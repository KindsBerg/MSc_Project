"""
train_model.py — leave-one-subject-out evaluation on the WESAD feature table.

    python train_model.py --diagnose                 # time-confound check
    python train_model.py --features clean           # full report, one config
    python train_model.py --task both --sweep        # the ablation table
    python train_model.py --ref-sweep                # baseline-buffer size sweep

--ref-sweep answers a different question from --sweep: not 'which features',
but 'how many windows does the per-subject scaling reference actually need'.
It holds the feature set at 'all' and varies N_REF_WINDOWS. The result is the
device's warm-up latency, which is N_REF_WINDOWS windows of wall-clock before
the first classification can be emitted.

Rationale for the validation scheme, the standardisation cascade and the
session-order control is in Project_ML_Notes.

'clean' under the 36-feature contract drops the posture means only; it is NOT
the same 'clean' as the 42-feature sweep, which also kept temp_slope and
temp_baseline_delta. Old sweep preserved in results/ablation_sweep_prior_contract.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import features as F

# Standardisation reference: first N ACCEPTED windows per subject, not a
# wall-clock cutoff. A time cutoff gave S6 zero reference windows, silently
# fell back to whole-recording stats for that subject alone, and collapsed
# the fold to 0.500.
N_REF_WINDOWS = 100
Z_CLIP = 20.0  # post-scaling clip; beyond this is a degenerate divisor

# Buffer sizes for --ref-sweep. Dense below 30 because that is where the
# scaling estimate is expected to still be moving; coarse above it because
# each extra window there costs 5 s of warm-up for no expected gain.
REF_GRID = [10, 15, 20, 25, 30, 40, 50, 75, 100]

# Windows before this per-subject index are excluded from scoring in the
# 'fixed' column. Must be >= max(REF_GRID) or the comparison is not like for
# like: a larger N would otherwise be scored on more of its own reference.
FIXED_EVAL_AFTER = max(REF_GRID)

CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")

STRESS_LABEL = 2
CEILINGS = {"binary": 0.88, "3class": 0.76}  # published WESAD all-wrist
TIME_COL = "t_start"

# Mean acceleration = gravity projection = posture. Drifts over a long
# recording, so correlates with session position.
POSTURE_FEATURES = ["acc_x_mean", "acc_y_mean", "acc_z_mean", "acc_mag_mean"]


def _drop(cols, remove) -> list:
    rm = set(remove)
    return [c for c in cols if c not in rm]


FEATURE_SETS = {
    "all": list(F.FEATURE_NAMES),
    "clean": _drop(F.FEATURE_NAMES, POSTURE_FEATURES),
    "eda_only": list(F.EDA_FEATURES),
    "eda_hr": list(F.EDA_FEATURES) + list(F.HR_FEATURES),
    "hr_only": list(F.HR_FEATURES),
    "time_only": [TIME_COL],
    "clean_plus_time": _drop(F.FEATURE_NAMES, POSTURE_FEATURES) + [TIME_COL],
}

# Retired at the 36-feature revision; both would now duplicate another row.
RETIRED_SETS = {
    "no_temp": "identical to 'all' — the TEMP block no longer exists",
    "no_posture": "identical to 'clean'",
    "eda_plus_time": "duplicate probe — use 'clean_plus_time'",
}

SWEEP_ORDER = ["all", "clean", "eda_only", "time_only", "clean_plus_time"]


def load_table(cache_dir: Path) -> pd.DataFrame:
    for name in ("wesad_features.parquet", "wesad_features.csv.gz"):
        p = cache_dir / name
        if p.is_file():
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            print(f"loaded {p}  ({len(df)} rows)")
            return df
    parts = sorted(cache_dir.glob("S*_features.parquet")) + sorted(
        cache_dir.glob("S*_features.csv.gz")
    )
    if not parts:
        raise FileNotFoundError(f"no feature cache in {cache_dir}")
    df = pd.concat(
        [pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p) for p in parts],
        ignore_index=True,
    )
    print(f"loaded {len(parts)} per-subject caches  ({len(df)} rows)")
    return df


def load_baseline_ref_s(cache_dir: Path) -> float | None:
    """Read the HR-baseline reference window baked into this feature cache.

    hr_baseline_delta is computed at build time (build_dataset.py
    --baseline-ref-s) against the first `baseline_ref_s` seconds of each
    recording and does NOT shrink with N_REF_WINDOWS. Any warm-up figure for
    a feature set touching the HR block is a lower bound unless this number
    is folded in — see _warmup_binding().

    Returns None for a cache built before this metadata existed, in which
    case the caller must treat the HR floor as unknown, not zero.
    """
    combined_meta = cache_dir / "wesad_features.meta.json"
    if combined_meta.is_file():
        return json.loads(combined_meta.read_text())["baseline_ref_s"]

    metas = sorted(cache_dir.glob("S*_features.meta.json"))
    if not metas:
        return None
    values = {json.loads(p.read_text())["baseline_ref_s"] for p in metas}
    if len(values) > 1:
        raise RuntimeError(
            f"cache built with inconsistent --baseline-ref-s across subjects: "
            f"{sorted(values)} -- rebuild with --force so every subject shares one value"
        )
    return values.pop()


def make_target(df: pd.DataFrame, task: str):
    if task == "binary":
        return (df["label"].to_numpy() == STRESS_LABEL).astype(int), [
            "non-stress",
            "stress",
        ]
    y = df["label"].to_numpy()
    lookup = {1: "baseline", 2: "stress", 3: "amusement"}
    return y, [lookup.get(int(c), str(c)) for c in sorted(np.unique(y))]


def diagnose_time_confound(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Per-subject correlation of each feature with elapsed recording time.

    Distinguishes 'the model reads a clock' from 'the model reads physiology'.
    |r| > 0.8 is a strong clock; 0.5-0.8 warrants comment.
    """
    cols = [c for c in F.FEATURE_NAMES if c in df.columns]
    per_subject = {}
    for sid, block in df.groupby("subject"):
        t = block[TIME_COL].to_numpy(dtype=np.float64)
        if t.size < 10 or np.std(t) == 0:
            continue
        rs = {}
        for c in cols:
            v = block[c].to_numpy(dtype=np.float64)
            ok = np.isfinite(v)
            rs[c] = (
                np.nan
                if ok.sum() < 10 or np.std(v[ok]) == 0
                else float(np.corrcoef(t[ok], v[ok])[0, 1])
            )
        per_subject[sid] = rs

    raw = pd.DataFrame(per_subject).T
    out = (
        pd.DataFrame(
            {
                "feature": raw.columns,
                "mean_abs_r": raw.abs().mean(axis=0).to_numpy(),
                "sd_abs_r": raw.abs().std(axis=0).to_numpy(),
                "mean_signed_r": raw.mean(axis=0).to_numpy(),
            }
        )
        .sort_values("mean_abs_r", ascending=False)
        .reset_index(drop=True)
    )

    print("\nTIME-CONFOUND DIAGNOSTIC  (|r| between each feature and elapsed time)")
    for _, r in out.head(top).iterrows():
        flag = (
            "  <-- strong"
            if r["mean_abs_r"] > 0.8
            else "  <-- moderate" if r["mean_abs_r"] > 0.5 else ""
        )
        print(
            f"  {r['feature']:<24} |r|={r['mean_abs_r']:.3f} "
            f"+/-{r['sd_abs_r']:.3f}  signed={r['mean_signed_r']:+.3f}{flag}"
        )

    strong = out[out["mean_abs_r"] > 0.8]["feature"].tolist()
    print(
        f"\n  {len(strong)} feature(s) with |r| > 0.8: {', '.join(strong)}"
        if strong
        else "\n  no feature exceeds |r| > 0.8"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "time_confound.csv", index=False)
    print(f"saved -> {RESULTS_DIR}/time_confound.csv")
    return out


def _standardise_core(
    df: pd.DataFrame, mode: str, n_ref: int = N_REF_WINDOWS
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Per-subject robust scaling. Never uses labels.

    Median and IQR, not mean and SD: a near-zero SD divisor on a short
    reference window produced z-scores in the hundreds. Divisor cascade is
    reference IQR -> subject IQR -> cohort IQR; a fixed floor was tried and
    pushed median max|z| from 34 to 2951.

    Too few reference windows is a hard error. A per-subject fallback is what
    silently invalidated S6's fold.

    t_start is left unscaled — it is a probe input.

    Returns (scaled_df, fallback_table). fallback_table is a subject x
    feature frame of which cascade level actually set the divisor
    ("reference"/"subject"/"cohort"/"unit"), or None for mode="none". A high
    subject/cohort rate confounds any comparison across n_ref, because those
    rows are partly borrowing scale the reference block didn't earn on its
    own — see fallback_report().
    """
    if mode == "none":
        return df, None

    out = df.copy()
    cols = [c for c in F.FEATURE_NAMES if c in df.columns]
    cohort_iqr = df[cols].quantile(0.75) - df[cols].quantile(0.25)
    fallback_rows = []

    for sid, idx in df.groupby("subject").groups.items():
        block = df.loc[idx, cols]
        if mode == "baseline":
            ref = df.loc[idx].sort_values(TIME_COL).head(n_ref)[cols]
            if len(ref) < 5:
                raise ValueError(f"{sid}: only {len(ref)} reference windows")
        elif mode == "subject":
            ref = block  # transductive; comparison only
        else:
            raise ValueError(f"unknown standardise mode: {mode}")

        centre = ref.median()
        iqr = ref.quantile(0.75) - ref.quantile(0.25)
        subj_iqr = block.quantile(0.75) - block.quantile(0.25)

        # Mirror the exact selection the scale computation below makes, so
        # the level recorded here is never out of step with the divisor
        # actually used (NaN compares False either direction, same as .where).
        used_ref = iqr > 0
        used_subj = (~used_ref) & (subj_iqr > 0)
        used_cohort = (~used_ref) & (~used_subj) & (cohort_iqr > 0)
        level = pd.Series("reference", index=cols, dtype=object)
        level[used_subj] = "subject"
        level[used_cohort] = "cohort"
        level[(~used_ref) & (~used_subj) & (~used_cohort)] = "unit"
        fallback_rows.append(pd.Series(level, name=sid))

        scale = iqr.where(iqr > 0, subj_iqr).where(lambda s: s > 0, cohort_iqr)
        scale = scale.where(scale > 0, 1.0)
        out.loc[idx, cols] = ((block - centre) / scale).to_numpy()

    out[cols] = out[cols].clip(-Z_CLIP, Z_CLIP).fillna(0.0)
    return out, pd.DataFrame(fallback_rows)


def standardise(df: pd.DataFrame, mode: str, n_ref: int = N_REF_WINDOWS) -> pd.DataFrame:
    return _standardise_core(df, mode, n_ref)[0]


def fallback_report(fallback_table: pd.DataFrame | None) -> None:
    """Divisor-cascade fallback rates: how often each feature borrowed scale
    from the subject or cohort rather than its own reference block.

    A handful of near-zero-variance features falling through on nearly every
    subject means the sweep's small-N rows are partly measuring cross-subject
    or whole-session scale rather than a stable reference estimate of their
    own — the confound has to be visible next to the accuracy it distorts,
    not left implicit.
    """
    if fallback_table is None or fallback_table.empty:
        return

    print("\nDIVISOR CASCADE")
    flat = pd.Series(fallback_table.to_numpy().ravel())
    overall = flat.value_counts(normalize=True)
    for level in ("reference", "subject", "cohort", "unit"):
        if level in overall.index:
            print(f"  {level:<10} {overall[level]:.1%}")

    per_subject = (fallback_table != "reference").mean(axis=1).sort_values(ascending=False)
    print("\n  fallback rate by subject (any non-reference level):")
    for sid, rate in per_subject.items():
        print(f"    {sid:<5} {rate:.1%}")

    per_feature = (fallback_table != "reference").mean(axis=0).sort_values(ascending=False)
    print("\n  top 10 features by fallback rate:")
    for feat, rate in per_feature.head(10).items():
        print(f"    {feat:<24} {rate:.1%}")


def scale_report(std_df: pd.DataFrame) -> None:
    """Post-standardisation magnitude check across subjects.

    Reports the 99th percentile, not the max: max saturates at Z_CLIP for
    every subject, so an outlier check on it can never fire. One subject
    reaching magnitudes the others never produce means that fold is lost to
    scaling rather than physiology — the S6 failure mode.
    """
    cols = [c for c in F.FEATURE_NAMES if c in std_df.columns]
    rows = []
    for sid, block in std_df.groupby("subject"):
        v = block[cols].to_numpy(dtype=np.float64)
        v = np.abs(v[np.isfinite(v)])
        rows.append(
            {
                "subject": sid,
                "p99": float(np.percentile(v, 99)) if v.size else np.nan,
                "clipped": float((v >= Z_CLIP).mean()) if v.size else np.nan,
            }
        )
    rep = pd.DataFrame(rows)
    med = rep["p99"].median()
    print(
        f"scale check: median p99|z| = {med:.2f}, "
        f"range {rep['p99'].min():.2f}..{rep['p99'].max():.2f}, "
        f"clipped {rep['clipped'].mean():.3%}"
    )
    for _, r in rep[(rep["p99"] > 3 * med) | (rep["p99"] < med / 3)].iterrows():
        print(f"  SCALE OUTLIER {r['subject']}: p99|z|={r['p99']:.2f}")


def make_model(kind: str, seed: int):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "hgb":
        # class_weight to match the RF branch above -- sklearn 1.9 supports it
        # on HGB. Without it, "--model both" compares a class-balanced RF
        # against an unbalanced HGB, which isn't a like-for-like comparison.
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, l2_regularization=1.0,
            class_weight="balanced", random_state=seed,
        )
    raise ValueError(f"unknown model: {kind}")


def resolve_features(name: str, df: pd.DataFrame) -> list:
    if name in RETIRED_SETS:
        raise ValueError(f"'{name}' was retired: {RETIRED_SETS[name]}")
    if name not in FEATURE_SETS:
        raise ValueError(f"unknown feature set: {name}")
    cols = [c for c in FEATURE_SETS[name] if c in df.columns]
    if not cols:
        raise ValueError(f"feature set '{name}' resolved to nothing")
    return cols


def window_rank(df: pd.DataFrame) -> np.ndarray:
    """Per-subject 0-based index of each window in recording order.

    The reference block is defined by position, not wall-clock, so any mask
    that excludes reference windows from scoring has to be defined the same
    way or the two disagree on dropped/rejected windows.
    """
    return (
        df.groupby("subject")[TIME_COL].rank(method="first").to_numpy(dtype=np.int64) - 1
    )


def _loso_folds(df, cols, task, model_kind, seed, do_permutation, verbose):
    """Yield one trained fold at a time.

    Split out so a caller can score the same predictions under more than one
    evaluation mask without retraining. Training is the expensive part and is
    identical regardless of which test windows are later counted.
    """
    X_all = df[cols].to_numpy(dtype=np.float64)
    y_all, class_names = make_target(df, task)
    subjects = df["subject"].to_numpy()

    for held in sorted(pd.unique(subjects)):
        te = subjects == held
        tr = ~te
        # Silent when it happens, so assert every fold.
        assert held not in set(subjects[tr]), "subject leaked across folds"

        y_tr, y_te = y_all[tr], y_all[te]
        if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
            continue

        imp = SimpleImputer(strategy="median").fit(X_all[tr])
        clf = make_model(model_kind, seed).fit(imp.transform(X_all[tr]), y_tr)
        X_te = imp.transform(X_all[te])
        y_hat = clf.predict(X_te)

        importance = None
        if do_permutation:
            importance = permutation_importance(
                clf, X_te, y_te, n_repeats=5, random_state=seed, n_jobs=-1,
            ).importances_mean
        elif hasattr(clf, "feature_importances_"):
            importance = clf.feature_importances_

        yield {
            "subject": held,
            "test_mask": te,
            "y_true": y_te,
            "y_pred": y_hat,
            "importance": importance,
            "class_names": class_names,
        }


def _fold_metrics(subject, y_true, y_pred) -> dict:
    return {
        "subject": subject,
        "n_test": int(len(y_true)),
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_loso(
    df: pd.DataFrame,
    task: str,
    model_kind: str,
    feature_set: str = "all",
    seed: int = 0,
    do_permutation: bool = False,
    verbose: bool = True,
):
    """One leave-one-subject-out sweep. Fold order and seed are fixed, so
    feature sets are compared over identical folds rather than fold noise."""
    cols = resolve_features(feature_set, df)

    fold_rows, y_true_all, y_pred_all, importances = [], [], [], []
    class_names = None

    for fold in _loso_folds(df, cols, task, model_kind, seed, do_permutation, verbose):
        y_te, y_hat = fold["y_true"], fold["y_pred"]
        class_names = fold["class_names"]

        y_true_all.append(y_te)
        y_pred_all.append(y_hat)
        fold_rows.append(_fold_metrics(fold["subject"], y_te, y_hat))
        if verbose:
            r = fold_rows[-1]
            print(
                f"  {fold['subject']}: acc={r['accuracy']:.3f} bal={r['balanced_acc']:.3f} "
                f"macroF1={r['macro_f1']:.3f}  (n={r['n_test']})"
            )
        if fold["importance"] is not None:
            importances.append(fold["importance"])

    y_true, y_pred = np.concatenate(y_true_all), np.concatenate(y_pred_all)
    folds = pd.DataFrame(fold_rows)

    summary = {
        "task": task,
        "model": model_kind,
        "feature_set": feature_set,
        "n_features": len(cols),
        "contract_n_features": F.N_FEATURES,
        "n_folds": len(folds),
        "accuracy_mean": float(folds["accuracy"].mean()),
        "accuracy_sd": float(folds["accuracy"].std()),
        "balanced_acc_mean": float(folds["balanced_acc"].mean()),
        "balanced_acc_sd": float(folds["balanced_acc"].std()),
        "macro_f1_mean": float(folds["macro_f1"].mean()),
        "macro_f1_sd": float(folds["macro_f1"].std()),
        "pooled_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "worst_fold_macro_f1": float(folds["macro_f1"].min()),
        "best_fold_macro_f1": float(folds["macro_f1"].max()),
    }

    imp_df = None
    if importances:
        arr = np.vstack(importances)
        imp_df = (
            pd.DataFrame(
                {
                    "feature": cols,
                    "importance_mean": arr.mean(axis=0),
                    "importance_sd": arr.std(axis=0),
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )

    return folds, summary, (y_true, y_pred, class_names), imp_df


def report(summary, evald, folds, imp_df):
    y_true, y_pred, class_names = evald
    print("\n" + "=" * 68)
    print(
        f"{summary['task'].upper()} / {summary['model'].upper()} / "
        f"{summary['feature_set']} ({summary['n_features']}) / "
        f"{summary['n_folds']} folds LOSO"
    )
    print("=" * 68)
    for k, lab in (("accuracy", "accuracy    "), ("balanced_acc", "balanced acc"),
                   ("macro_f1", "macro F1    ")):
        print(f"{lab}  {summary[k + '_mean']:.3f} +/- {summary[k + '_sd']:.3f}")
    print(
        f"fold range    {summary['worst_fold_macro_f1']:.3f} .. "
        f"{summary['best_fold_macro_f1']:.3f} macro F1"
    )

    # Near-perfect on some subjects and at chance on others is the signature
    # of a deterministic separator rather than graded physiology.
    near_perfect = int((folds["macro_f1"] > 0.97).sum())
    at_chance = int((folds["balanced_acc"] < 0.55).sum())
    if near_perfect and at_chance:
        print(
            f"\n  NOTE: {near_perfect} fold(s) near-perfect, {at_chance} at chance — "
            "bimodal. Check --diagnose."
        )

    print("\nper-class (pooled)")
    print(classification_report(y_true, y_pred, target_names=class_names,
                                digits=3, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print("confusion matrix (rows = true)")
    print(f"{'':>13}" + "".join(f"{c[:9]:>11}" for c in class_names))
    for i, c in enumerate(class_names):
        print(f"{c[:12]:>13}" + "".join(f"{v:>11}" for v in cm[i]))

    print("\nweakest subjects (macro F1)")
    for _, r in folds.nsmallest(3, "macro_f1").iterrows():
        print(f"  {r['subject']}: {r['macro_f1']:.3f}")

    if imp_df is not None and len(imp_df) > 1:
        print("\ntop 15 features")
        for _, r in imp_df.head(15).iterrows():
            print(f"  {r['feature']:<24} {r['importance_mean']:.4f} "
                  f"+/- {r['importance_sd']:.4f}")
        total = imp_df["importance_mean"].sum() or 1.0
        shares = {
            n: imp_df[imp_df["feature"].isin(b)]["importance_mean"].sum() / total
            for n, b in (("EDA", F.EDA_FEATURES), ("HR", F.HR_FEATURES),
                         ("IMU", F.IMU_FEATURES), ("CROSS", F.CROSS_FEATURES))
        }
        print("\nblock share:  " + "   ".join(f"{k} {v:.1%}" for k, v in shares.items()))
        if shares["HR"] > shares["EDA"]:
            print("  NOTE: HR outweighs EDA. HR transfers worst to the hardware —\n"
                  "  NeuroKit2 beat detection vs the SEN0344's on-board estimator.")

    ceil = CEILINGS.get(summary["task"])
    if ceil:
        print(
            f"\nWESAD published all-wrist ceiling ~{ceil:.2f}, you "
            f"{summary['accuracy_mean']:.3f} ({summary['accuracy_mean'] - ceil:+.3f})."
            "\n  Context only: elapsed time alone scores above both published"
            "\n  ceilings here, and those rows include a temperature channel"
            "\n  this build does not use."
        )


def save(tag: str, folds, summary, imp_df, evald):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(RESULTS_DIR / f"{tag}_folds.csv", index=False)
    (RESULTS_DIR / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    if imp_df is not None:
        imp_df.to_csv(RESULTS_DIR / f"{tag}_importance.csv", index=False)
    y_true, y_pred, names = evald
    pd.DataFrame(confusion_matrix(y_true, y_pred), index=names, columns=names).to_csv(
        RESULTS_DIR / f"{tag}_confusion.csv"
    )


def sweep_table(rows: list) -> None:
    print("\n" + "=" * 72)
    print(f"ABLATION SWEEP  (identical folds and seed, {F.N_FEATURES}-feature contract)")
    print("=" * 72)
    print(f"{'task':<8}{'model':<6}{'features':<18}{'n':>4}"
          f"{'acc':>9}{'bal':>9}{'macroF1':>10}{'+/-':>8}")
    for r in rows:
        print(f"{r['task']:<8}{r['model']:<6}{r['feature_set']:<18}{r['n_features']:>4}"
              f"{r['accuracy_mean']:>9.3f}{r['balanced_acc_mean']:>9.3f}"
              f"{r['macro_f1_mean']:>10.3f}{r['macro_f1_sd']:>8.3f}")

    print("\ndelta vs 'all' (balanced accuracy)")
    for (task, model), grp in pd.DataFrame(rows).groupby(["task", "model"]):
        base = grp[grp["feature_set"] == "all"]
        if base.empty:
            continue
        b = float(base["balanced_acc_mean"].iloc[0])
        for _, r in grp[grp["feature_set"] != "all"].iterrows():
            print(f"  {task}/{model}  {r['feature_set']:<18} "
                  f"{r['balanced_acc_mean'] - b:+.3f}")

    print(
        "\n  time_only        control, not a result. Whatever it scores is"
        "\n                   recoverable from session position alone, and every"
        "\n                   other row inherits that as a caveat."
        "\n  clean            headline configuration: posture means removed."
        "\n  eda_only         deployment-realistic figure — the custom EDA"
        "\n                   front-end alone, and the only row whose inputs all"
        "\n                   exist outside a fixed-order lab protocol."
        "\n  clean_plus_time  redundancy probe. If it matches time_only, the"
        "\n                   physiology adds nothing the clock does not carry."
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "ablation_sweep.csv", index=False)
    print(f"\nsaved -> {RESULTS_DIR}/ablation_sweep.csv")


def reference_convergence(df: pd.DataFrame, grid=REF_GRID) -> pd.DataFrame:
    """How far the N-window median/IQR sits from the whole-session estimate.

    Label-free and classifier-free, so it answers 'when do the scaling
    statistics stop moving' without inheriting anything from the model. Run
    on RAW features — this describes the reference block itself, and
    standardising first would be circular.

    Centre error is expressed in units of the subject's full-session IQR so
    it is comparable across features on different scales. Scale error is a
    plain relative error on the IQR.
    """
    cols = [c for c in F.FEATURE_NAMES if c in df.columns]
    rank = window_rank(df)
    rows = []

    for sid, idx in df.groupby("subject").groups.items():
        block = df.loc[idx, cols]
        r = rank[df.index.get_indexer(idx)]
        order = np.argsort(r, kind="stable")
        block = block.iloc[order]

        full_med = block.median()
        full_iqr = block.quantile(0.75) - block.quantile(0.25)
        denom = full_iqr.where(full_iqr > 0, np.nan)

        for n in grid:
            if len(block) < n:
                continue
            ref = block.head(n)
            ref_med = ref.median()
            ref_iqr = ref.quantile(0.75) - ref.quantile(0.25)

            centre_err = ((ref_med - full_med).abs() / denom).replace(
                [np.inf, -np.inf], np.nan
            )
            scale_err = ((ref_iqr - full_iqr).abs() / denom).replace(
                [np.inf, -np.inf], np.nan
            )

            row = {
                "subject": sid,
                "n_ref": n,
                "centre_err_median": float(centre_err.median(skipna=True)),
                "centre_err_p90": float(centre_err.quantile(0.90)),
                "scale_err_median": float(scale_err.median(skipna=True)),
                "scale_err_p90": float(scale_err.quantile(0.90)),
            }
            for name, blk in (("EDA", F.EDA_FEATURES), ("HR", F.HR_FEATURES),
                              ("IMU", F.IMU_FEATURES), ("CROSS", F.CROSS_FEATURES)):
                sub = [c for c in blk if c in centre_err.index]
                row[f"centre_err_{name}"] = (
                    float(centre_err[sub].median(skipna=True)) if sub else np.nan
                )
            rows.append(row)

    per_subject = pd.DataFrame(rows)
    metric_cols = [c for c in per_subject.columns if c not in ("subject", "n_ref")]
    agg = per_subject.groupby("n_ref")[metric_cols].agg(["mean", "max"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_subject.to_csv(RESULTS_DIR / "ref_convergence_per_subject.csv", index=False)
    agg.to_csv(RESULTS_DIR / "ref_convergence.csv", index=False)

    print("\n" + "=" * 72)
    print("REFERENCE STATISTIC CONVERGENCE  (raw features, no labels, no model)")
    print("=" * 72)
    print("  centre err = |median_N - median_full| / IQR_full")
    print("  scale err  = |IQR_N - IQR_full| / IQR_full")
    print(f"\n{'N':>5}{'warmup':>10}{'centre':>9}{'ctr p90':>9}"
          f"{'scale':>9}{'scl p90':>9}{'worst subj':>12}")
    for _, r in agg.iterrows():
        n = int(r["n_ref"])
        print(f"{n:>5}{_warmup_str(n):>10}"
              f"{r['centre_err_median_mean']:>9.3f}{r['centre_err_p90_mean']:>9.3f}"
              f"{r['scale_err_median_mean']:>9.3f}{r['scale_err_p90_mean']:>9.3f}"
              f"{r['centre_err_median_max']:>12.3f}")

    print("\n  Read the knee, not the minimum: the estimate is monotonically")
    print("  better with N by construction, since it converges on the full-session")
    print("  value it is measured against. The question is where the curve flattens.")
    print(f"\nsaved -> {RESULTS_DIR}/ref_convergence.csv")
    return agg


def _fmt_mmss(seconds: float) -> str:
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def _warmup_str(n: int, win_s: float = F.WIN_EDA_S, step_s: float = F.WIN_STEP_S) -> str:
    """Wall-clock cost of an N-window buffer, excluding electrode settling."""
    return _fmt_mmss(win_s + (n - 1) * step_s)


# Only these two are actually referenced against BASELINE_REF_S at build time
# (build_dataset.py); the rest of HR_FEATURES is computed purely from the
# short window itself and shares no dependency on the resting prior. A
# feature set containing either has a warm-up floor independent of N: the
# scaling buffer can be made arbitrarily small, the HR prior still takes
# baseline_ref_s to fill.
HR_DEPENDENT_FEATURES = {"hr_baseline_delta", "hr_delta_x_still"}


def _warmup_binding(n_window_s: float, hr_floor_s: float) -> tuple:
    """Wall-clock warm-up for one N, and which of two constraints binds.

    Two independent clocks can gate the first classification: the N-window
    scaling buffer (n_window_s) and, for a feature set touching the HR
    baseline, the fixed HR-reference prior (hr_floor_s). Whichever is larger
    is what the device actually waits on.
    """
    if hr_floor_s > n_window_s:
        return hr_floor_s, "hr_baseline"
    return n_window_s, "n_window"


def ref_window_sweep(
    raw_df: pd.DataFrame,
    task: str,
    model_kind: str,
    feature_set: str = "all",
    seed: int = 0,
    grid=REF_GRID,
    fixed_after: int = FIXED_EVAL_AFTER,
    baseline_ref_s: float | None = None,
) -> pd.DataFrame:
    """Balanced accuracy as a function of the scaling reference size.

    Scored two ways per N, from one training pass each:

      fixed   windows at per-subject index >= fixed_after only. The honest
              comparison: every N is scored on exactly the same test windows,
              none of which were in any reference block.
      per_n   windows at index >= N. Uses more of each recording, but the
              test set grows as N shrinks, so the columns are not directly
              comparable to each other. Present for completeness.

    Neither scores a window that fed its own subject's scaling. Doing so
    inflates larger N specifically, because a larger reference block is a
    larger share of the windows it is then evaluated on.

    `baseline_ref_s`, if given, is folded into the warm-up figure whenever
    `feature_set` touches HR_DEPENDENT_FEATURES (see _warmup_binding) — a
    feature set with no HR dependency (eda_only, time_only) is unaffected and
    its warm-up is exactly the N-window buffer time.
    """
    cols_check = resolve_features(feature_set, raw_df)
    rank = window_rank(raw_df)
    rows, fold_frames = [], []

    has_hr = any(c in HR_DEPENDENT_FEATURES for c in cols_check)
    if has_hr and baseline_ref_s is None:
        print("  WARNING: feature set touches the HR baseline but no "
              "baseline_ref_s metadata was found on this cache -- warm-up "
              "figures below ignore the HR prior entirely")
    hr_floor_s = baseline_ref_s if (has_hr and baseline_ref_s is not None) else 0.0

    print("\n" + "=" * 72)
    print(f"REFERENCE WINDOW SWEEP  ({task} / {model_kind} / {feature_set}, "
          f"{len(cols_check)} features, hr_floor={hr_floor_s:.0f}s)")
    print("=" * 72)

    for n in grid:
        df_n, fallback_table = _standardise_core(raw_df, "baseline", n_ref=n)
        fallback_rate = (
            float((fallback_table != "reference").to_numpy().mean())
            if fallback_table is not None else float("nan")
        )
        eligible = {"fixed": rank >= fixed_after, "per_n": rank >= n}

        acc = {k: {"true": [], "pred": [], "folds": []} for k in eligible}

        for fold in _loso_folds(
            df_n, cols_check, task, model_kind, seed,
            do_permutation=False, verbose=False,
        ):
            te = fold["test_mask"]
            for key, keep in eligible.items():
                sel = keep[te]
                if sel.sum() == 0:
                    continue
                y_t, y_p = fold["y_true"][sel], fold["y_pred"][sel]
                if len(np.unique(y_t)) < 2:
                    continue
                acc[key]["true"].append(y_t)
                acc[key]["pred"].append(y_p)
                acc[key]["folds"].append(_fold_metrics(fold["subject"], y_t, y_p))

        n_window_s = F.WIN_EDA_S + (n - 1) * F.WIN_STEP_S
        warmup_s, warmup_binding = _warmup_binding(n_window_s, hr_floor_s)
        row = {
            "n_ref": n,
            "n_window_s": n_window_s,
            "hr_floor_s": hr_floor_s,
            "warmup_s": warmup_s,
            "warmup_binding": warmup_binding,
            "fallback_rate": fallback_rate,
            "task": task,
            "model": model_kind,
            "feature_set": feature_set,
            "n_features": len(cols_check),
        }
        for key in eligible:
            f_df = pd.DataFrame(acc[key]["folds"])
            if f_df.empty:
                continue
            row[f"{key}_bal_mean"] = float(f_df["balanced_acc"].mean())
            row[f"{key}_bal_sd"] = float(f_df["balanced_acc"].std())
            row[f"{key}_bal_min"] = float(f_df["balanced_acc"].min())
            row[f"{key}_bal_p10"] = float(f_df["balanced_acc"].quantile(0.10))
            row[f"{key}_macro_f1_mean"] = float(f_df["macro_f1"].mean())
            row[f"{key}_n_test_total"] = int(f_df["n_test"].sum())
            row[f"{key}_n_folds"] = int(len(f_df))
            f_df = f_df.assign(n_ref=n, eval=key)
            fold_frames.append(f_df)

        rows.append(row)
        print(f"  N={n:>3} ({_fmt_mmss(warmup_s):>6}, bind={warmup_binding:<11} fb={fallback_rate:.1%})  "
              f"fixed bal={row.get('fixed_bal_mean', float('nan')):.3f} "
              f"(min {row.get('fixed_bal_min', float('nan')):.3f})   "
              f"per_n bal={row.get('per_n_bal_mean', float('nan')):.3f} "
              f"(min {row.get('per_n_bal_min', float('nan')):.3f})")

    out = pd.DataFrame(rows)
    # A grid whose fixed_after exceeds what any subject's recording supports
    # leaves the 'fixed' eval empty at EVERY N (0 test windows with both
    # classes present past that index, for every subject) -- the aggregate
    # columns then never appear in any row dict at all, not just as NaN.
    # That's a real result (this fixed_after is unusable), not a crash.
    for key in ("fixed", "per_n"):
        for suffix in ("_bal_mean", "_bal_sd", "_bal_min", "_bal_p10", "_macro_f1_mean"):
            col = f"{key}{suffix}"
            if col not in out.columns:
                out[col] = np.nan
        for suffix in ("_n_test_total", "_n_folds"):
            col = f"{key}{suffix}"
            if col not in out.columns:
                out[col] = 0

    print(f"\n{'N':>5}{'warmup':>9}{'bind':>12}{'fb%':>7}{'fixed':>9}{'sd':>7}{'min':>8}{'p10':>8}"
          f"{'per_n':>9}{'min':>8}{'ntest':>8}")
    for _, r in out.iterrows():
        ntest = int(r["fixed_n_test_total"]) if np.isfinite(r["fixed_n_test_total"]) else 0
        print(f"{int(r['n_ref']):>5}{_fmt_mmss(r['warmup_s']):>9}{r['warmup_binding']:>12}"
              f"{r['fallback_rate']:>7.1%}"
              f"{r['fixed_bal_mean']:>9.3f}{r['fixed_bal_sd']:>7.3f}"
              f"{r['fixed_bal_min']:>8.3f}{r['fixed_bal_p10']:>8.3f}"
              f"{r['per_n_bal_mean']:>9.3f}{r['per_n_bal_min']:>8.3f}"
              f"{ntest:>8}")

    if out["fixed_bal_mean"].notna().any():
        best = out.loc[out["fixed_bal_mean"].idxmax()]
        ref100 = out[out["n_ref"] == 100]
        if not ref100.empty and np.isfinite(ref100["fixed_bal_mean"].iloc[0]):
            b100 = float(ref100["fixed_bal_mean"].iloc[0])
            within = out[out["fixed_bal_mean"] >= b100 - 0.01]
            if not within.empty:
                k = int(within["n_ref"].min())
                k_row = out[out["n_ref"] == k].iloc[0]
                print(f"\n  smallest N within 0.01 of N=100: {k} "
                      f"({_fmt_mmss(k_row['warmup_s'])} vs {_fmt_mmss(ref100['warmup_s'].iloc[0])} warm-up)")
        print(f"  best mean at N={int(best['n_ref'])} ({best['fixed_bal_mean']:.3f})")
    else:
        print(
            f"\n  NOTE: 'fixed' is empty at every N in this grid — fixed_after="
            f"{fixed_after} leaves no subject with test windows of both classes "
            "past that index. Read 'per_n' only below, with its usual confound "
            "(the test set is not held constant across N)."
        )

    print(
        "\n  Compare the 'fixed' column across rows; it is the only one with a"
        "\n  constant test set. 'per_n' is reported because it is what the"
        "\n  deployed device would actually score on, but its test set grows as"
        "\n  N falls, so a difference between two per_n rows confounds buffer"
        "\n  size with how much of the recording was scored."
        "\n"
        "\n  The min and p10 columns carry the argument, not the mean. A small"
        "\n  buffer fails by landing on an unrepresentative opening stretch for"
        "\n  one subject, which moves the worst fold long before it moves the"
        "\n  average."
        "\n"
        "\n  WESAD opens with a long resting baseline, so every reference block"
        "\n  here is drawn from rest. That is an assumption the device inherits:"
        "\n  it must be donned and settled at rest. Not a property of the method."
        "\n"
        "\n  'bind' names which clock actually gates warm-up: n_window means the"
        "\n  scaling buffer is still the bottleneck; hr_baseline means the fixed"
        "\n  HR-reference prior already exceeds it, so shrinking N further buys"
        "\n  nothing on its own. 'fb%' is the divisor-cascade fallback rate at"
        "\n  this N (see fallback_report) -- if small N borrows more cross-"
        "\n  subject/session scale than large N, that confounds the comparison"
        "\n  in large N's favour."
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "ref_window_sweep.csv", index=False)
    if fold_frames:
        pd.concat(fold_frames, ignore_index=True).to_csv(
            RESULTS_DIR / "ref_window_sweep_folds.csv", index=False
        )
    print(f"\nsaved -> {RESULTS_DIR}/ref_window_sweep.csv")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--task", choices=["binary", "3class", "both"], default="binary")
    ap.add_argument("--model", choices=["rf", "hgb", "both"], default="rf")
    ap.add_argument("--features", default="all",
                    help=f"one of: {', '.join(sorted(FEATURE_SETS))}")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--ref-sweep", action="store_true",
                    help="sweep the scaling reference size over REF_GRID")
    ap.add_argument("--ref-grid", default=None,
                    help="comma-separated override for the sweep grid")
    ap.add_argument("--no-convergence", action="store_true",
                    help="skip the label-free statistic convergence table")
    ap.add_argument("--baseline-n", type=int, default=N_REF_WINDOWS,
                    help="scaling reference size for a single run")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--standardise", choices=["none", "baseline", "subject"],
                    default="baseline")
    ap.add_argument("--permutation", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.features in RETIRED_SETS:
        print(f"'{args.features}' was retired: {RETIRED_SETS[args.features]}")
        return 1
    if args.features not in FEATURE_SETS:
        print(f"unknown feature set '{args.features}'. "
              f"Available: {', '.join(sorted(FEATURE_SETS))}")
        return 1

    df = load_table(Path(args.cache))

    # A 42-column table would train silently on the 36-column subset.
    try:
        F.assert_no_retired(df.columns)
    except RuntimeError as e:
        print(f"\nSTALE FEATURE TABLE — {e}")
        return 1
    missing = [c for c in F.FEATURE_NAMES if c not in df.columns]
    if missing or TIME_COL not in df.columns:
        print(f"stale cache — missing {missing or [TIME_COL]}; rebuild with --force")
        return 1

    print(f"subjects={df['subject'].nunique()}  features={F.N_FEATURES}")
    print(df["label_name"].value_counts(normalize=True).round(3).to_string())

    baseline_ref_s = load_baseline_ref_s(Path(args.cache))
    if baseline_ref_s is not None:
        print(f"HR baseline reference: {baseline_ref_s:.0f}s (baked into hr_baseline_delta at cache-build time)")
    else:
        print("HR baseline reference: UNKNOWN — cache predates baseline metadata; rebuild with --force")

    # Diagnostic runs on RAW features — standardising rescales but does not
    # decorrelate, and raw is what the report should quote.
    if args.diagnose:
        diagnose_time_confound(df)
        return 0

    if args.ref_sweep:
        grid = (
            [int(x) for x in args.ref_grid.split(",")] if args.ref_grid else REF_GRID
        )
        if not args.no_convergence:
            reference_convergence(df, grid=grid)
        fixed_after = max(max(grid), FIXED_EVAL_AFTER)
        ref_window_sweep(
            df, args.task if args.task != "both" else "binary",
            args.model if args.model != "both" else "rf",
            feature_set=args.features, seed=args.seed,
            grid=grid, fixed_after=fixed_after,
            baseline_ref_s=baseline_ref_s,
        )
        return 0

    df, fallback_table = _standardise_core(df, args.standardise, n_ref=args.baseline_n)
    print(f"standardisation: {args.standardise} (n_ref={args.baseline_n})")
    if args.standardise != "none":
        scale_report(df)
        fallback_report(fallback_table)

    tasks = ["binary", "3class"] if args.task == "both" else [args.task]
    models = ["rf", "hgb"] if args.model == "both" else [args.model]
    sets = SWEEP_ORDER if args.sweep else [args.features]

    sweep_rows = []
    for task in tasks:
        for kind in models:
            for fs in sets:
                folds, summary, evald, imp_df = run_loso(
                    df, task, kind, feature_set=fs, seed=args.seed,
                    do_permutation=args.permutation,
                    verbose=not (args.quiet or args.sweep),
                )
                summary["standardise"] = args.standardise
                summary["n_ref_windows"] = args.baseline_n
                if args.sweep:
                    print(f"  {task:<7} {fs:<18} bal={summary['balanced_acc_mean']:.3f}")
                else:
                    report(summary, evald, folds, imp_df)
                # Suffix only on a non-default N, so existing result filenames
                # from the 100-window runs are not silently orphaned.
                tag = f"{task}_{kind}_{fs}_{args.standardise}"
                if args.baseline_n != N_REF_WINDOWS:
                    tag += f"_n{args.baseline_n}"
                save(tag, folds, summary, imp_df, evald)
                sweep_rows.append(summary)

    if args.sweep:
        sweep_table(sweep_rows)
    else:
        print(f"\nsaved -> {RESULTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())