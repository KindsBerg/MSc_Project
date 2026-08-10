"""
train_model.py — leave-one-subject-out evaluation on the WESAD feature table.

    python train_model.py --diagnose                    # time-confound check
    python train_model.py --features clean --permutation
    python train_model.py --task both --sweep           # the ablation table

Standardisation reference: the first N_REF_WINDOWS ACCEPTED windows per
subject, not a wall-clock cutoff. A time cutoff gave S6 zero reference
windows, silently fell back to whole-recording stats for that subject alone,
and collapsed the fold to 0.500.

N_REF_WINDOWS = 40 was chosen from a sweep over 10-400 windows. Balanced
accuracy showed no significant dependence on it anywhere in that range
(best p = 0.074), so the value was selected on worst-case fold performance
and warm-up latency instead: 40 is the smallest buffer at which no fold
collapses to a single-class predictor, and it costs 4m15s of warm-up.
That sweep is finished; its outputs are in results/ref_window_sweep*.csv.

'clean' under the 36-feature contract drops the posture means only; it is NOT
the same 'clean' as the 42-feature sweep. The old sweep is preserved in
results/ablation_sweep_prior_contract.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import features as F

N_REF_WINDOWS = 40
Z_CLIP = 20.0  # post-scaling clip; beyond this is a degenerate divisor

CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")

STRESS_LABEL = 2
CEILINGS = {"binary": 0.88, "3class": 0.76}  # published WESAD all-wrist
TIME_COL = "t_start"

# Mean acceleration = gravity projection = posture. Drifts over a long
# recording, so it correlates with session position.
POSTURE_FEATURES = ["acc_x_mean", "acc_y_mean", "acc_z_mean", "acc_mag_mean"]


def _drop(cols, remove) -> list:
    rm = set(remove)
    return [c for c in cols if c not in rm]


# IMU contribution to `device`: motion magnitude and the motion flag only —
# not the per-axis or sd/absint/peakfreq columns, which the SEN0344/EDA/IMU
# stack doesn't need to reproduce for a stress read.
DEVICE_IMU_FEATURES = ["acc_mag_mean", "motion_flag"]

# Everything the deployable hardware can actually produce: the custom EDA
# front end, the four-feature HR block (see features.py's HR_FEATURES), and
# minimal IMU motion context. Excluded, and why:
#   HRV features     — SEN0344 firmware blocks raw BVP FIFO access, so no
#                       beat-to-beat intervals exist to compute HRV from.
#   temperature      — the LM75BD on this PCB reads board self-heating, not
#                       skin temperature; a different instrument entirely.
#   absolute posture — wrist orientation from WESAD's Empatica placement
#                       doesn't transfer to this device's wrist mount.
DEVICE_FEATURES = list(F.EDA_FEATURES) + list(F.HR_FEATURES) + DEVICE_IMU_FEATURES

# Six sets, each with a distinct job:
#   all                every feature in the contract; defined for --features
#                      but not part of the reported sweep
#   clean              headline configuration, posture means removed
#   eda_only           deployment-realistic: the custom EDA front end alone;
#                      defined for --features but not part of the reported sweep
#   device             the deployable hardware configuration — see DEVICE_FEATURES
#   device_nomotionflag  device minus motion_flag — see its definition below
#   time_only          control, not a result — see sweep_table()
FEATURE_SETS = {
    "all": list(F.FEATURE_NAMES),
    "clean": _drop(F.FEATURE_NAMES, POSTURE_FEATURES),
    "eda_only": list(F.EDA_FEATURES),
    "device": DEVICE_FEATURES,
    # motion_flag's reference-window IQR is zero for 13/15 subjects, so it
    # falls through the divisor cascade to the unit path and enters the
    # matrix untransformed while every other feature is standardised.
    "device_nomotionflag": _drop(DEVICE_FEATURES, ["motion_flag"]),
    "time_only": [TIME_COL],
}

SWEEP_ORDER = ["clean", "device", "device_nomotionflag", "time_only"]


def load_table(cache_dir: Path) -> pd.DataFrame:
    """Load the feature cache, pinned to features.py's current pipeline version.

    Filenames are derived from FEATURE_PIPELINE_VERSION rather than hardcoded
    here, so build_dataset.py (which writes them) and this loader can't drift
    apart. Deliberately not permissive: a cache built under an older version
    won't match and won't be silently substituted — see FEATURE_PIPELINE_VERSION
    in features.py for why that matters.
    """
    version = F.FEATURE_PIPELINE_VERSION
    for name in (f"wesad_features_v{version}.parquet", f"wesad_features_v{version}.csv.gz"):
        p = cache_dir / name
        if p.is_file():
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            print(f"loaded {p}  ({len(df)} rows)")
            return df
    parts = sorted(cache_dir.glob(f"S*_features_v{version}.parquet")) + sorted(
        cache_dir.glob(f"S*_features_v{version}.csv.gz")
    )
    if not parts:
        raise FileNotFoundError(
            f"no feature_pipeline_version={version} cache found in {cache_dir} "
            f"(looked for wesad_features_v{version}.parquet/.csv.gz and "
            f"S*_features_v{version}.parquet/.csv.gz)"
        )
    df = pd.concat(
        [pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p) for p in parts],
        ignore_index=True,
    )
    print(f"loaded {len(parts)} per-subject caches  ({len(df)} rows)")
    return df


def load_baseline_ref_s(cache_dir: Path) -> float | None:
    """Read the HR-baseline reference window baked into this feature cache.

    hr_baseline_delta is computed at build time and does NOT shrink with
    N_REF_WINDOWS, so it sets its own floor on device warm-up for any feature
    set touching the HR block. Returns None for a cache built before this
    metadata existed — treat that as unknown, not zero.
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
            f"{sorted(values)} — rebuild with --force"
        )
    return values.pop()


def make_target(df: pd.DataFrame, task: str):
    if task == "binary":
        return (df["label"].to_numpy() == STRESS_LABEL).astype(int), ["non-stress", "stress"]
    y = df["label"].to_numpy()
    lookup = {1: "baseline", 2: "stress", 3: "amusement"}
    return y, [lookup.get(int(c), str(c)) for c in sorted(np.unique(y))]


def diagnose_time_confound(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Per-subject correlation of each feature with elapsed recording time.

    Distinguishes 'the model reads a clock' from 'the model reads physiology'.
    |r| > 0.8 is a strong clock; 0.5-0.8 warrants comment. WESAD runs its
    condition blocks in a fixed order, so this is the confound behind
    time_only outscoring every physiological configuration.
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
            "  <-- strong" if r["mean_abs_r"] > 0.8
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


# --------------------------------------------------------------------------
# Standardisation
# --------------------------------------------------------------------------


def _standardise_core(
    df: pd.DataFrame, mode: str, n_ref: int = N_REF_WINDOWS
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Per-subject robust scaling. Never uses labels.

    Median and IQR, not mean and SD: a near-zero SD divisor on a short
    reference window produced z-scores in the hundreds. The divisor cascade is
    reference IQR -> subject IQR -> cohort IQR; a fixed floor was tried and
    pushed median max|z| from 34 to 2951.

    Too few reference windows is a hard error — a silent per-subject fallback
    is what invalidated S6's fold.

    t_start is left unscaled; it is a probe input.

    Returns (scaled_df, fallback_table). fallback_table is a subject x feature
    frame of which cascade level set the divisor, or None for mode="none".
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

        # Mirror the exact selection the scale computation below makes, so the
        # level recorded is never out of step with the divisor actually used
        # (NaN compares False either direction, same as .where).
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
    """How often each feature borrowed scale from the subject or cohort rather
    than its own reference block.

    Binary and near-zero-variance features fall through on most subjects,
    which means those columns are not really being standardised per subject at
    all. Worth seeing next to the accuracy it affects.
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
    every subject, so an outlier check on it could never fire. One subject
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


# --------------------------------------------------------------------------
# Model and evaluation
# --------------------------------------------------------------------------


def make_model(kind: str, seed: int):
    if kind != "rf":
        raise ValueError(f"unknown model: {kind}")
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def resolve_features(name: str, df: pd.DataFrame) -> list:
    if name not in FEATURE_SETS:
        raise ValueError(f"unknown feature set: {name}")
    cols = [c for c in FEATURE_SETS[name] if c in df.columns]
    if not cols:
        raise ValueError(f"feature set '{name}' resolved to nothing")
    return cols


def _loso_folds(df, cols, task, model_kind, seed, do_permutation, verbose):
    """Yield one trained fold at a time.

    Split out so a caller can score the same predictions under more than one
    evaluation mask without retraining. Used by verify_subjects.py.
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
                clf, X_te, y_te, n_repeats=5, random_state=seed, n_jobs=-1
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


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


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


def result_tag(task: str, model: str, feature_set: str, mode: str, n_ref: int) -> str:
    """Result filename stem. n_ref is always included so runs at different
    buffer sizes never overwrite each other — including the ones already in
    results/ from earlier sweeps."""
    return f"{task}_{model}_{feature_set}_{mode}_n{n_ref}"


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

    delta_base = "clean"
    print(f"\ndelta vs '{delta_base}' (balanced accuracy)")
    for (task, model), grp in pd.DataFrame(rows).groupby(["task", "model"]):
        base = grp[grp["feature_set"] == delta_base]
        if base.empty:
            continue
        b = float(base["balanced_acc_mean"].iloc[0])
        for _, r in grp[grp["feature_set"] != delta_base].iterrows():
            print(f"  {task}/{model}  {r['feature_set']:<18} "
                  f"{r['balanced_acc_mean'] - b:+.3f}")

    print(
        "\n  time_only   control, not a result. Whatever it scores is"
        "\n              recoverable from session position alone, and every"
        "\n              other row inherits that as a caveat."
        "\n  clean       headline configuration: posture means removed."
        "\n  device      deployable hardware configuration — EDA front end +"
        "\n              the four-feature HR block + motion magnitude/flag,"
        "\n              exactly what the SEN0344/EDA/IMU stack can produce."
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "ablation_sweep.csv", index=False)
    print(f"\nsaved -> {RESULTS_DIR}/ablation_sweep.csv")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--task", choices=["binary", "3class", "both"], default="binary")
    ap.add_argument("--model", choices=["rf"], default="rf")
    ap.add_argument("--features", default="all",
                    help=f"one of: {', '.join(sorted(FEATURE_SETS))}")
    ap.add_argument("--sweep", action="store_true", help="run the ablation table")
    ap.add_argument("--baseline-n", type=int, default=N_REF_WINDOWS,
                    help=f"scaling reference size (default {N_REF_WINDOWS})")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--standardise", choices=["none", "baseline", "subject"],
                    default="baseline")
    ap.add_argument("--permutation", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.features not in FEATURE_SETS:
        print(f"unknown feature set '{args.features}'. "
              f"Available: {', '.join(sorted(FEATURE_SETS))}")
        return 1

    df = load_table(Path(args.cache))

    missing = [c for c in F.FEATURE_NAMES if c not in df.columns]
    if missing or TIME_COL not in df.columns:
        print(f"stale cache — missing {missing or [TIME_COL]}; rebuild with --force")
        return 1

    print(f"subjects={df['subject'].nunique()}  features={F.N_FEATURES}")
    print(df["label_name"].value_counts(normalize=True).round(3).to_string())

    baseline_ref_s = load_baseline_ref_s(Path(args.cache))
    if baseline_ref_s is not None:
        print(f"HR baseline reference: {baseline_ref_s:.0f}s "
              "(baked into hr_baseline_delta at cache-build time)")
    else:
        print("HR baseline reference: UNKNOWN — cache predates the metadata; "
              "rebuild with --force")

    # The diagnostic runs on RAW features: standardising rescales but does not
    # decorrelate, and raw is what the report should quote.
    if args.diagnose:
        diagnose_time_confound(df)
        return 0

    df, fallback_table = _standardise_core(df, args.standardise, n_ref=args.baseline_n)
    print(f"standardisation: {args.standardise} (n_ref={args.baseline_n})")
    if args.standardise != "none":
        scale_report(df)
        fallback_report(fallback_table)

    tasks = ["binary", "3class"] if args.task == "both" else [args.task]
    sets = SWEEP_ORDER if args.sweep else [args.features]

    sweep_rows = []
    for task in tasks:
        for fs in sets:
            folds, summary, evald, imp_df = run_loso(
                df, task, args.model, feature_set=fs, seed=args.seed,
                do_permutation=args.permutation,
                verbose=not (args.quiet or args.sweep),
            )
            summary["standardise"] = args.standardise
            summary["n_ref_windows"] = args.baseline_n
            if args.sweep:
                print(f"  {task:<7} {fs:<18} bal={summary['balanced_acc_mean']:.3f}")
            else:
                report(summary, evald, folds, imp_df)
            save(
                result_tag(task, args.model, fs, args.standardise, args.baseline_n),
                folds, summary, imp_df, evald,
            )
            sweep_rows.append(summary)

    if args.sweep:
        sweep_table(sweep_rows)
    else:
        print(f"\nsaved -> {RESULTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())