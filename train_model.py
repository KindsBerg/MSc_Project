"""
train_model.py — leave-one-subject-out evaluation on the WESAD feature table.

    python train_model.py --diagnose                 # time-confound check
    python train_model.py --features clean           # full report, one config
    python train_model.py --task both --sweep        # the ablation table

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


def standardise(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Per-subject robust scaling. Never uses labels.

    Median and IQR, not mean and SD: a near-zero SD divisor on a short
    reference window produced z-scores in the hundreds. Divisor cascade is
    reference IQR -> subject IQR -> cohort IQR; a fixed floor was tried and
    pushed median max|z| from 34 to 2951.

    Too few reference windows is a hard error. A per-subject fallback is what
    silently invalidated S6's fold.

    t_start is left unscaled — it is a probe input.
    """
    if mode == "none":
        return df

    out = df.copy()
    cols = [c for c in F.FEATURE_NAMES if c in df.columns]
    cohort_iqr = df[cols].quantile(0.75) - df[cols].quantile(0.25)

    for sid, idx in df.groupby("subject").groups.items():
        block = df.loc[idx, cols]
        if mode == "baseline":
            ref = df.loc[idx].sort_values(TIME_COL).head(N_REF_WINDOWS)[cols]
            if len(ref) < 5:
                raise ValueError(f"{sid}: only {len(ref)} reference windows")
        elif mode == "subject":
            ref = block  # transductive; comparison only
        else:
            raise ValueError(f"unknown standardise mode: {mode}")

        centre = ref.median()
        iqr = ref.quantile(0.75) - ref.quantile(0.25)
        subj_iqr = block.quantile(0.75) - block.quantile(0.25)
        scale = iqr.where(iqr > 0, subj_iqr).where(lambda s: s > 0, cohort_iqr)
        scale = scale.where(scale > 0, 1.0)
        out.loc[idx, cols] = ((block - centre) / scale).to_numpy()

    out[cols] = out[cols].clip(-Z_CLIP, Z_CLIP).fillna(0.0)
    return out


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
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, l2_regularization=1.0, random_state=seed
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
    X_all = df[cols].to_numpy(dtype=np.float64)
    y_all, class_names = make_target(df, task)
    subjects = df["subject"].to_numpy()

    fold_rows, y_true_all, y_pred_all, importances = [], [], [], []

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
        y_hat = clf.predict(imp.transform(X_all[te]))

        y_true_all.append(y_te)
        y_pred_all.append(y_hat)
        fold_rows.append(
            {
                "subject": held,
                "n_test": int(te.sum()),
                "accuracy": float((y_hat == y_te).mean()),
                "balanced_acc": float(balanced_accuracy_score(y_te, y_hat)),
                "macro_f1": float(f1_score(y_te, y_hat, average="macro", zero_division=0)),
            }
        )
        if verbose:
            r = fold_rows[-1]
            print(
                f"  {held}: acc={r['accuracy']:.3f} bal={r['balanced_acc']:.3f} "
                f"macroF1={r['macro_f1']:.3f}  (n={r['n_test']})"
            )

        if do_permutation:
            importances.append(
                permutation_importance(
                    clf, imp.transform(X_all[te]), y_te, n_repeats=5,
                    random_state=seed, n_jobs=-1,
                ).importances_mean
            )
        elif hasattr(clf, "feature_importances_"):
            importances.append(clf.feature_importances_)

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--task", choices=["binary", "3class", "both"], default="binary")
    ap.add_argument("--model", choices=["rf", "hgb", "both"], default="rf")
    ap.add_argument("--features", default="all",
                    help=f"one of: {', '.join(sorted(FEATURE_SETS))}")
    ap.add_argument("--sweep", action="store_true")
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

    # Diagnostic runs on RAW features — standardising rescales but does not
    # decorrelate, and raw is what the report should quote.
    if args.diagnose:
        diagnose_time_confound(df)
        return 0

    df = standardise(df, args.standardise)
    print(f"standardisation: {args.standardise}")
    if args.standardise != "none":
        scale_report(df)

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
                if args.sweep:
                    print(f"  {task:<7} {fs:<18} bal={summary['balanced_acc_mean']:.3f}")
                else:
                    report(summary, evald, folds, imp_df)
                save(f"{task}_{kind}_{fs}_{args.standardise}", folds, summary, imp_df, evald)
                sweep_rows.append(summary)

    if args.sweep:
        sweep_table(sweep_rows)
    else:
        print(f"\nsaved -> {RESULTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())