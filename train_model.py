"""
train_model.py — leave-one-subject-out evaluation on the WESAD feature table.

Produces the numbers for the report: per-class F1, macro F1, balanced
accuracy, aggregated confusion matrix, cross-fold feature importances with
spread, and — the reason this file was rewritten — a feature-set ablation
that quantifies how much of the result depends on features suspected of
encoding session order rather than affect.

Run:
    python train_model.py                                  # binary, RF, all features
    python train_model.py --diagnose                       # confound check, no training
    python train_model.py --sweep                          # the ablation, one table
    python train_model.py --task both --model both --sweep # everything, for the report
    python train_model.py --features clean --task 3class
    python train_model.py --standardise none               # scaling ablation

--------------------------------------------------------------------------
Why leave-one-subject-out, and why NOT a random split
--------------------------------------------------------------------------
Windows slide with a 5 s step over a 60 s window, so consecutive rows share
55 s of their source signal. A random shuffle puts near-duplicate rows either
side of the split and the resulting accuracy measures memorisation of a
recording, not generalisation to a person. Every split here is by subject and
an assertion enforces it on every fold.

--------------------------------------------------------------------------
The session-order confound, and what an ablation can and cannot show
--------------------------------------------------------------------------
WESAD's protocol runs in contiguous, time-ordered blocks: baseline, then
stress, then amusement, then meditation. Any feature that drifts
monotonically over a ~100 minute recording is therefore correlated with
elapsed time, and elapsed time is correlated with the label. Skin temperature
is the obvious candidate; mean per-axis acceleration (gravity projection, so
really posture) is the second.

IMPORTANT: an ablation bounds a feature block's CONTRIBUTION. It does not
identify the MECHANISM. If accuracy falls when TEMP is removed, that is
equally consistent with (a) the model was reading a clock, and (b) the model
was reading genuine stress-related distal cooling, which is real physiology
(Vinkers et al.). Distinguishing them needs the --diagnose correlation check
and the time_only probe, not the ablation alone. Report all three together.

Feature sets available to --features:
    all         every feature in features.FEATURE_NAMES
    no_temp     drop the whole TEMP block
    no_posture  drop acc_{x,y,z,mag}_mean (gravity projection = posture)
    clean       drop TEMP absolutes and posture means, KEEP temp_slope and
                temp_baseline_delta, which are far less time-confounded than
                temp_max/mean/min. The defensible headline configuration.
    eda_only    the EDA block alone — what the custom AFE contributes
    hr_only     the HR block alone — the weakest-transferring modality
    eda_hr      EDA + HR, no temperature and no motion
    time_only   PROBE, NOT A MODEL. Elapsed time as the single feature. If
                this alone scores well, the protocol's time-ordering is
                directly recoverable and every other number needs that
                caveat attached.
    clean_plus_time   REDUNDANCY PROBE. 'clean' with elapsed time added. If
                it matches 'clean', the physiological features already carry
                everything the clock offers. If it improves substantially,
                they carry information session position does not.
    eda_plus_time     Same probe against the EDA block alone.

--------------------------------------------------------------------------
Standardisation modes (--standardise)
--------------------------------------------------------------------------
none      Raw features. Ablation baseline.
baseline  DEFAULT. Each subject centred on the median and scaled by the IQR
          of their first N_REF_WINDOWS accepted windows. Causal, label-blind,
          and exactly what the live system can do from the first minutes of
          usable wear. Robust statistics are used deliberately: mean/SD on a
          short reference window produced z-scores in the hundreds and left
          subjects on incomparable scales. Degenerate divisors fall back
          through a cascade (reference IQR -> subject IQR -> cohort IQR)
          rather than onto a fixed floor, which was tried and made the
          scaling worse by a factor of ~90.
subject   Whole-recording per-subject statistics. Stronger but TRANSDUCTIVE:
          it uses the full test recording before predicting on it, which the
          live system cannot. Comparison only; state the assumption if
          reported.

--------------------------------------------------------------------------
Reading the result
--------------------------------------------------------------------------
WESAD's published all-wrist ceiling is ~76% (3-class) and ~88% (binary).
Landing near those reproduces the benchmark. Landing ABOVE them with a
REDUCED feature set (no HRV, no respiration) is a reason to hunt for a
confound, not to celebrate.
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

# Must match build_dataset.BASELINE_REF_S. Retained because build_dataset
# uses it for the baseline-delta features; standardisation no longer keys on
# it (see N_REF_WINDOWS).
BASELINE_REF_S = 600.0

# Standardisation reference: the first N ACCEPTED windows per subject, not a
# wall-clock cutoff.
#
# Why this changed. A time cutoff assumes every recording opens with usable
# windows. S6's does not — its label track begins with undefined/transient
# codes, every early window was rejected, and the subject ended up with ZERO
# reference windows before t=600 s. The old code then silently fell back to
# whole-recording statistics for that subject alone, applying a different
# transform to S6 than to the 14 subjects it was compared against. S6's
# features came out compressed (max |z| 9 against a cohort median of 34), the
# model saw nothing resembling stress, and the fold collapsed to balanced
# accuracy exactly 0.500 — despite S6 having the second-strongest EDA
# response in the whole cohort (Cohen's d 11.2 on eda_scl_mean).
#
# A window count is robust to that and remains causal: these are simply the
# earliest usable data available, which is also what the live system will do
# — take a resting reference from the first minutes of usable wear rather
# than from whenever the clock happened to start.
N_REF_WINDOWS = 100

# Hard clip applied after robust scaling. No legitimate robust z-score
# reaches this; anything beyond it is a degenerate divisor rather than a real
# excursion, and clipping stops one feature dominating the tree splits.
Z_CLIP = 20.0

CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")

# Binary task: stress against everything else. Standard WESAD framing.
STRESS_LABEL = 2

# Published all-wrist ceilings, for the sanity check.
CEILINGS = {"binary": 0.88, "3class": 0.76}

# Gravity projection: mean acceleration encodes wrist orientation, i.e. posture.
POSTURE_FEATURES = ["acc_x_mean", "acc_y_mean", "acc_z_mean", "acc_mag_mean"]

# Absolute temperature drifts monotonically over a long recording. Slope and
# baseline-delta are differential and far less time-confounded.
TEMP_ABSOLUTE = ["temp_mean", "temp_min", "temp_max", "temp_range"]

TIME_COL = "t_start"


def _drop(cols, remove) -> list:
    rm = set(remove)
    return [c for c in cols if c not in rm]


FEATURE_SETS = {
    "all": list(F.FEATURE_NAMES),
    "no_temp": _drop(F.FEATURE_NAMES, F.TEMP_FEATURES),
    "no_posture": _drop(F.FEATURE_NAMES, POSTURE_FEATURES),
    "clean": _drop(F.FEATURE_NAMES, TEMP_ABSOLUTE + POSTURE_FEATURES),
    "eda_only": list(F.EDA_FEATURES),
    "hr_only": list(F.HR_FEATURES),
    "eda_hr": list(F.EDA_FEATURES) + list(F.HR_FEATURES),
    "time_only": [TIME_COL],
    # Redundancy probes. If clean_plus_time ~= clean, the physiological
    # features already encode everything elapsed time offers, i.e. they are
    # largely redundant with session position. A substantial improvement
    # means they carry information the clock does not.
    "clean_plus_time": _drop(F.FEATURE_NAMES, TEMP_ABSOLUTE + POSTURE_FEATURES)
    + [TIME_COL],
    "eda_plus_time": list(F.EDA_FEATURES) + [TIME_COL],
}

# Order used by --sweep. Chosen so the table reads as a narrative: full
# result, then each suspected confound removed, then the probe.
SWEEP_ORDER = [
    "all",
    "no_posture",
    "no_temp",
    "clean",
    "eda_only",
    "time_only",
    "clean_plus_time",
    "eda_plus_time",
]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_table(cache_dir: Path) -> pd.DataFrame:
    """Load the combined table, or concatenate per-subject caches."""
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
        raise FileNotFoundError(
            f"no feature cache in {cache_dir} — run build_dataset.py --combine first"
        )
    frames = [
        pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p) for p in parts
    ]
    df = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(parts)} per-subject caches  ({len(df)} rows)")
    return df


def make_target(df: pd.DataFrame, task: str):
    """Return (y, class_names) for the requested task."""
    if task == "binary":
        y = (df["label"].to_numpy() == STRESS_LABEL).astype(int)
        return y, ["non-stress", "stress"]
    y = df["label"].to_numpy()
    lookup = {1: "baseline", 2: "stress", 3: "amusement"}
    return y, [lookup.get(int(c), str(c)) for c in sorted(np.unique(y))]


# --------------------------------------------------------------------------
# Confound diagnostic
# --------------------------------------------------------------------------


def diagnose_time_confound(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Per-subject correlation of each feature with elapsed recording time.

    This is the check that distinguishes 'the model reads a clock' from 'the
    model reads physiology'. A feature whose mean |r| with t_start is high
    across subjects is, within WESAD's time-ordered protocol, substantially a
    proxy for which condition block the window came from.

    Interpretation: |r| > 0.8 is a strong clock; 0.5-0.8 warrants comment;
    below ~0.3 the feature is not meaningfully time-coupled.
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
            if ok.sum() < 10 or np.std(v[ok]) == 0:
                rs[c] = np.nan
                continue
            rs[c] = float(np.corrcoef(t[ok], v[ok])[0, 1])
        per_subject[sid] = rs

    raw = pd.DataFrame(per_subject).T  # subjects x features
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

    print("\n" + "=" * 68)
    print("TIME-CONFOUND DIAGNOSTIC")
    print("per-subject |correlation| between each feature and elapsed time")
    print("=" * 68)
    for _, r in out.head(top).iterrows():
        if r["mean_abs_r"] > 0.8:
            flag = "  <-- strong"
        elif r["mean_abs_r"] > 0.5:
            flag = "  <-- moderate"
        else:
            flag = ""
        print(
            f"  {r['feature']:<24} |r|={r['mean_abs_r']:.3f} "
            f"+/-{r['sd_abs_r']:.3f}  signed={r['mean_signed_r']:+.3f}{flag}"
        )

    strong = out[out["mean_abs_r"] > 0.8]["feature"].tolist()
    if strong:
        print(f"\n  {len(strong)} feature(s) with |r| > 0.8 against elapsed time:")
        print(f"    {', '.join(strong)}")
        print(
            "  Within a time-ordered protocol these are partly proxies for\n"
            "  which block the window came from. Run --sweep and report the\n"
            "  'clean' configuration alongside 'all'."
        )
    else:
        print("\n  No feature exceeds |r| > 0.8. The clock hypothesis is weak;")
        print("  a drop under ablation is more likely genuine physiology.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "time_confound.csv", index=False)
    print(f"\nsaved -> {RESULTS_DIR}/time_confound.csv")
    return out


# --------------------------------------------------------------------------
# Standardisation
# --------------------------------------------------------------------------


def standardise(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Per-subject robust scaling. Never uses labels, in any mode.

    Uses median and IQR rather than mean and SD. The reference window is
    short and contains outliers, and a near-zero SD divisor produced
    z-scores in the hundreds — leaving subjects on incomparable scales,
    which is the opposite of what standardising is for.

    A subject with too few reference windows is a HARD ERROR, not a silent
    fallback. The previous fallback applied whole-recording statistics to one
    subject while the other 14 got a baseline-window transform, and that
    single inconsistency destroyed the fold without any visible failure.

    t_start is deliberately left unscaled — it is a probe input, and scaling
    it would change what the time_only result means.
    """
    if mode == "none":
        return df

    out = df.copy()
    cols = [c for c in F.FEATURE_NAMES if c in df.columns]

    # Cohort-wide spread, used as the third rung of the divisor cascade.
    cohort_iqr = df[cols].quantile(0.75) - df[cols].quantile(0.25)

    for sid, idx in df.groupby("subject").groups.items():
        block = df.loc[idx, cols]

        if mode == "baseline":
            # First N accepted windows, in recording order.
            ref = df.loc[idx].sort_values(TIME_COL).head(N_REF_WINDOWS)[cols]
            if len(ref) < 5:
                raise ValueError(
                    f"{sid}: only {len(ref)} windows available for a "
                    "standardisation reference. Investigate the subject rather "
                    "than working around it — a per-subject fallback is what "
                    "silently invalidated S6's fold previously."
                )
        elif mode == "subject":
            ref = block
        else:
            raise ValueError(f"unknown standardise mode: {mode}")

        centre = ref.median()
        iqr = ref.quantile(0.75) - ref.quantile(0.25)

        # Divisor cascade: reference IQR -> this subject's whole-recording
        # IQR -> cohort IQR -> 1.0.
        #
        # A fixed floor was tried and was actively harmful: a feature whose
        # IQR collapses to zero in the short reference window got divided by
        # the floor, multiplying it by 1/floor and pushing median max|z| from
        # 34 to 2951. A feature that is constant over 100 reference windows is
        # usually perfectly variable over the full recording, so widening the
        # window is the right fallback — it keeps the divisor on the feature's
        # own real scale instead of inventing one.
        subj_iqr = block.quantile(0.75) - block.quantile(0.25)
        scale = iqr.where(iqr > 0, subj_iqr)
        scale = scale.where(scale > 0, cohort_iqr)
        scale = scale.where(scale > 0, 1.0)

        out.loc[idx, cols] = ((block - centre) / scale).to_numpy()

    out[cols] = out[cols].clip(-Z_CLIP, Z_CLIP).fillna(0.0)
    return out


def scale_report(df: pd.DataFrame, std_df: pd.DataFrame) -> None:
    """Post-standardisation magnitude check across subjects.

    Subjects should land on comparable scales. One subject reaching
    magnitudes the others never produce means the model has seen nothing like
    them in training, and that fold is lost to scaling rather than to
    physiology — the S6 failure mode. Printed every run so a regression is
    visible rather than discovered in a fold table.
    """
    cols = [c for c in F.FEATURE_NAMES if c in std_df.columns]
    rows = []
    for sid, block in std_df.groupby("subject"):
        v = block[cols].to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        rows.append(
            {
                "subject": sid,
                "n_windows": len(block),
                "first_t": float(df[df["subject"] == sid][TIME_COL].min()),
                "p99_abs": float(np.percentile(np.abs(v), 99)) if v.size else np.nan,
                "max_abs": float(np.max(np.abs(v))) if v.size else np.nan,
            }
        )
    rep = pd.DataFrame(rows)
    med = rep["max_abs"].median()
    outliers = rep[rep["max_abs"] > 5 * med]

    print(
        f"scale check: median max|z| = {med:.1f}, "
        f"range {rep['max_abs'].min():.1f}..{rep['max_abs'].max():.1f}"
    )
    if len(outliers):
        print("  SCALE OUTLIERS — these folds are not comparable to the rest:")
        for _, r in outliers.iterrows():
            print(
                f"    {r['subject']}: max|z|={r['max_abs']:.1f} "
                f"(first window at t={r['first_t']:.0f}s)"
            )


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


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
            max_iter=300,
            learning_rate=0.08,
            l2_regularization=1.0,
            random_state=seed,
        )
    raise ValueError(f"unknown model: {kind}")


# --------------------------------------------------------------------------
# LOSO
# --------------------------------------------------------------------------


def resolve_features(name: str, df: pd.DataFrame) -> list:
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
    """One full leave-one-subject-out sweep.

    Fold order is the sorted subject list and the seed is fixed, so results
    from different feature sets are computed over identical folds and are
    directly comparable rather than differing by fold noise.
    """
    cols = resolve_features(feature_set, df)
    X_all = df[cols].to_numpy(dtype=np.float64)
    y_all, class_names = make_target(df, task)
    subjects = df["subject"].to_numpy()
    uniq = sorted(pd.unique(subjects))

    fold_rows = []
    y_true_all, y_pred_all = [], []
    importances = []

    for held in uniq:
        te = subjects == held
        tr = ~te

        # A subject appearing in both halves is the most damaging bug
        # available here, and it is silent. Assert it every fold.
        assert held not in set(subjects[tr]), "subject leaked across folds"

        y_tr, y_te = y_all[tr], y_all[te]
        if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
            if verbose:
                print(f"  {held}: skipped (degenerate fold)")
            continue

        imp = SimpleImputer(strategy="median").fit(X_all[tr])
        X_tr, X_te = imp.transform(X_all[tr]), imp.transform(X_all[te])

        clf = make_model(model_kind, seed).fit(X_tr, y_tr)
        y_hat = clf.predict(X_te)

        y_true_all.append(y_te)
        y_pred_all.append(y_hat)

        fold_rows.append(
            {
                "subject": held,
                "n_test": int(te.sum()),
                "accuracy": float((y_hat == y_te).mean()),
                "balanced_acc": float(balanced_accuracy_score(y_te, y_hat)),
                "macro_f1": float(
                    f1_score(y_te, y_hat, average="macro", zero_division=0)
                ),
            }
        )
        if verbose:
            r = fold_rows[-1]
            print(
                f"  {held}: acc={r['accuracy']:.3f} bal={r['balanced_acc']:.3f} "
                f"macroF1={r['macro_f1']:.3f}  (n={r['n_test']})"
            )

        if do_permutation:
            pi = permutation_importance(
                clf, X_te, y_te, n_repeats=5, random_state=seed, n_jobs=-1
            )
            importances.append(pi.importances_mean)
        elif hasattr(clf, "feature_importances_"):
            importances.append(clf.feature_importances_)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    folds = pd.DataFrame(fold_rows)

    summary = {
        "task": task,
        "model": model_kind,
        "feature_set": feature_set,
        "n_features": len(cols),
        "n_folds": len(folds),
        "accuracy_mean": float(folds["accuracy"].mean()),
        "accuracy_sd": float(folds["accuracy"].std()),
        "balanced_acc_mean": float(folds["balanced_acc"].mean()),
        "balanced_acc_sd": float(folds["balanced_acc"].std()),
        "macro_f1_mean": float(folds["macro_f1"].mean()),
        "macro_f1_sd": float(folds["macro_f1"].std()),
        "pooled_macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
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


def ceiling_note(task: str, acc: float) -> None:
    """Compare against the published ceiling, with the necessary caveat.

    The comparison is largely VOID for this dataset. WESAD's protocol runs
    the same condition blocks in the same order at similar wall-clock offsets
    for every subject, so elapsed time alone (the time_only probe) reaches
    ~0.96 binary / ~0.93 3-class balanced accuracy under LOSO. Any published
    figure on WESAD — including the ~0.88 / ~0.76 wrist ceilings — is an
    upper bound of uncertain composition, because no such paper reports a
    session-position control. Quote the ceiling for context only, and quote
    the time_only control alongside it.
    """
    ceil = CEILINGS.get(task)
    if ceil is None:
        return
    delta = acc - ceil
    print(f"\nWESAD published all-wrist ceiling: ~{ceil:.2f}   you: {acc:.3f}")
    print(f"  delta {delta:+.3f}")
    print(
        "  CAVEAT: elapsed time alone scores above both published ceilings on\n"
        "  this dataset (see --sweep, time_only). Treat the comparison as\n"
        "  context, not validation, and report the time control with it."
    )


def report(summary, evald, folds, imp_df):
    y_true, y_pred, class_names = evald

    print("\n" + "=" * 68)
    print(
        f"{summary['task'].upper()}  /  {summary['model'].upper()}  /  "
        f"features={summary['feature_set']} ({summary['n_features']})  /  "
        f"{summary['n_folds']} folds LOSO"
    )
    print("=" * 68)
    print(
        f"accuracy      {summary['accuracy_mean']:.3f} +/- {summary['accuracy_sd']:.3f}"
    )
    print(
        f"balanced acc  {summary['balanced_acc_mean']:.3f} "
        f"+/- {summary['balanced_acc_sd']:.3f}"
    )
    print(
        f"macro F1      {summary['macro_f1_mean']:.3f} +/- {summary['macro_f1_sd']:.3f}"
    )
    print(
        f"fold range    {summary['worst_fold_macro_f1']:.3f} .. "
        f"{summary['best_fold_macro_f1']:.3f} macro F1"
    )

    # A bimodal fold distribution — near-perfect on some subjects, at chance
    # on others — is the signature of a deterministic separator rather than a
    # noisy physiological one, and is worth flagging explicitly.
    near_perfect = int((folds["macro_f1"] > 0.97).sum())
    at_chance = int((folds["balanced_acc"] < 0.55).sum())
    if near_perfect and at_chance:
        print(
            f"\n  NOTE: {near_perfect} fold(s) near-perfect and {at_chance} at "
            "chance.\n  That bimodality suggests a deterministic separator rather "
            "than\n  graded physiology. Check --diagnose."
        )

    print("\nper-class (pooled across folds)")
    print(
        classification_report(
            y_true, y_pred, target_names=class_names, digits=3, zero_division=0
        )
    )

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
            print(
                f"  {r['feature']:<24} {r['importance_mean']:.4f} "
                f"+/- {r['importance_sd']:.4f}"
            )

        total = imp_df["importance_mean"].sum() or 1.0
        shares = {}
        for name, block in (
            ("EDA", F.EDA_FEATURES),
            ("HR", F.HR_FEATURES),
            ("IMU", F.IMU_FEATURES),
            ("TEMP", F.TEMP_FEATURES),
        ):
            shares[name] = (
                imp_df[imp_df["feature"].isin(block)]["importance_mean"].sum() / total
            )
        print(
            "\nblock share:  " + "   ".join(f"{k} {v:.1%}" for k, v in shares.items())
        )
        if shares["HR"] > shares["EDA"]:
            print(
                "  NOTE: HR outweighs EDA. HR transfers worst to your hardware\n"
                "  (NeuroKit2 beat detection vs the SEN0344's undocumented\n"
                "  on-board estimator). Expect a larger deployment drop than\n"
                "  these numbers suggest, and say so."
            )
        if shares["TEMP"] > 0.25:
            print(
                "  NOTE: TEMP carries >25% of importance. On your hardware the\n"
                "  LM75BD sits on the PCB and reads skin temperature mixed with\n"
                "  board self-heating, so this block transfers poorly too."
            )

    ceiling_note(summary["task"], summary["accuracy_mean"])


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
    """Side-by-side comparison of feature sets over identical folds."""
    print("\n" + "=" * 78)
    print("ABLATION SWEEP  (identical folds and seed across rows)")
    print("=" * 78)
    print(
        f"{'task':<8}{'model':<6}{'features':<12}{'n':>4}"
        f"{'acc':>9}{'bal':>9}{'macroF1':>10}{'+/-':>8}"
    )
    for r in rows:
        print(
            f"{r['task']:<8}{r['model']:<6}{r['feature_set']:<12}"
            f"{r['n_features']:>4}{r['accuracy_mean']:>9.3f}"
            f"{r['balanced_acc_mean']:>9.3f}{r['macro_f1_mean']:>10.3f}"
            f"{r['macro_f1_sd']:>8.3f}"
        )

    # Deltas against the 'all' row within each task/model combination.
    print("\ndelta vs 'all' (balanced accuracy)")
    for (task, model), grp in pd.DataFrame(rows).groupby(["task", "model"]):
        base = grp[grp["feature_set"] == "all"]
        if base.empty:
            continue
        b = float(base["balanced_acc_mean"].iloc[0])
        for _, r in grp.iterrows():
            if r["feature_set"] == "all":
                continue
            print(
                f"  {task}/{model}  {r['feature_set']:<12} "
                f"{r['balanced_acc_mean'] - b:+.3f}"
            )

    print(
        "\nHow to read this."
        "\n  time_only        whatever this scores is recoverable from session"
        "\n                   position alone. It is the control, not a result."
        "\n                   Every other row inherits it as a caveat, because"
        "\n                   any time-correlated feature gets label information"
        "\n                   for free within a fixed-order protocol."
        "\n  no_temp          bounds DEPENDENCE on temperature. It does not"
        "\n                   prove temperature is an artefact — distal cooling"
        "\n                   under stress is real physiology (Vinkers et al.)."
        "\n                   Cross-reference --diagnose."
        "\n  clean            the defensible headline configuration: temperature"
        "\n                   absolutes and posture means removed, differential"
        "\n                   temperature features kept."
        "\n  eda_only         the honest floor for the custom EDA front-end's"
        "\n                   own contribution."
        "\n  clean_plus_time  if this matches 'clean', the physiology is largely"
        "\n                   redundant with the clock. If it beats 'clean'"
        "\n                   substantially, the physiology carries independent"
        "\n                   information — the better outcome."
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
    ap.add_argument("--model", choices=["rf", "hgb", "both"], default="rf")
    ap.add_argument(
        "--features",
        choices=sorted(FEATURE_SETS),
        default="all",
        help="feature subset (ignored when --sweep is given)",
    )
    ap.add_argument("--sweep", action="store_true", help="run the ablation sweep")
    ap.add_argument(
        "--diagnose", action="store_true", help="time-confound check, then exit"
    )
    ap.add_argument(
        "--standardise", choices=["none", "baseline", "subject"], default="baseline"
    )
    ap.add_argument("--permutation", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true", help="suppress per-fold lines")
    args = ap.parse_args()

    df = load_table(Path(args.cache))

    missing = [c for c in F.FEATURE_NAMES if c not in df.columns]
    if missing:
        print(f"feature table is stale — missing {missing}\nrebuild with --force")
        return 1
    if TIME_COL not in df.columns:
        print(f"'{TIME_COL}' column absent — rebuild the cache to enable time probes")
        return 1

    print(f"subjects={df['subject'].nunique()}  features={len(F.FEATURE_NAMES)}")
    print(df["label_name"].value_counts(normalize=True).round(3).to_string())

    # Diagnostic runs on RAW features — standardising first would rescale but
    # not decorrelate, and raw is what the report should quote.
    if args.diagnose:
        diagnose_time_confound(df)
        return 0

    raw_df = df
    df = standardise(df, args.standardise)
    print(f"standardisation: {args.standardise}")
    if args.standardise != "none":
        scale_report(raw_df, df)

    tasks = ["binary", "3class"] if args.task == "both" else [args.task]
    models = ["rf", "hgb"] if args.model == "both" else [args.model]
    sets = SWEEP_ORDER if args.sweep else [args.features]

    sweep_rows = []
    for task in tasks:
        for kind in models:
            for fs in sets:
                if not args.quiet:
                    print(f"\n--- LOSO: {task} / {kind} / {fs} ---")
                folds, summary, evald, imp_df = run_loso(
                    df,
                    task,
                    kind,
                    feature_set=fs,
                    seed=args.seed,
                    do_permutation=args.permutation,
                    verbose=not (args.quiet or args.sweep),
                )
                summary["standardise"] = args.standardise
                if args.sweep:
                    print(
                        f"  {fs:<12} acc={summary['accuracy_mean']:.3f} "
                        f"bal={summary['balanced_acc_mean']:.3f} "
                        f"macroF1={summary['macro_f1_mean']:.3f}"
                    )
                    if fs == "time_only":
                        print(
                            "    ^ PROBE: elapsed time as the only input. "
                            "This is not a model."
                        )
                else:
                    report(summary, evald, folds, imp_df)
                save(
                    f"{task}_{kind}_{fs}_{args.standardise}",
                    folds,
                    summary,
                    imp_df,
                    evald,
                )
                sweep_rows.append(summary)

    if args.sweep:
        sweep_table(sweep_rows)
    else:
        print(f"\nsaved -> {RESULTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())