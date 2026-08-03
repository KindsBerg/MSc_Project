"""
verify_subjects.py — targeted forensics for the two folds that survived every
fix to the reference-window pipeline.

    python verify_subjects.py                       # S2 and S3, eda_only
    python verify_subjects.py --subjects S2 S3 S17  # add S17
    python verify_subjects.py --features clean      # different feature set
    python verify_subjects.py --n-ref 30 100        # compare two buffer sizes

Two distinct hypotheses are under test, one per subject. They are NOT the same
failure and should not be reported as one.

S2 sits at exactly 0.500 balanced accuracy for every N <= 30. Balanced accuracy
of exactly 0.500 on a two-class problem is the signature of a predictor that
emitted a single class for the entire fold, not of a model that got half its
guesses wrong. The candidate causes, in the order this script tests them:

    (a) degenerate prediction   — one class emitted for every test window
    (b) scaling collapse        — a near-zero reference IQR inflating z scores
                                  until the fold sits outside the training
                                  manifold. This is the S6 failure mode, which
                                  was fixed once already, so it is worth
                                  excluding explicitly rather than assuming.
    (c) no separation           — the subject's stress windows genuinely do not
                                  differ from their baseline windows

S3 is flat at 0.62-0.70 across a 40x range of N, and earlier forensics gave it
Cohen's d = 1.01 on SCL. A subject with a large effect size that a cross-subject
model cannot use is the signature of an inverted or idiosyncratic response, not
of a scaling problem. The specific thing to look for is SIGN DISAGREEMENT: a
feature that moves up under stress for the cohort and down for this subject is
actively misleading to a model trained on everyone else, and no amount of
reference-window tuning will recover it.

Reads the cache and reuses train_model's standardisation and fold machinery, so
it cannot disagree with the sweep about how features were scaled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
import train_model as T

RESULTS_DIR = Path("results")
DEFAULT_SUBJECTS = ["S2", "S3"]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD effect size. Sign convention: positive means group a is higher."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if not np.isfinite(pooled) or pooled == 0:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled)


def class_balance(df: pd.DataFrame, subjects: list, task: str) -> None:
    """Test-set composition. A fold cannot score above chance on a class it
    does not contain, and balanced accuracy is undefined in a way that silently
    reads as 0.500 when one class is absent."""
    print("\n" + "=" * 72)
    print("CLASS BALANCE  (held-out set composition)")
    print("=" * 72)
    y_all, names = T.make_target(df, task)
    subj = df["subject"].to_numpy()

    print(f"{'subject':<10}{'windows':>9}" + "".join(f"{n[:10]:>12}" for n in names)
          + f"{'minority':>11}")
    for sid in sorted(pd.unique(subj), key=lambda s: (s not in subjects, s)):
        m = subj == sid
        counts = [int((y_all[m] == c).sum()) for c in sorted(np.unique(y_all))]
        frac = min(counts) / max(sum(counts), 1)
        mark = "  <--" if sid in subjects else ""
        print(f"{sid:<10}{int(m.sum()):>9}"
              + "".join(f"{c:>12}" for c in counts)
              + f"{frac:>11.3f}{mark}")
        if 0 in counts:
            print(f"           WARNING {sid} is missing a class entirely — "
                  "balanced accuracy for this fold is not interpretable")


def prediction_profile(df: pd.DataFrame, subjects: list, task: str,
                       model_kind: str, feature_set: str, seed: int) -> pd.DataFrame:
    """What the model actually emitted for each target fold.

    Distinguishes 'wrong half the time' from 'emitted one class'. Only the
    second is a degenerate predictor, and only the second is fixable by
    changing how the fold is scaled.
    """
    cols = T.resolve_features(feature_set, df)
    rows = []

    print("\n" + "=" * 72)
    print(f"PREDICTION PROFILE  ({feature_set}, {len(cols)} features)")
    print("=" * 72)

    for fold in T._loso_folds(df, cols, task, model_kind, seed,
                              do_permutation=False, verbose=False):
        sid = fold["subject"]
        if sid not in subjects:
            continue
        y_t, y_p = fold["y_true"], fold["y_pred"]
        names = fold["class_names"]

        pred_counts = {int(c): int((y_p == c).sum()) for c in np.unique(y_t)}
        true_counts = {int(c): int((y_t == c).sum()) for c in np.unique(y_t)}
        n_pred_classes = len(np.unique(y_p))
        bal = T.balanced_accuracy_score(y_t, y_p)

        print(f"\n{sid}:  balanced acc = {bal:.4f}   "
              f"distinct predicted classes = {n_pred_classes}")
        print(f"  true      " + "  ".join(
            f"{names[i]}={true_counts.get(c, 0)}"
            for i, c in enumerate(sorted(true_counts))))
        print(f"  predicted " + "  ".join(
            f"{names[i]}={pred_counts.get(c, 0)}"
            for i, c in enumerate(sorted(true_counts))))

        if n_pred_classes == 1:
            only = int(np.unique(y_p)[0])
            label = names[sorted(true_counts).index(only)] if only in true_counts \
                else str(only)
            print(f"  DEGENERATE: emitted '{label}' for all {len(y_p)} windows.")
            print("    Not a marginal fold. The model placed every window of this")
            print("    subject on one side of every split it cared about, which")
            print("    points at the subject's position in feature space rather")
            print("    than at the difficulty of its stress response.")
        else:
            per_class = {
                names[i]: float((y_p[y_t == c] == c).mean())
                for i, c in enumerate(sorted(true_counts))
            }
            print("  per-class recall  " + "  ".join(
                f"{k}={v:.3f}" for k, v in per_class.items()))
            minority_share = min(pred_counts.values()) / max(len(y_p), 1)
            if minority_share < 0.05:
                print(f"  NEAR-DEGENERATE: the minority class is only "
                      f"{minority_share:.1%} of predictions.")
                print("    Reads as a two-class predictor but behaves as a one-class")
                print("    one. Treat this the same as the degenerate case.")
            else:
                print("    Both classes emitted in quantity, so this is a separation")
                print("    problem, not a degenerate predictor. Rescaling will not")
                print("    help; look at response direction above.")

        rows.append({"subject": sid, "balanced_acc": float(bal),
                     "n_pred_classes": n_pred_classes, "n_test": len(y_t)})

    return pd.DataFrame(rows)


def scaling_health(raw: pd.DataFrame, subjects: list, n_ref: int,
                   feature_set: str) -> pd.DataFrame:
    """Reference-block condition and post-standardisation magnitude.

    A near-zero reference IQR is the divisor that produced the original S6
    collapse. Checked here per feature, not as a subject-level summary, so a
    single bad divisor cannot hide behind well-behaved neighbours.
    """
    cols = [c for c in T.FEATURE_SETS[feature_set] if c in raw.columns
            and c != T.TIME_COL]
    std = T.standardise(raw, "baseline", n_ref=n_ref)
    rows = []

    print("\n" + "=" * 72)
    print(f"SCALING HEALTH  (n_ref={n_ref}, {feature_set})")
    print("=" * 72)
    print(f"{'subject':<10}{'p99|z|':>9}{'max|z|':>9}{'clipped':>10}"
          f"{'degen div':>11}{'ref rows':>10}")

    cohort_p99 = []
    for sid, idx in raw.groupby("subject").groups.items():
        ref = raw.loc[idx].sort_values(T.TIME_COL).head(n_ref)[cols]
        iqr = ref.quantile(0.75) - ref.quantile(0.25)
        degenerate = [c for c in cols if not np.isfinite(iqr[c]) or iqr[c] <= 0]

        v = std.loc[idx, cols].to_numpy(dtype=np.float64)
        v = np.abs(v[np.isfinite(v)])
        p99 = float(np.percentile(v, 99)) if v.size else np.nan
        mx = float(v.max()) if v.size else np.nan
        clip = float((v >= T.Z_CLIP).mean()) if v.size else np.nan
        cohort_p99.append(p99)

        mark = "  <--" if sid in subjects else ""
        rows.append({"subject": sid, "n_ref": n_ref, "p99_abs_z": p99,
                     "max_abs_z": mx, "clipped_frac": clip,
                     "n_degenerate_divisors": len(degenerate),
                     "degenerate_features": ";".join(degenerate)})
        print(f"{sid:<10}{p99:>9.2f}{mx:>9.2f}{clip:>10.3%}"
              f"{len(degenerate):>11}{len(ref):>10}{mark}")
        if sid in subjects and degenerate:
            print(f"           zero/undefined reference IQR: "
                  f"{', '.join(degenerate)}")

    med = float(np.nanmedian(cohort_p99))
    print(f"\n  cohort median p99|z| = {med:.2f}")
    for r in rows:
        if r["subject"] in subjects:
            ratio = r["p99_abs_z"] / med if med else np.nan
            verdict = ("OUTLIER — this fold is scaled onto a different range "
                       "than the training subjects"
                       if ratio > 3 or ratio < 1 / 3 else
                       "in range — scaling is NOT the explanation for this fold")
            print(f"  {r['subject']}: p99|z| = {r['p99_abs_z']:.2f} "
                  f"({ratio:.2f}x cohort median) — {verdict}")

    return pd.DataFrame(rows)


def response_direction(raw: pd.DataFrame, subjects: list, feature_set: str,
                       top: int = 12) -> pd.DataFrame:
    """Per-feature effect size for each subject against the cohort consensus.

    The load-bearing column is 'sign_agrees'. A feature with a large |d| whose
    sign opposes the cohort is worse for a cross-subject model than a feature
    with no effect at all: the model has learned a direction from 14 subjects
    and this subject moves the other way, so the feature actively drives the
    prediction wrong. That is the specific pathology a responder with high d
    and low accuracy would show, and it cannot be fixed by rescaling.
    """
    cols = [c for c in T.FEATURE_SETS[feature_set] if c in raw.columns
            and c != T.TIME_COL]
    lab = raw["label"].to_numpy()
    subj = raw["subject"].to_numpy()

    per_subj = {}
    for sid in pd.unique(subj):
        m = subj == sid
        s_mask = m & (lab == T.STRESS_LABEL)
        n_mask = m & (lab != T.STRESS_LABEL)
        per_subj[sid] = {
            c: cohens_d(raw.loc[s_mask, c].to_numpy(dtype=np.float64),
                        raw.loc[n_mask, c].to_numpy(dtype=np.float64))
            for c in cols
        }

    dmat = pd.DataFrame(per_subj).T  # subjects x features
    out_rows = []

    print("\n" + "=" * 72)
    print("RESPONSE DIRECTION  (Cohen's d, stress vs non-stress, raw features)")
    print("=" * 72)

    for sid in subjects:
        if sid not in dmat.index:
            continue
        others = dmat.drop(index=sid)
        cohort_d = others.median(axis=0, skipna=True)
        subj_d = dmat.loc[sid]

        tab = pd.DataFrame({
            "feature": cols,
            "d_subject": subj_d[cols].to_numpy(dtype=float),
            "d_cohort": cohort_d[cols].to_numpy(dtype=float),
        })
        tab["sign_agrees"] = np.sign(tab["d_subject"]) == np.sign(tab["d_cohort"])
        tab["abs_d_subject"] = tab["d_subject"].abs()
        tab["abs_d_cohort"] = tab["d_cohort"].abs()
        tab = tab.sort_values("abs_d_cohort", ascending=False)
        tab.insert(0, "subject", sid)

        mean_abs = float(tab["abs_d_subject"].mean(skipna=True))
        cohort_mean = float(tab["abs_d_cohort"].mean(skipna=True))
        # Only count disagreement where the cohort has a direction worth
        # opposing; a sign flip on a feature nobody responds to is noise.
        meaningful = tab[tab["abs_d_cohort"] > 0.3]
        n_flip = int((~meaningful["sign_agrees"]).sum())
        n_meaningful = int(len(meaningful))

        print(f"\n{sid}:  mean |d| = {mean_abs:.3f}  "
              f"(cohort median-feature mean |d| = {cohort_mean:.3f})")
        print(f"  sign disagreement on {n_flip}/{n_meaningful} features "
              f"where the cohort effect exceeds |d|=0.3")

        print(f"\n  {'feature':<24}{'d_subj':>9}{'d_cohort':>10}{'agrees':>9}")
        for _, r in tab.head(top).iterrows():
            flag = "" if r["sign_agrees"] else "   <-- INVERTED"
            print(f"  {r['feature']:<24}{r['d_subject']:>9.3f}"
                  f"{r['d_cohort']:>10.3f}{str(bool(r['sign_agrees'])):>9}{flag}")

        if n_meaningful and n_flip / n_meaningful > 0.3:
            print("\n  VERDICT: inverted responder. A substantial share of the")
            print("  features the cohort relies on move the opposite way for this")
            print("  subject. A model trained on the other 14 will be confidently")
            print("  wrong here, and the effect size being large makes it worse,")
            print("  not better. Not recoverable by tuning the reference window.")
        elif mean_abs < 0.3:
            print("\n  VERDICT: non-responder. Little separation to find in any")
            print("  direction, so the fold is near chance for a physiological")
            print("  reason rather than a modelling one.")
        else:
            print("\n  VERDICT: directions broadly agree and separation exists.")
            print("  The failure is not response direction — look at magnitude")
            print("  scaling or at whether the subject's operating point sits")
            print("  outside the training range.")

        out_rows.append(tab)

    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()


def stability_across_n(raw: pd.DataFrame, subjects: list, task: str,
                       model_kind: str, feature_set: str, seed: int,
                       n_list: list) -> pd.DataFrame:
    """Balanced accuracy for the target subjects only, across buffer sizes.

    Cheap because only the target folds are scored, but every fold is still
    trained, so the numbers match the full sweep exactly.
    """
    rows = []
    print("\n" + "=" * 72)
    print(f"ACROSS BUFFER SIZES  ({feature_set})")
    print("=" * 72)
    cols = T.resolve_features(feature_set, raw)

    for n in n_list:
        std = T.standardise(raw, "baseline", n_ref=n)
        for fold in T._loso_folds(std, cols, task, model_kind, seed,
                                  do_permutation=False, verbose=False):
            if fold["subject"] not in subjects:
                continue
            y_t, y_p = fold["y_true"], fold["y_pred"]
            rows.append({
                "subject": fold["subject"], "n_ref": n,
                "balanced_acc": float(T.balanced_accuracy_score(y_t, y_p)),
                "n_pred_classes": int(len(np.unique(y_p))),
            })

    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab
    piv = tab.pivot(index="subject", columns="n_ref", values="balanced_acc")
    cls = tab.pivot(index="subject", columns="n_ref", values="n_pred_classes")
    print("\nbalanced accuracy")
    print(piv.round(3).to_string())
    print("\ndistinct predicted classes (1 = degenerate)")
    print(cls.to_string())
    return tab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--features", default="eda_only")
    ap.add_argument("--task", default="binary", choices=["binary", "3class"])
    ap.add_argument("--model", default="rf", choices=["rf", "hgb"])
    ap.add_argument("--n-ref", nargs="+", type=int, default=[30, 100])
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    raw = T.load_table(Path(args.cache))
    missing = [s for s in args.subjects if s not in set(raw["subject"])]
    if missing:
        print(f"subjects not in cache: {missing}", file=sys.stderr)
        return 2

    print(f"\ntargets: {', '.join(args.subjects)}   "
          f"features: {args.features}   task: {args.task}   model: {args.model}")

    class_balance(raw, args.subjects, args.task)

    # Raw-feature checks first: they do not depend on a buffer size, so if the
    # answer is here it is the same answer at every N.
    direction = response_direction(raw, args.subjects, args.features)

    scale_frames = []
    for n in args.n_ref:
        scale_frames.append(scaling_health(raw, args.subjects, n, args.features))

    primary_n = args.n_ref[-1]
    std = T.standardise(raw, "baseline", n_ref=primary_n)
    print(f"\n(prediction profile below uses n_ref={primary_n})")
    profile = prediction_profile(std, args.subjects, args.task,
                                 args.model, args.features, args.seed)

    stability = stability_across_n(raw, args.subjects, args.task, args.model,
                                   args.features, args.seed, args.n_ref)

    print("\n" + "=" * 72)
    print("HOW TO READ THIS")
    print("=" * 72)
    print("""
  Degenerate prediction + p99|z| within cohort range + directions agree
      The subject's features are scaled correctly and point the right way,
      but its operating point sits outside the range the other 14 span.
      A per-subject baseline offset would be the lever, not the buffer size.

  Degenerate prediction + p99|z| outside cohort range
      Scaling collapse. Check the degenerate-divisor column for the feature
      that caused it. This is the S6 failure mode recurring.

  Both classes predicted + high |d| + sign disagreement
      Inverted responder. The model has learned the cohort's direction and
      this subject moves the other way. Not fixable in this pipeline; it is
      a limitation to report, and an argument for per-subject calibration
      as future work.

  Both classes predicted + low |d| across the board
      Non-responder under this protocol. A physiological result, not a bug.
      WESAD's stress condition does not elicit a measurable EDA response in
      every participant, and reporting that honestly is better than tuning
      until it disappears.
""")

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        tag = f"verify_{'_'.join(args.subjects)}_{args.features}"
        if not direction.empty:
            direction.to_csv(RESULTS_DIR / f"{tag}_direction.csv", index=False)
        if scale_frames:
            pd.concat(scale_frames, ignore_index=True).to_csv(
                RESULTS_DIR / f"{tag}_scaling.csv", index=False)
        if not profile.empty:
            profile.to_csv(RESULTS_DIR / f"{tag}_profile.csv", index=False)
        if not stability.empty:
            stability.to_csv(RESULTS_DIR / f"{tag}_stability.csv", index=False)
        print(f"saved -> {RESULTS_DIR}/{tag}_*.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())