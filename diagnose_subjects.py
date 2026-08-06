"""
diagnose_subjects.py — why do some folds fail while others are near-perfect?

Three explanations produce similar fold tables and need different responses:

  (A) NON-RESPONDER. The subject's EDA does not modulate with arousal.
      Documented in roughly 10% of people. A FINDING, not a bug, and a real
      limitation of any EDA-led wearable.
  (B) INVERTED RESPONDER. EDA modulates strongly but in the wrong direction —
      skin conductance FALLS under stress. Also a finding, and a stronger one:
      it means direction cannot be assumed across wearers.
  (C) SCALING FAULT. The subject's standardisation reference is degenerate,
      their features land on a scale the training folds never saw, and the
      fold is lost to arithmetic rather than physiology. A BUG.

This script separates them.

    python diagnose_subjects.py
    python diagnose_subjects.py --subjects S14 S3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
from train_model import FEATURE_SETS, N_REF_WINDOWS, TIME_COL, Z_CLIP, load_table, standardise

# Tonic and phasic are separated deliberately: a subject can be flat on one
# and strong on the other, and that distinction matters physiologically.
TONIC = ["eda_scl_mean", "eda_scl_sd"]
PHASIC = ["eda_scr_count", "eda_scr_amp_sum", "eda_range"]
KEY_EDA = TONIC[:1] + PHASIC


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference. |d| < 0.2 is negligible separation."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    pooled = np.sqrt(
        ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
        / (a.size + b.size - 2)
    )
    return 0.0 if pooled == 0 else float((a.mean() - b.mean()) / pooled)


def responder_check(df: pd.DataFrame) -> pd.DataFrame:
    """Per-subject effect size of stress vs non-stress on the EDA features.

    Computed on RAW features — standardisation rescales but does not change
    separation, and raw is what the report should quote.

    A responder shows positive d on eda_scl_mean and eda_scr_count: level
    rises and responses become more frequent under sympathetic activation.
    Sign is kept, not absolute, because a negative d is a different finding
    from a zero one.
    """
    rows = []
    for sid, block in df.groupby("subject"):
        stress = block[block["label"] == 2]
        rest = block[block["label"] != 2]
        r = {"subject": sid, "n_stress": len(stress), "n_rest": len(rest)}
        for c in KEY_EDA + ["eda_scl_sd"]:
            r[c] = cohens_d(stress[c].to_numpy(), rest[c].to_numpy())
        # Best phasic separation available, for the tonic/phasic split.
        r["phasic_best"] = max(
            (r[c] for c in PHASIC if np.isfinite(r[c])), key=abs, default=np.nan
        )
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("eda_scl_mean").reset_index(drop=True)

    print("=" * 76)
    print("EDA RESPONDER CHECK — Cohen's d, stress vs non-stress, raw features")
    print("=" * 76)
    print(f"{'subj':<6}{'SCL':>9}{'#SCR':>9}{'ampSum':>9}{'range':>9}   verdict")
    for _, r in out.iterrows():
        d = r["eda_scl_mean"]
        ph = r["phasic_best"]
        if abs(d) < 0.2 and abs(ph) < 0.5:
            v = "NON-RESPONDER"
        elif d < -0.5:
            v = "INVERTED"
        elif abs(d) < 0.2:
            v = "tonic-flat, phasic-strong"
        elif abs(d) < 0.5:
            v = "weak"
        else:
            v = "responder"
        print(
            f"{r['subject']:<6}{d:>9.2f}{r['eda_scr_count']:>9.2f}"
            f"{r['eda_scr_amp_sum']:>9.2f}{r['eda_range']:>9.2f}   {v}"
        )
    print("\n|d| < 0.2 = negligible. Negative = conductance falls under stress.")
    return out


def reference_check(df: pd.DataFrame) -> pd.DataFrame:
    """Health of each subject's standardisation reference.

    Mirrors train_model.standardise: the reference is the first
    N_REF_WINDOWS ACCEPTED windows in recording order, and the divisor is
    the IQR, not the SD. A zero IQR falls through the cascade to the
    subject's whole-recording IQR rather than exploding, so a nonzero count
    here is informative rather than fatal — but a large one means most of
    that subject's scaling came from the fallback, not the reference.
    """
    cols = [c for c in F.FEATURE_NAMES if c in df.columns]
    rows = []
    for sid, block in df.groupby("subject"):
        ref = block.sort_values(TIME_COL).head(N_REF_WINDOWS)[cols]
        iqr = ref.quantile(0.75) - ref.quantile(0.25)
        rows.append(
            {
                "subject": sid,
                "n_ref": len(ref),
                "first_t": float(block[TIME_COL].min()),
                "n_zero_iqr": int((iqr == 0).sum()),
                "min_iqr": float(iqr[iqr > 0].min()) if (iqr > 0).any() else np.nan,
                "worst_feature": iqr.idxmin(),
            }
        )
    out = pd.DataFrame(rows).sort_values("n_zero_iqr", ascending=False).reset_index(
        drop=True
    )

    print("\n" + "=" * 76)
    print("STANDARDISATION REFERENCE HEALTH")
    print("=" * 76)
    print(f"{'subj':<6}{'n_ref':>7}{'first_t':>9}{'zeroIQR':>9}{'min_iqr':>11}   "
          "smallest-IQR feature")
    for _, r in out.iterrows():
        flag = "  <-- thin" if r["n_ref"] < N_REF_WINDOWS else ""
        print(
            f"{r['subject']:<6}{r['n_ref']:>7}{r['first_t']:>9.0f}"
            f"{r['n_zero_iqr']:>9}{r['min_iqr']:>11.2e}   {r['worst_feature']}{flag}"
        )
    return out


def zero_iqr_exposure(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Which columns of `feature_set` have zero IQR over each subject's first
    N_REF_WINDOWS accepted windows — the compute_scale() condition that used
    to route training and live inference to different fallback tiers before
    the wider-block tier was removed (export_model.py). One row per subject.
    """
    cols = [c for c in FEATURE_SETS[feature_set] if c in df.columns]
    rows = []
    for sid, block in df.groupby("subject"):
        ref = block.sort_values(TIME_COL).head(N_REF_WINDOWS)[cols]
        iqr = ref.quantile(0.75) - ref.quantile(0.25)
        zero_cols = iqr[iqr == 0].index.tolist()
        rows.append(
            {
                "subject": sid,
                "feature_set": feature_set,
                "n_ref": len(ref),
                "n_zero_iqr_cols": len(zero_cols),
                "zero_iqr_cols": ";".join(zero_cols),
            }
        )
    out = pd.DataFrame(rows).sort_values("n_zero_iqr_cols", ascending=False).reset_index(
        drop=True
    )

    col_counts = pd.Series(0, index=cols, dtype=int)
    for cs in out["zero_iqr_cols"]:
        for c in filter(None, cs.split(";")):
            col_counts[c] += 1
    col_counts = col_counts[col_counts > 0].sort_values(ascending=False)

    print("\n" + "=" * 76)
    print(f"ZERO-IQR EXPOSURE — feature_set={feature_set!r}, n_ref={N_REF_WINDOWS}")
    print("=" * 76)
    print(f"{'subj':<6}{'n_zero_iqr_cols':>16}   columns")
    for _, r in out.iterrows():
        print(f"{r['subject']:<6}{r['n_zero_iqr_cols']:>16}   {r['zero_iqr_cols']}")
    print(f"\nper-column count across {df['subject'].nunique()} subjects:")
    if col_counts.empty:
        print("  none — every column has nonzero reference IQR in every subject")
    else:
        for c, n in col_counts.items():
            print(f"  {c:<24} {n}")
    return out


def magnitude_check(df: pd.DataFrame) -> pd.DataFrame:
    """Post-standardisation magnitudes per subject.

    Reports the 99th percentile and the clipped fraction, NOT the max: the
    max saturates at Z_CLIP for nearly every subject, so an outlier test on
    it can never fire. A subject reaching magnitudes the other fourteen never
    produce has been placed somewhere the model has no training data.
    """
    std_df = standardise(df, "baseline")
    cols = [c for c in F.FEATURE_NAMES if c in df.columns]
    rows = []
    for sid, block in std_df.groupby("subject"):
        v = np.abs(block[cols].to_numpy(dtype=np.float64))
        v = v[np.isfinite(v)]
        rows.append(
            {
                "subject": sid,
                "p50": float(np.percentile(v, 50)),
                "p99": float(np.percentile(v, 99)),
                "frac_clipped": float((v >= Z_CLIP).mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values("p99", ascending=False).reset_index(drop=True)

    med = out["p99"].median()
    print("\n" + "=" * 76)
    print("POST-STANDARDISATION MAGNITUDE (baseline mode)")
    print("=" * 76)
    print(f"{'subj':<6}{'p50|z|':>9}{'p99|z|':>9}{'clipped':>10}")
    for _, r in out.iterrows():
        flag = ""
        if r["p99"] > 3 * med:
            flag = "  <-- inflated"
        elif r["p99"] < med / 3:
            flag = "  <-- compressed"
        print(
            f"{r['subject']:<6}{r['p50']:>9.2f}{r['p99']:>9.2f}"
            f"{r['frac_clipped']:>10.4f}{flag}"
        )
    print(f"\ncohort median p99|z| = {med:.2f}; flagged outside 3x either way.")
    return out


def verdict(resp: pd.DataFrame, mag: pd.DataFrame, suspects: list) -> None:
    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)
    med = mag["p99"].median()
    for sid in suspects:
        r = resp[resp["subject"] == sid]
        m = mag[mag["subject"] == sid]
        if r.empty or m.empty:
            print(f"\n{sid}: not in the table")
            continue
        d = float(r["eda_scl_mean"].iloc[0])
        ph = float(r["phasic_best"].iloc[0])
        p99 = float(m["p99"].iloc[0])

        print(f"\n{sid}: d(SCL) = {d:+.2f}, best phasic d = {ph:+.2f}, "
              f"p99|z| = {p99:.1f}")

        if p99 > 3 * med or p99 < med / 3:
            print("  -> SCALING FAULT. This subject sits on a scale the other")
            print("     folds never produce, so the failure is arithmetic, not")
            print("     physiology. Investigate the reference window before")
            print("     drawing any conclusion about this subject's response.")
        elif d < -0.5:
            print("  -> INVERTED RESPONDER. Skin conductance FALLS under stress.")
            print("     Report it: response direction cannot be assumed across")
            print("     wearers, which is why a per-wearer reference matters and")
            print("     why a fixed threshold would fail on this person.")
        elif abs(d) < 0.2 and abs(ph) < 0.5:
            print("  -> NON-RESPONDER. Neither tonic nor phasic separates.")
            print("     Report it: EDA-led inference cannot work on subjects whose")
            print("     electrodermal activity does not modulate with arousal.")
            print("     A real deployment limitation of the device.")
        elif abs(d) < 0.2:
            print("  -> TONIC-FLAT, PHASIC-RESPONSIVE. Level does not shift but")
            print("     discrete responses do. Argues for keeping the phasic")
            print("     features rather than relying on mean SCL alone.")
        else:
            print("  -> EDA separates and scaling is sound, yet the fold fails.")
            print("     Look at whether HR disagrees with EDA for this subject,")
            print("     and compare the fold under eda_only against clean.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--subjects", nargs="*", default=["S14", "S3", "S2", "S17", "S11"])
    ap.add_argument("--out", default="results/subject_diagnostics.csv")
    ap.add_argument(
        "--zero-iqr", nargs="*", default=None, metavar="FEATURE_SET",
        help="run zero_iqr_exposure() for these feature sets (e.g. clean eda_only) "
             "instead of the responder/reference/magnitude report",
    )
    ap.add_argument("--zero-iqr-out", default="results/zero_iqr_exposure.csv")
    args = ap.parse_args()

    df = load_table(Path(args.cache))

    if args.zero_iqr is not None:
        merged = pd.concat(
            [zero_iqr_exposure(df, fs) for fs in args.zero_iqr], ignore_index=True
        )
        out = Path(args.zero_iqr_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out, index=False)
        print(f"\nsaved -> {out}")
        return 0

    resp = responder_check(df)
    ref = reference_check(df)
    mag = magnitude_check(df)
    verdict(resp, mag, args.subjects)

    merged = resp.merge(ref, on="subject").merge(mag, on="subject")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())