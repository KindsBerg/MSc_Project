"""
diagnose_subjects.py — why do S6 and S14 fail while others are near-perfect?

Two hypotheses produce identical fold tables and need different responses:

  (A) ELECTRODERMAL NON-RESPONDER. The subject's EDA genuinely does not
      modulate with arousal. A documented phenomenon affecting roughly 10%
      of people; 2 of 15 fits. This is a FINDING to report, not a bug, and
      it is a real limitation of any EDA-led wearable.

  (B) STANDARDISATION BLOWUP. standardise() divides each subject's features
      by the SD of their first BASELINE_REF_S seconds. If that window is
      unusually flat, the divisor is tiny and the subject's features explode
      to magnitudes absent from the training folds. That is a BUG in the
      pipeline and is fixable.

This script separates them.

Run:
    python diagnose_subjects.py
    python diagnose_subjects.py --subjects S6 S14
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
from train_model import BASELINE_REF_S, TIME_COL, load_table, standardise

KEY_EDA = ["eda_scl_mean", "eda_range", "eda_scr_count", "eda_scr_amp_sum"]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference. |d| < 0.2 is negligible separation."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    na, nb = a.size, b.size
    pooled = np.sqrt(
        ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    )
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def responder_check(df: pd.DataFrame) -> pd.DataFrame:
    """Per-subject effect size of stress vs non-stress on each EDA feature.

    A responder shows a positive d on eda_scl_mean and eda_scr_count: skin
    conductance level rises and responses become more frequent under
    sympathetic activation. A non-responder sits near zero on both.
    """
    rows = []
    for sid, block in df.groupby("subject"):
        stress = block[block["label"] == 2]
        rest = block[block["label"] != 2]
        r = {"subject": sid, "n_stress": len(stress), "n_rest": len(rest)}
        for c in KEY_EDA:
            r[c] = cohens_d(stress[c].to_numpy(), rest[c].to_numpy())
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("eda_scl_mean").reset_index(drop=True)

    print("=" * 72)
    print("EDA RESPONDER CHECK — Cohen's d, stress vs non-stress (RAW features)")
    print("=" * 72)
    print(f"{'subj':<6}{'SCL':>9}{'range':>9}{'#SCR':>9}{'ampSum':>9}   verdict")
    for _, r in out.iterrows():
        d = abs(r["eda_scl_mean"])
        if d < 0.2:
            verdict = "NON-RESPONDER"
        elif d < 0.5:
            verdict = "weak"
        else:
            verdict = "responder"
        print(
            f"{r['subject']:<6}{r['eda_scl_mean']:>9.2f}{r['eda_range']:>9.2f}"
            f"{r['eda_scr_count']:>9.2f}{r['eda_scr_amp_sum']:>9.2f}   {verdict}"
        )
    print("\n|d| < 0.2 = negligible separation; the EDA block cannot help there.")
    return out


def baseline_ref_check(df: pd.DataFrame) -> pd.DataFrame:
    """Health of each subject's standardisation reference window.

    Reports how many windows fall in the first BASELINE_REF_S seconds and how
    small the smallest per-feature SD is. A near-zero SD becomes a near-zero
    divisor, which is how hypothesis (B) happens.
    """
    cols = [c for c in F.FEATURE_NAMES if c in df.columns]
    rows = []
    for sid, block in df.groupby("subject"):
        ref = block[block[TIME_COL] < BASELINE_REF_S][cols]
        sd = ref.std()
        rows.append(
            {
                "subject": sid,
                "n_ref": len(ref),
                "min_sd": float(sd.min()) if len(ref) else np.nan,
                "n_zero_sd": int((sd == 0).sum()) if len(ref) else -1,
                "worst_feature": sd.idxmin() if len(ref) else "",
            }
        )
    out = pd.DataFrame(rows).sort_values("min_sd").reset_index(drop=True)

    print("\n" + "=" * 72)
    print("BASELINE REFERENCE WINDOW HEALTH")
    print("=" * 72)
    print(f"{'subj':<6}{'n_ref':>7}{'min_sd':>12}{'zero_sd':>9}   smallest-SD feature")
    for _, r in out.iterrows():
        flag = "  <-- thin" if r["n_ref"] < 20 else ""
        print(
            f"{r['subject']:<6}{r['n_ref']:>7}{r['min_sd']:>12.2e}"
            f"{r['n_zero_sd']:>9}   {r['worst_feature']}{flag}"
        )
    return out


def blowup_check(df: pd.DataFrame) -> pd.DataFrame:
    """Post-standardisation magnitudes per subject.

    If one subject's standardised features reach magnitudes the other 14
    never produce, the model has never seen anything like them in training
    and the fold is lost to scaling rather than to physiology.
    """
    std_df = standardise(df, "baseline")
    cols = [c for c in F.FEATURE_NAMES if c in df.columns]
    rows = []
    for sid, block in std_df.groupby("subject"):
        v = block[cols].to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        rows.append(
            {
                "subject": sid,
                "p99_abs": float(np.percentile(np.abs(v), 99)),
                "max_abs": float(np.max(np.abs(v))),
                "frac_gt_10": float((np.abs(v) > 10).mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values("max_abs", ascending=False).reset_index(
        drop=True
    )

    print("\n" + "=" * 72)
    print("POST-STANDARDISATION MAGNITUDE (baseline mode)")
    print("=" * 72)
    print(f"{'subj':<6}{'p99|z|':>10}{'max|z|':>12}{'frac>10':>10}")
    median_max = out["max_abs"].median()
    for _, r in out.iterrows():
        flag = "  <-- outlier" if r["max_abs"] > 5 * median_max else ""
        print(
            f"{r['subject']:<6}{r['p99_abs']:>10.2f}{r['max_abs']:>12.2f}"
            f"{r['frac_gt_10']:>10.4f}{flag}"
        )
    return out


def verdict(resp: pd.DataFrame, blow: pd.DataFrame, suspects: list) -> None:
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    median_max = blow["max_abs"].median()
    for sid in suspects:
        r = resp[resp["subject"] == sid]
        b = blow[blow["subject"] == sid]
        if r.empty or b.empty:
            continue
        d = abs(float(r["eda_scl_mean"].iloc[0]))
        mx = float(b["max_abs"].iloc[0])
        print(f"\n{sid}: |d| on eda_scl_mean = {d:.2f}, max|z| = {mx:.1f}")
        if mx > 5 * median_max:
            print("  -> STANDARDISATION BLOWUP (hypothesis B). Pipeline bug.")
            print("     Fix: floor the divisor, or use a robust scale (IQR/MAD)")
            print("     instead of SD, in train_model.standardise().")
        elif d < 0.2:
            print("  -> NON-RESPONDER (hypothesis A). Not a bug.")
            print("     Report it: EDA-led inference fails on subjects whose")
            print("     electrodermal activity does not modulate with arousal.")
            print("     This is a real deployment limitation of the device.")
        else:
            print("  -> Neither cleanly. EDA separates but the fold still fails;")
            print("     look at whether HR/TEMP disagree with EDA for this subject.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--subjects", nargs="*", default=["S6", "S14", "S2", "S3"])
    args = ap.parse_args()

    df = load_table(Path(args.cache))
    resp = responder_check(df)
    baseline_ref_check(df)
    blow = blowup_check(df)
    verdict(resp, blow, args.subjects)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())