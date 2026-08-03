"""
build_dataset.py — WESAD pickles -> windowed feature table.

Slides the 60 s EDA window over each subject, computes the shared feature
vector from features.py, attaches the protocol label, caches per subject.

    python build_dataset.py --check             # dependencies only
    python build_dataset.py --force --combine   # full rebuild

Design notes (baseline reference without label leakage, drop accounting,
dependency gating, cache invalidation) are in Project_ML_Notes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
from wesad_loader import LABEL_FS, LABEL_NAMES, SUBJECTS, WRIST_FS, load_subject

try:
    from config import WESAD_ROOT
except Exception:  # noqa: BLE001
    WESAD_ROOT = None

BASELINE_REF_S = 600.0  # resting reference taken from the opening of the recording
CACHE_DIR = Path("cache")
META_COLS = ["subject", "label", "label_name", "t_start"]
EXPECTED_COLS = F.FEATURE_NAMES + META_COLS


def preflight() -> bool:
    """cvxopt missing => cvxEDA silently becomes a median filter. Gate on it."""
    ok = True
    try:
        import neurokit2 as nk

        print(f"  neurokit2   {nk.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"  neurokit2   MISSING ({e})")
        return False
    try:
        import cvxopt

        print(f"  cvxopt      {cvxopt.__version__}")
    except Exception:  # noqa: BLE001
        print("  cvxopt      MISSING -> EDA features would come from the fallback")
        ok = False

    t = np.arange(0, 60, 1 / F.EDA_FS)
    F.reset_stats()
    F.eda_features(2.0 + 0.002 * t + 0.3 * np.exp(-((t - 20) ** 2) / 4), F.EDA_FS)
    if F.STATS["cvxeda_ok"]:
        print("  cvxEDA      OK on synthetic window")
    else:
        print(f"  cvxEDA      FALLBACK — {F.STATS['cvxeda_error']}")
        ok = False
    F.reset_stats()

    F.assert_no_retired(F.FEATURE_NAMES)
    print(f"  contract    {F.N_FEATURES} features")
    return ok


def _slice(x: np.ndarray, fs: float, t0: float, t1: float) -> np.ndarray:
    return x[max(int(np.floor(t0 * fs)), 0) : min(int(np.ceil(t1 * fs)), len(x))]


def _reject_reason(sub, t0: float, t1: float) -> str:
    seg = _slice(sub.label, LABEL_FS, t0, t1)
    if seg.size == 0:
        return "empty"
    _, counts = np.unique(seg, return_counts=True)
    return "impure" if counts.max() / seg.size < F.WINDOW_PURITY else "out_of_scope"


def _baseline_hr(hr: np.ndarray, fs: float) -> float | None:
    ref = _slice(hr, fs, 0.0, BASELINE_REF_S)
    ref = ref[np.isfinite(ref) & (ref > 20.0) & (ref < 220.0)]
    return float(np.mean(ref)) if ref.size else None


def build_subject(sid: str, root: str, limit: int | None = None) -> pd.DataFrame:
    t_start = time.time()
    sub = load_subject(sid, root)

    # BVP -> HR once for the whole recording, not per window: it is the
    # expensive step and windowing it would break at the edges.
    hr_series = F.bvp_to_hr(sub.bvp, fs=WRIST_FS["bvp"], out_fs=1.0)
    hr, hr_fs = hr_series.values, hr_series.fs

    base_hr = _baseline_hr(hr, hr_fs)
    if base_hr is None:
        print(f"  {sid}: WARNING — no resting HR reference; delta features NaN")

    rows: list[dict] = []
    drops = {"impure": 0, "out_of_scope": 0, "empty": 0, "no_hr": 0}
    duration = sub.duration()

    t0 = 0.0
    while t0 + F.WIN_EDA_S <= duration:
        t1 = t0 + F.WIN_EDA_S
        label = sub.window_label(t0, t1, purity=F.WINDOW_PURITY)
        if label is None:
            drops[_reject_reason(sub, t0, t1)] += 1
            t0 += F.WIN_STEP_S
            continue

        hr_blocks, imu_blocks = [], []
        s = t0
        while s + F.WIN_SHORT_S <= t1:
            e = s + F.WIN_SHORT_S
            hr_blocks.append(
                F.hr_features(_slice(hr, hr_fs, s, e), hr_fs, baseline_hr=base_hr)
            )
            imu_blocks.append(
                F.imu_features(_slice(sub.acc, WRIST_FS["acc"], s, e), WRIST_FS["acc"])
            )
            s = e
        if not hr_blocks:
            drops["no_hr"] += 1
            t0 += F.WIN_STEP_S
            continue

        row = F.feature_vector(
            eda_win=_slice(sub.eda, WRIST_FS["eda"], t0, t1),
            hr_short_blocks=hr_blocks,
            imu_short_blocks=imu_blocks,
            eda_fs=WRIST_FS["eda"],
        )
        row.update(
            subject=sid,
            label=int(label),
            label_name=LABEL_NAMES[label],
            t_start=round(t0, 3),
        )
        rows.append(row)
        if limit and len(rows) >= limit:
            break
        t0 += F.WIN_STEP_S

    df = pd.DataFrame(rows, columns=EXPECTED_COLS)
    counts = df["label_name"].value_counts().to_dict() if len(df) else {}
    print(
        f"  {sid}: {len(df):>5} kept "
        f"({' '.join(f'{k[:4]}={v}' for k, v in sorted(counts.items()))})  "
        f"| scope={drops['out_of_scope']} boundary={drops['impure']} "
        f"no_hr={drops['no_hr']}  {time.time() - t_start:.0f}s"
    )
    return df


def _cache_path(sid: str) -> Path:
    return CACHE_DIR / f"{sid}_features.parquet"


def _write(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:  # noqa: BLE001 — no pyarrow
        alt = path.with_suffix(".csv.gz")
        df.to_csv(alt, index=False, compression="gzip")
        return alt


def _existing(sid: str) -> Path | None:
    for p in (_cache_path(sid), _cache_path(sid).with_suffix(".csv.gz")):
        if p.is_file():
            return p
    return None


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    # A cache carries no record of the contract that built it; read back under
    # a different FEATURE_NAMES it would concatenate silently.
    if set(df.columns) != set(EXPECTED_COLS):
        raise RuntimeError(
            f"{path.name} was built by a different features.py — delete the "
            "cache directory and rebuild with --force"
        )
    return df


def quality_report(df: pd.DataFrame) -> int:
    """All-NaN or constant columns get imputed away silently. Surface them."""
    problems = 0
    print("\nFEATURE QUALITY")
    for col in F.FEATURE_NAMES:
        nan_frac = float(df[col].isna().mean())
        if nan_frac > 0.5:
            print(f"  [BAD] {col}: {nan_frac:.0%} NaN")
            problems += 1
        elif df[col].nunique(dropna=True) <= 1:
            print(f"  [BAD] {col}: constant")
            problems += 1
        elif nan_frac > 0.01:
            print(f"  [warn] {col}: {nan_frac:.1%} NaN")
    if not problems:
        print("  no all-NaN or constant columns")

    st = F.STATS
    if st["eda_windows"]:
        rate = st["cvxeda_fallback"] / st["eda_windows"]
        print(
            f"  cvxEDA: {st['cvxeda_ok']}/{st['eda_windows']} converged "
            f"({rate:.1%} fallback)"
        )
        if rate > 0.5:
            print(f"  [BAD] systematic — {st['cvxeda_error']}")
            problems += 1
    if st.get("scr_fallback"):
        print(f"  SCR detection fell back {st['scr_fallback']}x: {st['scr_error']}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=WESAD_ROOT, required=WESAD_ROOT is None)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--outdir", default="cache")
    args = ap.parse_args()

    global CACHE_DIR
    CACHE_DIR = Path(args.outdir)

    if not args.skip_preflight:
        print("PREFLIGHT")
        ok = preflight()
        if args.check:
            return 0 if ok else 1
        if not ok:
            print("\nDependencies degraded — fix, or --skip-preflight if deliberate")
            return 1
        print()

    F.reset_stats()
    print(
        f"window={F.WIN_EDA_S:.0f}s step={F.WIN_STEP_S:.0f}s "
        f"short={F.WIN_SHORT_S:.0f}s features={F.N_FEATURES}"
    )

    built: list[Path] = []
    for sid in args.subjects or SUBJECTS:
        cached = _existing(sid)
        if cached and not args.force:
            print(f"  {sid}: cached")
            built.append(cached)
            continue
        try:
            df = build_subject(sid, args.root, limit=args.limit)
        except Exception as e:  # noqa: BLE001
            print(f"  {sid}: FAILED — {type(e).__name__}: {e}")
            continue
        if not df.empty:
            built.append(_write(df, _cache_path(sid)))

    if not built:
        print("\nnothing built")
        return 1

    try:
        all_df = pd.concat([_read(p) for p in built], ignore_index=True)
    except RuntimeError as e:
        print(f"\nSTALE CACHE — {e}")
        return 1

    print(f"\nTOTAL {len(all_df)} windows across {all_df['subject'].nunique()} subjects")
    print(all_df["label_name"].value_counts(normalize=True).round(3).to_string())
    problems = quality_report(all_df)

    if args.combine:
        print(f"\ncombined -> {_write(all_df, CACHE_DIR / 'wesad_features.parquet')}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())