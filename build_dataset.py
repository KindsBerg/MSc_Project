"""
build_dataset.py — WESAD pickles -> windowed feature table.

Walks each subject, slides the 60 s EDA window, computes the shared feature
vector from features.py, attaches the protocol label, and writes a per-subject
cache. Caching per subject means a failure or a parameter change costs one
subject's recomputation, not the whole set, and leave-one-subject-out CV
becomes a cheap re-index rather than a re-parse.

Run:
    python build_dataset.py                                  # all subjects
    python build_dataset.py --subjects S2 --limit 20         # smoke test
    python build_dataset.py --force --combine                # full rebuild
    python build_dataset.py --check                          # deps only, no work

--------------------------------------------------------------------------
Three design decisions worth defending in the report
--------------------------------------------------------------------------
1. BASELINE REFERENCE WITHOUT LABEL LEAKAGE.
   hr_baseline_delta and temp_baseline_delta need a personal resting
   reference. The obvious way to get one is to average over every window
   labelled 'baseline' — but that uses the target to build a predictor,
   which is label leakage and would inflate cross-validated accuracy.
   Instead the reference is taken from the first BASELINE_REF_S seconds of
   the recording, chosen without reference to any label. In WESAD this
   period falls inside the ~20 min baseline block by protocol design, and it
   mirrors what the live system will do: take a resting reference from the
   first few minutes of wear.

2. DISCARDED WINDOWS ARE COUNTED BY REASON, NOT LUMPED TOGETHER.
   A window is rejected either because its labels disagree (it straddles a
   protocol boundary and has no single valid target) or because its label is
   out of scope (meditation, undefined, or transient). These are completely
   different things: the first is unavoidable data loss, the second is
   deliberate exclusion by the project's stated 3-class scope. Reporting a
   single combined figure makes ~60% of windows look discarded when most of
   it is meditation and inter-condition rest being correctly excluded.

3. DEPENDENCY STATE IS CHECKED AND RECORDED, NOT ASSUMED.
   cvxEDA needs `cvxopt` underneath NeuroKit2. If it is missing, the feature
   module falls back to a median-filter decomposition and still produces
   plausible numbers — so the run looks fine while the method described in
   the report is not the method that ran. Preflight catches this before the
   build, and the fallback rate is printed after it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
from wesad_loader import (
    LABEL_FS,
    LABEL_NAMES,
    SUBJECTS,
    WRIST_FS,
    load_subject,
)

try:
    from config import WESAD_ROOT
except Exception:  # noqa: BLE001 — config.py is optional
    WESAD_ROOT = None

# Seconds from the start of the recording used as the personal resting
# reference. See design note 1 above.
BASELINE_REF_S = 600.0

CACHE_DIR = Path("cache")

META_COLS = ["subject", "label", "label_name", "t_start"]


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------


def preflight() -> bool:
    """Report the state of every dependency that silently degrades output."""
    ok = True
    print("PREFLIGHT")

    try:
        import neurokit2 as nk

        print(f"  neurokit2   {nk.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"  neurokit2   MISSING ({e})")
        print("              -> EDA decomposition and BVP->HR both unavailable")
        return False

    try:
        import cvxopt

        print(f"  cvxopt      {cvxopt.__version__}")
    except Exception:  # noqa: BLE001
        print("  cvxopt      MISSING")
        print("              -> cvxEDA cannot run; EDA features will come from")
        print("                 the median-filter fallback, NOT the method your")
        print("                 report cites. Fix: pip install cvxopt")
        ok = False

    # Live-fire test of the decomposition on synthetic EDA, so a silent
    # failure surfaces here rather than 6000 windows later.
    fs = F.EDA_FS
    t = np.arange(0, 60, 1 / fs)
    synth = 2.0 + 0.002 * t + 0.3 * np.exp(-((t - 20) ** 2) / 4)
    F.reset_stats()
    F.eda_features(synth, fs)
    if F.STATS["cvxeda_ok"]:
        print("  cvxEDA      OK on synthetic window")
    else:
        print(f"  cvxEDA      FALLBACK — {F.STATS['cvxeda_error']}")
        ok = False
    F.reset_stats()

    try:
        import pyarrow  # noqa: F401

        print("  pyarrow     present (parquet cache)")
    except Exception:  # noqa: BLE001
        print("  pyarrow     missing — falling back to gzipped CSV")

    return ok


# --------------------------------------------------------------------------
# Per-subject build
# --------------------------------------------------------------------------


def _slice(x: np.ndarray, fs: float, t0: float, t1: float) -> np.ndarray:
    """Samples of a stream falling in [t0, t1)."""
    i0 = int(np.floor(t0 * fs))
    i1 = int(np.ceil(t1 * fs))
    return x[max(i0, 0) : min(i1, len(x))]


def _reject_reason(sub, t0: float, t1: float) -> str:
    """Why window_label() returned None: 'impure' or 'out_of_scope'."""
    i0 = int(np.floor(t0 * LABEL_FS))
    i1 = int(np.ceil(t1 * LABEL_FS))
    seg = sub.label[max(i0, 0) : min(i1, len(sub.label))]
    if seg.size == 0:
        return "empty"
    _, counts = np.unique(seg, return_counts=True)
    if counts.max() / seg.size < F.WINDOW_PURITY:
        return "impure"
    return "out_of_scope"


def _baseline_refs(sub, hr: np.ndarray, hr_fs: float):
    """Resting HR and TEMP from the opening BASELINE_REF_S of the recording."""
    ref_hr = _slice(hr, hr_fs, 0.0, BASELINE_REF_S)
    ref_hr = ref_hr[np.isfinite(ref_hr) & (ref_hr > 20.0) & (ref_hr < 220.0)]
    ref_temp = _slice(sub.temp, WRIST_FS["temp"], 0.0, BASELINE_REF_S)
    ref_temp = ref_temp[np.isfinite(ref_temp)]
    return (
        float(np.mean(ref_hr)) if ref_hr.size else None,
        float(np.mean(ref_temp)) if ref_temp.size else None,
    )


def build_subject(sid: str, root: str, limit: int | None = None) -> pd.DataFrame:
    """Feature rows for one subject. One row per accepted 60 s window."""
    t_start = time.time()
    sub = load_subject(sid, root)

    # BVP -> HR once for the whole recording. This is the expensive step and
    # the one that emulates the SEN0344's computed-HR output; doing it per
    # window would be both slow and inconsistent at the window edges.
    hr_series = F.bvp_to_hr(sub.bvp, fs=WRIST_FS["bvp"], out_fs=1.0)
    hr, hr_fs = hr_series.values, hr_series.fs

    base_hr, base_temp = _baseline_refs(sub, hr, hr_fs)
    if base_hr is None or base_temp is None:
        print(f"  {sid}: WARNING — no baseline reference; delta features will be NaN")

    duration = sub.duration()
    rows: list[dict] = []
    drops = {"impure": 0, "out_of_scope": 0, "empty": 0, "no_hr": 0}

    t0 = 0.0
    while t0 + F.WIN_EDA_S <= duration:
        t1 = t0 + F.WIN_EDA_S

        label = sub.window_label(t0, t1, purity=F.WINDOW_PURITY)
        if label is None:
            drops[_reject_reason(sub, t0, t1)] += 1
            t0 += F.WIN_STEP_S
            continue

        eda_win = _slice(sub.eda, WRIST_FS["eda"], t0, t1)
        temp_win = _slice(sub.temp, WRIST_FS["temp"], t0, t1)

        # Short sub-windows tiling the 60 s window, non-overlapping.
        hr_blocks: list[dict] = []
        imu_blocks: list[dict] = []
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
            eda_win=eda_win,
            temp_win=temp_win,
            hr_short_blocks=hr_blocks,
            imu_short_blocks=imu_blocks,
            baseline_temp=base_temp,
            eda_fs=WRIST_FS["eda"],
            temp_fs=WRIST_FS["temp"],
        )
        row["subject"] = sid
        row["label"] = int(label)
        row["label_name"] = LABEL_NAMES[label]
        row["t_start"] = round(t0, 3)
        rows.append(row)

        if limit and len(rows) >= limit:
            break
        t0 += F.WIN_STEP_S

    df = pd.DataFrame(rows, columns=F.FEATURE_NAMES + META_COLS)
    elapsed = time.time() - t_start
    counts = df["label_name"].value_counts().to_dict() if len(df) else {}
    dist = " ".join(f"{k[:4]}={v}" for k, v in sorted(counts.items()))
    print(
        f"  {sid}: {len(df):>5} kept ({dist})  "
        f"| scope={drops['out_of_scope']} boundary={drops['impure']} "
        f"no_hr={drops['no_hr']}  {elapsed:.0f}s"
    )
    return df


# --------------------------------------------------------------------------
# Cache IO
# --------------------------------------------------------------------------


def _cache_path(sid: str) -> Path:
    return CACHE_DIR / f"{sid}_features.parquet"


def _write(df: pd.DataFrame, path: Path) -> Path:
    """Parquet if pyarrow is available, gzipped CSV otherwise."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:  # noqa: BLE001 — pyarrow/fastparquet absent
        alt = path.with_suffix(".csv.gz")
        df.to_csv(alt, index=False, compression="gzip")
        return alt


def _existing(sid: str) -> Path | None:
    for p in (_cache_path(sid), _cache_path(sid).with_suffix(".csv.gz")):
        if p.is_file():
            return p
    return None


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# Quality report
# --------------------------------------------------------------------------


def quality_report(df: pd.DataFrame) -> int:
    """Flag columns that would silently damage training. Returns problem count.

    An all-NaN column gets imputed or dropped by most estimators without
    comment; a constant column contributes nothing but still takes a slot in
    the feature-importance ranking you intend to defend in the viva. Both are
    bugs rather than findings, so they are surfaced loudly here.
    """
    print("\nFEATURE QUALITY")
    problems = 0
    for col in F.FEATURE_NAMES:
        s = df[col]
        nan_frac = float(s.isna().mean())
        if nan_frac > 0.5:
            print(f"  [BAD] {col}: {nan_frac:.0%} NaN")
            problems += 1
        elif s.nunique(dropna=True) <= 1:
            val = s.dropna().iloc[0] if s.notna().any() else "all NaN"
            print(f"  [BAD] {col}: constant ({val})")
            problems += 1
        elif nan_frac > 0.01:
            print(f"  [warn] {col}: {nan_frac:.1%} NaN")

    if not problems:
        print("  no all-NaN or constant columns")

    st = F.STATS
    if st["eda_windows"]:
        rate = st["cvxeda_fallback"] / st["eda_windows"]
        tag = "[BAD]" if rate > 0.05 else "     "
        print(
            f"\n{tag} cvxEDA: {st['cvxeda_ok']}/{st['eda_windows']} converged, "
            f"{st['cvxeda_fallback']} fell back ({rate:.1%})"
        )
        if st["cvxeda_error"]:
            print(f"        first reason: {st['cvxeda_error']}")
        if rate > 0.5:
            print("        -> this is systematic, not non-convergence.")
            print("           EDA features did NOT come from cvxEDA.")
            problems += 1
    if st.get("scr_fallback"):
        print(f"      SCR detection fell back {st['scr_fallback']} times")
        if st["scr_error"]:
            print(f"        first reason: {st['scr_error']}")

    return problems


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=WESAD_ROOT,
        required=WESAD_ROOT is None,
        help="folder containing S2, S3, ... (default: config.WESAD_ROOT)",
    )
    ap.add_argument("--subjects", nargs="*", default=None, help="subset, e.g. S2 S3")
    ap.add_argument("--force", action="store_true", help="rebuild existing caches")
    ap.add_argument("--limit", type=int, default=None, help="max windows per subject")
    ap.add_argument("--combine", action="store_true", help="merge caches into one file")
    ap.add_argument("--check", action="store_true", help="preflight only, then exit")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--outdir", default="cache")
    args = ap.parse_args()

    global CACHE_DIR
    CACHE_DIR = Path(args.outdir)

    if not args.skip_preflight:
        deps_ok = preflight()
        if args.check:
            return 0 if deps_ok else 1
        if not deps_ok:
            print(
                "\nDependencies are degraded. Continuing would produce a feature\n"
                "table built by the fallback path. Fix the above, or re-run with\n"
                "--skip-preflight if the fallback is a deliberate choice.\n"
            )
            return 1
        print()

    targets = args.subjects or SUBJECTS
    F.reset_stats()

    print(
        f"window={F.WIN_EDA_S:.0f}s step={F.WIN_STEP_S:.0f}s "
        f"short={F.WIN_SHORT_S:.0f}s purity={F.WINDOW_PURITY} "
        f"features={F.N_FEATURES}"
    )

    built: list[Path] = []
    for sid in targets:
        cached = _existing(sid)
        if cached and not args.force:
            print(f"  {sid}: cached ({cached.name}) — use --force to rebuild")
            built.append(cached)
            continue
        try:
            df = build_subject(sid, args.root, limit=args.limit)
        except Exception as e:  # noqa: BLE001
            print(f"  {sid}: FAILED — {type(e).__name__}: {e}")
            continue
        if df.empty:
            print(f"  {sid}: no usable windows — not cached")
            continue
        built.append(_write(df, _cache_path(sid)))

    if not built:
        print("\nnothing built")
        return 1

    all_df = pd.concat([_read(p) for p in built], ignore_index=True)

    print(
        f"\nTOTAL {len(all_df)} windows across {all_df['subject'].nunique()} subjects"
    )
    print(all_df["label_name"].value_counts(normalize=True).round(3).to_string())

    problems = quality_report(all_df)

    if args.combine:
        out = _write(all_df, CACHE_DIR / "wesad_features.parquet")
        print(f"\ncombined -> {out}")

    if problems:
        print(f"\n{problems} problem(s) above — resolve before training.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())