"""
verify_wesad.py — prove the dataset loads and the time alignment is correct
BEFORE any feature code is written.

Run:
    python verify_wesad.py --root "C:\\dev\\wesad\\raw\\WESAD"
    python verify_wesad.py --root ... --plot S2

Checks, per subject:
  1. pickle loads
  2. every wrist stream has the expected shape/dtype
  3. implied durations from all five tracks agree within tolerance
  4. label distribution over kept classes is near the published 53/30/17
  5. no NaNs, and TEMP/EDA sit in physically plausible ranges
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from wesad_loader import (
    KEEP_LABELS,
    LABEL_FS,
    LABEL_NAMES,
    SUBJECTS,
    WRIST_FS,
    label_distribution,
    load_subject,
)

# Published WESAD proportions for baseline / stress / amusement.
EXPECTED = {"baseline": 0.53, "stress": 0.30, "amusement": 0.17}

# Streams whose implied duration should agree with the label track.
DURATION_TOL_S = 5.0


def check(sub, sid: str) -> list[str]:
    problems = []

    # --- shapes ---------------------------------------------------------
    if sub.acc.ndim != 2 or sub.acc.shape[1] != 3:
        problems.append(f"ACC shape {sub.acc.shape}, expected (N, 3)")
    for name in ("bvp", "eda", "temp"):
        if sub.stream(name).ndim != 1:
            problems.append(f"{name.upper()} is not 1-D")

    # --- durations agree ------------------------------------------------
    durs = {k: len(sub.stream(k)) / WRIST_FS[k] for k in WRIST_FS}
    durs["label"] = len(sub.label) / LABEL_FS
    spread = max(durs.values()) - min(durs.values())
    if spread > DURATION_TOL_S:
        detail = ", ".join(f"{k}={v:.1f}s" for k, v in durs.items())
        problems.append(f"duration spread {spread:.1f}s > {DURATION_TOL_S}s ({detail})")

    # --- NaNs -----------------------------------------------------------
    for name in ("bvp", "eda", "temp"):
        if np.isnan(sub.stream(name)).any():
            problems.append(f"{name.upper()} contains NaN")
    if np.isnan(sub.acc).any():
        problems.append("ACC contains NaN")

    # --- plausible ranges ----------------------------------------------
    if sub.eda.min() < -0.5:
        problems.append(f"EDA min {sub.eda.min():.2f} uS is negative")
    if not (10.0 <= np.median(sub.temp) <= 45.0):
        problems.append(f"TEMP median {np.median(sub.temp):.1f} C implausible")

    # --- labels ---------------------------------------------------------
    present = set(np.unique(sub.label).tolist())
    missing = [LABEL_NAMES[c] for c in KEEP_LABELS if c not in present]
    if missing:
        problems.append(f"missing class(es): {', '.join(missing)}")

    dist = label_distribution(sub)
    for cls, exp in EXPECTED.items():
        got = dist.get(cls, 0.0)
        if abs(got - exp) > 0.12:
            problems.append(f"{cls} share {got:.2f} vs expected ~{exp:.2f}")

    # --- alignment spot check ------------------------------------------
    # EDA sample times must map back to the same labels as direct indexing.
    t = sub.t("eda")
    direct = sub.label[np.clip((t * LABEL_FS).astype(int), 0, len(sub.label) - 1)]
    if not np.array_equal(direct, sub.label_for("eda")):
        problems.append("label_for('eda') disagrees with direct index")

    return problems


def summarise(sub, sid: str) -> str:
    dist = label_distribution(sub)
    parts = " ".join(f"{k[:4]}={v:.2f}" for k, v in dist.items())
    return (
        f"{sid:>4}  {sub.duration()/60:6.1f} min  "
        f"eda={len(sub.eda):>7}  bvp={len(sub.bvp):>8}  "
        f"acc={sub.acc.shape[0]:>7}  {parts}"
    )


def plot_subject(sub, sid: str) -> None:
    import matplotlib.pyplot as plt

    t = sub.t("eda") / 60.0
    lab = sub.label_for("eda")
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(t, sub.eda, lw=0.6)
    ax[0].set_ylabel("EDA (uS)")
    ax[1].step(t, lab, where="post", lw=0.8)
    ax[1].set_yticks(sorted(LABEL_NAMES))
    ax[1].set_yticklabels([LABEL_NAMES[c] for c in sorted(LABEL_NAMES)])
    ax[1].set_xlabel("minutes")
    fig.suptitle(f"{sid}: EDA vs protocol label (visual alignment check)")
    fig.tight_layout()
    plt.show()


def main() -> int:
    ap = argparse.ArgumentParser()
    from config import WESAD_ROOT
    ap.add_argument("--root", default=WESAD_ROOT, help="path to the WESAD folder")
    ap.add_argument("--plot", default=None, help="subject id to plot, e.g. S2")
    args = ap.parse_args()

    print(f"{'sid':>4}  {'dur':>10}  streams / label shares")
    failed = {}
    for sid in SUBJECTS:
        try:
            sub = load_subject(sid, args.root)
        except Exception as e:  # noqa: BLE001
            failed[sid] = [f"load failed: {e}"]
            print(f"{sid:>4}  LOAD FAILED: {e}")
            continue
        print(summarise(sub, sid))
        problems = check(sub, sid)
        if problems:
            failed[sid] = problems
        if args.plot and sid == args.plot:
            plot_subject(sub, sid)

    print()
    if failed:
        print("PROBLEMS")
        for sid, ps in failed.items():
            for p in ps:
                print(f"  {sid}: {p}")
        return 1

    print(f"All {len(SUBJECTS)} subjects loaded and passed alignment checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())