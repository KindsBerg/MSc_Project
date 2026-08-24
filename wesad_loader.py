"""
wesad_loader.py — parse WESAD subject pickles into aligned wrist streams.

WESAD pickles were written under Python 2, so they need encoding='latin1'.
Wrist channels are separate arrays at different sample rates with no
timestamps: all streams start at t=0, so sample n of a stream at rate fs sits
at t = n / fs seconds.

Chest data (RespiBAN, 700 Hz) is ignored — there is no chest sensor in this
project's hardware.

    from wesad_loader import load_subject, SUBJECTS, WRIST_FS
    s = load_subject("S2", root=r"C:\\dev\\wesad\\raw\\WESAD")
    s.eda                       # (N,) float array @ 4 Hz
    s.window_label(0.0, 60.0)   # one label for [0, 60) s, or None
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# S1 and S12 were discarded by the WESAD authors for sensor faults.
SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]

# Empatica E4 wrist sample rates (Hz).
WRIST_FS = {"acc": 32.0, "bvp": 64.0, "eda": 4.0, "temp": 4.0}
LABEL_FS = 700.0

# Protocol label codes. 0 = undefined, 5/6/7 = transient. Meditation (4) is
# loaded but dropped from KEEP_LABELS by scope.
LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
KEEP_LABELS = (1, 2, 3)

# WESAD readme §II.2: E4 ACC.csv is raw counts in units of 1/64 g, not g.
# Loading it unconverted leaves every g-scaled threshold downstream (the IMU
# motion gate in features.py) firing against numbers ~64x too large, so the
# gate never trips. Convert once, here, so nothing downstream sees raw counts.
ACC_LSB_PER_G = 64.0

# Gravity alone should put median |ACC| near 1 g across a whole recording.
# Outside this band means the unit conversion is wrong, not that the subject
# is unusual — so it fails hard rather than warning.
ACC_REST_MAG_MIN_G = 0.7
ACC_REST_MAG_MAX_G = 1.3


@dataclass
class Subject:
    """One WESAD subject's wrist streams plus the raw 700 Hz label track."""

    sid: str
    acc: np.ndarray    # (N, 3) @ 32 Hz, g
    bvp: np.ndarray    # (N,)   @ 64 Hz, blood volume pulse
    eda: np.ndarray    # (N,)   @ 4 Hz, microsiemens
    temp: np.ndarray   # (N,)   @ 4 Hz, E4 skin thermistor — NOT a model input
    label: np.ndarray  # (M,)   @ 700 Hz, int

    def stream(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def duration(self) -> float:
        """Recording duration, bounded by the shortest track."""
        ends = [len(self.stream(k)) / WRIST_FS[k] for k in WRIST_FS]
        ends.append(len(self.label) / LABEL_FS)
        return float(min(ends))

    def window_label(self, t0: float, t1: float, purity: float = 0.9):
        """One label for [t0, t1), or None if impure or out of scope.

        A window spanning a condition change has no valid target, so it is
        rejected unless at least `purity` of its label samples agree and the
        majority class is in KEEP_LABELS.
        """
        i0 = int(np.floor(t0 * LABEL_FS))  # first label sample covered by the window
        i1 = int(np.ceil(t1 * LABEL_FS))   # one past the last covered sample

        # A window PARTIALLY outside the label track is a caller bug (bad
        # timestamps), not an edge case. Clamping it would score the window on
        # whatever fraction happened to exist, so raise instead. Fully
        # out-of-range falls through to the empty-segment return below.
        partial = (i1 > len(self.label) or i0 < 0) and not (
            i0 >= len(self.label) or i1 <= 0
        )
        if partial:
            raise ValueError(
                f"window [{t0}, {t1})s partially outside label track "
                f"(0..{len(self.label) / LABEL_FS:.1f}s)"
            )

        seg = self.label[max(i0, 0) : min(i1, len(self.label))]  # label samples inside the window
        if seg.size == 0:
            return None  # window sits fully outside the label track
        vals, counts = np.unique(seg, return_counts=True)  # tally each label code present
        top = int(vals[np.argmax(counts)])  # majority label
        if counts.max() / seg.size < purity or top not in KEEP_LABELS:
            return None  # impure (straddles a boundary) or out of scope
        return top


def load_subject(sid: str, root: str | Path) -> Subject:
    """Load one subject pickle from <root>/<sid>/<sid>.pkl."""
    path = Path(root) / sid / f"{sid}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"missing pickle: {path}")
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")  # WESAD pickles are Python 2 era
    w = d["signal"]["wrist"]  # chest streams exist in d but are never read

    def flat(a):
        # 1-D float array, matching the live data format.
        return np.asarray(a, dtype=np.float64).reshape(-1)

    acc = np.asarray(w["ACC"], dtype=np.float64).reshape(-1, 3) / ACC_LSB_PER_G  # raw 1/64 g counts -> g

    mag_med = float(np.median(np.linalg.norm(acc, axis=1)))
    if not (ACC_REST_MAG_MIN_G <= mag_med <= ACC_REST_MAG_MAX_G):
        raise RuntimeError(
            f"{sid}: median |ACC| = {mag_med:.3f} g, expected "
            f"{ACC_REST_MAG_MIN_G}-{ACC_REST_MAG_MAX_G} g (gravity). "
            "A unit conversion is wrong upstream."
        )

    return Subject(
        sid=str(d.get("subject", sid)),
        acc=acc,
        bvp=flat(w["BVP"]),
        eda=flat(w["EDA"]),
        temp=flat(w["TEMP"]),
        label=np.asarray(d["label"]).reshape(-1).astype(np.int16),
    )