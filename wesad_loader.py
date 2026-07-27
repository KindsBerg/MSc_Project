"""
wesad_loader.py — parse WESAD subject pickles into aligned wrist streams.

WESAD pickles were written under Python 2, so they require encoding='latin1'.
Wrist channels are stored as separate arrays at different sampling rates with
no timestamps; all streams are zero-aligned at recording start, so sample n of
a stream at rate fs corresponds to t = n / fs seconds.

Chest data (RespiBAN, 700 Hz) is deliberately ignored — there is no chest
analogue in this project's hardware.

Usage:
    from wesad_loader import load_subject, SUBJECTS, WRIST_FS
    s = load_subject("S2", root=r"C:\\dev\\wesad\\raw\\WESAD")
    s.eda                       # (N,) float array @ 4 Hz
    s.window_label(0.0, 60.0)   # single label for the [0, 60) s window, or None
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# S1 and S12 were discarded by the WESAD authors for sensor faults.
SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]

# Empatica E4 wrist sampling rates (Hz).
WRIST_FS = {"acc": 32.0, "bvp": 64.0, "eda": 4.0, "temp": 4.0}

# The label array is sampled at the chest device rate.
LABEL_FS = 700.0

# Protocol label codes. 0 = undefined, 5/6/7 = transient/ignore.
LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}

# Classes kept for the standard 3-class task (meditation dropped by scope).
KEEP_LABELS = (1, 2, 3)


@dataclass
class Subject:
    """One WESAD subject's wrist streams plus the raw 700 Hz label track."""

    sid: str
    acc: np.ndarray    # (N, 3) @ 32 Hz
    bvp: np.ndarray    # (N,)   @ 64 Hz
    eda: np.ndarray    # (N,)   @ 4 Hz
    temp: np.ndarray   # (N,)   @ 4 Hz
    label: np.ndarray  # (M,)   @ 700 Hz, int

    # ---- streams --------------------------------------------------------

    def stream(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def duration(self) -> float:
        """Recording duration bounded by the shortest available track."""
        ends = [len(self.stream(k)) / WRIST_FS[k] for k in WRIST_FS]
        ends.append(len(self.label) / LABEL_FS)
        return float(min(ends))

    # ---- labels ---------------------------------------------------------

    def window_label(self, t0: float, t1: float, purity: float = 0.9):
        """Single label for the window [t0, t1) or None if impure/out of scope.

        A window spanning a condition change carries no valid target, so it is
        rejected unless at least `purity` of its label samples agree and the
        majority class is one of KEEP_LABELS.
        """
        i0 = int(np.floor(t0 * LABEL_FS))
        i1 = int(np.ceil(t1 * LABEL_FS))
        seg = self.label[max(i0, 0) : min(i1, len(self.label))]
        if seg.size == 0:
            return None
        vals, counts = np.unique(seg, return_counts=True)
        top = int(vals[np.argmax(counts)])
        if counts.max() / seg.size < purity:
            return None
        if top not in KEEP_LABELS:
            return None
        return top


def load_subject(sid: str, root: str | Path) -> Subject:
    """Load one subject pickle from <root>/<sid>/<sid>.pkl."""
    path = Path(root) / sid / f"{sid}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"missing pickle: {path}")
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")  # Python 2 pickle
    w = d["signal"]["wrist"]

    def flat(a):
        return np.asarray(a, dtype=np.float64).reshape(-1)

    return Subject(
        sid=str(d.get("subject", sid)),
        acc=np.asarray(w["ACC"], dtype=np.float64).reshape(-1, 3),
        bvp=flat(w["BVP"]),
        eda=flat(w["EDA"]),
        temp=flat(w["TEMP"]),
        label=np.asarray(d["label"]).reshape(-1).astype(np.int16),
    )