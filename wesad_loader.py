"""
wesad_loader.py -- loads the WESAD subject pickles and lines up the wrist streams.

these pickles are old python 2 files so need encoding='latin1'. wrist streams
have no timestamps, just different sample rates, so sample n of a stream at
rate fs is just at t = n/fs.

not using the chest sensor (RespiBAN) at all, my hardware doesn't have one.

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

# S1 and S12 got dropped by the WESAD authors, sensor faults
SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]

# Empatica E4 wrist sample rates (Hz)
WRIST_FS = {"acc": 32.0, "bvp": 64.0, "eda": 4.0, "temp": 4.0}
LABEL_FS = 700.0

# protocol label codes. 0 = undefined, 5/6/7 = transient, meditation (4) is
# loaded but not kept -- out of scope
LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
KEEP_LABELS = (1, 2, 3)

# WESAD readme says E4 ACC.csv is raw counts (1/64 g), not g. need to convert
# or every g-based threshold downstream (motion gate in features.py) ends up
# ~64x too big and never trips. converting once here so nothing else has to
# worry about it.
ACC_LSB_PER_G = 64.0

# gravity alone should put median |ACC| near 1g across a whole recording. if
# it's not in this range the unit conversion is broken, not the subject being
# weird -- so blow up loudly instead of just warning.
ACC_REST_MAG_MIN_G = 0.7
ACC_REST_MAG_MAX_G = 1.3


@dataclass
class Subject:
    """one subject's wrist streams + the raw 700hz label track"""

    sid: str
    acc: np.ndarray    # (N, 3) @ 32 Hz, g
    bvp: np.ndarray    # (N,)   @ 64 Hz, blood volume pulse
    eda: np.ndarray    # (N,)   @ 4 Hz, microsiemens
    temp: np.ndarray   # (N,)   @ 4 Hz, E4 skin thermistor -- not used in the model
    label: np.ndarray  # (M,)   @ 700 Hz, int

    def stream(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def duration(self) -> float:
        """recording length, capped by whichever stream is shortest"""
        ends = [len(self.stream(k)) / WRIST_FS[k] for k in WRIST_FS]
        ends.append(len(self.label) / LABEL_FS)
        return float(min(ends))

    def window_label(self, t0: float, t1: float, purity: float = 0.9):
        """one label for [t0, t1), or None if it's too mixed / out of scope.

        if a window straddles a condition change there's no clean target, so
        reject it unless most of the labels agree and the winner is one we
        actually keep.
        """
        i0 = int(np.floor(t0 * LABEL_FS))  # first label sample in the window
        i1 = int(np.ceil(t1 * LABEL_FS))   # one past the last label sample

        # partially outside the label track = caller bug (bad timestamps
        # somewhere), not something to clamp and score anyway. fully outside
        # is fine, handled below.
        partial = (i1 > len(self.label) or i0 < 0) and not (
            i0 >= len(self.label) or i1 <= 0
        )
        if partial:
            raise ValueError(
                f"window [{t0}, {t1})s partially outside label track "
                f"(0..{len(self.label) / LABEL_FS:.1f}s)"
            )

        seg = self.label[max(i0, 0) : min(i1, len(self.label))]  # labels inside the window
        if seg.size == 0:
            return None  # window is entirely outside the label track
        vals, counts = np.unique(seg, return_counts=True)  # tally each label code
        top = int(vals[np.argmax(counts)])  # whichever label wins
        if counts.max() / seg.size < purity or top not in KEEP_LABELS:
            return None  # too mixed, or not a label we care about
        return top


def load_subject(sid: str, root: str | Path) -> Subject:
    """loads one subject's pickle from <root>/<sid>/<sid>.pkl"""
    path = Path(root) / sid / f"{sid}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"missing pickle: {path}")
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")  # old python2 pickle
    w = d["signal"]["wrist"]  # chest data lives in here too but never used

    def flat(a):
        # just flattening to 1-D float, same shape as the live data
        return np.asarray(a, dtype=np.float64).reshape(-1)

    acc = np.asarray(w["ACC"], dtype=np.float64).reshape(-1, 3) / ACC_LSB_PER_G  # counts -> g

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
