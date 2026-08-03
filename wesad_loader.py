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
#import statements
from __future__ import annotations #used for use of a class name before its fully defined
import pickle # need to unpack from data files --> back into python objects
from dataclasses import dataclass # used for creating subject class
from pathlib import Path # used for file path manipulation
import numpy as np # used for numerical operations on arrays

#Constants
SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12] # S1 and S12 were discarded by the WESAD authors for sensor faults.
WRIST_FS = {"acc": 32.0, "bvp": 64.0, "eda": 4.0, "temp": 4.0} # Empatica E4 wrist sampling rates (Hz)
LABEL_FS = 700.0 # Label sampling rate (Hz)
LABEL_NAMES = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"} # Protocol label codes. 0 = undefined, 5/6/7 = transient/ignore.
KEEP_LABELS = (1, 2, 3) # Classes kept for the standard 3-class task (meditation dropped by scope).

# WESAD readme §II.2: E4 ACC.csv is raw counts in units of 1/64 g, not g.
# Loading it unconverted means every g-scaled threshold downstream (the IMU
# motion gate in features.py) never fires against real data -- it fires on
# raw counts two orders of magnitude larger than the g values it was tuned
# against. Convert once, here, so nothing downstream ever sees raw counts.
ACC_LSB_PER_G = 64.0

# Sanity bound on median resting |ACC| once expressed in g: gravity alone
# should put a wrist-worn accelerometer within this band on average, even
# across a whole recording with varying posture and movement. A median
# outside it is the signature of a unit error slipping back in, not of a
# genuinely unusual subject, so this is a hard failure rather than a warning.
ACC_REST_MAG_MIN_G = 0.7
ACC_REST_MAG_MAX_G = 1.3


@dataclass # @dataclass - to generate specific methods for Subject class
class Subject: 
    """One WESAD subject's wrist streams plus the raw 700 Hz label track."""
 # this part gives defines the attributes of subject class
    sid: str
    acc: np.ndarray    # (N, 3) @ 32 Hz - accelerometer, g
    bvp: np.ndarray    # (N,)   @ 64 Hz - blood volume pulse
    eda: np.ndarray    # (N,)   @ 4 Hz - electrodermal activity
    temp: np.ndarray   # (N,)   @ 4 Hz - temperature
    label: np.ndarray  # (M,)   @ 700 Hz, int

    # ---- streams --------------------------------------------------------

    def stream(self, name: str) -> np.ndarray:
        return getattr(self, name) # method returns the name of some stream which is a np array.

    def duration(self) -> float:
        """Recording duration bounded by the shortest available track."""
        ends = [len(self.stream(k)) / WRIST_FS[k] for k in WRIST_FS] # end = length of stream / sample rate for each stream in WRIST_FS, which is a dictionary of streams and their sample rates.
        ends.append(len(self.label) / LABEL_FS) # appends ends array with length of label stream / sample rate for label stream. the duration method cannot function without all data streams, so the label with shortest duration is used to determine what duration to use.
        return float(min(ends)) #retuns lowest duration of all streams, which is the duration of the recording.

    # ---- labels ---------------------------------------------------------

    def window_label(self, t0: float, t1: float, purity: float = 0.9): #purity = number of occurrences of top label / total number of labels in segment.
        """Single label for the window [t0, t1) or None if impure/out of scope.

        A window spanning a condition change carries no valid target, so it is
        rejected unless at least `purity` of its label samples agree and the
        majority class is one of KEEP_LABELS.
        """
        i0 = int(np.floor(t0 * LABEL_FS)) #i0 = gives a value for the start of label_fs window, gives start of window
        i1 = int(np.ceil(t1 * LABEL_FS)) # i1 gives end of the window
        # i0/i1 clamped below silently used to mean two different things: a
        # window that's legitimately empty (t0 past the recording), and a
        # window that's PARTIALLY out of bounds, silently scored on whatever
        # fraction of it happened to exist. The second case is a caller bug
        # (bad timestamps), not a valid edge case, so it must raise rather
        # than quietly return a label computed from a truncated segment.
        if i1 > len(self.label) or i0 < 0:
            if not (i0 >= len(self.label) or i1 <= 0):  # i.e. not fully out of range -> partial overlap
                raise ValueError(
                    f"window [{t0}, {t1})s partially outside label track "
                    f"(0..{len(self.label) / LABEL_FS:.1f}s) -- caller passed an out-of-range timestamp"
                )
        seg = self.label[max(i0, 0) : min(i1, len(self.label))] # this line gives segment of label stream that is withiin t0 and t1.
        if seg.size == 0: # if segment is empty, return None.
            return None
        vals, counts = np.unique(seg, return_counts=True) # vals = unique values in segment, counts = number of occurrences of each value
        top = int(vals[np.argmax(counts)]) # top = the label with the most occurrences
        if counts.max() / seg.size < purity: #if purity is < 0.9, return none. data not good enough to be used for training. 
            return None
        if top not in KEEP_LABELS: # means that the label is not one we keep for training.
            return None
        return top # 


def load_subject(sid: str, root: str | Path) -> Subject:
    """Load one subject pickle from <root>/<sid>/<sid>.pkl."""
    path = Path(root) / sid / f"{sid}.pkl" # finds path root for one subject in the file folder path.
    if not path.is_file():
        raise FileNotFoundError(f"missing pickle: {path}") # debug case for not finding path
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")  # Python 2 pickle -- necessary since wesad data is from python 2.
    w = d["signal"]["wrist"] # w means wrist data, this give dictionary of wrist data from the pickle file.

    def flat(a):
        return np.asarray(a, dtype=np.float64).reshape(-1) # this method takes np.asarray of a, converts to float64 and reshapes to 1d array. gives data in correct training format
        # for above, we need 1d array to match the live data format, which is a 1d array of data points.

    acc = np.asarray(w["ACC"], dtype=np.float64).reshape(-1, 3) / ACC_LSB_PER_G # raw E4 counts (1/64 g) -> g

    # Gravity alone should put median |ACC| near 1 g over a whole recording.
    # This is exactly the check that would have caught the 1/64 g unit bug
    # instead of it silently zeroing out the motion gate downstream.
    mag_med = float(np.median(np.linalg.norm(acc, axis=1)))
    if not (ACC_REST_MAG_MIN_G <= mag_med <= ACC_REST_MAG_MAX_G):
        raise RuntimeError(
            f"{sid}: median |ACC| = {mag_med:.3f} g, expected "
            f"{ACC_REST_MAG_MIN_G}-{ACC_REST_MAG_MAX_G} g (gravity). "
            "A unit conversion is wrong upstream of this check."
        )

    return Subject(
        sid=str(d.get("subject", sid)),
        acc=acc,
        bvp=flat(w["BVP"]),
        eda=flat(w["EDA"]),
        temp=flat(w["TEMP"]),
        label=np.asarray(d["label"]).reshape(-1).astype(np.int16),
    ) # retursn the subject class data stream information for one subject.