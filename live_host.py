"""
live_host.py — streaming inference engine for the wearable.

Sits between a sample source and the exported model. Accumulates samples as
they arrive, closes a 60 s window every WIN_STEP_S seconds, computes the
feature row through the SAME features.py used in training, and hands it to
StressModel.

    python live_host.py --replay S2          # replay a WESAD subject
    python live_host.py --replay S2 --speed 0  # as fast as possible

The source is deliberately abstracted. Replay exercises the entire inference
path with no hardware attached; a WebSocket source swaps in later without
touching anything below it.

--------------------------------------------------------------------------
Two things this file gets right on purpose
--------------------------------------------------------------------------
WINDOW GEOMETRY MATCHES TRAINING. 60 s EDA window, 5 s step, four
non-overlapping 15 s sub-windows for HR and IMU, aggregated at window close.
Any divergence here is a silent accuracy loss, so the constants come from
features.py rather than being restated.

PARTIAL ROWS ARE LEGAL, MISSING MODEL INPUTS ARE NOT. The eda_only artefact
needs ten EDA columns and nothing else, so the host must run with no PPG
attached. Blocks with no data yield NaN; the row is then checked against the
model's OWN column list, and only a missing or non-finite column the model
actually uses is an error.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

import features as F
from export_model import StressModel


class Stream:
    """Timestamped ring buffer for one sensor channel.

    Holds a little more than one window so a 60 s slice is always available
    without unbounded growth. Samples must arrive in time order.
    """

    def __init__(self, fs: float, span_s: float, width: int = 1):
        self.fs = fs
        self.span_s = span_s
        self.width = width
        self.t: deque = deque()
        self.v: deque = deque()

    def push(self, t: float, value) -> None:
        self.t.append(float(t))
        self.v.append(value)
        cutoff = t - self.span_s
        while self.t and self.t[0] < cutoff:
            self.t.popleft()
            self.v.popleft()

    def slice(self, t0: float, t1: float) -> np.ndarray:
        """Samples with t0 <= t < t1, in arrival order."""
        ts = np.fromiter(self.t, dtype=np.float64)
        if ts.size == 0:
            return np.empty((0, self.width)) if self.width > 1 else np.empty(0)
        sel = (ts >= t0) & (ts < t1)
        vals = [x for x, keep in zip(self.v, sel) if keep]
        if not vals:
            return np.empty((0, self.width)) if self.width > 1 else np.empty(0)
        arr = np.asarray(vals, dtype=np.float64)
        return arr.reshape(-1, self.width) if self.width > 1 else arr.reshape(-1)

    @property
    def latest(self) -> float:
        return self.t[-1] if self.t else -np.inf


class LiveEngine:
    """Accumulates samples, closes windows, produces predictions.

    Feed it with push_eda / push_hr / push_acc, then call step() whenever
    time advances. step() returns a result dict at each window close and
    None otherwise.
    """

    def __init__(self, model: StressModel, hr_fs: float = 1.0):
        self.model = model
        self.hr_fs = hr_fs

        # One window plus one step of slack, so a slice is never truncated by
        # eviction happening a fraction early.
        span = F.WIN_EDA_S + F.WIN_STEP_S
        self.eda = Stream(F.EDA_FS, span)
        self.hr = Stream(hr_fs, span)
        self.acc = Stream(F.ACC_FS, span, width=3)

        self.t_origin: float | None = None
        self.next_close: float | None = None
        self.baseline_hr: float | None = None
        self.n_windows = 0

        # Which columns the loaded artefact actually needs. Everything else
        # in the 36-feature row may legitimately be NaN.
        self.required = set(model.feature_names)

    # -- ingest ----------------------------------------------------------

    def _mark(self, t: float) -> None:
        if self.t_origin is None:
            self.t_origin = t
            self.next_close = t + F.WIN_EDA_S

    def push_eda(self, t: float, microsiemens: float) -> None:
        self._mark(t)
        self.eda.push(t, microsiemens)

    def push_hr(self, t: float, bpm: float) -> None:
        self._mark(t)
        self.hr.push(t, bpm)

    def push_acc(self, t: float, x: float, y: float, z: float) -> None:
        self._mark(t)
        self.acc.push(t, (x, y, z))

    # -- feature assembly -------------------------------------------------

    def _short_blocks(self, t0: float, t1: float):
        """Tile the window with non-overlapping short sub-windows.

        Identical geometry to build_dataset.build_subject, which is the only
        reason the live features mean the same thing as the trained ones.
        """
        hr_blocks, imu_blocks = [], []
        s = t0
        while s + F.WIN_SHORT_S <= t1:
            e = s + F.WIN_SHORT_S
            hr_win = self.hr.slice(s, e)
            if hr_win.size:
                hr_blocks.append(
                    F.hr_features(hr_win, self.hr_fs, baseline_hr=self.baseline_hr)
                )
            acc_win = self.acc.slice(s, e)
            if acc_win.size:
                imu_blocks.append(F.imu_features(acc_win, F.ACC_FS))
            s = e
        return hr_blocks, imu_blocks

    def _row(self, t0: float, t1: float) -> dict:
        """One full 36-column row. Absent streams yield NaN, not an error."""
        eda_win = self.eda.slice(t0, t1)
        hr_blocks, imu_blocks = self._short_blocks(t0, t1)

        row: dict = {}
        row.update(
            F.eda_features(eda_win, F.EDA_FS)
            if eda_win.size
            else {k: np.nan for k in F.EDA_FEATURES}
        )
        row.update(F.aggregate_short_windows(hr_blocks, F.HR_FEATURES))
        row.update(F.aggregate_short_windows(imu_blocks, F.IMU_FEATURES))
        row.update(F.cross_features(row))
        return {k: row[k] for k in F.FEATURE_NAMES}

    def _check_required(self, row: dict) -> list:
        """Columns the model needs that this row cannot supply."""
        return [
            c
            for c in self.model.feature_names
            if c not in row or not np.isfinite(row[c])
        ]

    # -- the loop --------------------------------------------------------

    def step(self, now: float):
        """Close a window if one is due. Returns a result dict or None."""
        if self.next_close is None or now < self.next_close:
            return None

        t1 = self.next_close
        t0 = t1 - F.WIN_EDA_S
        self.next_close = t1 + F.WIN_STEP_S
        self.n_windows += 1

        row = self._row(t0, t1)
        bad = self._check_required(row)
        if bad:
            return {
                "t": t1,
                "state": "incomplete",
                "detail": f"{len(bad)} required column(s) unusable: {bad[:3]}",
            }

        # Warm-up: the wearer's own resting reference. Cannot be inherited
        # from WESAD subjects — different skin, different baseline.
        if not self.model.ready:
            self.model.add_reference(row)
            return {
                "t": t1,
                "state": "warmup",
                "n_ref": self.model.n_reference,
                "n_required": self.model.n_ref_required,
            }

        label, proba = self.model.predict(row)
        return {"t": t1, "state": "ok", "label": label, "proba": proba, "row": row}

    def set_baseline_hr(self, bpm: float) -> None:
        """Resting HR for hr_baseline_delta. Unused by the eda_only model."""
        self.baseline_hr = float(bpm)


# ==========================================================================
# Sources
# ==========================================================================


def replay_wesad(engine: LiveEngine, sid: str, root: str, speed: float = 1.0):
    """Push one WESAD subject through the engine as if it were arriving live.

    This is the integration test for the whole inference path: same feature
    module, same window geometry, same model, no hardware. If a subject
    replays to sensible predictions here, the only remaining unknown is the
    sensors themselves.

    speed=1.0 is real time; speed=0 runs as fast as the CPU allows.
    """
    from wesad_loader import WRIST_FS, load_subject

    sub = load_subject(sid, root)
    hr_series = F.bvp_to_hr(sub.bvp, fs=WRIST_FS["bvp"], out_fs=engine.hr_fs)
    hr = hr_series.values

    # Resting reference from the opening of the recording, exactly as
    # build_dataset does — chosen without reference to any label.
    ref = hr[: int(600 * engine.hr_fs)]
    ref = ref[np.isfinite(ref) & (ref > 20) & (ref < 220)]
    if ref.size:
        engine.set_baseline_hr(float(np.mean(ref)))

    duration = sub.duration()

    # Merge every channel into one time-ordered event list, so the engine
    # sees the same interleaving a real multi-rate stream would produce.
    events = []
    for i, v in enumerate(sub.eda):
        t = i / WRIST_FS["eda"]
        if t < duration:
            events.append((t, "eda", float(v)))
    for i, v in enumerate(hr):
        t = i / engine.hr_fs
        if t < duration:
            events.append((t, "hr", float(v)))
    for i, v in enumerate(sub.acc):
        t = i / WRIST_FS["acc"]
        if t < duration:
            events.append((t, "acc", tuple(float(x) for x in v)))
    events.sort(key=lambda e: e[0])

    print(f"replaying {sid}: {len(events)} samples over {duration:.0f}s")
    wall0 = time.time()
    last_report = -1.0

    for t, kind, val in events:
        if speed > 0:
            behind = (t / speed) - (time.time() - wall0)
            if behind > 0:
                time.sleep(behind)

        if kind == "eda":
            engine.push_eda(t, val)
        elif kind == "hr":
            engine.push_hr(t, val)
        else:
            engine.push_acc(t, *val)

        res = engine.step(t)
        if res:
            yield res

        if speed == 0 and t - last_report >= 300:
            last_report = t
            print(f"  ... t={t:.0f}s")


# ==========================================================================
# Entry point
# ==========================================================================


def format_result(r: dict) -> str:
    if r["state"] == "warmup":
        return f"t={r['t']:7.0f}s  warm-up {r['n_ref']:>3}/{r['n_required']}"
    if r["state"] == "incomplete":
        return f"t={r['t']:7.0f}s  SKIP  {r['detail']}"
    p = "  ".join(f"{k}={v:.2f}" for k, v in sorted(r["proba"].items()))
    return f"t={r['t']:7.0f}s  {r['label']:<11} {p}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/stress_model_eda.joblib")
    ap.add_argument("--replay", default=None, help="WESAD subject id, e.g. S2")
    ap.add_argument("--root", default=None, help="WESAD root (default config.WESAD_ROOT)")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="1.0 = real time, 0 = as fast as possible")
    ap.add_argument("--hr-fs", type=float, default=1.0,
                    help="HR update rate in Hz — must match training")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"no artefact at {model_path} — run export_model.py first")
        return 1

    m = StressModel.load(model_path)
    print(m.describe())
    print()

    engine = LiveEngine(m, hr_fs=args.hr_fs)

    if not args.replay:
        print("no source selected. --replay <subject> is the only source "
              "implemented; the WebSocket source lands with the hardware.")
        return 1

    root = args.root
    if root is None:
        try:
            from config import WESAD_ROOT

            root = WESAD_ROOT
        except Exception:  # noqa: BLE001
            print("no --root and no config.WESAD_ROOT")
            return 1

    counts: dict = {}
    t_first_prediction = None
    for res in replay_wesad(engine, args.replay, root, speed=args.speed):
        if res["state"] == "ok":
            counts[res["label"]] = counts.get(res["label"], 0) + 1
            if t_first_prediction is None:
                t_first_prediction = res["t"]
        if not args.quiet:
            print(format_result(res))

    print(f"\nwindows closed:     {engine.n_windows}")
    print(f"first prediction at: {t_first_prediction}s"
          if t_first_prediction
          else "\nnever reached readiness")
    total = sum(counts.values()) or 1
    for k, v in sorted(counts.items()):
        print(f"  {k:<12} {v:>5}  ({v / total:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())