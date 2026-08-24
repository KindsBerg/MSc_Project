"""
live_host.py -- streaming inference engine for the wearable.

sits between a sample source and the exported model. buffers samples as they
come in, closes a 60s window every WIN_STEP_S seconds, runs the feature row
through the SAME features.py used in training, hands it to StressModel.

    python live_host.py --replay S2          # replay a WESAD subject
    python live_host.py --replay S2 --speed 0  # as fast as possible

source is deliberately abstracted -- replay exercises the whole inference
path with no hardware attached, a WebSocket source can swap in later without
touching anything below it.

--------------------------------------------------------------------------
two things I want to make sure I got right here
--------------------------------------------------------------------------
WINDOW GEOMETRY MATCHES TRAINING. 60s EDA window, 5s step. HR tiled in
non-overlapping 40s sub-windows, IMU in 15s sub-windows -- they split apart
once the SEN0344's real ~0.25Hz cadence got measured. aggregated at window
close. any mismatch here is a silent accuracy loss, so the constants come
straight from features.py instead of getting restated here.

UNIT CONVERSION HAPPENS HERE, NOT IN features.py. LSM6DS3TR-C reports accel
in m/s^2, every g-scaled threshold in features.py (the motion gate
especially) is calibrated in g. see ms2_to_g() / prepare_live_accel() below
-- features.py must only ever see g.

PARTIAL ROWS ARE FINE, MISSING MODEL INPUTS ARE NOT. the eda_only artefact
only needs the ten EDA columns, so the host has to work with no PPG
attached. blocks with no data give NaN; the row then gets checked against
the model's OWN column list, and only a missing/non-finite column the model
actually uses is an error.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

import features as F
from export_model import StressModel

# --------------------------------------------------------------------------
# IMU ingest -- the only place raw LSM6DS3TR-C output is allowed to become
# the g-valued, WESAD-rate samples features.py expects. real hardware
# sources have to go through this; push_acc() itself takes pre-converted g
# and stays the entry point for sources that already are (WESAD replay).
# --------------------------------------------------------------------------

SENSORS_GRAVITY_STANDARD = 9.80665  # m/s^2 per g (Adafruit_Sensor convention)
LSM6DS3TR_C_NATIVE_ODR_HZ = 104.0   # Adafruit_LSM6DS::_init() library default


@dataclass
class RawIMUSample:
    """one reading straight off the LSM6DS3TR-C, before any ingest processing.

    accel_ms2 is m/s^2 (Adafruit_LSM6DS::_read() scaling), gyro_rads is
    rad/s. neither is unit-converted, decimated, or quantised yet -- that's
    prepare_live_accel()'s job, not the source's.
    """

    t: float
    accel_ms2: tuple[float, float, float]
    gyro_rads: tuple[float, float, float]


def ms2_to_g(acc_ms2: np.ndarray) -> np.ndarray:
    """converts LSM6DS3TR-C acceleration from m/s^2 to g"""
    return np.asarray(acc_ms2, dtype=np.float64) / SENSORS_GRAVITY_STANDARD


def assert_accel_units_g(acc_g: np.ndarray, lo: float = 0.7, hi: float = 1.3) -> None:
    """raises unless a still-window's acceleration magnitude sits near 1g.

    guards the highest-risk mistake in the SEN0344/LSM6DS3TR-C migration:
    MOTION_STD_THRESHOLD_G is calibrated in g. if the m/s^2 -> g conversion
    ever silently gets skipped upstream, the motion gate desyncs from the
    real signal and both CROSS features quietly degrade with no error. this
    is a calibration check -- call it on a known-still block (e.g. warm-up),
    not on arbitrary in-motion data which will legitimately fail it.
    """
    acc_g = np.asarray(acc_g, dtype=np.float64)
    mag = np.linalg.norm(acc_g, axis=-1) if acc_g.ndim > 1 else np.abs(acc_g)
    mean_mag = float(np.mean(mag))
    if not (lo <= mean_mag <= hi):
        raise ValueError(
            f"still-window |acc| mean = {mean_mag:.3f}, expected {lo}-{hi} g. "
            f"Looks like m/s^2 (~{SENSORS_GRAVITY_STANDARD:.2f}) reached the "
            "ingest boundary unconverted."
        )


def decimate_imu(x: np.ndarray, native_fs: float, target_fs: float) -> np.ndarray:
    """polyphase-resamples a native-rate IMU block down to target_fs.

    LSM6DS3TR-C's 104Hz default ODR has no clean integer stride down to
    WESAD's 32Hz (104/32 = 3.25), so uses resample_poly's exact rational
    ratio instead of naive stride slicing.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    from math import gcd

    n, d = int(round(target_fs)), int(round(native_fs))
    g = gcd(n, d)
    return resample_poly(x, n // g, d // g, axis=0)


def quantise_to_e4_grid(acc_g: np.ndarray) -> np.ndarray:
    """rounds acceleration (g) onto the Empatica E4's 1/64g grid.

    LSM6DS3TR-C resolves 0.061 mg/LSB at +-2g, E4 resolves ~15.6mg -- ~250x
    coarser. on still windows the E4's acc_sd features are quantisation-
    floored in a way ours wouldn't be, so live would read systematically
    lower noise than anything in the training distribution.
    """
    return np.round(acc_g / F.E4_ACC_LSB_G) * F.E4_ACC_LSB_G


def prepare_live_accel(
    acc_ms2: np.ndarray,
    native_fs: float = LSM6DS3TR_C_NATIVE_ODR_HZ,
    target_fs: float = F.IMU_TARGET_FS_HZ,
) -> np.ndarray:
    """full live accel ingest: m/s^2 -> g -> decimate -> E4 grid.

    order matters (see the migration notes on features.py) -- convert units
    first, decimate second, quantise last and only if
    EMULATE_E4_QUANTISATION is set. WESAD training data is already on the
    E4 grid and shouldn't get quantised a second time, which is why this
    path is live-only.
    """
    acc_g = ms2_to_g(acc_ms2)
    acc_g = decimate_imu(acc_g, native_fs, target_fs)
    if F.EMULATE_E4_QUANTISATION:
        acc_g = quantise_to_e4_grid(acc_g)
    return acc_g


class Stream:
    """timestamped ring buffer for one sensor channel.

    holds a bit more than one window so a 60s slice is always available
    without unbounded growth. samples need to arrive in time order.
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
        """samples with t0 <= t < t1, in arrival order"""
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


# --------------------------------------------------------------------------
# resting-HR reference. two paths build it the same way: replay_wesad takes
# one slice straight from the pre-loaded recording (train-parity);
# LiveEngine's hardware path builds it up incrementally from push_hr, since
# real hardware has no pre-loaded array to slice. both use the same span,
# mask, and mean, so hr_baseline_delta means the same thing on every path
# that builds it -- build_dataset, replay, and live hardware.
# --------------------------------------------------------------------------

BASELINE_HR_REF_S = 300.0  # matches build_dataset.py's BASELINE_REF_S default


def _baseline_hr_from_samples(hr: np.ndarray) -> float | None:
    """resting-HR mean over a batch of samples, None if nothing usable.

    same validity mask as replay_wesad / build_dataset: finite, >20, <220 bpm.
    """
    hr = np.asarray(hr, dtype=np.float64)
    valid = hr[np.isfinite(hr) & (hr > 20) & (hr < 220)]
    return float(np.mean(valid)) if valid.size else None


class LiveEngine:
    """buffers samples, closes windows, produces predictions.

    feed it with push_eda / push_hr / push_acc, then call step() whenever
    time advances. step() returns a result dict at each window close and
    None otherwise.
    """

    def __init__(self, model: StressModel, hr_fs: float = F.SEN0344_HR_FS_HZ):
        self.model = model
        self.hr_fs = hr_fs

        # one window plus one step of slack, so a slice never gets truncated
        # by eviction happening a fraction early
        span = F.WIN_EDA_S + F.WIN_STEP_S
        self.eda = Stream(F.EDA_FS, span)
        self.hr = Stream(hr_fs, span)
        self.acc = Stream(F.ACC_FS, span, width=3)
        # not fed to the model (WESAD has no gyro) -- logged in rad/s just
        # for signal-quality gating
        self.gyro = Stream(F.IMU_TARGET_FS_HZ, span, width=3)

        self.t_origin: float | None = None
        self.next_close: float | None = None
        self.baseline_hr: float | None = None
        self.n_windows = 0

        # hardware-path resting-HR warm-up (see push_hr / _finalize_baseline_hr).
        # unused, never populated, when the loaded model doesn't need
        # hr_baseline_delta
        self._hr_ref_buffer: list[float] = []
        self._hr_ref_finalized = False

        # which columns the loaded artefact actually needs. everything else
        # in the row can legitimately be NaN
        self.required = set(model.feature_names)

        # derived from the loaded bundle's own feature list, not a
        # hardcoded set name -- whatever got exported, this follows it
        self._needs_baseline_hr = "hr_baseline_delta" in self.required

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

        # accumulating toward the resting-HR reference, same span as
        # replay_wesad's opening slice. self.hr can't be used for this --
        # it's a short ring buffer sized for one window, not
        # BASELINE_HR_REF_S seconds -- so warm-up needs its own buffer.
        # stops accumulating the moment baseline_hr gets set by any path
        # (including replay's direct call, see set_baseline_hr), so this
        # never recomputes or overwrites an already-established reference.
        if self._needs_baseline_hr and not self._hr_ref_finalized:
            if t - self.t_origin < BASELINE_HR_REF_S:
                self._hr_ref_buffer.append(bpm)

    def push_acc(self, t: float, x: float, y: float, z: float) -> None:
        """pushes one accel sample, already in g at F.ACC_FS.

        entry point for pre-converted sources (WESAD replay). real hardware
        needs to go through push_acc_raw() / prepare_live_accel() first --
        this one does no unit conversion, decimation, or quantisation.
        """
        self._mark(t)
        self.acc.push(t, (x, y, z))

    def push_acc_raw(
        self, t0: float, acc_ms2_block: np.ndarray, native_fs: float = LSM6DS3TR_C_NATIVE_ODR_HZ
    ) -> None:
        """ingests one batch of raw LSM6DS3TR-C samples (m/s^2, native ODR).

        matches the ESP32's batched-packet transport (notes.txt A.2) --
        decimation needs a block anyway since a lone raw sample has no
        frequency content to resample. t0 is the first raw sample's
        timestamp; pushed samples get re-timestamped at the decimated
        IMU_TARGET_FS_HZ spacing. runs prepare_live_accel() (unit
        conversion, decimation, E4-grid quantisation) before buffering.
        push_acc() is still the entry point for already-converted sources
        (WESAD replay).
        """
        acc_ms2_block = np.asarray(acc_ms2_block, dtype=np.float64).reshape(-1, 3)
        acc_g = prepare_live_accel(acc_ms2_block, native_fs=native_fs)
        dt_out = 1.0 / F.IMU_TARGET_FS_HZ
        for i, row in enumerate(acc_g):
            self.push_acc(t0 + i * dt_out, *row)

    def push_gyro(self, t: float, x_rads: float, y_rads: float, z_rads: float) -> None:
        """logs a gyro sample (rad/s). diagnostic only, never feeds feature_vector."""
        self._mark(t)
        self.gyro.push(t, (x_rads, y_rads, z_rads))

    # -- feature assembly -------------------------------------------------

    def _short_blocks(self, t0: float, t1: float):
        """tiles the window with non-overlapping short sub-windows.

        HR (40s) and IMU (15s) get tiled separately -- they split apart
        once the SEN0344's real cadence got measured. same geometry as
        build_dataset.build_subject, which is the only reason live
        features mean the same thing as trained ones.

        HR is anchored to window CLOSE (t1), not start -- a still-start
        anchor left the most recent 20s of the 60s window unused at
        prediction time, so the HR block was up to 20s stale. tiling walks
        backward from t1 in 40s steps so the LAST block always ends
        exactly at t1. with the current constants there's only ever 1 HR
        block per window so this loop looks like a no-op -- kept anyway
        because it's the real code path if either constant changes, don't
        mistake it for dead code.

        imu_blocks_hr_span = the IMU sub-blocks overlapping the HR blocks'
        span, by sub-window midpoint -- 15s doesn't evenly divide 40s so
        exact containment isn't possible, midpoint overlap is the standard
        approximation. this is what scopes the hr_delta_x_still cross-term,
        see features.cross_features().
        """
        n_hr_blocks = int((t1 - t0) // F.WIN_SHORT_HR_S)
        hr_blocks = []
        s = t1 - n_hr_blocks * F.WIN_SHORT_HR_S
        hr_span_start = s
        while s + F.WIN_SHORT_HR_S <= t1:
            e = s + F.WIN_SHORT_HR_S
            hr_win = self.hr.slice(s, e)
            if hr_win.size:
                hr_blocks.append(
                    F.hr_features(hr_win, self.hr_fs, baseline_hr=self.baseline_hr)
                )
            s = e
        hr_span_end = s  # == t1 whenever n_hr_blocks > 0

        imu_blocks, imu_blocks_hr_span = [], []
        s = t0
        while s + F.WIN_SHORT_IMU_S <= t1:
            e = s + F.WIN_SHORT_IMU_S
            acc_win = self.acc.slice(s, e)
            if acc_win.size:
                blk = F.imu_features(acc_win, F.ACC_FS)
                imu_blocks.append(blk)
                mid = (s + e) / 2.0
                if hr_span_start <= mid < hr_span_end:
                    imu_blocks_hr_span.append(blk)
            s = e
        return hr_blocks, imu_blocks, imu_blocks_hr_span

    def _row(self, t0: float, t1: float) -> dict:
        """one full feature row. missing streams give NaN, not an error."""
        eda_win = self.eda.slice(t0, t1)
        hr_blocks, imu_blocks, imu_blocks_hr_span = self._short_blocks(t0, t1)

        row: dict = {}
        row.update(
            F.eda_features(eda_win, F.EDA_FS)
            if eda_win.size
            else {k: np.nan for k in F.EDA_FEATURES}
        )
        row.update(F.aggregate_short_windows(hr_blocks, F.HR_FEATURES))
        row.update(F.aggregate_short_windows(imu_blocks, F.IMU_FEATURES))

        hr_span_fraction = F.aggregate_short_windows(
            imu_blocks_hr_span, F.IMU_FEATURES
        ).get("motion_fraction", np.nan)
        row.update(F.cross_features(row, hr_span_fraction))
        return {k: row[k] for k in F.FEATURE_NAMES}

    def _check_required(self, row: dict) -> list:
        """which columns the model needs that this row can't supply.

        hr_baseline_delta gets skipped while its own reference is still
        pending (see step()) -- expected to be NaN until baseline_hr is
        set, that's a warm-up state not a missing-data error.
        """
        skip = (
            {"hr_baseline_delta"}
            if self._needs_baseline_hr and self.baseline_hr is None
            else set()
        )
        return [
            c
            for c in self.model.feature_names
            if c not in skip and (c not in row or not np.isfinite(row[c]))
        ]

    def _finalize_baseline_hr(self) -> None:
        """one-shot: turns the warm-up buffer into baseline_hr.

        only called once the full BASELINE_HR_REF_S has actually elapsed
        (see step()), never on a partial buffer -- a short reference is
        worse than none, since hr_baseline_delta would then mean something
        different for this session than the ones it was trained against.
        if nothing survived the mask, baseline_hr just stays None and the
        engine stays not-ready -- see set_baseline_hr.
        """
        self._hr_ref_finalized = True
        baseline = _baseline_hr_from_samples(np.asarray(self._hr_ref_buffer, dtype=np.float64))
        if baseline is not None:
            self.set_baseline_hr(baseline)

    # -- the loop --------------------------------------------------------

    def step(self, now: float):
        """closes a window if one's due. returns a result dict or None."""
        if self.next_close is None or now < self.next_close:
            return None

        t1 = self.next_close
        t0 = t1 - F.WIN_EDA_S
        self.next_close = t1 + F.WIN_STEP_S
        self.n_windows += 1

        if (
            self._needs_baseline_hr
            and not self._hr_ref_finalized
            and t1 - self.t_origin >= BASELINE_HR_REF_S
        ):
            self._finalize_baseline_hr()

        row = self._row(t0, t1)
        bad = self._check_required(row)
        if bad:
            return {
                "t": t1,
                "state": "incomplete",
                "detail": f"{len(bad)} required column(s) unusable: {bad[:3]}",
            }

        # warm-up: two independent references, neither inherited from
        # WESAD subjects -- different skin, different baseline
        #   model.ready    — EDA standardisation reference (model.add_reference)
        #   baseline_hr    — wearer's own resting HR, only gated in when the
        #                    loaded model's feature set needs it
        waiting_on_baseline_hr = self._needs_baseline_hr and self.baseline_hr is None
        if not self.model.ready or waiting_on_baseline_hr:
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
        """resting HR for hr_baseline_delta -- required by feature sets
        that include it (e.g. device), unused otherwise (e.g. eda_only).

        marks the warm-up buffer finalized regardless of caller, so
        whichever path sets this first -- replay's direct call or this
        engine's own hardware-path accumulator -- the other never
        re-triggers or overwrites it.
        """
        self.baseline_hr = float(bpm)
        self._hr_ref_finalized = True


# ==========================================================================
# sources
# ==========================================================================


def replay_wesad(engine: LiveEngine, sid: str, root: str, speed: float = 1.0):
    """pushes one WESAD subject through the engine as if it arrived live.

    this is the integration test for the whole inference path: same
    feature module, same window geometry, same model, no hardware. if a
    subject replays to sensible predictions here, the only unknown left is
    the sensors themselves.

    speed=1.0 is real time, speed=0 runs as fast as the CPU allows.
    """
    from wesad_loader import WRIST_FS, load_subject

    sub = load_subject(sid, root)
    hr_series = F.bvp_to_hr(sub.bvp, fs=WRIST_FS["bvp"], out_fs=engine.hr_fs)
    hr = hr_series.values

    # resting reference from the start of the recording, exactly like
    # build_dataset does -- chosen without looking at any label. same
    # span, mask, and mean as LiveEngine's hardware-path accumulator, see
    # BASELINE_HR_REF_S / _baseline_hr_from_samples
    ref = hr[: int(BASELINE_HR_REF_S * engine.hr_fs)]
    baseline = _baseline_hr_from_samples(ref)
    if baseline is not None:
        engine.set_baseline_hr(baseline)

    duration = sub.duration()

    # merge every channel into one time-ordered event list, so the engine
    # sees the same interleaving a real multi-rate stream would produce
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
# entry point
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
    ap.add_argument("--hr-fs", type=float, default=F.SEN0344_HR_FS_HZ,
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
