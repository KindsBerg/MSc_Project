"""
features.py — THE shared feature module.

This file is imported by BOTH the WESAD training script and the live host
pipeline. It is the single implementation of every feature. Nothing here may
depend on which side is calling it, and no feature may be computed anywhere
else in the project.

Feature parity is a hard rule: the same columns, in the same order, over the
same per-stream window, in training and at inference. FEATURE_NAMES is the
single source of truth for that order; feature_vector() asserts against it on
every call so a drift becomes an immediate error rather than a silent
misalignment between the trained model and the live input.

--------------------------------------------------------------------------
Design constraints inherited from the sensor set (see WESAD_NOTES §2.2)
--------------------------------------------------------------------------
* NO HRV. RMSSD/SDNN/pNN50/LF/HF all need beat-to-beat intervals from raw
  BVP. The SEN0344 v2.0 firmware does not expose the MAX30102 FIFO, so only
  a computed HR number is available. The HR block is therefore statistics on
  the HR *number*, not on a pulse waveform. SD_HR below is the variability of
  the HR trend and must never be described as HRV.
* NO GYROSCOPE in the trained vector. The MPU6050 has a gyro; the Empatica
  E4 that recorded WESAD does not. Features are extracted on the
  intersection of the two, not the union, so gyro is excluded here. It may
  still be logged and used for live signal-quality gating outside the model.
* NO SpO2. Same data-access reason as HRV.

--------------------------------------------------------------------------
Window scheme
--------------------------------------------------------------------------
EDA/TEMP  60 s  — matches WESAD's own 60 s physiological window exactly.
HR/IMU    WIN_SHORT_S — shorter, per the multi-rate architecture. WESAD HR
          features MUST be recomputed on this same short window during
          training, or "mean HR" means two different things either side of
          the pipeline.

WIN_SHORT_S is PROVISIONAL. It is set to 15 s pending measurement of the
SEN0344's actual HR update cadence on the bench. If the sensor emits HR
every ~1-4 s, a 15 s window holds only 4-15 samples and SD/slope will be
coarse. Once the real cadence is known, set WIN_SHORT_S here and rebuild
the WESAD feature table — it is a single constant precisely so that
recomputation is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

# --------------------------------------------------------------------------
# Window and threshold configuration — change here, nowhere else.
# --------------------------------------------------------------------------

WIN_EDA_S = 60.0  # EDA/TEMP window, matches WESAD
WIN_SHORT_S = 15.0  # HR/IMU window — PROVISIONAL, see module docstring
WIN_STEP_S = 5.0  # slide between consecutive EDA windows
WINDOW_PURITY = 0.9  # min fraction of agreeing labels for a valid window

EDA_FS = 4.0  # Hz, WESAD wrist EDA and target for the live AFE
TEMP_FS = 4.0  # Hz
ACC_FS = 32.0  # Hz, E4; MPU6050 configured to match

# SCR detection: amplitude below this is treated as noise, not a response.
SCR_MIN_AMP_US = 0.01

# IMU motion gate. Std of 3D acceleration magnitude, in g, above which the
# window is flagged as motion-contaminated. Provisional — retune against
# your own worn recordings, then report the value used.
MOTION_STD_THRESHOLD_G = 0.05

# Band for the accelerometer peak-frequency feature (human movement).
ACC_BAND_HZ = (0.3, 10.0)

# Run-time counters. cvxEDA occasionally fails to converge on a short window
# and silently falls back to the median-filter decomposition; the fallback
# rate is a number that belongs in the report, so it is counted rather than
# lost. Reset with reset_stats() at the start of a batch run.
STATS = {
    "eda_windows": 0,
    "cvxeda_ok": 0,
    "cvxeda_fallback": 0,
    "cvxeda_error": None,   # first fallback reason, for diagnosis
    "scr_windows": 0,
    "scr_fallback": 0,
    "scr_error": None,
}


def reset_stats() -> None:
    for k in STATS:
        STATS[k] = None if k.endswith("_error") else 0


# --------------------------------------------------------------------------
# Feature name registry — the contract between training and inference.
# --------------------------------------------------------------------------

EDA_FEATURES = [
    "eda_scl_mean",
    "eda_scl_sd",
    "eda_scr_sd",
    "eda_range",
    "eda_slope",
    "eda_scr_count",
    "eda_scr_amp_sum",
    "eda_scr_amp_mean",
    "eda_scr_risetime_mean",
    "eda_scr_duration_sum",
]

HR_FEATURES = [
    "hr_mean",
    "hr_sd",
    "hr_min",
    "hr_max",
    "hr_range",
    "hr_slope",
    "hr_baseline_delta",
]

IMU_FEATURES = [
    "acc_x_mean", "acc_y_mean", "acc_z_mean", "acc_mag_mean",
    "acc_x_sd", "acc_y_sd", "acc_z_sd", "acc_mag_sd",
    "acc_x_absint", "acc_y_absint", "acc_z_absint", "acc_mag_absint",
    "acc_x_peakfreq", "acc_y_peakfreq", "acc_z_peakfreq",
    "motion_flag",
    "motion_fraction",
]

TEMP_FEATURES = [
    "temp_mean",
    "temp_min",
    "temp_max",
    "temp_range",
    "temp_slope",
    "temp_baseline_delta",
]

CROSS_FEATURES = [
    "hr_delta_x_still",   # hr_baseline_delta * (1 - motion_fraction)
    "eda_range_gated",    # eda_range * (1 - motion_fraction)
]

FEATURE_NAMES: list[str] = (
    EDA_FEATURES + HR_FEATURES + IMU_FEATURES + TEMP_FEATURES + CROSS_FEATURES
)

N_FEATURES = len(FEATURE_NAMES)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _slope(x: np.ndarray, fs: float) -> float:
    """Least-squares slope in units per second. NaN if under two samples."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        return np.nan
    t = np.arange(n, dtype=np.float64) / fs
    return float(np.polyfit(t, x, 1)[0])


def _abs_integral(x: np.ndarray, fs: float) -> float:
    """Integral of |x| over the window, in signal-units x seconds."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.nan
    return float(np.sum(np.abs(x)) / fs)


def _peak_freq(x: np.ndarray, fs: float, band=ACC_BAND_HZ) -> float:
    """Dominant in-band frequency via periodogram. NaN if window too short."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8:
        return np.nan
    x = x - x.mean()
    if not np.any(x):
        return 0.0
    f, p = sps.periodogram(x, fs=fs, scaling="density")
    sel = (f >= band[0]) & (f <= band[1])
    if not sel.any():
        return np.nan
    return float(f[sel][int(np.argmax(p[sel]))])


def _nan_block(names: list[str]) -> dict:
    return {k: np.nan for k in names}


# --------------------------------------------------------------------------
# EDA block — 60 s window
# --------------------------------------------------------------------------


def _decompose_eda(eda: np.ndarray, fs: float):
    """Split EDA into tonic (SCL) and phasic (SCR) components.

    Prefers NeuroKit2's cvxEDA (Greco et al.), which is the modern convex
    -optimisation replacement for WESAD's Choi decomposition. Falls back to a
    median-filter tonic estimate if NeuroKit2 is unavailable or cvxEDA fails
    to converge on a short 240-sample window, which it occasionally does.

    Returns (tonic, phasic, used_cvxeda).
    """
    try:
        import neurokit2 as nk

        df = nk.eda_phasic(eda, sampling_rate=fs, method="cvxeda")
        return (
            df["EDA_Tonic"].to_numpy(),
            df["EDA_Phasic"].to_numpy(),
            True,
        )
    except Exception as e:  # noqa: BLE001 — any failure falls back, by design
        # Record the FIRST reason only. A 100% fallback rate means cvxEDA
        # never ran at all (usually a missing `cvxopt` install), which is a
        # different problem from occasional non-convergence and must not be
        # allowed to look like one.
        if STATS["cvxeda_error"] is None:
            STATS["cvxeda_error"] = f"{type(e).__name__}: {e}"
        # Median filter wide enough to pass SCL but reject SCRs (~1-5 s).
        k = int(fs * 8) | 1  # odd kernel
        k = max(k, 3)
        if eda.size < k:
            tonic = np.full_like(eda, np.median(eda))
        else:
            tonic = sps.medfilt(eda, kernel_size=k)
        return tonic, eda - tonic, False


def _find_scrs(phasic: np.ndarray, fs: float):
    """Detect SCRs, returning (amplitudes, rise_times_s, durations_s)."""
    STATS["scr_windows"] += 1
    try:
        import neurokit2 as nk

        _, info = nk.eda_peaks(phasic, sampling_rate=fs, method="neurokit")
        amp = np.asarray(info.get("SCR_Amplitude", []), dtype=np.float64)
        rise = np.asarray(info.get("SCR_RiseTime", []), dtype=np.float64)
        rec = np.asarray(info.get("SCR_RecoveryTime", []), dtype=np.float64)
        keep = np.isfinite(amp) & (amp >= SCR_MIN_AMP_US)
        amp = amp[keep]
        rise = rise[keep] if rise.size == amp.size else rise[: amp.size]
        rec = rec[keep] if rec.size == amp.size else rec[: amp.size]
        # Recovery time is frequently NaN when the window truncates the tail;
        # duration falls back to rise time alone in that case.
        dur = np.where(np.isfinite(rec), rise + rec, rise)
        return amp, rise, dur
    except Exception as e:  # noqa: BLE001
        if STATS["scr_error"] is None:
            STATS["scr_error"] = f"{type(e).__name__}: {e}"
        STATS["scr_fallback"] += 1
        # Fallback: simple prominence-based peak count on the phasic driver.
        pk, props = sps.find_peaks(
            phasic, prominence=SCR_MIN_AMP_US, width=1
        )
        amp = np.asarray(props.get("prominences", []), dtype=np.float64)
        widths = np.asarray(props.get("widths", []), dtype=np.float64) / fs
        return amp, widths * 0.5, widths


def eda_features(eda: np.ndarray, fs: float = EDA_FS) -> dict:
    """Ten EDA features over one 60 s window. Input in microsiemens."""
    eda = np.asarray(eda, dtype=np.float64).reshape(-1)
    if eda.size < int(fs * 5):  # under 5 s of data is not usable
        return _nan_block(EDA_FEATURES)

    tonic, phasic, used_cvxeda = _decompose_eda(eda, fs)
    STATS["eda_windows"] += 1
    STATS["cvxeda_ok" if used_cvxeda else "cvxeda_fallback"] += 1
    amp, rise, dur = _find_scrs(phasic, fs)

    # NeuroKit2 returns NaN rise/recovery times for SCRs whose onset falls
    # before the window opens. Filtering to finite values first avoids a
    # "Mean of empty slice" warning AND, more importantly, stops the feature
    # silently becoming NaN when every SCR in the window is truncated.
    # Zero is the correct fallback: "no measurable rise time in this window"
    # is nearer zero than missing, and it keeps the column dense.
    amp_ok = amp[np.isfinite(amp)]
    rise_ok = rise[np.isfinite(rise)]
    dur_ok = dur[np.isfinite(dur)]

    return {
        "eda_scl_mean": float(np.mean(tonic)),
        "eda_scl_sd": float(np.std(tonic)),
        "eda_scr_sd": float(np.std(phasic)),
        "eda_range": float(np.ptp(eda)),
        "eda_slope": _slope(eda, fs),
        "eda_scr_count": float(amp_ok.size),
        "eda_scr_amp_sum": float(np.sum(amp_ok)) if amp_ok.size else 0.0,
        "eda_scr_amp_mean": float(np.mean(amp_ok)) if amp_ok.size else 0.0,
        "eda_scr_risetime_mean": float(np.mean(rise_ok)) if rise_ok.size else 0.0,
        "eda_scr_duration_sum": float(np.sum(dur_ok)) if dur_ok.size else 0.0,
    }


# --------------------------------------------------------------------------
# HR block — short window
# --------------------------------------------------------------------------


def hr_features(hr: np.ndarray, fs: float, baseline_hr: float | None = None) -> dict:
    """Statistics on the computed HR stream over one short window.

    `hr`  : sequence of HR values in bpm, already at the device's update rate.
    `fs`  : that update rate in Hz (WESAD side must be resampled to match).
    `baseline_hr` : rolling personal resting HR; None yields NaN for the delta.

    Deliberately contains no HRV. See module docstring.
    """
    hr = np.asarray(hr, dtype=np.float64).reshape(-1)
    hr = hr[np.isfinite(hr) & (hr > 20.0) & (hr < 220.0)]  # drop implausible
    if hr.size == 0:
        return _nan_block(HR_FEATURES)

    mean = float(np.mean(hr))
    return {
        "hr_mean": mean,
        "hr_sd": float(np.std(hr)) if hr.size > 1 else 0.0,
        "hr_min": float(np.min(hr)),
        "hr_max": float(np.max(hr)),
        "hr_range": float(np.ptp(hr)),
        "hr_slope": _slope(hr, fs),
        "hr_baseline_delta": (
            mean - float(baseline_hr) if baseline_hr is not None else np.nan
        ),
    }


# --------------------------------------------------------------------------
# IMU block — short window, accelerometer only
# --------------------------------------------------------------------------


def imu_features(acc: np.ndarray, fs: float = ACC_FS) -> dict:
    """Accelerometer features. `acc` is (N, 3) in g. Gyro excluded by design."""
    acc = np.asarray(acc, dtype=np.float64)
    if acc.ndim != 2 or acc.shape[1] != 3 or acc.shape[0] < 4:
        return _nan_block(IMU_FEATURES)

    mag = np.linalg.norm(acc, axis=1)
    out: dict = {}
    for i, ax in enumerate("xyz"):
        out[f"acc_{ax}_mean"] = float(np.mean(acc[:, i]))
        out[f"acc_{ax}_sd"] = float(np.std(acc[:, i]))
        out[f"acc_{ax}_absint"] = _abs_integral(acc[:, i] - np.mean(acc[:, i]), fs)
        out[f"acc_{ax}_peakfreq"] = _peak_freq(acc[:, i], fs)

    out["acc_mag_mean"] = float(np.mean(mag))
    out["acc_mag_sd"] = float(np.std(mag))
    out["acc_mag_absint"] = _abs_integral(mag - np.mean(mag), fs)

    # Motion gate. In a single short window the flag is binary and the
    # fraction is its float form; when short windows are aggregated into a
    # 60 s vector, motion_fraction becomes the share of flagged sub-windows.
    flag = 1.0 if out["acc_mag_sd"] > MOTION_STD_THRESHOLD_G else 0.0
    out["motion_flag"] = flag
    out["motion_fraction"] = flag
    return out


# --------------------------------------------------------------------------
# TEMP block — slow context, attached to the 60 s window
# --------------------------------------------------------------------------


def temp_features(
    temp: np.ndarray, fs: float = TEMP_FS, baseline_temp: float | None = None
) -> dict:
    """Skin-temperature features.

    Only slope and baseline-delta are strongly trustworthy on the target
    hardware: the LM75BD sits on the PCB and reads a mixture of skin
    temperature and board self-heating from the ESP32 and regulators. The
    absolute mean is retained because WESAD has it, but should be expected
    to carry an offset at inference and to be down-weighted accordingly.
    """
    temp = np.asarray(temp, dtype=np.float64).reshape(-1)
    temp = temp[np.isfinite(temp)]
    if temp.size == 0:
        return _nan_block(TEMP_FEATURES)

    mean = float(np.mean(temp))
    return {
        "temp_mean": mean,
        "temp_min": float(np.min(temp)),
        "temp_max": float(np.max(temp)),
        "temp_range": float(np.ptp(temp)),
        "temp_slope": _slope(temp, fs),
        "temp_baseline_delta": (
            mean - float(baseline_temp) if baseline_temp is not None else np.nan
        ),
    }


# --------------------------------------------------------------------------
# Short-window aggregation
# --------------------------------------------------------------------------


def aggregate_short_windows(blocks: list[dict], names: list[str]) -> dict:
    """Collapse buffered short-window feature dicts into one 60 s summary.

    Applied at each EDA-window close to the HR and IMU blocks. The rule per
    feature is chosen to preserve its meaning: extremes take extremes, counts
    and flags take means, everything else takes the mean.
    """
    if not blocks:
        return _nan_block(names)

    out: dict = {}
    for k in names:
        vals = np.array([b.get(k, np.nan) for b in blocks], dtype=np.float64)
        if np.all(np.isnan(vals)):
            out[k] = np.nan
        elif k.endswith("_min"):
            out[k] = float(np.nanmin(vals))
        elif k.endswith("_max"):
            out[k] = float(np.nanmax(vals))
        else:
            out[k] = float(np.nanmean(vals))

    # motion_fraction is the share of flagged sub-windows, not their mean
    # magnitude — recompute it explicitly so the semantics are unambiguous.
    if "motion_flag" in names:
        flags = np.array(
            [b.get("motion_flag", np.nan) for b in blocks], dtype=np.float64
        )
        if not np.all(np.isnan(flags)):
            frac = float(np.nanmean(flags))
            out["motion_fraction"] = frac
            out["motion_flag"] = 1.0 if frac > 0.5 else 0.0
    return out


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def cross_features(merged: dict) -> dict:
    """Interaction terms letting the model discount motion-driven change."""
    frac = merged.get("motion_fraction", np.nan)
    still = 1.0 - frac if np.isfinite(frac) else np.nan
    hr_d = merged.get("hr_baseline_delta", np.nan)
    eda_r = merged.get("eda_range", np.nan)
    return {
        "hr_delta_x_still": hr_d * still if np.isfinite(hr_d) else np.nan,
        "eda_range_gated": eda_r * still if np.isfinite(eda_r) else np.nan,
    }


def feature_vector(
    eda_win: np.ndarray,
    temp_win: np.ndarray,
    hr_short_blocks: list[dict],
    imu_short_blocks: list[dict],
    baseline_temp: float | None = None,
    eda_fs: float = EDA_FS,
    temp_fs: float = TEMP_FS,
) -> dict:
    """Assemble one complete feature row at an EDA-window close.

    `hr_short_blocks` / `imu_short_blocks` are the buffered outputs of
    hr_features() / imu_features() over the short windows falling inside this
    60 s window. Baseline HR is applied inside hr_features() at buffer time,
    not here.

    Returns a dict whose keys are exactly FEATURE_NAMES, in that order.
    """
    row: dict = {}
    row.update(eda_features(eda_win, eda_fs))
    row.update(aggregate_short_windows(hr_short_blocks, HR_FEATURES))
    row.update(aggregate_short_windows(imu_short_blocks, IMU_FEATURES))
    row.update(temp_features(temp_win, temp_fs, baseline_temp))
    row.update(cross_features(row))

    missing = set(FEATURE_NAMES) - set(row)
    extra = set(row) - set(FEATURE_NAMES)
    if missing or extra:
        raise RuntimeError(
            f"feature contract violated — missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
    return {k: row[k] for k in FEATURE_NAMES}


def to_array(row: dict) -> np.ndarray:
    """Ordered numeric array for the model. Order is FEATURE_NAMES, always."""
    return np.array([row[k] for k in FEATURE_NAMES], dtype=np.float64)


# --------------------------------------------------------------------------
# WESAD-side adapter — training only, NOT part of the shared contract.
# --------------------------------------------------------------------------


@dataclass
class HRSeries:
    values: np.ndarray
    fs: float


def bvp_to_hr(bvp: np.ndarray, fs: float = 64.0, out_fs: float = 1.0) -> HRSeries:
    """Derive an HR series from WESAD's raw BVP, emulating the SEN0344 output.

    This exists ONLY on the training side. WESAD provides a 64 Hz BVP
    waveform; the target hardware provides a computed HR number at a low
    update rate. Training on the waveform and inferring on the number would
    not transfer, so the waveform is reduced to a number here first.

    `out_fs` is the emulated HR update rate and is PROVISIONAL at 1 Hz. It
    must be set to the SEN0344's measured cadence once the board is on the
    bench, and the WESAD feature table rebuilt.

    Note for the report: this is a genuine domain-gap contributor. NeuroKit2's
    beat detector and DFRobot's undocumented on-board estimator are different
    algorithms with different latency and different motion rejection, so
    "mean HR" is the same quantity but not the same measurement either side
    of the pipeline. State it rather than let it be found.
    """
    import neurokit2 as nk

    bvp = np.asarray(bvp, dtype=np.float64).reshape(-1)
    sig, info = nk.ppg_process(bvp, sampling_rate=fs)
    inst = sig["PPG_Rate"].to_numpy()

    # Decimate the per-sample instantaneous rate to the emulated update rate
    # by averaging within each output interval — closer to what an on-board
    # estimator reports than naive subsampling would be.
    step = int(round(fs / out_fs))
    n_out = inst.size // step
    if n_out == 0:
        return HRSeries(np.array([]), out_fs)
    trimmed = inst[: n_out * step].reshape(n_out, step)
    return HRSeries(np.nanmean(trimmed, axis=1), out_fs)


if __name__ == "__main__":
    print(f"{N_FEATURES} features")
    for i, n in enumerate(FEATURE_NAMES):
        print(f"{i:3d}  {n}")