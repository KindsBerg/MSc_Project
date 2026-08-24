"""
features.py -- the one shared feature module. used by both the WESAD training
side and the live host, so it's the only place a feature actually gets
computed. same columns, same order, every time -- feature_vector() checks
against FEATURE_NAMES on every call so if something drifts it errors instead
of quietly messing up the model.

stuff I'm NOT using, and why:
  no HRV     needs beat-to-beat intervals, but the SEN0344 firmware only
             gives a computed HR number, no raw waveform access. hr_sd is
             just variability of the HR trend, NOT hrv.
  no gyro    E4 (what recorded WESAD) doesn't have one, so can't use it.
  no SpO2    same FIFO access problem as HRV.
  no temp    WESAD's TEMP is a skin thermistor, the LM75BD on my board reads
             board temp (mostly self heating) -- different thing entirely.
             dropping it only cost -0.004 balanced acc, well inside the fold
             noise, and it was correlating with elapsed time anyway.

windows:
  EDA   60s, matches WESAD's own window
  HR    40s -- SEN0344 only updates HR every 4s so need a wide window to get
        enough samples for sd/slope to mean anything
  IMU   15s, unchanged, accel is fast enough it doesn't need the wide window

HR and IMU windows used to be one constant (WIN_SHORT_S) before I actually
measured the SEN0344's real update rate on the bench.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal as sps
from scipy.ndimage import uniform_filter1d

# --------------------------------------------------------------------------
# config -- change stuff here, nowhere else
# --------------------------------------------------------------------------

WIN_EDA_S = 60.0          # EDA window, matches WESAD
WIN_SHORT_HR_S = 40.0     # HR sub-window -- ~10 samples at SEN0344's 0.25Hz
WIN_SHORT_IMU_S = 15.0    # IMU sub-window -- unchanged
WIN_STEP_S = 5.0          # slide between windows
WINDOW_PURITY = 0.9       # min fraction of agreeing labels to keep a window

EDA_FS = 4.0   # Hz, WESAD wrist EDA + target rate for my AFE
ACC_FS = 32.0  # Hz, E4 rate -- live IMU gets decimated down to match

# below this an SCR amplitude is just noise
SCR_MIN_AMP_US = 0.01

# motion gate: sd of 3D accel magnitude (g) above this = window is "moving".
# still provisional, need to retune against real worn data
MOTION_STD_THRESHOLD_G = 0.05

# live samples come in at 104Hz (library default), need to decimate to 32Hz
# before features.py sees them
IMU_TARGET_FS_HZ = 32.0
# matches E4's clipping range (LSM6DS3TR-C default is actually +/-4g)
IMU_RANGE_G = 2.0
# E4's quantisation step -- ~250x coarser than the LSM6DS3TR-C's real
# resolution, see quantise_to_e4_grid() in live_host.py
E4_ACC_LSB_G = 1.0 / 64.0
EMULATE_E4_QUANTISATION = True

# bump this by hand whenever a change here changes cached feature VALUES,
# even if FEATURE_NAMES itself didn't change -- forces a cache rebuild
# instead of silently reading stale data
FEATURE_PIPELINE_VERSION = 2

# SEN0344's real HR update rate, measured off the vendor library
SEN0344_HR_FS_HZ = 0.25

# smoothing applied to instantaneous HR before decimating down to
# SEN0344_HR_FS_HZ. kept separate from the 4s decimation on purpose so I can
# tune them independently. 8s is a starting guess (2x device interval), not
# bench measured
AVERAGING_WINDOW_SECONDS = 8.0

# band for the accel peak-frequency feature (human movement range)
ACC_BAND_HZ = (0.3, 10.0)

# cvxEDA sometimes fails to converge and falls back to a median filter --
# want that fallback rate for the report, so it's counted here instead of
# just getting lost. call reset_stats() before a batch run.
STATS = {
    "eda_windows": 0,       # EDA windows processed
    "cvxeda_ok": 0,         # windows where cvxEDA actually converged
    "cvxeda_fallback": 0,   # windows that used the median-filter fallback
    "cvxeda_error": None,   # first fallback reason, for reference
    "scr_windows": 0,       # windows run through SCR peak detection
    "scr_fallback": 0,      # windows where neurokit2 peaks failed -> scipy fallback
    "scr_error": None,      # first SCR fallback reason
}


def reset_stats() -> None:
    for k in STATS:  # zero everything out, clear error strings
        STATS[k] = None if k.endswith("_error") else 0


# --------------------------------------------------------------------------
# feature list -- the contract between training and inference
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

# no hr_min/max/range -- with only ~10 samples per window those just pick up
# whichever beat happened to land in the window, not anything stable
HR_FEATURES = [
    "hr_mean",
    "hr_sd",
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

CROSS_FEATURES = [
    "hr_delta_x_still",   # hr_baseline_delta * (1 - motion_fraction)
    "eda_range_gated",    # eda_range * (1 - motion_fraction)
]

FEATURE_NAMES: list[str] = EDA_FEATURES + HR_FEATURES + IMU_FEATURES + CROSS_FEATURES  # 33 total, this order matters
N_FEATURES = len(FEATURE_NAMES)

assert len(set(FEATURE_NAMES)) == N_FEATURES, "duplicate name in FEATURE_NAMES"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _slope(x: np.ndarray, fs: float, t: np.ndarray | None = None) -> float:
    """least squares slope, units per second. NaN if under 2 samples.

    pass `t` if some samples already got dropped by a filter -- rebuilding a
    plain arange(n)/fs axis would compress the gap and bias the slope.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        return np.nan  # can't get a slope from 1 point
    t = np.arange(n, dtype=np.float64) / fs if t is None else np.asarray(t, dtype=np.float64)  # time axis in seconds
    return float(np.polyfit(t, x, 1)[0])  # slope of the best fit line



def _abs_integral(x: np.ndarray, fs: float) -> float:
    """integral of |x| over the window"""
    x = np.asarray(x, dtype=np.float64)  # evenly spaced samples so this is just a scaled sum
    return float(np.sum(np.abs(x)) / fs) if x.size else np.nan  # sum(|x|) * dt


def _peak_freq(x: np.ndarray, fs: float, band=ACC_BAND_HZ) -> float:
    """dominant frequency in the band, via periodogram. NaN if window's too short."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8:
        return np.nan  # not enough samples for a real spectrum
    x = x - x.mean()  # remove DC so 0Hz can't win
    if not np.any(x):
        return 0.0
    f, p = sps.periodogram(x, fs=fs, scaling="density")  # f = freq bins, p = power
    sel = (f >= band[0]) & (f <= band[1])  # only look in the movement band
    return float(f[sel][int(np.argmax(p[sel]))]) if sel.any() else np.nan  # freq with the most power


def _nan_block(names: list[str]) -> dict:
    return {k: np.nan for k in names}  # "no usable data" row

# --------------------------------------------------------------------------
# EDA block -- 60s window
# --------------------------------------------------------------------------


def _decompose_eda(eda: np.ndarray, fs: float):
    """splits EDA into tonic (SCL) and phasic (SCR). returns (tonic, phasic, used_cvxeda).

    tries cvxEDA first (the proper convex optimisation decomposition), falls
    back to a median filter if neurokit2 is missing or cvxEDA won't converge.
    """
    try:
        import neurokit2 as nk

        df = nk.eda_phasic(eda, sampling_rate=fs, method="cvxeda")
        return df["EDA_Tonic"].to_numpy(), df["EDA_Phasic"].to_numpy(), True
    except Exception as e:  # noqa: BLE001 -- any failure just falls back
        # only keep the first reason -- if EVERY window falls back that's
        # cvxopt missing, which is a different problem than occasional
        # non-convergence
        if STATS["cvxeda_error"] is None:
            STATS["cvxeda_error"] = f"{type(e).__name__}: {e}"
        k = max(int(fs * 8) | 1, 3)  # odd kernel, wide enough to pass SCL through
        tonic = (
            np.full_like(eda, np.median(eda)) if eda.size < k
            else sps.medfilt(eda, kernel_size=k)
        )
        return tonic, eda - tonic, False


def _find_scrs(phasic: np.ndarray, fs: float):
    """finds SCR peaks. returns (amplitudes, rise_times_s, durations_s)."""
    STATS["scr_windows"] += 1
    try:
        import neurokit2 as nk

        _, info = nk.eda_peaks(phasic, sampling_rate=fs, method="neurokit")
        amp_raw = np.asarray(info.get("SCR_Amplitude", []), dtype=np.float64)
        rise_raw = np.asarray(info.get("SCR_RiseTime", []), dtype=np.float64)
        rec_raw = np.asarray(info.get("SCR_RecoveryTime", []), dtype=np.float64)

        # keep is built off the raw arrays and applied to all three the same
        # way so amp/rise/rec all stay lined up to the same peaks
        keep = np.isfinite(amp_raw) & (amp_raw >= SCR_MIN_AMP_US)
        amp = amp_raw[keep]
        rise = rise_raw[keep] if rise_raw.size == amp_raw.size else np.full(keep.sum(), np.nan)
        rec = rec_raw[keep] if rec_raw.size == amp_raw.size else np.full(keep.sum(), np.nan)

        # recovery is often NaN if the window cuts off the tail -- just use
        # rise time alone then
        dur = np.where(np.isfinite(rec), rise + rec, rise)
        return amp, rise, dur
    except Exception as e:  # noqa: BLE001
        if STATS["scr_error"] is None:
            STATS["scr_error"] = f"{type(e).__name__}: {e}"
        STATS["scr_fallback"] += 1
        pk, props = sps.find_peaks(phasic, prominence=SCR_MIN_AMP_US, width=1)
        amp = np.asarray(props.get("prominences", []), dtype=np.float64)
        widths = np.asarray(props.get("widths", []), dtype=np.float64) / fs
        return amp, widths * 0.5, widths


def eda_features(eda: np.ndarray, fs: float = EDA_FS) -> dict:
    """the 10 EDA features for one 60s window. input in microsiemens."""
    eda = np.asarray(eda, dtype=np.float64).reshape(-1)  # flatten to float64
    if eda.size < int(fs * 5):  # under 5s isn't enough to work with
        return _nan_block(EDA_FEATURES)

    tonic, phasic, used_cvxeda = _decompose_eda(eda, fs)
    STATS["eda_windows"] += 1
    STATS["cvxeda_ok" if used_cvxeda else "cvxeda_fallback"] += 1
    amp, rise, dur = _find_scrs(phasic, fs)

    # neurokit2 gives NaN rise/recovery when an SCR's onset is before the
    # window starts. filtering those out keeps the column dense -- 0 makes
    # more sense than NaN for "no measurable rise here"
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
# HR block -- short window
# --------------------------------------------------------------------------


def hr_features(hr: np.ndarray, fs: float, baseline_hr: float | None = None) -> dict:
    """stats on the computed HR stream over one short window.

    hr           : bpm values, already at the device's update rate
    fs           : that update rate in Hz
    baseline_hr  : subject's resting HR, None -> delta is NaN

    no HRV in here, see module docstring for why.
    """
    hr = np.asarray(hr, dtype=np.float64).reshape(-1)  # flat float64 bpm series
    valid = np.isfinite(hr) & (hr > 20.0) & (hr < 220.0)  # drop anything physiologically impossible
    t = np.arange(hr.size, dtype=np.float64) / fs  # build time axis BEFORE filtering so gaps stay real
    t, hr = t[valid], hr[valid]  # keep t and hr lined up through the filter
    if hr.size == 0:
        return _nan_block(HR_FEATURES)  # nothing usable in this window

    mean = float(np.mean(hr))
    return {
        "hr_mean": mean,
        "hr_sd": float(np.std(hr)) if hr.size > 1 else 0.0,
        "hr_slope": _slope(hr, fs, t=t),
        "hr_baseline_delta": mean - float(baseline_hr) if baseline_hr is not None else np.nan,
    }


# --------------------------------------------------------------------------
# IMU block -- short window, accel only
# --------------------------------------------------------------------------


def imu_features(acc: np.ndarray, fs: float = ACC_FS) -> dict:
    """accel features. `acc` is (N, 3) in g. no gyro, by design."""
    acc = np.asarray(acc, dtype=np.float64)
    if acc.ndim != 2 or acc.shape[1] != 3 or acc.shape[0] < 4:
        return _nan_block(IMU_FEATURES)

    mag = np.linalg.norm(acc, axis=1)  # 3D magnitude per sample
    out: dict = {}
    for i, ax in enumerate("xyz"):
        out[f"acc_{ax}_mean"] = float(np.mean(acc[:, i]))
        out[f"acc_{ax}_sd"] = float(np.std(acc[:, i]))
        out[f"acc_{ax}_absint"] = _abs_integral(acc[:, i] - np.mean(acc[:, i]), fs)  # movement amount
        out[f"acc_{ax}_peakfreq"] = _peak_freq(acc[:, i], fs)

    out["acc_mag_mean"] = float(np.mean(mag))
    out["acc_mag_sd"] = float(np.std(mag))
    out["acc_mag_absint"] = _abs_integral(mag - np.mean(mag), fs)

    # in a single short window this is just a binary flag / its float form.
    # once these get aggregated into the 60s vector, motion_fraction becomes
    # the share of sub-windows that were flagged
    flag = 1.0 if out["acc_mag_sd"] > MOTION_STD_THRESHOLD_G else 0.0
    out["motion_flag"] = flag
    out["motion_fraction"] = flag
    return out


# --------------------------------------------------------------------------
# putting it together
# --------------------------------------------------------------------------


def aggregate_short_windows(blocks: list[dict], names: list[str]) -> dict:
    """collapses a bunch of short-window dicts into one 60s summary.

    default rule is mean, keeps things simple
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

    # motion_fraction should be the SHARE flagged, not the average magnitude,
    # so recompute it separately to be sure
    if "motion_flag" in names:
        flags = np.array([b.get("motion_flag", np.nan) for b in blocks], dtype=np.float64)
        if not np.all(np.isnan(flags)):
            frac = float(np.nanmean(flags))
            out["motion_fraction"] = frac
            out["motion_flag"] = 1.0 if frac > 0.5 else 0.0
    return out


def cross_features(merged: dict, hr_span_motion_fraction: float | None = None) -> dict:
    """interaction terms so the model can discount motion-driven change.

    hr_delta_x_still needs a motion_fraction from the SAME span as
    hr_baseline_delta -- since the HR (40s) and IMU (15s) windows don't match
    anymore, the caller has to pass in the motion_fraction computed just over
    the HR span. falls back to merged's own value if not given.

    eda_range_gated doesn't need this -- EDA and its gate both live on the
    full 60s window already.
    """
    frac_eda = merged.get("motion_fraction", np.nan)
    still_eda = 1.0 - frac_eda if np.isfinite(frac_eda) else np.nan

    frac_hr = frac_eda if hr_span_motion_fraction is None else hr_span_motion_fraction
    still_hr = 1.0 - frac_hr if np.isfinite(frac_hr) else np.nan

    hr_d = merged.get("hr_baseline_delta", np.nan)
    eda_r = merged.get("eda_range", np.nan)
    return {
        "hr_delta_x_still": hr_d * still_hr if np.isfinite(hr_d) else np.nan,
        "eda_range_gated": eda_r * still_eda if np.isfinite(eda_r) else np.nan,
    }


def feature_vector(
    eda_win: np.ndarray,
    hr_short_blocks: list[dict],
    imu_short_blocks: list[dict],
    imu_short_blocks_hr_span: list[dict] | None = None,
    eda_fs: float = EDA_FS,
) -> dict:
    """builds one full feature row at an EDA-window close.

    hr_short_blocks / imu_short_blocks are the buffered short-window outputs
    from inside this 60s window. baseline HR already got applied inside
    hr_features().

    imu_short_blocks_hr_span is just the imu blocks that overlap the HR
    block's span -- only used to scope hr_delta_x_still right, see
    cross_features(). defaults to imu_short_blocks if not given.

    returns a dict with exactly FEATURE_NAMES as keys, in order.
    """
    row: dict = {}
    row.update(eda_features(eda_win, eda_fs))
    row.update(aggregate_short_windows(hr_short_blocks, HR_FEATURES))
    row.update(aggregate_short_windows(imu_short_blocks, IMU_FEATURES))

    hr_span_blocks = imu_short_blocks if imu_short_blocks_hr_span is None else imu_short_blocks_hr_span
    hr_span_motion_fraction = aggregate_short_windows(hr_span_blocks, IMU_FEATURES).get(
        "motion_fraction", np.nan
    )
    row.update(cross_features(row, hr_span_motion_fraction))

    missing = set(FEATURE_NAMES) - set(row)
    extra = set(row) - set(FEATURE_NAMES)
    if missing or extra:
        raise RuntimeError(
            f"feature contract violated — missing={sorted(missing)} extra={sorted(extra)}"
        )
    return {k: row[k] for k in FEATURE_NAMES}


# --------------------------------------------------------------------------
# WESAD-side adapter -- training only, not part of the shared contract
# --------------------------------------------------------------------------


@dataclass
class HRSeries:
    values: np.ndarray
    fs: float


def bvp_to_hr(
    bvp: np.ndarray,
    fs: float = 64.0,
    out_fs: float = SEN0344_HR_FS_HZ,
    avg_window_s: float = AVERAGING_WINDOW_SECONDS,
) -> HRSeries:
    """turns WESAD's raw BVP into an HR series, emulating what the SEN0344 outputs.

    training side only. WESAD gives a 64Hz waveform, my hardware gives a
    computed HR number at a low rate -- so reduce the waveform down to a
    number here first, otherwise training and inference wouldn't match.

    out_fs = 0.25Hz is the SEN0344's real measured update rate (from the
    DFRobot arduino example, delay(4000)ms). not a guess anymore, bench
    measured.

    smooth first, THEN decimate -- avg_window_s knocks out the beat-to-beat
    detail the device would never report, before dropping down to out_fs.
    doing it the other way round would alias that detail back in and inflate
    hr_sd.

    worth noting for the report: this is a real domain gap. neurokit2's beat
    detector and the SEN0344's onboard estimator are different algorithms
    with different latency/motion handling, so "mean HR" isn't really the
    same measurement on both sides even though it's the same quantity.
    """
    import neurokit2 as nk

    bvp = np.asarray(bvp, dtype=np.float64).reshape(-1)
    sig, info = nk.ppg_process(bvp, sampling_rate=fs)
    inst = sig["PPG_Rate"].to_numpy()

    # rolling mean at the native rate. mode="nearest" holds the edge value
    # instead of zero-padding so the start/end don't get biased toward zero
    win = max(int(round(avg_window_s * fs)), 1)
    smoothed = uniform_filter1d(inst, size=win, mode="nearest")

    # take one smoothed sample per output interval, from the END of the
    # interval -- matches "device holds its last computed value"
    step = int(round(fs / out_fs))
    n_out = smoothed.size // step
    if n_out == 0:
        return HRSeries(np.array([]), out_fs)
    return HRSeries(smoothed[step - 1 : n_out * step : step], out_fs)


if __name__ == "__main__":
    # just prints the frozen feature list, index + name in order
    print(f"{N_FEATURES} features")
    for i, n in enumerate(FEATURE_NAMES):
        print(f"{i:3d}  {n}")
