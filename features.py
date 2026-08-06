"""
features.py — the shared feature module.

Imported by BOTH the WESAD training pipeline and the live host. It is the only
place any feature is computed. Same columns, same order, same per-stream
window, on both sides. FEATURE_NAMES is the single source of truth for that
order and feature_vector() checks against it on every call, so a drift raises
instead of silently misaligning the model against its input.

Exclusions, and why (see WESAD_NOTES §2.2):

  NO HRV      RMSSD/SDNN/pNN50/LF/HF all need beat-to-beat intervals. The
              SEN0344 v2.0 firmware does not expose the MAX30102 FIFO, so only
              a computed HR number is available. The HR block is statistics on
              that number. hr_sd is the variability of the HR trend and must
              never be called HRV.
  NO GYRO     The LSM6DS3TR-C has one; the Empatica E4 that recorded WESAD
              does not. Features come from the intersection of the two
              sensors.
  NO SpO2     Same data-access reason as HRV.
  NO TEMP     WESAD's TEMP is a skin thermistor. The LM75BD on this PCB reads
              board temperature dominated by self-heating — a different
              instrument measuring a different thing. Removing the block cost
              -0.004 binary balanced accuracy (inside the +/-0.127 fold
              spread), and its absolute features correlated with elapsed
              session time at |r| ~ 0.65. Device thermal telemetry is handled
              on the ESP32 and never enters the model. (Two more thermal
              sources exist on the live path — SEN0344 register 0x14 and the
              LSM6DS3TR-C die sensor — both diagnostics only, same reason.)

Windows:
  EDA        60 s, matching WESAD's own physiological window.
  HR         WIN_SHORT_HR_S = 40 s. Measured off the SEN0344 vendor library:
             the sensor updates its computed HR once every 4 s, so a 40 s
             window holds ~10 samples — enough for sd/slope to mean something.
             A 15 s window at that cadence would hold 3-4.
  IMU        WIN_SHORT_IMU_S = 15 s, unchanged. The accelerometer's native
             rate is high regardless of PPG cadence, so it keeps the
             finer-grained window.

Both were PROVISIONAL as one constant (WIN_SHORT_S) pending a bench
measurement of the SEN0344's real HR update cadence; that measurement is now
in and the constant has been split accordingly (see the vendor comment on
bvp_to_hr's out_fs below).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal as sps

# --------------------------------------------------------------------------
# Configuration — change here, nowhere else.
# --------------------------------------------------------------------------

WIN_EDA_S = 60.0          # EDA window, matches WESAD
WIN_SHORT_HR_S = 40.0     # HR sub-window — ~10 samples at the SEN0344's 0.25 Hz cadence
WIN_SHORT_IMU_S = 15.0    # IMU sub-window — unchanged, IMU rate is high either way
WIN_STEP_S = 5.0          # slide between consecutive EDA windows
WINDOW_PURITY = 0.9       # min fraction of agreeing labels for a valid window

EDA_FS = 4.0   # Hz, WESAD wrist EDA and the target for the live AFE
ACC_FS = 32.0  # Hz, E4; live LSM6DS3TR-C decimated down to match, see IMU_TARGET_FS_HZ

# SCR amplitude below this is noise, not a response.
SCR_MIN_AMP_US = 0.01

# IMU motion gate: sd of 3-D acceleration magnitude, in g, above which a
# window is flagged motion-contaminated. Provisional — retune against your own
# worn recordings and report the value used.
MOTION_STD_THRESHOLD_G = 0.05

# LSM6DS3TR-C ingest: live samples arrive at the library's default 104 Hz ODR
# and must be decimated to WESAD's 32 Hz before imu_features() sees them.
IMU_TARGET_FS_HZ = 32.0
# Configured range, matching the E4's own clipping behaviour (decision, not a
# library default — the LSM6DS3TR-C defaults to +/-4 g).
IMU_RANGE_G = 2.0
# E4 quantisation step: 1/64 g, vs the LSM6DS3TR-C's 0.061 mg/LSB at +/-2 g —
# roughly 250x coarser. See quantise_to_e4_grid() in live_host.py.
E4_ACC_LSB_G = 1.0 / 64.0
EMULATE_E4_QUANTISATION = True

# SEN0344 computed-HR update cadence, measured off the vendor library (see
# bvp_to_hr below). Named here so callers can reference it explicitly instead
# of relying on bvp_to_hr's default.
SEN0344_HR_FS_HZ = 0.25

# Band for the accelerometer peak-frequency feature (human movement).
ACC_BAND_HZ = (0.3, 10.0)

# cvxEDA occasionally fails to converge on a short window and falls back to a
# median-filter decomposition. The fallback rate belongs in the report, so it
# is counted rather than lost. reset_stats() at the start of a batch run.
#STATS purpose is to define a dictionary that holds stats about EDA processing.
STATS = {
    "eda_windows": 0, # number of EDA windows processed
    "cvxeda_ok": 0, # cvxeda stands for convex optimization EDA. ok means they succeeded. convex optimiation is a method for decomposing EDA signals into tonic and phasic components.
    "cvxeda_fallback": 0, # convex optimization eda num failed
    "cvxeda_error": None,   # first fallback reason only
    "scr_windows": 0, # scr means skin conductance response, this holds the num of scr windows processed
    "scr_fallback": 0, #similar, scr fallback / failure count
    "scr_error": None, #this is scr error reason.
}


def reset_stats() -> None:
    for k in STATS:
        STATS[k] = None if k.endswith("_error") else 0


# --------------------------------------------------------------------------
# Feature registry — the contract between training and inference.
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

CROSS_FEATURES = [
    "hr_delta_x_still",   # hr_baseline_delta * (1 - motion_fraction)
    "eda_range_gated",    # eda_range * (1 - motion_fraction)
]

FEATURE_NAMES: list[str] = EDA_FEATURES + HR_FEATURES + IMU_FEATURES + CROSS_FEATURES
N_FEATURES = len(FEATURE_NAMES)

assert len(set(FEATURE_NAMES)) == N_FEATURES, "duplicate name in FEATURE_NAMES"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _slope(x: np.ndarray, fs: float, t: np.ndarray | None = None) -> float:
    """Least-squares slope in units per second. NaN under two samples.

    Pass `t` (seconds) whenever `x` has already had samples dropped by a
    validity filter. Rebuilding a contiguous arange(n)/fs axis from a filtered
    array compresses the gap the dropped samples left and biases the slope.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        return np.nan
    t = np.arange(n, dtype=np.float64) / fs if t is None else np.asarray(t, dtype=np.float64)
    return float(np.polyfit(t, x, 1)[0])


def _abs_integral(x: np.ndarray, fs: float) -> float:
    """Integral of |x| over the window, in signal-units x seconds."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(np.abs(x)) / fs) if x.size else np.nan


def _peak_freq(x: np.ndarray, fs: float, band=ACC_BAND_HZ) -> float:
    """Dominant in-band frequency via periodogram. NaN if the window is short."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8:
        return np.nan
    x = x - x.mean()
    if not np.any(x):
        return 0.0
    f, p = sps.periodogram(x, fs=fs, scaling="density")
    sel = (f >= band[0]) & (f <= band[1])
    return float(f[sel][int(np.argmax(p[sel]))]) if sel.any() else np.nan


def _nan_block(names: list[str]) -> dict:
    return {k: np.nan for k in names}


# --------------------------------------------------------------------------
# EDA block — 60 s window
# --------------------------------------------------------------------------


def _decompose_eda(eda: np.ndarray, fs: float):
    """Split EDA into tonic (SCL) and phasic (SCR). Returns (tonic, phasic, used_cvxeda).

    Prefers NeuroKit2's cvxEDA (Greco et al.), the convex-optimisation
    replacement for WESAD's Choi decomposition. Falls back to a median-filter
    tonic estimate if NeuroKit2 is missing or cvxEDA fails to converge.
    """
    try:
        import neurokit2 as nk

        df = nk.eda_phasic(eda, sampling_rate=fs, method="cvxeda")
        return df["EDA_Tonic"].to_numpy(), df["EDA_Phasic"].to_numpy(), True
    except Exception as e:  # noqa: BLE001 — any failure falls back, by design
        # First reason only. A 100% fallback rate means cvxEDA never ran at all
        # (usually a missing cvxopt), which is a different problem from
        # occasional non-convergence and must not look like one.
        if STATS["cvxeda_error"] is None:
            STATS["cvxeda_error"] = f"{type(e).__name__}: {e}"
        k = max(int(fs * 8) | 1, 3)  # odd kernel, wide enough to pass SCL
        tonic = (
            np.full_like(eda, np.median(eda)) if eda.size < k
            else sps.medfilt(eda, kernel_size=k)
        )
        return tonic, eda - tonic, False


def _find_scrs(phasic: np.ndarray, fs: float):
    """Detect SCRs. Returns (amplitudes, rise_times_s, durations_s)."""
    STATS["scr_windows"] += 1
    try:
        import neurokit2 as nk

        _, info = nk.eda_peaks(phasic, sampling_rate=fs, method="neurokit")
        amp_raw = np.asarray(info.get("SCR_Amplitude", []), dtype=np.float64)
        rise_raw = np.asarray(info.get("SCR_RiseTime", []), dtype=np.float64)
        rec_raw = np.asarray(info.get("SCR_RecoveryTime", []), dtype=np.float64)

        # `keep` is built against the RAW arrays and applied to all three
        # identically, so amp/rise/rec stay aligned to the same peaks.
        keep = np.isfinite(amp_raw) & (amp_raw >= SCR_MIN_AMP_US)
        amp = amp_raw[keep]
        rise = rise_raw[keep] if rise_raw.size == amp_raw.size else np.full(keep.sum(), np.nan)
        rec = rec_raw[keep] if rec_raw.size == amp_raw.size else np.full(keep.sum(), np.nan)

        # Recovery is often NaN when the window truncates the tail; duration
        # falls back to rise time alone there.
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
    """Ten EDA features over one 60 s window. Input in microsiemens."""
    eda = np.asarray(eda, dtype=np.float64).reshape(-1) # turns2d np array to float-64 1d arrray
    if eda.size < int(fs * 5):  # under 5 s is not usable
        return _nan_block(EDA_FEATURES)

    tonic, phasic, used_cvxeda = _decompose_eda(eda, fs)
    STATS["eda_windows"] += 1
    STATS["cvxeda_ok" if used_cvxeda else "cvxeda_fallback"] += 1
    amp, rise, dur = _find_scrs(phasic, fs)

    # NeuroKit2 returns NaN rise/recovery for SCRs whose onset precedes the
    # window. Filtering to finite values keeps the column dense; zero is the
    # right fallback, since "no measurable rise here" is nearer 0 than missing.
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

    hr           : HR values in bpm, already at the device's update rate.
    fs           : that update rate in Hz (WESAD side is resampled to match).
    baseline_hr  : personal resting HR; None yields NaN for the delta.

    Contains no HRV, by design. See module docstring.
    """
    hr = np.asarray(hr, dtype=np.float64).reshape(-1)
    valid = np.isfinite(hr) & (hr > 20.0) & (hr < 220.0)  # drop implausible
    t = np.arange(hr.size, dtype=np.float64) / fs
    t, hr = t[valid], hr[valid]
    if hr.size == 0:
        return _nan_block(HR_FEATURES)

    mean = float(np.mean(hr))
    return {
        "hr_mean": mean,
        "hr_sd": float(np.std(hr)) if hr.size > 1 else 0.0,
        "hr_min": float(np.min(hr)),
        "hr_max": float(np.max(hr)),
        "hr_range": float(np.ptp(hr)),
        "hr_slope": _slope(hr, fs, t=t),
        "hr_baseline_delta": mean - float(baseline_hr) if baseline_hr is not None else np.nan,
    }


# --------------------------------------------------------------------------
# IMU block — short window, accelerometer only
# --------------------------------------------------------------------------


def imu_features(acc: np.ndarray, fs: float = ACC_FS) -> dict:
    """Accelerometer features. `acc` is (N, 3) in g. Gyro excluded by design."""
    acc = np.asarray(acc, dtype=np.float64)
    if acc.ndim != 2 or acc.shape[1] != 3 or acc.shape[0] < 4:
        return _nan_block(IMU_FEATURES)

    mag = np.linalg.norm(acc, axis=1) # mag is the magnitude of acc vector, sqrt root of (x^2 + y^2 + z^2).
    out: dict = {}
    for i, ax in enumerate("xyz"):
        out[f"acc_{ax}_mean"] = float(np.mean(acc[:, i]))
        out[f"acc_{ax}_sd"] = float(np.std(acc[:, i]))
        out[f"acc_{ax}_absint"] = _abs_integral(acc[:, i] - np.mean(acc[:, i]), fs)
        out[f"acc_{ax}_peakfreq"] = _peak_freq(acc[:, i], fs)

    out["acc_mag_mean"] = float(np.mean(mag))
    out["acc_mag_sd"] = float(np.std(mag))
    out["acc_mag_absint"] = _abs_integral(mag - np.mean(mag), fs)

    # In one short window the flag is binary and the fraction is its float
    # form. Once short windows are aggregated into a 60 s vector,
    # motion_fraction becomes the share of flagged sub-windows.
    flag = 1.0 if out["acc_mag_sd"] > MOTION_STD_THRESHOLD_G else 0.0
    out["motion_flag"] = flag
    out["motion_fraction"] = flag
    return out


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def aggregate_short_windows(blocks: list[dict], names: list[str]) -> dict:
    """Collapse buffered short-window dicts into one 60 s summary.

    The rule per feature preserves its meaning: extremes take extremes,
    everything else takes the mean.
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
    # magnitude — recompute explicitly so the semantics are unambiguous.
    if "motion_flag" in names:
        flags = np.array([b.get("motion_flag", np.nan) for b in blocks], dtype=np.float64)
        if not np.all(np.isnan(flags)):
            frac = float(np.nanmean(flags))
            out["motion_fraction"] = frac
            out["motion_flag"] = 1.0 if frac > 0.5 else 0.0
    return out


def cross_features(merged: dict, hr_span_motion_fraction: float | None = None) -> dict:
    """Interaction terms letting the model discount motion-driven change.

    hr_delta_x_still must be paired with a motion_fraction computed over the
    SAME span as hr_baseline_delta. Since WIN_SHORT_HR_S (40 s) and
    WIN_SHORT_IMU_S (15 s) diverged, `merged["motion_fraction"]` — aggregated
    over the full 60 s EDA window — is no longer that span; the caller must
    pass the motion_fraction aggregated over just the 40 s HR sub-window(s)
    as `hr_span_motion_fraction`. Falls back to `merged`'s value when omitted,
    for callers that don't need the distinction (e.g. it's already scoped).

    eda_range_gated is unaffected — EDA and its motion gate both still live on
    the full 60 s window, so it keeps using `merged["motion_fraction"]`.
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
    """Assemble one complete feature row at an EDA-window close.

    hr_short_blocks / imu_short_blocks are the buffered outputs of
    hr_features() / imu_features() over the short windows inside this 60 s
    window. Baseline HR is applied inside hr_features() at buffer time.

    imu_short_blocks_hr_span is the subset of imu_short_blocks whose sub-
    windows fall inside the same span as hr_short_blocks (WIN_SHORT_HR_S),
    used only to scope the hr_delta_x_still cross-term correctly — see
    cross_features(). Defaults to imu_short_blocks when omitted.

    Returns a dict whose keys are exactly FEATURE_NAMES, in that order.
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
# WESAD-side adapter — training only, not part of the shared contract.
# --------------------------------------------------------------------------


@dataclass
class HRSeries:
    values: np.ndarray
    fs: float


def bvp_to_hr(bvp: np.ndarray, fs: float = 64.0, out_fs: float = SEN0344_HR_FS_HZ) -> HRSeries:
    """Derive an HR series from WESAD's raw BVP, emulating the SEN0344 output.

    Training side only. WESAD gives a 64 Hz BVP waveform; the hardware gives a
    computed HR number at a low update rate. Training on the waveform and
    inferring on the number would not transfer, so the waveform is reduced to
    a number here first.

    `out_fs` is the emulated update rate: 0.25 Hz, the SEN0344's measured
    cadence per the DFRobot_BloodOxygen_S Arduino example (`delay(4000)` — one
    new HR value every 4 s, with the device holding its last value between
    updates). No longer provisional; this is the bench-measured figure.

    For the report: this is a real domain-gap contributor. NeuroKit2's beat
    detector and DFRobot's undocumented on-board estimator are different
    algorithms with different latency and motion rejection, so "mean HR" is
    the same quantity but not the same measurement on both sides.
    """
    import neurokit2 as nk

    bvp = np.asarray(bvp, dtype=np.float64).reshape(-1)
    sig, info = nk.ppg_process(bvp, sampling_rate=fs)
    inst = sig["PPG_Rate"].to_numpy()

    # Average the per-sample instantaneous rate within each output interval —
    # closer to an on-board estimator than naive subsampling.
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