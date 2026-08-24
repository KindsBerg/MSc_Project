"""
validate_migration.py -- gate for the SEN0344/LSM6DS3TR-C hardware migration.

checks that the measured-hardware constants (WIN_SHORT_HR_S=40, out_fs=0.25,
the LSM6DS3TR-C unit/rate constants) are actually in place, that no stale
pre-migration cache can sneak back in, that the live ingest path converts
and rejects units correctly, that a 40s HR window actually holds enough
samples to be worth computing sd/slope on, that the two independent
standardisation implementations agree, and that artefact provenance
survives an export/reload round trip and is actually checked, not just
carried along for the ride.

    python validate_migration.py

prints one line per check plus a pass/fail summary. exits non-zero on any
failure.
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
import live_host
import export_model
from train_model import TIME_COL, Z_CLIP

CACHE_DIR = Path("cache")

Result = tuple[str, bool, str]


# --------------------------------------------------------------------------
# 1. constants
# --------------------------------------------------------------------------


def check_constants() -> Result:
    """S2 -- a partial migration should fail fast, not silently mix old and new values"""
    problems = []

    if hasattr(F, "WIN_SHORT_S"):
        problems.append("WIN_SHORT_S still exists — must be removed, not aliased")

    expected = {
        "WIN_SHORT_HR_S": 40.0,
        "WIN_SHORT_IMU_S": 15.0,
        "IMU_TARGET_FS_HZ": 32.0,
        "IMU_RANGE_G": 2.0,
        "E4_ACC_LSB_G": 1.0 / 64.0,
        "EMULATE_E4_QUANTISATION": True,
        "MOTION_STD_THRESHOLD_G": 0.05,
        "SEN0344_HR_FS_HZ": 0.25,
    }
    for name, want in expected.items():
        got = getattr(F, name, None)
        if got is None:
            problems.append(f"{name} is missing")
        elif got != want:
            problems.append(f"{name} = {got!r}, expected {want!r}")

    default_out_fs = inspect.signature(F.bvp_to_hr).parameters["out_fs"].default
    if default_out_fs != 0.25:
        problems.append(f"bvp_to_hr(out_fs=...) defaults to {default_out_fs!r}, expected 0.25")

    if problems:
        return "constants (S2)", False, "; ".join(problems)
    return "constants (S2)", True, "all measured-hardware constants in place"


# --------------------------------------------------------------------------
# 2. cache invalidation
# --------------------------------------------------------------------------


def check_cache_freshness() -> Result:
    """S7.2 -- parquet caches built under the old windows should never get
    silently reused. refuses (fails the check) rather than deleting anything itself."""
    if not CACHE_DIR.is_dir():
        return "cache freshness (S7.2)", True, "no cache/ directory — nothing to invalidate"

    meta_files = sorted(CACHE_DIR.glob("*.meta.json"))
    if not meta_files:
        return "cache freshness (S7.2)", True, "cache/ has no sidecars — nothing built yet"

    stale = []
    for p in meta_files:
        try:
            meta = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            stale.append(p.name)
            continue
        if (
            meta.get("win_short_hr_s") != F.WIN_SHORT_HR_S
            or meta.get("win_short_imu_s") != F.WIN_SHORT_IMU_S
            or meta.get("bvp_to_hr_out_fs") != F.SEN0344_HR_FS_HZ
        ):
            stale.append(p.name)

    if stale:
        shown = ", ".join(stale[:5]) + ("..." if len(stale) > 5 else "")
        return (
            "cache freshness (S7.2)",
            False,
            f"{len(stale)} stale cache file(s) built under the pre-migration "
            f"windows: {shown} — delete cache/ and rerun "
            "`python build_dataset.py --force --combine`",
        )
    return (
        "cache freshness (S7.2)",
        True,
        f"{len(meta_files)} cache sidecar(s) all match the live window constants",
    )


# --------------------------------------------------------------------------
# 3. unit regression at the live ingest boundary
# --------------------------------------------------------------------------


def check_unit_regression() -> Result:
    """S7.3 -- a still-hand signal in g must survive live ingest near 1.0,
    and the same signal mislabelled as m/s^2 must get rejected, not accepted."""
    native_fs = 104.0
    n = int(native_fs * 5)  # 5 s block
    rng = np.random.default_rng(0)
    still_hand_g = np.tile([0.0, 0.0, 1.0], (n, 1)) + rng.normal(0.0, 0.01, (n, 3))

    # correct case: sensor really does output m/s^2 -- feed the g signal
    # scaled up, exactly like real hardware would, through the full pipeline
    still_hand_ms2 = still_hand_g * live_host.SENSORS_GRAVITY_STANDARD
    acc_g_out = live_host.prepare_live_accel(
        still_hand_ms2, native_fs=native_fs, target_fs=F.IMU_TARGET_FS_HZ
    )
    mag = np.linalg.norm(acc_g_out, axis=1)
    if not np.isfinite(mag).all() or abs(float(mag.mean()) - 1.0) > 0.15:
        return (
            "unit regression — accept (S7.3)",
            False,
            f"mean|acc| after ingest = {mag.mean():.3f}, expected ~1.0 g",
        )

    # buggy case: g-valued numbers reach the pipeline mislabelled as m/s^2
    # (conversion step skipped upstream). the pipeline's own /9.80665 then
    # makes the result ~9.8x too small -- assert_accel_units_g needs to
    # catch that, not silently accept it
    acc_g_bug = live_host.prepare_live_accel(
        still_hand_g, native_fs=native_fs, target_fs=F.IMU_TARGET_FS_HZ
    )
    try:
        live_host.assert_accel_units_g(acc_g_bug)
    except ValueError:
        pass
    else:
        return (
            "unit regression — reject (S7.3)",
            False,
            "pipeline accepted g-valued input mislabelled as m/s^2 instead of rejecting it",
        )

    return (
        "unit regression (S7.3)",
        True,
        "still-hand signal round-trips to ~1.0 g; mislabelled units are rejected",
    )


# --------------------------------------------------------------------------
# 4. HR sample count at the new window
# --------------------------------------------------------------------------


def check_hr_sample_count() -> Result:
    """S7.4 -- a 40s HR window at 0.25Hz has to hold enough samples for
    sd/slope to actually mean something."""
    n_samples = F.WIN_SHORT_HR_S * F.SEN0344_HR_FS_HZ
    if n_samples < 9:
        return (
            "HR sample count (S7.4)",
            False,
            f"WIN_SHORT_HR_S={F.WIN_SHORT_HR_S}s at SEN0344_HR_FS_HZ="
            f"{F.SEN0344_HR_FS_HZ}Hz yields only {n_samples:.1f} samples (<9)",
        )
    return (
        "HR sample count (S7.4)",
        True,
        f"WIN_SHORT_HR_S={F.WIN_SHORT_HR_S}s at {F.SEN0344_HR_FS_HZ}Hz "
        f"yields {n_samples:.0f} samples",
    )


# --------------------------------------------------------------------------
# 5. standardisation agreement
# --------------------------------------------------------------------------


def _synthetic_feature_table(
    n_subjects: int = 3,
    n_windows: int = 90,
    seed: int = 0,
    nan_frac: float = 0.02,
    zero_ref_iqr_col: str | None = None,
    n_ref: int = 40,
) -> pd.DataFrame:
    """deliberately not a nice matrix: log-uniform per-column scales spanning
    six orders of magnitude (real columns span very different natural units
    -- eda_scl_mean in microsiemens vs acc_x_absint in g*s), a sprinkle of
    NaN (real rows do carry NaN, e.g. hr_baseline_delta pre-warm-up), and
    optionally one column forced to zero variance within the first n_ref
    rows of subject 0 so the ref-IQR fallback path actually gets exercised
    instead of silently never firing.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for si in range(n_subjects):
        sid = f"SYN{si}"
        # log-uniform in [1e-3, 1e3), signed
        log_scales = 10.0 ** rng.uniform(-3, 3, size=len(F.FEATURE_NAMES))
        signs = rng.choice([-1.0, 1.0], size=len(F.FEATURE_NAMES))
        centres = signs * rng.uniform(0.1, 10.0, size=len(F.FEATURE_NAMES)) * log_scales
        for wi in range(n_windows):
            vals = centres + rng.normal(0, 1, size=len(F.FEATURE_NAMES)) * log_scales
            row = dict(zip(F.FEATURE_NAMES, vals))
            row.update(subject=sid, t_start=float(wi * F.WIN_STEP_S))
            rows.append(row)
    df = pd.DataFrame(rows)

    nan_mask = rng.random(df[F.FEATURE_NAMES].shape) < nan_frac
    df.loc[:, F.FEATURE_NAMES] = df[F.FEATURE_NAMES].mask(nan_mask)

    if zero_ref_iqr_col is not None:
        idx = df.index[(df["subject"] == "SYN0")].to_numpy()[:n_ref]
        df.loc[idx, zero_ref_iqr_col] = 7.0  # constant across the whole reference block

    return df


def _bundle_for(cols, cohort_iqr, n_ref) -> export_model.StressModel:
    bundle = {
        "artefact_version": export_model.ARTEFACT_VERSION,
        "classifier": _DummyClassifier(),
        "feature_names": cols,
        "class_names": ["a", "b"],
        "cohort_iqr": cohort_iqr.to_dict(),
        "n_ref_windows": n_ref,
        "z_clip": Z_CLIP,
        "provenance": export_model.current_provenance(),
        "metadata": {"task": "binary", "model": "dummy"},
    }
    return export_model.StressModel(bundle)


def _max_diff_for_subject(df, sid, cols, n_ref, scaled_train) -> float:
    cohort_iqr = df[cols].quantile(0.75) - df[cols].quantile(0.25)
    model = _bundle_for(cols, cohort_iqr, n_ref)
    # keeps df's ORIGINAL index -- scaled_train is indexed on it too. an
    # earlier version of this check called .reset_index(drop=True) here,
    # which for any subject but the first in construction order silently
    # looked up the WRONG row out of scaled_train (only self-aligned by
    # coincidence for subject 0). that was a bug in this test, not the product.
    block = df[df["subject"] == sid].sort_values(TIME_COL)
    for r in block.head(n_ref)[cols].to_dict("records"):
        model.add_reference(r)
    if not model.ready:
        raise RuntimeError("StressModel never became ready")

    max_diff = 0.0
    for idx_label, row in block[cols].iterrows():
        z_live = model.transform(row.to_dict())[0]
        z_train = scaled_train.loc[idx_label, cols].to_numpy(dtype=np.float64)
        d = np.abs(z_live - z_train)
        d = d[np.isfinite(d)]  # both sides fillna(0) NaNs the same way, skip if already equal
        if d.size:
            max_diff = max(max_diff, float(np.max(d)))
    return max_diff


def check_standardisation_agreement() -> Result:
    """S7.5 -- training_standardise and StressModel.transform both drive
    compute_scale(). checks agreement on realistic data (wide per-column
    scales, injected NaN) AND on the zero-ref-IQR case that used to diverge
    (block-wide vs reference-only "wider" tier, measured at 19.47) before
    that tier got removed from compute_scale — both need to be exact now."""
    n_ref = 40
    cols = list(F.FEATURE_NAMES)

    # main case: realistic data, every column has nonzero reference IQR
    # (true for basically all continuous physiological features in
    # practice) -- this is the path production traffic actually takes
    df = _synthetic_feature_table(n_subjects=3, n_windows=90, n_ref=n_ref)
    cohort_iqr = df[cols].quantile(0.75) - df[cols].quantile(0.25)
    scaled_train = export_model.training_standardise(df, cols, cohort_iqr, n_ref)
    try:
        main_diffs = {
            sid: _max_diff_for_subject(df, sid, cols, n_ref, scaled_train)
            for sid in df["subject"].unique()
        }
    except RuntimeError as e:
        return "standardisation agreement (S7.5)", False, str(e)
    main_max = max(main_diffs.values())

    # former divergence probe: force zero IQR in one column's reference
    # window for SYN0 -- used to send training to the subject-wide IQR and
    # live to the reference-only IQR for that column. now both fall
    # through to the same cohort_iqr tier, so this needs to be exact too
    probe_col = cols[0]
    df_probe = _synthetic_feature_table(
        n_subjects=3, n_windows=90, n_ref=n_ref, zero_ref_iqr_col=probe_col
    )
    cohort_iqr_probe = df_probe[cols].quantile(0.75) - df_probe[cols].quantile(0.25)
    scaled_train_probe = export_model.training_standardise(df_probe, cols, cohort_iqr_probe, n_ref)
    try:
        probe_diff = _max_diff_for_subject(df_probe, "SYN0", cols, n_ref, scaled_train_probe)
    except RuntimeError as e:
        return "standardisation agreement (S7.5)", False, str(e)

    overall_max = max(main_max, probe_diff)
    if overall_max > 1e-8:
        return (
            "standardisation agreement (S7.5)",
            False,
            f"disagree by up to {overall_max:.3e} — realistic-data per-subject: "
            f"{ {k: round(v, 3) for k, v in main_diffs.items()} }, "
            f"zero-ref-IQR probe on {probe_col!r}: {probe_diff:.3e}",
        )
    return (
        "standardisation agreement (S7.5)",
        True,
        f"agree exactly on realistic data (max diff {main_max:.1e} across "
        f"{len(main_diffs)} subjects) and on the former zero-ref-IQR "
        f"divergence probe (max diff {probe_diff:.1e} on column {probe_col!r}, "
        f"was 19.47 before the wider tier was removed)",
    )


class _DummyClassifier:
    """minimal stand-in classifier -- only .predict gets exercised here"""

    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


# --------------------------------------------------------------------------
# 6. artefact provenance round trip
# --------------------------------------------------------------------------


def check_artefact_provenance_roundtrip() -> Result:
    """S7.6 -- export, reload, confirm every S6 provenance field survives
    AND is actually checked (a corrupted field has to raise on load)."""
    import joblib

    cols = list(F.FEATURE_NAMES)
    provenance = export_model.current_provenance()
    missing = [k for k in export_model.PROVENANCE_KEYS if k not in provenance]
    if missing:
        return (
            "artefact provenance round trip (S7.6)",
            False,
            f"current_provenance() is missing key(s): {missing}",
        )

    bundle = {
        "artefact_version": export_model.ARTEFACT_VERSION,
        "classifier": _DummyClassifier(),
        "feature_names": cols,
        "class_names": ["a", "b"],
        "cohort_iqr": {c: 1.0 for c in cols},
        "n_ref_windows": 5,
        "z_clip": Z_CLIP,
        "provenance": provenance,
        "metadata": {"task": "binary", "model": "dummy"},
    }

    with tempfile.TemporaryDirectory() as tmp:
        good_path = Path(tmp) / "roundtrip.joblib"
        joblib.dump(bundle, good_path)
        reloaded = export_model.StressModel.load(good_path)
        if reloaded.bundle["provenance"] != provenance:
            return (
                "artefact provenance round trip (S7.6)",
                False,
                "provenance dict changed across an export/reload round trip",
            )

        # a corrupted field has to raise on load, not warn
        bad_bundle = dict(bundle)
        bad_provenance = dict(provenance)
        bad_provenance["win_short_hr_s"] = 15.0
        bad_bundle["provenance"] = bad_provenance
        bad_path = Path(tmp) / "corrupt.joblib"
        joblib.dump(bad_bundle, bad_path)
        try:
            export_model.StressModel.load(bad_path)
        except ValueError:
            pass
        else:
            return (
                "artefact provenance round trip (S7.6)",
                False,
                "a corrupted provenance field loaded without raising",
            )

    return (
        "artefact provenance round trip (S7.6)",
        True,
        f"{len(export_model.PROVENANCE_KEYS)} provenance field(s) survive export/reload "
        "and a mismatch correctly raises",
    )


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

CHECKS = [
    check_constants,
    check_cache_freshness,
    check_unit_regression,
    check_hr_sample_count,
    check_standardisation_agreement,
    check_artefact_provenance_roundtrip,
]


def main() -> int:
    results: list[Result] = []
    for check in CHECKS:
        try:
            results.append(check())
        except Exception as e:  # noqa: BLE001
            results.append((check.__name__, False, f"raised {type(e).__name__}: {e}"))

    print("MIGRATION VALIDATION")
    print("=" * 72)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("=" * 72)
    if n_fail:
        print(f"FAIL — {n_fail}/{len(results)} check(s) failed")
        return 1
    print(f"PASS — {len(results)}/{len(results)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
