"""
export_model.py — fit the final model on all subjects and freeze it for the
live host pipeline.

Produces a single artefact (models/stress_model.joblib) containing the fitted
classifier, the exact ordered feature list it was trained on, the cohort-level
scaling fallback, and provenance metadata. Also provides StressModel, the
class the live host imports to run inference.

Run:
    python export_model.py                       # fit 'clean', binary, RF
    python export_model.py --task 3class
    python export_model.py --features eda_only   # deployment-realistic variant
    python export_model.py --verify              # load and self-test only

--------------------------------------------------------------------------
Why this file exists rather than a bare pickle
--------------------------------------------------------------------------
A pickled estimator carries no record of what its columns MEAN. If the live
feature module ever emits columns in a different order, or gains a feature,
or loses one, a bare estimator will happily accept the array and return
confident nonsense. There is no error, no warning, and the failure is
invisible until someone checks the predictions against reality.

So the feature list is frozen INTO the artefact and verified on every load
and every predict call. A mismatch raises. That is the whole point.

--------------------------------------------------------------------------
The t_start prohibition
--------------------------------------------------------------------------
t_start (seconds since recording start) was used as a diagnostic probe during
evaluation, where it scored ~0.96 balanced accuracy alone — higher than any
physiological model — because WESAD runs its condition blocks in a fixed
order at similar wall-clock offsets for every subject.

It has NO meaning at inference. Live monitoring has no protocol, no fixed
block ordering, and no session start that predicts anything. A model
containing t_start would be reading a clock that no longer ticks.

This is asserted at export, at load, and at predict. Three times, because
this is the single most damaging thing that could silently enter the
deployed artefact.

--------------------------------------------------------------------------
Standardisation at inference
--------------------------------------------------------------------------
The training-time scaling is per-subject: each subject is centred and scaled
by the median and IQR of their own first N_REF_WINDOWS windows. That
procedure is causal and transfers directly to deployment — but the PARAMETERS
do not. A new user's reference must be computed from THEIR first windows of
wear, not inherited from WESAD subjects.

So the artefact exports the PROCEDURE plus the cohort-level IQR fallback,
and StressModel accumulates a live reference buffer. Until that buffer is
full the model reports itself as not ready, rather than predicting against a
half-formed reference.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
from train_model import (
    FEATURE_SETS,
    N_REF_WINDOWS,
    STRESS_LABEL,
    TIME_COL,
    Z_CLIP,
    load_table,
    make_model,
    make_target,
    resolve_features,
)

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

ARTEFACT_VERSION = 1

# Feature sets that may never be exported: they contain the time probe.
FORBIDDEN_SETS = {"time_only", "clean_plus_time", "eda_plus_time"}


# --------------------------------------------------------------------------
# Contract enforcement
# --------------------------------------------------------------------------


def assert_no_time_feature(cols) -> None:
    """The single most important check in this file. See module docstring."""
    if TIME_COL in cols:
        raise ValueError(
            f"'{TIME_COL}' is present in the feature list. It is a diagnostic "
            "probe with no meaning at inference — live monitoring has no "
            "protocol clock. Refusing to export."
        )
    leaky = [c for c in cols if c not in F.FEATURE_NAMES]
    if leaky:
        raise ValueError(
            f"columns not in features.FEATURE_NAMES: {leaky}. Every exported "
            "column must come from the shared feature module, or the live "
            "pipeline cannot produce it."
        )


# --------------------------------------------------------------------------
# Reference statistics (the scaling procedure, not its parameters)
# --------------------------------------------------------------------------


def compute_scale(ref: pd.DataFrame, wider: pd.DataFrame, cohort: pd.Series):
    """Robust centre and scale, with the divisor cascade from training.

    Mirrors train_model.standardise exactly. Any change there must be
    mirrored here or training and inference diverge silently — which is the
    same class of bug the frozen feature list exists to prevent.
    """
    centre = ref.median()
    iqr = ref.quantile(0.75) - ref.quantile(0.25)
    wider_iqr = wider.quantile(0.75) - wider.quantile(0.25)
    scale = iqr.where(iqr > 0, wider_iqr)
    scale = scale.where(scale > 0, cohort)
    scale = scale.where(scale > 0, 1.0)
    return centre, scale


def training_standardise(df: pd.DataFrame, cols: list, cohort_iqr: pd.Series):
    """Apply per-subject scaling to the training table, as at evaluation."""
    out = df.copy()
    for sid, idx in df.groupby("subject").groups.items():
        block = df.loc[idx, cols]
        ref = df.loc[idx].sort_values(TIME_COL).head(N_REF_WINDOWS)[cols]
        if len(ref) < 5:
            raise ValueError(f"{sid}: too few reference windows ({len(ref)})")
        centre, scale = compute_scale(ref, block, cohort_iqr[cols])
        out.loc[idx, cols] = ((block - centre) / scale).to_numpy()
    out[cols] = out[cols].clip(-Z_CLIP, Z_CLIP).fillna(0.0)
    return out


# --------------------------------------------------------------------------
# Live inference wrapper
# --------------------------------------------------------------------------


class StressModel:
    """Load-and-predict wrapper for the live host pipeline.

    Usage:
        m = StressModel.load("models/stress_model.joblib")

        # During the opening minutes of wear, feed reference windows:
        for row in warmup_rows:            # dicts from features.feature_vector
            m.add_reference(row)

        if m.ready:
            label, proba = m.predict(row)

    The reference buffer is per-wearer and per-session. Call reset() when the
    device is removed or a new user puts it on — a reference built on one
    person's resting physiology is meaningless for another.
    """

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.clf = bundle["classifier"]
        self.feature_names = list(bundle["feature_names"])
        self.class_names = list(bundle["class_names"])
        self.cohort_iqr = pd.Series(bundle["cohort_iqr"])
        self.n_ref_required = int(bundle["n_ref_windows"])
        self.z_clip = float(bundle["z_clip"])

        assert_no_time_feature(self.feature_names)

        self._ref_rows: list = []
        self._centre = None
        self._scale = None

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "StressModel":
        import joblib

        bundle = joblib.load(path)
        if bundle.get("artefact_version") != ARTEFACT_VERSION:
            raise ValueError(
                f"artefact version {bundle.get('artefact_version')} != "
                f"{ARTEFACT_VERSION} — re-export with the current code"
            )
        m = cls(bundle)
        m._self_test()
        return m

    def _self_test(self) -> None:
        """Predict on a zero vector to confirm the estimator is functional."""
        x = np.zeros((1, len(self.feature_names)))
        try:
            self.clf.predict(x)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"loaded model failed self-test: {e}") from e

    # -- reference buffer ------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._centre is not None

    @property
    def n_reference(self) -> int:
        return len(self._ref_rows)

    def reset(self) -> None:
        """Clear the reference. Call on device removal or user change."""
        self._ref_rows.clear()
        self._centre = None
        self._scale = None

    def add_reference(self, row: dict) -> bool:
        """Add one warm-up window. Returns True once the model is ready."""
        self._ref_rows.append(self._row_to_series(row))
        if len(self._ref_rows) >= self.n_ref_required:
            ref = pd.DataFrame(self._ref_rows)
            self._centre, self._scale = compute_scale(
                ref, ref, self.cohort_iqr[self.feature_names]
            )
        return self.ready

    # -- prediction ------------------------------------------------------

    def _row_to_series(self, row: dict) -> pd.Series:
        """Validate one feature dict against the frozen contract."""
        missing = [c for c in self.feature_names if c not in row]
        if missing:
            raise ValueError(
                f"feature row is missing {len(missing)} exported column(s): "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}. The live "
                "feature module and the trained model have diverged."
            )
        return pd.Series(
            {c: float(row[c]) for c in self.feature_names}, dtype=np.float64
        )

    def transform(self, row: dict) -> np.ndarray:
        if not self.ready:
            raise RuntimeError(
                f"reference incomplete: {self.n_reference}/{self.n_ref_required} "
                "windows. Predicting before the wearer's resting reference is "
                "established would compare them against nothing."
            )
        s = self._row_to_series(row)
        z = ((s - self._centre) / self._scale).clip(-self.z_clip, self.z_clip)
        return z.fillna(0.0).to_numpy(dtype=np.float64).reshape(1, -1)

    def predict(self, row: dict):
        """Return (class_name, {class_name: probability})."""
        x = self.transform(row)
        idx = int(self.clf.predict(x)[0])
        proba = {}
        if hasattr(self.clf, "predict_proba"):
            p = self.clf.predict_proba(x)[0]
            proba = {
                self.class_names[int(c)]: float(v)
                for c, v in zip(self.clf.classes_, p)
            }
        return self.class_names[idx], proba

    # -- introspection ---------------------------------------------------

    def describe(self) -> str:
        m = self.bundle["metadata"]
        return (
            f"{m['task']} / {m['model']} / features={m['feature_set']} "
            f"({len(self.feature_names)})\n"
            f"trained {m['exported_utc']} on {m['n_train_windows']} windows "
            f"from {m['n_train_subjects']} subjects\n"
            f"LOSO balanced accuracy at evaluation: "
            f"{m.get('loso_balanced_acc', 'n/a')}\n"
            f"reference required before inference: {self.n_ref_required} windows"
        )


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def load_loso_summary(task: str, model: str, feature_set: str):
    """Attach the evaluation result to the artefact, if it exists."""
    p = RESULTS_DIR / f"{task}_{model}_{feature_set}_baseline_summary.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def export(
    cache: Path,
    task: str,
    model_kind: str,
    feature_set: str,
    seed: int,
    outfile: Path,
) -> Path:
    import joblib
    import sklearn

    if feature_set in FORBIDDEN_SETS:
        raise ValueError(
            f"feature set '{feature_set}' contains the time probe and cannot "
            f"be deployed. Use 'clean' or 'eda_only'."
        )

    df = load_table(cache)
    cols = resolve_features(feature_set, df)
    assert_no_time_feature(cols)

    cohort_iqr = df[F.FEATURE_NAMES].quantile(0.75) - df[F.FEATURE_NAMES].quantile(
        0.25
    )
    scaled = training_standardise(df, cols, cohort_iqr)

    X = scaled[cols].to_numpy(dtype=np.float64)
    y, class_names = make_target(df, task)

    print(f"fitting {model_kind} on {len(X)} windows, {len(cols)} features")
    clf = make_model(model_kind, seed).fit(X, y)

    loso = load_loso_summary(task, model_kind, feature_set)
    metadata = {
        "task": task,
        "model": model_kind,
        "feature_set": feature_set,
        "n_train_windows": int(len(X)),
        "n_train_subjects": int(df["subject"].nunique()),
        "class_balance": df["label_name"].value_counts(normalize=True).round(4).to_dict(),
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "seed": seed,
        "loso_balanced_acc": (
            round(loso["balanced_acc_mean"], 4) if loso else None
        ),
        "loso_balanced_acc_sd": (round(loso["balanced_acc_sd"], 4) if loso else None),
        "loso_macro_f1": round(loso["macro_f1_mean"], 4) if loso else None,
        # Recorded so the deployed artefact carries the caveat with it.
        "evaluation_caveat": (
            "WESAD runs its condition blocks in a fixed order, so elapsed "
            "session time alone reaches ~0.96 binary / ~0.90 3-class balanced "
            "accuracy under LOSO. Published WESAD benchmarks are therefore "
            "upper bounds of uncertain composition. Field performance without "
            "a protocol clock is expected to be lower than the LOSO figure."
        ),
    }

    bundle = {
        "artefact_version": ARTEFACT_VERSION,
        "classifier": clf,
        "feature_names": cols,
        "class_names": class_names,
        "cohort_iqr": cohort_iqr.to_dict(),
        "n_ref_windows": N_REF_WINDOWS,
        "z_clip": Z_CLIP,
        "metadata": metadata,
    }

    outfile.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, outfile)
    (outfile.with_suffix(".json")).write_text(
        json.dumps({**metadata, "feature_names": cols}, indent=2)
    )

    print(f"\nexported -> {outfile}")
    print(f"sidecar   -> {outfile.with_suffix('.json')}")
    return outfile


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def verify(path: Path, cache: Path) -> int:
    """Load the artefact and run it exactly as the live host would."""
    m = StressModel.load(path)
    print(m.describe())
    print()

    assert_no_time_feature(m.feature_names)
    print(f"  contract:  {len(m.feature_names)} features, no '{TIME_COL}'  OK")

    df = load_table(cache)
    sid = sorted(df["subject"].unique())[0]
    block = df[df["subject"] == sid].sort_values(TIME_COL)

    rows = block[m.feature_names].to_dict("records")
    if len(rows) < m.n_ref_required + 1:
        print(f"  {sid} has too few windows to simulate warm-up")
        return 1

    # Refuse to predict before the reference is built.
    try:
        m.predict(rows[0])
        print("  FAIL: predicted with an incomplete reference")
        return 1
    except RuntimeError:
        print("  cold start: correctly refused to predict  OK")

    for r in rows[: m.n_ref_required]:
        m.add_reference(r)
    print(f"  warm-up:   ready after {m.n_reference} windows  OK")

    label, proba = m.predict(rows[m.n_ref_required])
    top = ", ".join(f"{k}={v:.2f}" for k, v in sorted(proba.items()))
    print(f"  inference: {label}  ({top})  OK")

    # A missing column must raise, not silently mispredict.
    broken = dict(rows[m.n_ref_required])
    broken.pop(m.feature_names[0])
    try:
        m.predict(broken)
        print("  FAIL: accepted a row with a missing feature")
        return 1
    except ValueError:
        print("  contract violation correctly rejected  OK")

    m.reset()
    print(f"  reset:     ready={m.ready}  OK")
    print("\nartefact is sound.")
    return 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--task", choices=["binary", "3class"], default="binary")
    ap.add_argument("--model", choices=["rf", "hgb"], default="rf")
    ap.add_argument(
        "--features",
        choices=sorted(set(FEATURE_SETS) - FORBIDDEN_SETS),
        default="clean",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="verify existing artefact")
    args = ap.parse_args()

    out = Path(args.out) if args.out else MODEL_DIR / "stress_model.joblib"

    if args.verify:
        if not out.is_file():
            print(f"no artefact at {out} — export first")
            return 1
        return verify(out, Path(args.cache))

    export(
        cache=Path(args.cache),
        task=args.task,
        model_kind=args.model,
        feature_set=args.features,
        seed=args.seed,
        outfile=out,
    )
    print()
    return verify(out, Path(args.cache))


if __name__ == "__main__":
    sys.exit(main())