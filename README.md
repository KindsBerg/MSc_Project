# MSc_Project — Wearable Physiological Sensing: ML Pipeline

Data analytics pipeline for a wrist-worn multimodal biosensor measuring galvanic skin
response (GSR/EDA), PPG-derived heart rate, tri-axial motion and skin temperature.

This repository contains the **offline training half** of the system: it trains a
stress/affect classifier on the WESAD benchmark and exports a frozen model artefact
that the live host pipeline loads at runtime. Nothing here touches the device
directly — see `notes.txt` for how the training path (B) connects to the live signal
path (A).

MSc Individual Project, University of Hertfordshire.

---

## Pipeline

| File | Role |
|---|---|
| `wesad_loader.py` | Parses WESAD subject pickles into aligned wrist streams |
| `verify_wesad.py` | Acceptance test for the raw dataset — run before anything else |
| `features.py` | **The shared feature module.** Single implementation of all 42 features |
| `build_dataset.py` | Slides windows, extracts features, attaches labels, caches per subject |
| `train_model.py` | Leave-one-subject-out evaluation, ablation sweep, confound diagnostics |
| `diagnose_subjects.py` | Forensics for failing folds |
| `export_model.py` | Fits the final model and freezes the feature contract into an artefact |
| `validate_migration.py` | Gate for hardware-constant migrations: constants, cache freshness, unit regression, standardisation agreement, artefact provenance |

`features.py` is imported by both the training build and the live host pipeline. One
implementation, no drift — the same columns in the same order on both sides.

---

## Setup

### 1. Dependencies

```bash
python -m pip install numpy scipy pandas scikit-learn neurokit2 cvxopt pyarrow joblib matplotlib
```

`cvxopt` is **required**, not optional. Without it NeuroKit2 cannot run cvxEDA and the
EDA decomposition silently falls back to a median filter — producing plausible numbers
by a method the report does not describe. `build_dataset.py --check` will refuse to
build if this happens.

### 2. Dataset

WESAD is not included (~2 GB, and redistribution isn't ours to make). Download it from
the University of Siegen:

```
https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download
```

Unzip so that the subject folders `S2`, `S3`, … sit directly inside one directory.
S1 and S12 are absent — excluded by the original authors for sensor faults.

### 3. Local config

```bash
cp config.example.py config.py
```

Then edit `config.py` to point at your unzipped dataset. It is gitignored because the
path is machine-specific.

---

## Running

```bash
python verify_wesad.py                       # confirm the dataset loads and aligns
python verify_wesad.py --plot S2             # visual alignment check (needs matplotlib)

python build_dataset.py --check              # dependency preflight
python build_dataset.py --force --combine    # build the feature table (~10 min)

python train_model.py --diagnose             # time-confound correlation check
python train_model.py --task both --model rf --sweep

python export_model.py                                   # clean + RF, binary
python export_model.py --features eda_only --out models/stress_model_eda.joblib
python export_model.py --verify                          # load and self-test
```

Run `verify_wesad.py` first and don't proceed until it passes. Every downstream stage
assumes the label track is correctly aligned to the wrist streams, and a constant
offset would pass most assertions silently — which is why that script also plots EDA
against the protocol labels.

---

## Results

Leave-one-subject-out, 15 folds, Random Forest, per-subject robust scaling.
Full numbers in `results/ablation_sweep.csv`. Current figures are post-migration (SEN0344
0.25 Hz HR cadence, 40 s HR window) — see "Migration: before/after" below for what changed
and why.

| Feature set | n | Binary balanced acc | 3-class balanced acc |
|---|---|---|---|
| `all` | 36 | 0.844 | 0.648 |
| **`clean`** | **32** | **0.860** | **0.669** |
| `eda_only` | 10 | 0.829 | 0.583 |
| `time_only` | 1 | 0.960 | 0.903 |

`clean` drops the posture (mean-acceleration) channels. It beats the full feature set with
four fewer features.

`eda_only` uses the EDA block alone — the signal the custom analog front-end exists to
measure — and is the most deployment-realistic figure.

### Migration: before/after

The SEN0344 and LSM6DS3TR-C hardware facts (§1-§5 of the migration, not reproduced here)
resolved two provisional constants and replaced the IMU part. Both sides of this table use
the standard `train_model.py` sweep at its own `N_REF_WINDOWS=40` default, so the
comparison is config-matched, not just same-codebase:

| Config | Binary balanced acc — before (n_ref=40) | after (n_ref=40) | Δ |
|---|---|---|---|
| `clean` | 0.8724 ± 0.1321 | 0.8596 ± 0.1182 | −0.0128 |
| `eda_only` | 0.8285 ± 0.1704 | 0.8285 ± 0.1704 | **0.0000** (bit-exact) |

`eda_only` is untouched by construction — it only reads the EDA block, which this
migration never modified — and the two runs agree to 15 decimal places, which is itself a
useful reproducibility check (RF `random_state` is pinned). `clean`'s margin over
`eda_only` narrowed from +0.044 to +0.031. Smaller than initially estimated: an earlier
draft of this table compared against a stale `n_ref=100` run and additionally carried a
since-fixed bug where the 40 s HR sub-window was anchored to window *start* rather than
window *close*, leaving the most recent 20 s of every 60 s window's HR signal unused at
prediction time. Both are corrected here. The qualitative result is unchanged: `clean`
converges toward `eda_only` rather than holding its pre-migration margin, which is evidence
for the project's central claim — that direct EDA measurement, not HR derived from a
rate-limited PPG estimator, is the defensible core of this pipeline.

Pre-migration comparison baseline: `results/pre_migration_sen0344_lsm6ds3/
binary_rf_{clean,eda_only}_baseline_n40_summary.json`. A separately-recorded pre-migration
figure of `eda_only=0.846` traces to `results/ablation_sweep_prior_contract.csv`, the
**older 42-feature contract** — a different feature set (this codebase has been on the
36-feature contract for some time) and not comparable to any number in this table.
`results/pre_migration_n_ref_100_not_comparable/` holds a full `n_ref=100` sweep that was
briefly (and incorrectly) used as the "before" figure; kept for the record, excluded from
comparison — see the README in that directory.

---

## Methodological notes

**Leave-one-subject-out, never a random split.** Windows slide 5 s over a 60 s window,
so consecutive rows share 55 s of source signal. A random shuffle puts near-duplicates
either side of the split and measures memorisation of a recording rather than
generalisation to a person.

**Session-order control.** `time_only` is elapsed time since recording start as a single
feature, with no physiology at all. It scores 0.960 binary balanced accuracy — above
every physiological configuration and above both published WESAD wrist-only ceilings
(~0.88 / ~0.76). WESAD runs its condition blocks in a fixed order at similar wall-clock
offsets for every subject, so session position generalises across subjects. It is
reported as a control, not a result, and every other figure should be read alongside it.

**Baseline references avoid label leakage.** Per-subject standardisation uses the first
N accepted windows of each recording, chosen without reference to any label. Averaging
over windows labelled *baseline* would use the target to construct a predictor.

**Feature parity is enforced, not assumed.** `feature_vector()` asserts against
`FEATURE_NAMES` on every call, and the exported artefact carries its ordered column list
and re-checks it on load and on every prediction.

**`t_start` is a diagnostic only.** It has no meaning at inference — live monitoring has
no protocol clock. It is blocked at export, at load and at predict, and the three
time-containing feature sets cannot be exported.

**Known limitation: standardisation can diverge in one narrow, by-design edge case.**
`export_model.training_standardise` and `StressModel.transform` both drive the same
`compute_scale()` primitive, but with a different "wider" fallback operand when a
reference window's IQR is zero for some feature: training falls back to that *subject's
whole block* (all its windows, training-time only information); live inference falls back
to the *reference buffer alone*, since a live wearer's future windows don't exist yet.
`validate_migration.py`'s standardisation-agreement check confirmed the two procedures
agree exactly on realistic data, then deliberately triggered this fallback branch and
measured a **19.47** divergence on the probe column — real, and by design, not a bug, but
worth knowing about if a real feature column ever has zero variance across a wearer's
reference window (unlikely for continuous physiological signals over 40 windows, but not
provably impossible). Not changed as part of this migration; flagged for awareness.

---

## Scope limits

Stated here because they are design decisions, not oversights:

- **No HRV.** RMSSD, SDNN, pNN50 and LF/HF need beat-to-beat intervals from raw BVP. The
  SEN0344 v2.0 firmware does not expose the MAX30102 FIFO, so only a computed HR number
  is available. `hr_sd` is variability of the HR *trend* and is not HRV. That computed
  number updates once every 4 s (measured off the `DFRobot_BloodOxygen_S` Arduino
  example — `delay(4000)`, with the device holding its last value between updates), which
  is why the HR block runs on its own 40 s window rather than the IMU's 15 s one: at that
  cadence a 15 s window would hold only 3-4 samples, not enough for `hr_sd`/`hr_slope` to
  mean anything. See "Hardware notes" below.
- **No SpO₂.** Same reason — no raw dual-channel FIFO access. Out of scope by data
  access, not by hardware capability.
- **No gyroscope in the model.** The LSM6DS3TR-C has one; the Empatica E4 that recorded
  WESAD does not. Features are extracted on the intersection of dataset and hardware, not
  the union. Gyro is still logged, in rad/s, for live signal-quality gating outside the
  model — the physiological argument (confound-rejection, not fusion) is unaffected by the
  part swap below.
- **No absolute temperature in the model.** WESAD's `TEMP` is a skin thermistor; the
  LM75BD on this PCB reads board temperature dominated by self-heating — a different
  instrument measuring a different thing. Two more thermal sources exist on the live path
  — the SEN0344's own register `0x14` and the LSM6DS3TR-C's die sensor — and neither is in
  the feature contract either; both are diagnostics only, logged but not modelled, for the
  same reason the LM75BD's *absolute* reading was dropped (see `features.py` module
  docstring for the ablation number).
- **Arousal, not emotion.** Targets are WESAD's labelled affective states, not discrete
  emotions. Peripheral autonomic signatures are not emotion-specific.

---

## Hardware notes

The wearable's PPG and IMU parts were fixed and bench-measured after the pipeline above
was first built; two provisional constants were resolved as a result, and one part
changed. Both are one-off migrations, not ongoing tuning:

- **PPG — DFRobot SEN0344 (MAX30102 + on-board MCU).** Outputs a computed HR number only
  (no raw waveform) at **0.25 Hz** — one new value every 4 s, held between updates and
  algorithm-smoothed, so consecutive samples are autocorrelated. `bvp_to_hr(out_fs=...)`
  emulates this cadence on the training side; `WIN_SHORT_HR_S = 40 s` is the HR block's
  window in response (see above).
- **IMU — Adafruit LSM6DS3TR-C** (replaces the MPU6050 referenced in earlier notes).
  Returns acceleration in **m/s²** and gyro in **rad/s** at up to 6.66 kHz; the library
  default is 104 Hz / ±4 g. The live pipeline configures **±2 g** (to match the Empatica
  E4's own clipping behaviour) and decimates 104 Hz down to the E4's **32 Hz**
  (`IMU_TARGET_FS_HZ`) before any feature code sees it. The m/s² → g conversion happens at
  the ingest boundary (`live_host.py`), never inside `features.py` — every g-scaled
  threshold there, `MOTION_STD_THRESHOLD_G` in particular, is calibrated in g, and
  `features.py` must only ever receive g.
- **ACC quantisation asymmetry.** The LSM6DS3TR-C resolves 0.061 mg/LSB at ±2 g; the E4
  resolves ~15.6 mg (1/64 g) — roughly 250x coarser. Left alone, the live signal would read
  systematically lower noise on still windows than anything in the training distribution,
  a train/serve mismatch orthogonal to the unit conversion above. `EMULATE_E4_QUANTISATION`
  (default on) rounds live, already-decimated g values onto the E4's 1/64 g grid to match;
  WESAD training data is already on that grid and is never quantised a second time. This is
  a characterised trade-off with a measured cost, not an optimisation — its state is
  recorded in every exported artefact's provenance.
- **Board temperature — three sources, one in the model.** The LM75BD, the SEN0344's
  register `0x14`, and the LSM6DS3TR-C's die sensor (256 LSB/°C, +25.0 °C offset) all
  report *something* thermal. Only the LM75BD (as a slow-context, non-windowed signal) was
  ever in the feature contract, and even that was dropped — see Scope limits above. The
  other two are diagnostics only; that is a decision, not an oversight.

---

## Repository layout

```
├── wesad_loader.py          # dataset parsing
├── verify_wesad.py          # dataset acceptance test
├── features.py              # shared feature module (training + live)
├── build_dataset.py         # windowing and feature extraction
├── train_model.py           # LOSO evaluation and ablation
├── diagnose_subjects.py     # per-subject forensics
├── export_model.py          # model freezing and live inference wrapper
├── validate_migration.py    # hardware-constant migration gate
├── config.example.py        # copy to config.py and set your dataset path
├── notes.txt                # signal architecture: live path, training path, and the join
└── results/                 # committed evaluation output — the evidence for the numbers above
```

Not tracked: `WESAD/` (the dataset), `cache/` (regenerable feature tables), `models/`
(regenerable artefacts), `config.py` (machine-specific).

---

## Status

The offline pipeline is complete and produces a deployable artefact. No data from the
physical device has entered it yet — all figures above are WESAD-to-WESAD.

Two constants that were provisional pending hardware are now resolved from bench
measurement, not guesses:

- `WIN_SHORT_HR_S = 40.0`, `WIN_SHORT_IMU_S = 15.0` (was one constant, `WIN_SHORT_S`) —
  split because the SEN0344's measured HR update cadence (0.25 Hz, see Hardware notes)
  needed a wider window than the IMU does
- `bvp_to_hr(out_fs=0.25)` — the SEN0344's measured cadence, cited from the
  `DFRobot_BloodOxygen_S` vendor library, not assumed

One constant remains provisional pending worn recordings:

- `MOTION_STD_THRESHOLD_G = 0.05` — retune against real worn data once available

Each is a single constant by design; changing it costs one rebuild of the feature table.

---

## References

1. P. Schmidt et al., "Introducing WESAD, a multimodal dataset for wearable stress and
   affect detection," *ICMI 2018*, doi:10.1145/3242969.3242985
2. A. Greco et al., "cvxEDA: A convex optimization approach to electrodermal activity
   processing," *IEEE TBME* 63(4), 2016, doi:10.1109/TBME.2015.2474131
3. D. Makowski et al., "NeuroKit2: A Python toolbox for neurophysiological signal
   processing," *Behavior Research Methods* 53, 2021, doi:10.3758/s13428-020-01516-y
4. W. Boucsein et al., "Publication recommendations for electrodermal measurements,"
   *Psychophysiology* 49(8), 2012, doi:10.1111/j.1469-8986.2012.01384.x