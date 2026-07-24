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
Full numbers in `results/ablation_sweep.csv`.

| Feature set | n | Binary balanced acc | 3-class balanced acc |
|---|---|---|---|
| `all` | 42 | 0.887 | 0.756 |
| **`clean`** | **34** | **0.913** | **0.745** |
| `eda_only` | 10 | 0.846 | 0.616 |
| `time_only` | 1 | 0.960 | 0.903 |

`clean` drops absolute temperature and the posture (mean-acceleration) channels. It
beats the full feature set with eight fewer features.

`eda_only` uses the EDA block alone — the signal the custom analog front-end exists to
measure — and is the most deployment-realistic figure.

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

---

## Scope limits

Stated here because they are design decisions, not oversights:

- **No HRV.** RMSSD, SDNN, pNN50 and LF/HF need beat-to-beat intervals from raw BVP. The
  SEN0344 v2.0 firmware does not expose the MAX30102 FIFO, so only a computed HR number
  is available. `hr_sd` is variability of the HR *trend* and is not HRV.
- **No SpO₂.** Same reason — no raw dual-channel FIFO access. Out of scope by data
  access, not by hardware capability.
- **No gyroscope in the model.** The MPU6050 has one; the Empatica E4 that recorded WESAD
  does not. Features are extracted on the intersection of dataset and hardware, not the
  union. Gyro is still logged for live signal-quality gating outside the model.
- **Arousal, not emotion.** Targets are WESAD's labelled affective states, not discrete
  emotions. Peripheral autonomic signatures are not emotion-specific.

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
├── config.example.py        # copy to config.py and set your dataset path
├── notes.txt                # signal architecture: live path, training path, and the join
└── results/                 # committed evaluation output — the evidence for the numbers above
```

Not tracked: `WESAD/` (the dataset), `cache/` (regenerable feature tables), `models/`
(regenerable artefacts), `config.py` (machine-specific).

---

## Status

The offline pipeline is complete and produces a deployable artefact. No data from the
physical device has entered it yet — all figures above are WESAD-to-WESAD. Two
constants remain provisional pending hardware:

- `WIN_SHORT_S = 15.0` — set once the SEN0344's HR update cadence is measured
- `bvp_to_hr(out_fs=1.0)` — the emulated HR cadence on the training side

Both are single constants by design; changing them costs one rebuild of the feature
table.

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