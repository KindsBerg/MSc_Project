These files were part of the original results/pre_migration_sen0344_lsm6ds3/
backup but are NOT valid pre-migration comparators for the SEN0344/
LSM6DS3TR-C hardware migration.

They all record n_ref_windows=100 (see ablation_sweep.csv and each
*_summary.json's "n_ref_windows" field), a different reference-buffer
size than the n_ref=40 used both by train_model.py's own N_REF_WINDOWS
default and by every post-migration run in this project. Comparing
them against post-migration n_ref=40 numbers is an apples-to-oranges
config mismatch, not a hardware-migration effect — that mismatch is
what originally produced an incorrect "before" figure of 0.905 for
`clean` in README.md.

Moved here (2026-08-04) so they can't be picked up again by a future
before/after comparison. The valid pre-migration baseline, matched on
n_ref=40, is still in ../pre_migration_sen0344_lsm6ds3/:
  binary_rf_clean_baseline_n40_summary.json      (clean:    0.8724 +/- 0.1321)
  binary_rf_eda_only_baseline_n40_summary.json   (eda_only: 0.8285)

Nothing here was deleted, only relocated.
