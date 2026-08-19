# Run outputs

Typical primary-run outputs include:

- `run_config.json` — resolved CLI configuration
- `grid_data_summary.json` / `dataset_audit.json` — dataset integrity information
- `split_audit.json` — post-split class counts and effective batch size
- `history.jsonl` — epoch-by-epoch training and validation metrics
- `best.pt` — validation-selected checkpoint
- `last.pt` — final optimization state
- `brisc_train_metrics.json`
- `brisc_validation_metrics.json`
- `brisc_internal_test_metrics.json`
- `brisc_final_metrics.json`
- `pmram_external_metrics.json`
- `figshare_external_metrics.json` (optional)
- `figshare_overlap_audit.json` (optional)
- `final_complete_summary.json`

The morphology-ablation pipeline additionally writes seed-level and aggregate CSV/JSON tables, per-run profiles, confusion matrices, learning curves, and paper-oriented summary tables.

Large binary checkpoints are ignored by the repository `.gitignore` by default. For public releases, attach checkpoints to a GitHub Release or an archival object store and publish checksums.
