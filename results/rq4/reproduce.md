# RQ4 Reproduction Notes

This directory keeps the final RQ4 results in two sub-experiment folders.

## Part A

Boundary-search sensitivity can be regenerated into its sub-experiment directory:

```powershell
python -m evaluation.rq4.rq4_eval `
  --root . `
  --rq4-dir results\rq4\part_a_boundary_search_sensitivity `
  --run-boundary-sweep `
  --workers 16
```

The run writes `rq4_boundary_sensitivity.csv/json`. Rename these to
`boundary_sensitivity.csv/json` if replacing the curated files.

## Part B

First regenerate the Stage 3 representative regions used by the partition
sensitivity experiment:

```powershell
python -m src.stage3.subregions `
  --platform boiler_ccs `
  --hazard-id H-PRESSURE-001 `
  --hazard-driver fuel_command `
  --base-path results\stage2\boiler_ccs\fuel_command\boundary_results.json `
  --conditional-path results\stage2\boiler_ccs\fuel_command\conditional_results.json `
  --extraction-path results\stage1\boilerCCS\extraction.json `
  --manifest-path simulators\boiler_ccs\system_manifest.json `
  --output-root results\rq4\part_b_masking_partition_sensitivity\work\stage3_representative_regions `
  --seed 460 `
  --workers 16 `
  --representative-samples 10 `
  --sample-duration-mod 50
```

Then regenerate the fine partition sweep:

```powershell
python -m evaluation.rq4.rq4_eval `
  --root . `
  --rq4-dir results\rq4\part_b_masking_partition_sensitivity `
  --run-partition-sweep `
  --partition-validation-samples 0 `
  --partition-alarm-accuracy-samples 1000 `
  --partition-representative-root results\rq4\part_b_masking_partition_sensitivity\work\stage3_representative_regions `
  --partition-driver fuel_command `
  --partition-factor 0.1 `
  --partition-factor 0.2 `
  --partition-factor 0.3 `
  --partition-factor 0.4 `
  --partition-factor 0.5 `
  --partition-factor 0.6 `
  --partition-factor 0.7 `
  --partition-factor 0.8 `
  --partition-factor 0.9 `
  --partition-factor 1.0 `
  --partition-mask-offset-fraction 0.03 `
  --workers 16
```

The final Part B result uses:

- `representative_samples = 10`
- `sample_duration_mod = 50`
- `dynamic_slope_tol = 1e-5`
- `mask_offset_fraction = 0.03`
- `partition_alarm_accuracy_samples = 1000`

These are explicit reproduction parameters, not temporary debug knobs.
