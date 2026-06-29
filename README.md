# CRAVE Open Reproducibility Artifact

This repository contains an anonymized artifact for evaluating CRAVE-style
reach-avoid vulnerability discovery on Boiler CCS and Tennessee Eastman models.

## Contents

- `src/stage1/`: semantic extraction, sensitivity, and hazard ranking.
- `src/stage2/`: base and conditional search.
- `src/stage3/`: coordinated subregion construction.
- `simulators/`: platform models, manifests, variable tables, and runtime executors.
- `evaluation/`: RQ1-RQ4 artifact consumers and replay scripts.
- `results/`: compact release snapshots used by the paper tables.

## Setup

Use Python 3.12 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run a quick script check:

```powershell
$files = Get-ChildItem -Recurse -Filter *.py | ForEach-Object { $_.FullName }
python -m py_compile $files
python -m src.stage2.base_search --help
python -m evaluation.rq4.rq4_eval --help
```

## Core Commands

The commands below regenerate full artifacts and can run for several minutes.

Stage 2 base search:

```powershell
python -m src.stage2.base_search --platform boiler_ccs --hazard-driver fuel_command --workers 16
```

Stage 2 conditional search:

```powershell
python -m src.stage2.conditional_search `
  --platform boiler_ccs `
  --boundary-path results/stage2/boiler_ccs/fuel_command/boundary_results.json `
  --extraction-path results/stage1/boilerCCS/extraction.json `
  --manifest-path simulators/boiler_ccs/system_manifest.json `
  --output-path results/stage2/boiler_ccs/fuel_command/conditional_results.json `
  --output-root . `
  --workers 16
```

Stage 3 region construction:

```powershell
python -m src.stage3.subregions `
  --platform boiler_ccs `
  --hazard-id H-PRESSURE-001 `
  --hazard-driver fuel_command `
  --base-path results/stage2/boiler_ccs/fuel_command/boundary_results.json `
  --conditional-path results/stage2/boiler_ccs/fuel_command/conditional_results.json `
  --extraction-path results/stage1/boilerCCS/extraction.json `
  --manifest-path simulators/boiler_ccs/system_manifest.json `
  --output-root . `
  --workers 16
```

Read the retained RQ summaries:

```powershell
Get-Content results\rq1\rq1_main_tables.md
Get-Content results\rq2\rq2_main_table.md
Get-Content results\rq3\TE\summary.md
Get-Content results\rq4\README.md
```

RQ scripts can regenerate summaries and sensitivity analyses. RQ1 rebuilds its
paper table from the retained replay scan by default; pass `--run-w-scan` only
when you want to replace the retained RQ1 scan snapshot. Some other commands run
for minutes and write fresh outputs under `results/`, so run them when you want
to replace the retained snapshots:

```powershell
python -m evaluation.rq1.rq1_eval --root .
python -m evaluation.rq2.rq2_eval --root . --boundary-path results\rq2\te_crave_safe_boundary.json --budget 2939 --workers 4
python -m evaluation.RQ3.rq3_eval --root . --workers 4
```

For RQ4 Part A and Part B sweeps, use `results/rq4/reproduce.md`.

## Anonymization

This release intentionally avoids author names, affiliations, private paths,
and development-history notes. Generated replay traces and work directories are
excluded from version control; rerunning the commands above recreates them when
needed.
