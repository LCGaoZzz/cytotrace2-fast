---
id: cytotrace2-fast
name: CytoTRACE2 Fast
description: Run or interpret cytotrace2-fast in Omicos, prepare explicitly selected h5ad expression slots, and retain CLI parameters, logs, missingness and cell-aligned results.
tier: community
category: general_omics_analysis
summary: Portable CytoTRACE2 Rust execution with explicit input provenance and traceable results.
execution_mode: packaged_python
runtime_entrypoint: scripts/run_cytotrace2.py
---

# CytoTRACE2 Fast analysis

Use the existing Rust binary (`cytotrace2-fast >=1.3.1`) and original asset bundle.
The runner requires only Python >=3.10; h5ad export additionally needs the existing
anndata/numpy/scipy environment. Do not rebuild or install on every invocation.
Choose the public CLI directly for unsupported custom options; do not invent
JSON fields or silently ignore user parameters to fit this adapter.

Preserve species, expression source, cells, genes, seeds and both batch sizes.
The adapter never normalizes, log-transforms, imputes, renames or downsamples.
Only unlogged counts/CPM/TPM are supported. Input provenance, not a successful
numeric check, establishes whether a matrix is appropriate.

## Portable entrypoints

Resolve `scripts/run_cytotrace2.py` with `skill_resource` and
`include_runtime_path: true`. Keep the sibling scripts together; no original
checkout or guessed cache path is required by the copied runtime.

```text
<python> <resolved scripts/run_cytotrace2.py> run --request <request.json>
<python> <resolved scripts/run_cytotrace2.py> validate --request <request.json> --full
<python> <resolved scripts/run_cytotrace2.py> doctor --request <request.json>
<python> <resolved scripts/run_cytotrace2.py> status --output-dir <run-directory>
```

These are independent tools, not mandatory stages. `run` cheaply checks the
header/first row before the Rust parser reads the input. For unvalidated text,
use `--full` on `validate` or `run` to scan all values. Do not repeatedly rescan a
large unchanged export without a reason. `status` reports saved state, not live
process health. Use the host job runner for lifecycle management.

For h5ad, resolve `scripts/prepare_h5ad.py` and explicitly select `X`, `raw.X` or
`layers/NAME` and the gene-symbol column. The native TSV header has **cell IDs
only**, with no leading tab or gene-column label. This differs from ordinary
pandas `to_csv` output. See the input reference before exporting.

Completion requires the actual result table, exact cell-ID/order alignment and
missing-value diagnostics, not just exit code zero. Existing output directories
are never overwritten. No plots or modified h5ad are produced automatically.

## References on demand

- [Requests, input format and outputs](references/inputs-and-results.md)
- [Runtime, scaling, interpretation and plotting](references/runtime-and-interpretation.md)
