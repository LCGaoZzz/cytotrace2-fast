---
id: cytotrace2-fast
name: CytoTRACE2 Fast
description: Run or interpret cytotrace2-fast in Omicos; assess counts/lognorm and other expression scales, prepare suitable input with existing tools, and retain cell-aligned results and provenance.
tier: community
category: general_omics_analysis
summary: LLM-led input assessment and preparation with portable CytoTRACE2 Rust execution and traceable results.
execution_mode: packaged_python
runtime_entrypoint: scripts/run_cytotrace2.py
---

# CytoTRACE2 Fast analysis

Use the existing Rust binary (`cytotrace2-fast >=1.3.1`) and original asset bundle.
The runner requires only Python >=3.10; h5ad export additionally needs the existing
anndata/numpy/scipy environment. Do not rebuild or install on every invocation.
Choose the public CLI directly for unsupported custom options; do not invent
JSON fields or silently ignore user parameters to fit this adapter.

Preserve user-fixed species, expression source, cells, genes, seeds and both
batch sizes. Only unlogged counts/CPM/TPM are supported by the execution path.
The scripts never infer scale, normalize, undo logs, impute, rename or downsample;
that does not prevent you from assessing and preparing inputs with existing tools.

## LLM-led input preparation

For an unprepared h5ad or matrix, inspect the available slots, matching gene IDs,
processing records and representative values yourself. Use that evidence to
choose a suitable matrix and fill the explicit `matrix-source`, `gene-column`
and `expression_scale` arguments. These are agent-resolved parameters, not a
requirement that the user manually classify their data. Prefer verified unlogged
counts/CPM/TPM; do not transform an already suitable input unnecessarily.

If only logged data are available and the history supports a reliable inverse,
prepare a separate linear-scale artifact with the correct scale declaration.
Never blindly apply `expm1` to an unknown matrix or call inverse-normalized values
raw counts. Consult [input assessment and conversion](references/inputs-and-results.md#llm-led-input-assessment)
for evidence, log-base handling and unsupported inputs. Record the decision and
any transformations in the existing notebook/analysis record, preserve the source,
then export, validate as needed and run. Routine justified preparation needs no
additional approval. Ask only about material unresolved ambiguity or a conflict
with a user-fixed choice, after inspecting the available evidence.

This is an instruction to the LLM, not a new automatic classifier in the scripts.
A successful numeric check does not verify expression scale; do not claim it does.

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

For h5ad, resolve `scripts/prepare_h5ad.py` and explicitly pass the source you
selected (`X`, `raw.X` or `layers/NAME`) and its gene-symbol column. If preparation
required a transformation, export your prepared artifact, not the original logged
slot. The native TSV header has **cell IDs only**, with no leading tab or
gene-column label. This differs from ordinary pandas `to_csv` output.

Completion requires the actual result table, exact cell-ID/order alignment and
missing-value diagnostics, not just exit code zero. Existing output directories
are never overwritten. No plots or modified h5ad are produced automatically by
the runner; agent-created preparation artifacts are separate and must be recorded.

## References on demand

- [Input assessment, requests, input format and outputs](references/inputs-and-results.md)
- [Runtime, scaling, interpretation and plotting](references/runtime-and-interpretation.md)
