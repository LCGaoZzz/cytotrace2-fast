# Requests, input format and outputs

## Saved JSON request

```json
{
  "input_path": "counts.tsv",
  "output_dir": "runs/cytotrace2-001",
  "species": "human",
  "expression_scale": "counts",
  "binary": "/absolute/path/to/cytotrace2",
  "assets": "/absolute/path/to/assets",
  "seed": 14,
  "batch_size": 10000,
  "smooth_batch_size": 1000,
  "max_cores": 4,
  "timeout_seconds": null
}
```

Required: `input_path`, `output_dir`, `species`, `expression_scale`, `assets`.
Other fields above show defaults; `binary` defaults to `cytotrace2` on PATH.
Paths are resolved relative to the **request file**, not the current directory.
A binary containing a slash is a path; a bare binary name is resolved on PATH.
There are no remote URLs, shell command strings, arbitrary extra arguments or
automatic package installs. This is a local trusted-workspace adapter, **not a
security sandbox**; the user-selected executable is executed with host permissions.

Species is exactly `human` or `mouse`; scale is exactly `counts`, `CPM` or `TPM`.
Seed is an integer in [0, 2^32-1]. Batch sizes and max_cores are positive integers;
booleans are rejected. Timeout is null or positive finite seconds. Unknown fields
are errors. `max_cores=4` is the harness's conservative resource default, not the
Rust CLI's automatic default. No annotation-path field is offered: the Rust CLI
parses that compatibility option but does not read the annotation file.

## Native text contract

UTF-8, uncompressed **TSV**, genes in rows, cells in columns. The first line is
ONLY cell IDs. There is no gene header and no leading empty field:

```text
cell-A<TAB>cell-B
GeneA<TAB>1<TAB>4
GeneB<TAB>3<TAB>0
```

`<TAB>` denotes a literal tab. IDs must be unique and nonempty; no quoted fields,
tabs/newlines, or leading/trailing whitespace. Do not pass a normal dataframe
CSV or TSV with an index-column header directly. For an already prepared small
**genes-by-cells** dataframe `df`, explicitly write the cell header and then
`df.to_csv(stream, sep="\t", header=False, index=True)`; validate the result.
This adapter does not accept CSV, gzipped text or h5ad as the native binary input.

Source verification at base commit
`a3c6e7f7949b218edb0d13d41141b4bbf1fc881e`:
`src/pipeline.rs::read_matrix_once` splits the header on literal tabs and treats
every field as a cell ID. It does not use a CSV parser. This is why the adapter
follows the implementation rather than the root README's broader CSV wording.
`src/main.rs::parse_args` defines the CLI parameters and defaults.

The default check reads the header and first gene row only. Full validation
streams every row, rejects duplicate genes, malformed dimensions, empty values,
negative/non-finite numbers and an all-zero matrix. It does **not** verify species,
gene-symbol semantics, counts provenance, per-cell library quality, model feature
coverage or biological applicability. Those remain scientific preflight checks.
Full scans can be expensive; the manifest records their scope rather than
mislabeling a header-only check as full validation.

## h5ad export

Resolve both scripts in the same installed Skill and use the prepared interpreter:

```bash
python /resolved/scripts/prepare_h5ad.py \
  --input /data/input.h5ad --output /workspace/counts.tsv \
  --matrix-source layers/counts --gene-column gene_symbols \
  --expression-scale counts --gene-chunk-size 16
```

`--matrix-source` is required: `X`, `raw.X`, or `layers/NAME`. The default gene
column is `var_names`; a named column comes from the selected matrix's `var`
(`raw.var` for `raw.X`). Duplicate or missing IDs fail; there is no gene collapse,
alias correction or implicit raw-slot fallback. Check human/mouse gene-symbol
provenance and upstream mapping coverage before inference, especially for
Ensembl IDs or targeted panels.

The helper reads in backed mode and densifies only an export block of genes at
a time. This is **not a promise of bounded total RAM**: anndata can materialize
layers/metadata, and sparse column slicing can be expensive. Dense TSV size can
also dominate disk usage. Preserve and reuse a valid export where appropriate.
No transformation is applied; values are emitted with 17 significant digits for
the Rust f64 parser. A `.provenance.json` sidecar records input metadata, selected
slot, gene column, declared scale, dimensions and anndata version. Failures remove
only output files created by that invocation. Existing outputs/sidecars are refused.
The original h5ad is opened read-only and never modified.

## Results and joining

```text
<output_dir>/
  cytotrace2_results.txt
  results_manifest.json
  summary.json
  stdout.log
  stderr.log
```

The table is the **unaltered Rust output**: an unnamed cell-ID index followed by
`CytoTRACE2_Score`, `CytoTRACE2_Potency`, `CytoTRACE2_Relative`,
`preKNN_CytoTRACE2_Score`, `preKNN_CytoTRACE2_Potency`.
The wrapper verifies schema, exact cell order/count and finite score ranges.
Missing numeric entries remain missing and produce `completed_with_warnings`;
all missing final scores fail. Potency counts include rows with missing scores:
filter/report missingness explicitly before biological comparisons.

A manifest records the resolved request, exact argument list, binary version and
SHA-256, asset-manifest SHA-256, selected engine environment, input size/mtime,
return code, runtime state and elapsed execution seconds **after preflight**.
Asset existence checks are not payload hash validation; input size/mtime are not
content hashes. Hash large inputs/assets separately when an audit requires it.
Export/full-validation time is separate from Rust execution; do not call wrapper
wall time an algorithm-only benchmark. Native stage timings remain in stdout.log.

Failures after the run directory is created retain `failed`/`interrupted` metadata
and logs. Preflight failures create no run directory. `status` is a recorded-state
read: after SIGKILL or a host crash, `running` can be stale and partial files do
not establish completion. Never reuse that directory for a fresh run.

Join by literal IDs, never positional assignment to a different object:

```python
import pandas as pd
result = pd.read_csv(result_path, sep="\t", index_col=0)
assert result.index.is_unique and adata.obs_names.is_unique
assert set(result.index) == set(adata.obs_names)
assert not set(result.columns) & set(adata.obs.columns), "refuse silent column overwrite"
adata.obs = adata.obs.join(result, how="left", validate="one_to_one")
# Write a NEW h5ad only when requested; never overwrite the source.
```
