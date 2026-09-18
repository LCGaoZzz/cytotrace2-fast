# cytotrace2-fast

`cytotrace2-fast` is a single-binary Rust implementation of [CytoTRACE 2](https://github.com/digitalcytometry/cytotrace2). It keeps the official command-line input and output contract, while replacing the Python/PyTorch runtime with a streamed, parallel Rust pipeline.

On the official mouse-pancreas vignette (2,850 cells), the packaged build completed in **4.49 s** versus **62.84 s** for the official Python CLI (**14.0× faster**) and used **2.98 GB** peak RSS versus **5.63 GB**. The output was byte-identical across repeated runs and matched the official result with a maximum absolute numeric difference of **1.72e-9**.

CytoTRACE 2's biological method, pretrained models, feature space, and mapping tables remain the upstream project's work. This repository contains only the best-performing v1.2.0 implementation and its runtime assets; earlier experimental implementations and the original Python calculation code are not included.

## What it predicts

CytoTRACE 2 predicts a continuous developmental-potential score and a potency category for every cell in a single-cell RNA-seq expression matrix. The output columns are the same as the official CLI:

```text
CytoTRACE2_Score
CytoTRACE2_Potency
CytoTRACE2_Relative
preKNN_CytoTRACE2_Score
preKNN_CytoTRACE2_Potency
```

For the biological interpretation and citation, see the [upstream CytoTRACE 2 documentation](https://github.com/digitalcytometry/cytotrace2) and Kang et al., *Nature Methods* (2025), doi:[10.1038/s41592-025-02857-2](https://doi.org/10.1038/s41592-025-02857-2).

## Installation

The repository builds with a recent stable Rust toolchain (tested with Rust 1.88 and 1.98) on Linux, macOS, or WSL2. No Python, PyTorch, R, or network access is required at run time.

```bash
git clone https://github.com/LCGaoZzz/cytotrace2-fast.git
cd cytotrace2-fast
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

The binary is `target/release/cytotrace2`. The `assets/` directory is part of the checkout and is found automatically when running from the repository or from the standard Cargo release layout. To store the assets elsewhere, set `CYTOTRACE2_ASSETS=/path/to/assets` or pass `--assets /path/to/assets`.

`target/` is a local build directory and is intentionally not part of the repository. `RUSTFLAGS="-C target-cpu=native"` gives the best performance on the build machine; omit it for a more portable binary.

## Running CytoTRACE 2

The invocation follows the official Python CLI:

```bash
./target/release/cytotrace2 \
  -f expression_data.txt \
  -a cell_annotations.txt \
  -sp mouse \
  --seed 14 \
  -o cytotrace2_results
```

The expression file is a tab- or comma-separated matrix with genes in rows, cells in columns, and gene names in the first column. Use raw counts or CPM/TPM values, not log2-transformed values. The annotation file is accepted with the same validation semantics as the official CLI. The result is written to `cytotrace2_results/cytotrace2_results.txt`.

Supported options:

| Short | Long | Default | Description |
| --- | --- | --- | --- |
| `-f` | `--input-path` | required | Expression matrix |
| `-a` | `--annotation-path` | required | Cell annotation file |
| `-sp` | `--species` | `mouse` | `mouse` or `human` |
| `-bs` | `--batch-size` | `10000` | Model prediction batch size |
| `-sbs` | `--smooth-batch-size` | `1000` | Smoothing batch size |
| `-r` | `--seed` | `14` | NumPy-compatible MT19937 seed |
| `-o` | `--output-dir` | `cytotrace2_results` | Output directory |
| `-dpa` | `--disable-parallelization` | off | Use one worker |
| `-mc` | `--max-cores` | automatic | Cap worker count |
| `-dpl` | `--disable-plotting` | off | Accepted for CLI compatibility; this release writes the result table only |
| `-dv` | `--disable-verbose` | off | Suppress parameter and progress messages |
|  | `--assets` | auto | Override the asset directory |
|  | `--dump-stage DIR` | off | Write intermediate arrays for debugging |

The binary also accepts `--version`.

## How the calculation differs from the official implementation

The model and biological semantics are kept fixed. The differences are in representation, scheduling, and numerical execution:

1. **Runtime**: the official package starts a Python/PyTorch/scanpy process and uses Python plotting; this release is one Rust binary and does not import those runtimes.
2. **Model assets**: the 19 official PyTorch state dictionaries are losslessly exported once to `.npz`; the background graph, feature list, aliases, and ortholog table are bundled in `assets/`. No model is retrained or approximated.
3. **Preprocessing**: input parsing, CPM/log transform, descending-average ranks, and QC checks are fused into streaming row passes. The background graph is loaded directly into its CSR representation instead of materializing the dense intermediate.
4. **Ensemble forward pass**: model weights are evaluated in cell tiles of 16 with fused layer sweeps. Accumulation stays in deterministic `f64` order so the official anchor remains within the parity gate.
5. **Smoothing and linear algebra**: Rayon parallelizes the row kernels and the fixed-size numerical work; the vendored `faer` backend handles the required eigendecomposition without a Python/SciPy dependency.
6. **Randomness and output**: NumPy's legacy MT19937 stream and shuffle consumption are reproduced, and output floats use Python-compatible formatting. Repeated runs therefore produce byte-identical result files under the default thread configuration.
7. **Plotting**: `-dpl` is accepted for command-line compatibility, but this release intentionally produces the result table and log only. Use the upstream plotting helpers or your own analysis code for figures.

The detailed stage-level comparison and scope boundaries are in [`docs/DIFFERENCES.md`](docs/DIFFERENCES.md).

## Accuracy and parity check

The official vignette anchor is included as a small expected-output fixture at `bench/fixtures/official_vignette_results.txt`. After running the same input with both implementations, compare the files with:

```bash
python3 bench/parity_check.py \
  bench/fixtures/official_vignette_results.txt \
  cytotrace2_results/cytotrace2_results.txt \
  --atol 1e-6 --label cytotrace2-fast
```

The gate requires the same cell set, maximum absolute error <= `1e-6` for the three numeric columns, exact equality for both potency columns, and Spearman correlation >= `1 - 1e-9`.

Verified on the official vignette anchor with `-sp mouse --seed 14`:

| Check | Result |
| --- | ---: |
| `CytoTRACE2_Score` max absolute difference | `1.72e-09` |
| `CytoTRACE2_Relative` max absolute difference | `2.69e-09` |
| `preKNN_CytoTRACE2_Score` max absolute difference | `0` |
| Exact `CytoTRACE2_Potency` match | `100%` |
| Exact `preKNN_CytoTRACE2_Potency` match | `100%` |
| Spearman correlation of score | `1.0` |

The 10,000-cell synthetic stress input showed a systematic numeric gap of `7.1e-06` in the score column while potency stayed at `100%` and Spearman was `0.9999999996`. This is a known large-graph floating-point reduction difference between the two implementations; the official vignette anchor remains the release parity reference.

## Measured speed and memory

Measurements were made on WSL2 with 32 logical cores and 94 GB RAM. The official side was `cytotrace2-py 1.1.0.4` with cached models; each number is the median of at least three runs. Official timings include interpreter startup and the default plotting path. Synthetic rows are scaling diagnostics, not biological benchmark data.

| Input | Official Python | cytotrace2-fast v1.2.0 | Speed result | Peak RSS (official -> fast) |
| --- | ---: | ---: | ---: | ---: |
| Official mouse pancreas, 2,850 cells | `62.84 s` | **`4.49 s`** | **`14.0x`** | `5.63 -> 2.98 GB` |
| Synthetic, 10,000 cells | `162.4 s` | **`84.9 s`** | **`1.91x`** | `15.7 -> 10.0 GB` |
| Synthetic, 50,000 cells | killed at 40 min | **`444 s`** | official did not finish | `49.7 -> 23.6 GB` |
| Synthetic, 200,000 cells | not feasible (>200 GB estimated) | **`1,790.6 s`** | fast path completed | `- -> 56.9 GB` |

The benchmark protocol and the machine-readable table are in [`docs/DIFFERENCES.md`](docs/DIFFERENCES.md) and [`bench/benchmark_results.csv`](bench/benchmark_results.csv). To time your own input three times and run the parity gate, use:

```bash
scripts/run_bench.sh expression_data.txt cell_annotations.txt official_results.txt
```

## Assets and provenance

`assets/MANIFEST.json` records the SHA-256 and tensor metadata for every runtime asset. `scripts/export_assets.py` can re-create and verify the assets from an installed official `cytotrace2-py` environment; it is a packaging utility, not an alternative calculation implementation. The checked-in assets were verified byte-for-byte against the exported reference bundle.

## Citation and license

This is a derivative reimplementation of CytoTRACE 2 by the [Newman Lab at Stanford](https://github.com/digitalcytometry). If you use it, cite:

> Improved reconstruction of single-cell developmental potential with CytoTRACE 2. *Nature Methods*, 2025. Kang M, Gulati GS, Brown EL, Qi Z, Avagyan S, Almagro Armenteros JJ, Gleyzer R, Zhang W, Steen CB, D'Silva JP, Schwab J, Clarke MF, Chaudhuri AA, Newman AM. doi:10.1038/s41592-025-02857-2.

The upstream Stanford Non-Commercial Software License Agreement applies to the method, derived implementation, pretrained model assets, and reference data. Use is restricted to non-commercial purposes. See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md). The vendored `faer` backend is MIT-licensed; its upstream notice is retained in [`vendor/faer/README.md`](vendor/faer/README.md).
