# cytotrace2-fast

`cytotrace2-fast` is a single-binary Rust implementation of [CytoTRACE 2](https://github.com/digitalcytometry/cytotrace2). It preserves the model assets and result-table interface while replacing the Python/PyTorch runtime. It does not generate the upstream plots.

**v1.3.0** replaces the per-prediction-batch full Gram eigendecomposition with matrix-free top-30 Lanczos PCA and removes the four-group tridiagonalization default. The finalized solver checks numerical residuals and fails explicitly on invalid or nonconverged input; it never silently falls back to an expensive full decomposition.

The historical development campaign reported **28,216 s → 148.13 s** on one 265,480-cell human input on a 224-thread server, comparing the old and new **Rust implementations**, not the official Python package. Those timings describe the **pre-hardening candidate represented by `aebce88`**, before the residual checks were added. They are not measurements of the finalized HEAD. See [measurement provenance](docs/DIFFERENCES.md) and [release validation](docs/RELEASE_VALIDATION.md).

CytoTRACE 2's biological method, pretrained models, feature space, and mapping tables are the upstream project's work. No model is retrained in this repository.

## What it predicts

CytoTRACE 2 predicts developmental potential and potency categories from single-cell expression data. The output columns are:

```text
CytoTRACE2_Score
CytoTRACE2_Potency
CytoTRACE2_Relative
preKNN_CytoTRACE2_Score
preKNN_CytoTRACE2_Potency
```

For biological interpretation, see the [upstream documentation](https://github.com/digitalcytometry/cytotrace2) and Kang et al., *Nature Methods* (2025), doi:[10.1038/s41592-025-02857-2](https://doi.org/10.1038/s41592-025-02857-2). Shared missing values are not valid predictions and must remain visible in downstream quality checks.

## Installation

Build with a recent stable Rust toolchain on Linux, macOS, or WSL2. The binary does not require Python, PyTorch, R, or network access at run time.

```bash
git clone https://github.com/LCGaoZzz/cytotrace2-fast.git
cd cytotrace2-fast
RUSTFLAGS="-C target-cpu=native" cargo build --release --locked
./target/release/cytotrace2 --version
# cytotrace2-fast 1.3.0
```

Omit `target-cpu=native` when building for a different CPU. `target/` is a local build directory, not part of the repository. The checked-in `assets/` directory is found from the working directory or the standard Cargo release layout. Alternatively use `CYTOTRACE2_ASSETS=/path/to/assets` or `--assets /path/to/assets`.

## Running CytoTRACE 2

```bash
./target/release/cytotrace2 \
  -f expression_data.txt \
  -a cell_annotations.txt \
  -sp mouse \
  --seed 14 \
  -o cytotrace2_results
```

The expression file is a tab- or comma-separated matrix with genes in rows, cells in columns, and gene names in the first column. Use raw counts or CPM/TPM, not log-transformed values. The annotation argument is parsed for compatibility but is not read by this implementation. The output is `cytotrace2_results/cytotrace2_results.txt`.

| Short | Long | Default | Description |
| --- | --- | --- | --- |
| `-f` | `--input-path` | required | Expression matrix |
| `-a` | `--annotation-path` | empty | Parsed compatibility argument |
| `-sp` | `--species` | `mouse` | `mouse` or `human` |
| `-bs` | `--batch-size` | `10000` | Prediction/PCA batch size |
| `-sbs` | `--smooth-batch-size` | `1000` | Diffusion smoothing batch size |
| `-r` | `--seed` | `14` | NumPy-compatible pipeline MT19937 seed |
| `-o` | `--output-dir` | `cytotrace2_results` | Output directory |
| `-dpa` | `--disable-parallelization` | off | Upstream-compatible parallelization flag |
| `-mc` | `--max-cores` | automatic | Prediction/KNN worker setting |
| `-dpl` | `--disable-plotting` | off | Parsed; no figures are written either way |
| `-dv` | `--disable-verbose` | off | Suppress parameter summary |
| | `--assets` | auto | Asset directory |
| | `--dump-stage DIR` | off | Intermediate arrays for debugging |
| | `--version` | | Version from `Cargo.toml` |

Prediction and diffusion batch sizes are separate. Large inputs should keep bounded prediction batches (defaults, or `--batch-size 50000`) and bounded diffusion batches (default `1000`). The companion recipe retains its >30,000-cell whole-input-single-batch guard as a conservative support boundary. It is not a claim that prediction batch size alone always allocates a full-dataset diffusion matrix. KNN still computes all-pairs distances within a prediction batch; removing the PCA Gram matrix does not make the entire pipeline linear.

## Numerical implementation and validation

The pipeline keeps the 19 pretrained model exports, gene mappings, preprocessing, upstream RNG stream, diffusion, binning, and final KNN rule. Lanczos computes `Gv = Xc * (Xc^T * v)` without storing `G`. Two-pass reorthogonalization, fixed reduction order, a separate fixed `MT19937(14)` PCA start, and a sign convention support repeatability. The PCA seed does not replace or consume the user-controlled pipeline RNG.

The finalized solver requires an estimated residual at its candidate stopping point and explicitly checks every returned Ritz vector with `||G*u - lambda*u|| / (lambda_max * ||u||) <= 1e-10`. The iteration limit is `min(n, 300)`. A limit reached without certification, non-finite data, or a degenerate zero spectrum produces a nonzero exit rather than a result. These are numerical checks, not a universal guarantee of biological accuracy or of identical neighbors for arbitrarily near-tied spectra.

`C2RUST_PCA=full` explicitly requests the legacy full-EVD path; it is not an automatic fallback and may need quadratic memory. `C2RUST_TRI_GROUPS=<n>` overrides the tridiagonalization scheduling width. `C2RUST_TIME_SUB=1` includes residual and iteration diagnostics.

Run lightweight tests without model inference or large benchmark data:

```bash
python3 -m unittest discover -s bench -p 'test_*.py' -v
cargo test --locked --bin cytotrace2
cargo fmt --check
```

For two result files with matching input, seed, and batch settings:

```bash
python3 bench/compare_two.py reference.txt candidate.txt \
  --atol 1e-6 --rtol 0 --min-spearman 0.999999999
# Add --byte-check only for a separate exact repeatability check.
```

This gate requires matching schema, unique cell IDs in the same order, matching missing masks, finite-value agreement, and exact potency strings. Empty fields and `nan` are missing, never silently treated as zero. Mismatches and invalid inputs exit nonzero. The JSON report records hashes, compared counts, missing counts, tolerances, and failures. Undefined Spearman is `null`; a requested correlation threshold then fails. No-finite-data comparisons fail.

The official vignette fixture and its existing helper remain available:

```bash
python3 bench/parity_check.py \
  bench/fixtures/official_vignette_results.txt \
  cytotrace2_results/cytotrace2_results.txt --atol 1e-6
```

Historical vignette agreement was Score max absolute difference `1.72e-9`, Relative `2.69e-9`, preKNN `0`, both potency columns `100%`, and Spearman `1.0`. A 10k synthetic comparison to official Python instead had Score difference `7.1e-6`, potency `100%`, and Spearman `0.9999999996`. Do not generalize the vignette tolerance to all inputs. Historical official-anchor and 265k comparisons have **not been rerun on the residual-checked HEAD**; the small numerical regression tests and historical evidence are distinguished in the release-validation note.

## Historical speed and memory

The original v1.2.0 campaign on WSL2 (32 logical cores, 94 GB RAM) reported 4.49 s versus 62.84 s for the official 2,850-cell vignette, with 2.98 versus 5.63 GB reported peak RSS. Official timing included interpreter startup and plotting; the fast binary did not plot. This is not a plotting-matched algorithm-only benchmark.

The many-core campaign used a 224-thread Xeon Platinum 8480C, approximately 503 GB RAM, and native-CPU builds. The following are archived reports for the pre-hardening candidate, not new final-HEAD measurements:

| Input | v1.2.0 default (s) | v1.2.0 groups=224 (s) | Pre-hardening candidate (s) | Candidate statistic | Candidate RSS (GiB, harness) |
| --- | ---: | ---: | ---: | --- | ---: |
| Vignette, 2,850 | 9.0 | 13.53 | 7.02 | Median of 3 | 2.52 |
| Synthetic, 10,000 | 85.2 | 69.16 | 9.53 | Median of 3 | 6.22 |
| Synthetic, 50,000, bs=50000 | — | 2818.23 | 31.14 | Median of 3 | 29.0 |
| Synthetic, 100,000 | — | — | 60.76 | Single diagnostic run | 31.22 |
| Synthetic, 250,000 | — | — | 152.77 | Single diagnostic run | 36.90 |
| Human, 265,480 × 19,697, bs=50000 | 28216 | — | 148.13 | Median of 3 | 64.4 |

The 265k old reference is **one completed production run** on September 19, 2026; it is not a three-run paired benchmark in the same frozen-load window. Its result was reused instead of rerunning it. Approximately 106.7 GiB was sampled during an interrupted old-version twin; that is not a standardized peak for the completed reference. No precise cross-version RSS-reduction percentage is claimed. The unmeasured 50k default extrapolation is excluded from the measured table.

Machine-readable records are separate to avoid mixing implementations or seconds with speedup ratios: [official-vs-v1.2.0](bench/benchmark_results.csv) and [many-core archived campaign](bench/manycore_results.csv). Further context and limitations are in [DIFFERENCES.md](docs/DIFFERENCES.md).

## Assets and provenance

`assets/MANIFEST.json` records SHA-256 hashes and tensor metadata. `scripts/export_assets.py` recreates/verifies exports from an installed official `cytotrace2-py` environment; it is a packaging utility, not a second analysis implementation. Model assets were not changed by this PR.

## Citation and license

This is a derivative reimplementation of CytoTRACE 2 by the [Newman Lab at Stanford](https://github.com/digitalcytometry). Cite:

> Improved reconstruction of single-cell developmental potential with CytoTRACE 2. *Nature Methods*, 2025. Kang M, Gulati GS, Brown EL, Qi Z, Avagyan S, Almagro Armenteros JJ, Gleyzer R, Zhang W, Steen CB, D'Silva JP, Schwab J, Clarke MF, Chaudhuri AA, Newman AM. doi:10.1038/s41592-025-02857-2.

The upstream Stanford Non-Commercial Software License Agreement applies to the derived implementation, pretrained model assets, and reference data. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). The vendored `faer` notice remains in [vendor/faer/README.md](vendor/faer/README.md).
