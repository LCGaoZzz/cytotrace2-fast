# cytotrace2-fast

`cytotrace2-fast` is a single-binary Rust implementation of [CytoTRACE 2](https://github.com/digitalcytometry/cytotrace2). It preserves the model assets and result-table interface while replacing the Python/PyTorch runtime. It does not generate the upstream plots.

## v1.3.0 — the large-scale breakthrough

**v1.3.0 is not a micro-optimization. It removes the dominant asymptotic bottleneck that made the previous Rust implementation collapse at large prediction-batch sizes.**

The old Rust PCA path did this:

```text
centered expression Xc
        ↓
materialize G = Xc · Xcᵀ        O(n²) memory
        ↓
full symmetric eigendecomposition O(n³) work
        ↓
discard almost everything
        ↓
keep only the top 30 PCs
```

v1.3.0 instead does this:

```text
centered expression Xc
        ↓
apply G only as Gv = Xc · (Xcᵀ · v)
        ↓
matrix-free Lanczos
        ↓
solve only the top 30 Ritz pairs
        ↓
explicitly certify residuals
```

That changes the PCA stage from **building a dense cell×cell Gram matrix and solving the full spectrum** to **computing only the 30 directions that CytoTRACE 2 actually consumes**. No model is retrained, no biological feature space is changed, and the KNN stage still receives a 30-dimensional PCA embedding.

### What that meant in practice

Historical many-core measurements on a 224-thread Xeon Platinum 8480C:

| Input | Previous Rust v1.2.0 | v1.3.0 pre-hardening candidate | Improvement |
| --- | ---: | ---: | ---: |
| Synthetic 10k | 85.2 s | 9.53 s | **8.9×** |
| Synthetic 50k, v1.2 with `TRI_GROUPS=224` | 2,818.23 s | 31.14 s | **90.5×** |
| Real human 265,480 × 19,697, `bs=50000` | 28,216 s (7 h 50 m) | 148.13 s | **~190×** |

The scaling pattern is the important part: the improvement becomes enormous exactly where the old full-spectrum eigendecomposition becomes dominant.

The memory story is equally fundamental. For a prediction batch of roughly 44k cells, a single dense `f64` Gram matrix is already about **15.7 GiB** before eigensolver copies/workspace and the expression matrices are counted. v1.3.0 removes that PCA-side `n×n` object entirely. In the archived 265k campaign the new candidate peaked at **64.4 GiB**; an interrupted old-version twin had been sampled around **106.7 GiB**, although those are not standardized matched peak measurements and no exact memory-reduction percentage is claimed.

This is why v1.3.0 can be **both dramatically faster and lower-memory**: it is not trading memory for speed. It stops computing and storing a full eigensystem that downstream CytoTRACE 2 never uses.

### Why the numerical result stays essentially unchanged

Both paths target the same mathematical object: the leading 30-dimensional PCA subspace. The difference is whether the program computes the whole spectrum first or solves the needed extremal subspace directly.

Historical old-Rust vs new-Rust parity:

- **50k:** final Score max|Δ| = `1.6e-10`, Spearman = `1.0`; preKNN is bit-identical.
- **265k real input:** final Score max|Δ| = `1.33e-15`, Relative max|Δ| = `2.78e-15`, Spearman = `1.0`; preKNN max|Δ| = `0` over 265,474 finite entries.
- Repeated candidate runs were byte-identical to each other.
- The finalized solver now additionally certifies every returned Ritz pair with
  `||G·u - λu|| / (λmax·||u||) <= 1e-10` and fails closed if convergence is not demonstrated.

So the speedup does **not** come from reducing the model ensemble, changing the biological method, lowering floating-point precision, skipping KNN smoothing, or accepting a loose approximate answer. It comes from fixing the linear-algebra strategy.

> **Important scope:** the headline **~190× is v1.2.0 Rust → v1.3.0 Rust** on the archived 265k real-data campaign. It is not a claim that v1.3.0 is 190× faster than the official Python CytoTRACE 2. The official Python implementation already uses truncated top-k PCA; v1.3.0 fixes a complexity regression in the previous Rust reimplementation.

The archived timing numbers above describe the **pre-hardening candidate at `aebce88`**. The final v1.3.0 solver adds explicit residual certification and has not been re-benchmarked at 265k, so `148.13 s` is retained as historical provenance rather than presented as a fresh timing of the hardened HEAD. See [measurement provenance](docs/DIFFERENCES.md) and [release validation](docs/RELEASE_VALIDATION.md).

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
# cytotrace2-fast 1.3.1
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
| `-mc` | `--max-cores` | automatic | Global Rayon worker cap; also bounds prediction/KNN pools |
| `-dpl` | `--disable-plotting` | off | Parsed; no figures are written either way |
| `-dv` | `--disable-verbose` | off | Suppress parameter summary |
| | `--assets` | auto | Asset directory |
| | `--dump-stage DIR` | off | Intermediate arrays for debugging |
| | `--version` | | Version from `Cargo.toml` |

### CPU/thread limiting (v1.3.1)

`--max-cores N` now configures Rayon's **global** worker pool before any parallel work starts. This closes a real gap in v1.3.0: prediction/KNN-local pools respected the CLI cap, while preprocessing, PCA and vendored faer paths could still initialize/use the machine-wide Rayon pool. `--disable-parallelization` now forces the same global pool to one worker.

When `--max-cores` is omitted, the program leaves Rayon's normal environment-based configuration untouched, so `RAYON_NUM_THREADS` continues to work as before.

This is an **in-process Rayon limit**, not an operating-system CPU-affinity boundary. It does not hide CPUs from unrelated Python/native libraries that call their own `cpu_count()`, and the asset loader still uses a small number of explicit standard threads for I/O/model loading. If a workflow needs a hard CPU allocation across mixed libraries, enforce affinity/cgroups at the runner level (for example `taskset`) in addition to library-specific thread variables.

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
