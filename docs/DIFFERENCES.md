# Official CytoTRACE 2 versus cytotrace2-fast

This note records the measured computational differences behind the public release. The model architecture, pretrained weights, feature list, gene mappings, potency labels, smoothing semantics, and output schema are inherited from the official CytoTRACE 2 implementation. The changes are an implementation replacement for the same pipeline.

## Pipeline comparison

| Stage | Official implementation | cytotrace2-fast v1.2.0 | Effect |
| --- | --- | --- | --- |
| Process startup | Python, PyTorch, scanpy, SciPy, and plotting imports | One Rust executable | Removes interpreter and plotting startup from the critical path |
| Runtime model format | 19 PyTorch state dictionaries | Lossless `.npz` exports with a checked-in SHA-256 manifest | No PyTorch dependency at run time |
| Matrix input | Python text loading and intermediate arrays | Fixed-offset reads, bounded validation/re-read, and blocked transpose | Reads large files once while avoiding a dense duplicate |
| CPM/log/rank/QC | Separate full-array passes | One fused row sweep; exact-zero rows use an analytic tie rank | Less memory traffic |
| Ensemble forward | Python/Torch tensor calls per model and cell batch | Cell tiles of 16, fused layer sweeps, Rayon workers | The largest vignette stage is reduced from about 2.42 s to about 1.37 s |
| Background graph | Dense intermediate followed by CSR construction | Direct COO-to-CSR construction | Removes an approximately 815 MB dense intermediate |
| Smoothing and EVD | Python/SciPy/scanpy path | Rust row kernels and vendored `faer` EVD | Parallel native kernels without Python |
| Randomness | NumPy legacy MT19937 and its exact consumption order | Compatible MT19937, Fisher-Yates and resampling sequence | Preserves cell membership, bins, and potency labels |
| Plotting | Default Python plotting path | `-dpl` is parsed for compatibility; no figures are written | Plotting is intentionally out of scope for this binary |

The Rust implementation deliberately keeps `f64` accumulation and the official RNG consumption order. The campaign measured that changing the ensemble to `f32` produced a score drift of `1.6e-2`, while changing only RNG consumption produced a `0.127` score shift and 64 potency-bin flips. Those shortcuts are not used in this release.

## Official-anchor parity

The reference is the official mouse-pancreas vignette (2,850 cells, default `--seed 14`). The packaged output is compared with `bench/parity_check.py`.

| Metric | Measured result |
| --- | ---: |
| Score maximum absolute difference | `1.72e-09` |
| Relative maximum absolute difference | `2.69e-09` |
| pre-KNN score maximum absolute difference | `0` |
| CytoTRACE2 potency exact match | `100%` |
| pre-KNN potency exact match | `100%` |
| Score Spearman correlation | `1.0` |
| Repeated output SHA-256 | `6741245a...` (same for three runs) |

The parity gate's tolerance is `1e-6`. The official anchor is the release gate because it is the published vignette and the expected-output fixture is stable.

## Performance measurements

The measurements below use WSL2, 32 logical cores, 94 GB RAM, cached official models, and medians of at least three runs. The official command included its interpreter startup and default plotting path. Synthetic inputs are column-replicated diagnostic matrices with small perturbations; they are used only to show scaling.

| Dataset | Official wall | Fast wall | Fast/official | Official peak RSS | Fast peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vignette, 2,850 cells | 62.84 s | 4.49 s | 14.0x | 5.63 GB | 2.98 GB |
| Synthetic, 10,000 cells | 162.4 s | 84.9 s | 1.91x | 15.7 GB | 10.0 GB |
| Synthetic, 50,000 cells | killed at 40 min | 444 s | official did not finish | 49.7 GB at kill | 23.6 GB |
| Synthetic, 200,000 cells | estimated infeasible (>200 GB) | 1,790.6 s | fast path completed | — | 56.9 GB |

At 10,000 cells the fast output remains deterministic, but the score difference against the official run grows to `7.1e-06`; potency remains `100%` identical and Spearman is `0.9999999996`. This is a known large-graph reduction-order difference between Torch/SciPy and the Rust/f64 path, rather than run-to-run noise.

## Reproducing the measurements

Build the release binary with the CPU's native instruction set:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Run the benchmark helper against an expression matrix, annotation file, and an official result file:

```bash
scripts/run_bench.sh expression_data.txt cell_annotations.txt official_results.txt
```

The helper writes three output directories, wall/RSS logs, output hashes, and a JSON parity report under `runs_bench/`. The checked-in `bench/benchmark_results.csv` is the source table for the release numbers above.

## Scope boundaries

- This repository contains the single best v1.2.0 Rust implementation only. Earlier candidates, profiling harnesses, and the original Python calculation source are intentionally absent.
- The binary writes `cytotrace2_results.txt`; it does not recreate upstream plot files.
- Default Rayon scheduling gives byte-identical output. Changing `RAYON_NUM_THREADS` can change the last floating-point bits while remaining within the measured tolerance.
- Human gene mapping is implemented; the published parity anchor is the mouse vignette.
- The upstream Stanford non-commercial license and attribution apply to the derived implementation and bundled model assets.
