# Official CytoTRACE 2 versus cytotrace2-fast

This file separates implementation changes, archived measurements, and finalization checks. The detailed current verification scope is in [RELEASE_VALIDATION.md](RELEASE_VALIDATION.md).

## Implementation

The model architecture, pretrained weights, feature list, gene mappings, potency labels, and result schema are inherited from the official implementation. No weights or feature list changed in PR #2.

| Stage | Official Python implementation | Rust implementation |
| --- | --- | --- |
| Startup | Python/PyTorch/scanpy and plotting imports | One executable; plotting omitted |
| Models | 19 PyTorch state dictionaries | Lossless NPZ exports with checked-in hashes |
| Input | Text loading and intermediate arrays | Blocked reads/transposes; still holds dense expression data |
| Preprocessing | Separate array operations | Fused CPM/log/rank/QC row sweeps |
| Ensemble | Tensor calls per model and batch | Tiled/fused kernels with preserved accumulation order |
| Background graph | Dense intermediate then sparse construction | Direct COO-to-CSR construction |
| Diffusion/binning | Upstream smoothing and binning | Existing Rust kernels; not changed in this PR |
| Final KNN PCA | Top PCs via sklearn ARPACK | v1.2.0: full Gram EVD; v1.3.0: matrix-free Lanczos |
| Final KNN | Adaptive neighborhood averaging | Existing all-pairs distance and adaptive averaging rule |

The official Python code already selects only the top 30 PCs. The v1.3.0 change corrects the expensive full-spectrum strategy in the earlier Rust port; it is not an invention of truncated PCA and does not establish a 190x advantage over official Python.

For a prediction batch of `n` cells and `f=14271` features, Lanczos applies `Gv=Xc*(Xc^T*v)` without storing the `n*n` Gram matrix. With `m` steps, the principal work includes `O(n*f*m + n*m^2)` for products and full reorthogonalization, plus small projected eigendecompositions and embedding/residual construction. Working memory is `O(n*(f+m))` for this PCA stage. The rest of the pipeline is not thereby linear: dense input copies, diffusion batches, and within-batch all-pairs KNN remain relevant.

The vendored tridiagonalization's default task-group count is the Rayon pool width instead of four. More threads are not guaranteed to help every matrix or CPU topology. `C2RUST_TRI_GROUPS` remains an override, and `C2RUST_PCA=full` explicitly selects the memory-intensive legacy solver. There is no automatic dense fallback.

Lanczos uses a separate hardcoded `MT19937(14)` start, two-pass full reorthogonalization, fixed reduction partitions/order, and canonical signs. The user-supplied pipeline RNG remains separate. Fixed numerical order supports repeatability on a fixed build and configuration; CPU/compiler/library/thread changes are not promised to preserve every bit.

## Finalized numerical acceptance

Ritz-value stability (`1e-12`, two consecutive checks) is only a stopping heuristic. A candidate stop must also satisfy the Lanczos residual estimate. Every returned vector then undergoes an explicit matrix-free residual check:

`||G*u_j - lambda_j*u_j|| / (lambda_max * ||u_j||) <= 1e-10`.

The global leading-Ritz-value scale accommodates near-null components in rank-deficient matrices. Non-finite values, invalid dimensions, failed residuals, or an uncertified step limit return an error and the CLI exits nonzero. The limit is `min(n,300)`; reaching it is not itself success. Zero-spectrum data are rejected rather than passed to an undefined all-zero-distance KNN calculation.

Residual checks do not prove that an arbitrary difficult spectrum's entire leading invariant subspace was found, nor do they establish biological equivalence. Small tests compare distances, neighbor ordering and scores against full EVD, including rank deficiency, repeated leading eigenvalues and a near-tied truncation boundary. The exact 30/31 degeneracy has no unique rank-30 subspace; do not promise identical neighbors in that regime.

## Archived official-anchor evidence (not rerun during finalization)

The v1.2.0 official mouse-pancreas vignette report (2,850 cells, seed 14) records Score max difference `1.72e-9`, Relative `2.69e-9`, preKNN `0`, both potency matches `100%`, and Spearman `1.0`. The packaged fixture is `bench/fixtures/official_vignette_results.txt`. The same approximate gate results were reported for the pre-hardening candidate.

For the old 10k synthetic official-vs-fast comparison, Score difference reached `7.1e-6`, potency remained `100%`, and Spearman was `0.9999999996`. Thus the vignette's `1e-6` gate is not a global bound on every input. Synthetic matrices were column-replicated diagnostics with small perturbations, not independent biological cohorts.

The original WSL2 campaign (32 logical cores, 94 GB RAM) reported:

| Input | Official wall | v1.2.0 wall | Reported official / fast RSS |
| --- | ---: | ---: | --- |
| Vignette 2,850 | 62.84 s | 4.49 s | 5.63 / 2.98 GB |
| Synthetic 10k | 162.4 s | 84.9 s | 15.7 / 10.0 GB |
| Synthetic 50k | Killed at 40 min | 444 s | 49.7 GB at kill / 23.6 GB |
| Synthetic 200k | Not run to completion; >200 GB estimated | 1790.6 s | Not measured / 56.9 GB |

Completed measurements were described as medians of at least three runs; exact counts are not reconstructed here. The official CLI included startup and default plotting; the Rust binary did not plot. RSS units above preserve the legacy report's GB label; the original raw logs were not re-audited during finalization. See `bench/benchmark_results.csv`.

## Archived many-core campaign (before residual checks)

Source snapshot: `aebce88d422c9bc7ea16d185e1e30508aa446cf3`, PR #2's original description and table. The previous Rust reference was `d334c0eb017aee99b82ed07ace3f0f2ab3ead400`. The machine was a 224-thread Xeon Platinum 8480C with approximately 503 GB RAM and native-CPU builds. `bench/manycore_results.csv` retains these numbers with explicit version/configuration/statistic fields. Its candidate source snapshot identifies the historical report, not a newly measured binary hash.

The 265,480 × 19,697 human input used `bs=50000`. The 28,216-second v1.2.0 reference was one completed production run on September 19, 2026, 14:06:43–21:56:59 in the original run's recorded clock. It was reused for comparison; it was not rerun in a paired frozen-load window with the candidate. The 148.13-second candidate value was reported as the median of three runs. The quotient is approximately 190.48x **relative to old Rust**, conditional on that host/input and the historical measurement protocol.

Candidate vignette, 10k, 50k and 265k numbers were reported as three-run medians; 100k and 250k are single diagnostic runs. Repeat counts for the other old-version measurements are not invented. An unmeasured ~17,400-second 50k default extrapolation is not included in measured CSV fields. The abandoned reference run is not counted as a completed measurement.

The benchmark harness reports RSS in GiB (Linux `time -v` KiB divided by `2**20`). Its candidate 265k reported peak is 64.4 GiB. The old approximately 106.7 GiB value is an observed sample from an interrupted twin, not a standardized maximum for the completed reference. Therefore this release does not claim an exact 40% measured peak-memory reduction. The quadratic-allocation removal is a code fact; the sampled memory comparison has a weaker measurement basis.

Historical parity claims against v1.2.0:

- 10k in-process: eigenvalue relative difference up to `1.9e-14`, identical 30-neighbor sets/order.
- 50k: preKNN difference `0`, final Score difference `1.6e-10`, Spearman `1.0`.
- 265k: matching headers/order, 20 shared missing final Scores, 265,460 common final scores; preKNN difference `0` over 265,474 common finite entries; Score difference `1.33e-15`, Relative `2.78e-15`, Spearman `1.0`.
- Three pre-hardening candidate outputs were reported to share SHA-256 `10e367091ecab2bb7d70b5441a5c7e137263490567c74ef97dcaa9f63a41421c`, also matching an earlier candidate run. This is repeatability **within that candidate**, not byte identity between v1.2.0 and v1.3.0.

These are retained author-reported measurements, not newly reproduced results. The large original result files and the exact historical NaN-aware comparison procedure were not available in the finalization environment. The original `compare_two.py` could not directly process empty fields; the replacement now can, but that fact does not retroactively rerun the 265k comparison. `preKNN` lies upstream of the modified PCA and is a control for unaffected stages, not sufficient evidence of PCA correctness. Matching missing masks preserves behavior but does not repair missing biological predictions.

## Repeating a comparison without repeating model inference

When the owner has both existing result files, their comparison alone requires no rerun of CytoTRACE2:

```bash
python3 bench/compare_two.py OLD_RESULTS.txt NEW_RESULTS.txt \
  --atol 1e-6 --rtol 0 --min-spearman 0.999999999 \
  --output comparison.json
```

The report records file hashes, missing masks/counts, common finite counts, category agreement and numeric thresholds. Optional `--byte-check` is appropriate for separate repeats of the same build, not for a floating-point-tolerant old/new comparison. Do not impute or drop rows to make a gate pass.

No large inference, benchmark rerun or new performance number was added during finalization. The new explicit residual checks add matrix products; historical runtime and output hashes must not be assigned to the finalized HEAD without measurements.

## Boundaries

The companion recipe retains its >30k whole-input-single-batch refusal. Prediction and diffusion batch sizes are independent controls: a large prediction batch does not by itself mean full-dataset diffusion. Remaining quadratic diffusion behavior and all-pairs KNN justify conservative bounded-batch guidance, not an invented universal RAM formula. The binary does not recreate upstream plots. Human mapping is implemented, while the published fixture anchor is mouse. The upstream license and attribution remain unchanged.
