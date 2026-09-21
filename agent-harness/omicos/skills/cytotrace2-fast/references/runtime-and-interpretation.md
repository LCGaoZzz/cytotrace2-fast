# Runtime, scaling and interpretation

## Runtime and resource allocation

Prepare the binary once using the repository build instructions (`cargo build
--release --locked`). Supply its actual executable path and the original
`assets/` directory. The copied Skill requires the installed binary/assets, not
a Python cytotrace2 package or a checkout-relative import. The helper does not
bundle/relicense the original models; the repository's Stanford Non-Commercial
Software License and upstream attribution continue to apply.

The runner rejects versions below 1.3.1 and the upstream Python CLI, and checks
required model/mapping files exist. It explicitly passes `--max-cores` and sets
child `RAYON_NUM_THREADS` to the same allocation. In v1.3.1 this caps the global
Rayon worker pool, **not all OS threads**. The Rust loader still uses explicit
I/O/model threads. It is not a CPU-affinity boundary for other libraries or jobs.
Use the host's allocation/cgroup/affinity policy for a hard CPU boundary. Choose
CPU IDs from the process's actual allowed cpuset, not a hard-coded `0-63` range.
No parent-process environment or machine-wide thread settings are changed.

Prediction batch size and diffusion smoothing batch size are distinct. Defaults
are 10000 and 1000. Preserve the user's settings; reducing a batch can change
predictions and is not a semantics-free memory fix. The adapter refuses a
whole-input single batch over 30000 cells as a conservative support boundary,
not a theorem that every larger prediction batch is invalid. Bounded batches
such as 50000 within a larger input are permitted, subject to actual resources.
KNN still performs all-pairs distance work within prediction batches. Removing
the PCA Gram matrix does not make every pipeline stage linear-memory/time.

The main path uses the Rust solver without adding fallback algorithms. Existing
`C2RUST_*` settings are inherited and recorded. In particular, `C2RUST_PCA=full`
is an explicit legacy full-EVD override with potentially quadratic memory costs;
it must not be introduced silently after a numerical error. Check recorded
environment variables when results or memory use differ unexpectedly.

`timeout_seconds` terminates the child, waits, and force-kills after a grace
period when necessary. Interactive interruption and SIGTERM also clean up the child.
SIGKILL/host crashes cannot be handled; use the Omicos job runner and inspect
logs/state on recovery. No custom daemon, scheduler or MCP server is introduced.

## Interpretation boundaries

The repository describes CytoTRACE2 as prediction of developmental potential
and potency categories; it does not generate upstream plots. Read the upstream
CytoTRACE2 documentation and Kang et al., *Nature Methods* (2025),
DOI `10.1038/s41592-025-02857-2`, for biological interpretation. Higher score is
not by itself evidence of a traced ancestor, a selected root or a causal transition.
Keep final score, preKNN score, potency and within-analysis relative scaling
separate. A relative score should not be treated as a common absolute calibration
across independently analyzed subsets.

**Transfer guidance, not a validated new workflow:** do not assume a targeted
Xenium panel is interchangeable with broad scRNA-seq expression. Inspect gene
coverage and assay characteristics, and use matched broad-expression data or
independent evidence to assess applicability. Spatial localization of a score
does not establish ancestry, progression time or a causal niche mechanism.

Reuse existing UMAP/spatial coordinates, join scores by literal cell ID, and show
missing predictions separately. Useful diagnostics are final versus preKNN
score, missingness by sample, score distributions within comparable cell types,
and sensitivity to justified input choices. Group comparisons need biological
replicates and sample/donor-aware analysis; cells alone are not independent
experimental replicates. These are downstream analysis recommendations, not
extra computations performed by the harness.

Report the observed run, hardware allocation and timings. The repository's
historical ~190x headline compares an archived v1.3.0 candidate with old Rust
v1.2.0 on one large-data campaign; it is not a universal speedup versus official
Python or a fresh benchmark of this integration. Consult `docs/DIFFERENCES.md`
and `docs/RELEASE_VALIDATION.md` rather than manufacturing performance claims.

## Source anchors

The integration was reviewed against cytotrace2-fast base commit
`a3c6e7f7949b218edb0d13d41141b4bbf1fc881e`:

- `src/main.rs::configure_global_rayon`, `parse_args`, `default_assets_dir`:
  worker cap, flags/defaults, asset location and version output.
- `src/pipeline.rs::read_matrix_once`: native TSV header and matrix parsing.
- `src/main.rs`, output-writing block: five result columns and literal cell IDs.
- `README.md`, `docs/DIFFERENCES.md`, `docs/RELEASE_VALIDATION.md`:
  input scale, numerical/performance boundaries and historical validation scope.

Agent/Skill packaging follows iobrx commit
`5ca23a985d722ce346e79846a90e6e1b9161e179`, `agent-harness/omicos/`.
Adapter contract tests do not establish live Omicos compatibility, model
numerical parity or biological accuracy. Verify deployment in the target host.
