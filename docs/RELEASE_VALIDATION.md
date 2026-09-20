# v1.3.0 finalization: validation scope

## Source and executed checks

The hardening implementation is commit `954b85345b55b38d276cc2ab4f9fc785f8db9286`,
after the historical performance snapshot `aebce88d422c9bc7ea16d185e1e30508aa446cf3`.
[Its CI run](https://github.com/LCGaoZzz/cytotrace2-fast/actions/runs/35488653468)
actually passed the 22 strict-comparator tests, `cargo check --locked`, all 9
Rust unit tests, and the CLI version assertion. That run's remaining failure was
formatting; the follow-up finalization commit did not fully satisfy `cargo fmt`,
and the benchmark-harness closeout completes that mechanical reformat. The
current commit's **Checks** tab is authoritative for final-HEAD status; this
parent-run record must not be described as an all-green final-HEAD run.

Three metadata regression tests have additionally been executed locally, giving
25 passing Python tests. They check the Cargo/lockfile/CLI version source,
benchmark units and ratios, and historical source/statistic labels. The same
suite is now discovered by CI; with the seven harness tests below it counts 32
Python tests. No external Python packages are needed.

The 9 Rust unit tests comprise one formatting-utility test and eight new
numerical regression tests. New coverage includes:

- Full-EVD versus Lanczos distances, 30-neighbor ordering, KNN scores and potency
  on a small centered random matrix.
- Small dimensions, rank deficiency, repeated retained eigenvalues, and close
  eigenvalues on either side of the truncation boundary.
- Forced iteration exhaustion, invalid dimensions, non-finite input and zero
  spectrum rejection.
- Same-build repeatability and a one-thread/two-thread comparison.

These tests use matrices of at most 128 cells, not model inference or large
expression datasets. The current workflow also requires formatting and checks
that the executable reports `cytotrace2-fast 1.3.0` from Cargo's package version.

## Fail-closed numerical and output contracts

`src/knn.rs` checks estimated and explicit Ritz residuals at tolerance `1e-10`.
Uncertified exhaustion at `min(n,300)` steps is an error. The CLI exits nonzero;
no automatic full-Gram fallback or output imputation was introduced. Numerical
certification is not proof of universal leading-subspace or biological parity.

`bench/compare_two.py` fails on schema, cell identity/order, missing-mask,
category or finite-value mismatches. Empty and `nan` fields are missing; one-sided
missingness cannot disappear into a zero maximum delta. An entirely missing
numeric column cannot pass. It writes strict JSON with hashes and thresholds.
Byte identity is an optional separate requirement, not a synonym for tolerance
agreement. The negative cases are checked through actual subprocess exit codes.

## Benchmark harness hardening

`bench/run_one.py` was rewritten so a benchmark cannot silently lie or hang:
child stdout/stderr stream directly to files instead of undrained pipes, a
nonzero child exit propagates as the wrapper's exit code, an existing run
directory is refused instead of reused, and a missing GNU `/usr/bin/time` or
binary is an explicit error before launch. Each run records the exact command,
environment overrides, binary and result SHA-256, GNU-time wall/CPU/peak-RSS,
the child exit code, and supplementary `/proc` snapshots; a sampling failure
never aborts the run. Seven wrapper regression tests (`bench/test_run_one.py`)
drive tiny fake executables through success, child failure, missing output,
multi-megabyte output, directory reuse, label escaping, and malformed
STAGE_TIMINGS; they require GNU time and skip where it is absent. CI installs
the `time` package so they always execute there. No real model run is involved.

## Deliberately not rerun

The finalization did **not** run the official vignette end-to-end, a new human
cohort, a 265k inference, the eight-hour old reference, or a new speed/RSS
benchmark. The existing large result files were unavailable here, so their
NaN-aware comparison was not rerun either. The reported historical 20 missing
final scores remain a known behavior, not repaired predictions.

The historic 148.13-second result and candidate output SHA belong to the
pre-hardening campaign only. Explicit residual checking adds matrix products,
so they are not performance/hash claims for this final HEAD. Consult
[DIFFERENCES.md](DIFFERENCES.md) and
[the historical many-core table](../bench/manycore_results.csv) for the exact
measurement boundaries. No measurement or repeat count has been invented.

## Commands for lightweight acceptance

```bash
python3 -m unittest discover -s bench -p 'test_*.py' -v
cargo check --locked
cargo test --locked --bin cytotrace2
cargo fmt --check
test "$(cargo run --locked --quiet -- --version)" = "cytotrace2-fast 1.3.0"
```

When original result files are available, comparison alone does not rerun the
model:

```bash
python3 bench/compare_two.py OLD_RESULTS.txt NEW_RESULTS.txt \
  --atol 1e-6 --rtol 0 --min-spearman 0.999999999 \
  --output comparison.json
```

Keep input identity, seed, batch configuration, source revision, binary hash
and missing-cell report with any such comparison. Do not fill missing values or
change batches simply to make the gate pass.
