#!/usr/bin/env bash
# Run three timed executions of cytotrace2 plus the output parity gate.
#
# Usage:
#   scripts/run_bench.sh <expression.txt> <annotation.txt> <official_baseline_results.txt> [outdir_root]
#
# Environment overrides:
#   BIN          path to the cytotrace2 binary (default: <repo>/target/release/cytotrace2)
#   SPECIES      mouse|human (default mouse)      SEED  (default 14)
#   N_RUNS       timed repetitions (default 3)
#
# Produces, under [outdir_root] (default: ./runs_bench):
#   run_r1/…/run_rN/cytotrace2_results.txt   run_rN.stdout.txt   run_rN.time_v.txt
#   parity.json        parity gate output vs the official baseline
#   bench_summary.txt  per-run wall time / peak RSS / output sha256

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXPR="${1:?usage: run_bench.sh <expression.txt> <annotation.txt> <official_baseline_results.txt> [outdir_root]}"
ANNO="${2:?missing annotation path}"
BASELINE="${3:?missing official baseline results path}"
ROOT="${4:-$REPO/runs_bench}"
BIN="${BIN:-$REPO/target/release/cytotrace2}"
SPECIES="${SPECIES:-mouse}"
SEED="${SEED:-14}"
N_RUNS="${N_RUNS:-3}"

[ -x "$BIN" ] || { echo "binary not found/executable: $BIN (build first: RUSTFLAGS='-C target-cpu=native' cargo build --release)" >&2; exit 1; }

mkdir -p "$ROOT"
SUMMARY="$ROOT/bench_summary.txt"
: > "$SUMMARY"

for i in $(seq 1 "$N_RUNS"); do
  OUT="$ROOT/run_r$i"
  echo "=== run $i -> $OUT"
  /usr/bin/time -v -o "$ROOT/run_r${i}.time_v.txt" \
    "$BIN" -f "$EXPR" -a "$ANNO" -sp "$SPECIES" --seed "$SEED" -o "$OUT" \
    > "$ROOT/run_r${i}.stdout.txt" 2> "$ROOT/run_r${i}.stderr.txt"
  WALL=$(awk -F': ' '/Elapsed \(wall clock\) time/ {print $2}' "$ROOT/run_r${i}.time_v.txt")
  RSS=$(awk -F': ' '/Maximum resident set size/ {print $2}' "$ROOT/run_r${i}.time_v.txt")
  SHA=$(sha256sum "$OUT/cytotrace2_results.txt" | cut -d' ' -f1)
  echo "run_r$i  wall=$WALL  peak_rss_kb=$RSS  sha256=$SHA" | tee -a "$SUMMARY"
done

echo "=== parity gate vs $BASELINE"
python3 "$REPO/bench/parity_check.py" "$BASELINE" "$ROOT/run_r1/cytotrace2_results.txt" \
  --atol 1e-6 --label run_bench --json "$ROOT/parity.json"

echo "summary written to $SUMMARY"
