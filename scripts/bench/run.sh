#!/usr/bin/env bash
# Per-stage performance benchmark for the PDF pipeline (Phase 4).
#
# Emits one row per (fixture, config) with the stage breakdown that
# `DOCLING_RS_TIMING=1` accumulates, so a run on a GPU box can be diffed
# against the committed CPU baseline in docs/PERFORMANCE.md.
#
#   scripts/bench/run.sh                      # every config the host supports
#   scripts/bench/run.sh cpu cuda             # only these
#   BENCH_RUNS=3 scripts/bench/run.sh         # medians over N runs (default 3)
#
# GPU utilization is the diagnostic that decides whether a serving runtime
# would help at all (see performance-plan.md §4.6): a decoder stage that
# dominates wall clock at *low* utilization is launch-bound and is the case for
# vLLM / TensorRT-LLM; at high utilization it is compute-bound and wants a
# smaller model instead. Sampled here whenever nvidia-smi is present.
set -uo pipefail
cd "$(dirname "$0")/../.."

RUNS="${BENCH_RUNS:-3}"
OUT="${BENCH_OUT:-bench-results}"
mkdir -p "$OUT"

# Fixtures: each one exercises a different stage. Paths that are absent are
# skipped with a note rather than failing the run — a contributor checkout has
# the corpus, a GPU box may not have the large external scans.
FIXTURES=(
  "digital_paper:tests/data/pdf/sources/2206.01062.pdf::"
  "two_column:${BENCH_GFS:-}::"
  "true_scan:${BENCH_SCAN:-}::"
  "scan_formulas:${BENCH_MATH:-}:--enrich-formula:"
  "tables:tests/data/pdf/sources/redp5110_sampled.pdf::"
)

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(cpu)
  command -v nvidia-smi >/dev/null 2>&1 && CONFIGS+=(cuda tensorrt)
fi

BIN=target/release/docling-rs
[ -x "$BIN" ] || cargo build --release -q -p docling-cli || exit 1

# Median of a whitespace-separated list of numbers.
median() { tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }

printf '%-16s %-10s %10s %10s  %s\n' fixture config "wall(s)" "gpu(%)" "top stages"
for cfg in "${CONFIGS[@]}"; do
  for entry in "${FIXTURES[@]}"; do
    IFS=':' read -r name path extra _ <<< "$entry"
    [ -n "$path" ] && [ -f "$path" ] || { printf '%-16s %-10s %10s  (fixture not present — set BENCH_* to point at it)\n' "$name" "$cfg" "skip"; continue; }

    times=""
    log="$OUT/${name}.${cfg}.log"
    : > "$log"
    for _ in $(seq "$RUNS"); do
      gpulog="$OUT/${name}.${cfg}.gpu"
      if [ "$cfg" != cpu ] && command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$gpulog" 2>/dev/null &
        SAMPLER=$!
      else
        SAMPLER=""
      fi
      start=$(date +%s.%N)
      env DOCLING_RS_TIMING=1 \
          ${cfg:+DOCLING_RS_EP=$cfg} \
          "$BIN" --to json $extra "$path" >/dev/null 2>>"$log"
      end=$(date +%s.%N)
      [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null
      times="$times $(echo "$end $start" | awk '{printf "%.2f", $1-$2}')"
    done

    wall=$(echo "$times" | median)
    gpu="-"
    [ -s "$OUT/${name}.${cfg}.gpu" ] && gpu=$(awk -F, '{s+=$1; n++} END{if(n) printf "%.0f", s/n}' "$OUT/${name}.${cfg}.gpu")
    # The three costliest stages of the last run, from the timing table.
    # Stage lines read "  name   12345.6 ms   72.8%  (11 calls)" — keep name + share.
    top=$(awk '/timing \(per stage/{f=1;next} /TOTAL/{f=0} f && NF' "$log" | head -3 | awk '{printf "%s %s  ", $1, $4}')
    printf '%-16s %-10s %10s %10s  %s\n' "$name" "$cfg" "$wall" "$gpu" "$top"
  done
done

echo
echo "logs in $OUT/  ·  stage tables are the DOCLING_RS_TIMING output in each .log"
echo "commit the table into docs/PERFORMANCE.md so regressions are diffable"
