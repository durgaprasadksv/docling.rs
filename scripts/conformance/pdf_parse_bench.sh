#!/usr/bin/env bash
# Every PDF of pdf-parse-bench, scored element by element and checked
# against tests/data/pdf/pdf_parse_bench/expected.json (see
# scripts/conformance/pdf_parse_bench.py for the statuses).
#
#   scripts/conformance/pdf_parse_bench.sh            # convert, score, check
#   scripts/conformance/pdf_parse_bench.sh --update   # ... and accept the result
#
# Needs the layout model (.models/); the formula VLM is switched off so the
# run measures the text layer alone and stays fast and deterministic. The
# dataset (~36 MB of PDFs) is fetched once into target/pdf-parse-bench at the
# pinned commit; nothing of it is committed but the expectations and the
# formula-region fixtures (tests/data/pdf/formula_text/).
set -euo pipefail
cd "$(dirname "$0")/../.."

COMMIT=de72a161a67a8e17afc7a413142fed340af4776b
DATA=target/pdf-parse-bench
RUN=target/pdf-parse-bench-run
EXPECTED=tests/data/pdf/pdf_parse_bench/expected.json
BIN=${DOCLING_RS_BIN:-target/release/docling-rs}

if [ ! -d "$DATA/.git" ]; then
  git clone -q https://github.com/phorn1/pdf-parse-bench "$DATA"
fi
git -C "$DATA" -c advice.detachedHead=false checkout -q "$COMMIT"
[ -x "$BIN" ] || cargo build --release -p docling-cli

for set in 2026-q1-formulas-only 2026-q1-tables-only 2025-q4; do
  rm -rf "$RUN/$set"
  mkdir -p "$RUN/$set/out"
  for pdf in "$DATA/data/$set/pdfs/"*.pdf; do
    name=$(basename "$pdf" .pdf)
    mkdir -p "$RUN/$set/$name"
    DOCLING_CODE_FORMULA_DIR=/nonexistent \
      DOCLING_RS_FORMULA_TEXT_DUMP="$RUN/$set/$name" \
      "$BIN" --enrich-formula --to md,json --output "$RUN/$set/out" "$pdf" \
      >/dev/null 2>>"$RUN/$set.log"
  done
  echo "$set converted"
done

if [ "${1:-}" = "--update" ]; then
  python3 scripts/conformance/pdf_parse_bench.py "$DATA/data" "$RUN" \
    --update "$EXPECTED" --fixtures tests/data/pdf/formula_text
else
  python3 scripts/conformance/pdf_parse_bench.py "$DATA/data" "$RUN" --check "$EXPECTED"
fi
