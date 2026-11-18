#!/usr/bin/env bash
# Replay OpenDataLoader's synthetic page cases (tests/data/pdf/opendataloader/)
# against this build — the reading-order and list-structure assertions of its
# issue tests plus a word recall against its own Markdown. Needs `.models/`.
#
#   scripts/conformance/opendataloader.sh            # the Rust harness, release build
#   scripts/conformance/opendataloader.sh --report   # also a Markdown report of every case
#
# `cases.json` pins what holds today; the harness fails on a regression, on an
# `xfail` that now passes (update its status), and on a recall drop. To refresh
# the file after a change, use --report: it converts every fixture with the CLI
# into a scratch directory and reruns the Python evaluator, which rewrites
# cases.json with the new statuses — review the diff before committing it.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "${1:-}" == "--report" ]]; then
    out="$(mktemp -d)"
    cargo build --release -p docling-cli
    ./target/release/docling-rs --input "tests/data/pdf/opendataloader/**/*.pdf" \
        --output "$out" --to md,reader --jobs 4 --no-progress
    python3 -I scripts/conformance/opendataloader_cases.py evaluate \
        tests/data/pdf/opendataloader/cases.json "$out" tests/data/pdf/opendataloader \
        tests/data/pdf/opendataloader/cases.json "$out/report.md"
    echo "report: $out/report.md"
    exit 0
fi

DOCLING_RS_ODL_CORPUS=1 cargo test --release -p docling --test opendataloader -- --nocapture
