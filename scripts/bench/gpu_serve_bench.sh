#!/usr/bin/env bash
# Throughput benchmark against a running docling-serve (#80): posts each
# PDF (optionally a page window) to `/v1/convert/file` with formula
# enrichment on, and tabulates the server's own `processing_time` and
# per-stage `timings` — the numbers a GPU deployment (RunPod serverless,
# docling-rs-serve-cuda) is sized by. Everything is measured by the server,
# so the client's network does not leak into the stages; the wall time
# column is the client's. Start the server with `DOCLING_RS_TIMING=1` and
# `--concurrency 1`, or `timings` stays empty / mixes requests.
#
# Usage:
#   scripts/bench/gpu_serve_bench.sh http://host:5001 file.pdf[:A-B] [file.pdf[:A-B] …]
#   DOCLING_BENCH_OPTS="do_formula_enrichment=true" (default) sets the form
#   options; DOCLING_BENCH_RUNS=3 repeats each file (first run = cold).
#
# Example corpus (the pages measured on CPU in docs/PDF_CONFORMANCE.md):
#   oldscannedmath.pdf:220-221 university-physics-volume-2.pdf:280-281 \
#   chemistry-2e.pdf:620-620 tests/data/pdf/sources/code_and_formula.pdf
set -euo pipefail
base="${1:?server base URL}"; shift
opts="${DOCLING_BENCH_OPTS:-do_formula_enrichment=true}"
runs="${DOCLING_BENCH_RUNS:-1}"
curl -fsS "$base/health" >/dev/null || { echo "server not healthy at $base" >&2; exit 1; }
python3 - "$base" "$opts" "$runs" "$@" <<'PY'
import json, subprocess, sys, time
base, opts, runs, *inputs = sys.argv[1:]
rows = []
for spec in inputs:
    path, _, pages = spec.partition(":")
    for run in range(int(runs)):
        form = ["-F", f"files=@{path}", "-F", "to_formats=json"]
        if pages:
            a, _, b = pages.partition("-")
            form += ["-F", f"page_range={a}", "-F", f"page_range={b or a}"]
        for kv in opts.split(","):
            if kv:
                form += ["-F", kv]
        t0 = time.monotonic()
        out = subprocess.run(["curl", "-fsS", *form, f"{base}/v1/convert/file"], capture_output=True, text=True)
        wall = time.monotonic() - t0
        if out.returncode != 0:
            print(f"{spec}: request failed: {out.stderr.strip()}", file=sys.stderr); continue
        r = json.loads(out.stdout)
        timings = r.get("timings") or {}
        # stages come as {name: {total_ms|ms|secs...}} or {name: number}; keep the number
        def ms(v):
            if isinstance(v, (int, float)): return float(v)
            if isinstance(v, dict):
                for k in ("total_ms", "ms", "millis"):
                    if k in v: return float(v[k])
                for k in ("total", "secs", "seconds"):
                    if k in v: return float(v[k]) * 1000.0
            return 0.0
        stages = sorted(((k, ms(v)) for k, v in timings.items()), key=lambda kv: -kv[1])
        doc = (r.get("document") or {}).get("json_content") or {}
        n_pages = len(doc.get("pages") or {}) if isinstance(doc, dict) else 0
        formulas = sum(1 for t in (doc.get("texts") or []) if isinstance(t, dict) and t.get("label") == "formula")
        rows.append((spec, run, wall, float(r.get("processing_time") or 0.0), n_pages, formulas, stages))
print(f"{'input':44s} {'run':>3s} {'wall s':>7s} {'server s':>8s} {'pages':>5s} {'formulas':>8s}  top stages (ms)")
for spec, run, wall, srv, n, f, stages in rows:
    top = ", ".join(f"{k} {v:.0f}" for k, v in stages[:4])
    print(f"{spec[-44:]:44s} {run:3d} {wall:7.1f} {srv:8.1f} {n:5d} {f:8d}  {top}")
PY
