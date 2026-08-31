# Phase 4 — performance

**Runs after the correctness phases.** Phase 0 (scanned books), Phase 1
(formulas) and Phase 3 (hybrid / digital pages) all land first. Nothing here
changes what the pipeline *produces* — it changes what it costs. Optimizing
before the output is settled means re-measuring every time the output moves,
and risks trading correctness for speed while correctness is still in flight.

**Deliverable is a decision and a repeatable benchmark, not a port.**

**Deliverable is a decision and a repeatable benchmark, not a port.**

## 4.1 Why this phase exists

A CPU profile of the 12-page scanned book (taken under load, so absolute numbers
are inflated — proportions only):

```
layout.predict        72.8%   (11 calls)
pdfium.render         13.8%   (12 calls)
tableformer            6.4% + 3.0%
assemble_page          0.0%
```

Layout dominates; everything Phase 0 touched is 0.0%. OCR does not appear at all
because that fixture has a text layer — a true scan adds a fourth consumer.
Separately, formula decode runs at **~45 s per formula** on this CPU, which
dwarfs all of the above when formulas are present.

None of this ranking is trustworthy for a GPU deployment, which is why the phase
starts with measurement rather than migration.

## 4.2 Setup

A CUDA box matching production (GPU class matters for small models). Build with
`docling-onnx`'s `cuda` / `tensorrt` features; select via `DOCLING_RS_EP`.

## 4.3 Corpus

| Fixture | Exercises |
| --- | --- |
| `2206.01062` (9pp) | digital text, tables, pictures |
| GFS paper (15pp) | two-column reading order at scale |
| McKusick 4.4BSD ch.9 (26pp) | **true scan, no text layer** — the only OCR exercise |
| `oldscannedmath` pp.37–48 | scan + bad OCR layer + 32 formulas |
| `redp5110_sampled` | table-heavy, TableFormer |

Three runs each, median reported.

## 4.4 Metrics

`DOCLING_RS_TIMING` already records per-stage totals *and* call counts, so
per-call means come free. Add:

- wall clock and pages/sec
- **GPU utilization and memory sampled per stage** — the key diagnostic
- CPU-vs-GPU split (pdfium render is CPU-bound and will not move)
- formula decode: per-call latency **and output token count** (the one thing not
  currently instrumented)

## 4.5 Matrix

`CPU baseline · CUDA fp32 · CUDA fp16 · TensorRT EP` × five documents. No code
changes in this pass.

## 4.6 Decision rules — fixed before the data is seen

- Layout falls below ~20% of wall clock → **stop optimizing layout**, whatever
  the CPU profile said.
- Formula decode dominates **and GPU utilization during it is low (<30%)** →
  launch-bound → *this* is what justifies vLLM or TensorRT-LLM.
- Formula decode dominates but GPU utilization is high → compute-bound → a
  smaller model (`pix2text-mfr`, 120 MB vs 761 MB) beats any runtime change.
- pdfium render rises to the top → CPU-bound → process parallelism, not a
  runtime question.

Low GPU utilization is the entire reason a serving runtime helps. If it is never
observed, the migration was never the answer.

## 4.7 Cheap wins measured in the same pass

All inside ORT, no migration:

- batch layout across pages (one 640×640 forward per page today)
- batch the formula vision tower; padded-batch the decoder so 32 formulas cost
  `max(tokens)` steps instead of `sum(tokens)`
- re-test int8 layout on GPU — it mislabels a table on one Intel CPU, and that
  verdict may not transfer
- cap decode tokens

These are expected to reorder the profile enough that 4.6's rules fire
differently.

## 4.8 On runtimes

For the record, so the question is not relitigated from scratch:

- **vLLM cannot run five of the six models.** It serves language models and
  VLMs; RT-DETR, DBNet, PP-OCR, EfficientNet and TableFormer have no place in
  it. Its scope here is the CodeFormula decoder alone, as a sidecar.
- **TensorRT can run everything**, and `DOCLING_RS_EP=tensorrt` already routes
  ORT through it with zero code. Standalone TRT adds engine management (per GPU
  arch, per version, per shape profile), dynamic-shape profiles for
  variable-width OCR crops, and thin Rust bindings.
- **The binding constraint is that docling.rs is not only a server**: wasm
  compiles without ONNX Runtime at all, CoreML/DirectML cover Mac and Windows,
  CPU-only is the default and what CI runs, and there are Python and Node
  bindings. An NVIDIA-native stack means maintaining two inference paths
  permanently — in a repo whose work goes upstream.

Order the levers so quantization is last: EP → batching → CUDA graphs /
continuous batching → fp16 → int8 per-model with an accuracy gate. Never
blanket int8; the disabled `layout_heron_int8.onnx` is the standing reminder.

## 4.9 Repeatability

`scripts/bench/` emitting the table, and `docs/PERFORMANCE.md` with committed
numbers — the discipline `docs/PDF_CONFORMANCE.md` already uses. Today's CPU
table is the first row, so a change that quietly doubles layout time is caught.
