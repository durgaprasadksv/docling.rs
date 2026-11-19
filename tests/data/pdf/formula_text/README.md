# Formula regions rebuilt from the text layer

`cases.json`: every display formula of
[pdf-parse-bench](https://huggingface.co/datasets/piushorn/pdf-parse-bench)
(see `../pdf_parse_bench/README.md`) whose region the layout model found —
the region's glyphs (character, box, baseline, size, and for a code without
ToUnicode its font-program glyph name) and thin rules, the ground-truth LaTeX
(`gt`, from Wikipedia, CC BY-SA 4.0), the output `formula_text` recorded for
it (`latex`, `null` when it left the region to the model) and that output's
status against the ground truth after canonicalisation (`exact`, `near`,
`wrong`, `refused`).

`formula_text::tests::benchmark_formulas_rebuild_as_recorded` replays every
case with no PDF and no model and fails on any change of output. After an
intended change, regenerate with `scripts/conformance/pdf_parse_bench.sh
--update` (or, from saved dumps, `pdf_parse_bench.py <data> <run> --update
… --fixtures tests/data/pdf/formula_text`) and review what moved.
