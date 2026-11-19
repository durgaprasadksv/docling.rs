# pdf-parse-bench: expected statuses

`expected.json` holds, for every PDF of
[pdf-parse-bench](https://huggingface.co/datasets/piushorn/pdf-parse-bench)
(the 2026-q1 formulas and tables configs, plus the repository's 2025-q4 set,
[github.com/phorn1/pdf-parse-bench](https://github.com/phorn1/pdf-parse-bench)
at commit `de72a161`, MIT, Pius Horn), the status docling.rs reached on each
ground-truth element, in reading order: text passages, inline formulas,
display formulas and tables. The PDFs are born-digital, compiled from LaTeX
with randomised fonts, classes and layouts; their formulas come from
[`piushorn/wikipedia-latex-formulas-319k`](https://huggingface.co/datasets/piushorn/wikipedia-latex-formulas-319k)
(Wikipedia, CC BY-SA 4.0).

`scripts/conformance/pdf_parse_bench.sh` fetches the PDFs (not committed),
converts each one (layout model on, formula VLM off: the text layer is what
is measured), scores every element (`scripts/conformance/pdf_parse_bench.py`
describes the statuses) and fails when any element got worse than recorded
here; `--update` accepts a new result.
