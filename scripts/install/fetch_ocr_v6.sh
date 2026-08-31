#!/usr/bin/env bash
# Fetch the PP-OCRv6 small recognition model and the PP-OCRv3 DBNet detector.
#
# Both are Apache-2.0 and ONNX. They are kept out of the main dependency
# download because each changes pipeline behaviour and is opt-in per deployment:
#   * the detector only runs on crops the layout model failed on;
#   * the v6 recognizer supersedes the en_ pair when present (see
#     `docling-pdf::ocr::resolve_rec_pair` for the measured tradeoffs).
#
# The v6 model ships no dictionary file — its ~18.7k characters are embedded in
# `inference.yml` as a YAML list. They must be unquoted per YAML rules before
# use: a literal apostrophe is written `''''` there, and stripping only the outer
# quotes yields a two-character entry, which silently doubles every apostrophe
# in the output. Every entry must end up exactly one character; the check below
# is what catches that.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p .models

DET_URL="https://huggingface.co/breezedeus/cnstd-ppocr-en_PP-OCRv3_det/resolve/main/en_PP-OCRv3_det_infer.onnx"
REC_URL="https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/main/inference.onnx"
YML_URL="https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/main/inference.yml"

echo "fetching text detector -> .models/ocr_det_en.onnx"
curl -fsSL -o .models/ocr_det_en.onnx "$DET_URL"

echo "fetching PP-OCRv6 small recognition -> .models/ocr_rec_v6.onnx"
curl -fsSL -o .models/ocr_rec_v6.onnx "$REC_URL"
curl -fsSL -o .models/.v6_inference.yml "$YML_URL"

echo "extracting the character dictionary -> .models/v6_dict.txt"
python3 - <<'PY'
import ast
lines = open('.models/.v6_inference.yml', encoding='utf-8').read().split('\n')
i = next(n for n, l in enumerate(lines) if l.strip() == 'character_dict:')
raw = []
for l in lines[i + 1:]:
    if not l.startswith('  - '):
        break
    raw.append(l[4:])

def unyaml(c):
    if len(c) >= 2 and c[0] == c[-1] == "'":
        return c[1:-1].replace("''", "'")      # YAML single-quote escaping
    if len(c) >= 2 and c[0] == c[-1] == '"':
        try:
            return ast.literal_eval(c)
        except Exception:
            return c[1:-1]
    return c

out = [unyaml(c) for c in raw]
bad = [(n, c) for n, c in enumerate(out) if len(c) != 1]
if bad:
    raise SystemExit(f"dictionary mis-parse: {len(bad)} entries are not one character, e.g. {bad[:3]}")
open('.models/v6_dict.txt', 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print(f"  {len(out)} characters, all single-character")
PY
rm -f .models/.v6_inference.yml
echo "done"
