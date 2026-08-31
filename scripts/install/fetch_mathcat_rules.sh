#!/usr/bin/env bash
# Install the MathCAT rules the `speech` feature needs at runtime.
#
#   scripts/install/fetch_mathcat_rules.sh          # English only (~760 KB)
#   MATHCAT_LANGS="en de es" scripts/install/...    # more languages
#
# Unlike the ONNX models there is nothing to download: the rules ship inside
# the `mathcat` crate, so this copies them out of the cargo registry into
# .models/mathcat/ where the CWD-relative asset lookup finds them. Run it after
# a build that enabled the feature (that is what puts the crate on disk):
#
#   cargo build -p docling-cli --features speech
#
# MathCAT insists on a Braille directory even for speech-only use — it fails
# with "Wasn't able to find/read directory for language UEB" without one — so
# UEB comes along regardless of the language selection.
set -euo pipefail
cd "$(dirname "$0")/../.."

LANGS="${MATHCAT_LANGS:-en}"
DEST=.models/mathcat

SRC="$(find "${CARGO_HOME:-$HOME/.cargo}/registry/src" -maxdepth 2 -type d -name 'mathcat-*' 2>/dev/null | sort -V | tail -1)"
if [ -z "${SRC:-}" ] || [ ! -d "$SRC/Rules" ]; then
  echo "mathcat crate not found under the cargo registry." >&2
  echo "Build once with the feature first:  cargo build -p docling-cli --features speech" >&2
  exit 1
fi
echo "mathcat rules source: $SRC/Rules"

mkdir -p "$DEST/Languages" "$DEST/Braille"
cp "$SRC/Rules"/*.yaml "$DEST/"
cp -R "$SRC/Rules/Intent" "$DEST/"
# `zz` is MathCAT's fallback language definition; without it a missing rule in
# the selected language has nowhere to resolve.
for l in $LANGS zz; do
  [ -d "$SRC/Rules/Languages/$l" ] || { echo "  no such language: $l" >&2; continue; }
  cp -R "$SRC/Rules/Languages/$l" "$DEST/Languages/"
done
cp -R "$SRC/Rules/Braille/UEB" "$DEST/Braille/"

echo "installed $(du -sh "$DEST" | cut -f1) into $DEST (languages: $LANGS)"
echo "verify:  DOCLING_RS_SPEECH=1 cargo run -p docling-cli --features speech -- --to json --enrich-formula <pdf>"
