# Phase 0 — scanned books read correctly, end to end

**Status:** proposed, 2026-08-30
**Scope:** `docling-pdf`, `docling-core`, `docling-serve`; consumer changes in `~/TalkingText`
**Out of scope:** formulas, verbalization, the hybrid trigger — those are Phase 1
and Phase 3.

Phase 0 contains **no formula handling**. That keeps the first end-to-end pass
free of ML-formula complexity and proves out the whole chain. Phase 1 then adds
formulas on top of a chain already known to work, and Phase 3 (hybrid, digital
pages) reuses all of it behind a different trigger.

---

## Objectives

Three sequential outcomes, each independently demonstrable.

| # | Outcome | Where | Demo |
|---|---|---|---|
| **1** | Word-level bounding boxes in the output | server | CLI on a scanned PDF → boxes drawn on the raster land on words |
| **2** | Reading order and document metadata preserved | server + one client fix | two-column scan reads down column 1 then column 2; headings are headings |
| **3** | Merged into the on-device structure and played | client + endpoint | open a scanned book, press play, highlight follows the voice |

**Objectives 1 and 2 need no server and no app** — the `docling-rs` CLI is the
whole test harness. Only objective 3 introduces transport.

---

## Objective 1 — word-level bounding boxes

**Done when:** `docling-rs` converts a scanned PDF and the JSON carries a
per-word box keyed to character offsets in each item's text, and a debug overlay
shows those boxes sitting on the words.

Digital pages come along for free — their `word_cells` are already populated;
only the serialization is missing.

### 1.1 Where the geometry is lost today

Line and word geometry is **computed and discarded**. None of it needs inventing.

| Path | Word cells | Status |
|---|---|---|
| Digital page | `line_and_word_cells` (`dp_lines.rs:428`) → `PdfPage.word_cells` | computed, dropped at assembly |
| Scanned, table regions | `prep_table_words` (`ocr_prep.rs:312`) → `ocr_table_words` | computed, dropped at assembly |
| Scanned, body text | — | never computed |

**Loss at assembly, affecting every path:**

```rust
region_texts_exclusive(&regions, &cells) -> Vec<String>   // assemble.rs:1114
```

Works out which cells belong to which region, joins their text, **throws the
cells away**. After this a node knows its own outline and nothing else.

On a scanned page `page.cells` after OCR is **one `TextCell` per line** with a
real page-point box — structurally identical to a digital page's line cells. OCR
is just a different way of producing them, so this single fix serves both.

**Loss at CTC decode, scanned body text only:** `ocr_page` (`ocr.rs:266`)
recognizes one **line** at a time via `prep_region_lines`, so body text never
gets word cells. The positional information does exist — see 1.2.

**Not in scope: bbox quantization.** `take_prov` (`json.rs:182`) reconstructs
`prov.bbox` from the 0–511 DocLang grid (`x0 * cur_w / 512.0`) — ~1.20 pt/step
horizontally on a 612 pt page, ~1.55 pt/step vertically on 792 pt. Fine for a
block outline, too coarse for anything tighter. Fixing it means carrying the
float box alongside the grid: `Node::Located` gains `bbox_pt: Option<[f32;4]>`,
`take_prov` prefers it, DocLang/DocTags keep reading `location` so their output
stays byte-identical. **Tracked separately** — the boxes in this objective come
from cells, in points, and never touch the 512 grid.

### 1.2 How OCR produces text, and why positions are recoverable

PP-OCR's recognition model does **no word segmentation**. A space is a character
it predicts, like `a` or `7`.

1. Line crop → resized to height `REC_HEIGHT` (32), width preserved by aspect
2. CNN → `t_len` feature columns (horizontal downsample ~/4–/8)
3. Each column: a distribution over `nc` characters, including a literal space
   and a CTC **blank** at index 0
4. Greedy CTC decode collapses that to a string

`decode_row_scored` (`ocr_prep.rs:73`) is the collapse:

```rust
for row in probs.chunks_exact(nc) {   // one row per timestep
    let best = argmax(row);
    if best != prev && best != 0 {    // merge repeats; drop blanks
        out.push_str(chars[best]);    // <- char only; the index is discarded
    }
    prev = best;
}
```

`best != prev` merges repeats, `best != 0` drops blanks. Blank exists so double
letters survive the merge (`hello` needs one between the two `l`s).

**The loop index is the timestep, and timestep maps linearly to x** because the
CNN downsamples horizontally by a fixed factor and never mixes columns
arbitrarily:

```
x_in_crop ~= (t / t_len) * crop_width
```

So the position of every emitted character — and therefore every space — is
available for free. The function simply does not record it.

**Why not `segment_words`.** It (`ocr_prep.rs:263`) ink-projects a line and splits
on gaps; that is how `prep_table_words` gets table word boxes. Applied to body
text it gives N boxes and M recognized words with no way to pair them when
N != M — punctuation, tight kerning, italics, hyphens. `prep_table_words`
sidesteps that by recognizing each word crop *separately*, which costs ~5–10x the
inference and is *less* accurate, since PP-OCR rec is trained on line crops. CTC
alignment costs nothing and is aligned by construction. Keep `segment_words` only
as a fallback for lines whose decode contains no space at all.

### 1.3 Work

**A. Keep the cells at assembly** (`docling-pdf`)

1. `assemble.rs:1114` — add a variant of `region_texts_exclusive` returning
   `Vec<Vec<&TextCell>>` instead of `Vec<String>`. Identical assignment logic
   (intersection-over-self > 0.2, exclusive best region); just do not collapse.
2. While `cells_text()` joins, **record each cell's char offset into the joined
   string**. This must happen *during* the join: `dp_lines` inserts spaces per
   its merge rules, so recovering offsets afterwards by string-matching breaks on
   ligatures and hyphenation.

Result, per node: `Vec<(char_start, char_end, l, t, r, b)>`.

**B. Recover word positions from CTC** (`docling-pdf`)

1. `ocr_prep.rs:73` — add a sibling, leaving `decode_row_scored` untouched so
   `score_lines` and `ocr_table_words` are unaffected:
   ```rust
   pub fn decode_row_positions(chars: &[String], probs: &[f32], nc: usize)
       -> (String, f32, Vec<usize>)
   ```
   Same loop; also push the chunk index whenever a character is emitted.
2. `ocr.rs recognize_batch` — thread the positions out. Both widths are already
   in scope: `w` (padded, from `width_batches`) and `lines[i].w` (the crop's true
   width after the aspect-preserving resize, stored on `PrepLine`).
3. `ocr.rs ocr_page` — per line, walk the decoded string, split on spaces, and map
   each word's first/last timestep to page points:
   ```
   x_page = l + (t * w_padded) / (t_len * line.w) * (r - l)     clamp to [l, r]
   ```
   `w_padded` because the batch pads to a shared width; dividing by `w` instead of
   `line.w` drifts every box right. Emit word `TextCell`s reusing the line's
   `t`/`b`. Push to `page.word_cells`.
4. Fallback — a line whose decode contains no space emits no word cells;
   consumers fall back to the line box.

**C. Serialize** (`docling-core`)

New **opt-in** option, default off:

```jsonc
{ "text": "...",
  "prov":  [ { "page_no": 1, "bbox": {...}, "charspan": [0, 42] } ],
  "words": [ { "charspan": [0, 3],
               "bbox": { "l":.., "t":.., "r":.., "b":..,
                         "coord_origin": "BOTTOMLEFT" } } ] }
```

Additive, not a schema change. `prov[0]` stays the block bbox, so existing
consumers are untouched and conformance snapshots do not move with the flag off.

> Do **not** overload `prov` with one entry per word. It is legal — `prov` is a
> list and `charspan` already exists — but consumers read `prov[0]` as *the* bbox
> and would silently receive the first word's box instead.

Plumb the option through every surface in one PR, as the repo's working notes
require: `DocumentConverter` builder → CLI flag → serve (multipart field + JSON
body + query param) → Python kwarg → Node option struct. Grep `page_range` for
the pattern.

### 1.4 Verification

- **Visual, mandatory.** A debug renderer behind `DOCLING_RS_DEBUG` that draws the
  boxes onto the page raster. The only reliable way to catch the padding/`line.w`
  mistake, which produces boxes that look plausible but drift right.
- **Snapshot.** Word count per line, monotonic non-overlapping x within a line.
- **Peakiness.** CTC emits a character in one sharp spike, usually near the *end*
  of its visual evidence rather than the centre, so boundaries may sit slightly
  right of the visual gap. Measure it; subtract half a character width if the
  offset is systematic.

### 1.5 Increments

- **A + C alone** ships line-level boxes on scans and **word-level on digital
  pages** (their `word_cells` are already populated).
- **+ B** upgrades scans to word level, with no further format change.

---

## Objective 2 — reading order and document metadata

**Done when:** a two-column scanned page converts to text that reads down column
one then column two; headings are labelled headings; running heads and page
numbers are not read aloud as body text.

### 2.1 Reading order

The server already gets this right. `order_page` runs, `assemble_page` emits in
that order, and the JSON's node order **is** the reading order.

The loss is entirely client-side: the transformer discards it and re-sorts
geometrically by top-Y then left-X, verbatim from upstream's
`DoclingSchemaTransformer`. On a two-column scan that **interleaves the columns**
— for TTS the worst possible failure, because the narration becomes incoherent
rather than merely wrong.

Fix: trust the server's order, drop the geometric re-sort. ~30 lines, and the
largest single correctness win in Phase 0. Not blocked by objective 1.

### 2.2 Metadata that has to survive

"Preserved" needs a concrete list, or there is no definition of done. Each of
these changes how the text is spoken:

| Metadata | Why it matters for playback |
|---|---|
| Node order | narration coherence — the item above |
| Block label (text / section header / list item / caption / footnote / table) | heading pause and pitch; list item prefixes; captions and footnotes may be skipped or deferred |
| Heading level | prosody, and the chapter navigation tree |
| Page number | the resume cursor, and page-anchored highlight |
| Furniture vs body (running heads, folios) | must **not** be read aloud; docling classifies these, the classification has to reach the client |
| Table structure | a table read cell-by-cell in the wrong order is noise |
| Caption ↔ figure association | decides whether a caption is read in place or skipped |

Work: audit the JSON on a scanned fixture against this list, and confirm each
survives the client transformer. Anything that does not is either a serialization
gap on the server or a drop in the transformer — fix wherever it falls.

### 2.3 Verification

A two-column scanned fixture asserted end-to-end, plus a fixture with running
heads. Both must be in the test suite, not checked by eye once.

---

## Objective 3 — merge into the on-device structure and play

**Done when:** a scanned book opens in the app, plays in the right order, and the
highlight tracks the spoken word.

### 3.1 Conversion endpoint (`docling-serve`)

Accept a **sliced PDF** — the pages needing conversion, extracted client-side via
`FPDF_ImportPagesByIndex` — plus a page map, and return docling JSON.

```
POST /v1/pages
  files:   <sliced PDF, N pages>
  meta:    { "page_map": [42, 51], "options": { "ocr": "auto", ... } }
```

- The slice **is** the selection — no page-range or page-list parameter needed.
- `page_map` is required: the server sees pages 0..N, the client needs 42 and 51
  back.
- Content-hash the slice for a server-side cache. Scanned books are where users
  overlap (same textbook, same public-domain scan), so this turns a per-user cost
  into a per-title one.
- Stream the response, one page per line, so page 3 is usable while page 20 is
  still converting.

### 3.2 Client

- Batch pages ahead of the playhead, not by parse-window boundary. Font resources
  are copied once per slice, so a 10-page slice costs barely more than a 2-page
  one — bigger batches are cheaper per page.
- `docling_schema_transformer.dart:22` currently synthesises `symbolEnds` by even
  spacing. Read the `words` array and build real ones.
- Where a line has no word boxes, emit N identical rects — one per word, all
  carrying the line's box. The highlight then advances line by line. **This is the
  same mechanism formulas will use in Phase 1**, so building it here costs
  nothing later.
- Feed the merged page to `PageWordCounter`, then `SentenceChunkSplitter`.

### 3.3 Verification

Open a scanned book on device. Press play. The words are right, the order is
right, the highlight follows. That is the Phase 0 exit criterion.

---

## Risks

| Risk | Mitigation |
|---|---|
| Two-column scans narrate incoherently | Objective 2.1, with a two-column fixture in the suite |
| Padding/`line.w` confusion silently shifts every box right | The visual check is mandatory, not optional |
| CTC peakiness biases boundaries right | Measure; constant offset if systematic |
| Unspaced scripts emit no space class | Fallback to the line box |
| Conformance drift | Option defaults off; DocLang/DocTags never read the new field |
| Char offsets wrong after the join | Track during the join, never reconstruct after |
| Furniture read aloud as body text | Objective 2.2 audit, with a running-heads fixture |

---

## Effort

| Item | Lines |
|---|---|
| Obj 1 — keep cells at assembly | ~100 |
| Obj 1 — CTC positions | ~150 |
| Obj 1 — serialize + option plumbing | ~150 + plumbing |
| Obj 1 — verification | ~150 |
| Obj 2 — reading order (client) | ~30 |
| Obj 2 — metadata audit + fixtures | ~150 |
| Obj 3 — endpoint | ~200 |
| Obj 3 — client consumption | ~100 |

Branch `claude/word-provenance` off fresh `origin/master`, signed off. Not a
`fix:` — it cuts a 0.x minor.

---

## Explicitly deferred

- **Formulas** — CodeFormulaV2 on the server, and whether it produces usable
  LaTeX on this corpus. Phase 1.
- **Verbalization** — LaTeX → spoken text. Nothing currently does this anywhere.
  Server stage inside docling.rs, or client-side Rust with MathCAT. **Undecided,
  and it gates Phase 1**; if it lands client-side it needs an FFI library with
  real setup cost, so decide before Phase 0 ships.
- **Hybrid trigger and digital pages** — Phase 3. The trigger is already built and
  corpus-measured.
- **Bbox quantization** — see objective 1.1.

---

# Phase 1 — formulas

Phase 0 objectives 1 and 2 are done (word boxes; reading order and metadata,
both measured against the Python groundtruth). Objective 3 is deferred until
after Phase 3 by decision — the client merge is not on the critical path for
getting formulas right.

## What was already built

`crates/docling-pdf/src/enrich.rs` carries a **complete CodeFormulaV2 port** —
Idefics3 preprocessing (longest-edge 2048 → multiple-of-512 → 512×512 tiling
plus a squashed global tile), the tiled `<image>` prompt, and greedy decode over
three ONNX graphs, with the KV-cached decoder step verified argmax-identical to
`transformers.generate`. It is wired to `--enrich-formula` and gated on
`.models/code_formula/` (761 MB: vision 374, embed 231, int8 decoder 165,
tokenizer 7).

So Phase 1 is not "build formula recognition". It is **turn on what exists,
measure it, and close the gaps around it.**

## 1.1 What the measurements showed

Detection is not the problem. On a genuine 1900s scanned arithmetic textbook,
the layout model found **32 formula regions across 12 pages**, correctly placed
on the worked equations. `formula` is class 2 of heron's 17, so detection costs
nothing extra — it is the same pass that finds text and tables. There is no
separate MFD stage as in pix2text.

The gap is entirely in decoding, and it differs by document kind:

| | `text` (LaTeX) | `orig` (fallback) |
| --- | --- | --- |
| digital PDF | empty without the flag | `'a 2 +8 = 12'` — a linearized glyph dump; the superscript is gone |
| scanned, no text layer | empty | **empty** |
| scanned, bad OCR layer | empty | `'Jl X ^ =: -·''^" * » 5X0-'` — worse than nothing |

Three distinct failure modes, and the third is the dangerous one: a reader that
speaks `orig` would narrate noise. **Digital degrades to wrong; scanned degrades
to silent; a bad OCR layer degrades to gibberish.**

Note also that a page with *any* text layer skips OCR entirely, so a bad
embedded layer wins over our own recognizer. `--force-full-page-ocr` overrides
it, but our OCR is a text recognizer and will not read math either. Only an MFR
model closes this.

Consequence for Phase 3: **a digital page containing a formula still needs
server processing.** The text layer cannot be trusted for math, so the hybrid
trigger cannot route such a page client-side merely because it has text.

## 1.2 Decoding

With `--enrich-formula`, the clean fixture decodes to
`a ^ { 2 } + 8 = 1 2` — byte-identical to docling's enriched groundtruth, so the
port is confirmed against upstream rather than merely plausible.

Work:

- Fallback when decode is unavailable or fails: **emit nothing rather than
  `orig`**. The item keeps its `formula` label with empty text, which is enough
  for a reader to say "formula" and move on. Never narrate the OCR layer.
- Cap decode tokens — a runaway decode on a degraded crop is the worst case, and
  degraded crops are common on real scans.
- Formula word boxes: N identical rects, one per spoken word, all carrying the
  region box, so the highlight rests on the formula while it is read. The
  mechanism already exists from objective 1.

## 1.3 Verbalization — LaTeX to natural text

Decided: **server-side, in Rust.** The chain is validated end to end.

```
CodeFormula LaTeX → normalize → MathML → speech
```

- **`latex2mathml`** for MathML, *not* `pulldown-latex`: the latter mis-parses
  integrals, emitting `<msub><mo>∫</mo>…</msub><msup><mrow></mrow>…` — the limits
  split across separate scripts with an empty base — where `latex2mathml`
  correctly produces `<msubsup>`. It also wraps every script in a redundant
  `<mrow>`.
- **A token-rejoining normalizer is mandatory.** CodeFormula emits
  space-separated tokens, so `= 1 2` parses as two `<mn>` nodes and a speech
  engine says "one two" instead of "twelve"; `1 0 0 . 5` shatters into five.
  The normalizer strips those spaces while keeping the one that terminates a
  control word (`\cdot E` keeps its space, `\rightarrow2` does not need one).
- **MathCAT** (`mathcat`, MIT, DAISY — the engine behind NVDA and JAWS) for
  MathML → speech. It is written in Rust, so this needs no FFI, no per-platform
  binaries, and stays wasm-compatible. It ships its own `Rules` directory;
  English-only is a fraction of the 6.5 MB full set and fits the `.models/`
  asset pattern (missing → warn and degrade).

Measured output of the full chain:

```
a ^ { 2 } + 8 = 1 2   → "eigh squared plus 8, is equal to 12"
E = m c ^ { 2 }       → "cap e is equal to m c squared"
\int_0^\infty …       → "the integral from 0 to infinity of; e raised to the
                         negative x power, d x; is equal to 1"
\nabla \cdot E = …    → "divergence of cap e, is equal to; the fraction with
                         numerator rho; and denominator epsilon sub 0"
2H₂ + O₂ → 2H₂O       → "2 cap h sub 2, plus cap o sub 2; REACTS TO FORM, …"
9½ - 6¼ = 3¼          → "9 and 1 half, minus 6 and 1 fourth; is equal to, …"
```

`∇·` becomes "divergence of", and MathCAT's chemistry heuristics fire on their
own — the arrow reads as "reacts to form", not "right arrow". **Chemistry needs
no separate work**, which collapses it into the same path as math and physics.

Surface: an additive **`spoken` field** on the formula item, JSON-only, off by
default behind a flag — the pattern the `words` array established, so
conformance output is untouched and no groundtruth is regenerated.

## 1.4 Science scope

| Tier | New model | Status |
| --- | --- | --- |
| display equations — math, physics | none | the chain above |
| display chemical equations | none | same chain; MathCAT detects chemistry |
| structure diagrams (benzene rings) | — | **out of scope**; label only |
| inline math/chemistry in prose | none, but new detection | later |

Structural formulas are pictures, not notation: reading them needs OCSR
(image → SMILES → IUPAC name), which is a project of its own and out of scope.
The honest fallback is labelling — docling's picture classifier already has a
`chemistry_structure` class among its 26 and is already implemented, so a reader
can say "chemical structure diagram" instead of skipping silently.

Inline math never gets a box: heron has only a *display* formula class, so
`H₂O` or `E=mc²` inside a sentence arrives as ordinary text. That needs
detection in the text stream rather than in a region, touches ordinary prose,
and carries regression risk. Later phase.

---

## Phases after this one

- **Phase 3 — hybrid / digital pages.** The trigger is already built and
  corpus-measured. Note the Phase 1 finding that constrains it: a digital page
  containing a formula still needs server processing, because the text layer
  cannot be trusted for math.
- **Phase 4 — performance.** See `performance-plan.md`. Deliberately last:
  correctness first, and optimizing before the output is settled means
  re-measuring every time the output moves.
