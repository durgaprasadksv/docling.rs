# The `reader` output format

`--to reader` (CLI), `to=reader` (`/v1/convert`), `to_formats=reader`
(`/v1/convert/file`, answered in `reader_content`), library
`export_to_reader_json()` / `export_to_reader_value()`.

A document as the chunks a read-aloud client plays: page by page, in reading
order, each piece of text with its **role**, the **box** to scroll to, the box
of **every word** to highlight as the voice reaches it, and for a formula the
**spoken forms**. docling's own JSON carries all of this across a `body` tree
of `$ref`s, `prov` arrays, `words` arrays and a `speech` map; this export is
that information flattened into one list per page, in the shape a reader keeps
in its cache, so it is decoded, not transformed.

```jsonc
{
  "schema": "docling-rs/reader/1",
  "source": "chemistry-2e.pdf",
  "firstPageNumber": 759,            // 0-based index of pages[0]
  "pageSizes": [[612.0, 792.0]],     // points; one entry per entry of pages
  "pages": [
    [                                // the chunks of page 759, in reading order
      { "id": "#/texts/1", "kind": "page_header",
        "t": "15 · Equilibria of Other Reaction Classes", "p": 759,
        "s": 1, "e": 7,
        "l": 290.1, "tp": 767.3, "r": 540.0, "b": 758.0,
        "w": [[0, 2, 290.1, 767.3, 303.9, 758.0], [3, 4, 307.1, 767.3, 311.2, 758.0], …] },
      { "id": "#/texts/4", "kind": "heading", "level": 1,
        "t": "Common Ion Effect on Solubility", "p": 759, "s": 8, "e": 13, … },
      { "id": "#/texts/9", "kind": "formula",
        "t": "cap m g, open paren cap o cap h close paren sub 2, solid, is in equilibrium with, …",
        "latex": "M g ( O H ) _ { 2 } ( s ) \\rightleftharpoons M g ^ { 2 + } ( a q ) + 2 O H ^ { - } ( a q )",
        "orig": "",
        "speech": { "clearspeak": "cap m g, …", "simplespeak": "cap m g, …" },
        "p": 759, "s": 60, "e": 84, "l": 100.5, "tp": 560.2, "r": 420.0, "b": 540.8 }
    ]
  ]
}
```

## Every chunk

| Field | Meaning |
|---|---|
| `id` | docling's item ref (`#/texts/12`, `#/tables/0`, `#/tables/0/r3c1` for a cell, `#/tables/0/r3` for a contents entry) — stable within one conversion, for de-duplication and splicing |
| `kind` | the role; the table below |
| `t` | the text to speak. For a formula, its ClearSpeak form, or `""` when none could be produced |
| `p` | 0-based page index (docling's `page_no − 1`) |
| `s`, `e` | half-open range of whitespace-separated words on the page before / after the chunk, counted over `t` (whitespace: space, tab, LF, CR) — a page-global word index |
| `l`, `tp`, `r`, `b` | the chunk's box: PDF points, bottom-left origin, `tp > b` |
| `w` | word boxes `[charStart, charEnd, l, t, r, b]`; absent when the chunk has none |

### Coordinates

Every box — chunk, word, table cell — is PDF user space in points,
**bottom-left origin**, top above bottom. docling writes table cells top-left;
they are flipped here against the page height.

### Word boxes

`charStart`/`charEnd` index `t` in **UTF-16 code units**, the way Dart and
JavaScript index strings (docling's own `words` use Unicode scalar offsets;
the two differ only past U+FFFF). Coverage can be partial: a paragraph stitched
from a continuation on another page, or a scanned line the recognizer could not
split into words, has fewer entries than words. A consumer highlights the
covered words from their boxes and falls back to spacing the rest evenly over
the chunk box. Formula chunks carry no `w`: the spoken form is not the item's
text.

### Dot leaders

`Notices . . . . . . . . 1` — four or more periods, optionally
space-separated as PDFs place them — are dropped from `t` with the whitespace
around them, for a single space, and `w` is adjusted (boxes inside the run go,
boxes after it move back). A reader would otherwise voice them for seconds and
a sentence splitter would cut at every period. A genuine ellipsis (`...`)
survives.

### Nothing on the page is dropped

Running headers and footers, text inside figures, table cells, undecoded
formulas, checkboxes: every item is a chunk with its `kind`. The client decides
what to skip and how to voice each role, and can change its mind without
converting again. An item with no `prov` at all is on no page and is left out.

### The mapping is OpenDataLoader's

The rules follow OpenDataLoader-PDF's `DoclingSchemaTransformer`, the class a
client of that family applies to docling JSON when it does not receive this
export, so the two agree: an item without `prov` is skipped; a heading's
`level` is docling's integer, 1 when absent or unusable, clamped to 1–6; an
empty `text` falls back to `orig`; a picture's `description` is
`meta.description.text`, else its `description` annotation, and is the
picture's `t`; a table without `data` is skipped, its cells carry `rowspan` /
`colspan` when above 1, a cell without a box takes its share of the table's,
and when no cell carries a header flag the first row is the header. Where that
class flattens every other label to a paragraph, this export keeps the role.
When the JSON has no `body` tree, items are read from the `texts`, `tables`
and `pictures` arrays and ordered per page top to bottom, left to right, as
that class orders them; with a `body`, docling's reading order is kept.

## Roles

| `kind` | Extra fields | Notes |
|---|---|---|
| `paragraph` | — | body text, including footnote-less notes |
| `title` | `level: 0` | the document title (docling's `title` label) |
| `heading` | `level` (1 = top) | docling's `section_header`; on PDFs the level is the layout model's, not a document-global hierarchy |
| `page_header`, `page_footer` | — | running furniture; skip, or read once |
| `list_item` | `marker`, `enumerated`, `list` (the group's ref), `depth` (1 = top) | `t` is the item's text **without** its marker; `marker` is `"-"`, `"1."`, `"(a)"`, … or `null` |
| `caption` | `of` (ref of the picture or table) | `null` when docling attached it to no figure |
| `footnote` | — | |
| `code` | `language` or `null` | docling's `code` label; `t` keeps line breaks |
| `formula` | `latex`, `orig`, `speech` | `speech` maps style → text (`clearspeak`, `simplespeak`); `t` is the ClearSpeak form; `orig` the raw glyph text, routinely unreadable on scans; an undecoded formula has `latex: ""`, `speech: null`, `t: ""` and still its box |
| `checkbox` | `checked` | `t` is the option's label |
| `picture` | `captions` (refs), `description` | `t` is the description when docling produced one, else `""`; the box is the figure's |
| `picture_text` | `of` (the picture's ref) | text docling found inside a figure: axis labels, legends, scanned text in an image |
| `table` | `rows`, `cols`, `captions` (refs) | `t` is `""`; one chunk per table, before its cells |
| `table_cell` | `table` (ref), `row`, `col`, `header`, `rowspan` / `colspan` when above 1 | row-major, empty cells omitted; `header` for a column or row header, or the first row when the data flags none |
| `toc_entry` | `table` (ref), `row`, `page` | a contents table, row by row: the layout model's `document_index`, or a table of one or two columns whose rows end in page numbers; `page` when the row does |
| `key_value`, `form` | — | docling's key-value and form items, their cells' text joined |

## Producing it elsewhere

The export is a pure function of docling JSON:
`docling_core::reader::from_docling_json(&value)`. A consumer holding docling
JSON with `words` (`DOCLING_RS_WORD_PROV=1`) and `speech`
(`DOCLING_RS_SPEECH=1`) can build the same chunks; the `reader` format itself
switches both on for the conversion whatever the flags say.

## Versioning

`schema` names the shape. A field changing meaning bumps the trailing number;
adding a field or a `kind` does not — consumers should ignore kinds they do not
know rather than fail.
