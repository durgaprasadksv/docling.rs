//! The `reader` export: a document as the chunks a read-aloud client plays.
//!
//! A screen reader or a talking-book app does not want a document tree. It
//! wants, page by page and in reading order, the pieces of text it will
//! speak, each with the box to scroll to, the box of every word to highlight
//! as the voice reaches it, and the *role* of the piece — so that running
//! headers and footers can be skipped, headings announced, tables summarized
//! or read cell by cell, formulas spoken in the user's chosen style or passed
//! over. docling's own JSON has all of that, spread over a `body` tree of
//! `$ref`s, `prov` arrays, `words` arrays and a `speech` map; this export is
//! that information flattened into the shape such a client already stores:
//!
//! ```jsonc
//! { "schema": "docling-rs/reader/1",
//!   "source": "chemistry-2e.pdf",
//!   "firstPageNumber": 759,                 // 0-based index of pages[0]
//!   "pageSizes": [[612.0, 792.0]],          // points, one per entry of pages
//!   "pages": [[
//!     { "id": "#/texts/4", "kind": "heading", "level": 1,
//!       "t": "Common Ion Effect on Solubility", "p": 759,
//!       "s": 12, "e": 17,                    // word range on the page
//!       "l": 72.0, "tp": 701.6, "r": 300.2, "b": 689.9,
//!       "w": [[0, 6, 72.0, 701.6, 118.3, 689.9], …] },
//!     { "id": "#/texts/9", "kind": "formula", "t": "cap m g, …", "p": 759, …,
//!       "latex": "M g ( O H ) _ { 2 } …", "orig": "",
//!       "speech": { "clearspeak": "cap m g, …", "simplespeak": "…" } }
//!   ]] }
//! ```
//!
//! Conventions, chosen to match what a client keeps in its own cache rather
//! than what docling writes:
//!
//! - **Coordinates** are PDF points, bottom-left origin, `tp > b` — on every
//!   box, including word boxes and table cells (docling writes cells
//!   top-left; they are flipped here).
//! - **Word boxes** `w` are `[charStart, charEnd, l, t, r, b]` with the
//!   character range in **UTF-16 code units** of `t`, because the clients
//!   this is for (Dart, JavaScript) index strings that way; docling's own
//!   `words` use Unicode scalar offsets. Coverage can be partial: a chunk
//!   whose geometry came from a merged continuation, or from a scanned line
//!   the recognizer could not split, has fewer entries than words, and a
//!   consumer falls back to even spacing for the uncovered characters.
//! - **`s`/`e`** are the half-open range of whitespace-separated words on
//!   the page before and after the chunk, counted over `t` after cleaning,
//!   so a client can address a word by page-global index.
//! - **Dot leaders** (`Notices . . . . . . 1`, four or more periods) are
//!   dropped from `t` and `w` adjusted, as a reader would otherwise voice
//!   them for seconds; a genuine ellipsis survives.
//! - **Nothing is dropped.** Page furniture, picture text, table cells,
//!   formulas without a spoken form all appear with their `kind`; the client
//!   decides what to skip, and can change its mind without re-converting.
//!
//! `kind` and the extras each one carries:
//!
//! | `kind` | extras |
//! |---|---|
//! | `paragraph` | — |
//! | `title`, `heading` | `level` (1 = top) |
//! | `page_header`, `page_footer` | — |
//! | `list_item` | `marker`, `enumerated`, `list` (the group's ref), `depth` |
//! | `caption` | `of` (the picture's or table's ref) |
//! | `footnote` | — |
//! | `code` | `language` when docling detected one |
//! | `formula` | `latex`, `orig`, `speech` (style → text); `t` is the ClearSpeak form or `""` |
//! | `checkbox` | `checked` |
//! | `picture` | `t` is `""`; `captions` (refs) |
//! | `picture_text` | `of` (the picture): text docling found inside a figure |
//! | `table` | `t` is `""`; `rows`, `cols`, `captions` |
//! | `table_cell` | `table` (the ref), `row`, `col`, `header` |
//! | `toc_entry` | `page` when the entry ends in a page number; a `document_index` table, or a table that reads as one, row by row |
//! | `key_value`, `form` | docling's key-value and form items, text only |
//!
//! The export is built from the document's JSON (`DoclingDocument::export_to_json_value`
//! with word provenance and speech forced on), so a client holding docling
//! JSON can produce it too: [`from_docling_json`].

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

/// The `schema` value every export carries; bump the trailing number when a
/// field changes meaning.
pub const SCHEMA: &str = "docling-rs/reader/1";

/// Build the reader export from a docling-JSON document.
///
/// Items without `prov` (declarative formats have none) land on the page of
/// the previous chunk with a zero box, so no text is lost. Word boxes come
/// only from items that carry a `words` array (PDF conversions with word
/// provenance on), speech only from formula items with a `speech` map.
pub fn from_docling_json(doc: &Value) -> Value {
    let mut b = Builder::new(doc);
    b.walk_children(doc.get("body"), &Ctx::default());
    // docling's furniture tree is empty for PDFs (headers and footers sit in
    // the body with `content_layer: furniture`); other backends may fill it.
    b.walk_children(doc.get("furniture"), &Ctx::default());
    b.finish()
}

/// Where in the tree a chunk is produced.
#[derive(Clone, Default)]
struct Ctx {
    /// The innermost enclosing list group's ref and the nesting depth.
    list: Option<(String, usize)>,
    /// The enclosing picture's ref: its nested texts are `picture_text`.
    picture: Option<String>,
}

struct Chunk {
    page: usize,
    value: Map<String, Value>,
}

struct Builder<'a> {
    doc: &'a Value,
    /// `page_no` (1-based) → `(width, height)` in points.
    sizes: BTreeMap<usize, (f64, f64)>,
    chunks: Vec<Chunk>,
    /// Words emitted so far per page (0-based), for `s`/`e`.
    words_so_far: BTreeMap<usize, u64>,
    last_page: usize,
}

impl<'a> Builder<'a> {
    fn new(doc: &'a Value) -> Self {
        let mut sizes = BTreeMap::new();
        if let Some(pages) = doc.get("pages").and_then(Value::as_object) {
            for (k, v) in pages {
                let no = v
                    .get("page_no")
                    .and_then(Value::as_u64)
                    .map(|n| n as usize)
                    .or_else(|| k.parse().ok());
                let w = v.pointer("/size/width").and_then(Value::as_f64);
                let h = v.pointer("/size/height").and_then(Value::as_f64);
                if let (Some(no), Some(w), Some(h)) = (no, w, h) {
                    sizes.insert(no, (w, h));
                }
            }
        }
        let last_page = sizes.keys().next().copied().unwrap_or(1).saturating_sub(1);
        Self {
            doc,
            sizes,
            chunks: Vec::new(),
            words_so_far: BTreeMap::new(),
            last_page,
        }
    }

    fn resolve(&self, r: &Value) -> Option<(String, &'a Value)> {
        let path = r.get("$ref").and_then(Value::as_str)?;
        let v = self.doc.pointer(path.strip_prefix('#')?)?;
        Some((path.to_string(), v))
    }

    fn walk_children(&mut self, node: Option<&Value>, ctx: &Ctx) {
        let Some(children) = node
            .and_then(|n| n.get("children"))
            .and_then(Value::as_array)
        else {
            return;
        };
        for c in children {
            if let Some((path, item)) = self.resolve(c) {
                self.walk(&path, item, ctx);
            }
        }
    }

    fn walk(&mut self, path: &str, item: &Value, ctx: &Ctx) {
        let label = item.get("label").and_then(Value::as_str).unwrap_or("");
        if path.starts_with("#/groups/") {
            let mut ctx = ctx.clone();
            if label == "list" || label == "ordered_list" {
                let depth = ctx.list.as_ref().map_or(1, |(_, d)| d + 1);
                ctx.list = Some((path.to_string(), depth));
            }
            self.walk_children(Some(item), &ctx);
            return;
        }
        if path.starts_with("#/tables/") {
            self.table(path, item);
            return;
        }
        if path.starts_with("#/pictures/") {
            let mut m = self.base(item, "picture", "");
            m.insert("captions".into(), refs(item.get("captions")));
            self.push(path, m);
            let ctx = Ctx {
                picture: Some(path.to_string()),
                ..ctx.clone()
            };
            self.walk_children(Some(item), &ctx);
            return;
        }
        if path.starts_with("#/key_value_items/") || path.starts_with("#/form_items/") {
            let kind = if path.starts_with("#/form_items/") {
                "form"
            } else {
                "key_value"
            };
            let text = kv_text(item);
            let m = self.base(item, kind, &text);
            self.push(path, m);
            self.walk_children(Some(item), ctx);
            return;
        }
        // A text item.
        let text = item.get("text").and_then(Value::as_str).unwrap_or("");
        let parent = item.pointer("/parent/$ref").and_then(Value::as_str);
        let (kind, extra): (&str, Vec<(&str, Value)>) = match label {
            "section_header" => (
                "heading",
                vec![(
                    "level",
                    json!(item.get("level").and_then(Value::as_u64).unwrap_or(1)),
                )],
            ),
            "title" => ("title", vec![("level", json!(0))]),
            "page_header" => ("page_header", vec![]),
            "page_footer" => ("page_footer", vec![]),
            "list_item" => {
                let (list, depth) = ctx
                    .list
                    .as_ref()
                    .map(|(l, d)| (json!(l), json!(d)))
                    .unwrap_or((Value::Null, json!(0)));
                // docling gives an item it found no marker for the default
                // `-`; a reader voicing markers would say "dash" before
                // `(a) MgCl2`. Only a marker the text does not already carry
                // and that is not that default is one.
                let marker = item.get("marker").and_then(Value::as_str).filter(|m| {
                    !(*m == "-"
                        && !item
                            .get("enumerated")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                        && !text.trim_start().starts_with('-'))
                });
                vec![
                    ("marker", marker.map_or(Value::Null, |m| json!(m))),
                    (
                        "enumerated",
                        item.get("enumerated").cloned().unwrap_or(json!(false)),
                    ),
                    ("list", list),
                    ("depth", depth),
                ]
                .pipe(|e| ("list_item", e))
            }
            "caption" => (
                "caption",
                vec![(
                    "of",
                    parent
                        .filter(|p| !p.starts_with("#/body") && !p.starts_with("#/groups"))
                        .map_or(Value::Null, |p| json!(p)),
                )],
            ),
            "footnote" => ("footnote", vec![]),
            "code" => (
                "code",
                vec![(
                    "language",
                    item.get("code_language").cloned().unwrap_or(Value::Null),
                )],
            ),
            "formula" => {
                let speech = item.get("speech").cloned().unwrap_or(Value::Null);
                let spoken = speech
                    .get("clearspeak")
                    .and_then(Value::as_str)
                    .or_else(|| {
                        speech
                            .as_object()
                            .and_then(|m| m.values().next())
                            .and_then(Value::as_str)
                    })
                    .unwrap_or("")
                    .to_string();
                let mut m = self.base(item, "formula", &spoken);
                m.insert("latex".into(), json!(text));
                m.insert(
                    "orig".into(),
                    item.get("orig").cloned().unwrap_or(Value::Null),
                );
                m.insert("speech".into(), speech);
                // The spoken form is not the item's text, so docling's word
                // boxes (which index the LaTeX) do not apply; the whole box
                // stands for every word, as a highlight of the formula.
                m.remove("w");
                self.push(path, m);
                return;
            }
            "checkbox_selected" | "checkbox_unselected" => (
                "checkbox",
                vec![("checked", json!(label == "checkbox_selected"))],
            ),
            _ if ctx.picture.is_some() => (
                "picture_text",
                vec![("of", json!(ctx.picture.as_deref().unwrap()))],
            ),
            _ => ("paragraph", vec![]),
        };
        let mut m = self.base(item, kind, text);
        for (k, v) in extra {
            m.insert(k.into(), v);
        }
        self.push(path, m);
        // Nested items (a caption whose parent is a text item, a list item's
        // sub-list) follow their parent.
        self.walk_children(Some(item), ctx);
    }

    /// A table: one `table` chunk, then its cells — or, for a table of
    /// contents, one `toc_entry` per row.
    fn table(&mut self, path: &str, item: &Value) {
        let data = item.get("data");
        let cells: Vec<&Value> = data
            .and_then(|d| d.get("table_cells"))
            .and_then(Value::as_array)
            .map(|a| a.iter().collect())
            .unwrap_or_default();
        let rows = data
            .and_then(|d| d.get("num_rows"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let cols = data
            .and_then(|d| d.get("num_cols"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let label = item.get("label").and_then(Value::as_str).unwrap_or("table");
        let page_no = prov_page(item);
        let height = page_no.and_then(|p| self.sizes.get(&p)).map(|s| s.1);

        // Rows as the texts of their cells, left to right.
        let mut by_row: BTreeMap<u64, Vec<(u64, &Value)>> = BTreeMap::new();
        for c in &cells {
            let r = c
                .get("start_row_offset_idx")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let col = c
                .get("start_col_offset_idx")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            by_row.entry(r).or_default().push((col, c));
        }
        for v in by_row.values_mut() {
            v.sort_by_key(|(c, _)| *c);
        }

        if label == "document_index" || looks_like_toc(&by_row) {
            for (r, row) in &by_row {
                let text = row
                    .iter()
                    .map(|(_, c)| c.get("text").and_then(Value::as_str).unwrap_or("").trim())
                    .filter(|t| !t.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ");
                if text.trim().is_empty() {
                    continue;
                }
                let bbox = union_boxes(row.iter().map(|(_, c)| c.get("bbox")), height);
                let mut m = Map::new();
                m.insert("kind".into(), json!("toc_entry"));
                m.insert("table".into(), json!(path));
                m.insert("row".into(), json!(r));
                if let Some(page) = trailing_number(&text) {
                    m.insert("page".into(), json!(page));
                }
                let (page, bbox) = (page_no.unwrap_or(self.last_page + 1), bbox);
                self.push_text(format!("{path}/r{r}"), m, &text, page, bbox, None);
            }
            return;
        }

        let mut m = self.base(item, "table", "");
        m.insert("rows".into(), json!(rows));
        m.insert("cols".into(), json!(cols));
        m.insert("captions".into(), refs(item.get("captions")));
        self.push(path, m);
        for (r, row) in &by_row {
            for (col, c) in row {
                let text = c.get("text").and_then(Value::as_str).unwrap_or("");
                if text.trim().is_empty() {
                    continue;
                }
                let header = c
                    .get("column_header")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                    || c.get("row_header")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                let mut m = Map::new();
                m.insert("kind".into(), json!("table_cell"));
                m.insert("table".into(), json!(path));
                m.insert("row".into(), json!(r));
                m.insert("col".into(), json!(col));
                m.insert("header".into(), json!(header));
                let bbox = bbox_of(c.get("bbox"), height);
                let page = page_no.unwrap_or(self.last_page + 1);
                self.push_text(format!("{path}/r{r}c{col}"), m, text, page, bbox, None);
            }
        }
        // Captions and footnotes are children in docling's tree.
        self.walk_children(Some(item), &Ctx::default());
    }

    /// The common fields of an item's chunk: text, page, box, word boxes.
    fn base(&mut self, item: &Value, kind: &str, text: &str) -> Map<String, Value> {
        let page_no = prov_page(item).unwrap_or(self.last_page + 1);
        let height = self.sizes.get(&page_no).map(|s| s.1);
        let bbox = bbox_of(item.pointer("/prov/0/bbox"), height);
        let words = item.get("words").and_then(Value::as_array).map(|ws| {
            ws.iter()
                .filter_map(|w| {
                    let cs = w.pointer("/charspan/0").and_then(Value::as_u64)? as usize;
                    let ce = w.pointer("/charspan/1").and_then(Value::as_u64)? as usize;
                    let b = bbox_of(w.get("bbox"), height);
                    Some(WordBox { cs, ce, b })
                })
                .collect::<Vec<_>>()
        });
        let mut m = Map::new();
        m.insert("kind".into(), json!(kind));
        self.text_fields(&mut m, text, page_no, bbox, words);
        m
    }

    fn push_text(
        &mut self,
        id: String,
        mut m: Map<String, Value>,
        text: &str,
        page_no: usize,
        bbox: [f64; 4],
        words: Option<Vec<WordBox>>,
    ) {
        self.text_fields(&mut m, text, page_no, bbox, words);
        self.push(&id, m);
    }

    /// Fill `t p s e l tp r b w` — cleaning the text, converting word offsets
    /// to UTF-16, counting words into the page's running total.
    fn text_fields(
        &mut self,
        m: &mut Map<String, Value>,
        text: &str,
        page_no: usize,
        bbox: [f64; 4],
        words: Option<Vec<WordBox>>,
    ) {
        let (text, words) = drop_dot_leaders(text, words.unwrap_or_default());
        let p = page_no.saturating_sub(1);
        self.last_page = p;
        let n = count_words(&text);
        let start = *self.words_so_far.get(&p).unwrap_or(&0);
        self.words_so_far.insert(p, start + n);
        m.insert("t".into(), json!(text));
        m.insert("p".into(), json!(p));
        m.insert("s".into(), json!(start));
        m.insert("e".into(), json!(start + n));
        m.insert("l".into(), json!(round2(bbox[0])));
        m.insert("tp".into(), json!(round2(bbox[1])));
        m.insert("r".into(), json!(round2(bbox[2])));
        m.insert("b".into(), json!(round2(bbox[3])));
        if !words.is_empty() {
            let utf16 = utf16_offsets(&text);
            let w: Vec<Value> = words
                .iter()
                .filter_map(|w| {
                    let cs = *utf16.get(w.cs)?;
                    let ce = *utf16.get(w.ce)?;
                    Some(json!([
                        cs,
                        ce,
                        round2(w.b[0]),
                        round2(w.b[1]),
                        round2(w.b[2]),
                        round2(w.b[3])
                    ]))
                })
                .collect();
            m.insert("w".into(), Value::Array(w));
        }
    }

    fn push(&mut self, id: &str, mut m: Map<String, Value>) {
        m.insert("id".into(), json!(id));
        let page = m.get("p").and_then(Value::as_u64).unwrap_or(0) as usize;
        self.chunks.push(Chunk { page, value: m });
    }

    fn finish(self) -> Value {
        let first = self
            .chunks
            .iter()
            .map(|c| c.page)
            .chain(self.sizes.keys().map(|k| k.saturating_sub(1)))
            .min()
            .unwrap_or(0);
        let last = self
            .chunks
            .iter()
            .map(|c| c.page)
            .chain(self.sizes.keys().map(|k| k.saturating_sub(1)))
            .max()
            .unwrap_or(0);
        let mut pages: Vec<Vec<Value>> = vec![Vec::new(); last - first + 1];
        for c in self.chunks {
            pages[c.page - first].push(Value::Object(c.value));
        }
        let sizes: Vec<Value> = (first..=last)
            .map(|p| match self.sizes.get(&(p + 1)) {
                Some((w, h)) => json!([round2(*w), round2(*h)]),
                None => Value::Null,
            })
            .collect();
        let mut out = Map::new();
        out.insert("schema".into(), json!(SCHEMA));
        if let Some(name) = self.doc.get("name").and_then(Value::as_str) {
            out.insert("source".into(), json!(name));
        }
        out.insert("firstPageNumber".into(), json!(first));
        out.insert("pageSizes".into(), Value::Array(sizes));
        out.insert(
            "pages".into(),
            Value::Array(pages.into_iter().map(Value::Array).collect()),
        );
        Value::Object(out)
    }
}

/// A word's box with its character range (Unicode scalar offsets into the
/// chunk text, as docling's `words` give them).
#[derive(Clone, Debug, PartialEq)]
struct WordBox {
    cs: usize,
    ce: usize,
    b: [f64; 4],
}

trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
        f(self)
    }
}
impl<T> Pipe for T {}

fn refs(v: Option<&Value>) -> Value {
    Value::Array(
        v.and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|r| r.get("$ref").cloned()).collect())
            .unwrap_or_default(),
    )
}

fn prov_page(item: &Value) -> Option<usize> {
    item.pointer("/prov/0/page_no")
        .and_then(Value::as_u64)
        .map(|n| n as usize)
}

/// A docling `bbox` as `[l, top, r, bottom]` in bottom-left points: flipped
/// when it is TOPLEFT and the page height is known.
fn bbox_of(b: Option<&Value>, height: Option<f64>) -> [f64; 4] {
    let Some(b) = b else { return [0.0; 4] };
    let g = |k: &str| b.get(k).and_then(Value::as_f64).unwrap_or(0.0);
    let (l, t, r, bt) = (g("l"), g("t"), g("r"), g("b"));
    let top_left = b.get("coord_origin").and_then(Value::as_str) == Some("TOPLEFT");
    match (top_left, height) {
        (true, Some(h)) => [l, h - t, r, h - bt],
        // TOPLEFT without a page height: keep the numbers, put the larger
        // one on top so `tp > b` still holds.
        (true, None) => [l, t.max(bt), r, t.min(bt)],
        _ => [l, t.max(bt), r, t.min(bt)],
    }
}

fn union_boxes<'a>(
    boxes: impl Iterator<Item = Option<&'a Value>>,
    height: Option<f64>,
) -> [f64; 4] {
    let mut acc: Option<[f64; 4]> = None;
    for b in boxes.flatten() {
        let x = bbox_of(Some(b), height);
        acc = Some(match acc {
            None => x,
            Some(a) => [
                a[0].min(x[0]),
                a[1].max(x[1]),
                a[2].max(x[2]),
                a[3].min(x[3]),
            ],
        });
    }
    acc.unwrap_or([0.0; 4])
}

/// The text of a key-value or form item: whatever text its cells carry.
fn kv_text(item: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(cells) = item.pointer("/graph/cells").and_then(Value::as_array) {
        for c in cells {
            if let Some(t) = c.get("text").and_then(Value::as_str) {
                if !t.trim().is_empty() {
                    parts.push(t.trim().to_string());
                }
            }
        }
    }
    parts.join(" ")
}

/// A table that is really a table of contents: three or more rows, four in
/// five ending in a page number and one or two columns wide. The layout model
/// labels most of these `document_index`, but the label is not kept on the
/// table today; the shape is.
fn looks_like_toc(rows: &BTreeMap<u64, Vec<(u64, &Value)>>) -> bool {
    if rows.len() < 3 || rows.values().any(|r| r.len() > 2) {
        return false;
    }
    let numbered = rows
        .values()
        .filter(|row| {
            let text = row
                .iter()
                .map(|(_, c)| c.get("text").and_then(Value::as_str).unwrap_or(""))
                .collect::<Vec<_>>()
                .join(" ");
            trailing_number(&text).is_some()
        })
        .count();
    numbered * 5 >= rows.len() * 4
}

/// The page number a contents entry ends in (`Notices . . . . 1` → 1).
fn trailing_number(text: &str) -> Option<u64> {
    let t = text.trim_end();
    let digits = t.chars().rev().take_while(char::is_ascii_digit).count();
    if digits == 0 || digits > 5 {
        return None;
    }
    let before = t[..t.len() - digits].trim_end();
    // A bare number row (`12`) is not an entry; the number must follow text.
    if before.is_empty() {
        return None;
    }
    t[t.len() - digits..].parse().ok()
}

/// The client's word count: runs of non-whitespace, where whitespace is
/// space, tab, LF and CR.
fn count_words(s: &str) -> u64 {
    let mut n = 0;
    let mut in_word = false;
    for c in s.chars() {
        let ws = is_ws(c);
        if !ws && !in_word {
            n += 1;
        }
        in_word = !ws;
    }
    n
}

fn is_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

/// Drop dot leaders — four or more periods, optionally space-separated as a
/// PDF often places each glyph, with the whitespace around the run — for a
/// single space, keeping the word boxes aligned: boxes inside a run go,
/// boxes after it move back, a box straddling the edge is clipped.
fn drop_dot_leaders(text: &str, words: Vec<WordBox>) -> (String, Vec<WordBox>) {
    let chars: Vec<char> = text.chars().collect();
    // Find runs as char ranges [start, end).
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '.' {
            i += 1;
            continue;
        }
        // Count periods allowing whitespace between them.
        let mut j = i;
        let mut dots = 0;
        let mut last_dot = i;
        while j < chars.len() && (chars[j] == '.' || is_ws(chars[j])) {
            if chars[j] == '.' {
                dots += 1;
                last_dot = j;
            }
            j += 1;
        }
        if dots >= 4 {
            // Extend over the whitespace before and after.
            let mut s = i;
            while s > 0 && is_ws(chars[s - 1]) {
                s -= 1;
            }
            let mut e = last_dot + 1;
            while e < chars.len() && is_ws(chars[e]) {
                e += 1;
            }
            runs.push((s, e));
            i = e;
        } else {
            i = last_dot + 1;
        }
    }
    if runs.is_empty() {
        return (text.to_string(), words);
    }
    let mut out = String::with_capacity(text.len());
    let mut kept: Vec<WordBox> = Vec::new();
    // Map an old char index to its new one; `None` inside a dropped run.
    let mut new_index: Vec<Option<usize>> = Vec::with_capacity(chars.len() + 1);
    let mut r = 0;
    let mut n = 0usize;
    for (idx, &c) in chars.iter().enumerate() {
        if r < runs.len() && idx >= runs[r].0 && idx < runs[r].1 {
            if idx == runs[r].0 {
                // The run becomes one space, unless it starts or ends the text.
                if !out.is_empty() && runs[r].1 < chars.len() {
                    out.push(' ');
                    n += 1;
                }
            }
            new_index.push(None);
            if idx + 1 == runs[r].1 {
                r += 1;
            }
            continue;
        }
        new_index.push(Some(n));
        out.push(c);
        n += 1;
    }
    new_index.push(Some(n));
    for w in words {
        // Clip to the kept characters.
        let mut cs = w.cs;
        let mut ce = w.ce;
        while cs < ce && new_index.get(cs).copied().flatten().is_none() {
            cs += 1;
        }
        while ce > cs
            && new_index
                .get(ce.saturating_sub(1))
                .copied()
                .flatten()
                .is_none()
        {
            ce -= 1;
        }
        if cs >= ce {
            continue;
        }
        let (Some(Some(ncs)), Some(Some(nce))) = (
            new_index.get(cs).copied(),
            new_index.get(ce - 1).copied().map(|x| x.map(|x| x + 1)),
        ) else {
            continue;
        };
        kept.push(WordBox {
            cs: ncs,
            ce: nce,
            b: w.b,
        });
    }
    // A trailing space left by a run at the very end is trimmed by the
    // run rule above; collapse a double space a run may have produced.
    (out.trim_end().to_string(), kept)
}

/// For each Unicode scalar offset `i` (0..=chars) the UTF-16 offset.
fn utf16_offsets(s: &str) -> Vec<u32> {
    let mut out = Vec::with_capacity(s.chars().count() + 1);
    let mut n = 0u32;
    out.push(0);
    for c in s.chars() {
        n += c.len_utf16() as u32;
        out.push(n);
    }
    out
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc() -> Value {
        json!({
            "schema_name": "DoclingDocument",
            "name": "sample",
            "body": {"self_ref": "#/body", "children": [
                {"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}, {"$ref": "#/groups/0"},
                {"$ref": "#/texts/4"}, {"$ref": "#/tables/0"}, {"$ref": "#/pictures/0"},
                {"$ref": "#/texts/7"}
            ]},
            "furniture": {"self_ref": "#/furniture", "children": []},
            "groups": [
                {"self_ref": "#/groups/0", "parent": {"$ref": "#/body"}, "label": "list", "name": "list",
                 "children": [{"$ref": "#/texts/2"}, {"$ref": "#/texts/3"}]}
            ],
            "texts": [
                {"self_ref": "#/texts/0", "parent": {"$ref": "#/body"}, "label": "page_header",
                 "text": "746", "prov": [{"page_no": 2, "bbox": {"l": 28.0, "t": 767.0, "r": 42.0, "b": 758.0, "coord_origin": "BOTTOMLEFT"}}]},
                {"self_ref": "#/texts/1", "parent": {"$ref": "#/body"}, "label": "section_header", "level": 1,
                 "text": "Common Ion Effect", "prov": [{"page_no": 2, "bbox": {"l": 72.0, "t": 701.6, "r": 300.0, "b": 690.0, "coord_origin": "BOTTOMLEFT"}}],
                 "words": [
                   {"charspan": [0, 6], "text": "Common", "bbox": {"l": 72.0, "t": 701.6, "r": 118.0, "b": 690.0, "coord_origin": "BOTTOMLEFT"}},
                   {"charspan": [7, 10], "text": "Ion", "bbox": {"l": 122.0, "t": 701.6, "r": 140.0, "b": 690.0, "coord_origin": "BOTTOMLEFT"}},
                   {"charspan": [11, 17], "text": "Effect", "bbox": {"l": 144.0, "t": 701.6, "r": 190.0, "b": 690.0, "coord_origin": "BOTTOMLEFT"}}
                 ]},
                {"self_ref": "#/texts/2", "parent": {"$ref": "#/groups/0"}, "label": "list_item", "marker": "(a)", "enumerated": true,
                 "text": "MgCl2", "prov": [{"page_no": 2, "bbox": {"l": 90.0, "t": 654.0, "r": 140.0, "b": 644.0, "coord_origin": "BOTTOMLEFT"}}]},
                {"self_ref": "#/texts/3", "parent": {"$ref": "#/groups/0"}, "label": "list_item", "marker": "(b)", "enumerated": true,
                 "text": "KOH", "prov": [{"page_no": 2, "bbox": {"l": 90.0, "t": 634.0, "r": 140.0, "b": 624.0, "coord_origin": "BOTTOMLEFT"}}]},
                {"self_ref": "#/texts/4", "parent": {"$ref": "#/body"}, "label": "formula",
                 "text": "M g ^ { 2 + }", "orig": "", "speech": {"clearspeak": "cap m g, super 2 plus", "simplespeak": "cap m g super 2 plus"},
                 "prov": [{"page_no": 2, "bbox": {"l": 100.0, "t": 560.0, "r": 400.0, "b": 540.0, "coord_origin": "BOTTOMLEFT"}}]},
                {"self_ref": "#/texts/5", "parent": {"$ref": "#/pictures/0"}, "label": "caption",
                 "text": "FIGURE 1 A plot.", "prov": [{"page_no": 2, "bbox": {"l": 72.0, "t": 300.0, "r": 300.0, "b": 290.0, "coord_origin": "BOTTOMLEFT"}}]},
                {"self_ref": "#/texts/6", "parent": {"$ref": "#/pictures/0"}, "label": "text",
                 "text": "axis label", "prov": [{"page_no": 2, "bbox": {"l": 80.0, "t": 400.0, "r": 120.0, "b": 392.0, "coord_origin": "BOTTOMLEFT"}}]},
                {"self_ref": "#/texts/7", "parent": {"$ref": "#/body"}, "label": "text",
                 "text": "Notices . . . . . . . 1 and café ☃ done", "prov": [{"page_no": 3, "bbox": {"l": 72.0, "t": 700.0, "r": 500.0, "b": 690.0, "coord_origin": "BOTTOMLEFT"}}],
                 "words": [
                   {"charspan": [0, 7], "text": "Notices", "bbox": {"l": 72.0, "t": 700.0, "r": 110.0, "b": 690.0, "coord_origin": "BOTTOMLEFT"}},
                   {"charspan": [8, 9], "text": ".", "bbox": {"l": 112.0, "t": 700.0, "r": 114.0, "b": 690.0, "coord_origin": "BOTTOMLEFT"}},
                   {"charspan": [22, 23], "text": "1", "bbox": {"l": 200.0, "t": 700.0, "r": 206.0, "b": 690.0, "coord_origin": "BOTTOMLEFT"}},
                   {"charspan": [33, 34], "text": "☃", "bbox": {"l": 300.0, "t": 700.0, "r": 310.0, "b": 690.0, "coord_origin": "BOTTOMLEFT"}},
                   {"charspan": [35, 39], "text": "done", "bbox": {"l": 320.0, "t": 700.0, "r": 350.0, "b": 690.0, "coord_origin": "BOTTOMLEFT"}}
                 ]}
            ],
            "pictures": [
                {"self_ref": "#/pictures/0", "parent": {"$ref": "#/body"}, "label": "picture", "captions": [{"$ref": "#/texts/5"}],
                 "children": [{"$ref": "#/texts/5"}, {"$ref": "#/texts/6"}],
                 "prov": [{"page_no": 2, "bbox": {"l": 72.0, "t": 500.0, "r": 300.0, "b": 310.0, "coord_origin": "BOTTOMLEFT"}}]}
            ],
            "tables": [
                {"self_ref": "#/tables/0", "parent": {"$ref": "#/body"}, "label": "table", "captions": [], "children": [],
                 "prov": [{"page_no": 2, "bbox": {"l": 72.0, "t": 520.0, "r": 500.0, "b": 480.0, "coord_origin": "BOTTOMLEFT"}}],
                 "data": {"num_rows": 2, "num_cols": 2, "table_cells": [
                   {"start_row_offset_idx": 0, "start_col_offset_idx": 0, "text": "Ion", "column_header": true, "row_header": false,
                    "bbox": {"l": 72.0, "t": 272.0, "r": 200.0, "b": 290.0, "coord_origin": "TOPLEFT"}},
                   {"start_row_offset_idx": 0, "start_col_offset_idx": 1, "text": "Ksp", "column_header": true, "row_header": false,
                    "bbox": {"l": 200.0, "t": 272.0, "r": 500.0, "b": 290.0, "coord_origin": "TOPLEFT"}},
                   {"start_row_offset_idx": 1, "start_col_offset_idx": 0, "text": "Mg(OH)2", "column_header": false, "row_header": false,
                    "bbox": {"l": 72.0, "t": 292.0, "r": 200.0, "b": 312.0, "coord_origin": "TOPLEFT"}},
                   {"start_row_offset_idx": 1, "start_col_offset_idx": 1, "text": "", "column_header": false, "row_header": false,
                    "bbox": {"l": 200.0, "t": 292.0, "r": 500.0, "b": 312.0, "coord_origin": "TOPLEFT"}}
                 ]}}
            ],
            "pages": {"2": {"size": {"width": 612.0, "height": 792.0}, "page_no": 2},
                      "3": {"size": {"width": 612.0, "height": 792.0}, "page_no": 3}}
        })
    }

    fn kinds(page: &Value) -> Vec<String> {
        page.as_array()
            .unwrap()
            .iter()
            .map(|c| c["kind"].as_str().unwrap().to_string())
            .collect()
    }

    /// The whole shape: pages anchored at the first converted page, chunks in
    /// tree order with their roles, nothing dropped.
    #[test]
    fn pages_chunks_and_kinds() {
        let out = from_docling_json(&doc());
        assert_eq!(out["schema"], SCHEMA);
        assert_eq!(out["source"], "sample");
        assert_eq!(out["firstPageNumber"], 1); // docling page 2 → 0-based 1
        assert_eq!(out["pageSizes"], json!([[612.0, 792.0], [612.0, 792.0]]));
        let pages = out["pages"].as_array().unwrap();
        assert_eq!(pages.len(), 2);
        assert_eq!(
            kinds(&pages[0]),
            [
                "page_header",
                "heading",
                "list_item",
                "list_item",
                "formula",
                "table",
                "table_cell",
                "table_cell",
                "table_cell",
                "picture",
                "caption",
                "picture_text",
            ]
        );
        assert_eq!(kinds(&pages[1]), ["paragraph"]);
    }

    /// Word boxes: UTF-16 offsets into `t`, bottom-left boxes, and the page
    /// word counter running across chunks.
    #[test]
    fn word_boxes_and_counts() {
        let out = from_docling_json(&doc());
        let p0 = &out["pages"][0];
        let heading = &p0[1];
        assert_eq!(heading["t"], "Common Ion Effect");
        assert_eq!(heading["s"], 1); // "746" was one word
        assert_eq!(heading["e"], 4);
        assert_eq!(heading["level"], 1);
        assert_eq!(
            heading["w"],
            json!([
                [0, 6, 72.0, 701.6, 118.0, 690.0],
                [7, 10, 122.0, 701.6, 140.0, 690.0],
                [11, 17, 144.0, 701.6, 190.0, 690.0]
            ])
        );
        let (l, tp, r, b) = (
            heading["l"].as_f64().unwrap(),
            heading["tp"].as_f64().unwrap(),
            heading["r"].as_f64().unwrap(),
            heading["b"].as_f64().unwrap(),
        );
        assert!(tp > b && r > l, "bottom-left box: {heading}");
    }

    /// A list item knows its list and marker; a caption knows what it captions;
    /// text inside a figure is marked as such.
    #[test]
    fn roles_carry_their_context() {
        let out = from_docling_json(&doc());
        let p0 = &out["pages"][0];
        let item = &p0[2];
        assert_eq!(item["marker"], "(a)");
        assert_eq!(item["enumerated"], true);
        assert_eq!(item["list"], "#/groups/0");
        assert_eq!(item["depth"], 1);
        let caption = &p0[10];
        assert_eq!(caption["of"], "#/pictures/0");
        // docling's default `-` marker on an item whose text does not start
        // with a dash is not a marker.
        let mut d = doc();
        d["texts"][2]["marker"] = json!("-");
        d["texts"][2]["enumerated"] = json!(false);
        let out = from_docling_json(&d);
        assert_eq!(out["pages"][0][2]["marker"], Value::Null);
        d["texts"][2]["text"] = json!("- MgCl2");
        let out = from_docling_json(&d);
        assert_eq!(out["pages"][0][2]["marker"], "-");
        let inside = &p0[11];
        assert_eq!(inside["kind"], "picture_text");
        assert_eq!(inside["of"], "#/pictures/0");
    }

    /// A formula speaks its ClearSpeak form as `t`, keeps the LaTeX and the
    /// whole speech map, and has no word boxes (they would index the LaTeX).
    #[test]
    fn formula_speaks() {
        let out = from_docling_json(&doc());
        let f = &out["pages"][0][4];
        assert_eq!(f["t"], "cap m g, super 2 plus");
        assert_eq!(f["latex"], "M g ^ { 2 + }");
        assert_eq!(f["speech"]["simplespeak"], "cap m g super 2 plus");
        assert!(f.get("w").is_none());
    }

    /// Table cells are flipped to bottom-left and know their place; the empty
    /// cell is not a chunk.
    #[test]
    fn table_cells() {
        let out = from_docling_json(&doc());
        let p0 = &out["pages"][0];
        assert_eq!(p0[5]["rows"], 2);
        let cell = &p0[6];
        assert_eq!(cell["t"], "Ion");
        assert_eq!(cell["header"], true);
        assert_eq!(cell["row"], 0);
        assert_eq!(cell["col"], 0);
        assert_eq!(cell["tp"], 792.0 - 272.0);
        assert_eq!(cell["b"], 792.0 - 290.0);
    }

    /// Dot leaders go, boxes follow; a non-BMP character counts twice in the
    /// UTF-16 offsets the client indexes by.
    #[test]
    fn leaders_dropped_and_utf16_offsets() {
        let out = from_docling_json(&doc());
        let para = &out["pages"][1][0];
        assert_eq!(para["t"], "Notices 1 and café ☃ done");
        let w = para["w"].as_array().unwrap();
        // "Notices", "1", "☃", "done" — the lone "." inside the run is gone.
        assert_eq!(w.len(), 4);
        assert_eq!(w[0][0], 0);
        assert_eq!(w[0][1], 7);
        assert_eq!(w[1][0], 8); // "1" right after the single space
        assert_eq!(w[1][1], 9);
        // "Notices 1 and café " is 19 chars; "☃" is one char, one UTF-16 unit.
        assert_eq!(w[2][0], 19);
        assert_eq!(w[2][1], 20);
        assert_eq!(w[3][0], 21);
        assert_eq!(w[3][1], 25);
        assert_eq!(para["s"], 0);
        assert_eq!(para["e"], 6);
    }

    #[test]
    fn utf16_counts_surrogate_pairs() {
        // U+1D400 (mathematical bold A) is two UTF-16 units.
        let (t, w) = drop_dot_leaders(
            "𝐀 x",
            vec![WordBox {
                cs: 2,
                ce: 3,
                b: [0.0; 4],
            }],
        );
        assert_eq!(t, "𝐀 x");
        let off = utf16_offsets(&t);
        assert_eq!(off[w[0].cs], 3);
        assert_eq!(off[w[0].ce], 4);
    }

    /// Three periods are an ellipsis and stay; four are a leader.
    #[test]
    fn ellipsis_survives() {
        let (t, _) = drop_dot_leaders("wait... what", vec![]);
        assert_eq!(t, "wait... what");
        let (t, _) = drop_dot_leaders("Chapter 1 . . . . 12", vec![]);
        assert_eq!(t, "Chapter 1 12");
        let (t, _) = drop_dot_leaders("Chapter 1 ......", vec![]);
        assert_eq!(t, "Chapter 1");
    }

    /// A one- or two-column table whose rows end in page numbers is a table
    /// of contents and comes out as entries, each with its page.
    #[test]
    fn contents_table_becomes_entries() {
        let mut d = doc();
        let cells = json!([
            {"start_row_offset_idx": 0, "start_col_offset_idx": 0, "text": "Notices . . . . . . 1",
             "bbox": {"l": 72.0, "t": 100.0, "r": 500.0, "b": 110.0, "coord_origin": "TOPLEFT"}},
            {"start_row_offset_idx": 1, "start_col_offset_idx": 0, "text": "Preface . . . . . . 3",
             "bbox": {"l": 72.0, "t": 112.0, "r": 500.0, "b": 122.0, "coord_origin": "TOPLEFT"}},
            {"start_row_offset_idx": 2, "start_col_offset_idx": 0, "text": "Chapter 1. Overview . . . . 5",
             "bbox": {"l": 72.0, "t": 124.0, "r": 500.0, "b": 134.0, "coord_origin": "TOPLEFT"}}
        ]);
        d["tables"][0]["data"]["table_cells"] = cells;
        d["tables"][0]["data"]["num_rows"] = json!(3);
        d["tables"][0]["data"]["num_cols"] = json!(1);
        let out = from_docling_json(&d);
        let p0 = out["pages"][0].as_array().unwrap();
        let entries: Vec<&Value> = p0.iter().filter(|c| c["kind"] == "toc_entry").collect();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0]["t"], "Notices 1");
        assert_eq!(entries[0]["page"], 1);
        assert_eq!(entries[2]["t"], "Chapter 1. Overview 5");
        assert_eq!(entries[2]["page"], 5);
        assert!(p0.iter().all(|c| c["kind"] != "table"));
    }

    /// An item without provenance stays in the output, on the page of the
    /// chunk before it, with a zero box.
    #[test]
    fn nothing_is_dropped() {
        let mut d = doc();
        d["texts"][7]["prov"] = json!([]);
        let out = from_docling_json(&d);
        let p0 = out["pages"][0].as_array().unwrap();
        let last = p0.last().unwrap();
        assert_eq!(last["t"], "Notices 1 and café ☃ done");
        assert_eq!(last["l"], 0.0);
    }

    #[test]
    fn word_count_matches_the_client() {
        assert_eq!(count_words("a  b\tc\nd"), 4);
        assert_eq!(count_words("  "), 0);
        assert_eq!(count_words("x\u{a0}y"), 1); // NBSP is not a separator there
    }
}
