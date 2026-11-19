//! OpenDataLoader's synthetic page fixtures (`tests/data/pdf/opendataloader/`,
//! see its README): every upstream issue-test assertion that has a counterpart
//! on our output, replayed against docling.rs, plus a per-fixture word recall
//! against OpenDataLoader's own Markdown.
//!
//! `cases.json` records what held when it was written: a case is `pass`,
//! `xfail` (a measured gap, kept visible) or `unchecked` (the upstream
//! assertion is on OpenDataLoader's JSON shape). This test fails on a `pass`
//! that no longer passes, on an `xfail` that now passes — update the status,
//! the gap is closed — and on a fixture whose recall drops more than five
//! points. `scripts/conformance/opendataloader_cases.py` regenerates the file.
//!
//! Needs the layout model and converts ~170 pages (minutes in a debug build),
//! so it runs only with `DOCLING_RS_ODL_CORPUS=1`
//! (`scripts/conformance/opendataloader.sh`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use docling::{DocumentConverter, SourceDocument};
use serde_json::Value;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn ready() -> bool {
    if std::env::var_os("DOCLING_RS_ODL_CORPUS").is_none() {
        eprintln!("skipping: set DOCLING_RS_ODL_CORPUS=1 to run the OpenDataLoader corpus");
        return false;
    }
    let root = repo_root();
    if !root.join(".models/layout_heron.onnx").exists() {
        eprintln!("skipping: layout model not found");
        return false;
    }
    std::env::set_current_dir(&root).is_ok()
}

fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn words(s: &str) -> std::collections::HashSet<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| w.len() > 1)
        .map(str::to_string)
        .collect()
}

struct Output {
    md: String,
    chunks: Vec<Value>,
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// The failures of one case against our output — the same checks, with the
/// same meaning, as the Python evaluator that wrote `cases.json`.
fn check(case: &Value, out: &Output) -> Vec<String> {
    let ch = &case["checks"];
    let mut fails = Vec::new();
    let n = norm(&out.md);
    let lines: Vec<&str> = out.md.lines().collect();
    for seq in ch["order"].as_array().into_iter().flatten() {
        let mut pos: Option<usize> = None;
        for s in strs(seq) {
            let from = pos.map_or(0, |p| p + 1);
            match n.get(from..).and_then(|rest| rest.find(&norm(&s))) {
                Some(i) => pos = Some(from + i),
                None => {
                    fails.push(format!("order: {s:?} missing after position {pos:?}"));
                    break;
                }
            }
        }
    }
    for s in strs(&ch["contains"]) {
        if !n.contains(&norm(&s)) {
            fails.push(format!("contains: {s:?} missing"));
        }
    }
    for s in strs(&ch["not_contains"]) {
        if n.contains(&norm(&s)) {
            fails.push(format!("not_contains: {s:?} present"));
        }
    }
    for pair in ch["line_starts"].as_array().into_iter().flatten() {
        let p = strs(pair);
        let (Some(needle), Some(prefix)) = (p.first(), p.get(1)) else {
            continue;
        };
        match lines.iter().find(|l| l.contains(needle.as_str())) {
            None => fails.push(format!("line_starts: no line with {needle:?}")),
            Some(l) if !l.starts_with(prefix.as_str()) => fails.push(format!(
                "line_starts: line {:?} does not start with {prefix:?}",
                &l[..l.len().min(60)]
            )),
            _ => {}
        }
    }
    for group in ch["same_line"].as_array().into_iter().flatten() {
        let g = strs(group);
        // On one line, in this order.
        let in_order = |l: &str| {
            let nl = norm(l);
            let mut pos: Option<usize> = None;
            for s in &g {
                let from = pos.map_or(0, |p| p + 1);
                match nl.get(from..).and_then(|rest| rest.find(&norm(s))) {
                    Some(i) => pos = Some(from + i),
                    None => return false,
                }
            }
            true
        };
        if !lines.iter().any(|l| in_order(l)) {
            fails.push(format!("same_line: {g:?} not on one line in order"));
        }
    }
    let items: Vec<&Value> = out
        .chunks
        .iter()
        .filter(|c| c["kind"] == "list_item")
        .collect();
    // An upstream check names an item with its marker (`1. Find the length`);
    // the reader chunk keeps the marker apart from `t`.
    let item_text = |c: &Value| {
        norm(&format!(
            "{} {}",
            c["marker"].as_str().unwrap_or(""),
            c["t"].as_str().unwrap_or("")
        ))
    };
    for group in ch["one_list"].as_array().into_iter().flatten() {
        let mut refs = std::collections::HashSet::new();
        for s in strs(group) {
            match items.iter().find(|c| item_text(c).contains(&norm(&s))) {
                None => fails.push(format!("one_list: {s:?} is not in a list item")),
                Some(c) => {
                    refs.insert(c["list"].as_str().unwrap_or("").to_string());
                }
            }
        }
        if refs.len() > 1 {
            fails.push(format!("one_list: items spread over {} lists", refs.len()));
        }
    }
    for pair in ch["separate_lists"].as_array().into_iter().flatten() {
        let p = strs(pair);
        let refs: Vec<Option<&str>> = p
            .iter()
            .map(|s| {
                items
                    .iter()
                    .find(|c| item_text(c).contains(&norm(s)))
                    .map(|c| c["list"].as_str().unwrap_or(""))
            })
            .collect();
        if refs.iter().any(Option::is_none) {
            fails.push(format!(
                "separate_lists: an item of {p:?} is not a list item"
            ));
        } else if refs.len() == 2 && refs[0] == refs[1] {
            fails.push(format!("separate_lists: {p:?} are in one list"));
        }
    }
    for s in strs(&ch["in_list"]) {
        if !items.iter().any(|c| item_text(c).contains(&norm(&s))) {
            fails.push(format!("in_list: {s:?} is not a list item"));
        }
    }
    for s in strs(&ch["not_in_list"]) {
        if items.iter().any(|c| item_text(c).contains(&norm(&s))) {
            fails.push(format!("not_in_list: {s:?} is a list item"));
        }
    }
    if let Some(n) = case["list_items"].as_u64() {
        if items.len() as u64 != n {
            fails.push(format!("list_items: {} != {n}", items.len()));
        }
    }
    fails
}

// ---- Structural parity with OpenDataLoader -----------------------------------
//
// The property a client needs when a page may be parsed on the device (the
// Dart port of OpenDataLoader) or here: the same sequence of spoken units with
// the same roles. Both outputs flatten to (role, normalized text, box, depth),
// body content only, and are compared on four measures — the same ones, with
// the same meaning, as the Python evaluator that recorded them in cases.json.

#[derive(Clone, Debug)]
struct Elem {
    role: &'static str,
    text: String,
    bbox: [f64; 4],
}

/// Lower-cased, whitespace-collapsed, leading list marker dropped.
fn norm_text(s: &str) -> String {
    let t = norm(s).to_lowercase();
    let t = t.trim();
    // A bullet glyph, `-`/`*`/`+`/`o`, or `(a)`/`1.`/`b)`-style enumerator, then whitespace.
    let bullets = [
        '\u{2022}', '\u{2023}', '\u{25e6}', '\u{2043}', '\u{204c}', '\u{204d}', '\u{2219}',
        '\u{25aa}', '\u{25ab}', '\u{25cf}', '\u{25cb}', '-', '*', '+', '\u{00b7}', 'o',
    ];
    let mut chars = t.chars();
    if let Some(c) = chars.next() {
        let rest = chars.as_str();
        if bullets.contains(&c) && rest.starts_with(' ') {
            return rest.trim_start().to_string();
        }
    }
    // `(a)`, `a)`, `1.`, `12)`: up to three alphanumerics, `.` or `)`, space.
    let bytes = t.as_bytes();
    let mut i = 0;
    if bytes.first() == Some(&b'(') {
        i = 1;
    }
    let start = i;
    while i < bytes.len() && i - start < 3 && bytes[i].is_ascii_alphanumeric() {
        i += 1;
    }
    if i > start
        && i < bytes.len()
        && (bytes[i] == b'.' || bytes[i] == b')')
        && bytes.get(i + 1) == Some(&b' ')
    {
        return t[i + 1..].trim_start().to_string();
    }
    t.to_string()
}

fn odl_box(v: Option<&Value>) -> [f64; 4] {
    let a = v.and_then(Value::as_array);
    let g = |i: usize| {
        a.and_then(|a| a.get(i))
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
    };
    [g(0), g(1), g(2), g(3)]
}

fn odl_elements(doc: &Value) -> Vec<Elem> {
    fn walk(node: &Value, in_cell: bool, out: &mut Vec<Elem>) {
        if let Some(arr) = node.as_array() {
            for n in arr {
                walk(n, in_cell, out);
            }
            return;
        }
        let Some(obj) = node.as_object() else { return };
        let t = obj.get("type").and_then(Value::as_str).unwrap_or("");
        let text = obj.get("content").and_then(Value::as_str).unwrap_or("");
        let bbox = odl_box(obj.get("bounding box"));
        let mut push = |role: &'static str| {
            let text = norm_text(text);
            if !text.is_empty() {
                out.push(Elem { role, text, bbox });
            }
        };
        match t {
            "heading" => push("heading"),
            "paragraph" => push(if in_cell { "table_cell" } else { "paragraph" }),
            "caption" => push("caption"),
            "list item" => {
                push("list_item");
                walk(obj.get("kids").unwrap_or(&Value::Null), in_cell, out);
                return;
            }
            "list" => {
                walk(obj.get("list items").unwrap_or(&Value::Null), in_cell, out);
                return;
            }
            "header" | "footer" => return,
            "table" => {
                for row in obj
                    .get("rows")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    walk(row.get("cells").unwrap_or(&Value::Null), true, out);
                }
                return;
            }
            "table cell" => {
                walk(obj.get("kids").unwrap_or(&Value::Null), true, out);
                return;
            }
            _ => {}
        }
        walk(obj.get("kids").unwrap_or(&Value::Null), in_cell, out);
    }
    let mut out = Vec::new();
    walk(doc.get("kids").unwrap_or(&Value::Null), false, &mut out);
    out
}

fn reader_elements(chunks: &[Value]) -> Vec<Elem> {
    chunks
        .iter()
        .filter_map(|c| {
            let role = match c["kind"].as_str()? {
                "heading" | "title" => "heading",
                "paragraph" | "toc_entry" | "picture_text" | "code" | "footnote" => "paragraph",
                "list_item" => "list_item",
                "table_cell" => "table_cell",
                "caption" => "caption",
                _ => return None,
            };
            let text = norm_text(c["t"].as_str()?);
            if text.is_empty() {
                return None;
            }
            let f = |k: &str| c[k].as_f64().unwrap_or(0.0);
            Some(Elem {
                role,
                text,
                bbox: [f("l"), f("b"), f("r"), f("tp")],
            })
        })
        .collect()
}

fn tokens(s: &str) -> std::collections::HashSet<String> {
    s.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

fn iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let (il, ib, ir, it) = (
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    );
    if ir <= il || it <= ib {
        return 0.0;
    }
    let inter = (ir - il) * (it - ib);
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// Length of the longest increasing subsequence.
fn lis(seq: &[usize]) -> usize {
    let mut tails: Vec<usize> = Vec::new();
    for &x in seq {
        let i = tails.partition_point(|&t| t < x);
        if i == tails.len() {
            tails.push(x);
        } else {
            tails[i] = x;
        }
    }
    tails.len()
}

/// (elements, roles, order, boxes), as the Python evaluator defines them.
fn structure(theirs: &[Elem], ours: &[Elem]) -> Option<[f64; 4]> {
    if theirs.is_empty() {
        return None;
    }
    let our_tokens: Vec<_> = ours.iter().map(|e| tokens(&e.text)).collect();
    let mut matches: Vec<(usize, usize)> = Vec::new();
    for (i, e) in theirs.iter().enumerate() {
        let tt = tokens(&e.text);
        let mut best = (0.0f64, None);
        for (j, ot) in our_tokens.iter().enumerate() {
            if tt.is_empty() || ot.is_empty() {
                continue;
            }
            let inter = tt.intersection(ot).count() as f64;
            let jac = inter / tt.union(ot).count() as f64;
            if jac > best.0 {
                best = (jac, Some(j));
            }
        }
        if best.0 >= 0.6 {
            matches.push((i, best.1.unwrap()));
        }
    }
    if matches.is_empty() {
        return Some([0.0; 4]);
    }
    let n = matches.len() as f64;
    let same_role = matches
        .iter()
        .filter(|(i, j)| theirs[*i].role == ours[*j].role)
        .count() as f64;
    let order = lis(&matches.iter().map(|(_, j)| *j).collect::<Vec<_>>()) as f64;
    let boxes = matches
        .iter()
        .map(|(i, j)| iou(&theirs[*i].bbox, &ours[*j].bbox))
        .sum::<f64>()
        / n;
    Some([n / theirs.len() as f64, same_role / n, order / n, boxes])
}

#[test]
fn opendataloader_cases_hold() {
    if !ready() {
        return;
    }
    let dir = Path::new("tests/data/pdf/opendataloader");
    let cases: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("cases.json")).unwrap())
            .expect("cases.json parses");
    let converter = DocumentConverter::new();
    let mut outputs: HashMap<String, Output> = HashMap::new();
    let mut problems: Vec<String> = Vec::new();
    let (mut pass, mut xfail, mut unchecked, mut fixed) = (0, 0, 0, 0);
    let mut recalls: Vec<f64> = Vec::new();
    let mut structures: Vec<f64> = Vec::new();
    let mut improved: Vec<String> = Vec::new();
    for case in cases["cases"].as_array().unwrap() {
        let Some(pdf) = case["pdf"].as_str() else {
            continue;
        };
        if !outputs.contains_key(pdf) {
            let doc = converter
                .convert(SourceDocument::from_file(dir.join(pdf)).expect("fixture"))
                .unwrap_or_else(|e| panic!("{pdf}: {e}"))
                .document;
            let reader = doc.export_to_reader_value();
            let chunks = reader["pages"]
                .as_array()
                .map(|pages| {
                    pages
                        .iter()
                        .flat_map(|p| p.as_array().cloned().unwrap_or_default())
                        .collect()
                })
                .unwrap_or_default();
            let out = Output {
                md: doc.export_to_markdown(),
                chunks,
            };
            // Word recall against OpenDataLoader's Markdown, beside the fixture.
            let odl = dir.join(pdf).with_extension("odl.md");
            if let (Ok(theirs), Some(recorded)) =
                (std::fs::read_to_string(&odl), case["recall"].as_f64())
            {
                let g = words(&theirs);
                let recall = if g.is_empty() {
                    1.0
                } else {
                    g.intersection(&words(&out.md)).count() as f64 / g.len() as f64
                };
                recalls.push(recall);
                if recall + 0.05 < recorded {
                    problems.push(format!(
                        "{pdf}: word recall {recall:.2} fell from {recorded:.2}"
                    ));
                }
            }
            // Structural parity against OpenDataLoader's JSON, beside the fixture.
            let odl_json = dir.join(pdf).with_extension("odl.json");
            if let (Ok(text), Some(recorded)) = (
                std::fs::read_to_string(&odl_json),
                case["structure"].as_object(),
            ) {
                if let Ok(theirs) = serde_json::from_str::<Value>(&text) {
                    if let Some(now) =
                        structure(&odl_elements(&theirs), &reader_elements(&out.chunks))
                    {
                        for (k, v) in ["elements", "roles", "order", "boxes"].iter().zip(now) {
                            let was = recorded.get(*k).and_then(Value::as_f64).unwrap_or(0.0);
                            structures.push(v);
                            if v + 0.1 < was {
                                problems.push(format!(
                                    "{pdf}: structural {k} {v:.2} fell from {was:.2}"
                                ));
                            } else if v > was + 0.1 {
                                improved.push(format!("{pdf}: structural {k} {v:.2} up from {was:.2} — regenerate cases.json"));
                            }
                        }
                    }
                }
            }
            outputs.insert(pdf.to_string(), out);
        }
        let out = &outputs[pdf];
        let status = case["status"].as_str().unwrap_or("");
        let id = case["id"].as_str().unwrap_or("?");
        match status {
            "unchecked" | "no-output" => unchecked += 1,
            "pass" => {
                let fails = check(case, out);
                if fails.is_empty() {
                    pass += 1;
                } else {
                    problems.push(format!("{id} ({pdf}) regressed: {}", fails.join("; ")));
                }
            }
            "xfail" => {
                let fails = check(case, out);
                if fails.is_empty() {
                    fixed += 1;
                    problems.push(format!(
                        "{id} ({pdf}) now passes — set its status to \"pass\" in cases.json"
                    ));
                } else {
                    xfail += 1;
                }
            }
            other => problems.push(format!("{id}: unknown status {other:?}")),
        }
    }
    let mean = if recalls.is_empty() {
        0.0
    } else {
        recalls.iter().sum::<f64>() / recalls.len() as f64
    };
    let smean = if structures.is_empty() {
        0.0
    } else {
        structures.iter().sum::<f64>() / structures.len() as f64
    };
    eprintln!(
        "opendataloader corpus: {pass} pass, {xfail} xfail, {fixed} newly passing, {unchecked} unchecked; \
         mean word recall {mean:.3} over {} fixtures; mean structural parity {smean:.3}",
        recalls.len()
    );
    for line in &improved {
        eprintln!("improved: {line}");
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
