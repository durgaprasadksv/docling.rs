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
        if !lines.iter().any(|l| {
            let nl = norm(l);
            g.iter().all(|s| nl.contains(&norm(s)))
        }) {
            fails.push(format!("same_line: {g:?} not on one line"));
        }
    }
    let items: Vec<&Value> = out
        .chunks
        .iter()
        .filter(|c| c["kind"] == "list_item")
        .collect();
    let item_text = |c: &Value| norm(c["t"].as_str().unwrap_or(""));
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
    eprintln!(
        "opendataloader corpus: {pass} pass, {xfail} xfail, {fixed} newly passing, {unchecked} unchecked; \
         mean word recall {mean:.3} over {} fixtures",
        recalls.len()
    );
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
