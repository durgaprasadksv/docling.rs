//! Replay `formula_text` over saved formula regions — no PDF, no models.
//!
//! `formula_text_replay <cases.json | dump-dir>...` reads either the
//! regression fixtures (`tests/data/pdf/formula_text/cases.json`) or a
//! directory of `DOCLING_RS_FORMULA_TEXT_DUMP` records, rebuilds each region
//! with the current code, and prints one JSON line per region:
//! `{"id", "latex", "error"}`. `scripts/conformance/formula_text_eval.py
//! --replay` scores the lines against the ground truth.
use docling_pdf::formula_text::{explain, layer_from_json};

fn main() {
    for arg in std::env::args().skip(1) {
        let path = std::path::Path::new(&arg);
        let records: Vec<(String, serde_json::Value)> = if path.is_dir() {
            let mut files: Vec<_> = std::fs::read_dir(path)
                .expect("dump dir")
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "json"))
                .collect();
            files.sort();
            files
                .into_iter()
                .map(|p| {
                    let v = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
                    (p.display().to_string(), v)
                })
                .collect()
        } else {
            let v: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            v.as_array()
                .expect("a JSON array of cases")
                .iter()
                .map(|c| (c["id"].as_str().unwrap_or_default().to_string(), c.clone()))
                .collect()
        };
        for (id, rec) in records {
            let Some((layer, bbox)) = layer_from_json(&rec) else {
                eprintln!("{id}: unreadable record");
                continue;
            };
            let out = explain(&layer, bbox);
            println!(
                "{}",
                serde_json::json!({"id": id, "latex": out.as_ref().ok(), "error": out.as_ref().err()})
            );
        }
    }
}
