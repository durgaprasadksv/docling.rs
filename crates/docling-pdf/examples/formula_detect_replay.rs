//! Replay the display-formula detector over saved pages — no PDF, no
//! models. `formula_detect_replay <page-dump>...` reads
//! `DOCLING_RS_FORMULA_PAGE_DUMP` records and prints one JSON line per
//! region the detector splits: `{"page", "region", "label", "bbox",
//! "pieces": [{"latex", "bbox"}]}`.
use docling_pdf::formula_detect::detect;
use docling_pdf::formula_text::layer_from_json;

fn main() {
    for arg in std::env::args().skip(1) {
        let rec: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&arg).expect("page dump")).expect("JSON");
        let Some((layer, _)) = layer_from_json(&rec) else {
            eprintln!("{arg}: unreadable");
            continue;
        };
        let regions: Vec<(String, [f32; 4])> = rec["regions"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|r| {
                        let f = |k: usize| r[k].as_f64().map(|v| v as f32);
                        Some((r[0].as_str()?.to_string(), [f(1)?, f(2)?, f(3)?, f(4)?]))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let refs: Vec<(&str, [f32; 4])> = regions.iter().map(|(l, b)| (l.as_str(), *b)).collect();
        for (i, pieces) in detect(&layer, &refs) {
            let pieces: Vec<serde_json::Value> = pieces
                .iter()
                .map(|p| serde_json::json!({"latex": p.latex, "bbox": p.bbox}))
                .collect();
            println!(
                "{}",
                serde_json::json!({"page": arg, "region": i, "label": refs[i].0, "bbox": refs[i].1, "pieces": pieces})
            );
        }
    }
}
