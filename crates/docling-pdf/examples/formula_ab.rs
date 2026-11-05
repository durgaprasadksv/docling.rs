//! `cargo run --release -p docling-pdf --example formula_ab -- crop1.png crop2.png …`
//!
//! Runs the code/formula VLM on crops dumped by `DOCLING_RS_ENRICH_DEBUG`
//! and prints, per crop, the decode time and the post-processed LaTeX.
//! Point `DOCLING_CODE_FORMULA_DIR` at a model directory to A/B recognizers
//! (CodeFormulaV2 vs. a granite-docling export) on identical inputs.

use std::time::Instant;

use docling_pdf::enrich::{CodeFormula, CodeFormulaKind};

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: formula_ab <crop.png>…");
        std::process::exit(2);
    }
    let threads = docling_core::env::parse::<usize>("DOCLING_RS_TF_INTRA").unwrap_or(4);
    let t0 = Instant::now();
    let mut model = CodeFormula::load_with(threads).expect("load code/formula model");
    eprintln!("model loaded in {:.1}s", t0.elapsed().as_secs_f32());
    for p in &paths {
        let crop = image::open(p).expect("open crop").to_rgb8();
        let t = Instant::now();
        let out = model.predict(&crop, CodeFormulaKind::Formula);
        let dt = t.elapsed().as_secs_f32();
        match out {
            Ok(latex) => println!("{p}\t{dt:.1}s\t{latex}"),
            Err(e) => println!("{p}\t{dt:.1}s\tERROR {e}"),
        }
    }
}
