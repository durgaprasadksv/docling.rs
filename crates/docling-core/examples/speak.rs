//! Speak LaTeX from stdin, one formula per line, as the JSON `speech` map
//! would — a quick way to hear what a textbook formula becomes.
//!
//! ```sh
//! echo 'M g ^ { 2 + } ( a q ) + 2 O H ^ { - } ( a q )' \
//!   | cargo run -p docling-core --features speech --example speak
//! ```
//!
//! Needs MathCAT's rules at `.models/mathcat` (or `DOCLING_RS_MATHCAT_RULES`).
fn main() {
    use std::io::BufRead;
    for line in std::io::stdin().lock().lines() {
        let line = line.expect("stdin");
        let latex = line.trim();
        if latex.is_empty() || latex.starts_with('#') {
            continue;
        }
        println!("{latex}");
        match docling_core::speech::speak_latex(latex) {
            Some(styles) => {
                for (style, text) in styles {
                    println!("  {style:<12}{text}");
                }
            }
            None => println!("  (no spoken form)"),
        }
    }
}
