//! Display formulas found from the glyphs themselves, not the layout model's
//! label. On a digital page the text layer says which glyphs are math: TeX
//! sets every math variable in math italic, relations and operators in the
//! symbol face, big operators in the extension face
//! ([`crate::formula_text::Face`]), and prose never uses them. A region the
//! layout model labelled text (or a heading) is read row by row; a run of
//! rows with no prose word and with math in it is a display formula, and
//! becomes a formula region of its own when [`formula_text::explain`]
//! rebuilds it — otherwise nothing changes, so a guess never reaches the
//! output. Measured on pdf-parse-bench (`scripts/conformance/pdf_parse_bench.py`).

use crate::formula_text::{self, Face, MathGlyph, MathLayer};

/// One piece of a region the detector split, top to bottom.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    /// `Some(latex)`: a display formula, rebuilt; `None`: the region's own
    /// content (prose) in this band.
    pub latex: Option<String>,
    pub bbox: [f32; 4],
}

/// Words that are math in an upright face (function names, `mod`): never
/// prose.
const MATH_WORDS: [&str; 34] = [
    "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "sin", "cos", "tan", "cot", "sec", "csc",
    "log", "ln", "lg", "exp", "lim", "liminf", "limsup", "max", "min", "det", "deg", "gcd", "lcm",
    "sup", "inf", "mod", "dim", "ker", "arg", "Pr", "hom", "tr",
];

/// Whether `c` reads as math wherever it appears: an operator, relation,
/// arrow or Greek letter.
fn math_char(c: char) -> bool {
    matches!(c, '=' | '+' | '<' | '>' | '±' | '×' | '÷' | '√' | '∞' | '∓')
        || ('\u{0391}'..='\u{03C9}').contains(&c)
        || ('\u{2190}'..='\u{22FF}').contains(&c)
        || ('\u{27C0}'..='\u{27FF}').contains(&c)
        || ('\u{2980}'..='\u{2AFF}').contains(&c)
}

/// The vertical position a glyph belongs to its row by: the baseline, or
/// the centre of an extension glyph (it hangs from its baseline).
fn row_y(g: &MathGlyph) -> f32 {
    if g.hangs() || g.base.is_nan() {
        (g.t + g.b) / 2.0
    } else {
        g.base
    }
}

struct Row {
    y: f32,
    glyphs: Vec<usize>,
}

/// The display formulas in a region the layout model did not label a
/// formula (`bbox`: its box, top-left page points): `None` when it holds
/// none (or none that rebuilds), otherwise the region cut into pieces, top
/// to bottom — prose bands keep the region's label, formula bands carry
/// their LaTeX.
pub fn display_pieces(layer: &MathLayer, bbox: [f32; 4]) -> Option<Vec<Piece>> {
    const SLACK: f32 = 1.0;
    let [l, t, r, b] = bbox;
    let glyphs: Vec<&MathGlyph> = layer
        .glyphs
        .iter()
        .filter(|g| {
            let (x, y) = ((g.l + g.r) / 2.0, (g.t + g.b) / 2.0);
            x >= l - SLACK && x <= r + SLACK && y >= t - SLACK && y <= b + SLACK
        })
        .collect();
    if glyphs.len() < 2 || glyphs.iter().any(|g| g.tilted) {
        return None;
    }
    // Nothing set in a math face, no math symbol: nothing to look for.
    if !glyphs
        .iter()
        .any(|g| matches!(g.face, Face::Italic | Face::Math | Face::Extension) || math_char(g.ch))
    {
        return None;
    }
    let script = scripts(&glyphs);
    let main = formula_text::median(
        (0..glyphs.len())
            .filter(|&i| !script[i])
            .map(|i| glyphs[i].size)
            .collect(),
    );
    // Rows: the glyphs on a baseline of their own (not scripts) by
    // baseline; a script joins the row nearest to it.
    let mut order: Vec<usize> = (0..glyphs.len()).filter(|&i| !script[i]).collect();
    order.sort_by(|&a, &b| row_y(glyphs[a]).total_cmp(&row_y(glyphs[b])));
    let mut rows: Vec<Row> = Vec::new();
    for i in order {
        let y = row_y(glyphs[i]);
        match rows.last_mut() {
            Some(row) if y - row.y < 0.35 * main => row.glyphs.push(i),
            _ => rows.push(Row { y, glyphs: vec![i] }),
        }
    }
    if rows.is_empty() {
        return None;
    }
    for (i, g) in glyphs.iter().enumerate() {
        if !script[i] {
            continue;
        }
        let y = (g.t + g.b) / 2.0;
        let k = (0..rows.len())
            .min_by(|&a, &b| (rows[a].y - y).abs().total_cmp(&(rows[b].y - y).abs()))
            .unwrap_or(0);
        rows[k].glyphs.push(i);
    }
    let prose: Vec<bool> = rows
        .iter()
        .map(|row| has_prose_word(&glyphs, &row.glyphs, &script))
        .collect();
    // Formula blocks: maximal runs of rows without a prose word that hold
    // math (two signals at least — one italic letter is a variable named
    // in passing as often as a formula).
    let mut blocks: Vec<(usize, usize)> = Vec::new();
    let mut k = 0;
    while k < rows.len() {
        if prose[k] {
            k += 1;
            continue;
        }
        let start = k;
        while k < rows.len() && !prose[k] {
            k += 1;
        }
        let ids: Vec<usize> = rows[start..k]
            .iter()
            .flat_map(|row| row.glyphs.iter().copied())
            .collect();
        let signals = ids
            .iter()
            .filter(|&&i| {
                let g = glyphs[i];
                matches!(g.face, Face::Italic | Face::Math | Face::Extension) || math_char(g.ch)
            })
            .count();
        // A block of script-size glyphs only is the exponent or the limits
        // of something on a prose line (`2^{1+⌊…⌋}`, `∏_{i∈I}`), never a
        // display of its own.
        let full = ids
            .iter()
            .any(|&i| glyphs[i].size >= 0.85 * main && glyphs[i].face != Face::Extension);
        if signals >= 2 && ids.len() >= 3 && full {
            blocks.push((start, k));
        }
    }
    if blocks.is_empty() {
        return None;
    }
    let extent = |ids: &mut dyn Iterator<Item = usize>| -> [f32; 4] {
        let mut e = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        for i in ids {
            let g = glyphs[i];
            e = [e[0].min(g.l), e[1].min(g.t), e[2].max(g.r), e[3].max(g.b)];
        }
        e
    };
    let mut pieces: Vec<Piece> = Vec::new();
    let mut any = false;
    let mut k = 0;
    let flush_text = |pieces: &mut Vec<Piece>, from: usize, to: usize| {
        let mut ids = rows[from..to]
            .iter()
            .flat_map(|row| row.glyphs.iter().copied());
        let [_, et, _, eb] = extent(&mut ids);
        pieces.push(Piece {
            latex: None,
            bbox: [l, et, r, eb],
        });
    };
    for &(start, end) in &blocks {
        let mut ids = rows[start..end]
            .iter()
            .flat_map(|row| row.glyphs.iter().copied());
        let mut fb = extent(&mut ids);
        // The rules inside the block's rows (fraction bars, vincula).
        for ru in &layer.rules {
            let y = (ru.t + ru.b) / 2.0;
            if y >= fb[1] && y <= fb[3] && ru.r > fb[0] && ru.l < fb[2] {
                fb = [fb[0].min(ru.l), fb[1], fb[2].max(ru.r), fb[3]];
            }
        }
        // Several formulas in one block (consecutive displays) read one by
        // one, like a layout box around them.
        let bands = match formula_text::split_rows(layer, fb) {
            v if v.len() >= 2 => v,
            _ => vec![fb],
        };
        // A display has its lines to itself: a glyph of the region beside a
        // band (prose left of a text-style fraction whose numerator and
        // denominator made rows of their own) means the formula is inline,
        // and cutting it out would take that prose with it.
        let alone = |bb: &[f32; 4]| {
            !glyphs.iter().any(|g| {
                let (x, y) = ((g.l + g.r) / 2.0, (g.t + g.b) / 2.0);
                y > bb[1] && y < bb[3] && (x < bb[0] - 1.0 || x > bb[2] + 1.0)
            })
        };
        let read: Vec<Option<String>> = bands
            .iter()
            .map(|bb| {
                alone(bb)
                    .then(|| formula_text::explain(layer, *bb).ok())
                    .flatten()
            })
            .collect();
        if read.iter().all(Option::is_none) {
            continue;
        }
        // The text rows before the block.
        if k < start {
            flush_text(&mut pieces, k, start);
        }
        for (bb, latex) in bands.into_iter().zip(read) {
            any |= latex.is_some();
            pieces.push(Piece { latex, bbox: bb });
        }
        k = end;
    }
    if !any {
        return None;
    }
    if k < rows.len() {
        flush_text(&mut pieces, k, rows.len());
    }
    Some(pieces)
}

/// Which glyphs are scripts: set smaller than the glyph just before them
/// and off its baseline (`ν_{initial}`, `x^2`, a footnote mark), or
/// continuing such a script at its size and baseline (the rest of
/// `initial`). Decided glyph by glyph rather than by one size for the
/// region: a region's largest glyph can be a big symbol, its most common
/// size a subscript word longer than its base.
fn scripts(glyphs: &[&MathGlyph]) -> Vec<bool> {
    let mut order: Vec<usize> = (0..glyphs.len()).collect();
    order.sort_by(|&a, &b| glyphs[a].l.total_cmp(&glyphs[b].l));
    let mut script = vec![false; glyphs.len()];
    for (k, &i) in order.iter().enumerate() {
        let g = glyphs[i];
        if g.face == Face::Extension || g.base.is_nan() {
            continue;
        }
        // The nearest glyph ending just before this one, on its line.
        let prev = order[..k]
            .iter()
            .rev()
            .copied()
            .filter(|&j| {
                let p = glyphs[j];
                p.r <= g.l + 0.3 * g.size
                    && g.l - p.r < 0.5 * p.size.max(g.size)
                    && (row_y(p) - g.base).abs() < 0.8 * p.size.max(g.size)
            })
            .max_by(|&a, &b| glyphs[a].r.total_cmp(&glyphs[b].r));
        let Some(j) = prev else { continue };
        let p = glyphs[j];
        let shifted = (p.base - g.base).abs() > 0.1 * g.size || p.hangs();
        script[i] = (p.size >= 1.15 * g.size && shifted)
            || (script[j] && (p.size - g.size).abs() < 0.05 * g.size && !shifted);
    }
    script
}

/// Whether a row holds prose: a word of two letters or more, some of them
/// lowercase, set in a text face on the line (not a script), that is not a
/// function name. (A word in a script — `I_{rotation}` — does not count.)
fn has_prose_word(glyphs: &[&MathGlyph], ids: &[usize], script: &[bool]) -> bool {
    let mut letters: Vec<&MathGlyph> = ids
        .iter()
        .filter(|&&i| !script[i])
        .map(|&i| glyphs[i])
        .filter(|g| g.face == Face::Text && g.ch.is_alphabetic())
        .collect();
    letters.sort_by(|a, b| a.l.total_cmp(&b.l));
    let signals = ids
        .iter()
        .filter(|&&i| {
            let g = glyphs[i];
            matches!(g.face, Face::Italic | Face::Math | Face::Extension) || math_char(g.ch)
        })
        .count();
    let mut words: Vec<String> = Vec::new();
    let mut word = String::new();
    let mut last: Option<&MathGlyph> = None;
    for g in letters {
        let joined = last.is_some_and(|p| {
            // Letter boxes are their advances, so the letters of a word
            // touch; a justified line's word space can be as narrow as
            // 0.2 em.
            g.l - p.r < 0.08 * g.size && (g.base - p.base).abs() < 0.2 * g.size
        });
        if !joined && !word.is_empty() {
            words.push(std::mem::take(&mut word));
        }
        word.push(g.ch);
        last = Some(g);
    }
    if !word.is_empty() {
        words.push(word);
    }
    let words: Vec<&String> = words
        .iter()
        .filter(|w| w.chars().count() >= 2 && !MATH_WORDS.contains(&w.as_str()))
        .collect();
    // A capitalised word is prose too: an operator name (`Core`) and the
    // first word of the next sentence on the formula's line (`… = 1 While`)
    // look the same, and the second must not be read into the formula.
    // All-capital runs (`NFA`, `SRSWOR`) are names; with no math around
    // them they are prose as well.
    words
        .iter()
        .any(|w| w.chars().any(char::is_lowercase) || signals == 0)
}

/// The layout labels a display formula is looked for in: the regions that
/// hold running text (a formula the model took for prose or a heading).
const SCANNED: [&str; 9] = [
    "text",
    "page_header",
    "page_footer",
    "paragraph",
    "section_header",
    "title",
    "list_item",
    "caption",
    "footnote",
];

/// Every region of a page (`regions`: label and box) that holds a display
/// formula, with the pieces it splits into ([`display_pieces`]). A region
/// that a formula region overlaps is left alone: the layout model found
/// that formula, and the formula pass reads it.
pub fn detect(layer: &MathLayer, regions: &[(&str, [f32; 4])]) -> Vec<(usize, Vec<Piece>)> {
    let area = |b: &[f32; 4]| ((b[2] - b[0]) * (b[3] - b[1])).max(1e-3);
    let inter = |a: &[f32; 4], b: &[f32; 4]| {
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0)
    };
    regions
        .iter()
        .enumerate()
        .filter(|(_, (label, _))| SCANNED.contains(label))
        .filter(|(_, (_, bb))| {
            !regions.iter().any(|(label, fb)| {
                *label == "formula" && inter(bb, fb) > 0.1 * area(bb).min(area(fb))
            })
        })
        // Text inside a table or a picture is a cell or a label: the table
        // reads it.
        .filter(|(_, (_, bb))| {
            !regions.iter().any(|(label, tb)| {
                matches!(*label, "table" | "picture" | "document_index" | "form")
                    && inter(bb, tb) > 0.5 * area(bb)
            })
        })
        .filter_map(|(i, (_, bb))| display_pieces(layer, *bb).map(|p| (i, p)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula_text::Rule;

    /// A glyph at `x` on baseline `base` (top-left frame), `size` points,
    /// in `face`: 0.55 em advance, 0.75 em ascent, 0.25 em descent.
    fn g(ch: char, x: f32, base: f32, size: f32, face: Face) -> MathGlyph {
        MathGlyph {
            ch,
            name: None,
            l: x,
            t: base - 0.75 * size,
            r: x + 0.55 * size,
            b: base + 0.25 * size,
            base,
            size,
            tilted: false,
            face,
        }
    }

    /// A line of words: letters touch, words are 0.3 em apart; a word in
    /// `[…]` is set in math italic.
    fn line(s: &str, x: f32, base: f32, size: f32) -> Vec<MathGlyph> {
        let mut out = Vec::new();
        let mut at = x;
        for word in s.split(' ') {
            let (w, face) = match word.strip_prefix('[').and_then(|w| w.strip_suffix(']')) {
                Some(w) => (w, Face::Italic),
                None => (word, Face::Text),
            };
            for c in w.chars() {
                let face = if c.is_alphabetic() { face } else { Face::Text };
                out.push(g(c, at, base, size, face));
                at += 0.55 * size;
            }
            at += 0.3 * size;
        }
        out
    }

    fn pieces(glyphs: Vec<MathGlyph>) -> Option<Vec<Piece>> {
        let layer = MathLayer {
            glyphs,
            rules: Vec::<Rule>::new(),
        };
        display_pieces(&layer, [0.0, 0.0, 600.0, 800.0])
    }

    #[test]
    fn a_display_between_prose_lines_is_cut_out() {
        let mut gs = line("the sum of two terms", 50.0, 100.0, 10.0);
        gs.extend(line("[x] = [a] + [b]", 120.0, 125.0, 10.0));
        gs.extend(line("where both are real", 50.0, 150.0, 10.0));
        let p = pieces(gs).expect("a formula");
        assert_eq!(p.len(), 3);
        assert!(p[0].latex.is_none() && p[2].latex.is_none());
        assert_eq!(p[1].latex.as_deref(), Some("x = a + b"));
    }

    #[test]
    fn a_formula_sharing_its_line_with_prose_is_left_alone() {
        // `… = 1 While` — the next sentence starts on the formula's line.
        let mut gs = line("the sum of two terms", 50.0, 100.0, 10.0);
        gs.extend(line("[x] = [a] + 1 While", 50.0, 125.0, 10.0));
        assert_eq!(pieces(gs), None);
    }

    #[test]
    fn a_subscript_word_is_not_prose() {
        // `ν_{initial} = c`: the subscript outnumbers its base.
        let mut gs = vec![g('ν', 100.0, 100.0, 11.0, Face::Italic)];
        gs.extend(line("initial", 106.0, 102.5, 7.5));
        gs.extend(line("= [c]", 140.0, 100.0, 11.0));
        let p = pieces(gs).expect("a formula");
        assert_eq!(p.len(), 1);
        assert!(p[0]
            .latex
            .as_deref()
            .is_some_and(|l| l.starts_with("\\nu _ {")));
    }

    #[test]
    fn a_row_of_scripts_alone_is_no_display() {
        // The limits of a `∏` on a prose line, on a row of their own.
        let mut gs = line("member of the product", 50.0, 100.0, 10.0);
        gs.extend(line("[i] ∈ [I]", 80.0, 109.0, 7.0));
        gs.extend(line("over every index", 50.0, 125.0, 10.0));
        assert_eq!(pieces(gs), None);
    }

    #[test]
    fn prose_alone_is_left_alone() {
        let mut gs = line("plain words", 50.0, 100.0, 10.0);
        gs.extend(line("and more words", 50.0, 115.0, 10.0));
        assert_eq!(pieces(gs), None);
    }

    #[test]
    fn faces_by_font_name() {
        assert_eq!(Face::of_font_name("ABCDEF+CMMI10"), Face::Italic);
        assert_eq!(Face::of_font_name("LMMathItalic10-Regular"), Face::Italic);
        assert_eq!(Face::of_font_name("CMSY7"), Face::Symbol);
        assert_eq!(Face::of_font_name("MSBM10"), Face::Symbol);
        assert_eq!(Face::of_font_name("CMEX10"), Face::Extension);
        assert_eq!(
            Face::of_font_name("LMMathExtension10-Regular"),
            Face::Extension
        );
        assert_eq!(Face::of_font_name("CambriaMath"), Face::Math);
        assert_eq!(Face::of_font_name("STIXTwoMath-Regular"), Face::Math);
        assert_eq!(Face::of_font_name("CMR10"), Face::Text);
        assert_eq!(Face::of_font_name("LMRoman10-Regular"), Face::Text);
        assert_eq!(Face::of_font_name("TimesNewRomanPSMT"), Face::Text);
    }
}
