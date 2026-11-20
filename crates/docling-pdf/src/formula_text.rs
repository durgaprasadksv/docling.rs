//! LaTeX for a formula region rebuilt from the PDF's own text layer — no model.
//!
//! A formula typeset as text (TeX's Computer Modern faces, a word processor's
//! equation runs) already carries everything the recognizer reads off the
//! pixels: every symbol, its exact size, and its baseline. What the flat text
//! layer loses is the *structure* — `a2 + 8 = 12` for `a^2 + 8 = 12` — and that
//! structure is geometry: a script is a smaller glyph raised or lowered from
//! its base's baseline, a fraction is two stacks either side of a drawn rule,
//! a square root is a radical glyph with a rule over its body. This module
//! rebuilds it, which is microseconds of arithmetic where the CodeFormula VLM
//! is seconds of GPU per formula, and exact where the VLM can misread a glyph.
//!
//! It is deliberately conservative: [`reconstruct`] returns `None` — and the
//! caller falls back to the VLM — whenever anything in the region is left
//! unexplained: a glyph without a usable character (a TeX extensible-delimiter
//! piece, an unmapped symbol), a rotated glyph, a drawn rule that is neither a
//! fraction bar, a vinculum nor an over/underline, two full-size glyphs
//! stacked on each other (a matrix, `\overset`), a glyph whose baseline is
//! unknown. A formula drawn as vector outlines (MathJax output printed to PDF)
//! has no glyphs at all and is never touched.
//!
//! The output is spelled the way docling's CodeFormula model spells its
//! answers — space-separated tokens, digits one by one, `^ { 2 }` — so a
//! formula rebuilt here reads byte-identical to the model's on the formulas
//! the two agree on (`code_and_formula.pdf`'s `a ^ { 2 } + 8 = 1 2`), and the
//! speech chain sees the same LaTeX dialect either way.

use crate::checkbox::Ink;
use crate::pdfium_backend::Glyph;

/// One glyph of the page in top-left page points (the cells' frame).
#[derive(Debug, Clone, PartialEq)]
pub struct MathGlyph {
    /// The glyph's character: the text layer's, or — for a code without
    /// ToUnicode that the font program names — the reading of `name`.
    pub ch: char,
    /// That program glyph name, kept so the dump and the regression
    /// fixtures re-read it (`layer_from_json`) when the name table changes.
    pub name: Option<std::sync::Arc<str>>,
    /// The glyph's loose box (advance × font ascent/descent).
    pub l: f32,
    pub t: f32,
    pub r: f32,
    pub b: f32,
    /// Baseline y (the pen's, rise included).
    pub base: f32,
    /// Font size in points.
    pub size: f32,
    /// Set on a glyph whose baseline is not upright (rotated or tilted text):
    /// a region holding one is not rebuilt.
    pub tilted: bool,
}

impl MathGlyph {
    /// A TeX extension-font glyph (a display `∑`, a `\Big(`) hangs from
    /// its baseline: its real extent is about zero above it and a fixed
    /// depth below, which the font's generic ascent/descent box does not
    /// show (a display `∑` reads a full em above its row). Rebuild the box
    /// from the glyph's standard depth, so the glyph sits where it is drawn
    /// — centred on the math axis, where TeX puts it. Idempotent: the box is
    /// recomputed from the baseline, never from itself.
    fn with_true_extent(mut self) -> Self {
        if let Some(depth) = self.name.as_deref().and_then(extension_depth) {
            self.t = self.base;
            self.b = self.base + depth * self.size;
        }
        self
    }

    /// Whether this glyph is an extension-font glyph placed by its centre,
    /// not its baseline.
    fn hangs(&self) -> bool {
        self.name.as_deref().and_then(extension_depth).is_some()
    }
}

/// The depth below the baseline, in em, of a sized TeX extension glyph
/// (Computer Modern's CMEX metrics, which Latin Modern and the AMS fonts
/// share closely enough to place a glyph on its row); `None` for any other
/// glyph.
fn extension_depth(name: &str) -> Option<f32> {
    let name = name.trim_end_matches(|c: char| c.is_ascii_digit());
    let op = |stem: &str| {
        [
            "summation",
            "product",
            "coproduct",
            "union",
            "intersection",
            "circleplus",
            "circlemultiply",
            "circledot",
            "unionmulti",
            "logicaland",
            "logicalor",
        ]
        .contains(&stem)
    };
    if let Some(stem) = name.strip_suffix("display") {
        return match stem {
            "integral" | "contintegral" => Some(2.22),
            s if op(s) => Some(1.4),
            _ => None,
        };
    }
    if let Some(stem) = name.strip_suffix("text") {
        return match stem {
            "integral" | "contintegral" => Some(1.11),
            s if op(s) => Some(1.0),
            _ => None,
        };
    }
    for (suffix, depth) in [("Bigg", 3.0), ("bigg", 2.4), ("Big", 1.8), ("big", 1.2)] {
        if name.ends_with(suffix) {
            return Some(depth);
        }
    }
    None
}

/// A thin horizontal rule painted on the page (a fraction bar, a radical's
/// vinculum, an overline), top-left page points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rule {
    pub l: f32,
    pub t: f32,
    pub r: f32,
    pub b: f32,
}

/// What [`reconstruct`] reads: a page's glyphs and thin rules.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MathLayer {
    pub glyphs: Vec<MathGlyph>,
    pub rules: Vec<Rule>,
}

/// Thickest stroke or fill read as a rule, points (TeX's default rule is
/// 0.4 pt; a heavy display bar in a large size stays well under this).
const MAX_RULE_THICKNESS: f64 = 2.0;

impl MathLayer {
    /// The layer of a page whose glyphs and painted pieces the text parser
    /// walked (`h`: page height, for the y-flip).
    pub(crate) fn new(glyphs: &[Glyph], inks: &[Ink], h: f32) -> Self {
        // Whitespace is dropped after the glyph's name is read, not before:
        // a code whose base-encoding guess is a space can be a `ψ`
        // (`u1D713`) — dropping it first lost the symbol silently.
        let glyphs = glyphs
            .iter()
            .filter(|g| g.ll.is_finite() && g.lt.is_finite())
            .filter(|g| {
                let named = g.unmapped_math && g.program_name.is_some();
                named || !g.ch.is_whitespace()
            })
            .map(|g| {
                let (base, size) = match g.pen {
                    Some([_, y, size]) => (h - y, size),
                    None => (f32::NAN, g.lt - g.lb),
                };
                // A code without ToUnicode in a font whose Type1 program
                // names it: the character the text layer shows is a
                // base-encoding guess (CMEX's bracket pieces read as
                // digits), so the program's glyph name decides, and a name
                // with no reading marks the glyph unreadable. A code the
                // program's encoding vector does not list (a font on
                // StandardEncoding names only its exceptions) keeps the
                // encoding's character, which is then the right one.
                let name = g.program_name.clone().filter(|_| g.unmapped_math);
                let ch = match name.as_deref() {
                    Some(name) => read_name(name, g.ch),
                    None => g.ch,
                };
                MathGlyph {
                    ch,
                    name,
                    l: g.ll,
                    t: h - g.lt,
                    r: g.lr,
                    b: h - g.lb,
                    base,
                    size,
                    tilted: g.quad.is_some(),
                }
                .with_true_extent()
            })
            .filter(|m| !m.ch.is_whitespace())
            .collect();
        let rules = inks
            .iter()
            .filter_map(|ink| {
                let (l, b, r, t) = match *ink {
                    Ink::Seg { x0, y0, x1, y1 } if (y0 - y1).abs() < 0.25 => {
                        // A stroked line: its width is the pen's, unknown
                        // here — give it a hairline's thickness.
                        let y = (y0 + y1) / 2.0;
                        (x0.min(x1), y - 0.2, x0.max(x1), y + 0.2)
                    }
                    Ink::Rect { l, b, r, t } | Ink::Blot { l, b, r, t } => (l, b, r, t),
                    Ink::Seg { .. } => return None,
                };
                let (w, th) = (r - l, t - b);
                (th <= MAX_RULE_THICKNESS && w >= 1.5 && w >= 3.0 * th).then_some(Rule {
                    l: l as f32,
                    t: h - t as f32,
                    r: r as f32,
                    b: h - b as f32,
                })
            })
            .collect();
        Self { glyphs, rules }
    }

    /// Whether the layer has anything to rebuild from.
    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }
}

#[cfg(feature = "ml")]
/// One formula region as JSON — its glyphs and rules (relative to nothing:
/// page points) and the outcome — for the evaluation harness
/// (`DOCLING_RS_FORMULA_TEXT_DUMP=<dir>`, `scripts/conformance/formula_text_eval.py`)
/// and the regression fixtures it writes.
pub fn dump_json(
    layer: &MathLayer,
    bbox: [f32; 4],
    outcome: &Result<String, String>,
) -> serde_json::Value {
    const SLACK: f32 = 1.0;
    let [l, t, r, b] = bbox;
    let inside =
        |x: f32, y: f32| x >= l - SLACK && x <= r + SLACK && y >= t - SLACK && y <= b + SLACK;
    let r2 = |v: f32| (v * 100.0).round() / 100.0;
    // The formula's glyphs and, for the guards that read the surroundings
    // (`continues_outside`), every glyph within 12 pt of the box: a replay
    // selects the formula from them exactly as `explain` does on the page.
    let near = |g: &MathGlyph| {
        let (x, y) = ((g.l + g.r) / 2.0, (g.t + g.b) / 2.0);
        x >= l - 12.0 && x <= r + 12.0 && y >= t - 12.0 && y <= b + 12.0
    };
    let mut kept = select(layer, bbox);
    let around: Vec<MathGlyph> = layer
        .glyphs
        .iter()
        .filter(|g| {
            near(g)
                && !kept
                    .iter()
                    .any(|k| k.l == g.l && k.t == g.t && k.ch == g.ch)
        })
        .cloned()
        .collect();
    kept.extend(around);
    let glyphs: Vec<serde_json::Value> = kept
        .iter()
        .map(|g| {
            serde_json::json!([
                g.ch.to_string(),
                r2(g.l),
                r2(g.t),
                r2(g.r),
                r2(g.b),
                r2(g.base),
                r2(g.size),
                g.tilted,
                g.name.as_deref()
            ])
        })
        .collect();
    // Rules up to 12 pt above or below the box too: `explain` reads a bar
    // just outside it as a cut formula, and a replay must see it.
    let rules: Vec<serde_json::Value> = layer
        .rules
        .iter()
        .filter(|ru| {
            let (x, y) = ((ru.l + ru.r) / 2.0, (ru.t + ru.b) / 2.0);
            inside(x, y) || (x >= l - SLACK && x <= r + SLACK && y >= t - 12.0 && y <= b + 12.0)
        })
        .map(|ru| serde_json::json!([r2(ru.l), r2(ru.t), r2(ru.r), r2(ru.b)]))
        .collect();
    serde_json::json!({
        "bbox": bbox.map(r2),
        "glyphs": glyphs,
        "rules": rules,
        "latex": outcome.as_ref().ok(),
        "error": outcome.as_ref().err(),
    })
}

#[cfg(feature = "ml")]
/// The [`MathLayer`] of a [`dump_json`] record (the regression fixtures).
pub fn layer_from_json(v: &serde_json::Value) -> Option<(MathLayer, [f32; 4])> {
    let f = |x: &serde_json::Value| x.as_f64().map(|v| v as f32);
    let bb = v["bbox"].as_array()?;
    let bbox = [f(&bb[0])?, f(&bb[1])?, f(&bb[2])?, f(&bb[3])?];
    let glyphs = v["glyphs"]
        .as_array()?
        .iter()
        .map(|g| {
            let g = g.as_array()?;
            let shown = g[0].as_str()?.chars().next()?;
            // A program glyph name is read again, so a fixture follows the
            // current name table.
            let name: Option<std::sync::Arc<str>> =
                g.get(8).and_then(|n| n.as_str()).map(std::sync::Arc::from);
            Some(
                MathGlyph {
                    ch: name.as_deref().map_or(shown, |n| read_name(n, shown)),
                    name,
                    l: f(&g[1])?,
                    t: f(&g[2])?,
                    r: f(&g[3])?,
                    b: f(&g[4])?,
                    base: f(&g[5])?,
                    size: f(&g[6])?,
                    tilted: g[7].as_bool()?,
                }
                .with_true_extent(),
            )
        })
        .collect::<Option<Vec<_>>>()?;
    let rules = v["rules"]
        .as_array()?
        .iter()
        .map(|r| {
            let r = r.as_array()?;
            Some(Rule {
                l: f(&r[0])?,
                t: f(&r[1])?,
                r: f(&r[2])?,
                b: f(&r[3])?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some((MathLayer { glyphs, rules }, bbox))
}

/// Whether formulas are rebuilt from the text layer before the VLM is asked
/// (`DOCLING_RS_FORMULA_TEXT_LAYER`, on unless set falsy).
pub fn enabled() -> bool {
    docling_core::env::nonempty("DOCLING_RS_FORMULA_TEXT_LAYER")
        .is_none_or(|_| docling_core::env::flag("DOCLING_RS_FORMULA_TEXT_LAYER"))
}

/// The formula in `bbox` (`[l, t, r, b]`, top-left page points) as LaTeX in
/// the CodeFormula model's spelling, or why it cannot be rebuilt with
/// confidence (see the module docs) — the caller then asks the model.
/// The glyphs of the formula in `bbox`: those centred inside it, plus a
/// tall (TeX extension) bracket hanging just outside its left or right edge
/// on its rows — the layout box is drawn to the ink of the formula's body
/// and routinely leaves out the `\left(` before it or the `\right)` after.
fn select(layer: &MathLayer, bbox: [f32; 4]) -> Vec<MathGlyph> {
    const SLACK: f32 = 1.0;
    let [l, t, r, b] = bbox;
    let inside =
        |x: f32, y: f32| x >= l - SLACK && x <= r + SLACK && y >= t - SLACK && y <= b + SLACK;
    layer
        .glyphs
        .iter()
        .filter(|g| {
            let (xc, yc) = ((g.l + g.r) / 2.0, (g.t + g.b) / 2.0);
            inside(xc, yc)
                || (g.hangs()
                    && delimiter(g.ch)
                    && yc >= t
                    && yc <= b
                    && ((g.r <= l + SLACK && l - g.r < 0.6 * g.size)
                        || (g.l >= r - SLACK && g.l - r < 0.6 * g.size)))
        })
        .cloned()
        .collect()
}

/// The formulas one layout box holds, as one box per formula. The layout
/// model routinely draws a single box around two or three consecutive
/// display formulas; read as one formula that is several rows, which was
/// refused (and, measured on pdf-parse-bench, cost 113 formulas the model had
/// in fact found). The glyphs' own vertical extents — fraction bars and
/// hanging operators bridging the stacks of one formula — fall into bands
/// separated by display spacing; a gap of more than 0.6 em between bands
/// separates formulas, the much smaller gap between the lines of one
/// aligned display does not. Empty when the box holds one band (or no
/// glyphs): the box stands as it is.
pub fn split_rows(layer: &MathLayer, bbox: [f32; 4]) -> Vec<[f32; 4]> {
    let glyphs = select(layer, bbox);
    if glyphs.len() < 2 {
        return Vec::new();
    }
    let size = median(glyphs.iter().map(|g| g.size).collect());
    let [l, t, r, b] = bbox;
    let rules: Vec<Rule> = layer
        .rules
        .iter()
        .filter(|ru| {
            let (x, y) = ((ru.l + ru.r) / 2.0, (ru.t + ru.b) / 2.0);
            x >= l - 1.0 && x <= r + 1.0 && y >= t - 1.0 && y <= b + 1.0
        })
        .copied()
        .collect();
    // Vertical extents: glyph boxes and rules, as (top, bottom, glyph index).
    let mut spans: Vec<(f32, f32, Option<usize>)> = glyphs
        .iter()
        .enumerate()
        .map(|(i, g)| (g.t, g.b, Some(i)))
        .chain(rules.iter().map(|ru| (ru.t, ru.b, None)))
        .collect();
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    let gap = 0.6 * size;
    let mut bands: Vec<(f32, f32, Vec<usize>)> = Vec::new();
    for (top, bottom, gi) in spans {
        match bands.last_mut() {
            Some(band) if top <= band.1 + gap => {
                band.1 = band.1.max(bottom);
                band.2.extend(gi);
            }
            _ => bands.push((top, bottom, gi.into_iter().collect())),
        }
    }
    bands.retain(|band| !band.2.is_empty());
    // A band of script-size glyphs only (the limits of a `∑` whose box
    // hides its height, a lone subscript) is never a formula of its own:
    // it joins the band it is closest to.
    let full = glyphs.iter().map(|g| g.size).fold(0.0, f32::max);
    loop {
        let small = bands
            .iter()
            .position(|band| band.2.iter().all(|&i| glyphs[i].size < 0.85 * full));
        match small {
            Some(k) if bands.len() > 1 => {
                let gap_up = (k > 0).then(|| bands[k].0 - bands[k - 1].1);
                let gap_down = (k + 1 < bands.len()).then(|| bands[k + 1].0 - bands[k].1);
                let into = match (gap_up, gap_down) {
                    (Some(u), Some(d)) if d < u => k + 1,
                    (Some(_), _) => k - 1,
                    _ => k + 1,
                };
                let band = bands.remove(k);
                let target = if into > k { into - 1 } else { into };
                let t = &mut bands[target];
                t.0 = t.0.min(band.0);
                t.1 = t.1.max(band.1);
                t.2.extend(band.2);
            }
            _ => break,
        }
    }
    if bands.len() < 2 {
        return Vec::new();
    }
    bands
        .iter()
        .map(|(_, _, ids)| {
            let (mut bl, mut bt, mut br, mut bb) = (
                f32::INFINITY,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            );
            for &i in ids {
                let g = &glyphs[i];
                (bl, bt, br, bb) = (bl.min(g.l), bt.min(g.t), br.max(g.r), bb.max(g.b));
            }
            [bl, bt, br, bb]
        })
        .collect()
}

/// Whether a fraction bar lies just outside `bbox`, above or below the
/// glyphs and over them: the layout model drew the box through a fraction
/// (numerators and bars in one box, denominators in the next), and what is
/// inside reads as a formula it is only half of.
fn cut_through_a_fraction(layer: &MathLayer, bbox: [f32; 4], glyphs: &[MathGlyph]) -> bool {
    if glyphs.is_empty() {
        return false;
    }
    let [_, t, _, b] = bbox;
    let size = median(glyphs.iter().map(|g| g.size).collect());
    let (gl, gr) = glyphs
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, r), g| {
            (l.min(g.l), r.max(g.r))
        });
    layer.rules.iter().any(|ru| {
        let y = (ru.t + ru.b) / 2.0;
        let outside = y < t - 1.0 || y > b + 1.0;
        let near = (t - y).max(y - b) < 0.6 * size;
        let over = ru.r.min(gr) - ru.l.max(gl) > 0.5 * (ru.r - ru.l);
        outside && near && over
    })
}

/// Whether glyphs just outside `bbox` belong to the formula in it: on its
/// rows within half an em of its left or right edge (a `∑` or a `ρ` the box
/// left out), or under/over its glyphs within half an em of its bottom or
/// top (the limit of a `\max`). A display formula stands clear of its
/// neighbours by more than that; glyphs this close are the formula
/// continuing past a box drawn too tight.
fn continues_outside(layer: &MathLayer, bbox: [f32; 4], glyphs: &[MathGlyph]) -> bool {
    if glyphs.is_empty() {
        return false;
    }
    let [l, t, r, b] = bbox;
    let size = median(glyphs.iter().map(|g| g.size).collect());
    let (gl, gr) = glyphs
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, z), g| {
            (a.min(g.l), z.max(g.r))
        });
    let selected = |g: &MathGlyph| {
        glyphs
            .iter()
            .any(|s| s.l == g.l && s.t == g.t && s.ch == g.ch)
    };
    layer.glyphs.iter().any(|g| {
        if selected(g) || g.ch.is_whitespace() {
            return false;
        }
        let (xc, yc) = ((g.l + g.r) / 2.0, (g.t + g.b) / 2.0);
        let beside = yc > t
            && yc < b
            && ((g.r <= l && l - g.r < 0.5 * size) || (g.l >= r && g.l - r < 0.5 * size));
        // over or under: only script-size glyphs (a limit); the next line
        // of body text can sit this close under a display
        let stacked = g.size < 0.85 * size
            && xc > gl
            && xc < gr
            && ((g.b <= t && t - g.b < 0.5 * size) || (g.t >= b && g.t - b < 0.5 * size));
        beside || stacked
    })
}

/// Whether the tall brackets among `glyphs` pair up: a `\Big)` without a
/// `\Big(` means part of the formula is missing.
fn tall_brackets_pair(glyphs: &[MathGlyph]) -> bool {
    let side = |g: &MathGlyph| -> i32 {
        match g.name.as_deref() {
            Some(n) if g.hangs() && (n.contains("left") || n.starts_with("angbracketleft")) => 1,
            Some(n) if g.hangs() && n.contains("right") => -1,
            _ => 0,
        }
    };
    glyphs.iter().map(side).filter(|&s| s > 0).count()
        == glyphs.iter().map(side).filter(|&s| s < 0).count()
}

pub fn explain(layer: &MathLayer, bbox: [f32; 4]) -> Result<String, String> {
    let glyphs = select(layer, bbox);
    if cut_through_a_fraction(layer, bbox, &glyphs) {
        return Err("a fraction bar just outside the box (the formula is cut)".into());
    }
    if continues_outside(layer, bbox, &glyphs) {
        return Err("glyphs hugging the box from outside (the formula continues)".into());
    }
    if !tall_brackets_pair(&glyphs) {
        return Err("a tall bracket without its partner (part of the formula is missing)".into());
    }
    if glyphs.is_empty() {
        return Err("no glyphs in the region (drawn as outlines, or an image)".into());
    }
    if glyphs.len() > 600 {
        return Err(format!(
            "{} glyphs: too large to be one formula",
            glyphs.len()
        ));
    }
    if let Some(g) = glyphs.iter().find(|g| g.tilted) {
        return Err(format!("rotated glyph {:?}", g.ch));
    }
    if let Some(g) = glyphs
        .iter()
        .find(|g| !g.base.is_finite() || g.size.is_nan() || g.size <= 0.5)
    {
        return Err(format!("glyph {:?} without a baseline or size", g.ch));
    }
    if let Some(g) = glyphs.iter().find(|g| token(g.ch).is_none()) {
        return Err(format!("unmapped glyph {:?} (U+{:04X})", g.ch, g.ch as u32));
    }
    let [l, t, r, b] = bbox;
    let rules: Vec<Rule> = layer
        .rules
        .iter()
        .filter(|ru| {
            let (x, y) = ((ru.l + ru.r) / 2.0, (ru.t + ru.b) / 2.0);
            x >= l - 1.0 && x <= r + 1.0 && y >= t - 1.0 && y <= b + 1.0
        })
        .copied()
        .collect();
    let tex = Builder::new(glyphs, rules).run()?;
    // Three centred dots set one by one are `\cdots` (TeX draws `\cdots` as
    // three `\cdot` glyphs).
    let tex = tex
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace("\\cdot \\cdot \\cdot", "\\cdots");
    if tex.is_empty() {
        return Err("nothing readable".into());
    }
    Ok(tex)
}

/// A unit of the formula under construction: a glyph, or a structure already
/// rebuilt (fraction, root, accented glyph, operator with limits).
#[derive(Debug, Clone)]
struct Atom {
    l: f32,
    t: f32,
    r: f32,
    b: f32,
    base: f32,
    size: f32,
    tex: String,
    /// The glyph, for a plain glyph atom.
    ch: Option<char>,
    /// A rebuilt structure: always on its row's baseline, never a script by
    /// position (a fraction's centre is not a baseline).
    structure: bool,
}

impl Atom {
    fn glyph(g: &MathGlyph) -> Self {
        // A hanging extension glyph is centred on the math axis, a quarter
        // em above its row's baseline: that is its row, not its own
        // baseline (see `MathGlyph::with_true_extent`).
        let base = if g.hangs() {
            (g.t + g.b) / 2.0 + 0.25 * g.size
        } else {
            g.base
        };
        Atom {
            l: g.l,
            t: g.t,
            r: g.r,
            b: g.b,
            base,
            size: g.size,
            tex: token(g.ch).unwrap_or_default(),
            ch: Some(g.ch),
            structure: false,
        }
    }
    fn xc(&self) -> f32 {
        (self.l + self.r) / 2.0
    }
    fn yc(&self) -> f32 {
        (self.t + self.b) / 2.0
    }
}

fn union(atoms: &[Atom]) -> (f32, f32, f32, f32) {
    atoms.iter().fold(
        (
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ),
        |(l, t, r, b), a| (l.min(a.l), t.min(a.t), r.max(a.r), b.max(a.b)),
    )
}

fn median(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

struct Builder {
    atoms: Vec<Atom>,
    rules: Vec<Rule>,
}

impl Builder {
    fn new(glyphs: Vec<MathGlyph>, rules: Vec<Rule>) -> Self {
        Self {
            atoms: glyphs.iter().map(Atom::glyph).collect(),
            rules,
        }
    }

    fn run(mut self) -> Result<String, String> {
        self.joins()?;
        self.accents();
        // Narrowest rule first: an inner fraction or root is always narrower
        // than the one around it, so by the time a rule is read every
        // structure inside its stacks is already a single atom.
        let mut rules = std::mem::take(&mut self.rules);
        rules.sort_by(|a, b| (a.r - a.l).total_cmp(&(b.r - b.l)));
        for k in 0..rules.len() {
            let rule = rules[k];
            let wider: Vec<Rule> = rules[k + 1..].to_vec();
            match self.rule(rule, &wider) {
                Some(true) => {}
                Some(false) => {
                    return Err(format!(
                        "a rule nothing explains at ({:.0}, {:.0})–({:.0}, {:.0})",
                        rule.l, rule.t, rule.r, rule.b
                    ))
                }
                None => return Err("a fraction or root whose parts do not read".into()),
            }
        }
        self.word_limits();
        self.limits();
        let tex = rows(std::mem::take(&mut self.atoms))
            .ok_or_else(|| "stacked or unreadable structure (a matrix, a binomial)".to_string())?;
        // Several rows are a multi-line display only when every row is an
        // equation (has a relation); rows of bare entries are a matrix, an
        // array or `cases`, whose columns this reader does not rebuild —
        // read as rows they come out wrong, so they go to the model.
        if tex.contains(" \\\\ ") && !tex.split(" \\\\ ").all(has_relation) {
            return Err("rows that are not all equations (a matrix, an array, cases)".into());
        }
        // Even rows that are all equations are, measured on pdf-parse-bench,
        // more often a neighbouring formula the layout box took in than one
        // multi-line display (27 wrong to 6 right): until formulas are
        // delimited from the glyphs themselves, several rows go to the model.
        if tex.contains(" \\\\ ") {
            return Err("several rows (one display, or a neighbour in the box)".into());
        }
        incomplete(&tex)?;
        Ok(tex)
    }

    /// Glyphs TeX overlays or abuts to draw one symbol, joined into it:
    /// `\mapsto` (a zero-width bar + `→`), `\longrightarrow` (`−` + `→`),
    /// `\Longleftrightarrow` (`⇐` + `⇒`), and a negation slash over a
    /// relation (`≠`, `∉`). A negation slash over anything else is
    /// unexplained.
    fn joins(&mut self) -> Result<(), String> {
        const NEGATION: char = '\u{0338}';
        loop {
            let mut order: Vec<usize> = (0..self.atoms.len()).collect();
            order.sort_by(|&x, &y| self.atoms[x].l.total_cmp(&self.atoms[y].l));
            let mut hit = None;
            for w in order.windows(2) {
                let (a, b) = (&self.atoms[w[0]], &self.atoms[w[1]]);
                let (Some(ca), Some(cb)) = (a.ch, b.ch) else {
                    continue;
                };
                let touching = b.l <= a.r + 0.12 * a.size.max(b.size)
                    && (a.base - b.base).abs() < 0.15 * a.size.max(b.size);
                if !touching {
                    continue;
                }
                let joined = match (ca, cb) {
                    ('↦', '→') | ('↦', '⟶') => Some('↦'),
                    ('−', '→') | ('-', '→') => Some('⟶'),
                    ('←', '−') => Some('⟵'),
                    ('=', '⇒') => Some('⟹'),
                    ('⇐', '=') => Some('⟸'),
                    ('⇐', '⇒') => Some('⟺'),
                    ('←', '→') => Some('⟷'),
                    ('↦', '−') => Some('↦'),
                    ('|', '=') => Some('⊨'),
                    _ => None,
                };
                if let Some(c) = joined {
                    hit = Some((w[0], w[1], c));
                    break;
                }
            }
            // Symbols TeX stacks: `\cong` is `∼` set over `=`.
            if hit.is_none() {
                for (i, a) in self.atoms.iter().enumerate() {
                    if a.ch != Some('∼') {
                        continue;
                    }
                    let under = self.atoms.iter().position(|g| {
                        g.ch == Some('=')
                            && g.l.max(a.l) < g.r.min(a.r)
                            && (g.r.min(a.r) - g.l.max(a.l)) > 0.6 * (g.r - g.l).min(a.r - a.l)
                            && a.base < g.base
                            && g.base - a.base < 0.6 * g.size
                    });
                    if let Some(j) = under {
                        hit = Some((i, j, '≅'));
                        break;
                    }
                }
            }
            // A negation slash overlapping a relation: the negated relation.
            if hit.is_none() {
                if let Some(si) = self.atoms.iter().position(|a| a.ch == Some(NEGATION)) {
                    let s = self.atoms[si].clone();
                    let under = (0..self.atoms.len()).find(|&j| {
                        let g = &self.atoms[j];
                        j != si && g.l < s.r && s.l < g.r && (g.base - s.base).abs() < 0.3 * s.size
                    });
                    let negated = under.and_then(|j| {
                        Some(match self.atoms[j].ch? {
                            '=' => '≠',
                            '∈' => '∉',
                            '≡' => '≢',
                            '<' => '≮',
                            '>' => '≯',
                            '⊂' => '⊄',
                            '⊆' => '⊈',
                            '∼' => '≁',
                            '≤' => '≰',
                            '≥' => '≱',
                            _ => return None,
                        })
                    });
                    match (under, negated) {
                        (Some(j), Some(c)) => hit = Some((si, j, c)),
                        _ => return Err("a negation slash over no relation".into()),
                    }
                }
            }
            let Some((i, j, c)) = hit else { return Ok(()) };
            let (a, b) = (self.atoms[i].clone(), self.atoms[j].clone());
            let (hi, lo) = (i.max(j), i.min(j));
            self.atoms.remove(hi);
            self.atoms.remove(lo);
            // The joined symbol sits on the row of the glyph that carries
            // it: the relation under a negation slash or under the `∼` of `≅`.
            let host = if a.ch == Some(NEGATION) || c == '≅' {
                &b
            } else {
                &a
            };
            self.atoms.push(Atom {
                l: a.l.min(b.l),
                t: a.t.min(b.t),
                r: a.r.max(b.r),
                b: a.b.max(b.b),
                base: host.base,
                size: host.size.max(b.size),
                tex: token(c).ok_or_else(|| format!("no reading for joined {c:?}"))?,
                ch: Some(c),
                structure: false,
            });
        }
    }

    /// An accent glyph sitting on a glyph below it becomes `\hat { x }`.
    fn accents(&mut self) {
        loop {
            let found = self.atoms.iter().enumerate().find_map(|(i, a)| {
                let cmd = accent(a.ch?)?;
                let below = self
                    .atoms
                    .iter()
                    .enumerate()
                    .filter(|(j, g)| {
                        *j != i
                            && g.ch.is_some_and(|c| accent(c).is_none())
                            && a.xc() >= g.l - 0.15 * g.size
                            && a.xc() <= g.r + 0.15 * g.size
                            // Above the glyph: TeX raises an accent only by
                            // the base's height over the x-height (nothing
                            // for `x`, a quarter em for `W`), so the test is
                            // the accent's middle against the glyph's, not
                            // a fixed lift.
                            // (the box can mislead: a `\vec` arrow mapped
                            // to U+20D7 keeps a box below the letter's
                            // middle — its raised baseline still shows it)
                            && (a.yc() < g.yc() || a.base < g.base - 0.1 * g.size)
                            && a.base <= g.base + 0.05 * g.size
                            && a.base > g.base - 1.6 * g.size
                    })
                    // The full-size letter under it before a script beside
                    // that letter (`\bar{\xi_1}`'s bar is over the ξ, not
                    // the 1), then the nearest.
                    .min_by(|x, y| {
                        y.1.size.total_cmp(&x.1.size).then(
                            (x.1.xc() - a.xc())
                                .abs()
                                .total_cmp(&(y.1.xc() - a.xc()).abs()),
                        )
                    })?
                    .0;
                Some((i, below, cmd))
            });
            let Some((i, j, cmd)) = found else { break };
            let base = self.atoms[j].clone();
            let acc = self.atoms[i].clone();
            let merged = Atom {
                l: base.l.min(acc.l),
                t: base.t.min(acc.t),
                r: base.r.max(acc.r),
                b: base.b,
                base: base.base,
                size: base.size,
                tex: format!("{cmd} {{ {} }}", base.tex),
                // Still its letter for the row's baseline (an accented
                // letter is a full-size glyph on the line, not a structure).
                ch: base.ch,
                structure: false,
            };
            let (hi, lo) = (i.max(j), i.min(j));
            self.atoms.remove(hi);
            self.atoms.remove(lo);
            self.atoms.push(merged);
        }
    }

    /// Read one rule: a radical's vinculum, a fraction bar, an overline or an
    /// underline. `Some(false)` when it is none of them; `None` when a part
    /// cannot be rebuilt.
    fn rule(&mut self, rule: Rule, wider: &[Rule]) -> Option<bool> {
        let ry = (rule.t + rule.b) / 2.0;
        // Radical: a √ glyph ending where the rule starts, its top at the
        // rule's height.
        let radical = self.atoms.iter().position(|a| {
            a.ch == Some('√')
                && rule.l >= a.l - 1.0
                && rule.l <= a.r + 0.5 * a.size
                && ry >= a.t - 0.8 * a.size
                && ry <= a.base
        });
        if let Some(ri) = radical {
            let sign = self.atoms[ri].clone();
            // The body: under the vinculum, within its span, down to the
            // sign's foot — a subscript in the body reaches below a text
            // radical's nominal box — but never past a wider rule below the
            // root (the bar of a fraction the root sits in).
            // (A text-size `√` hangs from its baseline like the extension
            // ones but keeps the generic box: measure from the vinculum.)
            let floor = sign
                .b
                .max(sign.base + 0.5 * sign.size)
                .max(ry + 1.4 * sign.size);
            let walled = |a: &Atom| {
                wider.iter().any(|w| {
                    let wy = (w.t + w.b) / 2.0;
                    a.xc() >= w.l && a.xc() <= w.r && wy > ry && a.yc() > wy
                })
            };
            let body = self.take(
                |a| {
                    a.xc() > sign.r - 0.5
                        && a.xc() < rule.r + 0.5
                        && a.yc() > ry
                        && a.t < floor
                        && !walled(a)
                },
                ri,
            );
            if body.is_empty() {
                return Some(false);
            }
            // The index (`\sqrt[3]{…}`): small, up and to the left inside
            // the sign's crook — raised against the body's baseline (a
            // radical's own baseline is its top: it hangs).
            let row = median(body.iter().map(|a| a.base).collect());
            let index = self.take_where(|a| {
                // inside the crook: from the sign's left edge (an exponent
                // of whatever precedes the root ends left of it), ending
                // before the sign's middle
                a.l >= sign.l - 0.15 * sign.size
                    && a.xc() < sign.xc()
                    && a.size < 0.85 * sign.size
                    && (a.base < row - 0.25 * sign.size || a.base < sign.base - 0.25 * sign.size)
            });
            let ri = self
                .atoms
                .iter()
                .position(|a| a.ch == Some('√') && a.l == sign.l && a.t == sign.t)?;
            self.atoms.remove(ri);
            let inner = rows(body.clone())?;
            let tex = match index.is_empty() {
                true => format!("\\sqrt {{ {inner} }}"),
                false => format!("\\sqrt [ {} ] {{ {inner} }}", rows(index)?),
            };
            let (_, t, r, b) = union(&body);
            // The root sits on its body's baseline: a large radical sign
            // hangs from its own, which reads an em too high and made the
            // whole root a superscript (`\frac{1}{60} ^ { \sqrt{5} }`).
            let body_base = {
                let main = main_size(&body);
                let on_line: Vec<f32> = body
                    .iter()
                    .filter(|a| a.size >= 0.85 * main)
                    .map(|a| a.base)
                    .collect();
                if on_line.is_empty() {
                    sign.base
                } else {
                    median(on_line)
                }
            };
            self.atoms.push(Atom {
                l: sign.l,
                t: t.min(rule.t),
                r: r.max(rule.r),
                b: b.max(sign.b),
                base: body_base,
                size: main_size(&body).max(sign.size * 0.85),
                tex,
                ch: None,
                structure: true,
            });
            return Some(true);
        }
        let in_span = |a: &Atom| a.xc() >= rule.l - 0.5 && a.xc() <= rule.r + 0.5;
        // A wider rule between this one and an atom fences the atom off: it
        // belongs to the structure that rule makes.
        let fenced = |a: &Atom| {
            wider.iter().any(|w| {
                let wy = (w.t + w.b) / 2.0;
                a.xc() >= w.l
                    && a.xc() <= w.r
                    && ((wy < ry && a.yc() < wy) || (wy > ry && a.yc() > wy))
            })
        };
        let above = self.band(&rule, true, &in_span, &fenced);
        let below = self.band(&rule, false, &in_span, &fenced);
        let (num, den) = (!above.is_empty(), !below.is_empty());
        // A rule with glyphs only above it reads as an underline, which in
        // display math is almost never what it is: measured, it was the bar
        // of a fraction whose numerator carries an overline
        // (`\frac{\overline{AM}}{\overline{MH}}`), read wrong. Unexplained.
        if !den {
            return Some(false);
        }
        let take_ids = |ids: &[usize], atoms: &mut Vec<Atom>| -> Vec<Atom> {
            let mut ids = ids.to_vec();
            ids.sort_unstable_by(|a, b| b.cmp(a));
            let mut out: Vec<Atom> = ids.into_iter().map(|i| atoms.remove(i)).collect();
            out.reverse();
            out
        };
        let mut all: Vec<usize> = above.iter().chain(&below).copied().collect();
        all.sort_unstable();
        let parts = take_ids(&all, &mut self.atoms);
        let (top, bottom): (Vec<Atom>, Vec<Atom>) = parts.into_iter().partition(|a| a.yc() < ry);
        let size = median(top.iter().chain(&bottom).map(|a| a.size).collect());
        let run_base = |v: &[Atom]| median(v.iter().map(|a| a.base).collect());
        let (top_base, bottom_base) = (run_base(&top), run_base(&bottom));
        let (ul, ut, ur, ub) = union(&top.iter().chain(&bottom).cloned().collect::<Vec<_>>());
        let tex = match (num, den) {
            (true, true) => format!("\\frac {{ {} }} {{ {} }}", rows(top)?, rows(bottom)?),
            (false, true) => format!("\\overline {{ {} }}", rows(bottom.clone())?),
            (_, false) => unreachable!("refused above"),
        };
        // A fraction sits on the math axis, about a quarter em above the
        // baseline; an overlined run keeps its own baseline.
        let base = match (num, den) {
            (true, true) => ry + 0.25 * size,
            _ => bottom_base,
        };
        let _ = top_base;
        let base = if base.is_finite() {
            base
        } else {
            ry + 0.25 * size
        };
        self.atoms.push(Atom {
            l: ul.min(rule.l),
            t: ut.min(rule.t),
            r: ur.max(rule.r),
            b: ub.max(rule.b),
            base,
            size,
            tex,
            ch: None,
            structure: true,
        });
        Some(true)
    }

    /// The atoms stacked above (or below) a rule within its span: the
    /// nearest first, each next one taken while it nearly touches the stack.
    fn band(
        &self,
        rule: &Rule,
        up: bool,
        in_span: &dyn Fn(&Atom) -> bool,
        fenced: &dyn Fn(&Atom) -> bool,
    ) -> Vec<usize> {
        let ry = (rule.t + rule.b) / 2.0;
        let mut cand: Vec<usize> = (0..self.atoms.len())
            .filter(|&i| {
                let a = &self.atoms[i];
                in_span(a)
                    && !fenced(a)
                    && if up {
                        a.yc() < ry && a.base <= ry + 0.5
                    } else {
                        a.yc() > ry
                    }
            })
            .collect();
        cand.sort_by(|&x, &y| {
            let d = |a: &Atom| if up { ry - a.b } else { a.t - ry };
            d(&self.atoms[x]).total_cmp(&d(&self.atoms[y]))
        });
        let mut edge = if up { rule.t } else { rule.b };
        let mut out = Vec::new();
        for i in cand {
            let a = &self.atoms[i];
            let gap = if up { edge - a.b } else { a.t - edge };
            if gap > 0.6 * a.size {
                break;
            }
            edge = if up { edge.min(a.t) } else { edge.max(a.b) };
            out.push(i);
        }
        out
    }

    /// Remove and return the atoms `keep` selects, never the one at `skip`.
    fn take(&mut self, keep: impl Fn(&Atom) -> bool, skip: usize) -> Vec<Atom> {
        let mut out = Vec::new();
        let mut k = 0;
        self.atoms.retain(|a| {
            let idx = k;
            k += 1;
            let taken = idx != skip && keep(a);
            if taken {
                out.push(a.clone());
            }
            !taken
        });
        out
    }

    fn take_where(&mut self, keep: impl Fn(&Atom) -> bool) -> Vec<Atom> {
        self.take(keep, usize::MAX)
    }

    /// A word operator — `lim`, `max`, `liminf`, … — with smaller atoms
    /// centred under it: `\lim _ { n \rightarrow \infty }`. Without this the
    /// limit's glyphs, lower than the word, read as subscripts of its
    /// letters (`l _ { \rightarrow } i m`).
    fn word_limits(&mut self) {
        const WORDS: [&str; 7] = ["liminf", "limsup", "lim", "max", "min", "sup", "inf"];
        loop {
            let mut letters: Vec<usize> = (0..self.atoms.len())
                .filter(|&i| {
                    let a = &self.atoms[i];
                    !a.structure && a.ch.is_some_and(|c| c.is_ascii_lowercase())
                })
                .collect();
            letters.sort_by(|&x, &y| self.atoms[x].l.total_cmp(&self.atoms[y].l));
            let mut found = None;
            'scan: for start in 0..letters.len() {
                // The word's letters: those on this letter's baseline, from
                // it rightwards — the limit's own letters below (`s → 1`)
                // fall between them in x and must not interleave.
                let head = &self.atoms[letters[start]];
                let line_ids: Vec<usize> = letters[start..]
                    .iter()
                    .copied()
                    .filter(|&i| (self.atoms[i].base - head.base).abs() < 0.1 * head.size)
                    .collect();
                for word in WORDS {
                    let len = word.len();
                    if len > line_ids.len() {
                        continue;
                    }
                    let run: Vec<&Atom> = line_ids[..len].iter().map(|&i| &self.atoms[i]).collect();
                    let first = run[0];
                    let spelled: String = run.iter().filter_map(|a| a.ch).collect();
                    let tight = run.windows(2).all(|w| {
                        (w[1].base - first.base).abs() < 0.1 * first.size
                            && w[1].l - w[0].r < 0.3 * first.size
                    });
                    if spelled != word || !tight {
                        continue;
                    }
                    let (l, r) = (first.l, run[len - 1].r);
                    let under: Vec<usize> = (0..self.atoms.len())
                        .filter(|&i| {
                            let a = &self.atoms[i];
                            !line_ids[..len].contains(&i)
                                && a.xc() >= l - 0.3 * first.size
                                && a.xc() <= r + 0.3 * first.size
                                && a.base > first.base + 0.35 * first.size
                                && a.base < first.base + 2.0 * first.size
                                && a.size < 0.9 * first.size
                        })
                        .collect();
                    if !under.is_empty() {
                        found = Some((line_ids[..len].to_vec(), under, word));
                        break 'scan;
                    }
                }
            }
            let Some((word_ids, under, word)) = found else {
                return;
            };
            let first = self.atoms[word_ids[0]].clone();
            let mut ids: Vec<usize> = word_ids.iter().chain(&under).copied().collect();
            ids.sort_unstable_by(|a, b| b.cmp(a));
            let taken: Vec<(usize, Atom)> =
                ids.iter().map(|&i| (i, self.atoms.remove(i))).collect();
            let below: Vec<Atom> = taken
                .iter()
                .filter(|(i, _)| under.contains(i))
                .map(|(_, a)| a.clone())
                .collect();
            let all: Vec<Atom> = taken.iter().map(|(_, a)| a.clone()).collect();
            let Some(limit) = rows(below) else {
                // Put everything back; the gate refuses the region later.
                self.atoms.extend(all);
                return;
            };
            let (l, t, r, b) = union(&all);
            self.atoms.push(Atom {
                l,
                t,
                r,
                b,
                base: first.base,
                size: first.size,
                tex: format!("\\{word} _ {{ {limit} }}"),
                ch: None,
                structure: true,
            });
        }
    }

    /// A large operator (`∑`, `∏`, `∫`, …) with smaller atoms centred above
    /// and below it: `\sum _ { below } ^ { above }`.
    fn limits(&mut self) {
        let ops: Vec<usize> = (0..self.atoms.len())
            .filter(|&i| self.atoms[i].ch.is_some_and(big_operator))
            .collect();
        for &oi in ops.iter().rev() {
            let op = self.atoms[oi].clone();
            let w = op.r - op.l;
            let near = |a: &Atom| {
                a.xc() >= op.l - 0.2 * w && a.xc() <= op.r + 0.2 * w && a.size < 0.9 * op.size
            };
            let below_ids: Vec<usize> = (0..self.atoms.len())
                .filter(|&i| {
                    i != oi && near(&self.atoms[i]) && self.atoms[i].base > op.base + 0.35 * op.size
                })
                .collect();
            let above_ids: Vec<usize> = (0..self.atoms.len())
                .filter(|&i| {
                    i != oi && near(&self.atoms[i]) && self.atoms[i].base < op.base - 0.8 * op.size
                })
                .collect();
            if below_ids.is_empty() && above_ids.is_empty() {
                continue;
            }
            // A limit wider than the operator (`(g-3)/2` over a `∑`) is one
            // run: grow each from what sits over the operator to the glyphs
            // on its baseline that touch it, so its ends are not left as
            // scripts of the neighbours.
            let grow = |seed: Vec<usize>, atoms: &[Atom]| -> Vec<usize> {
                let mut set = seed;
                loop {
                    let next = (0..atoms.len()).find(|&j| {
                        j != oi
                            && !set.contains(&j)
                            && atoms[j].size < 0.9 * op.size
                            && set.iter().any(|&m| {
                                let (a, b) = (&atoms[m], &atoms[j]);
                                (a.base - b.base).abs() < 0.15 * a.size
                                    && b.l - a.r < 0.4 * a.size
                                    && a.l - b.r < 0.4 * a.size
                            })
                    });
                    match next {
                        Some(j) => set.push(j),
                        None => return set,
                    }
                }
            };
            let below_ids = grow(below_ids, &self.atoms);
            let above_ids = grow(above_ids, &self.atoms);
            if below_ids.iter().any(|i| above_ids.contains(i)) {
                continue;
            }
            let mut ids: Vec<usize> = below_ids
                .iter()
                .chain(&above_ids)
                .copied()
                .chain([oi])
                .collect();
            ids.sort_unstable_by(|a, b| b.cmp(a));
            let mut taken: Vec<(usize, Atom)> =
                ids.iter().map(|&i| (i, self.atoms.remove(i))).collect();
            taken.reverse();
            let pick = |set: &[usize]| -> Vec<Atom> {
                taken
                    .iter()
                    .filter(|(i, _)| set.contains(i))
                    .map(|(_, a)| a.clone())
                    .collect()
            };
            let (below, above) = (pick(&below_ids), pick(&above_ids));
            let mut tex = op.tex.clone();
            if !below.is_empty() {
                match rows(below.clone()) {
                    Some(t) => tex.push_str(&format!(" _ {{ {t} }}")),
                    None => {
                        // Put everything back untouched; the gate rejects
                        // the stack later.
                        self.atoms.extend(taken.into_iter().map(|(_, a)| a));
                        return;
                    }
                }
            }
            if !above.is_empty() {
                match rows(above.clone()) {
                    Some(t) => tex.push_str(&format!(" ^ {{ {t} }}")),
                    None => {
                        self.atoms.extend(taken.into_iter().map(|(_, a)| a));
                        return;
                    }
                }
            }
            let all: Vec<Atom> = below
                .iter()
                .chain(&above)
                .cloned()
                .chain([op.clone()])
                .collect();
            let (l, t, r, b) = union(&all);
            self.atoms.push(Atom {
                l,
                t,
                r,
                b,
                base: op.base,
                size: op.size,
                tex,
                ch: None,
                structure: true,
            });
        }
    }
}

/// Split atoms into rows (a multi-line display: one baseline per equation)
/// and lay each out; rows join with `\\`, the way the caller already joins
/// a formula block decoded line by line.
fn rows(atoms: Vec<Atom>) -> Option<String> {
    if atoms.is_empty() {
        return Some(String::new());
    }
    let main = main_size(&atoms);
    // Full-size, non-structure baselines, clustered.
    let mut bases: Vec<f32> = atoms
        .iter()
        .filter(|a| a.size >= 0.85 * main || a.structure)
        .map(|a| a.base)
        .collect();
    bases.sort_by(f32::total_cmp);
    let mut rows_at: Vec<f32> = Vec::new();
    for b in bases {
        match rows_at.last() {
            Some(&last) if b - last <= 1.0 * main => {}
            _ => rows_at.push(b),
        }
    }
    if rows_at.len() <= 1 {
        return line(atoms);
    }
    let mut groups: Vec<Vec<Atom>> = vec![Vec::new(); rows_at.len()];
    for a in atoms {
        let k = (0..rows_at.len())
            .min_by(|&x, &y| {
                (rows_at[x] - a.base)
                    .abs()
                    .total_cmp(&(rows_at[y] - a.base).abs())
            })
            .unwrap_or(0);
        groups[k].push(a);
    }
    let parts: Option<Vec<String>> = groups
        .into_iter()
        .filter(|g| !g.is_empty())
        .map(line)
        .collect();
    Some(parts?.join(" \\\\ "))
}

/// The shapes a formula with a missing piece leaves — measured on
/// pdf-parse-bench, where the layout box cut a glyph off a formula's edge:
/// a script with nothing to attach to (`{ } _ { 0 } = …`, the `μ` of `μ_0`
/// lost) and brackets that do not pair (a `\{ … \}` whose outer braces
/// fell outside). Each is refused: the model reads the region from its
/// pixels instead. (Tried and dropped: refusing a formula that opens on a
/// relation — continuation lines `= \lim …` do that honestly more often than
/// a cut-off does — and unpaired angle brackets, which bra-ket `|ψ⟩` has by
/// design.)
fn incomplete(tex: &str) -> Result<(), String> {
    let toks: Vec<&str> = tex.split_whitespace().collect();
    if tex.contains("{ } _") || tex.contains("{ } ^") {
        return Err("a script with no base (a glyph is missing)".into());
    }
    let count = |t: &str| toks.iter().filter(|&&x| x == t).count();
    for (open, close) in [("(", ")"), ("[", "]"), ("\\{", "\\}")] {
        if count(open) != count(close) {
            // A half-open interval (`[0, 1)`) is the one honest exception;
            // it is rare enough in display math to leave to the model.
            return Err(format!(
                "unpaired {open} {close} (part of the formula is missing)"
            ));
        }
    }
    Ok(())
}

/// Whether a row of output holds a relation — what makes it an equation
/// rather than a row of a matrix.
fn has_relation(row: &str) -> bool {
    const RELATIONS: [&str; 30] = [
        "=",
        "<",
        ">",
        "\\leq",
        "\\geq",
        "\\neq",
        "\\approx",
        "\\equiv",
        "\\sim",
        "\\simeq",
        "\\cong",
        "\\propto",
        "\\rightarrow",
        "\\leftarrow",
        "\\Rightarrow",
        "\\Leftarrow",
        "\\Leftrightarrow",
        "\\leftrightarrow",
        "\\longrightarrow",
        "\\Longrightarrow",
        "\\Longleftrightarrow",
        "\\mapsto",
        "\\in",
        "\\notin",
        "\\subset",
        "\\subseteq",
        "\\supset",
        "\\supseteq",
        "\\ll",
        "\\gg",
    ];
    row.split_whitespace().any(|t| RELATIONS.contains(&t))
}

fn main_size(atoms: &[Atom]) -> f32 {
    let sizes: Vec<f32> = atoms
        .iter()
        .filter(|a| a.ch.is_some_and(|c| !delimiter(c) && !big_operator(c)) || a.structure)
        .map(|a| a.size)
        .collect();
    let pool = if sizes.is_empty() {
        atoms.iter().map(|a| a.size).collect()
    } else {
        sizes
    };
    pool.into_iter().fold(0.0, f32::max)
}

/// One row: base-level atoms in x order, each followed by the scripts that
/// sit raised or lowered after it (recursively — a script's own scripts are
/// read inside its group).
fn line(mut atoms: Vec<Atom>) -> Option<String> {
    if atoms.is_empty() {
        return Some(String::new());
    }
    atoms.sort_by(|a, b| a.l.total_cmp(&b.l));
    let main = main_size(&atoms);
    let full: Vec<f32> = atoms
        .iter()
        .filter(|a| {
            !a.structure
                && a.size >= 0.85 * main
                && a.ch.is_some_and(|c| !delimiter(c) && !big_operator(c))
        })
        .map(|a| a.base)
        .collect();
    let base = if full.is_empty() {
        median(atoms.iter().map(|a| a.base).collect())
    } else {
        median(full)
    };
    #[derive(PartialEq, Clone, Copy)]
    enum Level {
        Base,
        Sup,
        Sub,
    }
    let level = |a: &Atom| -> Level {
        let big = a.size >= 0.85 * main;
        // A rebuilt structure's size says nothing about its level: a `\tfrac`
        // on the baseline is set in script size. Its estimated baseline does
        // (the fraction's axis put back a quarter em): only one raised or
        // lowered well clear of the row — a fraction in an exponent — is a
        // script.
        // (A root or limit in script size, though, is a script by its
        // baseline like any glyph: `\beta _ { \sqrt { s } }`.)
        if a.structure && (big || a.tex.starts_with("\\frac")) {
            let shift = a.base - base;
            return if shift < -0.35 * main {
                Level::Sup
            } else if shift > 0.35 * main {
                Level::Sub
            } else {
                Level::Base
            };
        }
        if let Some(c) = a.ch {
            if big && (delimiter(c) || big_operator(c)) {
                return Level::Base;
            }
            // a radical sign without a vinculum (`\surd 6`) hangs from its
            // own baseline like the extension glyphs; it is on the line
            if c == '√' && a.size >= 0.85 * main {
                return Level::Base;
            }
        }
        let shift = a.base - base; // + is lower
        if shift < -0.12 * main || (!big && shift < -0.05 * main) {
            Level::Sup
        } else if shift > 0.1 * main || (!big && shift > 0.05 * main) {
            Level::Sub
        } else {
            Level::Base
        }
    };
    struct Item {
        tex: String,
        r: f32,
        size: f32,
        sub: Vec<Atom>,
        sup: Vec<Atom>,
        ch: Option<char>,
        /// The script group the last script atom joined, and its right edge.
        last: Option<(Level, f32)>,
    }
    // A script-level atom standing over a base atom of the row (an arrow
    // over `=`, `\overset{→}{=}`) is stacked notation, not a script.
    let bases: Vec<(f32, f32)> = atoms
        .iter()
        .filter(|a| level(a) == Level::Base && has_relation(&a.tex))
        .map(|a| (a.l, a.r))
        .collect();
    if atoms.iter().any(|a| {
        level(a) != Level::Base
            && bases
                .iter()
                .any(|&(l, r)| a.xc() > l + 0.2 * (r - l) && a.xc() < r - 0.2 * (r - l))
    }) {
        return None;
    }
    let mut items: Vec<Item> = Vec::new();
    for a in atoms {
        let mut lv = level(&a);
        // A script-size atom back near the baseline right after a script is
        // that script's own script (`Δ _ { S ^ { n - 1 } }`: the
        // superscript of a subscript sits about where the base line is).
        if lv == Level::Base && a.size < 0.85 * main && !a.structure {
            if let Some(Item {
                last: Some((kind, r)),
                ..
            }) = items.last()
            {
                if a.l - r < 0.3 * a.size {
                    lv = *kind;
                }
            }
        }
        match lv {
            Level::Base => {
                if let Some(last) = items.last() {
                    // Two full-size atoms stacked on one another — a matrix
                    // column, an `\overset` — is structure this reader does
                    // not rebuild.
                    let overlap = last.r - a.l;
                    if overlap > 0.5 * (a.r - a.l).min(last.size)
                        && last.sub.is_empty()
                        && last.sup.is_empty()
                    {
                        return None;
                    }
                }
                items.push(Item {
                    tex: a.tex.clone(),
                    r: a.r,
                    size: a.size,
                    sub: Vec::new(),
                    sup: Vec::new(),
                    ch: a.ch,
                    last: None,
                });
            }
            lv => {
                if items.is_empty() {
                    items.push(Item {
                        tex: "{ }".into(),
                        r: a.l,
                        size: a.size,
                        sub: Vec::new(),
                        sup: Vec::new(),
                        ch: None,
                        last: None,
                    });
                }
                let it = items.last_mut().unwrap();
                // Two stacked atoms right after an opening bracket are a
                // binomial (`\binom{n}{k}`), not scripts of the bracket; and
                // a relation carries no scripts (`Z = _ { n } \sum …` is a
                // limit the operator lost).
                let opening = matches!(it.ch, Some('(' | '[' | '{' | '⟨'));
                let other = if lv == Level::Sup { &it.sub } else { &it.sup };
                if opening && !other.is_empty() {
                    return None;
                }
                // A script standing over its base rather than after it is
                // stacked notation — `\overset`, `\binom`, `\underset` —
                // which reads wrong as a script.
                if it.ch.is_some() && a.xc() < it.r - 0.15 * it.size && it.last.is_none() {
                    return None;
                }
                it.last = Some((lv, it.last.map_or(a.r, |(_, r)| r.max(a.r))));
                if lv == Level::Sup {
                    it.sup.push(a);
                } else {
                    it.sub.push(a);
                }
            }
        }
    }
    {
        let letters: Vec<Option<char>> = items
            .iter()
            .map(|it| it.ch.filter(char::is_ascii_alphabetic))
            .collect();
        let scripted: Vec<bool> = items
            .iter()
            .map(|it| !it.sub.is_empty() || !it.sup.is_empty())
            .collect();
        for (k, tex) in function_names(&letters, &scripted) {
            items[k].tex = tex;
        }
    }
    // A relation carrying scripts right before a large operator: the
    // operator's limit, lost to it (`Z = _ { n } \sum …`). (A relation's
    // script before anything else can be a prescript: `= {}_1 F_1`.)
    for w in items.windows(2) {
        let scripted = !w[0].sub.is_empty() || !w[0].sup.is_empty();
        if scripted
            && w[0].ch.is_some()
            && has_relation(&w[0].tex)
            && w[1].ch.is_some_and(big_operator)
        {
            return None;
        }
    }
    let mut out: Vec<String> = Vec::new();
    let mut root_next = false;
    for it in items {
        let mut s = if root_next {
            root_next = false;
            format!("{{ {} }}", it.tex)
        } else {
            it.tex.clone()
        };
        if it.ch == Some('√') && it.sub.is_empty() && it.sup.is_empty() {
            root_next = true;
            out.push(s);
            continue;
        }
        // A script group that reads as nothing (an accent that found no
        // letter) is a misreading, not an empty script.
        let script = |v: Vec<Atom>| line(v).filter(|t| !t.trim().is_empty());
        if !it.sub.is_empty() {
            s.push_str(&format!(" _ {{ {} }}", script(it.sub)?));
        }
        if !it.sup.is_empty() {
            s.push_str(&format!(" ^ {{ {} }}", script(it.sup)?));
        }
        out.push(s);
    }
    Some(out.join(" "))
}

/// `s i n` → `\sin`: a run of letters spelling a function name reads as the
/// operator (its scripts, if any, stay on the last letter's item — `\sin ^ { 2 }`).
fn function_names(letters: &[Option<char>], scripted: &[bool]) -> Vec<(usize, String)> {
    const NAMES: [&str; 22] = [
        "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "sin", "cos", "tan", "cot", "sec",
        "csc", "log", "ln", "exp", "lim", "max", "min", "det", "deg", "gcd", "sup",
    ];
    let n = letters.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let hit = NAMES.iter().find(|name| {
            let len = name.len();
            i + len <= n
                && letters[i..i + len].iter().copied().collect::<Option<String>>().as_deref()
                    == Some(**name)
                // Only the run's last letter may carry scripts.
                && scripted[i..i + len - 1].iter().all(|s| !s)
        });
        match hit {
            Some(name) => {
                let len = name.len();
                for k in i..i + len - 1 {
                    out.push((k, String::new()));
                }
                out.push((i + len - 1, format!("\\{name}")));
                i += len;
            }
            None => i += 1,
        }
    }
    out
}

/// The LaTeX token for a glyph, in the CodeFormula model's spelling; `None`
/// for a character with no confident reading (the region is then left to the
/// model).
fn token(c: char) -> Option<String> {
    // A styled letter from the text layer (𝑥 from a ToUnicode map) reads
    // as the letter.
    let c = fold_math_alphanumeric(c);
    if c.is_ascii_alphanumeric() {
        return Some(c.to_string());
    }
    let t = match c {
        '+' | '=' | '(' | ')' | '[' | ']' | '<' | '>' | '/' | '|' | '!' | ',' | '.' | ';' | ':'
        | '\'' | '*' | '?' => return Some(c.to_string()),
        '-' | '−' | '–' => "-",
        '∶' => ":", // RATIO, how unicode-math fonts draw `:` in math
        '{' => "\\{",
        '}' => "\\}",
        '%' => "\\%",
        '×' => "\\times",
        '÷' => "\\div",
        '·' | '⋅' | '∙' => "\\cdot",
        '±' => "\\pm",
        '∓' => "\\mp",
        '≤' | '⩽' => "\\leq",
        '≥' | '⩾' => "\\geq",
        '≠' => "\\neq",
        '≈' => "\\approx",
        '≡' => "\\equiv",
        '∼' => "\\sim",
        '≃' => "\\simeq",
        '∝' => "\\propto",
        '→' | '⟶' => "\\rightarrow",
        '←' | '⟵' => "\\leftarrow",
        '↔' | '⟷' => "\\leftrightarrow",
        '⇒' | '⟹' => "\\Rightarrow",
        '⇔' | '⟺' => "\\Leftrightarrow",
        '⇌' => "\\rightleftharpoons",
        '∞' => "\\infty",
        '∂' => "\\partial",
        '∇' => "\\nabla",
        '∑' => "\\sum",
        '∏' => "\\prod",
        '∫' => "\\int",
        '∮' => "\\oint",
        '√' => "\\sqrt",
        '∈' => "\\in",
        '∉' => "\\notin",
        '⊂' => "\\subset",
        '⊆' => "\\subseteq",
        '∪' | '⋃' => "\\cup",
        '∩' | '⋂' => "\\cap",
        '∀' => "\\forall",
        '∃' => "\\exists",
        '∅' => "\\emptyset",
        '′' => "\\prime",
        '°' => "\\circ",
        '∘' => "\\circ",
        '…' | '⋯' => "\\cdots",
        '∗' | '⋆' => "*",
        '∆' => "\\Delta",
        '◦' => "\\circ",
        '∥' | '‖' => "\\|",
        '∣' => "|",
        '⊗' => "\\otimes",
        '⊕' => "\\oplus",
        '∧' => "\\wedge",
        '∨' => "\\vee",
        '¬' => "\\neg",
        '⟨' => "\\langle",
        '⟩' => "\\rangle",
        '∖' => "\\setminus",
        '⊥' => "\\perp",
        '∠' => "\\angle",
        '†' => "\\dagger",
        '≪' => "\\ll",
        '≫' => "\\gg",
        '⊃' => "\\supset",
        '⊇' => "\\supseteq",
        '∋' => "\\ni",
        '↦' => "\\mapsto",
        '⇐' | '⟸' => "\\Leftarrow",
        '↑' => "\\uparrow",
        '↓' => "\\downarrow",
        '≺' => "\\prec",
        '≻' => "\\succ",
        '≅' => "\\cong",
        '⌊' => "\\lfloor",
        '⌋' => "\\rfloor",
        '⌈' => "\\lceil",
        '⌉' => "\\rceil",
        '⊙' => "\\odot",
        '≜' => "\\triangleq",
        'ℜ' => "\\Re",
        'ℑ' => "\\Im",
        '⊨' => "\\models",
        '\u{2126}' => "\\Omega", // OHM SIGN
        'ℏ' => "\\hbar",
        'ℓ' => "\\ell",
        'α' => "\\alpha",
        'β' => "\\beta",
        'γ' => "\\gamma",
        'δ' => "\\delta",
        'ε' | 'ϵ' => "\\epsilon",
        'ζ' => "\\zeta",
        'η' => "\\eta",
        'θ' | 'ϑ' => "\\theta",
        'ι' => "\\iota",
        'κ' => "\\kappa",
        'λ' => "\\lambda",
        'μ' | 'µ' => "\\mu",
        'ν' => "\\nu",
        'ξ' => "\\xi",
        'π' => "\\pi",
        'ρ' | 'ϱ' => "\\rho",
        'σ' | 'ς' => "\\sigma",
        'τ' => "\\tau",
        'υ' => "\\upsilon",
        'φ' | 'ϕ' => "\\phi",
        'χ' => "\\chi",
        'ψ' => "\\psi",
        'ω' => "\\omega",
        'Γ' => "\\Gamma",
        'Δ' => "\\Delta",
        'Θ' => "\\Theta",
        'Λ' => "\\Lambda",
        'Ξ' => "\\Xi",
        'Π' => "\\Pi",
        'Σ' => "\\Sigma",
        'Υ' => "\\Upsilon",
        'Φ' => "\\Phi",
        'Ψ' => "\\Psi",
        'Ω' => "\\Omega",
        // Accent glyphs read as accents in `accents`; one left over is not
        // understood.
        _ if accent(c).is_some() => return Some(String::new()),
        // A negation slash: read with the relation it crosses (`joins`).
        '\u{0338}' => return Some(String::new()),
        // A piece of an extensible bracket, brace or integral (`⎛`, `⎜`,
        // `⎝`, `⌠`, …): a tall delimiter this reader does not assemble.
        '\u{2320}' | '\u{2321}' | '\u{239B}'..='\u{23B3}' => return None,
        // Any other symbol the LaTeX → MathML converter (and the speech
        // chain behind it) knows: its own name for it, from its symbol
        // table. The names above win where TeX's and unicode-math's
        // disagree (`ε` is `\epsilon` to docling's model, `\varepsilon` to
        // unicode-math). Nothing known: the region goes to the model.
        _ => return docling_core::latex_command(c).map(str::to_string),
    };
    Some(t.to_string())
}

/// The character a TeX math glyph name stands for: the Adobe glyph list's
/// reading, else Computer Modern's own names for sized delimiters and large
/// operators (CMEX, the `big`/`Big`/`bigg`/`Bigg` and `text`/`display`
/// variants). The *pieces* an extensible delimiter is stacked from
/// (`parenlefttp`, `parenleftex`, `parenleftbt`, …) have no reading of their
/// own: the region is left to the model.
/// The character a named glyph stands for; U+FFFD — unreadable, the region
/// goes to the model — for a name with no reading.
fn read_name(name: &str, shown: char) -> char {
    tex_glyph_name(name).unwrap_or_else(|| {
        docling_core::debug_log!(
            "formula_text: no reading for glyph name {name:?} (shown as {shown:?})"
        );
        char::REPLACEMENT_CHARACTER
    })
}

/// A Unicode-named glyph (`u1D45B`, `uni03B1`, `u1D463.alt` — the
/// convention OpenType-derived math fonts use) as its character, a
/// mathematical alphanumeric folded to the plain letter, digit or Greek
/// letter it is a style of (𝑛 → n, 𝜃 → θ): the style is a typeface the
/// reading does not carry.
fn unicode_glyph_name(name: &str) -> Option<char> {
    let base = name.split('.').next().unwrap_or(name);
    let hex = base
        .strip_prefix("uni")
        .or_else(|| base.strip_prefix('u'))?;
    if !(4..=6).contains(&hex.len()) || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let cp = u32::from_str_radix(hex, 16).ok()?;
    Some(fold_math_alphanumeric(char::from_u32(cp)?))
}

/// The plain character a Mathematical Alphanumeric Symbols code point
/// (U+1D400–U+1D7FF) styles; any other character unchanged.
fn fold_math_alphanumeric(c: char) -> char {
    const GREEK: &str = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡϴΣΤΥΦΧΨΩ∇αβγδεζηθικλμνξοπρςστυφχψω∂ϵϑϰϕϱϖ";
    // Greek capitals drawn like Latin ones read as the Latin letter.
    let latin_look = |g: char| match g {
        'Α' => 'A',
        'Β' => 'B',
        'Ε' => 'E',
        'Ζ' => 'Z',
        'Η' => 'H',
        'Ι' => 'I',
        'Κ' => 'K',
        'Μ' => 'M',
        'Ν' => 'N',
        'Ο' => 'O',
        'Ρ' => 'P',
        'Τ' => 'T',
        'Χ' => 'X',
        'ϴ' => 'Θ',
        g => g,
    };
    let cp = c as u32;
    match cp {
        0x210E => 'h', // ℎ, the italic h the block leaves out
        // The Letterlike Symbols the block leaves out: double-struck
        // (ℂ ℍ ℕ ℙ ℚ ℝ ℤ), script (ℬ ℰ ℱ ℋ ℐ ℒ ℳ ℛ ℯ ℊ ℴ) and Fraktur (ℭ ℌ ℑ ℜ ℨ).
        0x2102 | 0x212D => 'C',
        0x210B..=0x210D => 'H', // ℋ ℌ ℍ
        0x2115 => 'N',
        0x2119 => 'P',
        0x211A => 'Q',
        // (ℜ and ℑ are the real- and imaginary-part operators, not
        // letters: they keep their own reading.)
        0x211D | 0x211B => 'R',
        0x2124 | 0x2128 => 'Z',
        0x212C => 'B',
        0x2130 => 'E',
        0x2131 => 'F',
        0x2110 => 'I',
        0x2112 => 'L',
        0x2133 => 'M',
        0x212F => 'e',
        0x210A => 'g',
        0x2134 => 'o',
        0x1D400..=0x1D6A3 => {
            let k = ((cp - 0x1D400) % 52) as u8;
            if k < 26 {
                (b'A' + k) as char
            } else {
                (b'a' + k - 26) as char
            }
        }
        0x1D6A8..=0x1D7C9 => GREEK
            .chars()
            .nth(((cp - 0x1D6A8) % 58) as usize)
            .map_or(c, latin_look),
        0x1D7CE..=0x1D7FF => (b'0' + ((cp - 0x1D7CE) % 10) as u8) as char,
        _ => c,
    }
}

fn tex_glyph_name(name: &str) -> Option<char> {
    if let Some(c) = unicode_glyph_name(name) {
        return Some(c);
    }
    // Blackboard and calligraphic capitals (AMS `Rbbb`, `Fcal`): the plain
    // letter — the alphabet is a typeface the reading does not carry.
    for suffix in ["bbb", "cal"] {
        if let Some(letter) = name.strip_suffix(suffix) {
            let mut cs = letter.chars();
            if let (Some(c), None) = (cs.next(), cs.next()) {
                if c.is_ascii_alphabetic() {
                    return Some(c);
                }
            }
        }
    }
    // `parenrightbig4`, `radicalbig1`: a numbered size of a sized glyph.
    let name = name.trim_end_matches(|c: char| c.is_ascii_digit());
    let stem = name
        .trim_end_matches("display")
        .trim_end_matches("text")
        .trim_end_matches("Bigg")
        .trim_end_matches("bigg")
        .trim_end_matches("Big")
        .trim_end_matches("big");
    let sized = stem.len() < name.len();
    if sized {
        let ch = match stem {
            "parenleft" => '(',
            "parenright" => ')',
            "bracketleft" => '[',
            "bracketright" => ']',
            "braceleft" => '{',
            "braceright" => '}',
            "angbracketleft" => '⟨',
            "angbracketright" => '⟩',
            "slash" => '/',
            "radical" => '√',
            "summation" => '∑',
            "product" => '∏',
            "integral" => '∫',
            "contintegral" => '∮',
            "union" => '⋃',
            "intersection" => '⋂',
            "circleplus" => '⊕',
            "circlemultiply" => '⊗',
            "circledot" => '⊙',
            _ => return None,
        };
        return Some(ch);
    }
    // Computer Modern names the Adobe glyph list does not know.
    let tex = match name {
        "vector" => Some('⃗'),
        "prime" => Some('′'),
        "asteriskmath" => Some('∗'),
        "minus" => Some('−'),
        "mapsto" | "mapstochar" => Some('↦'),
        "negationslash" => Some('\u{0338}'),
        "nequal" => Some('≠'),
        "nelement" => Some('∉'),
        "radicallow" => Some('√'),
        n if n.starts_with("hatwide") => Some('ˆ'),
        n if n.starts_with("tildewide") => Some('˜'),
        _ => None,
    };
    tex.or_else(|| crate::textparse::glyph_name_to_char(name.as_bytes()))
}

fn accent(c: char) -> Option<&'static str> {
    // Spacing accents, and the combining marks unicode-math fonts map them
    // to (U+0300 block, U+20D7).
    Some(match c {
        'ˆ' | '^' | '\u{0302}' => "\\hat",
        '˜' | '~' | '\u{0303}' => "\\tilde",
        '¯' | 'ˉ' | '\u{0304}' | '\u{0305}' => "\\bar",
        '⃗' => "\\vec",
        '˙' | '\u{0307}' => "\\dot",
        '¨' | '\u{0308}' => "\\ddot",
        'ˇ' | '\u{030C}' => "\\check",
        '´' | '\u{0301}' => "\\acute",
        '`' | '\u{0300}' => "\\grave",
        '˘' | '\u{0306}' => "\\breve",
        _ => return None,
    })
}

fn delimiter(c: char) -> bool {
    matches!(
        c,
        '(' | ')' | '[' | ']' | '{' | '}' | '|' | '⟨' | '⟩' | '⌊' | '⌋' | '⌈' | '⌉' | '‖' | '∥'
    )
}

fn big_operator(c: char) -> bool {
    matches!(c, '∑' | '∏' | '∫' | '∮' | '⋃' | '⋂')
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A glyph at `x` on baseline `base` (top-left frame), `size` points:
    /// a box of 0.55 em advance, 0.75 em ascent, 0.25 em descent.
    fn g(ch: char, x: f32, base: f32, size: f32) -> MathGlyph {
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
        }
    }

    /// Glyphs laid left to right from `x` on one baseline.
    fn run(s: &str, x: f32, base: f32, size: f32) -> Vec<MathGlyph> {
        s.chars()
            .enumerate()
            .map(|(i, c)| g(c, x + i as f32 * 0.6 * size, base, size))
            .collect()
    }

    fn rebuild(glyphs: Vec<MathGlyph>, rules: Vec<Rule>) -> Option<String> {
        explain(&MathLayer { glyphs, rules }, [0.0, 0.0, 600.0, 800.0]).ok()
    }

    #[test]
    fn a_superscript_is_a_smaller_raised_glyph() {
        // code_and_formula.pdf's `a^2 + 8 = 12`, spelled as docling's model
        // spells it.
        let mut gs = vec![g('a', 100.0, 200.0, 10.0), g('2', 105.5, 196.4, 7.0)];
        gs.extend(run("+8=12", 112.0, 200.0, 10.0));
        assert_eq!(rebuild(gs, vec![]).as_deref(), Some("a ^ { 2 } + 8 = 1 2"));
    }

    #[test]
    fn chemistry_subscripts_and_charges() {
        // K_sp = [Cd^{2+}]: subscripts lowered, a charge raised.
        let mut gs = vec![g('K', 100.0, 200.0, 9.0)];
        gs.extend(run("sp", 105.7, 201.8, 7.5));
        gs.extend(run("=[Cd", 116.0, 200.0, 9.0));
        gs.extend(run("2+", 137.6, 197.0, 7.5));
        gs.push(g(']', 147.0, 200.0, 9.0));
        assert_eq!(
            rebuild(gs, vec![]).as_deref(),
            Some("K _ { s p } = [ C d ^ { 2 + } ]")
        );
    }

    #[test]
    fn a_fraction_is_two_stacks_either_side_of_a_rule() {
        // x = \frac{a+1}{2b}
        let mut gs = run("x=", 100.0, 200.0, 10.0);
        gs.extend(run("a+1", 115.0, 193.0, 10.0));
        gs.extend(run("2b", 118.0, 210.0, 10.0));
        let bar = Rule {
            l: 114.0,
            t: 197.3,
            r: 134.0,
            b: 197.7,
        };
        assert_eq!(
            rebuild(gs, vec![bar]).as_deref(),
            Some("x = \\frac { a + 1 } { 2 b }")
        );
    }

    #[test]
    fn nested_fractions_resolve_inside_out() {
        // \frac{\frac{1}{2}}{3}
        let mut gs = vec![g('1', 103.0, 180.0, 10.0), g('2', 103.0, 196.0, 10.0)];
        gs.push(g('3', 103.0, 214.0, 10.0));
        let inner = Rule {
            l: 102.0,
            t: 184.8,
            r: 109.0,
            b: 185.2,
        };
        let outer = Rule {
            l: 100.0,
            t: 201.8,
            r: 112.0,
            b: 202.2,
        };
        assert_eq!(
            rebuild(gs, vec![inner, outer]).as_deref(),
            Some("\\frac { \\frac { 1 } { 2 } } { 3 }")
        );
    }

    #[test]
    fn a_root_is_a_radical_and_its_vinculum() {
        // \sqrt{x+1}
        let mut gs = vec![g('√', 100.0, 200.0, 10.0)];
        gs.extend(run("x+1", 106.0, 200.0, 10.0));
        let vinculum = Rule {
            l: 105.5,
            t: 191.8,
            r: 124.0,
            b: 192.2,
        };
        assert_eq!(
            rebuild(gs, vec![vinculum]).as_deref(),
            Some("\\sqrt { x + 1 }")
        );
    }

    #[test]
    fn function_names_and_nested_scripts() {
        // \sin^2 x + e^{x^2}
        let mut gs = run("sin", 100.0, 200.0, 10.0);
        gs.push(g('2', 118.0, 196.4, 7.0));
        gs.extend(run("x+e", 123.0, 200.0, 10.0));
        gs.push(g('x', 141.0, 196.4, 7.0));
        gs.push(g('2', 145.0, 194.0, 5.0));
        assert_eq!(
            rebuild(gs, vec![]).as_deref(),
            Some("\\sin ^ { 2 } x + e ^ { x ^ { 2 } }")
        );
    }

    #[test]
    fn unicode_named_and_styled_letters_read_as_plain_letters() {
        use super::{tex_glyph_name, token};
        assert_eq!(tex_glyph_name("u1D45B"), Some('n')); // 𝑛
        assert_eq!(tex_glyph_name("u1D463.alt"), Some('v'));
        assert_eq!(tex_glyph_name("u1D6FC"), Some('α')); // 𝛼
        assert_eq!(tex_glyph_name("u1D703"), Some('θ')); // 𝜃
        assert_eq!(tex_glyph_name("uni03B1"), Some('α'));
        assert_eq!(tex_glyph_name("u1D7D0"), Some('2')); // 𝟐
        assert_eq!(token('𝑥').as_deref(), Some("x"));
        assert_eq!(token('ℎ').as_deref(), Some("h"));
        assert_eq!(token('𝛤').as_deref(), Some("\\Gamma"));
    }

    /// Symbols outside the hand-written table read through the speech
    /// converter's own symbol table, so every one is a command it knows.
    #[test]
    fn rarer_symbols_read_through_the_converters_table() {
        use super::token;
        for (c, want) in [
            ('⊢', "\\vdash"),
            ('∴', "\\therefore"),
            ('⊊', "\\subsetneq"),
            ('ℵ', "\\aleph"),
        ] {
            assert_eq!(token(c).as_deref(), Some(want), "{c}");
        }
        // The hand-written names still win.
        assert_eq!(token('ε').as_deref(), Some("\\epsilon"));
        assert_eq!(token('≤').as_deref(), Some("\\leq"));
    }

    #[test]
    fn what_cannot_be_read_is_left_to_the_model() {
        // An unmapped glyph (a TeX extensible-delimiter piece).
        let gs = vec![g('⎛', 100.0, 200.0, 10.0), g('x', 108.0, 200.0, 10.0)];
        assert_eq!(rebuild(gs, vec![]), None);
        // A rule with nothing above or below it.
        let gs = run("x=1", 100.0, 200.0, 10.0);
        let stray = Rule {
            l: 200.0,
            t: 150.0,
            r: 260.0,
            b: 150.4,
        };
        assert_eq!(rebuild(gs, vec![stray]), None);
        // Two full-size glyphs stacked (a matrix column).
        let gs = vec![g('a', 100.0, 190.0, 10.0), g('b', 100.0, 191.0, 10.0)];
        assert_eq!(rebuild(gs, vec![]), None);
        // A rotated glyph.
        let mut t = g('x', 100.0, 200.0, 10.0);
        t.tilted = true;
        assert_eq!(rebuild(vec![t], vec![]), None);
        // Nothing in the region: a formula drawn as outlines.
        assert_eq!(rebuild(vec![], vec![]), None);
    }

    #[test]
    fn rows_of_a_display_are_left_to_the_model_for_now() {
        let mut gs = run("x=1", 100.0, 200.0, 10.0);
        gs.extend(run("y=2", 100.0, 216.0, 10.0));
        // Several rows are left to the model until formulas are delimited
        // from their own glyphs (see `Builder::run`).
        assert_eq!(rebuild(gs, vec![]), None);
    }

    /// Every display formula of the pdf-parse-bench corpus whose region the
    /// layout model found (`tests/data/pdf/formula_text/`), replayed from its
    /// recorded glyph layer: the output must be the one recorded, byte for
    /// byte, refusals included. The recording, and each case's status
    /// against the ground-truth LaTeX (exact / near / wrong / refused), come
    /// from `scripts/conformance/formula_text_eval.py`; after an intended
    /// change, re-run it with `--replay --fixtures` and review what moved.
    #[cfg(feature = "ml")]
    #[test]
    fn benchmark_formulas_rebuild_as_recorded() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/data/pdf/formula_text/cases.json");
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("skipped: {} not found", path.display());
            return;
        };
        let cases: Vec<serde_json::Value> = serde_json::from_slice(&bytes).expect("cases.json");
        let mut changed = Vec::new();
        let mut tally = std::collections::BTreeMap::<String, usize>::new();
        for case in &cases {
            *tally
                .entry(case["status"].as_str().unwrap_or("?").to_string())
                .or_default() += 1;
            let (layer, bbox) = layer_from_json(case).expect("a readable case");
            let got = explain(&layer, bbox).ok();
            let want = case["latex"].as_str();
            if got.as_deref() != want {
                changed.push(format!(
                    "{} [{}]\n    recorded: {want:?}\n    now:      {got:?}",
                    case["id"].as_str().unwrap_or("?"),
                    case["status"].as_str().unwrap_or("?"),
                ));
            }
        }
        eprintln!("formula_text corpus: {} cases {tally:?}", cases.len());
        assert!(
            changed.is_empty(),
            "{} of {} cases changed (re-score with formula_text_eval.py --replay):\n{}",
            changed.len(),
            cases.len(),
            changed
                .iter()
                .take(20)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}
