//! Spoken-form rendering of formula items: LaTeX → MathML → English.
//!
//! A screen reader speaks a formula; a TTS reader has to as well, and neither
//! can do anything useful with `$$a^{2}+8=12$$`. This module turns the LaTeX a
//! [`Node::Formula`](crate::Node::Formula) carries into a sentence
//! ("eigh squared plus 8, is equal to 12") that lands on the JSON item as the
//! additive `speech` map, one entry per speech style.
//!
//! The chain is pure Rust, which is why it lives server-side rather than in a
//! client:
//!
//! 1. [`normalize_latex`] rejoins the space-separated tokens the CodeFormula
//!    model emits — the step without which every multi-digit number breaks.
//! 2. `crate::mathml`, the HTML export's port of Python `latex2mathml`
//!    3.81, parses that into MathML. The `latex2mathml` 0.2 *crate* this
//!    first ran on nested `X_{a}^{b}` as a superscript inside the subscript
//!    (so `SO_4^{2-}` carried its charge on the 4), wrote `<` raw, and knew
//!    no `\xrightarrow`, `\prime`, `\rm` or `array`; on the fixture corpus
//!    of 776 formula lines it voiced 466 against the port's 679 (`math-core`
//!    0.9, the maintained crate, 540 — it also needs a 1.96 toolchain). The
//!    port voices everything either crate does.
//! 3. MathCAT (the DAISY engine behind NVDA and JAWS) speaks the MathML. Its
//!    chemistry heuristics fire on their own — `\rightarrow` in a chemical
//!    equation reads as "reacts to form", not "right arrow" — so chemistry
//!    needs no separate speech path. It does need the MathML a chemist would
//!    write, and a formula recognizer does not produce that: the LaTeX
//!    rewrites between steps 1 and 2 (`group_elements`, `chemical_arrows`,
//!    `segments`, …) close that gap. Each exists because a real textbook
//!    equation came out as "cap m, the 2 plus power of, g of eigh q" instead of
//!    "cap m g, super 2 plus, aqueous".
//!
//! Everything here is behind the `speech` feature; with it off the JSON export
//! is byte-identical, so no groundtruth moves.

/// Rejoin the space-separated tokens a formula recognizer emits.
///
/// CodeFormulaV2 writes `a ^ { 2 } + 8 = 1 2`, one token at a time. Fed to a
/// MathML parser as-is, `1 2` becomes two separate `<mn>` nodes and a speech
/// engine says "one two" instead of "twelve"; `1 0 0 . 5` shatters into five
/// nodes. Every number in every formula is wrong, quietly.
///
/// Removing *all* spaces would break the other direction: a control word needs
/// its terminating space when a letter follows (`\cdot E` must not become
/// `\cdotE`). So a space survives only in two cases — between a control word
/// and a following letter, and after a lone backslash, where it *is* the
/// command. `\rightarrow 2` needs no space because a digit cannot extend the
/// command name.
pub fn normalize_latex(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut chars = src.chars().peekable();
    while let Some(c) = chars.next() {
        if c != ' ' {
            out.push(c);
            continue;
        }
        // `\ ` is a command of its own (a control space, `285.8\ \mathrm{kJ}`):
        // dropping its space would fuse the backslash with the next token.
        let escaped = out.len() - out.trim_end_matches('\\').len();
        if escaped % 2 == 1 {
            out.push(' ');
            continue;
        }
        // Is what we've emitted so far ending in a control word (`\` followed by
        // letters only)? Then a following letter would extend it.
        let ends_control_word = out.rfind('\\').is_some_and(|i| {
            let tail = &out[i + 1..];
            !tail.is_empty() && tail.chars().all(|c| c.is_ascii_alphabetic())
        });
        if ends_control_word && chars.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
            out.push(' ');
        }
    }
    out
}

/// Is this LaTeX unambiguously chemistry?
///
/// The rewrites below change what a formula *means* when they are wrong
/// (`Te` is tellurium in a reaction and `T·e` in `kTe^{-E/kT}`; `Re(s)` is a
/// real part), so they only run on evidence a mathematician never produces:
///
/// - an `(aq)` state mark, which has no reading in mathematics;
/// - an ionic charge — a superscript that is only a sign, or digits and a
///   sign, in either order (`^{+}`, `^{2+}`, `^{+2}`). `e^{-x}` is not one;
/// - an equilibrium arrow (`\rightleftharpoons`);
/// - any other state mark (`(s)`, `(g)`, `(l)`) **together with** a reaction
///   arrow. Alone, `(s)` is a Laplace variable and `(g)` a function argument.
#[cfg_attr(not(feature = "speech"), allow(dead_code))] // read by the speech chain only
fn is_chemistry(latex: &str) -> bool {
    let flat: String = latex.chars().filter(|c| !c.is_whitespace()).collect();
    if flat.contains("(aq)") || flat.contains("\\rightleftharpoons") {
        return true;
    }
    let charged = flat.match_indices("^{").any(|(i, _)| {
        let inner: &str = flat[i + 2..].split('}').next().unwrap_or("");
        let is_sign = |c: char| matches!(c, '+' | '-' | '−');
        let digits_then_sign = inner.trim_end_matches(is_sign);
        let sign_then_digits = inner.trim_start_matches(is_sign);
        !inner.is_empty()
            && ((digits_then_sign.len() + 1 == inner.len()
                && digits_then_sign.chars().all(|c| c.is_ascii_digit()))
                || (sign_then_digits.len() + 1 == inner.len()
                    && sign_then_digits.chars().all(|c| c.is_ascii_digit())))
    });
    if charged {
        return true;
    }
    let state = ["(s)", "(g)", "(l)"].iter().any(|m| flat.contains(m));
    let arrow = [
        "\\rightarrow",
        "\\longrightarrow",
        "\\Rightarrow",
        "\\Longrightarrow",
        "\\leftrightarrow",
        "\\Leftrightarrow",
        "\\longleftrightarrow",
        "\\Longleftrightarrow",
        "\\xrightarrow",
        "\\to",
    ]
    .iter()
    .any(|a| flat.contains(a));
    state && arrow
}

/// The two-letter element symbols of the periodic table.
///
/// One-letter symbols (`H`, `C`, `O`, …) need no grouping: a single `<mi>` is
/// already one token to MathCAT. Three-letter placeholders for undiscovered
/// elements do not occur in textbooks.
#[cfg_attr(not(feature = "speech"), allow(dead_code))] // read by the speech chain only
const ELEMENTS_2: &[&str] = &[
    "He", "Li", "Be", "Ne", "Na", "Mg", "Al", "Si", "Cl", "Ar", "Ca", "Sc", "Ti", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Zr", "Nb", "Mo", "Tc",
    "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "Np", "Pu",
    "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

/// Byte index just past the control word (or single escaped character)
/// starting at the backslash `at`.
#[cfg_attr(not(feature = "speech"), allow(dead_code))] // read by the speech chain only
fn control_word_end(s: &str, at: usize) -> usize {
    debug_assert_eq!(s.as_bytes()[at], b'\\');
    let letters = s[at + 1..]
        .bytes()
        .take_while(|b| b.is_ascii_alphabetic())
        .count();
    if letters > 0 {
        at + 1 + letters
    } else {
        // `\{`, `\,`, `\\`: one escaped character (ASCII in practice).
        at + 1 + s[at + 1..].chars().next().map_or(0, char::len_utf8)
    }
}

/// Byte index just past the brace group opening at `at`; the string's end
/// when the group never closes.
#[cfg_attr(not(feature = "speech"), allow(dead_code))] // read by the speech chain only
fn group_end(s: &str, at: usize) -> usize {
    debug_assert_eq!(s.as_bytes()[at], b'{');
    let mut depth = 0usize;
    let mut i = at;
    let b = s.as_bytes();
    while i < b.len() {
        match b[i] {
            b'\\' => {
                i = control_word_end(s, i);
                continue;
            }
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    b.len()
}

/// Brace each two-letter element symbol so it reaches MathML as one group.
///
/// A formula recognizer emits one character per token, and a MathML
/// converter turns `Mg^{2+}` into `<mi>M</mi><msup><mi>g</mi>…` — the charge
/// sits on the `g` alone, and MathCAT, which merges adjacent single-letter
/// `<mi>`s into element symbols, cannot merge across that boundary. It then
/// reads "cap m, the 2 plus power of, g" and, because the two sides of the
/// equation no longer hold the same elements, rejects the whole equation as
/// chemistry: the `⇌` is spoken as "right harpoon over left harpoon".
/// Grouped as `{Mg}^{2+}` the charge lands on an `<mrow>` of the two letters,
/// MathCAT merges them, and the equation is narrated as a reaction. (Not
/// `\mathrm{Mg}`: a converter may emit that as one `<mi>Mg</mi>`, which
/// outside a recognized equation is spoken as the word "Mg".)
///
/// Only called when [`is_chemistry`] holds: in mathematics `Pa`, `Re`, `Te`
/// and `In` are products of variables. Letters inside a control word
/// (`\Gamma`, `\Sigma`, `\Lambda` contain `Ga`, `Si`, `La`) and inside
/// `\text{…}` are never touched, and a lowercase letter that continues into
/// a word (`Natural`) is not a symbol.
#[cfg_attr(not(feature = "speech"), allow(dead_code))] // read by the speech chain only
fn group_elements(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len() + 16);
    let mut i = 0;
    // Whether the previous character was the end of a symbol just grouped —
    // the `a` of `{Na}` in `NaCl` is not a word continuing into `Cl`.
    let mut after_symbol = false;
    while i < b.len() {
        if b[i] == b'\\' {
            after_symbol = false;
            let end = control_word_end(s, i);
            out.push_str(&s[i..end]);
            i = end;
            if &s[end - 4..end] == "text" && s[i - 5..i].starts_with('\\') {
                // Copy `\text{…}` whole: its letters are words.
                let mut j = i;
                while b.get(j) == Some(&b' ') {
                    j += 1;
                }
                if b.get(j) == Some(&b'{') {
                    let e = group_end(s, j);
                    out.push_str(&s[i..e]);
                    i = e;
                }
            }
            continue;
        }
        if b[i].is_ascii_uppercase()
            && b.get(i + 1).is_some_and(u8::is_ascii_lowercase)
            && !b.get(i + 2).is_some_and(u8::is_ascii_lowercase)
            && !(i > 0 && b[i - 1].is_ascii_lowercase() && !after_symbol)
            && ELEMENTS_2.contains(&&s[i..i + 2])
        {
            out.push('{');
            out.push_str(&s[i..i + 2]);
            out.push('}');
            i += 2;
            after_symbol = true;
            continue;
        }
        after_symbol = false;
        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Drop the `\mathrm{…}` around element symbols in a chemical formula:
/// `\mathrm{COOH}(aq)` → `COOH(aq)`.
///
/// A formula recognizer sets every formula upright (`\mathrm{CH}_{3}\mathrm{COOH}`).
/// The converter turns a multi-letter `\mathrm` group into one `<mi>`, and
/// MathCAT reads a parenthesis right after a multi-letter identifier as its
/// argument: `\mathrm{CO}(g)` was "cap c cap o, times g", and one such group
/// in an equation fails MathCAT's chemistry check for the whole of it (an
/// acid-base equilibrium came out "times eigh q … right harpoon over left
/// harpoon"). Bare letters are what [`group_elements`] expects and what
/// MathCAT merges into element symbols. Only groups of letters and digits are
/// unwrapped (spaces from the recognizer's tokenization dropped); anything
/// else in an `\mathrm` stays as it is. Only called for chemistry.
#[cfg_attr(not(feature = "speech"), allow(dead_code))] // read by the speech chain only
fn unwrap_upright(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(at) = rest.find("\\mathrm") {
        out.push_str(&rest[..at]);
        let after = &rest[at + "\\mathrm".len()..];
        let open = after.len() - after.trim_start_matches(' ').len();
        let body = &after[open..];
        let close = body.find('}');
        match (body.starts_with('{'), close) {
            (true, Some(end))
                if body[1..end].chars().any(|c| c.is_ascii_alphanumeric())
                    && body[1..end]
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == ' ') =>
            {
                // `\to\mathrm{CH}` must not become `\toCH`: after a control word
                // the letters need the space that ends it.
                let tail = out.rsplit('\\').next().unwrap_or("");
                if out.contains('\\')
                    && !tail.is_empty()
                    && tail.chars().all(|c| c.is_ascii_alphabetic())
                {
                    out.push(' ');
                }
                out.extend(body[1..end].chars().filter(|c| *c != ' '));
                rest = &body[end + 1..];
            }
            _ => {
                out.push_str("\\mathrm");
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Spell `\xrightarrow[below]{above}` and `\xleftarrow` as `\overset` /
/// `\underset` over a plain arrow.
///
/// A reaction condition over the arrow (`\xrightarrow{\Delta}`,
/// `\xrightarrow{\text{heat}}`) is textbook routine. The converter renders
/// the bracketed form as an `<munderover>` of `<mstyle>`/`<mpadded>` that
/// MathCAT refuses ("Pattern match/replacement failure"), and the reader
/// hears nothing. `\overset{\Delta}{\longrightarrow}` is a plain `<mover>`
/// on the arrow, which MathCAT voices as "reacts to form with cap delta
/// above"; both forms are rewritten so they speak alike.
#[cfg_attr(not(feature = "speech"), allow(dead_code))] // read by the speech chain only
fn unfold_extensible_arrows(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    let mut rest = s;
    loop {
        let Some(pos) = rest.find("\\x") else {
            out.push_str(rest);
            return out;
        };
        let end = control_word_end(rest, pos);
        let arrow = match &rest[pos..end] {
            "\\xrightarrow" => "\\longrightarrow",
            "\\xleftarrow" => "\\longleftarrow",
            _ => {
                out.push_str(&rest[..end]);
                rest = &rest[end..];
                continue;
            }
        };
        let mut i = end;
        let b = rest.as_bytes();
        let below = if b.get(i) == Some(&b'[') {
            let close = rest[i..].find(']').map(|c| i + c);
            match close {
                Some(c) => {
                    let below = &rest[i + 1..c];
                    i = c + 1;
                    Some(below)
                }
                None => None,
            }
        } else {
            None
        };
        if b.get(i) != Some(&b'{') {
            // Malformed: leave it, the parse error surfaces downstream.
            out.push_str(&rest[..end]);
            rest = &rest[end..];
            continue;
        }
        let g_end = group_end(rest, i);
        let above = &rest[i + 1..g_end - 1];
        out.push_str(&rest[..pos]);
        match below {
            Some(under) => {
                out.push_str("\\underset{");
                out.push_str(under);
                out.push_str("}{\\overset{");
                out.push_str(above);
                out.push_str("}{");
                out.push_str(arrow);
                out.push_str("}}");
            }
            None => {
                out.push_str("\\overset{");
                out.push_str(above);
                out.push_str("}{");
                out.push_str(arrow);
                out.push('}');
            }
        }
        rest = &rest[g_end..];
    }
}

/// Take state marks out of `\text{…}`: `\text{(aq)}` → `(aq)`.
///
/// Both forms are written, but an `<mtext>(aq)</mtext>` is prose to MathCAT
/// (spoken "(aq)"), while bare `(aq)` is exactly the
/// `<mo>(</mo><mi>aq</mi><mo>)</mo>` its state detection wants.
#[cfg_attr(not(feature = "speech"), allow(dead_code))] // read by the speech chain only
fn bare_states(s: &str) -> String {
    let mut out = s.to_string();
    for state in ["(aq)", "(s)", "(g)", "(l)"] {
        for form in [format!("\\text{{{state}}}"), format!("\\mathrm{{{state}}}")] {
            out = out.replace(&form, state);
        }
    }
    out
}

/// Split a formula at its top-level `\quad` / `\qquad` gaps.
///
/// A display equation and its annotation share one formula item:
/// `CH_4 + 2O_2 ⟶ CO_2 + 2H_2O \quad \Delta H = -890 kJ`. Fed to MathCAT as
/// one expression, the trailing equation makes the whole thing fail the
/// chemistry check (the arrow becomes "long rightwards arrow" and the two
/// parts run together as a product). Spoken part by part, each is what it
/// is. The split is only at brace depth zero, so a `\quad` inside a group
/// stays where it is.
#[cfg_attr(not(feature = "speech"), allow(dead_code))] // read by the speech chain only
fn segments(latex: &str) -> Vec<&str> {
    let b = latex.as_bytes();
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'{' => depth += 1,
            b'}' => depth = depth.saturating_sub(1),
            b'\\' => {
                let end = control_word_end(latex, i);
                if depth == 0 && matches!(&latex[i..end], "\\quad" | "\\qquad") {
                    parts.push(&latex[start..i]);
                    start = end;
                }
                i = end;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    parts.push(&latex[start..]);
    parts
        .into_iter()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect()
}

/// A word label in front of a formula, `\mathrm{Net}: …` or `\text{Overall:} …`,
/// and the formula after it.
///
/// Textbooks head an equation with a word ("Net:", "Neto:", "Overall:").
/// Spoken inside the expression its letters are variables or, in chemistry,
/// element symbols, and they fail MathCAT's chemistry check for the whole
/// equation (an equilibrium read "right harpoon over left harpoon"). A label
/// is a run of at least three letters (spaces from a recognizer's
/// tokenization dropped when every token is one letter) in `\mathrm` or
/// `\text`, ending in a colon inside or right after the group.
#[cfg_attr(not(feature = "speech"), allow(dead_code))] // read by the speech chain only
fn leading_label(seg: &str) -> (Option<String>, &str) {
    let s = seg.trim_start();
    let Some(after) = s
        .strip_prefix("\\mathrm")
        .or_else(|| s.strip_prefix("\\text"))
    else {
        return (None, seg);
    };
    let after = after.trim_start();
    let Some(body) = after.strip_prefix('{') else {
        return (None, seg);
    };
    let Some(end) = body.find('}') else {
        return (None, seg);
    };
    let inner = body[..end].trim();
    let (inner, colon_inside) = match inner.strip_suffix(':') {
        Some(i) => (i.trim_end(), true),
        None => (inner, false),
    };
    let mut rest = &body[end + 1..];
    if !colon_inside {
        match rest.trim_start().strip_prefix(':') {
            Some(r) => rest = r,
            None => return (None, seg),
        }
    }
    let toks: Vec<&str> = inner.split_whitespace().collect();
    let word = if toks.len() > 1 && toks.iter().all(|t| t.chars().count() == 1) {
        toks.concat()
    } else {
        toks.join(" ")
    };
    if word.chars().filter(|c| c.is_alphabetic()).count() < 3
        || !word.chars().all(|c| c.is_alphabetic() || c == ' ')
    {
        return (None, seg);
    }
    (Some(word), rest)
}

/// A formula segment's prose, if it is nothing but a `\text{…}` note (with or
/// without parentheses around it).
#[cfg_attr(not(feature = "speech"), allow(dead_code))] // read by the speech chain only
fn words_only(seg: &str) -> Option<&str> {
    let s = seg.trim();
    let inner = s
        .strip_prefix('(')
        .and_then(|r| r.strip_suffix(')'))
        .unwrap_or(s)
        .trim();
    let rest = inner.strip_prefix("\\text")?.trim_start();
    let note = rest.strip_prefix('{')?.strip_suffix('}')?.trim();
    (!note.is_empty() && !note.contains('{') && !note.contains('\\')).then_some(note)
}

/// Rewrite logic arrows as chemical ones when the formula is plainly chemistry.
///
/// A formula recognizer reaches for the arrow that *looks* right and gets the
/// meaning wrong: it writes `\Longrightarrow` (implication) for a reaction and
/// `\Longleftrightarrow` (equivalence) for an equilibrium. MathCAT then reads
/// them exactly as written — "implies", "if and only if" — so a chemical
/// equation is narrated as a logical proposition. With the chemical arrows it
/// says "reacts to form" and "is in equilibrium with" instead.
///
/// Only applied when the expression carries unambiguous chemistry: a state
/// symbol (`(aq)`, `(s)`, `(g)`, `(l)`) or an ionic charge (`^{+…}`/`^{-…}`).
/// In mathematics those arrows mean what they say, and a generic
/// `aA + bB ⟺ eE + fF` with neither marker is genuinely ambiguous without
/// document context — it is left alone rather than guessed at.
#[cfg_attr(not(feature = "speech"), allow(dead_code))] // read by the speech chain only
fn chemical_arrows(latex: &str) -> std::borrow::Cow<'_, str> {
    if !is_chemistry(latex) {
        return std::borrow::Cow::Borrowed(latex);
    }
    let mut out = latex.to_string();
    for (logic, chem) in [
        ("\\Longleftrightarrow", "\\rightleftharpoons"),
        ("\\Leftrightarrow", "\\rightleftharpoons"),
        ("\\longleftrightarrow", "\\rightleftharpoons"),
        ("\\leftrightarrow", "\\rightleftharpoons"),
        ("\\Longrightarrow", "\\longrightarrow"),
        ("\\Rightarrow", "\\longrightarrow"),
    ] {
        out = out.replace(logic, chem);
    }
    std::borrow::Cow::Owned(out)
}

/// Split a trailing prose annotation off a formula: `… \quad ( \text {balanced} )`
/// becomes the equation and `"balanced"`.
///
/// Textbooks annotate display equations this way, and the annotation is content
/// a reader should hear — but leaving it attached costs more than it gives.
/// MathCAT decides whether an expression is *chemistry* from its shape, and the
/// trailing group defeats that: `2H₂O ⟶ 2H₂ + O₂` speaks its arrow as "reacts
/// to form", while the same equation with `(balanced)` appended degrades to
/// "long rightwards arrow ... of, open paren balanced close paren".
///
/// So the equation is spoken as chemistry and the annotation appended as words.
#[cfg_attr(not(feature = "speech"), allow(dead_code))] // read by the speech chain only
pub(crate) fn split_annotation(latex: &str) -> (&str, Option<&str>) {
    let s = latex.trim_end();
    let Some(body) = s.strip_suffix(')') else {
        return (latex, None);
    };
    let Some(open) = body.rfind('(') else {
        return (latex, None);
    };
    // The parenthesised group has to be exactly one `\text{…}` — anything else
    // is part of the maths (`(a + b)`) and must stay.
    let inner = body[open + 1..].trim();
    let Some(rest) = inner.strip_prefix(r"\text") else {
        return (latex, None);
    };
    let rest = rest.trim_start();
    let Some(rest) = rest.strip_prefix('{').and_then(|r| r.strip_suffix('}')) else {
        return (latex, None);
    };
    let note = rest.trim();
    if note.is_empty() || note.contains('{') {
        return (latex, None);
    }
    let head = body[..open].trim_end();
    let head = head.strip_suffix(r"\quad").unwrap_or(head).trim_end();
    if head.is_empty() {
        return (latex, None);
    }
    (head, Some(note))
}

#[cfg(feature = "speech")]
mod engine {
    use std::sync::mpsc::{channel, Sender};
    use std::sync::OnceLock;

    /// One formula to speak and where to send its styles.
    struct Job {
        mathml: String,
        reply: Sender<Option<super::SpeechStyles>>,
    }

    /// MathCAT keeps its rules, its preferences and the parsed expression in
    /// `thread_local` state, so the thread that loaded the rules is the only
    /// one that has them — and a server hands each request to whichever
    /// thread its pool has free. (This showed as a test that failed only
    /// when run beside another MathCAT test: the second thread to arrive
    /// had no rules.) So one worker thread owns MathCAT for the process:
    /// callers queue a job and wait for the answer, which also serializes
    /// the engine's single-expression state. Formula verbalization is not on
    /// any hot path — a page has a handful of formulas, each of which cost a
    /// VLM decode to produce — so the round trip is noise.
    ///
    /// `None` when the rules directory is missing or refused: speech
    /// degrades to no `speech` field rather than failing the conversion,
    /// like every other optional asset.
    static WORKER: OnceLock<Option<Sender<Job>>> = OnceLock::new();

    fn worker() -> Option<&'static Sender<Job>> {
        WORKER
            .get_or_init(|| {
                let Some(dir) = rules_dir() else {
                    crate::debug_log!(
                        "speech: MathCAT rules not found (.models/mathcat); \
                         formula items will carry no `speech` field"
                    );
                    return None;
                };
                let (tx, rx) = channel::<Job>();
                let (ready_tx, ready_rx) = channel::<bool>();
                std::thread::Builder::new()
                    .name("docling-speech".into())
                    .spawn(move || {
                        let dir_str = dir.to_string_lossy().into_owned();
                        let ok = match libmathcat::interface::set_rules_dir(dir_str) {
                            Ok(()) => true,
                            Err(e) => {
                                crate::debug_log!("speech: MathCAT rules {}: {e}", dir.display());
                                false
                            }
                        };
                        let _ = ready_tx.send(ok);
                        if !ok {
                            return;
                        }
                        for job in rx {
                            let _ = job.reply.send(speak_on_this_thread(job.mathml));
                        }
                    })
                    .ok()?;
                ready_rx.recv().ok().filter(|&ok| ok).map(|_| tx)
            })
            .as_ref()
    }

    /// `.models/mathcat/`, resolved CWD-relative then against
    /// `DOCLING_RS_MODELS_DIR`, matching how every other runtime asset is found.
    fn rules_dir() -> Option<std::path::PathBuf> {
        if let Some(over) = crate::env::nonempty("DOCLING_RS_MATHCAT_RULES") {
            let p = std::path::PathBuf::from(over);
            return p.is_dir().then_some(p);
        }
        let cwd = std::path::PathBuf::from(".models/mathcat");
        if cwd.is_dir() {
            return Some(cwd);
        }
        let base = crate::env::nonempty("DOCLING_RS_MODELS_DIR")?;
        let p = std::path::Path::new(&base).join(".models/mathcat");
        p.is_dir().then_some(p)
    }

    /// MathCAT's speech styles, as the JSON `speech` map keys them (lower
    /// case) and as MathCAT's `SpeechStyle` preference names them. Every
    /// style is produced for every formula: verbalization costs microseconds
    /// where the formula cost a VLM decode, and the style is a reader
    /// setting — shipping one would make flipping it re-run the conversion.
    pub(super) const STYLES: &[(&str, &str)] =
        &[("clearspeak", "ClearSpeak"), ("simplespeak", "SimpleSpeak")];

    /// Speak one MathML string in every style, or `None` if MathCAT is
    /// unavailable or refuses it. Never propagates an error: a formula we
    /// cannot voice is a formula the reader announces as "formula", not a
    /// failed conversion.
    pub(super) fn speak(mathml: String) -> Option<super::SpeechStyles> {
        let (reply, rx) = channel();
        worker()?.send(Job { mathml, reply }).ok()?;
        rx.recv().ok()?
    }

    /// The engine call itself; only ever runs on the worker thread.
    fn speak_on_this_thread(mathml: String) -> Option<super::SpeechStyles> {
        libmathcat::interface::set_mathml(mathml).ok()?;
        let mut out = super::SpeechStyles::new();
        for (key, style) in STYLES {
            let set = libmathcat::interface::set_preference(
                String::from("SpeechStyle"),
                String::from(*style),
            );
            if set.is_err() {
                continue;
            }
            let s = libmathcat::interface::get_spoken_text()
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                // MathCAT reports an unparseable subexpression *inside* the
                // speech rather than failing, so the string can contain
                // "[PARSE ERROR: Undefined(...)]". A reader would narrate
                // the error text aloud. Treat it as no spoken form.
                .filter(|s| !s.contains("PARSE ERROR"))
                // A command the converter does not know is passed through
                // as its own text (`\end{equation}` from a recognizer that
                // emitted a stray environment, `\foo`), and a reader would
                // then pronounce LaTeX. No formula speaks a backslash.
                .filter(|s| !s.contains('\\'))?;
            out.insert((*key).to_string(), s);
        }
        (!out.is_empty()).then_some(out)
    }
}

/// A formula's spoken forms by style — `clearspeak`, `simplespeak` — the
/// JSON item's `speech` map. The reader picks the style; a style missing
/// from the map falls back to any present one on the client.
pub type SpeechStyles = std::collections::BTreeMap<String, String>;

/// The spoken forms of a formula's LaTeX, or `None` when they cannot be
/// produced.
///
/// `None` is the honest answer in several cases and they are all treated alike:
/// the formula was never decoded (empty LaTeX), the LaTeX does not parse, or
/// the MathCAT rules are not installed. **The caller must not substitute the
/// raw extraction.** On a scanned page a formula's `orig` is whatever the OCR
/// layer produced, which on real scans is routinely unreadable
/// (`'Jl X ^ =: -·''^" * » 5X0-'`); narrating that is worse than saying
/// "formula" and moving on.
#[cfg(feature = "speech")]
pub fn speak_latex(latex: &str) -> Option<SpeechStyles> {
    let latex = latex.trim();
    if latex.is_empty() {
        return None;
    }
    let mut styles: Option<SpeechStyles> = None;
    let append = |styles: &mut Option<SpeechStyles>, part: SpeechStyles| match styles {
        None => *styles = Some(part),
        Some(acc) => {
            for (k, v) in part {
                let slot = acc.entry(k).or_default();
                if !slot.is_empty() {
                    slot.push_str(", ");
                }
                slot.push_str(&v);
            }
        }
    };
    let words = |note: &str| -> SpeechStyles {
        engine::STYLES
            .iter()
            .map(|(k, _)| ((*k).to_string(), note.to_string()))
            .collect()
    };
    for seg in segments(latex) {
        let (label, seg) = leading_label(seg);
        if let Some(l) = label {
            append(&mut styles, words(&l));
        }
        if let Some(note) = words_only(seg) {
            append(&mut styles, words(note));
            continue;
        }
        let (head, note) = split_annotation(seg);
        append(&mut styles, speak_expression(head)?);
        if let Some(n) = note {
            append(&mut styles, words(n));
        }
    }
    styles
}

/// One mathematical (or chemical) expression through the rewrite chain and
/// MathCAT.
#[cfg(feature = "speech")]
fn speak_expression(latex: &str) -> Option<SpeechStyles> {
    let latex = chemical_arrows(latex);
    let mut tight = bare_states(&normalize_latex(&latex));
    if is_chemistry(&tight) {
        tight = group_elements(&unwrap_upright(&tight));
    }
    tight = unfold_extensible_arrows(&tight);
    let mathml = crate::mathml::to_xml(&tight).ok()?;
    engine::speak(mathml)
}

/// Without the `speech` feature there is no spoken form, and the JSON export
/// stays byte-identical.
#[cfg(not(feature = "speech"))]
pub fn speak_latex(_latex: &str) -> Option<SpeechStyles> {
    None
}

#[cfg(test)]
mod tests {
    use super::normalize_latex;

    #[test]
    fn splits_a_leading_word_label() {
        use super::leading_label;
        assert_eq!(
            leading_label(r"\mathrm {Neto}: \mathrm {Al}^{3+}"),
            (Some("Neto".to_string()), r" \mathrm {Al}^{3+}")
        );
        assert_eq!(
            leading_label(r"\text{Overall:} x"),
            (Some("Overall".to_string()), " x")
        );
        assert_eq!(
            leading_label(r"\mathrm {N e t}: x"),
            (Some("Net".to_string()), " x")
        );
        assert_eq!(leading_label(r"\mathrm{Al}: x").0, None); // two letters: a symbol, not a word
        assert_eq!(leading_label(r"\mathrm{Neto} x").0, None); // no colon
    }

    #[test]
    fn unwraps_upright_element_groups_only() {
        use super::unwrap_upright;
        assert_eq!(
            unwrap_upright(r"\mathrm{CH}_{3}\mathrm {C O O H}(aq)"),
            "CH_{3}COOH(aq)"
        );
        assert_eq!(
            unwrap_upright(r"\mathrm{(aq)} + \mathrm{}"),
            r"\mathrm{(aq)} + \mathrm{}"
        );
        assert_eq!(unwrap_upright(r"\mathrm"), r"\mathrm");
        assert_eq!(unwrap_upright(r"\to\mathrm{CH}_{4}"), r"\to CH_{4}");
        assert_eq!(unwrap_upright(r"x^{2}\mathrm{CO}"), r"x^{2}CO");
    }

    /// The whole reason this step exists: a recognizer emits one token at a
    /// time, so consecutive digits arrive separated and would parse as separate
    /// numbers ("one two" for 12, five nodes for 100.5).
    #[test]
    fn rejoins_multi_digit_numbers() {
        assert_eq!(normalize_latex("a ^ { 2 } + 8 = 1 2"), "a^{2}+8=12");
        assert_eq!(normalize_latex("x = 1 0 0 . 5"), "x=100.5");
    }

    /// A control word keeps the space that terminates it when a letter follows,
    /// and loses it otherwise — `\cdotE` is a different (undefined) command,
    /// while `\rightarrow2` is unambiguous.
    #[test]
    fn keeps_only_the_space_that_terminates_a_control_word() {
        assert_eq!(
            normalize_latex(r"\nabla \cdot E = \frac { \rho } { \varepsilon _ { 0 } }"),
            r"\nabla\cdot E=\frac{\rho}{\varepsilon_{0}}"
        );
        assert_eq!(
            normalize_latex(r"2 H _ { 2 } + O _ { 2 } \rightarrow 2 H _ { 2 } O"),
            r"2H_{2}+O_{2}\rightarrow2H_{2}O"
        );
    }

    /// Structure survives: fractions, scripts and integral limits are unchanged
    /// apart from the removed spacing.
    #[test]
    fn preserves_structure() {
        assert_eq!(
            normalize_latex(r"\int _ { 0 } ^ { \infty } e ^ { - x } d x = 1"),
            r"\int_{0}^{\infty}e^{-x}dx=1"
        );
        assert_eq!(
            normalize_latex(r"9 \frac { 1 } { 2 } - 6 \frac { 1 } { 4 }"),
            r"9\frac{1}{2}-6\frac{1}{4}"
        );
    }

    /// Already-tight LaTeX (any other backend's) passes through untouched.
    #[test]
    fn tight_latex_is_unchanged() {
        for s in [r"a^{2}+8=12", r"\frac{1}{2}", r"E=mc^{2}"] {
            assert_eq!(normalize_latex(s), s);
        }
    }

    /// A textbook's `(balanced)` / `(unbalanced)` annotation comes off the
    /// equation so the maths can be recognised as chemistry, and is spoken
    /// after it rather than dropped.
    #[test]
    fn a_trailing_prose_annotation_is_split_off() {
        let (head, note) = super::split_annotation(
            r"2 H _ { 2 } O \longrightarrow 2 H _ { 2 } + O _ { 2 } \quad ( \text {balanced} )",
        );
        assert_eq!(
            head,
            r"2 H _ { 2 } O \longrightarrow 2 H _ { 2 } + O _ { 2 }"
        );
        assert_eq!(note, Some("balanced"));
    }

    /// Ordinary parenthesised maths is never mistaken for an annotation.
    #[test]
    fn real_parentheses_survive() {
        for s in [
            r"y = ( \tan \theta _ { 0 } ) x",
            r"a = ( b + c )",
            r"R = \frac { v ^ { 2 } } { g } ( 2 )",
            r"x = 1",
        ] {
            let (head, note) = super::split_annotation(s);
            assert_eq!(head, s, "unchanged: {s}");
            assert_eq!(note, None);
        }
    }

    /// A lone backslash is the control-space command; its space stays.
    #[test]
    fn control_space_survives() {
        assert_eq!(
            normalize_latex(r"- 2 8 5 . 8 \ \mathrm { k J }"),
            r"-285.8\ \mathrm{kJ}"
        );
    }

    /// The chemistry gate: what a mathematician never writes.
    #[test]
    fn chemistry_is_recognized_only_on_unambiguous_evidence() {
        for chem in [
            r"N a ^ { + } ( a q )",
            r"M g ^ { 2 + }",
            r"C u ^ { + 2 }",
            r"N _ { 2 } + 3 H _ { 2 } \rightleftharpoons 2 N H _ { 3 }",
            r"2 H _ { 2 } ( g ) + O _ { 2 } ( g ) \longrightarrow 2 H _ { 2 } O ( l )",
        ] {
            assert!(super::is_chemistry(chem), "chemistry: {chem}");
        }
        for math in [
            r"e ^ { - x }",
            r"k T e ^ { - E / k T }",
            r"F ( s ) = \int f ( t ) e ^ { - s t } d t",
            r"f ( g ) + h ( l )",
            r"P a = N / m ^ { 2 }",
        ] {
            assert!(!super::is_chemistry(math), "mathematics: {math}");
        }
    }

    /// Two-letter symbols become one group so a charge attaches to the whole
    /// symbol; control words and prose are never touched.
    #[test]
    fn two_letter_elements_are_grouped() {
        assert_eq!(
            super::group_elements(r"Mg(OH)_{2}(s)\rightleftharpoons Mg^{2+}(aq)+2OH^{-}(aq)"),
            r"{Mg}(OH)_{2}(s)\rightleftharpoons {Mg}^{2+}(aq)+2OH^{-}(aq)"
        );
        assert_eq!(super::group_elements(r"NaCl(s)"), r"{Na}{Cl}(s)");
        // `\Gamma`, `\Sigma`, `\Lambda` contain Ga, Si, La.
        assert_eq!(
            super::group_elements(r"\Gamma\Sigma\Lambda(aq)"),
            r"\Gamma\Sigma\Lambda(aq)"
        );
        assert_eq!(
            super::group_elements(r"\text{Natural He}(aq)"),
            r"\text{Natural He}(aq)"
        );
        // A lowercase run is a word, not a symbol.
        assert_eq!(super::group_elements(r"Heat(aq)"), r"Heat(aq)");
    }

    /// An extensible arrow becomes a plain arrow with its condition above
    /// (and below), in both forms.
    #[test]
    fn extensible_arrows_unfold() {
        assert_eq!(
            super::unfold_extensible_arrows(r"A\xrightarrow{\Delta}B"),
            r"A\overset{\Delta}{\longrightarrow}B"
        );
        assert_eq!(
            super::unfold_extensible_arrows(r"A\xrightarrow[cat]{\Delta}B"),
            r"A\underset{cat}{\overset{\Delta}{\longrightarrow}}B"
        );
        assert_eq!(
            super::unfold_extensible_arrows(r"A\xleftarrow{h\nu}B"),
            r"A\overset{h\nu}{\longleftarrow}B"
        );
        // Not an extensible arrow; nothing to do.
        assert_eq!(super::unfold_extensible_arrows(r"\xi+x"), r"\xi+x");
    }

    /// State marks come out of `\text{…}`.
    #[test]
    fn state_marks_leave_text() {
        assert_eq!(
            super::bare_states(r"Mg^{2+}\text{(aq)}+2OH^{-}\text{(aq)}"),
            r"Mg^{2+}(aq)+2OH^{-}(aq)"
        );
    }

    /// A formula splits at top-level `\quad` gaps only.
    #[test]
    fn segments_split_at_top_level_quad_only() {
        assert_eq!(
            super::segments(r"A \longrightarrow B \quad \Delta H = - 8 9 0"),
            vec![r"A \longrightarrow B", r"\Delta H = - 8 9 0"]
        );
        assert_eq!(
            super::segments(r"\frac { a \quad b } { c } \qquad d"),
            vec![r"\frac { a \quad b } { c }", "d"]
        );
        assert_eq!(super::segments("x = 1"), vec!["x = 1"]);
        assert_eq!(super::words_only(r"( \text {balanced} )"), Some("balanced"));
        assert_eq!(super::words_only(r"\text { and }"), Some("and"));
        assert_eq!(super::words_only(r"\text{a}+b"), None);
    }

    /// A recognizer reaches for the arrow that looks right and gets the meaning
    /// wrong, so a reaction is narrated as "implies" and an equilibrium as "if
    /// and only if". Rewritten only where the chemistry is unambiguous.
    #[test]
    fn logic_arrows_become_chemical_ones_only_in_chemistry() {
        let chem = super::chemical_arrows(
            r"O H ^ { - } ( a q ) + H ^ { + } ( a q ) \Longleftrightarrow H _ { 2 } O",
        );
        assert!(chem.contains(r"\rightleftharpoons"), "{chem}");

        let reaction =
            super::chemical_arrows(r"Z n ( s ) + C u ^ { + 2 } \Longrightarrow Z n ^ { + 2 }");
        assert!(reaction.contains(r"\longrightarrow"), "{reaction}");

        // Mathematics means what it says: implication and equivalence stay.
        for m in [
            r"x > 0 \Longrightarrow x ^ { 2 } > 0",
            r"P \Leftrightarrow Q",
            // No state symbol and no charge: genuinely ambiguous, left alone.
            r"a A + b B \Longleftrightarrow e E + f F",
        ] {
            assert_eq!(super::chemical_arrows(m), m, "unchanged: {m}");
        }
    }

    /// A formula that was never decoded has no spoken form — and, critically,
    /// the caller gets `None` rather than a fallback to the raw extraction.
    #[test]
    fn undecoded_formula_has_no_spoken_form() {
        assert_eq!(super::speak_latex(""), None);
        assert_eq!(super::speak_latex("   "), None);
    }

    /// Through MathCAT, when its rules are installed at the repo root. Each
    /// case is a textbook formula that was spoken wrong or not at all before
    /// the rewrite it exercises.
    #[cfg(feature = "speech")]
    #[test]
    fn textbook_formulas_speak_as_intended() {
        // Tests run with CWD = the crate dir; the rules live at the repo root.
        let rules = concat!(env!("CARGO_MANIFEST_DIR"), "/../../.models/mathcat");
        if !std::path::Path::new(rules).is_dir() {
            eprintln!("skipping: {rules} is not present");
            return;
        }
        std::env::set_var("DOCLING_RS_MATHCAT_RULES", rules);
        let clear = |latex: &str| -> String {
            super::speak_latex(latex)
                .unwrap_or_else(|| panic!("no spoken form for {latex}"))
                .remove("clearspeak")
                .expect("clearspeak style")
        };
        let has = |spoken: &str, parts: &[&str]| {
            for p in parts {
                assert!(spoken.contains(p), "{p:?} missing from {spoken:?}");
            }
        };

        // Two-letter element with a charge: an equilibrium, not harpoons.
        let s = clear(
            r"M g ( O H ) _ { 2 } ( s ) \rightleftharpoons M g ^ { 2 + } ( a q ) + 2 O H ^ { - } ( a q )",
        );
        has(
            &s,
            &[
                "cap m g",
                "solid",
                "is in equilibrium with",
                "super 2 plus",
                "aqueous",
            ],
        );
        assert!(!s.contains("harpoon"), "{s}");

        // Upright multi-letter groups before a state (GLM-OCR sets formulas in
        // \mathrm): the state is a state, not an argument, and the equation stays
        // chemistry.
        let s = clear(
            r"\mathrm {C H} _ {3} \mathrm {C O O H} (a q) + \mathrm {H} _ {2} \mathrm {O} (l) \rightleftharpoons \mathrm {C H} _ {3} \mathrm {C O O} ^ {-} (a q) + \mathrm {H} _ {3} \mathrm {O} ^ {+} (a q)",
        );
        has(
            &s,
            &["aqueous", "liquid", "is in equilibrium with", "super minus"],
        );
        assert!(!s.contains("harpoon") && !s.contains("times"), "{s}");
        let s = clear(
            r"\mathrm {Neto}: \mathrm {Al} (\mathrm {OH}) _ {3} (s) + \mathrm {OH} ^ {-} (a q) \rightleftharpoons \mathrm {Al} (\mathrm {OH}) _ {4} ^ {-} (a q)",
        );
        has(&s, &["Neto", "solid", "is in equilibrium with", "aqueous"]);
        assert!(!s.contains("harpoon"), "{s}");
        let s = clear(r"\mathrm{HCl}(aq) \rightarrow \mathrm{H}^{+}(aq) + \mathrm{Cl}^{-}(aq)");
        has(&s, &["cap c l", "reacts to form", "aqueous"]);

        // Stacked scripts: the charge belongs to the ion, not to the 4.
        has(
            &clear(r"S O _ { 4 } ^ { 2 - } ( a q )"),
            &["sub 4", "2 minus", "aqueous"],
        );

        // An annotation after `\quad` must not demote the equation to maths.
        let s = clear(
            r"C H _ { 4 } ( g ) + 2 O _ { 2 } ( g ) \longrightarrow C O _ { 2 } ( g ) + 2 H _ { 2 } O ( l ) \quad \Delta H = - 8 9 0 \mathrm { k J }",
        );
        has(&s, &["reacts to form", "cap delta", "890"]);

        // A condition over the arrow.
        has(
            &clear(
                r"2 K C l O _ { 3 } ( s ) \xrightarrow { \Delta } 2 K C l ( s ) + 3 O _ { 2 } ( g )",
            ),
            &["reacts to form", "cap delta"],
        );

        // Nuclide notation and a decay.
        has(
            &clear(
                r"{ } _ { 9 2 } ^ { 2 3 8 } U \longrightarrow { } _ { 9 0 } ^ { 2 3 4 } T h + { } _ { 2 } ^ { 4 } H e",
            ),
            &["238", "92", "cap u", "reacts to form"],
        );

        // Mathematics the previous converter could not voice at all.
        has(&clear("k < n"), &["less than"]);
        has(
            &clear(r"f ^ { \prime } ( x ) = 3 x ^ { 2 }"),
            &["prime", "squared"],
        );
        has(&clear(r"\vec { F } = m \vec { a }"), &["vector"]);
        has(
            &clear(r"\Delta H ^ { \circ } = - 2 8 5 . 8 \ \mathrm { k J / m o l }"),
            &["285.8"],
        );

        // Mathematics stays mathematics.
        let s = clear(r"\Gamma ( n ) = ( n - 1 ) !");
        has(&s, &["cap gamma", "factorial"]);
        has(&clear(r"P a = N / m ^ { 2 }"), &["cap p eigh"]);

        // Recognizer junk is silence, not spoken LaTeX.
        assert_eq!(super::speak_latex(r"\end{equation}"), None);
    }
}
