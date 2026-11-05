//! Spoken-form rendering of formula items: LaTeX → MathML → English.
//!
//! A screen reader speaks a formula; a TTS reader has to as well, and neither
//! can do anything useful with `$$a^{2}+8=12$$`. This module turns the LaTeX a
//! [`Node::Formula`](crate::Node::Formula) carries into a sentence
//! ("eigh squared plus 8, is equal to 12") that lands on the JSON item as the
//! additive `spoken` field.
//!
//! The chain is pure Rust, which is why it lives server-side rather than in a
//! client:
//!
//! 1. [`normalize_latex`] rejoins the space-separated tokens the CodeFormula
//!    model emits — the step without which every multi-digit number breaks.
//! 2. `latex2mathml` parses that into MathML. Chosen over `pulldown-latex`,
//!    which mis-parses `\int_{0}^{\infty}` into a `<msub>` plus an `<msup>`
//!    over an empty base instead of one `<msubsup>`.
//! 3. MathCAT (the DAISY engine behind NVDA and JAWS) speaks the MathML. Its
//!    chemistry heuristics fire on their own — `\rightarrow` in a chemical
//!    equation reads as "reacts to form", not "right arrow" — so chemistry
//!    needs no separate path.
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
/// `\cdotE`). So a space survives only in that one case — between a control
/// word and a following letter. `\rightarrow 2` needs no space because a digit
/// cannot extend the command name.
pub fn normalize_latex(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut chars = src.chars().peekable();
    while let Some(c) = chars.next() {
        if c != ' ' {
            out.push(c);
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
    let flat: String = latex.chars().filter(|c| !c.is_whitespace()).collect();
    let charged = flat.contains("^{+") || flat.contains("^{-");
    let state = ["(aq)", "(s)", "(g)", "(l)"]
        .iter()
        .any(|m| flat.contains(m));
    if !(charged || state) {
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
    use std::sync::Mutex;

    /// MathCAT keeps its parsed expression in `thread_local` state, so one call
    /// per process at a time. Formula verbalization is not on any hot path (a
    /// page has a handful of formulas, each of which cost a VLM decode to
    /// produce), so a mutex is the right trade against per-thread rule loading.
    static LOCK: Mutex<()> = Mutex::new(());

    /// Point MathCAT at its rules directory, once. Returns whether speech is
    /// available — a missing rules directory degrades to no `spoken` field
    /// rather than failing the conversion, like every other optional asset.
    fn ready() -> bool {
        use std::sync::OnceLock;
        static READY: OnceLock<bool> = OnceLock::new();
        *READY.get_or_init(|| {
            let Some(dir) = rules_dir() else {
                crate::debug_log!(
                    "speech: MathCAT rules not found (.models/mathcat); \
                     formula items will carry no `spoken` field"
                );
                return false;
            };
            match libmathcat::interface::set_rules_dir(dir.to_string_lossy().into_owned()) {
                Ok(()) => true,
                Err(e) => {
                    crate::debug_log!("speech: MathCAT rules {}: {e}", dir.display());
                    false
                }
            }
        })
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

    /// Speak one MathML string, or `None` if MathCAT is unavailable or refuses
    /// it. Never propagates an error: a formula we cannot voice is a formula
    /// the reader announces as "formula", not a failed conversion.
    pub(super) fn speak(mathml: String) -> Option<String> {
        if !ready() {
            return None;
        }
        let _guard = LOCK.lock().ok()?;
        libmathcat::interface::set_mathml(mathml)
            .ok()
            .and_then(|_| libmathcat::interface::get_spoken_text().ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            // MathCAT reports an unparseable subexpression *inside* the speech
            // rather than failing, so the string can contain
            // "[PARSE ERROR: Undefined(...)]". A recognizer emitting a
            // malformed command (`\right leftrightarrow` for
            // `\rightleftharpoons`) produces exactly that, and a reader would
            // narrate the error text aloud. Treat it as no spoken form.
            .filter(|s| !s.contains("PARSE ERROR"))
    }
}

/// The spoken form of a formula's LaTeX, or `None` when it cannot be produced.
///
/// `None` is the honest answer in several cases and they are all treated alike:
/// the formula was never decoded (empty LaTeX), the LaTeX does not parse, or
/// the MathCAT rules are not installed. **The caller must not substitute the
/// raw extraction.** On a scanned page a formula's `orig` is whatever the OCR
/// layer produced, which on real scans is routinely unreadable
/// (`'Jl X ^ =: -·''^" * » 5X0-'`); narrating that is worse than saying
/// "formula" and moving on.
#[cfg(feature = "speech")]
pub fn speak_latex(latex: &str) -> Option<String> {
    let latex = latex.trim();
    if latex.is_empty() {
        return None;
    }
    let (head, note) = split_annotation(latex);
    let head = chemical_arrows(head);
    let normalized = normalize_latex(&head);
    let mathml =
        latex2mathml::latex_to_mathml(&normalized, latex2mathml::DisplayStyle::Block).ok()?;
    let spoken = engine::speak(mathml)?;
    Some(match note {
        Some(n) => format!("{spoken}, {n}"),
        None => spoken,
    })
}

/// Without the `speech` feature there is no spoken form, and the JSON export
/// stays byte-identical.
#[cfg(not(feature = "speech"))]
pub fn speak_latex(_latex: &str) -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::normalize_latex;

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
}
