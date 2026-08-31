//! OCR for scanned pages, via the PP-OCRv3 recognition model (CRNN/SVTR) run
//! with `ort`. The layout model already locates text regions on the page image
//! (it works without a text layer), so OCR only needs *recognition*: each text
//! region is cropped, split into lines by horizontal projection, and each line
//! is recognised and decoded with CTC — producing [`TextCell`]s the normal
//! layout assembly then consumes. This avoids a separate text-detection model.

use image::RgbImage;
use ort::session::Session;
use ort::value::Tensor;

use crate::layout::Region;
// The ONNX-free half (line prep, batching, CTC decode) lives in `ocr_prep`
// so the wasm build shares it verbatim (#79 phase 2).
use crate::ocr_prep::{
    batch_input, decode_row_scored, dict_chars, prep_region_lines, prep_table_words, width_batches,
    PrepLine, REC_HEIGHT,
};
use crate::pdfium_backend::TextCell;

/// One recognized line: its decoded text, the mean per-character confidence,
/// and the CTC timestep that emitted each character (what `line_words` turns
/// into per-word boxes).
type PositionedLine = (String, f32, Vec<usize>);

/// One page's OCR result: the line cells with their confidences, plus the
/// word-level cells recovered from the same recognition run.
type PageOcr = (Vec<(TextCell, f32)>, Vec<TextCell>);

pub struct OcrModel {
    rec: Session,
    /// DBNet text detector, loaded on first use — only the fallback path needs
    /// it, and a build without the asset must still convert.
    det: Option<Session>,
    det_tried: bool,
    /// CTC classes: index 0 = blank, 1..=6623 = dictionary, 6624 = space.
    chars: Vec<String>,
}

/// OCR recognition language: which PP-OCRv3 model + dictionary pair runs.
///
/// The default is **English** (`.models/ocr_rec_en.onnx` + `.models/en_dict.txt`):
/// the multilingual `ch_` model reads Latin scripts with badly degraded word
/// spacing (glued words on ordinary English scans), which is the common
/// real-world case. `Ch` selects the `ch_` pair (`.models/ocr_rec.onnx` +
/// `.models/ppocr_keys_v1.txt`) — that is what upstream docling conformance is
/// measured with, and `scripts/conformance/pdf_*.sh` pin it explicitly (by
/// path, which wins over this selector).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OcrLang {
    /// en_PP-OCRv3 — English-only, proper Latin word spacing.
    #[default]
    En,
    /// ch_PP-OCRv3 — multilingual; the docling-conformance model.
    Ch,
}

impl OcrLang {
    /// Parse a user-supplied language id. `None` for anything but `en`/`ch`
    /// (trimmed, case-insensitive) — callers surface their own error/warning.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "en" => Some(Self::En),
            "ch" => Some(Self::Ch),
            _ => None,
        }
    }

    /// The process-level choice from `DOCLING_RS_OCR_LANG` (empty/unset → the
    /// English default; unknown values warn and use English).
    pub fn from_env() -> Self {
        let Some(raw) = docling_core::env::nonempty("DOCLING_RS_OCR_LANG") else {
            return Self::default();
        };
        Self::parse(&raw).unwrap_or_else(|| {
            eprintln!("docling-pdf: DOCLING_RS_OCR_LANG={raw:?} is not en|ch; using en");
            Self::default()
        })
    }
}

/// Which document regions feed the OCR — docling 2.116's `OcrMode` (#254,
/// upstream docling#3710). Upstream restructured its pipeline so OCR runs
/// *after* layout, on layout regions filtered by the PDF text layer — the
/// architecture this port has always had — and named the strategies:
///
/// - `PdfAwareLayoutRegions` (upstream's **default**): OCR only layout regions
///   the embedded text layer can't cover. Exactly the standard path here —
///   scanned pages OCR their regions, digital pages OCR only text-less bitmap
///   areas.
/// - `FullPage` / `LayoutRegions`: ignore the PDF text layer and OCR
///   everything. Both map onto the [`force_full_page_ocr`] machinery (discard
///   the text layer, OCR every layout region): the upstream distinction —
///   whole-page vs per-region *detector* input — has no analogue in this
///   engine, whose PP-OCR recognizer always consumes per-region line crops.
///
/// [`force_full_page_ocr`]: crate::Pipeline::force_full_page_ocr
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OcrMode {
    /// Upstream's `default`: currently wired to `PdfAwareLayoutRegions`.
    #[default]
    Default,
    /// OCR the full page, text layer ignored (docling's `full_page`; the
    /// mode-shaped spelling of `force_full_page_ocr`).
    FullPage,
    /// OCR every layout region, text layer ignored (docling's
    /// `layout_regions`).
    LayoutRegions,
    /// OCR layout regions the text layer can't cover (docling's
    /// `pdf_aware_layout_regions` — the default behavior).
    PdfAwareLayoutRegions,
}

impl OcrMode {
    /// Parse docling's mode ids. `None` for anything else — callers surface
    /// their own error/warning.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "default" => Some(Self::Default),
            "full_page" => Some(Self::FullPage),
            "layout_regions" => Some(Self::LayoutRegions),
            "pdf_aware_layout_regions" => Some(Self::PdfAwareLayoutRegions),
            _ => None,
        }
    }

    /// The process-level choice from `DOCLING_RS_OCR_MODE` (empty/unset → the
    /// default; unknown values warn and use the default).
    pub fn from_env() -> Self {
        let Some(raw) = docling_core::env::nonempty("DOCLING_RS_OCR_MODE") else {
            return Self::default();
        };
        Self::parse(&raw).unwrap_or_else(|| {
            eprintln!(
                "docling-pdf: DOCLING_RS_OCR_MODE={raw:?} is not \
                 default|full_page|layout_regions|pdf_aware_layout_regions; using default"
            );
            Self::default()
        })
    }

    /// Whether this mode discards the embedded text layer — the engine truth
    /// both non-default modes reduce to.
    pub fn forces_full_page(self) -> bool {
        matches!(self, Self::FullPage | Self::LayoutRegions)
    }
}

/// The process-level OCR render scale from `DOCLING_RS_OCR_SCALE` (#254,
/// upstream docling#3877's `OcrOptions.scale`): pixels per PDF point fed to
/// the recognizer. Unset/empty → `None` (OCR reads the pipeline's own page
/// render, 2.0 px/pt); non-positive or unparsable values warn and are ignored.
pub fn scale_from_env() -> Option<f32> {
    let raw = docling_core::env::nonempty("DOCLING_RS_OCR_SCALE")?;
    match raw.parse::<f32>() {
        Ok(s) if s > 0.0 && s.is_finite() => Some(s),
        _ => {
            eprintln!(
                "docling-pdf: DOCLING_RS_OCR_SCALE={raw:?} is not a positive number; ignored"
            );
            None
        }
    }
}

/// Resolve the recognition model + dictionary pair for `lang`. An English
/// default that isn't on disk (older model checkouts) degrades to the `ch_`
/// pair with a warning rather than failing — the usual missing-optional-asset
/// convention. Explicit `DOCLING_OCR_REC_ONNX` / `DOCLING_OCR_DICT` paths win
/// over all of this; they are a pair, so set both together.
pub(crate) fn resolve_rec_pair(lang: OcrLang) -> (String, String) {
    const CH: (&str, &str) = (".models/ocr_rec.onnx", ".models/ppocr_keys_v1.txt");
    const EN: (&str, &str) = (".models/ocr_rec_en.onnx", ".models/en_dict.txt");
    // PP-OCRv6 small, preferred for the non-`ch` selection when installed. Its
    // ~18.7k-character vocabulary makes it multilingual rather than English-only,
    // so it supersedes the `en_` pair rather than sitting beside it; the older
    // model stays the fallback so existing `.models/` trees keep working.
    //
    // Measured against an independent engine over 18 pages of a scanned book it
    // is a wash on prose and *behind* the v3 English pair on symbol-dense text,
    // where it abstains on glyphs it cannot identify instead of guessing. It wins
    // decisively on display type — covers and plates, the pages the detector path
    // exists for — reads word spacing far more reliably, and is faster despite
    // being larger (most of its size is the output projection for that
    // vocabulary, one wide matmul, not deeper computation).
    const V6: (&str, &str) = (".models/ocr_rec_v6.onnx", ".models/v6_dict.txt");
    let want_ch = lang == OcrLang::Ch;
    let have = |p: &(&str, &str)| {
        std::path::Path::new(&crate::resolve_asset(p.0)).exists()
            && std::path::Path::new(&crate::resolve_asset(p.1)).exists()
    };
    let pick = if want_ch {
        CH
    } else if have(&V6) {
        V6
    } else {
        EN
    };
    let (mut rec, mut dict) = (crate::resolve_asset(pick.0), crate::resolve_asset(pick.1));
    if !want_ch && (!std::path::Path::new(&rec).exists() || !std::path::Path::new(&dict).exists()) {
        let (ch_rec, ch_dict) = (crate::resolve_asset(CH.0), crate::resolve_asset(CH.1));
        if std::path::Path::new(&ch_rec).exists() && std::path::Path::new(&ch_dict).exists() {
            eprintln!(
                "docling-pdf: English OCR model not found ({rec}); falling back to the \
                 multilingual ch_ model — expect weak Latin word spacing. Fetch it with \
                 scripts/install/download_dependencies.sh"
            );
            (rec, dict) = (ch_rec, ch_dict);
        }
    }
    (
        docling_core::env::nonempty("DOCLING_OCR_REC_ONNX").unwrap_or(rec),
        docling_core::env::nonempty("DOCLING_OCR_DICT").unwrap_or(dict),
    )
}

impl OcrModel {
    /// Load the recognition model and its character dictionary for `lang` —
    /// see [`resolve_rec_pair`] for the selection rules (explicit
    /// `DOCLING_OCR_REC_ONNX`/`DOCLING_OCR_DICT` paths win).
    pub fn load(lang: OcrLang) -> Result<Self, String> {
        let (rec_path, dict_path) = resolve_rec_pair(lang);
        // Single-threaded: ORT's multi-threaded float-reduction order varies
        // across runs, which flips the CTC argmax on low-confidence characters
        // (e.g. noisy faxes) and makes the snapshot output non-deterministic. The
        // recognition inputs are tiny per-line crops, so the throughput cost is
        // negligible.
        let builder = Session::builder()
            .map_err(|e| format!("ocr: builder: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| format!("ocr: intra_threads: {e}"))?;
        let rec = docling_onnx::apply(builder)
            .map_err(|e| format!("ocr: {e}"))?
            .commit_from_file(&rec_path)
            .map_err(|e| format!("ocr: load {rec_path}: {e}"))?;
        let dict = std::fs::read_to_string(&dict_path)
            .map_err(|e| format!("ocr: read dict {dict_path}: {e}"))?;
        Ok(Self {
            rec,
            det: None,
            det_tried: false,
            chars: dict_chars(&dict),
        })
    }

    /// Recognise a batch of prepared *same-width* lines in one session run.
    ///
    /// Only equal widths ever share a run: same-width batching is
    /// bit-identical to one-at-a-time recognition (each sample keeps its own
    /// data and per-sample kernel reduction order — verified empirically on
    /// the scanned corpus), whereas width-padding leaks into the real
    /// timesteps through the model's global-attention blocks and measurably
    /// changes low-confidence characters.
    fn recognize_batch(
        &mut self,
        w: usize,
        chunk: &[usize],
        lines: &[PrepLine],
    ) -> Result<Vec<(String, f32)>, String> {
        let n = chunk.len();
        let data = batch_input(w, chunk, lines);
        let input = Tensor::from_array(([n, 3, REC_HEIGHT as usize, w], data))
            .map_err(|e| format!("ocr: input tensor: {e}"))?;
        let outputs = self
            .rec
            .run(ort::inputs!["x" => input])
            .map_err(|e| format!("ocr: rec inference: {e}"))?;
        let (shape, probs) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("ocr: extract rec: {e}"))?;
        let t_len = shape[1] as usize;
        let nc = shape[2] as usize;
        Ok((0..n)
            .map(|i| {
                decode_row_scored(
                    &self.chars,
                    &probs[i * t_len * nc..(i + 1) * t_len * nc],
                    nc,
                )
            })
            .collect())
    }

    /// [`Self::recognize_batch`], additionally returning each character's CTC
    /// timestep and the sequence length, which together locate a character
    /// horizontally inside its crop.
    fn recognize_batch_positioned(
        &mut self,
        w: usize,
        chunk: &[usize],
        lines: &[PrepLine],
    ) -> Result<(Vec<PositionedLine>, usize), String> {
        let n = chunk.len();
        let data = batch_input(w, chunk, lines);
        let input = Tensor::from_array(([n, 3, REC_HEIGHT as usize, w], data))
            .map_err(|e| format!("ocr: input tensor: {e}"))?;
        let outputs = self
            .rec
            .run(ort::inputs!["x" => input])
            .map_err(|e| format!("ocr: rec inference: {e}"))?;
        let (shape, probs) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("ocr: extract rec: {e}"))?;
        let t_len = shape[1] as usize;
        let nc = shape[2] as usize;
        Ok((
            (0..n)
                .map(|i| {
                    crate::ocr_prep::decode_row_positions(
                        &self.chars,
                        &probs[i * t_len * nc..(i + 1) * t_len * nc],
                        nc,
                    )
                })
                .collect(),
            t_len,
        ))
    }

    /// Recognize `lines` and reduce to orientation-probe evidence: the
    /// confidence-weighted character count `Σ(conf × chars)` plus the raw
    /// character total (#225). Same deterministic width-batching as page OCR.
    pub(crate) fn score_lines(&mut self, lines: &[PrepLine]) -> Result<(f32, usize), String> {
        let mut weighted = 0.0f32;
        let mut chars = 0usize;
        for (w, chunk) in width_batches(lines) {
            for (text, conf) in self.recognize_batch(w, &chunk, lines)? {
                let n = text.trim().chars().count();
                weighted += conf * n as f32;
                chars += n;
            }
        }
        Ok((weighted, chars))
    }

    /// OCR a page: produce text cells (page points) for every line found inside
    /// the text regions, each paired with its recognition confidence (mean
    /// emitted-character probability — feeds the page `ocr_score`, #183).
    /// `scale` is image-px per page-point.
    /// The DBNet session, loaded once on first use. `None` when the asset is
    /// absent — a missing optional model degrades (the caller falls back to ink
    /// projection) rather than failing the conversion.
    fn det_session(&mut self) -> Option<&mut Session> {
        if !self.det_tried {
            self.det_tried = true;
            let path = crate::resolve_asset(".models/ocr_det_en.onnx");
            if std::path::Path::new(&path).exists() {
                self.det = Session::builder()
                    .ok()
                    .and_then(|b| b.with_intra_threads(1).ok())
                    .and_then(|b| docling_onnx::apply(b).ok())
                    .and_then(|mut b| b.commit_from_file(&path).ok());
                if self.det.is_none() {
                    eprintln!("docling-pdf: text detector present but failed to load ({path})");
                }
            } else {
                docling_core::debug_log!(
                    "docling-pdf: no text detector at {path}; picture-interior OCR \
                     falls back to ink projection"
                );
            }
        }
        self.det.as_mut()
    }

    /// Recognize `regions` using **detected** line boxes rather than ink
    /// projection. `Ok(None)` when no detector is available, so the caller can
    /// fall back.
    ///
    /// This is the path for crops the layout model failed on — a whole page
    /// boxed as a picture. Ink projection assumes its input is one isolated
    /// block of prose; on artwork it profiles the picture and loses the text.
    pub fn ocr_regions_detected(
        &mut self,
        img: &RgbImage,
        regions: &[Region],
        scale: f32,
    ) -> Result<Option<Vec<(TextCell, f32)>>, String> {
        let (iw, ih) = img.dimensions();
        let mut boxes: Vec<(u32, u32, u32, u32)> = Vec::new();
        {
            let Some(det) = self.det_session() else {
                return Ok(None);
            };
            for region in regions {
                let l = (region.l * scale).max(0.0) as u32;
                let t = (region.t * scale).max(0.0) as u32;
                let r = ((region.r * scale).max(0.0) as u32).min(iw);
                let b = ((region.b * scale).max(0.0) as u32).min(ih);
                if r <= l || b <= t {
                    continue;
                }
                let crop = image::imageops::crop_imm(img, l, t, r - l, b - t).to_image();
                // Detected boxes are crop-relative; lift them back to page pixels.
                for (bl, bt, br, bb) in crate::det::detect(det, &crop)? {
                    boxes.push((l + bl, t + bt, l + br, t + bb));
                }
            }
        }
        if boxes.is_empty() {
            return Ok(Some(Vec::new()));
        }
        // Top-to-bottom, then left-to-right, so cells reach the assembler in the
        // same page order the projection path produced.
        boxes.sort_by_key(|&(l, t, _, _)| (t, l));

        let mut lines = Vec::with_capacity(boxes.len());
        let mut kept = Vec::with_capacity(boxes.len());
        for &(l, t, r, b) in &boxes {
            let crop = image::imageops::crop_imm(img, l, t, r - l, b - t).to_image();
            if let Some(pl) = crate::ocr_prep::prep_line(&crate::ocr_prep::normalize_polarity(crop))
            {
                lines.push(pl);
                kept.push((l, t, r, b));
            }
        }
        let mut texts = vec![(String::new(), 0.0f32); lines.len()];
        for (w, chunk) in width_batches(&lines) {
            for (&i, text) in chunk.iter().zip(self.recognize_batch(w, &chunk, &lines)?) {
                texts[i] = text;
            }
        }
        let mut cells = Vec::new();
        for ((l, t, r, b), (text, conf)) in kept.into_iter().zip(texts) {
            let text = text.trim().to_string();
            if text.is_empty() {
                continue;
            }
            cells.push((
                TextCell {
                    text,
                    l: l as f32 / scale,
                    t: t as f32 / scale,
                    r: r as f32 / scale,
                    b: b as f32 / scale,
                },
                conf,
            ));
        }
        Ok(Some(cells))
    }

    pub fn ocr_page(
        &mut self,
        img: &RgbImage,
        regions: &[Region],
        scale: f32,
    ) -> Result<Vec<(TextCell, f32)>, String> {
        Ok(self.ocr_page_words(img, regions, scale)?.0)
    }

    /// [`Self::ocr_page`], additionally returning per-**word** cells.
    ///
    /// The recognition model reads a whole line and emits a string; it never
    /// produces word boxes. But it does emit each character at a CTC timestep,
    /// and the crop is downsampled horizontally by a fixed factor, so a timestep
    /// maps linearly onto x. Splitting the decode at its space characters and
    /// mapping the surrounding timesteps therefore recovers word geometry from
    /// the run that already happened — no second pass, and aligned to the words
    /// the model actually read (unlike an ink-gap split, which yields boxes with
    /// nothing to pair them to when its count disagrees with the decode).
    ///
    /// A line whose decode holds no space yields no words; callers fall back to
    /// the line box.
    pub fn ocr_page_words(
        &mut self,
        img: &RgbImage,
        regions: &[Region],
        scale: f32,
    ) -> Result<PageOcr, String> {
        // Gather every line crop on the page first (shared with the browser
        // path), so equal-width lines can share a recognition run regardless
        // of which region they came from.
        let (bboxes, lines) = prep_region_lines(img, regions, scale);

        // Deterministic width-batching (shared with the wasm path).
        let mut texts = vec![(String::new(), 0.0f32, Vec::new()); lines.len()];
        // The padded batch width each line was recognized at: a timestep spans
        // the *padded* crop, so the true width is what turns it into a fraction.
        let mut batch_w = vec![0usize; lines.len()];
        let mut t_lens = vec![0usize; lines.len()];
        for (w, chunk) in width_batches(&lines) {
            let (decoded, t_len) = self.recognize_batch_positioned(w, &chunk, &lines)?;
            for (&i, row) in chunk.iter().zip(decoded) {
                texts[i] = row;
                batch_w[i] = w;
                t_lens[i] = t_len;
            }
        }

        // Emit cells in page order, exactly as the sequential walk did.
        let mut cells = Vec::new();
        let mut words = Vec::new();
        for (i, ((l, t, r, b), (text, conf, at))) in bboxes.into_iter().zip(texts).enumerate() {
            words.extend(line_words(
                &text,
                &at,
                t_lens[i],
                batch_w[i],
                lines[i].w,
                (l, t, r, b),
            ));
            let text = text.trim().to_string();
            if text.is_empty() {
                continue;
            }
            cells.push((TextCell { text, l, t, r, b }, conf));
        }
        Ok((cells, words))
    }

    /// Recognize the *word* crops inside the page's table regions (mirroring
    /// the browser scanned path): [`ocr_page`](Self::ocr_page) deliberately
    /// skips table labels, so a scanned table would otherwise reach the cell
    /// matcher with no words at all and dissolve (#173). Returns word-level
    /// [`TextCell`]s in page points.
    pub fn ocr_table_words(
        &mut self,
        img: &RgbImage,
        regions: &[Region],
        scale: f32,
    ) -> Result<Vec<(TextCell, f32)>, String> {
        let (bboxes, lines) = prep_table_words(img, regions, scale);
        let mut texts = vec![(String::new(), 0.0f32); lines.len()];
        for (w, chunk) in width_batches(&lines) {
            for (&i, text) in chunk.iter().zip(self.recognize_batch(w, &chunk, &lines)?) {
                texts[i] = text;
            }
        }
        let mut cells = Vec::new();
        for ((l, t, r, b), (text, conf)) in bboxes.into_iter().zip(texts) {
            let text = text.trim().to_string();
            if text.is_empty() {
                continue;
            }
            cells.push((TextCell { text, l, t, r, b }, conf));
        }
        Ok(cells)
    }
}

/// Columns to extend a word's box back from its first character's timestep, and
/// forward from its last.
///
/// CTC is peaky: the model commits a character in one sharp spike, typically
/// near the *end* of that glyph's visual evidence rather than its centre. Taking
/// the raw timesteps therefore produces boxes inset on both sides — measured
/// against an independent OCR engine on a 200 dpi scan, the left edge sat
/// +2.3 pt inside the glyph and the right edge -1.0 pt, a deficit that stayed
/// constant across word length (so a fixed inset, not a scale error). A
/// single-character word degenerated worst of all: one timestep wide, ~24 % of
/// its glyph.
///
/// Expressed in columns rather than points so the correction scales with font
/// size and render scale instead of being tuned to one document. One column is
/// ~2.3 pt on that scan, which is what makes these values ~1.0 and ~1.5.
const LEAD: f32 = 1.0;
const TRAIL: f32 = 1.5;

/// Split one line's decode at its space characters and map each word's span of
/// CTC timesteps onto a box inside the line's page-point rect.
///
/// A timestep indexes the *padded* batch crop, so the fraction of the real crop
/// it sits at is `t / t_len * padded_w / line_w` — dividing by the padded width
/// instead is the mistake that shifts every box right by the padding.
fn line_words(
    text: &str,
    at: &[usize],
    t_len: usize,
    padded_w: usize,
    line_w: usize,
    (l, t, r, b): (f32, f32, f32, f32),
) -> Vec<TextCell> {
    if t_len == 0 || line_w == 0 || at.len() != text.chars().count() || !text.contains(' ') {
        return Vec::new();
    }
    let span = r - l;
    let frac = |step: f32| -> f32 {
        ((step / t_len as f32) * padded_w as f32 / line_w as f32).clamp(0.0, 1.0)
    };
    let mut out = Vec::new();
    let mut word = String::new();
    let mut first: Option<usize> = None;
    let mut last = 0usize;
    let mut flush = |word: &mut String, first: &mut Option<usize>, last: usize| {
        if let Some(f) = first.take() {
            let w = word.trim();
            if !w.is_empty() {
                out.push(TextCell {
                    text: w.to_string(),
                    l: l + frac(f as f32 - LEAD) * span,
                    t,
                    r: l + frac(last as f32 + TRAIL) * span,
                    b,
                });
            }
        }
        word.clear();
    };
    for (c, &step) in text.chars().zip(at) {
        if c == ' ' {
            flush(&mut word, &mut first, last);
            continue;
        }
        if first.is_none() {
            first = Some(step);
        }
        last = step;
        word.push(c);
    }
    flush(&mut word, &mut first, last);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #254: docling's four `OcrMode` ids parse; `full_page`/`layout_regions`
    /// reduce to the force-full-page machinery, the default/pdf-aware pair to
    /// the standard text-layer-aware path. Unknown ids parse to nothing.
    #[test]
    fn ocr_mode_ids_parse_and_map_to_forcing() {
        for (id, mode, forces) in [
            ("default", OcrMode::Default, false),
            ("full_page", OcrMode::FullPage, true),
            ("layout_regions", OcrMode::LayoutRegions, true),
            (
                "pdf_aware_layout_regions",
                OcrMode::PdfAwareLayoutRegions,
                false,
            ),
        ] {
            assert_eq!(OcrMode::parse(id), Some(mode));
            assert_eq!(mode.forces_full_page(), forces, "{id}");
        }
        assert_eq!(OcrMode::parse(" Full_Page "), Some(OcrMode::FullPage));
        assert_eq!(OcrMode::parse("easyocr"), None);
        assert_eq!(OcrMode::parse(""), None);
    }
}
