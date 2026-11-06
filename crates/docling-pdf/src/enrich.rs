//! Optional enrichment models (docling's `do_picture_classification` /
//! `do_code_enrichment` / `do_formula_enrichment`, issue #76).
//!
//! * [`PictureClassifier`] — docling-project/DocumentFigureClassifier-v2.5, an
//!   EfficientNet image classifier over 26 figure classes (bar_chart, logo,
//!   signature, …). The HF repo ships the ONNX graph as-is; docling's ViT
//!   preprocessing is 224×224 bilinear + rescale + normalize, and the raw
//!   logits are softmaxed and sorted descending — the full distribution lands
//!   on the picture item like docling's `PictureClassificationData`.
//!
//! * [`CodeFormula`] — docling-project/CodeFormulaV2, an Idefics3/SmolVLM-class
//!   VLM that rewrites a code crop as clean source text prefixed with
//!   `<_language_>`, or a formula crop as LaTeX. Exported to three graphs by
//!   `scripts/install/export_code_formula.py` (vision tower+connector, token
//!   embeddings, and a KV-cached Llama decoder step verified argmax-identical
//!   to `transformers.generate`); this module ports the Idefics3 preprocessing
//!   (longest-edge 2048 resize → multiple-of-512 resize → 512×512 tiling + a
//!   squashed global tile), the tiled `<image>` prompt, and the greedy decode.

use image::RgbImage;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use docling_core::PictureClass;

/// docling crops enrichment inputs at `images_scale` pixels per point:
/// 2.0 for the picture classifier, 1.67 (≈120 dpi) for CodeFormula.
pub const CLASSIFIER_SCALE: f32 = 2.0;
pub const CODE_FORMULA_SCALE: f32 = 1.67;

/// The scale to render a page at so its formula crops carry real detail,
/// derived from the recognizer's own input size.
///
/// Idefics3 preprocessing resizes a crop's longest edge to 2048, so that is
/// exactly the resolution worth supplying: below it the model interpolates,
/// above it the pixels are discarded. A display equation typically spans
/// about half the text width, so targeting 2048px across half the page gives
/// `2 * 2048 / page_width` — about 6.7 px/pt on US Letter, 14 on a small
/// octavo page.
///
/// The ordinary 2 px/pt page bitmap (144 dpi) is ample for vector typesetting
/// and destructive for a scan: the exercises page of a 1900s arithmetic
/// textbook carries a 600 dpi embedded image, so a superscript ~20px tall in
/// the source arrived ~5px tall at the model, where `3` and `8` are genuinely
/// indistinguishable. Upscaling the 2 px/pt crop cannot recover it — measured,
/// and the output was identical to the pixel; rendering the page afresh at
/// this scale took one page's exercises from 2 of 6 correct to 6 of 6.
///
/// Clamped at both ends: never below the ordinary page render (there would be
/// nothing to gain), and never above [`MAX_ENRICH_SCALE`], since the render
/// is a full-page RGB bitmap and the pipeline holds several pages at once.
/// `DOCLING_RS_ENRICH_SCALE` overrides it outright.
pub fn auto_enrich_scale(page_width_pt: f32) -> f32 {
    if let Some(v) = docling_core::env::parse::<f32>("DOCLING_RS_ENRICH_SCALE") {
        if v > 0.1 && v <= 16.0 {
            return v;
        }
    }
    let want = 2.0 * LONGEST_EDGE as f32 / page_width_pt.max(1.0);
    want.clamp(crate::pdfium_backend::RENDER_SCALE, MAX_ENRICH_SCALE)
}

/// Ceiling for [`auto_enrich_scale`]: a US Letter page at 8 px/pt is a
/// 4896x6336 RGB bitmap (~93 MB), and pages are processed several at a time.
pub const MAX_ENRICH_SCALE: f32 = 8.0;
/// CodeFormula expands the region box by 18% of its size on every side
/// (docling's `expansion_factor`) before cropping.
pub const CODE_FORMULA_EXPANSION: f32 = 0.18;

// ---------------------------------------------------------------------------
// Picture classifier
// ---------------------------------------------------------------------------

/// The 26 classes of DocumentFigureClassifier-v2.5, indexed by model class id
/// (`config.json` `id2label`).
pub const PICTURE_CLASSES: [&str; 26] = [
    "logo",
    "photograph",
    "icon",
    "engineering_drawing",
    "line_chart",
    "bar_chart",
    "other",
    "table",
    "flow_chart",
    "screenshot_from_computer",
    "signature",
    "screenshot_from_manual",
    "geographical_map",
    "pie_chart",
    "page_thumbnail",
    "stamp",
    "music",
    "calendar",
    "qr_code",
    "bar_code",
    "full_page_image",
    "scatter_plot",
    "chemistry_structure",
    "topographical_map",
    "crossword_puzzle",
    "box_plot",
];

const CLASSIFIER_SIDE: u32 = 224;
/// ViT preprocessing constants from the model's `preprocessor_config.json`.
const CLASSIFIER_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const CLASSIFIER_STD: [f32; 3] = [0.478_539_44, 0.473_286_4, 0.474_341_63];

pub struct PictureClassifier {
    session: Session,
}

impl PictureClassifier {
    /// Load from `DOCLING_PICTURE_CLASSIFIER_ONNX` /
    /// `.models/picture_classifier(_int8).onnx`. `None` when the graph is
    /// absent — the caller warns once and skips classification.
    pub fn load_with(intra: usize) -> Option<Self> {
        let path = crate::model_path(
            "DOCLING_PICTURE_CLASSIFIER_ONNX",
            ".models/picture_classifier.onnx",
            ".models/picture_classifier_int8.onnx",
        );
        if !std::path::Path::new(&path).exists() {
            eprintln!(
                "docling-pdf: picture classifier model not found ({path}); \
                 picture classification skipped. Run scripts/install/download_dependencies.sh."
            );
            return None;
        }
        let builder = docling_onnx::session_builder()
            .map_err(|e| eprintln!("docling-pdf: picture classifier: {e}"))
            .ok()?
            .with_intra_threads(intra)
            .ok()?;
        let builder = docling_onnx::apply(builder)
            .and_then(docling_onnx::cap_before_1_29)
            .map_err(|e| eprintln!("docling-pdf: picture classifier: {e}"))
            .ok()?;
        let session = docling_onnx::commit_uncached(builder, &path)
            .map_err(|e| eprintln!("docling-pdf: picture classifier load {path}: {e}"))
            .ok()?;
        Some(Self { session })
    }

    /// Classify one picture crop: the full 26-class distribution, descending
    /// confidence (docling attaches all predicted classes, not just the top).
    pub fn classify(&mut self, crop: &RgbImage) -> Result<Vec<PictureClass>, String> {
        let resized = image::imageops::resize(
            crop,
            CLASSIFIER_SIDE,
            CLASSIFIER_SIDE,
            image::imageops::FilterType::Triangle,
        );
        let n = (CLASSIFIER_SIDE * CLASSIFIER_SIDE) as usize;
        let mut data = vec![0f32; 3 * n];
        for (i, px) in resized.pixels().enumerate() {
            for c in 0..3 {
                data[c * n + i] = (px[c] as f32 / 255.0 - CLASSIFIER_MEAN[c]) / CLASSIFIER_STD[c];
            }
        }
        let input = Tensor::from_array((
            [
                1usize,
                3,
                CLASSIFIER_SIDE as usize,
                CLASSIFIER_SIDE as usize,
            ],
            data,
        ))
        .map_err(|e| format!("picture classifier: input: {e}"))?;
        let outputs = docling_onnx::run(&mut self.session, ort::inputs!["input" => input])
            .map_err(|e| format!("picture classifier: inference: {e}"))?;
        let (_, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("picture classifier: output: {e}"))?;
        // softmax → (class, prob) sorted descending, like docling's engine.
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let mut preds: Vec<PictureClass> = exp
            .iter()
            .enumerate()
            .map(|(i, &e)| PictureClass {
                class_name: PICTURE_CLASSES.get(i).copied().unwrap_or("other").into(),
                confidence: e / sum,
            })
            .collect();
        preds.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        Ok(preds)
    }
}

// ---------------------------------------------------------------------------
// CodeFormula (Idefics3 VLM)
// ---------------------------------------------------------------------------

/// Which prompt the model gets for a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeFormulaKind {
    Code,
    Formula,
}

/// Idefics3 processor constants (CodeFormulaV2's `preprocessor_config.json` /
/// tokenizer). The token ids are fixed by the checkpoint's tokenizer.json.
const TILE: u32 = 512; // max_image_size.longest_edge
const LONGEST_EDGE: u32 = 2048; // size.longest_edge
const MAX_IMAGE_SIZE: u32 = 4096; // transformers' hard upper bound
const IMAGE_SEQ_LEN: usize = 64; // visual tokens per tile
const IMAGE_TOKEN_ID: i64 = 100270; // <image>
const EOS_ID: i64 = 100338; // <end_of_utterance>
const MODEL_MAX_LEN: usize = 8192;
/// Generation budget for a *formula* crop — generated tokens, the prompt
/// not counted. See the decode loop.
const FORMULA_MAX_NEW_TOKENS: usize = 1024;
const HIDDEN: usize = 576;
const N_LAYERS: usize = 30;
const N_KV: usize = 3;
const HEAD_DIM: usize = 64;

/// Which checkpoint family the loaded graphs came from. Both are Idefics3
/// with the same SigLIP2 tower, pixel-shuffle connector and 165M Llama-style
/// decoder (576 hidden, 30 layers, 3 KV heads), so the ONNX graphs and the
/// decode loop are shared; only the chat template, the task query and the
/// end-of-turn token differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flavor {
    /// `docling-project/CodeFormulaV2`: `user:<image><formula><end_of_utterance>`.
    CodeFormulaV2,
    /// `ibm-granite/granite-docling-258M` (granite chat template, task
    /// prompts like "Convert formula to LaTeX.", `<|end_of_text|>` ends a turn).
    GraniteDocling,
}

pub struct CodeFormula {
    vision: Session,
    embed: Session,
    decoder: Session,
    tokenizer: Tokenizer,
    flavor: Flavor,
    /// Token ids that end the answer. Looked up by name because the two
    /// checkpoints number their special tokens differently (CodeFormulaV2's
    /// `<end_of_utterance>` is 100338; granite-docling's is 100352 and its
    /// template closes turns with `<|end_of_text|>` 100257 instead).
    eos: Vec<i64>,
}

impl CodeFormula {
    /// Load the three graphs + tokenizer from `DOCLING_CODE_FORMULA_DIR`
    /// (default `.models/code_formula/`). `None` when absent, with a one-time
    /// warning from the caller's slot.
    pub fn load_with(intra: usize) -> Option<Self> {
        let dir = docling_core::env::nonempty("DOCLING_CODE_FORMULA_DIR")
            .unwrap_or_else(|| crate::resolve_asset(".models/code_formula"));
        let file = |name: &str| format!("{dir}/{name}");
        if !std::path::Path::new(&file("tokenizer.json")).exists() {
            eprintln!(
                "docling-pdf: CodeFormula model not found ({dir}/tokenizer.json); code/formula \
                 enrichment skipped. Run scripts/install/download_dependencies.sh."
            );
            return None;
        }
        let tokenizer = Tokenizer::from_file(file("tokenizer.json"))
            .map_err(|e| eprintln!("docling-pdf: CodeFormula tokenizer: {e}"))
            .ok()?;
        // Both checkpoints descend from the granite vocabulary (role markers
        // and all), so the discriminator is the SmolVLM turn terminator:
        // CodeFormulaV2's tokenizer has `<end_of_utterance>` (its EOS, 100338),
        // granite-docling's does not and closes a turn with `<|end_of_text|>`.
        // `DOCLING_CODE_FORMULA_FLAVOR=granite|codeformula` forces it.
        let eou = tokenizer.token_to_id("<end_of_utterance>").map(i64::from);
        let flavor = match docling_core::env::nonempty("DOCLING_CODE_FORMULA_FLAVOR").as_deref() {
            Some("granite") => Flavor::GraniteDocling,
            Some("codeformula") => Flavor::CodeFormulaV2,
            _ if eou.is_some() => Flavor::CodeFormulaV2,
            _ => Flavor::GraniteDocling,
        };
        let eos: Vec<i64> = match flavor {
            Flavor::CodeFormulaV2 => vec![eou.unwrap_or(EOS_ID)],
            Flavor::GraniteDocling => tokenizer
                .token_to_id("<|end_of_text|>")
                .map(i64::from)
                .into_iter()
                .collect(),
        };
        if eos.is_empty() {
            eprintln!("docling-pdf: CodeFormula: no end-of-turn token in {dir}/tokenizer.json");
            return None;
        }
        // INT8 variants take priority when present, like the other models —
        // except for granite-docling's decoder: its dynamic per-tensor INT8
        // quantization is not near-exact the way CodeFormulaV2's is. Measured
        // on the same 52 crops, the INT8 decoder ran away to the token budget
        // on 13 of 32 real crops where fp32 finished every one, so the
        // flavor pins fp32 (the file is simply ignored if present).
        // A GPU build prefers the fp32 graphs (`prefer_fp32`), but the
        // release ships CodeFormulaV2's decoder as INT8 only (~165 MB in place
        // of the 655 MB fp32 export), so a CUDA server that insisted on fp32
        // silently skipped every formula. Take the preferred variant when it
        // is on disk and the other one when it is not: INT8 is near-exact for
        // CodeFormulaV2, and a decoded formula beats none.
        let allow_int8 = flavor == Flavor::CodeFormulaV2;
        let graph = |base: &str| {
            let fp32 = file(&format!("{base}.onnx"));
            let int8 = file(&format!("{base}_int8.onnx"));
            let have = |p: &str| std::path::Path::new(p).exists();
            let want_int8 = allow_int8 && !crate::prefer_fp32();
            match (want_int8, have(&int8), have(&fp32)) {
                (true, true, _) => int8,
                (false, _, true) => fp32,
                (_, true, false) if allow_int8 => int8,
                _ => fp32,
            }
        };
        for f in [&graph("vision"), &graph("embed"), &graph("decoder_kv")] {
            if !std::path::Path::new(f.as_str()).exists() {
                eprintln!(
                    "docling-pdf: CodeFormula model not found ({f}); code/formula \
                     enrichment skipped. Run scripts/install/download_dependencies.sh."
                );
                return None;
            }
        }
        let load = |p: String| {
            let builder = docling_onnx::session_builder()
                .map_err(|e| eprintln!("docling-pdf: CodeFormula: {e}"))
                .ok()?
                .with_intra_threads(intra)
                .ok()?;
            let builder = docling_onnx::apply(builder)
                .map_err(|e| eprintln!("docling-pdf: CodeFormula: {e}"))
                .ok()?;
            docling_onnx::commit_uncached(builder, &p)
                .map_err(|e| eprintln!("docling-pdf: CodeFormula load {p}: {e}"))
                .ok()
        };
        crate::debug_log!("code-formula: {flavor:?} from {dir}, eos ids {eos:?}");
        Some(Self {
            vision: load(graph("vision"))?,
            embed: load(graph("embed"))?,
            decoder: load(graph("decoder_kv"))?,
            tokenizer,
            flavor,
            eos,
        })
    }

    /// Run the VLM on a code/formula crop and return the post-processed text
    /// (still carrying the `<_language_>` prefix for code — see
    /// [`extract_code_language`]).
    pub fn predict(&mut self, crop: &RgbImage, kind: CodeFormulaKind) -> Result<String, String> {
        // Debug aid: dump each crop the VLM sees (compare against docling's
        // `prepare_element` output when chasing a generation divergence).
        if let Some(dir) = docling_core::env::nonempty("DOCLING_RS_ENRICH_DEBUG") {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static N: AtomicUsize = AtomicUsize::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let _ = crop.save(format!("{dir}/rs_crop_{n}.png"));
        }
        let (tiles, rows, cols) =
            crate::timing::timed("code_formula.preprocess", || preprocess_idefics3(crop));
        let n_tiles = tiles.len() / (3 * (TILE * TILE) as usize);

        // Vision tower over the tile batch → [T, 64, 576]. (Scoped: the
        // session outputs hold a mutable borrow of the session until dropped.)
        let feats: Vec<f32> = {
            let input = Tensor::from_array(([n_tiles, 3, TILE as usize, TILE as usize], tiles))
                .map_err(|e| format!("code-formula: vision input: {e}"))?;
            let outputs = crate::timing::timed("code_formula.vision", || {
                docling_onnx::run(&mut self.vision, ort::inputs!["pixel_values" => input])
            })
            .map_err(|e| format!("code-formula: vision: {e}"))?;
            let (_, feats) = outputs["image_features"]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("code-formula: vision output: {e}"))?;
            feats.to_vec()
        };

        // Prompt: the chat template with the single <image> expanded into the
        // per-tile token grid (transformers' Idefics3Processor layout).
        let prompt = self.prompt(kind, rows, cols);
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| format!("code-formula: tokenize: {e}"))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&v| v as i64).collect();
        let seq = ids.len();

        // Token embeddings, then scatter the visual tokens into the <image>
        // positions (Idefics3's inputs_merger).
        let mut embeds = crate::timing::timed("code_formula.embed", || self.embed_ids(&ids))?;
        let image_positions: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|(_, &t)| t == IMAGE_TOKEN_ID)
            .map(|(i, _)| i)
            .collect();
        if image_positions.len() != n_tiles * IMAGE_SEQ_LEN {
            return Err(format!(
                "code-formula: {} image tokens for {} tiles",
                image_positions.len(),
                n_tiles
            ));
        }
        for (v, &pos) in image_positions.iter().enumerate() {
            embeds[pos * HIDDEN..(pos + 1) * HIDDEN]
                .copy_from_slice(&feats[v * HIDDEN..(v + 1) * HIDDEN]);
        }

        // Greedy KV-cache decode until <end_of_utterance> (docling caps
        // generation at the model's 8192 context). The K/V caches stay owned
        // `ort` values fed straight back into the next step — never extracted
        // or copied (they grow every step, so per-step copies would be
        // O(steps²) float traffic; same pattern as the TableFormer decoder).
        // ort's array constructors reject a 0-length dim, so the zero-`past`
        // first-step tensors go through the session allocator.
        let mut cache: Option<(ort::value::DynValue, ort::value::DynValue)> = None;
        let empty = {
            let mk = || {
                Tensor::<f32>::new(
                    self.decoder.allocator(),
                    [N_LAYERS, 1, N_KV, 0usize, HEAD_DIM],
                )
                .map_err(|e| format!("code-formula: empty kv cache: {e}"))
            };
            (mk()?, mk()?)
        };
        let mut past_len = 0usize;
        let mut positions: Vec<i64> = (0..seq as i64).collect();
        let mut x = embeds;
        let mut x_seq = seq;
        let mut out_ids: Vec<u32> = Vec::new();
        // Under a GPU provider the loop runs through an `IoBinding`, like
        // TableFormer's `KvLayered` decode: with plain runs every step handed
        // ONNX Runtime host tensors to copy to the device and got the whole
        // KV cache back as host tensors — 30 layers of it, one token longer
        // each step, both directions — so on an RTX PRO 4000 a formula took
        // 3–9 s, no faster than a laptop CPU. Bound, the cache outputs stay in
        // device memory and are the next step's inputs; only `logits` (a
        // 100k-float row) comes back. Same graph, same kernels: the output is
        // the plain loop's. `DOCLING_RS_ENRICH_IOBINDING=0` forces the plain
        // loop for comparison.
        let device = match docling_core::env::nonempty("DOCLING_RS_ENRICH_IOBINDING") {
            Some(_) if !docling_core::env::flag("DOCLING_RS_ENRICH_IOBINDING") => None,
            _ => docling_onnx::device_memory_info(),
        };
        let mut binding = match &device {
            Some(_) => Some(
                self.decoder
                    .create_binding()
                    .map_err(|e| format!("code-formula: io binding: {e}"))?,
            ),
            None => None,
        };
        let host = ort::memory::MemoryInfo::default();
        // A greedy decode that never emits EOS runs to the model's whole
        // context — ~7,800 steps, each a separate decoder invocation. That is
        // not a hypothetical: a crop the model cannot parse (a table the layout
        // stage labelled `formula`, a block of several equations) does exactly
        // this, and one such crop costs minutes where a real equation costs a
        // second or two.
        //
        // Past a few hundred tokens a formula's output is garbage anyway, so
        // the cap loses nothing and bounds the worst case. Chosen against
        // PP-FormulaNet, which caps formula generation at 1,024 tokens (2,560
        // in its largest variant) — real equations here decode in well under
        // 200. Code blocks keep the full budget: unlike an equation, a long one
        // is legitimate.
        let budget = match kind {
            CodeFormulaKind::Formula => FORMULA_MAX_NEW_TOKENS,
            CodeFormulaKind::Code => MODEL_MAX_LEN,
        };
        // The budget counts *generated* tokens. It used to include the prompt,
        // which was harmless at docling's 1.67 px/pt crops (a few tiles) and
        // fatal at the native-resolution ones: a tall equation block tiles to
        // 17 × 64 image tokens, more than the whole formula budget, so the
        // decode started with a 0-token budget and the block was dropped
        // ("generation hit the 0-token budget"). The model's context still
        // caps it.
        let max_new = match kind {
            CodeFormulaKind::Formula => budget,
            CodeFormulaKind::Code => budget.saturating_sub(seq),
        }
        .min(MODEL_MAX_LEN.saturating_sub(seq));
        let mut finished = false;
        for _ in 0..max_new {
            let embeds_t = Tensor::from_array(([1usize, x_seq, HIDDEN], x))
                .map_err(|e| format!("code-formula: embeds: {e}"))?;
            let pos_t = Tensor::from_array(([1usize, positions.len()], positions.clone()))
                .map_err(|e| format!("code-formula: positions: {e}"))?;
            let stage = if cache.is_none() {
                "code_formula.prefill"
            } else {
                "code_formula.step"
            };
            let t_step = std::time::Instant::now();
            let next = {
                let mut out = match (&mut binding, &device) {
                    (Some(b), Some(dev)) => {
                        let err = |what: &str, e: ort::Error| {
                            format!("code-formula: io binding {what}: {e}")
                        };
                        b.bind_input("inputs_embeds", &embeds_t)
                            .map_err(|e| err("inputs_embeds", e))?;
                        b.bind_input("position_ids", &pos_t)
                            .map_err(|e| err("position_ids", e))?;
                        match cache.as_ref() {
                            Some((k, v)) => {
                                b.bind_input("past_k", k).map_err(|e| err("past_k", e))?;
                                b.bind_input("past_v", v).map_err(|e| err("past_v", e))?;
                            }
                            None => {
                                b.bind_input("past_k", &empty.0)
                                    .map_err(|e| err("past_k", e))?;
                                b.bind_input("past_v", &empty.1)
                                    .map_err(|e| err("past_v", e))?;
                            }
                        }
                        // Re-bound every step: the cache outputs are one token
                        // longer each time, so ONNX Runtime allocates them
                        // afresh on the device rather than reuse the last.
                        b.bind_output_to_device("new_k", dev)
                            .map_err(|e| err("new_k", e))?;
                        b.bind_output_to_device("new_v", dev)
                            .map_err(|e| err("new_v", e))?;
                        b.bind_output_to_device("logits", &host)
                            .map_err(|e| err("logits", e))?;
                        // Arena shrinkage once per decode (#842), as the
                        // TableFormer loop does: per step it is a
                        // cudaFree/cudaMalloc pair per token.
                        match (cache.is_none(), docling_onnx::run_options()) {
                            (true, Some(opts)) => self.decoder.run_binding_with_options(b, opts),
                            _ => self.decoder.run_binding(b),
                        }
                    }
                    _ => match cache.as_ref() {
                        Some((k, v)) => docling_onnx::run(
                            &mut self.decoder,
                            ort::inputs![
                            "inputs_embeds" => embeds_t, "position_ids" => pos_t,
                            "past_k" => k, "past_v" => v],
                        ),
                        None => docling_onnx::run(
                            &mut self.decoder,
                            ort::inputs![
                            "inputs_embeds" => embeds_t, "position_ids" => pos_t,
                            "past_k" => &empty.0, "past_v" => &empty.1],
                        ),
                    },
                }
                .map_err(|e| format!("code-formula: decoder: {e}"))?;
                let (_, logits) = out["logits"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("code-formula: logits: {e}"))?;
                let next = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i as i64)
                    .unwrap_or(self.eos[0]);
                cache = Some((
                    out.remove("new_k")
                        .ok_or_else(|| "code-formula: new_k missing".to_string())?,
                    out.remove("new_v")
                        .ok_or_else(|| "code-formula: new_v missing".to_string())?,
                ));
                next
            };
            crate::timing::record(stage, t_step.elapsed().as_nanos());
            past_len += x_seq;
            if self.eos.contains(&next) {
                finished = true;
                break;
            }
            out_ids.push(next as u32);
            // A greedy decode that has started looping never emits EOS; it
            // runs to the budget — 1,024 steps at ~10 ms each on a GPU, and
            // on one 12-page scanned algebra chapter six such decodes per run
            // were most of the decode time, producing nothing. Stop at the
            // first repetition instead; the result is the same (no formula).
            if looping(&out_ids) {
                return Err(format!(
                    "generation looped after {} tokens (a repeated cycle)",
                    out_ids.len()
                ));
            }
            x = crate::timing::timed("code_formula.embed", || self.embed_ids(&[next]))?;
            x_seq = 1;
            positions = vec![past_len as i64];
        }

        // A decode that exhausts its budget without ever emitting EOS did not
        // finish — it ran away. Its output is not a partial answer to be
        // salvaged: it is the model looping, and on a page of worn type one
        // such crop produced `x_1, y_1, z_1, …` repeated to the cap. Reporting
        // that as a formula would have a reader narrate noise with no signal
        // that anything was wrong, which is worse than saying nothing. Errors
        // here degrade to an undecoded formula, which the reader announces.
        if !finished {
            return Err(format!(
                "generation hit the {max_new}-token budget without finishing"
            ));
        }
        let text = self
            .tokenizer
            .decode(&out_ids, false)
            .map_err(|e| format!("code-formula: decode: {e}"))?;
        Ok(post_process(&text))
    }

    /// The full chat-templated prompt for a `rows`×`cols` tile grid: each
    /// checkpoint's own template around the task query. The query can be
    /// overridden for experiments (`DOCLING_RS_FORMULA_PROMPT` /
    /// `DOCLING_RS_CODE_PROMPT`), e.g. to try granite-docling's long-form
    /// instruction ("Convert formula to LaTeX.") against the default.
    fn prompt(&self, kind: CodeFormulaKind, rows: u32, cols: u32) -> String {
        let default = match (self.flavor, kind) {
            (Flavor::CodeFormulaV2, CodeFormulaKind::Code) => "<code>",
            (Flavor::CodeFormulaV2, CodeFormulaKind::Formula) => "<formula>",
            // granite-docling's model card lists both the long instruction
            // ("Convert formula to LaTeX.") and this short form. Measured on
            // 52 crops (scanned 1900s algebra, OpenStax physics/chemistry,
            // synthetic): the long form ran away to the token budget on half
            // the real crops and wrapped answers in prose; the short form —
            // the DocTags tag the model emits in full-page mode — did not.
            (Flavor::GraniteDocling, CodeFormulaKind::Code) => "<code>",
            (Flavor::GraniteDocling, CodeFormulaKind::Formula) => "<formula>",
        };
        let env = match kind {
            CodeFormulaKind::Code => "DOCLING_RS_CODE_PROMPT",
            CodeFormulaKind::Formula => "DOCLING_RS_FORMULA_PROMPT",
        };
        let query = docling_core::env::nonempty(env).unwrap_or_else(|| default.to_string());
        let img = image_prompt(rows, cols);
        match self.flavor {
            Flavor::CodeFormulaV2 => {
                format!("<|start_of_role|>user:{img}{query}<end_of_utterance>\nassistant:")
            }
            Flavor::GraniteDocling => format!(
                "<|start_of_role|>user<|end_of_role|>{img}{query}<|end_of_text|>\n\
                 <|start_of_role|>assistant<|end_of_role|>"
            ),
        }
    }

    fn embed_ids(&mut self, ids: &[i64]) -> Result<Vec<f32>, String> {
        let input = Tensor::from_array(([1usize, ids.len()], ids.to_vec()))
            .map_err(|e| format!("code-formula: ids: {e}"))?;
        let out = docling_onnx::run(&mut self.embed, ort::inputs!["input_ids" => input])
            .map_err(|e| format!("code-formula: embed: {e}"))?;
        let (_, embeds) = out["inputs_embeds"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("code-formula: embed output: {e}"))?;
        Ok(embeds.to_vec())
    }
}

/// The `<image>` expansion for a rows×cols tile grid + global tile
/// (transformers' `Idefics3Processor.replace_image_token`).
fn image_prompt(rows: u32, cols: u32) -> String {
    let img = "<image>".repeat(IMAGE_SEQ_LEN);
    let mut s = String::new();
    for r in 1..=rows {
        for c in 1..=cols {
            s.push_str(&format!("<fake_token_around_image><row_{r}_col_{c}>{img}"));
        }
        s.push('\n');
    }
    s.push_str(&format!(
        "\n<fake_token_around_image><global-img>{img}<fake_token_around_image>"
    ));
    s
}

/// Idefics3 image preprocessing: longest-edge 2048 resize (LANCZOS), resize to
/// multiples of 512, split into 512×512 tiles (row-major) plus the whole image
/// squashed to 512×512 as the trailing global tile, then `(x/255 - 0.5)/0.5`.
/// Returns the flattened `[T,3,512,512]` tensor data and the grid shape.
fn preprocess_idefics3(crop: &RgbImage) -> (Vec<f32>, u32, u32) {
    // 1. longest edge → 2048 exactly (up- or down-scale), short side rounded
    //    to even, both clamped below 4096 (rescale_to_max_len).
    let (w0, h0) = crop.dimensions();
    let (mut w, mut h) = rescale_to_max_len(w0, h0, LONGEST_EDGE);
    (h, w) = scale_below_upper_bound(h, w, MAX_IMAGE_SIZE);
    let img = lanczos3(crop, w, h);

    // 2. ceil each side to a multiple of the 512 tile (resize_for_vision_encoder).
    let (tw, th) = if w >= h {
        let tw = w.div_ceil(TILE) * TILE;
        let th0 = (tw as f64 / (w as f64 / h as f64)) as u32;
        (tw, th0.div_ceil(TILE) * TILE)
    } else {
        let th = h.div_ceil(TILE) * TILE;
        let tw0 = (th as f64 * (w as f64 / h as f64)) as u32;
        (tw0.div_ceil(TILE) * TILE, th)
    };
    let img = lanczos3(&img, tw, th);

    // 3. tiles (row-major) + the global squash.
    let (rows, cols) = (th / TILE, tw / TILE);
    let mut tensor = Vec::with_capacity(((rows * cols + 1) * 3 * TILE * TILE) as usize);
    for r in 0..rows {
        for c in 0..cols {
            let tile = image::imageops::crop_imm(&img, c * TILE, r * TILE, TILE, TILE).to_image();
            push_normalized(&mut tensor, &tile);
        }
    }
    let global = lanczos3(&img, TILE, TILE);
    push_normalized(&mut tensor, &global);
    (tensor, rows, cols)
}

/// Lanczos-3 resize through `fast_image_resize` (SIMD convolution, the same
/// kernel as `image::imageops::resize(.., Lanczos3)`). The three resizes of a
/// crop's preprocessing cost 288 ms per crop with the scalar `image` resize —
/// 6.6 s of the 34 s two scanned pages took on a GPU, where the decode itself
/// had become the only other cost. `DOCLING_RS_SLOW_RESIZE=1` restores the
/// scalar path, as it does for the page bitmaps.
fn lanczos3(src: &RgbImage, dw: u32, dh: u32) -> RgbImage {
    use fast_image_resize as fir;
    static SLOW: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let slow = *SLOW.get_or_init(|| docling_core::env::flag("DOCLING_RS_SLOW_RESIZE"));
    if !slow && (dw, dh) != src.dimensions() {
        if let Some(out) = (|| {
            let s = fir::images::ImageRef::new(
                src.width(),
                src.height(),
                src.as_raw(),
                fir::PixelType::U8x3,
            )
            .ok()?;
            let mut dst = fir::images::Image::new(dw, dh, fir::PixelType::U8x3);
            fir::Resizer::new()
                .resize(
                    &s,
                    &mut dst,
                    &fir::ResizeOptions::new()
                        .resize_alg(fir::ResizeAlg::Convolution(fir::FilterType::Lanczos3)),
                )
                .ok()?;
            RgbImage::from_raw(dw, dh, dst.into_vec())
        })() {
            return out;
        }
    }
    image::imageops::resize(src, dw, dh, image::imageops::FilterType::Lanczos3)
}

/// transformers' `_resize_output_size_rescale_to_max_len`: longest edge to
/// `max_len` exactly, the short side `int(long/aspect)` bumped to even.
fn rescale_to_max_len(w0: u32, h0: u32, max_len: u32) -> (u32, u32) {
    let aspect = w0 as f64 / h0 as f64;
    let (w, h) = if w0 >= h0 {
        let w = max_len;
        let mut h = (w as f64 / aspect) as u32;
        if !h.is_multiple_of(2) {
            h += 1;
        }
        (w, h)
    } else {
        let h = max_len;
        let mut w = (h as f64 * aspect) as u32;
        if !w.is_multiple_of(2) {
            w += 1;
        }
        (w, h)
    };
    (w.max(1), h.max(1))
}

/// transformers' `_resize_output_size_scale_below_upper_bound` (a no-op unless
/// the even-bump pushed a side past the hard 4096 cap).
fn scale_below_upper_bound(h0: u32, w0: u32, max_len: u32) -> (u32, u32) {
    let aspect = w0 as f64 / h0 as f64;
    let (h, w) = if w0 >= h0 && w0 > max_len {
        let w = max_len;
        (((w as f64 / aspect) as u32).max(1), w)
    } else if h0 > w0 && h0 > max_len {
        let h = max_len;
        (h, ((h as f64 * aspect) as u32).max(1))
    } else {
        (h0, w0)
    };
    (h.max(1), w.max(1))
}

/// Append one 512×512 tile as CHW `(x/255 - 0.5)/0.5`.
fn push_normalized(tensor: &mut Vec<f32>, tile: &RgbImage) {
    let n = (TILE * TILE) as usize;
    let base = tensor.len();
    tensor.resize(base + 3 * n, 0.0);
    for (i, px) in tile.pixels().enumerate() {
        for c in 0..3 {
            tensor[base + c * n + i] = px[c] as f32 / 255.0 * 2.0 - 1.0;
        }
    }
}

/// Whether a greedy decode has fallen into a cycle: its last `3 × n` tokens
/// are the same `n`-token sequence three times over, for some cycle length
/// `n` up to 32. A real equation repeats a short token run that often only
/// in degenerate cases (`x_1, y_1, z_1, …` on a page of worn type was the
/// decode that motivated this), and such a decode never finishes anyway.
/// Checked from 24 tokens on so the cheapest cycles (one or two tokens
/// stuttering) are caught early, and a cycle has to fill at least 24 tokens
/// so `\\ a & b` rows of a genuine matrix do not trip it.
fn looping(out: &[u32]) -> bool {
    let len = out.len();
    if len < 24 {
        return false;
    }
    (1..=32).any(|n| {
        // The window: at least three copies of the cycle, and at least 24
        // tokens of it, whichever is longer.
        let w = (3 * n).max(24);
        len >= w && (0..w - n).all(|i| out[len - 1 - i] == out[len - 1 - i - n])
    })
}

/// docling's CodeFormulaModel post-processing: truncate at
/// `<end_of_utterance>`, remove the closing/query artifacts, strip leading
/// whitespace.
fn post_process(text: &str) -> String {
    let mut t = match text.find("<end_of_utterance>") {
        Some(i) => &text[..i],
        None => text,
    }
    .to_string();
    for tok in ["</code>", "</formula>", "<loc_0><loc_0><loc_500><loc_500>"] {
        t = t.replace(tok, "");
    }
    // granite-docling answers a crop prompt DocTags-style — the opening task
    // tag and a `<loc_…>` box run can precede the content. Strip them all.
    t = strip_loc_tokens(&t);
    for tok in ["<formula>", "<code>"] {
        if let Some(rest) = t.trim_start().strip_prefix(tok) {
            t = rest.to_string();
        }
    }
    t.trim_start().to_string()
}

/// Remove every `<loc_N>` token from `s`.
fn strip_loc_tokens(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find("<loc_") {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        match tail.find('>') {
            Some(j) if j > 5 && tail[5..j].bytes().all(|b| b.is_ascii_digit()) => {
                rest = &tail[j + 1..]
            }
            _ => {
                out.push_str("<loc_");
                rest = &tail[5..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Collapse an immediately repeated line in a formula's LaTeX.
///
/// A greedy decoder handed a multi-line equation sometimes emits one of its
/// lines twice — on a scanned textbook page, exercise 11 (which wraps onto a
/// second line) came back as its own first line, repeated. Nothing downstream
/// can tell that apart from a genuine repetition, and a reader voices it twice.
///
/// Only *consecutive* identical lines are collapsed, and only for formulas: two
/// identical adjacent lines are near-meaningless in an equation, while in code
/// they are ordinary.
pub fn dedupe_repeated_lines(latex: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for seg in latex.split("\\\\") {
        if out.last().map(|p: &&str| p.trim()) != Some(seg.trim()) {
            out.push(seg);
        }
    }
    out.join("\\\\")
}

/// docling's `_extract_code_language`: an output beginning with
/// `<_language_>` yields `(remainder, Some(language))`.
pub fn extract_code_language(s: &str) -> (String, Option<String>) {
    let rest = match s.strip_prefix("<_") {
        Some(r) => r,
        None => return (s.to_string(), None),
    };
    // The language is everything up to the closing `_>` that contains neither
    // `_` nor `>` (docling's `[^_>]+`).
    match rest.find("_>") {
        Some(end) if !rest[..end].is_empty() && !rest[..end].contains(['_', '>']) => {
            let lang = rest[..end].to_string();
            let remainder = rest[end + 2..].trim_start().to_string();
            (remainder, Some(lang))
        }
        _ => (s.to_string(), None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_prefix_extraction() {
        assert_eq!(
            extract_code_language("<_JavaScript_> function f() {}"),
            (
                "function f() {}".to_string(),
                Some("JavaScript".to_string())
            )
        );
        assert_eq!(
            extract_code_language("plain text"),
            ("plain text".to_string(), None)
        );
        assert_eq!(
            extract_code_language("<_x_y_> t"),
            ("<_x_y_> t".to_string(), None)
        );
    }

    #[test]
    fn idefics3_grid_matches_processor() {
        // An 800×300 crop resizes to 2048×768, tiles to 2048×1024 → 4×2 grid
        // (+ global) — the shape verified against transformers' processor.
        let img = RgbImage::new(800, 300);
        let (tensor, rows, cols) = preprocess_idefics3(&img);
        assert_eq!((rows, cols), (2, 4));
        assert_eq!(tensor.len(), 9 * 3 * 512 * 512);
    }

    #[test]
    fn looping_decodes_are_caught_and_real_runs_are_not() {
        // A one-token stutter: caught as soon as 24 tokens of it exist.
        let stutter: Vec<u32> = std::iter::repeat_n(7, 24).collect();
        assert!(looping(&stutter));
        assert!(!looping(&stutter[..23]));
        // `x_1, y_1, z_1,` repeated: a 6-token cycle, three times = 18 tokens,
        // below the 24-token floor; four times trips it (at n=8 the window
        // of 24 is three 8-token runs of a 6-cycle… only when aligned, so
        // check the natural length too).
        let cycle: Vec<u32> = [1, 2, 3, 4, 5, 6].repeat(3);
        assert!(!looping(&cycle));
        let cycle: Vec<u32> = [1, 2, 3, 4, 5, 6].repeat(4);
        assert!(looping(&cycle));
        // A long non-repeating run, and a matrix-like row structure whose
        // rows differ, are not loops.
        let distinct: Vec<u32> = (0..200).collect();
        assert!(!looping(&distinct));
        let rows: Vec<u32> = (0..6)
            .flat_map(|r| [100, 200 + r, 300, 400 + r, 500])
            .collect();
        assert!(!looping(&rows));
    }

    #[test]
    fn granite_crop_answer_is_stripped_to_content() {
        // granite-docling answers a crop DocTags-style: task tag, a full-page
        // `<loc_…>` box run, the LaTeX, closing tag.
        assert_eq!(
            post_process("<formula><loc_0><loc_0><loc_500><loc_500>\\frac { a } { b }</formula>"),
            "\\frac { a } { b }"
        );
        // Only well-formed `<loc_N>` runs are removed; look-alikes survive.
        assert_eq!(
            strip_loc_tokens("x<loc_12>y<loc_>z<loc_a>"),
            "xy<loc_>z<loc_a>"
        );
    }

    #[test]
    fn image_prompt_layout() {
        let p = image_prompt(1, 2);
        assert!(p.starts_with("<fake_token_around_image><row_1_col_1><image>"));
        assert!(p.contains("<row_1_col_2>"));
        let tail = "<fake_token_around_image><global-img>".to_owned()
            + &"<image>".repeat(64)
            + "<fake_token_around_image>";
        assert!(p.ends_with(&tail));
        assert_eq!(p.matches("<image>").count(), 3 * 64);
    }
}
