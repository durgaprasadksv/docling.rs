//! Text detection (PP-OCRv3 DBNet), the stage `ocr_prep::segment_lines` stands
//! in for on the normal path.
//!
//! Line finding by horizontal ink projection works inside a region the layout
//! model has already isolated as prose: the crop is one block of text, so a row
//! profile separates its lines cleanly. It does *not* work when the crop is a
//! whole page the layout model mislabelled — a cover, a part-opener plate, an
//! illustration with lettering. There the projection is being asked to do scene
//! detection, and artwork inks every row: on one measured cover, `mean * 0.7`
//! marked 1617 of 1626 rows as ink and the whole page collapsed into three
//! meaningless bands.
//!
//! Every upstream docling OCR backend (EasyOCR/CRAFT, RapidOCR/PP-OCR,
//! Tesseract) carries a detection stage; this restores it for the one path where
//! its absence loses whole pages. The normal region-scoped path is untouched, so
//! conformance cannot move.
//!
//! DBNet emits a per-pixel text probability map. Post-processing is the standard
//! recipe minus the parts this pipeline does not need: binarize, take connected
//! components, and keep each one's *axis-aligned* bounding box — PaddleOCR fits
//! rotated min-area rectangles and offsets the polygon with Vatti clipping, but
//! the crops here are horizontal and the recognizer wants an upright box anyway.

use image::{imageops, imageops::FilterType, RgbImage};
use ort::session::Session;
use ort::value::Tensor;

/// Longest side the detector sees. PaddleOCR's `det_limit_side_len`; the network
/// is fully convolutional, so this only trades resolution for time.
const LIMIT_SIDE: u32 = 960;

/// DBNet's output is trained on *shrunk* text regions, so a raw component box
/// sits inside the glyphs. PaddleOCR dilates it back by `unclip_ratio`; applied
/// here as a proportional expansion of the rectangle.
const UNCLIP: f32 = 1.6;

/// Probability above which a pixel counts as text (`det_db_thresh`).
const PIXEL_THRESH: f32 = 0.3;

/// Mean probability a component must reach to be kept (`det_db_box_thresh`).
const BOX_THRESH: f32 = 0.5;

/// Smallest side, in detector pixels, of a box worth recognizing — drops
/// speckle without touching real lettering.
const MIN_SIDE: u32 = 3;

/// ImageNet normalization, as PaddleOCR's detector preprocessing uses — note
/// this differs from the recognizer's `(x/255 - 0.5) / 0.5`.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Run DBNet over `img` and return text-line boxes in the image's own pixel
/// coordinates. Empty when the model finds nothing.
pub fn detect(session: &mut Session, img: &RgbImage) -> Result<Vec<(u32, u32, u32, u32)>, String> {
    let (iw, ih) = img.dimensions();
    if iw == 0 || ih == 0 {
        return Ok(Vec::new());
    }
    // Resize to at most LIMIT_SIDE on the long edge, both sides a multiple of 32
    // (the network's total stride); the ratio back to source is applied to every
    // box at the end.
    let scale = (LIMIT_SIDE as f32 / iw.max(ih) as f32).min(1.0);
    let round32 = |v: f32| ((v / 32.0).round().max(1.0) as u32) * 32;
    let (nw, nh) = (round32(iw as f32 * scale), round32(ih as f32 * scale));
    let resized = imageops::resize(img, nw, nh, FilterType::Triangle);

    let mut data = vec![0f32; 3 * (nw * nh) as usize];
    let plane = (nw * nh) as usize;
    for (i, px) in resized.pixels().enumerate() {
        for c in 0..3 {
            data[c * plane + i] = (px.0[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    let input = Tensor::from_array(([1usize, 3, nh as usize, nw as usize], data))
        .map_err(|e| format!("det: input tensor: {e}"))?;
    let outputs = session
        .run(ort::inputs!["x" => input])
        .map_err(|e| format!("det: inference: {e}"))?;
    let (shape, probs) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("det: extract: {e}"))?;
    // [1, 1, H, W]
    let (ph, pw) = (shape[2] as u32, shape[3] as u32);

    let boxes = components(probs, pw, ph);
    let (sx, sy) = (iw as f32 / pw as f32, ih as f32 / ph as f32);
    Ok(boxes
        .into_iter()
        .map(|(l, t, r, b)| {
            (
                ((l as f32 * sx) as u32).min(iw),
                ((t as f32 * sy) as u32).min(ih),
                ((r as f32 * sx) as u32).min(iw),
                ((b as f32 * sy) as u32).min(ih),
            )
        })
        .filter(|(l, t, r, b)| r > l && b > t)
        .collect())
}

/// Connected components of the thresholded probability map, each returned as an
/// unclipped bounding box scored by the mean probability inside it.
///
/// Iterative flood fill (an explicit stack, not recursion): a full-page text
/// region can be tens of thousands of pixels and a recursive fill would blow the
/// stack on exactly the pages this path exists to rescue.
fn components(probs: &[f32], w: u32, h: u32) -> Vec<(u32, u32, u32, u32)> {
    let idx = |x: u32, y: u32| (y * w + x) as usize;
    let mut seen = vec![false; (w * h) as usize];
    let mut out = Vec::new();
    let mut stack: Vec<(u32, u32)> = Vec::new();
    for y0 in 0..h {
        for x0 in 0..w {
            if seen[idx(x0, y0)] || probs[idx(x0, y0)] < PIXEL_THRESH {
                continue;
            }
            let (mut l, mut t, mut r, mut b) = (x0, y0, x0, y0);
            let (mut sum, mut n) = (0.0f32, 0usize);
            stack.push((x0, y0));
            seen[idx(x0, y0)] = true;
            while let Some((x, y)) = stack.pop() {
                let p = probs[idx(x, y)];
                sum += p;
                n += 1;
                l = l.min(x);
                t = t.min(y);
                r = r.max(x);
                b = b.max(y);
                let mut push = |nx: u32, ny: u32, stack: &mut Vec<(u32, u32)>| {
                    if !seen[idx(nx, ny)] && probs[idx(nx, ny)] >= PIXEL_THRESH {
                        seen[idx(nx, ny)] = true;
                        stack.push((nx, ny));
                    }
                };
                if x > 0 {
                    push(x - 1, y, &mut stack);
                }
                if x + 1 < w {
                    push(x + 1, y, &mut stack);
                }
                if y > 0 {
                    push(x, y - 1, &mut stack);
                }
                if y + 1 < h {
                    push(x, y + 1, &mut stack);
                }
            }
            if n == 0 || sum / (n as f32) < BOX_THRESH {
                continue;
            }
            let (bw, bh) = (r - l + 1, b - t + 1);
            if bw < MIN_SIDE || bh < MIN_SIDE {
                continue;
            }
            // Unclip: grow the rectangle about its centre. DBNet shrinks regions
            // during training, so the raw component stops inside the glyphs.
            let gx = ((bw as f32 * (UNCLIP - 1.0)) / 2.0).round() as u32;
            let gy = ((bh as f32 * (UNCLIP - 1.0)) / 2.0).round() as u32;
            out.push((
                l.saturating_sub(gx),
                t.saturating_sub(gy),
                (r + gx).min(w - 1),
                (b + gy).min(h - 1),
            ));
        }
    }
    out
}
