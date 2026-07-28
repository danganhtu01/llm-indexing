//! Faces sub-tier: YuNet detection + SFace embedding, on the SAME `ort` the
//! tags-tier detector already uses.
//!
//! The pair is the one decided in `plan.md` §7 B1 and surveyed in
//! `docs/VISION-RESEARCH.md` §2/§6: **YuNet** (`face_detection_yunet_2023mar`)
//! locates faces and their five landmarks, **SFace**
//! (`face_recognition_sface_2021dec`) turns each aligned crop into a 128-d
//! vector. Both are OpenCV-Zoo ONNX artifacts under Apache-2.0 — the only
//! fully-clean face pair in the survey (the InsightFace `buffalo_l` packs are
//! more accurate but their weights are non-commercial, so they are never
//! shipped here).
//!
//! ## Why this ships OFF by default
//!
//! A face embedding is a biometric identifier for a person who did not ask to
//! be enrolled — including people who merely appear in the background of
//! someone else's photo. Everything else this engine writes is a fact about a
//! *file*; this is a fact about a *person*. So it is opt-in at every layer and
//! each layer defaults to off:
//!
//! 1. the models are not downloaded by `fetch-data --vision` — only by an
//!    explicit `fetch-data --faces`;
//! 2. `vision.faces` defaults to `"off"` in config, in the per-job settings
//!    merge, and on the CLI;
//! 3. absent models make the capability *absent*, never an error — a job that
//!    asks for faces on a box that has not staged them runs the rest of its
//!    tier unchanged (see [`available`]);
//! 4. `GET /settings` reports `faces[].present` so an app can grey the control
//!    out rather than offering something the box cannot do.
//!
//! **Local-only.** Detection and embedding run in-process through `ort`; there
//! is no network call at index time (the models arrive only via the operator-run
//! fetch). The vectors are written into the corpus's own `faces` table on the
//! machine that indexed the file and are never rendered into `fts.content`,
//! sidecars, manifests or job summaries — see [`FaceDetection`]. Nothing in this
//! engine sends them anywhere.
//!
//! ## Determinism
//!
//! Same file + same model files ⇒ same faces, same order, same bytes. There is
//! no augmentation, no multi-crop voting, no RNG, and no test-time resizing
//! choice left to chance: the image is letterboxed into the model's fixed
//! 640x640 input with a fixed filter, decoded against fixed priors, filtered at
//! fixed thresholds, ordered by score with positional tie-breaks, and each crop
//! is aligned by a closed-form similarity transform onto fixed reference
//! landmarks. The same caveat the rest of the vision tiers carry applies here
//! too: identical bytes are guaranteed for a given build, model file and ONNX
//! Runtime execution provider, not across different ones.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use image::{imageops::FilterType, DynamicImage, RgbImage};
use ort::session::Session;
use ort::value::Tensor;

use crate::config::VisionConfig;
use crate::vision::types::{FaceDetection, VisionResult};

/// Face **detector** file under `<data_dir>/vision` — the OpenCV-Zoo YuNet
/// export. Named by the `VISION_MODELS` registry entry in `mod.rs` so the
/// staging command and the presence check agree on one filename.
pub const YUNET_ONNX: &str = "yunet.onnx";
/// Face **embedder** file under `<data_dir>/vision` — the OpenCV-Zoo SFace
/// export.
pub const SFACE_ONNX: &str = "sface.onnx";

/// The id recorded in `faces.model` and `vision.faces_model`, and accepted by
/// the `vision.faces` toggle. One id for the pair: SFace embeddings are only
/// comparable when they come from crops aligned by YuNet's landmarks, so the two
/// models are selected and versioned together, never independently.
pub const FACE_MODEL_ID: &str = "yunet-sface";

/// Square input the pinned YuNet export declares (`input: [1, 3, 640, 640]`).
/// The 2023mar ONNX has a STATIC input shape, so — unlike OpenCV's
/// `FaceDetectorYN`, which reshapes the graph per image — every image is
/// letterboxed into this box. That is the stricter of the two: it fixes the
/// prior grid, so a given image always meets the same anchors.
const INPUT_SIZE: u32 = 640;

/// Feature-map strides YuNet emits, one output triple per stride
/// (`cls_N`/`obj_N`/`bbox_N`/`kps_N`). At 640 they give 80x80 + 40x40 + 20x20
/// prior positions.
const STRIDES: [u32; 3] = [8, 16, 32];

/// IoU above which the lower-scoring of two overlapping boxes is dropped.
/// OpenCV's `FaceDetectorYN` default, kept as a constant rather than a knob:
/// it trades one duplicate against one missed neighbour and has no per-corpus
/// right answer, so exposing it would only add a way to make results
/// incomparable between jobs.
const NMS_IOU: f32 = 0.3;

/// Faces whose shorter side is under this many ORIGINAL-image pixels are
/// dropped before embedding. Below roughly this size the aligned 112x112 crop is
/// mostly upsampling artefact and its embedding is noise wearing a person's
/// name — and per digiKam 8.6's published lesson (VISION-RESEARCH §6) a blurry
/// or tiny face poisons a cluster far more than a missed face costs. Fixed, not
/// a knob, for the same reason as [`NMS_IOU`].
const MIN_FACE_PX: u32 = 24;

/// Side of the aligned crop SFace consumes (`data: [1, 3, 112, 112]`).
const CROP_SIZE: u32 = 112;

/// Dimensionality of an SFace embedding (`fc1: [1, 128]`).
pub const EMBEDDING_DIMS: usize = 128;

/// The canonical 5-point face template SFace was trained against (ArcFace's
/// 112x112 reference: left eye, right eye, nose tip, left mouth corner, right
/// mouth corner), copied from OpenCV's `FaceRecognizerSF::alignCrop`. Every crop
/// is warped onto these coordinates, which is what makes two photos of one
/// person comparable at all.
const REFERENCE_LANDMARKS: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// A face as YuNet reports it, before it is mapped back out of the letterbox:
/// box plus the five landmarks the aligner needs.
#[derive(Debug, Clone, PartialEq)]
struct RawFace {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    score: f32,
    landmarks: [[f32; 2]; 5],
}

/// Whether the faces capability exists on this box: both model files are
/// present under `models_dir`.
///
/// Deliberately a presence check and nothing more — it runs once per job, and
/// the point of the call is to answer "can this run?" cheaply enough that the
/// answer costs nothing when it is no. Integrity (the pinned SHA-256) is the
/// separate question [`crate::vision::faces_present`] answers for `GET
/// /settings` and [`crate::vision::corrupt_face_models`] answers at job start.
///
/// An absent pair is NOT an error anywhere: it means the capability is absent,
/// and a job that asked for faces simply runs the rest of its tier. That is the
/// deliberate asymmetry with the tags tier, whose missing models fail a job at
/// submit — a corpus with no faces in it is a corpus that told the truth, while
/// a job killed because an opt-in privacy feature was unavailable would push an
/// operator toward staging biometric models they did not want.
pub fn available(models_dir: &Path) -> bool {
    models_dir.join(YUNET_ONNX).is_file() && models_dir.join(SFACE_ONNX).is_file()
}

/// Letterbox `image` into a fixed `size`x`size` BGR planar `f32` tensor, and
/// report the scale used so boxes can be mapped back.
///
/// Aspect ratio is preserved (the remainder is left black) because a squashed
/// face is a face YuNet was not trained on; the scale is `min(size/w, size/h)`
/// so the whole frame is always visible. Pixel values stay in `[0, 255]` and the
/// channel order is B,G,R — both models are OpenCV exports and carry their own
/// normalization inside the graph (SFace's leading `Sub 127.5` / `Mul 1/128`),
/// so pre-scaling here would double-normalize.
fn letterbox(image: &RgbImage, size: u32) -> (Vec<f32>, f32) {
    let (width, height) = (image.width().max(1), image.height().max(1));
    let scale = (f64::from(size) / f64::from(width)).min(f64::from(size) / f64::from(height));
    let target_w = ((f64::from(width) * scale).round() as u32).clamp(1, size);
    let target_h = ((f64::from(height) * scale).round() as u32).clamp(1, size);
    let resized = image::imageops::resize(image, target_w, target_h, FilterType::Triangle);
    let plane = (size * size) as usize;
    let mut data = vec![0.0_f32; 3 * plane];
    for (x, y, pixel) in resized.enumerate_pixels() {
        let offset = (y * size + x) as usize;
        // BGR planar.
        data[offset] = f32::from(pixel[2]);
        data[plane + offset] = f32::from(pixel[1]);
        data[2 * plane + offset] = f32::from(pixel[0]);
    }
    (data, scale as f32)
}

/// Decode one stride's YuNet outputs into candidate faces above `score_min`.
///
/// The head is anchor-free: prior `(r, c)` on the stride grid predicts a centre
/// offset in cells, a log-space size, and five landmark offsets in cells. The
/// score is `sqrt(cls * obj)` — the geometric mean of the classification and
/// objectness branches, both already sigmoided in the graph — which is exactly
/// OpenCV's `FaceDetectorYN` post-processing, clamps included.
fn decode_stride(
    cls: &[f32],
    obj: &[f32],
    bbox: &[f32],
    kps: &[f32],
    stride: u32,
    size: u32,
    score_min: f32,
) -> Vec<RawFace> {
    let cols = (size / stride) as usize;
    let rows = cols;
    let priors = rows * cols;
    let usable = priors
        .min(cls.len())
        .min(obj.len())
        .min(bbox.len() / 4)
        .min(kps.len() / 10);
    let stride = stride as f32;
    let mut faces = Vec::new();
    for index in 0..usable {
        let score = (cls[index].clamp(0.0, 1.0) * obj[index].clamp(0.0, 1.0)).sqrt();
        if score < score_min {
            continue;
        }
        let column = (index % cols) as f32;
        let row = (index / cols) as f32;
        let center_x = (column + bbox[index * 4]) * stride;
        let center_y = (row + bbox[index * 4 + 1]) * stride;
        let width = bbox[index * 4 + 2].exp() * stride;
        let height = bbox[index * 4 + 3].exp() * stride;
        let mut landmarks = [[0.0_f32; 2]; 5];
        for (point, slot) in landmarks.iter_mut().enumerate() {
            slot[0] = (kps[index * 10 + point * 2] + column) * stride;
            slot[1] = (kps[index * 10 + point * 2 + 1] + row) * stride;
        }
        faces.push(RawFace {
            x: center_x - width / 2.0,
            y: center_y - height / 2.0,
            width,
            height,
            score,
            landmarks,
        });
    }
    faces
}

/// Intersection-over-union of two boxes given as `(x, y, w, h)`.
fn iou(left: &RawFace, right: &RawFace) -> f32 {
    let overlap_x = (left.x + left.width).min(right.x + right.width) - left.x.max(right.x);
    let overlap_y = (left.y + left.height).min(right.y + right.height) - left.y.max(right.y);
    if overlap_x <= 0.0 || overlap_y <= 0.0 {
        return 0.0;
    }
    let intersection = overlap_x * overlap_y;
    let union = left.width * left.height + right.width * right.height - intersection;
    if union <= 0.0 {
        return 0.0;
    }
    intersection / union
}

/// Greedy non-maximum suppression, highest score first.
///
/// The sort is TOTAL — score descending, then top edge, then left edge — so two
/// equally-confident faces (a mirrored pair, a repeated face in a collage) keep
/// a fixed order across runs instead of inheriting whatever order the prior grid
/// happened to produce. That order is also what `face_index` and the `max_faces`
/// truncation below key on, which is what makes the stored rows reproducible.
fn suppress(mut faces: Vec<RawFace>, threshold: f32) -> Vec<RawFace> {
    faces.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left.y
                    .partial_cmp(&right.y)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                left.x
                    .partial_cmp(&right.x)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    let mut kept: Vec<RawFace> = Vec::new();
    for face in faces {
        if kept.iter().all(|keep| iou(keep, &face) < threshold) {
            kept.push(face);
        }
    }
    kept
}

/// The 2x3 similarity transform (rotation + uniform scale + translation) that
/// best maps `landmarks` onto [`REFERENCE_LANDMARKS`], as `[a, -b, tx, b, a, ty]`.
///
/// This is the closed-form least-squares similarity fit — the same answer
/// OpenCV's `alignCrop` gets from Umeyama, without the SVD: for a rotation the
/// determinant is positive by construction, and Umeyama's reflection correction
/// only ever fires on a mirrored configuration, which five landmarks of a real
/// face are not. Returns `None` for a degenerate (all-coincident) landmark set,
/// where no scale exists.
fn similarity_transform(landmarks: &[[f32; 2]; 5]) -> Option<[f32; 6]> {
    let count = landmarks.len() as f64;
    let src_mean = [
        landmarks
            .iter()
            .map(|point| f64::from(point[0]))
            .sum::<f64>()
            / count,
        landmarks
            .iter()
            .map(|point| f64::from(point[1]))
            .sum::<f64>()
            / count,
    ];
    let dst_mean = [
        REFERENCE_LANDMARKS
            .iter()
            .map(|point| f64::from(point[0]))
            .sum::<f64>()
            / count,
        REFERENCE_LANDMARKS
            .iter()
            .map(|point| f64::from(point[1]))
            .sum::<f64>()
            / count,
    ];
    let (mut variance, mut dot, mut cross) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (source, target) in landmarks.iter().zip(REFERENCE_LANDMARKS.iter()) {
        let sx = f64::from(source[0]) - src_mean[0];
        let sy = f64::from(source[1]) - src_mean[1];
        let dx = f64::from(target[0]) - dst_mean[0];
        let dy = f64::from(target[1]) - dst_mean[1];
        variance += sx * sx + sy * sy;
        dot += sx * dx + sy * dy;
        cross += sx * dy - sy * dx;
    }
    if variance <= f64::EPSILON {
        return None;
    }
    let a = dot / variance;
    let b = cross / variance;
    let tx = dst_mean[0] - (a * src_mean[0] - b * src_mean[1]);
    let ty = dst_mean[1] - (b * src_mean[0] + a * src_mean[1]);
    Some([
        a as f32, -b as f32, tx as f32, b as f32, a as f32, ty as f32,
    ])
}

/// Invert a 2x3 similarity `[a, -b, tx, b, a, ty]`, so the warp below can be
/// written as an inverse map (for each destination pixel, where does it come
/// from) — the only form that fills every output pixel exactly once.
fn invert_similarity(matrix: [f32; 6]) -> Option<[f32; 6]> {
    let (a, b) = (f64::from(matrix[0]), f64::from(matrix[3]));
    let determinant = a * a + b * b;
    if determinant <= f64::EPSILON {
        return None;
    }
    let (inv_a, inv_b) = (a / determinant, -b / determinant);
    let (tx, ty) = (f64::from(matrix[2]), f64::from(matrix[5]));
    let inv_tx = -(inv_a * tx - inv_b * ty);
    let inv_ty = -(inv_b * tx + inv_a * ty);
    Some([
        inv_a as f32,
        -inv_b as f32,
        inv_tx as f32,
        inv_b as f32,
        inv_a as f32,
        inv_ty as f32,
    ])
}

/// Warp `image` onto the aligned `CROP_SIZE` square through `matrix`, sampled
/// bilinearly, and lay the result out as SFace's BGR planar `[1, 3, 112, 112]`
/// input. Samples outside the source are black, matching OpenCV `warpAffine`'s
/// default constant border, so a face at the very edge of a frame aligns the
/// same way here as under the reference implementation.
fn warp_crop(image: &RgbImage, matrix: [f32; 6]) -> Option<Vec<f32>> {
    let inverse = invert_similarity(matrix)?;
    let side = CROP_SIZE as usize;
    let plane = side * side;
    let mut data = vec![0.0_f32; 3 * plane];
    for y in 0..side {
        for x in 0..side {
            let (dx, dy) = (x as f32 + 0.5, y as f32 + 0.5);
            let source_x = inverse[0] * dx + inverse[1] * dy + inverse[2] - 0.5;
            let source_y = inverse[3] * dx + inverse[4] * dy + inverse[5] - 0.5;
            let [red, green, blue] = sample_bilinear(image, source_x, source_y);
            let offset = y * side + x;
            data[offset] = blue;
            data[plane + offset] = green;
            data[2 * plane + offset] = red;
        }
    }
    Some(data)
}

/// Bilinear sample of `image` at a continuous pixel coordinate, black outside.
fn sample_bilinear(image: &RgbImage, x: f32, y: f32) -> [f32; 3] {
    let (width, height) = (image.width() as i64, image.height() as i64);
    let x0 = x.floor();
    let y0 = y.floor();
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as i64, y0 as i64);
    let mut out = [0.0_f32; 3];
    for (dy, weight_y) in [(0_i64, 1.0 - fy), (1, fy)] {
        for (dx, weight_x) in [(0_i64, 1.0 - fx), (1, fx)] {
            let weight = weight_x * weight_y;
            if weight == 0.0 {
                continue;
            }
            let (px, py) = (x0 + dx, y0 + dy);
            if px < 0 || py < 0 || px >= width || py >= height {
                continue;
            }
            let pixel = image.get_pixel(px as u32, py as u32);
            for channel in 0..3 {
                out[channel] += weight * f32::from(pixel[channel]);
            }
        }
    }
    out
}

/// The two loaded ONNX sessions, each behind its own `Mutex` so a detection and
/// an embedding never contend on one lock.
struct FaceSessions {
    detector: Mutex<Session>,
    embedder: Mutex<Session>,
}

/// Build the YuNet + SFace pair.
///
/// `intra_cap` is the headroom core cap (`VisionConfig::headroom_cores_cap`):
/// `Some` builds each session with `.with_intra_threads(cap)`, `None` keeps
/// ONNX Runtime's every-core default — byte-identical to before the knob
/// existed. The `map_err(ort::Error::from)` converts the builder's
/// `Error<SessionBuilder>` (not `Send + Sync`, so not anyhow-compatible) into
/// the plain `Error<()>` that is — see `detector::build_session`, which owns
/// the full explanation.
fn build_sessions(models_dir: &Path, intra_cap: Option<usize>) -> Result<FaceSessions> {
    let open = |relative: &str| -> Result<Mutex<Session>> {
        let path = models_dir.join(relative);
        anyhow::ensure!(
            path.is_file(),
            "face model not found at {} (run `llm-index fetch-data --faces`)",
            path.display()
        );
        let mut builder = Session::builder().context("creating face session builder")?;
        if let Some(cap) = intra_cap {
            builder = builder
                .with_intra_threads(cap.max(1))
                .map_err(ort::Error::<()>::from)
                .context("capping face session intra-op threads")?;
        }
        let session = builder
            .commit_from_file(&path)
            .with_context(|| format!("loading face model {}", path.display()))?;
        Ok(Mutex::new(session))
    };
    Ok(FaceSessions {
        detector: open(YUNET_ONNX)?,
        embedder: open(SFACE_ONNX)?,
    })
}

/// The cached process-wide face sessions. Only a *successful* build is cached,
/// exactly as `detector.rs` does it: a model file momentarily truncated by a
/// re-fetch must not poison a resident `serve` process until restart. A
/// dedicated init lock serializes construction so the rayon workers do not all
/// load 38 MB of SFace at once.
///
/// HEADROOM LIMITATION: `intra_cap` is baked into the sessions on the FIRST
/// build, so the first job's headroom wins for the process lifetime — the same
/// cache property `detector::session` documents in full.
fn sessions(models_dir: &Path, intra_cap: Option<usize>) -> Result<&'static FaceSessions> {
    static SESSIONS: OnceLock<FaceSessions> = OnceLock::new();
    static INIT: Mutex<()> = Mutex::new(());
    if let Some(sessions) = SESSIONS.get() {
        return Ok(sessions);
    }
    let _guard = INIT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(sessions) = SESSIONS.get() {
        return Ok(sessions);
    }
    let built = build_sessions(models_dir, intra_cap).context("face model init failed")?;
    Ok(SESSIONS.get_or_init(|| built))
}

/// Run YuNet over `image`, returning the surviving faces in ORIGINAL-image
/// coordinates, best first.
fn detect(image: &RgbImage, sessions: &FaceSessions, cfg: &VisionConfig) -> Result<Vec<RawFace>> {
    let (input, scale) = letterbox(image, INPUT_SIZE);
    let mut session = sessions
        .detector
        .lock()
        .map_err(|_| anyhow::anyhow!("face detector session mutex poisoned"))?;
    let input_name = session
        .inputs()
        .first()
        .context("face detector model has no input")?
        .name()
        .to_string();
    let tensor = Tensor::from_array((
        [1_usize, 3, INPUT_SIZE as usize, INPUT_SIZE as usize],
        input,
    ))
    .context("building face detector input tensor")?;
    // Scoped so the borrowed outputs are dropped — and the session lock with
    // them — before the (pure) suppression below.
    let candidates = {
        let outputs = session
            .run(ort::inputs![input_name => tensor])
            .context("running face detection")?;
        let mut candidates = Vec::new();
        for stride in STRIDES {
            let plane = |prefix: &str| -> Result<Vec<f32>> {
                let key = format!("{prefix}_{stride}");
                let value = outputs
                    .get(key.as_str())
                    .with_context(|| format!("face detector produced no '{key}' output"))?;
                let (_, data) = value
                    .try_extract_tensor::<f32>()
                    .with_context(|| format!("face detector output '{key}' is not f32"))?;
                Ok(data.to_vec())
            };
            candidates.extend(decode_stride(
                &plane("cls")?,
                &plane("obj")?,
                &plane("bbox")?,
                &plane("kps")?,
                stride,
                INPUT_SIZE,
                cfg.face_score,
            ));
        }
        candidates
    };
    drop(session);

    // Suppress in letterbox space (where the boxes were predicted), then map
    // the survivors back through the single scale factor.
    let inverse = if scale > 0.0 { 1.0 / scale } else { 1.0 };
    Ok(suppress(candidates, NMS_IOU)
        .into_iter()
        .map(|face| RawFace {
            x: face.x * inverse,
            y: face.y * inverse,
            width: face.width * inverse,
            height: face.height * inverse,
            score: face.score,
            landmarks: face
                .landmarks
                .map(|point| [point[0] * inverse, point[1] * inverse]),
        })
        .collect())
}

/// Embed one aligned crop with SFace. `None` when the landmarks are degenerate;
/// an inference failure propagates.
fn embed(image: &RgbImage, face: &RawFace, sessions: &FaceSessions) -> Result<Option<Vec<f32>>> {
    let Some(matrix) = similarity_transform(&face.landmarks) else {
        return Ok(None);
    };
    let Some(crop) = warp_crop(image, matrix) else {
        return Ok(None);
    };
    let mut session = sessions
        .embedder
        .lock()
        .map_err(|_| anyhow::anyhow!("face embedder session mutex poisoned"))?;
    // SFace's ONNX export lists its weights as graph inputs alongside the image
    // (an MXNet-export trait), so the image input is selected by name rather
    // than by position; every other input is initializer-backed and left alone.
    let input_name = session
        .inputs()
        .iter()
        .map(|input| input.name().to_string())
        .find(|name| name == "data")
        .or_else(|| {
            session
                .inputs()
                .first()
                .map(|input| input.name().to_string())
        })
        .context("face embedder model has no input")?;
    let tensor = Tensor::from_array(([1_usize, 3, CROP_SIZE as usize, CROP_SIZE as usize], crop))
        .context("building face embedder input tensor")?;
    let outputs = session
        .run(ort::inputs![input_name => tensor])
        .context("running face embedding")?;
    let key = outputs
        .keys()
        .next()
        .context("face embedder produced no output")?
        .to_string();
    let value = outputs
        .get(key.as_str())
        .context("face embedder output vanished")?;
    let (_, data) = value
        .try_extract_tensor::<f32>()
        .context("face embedder output is not f32")?;
    Ok(Some(data.to_vec()))
}

/// Detect and embed faces in `image`, filling `out.faces` and stamping
/// `out.faces_model`.
///
/// The stamp is written even when nothing was found: "scanned, no faces" is a
/// different fact from "never scanned", and only the former lets a resume leave
/// the file alone (see [`VisionResult::faces_model`]).
pub(crate) fn fill(
    image: &DynamicImage,
    models_dir: &Path,
    cfg: &VisionConfig,
    out: &mut VisionResult,
) -> Result<()> {
    let sessions = sessions(models_dir, cfg.headroom_cores_cap)?;
    let rgb = image.to_rgb8();
    let detections = detect(&rgb, sessions, cfg)?;
    let mut faces = Vec::new();
    for face in detections {
        let width = face.width.round().max(0.0) as u32;
        let height = face.height.round().max(0.0) as u32;
        if width.min(height) < MIN_FACE_PX {
            continue;
        }
        if faces.len() >= cfg.max_faces {
            break;
        }
        let embedding = embed(&rgb, &face, sessions)?;
        faces.push(FaceDetection {
            x: face.x.round() as i32,
            y: face.y.round() as i32,
            width,
            height,
            quality: face.score,
            embedding,
            frame: None,
        });
    }
    out.faces = faces;
    out.faces_model = Some(FACE_MODEL_ID.to_string());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(x: f32, y: f32, size: f32, score: f32) -> RawFace {
        RawFace {
            x,
            y,
            width: size,
            height: size,
            score,
            landmarks: [[0.0; 2]; 5],
        }
    }

    #[test]
    fn available_needs_both_model_files() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!available(dir.path()));
        std::fs::write(dir.path().join(YUNET_ONNX), b"stub").unwrap();
        assert!(!available(dir.path()), "the detector alone is not the pair");
        std::fs::write(dir.path().join(SFACE_ONNX), b"stub").unwrap();
        assert!(available(dir.path()));
        std::fs::remove_file(dir.path().join(YUNET_ONNX)).unwrap();
        assert!(!available(dir.path()));
    }

    #[test]
    fn letterbox_preserves_aspect_and_pads_the_rest() {
        // A 40x20 image into a 40-box: scale 1.0 horizontally, bottom half pad.
        let image = RgbImage::from_pixel(40, 20, image::Rgb([10, 20, 30]));
        let (data, scale) = letterbox(&image, 40);
        assert!((scale - 1.0).abs() < 1e-6);
        assert_eq!(data.len(), 3 * 40 * 40);
        let plane = 40 * 40;
        // Top-left pixel carries the source colour, in BGR planar order.
        assert!((data[0] - 30.0).abs() < 1e-3);
        assert!((data[plane] - 20.0).abs() < 1e-3);
        assert!((data[2 * plane] - 10.0).abs() < 1e-3);
        // The padded bottom half is black in every plane.
        let padded = 30 * 40;
        for channel in 0..3 {
            assert_eq!(data[channel * plane + padded], 0.0);
        }
        // A wider-than-tall image scales by the long side.
        let (_, scale) = letterbox(&RgbImage::new(100, 25), 50);
        assert!((scale - 0.5).abs() < 1e-6);
    }

    #[test]
    fn decode_stride_reads_the_anchor_free_head() {
        // One 2x2 grid at stride 16 (so `size` 32): only prior (1, 0) is
        // confident. cls/obj are already sigmoided by the graph, so a score of
        // sqrt(0.81 * 1.0) = 0.9 clears a 0.9 gate.
        let cls = [0.0, 0.0, 0.81, 0.0];
        let obj = [0.0, 0.0, 1.0, 0.0];
        let mut bbox = [0.0_f32; 16];
        // Prior index 2 == row 1, column 0. Centre at (0.5, 1.5) cells, size e^0.
        bbox[8] = 0.5;
        bbox[9] = 0.5;
        bbox[10] = 0.0;
        bbox[11] = 0.0;
        let mut kps = [0.0_f32; 40];
        kps[20] = 0.25; // first landmark x offset, prior 2
        kps[21] = 0.75;
        let faces = decode_stride(&cls, &obj, &bbox, &kps, 16, 32, 0.9);
        assert_eq!(faces.len(), 1);
        let face = &faces[0];
        assert!((face.score - 0.9).abs() < 1e-6);
        assert!((face.width - 16.0).abs() < 1e-4);
        // centre (0 + 0.5) * 16 = 8, (1 + 0.5) * 16 = 24 -> top-left (0, 16).
        assert!((face.x - 0.0).abs() < 1e-4, "{face:?}");
        assert!((face.y - 16.0).abs() < 1e-4, "{face:?}");
        assert!((face.landmarks[0][0] - 4.0).abs() < 1e-4);
        assert!((face.landmarks[0][1] - 28.0).abs() < 1e-4);
        // A higher gate drops it entirely.
        assert!(decode_stride(&cls, &obj, &bbox, &kps, 16, 32, 0.95).is_empty());
    }

    #[test]
    fn decode_stride_tolerates_short_outputs() {
        // Fewer values than the grid implies: the extra priors are not read.
        let faces = decode_stride(&[1.0], &[1.0], &[0.0; 4], &[0.0; 10], 16, 64, 0.5);
        assert_eq!(faces.len(), 1);
        assert!(decode_stride(&[], &[], &[], &[], 8, 64, 0.5).is_empty());
    }

    #[test]
    fn suppress_keeps_the_best_of_an_overlapping_pair() {
        let faces = vec![
            raw(0.0, 0.0, 10.0, 0.80),
            raw(1.0, 1.0, 10.0, 0.95), // ~68% IoU with the first
            raw(50.0, 50.0, 10.0, 0.90),
        ];
        let kept = suppress(faces, NMS_IOU);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].score - 0.95).abs() < 1e-6);
        assert!((kept[1].score - 0.90).abs() < 1e-6);
    }

    #[test]
    fn suppress_orders_equal_scores_by_position() {
        // Three non-overlapping, equally-confident faces fed in a scrambled
        // order come back in a fixed one — top-to-bottom, then left-to-right.
        let ordered = |input: Vec<RawFace>| {
            suppress(input, NMS_IOU)
                .into_iter()
                .map(|face| (face.x as i32, face.y as i32))
                .collect::<Vec<_>>()
        };
        let expected = vec![(0, 0), (30, 0), (0, 30)];
        assert_eq!(
            ordered(vec![
                raw(0.0, 30.0, 10.0, 0.9),
                raw(30.0, 0.0, 10.0, 0.9),
                raw(0.0, 0.0, 10.0, 0.9),
            ]),
            expected
        );
        assert_eq!(
            ordered(vec![
                raw(0.0, 0.0, 10.0, 0.9),
                raw(0.0, 30.0, 10.0, 0.9),
                raw(30.0, 0.0, 10.0, 0.9),
            ]),
            expected
        );
    }

    #[test]
    fn similarity_transform_maps_the_reference_onto_itself() {
        let identity = similarity_transform(&REFERENCE_LANDMARKS).expect("non-degenerate");
        assert!((identity[0] - 1.0).abs() < 1e-4, "{identity:?}");
        assert!(identity[1].abs() < 1e-4);
        assert!(identity[2].abs() < 1e-3);
        assert!(identity[3].abs() < 1e-4);
        assert!((identity[4] - 1.0).abs() < 1e-4);
        assert!(identity[5].abs() < 1e-3);
    }

    #[test]
    fn similarity_transform_undoes_a_scale_rotation_and_shift() {
        // Take the reference, rotate 30 degrees, scale 2x, shift — the fit must
        // recover the inverse and land the points back on the reference.
        let (sin, cos) = (30.0_f32.to_radians().sin(), 30.0_f32.to_radians().cos());
        let mut moved = [[0.0_f32; 2]; 5];
        for (slot, point) in moved.iter_mut().zip(REFERENCE_LANDMARKS.iter()) {
            slot[0] = 2.0 * (cos * point[0] - sin * point[1]) + 17.0;
            slot[1] = 2.0 * (sin * point[0] + cos * point[1]) - 9.0;
        }
        let matrix = similarity_transform(&moved).expect("non-degenerate");
        for (point, reference) in moved.iter().zip(REFERENCE_LANDMARKS.iter()) {
            let x = matrix[0] * point[0] + matrix[1] * point[1] + matrix[2];
            let y = matrix[3] * point[0] + matrix[4] * point[1] + matrix[5];
            assert!((x - reference[0]).abs() < 1e-2, "{x} vs {}", reference[0]);
            assert!((y - reference[1]).abs() < 1e-2, "{y} vs {}", reference[1]);
        }
    }

    #[test]
    fn similarity_transform_declines_a_degenerate_landmark_set() {
        assert!(similarity_transform(&[[5.0, 5.0]; 5]).is_none());
    }

    #[test]
    fn invert_similarity_round_trips() {
        let forward = [2.0, -1.0, 7.0, 1.0, 2.0, -3.0];
        let inverse = invert_similarity(forward).expect("invertible");
        for (x, y) in [(0.0_f32, 0.0_f32), (13.0, -4.0), (100.5, 60.25)] {
            let fx = forward[0] * x + forward[1] * y + forward[2];
            let fy = forward[3] * x + forward[4] * y + forward[5];
            let bx = inverse[0] * fx + inverse[1] * fy + inverse[2];
            let by = inverse[3] * fx + inverse[4] * fy + inverse[5];
            assert!((bx - x).abs() < 1e-3, "{bx} vs {x}");
            assert!((by - y).abs() < 1e-3, "{by} vs {y}");
        }
        assert!(invert_similarity([0.0; 6]).is_none());
    }

    #[test]
    fn warp_crop_resamples_into_a_bgr_plane() {
        // Identity transform over a solid image: every sampled pixel is the
        // source colour, laid out BGR-planar.
        let image = RgbImage::from_pixel(200, 200, image::Rgb([200, 100, 50]));
        let data = warp_crop(&image, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).expect("invertible");
        let plane = (CROP_SIZE * CROP_SIZE) as usize;
        assert_eq!(data.len(), 3 * plane);
        for index in [0, plane / 2, plane - 1] {
            assert!((data[index] - 50.0).abs() < 1e-3);
            assert!((data[plane + index] - 100.0).abs() < 1e-3);
            assert!((data[2 * plane + index] - 200.0).abs() < 1e-3);
        }
        // Shifted entirely off the source: the border is black, not an error.
        let off = warp_crop(&image, [1.0, 0.0, -1000.0, 0.0, 1.0, -1000.0]).expect("invertible");
        assert!(off.iter().all(|value| *value == 0.0));
    }

    /// The end-to-end path, gated on a staged model pair exactly like
    /// `detector.rs`'s live test: with `LLM_INDEX_VISION_MODELS` unset, or the
    /// pair absent under it, this skips and CI stays green with no 38 MB
    /// download.
    ///
    /// The fixture is `tests/fixtures/faces-two.jpg` — the SAME public-domain
    /// NASA portrait (`astronaut.png`, Eileen Collins, distributed by
    /// scikit-image) composited twice at two scales on one canvas. Two faces
    /// makes it test what one cannot: that `face_index` ordering is real, that
    /// the per-face crop is actually per-face, and — because both faces are the
    /// same person — that the embeddings carry identity rather than noise. A
    /// wholly synthetic drawn face was tried first and YuNet does not fire on
    /// one (measured: no detection at thresholds down to 0.3), which is why a
    /// photograph is in the tree at all.
    ///
    /// Measured through THIS code path when the fixture was pinned (2026-07-26):
    /// boxes `(44, 35, 77x96)` @ `0.9505` and `(199, 48, 56x68)` @ `0.9393`
    /// against the `0.9` default gate, cosine `0.9597` between the two crops.
    /// The same fixture through OpenCV's own `FaceDetectorYN`/`FaceRecognizerSF`
    /// gives `0.9502`/`0.9393` and `0.9612` — i.e. this hand-written decode,
    /// alignment and warp agree with the reference implementation, which is the
    /// claim that could not be made by reading the code.
    #[test]
    fn faces_are_detected_embedded_and_byte_stable_when_models_present() {
        let Ok(dir) = std::env::var("LLM_INDEX_VISION_MODELS") else {
            eprintln!("skipping faces live test: LLM_INDEX_VISION_MODELS unset");
            return;
        };
        let models_dir = Path::new(&dir);
        if !available(models_dir) {
            eprintln!("skipping faces live test: yunet/sface not staged under {dir}");
            return;
        }
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/faces-two.jpg");
        let image = image::open(&fixture).expect("decoding the face fixture");
        let cfg = VisionConfig::default();

        let mut first = VisionResult::default();
        fill(&image, models_dir, &cfg, &mut first).expect("face extraction");
        assert_eq!(first.faces_model.as_deref(), Some(FACE_MODEL_ID));
        assert_eq!(first.faces.len(), 2, "{:?}", first.faces);

        // Ordered best-first, both clearing the default gate, both plausibly
        // sized inside a 300x160 canvas.
        assert!(first.faces[0].quality >= first.faces[1].quality);
        for face in &first.faces {
            assert!(face.quality >= cfg.face_score, "{face:?}");
            assert!(
                face.width >= MIN_FACE_PX && face.height >= MIN_FACE_PX,
                "{face:?}"
            );
            assert!(face.x >= 0 && face.y >= 0, "{face:?}");
            assert!(face.frame.is_none(), "a still has no keyframe ordinal");
            let embedding = face.embedding.as_ref().expect("embedded");
            assert_eq!(embedding.len(), EMBEDDING_DIMS);
            assert!(
                embedding.iter().any(|value| value.abs() > 1e-6),
                "an all-zero embedding means the crop never reached the model"
            );
        }
        // The two crops are the same person at two scales, so their vectors must
        // be close. A pipeline that mixed up crops, skipped alignment, or fed
        // the wrong channel order does not clear this.
        let cosine = |left: &[f32], right: &[f32]| {
            let dot: f32 = left.iter().zip(right).map(|(a, b)| a * b).sum();
            let norm = |vector: &[f32]| vector.iter().map(|v| v * v).sum::<f32>().sqrt();
            dot / (norm(left) * norm(right))
        };
        let similarity = cosine(
            first.faces[0].embedding.as_ref().unwrap(),
            first.faces[1].embedding.as_ref().unwrap(),
        );
        assert!(
            similarity > 0.9,
            "same person, two scales: cosine {similarity}"
        );

        // Determinism: the same bytes through the same models again produce the
        // same faces, box for box and float for float.
        let mut second = VisionResult::default();
        fill(&image, models_dir, &cfg, &mut second).expect("face extraction");
        assert_eq!(first.faces, second.faces);
    }

    /// A raised gate is a gate: the same fixture at a threshold above both
    /// measured scores yields a SCAN with no faces — proving the stamp records
    /// "looked, found nothing" rather than "never looked".
    #[test]
    fn a_gate_above_every_score_records_a_scan_with_no_faces() {
        let Ok(dir) = std::env::var("LLM_INDEX_VISION_MODELS") else {
            eprintln!("skipping faces threshold test: LLM_INDEX_VISION_MODELS unset");
            return;
        };
        let models_dir = Path::new(&dir);
        if !available(models_dir) {
            eprintln!("skipping faces threshold test: yunet/sface not staged under {dir}");
            return;
        }
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/faces-two.jpg");
        let image = image::open(&fixture).expect("decoding the face fixture");
        let cfg = VisionConfig {
            face_score: 0.99,
            ..Default::default()
        };
        let mut result = VisionResult::default();
        fill(&image, models_dir, &cfg, &mut result).expect("face extraction");
        assert!(result.faces.is_empty(), "{:?}", result.faces);
        assert_eq!(result.faces_model.as_deref(), Some(FACE_MODEL_ID));
    }

    #[test]
    fn sample_bilinear_interpolates_and_blackens_outside() {
        let mut image = RgbImage::new(2, 1);
        image.put_pixel(0, 0, image::Rgb([0, 0, 0]));
        image.put_pixel(1, 0, image::Rgb([100, 100, 100]));
        let middle = sample_bilinear(&image, 0.5, 0.0);
        assert!((middle[0] - 50.0).abs() < 1e-3, "{middle:?}");
        assert_eq!(sample_bilinear(&image, -5.0, 0.0), [0.0; 3]);
        assert_eq!(sample_bilinear(&image, 0.0, 9.0), [0.0; 3]);
    }
}
