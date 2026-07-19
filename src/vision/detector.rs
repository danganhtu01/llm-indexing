//! Tags tier (V3): object detection via RF-DETR-Nano ONNX on `ort`.
//!
//! Per the VISION-SPEC AMENDMENT 2026-07-19 the detector is **RF-DETR-Nano**
//! (Apache-2.0), not Ultralytics YOLO (AGPL-3.0). RF-DETR is a DETR-style
//! detector: each of the N object queries yields one box + per-class logits, so
//! post-processing is a per-query sigmoid + threshold — **no NMS**. We only
//! store aggregated `{label, count, max_conf}`, so the decoded box coordinates
//! are never needed (no letterbox / coordinate mapping here).
//!
//! The ONNX session is shared process-wide behind a `Mutex`/`OnceLock`
//! (VISION-SPEC section 3). Model-dependent behaviour is only exercised when the
//! model file is present; the unit tests below cover the pure math and the
//! label mapping, and skip the live path when the model is absent.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use image::{imageops::FilterType, DynamicImage};
use ort::session::Session;
use ort::value::Tensor;

use crate::config::VisionConfig;
use crate::vision::types::{ObjectDetection, VisionResult};

/// Detector model file under `<data_dir>/vision`. The V1-owned `VISION_MODELS`
/// registry in `mod.rs` points its tags-tier entry here (Apache-2.0
/// RF-DETR-Nano) so the submit pre-flight and `fetch-data --vision` agree on the
/// filename.
pub const RFDETR_NANO_ONNX: &str = "rf-detr-nano.onnx";

/// Square input resolution RF-DETR-Nano expects (its published default; must be
/// divisible by 56). Confirmed 2026-07-19 against onnx-community/rfdetr_nano-ONNX
/// together with the ImageNet mean/std below.
const INPUT_SIZE: u32 = 384;

/// ImageNet normalization applied after scaling pixels to `[0, 1]` (RF-DETR's
/// standard preprocessing).
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// COCO-80 class names, in the canonical contiguous order.
const COCO_80: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// The 91-space COCO category id for each of the 80 contiguous classes above.
/// DETR-family heads that emit 90/91 logits index them by these ids (with gaps
/// for removed categories); this table maps such an index back to a name.
const COCO_90_IDS: [u16; 80] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
];

/// Map a raw class index to a COCO label, given the head's class count.
/// Heads with ~80 logits are contiguous; wider (90/91) heads index by COCO
/// category id. Unmapped indices (e.g. a background/no-object slot) yield
/// `None` and are dropped.
fn coco_label(class: usize, num_classes: usize) -> Option<&'static str> {
    if num_classes <= COCO_80.len() + 1 {
        return COCO_80.get(class).copied();
    }
    COCO_90_IDS
        .iter()
        .position(|&id| usize::from(id) == class)
        .and_then(|pos| COCO_80.get(pos).copied())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Preprocess an image into the model's NCHW `f32` input: resize to a square,
/// scale to `[0, 1]`, then apply ImageNet normalization. Layout is
/// channel-major (`[R plane, G plane, B plane]`), length `3 * size * size`.
fn preprocess(image: &DynamicImage, size: u32) -> Vec<f32> {
    let resized = image
        .resize_exact(size, size, FilterType::Triangle)
        .to_rgb8();
    let plane = (size * size) as usize;
    let mut data = vec![0.0_f32; 3 * plane];
    for (x, y, pixel) in resized.enumerate_pixels() {
        let offset = (y * size + x) as usize;
        for channel in 0..3 {
            let value = f32::from(pixel[channel]) / 255.0;
            data[channel * plane + offset] =
                (value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
        }
    }
    data
}

/// DETR-style post-processing over the `[N, num_classes]` logits (flattened
/// row-major): per query take the highest-scoring class after a sigmoid, keep
/// it when the score clears `conf`. No NMS — each query is one object. Pure and
/// model-free for unit testing.
fn decode(logits: &[f32], num_queries: usize, num_classes: usize, conf: f32) -> Vec<(usize, f32)> {
    if num_classes == 0 {
        return Vec::new();
    }
    let queries = num_queries.min(logits.len() / num_classes);
    let mut detections = Vec::new();
    for query in 0..queries {
        let row = &logits[query * num_classes..(query + 1) * num_classes];
        let mut best_class = 0;
        let mut best_score = 0.0_f32;
        for (class, &logit) in row.iter().enumerate() {
            let score = sigmoid(logit);
            if score > best_score {
                best_score = score;
                best_class = class;
            }
        }
        if best_score >= conf {
            detections.push((best_class, best_score));
        }
    }
    detections
}

/// Aggregate raw `(class, score)` detections into per-label `{count, max_conf}`,
/// dropping classes with no COCO label. Sorted by count (desc) then label, for a
/// stable, deterministic result.
fn aggregate(detections: &[(usize, f32)], num_classes: usize) -> Vec<ObjectDetection> {
    let mut tallies: HashMap<&'static str, (u32, f32)> = HashMap::new();
    for &(class, score) in detections {
        if let Some(label) = coco_label(class, num_classes) {
            let entry = tallies.entry(label).or_insert((0, 0.0));
            entry.0 += 1;
            if score > entry.1 {
                entry.1 = score;
            }
        }
    }
    let mut objects = tallies
        .into_iter()
        .map(|(label, (count, max_conf))| ObjectDetection {
            label: label.to_string(),
            count,
            max_conf,
        })
        .collect::<Vec<_>>();
    objects.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.label.cmp(&right.label))
    });
    objects
}

/// Load the RF-DETR ONNX session from `<models_dir>/rf-detr-nano.onnx`.
fn build_session(models_dir: &Path) -> Result<Mutex<Session>> {
    let path = models_dir.join(RFDETR_NANO_ONNX);
    anyhow::ensure!(
        path.is_file(),
        "object detector model not found at {} (run `llm-index fetch-data --vision`)",
        path.display()
    );
    // ONNX Runtime defaults to all cores intra-op. We keep that default: its
    // thread-config builder returns an error type that carries the builder for
    // recovery and is not `anyhow`-compatible, and detector calls are already
    // serialized by the shared `Mutex`.
    let session = Session::builder()
        .context("creating detector session builder")?
        .commit_from_file(&path)
        .with_context(|| format!("loading detector model {}", path.display()))?;
    Ok(Mutex::new(session))
}

/// The cached process-wide detector session, initialised on first use. Only a
/// *successful* build is cached; a transient failure (a model file momentarily
/// truncated during a re-fetch) caches nothing, so the next job retries rather
/// than the resident `serve` process being poisoned until restart. A dedicated
/// init lock serializes construction so the rayon workers don't all build at once.
fn session(models_dir: &Path) -> Result<&'static Mutex<Session>> {
    static SESSION: OnceLock<Mutex<Session>> = OnceLock::new();
    static INIT: Mutex<()> = Mutex::new(());
    if let Some(session) = SESSION.get() {
        return Ok(session);
    }
    let _guard = INIT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(session) = SESSION.get() {
        return Ok(session);
    }
    let session = build_session(models_dir).context("detector init failed")?;
    Ok(SESSION.get_or_init(|| session))
}

/// Detect objects and aggregate per-label counts into `out.objects`.
pub(crate) fn fill(
    image: &DynamicImage,
    models_dir: &Path,
    cfg: &VisionConfig,
    out: &mut VisionResult,
) -> Result<()> {
    let session = session(models_dir)?;
    let mut session = session
        .lock()
        .map_err(|_| anyhow::anyhow!("detector session mutex poisoned"))?;

    let input_name = session
        .inputs()
        .first()
        .context("detector model has no input")?
        .name()
        .to_string();
    let input = preprocess(image, INPUT_SIZE);
    let tensor = Tensor::from_array((
        [1_usize, 3, INPUT_SIZE as usize, INPUT_SIZE as usize],
        input,
    ))
    .context("building detector input tensor")?;

    let outputs = session
        .run(ort::inputs![input_name => tensor])
        .context("running detector inference")?;

    // RF-DETR emits a boxes tensor ([1, N, 4]) and a logits tensor
    // ([1, N, num_classes]); we only need the latter. Pick the 3-D f32 output
    // whose last dim is not 4.
    let mut logits: Option<(Vec<f32>, usize, usize)> = None;
    let keys = outputs.keys().map(str::to_string).collect::<Vec<_>>();
    for key in &keys {
        let Some(value) = outputs.get(key) else {
            continue;
        };
        let Ok((shape, data)) = value.try_extract_tensor::<f32>() else {
            continue;
        };
        let dims = shape.iter().copied().collect::<Vec<_>>();
        if dims.len() == 3 {
            let queries = dims[1].max(0) as usize;
            let classes = dims[2].max(0) as usize;
            if classes != 4 && classes > 0 && queries > 0 {
                logits = Some((data.to_vec(), queries, classes));
                break;
            }
        }
    }
    let (logits, num_queries, num_classes) =
        logits.context("detector produced no [1, N, classes] logits output")?;

    let detections = decode(&logits, num_queries, num_classes, cfg.detector_conf);
    out.objects = aggregate(&detections, num_classes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_is_monotonic_and_centered() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(6.0) > 0.99);
        assert!(sigmoid(-6.0) < 0.01);
        assert!(sigmoid(1.0) > sigmoid(0.0));
    }

    #[test]
    fn preprocess_shapes_and_normalizes() {
        // A solid mid-gray image: every normalized value is (0.5 - mean)/std.
        let gray = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            10,
            7,
            image::Rgb([128, 128, 128]),
        ));
        let data = preprocess(&gray, 32);
        assert_eq!(data.len(), 3 * 32 * 32);
        let plane = 32 * 32;
        for channel in 0..3 {
            let expected = (128.0 / 255.0 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
            // Check a couple of samples in each channel plane.
            for &idx in &[0usize, plane / 2, plane - 1] {
                assert!((data[channel * plane + idx] - expected).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn coco_label_maps_contiguous_and_id_spaces() {
        // Contiguous 80-class head.
        assert_eq!(coco_label(0, 80), Some("person"));
        assert_eq!(coco_label(2, 80), Some("car"));
        assert_eq!(coco_label(16, 80), Some("dog"));
        assert_eq!(coco_label(79, 80), Some("toothbrush"));
        assert_eq!(coco_label(80, 80), None);
        // 91-space head indexes by COCO category id.
        assert_eq!(coco_label(1, 91), Some("person"));
        assert_eq!(coco_label(3, 91), Some("car"));
        assert_eq!(coco_label(18, 91), Some("dog"));
        assert_eq!(coco_label(90, 91), Some("toothbrush"));
        // 12 is a gap in the 91-space id table -> unmapped.
        assert_eq!(coco_label(12, 91), None);
    }

    #[test]
    fn decode_keeps_confident_queries_and_picks_best_class() {
        // 3 queries x 3 classes (row-major logits). Large positive -> sigmoid≈1.
        let logits = [
            -5.0, 5.0, -5.0, // query 0 -> class 1, high
            -5.0, -5.0, -5.0, // query 1 -> all low, dropped
            2.0, -1.0, 4.0, // query 2 -> class 2, high
        ];
        let detections = decode(&logits, 3, 3, 0.5);
        assert_eq!(detections.len(), 2);
        assert_eq!(detections[0].0, 1);
        assert_eq!(detections[1].0, 2);
        assert!(detections.iter().all(|(_, score)| *score >= 0.5));
    }

    #[test]
    fn decode_handles_zero_classes_and_ragged_input() {
        assert!(decode(&[], 5, 0, 0.5).is_empty());
        // Only enough data for one full query; the second is not read.
        assert_eq!(decode(&[10.0, -10.0], 5, 2, 0.5).len(), 1);
    }

    #[test]
    fn aggregate_counts_and_keeps_max_conf_sorted() {
        // Two persons (class 0) and one dog (class 16), contiguous 80-space.
        let detections = [(0usize, 0.7_f32), (16, 0.8), (0, 0.9)];
        let objects = aggregate(&detections, 80);
        assert_eq!(objects.len(), 2);
        // Sorted by count desc: person(2) before dog(1).
        assert_eq!(objects[0].label, "person");
        assert_eq!(objects[0].count, 2);
        assert!((objects[0].max_conf - 0.9).abs() < 1e-6);
        assert_eq!(objects[1].label, "dog");
        assert_eq!(objects[1].count, 1);
    }

    #[test]
    fn aggregate_drops_unmapped_classes() {
        // Class 80 has no label in an 80-class head -> dropped entirely.
        assert!(aggregate(&[(80usize, 0.9_f32)], 80).is_empty());
    }

    /// Live inference path — runs only when a real model is present under the
    /// dir named by `LLM_INDEX_VISION_MODELS`; otherwise it skips (CI stays
    /// green with no download), per VISION-SPEC section 6.
    #[test]
    fn detector_runs_when_model_present() {
        let Ok(dir) = std::env::var("LLM_INDEX_VISION_MODELS") else {
            eprintln!("skipping detector live test: LLM_INDEX_VISION_MODELS unset");
            return;
        };
        let models_dir = Path::new(&dir);
        if !models_dir.join(RFDETR_NANO_ONNX).is_file() {
            eprintln!("skipping detector live test: {RFDETR_NANO_ONNX} absent");
            return;
        }
        let cfg = VisionConfig::default();
        let image = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            640,
            480,
            image::Rgb([120, 120, 120]),
        ));
        let mut out = VisionResult::default();
        fill(&image, models_dir, &cfg, &mut out).expect("detector inference");
        // A blank frame may legitimately yield no objects; we only assert the
        // path completed and produced well-formed rows. (Validated 2026-07-19
        // against onnx-community/rfdetr_nano-ONNX on ultralytics bus.jpg:
        // person x4 @0.95, bus x1 @0.95 — labels, decode, and preprocessing
        // all correct.)
        assert!(out.objects.iter().all(|object| object.count >= 1));
    }
}
