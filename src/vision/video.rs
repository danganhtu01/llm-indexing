//! Video (V4): ffmpeg scene-change keyframes -> per-frame tags -> aggregation,
//! merged with the existing whisper transcript.
//!
//! Owned by the V4 worker. Keyframes are pulled with a fixed ffmpeg argv
//! (`-nostdin`, explicit filters) into a fresh tempdir that is cleaned up on
//! every path, run through the shared per-image orchestrator
//! ([`super::VisionAnalyzer::analyze_image`]), then aggregated into a single
//! [`VisionResult`]. The whisper transcript is merged one level up: the
//! pipeline extracts the transcript into the file's content and appends this
//! result's [`VisionResult::content_block`] beside it (see
//! `pipeline::process`), so there is nothing transcript-shaped to thread
//! through here.
//!
//! INTERFACE FLAG (for V1): the spec says video keyframes "run through the
//! image tier pipeline (call the orchestrator's per-image path)", but
//! `analyze` only receives `models_dir` + `cfg`, not the `VisionAnalyzer`
//! itself. We reconstruct an analyzer from those two fields (they are exactly
//! what `VisionAnalyzer` holds) to call `analyze_image`. Cleaner would be for
//! `mod.rs` to pass `&self` (or an `&dyn Fn(&Path, VisionMode) -> VisionResult`
//! seam) into `video::analyze`; adapted here to avoid touching V1-owned files.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use super::types::{ObjectDetection, TagScore, VisionMode, VisionResult};
use crate::config::VisionConfig;

/// Prefix on the error string a per-file ffmpeg/ffprobe timeout produces, so
/// [`analyze`] can tell a pathological clip (record the tier, don't retry every
/// resume) from an environment failure such as a missing ffmpeg (retry later).
const VIDEO_TIMEOUT: &str = "vision-timeout";

/// Scene-change score above which ffmpeg's `select` filter emits a keyframe.
const SCENE_THRESHOLD: f32 = 0.3;
/// If scene detection yields fewer than this many frames, fall back to a
/// fixed-interval sample so short/low-motion clips still get coverage.
const MIN_SCENE_FRAMES: usize = 3;
/// Hard ceiling on how many scene frames ffmpeg writes before we sample down to
/// `max_frames`, so a long busy video cannot spill thousands of PNGs into the
/// tempdir.
const SCENE_HARD_CAP: usize = 240;

/// Analyse a video: scene-change keyframes -> per-frame image pipeline ->
/// aggregated objects/tags/embedding + frame count.
///
/// Never panics: a missing/failing ffmpeg or an unreadable clip records an error
/// on the result and returns rather than propagating. Environment failures
/// (ffmpeg absent, tempdir) leave `mode` at `off` so a later `--resume` retries
/// once the host is fixed; genuine per-file failures (a pathological clip that
/// times out, no decodable frames) record the requested tier so they are not
/// re-attempted every resume.
pub(super) fn analyze(
    path: &Path,
    models_dir: &Path,
    cfg: &VisionConfig,
    mode: VisionMode,
    faces_ready: bool,
) -> VisionResult {
    let started = Instant::now();
    let mut result = VisionResult::default();

    // A single fresh tempdir for both extraction passes; TempDir cleans itself
    // on drop, covering every early return below.
    let temp = match tempfile::tempdir() {
        Ok(temp) => temp,
        Err(error) => {
            // Host/environment failure — leave `mode` at off so resume retries.
            result.error = Some(format!("vision video tempdir: {error}"));
            result.elapsed_ms = Some(started.elapsed().as_millis() as u64);
            return result;
        }
    };

    let frames = match extract_keyframes(path, temp.path(), cfg.max_frames, cfg.timeout_secs) {
        Ok(frames) => frames,
        Err(error) => {
            // A per-file timeout records the tier (don't burn the timeout on the
            // same bad clip every resume); a missing/unspawnable ffmpeg leaves
            // `mode` at off so resume retries once ffmpeg is installed.
            if error.starts_with(VIDEO_TIMEOUT) {
                result.mode = mode;
            }
            result.error = Some(error);
            result.elapsed_ms = Some(started.elapsed().as_millis() as u64);
            return result;
        }
    };

    if frames.is_empty() {
        // ffmpeg ran but produced nothing decodable — a per-file property.
        result.mode = mode;
        result.error = Some("video-no-frames".into());
        result.elapsed_ms = Some(started.elapsed().as_millis() as u64);
        return result;
    }

    // Frames run through the shared per-image pipeline. Cap the per-frame tier
    // at `tags`: object/tag/embedding aggregation is the point of video vision,
    // and per-frame Florence-2 captioning (captions tier) would be pathological
    // across `max_frames` frames.
    let analyzer = super::VisionAnalyzer {
        cfg: cfg.clone(),
        models_dir: models_dir.to_path_buf(),
        faces_ready,
    };
    let frame_target = mode.min(VisionMode::Tags);
    let per_frame: Vec<VisionResult> = frames
        .iter()
        .map(|frame| analyzer.analyze_image(frame, frame_target))
        .collect();

    aggregate_into(&mut result, &per_frame, cfg.tag_top_k, cfg.max_faces);
    result.frames = Some(per_frame.len());
    // Record the full requested `mode` only when every frame reached the
    // per-frame target tier; if a frame fell short (e.g. tag models missing, so
    // it downgraded to `meta`), record that lower achieved tier so a later
    // resume retries the video rather than treating it as done.
    let achieved = per_frame
        .iter()
        .map(|frame| frame.mode)
        .min()
        .unwrap_or(frame_target);
    result.mode = if achieved >= frame_target {
        mode
    } else {
        achieved
    };
    // Surface the first per-frame error (e.g. missing tag models) without
    // dropping the frames we did extract.
    if result.error.is_none() {
        result.error = per_frame.iter().find_map(|frame| frame.error.clone());
    }
    result.elapsed_ms = Some(started.elapsed().as_millis() as u64);
    result
}

/// Fold per-frame image results into the aggregate video result: representative
/// dimensions, union of objects (summed counts), union of tags (best score),
/// a mean-pooled CLIP embedding, and the concatenated per-keyframe faces.
fn aggregate_into(
    result: &mut VisionResult,
    frames: &[VisionResult],
    tag_top_k: usize,
    max_faces: usize,
) {
    if let Some(first) = frames.iter().find(|frame| frame.width.is_some()) {
        result.width = first.width;
        result.height = first.height;
    }
    result.objects = aggregate_objects(frames);
    result.tags = aggregate_tags(frames, tag_top_k);
    if let Some((embedding, model, dimensions)) = mean_pool(frames) {
        result.embedding = Some(embedding);
        result.embedding_model = Some(model);
        result.dimensions = Some(dimensions);
    }
    aggregate_faces(result, frames, max_faces);
}

/// Concatenate the faces found in each keyframe, stamping the keyframe ordinal
/// so a box means something: it is in THAT frame's pixel space, not the video's.
///
/// Deliberately a concatenation, not a de-duplication — one person walking
/// through five keyframes yields five faces here. Deciding that those five are
/// one person is cross-file/cross-frame identity work, which is the app's job
/// (plan §6 F2); the engine's job is per-file facts. `max_faces` bounds the
/// whole file rather than each frame, so a crowd scene cannot multiply by
/// `max_frames`.
///
/// `faces_model` is carried up from the frames so the video row records that it
/// WAS scanned even when no keyframe held a face — the same
/// scanned-versus-never-scanned distinction stills rely on.
fn aggregate_faces(result: &mut VisionResult, frames: &[VisionResult], max_faces: usize) {
    result.faces_model = frames.iter().find_map(|frame| frame.faces_model.clone());
    if result.faces_model.is_none() {
        return;
    }
    let mut faces = Vec::new();
    for (ordinal, frame) in frames.iter().enumerate() {
        for face in &frame.faces {
            if faces.len() >= max_faces {
                result.faces = faces;
                return;
            }
            let mut face = face.clone();
            face.frame = Some(ordinal as u32);
            faces.push(face);
        }
    }
    result.faces = faces;
}

/// Union object detections across frames: sum counts per label, keep the peak
/// confidence, and order by count (desc) then label for determinism.
fn aggregate_objects(frames: &[VisionResult]) -> Vec<ObjectDetection> {
    let mut by_label: BTreeMap<&str, (u32, f32)> = BTreeMap::new();
    for object in frames.iter().flat_map(|frame| &frame.objects) {
        let entry = by_label.entry(&object.label).or_insert((0, 0.0));
        entry.0 = entry.0.saturating_add(object.count);
        entry.1 = entry.1.max(object.max_conf);
    }
    let mut objects: Vec<ObjectDetection> = by_label
        .into_iter()
        .map(|(label, (count, max_conf))| ObjectDetection {
            label: label.to_string(),
            count,
            max_conf,
        })
        .collect();
    objects.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.label.cmp(&b.label)));
    objects
}

/// Union zero-shot tags across frames keeping each tag's best score, ordered by
/// score (desc) then tag, capped at `top_k`.
fn aggregate_tags(frames: &[VisionResult], top_k: usize) -> Vec<TagScore> {
    let mut by_tag: BTreeMap<&str, f32> = BTreeMap::new();
    for tag in frames.iter().flat_map(|frame| &frame.tags) {
        let entry = by_tag.entry(&tag.tag).or_insert(f32::MIN);
        *entry = entry.max(tag.score);
    }
    let mut tags: Vec<TagScore> = by_tag
        .into_iter()
        .map(|(tag, score)| TagScore {
            tag: tag.to_string(),
            score,
        })
        .collect();
    tags.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.tag.cmp(&b.tag))
    });
    tags.truncate(top_k);
    tags
}

/// Mean-pool the per-frame CLIP embeddings (element-wise average over the
/// frames that produced one of the modal dimension). Returns the pooled vector,
/// the embedding model name, and its dimension.
fn mean_pool(frames: &[VisionResult]) -> Option<(Vec<f32>, String, usize)> {
    // Pick the modal (most common) embedding dimension so a stray malformed
    // vector cannot corrupt the average.
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for frame in frames {
        if let Some(embedding) = &frame.embedding {
            *counts.entry(embedding.len()).or_insert(0) += 1;
        }
    }
    let dimensions = counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)))
        .map(|(len, _)| len)
        .filter(|len| *len > 0)?;

    let mut sum = vec![0.0f64; dimensions];
    let mut model = None;
    let mut used = 0usize;
    for frame in frames {
        match &frame.embedding {
            Some(embedding) if embedding.len() == dimensions => {
                for (slot, value) in sum.iter_mut().zip(embedding) {
                    *slot += f64::from(*value);
                }
                used += 1;
                if model.is_none() {
                    model = frame.embedding_model.clone();
                }
            }
            _ => {}
        }
    }
    if used == 0 {
        return None;
    }
    let pooled = sum
        .into_iter()
        .map(|value| (value / used as f64) as f32)
        .collect();
    Some((
        pooled,
        model.unwrap_or_else(|| "clip".to_string()),
        dimensions,
    ))
}

/// Extract keyframes into `dir`: scene-change detection first, fixed-interval
/// fallback when it yields fewer than [`MIN_SCENE_FRAMES`], sampled down to
/// `max_frames`. Returns the sorted PNG paths, or an `Err` string when ffmpeg
/// is unavailable.
fn extract_keyframes(
    path: &Path,
    dir: &Path,
    max_frames: usize,
    timeout_secs: u64,
) -> Result<Vec<PathBuf>, String> {
    let scene = run_ffmpeg_frames(
        path,
        dir,
        "scene",
        &format!("select='gt(scene\\,{SCENE_THRESHOLD})',scale='min(1280,iw)':-2:flags=bicubic"),
        Some("vfr"),
        SCENE_HARD_CAP,
        timeout_secs,
    )?;

    let frames = if scene.len() >= MIN_SCENE_FRAMES {
        scene
    } else {
        // Fixed-interval fallback: spread `max_frames` samples across the clip
        // using its probed duration, or one frame/second when duration is
        // unknown. Falls back cleanly for very short synthesized clips.
        let filter = match probe_duration(path, timeout_secs).filter(|d| *d > 0.0) {
            Some(duration) => {
                let interval = (duration / (max_frames.max(1) as f64)).max(0.001);
                format!("fps=1/{interval:.6},scale='min(1280,iw)':-2:flags=bicubic")
            }
            None => "fps=1,scale='min(1280,iw)':-2:flags=bicubic".to_string(),
        };
        let interval = run_ffmpeg_frames(
            path,
            dir,
            "interval",
            &filter,
            None,
            max_frames,
            timeout_secs,
        )?;
        // Prefer whichever pass gave more coverage.
        if interval.len() >= scene.len() {
            interval
        } else {
            scene
        }
    };

    Ok(cap_evenly(frames, max_frames))
}

/// Wait for `child` up to `timeout`, killing and reaping it on expiry. Returns
/// the exit status, or `None` when the timeout fired. Polls rather than blocking
/// so a pathological ffmpeg/ffprobe cannot wedge the worker forever (the spec's
/// per-file `vision_timeout_secs`); child stdio is redirected to null/piped by
/// the caller, so there is no pipe to drain here.
fn wait_bounded(child: &mut Child, timeout: Duration) -> std::io::Result<Option<ExitStatus>> {
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Run ffmpeg with a fixed argv to write `<dir>/<prefix>-%06d.png` and return
/// the resulting frame paths, sorted, bounded by `timeout_secs`. `Err` when the
/// ffmpeg binary is missing/unspawnable or the run exceeds the timeout (a
/// [`VIDEO_TIMEOUT`]-prefixed message); a non-zero exit yields an empty list (so
/// the fallback can engage). stdout/stderr are dropped (frames land on disk), so
/// the bounded wait never risks a pipe-fill deadlock.
fn run_ffmpeg_frames(
    path: &Path,
    dir: &Path,
    prefix: &str,
    filter: &str,
    fps_mode: Option<&str>,
    limit: usize,
    timeout_secs: u64,
) -> Result<Vec<PathBuf>, String> {
    let pattern = dir.join(format!("{prefix}-%06d.png"));
    let mut command = Command::new("ffmpeg");
    command.args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i"]);
    command.arg(path);
    command.args(["-vf", filter]);
    if let Some(mode) = fps_mode {
        command.args(["-fps_mode", mode]);
    }
    command.args(["-frames:v", &limit.max(1).to_string()]);
    command.arg(&pattern);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err("ffmpeg not found; video vision requires ffmpeg on PATH".to_string());
        }
        Err(error) => return Err(format!("ffmpeg keyframe extraction failed: {error}")),
    };
    match wait_bounded(&mut child, Duration::from_secs(timeout_secs.max(1))) {
        Ok(Some(status)) if status.success() => Ok(collect_frames(dir, prefix)),
        Ok(Some(_)) => Ok(Vec::new()),
        Ok(None) => Err(format!(
            "{VIDEO_TIMEOUT}: ffmpeg exceeded {timeout_secs}s on {}",
            path.display()
        )),
        Err(error) => Err(format!("ffmpeg keyframe extraction failed: {error}")),
    }
}

/// Collect and sort the `<prefix>-*.png` frames ffmpeg wrote into `dir`.
fn collect_frames(dir: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut frames: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|entry| {
            entry
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(prefix) && name.ends_with(".png"))
        })
        .collect();
    frames.sort();
    frames
}

/// Probe a clip's duration (seconds) with ffprobe, if it is available, bounded by
/// `timeout_secs`. Returns `None` when ffprobe is absent, times out, or the
/// output is unparsable — callers fall back to a fixed frame rate. stderr is
/// dropped and stdout (a single number) is read only after exit, so the bounded
/// wait cannot deadlock on a full pipe.
fn probe_duration(path: &Path, timeout_secs: u64) -> Option<f64> {
    let mut child = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    match wait_bounded(&mut child, Duration::from_secs(timeout_secs.max(1))) {
        Ok(Some(status)) if status.success() => {}
        _ => return None,
    }
    let mut stdout = String::new();
    child.stdout.take()?.read_to_string(&mut stdout).ok()?;
    stdout.trim().parse().ok()
}

/// Sample `items` down to at most `max` entries, evenly spaced across the input
/// so coverage spans the whole clip rather than truncating to the front.
fn cap_evenly<T>(items: Vec<T>, max: usize) -> Vec<T> {
    if max == 0 {
        return Vec::new();
    }
    let len = items.len();
    if len <= max {
        return items;
    }
    // Monotonic, distinct indices 0..max mapped across 0..len.
    let keep: Vec<usize> = (0..max).map(|k| k * len / max).collect();
    items
        .into_iter()
        .enumerate()
        .filter_map(|(index, item)| keep.contains(&index).then_some(item))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(name: &str) -> PathBuf {
        PathBuf::from(name)
    }

    fn frames(names: &[&str]) -> Vec<PathBuf> {
        names.iter().map(|name| frame(name)).collect()
    }

    // --- keyframe selection / capping (mocked frame lists) ---------------

    #[test]
    fn cap_evenly_keeps_all_when_under_cap() {
        let input = frames(&["a", "b", "c"]);
        assert_eq!(cap_evenly(input.clone(), 12), input);
    }

    #[test]
    fn cap_evenly_samples_spread_across_input() {
        let input = frames(&["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]);
        let out = cap_evenly(input, 4);
        assert_eq!(out.len(), 4);
        // Evenly spaced indices 0,2,5,7 (k*10/4) — spans start to end, distinct.
        assert_eq!(out, frames(&["0", "2", "5", "7"]));
    }

    #[test]
    fn cap_evenly_zero_max_is_empty() {
        assert!(cap_evenly(frames(&["a", "b"]), 0).is_empty());
    }

    #[test]
    fn cap_evenly_indices_are_distinct_and_monotonic() {
        let input: Vec<PathBuf> = (0..100).map(|n| PathBuf::from(n.to_string())).collect();
        let out = cap_evenly(input, 12);
        assert_eq!(out.len(), 12);
        let nums: Vec<i32> = out
            .iter()
            .map(|p| p.to_str().unwrap().parse().unwrap())
            .collect();
        assert!(nums.windows(2).all(|w| w[0] < w[1]), "strictly increasing");
        assert_eq!(nums[0], 0);
    }

    /// Mirrors the scene-vs-fallback choice in `extract_keyframes` over mocked
    /// lists so the branch is covered without ffmpeg.
    fn choose(scene: Vec<PathBuf>, fallback: Vec<PathBuf>, max: usize) -> Vec<PathBuf> {
        let chosen = if scene.len() >= MIN_SCENE_FRAMES {
            scene
        } else if fallback.len() >= scene.len() {
            fallback
        } else {
            scene
        };
        cap_evenly(chosen, max)
    }

    #[test]
    fn enough_scene_frames_uses_scene() {
        let scene = frames(&["s0", "s1", "s2", "s3"]);
        let fallback = frames(&["f0", "f1", "f2", "f3", "f4"]);
        assert_eq!(choose(scene.clone(), fallback, 12), scene);
    }

    #[test]
    fn too_few_scene_frames_falls_back() {
        let scene = frames(&["s0", "s1"]);
        let fallback = frames(&["f0", "f1", "f2", "f3"]);
        assert_eq!(choose(scene, fallback.clone(), 12), fallback);
    }

    #[test]
    fn fallback_below_scene_keeps_scene() {
        // Scene under the min but fallback produced even less -> keep scene.
        let scene = frames(&["s0", "s1"]);
        let fallback = frames(&["f0"]);
        assert_eq!(choose(scene.clone(), fallback, 12), scene);
    }

    // --- aggregation ------------------------------------------------------

    fn result_with(objects: Vec<ObjectDetection>, tags: Vec<TagScore>) -> VisionResult {
        VisionResult {
            objects,
            tags,
            ..Default::default()
        }
    }

    #[test]
    fn aggregate_objects_unions_and_sums_counts() {
        let a = result_with(
            vec![
                ObjectDetection {
                    label: "person".into(),
                    count: 2,
                    max_conf: 0.7,
                },
                ObjectDetection {
                    label: "dog".into(),
                    count: 1,
                    max_conf: 0.6,
                },
            ],
            vec![],
        );
        let b = result_with(
            vec![ObjectDetection {
                label: "person".into(),
                count: 3,
                max_conf: 0.9,
            }],
            vec![],
        );
        let objects = aggregate_objects(&[a, b]);
        assert_eq!(objects.len(), 2);
        // person: 2+3=5, sorted first by count.
        assert_eq!(objects[0].label, "person");
        assert_eq!(objects[0].count, 5);
        assert!((objects[0].max_conf - 0.9).abs() < 1e-6);
        assert_eq!(objects[1].label, "dog");
        assert_eq!(objects[1].count, 1);
    }

    #[test]
    fn aggregate_tags_keeps_best_score_and_caps_top_k() {
        let a = result_with(
            vec![],
            vec![
                TagScore {
                    tag: "beach".into(),
                    score: 0.4,
                },
                TagScore {
                    tag: "sunset".into(),
                    score: 0.2,
                },
            ],
        );
        let b = result_with(
            vec![],
            vec![
                TagScore {
                    tag: "beach".into(),
                    score: 0.9,
                },
                TagScore {
                    tag: "dog".into(),
                    score: 0.5,
                },
            ],
        );
        let tags = aggregate_tags(&[a, b], 2);
        assert_eq!(tags.len(), 2, "capped at top_k");
        assert_eq!(tags[0].tag, "beach");
        assert!((tags[0].score - 0.9).abs() < 1e-6, "kept the best score");
        assert_eq!(tags[1].tag, "dog");
    }

    #[test]
    fn aggregate_tags_empty_when_no_tags() {
        assert!(aggregate_tags(&[result_with(vec![], vec![])], 8).is_empty());
    }

    #[test]
    fn mean_pool_averages_matching_dimensions() {
        let a = VisionResult {
            embedding: Some(vec![0.0, 2.0, 4.0]),
            embedding_model: Some("clip-vit-b32".into()),
            dimensions: Some(3),
            ..Default::default()
        };
        let b = VisionResult {
            embedding: Some(vec![2.0, 4.0, 8.0]),
            embedding_model: Some("clip-vit-b32".into()),
            dimensions: Some(3),
            ..Default::default()
        };
        let (pooled, model, dims) = mean_pool(&[a, b]).unwrap();
        assert_eq!(dims, 3);
        assert_eq!(model, "clip-vit-b32");
        assert_eq!(pooled, vec![1.0, 3.0, 6.0]);
    }

    #[test]
    fn mean_pool_ignores_off_dimension_vectors() {
        // Modal dimension is 3 (two frames); the stray 2-dim vector is dropped.
        let good_a = VisionResult {
            embedding: Some(vec![1.0, 1.0, 1.0]),
            ..Default::default()
        };
        let good_b = VisionResult {
            embedding: Some(vec![3.0, 3.0, 3.0]),
            ..Default::default()
        };
        let stray = VisionResult {
            embedding: Some(vec![9.0, 9.0]),
            ..Default::default()
        };
        let (pooled, _, dims) = mean_pool(&[good_a, stray, good_b]).unwrap();
        assert_eq!(dims, 3);
        assert_eq!(pooled, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn mean_pool_none_without_embeddings() {
        assert!(mean_pool(&[VisionResult::default()]).is_none());
    }

    fn face(x: i32) -> crate::vision::types::FaceDetection {
        crate::vision::types::FaceDetection {
            x,
            y: 0,
            width: 40,
            height: 40,
            quality: 0.95,
            embedding: Some(vec![0.5; 4]),
            frame: None,
        }
    }

    fn scanned(faces: Vec<crate::vision::types::FaceDetection>) -> VisionResult {
        VisionResult {
            faces,
            faces_model: Some("yunet-sface".into()),
            ..Default::default()
        }
    }

    #[test]
    fn aggregate_faces_stamps_the_keyframe_each_face_came_from() {
        let frames = vec![
            scanned(vec![face(10)]),
            scanned(vec![]),
            scanned(vec![face(20), face(30)]),
        ];
        let mut result = VisionResult::default();
        aggregate_faces(&mut result, &frames, 20);
        assert_eq!(result.faces_model.as_deref(), Some("yunet-sface"));
        assert_eq!(
            result
                .faces
                .iter()
                .map(|face| (face.x, face.frame))
                .collect::<Vec<_>>(),
            vec![(10, Some(0)), (20, Some(2)), (30, Some(2))]
        );
    }

    #[test]
    fn aggregate_faces_caps_the_whole_file_not_each_frame() {
        let frames = vec![
            scanned(vec![face(1), face(2)]),
            scanned(vec![face(3), face(4)]),
        ];
        let mut result = VisionResult::default();
        aggregate_faces(&mut result, &frames, 3);
        assert_eq!(result.faces.len(), 3);
        assert_eq!(result.faces[2].frame, Some(1));
    }

    #[test]
    fn aggregate_faces_records_a_scan_that_found_nothing_but_not_an_unscanned_video() {
        // Every keyframe scanned, none held a face: the video is marked scanned
        // so a resume leaves it alone.
        let mut scanned_result = VisionResult::default();
        aggregate_faces(&mut scanned_result, &[scanned(vec![]), scanned(vec![])], 20);
        assert_eq!(scanned_result.faces_model.as_deref(), Some("yunet-sface"));
        assert!(scanned_result.faces.is_empty());

        // Faces off (or unstaged): no stamp, so the video stays eligible.
        let mut unscanned = VisionResult::default();
        aggregate_faces(&mut unscanned, &[VisionResult::default()], 20);
        assert_eq!(unscanned.faces_model, None);
        assert!(unscanned.faces.is_empty());
    }

    // --- end-to-end (gated: needs ffmpeg to synthesize + extract) ---------

    fn ffmpeg_available() -> bool {
        Command::new("ffmpeg")
            .args(["-hide_banner", "-version"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Synthesize a tiny clip (testsrc pattern) and run the real keyframe +
    /// per-frame pipeline at the `meta` tier (no model downloads needed).
    #[test]
    fn end_to_end_meta_over_synthesized_clip() {
        if !ffmpeg_available() {
            eprintln!("skipping: ffmpeg not on PATH");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let clip = dir.path().join("clip.mp4");
        let status = Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=2:size=320x240:rate=10",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:size=320x240:duration=2:rate=10",
                "-filter_complex",
                "[0:v][1:v]blend=all_mode=average",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&clip)
            .status()
            .unwrap();
        assert!(status.success(), "synthesizing test clip");

        let cfg = VisionConfig::default();
        let models = dir.path(); // unused at the meta tier
        let result = analyze(&clip, models, &cfg, VisionMode::Meta, false);

        assert_eq!(result.error, None, "meta-tier video should not error");
        assert_eq!(result.mode, VisionMode::Meta);
        let frames = result.frames.expect("frame count recorded");
        assert!(frames >= 1, "at least one keyframe analyzed, got {frames}");
        assert!(frames <= cfg.max_frames, "capped at max_frames");
        assert_eq!(result.width, Some(320));
        assert_eq!(result.height, Some(240));
        assert!(result.elapsed_ms.is_some());
        // Meta tier produces no models-backed objects/tags/embedding.
        assert!(result.objects.is_empty());
        assert!(result.tags.is_empty());
        assert!(result.embedding.is_none());
    }
}
