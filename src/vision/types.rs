use std::str::FromStr;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// Vision analysis tier. The ordering is meaningful — `off < meta < tags <
/// captions` — and each tier includes every tier below it. The `derive`d
/// `PartialOrd`/`Ord` follow declaration order, so keep the variants in
/// ascending-tier order.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize, ValueEnum,
)]
#[serde(rename_all = "lowercase")]
pub enum VisionMode {
    /// No vision analysis (default everywhere).
    #[default]
    Off,
    /// Pure-code metadata: dimensions, EXIF, perceptual hash, quality metrics.
    Meta,
    /// `meta` plus local CV models: CLIP zero-shot tags and RF-DETR-Nano object
    /// counts.
    Tags,
    /// `tags` plus a best-effort Florence-2 caption (opt-in, needs a big model).
    Captions,
}

impl VisionMode {
    /// The canonical lowercase token, as stored in `vision.mode` and accepted on
    /// the wire.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Meta => "meta",
            Self::Tags => "tags",
            Self::Captions => "captions",
        }
    }

    /// Every tier in ascending order — used to build human-readable error
    /// messages when validation rejects an unknown value.
    pub const ALL: &'static [VisionMode] = &[
        VisionMode::Off,
        VisionMode::Meta,
        VisionMode::Tags,
        VisionMode::Captions,
    ];

    /// Whether this tier needs downloaded model files (`tags` and `captions`).
    /// `off` and `meta` are pure code, so they never require a model.
    pub fn needs_models(&self) -> bool {
        matches!(self, Self::Tags | Self::Captions)
    }

    /// Whether running `self` covers everything `other` would produce — i.e.
    /// `self` is at least as high a tier as `other`.
    pub fn includes(&self, other: VisionMode) -> bool {
        *self >= other
    }
}

impl std::fmt::Display for VisionMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for VisionMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "meta" => Ok(Self::Meta),
            "tags" => Ok(Self::Tags),
            "captions" => Ok(Self::Captions),
            other => Err(format!(
                "unknown vision tier '{other}' (expected off, meta, tags, or captions)"
            )),
        }
    }
}

/// One WGS84 coordinate pulled from EXIF GPS tags.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GpsCoord {
    pub lat: f64,
    pub lon: f64,
}

/// Camera/context metadata parsed from EXIF. `camera`, `datetime` and `gps`
/// drive the searchable `[vision]` block; anything else a tier wants to keep is
/// flattened into `fields` so `exif_json` can carry it without a schema change.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExifInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub datetime: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gps: Option<GpsCoord>,
    /// Any additional EXIF tags a tier keeps; flattened into `exif_json`.
    #[serde(flatten)]
    pub fields: serde_json::Map<String, serde_json::Value>,
}

impl ExifInfo {
    /// The `camera: …` line for the searchable vision block, or `None` when no
    /// human-meaningful camera/date/GPS metadata was parsed.
    pub fn summary_line(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(camera) = self.camera.as_deref().filter(|value| !value.is_empty()) {
            parts.push(camera.to_string());
        }
        if let Some(datetime) = self.datetime.as_deref().filter(|value| !value.is_empty()) {
            parts.push(datetime.to_string());
        }
        if let Some(gps) = &self.gps {
            parts.push(format!("GPS {:.2},{:.2}", gps.lat, gps.lon));
        }
        (!parts.is_empty()).then(|| format!("camera: {}", parts.join(", ")))
    }
}

/// One object class detected by the object detector, aggregated over a
/// file/video.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectDetection {
    pub label: String,
    pub count: u32,
    pub max_conf: f32,
}

/// One zero-shot tag scored above the configured threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagScore {
    pub tag: String,
    pub score: f32,
}

/// One face located by the (opt-in) faces sub-tier, with its SFace embedding.
///
/// **Privacy posture.** These are biometric facts about identifiable people, so
/// they are the one part of a vision result that is deliberately *not* woven
/// into anything that travels: they never reach [`VisionResult::content_block`]
/// (so they are never searchable text, never a sidecar, never part of a report),
/// they are written only into the corpus's own `faces` table on the machine that
/// indexed the file, and no code path in this engine sends them anywhere. The
/// sub-tier that produces them is off by default at every layer — see
/// [`crate::vision::faces`].
///
/// Geometry is in the pixel space of the image that was analysed: the file
/// itself for a still, or the keyframe named by `frame` for a video.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaceDetection {
    /// Left edge of the face box, in analysed-image pixels.
    pub x: i32,
    /// Top edge of the face box, in analysed-image pixels.
    pub y: i32,
    /// Face box width, in analysed-image pixels.
    pub width: u32,
    /// Face box height, in analysed-image pixels.
    pub height: u32,
    /// Detector confidence in `[0, 1]` — the face-quality gate an app clusters
    /// on (a low-scoring face is typically small, blurred, or steeply posed, and
    /// its embedding is the kind that poisons clusters).
    pub quality: f32,
    /// The 128-d SFace embedding of the aligned crop, or `None` when the
    /// embedder could not produce one for this face.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// Video only: the 0-based ordinal of the keyframe this face was found in
    /// (`None` for stills). Keyframes are extracted in a fixed order, so the
    /// ordinal is stable for a given file and `max_frames`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<u32>,
}

/// The full result of analysing one image or video. Fields map 1:1 onto the
/// `vision` table columns (see `store::SCHEMA`); tiers fill in what they own and
/// leave the rest at their defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VisionResult {
    /// Highest tier that actually ran for this file.
    pub mode: VisionMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    /// 16-hex-char 64-bit DCT perceptual hash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exif: Option<ExifInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub objects: Vec<ObjectDetection>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<TagScore>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caption: Option<String>,
    /// CLIP image embedding (kept out of `chunks`, whose vectors are 384-dim e5
    /// text vectors consumed by llm-search).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<usize>,
    /// Faces located by the opt-in faces sub-tier. Empty unless a job turned
    /// faces on AND the models are staged; never rendered into
    /// [`VisionResult::content_block`] (see [`FaceDetection`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub faces: Vec<FaceDetection>,
    /// The face model pair that actually scanned this file (`yunet-sface`), or
    /// `None` when the faces sub-tier did not run for it.
    ///
    /// This is the *evidence that a scan happened*, not the evidence that a face
    /// was found: a photo of a landscape gets `Some("yunet-sface")` and an empty
    /// [`faces`](Self::faces). It is what lets a later resume tell "already
    /// scanned, no faces here" from "never scanned", so turning faces on for an
    /// existing corpus reprocesses exactly once instead of either never (the row
    /// looks complete) or every run (no row is ever proof of a scan).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub faces_model: Option<String>,
    /// Video: number of keyframes analysed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<u64>,
    /// `decode-limit`, `decode-error`, or a tier-specific failure; `None` on
    /// success.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl VisionResult {
    /// Render the searchable `[vision]` block appended to a file's FTS content,
    /// or `None` when nothing worth indexing was produced (so the off-path and
    /// empty results append nothing). Example:
    ///
    /// ```text
    /// [vision] caption: two people walking a dog on a beach at sunset
    /// objects: person(2), dog(1)
    /// tags: beach, sunset, outdoors, family
    /// camera: Apple iPhone 15 Pro, 2024-06-01T18:22, GPS 10.79,106.70
    /// ```
    pub fn content_block(&self) -> Option<String> {
        let mut lines = Vec::new();
        if let Some(caption) = self.caption.as_deref().map(str::trim) {
            if !caption.is_empty() {
                lines.push(format!("caption: {caption}"));
            }
        }
        if !self.objects.is_empty() {
            let objects = self
                .objects
                .iter()
                .map(|object| format!("{}({})", object.label, object.count))
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("objects: {objects}"));
        }
        if !self.tags.is_empty() {
            let tags = self
                .tags
                .iter()
                .map(|tag| tag.tag.clone())
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("tags: {tags}"));
        }
        if let Some(line) = self.exif.as_ref().and_then(ExifInfo::summary_line) {
            lines.push(line);
        }
        (!lines.is_empty()).then(|| format!("[vision] {}", lines.join("\n")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers_order_off_below_meta_below_tags_below_captions() {
        assert!(VisionMode::Off < VisionMode::Meta);
        assert!(VisionMode::Meta < VisionMode::Tags);
        assert!(VisionMode::Tags < VisionMode::Captions);
        // A higher tier includes every lower one.
        assert!(VisionMode::Captions.includes(VisionMode::Off));
        assert!(VisionMode::Tags.includes(VisionMode::Meta));
        assert!(!VisionMode::Meta.includes(VisionMode::Tags));
    }

    #[test]
    fn parses_and_renders_every_tier_round_trip() {
        for mode in VisionMode::ALL {
            assert_eq!(mode.as_str().parse::<VisionMode>().unwrap(), *mode);
            assert_eq!(mode.to_string(), mode.as_str());
        }
        assert_eq!("  Tags ".parse::<VisionMode>().unwrap(), VisionMode::Tags);
        assert!("blur".parse::<VisionMode>().is_err());
    }

    #[test]
    fn only_tags_and_captions_need_models() {
        assert!(!VisionMode::Off.needs_models());
        assert!(!VisionMode::Meta.needs_models());
        assert!(VisionMode::Tags.needs_models());
        assert!(VisionMode::Captions.needs_models());
    }

    #[test]
    fn default_result_renders_no_block() {
        assert!(VisionResult::default().content_block().is_none());
    }

    /// The privacy invariant [`FaceDetection`] documents, asserted rather than
    /// asserted-by-reading: faces are the one vision output that must never be
    /// rendered into the searchable content block (which becomes `fts.content`
    /// and, through the sidecar writer, a `.txt` next to the original). A result
    /// carrying ONLY faces renders no block at all, and adding faces to a result
    /// that does render one leaves that block byte-identical.
    #[test]
    fn faces_never_reach_the_searchable_content_block() {
        let face = FaceDetection {
            x: 10,
            y: 20,
            width: 64,
            height: 80,
            quality: 0.97,
            embedding: Some(vec![0.1; 128]),
            frame: None,
        };
        let faces_only = VisionResult {
            mode: VisionMode::Tags,
            faces: vec![face.clone()],
            faces_model: Some("yunet-sface".into()),
            ..Default::default()
        };
        assert!(faces_only.content_block().is_none());

        let mut tagged = VisionResult {
            mode: VisionMode::Tags,
            tags: vec![TagScore {
                tag: "beach".into(),
                score: 0.4,
            }],
            ..Default::default()
        };
        let without = tagged.content_block();
        tagged.faces = vec![face];
        tagged.faces_model = Some("yunet-sface".into());
        assert_eq!(without, tagged.content_block());
    }

    #[test]
    fn content_block_matches_the_spec_layout() {
        let result = VisionResult {
            mode: VisionMode::Captions,
            caption: Some("two people walking a dog on a beach at sunset".into()),
            objects: vec![
                ObjectDetection {
                    label: "person".into(),
                    count: 2,
                    max_conf: 0.9,
                },
                ObjectDetection {
                    label: "dog".into(),
                    count: 1,
                    max_conf: 0.8,
                },
            ],
            tags: vec![
                TagScore {
                    tag: "beach".into(),
                    score: 0.4,
                },
                TagScore {
                    tag: "sunset".into(),
                    score: 0.3,
                },
            ],
            exif: Some(ExifInfo {
                camera: Some("Apple iPhone 15 Pro".into()),
                datetime: Some("2024-06-01T18:22".into()),
                gps: Some(GpsCoord {
                    lat: 10.79,
                    lon: 106.70,
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let block = result.content_block().unwrap();
        assert_eq!(
            block,
            "[vision] caption: two people walking a dog on a beach at sunset\n\
             objects: person(2), dog(1)\n\
             tags: beach, sunset\n\
             camera: Apple iPhone 15 Pro, 2024-06-01T18:22, GPS 10.79,106.70"
        );
    }
}
