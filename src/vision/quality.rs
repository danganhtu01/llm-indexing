//! Meta tier (V2): blur (variance of Laplacian) + exposure histogram metrics.
//!
//! Pure code over the decoded luma plane. Blur is the variance of a 3x3
//! Laplacian response (low variance => few edges => blurry); exposure is the
//! fraction of pixels clipped into the darkest/brightest histogram bins.

use anyhow::Result;
use image::DynamicImage;
use serde_json::json;

use super::types::VisionResult;

/// Laplacian-variance below this reads as blurry (fixed, deterministic).
const BLUR_VAR_THRESHOLD: f64 = 100.0;
/// Luma at/below this counts as clipped-dark.
const CLIP_LOW: u8 = 16;
/// Luma at/above this counts as clipped-bright.
const CLIP_HIGH: u8 = 239;
/// Clipped fraction above this raises the under/over-exposure flag.
const CLIP_FRACTION_THRESHOLD: f64 = 0.5;

/// Assess image quality into `out.quality` (JSON).
pub(super) fn fill(image: &DynamicImage, out: &mut VisionResult) -> Result<()> {
    let luma = image.to_luma8();
    let (width, height) = luma.dimensions();
    let samples = luma.as_raw();

    let blur_var = laplacian_variance(samples, width, height);
    let (clipped_low, clipped_high, mean_luma) = exposure(samples);

    let blurry = blur_var < BLUR_VAR_THRESHOLD;
    let underexposed = clipped_low > CLIP_FRACTION_THRESHOLD;
    let overexposed = clipped_high > CLIP_FRACTION_THRESHOLD;

    out.quality = Some(json!({
        "blur_var": round(blur_var),
        "blurry": blurry,
        "mean_luma": round(mean_luma),
        "clipped_low": round(clipped_low),
        "clipped_high": round(clipped_high),
        "underexposed": underexposed,
        "overexposed": overexposed,
    }));
    Ok(())
}

/// Variance of the 3x3 Laplacian response over interior pixels. Returns `0.0`
/// for images too small to have an interior (treated as maximally blurry).
fn laplacian_variance(samples: &[u8], width: u32, height: u32) -> f64 {
    if width < 3 || height < 3 {
        return 0.0;
    }
    let width = width as usize;
    let height = height as usize;
    let at = |x: usize, y: usize| f64::from(samples[y * width + x]);

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0.0;
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let laplacian =
                at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4.0 * at(x, y);
            sum += laplacian;
            sum_sq += laplacian * laplacian;
            count += 1.0;
        }
    }
    let mean = sum / count;
    sum_sq / count - mean * mean
}

/// `(clipped_low_fraction, clipped_high_fraction, mean_luma)` over the luma
/// plane. An empty plane reports all zeros.
fn exposure(samples: &[u8]) -> (f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut low = 0u64;
    let mut high = 0u64;
    let mut total = 0u64;
    for &sample in samples {
        if sample <= CLIP_LOW {
            low += 1;
        }
        if sample >= CLIP_HIGH {
            high += 1;
        }
        total += u64::from(sample);
    }
    let count = samples.len() as f64;
    (
        low as f64 / count,
        high as f64 / count,
        total as f64 / count,
    )
}

/// Round to three decimals so the serialized JSON is stable and compact.
fn round(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GrayImage, Luma};

    fn quality(image: &DynamicImage) -> serde_json::Value {
        let mut out = VisionResult::default();
        fill(image, &mut out).unwrap();
        out.quality.unwrap()
    }

    /// A high-contrast checkerboard — many edges, so high Laplacian variance.
    fn checkerboard(size: u32) -> DynamicImage {
        let mut image = GrayImage::new(size, size);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = Luma([if (x + y) % 2 == 0 { 0 } else { 255 }]);
        }
        DynamicImage::ImageLuma8(image)
    }

    fn solid(size: u32, value: u8) -> DynamicImage {
        DynamicImage::ImageLuma8(GrayImage::from_pixel(size, size, Luma([value])))
    }

    #[test]
    fn flat_image_reads_as_blurry() {
        let q = quality(&solid(32, 128));
        assert_eq!(q["blurry"], serde_json::Value::Bool(true));
        assert_eq!(q["blur_var"].as_f64().unwrap(), 0.0);
    }

    #[test]
    fn checkerboard_reads_as_sharp() {
        let q = quality(&checkerboard(32));
        assert_eq!(q["blurry"], serde_json::Value::Bool(false));
        assert!(q["blur_var"].as_f64().unwrap() > BLUR_VAR_THRESHOLD);
    }

    #[test]
    fn dark_image_is_underexposed() {
        let q = quality(&solid(16, 2));
        assert_eq!(q["underexposed"], serde_json::Value::Bool(true));
        assert_eq!(q["overexposed"], serde_json::Value::Bool(false));
        assert_eq!(q["clipped_low"].as_f64().unwrap(), 1.0);
    }

    #[test]
    fn bright_image_is_overexposed() {
        let q = quality(&solid(16, 250));
        assert_eq!(q["overexposed"], serde_json::Value::Bool(true));
        assert_eq!(q["underexposed"], serde_json::Value::Bool(false));
        assert_eq!(q["clipped_high"].as_f64().unwrap(), 1.0);
    }

    #[test]
    fn mid_gray_is_neither_clipped() {
        let q = quality(&solid(16, 128));
        assert_eq!(q["underexposed"], serde_json::Value::Bool(false));
        assert_eq!(q["overexposed"], serde_json::Value::Bool(false));
    }
}
