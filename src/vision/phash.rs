//! Meta tier (V2): 64-bit DCT perceptual hash rendered as 16 hex chars.
//!
//! Pure code, no dependency beyond `image`: 32x32 grayscale -> separable 2D
//! DCT-II -> top-left 8x8 -> median threshold (excluding the DC term) -> 64-bit
//! hash. Near-duplicate images (resizes, re-encodes) land a small Hamming
//! distance; unrelated images land a large one.

use std::f64::consts::PI;

use anyhow::Result;
use image::imageops::FilterType;
use image::DynamicImage;

/// Side length of the downscaled grayscale image fed to the DCT.
const DCT_SIZE: usize = 32;
/// Side length of the low-frequency block kept from the DCT.
const HASH_SIZE: usize = 8;

/// Compute the perceptual hash of `image` into `out.phash`.
pub(super) fn fill(image: &DynamicImage, out: &mut super::types::VisionResult) -> Result<()> {
    out.phash = Some(hash(image));
    Ok(())
}

/// The 16-hex-char 64-bit DCT perceptual hash of `image`.
fn hash(image: &DynamicImage) -> String {
    // Downscale to a fixed grayscale grid so re-encodes and resizes collapse to
    // the same low-frequency content. A fixed filter keeps this deterministic.
    let gray = image.to_luma8();
    let small = image::imageops::resize(
        &gray,
        DCT_SIZE as u32,
        DCT_SIZE as u32,
        FilterType::Triangle,
    );

    let mut matrix = vec![0.0f64; DCT_SIZE * DCT_SIZE];
    for (index, pixel) in small.pixels().enumerate() {
        matrix[index] = f64::from(pixel[0]);
    }

    let dct = dct_2d(&matrix);

    // Collect the top-left 8x8 low-frequency coefficients (row-major).
    let mut low = [0.0f64; HASH_SIZE * HASH_SIZE];
    for row in 0..HASH_SIZE {
        for col in 0..HASH_SIZE {
            low[row * HASH_SIZE + col] = dct[row * DCT_SIZE + col];
        }
    }

    // Median of the 63 non-DC coefficients (index 0 is the DC term).
    let mut non_dc: Vec<f64> = low[1..].to_vec();
    non_dc.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = (non_dc[non_dc.len() / 2 - 1] + non_dc[non_dc.len() / 2]) / 2.0;

    // One bit per coefficient (including DC), high bit first.
    let mut bits: u64 = 0;
    for &coeff in low.iter() {
        bits <<= 1;
        if coeff > median {
            bits |= 1;
        }
    }
    format!("{bits:016x}")
}

/// Separable 2D DCT-II of an `N x N` (N = [`DCT_SIZE`]) row-major matrix: 1D DCT
/// across each row, then across each column of the result.
fn dct_2d(input: &[f64]) -> Vec<f64> {
    let n = DCT_SIZE;
    let cosines = cosine_table();

    // Rows.
    let mut rows = vec![0.0f64; n * n];
    for row in 0..n {
        let src = &input[row * n..row * n + n];
        let dst = &mut rows[row * n..row * n + n];
        dct_1d(src, dst, &cosines);
    }

    // Columns.
    let mut out = vec![0.0f64; n * n];
    let mut column = [0.0f64; DCT_SIZE];
    let mut transformed = [0.0f64; DCT_SIZE];
    for col in 0..n {
        for row in 0..n {
            column[row] = rows[row * n + col];
        }
        dct_1d(&column, &mut transformed, &cosines);
        for row in 0..n {
            out[row * n + col] = transformed[row];
        }
    }
    out
}

/// 1D DCT-II: `dst[k] = sum_n src[n] * cos(pi/N * (n + 0.5) * k)`. The overall
/// scale is irrelevant here — only the sign relative to the median matters.
fn dct_1d(src: &[f64], dst: &mut [f64], cosines: &[f64]) {
    let n = DCT_SIZE;
    for (k, out) in dst.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (sample_index, &sample) in src.iter().enumerate() {
            sum += sample * cosines[k * n + sample_index];
        }
        *out = sum;
    }
}

/// Precomputed `cos(pi/N * (n + 0.5) * k)` table, indexed `[k * N + n]`.
fn cosine_table() -> Vec<f64> {
    let n = DCT_SIZE;
    let mut table = vec![0.0f64; n * n];
    for k in 0..n {
        for sample in 0..n {
            table[k * n + sample] = (PI / n as f64 * (sample as f64 + 0.5) * k as f64).cos();
        }
    }
    table
}

/// Hamming distance between two 16-hex-char perceptual hashes, or `None` if
/// either side is not valid 16-char hex. Near-duplicate detection helper
/// (VISION-SPEC §V2), re-exported from `mod.rs` as `vision::hamming` so
/// consumers (and the on-box phash-dedup smoke) can compare stored `vision.phash`
/// values.
pub fn hamming(a: &str, b: &str) -> Option<u32> {
    let left = u64::from_str_radix(a, 16).ok()?;
    let right = u64::from_str_radix(b, 16).ok()?;
    Some((left ^ right).count_ones())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, Rgb, RgbImage};

    /// A deterministic, texture-rich test image seeded by `seed` so distinct
    /// seeds produce visually unrelated content.
    fn synth(width: u32, height: u32, seed: u32) -> DynamicImage {
        let mut image = RgbImage::new(width, height);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            // A couple of incommensurate sinusoids + the seed give structured
            // but seed-dependent content (avoids a flat, hash-degenerate image).
            let a = ((x * 7 + y * 13 + seed * 101) % 256) as u8;
            let b = (((x * x + y * y + seed * 53) / 3) % 256) as u8;
            let c = ((x.wrapping_mul(y).wrapping_add(seed * 17)) % 256) as u8;
            *pixel = Rgb([a, b, c]);
        }
        DynamicImage::ImageRgb8(image)
    }

    fn phash(image: &DynamicImage) -> String {
        let mut out = crate::vision::types::VisionResult::default();
        fill(image, &mut out).unwrap();
        out.phash.unwrap()
    }

    #[test]
    fn hash_is_16_hex_chars_and_stable() {
        let image = synth(200, 150, 1);
        let first = phash(&image);
        assert_eq!(first.len(), 16);
        assert!(first.chars().all(|c| c.is_ascii_hexdigit()));
        // Deterministic: same input, same hash.
        assert_eq!(first, phash(&image));
    }

    #[test]
    fn resized_copy_is_near_identical() {
        let original = synth(256, 256, 7);
        let resized = original.resize_exact(180, 200, FilterType::Lanczos3);
        let distance = hamming(&phash(&original), &phash(&resized)).unwrap();
        // A resize of the same content must stay close.
        assert!(distance <= 8, "resized distance was {distance}");
    }

    #[test]
    fn unrelated_images_are_far_apart() {
        let a = synth(200, 200, 3);
        let b = synth(200, 200, 999);
        let distance = hamming(&phash(&a), &phash(&b)).unwrap();
        // Unrelated content should differ in many bits.
        assert!(distance >= 18, "unrelated distance was {distance}");
    }

    #[test]
    fn hamming_rejects_non_hex() {
        assert_eq!(hamming("0000000000000000", "ffffffffffffffff"), Some(64));
        assert_eq!(hamming("zz", "0000000000000000"), None);
    }
}
