//! Meta tier (V2): EXIF camera / datetime / GPS parsing via `kamadak-exif`.
//!
//! Pure code — reads the file's own EXIF block, no network. Images without EXIF
//! (e.g. synthesized PNGs) are not an error: `out.exif` simply stays `None`.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::Result;
use exif::{DateTime, Exif, In, Reader, Tag, Value};

use super::types::{ExifInfo, GpsCoord, VisionResult};

/// Parse EXIF metadata from `path` into `out.exif`.
///
/// Extracts camera make/model, `DateTimeOriginal`, decimal GPS, and orientation
/// (kept in [`ExifInfo::fields`]). A file with no readable EXIF leaves
/// `out.exif` untouched and returns `Ok(())`.
pub(super) fn read(path: &Path, out: &mut VisionResult) -> Result<()> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    // A missing/invalid EXIF block is normal for many images; treat it as "no
    // metadata" rather than a tier error so a plain PNG doesn't record an error.
    let exif = match Reader::new().read_from_container(&mut reader) {
        Ok(exif) => exif,
        Err(_) => return Ok(()),
    };

    let mut info = ExifInfo {
        camera: camera(&exif),
        datetime: datetime(&exif),
        gps: gps(&exif),
        ..Default::default()
    };
    if let Some(orientation) = exif
        .get_field(Tag::Orientation, In::PRIMARY)
        .and_then(|field| field.value.get_uint(0))
    {
        info.fields
            .insert("orientation".to_string(), orientation.into());
    }

    // Only attach a row when something human-meaningful was actually parsed.
    if info.camera.is_some()
        || info.datetime.is_some()
        || info.gps.is_some()
        || !info.fields.is_empty()
    {
        out.exif = Some(info);
    }
    Ok(())
}

/// Combine `Make` and `Model` into a single display string ("Apple iPhone 15
/// Pro"), collapsing the redundancy when the model already names the maker.
fn camera(exif: &Exif) -> Option<String> {
    let make = ascii(exif, Tag::Make);
    let model = ascii(exif, Tag::Model);
    match (make, model) {
        (Some(make), Some(model)) => {
            if model
                .to_ascii_lowercase()
                .starts_with(&make.to_ascii_lowercase())
            {
                Some(model)
            } else {
                Some(format!("{make} {model}"))
            }
        }
        (Some(make), None) => Some(make),
        (None, Some(model)) => Some(model),
        (None, None) => None,
    }
}

/// `DateTimeOriginal` reformatted as `YYYY-MM-DDTHH:MM` (falling back to
/// `DateTime` when the original is absent).
fn datetime(exif: &Exif) -> Option<String> {
    let field = exif
        .get_field(Tag::DateTimeOriginal, In::PRIMARY)
        .or_else(|| exif.get_field(Tag::DateTime, In::PRIMARY))?;
    let bytes = match &field.value {
        Value::Ascii(chunks) => chunks.first()?,
        _ => return None,
    };
    let stamp = DateTime::from_ascii(bytes).ok()?;
    Some(format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}",
        stamp.year, stamp.month, stamp.day, stamp.hour, stamp.minute
    ))
}

/// Decimal WGS84 GPS coordinate from the `GPSLatitude`/`GPSLongitude`
/// degrees-minutes-seconds rationals and their N/S/E/W refs.
fn gps(exif: &Exif) -> Option<GpsCoord> {
    let lat = dms(exif, Tag::GPSLatitude, Tag::GPSLatitudeRef, b'S')?;
    let lon = dms(exif, Tag::GPSLongitude, Tag::GPSLongitudeRef, b'W')?;
    Some(GpsCoord { lat, lon })
}

/// Convert one DMS triple + hemisphere ref into a signed decimal degree.
/// `negative_ref` is the ASCII letter (`S`/`W`) that flips the sign.
fn dms(exif: &Exif, coord: Tag, reference: Tag, negative_ref: u8) -> Option<f64> {
    let field = exif.get_field(coord, In::PRIMARY)?;
    let parts = match &field.value {
        Value::Rational(values) if values.len() >= 3 => values,
        _ => return None,
    };
    let degrees = parts[0].to_f64() + parts[1].to_f64() / 60.0 + parts[2].to_f64() / 3600.0;
    let sign = ascii(exif, reference)
        .and_then(|reference| reference.bytes().next())
        .filter(|byte| byte.eq_ignore_ascii_case(&negative_ref))
        .map_or(1.0, |_| -1.0);
    Some(sign * degrees)
}

/// The trimmed first ASCII component of `tag`, or `None` when absent/empty.
fn ascii(exif: &Exif, tag: Tag) -> Option<String> {
    let field = exif.get_field(tag, In::PRIMARY)?;
    let bytes = match &field.value {
        Value::Ascii(chunks) => chunks.first()?,
        _ => return None,
    };
    let text = String::from_utf8_lossy(bytes);
    let trimmed = text.trim().trim_end_matches('\u{0}').trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal but valid JPEG carrying an EXIF APP1 segment with Make, Model,
    /// DateTimeOriginal, and a southern/western GPS fix. Built by hand so the
    /// test needs no fixture file on disk.
    fn jpeg_with_exif() -> Vec<u8> {
        // TIFF header (big-endian) + IFD0 with Make/Model/DateTimeOriginal/GPS
        // pointer, then a GPS IFD. Offsets are relative to the TIFF header start.
        let mut tiff: Vec<u8> = Vec::new();
        tiff.extend_from_slice(b"MM\x00\x2a"); // big-endian, magic 42
        tiff.extend_from_slice(&8u32.to_be_bytes()); // IFD0 at offset 8

        // We lay out variable data after the two IFDs. Compute offsets as we go.
        // IFD0 has 4 entries (Make, Model, DateTimeOriginal, GPSInfoIFDPointer).
        let ifd0_offset = 8u32;
        let ifd0_count = 4u16;
        // Each IFD: 2 (count) + 12*n (entries) + 4 (next-IFD pointer).
        let ifd0_size = 2 + 12 * u32::from(ifd0_count) + 4;
        let gps_ifd_offset = ifd0_offset + ifd0_size;
        let gps_count = 4u16; // LatRef, Lat, LonRef, Lon
        let gps_ifd_size = 2 + 12 * u32::from(gps_count) + 4;
        let data_offset = gps_ifd_offset + gps_ifd_size;

        let make = b"Apple\0";
        let model = b"iPhone 15 Pro\0";
        let dto = b"2024:06:01 18:22:33\0";
        // Blobs placed sequentially in the data area.
        let make_off = data_offset;
        let model_off = make_off + make.len() as u32;
        let dto_off = model_off + model.len() as u32;
        // GPS rationals: lat = 10 deg 47' 24" (=> 10.79), lon = 106 deg 42' 0".
        let lat_off = dto_off + dto.len() as u32;
        let lon_off = lat_off + 24; // 3 rationals * 8 bytes

        // --- IFD0 ---
        tiff.extend_from_slice(&ifd0_count.to_be_bytes());
        // Make (0x010f) ASCII
        entry(&mut tiff, 0x010f, 2, make.len() as u32, make_off);
        // Model (0x0110) ASCII
        entry(&mut tiff, 0x0110, 2, model.len() as u32, model_off);
        // DateTimeOriginal is an Exif-IFD tag, but readers also accept it here;
        // use DateTime (0x0132) in IFD0 which our parser falls back to.
        entry(&mut tiff, 0x0132, 2, dto.len() as u32, dto_off);
        // GPSInfoIFDPointer (0x8825) LONG -> gps_ifd_offset
        entry(&mut tiff, 0x8825, 4, 1, gps_ifd_offset);
        tiff.extend_from_slice(&0u32.to_be_bytes()); // no next IFD

        // --- GPS IFD ---
        tiff.extend_from_slice(&gps_count.to_be_bytes());
        // GPSLatitudeRef (0x0001) ASCII "S\0" inline
        entry_inline(&mut tiff, 0x0001, 2, 2, b"S\0\0\0");
        // GPSLatitude (0x0002) RATIONAL x3 -> lat_off
        entry(&mut tiff, 0x0002, 5, 3, lat_off);
        // GPSLongitudeRef (0x0003) ASCII "W\0" inline
        entry_inline(&mut tiff, 0x0003, 2, 2, b"W\0\0\0");
        // GPSLongitude (0x0004) RATIONAL x3 -> lon_off
        entry(&mut tiff, 0x0004, 5, 3, lon_off);
        tiff.extend_from_slice(&0u32.to_be_bytes()); // no next IFD

        // --- data area ---
        tiff.extend_from_slice(make);
        tiff.extend_from_slice(model);
        tiff.extend_from_slice(dto);
        // lat 10/1, 47/1, 24/1
        for (num, den) in [(10u32, 1u32), (47, 1), (24, 1)] {
            tiff.extend_from_slice(&num.to_be_bytes());
            tiff.extend_from_slice(&den.to_be_bytes());
        }
        // lon 106/1, 42/1, 0/1
        for (num, den) in [(106u32, 1u32), (42, 1), (0, 1)] {
            tiff.extend_from_slice(&num.to_be_bytes());
            tiff.extend_from_slice(&den.to_be_bytes());
        }

        // Wrap the TIFF block in a JPEG APP1 (Exif) segment.
        let mut app1: Vec<u8> = Vec::new();
        app1.extend_from_slice(b"Exif\0\0");
        app1.extend_from_slice(&tiff);
        let seg_len = (app1.len() + 2) as u16;

        let mut jpeg: Vec<u8> = Vec::new();
        jpeg.extend_from_slice(&[0xff, 0xd8]); // SOI
        jpeg.extend_from_slice(&[0xff, 0xe1]); // APP1
        jpeg.extend_from_slice(&seg_len.to_be_bytes());
        jpeg.extend_from_slice(&app1);
        jpeg.extend_from_slice(&[0xff, 0xd9]); // EOI
        jpeg
    }

    fn entry(buf: &mut Vec<u8>, tag: u16, kind: u16, count: u32, value_offset: u32) {
        buf.extend_from_slice(&tag.to_be_bytes());
        buf.extend_from_slice(&kind.to_be_bytes());
        buf.extend_from_slice(&count.to_be_bytes());
        buf.extend_from_slice(&value_offset.to_be_bytes());
    }

    fn entry_inline(buf: &mut Vec<u8>, tag: u16, kind: u16, count: u32, inline: &[u8; 4]) {
        buf.extend_from_slice(&tag.to_be_bytes());
        buf.extend_from_slice(&kind.to_be_bytes());
        buf.extend_from_slice(&count.to_be_bytes());
        buf.extend_from_slice(inline);
    }

    #[test]
    fn parses_camera_datetime_and_gps() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("exif.jpg");
        std::fs::write(&path, jpeg_with_exif()).unwrap();

        let mut out = VisionResult::default();
        read(&path, &mut out).unwrap();
        let info = out.exif.expect("exif parsed");

        assert_eq!(info.camera.as_deref(), Some("Apple iPhone 15 Pro"));
        assert_eq!(info.datetime.as_deref(), Some("2024-06-01T18:22"));
        let gps = info.gps.expect("gps parsed");
        assert!((gps.lat - -10.79).abs() < 0.01, "lat was {}", gps.lat);
        assert!((gps.lon - -106.70).abs() < 0.01, "lon was {}", gps.lon);
    }

    #[test]
    fn no_exif_is_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plain.png");
        image::RgbImage::new(4, 4).save(&path).unwrap();

        let mut out = VisionResult::default();
        read(&path, &mut out).unwrap();
        assert!(out.exif.is_none());
        assert!(out.error.is_none());
    }

    #[test]
    fn summary_line_matches_the_vision_block_layout() {
        let info = ExifInfo {
            camera: Some("Apple iPhone 15 Pro".into()),
            datetime: Some("2024-06-01T18:22".into()),
            gps: Some(GpsCoord {
                lat: 10.79,
                lon: 106.70,
            }),
            ..Default::default()
        };
        assert_eq!(
            info.summary_line().as_deref(),
            Some("camera: Apple iPhone 15 Pro, 2024-06-01T18:22, GPS 10.79,106.70")
        );
    }
}
