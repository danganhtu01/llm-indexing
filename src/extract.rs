use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use chardetng::EncodingDetector;
use mailparse::{parse_mail, MailHeaderMap, ParsedMail};
use quick_xml::events::Event;
use quick_xml::Reader;
use sha1::{Digest, Sha1};
use tempfile::tempdir;
use zip::ZipArchive;

use crate::config::Config;
use crate::failure::{CapabilityUnavailable, EncryptedDocument, HeicDecodeFailed};
use crate::media::Transcriber;
use crate::ocr::TesseractOcr;

const TEXT_EXTS: &[&str] = &[
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".log",
    ".json",
    ".xml",
    ".html",
    ".htm",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".rtf",
    ".srt",
    ".vtt",
];
const CODE_EXTS: &[&str] = &[
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".cs", ".go", ".rs", ".rb",
    ".php", ".sql", ".sh", ".ps1", ".bat", ".r", ".css", ".scss",
];
const IMAGE_EXTS: &[&str] = &[
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif",
];
/// `.heic` (Apple Photos/iCloud export): not in [`IMAGE_EXTS`] because the
/// `image` crate has no HEIF feature and this build carries no libheif
/// binding, so it cannot be handed to the OCR arm directly. Decoded to a JPEG
/// frame first by [`heic`] via ffmpeg, then run through the same
/// [`TesseractOcr::image_to_text`] the `IMAGE_EXTS` arm uses. 18,165 rows of
/// the live corpus sat at `name-only-partial` for this before ffmpeg was
/// confirmed (2026-07-25, this box) to decode them. `.heif` — the same
/// container format under its generic extension — is deliberately NOT
/// included: nothing in the live corpus needed it, and it was never verified
/// against a real file the way `.heic` was; add it here (and to `heic()`'s
/// doc comment) once it is.
const HEIC_EXTS: &[&str] = &[".heic"];
const EMAIL_EXTS: &[&str] = &[".eml", ".wdseml", ".emlx"];
const AUDIO_EXTS: &[&str] = &[".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"];
const VIDEO_EXTS: &[&str] = &[".mkv", ".mp4", ".mov", ".m4v", ".avi", ".webm"];
const ARCHIVE_EXTS: &[&str] = &[".zip", ".rar", ".7z", ".tar", ".gz", ".tgz"];
/// The extensions `extract_inner`'s dispatch handles by name. Duplicated out of
/// that `match` so [`extractor_revision`] moves when the dispatch gains or loses
/// a format; `dispatched_extensions_are_not_reported_unsupported` pins them
/// together.
const DOCUMENT_EXTS: &[&str] = &[
    ".pdf", ".doc", ".docx", ".xlsx", ".xlsm", ".pptx", ".odt", ".ods", ".odp",
];

/// A fingerprint of what THIS BUILD is able to extract, recorded on the corpus
/// so a resume can tell that the engine's capability changed since the rows it
/// is looking at were written.
///
/// It exists for the attempt cap ([`crate::pipeline::MAX_ATTEMPTS`]). A row
/// capped for a file the old build genuinely could not read must not stay capped
/// once a build that CAN read it is deployed — the cap would otherwise freeze
/// every `.heic`, every newly dispatched format, on the verdict of the code that
/// had no decoder for it. Derived from the extension tables rather than
/// hand-maintained so it moves on its own the moment one gains an entry, with
/// the crate version folded in so a release that changes extraction some other
/// way (a fixed OCR fallback, a new error path) also grants capped rows one more
/// attempt. Cheap enough to call once per run; never on a per-file path.
pub fn extractor_revision() -> String {
    let mut hash = Sha1::new();
    hash.update(env!("CARGO_PKG_VERSION"));
    for table in [
        TEXT_EXTS,
        CODE_EXTS,
        IMAGE_EXTS,
        HEIC_EXTS,
        EMAIL_EXTS,
        AUDIO_EXTS,
        VIDEO_EXTS,
        ARCHIVE_EXTS,
        DOCUMENT_EXTS,
    ] {
        for ext in table {
            hash.update(ext);
        }
    }
    let digest = format!("{:x}", hash.finalize());
    digest[..12].to_string()
}

#[derive(Debug, Clone)]
pub struct Extracted {
    pub text: String,
    pub method: String,
    pub ocr_used: bool,
    pub pages: usize,
    /// `(page_number, text)` for the extraction paths that can attribute text
    /// to a specific page (currently only the two PDF paths that keep
    /// `pdftotext`'s per-page structure intact), 1-based to match `pages` and
    /// `pdf_pages`. Empty for every other extraction method and for a PDF
    /// path that merged page-agnostic OCR text into `text` in a way that
    /// cannot be re-split honestly (P0-8: this is what lets the chunker
    /// attribute a search hit to "p. 14" instead of just a filename).
    pub page_segments: Vec<(usize, String)>,
}

impl Extracted {
    /// Nothing was extracted, but something COULD have been: the file was over
    /// `max_bytes`, or its extension is in `skip_exts`, or a stage that handles
    /// it was switched off. Every one of those is a configuration verdict, so the
    /// row stays retryable (`name-only-partial`) and a later run with a larger
    /// budget or a stage enabled reprocesses it.
    fn empty() -> Self {
        Self {
            text: String::new(),
            method: "name-only".into(),
            ocr_used: false,
            pages: 0,
            page_segments: Vec::new(),
        }
    }

    /// No extractor in this build handles this extension at all — object files,
    /// game archives, compiler intermediates: 48k of the live corpus. TERMINAL,
    /// not a failure. `excluded:` is the repo's "deliberately not processed"
    /// marker and resume treats it as a finished row, so unlike the
    /// `name-only-partial` this used to produce, it is not re-attempted on every
    /// resume forever. A build that later grows a decoder for it moves
    /// [`extractor_revision`], which is what re-opens these rows.
    fn unsupported() -> Self {
        Self {
            text: String::new(),
            method: "excluded:unsupported".into(),
            ocr_used: false,
            pages: 0,
            page_segments: Vec::new(),
        }
    }
}

pub fn extract(
    path: &Path,
    ext: &str,
    size: u64,
    config: &Config,
    ocr: &TesseractOcr,
    transcriber: &Transcriber,
) -> Result<Extracted> {
    extract_inner(path, ext, size, config, ocr, transcriber, 0)
}

fn extract_inner(
    path: &Path,
    ext: &str,
    size: u64,
    config: &Config,
    ocr: &TesseractOcr,
    transcriber: &Transcriber,
    depth: usize,
) -> Result<Extracted> {
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("~$"))
    {
        return Ok(Extracted {
            text: "Temporary Office lock file; no document body exists.".into(),
            method: "excluded:office-lock".into(),
            ocr_used: false,
            pages: 0,
            page_segments: Vec::new(),
        });
    }
    if (size > config.max_bytes && config.ocr != "exhaustive") || config.skip_ext(ext) {
        return Ok(Extracted::empty());
    }
    let max_chars = if config.ocr == "exhaustive" {
        usize::MAX / 4
    } else {
        config.max_chars
    };
    if TEXT_EXTS.contains(&ext) || CODE_EXTS.contains(&ext) {
        let raw = read_limited(path, max_chars.saturating_mul(4))?;
        return Ok(Extracted {
            text: decode(&raw, max_chars),
            method: "text".into(),
            ocr_used: false,
            pages: 0,
            page_segments: Vec::new(),
        });
    }
    if EMAIL_EXTS.contains(&ext) {
        return Ok(Extracted {
            text: email(path, max_chars)?,
            method: "email".into(),
            ocr_used: false,
            pages: 0,
            page_segments: Vec::new(),
        });
    }
    match ext {
        ".pdf" if config.ocr == "exhaustive" => pdf_exhaustive(path, config, ocr),
        ".pdf" => pdf(path, config, ocr),
        ".doc" => legacy_doc(path, max_chars),
        ".docx" => office_archive(path, &["word/"], &["word/media/"], "docx", config, ocr),
        ".xlsx" | ".xlsm" => office_archive(path, &["xl/"], &["xl/media/"], "xlsx", config, ocr),
        ".pptx" => office_archive(
            path,
            &["ppt/slides/", "ppt/notesSlides/"],
            &["ppt/media/"],
            "pptx",
            config,
            ocr,
        ),
        ".odt" | ".ods" | ".odp" => {
            office_archive(path, &["content.xml"], &["Pictures/"], "odf", config, ocr)
        }
        _ if IMAGE_EXTS.contains(&ext)
            && matches!(config.ocr.as_str(), "auto" | "on" | "exhaustive")
            && ocr.available =>
        {
            let text = truncate(ocr.image_to_text(path), max_chars);
            Ok(Extracted {
                ocr_used: !text.trim().is_empty(),
                text,
                method: "ocr".into(),
                pages: 1,
                page_segments: Vec::new(),
            })
        }
        _ if HEIC_EXTS.contains(&ext)
            && matches!(config.ocr.as_str(), "auto" | "on" | "exhaustive")
            && ocr.available =>
        {
            heic(path, max_chars, ocr, config.headroom_cores_cap())
        }
        _ if AUDIO_EXTS.contains(&ext) || VIDEO_EXTS.contains(&ext) => media(
            path,
            ext,
            max_chars,
            ocr,
            transcriber,
            config.headroom_cores_cap(),
        ),
        _ if ARCHIVE_EXTS.contains(&ext) && depth < 4 => {
            archive(path, config, ocr, transcriber, depth + 1, max_chars)
        }
        // A format this build DOES handle whose arm above declined on a runtime
        // condition — OCR off or Tesseract absent for an image or a HEIC, an
        // archive past the nesting limit. The capability exists, so the row
        // must stay retryable rather than be written off as unsupported.
        _ if IMAGE_EXTS.contains(&ext)
            || HEIC_EXTS.contains(&ext)
            || AUDIO_EXTS.contains(&ext)
            || VIDEO_EXTS.contains(&ext)
            || ARCHIVE_EXTS.contains(&ext) =>
        {
            Ok(Extracted::empty())
        }
        _ => Ok(Extracted::unsupported()),
    }
}

fn legacy_doc(path: &Path, max_chars: usize) -> Result<Extracted> {
    let output = Command::new("antiword")
        .arg(path)
        .output()
        .with_context(|| format!("running antiword for {}", path.display()))?;
    if !output.status.success() {
        anyhow::bail!(
            "antiword failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
    }
    let text = truncate(
        String::from_utf8_lossy(&output.stdout).into_owned(),
        max_chars,
    );
    if text.trim().is_empty() {
        anyhow::bail!("antiword produced no document text")
    }
    Ok(Extracted {
        text,
        method: "doc".into(),
        ocr_used: false,
        pages: 0,
        page_segments: Vec::new(),
    })
}

fn archive(
    path: &Path,
    config: &Config,
    ocr: &TesseractOcr,
    transcriber: &Transcriber,
    depth: usize,
    max_chars: usize,
) -> Result<Extracted> {
    let listing = bsdtar().args(["-tf"]).arg(path).output()?;
    if !listing.status.success() {
        anyhow::bail!(
            "archive listing failed: {}",
            String::from_utf8_lossy(&listing.stderr).trim()
        )
    }
    for name in String::from_utf8_lossy(&listing.stdout).lines() {
        let candidate = Path::new(name);
        if candidate.is_absolute()
            || candidate
                .components()
                .any(|part| matches!(part, std::path::Component::ParentDir))
        {
            anyhow::bail!("archive contains an unsafe path")
        }
    }
    let temp = tempdir()?;
    let output = bsdtar()
        .args(["-xf"])
        .arg(path)
        .arg("-C")
        .arg(temp.path())
        .output()?;
    if !output.status.success() {
        anyhow::bail!(
            "archive extraction failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
    }
    let mut stack = vec![temp.path().to_path_buf()];
    let mut files = Vec::new();
    while let Some(directory) = stack.pop() {
        for entry in fs::read_dir(directory)?.flatten() {
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                stack.push(entry.path());
            } else if file_type.is_file() {
                files.push(entry.path());
            }
            if files.len() > 10_000 {
                anyhow::bail!("archive exceeds the 10,000-entry safety limit")
            }
        }
    }
    files.sort();
    let mut parts = Vec::new();
    let mut partial = false;
    let mut ocr_used = false;
    let mut pages = 0;
    for file in files {
        let metadata = file.metadata()?;
        let extension = file
            .extension()
            .map(|value| format!(".{}", value.to_string_lossy().to_lowercase()))
            .unwrap_or_default();
        match extract_inner(
            &file,
            &extension,
            metadata.len(),
            config,
            ocr,
            transcriber,
            depth,
        ) {
            Ok(extracted) => {
                partial |= extracted.method == "name-only"
                    || extracted.method.starts_with("error:")
                    || extracted.method.ends_with("-partial");
                ocr_used |= extracted.ocr_used;
                pages += extracted.pages;
                if !extracted.text.trim().is_empty() {
                    let relative = file.strip_prefix(temp.path()).unwrap_or(&file);
                    parts.push(format!(
                        "[archive entry: {}]\n{}",
                        relative.display(),
                        extracted.text
                    ));
                }
            }
            Err(error) => {
                partial = true;
                let relative = file.strip_prefix(temp.path()).unwrap_or(&file);
                parts.push(format!(
                    "[archive entry error: {}] {error:#}",
                    relative.display()
                ));
            }
        }
        if parts.iter().map(String::len).sum::<usize>() >= max_chars {
            break;
        }
    }
    if parts.is_empty() {
        anyhow::bail!("archive contains no extractable content")
    }
    Ok(Extracted {
        text: truncate(parts.join("\n\n"), max_chars),
        method: if partial {
            "archive-partial"
        } else {
            "archive"
        }
        .into(),
        ocr_used,
        pages,
        page_segments: Vec::new(),
    })
}

/// libarchive converts archive entry names through the process locale. The
/// service intentionally runs without a generated locale package, so select
/// Debian's built-in UTF-8 locale explicitly for Vietnamese and other Unicode
/// filenames. A plain `C` locale makes bsdtar skip otherwise valid ZIP entries
/// and return a false extraction error.
fn bsdtar() -> Command {
    let mut command = Command::new("bsdtar");
    command.env("LANG", "C.UTF-8").env("LC_ALL", "C.UTF-8");
    command
}

fn media(
    path: &Path,
    ext: &str,
    max_chars: usize,
    ocr: &TesseractOcr,
    transcriber: &Transcriber,
    ffmpeg_threads: Option<usize>,
) -> Result<Extracted> {
    if !transcriber.available() {
        return Err(
            CapabilityUnavailable("local Whisper transcription model is unavailable").into(),
        );
    }
    let transcript = transcriber.transcribe(path)?;
    let mut sections = vec![format!("[Audio transcript]\n{transcript}")];
    let mut frame_count = 0;
    if VIDEO_EXTS.contains(&ext) && ocr.available {
        let temp = tempdir()?;
        let pattern = temp.path().join("frame-%06d.png");
        let output = Command::new("ffmpeg")
            .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(path)
            .args([
                "-vf",
                "fps=1/30,scale='min(1920,iw)':-2",
                "-frames:v",
                "1000",
            ])
            // `-threads <cores_cap>` under headroom, nothing otherwise (see
            // `headroom::ffmpeg_thread_args`).
            .args(crate::headroom::ffmpeg_thread_args(ffmpeg_threads))
            .arg(&pattern)
            .output()?;
        if output.status.success() {
            let mut frames = fs::read_dir(temp.path())?
                .flatten()
                .map(|entry| entry.path())
                .collect::<Vec<_>>();
            frames.sort();
            frame_count = frames.len();
            let mut seen = HashSet::new();
            let visual = frames
                .into_iter()
                .filter_map(|frame| {
                    let text = ocr.image_to_text(&frame).trim().to_string();
                    (!text.is_empty() && seen.insert(text.clone())).then_some(text)
                })
                .collect::<Vec<_>>()
                .join("\n");
            if !visual.is_empty() {
                sections.push(format!("[Video frame OCR]\n{visual}"));
            }
        }
    }
    Ok(Extracted {
        text: truncate(sections.join("\n\n"), max_chars),
        method: if VIDEO_EXTS.contains(&ext) {
            "video-transcript-ocr"
        } else {
            "audio-transcript"
        }
        .into(),
        ocr_used: frame_count > 0,
        pages: frame_count,
        page_segments: Vec::new(),
    })
}

/// Bound on how long a single HEIC-to-JPEG ffmpeg decode may run before it is
/// killed. This is the first extraction call site that bounds its subprocess
/// wait at all — every other `Command::output()` call in this file blocks
/// without a deadline (see `failure.rs`'s note that
/// [`crate::failure::FailureClass::Timeout`] was previously unreachable) — so
/// a pathological/corrupt HEIC frees the extract worker instead of hanging it
/// for the life of the job. 60s mirrors `vision.timeout_secs`'s default
/// (`config.rs::default_vision_timeout`), the closest existing per-file media
/// budget in this build; a still-image decode should never approach it.
const HEIC_DECODE_TIMEOUT_SECS: u64 = 60;

/// Decode one still frame from a `.heic` file to a temp JPEG via ffmpeg, then
/// run it through the same [`TesseractOcr::image_to_text`] the `IMAGE_EXTS`
/// OCR arm uses.
///
/// ffmpeg has no dedicated HEIF demuxer in this build (`ffmpeg -formats`
/// shows none) but reads HEIC/HEIF anyway: the format shares its ISOBMFF
/// container with MP4, so ffmpeg's `mov,mp4,m4a,3gp,3g2,mj2` demuxer opens it
/// and decodes the embedded HEVC still picture(s) via its ordinary HEVC
/// decoder — including reassembling a tiled grid and applying the stored
/// orientation, so a plain `-frames:v 1` extraction with no explicit `-map`
/// is sufficient. Verified against a real iCloud-export `.heic` on this box
/// (ffmpeg 8.1.2, 2026-07-25): a 48-tile 4032x3024 grid decoded correctly,
/// rotated, in ~150ms. NOT verified on other ffmpeg builds — this depends on
/// mov-demuxer HEIF stream-group support that is new enough that older
/// ffmpeg builds may only see the first tile or refuse the file; a build
/// that can't cope fails the same way a corrupt file does (see below), which
/// is the safe direction to fail in.
///
/// Three distinct failure shapes reach three distinct [`FailureClass`]es via
/// [`crate::failure::classify`]:
/// - ffmpeg missing from `PATH`: `Command::spawn` returns an `io::Error` of
///   kind `NotFound`, propagated as-is (no typed marker needed — `classify`
///   already maps that kind to `IoNotFound`, same as a missing `antiword`).
/// - ffmpeg wedged past [`HEIC_DECODE_TIMEOUT_SECS`]: [`wait_bounded`]
///   returns an `io::Error` of kind `TimedOut`, which `classify` maps to
///   `Timeout` — the class this call site is what makes reachable at all.
/// - ffmpeg ran and exited but rejected the bytes (non-zero exit, or a zero
///   exit with no output file — both observed from real ffmpeg builds on
///   malformed input): [`HeicDecodeFailed`], which `classify` maps to
///   `Decode`.
///
/// [`FailureClass`]: crate::failure::FailureClass
fn heic(
    path: &Path,
    max_chars: usize,
    ocr: &TesseractOcr,
    ffmpeg_threads: Option<usize>,
) -> Result<Extracted> {
    let temp = tempdir()?;
    let frame = temp.path().join("frame.jpg");
    let mut command = Command::new("ffmpeg");
    command.args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i"]);
    command.arg(path);
    command.args(["-frames:v", "1", "-q:v", "2"]);
    // `-threads <cores_cap>` under headroom, nothing otherwise (see
    // `headroom::ffmpeg_thread_args`). Bounds the tiled-HEVC decode, the one
    // CPU-heavy step here.
    command.args(crate::headroom::ffmpeg_thread_args(ffmpeg_threads));
    command.arg(&frame);
    // stdout/stderr dropped, matching `vision::video`'s bounded ffmpeg calls:
    // the frame lands on disk, so there is no pipe to drain, and draining one
    // is what the bounded wait below is not built to do.
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut child = command
        .spawn()
        .with_context(|| format!("spawning ffmpeg for {}", path.display()))?;
    let status = wait_bounded(&mut child, Duration::from_secs(HEIC_DECODE_TIMEOUT_SECS))
        .with_context(|| format!("running ffmpeg for {}", path.display()))?;
    if !status.success() || !frame.is_file() {
        return Err(HeicDecodeFailed.into());
    }
    let text = truncate(ocr.image_to_text(&frame), max_chars);
    Ok(Extracted {
        ocr_used: !text.trim().is_empty(),
        text,
        method: "heic-ocr".into(),
        pages: 1,
        // A single OCR'd frame: no page structure to attribute (P0-8).
        page_segments: Vec::new(),
    })
}

/// Wait for `child` up to `timeout`, killing and reaping it on expiry.
/// Duplicated from `vision::video::wait_bounded` rather than shared — that
/// helper is private to a module owned by the V4 worker (see its file-level
/// doc comment) and returns `Option<ExitStatus>` for a caller that wants a
/// distinguishable-by-string timeout, where this caller wants an `io::Error`
/// of kind `TimedOut` so [`crate::failure::classify`] can recognize it by
/// TYPE. Polls rather than blocking so a pathological ffmpeg cannot wedge the
/// extract worker forever.
fn wait_bounded(child: &mut Child, timeout: Duration) -> std::io::Result<ExitStatus> {
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!(
                    "ffmpeg exceeded {}s decoding a HEIC frame",
                    timeout.as_secs()
                ),
            ));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn read_limited(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(limit as u64).read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn decode(raw: &[u8], max_chars: usize) -> String {
    let mut detector = EncodingDetector::new();
    detector.feed(raw, true);
    let encoding = detector.guess(None, true);
    let (text, _, _) = encoding.decode(raw);
    truncate(text.into_owned(), max_chars)
}

fn truncate(text: String, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

fn email(path: &Path, max_chars: usize) -> Result<String> {
    let bytes = fs::read(path)?;
    let parsed = parse_mail(&bytes)?;
    let mut parts = Vec::new();
    for header in ["Subject", "From", "To", "Cc", "Date"] {
        if let Some(value) = parsed.headers.get_first_value(header) {
            parts.push(format!("{header}: {value}"));
        }
    }
    collect_mail(&parsed, &mut parts);
    Ok(truncate(parts.join("\n"), max_chars))
}

fn collect_mail(mail: &ParsedMail<'_>, parts: &mut Vec<String>) {
    if mail.subparts.is_empty() {
        if mail.ctype.mimetype == "text/plain" {
            if let Ok(body) = mail.get_body() {
                parts.push(body)
            }
        } else if mail.ctype.mimetype == "text/html" {
            if let Ok(body) = mail.get_body() {
                parts.push(strip_html(&body))
            }
        }
        if let Some(disposition) = mail.get_headers().get_first_value("Content-Disposition") {
            if let Some((_, filename)) = disposition.split_once("filename=") {
                parts.push(format!(
                    "[attachment: {}]",
                    filename.trim_matches(['\"', '\''])
                ));
            }
        }
    } else {
        for subpart in &mail.subparts {
            collect_mail(subpart, parts)
        }
    }
}

fn strip_html(html: &str) -> String {
    let mut output = String::with_capacity(html.len());
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                output.push(' ');
            }
            _ if !in_tag => output.push(c),
            _ => {}
        }
    }
    output
}

fn office_archive(
    path: &Path,
    prefixes: &[&str],
    media_prefixes: &[&str],
    method: &str,
    config: &Config,
    ocr: &TesseractOcr,
) -> Result<Extracted> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    let mut parts = Vec::new();
    let mut images = Vec::new();
    let exhaustive = config.ocr == "exhaustive";
    let temp = tempdir()?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        if name.ends_with(".xml") && prefixes.iter().any(|prefix| name.starts_with(prefix)) {
            let mut xml = Vec::new();
            entry.read_to_end(&mut xml)?;
            parts.extend(xml_text(&xml));
        } else if exhaustive
            && ocr.available
            && media_prefixes.iter().any(|prefix| name.starts_with(prefix))
            && IMAGE_EXTS
                .iter()
                .any(|ext| name.to_lowercase().ends_with(ext))
        {
            let extension = Path::new(&name)
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("png");
            let target = temp.path().join(format!("image-{i}.{extension}"));
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes)?;
            fs::write(&target, bytes)?;
            images.push(target);
        }
    }
    let mut ocr_used = false;
    for image in &images {
        let text = ocr.image_to_text(image);
        if !text.trim().is_empty() {
            ocr_used = true;
            parts.push(format!("[Embedded image OCR]\n{text}"));
        }
    }
    Ok(Extracted {
        text: truncate(
            parts.join("\n"),
            if exhaustive {
                usize::MAX
            } else {
                config.max_chars
            },
        ),
        method: if ocr_used {
            format!("{method}-ocr")
        } else {
            method.into()
        },
        ocr_used,
        pages: images.len(),
        page_segments: Vec::new(),
    })
}

fn xml_text(xml: &[u8]) -> Vec<String> {
    let mut reader = Reader::from_reader(Cursor::new(xml));
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut capture = false;
    let mut out = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(tag)) => {
                let name = tag.name();
                let local = name
                    .as_ref()
                    .rsplit(|b| *b == b':')
                    .next()
                    .unwrap_or(name.as_ref());
                capture = matches!(local, b"t" | b"v");
            }
            Ok(Event::Text(text)) if capture => {
                let value = String::from_utf8_lossy(text.as_ref()).trim().to_string();
                if !value.is_empty() {
                    out.push(value)
                }
            }
            Ok(Event::End(_)) => capture = false,
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    out
}

fn pdf_exhaustive(path: &Path, config: &Config, ocr: &TesseractOcr) -> Result<Extracted> {
    if !ocr.available {
        // A typed error, not `anyhow::bail!`, so `crate::failure::classify`
        // recognizes this as `Unsupported` by downcasting rather than parsing
        // the message — see `failure.rs`.
        return Err(
            CapabilityUnavailable("Tesseract is unavailable for exhaustive PDF OCR").into(),
        );
    }
    let info = pdf_info(path);
    if info.password_required {
        // A typed error, not `anyhow::bail!`, so `crate::failure::classify`
        // recognizes this as `Encrypted` by downcasting rather than parsing
        // the message — see `failure.rs`. Bailing here saves every
        // per-page pdftotext/pdftoppm spawn this loop would otherwise make,
        // all doomed to the same empty result.
        return Err(EncryptedDocument.into());
    }
    let pages = info.pages;
    if pages == 0 {
        anyhow::bail!("PDF page count is unavailable")
    }
    let temp = tempdir()?;
    let dpi = config.ocr_dpi.to_string();
    let mut parts = Vec::with_capacity(pages);
    let mut page_segments = Vec::with_capacity(pages);
    let mut used_ocr = false;
    for page in 1..=pages {
        let output = Command::new("pdftotext")
            .args(["-f", &page.to_string(), "-l", &page.to_string()])
            .arg(path)
            .arg("-")
            .output();
        let text = output
            .ok()
            .filter(|result| result.status.success())
            .map(|result| String::from_utf8_lossy(&result.stdout).into_owned())
            .unwrap_or_default();
        let prefix = temp.path().join(format!("page-{page}"));
        let rendered = Command::new("pdftoppm")
            .args([
                "-f",
                &page.to_string(),
                "-l",
                &page.to_string(),
                "-singlefile",
                "-png",
                "-r",
                &dpi,
            ])
            .arg(path)
            .arg(&prefix)
            .output()?;
        if !rendered.status.success() {
            anyhow::bail!("failed to rasterize PDF page {page}")
        }
        let image = prefix.with_extension("png");
        let recognized = ocr.image_to_text(&image);
        let _ = fs::remove_file(image);
        let page_text = match (text.trim().is_empty(), recognized.trim().is_empty()) {
            (true, true) => format!("[Page {page}: no textual content detected]"),
            (false, true) => format!("[Text layer]\n{text}"),
            (true, false) => {
                used_ocr = true;
                format!("[OCR]\n{recognized}")
            }
            (false, false) => {
                used_ocr = true;
                format!("[Text layer]\n{text}\n[OCR]\n{recognized}")
            }
        };
        parts.push(format!("[Page {page}]\n{page_text}"));
        // Exhaustive mode never truncates `text` (its `max_chars` is
        // effectively unlimited — see `extract_inner`), so the per-page
        // segments built alongside it need no truncation either: the two
        // always agree on how much of the document was kept.
        page_segments.push((page, page_text));
    }
    Ok(Extracted {
        text: parts.join("\n\n"),
        method: if used_ocr {
            "pdf-exhaustive-ocr"
        } else {
            "pdf-exhaustive-text"
        }
        .into(),
        ocr_used: used_ocr,
        pages,
        page_segments,
    })
}

/// Bound on how much pdftoppm stderr a single OCR-fallback failure logs.
/// A malformed PDF can make poppler emit stderr without limit (repeated
/// "Syntax Error" / "Bad block header" lines); capping this is what keeps
/// that failure mode from flooding engine.log the way the old inherited-stdio
/// path did.
const STDERR_LOG_CHARS: usize = 2000;

fn pdf(path: &Path, config: &Config, ocr: &TesseractOcr) -> Result<Extracted> {
    let info = pdf_info(path);
    if info.password_required {
        // Bail BEFORE spawning pdftotext/pdftoppm below — both would run
        // and fail identically (12 live-corpus PDFs did exactly this,
        // silently, every one becoming an empty `pdf-text-partial` row). A
        // typed error, not `anyhow::bail!`, so `crate::failure::classify`
        // recognizes this as `Encrypted` by downcasting rather than parsing
        // the message — see `failure.rs`.
        return Err(EncryptedDocument.into());
    }
    let pages = info.pages;
    let text = Command::new("pdftotext")
        .arg(path)
        .arg("-")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).to_string())
        .unwrap_or_default();
    // `pdftotext` inserts a form-feed between pages by default (no `-nopgbrk`
    // is passed anywhere here), which is a free, page-numbered split of text
    // that already went through this exact call — no extra `pdftotext -f -l`
    // invocation per page needed for the common (non-OCR) case.
    let page_segments =
        truncate_page_segments(page_segments_from_form_feeds(&text), config.max_chars);
    let need_ocr = config.ocr == "on"
        || (config.ocr == "auto" && text.trim().chars().count() < 20 * pages.max(1));
    if !need_ocr || !ocr.available {
        return Ok(Extracted {
            text: truncate(text, config.max_chars),
            method: "pdf-text".into(),
            ocr_used: false,
            pages,
            page_segments,
        });
    }
    let temp = tempdir()?;
    let prefix = temp.path().join("page");
    let max_page = pages.max(1).min(config.ocr_max_pages);
    let dpi = config.ocr_dpi.to_string();
    let rendered = Command::new("pdftoppm")
        .args(["-f", "1", "-l", &max_page.to_string(), "-png", "-r", &dpi])
        .arg(path)
        .arg(&prefix)
        .output();
    let succeeded = match &rendered {
        Ok(output) if output.status.success() => true,
        Ok(output) => {
            tracing::warn!(
                path = %path.display(),
                status = %output.status,
                stderr = %truncate(String::from_utf8_lossy(&output.stderr).trim().to_string(), STDERR_LOG_CHARS),
                "pdftoppm failed for OCR fallback"
            );
            false
        }
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "pdftoppm could not be spawned");
            false
        }
    };
    let mut ocr_parts = Vec::new();
    if succeeded {
        let mut images = fs::read_dir(temp.path())?
            .flatten()
            .map(|e| e.path())
            .collect::<Vec<PathBuf>>();
        images.sort();
        for image in images {
            ocr_parts.push(ocr.image_to_text(&image));
            if ocr_parts.iter().map(String::len).sum::<usize>() >= config.max_chars {
                break;
            }
        }
    }
    let ocr_text = ocr_parts.join("\n");
    if ocr_text.trim().len() > text.trim().len() {
        Ok(Extracted {
            text: truncate(format!("{text}\n{ocr_text}"), config.max_chars),
            method: "pdf-ocr".into(),
            ocr_used: true,
            pages,
            // The merged text prefixes the whole-document `pdftotext` output
            // ahead of a flat join of per-image OCR text, so the page split
            // computed from `text` alone no longer lines up with what is
            // actually stored — leave this run page-agnostic rather than
            // attribute OCR'd content to the wrong page.
            page_segments: Vec::new(),
        })
    } else {
        Ok(Extracted {
            text: truncate(text, config.max_chars),
            method: "pdf-text".into(),
            ocr_used: false,
            pages,
            page_segments,
        })
    }
}

/// Split `pdftotext`'s default output on the form-feed page breaks it inserts
/// between pages into one `(page_number, text)` segment per page, 1-based.
/// Blank pages are dropped from the result but not from the numbering — the
/// `enumerate` runs over every split part before `filter_map` discards the
/// empty ones, so a document with a blank page 2 still reports page 3
/// correctly rather than shifting it down to 2.
fn page_segments_from_form_feeds(text: &str) -> Vec<(usize, String)> {
    text.split('\u{000c}')
        .enumerate()
        .filter(|(_, part)| !part.trim().is_empty())
        .map(|(index, part)| (index + 1, part.to_string()))
        .collect()
}

/// Trim page segments so their concatenated length never exceeds `max_chars`
/// — the same budget `truncate` applies to the flat `Extracted::text` — by
/// keeping whole pages up to the budget and, once it runs out mid-page,
/// truncating that page's text and dropping every later page entirely.
/// Without this the chunker (which chunks `page_segments` directly when they
/// are present, see `embedding::chunk_spans`) would embed an amount of text
/// the rest of the pipeline never agreed to.
fn truncate_page_segments(
    segments: Vec<(usize, String)>,
    max_chars: usize,
) -> Vec<(usize, String)> {
    let mut budget = max_chars;
    let mut out = Vec::with_capacity(segments.len());
    for (page, text) in segments {
        if budget == 0 {
            break;
        }
        let kept = truncate(text, budget);
        budget -= kept.chars().count();
        if !kept.trim().is_empty() {
            out.push((page, kept));
        }
    }
    out
}

/// What a single `pdfinfo` invocation tells us before any of pdftotext or
/// pdftoppm gets spawned: the page count, and whether the document is one
/// pdftotext/pdftoppm can actually read.
struct PdfInfo {
    pages: usize,
    /// A user password is required and pdfinfo could not open the document
    /// with poppler's default (empty) password — pdftotext/pdftoppm will
    /// fail identically, so callers must bail rather than march through them
    /// for an empty result. See [`pdf_info`] for how this is distinguished
    /// from an owner-password-only PDF, which opens fine and must NOT set
    /// this.
    password_required: bool,
}

/// Run `pdfinfo` once and read both the page count and the encryption
/// signal off its already-captured output — the same single subprocess this
/// helper always spawned, just no longer throwing its stderr away.
///
/// poppler's `pdfinfo` opens every document with its default (empty) user
/// password before printing anything. Two outcomes matter here:
///
/// - **Opens fine** (no user password, or an owner-password-only PDF whose
///   user password is blank): pdfinfo prints its normal structured field
///   block, including a `Pages:` line and — only when the document actually
///   carries a permissions dictionary — an `Encrypted:` field such as
///   `Encrypted: yes (print:no copy:no ...)`. Text extraction still works
///   here (poppler's CLI tools don't enforce the permission bits), so this
///   case must extract normally regardless of what `Encrypted:` says. We
///   therefore don't need to parse the `Encrypted:` field at all: its mere
///   presence is redundant with `Pages:` being present, which is the signal
///   we already need for the page count.
/// - **Cannot open** (a real user password is set): pdfinfo never reaches
///   the point of printing any structured fields — no `Pages:`, no
///   `Encrypted:` — and instead writes a password complaint to stderr
///   (`Command Line Error: Incorrect password` on the poppler builds this
///   repo has seen; other poppler versions/locales are known to word the
///   lead-in differently, so we match the substring `incorrect password`
///   case-insensitively rather than the whole line).
///
/// So the definitive, version-robust signal that extraction cannot proceed
/// is the STRUCTURED one — no `Pages:` field, i.e. `pages == 0` — with the
/// stderr text used only to confirm the reason is a password and not some
/// other open failure (a corrupt file also yields `pages == 0` but no
/// password wording, and must keep classifying as a decode/unknown failure,
/// not `encrypted`).
fn pdf_info(path: &Path) -> PdfInfo {
    let output = Command::new("pdfinfo").arg(path).output().ok();
    let stdout = output
        .as_ref()
        .map(|out| String::from_utf8_lossy(&out.stdout).into_owned())
        .unwrap_or_default();
    let stderr = output
        .as_ref()
        .map(|out| String::from_utf8_lossy(&out.stderr).into_owned())
        .unwrap_or_default();
    parse_pdf_info(&stdout, &stderr)
}

/// The pure half of [`pdf_info`]: turn `pdfinfo`'s captured stdout/stderr
/// into a [`PdfInfo`]. Split out from the `Command` invocation so the
/// classification rule itself — the part with version/wording risk — is
/// testable against real captured `pdfinfo` output without needing poppler
/// installed wherever the tests run.
fn parse_pdf_info(stdout: &str, stderr: &str) -> PdfInfo {
    let pages = stdout
        .lines()
        .find(|line| line.starts_with("Pages:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|n| n.parse().ok())
        .unwrap_or(0);
    let password_required =
        pages == 0 && stderr.to_ascii_lowercase().contains("incorrect password");
    PdfInfo {
        pages,
        password_required,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extracted(name: &str) -> Extracted {
        let mut config = Config::default();
        config.ocr = "off".into();
        config.finalize();
        let temp = tempdir().unwrap();
        let path = temp.path().join(name);
        fs::write(&path, b"not a real document").unwrap();
        // Exactly how the walker derives `ext`, empty extension included — the
        // 26,442 extensionless rows on the live corpus reach `extract` this way.
        let ext = path
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
            .unwrap_or_default();
        extract(
            &path,
            &ext,
            19,
            &config,
            &TesseractOcr::new(&config),
            &Transcriber::new(&config),
        )
        .unwrap()
    }

    #[test]
    fn an_extension_no_extractor_handles_is_terminal_not_a_failure() {
        // The 48k `.o`/`.xnb`/`.rlib` rows: nothing in this build reads them, and
        // nothing ever will, so they must land on the terminal `excluded:` marker
        // rather than the `name-only-partial` that resume re-attempted forever.
        for name in ["build.o", "assets.xnb", "libcore.rlib", "no_extension"] {
            assert_eq!(
                extracted(name).method,
                "excluded:unsupported",
                "{name} has no extractor in this build"
            );
        }
    }

    #[test]
    fn a_format_this_build_handles_stays_retryable_when_a_stage_declines() {
        // OCR is off here, so the image arm declines — but the capability exists,
        // and a later run with OCR on must read the file. Writing it off as
        // unsupported would make a config choice permanent.
        assert_eq!(extracted("scan.png").method, "name-only");
    }

    #[test]
    fn heic_is_dispatched_by_name_not_written_off_as_unsupported() {
        // Before this workstream `.heic` fell through every arm to the
        // catch-all `_ => Ok(Extracted::unsupported())`, i.e. the SAME
        // terminal `excluded:unsupported` `an_extension_no_extractor_handles_
        // is_terminal_not_a_failure` pins for `.o`/`.xnb`/genuinely-unhandled
        // extensions — permanently freezing every `.heic` row the moment the
        // attempt cap (`pipeline::MAX_ATTEMPTS`) landed, on the verdict of a
        // build with no HEIC path at all. With OCR off (as here) the new HEIC
        // arm declines on the same runtime condition the PNG arm above does,
        // landing on `name-only`, not `excluded:unsupported` — proof the
        // extension is dispatched by name rather than fell through.
        assert_eq!(extracted("photo.heic").method, "name-only");
    }

    /// Real ffmpeg + real tesseract, gated on both actually being on `PATH` —
    /// same shape as `tests/indexing.rs`'s `pdfinfo_available`-gated B2 tests.
    /// Garbage bytes named `.heic`: ffmpeg opens the file, cannot make sense
    /// of it, and exits non-zero — the "ffmpeg ran but rejected the bytes"
    /// shape `heic()`'s doc comment describes, which must classify as
    /// [`crate::failure::FailureClass::Decode`], not `Unknown`. The
    /// successful-decode counterpart is
    /// `heic_dispatch_decodes_a_real_frame_and_ocrs_it` below, over
    /// `tests/fixtures/heic-tiny.heic`.
    #[test]
    fn a_heic_ffmpeg_cannot_decode_classifies_as_decode_not_unknown() {
        let mut config = Config::default();
        config.ocr = "on".into();
        config.finalize();
        let ocr = TesseractOcr::new(&config);
        if !ffmpeg_available() {
            eprintln!("skipping HEIC decode-failure live test: ffmpeg not on PATH");
            return;
        }
        if !ocr.available {
            eprintln!("skipping HEIC decode-failure live test: tesseract not on PATH");
            return;
        }
        let temp = tempdir().unwrap();
        let path = temp.path().join("corrupt.heic");
        fs::write(&path, b"not a real heic file").unwrap();
        let error = extract(
            &path,
            ".heic",
            21,
            &config,
            &ocr,
            &Transcriber::new(&config),
        )
        .expect_err("garbage bytes are not a decodable HEIC frame");
        assert_eq!(
            crate::failure::classify(&error),
            crate::failure::FailureClass::Decode
        );
    }

    fn ffmpeg_available() -> bool {
        Command::new("ffmpeg")
            .arg("-version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    /// The successful-decode counterpart to the decode-failure test above,
    /// over a real (if synthetic) HEIC-shaped file — `tests/fixtures/
    /// heic-tiny.heic`: a single 64x64 red HEVC still frame in an ISOBMFF
    /// container, produced with
    /// `ffmpeg -f lavfi -i color=c=red:s=64x64 -frames:v 1 -c:v hevc
    /// -tag:v hvc1 -f mp4 heic-tiny.heic` (ffmpeg has no HEIC muxer to encode
    /// a "real" `major_brand: heic` file with — see `heic()`'s doc comment —
    /// so `-f mp4` is forced; the demuxer that reads it back, `mov,mp4,m4a,
    /// 3gp,3g2,mj2`, is the SAME one that opens a real Apple `.heic`, whose
    /// tile-grid HEVC decode was confirmed BY HAND against a real
    /// iCloud-export file on this box on 2026-07-25 — this fixture pins the
    /// end-to-end wiring (dispatch → ffmpeg → temp JPEG → tesseract →
    /// `method`), not the mov demuxer's HEIF-brand handling itself, which a
    /// synthetic single-tile file cannot exercise).
    #[test]
    fn heic_dispatch_decodes_a_real_frame_and_ocrs_it() {
        let mut config = Config::default();
        config.ocr = "on".into();
        config.finalize();
        let ocr = TesseractOcr::new(&config);
        if !ffmpeg_available() {
            eprintln!("skipping HEIC decode-success live test: ffmpeg not on PATH");
            return;
        }
        if !ocr.available {
            eprintln!("skipping HEIC decode-success live test: tesseract not on PATH");
            return;
        }
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/heic-tiny.heic");
        let extracted = extract(
            &fixture,
            ".heic",
            fs::metadata(&fixture).unwrap().len(),
            &config,
            &ocr,
            &Transcriber::new(&config),
        )
        .expect("a decodable HEIC frame must extract, not error");
        assert_eq!(extracted.method, "heic-ocr");
        assert_eq!(extracted.pages, 1);
    }

    #[test]
    fn dispatched_extensions_are_not_reported_unsupported() {
        // DOCUMENT_EXTS is a copy of the dispatch's own arms, kept only so the
        // capability fingerprint moves when a format is added. This is what stops
        // the copy from drifting: every extension it claims must reach a real
        // extractor, which on this junk input either errors or extracts nothing —
        // never `excluded:unsupported`.
        let mut config = Config::default();
        config.ocr = "off".into();
        config.finalize();
        let temp = tempdir().unwrap();
        for ext in DOCUMENT_EXTS {
            let path = temp.path().join(format!("sample{ext}"));
            fs::write(&path, b"not a real document").unwrap();
            let method = extract(
                &path,
                ext,
                19,
                &config,
                &TesseractOcr::new(&config),
                &Transcriber::new(&config),
            )
            .map(|extracted| extracted.method)
            .unwrap_or_else(|_| "error".into());
            assert_ne!(
                method, "excluded:unsupported",
                "{ext} is dispatched by name"
            );
        }
    }

    #[test]
    fn the_extractor_revision_is_stable_and_moves_with_the_capability_tables() {
        // Stable, because a fingerprint that changed between two runs of the same
        // binary would hand every capped row a free attempt on every resume.
        assert_eq!(extractor_revision(), extractor_revision());
        assert_eq!(extractor_revision().len(), 12);
        // And it is derived from the tables rather than hand-maintained, so
        // teaching this build a new format moves it without anyone remembering
        // to. `.avif` stands in for "the next format this build learns" —
        // `.heic` no longer works for that: it is one of the real tables
        // (`HEIC_EXTS`) now, so hashing it a second time here would no longer
        // differ from `extractor_revision()` and this assertion would start
        // failing for the wrong reason.
        let mut with_avif = Sha1::new();
        with_avif.update(env!("CARGO_PKG_VERSION"));
        for table in [
            TEXT_EXTS,
            CODE_EXTS,
            IMAGE_EXTS,
            HEIC_EXTS,
            EMAIL_EXTS,
            AUDIO_EXTS,
            VIDEO_EXTS,
            ARCHIVE_EXTS,
            DOCUMENT_EXTS,
        ] {
            for ext in table {
                with_avif.update(ext);
            }
        }
        with_avif.update(".avif");
        assert_ne!(
            extractor_revision(),
            format!("{:x}", with_avif.finalize())[..12].to_string()
        );
    }

    /// `parse_pdf_info` classification, pinned against REAL `pdfinfo`
    /// (poppler 25.11.0) output captured by hand — not fabricated strings —
    /// so these tests fail if the actual field/message wording this build
    /// depends on ever drifts. Generated with a tiny hand-built one-page PDF
    /// (plain, then re-saved through `pypdf` with `user_password=""` and
    /// with a real user password) and Calibre's bundled poppler binaries;
    /// see B2's task notes for how the fixtures were produced. No live
    /// process spawn here — `pdf_info()` (the thin wrapper that actually
    /// runs `pdfinfo`) is exercised separately by the end-to-end tests in
    /// `tests/indexing.rs`, gated on poppler actually being on `PATH`.
    mod pdf_info_classification {
        use super::*;

        /// A normal, unencrypted PDF: `Pages:` present, no stderr at all.
        /// Must extract normally.
        #[test]
        fn an_unencrypted_pdf_is_not_password_required() {
            let stdout = "Producer:        pypdf\n\
                Pages:           1\n\
                Encrypted:       no\n\
                Page size:       200 x 200 pts\n\
                PDF version:     1.4\n";
            let info = parse_pdf_info(stdout, "");
            assert_eq!(info.pages, 1);
            assert!(!info.password_required);
        }

        /// Owner-password-only: blank user password opens it fine, so
        /// pdfinfo prints the full field block INCLUDING `Pages:` — the
        /// `Encrypted: yes (print:.. copy:..)` line is present but must not
        /// by itself trigger a bail, because pdftotext/pdftoppm can still
        /// read the document (poppler's CLI tools don't enforce the
        /// permission bits). This is the case B2's task explicitly calls
        /// out as must-not-block.
        #[test]
        fn an_owner_password_only_pdf_is_not_password_required_even_with_all_permissions_denied() {
            let stdout = "Producer:        pypdf\n\
                Pages:           1\n\
                Encrypted:       yes (print:no copy:no change:no addNotes:no algorithm:RC4)\n\
                Page size:       200 x 200 pts\n\
                PDF version:     1.4\n";
            let info = parse_pdf_info(stdout, "");
            assert_eq!(info.pages, 1);
            assert!(!info.password_required);
        }

        /// A real user password: pdfinfo never reaches the point of
        /// printing ANY structured field — no `Pages:`, no `Encrypted:` —
        /// stdout is empty and the password complaint lands on stderr. This
        /// is the exact live-corpus shape (12 PDFs, evidence doc §pdfinfo).
        #[test]
        fn a_user_password_required_pdf_is_password_required() {
            let info = parse_pdf_info("", "Command Line Error: Incorrect password\n");
            assert_eq!(info.pages, 0);
            assert!(info.password_required);
        }

        /// Poppler version/build wording varies (older builds, different
        /// leading text) — match must be robust to that, so this checks a
        /// handful of plausible variants rather than the one exact string
        /// above, per the task's "keep it robust to poppler version
        /// wording" requirement.
        #[test]
        fn password_detection_is_robust_to_poppler_wording_variants() {
            for stderr in [
                "Command Line Error: Incorrect password\n",
                "Command Line Error: Incorrect password",
                "Error: Incorrect password\n",
                "incorrect password\n",
                "INCORRECT PASSWORD\n",
                "Syntax Warning: Incorrect password\n",
            ] {
                let info = parse_pdf_info("", stderr);
                assert!(info.password_required, "{stderr:?} should be detected");
            }
        }

        /// A corrupt/truncated file also yields no `Pages:` field, but for a
        /// different reason — it must NOT classify as encrypted, or a
        /// genuinely damaged (not password-protected) PDF would be
        /// misreported and its resume behavior would follow the wrong path.
        /// Real captured stderr from running pdfinfo against 27 bytes of
        /// non-PDF garbage.
        #[test]
        fn a_corrupt_non_pdf_file_is_not_password_required() {
            let stderr = "Syntax Warning: May not be a PDF file (continuing anyway)\n\
                Syntax Error: Couldn't find trailer dictionary\n\
                Syntax Error: Couldn't find trailer dictionary\n\
                Syntax Error: Couldn't read xref table\n";
            let info = parse_pdf_info("", stderr);
            assert_eq!(info.pages, 0);
            assert!(!info.password_required);
        }

        /// pdfinfo missing from `PATH` entirely: `pdf_info()`'s `.output()`
        /// call fails and both strings are empty, same as this function
        /// sees it. Must not be misread as an encrypted document — an
        /// unrelated environment problem is `pages == 0`/`Unknown`, not
        /// `Encrypted`, downstream in `pdf()`/`pdf_exhaustive()`.
        #[test]
        fn empty_output_from_a_missing_pdfinfo_binary_is_not_password_required() {
            let info = parse_pdf_info("", "");
            assert_eq!(info.pages, 0);
            assert!(!info.password_required);
        }
    }

    // The pdftoppm OCR-fallback path itself needs poppler + tesseract binaries
    // and can't be exercised as a plain unit test; what's testable in-process
    // is the bound that keeps a pathological PDF's stderr from flooding
    // engine.log the way the old inherited-stdio call did.
    #[test]
    fn stderr_log_bound_caps_pathological_output() {
        let flood = "Syntax Error: Couldn't find trailer dictionary\n".repeat(500);
        assert!(flood.chars().count() > STDERR_LOG_CHARS);
        let capped = truncate(flood, STDERR_LOG_CHARS);
        assert_eq!(capped.chars().count(), STDERR_LOG_CHARS);
    }

    #[test]
    fn stderr_log_bound_leaves_short_output_untouched() {
        let short = "Command Line Error: Incorrect password".to_string();
        assert_eq!(truncate(short.clone(), STDERR_LOG_CHARS), short);
    }
}
