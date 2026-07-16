use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use chardetng::EncodingDetector;
use mailparse::{parse_mail, MailHeaderMap, ParsedMail};
use quick_xml::events::Event;
use quick_xml::Reader;
use tempfile::tempdir;
use zip::ZipArchive;

use crate::config::Config;
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
const EMAIL_EXTS: &[&str] = &[".eml", ".wdseml", ".emlx"];
const AUDIO_EXTS: &[&str] = &[".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"];
const VIDEO_EXTS: &[&str] = &[".mkv", ".mp4", ".mov", ".m4v", ".avi", ".webm"];
const ARCHIVE_EXTS: &[&str] = &[".zip", ".rar", ".7z", ".tar", ".gz", ".tgz"];

#[derive(Debug, Clone)]
pub struct Extracted {
    pub text: String,
    pub method: String,
    pub ocr_used: bool,
    pub pages: usize,
}

impl Extracted {
    fn empty() -> Self {
        Self {
            text: String::new(),
            method: "name-only".into(),
            ocr_used: false,
            pages: 0,
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
        });
    }
    if EMAIL_EXTS.contains(&ext) {
        return Ok(Extracted {
            text: email(path, max_chars)?,
            method: "email".into(),
            ocr_used: false,
            pages: 0,
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
            })
        }
        _ if AUDIO_EXTS.contains(&ext) || VIDEO_EXTS.contains(&ext) => {
            media(path, ext, max_chars, ocr, transcriber)
        }
        _ if ARCHIVE_EXTS.contains(&ext) && depth < 4 => {
            archive(path, config, ocr, transcriber, depth + 1, max_chars)
        }
        _ => Ok(Extracted::empty()),
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
) -> Result<Extracted> {
    if !transcriber.available() {
        anyhow::bail!("local Whisper transcription model is unavailable")
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
    })
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
        anyhow::bail!("Tesseract is unavailable for exhaustive PDF OCR")
    }
    let pages = pdf_pages(path);
    if pages == 0 {
        anyhow::bail!("PDF page count is unavailable")
    }
    let temp = tempdir()?;
    let dpi = config.ocr_dpi.to_string();
    let mut parts = Vec::with_capacity(pages);
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
    })
}

fn pdf(path: &Path, config: &Config, ocr: &TesseractOcr) -> Result<Extracted> {
    let pages = pdf_pages(path);
    let text = Command::new("pdftotext")
        .arg(path)
        .arg("-")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).to_string())
        .unwrap_or_default();
    let need_ocr = config.ocr == "on"
        || (config.ocr == "auto" && text.trim().chars().count() < 20 * pages.max(1));
    if !need_ocr || !ocr.available {
        return Ok(Extracted {
            text: truncate(text, config.max_chars),
            method: "pdf-text".into(),
            ocr_used: false,
            pages,
        });
    }
    let temp = tempdir()?;
    let prefix = temp.path().join("page");
    let max_page = pages.max(1).min(config.ocr_max_pages);
    let dpi = config.ocr_dpi.to_string();
    let status = Command::new("pdftoppm")
        .args(["-f", "1", "-l", &max_page.to_string(), "-png", "-r", &dpi])
        .arg(path)
        .arg(&prefix)
        .status();
    let mut ocr_parts = Vec::new();
    if status.map(|s| s.success()).unwrap_or(false) {
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
        })
    } else {
        Ok(Extracted {
            text: truncate(text, config.max_chars),
            method: "pdf-text".into(),
            ocr_used: false,
            pages,
        })
    }
}

fn pdf_pages(path: &Path) -> usize {
    let output = Command::new("pdfinfo").arg(path).output().ok();
    output
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("Pages:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|n| n.parse().ok())
        })
        .unwrap_or(0)
}
