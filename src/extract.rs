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
) -> Result<Extracted> {
    if size > config.max_bytes || config.skip_ext(ext) {
        return Ok(Extracted::empty());
    }
    if TEXT_EXTS.contains(&ext) || CODE_EXTS.contains(&ext) {
        let raw = read_limited(path, config.max_chars.saturating_mul(4))?;
        return Ok(Extracted {
            text: decode(&raw, config.max_chars),
            method: "text".into(),
            ocr_used: false,
            pages: 0,
        });
    }
    if EMAIL_EXTS.contains(&ext) {
        return Ok(Extracted {
            text: email(path, config.max_chars)?,
            method: "email".into(),
            ocr_used: false,
            pages: 0,
        });
    }
    match ext {
        ".pdf" => pdf(path, config, ocr),
        ".docx" => Ok(Extracted {
            text: office_xml(path, &["word/"], config.max_chars)?,
            method: "docx".into(),
            ocr_used: false,
            pages: 0,
        }),
        ".xlsx" | ".xlsm" => Ok(Extracted {
            text: office_xml(path, &["xl/"], config.max_chars)?,
            method: "xlsx".into(),
            ocr_used: false,
            pages: 0,
        }),
        ".pptx" => Ok(Extracted {
            text: office_xml(path, &["ppt/slides/", "ppt/notesSlides/"], config.max_chars)?,
            method: "pptx".into(),
            ocr_used: false,
            pages: 0,
        }),
        _ if IMAGE_EXTS.contains(&ext)
            && matches!(config.ocr.as_str(), "auto" | "on")
            && ocr.available =>
        {
            let text = truncate(ocr.image_to_text(path), config.max_chars);
            Ok(Extracted {
                ocr_used: !text.trim().is_empty(),
                text,
                method: "ocr".into(),
                pages: 1,
            })
        }
        _ => Ok(Extracted::empty()),
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

fn office_xml(path: &Path, prefixes: &[&str], max_chars: usize) -> Result<String> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    let mut parts = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        if !name.ends_with(".xml") || !prefixes.iter().any(|prefix| name.starts_with(prefix)) {
            continue;
        }
        let mut xml = Vec::new();
        entry.read_to_end(&mut xml)?;
        parts.extend(xml_text(&xml));
        if parts.iter().map(String::len).sum::<usize>() >= max_chars {
            break;
        }
    }
    Ok(truncate(parts.join("\n"), max_chars))
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
    let status = Command::new("pdftoppm")
        .args(["-f", "1", "-l", &max_page.to_string(), "-png", "-r", "200"])
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
