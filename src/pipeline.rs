use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rayon::prelude::*;
use sha1::{Digest, Sha1};

use crate::config::Config;
use crate::extract::extract;
use crate::model::{FileRec, IndexStats, ProcessedFile};
use crate::normalize::Normalizer;
use crate::ocr::TesseractOcr;
use crate::store::{analyze, connect, IndexStore};
use crate::walker::walk;

pub struct IndexRequest<'a> {
    pub paths: &'a [PathBuf],
    pub out: &'a Path,
    pub config: Config,
    pub resume: bool,
    pub artifacts: bool,
}

pub fn run_index(mut request: IndexRequest<'_>) -> Result<IndexStats> {
    request.config.finalize();
    let started = Instant::now();
    let normalizer = Arc::new(Normalizer::load(&request.config));
    let ocr = Arc::new(TesseractOcr::new(&request.config));
    let mut store = IndexStore::open(
        request.out,
        &request.config,
        request.resume,
        request.artifacts,
    )?;
    let existing = if request.resume {
        store.existing_keys()?
    } else {
        Default::default()
    };
    let mut records = walk(request.paths, &request.config);
    let before = records.len();
    if request.resume {
        records.retain(|record| {
            existing
                .get(&record.path)
                .map(|(size, mtime)| *size != record.size || *mtime != record.mtime as i64)
                .unwrap_or(true)
        });
    }
    let skipped = before - records.len();
    eprintln!(
        "llm-index {} -> {} file(s), OCR {}, workers {}",
        env!("CARGO_PKG_VERSION"),
        records.len(),
        if ocr.available {
            &ocr.langs
        } else {
            "unavailable"
        },
        request.config.workers
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(request.config.workers)
        .build()?;
    let config = Arc::new(request.config.clone());
    let processed = pool.install(|| {
        records
            .par_iter()
            .map(|record| process(record.clone(), &config, &normalizer, &ocr))
            .collect::<Vec<_>>()
    });
    let mut stats = IndexStats {
        skipped,
        ..Default::default()
    };
    for file in processed {
        stats.files += 1;
        stats.bytes += file.rec.size;
        stats.ocr_files += usize::from(file.ocr_used);
        stats.errors += usize::from(file.method.starts_with("error:"));
        store.add(&file, now())?;
    }
    store.finish()?;
    stats.elapsed_seconds = started.elapsed().as_secs_f64();
    if request.artifacts {
        write_analysis(request.out, request.paths)?;
    }
    Ok(stats)
}

fn process(
    record: FileRec,
    config: &Config,
    normalizer: &Normalizer,
    ocr: &TesseractOcr,
) -> ProcessedFile {
    let path = Path::new(&record.path);
    match extract(path, &record.ext, record.size, config, ocr) {
        Ok(extracted) => {
            let content = if extracted.text.trim().is_empty() {
                format!("{} {}", record.name, record.dir)
            } else {
                extracted.text
            };
            let token_source = format!(
                "{} {} {}",
                content,
                record.name,
                record.path.replace(['\\', '/'], " ")
            );
            let lang = normalizer.detect_lang(&content);
            let tokens = normalizer.enrich(&token_source, &lang);
            let hash = config.hash.then(|| sha1(path)).flatten();
            ProcessedFile {
                rec: record,
                content,
                tokens,
                lang,
                method: extracted.method,
                ocr_used: extracted.ocr_used,
                pages: extracted.pages,
                sha1: hash,
            }
        }
        Err(error) => ProcessedFile {
            content: format!("{} {}", record.name, record.dir),
            tokens: Vec::new(),
            lang: "und".into(),
            method: format!("error:{}", error_chain_name(&error)),
            ocr_used: false,
            pages: 0,
            sha1: None,
            rec: record,
        },
    }
}

fn sha1(path: &Path) -> Option<String> {
    let mut file = fs::File::open(path).ok()?;
    let mut hash = Sha1::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).ok()?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Some(format!("{:x}", hash.finalize()))
}

fn error_chain_name(error: &anyhow::Error) -> String {
    error
        .root_cause()
        .to_string()
        .split_whitespace()
        .next()
        .unwrap_or("extract")
        .to_string()
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or_default()
}

fn write_analysis(out: &Path, paths: &[PathBuf]) -> Result<()> {
    let connection = connect(out)?;
    let report = analyze(&connection)?;
    let reports = out.join("reports");
    fs::create_dir_all(&reports)?;
    fs::write(
        reports.join("analysis.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    let mut markdown = format!(
        "# Index analysis\n\n**Sources:** {}\n\n",
        paths
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    markdown.push_str(&format!(
        "- Files: {}\n- Bytes: {}\n- OCR files: {}\n",
        report["files"], report["bytes"], report["ocr_files"]
    ));
    fs::write(reports.join("analysis.md"), markdown).context("writing analysis report")?;
    Ok(())
}
