use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rayon::prelude::*;
use sha1::{Digest, Sha1};

use crate::config::Config;
use crate::embedding::Embedder;
use crate::extract::extract;
use crate::media::Transcriber;
use crate::model::{FileRec, IndexStats, ProcessedFile};
use crate::normalize::Normalizer;
use crate::ocr::TesseractOcr;
use crate::store::{analyze, connect, database_path, remove_database, IndexStore};
use crate::vision::{
    is_video_ext, is_vision_ext, needs_vision_reprocess, VisionAnalyzer, VisionMode, VisionResult,
};
use crate::walker::walk;

pub struct IndexRequest<'a> {
    pub paths: &'a [PathBuf],
    pub out: &'a Path,
    pub config: Config,
    pub resume: bool,
    /// Delete an existing database before indexing into it. Honoured here
    /// rather than by the caller so the deletion happens after every
    /// prerequisite this run needs has been loaded and checked — see
    /// `run_index`. Ignored when `resume` is set, which exists to keep what is
    /// there.
    pub overwrite: bool,
    pub artifacts: bool,
    pub include_paths: Option<HashSet<String>>,
    pub cancellation: Option<Arc<AtomicBool>>,
    pub progress: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>,
}

pub fn run_index(mut request: IndexRequest<'_>) -> Result<IndexStats> {
    request.config.finalize();
    let started = Instant::now();
    let normalizer = Arc::new(Normalizer::load(&request.config));
    let ocr = Arc::new(TesseractOcr::new(&request.config));
    let transcriber = Arc::new(Transcriber::new(&request.config));
    let vision = Arc::new(VisionAnalyzer::new(&request.config)?);
    let mut embedder = Embedder::new(&request.config)?;
    // The last possible moment to destroy the previous corpus. Everything that
    // can predictably fail on an operator mistake — an unreadable config, a
    // missing or corrupt vision model, an embedding model that has to be
    // fetched into a cold cache — has already run above, so none of them can
    // leave the destination deleted and nothing in its place. What remains
    // between here and the first write is `IndexStore::open` itself: a failure
    // there (unwritable directory, corrupt schema) still costs the old corpus.
    if request.overwrite && !request.resume {
        remove_database(request.out)?;
    }
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
    let vision_modes = if request.resume && request.config.vision.max != VisionMode::Off {
        store.existing_vision_modes()?
    } else {
        HashMap::new()
    };
    let mut records = walk(request.paths, &request.config);
    let current = records
        .iter()
        .map(|record| record.path.clone())
        .collect::<HashSet<_>>();
    // Pruning deletes rows for sources that have disappeared. It used to run
    // against a throwaway copy that a failed run discarded; now it lands in the
    // published corpus and the next batch commit makes it permanent. An empty
    // walk is therefore not treated as "every source was deleted" — over a
    // mounted tree that is far more often a root that failed to mount or a
    // mistyped path, and that reading would silently empty the whole corpus for
    // a run that then reports success. Rebuilding from empty is what
    // `overwrite` is for, and it is explicit.
    let removed = if request.resume && !current.is_empty() {
        store.prune_missing(&current)?
    } else {
        0
    };
    if let Some(include_paths) = &request.include_paths {
        records.retain(|record| include_paths.contains(&record.path));
    }
    let before = records.len();
    if request.resume {
        records.retain(|record| {
            existing
                .get(&record.path)
                .map(|(size, mtime, method, has_chunks)| {
                    *size != record.size
                        || *mtime != record.mtime as i64
                        || !*has_chunks
                        || incomplete_method(method)
                        || (request.config.ocr == "exhaustive"
                            && record.ext == ".pdf"
                            && !method.starts_with("pdf-exhaustive"))
                        || needs_vision_reprocess(
                            request.config.vision.max,
                            vision_modes
                                .get(&record.path)
                                .and_then(|mode| mode.parse().ok()),
                            &record.ext,
                        )
                })
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
    let total = records.len();
    if let Some(progress) = &request.progress {
        progress(0, total);
    }
    let completed = Arc::new(AtomicUsize::new(0));
    let cancellation = request.cancellation.clone();
    let progress = request.progress.clone();
    let mut stats = IndexStats {
        skipped,
        removed,
        ..Default::default()
    };
    // Extraction fans out over the rayon pool and hands each finished file to
    // this thread, which owns the embedder and the (non-Sync) SQLite connection
    // and is therefore the single writer. Streaming rather than collecting keeps
    // committed work on disk as the run proceeds and bounds peak memory to the
    // channel depth instead of the whole corpus: a large drive no longer has to
    // fit every extracted document in RAM before a single row is written.
    let (sender, receiver) = mpsc::sync_channel::<ProcessedFile>(config.workers.max(1) * 4);
    let outcome = std::thread::scope(|scope| -> Result<bool> {
        scope.spawn(move || {
            let stopped = AtomicBool::new(false);
            pool.install(|| {
                records.par_iter().for_each(|record| {
                    if stopped.load(Ordering::Relaxed)
                        || cancellation
                            .as_ref()
                            .is_some_and(|flag| flag.load(Ordering::Relaxed))
                    {
                        return;
                    }
                    let file = process(
                        record.clone(),
                        &config,
                        &normalizer,
                        &ocr,
                        &transcriber,
                        &vision,
                    );
                    let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    if let Some(progress) = &progress {
                        progress(count, total);
                    }
                    // The writer hangs up (error or cancellation) by dropping the
                    // receiver, which also unblocks a full channel; the remaining
                    // records then wind down without extracting anything.
                    if sender.send(file).is_err() {
                        stopped.store(true, Ordering::Relaxed);
                    }
                })
            })
        });
        // Consumed by value: returning early here drops the receiver, so the
        // extraction side is never left blocked on a send.
        for mut file in receiver {
            if request
                .cancellation
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Relaxed))
            {
                return Ok(true);
            }
            if !incomplete_method(&file.method) && !file.method.starts_with("excluded:") {
                file.chunks = embedder.embed_document(&file.content)?;
            }
            stats.files += 1;
            stats.bytes += file.rec.size;
            stats.ocr_files += usize::from(file.ocr_used);
            stats.errors += usize::from(file.method.starts_with("error:"));
            stats.incomplete += usize::from(incomplete_method(&file.method));
            stats.embedded_chunks += file.chunks.len();
            stats.vision_files += usize::from(file.vision.is_some());
            store.add(&file, now())?;
        }
        Ok(false)
    });
    // Commit before propagating anything: success, cancellation and a mid-run
    // failure all leave their finished files in the destination corpus, which is
    // what a later resume continues from. The run's own error still wins over a
    // commit error, since it is the one that explains the outcome.
    let committed = store.finish();
    let cancelled = outcome?;
    committed?;
    if cancelled {
        anyhow::bail!("indexing cancelled; {} file(s) committed", stats.files)
    }
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
    transcriber: &Transcriber,
    analyzer: &VisionAnalyzer,
) -> ProcessedFile {
    let path = Path::new(&record.path);
    match extract(path, &record.ext, record.size, config, ocr, transcriber) {
        Ok(extracted) => {
            let empty = extracted.text.trim().is_empty();
            let mut content = nfc(if empty {
                format!("{} {}", record.name, record.dir)
            } else {
                extracted.text
            });
            // Vision composes WITH the extracted (OCR) text: append the
            // searchable `[vision]` block, leaving `method` untouched so
            // consumers keying on it are unaffected.
            let vision = run_vision(analyzer, path, &record.ext, config.vision.max);
            if let Some(block) = vision.as_ref().and_then(VisionResult::content_block) {
                content = nfc(format!("{content}\n{block}"));
            }
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
                method: if empty && !extracted.method.starts_with("excluded:") {
                    format!("{}-partial", extracted.method)
                } else {
                    extracted.method
                },
                ocr_used: extracted.ocr_used,
                pages: extracted.pages,
                sha1: hash,
                chunks: Vec::new(),
                vision,
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
            chunks: Vec::new(),
            vision: None,
            rec: record,
        },
    }
}

/// Run vision analysis for an eligible file, returning a result only when a tier
/// actually ran (or recorded a decode error). Non-vision extensions and the
/// off-path return `None`, so nothing is attached and the off-path is inert.
fn run_vision(
    analyzer: &VisionAnalyzer,
    path: &Path,
    ext: &str,
    mode: VisionMode,
) -> Option<VisionResult> {
    if mode == VisionMode::Off || !is_vision_ext(ext) {
        return None;
    }
    let result = if is_video_ext(ext) {
        analyzer.analyze_video(path, mode)
    } else {
        analyzer.analyze_image(path, mode)
    };
    (result.mode != VisionMode::Off || result.error.is_some()).then_some(result)
}

fn incomplete_method(method: &str) -> bool {
    method == "name-only" || method.starts_with("error:") || method.ends_with("-partial")
}

/// Normalize content to Unicode NFC at the storage boundary. OCR (Tesseract)
/// emits Vietnamese diacritics as decomposed sequences (base letter + combining
/// marks); stored and displayed content must be precomposed NFC so search and
/// rendering behave consistently. Fast-path already-NFC text to avoid a re-alloc.
fn nfc(text: String) -> String {
    use unicode_normalization::{is_nfc, UnicodeNormalization};
    if is_nfc(&text) {
        text
    } else {
        text.nfc().collect()
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
    // `out` may name the database itself rather than a directory, the shape
    // service jobs use; reports belong beside it either way.
    let database = database_path(out);
    let connection = connect(&database)?;
    let report = analyze(&connection)?;
    let reports = database
        .parent()
        .unwrap_or(Path::new("."))
        .join("reports");
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

#[cfg(test)]
mod tests {
    use super::nfc;

    #[test]
    fn nfc_precomposes_decomposed_vietnamese() {
        // "tiếng Việt" typed with combining marks, as OCR would emit it.
        let decomposed = "tie\u{0302}\u{0301}ng Vie\u{0323}\u{0302}t".to_string();
        assert_eq!(nfc(decomposed), "tiếng Việt");
    }

    #[test]
    fn nfc_leaves_precomposed_unchanged() {
        let precomposed = "tiếng Việt".to_string();
        assert_eq!(nfc(precomposed.clone()), precomposed);
    }
}
