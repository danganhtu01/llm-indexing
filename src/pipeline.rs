use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rayon::prelude::*;
use sha1::{Digest, Sha1};

use crate::config::{Config, MAX_WORKERS};
use crate::embedding::Embedder;
use crate::extract::extract;
use crate::media::Transcriber;
use crate::model::{FileRec, IndexStats, ProcessedFile};
use crate::normalize::Normalizer;
use crate::ocr::TesseractOcr;
use crate::runtime::{Admission, EmbedderPool, RuntimeKnobs, EMBED_RANGE};
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
    /// Live stage settings for this run, shared with whoever may retune it
    /// mid-flight. Carried alongside `cancellation` and reaching the workers by
    /// the same route, because it is the same problem: an out-of-band signal
    /// that must land on work already running. `None` derives fixed settings
    /// from the config, which is what the CLI does.
    pub runtime: Option<Arc<RuntimeKnobs>>,
    pub progress: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>,
}

pub fn run_index(mut request: IndexRequest<'_>) -> Result<IndexStats> {
    request.config.finalize();
    let started = Instant::now();
    let runtime = request
        .runtime
        .clone()
        .unwrap_or_else(|| Arc::new(RuntimeKnobs::from_config(&request.config)));
    let normalizer = Arc::new(Normalizer::load(&request.config));
    let ocr = Arc::new(TesseractOcr::new(&request.config).with_runtime(runtime.clone()));
    let transcriber = Arc::new(Transcriber::new(&request.config));
    let vision = Arc::new(VisionAnalyzer::new(&request.config)?);
    // Built eagerly, and BEFORE the overwrite below, for the reason spelled out
    // there: a cold model cache is fetched over the network and that failure must
    // not land after the previous corpus has been deleted. The embedder pool is
    // seeded with this instance and grows lazily from it.
    let first_embedder = Embedder::with_intra_threads(&request.config, runtime.embed_intra_threads())?;
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
        // The live target, not `config.workers` — they are the same at startup
        // but diverge the moment the extract stage is retuned, and a log line
        // that keeps reporting the seed value would be actively misleading.
        runtime.extract()
    );
    // The pool is built ONCE at the ceiling and never rebuilt. How many of its
    // threads are actually working is decided by `admission`, which re-reads the
    // live `extract` setting before every file — that indirection is the whole
    // reason the stage can be retuned mid-job. Sizing the pool to the current
    // setting instead would cap the job forever at whatever it started with.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(MAX_WORKERS)
        .build()?;
    let admission = Admission::new(runtime.clone());
    let embedders = EmbedderPool::seeded(&request.config, runtime.clone(), first_embedder);
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
    // Three stages, not two. Extraction fans out over the rayon pool; a pool of
    // embed workers turns extracted text into vectors; this thread owns the
    // (non-Sync) SQLite connection and does nothing but write. Embedding used to
    // run INLINE on this thread, which made the single writer also the single
    // embedder and capped embedding throughput at one document at a time no
    // matter how wide extraction ran.
    //
    // Streaming rather than collecting still keeps committed work on disk as the
    // run proceeds and bounds peak memory to the channel depths instead of the
    // whole corpus.
    //
    // Capacity is fixed at construction and can never be resized, so it is sized
    // for the CEILING rather than for today's setting: a channel cut to
    // `config.workers` would throttle the job the moment `extract` was raised.
    //
    // One slot per possible extract worker, not the old FOUR per worker. Each
    // slot can hold a document up to `max_chars`, so the old multiplier applied
    // to the ceiling would have put a ~512 MB floor under peak memory on top of
    // the embedder pool — the same OOM footgun the pool defaults guard against.
    // The multiplier only ever bought jitter smoothing; it never bounded
    // extract concurrency, because a full channel means the writer is behind,
    // and making producers wait for it is backpressure working as designed.
    let (extracted_sender, extracted_receiver) = mpsc::sync_channel::<ProcessedFile>(MAX_WORKERS);
    let (embedded_sender, embedded_receiver) =
        mpsc::sync_channel::<Result<ProcessedFile>>(EMBED_RANGE.1 * 2);
    // One predicate, shared by every embed worker and by the writer loop. The
    // embed stage needs the same cancellation awareness the extract stage has
    // had all along: without it a worker only learns the job is over when its
    // send fails, which is AFTER it has checked out an embedder and embedded a
    // whole document. `thread::scope` cannot return until all of them join, so
    // that is up to `EMBED_RANGE.1` further inferences — serialized through a
    // single pooled instance if `embed` has been turned down to 1 — before
    // `store.finish()` runs and the job leaves `cancelling`.
    let cancellation_flag = request.cancellation.clone();
    let job_cancelled = move || {
        cancellation_flag
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
    };
    let job_cancelled: &(dyn Fn() -> bool + Sync) = &job_cancelled;
    let outcome = std::thread::scope(|scope| -> Result<bool> {
        scope.spawn(move || {
            let stopped = AtomicBool::new(false);
            let halted = || {
                stopped.load(Ordering::Relaxed)
                    || cancellation
                        .as_ref()
                        .is_some_and(|flag| flag.load(Ordering::Relaxed))
            };
            pool.install(|| {
                records.par_iter().for_each(|record| {
                    if halted() {
                        return;
                    }
                    // Admission, not pool size, is what bounds concurrency. A
                    // thread that cannot get a slot parks here until one frees,
                    // the target is raised, or the job stops — so lowering the
                    // setting mid-job genuinely quiesces the surplus workers
                    // rather than merely applying to the next job.
                    let file = {
                        let Some(_slot) = admission.acquire(&halted) else {
                            return;
                        };
                        process(
                            record.clone(),
                            &config,
                            &normalizer,
                            &ocr,
                            &transcriber,
                            &vision,
                        )
                        // `_slot` is released HERE, before the send below: a
                        // thread blocked on backpressure is not doing work and
                        // must not hold a slot the gate could give to someone
                        // who would.
                    };
                    let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    if let Some(progress) = &progress {
                        progress(count, total);
                    }
                    // The embed stage hangs up (error or cancellation) by
                    // dropping the receiver, which also unblocks a full channel;
                    // the remaining records then wind down without extracting.
                    if extracted_sender.send(file).is_err() {
                        stopped.store(true, Ordering::Relaxed);
                    }
                })
            })
        });
        // One receiver shared by every embed worker. `recv` blocks with the lock
        // held, so exactly one worker waits for the next file at a time — but it
        // releases the lock the instant it has one, so the others are free to go
        // embed. Embedding dominates the loop by orders of magnitude; the
        // handoff does not contend.
        let extracted_receiver = Arc::new(Mutex::new(extracted_receiver));
        for _ in 0..EMBED_RANGE.1 {
            let receiver = extracted_receiver.clone();
            let sender = embedded_sender.clone();
            let embedders = embedders.clone();
            scope.spawn(move || embed_worker(&receiver, &sender, &embedders, job_cancelled));
        }
        // Both originals must go before the writer loop, or the loop never ends
        // and — worse — the extract side never learns it has been hung up on.
        // Once these are dropped the workers hold the only handles, so when they
        // exit the extract channel closes and the producer winds down. Without
        // this the error path below deadlocks: the writer leaves, the workers
        // fail their sends and exit, and the producer blocks forever on a full
        // channel whose receiver is still alive in this frame.
        drop(embedded_sender);
        drop(extracted_receiver);
        for message in embedded_receiver {
            if job_cancelled() {
                return Ok(true);
            }
            let file = message?;
            stats.files += 1;
            stats.bytes += file.rec.size;
            stats.ocr_files += usize::from(file.ocr_used);
            stats.errors += usize::from(file.method.starts_with("error:"));
            stats.incomplete += usize::from(incomplete_method(&file.method));
            stats.embedded_chunks += file.chunks.len();
            stats.vision_files += usize::from(file.vision.is_some());
            store.add(&file, now())?;
        }
        // Not a bare `Ok(false)`. Now that the embed workers drain on
        // cancellation, a cancel can close this channel with nothing left in it,
        // and the loop above would then never observe the flag — reporting a
        // cancelled run as a clean success to the CLI.
        Ok(job_cancelled())
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

/// Pull extracted files, embed them, hand them to the writer.
///
/// Runs on its own thread; the concurrency that matters is not how many of these
/// exist but how many [`Embedder`] instances [`EmbedderPool`] is currently
/// willing to lend out, which is the live `embed` setting. A worker that cannot
/// get one parks in `checkout`, so shrinking the pool quiesces workers without
/// stopping or restarting anything.
///
/// Embedding errors are forwarded rather than swallowed so the writer can fail
/// the run exactly as it did when embedding happened inline on its own thread.
///
/// `cancelled` is the same flag the extract stage's admission gate takes, and it
/// is checked at both points where this worker would otherwise commit to a long,
/// pointless piece of work: after a file arrives, and (inside `checkout`) while
/// parked waiting for an instance. A failed send is NOT sufficient — by the time
/// it fails, a whole `max_chars` document has already been through the model.
fn embed_worker(
    receiver: &Mutex<mpsc::Receiver<ProcessedFile>>,
    sender: &mpsc::SyncSender<Result<ProcessedFile>>,
    embedders: &EmbedderPool,
    cancelled: &(dyn Fn() -> bool + Sync),
) {
    loop {
        let next = {
            let guard = receiver
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.recv()
        };
        let Ok(mut file) = next else {
            return;
        };
        // Before the embed, not after it. A file that arrived just as the job was
        // cancelled is dropped here; the writer is returning `Ok(true)` and will
        // not consume it, and embedding it first only delays the join that
        // `run_index` — and therefore `store.finish()` and the job queue behind
        // it — is waiting on.
        if cancelled() {
            return;
        }
        // Same predicate the inline path used: name-only, error, partial and
        // excluded files carry no embeddable content.
        let message = if incomplete_method(&file.method) || file.method.starts_with("excluded:") {
            Ok(file)
        } else {
            match embedders.checkout(cancelled) {
                // Cancelled while parked for an instance.
                Ok(None) => return,
                Ok(Some(mut embedder)) => {
                    let embedded = embedder.embed_document(&file.content);
                    // Explicitly BEFORE the send below: a worker blocked on
                    // backpressure must not hold a pooled instance, or shrinking
                    // the pool cannot reclaim it and other workers park behind a
                    // model nobody is using.
                    drop(embedder);
                    match embedded {
                        Ok(chunks) => {
                            file.chunks = chunks;
                            Ok(file)
                        }
                        Err(error) => Err(error),
                    }
                }
                Err(error) => Err(error),
            }
        };
        if sender.send(message).is_err() {
            return;
        }
    }
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
    use super::*;

    /// An ordinary extracted text file — `method` is complete, so it takes the
    /// embed path rather than the pass-through one.
    fn embeddable(path: &str) -> ProcessedFile {
        ProcessedFile {
            rec: FileRec {
                path: path.into(),
                name: path.into(),
                ext: ".txt".into(),
                dir: ".".into(),
                drive: String::new(),
                size: 12,
                mtime: 0.0,
            },
            content: "compliance report".into(),
            tokens: Vec::new(),
            lang: "en".into(),
            method: "text".into(),
            ocr_used: false,
            pages: 0,
            sha1: None,
            chunks: Vec::new(),
            vision: None,
        }
    }

    #[test]
    fn an_embed_worker_drains_instead_of_embedding_when_the_job_is_cancelled() {
        // The defect this pins: an embed worker used to learn a job was over ONLY
        // from a failed send, i.e. after it had already checked out an embedder
        // and pushed a whole document through the model. `run_index` cannot
        // return until every one of these threads joins, so a cancel cost up to
        // one full inference per worker — serialized through a single instance
        // when `embed` is turned down to 1, which the feature explicitly invites.
        // Meanwhile `store.finish()` is deferred, the job row sits in
        // `cancelling`, and the service's one-at-a-time worker loop blocks the
        // whole queue.
        //
        // The pool here is SATURATED, so a checkout can only be satisfied by
        // cancellation — the exact state a worker is in when it is starved
        // mid-cancel.
        let runtime = Arc::new(RuntimeKnobs::from_config(&Config::default()));
        let pool = EmbedderPool::saturated(&Config::default(), runtime);
        let (extracted_sender, extracted_receiver) = mpsc::sync_channel::<ProcessedFile>(1);
        let (embedded_sender, embedded_receiver) = mpsc::sync_channel::<Result<ProcessedFile>>(4);
        extracted_sender.send(embeddable("a.txt")).expect("queued");
        drop(extracted_sender);

        let cancel = Arc::new(AtomicBool::new(true));
        let receiver = Arc::new(Mutex::new(extracted_receiver));
        let (done_sender, done_receiver) = mpsc::channel();
        // DETACHED, not scoped: against the unfixed worker this thread parks in
        // `checkout` forever, and a scoped join would hang the run instead of
        // failing it.
        std::thread::spawn(move || {
            let cancelled = move || cancel.load(Ordering::Relaxed);
            embed_worker(&receiver, &embedded_sender, &pool, &cancelled);
            let _ = done_sender.send(());
        });

        done_receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("a cancelled embed worker must return promptly, not embed one more document");
        // And it must not have produced work the writer is no longer consuming.
        assert!(
            embedded_receiver.try_recv().is_err(),
            "a cancelled worker must not embed and forward the file it had in hand"
        );
    }

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
