use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use llm_indexing::config::Config;
use llm_indexing::model::IndexStats;
use llm_indexing::normalize::Normalizer;
use llm_indexing::pipeline::{run_index, IndexRequest, Progress};
use llm_indexing::store::{connect, search, top_folders};

/// Every test here loads the embedding model, and fastembed's HuggingFace cache
/// takes a per-blob file lock that fails outright instead of waiting when two
/// processes populate it at once — a cold cache would otherwise fail whichever
/// tests lost the race. A poisoned guard is recovered because the test holding
/// it has already failed on its own terms.
static MODEL: Mutex<()> = Mutex::new(());

fn model_lock() -> MutexGuard<'static, ()> {
    MODEL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn indexes_and_searches_english_and_vietnamese() {
    let _serialized = model_lock();
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    fs::create_dir_all(input.join("sub")).unwrap();
    fs::write(
        input.join("report_en.txt"),
        "Anti money laundering compliance report. Suspicious activity detected.",
    )
    .unwrap();
    fs::write(
        input.join("bao_cao_vi.txt"),
        "Báo cáo giao dịch đáng ngờ tại ngân hàng. Khách hàng rủi ro cao.",
    )
    .unwrap();
    fs::write(
        input.join("sub/notes.md"),
        "KYC and CDD notes for bank review.",
    )
    .unwrap();

    let mut config = Config::default();
    config.ocr = "off".into();
    config.sidecar = "none".into();
    config.workers = 2;
    config.data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
    let stats = run_index(IndexRequest {
        paths: std::slice::from_ref(&input),
        out: &output,
        config: config.clone(),
        resume: false,
        overwrite: false,
        artifacts: true,
        retry_errors: false,
        include_paths: None,
        cancellation: None,
        runtime: None,
        progress: None,
    })
    .unwrap();
    assert_eq!(stats.files, 3);

    let connection = connect(&output).unwrap();

    // Migration harness: a freshly-written corpus lands on the version this
    // binary writes, not the SQLite default of 0.
    assert_eq!(
        llm_indexing::store::schema_version(&connection).unwrap(),
        llm_indexing::store::CURRENT_SCHEMA_VERSION
    );
    // `GET /corpus/status`'s `pending_files` is derived from exactly this meta
    // key — the pipeline's own last-discovery snapshot — rather than a
    // filesystem walk; this is what proves the real pipeline actually writes
    // it (the service-level tests only ever fabricate it via a fixture).
    let discovered = llm_indexing::store::read_meta(&connection, "last_discovered_files")
        .unwrap()
        .and_then(|value| value.parse::<i64>().ok());
    assert_eq!(
        discovered,
        Some(3),
        "all 3 discovered files, not just the 3 processed"
    );

    let normalizer = Normalizer::load(&config);
    assert!(!search(&connection, &normalizer, "launder", 5, false)
        .unwrap()
        .is_empty());
    assert!(!search(&connection, &normalizer, "ngan hang", 5, false)
        .unwrap()
        .is_empty());
    assert!(
        !search(&connection, &normalizer, "know your customer", 5, false)
            .unwrap()
            .is_empty()
    );
    assert!(!top_folders(&connection, &normalizer, "bank", 5)
        .unwrap()
        .is_empty());
}

// ── Durability: the corpus is written in place, so interrupted work is kept ──
//
// These index straight into a `corpus.sqlite` destination, the shape service
// jobs use, and assert that a run which does not finish still leaves usable
// rows behind and that `resume` continues from exactly those rows.

fn durability_config() -> Config {
    let mut config = Config::default();
    config.ocr = "off".into();
    config.sidecar = "none".into();
    config.workers = 2;
    config.data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data");
    config
}

fn sample_tree(input: &Path, count: usize) -> Vec<String> {
    fs::create_dir_all(input).unwrap();
    for index in 0..count {
        fs::write(
            input.join(format!("report_{index}.txt")),
            format!("Suspicious activity report number {index} for the compliance team."),
        )
        .unwrap();
    }
    let input = input.canonicalize().unwrap();
    (0..count)
        .map(|index| {
            input
                .join(format!("report_{index}.txt"))
                .to_string_lossy()
                .to_string()
        })
        .collect()
}

fn index(
    input: &Path,
    destination: &Path,
    resume: bool,
    include_paths: Option<HashSet<String>>,
    cancellation: Option<Arc<AtomicBool>>,
    progress: Option<Arc<dyn Fn(Progress) + Send + Sync>>,
) -> anyhow::Result<IndexStats> {
    let _serialized = model_lock();
    run_index(IndexRequest {
        paths: std::slice::from_ref(&input.to_path_buf()),
        out: destination,
        config: durability_config(),
        resume,
        overwrite: false,
        artifacts: false,
        retry_errors: false,
        include_paths,
        cancellation,
        runtime: None,
        progress,
    })
}

fn indexed_files(destination: &Path) -> i64 {
    connect(destination)
        .unwrap()
        .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))
        .unwrap()
}

#[test]
fn resume_continues_from_a_partially_written_corpus() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let destination = temp.path().join("corpus.sqlite");
    let paths = sample_tree(&input, 4);

    // Stand in for a run that died after two files: the corpus holds those two
    // and nothing else, exactly as a killed process would leave it.
    let partial = index(
        &input,
        &destination,
        false,
        Some(paths[..2].iter().cloned().collect()),
        None,
        None,
    )
    .unwrap();
    assert_eq!(partial.files, 2);
    assert!(destination.is_file(), "the corpus is the published file");
    assert_eq!(indexed_files(&destination), 2);

    // Resume must pick the remaining two up rather than restart the tree.
    let finished = index(&input, &destination, true, None, None, None).unwrap();
    assert_eq!(finished.skipped, 2, "committed files are not redone");
    assert_eq!(finished.files, 2);
    assert_eq!(indexed_files(&destination), 4);

    let connection = connect(&destination).unwrap();
    let normalizer = Normalizer::load(&durability_config());
    assert_eq!(
        search(&connection, &normalizer, "suspicious", 10, false)
            .unwrap()
            .len(),
        4
    );
}

/// A sub-path resume must prune only its OWN stale rows. Rows outside the
/// resumed root belong to the rest of a whole-drive corpus — the sub-path
/// job's walk never saw them, so their absence from it is not evidence of
/// deletion, and pruning them (the old behavior) let a targeted re-index of
/// one folder silently destroy every other folder's rows.
#[test]
fn a_sub_path_resume_prunes_only_under_its_own_root() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let docs = input.join("docs");
    let photos = input.join("photos");
    fs::create_dir_all(&docs).unwrap();
    fs::create_dir_all(&photos).unwrap();
    fs::write(docs.join("keep.txt"), "Docs report that stays on disk.").unwrap();
    fs::write(
        docs.join("gone.txt"),
        "Docs report deleted before the resume.",
    )
    .unwrap();
    fs::write(
        photos.join("outside.txt"),
        "Photos report outside the resumed root.",
    )
    .unwrap();
    let destination = temp.path().join("corpus.sqlite");

    // Whole-tree first: all three rows land in the per-drive corpus.
    let whole = index(&input, &destination, false, None, None, None).unwrap();
    assert_eq!(whole.files, 3);
    assert_eq!(indexed_files(&destination), 3);

    // One docs file vanishes; resume ONLY the docs subtree.
    fs::remove_file(docs.join("gone.txt")).unwrap();
    let scoped = index(&docs, &destination, true, None, None, None).unwrap();
    assert_eq!(scoped.removed, 1, "the vanished in-root file is pruned");
    assert_eq!(scoped.skipped, 1, "the unchanged in-root file is reused");
    assert_eq!(scoped.files, 0);

    let connection = connect(&destination).unwrap();
    let mut remaining = connection
        .prepare("SELECT path FROM files ORDER BY path")
        .unwrap()
        .query_map([], |row| row.get::<_, String>(0))
        .unwrap()
        .flatten()
        .collect::<Vec<_>>();
    remaining.sort();
    assert_eq!(remaining.len(), 2, "photos row must survive: {remaining:?}");
    assert!(
        remaining.iter().any(|p| p.ends_with("keep.txt")),
        "in-root unchanged row kept: {remaining:?}"
    );
    assert!(
        remaining.iter().any(|p| p.ends_with("outside.txt")),
        "out-of-root row NOT this job's to delete: {remaining:?}"
    );
    assert!(
        !remaining.iter().any(|p| p.ends_with("gone.txt")),
        "vanished in-root row pruned: {remaining:?}"
    );
}

/// The corpus self-describes (meta table) and the embedding-model identity
/// gates resume: an unchanged model skips unchanged files as before, but a
/// recorded model different from the current one forces a full re-embed —
/// previously a model upgrade silently left a mixed-vector corpus.
#[test]
fn a_changed_embedding_model_reprocesses_the_whole_corpus() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let destination = temp.path().join("corpus.sqlite");
    sample_tree(&input, 2);

    let first = index(&input, &destination, false, None, None, None).unwrap();
    assert_eq!(first.files, 2);

    let connection = connect(&destination).unwrap();
    let meta = |key: &str| {
        connection
            .query_row("SELECT value FROM meta WHERE key=?1", [key], |row| {
                row.get::<_, String>(0)
            })
            .ok()
    };
    assert_eq!(
        meta("embed_model").as_deref(),
        Some("intfloat/multilingual-e5-small"),
        "the corpus records which embedding model produced it"
    );
    let started = meta("last_job_started_at").expect("started_at stamped");
    let finished = meta("last_job_finished_at").expect("finished_at stamped on completion");
    assert!(
        finished.parse::<f64>().unwrap() >= started.parse::<f64>().unwrap(),
        "finished_at >= started_at on a completed run"
    );

    // Same model: resume skips everything.
    let same = index(&input, &destination, true, None, None, None).unwrap();
    assert_eq!(same.skipped, 2);
    assert_eq!(same.files, 0);

    // Recorded model differs from the loaded one: everything is re-embedded.
    connection
        .execute(
            "UPDATE meta SET value='some/older-model' WHERE key='embed_model'",
            [],
        )
        .unwrap();
    drop(connection);
    let upgraded = index(&input, &destination, true, None, None, None).unwrap();
    assert_eq!(upgraded.files, 2, "a model change must re-embed every file");
    assert_eq!(upgraded.skipped, 0);
    let connection = connect(&destination).unwrap();
    let recorded: String = connection
        .query_row(
            "SELECT value FROM meta WHERE key='embed_model'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        recorded, "intfloat/multilingual-e5-small",
        "the current model is re-recorded after the re-embed"
    );
}

#[test]
fn cancellation_keeps_committed_work_and_resume_finishes_it() {
    const FILES: i64 = 240;
    // Cancel late enough that the writer must already hold files.
    //
    // `progress` counts EXTRACTED files, so this bound is arithmetic on how far
    // extraction can run ahead of the writer. The pipeline buffers at most
    // MAX_WORKERS (64, extract→embed channel) + MAX_WORKERS (64, extract threads
    // blocked in `send`, already counted as processed) + EMBED_RANGE.1 (8, one
    // per embed worker) + EMBED_RANGE.1 × 2 (16, embed→writer channel) = 152
    // files it has not yet written. Cancelling at 200 therefore guarantees ~48
    // reached the writer, while leaving 40 unstarted so the run is genuinely
    // interrupted.
    //
    // These numbers grew with the pipeline: the channel is now sized for the
    // MAX_WORKERS ceiling rather than for `config.workers`, because `extract` is
    // retunable mid-job and a capacity cut to the starting value would throttle
    // a job that was later widened. A 24-file corpus no longer outruns that
    // buffer at all — every file would sit in the channel, the writer would see
    // the flag before storing anything, and the assertion below would be
    // vacuous rather than wrong.
    const CANCEL_AFTER: usize = 200;

    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let destination = temp.path().join("corpus.sqlite");
    sample_tree(&input, FILES as usize);

    let cancellation = Arc::new(AtomicBool::new(false));
    let flag = cancellation.clone();
    let error = index(
        &input,
        &destination,
        false,
        None,
        Some(cancellation),
        Some(Arc::new(move |update: Progress| {
            if update.processed >= CANCEL_AFTER {
                flag.store(true, Ordering::Relaxed);
            }
        })),
    )
    .expect_err("a cancelled run reports cancellation");
    assert!(format!("{error:#}").contains("cancelled"), "{error:#}");

    // The old contract deleted the whole build here. Now the work that reached
    // the writer is committed and stays.
    assert!(destination.is_file(), "the partial corpus survives");
    let retained = indexed_files(&destination);
    assert!(retained > 0, "committed work must survive cancellation");
    assert!(
        retained < FILES,
        "the run really was interrupted: {retained}"
    );

    // Resume skips precisely what was kept and finishes the rest.
    let finished = index(&input, &destination, true, None, None, None).unwrap();
    assert_eq!(finished.skipped as i64, retained);
    assert_eq!(finished.files as i64, FILES - retained);
    assert_eq!(indexed_files(&destination), FILES);
}

fn indexed_chunks(destination: &Path) -> i64 {
    connect(destination)
        .unwrap()
        .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
        .unwrap()
}

/// Embedding moved OFF the single writer thread into a pool of `Embedder`
/// instances that run concurrently. That is a correctness risk, not just a
/// performance change: files now cross a second channel and are embedded out of
/// order by whichever worker got a model, so a mistake there loses chunks,
/// duplicates them, or attaches them to the wrong file.
///
/// Pinning it as an INVARIANT across pool sizes is what makes this meaningful —
/// one pool size proves nothing, because a bug that drops work would drop it
/// consistently. Widening the pool must change only how fast the work happens.
#[test]
fn pooled_embedding_is_invariant_to_the_embed_pool_size() {
    use llm_indexing::runtime::RuntimeKnobs;
    use serde_json::{json, Map, Value};

    let _serialized = model_lock();
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    // Enough files to keep several embedders genuinely busy at once.
    sample_tree(&input, 24);

    let mut observed = Vec::new();
    for embed in [1_u64, 3] {
        let destination = temp.path().join(format!("corpus_{embed}.sqlite"));
        let config = durability_config();
        let runtime = Arc::new(RuntimeKnobs::from_config(&config));
        let body: Map<String, Value> = json!({"embed": embed}).as_object().expect("object").clone();
        runtime.apply(&body).expect("embed is a valid stage");

        let stats = run_index(IndexRequest {
            paths: std::slice::from_ref(&input),
            out: &destination,
            config,
            resume: false,
            overwrite: false,
            artifacts: false,
            retry_errors: false,
            include_paths: None,
            cancellation: None,
            runtime: Some(runtime),
            progress: None,
        })
        .unwrap();

        assert_eq!(stats.files, 24, "embed={embed}");
        assert!(
            stats.embedded_chunks > 0,
            "embed={embed}: the run must actually embed something, or the \
             equality below would hold vacuously at zero"
        );
        // Every file is one short sentence, so each contributes exactly one chunk.
        assert_eq!(stats.embedded_chunks, 24, "embed={embed}");
        assert_eq!(
            indexed_chunks(&destination),
            stats.embedded_chunks as i64,
            "embed={embed}: stored chunks must match reported chunks"
        );
        observed.push(stats.embedded_chunks);
    }
    assert_eq!(
        observed[0], observed[1],
        "pool size must not change what gets embedded, only how fast"
    );
}

/// Write a minimal but VALID `.docx` (a zip carrying `word/document.xml`) whose
/// `<w:t>` run holds `text`, so extraction yields a complete, embeddable row.
fn write_docx(path: &Path, text: &str) {
    use std::io::Write;
    use zip::write::SimpleFileOptions;
    use zip::CompressionMethod;

    let file = fs::File::create(path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    // `Stored` needs no compression feature and keeps the fixture trivial.
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file("word/document.xml", options).unwrap();
    let xml = format!(
        "<?xml version=\"1.0\"?><w:document xmlns:w=\"urn:x\"><w:body><w:p><w:r>\
         <w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
    );
    zip.write_all(xml.as_bytes()).unwrap();
    zip.finish().unwrap();
}

/// The single corpus row's `(method, size)` — every test here indexes exactly one
/// file, so no path predicate is needed (paths are stored canonicalized).
fn only_file_row(destination: &Path) -> (String, i64) {
    connect(destination)
        .unwrap()
        .query_row("SELECT method, size FROM files", [], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })
        .unwrap()
}

/// keep-on-failure protects a complete row ONLY when the file is unchanged. This
/// is the companion guard: when the file has CHANGED and the reprocess fails, the
/// error row must still REPLACE the old complete row (the ordinary contract the
/// feature must not weaken). Uses a `.docx` because it extracts real text when a
/// valid zip (complete row) yet errors outright once the bytes are no longer a
/// zip — a deterministic, tool-free way to turn a good row into a failing one.
#[test]
fn a_changed_file_whose_reprocess_fails_still_replaces_the_old_row() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    fs::create_dir_all(&input).unwrap();
    let destination = temp.path().join("corpus.sqlite");
    let doc = input.join("report.docx");

    // A valid docx -> a COMPLETE row with chunks (exactly the row keep-on-failure
    // would preserve if the file were UNCHANGED).
    write_docx(
        &doc,
        "Suspicious activity compliance report for the bank review team.",
    );
    let first = index(&input, &destination, false, None, None, None).unwrap();
    assert_eq!(first.files, 1);
    assert_eq!(first.errors, 0, "the valid docx extracts cleanly");
    let (method, _) = only_file_row(&destination);
    assert!(
        !method.starts_with("error:"),
        "stored a complete row: {method}"
    );

    // The bytes CHANGE to non-zip garbage: reprocessing now fails. Because the
    // file changed, the error must replace the old row rather than be kept.
    fs::write(
        &doc,
        b"not a zip archive at all -- just some plain garbage bytes",
    )
    .unwrap();
    let resumed = index(&input, &destination, true, None, None, None).unwrap();
    assert_eq!(resumed.files, 1, "the changed file is reprocessed");
    assert_eq!(resumed.errors, 1, "the reprocess fails");

    let (method, size) = only_file_row(&destination);
    assert!(
        method.starts_with("error:"),
        "a changed file's error must REPLACE the old complete row, got {method}"
    );
    let on_disk = fs::metadata(&doc).unwrap().len() as i64;
    assert_eq!(
        size, on_disk,
        "the replaced row carries the changed file's size, proving it was rewritten"
    );
}

// ── P0-8: page anchoring from PDF extraction into stored chunks ─────────────

/// Word-wrapped filler lines of (at least) `total_chars` of `word`, so a page
/// built from them is long enough to force more than one embedding chunk
/// (`CHUNK_CHARS` is 1,200) without relying on any single `Tj` string wider
/// than a page — poppler's text extraction does not reliably recover text
/// positioned off the visible `MediaBox`, which a single unwrapped multi-KB
/// line easily runs into at any normal font size.
fn padded_lines(word: &str, total_chars: usize) -> Vec<String> {
    let mut words = Vec::new();
    let mut length = 0;
    while length < total_chars {
        words.push(word);
        length += word.len() + 1;
    }
    words.chunks(10).map(|line| line.join(" ")).collect()
}

/// Write a minimal but VALID multi-page PDF by hand — one Type1/Helvetica
/// content stream per page, a correct object-offset xref table, no external
/// tool involved — so `pdftotext`'s page breaks (and therefore
/// `extract::page_segments_from_form_feeds`) are exercised against a REAL
/// multi-page document rather than simulated. Mirrors `write_docx`'s
/// handcraft-a-minimal-valid-file approach one file up.
fn write_pdf(path: &Path, pages: &[Vec<String>]) {
    let mut objects: Vec<String> = Vec::new();
    objects.push("<< /Type /Catalog /Pages 2 0 R >>".to_string()); // 1
    let kids = (0..pages.len())
        .map(|i| format!("{} 0 R", 3 + 2 * i))
        .collect::<Vec<_>>()
        .join(" ");
    objects.push(format!(
        "<< /Type /Pages /Kids [{kids}] /Count {} >>",
        pages.len()
    )); // 2
    let font_obj = 3 + 2 * pages.len();
    for (index, lines) in pages.iter().enumerate() {
        let content_obj = 4 + 2 * index;
        objects.push(format!(
            "<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 {font_obj} 0 R >> >> \
             /MediaBox [0 0 612 792] /Contents {content_obj} 0 R >>"
        ));
        let mut parts = vec!["BT /F1 10 Tf 72 720 Td".to_string()];
        for (line_index, line) in lines.iter().enumerate() {
            if line_index == 0 {
                parts.push(format!("({line}) Tj"));
            } else {
                parts.push(format!("0 -14 Td ({line}) Tj"));
            }
        }
        parts.push("ET".to_string());
        let stream = parts.join("\n");
        objects.push(format!(
            "<< /Length {} >>\nstream\n{stream}\nendstream",
            stream.len()
        ));
    }
    objects.push("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string());

    let header = "%PDF-1.4\n";
    let mut body = String::new();
    let mut offsets = Vec::with_capacity(objects.len());
    for (index, object) in objects.iter().enumerate() {
        offsets.push(header.len() + body.len());
        body.push_str(&format!("{} 0 obj\n{object}\nendobj\n", index + 1));
    }
    let xref_offset = header.len() + body.len();
    let mut xref = format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1);
    for offset in &offsets {
        xref.push_str(&format!("{offset:010} 00000 n \n"));
    }
    let trailer = format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF",
        objects.len() + 1
    );
    fs::write(path, format!("{header}{body}{xref}{trailer}")).unwrap();
}

/// End-to-end: a real multi-page PDF, indexed through the real pipeline
/// (extraction -> chunking -> the corpus database), must leave every stored
/// chunk with a page range that is present, in bounds, and never regresses
/// across `chunk_index` — the whole point of P0-8's `page_start`/`page_end`.
/// Padded long enough (2,000+ chars/page, `CHUNK_CHARS` is 1,200) that the
/// three pages produce more than one chunk, including at least one that
/// straddles a page boundary — the exact case a per-file-only test cannot
/// exercise, because it depends on `extract::pdf` actually handing the
/// chunker real, page-numbered segments rather than a fabricated slice.
#[test]
fn page_boundaries_survive_from_pdf_extraction_into_stored_chunks() {
    // NOT `model_lock()` here too: `index()` below takes it itself (see its
    // doc comment), and the plain `std::sync::Mutex` it wraps is not
    // reentrant — a second `.lock()` from the same thread that already holds
    // it deadlocks rather than blocking briefly.
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    fs::create_dir_all(&input).unwrap();
    let destination = temp.path().join("corpus.sqlite");

    let pages = vec![
        padded_lines("alpha", 2000),
        padded_lines("bravo", 2000),
        padded_lines("charlie", 2000),
    ];
    write_pdf(&input.join("report.pdf"), &pages);

    let stats = index(&input, &destination, false, None, None, None).unwrap();
    assert_eq!(stats.files, 1);
    assert_eq!(stats.errors, 0, "the handcrafted PDF must extract cleanly");

    let connection = connect(&destination).unwrap();
    let pages_recorded: i64 = connection
        .query_row("SELECT pages FROM files", [], |row| row.get(0))
        .unwrap();
    assert_eq!(pages_recorded, 3, "pdfinfo's page count is unaffected");

    let mut statement = connection
        .prepare("SELECT chunk_index, page_start, page_end FROM chunks ORDER BY chunk_index")
        .unwrap();
    let rows: Vec<(i64, Option<i64>, Option<i64>)> = statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .unwrap()
        .flatten()
        .collect();

    assert!(
        rows.len() > 1,
        "the padded pages must produce more than one chunk: {rows:?}"
    );
    assert!(
        rows.iter()
            .all(|(_, start, end)| start.is_some() && end.is_some()),
        "every chunk of a page-segmented PDF must carry a page range: {rows:?}"
    );
    assert_eq!(
        rows.first().unwrap().1,
        Some(1),
        "the first chunk starts on page 1: {rows:?}"
    );
    assert_eq!(
        rows.last().unwrap().2,
        Some(3),
        "the last chunk ends on page 3: {rows:?}"
    );
    assert!(
        rows.iter().any(|(_, start, end)| start != end),
        "at least one chunk must straddle a page boundary given how long each page is: {rows:?}"
    );
    // Page ranges never point past the 3 real pages, and — since a chunk's
    // character window only ever starts and ends later than the previous
    // chunk's (the overlap subtracts from `end`, never producing an earlier
    // `start`) — `page_start` and `page_end` are each non-decreasing across
    // `chunk_index`. They are NOT compared cross-field (`start` against the
    // previous chunk's `end`): overlapping windows can revisit a tail of an
    // earlier page a later-starting chunk has already moved past in `page_end`,
    // so `page_start` alone can sit behind the previous chunk's `page_end`
    // without that being a regression.
    let (mut previous_start, mut previous_end) = (1, 1);
    for (chunk_index, start, end) in &rows {
        let (start, end) = (start.unwrap(), end.unwrap());
        assert!(
            (1..=3).contains(&start) && start <= end && end <= 3,
            "chunk {chunk_index}: page range out of bounds: {rows:?}"
        );
        assert!(
            start >= previous_start && end >= previous_end,
            "chunk {chunk_index}: page range regressed: {rows:?}"
        );
        previous_start = start;
        previous_end = end;
    }
}

/// The interaction keep-on-failure must NOT break: a pure EMBED-MODEL upgrade
/// re-embeds the whole corpus, and when an unchanged file's reprocess fails,
/// keep-on-failure preserves its old (old-model) row. That file is therefore
/// still un-migrated, so the corpus `embed_model` marker must NOT advance — or
/// `embed_model_changed` would read false forever after and no resume would ever
/// revisit the stranded file. The upgrade gate stays OPEN until the migration
/// truly lands. (Before the fix the marker flipped up front, closing the gate and
/// stranding the file with no signal.)
///
/// The file is made to FAIL its reprocess while staying byte-for-byte "unchanged"
/// to the retain predicate — garbage bytes of the original length, with the
/// original mtime restored — the exact shape keep-on-failure preserves.
#[test]
fn a_model_change_that_keeps_an_old_row_leaves_the_upgrade_gate_open() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    fs::create_dir_all(&input).unwrap();
    let destination = temp.path().join("corpus.sqlite");
    let doc = input.join("report.docx");
    let text = "Suspicious activity compliance report for the bank review team.";

    // A valid docx -> a COMPLETE row with chunks, recorded under the current model.
    write_docx(&doc, text);
    let first = index(&input, &destination, false, None, None, None).unwrap();
    assert_eq!(first.files, 1);
    assert_eq!(first.errors, 0, "the valid docx extracts cleanly");
    let (good_method, good_size) = only_file_row(&destination);
    assert!(
        !good_method.starts_with("error:"),
        "stored a complete row: {good_method}"
    );

    let marker = |dest: &Path| -> String {
        connect(dest)
            .unwrap()
            .query_row(
                "SELECT value FROM meta WHERE key='embed_model'",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap()
    };
    let current = marker(&destination);

    // Corrupt the bytes so a reprocess FAILS, but keep the file "unchanged" as the
    // retain predicate sees it: same length, and the ORIGINAL mtime restored.
    // keep-on-failure must therefore preserve the old row on the upgrade run.
    let original_mtime = fs::metadata(&doc).unwrap().modified().unwrap();
    fs::write(&doc, vec![b'x'; good_size as usize]).unwrap();
    fs::File::options()
        .write(true)
        .open(&doc)
        .unwrap()
        .set_modified(original_mtime)
        .unwrap();

    // Force a pure embed-model upgrade: the recorded model differs from the loaded
    // one, so resume bypasses the retain predicate and re-embeds the whole corpus.
    {
        let connection = connect(&destination).unwrap();
        connection
            .execute(
                "UPDATE meta SET value='some/older-model' WHERE key='embed_model'",
                [],
            )
            .unwrap();
    }

    let upgraded = index(&input, &destination, true, None, None, None).unwrap();
    assert_eq!(
        upgraded.files, 1,
        "the unchanged file is reprocessed by the model upgrade"
    );
    assert_eq!(upgraded.errors, 1, "its reprocess fails");

    // keep-on-failure kept the OLD complete row rather than the error.
    let (kept_method, kept_size) = only_file_row(&destination);
    assert_eq!(
        kept_method, good_method,
        "the old complete row was kept, not replaced by the error"
    );
    assert_eq!(kept_size, good_size);

    // The crux: the marker did NOT advance, because that file is still on
    // old-model vectors. A closed gate here would strand it permanently.
    assert_eq!(
        marker(&destination),
        "some/older-model",
        "a kept old-model row must leave the upgrade gate OPEN for the next resume"
    );

    // Repair the file; the still-open gate lets the next resume finally migrate it,
    // and only once the whole corpus is on the new model does the marker advance.
    write_docx(&doc, text);
    let healed = index(&input, &destination, true, None, None, None).unwrap();
    assert_eq!(healed.errors, 0, "the repaired file re-embeds cleanly");
    let (healed_method, _) = only_file_row(&destination);
    assert!(
        !healed_method.starts_with("error:"),
        "healed to a complete row: {healed_method}"
    );
    assert_eq!(
        marker(&destination),
        current,
        "once every file is migrated the marker finally advances to the current model"
    );
}

/// The corpus's own record of how a path has fared: `(method, attempts)`.
fn only_file_attempts(destination: &Path) -> (String, i64) {
    connect(destination)
        .unwrap()
        .query_row("SELECT method, attempts FROM files", [], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })
        .unwrap()
}

/// The whole workstream, end to end: a file that cannot be read is attempted
/// three times across three resumes and then stops costing anything, and the
/// escape hatches still work. A corrupt `.docx` is used because it fails
/// deterministically with no external tool involved.
#[test]
fn an_unreadable_file_is_attempted_three_times_and_then_left_alone() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    fs::create_dir_all(&input).unwrap();
    let destination = temp.path().join("corpus.sqlite");
    let doc = input.join("broken.docx");
    fs::write(&doc, b"this was never a zip").unwrap();

    // The first run plus two resumes: each one pays the file's full cost and
    // records the failure.
    for expected in 1..=3 {
        let stats = index(&input, &destination, expected > 1, None, None, None).unwrap();
        assert_eq!(stats.errors, 1, "attempt {expected} must run and fail");
        assert_eq!(
            stats.capped, 0,
            "nothing is capped before the budget is out"
        );
        let (method, attempts) = only_file_attempts(&destination);
        assert!(method.starts_with("error:"), "{method}");
        assert_eq!(attempts, i64::from(expected), "attempt {expected} recorded");
    }

    // The fourth resume is the point of the workstream: the file is not read,
    // not OCR'd, not embedded. Before the cap this repeated forever, once per
    // resume, for ~181k of the live corpus's 263k rows.
    let converged = index(&input, &destination, true, None, None, None).unwrap();
    assert_eq!(converged.files, 0, "a capped file is not processed at all");
    assert_eq!(converged.errors, 0);
    assert_eq!(
        converged.capped, 1,
        "and it is reported as capped, not done"
    );
    assert_eq!(only_file_attempts(&destination).1, 3, "no further attempts");

    // Escape hatch one: the file changes. New bytes get a full budget, so the
    // repaired document is read even though the path had burned every attempt.
    write_docx(&doc, "Suspicious activity report for the compliance team.");
    let repaired = index(&input, &destination, true, None, None, None).unwrap();
    assert_eq!(repaired.files, 1, "changed bytes are always reprocessed");
    assert_eq!(repaired.errors, 0);
    let (method, attempts) = only_file_attempts(&destination);
    assert_eq!(method, "docx");
    assert_eq!(attempts, 0, "a finished row carries no failed attempts");
}

/// Escape hatch two: `retry_errors`. The operator has fixed the reason the rows
/// failed — a drive that was not mounted, a dependency that was not installed —
/// and asks for the capped rows back. OFF by default, which is what the test
/// above depends on.
#[test]
fn retry_errors_reopens_a_capped_row() {
    let _serialized = model_lock();
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    fs::create_dir_all(&input).unwrap();
    let destination = temp.path().join("corpus.sqlite");
    fs::write(input.join("broken.docx"), b"this was never a zip").unwrap();

    for resume in [false, true, true, true] {
        run_index(IndexRequest {
            paths: std::slice::from_ref(&input),
            out: &destination,
            config: durability_config(),
            resume,
            overwrite: false,
            artifacts: false,
            retry_errors: false,
            include_paths: None,
            cancellation: None,
            runtime: None,
            progress: None,
        })
        .unwrap();
    }
    assert_eq!(only_file_attempts(&destination).1, 3, "budget spent");

    let retried = run_index(IndexRequest {
        paths: std::slice::from_ref(&input),
        out: &destination,
        config: durability_config(),
        resume: true,
        overwrite: false,
        artifacts: false,
        retry_errors: true,
        include_paths: None,
        cancellation: None,
        runtime: None,
        progress: None,
    })
    .unwrap();
    assert_eq!(retried.files, 1, "retry_errors reopens the capped row");
    assert_eq!(retried.capped, 0);
    assert_eq!(
        only_file_attempts(&destination).1,
        4,
        "the extra attempt is counted like any other"
    );
}

/// B2: the encrypted-PDF fast path. `pdfinfo` must actually be on `PATH` for
/// any of this to exercise real detection instead of the harmless "binary
/// missing" fallback `parse_pdf_info` also has to handle — see
/// `pdf_info_classification::empty_output_from_a_missing_pdfinfo_binary_is_not_password_required`
/// in `extract.rs` for that case covered in isolation. Skips cleanly (like
/// the vision live tests) rather than failing on a box without poppler
/// installed.
mod encrypted_pdf_fast_path {
    use super::*;

    fn pdfinfo_available() -> bool {
        std::process::Command::new("pdfinfo")
            .arg("-v")
            .output()
            .is_ok()
    }

    /// Three fixtures, all built from the SAME one-page PDF with the text
    /// "Owner password only test": `pdf-plain.pdf` unencrypted,
    /// `pdf-owner-password-only.pdf` re-saved with `user_password=""` (opens
    /// under poppler's default password, permissions fully denied),
    /// `pdf-user-password-required.pdf` re-saved with a real user password.
    /// Verified against Calibre's bundled poppler 25.11.0 while writing this
    /// test: `pdftotext` reads the first two and fails identically to
    /// `pdfinfo` on the third with "Command Line Error: Incorrect password".
    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    #[test]
    fn an_encrypted_pdf_becomes_an_honest_error_row_and_counts_separately() {
        if !pdfinfo_available() {
            eprintln!("skipping encrypted-PDF live test: pdfinfo not on PATH");
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("input");
        fs::create_dir_all(&input).unwrap();
        let destination = temp.path().join("corpus.sqlite");
        fs::copy(
            fixture("pdf-user-password-required.pdf"),
            input.join("locked.pdf"),
        )
        .unwrap();

        let stats = index(&input, &destination, false, None, None, None).unwrap();
        assert_eq!(stats.files, 1);
        assert_eq!(stats.errors, 1, "a password-required PDF is an error row");
        assert_eq!(
            stats.encrypted, 1,
            "the operator-visible encrypted counter, separate from stats.errors"
        );
        let (method, _) = only_file_row(&destination);
        assert_eq!(
            method, "error:encrypted",
            "honest error:encrypted, not a silent empty pdf-text-partial"
        );
    }

    /// The must-not-block case: an owner-password-only PDF, with EVERY
    /// permission bit denied in the fixture, still has its text pulled by
    /// `pdftotext` under poppler's default empty user password. B2 must not
    /// mistake `Encrypted: yes` for a reason to bail.
    #[test]
    fn an_owner_password_only_pdf_still_extracts() {
        if !pdfinfo_available() {
            eprintln!("skipping encrypted-PDF live test: pdfinfo not on PATH");
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("input");
        fs::create_dir_all(&input).unwrap();
        let destination = temp.path().join("corpus.sqlite");
        fs::copy(
            fixture("pdf-owner-password-only.pdf"),
            input.join("restricted.pdf"),
        )
        .unwrap();

        let stats = index(&input, &destination, false, None, None, None).unwrap();
        assert_eq!(stats.files, 1);
        assert_eq!(
            stats.errors, 0,
            "owner-password-only must not be treated as blocking"
        );
        assert_eq!(stats.encrypted, 0);
        let (method, _) = only_file_row(&destination);
        assert!(!method.starts_with("error:"), "must extract, got {method}");
    }

    /// Control: an ordinary unencrypted PDF is unaffected by the fast path.
    #[test]
    fn a_normal_pdf_is_unaffected() {
        if !pdfinfo_available() {
            eprintln!("skipping encrypted-PDF live test: pdfinfo not on PATH");
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("input");
        fs::create_dir_all(&input).unwrap();
        let destination = temp.path().join("corpus.sqlite");
        fs::copy(fixture("pdf-plain.pdf"), input.join("plain.pdf")).unwrap();

        let stats = index(&input, &destination, false, None, None, None).unwrap();
        assert_eq!(stats.files, 1);
        assert_eq!(stats.errors, 0);
        assert_eq!(stats.encrypted, 0);
        let (method, _) = only_file_row(&destination);
        assert_eq!(method, "pdf-text");
    }
}

/// The other half of P0-8, and the one a live deployment actually depends on:
/// `SCHEMA_V2` adds `chunks.page_start`/`page_end` NULLABLE and backfills
/// nothing, so every chunk of an ALREADY-INDEXED corpus reads back NULL — the
/// migration migrates the schema, not the data. Without a per-file signal the
/// only way to make an existing corpus citable would be a destructive
/// `{overwrite:true}` re-index of the whole thing, which is not something a
/// deploy should have to do.
///
/// Reproduces the live shape exactly (index, then blank the anchors, which is
/// what a pre-migration corpus looks like once the columns are added) and
/// asserts an ORDINARY resume repairs it — and then, just as importantly, that
/// the next resume leaves it alone. A rule that re-schedules an unanchored file
/// it cannot anchor would redo those files on every run for the life of the
/// corpus.
#[test]
fn an_unanchored_corpus_regains_its_page_anchors_on_an_ordinary_resume() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    fs::create_dir_all(&input).unwrap();
    let destination = temp.path().join("corpus.sqlite");

    write_pdf(
        &input.join("report.pdf"),
        &[padded_lines("alpha", 2000), padded_lines("bravo", 2000)],
    );
    assert_eq!(
        index(&input, &destination, false, None, None, None)
            .unwrap()
            .files,
        1
    );
    assert!(anchored_chunks(&destination) > 0, "indexed with anchors");

    // Become a corpus indexed before page attribution existed.
    rusqlite::Connection::open(&destination)
        .unwrap()
        .execute_batch("UPDATE chunks SET page_start=NULL, page_end=NULL")
        .unwrap();
    assert_eq!(anchored_chunks(&destination), 0);

    // Nothing about the FILE changed — same size, same mtime, complete method,
    // chunks present — so every pre-existing resume rule would skip it. Only
    // the missing anchors schedule it.
    let repaired = index(&input, &destination, true, None, None, None).unwrap();
    assert_eq!(
        repaired.files, 1,
        "an unanchored PDF must be redone by a plain resume"
    );
    assert!(
        anchored_chunks(&destination) > 0,
        "and must come back with its page anchors"
    );

    // Termination: the file is anchored now, so the signal is gone and the
    // next resume skips it like any other unchanged file.
    let settled = index(&input, &destination, true, None, None, None).unwrap();
    assert_eq!(
        (settled.files, settled.skipped),
        (0, 1),
        "a repaired file must not be re-scheduled forever"
    );
}

fn anchored_chunks(destination: &Path) -> i64 {
    connect(destination)
        .unwrap()
        .query_row(
            "SELECT COUNT(*) FROM chunks WHERE page_start IS NOT NULL",
            [],
            |row| row.get(0),
        )
        .unwrap()
}

// ── sha1 backfill lane (`hash_backfill`) ─────────────────────────────────────
//
// A corpus indexed before `hash: true` was set carries `sha1 IS NULL` on every
// row, and resume never repairs that on its own: those rows are FINISHED, so the
// forward hash path — which only runs on a file the job is actually indexing —
// never sees them again, however many times the corpus is resumed. The backfill
// lane is the only route from that corpus to a hashed one, and these tests pin
// what it may and may not do on the way.
//
// The load-bearing invariant is the negative one. The lane's writer is a bare
// `UPDATE files SET sha1`, deliberately NOT the `INSERT OR REPLACE` the indexing
// path uses, because that would restate `method`, `chars`, `pages`,
// `indexed_at`, `attempts`, `last_attempt_at` and `elapsed_ms` — the exact
// columns the resume predicate and the attempt cap read — from a `ProcessedFile`
// the lane never built.

/// One stored row, every column a backfill must not disturb plus the one it
/// must set. Compared whole, so "changed nothing else" is asserted against the
/// row rather than against a list of columns someone remembered to check.
type StoredRow = (String, String, i64, i64, f64, i64, f64, i64, Option<String>);

fn file_rows(destination: &Path) -> Vec<StoredRow> {
    let connection = connect(destination).unwrap();
    let mut statement = connection
        .prepare(
            "SELECT path,method,chars,pages,indexed_at,attempts,last_attempt_at,elapsed_ms,sha1 \
             FROM files ORDER BY path",
        )
        .unwrap();
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?,
                row.get(7)?,
                row.get(8)?,
            ))
        })
        .unwrap();
    rows.map(|row| row.unwrap()).collect()
}

/// The hash the lane must produce, computed the way anything else would.
fn expected_sha1(path: &Path) -> String {
    use sha1::{Digest, Sha1};
    let mut hash = Sha1::new();
    hash.update(fs::read(path).unwrap());
    format!("{:x}", hash.finalize())
}

/// `durability_config` plus the two gates, so a test can arm exactly one of
/// them and prove the other still holds the lane shut.
fn backfill_config(hash: bool, hash_backfill: bool) -> Config {
    let mut config = durability_config();
    config.hash = hash;
    config.hash_backfill = hash_backfill;
    config
}

/// A shared `(processed, worked, total)` log for the progress assertions.
type ProgressSink = Arc<Mutex<Vec<(usize, usize, usize)>>>;

fn progress_sink() -> ProgressSink {
    Arc::new(Mutex::new(Vec::new()))
}

/// A progress callback that appends every observation to `sink` verbatim — all
/// three counters, so an assertion can read the work fraction the consuming app
/// computes rather than only the pair the engine used to report.
fn record_progress(sink: ProgressSink) -> Arc<dyn Fn(Progress) + Send + Sync> {
    Arc::new(move |update: Progress| {
        sink.lock()
            .unwrap()
            .push((update.processed, update.worked, update.total))
    })
}

fn index_with(
    input: &Path,
    destination: &Path,
    config: Config,
    resume: bool,
    cancellation: Option<Arc<AtomicBool>>,
    progress: Option<Arc<dyn Fn(Progress) + Send + Sync>>,
) -> anyhow::Result<IndexStats> {
    let _serialized = model_lock();
    run_index(IndexRequest {
        paths: std::slice::from_ref(&input.to_path_buf()),
        out: destination,
        config,
        resume,
        overwrite: false,
        artifacts: false,
        retry_errors: false,
        include_paths: None,
        cancellation,
        runtime: None,
        progress,
    })
}

fn hashed_rows(destination: &Path) -> i64 {
    connect(destination)
        .unwrap()
        .query_row(
            "SELECT COUNT(*) FROM files WHERE sha1 IS NOT NULL",
            [],
            |row| row.get(0),
        )
        .unwrap()
}

/// The whole acceptance in one pass: a resume with both gates armed hashes the
/// rows it would otherwise have skipped, writes the RIGHT hash, and leaves
/// everything else about those rows exactly as it found it.
#[test]
fn a_backfill_pass_hashes_skipped_rows_and_touches_nothing_else() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let destination = temp.path().join("corpus.sqlite");
    let paths = sample_tree(&input, 4);

    // The corpus as it exists on the live box: fully indexed, and every row
    // written by a build that was not hashing.
    let first = index_with(
        &input,
        &destination,
        backfill_config(false, false),
        false,
        None,
        None,
    )
    .unwrap();
    assert_eq!(first.files, 4);
    assert_eq!(first.hashed, 0, "the lane cannot run on a non-resume");
    let before = file_rows(&destination);
    assert!(
        before.iter().all(|row| row.8.is_none()),
        "the fixture must start unhashed or it proves nothing"
    );

    let stats = index_with(
        &input,
        &destination,
        backfill_config(true, true),
        true,
        None,
        None,
    )
    .unwrap();

    // Nothing was indexed: no file was extracted, embedded or re-written.
    assert_eq!(stats.files, 0, "a backfill pass writes no file rows");
    assert_eq!(stats.embedded_chunks, 0, "a backfill pass embeds nothing");
    assert_eq!(stats.errors, 0);
    // The rows left `skipped` for `hashed` — this run did touch them.
    assert_eq!(stats.hashed, 4);
    assert_eq!(stats.hash_failed, 0, "every owed file was readable");
    assert_eq!(stats.skipped, 0, "hashed rows are not also skipped");
    assert_eq!(stats.capped, 0);

    let after = file_rows(&destination);
    assert_eq!(after.len(), before.len(), "no row was added or removed");
    for (old, new) in before.iter().zip(after.iter()) {
        assert_eq!(old.0, new.0, "path");
        assert_eq!(old.1, new.1, "method must survive a backfill");
        assert_eq!(old.2, new.2, "chars must survive a backfill");
        assert_eq!(old.3, new.3, "pages must survive a backfill");
        assert_eq!(old.4, new.4, "indexed_at must survive a backfill");
        assert_eq!(old.5, new.5, "attempts must survive a backfill");
        assert_eq!(old.6, new.6, "last_attempt_at must survive a backfill");
        assert_eq!(old.7, new.7, "elapsed_ms must survive a backfill");
    }
    for path in &paths {
        let stored = after
            .iter()
            .find(|row| &row.0 == path)
            .unwrap_or_else(|| panic!("{path} must still have a row"));
        assert_eq!(
            stored.8.as_deref(),
            Some(expected_sha1(Path::new(path)).as_str()),
            "the backfilled hash must be the file's actual sha1"
        );
    }

    // And it is a ONE-TIME cost: the rows now carry hashes, so the next armed
    // resume owes nothing and reports them as ordinary skips again.
    let settled = index_with(
        &input,
        &destination,
        backfill_config(true, true),
        true,
        None,
        None,
    )
    .unwrap();
    assert_eq!(settled.hashed, 0, "a hashed row must not be re-hashed");
    assert_eq!(settled.skipped, 4);
}

/// Both gates, independently. Either one off and the resume is the resume this
/// build shipped before the lane existed — same counters, and not one byte
/// written to `sha1`. This is what makes the feature safe to deploy inert.
#[test]
fn either_gate_off_leaves_the_resume_exactly_as_it_was() {
    for (hash, hash_backfill) in [(false, false), (true, false), (false, true)] {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("input");
        let destination = temp.path().join("corpus.sqlite");
        sample_tree(&input, 3);

        index_with(
            &input,
            &destination,
            backfill_config(false, false),
            false,
            None,
            None,
        )
        .unwrap();
        let before = file_rows(&destination);

        let stats = index_with(
            &input,
            &destination,
            backfill_config(hash, hash_backfill),
            true,
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            stats.skipped, 3,
            "hash={hash} hash_backfill={hash_backfill}: every row stays a plain skip"
        );
        assert_eq!(stats.hashed, 0, "hash={hash} hash_backfill={hash_backfill}");
        assert_eq!(stats.files, 0, "hash={hash} hash_backfill={hash_backfill}");
        assert_eq!(
            file_rows(&destination),
            before,
            "hash={hash} hash_backfill={hash_backfill}: the corpus must be untouched"
        );
    }
}

/// Total GROWS when the lane is armed, and that is the intended reading: the run
/// genuinely has the owed rows to do. What must not happen is the two counters
/// disagreeing — a `processed` that outruns `total`, or a `total` that hides
/// work the run is spending time on and skews every rate derived from it.
#[test]
fn an_armed_backfill_grows_the_total_and_processed_reaches_it() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let destination = temp.path().join("corpus.sqlite");
    sample_tree(&input, 4);

    index_with(
        &input,
        &destination,
        backfill_config(false, false),
        false,
        None,
        None,
    )
    .unwrap();

    // A plain resume of a finished corpus: nothing to do, and `total` says so.
    let plain = progress_sink();
    let sink = plain.clone();
    index_with(
        &input,
        &destination,
        backfill_config(true, false),
        true,
        None,
        Some(record_progress(sink)),
    )
    .unwrap();
    assert_eq!(
        plain.lock().unwrap().as_slice(),
        [(0, 0, 0)],
        "an unarmed resume of a finished corpus has nothing to report"
    );

    let armed = progress_sink();
    let sink = armed.clone();
    let stats = index_with(
        &input,
        &destination,
        backfill_config(true, true),
        true,
        None,
        Some(record_progress(sink)),
    )
    .unwrap();

    let samples = armed.lock().unwrap().clone();
    assert_eq!(
        samples,
        vec![(0, 0, 4), (1, 0, 4), (2, 0, 4), (3, 0, 4), (4, 0, 4)],
        "the owed rows are in `total` from the first sample and `processed` reaches it — \
         and `worked` stays FLAT at zero the whole way, because a hash is not indexing"
    );
    assert_eq!(stats.hashed, 4);
}

/// **The gate signal.** A pure-hash prefix must advance `processed` while
/// `worked` stays flat, and the indexing pass that follows must advance both
/// together — because the consumer's ETA gate is `Δworked/Δprocessed` over a
/// recent window, and the two stretches run at wildly different speeds off the
/// same `(processed, total)` pair.
///
/// The run below is the real shape: an armed resume over a corpus with rows
/// owing a hash AND a new file to index, in that order (the lane runs first by
/// design). Without `worked` the app sees one undifferentiated counter and
/// projects the hash rate across the indexing pass.
#[test]
fn the_hash_prefix_advances_processed_while_worked_stays_flat() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let destination = temp.path().join("corpus.sqlite");
    sample_tree(&input, 3);

    // Seed unhashed: these three rows are what the lane will owe.
    index_with(
        &input,
        &destination,
        backfill_config(false, false),
        false,
        None,
        None,
    )
    .unwrap();
    // One genuinely new file, so the run has an indexing pass after the lane.
    fs::write(input.join("fresh.txt"), "a new file to index this run").unwrap();

    let samples = progress_sink();
    let sink = samples.clone();
    let stats = index_with(
        &input,
        &destination,
        backfill_config(true, true),
        true,
        None,
        Some(record_progress(sink)),
    )
    .unwrap();
    assert_eq!(stats.hashed, 3, "the three seeded rows were owed a hash");
    assert_eq!(stats.files, 1, "and one new file was indexed");

    let samples = samples.lock().unwrap().clone();
    // total = 3 owed hashes + 1 file to index.
    assert_eq!(
        samples,
        vec![(0, 0, 4), (1, 0, 4), (2, 0, 4), (3, 0, 4), (4, 1, 4)],
        "worked is FLAT across the hash prefix and advances only on the indexed file"
    );

    // Stated as the two fractions a consumer actually computes, so a change to
    // the counters that broke the gate would fail here and not only above.
    let prefix = &samples[..4];
    let (p0, w0, _) = prefix[0];
    let (p1, w1, _) = *prefix.last().unwrap();
    assert!(p1 > p0, "the prefix moved `processed`");
    assert_eq!(
        w1 - w0,
        0,
        "f = 0 across the hash prefix: the consumer's ETA gate is shut"
    );
    let (p2, w2, _) = *samples.last().unwrap();
    assert_eq!(
        (p2 - p1, w2 - w1),
        (1, 1),
        "f = 1 across the indexing pass: the gate opens and the ETA is honest"
    );
}

/// With the lane off — every run by default — `worked` is identical to
/// `processed` at every observation. The counter is additive: it cannot change
/// what an unarmed run reports, and the consumer's f is 1 throughout exactly as
/// it was before the field existed.
#[test]
fn worked_tracks_processed_exactly_when_the_backfill_lane_is_off() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let destination = temp.path().join("corpus.sqlite");
    sample_tree(&input, 3);

    let samples = progress_sink();
    let sink = samples.clone();
    index_with(
        &input,
        &destination,
        backfill_config(true, false),
        false,
        None,
        Some(record_progress(sink)),
    )
    .unwrap();

    let samples = samples.lock().unwrap().clone();
    assert_eq!(
        samples.len(),
        4,
        "one seed tick plus one per file: {samples:?}"
    );
    for (processed, worked, total) in samples {
        assert_eq!(
            processed, worked,
            "no hash lane means every processed file was indexed"
        );
        assert_eq!(total, 3);
    }
}

/// A file the lane claims but cannot READ must be counted, not dropped.
///
/// This is the common case on a real drive, not an edge: locked mail stores, VM
/// disks a hypervisor holds open, anything another process has exclusive,
/// anything the account cannot read. Such a row is out of `skipped` (the lane
/// claimed it), gets no `sha1`, writes no `error:` row and so lands in neither
/// `files` nor `errors` — so without `hash_failed` its only trace would be an
/// unexplained gap between the owed count the run announced and the hashes it
/// produced, which is indistinguishable from a bug in the lane.
///
/// Unreadability is staged by DELETING the file after the walk has recorded it
/// and before the lane reaches it, driven off the opening `progress(0, total)`
/// sample. That is deterministic and identical on every platform, unlike a
/// permissions bit (which root ignores) or a deny-read share mode (Windows
/// only), and it reproduces exactly the shape that matters: a row whose file
/// the walk saw and the hash read cannot get at.
#[test]
fn a_file_the_lane_cannot_read_is_counted_not_silently_dropped() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let destination = temp.path().join("corpus.sqlite");
    let paths = sample_tree(&input, 2);

    index_with(
        &input,
        &destination,
        backfill_config(false, false),
        false,
        None,
        None,
    )
    .unwrap();
    let before = file_rows(&destination);

    // Removed after the walk that records it, before the lane reads it. The
    // walk drives pruning off its own snapshot, so the row survives the run.
    let doomed = paths[1].clone();
    // Captured so the file can be put back byte-for-byte, mtime included, for
    // the second armed run below — anything else would make the row look CHANGED
    // and send it down the indexing path instead of the lane.
    let doomed_bytes = fs::read(&doomed).unwrap();
    let doomed_mtime = fs::metadata(&doomed).unwrap().modified().unwrap();
    let samples = progress_sink();
    let sink = samples.clone();
    let stats = index_with(
        &input,
        &destination,
        backfill_config(true, true),
        true,
        None,
        Some(Arc::new(move |update: Progress| {
            if update.processed == 0 {
                fs::remove_file(&doomed).unwrap();
            }
            sink.lock()
                .unwrap()
                .push((update.processed, update.worked, update.total))
        })),
    )
    .unwrap();

    assert_eq!(stats.hashed, 1, "the readable file is still hashed");
    assert_eq!(stats.hash_failed, 1, "the unreadable one is COUNTED");
    // The whole point: nothing is left unattributed. Both rows were owed, both
    // were attempted, and the two counters add back up to the owed count.
    assert_eq!(
        stats.hashed + stats.hash_failed,
        2,
        "the lane's accounting must close — no unexplained remainder"
    );
    assert_eq!(stats.skipped, 0, "a claimed row is not also a skip");
    assert_eq!(stats.files, 0, "nothing was indexed");
    assert_eq!(stats.errors, 0, "a hash miss writes no error row");
    assert_eq!(
        samples.lock().unwrap().as_slice(),
        [(0, 0, 2), (1, 0, 2), (2, 0, 2)],
        "an unreadable file still advances `processed` — the read was attempted — \
         and neither arm of the lane advances `worked`"
    );

    // The row itself survives untouched, including its NULL sha1: a hash that
    // could not be taken says nothing about an extraction that succeeded.
    let after = file_rows(&destination);
    assert_eq!(after.len(), 2, "the unreadable file keeps its row");
    let missed = after
        .iter()
        .find(|row| row.0 == paths[1])
        .expect("the row for the removed file is not pruned by its own run");
    assert_eq!(missed.8, None, "no hash landed");
    let original = before.iter().find(|row| row.0 == paths[1]).unwrap();
    assert_eq!(missed, original, "the row is byte-for-byte as it was");

    // And because no hash ever lands, `backfill_candidate` re-claims it on every
    // armed run — the honest counterpart to the one-time-cost property that
    // holds for rows the lane CAN read. Restored identically (same bytes, same
    // mtime) so the resume predicate still calls it finished and unchanged, then
    // removed again at the same point in the run.
    fs::write(&paths[1], &doomed_bytes).unwrap();
    fs::OpenOptions::new()
        .write(true)
        .open(&paths[1])
        .unwrap()
        .set_modified(doomed_mtime)
        .unwrap();
    let doomed = paths[1].clone();
    let again = index_with(
        &input,
        &destination,
        backfill_config(true, true),
        true,
        None,
        Some(Arc::new(move |update: Progress| {
            if update.processed == 0 {
                fs::remove_file(&doomed).unwrap();
            }
        })),
    )
    .unwrap();
    assert_eq!(
        (again.hashed, again.hash_failed, again.skipped),
        (0, 1, 1),
        "the hashed row is now a plain skip; the unreadable one is owed again"
    );
}

/// A killed backfill keeps what it hashed and the next run owes only the rest —
/// the same durability contract the indexing pass has, reached through the same
/// batched commit. Cancellation is driven from the progress callback, which the
/// lane calls once per row, so the kill lands at an exact known row.
#[test]
fn a_cancelled_backfill_keeps_its_hashes_and_the_next_run_owes_the_rest() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let destination = temp.path().join("corpus.sqlite");
    sample_tree(&input, 4);

    let mut seed = backfill_config(false, false);
    // One row per commit, so the assertion below is about the checkpoint the
    // lane shares with `add` rather than about `finish` saving it at the end.
    seed.commit_batch = 1;
    index_with(&input, &destination, seed.clone(), false, None, None).unwrap();

    let cancellation = Arc::new(AtomicBool::new(false));
    let flag = cancellation.clone();
    let mut armed = seed.clone();
    armed.hash = true;
    armed.hash_backfill = true;
    let error = index_with(
        &input,
        &destination,
        armed.clone(),
        true,
        Some(cancellation),
        Some(Arc::new(move |update: Progress| {
            if update.processed >= 2 {
                flag.store(true, Ordering::Relaxed);
            }
        })),
    )
    .expect_err("a cancelled run reports itself cancelled");
    assert_eq!(
        error.to_string(),
        "indexing cancelled; 0 file(s) and 2 sha1 backfill(s) committed",
        "a backfill cancel takes the run's ordinary cancelled path, and says what \
         it actually committed rather than reporting the zero files it indexed"
    );
    assert_eq!(
        hashed_rows(&destination),
        2,
        "the hashes taken before the cancel are committed, and no more were taken"
    );

    // The resume owes exactly the remainder — not all four, and not none.
    let finished = index_with(&input, &destination, armed, true, None, None).unwrap();
    assert_eq!(
        finished.hashed, 2,
        "a resumed backfill owes only what is left"
    );
    assert_eq!(
        finished.skipped, 2,
        "the already-hashed rows are plain skips"
    );
    assert_eq!(hashed_rows(&destination), 4);
}

/// The lane's 1 GiB ceiling, on a file that reports its size without occupying
/// it. Above the ceiling the drives-analytics app does not compare hashes
/// either, so reading a multi-gigabyte file end to end would buy a value nothing
/// consults — the row is left unhashed and stays an ordinary skip.
///
/// Note what this does NOT assert: the forward path has no such ceiling and does
/// not gain one here. That is live behaviour on a running deployment, so the
/// engine/app agreement is partial by design.
#[test]
fn a_file_over_the_backfill_ceiling_is_left_unhashed() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let destination = temp.path().join("corpus.sqlite");
    fs::create_dir_all(&input).unwrap();
    fs::write(
        input.join("small.txt"),
        "A small report the lane will hash.",
    )
    .unwrap();
    // `.bin` is an extension no extractor in this build handles, which lands the
    // row on the terminal `excluded:unsupported` method — a COMPLETE row, so
    // resume skips it and the lane is the only thing that could ever hash it.
    // That is the case the ceiling has to stop; an INCOMPLETE row would be
    // re-indexed instead and never reach the lane at all.
    let huge = input.join("huge.bin");
    sparse_file(&huge, (1 << 30) + 1);

    let mut config = backfill_config(false, false);
    // Above the oversize cut-off, or `extract` short-circuits to `name-only`
    // (an incomplete row) before the unsupported-extension verdict is reached.
    config.max_bytes = 4 << 30;
    index_with(&input, &destination, config.clone(), false, None, None).unwrap();

    let mut armed = config;
    armed.hash = true;
    armed.hash_backfill = true;
    let stats = index_with(&input, &destination, armed, true, None, None).unwrap();

    assert_eq!(stats.hashed, 1, "only the small file is under the ceiling");
    assert_eq!(
        stats.skipped, 1,
        "the oversized row is not claimed by the lane, so it stays a plain skip"
    );
    let rows = file_rows(&destination);
    let huge_row = rows
        .iter()
        .find(|row| row.0.ends_with("huge.bin"))
        .expect("the oversized file is still indexed, just not hashed");
    assert_eq!(huge_row.1, "excluded:unsupported");
    assert_eq!(
        huge_row.8, None,
        "a file at or above the ceiling must be left unhashed"
    );
    let small_row = rows
        .iter()
        .find(|row| row.0.ends_with("small.txt"))
        .expect("the small file still has a row");
    assert_eq!(
        small_row.8.as_deref(),
        Some(expected_sha1(&input.join("small.txt")).as_str())
    );
}

/// A file that REPORTS `len` bytes without occupying them. The lane's ceiling
/// reads `FileRec::size`, i.e. the walker's `metadata().len()`, so a sparse file
/// exercises it exactly as a real one would at no cost in disk or time.
fn sparse_file(path: &Path, len: u64) {
    fs::File::create(path).unwrap();
    // NTFS needs the attribute set BEFORE the extension or it allocates the
    // whole range; POSIX filesystems make a hole out of `set_len` on their own.
    // A failure here is not fatal — the file is simply allocated, and nothing in
    // this test ever reads it.
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("fsutil")
            .args(["sparse", "setflag"])
            .arg(path)
            .status();
    }
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(len)
        .unwrap();
}
