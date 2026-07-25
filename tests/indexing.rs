use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use llm_indexing::config::Config;
use llm_indexing::model::IndexStats;
use llm_indexing::normalize::Normalizer;
use llm_indexing::pipeline::{run_index, IndexRequest};
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
    progress: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>,
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
        Some(Arc::new(move |processed, _| {
            if processed >= CANCEL_AFTER {
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
