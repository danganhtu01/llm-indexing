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
    MODEL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
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
        include_paths: None,
        cancellation: None,
        progress: None,
    })
    .unwrap();
    assert_eq!(stats.files, 3);

    let connection = connect(&output).unwrap();
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
        include_paths,
        cancellation,
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

#[test]
fn cancellation_keeps_committed_work_and_resume_finishes_it() {
    const FILES: i64 = 24;
    // Cancel late enough that the writer must already hold files: extraction can
    // only run ahead of it by the channel depth (workers × 4) plus one in-flight
    // send per worker, so 20 extracted files means at least ~10 were received,
    // embedded and stored before the flag was seen.
    const CANCEL_AFTER: usize = 20;

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
    assert!(retained < FILES, "the run really was interrupted: {retained}");

    // Resume skips precisely what was kept and finishes the rest.
    let finished = index(&input, &destination, true, None, None, None).unwrap();
    assert_eq!(finished.skipped as i64, retained);
    assert_eq!(finished.files as i64, FILES - retained);
    assert_eq!(indexed_files(&destination), FILES);
}
