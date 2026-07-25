use std::fs;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use llm_indexing::jobs_store::{JobsStore, INTERRUPTED_ERROR};
use llm_indexing::service::{router, ServiceConfig};
use llm_indexing::vision::VisionMode;
use serde_json::{json, Value};
use tower::ServiceExt;

#[tokio::test]
async fn http_job_publishes_only_sqlite_and_confines_paths() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    fs::create_dir_all(&input).unwrap();
    fs::write(
        input.join("hello.txt"),
        "Vietnamese ngân hàng and English compliance.",
    )
    .unwrap();
    let app = router(ServiceConfig {
        output_root: output.clone(),
        allowed_roots: vec![input.clone()],
        default_paths: vec![input.clone()],
        config_path: None,
        ocr_langs: "vie+eng".into(),
        workers: 1,
        max_pending: 2,
        max_body: 1024 * 1024,
        vision_max: VisionMode::Off,
    })
    .unwrap();

    let health = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/index")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"id":"job-1","paths":[input],"output":"corpus.sqlite",
                                "ocr":"off","workers":1})
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let job = wait_for_job(&app, "job-1").await;
    assert_eq!(job["status"], "complete", "{job}");
    // Only the published corpus plus P0-11's persisted job store
    // (`jobs.sqlite`) live under `output_root` — nothing else.
    assert_eq!(output_dir_names(&output), vec!["corpus.sqlite", "jobs.sqlite"]);
    assert!(output.join("corpus.sqlite").is_file());

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/index")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"id":"job-2","paths":[temp.path()],"output":"bad.sqlite",
                                "ocr":"off"})
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let denied = wait_for_job(&app, "job-2").await;
    assert_eq!(denied["status"], "error");
    assert!(denied["error"]
        .as_str()
        .unwrap()
        .contains("INDEX_ALLOWED_ROOTS"));
}

// ── Destination guards ───────────────────────────────────────────────────────
//
// Jobs write straight into the published database, so the guards deciding
// whether an existing corpus may be touched are the whole safety contract.

/// Sorted top-level filenames under `output_root` — used to assert nothing
/// stray was left behind (P0-11 makes `jobs.sqlite` a permanent, expected
/// sibling of the published corpora).
fn output_dir_names(output: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(output)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn guard_router(output: &std::path::Path, input: &std::path::Path) -> axum::Router {
    guard_router_with_config(output, input, None)
}

fn guard_router_with_config(
    output: &std::path::Path,
    input: &std::path::Path,
    config_path: Option<std::path::PathBuf>,
) -> axum::Router {
    fs::create_dir_all(input).unwrap();
    fs::create_dir_all(output).unwrap();
    router(ServiceConfig {
        output_root: output.to_path_buf(),
        allowed_roots: vec![input.to_path_buf()],
        default_paths: vec![input.to_path_buf()],
        config_path,
        ocr_langs: "vie+eng".into(),
        workers: 1,
        max_pending: 2,
        max_body: 1024 * 1024,
        vision_max: VisionMode::Off,
    })
    .unwrap()
}

/// A corpus holding one row for a file that is not in the input tree, so an
/// overwrite is visible as its disappearance.
fn write_stale_corpus(destination: &std::path::Path) {
    let connection = rusqlite::Connection::open(destination).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE files(
                id INTEGER PRIMARY KEY, path TEXT UNIQUE, drive TEXT, dir TEXT,
                name TEXT, ext TEXT, size INTEGER, mtime REAL, lang TEXT,
                method TEXT, ocr_used INTEGER, pages INTEGER, chars INTEGER,
                sha1 TEXT, indexed_at REAL
             );
             INSERT INTO files(id,path,name,ext,size,mtime,lang,method,ocr_used,pages,chars,indexed_at)
             VALUES (1,'/gone/stale.txt','stale.txt','.txt',1,0.0,'en','text',0,0,1,0.0);",
        )
        .unwrap();
}

#[tokio::test]
async fn an_existing_corpus_is_refused_without_resume_or_overwrite() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    let app = guard_router(&output, &input);
    fs::write(input.join("hello.txt"), "compliance report").unwrap();
    write_stale_corpus(&output.join("corpus.sqlite"));

    let (status, _) = submit_body(
        &app,
        json!({"id":"guard","paths":[input],"output":"corpus.sqlite","ocr":"off"}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let job = wait_for_job(&app, "guard").await;
    assert_eq!(job["status"], "error", "{job}");
    assert!(
        job["error"].as_str().unwrap().contains("already exists"),
        "{job}"
    );
    // Refusing must not touch what is there.
    let connection = rusqlite::Connection::open(output.join("corpus.sqlite")).unwrap();
    let files: i64 = connection
        .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))
        .unwrap();
    assert_eq!(files, 1);
}

#[tokio::test]
async fn overwrite_replaces_the_existing_corpus() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    let app = guard_router(&output, &input);
    fs::write(input.join("hello.txt"), "compliance report").unwrap();
    let destination = output.join("corpus.sqlite");
    write_stale_corpus(&destination);

    let (status, _) = submit_body(
        &app,
        json!({"id":"over","paths":[input],"output":"corpus.sqlite","ocr":"off",
               "workers":1,"overwrite":true}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let job = wait_for_job(&app, "over").await;
    assert_eq!(job["status"], "complete", "{job}");

    // Truncated up front: the stale row is gone, not merged into the new run.
    let connection = rusqlite::Connection::open(&destination).unwrap();
    let stale: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM files WHERE path='/gone/stale.txt'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(stale, 0);
    assert_eq!(job["files"], 1);
    // Nothing but the published database and the persisted job store are left
    // behind.
    assert_eq!(output_dir_names(&output), vec!["corpus.sqlite", "jobs.sqlite"]);
}

#[tokio::test]
async fn a_failed_overwrite_leaves_the_existing_corpus_intact() {
    // An overwrite job writes into the destination, so the previous corpus has
    // to be deleted at some point — but only once everything that can
    // predictably fail has succeeded. An unreadable config is the cheapest
    // stand-in for that class (a missing vision model or a cold embedding
    // cache fail the same way, earlier than any write): the job must fail with
    // the old corpus untouched rather than destroy it and put nothing back.
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    let config = temp.path().join("broken.yaml");
    fs::write(&config, "workers: [this is not a scalar\n").unwrap();
    let app = guard_router_with_config(&output, &input, Some(config));
    fs::write(input.join("hello.txt"), "compliance report").unwrap();
    let destination = output.join("corpus.sqlite");
    write_stale_corpus(&destination);

    let (status, _) = submit_body(
        &app,
        json!({"id":"doomed","paths":[input],"output":"corpus.sqlite","ocr":"off",
               "workers":1,"overwrite":true}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let job = wait_for_job(&app, "doomed").await;
    assert_eq!(job["status"], "error", "{job}");
    // Specifically the config load, i.e. a step that runs before any write and
    // must therefore run before the deletion too.
    assert!(
        job["error"].as_str().unwrap().contains("YAML config"),
        "{job}"
    );

    assert!(destination.is_file(), "the previous corpus was destroyed");
    let connection = rusqlite::Connection::open(&destination).unwrap();
    let stale: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM files WHERE path='/gone/stale.txt'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        stale, 1,
        "a job that never indexed anything kept the corpus"
    );
}

// ── P0-11: restart/crash reconciliation ─────────────────────────────────────

#[tokio::test]
async fn a_completed_job_is_served_from_the_persisted_store_after_a_restart() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    fs::create_dir_all(&input).unwrap();
    fs::write(input.join("hello.txt"), "compliance report").unwrap();
    let app = guard_router(&output, &input);

    let (status, _) = submit_body(
        &app,
        json!({"id":"persisted","paths":[input.clone()],"output":"corpus.sqlite","ocr":"off"}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let job = wait_for_job(&app, "persisted").await;
    assert_eq!(job["status"], "complete", "{job}");

    // A brand-new router over the SAME output_root is exactly what a service
    // restart looks like: a fresh in-memory `jobs` map with nothing in it,
    // backed by the same `jobs.sqlite` already on disk. The old router (and
    // its in-memory state) is dropped here, never consulted again.
    drop(app);
    let restarted = guard_router(&output, &input);
    let (status, body) = get_json_status(&restarted, "/jobs/persisted").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["id"], "persisted");
    assert_eq!(body["status"], "complete", "{body}");
}

#[tokio::test]
async fn an_unknown_job_id_still_404s_after_a_restart() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    fs::create_dir_all(&input).unwrap();
    fs::create_dir_all(&output).unwrap();
    let app = guard_router(&output, &input);
    let (status, _) = get_json_status(&app, "/jobs/never-existed").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn a_startup_sweep_rewrites_jobs_left_non_terminal_by_a_prior_instance_to_error() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    fs::create_dir_all(&input).unwrap();
    fs::create_dir_all(&output).unwrap();

    // Reproduce what a process killed mid-job leaves on disk: `queued`,
    // `running` and `cancelling` rows in `jobs.sqlite` with no worker left
    // that will ever finish them — nothing about this depends on the HTTP
    // layer, so it is written directly through the store the way a crash
    // would leave it, rather than raced against a real (near-instant) job.
    {
        let store = JobsStore::open(&output).unwrap();
        store
            .record(
                "was-queued",
                &json!({"id":"was-queued","status":"queued","submitted_at":1.0}),
            )
            .unwrap();
        store
            .record(
                "was-running",
                &json!({"id":"was-running","status":"running","processed":3,"total":10}),
            )
            .unwrap();
        store
            .record(
                "already-done",
                &json!({"id":"already-done","status":"complete","files":2}),
            )
            .unwrap();
    }

    // `router(...)` runs the P0-11 startup sweep before returning — this is
    // the moment a restarted process would run it, before it ever binds a
    // listener.
    let app = guard_router(&output, &input);

    for id in ["was-queued", "was-running"] {
        let (status, body) = get_json_status(&app, &format!("/jobs/{id}")).await;
        assert_eq!(status, StatusCode::OK, "{id}: {body}");
        assert_eq!(body["status"], "error", "{id}: {body}");
        assert_eq!(body["error"], INTERRUPTED_ERROR, "{id}: {body}");
        assert!(body["completed_at"].is_number(), "{id}: {body}");
    }
    // Progress the running job had made survives the rewrite — only
    // status/error/completed_at change.
    let (_, running) = get_json_status(&app, "/jobs/was-running").await;
    assert_eq!(running["processed"], 3);
    assert_eq!(running["total"], 10);

    // A job that was already terminal before the restart is untouched.
    let (status, done) = get_json_status(&app, "/jobs/already-done").await;
    assert_eq!(status, StatusCode::OK, "{done}");
    assert_eq!(done["status"], "complete", "{done}");
    assert_eq!(done["files"], 2);
}

// ── Cancellation ─────────────────────────────────────────────────────────
//
// A job's cancellation flag lives only in this process's memory (see
// `AppState::cancellations`), never in `jobs.sqlite` — so an id this process
// has no live flag for (aged out, or genuinely from a prior instance) must
// still get an honest answer instead of a bare 404, and cancelling it must
// never flip a persisted terminal row to "cancelled".

#[tokio::test]
async fn cancelling_a_truly_unknown_job_id_404s() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    fs::create_dir_all(&input).unwrap();
    fs::create_dir_all(&output).unwrap();
    let app = guard_router(&output, &input);

    let (status, body) = post_json(&app, "/jobs/never-existed/cancel", json!({})).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
}

#[tokio::test]
async fn cancelling_a_job_this_process_only_knows_from_jobs_sqlite_reports_conflict_not_404() {
    // Reproduces the P0-11-era gap: a job this process has no in-memory
    // cancellation flag for (aged out of `cancellations`, or — the far more
    // common case — swept to a terminal `error` by `sweep_interrupted` at
    // startup, exactly as `a_startup_sweep_rewrites_...` above sets up) must
    // not 404 just because `GET /jobs/{id}` would happily answer for it from
    // `jobs.sqlite`. And crucially: this call must not be the thing that
    // marks a completed/errored run "cancelled" — there is no live worker
    // behind it to cancel.
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    fs::create_dir_all(&input).unwrap();
    fs::create_dir_all(&output).unwrap();
    {
        let store = JobsStore::open(&output).unwrap();
        store
            .record(
                "was-running",
                &json!({"id":"was-running","status":"running","processed":3,"total":10}),
            )
            .unwrap();
        store
            .record(
                "already-done",
                &json!({"id":"already-done","status":"complete","files":2}),
            )
            .unwrap();
    }
    // `router(...)` (via `guard_router`) runs the startup sweep, so
    // "was-running" is now a terminal `error` row before this process ever
    // saw a cancellation request for it — the same state a real restart
    // leaves.
    let app = guard_router(&output, &input);

    for id in ["was-running", "already-done", "unknown-but-plausible"] {
        let plausible_but_unknown = id == "unknown-but-plausible";
        let (status, body) = post_json(&app, &format!("/jobs/{id}/cancel"), json!({})).await;
        if plausible_but_unknown {
            assert_eq!(status, StatusCode::NOT_FOUND, "{id}: {body}");
            continue;
        }
        assert_eq!(
            status,
            StatusCode::CONFLICT,
            "{id}: expected \"not active\", not a bare 404: {body}"
        );
    }

    // Neither persisted row was mutated by the cancel attempt.
    let (_, was_running) = get_json_status(&app, "/jobs/was-running").await;
    assert_eq!(was_running["status"], "error", "{was_running}");
    assert_eq!(was_running["error"], INTERRUPTED_ERROR, "{was_running}");
    let (_, already_done) = get_json_status(&app, "/jobs/already-done").await;
    assert_eq!(already_done["status"], "complete", "{already_done}");
}

#[tokio::test]
async fn cancelling_a_genuinely_running_job_still_accepts_and_marks_it_cancelling() {
    // The live path (a cancellation flag this process itself created via
    // `submit`) must keep working exactly as before.
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    sample_corpus(&input, 40);
    let app = guard_router(&output, &input);

    let (status, _) = post_json(
        &app,
        "/index",
        json!({"id":"job-cancel-me","paths":[input],"output":"corpus.sqlite","ocr":"off"}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    wait_until_running(&app, "job-cancel-me").await;

    let (status, body) = post_json(&app, "/jobs/job-cancel-me/cancel", json!({})).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{body}");
    assert_eq!(body["status"], "cancelling", "{body}");

    // `wait_for_job` only recognises "complete"/"error" as terminal, so a
    // cancelled run needs its own poll here.
    let mut job = json!({});
    for _ in 0..1500 {
        job = get_json(&app, "/jobs/job-cancel-me").await;
        if job["status"] == "cancelled" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(job["status"], "cancelled", "{job}");
}

#[tokio::test]
async fn an_unreadable_corpus_is_reported_rather_than_read_as_empty() {
    // The failure mode this guards: a corpus that cannot be read answering
    // `indexed_files: 0`. A consumer cannot tell that from a corpus that is
    // genuinely empty, and "your documents are gone" is the wrong thing to
    // show over a database whose rows are all still there.
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    let app = guard_router(&output, &input);
    fs::write(
        output.join("corpus.sqlite"),
        b"this is not a sqlite database",
    )
    .unwrap();

    let response = get(&app, "/corpus/status").await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["error"], "corpus database unreadable", "{body}");
    assert!(body["indexed_files"].is_null(), "{body}");

    // The tree and document routes answer the same way rather than presenting
    // an empty corpus as a complete one.
    let tree = get(&app, "/corpus/tree?root=input").await;
    assert_eq!(tree.status(), StatusCode::SERVICE_UNAVAILABLE);
    let text = get(&app, "/corpus/documents/1/text").await;
    assert_eq!(text.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn an_absent_corpus_still_reads_as_empty_rather_than_an_error() {
    // The counterpart: "no job has written this output yet" is a normal state
    // and must stay a zeroed 200, not get swept into the error above.
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    let app = guard_router(&output, &input);

    let status = get_json(&app, "/corpus/status").await;
    assert_eq!(status["indexed_files"], 0, "{status}");
    assert_eq!(status["pending_files"], 0, "{status}");
    assert_eq!(status["writing"], false, "{status}");
    let tree = get(&app, "/corpus/tree?root=input").await;
    assert_eq!(tree.status(), StatusCode::OK);
    let text = get(&app, "/corpus/documents/1/text").await;
    assert_eq!(text.status(), StatusCode::NOT_FOUND);
}

async fn wait_for_job(app: &axum::Router, id: &str) -> Value {
    // Budget generously: a real `/index` job loads the e5 embedding model
    // (several seconds of CPU init on a cold cache) before it can complete, so
    // a tight poll window flakes as "job did not finish" on slower boxes even
    // though the job succeeds. 30s stays bounded enough to catch a truly hung
    // job while never racing correct-but-slow model loading.
    for _ in 0..1500 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/jobs/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let job: Value = serde_json::from_slice(&bytes).unwrap();
        if matches!(job["status"].as_str(), Some("complete" | "error")) {
            return job;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("job did not finish")
}

// ── /corpus/tree, /corpus/documents/{id}/text, /corpus/status ───────────────
//
// These write the `corpus.sqlite` schema directly with rusqlite rather than
// running a real `/index` job, so they stay independent of the pre-existing
// model-download flake in `http_job_publishes_only_sqlite_and_confines_paths`
// (embedding needs a network-fetched model; these routes only ever read).

/// Publish a corpus.sqlite with one indexed file, matching llm-indexing's
/// `files`/`fts` schema (see `src/store.rs`), keyed by `indexed_path` — the
/// absolute path `/corpus/tree`'s walk must reproduce to join it.
fn write_fixture_corpus(output_root: &std::path::Path, indexed_path: &str) {
    let connection = rusqlite::Connection::open(output_root.join("corpus.sqlite")).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE files(
                id INTEGER PRIMARY KEY, path TEXT UNIQUE, drive TEXT, dir TEXT,
                name TEXT, ext TEXT, size INTEGER, mtime REAL, lang TEXT,
                method TEXT, ocr_used INTEGER, pages INTEGER, chars INTEGER,
                sha1 TEXT, indexed_at REAL
             );
             CREATE VIRTUAL TABLE fts USING fts5(name, path, content, tokens);",
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO files(id,path,dir,name,ext,size,mtime,lang,method,ocr_used,pages,chars,sha1,indexed_at)
             VALUES (1,?1,'folder','indexed.txt','.txt',900,0.0,'en','text',0,1,900,NULL,0.0)",
            rusqlite::params![indexed_path],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO fts(rowid,name,path,content,tokens) VALUES (1,'indexed.txt',?1,'money laundering typologies','')",
            rusqlite::params![indexed_path],
        )
        .unwrap();
}

async fn get(app: &axum::Router, uri: &str) -> axum::response::Response {
    app.clone()
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap()
}

async fn get_json(app: &axum::Router, uri: &str) -> Value {
    let response = get(app, uri).await;
    let bytes = to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// A GET that keeps the status code, for the error-path assertions.
async fn get_json_status(app: &axum::Router, uri: &str) -> (StatusCode, Value) {
    let response = get(app, uri).await;
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn corpus_tree_rejects_an_unknown_root() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    fs::create_dir_all(&input).unwrap();
    let app = router(ServiceConfig {
        output_root: temp.path().join("output"),
        allowed_roots: vec![input.clone()],
        default_paths: vec![input],
        config_path: None,
        ocr_langs: "vie+eng".into(),
        workers: 1,
        max_pending: 2,
        max_body: 1024 * 1024,
        vision_max: VisionMode::Off,
    })
    .unwrap();

    let response = get(&app, "/corpus/tree?root=nope").await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn corpus_surface_degrades_to_empty_before_any_index_job() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    fs::create_dir_all(input.join("folder")).unwrap();
    fs::write(
        input.join("folder/indexed.txt"),
        "money laundering typologies",
    )
    .unwrap();
    let input = input.canonicalize().unwrap();
    let app = router(ServiceConfig {
        output_root: temp.path().join("output"),
        allowed_roots: vec![input.clone()],
        default_paths: vec![input],
        config_path: None,
        ocr_langs: "vie+eng".into(),
        workers: 1,
        max_pending: 2,
        max_body: 1024 * 1024,
        vision_max: VisionMode::Off,
    })
    .unwrap();

    // No corpus.sqlite published yet: the tree still walks the live root, but
    // every entry carries no document fields.
    let entries = get_json(&app, "/corpus/tree?root=input").await;
    let file = entries
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == "indexed.txt")
        .unwrap();
    assert_eq!(file["kind"], "file");
    assert!(file.get("document_id").is_none());

    let status = get_json(&app, "/corpus/status").await;
    assert_eq!(status["indexed_files"], 0);
    assert_eq!(status["total_characters"], 0);

    let missing_text = get(&app, "/corpus/documents/1/text").await;
    assert_eq!(missing_text.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn corpus_tree_status_and_document_text_join_the_published_database() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    fs::create_dir_all(input.join("folder")).unwrap();
    fs::write(
        input.join("folder/indexed.txt"),
        "money laundering typologies",
    )
    .unwrap();
    fs::write(input.join("pending.txt"), "not yet indexed").unwrap();
    let input = input.canonicalize().unwrap();
    let app = router(ServiceConfig {
        output_root: output.clone(),
        allowed_roots: vec![input.clone()],
        default_paths: vec![input.clone()],
        config_path: None,
        ocr_langs: "vie+eng".into(),
        workers: 1,
        max_pending: 2,
        max_body: 1024 * 1024,
        vision_max: VisionMode::Off,
    })
    .unwrap();

    let indexed_path = input.join("folder/indexed.txt");
    write_fixture_corpus(&output, indexed_path.to_str().unwrap());

    let entries = get_json(&app, "/corpus/tree?root=input").await;
    let entries = entries.as_array().unwrap();

    let folder = entries.iter().find(|e| e["name"] == "folder").unwrap();
    assert_eq!(folder["kind"], "dir");
    assert_eq!(folder["depth"], 0);
    assert!(folder.get("document_id").is_none());

    let indexed = entries.iter().find(|e| e["name"] == "indexed.txt").unwrap();
    assert_eq!(indexed["kind"], "file");
    assert_eq!(indexed["depth"], 1);
    assert_eq!(indexed["path"], "folder/indexed.txt");
    assert_eq!(indexed["document_id"], 1);
    assert_eq!(indexed["character_count"], 900);
    assert_eq!(indexed["method"], "text");
    assert_eq!(indexed["lang"], "en");
    assert!(indexed["snippet"].as_str().unwrap().contains("laundering"));

    let pending = entries.iter().find(|e| e["name"] == "pending.txt").unwrap();
    assert_eq!(pending["kind"], "file");
    assert!(pending.get("document_id").is_none());

    let status = get_json(&app, "/corpus/status").await;
    assert_eq!(status["indexed_files"], 1);
    assert_eq!(status["total_characters"], 900);
    assert_eq!(status["languages"], json!([["en", 1]]));

    let text = get(&app, "/corpus/documents/1/text").await;
    assert_eq!(text.status(), StatusCode::OK);
    assert_eq!(
        text.headers().get("content-type").unwrap(),
        "text/plain; charset=utf-8"
    );
    let bytes = to_bytes(text.into_body(), 1024 * 1024).await.unwrap();
    assert_eq!(bytes, "money laundering typologies".as_bytes());

    let missing = get(&app, "/corpus/documents/999/text").await;
    assert_eq!(missing.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn corpus_status_reports_pending_files_from_meta_not_a_tree_walk() {
    // The fix this proves: a consumer used to learn "pending" by re-fetching
    // `/corpus/tree` (a full sorted `fs::read_dir` walk plus a snippet-carrying
    // join) a second time just to count null `document_id`s. `pending_files`
    // must come back from `/corpus/status` alone, derived from the pipeline's
    // own `last_discovered_files` meta key against a plain `COUNT(*)`.
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    fs::create_dir_all(&input).unwrap();
    let app = router(ServiceConfig {
        output_root: output.clone(),
        allowed_roots: vec![input.clone()],
        default_paths: vec![input.clone()],
        config_path: None,
        ocr_langs: "vie+eng".into(),
        workers: 1,
        max_pending: 2,
        max_body: 1024 * 1024,
        vision_max: VisionMode::Off,
    })
    .unwrap();

    write_fixture_corpus(&output, input.join("indexed.txt").to_str().unwrap());
    let connection = rusqlite::Connection::open(output.join("corpus.sqlite")).unwrap();
    connection
        .execute_batch("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);")
        .unwrap();
    connection
        .execute(
            "INSERT INTO meta(key,value) VALUES('last_discovered_files','3')",
            [],
        )
        .unwrap();
    drop(connection);

    let status = get_json(&app, "/corpus/status").await;
    assert_eq!(status["indexed_files"], 1, "{status}");
    assert_eq!(
        status["pending_files"], 2,
        "3 last discovered minus 1 indexed = 2 still pending: {status}"
    );
    // The fixture never ran through the migration harness, so its raw
    // `user_version` is 0 — distinct from what this binary would write. That
    // gap IS the version-skew signal a consumer's health banner reads.
    assert_eq!(status["schema_version"], 0, "{status}");
    // The fixture's `write_fixture_corpus` never creates a `chunks` table
    // either (it predates the migration harness) — that must read as "0
    // embedded" rather than fail the whole status read.
    assert_eq!(status["embedded_chunks"], 0, "{status}");
}

#[tokio::test]
async fn corpus_status_reports_embedded_chunks_when_the_table_exists() {
    // The fix this proves: a consumer (da-academic's Search tab semantic
    // diagnostic) reads `embedded_chunks` off `/corpus/status` to tell "no
    // chunks embedded yet" apart from "this build doesn't report the count at
    // all". Before this, the field never appeared in the response no matter
    // how many rows `chunks` held.
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    fs::create_dir_all(&input).unwrap();
    let app = router(ServiceConfig {
        output_root: output.clone(),
        allowed_roots: vec![input.clone()],
        default_paths: vec![input.clone()],
        config_path: None,
        ocr_langs: "vie+eng".into(),
        workers: 1,
        max_pending: 2,
        max_body: 1024 * 1024,
        vision_max: VisionMode::Off,
    })
    .unwrap();

    write_fixture_corpus(&output, input.join("indexed.txt").to_str().unwrap());
    let connection = rusqlite::Connection::open(output.join("corpus.sqlite")).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE chunks(
                id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL, embedding BLOB NOT NULL, dimensions INTEGER NOT NULL,
                model TEXT NOT NULL
             );",
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO chunks(id,file_id,chunk_index,content,embedding,dimensions,model)
             VALUES (1,1,0,'chunk one',x'00',4,'test-model'),
                    (2,1,1,'chunk two',x'00',4,'test-model')",
            [],
        )
        .unwrap();
    drop(connection);

    let status = get_json(&app, "/corpus/status").await;
    assert_eq!(status["embedded_chunks"], 2, "{status}");
}

#[tokio::test]
async fn corpus_status_pending_files_never_goes_negative() {
    // A corpus that has indexed MORE than the last recorded discovery (a
    // second job covering more paths, or files added between calls) must not
    // report a negative "still pending" count.
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    let output = temp.path().join("output");
    fs::create_dir_all(&input).unwrap();
    let app = router(ServiceConfig {
        output_root: output.clone(),
        allowed_roots: vec![input.clone()],
        default_paths: vec![input.clone()],
        config_path: None,
        ocr_langs: "vie+eng".into(),
        workers: 1,
        max_pending: 2,
        max_body: 1024 * 1024,
        vision_max: VisionMode::Off,
    })
    .unwrap();

    write_fixture_corpus(&output, input.join("indexed.txt").to_str().unwrap());
    let connection = rusqlite::Connection::open(output.join("corpus.sqlite")).unwrap();
    connection
        .execute_batch("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);")
        .unwrap();
    connection
        .execute(
            "INSERT INTO meta(key,value) VALUES('last_discovered_files','0')",
            [],
        )
        .unwrap();
    drop(connection);

    let status = get_json(&app, "/corpus/status").await;
    assert_eq!(status["indexed_files"], 1, "{status}");
    assert_eq!(status["pending_files"], 0, "{status}");
}

// ── Vision submit validation (--vision-max cap, unknown tier, missing models) ──
//
// All three reject at submit before the job is queued, so they never touch the
// (network-fetched) embedding model.

fn vision_router(vision_max: VisionMode) -> axum::Router {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    fs::create_dir_all(&input).unwrap();
    router(ServiceConfig {
        output_root: temp.path().join("output"),
        allowed_roots: vec![input.clone()],
        default_paths: vec![input],
        config_path: None,
        ocr_langs: "vie+eng".into(),
        workers: 1,
        max_pending: 2,
        max_body: 1024 * 1024,
        vision_max,
    })
    .unwrap()
}

async fn submit_vision(app: &axum::Router, id: &str, vision: &str) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/index")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"id": id, "ocr": "off", "vision": vision}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn submit_rejects_an_unknown_vision_tier() {
    let app = vision_router(VisionMode::Captions);
    let (status, body) = submit_vision(&app, "v-unknown", "blurry").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("unknown vision tier"));
}

#[tokio::test]
async fn submit_rejects_a_tier_above_the_server_cap() {
    // Default cap is off, so even `meta` is refused.
    let app = vision_router(VisionMode::Off);
    let (status, body) = submit_vision(&app, "v-capped", "tags").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].as_str().unwrap().contains("maximum"));
}

#[tokio::test]
async fn submit_rejects_when_vision_models_are_missing() {
    // Cap allows tags, but no model files exist under the default data dir.
    let app = vision_router(VisionMode::Tags);
    let (status, body) = submit_vision(&app, "v-nomodels", "tags").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("fetch-data --vision"));
}

#[tokio::test]
async fn submit_accepts_off_vision_without_models() {
    // `off` (and an omitted field) must never trip vision validation.
    let app = vision_router(VisionMode::Off);
    let (status, _) = submit_vision(&app, "v-off", "off").await;
    assert_eq!(status, StatusCode::ACCEPTED);
}

// ── Per-job settings validation (ocr_opts / vision_opts) rejected at submit ──
//
// Each rejects with a field-specific 400 BEFORE the job is queued, so they
// never load the (network-fetched) embedding model.

async fn submit_body(app: &axum::Router, body: Value) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/index")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn submit_rejects_out_of_range_ocr_dpi() {
    let app = vision_router(VisionMode::Off);
    let (status, body) = submit_body(
        &app,
        json!({"id":"dpi-bad","ocr":"off","ocr_opts":{"dpi":5000}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body["error"].as_str().unwrap().contains("ocr.dpi"),
        "{body}"
    );
}

#[tokio::test]
async fn submit_rejects_uninstalled_ocr_language() {
    // "zzz" is not a real tesseract pack, so it is absent from the bundled and
    // system tessdata alike regardless of what the box has installed.
    let app = vision_router(VisionMode::Off);
    let (status, body) = submit_body(
        &app,
        json!({"id":"lang-bad","ocr":"off","ocr_opts":{"langs":"zzz"}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].as_str().unwrap().contains("zzz"), "{body}");
    assert!(
        body["error"].as_str().unwrap().contains("ocr.langs"),
        "{body}"
    );
}

#[tokio::test]
async fn submit_rejects_out_of_range_vision_setting() {
    // vision_opts is validated even with the tier off/omitted.
    let app = vision_router(VisionMode::Off);
    let (status, body) = submit_body(
        &app,
        json!({"id":"vk-bad","ocr":"off","vision_opts":{"tag_top_k":99}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body["error"].as_str().unwrap().contains("tag_top_k"),
        "{body}"
    );
}

#[tokio::test]
async fn settings_route_serves_the_capability_contract() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    fs::create_dir_all(&input).unwrap();
    let app = router(ServiceConfig {
        output_root: temp.path().join("output"),
        allowed_roots: vec![input.clone()],
        default_paths: vec![input],
        config_path: None,
        ocr_langs: "vie+eng".into(),
        workers: 7,
        max_pending: 2,
        max_body: 1024 * 1024,
        // A high cap; with no models staged the endpoint still only offers `meta`.
        vision_max: VisionMode::Tags,
    })
    .unwrap();

    let response = get(&app, "/settings").await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let settings: Value = serde_json::from_slice(&body).unwrap();

    // The process advertises its real worker default/ceiling and OCR modes.
    assert_eq!(settings["workers"]["default"], 7);
    assert_eq!(settings["workers"]["max"], 64);
    assert_eq!(
        settings["ocr"]["modes"],
        json!(["auto", "on", "off", "exhaustive"])
    );
    assert!(settings["ocr"]["langs_installed"].is_array());
    // Vision: the cap is echoed, and the tags tier is gated OUT because its
    // models were never staged — only pure-code `meta` is available.
    assert_eq!(settings["vision"]["max_tier"], "tags");
    assert_eq!(settings["vision"]["tiers_available"], json!(["meta"]));
    assert_eq!(settings["vision"]["detectors"][0]["id"], "nano");
    assert_eq!(settings["vision"]["detectors"][0]["present"], false);
}

// ── Live stage tuning (GET/POST /runtime, POST /jobs/{id}/runtime) ───────────

fn runtime_app(input: &std::path::Path, output: &std::path::Path) -> axum::Router {
    router(ServiceConfig {
        output_root: output.to_path_buf(),
        allowed_roots: vec![input.to_path_buf()],
        default_paths: vec![input.to_path_buf()],
        config_path: None,
        ocr_langs: "vie+eng".into(),
        workers: 1,
        max_pending: 4,
        max_body: 1024 * 1024,
        vision_max: VisionMode::Off,
    })
    .unwrap()
}

async fn post_json(app: &axum::Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

/// Poll until the job is genuinely in flight. `run_index` loads the ~448 MB
/// embedding model while the job already reports `running`, so this is a wide
/// window rather than a race.
async fn wait_until_running(app: &axum::Router, id: &str) {
    for _ in 0..1500 {
        let job = get_json(app, &format!("/jobs/{id}")).await;
        match job["status"].as_str() {
            Some("running") => return,
            Some("complete" | "error" | "cancelled") => {
                panic!("job {id} finished before it could be retuned: {job}")
            }
            _ => tokio::time::sleep(Duration::from_millis(10)).await,
        }
    }
    panic!("job {id} never started running")
}

fn sample_corpus(input: &std::path::Path, count: usize) {
    fs::create_dir_all(input).unwrap();
    for index in 0..count {
        fs::write(
            input.join(format!("report_{index}.txt")),
            format!("Suspicious activity report {index} for the compliance team."),
        )
        .unwrap();
    }
}

#[tokio::test]
async fn runtime_reports_every_stage_and_never_overclaims_liveness() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    fs::create_dir_all(&input).unwrap();
    let app = runtime_app(&input, &temp.path().join("output"));

    let view = get_json(&app, "/runtime").await;
    let stages = &view["stages"];
    // The three canonical names the contract shares across engines.
    for stage in ["extract", "embed", "ocr"] {
        assert!(stages[stage].is_object(), "missing stage {stage}: {view}");
        for field in ["value", "min", "max", "live", "unit"] {
            assert!(
                stages[stage].get(field).is_some(),
                "stage {stage} is missing {field}: {view}"
            );
        }
        assert!(stages[stage]["value"].is_i64(), "{stage} value must be int");
    }
    // `extract` seeds from `serve --workers`.
    assert_eq!(stages["extract"]["value"], json!(1));

    // The liveness flag is the contract's core promise, so it is asserted rather
    // than assumed: a stage that says `live: true` must not also name a boundary,
    // and one that says `false` MUST name the boundary it applies at.
    //
    // `ocr` is false: OMP_THREAD_LIMIT is resolved when tesseract is spawned,
    // once per file, so a change cannot reach the scan already being recognised.
    for (stage, live) in [("extract", true), ("embed", true), ("ocr", false)] {
        assert_eq!(stages[stage]["live"], json!(live), "{stage}");
        assert_eq!(
            stages[stage].get("applies").is_some(),
            !live,
            "{stage}: `applies` must be present exactly when live is false"
        );
    }
    assert_eq!(stages["ocr"]["applies"], json!("next-file"));

    // Exactly the shared llm set — no engine-local extras. The app merges every
    // stage reported here into its Settings UI but validates writes against its
    // own copy of this list, so a name only this engine knows renders as a
    // control whose save 400s, taking the rest of the caller's body down with it.
    let mut names: Vec<&str> = stages
        .as_object()
        .expect("stages object")
        .keys()
        .map(String::as_str)
        .collect();
    names.sort_unstable();
    assert_eq!(names, ["embed", "extract", "ocr"], "{view}");
}

#[tokio::test]
async fn posting_runtime_clamps_out_of_range_values_and_reports_the_clamp() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    fs::create_dir_all(&input).unwrap();
    let app = runtime_app(&input, &temp.path().join("output"));

    let max = get_json(&app, "/runtime").await["stages"]["extract"]["max"].clone();
    let (status, body) = post_json(&app, "/runtime", json!({"extract": 100_000})).await;
    assert_eq!(status, StatusCode::OK);
    // Clamped, NOT rejected — and the response says what actually landed, so the
    // caller can see it did not get what it asked for.
    assert_eq!(body["stages"]["extract"]["value"], max);
    assert_eq!(
        get_json(&app, "/runtime").await["stages"]["extract"]["value"],
        max,
        "the clamped value must persist as the process-wide default"
    );
}

#[tokio::test]
async fn posting_an_unknown_stage_is_rejected_and_lists_the_valid_names() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    fs::create_dir_all(&input).unwrap();
    let app = runtime_app(&input, &temp.path().join("output"));

    // `gpu_layers` is a REAL stage — on the other engine. Sending it here must
    // not silently succeed.
    let (status, body) = post_json(&app, "/runtime", json!({"gpu_layers": 8})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let error = body["error"].as_str().unwrap_or_default();
    for name in ["extract", "embed", "ocr"] {
        assert!(error.contains(name), "400 must list valid names: {error}");
    }
}

#[tokio::test]
async fn per_job_runtime_rejects_unknown_and_terminal_jobs() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    fs::create_dir_all(&input).unwrap();
    fs::write(input.join("a.txt"), "compliance report").unwrap();
    let app = runtime_app(&input, &temp.path().join("output"));

    let (status, _) = post_json(&app, "/jobs/nope/runtime", json!({"extract": 2})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, _) = post_json(
        &app,
        "/index",
        json!({"id":"done","paths":[input],"output":"corpus.sqlite","ocr":"off"}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let job = wait_for_job(&app, "done").await;
    assert_eq!(job["status"], "complete", "{job}");

    // Terminal is a 409, distinct from the 404 above: "you cannot retune this"
    // must not be indistinguishable from "no such job".
    let (status, _) = post_json(&app, "/jobs/done/runtime", json!({"extract": 2})).await;
    assert_eq!(status, StatusCode::CONFLICT);
}

#[tokio::test]
async fn retuning_a_running_job_hits_that_job_and_leaves_the_defaults_alone() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    sample_corpus(&input, 40);
    let app = runtime_app(&input, &temp.path().join("output"));

    let (status, _) = post_json(
        &app,
        "/index",
        json!({"id":"live","paths":[input],"output":"corpus.sqlite","ocr":"off"}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    wait_until_running(&app, "live").await;

    let (status, body) = post_json(&app, "/jobs/live/runtime", json!({"extract": 4})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["stages"]["extract"]["value"], json!(4));

    // The point of the per-job route: it reaches THAT job, not the process-wide
    // defaults. A caller retuning one job must not silently reconfigure the box.
    assert_eq!(
        get_json(&app, "/runtime").await["stages"]["extract"]["value"],
        json!(1),
        "a per-job retune must not leak into the process-wide defaults"
    );

    // GET /jobs/{id}/runtime reads back the job's OWN live value (4), while the
    // process-wide GET still reports the default (1). Without this readback a
    // caller had no way to confirm a retune stuck — every later read showed the
    // default, which is indistinguishable from the setting being lost.
    let per_job = get_json(&app, "/jobs/live/runtime").await;
    assert_eq!(
        per_job["stages"]["extract"]["value"],
        json!(4),
        "the per-job GET must report the job's retuned value, not the default"
    );

    let job = wait_for_job(&app, "live").await;
    assert_eq!(job["status"], "complete", "{job}");
    assert_eq!(job["files"], json!(40), "retuning must not lose files");
}

/// GET on a job that never existed is 404; on a finished one it is 409 (its
/// per-job knobs are reaped), never a 404 that would read as "never ran".
#[tokio::test]
async fn per_job_runtime_get_rejects_unknown_and_terminal_jobs() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    sample_corpus(&input, 4);
    let app = runtime_app(&input, &temp.path().join("output"));

    let (status, _) = get_json_status(&app, "/jobs/nope/runtime").await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, _) = post_json(
        &app,
        "/index",
        json!({"id":"done","paths":[input],"output":"corpus.sqlite","ocr":"off"}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let job = wait_for_job(&app, "done").await;
    assert_eq!(job["status"], "complete", "{job}");
    let (status, _) = get_json_status(&app, "/jobs/done/runtime").await;
    assert_eq!(status, StatusCode::CONFLICT);
}

#[tokio::test]
async fn process_defaults_do_not_retune_jobs_already_in_flight() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    sample_corpus(&input, 40);
    let app = runtime_app(&input, &temp.path().join("output"));

    let (status, _) = post_json(
        &app,
        "/index",
        json!({"id":"isolated","paths":[input],"output":"corpus.sqlite","ocr":"off"}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    wait_until_running(&app, "isolated").await;

    let (status, _) = post_json(&app, "/runtime", json!({"extract": 9})).await;
    assert_eq!(status, StatusCode::OK);
    // An empty body applies nothing and simply reports this job's own view.
    let (status, body) = post_json(&app, "/jobs/isolated/runtime", json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body["stages"]["extract"]["value"],
        json!(1),
        "POST /runtime sets defaults for FUTURE jobs, not work in flight"
    );

    let job = wait_for_job(&app, "isolated").await;
    assert_eq!(job["status"], "complete", "{job}");
}

#[tokio::test]
async fn an_explicit_per_job_workers_seeds_that_jobs_extract_stage() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input");
    sample_corpus(&input, 40);
    let app = runtime_app(&input, &temp.path().join("output"));

    // `workers` on the submit body is the caller stating this job's extract
    // width, so it must outrank the process-wide default it snapshotted from.
    let (status, _) = post_json(
        &app,
        "/index",
        json!({"id":"seeded","paths":[input],"output":"corpus.sqlite","ocr":"off","workers":5}),
    )
    .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    wait_until_running(&app, "seeded").await;

    let (status, body) = post_json(&app, "/jobs/seeded/runtime", json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["stages"]["extract"]["value"], json!(5));
    assert_eq!(
        get_json(&app, "/runtime").await["stages"]["extract"]["value"],
        json!(1),
        "a per-job workers request must not move the process-wide default"
    );

    let job = wait_for_job(&app, "seeded").await;
    assert_eq!(job["status"], "complete", "{job}");
}
