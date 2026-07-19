use std::fs;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
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
    assert_eq!(fs::read_dir(&output).unwrap().count(), 1);
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
