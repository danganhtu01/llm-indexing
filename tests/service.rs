use std::fs;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use llm_indexing::service::{router, ServiceConfig};
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
    for _ in 0..100 {
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
