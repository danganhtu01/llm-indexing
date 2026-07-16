use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use axum::extract::{DefaultBodyLimit, Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tempfile::Builder;
use tokio::sync::{mpsc, RwLock};
use tower_http::trace::TraceLayer;
use uuid::Uuid;

use crate::config::Config;
use crate::pipeline::{run_index, IndexRequest};
use crate::VERSION;

const MAX_HISTORY: usize = 1_000;

#[derive(Debug, Clone)]
pub struct ServiceConfig {
    pub output_root: PathBuf,
    pub allowed_roots: Vec<PathBuf>,
    pub default_paths: Vec<PathBuf>,
    pub config_path: Option<PathBuf>,
    pub ocr_langs: String,
    pub workers: usize,
    pub max_pending: usize,
    pub max_body: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRequest {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub paths: Option<Vec<PathBuf>>,
    #[serde(default = "default_output")]
    pub output: String,
    #[serde(default = "default_ocr")]
    pub ocr: String,
    #[serde(default)]
    pub ocr_langs: Option<String>,
    #[serde(default)]
    pub workers: Option<usize>,
    #[serde(default)]
    pub resume: bool,
    #[serde(default)]
    pub overwrite: bool,
    #[serde(default)]
    pub include_paths: Option<Vec<String>>,
}

fn default_output() -> String {
    "corpus.sqlite".into()
}
fn default_ocr() -> String {
    "auto".into()
}

#[derive(Clone)]
struct AppState {
    jobs: Arc<RwLock<HashMap<String, Value>>>,
    cancellations: Arc<RwLock<HashMap<String, Arc<AtomicBool>>>>,
    sender: mpsc::Sender<(String, JobRequest)>,
}

pub fn router(config: ServiceConfig) -> Result<Router> {
    fs::create_dir_all(&config.output_root)?;
    let mut normalized = config;
    normalized.output_root = normalized.output_root.canonicalize()?;
    normalized.allowed_roots = normalized
        .allowed_roots
        .iter()
        .map(|path| {
            path.canonicalize()
                .with_context(|| format!("allowed root {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let jobs = Arc::new(RwLock::new(HashMap::new()));
    let cancellations = Arc::new(RwLock::new(HashMap::new()));
    let (sender, receiver) = mpsc::channel(normalized.max_pending);
    let max_body = normalized.max_body;
    tokio::spawn(worker(
        receiver,
        jobs.clone(),
        cancellations.clone(),
        normalized.clone(),
    ));
    let state = AppState {
        jobs,
        cancellations,
        sender,
    };
    Ok(Router::new()
        .route("/health", get(health))
        .route("/index", post(submit))
        .route("/jobs/{id}", get(job))
        .route("/jobs/{id}/cancel", post(cancel_job))
        .layer(DefaultBodyLimit::max(max_body))
        .layer(TraceLayer::new_for_http())
        .with_state(state))
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    let busy = state.jobs.read().await.values().any(|job| {
        matches!(
            job["status"].as_str(),
            Some("queued" | "running" | "cancelling")
        )
    });
    Json(json!({"ok": true, "service": "llm-indexing", "version": VERSION, "busy": busy}))
}

async fn submit(State(state): State<AppState>, Json(mut request): Json<JobRequest>) -> Response {
    let id = request
        .id
        .take()
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    {
        let mut jobs = state.jobs.write().await;
        if jobs.contains_key(&id) {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"status":"error","error":"job id already exists"})),
            )
                .into_response();
        }
        prune_history(&mut jobs);
        if jobs.len() >= MAX_HISTORY {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(json!({"status":"error","error":"job history is full"})),
            )
                .into_response();
        }
        jobs.insert(
            id.clone(),
            json!({"id":id,"status":"queued","submitted_at":now()}),
        );
    }
    state
        .cancellations
        .write()
        .await
        .insert(id.clone(), Arc::new(AtomicBool::new(false)));
    request.id = Some(id.clone());
    match state.sender.try_send((id.clone(), request)) {
        Ok(()) => (
            StatusCode::ACCEPTED,
            Json(json!({"id":id,"status":"queued","submitted_at":now()})),
        )
            .into_response(),
        Err(_) => {
            state.jobs.write().await.remove(&id);
            state.cancellations.write().await.remove(&id);
            (
                StatusCode::TOO_MANY_REQUESTS,
                Json(json!({"status":"error","error":"indexing queue is full"})),
            )
                .into_response()
        }
    }
}

async fn job(State(state): State<AppState>, AxumPath(id): AxumPath<String>) -> Response {
    state
        .jobs
        .read()
        .await
        .get(&id)
        .cloned()
        .map(|value| Json(value).into_response())
        .unwrap_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(json!({"error":"job not found"})),
            )
                .into_response()
        })
}

async fn cancel_job(State(state): State<AppState>, AxumPath(id): AxumPath<String>) -> Response {
    let Some(cancellation) = state.cancellations.read().await.get(&id).cloned() else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error":"job not found"})),
        )
            .into_response();
    };
    let mut jobs = state.jobs.write().await;
    let Some(job) = jobs.get_mut(&id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error":"job not found"})),
        )
            .into_response();
    };
    match job["status"].as_str() {
        Some("queued" | "running" | "cancelling") => {
            cancellation.store(true, Ordering::Relaxed);
            job["status"] = json!("cancelling");
            job["message"] = json!("cancellation requested");
            (StatusCode::ACCEPTED, Json(job.clone())).into_response()
        }
        _ => (
            StatusCode::CONFLICT,
            Json(json!({"error":"job is not active"})),
        )
            .into_response(),
    }
}

async fn worker(
    mut receiver: mpsc::Receiver<(String, JobRequest)>,
    jobs: Arc<RwLock<HashMap<String, Value>>>,
    cancellations: Arc<RwLock<HashMap<String, Arc<AtomicBool>>>>,
    config: ServiceConfig,
) {
    while let Some((id, request)) = receiver.recv().await {
        let cancellation = cancellations
            .read()
            .await
            .get(&id)
            .cloned()
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        if cancellation.load(Ordering::Relaxed) {
            jobs.write().await.insert(
                id.clone(),
                json!({"id":id,"status":"cancelled","message":"cancelled before start","completed_at":now()}),
            );
            continue;
        }
        jobs.write().await.insert(
            id.clone(),
            json!({"id":id,"status":"running","processed":0,"total":0,"started_at":now()}),
        );
        let run_config = config.clone();
        let job_id = id.clone();
        let job_states = jobs.clone();
        let worker_cancellation = cancellation.clone();
        let result = tokio::task::spawn_blocking(move || {
            run_job(
                &job_id,
                request,
                &run_config,
                job_states,
                worker_cancellation,
            )
        })
        .await;
        let value = if cancellation.load(Ordering::Relaxed) {
            json!({"id":id,"status":"cancelled","message":"indexing cancelled; previous atomic corpus preserved","completed_at":now()})
        } else {
            match result {
                Ok(Ok(value)) => value,
                Ok(Err(error)) => {
                    json!({"id":id,"status":"error","error":format!("{error:#}"),"completed_at":now()})
                }
                Err(error) => {
                    json!({"id":id,"status":"error","error":format!("worker join: {error}"),"completed_at":now()})
                }
            }
        };
        jobs.write().await.insert(id, value);
    }
}

fn run_job(
    id: &str,
    request: JobRequest,
    service: &ServiceConfig,
    jobs: Arc<RwLock<HashMap<String, Value>>>,
    cancellation: Arc<AtomicBool>,
) -> Result<Value> {
    let paths = request
        .paths
        .unwrap_or_else(|| service.default_paths.clone());
    if paths.is_empty() {
        anyhow::bail!("paths must be a non-empty array of mounted directories")
    }
    let paths = paths
        .into_iter()
        .map(|path| {
            path.canonicalize()
                .with_context(|| format!("input path does not exist: {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    if paths.iter().any(|path| !path.is_dir()) {
        anyhow::bail!("every input path must be a directory")
    }
    if paths
        .iter()
        .any(|path| !service.allowed_roots.iter().any(|root| within(path, root)))
    {
        anyhow::bail!("input path is outside INDEX_ALLOWED_ROOTS")
    }
    let include_paths = requested_paths(&paths, request.include_paths)?;
    if Path::new(&request.output)
        .file_name()
        .and_then(|n| n.to_str())
        != Some(&request.output)
        || !request.output.ends_with(".sqlite")
    {
        anyhow::bail!("output must be a plain filename ending in .sqlite")
    }
    if !matches!(request.ocr.as_str(), "auto" | "on" | "off" | "exhaustive") {
        anyhow::bail!("ocr must be auto, on, off, or exhaustive")
    }
    let destination = service.output_root.join(&request.output);
    if destination.exists() && !request.resume && !request.overwrite {
        anyhow::bail!("output already exists; set resume or overwrite")
    }
    let work = Builder::new()
        .prefix(".indexing-")
        .tempdir_in(&service.output_root)?;
    if request.resume && destination.exists() {
        fs::copy(&destination, work.path().join("index.sqlite"))?;
    }
    let mut config = Config::load(service.config_path.as_deref())?;
    config.ocr = request.ocr;
    config.ocr_langs = request
        .ocr_langs
        .unwrap_or_else(|| service.ocr_langs.clone());
    config.workers = request.workers.unwrap_or(service.workers).clamp(1, 64);
    config.sidecar = "none".into();
    let progress_id = id.to_owned();
    let stats = run_index(IndexRequest {
        paths: &paths,
        out: work.path(),
        config: config.clone(),
        resume: request.resume,
        artifacts: false,
        include_paths,
        cancellation: Some(cancellation),
        progress: Some(Arc::new(move |processed, total| {
            let mut jobs = jobs.blocking_write();
            if let Some(job) = jobs.get_mut(&progress_id) {
                job["processed"] = json!(processed);
                job["total"] = json!(total);
            }
        })),
    })?;
    let database = work.path().join("index.sqlite");
    fs::rename(database, &destination)?;
    Ok(json!({
        "id":id,"status":"complete","database":destination,"files":stats.files,
        "ocr_files":stats.ocr_files,"errors":stats.errors,"skipped":stats.skipped,
        "incomplete":stats.incomplete,"embedded_chunks":stats.embedded_chunks,"removed":stats.removed,
        "elapsed_seconds":stats.elapsed_seconds,"ocr_langs":config.ocr_langs,"completed_at":now()
    }))
}

fn requested_paths(
    roots: &[PathBuf],
    requested: Option<Vec<String>>,
) -> Result<Option<HashSet<String>>> {
    let Some(requested) = requested else {
        return Ok(None);
    };
    let mut paths = HashSet::with_capacity(requested.len());
    for relative in requested {
        let relative_path = Path::new(&relative);
        if relative_path.is_absolute()
            || relative_path.components().any(|component| {
                matches!(
                    component,
                    std::path::Component::ParentDir
                        | std::path::Component::RootDir
                        | std::path::Component::Prefix(_)
                )
            })
        {
            anyhow::bail!("include path must be relative and confined: {relative}")
        }
        let source = roots
            .iter()
            .filter_map(|root| root.join(relative_path).canonicalize().ok())
            .find(|candidate| {
                candidate.is_file() && roots.iter().any(|root| within(candidate, root))
            })
            .with_context(|| format!("included source file does not exist: {relative}"))?;
        paths.insert(source.to_string_lossy().to_string());
    }
    Ok(Some(paths))
}

fn within(path: &Path, root: &Path) -> bool {
    path == root || path.starts_with(root)
}

fn prune_history(jobs: &mut HashMap<String, Value>) {
    if jobs.len() < MAX_HISTORY {
        return;
    }
    let mut finished = jobs
        .values()
        .filter(|job| {
            matches!(
                job["status"].as_str(),
                Some("complete" | "error" | "cancelled")
            )
        })
        .filter_map(|job| {
            Some((
                job["id"].as_str()?.to_string(),
                job["completed_at"].as_f64().unwrap_or(0.0),
            ))
        })
        .collect::<Vec<_>>();
    finished.sort_by(|a, b| a.1.total_cmp(&b.1));
    for (id, _) in finished.into_iter().take(jobs.len() - MAX_HISTORY + 1) {
        jobs.remove(&id);
    }
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::requested_paths;

    #[test]
    fn requested_paths_are_exact_and_confined() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("input");
        std::fs::create_dir_all(root.join("folder")).unwrap();
        std::fs::write(root.join("folder/changed.txt"), "changed").unwrap();
        std::fs::write(root.join("unchanged.txt"), "unchanged").unwrap();
        let root = root.canonicalize().unwrap();

        let selected = requested_paths(
            std::slice::from_ref(&root),
            Some(vec!["folder/changed.txt".into()]),
        )
        .unwrap()
        .unwrap();
        assert_eq!(selected.len(), 1);
        assert!(selected.contains(
            &root
                .join("folder/changed.txt")
                .canonicalize()
                .unwrap()
                .to_string_lossy()
                .to_string()
        ));
        assert!(requested_paths(&[root], Some(vec!["../escape.txt".into()])).is_err());
    }
}
