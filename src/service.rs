use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path as AxumPath, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rusqlite::{params, Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tempfile::Builder;
use tokio::sync::{mpsc, RwLock};
use tower_http::trace::TraceLayer;
use uuid::Uuid;

use crate::config::Config;
use crate::pipeline::{run_index, IndexRequest};
use crate::store::grouped;
use crate::vision::{corrupt_models, missing_vision_prereqs, VisionMode};
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
    /// Highest vision tier this server will accept (`serve --vision-max`,
    /// default `off`); requests above it are rejected at submit.
    pub vision_max: VisionMode,
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
    /// Requested vision tier (`off`|`meta`|`tags`|`captions`); `None` means
    /// `off`. Validated at submit against the server's `--vision-max` cap.
    #[serde(default)]
    pub vision: Option<String>,
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
    output_root: PathBuf,
    /// Allowed input roots keyed by their directory name (e.g. `/input` ->
    /// `"input"`), the `root` query param accepted by `/corpus/tree`.
    roots: Arc<HashMap<String, PathBuf>>,
    /// Highest vision tier accepted at submit.
    vision_max: VisionMode,
    /// Config source used to resolve vision model paths for the submit
    /// pre-flight.
    config_path: Option<PathBuf>,
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
    let mut roots = HashMap::with_capacity(normalized.allowed_roots.len());
    for root in &normalized.allowed_roots {
        let name = root_name(root);
        if roots.insert(name.clone(), root.clone()).is_some() {
            anyhow::bail!("allowed roots must have unique directory names (duplicate: {name})")
        }
    }
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
        output_root: normalized.output_root.clone(),
        roots: Arc::new(roots),
        vision_max: normalized.vision_max,
        config_path: normalized.config_path.clone(),
    };
    Ok(Router::new()
        .route("/health", get(health))
        .route("/index", post(submit))
        .route("/jobs/{id}", get(job))
        .route("/jobs/{id}/cancel", post(cancel_job))
        .route("/corpus/tree", get(corpus_tree))
        .route("/corpus/documents/{id}/text", get(corpus_document_text))
        .route("/corpus/status", get(corpus_status_handler))
        .layer(DefaultBodyLimit::max(max_body))
        .layer(TraceLayer::new_for_http())
        .with_state(state))
}

/// The `root` query-param name for an allowed input root: its directory name
/// (`/input` -> `"input"`), or the full path string for the rare case of a
/// nameless root (e.g. `/`).
fn root_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string())
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

/// Validate a job's requested vision tier against the server cap and, when the
/// tier needs models, the on-disk model files. Returns the resolved tier or a
/// small error tuple (kept out of `Response`, which `clippy::result_large_err`
/// flags) that the caller turns into a job-level `400`.
fn validate_vision(
    state: &AppState,
    requested: Option<&str>,
) -> Result<VisionMode, (StatusCode, Json<Value>)> {
    let mode = requested
        .unwrap_or("off")
        .parse::<VisionMode>()
        .map_err(|error| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({"status":"error","error": error})),
            )
        })?;
    if mode > state.vision_max {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({
                "status":"error",
                "error": format!(
                    "vision tier '{}' exceeds this server's maximum '{}'",
                    mode, state.vision_max
                )
            })),
        ));
    }
    if mode.needs_models() {
        let config = Config::load(state.config_path.as_deref()).map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"status":"error","error": format!("loading config: {error:#}")})),
            )
        })?;
        if !missing_vision_prereqs(&config.vision_models_dir(), mode).is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "status":"error",
                    "error":"vision models missing; run llm-index fetch-data --vision"
                })),
            ));
        }
    }
    Ok(mode)
}

async fn submit(State(state): State<AppState>, Json(mut request): Json<JobRequest>) -> Response {
    if let Err((status, body)) = validate_vision(&state, request.vision.as_deref()) {
        return (status, body).into_response();
    }
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
    if !valid_output_name(&request.output) {
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
    // Resolve the vision tier, clamped to the server cap as defence in depth
    // (submit already validated it against the same cap and model presence).
    let requested_vision: VisionMode = request
        .vision
        .as_deref()
        .unwrap_or("off")
        .parse()
        .unwrap_or(VisionMode::Off);
    config.vision.max = requested_vision.min(service.vision_max);
    if config.vision.max.needs_models() {
        let models_dir = config.vision_models_dir();
        if !missing_vision_prereqs(&models_dir, config.vision.max).is_empty() {
            anyhow::bail!("vision models missing; run llm-index fetch-data --vision")
        }
        // Integrity gate — runs on this blocking worker thread (never the async
        // submit path), so hashing the ~100 MB detector is safe. A present but
        // corrupt/tampered pinned model fails the job as a whole, before any file
        // is processed, rather than surfacing as per-file errors mid-run.
        let corrupt = corrupt_models(&models_dir, config.vision.max);
        if !corrupt.is_empty() {
            anyhow::bail!(
                "vision model integrity check failed (corrupt/truncated/tampered); \
                 re-run llm-index fetch-data --vision --force: {corrupt:?}"
            )
        }
    }
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
        "vision_files":stats.vision_files,"vision":config.vision.max.as_str(),
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

/// A published corpus database must be a plain filename (no directories, no
/// traversal) ending in `.sqlite`, confining every job's output under
/// `output_root`.
fn valid_output_name(name: &str) -> bool {
    Path::new(name).file_name().and_then(|n| n.to_str()) == Some(name) && name.ends_with(".sqlite")
}

// ── Corpus read surface (GET /corpus/tree, /corpus/documents/{id}/text, /corpus/status) ──
//
// READ-ONLY over whatever `corpus.sqlite` the most recent job published.
// Consumer apps used to open the SQLite file directly; this surface lets them
// stop decoding the schema themselves. The database is absent until the first
// job completes — every route below degrades to an empty/zeroed result rather
// than an error.

#[derive(Debug, Deserialize)]
struct TreeQuery {
    root: String,
    #[serde(default)]
    output: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct OutputQuery {
    #[serde(default)]
    output: Option<String>,
}

/// Resolve the `?output=` query param (default `corpus.sqlite`) to a path
/// under `output_root`, rejecting anything that is not a confined plain
/// filename ending in `.sqlite`. The error is small (kept out of `Response`,
/// which `clippy::result_large_err` flags) and converts at each call site.
fn resolve_output(
    state: &AppState,
    requested: Option<&str>,
) -> Result<PathBuf, (StatusCode, Json<Value>)> {
    let name = requested.unwrap_or("corpus.sqlite");
    if !valid_output_name(name) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error":"output must be a plain filename ending in .sqlite"})),
        ));
    }
    Ok(state.output_root.join(name))
}

/// Open a corpus database read-only; `None` when it is absent or unreadable.
fn open_ro(path: &Path) -> Option<Connection> {
    if !path.exists() {
        return None;
    }
    Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).ok()
}

struct DocMeta {
    id: i64,
    character_count: i64,
    method: String,
    lang: String,
    snippet: String,
}

/// One row per indexed file, keyed by its absolute path (`files.path`), which
/// is exactly how the tree walk below reconstructs each entry's path — an
/// exact join, unlike a by-name join that can collide across directories.
fn corpus_index(path: &Path) -> HashMap<String, DocMeta> {
    let Some(connection) = open_ro(path) else {
        return HashMap::new();
    };
    let Ok(mut statement) = connection.prepare(
        "SELECT f.path, f.id, COALESCE(f.chars,0), COALESCE(f.method,''), COALESCE(f.lang,''), \
                COALESCE(substr(fts.content,1,400),'') \
         FROM files f JOIN fts ON fts.rowid = f.id",
    ) else {
        return HashMap::new();
    };
    let Ok(rows) = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            DocMeta {
                id: row.get(1)?,
                character_count: row.get(2)?,
                method: row.get(3)?,
                lang: row.get(4)?,
                snippet: row.get(5)?,
            },
        ))
    }) else {
        return HashMap::new();
    };
    rows.flatten().collect()
}

/// GET /corpus/tree?root=NAME[&output=corpus.sqlite]
///
/// A sorted recursive walk of one allowed input root, joined by absolute path
/// against the published corpus database. `root` must name one of the
/// service's configured allowed roots (its directory name); anything else is
/// `400`. A root that doesn't (yet) exist on disk walks to an empty array,
/// same as an absent corpus database.
async fn corpus_tree(State(state): State<AppState>, Query(query): Query<TreeQuery>) -> Response {
    let Some(root) = state.roots.get(&query.root).cloned() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":"unknown root"})),
        )
            .into_response();
    };
    let output = match resolve_output(&state, query.output.as_deref()) {
        Ok(path) => path,
        Err(response) => return response.into_response(),
    };
    let entries = tokio::task::spawn_blocking(move || tree_entries(&root, &output))
        .await
        .unwrap_or_default();
    Json(entries).into_response()
}

fn tree_entries(root: &Path, corpus_db: &Path) -> Vec<Value> {
    let index = corpus_index(corpus_db);
    let mut rows = Vec::new();
    if root.is_dir() {
        let _ = collect_tree(root, root, 0, &index, &mut rows);
    }
    rows
}

fn collect_tree(
    root: &Path,
    directory: &Path,
    depth: usize,
    index: &HashMap<String, DocMeta>,
    rows: &mut Vec<Value>,
) -> Result<()> {
    let mut children = fs::read_dir(directory)?
        .filter_map(Result::ok)
        .collect::<Vec<_>>();
    children.sort_by(|left, right| {
        let left_dir = left.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let right_dir = right.file_type().map(|t| t.is_dir()).unwrap_or(false);
        right_dir
            .cmp(&left_dir)
            .then_with(|| left.file_name().cmp(&right.file_name()))
    });
    for child in children {
        // Mirror the indexing walker's default: symlinks are never followed,
        // so the tree stays confined and matches what was actually indexed.
        if child.file_type().map(|t| t.is_symlink()).unwrap_or(true) {
            continue;
        }
        let path = child.path();
        let Ok(metadata) = fs::metadata(&path) else {
            continue;
        };
        let is_dir = metadata.is_dir();
        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let name = child.file_name().to_string_lossy().into_owned();
        let document = (!is_dir)
            .then(|| index.get(path.to_string_lossy().as_ref()))
            .flatten();
        let mut entry = json!({
            "path": relative,
            "name": name,
            "kind": if is_dir { "dir" } else { "file" },
            "depth": depth as i64,
            "size_bytes": if is_dir { 0 } else { metadata.len().min(i64::MAX as u64) as i64 },
            "modified_at": metadata
                .modified()
                .ok()
                .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
                .map(|value| value.as_secs() as i64)
                .unwrap_or_default(),
        });
        if let Some(document) = document {
            entry["document_id"] = json!(document.id);
            entry["character_count"] = json!(document.character_count);
            entry["method"] = json!(document.method);
            entry["lang"] = json!(document.lang);
            entry["snippet"] = json!(document.snippet);
        }
        rows.push(entry);
        if is_dir {
            collect_tree(root, &path, depth + 1, index, rows)?;
        }
    }
    Ok(())
}

/// GET /corpus/documents/{id}/text[?output=corpus.sqlite]
///
/// Streams the extracted text for one document as `text/plain`. `404` when
/// the corpus database is absent or holds no matching id.
async fn corpus_document_text(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<i64>,
    Query(query): Query<OutputQuery>,
) -> Response {
    let output = match resolve_output(&state, query.output.as_deref()) {
        Ok(path) => path,
        Err(response) => return response.into_response(),
    };
    let text = tokio::task::spawn_blocking(move || document_text(&output, id)).await;
    let Ok(Some(content)) = text else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error":"document not found"})),
        )
            .into_response();
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(Body::from(content))
        .unwrap_or_else(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error":"response error"})),
            )
                .into_response()
        })
}

fn document_text(corpus_db: &Path, id: i64) -> Option<String> {
    let connection = open_ro(corpus_db)?;
    connection
        .query_row(
            "SELECT fts.content FROM files f JOIN fts ON fts.rowid = f.id WHERE f.id = ?1",
            params![id],
            |row| row.get(0),
        )
        .ok()
}

/// GET /corpus/status[?output=corpus.sqlite]
///
/// Cheap corpus-wide aggregates: total indexed files/characters/bytes, OCR
/// count, and language/method breakdowns. Zeroed when the database is absent.
async fn corpus_status_handler(
    State(state): State<AppState>,
    Query(query): Query<OutputQuery>,
) -> Response {
    let output = match resolve_output(&state, query.output.as_deref()) {
        Ok(path) => path,
        Err(response) => return response.into_response(),
    };
    let value = tokio::task::spawn_blocking(move || corpus_status(&output))
        .await
        .unwrap_or_else(|_| empty_status());
    Json(value).into_response()
}

fn corpus_status(path: &Path) -> Value {
    let Some(connection) = open_ro(path) else {
        return empty_status();
    };
    let indexed_files: i64 = connection
        .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
        .unwrap_or(0);
    let total_characters: i64 = connection
        .query_row("SELECT COALESCE(SUM(chars),0) FROM files", [], |r| r.get(0))
        .unwrap_or(0);
    let total_bytes: i64 = connection
        .query_row("SELECT COALESCE(SUM(size),0) FROM files", [], |r| r.get(0))
        .unwrap_or(0);
    let ocr_files: i64 = connection
        .query_row("SELECT COALESCE(SUM(ocr_used),0) FROM files", [], |r| {
            r.get(0)
        })
        .unwrap_or(0);
    json!({
        "indexed_files": indexed_files,
        "total_characters": total_characters,
        "total_bytes": total_bytes,
        "ocr_files": ocr_files,
        "languages": grouped(&connection, "lang", 10).unwrap_or_default(),
        "methods": grouped(&connection, "method", 20).unwrap_or_default(),
    })
}

fn empty_status() -> Value {
    json!({
        "indexed_files": 0, "total_characters": 0, "total_bytes": 0, "ocr_files": 0,
        "languages": Vec::<(String, i64)>::new(), "methods": Vec::<(String, i64)>::new(),
    })
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
    use super::{requested_paths, root_name, valid_output_name};
    use std::path::Path;

    #[test]
    fn root_name_uses_the_final_path_component() {
        assert_eq!(root_name(Path::new("/input")), "input");
        assert_eq!(root_name(Path::new("/input/downloads")), "downloads");
    }

    #[test]
    fn root_name_falls_back_to_the_full_path_when_nameless() {
        assert_eq!(root_name(Path::new("/")), "/");
    }

    #[test]
    fn valid_output_name_rejects_paths_and_wrong_extension() {
        assert!(valid_output_name("corpus.sqlite"));
        assert!(!valid_output_name("../corpus.sqlite"));
        assert!(!valid_output_name("sub/corpus.sqlite"));
        assert!(!valid_output_name("corpus.db"));
        assert!(!valid_output_name("/etc/corpus.sqlite"));
    }

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
