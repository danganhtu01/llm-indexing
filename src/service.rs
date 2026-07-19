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

use crate::config::{clamp_workers, Config, MAX_WORKERS};
use crate::pipeline::{run_index, IndexRequest};
use crate::settings::{
    installed_tessdata_langs, tessdata_sources, OcrSettings, VisionSettings, CAPTIONERS, DETECTORS,
    OCR_DPI_RANGE, OCR_MAX_PAGES_RANGE, OCR_PSM_RANGE, TAGGERS,
};
use crate::store::grouped;
use crate::vision::{
    available_tiers, captioner_present, corrupt_models, detector_present, missing_vision_prereqs,
    tagger_present, VisionMode,
};
use crate::VERSION;

const MAX_HISTORY: usize = 1_000;

/// Accepted `ocr` modes — the single definition backing both the submit-time
/// validation and the list `GET /settings` advertises.
const OCR_MODES: &[&str] = &["auto", "on", "off", "exhaustive"];

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
    /// Per-job OCR quality overrides (dpi/psm/preprocess/max_pages/langs),
    /// merged over the service config via the single settings path. Validated at
    /// submit → `400` naming the field. Absent ⇒ exactly today's behavior.
    #[serde(default)]
    pub ocr_opts: Option<OcrSettings>,
    /// Per-job vision overrides (detector/tagger/captioner + numeric knobs),
    /// active only when the requested tier != off and capped by `--vision-max`.
    #[serde(default)]
    pub vision_opts: Option<VisionSettings>,
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
    /// Default worker count this serve process runs jobs with; advertised by
    /// `GET /settings` as `workers.default`.
    workers: usize,
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
        workers: normalized.workers,
    };
    Ok(Router::new()
        .route("/health", get(health))
        .route("/settings", get(settings))
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

/// GET /settings — read-only capability discovery (SETTINGS-SPEC §2).
///
/// The contract the consumer apps (ff-lc-app / da-academic / drives-analytics)
/// render their OCR/vision settings UIs from, so no client hardcodes ranges,
/// installed languages, or which vision tiers this process can actually run.
/// Purely additive; touches no job state.
///
/// The probe reads the config file, enumerates the tessdata dir, execs
/// `tesseract --list-langs`, and hash-verifies the (up to ~100 MB) vision model
/// files — all blocking — so it runs on a blocking worker, never the async
/// executor.
async fn settings(State(state): State<AppState>) -> Response {
    let config_path = state.config_path.clone();
    let vision_max = state.vision_max;
    let workers = state.workers;
    match tokio::task::spawn_blocking(move || {
        build_settings(config_path.as_deref(), vision_max, workers)
    })
    .await
    {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(error)) => {
            // Log the full chain server-side but keep the client body generic — the
            // anyhow context embeds the absolute server-side config path.
            tracing::error!(error = %format!("{error:#}"), "building /settings response failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"status":"error","error":"loading settings"})),
            )
                .into_response()
        }
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"status":"error","error":"settings probe failed"})),
        )
            .into_response(),
    }
}

/// Build the `GET /settings` body. Ranges come from the single `settings.rs`
/// bound consts and defaults from the loaded [`Config`] (the same fields the W1
/// `OcrSettings`/`VisionSettings` bases read), so nothing here re-defines a knob.
fn build_settings(
    config_path: Option<&Path>,
    vision_max: VisionMode,
    workers: usize,
) -> Result<Value> {
    let config = Config::load(config_path)?;
    let models_dir = config.vision_models_dir();
    let langs_installed: Vec<String> = installed_tessdata_langs(&config).into_iter().collect();
    let psm_values: Vec<String> = (OCR_PSM_RANGE.0..=OCR_PSM_RANGE.1)
        .map(|value| value.to_string())
        .collect();
    // Only tiers within this process's `--vision-max` cap AND with their models
    // present/verified are offered.
    let tiers_available: Vec<&str> = available_tiers(&models_dir)
        .into_iter()
        .filter(|tier| *tier <= vision_max)
        .map(|tier| tier.as_str())
        .collect();
    Ok(json!({
        "version": VERSION,
        "ocr": {
            "modes": OCR_MODES,
            "langs_installed": langs_installed,
            "dpi": {"min": OCR_DPI_RANGE.0, "max": OCR_DPI_RANGE.1, "default": config.ocr_dpi},
            "psm": {"values": psm_values, "default": config.ocr_psm},
            "preprocess_default": config.ocr_preprocess,
            "max_pages": {
                "min": OCR_MAX_PAGES_RANGE.0, "max": OCR_MAX_PAGES_RANGE.1,
                "default": config.ocr_max_pages
            },
        },
        "vision": {
            "max_tier": vision_max.as_str(),
            "tiers_available": tiers_available,
            "detectors": sub_models(DETECTORS, detector_present(&models_dir)),
            "taggers": sub_models(TAGGERS, tagger_present(&models_dir)),
            "captioners": sub_models(CAPTIONERS, captioner_present(&models_dir)),
            "defaults": {
                "detector_conf": config.vision.detector_conf,
                "tag_threshold": config.vision.tag_score,
                "tag_top_k": config.vision.tag_top_k,
                "max_frames": config.vision.max_frames,
                "timeout_secs": config.vision.timeout_secs,
            },
        },
        // Route the advertised default through the SAME clamp `run_job` applies, so
        // /settings never reports a default outside its own `max` (or below 1).
        "workers": {"default": clamp_workers(workers), "max": MAX_WORKERS},
    }))
}

/// One `{"id","present"}` entry per selectable sub-model id (the accepted enum
/// values from `settings.rs` minus the `off` toggle), tagged with whether its
/// backing model files are staged. In v1 each category has a single model, so
/// they share one `present` flag.
fn sub_models(ids: &[&str], present: bool) -> Vec<Value> {
    ids.iter()
        .filter(|id| **id != "off")
        .map(|id| json!({"id": id, "present": present}))
        .collect()
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
        let config = Config::load(state.config_path.as_deref()).map_err(|_error| {
            // Generic body — the anyhow context embeds the server-side config path.
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"status":"error","error":"loading service configuration"})),
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

/// Cheap (no-I/O) per-field range/enum validation of a job's OCR/vision
/// overrides, using the same merge structs the pipeline later applies. Returns a
/// field-specific `400` (small tuple kept out of `Response`, which
/// `clippy::result_large_err` flags). The OCR language check is deliberately
/// NOT here: it reads the config file and execs `tesseract --list-langs`, so
/// submit runs it via [`validate_request_langs`] on a blocking worker rather than
/// blocking the async executor.
fn validate_job_fields(request: &JobRequest) -> Result<(), (StatusCode, Json<Value>)> {
    if let Some(ocr) = &request.ocr_opts {
        ocr.validate().map_err(bad_field)?;
    }
    if let Some(vision) = &request.vision_opts {
        vision.validate().map_err(bad_field)?;
    }
    Ok(())
}

/// The per-request OCR language selection actually in effect: `ocr_opts.langs`
/// wins over the legacy top-level `ocr_langs`, matching `run_job`'s precedence.
/// `None` ⇒ the client supplied no language, so the (trusted) service default is
/// used and there is nothing per-request to validate. Guarding on this closes the
/// bypass where the legacy `ocr_langs` alias reached tesseract unvalidated while
/// `ocr_opts.langs` was gated.
fn effective_request_langs(request: &JobRequest) -> Option<String> {
    request
        .ocr_opts
        .as_ref()
        .and_then(|ocr| ocr.langs.clone())
        .or_else(|| request.ocr_langs.clone())
}

/// Blocking: validate `langs` against the installed tessdata using the same
/// source-aware resolution `TesseractOcr` uses. Reads the config file and execs
/// `tesseract --list-langs`, so callers run it via `spawn_blocking`. On rejection
/// returns the HTTP status + message; a config-load failure is reported
/// generically, never echoing the server-side config path.
fn validate_request_langs(
    config_path: Option<PathBuf>,
    langs: String,
) -> Result<(), (StatusCode, String)> {
    let config = Config::load(config_path.as_deref()).map_err(|_error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "loading service configuration".to_string(),
        )
    })?;
    let (bundled, system) = tessdata_sources(&config);
    OcrSettings {
        langs: Some(langs),
        ..Default::default()
    }
    .validate_langs(&bundled, &system)
    .map_err(|message| (StatusCode::BAD_REQUEST, message))
}

/// A field-specific submit rejection, matching the `{"status":"error","error"}`
/// shape the other submit validations use.
fn bad_field(message: String) -> (StatusCode, Json<Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"status":"error","error": message})),
    )
}

async fn submit(State(state): State<AppState>, Json(mut request): Json<JobRequest>) -> Response {
    if let Err((status, body)) = validate_vision(&state, request.vision.as_deref()) {
        return (status, body).into_response();
    }
    if let Err((status, body)) = validate_job_fields(&request) {
        return (status, body).into_response();
    }
    // OCR language validation reads the config file and execs `tesseract
    // --list-langs`; run it on a blocking worker so a slow/stalled tesseract never
    // pins the async executor thread (the identical /settings probe does the same).
    if let Some(langs) = effective_request_langs(&request) {
        let config_path = state.config_path.clone();
        match tokio::task::spawn_blocking(move || validate_request_langs(config_path, langs)).await
        {
            Ok(Ok(())) => {}
            Ok(Err((status, message))) => {
                return (status, Json(json!({"status":"error","error": message}))).into_response();
            }
            Err(_) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"status":"error","error":"settings validation failed"})),
                )
                    .into_response();
            }
        }
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
    if !OCR_MODES.contains(&request.ocr.as_str()) {
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
    config.workers = clamp_workers(request.workers.unwrap_or(service.workers));
    config.sidecar = "none".into();
    // Per-job OCR quality knobs merged over the (built-in ⊕ YAML ⊕ legacy-langs)
    // base through the single settings path; submit already validated them. When
    // `ocr_opts` is absent this reproduces the config verbatim (off-path
    // unchanged). An `ocr_opts.langs` here wins over the legacy top-level
    // `ocr_langs` set just above (it is the merge base).
    OcrSettings::resolve(&config, request.ocr_opts.as_ref()).apply_to(&mut config);
    // Resolve the vision tier, clamped to the server cap as defence in depth
    // (submit already validated it against the same cap and model presence).
    let requested_vision: VisionMode = request
        .vision
        .as_deref()
        .unwrap_or("off")
        .parse()
        .unwrap_or(VisionMode::Off);
    config.vision.max = requested_vision.min(service.vision_max);
    // Per-job vision knobs (detector_conf/tag_threshold/tag_top_k/max_frames/
    // timeout_secs) merged over the config base; inert when the tier is off.
    VisionSettings::resolve(&config, request.vision_opts.as_ref()).apply_to(&mut config);
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
    use super::{build_settings, requested_paths, root_name, valid_output_name};
    use crate::config::{Config, MAX_WORKERS};
    use crate::settings::{
        OcrSettings, VisionSettings, OCR_DPI_RANGE, OCR_MAX_PAGES_RANGE, OCR_PSM_RANGE,
    };
    use crate::vision::VisionMode;
    use std::path::Path;

    /// Write a config file whose `data_dir` resolves to `dir`, so `build_settings`
    /// enumerates the fixture tessdata/vision trees created under it.
    fn config_pointing_at(dir: &Path) -> std::path::PathBuf {
        let path = dir.join("config.yaml");
        std::fs::write(&path, "data_dir: .\n").unwrap();
        path
    }

    #[test]
    fn settings_reports_the_spec_shape() {
        let value = build_settings(None, VisionMode::Off, 4).unwrap();
        // Top-level blocks.
        assert_eq!(value["version"], crate::VERSION);
        assert_eq!(value["workers"]["default"], 4);
        assert_eq!(value["workers"]["max"], MAX_WORKERS);
        // OCR block: modes list + range triples + defaults are all present.
        assert_eq!(
            value["ocr"]["modes"],
            serde_json::json!(["auto", "on", "off", "exhaustive"])
        );
        assert!(value["ocr"]["langs_installed"].is_array());
        assert!(value["ocr"]["dpi"]["min"].is_number());
        assert_eq!(value["ocr"]["psm"]["values"].as_array().unwrap().len(), 14);
        assert!(value["ocr"]["preprocess_default"].is_boolean());
        // Vision block: cap, gated tiers, per-sub-model present flags, defaults.
        assert_eq!(value["vision"]["max_tier"], "off");
        assert!(value["vision"]["tiers_available"].is_array());
        for category in ["detectors", "taggers", "captioners"] {
            let list = value["vision"][category].as_array().unwrap();
            assert_eq!(list.len(), 1, "{category}");
            assert!(list[0]["id"].is_string());
            assert!(list[0]["present"].is_boolean());
        }
        assert!(value["vision"]["defaults"]["detector_conf"].is_number());
    }

    #[test]
    fn settings_enumerates_the_fixture_tessdata_dir() {
        let temp = tempfile::tempdir().unwrap();
        let tessdata = temp.path().join("tessdata");
        std::fs::create_dir_all(&tessdata).unwrap();
        for name in ["eng.traineddata", "vie.traineddata", "readme.txt"] {
            std::fs::write(tessdata.join(name), b"x").unwrap();
        }
        let config = config_pointing_at(temp.path());

        let value = build_settings(Some(&config), VisionMode::Off, 4).unwrap();
        let langs: Vec<String> = value["ocr"]["langs_installed"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect();
        // The bundled fixture stems appear; the non-traineddata file does not.
        // (System `tesseract --list-langs` packs may add more — assert a subset.)
        assert!(langs.contains(&"eng".to_string()), "{langs:?}");
        assert!(langs.contains(&"vie".to_string()), "{langs:?}");
        assert!(!langs.contains(&"readme".to_string()), "{langs:?}");
    }

    #[test]
    fn tiers_available_gate_on_models_and_respect_the_cap() {
        let temp = tempfile::tempdir().unwrap();
        let config = config_pointing_at(temp.path());

        // A high cap but no staged vision models: only the pure-code `meta` tier
        // is offered; `tags`/`captions` are gated out and every sub-model reads
        // not-present.
        let value = build_settings(Some(&config), VisionMode::Captions, 4).unwrap();
        assert_eq!(value["vision"]["max_tier"], "captions");
        assert_eq!(
            value["vision"]["tiers_available"],
            serde_json::json!(["meta"])
        );
        assert_eq!(value["vision"]["detectors"][0]["present"], false);
        assert_eq!(value["vision"]["captioners"][0]["present"], false);

        // Planting a wrongly-hashed detector must NOT flip `tags` on — the
        // pinned-hash gate rejects it, so the offered tiers are unchanged.
        let vision_dir = temp.path().join("vision");
        std::fs::create_dir_all(&vision_dir).unwrap();
        std::fs::write(vision_dir.join("rf-detr-nano.onnx"), b"bogus").unwrap();
        let value = build_settings(Some(&config), VisionMode::Captions, 4).unwrap();
        assert_eq!(
            value["vision"]["tiers_available"],
            serde_json::json!(["meta"])
        );
        assert_eq!(value["vision"]["detectors"][0]["present"], false);

        // The cap itself gates the list: at `off` nothing is offered.
        let capped = build_settings(Some(&config), VisionMode::Off, 4).unwrap();
        assert_eq!(capped["vision"]["tiers_available"], serde_json::json!([]));
    }

    #[test]
    fn settings_defaults_and_ranges_mirror_the_w1_source() {
        // Ranges come from the single settings.rs bound consts; defaults from the
        // same Config fields the W1 OcrSettings/VisionSettings bases read — no
        // knob is redefined in the /settings builder.
        let value = build_settings(None, VisionMode::Off, 4).unwrap();
        let config = Config::default();
        let ocr = OcrSettings::from_config(&config);
        let vision = VisionSettings::from_config(&config);

        assert_eq!(value["ocr"]["dpi"]["min"], OCR_DPI_RANGE.0);
        assert_eq!(value["ocr"]["dpi"]["max"], OCR_DPI_RANGE.1);
        assert_eq!(value["ocr"]["dpi"]["default"], ocr.dpi.unwrap());
        assert_eq!(value["ocr"]["psm"]["default"], ocr.psm.clone().unwrap());
        assert_eq!(
            value["ocr"]["psm"]["values"].as_array().unwrap().len(),
            (OCR_PSM_RANGE.1 - OCR_PSM_RANGE.0 + 1) as usize
        );
        assert_eq!(value["ocr"]["preprocess_default"], ocr.preprocess.unwrap());
        assert_eq!(value["ocr"]["max_pages"]["min"], OCR_MAX_PAGES_RANGE.0);
        assert_eq!(value["ocr"]["max_pages"]["max"], OCR_MAX_PAGES_RANGE.1);
        assert_eq!(value["ocr"]["max_pages"]["default"], ocr.max_pages.unwrap());

        let defaults = &value["vision"]["defaults"];
        assert_eq!(defaults["detector_conf"], vision.detector_conf.unwrap());
        assert_eq!(defaults["tag_threshold"], vision.tag_threshold.unwrap());
        assert_eq!(defaults["tag_top_k"], vision.tag_top_k.unwrap());
        assert_eq!(defaults["max_frames"], vision.max_frames.unwrap());
        assert_eq!(defaults["timeout_secs"], vision.timeout_secs.unwrap());
    }

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
