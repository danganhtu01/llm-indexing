use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path as AxumPath, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use tokio::sync::{mpsc, RwLock};
use tower_http::trace::TraceLayer;
use uuid::Uuid;

use crate::config::{clamp_workers, Config, MAX_WORKERS};
use crate::embedding::{
    rank_chunks, rank_chunks_fast, Embedder, VectorScan, EMBEDDING_MODEL, MAX_HITS,
};
use crate::jobs_store::{JobsStore, MAX_PERSISTED_HISTORY, RESERVED_OUTPUT_NAME};
use crate::pipeline::{run_index, IndexRequest};
use crate::runtime::RuntimeKnobs;
use crate::settings::{
    installed_tessdata_langs, tessdata_sources, OcrSettings, VisionSettings, CAPTIONERS, DETECTORS,
    FACE_MODELS, OCR_DPI_RANGE, OCR_MAX_PAGES_RANGE, OCR_PSM_RANGE, TAGGERS,
};
use crate::store::{grouped, journal_path, BUSY_TIMEOUT, READ_BUSY_TIMEOUT};
use crate::vision::{
    available_tiers, captioner_present, corrupt_face_models, corrupt_models, detector_present,
    faces_present, missing_vision_prereqs, tagger_present, VisionMode,
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
    /// Re-attempt rows that have already burned `MAX_ATTEMPTS` without
    /// finishing. Default FALSE, and left that way except when the reason those
    /// rows failed has been fixed outside the engine: a resume with this set
    /// re-runs extraction, OCR and embedding over every file that has failed
    /// everything so far — ~69% of the rows on the live corpus. It changes which
    /// rows are attempted, never how one is processed.
    #[serde(default)]
    pub retry_errors: bool,
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
    /// Per-job live stage settings, keyed exactly like `cancellations` and
    /// managed on the same lifecycle: inserted at submit, handed to the blocking
    /// job, reachable by `POST /jobs/{id}/runtime` for as long as it runs.
    runtimes: Arc<RwLock<HashMap<String, Arc<RuntimeKnobs>>>>,
    /// Process-wide defaults every new job is snapshotted from. Never aliased
    /// into a running job, so `POST /runtime` cannot retune work in flight —
    /// that is what the per-job route is for.
    defaults: Arc<RuntimeKnobs>,
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
    /// Lazily loaded query-side embedding model, shared by every
    /// `/corpus/search?mode=semantic` request.
    embedder: Arc<QueryEmbedder>,
    /// Persisted job envelopes (P0-11) — `jobs.sqlite` under `output_root`.
    /// Written to on every status transition worth reconciling on; read by
    /// `GET /jobs/{id}` once a job has aged out of (or never existed in, after
    /// a restart) the in-memory `jobs` map.
    jobs_store: Arc<JobsStore>,
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
    let runtimes = Arc::new(RwLock::new(HashMap::new()));
    // Seed the process-wide stage defaults from the same config the jobs load,
    // with `serve --workers` taking precedence for the extract stage exactly as
    // it does in `run_job`.
    //
    // An unreadable config falls back to the built-in defaults rather than
    // failing startup. Reporting it here would MOVE an existing failure: a bad
    // config is already surfaced per job, by the job, which is what leaves the
    // published corpus untouched when a job cannot run. Turning it into a
    // startup panic would take the whole service down for a fault the job-level
    // path already handles correctly.
    let defaults = {
        let mut config = match Config::load(normalized.config_path.as_deref()) {
            Ok(config) => config,
            Err(error) => {
                tracing::warn!(
                    error = %format!("{error:#}"),
                    "config unreadable; runtime stage defaults fall back to built-ins \
                     (jobs will still report the config error)"
                );
                Config::default()
            }
        };
        config.workers = clamp_workers(normalized.workers);
        Arc::new(RuntimeKnobs::from_config(&config))
    };
    // P0-11: open (or create) the persisted job store before anything else
    // touches `jobs`, sweep any row a previous process left non-terminal, and
    // bound the history it starts this run with. Sweeping here — before the
    // HTTP listener binds — means the very first `GET /jobs/{id}` a caller
    // can make already sees the honest post-restart state, never a stale
    // "running" that no worker will ever finish.
    let jobs_store =
        Arc::new(JobsStore::open(&normalized.output_root).context("opening jobs.sqlite")?);
    let swept = jobs_store
        .sweep_interrupted()
        .context("sweeping interrupted jobs at startup")?;
    if !swept.is_empty() {
        tracing::warn!(
            count = swept.len(),
            ids = ?swept,
            "rewrote jobs left queued/running/cancelling by a prior instance to a terminal \
             error (\"{}\")",
            crate::jobs_store::INTERRUPTED_ERROR,
        );
    }
    jobs_store
        .prune(MAX_PERSISTED_HISTORY)
        .context("pruning persisted job history at startup")?;
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
        runtimes.clone(),
        normalized.clone(),
        jobs_store.clone(),
    ));
    let state = AppState {
        jobs,
        cancellations,
        runtimes,
        defaults,
        sender,
        output_root: normalized.output_root.clone(),
        roots: Arc::new(roots),
        vision_max: normalized.vision_max,
        config_path: normalized.config_path.clone(),
        workers: normalized.workers,
        embedder: Arc::new(QueryEmbedder::new(normalized.config_path.clone())),
        jobs_store,
    };
    Ok(Router::new()
        .route("/health", get(health))
        .route("/settings", get(settings))
        .route("/index", post(submit))
        .route("/jobs/{id}", get(job))
        .route("/jobs/{id}/cancel", post(cancel_job))
        .route("/runtime", get(runtime_defaults).post(set_runtime_defaults))
        .route(
            "/jobs/{id}/runtime",
            get(get_job_runtime).post(set_job_runtime),
        )
        .route("/corpus/tree", get(corpus_tree))
        .route("/corpus/documents/{id}/text", get(corpus_document_text))
        .route("/corpus/status", get(corpus_status_handler))
        .route("/corpus/search", get(corpus_search))
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
            // Faces is enumerated exactly like the other sub-models, and for the
            // same reason: an app must be able to tell "this box cannot do
            // faces" from "this box will not", without guessing. `present` is
            // false on every box that has not deliberately staged the pair, and
            // the default below is `off` on every box.
            "faces": sub_models(FACE_MODELS, faces_present(&models_dir)),
            "defaults": {
                "detector_conf": config.vision.detector_conf,
                "tag_threshold": config.vision.tag_score,
                "tag_top_k": config.vision.tag_top_k,
                "faces": config.vision.faces,
                "face_score": config.vision.face_score,
                "max_faces": config.vision.max_faces,
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

/// Persist one job envelope to `jobs.sqlite`, off the async executor. Best
/// effort: a write failure here is logged and swallowed rather than turned
/// into an HTTP error — the in-memory `jobs` map (the live queue's source of
/// truth) is unaffected either way, and the cost of a lost persisted write is
/// bounded to "this one transition doesn't survive an immediate restart",
/// never a wedged request.
async fn persist_job(jobs_store: &Arc<JobsStore>, id: &str, envelope: &Value) {
    let store = jobs_store.clone();
    let id = id.to_string();
    let envelope = envelope.clone();
    match tokio::task::spawn_blocking(move || store.record(&id, &envelope)).await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            tracing::warn!(error = %format!("{error:#}"), "failed to persist job envelope")
        }
        Err(error) => tracing::warn!(error = %format!("{error}"), "job-store write task failed"),
    }
}

/// [`persist_job`] plus a bound on the persisted history — called on every
/// terminal transition, so the store's row count stays checked at exactly the
/// points where new terminal rows are created.
async fn persist_terminal_job(jobs_store: &Arc<JobsStore>, id: &str, envelope: &Value) {
    persist_job(jobs_store, id, envelope).await;
    let store = jobs_store.clone();
    match tokio::task::spawn_blocking(move || store.prune(MAX_PERSISTED_HISTORY)).await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            tracing::warn!(error = %format!("{error:#}"), "failed to prune persisted job history")
        }
        Err(error) => tracing::warn!(error = %format!("{error}"), "job-store prune task failed"),
    }
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
        // `output` rides along from the first record onward: it is how
        // `/corpus/status` tells a reader that the database it is querying is
        // being written into right now.
        jobs.insert(
            id.clone(),
            json!({"id":id,"status":"queued","output":request.output,"submitted_at":now()}),
        );
    }
    state
        .cancellations
        .write()
        .await
        .insert(id.clone(), Arc::new(AtomicBool::new(false)));
    // A detached snapshot, not a share of the defaults: a later POST /runtime
    // must not retune this job behind the caller's back.
    let runtime = Arc::new(state.defaults.snapshot());
    // An explicit per-job `workers` is the caller stating this job's extract
    // width, so it outranks the process-wide default it was snapshotted from.
    if let Some(workers) = request.workers {
        let _ = runtime.apply(&Map::from_iter([(
            crate::runtime::EXTRACT.to_string(),
            json!(clamp_workers(workers)),
        )]));
    }
    state.runtimes.write().await.insert(id.clone(), runtime);
    request.id = Some(id.clone());
    let output = request.output.clone();
    let queued_envelope = json!({"id":id,"status":"queued","output":output,"submitted_at":now()});
    // Persisted BEFORE `try_send` below, not after: `try_send` is what makes
    // this job visible to the worker loop, which persists its own "running"
    // transition the moment it dequeues. Persisting "queued" first establishes
    // a strict happens-before between the two writes to the same row — were
    // this the other way around, a worker that dequeues and writes "running"
    // faster than this request resumes from its own await could have its
    // write clobbered back to "queued" moments later.
    persist_job(&state.jobs_store, &id, &queued_envelope).await;
    match state.sender.try_send((id.clone(), request)) {
        Ok(()) => (StatusCode::ACCEPTED, Json(queued_envelope)).into_response(),
        Err(_) => {
            state.jobs.write().await.remove(&id);
            state.cancellations.write().await.remove(&id);
            state.runtimes.write().await.remove(&id);
            // The job was never actually queued — leaving the "queued" row
            // above in `jobs.sqlite` would advertise a job through
            // `GET /jobs/{id}` that this response is telling the caller was
            // rejected. Overwrite it with the terminal state instead of
            // deleting it, so a caller that already saw "queued" and polls
            // afterward gets a coherent answer rather than a 404.
            let rejected = json!({"id":id,"status":"error",
                "error":"indexing queue is full","completed_at":now()});
            persist_terminal_job(&state.jobs_store, &id, &rejected).await;
            (
                StatusCode::TOO_MANY_REQUESTS,
                Json(json!({"status":"error","error":"indexing queue is full"})),
            )
                .into_response()
        }
    }
}

/// GET /jobs/{id} — the in-memory `jobs` map first (the live/fast path for a
/// job this process is actively tracking), falling back to the persisted
/// store for a job that has aged out of it (or, after a restart, was never in
/// it to begin with — P0-11's `jobs.sqlite` is what makes that case a served
/// terminal row instead of a bare 404).
async fn job(State(state): State<AppState>, AxumPath(id): AxumPath<String>) -> Response {
    if let Some(value) = state.jobs.read().await.get(&id).cloned() {
        return Json(value).into_response();
    }
    let store = state.jobs_store.clone();
    let lookup = id.clone();
    match tokio::task::spawn_blocking(move || store.get(&lookup)).await {
        Ok(Ok(Some(value))) => Json(value).into_response(),
        Ok(Ok(None)) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error":"job not found"})),
        )
            .into_response(),
        Ok(Err(error)) => {
            tracing::warn!(
                error = %format!("{error:#}"),
                "failed to read persisted job envelope"
            );
            (
                StatusCode::NOT_FOUND,
                Json(json!({"error":"job not found"})),
            )
                .into_response()
        }
        Err(error) => {
            tracing::warn!(error = %format!("{error}"), "job-store read task failed");
            (
                StatusCode::NOT_FOUND,
                Json(json!({"error":"job not found"})),
            )
                .into_response()
        }
    }
}

async fn cancel_job(State(state): State<AppState>, AxumPath(id): AxumPath<String>) -> Response {
    // A cancellation flag exists ONLY for a job this very process admitted
    // via `submit` — it is never persisted or reconstructed. So an id absent
    // here is either a job this process never ran (a restart, or an id that
    // simply does not exist) or one that aged out of `cancellations` — either
    // way, this process cannot cancel it, whatever its recorded state.
    let Some(cancellation) = state.cancellations.read().await.get(&id).cloned() else {
        return cancel_unmanaged(&state, &id).await;
    };
    // The write lock is scoped to the in-memory mutation only: persisting to
    // `jobs.sqlite` below awaits a `spawn_blocking` task, and holding an async
    // `RwLock` write guard across an `.await` would block every other job-map
    // reader/writer for that whole round trip.
    let cancelling = {
        let mut jobs = state.jobs.write().await;
        let Some(job) = jobs.get_mut(&id) else {
            // `prune_history` (see below) removes only terminal rows from
            // `jobs`, never from `cancellations` — so a cancellation flag can
            // outlive its job here. The row is gone from memory but its
            // terminal state is durable in `jobs.sqlite`; treat it the same
            // as any other id this process cannot presently act on rather
            // than 404ing a job `GET /jobs/{id}` still happily answers for.
            drop(jobs);
            return cancel_unmanaged(&state, &id).await;
        };
        match job["status"].as_str() {
            Some("queued" | "running" | "cancelling") => {
                cancellation.store(true, Ordering::Relaxed);
                job["status"] = json!("cancelling");
                job["message"] = json!("cancellation requested");
                Some(job.clone())
            }
            _ => None,
        }
    };
    match cancelling {
        Some(job) => {
            persist_job(&state.jobs_store, &id, &job).await;
            (StatusCode::ACCEPTED, Json(job)).into_response()
        }
        None => (
            StatusCode::CONFLICT,
            Json(json!({"error":"job is not active"})),
        )
            .into_response(),
    }
}

/// The fallback for `cancel_job` when `id` has no live cancellation flag in
/// this process. Consults `jobs.sqlite` — the same persisted fallback
/// `GET /jobs/{id}` reads — so a job this process merely lost live track of
/// (aged out of `jobs`/`cancellations`, or genuinely run by a prior instance
/// before a restart) is reported "not active" (409) rather than the
/// misleading "not found" (404) a caller would otherwise see for an id
/// `GET /jobs/{id}` still happily answers. This path can never itself flip a
/// job's status: without a cancellation flag there is no running work to
/// signal, so a persisted row — terminal by construction, since
/// `sweep_interrupted` rewrites anything a restart caught mid-run — is only
/// ever read, never mutated, which is what keeps a completed run from ever
/// being reported cancelled. An id in neither place is genuinely unknown.
async fn cancel_unmanaged(state: &AppState, id: &str) -> Response {
    let store = state.jobs_store.clone();
    let lookup = id.to_string();
    let found = match tokio::task::spawn_blocking(move || store.get(&lookup)).await {
        Ok(Ok(value)) => value.is_some(),
        Ok(Err(error)) => {
            tracing::warn!(
                error = %format!("{error:#}"),
                "failed to read persisted job envelope while cancelling"
            );
            false
        }
        Err(error) => {
            tracing::warn!(error = %format!("{error}"), "job-store read task failed");
            false
        }
    };
    if found {
        (
            StatusCode::CONFLICT,
            Json(json!({"error":"job is not active"})),
        )
            .into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(json!({"error":"job not found"})),
        )
            .into_response()
    }
}

/// GET /runtime — the process-wide stage settings future jobs start from.
async fn runtime_defaults(State(state): State<AppState>) -> Response {
    Json(state.defaults.view()).into_response()
}

/// POST /runtime — set the process-wide defaults.
///
/// Affects FUTURE jobs only. Jobs already running hold their own snapshot, so a
/// caller retuning the defaults cannot accidentally reach into work in flight;
/// `POST /jobs/{id}/runtime` is the deliberate way to do that.
async fn set_runtime_defaults(
    State(state): State<AppState>,
    Json(body): Json<Map<String, Value>>,
) -> Response {
    match state.defaults.apply(&body) {
        Ok(view) => Json(view).into_response(),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"status":"error","error":error})),
        )
            .into_response(),
    }
}

/// POST /jobs/{id}/runtime — retune a job that is running RIGHT NOW.
///
/// The settings this writes are the ones the job's extract admission gate and
/// embedder pool re-read as they work, so the change lands on files already in
/// flight rather than at the next job boundary.
/// GET /jobs/{id}/runtime — a job's LIVE per-job stage values, without changing
/// them.
///
/// The counterpart the mid-run POST was missing. `GET /runtime` reports the
/// process-wide defaults, not what a running job was individually retuned to, so
/// there was no way to read a job's true live values back — a caller could set
/// extract=20 on a job, get a 200, and then only ever see the process default
/// (12) on any later read, which reads exactly like the setting was lost. This
/// closes that: it returns the same per-job `RuntimeKnobs.view()` the POST does.
async fn get_job_runtime(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    let status = state
        .jobs
        .read()
        .await
        .get(&id)
        .and_then(|job| job["status"].as_str().map(str::to_string));
    let Some(status) = status else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error":"job not found"})),
        )
            .into_response();
    };
    // A terminal job's per-job knobs are reaped, so there is nothing live to
    // report — 409 rather than a 404 that would read as "never existed".
    if !matches!(status.as_str(), "queued" | "running" | "cancelling") {
        return (
            StatusCode::CONFLICT,
            Json(json!({"error":"job is not active","status":status})),
        )
            .into_response();
    }
    match state.runtimes.read().await.get(&id).cloned() {
        Some(runtime) => Json(runtime.view()).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({"error":"job not found"})),
        )
            .into_response(),
    }
}

async fn set_job_runtime(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Json(body): Json<Map<String, Value>>,
) -> Response {
    // Status first: a terminal job is a 409, and it must not read as a 404 just
    // because its settings were already reaped.
    let status = state
        .jobs
        .read()
        .await
        .get(&id)
        .and_then(|job| job["status"].as_str().map(str::to_string));
    let Some(status) = status else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error":"job not found"})),
        )
            .into_response();
    };
    if !matches!(status.as_str(), "queued" | "running" | "cancelling") {
        return (
            StatusCode::CONFLICT,
            Json(json!({"error":"job is not active","status":status})),
        )
            .into_response();
    }
    let Some(runtime) = state.runtimes.read().await.get(&id).cloned() else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error":"job not found"})),
        )
            .into_response();
    };
    match runtime.apply(&body) {
        Ok(view) => Json(view).into_response(),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"status":"error","error":error})),
        )
            .into_response(),
    }
}

async fn worker(
    mut receiver: mpsc::Receiver<(String, JobRequest)>,
    jobs: Arc<RwLock<HashMap<String, Value>>>,
    cancellations: Arc<RwLock<HashMap<String, Arc<AtomicBool>>>>,
    runtimes: Arc<RwLock<HashMap<String, Arc<RuntimeKnobs>>>>,
    config: ServiceConfig,
    jobs_store: Arc<JobsStore>,
) {
    while let Some((id, request)) = receiver.recv().await {
        let cancellation = cancellations
            .read()
            .await
            .get(&id)
            .cloned()
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        if cancellation.load(Ordering::Relaxed) {
            let value = json!({"id":id,"status":"cancelled","message":"cancelled before start","completed_at":now()});
            jobs.write().await.insert(id.clone(), value.clone());
            persist_terminal_job(&jobs_store, &id, &value).await;
            continue;
        }
        let output = request.output.clone();
        let running = json!({"id":id,"status":"running","output":output,"processed":0,"total":0,
                   "started_at":now()});
        jobs.write().await.insert(id.clone(), running.clone());
        persist_job(&jobs_store, &id, &running).await;
        // Same lookup shape as the cancellation above: the settings the HTTP
        // route can reach must be the very ones this job runs with, so take the
        // registered Arc rather than building a fresh one.
        let runtime = runtimes
            .read()
            .await
            .get(&id)
            .cloned()
            .unwrap_or_else(|| Arc::new(RuntimeKnobs::from_config(&Config::default())));
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
                runtime,
            )
        })
        .await;
        // A cancelled job keeps every file it had already committed to the
        // published database — there is no temporary build to discard and no
        // earlier corpus to fall back to. Resubmitting with `resume` continues
        // from exactly that point.
        let value = if cancellation.load(Ordering::Relaxed) {
            let message = format!(
                "indexing cancelled; partial corpus retained in {output}, \
                 resubmit with resume to continue"
            );
            json!({"id":id,"status":"cancelled","output":output,"message":message,
                   "completed_at":now()})
        } else {
            // A failed job is no longer a job that published nothing: it wrote
            // into the destination as it went, so whatever it had committed is
            // still there. Say so, and say what to do about it — a caller that
            // resubmits blind gets "output already exists" and no explanation.
            let partial = format!(
                "{output} may hold the files this job committed before it failed; \
                 resubmit with resume to continue or overwrite to start clean"
            );
            match result {
                Ok(Ok(value)) => value,
                Ok(Err(error)) => {
                    json!({"id":id,"status":"error","output":output,"error":format!("{error:#}"),
                           "partial_corpus":partial,"completed_at":now()})
                }
                Err(error) => {
                    json!({"id":id,"status":"error","output":output,
                           "error":format!("worker join: {error}"),
                           "partial_corpus":partial,"completed_at":now()})
                }
            }
        };
        jobs.write().await.insert(id.clone(), value.clone());
        persist_terminal_job(&jobs_store, &id, &value).await;
    }
}

fn run_job(
    id: &str,
    request: JobRequest,
    service: &ServiceConfig,
    jobs: Arc<RwLock<HashMap<String, Value>>>,
    cancellation: Arc<AtomicBool>,
    runtime: Arc<RuntimeKnobs>,
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
    // The job writes straight into the published database so that work survives
    // a crash and `resume` can continue from it. `overwrite` therefore deletes
    // rather than swapping a finished build in at the end: the previous corpus
    // cannot be held as a fallback while its replacement is written into the
    // same file. `resume` wins when both are set — it exists precisely to keep
    // what is there. An interrupted overwrite leaves a partial NEW corpus,
    // resumable with `resume`, not the superseded one.
    //
    // The deletion itself is deferred to `run_index`, which performs it only
    // once the config, the vision models and the embedding model have all
    // loaded. A job that is going to fail on a missing model or a bad config
    // must fail with the old corpus still on disk.
    let overwrite = request.overwrite && !request.resume;
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
    // Faces gets the integrity half of that gate but NOT the presence half. An
    // absent pair means the capability is absent and the job runs without it;
    // a pair that is present but does not match its pinned hash is bytes nobody
    // vouched for computing claims about people's identities, so the job stops.
    if config.vision.max != VisionMode::Off && config.vision.faces_enabled() {
        let corrupt = corrupt_face_models(&config.vision_models_dir());
        if !corrupt.is_empty() {
            anyhow::bail!(
                "face model integrity check failed (corrupt/truncated/tampered); \
                 re-run llm-index fetch-data --faces --force: {corrupt:?}"
            )
        }
    }
    let progress_id = id.to_owned();
    let stats = run_index(IndexRequest {
        paths: &paths,
        out: &destination,
        config: config.clone(),
        resume: request.resume,
        overwrite,
        artifacts: false,
        retry_errors: request.retry_errors,
        include_paths,
        cancellation: Some(cancellation),
        runtime: Some(runtime),
        progress: Some(Arc::new(move |processed, total| {
            let mut jobs = jobs.blocking_write();
            if let Some(job) = jobs.get_mut(&progress_id) {
                job["processed"] = json!(processed);
                job["total"] = json!(total);
            }
        })),
    })?;
    Ok(json!({
        "id":id,"status":"complete","output":request.output,"database":destination,"files":stats.files,
        "ocr_files":stats.ocr_files,"errors":stats.errors,"encrypted":stats.encrypted,"skipped":stats.skipped,
        "capped":stats.capped,"incomplete":stats.incomplete,"embedded_chunks":stats.embedded_chunks,"removed":stats.removed,
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
/// `output_root` — and it must not be `jobs.sqlite` (P0-11's persisted job
/// store), which shares that directory but is not a corpus: a job targeting
/// it would corrupt the restart-reconciliation state, and a `/corpus/*` read
/// against it would just fail confusingly on a schema it doesn't recognize.
fn valid_output_name(name: &str) -> bool {
    Path::new(name).file_name().and_then(|n| n.to_str()) == Some(name)
        && name.ends_with(".sqlite")
        && name != RESERVED_OUTPUT_NAME
}

// ── Corpus read surface (GET /corpus/tree, /corpus/documents/{id}/text, /corpus/status) ──
//
// READ-ONLY over whatever `corpus.sqlite` the most recent job wrote. Consumer
// apps used to open the SQLite file directly; this surface lets them stop
// decoding the schema themselves. The database is absent until the first job
// writes it — every route below degrades to an empty/zeroed result rather than
// an error — and, since jobs write in place, it can be read mid-run and answer
// from the rows committed so far.

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

/// The three states a corpus can be in when a read arrives. `Absent` and
/// `Unreadable` must never collapse into each other: "no job has written this
/// output yet" and "the database is there but its rows cannot be read" look
/// identical to a consumer that is handed a zero either way, and the second is
/// a fault it needs to see rather than act on.
enum Corpus {
    Absent,
    Ready(Connection),
    /// A writer holds the database. Distinct from `Unreadable`: the corpus is
    /// healthy and the caller should retry, not treat this as a fault.
    Busy,
    Unreadable(String),
}

/// Open a corpus database read-only.
///
/// Two failure modes look alike here and must not be conflated. A writer
/// spilling its page cache escalates to an EXCLUSIVE lock and holds it until
/// the batch commit, which surfaces as `SQLITE_BUSY` — the database is fine.
/// A writer *killed* mid-transaction leaves a hot rollback journal, which a
/// read-only connection cannot replay (SQLite refuses the database rather than
/// serve pages the journal is about to undo) — only that one needs recovery.
/// Reporting a locked corpus as unreadable would signal corruption during
/// ordinary indexing, i.e. exactly the read-while-writing case that writing in
/// place exists to enable.
fn open_ro(path: &Path) -> Corpus {
    if !path.exists() {
        return Corpus::Absent;
    }
    match read_only(path) {
        Ok(connection) => Corpus::Ready(connection),
        Err(error) if is_busy(&error) => Corpus::Busy,
        // Only a read-write open can roll a hot journal back, so do that here
        // and retry, rather than leaving the corpus unreadable until the next
        // job happens to run. A retry that is merely locked is still Busy.
        Err(_) => match recover_journal(path).and_then(|()| read_only(path)) {
            Ok(connection) => Corpus::Ready(connection),
            Err(error) if is_busy(&error) => Corpus::Busy,
            Err(error) => Corpus::Unreadable(error.to_string()),
        },
    }
}

/// Whether a failure means "someone else holds the lock" rather than "this
/// database cannot be read".
fn is_busy(error: &rusqlite::Error) -> bool {
    matches!(
        error.sqlite_error_code(),
        Some(rusqlite::ErrorCode::DatabaseBusy) | Some(rusqlite::ErrorCode::DatabaseLocked)
    )
}

/// Open read-only and touch the schema. The probe is the point: SQLite defers
/// real access until the first statement, so a database that cannot be read —
/// hot journal, corruption, bad permissions — would otherwise open cleanly here
/// and fail later where the failure is easier to swallow.
///
/// Reads take a much shorter busy timeout than the writer's: something polling
/// `/corpus/status` during a long index wants a prompt "busy, retry" far more
/// than it wants to block for the writer's whole commit window.
fn read_only(path: &Path) -> Result<Connection, rusqlite::Error> {
    // Before the open, or a corpus carrying a `vec0` shadow index would be
    // served by the scan forever: the module reaches a connection through
    // SQLite's auto-extension list, consulted once as the connection is
    // created. See [`crate::vec0::register`].
    crate::vec0::register();
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    connection.busy_timeout(READ_BUSY_TIMEOUT)?;
    connection.query_row("SELECT count(*) FROM sqlite_master", [], |row| {
        row.get::<_, i64>(0)
    })?;
    Ok(connection)
}

/// Roll back a journal left behind by an interrupted writer, by opening
/// read-write long enough for SQLite to notice it. Recovery is what the writer
/// itself would have done on its next open; doing it here means a killed job
/// does not leave the corpus unreadable to every consumer in the meantime.
fn recover_journal(path: &Path) -> Result<(), rusqlite::Error> {
    if !journal_path(path).exists() {
        return Ok(());
    }
    let connection = Connection::open(path)?;
    connection.busy_timeout(BUSY_TIMEOUT)?;
    connection.query_row("SELECT count(*) FROM sqlite_master", [], |row| {
        row.get::<_, i64>(0)
    })?;
    Ok(())
}

/// Why a corpus read could not be served. Kept apart from a plain string so a
/// contended database is never reported as a damaged one.
enum ReadError {
    /// A writer holds the lock. Healthy corpus, retryable.
    Busy,
    Unreadable(String),
}

impl From<rusqlite::Error> for ReadError {
    fn from(error: rusqlite::Error) -> Self {
        if is_busy(&error) {
            ReadError::Busy
        } else {
            ReadError::Unreadable(error.to_string())
        }
    }
}

impl From<anyhow::Error> for ReadError {
    /// Store helpers return anyhow; downcast so a lock contention arriving
    /// through one of them is still classified as busy rather than damaged.
    fn from(error: anyhow::Error) -> Self {
        match error.downcast_ref::<rusqlite::Error>() {
            Some(sqlite) if is_busy(sqlite) => ReadError::Busy,
            _ => ReadError::Unreadable(error.to_string()),
        }
    }
}

/// A corpus that exists but cannot be read is a fault to report, not an empty
/// result to return — but a corpus someone is *writing* is neither. Both are
/// 503, and the body is the difference: one says retry, the other says broken.
fn read_error(error: &ReadError) -> Response {
    let body = match error {
        ReadError::Busy => json!({
            "error": "corpus database busy",
            "detail": "a job is writing this corpus; retry shortly",
            "retryable": true,
        }),
        ReadError::Unreadable(detail) => json!({
            "error": "corpus database unreadable",
            "detail": detail,
            "retryable": false,
        }),
    };
    (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
}

/// A read task that failed to run at all (join error) is a service fault.
fn unreadable(detail: &str) -> Response {
    read_error(&ReadError::Unreadable(detail.to_string()))
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
/// `Ok` of an empty map means the corpus holds no documents (or none has been
/// written yet); a query that genuinely failed is an `Err`, never an empty map.
fn corpus_index(path: &Path) -> Result<HashMap<String, DocMeta>, ReadError> {
    let connection = match open_ro(path) {
        Corpus::Absent => return Ok(HashMap::new()),
        Corpus::Ready(connection) => connection,
        Corpus::Busy => return Err(ReadError::Busy),
        Corpus::Unreadable(error) => return Err(ReadError::Unreadable(error)),
    };
    let mut statement = connection.prepare(
        "SELECT f.path, f.id, COALESCE(f.chars,0), COALESCE(f.method,''), COALESCE(f.lang,''), \
                    COALESCE(substr(fts.content,1,400),'') \
             FROM files f JOIN fts ON fts.rowid = f.id",
    )?;
    let rows = statement.query_map([], |row| {
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
    })?;
    rows.collect::<Result<HashMap<_, _>, _>>()
        .map_err(ReadError::from)
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
    match tokio::task::spawn_blocking(move || tree_entries(&root, &output)).await {
        Ok(Ok(entries)) => Json(entries).into_response(),
        Ok(Err(error)) => read_error(&error),
        Err(error) => unreadable(&format!("read task failed: {error}")),
    }
}

fn tree_entries(root: &Path, corpus_db: &Path) -> Result<Vec<Value>, ReadError> {
    let index = corpus_index(corpus_db)?;
    let mut rows = Vec::new();
    if root.is_dir() {
        let _ = collect_tree(root, root, 0, &index, &mut rows);
    }
    Ok(rows)
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
    let content = match tokio::task::spawn_blocking(move || document_text(&output, id)).await {
        Ok(Ok(Some(content))) => content,
        // Only a corpus that was actually readable can say "no such document".
        Ok(Ok(None)) => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error":"document not found"})),
            )
                .into_response()
        }
        Ok(Err(error)) => return read_error(&error),
        Err(error) => return unreadable(&format!("read task failed: {error}")),
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

fn document_text(corpus_db: &Path, id: i64) -> Result<Option<String>, ReadError> {
    let connection = match open_ro(corpus_db) {
        Corpus::Absent => return Ok(None),
        Corpus::Ready(connection) => connection,
        Corpus::Busy => return Err(ReadError::Busy),
        Corpus::Unreadable(error) => return Err(ReadError::Unreadable(error)),
    };
    connection
        .query_row(
            "SELECT fts.content FROM files f JOIN fts ON fts.rowid = f.id WHERE f.id = ?1",
            params![id],
            |row| row.get(0),
        )
        .optional()
        .map_err(ReadError::from)
}

/// GET /corpus/status[?output=corpus.sqlite]
///
/// Cheap corpus-wide aggregates: indexed/pending file counts, total
/// characters/bytes, OCR count, language/method breakdowns, and the corpus's
/// schema version. Every one of these is a single `COUNT`/`meta` lookup —
/// deliberately, so a consumer polling this on a tight tick never pays for a
/// tree walk. Zeroed when the database is absent.
async fn corpus_status_handler(
    State(state): State<AppState>,
    Query(query): Query<OutputQuery>,
) -> Response {
    let output = match resolve_output(&state, query.output.as_deref()) {
        Ok(path) => path,
        Err(response) => return response.into_response(),
    };
    // Jobs write into the published database as they go, so a corpus can be
    // read while it is still being built. Nothing about the rows themselves
    // says so, and the atomic-publish guarantee that used to make a visible
    // corpus a finished one is gone, so report it explicitly.
    let writing = writing_output(&state, &output).await;
    match tokio::task::spawn_blocking(move || corpus_status(&output)).await {
        Ok(Ok(mut value)) => {
            value["writing"] = json!(writing);
            Json(value).into_response()
        }
        Ok(Err(error)) => read_error(&error),
        Err(error) => unreadable(&format!("read task failed: {error}")),
    }
}

/// Whether a queued, running or cancelling job is targeting this database.
async fn writing_output(state: &AppState, output: &Path) -> bool {
    let Some(name) = output.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    state.jobs.read().await.values().any(|job| {
        job["output"].as_str() == Some(name)
            && matches!(
                job["status"].as_str(),
                Some("queued" | "running" | "cancelling")
            )
    })
}

/// Aggregates over one corpus. Every count is read or the whole call fails:
/// `unwrap_or(0)` here would answer "0 files" over a database holding every row
/// a consumer has, which reads as an empty corpus rather than a failed read.
fn corpus_status(path: &Path) -> Result<Value, ReadError> {
    let connection = match open_ro(path) {
        Corpus::Absent => return Ok(empty_status()),
        Corpus::Ready(connection) => connection,
        Corpus::Busy => return Err(ReadError::Busy),
        Corpus::Unreadable(error) => return Err(ReadError::Unreadable(error)),
    };
    let count = |sql: &str| -> Result<i64, ReadError> {
        connection
            .query_row(sql, [], |row| row.get(0))
            .map_err(ReadError::from)
    };
    let indexed_files = count("SELECT COUNT(*) FROM files")?;
    // `pending_files` used to cost the caller a full `/corpus/tree` walk (sorted
    // recursive `fs::read_dir` plus a snippet-carrying join) just to count entries
    // whose `document_id` came back null. Deriving it from the pipeline's own
    // last-discovery snapshot (`crate::pipeline::run_index` stamps
    // `last_discovered_files` into `meta` before it filters down to what actually
    // needs work) turns that into one more `COUNT(*)` here — no tree walk, no
    // snippets. A corpus with no recorded discovery yet (never indexed through
    // this harness, or a bare test fixture) reads as 0 pending rather than an error.
    let discovered = crate::store::read_meta(&connection, "last_discovered_files")
        .map_err(ReadError::from)?
        .and_then(|value| value.parse::<i64>().ok());
    let pending_files = discovered
        .map(|total| (total - indexed_files).max(0))
        .unwrap_or(0);
    // A corpus predating the `chunks` table (or a hand-built fixture that
    // never ran through `migrate`) simply has nothing embedded yet — that
    // reads as 0, not as a fault, the same way a fresh `meta` table does above.
    let embedded_chunks = if table_exists(&connection, "chunks")? {
        count("SELECT COUNT(*) FROM chunks")?
    } else {
        0
    };
    Ok(json!({
        "indexed_files": indexed_files,
        "pending_files": pending_files,
        "total_characters": count("SELECT COALESCE(SUM(chars),0) FROM files")?,
        "total_bytes": count("SELECT COALESCE(SUM(size),0) FROM files")?,
        "ocr_files": count("SELECT COALESCE(SUM(ocr_used),0) FROM files")?,
        "embedded_chunks": embedded_chunks,
        "languages": grouped(&connection, "lang", 10)?,
        "methods": grouped(&connection, "method", 20)?,
        "schema_version": crate::store::schema_version(&connection).map_err(ReadError::from)?,
    }))
}

/// Whether `name` exists as a table in this connection's schema. Guards
/// aggregates over tables that a corpus might simply not have yet — added
/// after schema version 1, or absent from a hand-built test fixture — so
/// `/corpus/status` reads that as "nothing there" rather than failing the
/// whole response the way a bare `SELECT COUNT(*)` against a missing table
/// would (`ReadError::Unreadable`, indistinguishable from real corruption).
fn table_exists(connection: &Connection, name: &str) -> Result<bool, ReadError> {
    connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
            [name],
            |row| row.get::<_, i64>(0),
        )
        .map(|count| count > 0)
        .map_err(ReadError::from)
}

fn empty_status() -> Value {
    json!({
        "indexed_files": 0, "pending_files": 0, "total_characters": 0, "total_bytes": 0,
        "ocr_files": 0, "embedded_chunks": 0, "languages": Vec::<(String, i64)>::new(),
        "methods": Vec::<(String, i64)>::new(), "schema_version": 0,
    })
}

// ── Semantic search (GET /corpus/search) ────────────────────────────────────
//
// The `chunks` embeddings every index job has been writing since the corpus
// format's first release had exactly one reader — the `vector-search` CLI
// subcommand — so on the live corpora 4.1 GB of paid-for vectors were reachable
// only by shelling into the container. This route is that reader, over the same
// read-only corpus surface as `/corpus/tree` and friends.
//
// `POST /search/fts` and `POST /search/vector` were deliberately moved out of
// this service to `llm-search` (see docs/HTTP_API.md); this is not a walk-back
// of that. `llm-search` holds every chunk vector RESIDENT to serve a
// search-as-you-type socket — 2.68 M x 384 floats plus their text is a
// multi-gigabyte process, which is why the hub app does not run one. What is
// added back here is the streaming, nothing-resident half: one ranking pass per
// request, `O(limit)` memory, and no second service to deploy.
//
// That pass is a k-NN lookup when the corpus carries a `vec0` shadow index and
// an exhaustive scan when it does not (`crate::vec0`, `crate::embedding`). The
// choice is read off the corpus rather than configured here: this route stays
// read-only, and nothing it does can create, repair or invalidate an index.

/// Modes `/corpus/search` accepts. The list is what a rejected request is told,
/// so adding a keyword mode later stays a one-line change with no new failure
/// shape.
///
/// `semantic` is exact and is the default. `semantic_fast` is the opt-in that
/// exists because exactness has a price: it ranks through the corpus' QUANTISED
/// shadow index, which reads a fraction of the bytes and returns an
/// approximation of the same list. Two modes rather than one mode with a
/// tolerance knob, because approximate and exact are different promises and a
/// caller has to make that choice deliberately — see
/// [`crate::embedding::rank_chunks_fast`].
const SEARCH_MODES: &[&str] = &["semantic", "semantic_fast"];

/// The mode that ranks through the quantised index.
const FAST_MODE: &str = "semantic_fast";

/// Hits returned when the caller does not ask. Matches the `search` CLI
/// subcommand rather than `vector-search`'s 10: a search API's default page is
/// what a UI renders, and 20 is that.
const DEFAULT_SEARCH_LIMIT: usize = 20;

#[derive(Debug, Deserialize)]
struct SearchQuery {
    /// Optional in the type ONLY so a missing query answers the service's own
    /// JSON `400` instead of axum's plain-text rejection.
    #[serde(default)]
    q: Option<String>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    output: Option<String>,
}

/// GET /corpus/search?q=TEXT[&mode=semantic][&limit=20][&output=corpus.sqlite]
///
/// Embeds `q` with the same model the corpus rows were embedded with and ranks
/// `chunks` by cosine similarity. Everything expensive — loading the model,
/// embedding the query, scanning the corpus — happens on a blocking worker.
///
/// The response always carries `status`, and an empty `hits` is never left
/// ambiguous: a corpus indexed without embeddings, a corpus embedded by another
/// model, and a model that has not finished loading are three different
/// `status`/`reason` pairs, not three empty lists. Only a corpus that exists and
/// cannot be read is an error (`503`, shared with the rest of this surface).
///
/// It also carries `path`, because the ranking runs over a `vec0` shadow index
/// when the corpus has a usable one and over an exhaustive scan when it does
/// not. The two return the same hits; they do not take the same time, so which
/// one ran is a fact the caller is owed rather than one to infer from a
/// stopwatch.
async fn corpus_search(
    State(state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> Response {
    let mode = query.mode.as_deref().unwrap_or("semantic");
    if !SEARCH_MODES.contains(&mode) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": format!("unsupported search mode {mode:?}"),
                        "modes": SEARCH_MODES})),
        )
            .into_response();
    }
    let text = query.q.as_deref().unwrap_or_default().trim().to_string();
    if text.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":"q is required and must not be blank"})),
        )
            .into_response();
    }
    let fast = mode == FAST_MODE;
    let limit = query
        .limit
        .unwrap_or(DEFAULT_SEARCH_LIMIT)
        .clamp(1, MAX_HITS);
    let output = match resolve_output(&state, query.output.as_deref()) {
        Ok(path) => path,
        Err(response) => return response.into_response(),
    };
    let embedder = match state.embedder.acquire().await {
        Acquired::Ready(embedder) => embedder,
        Acquired::Warming { warming_ms } => {
            return Json(search_response(
                &text,
                mode,
                limit,
                SearchOutcome::Warming { warming_ms },
            ))
            .into_response()
        }
        Acquired::Unavailable { reason } => {
            return Json(search_response(
                &text,
                mode,
                limit,
                // `acquire` armed a fresh load on the way out; say so, or a
                // caller has no way to know retrying is worth anything.
                SearchOutcome::Unavailable {
                    reason,
                    retrying: true,
                },
            ))
            .into_response();
        }
    };
    let scan = {
        let text = text.clone();
        tokio::task::spawn_blocking(move || {
            let started = Instant::now();
            // Embed under the lock, then drop it before the scan: the scan is
            // the long half, and holding the single embedder across it would
            // serialize every concurrent search on the wrong resource.
            //
            // A panic inside `embed_query` (never observed, but the ONNX call
            // is not something this code controls) would otherwise poison the
            // mutex and brick every later search behind an opaque 503 with no
            // way back short of a restart — unlike a failed *load*, which
            // explicitly re-arms. `embed_query` only reads the model to
            // produce a `Result`, so the guarded data is not left structurally
            // broken by a panic while holding it; recovering the guard keeps
            // this failure mode self-healing like the rest of this surface.
            let query_vector = {
                let mut guard = embedder
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                guard.embed_query(&text)
            };
            let query_vector = match query_vector {
                Ok(vector) => vector,
                // A model that loaded but cannot embed is a service fault, not
                // an empty result.
                Err(error) => {
                    return Ok(SearchOutcome::Unavailable {
                        reason: format!("embedding the query failed: {error:#}"),
                        // The model is loaded and stays loaded: this query
                        // failed, not the embedder, so nothing is being retried.
                        retrying: false,
                    });
                }
            };
            semantic_scan(&output, &query_vector, limit, fast, started)
        })
        .await
    };
    match scan {
        Ok(Ok(outcome)) => Json(search_response(&text, mode, limit, outcome)).into_response(),
        Ok(Err(error)) => read_error(&error),
        Err(error) => unreadable(&format!("search task failed: {error}")),
    }
}

/// Rank one corpus against an already-embedded query.
///
/// `started` is passed in so the reported `elapsed_ms` covers the whole
/// server-side cost the caller waited on — embedding included — rather than
/// just the scan.
fn semantic_scan(
    output: &Path,
    query_vector: &[f32],
    limit: usize,
    fast: bool,
    started: Instant,
) -> Result<SearchOutcome, ReadError> {
    let connection = match open_ro(output) {
        Corpus::Absent => {
            let name = output.file_name().unwrap_or_default().to_string_lossy();
            return Ok(SearchOutcome::NoEmbeddings {
                reason: format!("no corpus database at {name} yet"),
                other_models: Vec::new(),
            });
        }
        Corpus::Ready(connection) => connection,
        Corpus::Busy => return Err(ReadError::Busy),
        Corpus::Unreadable(error) => return Err(ReadError::Unreadable(error)),
    };
    // A corpus written before the chunks table existed has no embeddings and no
    // table to scan; that is a shape of "nothing to search", not a failed query.
    let embedded: Option<i64> = connection
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'",
            [],
            |row| row.get(0),
        )
        .optional()?;
    if embedded.is_none() {
        return Ok(SearchOutcome::NoEmbeddings {
            reason: "this corpus has no chunks table; it was written by a build without \
                     embeddings"
                .into(),
            other_models: Vec::new(),
        });
    }
    let scan = if fast {
        rank_chunks_fast(&connection, EMBEDDING_MODEL, query_vector, limit)?
    } else {
        rank_chunks(&connection, EMBEDDING_MODEL, query_vector, limit)?
    };
    let elapsed_ms = started.elapsed().as_millis() as u64;
    if scan.compared > 0 {
        return Ok(SearchOutcome::Ranked { scan, elapsed_ms });
    }
    // Nothing comparable. Which of the two reasons it is matters: one says
    // "turn embedding on and reindex", the other says "this corpus is on a
    // different model and reindexing it would migrate, not fix".
    let reason = if scan.skipped > 0 {
        format!(
            "every one of the {} embeddings in this corpus was written by another model; \
             queries here are embedded with {EMBEDDING_MODEL}, and cosine across two \
             embedding spaces is meaningless",
            scan.skipped
        )
    } else {
        "this corpus holds no embeddings: no indexed file has been embedded yet".into()
    };
    Ok(SearchOutcome::NoEmbeddings {
        reason,
        other_models: scan.other_models,
    })
}

/// What one semantic request resolved to. Every variant is a `200`: the only
/// `/corpus/search` failures are a malformed request (`400`) and a corpus that
/// exists but cannot be read (`503`, via [`read_error`]).
enum SearchOutcome {
    /// A scan ran over comparable vectors. `hits` may still be short of `limit`
    /// — or empty, if the corpus holds fewer chunks than that.
    Ranked { scan: VectorScan, elapsed_ms: u64 },
    /// There was nothing to rank, and this is why.
    NoEmbeddings {
        reason: String,
        other_models: Vec<String>,
    },
    /// The query embedder is loading. Reported rather than waited on.
    Warming { warming_ms: u64 },
    /// The query embedder could not be loaded, or could not embed. `retrying`
    /// says whether a fresh load is already in flight.
    Unavailable { reason: String, retrying: bool },
}

/// The `/corpus/search` envelope.
///
/// `mode`, `status`, `limit` and `hits` are present in every response so a
/// consumer branches on `status` and never has to interpret an empty `hits`.
/// `hits` mirrors `llm-search`'s `/search/vector` rows (`path`, `name`,
/// `chunk_index`, `score`, `content`) so the two search surfaces stay one shape.
fn search_response(query: &str, mode: &str, limit: usize, outcome: SearchOutcome) -> Value {
    let mut body = json!({
        "mode": mode,
        "query": query,
        "limit": limit,
        "model": EMBEDDING_MODEL,
        "hits": Vec::<Value>::new(),
    });
    match outcome {
        SearchOutcome::Ranked { scan, elapsed_ms } => {
            body["status"] = json!("ready");
            body["hits"] = json!(scan.hits);
            body["compared_chunks"] = json!(scan.compared);
            body["skipped_chunks"] = json!(scan.skipped);
            body["elapsed_ms"] = json!(elapsed_ms);
            // Which ranking path served this. The exact paths' hits and scores
            // are the same either way; their latency is not, by more than an
            // order of magnitude on a large corpus, so a consumer that sees a
            // slow answer can tell "this corpus has no shadow index" from "this
            // corpus has one and it was not used" without guessing.
            body["path"] = json!(scan.path.as_str());
            // And whether that path is the scan's own answer, stated rather
            // than left to be looked up: `semantic_fast` over a corpus with no
            // quantised index is answered EXACTLY, and a consumer that has to
            // label its results has no other way to know which it got.
            body["exact"] = json!(scan.path.is_exact());
            if let Some(candidates) = scan.candidates {
                body["candidates"] = json!(candidates);
            }
            if let Some(note) = scan.index_note {
                body["index_note"] = json!(note);
            }
        }
        SearchOutcome::NoEmbeddings {
            reason,
            other_models,
        } => {
            body["status"] = json!("no_embeddings");
            body["reason"] = json!(reason);
            if !other_models.is_empty() {
                body["other_models"] = json!(other_models);
            }
        }
        SearchOutcome::Warming { warming_ms } => {
            body["status"] = json!("warming");
            body["reason"] = json!(
                "the query embedding model is loading (first semantic search in this \
                 process); retry shortly"
            );
            body["warming_ms"] = json!(warming_ms);
        }
        SearchOutcome::Unavailable { reason, retrying } => {
            body["status"] = json!("unavailable");
            body["reason"] = json!(reason);
            body["retrying"] = json!(retrying);
        }
    }
    body
}

/// The query half of semantic search: the embedding model, loaded once, lazily,
/// on the first `mode=semantic` request.
///
/// An index job builds its own embedder; a serve process that has only ever
/// answered reads has none, and building one is not free — it opens an ONNX
/// session and reads the model out of the fastembed cache (measured on the
/// workhorse: see docs/HTTP_API.md). Paying that inside the request would make
/// the first search sit there with nothing to tell the caller apart from a slow
/// scan, so the first request ARMS the load and answers `status: "warming"` at
/// once. The load runs on a blocking worker; a later request finds it ready.
///
/// This embedder only ever embeds QUERIES. It shares the model and the code
/// path with indexing but writes nothing, so no corpus row and no job outcome
/// depends on whether serve happens to have one loaded.
struct QueryEmbedder {
    config_path: Option<PathBuf>,
    state: RwLock<EmbedderState>,
}

#[derive(Clone)]
enum EmbedderState {
    /// Never asked for. The first request moves this to `Loading`.
    Cold,
    Loading {
        since: Instant,
    },
    Ready(Arc<Mutex<Embedder>>),
    /// The last load failed. Kept — a caller is owed the reason — but not
    /// terminal: the next request re-arms, so a transient failure (a cache not
    /// yet populated, a disk hiccup) does not disable search until restart.
    Failed(String),
}

/// What a caller gets when it asks for the embedder. Never blocks on a load.
enum Acquired {
    Ready(Arc<Mutex<Embedder>>),
    Warming {
        warming_ms: u64,
    },
    /// The previous load failed. Returning it also ARMS a fresh attempt, so the
    /// reason is history rather than a standing verdict.
    Unavailable {
        reason: String,
    },
}

impl QueryEmbedder {
    fn new(config_path: Option<PathBuf>) -> Self {
        Self {
            config_path,
            state: RwLock::new(EmbedderState::Cold),
        }
    }

    async fn acquire(self: &Arc<Self>) -> Acquired {
        // Fast path: a loaded embedder must not queue behind a write lock.
        if let EmbedderState::Ready(embedder) = &*self.state.read().await {
            return Acquired::Ready(embedder.clone());
        }
        let mut state = self.state.write().await;
        match state.clone() {
            EmbedderState::Ready(embedder) => Acquired::Ready(embedder),
            EmbedderState::Loading { since } => Acquired::Warming {
                warming_ms: since.elapsed().as_millis() as u64,
            },
            // `Loading` is claimed under the write lock, which is what keeps a
            // burst of first requests to exactly one load attempt.
            previous @ (EmbedderState::Cold | EmbedderState::Failed(_)) => {
                *state = EmbedderState::Loading {
                    since: Instant::now(),
                };
                drop(state);
                self.clone().spawn_load();
                match previous {
                    EmbedderState::Failed(reason) => Acquired::Unavailable { reason },
                    _ => Acquired::Warming { warming_ms: 0 },
                }
            }
        }
    }

    fn spawn_load(self: Arc<Self>) {
        let config_path = self.config_path.clone();
        tokio::spawn(async move {
            let loaded = tokio::task::spawn_blocking(move || {
                let started = Instant::now();
                let config = Config::load(config_path.as_deref())?;
                let embedder = Embedder::new(&config)?;
                Ok::<_, anyhow::Error>((embedder, started.elapsed()))
            })
            .await;
            let next = match loaded {
                Ok(Ok((embedder, elapsed))) => {
                    tracing::info!(
                        load_ms = elapsed.as_millis() as u64,
                        "query embedding model loaded; semantic search is ready"
                    );
                    EmbedderState::Ready(Arc::new(Mutex::new(embedder)))
                }
                Ok(Err(error)) => {
                    let detail = format!("{error:#}");
                    tracing::warn!(error = %detail, "loading the query embedding model failed");
                    EmbedderState::Failed(detail)
                }
                Err(error) => EmbedderState::Failed(format!("embedder load task failed: {error}")),
            };
            *self.state.write().await = next;
        });
    }
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
    use super::{build_settings, read_only, requested_paths, root_name, valid_output_name, Corpus};
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

    /// Reproduce what a writer killed mid-transaction leaves on disk: a
    /// database plus the rollback journal describing the transaction that never
    /// finished. Copying both out from under a live transaction gives a pair no
    /// connection holds locks on, which is exactly the state SQLite calls hot.
    fn corpus_with_hot_journal(directory: &Path) -> std::path::PathBuf {
        let source = directory.join("source.sqlite");
        let connection = rusqlite::Connection::open(&source).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE files(id INTEGER PRIMARY KEY, path TEXT);
                 INSERT INTO files(id,path) VALUES (1,'/a/committed.txt');",
            )
            .unwrap();
        // A tiny page cache forces the open transaction to spill to disk, which
        // is what actually writes the journal header and dirties the database.
        // Without a spill nothing has left the cache, the journal is inert, and
        // the pair copied below would not be the hot state this test is about.
        connection.execute_batch("PRAGMA cache_size = 10").unwrap();
        connection.execute_batch("BEGIN IMMEDIATE").unwrap();
        for id in 2..400 {
            connection
                .execute(
                    "INSERT INTO files(id,path) VALUES (?1,?2)",
                    rusqlite::params![id, format!("/a/mid_{id}_{}", "x".repeat(600))],
                )
                .unwrap();
        }

        let killed = directory.join("killed.sqlite");
        std::fs::copy(&source, &killed).unwrap();
        std::fs::copy(
            crate::store::journal_path(&source),
            crate::store::journal_path(&killed),
        )
        .unwrap();
        drop(connection);
        killed
    }

    #[test]
    fn a_hot_journal_is_recovered_rather_than_served_as_unreadable() {
        let temp = tempfile::tempdir().unwrap();
        let killed = corpus_with_hot_journal(temp.path());
        // The premise: a read-only connection cannot replay a journal, so this
        // is the state in which the corpus routes used to answer "0 files".
        assert!(
            read_only(&killed).is_err(),
            "a hot journal must defeat a plain read-only open"
        );

        let Corpus::Ready(connection) = super::open_ro(&killed) else {
            panic!("a hot journal should be recovered on open")
        };
        // Recovery rolls the unfinished transaction back: what was committed
        // before the writer died is there, what it was mid-way through is not.
        let paths: Vec<String> = connection
            .prepare("SELECT path FROM files ORDER BY id")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(paths, vec!["/a/committed.txt".to_string()]);
        assert!(
            !crate::store::journal_path(&killed).exists(),
            "recovery clears the journal"
        );
    }

    /// The read-while-writing case in-place writing exists to enable: a healthy
    /// corpus under an active writer must read as BUSY (retry), never as
    /// unreadable, which would signal corruption during ordinary indexing.
    #[test]
    fn a_locked_corpus_reads_as_busy_not_unreadable() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("locked.sqlite");
        let writer = rusqlite::Connection::open(&path).unwrap();
        writer
            .execute_batch("CREATE TABLE files(id INTEGER PRIMARY KEY, path TEXT);")
            .unwrap();
        // Hold the lock the way a batch commit does, forcing the page cache to
        // spill so the lock escalates past RESERVED to EXCLUSIVE.
        writer.execute_batch("PRAGMA cache_size = 10").unwrap();
        writer.execute_batch("BEGIN IMMEDIATE").unwrap();
        for id in 1..400 {
            writer
                .execute(
                    "INSERT INTO files(id,path) VALUES (?1,?2)",
                    rusqlite::params![id, format!("/a/f_{id}_{}", "x".repeat(600))],
                )
                .unwrap();
        }

        let started = std::time::Instant::now();
        let state = super::open_ro(&path);
        let elapsed = started.elapsed();
        match state {
            Corpus::Busy => {}
            Corpus::Unreadable(error) => {
                panic!("a writer's lock must not be reported as damage: {error}")
            }
            Corpus::Ready(_) => panic!("the writer holds an exclusive lock"),
            Corpus::Absent => panic!("the database exists"),
        }
        // Classifying it correctly is only half the fix. Falling through to
        // journal recovery — a read-write open the same writer also blocks —
        // costs a second full timeout, so a status poll during indexing hangs
        // for a minute before answering. Busy must be recognised on the FIRST
        // probe, bounding this by READ_BUSY_TIMEOUT rather than two
        // writer-length waits. Without that, this takes ~47s.
        assert!(
            elapsed < crate::store::BUSY_TIMEOUT,
            "a locked corpus must be reported promptly, took {elapsed:?}"
        );

        // The same corpus reads normally once the writer is done.
        writer.execute_batch("COMMIT").unwrap();
        drop(writer);
        assert!(matches!(super::open_ro(&path), Corpus::Ready(_)));
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
        for category in ["detectors", "taggers", "captioners", "faces"] {
            let list = value["vision"][category].as_array().unwrap();
            assert_eq!(list.len(), 1, "{category}");
            assert!(list[0]["id"].is_string());
            assert!(list[0]["present"].is_boolean());
        }
        assert!(value["vision"]["defaults"]["detector_conf"].is_number());
        // Faces is discoverable and its advertised default is `off` — an app
        // reading this can offer the control without ever pre-selecting it.
        assert_eq!(value["vision"]["faces"][0]["id"], "yunet-sface");
        assert_eq!(value["vision"]["defaults"]["faces"], "off");
        assert!(value["vision"]["defaults"]["face_score"].is_number());
        assert!(value["vision"]["defaults"]["max_faces"].is_number());
    }

    /// The capability half of the faces opt-in: a box that has not staged the
    /// pair says so, a box with a wrongly-hashed one still says so, and neither
    /// changes which vision TIERS are on offer.
    #[test]
    fn faces_presence_is_reported_without_touching_the_tier_list() {
        let temp = tempfile::tempdir().unwrap();
        let config = config_pointing_at(temp.path());
        let vision_dir = temp.path().join("vision");
        std::fs::create_dir_all(&vision_dir).unwrap();

        let value = build_settings(Some(&config), VisionMode::Captions, 4).unwrap();
        assert_eq!(value["vision"]["faces"][0]["present"], false);
        let tiers = value["vision"]["tiers_available"].clone();

        // Half a pair is not a capability.
        std::fs::write(vision_dir.join("yunet.onnx"), b"bogus").unwrap();
        let value = build_settings(Some(&config), VisionMode::Captions, 4).unwrap();
        assert_eq!(value["vision"]["faces"][0]["present"], false);
        // Both halves present but unpinned-hash bogus: still absent.
        std::fs::write(vision_dir.join("sface.onnx"), b"bogus").unwrap();
        let value = build_settings(Some(&config), VisionMode::Captions, 4).unwrap();
        assert_eq!(value["vision"]["faces"][0]["present"], false);
        assert_eq!(
            value["vision"]["tiers_available"], tiers,
            "face models must never move the tier gates"
        );
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
    fn valid_output_name_reserves_jobs_sqlite_for_the_job_store() {
        assert!(!valid_output_name("jobs.sqlite"));
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

    /// Semantic search below the HTTP layer: `semantic_scan` is where "no
    /// results" has to become a stated reason, and it is reachable without the
    /// embedding model because the query vector is already an argument.
    mod semantic {
        use super::super::{search_response, semantic_scan, ReadError, SearchOutcome, FAST_MODE};
        use crate::embedding::{vector_to_bytes, EMBEDDING_MODEL};
        use rusqlite::Connection;
        use serde_json::Value;
        use std::path::Path;
        use std::time::Instant;

        /// A corpus with `files` + `chunks`, holding one chunk per `(model,
        /// vector)`. `chunks: None` writes a corpus with no chunks TABLE at all
        /// — what a build older than embeddings left behind.
        fn corpus(path: &Path, chunks: Option<&[(&str, Vec<f32>)]>) {
            crate::vec0::register();
            let connection = Connection::open(path).unwrap();
            connection
                .execute_batch(
                    "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                     CREATE TABLE files(id INTEGER PRIMARY KEY, path TEXT UNIQUE, name TEXT);
                     INSERT INTO files(id,path,name) VALUES(1,'/corpus/a.txt','a.txt');",
                )
                .unwrap();
            let Some(chunks) = chunks else { return };
            connection
                .execute_batch(
                    "CREATE TABLE chunks(id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL,
                       chunk_index INTEGER NOT NULL, content TEXT NOT NULL,
                       embedding BLOB NOT NULL, dimensions INTEGER NOT NULL, model TEXT NOT NULL,
                       page_start INTEGER, page_end INTEGER);",
                )
                .unwrap();
            for (index, (model, vector)) in chunks.iter().enumerate() {
                connection
                    .execute(
                        "INSERT INTO chunks(file_id,chunk_index,content,embedding,dimensions,model) \
                         VALUES(1,?1,?2,?3,?4,?5)",
                        rusqlite::params![
                            index as i64,
                            format!("chunk {index}"),
                            vector_to_bytes(vector),
                            vector.len() as i64,
                            model
                        ],
                    )
                    .unwrap();
            }
        }

        fn scan(path: &Path) -> SearchOutcome {
            rank(path, false)
        }

        /// `mode=semantic_fast`: the same request, routed at the quantised
        /// index instead of the exact one.
        fn fast(path: &Path) -> SearchOutcome {
            rank(path, true)
        }

        fn rank(path: &Path, fast: bool) -> SearchOutcome {
            match semantic_scan(path, &[1.0, 0.0, 0.0], 5, fast, Instant::now()) {
                Ok(outcome) => outcome,
                Err(ReadError::Busy) => panic!("a fixture corpus cannot be busy"),
                Err(ReadError::Unreadable(detail)) => panic!("fixture unreadable: {detail}"),
            }
        }

        fn body(outcome: SearchOutcome) -> Value {
            search_response("beach at sunset", "semantic", 5, outcome)
        }

        fn fast_body(outcome: SearchOutcome) -> Value {
            search_response("beach at sunset", FAST_MODE, 5, outcome)
        }

        #[test]
        fn an_absent_corpus_answers_empty_with_a_reason_not_an_error() {
            let temp = tempfile::tempdir().unwrap();
            let body = body(scan(&temp.path().join("corpus.sqlite")));
            assert_eq!(body["status"], "no_embeddings");
            assert_eq!(body["hits"].as_array().unwrap().len(), 0);
            assert!(
                body["reason"]
                    .as_str()
                    .unwrap()
                    .contains("no corpus database"),
                "{body}"
            );
            // The honest fields a caller branches on are there either way.
            assert_eq!(body["mode"], "semantic");
            assert_eq!(body["model"], EMBEDDING_MODEL);
            assert_eq!(body["query"], "beach at sunset");
        }

        #[test]
        fn a_corpus_without_a_chunks_table_degrades_rather_than_failing() {
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("corpus.sqlite");
            corpus(&path, None);
            let body = body(scan(&path));
            assert_eq!(body["status"], "no_embeddings");
            assert!(
                body["reason"].as_str().unwrap().contains("no chunks table"),
                "{body}"
            );
        }

        #[test]
        fn a_corpus_indexed_with_embedding_off_says_so() {
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("corpus.sqlite");
            corpus(&path, Some(&[]));
            let body = body(scan(&path));
            assert_eq!(body["status"], "no_embeddings");
            assert!(
                body["reason"].as_str().unwrap().contains("no embeddings"),
                "{body}"
            );
            assert!(body.get("other_models").is_none(), "{body}");
        }

        #[test]
        fn a_corpus_embedded_by_another_model_names_it() {
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("corpus.sqlite");
            corpus(
                &path,
                Some(&[
                    ("clip-vit-b32", vec![1.0, 0.0, 0.0]),
                    ("clip-vit-b32", vec![0.9, 0.1, 0.0]),
                ]),
            );
            let body = body(scan(&path));
            assert_eq!(body["status"], "no_embeddings");
            assert_eq!(
                body["other_models"],
                serde_json::json!(["clip-vit-b32 (3d)"])
            );
            let reason = body["reason"].as_str().unwrap();
            assert!(reason.contains("another model"), "{reason}");
            assert!(reason.contains(EMBEDDING_MODEL), "{reason}");
        }

        #[test]
        fn a_ranked_scan_reports_hits_scores_and_what_it_compared() {
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("corpus.sqlite");
            corpus(
                &path,
                Some(&[
                    (EMBEDDING_MODEL, vec![0.0, 1.0, 0.0]),
                    (EMBEDDING_MODEL, vec![1.0, 0.0, 0.0]),
                    ("clip-vit-b32", vec![1.0, 0.0, 0.0]),
                ]),
            );
            let body = body(scan(&path));
            assert_eq!(body["status"], "ready");
            assert_eq!(body["compared_chunks"], 2);
            assert_eq!(body["skipped_chunks"], 1);
            let hits = body["hits"].as_array().unwrap();
            assert_eq!(hits.len(), 2);
            assert_eq!(hits[0]["content"], "chunk 1");
            assert_eq!(hits[0]["path"], "/corpus/a.txt");
            assert!((hits[0]["score"].as_f64().unwrap() - 1.0).abs() < 0.0001);
            assert!(hits[1]["score"].as_f64().unwrap().abs() < 0.0001);
            assert!(body["elapsed_ms"].is_number(), "{body}");
            // No shadow index, so the scan served it — stated, not implied, and
            // with nothing to say about an index that is not there.
            assert_eq!(body["path"], "scan");
            assert!(body.get("index_note").is_none(), "{body}");
        }

        /// The same corpus after `llm-index vector-index --tier TIER`.
        fn with_shadow_index(path: &Path, tier: crate::vec0::Tier, dimensions: usize) {
            let mut connection = Connection::open(path).unwrap();
            crate::vec0::build(
                &mut connection,
                tier,
                EMBEDDING_MODEL,
                dimensions,
                |_, _| {},
            )
            .unwrap();
        }

        #[test]
        fn a_shadow_index_answers_the_same_request_and_the_response_says_so() {
            // Same fixture, same query, same hits and scores as the scan test
            // above — the only difference a consumer can see is `path`.
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("corpus.sqlite");
            corpus(
                &path,
                Some(&[
                    (EMBEDDING_MODEL, vec![0.0, 1.0, 0.0]),
                    (EMBEDDING_MODEL, vec![1.0, 0.0, 0.0]),
                    ("clip-vit-b32", vec![1.0, 0.0, 0.0]),
                ]),
            );
            with_shadow_index(&path, crate::vec0::Tier::Float, 3);
            let body = body(scan(&path));
            assert_eq!(body["status"], "ready");
            assert_eq!(body["path"], "vec0");
            assert_eq!(body["exact"], true);
            assert_eq!(body["compared_chunks"], 2);
            assert_eq!(body["skipped_chunks"], 1);
            let hits = body["hits"].as_array().unwrap();
            assert_eq!(hits.len(), 2);
            assert_eq!(hits[0]["content"], "chunk 1");
            assert!((hits[0]["score"].as_f64().unwrap() - 1.0).abs() < 0.0001);
            assert!(body.get("index_note").is_none(), "{body}");
        }

        #[test]
        fn semantic_fast_ranks_through_the_quantised_index_and_labels_itself() {
            // The opt-in path as a consumer sees it: same envelope, same score
            // arithmetic, a `path` naming the quantisation and `exact: false`
            // so nothing has to infer the promise from the path name.
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("corpus.sqlite");
            corpus(
                &path,
                Some(&[
                    (EMBEDDING_MODEL, vec![0.0, 1.0, 0.0]),
                    (EMBEDDING_MODEL, vec![1.0, 0.0, 0.0]),
                    ("clip-vit-b32", vec![1.0, 0.0, 0.0]),
                ]),
            );
            with_shadow_index(&path, crate::vec0::Tier::Int8, 3);
            let body = fast_body(fast(&path));
            assert_eq!(body["mode"], FAST_MODE);
            assert_eq!(body["status"], "ready");
            assert_eq!(body["path"], "vec0_int8");
            assert_eq!(body["exact"], false);
            // The pool is `limit x` the measured oversample, bounded by what
            // the corpus actually holds: three chunks, two of them this model's.
            assert_eq!(body["candidates"], 2);
            let hits = body["hits"].as_array().unwrap();
            assert_eq!(hits[0]["content"], "chunk 1");
            // The score is the float cosine, not a quantised distance: the
            // quantisation chooses the candidates and never the numbers.
            assert!((hits[0]["score"].as_f64().unwrap() - 1.0).abs() < 0.0001);
            assert!(body.get("index_note").is_none(), "{body}");
        }

        #[test]
        fn semantic_fast_over_a_corpus_without_one_answers_exactly_and_says_so() {
            // The fallback that matters most: asking for the fast path on a
            // corpus that has no quantised index returns the EXACT answer, and
            // labels it exact rather than pretending the request was served.
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("corpus.sqlite");
            corpus(
                &path,
                Some(&[
                    (EMBEDDING_MODEL, vec![0.0, 1.0, 0.0]),
                    (EMBEDDING_MODEL, vec![1.0, 0.0, 0.0]),
                ]),
            );
            let body = fast_body(fast(&path));
            assert_eq!(body["status"], "ready");
            assert_eq!(body["path"], "scan");
            assert_eq!(body["exact"], true);
            assert!(
                body["index_note"].as_str().unwrap().contains("--tier int8"),
                "{body}"
            );
            assert_eq!(body["hits"][0]["content"], "chunk 1");
        }

        #[test]
        fn a_corpus_whose_index_cannot_be_trusted_says_which_path_served_it() {
            // The capability fallback as a consumer sees it: still `ready`, still
            // the right hits, but `path: scan` with the reason attached — the
            // difference between "no index here" and "the index is stale".
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("corpus.sqlite");
            corpus(&path, Some(&[(EMBEDDING_MODEL, vec![1.0, 0.0, 0.0])]));
            with_shadow_index(&path, crate::vec0::Tier::Float, 3);
            Connection::open(&path)
                .unwrap()
                .execute(
                    "INSERT INTO chunks(file_id,chunk_index,content,embedding,dimensions,model) \
                     VALUES(1,9,'written behind the index',?1,3,?2)",
                    rusqlite::params![vector_to_bytes(&[1.0, 0.0, 0.0]), EMBEDDING_MODEL],
                )
                .unwrap();

            let body = body(scan(&path));
            assert_eq!(body["status"], "ready");
            assert_eq!(body["path"], "scan");
            assert!(
                body["index_note"].as_str().unwrap().contains("stale"),
                "{body}"
            );
            // And the row the index never saw is in the answer.
            let hits = body["hits"].as_array().unwrap();
            assert_eq!(hits.len(), 2);
            assert!(hits
                .iter()
                .any(|hit| hit["content"] == "written behind the index"));
        }

        #[test]
        fn warming_and_unavailable_are_stated_never_hidden_behind_zero_hits() {
            let warming = body(SearchOutcome::Warming { warming_ms: 40 });
            assert_eq!(warming["status"], "warming");
            assert_eq!(warming["warming_ms"], 40);
            assert_eq!(warming["hits"].as_array().unwrap().len(), 0);
            assert!(warming["reason"].as_str().unwrap().contains("loading"));

            let broken = body(SearchOutcome::Unavailable {
                reason: "no model cache".into(),
                retrying: true,
            });
            assert_eq!(broken["status"], "unavailable");
            assert_eq!(broken["reason"], "no model cache");
            assert_eq!(broken["retrying"], true);
            assert_eq!(broken["hits"].as_array().unwrap().len(), 0);
        }
    }
}
