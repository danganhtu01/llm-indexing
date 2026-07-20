//! Live, per-stage concurrency knobs.
//!
//! Every stage this module exposes is tunable WHILE A JOB IS RUNNING. That is
//! the whole point: the settings are read at the moment work is admitted, never
//! captured once at job start, so a change reaches work already in flight
//! instead of waiting for the next job.
//!
//! The settable stage names are EXACTLY the three the HTTP contract shares with
//! the other engines: `extract`, `embed`, `ocr`. Nothing else is advertised by
//! `GET /runtime` or accepted by `POST /runtime`, because the app in front of
//! these engines validates the stage set strictly against that same list —
//! advertising a fourth name over GET that PUT then rejects is worse than not
//! having the knob, since the app's rejection discards the caller's WHOLE body,
//! including the stages it did understand.
//!
//! ONNX intra-op width (`Config::embed_intra_threads`) is therefore a
//! CONFIG-ONLY setting, not a runtime stage. It never could have been live —
//! ort bakes the thread count into a `Session` at construction — so exposing it
//! on a surface whose entire premise is "this reaches work in flight" bought
//! nothing. [`RuntimeKnobs`] still carries the resolved value so the embedder
//! pool can read it when it builds an instance; it is simply not settable over
//! HTTP.
//!
//! Two coordination primitives live here alongside the settings themselves:
//!
//! * [`Admission`] gates how many extract workers may run concurrently out of a
//!   rayon pool that is always built at [`MAX_WORKERS`]. Lowering the target
//!   parks the surplus workers; it does not tear the pool down.
//! * [`EmbedderPool`] hands out a bounded set of [`Embedder`] instances.
//!   `fastembed`'s `embed` takes `&mut self`, so instances cannot be shared —
//!   concurrency is bounded by how many exist, which makes the pool size itself
//!   the live knob.
//!
//! Both park on a [`Condvar`] with a short timeout rather than spinning. The
//! timeout is a correctness backstop, not the primary wakeup path: a setter
//! notifies every subscribed condvar, but it cannot hold the subscriber's mutex
//! while doing so, so a notify can race with a thread that has evaluated its
//! predicate and not yet slept. [`POLL`] bounds how long such a lost wakeup can
//! delay a change.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

use anyhow::Result;
use serde_json::{json, Map, Value};

use crate::config::{Config, MAX_WORKERS};
use crate::embedding::{EmbeddedChunk, Embedder};

/// Canonical stage name: extract-side concurrency (rayon admission).
pub const EXTRACT: &str = "extract";
/// Canonical stage name: number of live [`Embedder`] instances.
pub const EMBED: &str = "embed";
/// Canonical stage name: `OMP_THREAD_LIMIT` handed to each tesseract spawn.
pub const OCR: &str = "ocr";

/// Every settable stage name, in the order `GET /runtime` reports them. Also the
/// list a `400` names when a caller sends something else.
///
/// This is the shared contract's llm-engine set, verbatim and complete. Adding
/// an engine-local fourth name here is not a harmless extension: the app merges
/// every stage an engine reports into its Settings UI but validates writes
/// against its own copy of this list, so an extra name renders as a control
/// whose save fails — and fails the entire body with it.
pub const STAGES: &[&str] = &[EXTRACT, EMBED, OCR];

pub const EXTRACT_RANGE: (usize, usize) = (1, MAX_WORKERS);
/// Each instance is ~448 MB of resident weights, so the ceiling is deliberately
/// far below [`MAX_WORKERS`]: 8 concurrent embedders is already ~3.5 GB.
pub const EMBED_RANGE: (usize, usize) = (1, 8);
pub const OCR_RANGE: (usize, usize) = (1, 64);
/// ort's own practical ceiling for intra-op parallelism on this model. Bounds
/// the CONFIG value (see the module header); not a settable stage.
pub const EMBED_INTRA_THREADS_RANGE: (usize, usize) = (1, 8);

/// Default live embedder count. Conservative on purpose — see [`EMBED_RANGE`].
/// The pool also grows LAZILY, so a job that never embeds two documents at once
/// only ever builds one instance and this default costs nothing.
pub const DEFAULT_EMBED: usize = 2;

/// How long a parked worker sleeps before re-reading its target. Short enough
/// that a lost notify still lands "within a few seconds", long enough that 64
/// parked threads cost no measurable CPU.
const POLL: Duration = Duration::from_millis(200);

fn clamp(value: usize, range: (usize, usize)) -> usize {
    value.clamp(range.0, range.1)
}

/// What tesseract's OpenMP would already have used, had this knob not existed.
///
/// Defaulting to this rather than to `1` is what keeps the OCR stage from being
/// a silent behaviour change: passing the value OpenMP would have chosen on its
/// own is a no-op, so an untouched deployment OCRs exactly as it did before,
/// while an operator who wants to stop N extract workers each fanning out to
/// every core can now say so.
///
/// An `OMP_THREAD_LIMIT` already in the environment wins over the CPU count.
/// Since this stage sets that variable on the child, ignoring an operator's
/// existing value would not be a no-op at all — it would silently OVERRIDE
/// their tuning with a larger number the first time they upgraded.
fn default_ocr_threads() -> usize {
    std::env::var("OMP_THREAD_LIMIT")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|limit| *limit > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
}

/// Process-wide or per-job stage settings.
///
/// Cloning is deliberately not offered: these are shared through an [`Arc`] so
/// that a `POST /jobs/{id}/runtime` reaches the very atomics the running job's
/// workers are reading. A copy would silently become a no-op.
#[derive(Debug)]
pub struct RuntimeKnobs {
    extract: AtomicUsize,
    embed: AtomicUsize,
    ocr: AtomicUsize,
    /// Resolved from [`Config`] once and never written again — the HTTP surface
    /// cannot reach it. Kept here rather than read from the `Config` directly so
    /// the embedder pool has a single place to consult for stage-ish numbers.
    embed_intra_threads: AtomicUsize,
    /// Condvars belonging to [`Admission`]/[`EmbedderPool`] instances driven by
    /// these settings, woken on every change so a raise takes effect at once.
    subscribers: Mutex<Vec<Arc<Condvar>>>,
}

impl RuntimeKnobs {
    /// Seed from a [`Config`], preserving today's effective behaviour exactly:
    /// `extract` from `workers`, ONNX width from the same `workers`-derived
    /// value it used to be hardcoded to, OCR from the OpenMP default.
    pub fn from_config(config: &Config) -> Self {
        Self {
            extract: AtomicUsize::new(clamp(config.workers, EXTRACT_RANGE)),
            embed: AtomicUsize::new(clamp(config.embed_workers, EMBED_RANGE)),
            ocr: AtomicUsize::new(clamp(
                config.ocr_threads.unwrap_or_else(default_ocr_threads),
                OCR_RANGE,
            )),
            embed_intra_threads: AtomicUsize::new(clamp(
                config.resolved_embed_intra_threads(),
                EMBED_INTRA_THREADS_RANGE,
            )),
            subscribers: Mutex::new(Vec::new()),
        }
    }

    /// A detached copy of the CURRENT values — how a job takes its starting
    /// point from the process-wide defaults without aliasing them, so a later
    /// `POST /runtime` does not retroactively retune jobs already in flight.
    pub fn snapshot(&self) -> Self {
        Self {
            extract: AtomicUsize::new(self.extract()),
            embed: AtomicUsize::new(self.embed()),
            ocr: AtomicUsize::new(self.ocr()),
            embed_intra_threads: AtomicUsize::new(self.embed_intra_threads()),
            subscribers: Mutex::new(Vec::new()),
        }
    }

    pub fn extract(&self) -> usize {
        self.extract.load(Ordering::Relaxed)
    }
    pub fn embed(&self) -> usize {
        self.embed.load(Ordering::Relaxed)
    }
    pub fn ocr(&self) -> usize {
        self.ocr.load(Ordering::Relaxed)
    }
    pub fn embed_intra_threads(&self) -> usize {
        self.embed_intra_threads.load(Ordering::Relaxed)
    }

    /// Register a condvar to wake whenever any setting changes.
    fn subscribe(&self, condvar: Arc<Condvar>) {
        lock(&self.subscribers).push(condvar);
    }

    fn wake_subscribers(&self) {
        for condvar in lock(&self.subscribers).iter() {
            condvar.notify_all();
        }
    }

    /// Apply a `{"<stage>": <int>}` body.
    ///
    /// Out-of-range values are CLAMPED rather than rejected, and the caller sees
    /// what actually landed by reading the returned view. Unknown stage names
    /// are an error naming the valid set, and NOTHING is applied in that case —
    /// a body with one typo does not half-land.
    pub fn apply(&self, body: &Map<String, Value>) -> Result<Value, String> {
        let unknown: Vec<&str> = body
            .keys()
            .map(String::as_str)
            .filter(|key| !STAGES.contains(key))
            .collect();
        if !unknown.is_empty() {
            return Err(format!(
                "unknown stage(s): {}; valid stages: {}",
                unknown.join(", "),
                STAGES.join(", ")
            ));
        }
        for (key, value) in body {
            let Some(requested) = as_count(value) else {
                return Err(format!("stage {key} must be an integer"));
            };
            match key.as_str() {
                EXTRACT => self
                    .extract
                    .store(clamp(requested, EXTRACT_RANGE), Ordering::Relaxed),
                EMBED => self
                    .embed
                    .store(clamp(requested, EMBED_RANGE), Ordering::Relaxed),
                OCR => self.ocr.store(clamp(requested, OCR_RANGE), Ordering::Relaxed),
                _ => unreachable!("stage names were validated above"),
            }
        }
        self.wake_subscribers();
        Ok(self.view())
    }

    /// The `GET /runtime` body.
    ///
    /// `live` is a promise the caller is entitled to trust, so it is stated per
    /// stage rather than blanket-asserted:
    ///
    /// * `extract` — read by [`Admission`] before every file, so a change
    ///   retunes the job in flight.
    /// * `embed` — read by [`EmbedderPool`] on every checkout/checkin; shrink
    ///   drops instances as they come back, growth builds lazily.
    /// * `ocr` — NOT live. `OMP_THREAD_LIMIT` is resolved when tesseract is
    ///   SPAWNED, which happens once per file, so a change cannot reach the page
    ///   currently being recognised — a 900-page scan already running keeps the
    ///   width it started with. "next-file" is one of the boundaries the shared
    ///   contract enumerates for `live: false`, so that is what this reports.
    ///   Per-file is a fine granularity and a genuinely useful knob; it is just
    ///   not the promise `live: true` makes, and that flag is the one thing the
    ///   contract asks callers to trust.
    pub fn view(&self) -> Value {
        json!({"stages": {
            EXTRACT: {
                "value": self.extract(), "min": EXTRACT_RANGE.0, "max": EXTRACT_RANGE.1,
                "live": true, "unit": "threads",
            },
            EMBED: {
                "value": self.embed(), "min": EMBED_RANGE.0, "max": EMBED_RANGE.1,
                "live": true, "unit": "instances",
            },
            OCR: {
                "value": self.ocr(), "min": OCR_RANGE.0, "max": OCR_RANGE.1,
                "live": false, "applies": "next-file", "unit": "threads",
            },
        }})
    }
}

/// Accept any JSON integer, including a negative one — which then clamps up to
/// the stage minimum rather than being rejected, per the contract.
fn as_count(value: &Value) -> Option<usize> {
    if let Some(unsigned) = value.as_u64() {
        return Some(usize::try_from(unsigned).unwrap_or(usize::MAX));
    }
    value.as_i64().map(|signed| usize::try_from(signed).unwrap_or(0))
}

/// Take a lock, recovering rather than propagating a poisoned mutex. A worker
/// that panicked mid-file must not wedge the admission gate for the whole job.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

// ── extract admission ────────────────────────────────────────────────────────

/// Concurrency gate for the extract stage.
///
/// The rayon pool is built once at [`MAX_WORKERS`] and never rebuilt; this is
/// what actually decides how many of those threads are doing work. A thread that
/// cannot get a slot parks on the condvar — it does not spin, and it does not
/// hold a rayon worker busy.
#[derive(Debug)]
pub struct Admission {
    runtime: Arc<RuntimeKnobs>,
    admitted: Mutex<usize>,
    condvar: Arc<Condvar>,
}

impl Admission {
    pub fn new(runtime: Arc<RuntimeKnobs>) -> Arc<Self> {
        let condvar = Arc::new(Condvar::new());
        runtime.subscribe(condvar.clone());
        Arc::new(Self {
            runtime,
            admitted: Mutex::new(0),
            condvar,
        })
    }

    /// Wait for a slot. Returns `None` when `cancelled` fires first, which is
    /// how a cancelled job drains instead of deadlocking behind a low target.
    ///
    /// The target is re-read on every wakeup, so lowering it mid-job parks the
    /// surplus as each in-flight file finishes, and raising it admits more
    /// without restarting anything.
    pub fn acquire(&self, cancelled: &dyn Fn() -> bool) -> Option<AdmissionGuard<'_>> {
        let mut admitted = lock(&self.admitted);
        loop {
            if cancelled() {
                return None;
            }
            if *admitted < self.runtime.extract() {
                *admitted += 1;
                return Some(AdmissionGuard { admission: self });
            }
            let (next, _) = self
                .condvar
                .wait_timeout(admitted, POLL)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            admitted = next;
        }
    }

    /// Currently admitted workers — the observable concurrency, and what the
    /// regression tests assert against.
    pub fn in_flight(&self) -> usize {
        *lock(&self.admitted)
    }

    fn release(&self) {
        *lock(&self.admitted) -= 1;
        self.condvar.notify_one();
    }
}

/// Releases its admission slot on drop, so a panicking extract does not leak
/// concurrency out of the gate.
pub struct AdmissionGuard<'a> {
    admission: &'a Admission,
}

impl Drop for AdmissionGuard<'_> {
    fn drop(&mut self) {
        self.admission.release();
    }
}

// ── embedder pool ────────────────────────────────────────────────────────────

#[derive(Default)]
struct PoolState {
    idle: Vec<Embedder>,
    /// Instances that exist, whether idle or checked out. Reserved BEFORE the
    /// slow construction so two workers cannot both decide to build the last
    /// permitted instance.
    built: usize,
}

/// A bounded set of [`Embedder`] instances, sized by the live `embed` stage.
///
/// `fastembed`'s `TextEmbedding::embed` takes `&mut self`, so one model cannot
/// be shared across workers; N concurrent embeds require N independent models,
/// each ~448 MB resident. That is why the pool size IS the concurrency knob, and
/// why it grows lazily rather than eagerly to its target.
pub struct EmbedderPool {
    config: Config,
    runtime: Arc<RuntimeKnobs>,
    state: Mutex<PoolState>,
    condvar: Arc<Condvar>,
}

impl EmbedderPool {
    /// Seed the pool with an already-constructed instance.
    ///
    /// The caller builds that first embedder EAGERLY and early, before any
    /// destructive step: a cold model cache has to be fetched over the network,
    /// and that failure must happen while the previous corpus is still on disk.
    /// See the `overwrite` ordering in `run_index`.
    pub fn seeded(config: &Config, runtime: Arc<RuntimeKnobs>, first: Embedder) -> Arc<Self> {
        let condvar = Arc::new(Condvar::new());
        runtime.subscribe(condvar.clone());
        Arc::new(Self {
            config: config.clone(),
            runtime,
            state: Mutex::new(PoolState {
                idle: vec![first],
                built: 1,
            }),
            condvar,
        })
    }

    /// A pool at its target with every instance checked out, so any `checkout`
    /// must park.
    ///
    /// Test-only, and it holds no [`Embedder`] on purpose: one costs ~448 MB of
    /// resident weights and a model download, and the parking path this exists to
    /// exercise is reached precisely BECAUSE no instance is available. Building a
    /// real model would prove nothing extra and would make the test unrunnable
    /// on a cold cache.
    #[cfg(test)]
    pub(crate) fn saturated(config: &Config, runtime: Arc<RuntimeKnobs>) -> Arc<Self> {
        let condvar = Arc::new(Condvar::new());
        runtime.subscribe(condvar.clone());
        let built = runtime.embed();
        Arc::new(Self {
            config: config.clone(),
            runtime,
            state: Mutex::new(PoolState {
                idle: Vec::new(),
                built,
            }),
            condvar,
        })
    }

    /// Borrow an instance, building one if the target allows and none is idle.
    ///
    /// Returns `Ok(None)` when `cancelled` fires, which is the same drain
    /// discipline [`Admission::acquire`] uses and for the same reason: a worker
    /// parked here waiting for an instance it would only waste is the difference
    /// between a cancel landing now and a cancel landing after one more full
    /// model inference per worker. The check runs BEFORE a hand-out as well as
    /// in the wait loop — an idle instance being available is no reason to embed
    /// a document for a job that is over.
    ///
    /// Construction happens with the lock RELEASED: loading ~448 MB of weights
    /// takes seconds, and holding the pool lock across it would stall every
    /// other embed worker for the duration.
    pub fn checkout(&self, cancelled: &dyn Fn() -> bool) -> Result<Option<PooledEmbedder<'_>>> {
        loop {
            if cancelled() {
                return Ok(None);
            }
            let target = self.runtime.embed();
            let mut state = lock(&self.state);
            // Shrink: surplus instances are dropped as they become available.
            // Anything still checked out is dropped on check-in instead.
            let mut surplus = Vec::new();
            while state.built > target {
                match state.idle.pop() {
                    Some(extra) => {
                        state.built -= 1;
                        surplus.push(extra);
                    }
                    None => break,
                }
            }
            if let Some(embedder) = state.idle.pop() {
                drop(state);
                // Free the surplus outside the lock — dropping an ort session is
                // not instant and no other worker needs to wait behind it.
                drop(surplus);
                return Ok(Some(PooledEmbedder {
                    pool: self,
                    embedder: Some(embedder),
                }));
            }
            if state.built < target {
                state.built += 1;
                drop(state);
                drop(surplus);
                let intra = self.runtime.embed_intra_threads();
                return match Embedder::with_intra_threads(&self.config, intra) {
                    Ok(embedder) => Ok(Some(PooledEmbedder {
                        pool: self,
                        embedder: Some(embedder),
                    })),
                    Err(error) => {
                        // Give the reservation back, or the pool permanently
                        // believes it is at target and never retries.
                        lock(&self.state).built -= 1;
                        self.condvar.notify_all();
                        Err(error)
                    }
                };
            }
            drop(surplus);
            let (next, _) = self
                .condvar
                .wait_timeout(state, POLL)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            drop(next);
        }
    }

    fn checkin(&self, embedder: Embedder) {
        let mut state = lock(&self.state);
        if state.built > self.runtime.embed() {
            // Shrunk while this one was in use: retire it instead of parking it.
            state.built -= 1;
            drop(state);
            drop(embedder);
        } else {
            state.idle.push(embedder);
            drop(state);
        }
        self.condvar.notify_one();
    }
}

/// An [`Embedder`] borrowed from an [`EmbedderPool`], returned on drop.
pub struct PooledEmbedder<'a> {
    pool: &'a EmbedderPool,
    embedder: Option<Embedder>,
}

impl PooledEmbedder<'_> {
    pub fn embed_document(&mut self, content: &str) -> Result<Vec<EmbeddedChunk>> {
        self.embedder
            .as_mut()
            .expect("embedder is taken only in Drop")
            .embed_document(content)
    }
}

impl Drop for PooledEmbedder<'_> {
    fn drop(&mut self) {
        if let Some(embedder) = self.embedder.take() {
            self.pool.checkin(embedder);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn knobs() -> Arc<RuntimeKnobs> {
        Arc::new(RuntimeKnobs::from_config(&Config::default()))
    }

    fn body(value: Value) -> Map<String, Value> {
        value.as_object().expect("object body").clone()
    }

    #[test]
    fn defaults_track_config_workers_without_changing_behaviour() {
        let mut config = Config::default();
        config.workers = 12;
        let runtime = RuntimeKnobs::from_config(&config);
        assert_eq!(runtime.extract(), 12);
        // Unset `embed_intra_threads` still derives from `workers`, exactly as
        // the hardcoded `config.workers.clamp(1, 8)` used to.
        assert_eq!(runtime.embed_intra_threads(), 8);
        assert_eq!(runtime.embed(), DEFAULT_EMBED);
    }

    #[test]
    fn out_of_range_values_are_clamped_not_rejected() {
        let runtime = knobs();
        let view = runtime
            .apply(&body(json!({"extract": 9999, "embed": 0})))
            .expect("clamping never errors");
        assert_eq!(view["stages"]["extract"]["value"], json!(MAX_WORKERS));
        assert_eq!(view["stages"]["embed"]["value"], json!(EMBED_RANGE.0));
        // The response reports what actually landed, not what was asked for.
        assert_eq!(runtime.extract(), MAX_WORKERS);
    }

    #[test]
    fn negative_values_clamp_to_the_minimum() {
        let runtime = knobs();
        runtime
            .apply(&body(json!({"extract": -5})))
            .expect("clamped");
        assert_eq!(runtime.extract(), EXTRACT_RANGE.0);
    }

    #[test]
    fn unknown_stage_names_are_rejected_and_apply_nothing() {
        let runtime = knobs();
        let before = runtime.extract();
        let error = runtime
            .apply(&body(json!({"extract": 3, "gpu_layers": 4})))
            .expect_err("unknown stage");
        assert!(error.contains("gpu_layers"), "{error}");
        for stage in STAGES {
            assert!(error.contains(stage), "{error} should list {stage}");
        }
        assert_eq!(
            runtime.extract(),
            before,
            "a body with one bad name must not half-apply"
        );
    }

    #[test]
    fn ocr_is_reported_as_next_file_not_live() {
        // The contract's core promise: `live` is trustworthy. `OMP_THREAD_LIMIT`
        // is resolved when tesseract is SPAWNED — once per file — so a change
        // cannot reach the document currently being recognised. "next-file" is
        // exactly one of the boundaries the contract enumerates for live:false,
        // and claiming live:true here is the failure the flag exists to prevent:
        // an operator watching a 900-page scan would see a green "live" badge,
        // turn the stage down, and see nothing happen for 40 minutes.
        let view = knobs().view();
        assert_eq!(
            view["stages"]["ocr"]["live"],
            json!(false),
            "ocr lands at a boundary, so it must not claim to be live"
        );
        assert_eq!(view["stages"]["ocr"]["applies"], json!("next-file"));
        // The genuinely live stages are re-read by the admission gate and the
        // embedder pool as they work, so they claim liveness and must NOT also
        // name a boundary.
        for live in ["extract", "embed"] {
            assert_eq!(view["stages"][live]["live"], json!(true), "{live}");
            assert!(
                view["stages"][live].get("applies").is_none(),
                "{live} is live, so it must not claim a boundary"
            );
        }
    }

    #[test]
    fn only_the_shared_contract_stages_are_advertised_and_settable() {
        // The app in front of these engines merges every stage GET reports into
        // its Settings UI but validates writes against its own copy of the
        // canonical llm set. An engine-local extra name therefore renders as a
        // control whose save 400s — and the app's 400 discards the WHOLE body,
        // so an unrelated `scan` edit in the same visit is silently lost too.
        let view = knobs().view();
        let stages = view["stages"].as_object().expect("stages object");
        let mut names: Vec<&str> = stages.keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            ["embed", "extract", "ocr"],
            "GET /runtime must advertise exactly the shared llm stage set"
        );
        // Advertised and settable are the same set, in both directions.
        for stage in STAGES {
            assert!(
                stages.contains_key(*stage),
                "{stage} is settable but unadvertised"
            );
        }
        let error = knobs()
            .apply(&body(json!({"embed_intra_threads": 4})))
            .expect_err("a config-only knob is not a runtime stage");
        assert!(error.contains("embed_intra_threads"), "{error}");
    }

    #[test]
    fn intra_threads_still_resolves_from_config_though_it_is_not_a_stage() {
        // Dropping it from the HTTP surface must not drop the setting: the
        // embedder pool still reads it when it builds an instance.
        let mut config = Config::default();
        config.embed_intra_threads = Some(3);
        assert_eq!(RuntimeKnobs::from_config(&config).embed_intra_threads(), 3);
    }

    #[test]
    fn checkout_releases_parked_workers_on_cancellation() {
        use std::sync::atomic::AtomicBool;

        // The embed stage's half of the drain discipline. Without it the only
        // way a worker learns the job is over is a failed send — i.e. AFTER it
        // has checked out an embedder and embedded a whole document — so a
        // cancel waits for one full model inference per worker, `run_index`
        // cannot return, `store.finish()` is deferred, the job row sits in
        // `cancelling`, and the single-slot service worker loop blocks the whole
        // queue behind it.
        let runtime = knobs();
        runtime.apply(&body(json!({"embed": 1}))).expect("applied");
        let pool = EmbedderPool::saturated(&Config::default(), runtime);
        let cancel = Arc::new(AtomicBool::new(false));
        let (sender, receiver) = std::sync::mpsc::channel();

        // DETACHED, not scoped: against the unfixed pool this thread parks
        // forever, and a scoped join would turn a red test into a hung run.
        let worker_pool = pool.clone();
        let worker_cancel = cancel.clone();
        std::thread::spawn(move || {
            let cancelled = || worker_cancel.load(Ordering::Relaxed);
            let outcome = worker_pool.checkout(&cancelled);
            let _ = sender.send(matches!(outcome, Ok(None)));
        });

        // Merely being starved is not a reason to come back.
        assert!(
            receiver.recv_timeout(POLL * 3).is_err(),
            "checkout must keep waiting while the pool is only saturated"
        );
        cancel.store(true, Ordering::Relaxed);
        assert!(
            receiver
                .recv_timeout(Duration::from_secs(5))
                .expect("a parked worker must wake on cancellation, not on an instance"),
            "a cancelled checkout must report cancellation, not hand out an embedder"
        );
    }

    #[test]
    fn a_snapshot_does_not_alias_the_defaults() {
        // Per-job settings must diverge from the process-wide ones, or a later
        // POST /runtime would retune jobs already in flight.
        let defaults = knobs();
        let job = Arc::new(defaults.snapshot());
        job.apply(&body(json!({"extract": 3}))).expect("applied");
        assert_eq!(job.extract(), 3);
        assert_eq!(defaults.extract(), Config::default().workers);
    }

    #[test]
    fn admission_caps_concurrency_at_the_live_target() {
        use std::sync::atomic::AtomicBool;

        let runtime = knobs();
        runtime.apply(&body(json!({"extract": 2}))).expect("applied");
        let admission = Admission::new(runtime.clone());
        let never = AtomicBool::new(false);
        let cancelled = || never.load(Ordering::Relaxed);

        let first = admission.acquire(&cancelled).expect("slot");
        let second = admission.acquire(&cancelled).expect("slot");
        assert_eq!(admission.in_flight(), 2);

        // A third acquire must BLOCK, so prove the gate from another thread.
        // The waiter announces admission and then holds its slot until released,
        // rather than dropping it immediately: a waiter that acquired and
        // returned would leave `in_flight` back at 2 by the time this thread
        // looked, and the check below would pass against a gate that admits
        // everyone. The flag is the observation, not the counter.
        let admitted = AtomicBool::new(false);
        let release = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let handle = scope.spawn(|| {
                let guard = admission.acquire(&cancelled).expect("slot after raise");
                admitted.store(true, Ordering::SeqCst);
                // BOUNDED. `thread::scope` joins children before propagating a
                // panic, so an unbounded wait here would turn a failed assertion
                // in the parent into a hung test run instead of a red one.
                for _ in 0..2_000 {
                    if release.load(Ordering::SeqCst) {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                drop(guard);
            });
            std::thread::sleep(POLL * 3);
            assert!(
                !admitted.load(Ordering::SeqCst),
                "admission must not exceed the target: a third worker was let in \
                 while two slots of two were held"
            );
            assert_eq!(admission.in_flight(), 2);

            // Raising the target admits the waiter without restarting anything.
            runtime.apply(&body(json!({"extract": 3}))).expect("applied");
            let mut woke = false;
            for _ in 0..200 {
                if admitted.load(Ordering::SeqCst) {
                    woke = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            assert!(woke, "raising the target must admit the parked worker");
            assert_eq!(admission.in_flight(), 3);
            release.store(true, Ordering::SeqCst);
            handle.join().expect("waiter finished");
        });
        drop(first);
        drop(second);
        assert_eq!(admission.in_flight(), 0);
    }

    #[test]
    fn admission_releases_waiters_on_cancellation() {
        use std::sync::atomic::AtomicBool;

        let runtime = knobs();
        runtime.apply(&body(json!({"extract": 1}))).expect("applied");
        let admission = Admission::new(runtime);
        let cancel = AtomicBool::new(false);
        let cancelled = || cancel.load(Ordering::Relaxed);
        let held = admission.acquire(&cancelled).expect("slot");

        std::thread::scope(|scope| {
            let handle = scope.spawn(|| admission.acquire(&cancelled).is_none());
            std::thread::sleep(POLL);
            cancel.store(true, Ordering::Relaxed);
            assert!(
                handle.join().expect("waiter woke"),
                "a cancelled job must not deadlock behind a low target"
            );
        });
        drop(held);
    }
}
