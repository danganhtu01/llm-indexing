use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use llm_indexing::config::Config;
use llm_indexing::embedding::{vector_search, Embedder};
use llm_indexing::normalize::Normalizer;
use llm_indexing::pipeline::{run_index, IndexRequest};
use llm_indexing::service::{router, JobRequest, ServiceConfig};
use llm_indexing::settings::{tessdata_sources, OcrSettings, VisionSettings};
use llm_indexing::store::{analyze, connect, search, top_folders};
use llm_indexing::vision::{VisionMode, VISION_MODELS};
use llm_indexing::VERSION;
use serde_json::Value;

#[derive(Parser)]
#[command(name = "llm-index", version = VERSION,
          about = "Rust-native EN/VI full-text indexer with OCR and SQLite FTS5")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Index(IndexArgs),
    Search(SearchArgs),
    VectorSearch(VectorSearchArgs),
    TopFolder(TopFolderArgs),
    Analyze(AnalyzeArgs),
    Serve(ServeArgs),
    Request(RequestArgs),
    FetchData(FetchDataArgs),
    PrefetchModels(PrefetchModelsArgs),
}

#[derive(Debug, Clone, ValueEnum)]
enum OcrMode {
    Auto,
    On,
    Off,
    Exhaustive,
}
impl OcrMode {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
            Self::Exhaustive => "exhaustive",
        }
    }
}

#[derive(Args)]
struct IndexArgs {
    #[arg(required = true)]
    paths: Vec<PathBuf>,
    #[arg(long, default_value = "index_out")]
    out: PathBuf,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum)]
    ocr: Option<OcrMode>,
    #[arg(long)]
    ocr_langs: Option<String>,
    #[arg(long)]
    sidecar: Option<String>,
    #[arg(long)]
    workers: Option<usize>,
    #[arg(long)]
    max_bytes: Option<u64>,
    /// Vision analysis tier: off (default), meta, tags, or captions.
    #[arg(long, value_enum)]
    vision: Option<VisionMode>,
    #[arg(long)]
    resume: bool,
    // Per-job OCR quality overrides (feed the SAME settings merge as the HTTP
    // `ocr_opts`); language selection stays on the legacy `--ocr-langs` above.
    #[arg(long)]
    ocr_dpi: Option<u32>,
    #[arg(long)]
    ocr_psm: Option<String>,
    #[arg(long)]
    ocr_preprocess: Option<bool>,
    #[arg(long)]
    ocr_max_pages: Option<usize>,
    // Per-job vision overrides (feed the SAME settings merge as `vision_opts`).
    #[arg(long)]
    vision_detector: Option<String>,
    #[arg(long)]
    vision_detector_conf: Option<f32>,
    #[arg(long)]
    vision_tagger: Option<String>,
    #[arg(long)]
    vision_tag_threshold: Option<f32>,
    #[arg(long)]
    vision_tag_top_k: Option<usize>,
    #[arg(long)]
    vision_captioner: Option<String>,
    #[arg(long)]
    vision_max_frames: Option<usize>,
    #[arg(long)]
    vision_timeout_secs: Option<u64>,
}

#[derive(Args)]
struct SearchArgs {
    query: String,
    #[arg(long, default_value = "index_out")]
    index: PathBuf,
    #[arg(long, default_value_t = 20)]
    limit: usize,
    #[arg(long)]
    fuzzy: bool,
    #[arg(long)]
    config: Option<PathBuf>,
}

#[derive(Args)]
struct VectorSearchArgs {
    query: String,
    #[arg(long, default_value = "index_out/index.sqlite")]
    index: PathBuf,
    #[arg(long, default_value_t = 10)]
    limit: usize,
    #[arg(long)]
    config: Option<PathBuf>,
}

#[derive(Args)]
struct TopFolderArgs {
    query: String,
    #[arg(long, default_value = "index_out")]
    index: PathBuf,
    #[arg(short = 'n', long, default_value_t = 10)]
    limit: usize,
    #[arg(long)]
    config: Option<PathBuf>,
}

#[derive(Args)]
struct AnalyzeArgs {
    #[arg(long, default_value = "index_out")]
    index: PathBuf,
    #[arg(long)]
    json: Option<PathBuf>,
    #[arg(long)]
    markdown: Option<PathBuf>,
}

#[derive(Args)]
struct ServeArgs {
    #[arg(long, default_value = "0.0.0.0:9801")]
    listen: String,
    #[arg(long, default_value = "/output")]
    output_root: PathBuf,
    #[arg(long = "allowed-root")]
    allowed_roots: Vec<PathBuf>,
    #[arg(long = "default-path")]
    default_paths: Vec<PathBuf>,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, default_value = "vie+eng")]
    ocr_langs: String,
    #[arg(long, default_value_t = 4)]
    workers: usize,
    #[arg(long, default_value_t = 32)]
    max_pending: usize,
    #[arg(long, default_value_t = 1024 * 1024)]
    max_body: usize,
    /// Highest vision tier this server accepts (env fallback
    /// `INDEX_VISION_MAX`); requests above it are rejected. Default `off`.
    #[arg(long = "vision-max", value_enum)]
    vision_max: Option<VisionMode>,
}

#[derive(Args)]
struct RequestArgs {
    #[arg(long, default_value = "http://127.0.0.1:9801")]
    url: String,
    #[arg(long)]
    ping: bool,
    #[arg(long)]
    no_wait: bool,
    #[arg(long = "path")]
    paths: Vec<PathBuf>,
    #[arg(long, default_value = "corpus.sqlite")]
    output: String,
    #[arg(long, value_enum, default_value = "auto")]
    ocr: OcrMode,
    #[arg(long)]
    ocr_langs: Option<String>,
    #[arg(long)]
    workers: Option<usize>,
    /// Vision analysis tier: off (default), meta, tags, or captions.
    #[arg(long, value_enum, default_value = "off")]
    vision: VisionMode,
    #[arg(long)]
    resume: bool,
    #[arg(long)]
    overwrite: bool,
}

#[derive(Args)]
struct FetchDataArgs {
    #[arg(long, default_value = "data")]
    data_dir: PathBuf,
    #[arg(long)]
    force: bool,
    #[arg(long, conflicts_with = "ocr_only")]
    dictionaries_only: bool,
    #[arg(long, conflicts_with = "dictionaries_only")]
    ocr_only: bool,
    /// Fetch the vision models (RF-DETR-Nano detector / Florence-2) with pinned
    /// SHA-256 verification instead of dictionaries/OCR data.
    #[arg(long, conflicts_with_all = ["dictionaries_only", "ocr_only"])]
    vision: bool,
}

#[derive(Args)]
struct PrefetchModelsArgs {
    #[arg(long, default_value = "/app/models/fastembed")]
    embedding_cache: PathBuf,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Index(args) => index(args),
        Command::Search(args) => search_command(args),
        Command::VectorSearch(args) => vector_search_command(args),
        Command::TopFolder(args) => top_folder_command(args),
        Command::Analyze(args) => analyze_command(args),
        Command::Serve(args) => serve(args),
        Command::Request(args) => request(args),
        Command::FetchData(args) => fetch_data(args),
        Command::PrefetchModels(args) => prefetch_models(args),
    }
}

fn vector_search_command(args: VectorSearchArgs) -> Result<()> {
    let config = Config::load(args.config.as_deref())?;
    let hits = vector_search(&args.index, &config, &args.query, args.limit)?;
    println!("{}", serde_json::to_string_pretty(&hits)?);
    Ok(())
}

fn index(args: IndexArgs) -> Result<()> {
    let mut config = Config::load(args.config.as_deref())?;
    if let Some(ocr) = args.ocr {
        config.ocr = ocr.as_str().into()
    }
    if let Some(langs) = &args.ocr_langs {
        // Same installed-tessdata gate the HTTP `ocr_opts.langs` submit uses, so the
        // CLI cannot silently OCR every page empty with an uninstalled/cross-source
        // language selection.
        let (bundled, system) = tessdata_sources(&config);
        OcrSettings {
            langs: Some(langs.clone()),
            ..Default::default()
        }
        .validate_langs(&bundled, &system)
        .map_err(|error| anyhow::anyhow!(error))?;
        config.ocr_langs = langs.clone();
    }
    if let Some(sidecar) = args.sidecar {
        config.sidecar = sidecar
    }
    if let Some(workers) = args.workers {
        config.workers = workers
    }
    if let Some(max_bytes) = args.max_bytes {
        config.max_bytes = max_bytes
    }
    if let Some(vision) = args.vision {
        config.vision.max = vision
    }
    // Per-job OCR/vision quality knobs go through the SAME merge + validation
    // path as the HTTP `ocr_opts`/`vision_opts`, so CLI and service stay at
    // parity. `--ocr-langs` (legacy, above) remains the language selector.
    let ocr_over = OcrSettings {
        dpi: args.ocr_dpi,
        psm: args.ocr_psm.clone(),
        preprocess: args.ocr_preprocess,
        max_pages: args.ocr_max_pages,
        langs: None,
    };
    ocr_over
        .validate()
        .map_err(|error| anyhow::anyhow!(error))?;
    OcrSettings::resolve(&config, Some(&ocr_over)).apply_to(&mut config);
    let vision_over = VisionSettings {
        detector: args.vision_detector.clone(),
        detector_conf: args.vision_detector_conf,
        tagger: args.vision_tagger.clone(),
        tag_threshold: args.vision_tag_threshold,
        tag_top_k: args.vision_tag_top_k,
        captioner: args.vision_captioner.clone(),
        max_frames: args.vision_max_frames,
        timeout_secs: args.vision_timeout_secs,
    };
    vision_over
        .validate()
        .map_err(|error| anyhow::anyhow!(error))?;
    VisionSettings::resolve(&config, Some(&vision_over)).apply_to(&mut config);
    // Mirror the service's refusal: without --resume, indexing into an
    // existing corpus INSERT-OR-REPLACEs rows under new rowids, and stale FTS
    // text lingered as ghost search hits nothing ever cleaned up. (The vlm
    // CLI refuses the same way.)
    let database = llm_indexing::store::database_path(&args.out);
    if database.exists() && !args.resume {
        anyhow::bail!(
            "output {} already exists; pass --resume to continue it (or delete it first)",
            database.display()
        );
    }
    let stats = run_index(IndexRequest {
        paths: &args.paths,
        out: &args.out,
        config,
        resume: args.resume,
        overwrite: false,
        artifacts: true,
        include_paths: None,
        cancellation: None,
        runtime: None,
        progress: None,
    })?;
    println!("{}", serde_json::to_string_pretty(&stats)?);
    println!(
        "Index database: {}",
        args.out
            .canonicalize()
            .unwrap_or(args.out)
            .join("index.sqlite")
            .display()
    );
    Ok(())
}

fn normalizer(config: Option<&Path>) -> Result<Normalizer> {
    Ok(Normalizer::load(&Config::load(config)?))
}

fn search_command(args: SearchArgs) -> Result<()> {
    let connection = connect(&args.index)?;
    let normalizer = normalizer(args.config.as_deref())?;
    let hits = search(
        &connection,
        &normalizer,
        &args.query,
        args.limit,
        args.fuzzy,
    )?;
    for (i, hit) in hits.iter().enumerate() {
        println!(
            "{:>2}. {}\n    [{}/{}] {}",
            i + 1,
            hit.path,
            hit.lang,
            hit.method,
            hit.snippet
        );
    }
    let folders = top_folders(&connection, &normalizer, &args.query, args.limit)?;
    if let Some((folder, count)) = folders.first() {
        println!("\nFolder with most matches: {folder} ({count})");
    }
    Ok(())
}

fn top_folder_command(args: TopFolderArgs) -> Result<()> {
    let connection = connect(&args.index)?;
    let normalizer = normalizer(args.config.as_deref())?;
    for (folder, count) in top_folders(&connection, &normalizer, &args.query, args.limit)? {
        println!("{count:>6}  {folder}");
    }
    Ok(())
}

fn analyze_command(args: AnalyzeArgs) -> Result<()> {
    let connection = connect(&args.index)?;
    let value = analyze(&connection)?;
    let pretty = serde_json::to_string_pretty(&value)?;
    if let Some(path) = args.json {
        fs::write(path, &pretty)?
    }
    let markdown = format!(
        "# Index analysis\n\n- Files: {}\n- Bytes: {}\n- OCR files: {}\n",
        value["files"], value["bytes"], value["ocr_files"]
    );
    if let Some(path) = args.markdown {
        fs::write(path, &markdown)?
    }
    println!("{pretty}");
    Ok(())
}

fn serve(args: ServeArgs) -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let allowed_roots = if args.allowed_roots.is_empty() {
        env_paths("INDEX_ALLOWED_ROOTS", "/input")
    } else {
        args.allowed_roots
    };
    let default_paths = if args.default_paths.is_empty() {
        env_paths("INDEX_DEFAULT_PATHS", "/input")
    } else {
        args.default_paths
    };
    let vision_max = args
        .vision_max
        .or_else(|| {
            std::env::var("INDEX_VISION_MAX")
                .ok()
                .and_then(|value| value.parse().ok())
        })
        .unwrap_or(VisionMode::Off);
    let config = ServiceConfig {
        output_root: args.output_root,
        allowed_roots,
        default_paths,
        config_path: args.config,
        ocr_langs: args.ocr_langs,
        workers: args.workers,
        max_pending: args.max_pending,
        max_body: args.max_body,
        vision_max,
    };
    let address: SocketAddr = args.listen.parse().context("--listen must be HOST:PORT")?;
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move {
        let app = router(config)?;
        let listener = tokio::net::TcpListener::bind(address).await?;
        println!("llm-index listening on http://{}", listener.local_addr()?);
        axum::serve(listener, app).await?;
        Ok::<_, anyhow::Error>(())
    })
}

fn request(args: RequestArgs) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let base = args.url.trim_end_matches('/');
    if args.ping {
        let response = client
            .get(format!("{base}/health"))
            .send()?
            .error_for_status()?;
        println!(
            "{}",
            serde_json::to_string_pretty(&response.json::<Value>()?)?
        );
        return Ok(());
    }
    let payload = JobRequest {
        id: None,
        paths: (!args.paths.is_empty()).then_some(args.paths),
        output: args.output,
        ocr: args.ocr.as_str().into(),
        ocr_langs: args.ocr_langs,
        workers: args.workers,
        resume: args.resume,
        overwrite: args.overwrite,
        include_paths: None,
        vision: Some(args.vision.as_str().to_string()),
        ocr_opts: None,
        vision_opts: None,
    };
    let response = client
        .post(format!("{base}/index"))
        .json(&payload)
        .send()?
        .error_for_status()?;
    let queued = response.json::<Value>()?;
    if args.no_wait {
        println!("{}", serde_json::to_string_pretty(&queued)?);
        return Ok(());
    }
    let id = queued["id"]
        .as_str()
        .context("server response omitted job id")?;
    loop {
        let value = client
            .get(format!("{base}/jobs/{id}"))
            .send()?
            .error_for_status()?
            .json::<Value>()?;
        match value["status"].as_str() {
            Some("complete") => {
                println!("{}", serde_json::to_string_pretty(&value)?);
                return Ok(());
            }
            Some("error") => {
                anyhow::bail!("{}", value["error"].as_str().unwrap_or("indexing failed"))
            }
            _ => thread::sleep(Duration::from_millis(500)),
        }
    }
}

fn fetch_data(args: FetchDataArgs) -> Result<()> {
    if args.vision {
        return fetch_vision_models(&args);
    }
    const RAW: &str = "https://raw.githubusercontent.com";
    let files = [
        (
            "dict/en_US.dic",
            format!("{RAW}/wooorm/dictionaries/main/dictionaries/en/index.dic"),
            false,
        ),
        (
            "dict/en_US.aff",
            format!("{RAW}/wooorm/dictionaries/main/dictionaries/en/index.aff"),
            false,
        ),
        (
            "dict/vi_VN.dic",
            format!("{RAW}/wooorm/dictionaries/main/dictionaries/vi/index.dic"),
            false,
        ),
        (
            "dict/vi_VN.aff",
            format!("{RAW}/wooorm/dictionaries/main/dictionaries/vi/index.aff"),
            false,
        ),
        (
            "dict/vi_words.txt",
            format!("{RAW}/duyet/vietnamese-wordlist/master/Viet74K.txt"),
            false,
        ),
        (
            "tessdata/vie.traineddata",
            format!("{RAW}/tesseract-ocr/tessdata_best/main/vie.traineddata"),
            true,
        ),
        (
            "tessdata/eng.traineddata",
            format!("{RAW}/tesseract-ocr/tessdata_best/main/eng.traineddata"),
            true,
        ),
        (
            "tessdata/rus.traineddata",
            format!("{RAW}/tesseract-ocr/tessdata_best/main/rus.traineddata"),
            true,
        ),
        (
            "tessdata/deu.traineddata",
            format!("{RAW}/tesseract-ocr/tessdata_best/main/deu.traineddata"),
            true,
        ),
    ];
    let client = reqwest::blocking::Client::new();
    for (relative, url, is_ocr) in files {
        if args.dictionaries_only && is_ocr || args.ocr_only && !is_ocr {
            continue;
        }
        let destination = args.data_dir.join(relative);
        if destination.exists() && !args.force {
            continue;
        }
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?
        }
        let bytes = client.get(url).send()?.error_for_status()?.bytes()?;
        fs::write(&destination, &bytes)?;
        println!("{} {} bytes", destination.display(), bytes.len());
    }
    Ok(())
}

/// Fetch the vision models listed in `VISION_MODELS` into `<data_dir>/vision`,
/// verifying each against its pinned SHA-256 before writing. The tags-tier
/// detector (RF-DETR-Nano) is pinned and downloaded here; the captions-tier
/// Florence-2 files stay unpinned (`None`) while that tier is the v1 unsupported
/// stub, so they are reported as not-yet-pinned and skipped. The
/// verify-after-download path runs whenever a real hash is present.
fn fetch_vision_models(args: &FetchDataArgs) -> Result<()> {
    use sha2::{Digest, Sha256};
    let directory = args.data_dir.join("vision");
    let client = reqwest::blocking::Client::new();
    for model in VISION_MODELS {
        let destination = directory.join(model.relative);
        // Re-verify an already-present pinned file rather than trusting mere
        // existence. The atomic write below means an interrupted download never
        // lands here, but a file corrupted/truncated/swapped by other means is
        // caught and repaired by re-downloading. Unpinned files (the Florence
        // stub) keep the skip-if-present behaviour.
        if destination.exists() && !args.force {
            match model.sha256 {
                Some(expected) if file_sha256(&destination)?.eq_ignore_ascii_case(expected) => {
                    continue;
                }
                Some(_) => eprintln!(
                    "{} present but fails its pinned sha256 — re-fetching",
                    model.relative
                ),
                None => continue,
            }
        }
        let Some(url) = model.url else {
            eprintln!(
                "skipping {} — download URL not yet pinned (V3/V5 will pin it)",
                model.relative
            );
            continue;
        };
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?
        }
        let bytes = client.get(url).send()?.error_for_status()?.bytes()?;
        match model.sha256 {
            Some(expected) => {
                let actual = format!("{:x}", Sha256::digest(&bytes));
                if !actual.eq_ignore_ascii_case(expected) {
                    anyhow::bail!(
                        "sha256 mismatch for {}: expected {expected}, got {actual}",
                        model.relative
                    )
                }
            }
            None => eprintln!(
                "warning: {} has no pinned sha256 yet; skipping verification (V3/V5 will pin it)",
                model.relative
            ),
        }
        // Atomic: write a sibling temp file and rename in, so a crash mid-write
        // never leaves a truncated file that a later run would accept as done.
        write_atomic(&destination, &bytes)?;
        println!(
            "{} {} bytes ({})",
            destination.display(),
            bytes.len(),
            model.license
        );
    }
    // CLIP is served from fastembed's own cache (there is no single pinned file),
    // so stage it here — the ONLY sanctioned network fetch of CLIP (VISION-SPEC
    // §1) — so index-time tags jobs load it locally and the submit pre-flight can
    // require it instead of fastembed silently downloading it mid-job.
    println!("staging CLIP encoders under {} …", directory.display());
    llm_indexing::vision::prefetch_clip(&directory)?;
    println!("CLIP encoders staged under {}", directory.display());
    Ok(())
}

/// Stream a file's lowercase-hex SHA-256 (chunked, so a ~100 MB ONNX blob is not
/// loaded fully into memory).
fn file_sha256(path: &Path) -> Result<String> {
    use std::io::Read;

    use sha2::{Digest, Sha256};
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1 << 16];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Write `bytes` to `destination` atomically via a sibling temp file + rename, so
/// an interrupted write never leaves a partial file at `destination`.
fn write_atomic(destination: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let directory = destination.parent().unwrap_or_else(|| Path::new("."));
    let mut temp = tempfile::NamedTempFile::new_in(directory)?;
    temp.write_all(bytes)?;
    temp.flush()?;
    temp.persist(destination).map_err(|error| error.error)?;
    Ok(())
}

fn prefetch_models(args: PrefetchModelsArgs) -> Result<()> {
    let mut config = Config::default();
    config.embedding_cache = args.embedding_cache;
    let _ = Embedder::new(&config)?;
    println!(
        "embedding model cached at {}",
        config.embedding_cache.display()
    );
    Ok(())
}

fn env_paths(key: &str, default: &str) -> Vec<PathBuf> {
    std::env::var(key)
        .unwrap_or_else(|_| default.into())
        .split(':')
        .filter(|part| !part.is_empty())
        .map(PathBuf::from)
        .collect()
}
