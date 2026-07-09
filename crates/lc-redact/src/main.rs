//! Pipeline stage 2 — strip PII from the text the indexer extracted.
//!
//!   in : <index>/sidecar/**/*.txt   (read-only bind)
//!   out: <redacted>/sidecar/**/*.txt
//!        <redacted>/redactions.jsonl   one JSON object per redacted span
//!        <redacted>/run.json           the stage contract (lc_core::RunManifest)
//!
//! The audit log is the point. `redactions.jsonl` records file, rule, kind, byte offsets
//! and a salted digest — never the redacted value itself. That is enough to prove what
//! was removed and to spot the same identifier recurring across documents, and not enough
//! to reconstruct it.
//!
//! Fail-open per file, fail-closed per span: one unreadable file must never kill a run
//! over 500 documents, but a file that cannot be redacted is never written to the output.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use lc_core::{RunManifest, Stage};
use serde::Serialize;
use walkdir::WalkDir;

mod rules;
use rules::{Gating, Redactor};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum GatingArg {
    /// Redact bare national-ID-shaped numbers even with no adjacent keyword. Default.
    Strict,
    /// Require an adjacent keyword, as claudeops' `redact` crate does.
    ContextGated,
}

impl From<GatingArg> for Gating {
    fn from(g: GatingArg) -> Self {
        match g {
            GatingArg::Strict => Gating::Strict,
            GatingArg::ContextGated => Gating::ContextGated,
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "lc-redact",
    about = "Pipeline stage 2: redact PII from extracted text"
)]
struct Args {
    /// Input dir — the indexer's output. Read-only; we never write here.
    #[arg(long, default_value_os_t = Stage::Index.output_dir())]
    input: PathBuf,

    /// Output dir.
    #[arg(long, default_value_os_t = Stage::Redact.output_dir())]
    output: PathBuf,

    /// How to treat identifiers with no adjacent keyword.
    #[arg(long, value_enum, default_value_t = GatingArg::Strict)]
    gating: GatingArg,

    /// Extra surnames for the person-name gazetteer, one per line.
    #[arg(long)]
    names: Option<PathBuf>,

    /// Re-redact files that already exist in the output. Off by default: a re-run after a
    /// crash should be cheap.
    #[arg(long)]
    force: bool,
}

/// One line of `redactions.jsonl`.
#[derive(Serialize)]
struct AuditRow<'a> {
    /// Path relative to the input root. Never absolute: the container's view of the
    /// filesystem is not the host's.
    file: &'a str,
    #[serde(flatten)]
    span: &'a rules::Span,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let mut manifest = RunManifest::start(Stage::Redact);

    match run(&args, &mut manifest) {
        Ok(()) => {
            let _ = manifest.finish(&args.output, 0);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("lc-redact: {e:#}");
            manifest.warn(format!("fatal: {e:#}"));
            let _ = manifest.finish(&args.output, 1);
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args, manifest: &mut RunManifest) -> Result<()> {
    let src_root = args.input.join("sidecar");
    anyhow::ensure!(
        src_root.is_dir(),
        "no sidecar tree at {} -- has stage 1 (lc-index) run?",
        src_root.display()
    );

    let dst_root = args.output.join("sidecar");
    fs::create_dir_all(&dst_root)?;

    let extra: Vec<String> = match &args.names {
        Some(p) => fs::read_to_string(p)
            .with_context(|| format!("reading gazetteer {}", p.display()))?
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(str::to_string)
            .collect(),
        None => Vec::new(),
    };

    // Per-run salt, never persisted. Digests are therefore comparable *within* a run --
    // enough to spot one identifier recurring across documents -- and useless as a
    // rainbow-table target across runs.
    let salt = random_salt()?;
    let redactor = Redactor::new(args.gating.into(), &extra, salt);

    let audit_path = args.output.join("redactions.jsonl");
    let mut audit = BufWriter::new(
        fs::File::create(&audit_path)
            .with_context(|| format!("creating {}", audit_path.display()))?,
    );

    let (mut seen, mut written, mut skipped, mut failed, mut spans_total) = (0u64, 0, 0, 0, 0u64);

    for entry in WalkDir::new(&src_root).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if !entry.file_type().is_file() || path.extension().is_none_or(|e| e != "txt") {
            continue;
        }
        seen += 1;

        let rel = path.strip_prefix(&src_root).unwrap_or(path);
        let dst = dst_root.join(rel);

        if dst.exists() && !args.force {
            skipped += 1;
            continue;
        }

        match redact_one(&redactor, path, &dst, rel, &mut audit) {
            Ok(n) => {
                written += 1;
                spans_total += n;
            }
            Err(e) => {
                // Fail-open per file. The output simply lacks this document; it is never
                // written half-redacted.
                failed += 1;
                manifest.warn(format!("{}: {e:#}", rel.display()));
                eprintln!("lc-redact: skipping {}: {e:#}", rel.display());
            }
        }
    }

    audit.flush()?;

    manifest.count("files_seen", seen);
    manifest.count("files_written", written);
    manifest.count("files_skipped_existing", skipped);
    manifest.count("files_failed", failed);
    manifest.count("spans_redacted", spans_total);
    manifest.count(
        "gazetteer_surnames",
        (rules::VN_SURNAMES.len() + extra.len()) as u64,
    );

    if failed > 0 {
        manifest.warn(format!(
            "{failed} file(s) could not be redacted and were NOT written to the output"
        ));
    }

    println!(
        "lc-redact: {seen} seen, {written} written, {skipped} skipped, {failed} failed, \
         {spans_total} spans redacted"
    );
    Ok(())
}

fn redact_one(
    redactor: &Redactor,
    src: &Path,
    dst: &Path,
    rel: &Path,
    audit: &mut impl Write,
) -> Result<u64> {
    let text = fs::read_to_string(src).context("reading (not valid UTF-8?)")?;
    let spans = redactor.find(&text);
    let out = redactor.apply(&text, &spans);

    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    // Write via a temp file + rename: a stage killed mid-write (this box loses mains
    // power) must never leave a truncated, partially-redacted document behind.
    let tmp = dst.with_extension("txt.tmp");
    fs::write(&tmp, out.as_bytes())?;
    fs::rename(&tmp, dst)?;

    let file = rel.to_string_lossy();
    for span in &spans {
        writeln!(
            audit,
            "{}",
            serde_json::to_string(&AuditRow { file: &file, span })?
        )?;
    }
    Ok(spans.len() as u64)
}

/// 16 bytes from the OS. No `rand` dependency for this.
///
/// `read_exact`, never `fs::read` — /dev/urandom is an endless stream and reading it to
/// EOF never returns.
fn random_salt() -> Result<[u8; 16]> {
    use std::io::Read;
    let mut buf = [0u8; 16];
    fs::File::open("/dev/urandom")
        .context("opening /dev/urandom")?
        .read_exact(&mut buf)
        .context("reading /dev/urandom")?;
    Ok(buf)
}
