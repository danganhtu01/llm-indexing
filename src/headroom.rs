//! Resource headroom (`serve --headroom` / `headroom_pct`): leave a slice of
//! the machine free while a job runs.
//!
//! The owner's ask is simple — a whole-drive index job must not make the box
//! unusable — and the mechanism is deliberately simple to match: one integer
//! percent (`0..=50`, bounds in `settings.rs` with the other knob ranges)
//! yields one derived number, [`cores_cap`], and every CPU-shaped knob in the
//! process is capped at it. `0` (the default) is the feature fully off: every
//! helper here returns the untouched legacy value, no priority class is set,
//! no `-threads` flag is added — byte-identical behaviour to a build without
//! the feature.
//!
//! This is a CROSS-ENGINE CONTRACT shared with `vlm-indexing` and driven by
//! the drives-analytics app:
//!
//! * `serve --headroom <PCT>` / `index --headroom <PCT>` — integer percent,
//!   `0..=50`, validated by clap. The CLI flag wins over the YAML
//!   `headroom_pct` when provided ([`crate::config::Config::override_headroom`]).
//! * `headroom_pct: <PCT>` — top-level YAML config field, default 0, clamped
//!   in `Config::finalize`.
//! * effective `h = pct / 100.0`;
//!   `cores_cap = max(1, floor(available_parallelism * (1 - h)))`.
//!
//! What a non-zero percent does (each at its own call site, all consulting
//! [`crate::config::Config::headroom_cores_cap`]):
//!
//! 1. the process drops itself to `BELOW_NORMAL_PRIORITY_CLASS` at startup
//!    ([`lower_process_priority`]); child processes (tesseract, ffmpeg,
//!    magick, pdftoppm, antiword) inherit the class automatically, so the
//!    subprocess fleet needs no per-spawn handling;
//! 2. the extract worker target and embedder-pool size are seeded UNDER the
//!    cap at read time (`RuntimeKnobs::from_config` — the configured
//!    `workers`/`embed_workers` are never rewritten, so a CLI flag can always
//!    loosen a YAML percent), and the cap rides on every `RuntimeKnobs`
//!    instance as a CEILING that `POST /runtime` retunes clamp to;
//! 3. every ONNX intra-op width (embedding, CLIP, RF-DETR, YuNet/SFace) is
//!    capped where the session is BUILT — see the session caches' doc
//!    comments for the first-construction caveat;
//! 4. the tesseract `OMP_THREAD_LIMIT` derivation and the whisper
//!    `set_n_threads` derivation are capped;
//! 5. every ffmpeg spawn gains `-threads <cores_cap>` on BOTH sides of `-i`
//!    (input-side caps the decoder, output-side the encoder/filters — see
//!    [`ffmpeg_thread_args`] for why the position matters).
//!
//! Deliberately NOT a runtime stage: the app in front of these engines
//! validates `GET /runtime` stage names strictly against its own copy of the
//! stage list and rejects whole bodies on unknown keys (see the `runtime.rs`
//! module header), so headroom is Config/CLI only. The one deliberate
//! exemption is the interactive query-side search embedder — see
//! `service::QueryEmbedder::spawn_load`: it serves the human the feature
//! protects.

/// `cores_cap` for a machine with `cores` logical cores at `pct` percent
/// headroom: `max(1, floor(cores * (1 - pct/100)))`.
///
/// Integer arithmetic on purpose — `cores * (100 - pct) / 100` floors exactly
/// (usize division truncates toward zero and both operands are non-negative),
/// so the contract's `floor(cores * (1 - h))` is computed without ever
/// touching floating point. At `pct == 0` this is `cores` unchanged, which is
/// what makes an off headroom genuinely off.
pub fn cores_cap_for(cores: usize, pct: u8) -> usize {
    let pct = usize::from(pct.min(100));
    (cores * (100 - pct) / 100).max(1)
}

/// [`cores_cap_for`] against THIS machine's `available_parallelism`, with the
/// same `unwrap_or(1)` fallback `runtime::default_ocr_threads` uses when the
/// count is unavailable.
pub fn cores_cap(pct: u8) -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    cores_cap_for(cores, pct)
}

/// `value` capped at `cap` when headroom is active, `value` untouched when it
/// is not (`cap == None`). The single shape every derivation-capping call site
/// uses, so "no cap means no change" is decided in exactly one place.
pub fn capped(value: usize, cap: Option<usize>) -> usize {
    match cap {
        Some(cap) => value.min(cap),
        None => value,
    }
}

/// The argv fragment a headroom-aware ffmpeg spawn appends: `-threads <cap>`
/// when headroom is active, NOTHING when it is not — an ffmpeg invoked at
/// `headroom_pct == 0` must be argv-identical to one from before the feature
/// existed.
///
/// ffmpeg applies `-threads` POSITIONALLY: as an INPUT option (before `-i`)
/// it caps the decoder's thread pool; as an output option (before the output
/// path) it caps encoding and filtering. The decoder is the CPU-heavy half of
/// every spawn in this crate (tiled-HEVC HEIC stills, video frame and
/// keyframe extraction), and an output-side flag alone does NOT reach it —
/// measured on ffmpeg 8.1.2 on this box: with only output-side `-threads 1` a
/// decode still ran ~7.6x parallel, while input-side `-threads 1` serialized
/// it. Callers therefore insert this fragment on BOTH sides of `-i`: the
/// input-side copy bounds the decode, the output-side copy bounds the
/// encode/filter work, and at `cap == None` both collapse to nothing.
pub fn ffmpeg_thread_args(cap: Option<usize>) -> Vec<String> {
    match cap {
        Some(cap) => vec!["-threads".to_string(), cap.max(1).to_string()],
        None => Vec::new(),
    }
}

/// Drop THIS process to `BELOW_NORMAL_PRIORITY_CLASS` so the OS scheduler
/// favours interactive work over an index job. Returns whether the change was
/// applied.
///
/// Called once at startup (both `serve` and the native `index` command) and
/// only when the effective headroom percent is non-zero — an untouched
/// deployment keeps `NORMAL_PRIORITY_CLASS` exactly as before. Every
/// subprocess this engine spawns (tesseract, ffmpeg, magick, pdftoppm,
/// pdftotext, antiword) inherits the priority class from its parent, so this
/// single call covers the whole process tree without touching a spawn site.
///
/// Windows-only by contract; on other platforms this is a documented no-op
/// returning `false` (the thread caps above still deliver the headroom).
#[cfg(windows)]
pub fn lower_process_priority() -> bool {
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, SetPriorityClass, BELOW_NORMAL_PRIORITY_CLASS,
    };
    // SAFETY: GetCurrentProcess returns the pseudo-handle for this process
    // (never fails, never needs closing); SetPriorityClass on it with a valid
    // priority-class constant has no memory-safety concerns.
    let applied =
        unsafe { SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS) } != 0;
    if applied {
        tracing::info!("process priority lowered to BelowNormal (--headroom)");
    } else {
        tracing::warn!("SetPriorityClass(BELOW_NORMAL_PRIORITY_CLASS) failed; priority unchanged");
    }
    applied
}

/// Non-Windows twin of [`lower_process_priority`]: a documented no-op. The
/// contract specifies the Windows priority class only; the CPU-thread caps are
/// the portable half of the feature.
#[cfg(not(windows))]
pub fn lower_process_priority() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cores_cap_math_matches_the_contract() {
        // floor(cores * (1 - pct/100)), never below 1.
        assert_eq!(cores_cap_for(16, 10), 14); // floor(14.4)
        assert_eq!(cores_cap_for(8, 10), 7); // floor(7.2)
        assert_eq!(cores_cap_for(4, 50), 2);
        assert_eq!(cores_cap_for(2, 50), 1);
        assert_eq!(cores_cap_for(1, 50), 1); // the max(1, ..) floor
        assert_eq!(cores_cap_for(12, 25), 9);
    }

    #[test]
    fn zero_percent_leaves_the_core_count_untouched() {
        // The off-path guarantee: h == 0 must not change ANY derived number.
        for cores in [1, 2, 3, 8, 16, 64, 128] {
            assert_eq!(cores_cap_for(cores, 0), cores);
        }
    }

    #[test]
    fn the_machine_wide_cap_agrees_with_the_pure_math() {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        assert_eq!(cores_cap(10), cores_cap_for(cores, 10));
        assert_eq!(cores_cap(0), cores);
    }

    #[test]
    fn capped_is_inert_without_a_cap() {
        assert_eq!(capped(64, None), 64);
        assert_eq!(capped(64, Some(14)), 14);
        assert_eq!(capped(4, Some(14)), 4);
    }

    #[test]
    fn ffmpeg_gains_threads_only_when_headroom_is_active() {
        // The argv contract: presence at h > 0, byte-identical absence at 0.
        assert_eq!(ffmpeg_thread_args(Some(14)), vec!["-threads", "14"]);
        assert!(ffmpeg_thread_args(None).is_empty());
        // A cap can never ask ffmpeg for zero threads (which ffmpeg reads as
        // "auto" — every core, the opposite of what headroom wants).
        assert_eq!(ffmpeg_thread_args(Some(0)), vec!["-threads", "1"]);
    }

    /// The one contract item the math tests cannot pin: the Windows call must
    /// actually land BelowNormal on a real process. Runs against THIS test
    /// process and restores whatever class the harness started with, so the
    /// rest of the suite is not quietly demoted (priority is process-wide, but
    /// briefly running tests at BelowNormal is harmless).
    #[cfg(windows)]
    #[test]
    fn lowering_the_process_priority_lands_below_normal_on_windows() {
        use windows_sys::Win32::System::Threading::{
            GetCurrentProcess, GetPriorityClass, SetPriorityClass, BELOW_NORMAL_PRIORITY_CLASS,
        };
        let before = unsafe { GetPriorityClass(GetCurrentProcess()) };
        assert_ne!(before, 0, "GetPriorityClass failed");
        assert!(lower_process_priority());
        let after = unsafe { GetPriorityClass(GetCurrentProcess()) };
        assert_eq!(after, BELOW_NORMAL_PRIORITY_CLASS);
        assert!(
            unsafe { SetPriorityClass(GetCurrentProcess(), before) } != 0,
            "restoring the harness's original priority class failed"
        );
    }
}
