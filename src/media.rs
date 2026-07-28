use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use anyhow::{Context, Result};
use tempfile::tempdir;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::Config;
use crate::failure::CapabilityUnavailable;

#[derive(Clone)]
pub struct Transcriber {
    context: Option<Arc<WhisperContext>>,
    threads: i32,
    /// [`Config::headroom_cores_cap`] captured at construction: `Some` adds
    /// `-threads <cap>` to the audio-extraction ffmpeg spawn, `None` (headroom
    /// off) leaves that argv byte-identical to before the feature existed.
    ffmpeg_threads: Option<usize>,
}

impl Transcriber {
    pub fn new(config: &Config) -> Self {
        let context = config
            .whisper_model
            .is_file()
            .then(|| {
                WhisperContext::new_with_params(
                    config.whisper_model.to_string_lossy().as_ref(),
                    WhisperContextParameters::default(),
                )
                .ok()
                .map(Arc::new)
            })
            .flatten();
        Self {
            context,
            // The legacy `workers.clamp(1, 8)` derivation, held under the
            // headroom core cap AT THIS READ — the configured `workers` is
            // never rewritten by finalize (see `Config::finalize`), so the
            // cap must be applied to the derivation itself, with the percent
            // in effect at construction deciding.
            threads: crate::headroom::capped(
                config.workers.clamp(1, 8),
                config.headroom_cores_cap(),
            ) as i32,
            ffmpeg_threads: config.headroom_cores_cap(),
        }
    }

    pub fn available(&self) -> bool {
        self.context.is_some()
    }

    pub fn transcribe(&self, path: &Path) -> Result<String> {
        // A typed error, not `.context(..)` on the `Option`, so
        // `crate::failure::classify` recognizes this as `Unsupported` by
        // downcasting rather than parsing the message — see `failure.rs`.
        let context = self
            .context
            .as_ref()
            .ok_or(CapabilityUnavailable("Whisper model is unavailable"))?;
        let temp = tempdir()?;
        let wav = temp.path().join("audio.wav");
        let output = Command::new("ffmpeg")
            .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y"])
            // Under headroom, `-threads <cores_cap>` on BOTH sides of `-i`:
            // the flag is positional, so only an input-side copy reaches the
            // DECODER and only an output-side copy reaches the encode. Both
            // collapse to nothing at pct 0. See `headroom::ffmpeg_thread_args`.
            .args(crate::headroom::ffmpeg_thread_args(self.ffmpeg_threads))
            .arg("-i")
            .arg(path)
            .args(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"])
            .args(crate::headroom::ffmpeg_thread_args(self.ffmpeg_threads))
            .arg(&wav)
            .output()
            .with_context(|| format!("running ffmpeg for {}", path.display()))?;
        if !output.status.success() {
            anyhow::bail!(
                "ffmpeg audio extraction failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )
        }

        let reader = hound::WavReader::open(&wav).context("opening extracted WAV")?;
        let samples = reader
            .into_samples::<i16>()
            .collect::<Result<Vec<_>, _>>()?;
        if samples.is_empty() {
            anyhow::bail!("media contains no decodable audio samples")
        }
        let mut audio = vec![0_f32; samples.len()];
        whisper_rs::convert_integer_to_float_audio(&samples, &mut audio)
            .context("converting audio samples")?;

        let mut state = context.create_state().context("creating Whisper state")?;
        let mut params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: -1.0,
        });
        params.set_n_threads(self.threads);
        params.set_translate(false);
        params.set_language(None);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        state
            .full(params, &audio)
            .context("running local Whisper transcription")?;

        let transcript = state
            .as_iter()
            .map(|segment| {
                format!(
                    "[{}-{}] {}",
                    timestamp(segment.start_timestamp()),
                    timestamp(segment.end_timestamp()),
                    segment.to_string().trim()
                )
            })
            .filter(|line| !line.ends_with("] "))
            .collect::<Vec<_>>()
            .join("\n");
        if transcript.trim().is_empty() {
            anyhow::bail!("Whisper produced an empty transcript")
        }
        Ok(transcript)
    }
}

fn timestamp(centiseconds: i64) -> String {
    let seconds = centiseconds.max(0) / 100;
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3600,
        seconds % 3600 / 60,
        seconds % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_transcript_timestamps() {
        assert_eq!(timestamp(372_300), "01:02:03");
    }

    #[test]
    fn whisper_threads_follow_the_legacy_derivation_when_headroom_is_off() {
        // The off-path guarantee: `workers.clamp(1, 8)`, untouched.
        let mut config = Config::default();
        config.workers = 6;
        assert_eq!(Transcriber::new(&config).threads, 6);
        assert_eq!(Transcriber::new(&config).ffmpeg_threads, None);
    }

    #[test]
    fn headroom_caps_whisper_threads_and_arms_the_ffmpeg_flag() {
        let mut config = Config::default();
        config.workers = 8;
        config.headroom_pct = 50;
        let cap = config.headroom_cores_cap().expect("headroom is on");
        let transcriber = Transcriber::new(&config);
        assert_eq!(transcriber.threads as usize, 8_usize.min(cap));
        assert_eq!(transcriber.ffmpeg_threads, Some(cap));
    }
}
