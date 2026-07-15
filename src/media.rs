use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use anyhow::{Context, Result};
use tempfile::tempdir;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::Config;

#[derive(Clone)]
pub struct Transcriber {
    context: Option<Arc<WhisperContext>>,
    threads: i32,
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
            threads: config.workers.clamp(1, 8) as i32,
        }
    }

    pub fn available(&self) -> bool {
        self.context.is_some()
    }

    pub fn transcribe(&self, path: &Path) -> Result<String> {
        let context = self
            .context
            .as_ref()
            .context("Whisper model is unavailable")?;
        let temp = tempdir()?;
        let wav = temp.path().join("audio.wav");
        let output = Command::new("ffmpeg")
            .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(path)
            .args(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"])
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
}
