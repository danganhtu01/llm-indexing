use std::path::Path;

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::config::Config;

pub const EMBEDDING_MODEL: &str = "intfloat/multilingual-e5-small";
const CHUNK_CHARS: usize = 1_200;
const CHUNK_OVERLAP: usize = 200;

#[derive(Debug, Clone)]
pub struct EmbeddedChunk {
    pub index: usize,
    pub content: String,
    pub vector: Vec<f32>,
    /// First/last page (1-based) this chunk's text was drawn from — `None`
    /// for any document whose extraction carried no page segments, and for a
    /// chunk built before this field existed. P0-8: this is what lets a hit
    /// render as "p. 14" instead of just a filename.
    pub page_start: Option<usize>,
    pub page_end: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorHit {
    pub path: String,
    pub name: String,
    pub chunk_index: usize,
    pub score: f32,
    pub content: String,
    /// `#[serde(default)]`: a hub built against a wave-2 engine (a corpus
    /// predating this column, or a response serialized by an older binary)
    /// must decode this as an absent locator rather than fail outright — see
    /// the wave-3 deploy-order rule.
    #[serde(default)]
    pub page_start: Option<usize>,
    #[serde(default)]
    pub page_end: Option<usize>,
}

pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    /// Build with the config's own intra-op width — see
    /// [`Config::resolved_embed_intra_threads`].
    pub fn new(config: &Config) -> Result<Self> {
        Self::with_intra_threads(config, config.resolved_embed_intra_threads())
    }

    /// Build with an explicit ONNX intra-op thread count.
    ///
    /// This is where the `embed_intra_threads` knob stops being live: ort bakes
    /// `intra_threads` into the `Session` here, at construction. Changing the
    /// setting therefore affects only instances built AFTER the change, which is
    /// what `GET /runtime` reports (`live: false`, `applies:
    /// next-embedder-instance`) rather than pretending otherwise.
    pub fn with_intra_threads(config: &Config, intra_threads: usize) -> Result<Self> {
        if config.embedding_model != EMBEDDING_MODEL {
            anyhow::bail!("unsupported embedding model {}", config.embedding_model)
        }
        let model = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::MultilingualE5Small)
                .with_cache_dir(config.embedding_cache.clone())
                .with_show_download_progress(false)
                .with_intra_threads(intra_threads.clamp(1, 8)),
        )
        .context("loading multilingual embedding model")?;
        Ok(Self { model })
    }

    /// `page_segments` are the same `(page_number, text)` pairs
    /// `ProcessedFile::page_segments` carries — pass an empty slice for a
    /// document with no page structure, which chunks `content` exactly as
    /// before this field existed.
    pub fn embed_document(
        &mut self,
        content: &str,
        page_segments: &[(usize, String)],
    ) -> Result<Vec<EmbeddedChunk>> {
        let spans = chunk_spans(content, page_segments);
        if spans.is_empty() {
            return Ok(Vec::new());
        }
        let passages = spans
            .iter()
            .map(|span| format!("passage: {}", span.content))
            .collect::<Vec<_>>();
        let vectors = self.model.embed(passages, None)?;
        Ok(spans
            .into_iter()
            .zip(vectors)
            .enumerate()
            .map(|(index, (span, vector))| EmbeddedChunk {
                index,
                content: span.content,
                vector,
                page_start: span.page_start,
                page_end: span.page_end,
            })
            .collect())
    }

    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        self.model
            .embed(vec![format!("query: {query}")], None)?
            .pop()
            .context("embedding model returned no query vector")
    }
}

pub fn vector_search(
    index: &Path,
    config: &Config,
    query: &str,
    limit: usize,
) -> Result<Vec<VectorHit>> {
    let mut embedder = Embedder::new(config)?;
    let query_vector = embedder.embed_query(query)?;
    let connection = Connection::open(index)?;
    let mut statement = connection.prepare(
        "SELECT f.path,f.name,c.chunk_index,c.content,c.embedding,c.page_start,c.page_end \
         FROM chunks c JOIN files f ON f.id=c.file_id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)? as usize,
            row.get::<_, String>(3)?,
            row.get::<_, Vec<u8>>(4)?,
            row.get::<_, Option<i64>>(5)?,
            row.get::<_, Option<i64>>(6)?,
        ))
    })?;
    let mut hits = rows
        .flatten()
        .filter_map(
            |(path, name, chunk_index, content, bytes, page_start, page_end)| {
                let vector = bytes_to_vector(&bytes)?;
                Some(VectorHit {
                    path,
                    name,
                    chunk_index,
                    score: cosine(&query_vector, &vector),
                    content,
                    page_start: page_start.map(|value| value as usize),
                    page_end: page_end.map(|value| value as usize),
                })
            },
        )
        .collect::<Vec<_>>();
    hits.sort_by(|left, right| right.score.total_cmp(&left.score));
    hits.truncate(limit.clamp(1, 100));
    Ok(hits)
}

pub fn vector_to_bytes(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bytes_to_vector(bytes: &[u8]) -> Option<Vec<f32>> {
    bytes.len().is_multiple_of(4).then(|| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect()
    })
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return -1.0;
    }
    let dot = left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        -1.0
    } else {
        dot / (left_norm * right_norm)
    }
}

/// One windowed slice of text ready to embed, with the page range (if any) it
/// was drawn from.
struct ChunkSpan {
    content: String,
    page_start: Option<usize>,
    page_end: Option<usize>,
}

/// The same fixed-size sliding window `chunks` always used (`CHUNK_CHARS`
/// wide, `CHUNK_OVERLAP` overlap), run over `page_segments` when present so
/// every window can report which page(s) its characters came from — and over
/// `content` unchanged (every span's pages `None`) when they are not, which
/// is exactly the previous behaviour for every extraction method but PDF.
///
/// `page_segments` is walked as one flat, page-tagged character sequence
/// rather than character-offset-mapped back onto `content`: the latter would
/// have to agree, character for character, with whatever `content` became
/// after NFC normalization and (for some methods) an appended vision block —
/// agreement this function cannot assume. Chunking the segments directly
/// sidesteps that entirely, at the cost of the embedded chunk text being
/// built from the page segments rather than literally `content` for a PDF —
/// harmless, since `chunks.content` is a derived embedding input, not a
/// verbatim copy of anything stored elsewhere.
fn chunk_spans(content: &str, page_segments: &[(usize, String)]) -> Vec<ChunkSpan> {
    let characters: Vec<(char, Option<usize>)> = if page_segments.is_empty() {
        content.chars().map(|c| (c, None)).collect()
    } else {
        page_segments
            .iter()
            .flat_map(|(page, text)| text.chars().map(move |c| (c, Some(*page))))
            .collect()
    };
    let mut spans = Vec::new();
    let mut start = 0;
    while start < characters.len() {
        let end = (start + CHUNK_CHARS).min(characters.len());
        let slice = &characters[start..end];
        let text = slice.iter().map(|(c, _)| *c).collect::<String>();
        if !text.trim().is_empty() {
            let pages = slice.iter().filter_map(|(_, page)| *page);
            spans.push(ChunkSpan {
                content: text,
                page_start: pages.clone().min(),
                page_end: pages.max(),
            });
        }
        if end == characters.len() {
            break;
        }
        start = end.saturating_sub(CHUNK_OVERLAP);
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_overlap_without_dropping_text() {
        let text = "a".repeat(2_500);
        let output = chunk_spans(&text, &[]);
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].content.chars().count(), CHUNK_CHARS);
        assert_eq!(output[1].content.chars().count(), CHUNK_CHARS);
        // No page segments were given: every span is page-agnostic, the exact
        // previous behaviour for every method but PDF.
        assert!(output.iter().all(|span| span.page_start.is_none() && span.page_end.is_none()));
    }

    #[test]
    fn a_chunk_wholly_inside_one_page_reports_just_that_page() {
        let pages = vec![(3, "z".repeat(50))];
        let spans = chunk_spans("", &pages);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].page_start, Some(3));
        assert_eq!(spans[0].page_end, Some(3));
    }

    #[test]
    fn a_chunk_spanning_a_page_boundary_reports_the_full_range() {
        // One window (CHUNK_CHARS wide) straddling three short pages must
        // report the low and high page it actually drew text from, not just
        // the first or the last.
        let pages = vec![
            (1, "a".repeat(100)),
            (2, "b".repeat(100)),
            (3, "c".repeat(100)),
        ];
        let spans = chunk_spans("", &pages);
        assert_eq!(spans[0].page_start, Some(1));
        assert_eq!(spans[0].page_end, Some(3));
    }

    #[test]
    fn page_numbering_survives_a_skipped_blank_page() {
        // page 2 produced no segment (blank page dropped upstream); the
        // chunker must still see page 3 as page 3, not renumber it to 2.
        let pages = vec![(1, "a".repeat(50)), (3, "c".repeat(50))];
        let spans = chunk_spans("", &pages);
        assert_eq!(spans[0].page_start, Some(1));
        assert_eq!(spans[0].page_end, Some(3));
    }

    #[test]
    fn empty_page_segments_falls_back_to_flat_content_chunking() {
        // The non-PDF path: no page_segments, so chunking (and page
        // attribution) comes from `content` alone.
        let spans = chunk_spans("plain extracted text, no pages", &[]);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].content, "plain extracted text, no pages");
        assert_eq!(spans[0].page_start, None);
        assert_eq!(spans[0].page_end, None);
    }

    #[test]
    fn vector_round_trip_and_cosine() {
        let vector = vec![1.0, 2.0, 3.0];
        let decoded = bytes_to_vector(&vector_to_bytes(&vector)).unwrap();
        assert_eq!(decoded, vector);
        assert!((cosine(&vector, &decoded) - 1.0).abs() < 0.0001);
    }
}
