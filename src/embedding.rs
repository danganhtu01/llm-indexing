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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorHit {
    pub path: String,
    pub name: String,
    pub chunk_index: usize,
    pub score: f32,
    pub content: String,
}

pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    pub fn new(config: &Config) -> Result<Self> {
        if config.embedding_model != EMBEDDING_MODEL {
            anyhow::bail!("unsupported embedding model {}", config.embedding_model)
        }
        let model = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::MultilingualE5Small)
                .with_cache_dir(config.embedding_cache.clone())
                .with_show_download_progress(false)
                .with_intra_threads(config.workers.clamp(1, 8)),
        )
        .context("loading multilingual embedding model")?;
        Ok(Self { model })
    }

    pub fn embed_document(&mut self, content: &str) -> Result<Vec<EmbeddedChunk>> {
        let chunks = chunks(content);
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        let passages = chunks
            .iter()
            .map(|chunk| format!("passage: {chunk}"))
            .collect::<Vec<_>>();
        let vectors = self.model.embed(passages, None)?;
        Ok(chunks
            .into_iter()
            .zip(vectors)
            .enumerate()
            .map(|(index, (content, vector))| EmbeddedChunk {
                index,
                content,
                vector,
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
        "SELECT f.path,f.name,c.chunk_index,c.content,c.embedding \
         FROM chunks c JOIN files f ON f.id=c.file_id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)? as usize,
            row.get::<_, String>(3)?,
            row.get::<_, Vec<u8>>(4)?,
        ))
    })?;
    let mut hits = rows
        .flatten()
        .filter_map(|(path, name, chunk_index, content, bytes)| {
            let vector = bytes_to_vector(&bytes)?;
            Some(VectorHit {
                path,
                name,
                chunk_index,
                score: cosine(&query_vector, &vector),
                content,
            })
        })
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
    (bytes.len() % 4 == 0).then(|| {
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

fn chunks(content: &str) -> Vec<String> {
    let characters = content.chars().collect::<Vec<_>>();
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < characters.len() {
        let end = (start + CHUNK_CHARS).min(characters.len());
        let chunk = characters[start..end].iter().collect::<String>();
        if !chunk.trim().is_empty() {
            chunks.push(chunk);
        }
        if end == characters.len() {
            break;
        }
        start = end.saturating_sub(CHUNK_OVERLAP);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_overlap_without_dropping_text() {
        let text = "a".repeat(2_500);
        let output = chunks(&text);
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].chars().count(), CHUNK_CHARS);
        assert_eq!(output[1].chars().count(), CHUNK_CHARS);
    }

    #[test]
    fn vector_round_trip_and_cosine() {
        let vector = vec![1.0, 2.0, 3.0];
        let decoded = bytes_to_vector(&vector_to_bytes(&vector)).unwrap();
        assert_eq!(decoded, vector);
        assert!((cosine(&vector, &decoded) - 1.0).abs() < 0.0001);
    }
}
