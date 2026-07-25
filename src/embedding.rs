use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
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
    Ok(rank_chunks(&connection, &config.embedding_model, &query_vector, limit)?.hits)
}

/// Upper bound on `limit` for every ranking surface. The scan is exhaustive
/// either way, so this bounds the RESPONSE, never the work.
pub const MAX_HITS: usize = 100;

/// The outcome of one exhaustive cosine scan over a corpus' `chunks` table.
///
/// `compared` and `skipped` exist so an empty `hits` can be explained. "No
/// matches" over a corpus with no vectors at all, and "no matches" over a
/// corpus whose 2.7 M vectors were written by a DIFFERENT embedding model, are
/// the same empty list and completely different facts — the caller has to be
/// able to tell them apart.
#[derive(Debug, Clone)]
pub struct VectorScan {
    pub hits: Vec<VectorHit>,
    /// Chunks actually ranked: those whose stored vector came from the model
    /// the query was embedded with, at the same width.
    pub compared: usize,
    /// Chunks passed over because another model (or another width) wrote them.
    /// Cosine across two embedding spaces is a meaningless number, so they are
    /// never scored — but they are counted.
    pub skipped: usize,
    /// The `model (Nd)` labels behind `skipped`, deduplicated and capped. What
    /// turns "everything was skipped" into an actionable message.
    pub other_models: Vec<String>,
}

/// Rank a corpus' chunk embeddings against `query` by cosine similarity.
///
/// An exhaustive streaming scan: every stored vector is scored where SQLite
/// hands it over, so the top-k is exact rather than approximate and the whole
/// pass allocates nothing per row. Three things keep that affordable on the
/// live corpora (2.68 M vectors / 4.1 GB of BLOB in the largest):
///
/// * `content` is NOT selected. It is the bulk of a chunk row, it is needed for
///   at most `limit` of them, and pulling it for all 2.7 M would materialise
///   well over a gigabyte of `String` to throw away. The winners are hydrated
///   in a second keyed pass ([`hydrate`]).
/// * vectors are scored straight out of the SQLite blob — no `Vec<f32>` per
///   row, no copy into a staging buffer.
/// * a bounded [`TopK`] heap replaces sorting all N scores, so memory is
///   `O(limit)` however large the corpus is.
///
/// It is deliberately single-threaded. Measured on the live corpus (see
/// `scan_latency_over_a_real_corpus` and `docs/ARCHITECTURE.md`), scoring is
/// ~0.2 s per million vectors against ~2.3 s to read them: the pass is bound by
/// SQLite page reads, so handing the arithmetic to rayon would trade a few
/// percent for contention with the extraction pool a concurrent index job is
/// using.
///
/// Ordering is deterministic: descending score, ties broken by ascending
/// `chunks.id`.
pub fn rank_chunks(
    connection: &Connection,
    model: &str,
    query: &[f32],
    limit: usize,
) -> Result<VectorScan> {
    let limit = limit.clamp(1, MAX_HITS);
    let width = query.len() * 4;
    let query_norm = norm(query);
    let mut best = TopK::new(limit);
    let mut compared = 0usize;
    let mut skipped = 0usize;
    let mut other_models = BTreeSet::new();
    let mut statement = connection.prepare("SELECT id,model,dimensions,embedding FROM chunks")?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let row_model = row.get_ref(1)?.as_str()?;
        let blob = row.get_ref(3)?.as_blob()?;
        if row_model != model || blob.len() != width {
            skipped += 1;
            if other_models.len() < MAX_REPORTED_MODELS {
                let dimensions: i64 = row.get(2)?;
                other_models.insert(format!("{row_model} ({dimensions}d)"));
            }
            continue;
        }
        compared += 1;
        best.push(row.get(0)?, cosine_bytes(query, query_norm, blob));
    }
    let hits = hydrate(connection, &best.into_sorted())?;
    Ok(VectorScan {
        hits,
        compared,
        skipped,
        other_models: other_models.into_iter().collect(),
    })
}

/// Foreign `model (Nd)` labels reported by a scan before it stops collecting
/// them: enough to name what is in the corpus, bounded so a corpus with junk in
/// the column cannot grow the set without limit.
const MAX_REPORTED_MODELS: usize = 4;

/// Fetch path/name/content for the winners, in ranked order.
///
/// The inner join is the same one the scan's predecessor used: a chunk whose
/// `files` row has gone leaves the result rather than appearing pathless.
fn hydrate(connection: &Connection, ranked: &[Ranked]) -> Result<Vec<VectorHit>> {
    if ranked.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = vec!["?"; ranked.len()].join(",");
    let mut statement = connection.prepare(&format!(
        "SELECT c.id,f.path,f.name,c.chunk_index,c.content \
         FROM chunks c JOIN files f ON f.id=c.file_id WHERE c.id IN ({placeholders})"
    ))?;
    let ids = ranked.iter().map(|entry| entry.id).collect::<Vec<_>>();
    let rows = statement.query_map(rusqlite::params_from_iter(ids), |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, i64>(3)? as usize,
            row.get::<_, String>(4)?,
        ))
    })?;
    let mut fetched = HashMap::with_capacity(ranked.len());
    for row in rows {
        let (id, path, name, chunk_index, content) = row?;
        fetched.insert(id, (path, name, chunk_index, content));
    }
    Ok(ranked
        .iter()
        .filter_map(|entry| {
            let (path, name, chunk_index, content) = fetched.remove(&entry.id)?;
            Some(VectorHit {
                path,
                name,
                chunk_index,
                score: entry.score,
                content,
            })
        })
        .collect())
}

/// One ranked chunk while the scan is still running: the id and its score, and
/// nothing else, so the heap stays small however wide the corpus rows are.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Ranked {
    score: f32,
    id: i64,
}

impl Eq for Ranked {}

impl Ord for Ranked {
    /// Ordered WORST-first, which is what makes [`BinaryHeap`] — a max-heap —
    /// evict the weakest hit. "Worse" is a lower score, and on a tie the higher
    /// `chunks.id`: equal cosines therefore resolve to the earlier row, exactly
    /// as the stable sort over rowid order this replaced did.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then(self.id.cmp(&other.id))
    }
}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A bounded best-k accumulator over a stream of scores.
struct TopK {
    limit: usize,
    heap: BinaryHeap<Ranked>,
}

impl TopK {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            heap: BinaryHeap::with_capacity(limit + 1),
        }
    }

    fn push(&mut self, id: i64, score: f32) {
        let candidate = Ranked { score, id };
        if self.heap.len() < self.limit {
            self.heap.push(candidate);
        } else if self
            .heap
            .peek()
            .is_some_and(|worst| candidate.cmp(worst) == Ordering::Less)
        {
            self.heap.pop();
            self.heap.push(candidate);
        }
    }

    /// Best first. `into_sorted_vec` ascends by [`Ranked`]'s worst-first order,
    /// which is best-first by score.
    fn into_sorted(self) -> Vec<Ranked> {
        self.heap.into_sorted_vec()
    }
}

pub fn vector_to_bytes(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Cosine of `query` against one stored vector, read straight out of its SQLite
/// blob. The caller guarantees `bytes` is `query.len() * 4` long, and passes
/// `query_norm` in because it is the same for every row of a scan.
///
/// Decoding in place is the point: this is the innermost loop of a scan that
/// runs over every vector in the corpus, and a `Vec<f32>` per row would allocate
/// 2.7 M times on the live one.
fn cosine_bytes(query: &[f32], query_norm: f32, bytes: &[u8]) -> f32 {
    let mut dot = 0.0f32;
    let mut square = 0.0f32;
    for (value, stored) in query.iter().zip(bytes.chunks_exact(4)) {
        let stored = f32::from_le_bytes(stored.try_into().expect("four-byte chunk"));
        dot += value * stored;
        square += stored * stored;
    }
    let stored_norm = square.sqrt();
    if query_norm == 0.0 || stored_norm == 0.0 {
        return -1.0;
    }
    let score = dot / (query_norm * stored_norm);
    // A vector holding NaN/inf (a broken producer, a torn blob) ranks below
    // every real one instead of poisoning the comparison, which is the same
    // -1.0 floor a degenerate (zero-norm) vector gets above.
    if score.is_finite() {
        score
    } else {
        -1.0
    }
}

fn norm(vector: &[f32]) -> f32 {
    vector.iter().map(|value| value * value).sum::<f32>().sqrt()
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
        let stored = vector_to_bytes(&vector);
        assert_eq!(stored.len(), vector.len() * 4);
        assert!((cosine_bytes(&vector, norm(&vector), &stored) - 1.0).abs() < 0.0001);
        // Orthogonal scores 0, opposite scores -1: the ends of the range the
        // ranking is ordered by.
        let across = vec![0.0, 0.0, 1.0];
        assert!(cosine_bytes(&[1.0, 0.0, 0.0], 1.0, &vector_to_bytes(&across)).abs() < 0.0001);
        let opposite = vec![-1.0, 0.0, 0.0];
        assert!((cosine_bytes(&[1.0, 0.0, 0.0], 1.0, &vector_to_bytes(&opposite)) + 1.0) < 0.0001);
    }

    #[test]
    fn a_degenerate_stored_vector_ranks_last_instead_of_poisoning_the_order() {
        let query = [1.0, 0.0, 0.0];
        assert_eq!(
            cosine_bytes(&query, 1.0, &vector_to_bytes(&[0.0, 0.0, 0.0])),
            -1.0
        );
        assert_eq!(
            cosine_bytes(&query, 1.0, &vector_to_bytes(&[f32::NAN, 0.0, 0.0])),
            -1.0
        );
    }

    /// A corpus holding only what ranking reads: one `chunks` row per
    /// `(id, file_id, model, vector)`, and a `files` row per distinct file id.
    fn corpus(rows: &[(i64, i64, &str, Vec<f32>)]) -> Connection {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE files(id INTEGER PRIMARY KEY, path TEXT UNIQUE, name TEXT);
                 CREATE TABLE chunks(
                   id INTEGER PRIMARY KEY,
                   file_id INTEGER NOT NULL,
                   chunk_index INTEGER NOT NULL,
                   content TEXT NOT NULL,
                   embedding BLOB NOT NULL,
                   dimensions INTEGER NOT NULL,
                   model TEXT NOT NULL);",
            )
            .unwrap();
        for (id, file_id, model, vector) in rows {
            connection
                .execute(
                    "INSERT OR IGNORE INTO files(id,path,name) VALUES(?1,?2,?3)",
                    rusqlite::params![
                        file_id,
                        format!("/corpus/file{file_id}.txt"),
                        format!("file{file_id}.txt")
                    ],
                )
                .unwrap();
            connection
                .execute(
                    "INSERT INTO chunks(id,file_id,chunk_index,content,embedding,dimensions,model) \
                     VALUES(?1,?2,?3,?4,?5,?6,?7)",
                    rusqlite::params![
                        id,
                        file_id,
                        0,
                        format!("chunk {id}"),
                        vector_to_bytes(vector),
                        vector.len() as i64,
                        model
                    ],
                )
                .unwrap();
        }
        connection
    }

    const MODEL: &str = EMBEDDING_MODEL;

    #[test]
    fn ranking_orders_by_cosine_and_hydrates_the_winners() {
        let connection = corpus(&[
            (1, 10, MODEL, vec![0.0, 1.0, 0.0]),  // orthogonal
            (2, 11, MODEL, vec![1.0, 0.0, 0.0]),  // exact
            (3, 12, MODEL, vec![-1.0, 0.0, 0.0]), // opposite
            (4, 13, MODEL, vec![0.8, 0.6, 0.0]),  // close
        ]);
        let scan = rank_chunks(&connection, MODEL, &[1.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(scan.compared, 4);
        assert_eq!(scan.skipped, 0);
        let ranked = scan
            .hits
            .iter()
            .map(|hit| hit.content.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ranked, ["chunk 2", "chunk 4", "chunk 1"]);
        assert!((scan.hits[0].score - 1.0).abs() < 0.0001);
        assert!((scan.hits[1].score - 0.8).abs() < 0.0001);
        assert!(scan.hits[2].score.abs() < 0.0001);
        // The winners carry their file identity, fetched in the second pass.
        assert_eq!(scan.hits[0].path, "/corpus/file11.txt");
        assert_eq!(scan.hits[0].name, "file11.txt");
        assert_eq!(scan.hits[0].chunk_index, 0);
    }

    #[test]
    fn equal_scores_resolve_to_the_earlier_chunk() {
        let connection = corpus(&[
            (7, 10, MODEL, vec![1.0, 0.0, 0.0]),
            (8, 11, MODEL, vec![1.0, 0.0, 0.0]),
        ]);
        let scan = rank_chunks(&connection, MODEL, &[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(scan.hits.len(), 1);
        assert_eq!(scan.hits[0].content, "chunk 7");
    }

    #[test]
    fn vectors_from_another_model_are_counted_never_ranked() {
        // Cosine between two embedding spaces is a meaningless number, so a
        // corpus embedded with something else must come back empty WITH the
        // reason, not silently mis-ranked.
        let connection = corpus(&[
            (1, 10, "clip-vit-b32", vec![1.0, 0.0, 0.0]),
            (2, 11, "clip-vit-b32", vec![0.9, 0.1, 0.0]),
        ]);
        let scan = rank_chunks(&connection, MODEL, &[1.0, 0.0, 0.0], 5).unwrap();
        assert!(scan.hits.is_empty());
        assert_eq!(scan.compared, 0);
        assert_eq!(scan.skipped, 2);
        assert_eq!(scan.other_models, ["clip-vit-b32 (3d)"]);
    }

    #[test]
    fn a_vector_of_another_width_is_skipped_like_another_model() {
        let connection = corpus(&[
            (1, 10, MODEL, vec![1.0, 0.0, 0.0]),
            (2, 11, MODEL, vec![1.0, 0.0, 0.0, 0.0]),
        ]);
        let scan = rank_chunks(&connection, MODEL, &[1.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(scan.compared, 1);
        assert_eq!(scan.skipped, 1);
        assert_eq!(scan.other_models, [format!("{MODEL} (4d)")]);
    }

    #[test]
    fn a_chunk_whose_file_row_is_gone_drops_out_of_the_hits() {
        let connection = corpus(&[(1, 10, MODEL, vec![1.0, 0.0, 0.0])]);
        connection.execute("DELETE FROM files", []).unwrap();
        let scan = rank_chunks(&connection, MODEL, &[1.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(scan.compared, 1);
        assert!(scan.hits.is_empty());
    }

    #[test]
    fn a_long_scan_keeps_the_winners_wherever_they_sit() {
        // Thousands of rows with the winners planted at the extremes — the very
        // first row, the very last, and points in between — and one score
        // deliberately duplicated far apart. A heap that only ever holds five
        // entries has to survive an eviction stream this long without the
        // result depending on where in the scan a hit turned up.
        let count = 8_199usize;
        let planted = |id: i64| -> Option<f32> {
            match id {
                1 => Some(0.6),
                4_096 => Some(1.0),
                4_097 => Some(0.3),
                8_000 => Some(1.0),
                8_192 => Some(0.8),
                id if id == count as i64 => Some(0.5),
                _ => None,
            }
        };
        let rows = (1..=count as i64)
            .map(|id| {
                // Unit vectors, so the cosine against [1,0,0] IS the first
                // component; unplanted rows sit orthogonal at 0.0.
                let vector = match planted(id) {
                    Some(score) => vec![score, (1.0f32 - score * score).sqrt(), 0.0],
                    None => vec![0.0, 1.0, 0.0],
                };
                (id, id, MODEL, vector)
            })
            .collect::<Vec<_>>();
        let connection = corpus(&rows);
        let scan = rank_chunks(&connection, MODEL, &[1.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(scan.compared, count);
        let ranked = scan
            .hits
            .iter()
            .map(|hit| hit.content.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            ranked,
            [
                "chunk 4096".to_string(), // 1.0, earlier id wins the tie
                "chunk 8000".to_string(), // 1.0, thousands of rows later
                "chunk 8192".to_string(), // 0.8
                "chunk 1".to_string(),    // 0.6, the very first row scanned
                format!("chunk {count}"), // 0.5, the very last
            ]
        );
        for (hit, expected) in scan.hits.iter().zip([1.0, 1.0, 0.8, 0.6, 0.5]) {
            assert!(
                (hit.score - expected).abs() < 0.0001,
                "{} scored {}",
                hit.content,
                hit.score
            );
        }
    }

    /// Live scan latency over a real corpus — runs only when
    /// `LLM_INDEX_BENCH_CORPUS` names one, otherwise it skips, so CI needs no
    /// multi-GB fixture. `LLM_INDEX_BENCH_PASSES` (default 3) bounds how many
    /// full scans it costs; every pass reads the whole `chunks` table, so keep
    /// it at 1 when pointing this at something big. Build with `--release`: a
    /// debug build measures rustc's bounds checks, not the scan. The numbers
    /// this produced on the live corpora are in `docs/ARCHITECTURE.md`.
    #[test]
    fn scan_latency_over_a_real_corpus() {
        let Ok(path) = std::env::var("LLM_INDEX_BENCH_CORPUS") else {
            eprintln!("skipping scan latency measurement: LLM_INDEX_BENCH_CORPUS unset");
            return;
        };
        let passes = std::env::var("LLM_INDEX_BENCH_PASSES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(3);
        let connection = rusqlite::Connection::open_with_flags(
            &path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .expect("open corpus read-only");
        // A deterministic unit vector: the scan costs the same whatever the
        // query is, and this keeps the measurement free of the embedding model.
        let mut seed = 42u32;
        let query = (0..384)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1u32 << 23) as f32 - 0.5
            })
            .collect::<Vec<_>>();
        let query = query.iter().map(|v| v / norm(&query)).collect::<Vec<_>>();
        // The floor: step every row and touch every blob, scoring nothing and
        // never reading past `embedding`. What a full ranking costs beyond this
        // is the cosine arithmetic plus the per-row `model` check — the split
        // that decided both the threading and the provenance filter.
        let started = std::time::Instant::now();
        let mut statement = connection
            .prepare("SELECT id,embedding FROM chunks")
            .unwrap();
        let mut rows = statement.query([]).unwrap();
        let mut bytes = 0usize;
        while let Some(row) = rows.next().unwrap() {
            bytes += row.get_ref(1).unwrap().as_blob().unwrap().len();
        }
        eprintln!(
            "row walk over {bytes} vector bytes, no scoring: {:?} (this pass also \
             warms the page cache, so it carries the cold-read cost)",
            started.elapsed()
        );
        for pass in 1..=passes {
            let started = std::time::Instant::now();
            let scan = rank_chunks(&connection, EMBEDDING_MODEL, &query, 20).unwrap();
            eprintln!(
                "pass {pass}: compared {} skipped {} -> {} hits in {:?}; best {:?}",
                scan.compared,
                scan.skipped,
                scan.hits.len(),
                started.elapsed(),
                scan.hits
                    .first()
                    .map(|hit| (hit.path.clone(), hit.chunk_index, hit.score)),
            );
        }
    }

    #[test]
    fn limit_is_clamped_to_the_response_bound() {
        let rows = (1..=MAX_HITS as i64 + 10)
            .map(|id| (id, id, MODEL, vec![1.0, id as f32 / 1000.0, 0.0]))
            .collect::<Vec<_>>();
        let connection = corpus(&rows);
        let scan = rank_chunks(&connection, MODEL, &[1.0, 0.0, 0.0], 10_000).unwrap();
        assert_eq!(scan.hits.len(), MAX_HITS);
        let scan = rank_chunks(&connection, MODEL, &[1.0, 0.0, 0.0], 0).unwrap();
        assert_eq!(scan.hits.len(), 1);
    }
}
