use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::path::Path;

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use rusqlite::{Connection, OptionalExtension};
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
    // Before the open, never after: `vec0` reaches a connection through
    // SQLite's auto-extension list, which is consulted once as the connection
    // is created. See [`crate::vec0::register`].
    crate::vec0::register();
    let connection = Connection::open(index)?;
    Ok(rank_chunks(&connection, &config.embedding_model, &query_vector, limit)?.hits)
}

/// Upper bound on `limit` for every ranking surface. The scan is exhaustive
/// either way, so this bounds the RESPONSE, never the work.
pub const MAX_HITS: usize = 100;

/// The outcome of one ranking pass over a corpus' `chunks` table.
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
    /// the query was embedded with, at the same width. On the [`RankPath::Vec0`]
    /// path these were ranked by the shadow index rather than read one by one,
    /// so this is the index's row count — the same set, counted from the
    /// recorded state instead of from the pass.
    pub compared: usize,
    /// Chunks passed over because another model (or another width) wrote them.
    /// Cosine across two embedding spaces is a meaningless number, so they are
    /// never scored — but they are counted.
    pub skipped: usize,
    /// The `model (Nd)` labels behind `skipped`, deduplicated and capped. What
    /// turns "everything was skipped" into an actionable message. Empty on the
    /// [`RankPath::Vec0`] path, which never reads the rows it excludes.
    pub other_models: Vec<String>,
    /// Which of the two ranking paths produced `hits`.
    pub path: RankPath,
    /// Set only when a corpus HAS a shadow index that was not used, and says
    /// why. A search that quietly stops being fast is indistinguishable from
    /// one that was never made fast, and the two need completely different
    /// operator responses — see [`crate::vec0::usable`].
    pub index_note: Option<String>,
}

/// How a [`VectorScan`] was ranked.
///
/// Both paths return the same top-k with the same scores; they differ only in
/// what they read to get there — the whole `chunks` table against just the
/// vectors, 15.6 GB against 4.12 GB on the live corpus and roughly an order of
/// magnitude in latency (`docs/ARCHITECTURE.md`). Reported to the caller because
/// a 50 s answer and a 4 s one are otherwise the same JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankPath {
    /// The exhaustive cosine scan. Always available; the only path a corpus
    /// without a shadow index has.
    Scan,
    /// k-NN over the `vec0` shadow index, with the candidates re-scored against
    /// the stored BLOBs.
    Vec0,
}

impl RankPath {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Scan => "scan",
            Self::Vec0 => "vec0",
        }
    }
}

/// Rank a corpus' chunk embeddings against `query` by cosine similarity.
///
/// The entry point for every ranking surface, and the one place that decides
/// which path a corpus can be ranked by. A corpus carrying a usable `vec0`
/// shadow index is ranked by [`rank_by_index`]; every other corpus — which is
/// every corpus that has not been through `llm-index vector-index`, and every
/// corpus whose index cannot be vouched for — is ranked by [`scan_chunks`],
/// exactly as before the index existed.
///
/// The capability is discovered from the corpus itself rather than configured:
/// there is no flag to get wrong, and a corpus that loses (or never gains) its
/// index degrades in latency and in nothing else. Both paths return the same
/// hits with the same scores in the same order; [`VectorScan::path`] reports
/// which one ran.
pub fn rank_chunks(
    connection: &Connection,
    model: &str,
    query: &[f32],
    limit: usize,
) -> Result<VectorScan> {
    match crate::vec0::usable(connection, model, query.len())? {
        crate::vec0::Usable::Ready(state) => rank_by_index(connection, &state, query, limit),
        crate::vec0::Usable::Absent => scan_chunks(connection, model, query, limit, None),
        crate::vec0::Usable::Declined(reason) => {
            scan_chunks(connection, model, query, limit, Some(reason))
        }
    }
}

/// Rank through the `vec0` shadow index.
///
/// Two steps, and the second is what keeps this path honest: the index picks
/// the CANDIDATES, and their scores are then recomputed from
/// `chunks.embedding` with [`cosine_bytes`] — the scan's own arithmetic, on the
/// scan's own bytes — and ordered by the scan's own [`Ranked`] comparison. So a
/// `score` in a response is the same number whichever path produced it, and the
/// index can never introduce a ranking of its own.
///
/// The candidate list is deliberately wider than `limit` ([`KNN_TIE_MARGIN`]).
/// A k-NN returns SOME k nearest rows; where several vectors tie at the k-th
/// distance, which of them it returns is its traversal order rather than the
/// scan's "lower `chunks.id` wins". Over-fetching lets the re-score resolve
/// those ties the scan's way, and costs nothing — a `vec0` k-NN's work is
/// reading the vectors, not the size of its heap. Ties running deeper than the
/// margin (more than 100 chunks at exactly the boundary distance, i.e. that
/// many byte-identical vectors) can still pick a different member of the tie
/// than the scan would; the hits are equally correct, and the alternative is
/// giving up the index.
fn rank_by_index(
    connection: &Connection,
    state: &crate::vec0::IndexState,
    query: &[f32],
    limit: usize,
) -> Result<VectorScan> {
    let limit = limit.clamp(1, MAX_HITS);
    let width = query.len() * 4;
    let query_norm = norm(query);
    let candidates = crate::vec0::knn(connection, &vector_to_bytes(query), limit + KNN_TIE_MARGIN)?;
    let mut best = TopK::new(limit);
    let mut statement = connection.prepare("SELECT embedding FROM chunks WHERE id=?1")?;
    for id in candidates {
        // A candidate whose row has gone, or whose vector no longer has the
        // query's width, is dropped rather than scored — the index is a copy,
        // and this is the pass that treats `chunks` as the truth.
        let scored = statement
            .query_row([id], |row| {
                let blob = row.get_ref(0)?.as_blob()?;
                Ok((blob.len() == width).then(|| cosine_bytes(query, query_norm, blob)))
            })
            .optional()?
            .flatten();
        if let Some(score) = scored {
            best.push(id, score);
        }
    }
    let hits = hydrate(connection, &best.into_sorted())?;
    Ok(VectorScan {
        hits,
        // The index holds exactly the rows the scan would have compared, and
        // excludes exactly the rows it would have skipped — that is what
        // `vec0::build` filters on. Reporting the recorded counts keeps the two
        // paths' responses the same shape without reading 2.7 M rows to
        // recount what the corpus already knows.
        compared: state.vectors,
        skipped: state.chunks.saturating_sub(state.vectors),
        other_models: Vec::new(),
        path: RankPath::Vec0,
        index_note: None,
    })
}

/// Candidates fetched beyond `limit` so boundary ties resolve the scan's way.
/// See [`rank_by_index`].
const KNN_TIE_MARGIN: usize = MAX_HITS;

/// Rank a corpus' chunk embeddings by an exhaustive cosine scan.
///
/// Every stored vector is scored where SQLite hands it over, so the top-k is
/// exact rather than approximate and the whole pass allocates nothing per row.
/// Three things keep that affordable on the live corpora (2.68 M vectors /
/// 4.1 GB of BLOB in the largest):
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
///
/// `index_note` is carried straight through onto the result: this function is
/// reached with one set whenever a corpus HAS a shadow index that [`rank_chunks`]
/// declined, and the caller is owed that reason.
fn scan_chunks(
    connection: &Connection,
    model: &str,
    query: &[f32],
    limit: usize,
    index_note: Option<String>,
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
        path: RankPath::Scan,
        index_note,
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
        "SELECT c.id,f.path,f.name,c.chunk_index,c.content,c.page_start,c.page_end \
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
            row.get::<_, Option<i64>>(5)?,
            row.get::<_, Option<i64>>(6)?,
        ))
    })?;
    let mut fetched = HashMap::with_capacity(ranked.len());
    for row in rows {
        let (id, path, name, chunk_index, content, page_start, page_end) = row?;
        fetched.insert(id, (path, name, chunk_index, content, page_start, page_end));
    }
    Ok(ranked
        .iter()
        .filter_map(|entry| {
            let (path, name, chunk_index, content, page_start, page_end) =
                fetched.remove(&entry.id)?;
            Some(VectorHit {
                path,
                name,
                chunk_index,
                score: entry.score,
                content,
                page_start: page_start.map(|value| value as usize),
                page_end: page_end.map(|value| value as usize),
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
        assert!(output
            .iter()
            .all(|span| span.page_start.is_none() && span.page_end.is_none()));
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
        crate::vec0::register();
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                 CREATE TABLE files(id INTEGER PRIMARY KEY, path TEXT UNIQUE, name TEXT);
                 CREATE TABLE chunks(
                   id INTEGER PRIMARY KEY,
                   file_id INTEGER NOT NULL,
                   chunk_index INTEGER NOT NULL,
                   content TEXT NOT NULL,
                   embedding BLOB NOT NULL,
                   dimensions INTEGER NOT NULL,
                   model TEXT NOT NULL,
                   page_start INTEGER,
                   page_end INTEGER);",
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

    /// Live ranking latency over a real corpus — runs only when
    /// `LLM_INDEX_BENCH_CORPUS` names one, otherwise it skips, so CI needs no
    /// multi-GB fixture. `LLM_INDEX_BENCH_PASSES` (default 3) bounds how many
    /// full scans it costs; every pass reads the whole `chunks` table, so keep
    /// it at 1 when pointing this at something big. Build with `--release`: a
    /// debug build measures rustc's bounds checks, not the scan. The numbers
    /// this produced on the live corpora are in `docs/ARCHITECTURE.md`.
    ///
    /// When the corpus carries a `vec0` shadow index this measures BOTH paths,
    /// back to back on the same machine in the same session — the only way the
    /// two numbers are comparable, since the absolute cost of either is mostly
    /// the page cache. It then asserts they returned the same hits with the same
    /// scores, so the equivalence claim is checked against 2.68 M live vectors
    /// and not only against the fixtures above.
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
        crate::vec0::register();
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
            let scan = scan_chunks(&connection, EMBEDDING_MODEL, &query, 20, None).unwrap();
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
        // The other half of the same measurement, when the corpus has a shadow
        // index (`llm-index vector-index`): the k-NN path over the SAME corpus
        // and the SAME query, so the two numbers are comparable and the hits can
        // be checked against each other rather than merely timed.
        match crate::vec0::usable(&connection, EMBEDDING_MODEL, query.len()).unwrap() {
            crate::vec0::Usable::Absent => {
                eprintln!("no vec0 shadow index in this corpus; scan numbers only")
            }
            crate::vec0::Usable::Declined(reason) => eprintln!("shadow index not used: {reason}"),
            crate::vec0::Usable::Ready(state) => {
                eprintln!(
                    "shadow index: {} vectors over {} chunks, built by {}",
                    state.vectors, state.chunks, state.builder
                );
                let mut ranked = None;
                for pass in 1..=passes {
                    let started = std::time::Instant::now();
                    let indexed = rank_chunks(&connection, EMBEDDING_MODEL, &query, 20).unwrap();
                    eprintln!(
                        "vec0 pass {pass}: {} hits in {:?}; best {:?}",
                        indexed.hits.len(),
                        started.elapsed(),
                        indexed.hits.first().map(|hit| (
                            hit.path.clone(),
                            hit.chunk_index,
                            hit.score
                        )),
                    );
                    ranked = Some(indexed);
                }
                // Equivalence on the live corpus, not just on a fixture: the
                // same top-20, in the same order, with the same scores.
                let indexed = ranked.expect("at least one pass");
                let scanned = scan_chunks(&connection, EMBEDDING_MODEL, &query, 20, None).unwrap();
                assert_eq!(
                    indexed
                        .hits
                        .iter()
                        .map(|hit| (hit.path.clone(), hit.chunk_index, hit.score.to_bits()))
                        .collect::<Vec<_>>(),
                    scanned
                        .hits
                        .iter()
                        .map(|hit| (hit.path.clone(), hit.chunk_index, hit.score.to_bits()))
                        .collect::<Vec<_>>(),
                    "the index and the scan must agree over the live corpus"
                );
                eprintln!("index and scan agree on all {} hits", scanned.hits.len());
            }
        }
    }

    /// A corpus of `count` unit vectors spread over a half-circle, so no two
    /// share a score and the exact top-k is unambiguous, plus a handful of rows
    /// the index must exclude. Deterministic: no RNG, so a failure here is
    /// always reproducible.
    fn indexable_corpus(count: i64) -> Connection {
        let mut rows = (1..=count)
            .map(|id| {
                let angle = id as f32 * std::f32::consts::PI / (count as f32 + 1.0);
                (id, id, MODEL, vec![angle.cos(), angle.sin(), 0.0])
            })
            .collect::<Vec<_>>();
        rows.push((count + 1, count + 1, "clip-vit-b32", vec![1.0, 0.0, 0.0]));
        rows.push((count + 2, count + 2, MODEL, vec![1.0, 0.0, 0.0, 0.0]));
        corpus(&rows)
    }

    /// The query used by the equivalence tests. Off-axis on purpose: an axis
    /// query would put the winner at exactly 1.0 and hide any drift.
    const PROBE: [f32; 3] = [0.8, 0.6, 0.0];

    #[test]
    fn the_shadow_index_returns_exactly_what_the_scan_returns() {
        // The claim the whole index rests on: it changes how long a query takes
        // and nothing else. Same hits, same order, same scores to the bit —
        // which they must be, because the index only nominates candidates and
        // the scores are recomputed from the same BLOBs by the same cosine.
        let mut connection = indexable_corpus(500);
        let scanned = scan_chunks(&connection, MODEL, &PROBE, 20, None).unwrap();
        crate::vec0::build(&mut connection, MODEL, 3, |_, _| {}).unwrap();
        let indexed = rank_chunks(&connection, MODEL, &PROBE, 20).unwrap();

        assert_eq!(scanned.path, RankPath::Scan);
        assert_eq!(indexed.path, RankPath::Vec0);
        assert_eq!(indexed.index_note, None);
        assert_eq!(
            indexed
                .hits
                .iter()
                .map(|hit| (hit.content.clone(), hit.score.to_bits()))
                .collect::<Vec<_>>(),
            scanned
                .hits
                .iter()
                .map(|hit| (hit.content.clone(), hit.score.to_bits()))
                .collect::<Vec<_>>(),
        );
        // And the counts stay comparable: the index covers exactly the rows the
        // scan compared, and excludes exactly the ones it skipped.
        assert_eq!(indexed.compared, scanned.compared);
        assert_eq!(indexed.skipped, scanned.skipped);
    }

    #[test]
    fn a_rebuilt_index_agrees_with_the_scan_it_was_rebuilt_from() {
        // Rebuild-from-BLOBs is how the live corpora gain an index: nothing is
        // re-embedded, so the rebuilt index has to reproduce the scan's answer
        // over the vectors already stored. Rebuilt twice, because a rebuild
        // over an existing index is the repair path an operator actually runs.
        let mut connection = indexable_corpus(300);
        crate::vec0::build(&mut connection, MODEL, 3, |_, _| {}).unwrap();
        crate::vec0::build(&mut connection, MODEL, 3, |_, _| {}).unwrap();
        for limit in [1, 5, 20, MAX_HITS] {
            let indexed = rank_chunks(&connection, MODEL, &PROBE, limit).unwrap();
            let scanned = scan_chunks(&connection, MODEL, &PROBE, limit, None).unwrap();
            assert_eq!(indexed.path, RankPath::Vec0);
            assert_eq!(
                indexed
                    .hits
                    .iter()
                    .map(|hit| hit.content.clone())
                    .collect::<Vec<_>>(),
                scanned
                    .hits
                    .iter()
                    .map(|hit| hit.content.clone())
                    .collect::<Vec<_>>(),
                "limit {limit}"
            );
        }
    }

    #[test]
    fn a_corpus_without_an_index_ranks_by_the_scan_and_says_nothing_about_it() {
        // The default, and the capability fallback in its quiet form: no index,
        // no note, no difference from the release before this one.
        let connection = indexable_corpus(50);
        let scan = rank_chunks(&connection, MODEL, &PROBE, 5).unwrap();
        assert_eq!(scan.path, RankPath::Scan);
        assert_eq!(scan.index_note, None);
        assert_eq!(scan.hits.len(), 5);
    }

    #[test]
    fn a_stale_index_is_bypassed_and_the_answer_is_still_right() {
        // What an older build writing into an indexed corpus leaves behind. The
        // fallback is not a formality: the row it wrote is a genuine top hit,
        // and a k-NN over the index could not have found it.
        let mut connection = indexable_corpus(100);
        crate::vec0::build(&mut connection, MODEL, 3, |_, _| {}).unwrap();
        connection
            .execute(
                "INSERT INTO files(id,path,name) VALUES(9001,'/corpus/late.txt','late.txt')",
                [],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO chunks(id,file_id,chunk_index,content,embedding,dimensions,model) \
                 VALUES(9001,9001,0,'written behind the index',?1,3,?2)",
                rusqlite::params![vector_to_bytes(&PROBE), MODEL],
            )
            .unwrap();

        let scan = rank_chunks(&connection, MODEL, &PROBE, 5).unwrap();
        assert_eq!(scan.path, RankPath::Scan);
        assert!(
            scan.index_note
                .as_deref()
                .is_some_and(|note| note.contains("stale")),
            "{:?}",
            scan.index_note
        );
        assert_eq!(scan.hits[0].content, "written behind the index");
        assert!((scan.hits[0].score - 1.0).abs() < 0.0001);
    }

    #[test]
    fn the_index_drops_a_candidate_whose_chunk_row_has_gone() {
        // `chunks` is the truth and the index is a copy, so a candidate the copy
        // still knows about is dropped by the re-score rather than returned
        // pathless or scored from the stale vector.
        let mut connection = indexable_corpus(20);
        crate::vec0::build(&mut connection, MODEL, 3, |_, _| {}).unwrap();
        let top = rank_chunks(&connection, MODEL, &PROBE, 1).unwrap().hits[0]
            .content
            .clone();
        // Delete the winner's chunk row without touching the index, then keep
        // the staleness witness in step so the index is still used.
        connection
            .execute("DELETE FROM chunks WHERE content=?1", [&top])
            .unwrap();
        let mut state = crate::vec0::state(&connection).unwrap().unwrap();
        state.chunks -= 1;
        crate::vec0::write_state(&connection, &state).unwrap();

        let scan = rank_chunks(&connection, MODEL, &PROBE, 5).unwrap();
        assert_eq!(scan.path, RankPath::Vec0);
        assert!(
            scan.hits.iter().all(|hit| hit.content != top),
            "a deleted chunk must not come back through the index"
        );
        assert_eq!(scan.hits.len(), 5, "and the page is still full");
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
