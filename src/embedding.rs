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
    /// index paths, which never read the rows they exclude.
    pub other_models: Vec<String>,
    /// Which ranking path produced `hits`.
    pub path: RankPath,
    /// How many candidates an index path nominated before the float re-score
    /// picked `hits` out of them. `None` on the scan, which nominates nothing —
    /// it scores every vector there is.
    ///
    /// Reported because it is the one knob behind an approximate answer: a
    /// [`RankPath::Quantised`] result is only as good as the pool the
    /// quantisation put the right rows into, and a caller comparing two
    /// engines' answers is owed the number rather than left to infer it.
    pub candidates: Option<usize>,
    /// Set only when a corpus HAS a shadow index that was not used, and says
    /// why. A search that quietly stops being fast is indistinguishable from
    /// one that was never made fast, and the two need completely different
    /// operator responses — see [`crate::vec0::usable`].
    pub index_note: Option<String>,
}

/// How a [`VectorScan`] was ranked.
///
/// [`Scan`] and [`Vec0`] return the same top-k with the same scores; they differ
/// only in what they read to get there — the whole `chunks` table against just
/// the vectors, 15.6 GB against 4.12 GB on the live corpus and roughly an order
/// of magnitude in latency (`docs/ARCHITECTURE.md`). Reported to the caller
/// because a 50 s answer and a 4 s one are otherwise the same JSON.
///
/// [`Quantised`] is the one that does NOT return the same list, and it is
/// reachable only from [`rank_chunks_fast`] — see there.
///
/// [`Scan`]: RankPath::Scan
/// [`Vec0`]: RankPath::Vec0
/// [`Quantised`]: RankPath::Quantised
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankPath {
    /// The exhaustive cosine scan. Always available; the only path a corpus
    /// without a shadow index has.
    Scan,
    /// k-NN over the exact `vec0` shadow index, with the candidates re-scored
    /// against the stored BLOBs. Same answer as the scan.
    Vec0,
    /// k-NN over the corpus' QUANTISED shadow index, with the candidates
    /// re-scored against the stored BLOBs. The scores are exact; the SET of
    /// rows they were computed over is what the quantisation nominated, so this
    /// answer can differ from the scan's.
    Quantised(crate::vec0::Tier),
}

impl RankPath {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Scan => "scan",
            Self::Vec0 => "vec0",
            Self::Quantised(crate::vec0::Tier::Int8) => "vec0_int8",
            Self::Quantised(crate::vec0::Tier::Bit) => "vec0_bit",
            // Unreachable: `vec0::usable` declines a float index in the
            // quantised slot, which is the only way this pairing is built.
            Self::Quantised(crate::vec0::Tier::Float) => "vec0",
        }
    }

    /// Whether this path's `hits` are the scan's own answer.
    ///
    /// The question every consumer of an approximate result has to be able to
    /// ask, answered from the path rather than from a table of path names kept
    /// somewhere else.
    pub fn is_exact(self) -> bool {
        !matches!(self, Self::Quantised(_))
    }
}

/// Rank a corpus' chunk embeddings against `query` by cosine similarity —
/// EXACTLY.
///
/// The entry point for every ranking surface that promises the corpus' true
/// top-k, and the reason that promise survives the quantised tier: this looks in
/// the exact slot ([`crate::vec0::Slot::Exact`]) and nowhere else. A corpus
/// carrying a usable float `vec0` index is ranked by [`rank_by_index`] over its
/// candidates; every other corpus — one that has not been through `llm-index
/// vector-index`, one whose index cannot be vouched for, and one carrying only a
/// QUANTISED index — is ranked by [`scan_chunks`], exactly as before any index
/// existed. Building a fast index can therefore never change what this returns.
///
/// The capability is discovered from the corpus itself rather than configured:
/// there is no flag to get wrong, and a corpus that loses (or never gains) its
/// index degrades in latency and in nothing else. Both paths return the same
/// hits with the same scores in the same order; [`VectorScan::path`] reports
/// which one ran. [`rank_chunks_fast`] is the opted-into approximate sibling.
pub fn rank_chunks(
    connection: &Connection,
    model: &str,
    query: &[f32],
    limit: usize,
) -> Result<VectorScan> {
    let slot = crate::vec0::Slot::Exact;
    match crate::vec0::usable(connection, slot, model, query.len())? {
        crate::vec0::Usable::Ready(state) => {
            let pool = limit.clamp(1, MAX_HITS) + KNN_TIE_MARGIN;
            let candidates =
                crate::vec0::knn(connection, state.tier, &vector_to_bytes(query), pool)?;
            rank_by_index(connection, &state, candidates, query, limit, RankPath::Vec0)
        }
        crate::vec0::Usable::Absent => scan_chunks(connection, model, query, limit, None),
        crate::vec0::Usable::Declined(reason) => {
            scan_chunks(connection, model, query, limit, Some(reason))
        }
    }
}

/// Rank against the corpus' QUANTISED shadow index, falling back to the exact
/// path when it has none.
///
/// This is the only function that can reach a quantised index, and it exists
/// because [`rank_chunks`] must not: a quantised k-NN returns a DIFFERENT set of
/// rows from the scan, so serving it under the same request would silently
/// change what search means. A caller opts in per request
/// (`mode=semantic_fast`), gets [`RankPath::Quantised`] back when it was served
/// that way, and gets the exact answer with a [`VectorScan::index_note`] when
/// the corpus has no quantised index to serve it from.
///
/// What it buys, and what it costs, are measured on the live corpora in
/// `docs/ARCHITECTURE.md`: the quantised tiers read a quarter (int8) or a
/// thirty-second (bit) of the bytes the exact tier reads, and their answers are
/// reported there as recall@10 against the scan's own top-10.
///
/// The candidates it nominates are re-scored from `chunks.embedding` exactly as
/// on every other path, so every `score` in the response is a true cosine
/// against the stored vector. What quantisation changes is which rows were
/// scored at all — never what a score means.
pub fn rank_chunks_fast(
    connection: &Connection,
    model: &str,
    query: &[f32],
    limit: usize,
) -> Result<VectorScan> {
    let slot = crate::vec0::Slot::Quantised;
    let state = match crate::vec0::usable(connection, slot, model, query.len())? {
        crate::vec0::Usable::Ready(state) => state,
        // No quantised index, or one that cannot be vouched for. Either way the
        // caller gets a real answer — the EXACT one, from whichever path
        // `rank_chunks` can take — and a note saying the fast path is not what
        // ran. Falling back rather than refusing is the same rule the exact
        // path follows, and it is what lets a consumer ask for `semantic_fast`
        // unconditionally against corpora that have not all been indexed.
        other => {
            let mut exact = rank_chunks(connection, model, query, limit)?;
            let quantised = match other {
                crate::vec0::Usable::Declined(reason) => reason,
                _ => NO_QUANTISED_INDEX.to_string(),
            };
            // Both reasons, when there are two: the fast path did not run AND
            // the exact index was not used either are different facts with
            // different repairs, and a caller shown only one of them cannot act
            // on the other.
            exact.index_note = Some(match exact.index_note {
                Some(existing) => {
                    format!("{quantised}; the exact index was not used either: {existing}")
                }
                None => quantised,
            });
            return Ok(exact);
        }
    };
    let limit = limit.clamp(1, MAX_HITS);
    // The query goes through the SAME quantiser the corpus did, so the k-NN
    // compares like with like. For int8 that is what makes it work at all:
    // a per-vector scale is invisible to cosine only when both sides have one.
    let query_bytes = vector_to_bytes(query);
    let encoded = state
        .encode(&query_bytes)
        .context("encoding the query into the quantised index' tier")?;
    let pool = candidate_pool(limit);
    let candidates = crate::vec0::knn(connection, state.tier, encoded.as_ref(), pool)?;
    rank_by_index(
        connection,
        &state,
        candidates,
        query,
        limit,
        RankPath::Quantised(state.tier),
    )
}

/// Said when `mode=semantic_fast` reaches a corpus that has no quantised index.
///
/// `int8` and not `bit` in the suggestion: it is the tier the measurements
/// picked, because it is the only one that answers with the corpus' real top-k
/// (`docs/ARCHITECTURE.md`).
const NO_QUANTISED_INDEX: &str = "this corpus has no quantised shadow index, so the exact path \
                                  answered; build one with `llm-index vector-index --tier int8`";

/// Candidates a quantised k-NN nominates for a `limit`-hit page.
///
/// Oversampling is what turns an approximate index into an accurate answer: the
/// quantisation only has to put the right rows somewhere in the pool, and the
/// float re-score then picks the same top-k out of it that the scan would.
///
/// The multiplier is MEASURED, not chosen. `docs/ARCHITECTURE.md` carries
/// recall@10 against the exact answer at pools from 10 to 1,000 over the live
/// 2.68 M-vector corpus, for both quantisations. On the `int8` tier recall is
/// 0.9750 at pool 10 and **1.0000 from pool 20 upward**, so the bar is cleared
/// by every pool this can produce; latency is flat to pool 100 (best 1,655 ->
/// 1,750 ms) and then climbs with the re-score's keyed row reads (2,396 ms at
/// pool 200, 3,257 ms at pool 1,000). Ten is the largest multiplier still on the
/// flat part, i.e. all the margin that is free.
///
/// The pool is nearly free in the k-NN itself, whose work is reading 2.68 M
/// vectors rather than the size of its heap; what it costs is one keyed `chunks`
/// row read per candidate in the re-score, which is why it is bounded rather
/// than simply large.
const CANDIDATE_OVERSAMPLE: usize = 10;

/// Ceiling on the candidate pool, at the point where the measured latency starts
/// climbing. Reached at `limit` 20 — the `/corpus/search` default — and above.
const MAX_CANDIDATES: usize = 200;

fn candidate_pool(limit: usize) -> usize {
    limit
        .saturating_mul(CANDIDATE_OVERSAMPLE)
        .clamp(limit, MAX_CANDIDATES)
}

/// Rank a k-NN's candidate list by re-scoring it against `chunks.embedding`.
///
/// The second step is what keeps every index path honest: the index picks the
/// CANDIDATES, and their scores are then recomputed from `chunks.embedding`
/// with [`cosine_bytes`] — the scan's own arithmetic, on the scan's own bytes —
/// and ordered by the scan's own [`Ranked`] comparison. So a `score` in a
/// response is the same number whichever path produced it, and no index can
/// introduce a ranking of its own. On the exact tier that makes the answer
/// identical to the scan's; on a quantised tier it makes every score exact even
/// where the SET differs.
///
/// The exact tier's candidate list is deliberately wider than `limit`
/// ([`KNN_TIE_MARGIN`]). A k-NN returns SOME k nearest rows; where several
/// vectors tie at the k-th distance, which of them it returns is its traversal
/// order rather than the scan's "lower `chunks.id` wins". Over-fetching lets the
/// re-score resolve those ties the scan's way, and costs nothing — a `vec0`
/// k-NN's work is reading the vectors, not the size of its heap. Ties running
/// deeper than the margin (more than 100 chunks at exactly the boundary
/// distance, i.e. that many byte-identical vectors) can still pick a different
/// member of the tie than the scan would; the hits are equally correct, and the
/// alternative is giving up the index.
fn rank_by_index(
    connection: &Connection,
    state: &crate::vec0::IndexState,
    candidates: Vec<i64>,
    query: &[f32],
    limit: usize,
    path: RankPath,
) -> Result<VectorScan> {
    let limit = limit.clamp(1, MAX_HITS);
    let width = query.len() * 4;
    let query_norm = norm(query);
    let nominated = candidates.len();
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
        // `vec0::build` filters on. Reporting the recorded counts keeps the
        // paths' responses the same shape without reading 2.7 M rows to
        // recount what the corpus already knows.
        compared: state.vectors,
        skipped: state.chunks.saturating_sub(state.vectors),
        other_models: Vec::new(),
        path,
        candidates: Some(nominated),
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
        candidates: None,
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
        match crate::vec0::usable(
            &connection,
            crate::vec0::Slot::Exact,
            EMBEDDING_MODEL,
            query.len(),
        )
        .unwrap()
        {
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
        crate::vec0::build(
            &mut connection,
            crate::vec0::Tier::Float,
            MODEL,
            3,
            |_, _| {},
        )
        .unwrap();
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
        for _ in 0..2 {
            crate::vec0::build(
                &mut connection,
                crate::vec0::Tier::Float,
                MODEL,
                3,
                |_, _| {},
            )
            .unwrap();
        }
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
        crate::vec0::build(
            &mut connection,
            crate::vec0::Tier::Float,
            MODEL,
            3,
            |_, _| {},
        )
        .unwrap();
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
        crate::vec0::build(
            &mut connection,
            crate::vec0::Tier::Float,
            MODEL,
            3,
            |_, _| {},
        )
        .unwrap();
        let top = rank_chunks(&connection, MODEL, &PROBE, 1).unwrap().hits[0]
            .content
            .clone();
        // Delete the winner's chunk row without touching the index, then keep
        // the staleness witness in step so the index is still used.
        connection
            .execute("DELETE FROM chunks WHERE content=?1", [&top])
            .unwrap();
        let mut state = crate::vec0::state(&connection, crate::vec0::Slot::Exact)
            .unwrap()
            .unwrap();
        state.chunks -= 1;
        crate::vec0::write_state(&connection, crate::vec0::Slot::Exact, &state).unwrap();

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

    // ---- the quantised tier ------------------------------------------------
    //
    // Everything below is about ONE question: a quantised index returns a
    // different set of rows from the scan, so how different, and does the
    // caller ever get it without asking? The live-corpus numbers are in
    // `docs/ARCHITECTURE.md` and come from `quantised_recall_over_a_real_corpus`
    // at the bottom; the fixtures here pin the behaviour that has to hold
    // whatever those numbers are.

    /// A deterministic 32-bit LCG. No `rand` dependency, and a failure here is
    /// always the same failure.
    struct Lcg(u32);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 8) as f32 / (1u32 << 23) as f32 - 0.5
        }

        fn unit(&mut self, width: usize) -> Vec<f32> {
            let vector = (0..width).map(|_| self.next()).collect::<Vec<_>>();
            let length = norm(&vector);
            vector.into_iter().map(|value| value / length).collect()
        }
    }

    /// The width the shipped model embeds at, and the one bit quantisation has
    /// to be able to pack.
    const WIDTH: usize = 384;

    /// Documents per topic, and topics, in the fixture corpora below.
    const PER_TOPIC: usize = 25;

    /// How far a document sits from its topic's centre, and a query from the
    /// topic it is asking about. Chosen so intra-topic cosine lands near 0.75 —
    /// close enough that the ten nearest are genuinely the same subject, far
    /// enough apart that ranking them is a real task rather than a tie.
    const JITTER: f32 = 0.15;

    /// How much of every vector is the corpus' shared "hub" direction.
    ///
    /// The single most important number in this fixture. Text-embedding models
    /// do not produce vectors spread evenly over the sphere: they produce
    /// vectors clustered around one dominant mean direction, so any two
    /// documents already agree strongly before their topics are considered.
    /// A fixture without that is not a text corpus, and it cannot exercise the
    /// thing that makes `Tier::Bit` work — a sign bit taken about the ORIGIN
    /// carries information only if the corpus straddles the origin, and a
    /// hubbed corpus does not.
    const HUB: f32 = 2.0;

    /// A corpus shaped like a real one: one shared hub direction, `clusters`
    /// topics around it, [`PER_TOPIC`] documents around each topic, every vector
    /// normalised. The topic centres come back so a test can ask a question that
    /// HAS an answer.
    ///
    /// Every part of that shape is load-bearing, and none is decoration:
    ///
    /// * uniformly random 384-d vectors would be a dishonest CORPUS — they are
    ///   all nearly orthogonal, so a top-10 is ten members of one
    ///   undifferentiated cloud, and they straddle the origin in every dimension
    ///   so binary quantisation looks better here than it is;
    /// * a uniformly random QUERY would be a dishonest question — it sits near
    ///   nothing, so its top-10 is separated by cosine noise smaller than any
    ///   8-bit (let alone 1-bit) quantiser's rounding, and every quantisation
    ///   scores badly on it for a reason that has nothing to do with search.
    ///   Measured here, that fixture put `Tier::Bit` at recall@10 0.79 — a fact
    ///   about asking a corpus of documents about nothing, not about the tier.
    ///
    /// So the tests below query NEAR a topic centre, exactly as a real query
    /// lands near a subject, and the ranking task is discriminating between that
    /// subject's own documents — which is the task quantisation has to survive.
    fn clustered_corpus(clusters: usize) -> (Connection, Vec<Vec<f32>>) {
        let mut rng = Lcg(20_260_726);
        let hub = rng.unit(WIDTH);
        let mut rows = Vec::with_capacity(clusters * PER_TOPIC);
        let mut centres = Vec::with_capacity(clusters);
        let mut id = 1i64;
        for _ in 0..clusters {
            let topic = rng.unit(WIDTH);
            let centre = normalise(
                &hub.iter()
                    .zip(&topic)
                    .map(|(shared, own)| HUB * shared + own)
                    .collect::<Vec<_>>(),
            );
            for _ in 0..PER_TOPIC {
                rows.push((id, id, MODEL, jitter_around(&centre, &mut rng)));
                id += 1;
            }
            centres.push(centre);
        }
        (corpus(&rows), centres)
    }

    /// `centre` displaced by [`JITTER`] of noise, renormalised.
    fn jitter_around(centre: &[f32], rng: &mut Lcg) -> Vec<f32> {
        normalise(
            &centre
                .iter()
                .map(|value| value + JITTER * rng.next())
                .collect::<Vec<_>>(),
        )
    }

    fn normalise(vector: &[f32]) -> Vec<f32> {
        let length = norm(vector);
        vector.iter().map(|value| value / length).collect()
    }

    /// Share of the exact top-`k` that `candidate` also returned.
    ///
    /// Keyed on `(path, chunk_index)`, which identifies a chunk, and NOT on its
    /// text: a real corpus is full of chunks whose content is byte-identical
    /// (boilerplate headers, empty-ish OCR pages, the same document filed
    /// twice). Keying on text silently collapses those into one truth entry and
    /// counts every hit that matches any of them, which reports recall above
    /// 1.0 — as this measurement did until it keyed on identity.
    fn recall(exact: &VectorScan, candidate: &VectorScan) -> f64 {
        let key = |hit: &VectorHit| (hit.path.clone(), hit.chunk_index);
        let truth = exact.hits.iter().map(key).collect::<BTreeSet<_>>();
        if truth.is_empty() {
            return 1.0;
        }
        let found = candidate
            .hits
            .iter()
            .filter(|hit| truth.contains(&key(hit)))
            .count();
        found as f64 / truth.len() as f64
    }

    /// The recall bar this feature ships against — see plan §6 Q1.
    const RECALL_BAR: f64 = 0.95;

    /// The floor `Tier::Bit` has to stay above on this fixture.
    ///
    /// Not a bar it ships against — it does not clear [`RECALL_BAR`] here or on
    /// the live corpus, and `docs/ARCHITECTURE.md` says so with the numbers.
    /// This is the line between "coarse" and "broken": an uncentred bit index,
    /// or one whose packing disagreed with `sqlite-vec`'s Hamming distance,
    /// lands far below it, so it catches the failures that would otherwise look
    /// like the tier merely being lossy.
    const BIT_FLOOR: f64 = 0.5;

    #[test]
    fn a_quantised_index_clears_the_recall_bar_and_rebuilds_to_the_same_answer() {
        // Rebuild equivalence, stated the only way it can be stated for a lossy
        // index: not "the same list" — it is not the same list — but "the same
        // list to within the bar each tier is held to", reproduced by a rebuild
        // from the same BLOBs. Both quantisations, over 20 queries drawn from
        // the corpus' own geometry.
        //
        // The two bars differ because the measurements differ. `Int8` returns
        // the exact top-10 on the live 2.68 M-vector corpus (recall@10 1.0000
        // from pool 20 up) and is what `semantic_fast` is built on; `Bit` is
        // three times faster and does not come close (0.125 to 0.615), so what
        // is pinned here is that it still works as designed rather than that it
        // is good enough to prefer.
        let (mut connection, centres) = clustered_corpus(40);
        crate::vec0::build(
            &mut connection,
            crate::vec0::Tier::Float,
            MODEL,
            WIDTH,
            |_, _| {},
        )
        .unwrap();
        let mut rng = Lcg(7);
        let queries = centres
            .iter()
            .take(20)
            .map(|centre| jitter_around(centre, &mut rng))
            .collect::<Vec<_>>();

        for (tier, bar) in [
            (crate::vec0::Tier::Int8, RECALL_BAR),
            (crate::vec0::Tier::Bit, BIT_FLOOR),
        ] {
            // Built twice: a rebuild over an existing index is the repair path
            // an operator actually runs, and it has to land in the same place.
            let mut rounds = Vec::new();
            for _ in 0..2 {
                crate::vec0::build(&mut connection, tier, MODEL, WIDTH, |_, _| {}).unwrap();
                let mut total = 0.0;
                for query in &queries {
                    let exact = rank_chunks(&connection, MODEL, query, 10).unwrap();
                    let fast = rank_chunks_fast(&connection, MODEL, query, 10).unwrap();
                    assert_eq!(fast.path, RankPath::Quantised(tier));
                    assert!(!fast.path.is_exact());
                    assert_eq!(fast.hits.len(), exact.hits.len());
                    assert_eq!(fast.candidates, Some(candidate_pool(10)));
                    total += recall(&exact, &fast);
                }
                rounds.push(total / queries.len() as f64);
            }
            assert!(rounds[0] >= bar, "{tier:?} recall@10 {:.4}", rounds[0]);
            assert_eq!(
                rounds[0], rounds[1],
                "{tier:?}: a rebuild from the same BLOBs must reproduce the same answer"
            );
        }
    }

    #[test]
    fn the_quantised_path_scores_every_hit_from_the_stored_float_vector() {
        // The invariant that survives quantisation: what changes is which rows
        // were scored, never what a score means. Every hit the fast path
        // returns carries the cosine the scan would have computed for it, to
        // the bit.
        let (mut connection, centres) = clustered_corpus(10);
        crate::vec0::build(
            &mut connection,
            crate::vec0::Tier::Bit,
            MODEL,
            WIDTH,
            |_, _| {},
        )
        .unwrap();
        let mut rng = Lcg(11);
        let query = jitter_around(&centres[3], &mut rng);

        let fast = rank_chunks_fast(&connection, MODEL, &query, 10).unwrap();
        let exact = scan_chunks(&connection, MODEL, &query, MAX_HITS, None).unwrap();
        assert!(!fast.hits.is_empty());
        for hit in &fast.hits {
            let scanned = exact
                .hits
                .iter()
                .find(|candidate| candidate.content == hit.content)
                .unwrap_or_else(|| panic!("{} is not in the scan's top-100", hit.content));
            assert_eq!(
                hit.score.to_bits(),
                scanned.score.to_bits(),
                "{}",
                hit.content
            );
        }
        // And the page is ordered by that score, descending, like every other
        // path in this module.
        assert!(fast
            .hits
            .windows(2)
            .all(|pair| pair[0].score >= pair[1].score));
    }

    #[test]
    fn the_exact_path_never_reads_the_quantised_index() {
        // The separation the whole design rests on. A corpus carrying ONLY a
        // quantised index answers `mode=semantic` by the scan — it does not
        // quietly become approximate because an operator built a fast index.
        let mut connection = indexable_corpus(200);
        crate::vec0::build(
            &mut connection,
            crate::vec0::Tier::Int8,
            MODEL,
            3,
            |_, _| {},
        )
        .unwrap();

        let exact = rank_chunks(&connection, MODEL, &PROBE, 10).unwrap();
        let scanned = scan_chunks(&connection, MODEL, &PROBE, 10, None).unwrap();
        assert_eq!(exact.path, RankPath::Scan);
        assert!(exact.path.is_exact());
        assert_eq!(exact.index_note, None, "there is no EXACT index to explain");
        assert_eq!(
            exact
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
    }

    #[test]
    fn the_fast_path_over_a_corpus_without_a_quantised_index_answers_exactly() {
        // The absent-index fallback. A request for the fast path over a corpus
        // that has none is answered EXACTLY rather than refused, and the note
        // says both that it was and how to make it fast.
        let connection = indexable_corpus(120);
        let fast = rank_chunks_fast(&connection, MODEL, &PROBE, 10).unwrap();
        let scanned = scan_chunks(&connection, MODEL, &PROBE, 10, None).unwrap();

        assert_eq!(fast.path, RankPath::Scan);
        assert!(fast.path.is_exact());
        assert!(
            fast.index_note
                .as_deref()
                .is_some_and(|note| note.contains("no quantised shadow index")),
            "{:?}",
            fast.index_note
        );
        assert_eq!(fast.hits.len(), scanned.hits.len());
        assert_eq!(fast.hits[0].content, scanned.hits[0].content);
    }

    #[test]
    fn the_fast_path_uses_the_exact_index_when_that_is_all_the_corpus_has() {
        // Same fallback, one rung up: a corpus with a float index and no
        // quantised one serves `semantic_fast` from the float index — still
        // exact, still faster than the scan, and still labelled honestly.
        let mut connection = indexable_corpus(120);
        crate::vec0::build(
            &mut connection,
            crate::vec0::Tier::Float,
            MODEL,
            3,
            |_, _| {},
        )
        .unwrap();

        let fast = rank_chunks_fast(&connection, MODEL, &PROBE, 10).unwrap();
        assert_eq!(fast.path, RankPath::Vec0);
        assert!(fast.path.is_exact());
        assert!(fast
            .index_note
            .as_deref()
            .is_some_and(|note| note.contains("no quantised shadow index")));
    }

    #[test]
    fn a_stale_quantised_index_is_bypassed_and_the_answer_is_still_right() {
        // The staleness witness, on the quantised slot. A build without index
        // maintenance writes a genuine top hit behind the index; the fast path
        // must fall back rather than serve a page that cannot contain it.
        let mut connection = indexable_corpus(100);
        crate::vec0::build(
            &mut connection,
            crate::vec0::Tier::Int8,
            MODEL,
            3,
            |_, _| {},
        )
        .unwrap();
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

        let fast = rank_chunks_fast(&connection, MODEL, &PROBE, 5).unwrap();
        assert_eq!(fast.path, RankPath::Scan);
        assert!(
            fast.index_note
                .as_deref()
                .is_some_and(|note| note.contains("stale") && note.contains("chunks_vec_q")),
            "{:?}",
            fast.index_note
        );
        assert_eq!(fast.hits[0].content, "written behind the index");
    }

    #[test]
    fn the_candidate_pool_is_bounded_and_never_below_the_page() {
        assert_eq!(candidate_pool(10), 10 * CANDIDATE_OVERSAMPLE);
        assert_eq!(candidate_pool(1), CANDIDATE_OVERSAMPLE);
        assert_eq!(candidate_pool(MAX_HITS), MAX_CANDIDATES);
        assert!(
            candidate_pool(MAX_HITS) >= MAX_HITS,
            "a full page must never ask for fewer candidates than hits"
        );
    }

    /// Recall and latency of the quantised tiers over a real corpus — runs only
    /// when `LLM_INDEX_BENCH_CORPUS` names one, otherwise it skips, so CI needs
    /// no multi-GB fixture. Build with `--release`.
    ///
    /// This is the measurement the shipped [`CANDIDATE_OVERSAMPLE`] comes from,
    /// and the numbers it printed are in `docs/ARCHITECTURE.md`. It wants a
    /// corpus carrying BOTH indexes: the float one is the ground truth (proven
    /// equal to the scan by `scan_latency_over_a_real_corpus`) and answering
    /// 20 queries from it costs seconds instead of the ~20 minutes the same
    /// queries would cost by scanning.
    ///
    /// `FASTEMBED_CACHE_DIR` pointing at a staged `multilingual-e5-small` makes
    /// the queries REAL query embeddings, which is the only honest way to
    /// measure recall — a random unit vector is nowhere near the manifold a
    /// query lands on, and quantisation error is a property of where you are on
    /// it. Without the model the test says so and falls back to stored corpus
    /// vectors, which is the next best thing and clearly labelled.
    #[test]
    fn quantised_recall_over_a_real_corpus() {
        let Ok(path) = std::env::var("LLM_INDEX_BENCH_CORPUS") else {
            eprintln!("skipping quantised recall measurement: LLM_INDEX_BENCH_CORPUS unset");
            return;
        };
        crate::vec0::register();
        let connection = rusqlite::Connection::open_with_flags(
            &path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .expect("open corpus read-only");
        let quantised = match crate::vec0::usable(
            &connection,
            crate::vec0::Slot::Quantised,
            EMBEDDING_MODEL,
            WIDTH,
        )
        .unwrap()
        {
            crate::vec0::Usable::Ready(state) => state,
            crate::vec0::Usable::Absent => {
                eprintln!("no quantised shadow index in this corpus; nothing to measure");
                return;
            }
            crate::vec0::Usable::Declined(reason) => {
                eprintln!("quantised index not usable: {reason}");
                return;
            }
        };
        let exact_path = crate::vec0::usable(
            &connection,
            crate::vec0::Slot::Exact,
            EMBEDDING_MODEL,
            WIDTH,
        )
        .unwrap();
        eprintln!(
            "corpus {path}: {} vectors over {} chunks, quantised tier {}, exact index {}",
            quantised.vectors,
            quantised.chunks,
            quantised.tier.as_str(),
            match &exact_path {
                crate::vec0::Usable::Ready(_) => "present (ground truth is the k-NN)".to_string(),
                crate::vec0::Usable::Absent => "absent (ground truth is the full scan)".to_string(),
                crate::vec0::Usable::Declined(reason) => format!("declined: {reason}"),
            }
        );

        let queries = bench_queries(&connection);
        // The exact answer, once per query, reused by every pool size.
        let mut truth = Vec::new();
        let mut exact_ms = Vec::new();
        for query in &queries {
            let started = std::time::Instant::now();
            let scan = rank_chunks(&connection, EMBEDDING_MODEL, query, 10).unwrap();
            exact_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            truth.push(scan);
        }
        eprintln!(
            "exact path ({}) over {} queries: {}",
            truth[0].path.as_str(),
            queries.len(),
            summarise(&exact_ms)
        );

        // The SHIPPED path, exactly as `mode=semantic_fast` calls it, before the
        // sweep below explores the pools around it. Reported first so the table
        // can be read as "and here is why that is the pool it uses".
        let mut shipped_ms = Vec::new();
        let mut shipped_recall = Vec::new();
        for (query, exact) in queries.iter().zip(&truth) {
            let started = std::time::Instant::now();
            let fast = rank_chunks_fast(&connection, EMBEDDING_MODEL, query, 10).unwrap();
            shipped_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(fast.path, RankPath::Quantised(quantised.tier));
            assert_eq!(fast.candidates, Some(candidate_pool(10)));
            shipped_recall.push(recall(exact, &fast));
        }
        eprintln!(
            "SHIPPED rank_chunks_fast (limit 10, pool {}): recall@10 mean {:.4} worst {:.4}; {}",
            candidate_pool(10),
            shipped_recall.iter().sum::<f64>() / shipped_recall.len() as f64,
            shipped_recall.iter().cloned().fold(f64::INFINITY, f64::min),
            summarise(&shipped_ms)
        );

        // Pool 10 is the "no rerank headroom" pattern — the quantised k-NN's own
        // top-10, re-scored but never given anything to choose between. Every
        // larger pool is the rerank pattern at a different oversample.
        for pool in [10usize, 20, 50, 100, 200, 500, 1_000] {
            let mut recalls = Vec::new();
            let mut millis = Vec::new();
            for (query, exact) in queries.iter().zip(&truth) {
                let bytes = vector_to_bytes(query);
                let encoded = quantised.encode(&bytes).unwrap();
                let started = std::time::Instant::now();
                let candidates =
                    crate::vec0::knn(&connection, quantised.tier, encoded.as_ref(), pool).unwrap();
                let fast = rank_by_index(
                    &connection,
                    &quantised,
                    candidates,
                    query,
                    10,
                    RankPath::Quantised(quantised.tier),
                )
                .unwrap();
                millis.push(started.elapsed().as_secs_f64() * 1000.0);
                recalls.push(recall(exact, &fast));
            }
            let mean = recalls.iter().sum::<f64>() / recalls.len() as f64;
            let floor = recalls.iter().cloned().fold(f64::INFINITY, f64::min);
            eprintln!(
                "{} pool {pool:>5}: recall@10 mean {mean:.4} worst {floor:.4}; {}",
                quantised.tier.as_str(),
                summarise(&millis)
            );
        }
    }

    /// `n` query vectors for the bench, embedded by the real model when one is
    /// staged and drawn from the corpus otherwise.
    fn bench_queries(connection: &Connection) -> Vec<Vec<f32>> {
        let prompts = [
            "hoa don thanh toan",
            "invoice total amount due",
            "passport scan copy",
            "bang diem dai hoc",
            "insurance policy renewal",
            "hop dong lao dong",
            "photo of a beach at sunset",
            "meeting notes action items",
            "giay khai sinh",
            "bank statement transactions",
            "curriculum vitae experience",
            "so do nha dat",
            "medical test results",
            "software licence agreement",
            "thu moi hop",
            "tax return filing",
            "receipt for equipment purchase",
            "danh sach nhan vien",
            "travel itinerary booking",
            "warranty certificate",
        ];
        let config = crate::config::Config::default();
        match Embedder::new(&config) {
            Ok(mut embedder) => {
                eprintln!("queries: {} real embeddings of real prompts", prompts.len());
                prompts
                    .iter()
                    .map(|prompt| embedder.embed_query(prompt).unwrap())
                    .collect()
            }
            Err(error) => {
                eprintln!(
                    "queries: NO embedding model ({error:#}) — falling back to {} stored corpus \
                     vectors, which sit on the passage manifold rather than the query one",
                    prompts.len()
                );
                let mut statement = connection
                    .prepare(
                        "SELECT embedding FROM chunks WHERE model=?1 AND id % 9973 = 0 LIMIT ?2",
                    )
                    .unwrap();
                statement
                    .query_map(
                        rusqlite::params![EMBEDDING_MODEL, prompts.len() as i64],
                        |row| {
                            let blob = row.get_ref(0)?.as_blob()?;
                            Ok(blob
                                .chunks_exact(4)
                                .map(|bytes| {
                                    f32::from_le_bytes(bytes.try_into().expect("four bytes"))
                                })
                                .collect::<Vec<f32>>())
                        },
                    )
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap()
            }
        }
    }

    /// `best / median / worst` of a set of millisecond timings.
    fn summarise(millis: &[f64]) -> String {
        let mut sorted = millis.to_vec();
        sorted.sort_by(f64::total_cmp);
        format!(
            "best {:.0} ms, median {:.0} ms, worst {:.0} ms",
            sorted[0],
            sorted[sorted.len() / 2],
            sorted[sorted.len() - 1]
        )
    }
}
