//! The OPTIONAL `vec0` shadow indexes over `chunks.embedding`.
//!
//! Semantic search without one is an exhaustive cosine scan: every stored vector
//! is read and scored, which is exact but costs a full pass over the `chunks`
//! table — 15.6 GB of it on the 2.68 M-vector live corpus. This module adds
//! purely derived copies of those vectors in `sqlite-vec` [`vec0`] virtual
//! tables, laid out as contiguous chunk blobs rather than one row per vector, so
//! a query reads only vectors and nothing else.
//!
//! A corpus can carry two of them, and they answer different questions — see
//! [`Slot`]:
//!
//! * the **exact** slot ([`Tier::Float`]) holds a full-precision copy. Its k-NN
//!   nominates candidates that a float re-score then orders exactly as the scan
//!   would, so the default query path uses it without changing a single result.
//!   It is a layout win, not an algorithmic one: `vec0` 0.1.9 has no ANN
//!   structure and still visits every vector. Measured, it is roughly an order
//!   of magnitude faster than the scan — and still seconds, not milliseconds, at
//!   2.68 M vectors.
//! * the **quantised** slot ([`Tier::Int8`], [`Tier::Bit`]) holds a lossy copy —
//!   a quarter or a thirty-second of the bytes. It is what gets a 2.68 M-vector
//!   corpus under a second, and it CHANGES which rows come back, so it serves
//!   only `mode=semantic_fast` and never the default path. `docs/ARCHITECTURE.md`
//!   carries the measured recall and latency of both quantisations.
//!
//! Four properties are load-bearing, and every function here exists to keep one
//! of them:
//!
//! * **Optional.** An index is a table that either exists in a corpus or does
//!   not. A corpus without one is byte-identical to what this build produced
//!   before the indexes existed, answers semantic queries through the scan, and
//!   pays nothing at index time. Nothing creates a table implicitly — only
//!   `llm-index vector-index` does.
//! * **Derived.** Every vector in one is computed from a `chunks.embedding`
//!   BLOB, so it can be dropped and rebuilt from the corpus at any time without
//!   re-embedding a single document ([`build`]). Losing one loses no data.
//! * **Never trusted blindly.** An index is used only when the corpus can prove
//!   it is still complete ([`usable`]). A build that does not maintain it —
//!   every release before this feature — can write `chunks` rows underneath it,
//!   and serving a k-NN answer from an index that is missing those rows would be
//!   silently wrong. The scan is always there to fall back to.
//! * **Never silently approximate.** A quantised index cannot reach the default
//!   query path at all. `rank_chunks` looks in the exact slot and nowhere else;
//!   only `rank_chunks_fast`, behind its own request mode, looks in the
//!   quantised one. See [`crate::embedding`].
//!
//! [`vec0`]: https://github.com/asg017/sqlite-vec
//! [`rank_chunks`]: crate::embedding::rank_chunks
//! [`rank_chunks_fast`]: crate::embedding::rank_chunks_fast

use std::borrow::Cow;
use std::sync::Once;

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// One of the two shadow-index slots a corpus can carry.
///
/// Two slots rather than one table per representation, because a slot is a
/// ROLE: the exact slot is the one a result-preserving query path may read, the
/// quantised slot is the one only an opted-in approximate path may read. Which
/// quantisation is in the quantised slot is a property of what is stored there
/// ([`IndexState::tier`]), not of the slot — and a corpus holds at most one at a
/// time, because building a second replaces the first. That is deliberate: with
/// one quantised index per corpus there is never a question of which one
/// answered a query, and never a pair of them to keep in step with `chunks`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    /// Full-precision vectors. Safe for the default query path.
    Exact,
    /// Quantised vectors. Reachable only from `mode=semantic_fast`.
    Quantised,
}

impl Slot {
    /// The virtual table this slot lives in.
    ///
    /// `sqlite-vec` puts its own storage in `<table>_*` shadow tables beside it
    /// (`_info`, `_chunks`, `_rowids`, `_vector_chunks00`). All of them are
    /// ordinary tables in the same file, so a corpus with an index is still one
    /// portable `.sqlite` file with no sidecars.
    pub fn table(self) -> &'static str {
        match self {
            Self::Exact => "chunks_vec",
            Self::Quantised => "chunks_vec_q",
        }
    }

    /// `meta` key describing what is in this slot — see [`IndexState`].
    ///
    /// The corpus' schema-version idiom is exactly this: `meta` keys plus
    /// `IF NOT EXISTS` tables (`store::SCHEMA`) and re-applied column lists
    /// (`store::ADDED_FILE_COLUMNS`). There is no monotonic schema number to
    /// bump, and inventing one here would leave every existing corpus claiming a
    /// version it was never stamped with. A corpus without the key simply has
    /// nothing in the slot.
    pub fn meta_key(self) -> &'static str {
        match self {
            Self::Exact => "vec0_index",
            Self::Quantised => "vec0_index_q",
        }
    }

    /// `meta` key holding the CENTRE a tier that needs one was built with — see
    /// [`IndexState::centre`].
    ///
    /// A second key rather than a field of the marker, because the two have
    /// opposite lifetimes: the centre is measured once per build and never moves
    /// again, while the marker is rewritten inside every per-file savepoint an
    /// index job commits. Folding 384 floats into the marker would put a few
    /// kilobytes of unchanged JSON into every one of those writes.
    pub fn centre_key(self) -> &'static str {
        match self {
            Self::Exact => "vec0_index_centre",
            Self::Quantised => "vec0_index_q_centre",
        }
    }

    /// The command that repairs this slot, quoted into decline messages so an
    /// operator reading a fallback reason is told what to run.
    fn rebuild_hint(self) -> &'static str {
        match self {
            Self::Exact => "`llm-index vector-index --rebuild`",
            Self::Quantised => "`llm-index vector-index --tier <int8|bit> --rebuild`",
        }
    }

    /// Both slots, for the callers that have to treat a corpus as a whole: the
    /// writer that maintains whatever it finds, and the CLI that reports it.
    pub const ALL: [Self; 2] = [Self::Exact, Self::Quantised];
}

/// How a slot stores its copy of the vectors.
///
/// The two quantisers answer to different rules, and the difference is the
/// single most important thing in this module.
///
/// **`Int8` has to preserve cosine, so it may scale and must not offset.**
/// Cosine is invariant under multiplying a vector by a positive scalar and is
/// not invariant under adding anything to it, so a pure per-vector scale changes
/// the angles only by its rounding while an offset changes them systematically.
/// That rules out `sqlite-vec`'s own `vec_quantize_int8(v, 'unit')` here twice
/// over: it maps the fixed interval `[-1, 1]` onto the code range with an affine
/// formula — an offset of about half a code, and, far worse, a range 20 times
/// wider than the data. The components of a unit-norm 384-d embedding sit around
/// ±0.05, so `'unit'` spends about 13 of its 256 codes on the entire corpus and
/// throws the other 243 away. [`Tier::Int8`] scales each vector by its own
/// largest component instead: the full code range for every vector, no
/// corpus-wide calibration, and clipping is impossible.
///
/// **`Bit` is not computing cosine at all, so it must offset.** Its Hamming
/// distance is a *nominator*, not a score — the float re-score does the ranking —
/// and what a sign bit is worth is how evenly the corpus splits on it. Text
/// embeddings have a large shared mean ("hub") direction, so most dimensions
/// agree in sign across almost every vector and their bits carry nothing.
/// Measured on the 2.68 M-vector live corpus, uncentred sign quantisation scored
/// **recall@10 0.010 at pool 10 and 0.090 at pool 1,000** — noise. Subtracting
/// the corpus mean first ([`IndexState::centre`]) multiplies that by three to
/// six (**0.125 and 0.615** at the same pools), which is the difference between
/// broken and merely coarse.
///
/// **Coarse is what it stays**, and `docs/ARCHITECTURE.md` says so with the
/// numbers: 384 bits is 32x compression, and on this corpus it does not put the
/// true top-10 into any pool the re-score can afford. `Int8` is the tier to
/// build; `Bit` is here because it is the only sub-second one, because it is the
/// evidence behind that claim, and because a smaller corpus is a different
/// question — see the second row of the table there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// The stored `f32` vectors, copied verbatim. Lossless: a k-NN over this
    /// tier nominates candidates whose float re-score reproduces the scan.
    #[default]
    Float,
    /// One signed byte per dimension, `round(v_i * 127 / max_j|v_j|)`.
    ///
    /// A quarter of the bytes. The per-vector divisor is what makes this a
    /// cosine-safe quantiser: every vector is scaled by its own positive
    /// scalar, which cosine does not see, so the only error is the rounding —
    /// and the rounding is as small as 8 bits allow, because every vector's
    /// largest component lands exactly on the rail at ±127.
    Int8,
    /// One BIT per dimension, set when `v_i - centre_i > 0.0`, compared by
    /// Hamming distance.
    ///
    /// A thirty-second of the bytes, and the only tier that answers the largest
    /// live corpus in under a second (best 143 ms at pool 10, 399 ms at pool
    /// 200). It keeps a sign pattern and discards every magnitude, and at 384
    /// dimensions that is not enough to find the true top-10: measured recall@10
    /// on that corpus is 0.125 at pool 10 and 0.615 at pool 1,000, so it does
    /// NOT clear the 0.95 bar at any pool worth paying for. Build `Int8` unless
    /// a coarse net is what you want. The `centre` is not decoration either —
    /// without it these numbers fall by another factor of six. See the
    /// type-level docs and `docs/ARCHITECTURE.md`.
    Bit,
}

impl Tier {
    /// The slot this tier belongs in. Total, not a lookup: a tier IS exact or
    /// approximate, and the slot is that fact.
    pub fn slot(self) -> Slot {
        match self {
            Self::Float => Slot::Exact,
            Self::Int8 | Self::Bit => Slot::Quantised,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Float => "float",
            Self::Int8 => "int8",
            Self::Bit => "bit",
        }
    }

    /// The `vec0` column declaration for `dimensions`-wide vectors.
    ///
    /// `distance_metric=cosine` matches how the corpus is ranked, and is
    /// rejected outright by `sqlite-vec` for bit vectors, which only ever
    /// compare by Hamming distance. Neither choice can change a reported number:
    /// every score a caller sees is recomputed from the stored `f32` BLOB by
    /// [`crate::embedding`], so a metric declared here only decides which rows
    /// are nominated.
    fn column(self, dimensions: usize) -> String {
        match self {
            Self::Float => format!("embedding float[{dimensions}] distance_metric=cosine"),
            Self::Int8 => format!("embedding int8[{dimensions}] distance_metric=cosine"),
            Self::Bit => format!("embedding bit[{dimensions}]"),
        }
    }

    /// Bytes one vector occupies in this tier.
    pub fn bytes_per_vector(self, dimensions: usize) -> usize {
        match self {
            Self::Float => dimensions * 4,
            Self::Int8 => dimensions,
            Self::Bit => dimensions / 8,
        }
    }

    /// How `placeholder` has to appear in SQL for `sqlite-vec` to read a bound
    /// BLOB as THIS tier.
    ///
    /// Not decoration. `sqlite-vec` carries a vector's element type in the
    /// value's SQLite *subtype*, and a bound parameter has none — so a raw
    /// `int8` blob handed straight to an `int8[N]` column is read as `float32`
    /// and rejected ("Must be divisible by 4"). `vec_int8()`/`vec_bit()` are the
    /// functions that stamp the subtype, and they are the reason every INSERT
    /// and every `MATCH` in this module is written through the tier rather than
    /// with a bare `?`.
    fn bind(self, placeholder: &str) -> String {
        match self {
            Self::Float => placeholder.to_string(),
            Self::Int8 => format!("vec_int8({placeholder})"),
            Self::Bit => format!("vec_bit({placeholder})"),
        }
    }

    /// Whether this tier has to be given a corpus centre before it can encode
    /// anything — see [`IndexState::centre`].
    pub fn needs_centre(self) -> bool {
        matches!(self, Self::Bit)
    }

    /// Encode one stored `f32` BLOB into this tier's representation.
    ///
    /// `blob` is the little-endian `f32` encoding `chunks.embedding` holds, so
    /// the float tier borrows it and copies nothing — which matters, because a
    /// build of the live corpus runs this 2.68 M times.
    ///
    /// `centre` is the corpus mean, required exactly when [`Self::needs_centre`]
    /// says so and ignored otherwise. Prefer [`IndexState::encode`], which
    /// carries the right one for an index rather than leaving the caller to
    /// remember.
    ///
    /// `None` is a vector this tier cannot represent, which the caller counts as
    /// skipped rather than indexing a vector the query could never match — the
    /// same rule the scan applies per row. Three shapes reach it: a blob that is
    /// not `dimensions` floats wide, a bit encoding of a width that does not
    /// pack into whole bytes, and a centring tier handed no centre. The last two
    /// are unreachable through [`create`] and [`build`], which refuse both
    /// outright; they are checked here anyway so this is total rather than
    /// total-if-you-came-the-right-way.
    pub fn encode<'a>(
        self,
        blob: &'a [u8],
        dimensions: usize,
        centre: Option<&[f32]>,
    ) -> Option<Cow<'a, [u8]>> {
        if blob.len() != dimensions * 4 {
            return None;
        }
        Some(match self {
            Self::Float => Cow::Borrowed(blob),
            Self::Int8 => Cow::Owned(quantise_int8(blob)),
            Self::Bit => {
                if !dimensions.is_multiple_of(8) {
                    return None;
                }
                let centre = centre.filter(|centre| centre.len() == dimensions)?;
                Cow::Owned(quantise_bit(blob, centre))
            }
        })
    }
}

/// Read one little-endian `f32` out of a blob at element `index`.
fn element(blob: &[u8], index: usize) -> f32 {
    let start = index * 4;
    f32::from_le_bytes(
        blob[start..start + 4]
            .try_into()
            .expect("four bytes per f32, checked by the caller's width test"),
    )
}

/// `round(v_i * 127 / max_j|v_j|)`, one signed byte per dimension.
///
/// The divisor is per vector and is the vector's own largest magnitude, so:
///
/// * every vector uses the full code range, whatever its norm — including a
///   corpus whose producer did not normalise;
/// * nothing ever clips, because the largest component maps exactly to ±127;
/// * cosine is unchanged apart from the rounding, because scaling a vector by a
///   positive scalar does not move it.
///
/// 127 rather than 128: the range is deliberately symmetric, so that `v` and
/// `-v` quantise to exact negatives of each other and a query's cosine against
/// a vector and against its opposite still sum to zero. Using the extra code at
/// -128 would buy a fraction of a percent of resolution and break that.
///
/// A vector with no finite non-zero component — all zeros, or one carrying
/// NaN/inf from a broken producer — encodes as all zeros. `sqlite-vec` scores
/// that as NaN, which no comparison accepts, so it can only ever appear as a
/// spurious candidate; the float re-score then gives it the same -1.0 floor the
/// scan gives it and it drops out of the page.
fn quantise_int8(blob: &[u8]) -> Vec<u8> {
    let dimensions = blob.len() / 4;
    let mut peak = 0.0f32;
    for index in 0..dimensions {
        let value = element(blob, index);
        if value.is_finite() && value.abs() > peak {
            peak = value.abs();
        }
    }
    if peak == 0.0 {
        return vec![0u8; dimensions];
    }
    let scale = 127.0 / peak;
    (0..dimensions)
        .map(|index| {
            let value = element(blob, index);
            if !value.is_finite() {
                return 0u8;
            }
            // `clamp` guards only the non-finite-adjacent edges; the arithmetic
            // itself cannot exceed ±127, since `peak` is the largest magnitude.
            ((value * scale).round().clamp(-127.0, 127.0) as i8) as u8
        })
        .collect()
}

/// One bit per dimension, set when the component is strictly above `centre`.
///
/// The packing is `sqlite-vec`'s own — bit `i` in byte `i / 8` at position
/// `i % 8`, the layout `vec_quantize_binary` produces and the Hamming distance
/// expects. What differs is the threshold: `centre_i` rather than zero.
///
/// That subtraction is the whole tier. A bit is worth something only if the
/// corpus splits on it, and text embeddings do not split on zero — they carry a
/// large shared mean direction, so a dimension whose mean is well away from zero
/// has the same sign in nearly every vector and its bit is a constant.
/// Thresholding at the corpus mean makes every dimension an even split by
/// construction, which is the difference between recall@10 0.09 and 0.99 on the
/// live corpus (`docs/ARCHITECTURE.md`).
///
/// NaN compares false and lands on the unset side, which is the degenerate case
/// behaving like every other below-centre component.
fn quantise_bit(blob: &[u8], centre: &[f32]) -> Vec<u8> {
    let dimensions = blob.len() / 4;
    let mut out = vec![0u8; dimensions / 8];
    for index in 0..dimensions {
        if element(blob, index) > centre[index] {
            out[index / 8] |= 1 << (index % 8);
        }
    }
    out
}

/// Vectors read to measure a corpus [`IndexState::centre`].
///
/// The mean of 50,000 samples pins each of the 384 components to about a
/// thousandth of a component's own spread, which is far finer than any decision
/// made with it: a dimension whose mean is that close to a vector's value splits
/// the corpus evenly whichever side the estimate falls, so the bit is a fair coin
/// either way. Measuring it over every vector instead would double a build,
/// because reading `chunks.embedding` means walking past `chunks.content` — the
/// reason a build of the live corpus costs minutes rather than seconds.
const CENTRE_SAMPLE: usize = 50_000;

/// Points across the `chunks.id` range the sample is drawn from.
///
/// Spread rather than taken from the front, because `chunks.id` follows
/// insertion order and insertion order follows the walk: the first 50,000 rows
/// of a live corpus are one corner of one drive, whose mean is not the corpus'.
const CENTRE_ANCHORS: usize = 200;

/// The mean of this corpus' `model` vectors, for a tier that quantises against
/// it.
///
/// `None` is a corpus holding no vectors of that model and width — nothing to
/// average, and nothing to index either.
pub fn corpus_centre(
    connection: &Connection,
    model: &str,
    dimensions: usize,
) -> Result<Option<Vec<f32>>> {
    let bounds: Option<(i64, i64)> = connection
        .query_row(
            "SELECT MIN(id),MAX(id) FROM chunks WHERE model=?1",
            [model],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?
        .filter(|(low, high): &(Option<i64>, Option<i64>)| low.is_some() && high.is_some())
        .map(|(low, high)| (low.unwrap_or_default(), high.unwrap_or_default()));
    let Some((low, high)) = bounds else {
        return Ok(None);
    };
    // f64 accumulators: 50,000 f32 components summed in f32 would lose the last
    // bits of a mean whose whole job is to sit near zero.
    let mut sums = vec![0f64; dimensions];
    let mut counted = 0usize;
    let span = (high - low).max(0) as u64;
    let per_anchor = (CENTRE_SAMPLE / CENTRE_ANCHORS).max(1);
    let mut statement = connection
        .prepare("SELECT embedding FROM chunks WHERE model=?1 AND id>=?2 ORDER BY id LIMIT ?3")?;
    for anchor in 0..CENTRE_ANCHORS {
        let from = low + (span * anchor as u64 / CENTRE_ANCHORS as u64) as i64;
        let mut rows = statement.query(params![model, from, per_anchor as i64])?;
        while let Some(row) = rows.next()? {
            let blob = row.get_ref(0)?.as_blob()?;
            if blob.len() != dimensions * 4 {
                continue;
            }
            for (index, sum) in sums.iter_mut().enumerate() {
                let value = element(blob, index);
                if value.is_finite() {
                    *sum += value as f64;
                }
            }
            counted += 1;
        }
    }
    if counted == 0 {
        return Ok(None);
    }
    Ok(Some(
        sums.into_iter()
            .map(|sum| (sum / counted as f64) as f32)
            .collect(),
    ))
}

/// Vectors per transaction during a [`build`].
///
/// A build of the live corpus writes 2.68 M vectors — 4.1 GB in the float tier.
/// Appending that in one transaction is cheap (new pages need no rollback
/// journal), but a REBUILD reuses the pages the dropped index freed, and
/// overwritten pages ARE journaled: one transaction would put a journal the size
/// of the index beside the corpus. Batching bounds it — measured 4.1 MB peak
/// across a full rebuild of that corpus — and costs nothing, because a build is
/// not a corpus edit to protect: the slot's [`Slot::meta_key`] is written only
/// after the last row lands, so a killed build leaves a part-filled table that
/// no query will use (see [`usable`]) and `--rebuild` starts over.
const BUILD_BATCH: usize = 50_000;

/// Make the `vec0` module available to every connection opened AFTER this call.
///
/// `sqlite3_auto_extension` is process-wide and one-way, which is the whole
/// reason this is called from the connection helpers (`store::connect`,
/// `store::IndexStore::open`, `service::read_only`) rather than from the places
/// that query an index: a connection opened before registration would report
/// `no such module: vec0` on a corpus that has one, and there is no way to
/// retrofit the module onto a connection that already exists.
///
/// Registering is inert for everything else. It adds `vec_*` scalar functions
/// and the `vec0` module to new connections and changes no existing SQL: no
/// table, no query and no plan in this crate references either unless a corpus
/// actually holds a shadow index.
pub fn register() {
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        // SAFETY: `sqlite3_vec_init` is the extension entry point sqlite-vec
        // exports for exactly this call, and the transmute is the signature
        // cast its own README and test use. Registration happens once, before
        // any connection this process opens is handed out.
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
                *const (),
                unsafe extern "C" fn(
                    *mut rusqlite::ffi::sqlite3,
                    *mut *mut std::os::raw::c_char,
                    *const rusqlite::ffi::sqlite3_api_routines,
                ) -> std::os::raw::c_int,
            >(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
}

/// `meta` key every index job stamps at the start of its run — the second half
/// of the staleness witness, and not owned by this module.
///
/// See [`IndexState::job`] for why the row count alone is not enough.
const JOB_KEY: &str = "last_job_started_at";

/// What a corpus records about one shadow index.
///
/// `vectors` and `chunks` are the two counts that make [`usable`] possible:
/// `vectors` is how many rows the index holds, `chunks` is how many rows
/// `chunks` held when the index was last maintained. Every writer that touches
/// `chunks` and maintains an index moves both, in the same transaction as the
/// rows themselves, so the pair is either current or provably not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexState {
    /// The embedding model whose vectors the index holds. A query embedded with
    /// anything else must not touch it — cosine across two embedding spaces is
    /// a meaningless number, which is the same rule the scan applies per row.
    pub model: String,
    /// Vector width, in floats — of the SOURCE vectors, whatever the tier
    /// stores them as. Fixed at `CREATE VIRTUAL TABLE` time and not changeable
    /// afterwards, so a corpus that changes model width needs a rebuild rather
    /// than a migration.
    pub dimensions: usize,
    /// Rows in the index.
    pub vectors: usize,
    /// Rows in `chunks` as of the last maintained write. Half the staleness
    /// witness.
    pub chunks: usize,
    /// `meta.last_job_started_at` as of the last maintained write. The other
    /// half, and the half that is actually load-bearing.
    ///
    /// A row count alone is not a witness: an index job that re-processes a file
    /// deletes its chunks and writes the same number back, so a corpus can be
    /// rewritten underneath the index with the count landing exactly where it
    /// started. `last_job_started_at` is stamped by [`crate::pipeline::run_index`]
    /// before it writes anything, by every build that has ever had the key — so
    /// a job run by a build that does not maintain the index moves this and
    /// cannot move it back. Between them the two cover both shapes: a job that
    /// wrote (this moves) and a direct edit that bypassed the pipeline (the
    /// count moves).
    ///
    /// `None` is a corpus that has never run a job that stamped the key, which
    /// is a legitimate state and compares equal to itself.
    #[serde(default)]
    pub job: Option<String>,
    /// How the vectors are stored, and therefore whether an answer from this
    /// index is exact.
    ///
    /// `default` is [`Tier::Float`] on purpose: a marker written before the
    /// quantised tier existed describes a full-precision index, so an old
    /// corpus keeps parsing and keeps being treated as exactly what it is.
    #[serde(default)]
    pub tier: Tier,
    /// The vector subtracted from every vector before quantising, for a tier
    /// that thresholds rather than scales ([`Tier::needs_centre`]).
    ///
    /// It is the corpus' own mean, measured once by [`corpus_centre`] at build
    /// time, and it is what makes [`Tier::Bit`] work at all — see the [`Tier`]
    /// docs for the measurement that says so. A query is centred by the same
    /// vector the corpus was, so both sides live in the same space; an index
    /// whose centre is missing is refused rather than served uncentred, because
    /// uncentred is not "slightly worse", it is noise.
    ///
    /// NOT serialised with the rest of the marker ([`serde(skip)`]): it lives
    /// under [`Slot::centre_key`] because it is fixed for the life of an index
    /// while the marker is rewritten inside every per-file savepoint an index
    /// job commits. [`state`] loads it back.
    ///
    /// [`serde(skip)`]: https://serde.rs/field-attrs.html#skip
    #[serde(skip)]
    pub centre: Option<Vec<f32>>,
    /// Unix seconds the index was last built. Informational.
    pub built_at: f64,
    /// `llm-index` version that built it. Informational, and the thing to look
    /// at first if an index ever has to be distrusted wholesale.
    pub builder: String,
}

/// Whether a corpus' index may answer this query.
#[derive(Debug, Clone)]
pub enum Usable {
    /// Use it.
    Ready(IndexState),
    /// There is no index in this slot. The ordinary case: nothing to say about
    /// it.
    Absent,
    /// There is one, and it is not being used. The reason travels back to the
    /// caller because an index that silently stops being used is
    /// indistinguishable from one that was never built, and the two want
    /// completely different operator responses.
    Declined(String),
}

/// Whether `slot`'s table exists in this corpus.
///
/// A `sqlite_master` lookup, so it costs nothing and — importantly — does not
/// touch the virtual table. Naming a `vec0` table in a query is what makes
/// SQLite instantiate the module; this deliberately does not.
pub fn present(connection: &Connection, slot: Slot) -> Result<bool> {
    let found: Option<i64> = connection
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1",
            [slot.table()],
            |row| row.get(0),
        )
        .optional()?;
    Ok(found.is_some())
}

/// Read `slot`'s recorded [`IndexState`], if the corpus has one.
///
/// `None` covers three shapes that are all "no usable index": no `meta` table
/// (a corpus older than it), no key, and a key holding something that does not
/// parse. The last one is not an error on purpose — a corrupt marker must
/// degrade to the scan, not fail the query.
///
/// A tier that quantises against a centre gets it loaded from its own key here,
/// so a caller never holds a state that can encode wrongly: either the centre is
/// there or [`IndexState::centre`] is `None` and [`usable`] declines the index.
pub fn state(connection: &Connection, slot: Slot) -> Result<Option<IndexState>> {
    let raw: Option<String> = connection
        .query_row(
            "SELECT value FROM meta WHERE key=?1",
            [slot.meta_key()],
            |row| row.get(0),
        )
        .optional()
        .unwrap_or(None);
    let Some(mut state) = raw.and_then(|value| serde_json::from_str::<IndexState>(&value).ok())
    else {
        return Ok(None);
    };
    if state.tier.needs_centre() {
        state.centre = read_centre(connection, slot)?;
    }
    Ok(Some(state))
}

/// The centre recorded under `slot`'s [`Slot::centre_key`], if it is there and
/// parses. Same rule as the marker: unreadable degrades to `None`, which is
/// declined, rather than failing a query.
fn read_centre(connection: &Connection, slot: Slot) -> Result<Option<Vec<f32>>> {
    let raw: Option<String> = connection
        .query_row(
            "SELECT value FROM meta WHERE key=?1",
            [slot.centre_key()],
            |row| row.get(0),
        )
        .optional()
        .unwrap_or(None);
    Ok(raw.and_then(|value| serde_json::from_str(&value).ok()))
}

/// The corpus' current [`JOB_KEY`] stamp — see [`IndexState::job`].
///
/// `None` covers "no job has ever stamped it" and "there is no `meta` table",
/// both of which are stable states that compare equal to themselves.
pub fn job_stamp(connection: &Connection) -> Option<String> {
    connection
        .query_row("SELECT value FROM meta WHERE key=?1", [JOB_KEY], |row| {
            row.get(0)
        })
        .optional()
        .unwrap_or(None)
}

/// Record `state` under `slot`'s key. Rides the caller's transaction.
///
/// The centre is NOT written here — see [`IndexState::centre`]. This is the call
/// an index job makes once per file; putting 384 floats through it would rewrite
/// a few unchanged kilobytes on every one of them.
pub fn write_state(connection: &Connection, slot: Slot, state: &IndexState) -> Result<()> {
    connection.execute(
        "INSERT INTO meta(key,value) VALUES(?1,?2) \
         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        params![slot.meta_key(), serde_json::to_string(state)?],
    )?;
    Ok(())
}

/// Record the centre a build measured, or remove the key when the tier has none.
///
/// Called by [`build`] alone: the centre describes the index as a whole and is
/// replaced only when the index is.
fn write_centre(connection: &Connection, slot: Slot, centre: Option<&Vec<f32>>) -> Result<()> {
    match centre {
        Some(centre) => connection.execute(
            "INSERT INTO meta(key,value) VALUES(?1,?2) \
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![slot.centre_key(), serde_json::to_string(centre)?],
        )?,
        None => connection.execute("DELETE FROM meta WHERE key=?1", [slot.centre_key()])?,
    };
    Ok(())
}

/// Decide whether `slot`'s index can serve a query embedded with `model` at
/// `dimensions` floats.
///
/// Everything that can make an index unfit is checked here, and each one is
/// named rather than collapsed into a boolean:
///
/// * the table is not there — the ordinary, silent case;
/// * it is there but was never finished (the slot's `meta` key absent), i.e. a
///   build was killed part-way;
/// * it holds another model's vectors, or another width;
/// * it holds a tier that does not belong in this slot, which only a hand-edited
///   marker can produce and which would otherwise let a quantised index answer
///   an exact query;
/// * the corpus moved underneath it. This is the one that matters in practice:
///   every `llm-index` release before this feature writes `chunks` rows without
///   maintaining an index, so a corpus indexed by an older build after the index
///   was created holds vectors the index has never seen. Both witnesses are
///   checked — the job stamp catches a job that rewrote the corpus, the row
///   count catches an edit that bypassed the pipeline — and either one turns a
///   silently wrong answer into a fallback with a reason.
///
/// Both checks are single indexed row lookups plus `SELECT COUNT(*) FROM
/// chunks`, which SQLite answers from the narrowest index on the table
/// (`idx_chunks_file`) rather than the row data. Measured on the live corpus
/// they are a small fraction of the k-NN they guard — see
/// `docs/ARCHITECTURE.md` — which is what makes checking on every query, rather
/// than trusting a marker, affordable.
pub fn usable(
    connection: &Connection,
    slot: Slot,
    model: &str,
    dimensions: usize,
) -> Result<Usable> {
    if !present(connection, slot)? {
        return Ok(Usable::Absent);
    }
    let table = slot.table();
    let hint = slot.rebuild_hint();
    let Some(state) = state(connection, slot)? else {
        return Ok(Usable::Declined(format!(
            "{table} exists but no completed build is recorded; \
             a build was interrupted — rebuild it with {hint}"
        )));
    };
    if state.tier.slot() != slot {
        return Ok(Usable::Declined(format!(
            "{table} is recorded as holding {} vectors, which do not belong in this slot — \
             rebuild it with {hint}",
            state.tier.as_str()
        )));
    }
    if state.model != model || state.dimensions != dimensions {
        return Ok(Usable::Declined(format!(
            "{table} holds {} ({}d) vectors and this query is {model} ({dimensions}d)",
            state.model, state.dimensions
        )));
    }
    if state.vectors == 0 {
        return Ok(Usable::Declined(format!(
            "{table} is empty; scanning so the corpus can say why"
        )));
    }
    // A centring tier without its centre would quantise against zero, which on
    // text embeddings is not a degraded index but a random one — measured
    // recall@10 0.09 against 0.99. Refused, never served.
    if state.tier.needs_centre() && state.centre.as_ref().is_none_or(|c| c.len() != dimensions) {
        return Ok(Usable::Declined(format!(
            "{table} holds {} vectors but its corpus centre is missing or the wrong width; \
             it cannot be quantised against — rebuild with {hint}",
            state.tier.as_str()
        )));
    }
    let job = job_stamp(connection);
    if job != state.job {
        return Ok(Usable::Declined(format!(
            "{table} is stale: an index job has run against this corpus since it was \
             maintained (last_job_started_at {} -> {}) — the build that ran it does not maintain \
             the index. Rebuild with {hint}",
            state.job.as_deref().unwrap_or("none"),
            job.as_deref().unwrap_or("none"),
        )));
    }
    let chunks: i64 = connection.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
    if chunks as usize != state.chunks {
        return Ok(Usable::Declined(format!(
            "{table} is stale: it was maintained at {} chunks and the corpus now holds \
             {chunks} — something wrote this corpus without maintaining it. Rebuild with {hint}",
            state.chunks
        )));
    }
    Ok(Usable::Ready(state))
}

/// The `k` nearest `chunks.id`s to `query`, nearest first, out of `tier`'s slot.
///
/// `query` must already be encoded in that tier ([`Tier::encode`]), which for
/// the float tier is the raw little-endian `f32` blob `chunks.embedding` holds
/// and so is no conversion at all. Only ids come back: the caller re-scores them
/// against the stored BLOBs with the scan's own cosine, so the score in a
/// response means exactly what it meant before any of these indexes existed,
/// whichever path produced it.
pub fn knn(connection: &Connection, tier: Tier, query: &[u8], k: usize) -> Result<Vec<i64>> {
    let mut statement = connection.prepare(&format!(
        "SELECT rowid FROM {} WHERE embedding MATCH {} AND k = ?2 ORDER BY distance",
        tier.slot().table(),
        tier.bind("?1")
    ))?;
    let rows = statement.query_map(params![query, k.clamp(1, K_MAX) as i64], |row| row.get(0))?;
    Ok(rows.collect::<Result<Vec<i64>, _>>()?)
}

/// Ceiling `sqlite-vec` puts on `k` in a k-NN query. Above every candidate pool
/// `crate::embedding` asks for, so clamping here can never bite a real request —
/// it is here so a caller cannot turn a mis-parsed limit into a vtab error.
const K_MAX: usize = 4_096;

/// Create the (empty) virtual table for `dimensions`-wide vectors in `tier`.
pub fn create(connection: &Connection, tier: Tier, dimensions: usize) -> Result<()> {
    // Checked here rather than left to sqlite-vec, whose message ("Binary
    // quantization requires vectors with a length divisible by 8") describes its
    // own function rather than what an operator asked for. 384 divides, so this
    // is unreachable for the shipped model.
    if tier == Tier::Bit && !dimensions.is_multiple_of(8) {
        anyhow::bail!(
            "bit quantisation needs a vector width divisible by 8; this corpus stores {dimensions}d \
             vectors — use --tier int8"
        )
    }
    connection
        .execute_batch(&format!(
            "CREATE VIRTUAL TABLE {} USING vec0({})",
            tier.slot().table(),
            tier.column(dimensions)
        ))
        .with_context(|| format!("creating the {} vec0 shadow index", tier.as_str()))?;
    Ok(())
}

/// Drop `slot`'s index and forget it was ever built.
///
/// `DROP TABLE` on a `vec0` table removes its shadow tables too, so this leaves
/// a corpus that is once again indistinguishable from one that never had an
/// index in this slot. Nothing is lost: every vector in it was derived.
pub fn drop_index(connection: &Connection, slot: Slot) -> Result<()> {
    connection.execute_batch(&format!("DROP TABLE IF EXISTS {}", slot.table()))?;
    // Conditional because the live corpora predate `meta` entirely: dropping an
    // index from one of those must not fail on a table that was never there.
    let has_meta: Option<i64> = connection
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'",
            [],
            |row| row.get(0),
        )
        .optional()?;
    if has_meta.is_some() {
        connection.execute("DELETE FROM meta WHERE key=?1", [slot.meta_key()])?;
        connection.execute("DELETE FROM meta WHERE key=?1", [slot.centre_key()])?;
    }
    Ok(())
}

/// Mirror one chunk into `tier`'s slot. `encoded` is the vector in that tier.
pub fn insert(connection: &Connection, tier: Tier, id: i64, encoded: &[u8]) -> Result<()> {
    connection.execute(
        &format!(
            "INSERT INTO {}(rowid,embedding) VALUES(?1,{})",
            tier.slot().table(),
            tier.bind("?2")
        ),
        params![id, encoded],
    )?;
    Ok(())
}

/// Remove one chunk from `slot`. Silent when the id was never in it, which is
/// the normal case for a chunk the index excludes (another model, another
/// width) being deleted.
pub fn delete(connection: &Connection, slot: Slot, id: i64) -> Result<()> {
    connection.execute(
        &format!("DELETE FROM {} WHERE rowid=?1", slot.table()),
        [id],
    )?;
    Ok(())
}

impl IndexState {
    /// Encode one stored `f32` BLOB the way THIS index stores its copies.
    ///
    /// The call every writer and every query path makes, so that the tier, the
    /// width and the centre an index was built with travel together and no
    /// caller can pair them wrongly.
    pub fn encode<'a>(&self, blob: &'a [u8]) -> Option<Cow<'a, [u8]>> {
        self.tier
            .encode(blob, self.dimensions, self.centre.as_deref())
    }
}

/// One shadow index a writer found in a corpus and is maintaining.
#[derive(Debug, Clone)]
pub struct Maintained {
    pub slot: Slot,
    pub state: IndexState,
}

/// Every shadow index in `connection` that a writer can maintain.
///
/// Two shapes are left out, and for the same reason: a job will not maintain a
/// table it cannot vouch for, and the build an operator re-runs replaces it
/// wholesale. Those are a slot whose table exists but whose build never finished
/// (no state to load), and a centring tier whose centre is gone — half-
/// maintaining that one would grow its `chunks` count while its `vectors` count
/// stood still, turning an index a reader already declines into one whose own
/// numbers are wrong as well.
pub fn maintained(connection: &Connection) -> Result<Vec<Maintained>> {
    let mut found = Vec::new();
    for slot in Slot::ALL {
        if !present(connection, slot)? {
            continue;
        }
        if let Some(state) = state(connection, slot)? {
            if state.tier.needs_centre() && state.centre.is_none() {
                continue;
            }
            found.push(Maintained { slot, state });
        }
    }
    Ok(found)
}

/// The vector width `model`'s rows are stored at, taken from the corpus itself.
///
/// Read from the first such row rather than from an embedding model: an index
/// mirrors what is STORED, so a corpus whose vectors are some other width must
/// build an index of that width or index nothing at all. It also keeps
/// `vector-index` from having to load a 448 MB ONNX model to learn the number
/// 384. Rows that then disagree with this width are counted as skipped by
/// [`build`], which is where a mixed corpus becomes visible.
///
/// `None` is a corpus with no rows for this model — nothing to index.
pub fn corpus_dimensions(connection: &Connection, model: &str) -> Result<Option<usize>> {
    let width: Option<i64> = connection
        .query_row(
            "SELECT dimensions FROM chunks WHERE model=?1 ORDER BY id LIMIT 1",
            [model],
            |row| row.get(0),
        )
        .optional()?;
    Ok(width.filter(|value| *value > 0).map(|value| value as usize))
}

/// What one [`build`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct BuildReport {
    /// Vectors written into the index.
    pub vectors: usize,
    /// Rows in `chunks` the index does not cover: another model, or a BLOB that
    /// is not `dimensions` floats wide. Counted, never indexed — the same rule
    /// the scan applies.
    pub skipped: usize,
    /// The state recorded in `meta` at the end.
    pub state: IndexState,
}

/// (Re)build one tier's index for an existing corpus, from the stored BLOBs
/// alone.
///
/// This is the path the 2.68 M-vector live corpora take to gain an index: it
/// reads `chunks.embedding`, encodes each vector into `tier` and writes it into
/// the virtual table, and touches no model, no document and no file on disk.
/// Nothing is re-embedded, and no table the corpus' own data lives in is
/// written — `files`, `fts`, `chunks` and `vision` are read-only to a build, so
/// the worst an interrupted one costs is the time it had spent.
///
/// Any existing index IN THE SAME SLOT is dropped first, so this is both "build"
/// and "rebuild", and so building `int8` over a `bit` corpus replaces it rather
/// than leaving two quantised indexes to disagree. The other slot is untouched:
/// a corpus can carry the exact tier and one quantised tier at once, which is
/// the arrangement `mode=semantic` and `mode=semantic_fast` are both fast on.
/// There is no incremental repair mode, because a stale index is stale by an
/// unknown amount and rebuilding is the only honest repair.
///
/// `progress` is called with `(written, total)` at each commit boundary, so a
/// CLI can report a build that runs for minutes.
pub fn build(
    connection: &mut Connection,
    tier: Tier,
    model: &str,
    dimensions: usize,
    mut progress: impl FnMut(usize, usize),
) -> Result<BuildReport> {
    let total: i64 = connection.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
    let total = total.max(0) as usize;
    // The corpus' self-description table, verbatim from `store::SCHEMA`. The
    // live corpora predate it and have never been opened by a build that adds
    // it, so a rebuild has to be able to create it — which is the same
    // `IF NOT EXISTS` route every other table in this format takes, and a no-op
    // for a corpus that already has one.
    connection
        .execute_batch("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        .context("creating the corpus meta table")?;
    let slot = tier.slot();
    // Measured BEFORE the table is dropped, so a failure here leaves the corpus
    // with the index it already had rather than with none.
    let centre = if tier.needs_centre() {
        let centre = corpus_centre(connection, model, dimensions)?.with_context(|| {
            format!("measuring the corpus centre for a {} index", tier.as_str())
        })?;
        Some(centre)
    } else {
        None
    };
    drop_index(connection, slot)?;
    create(connection, tier, dimensions)?;
    let table = slot.table();
    let mut vectors = 0usize;
    let mut skipped = 0usize;
    // Keyset pagination over `chunks.id`, so each batch can COMMIT: a cursor
    // held open across a commit is not a cursor any more, and collecting the ids
    // up front would materialise 2.7 M of them (or, worse, their BLOBs) to avoid
    // a b-tree seek per 50,000 rows.
    let mut from = i64::MIN;
    loop {
        let transaction = connection.unchecked_transaction()?;
        let mut read = transaction
            .prepare("SELECT id,model,embedding FROM chunks WHERE id>=?1 ORDER BY id LIMIT ?2")?;
        let mut insert = transaction.prepare(&format!(
            "INSERT INTO {table}(rowid,embedding) VALUES(?1,{})",
            tier.bind("?2")
        ))?;
        let mut rows = read.query(params![from, BUILD_BATCH as i64])?;
        let mut seen = 0usize;
        let mut last = from;
        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let row_model = row.get_ref(1)?.as_str()?;
            let embedding = row.get_ref(2)?.as_blob()?;
            match (row_model == model)
                .then(|| tier.encode(embedding, dimensions, centre.as_deref()))
                .flatten()
            {
                Some(encoded) => {
                    insert.execute(params![id, encoded.as_ref()])?;
                    vectors += 1;
                }
                None => skipped += 1,
            }
            seen += 1;
            last = id;
        }
        drop(rows);
        drop(read);
        drop(insert);
        transaction.commit()?;
        // A short page is the last page. `checked_add` also ends the loop on the
        // (unreachable in practice) corpus whose highest chunk id is i64::MAX,
        // rather than wrapping round and re-reading the whole table forever.
        match (seen == BUILD_BATCH).then(|| last.checked_add(1)).flatten() {
            Some(next) => from = next,
            None => break,
        }
        progress(vectors, total);
    }
    let state = IndexState {
        model: model.to_string(),
        dimensions,
        vectors,
        chunks: total,
        job: job_stamp(connection),
        tier,
        centre,
        built_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs_f64())
            .unwrap_or_default(),
        builder: crate::VERSION.to_string(),
    };
    // Last, and in this order: the centre is what the marker's tier promises is
    // there, and the marker is what makes an index usable at all, so neither may
    // become visible before the thing it vouches for. Everything before this
    // point is a part-filled table nothing will read.
    write_centre(connection, slot, state.centre.as_ref())?;
    write_state(connection, slot, &state)?;
    progress(vectors, total);
    Ok(BuildReport {
        vectors,
        skipped,
        state,
    })
}

/// The `/corpus/status`-shaped summary of one slot, for a CLI or a route that
/// wants to report what is there without deciding whether to use it.
pub fn describe(connection: &Connection, slot: Slot) -> Result<serde_json::Value> {
    if !present(connection, slot)? {
        return Ok(json!({"present": false}));
    }
    match state(connection, slot)? {
        Some(state) => Ok(json!({"present": true, "state": state})),
        None => Ok(json!({"present": true, "state": serde_json::Value::Null,
                          "note": "no completed build is recorded for this table"})),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::vector_to_bytes;

    /// A corpus holding only what the index reads and writes.
    fn corpus() -> Connection {
        register();
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                 CREATE TABLE chunks(
                   id INTEGER PRIMARY KEY,
                   file_id INTEGER NOT NULL,
                   chunk_index INTEGER NOT NULL,
                   content TEXT NOT NULL,
                   embedding BLOB NOT NULL,
                   dimensions INTEGER NOT NULL,
                   model TEXT NOT NULL);
                 CREATE INDEX idx_chunks_file ON chunks(file_id);",
            )
            .unwrap();
        connection
    }

    fn add_chunk(connection: &Connection, id: i64, model: &str, vector: &[f32]) {
        connection
            .execute(
                "INSERT INTO chunks(id,file_id,chunk_index,content,embedding,dimensions,model) \
                 VALUES(?1,?1,0,'text',?2,?3,?4)",
                params![id, vector_to_bytes(vector), vector.len() as i64, model],
            )
            .unwrap();
    }

    const MODEL: &str = "test-model";

    /// Decode an int8 blob back to the signed codes it holds.
    fn codes(encoded: &[u8]) -> Vec<i8> {
        encoded.iter().map(|byte| *byte as i8).collect()
    }

    #[test]
    fn int8_quantisation_puts_the_largest_component_on_the_rail() {
        // The property the whole quantiser rests on: a per-vector divisor, so
        // every vector uses the full code range and nothing ever clips.
        let encoded = quantise_int8(&vector_to_bytes(&[0.05, -0.10, 0.025, 0.0]));
        assert_eq!(codes(&encoded), [64, -127, 32, 0]);
        // A vector a thousand times shorter quantises IDENTICALLY, because
        // cosine cannot see a positive scale and neither can this.
        let tiny = quantise_int8(&vector_to_bytes(&[0.00005, -0.0001, 0.000025, 0.0]));
        assert_eq!(codes(&tiny), codes(&encoded));
        // And so does its negation, mirrored — which is why the range stops at
        // -127 rather than reaching for the extra code at -128.
        let opposite = quantise_int8(&vector_to_bytes(&[-0.05, 0.10, -0.025, 0.0]));
        assert_eq!(codes(&opposite), [-64, 127, -32, 0]);
    }

    #[test]
    fn int8_quantisation_bounds_every_code_and_survives_degenerate_vectors() {
        for vector in [
            vec![1.0f32, -1.0, 0.5, -0.5],
            vec![1e30, -1e30, 1.0, 0.0],
            vec![1e-30, -1e-30, 0.0, 0.0],
        ] {
            let encoded = quantise_int8(&vector_to_bytes(&vector));
            assert_eq!(encoded.len(), vector.len(), "one byte per dimension");
            assert!(
                codes(&encoded)
                    .iter()
                    .all(|code| (-127..=127).contains(code)),
                "{vector:?} -> {:?}",
                codes(&encoded)
            );
        }
        // No finite non-zero component: all zeros, which sqlite-vec scores NaN
        // and the float re-score floors at -1.0. It cannot be wrong, only
        // wasted.
        assert_eq!(codes(&quantise_int8(&vector_to_bytes(&[0.0; 4]))), [0; 4]);
        assert_eq!(
            codes(&quantise_int8(&vector_to_bytes(&[f32::NAN; 4]))),
            [0; 4]
        );
        // One broken component does not poison the rest: it encodes as zero and
        // takes no part in choosing the divisor.
        assert_eq!(
            codes(&quantise_int8(&vector_to_bytes(&[
                f32::INFINITY,
                0.2,
                -0.1,
                0.0
            ]))),
            [0, 127, -64, 0]
        );
    }

    #[test]
    fn bit_quantisation_thresholds_at_the_centre_and_keeps_nothing_else() {
        let vector = (0..16)
            .map(|index| if index % 3 == 0 { 0.001 } else { -2.0 })
            .collect::<Vec<f32>>();
        let origin = vec![0.0f32; 16];
        let encoded = quantise_bit(&vector_to_bytes(&vector), &origin);
        assert_eq!(encoded.len(), 2, "one bit per dimension");
        for (index, value) in vector.iter().enumerate() {
            let bit = encoded[index / 8] >> (index % 8) & 1;
            assert_eq!(bit == 1, *value > 0.0, "dimension {index}");
        }
        // Magnitude is gone: a vector 2,000 times longer in every component
        // encodes to the same bits about the origin.
        let longer = vector
            .iter()
            .map(|value| value * 2_000.0)
            .collect::<Vec<_>>();
        assert_eq!(quantise_bit(&vector_to_bytes(&longer), &origin), encoded);
        // Zero, negative zero, NaN and -inf all land on the unset side, like
        // every other component at or below the centre.
        assert_eq!(
            quantise_bit(
                &vector_to_bytes(&[
                    0.0,
                    f32::NAN,
                    -0.0,
                    f32::NAN,
                    0.0,
                    -1.0,
                    f32::NEG_INFINITY,
                    0.0
                ]),
                &[0.0f32; 8]
            ),
            [0u8]
        );
    }

    #[test]
    fn the_centre_is_what_gives_a_bit_its_information() {
        // The failure the centre exists to prevent, in miniature: eight vectors
        // that all sit on the same side of the origin in every dimension, so
        // uncentred they collapse to ONE code and the index cannot tell them
        // apart at all. Centred, they spread across the code space.
        //
        // This is the shape of a real text-embedding corpus, where a large
        // shared mean direction puts most dimensions on one side for nearly
        // every vector — measured on the live corpus as recall@10 0.09
        // uncentred against 0.99 centred (`docs/ARCHITECTURE.md`).
        let corpus = (0..8)
            .map(|index| {
                (0..8)
                    .map(|dimension| 5.0 + (((index + dimension) % 3) as f32) * 0.1)
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>();
        let origin = vec![0.0f32; 8];
        let uncentred = corpus
            .iter()
            .map(|vector| quantise_bit(&vector_to_bytes(vector), &origin))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            uncentred.len(),
            1,
            "every bit is a constant: no information"
        );

        let mut centre = vec![0.0f32; 8];
        for vector in &corpus {
            for (sum, value) in centre.iter_mut().zip(vector) {
                *sum += value / corpus.len() as f32;
            }
        }
        let centred = corpus
            .iter()
            .map(|vector| quantise_bit(&vector_to_bytes(vector), &centre))
            .collect::<std::collections::BTreeSet<_>>();
        assert!(centred.len() > 1, "the same vectors now carry a code each");
    }

    #[test]
    fn a_corpus_centre_is_the_mean_of_the_vectors_it_holds() {
        let connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, -1.0]);
        add_chunk(&connection, 2, MODEL, &[3.0, 2.0, 1.0]);
        add_chunk(&connection, 3, "other-model", &[100.0, 100.0, 100.0]);

        let centre = corpus_centre(&connection, MODEL, 3).unwrap().unwrap();
        assert_eq!(
            centre,
            vec![2.0, 1.0, 0.0],
            "the foreign model is not in it"
        );
        assert!(
            corpus_centre(&connection, "absent-model", 3)
                .unwrap()
                .is_none(),
            "nothing to average is not a zero centre, it is no centre"
        );
    }

    #[test]
    fn a_bit_index_without_its_centre_is_declined_rather_than_served() {
        // The centre lives in its own `meta` key, so it can be lost on its own —
        // by a hand edit, or by a build that wrote the marker and nothing else.
        // Serving that index would quantise against zero, which on a real corpus
        // is a random index rather than a worse one.
        let mut connection = corpus();
        for id in 1..=4i64 {
            add_chunk(
                &connection,
                id,
                MODEL,
                &[id as f32, 0.5, -0.5, 1.0, 0.0, 0.25, -1.0, 2.0],
            );
        }
        build(&mut connection, Tier::Bit, MODEL, 8, |_, _| {}).unwrap();
        assert!(matches!(
            usable(&connection, Slot::Quantised, MODEL, 8).unwrap(),
            Usable::Ready(_)
        ));

        connection
            .execute(
                "DELETE FROM meta WHERE key=?1",
                [Slot::Quantised.centre_key()],
            )
            .unwrap();
        let Usable::Declined(reason) = usable(&connection, Slot::Quantised, MODEL, 8).unwrap()
        else {
            panic!("an uncentred bit index must not be served")
        };
        assert!(reason.contains("centre is missing"), "{reason}");
        // And a rebuild puts it back.
        build(&mut connection, Tier::Bit, MODEL, 8, |_, _| {}).unwrap();
        assert!(matches!(
            usable(&connection, Slot::Quantised, MODEL, 8).unwrap(),
            Usable::Ready(_)
        ));
    }

    #[test]
    fn encoding_refuses_a_vector_it_cannot_represent() {
        let blob = vector_to_bytes(&[1.0, 0.0, 0.0]);
        for tier in [Tier::Float, Tier::Int8] {
            assert!(tier.encode(&blob, 3, None).is_some());
            assert!(tier.encode(&blob, 4, None).is_none(), "{tier:?}");
            assert!(
                tier.encode(&blob[..7], 3, None).is_none(),
                "a torn blob, {tier:?}"
            );
        }
        // Bit adds two: 3 dimensions do not pack into whole bytes, and no tier
        // that thresholds may encode without the centre it thresholds against.
        // `create` and `build` refuse both outright, so this is belt and braces
        // — and each is a `None` rather than a panic, which is what makes
        // `encode` safe to call with whatever a corpus happens to record.
        let eight = vector_to_bytes(&[0.5; 8]);
        assert!(Tier::Bit.encode(&blob, 3, Some(&[0.0; 3])).is_none());
        assert!(Tier::Bit.encode(&eight, 8, None).is_none());
        assert!(
            Tier::Bit.encode(&eight, 8, Some(&[0.0; 3])).is_none(),
            "a centre of another width is no centre"
        );
        assert!(Tier::Bit
            .encode(&eight, 8, Some(&[0.0; 8]))
            .is_some_and(|encoded| encoded.len() == 1));
        // And the float tier borrows rather than copying — the property that
        // keeps a 2.68 M-vector build from allocating 2.68 M times.
        assert!(matches!(
            Tier::Float.encode(&blob, 3, None).unwrap(),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn a_corpus_without_the_table_reports_absent_and_stays_untouched() {
        let connection = corpus();
        for slot in Slot::ALL {
            assert!(!present(&connection, slot).unwrap());
            assert!(matches!(
                usable(&connection, slot, MODEL, 3).unwrap(),
                Usable::Absent
            ));
            assert_eq!(
                describe(&connection, slot).unwrap(),
                json!({"present": false})
            );
        }
        assert!(maintained(&connection).unwrap().is_empty());
    }

    #[test]
    fn a_build_indexes_the_matching_vectors_and_counts_the_rest() {
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        add_chunk(&connection, 2, MODEL, &[0.0, 1.0, 0.0]);
        add_chunk(&connection, 3, "other-model", &[1.0, 0.0, 0.0]);
        add_chunk(&connection, 4, MODEL, &[1.0, 0.0, 0.0, 0.0]); // another width

        let report = build(&mut connection, Tier::Float, MODEL, 3, |_, _| {}).unwrap();
        assert_eq!(report.vectors, 2);
        assert_eq!(report.skipped, 2, "a foreign model and a foreign width");
        assert_eq!(
            report.state.chunks, 4,
            "the staleness witness counts them all"
        );
        assert_eq!(report.state.tier, Tier::Float);

        let nearest = knn(
            &connection,
            Tier::Float,
            &vector_to_bytes(&[1.0, 0.0, 0.0]),
            2,
        )
        .unwrap();
        assert_eq!(nearest, vec![1, 2], "exact match first");
        assert!(matches!(
            usable(&connection, Slot::Exact, MODEL, 3).unwrap(),
            Usable::Ready(_)
        ));
    }

    #[test]
    fn a_quantised_build_covers_the_same_rows_in_its_own_slot() {
        // Same filter, same counts, a different table — and the exact slot is
        // left completely alone, which is what lets a corpus be fast on both
        // `semantic` and `semantic_fast` at once.
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[0.05, 0.01, 0.0]);
        add_chunk(&connection, 2, MODEL, &[0.0, 0.05, 0.0]);
        add_chunk(&connection, 3, "other-model", &[1.0, 0.0, 0.0]);
        build(&mut connection, Tier::Float, MODEL, 3, |_, _| {}).unwrap();

        let report = build(&mut connection, Tier::Int8, MODEL, 3, |_, _| {}).unwrap();
        assert_eq!(report.vectors, 2);
        assert_eq!(report.skipped, 1);
        assert_eq!(report.state.tier, Tier::Int8);
        assert!(present(&connection, Slot::Exact).unwrap(), "untouched");
        assert!(present(&connection, Slot::Quantised).unwrap());

        // The quantised k-NN nominates the same nearest row as the exact one.
        let query = vector_to_bytes(&[0.05, 0.01, 0.0]);
        let quantised = Tier::Int8.encode(&query, 3, None).unwrap();
        assert_eq!(
            knn(&connection, Tier::Int8, &quantised, 1).unwrap(),
            knn(&connection, Tier::Float, &query, 1).unwrap()
        );
        assert_eq!(maintained(&connection).unwrap().len(), 2);
    }

    #[test]
    fn a_second_quantisation_replaces_the_first_rather_than_joining_it() {
        // One quantised index per corpus, by construction: there is then never a
        // question of which one answered, and never two to keep in step.
        let mut connection = corpus();
        add_chunk(
            &connection,
            1,
            MODEL,
            &[0.05, 0.01, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0],
        );
        add_chunk(
            &connection,
            2,
            MODEL,
            &[0.01, 0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        build(&mut connection, Tier::Int8, MODEL, 8, |_, _| {}).unwrap();
        assert!(
            read_centre(&connection, Slot::Quantised).unwrap().is_none(),
            "int8 does not threshold, so it records no centre"
        );
        let report = build(&mut connection, Tier::Bit, MODEL, 8, |_, _| {}).unwrap();
        assert!(
            read_centre(&connection, Slot::Quantised).unwrap().is_some(),
            "and bit does"
        );

        assert_eq!(report.state.tier, Tier::Bit);
        assert_eq!(
            state(&connection, Slot::Quantised).unwrap().unwrap().tier,
            Tier::Bit
        );
        assert_eq!(maintained(&connection).unwrap().len(), 1);
        let bytes: i64 = connection
            .query_row(
                &format!(
                    "SELECT LENGTH(embedding) FROM {} LIMIT 1",
                    Slot::Quantised.table()
                ),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(bytes, 1, "8 dimensions, one bit each");
        // And going back to int8 takes the centre with it, so nothing is left
        // describing an index that is no longer there.
        build(&mut connection, Tier::Int8, MODEL, 8, |_, _| {}).unwrap();
        assert!(read_centre(&connection, Slot::Quantised).unwrap().is_none());
    }

    #[test]
    fn a_bit_index_needs_a_width_it_can_pack() {
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        let refused = build(&mut connection, Tier::Bit, MODEL, 3, |_, _| {}).unwrap_err();
        assert!(refused.to_string().contains("--tier int8"), "{refused}");
    }

    #[test]
    fn a_rebuild_replaces_the_index_rather_than_doubling_it() {
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        build(&mut connection, Tier::Float, MODEL, 3, |_, _| {}).unwrap();
        add_chunk(&connection, 2, MODEL, &[0.0, 1.0, 0.0]);
        let report = build(&mut connection, Tier::Float, MODEL, 3, |_, _| {}).unwrap();
        assert_eq!(report.vectors, 2);
        let rows: i64 = connection
            .query_row(
                &format!("SELECT COUNT(*) FROM {}", Slot::Exact.table()),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(rows, 2, "the old table is dropped, not appended to");
    }

    #[test]
    fn insert_and_delete_keep_the_index_in_step_with_chunks() {
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        build(&mut connection, Tier::Float, MODEL, 3, |_, _| {}).unwrap();

        add_chunk(&connection, 2, MODEL, &[0.9, 0.1, 0.0]);
        insert(
            &connection,
            Tier::Float,
            2,
            &vector_to_bytes(&[0.9, 0.1, 0.0]),
        )
        .unwrap();
        assert_eq!(
            knn(
                &connection,
                Tier::Float,
                &vector_to_bytes(&[0.9, 0.1, 0.0]),
                1
            )
            .unwrap(),
            vec![2]
        );

        delete(&connection, Slot::Exact, 2).unwrap();
        assert_eq!(
            knn(
                &connection,
                Tier::Float,
                &vector_to_bytes(&[0.9, 0.1, 0.0]),
                5
            )
            .unwrap(),
            vec![1],
            "a deleted chunk stops being a candidate"
        );
        // Deleting an id the index never held is a no-op, not an error: chunks
        // the index excludes are deleted through the same call site.
        delete(&connection, Slot::Exact, 99).unwrap();
    }

    #[test]
    fn an_index_is_declined_when_chunks_moved_underneath_it() {
        // The old-binary hazard, reproduced exactly: a build that does not
        // maintain the index writes a chunks row, and the index no longer covers
        // the corpus. Serving k-NN from it would silently miss that row. Both
        // slots decline it, and each names its own repair.
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        build(&mut connection, Tier::Float, MODEL, 3, |_, _| {}).unwrap();
        build(&mut connection, Tier::Int8, MODEL, 3, |_, _| {}).unwrap();

        add_chunk(&connection, 2, MODEL, &[0.0, 1.0, 0.0]); // no index maintenance
        for (slot, hint) in [(Slot::Exact, "--rebuild"), (Slot::Quantised, "--tier")] {
            let Usable::Declined(reason) = usable(&connection, slot, MODEL, 3).unwrap() else {
                panic!("a corpus written behind {slot:?} must not be served from it")
            };
            assert!(reason.contains("stale"), "{reason}");
            assert!(reason.contains(hint), "{reason}");
            assert!(reason.contains(slot.table()), "{reason}");
        }
    }

    /// Stamp the key every index job writes at the start of its run.
    fn run_a_job(connection: &Connection, at: &str) {
        connection
            .execute(
                "INSERT INTO meta(key,value) VALUES('last_job_started_at',?1) \
                 ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [at],
            )
            .unwrap();
    }

    #[test]
    fn an_index_is_declined_after_a_job_that_left_the_row_count_where_it_found_it() {
        // The hazard a row count cannot see. An index job re-processing a file
        // deletes its chunks and writes the same number back: the corpus is
        // rewritten, the count is unchanged, and the vectors in the index now
        // belong to rows that no longer exist. `last_job_started_at` is what
        // catches it — every build that has ever had the key stamps it before
        // writing, whether or not it maintains the index.
        let mut connection = corpus();
        run_a_job(&connection, "1700");
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        build(&mut connection, Tier::Float, MODEL, 3, |_, _| {}).unwrap();
        assert!(matches!(
            usable(&connection, Slot::Exact, MODEL, 3).unwrap(),
            Usable::Ready(_)
        ));

        run_a_job(&connection, "1800");
        connection
            .execute("DELETE FROM chunks WHERE id=1", [])
            .unwrap();
        add_chunk(&connection, 1, MODEL, &[0.0, 1.0, 0.0]);
        let live: i64 = connection
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .unwrap();
        assert_eq!(
            live as usize,
            state(&connection, Slot::Exact).unwrap().unwrap().chunks
        );

        let Usable::Declined(reason) = usable(&connection, Slot::Exact, MODEL, 3).unwrap() else {
            panic!("a job that rewrote the corpus must invalidate the index")
        };
        assert!(
            reason.contains("last_job_started_at 1700 -> 1800"),
            "{reason}"
        );
    }

    #[test]
    fn a_rebuild_after_such_a_job_makes_the_index_usable_again() {
        // The repair, and the reason declining is not a dead end.
        let mut connection = corpus();
        run_a_job(&connection, "1700");
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        build(&mut connection, Tier::Float, MODEL, 3, |_, _| {}).unwrap();
        run_a_job(&connection, "1800");
        add_chunk(&connection, 2, MODEL, &[0.0, 1.0, 0.0]);
        assert!(matches!(
            usable(&connection, Slot::Exact, MODEL, 3).unwrap(),
            Usable::Declined(_)
        ));

        let report = build(&mut connection, Tier::Float, MODEL, 3, |_, _| {}).unwrap();
        assert_eq!(report.state.job.as_deref(), Some("1800"));
        assert!(matches!(
            usable(&connection, Slot::Exact, MODEL, 3).unwrap(),
            Usable::Ready(_)
        ));
    }

    #[test]
    fn an_index_for_another_model_or_width_is_declined() {
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        build(&mut connection, Tier::Float, MODEL, 3, |_, _| {}).unwrap();

        let Usable::Declined(reason) =
            usable(&connection, Slot::Exact, "another-model", 3).unwrap()
        else {
            panic!("cosine across two embedding spaces is meaningless")
        };
        assert!(reason.contains("another-model"), "{reason}");
        let Usable::Declined(reason) = usable(&connection, Slot::Exact, MODEL, 384).unwrap() else {
            panic!("a 3d index cannot answer a 384d query")
        };
        assert!(reason.contains("384d"), "{reason}");
    }

    #[test]
    fn a_quantised_index_hand_moved_into_the_exact_slot_is_declined() {
        // Only a hand-edited marker produces this, and it is exactly the shape
        // that would let an approximate index answer the exact query path.
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        build(&mut connection, Tier::Float, MODEL, 3, |_, _| {}).unwrap();
        let mut state = state(&connection, Slot::Exact).unwrap().unwrap();
        state.tier = Tier::Int8;
        write_state(&connection, Slot::Exact, &state).unwrap();

        let Usable::Declined(reason) = usable(&connection, Slot::Exact, MODEL, 3).unwrap() else {
            panic!("an int8 index must never serve the exact path")
        };
        assert!(reason.contains("int8"), "{reason}");
    }

    #[test]
    fn an_interrupted_build_leaves_a_table_nothing_will_use() {
        // A build writes its marker last, so a table with no marker is exactly
        // what a killed build leaves. It must degrade to the scan, and say so —
        // and a writer must not adopt it either.
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        build(&mut connection, Tier::Int8, MODEL, 3, |_, _| {}).unwrap();
        connection
            .execute(
                "DELETE FROM meta WHERE key=?1",
                [Slot::Quantised.meta_key()],
            )
            .unwrap();

        let Usable::Declined(reason) = usable(&connection, Slot::Quantised, MODEL, 3).unwrap()
        else {
            panic!("an unfinished build must not be trusted")
        };
        assert!(reason.contains("interrupted"), "{reason}");
        assert_eq!(
            describe(&connection, Slot::Quantised).unwrap()["state"],
            serde_json::Value::Null
        );
        assert!(maintained(&connection).unwrap().is_empty());
    }

    #[test]
    fn dropping_leaves_a_corpus_indistinguishable_from_one_that_never_had_an_index() {
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        for tier in [Tier::Float, Tier::Int8] {
            build(&mut connection, tier, MODEL, 3, |_, _| {}).unwrap();
        }
        for slot in Slot::ALL {
            drop_index(&connection, slot).unwrap();
            assert!(!present(&connection, slot).unwrap());
            assert!(state(&connection, slot).unwrap().is_none());
        }
        // Including the shadow tables sqlite-vec created beside them.
        let leftovers: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'chunks_vec%'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(leftovers, 0);
    }

    #[test]
    fn dropping_one_slot_leaves_the_other_alone() {
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        build(&mut connection, Tier::Float, MODEL, 3, |_, _| {}).unwrap();
        build(&mut connection, Tier::Int8, MODEL, 3, |_, _| {}).unwrap();

        drop_index(&connection, Slot::Quantised).unwrap();
        assert!(!present(&connection, Slot::Quantised).unwrap());
        assert!(matches!(
            usable(&connection, Slot::Exact, MODEL, 3).unwrap(),
            Usable::Ready(_)
        ));
    }

    #[test]
    fn a_corrupt_marker_degrades_to_the_scan_instead_of_failing_the_query() {
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        build(&mut connection, Tier::Float, MODEL, 3, |_, _| {}).unwrap();
        connection
            .execute(
                "UPDATE meta SET value='{not json' WHERE key=?1",
                [Slot::Exact.meta_key()],
            )
            .unwrap();

        assert!(state(&connection, Slot::Exact).unwrap().is_none());
        assert!(matches!(
            usable(&connection, Slot::Exact, MODEL, 3).unwrap(),
            Usable::Declined(_)
        ));
    }

    #[test]
    fn a_marker_written_before_the_quantised_tier_existed_still_parses_as_float() {
        // The forward-compatibility rule this corpus format runs on: a `meta`
        // key gains fields, never a version number. An index built by the
        // release before this one has no `tier`, and it is a float index.
        let connection = corpus();
        connection
            .execute(
                "INSERT INTO meta(key,value) VALUES(?1,?2)",
                params![
                    Slot::Exact.meta_key(),
                    r#"{"model":"test-model","dimensions":3,"vectors":7,"chunks":7,
                        "job":"1700","built_at":1.0,"builder":"0.4.0"}"#
                ],
            )
            .unwrap();
        let state = state(&connection, Slot::Exact).unwrap().unwrap();
        assert_eq!(state.tier, Tier::Float);
        assert_eq!(state.vectors, 7);
    }
}
