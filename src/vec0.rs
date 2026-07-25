//! The OPTIONAL `vec0` shadow index over `chunks.embedding`.
//!
//! Semantic search without it is an exhaustive cosine scan: every stored vector
//! is read and scored, which is exact but costs a full pass over the `chunks`
//! table — 15.6 GB of it on the 2.68 M-vector live corpus. This module adds a
//! second, purely derived copy of the vectors in a `sqlite-vec` [`vec0`] virtual
//! table, laid out as contiguous chunk blobs rather than one row per vector, so a
//! query reads 4.12 GB of vectors and nothing else.
//!
//! That is a layout win, not an algorithmic one: `vec0` 0.1.9 has no ANN
//! structure and still visits every vector. Measured back to back on both live
//! corpora it is roughly an order of magnitude faster than the scan — best warm
//! passes 1.32 s at 869 k vectors and 3.9 s at 2.68 M — which is seconds rather
//! than milliseconds. See `docs/ARCHITECTURE.md`, which keeps the numbers and
//! says plainly what this does not buy.
//!
//! Three properties are load-bearing, and every function here exists to keep one
//! of them:
//!
//! * **Optional.** The index is a table that either exists in a corpus or does
//!   not. A corpus without it is byte-identical to what this build produced
//!   before the index existed, answers semantic queries through the scan, and
//!   pays nothing at index time. Nothing creates the table implicitly — only
//!   `llm-index vector-index` does.
//! * **Derived.** Every vector in it is a copy of a `chunks.embedding` BLOB, so
//!   it can be dropped and rebuilt from the corpus at any time without
//!   re-embedding a single document ([`build`]). Losing it loses no data.
//! * **Never trusted blindly.** An index is used only when the corpus can prove
//!   it is still complete ([`usable`]). A build that does not maintain it —
//!   every release before this one — can write `chunks` rows underneath it, and
//!   serving a k-NN answer from an index that is missing those rows would be
//!   silently wrong. The scan is always there to fall back to.
//!
//! [`vec0`]: https://github.com/asg017/sqlite-vec

use std::sync::Once;

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// The virtual table holding the shadow index.
///
/// `sqlite-vec` puts its own storage in `chunks_vec_*` shadow tables beside it
/// (`_info`, `_chunks`, `_rowids`, `_vector_chunks00`). All of them are ordinary
/// tables in the same file, so a corpus with an index is still one portable
/// `.sqlite` file with no sidecars.
pub const SHADOW_TABLE: &str = "chunks_vec";

/// `meta` key describing the index — see [`IndexState`].
///
/// The corpus' schema-version idiom is exactly this: `meta` keys plus
/// `IF NOT EXISTS` tables (`store::SCHEMA`) and re-applied column lists
/// (`store::ADDED_FILE_COLUMNS`). There is no monotonic schema number to bump,
/// and inventing one here would leave every existing corpus claiming a version
/// it was never stamped with. A corpus without this key simply has no index.
pub const META_KEY: &str = "vec0_index";

/// Vectors per transaction during a [`build`].
///
/// A build of the live corpus writes 2.68 M vectors — 4.1 GB. Appending that in
/// one transaction is cheap (new pages need no rollback journal), but a REBUILD
/// reuses the pages the dropped index freed, and overwritten pages ARE journaled:
/// one transaction would put a journal the size of the index beside the corpus.
/// Batching bounds it — measured 4.1 MB peak across a full rebuild of that corpus
/// — and costs nothing, because a build is not a corpus edit to protect:
/// [`META_KEY`] is written only after the last row lands, so a killed build
/// leaves a part-filled table that no query will use (see [`usable`]) and
/// `--rebuild` starts over.
const BUILD_BATCH: usize = 50_000;

/// Make the `vec0` module available to every connection opened AFTER this call.
///
/// `sqlite3_auto_extension` is process-wide and one-way, which is the whole
/// reason this is called from the connection helpers (`store::connect`,
/// `store::IndexStore::open`, `service::read_only`) rather than from the places
/// that query the index: a connection opened before registration would report
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

/// What a corpus records about its shadow index.
///
/// `vectors` and `chunks` are the two counts that make [`usable`] possible:
/// `vectors` is how many rows the index holds, `chunks` is how many rows
/// `chunks` held when the index was last maintained. Every writer that touches
/// `chunks` and maintains the index moves both, in the same transaction as the
/// rows themselves, so the pair is either current or provably not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexState {
    /// The embedding model whose vectors the index holds. A query embedded with
    /// anything else must not touch it — cosine across two embedding spaces is
    /// a meaningless number, which is the same rule the scan applies per row.
    pub model: String,
    /// Vector width, in floats. Fixed at `CREATE VIRTUAL TABLE` time and not
    /// changeable afterwards, so a corpus that changes model width needs a
    /// rebuild rather than a migration.
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
    /// There is no index. The ordinary case: nothing to say about it.
    Absent,
    /// There is one, and it is not being used. The reason travels back to the
    /// caller because an index that silently stops being used is indistinguishable
    /// from one that was never built, and the two want completely different
    /// operator responses.
    Declined(String),
}

/// Whether `SHADOW_TABLE` exists in this corpus.
///
/// A `sqlite_master` lookup, so it costs nothing and — importantly — does not
/// touch the virtual table. Naming a `vec0` table in a query is what makes
/// SQLite instantiate the module; this deliberately does not.
pub fn present(connection: &Connection) -> Result<bool> {
    let found: Option<i64> = connection
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1",
            [SHADOW_TABLE],
            |row| row.get(0),
        )
        .optional()?;
    Ok(found.is_some())
}

/// Read the recorded [`IndexState`], if the corpus has one.
///
/// `None` covers three shapes that are all "no usable index": no `meta` table
/// (a corpus older than it), no key, and a key holding something that does not
/// parse. The last one is not an error on purpose — a corrupt marker must
/// degrade to the scan, not fail the query.
pub fn state(connection: &Connection) -> Result<Option<IndexState>> {
    let raw: Option<String> = connection
        .query_row("SELECT value FROM meta WHERE key=?1", [META_KEY], |row| {
            row.get(0)
        })
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

/// Record `state` under [`META_KEY`]. Rides the caller's transaction.
pub fn write_state(connection: &Connection, state: &IndexState) -> Result<()> {
    connection.execute(
        "INSERT INTO meta(key,value) VALUES(?1,?2) \
         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        params![META_KEY, serde_json::to_string(state)?],
    )?;
    Ok(())
}

/// Decide whether this corpus' index can serve a query embedded with `model` at
/// `dimensions` floats.
///
/// Everything that can make an index unfit is checked here, and each one is
/// named rather than collapsed into a boolean:
///
/// * the table is not there — the ordinary, silent case;
/// * it is there but was never finished (`meta` key absent), i.e. a build was
///   killed part-way;
/// * it holds another model's vectors, or another width;
/// * the corpus moved underneath it. This is the one that matters in practice:
///   every `llm-index` release before this one writes `chunks` rows without
///   maintaining the index, so a corpus indexed by an older build after the
///   index was created holds vectors the index has never seen. Both witnesses
///   are checked — the job stamp catches a job that rewrote the corpus, the row
///   count catches an edit that bypassed the pipeline — and either one turns a
///   silently wrong answer into a fallback with a reason.
///
/// Both checks are single indexed row lookups plus `SELECT COUNT(*) FROM
/// chunks`, which SQLite answers from the narrowest index on the table
/// (`idx_chunks_file`) rather than the row data. Measured on the live corpus
/// they are a small fraction of the k-NN they guard — see
/// `docs/ARCHITECTURE.md` — which is what makes checking on every query, rather
/// than trusting a marker, affordable.
pub fn usable(connection: &Connection, model: &str, dimensions: usize) -> Result<Usable> {
    if !present(connection)? {
        return Ok(Usable::Absent);
    }
    let Some(state) = state(connection)? else {
        return Ok(Usable::Declined(format!(
            "{SHADOW_TABLE} exists but no completed build is recorded; \
             a build was interrupted — rebuild it with `llm-index vector-index --rebuild`"
        )));
    };
    if state.model != model || state.dimensions != dimensions {
        return Ok(Usable::Declined(format!(
            "the shadow index holds {} ({}d) vectors and this query is {model} ({dimensions}d)",
            state.model, state.dimensions
        )));
    }
    if state.vectors == 0 {
        return Ok(Usable::Declined(
            "the shadow index is empty; scanning so the corpus can say why".into(),
        ));
    }
    let job = job_stamp(connection);
    if job != state.job {
        return Ok(Usable::Declined(format!(
            "the shadow index is stale: an index job has run against this corpus since it was \
             maintained (last_job_started_at {} -> {}) — the build that ran it does not maintain \
             the index. Rebuild with `llm-index vector-index --rebuild`",
            state.job.as_deref().unwrap_or("none"),
            job.as_deref().unwrap_or("none"),
        )));
    }
    let chunks: i64 = connection.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
    if chunks as usize != state.chunks {
        return Ok(Usable::Declined(format!(
            "the shadow index is stale: it was maintained at {} chunks and the corpus now holds \
             {chunks} — something wrote this corpus without maintaining it. Rebuild with \
             `llm-index vector-index --rebuild`",
            state.chunks
        )));
    }
    Ok(Usable::Ready(state))
}

/// The `k` nearest `chunks.id`s to `query`, nearest first.
///
/// `query` is the raw little-endian `f32` BLOB — the same encoding
/// `chunks.embedding` holds, so no conversion happens anywhere on this path.
/// Only ids come back: the caller re-scores them against the stored BLOBs with
/// the scan's own cosine, so the score in a response means exactly what it
/// meant before this index existed, whichever path produced it.
pub fn knn(connection: &Connection, query: &[u8], k: usize) -> Result<Vec<i64>> {
    let mut statement = connection.prepare(&format!(
        "SELECT rowid FROM {SHADOW_TABLE} WHERE embedding MATCH ?1 AND k = ?2 ORDER BY distance"
    ))?;
    let rows = statement.query_map(params![query, k.clamp(1, K_MAX) as i64], |row| row.get(0))?;
    Ok(rows.collect::<Result<Vec<i64>, _>>()?)
}

/// Ceiling `sqlite-vec` puts on `k` in a k-NN query. Far above [`MAX_HITS`]
/// (100) and its tie margin, so clamping here can never bite a real request —
/// it is here so a caller cannot turn a mis-parsed limit into a vtab error.
///
/// [`MAX_HITS`]: crate::embedding::MAX_HITS
const K_MAX: usize = 4_096;

/// Create the (empty) virtual table for `dimensions`-wide vectors.
///
/// `distance_metric=cosine` matches how the corpus is ranked. It affects only
/// which candidates the k-NN returns; every score a caller sees is recomputed
/// from the stored BLOB by [`crate::embedding`], so the metric declared here can
/// never change a reported number.
pub fn create(connection: &Connection, dimensions: usize) -> Result<()> {
    connection
        .execute_batch(&format!(
            "CREATE VIRTUAL TABLE {SHADOW_TABLE} USING vec0(\
               embedding float[{dimensions}] distance_metric=cosine)"
        ))
        .context("creating the vec0 shadow index")?;
    Ok(())
}

/// Drop the index and forget it was ever built.
///
/// `DROP TABLE` on a `vec0` table removes its shadow tables too, so this leaves
/// a corpus that is once again indistinguishable from one that never had an
/// index. Nothing is lost: every vector in it was a copy.
pub fn drop_index(connection: &Connection) -> Result<()> {
    connection.execute_batch(&format!("DROP TABLE IF EXISTS {SHADOW_TABLE}"))?;
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
        connection.execute("DELETE FROM meta WHERE key=?1", [META_KEY])?;
    }
    Ok(())
}

/// Mirror one chunk into the index. The caller's `id` is `chunks.id`.
pub fn insert(connection: &Connection, id: i64, embedding: &[u8]) -> Result<()> {
    connection.execute(
        &format!("INSERT INTO {SHADOW_TABLE}(rowid,embedding) VALUES(?1,?2)"),
        params![id, embedding],
    )?;
    Ok(())
}

/// Remove one chunk from the index. Silent when the id was never in it, which
/// is the normal case for a chunk the index excludes (another model, another
/// width) being deleted.
pub fn delete(connection: &Connection, id: i64) -> Result<()> {
    connection.execute(&format!("DELETE FROM {SHADOW_TABLE} WHERE rowid=?1"), [id])?;
    Ok(())
}

/// The vector width `model`'s rows are stored at, taken from the corpus itself.
///
/// Read from the first such row rather than from an embedding model: the index
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

/// (Re)build the index for an existing corpus, from the stored BLOBs alone.
///
/// This is the path the 2.68 M-vector live corpora take to gain an index: it
/// reads `chunks.embedding` and writes it into the virtual table, and touches no
/// model, no document and no file on disk. Nothing is re-embedded, and no table
/// the corpus' own data lives in is written — `files`, `fts`, `chunks` and
/// `vision` are read-only to a build, so the worst an interrupted one costs is
/// the time it had spent.
///
/// Any existing index is dropped first, so this is both "build" and "rebuild":
/// there is no incremental repair mode, because a stale index is stale by an
/// unknown amount and rebuilding is the only honest repair.
///
/// `progress` is called with `(written, total)` at each commit boundary, so a
/// CLI can report a build that runs for minutes.
pub fn build(
    connection: &mut Connection,
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
    drop_index(connection)?;
    create(connection, dimensions)?;
    let width = dimensions * 4;
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
            "INSERT INTO {SHADOW_TABLE}(rowid,embedding) VALUES(?1,?2)"
        ))?;
        let mut rows = read.query(params![from, BUILD_BATCH as i64])?;
        let mut seen = 0usize;
        let mut last = from;
        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let row_model = row.get_ref(1)?.as_str()?;
            let embedding = row.get_ref(2)?.as_blob()?;
            if row_model == model && embedding.len() == width {
                insert.execute(params![id, embedding])?;
                vectors += 1;
            } else {
                skipped += 1;
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
        built_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs_f64())
            .unwrap_or_default(),
        builder: crate::VERSION.to_string(),
    };
    // Last, and on its own: the marker is what makes an index usable, so it must
    // not be visible until every vector it vouches for is committed. Everything
    // before this point is a part-filled table nothing will read.
    write_state(connection, &state)?;
    progress(vectors, total);
    Ok(BuildReport {
        vectors,
        skipped,
        state,
    })
}

/// The `/corpus/status`-shaped summary of a corpus' index, for a CLI or a route
/// that wants to report what is there without deciding whether to use it.
pub fn describe(connection: &Connection) -> Result<serde_json::Value> {
    if !present(connection)? {
        return Ok(json!({"present": false}));
    }
    match state(connection)? {
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

    #[test]
    fn a_corpus_without_the_table_reports_absent_and_stays_untouched() {
        let connection = corpus();
        assert!(!present(&connection).unwrap());
        assert!(matches!(
            usable(&connection, MODEL, 3).unwrap(),
            Usable::Absent
        ));
        assert_eq!(describe(&connection).unwrap(), json!({"present": false}));
    }

    #[test]
    fn a_build_indexes_the_matching_vectors_and_counts_the_rest() {
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        add_chunk(&connection, 2, MODEL, &[0.0, 1.0, 0.0]);
        add_chunk(&connection, 3, "other-model", &[1.0, 0.0, 0.0]);
        add_chunk(&connection, 4, MODEL, &[1.0, 0.0, 0.0, 0.0]); // another width

        let report = build(&mut connection, MODEL, 3, |_, _| {}).unwrap();
        assert_eq!(report.vectors, 2);
        assert_eq!(report.skipped, 2, "a foreign model and a foreign width");
        assert_eq!(
            report.state.chunks, 4,
            "the staleness witness counts them all"
        );

        let nearest = knn(&connection, &vector_to_bytes(&[1.0, 0.0, 0.0]), 2).unwrap();
        assert_eq!(nearest, vec![1, 2], "exact match first");
        assert!(matches!(
            usable(&connection, MODEL, 3).unwrap(),
            Usable::Ready(_)
        ));
    }

    #[test]
    fn a_rebuild_replaces_the_index_rather_than_doubling_it() {
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        build(&mut connection, MODEL, 3, |_, _| {}).unwrap();
        add_chunk(&connection, 2, MODEL, &[0.0, 1.0, 0.0]);
        let report = build(&mut connection, MODEL, 3, |_, _| {}).unwrap();
        assert_eq!(report.vectors, 2);
        let rows: i64 = connection
            .query_row(&format!("SELECT COUNT(*) FROM {SHADOW_TABLE}"), [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(rows, 2, "the old table is dropped, not appended to");
    }

    #[test]
    fn insert_and_delete_keep_the_index_in_step_with_chunks() {
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        build(&mut connection, MODEL, 3, |_, _| {}).unwrap();

        add_chunk(&connection, 2, MODEL, &[0.9, 0.1, 0.0]);
        insert(&connection, 2, &vector_to_bytes(&[0.9, 0.1, 0.0])).unwrap();
        assert_eq!(
            knn(&connection, &vector_to_bytes(&[0.9, 0.1, 0.0]), 1).unwrap(),
            vec![2]
        );

        delete(&connection, 2).unwrap();
        assert_eq!(
            knn(&connection, &vector_to_bytes(&[0.9, 0.1, 0.0]), 5).unwrap(),
            vec![1],
            "a deleted chunk stops being a candidate"
        );
        // Deleting an id the index never held is a no-op, not an error: chunks
        // the index excludes are deleted through the same call site.
        delete(&connection, 99).unwrap();
    }

    #[test]
    fn an_index_is_declined_when_chunks_moved_underneath_it() {
        // The old-binary hazard, reproduced exactly: a build that does not
        // maintain the index writes a chunks row, and the index no longer covers
        // the corpus. Serving k-NN from it would silently miss that row.
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        build(&mut connection, MODEL, 3, |_, _| {}).unwrap();
        assert!(matches!(
            usable(&connection, MODEL, 3).unwrap(),
            Usable::Ready(_)
        ));

        add_chunk(&connection, 2, MODEL, &[0.0, 1.0, 0.0]); // no index maintenance
        let Usable::Declined(reason) = usable(&connection, MODEL, 3).unwrap() else {
            panic!("a corpus written behind the index must not be served from it")
        };
        assert!(reason.contains("stale"), "{reason}");
        assert!(reason.contains("--rebuild"), "{reason}");
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
        build(&mut connection, MODEL, 3, |_, _| {}).unwrap();
        assert!(matches!(
            usable(&connection, MODEL, 3).unwrap(),
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
        assert_eq!(live as usize, state(&connection).unwrap().unwrap().chunks);

        let Usable::Declined(reason) = usable(&connection, MODEL, 3).unwrap() else {
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
        build(&mut connection, MODEL, 3, |_, _| {}).unwrap();
        run_a_job(&connection, "1800");
        add_chunk(&connection, 2, MODEL, &[0.0, 1.0, 0.0]);
        assert!(matches!(
            usable(&connection, MODEL, 3).unwrap(),
            Usable::Declined(_)
        ));

        let report = build(&mut connection, MODEL, 3, |_, _| {}).unwrap();
        assert_eq!(report.state.job.as_deref(), Some("1800"));
        assert!(matches!(
            usable(&connection, MODEL, 3).unwrap(),
            Usable::Ready(_)
        ));
    }

    #[test]
    fn an_index_for_another_model_or_width_is_declined() {
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        build(&mut connection, MODEL, 3, |_, _| {}).unwrap();

        let Usable::Declined(reason) = usable(&connection, "another-model", 3).unwrap() else {
            panic!("cosine across two embedding spaces is meaningless")
        };
        assert!(reason.contains("another-model"), "{reason}");
        let Usable::Declined(reason) = usable(&connection, MODEL, 384).unwrap() else {
            panic!("a 3d index cannot answer a 384d query")
        };
        assert!(reason.contains("384d"), "{reason}");
    }

    #[test]
    fn an_interrupted_build_leaves_a_table_nothing_will_use() {
        // A build writes its marker last, so a table with no marker is exactly
        // what a killed build leaves. It must degrade to the scan, and say so.
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        build(&mut connection, MODEL, 3, |_, _| {}).unwrap();
        connection
            .execute("DELETE FROM meta WHERE key=?1", [META_KEY])
            .unwrap();

        let Usable::Declined(reason) = usable(&connection, MODEL, 3).unwrap() else {
            panic!("an unfinished build must not be trusted")
        };
        assert!(reason.contains("interrupted"), "{reason}");
        assert_eq!(
            describe(&connection).unwrap()["state"],
            serde_json::Value::Null
        );
    }

    #[test]
    fn dropping_leaves_a_corpus_indistinguishable_from_one_that_never_had_an_index() {
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        build(&mut connection, MODEL, 3, |_, _| {}).unwrap();
        drop_index(&connection).unwrap();

        assert!(!present(&connection).unwrap());
        assert!(state(&connection).unwrap().is_none());
        // Including the shadow tables sqlite-vec created beside it.
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
    fn a_corrupt_marker_degrades_to_the_scan_instead_of_failing_the_query() {
        let mut connection = corpus();
        add_chunk(&connection, 1, MODEL, &[1.0, 0.0, 0.0]);
        build(&mut connection, MODEL, 3, |_, _| {}).unwrap();
        connection
            .execute("UPDATE meta SET value='{not json' WHERE key=?1", [META_KEY])
            .unwrap();

        assert!(state(&connection).unwrap().is_none());
        assert!(matches!(
            usable(&connection, MODEL, 3).unwrap(),
            Usable::Declined(_)
        ));
    }
}
