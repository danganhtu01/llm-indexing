//! What a build WITHOUT `sqlite-vec` sees in a corpus that has a shadow index.
//!
//! The index is written into the published corpus, and the published corpus is
//! read by binaries this one does not control: every `llm-index` release before
//! this one, the `vlm-indexing` and `drives-analytics` readers, and anything an
//! operator points at the file. None of them has the `vec0` module. If a
//! `CREATE VIRTUAL TABLE` in the schema were enough to break them, the index
//! could not ship at all.
//!
//! It is not enough, and this asserts exactly why: SQLite records a virtual
//! table's declaration in `sqlite_master` and instantiates its module only when
//! a statement NAMES the table. A build with no module therefore reads, writes
//! and migrates a corpus that has one, and fails only on the one query it was
//! never going to issue.
//!
//! Simulated rather than asserted from reading the manual, and simulated
//! honestly: `sqlite3_cancel_auto_extension` removes the module registration
//! this crate installs, so every connection opened afterwards is exactly what an
//! older binary gets. That is process-wide, which is why this lives in its own
//! integration test binary and is a single test: it must not race another test
//! opening a connection.

use llm_indexing::embedding::{vector_to_bytes, EMBEDDING_MODEL};
use llm_indexing::store::connect;
use llm_indexing::vec0;
use rusqlite::{params, Connection};

/// Every table an older `llm-index` created, written by hand so this fixture
/// keeps describing THAT build rather than tracking `store::SCHEMA`.
const LEGACY_SCHEMA: &str = "\
CREATE TABLE files(
  id INTEGER PRIMARY KEY, path TEXT UNIQUE, drive TEXT, dir TEXT, name TEXT, ext TEXT,
  size INTEGER, mtime REAL, lang TEXT, method TEXT, ocr_used INTEGER, pages INTEGER,
  chars INTEGER, sha1 TEXT, indexed_at REAL
);
CREATE VIRTUAL TABLE fts USING fts5(name, path, content, tokens);
CREATE TABLE chunks(
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL, content TEXT NOT NULL, embedding BLOB NOT NULL,
  dimensions INTEGER NOT NULL, model TEXT NOT NULL, UNIQUE(file_id, chunk_index)
);
CREATE INDEX idx_chunks_file ON chunks(file_id);
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
";

/// The `sqlite3_vec_init` pointer, cast the way SQLite's auto-extension list
/// wants it — the same cast `vec0::register` performs.
type Initializer = unsafe extern "C" fn(
    *mut rusqlite::ffi::sqlite3,
    *mut *mut std::os::raw::c_char,
    *const rusqlite::ffi::sqlite3_api_routines,
) -> std::os::raw::c_int;

fn initializer() -> Initializer {
    // SAFETY: the entry point sqlite-vec exports for exactly this call, cast as
    // its own README and this crate's `vec0::register` cast it.
    unsafe {
        std::mem::transmute::<*const (), Initializer>(sqlite_vec::sqlite3_vec_init as *const ())
    }
}

/// Stop handing the `vec0` module to newly opened connections.
///
/// The inverse of [`vec0::register`], and the only way to observe the
/// no-such-module world from inside a process that has the extension linked in.
fn forget_the_vec0_module() {
    // SAFETY: cancelling a registered auto-extension is a supported SQLite
    // operation and affects only connections opened after it.
    unsafe { rusqlite::ffi::sqlite3_cancel_auto_extension(Some(initializer())) };
}

/// Put it back. Not [`vec0::register`], which is `Once`-guarded and has already
/// fired in this process — re-registering after a cancel has to go to SQLite
/// directly.
fn remember_the_vec0_module() {
    // SAFETY: as `vec0::register`. SQLite ignores a duplicate registration.
    unsafe { rusqlite::ffi::sqlite3_auto_extension(Some(initializer())) };
}

#[test]
fn a_corpus_with_a_shadow_index_stays_readable_and_writable_without_the_module() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("corpus.sqlite");

    // 1. Build the corpus and its shadow index with the module available.
    {
        let mut connection = connect(&path).unwrap();
        connection.execute_batch(LEGACY_SCHEMA).unwrap();
        // What `run_index` stamps before it writes a thing — half the staleness
        // witness, and written by every build that has ever had the key.
        connection
            .execute(
                "INSERT INTO meta(key,value) VALUES('last_job_started_at','1700')",
                [],
            )
            .unwrap();
        for id in 1..=8i64 {
            connection
                .execute(
                    "INSERT INTO files(id,path,name,size,mtime,method,indexed_at) \
                     VALUES(?1,?2,?3,10,0.0,'text',1700.0)",
                    params![id, format!("/a/file{id}.txt"), format!("file{id}.txt")],
                )
                .unwrap();
            connection
                .execute(
                    "INSERT INTO fts(rowid,name,path,content,tokens) VALUES(?1,?2,?3,?4,'')",
                    params![
                        id,
                        format!("file{id}.txt"),
                        format!("/a/file{id}.txt"),
                        "some indexed text"
                    ],
                )
                .unwrap();
            connection
                .execute(
                    "INSERT INTO chunks(file_id,chunk_index,content,embedding,dimensions,model) \
                     VALUES(?1,0,?2,?3,3,?4)",
                    params![
                        id,
                        format!("chunk {id}"),
                        vector_to_bytes(&[id as f32, 1.0, 0.0]),
                        EMBEDDING_MODEL
                    ],
                )
                .unwrap();
        }
        for tier in [vec0::Tier::Float, vec0::Tier::Int8] {
            let report = vec0::build(&mut connection, tier, EMBEDDING_MODEL, 3, |_, _| {}).unwrap();
            assert_eq!(report.vectors, 8, "{tier:?}");
        }
    }

    // 2. Become an older binary: no `vec0` module from here on.
    forget_the_vec0_module();
    let old = Connection::open(&path).unwrap();

    // 3. Everything an older build does with a corpus still works. These are
    //    the statements that matter — the resume scan, the FTS search, the
    //    ranking scan, the `ALTER TABLE` migration, and a write — and none of
    //    them names the virtual table.
    let files: i64 = old
        .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))
        .unwrap();
    assert_eq!(files, 8);
    let resume: i64 = old
        .query_row(
            "SELECT COUNT(*) FROM files f WHERE EXISTS(SELECT 1 FROM chunks c WHERE c.file_id=f.id)",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(resume, 8);
    let hit: String = old
        .query_row(
            "SELECT f.path FROM fts JOIN files f ON f.id=fts.rowid WHERE fts MATCH 'indexed' \
             ORDER BY bm25(fts) LIMIT 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(hit.starts_with("/a/file"), "{hit}");
    let vectors: i64 = old
        .query_row(
            "SELECT COUNT(*) FROM (SELECT id,model,dimensions,embedding FROM chunks)",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        vectors, 8,
        "the exhaustive scan reads its own table as ever"
    );

    // The schema migration this build applies on every open, on a corpus that
    // now also holds a virtual table it knows nothing about.
    old.execute_batch(
        "BEGIN IMMEDIATE;
         ALTER TABLE files ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0;
         UPDATE files SET attempts = 3 WHERE method LIKE 'error:%';
         COMMIT;",
    )
    .unwrap();

    // And a write: an older job indexing into this corpus succeeds. It leaves
    // the shadow index stale, which is precisely what `vec0::usable` exists to
    // catch — see the assertion at the end. Note the shape deliberately: one
    // chunk added, one file's chunk removed, so the ROW COUNT lands exactly
    // where it started and only the job stamp gives it away.
    old.execute(
        "UPDATE meta SET value='1800' WHERE key='last_job_started_at'",
        [],
    )
    .unwrap();
    old.execute(
        "INSERT INTO chunks(file_id,chunk_index,content,embedding,dimensions,model) \
         VALUES(1,1,'written by an older build',?1,3,?2)",
        params![vector_to_bytes(&[1.0, 1.0, 0.0]), EMBEDDING_MODEL],
    )
    .unwrap();
    old.execute("DELETE FROM chunks WHERE file_id=8", [])
        .unwrap();
    old.execute("DELETE FROM files WHERE id=8", []).unwrap();

    // 4. The ONE thing it cannot do is query the virtual tables — which it
    //    never would, because nothing in a build without this feature mentions
    //    them. Both slots, because the quantised one is a second
    //    `CREATE VIRTUAL TABLE` in the same published corpus and inherits none
    //    of the first one's evidence.
    for slot in vec0::Slot::ALL {
        let refused = old
            .prepare(&format!("SELECT rowid FROM {} LIMIT 1", slot.table()))
            .expect_err("a build with no vec0 module cannot read a vec0 table");
        assert!(refused.to_string().contains("no such module"), "{refused}");
        // Their shadow storage is ordinary tables, so even that is readable.
        let shadow_rows: i64 = old
            .query_row(
                &format!("SELECT COUNT(*) FROM {}_rowids", slot.table()),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(shadow_rows, 8, "{slot:?}");
    }
    drop(old);

    // 5. Back in a build that HAS the module: the corpus the older binary wrote
    //    is intact, and its index is refused rather than trusted.
    remember_the_vec0_module();
    let connection = connect(&path).unwrap();
    let chunks: i64 = connection
        .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
        .unwrap();
    assert_eq!(chunks, 8, "7 files' chunks plus the one it added");
    for slot in vec0::Slot::ALL {
        assert_eq!(
            chunks as usize,
            vec0::state(&connection, slot).unwrap().unwrap().chunks,
            "the row count is no help here — that is the point"
        );
        let vec0::Usable::Declined(reason) =
            vec0::usable(&connection, slot, EMBEDDING_MODEL, 3).unwrap()
        else {
            panic!("an index written behind by an older binary must not be served from")
        };
        assert!(reason.contains("stale"), "{reason}");
        assert!(reason.contains("--rebuild"), "{reason}");
    }

    // And the repair works from here: a rebuild reads the corpus the older
    // binary left and the index is trustworthy again — with nothing
    // re-embedded, because every vector it needs is already in `chunks`.
    let mut connection = connection;
    for tier in [vec0::Tier::Float, vec0::Tier::Int8] {
        let report = vec0::build(&mut connection, tier, EMBEDDING_MODEL, 3, |_, _| {}).unwrap();
        assert_eq!(report.vectors, 8, "{tier:?}");
        assert!(
            matches!(
                vec0::usable(&connection, tier.slot(), EMBEDDING_MODEL, 3).unwrap(),
                vec0::Usable::Ready(_)
            ),
            "{tier:?}"
        );
    }
}
