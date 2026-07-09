# The L&C document pipeline

Turn the Compliance-Team Google Drive into a database the legal & compliance team can
generate documents from, with the PII stripped out on the way, and a web app that lets
them run each step by pressing a button.

Runs on **ArchFF** (see [`danganhtu01/archff`](https://github.com/danganhtu01/archff)).
All new code is **Rust**; the front end is **WebAssembly**.

---

## 1. The shape

```
/srv/lc/compliance-team/   stage 0  ingest    rclone copy, every minute   [archff/rclone/]
        /index/            stage 1  index     index.sqlite, manifest.jsonl, sidecar/**.txt
        /redacted/         stage 2  redact    redacted mirror + redactions.jsonl
        /corpus/           stage 3  corpus    corpus.sqlite — the document DB
        /generated/        stage 4  generate  emitted .docx
        /state/                     runs.sqlite, locks
```

Each stage is a **container**: its input directory bound read-only, its output directory
bound read-write, and a `run.json` written on completion. That is the entire interface —
see [`crates/lc-core/src/lib.rs`](../crates/lc-core/src/lib.rs). Because it is the entire
interface, the language a stage is written in is an implementation detail.

**Stage 0 is not ours.** `archff/rclone/compliance-team-copy.timer` already mirrors the
Drive every minute. Index the mirror at `/srv/lc/compliance-team`, **never** the FUSE
mount at `/srv/lc/compliance-team-live` — walking the mount drags 3.2 GiB through the VFS
cache on every pass.

### The security invariant

`/srv/lc/compliance-team` and `/srv/lc/index` hold **un-redacted PII** — `index.sqlite`
contains the full text of every document. Only `/srv/lc/redacted` and `/srv/lc/generated`
may ever be reachable from an HTTP handler. `Stage::is_web_servable()` encodes this, and
a unit test fails if anyone widens it.

---

## 2. Why systemd units and not a Docker socket

The obvious design gives the web app `/var/run/docker.sock` so it can start containers.
Don't. That socket is **root-equivalent**, and the web app is the one process you are
deliberately exposing to humans.

Instead each stage is a **systemd oneshot unit** whose `ExecStart` is the `docker run`.
`lc-api` runs unprivileged (group `lc`) and starts those units over D-Bus with `zbus`,
gated by a polkit rule permitting exactly `start` on exactly those units. You get:

- **no docker socket** anywhere near an HTTP handler;
- **non-overlap for free** — systemd will not re-trigger a running service;
- **logs in journald**, which the UI streams over SSE;
- and *"next automatic run"* is just the `NextElapseUSecRealtime` property on the timer,
  rather than a scheduler you have to write and then debug.

It is also the idiom this box already uses, for `compliance-team-copy.timer`.

Chain the stages with `OnSuccess=` so a manual index cascades into redact → corpus.

---

## 3. Stage 1 — index

The existing Python indexer (`src/claude_index/`) is **frozen behind the container
contract**. It is validated on an 80k-file drive, it handles PDF/DOCX/XLSX/PPTX/email and
OCRs scans with Tesseract (`vie+eng`), and its recall work — diacritic folding, Hunspell
lemmas, Vietnamese maximum-matching over a 74k lexicon, abbreviation expansion — is not
something to rewrite on day one to satisfy a language rule.

Three Windows assumptions must be overridden in the image:
`config.py` defaults `tesseract_cmd` to `C:\Program Files\Tesseract-OCR\tesseract.exe`;
`store._write_sidecar` calls `os.path.splitdrive`; and `REPO_ROOT` resolves `data/dict`
and `data/tessdata` relative to the source tree, so both must be baked in or mounted.

`--resume` keys on `(size, int(mtime))`, which is exactly what incremental re-indexing
every few minutes needs.

**`crates/lc-index/` is its intended Rust replacement** — `tantivy` for FTS5,
`pdfium-render` for PyMuPDF, `leptess` or `ocrs` for pytesseract, `calamine`/`docx-rs`/
`zip` for the Office readers, `whatlang` + `rust-stemmers` + `zspell` for the language
stack. When it lands, only `compose.yaml` changes. Budget two to three weeks; the
contract makes it a drop-in swap with zero downstream change.

---

## 4. Stage 2 — redact

`crates/lc-redact/` — **written**, tested, and the first stage of this pipeline that
exists.

### Why not just use claudeops' `redact` crate

`danganhtu01/claudeops` has a `redact` library crate (283 LOC, `regex` + `serde_json`).
Its rule set is good. But it is a *library with no entrypoint* — its own
`compose.governance.yml` says so:

> `classify/redact expose only a parity-test [[bin]] ... They each need a real CLI`
> `entrypoint before they can become an add-on container — do NOT fake one.`

And for a **document** corpus it has four gaps, three of which fail silently — by leaving
PII in place, not by erroring:

| Gap | Consequence |
|---|---|
| No person-name detection at all (pure regex, no NER, no gazetteer) | every signatory name survives |
| MST tax codes matched only as a JSON **key name** | no free-text pattern; every tax code in a contract survives |
| CCCD/CMND and bank accounts are **context-gated** | a bare 12-digit ID whose label is a table header three rows up survives |
| No audit trail | you cannot prove what was removed, or from where |

That third one is not theoretical. Against a real contract fixture:

```
Số: 079200001234              ← a CCCD, labelled, but not with a gated keyword

--gating context-gated  (claudeops behaviour) → 079200001234   LEAKS
--gating strict         (our default)         → [REDACTED]
```

So `lc-redact` runs **strict by default**: a bare 9- or 12-digit run is redacted with no
keyword required. It over-redacts some invoice numbers. It never leaks a national ID.
`--gating context-gated` restores claudeops' behaviour if you decide otherwise.

Bank accounts stay keyword-gated in **both** modes — an ungated 6–24-digit run matches
almost every invoice line, and the false-positive rate destroys the corpus rather than
protecting it.

### The audit trail

`redactions.jsonl`, one object per span:

```json
{"file":"hop-dong/hd-001.txt","start":72,"end":86,"kind":"tax_code",
 "rule":"tax_code_mst","digest":"0e93d0f9a0b5401a"}
```

`digest` is a **salted** SHA-256 of the matched text, truncated to 8 bytes. The salt is
generated per run and never persisted. So you can prove the same identifier recurred
across two documents **without storing the identifier**, and the log is useless as a
rainbow-table target. The redacted value itself never appears anywhere.

### Person names

Names cannot be done with a regex, and claudeops does not try. A **gazetteer** is the
honest 80% answer for Vietnamese: the surname space is tiny (~14 names cover most of the
population) and highly distinctive, so `<Surname> <Given> <Given>` in title case is a
strong signal. It over-matches place names that share a surname (Nguyễn Huệ is a street).
It misses foreign names entirely. `run.json` records `gazetteer_surnames` so nobody
mistakes this for NER. Extend with `--names <file>`.

### Failure model

Fail-open per file, fail-closed per span. One unreadable file must never kill a run over
500 documents — it is counted, warned about in `run.json`, and **not written to the
output**, so nothing is ever emitted half-redacted. Output files are written to a temp
name and renamed, because this box loses mains power and a truncated partially-redacted
document is worse than a missing one.

---

## 5. Stages 3 and 4 — corpus and generate

**Not yet written.** Both have most of their parts sitting in repos you already own.

`crates/lc-corpus/` builds `corpus.sqlite`. Lift the provenance patterns from
`FinFan-Compliance-App`'s `crates/db` verbatim: the append-only `audit_log`
(`IDENTITY` PK, no `UPDATE`/`DELETE` path), the frozen `case_sources.snapshot JSONB`, the
`case_events` timeline. Those are exactly the shape a documents/entities/clauses/
provenance schema needs. That app uses **Postgres** via `sqlx`'s runtime API; `sqlx`
speaks SQLite through the same API, and SQLite matches what stage 1 already emits and
keeps each stage's output a single file you can copy. **This is an open decision.**

`crates/lc-generate/` emits `.docx`. **Reuse `knit-md-docx`**, which is already written:
`rust_knit_md_docx::to_bytes(&markdown) -> Vec<u8>`. Render a template plus corpus rows
into Markdown, hand it over, stream the bytes. It covers headings with outline levels,
GFM tables, nested lists, images, footnotes.

Two traps. The crate is named **`rust_knit_md_docx`**, not `knit-md-docx`. And it depends
on a *fork* of `docx-rs` through a **relative path dependency** (`../docx-rs/docx-core`),
so the `git =` dependency its README suggests will not resolve — you need a workspace
`[patch]` pointing at `danganhtu01/knit-md-docx-rs`. It is native-only: generate `.docx`
server-side, never in the browser. It has no hard page breaks; add them via the
re-exported `docx_rs` API if a template needs one.

---

## 6. The web app

`crates/lc-api/` (axum + tokio + zbus) and `crates/lc-web/` (Leptos → WASM, built with
Trunk). **Not yet written.**

Fork the shell from `FinFan-Compliance-App`: its Axum 0.8 harness, its `Router::merge`
route assembly, and above all its `/api/lists/{fetch,clean,use}/{key}` staged endpoints,
which are a one-to-one template for *one button per pipeline stage*. Its Leptos 0.7 CSR
app and `gloo-net` client come across too.

Four things are genuinely absent from every repo we own and have to be written:

1. **systemd/D-Bus orchestration** (`zbus`) — no scheduler exists anywhere.
   `LIST_REFRESH_CRON` is documented in that app's `.env.example` and never implemented.
2. **"Next automatic run"** — read `NextElapseUSecRealtime` off the systemd Timer object.
3. **Live log streaming** — there is no SSE or WebSocket in any of these repos. Add an
   axum SSE endpoint tailing journald, and `web-sys::EventSource` in the UI.
4. **Full-text search** over the corpus — no `tantivy`, no `rusqlite` anywhere.

---

## 7. Hosting

This box is on Tailscale (`archff` = `100.104.247.21`).

**Recommended — Tailscale Serve.** `tailscale serve --bg --https=443 http://127.0.0.1:8080`
gives `https://archff.<tailnet>.ts.net` with a real Let's Encrypt certificate, no open
ports and no router changes. The reason this is *the* answer for a compliance app:
`lc-api` can read the `Tailscale-User-Login` header, so you get a per-user audit trail
without building authentication at all.

**LAN-only** (`0.0.0.0:8080`, firewalled to the subnet) is simplest but has no
authentication and no remote access.

**Public internet:** `tailscale funnel 443 on` is one command but publishes an
*unauthenticated* URL. If outside access is a hard requirement, use **Cloudflare Tunnel +
Cloudflare Access** — outbound-only, your own domain, SSO against the Google Workspace
that already owns the Drive. Port-forwarding Caddy to the public internet is the one
option to rule out.

---

## 8. Prerequisites on ArchFF

```bash
sudo pacman -S rust-wasm trunk docker-compose   # rust-wasm matches the `rust` pkg version exactly
sudo systemctl enable --now docker
```

Arch's `rust` ships **only** `x86_64-unknown-linux-gnu` std — `extra/rust-wasm` adds the
`wasm32-unknown-unknown` target at the same version, so no `rustup` migration is needed.
`docker compose` is a separate package. **`atdang` does not need to be in the `docker`
group**: the stage containers are run by root via systemd, and `lc-api` reaches them over
D-Bus. That is the point of §2.

## 9. Status

| Stage | Crate | State |
|---|---|---|
| 0 ingest | — | **live** (`archff/rclone/`) |
| 1 index | `src/claude_index/` (Python) | works; needs a Dockerfile |
| 1 index | `crates/lc-index/` (Rust) | not started |
| 2 redact | `crates/lc-redact/` | **written, 12 tests passing** |
| — | `crates/lc-core/` | **written** — the stage contract |
| 3 corpus | `crates/lc-corpus/` | not started; DB engine undecided |
| 4 generate | `crates/lc-generate/` | not started; reuse `knit-md-docx` |
| api | `crates/lc-api/` | not started |
| web | `crates/lc-web/` | not started |
| deploy | `deploy/` | not started |
