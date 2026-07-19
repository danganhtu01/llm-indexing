# llm-indexing — Engine settings surface (wave 2 spec)

> **WAVE 2 — implemented AFTER the vision-v1 worker fleet lands on this branch.**
> Current-wave workers: ignore this file (do not implement, do not flag as missing).

Owner request 2026-07-19: allow per-job inputs for the type and quality of OCR, the
type and quality of image analysis, etc. Build the ENGINE settings surface now; the
ff-lc-app / da-academic app UIs consume it later.

Constraints (owner directives): Rust only; flat structure — extend existing files
plus at most one new `src/settings.rs`; single source of truth — one settings struct
pair serves BOTH service-wide config defaults and per-job overrides via an explicit
merge, never two definitions of the same knob.

## 1. Per-job settings (JobRequest additions, all optional)

```jsonc
{
  "ocr_opts": {              // overrides service config for THIS job only
    "dpi": 300,              // 150..=1200
    "psm": "3",              // validated tesseract PSM 0..=13 (string, engine style)
    "preprocess": true,      // ImageMagick pre-pass on/off
    "max_pages": 20,         // 1..=500
    "langs": "vie+eng+rus"   // validated against INSTALLED tessdata; wins over the
  },                         //   legacy top-level ocr_langs (kept for compat)
  "vision_opts": {           // active only when vision tier != off; capped by --vision-max
    "detector": "nano",      // off | nano   (future sizes join here)
    "detector_conf": 0.5,    // 0.05..=0.95
    "tagger": "clip",        // off | clip
    "tag_threshold": 0.22,   // 0.0..=1.0
    "tag_top_k": 8,          // 1..=32
    "captioner": "florence2",// off | florence2 (only if models present)
    "max_frames": 12,        // 1..=64 (video keyframes)
    "timeout_secs": 60       // 5..=1800 per-file vision timeout
  }
}
```

- Validation at submit: `400` with a field-specific message on any out-of-range /
  unknown-enum value or uninstalled OCR language; unknown top-level fields remain
  permissively ignored (existing serde posture, forward-compat).
- Precedence: per-job `*_opts` field > service config > built-in default. Implement
  as `OcrSettings::merge(base, override)` / `VisionSettings::merge(...)` — the ONLY
  merge path, unit-tested.
- Native CLI parity: `index` gains matching flags (`--ocr-dpi`, `--ocr-psm`,
  `--ocr-preprocess`, `--ocr-max-pages`, `--vision-detector`, …) feeding the same
  merge.

## 2. `GET /settings` — capabilities discovery (the app-integration contract)

Read-only endpoint the apps will render their settings UIs from (no hardcoding in
ff-lc-app / da-academic / drives-analytics later):

```jsonc
{
  "version": "0.5.0",
  "ocr": {
    "modes": ["auto","on","off","exhaustive"],
    "langs_installed": ["vie","eng","rus","deu"],   // bundled tessdata ∪ system packs
    "dpi": {"min":150, "max":1200, "default":300},
    "psm": {"values":["0".."13"], "default":"3"},
    "preprocess_default": true,
    "max_pages": {"min":1, "max":500, "default":20}
  },
  "vision": {
    "max_tier": "tags",                    // this serve process's --vision-max cap
    "tiers_available": ["meta","tags"],    // gated on model files actually present
    "detectors": [{"id":"nano","present":true}],
    "taggers":   [{"id":"clip","present":true}],
    "captioners":[{"id":"florence2","present":false}],
    "defaults": {"detector_conf":0.5,"tag_threshold":0.22,"tag_top_k":8,
                 "max_frames":12,"timeout_secs":60}
  },
  "workers": {"default":4, "max":64}
}
```

- `langs_installed`: enumerate the bundled tessdata dir + system tessdata (the same
  resolution order the OCR module already uses); never a hardcoded list.
- `tiers_available` / `present`: model-file existence + hash check reuse the
  fetch-data verification helpers.

## 3. Config file

The same two settings structs provide the YAML config's `ocr:`/`vision:` sections
(service-wide defaults). No new config keys beyond the knobs above.

## 4. Compatibility

Every addition optional; absent fields = exactly today's behavior. `EngineJob`
response shape, `/corpus/*`, and existing clients unchanged. `/settings` is purely
additive.

## 5. Tests

Merge-precedence table tests; per-field bounds (accept edge, reject beyond); langs
validation rejects uninstalled packs with a helpful message; `/settings` shape +
tessdata enumeration against a fixture dir; plumb-through proof that a per-job `dpi`
actually reaches the tesseract invocation (inspect the built command/config in a
unit seam); CLI flag parity with HTTP.

## 6. Wave-2 worker plan (small fleet)

W1 opus: settings structs + merge + JobRequest/CLI wiring + submit validation.
W2 opus: `GET /settings` + capability probes + tests.
W3 sonnet: docs (HTTP_API.md update, README settings section, config reference).
Then integrate (gauntlet green) + review-lite (compat + validation-bypass lenses).
