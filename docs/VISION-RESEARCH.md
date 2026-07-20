# Local Vision Stack for the RTX 3070 Ti — Research Report
**Target:** Corsair Windows 11 desktop, NVIDIA RTX 3070 Ti (8 GB VRAM, Ampere sm_86), feeding llm-indexing (Rust, ONNX Runtime via fastembed). Scope: fully offline identification/classification/description of a 100k–500k-file photo+video library. Compiled 2026-07-19 from 5 research sweeps + gap-fill; every recommendation sourced in §8.

---

## 1. Executive summary

1. Run **native Windows 11** (not WSL2) with Rust + `ort` 2.0.0-rc.12 using the **CUDA EP** (CUDA 12.x + cuDNN 9); add the TensorRT EP with persisted engine caches once the model set freezes [ORT EP docs 2026; ort rc.12 2026-03].
2. **One embedding model is the backbone**: **SigLIP 2** (`ViT-B-16-SigLIP2__webli`, Apache-2.0, ready ONNX from Immich's HF org) serves text search, near-dup tier-2, zero-shot Places365 scenes, and the LAION aesthetic-MLP — four tiers, one vector, no extra VRAM [Immich searching docs 2026; LAION audit 2026-01].
3. **Objects: skip Ultralytics YOLO entirely** — AGPL-3.0 weights are a bundling hazard even for internal use per Ultralytics' own position. Use **RF-DETR-Nano/Medium or D-FINE-L (both Apache-2.0, official ONNX+TensorRT, COCO 48–55 AP ≈ YOLO-class)** [Ultralytics license 2026; RF-DETR 2025; D-FINE ICLR 2025].
4. **Tags:** RAM++ (Apache, open-set ~4,500 tags) after a one-time ONNX export; **captions:** **Florence-2-large (MIT, onnx-community export, batchable, <1 GB)** for the bulk pass, with **Qwen3-VL-4B Q4 via a llama.cpp sidecar** for rich descriptions on a curated subset only.
5. **Faces:** buffalo_l (SCRFD-10G + ArcFace-R50, best accuracy, weights **non-commercial** — fine for a personal library, decide at engine level) or the fully clean **YuNet + SFace (Apache)** pair; cluster with DBSCAN + a face-quality gate [InsightFace 2026; digiKam 8.6 2025-03].
6. **Dedup costs nothing new:** reuse the existing pHash (meta tier) + SigLIP2 vectors, indexed in **usearch** (native Rust, HNSW, Hamming + cosine in one lib) [USearch 2026].
7. **Video:** NVDEC-accelerated ffmpeg keyframes (H.264/HEVC/VP9/AV1 on Ampere) → same image pipeline; Whisper large-v3-turbo INT8 (~1.5 GB) for speech [NVDEC 2026; Whisper 2026].
8. Everything fits 8 GB **sequentially staged, one heavy model resident per phase**; embeddings+detection over 100k photos ≈ **hours**, captions are the bottleneck (**tier-gate them**: caption the post-dedup, post-quality subset, not everything).
9. Whole stack is achievable with **zero AGPL and zero cloud**; only faces carry a license decision.
10. First on-box validations: ort CUDA EP smoke test, SigLIP2 throughput, Florence-2 Rust generation-loop prototype (the one real integration risk).

---

## 2. Recommendation matrix

| Task | Recommended | Runner-up | VRAM | Speed on 3070 Ti (FP16, est.) | Quality evidence | License | Windows runtime path |
|---|---|---|---|---|---|---|---|
| **Tagging (multi-label)** | RAM++ (Swin-L) | SigLIP2 zero-shot over custom label lists; Florence-2 `<OD>` tags | ~2–3 GB | few hundred img/s batched | Beats prior SOTA on common+uncommon tags, ~4,500 open-set tags [RAM++ 2023–25] | Apache-2.0 | Manual ONNX export → ort CUDA EP |
| **Objects (closed-set)** | RF-DETR-Nano/Medium | D-FINE-L | ~1–2 GB | ~180–220 fps (Nano); T4 figures ×1.6–2 | Nano 48.4 / Medium 54.7 COCO AP; D-FINE-L 54.0 [RF-DETR 2025; D-FINE ICLR 2025] | Apache-2.0 (core) | Official ONNX+TRT export → ort |
| **Open-vocab detect (on-demand only)** | OWLv2 | Grounding DINO 1.5 (accuracy one-offs); YOLO-World (AGPL) | 3–6 GB | slow (~few fps) — never bulk | Strong LVIS rare-class recall [OWLv2 2024] | Apache-2.0 | HF ONNX → ort |
| **Captions (bulk)** | Florence-2-large ONNX | SmolVLM2-2.2B (ONNX, Apache) | <1 GB (q4f16 <500 MB) | sub-100 ms/img, **batchable** | COCO CIDEr 140.0, beats 80B Flamingo [CVPR 2024] | MIT | onnx-community 5-graph ONNX → ort (gen loop hand-rolled, §4) |
| **Captions (rich, subset)** | Qwen3-VL-4B Q4_K_M | PaliGemma2-3B-ft-docci-448 (densest paragraphs, Transformers); Moondream 2 | ~3.3 GB | few img/s | Best detailed captions ≤8 GB class [Qwen3-VL GGUF 2026] | Apache-2.0 | llama.cpp `mtmd`/Ollama sidecar (no ONNX yet mainstream; onnx-community Qwen3-4B-VL-ONNX exists but genai-format, no Rust binding) |
| **Faces** | buffalo_l (SCRFD-10G + ArcFace-R50, 512-D) | YuNet + SFace (only fully Apache pair) | <1 GB | thousands faces/s batched | Best hard-pose accuracy; digiKam ships YuNet+SFace [digiKam 8.6 2025-03] | code MIT / **weights NC** vs Apache-2.0 | ONNX → ort CUDA EP; DBSCAN cluster + FIQA gate |
| **Dedup / near-dup** | pHash tier-1 (dist<10) + SigLIP2 cosine tier-2 (~0.99) in usearch | imagededup-style CNN pass; SSCD | 0 (reuses existing) | CPU/RAM; sub-ms ANN queries | Two-tier is the literature consensus [MDPI 2025; arXiv 2312.07273] | MIT/Apache | `image-hasher` crate + usearch Rust crate |
| **Embeddings / search** | SigLIP2 ViT-B-16 (So400m if headroom) | DINOv2 ViT-S/B for pure-visual clustering (DINOv3 better but custom Meta license) | ~1 GB (B) / 2–3 GB (So400m) | hundreds img/s batched | Beats SigLIP/CLIP on retrieval+zero-shot, 109 langs [SigLIP2 2025-02] | Apache-2.0 | Immich HF ONNX → fastembed `try_new_from_user_defined` |
| **Video** | ffmpeg + NVDEC keyframes → image pipeline; TransNetV2 shots; Whisper large-v3-turbo INT8 speech | PySceneDetect (CPU, coarse cuts); TMK+PDQF video dedup | Whisper ~1.5 GB | NVDEC decodes in parallel with CUDA inference | Ampere decodes H.264/HEVC/VP9/AV1 [NVDEC 2026]; turbo ≈ large-v3 −0.39pp WER [2026] | MIT/Apache mix | `-hwaccel cuda`; TransNetV2/videohash = FFI (no native Rust) |
| **OCR (optional tier)** | RapidOCR (PP-OCR ONNX, ~80 MB) | Tesseract (already deployed) | <1 GB | ~120 pages/min-class GPU | PaddleOCR-class accuracy, curved 88.7% vs Tesseract 52.1% [Modal 2026] | Apache-2.0 | ONNX → ort |

---

## 3. The recommended stack on the RTX 3070 Ti (8 GB)

**Principle: sequential tier residency.** One heavy model loaded per phase; nothing needs co-residency except faces+embeddings (both tiny). All fps below are FP16, batch 8–32; public T4 numbers ×1.6–2 [Angle 3; Angle 5].

| Phase | Models resident | VRAM | Est. 100k-photo wall time |
|---|---|---|---|
| 0. Meta (CPU) | none — EXIF, pHash (`image-hasher`), reverse-geocode (`rrgeo`/`reverse_geocoder`) | 0 | I/O-bound; hours on NVMe, dominated by random reads |
| 1. Embed + faces | SigLIP2-B (~1 GB) + SCRFD/ArcFace or YuNet/SFace (<1 GB) | ~2–3 GB | **~1–3 h** (hundreds img/s; Immich anecdote: 300k CLIP-embedded overnight on a weaker iGPU) |
| 2. Detect + tag | RF-DETR-Nano (~1–2 GB), then RAM++ (~2–3 GB) | ≤3 GB each | **~1–2 h per pass** |
| 3. Dedup + bursts + aesthetics (CPU) | none — usearch HNSW (100k×512-D fp32 ≈ 0.2 GB RAM); EXIF-gap + pHash burst grouping; LAION aesthetic MLP on SigLIP2 vectors | 0 | minutes–tens of minutes |
| 4. Captions (gated subset) | Florence-2-large ONNX (<1 GB, batched) | ~1–6 GB w/ batch | Florence-2 batched: **hours for 100k**; contrast Moondream ~1 img/s ≈ 28 h — hence Florence-2 for bulk |
| 5. Rich captions (opt-in) | Qwen3-VL-4B Q4 via llama.cpp sidecar (~3.3 GB + ctx) | ~4–5 GB | only on user-flagged/top-aesthetic subset |
| 6. Video | NVDEC decode + phases 1–4 on keyframes; Whisper turbo INT8 (~1.5 GB) | ≤3 GB | scales with footage; NVDEC ≈ free alongside CUDA |

**CPU-side:** meta tier, ANN indexing, clustering (`petal-clustering` HDBSCAN / DBSCAN), burst logic, ffmpeg demux, DB writes. **Never bulk-run** open-vocab detection (10–50× slower) — on-demand only.
**Tier gating is mandatory at 500k scale:** captions after dedup+quality filtering, not before [gap-fill §H].
**Avoid on this box:** Immich ViT-H CLIP (~4.8 GB, quality overkill), Moondream 3 preview (18 GB fp16, BSL), Qwen3-VL-8B (6.1 GB, tight), FastVLM (Apple-runtime), Phi-4-mm (weak general-photo MMMU 54.3).

---

## 4. Integration with llm-indexing vision tiers

- **tags tier:** swap "CLIP" → SigLIP2 ONNX loaded through **fastembed `try_new_from_user_defined(...)`** (SigLIP2 is NOT in fastembed's built-ins; Immich's HF exports are directly consumable) [fastembed-rs v5.17.3, 2026-07-15]. Swap "YOLO" → RF-DETR/D-FINE ONNX via raw `ort` sessions. RAM++ needs a one-time PyTorch→ONNX export (BERT text branch is the friction point).
- **captions tier — the real engineering risk:** `onnxruntime-genai` has **no Rust binding** [microsoft/onnxruntime-genai 2026], so the autoregressive loop must be hand-rolled. Florence-2's ONNX is **5 graphs** (vision encoder, embed_tokens, encoder, decoder, decoder_merged); captions are bounded (~256 tok) so a fixed pre-allocated KV-cache loop is tractable; `rust-bert`'s generic seq2seq-ONNX pipeline is the closest reusable pattern. Fallback that avoids all of it: llama.cpp server sidecar (Qwen3-VL-4B / SmolVLM2 GGUF) called over local HTTP.
- **faces (new sub-tier):** detector → aligned crop → 512-D embed → incremental modified-DBSCAN (Immich's recipe: min-samples + max-distance knobs) with a **FIQA quality gate first** — digiKam 8.6's lesson: blurry/tiny faces poison clusters more than missed faces hurt.
- **dedup/bursts:** extend meta tier — usearch Hamming index over existing pHash; SigLIP2-cosine confirm pass; burst = sort by capture time → gap-segment (~1 min; 1–3 s for true bursts) → pHash/embedding confirm → pick sharpest (Laplacian variance or FIQA). No library does this turnkey (Immich has no burst grouping — open request #1885); build it.
- **ort EP notes:** enable the `cuda` cargo feature (prebuilt binaries downloadable for CUDA & TensorRT only; **DirectML requires a source build — skip it on this NVIDIA box**). Pin ort ≥ rc.12 (earlier RCs had TRT-EP breakage, issue #226). Match ort's expected CUDA 12.x + cuDNN 9 exactly; cuDNN 8↔9 are not cross-compatible. Add TensorRT EP later with `trt_engine_cache_enable` + `trt_timing_cache_enable` (first build minutes/model → ~seconds cached; cache invalidates on model/ORT/TRT/GPU change). `trt_fp16_enable` on; INT8 only for CNN detectors after A/B (ViT/CLIP INT8 is often no faster than FP16 on Ampere) [ORT TRT EP docs 2026; NVIDIA TRT issues #3147/#3352].

---

## 5. Requirements checklist

- [ ] **Driver:** latest NVIDIA **Studio** driver (covers CUDA 12.x runtimes via forward-compat; CUDA 13 toolkit unnecessary — Ampere gains nothing from it) [CUDA 13.3 notes 2026-05-27].
- [ ] **Runtime:** ONNX Runtime GPU build matching `ort` rc.12 (CUDA 12.x + **cuDNN 9**; both `bin` dirs on PATH; MSVC runtime installed).
- [ ] **Native Windows, not WSL2** — avoids the `/mnt/c` many-small-file penalty on a 100k+-file walk; everything here builds natively [Angle 5 §5].
- [ ] **RAM:** 32 GB comfortable (16 GB floor). **Disk:** library + index on **NVMe**; reserve tens of GB for model weights, TRT engine/timing caches, usearch index, DB.
- [ ] **Models to fetch (offline bundle):** SigLIP2-B ONNX (Immich HF), RF-DETR-Nano/Medium ONNX, RAM++ (export), Florence-2-large ONNX (onnx-community), face pack (buffalo_l or YuNet+SFace), Whisper large-v3-turbo INT8, optional Qwen3-VL-4B GGUF + mmproj, RapidOCR models.
- [ ] **Tools:** ffmpeg w/ NVDEC (`-hwaccel cuda`); llama.cpp (only if the sidecar path is used); PyTorch on a dev machine only for one-time exports (wheels bundle CUDA — no system toolkit needed).
- [ ] **Thermals:** undervolt (~0.85–0.90 V curve, MSI Afterburner) for multi-day batches — cuts power/heat 20–30% with minimal perf loss; investigate if sustained >85 °C.
- [ ] **Install order:** driver → ORT DLLs → verify `ort` CUDA EP smoke test → models → TRT EP + engine caches last.

---

## 6. Licensing cautions

| Item | Status | Consequence |
|---|---|---|
| **Ultralytics YOLO (v8/11/12/26, YOLO-World, YOLOE)** | AGPL-3.0 code **and weights**; vendor states even internal/private product use needs an Enterprise License | **Excluded from the stack.** Apache DETRs match it on COCO anyway [Ultralytics license 2026] |
| **InsightFace buffalo packs** | Code MIT; **weights non-commercial research only** | OK for owner's personal library; a landmine if llm-indexing ever ships commercially or touches FinFan work. Decide at engine level; clean fallback = YuNet+SFace |
| **EdgeFace / AdaFace weights** | CC BY-NC-SA / NC research | Not commercially clean despite permissive code — do not substitute for SFace |
| **DINOv3** | Custom Meta license (commercial-permitted, not OSI) | Use DINOv2 (Apache) unless terms reviewed |
| **ConvNeXt V2** | Some weights CC-BY-NC | Check per-variant before use |
| **RF-DETR** | Core Nano–Large Apache-2.0; **XL/2XL = PML-1.0** | Stay on core sizes |
| **Gemma 3/PaliGemma 2** | Gemma license (use restrictions) | Fine locally; note Gemma **4** is Apache-2.0 [google/gemma-4-E4B-it 2026] |
| **Moondream 3 preview** | BSL 1.1 (not open); 3.1 (2026-07-07) has a friendlier custom license | Prefer Moondream 2 (Apache) if used at all |
| **Immich / PhotoPrism (as reference code)** | AGPL-3.0 | Reference their recipes, don't vendor their code into a proprietary engine; LibrePhotos (MIT) is the liftable codebase |
| Fully clean core stack | SigLIP2, RF-DETR, D-FINE, OWLv2, RAM++, DINOv2, Florence-2 (MIT), SmolVLM2, Qwen3-VL, usearch, YuNet+SFace | **Zero copyleft / NC exposure end-to-end** |

---

## 7. Risks, unknowns, validate-first

**Validate first on the machine (in order):**
1. `ort` CUDA EP loads and runs SigLIP2 ONNX (the DLL/version-matching step is where Windows installs fail) — measure img/s at batch 16.
2. **Florence-2 5-graph generation loop in Rust** — prototype on 100 images before committing the captions tier to ONNX; if painful, fall back to llama.cpp sidecar immediately.
3. RF-DETR ONNX FP16 throughput + a TRT engine build (confirm cache persistence, time-to-first-build).
4. RAM++ ONNX export actually works (BERT branch); else zero-shot SigLIP2 tags carry the tier.
5. INT8 vs FP16 A/B on the detector only; keep all ViT-family models FP16.
6. Small VN/DE query-recall eval on SigLIP2 — **Vietnamese is a documented low-resource laggard** (German is fine) [LearnOpenCV SigLIP2 2026; uCLIP 2025-11].

**Known unknowns / gaps:**
- No measured 3070 Ti VLM benchmarks anywhere — all throughput figures extrapolated from 3090/3060/T4; the 100k-caption wall-time estimate has ±2× error bars.
- `ort` is still pre-stable (rc.12); API churn risk. ORT upstream is at ~1.23 (CUDA 13/TRT 10.14) while ort tracks the 1.22-era matrix — version-pin everything.
- Native-Rust **RAW decode** unresolved (libraw FFI needed); pure-Rust `heic` or `libheif-rs` covers HEIC.
- TransNetV2 (shot detection) and TMK+PDQF/videohash (video dedup) have no ONNX/Rust path — FFI or defer; PySceneDetect-style histogram cuts are the cheap interim.
- Immich's OCR VRAM leak (#26739) is only *partially* fixed in v2.6.0 — a caution for OCR-tier memory behavior generally; prefer RapidOCR under your own session management.
- Caption hallucination rates (POPE/CHAIR) for Florence-2/SmolVLM2/Qwen3-VL at this scale: unevaluated — treat captions as searchable hints, not ground truth.
- Cloud baseline for reference only: AWS Rekognition ≈ $100/100k labels, Google Vision ≈ $150/100k — cheap once, but recurring, per-feature-billed, and violates the offline constraint [AWS/GCP pricing 2026].

---

## 8. Sources (deduped, dated)

**Runtimes / Windows:** ONNX Runtime CUDA EP, TensorRT EP (matrix + cache timings), DirectML EP (sustained-engineering note), quantization docs — onnxruntime.ai, 2026 · ort 2.0.0-rc.12, docs.rs/crate/ort + pykeio/ort releases & issue #226, 2026-03 · microsoft/onnxruntime-genai (no Rust binding), github, 2026 · Anush008/fastembed-rs v5.17.3 (`try_new_from_user_defined`), github, 2026-07-15 · CUDA 13.3 release notes + CUDA compatibility r595, docs.nvidia.com, 2026-03/05 · onnxruntime releases (1.23.x) + Triton release notes, 2026 · InsiderLLM WSL2 guide + TechTimes WSL3, 2026 · NVIDIA TensorRT issues #3147/#3352, ModelOpt #167 · pytorch.org Get Started, 2026.
**Photo-manager stacks:** docs.immich.app (ml-hardware-acceleration, searching, facial-recognition, better-facial-clusters), 2025–26 · Immich discussions 11862/8497/17135/27047, issues 26739/23462, PR 27027, 2024–26 · huggingface.co/immich-app (CLIP/SigLIP2 ONNX, buffalo packs), 2025–26 · t0saki.com Immich GPU acceleration, 2025 · digikam.org 8.5/8.6/8.7 release notes, 2024–25 + phoronix digiKam 8.7, 2025 · docs.photoprism.app face-recognition + AGPL, 2025–26 · docs.librephotos.com (intro, face-recognition, captioning), 2025 · Damselfly face-recognition-in-net, 2024.
**VLMs:** Florence-2 CVPR 2024 + onnx-community/Florence-2-base/-large-ft, HF · SmolVLM2 blog + arXiv 2504.05299, 2025 · Qwen3-VL-4B/8B GGUF + onnx-community Qwen3-4B-VL-ONNX, HF, 2026 · moondream2 + moondream-2b-2025-04-14 + moondream3-preview LICENSE + moondream3.1-9B-A2B, HF, 2025–26 · Photon 1.2.0, moondream.ai, 2025–26 · paligemma2 blog + paligemma2-3b-ft-docci-448, HF · google/gemma-4-E4B-it + ai.google.dev/gemma/docs/releases, 2026 · ggml-org/gemma-3-4b-it-GGUF + unsloth gemma-3n/gemma-4 GGUF, 2026 · InternVL3 arXiv 2504.10479, 2025-04 · Phi-4-multimodal-instruct HF + arXiv 2503.01743, 2025 · apple/ml-fastvlm, 2025 · Roboflow moondream-2 benchmark + MNeMoNiCuZ/florence2-caption-batch, 2026.
**Detectors / taggers / embeddings:** Ultralytics YOLO26 docs + arXiv 2509.25164 + 2510.09653, 2025 · ultralytics.com/license + issue #19390 + Imagimob license note, 2026 · RF-DETR blog + arXiv 2511.09554 + github LICENSE, 2025–26 · D-FINE arXiv 2410.13842 (ICLR 2025) · RT-DETRv4 arXiv 2510.25257 + github, 2025–26 · DEIM CVPR 2025 · YOLO-World CVPR 2024 · Grounding DINO 1.5 arXiv 2405.10300, 2024 · RAM++ HF card + arXiv 2310.15200 + recognize-anything github · SmilingWolf WD v3 HF · SigLIP2 (EmergentMind, timm HF, LearnOpenCV review), 2025–26 · DINOv3 arXiv 2508.10104 + Meta blog, 2025-08 · InternImage arXiv 2211.05778 · CSAILVision/places365 · uCLIP arXiv 2511.13036, 2025-11.
**Faces / dedup:** deepinsight/insightface README + insightface.ai licensing, 2026 · opencv/opencv_zoo (YuNet/SFace), 2026 · serengil/retinaface + yakhyo/retinaface-pytorch (MIT), 2026 · Idiap/EdgeFace-S-GAMMA HF + mk-minchul/AdaFace, 2026 · idealo/imagededup + imagehash + benhoyt duplicate-image-detection · MDPI Electronics 15(7):1493, 2025 · arXiv 2312.07273, 2024 · unum-cloud/usearch + crate, 2026 · faiss crate + arXiv 2401.08281, 2024 · petal-clustering + hdbscan crate, 2026 · LAION-Aesthetics audit arXiv 2601.09896, 2026-01 · USPTO 10,430,456 / 11,443,469 · Immich discussion #1885.
**Video / OCR / misc:** NVDEC (Wikipedia) + NVIDIA Video Codec SDK 13.0 ffmpeg guide, 2026 · soCzech/TransNetV2 + Frontiers shot-boundary survey, 2026 · facebook/ThreatExchange (PDQ/TMK) + akamhy/videohash, 2026 · runaihome Whisper self-host + spokenly Whisper sizes, 2026 · Modal 8-OCR-models + imagetotable OCR, 2026 · libheif-rs + heic crate, 2026 · reverse_geocoder + rrgeo crates, 2026 · AWS Rekognition pricing + Google Cloud Vision pricing, 2026 · SilverPC 3070 Ti metrics, 2025-10 + undervolting guides.
