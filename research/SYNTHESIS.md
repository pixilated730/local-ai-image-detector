# Research Synthesis — Decisions & Plan

*Distilled from the four research reports in this directory (2026-08-13).*

## The headline findings

1. **The target is reachable.** An independent Feb 2026 benchmark (arXiv 2602.07814, 23 checkpoints, 2.6M images, 291 generators) measured the **Community Forensics ViT-S** checkpoint at **75.0% mean zero-shot accuracy** — exactly our bar — and it was the most stable across datasets. It's ~22M params (~22MB int8), plain ViT, trivially ONNX-exportable. This is our baseline.
2. **Data diversity beats architecture.** Generalization scales with the number of distinct generators in training (CommFor used 4,803), not method cleverness. Fancy architectures (AIDE, DIRE, LGrad) are either browser-infeasible or add only a few points.
3. **Degradation robustness is the battleground.** Web images are JPEG-recompressed and resized; artifact-based detectors collapse under this (fake recall 99%→0.8% under JPEG in one audit). Training with aggressive re-encode/resize augmentation applied identically to both classes is non-negotiable.
4. **Everything fails one-sided on hard fakes** (predicts "real"). Threshold **calibration on a degraded dev set** is the cheapest large win — the eval uses a fixed 0.65 cutoff, and a monotonic score remap is legitimate.
5. **The browser part is solved.** transformers.js v3 / onnxruntime-web ≥1.19, WebGPU in service workers since Chrome 124, offscreen document as robust fallback, Cache API for one-time weight download, extension-context fetch beats CORS. Working example repos exist.
6. **Nobody has shipped this yet.** Existing extensions are either server-backed or unvalidated — the niche is real.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Baseline model | **Community Forensics ViT-S** (OwensLab/commfor-model-224 or -384) | Only checkpoint independently at ~75%; tiny; MIT code (⚠ verify weight license on HF) |
| Runtime | transformers.js v3 / onnxruntime-web, **WebGPU → WASM fallback** | HF official pattern; ViT-S stays usable even on WASM |
| Inference host | **Offscreen document** (current skeleton) — long-lived, threads + WebGPU always available | SW-hosting is viable (Chrome ≥124) but offscreen avoids SW-lifetime + single-thread constraints |
| Weights | One-time download → Cache API/OPFS, `unlimitedStorage` | Bounty allows initial download, requires offline after |
| Score head | **Calibrated remap** of model logit so optimal balanced-accuracy operating point lands at 0.65 | Biggest cheap win per research |
| Aux signals | Metadata pre-pass (C2PA, IPTC trainedAlgorithmicMedia, XMP strings; maybe SD watermark) — presence ⇒ high-confidence AI | High precision, ~zero cost; rarely fires on re-encoded benchmarks but helps in real browsing |
| If baseline < bar | Fine-tune ViT-S/B from CommFor checkpoint with degradation augmentation + fresh 2024-25 fakes (OpenFake: Flux/Ideogram/Imagen/GPT-Image-1); optional SAFE-style small-CNN second branch, logit-fused | The proven recipe; hybrid adds a few points |

## Proxy evaluation (never train on it)

- **Chameleon** (primary hard set — ~70% there suggests 75% on a web-realistic private set)
- **WildRF** test split (platform-native degradation)
- Self-collected 2025-era fakes + reals, re-saved JPEG q~85 ≤1024px
- Report balanced accuracy per generator and per degradation level (JPEG q95/75/50, 512px downscale)
- **Bias audit first**: a classifier on (format, width, height) alone must not beat chance

## Phased plan

1. **Baseline (now)**: export CommFor ViT-S to ONNX → wire into the skeleton's `engine.js` → build eval harness → measure on proxy benchmark → calibrate to 0.65.
2. **If ≥ ~72% on proxy**: polish, package, submit — calibration may carry us over.
3. **If short**: fine-tune (Community Forensics data + OpenFake + degradation aug) — this is a GPU training project (not browser-side; bounty allows offline training, only inference must be in-browser).
4. **Ship**: reproducible build, install docs, MIT license, one-time weight download flow.

## Key risks

- CommFor weight license may not be MIT-compatible (dataset is CC-BY-NC-SA) — **verify before shipping; fine-tuning our own weights on permissive data is the fallback.**
- Private benchmark may skew harder (Chameleon-like) than "web-realistic" suggests → proxy eval must include Chameleon.
- Overfitting our proxy: keep one untouched holdout split for the final go/no-go measurement.
