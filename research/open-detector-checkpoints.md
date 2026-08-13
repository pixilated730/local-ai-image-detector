# Open-Source AI-Generated Image Detectors: Candidates for the Extension

## Key context from independent evaluations

- **"How well are open sourced AI-generated image detection models out-of-the-box" (arXiv 2602.07814, Feb 2026)** — zero-shot benchmark of 23 pretrained checkpoints across 16 methods, 12 datasets, 2.6M images, 291 generators. Best mean accuracy: **Community Forensics 75.0%**, then SAFE 68.8% (unstable, 3.2–99.8% per dataset), PatchCraft 67.5%. Worst: CNNSpot 37.5%, UnivFD 40.7%. On modern commercial generators everything craters: Flux Dev 21%, Firefly 18%, MJ v7 24%, Imagen 4 19%. Training-data diversity mattered more than architecture.
- **Chameleon / AIDE (ICLR 2025, arXiv 2406.19435)** — on human-Turing-test-passing fakes, every detector scores 50–64% (AIDE best 63.9%); most keep >90% real-accuracy but drop to <25% fake-accuracy. Self-reported 98%+ numbers are meaningless for OOD.
- Calibration on a small dev set recovers large amounts of accuracy (arXiv 2602.01973) — raw thresholds are the biggest single source of lost balanced accuracy.

## Candidate table (condensed)

| Model | Arch / input | Size | License | Independent OOD evidence | Browser portability |
|---|---|---|---|---|---|
| **Community Forensics** (HF: OwensLab/commfor-model-224 / -384) | ViT-S/16, 224/384px; trained on 2.7M imgs / 4,803 generators | ~22M params (~85MB fp32, ~22MB int8) | Code MIT; dataset CC-BY-NC-SA; **verify weight license** | **Best: 75.0% mean, most stable** | Excellent — plain ViT, trivial ONNX |
| **SAFE** (KDD 2025) | Small CNN + transform preprocessing, 256px | tens of MB | check repo | 68.8% mean but unstable (3.2–99.8%) | Good — simple ops; preprocessing in JS |
| **Organika/sdxl-detector** | Swin-base 224px | 87M (~85MB q8) | **CC-BY-NC-3.0** | none formal; weak on non-SDXL | Excellent — ONNX branch exists; transformers.js-ready |
| **HPAI-BSC/SuSy** | ResNet-18+MLP, patch-based | ~12M | open (check card) | paper only | Very good |
| **umm-maybe/AI-image-detector** | Swin, art-focused, 2022 | ~87M | CC-BY-ND | stale, predates SDXL | Obsolete |
| **dima806 / NYUAD ViT detectors** | ViT-base | 86M | Apache-2.0 | none; narrow data | Excellent tech, low expected OOD |
| **UnivFD (Ojha)** | CLIP ViT-L/14 + linear probe | ~300M | released | **40.7% mean**; ~57% Chameleon | Heavy |
| **CNNSpot** | ResNet-50 ProGAN-era | 25M | released | **37.5% (near-random)** | Portable but useless today |
| **NPR** | ResNet-50 on pixel residuals | 25M | released | ~57% Chameleon | Portable, weak OOD |
| **DIRE** | Diffusion reconstruction error | needs diffusion model | released | weak Chameleon | **Not portable** (diffusion inversion) |
| **AIDE** | DCT patch expert + CLIP-ConvNeXt | large | **CC-BY-NC-SA** | best Chameleon (63.9%), mid-pack elsewhere | Hard; NC license |
| **C2P-CLIP / FatFormer / RINE** | CLIP-L based | ~300M | mixed (FatFormer Apache-2.0) | no strong third-party OOD | Heavy; RINE needs custom export |
| **DRCT** | ConvNeXt-B / CLIP-ViT | 89–300M | released | mid-tier | ConvNeXt variant portable |

## Existing browser-based detectors

- **dejAIvu** (GPL-3.0) — open-source extension running ONNX locally with saliency overlays; art-focused undocumented model. Architectural reference only.
- **haywoodsloan/ai-image-detector** (Apache-2.0) — extension + training stack but server-side inference, 0.2B SwinV2.
- **BitMind extension** — commercial, server-side.
- **No existing extension ships a well-validated local detector — the niche is real.**

## Portability notes

- Plain ViT/Swin/ResNet/ConvNeXt: directly supported by Optimum ONNX export and transformers.js; WASM today, WebGPU via ORT-Web 1.17+.
- CLIP-L detectors: ~150MB int8, marginal. DIRE impossible; AIDE impractical.
- int8 quantization costs ~1-2 points; ViT-S int8 ≈ 22MB — comfortably shippable.

## Recommendation: top 3 to baseline

1. **Community Forensics ViT-S** — independently ~75% mean zero-shot (exactly the target), 4,803-generator training diversity, tiny, MIT code, trivial export. **Action: confirm HF weight license.**
2. **SAFE** — second-best independent mean; unstable solo but a strong ensemble partner. **CommFor (semantic ViT) + SAFE (low-level CNN) with calibrated fusion is the most promising route past 75%.**
3. **Organika/sdxl-detector** — fastest bring-up demo (ONNX exists) but NC license and no OOD validation; not the shipping model.

Cross-cutting: calibrate the decision threshold on our own web-realistic dev set; expect JPEG/resize to hurt artifact-based detectors more than semantic ones.
