# Generalizable AI-Image Detection (2023–2025): Method Survey

## Method comparison (condensed)

| Method | Approach | Backbone | Chameleon (hard set) | Browser feasibility |
|---|---|---|---|---|
| **UnivFD** (CVPR'23) | Linear probe on frozen CLIP | CLIP ViT-L/14 (~304M) | 57.2% (fake recall 3%) | Feasible w/ int8 (~310MB) |
| **CLIPping the Deception** | Prompt tuning of CLIP | CLIP ViT-L/14 | n/r | Same |
| **RINE** (ECCV'24) | Intermediate CLIP CLS tokens + head | CLIP ViT-L/14 | mediocre | Custom ONNX export |
| **FatFormer** (CVPR'24) | Forgery-aware adapters on CLIP | CLIP ViT-L + adapters | weak | Feasible, heavy |
| **C2P-CLIP** (AAAI'25) | Category prompt injection | CLIP ViT-L/14 | degrades hard (63% under degradation vs 80% robust) | Feasible |
| **Effort** (ICML'25) | SVD orthogonal-subspace fine-tune, 0.19M trainable | CLIP ViT-L/14 | best-in-class among CLIP methods (community) | **Good** — merges into vanilla ViT forward |
| **NPR** (CVPR'24) | Neighboring-pixel residual → CNN | ~ResNet-50 | 57.3% (fake recall 5%) | **Excellent** size; brittle to JPEG/resize |
| **SAFE** (KDD'25) | Native-res crops + patch masking, small CNN | ~1–5M CNN | moderate | **Excellent**; weakens on downscaled images |
| **PatchCraft/SSP** | Texture patch contrast | small CNN | 53.8% — collapses | Poor hard-set OOD |
| **LGrad** | GAN-discriminator gradients | — | fake recall ~3% | **Infeasible** (test-time gradients) |
| **DIRE** | Diffusion inversion error | ADM + ResNet | 58.2% | **Infeasible** (diffusion inversion) |
| **DRCT** (ICML'24) | Reconstruction hard-positives in training only | ConvNeXt-B / CLIP-ViT | n/r | **Feasible at inference** |
| **AIDE** (ICLR'25) | DCT patch experts + CLIP semantic branch | OpenCLIP + 2×ResNet-50 | **58.4→65.8% (best published)** | Borderline (~300–400MB, fiddly); NC license |
| **Community Forensics** (CVPR'25) | Nothing clever — 4,803 generators, 2.7M imgs, e2e fine-tune | **ViT-S (CLIP-pretrained)** 224/384 | among best in independent evals | **Best fit**: ~22M params, HF weights, trivial ONNX |
| **B-Free** (CVPR'25) | Fakes = SD reconstructions of the reals; content aug | DINOv2 ViT-L, 504px crops | strong in-the-wild | Heavy but pure forward pass |
| **AIGI-Holmes** | LVLM + visual expert | 7B+ | good | **Infeasible** |

## Key lessons

1. **Training-data diversity beats method cleverness — by a lot.** Accuracy scales with number of distinct generators in training (diminishing returns past ~1,000). A plain fine-tuned ViT-S on diverse data beats nearly every specialized architecture. All ProGAN-trained methods collapse to near-chance on Chameleon.
2. **Dataset bias is the silent killer (JPEG/resolution).** "Fake or JPEG?": detectors learn "PNG=fake, JPEG=real"; fixing it = +11 points. Must train with aggressive JPEG/WebP recompression, rescaling, blur, noise — identically on both classes.
3. **Low-level artifact methods are brittle to web transformations.** Downscaling + recompression (what CDNs do) destroys their signal. SAFE/B-Free mitigate with native-resolution crops. Semantic (CLIP/DINOv2) features degrade far more gracefully.
4. **The hard-set failure mode is one-sided: everything predicts "real"** (fake recall 0–14% on Chameleon). Threshold calibration on degraded validation data is essential; so are hard positives in training.
5. **Hybrid semantic + low-level fusion helps, but only on top of good data.** AIDE's fusion adds a few points; diverse data adds tens.
6. **GPT-4o-style autoregressive generation is not a special problem yet** — its decoder appears diffusion-based; detectors transfer (SAFE: 98.9% on GPT-4o outputs).

## Browser feasibility

- Infeasible: DIRE (diffusion inversion), LGrad (test-time gradients), LVLM detectors, any test-time optimization.
- Fully feasible: pure forward passes — ViT-S/B, ResNet/ConvNeXt, CLIP+linear/adapter (Effort merges ΔW → vanilla forward).
- Sizes: ViT-S ≈ 22M (45MB fp16), ViT-B ≈ 86M (~175MB fp16), ViT-L ≈ 304M (~310MB int8 = practical ceiling).

## Recommended approach

1. **Backbone**: CLIP-pretrained ViT-B/16 (or ViT-S for latency), starting from `OwensLab/CommunityForensics` checkpoint. Fine-tune end-to-end (frozen probes lose several points). Optionally Effort-style orthogonal fine-tuning (free at inference).
2. **Data**: Community Forensics 2.7M base + DRCT-2M + B-Free-style SD reconstructions of reals + fresh 2024–25 commercial samples (Flux, MJ v6, SD3.5, GPT-4o, Imagen 3). Reals: LAION/COCO + genuinely web-degraded photos.
3. **Augmentation (non-negotiable)**: random JPEG/WebP (q30–95), downscale 0.25–1.0× mixed interpolation, blur/noise, random crop — both classes identically.
4. **Inference**: multi-crop — whole-image 224 semantic view + 2–4 native-resolution 224² crops for artifact signal; average logits. Calibrate threshold on degraded validation set.
5. **Optional hybrid branch**: ~5MB SAFE/NPR-style CNN on native-res crops, late-fused (logit average); ship as separate small ONNX, disable for heavily downscaled inputs.

**Expected outcome**: ≥90% balanced accuracy on standard cross-generator suites, ~70–80% on Chameleon-grade fakes. The ≥75% target is realistic on web-realistic mixes — through data diversity + degradation training, not architecture tricks.
