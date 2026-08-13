# AI-Generated Image Detection: Datasets, Benchmarks, and Recipe

## 1. Training datasets

| Dataset | Size | Generators covered | Real sources | Availability / link |
|---|---|---|---|---|
| **GenImage** (NeurIPS 2023) | 2.68M (1.33M real / 1.35M fake) | Midjourney (v5), SD v1.4/1.5, Wukong, VQDM, ADM, GLIDE, BigGAN — 8 generators, no post-2023 models | ImageNet-1k | Public: https://github.com/GenImage-Dataset/GenImage , https://www.unbiased-genimage.org . **Caution: severe JPEG/size bias (see §3/§6)** |
| **Community Forensics** (CVPR 2025) | 2.7M fake images from **4,803 generator models** (systematically pulled from Hugging Face + commercial: DALL·E 3, MJ, Firefly, etc.) | Paired real from LAION, ImageNet, COCO, FFHQ etc. | HF: `OwensLab/CommunityForensics` (also a small subset version); https://github.com/JeongsooP/Community-Forensics . **Best single "generator diversity" training set** — key finding: generalization scales with number of generators, not images |
| **WildFake** (AAAI 2025) | ~3.5M fake / ~1M real, hierarchical (GANs, DMs, others) | Fakes scraped from Civitai, Midjourney community, etc. — real user prompts, real-world styles | Mixed web real images | arXiv 2402.11843; released via HF (search "WildFake"). Trained detectors show better degradation robustness |
| **ELSA D3** | ~2.3M records / 11.5M images (1 real + 4 fakes per record) | SD 1.4, SD 2.1, SDXL, DeepFloyd IF | LAION-400M (real image + prompt) | HF: `elsaEU/ELSA_D3`, streamable. Deliberately varied aspect ratios and encodings — less format bias than GenImage |
| **OpenFake** (2025) | ~4M total: 3M real + ~1M synthetic | SDXL, **Flux**, Ideogram, Imagen 3, DALL·E, **GPT Image 1** | Politically/news-focused real images from LAION-400M captions + 2025 social media | HF: `ComplexDataLab/OpenFake` (arXiv 2509.09495). **One of very few sets with GPT-4o-class fakes** |
| **DiffusionDB** | 14M images (fake only) | SD 1.x only, real user prompts | none (pair with own reals) | HF: `poloclub/diffusiondb`, CC0. Dated |
| **ArtiFact** | 2.5M (965k real / 1.53M fake) | 25 methods: 13 GANs, 7 diffusion, 5 other | COCO, FFHQ, LSUN, ImageNet, CelebA-HQ, Landscape | Kaggle: `awsaf49/artifact-dataset` (arXiv 2302.11970). Social-platform impairments applied |
| **CIFAKE** | 120k @ 32×32 | SD v1.4 only | CIFAR-10 | Skip — tiny resolution, one old generator |
| **TWIGMA** | ~800k AI images from Twitter (2021–2023) | MJ/SD/DALL·E-2 era, weakly labeled by hashtag | none | Zenodo 8031785. In-the-wild fake augmentation; noisy labels |
| **Kaggle "Detect AI vs Human-Generated Images"** (2025) | ~80k pairs | Modern gen, Shutterstock-paired | Shutterstock licensed photos | kaggle.com/competitions/detect-ai-vs-human-generated-images . Clean paired real/fake |
| **AI-GenBench** (2025) | 360k (180k/180k) | **36 generators ordered by release date** (GANs → SD3/Flux era) | ImageNet, COCO, LAION-400M, RAISE | https://github.com/MI-BioLab/AI-GenBench . Temporal train/eval splits |
| **GenImage++** (2025) | test-only extension | **Flux.1, SD3**, long prompts | — | Use as held-out eval only |

## 2. Hard/realistic evaluation benchmarks + SOTA

| Benchmark | What it is | SOTA balanced accuracy |
|---|---|---|
| **Chameleon** (AIDE paper, ICLR 2025) | 26,033 images (14,863 real Unsplash / 11,170 fake) that **passed a human "perception Turing test"** — MJ, DALL·E 3, SD+LoRA; 720p–4K | **Nearly all off-the-shelf detectors ≤ random (~50%)**; AIDE (best in paper): **63.9%**; PatchCraft ~55.7%; 2025-26 followups reach mid-to-high 60s. Best "hard mode" proxy. arXiv 2406.19435 |
| **WildRF** (LaDeDa) | 5,300 images scraped from Reddit/X/Facebook — native platform compression/edits | LaDeDa: **93.7% mAP** (vs ~99% on academic benchmarks); academic-trained detectors do much worse. arXiv 2406.09398 |
| **Community Forensics eval** | held-out unseen-generator eval | Their classifier generalizes well; public checkpoints usable as baseline/teacher |
| **AI-GenBench "next-period"** | train before time T, test after | AUROC drops notably on next-window generators |
| **AIGIBench** (arXiv 2505.12335) | multi-degradation, multi-generator audit | Real-image accuracy collapses under social-media degradations; CNNDetection fake-recall: 99% clean → **0.8%** under JPEG |
| **VFM baselines** (arXiv 2509.12995) | in-the-wild eval | Fine-tuned vision foundation models (DINOv2/SigLIP) **beat specialized forensic detectors** in the wild |

Takeaway: ~99% on GenImage-style benchmarks → ~50–65% on Chameleon. The 75% target is realistic only with degradation-robust training; NOT achieved by off-the-shelf detectors.

## 3. Degradation and augmentation (proven)

- **CNNSpot canon**: train with random JPEG (QF 30–100, p≈0.5) + Gaussian blur (σ 0–3, p≈0.5). Without it, fake recall → near 0 under compression.
- **GenImage degraded track**: JPEG QF=65 / blur / downsampling each cut cross-generator accuracy 10–30 points for non-augmented detectors.
- **"Fake or JPEG?" (arXiv 2403.17608)**: GenImage reals are JPEGs, fakes fixed-size PNGs — detectors learn the *format*. Equalizing treatment improved cross-generator accuracy by >11 points. Use unbiased-genimage.org variant.
- **DCPT (2026)**: clean/degraded consistency loss; +15.7–17.9 pts under JPEG, +3.8–10.3 under resize.
- Practical: re-encode JPEG (QF 30–95), WebP, downscale-then-upscale (0.25×–1×), random crop, slight blur/noise — **identical pipeline for reals and fakes**.

## 4. Real-image sources & distribution-mismatch pitfall

- Sources: COCO, Open Images, LAION (pre-2022, date-filtered — post-2022 contains AI images), ImageNet, RAISE, Unsplash, Flickr date-filtered, news photo sets (VisualNews).
- **Pitfall**: detectors cheat on any real/fake asymmetry — JPEG vs PNG, fixed sizes, resolution, content category. B-Free (arXiv 2412.17671) generates fakes as reconstructions of the same real images to match content.
- Rule: **match format, size distribution, and content topic between classes**; send both through the same re-encode pipeline; prompt-pair fakes to real captions.

## 5. Non-learned signals

- **C2PA Content Credentials**: DALL·E 3 / GPT-4o (since Feb 2024), Adobe Firefly, Bing. IPTC `digitalSourceType = trainedAlgorithmicMedia`. Does NOT survive screenshots/re-encoding/most platform uploads (X strips). High-precision/low-recall: presence ⇒ almost certainly AI; absence ⇒ no info.
- **SD invisible watermark** (`invisible-watermark`, DWT-DCT, 48-bit): default in reference pipelines, but A1111/ComfyUI mostly disable it. Cheap to decode; require exact payload match (random images decode to random bits). Weak auxiliary only.
- **SynthID**: not publicly detectable by third parties (portal + trusted-partner API only). Cannot rely on it.
- Recipe: cheap metadata/watermark pre-pass → positive ⇒ high-confidence AI; otherwise learned model carries the decision. Expect rare fires on a re-encoded private benchmark.

## 6. Building an honest proxy eval

1. **Held-out generators, not held-out images** (temporal split: train pre-T, test post-T).
2. **In-the-wild scrapes with platform-native processing** (WildRF/TWIGMA/Chameleon pattern).
3. **Human-filtered hard positives** (Chameleon's "both annotators fooled").
4. **Bias audit**: train a trivial model on (JPEG quality, width, height) only; if it beats chance, the eval is broken.
5. **Degradation sweeps**: report accuracy at JPEG QF ∈ {95, 75, 50} and 512px downscale.

## 7. Recommended recipe

**(a) Training set (~1–2M images, balanced):**
- Fakes: Community Forensics (diversity backbone) + OpenFake synthetic split (Flux/Ideogram/Imagen/GPT-Image-1) + WildFake slice + optionally self-generated Flux.1-dev/SD3.5/SDXL from COCO/LAION captions (B-Free-style pairing).
- Reals: COCO + Open Images + LAION (pre-2022) + Unsplash/RAISE + news photos, content-matched.
- Unified post-processing both classes: random JPEG QF 30–95 (p=0.7), WebP, resize 256–1024px, crop, slight blur/noise; strip metadata. Consider DCPT consistency loss.
- Model: fine-tune a **vision foundation model (DINOv2 / SigLIP / CLIP-L)** with shallow head — beats bespoke forensic architectures in-the-wild (arXiv 2509.12995); AIDE-style hybrid is the alternative.

**(b) Proxy benchmark (never trained on):**
- **Chameleon** (primary; ~70% balanced accuracy there ⇒ 75% on private web-realistic benchmark plausible)
- **WildRF test split**
- **GenImage++** and/or self-collected 2025-era fakes (GPT-4o from r/ChatGPT, MJ v6 showcase, Flux from Civitai) + reals from same platforms, re-saved JPEG QF~85 ≤1024px
- Report balanced accuracy per generator and per degradation level; run the format-bias audit first.
