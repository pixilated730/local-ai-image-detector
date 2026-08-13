# Extension (Manifest V3)

Status: **skeleton** — full pipeline wired end-to-end with a stub engine; the real model
plugs into `src/engine.js` once the research phase picks a checkpoint.

## Architecture

```
content.js (per page)                background.js (service worker)      offscreen.js
─────────────────────                ──────────────────────────────      ─────────────
find <img> (Intersection/            route ANALYZE_IMAGE                 fetch image
MutationObserver, size filter)  ──▶  cache results by URL           ──▶  decode (createImageBitmap)
overlay confidence badge        ◀──  ensure offscreen doc exists    ◀──  preprocess → engine.classify()
(shadow DOM, tracks position)                                            ONNX Runtime Web (WebGPU/WASM)
```

Why an offscreen document for inference:
- Full DOM APIs (OffscreenCanvas, createImageBitmap) for decoding
- Long-lived (service workers get killed); stable home for a loaded ONNX session
- Fetching images from the extension context uses `<all_urls>` host permission, so
  cross-origin images decode cleanly (no canvas tainting)

## Build & load for development

```
npm install
node build.mjs                                    # vendors onnxruntime-web into lib/ort/
python ../eval/export_onnx.py --repo OwensLab/commfor-model-384 --input-size 384
cp ../eval/models/commfor_vits_384.onnx models/detector.onnx
```

Then load it unpacked — see [../INSTALL.md](../INSTALL.md) for the click-by-click
(Brave and Chrome are identical here).

Model: **Community Forensics ViT-S/16 @384** (MIT weights, `OwensLab/commfor-model-384`),
fp32 ONNX (~87MB), calibration offset +7.6897 baked into `engine.js`.
Engine tries WebGPU, falls back to WASM.

## Tests

```
python test/serve.py 8124 --dir .     # dev server with correct .mjs/.wasm MIME types
```

- `test/parity.html` — browser preprocessing vs `eval/harness.py` on identical files
- `test/demo.html` — a feed page driving the real content script + engine via a
  `chrome.*` shim; `?scroll=9000` starts mid-feed to check viewport priority

## Remaining TODO
- [ ] Set `CALIBRATION_OFFSET` in engine.js from proxy-benchmark calibration
- [ ] One-time weights download + Cache API/OPFS persistence (instead of bundling)
- [ ] Verify browser preprocessing parity vs eval/harness.py on the same images
- [ ] blob:/data: URL images (decode via content-script canvas where untainted)
- [ ] Optional metadata/watermark auxiliary signals fused into the score
- [ ] int8/fp16 model variant for faster WASM (dynamic int8 was lossy — needs static quant)
