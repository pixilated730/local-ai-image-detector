# In-Browser Inference in a Chrome MV3 Extension — Technical Report

## TL;DR

Use **transformers.js v3 (`@huggingface/transformers`)** (wraps onnxruntime-web) with `device: "webgpu"` and WASM fallback. Inference can live in the **background service worker** (Chrome ≥124 exposes WebGPU to SWs) with an **offscreen document** as fallback — or in the offscreen document as the single robust tier (threads + WebGPU always available there). Weights auto-cache via the Cache API under the `chrome-extension://` origin → one-time download, offline forever after. This is HF's official extension pattern.

## Execution contexts

| Context | WASM | WebGPU | Notes |
|---|---|---|---|
| Service worker | Yes (numThreads=1) | **Yes since Chrome 124** | No nested workers → single-thread WASM; no dynamic import (need ORT ≥1.19 statically bundled) |
| Offscreen document | Yes (threads OK) | Yes | Most robust; full DOM; one per extension; debug via chrome://inspect/#other |
| Content script | avoid | avoid | Per-tab model copies, page interference |

## Required manifest bits

```json
"permissions": ["offscreen", "storage", "unlimitedStorage"],
"host_permissions": ["<all_urls>"],
"content_security_policy": {
  "extension_pages": "script-src 'self' 'wasm-unsafe-eval'; object-src 'self';"
},
"minimum_chrome_version": "124"
```

- `wasm-unsafe-eval` is mandatory for WASM compile; no CDN scripts allowed — bundle everything.
- `unlimitedStorage` prevents eviction of multi-hundred-MB cached weights.

## ORT/transformers.js config

```js
env.backends.onnx.wasm.numThreads = 1;   // service worker only; threads OK in offscreen
env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL("wasm/"); // bundled, no CDN
// pipeline("image-classification", model, { device: "webgpu", dtype: "fp16" })
// fallback: { device: "wasm", dtype: "q8" }
```

- Weight caching: default Cache API under extension origin (one-time download, offline after). Alternatives: custom cache (IndexedDB/OPFS) or ship weights in the CRX (`env.allowRemoteModels=false; env.localModelPath=...`).
- First WebGPU inference compiles shaders (1–10s) — warm up with a dummy tensor; keep input resolution fixed.

## CORS / pixels

- Content scripts CANNOT read cross-origin pixels (canvas taint) and can't fetch cross-origin (since Chrome 85).
- **Solution**: send the URL to the extension context; `fetch()` there is CORS-exempt for `host_permissions` hosts → `createImageBitmap(blob)` → OffscreenCanvas → tensor. Never touch the page's canvas.
- Edge cases: `blob:` URLs are page-scoped (content script must read them itself); auth-gated images (`credentials: "include"` where needed); use `img.currentSrc` for srcset.

## Performance (224px classifier, per image, rough)

| Setup | ViT-base | EffNet-B0/ConvNeXt-T |
|---|---|---|
| WASM 1 thread SIMD q8 | 300–1500 ms | 60–300 ms |
| WASM 4 threads (offscreen) | 100–500 ms | 25–100 ms |
| WebGPU fp16 discrete GPU | 10–50 ms | 5–20 ms |
| WebGPU integrated | 30–120 ms | 15–50 ms |

- On WebGPU, batch 4–8 images. On WASM, batching helps little — queue serially, prioritize viewport (IntersectionObserver).
- ViT-S/EffNet-class model keeps WASM fallback usable.

## Risks / gotchas

1. SW ~30s idle kill → session re-init + shader recompile on restart. Open message ports reset the timer; or host in offscreen (long-lived).
2. WebGPU unavailable on some machines (old GPUs, blocklists, policy) → always webgpu→wasm chain; probe `navigator.gpu.requestAdapter()`.
3. onnxruntime-web ≥1.19 / transformers.js v3+ required for SW (dynamic import ban).
4. transformers.js can cache failed responses — clear Cache Storage when changing model config.
5. `<all_urls>` triggers heavier Web Store review; disclose the model download.
6. ViT-base fp32 ≈ 350MB runtime footprint — quantize, keep one session.

## Example repos (verified)

- huggingface/transformers.js-examples/browser-extension — pipeline in SW, WebGPU
- GoogleChrome/chrome-extensions-samples functional-samples/sample.webgpu
- mlc-ai/web-llm chrome-extension-webgpu-service-worker
- kernel64/deepfake-detector-addon — ViT real/fake ONNX in-extension with overlays (domain precedent)
- Medium: "Transformers.js + ONNX Runtime WebGPU in Chrome extension" (offscreen pattern)
