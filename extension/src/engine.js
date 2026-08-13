// Inference engine: Community Forensics ViT-S/16 (224px) via ONNX Runtime Web.
// Everything outside this file only knows `classify(imageBitmap) -> p(AI)`.
//
// Preprocessing MUST stay in lockstep with eval/harness.py:
//   resize shorter side to 256 -> center crop 224 -> [0,1] -> ImageNet mean/std, NCHW.
// Output: single logit; sigmoid(logit + CALIBRATION_OFFSET) = p(AI-generated).

import * as ort from '../lib/ort/ort.all.bundle.min.mjs';

// Resolves inside the extension (chrome.runtime) and on plain test pages (import.meta),
// so parity tests exercise this exact file.
const BASE =
  typeof chrome !== 'undefined' && chrome.runtime?.getURL
    ? chrome.runtime.getURL('')
    : new URL('..', import.meta.url).href;

const MODEL_URL = new URL('models/detector.onnx', BASE).href;
const RESIZE = 440;
const CROP = 384;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// Logit offset that shifts the model's optimal balanced-accuracy operating point to the
// bounty's fixed 0.65 threshold. Model = CommFor ViT-S/384 with a re-fit head
// (eval/refit_head.py, lam=0.1, 2026-08-13): pooled BA@0.65 = 0.8915 across
// OpenFake/COCO/VOC/Commons/Unsplash eval sets; OOD fake recall preserved (99.6%).
const CALIBRATION_OFFSET = 2.29;

export async function createEngine() {
  ort.env.wasm.wasmPaths = new URL('lib/ort/', BASE).href;
  // Offscreen documents are not crossOriginIsolated by default -> no SharedArrayBuffer.
  if (!self.crossOriginIsolated) ort.env.wasm.numThreads = 1;

  for (const providers of [['webgpu'], ['wasm']]) {
    try {
      const session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: providers,
      });
      const engine = new OnnxEngine(session, providers[0]);
      await engine.warmup();
      return engine;
    } catch (e) {
      console.warn(`[engine] ${providers[0]} init failed:`, e);
    }
  }
  throw new Error('No ONNX execution provider available');
}

class OnnxEngine {
  constructor(session, backend) {
    this.session = session;
    this.backend = backend;
    this.name = 'commfor-vits-384';
    // Serialize inferences: one session, and WebGPU shader state dislikes interleaving.
    this.queue = Promise.resolve();
  }

  async warmup() {
    const zeros = new ort.Tensor('float32', new Float32Array(3 * CROP * CROP), [1, 3, CROP, CROP]);
    await this.session.run({ [this.session.inputNames[0]]: zeros });
  }

  classify(bitmap) {
    const run = async () => {
      const { tensor, graphic } = await preprocess(bitmap);
      const out = await this.session.run({ [this.session.inputNames[0]]: tensor });
      const rawLogit = out[this.session.outputNames[0]].data[0] + CALIBRATION_OFFSET;
      // Non-photographic content (charts, tables, UI screenshots): the photo detector's
      // score is out-of-distribution noise there, and such graphics are rarely
      // generator output — apply a fixed prior penalty. Measured effect on the photo
      // benchmark: -0.06pp balanced accuracy (see eval/graphic_gate.py).
      const logit = graphic ? rawLogit - GRAPHIC_PENALTY : rawLogit;
      return { aiProbability: 1 / (1 + Math.exp(-logit)), rawLogit, graphic };
    };
    const result = this.queue.then(run);
    this.queue = result.catch(() => {});
    return result;
  }
}

const GRAPHIC_PENALTY = 3.0; // logits; keeps only very confident detections on graphics

// Resize shorter side to RESIZE, center-crop CROP, ImageNet-normalize, NCHW float32.
async function preprocess(bitmap) {
  const scale = RESIZE / Math.min(bitmap.width, bitmap.height);
  const w = Math.round(bitmap.width * scale);
  const h = Math.round(bitmap.height * scale);

  // Canvas drawImage scaling with high smoothing tracks PIL's bilinear more closely
  // than createImageBitmap's resizer (measured in test/parity.html: 1.98 vs 3.04 max
  // logit diff; decisions at the 0.65 operating point match Python 16/16 either way).
  const canvas = new OffscreenCanvas(CROP, CROP);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  // Draw the resized image so its center lands on the canvas center (= center crop).
  ctx.drawImage(bitmap, (CROP - w) / 2, (CROP - h) / 2, w, h);

  const { data: rgba } = ctx.getImageData(0, 0, CROP, CROP);
  const plane = CROP * CROP;
  const chw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    chw[i] = (rgba[i * 4] / 255 - MEAN[0]) / STD[0];
    chw[i + plane] = (rgba[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    chw[i + 2 * plane] = (rgba[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  return {
    tensor: new ort.Tensor('float32', chw, [1, 3, CROP, CROP]),
    graphic: detectGraphic(rgba),
  };
}

// Graphic/screenshot detector (charts, tables, UI, text pages): huge flat-color runs
// and a tiny palette, unlike photos or generated art. Thresholds tuned in
// eval/graphic_gate.py — fires on 100% of rendered graphics, 1.5% of photos.
function detectGraphic(rgba) {
  const size = CROP;
  let flat = 0;
  const palette = new Set();
  for (let y = 0; y < size; y++) {
    const row = y * size;
    for (let x = 0; x < size; x++) {
      const i = (row + x) * 4;
      palette.add(((rgba[i] >> 3) << 10) | ((rgba[i + 1] >> 3) << 5) | (rgba[i + 2] >> 3));
      if (x + 1 < size) {
        const j = i + 4;
        if (rgba[i] === rgba[j] && rgba[i + 1] === rgba[j + 1] && rgba[i + 2] === rgba[j + 2]) flat++;
      }
    }
  }
  const flatFrac = flat / (size * (size - 1));
  const colors = palette.size / 1024;
  return flatFrac > 0.62 || (flatFrac > 0.45 && colors < 0.08);
}
