// Vendors onnxruntime-web runtime files into lib/ort/ so the extension is fully
// self-contained (MV3 CSP forbids CDN scripts). Run after `npm install`:
//   node build.mjs
import { cpSync, mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const dist = 'node_modules/onnxruntime-web/dist';
const out = 'lib/ort';
mkdirSync(out, { recursive: true });

cpSync(join(dist, 'ort.all.bundle.min.mjs'), join(out, 'ort.all.bundle.min.mjs'));
for (const f of readdirSync(dist)) {
  // .wasm binaries plus their ort-wasm*.mjs loader glue — ORT fetches both at runtime.
  if (f.endsWith('.wasm') || (f.startsWith('ort-wasm') && f.endsWith('.mjs')))
    cpSync(join(dist, f), join(out, f));
}
console.log('Vendored onnxruntime-web ->', out);
console.log('Model: place the exported ONNX at models/detector.onnx (see eval/export_onnx.py)');
