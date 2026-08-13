// Offscreen document: owns image fetching/decoding, the content-hash result cache, and
// model inference. Hashing lives here because this is where the raw bytes already are —
// a SHA-256 over the encoded image identifies it regardless of which page/URL served it.

import { createEngine } from './engine.js';

const enginePromise = createEngine();

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === 'OFFSCREEN_INFER') {
    infer(msg.url).then(sendResponse);
    return true;
  }
  if (msg?.type === 'OFFSCREEN_STATUS') {
    enginePromise
      .then((e) => sendResponse({ ready: true, backend: e.backend, model: e.name }))
      .catch((err) => sendResponse({ ready: false, error: String(err) }));
    return true;
  }
});

async function infer(url) {
  try {
    // Some hosts (grok.com, CDNs behind auth) reject anonymous requests — retry with
    // credentials before giving up. data: URLs come from the content script inline.
    let resp = await fetch(url, { credentials: 'omit', cache: 'force-cache' });
    if (!resp.ok && /^https?:/.test(url)) {
      resp = await fetch(url, { credentials: 'include', cache: 'force-cache' });
    }
    if (!resp.ok) throw new Error(`fetch ${resp.status}`);
    const bytes = await resp.arrayBuffer();

    // Offscreen documents only get chrome.runtime.* — no chrome.storage — so all
    // cache reads/writes go through the background worker via messaging.
    const hash = toHex(await crypto.subtle.digest('SHA-256', bytes));
    const cached = await chrome.runtime
      .sendMessage({ type: 'HASH_GET', hash })
      .catch(() => null);
    if (cached && typeof cached.p === 'number') {
      return { aiProbability: cached.p, graphic: !!cached.g, cached: true };
    }

    const engine = await enginePromise;
    const bitmap = await createImageBitmap(new Blob([bytes]));
    const { aiProbability, graphic } = await engine.classify(bitmap);
    bitmap.close?.();

    chrome.runtime
      .sendMessage({ type: 'HASH_PUT', hash, p: aiProbability, g: graphic })
      .catch(() => {});
    return { aiProbability, graphic };
  } catch (e) {
    return { error: String(e) };
  }
}

function toHex(buf) {
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}
