// Background service worker: routes analysis requests from content scripts to the
// offscreen document (which owns image decoding + model inference), and caches results.

const OFFSCREEN_URL = 'src/offscreen.html';

// In-memory result cache, keyed by image URL. The service worker can be killed at any
// time, so this is a best-effort cache; persistent caching can move to chrome.storage
// later if needed.
const resultCache = new Map();
const CACHE_MAX = 2000;
const AI_THRESHOLD = 0.65;

// Session stats shown in the popup. In-memory (reset when the worker restarts) and
// mirrored to storage.session so the popup can read them cheaply.
const stats = { analyzed: 0, flagged: 0, cacheHits: 0, totalMs: 0 };
function bumpStats(result, ms, fromCache) {
  if (fromCache || result?.cached) stats.cacheHits++;
  else {
    stats.analyzed++;
    stats.totalMs += ms;
  }
  if (typeof result?.aiProbability === 'number' && result.aiProbability >= AI_THRESHOLD)
    stats.flagged++;
  chrome.storage.session?.set({ stats }).catch?.(() => {});
}

let offscreenReady = null;

async function ensureOffscreen() {
  if (offscreenReady) return offscreenReady;
  offscreenReady = (async () => {
    const contexts = await chrome.runtime.getContexts({
      contextTypes: ['OFFSCREEN_DOCUMENT'],
    });
    if (contexts.length === 0) {
      await chrome.offscreen.createDocument({
        url: OFFSCREEN_URL,
        reasons: ['DOM_PARSER'], // full DOM APIs for image decode + WASM/WebGPU inference
        justification: 'Runs local ML inference to detect AI-generated images.',
      });
    }
  })();
  return offscreenReady;
}

function cachePut(key, value) {
  if (resultCache.size >= CACHE_MAX) {
    // Drop the oldest entry (Map preserves insertion order).
    const oldest = resultCache.keys().next().value;
    resultCache.delete(oldest);
  }
  resultCache.set(key, value);
}

// Content scripts declared in the manifest only load into pages opened AFTER install.
// Inject into already-open tabs so enabling detection works without a manual reload
// (important on SPAs like x.com, where navigation never triggers a fresh document).
async function injectExistingTabs() {
  const tabs = await chrome.tabs.query({ url: ['http://*/*', 'https://*/*'] });
  await Promise.all(
    tabs.map((tab) =>
      chrome.scripting
        .executeScript({ target: { tabId: tab.id, allFrames: true }, files: ['src/content.js'] })
        .catch(() => {}) // restricted pages (web store, other extensions) — expected
    )
  );
}

chrome.runtime.onInstalled.addListener(injectExistingTabs);
chrome.runtime.onStartup.addListener(injectExistingTabs);

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === 'ANALYZE_IMAGE') {
    handleAnalyze(msg).then(sendResponse);
    return true; // async response
  }
  if (msg?.type === 'GET_STATUS') {
    handleStatus().then(sendResponse);
    return true;
  }
  if (msg?.type === 'GET_STATS') {
    sendResponse({ ...stats });
    return false;
  }
  // Content-hash cache, serviced here because offscreen documents cannot touch
  // chrome.storage (they only get chrome.runtime messaging).
  if (msg?.type === 'HASH_GET') {
    hashGet(msg.hash).then(sendResponse);
    return true;
  }
  if (msg?.type === 'HASH_PUT') {
    void hashPut(msg.hash, msg.p, msg.g);
    sendResponse(true);
    return false;
  }
});

const HASH_CACHE_MAX = 5000; // entries in chrome.storage.local, pruned oldest-first

async function hashGet(hash) {
  try {
    const { hashCache = true } = await chrome.storage.local.get({ hashCache: true });
    if (!hashCache || !hash) return null;
    return (await chrome.storage.local.get(`h:${hash}`))[`h:${hash}`] ?? null;
  } catch {
    return null;
  }
}

async function hashPut(hash, p, g) {
  try {
    const { hashCache = true } = await chrome.storage.local.get({ hashCache: true });
    if (!hashCache || !hash || typeof p !== 'number') return;
    await chrome.storage.local.set({ [`h:${hash}`]: { p, g: !!g, t: Date.now() } });
    if (Math.random() < 0.01) {
      const all = await chrome.storage.local.get(null);
      const entries = Object.entries(all).filter(([k]) => k.startsWith('h:'));
      if (entries.length > HASH_CACHE_MAX) {
        entries.sort((a, b) => (a[1].t ?? 0) - (b[1].t ?? 0));
        await chrome.storage.local.remove(
          entries.slice(0, entries.length - HASH_CACHE_MAX).map(([k]) => k)
        );
      }
    }
  } catch {
    // cache is best-effort
  }
}

// createDocument() resolves before the offscreen module registers its onMessage
// listener, so the first send can fail with "Receiving end does not exist". Retry
// briefly instead of dropping the image.
async function sendToOffscreen(message, attempts = 8) {
  await ensureOffscreen();
  for (let i = 0; i < attempts; i++) {
    try {
      const result = await chrome.runtime.sendMessage(message);
      if (result !== undefined) return result;
    } catch (e) {
      if (!String(e).includes('Receiving end does not exist')) throw e;
    }
    await new Promise((r) => setTimeout(r, 150 * (i + 1)));
  }
  throw new Error('offscreen document did not respond');
}

async function handleAnalyze({ url }) {
  if (resultCache.has(url)) {
    const hit = resultCache.get(url);
    bumpStats(hit, 0, true);
    return hit;
  }
  try {
    const t0 = Date.now();
    const result = await sendToOffscreen({ type: 'OFFSCREEN_INFER', url });
    if (result && !result.error) {
      cachePut(url, result);
      bumpStats(result, Date.now() - t0, false);
    }
    return result;
  } catch (e) {
    return { error: String(e) };
  }
}

async function handleStatus() {
  try {
    return await sendToOffscreen({ type: 'OFFSCREEN_STATUS' });
  } catch (e) {
    return { ready: false, error: String(e) };
  }
}
