// Content script: finds every image a user can actually see — <img>, <picture>/srcset,
// CSS background images, video posters, images inside open shadow roots and iframes —
// queues them viewport-first, and asks the background worker to analyze them ONE AT A
// TIME. Detected images get a simple "AI generated" tag; clean images get nothing.
// Repeat sightings are free (element, URL, and content-hash caches).

(() => {
  // Declared in the manifest AND injected into already-open tabs by the background
  // worker, so guard against running twice in one document.
  if (window.__laidInjected) return;
  window.__laidInjected = true;

  const MIN_DISPLAY_SIZE = 96; // px, both dimensions — skip icons/avatars
  const MIN_NATURAL_SIZE = 128;
  const AI_THRESHOLD = 0.65;
  const MAX_INLINE_BYTES = 6e6; // cap for blob:/data: images passed inline
  const SCAN_ELEMENT_CAP = 4000; // bound background-image scanning cost per pass

  let enabled = false;
  let showScores = true; // bounty requirement: a confidence score on EVERY analyzed image
  const queued = new WeakSet(); // elements already queued
  const urlResults = new Map(); // url -> p(AI), page-lifetime cache
  const queue = []; // candidates awaiting analysis
  const badges = new Set();
  let draining = false;

  // Set `localStorage.laidDebug = 1` on a page to trace every decision.
  const DEBUG = (() => {
    try {
      return !!localStorage.getItem('laidDebug');
    } catch {
      return false;
    }
  })();
  const debug = (...a) => DEBUG && console.debug('[ai-detector]', ...a);

  // --- Settings ----------------------------------------------------------------

  chrome.storage.local
    .get({ enabled: false, showScores: true })
    .then(({ enabled: on, showScores: s }) => {
      showScores = s;
      if (on) start();
    });
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== 'local') return;
    if (changes.showScores) {
      showScores = changes.showScores.newValue;
      if (!showScores) {
        // Drop the neutral chips; keep the AI tags.
        for (const host of [...badges]) {
          if (+host.dataset.laidConfidence < AI_THRESHOLD) {
            host.remove();
            badges.delete(host);
          }
        }
      }
    }
    if (changes.enabled) changes.enabled.newValue ? start() : stop();
  });

  function start() {
    if (enabled) return;
    enabled = true;
    debug('enabled on', location.href);
    scan(document);
    observe(document);
    scheduleBackgroundScan();
  }

  function stop() {
    enabled = false;
    observers.forEach((o) => o.disconnect());
    observers.length = 0;
    io.disconnect();
    queue.length = 0;
    badges.forEach((h) => h.remove());
    badges.clear();
    debug('disabled');
  }

  // --- What counts as an image ---------------------------------------------------

  // Returns the image URL an element currently displays, or null.
  function urlOf(el) {
    if (el.tagName === 'IMG') return el.currentSrc || el.src || null;
    if (el.tagName === 'VIDEO') return el.poster || null;
    const bg = getComputedStyle(el).backgroundImage;
    if (bg && bg !== 'none') {
      // Take the first url(...) layer; ignore gradients.
      const m = /url\((['"]?)(.*?)\1\)/.exec(bg);
      if (m && m[2] && !m[2].startsWith('#')) return m[2];
    }
    return null;
  }

  function bigEnough(el) {
    const r = el.getBoundingClientRect();
    if (r.width < MIN_DISPLAY_SIZE || r.height < MIN_DISPLAY_SIZE) return false;
    // <img> also has intrinsic size; background images don't expose one.
    if (el.tagName === 'IMG' && el.naturalWidth) {
      return el.naturalWidth >= MIN_NATURAL_SIZE && el.naturalHeight >= MIN_NATURAL_SIZE;
    }
    return true;
  }

  // --- Discovery -----------------------------------------------------------------

  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          io.unobserve(e.target);
          enqueue(e.target);
        }
      }
    },
    { rootMargin: '300px 0px' } // prepare just off-screen so scrolling feels instant
  );

  function consider(el) {
    if (!enabled || queued.has(el)) return;
    io.observe(el);
  }

  // Walk a root (document or shadow root), including nested open shadow roots.
  function scan(root, depth = 0) {
    if (!root || depth > 8) return;
    for (const el of root.querySelectorAll?.('img, video') ?? []) consider(el);
    for (const el of root.querySelectorAll?.('*') ?? []) {
      if (el.shadowRoot) {
        scan(el.shadowRoot, depth + 1);
        observe(el.shadowRoot);
      }
    }
  }

  // Background images can be on any element, and computing styles is expensive, so
  // sweep on idle with a cap rather than on every mutation.
  let bgScanQueued = false;
  function scheduleBackgroundScan() {
    if (!enabled || bgScanQueued) return;
    bgScanQueued = true;
    const run = () => {
      bgScanQueued = false;
      if (!enabled) return;
      let checked = 0;
      for (const el of document.querySelectorAll('*')) {
        if (++checked > SCAN_ELEMENT_CAP) break;
        if (queued.has(el) || el.tagName === 'IMG' || el.tagName === 'VIDEO') continue;
        const r = el.getBoundingClientRect();
        if (r.width < MIN_DISPLAY_SIZE || r.height < MIN_DISPLAY_SIZE) continue;
        if (r.bottom < -1000 || r.top > innerHeight + 1000) continue; // near viewport only
        if (urlOf(el)) consider(el);
      }
    };
    (window.requestIdleCallback ?? setTimeout)(run, { timeout: 1000 });
  }

  const observers = [];
  function observe(root) {
    const mo = new MutationObserver((mutations) => {
      for (const m of mutations) {
        if (m.type === 'attributes') {
          // src/srcset/style changed — re-evaluate this element.
          queued.delete(m.target);
          consider(m.target);
          continue;
        }
        for (const node of m.addedNodes) {
          if (node.nodeType !== Node.ELEMENT_NODE) continue;
          if (node.tagName === 'IMG' || node.tagName === 'VIDEO') consider(node);
          scan(node);
        }
      }
      scheduleBackgroundScan();
    });
    mo.observe(root === document ? document.documentElement : root, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['src', 'srcset', 'style', 'poster', 'class'],
    });
    observers.push(mo);
  }

  addEventListener('scroll', scheduleBackgroundScan, { passive: true, capture: true });

  // --- Sequential, viewport-priority analysis ------------------------------------

  function enqueue(el) {
    if (queued.has(el)) return;
    queued.add(el);
    queue.push(el);
    void drain();
  }

  // Always take the queued element nearest the top of the CURRENT viewport, so fast
  // scrolling re-prioritizes automatically.
  function nextTarget() {
    let best = -1;
    let bestScore = Infinity;
    for (let i = 0; i < queue.length; i++) {
      const r = queue[i].getBoundingClientRect();
      const offscreen = r.bottom < 0 || r.top > innerHeight;
      const score = (offscreen ? 1e6 : 0) + Math.abs(r.top);
      if (score < bestScore) {
        bestScore = score;
        best = i;
      }
    }
    return best >= 0 ? queue.splice(best, 1)[0] : null;
  }

  async function drain() {
    if (draining) return;
    draining = true;
    try {
      let el;
      while (enabled && (el = nextTarget())) {
        await analyze(el).catch((e) => debug('analyze threw', String(e)));
      }
    } finally {
      draining = false;
    }
  }

  async function analyze(el) {
    if (!el.isConnected) return;

    // <img> may not have decoded yet; wait for it rather than skipping.
    if (el.tagName === 'IMG' && !el.complete) {
      await new Promise((res) => {
        el.addEventListener('load', res, { once: true });
        el.addEventListener('error', res, { once: true });
      });
    }

    const url = urlOf(el);
    if (!url) return debug('skip (no url)', el.tagName);
    if (!bigEnough(el)) return debug('skip (too small)', url.slice(0, 80));

    let r = urlResults.get(url);
    if (r === undefined) {
      // blob:/data: URLs are page-scoped and unreachable from the extension context —
      // read the bytes here and hand them over inline.
      let payload = url;
      if (!/^https?:/.test(url)) {
        payload = await inlineImage(url);
        if (!payload) return debug('skip (unreadable local url)', url.slice(0, 60));
      }
      let result;
      try {
        result = await chrome.runtime.sendMessage({ type: 'ANALYZE_IMAGE', url: payload });
      } catch (e) {
        return debug('message failed (extension reloaded?)', String(e));
      }
      if (!result || result.error || typeof result.aiProbability !== 'number') {
        return debug('analysis failed:', result?.error, url.slice(0, 80));
      }
      r = { p: result.aiProbability, graphic: !!result.graphic };
      urlResults.set(url, r);
    }

    debug(
      `p(AI)=${r.p.toFixed(3)}${r.graphic ? ' [graphic]' : ''}${r.p >= AI_THRESHOLD ? '  → TAGGED' : ''}`,
      url.slice(0, 80)
    );
    if (!enabled) return;
    // AI images always get the red tag; below threshold, a neutral score chip is shown
    // when "show scores" is on (the bounty requires a visible confidence score for
    // every analyzed image).
    if (r.p >= AI_THRESHOLD) badge(el, r.p, r.graphic);
    else if (showScores) badge(el, r.p, r.graphic);
  }

  async function inlineImage(url) {
    try {
      const blob = await (await fetch(url)).blob();
      if (blob.size > MAX_INLINE_BYTES) return null;
      return await new Promise((resolve) => {
        const fr = new FileReader();
        fr.onload = () => resolve(fr.result);
        fr.onerror = () => resolve(null);
        fr.readAsDataURL(blob);
      });
    } catch {
      return null;
    }
  }

  // --- "AI generated" tag ---------------------------------------------------------

  // Constructed stylesheet: CSSOM-based, so strict page CSP (x.com, grok.com) cannot
  // block it the way it blocks injected <style> elements. Shared by all badges.
  let badgeSheet = null;
  function getBadgeSheet() {
    if (badgeSheet) return badgeSheet;
    badgeSheet = new CSSStyleSheet();
    badgeSheet.replaceSync(`
      .pill {
        position: absolute; top: 0; left: 0;
        display: inline-flex; align-items: center; gap: 5px;
        font: 600 11px/1 system-ui, -apple-system, sans-serif;
        padding: 4px 9px; border-radius: 999px;
        white-space: nowrap; cursor: default; pointer-events: auto;
        box-shadow: 0 2px 6px rgba(0,0,0,.35);
        transition: transform .12s ease;
        -webkit-backdrop-filter: blur(4px); backdrop-filter: blur(4px);
      }
      .pill:hover { transform: scale(1.06); }
      .pill.ai { background: rgba(185, 28, 28, .92); color: #fff; }
      .pill.ok { background: rgba(22, 22, 24, .72); color: rgba(255,255,255,.92); }
      .pill.gfx { background: rgba(51, 65, 85, .78); color: rgba(255,255,255,.9); }
      .dot { width: 6px; height: 6px; border-radius: 50%; }
      .ai .dot { background: #fff; box-shadow: 0 0 5px rgba(255,255,255,.9); }
      .ok .dot { background: #34d399; }
      .gfx .dot { background: #93c5fd; }
      .pct { font-weight: 700; opacity: .95; }
      .pill .detail {
        display: none; font-weight: 400; opacity: .85; margin-left: 2px;
      }
      .pill:hover .detail { display: inline; }
    `);
    return badgeSheet;
  }

  function badge(target, p, graphic = false) {
    const isAI = p >= AI_THRESHOLD;
    const host = document.createElement('div');
    host.className = 'laid-badge-host';
    host.dataset.laidConfidence = p.toFixed(4);
    if (graphic) host.dataset.laidGraphic = '1';
    const shadow = host.attachShadow({ mode: 'closed' });
    try {
      shadow.adoptedStyleSheets = [getBadgeSheet()];
    } catch {
      /* very old Chrome: pill still renders unstyled-but-legible via title */
    }

    const el = document.createElement('div');
    el.className = `pill ${isAI ? 'ai' : graphic ? 'gfx' : 'ok'}`;
    el.title =
      `Local AI Image Detector — P(AI-generated) = ${(p * 100).toFixed(1)}%` +
      (graphic
        ? '\nLooks like a graphic/screenshot — photo-detector confidence is reduced here.'
        : '') +
      `\nModel: Community Forensics ViT-S (runs entirely on this device)`;

    const dot = document.createElement('span');
    dot.className = 'dot';
    const label = document.createElement('span');
    const pct = document.createElement('span');
    pct.className = 'pct';
    pct.textContent = `${Math.round(p * 100)}%`;
    if (isAI) {
      label.textContent = 'AI generated';
      el.append(dot, label, pct);
    } else if (graphic) {
      // Charts/tables/UI screenshots: the photo detector doesn't really apply —
      // say so instead of pretending the score is meaningful.
      label.textContent = 'graphic';
      el.append(dot, label, pct);
    } else {
      // Neutral chip: score only, unobtrusive; label appears on hover.
      const detail = document.createElement('span');
      detail.className = 'detail';
      detail.textContent = 'AI score';
      el.append(dot, pct, detail);
    }
    shadow.append(el);
    document.documentElement.appendChild(host);
    badges.add(host);

    const reposition = () => {
      if (!target.isConnected) {
        host.remove();
        badges.delete(host);
        return;
      }
      const r = target.getBoundingClientRect();
      Object.assign(host.style, {
        position: 'fixed',
        left: `${r.left}px`,
        top: `${r.top}px`,
        width: '0',
        height: '0',
        zIndex: '2147483647',
      });
    };
    reposition();
    new ResizeObserver(reposition).observe(target);
    addEventListener('scroll', reposition, { passive: true, capture: true });
    addEventListener('resize', reposition, { passive: true });
  }
})();
