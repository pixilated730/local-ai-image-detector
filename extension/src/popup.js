const $ = (id) => document.getElementById(id);

async function refreshStatus() {
  const { enabled } = await chrome.storage.local.get({ enabled: false });
  if (!enabled) {
    $('status').textContent = 'off';
    $('status').className = '';
    return;
  }
  $('status').textContent = 'starting…';
  const s = await chrome.runtime.sendMessage({ type: 'GET_STATUS' }).catch(() => null);
  $('engine').textContent = s?.model ?? '—';
  $('backend').textContent = s?.backend ?? '—';
  if (s?.ready) {
    $('status').textContent = 'ready';
    $('status').className = 'ok';
  } else {
    $('status').textContent = s?.error ? 'error' : 'loading model…';
    $('status').className = s?.error ? 'bad' : '';
    if (s?.error) $('status').title = s.error;
  }
}

async function refreshStats() {
  const st = await chrome.runtime.sendMessage({ type: 'GET_STATS' }).catch(() => null);
  if (!st) return;
  $('sAnalyzed').textContent = st.analyzed;
  $('sFlagged').textContent = st.flagged;
  $('sCache').textContent = st.cacheHits;
  $('sSpeed').textContent = st.analyzed ? `${Math.round(st.totalMs / st.analyzed)} ms` : '—';
}

chrome.storage.local
  .get({ enabled: false, hashCache: true, showScores: true })
  .then(({ enabled, hashCache, showScores }) => {
    $('enabled').checked = enabled;
    $('hashCache').checked = hashCache;
    $('showScores').checked = showScores;
    void refreshStatus();
    void refreshStats();
  });

setInterval(refreshStats, 1500);

$('enabled').addEventListener('change', async (e) => {
  await chrome.storage.local.set({ enabled: e.target.checked });
  void refreshStatus();
});
$('showScores').addEventListener('change', (e) =>
  chrome.storage.local.set({ showScores: e.target.checked })
);
$('hashCache').addEventListener('change', (e) =>
  chrome.storage.local.set({ hashCache: e.target.checked })
);

$('clearCache').addEventListener('click', async () => {
  const all = await chrome.storage.local.get(null);
  const keys = Object.keys(all).filter((k) => k.startsWith('h:'));
  await chrome.storage.local.remove(keys);
  $('clearCache').textContent = `Cleared ${keys.length} entries`;
  setTimeout(() => ($('clearCache').textContent = 'Clear image cache'), 1500);
});
