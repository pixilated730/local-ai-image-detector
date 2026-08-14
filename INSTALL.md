# Install & test (Brave / Chrome)

The extension is a plain unpacked Manifest V3 extension — Brave loads it exactly like
Chrome does. Everything it needs (model + ONNX Runtime) is already inside the folder,
so there's no build step and no network access required at runtime.

## Easiest path: prebuilt release (no build tools)

1. Download the zip from the repo's **Releases** page and unzip it
2. Open **`brave://extensions`** (or `chrome://extensions`)
3. Turn on **Developer mode** (toggle, top-right)
4. Click **Load unpacked** and select the unzipped `local-ai-image-detector` folder
5. Click the toolbar icon → switch **Detection** on

That's the whole install. The section below covers loading from a source checkout.

## Load it in Brave (from source checkout)

1. Open **`brave://extensions`** (paste into the address bar — it can't be linked to)
2. Turn on **Developer mode** (toggle, top-right)
3. Click **Load unpacked**
4. Select the folder that **contains `manifest.json`**:
   `...\ai-image-detector\extension`

   > **Pick `extension` itself, not a folder inside it.** In the Windows picker,
   > single-click `extension` to highlight it and press **Select Folder** — if a
   > subfolder like `icons` or `src` is highlighted instead, the load fails with
   > *"Manifest file is missing or unreadable."*

The "Local AI Image Detector" card appears. Pin it to the toolbar (puzzle-piece icon →
pin) so you can reach the toggle quickly.

## Turn it on

Detection is **off by default**. Click the extension icon and flip **Detection** on.
The popup shows:

- **Engine** — `commfor-vits-384` once the model is loaded
- **Backend** — `webgpu` (fast) or `wasm` (fallback, slower)
- **Status** — `ready` when the model has finished loading

The first load takes a few seconds (87 MB model + runtime init). After that the model
stays resident. Then just browse — x.com, Reddit, news sites, anywhere.

## What you should see

- Images are analyzed **one at a time**, starting with what's on your screen
- Only images judged AI-generated get a red **"AI generated"** tag in the top-left
  corner; real images get nothing. Hover the tag for the exact confidence
- Scrolling back over images you've already passed does **not** re-analyze them
- Small images (<128 px) are skipped — icons and avatars aren't worth analyzing

## Settings

- **Detection** — master on/off
- **Remember images (hash cache)** — stores a SHA-256 of each analyzed image with its
  score in local storage, so the same image is instantly recognized on any other page or
  after a restart, with no re-inference. Turn it off if you'd rather keep nothing.
- **Clear image cache** — wipes those stored hashes

## Privacy

No image bytes ever leave the browser. There is no server, no API call, no telemetry.
The one-time model file is bundled in the folder, so the extension works fully offline.

## Troubleshooting

- **Popup says `wasm` instead of `webgpu`** — WebGPU is unavailable (old GPU, driver
  blocklist, or disabled). It still works, just slower (~1 s/image vs well under that).
  Check `brave://gpu` for WebGPU status.
- **Status stuck on "loading model…"** — open `brave://extensions`, click the
  extension's **service worker** link, and check the console for errors.
- **No tags appear anywhere** — confirm Detection is on, and that the page's images are
  larger than 128 px. Reload the page after enabling.
- **Inspecting the inference host**: `brave://inspect/#other` → the offscreen document.
