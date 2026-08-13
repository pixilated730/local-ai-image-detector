"""Fetch modern REAL photos from Wikimedia Commons "Quality images".

Why this source: QI candidates are human-reviewed real photographs (AI-generated
images are explicitly ineligible), most are recent (2015+), high-resolution, and
professionally processed — exactly the "modern photography" distribution where the
detector false-positives. The API serves 1024px thumbnails directly, which matches
our web-realistic pipeline.

Writes:
  data/commons/train/real/*.jpg   (--train-n, default 900)
  data/commons/test/real/*.jpg    (--test-n, default 350)

Usage: python fetch_commons.py
"""

import argparse
import io
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "ai-image-detector-eval/0.1 (research; contact: njirokevz@gmail.com)"}
JPEG_Q = 85


def api(params: dict) -> dict:
    qs = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(f"{API}?{qs}", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def iter_quality_images():
    """Yield (title, thumb_url) for members of Category:Quality images."""
    cont = {}
    while True:
        resp = api({
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": "Category:Quality images",
            "gcmtype": "file",
            "gcmlimit": "200",
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": "1024",
            **cont,
        })
        for page in resp.get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("mime") in ("image/jpeg", "image/png") and info.get("thumburl"):
                yield info["thumburl"]
        cont = resp.get("continue")
        if not cont:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=900)
    ap.add_argument("--test-n", type=int, default=350)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent / "data" / "commons"
    train = root / "train" / "real"
    test = root / "test" / "real"
    train.mkdir(parents=True, exist_ok=True)
    test.mkdir(parents=True, exist_ok=True)

    counts = {"train": len(list(train.glob("*.jpg"))), "test": len(list(test.glob("*.jpg")))}
    targets = {"train": args.train_n, "test": args.test_n}
    lock = Lock()
    failures = 0

    def work(url, split):
        nonlocal failures
        img = None
        for attempt, delay in enumerate((0, 3, 8)):  # Commons rate-limits: retry w/ backoff
            if delay:
                import time
                time.sleep(delay)
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=30) as r:
                    img = Image.open(io.BytesIO(r.read())).convert("RGB")
                break
            except Exception:
                continue
        if img is None:
            with lock:
                failures += 1
            return
        try:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=JPEG_Q)  # uniform web-realistic re-encode
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        except Exception:
            with lock:
                failures += 1
            return
        with lock:
            if counts[split] >= targets[split]:
                return
            img.save((train if split == "train" else test) / f"{counts[split]:05d}.jpg",
                     "JPEG", quality=JPEG_Q)
            counts[split] += 1
            done = counts["train"] + counts["test"]
            if done % 100 == 0:
                print(f"  train={counts['train']} test={counts['test']} failures={failures}",
                      flush=True)

    print("listing Category:Quality images ...", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        i = 0
        for url in iter_quality_images():
            if all(counts[k] >= targets[k] for k in targets):
                break
            # interleave: every 4th image to test, others to train
            split = "test" if i % 4 == 0 else "train"
            if counts[split] < targets[split]:
                pool.submit(work, url, split)
            i += 1

    print(f"DONE train={counts['train']} test={counts['test']} failures={failures}", flush=True)


if __name__ == "__main__":
    main()
