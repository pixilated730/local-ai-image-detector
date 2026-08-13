"""Fetch modern REAL photos from the Unsplash Lite dataset (25k curated photos).

Provenance: all photos were submitted to Unsplash before 2022 (Lite 1.2.0 snapshot) —
pre-generator, provably real — and most carry camera EXIF. This is modern professional
photography: exactly the distribution where the detector currently false-positives.
Images come from the images.unsplash.com CDN, which resizes server-side (?w=1024).

Requires data/unsplash/lite.zip (downloaded from https://unsplash.com/data/lite/latest).

Writes:
  data/unsplash/train/real/*.jpg  (--train-n, default 900)
  data/unsplash/test/real/*.jpg   (--test-n, default 350)

Usage: python fetch_unsplash.py
"""

import argparse
import csv
import io
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from PIL import Image

JPEG_Q = 85
UA = {"User-Agent": "Mozilla/5.0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=900)
    ap.add_argument("--test-n", type=int, default=350)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent / "data" / "unsplash"
    train = root / "train" / "real"
    test = root / "test" / "real"
    train.mkdir(parents=True, exist_ok=True)
    test.mkdir(parents=True, exist_ok=True)

    # photos.tsv000 lives inside the lite zip
    zf = zipfile.ZipFile(root / "lite.zip")
    tsv_name = next(n for n in zf.namelist() if n.startswith("photos.tsv"))
    rows = list(csv.DictReader(io.TextIOWrapper(zf.open(tsv_name), encoding="utf-8"), delimiter="\t"))
    print(f"{len(rows)} photos in {tsv_name}", flush=True)

    # Require camera EXIF (extra realness guarantee) and drop any 2022+ submissions.
    def ok(r):
        sub = (r.get("photo_submitted_at") or "")[:4]
        return (r.get("exif_camera_make") or r.get("exif_camera_model")) and sub.isdigit() and int(sub) < 2022

    rows = [r for r in rows if ok(r)]
    print(f"{len(rows)} pre-2022 photos with camera EXIF", flush=True)

    counts = {"train": len(list(train.glob("*.jpg"))), "test": len(list(test.glob("*.jpg")))}
    targets = {"train": args.train_n, "test": args.test_n}
    lock = Lock()
    failures = 0

    def work(url, split):
        nonlocal failures
        try:
            req = urllib.request.Request(f"{url}?w=1024&q=85&fm=jpg", headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                img = Image.open(io.BytesIO(r.read())).convert("RGB")
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
                print(f"  train={counts['train']} test={counts['test']} failures={failures}", flush=True)

    with ThreadPoolExecutor(max_workers=24) as pool:
        for i, r in enumerate(rows):
            if all(counts[k] >= targets[k] for k in targets):
                break
            split = "test" if i % 4 == 0 else "train"
            if counts[split] < targets[split]:
                pool.submit(work, r["photo_image_url"], split)

    print(f"DONE train={counts['train']} test={counts['test']} failures={failures}", flush=True)


if __name__ == "__main__":
    main()
