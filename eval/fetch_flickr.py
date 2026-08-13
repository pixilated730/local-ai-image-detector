"""Fetch Flickr consumer photos WITH metadata, split by CAPTURE date (date_taken).

Why: the dataset is a 2022 scrape — every row has date_upload=2022, so upload-date
filtering is useless. But date_taken comes from camera EXIF: a photo TAKEN before 2022
cannot be modern-generator AI (the generators didn't exist), while still being modern
consumer photography (we accept 2015+). Photos taken in 2022 go to the "unknown" bucket
— that's where AI contamination can live.

Writes:
  data/flickr_dated/test_pre2022/real/*.jpg    guaranteed-real held-out test
  data/flickr_dated/train_pre2022/real/*.jpg   training pool (--train-n)
  data/flickr_dated/post2022/unknown/*.jpg     taken-2022 uploads (label unknown!)
  data/flickr_dated/meta.csv                   file,split,date_taken,width,height,url

Also saves ORIGINAL-resolution copies for a native-pixel diagnosis subset:
  data/flickr_dated/originals/*.jpg            full size, no re-encode

Usage: python fetch_flickr.py [--train-n 1200]
"""

import argparse
import csv
import io
import urllib.request
from pathlib import Path

from datasets import load_dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # originals can exceed PIL's default bomb limit

MAX_SIDE = 1024
JPEG_Q = 85
UA = {"User-Agent": "Mozilla/5.0"}


def web_realistic(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=JPEG_Q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def fetch(url: str, timeout=25) -> Image.Image | None:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return Image.open(io.BytesIO(r.read()))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=1200)
    ap.add_argument("--test-n", type=int, default=500)
    ap.add_argument("--post-n", type=int, default=300)
    ap.add_argument("--originals-n", type=int, default=80)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent / "data" / "flickr_dated"
    dirs = {
        "test": root / "test_pre2022" / "real",
        "train": root / "train_pre2022" / "real",
        "post": root / "post2022" / "unknown",
        "orig": root / "originals",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    targets = {"test": args.test_n, "train": args.train_n, "post": args.post_n}
    counts = {k: len(list(d.glob("*.jpg"))) for k, d in dirs.items()}

    meta_path = root / "meta.csv"
    meta_new = not meta_path.exists()
    meta = open(meta_path, "a", newline="", encoding="utf-8")
    w = csv.writer(meta)
    if meta_new:
        w.writerow(["file", "split", "date_upload", "width", "height", "url"])

    print("streaming Chr0my/public_flickr_photos_license_1 ...", flush=True)
    ds = load_dataset("Chr0my/public_flickr_photos_license_1", split="train", streaming=True)

    from concurrent.futures import ThreadPoolExecutor
    import threading

    lock = threading.Lock()
    failures = 0
    stop = False

    def route(year):
        """Pick the split for a row; None if that split is already full."""
        pre2022 = year < 2022
        if pre2022:
            split = "test" if counts["test"] * args.train_n <= counts["train"] * args.test_n else "train"
        else:
            split = "post"
        return split if counts[split] < targets[split] else None

    def work(url, date, year, split):
        nonlocal failures, stop
        img = fetch(url)
        with lock:
            if img is None:
                failures += 1
                if failures > 6000:
                    stop = True
                return
            if counts[split] >= targets[split]:
                return
            try:
                name = f"{counts[split]:05d}.jpg"
                if split != "post" and counts["orig"] < args.originals_n and min(img.size) >= 1400:
                    img.convert("RGB").save(dirs["orig"] / f"{counts['orig']:05d}.jpg", "JPEG", quality=95)
                    counts["orig"] += 1
                web_realistic(img).save(dirs[split] / name, "JPEG", quality=JPEG_Q)
                w.writerow([name, split, date, img.width, img.height, url])
                meta.flush()
                counts[split] += 1
            except Exception:
                failures += 1
                return
            done = sum(counts[k] for k in targets)
            if done % 50 == 0:
                print(f"  test={counts['test']} train={counts['train']} post={counts['post']} "
                      f"orig={counts['orig']} failures={failures}", flush=True)

    with ThreadPoolExecutor(max_workers=24) as pool:
        pending = 0
        for row in ds:
            if stop or all(counts[k] >= targets[k] for k in targets):
                break
            url, date = row.get("url"), str(row.get("date_taken") or "")
            if not url or len(date) < 4:
                continue
            year = int(date[:4]) if date[:4].isdigit() else 0
            # 2015+ keeps photos "modern" (high-res phones/mirrorless); <=2021 keeps
            # them provably pre-generator.
            if not 2015 <= year <= 2022:
                continue
            split = route(year)
            if split is None:
                continue
            pool.submit(work, url, date, year, split)
            pending += 1
            if pending % 500 == 0:  # let the pool drain so counts stay accurate
                import time
                time.sleep(2)

    print(f"DONE test={counts['test']} train={counts['train']} post={counts['post']} "
          f"orig={counts['orig']} failures={failures}", flush=True)
    meta.close()


if __name__ == "__main__":
    main()
