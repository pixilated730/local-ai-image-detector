"""Broaden the real-image side of the benchmark with non-OpenFake distributions.

The proxy benchmark's reals all came from OpenFake (LAION/news-style), so the
false-positive rate on ordinary web photos was unverified. This pulls reals from
distinctly different sources — consumer Flickr photos (COCO), older consumer photos
(Pascal VOC 2012), amateur/phone-camera Flickr uploads (public_flickr_photos), and
modern photography (Unsplash) — applies the SAME web-realistic re-encode used for the
rest of the benchmark, and writes them to a separate tree for FPR measurement.

Output: data/reals_broad/<source>/real/*.jpg   (label 0 only — this measures FPR)

Usage: python build_reals.py [--per-source 200]
"""

import argparse
import io
from pathlib import Path

from datasets import load_dataset
from PIL import Image

MAX_SIDE = 1024
JPEG_Q = 85

# (key, hf repo, config, split, image column) — each a distinct real distribution.
# Datasets that embed image bytes.
SOURCES = [
    ("coco", "detection-datasets/coco", None, "val", "image"),
    ("voc2012", "nateraw/pascal-voc-2012", None, "val", "image"),
]

# Datasets that only carry URLs — fetched directly. Flickr "license 1" photos are
# ordinary uploads (phones and consumer cameras, EXIF intact), which is the closest
# public stand-in for the everyday photos a browsing user actually sees.
URL_SOURCES = [
    ("flickr_consumer", "Chr0my/public_flickr_photos_license_1", "train", "url"),
]


def web_realistic(img: Image.Image) -> Image.Image:
    """Identical treatment to build_proxy.py so classes stay format-matched."""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=JPEG_Q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def pick_image(row, column):
    """Datasets differ in schema; find a PIL image in the row."""
    val = row.get(column)
    if isinstance(val, Image.Image):
        return val
    for v in row.values():
        if isinstance(v, Image.Image):
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=200)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "reals_broad"))
    args = ap.parse_args()

    out = Path(args.out)
    for key, repo, config, split, column in SOURCES:
        dest = out / key / "real"
        dest.mkdir(parents=True, exist_ok=True)
        have = len(list(dest.glob("*.jpg")))
        if have >= args.per_source:
            print(f"[{key}] already has {have}, skipping")
            continue

        print(f"[{key}] streaming {repo} ({split}) ...")
        try:
            ds = load_dataset(repo, config, split=split, streaming=True)
        except Exception as e:
            print(f"[{key}] FAILED to open: {str(e)[:120]}")
            continue

        n, errors = have, 0
        try:
            for row in ds:
                if n >= args.per_source:
                    break
                img = pick_image(row, column)
                if img is None:
                    errors += 1
                    if errors > 50:
                        print(f"[{key}] no image column found, giving up")
                        break
                    continue
                try:
                    web_realistic(img).save(dest / f"{n:05d}.jpg", "JPEG", quality=JPEG_Q)
                except Exception:
                    errors += 1
                    continue
                n += 1
                if n % 50 == 0:
                    print(f"  [{key}] {n}/{args.per_source}")
        except Exception as e:
            print(f"[{key}] stream error after {n}: {str(e)[:120]}")
        print(f"[{key}] done: {n} images ({errors} skipped)")

    # URL-only datasets: fetch the image bytes ourselves.
    import urllib.request

    for key, repo, split, url_col in URL_SOURCES:
        dest = out / key / "real"
        dest.mkdir(parents=True, exist_ok=True)
        n = len(list(dest.glob("*.jpg")))
        if n >= args.per_source:
            print(f"[{key}] already has {n}, skipping")
            continue
        print(f"[{key}] streaming {repo} ({split}) via URLs ...")
        try:
            ds = load_dataset(repo, split=split, streaming=True)
        except Exception as e:
            print(f"[{key}] FAILED to open: {str(e)[:120]}")
            continue
        errors = 0
        for row in ds:
            if n >= args.per_source:
                break
            url = row.get(url_col)
            if not url:
                continue
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    img = Image.open(io.BytesIO(r.read()))
                web_realistic(img).save(dest / f"{n:05d}.jpg", "JPEG", quality=JPEG_Q)
            except Exception:
                errors += 1
                if errors > 200:
                    print(f"[{key}] too many fetch failures, stopping")
                    break
                continue
            n += 1
            if n % 50 == 0:
                print(f"  [{key}] {n}/{args.per_source}")
        print(f"[{key}] done: {n} images ({errors} failed fetches)")


if __name__ == "__main__":
    main()
