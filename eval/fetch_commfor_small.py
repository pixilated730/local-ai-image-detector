"""Fetch a slice of OwensLab/CommunityForensics-Small as an out-of-distribution fake
check for the refit head (their generators are disjoint from OpenFake's frontier set).

Writes data/commfor_small/{fake,real}/*.jpg through the same web-realistic re-encode.

Usage: python fetch_commfor_small.py [--n 400]
"""

import argparse
import io
from pathlib import Path

from datasets import load_dataset, get_dataset_split_names
from PIL import Image

MAX_SIDE = 1024
JPEG_Q = 85


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


def label_of(row):
    for key in ("label", "labels", "is_fake", "fake"):
        if key in row:
            v = row[key]
            if isinstance(v, str):
                return 1 if v.lower() in ("fake", "generated", "1", "true") else 0
            return int(bool(v))
    return 1  # dataset is generated-image-centric; default fake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="per class")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent / "data" / "commfor_small"
    for c in ("fake", "real"):
        (root / c).mkdir(parents=True, exist_ok=True)

    splits = get_dataset_split_names("OwensLab/CommunityForensics-Small", "default")
    split = "test" if "test" in splits else splits[-1]
    print("using split:", split, "of", splits, flush=True)
    ds = load_dataset("OwensLab/CommunityForensics-Small", "default", split=split, streaming=True)
    ds = ds.shuffle(seed=5, buffer_size=300)

    counts = {"fake": 0, "real": 0}
    printed_keys = False
    for row in ds:
        if not printed_keys:
            print("columns:", list(row.keys()), flush=True)
            printed_keys = True
        if all(v >= args.n for v in counts.values()):
            break
        cls = "fake" if label_of(row) else "real"
        if counts[cls] >= args.n:
            continue
        img = row.get("image")
        if img is None:
            for v in row.values():
                if isinstance(v, Image.Image):
                    img = v
                    break
        if img is None:
            continue
        try:
            web_realistic(img).save(root / cls / f"{counts[cls]:05d}.jpg", "JPEG", quality=JPEG_Q)
        except Exception:
            continue
        counts[cls] += 1
        if sum(counts.values()) % 100 == 0:
            print(f"  fake={counts['fake']} real={counts['real']}", flush=True)
    print(f"DONE fake={counts['fake']} real={counts['real']}", flush=True)


if __name__ == "__main__":
    main()
