"""Fetch extra fakes from the HARD generators (flux-2, ideogram, nano-banana,
gpt-image-1.5, midjourney-7) out of OpenFake's VALIDATION split (train-safe; our eval
uses the test split only).

Round-3 recall at 0.65: flux-2-klein 66%, ideogram-2 67%, nano-banana 71%,
gpt-image-1.5 78%, midjourney-7 80%. The train pool capped every generator at ~95
images; these need more representation.

Writes: data/trainpool/fake_hard/*.jpg

Usage: python fetch_hard_fakes.py [--n 600]
"""

import argparse
import io
from pathlib import Path

from datasets import load_dataset
from PIL import Image

MAX_SIDE = 1024
JPEG_Q = 85
HARD = ("flux", "ideogram", "nano-banana", "gpt-image", "midjourney")
PER_GEN_CAP = 160


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()

    dest = Path(__file__).resolve().parent / "data" / "trainpool" / "fake_hard"
    dest.mkdir(parents=True, exist_ok=True)
    n = len(list(dest.glob("*.jpg")))
    if n >= args.n:
        print(f"already have {n}")
        return

    ds = load_dataset("ComplexDataLab/OpenFake", "core", split="validation", streaming=True)
    ds = ds.shuffle(seed=21, buffer_size=500)
    per_gen: dict[str, int] = {}

    for row in ds:
        if n >= args.n:
            break
        if str(row.get("label", "")).lower() != "fake":
            continue
        model = str(row.get("model", "")).lower()
        if not any(h in model for h in HARD):
            continue
        if per_gen.get(model, 0) >= PER_GEN_CAP:
            continue
        img = row.get("image")
        if img is None:
            continue
        try:
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in model)[:40]
            web_realistic(img).save(dest / f"{n:05d}_{safe}.jpg", "JPEG", quality=JPEG_Q)
        except Exception:
            continue
        per_gen[model] = per_gen.get(model, 0) + 1
        n += 1
        if n % 50 == 0:
            print(f"  {n}/{args.n} {dict(per_gen)}", flush=True)

    print(f"DONE {n}: {dict(per_gen)}", flush=True)


if __name__ == "__main__":
    main()
