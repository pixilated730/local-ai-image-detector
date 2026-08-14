"""Fetch pre-AI-era memes for the social-media-realistic benchmark expansion.

Memes are meme-laundered real content: old photos/frames, upscaled, text-overlaid,
recompressed many times. Live X testing showed the detector scoring several of them
just above the 0.65 threshold — a distribution our benchmark never covered. Memes from
datasets collected before 2022 are provably not modern-generator output, giving clean
"not-AI" labels for FPR measurement.

Tries several public HF meme datasets (Memotion-era, 2019-2020 collections) and takes
whichever streams successfully. Same web-realistic re-encode as the rest of the bench.

Writes: data/memes_pre2022/real/*.jpg

Usage: python fetch_memes.py [--n 400]
"""

import argparse
import io
from pathlib import Path

from datasets import load_dataset
from PIL import Image

MAX_SIDE = 1024
JPEG_Q = 85

CANDIDATES = [
    # (repo, config, split) — all collected pre-2022
    ("Ahren09/MMSoc_Memotion", None, "train"),
    ("mmathys/memotion7k", None, "train"),
    ("kirp/memotion", None, "train"),
    ("julien-c/reddit-memes", None, "train"),
    ("ceyda/memes", None, "train"),
]


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
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()

    dest = Path(__file__).resolve().parent / "data" / "memes_pre2022" / "real"
    dest.mkdir(parents=True, exist_ok=True)
    n = len(list(dest.glob("*.jpg")))
    if n >= args.n:
        print(f"already have {n}")
        return

    for repo, config, split in CANDIDATES:
        if n >= args.n:
            break
        try:
            ds = load_dataset(repo, config, split=split, streaming=True)
            print(f"[{repo}] streaming ...", flush=True)
        except Exception as e:
            print(f"[{repo}] FAILED: {str(e)[:100]}", flush=True)
            continue
        got_here = 0
        try:
            for row in ds:
                if n >= args.n:
                    break
                img = None
                for v in row.values():
                    if isinstance(v, Image.Image):
                        img = v
                        break
                if img is None:
                    continue
                if min(img.size) < 128:
                    continue
                try:
                    web_realistic(img).save(dest / f"{n:05d}.jpg", "JPEG", quality=JPEG_Q)
                except Exception:
                    continue
                n += 1
                got_here += 1
                if n % 50 == 0:
                    print(f"  {n}/{args.n}", flush=True)
        except Exception as e:
            print(f"[{repo}] stream error after {got_here}: {str(e)[:100]}", flush=True)
        print(f"[{repo}] contributed {got_here}", flush=True)
    print(f"DONE: {n} memes", flush=True)


if __name__ == "__main__":
    main()
