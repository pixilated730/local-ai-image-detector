"""Build a web-realistic proxy benchmark by streaming from public datasets.

Streams real and fake images from ComplexDataLab/OpenFake (modern generators: Flux,
Ideogram, Imagen, GPT-Image-1, SDXL + LAION-sourced reals), applies a uniform
"web-realistic" re-encode (downscale to <=1024px, JPEG q85) to BOTH classes so the
benchmark can't be solved by format cues, and writes:

    data/proxy/{real,fake}/...jpg      (dev split — for calibration)
    data/holdout/{real,fake}/...jpg    (holdout — final go/no-go only)

Usage: python build_proxy.py [--per-class 500] [--holdout-frac 0.3] [--seed 7]
"""

import argparse
import io
import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image

MAX_SIDE = 1024
JPEG_Q = 85


def web_realistic(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    # Round-trip through JPEG to simulate web delivery.
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=JPEG_Q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    for split in ("proxy", "holdout"):
        for cls in ("real", "fake"):
            (out / split / cls).mkdir(parents=True, exist_ok=True)

    print("Streaming ComplexDataLab/OpenFake (core/test) ...")
    ds = load_dataset("ComplexDataLab/OpenFake", "core", split="test", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=500)

    counts = {"real": 0, "fake": 0}
    gen_counts = {}
    target = args.per_class

    for row in ds:
        label = str(row.get("label", "")).lower()
        if label not in ("real", "fake"):
            continue
        if counts[label] >= target:
            if all(v >= target for v in counts.values()):
                break
            continue

        # Keep generator diversity for fakes: cap per-generator share.
        model = str(row.get("model", "unknown"))
        if label == "fake":
            cap = max(25, target // 8)
            if gen_counts.get(model, 0) >= cap:
                continue

        img = row.get("image")
        if img is None:
            continue
        try:
            img = web_realistic(img)
        except Exception:
            continue

        split = "holdout" if rng.random() < args.holdout_frac else "proxy"
        idx = counts[label]
        safe_model = "".join(c if c.isalnum() or c in "-_" else "_" for c in model)[:40]
        name = f"{idx:05d}_{safe_model}.jpg" if label == "fake" else f"{idx:05d}.jpg"
        img.save(out / split / label / name, "JPEG", quality=JPEG_Q)

        counts[label] += 1
        if label == "fake":
            gen_counts[model] = gen_counts.get(model, 0) + 1
        done = counts["real"] + counts["fake"]
        if done % 100 == 0:
            print(f"  real={counts['real']} fake={counts['fake']}")

    print(f"Done: real={counts['real']} fake={counts['fake']}")
    print("Fake generator distribution:")
    for k, v in sorted(gen_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
