"""Assemble the head-refit training pool. STRICTLY disjoint from all eval sets.

Eval sets (never touched here): OpenFake core/TEST (proxy+holdout), reals_broad
(COCO val / VOC val / original flickr_consumer), flickr_dated/test_pre2022.

Training pool sources:
  fakes: OpenFake core/VALIDATION split, generator-capped
  reals: OpenFake core/VALIDATION reals + COCO TRAIN split + VOC TRAIN split
         (+ flickr_dated/train_pre2022, fetched separately by fetch_flickr.py)

Same web-realistic re-encode as everywhere else.

Usage: python build_trainpool.py [--fakes 2500] [--coco 1000] [--voc 500] [--of-reals 1000]
"""

import argparse
import io
from pathlib import Path

from datasets import load_dataset
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


def save_stream(ds, dest: Path, n_target: int, keep, name_fn, log_tag: str):
    dest.mkdir(parents=True, exist_ok=True)
    n = len(list(dest.glob("*.jpg")))
    if n >= n_target:
        print(f"[{log_tag}] already {n}, skip", flush=True)
        return
    for row in ds:
        if n >= n_target:
            break
        if not keep(row):
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
            web_realistic(img).save(dest / name_fn(n, row), "JPEG", quality=JPEG_Q)
        except Exception:
            continue
        n += 1
        if n % 100 == 0:
            print(f"  [{log_tag}] {n}/{n_target}", flush=True)
    print(f"[{log_tag}] done: {n}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fakes", type=int, default=2500)
    ap.add_argument("--of-reals", type=int, default=1000)
    ap.add_argument("--coco", type=int, default=1000)
    ap.add_argument("--voc", type=int, default=500)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent / "data" / "trainpool"

    # --- OpenFake validation split: fakes (generator-capped) and reals ---
    of = load_dataset("ComplexDataLab/OpenFake", "core", split="validation", streaming=True)
    of = of.shuffle(seed=11, buffer_size=500)
    gen_counts: dict[str, int] = {}
    cap = max(50, args.fakes // 10)

    def keep_fake(row):
        if str(row.get("label", "")).lower() != "fake":
            return False
        m = str(row.get("model", "unknown"))
        if gen_counts.get(m, 0) >= cap:
            return False
        gen_counts[m] = gen_counts.get(m, 0) + 1
        return True

    def fake_name(n, row):
        m = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(row.get("model", "u")))[:40]
        return f"{n:05d}_{m}.jpg"

    save_stream(of, root / "fake", args.fakes, keep_fake, fake_name, "of-fakes")

    of2 = load_dataset("ComplexDataLab/OpenFake", "core", split="validation", streaming=True)
    of2 = of2.shuffle(seed=12, buffer_size=500)
    save_stream(
        of2, root / "real_openfake", args.of_reals,
        lambda r: str(r.get("label", "")).lower() == "real",
        lambda n, r: f"{n:05d}.jpg", "of-reals",
    )

    # --- COCO train / VOC train reals ---
    coco = load_dataset("detection-datasets/coco", split="train", streaming=True).shuffle(seed=13, buffer_size=200)
    save_stream(coco, root / "real_coco", args.coco, lambda r: True, lambda n, r: f"{n:05d}.jpg", "coco")

    voc = load_dataset("nateraw/pascal-voc-2012", split="train", streaming=True).shuffle(seed=14, buffer_size=200)
    save_stream(voc, root / "real_voc", args.voc, lambda r: True, lambda n, r: f"{n:05d}.jpg", "voc")


if __name__ == "__main__":
    main()
