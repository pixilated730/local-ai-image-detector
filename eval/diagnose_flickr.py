"""Isolate WHY modern consumer Flickr photos score as AI (60.7% FPR @0.65).

Hypothesis: downscaling huge originals to <=1024px strips sensor noise, making photos
look "too clean". If true, preprocessing views that preserve native pixels (center
crops without resizing) should rescue reals; if false, the photos are intrinsically
AI-looking to this model and only training fixes it.

Scores every view for reals (flickr / coco / voc / openfake) AND fakes (proxy+holdout),
because any fix must keep fake recall. Reports per-source flag-rate @0.65 (calibrated
+7.6897) per view, plus simple logit-fusion combos.

Usage: python diagnose_flickr.py
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import NORM_MEAN, NORM_STD  # noqa: E402

OFF = 7.6897
THRESH = 0.65
CROP = 384

SETS = {
    "flickr(real)": [Path("data/reals_broad/flickr_consumer/real")],
    "coco(real)": [Path("data/reals_broad/coco/real")],
    "voc(real)": [Path("data/reals_broad/voc2012/real")],
    "openfake(real)": [Path("data/proxy/real"), Path("data/holdout/real")],
    "fakes": [Path("data/proxy/fake"), Path("data/holdout/fake")],
}


def to_tensor(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - NORM_MEAN) / NORM_STD
    return arr.transpose(2, 0, 1)


def center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def resize_shorter(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    s = target / min(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)


def views(img: Image.Image) -> dict[str, np.ndarray]:
    """Each view is one 384x384 tensor."""
    out = {}
    # standard eval view (what ships today)
    out["std_440_384"] = to_tensor(center_crop(resize_shorter(img, 440), CROP))
    # fit whole image into 384 (max semantic context, max detail loss)
    out["fit_384"] = to_tensor(center_crop(resize_shorter(img, 384), CROP))
    # native-resolution center crop: zero resampling, preserves whatever noise remains
    if min(img.size) >= CROP:
        out["native_crop"] = to_tensor(center_crop(img, CROP))
    else:
        out["native_crop"] = out["std_440_384"]
    return out


def main():
    sess = ort.InferenceSession(
        "models/commfor_vits_384.onnx", providers=["CPUExecutionProvider"]
    )

    results = {}  # set_name -> view_name -> np.array of logits
    for name, dirs in SETS.items():
        files = [f for d in dirs for f in sorted(d.glob("*.jpg"))]
        per_view: dict[str, list] = {}
        batch: dict[str, list] = {}

        def flush():
            for v, tensors in batch.items():
                if not tensors:
                    continue
                lg = sess.run(None, {"pixel_values": np.stack(tensors)})[0].reshape(-1)
                per_view.setdefault(v, []).extend(lg.tolist())
                batch[v] = []

        for i, f in enumerate(files):
            try:
                img = Image.open(f).convert("RGB")
            except Exception:
                continue
            for v, t in views(img).items():
                batch.setdefault(v, []).append(t)
            if (i + 1) % 16 == 0:
                flush()
        flush()
        results[name] = {v: np.array(l) for v, l in per_view.items()}
        print(f"scored {name}: {len(files)} files", file=sys.stderr)

    view_names = ["std_440_384", "fit_384", "native_crop"]
    combos = {
        "mean(std,native)": lambda r: (r["std_440_384"] + r["native_crop"]) / 2,
        "mean(all3)": lambda r: (r["std_440_384"] + r["fit_384"] + r["native_crop"]) / 3,
        "min(std,native)": lambda r: np.minimum(r["std_440_384"], r["native_crop"]),
    }

    header = ["set"] + view_names + list(combos)
    print("\nflag-rate @0.65 (calibrated). For 'fakes' this is RECALL (higher better);")
    print("for real sets it is FPR (lower better).\n")
    print(" | ".join(f"{h:18s}" for h in header))
    for name, r in results.items():
        row = [f"{name:18s}"]
        for v in view_names:
            p = 1 / (1 + np.exp(-(r[v] + OFF)))
            row.append(f"{(p >= THRESH).mean():18.1%}")
        for fn in combos.values():
            p = 1 / (1 + np.exp(-(fn(r) + OFF)))
            row.append(f"{(p >= THRESH).mean():18.1%}")
        print(" | ".join(row))

    # Median logits — shows how far distributions move between views.
    print("\nmedian raw logit per view:")
    for name, r in results.items():
        meds = "  ".join(f"{v}:{np.median(r[v]):+7.2f}" for v in view_names)
        print(f"  {name:18s} {meds}")


if __name__ == "__main__":
    main()
