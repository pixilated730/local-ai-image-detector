"""Design + validate a 'graphic/screenshot' gate that separates non-photographic
images (charts, tables, UI screenshots, text) from photos and generated art.

Motivation: the detector's scores on charts/UI screenshots are out-of-distribution
noise — near-identical infographics score 58% vs 78%, and flat promo graphics can hit
96% "AI generated". The gate detects such graphics from simple pixel statistics and
lets the pipeline treat them as a distinct class instead of pretending the photo
detector's output is meaningful there.

Features (computed on the same 384x384 crop the model sees):
  flat  — fraction of pixels identical to their right neighbor (flat fills)
  colors— number of distinct quantized colors (5 bits/channel) / 1024
Rendered synthetic positives: matplotlib charts, tables, text pages, UI mockups.
Negatives: our real-photo and generated-image eval sets.

Usage: python graphic_gate.py
"""

import io
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import preprocess  # noqa: E402  (used for the same resize path)


# ---------------------------------------------------------------- gate features
def gate_features(img: Image.Image) -> dict:
    """Same crop geometry as the model input; cheap integer stats."""
    im = img.convert("RGB")
    w, h = im.size
    s = 384 / min(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    w, h = im.size
    left, top = (w - 384) // 2, (h - 384) // 2
    im = im.crop((left, top, left + 384, top + 384))
    a = np.asarray(im, dtype=np.uint8)

    flat = float((a[:, 1:] == a[:, :-1]).all(axis=2).mean())
    q = (a >> 3).astype(np.uint32)
    codes = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    colors = len(np.unique(codes)) / 1024.0
    return {"flat": flat, "colors": colors}


def is_graphic(feat: dict) -> bool:
    # Graphics: flat p5=0.71; photos: flat p95=0.53, fakes p95=0.48 (measured below).
    # flat>0.62 splits the distributions; the color term catches dark low-palette UI.
    return feat["flat"] > 0.62 or (feat["flat"] > 0.45 and feat["colors"] < 0.08)


# ---------------------------------------------------------------- synthetic graphics
def render_graphics(n=60, seed=3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind = i % 4
        if kind == 0:  # bar chart
            fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
            k = rng.randint(4, 9)
            ax.bar(range(k), [rng.random() for _ in range(k)],
                   color=rng.choice(["#6366f1", "#333", "#999", "#0ea5e9"]))
            ax.set_title("Evaluation results")
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            out.append(Image.open(buf).convert("RGB"))
        elif kind == 1:  # line chart
            fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
            for _ in range(rng.randint(1, 4)):
                ax.plot(np.cumsum(np.random.randn(50)))
            ax.grid(True)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            out.append(Image.open(buf).convert("RGB"))
        elif kind == 2:  # table / text page
            im = Image.new("RGB", (900, 700), rng.choice(["#fff", "#0d1117", "#f6f8fa"]))
            d = ImageDraw.Draw(im)
            fg = "#111" if im.getpixel((0, 0))[0] > 128 else "#e6edf3"
            for r in range(rng.randint(8, 16)):
                y = 40 + r * 40
                d.line([(30, y + 26), (870, y + 26)], fill="#8884", width=1)
                for c in range(4):
                    d.text((40 + c * 210, y), f"row{r} col{c} {rng.randint(0, 999)}", fill=fg)
            out.append(im)
        else:  # UI mockup: flat panels + buttons
            im = Image.new("RGB", (1000, 700), rng.choice(["#000", "#111827", "#ffffff"]))
            d = ImageDraw.Draw(im)
            for _ in range(rng.randint(3, 8)):
                x0, y0 = rng.randint(0, 600), rng.randint(0, 400)
                d.rounded_rectangle([x0, y0, x0 + rng.randint(150, 380), y0 + rng.randint(60, 260)],
                                    radius=12, fill=rng.choice(["#1f2937", "#3b82f6", "#e5e7eb", "#374151"]))
            d.text((60, 40), "Under the hood — aggregate stats on labels", fill="#888")
            out.append(im)
    # web-realistic: jpeg round trip like everything else
    final = []
    for im in out:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        buf.seek(0)
        final.append(Image.open(buf).convert("RGB"))
    return final


def main():
    print("rendering synthetic graphics ...")
    graphics = render_graphics(80)
    gfeat = [gate_features(im) for im in graphics]

    sets = {
        "real photos": [*Path("data/reals_broad/coco/real").glob("*.jpg")][:100]
        + [*Path("data/unsplash/test/real").glob("*.jpg")][:100],
        "generated (fakes)": [*Path("data/proxy/fake").glob("*.jpg")][:150]
        + [*Path("data/holdout/fake").glob("*.jpg")][:80],
    }

    def rate(feats):
        return np.mean([is_graphic(f) for f in feats])

    print(f"graphics    : gate fires {rate(gfeat):6.1%}  (want high)")
    for name, files in sets.items():
        feats = [gate_features(Image.open(f)) for f in files]
        flats = np.array([f["flat"] for f in feats])
        print(f"{name:12s}: gate fires {rate(feats):6.1%}  (want ~0)   flat p95={np.percentile(flats, 95):.3f}")

    fl = np.array([f["flat"] for f in gfeat])
    co = np.array([f["colors"] for f in gfeat])
    print(f"\ngraphics feature ranges: flat p5={np.percentile(fl, 5):.3f} median={np.median(fl):.3f}; "
          f"colors median={np.median(co):.3f}")


if __name__ == "__main__":
    main()
