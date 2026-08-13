"""Re-derive the calibration offset over a MIXED real-image distribution.

The original offset (+7.6897) was fit with reals drawn only from OpenFake. Broadening
the real side (COCO / VOC / Flickr consumer photos) showed that offset pushes many
ordinary photos over the 0.65 line — Flickr's high-resolution consumer uploads scored
60.7% FPR. This script pools all real sources with the OpenFake fakes, splits dev/test
by hash of filename, fits the offset on dev, and reports test metrics.

Usage: python recalibrate.py
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import balanced_accuracy, auc_score, preprocess  # noqa: E402

MODEL = "models/commfor_vits_384.onnx"
THRESHOLD = 0.65
BATCH = 16

REAL_DIRS = [
    ("openfake", Path("data/proxy/real")),
    ("openfake", Path("data/holdout/real")),
    ("coco", Path("data/reals_broad/coco/real")),
    ("voc2012", Path("data/reals_broad/voc2012/real")),
    ("flickr", Path("data/reals_broad/flickr_consumer/real")),
]
FAKE_DIRS = [
    ("openfake", Path("data/proxy/fake")),
    ("openfake", Path("data/holdout/fake")),
]


def is_dev(path: Path) -> bool:
    """Stable 60/40 dev/test split by filename hash (not by source)."""
    return int(hashlib.sha256(path.name.encode()).hexdigest(), 16) % 10 < 6


def score_all(sess, files):
    out = []
    for i in range(0, len(files), BATCH):
        chunk = files[i : i + BATCH]
        arr = np.stack([preprocess(f, 440, 384) for f in chunk])
        out.extend(sess.run(None, {"pixel_values": arr})[0].reshape(-1).tolist())
    return out


def main():
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])

    rows = []  # (source, label, logit, is_dev)
    for label, dirs in ((0, REAL_DIRS), (1, FAKE_DIRS)):
        for source, d in dirs:
            files = sorted(d.glob("*.jpg"))
            if not files:
                continue
            print(f"scoring {source} {d} ({len(files)}) ...")
            for f, lg in zip(files, score_all(sess, files)):
                rows.append((source, label, lg, is_dev(f)))

    src = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    lg = np.array([r[2] for r in rows])
    dev = np.array([r[3] for r in rows])

    # Fit offset on dev: choose b maximising balanced accuracy at the fixed threshold.
    target_logit = np.log(THRESHOLD / (1 - THRESHOLD))
    grid = np.linspace(-2, 14, 1601)
    best_b, best_dev = 0.0, -1
    for b in grid:
        p = 1 / (1 + np.exp(-(lg[dev] + b)))
        ba = balanced_accuracy(y[dev], p, THRESHOLD)
        if ba > best_dev:
            best_dev, best_b = ba, b

    p_test = 1 / (1 + np.exp(-(lg[~dev] + best_b)))
    print("\n=== Recalibrated on mixed real distribution ===")
    print(f"offset (dev-fitted)      : {best_b:+.4f}   (was +7.6897, OpenFake-only reals)")
    print(f"dev  balanced acc @0.65  : {best_dev:.4f}")
    print(f"test balanced acc @0.65  : {balanced_accuracy(y[~dev], p_test, THRESHOLD):.4f}")
    print(f"test AUC                 : {auc_score(y[~dev], p_test):.4f}")
    print(f"test real acc            : {(p_test[y[~dev] == 0] < THRESHOLD).mean():.4f}")
    print(f"test fake acc            : {(p_test[y[~dev] == 1] >= THRESHOLD).mean():.4f}")

    print("\nPer-source FPR @0.65 (test split, old vs new offset):")
    for s in sorted(set(src[y == 0])):
        m = (src == s) & (y == 0) & ~dev
        if not m.any():
            continue
        old = (1 / (1 + np.exp(-(lg[m] + 7.6897))) >= THRESHOLD).mean()
        new = (1 / (1 + np.exp(-(lg[m] + best_b))) >= THRESHOLD).mean()
        print(f"  {s:10s} n={m.sum():4d}   old {old:6.1%}  ->  new {new:6.1%}")

    m = (y == 1) & ~dev
    old = (1 / (1 + np.exp(-(lg[m] + 7.6897))) >= THRESHOLD).mean()
    new = (1 / (1 + np.exp(-(lg[m] + best_b))) >= THRESHOLD).mean()
    print(f"\nFake recall (test): old {old:.1%}  ->  new {new:.1%}")


if __name__ == "__main__":
    main()
