"""Evaluation harness for the AI-image detector.

Scores a directory tree of images with an ONNX detector and reports the metrics that
matter for the bounty: balanced accuracy at the fixed 0.65 threshold, the best
achievable balanced accuracy, and the calibration offset that moves the optimal
operating point to 0.65.

Data layout (labels from top-level subdirectory names):
    <data_dir>/real/**   -> label 0
    <data_dir>/fake/**   -> label 1

Usage:
    python harness.py --model models/commfor_vits_224.onnx --data data/smoke
    python harness.py --model models/commfor_vits_224_int8.onnx --data data/proxy --save-scores scores.csv
Optional: --apply-logit-offset <b> applies calibration p' = sigmoid(logit + b) before metrics.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path: Path, resize_size=256, crop_size=224) -> np.ndarray:
    """Match Community Forensics eval transforms: Resize(256) -> CenterCrop(224) -> normalize."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = resize_size / min(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - crop_size) // 2, (h - crop_size) // 2
    img = img.crop((left, top, left + crop_size, top + crop_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - NORM_MEAN) / NORM_STD
    return arr.transpose(2, 0, 1)  # HWC -> CHW


def collect(data_dir: Path):
    items = []
    for label_name, label in (("real", 0), ("fake", 1)):
        root = data_dir / label_name
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                items.append((p, label))
    return items


def balanced_accuracy(labels: np.ndarray, probs: np.ndarray, threshold: float) -> float:
    pred = probs >= threshold
    real, fake = labels == 0, labels == 1
    tnr = float((~pred[real]).mean()) if real.any() else float("nan")
    tpr = float(pred[fake].mean()) if fake.any() else float("nan")
    return (tpr + tnr) / 2


def auc_score(labels: np.ndarray, probs: np.ndarray) -> float:
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1)
    # average ties
    for v in np.unique(probs):
        m = probs == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=0.65, help="bounty threshold")
    ap.add_argument("--apply-logit-offset", type=float, default=0.0)
    ap.add_argument("--save-scores", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data)
    items = collect(data_dir)
    if not items:
        raise SystemExit(f"No images under {data_dir}/real or {data_dir}/fake")
    print(f"{len(items)} images ({sum(1 for _, l in items if l == 0)} real, "
          f"{sum(1 for _, l in items if l == 1)} fake)")

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    logits, labels, paths, failed = [], [], [], 0
    t0 = time.time()
    batch, batch_meta = [], []

    def flush():
        nonlocal batch, batch_meta
        if not batch:
            return
        out = sess.run(None, {input_name: np.stack(batch)})[0].reshape(-1)
        logits.extend(out.tolist())
        for p, l in batch_meta:
            paths.append(p)
            labels.append(l)
        batch, batch_meta = [], []

    for i, (path, label) in enumerate(items):
        try:
            batch.append(preprocess(path, args.resize, args.crop))
            batch_meta.append((path, label))
        except Exception:
            failed += 1
            continue
        if len(batch) >= args.batch:
            flush()
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(items)} ({(i + 1) / (time.time() - t0):.1f} img/s)")
    flush()

    labels = np.array(labels)
    logits = np.array(logits) + args.apply_logit_offset
    probs = 1 / (1 + np.exp(-logits))

    # Threshold sweep for the best balanced-accuracy operating point.
    candidates = np.unique(np.concatenate([probs, [args.threshold, 0.5]]))
    best_t, best_ba = max(
        ((t, balanced_accuracy(labels, probs, t)) for t in candidates), key=lambda x: x[1]
    )
    # Logit offset that moves the optimal operating point to the bounty threshold.
    eps = 1e-7
    offset = math.log(args.threshold / (1 - args.threshold)) - math.log(
        max(best_t, eps) / max(1 - best_t, eps)
    )

    report = {
        "model": args.model,
        "data": str(data_dir),
        "n_images": int(len(labels)),
        "n_failed": failed,
        "auc": round(auc_score(labels, probs), 4),
        "balanced_acc@0.50": round(balanced_accuracy(labels, probs, 0.5), 4),
        f"balanced_acc@{args.threshold:.2f}": round(
            balanced_accuracy(labels, probs, args.threshold), 4
        ),
        "best_balanced_acc": round(best_ba, 4),
        "best_threshold": round(float(best_t), 4),
        "calibration_logit_offset_to_0.65": round(offset, 4),
        "real_acc@best_t": round(float((probs[labels == 0] < best_t).mean()), 4),
        "fake_acc@best_t": round(float((probs[labels == 1] >= best_t).mean()), 4),
        "img_per_sec": round(len(labels) / (time.time() - t0), 2),
    }
    print(json.dumps(report, indent=2))

    if args.save_scores:
        with open(args.save_scores, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "prob_ai"])
            for p, l, pr in zip(paths, labels.tolist(), probs.tolist()):
                w.writerow([str(p), l, f"{pr:.6f}"])
        print(f"scores -> {args.save_scores}")


if __name__ == "__main__":
    main()
