"""Re-fit the classifier head on frozen CommFor features to fix the modern-consumer-
photo false-positive problem, without touching the backbone.

Objective: class-weighted BCE + lambda * ||w - w0||^2, where (w0, b0) is the original
head — the regularizer keeps the new boundary close to the shipped one so generator
knowledge encoded in feature space is preserved. Pure numpy (Adam, full batch).

Data hygiene:
  train   : data/trainpool/* + data/flickr_dated/train_pre2022 (all disjoint from eval)
  select  : 20% split of train (lambda choice)
  report  : untouched eval sets (proxy/holdout, reals_broad, flickr_dated/test_pre2022)

Outputs models/head_384_refit.npz and a full classifier models/commfor_vits_384_refit.onnx.

Usage: python refit_head.py [--skip-extract]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import preprocess, balanced_accuracy, auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
CACHE = ROOT / "data" / "feature_cache"
THRESH = 0.65

TRAIN_SETS = {
    # tag: (dir, label, source-group for real balancing)
    "tp_fake": ("data/trainpool/fake", 1, "fake"),
    "tp_real_of": ("data/trainpool/real_openfake", 0, "openfake"),
    "tp_real_coco": ("data/trainpool/real_coco", 0, "coco"),
    "tp_real_voc": ("data/trainpool/real_voc", 0, "voc"),
    "tp_real_flickr": ("data/flickr_dated/train_pre2022/real", 0, "modern"),
    "tp_real_commons": ("data/commons/train/real", 0, "modern"),
    "tp_real_unsplash": ("data/unsplash/train/real", 0, "modern"),
}
EVAL_SETS = {
    "eval_fake": (["data/proxy/fake", "data/holdout/fake"], 1),
    "eval_real_of": (["data/proxy/real", "data/holdout/real"], 0),
    "eval_real_coco": (["data/reals_broad/coco/real"], 0),
    "eval_real_voc": (["data/reals_broad/voc2012/real"], 0),
    "eval_real_commons": (["data/commons/test/real"], 0),
    "eval_real_unsplash": (["data/unsplash/test/real"], 0),
    "eval_real_flickr_pre2022": (["data/flickr_dated/test_pre2022/real"], 0),
    # NOT eval-real: 2022-scrape flickr contains AI uploads; keep for analysis only
    "flickr2022_contaminated": (["data/reals_broad/flickr_consumer/real"], -1),
    "post2022_unknown": (["data/flickr_dated/post2022/unknown"], -1),
    # OOD check: CommForensics generators are disjoint from OpenFake's — guards against
    # the refit narrowing to OpenFake-style fakes.
    "ood_commfor_fake": (["data/commfor_small/fake"], -1),
    "ood_commfor_real": (["data/commfor_small/real"], -1),
}


def extract(dirs, tag, sess):
    """Feature-extract a set of dirs with an npz cache keyed by tag + file count."""
    if isinstance(dirs, str):
        dirs = [dirs]
    files = [f for d in dirs for f in sorted((ROOT / d).glob("*.jpg"))]
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"{tag}_{len(files)}.npz"
    if cache_file.exists():
        return np.load(cache_file)["x"]
    feats, batch = [], []
    for i, f in enumerate(files):
        try:
            batch.append(preprocess(f, 440, 384))
        except Exception:
            continue
        if len(batch) == 16:
            feats.append(sess.run(None, {"pixel_values": np.stack(batch)})[0])
            batch = []
        if (i + 1) % 320 == 0:
            print(f"    {tag}: {i + 1}/{len(files)}", flush=True)
    if batch:
        feats.append(sess.run(None, {"pixel_values": np.stack(batch)})[0])
    x = np.concatenate(feats) if feats else np.zeros((0, 384), np.float32)
    np.savez_compressed(cache_file, x=x)
    return x


def fit_head(X, y, sw, w0, b0, lam, steps=4000, lr=0.01):
    """Full-batch Adam on weighted BCE + lam*||w-w0||^2, starting from (w0, b0)."""
    w, b = w0.copy(), float(b0)
    mw = np.zeros_like(w); vw = np.zeros_like(w); mb = vb = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        g = (p - y) * sw                      # d(weighted BCE)/dz
        gw = X.T @ g / len(y) + 2 * lam * (w - w0)
        gb = g.mean()
        mw = b1 * mw + (1 - b1) * gw; vw = b2 * vw + (1 - b2) * gw**2
        mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb**2
        w -= lr * (mw / (1 - b1**t)) / (np.sqrt(vw / (1 - b2**t)) + eps)
        b -= lr * (mb / (1 - b1**t)) / (np.sqrt(vb / (1 - b2**t)) + eps)
    return w, b


def best_ba(y, logits):
    p = 1 / (1 + np.exp(-logits))
    ts = np.unique(np.concatenate([p, [0.5]]))
    return max(balanced_accuracy(y, p, t) for t in ts)


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    fs = ort.InferenceSession(str(MODELS / "commfor_vits_384_features.onnx"),
                              providers=["CPUExecutionProvider"])
    d = np.load(MODELS / "head_384.npz")
    w0, b0 = d["w"].astype(np.float64), float(d["b"][0])

    # ---- extract features ----
    print("extracting features (cached after first run) ...")
    train_x, train_y, train_src = [], [], []
    for tag, (dir_, label, src) in TRAIN_SETS.items():
        x = extract(dir_, tag, fs)
        print(f"  {tag}: {len(x)}")
        train_x.append(x); train_y += [label] * len(x); train_src += [src] * len(x)
    X = np.concatenate(train_x).astype(np.float64)
    y = np.array(train_y, np.float64)
    src = np.array(train_src)

    ev = {}
    for tag, (dirs, label) in EVAL_SETS.items():
        ev[tag] = (extract(dirs, tag, fs).astype(np.float64), label)
        print(f"  {tag}: {len(ev[tag][0])}")

    # ---- sample weights: classes balanced; real sources balanced within the class ----
    sw = np.ones(len(y))
    n_fake, n_real = (y == 1).sum(), (y == 0).sum()
    real_sources = [s for s in np.unique(src) if s != "fake"]
    for s in real_sources:
        m = src == s
        sw[m] = (n_real / len(real_sources)) / m.sum()   # equalize real sources
    sw[y == 1] *= 1.0
    sw[y == 0] *= n_fake / n_real                         # balance classes overall
    sw /= sw.mean()

    # ---- train/select split ----
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, va = idx[:cut], idx[cut:]

    print("\nlambda sweep (selection = balanced acc on 20% held-out train):")
    best = None
    for lam in [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        w, b = fit_head(X[tr], y[tr], sw[tr], w0, b0, lam)
        ba = best_ba(y[va], X[va] @ w + b)
        drift = float(np.linalg.norm(w - w0) / np.linalg.norm(w0))
        print(f"  lam={lam:7.4f}  val_bestBA={ba:.4f}  |w-w0|/|w0|={drift:.3f}")
        if best is None or ba > best[0]:
            best = (ba, lam, w, b)
    ba, lam, w, b = best
    print(f"selected lam={lam} (val bestBA {ba:.4f}); refitting on all train data")
    w, b = fit_head(X, y, sw, w0, b0, lam)

    np.savez(MODELS / "head_384_refit.npz", w=w.astype(np.float32), b=np.array([b], np.float32))

    # ---- evaluate old vs new on untouched eval sets ----
    def report(name, wv, bv):
        print(f"\n=== {name} ===")
        lg_fake = ev["eval_fake"][0] @ wv + bv
        # calibration offset: fit on pooled eval reals+fakes at THRESH (like recalibrate.py)
        real_tags = [t for t in ev if t.startswith("eval_real") and len(ev[t][0])]
        lg_reals = {t: ev[t][0] @ wv + bv for t in real_tags}
        pool_lg = np.concatenate([lg_fake] + list(lg_reals.values()))
        pool_y = np.concatenate([np.ones(len(lg_fake))] + [np.zeros(len(v)) for v in lg_reals.values()])
        target = np.log(THRESH / (1 - THRESH))
        offs = np.linspace(-4, 14, 1801)
        bas = [balanced_accuracy(pool_y, 1 / (1 + np.exp(-(pool_lg + o))), THRESH) for o in offs]
        o = float(offs[int(np.argmax(bas))])
        print(f"offset={o:+.3f}  pooled BA@0.65={max(bas):.4f}  AUC={auc_score(pool_y, pool_lg):.4f}")
        rec = ((1 / (1 + np.exp(-(lg_fake + o)))) >= THRESH).mean()
        print(f"  fake recall          : {rec:6.1%}")
        accs = []
        for t in real_tags:
            fpr = ((1 / (1 + np.exp(-(lg_reals[t] + o)))) >= THRESH).mean()
            accs.append(1 - fpr)
            print(f"  {t:26s} FPR: {fpr:6.1%}")
        print(f"  source-balanced BA   : {(rec + np.mean(accs)) / 2:.4f}")
        for t in ("post2022_unknown", "flickr2022_contaminated", "ood_commfor_fake", "ood_commfor_real"):
            if len(ev.get(t, ([],))[0]):
                un = ev[t][0] @ wv + bv
                print(f"  {t:26s} flagged: {((1 / (1 + np.exp(-(un + o)))) >= THRESH).mean():6.1%}")
        return o

    report("ORIGINAL head", w0, b0)
    off_new = report(f"REFIT head (lam={lam})", w, b)

    # High-regularization variant: stays closer to the original boundary — likely safer
    # on out-of-distribution fakes. Compare before choosing what to ship.
    w_hi, b_hi = fit_head(X, y, sw, w0, b0, 0.1)
    np.savez(MODELS / "head_384_refit_hi.npz", w=w_hi.astype(np.float32), b=np.array([b_hi], np.float32))
    off_hi = report("REFIT head (lam=0.1, low-drift)", w_hi, b_hi)

    print(f"\ncalibration offsets — selected: {off_new:+.4f} | low-drift: {off_hi:+.4f}")


if __name__ == "__main__":
    main()
