"""Bake a refit head into a full classifier ONNX and (optionally) install it into the
extension with its calibration offset.

Usage:
  python bake_head.py --head models/head_384_refit.npz --offset 3.49 --install
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "vendor" / "Community-Forensics"))
from models import ViTClassifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="models/head_384_refit.npz")
    ap.add_argument("--out", default="models/commfor_vits_384_refit.onnx")
    ap.add_argument("--offset", type=float, required=True)
    ap.add_argument("--install", action="store_true", help="copy into extension + set offset")
    args = ap.parse_args()

    d = np.load(ROOT / args.head)
    w, b = d["w"].reshape(1, -1), d["b"].reshape(-1)

    model = ViTClassifier.from_pretrained("OwensLab/commfor-model-384", device="cpu")
    model.eval()
    with torch.no_grad():
        model.vit.head.weight.copy_(torch.from_numpy(w))
        model.vit.head.bias.copy_(torch.from_numpy(b))

    out = ROOT / args.out
    torch.onnx.export(
        model, torch.randn(1, 3, 384, 384), str(out),
        input_names=["pixel_values"], output_names=["logit"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    # Parity: ONNX logit must equal features @ w + b.
    import onnxruntime as ort
    fs = ort.InferenceSession(str(ROOT / "models/commfor_vits_384_features.onnx"),
                              providers=["CPUExecutionProvider"])
    cs = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = np.random.randn(2, 3, 384, 384).astype(np.float32)
    ref = fs.run(None, {"pixel_values": x})[0] @ w.reshape(-1) + b[0]
    got = cs.run(None, {"pixel_values": x})[0].reshape(-1)
    diff = float(np.abs(ref - got).max())
    print(f"baked {out.name}: parity diff {diff:.2e}")
    assert diff < 1e-3

    if args.install:
        ext = ROOT.parent / "extension"
        import shutil
        shutil.copy(out, ext / "models" / "detector.onnx")
        engine = ext / "src" / "engine.js"
        src = engine.read_text(encoding="utf-8")
        src, n = re.subn(r"const CALIBRATION_OFFSET = [-\d.]+;",
                         f"const CALIBRATION_OFFSET = {args.offset};", src)
        assert n == 1, "CALIBRATION_OFFSET line not found"
        engine.write_text(src, encoding="utf-8")
        print(f"installed into extension: detector.onnx + CALIBRATION_OFFSET={args.offset}")


if __name__ == "__main__":
    main()
