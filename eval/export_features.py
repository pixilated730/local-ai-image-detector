"""Export the CommFor ViT-S backbone as a 384-d feature extractor (head removed),
plus the original head weights (w, b) as npz.

Used by refit_head.py: features are extracted once on CPU, then a regularized logistic
head is fit in seconds. logit == features @ w + b reproduces the shipped classifier.

Usage: python export_features.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor" / "Community-Forensics"))
from models import ViTClassifier  # noqa: E402


def main():
    out_dir = Path(__file__).resolve().parent / "models"
    model = ViTClassifier.from_pretrained("OwensLab/commfor-model-384", device="cpu")
    model.eval()

    head = model.vit.head
    np.savez(
        out_dir / "head_384.npz",
        w=head.weight.detach().numpy().reshape(-1),  # (384,)
        b=head.bias.detach().numpy().reshape(-1),    # (1,)
    )

    model.vit.head = torch.nn.Identity()
    example = torch.randn(1, 3, 384, 384)
    feat_path = out_dir / "commfor_vits_384_features.onnx"
    torch.onnx.export(
        model, example, str(feat_path),
        input_names=["pixel_values"], output_names=["features"],
        dynamic_axes={"pixel_values": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    # Sanity: features @ w + b must reproduce the classifier logit.
    import onnxruntime as ort
    d = np.load(out_dir / "head_384.npz")
    fs = ort.InferenceSession(str(feat_path), providers=["CPUExecutionProvider"])
    cs = ort.InferenceSession(str(out_dir / "commfor_vits_384.onnx"), providers=["CPUExecutionProvider"])
    x = np.random.randn(2, 3, 384, 384).astype(np.float32)
    feats = fs.run(None, {"pixel_values": x})[0]
    recon = feats @ d["w"] + d["b"]
    ref = cs.run(None, {"pixel_values": x})[0].reshape(-1)
    print("feature dim:", feats.shape, "| max |recon - ref| =", float(np.abs(recon - ref).max()))
    assert np.abs(recon - ref).max() < 1e-3


if __name__ == "__main__":
    main()
