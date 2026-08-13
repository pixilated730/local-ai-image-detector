"""Export the Community Forensics ViT-S detector to ONNX for in-browser inference.

Loads OwensLab/commfor-model-224 (MIT-licensed weights, PytorchModelHubMixin),
exports fp32 ONNX + int8 dynamically-quantized ONNX, and verifies numerical parity
between PyTorch and ONNX Runtime on random inputs.

Usage: python export_onnx.py [--repo OwensLab/commfor-model-224] [--out ../extension/models]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Vendored Community Forensics repo provides the model class.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor" / "Community-Forensics"))
from models import ViTClassifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="OwensLab/commfor-model-224")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "models"))
    ap.add_argument("--input-size", type=int, default=224)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / f"commfor_vits_{args.input_size}.onnx"
    int8_path = out_dir / f"commfor_vits_{args.input_size}_int8.onnx"

    print(f"Loading {args.repo} ...")
    model = ViTClassifier.from_pretrained(args.repo, device="cpu")
    model.eval()

    example = torch.randn(1, 3, args.input_size, args.input_size)

    print(f"Exporting fp32 ONNX -> {fp32_path}")
    torch.onnx.export(
        model,
        example,
        str(fp32_path),
        input_names=["pixel_values"],
        output_names=["logit"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    print("Verifying fp32 parity (torch vs onnxruntime)...")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    max_diff = 0.0
    for _ in range(5):
        x = torch.randn(2, 3, args.input_size, args.input_size)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"pixel_values": x.numpy()})[0]
        max_diff = max(max_diff, float(np.abs(ref - got).max()))
    print(f"  max |logit diff| fp32: {max_diff:.6f}")
    assert max_diff < 1e-3, "fp32 export diverges from PyTorch"

    print(f"Quantizing (dynamic int8) -> {int8_path}")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    sess8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    diffs = []
    for _ in range(5):
        x = torch.randn(2, 3, args.input_size, args.input_size)
        with torch.no_grad():
            ref = torch.sigmoid(model(x)).numpy()
        got = 1 / (1 + np.exp(-sess8.run(None, {"pixel_values": x.numpy()})[0]))
        diffs.append(float(np.abs(ref - got).max()))
    print(f"  max |prob diff| int8 (random inputs): {max(diffs):.4f}")

    for p in (fp32_path, int8_path):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")
    print("Done. Note: judge int8 quality on real images via harness.py, not random inputs.")


if __name__ == "__main__":
    main()
