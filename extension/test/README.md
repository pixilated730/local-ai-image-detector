# Extension test pages

Serve the extension folder with correct MIME types, then open the pages:

```bash
python serve.py 8124 --dir ..
# http://localhost:8124/test/parity.html    — browser vs Python preprocessing parity
# http://localhost:8124/test/demo.html      — feed demo driving the real content script
# http://localhost:8124/test/hardcases.html — discovery coverage (bg/shadow/blob/iframe/...)
```

`img/` and `expected.json` are not committed (benchmark-derived images). Regenerate
after building the eval proxy set (`../../eval/build_proxy.py`):

```bash
cd ../../eval && python - <<'EOF'
import sys, json, shutil
from pathlib import Path
import numpy as np, onnxruntime as ort
sys.path.insert(0, ".")
from harness import preprocess
src, out = Path("data/proxy"), Path("../extension/test/img")
out.mkdir(parents=True, exist_ok=True)
picks = [(f, c) for c in ("real", "fake") for f in sorted((src/c).glob("*.jpg"))[:8]]
sess = ort.InferenceSession("models/commfor_vits_384_refit.onnx", providers=["CPUExecutionProvider"])
exp = []
for f, c in picks:
    name = f"{c}_{f.name}"; shutil.copy(f, out/name)
    lg = float(sess.run(None, {"pixel_values": preprocess(f, 440, 384)[None]})[0].item())
    exp.append({"file": name, "label": c, "logit": round(lg, 6), "prob": round(float(1/(1+np.exp(-lg))), 6)})
json.dump(exp, open("../extension/test/expected.json", "w"), indent=1)
print("done", len(exp))
EOF
```
