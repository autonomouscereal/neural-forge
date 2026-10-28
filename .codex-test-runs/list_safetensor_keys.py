from __future__ import annotations

import json
import sys

from safetensors import safe_open


for value in sys.argv[1:]:
    with safe_open(value, framework="pt", device="cpu") as handle:
        rows = []
        for key in handle.keys():
            tensor = handle.get_slice(key)
            rows.append({"key": key, "shape": list(tensor.get_shape())})
    print(json.dumps({"path": value, "tensors": rows}, sort_keys=True))
