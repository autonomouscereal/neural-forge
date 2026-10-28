from pathlib import Path

from safetensors import safe_open


path = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M/model.safetensors")
with safe_open(path, framework="pt", device="cpu") as handle:
    for key in handle.keys():
        if "feed_forward" not in key and ".mlp." not in key:
            continue
        tensor = handle.get_slice(key)
        print(f"{key}\t{tuple(tensor.get_shape())}")
