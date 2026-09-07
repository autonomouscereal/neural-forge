#!/usr/bin/env python3
"""Genuine CPU FPS ABI canary only; no models, GPU execution or capability proof."""
import importlib.metadata
import json
import platform
import sys


def main():
    import torch
    import torch_cluster

    if sys.version_info[:2] != (3, 10) or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("reviewed Linux x86_64 CPython 3.10 runtime required")
    if sys.platform != "linux" or torch.__version__ != "2.6.0+cu124" or torch.version.cuda != "12.4":
        raise RuntimeError("reviewed Torch 2.6.0+cu124 ABI required")
    version = importlib.metadata.version("torch-cluster")
    if version != "1.6.3+pt26cu124" or torch_cluster.fps.__module__ != "torch_cluster.fps":
        raise RuntimeError("genuine pinned torch-cluster implementation required")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only canary must not initialize CUDA")
    # No distance ties at the selected steps. Exercise the exact six-argument
    # API called by native HY3D as well as the registered compiled operator.
    points = torch.tensor([[0., 0, 0], [1, 0, 0], [4, 0, 0], [10, 0, 0],
                           [20, 0, 0], [22, 0, 0], [27, 0, 0], [35, 0, 0]], device="cpu")
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long, device="cpu")
    expected = torch.tensor([0, 3, 4, 7], dtype=torch.long, device="cpu")
    observed = torch_cluster.fps(points, batch, 0.5, False, 2, None)
    compiled = torch.ops.torch_cluster.fps(points, torch.tensor([0, 4, 8]), torch.tensor(0.5), False)
    if observed.device.type != "cpu" or observed.dtype != torch.int64 or not torch.equal(observed, expected):
        raise RuntimeError("native CPU FPS output disagrees with exact canary")
    if not torch.equal(observed, compiled) or torch.cuda.is_initialized():
        raise RuntimeError("compiled FPS/CPU-only execution check failed")
    print(json.dumps({"schema": "neural-forge.hy3d-fps-cpu-canary.v1", "torch": torch.__version__,
                      "torch_cluster": version, "device": "cpu", "indices": observed.tolist(),
                      "gpu_tested": False, "native_vae_tested": False, "folded_child_proven": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
