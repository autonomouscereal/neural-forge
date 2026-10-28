"""Scratch micro-benchmark: per-tensor overhead inside _tensor_sha256."""

from __future__ import annotations

import hashlib
import json
import statistics
import time

import torch

import neural_forge.hf_lfm2_circuit_trace_cache as m


def _values(count: int, elements: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        return [torch.randn(1, 16, elements, dtype=torch.float32) for _ in range(count)]


def _time(call, repeats: int = 7):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started)
    return min(samples), statistics.median(samples)


def main() -> None:
    values = _values(480, 2048)
    total = sum(int(v.numel() * v.element_size()) for v in values)
    buffers = [v.reshape(-1).view(torch.uint8).numpy() for v in values]
    print(f"{len(values)} tensors, {total / 1e6:.1f} MB\n")

    for label, call in (
        ("full _tensor_sha256", lambda: [m._tensor_sha256(torch, v) for v in values]),
        (
            "hash only (precomputed buffer)",
            lambda: [hashlib.sha256(b).hexdigest() for b in buffers],
        ),
        (
            "hash + torch buffer chain",
            lambda: [
                hashlib.sha256(
                    v.detach().to(device="cpu").contiguous()
                    .reshape(-1).view(torch.uint8).numpy()
                ).hexdigest()
                for v in values
            ],
        ),
        (
            "hash + minimal buffer chain",
            lambda: [
                hashlib.sha256(
                    v.reshape(-1).view(torch.uint8).numpy()
                ).hexdigest()
                for v in values
            ],
        ),
        (
            "json.dumps(shape) only",
            lambda: [
                json.dumps(
                    [int(s) for s in v.shape],
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                for v in values
            ],
        ),
        (
            "manual shape encoding only",
            lambda: [
                ("[" + ",".join(str(int(s)) for s in v.shape) + "]").encode("ascii")
                for v in values
            ],
        ),
        (
            "dtype string only",
            lambda: [
                str(v.dtype).replace("torch.", "").encode("ascii") for v in values
            ],
        ),
        (
            "torch buffer chain only",
            lambda: [
                v.detach().to(device="cpu").contiguous().reshape(-1)
                .view(torch.uint8).numpy()
                for v in values
            ],
        ),
    ):
        best, median = _time(call)
        print(f"  {label:<34} best={best * 1e3:8.2f} ms median={median * 1e3:8.2f} ms")

    # Byte equality of the manual shape encoding against canonical JSON.
    for v in values[:3] + [torch.zeros(())]:
        shape = [int(s) for s in v.shape]
        assert m._canonical_json(shape) == (
            "[" + ",".join(str(s) for s in shape) + "]"
        ).encode("ascii"), shape
    print("\nmanual shape encoding == _canonical_json: OK")


if __name__ == "__main__":
    main()
