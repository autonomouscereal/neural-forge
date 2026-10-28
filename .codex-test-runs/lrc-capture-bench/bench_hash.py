"""Scratch micro-benchmark: where does tensor SHA-256 time actually go?"""

from __future__ import annotations

import hashlib
import statistics
import time

import torch

import neural_forge.hf_lfm2_circuit_trace_cache as m


def _values(count: int, elements: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        return [torch.randn(1, 16, elements, dtype=torch.float32) for _ in range(count)]


def _time(call, repeats: int = 5):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append(time.perf_counter() - started)
    return min(samples), statistics.median(samples)


def _raw_bytes(values):
    return [v.reshape(-1).view(torch.uint8).numpy() for v in values]


def main() -> None:
    for count, elements in ((480, 2048), (120, 65536), (32, 1048576)):
        values = _values(count, elements)
        total = sum(int(v.numel() * v.element_size()) for v in values)
        buffers = _raw_bytes(values)
        print(f"\n{count} tensors x {total / count / 1e6:.2f} MB = {total / 1e6:.1f} MB")

        best, _ = _time(lambda: [m._tensor_sha256(torch, v) for v in values])
        print(f"  serial _tensor_sha256          {best * 1e3:8.2f} ms "
              f"({total / best / 1e9:.2f} GB/s)")

        best_raw, _ = _time(lambda: [hashlib.sha256(b).hexdigest() for b in buffers])
        print(f"  serial raw hashlib only        {best_raw * 1e3:8.2f} ms "
              f"({total / best_raw / 1e9:.2f} GB/s)")

        for workers in (2, 4, 6, 12):
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                chunks = m._byte_balanced_chunks(
                    [int(v.numel() * v.element_size()) for v in values],
                    workers,
                )

                def run():
                    futures = [
                        pool.submit(m._ordered_chunk_sha256, torch, values[a:b])
                        for a, b in chunks
                    ]
                    for future in futures:
                        future.result()

                best_p, _ = _time(run)
                print(f"  chunked x{workers:<3}                  "
                      f"{best_p * 1e3:8.2f} ms "
                      f"({best / best_p:.2f}x, {total / best_p / 1e9:.2f} GB/s)")

                def run_raw():
                    futures = [
                        pool.submit(
                            lambda part: [hashlib.sha256(b).hexdigest() for b in part],
                            buffers[a:b],
                        )
                        for a, b in chunks
                    ]
                    for future in futures:
                        future.result()

                best_pr, _ = _time(run_raw)
                print(f"    raw hashlib x{workers:<3}            "
                      f"{best_pr * 1e3:8.2f} ms "
                      f"({best_raw / best_pr:.2f}x, {total / best_pr / 1e9:.2f} GB/s)")


if __name__ == "__main__":
    main()
