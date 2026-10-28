"""Scratch benchmark: serial baseline collector vs the optimized collector.

Not part of the test suite.  Runs synthetic tiny captures on CPU so parallel
hashing, the seal pipeline, the parallel parent fingerprint, and load_batch
verified-once reuse can be timed against the pre-change implementation.  The
accelerator staging path cannot be timed here: this box has no CUDA device.
"""

from __future__ import annotations

import importlib.util
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
import transformers

import neural_forge.hf_lfm2_circuit_trace_cache as optimized


BASELINE_PATH = Path(__file__).resolve().parent / "baseline_collector.py"


def _load_baseline():
    spec = importlib.util.spec_from_file_location(
        "baseline_lfm2_circuit_trace_cache",
        BASELINE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _model(*, seed: int, hidden: int, layers: int):
    config = transformers.Lfm2Config(
        vocab_size=512,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=8,
        num_key_value_heads=4,
        layer_types=["conv", "full_attention"] * (layers // 2),
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return transformers.Lfm2ForCausalLM(config)


def _batches(count: int, prompt_length: int, forced_length: int):
    result = []
    for index in range(count):
        ids = (
            torch.arange(
                1 + index * prompt_length,
                1 + (index + 1) * prompt_length,
                dtype=torch.long,
            )
            .remainder(500)
            .add(1)
            .reshape(1, prompt_length)
        )
        result.append(
            {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "forced_decode_ids": torch.arange(
                    1,
                    forced_length + 1,
                    dtype=torch.long,
                ).reshape(1, forced_length),
            }
        )
    return result


def _capture(module, model, batches, repeats: int):
    config = module.Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=16,
        max_decode_steps=8,
        max_capture_bytes=8 * 1024 * 1024 * 1024,
        decode_trace_mode=module.FULL_PREFIX_CANONICAL_TRACE_MODE,
    )
    samples = []
    replay = None
    for _ in range(repeats):
        started = time.perf_counter()
        replay = module.capture_lfm2_child_replay(
            model,
            batches,
            selected_layer_indices=list(range(len(model.model.layers))),
            config=config,
            input_device="cpu",
        )
        samples.append(time.perf_counter() - started)
    return samples, replay


def _best(label, samples):
    print(
        f"  {label:<34} best={min(samples):8.4f}s "
        f"median={statistics.median(samples):8.4f}s"
    )
    return min(samples)


def main() -> None:
    baseline = _load_baseline()
    model = _model(seed=4321, hidden=256, layers=4)
    batches = _batches(count=4, prompt_length=96, forced_length=4)
    repeats = 5

    print(f"cpu_count={os.cpu_count()} "
          f"hash_workers={optimized._tensor_hash_worker_count()}")

    print("\n[1] end-to-end capture_lfm2_child_replay (4 batches, 4 layers)")
    base_samples, base_replay = _capture(baseline, model, batches, repeats)
    fast_samples, fast_replay = _capture(optimized, model, batches, repeats)
    base_best = _best("baseline", base_samples)
    fast_best = _best("optimized", fast_samples)
    print(f"  speedup                            {base_best / fast_best:.2f}x")
    print(f"  timing block: {fast_replay.capture_timing}")

    for label, tensors in (
        ("small capture", dict(fast_replay.load_batch(0).tensors)),
        ("production-scale capture", _synthetic_inventory()),
    ):
        total_bytes = sum(
            int(value.numel() * value.element_size()) for value in tensors.values()
        )
        print(
            f"\n[2] sealed inventory hashing - {label} "
            f"({len(tensors)} tensors, {total_bytes / 1e6:.1f} MB)"
        )
        serial_samples = []
        parallel_samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            serial = {
                name: baseline._tensor_descriptor(torch, value)
                for name, value in sorted(tensors.items())
            }
            serial_samples.append(time.perf_counter() - started)
            started = time.perf_counter()
            parallel = optimized._ordered_tensor_descriptors(torch, tensors)
            parallel_samples.append(time.perf_counter() - started)
        serial_best = _best("serial descriptors", serial_samples)
        parallel_best = _best("chunked parallel descriptors", parallel_samples)
        print(f"  speedup                            {serial_best / parallel_best:.2f}x")
        assert optimized._canonical_json(serial) == optimized._canonical_json(parallel)
        print("  inventory bytes                    IDENTICAL")

    big = _model(seed=99, hidden=512, layers=6)
    print(f"\n[3] fingerprint_lfm2_parent ({len(big.state_dict())} tensors)")
    base_fp = [
        _timed(lambda: baseline.fingerprint_lfm2_parent(big)) for _ in range(repeats)
    ]
    fast_fp = [
        _timed(lambda: optimized.fingerprint_lfm2_parent(big)) for _ in range(repeats)
    ]
    base_fp_best = _best("baseline", [value for value, _ in base_fp])
    fast_fp_best = _best("optimized", [value for value, _ in fast_fp])
    print(f"  speedup                            {base_fp_best / fast_fp_best:.2f}x")
    assert base_fp[0][1].identity_sha256 == fast_fp[0][1].identity_sha256
    print("  identity_sha256                    IDENTICAL")

    print("\n[4] repeated Lfm2CircuitTraceCache.load_batch")
    # The baseline copy has a different collector source hash, so it cannot
    # open a cache this build wrote.  Measure the reuse win within one build
    # instead: forced rehash (guards released) versus guarded reuse.
    with tempfile.TemporaryDirectory() as root:
        cache_config = optimized.Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=16,
            max_decode_steps=4,
            max_capture_bytes=8 * 1024 * 1024 * 1024,
        )
        built = optimized.build_lfm2_circuit_trace_cache(
            Path(root) / "cache",
            model,
            batches[:2],
            config=cache_config,
            input_device="cpu",
        )
        cache = optimized.Lfm2CircuitTraceCache.open(built.path, verify=False)
        rehash_loads = []
        guarded_loads = []
        first_record = cache.load_batch(0).record
        for _ in range(8):
            cache.release_verified_batch_guards()
            rehash_loads.append(_timed(lambda: cache.load_batch(0))[0])
            guarded_loads.append(_timed(lambda: cache.load_batch(0))[0])
        rehash_best = _best("forced rehash", rehash_loads)
        guarded_best = _best("guarded reuse", guarded_loads)
        print(f"  speedup                            {rehash_best / guarded_best:.2f}x")
        assert cache.load_batch(0).record == first_record
        print("  record                             IDENTICAL")
        cache.release_verified_batch_guards()

    print("\n[5] determinism of the sealed replay records")
    for index in range(len(batches)):
        assert (
            base_replay.load_batch(index).record
            == fast_replay.load_batch(index).record
        )
        assert baseline._canonical_json(
            base_replay.load_batch(index).record
        ) == optimized._canonical_json(fast_replay.load_batch(index).record)
    print("  every replay_batch_sha256          BYTE-IDENTICAL")
    base_metadata = base_replay.metadata
    fast_metadata = fast_replay.metadata
    # The baseline copy lives at a different path, so its collector source hash
    # legitimately differs.  Everything else must match exactly.
    base_metadata["implementation"] = fast_metadata["implementation"]
    base_metadata.pop("replay_sha256")
    fast_metadata.pop("replay_sha256")
    assert baseline._canonical_json(base_metadata) == optimized._canonical_json(
        fast_metadata
    )
    print("  canonical replay metadata          BYTE-IDENTICAL")


def _synthetic_inventory():
    """Endpoint set shaped like one production capture-point batch."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        return {
            f"phase.{phase:03d}.layer.{layer:03d}.endpoint.{slot:02d}": torch.randn(
                1,
                16,
                2048,
                dtype=torch.float32,
            )
            for phase in range(3)
            for layer in range(16)
            for slot in range(10)
        }


def _timed(call):
    started = time.perf_counter()
    result = call()
    return time.perf_counter() - started, result


if __name__ == "__main__":
    main()
