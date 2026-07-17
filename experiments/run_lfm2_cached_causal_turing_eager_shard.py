#!/usr/bin/env python3
"""Run only the sealed GPU6 shard with eager LFM2 attention.

The cached-causal producer and its scientific contract remain byte-identical.
This wrapper is a narrowly scoped runtime compatibility boundary for the
Turing GPU: it changes only the Hugging Face attention dispatch stored on
freshly loaded model configuration objects, then installs a process-local
tripwire that rejects every SDPA call.  Model parameters and buffers are not
modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional


RUNTIME_RECEIPT_SCHEMA = "neural-forge-lfm2-turing-eager-runtime-v1"
EXPECTED_MODE = "shard"
EXPECTED_SHARD_INDEX = 2
EXPECTED_DEVICE_INDEX = 6
EXPECTED_COMPUTE_CAPABILITY = (7, 5)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class Lfm2TuringEagerRuntimeError(RuntimeError):
    """The isolated Turing eager-attention boundary was violated."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2TuringEagerRuntimeError(
            "Runtime receipt must be finite canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _required_digest_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if _DIGEST.fullmatch(value) is None:
        raise Lfm2TuringEagerRuntimeError(
            f"{name} must contain one lowercase SHA-256 digest."
        )
    return value


def _required_option(argv: Sequence[str], name: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) != 1:
        raise Lfm2TuringEagerRuntimeError(
            f"The guarded runtime requires exactly one {name} option."
        )
    position = positions[0]
    if position + 1 >= len(argv) or argv[position + 1].startswith("--"):
        raise Lfm2TuringEagerRuntimeError(f"{name} has no value.")
    return str(argv[position + 1])


def validate_guarded_scope(
    argv: Sequence[str],
    *,
    cuda: Any = torch.cuda,
) -> dict[str, Any]:
    """Fail closed unless this is exactly shard 2 on physical CUDA device 6."""

    mode = _required_option(argv, "--mode")
    shard_index = int(_required_option(argv, "--shard-index"))
    device = _required_option(argv, "--device")
    match = re.fullmatch(r"cuda:(\d+)", device)
    if (
        mode != EXPECTED_MODE
        or shard_index != EXPECTED_SHARD_INDEX
        or match is None
        or int(match.group(1)) != EXPECTED_DEVICE_INDEX
    ):
        raise Lfm2TuringEagerRuntimeError(
            "This runtime is restricted to --mode shard --shard-index 2 "
            "--device cuda:6."
        )
    if not cuda.is_available():
        raise Lfm2TuringEagerRuntimeError("CUDA is unavailable.")
    device_index = int(match.group(1))
    capability = tuple(int(value) for value in cuda.get_device_capability(device_index))
    if capability != EXPECTED_COMPUTE_CAPABILITY:
        raise Lfm2TuringEagerRuntimeError(
            "The eager compatibility runtime is restricted to CUDA capability "
            f"{EXPECTED_COMPUTE_CAPABILITY}, got {capability}."
        )
    return {
        "mode": mode,
        "shard_index": shard_index,
        "device": device,
        "device_name": str(cuda.get_device_name(device_index)),
        "compute_capability": list(capability),
    }


def _tensor_bindings(model: nn.Module) -> tuple[tuple[Any, ...], ...]:
    values = tuple(model.named_parameters(remove_duplicate=False)) + tuple(
        model.named_buffers(remove_duplicate=False)
    )
    return tuple(
        (
            name,
            id(value),
            int(value.untyped_storage().data_ptr()),
            int(value.storage_offset()),
            tuple(int(size) for size in value.shape),
            str(value.dtype),
            int(value._version),
        )
        for name, value in values
    )


def force_lfm2_eager_attention(model: nn.Module) -> Mapping[str, Any]:
    """Bind LFM2 attention modules to eager dispatch without touching tensors."""

    before_tensors = _tensor_bindings(model)
    configs: dict[int, Any] = {}
    root_config = getattr(model, "config", None)
    if root_config is None:
        raise Lfm2TuringEagerRuntimeError("Loaded LFM2 model has no config.")
    configs[id(root_config)] = root_config
    attention_module_count = 0
    for module in model.modules():
        if type(module).__name__ != "Lfm2Attention":
            continue
        attention_module_count += 1
        config = getattr(module, "config", None)
        if config is None:
            raise Lfm2TuringEagerRuntimeError(
                "Lfm2Attention module has no runtime config."
            )
        configs[id(config)] = config
    if attention_module_count == 0:
        raise Lfm2TuringEagerRuntimeError(
            "Loaded model has no concrete Lfm2Attention modules."
        )
    original_backends = sorted(
        {
            str(getattr(config, "_attn_implementation", None) or "eager")
            for config in configs.values()
        }
    )
    for config in configs.values():
        config._attn_implementation = "eager"
    if any(
        getattr(config, "_attn_implementation", None) != "eager"
        for config in configs.values()
    ):
        raise Lfm2TuringEagerRuntimeError(
            "An LFM2 attention config rejected eager dispatch."
        )
    if _tensor_bindings(model) != before_tensors:
        raise Lfm2TuringEagerRuntimeError(
            "Attention dispatch binding changed a parameter or buffer."
        )
    return {
        "original_attention_backends": original_backends,
        "effective_attention_backend": "eager",
        "attention_module_count": attention_module_count,
        "config_object_count": len(configs),
        "tensor_bindings_unchanged": True,
    }


@contextmanager
def forbid_sdpa_dispatch() -> Iterator[dict[str, int]]:
    """Reject SDPA so eager cannot silently regress to Flash dispatch."""

    original = functional.scaled_dot_product_attention
    audit = {"dispatch_attempts": 0}

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        audit["dispatch_attempts"] += 1
        raise Lfm2TuringEagerRuntimeError(
            "SDPA dispatch is forbidden inside the Turing eager shard."
        )

    functional.scaled_dot_product_attention = forbidden
    try:
        yield audit
    finally:
        functional.scaled_dot_product_attention = original


def _read_json_file(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise Lfm2TuringEagerRuntimeError(f"{name} is missing or unsafe: {path}.")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2TuringEagerRuntimeError(f"{name} is not canonical JSON.") from exc
    if not isinstance(value, Mapping):
        raise Lfm2TuringEagerRuntimeError(f"{name} must be a JSON object.")
    return value


def authenticate_sealed_scope(argv: Sequence[str]) -> dict[str, Any]:
    """Authenticate the unchanged contract and existing shard ownership."""

    from neural_forge.hf_lfm2_cached_causal_sharding import (
        verify_lfm2_cached_causal_run_contract,
        verify_lfm2_cached_causal_shard_spec,
    )

    contract_path = Path(
        _required_option(argv, "--run-contract-document")
    ).resolve(strict=True)
    work_root = Path(_required_option(argv, "--work-root")).resolve(strict=True)
    spec_path = (work_root / "shard-spec.json").resolve(strict=True)
    expected_contract_file_sha = _required_digest_environment(
        "NEURAL_FORGE_EXPECTED_RUN_CONTRACT_FILE_SHA256"
    )
    expected_spec_file_sha = _required_digest_environment(
        "NEURAL_FORGE_EXPECTED_SHARD_SPEC_FILE_SHA256"
    )
    if _file_sha256(contract_path) != expected_contract_file_sha:
        raise Lfm2TuringEagerRuntimeError("Sealed run-contract file changed.")
    if _file_sha256(spec_path) != expected_spec_file_sha:
        raise Lfm2TuringEagerRuntimeError("Sealed shard-spec file changed.")
    contract = _read_json_file(contract_path, "run contract")
    spec = _read_json_file(spec_path, "shard spec")
    verify_lfm2_cached_causal_run_contract(contract)
    verify_lfm2_cached_causal_shard_spec(spec, contract)
    if (
        spec.get("shard_index") != EXPECTED_SHARD_INDEX
        or spec.get("shard_count") != 3
        or spec.get("owned_layer_indices") != [10, 11, 12, 13]
        or spec.get("expected_candidate_count") != 24
    ):
        raise Lfm2TuringEagerRuntimeError(
            "Authenticated shard spec is not the expected GPU6 ownership."
        )
    return {
        "run_contract_path": str(contract_path),
        "run_contract_file_sha256": expected_contract_file_sha,
        "run_contract_sha256": str(contract["run_contract_sha256"]),
        "shard_spec_path": str(spec_path),
        "shard_spec_file_sha256": expected_spec_file_sha,
        "shard_spec_sha256": str(spec["shard_spec_sha256"]),
        "expected_candidate_count": 24,
        "owned_layer_indices": [10, 11, 12, 13],
    }


def _receipt_path() -> Path:
    raw = os.environ.get("NEURAL_FORGE_LFM2_EAGER_RECEIPT", "")
    if not raw:
        raise Lfm2TuringEagerRuntimeError(
            "NEURAL_FORGE_LFM2_EAGER_RECEIPT is required."
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise Lfm2TuringEagerRuntimeError("Runtime receipt path must be absolute.")
    parent = path.parent.resolve(strict=True)
    result = parent / path.name
    if result.exists() and (not result.is_file() or result.is_symlink()):
        raise Lfm2TuringEagerRuntimeError("Runtime receipt path is unsafe.")
    return result


def _write_receipt(path: Path, body: Mapping[str, Any]) -> None:
    document = {**dict(body), "runtime_receipt_sha256": _json_sha256(body)}
    payload = _canonical_json(document) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(argv: Sequence[str]) -> int:
    """Execute the unchanged producer under the sealed eager boundary."""

    from experiments import build_lfm2_cached_causal_evidence as producer

    scope = validate_guarded_scope(argv)
    sealed_scope = authenticate_sealed_scope(argv)
    producer_path = Path(producer.__file__).resolve(strict=True)
    expected_producer_sha = _required_digest_environment(
        "NEURAL_FORGE_EXPECTED_PRODUCER_FILE_SHA256"
    )
    if _file_sha256(producer_path) != expected_producer_sha:
        raise Lfm2TuringEagerRuntimeError("Cached-causal producer file changed.")
    receipt_path = _receipt_path()
    receipt: dict[str, Any] = {
        "schema": RUNTIME_RECEIPT_SCHEMA,
        "status": "starting",
        "scope": scope,
        "sealed_scope": sealed_scope,
        "producer_file_sha256": expected_producer_sha,
        "wrapper_file_sha256": _file_sha256(Path(__file__).resolve(strict=True)),
        "attention_backend": "eager",
        "sdpa_forbidden": True,
        "model_loads": [],
        "sdpa_dispatch_attempts": 0,
    }
    _write_receipt(receipt_path, receipt)
    original_loader = producer.residual.load_local_lfm2_model

    def eager_loader(*args: Any, **kwargs: Any) -> nn.Module:
        model = original_loader(*args, **kwargs)
        binding = dict(force_lfm2_eager_attention(model))
        loads = list(receipt["model_loads"])
        binding["load_index"] = len(loads)
        binding["role"] = "parent-authentication" if not loads else "child-worker"
        loads.append(binding)
        receipt["model_loads"] = loads
        receipt["status"] = "running"
        _write_receipt(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "event": "turing_eager_attention_bound",
                    **binding,
                    "runtime_receipt": str(receipt_path),
                },
                allow_nan=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return model

    producer.residual.load_local_lfm2_model = eager_loader
    try:
        with forbid_sdpa_dispatch() as sdpa_audit:
            try:
                result = producer.main(argv)
            except BaseException as exc:
                receipt["status"] = "failed"
                receipt["failure_type"] = type(exc).__name__
                receipt["failure_message"] = str(exc)
                raise
            finally:
                receipt["sdpa_dispatch_attempts"] = int(
                    sdpa_audit["dispatch_attempts"]
                )
                _write_receipt(receipt_path, receipt)
        if len(receipt["model_loads"]) != 2:
            raise Lfm2TuringEagerRuntimeError(
                "Shard completed without exactly two authenticated model loads."
            )
        receipt["status"] = "completed"
        _write_receipt(receipt_path, receipt)
        return int(result)
    finally:
        producer.residual.load_local_lfm2_model = original_loader


def main(argv: Sequence[str] | None = None) -> int:
    return run(tuple(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
