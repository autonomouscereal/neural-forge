#!/usr/bin/env python3
"""Replay one authenticated normalized-input compiler request on CPU only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from neural_forge.dynamic_circuit_ir import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from neural_forge.hf_lfm2_normalized_input_decoder import (  # noqa: E402
    compile_lfm2_normalized_input_decoder,
)
from neural_forge.hf_lfm2_normalized_input_replay_capsule import (  # noqa: E402
    load_lfm2_normalized_input_replay_capsule,
)


RESULT_SCHEMA = "neural-forge-lfm2-normalized-input-replay-result-v1"


class Lfm2NormalizedInputReplayError(RuntimeError):
    """The standalone replay result boundary is unsafe."""


def _new_result_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser().absolute()
    parent = path.parent
    is_junction = getattr(os.path, "isjunction", None)
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or bool(callable(is_junction) and is_junction(parent))
        or parent.resolve(strict=True) != parent
        or os.path.lexists(path)
    ):
        raise Lfm2NormalizedInputReplayError(
            "Replay result path must be new beneath an ordinary local directory."
        )
    return path


def _write_result(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    result = {**body, "result_sha256": canonical_sha256(body)}
    payload = canonical_json_bytes(result) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except (FileExistsError, OSError) as exc:
        raise Lfm2NormalizedInputReplayError(
            "Replay result publication refused to overwrite a file."
        ) from exc
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    except OSError as exc:
        raise Lfm2NormalizedInputReplayError(
            "Replay result could not be durably written."
        ) from exc
    finally:
        os.close(descriptor)
    return result


def replay(
    *,
    capsule_directory: Path,
    result_json: Path,
) -> dict[str, Any]:
    """Authenticate one capsule, invoke the production compiler, and report."""

    output = _new_result_path(result_json)
    capsule = load_lfm2_normalized_input_replay_capsule(capsule_directory)
    common = {
        "schema": RESULT_SCHEMA,
        "capsule_sha256": capsule.capsule_sha256,
        "capsule_layer_index": capsule.layer_index,
        "capsule_manifest_schema": capsule.manifest["schema"],
        "capsule_authenticated_before_compiler_invocation": True,
        "production_compiler_callable": (
            "neural_forge.hf_lfm2_normalized_input_decoder."
            "compile_lfm2_normalized_input_decoder"
        ),
        "parent_checkpoint_loads": 0,
        "trace_cache_loads": 0,
        "target_model_loads": 0,
        "model_forward_calls": 0,
        "gpu_operations": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "pickle_loads": 0,
    }
    try:
        compiled = compile_lfm2_normalized_input_decoder(**capsule.compiler_kwargs())
    except Exception as exc:
        return _write_result(
            output,
            {
                **common,
                "status": "production-compiler-failed",
                "compiler_invocations": 1,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:4096],
                "compiler_receipt": None,
            },
        )
    verification = dict(compiled.verify())
    if verification.get("verified") is not True:
        raise Lfm2NormalizedInputReplayError(
            "Production compiler returned an unauthenticated result."
        )
    return _write_result(
        output,
        {
            **common,
            "status": "production-compiler-succeeded",
            "compiler_invocations": 1,
            "normalized_input_decoder_sha256": (
                compiled.normalized_input_decoder_sha256
            ),
            "normalized_input_decoder_rank": (compiled.normalized_input_decoder_rank),
            "source_hidden_size": compiled.source_hidden_size,
            "target_hidden_size": compiled.target_hidden_size,
            "compiler_receipt_sha256": compiled.receipt_sha256,
            "compiler_receipt": compiled.as_dict(),
            "compiler_verification": verification,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule-directory", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = replay(
        capsule_directory=args.capsule_directory,
        result_json=args.result_json,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result["status"] == "production-compiler-succeeded" else 1


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
