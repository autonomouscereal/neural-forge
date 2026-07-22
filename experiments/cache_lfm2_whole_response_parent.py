#!/usr/bin/env python3
"""Cache long-form responses from one explicit local LFM2 parent exactly once."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from experiments import run_lfm2_residual_attention_fold as residual  # noqa: E402
from experiments.review_lfm2_semantic_frontier import (  # noqa: E402
    _generate_rows,
    load_review_prompts,
)
from experiments.run_lfm2_dynamic_catalog import (  # noqa: E402
    fingerprint_local_tokenizer,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
)


PARENT_ENVELOPE_SCHEMA = "neural-forge-lfm2-whole-response-parent-envelope-v1"


class Lfm2WholeResponseParentCacheError(RuntimeError):
    """The one-time local parent response cache could not be published."""


def _digest(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise Lfm2WholeResponseParentCacheError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return normalized


def _local_directory(value: Path, *, label: str) -> Path:
    lexical = value.expanduser().absolute()
    if lexical.is_symlink():
        raise Lfm2WholeResponseParentCacheError(f"{label} cannot be a link.")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2WholeResponseParentCacheError(f"{label} is missing.") from exc
    if resolved != lexical or not resolved.is_dir():
        raise Lfm2WholeResponseParentCacheError(
            f"{label} must be one canonical real directory."
        )
    return resolved


def _new_output_file(value: Path) -> Path:
    output = value.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise Lfm2WholeResponseParentCacheError(
            f"Refusing to replace existing parent response envelope: {output}."
        )
    parent = output.parent.resolve(strict=True)
    if parent != output.parent or not parent.is_dir() or parent.is_symlink():
        raise Lfm2WholeResponseParentCacheError(
            "Parent response envelope parent must be one canonical real directory."
        )
    return output


def _atomic_write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise Lfm2WholeResponseParentCacheError(
                f"Refusing to replace existing parent response envelope: {path}."
            )
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def cache_parent_responses(
    *,
    parent_model_directory: Path,
    tokenizer_directory: Path,
    prompts_json: Path,
    output_json: Path,
    device: str,
    dtype: str,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
    expected_parent_state_sha256: str | None = None,
    expected_parent_identity_sha256: str | None = None,
    expected_tokenizer_sha256: str | None = None,
) -> dict[str, Any]:
    """Run each fixed review prompt once and seal the immutable response bank."""

    if not 1 <= maximum_prompt_tokens <= 8192:
        raise Lfm2WholeResponseParentCacheError(
            "maximum_prompt_tokens must be between 1 and 8192."
        )
    if not 16 <= maximum_new_tokens <= 2048:
        raise Lfm2WholeResponseParentCacheError(
            "maximum_new_tokens must be between 16 and 2048."
        )
    parent_path = _local_directory(parent_model_directory, label="parent model")
    tokenizer_path = _local_directory(tokenizer_directory, label="parent tokenizer")
    output = _new_output_file(output_json)
    if (
        output == parent_path
        or output == tokenizer_path
        or parent_path in output.parents
        or tokenizer_path in output.parents
    ):
        raise Lfm2WholeResponseParentCacheError(
            "Parent response output cannot be inside a model or tokenizer input."
        )
    prompts = load_review_prompts(prompts_json)
    prompt_rows = [dict(row) for row in prompts]
    prompt_set_sha256 = canonical_sha256(prompt_rows)
    tokenizer_sha256 = fingerprint_local_tokenizer(tokenizer_path)
    expected_tokenizer = _digest(
        expected_tokenizer_sha256,
        label="expected tokenizer SHA-256",
    )
    if expected_tokenizer is not None and tokenizer_sha256 != expected_tokenizer:
        raise Lfm2WholeResponseParentCacheError(
            "Local tokenizer differs from its external identity pin."
        )

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyTorch is required for parent response caching.") from exc
    model = residual.load_local_lfm2_model(
        parent_path,
        device=torch.device(device),
        dtype=residual._torch_dtype(dtype),
    )
    tokenizer = residual.load_local_tokenizer(tokenizer_path)
    state_before = residual.model_state_sha256(model)
    identity_before = fingerprint_lfm2_parent(model).identity_sha256
    parameter_count_before = residual.unique_parameter_count(model)
    expected_state = _digest(
        expected_parent_state_sha256,
        label="expected parent state SHA-256",
    )
    expected_identity = _digest(
        expected_parent_identity_sha256,
        label="expected parent identity SHA-256",
    )
    if expected_state is not None and state_before != expected_state:
        raise Lfm2WholeResponseParentCacheError(
            "Local parent state differs from its external identity pin."
        )
    if expected_identity is not None and identity_before != expected_identity:
        raise Lfm2WholeResponseParentCacheError(
            "Local parent topology identity differs from its external pin."
        )
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise Lfm2WholeResponseParentCacheError(
            "Parent model already contains gradients before inference."
        )

    responses = _generate_rows(
        model,
        tokenizer,
        prompts,
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
    )
    state_after = residual.model_state_sha256(model)
    identity_after = fingerprint_lfm2_parent(model).identity_sha256
    parameter_count_after = residual.unique_parameter_count(model)
    if (
        state_after != state_before
        or identity_after != identity_before
        or parameter_count_after != parameter_count_before
        or any(parameter.grad is not None for parameter in model.parameters())
    ):
        raise Lfm2WholeResponseParentCacheError(
            "Parent state, topology, parameter count, or gradients changed."
        )
    response_rows = [
        {
            "id": row["id"],
            "category": row["category"],
            "response": row["response"],
            "response_token_ids": row["response_token_ids"],
            "diagnostics": row["diagnostics"],
        }
        for row in responses
    ]
    body = {
        "schema": PARENT_ENVELOPE_SCHEMA,
        "status": "complete-one-time-parent-response-cache",
        "parent_model_directory": str(parent_path),
        "tokenizer_directory": str(tokenizer_path),
        "prompt_source": str(prompts_json.expanduser().absolute()),
        "prompt_set_sha256": prompt_set_sha256,
        "prompt_count": len(prompts),
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "maximum_new_tokens": maximum_new_tokens,
        "inference_device": device,
        "inference_dtype": dtype,
        "parent_state_sha256": state_before,
        "parent_identity_sha256": identity_before,
        "parent_unique_parameters": parameter_count_before,
        "tokenizer_sha256": tokenizer_sha256,
        "responses": response_rows,
        "response_bank_sha256": canonical_sha256(response_rows),
        "execution_audit": {
            "parent_model_loads": 1,
            "parent_generate_invocations": len(prompts),
            "response_cache_publications": 1,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "autograd_grad_calls": 0,
            "hub_loader_used": False,
            "network_access": False,
            "parent_state_unchanged": True,
            "parent_parameter_count_unchanged": True,
        },
    }
    document = {**body, "envelope_sha256": canonical_sha256(body)}
    _atomic_write_new_json(output, document)
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate each fixed whole-response review prompt exactly once from "
            "an explicit local LFM2 parent, then seal the reusable text cache."
        )
    )
    parser.add_argument("--parent-model-directory", required=True, type=Path)
    parser.add_argument("--tokenizer-directory", required=True, type=Path)
    parser.add_argument("--prompts-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--maximum-prompt-tokens", type=int, default=1024)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    parser.add_argument("--expected-parent-state-sha256")
    parser.add_argument("--expected-parent-identity-sha256")
    parser.add_argument("--expected-tokenizer-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = cache_parent_responses(
        parent_model_directory=args.parent_model_directory,
        tokenizer_directory=args.tokenizer_directory,
        prompts_json=args.prompts_json,
        output_json=args.output_json,
        device=args.device,
        dtype=args.dtype,
        maximum_prompt_tokens=args.maximum_prompt_tokens,
        maximum_new_tokens=args.maximum_new_tokens,
        expected_parent_state_sha256=args.expected_parent_state_sha256,
        expected_parent_identity_sha256=args.expected_parent_identity_sha256,
        expected_tokenizer_sha256=args.expected_tokenizer_sha256,
    )
    print(
        json.dumps(
            {
                "status": document["status"],
                "prompt_count": document["prompt_count"],
                "maximum_new_tokens": document["maximum_new_tokens"],
                "envelope_sha256": document["envelope_sha256"],
                "output": str(args.output_json),
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
