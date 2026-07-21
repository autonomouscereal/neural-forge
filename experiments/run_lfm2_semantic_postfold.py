#!/usr/bin/env python3
"""Run the saved-child, no-training LFM2 semantic postfold transaction."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


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

from experiments.run_lfm2_residual_attention_fold import (  # noqa: E402
    load_local_tokenizer,
)
from neural_forge.hf_lfm2_semantic_postfold import (  # noqa: E402
    Lfm2SemanticPostfoldBlockedError,
    SEMANTIC_ATTENTION_INSTALLER_CALLABLE,
    SEMANTIC_LAYER_STACK_INSTALLER_CALLABLE,
    run_lfm2_semantic_postfold,
)


def _new_file(value: str, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise ValueError(f"{name} already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError(f"{name} parent must be a real existing directory.")
    return path


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
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
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _installer(module_name: str | None, callable_name: str) -> Any | None:
    if module_name is None:
        return None
    module = importlib.import_module(module_name)
    value = getattr(module, callable_name, None)
    if not callable(value):
        raise RuntimeError(f"{module_name}.{callable_name} is missing.")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a complete causal semantic layer-stack refold and one fused "
            "final-MLP/terminal/lexical W2 solve to a saved standalone CPCF-v5 "
            "LFM2 child without training."
        )
    )
    parser.add_argument("--input-checkpoint", required=True)
    parser.add_argument("--expected-input-artifact-sha256", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--receipt-json", required=True)
    parser.add_argument(
        "--trace-cache",
        required=True,
        help=(
            "Complete semantic-evidence cache. The distinct c61 compiler cache "
            "is reopened from the pinned compiler diagnostic."
        ),
    )
    parser.add_argument(
        "--expected-cache-key",
        required=True,
        help="Externally pinned identity of the semantic-evidence cache.",
    )
    parser.add_argument("--bridge-directory", required=True)
    parser.add_argument("--expected-bridge-artifact-sha256", required=True)
    parser.add_argument("--donor-tokenizer-directory", required=True)
    parser.add_argument("--target-tokenizer-directory", required=True)
    parser.add_argument("--parent-model-directory", required=True)
    parser.add_argument("--compiler-diagnostic-json", required=True)
    parser.add_argument(
        "--expected-compiler-diagnostic-sha256",
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument(
        "--parent-final-norm-tensor-key",
        default="model.embedding_norm.weight",
    )
    parser.add_argument("--parent-norm-epsilon-config-key", default="norm_eps")
    parser.add_argument(
        "--parent-tied-tensor-key",
        default="model.embed_tokens.weight",
    )
    parser.add_argument(
        "--attention-installer-module",
        help=(
            "Module exporting the exact callable "
            f"{SEMANTIC_ATTENTION_INSTALLER_CALLABLE}; omitted means use the "
            "canonical built-in resolver and fail closed if unavailable."
        ),
    )
    parser.add_argument(
        "--semantic-layer-stack-compiler-module",
        help=(
            "Module exporting the complete causal callable "
            f"{SEMANTIC_LAYER_STACK_INSTALLER_CALLABLE}; omitted means use "
            "the canonical resolver and fail closed if unavailable."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = _new_file(args.receipt_json, "receipt JSON")
    output = Path(args.output_checkpoint).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"output checkpoint already exists: {output}")
    donor = load_local_tokenizer(args.donor_tokenizer_directory)
    target = load_local_tokenizer(args.target_tokenizer_directory)
    try:
        result = run_lfm2_semantic_postfold(
            input_checkpoint_directory=args.input_checkpoint,
            expected_input_artifact_sha256=(args.expected_input_artifact_sha256),
            output_checkpoint_directory=output,
            cache_directory=args.trace_cache,
            expected_cache_key=args.expected_cache_key,
            bridge_directory=args.bridge_directory,
            expected_bridge_artifact_sha256=(args.expected_bridge_artifact_sha256),
            donor_tokenizer=donor,
            target_tokenizer=target,
            parent_model_directory=args.parent_model_directory,
            compiler_diagnostic_json=args.compiler_diagnostic_json,
            expected_compiler_diagnostic_sha256=(
                args.expected_compiler_diagnostic_sha256
            ),
            device=args.device,
            dtype=args.dtype,
            parent_final_norm_tensor_key=(args.parent_final_norm_tensor_key),
            parent_norm_epsilon_config_key=(args.parent_norm_epsilon_config_key),
            parent_tied_tensor_key=args.parent_tied_tensor_key,
            attention_installer=_installer(
                args.attention_installer_module,
                SEMANTIC_ATTENTION_INSTALLER_CALLABLE,
            ),
            semantic_layer_stack_compiler=_installer(
                args.semantic_layer_stack_compiler_module,
                SEMANTIC_LAYER_STACK_INSTALLER_CALLABLE,
            ),
        )
    except Lfm2SemanticPostfoldBlockedError as exc:
        _atomic_json(receipt, dict(exc.report))
        print(json.dumps(dict(exc.report), indent=2, sort_keys=True))
        return 2
    _atomic_json(receipt, dict(result))
    print(json.dumps(dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
