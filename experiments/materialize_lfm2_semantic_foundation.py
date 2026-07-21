#!/usr/bin/env python3
"""Materialize the exact 57,405,368-parameter LFM2 semantic foundation.

This is the fast replacement for running the historical 14-layer CPCF-v5
compiler merely to obtain a child shell.  It opens only direct-local assets,
synthesizes the exact planner-owned zero blank, folds the tied interface from
the sealed final-normalization evidence, installs neutral norms, and saves and
reloads a standalone dense checkpoint.  Attention/conv/MLP folding remains the
next semantic-postfold stage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_semantic_foundation import (  # noqa: E402
    EXACT_C61_TARGET_PARAMETERS,
    Lfm2SemanticFoundationError,
    materialize_lfm2_semantic_foundation,
    read_lfm2_semantic_foundation_catalog_preparation,
    write_lfm2_semantic_foundation_receipt,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    load_strict_tokenizer_bridge_artifact,
)


def _digest(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticFoundationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model-directory", required=True, type=Path)
    parser.add_argument("--donor-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--target-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument("--compiler-cache-directory", required=True, type=Path)
    parser.add_argument("--trajectory-corpus", required=True, type=Path)
    parser.add_argument(
        "--selective-catalog-store-directory",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--selective-catalog-receipt-json",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--expected-selective-catalog-receipt-sha256",
        required=True,
    )
    parser.add_argument("--output-checkpoint-directory", required=True, type=Path)
    parser.add_argument("--foundation-receipt-json", required=True, type=Path)
    parser.add_argument("--expected-parent-identity-sha256", required=True)
    parser.add_argument("--expected-cache-key", required=True)
    parser.add_argument("--expected-bridge-artifact-sha256", required=True)
    parser.add_argument("--expected-target-plan-fingerprint", required=True)
    parser.add_argument(
        "--expected-selected-candidate-fingerprint",
        required=True,
    )
    parser.add_argument("--parent-name", default="LiquidAI-LFM2.5-230M")
    parser.add_argument(
        "--child-name",
        default="lfm2-semantic-foundation-c61-4x",
    )
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    from experiments.run_lfm2_residual_attention_fold import (
        load_local_lfm2_model,
        load_local_tokenizer,
    )

    parent_identity = _digest(
        args.expected_parent_identity_sha256,
        "expected parent identity",
    )
    cache_key = _digest(args.expected_cache_key, "expected cache key")
    bridge_sha = _digest(
        args.expected_bridge_artifact_sha256,
        "expected bridge artifact",
    )
    cache = open_lfm2_circuit_trace_cache(
        args.compiler_cache_directory,
        verify=True,
    )
    if cache.cache_key != cache_key or cache.parent_fingerprint != parent_identity:
        raise Lfm2SemanticFoundationError(
            "Compiler cache differs from its external identity pins."
        )
    donor_tokenizer = load_local_tokenizer(args.donor_tokenizer_directory)
    target_tokenizer = load_local_tokenizer(args.target_tokenizer_directory)
    bridge = load_strict_tokenizer_bridge_artifact(
        args.bridge_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=bridge_sha,
    )
    preparation = read_lfm2_semantic_foundation_catalog_preparation(
        args.selective_catalog_receipt_json,
        expected_receipt_sha256=(
            args.expected_selective_catalog_receipt_sha256
        ),
        cache=cache,
        catalog_store_directory=args.selective_catalog_store_directory,
    )
    import torch

    if not torch.cuda.is_available() or not str(args.model_device).startswith("cuda:"):
        raise Lfm2SemanticFoundationError(
            "The production semantic foundation requires one explicit CUDA device."
        )
    parent = load_local_lfm2_model(
        args.parent_model_directory,
        device=torch.device(args.model_device),
        dtype=torch.bfloat16,
    )
    parent.requires_grad_(False)
    parent.eval()
    receipt = materialize_lfm2_semantic_foundation(
        parent_model=parent,
        parent_model_directory=args.parent_model_directory,
        parent_name=args.parent_name,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        bridge=bridge,
        compiler_cache=cache,
        trajectory_corpus=args.trajectory_corpus,
        catalog_preparation=preparation,
        output_checkpoint_directory=args.output_checkpoint_directory,
        expected_parent_identity_sha256=parent_identity,
        expected_target_plan_fingerprint=_digest(
            args.expected_target_plan_fingerprint,
            "expected target plan",
        ),
        expected_selected_candidate_fingerprint=_digest(
            args.expected_selected_candidate_fingerprint,
            "expected selected candidate",
        ),
        child_name=args.child_name,
        device=args.model_device,
        dtype=args.dtype,
    )
    if receipt.get("target_unique_parameters") != EXACT_C61_TARGET_PARAMETERS:
        raise Lfm2SemanticFoundationError(
            "Materializer returned the wrong exact c61 parameter count."
        )
    file_sha256 = write_lfm2_semantic_foundation_receipt(
        args.foundation_receipt_json,
        receipt,
    )
    return {
        "schema": "neural-forge-lfm2-semantic-foundation-launch-v1",
        "status": "semantic-foundation-materialized",
        "foundation_receipt_sha256": receipt["receipt_sha256"],
        "foundation_receipt_file_sha256": file_sha256,
        "foundation_receipt_json": str(args.foundation_receipt_json.resolve()),
        "checkpoint_directory": str(args.output_checkpoint_directory.resolve()),
        "artifact_sha256": receipt["standalone_artifact"]["artifact_sha256"],
        "model_state_sha256": receipt["standalone_artifact"][
            "model_state_sha256"
        ],
        "target_unique_parameters": receipt["target_unique_parameters"],
        "legacy_cpcf_layer_compiler_executed": False,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "parent_forward_calls": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except BaseException as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
