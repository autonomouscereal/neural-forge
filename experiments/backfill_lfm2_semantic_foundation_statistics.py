#!/usr/bin/env python3
"""Backfill compact tied-interface statistics for one saved LFM2 foundation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
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
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from experiments.run_lfm2_residual_attention_fold import (  # noqa: E402
    load_local_tokenizer,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_long_horizon_terminal_refold import (  # noqa: E402
    load_lfm2_parent_final_rmsnorm_from_local_safetensors,
)
from neural_forge.hf_lfm2_semantic_foundation import (  # noqa: E402
    Lfm2SemanticFoundationError,
    backfill_lfm2_semantic_foundation_interface_statistics,
)
from neural_forge.hf_lfm2_semantic_postfold import (  # noqa: E402
    load_lfm2_cpcf_v5_standalone_child,
    load_lfm2_mapped_parent_tied_table_from_local_safetensors,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
)


def _digest(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticFoundationError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _new_file(value: str | os.PathLike[str], name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise Lfm2SemanticFoundationError(f"{name} already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise Lfm2SemanticFoundationError(
            f"{name} parent must be an ordinary existing directory."
        )
    return path


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    raw = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
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
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-checkpoint-directory", required=True, type=Path)
    parser.add_argument("--expected-input-artifact-sha256", required=True)
    parser.add_argument("--semantic-cache-directory", required=True, type=Path)
    parser.add_argument("--expected-semantic-cache-key", required=True)
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument("--expected-bridge-artifact-sha256", required=True)
    parser.add_argument("--donor-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--target-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--parent-model-directory", required=True, type=Path)
    parser.add_argument("--foundation-receipt-json", required=True, type=Path)
    parser.add_argument("--expected-foundation-receipt-sha256", required=True)
    parser.add_argument("--artifact-store-directory", required=True, type=Path)
    parser.add_argument("--backfill-receipt-json", required=True, type=Path)
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
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    semantic_cache_key = _digest(
        args.expected_semantic_cache_key,
        "expected semantic cache key",
    )
    bridge_sha256 = _digest(
        args.expected_bridge_artifact_sha256,
        "expected bridge artifact",
    )
    expected_input_sha256 = _digest(
        args.expected_input_artifact_sha256,
        "expected input artifact",
    )
    foundation_receipt_sha256 = _digest(
        args.expected_foundation_receipt_sha256,
        "expected foundation receipt",
    )
    receipt_path = _new_file(args.backfill_receipt_json, "backfill receipt JSON")
    semantic_cache = open_lfm2_circuit_trace_cache(
        args.semantic_cache_directory,
        verify=True,
    )
    if semantic_cache.cache_key != semantic_cache_key:
        raise Lfm2SemanticFoundationError(
            "Semantic evidence cache differs from its external identity pin."
        )
    donor_tokenizer = load_local_tokenizer(args.donor_tokenizer_directory)
    target_tokenizer = load_local_tokenizer(args.target_tokenizer_directory)
    bridge = load_strict_tokenizer_bridge_artifact(
        args.bridge_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=bridge_sha256,
    )
    target_model, saved_tokenizer, input_metadata = load_lfm2_cpcf_v5_standalone_child(
        args.input_checkpoint_directory,
        expected_artifact_sha256=expected_input_sha256,
        device=args.device,
        dtype=args.dtype,
    )
    if fingerprint_hf_tokenizer_semantics(
        saved_tokenizer
    ) != fingerprint_hf_tokenizer_semantics(target_tokenizer):
        raise Lfm2SemanticFoundationError(
            "Saved child tokenizer differs from the target bridge package."
        )
    parent_norm = load_lfm2_parent_final_rmsnorm_from_local_safetensors(
        semantic_cache,
        args.parent_model_directory,
        tensor_key=args.parent_final_norm_tensor_key,
        epsilon_config_key=args.parent_norm_epsilon_config_key,
    )
    source_table, _source_table_report = (
        load_lfm2_mapped_parent_tied_table_from_local_safetensors(
            args.parent_model_directory,
            cache=semantic_cache,
            bridge=bridge,
            authenticated_parent_final_norm=parent_norm,
            tensor_key=args.parent_tied_tensor_key,
        )
    )
    result = backfill_lfm2_semantic_foundation_interface_statistics(
        target_model,
        source_table,
        semantic_cache=semantic_cache,
        bridge=bridge,
        target_tokenizer=target_tokenizer,
        parent_model_directory=args.parent_model_directory,
        input_metadata=input_metadata,
        foundation_receipt_json=args.foundation_receipt_json,
        expected_foundation_receipt_sha256=foundation_receipt_sha256,
        artifact_store_directory=args.artifact_store_directory,
    )
    report = dict(result.report)
    _atomic_json(receipt_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except BaseException as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
