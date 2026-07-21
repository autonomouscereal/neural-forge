#!/usr/bin/env python3
"""Recache one sealed LFM2 corpus with its complete causal circuit lattice.

This is a one-pass evidence operation, not a fold or a training stage.  It
reads only the authenticated input tensors from an existing circuit cache,
loads the explicitly local parent once, and publishes a new content-addressed
cache containing:

* every prompt position instead of only a sampled prefill suffix;
* complete attention K/V history, pre-RoPE Q/K, scores, probabilities, masks,
  and contexts;
* complete short-convolution state/response evidence; and
* the final-normalization/output boundary.

The resulting cache is reusable by all later closed-form folding passes.  No
token generation, gradient, backward pass, optimizer, pruning, quantization,
network access, or Hugging Face auto-loader is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch

from experiments.run_lfm2_residual_attention_fold import (
    direct_local_io_audit,
    load_local_lfm2_model,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    CACHED_VERIFIED_TRACE_MODE,
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)


REPORT_SCHEMA = "neural-forge-lfm2-complete-causal-evidence-recache-v1"
_DIGEST_LENGTH = 64
_REQUIRED_INPUT_TENSORS = (
    "input.attention_mask",
    "input.forced_decode_ids",
    "input.input_ids",
)


class Lfm2CompleteCausalEvidenceRecacheError(RuntimeError):
    """The source cache, local parent, or complete recache failed closed."""


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
        raise Lfm2CompleteCausalEvidenceRecacheError(
            "Recache evidence must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: str, name: str) -> str:
    normalized = str(value).strip().lower()
    if (
        len(normalized) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise Lfm2CompleteCausalEvidenceRecacheError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return normalized


def _dtype(name: str) -> torch.dtype:
    try:
        return {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }[name.strip().lower()]
    except KeyError as exc:
        raise Lfm2CompleteCausalEvidenceRecacheError(
            f"Unsupported dtype {name!r}."
        ) from exc


def _source_input_batches(
    cache: Lfm2CircuitTraceCache,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    with cache.selective_authenticated_session() as session:
        for reference in session.batch_references:
            available = frozenset(reference.ordered_tensor_names)
            missing = sorted(set(_REQUIRED_INPUT_TENSORS).difference(available))
            if missing:
                raise Lfm2CompleteCausalEvidenceRecacheError(
                    f"Source batch {reference.index} lacks input tensors: {missing}."
                )
            selected_names = list(_REQUIRED_INPUT_TENSORS)
            if "input.position_ids" in available:
                selected_names.append("input.position_ids")
            names = tuple(sorted(selected_names))
            payload = session.load_selected_tensors(reference.index, names)
            try:
                tensors = payload.tensors
                batch = {
                    "input_ids": tensors["input.input_ids"].clone(),
                    "attention_mask": tensors["input.attention_mask"].clone(),
                    "forced_decode_ids": tensors[
                        "input.forced_decode_ids"
                    ].clone(),
                }
                if "input.position_ids" in tensors:
                    batch["position_ids"] = tensors["input.position_ids"].clone()
                batches.append(batch)
            finally:
                session.release_selected_tensors(payload)
        session.verify()
        receipt = session.as_dict()
    if len(batches) != cache.batch_count:
        raise Lfm2CompleteCausalEvidenceRecacheError(
            "Authenticated input extraction did not preserve every source batch."
        )
    return tuple(batches), receipt


def _batch_geometry(
    batches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prompt_lengths: list[int] = []
    decode_lengths: list[int] = []
    full_lengths: list[int] = []
    for index, batch in enumerate(batches):
        input_ids = batch["input_ids"]
        forced_ids = batch["forced_decode_ids"]
        if (
            not torch.is_tensor(input_ids)
            or not torch.is_tensor(forced_ids)
            or input_ids.ndim != 2
            or forced_ids.ndim != 2
            or int(input_ids.shape[0]) != 1
            or int(forced_ids.shape[0]) != 1
        ):
            raise Lfm2CompleteCausalEvidenceRecacheError(
                f"Source batch {index} is not a single-sequence causal record."
            )
        prompt = int(input_ids.shape[1])
        decode = int(forced_ids.shape[1])
        if prompt <= 0 or decode <= 0:
            raise Lfm2CompleteCausalEvidenceRecacheError(
                f"Source batch {index} has an empty prompt or forced horizon."
            )
        prompt_lengths.append(prompt)
        decode_lengths.append(decode)
        full_lengths.append(prompt + decode)
    return {
        "batch_count": len(batches),
        "prompt_token_counts": prompt_lengths,
        "forced_decode_token_counts": decode_lengths,
        "full_prefix_token_counts": full_lengths,
        "maximum_prompt_tokens": max(prompt_lengths),
        "maximum_decode_tokens": max(decode_lengths),
        "maximum_full_prefix_tokens": max(full_lengths),
    }


def _validate_complete_manifest(
    cache: Lfm2CircuitTraceCache,
    *,
    expected_parent_fingerprint: str,
    geometry: Mapping[str, Any],
) -> None:
    manifest = cache.manifest
    capture = manifest.get("capture")
    if not isinstance(capture, Mapping):
        raise Lfm2CompleteCausalEvidenceRecacheError(
            "Completed cache has no capture contract."
        )
    maximum_full = int(geometry["maximum_full_prefix_tokens"])
    required = {
        "capture_causal_circuit_evidence": True,
        "attention_pre_rope_qk": True,
        "attention_exact_causal_scores_and_probabilities": True,
        "attention_complete_kv_history": True,
        "short_conv_exact_fir_state_and_response": True,
        "final_normalization_boundary": True,
    }
    if any(capture.get(name) != value for name, value in required.items()):
        raise Lfm2CompleteCausalEvidenceRecacheError(
            "Completed cache does not authenticate the full causal-circuit "
            "evidence contract."
        )
    # Older cached-verified manifests encode the default mode through these
    # execution booleans and omit the redundant decode_trace_mode spelling.
    # Full-prefix mode always writes its explicit non-default mode.
    if (
        capture.get("decode_trace_mode", CACHED_VERIFIED_TRACE_MODE)
        != CACHED_VERIFIED_TRACE_MODE
        or capture.get("cached_prefill_use_cache") is not True
        or capture.get("forced_decode_use_cache") is not True
        or capture.get("full_prefix_verification_use_cache") is not False
    ):
        raise Lfm2CompleteCausalEvidenceRecacheError(
            "Completed cache is not the cached-verified causal trace mode."
        )
    if int(capture.get("sample_last_tokens", 0)) < maximum_full:
        raise Lfm2CompleteCausalEvidenceRecacheError(
            "Completed cache still truncates the maximum causal prefix."
        )
    if cache.parent_fingerprint != expected_parent_fingerprint:
        raise Lfm2CompleteCausalEvidenceRecacheError(
            "Completed cache is bound to the wrong parent state."
        )


def recache_complete_causal_evidence(
    *,
    parent_model_directory: Path,
    source_cache_directory: Path,
    output_cache_root: Path,
    report_path: Path,
    expected_source_cache_key: str,
    expected_source_manifest_sha256: str,
    expected_parent_fingerprint: str,
    device: str,
    dtype: str,
    maximum_capture_bytes_per_batch: int,
    logit_rtol: float,
    logit_atol: float,
) -> dict[str, Any]:
    expected_cache = _digest(
        expected_source_cache_key,
        "expected_source_cache_key",
    )
    expected_manifest = _digest(
        expected_source_manifest_sha256,
        "expected_source_manifest_sha256",
    )
    expected_parent = _digest(
        expected_parent_fingerprint,
        "expected_parent_fingerprint",
    )
    if report_path.exists():
        raise Lfm2CompleteCausalEvidenceRecacheError(
            f"Refusing to overwrite report: {report_path}."
        )

    source = open_lfm2_circuit_trace_cache(
        source_cache_directory,
        verify=False,
    )
    source_manifest = source.manifest
    if (
        source.cache_key != expected_cache
        or source_manifest.get("manifest_sha256") != expected_manifest
        or source.parent_fingerprint != expected_parent
    ):
        raise Lfm2CompleteCausalEvidenceRecacheError(
            "Source cache identity differs from the explicit expected seals."
        )
    batches, source_session = _source_input_batches(source)
    geometry = _batch_geometry(batches)
    source_capture = source_manifest.get("capture")
    if not isinstance(source_capture, Mapping):
        raise Lfm2CompleteCausalEvidenceRecacheError(
            "Source cache has no authenticated capture policy."
        )
    source_require_top1 = source_capture.get("require_top1_agreement")
    source_allow_numeric_tie = source_capture.get(
        "allow_top1_disagreement_within_numeric_error_envelope",
        False,
    )
    if (
        not isinstance(source_require_top1, bool)
        or not isinstance(source_allow_numeric_tie, bool)
    ):
        raise Lfm2CompleteCausalEvidenceRecacheError(
            "Source cache top-1 verification policy is invalid."
        )

    parent = load_local_lfm2_model(
        parent_model_directory,
        device=device,
        dtype=_dtype(dtype),
    )
    config = getattr(parent, "config", None)
    if config is None:
        raise Lfm2CompleteCausalEvidenceRecacheError(
            "The local parent exposes no runtime configuration."
        )
    config._attn_implementation = "eager"
    if getattr(config, "_attn_implementation", None) != "eager":
        raise Lfm2CompleteCausalEvidenceRecacheError(
            "Unable to select the exact eager attention operator."
        )
    live_parent = fingerprint_lfm2_parent(parent)
    if live_parent.identity_sha256 != expected_parent:
        raise Lfm2CompleteCausalEvidenceRecacheError(
            "The explicitly local parent state differs from the source cache."
        )

    capture = Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=int(geometry["maximum_full_prefix_tokens"]),
        max_decode_steps=int(geometry["maximum_decode_tokens"]),
        max_capture_bytes=maximum_capture_bytes_per_batch,
        logit_rtol=logit_rtol,
        logit_atol=logit_atol,
        # Preserve the already-sealed corpus verification contract.  This
        # source cache intentionally accepts a few BF16 cached/full-prefix
        # top-1 tie breaks; requiring a stronger decision here would make the
        # exact cached causal operator impossible to recapture.  Attention
        # scores/probabilities/context are independently reconstructed and
        # authenticated on the execution device.
        require_top1_agreement=source_require_top1,
        allow_top1_disagreement_within_numeric_error_envelope=(
            source_allow_numeric_tie
        ),
        decode_trace_mode=CACHED_VERIFIED_TRACE_MODE,
        capture_causal_circuit_evidence=True,
    )
    output_cache_root.mkdir(parents=True, exist_ok=True)
    existing = frozenset(
        path.resolve()
        for path in output_cache_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    )
    result = build_lfm2_circuit_trace_cache(
        output_cache_root,
        parent,
        batches,
        config=capture,
        input_device=torch.device(device),
    )
    _validate_complete_manifest(
        result,
        expected_parent_fingerprint=expected_parent,
        geometry=geometry,
    )
    reused = result.path.resolve() in existing
    expected_forward_calls = sum(
        2 + 2 * count for count in geometry["forced_decode_token_counts"]
    )
    body = {
        "schema": REPORT_SCHEMA,
        "source_cache": {
            "path": str(source.path),
            "cache_key": source.cache_key,
            "manifest_sha256": source_manifest["manifest_sha256"],
            "parent_fingerprint": source.parent_fingerprint,
            "selective_session_sha256": source_session["session_sha256"],
            "selected_input_tensors_only": True,
            "parent_forward_calls": 0,
            "inherited_verification_policy": {
                "require_top1_agreement": source_require_top1,
                "allow_top1_disagreement_within_numeric_error_envelope": (
                    source_allow_numeric_tie
                ),
            },
        },
        "local_parent": {
            **direct_local_io_audit(parent_model_directory),
            "identity_sha256": live_parent.identity_sha256,
            "attention_backend": "eager",
            "loads": 1,
        },
        "corpus_geometry": geometry,
        "complete_cache": {
            "path": str(result.path),
            "cache_key": result.cache_key,
            "manifest_sha256": result.manifest["manifest_sha256"],
            "corpus_sha256": result.corpus_fingerprint,
            "layer_topology_sha256": result.layer_topology_fingerprint,
            "capture": result.manifest["capture"],
            "reused_existing_content_addressed_cache": reused,
            "causal_lattice_complete": True,
            "prompt_suffix_truncation": False,
            "rolling_kv_truncation": False,
        },
        "operation_audit": {
            "evidence_capture_passes": 0 if reused else 1,
            "parent_model_loads": 1,
            "parent_forward_calls": 0 if reused else expected_forward_calls,
            "parent_generation_calls": 0,
            "child_model_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "network_calls": 0,
        },
        "purpose": (
            "one reusable complete causal evidence cache for closed-form "
            "semantic-span residual-attention folding"
        ),
    }
    report = {**body, "report_sha256": _sha256(body)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(report_path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(report) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        report_path.unlink(missing_ok=True)
        raise
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model-directory", type=Path, required=True)
    parser.add_argument("--source-cache-directory", type=Path, required=True)
    parser.add_argument("--output-cache-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-source-cache-key", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-parent-fingerprint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--maximum-capture-bytes-per-batch",
        type=int,
        default=2 * 1024**3,
    )
    parser.add_argument("--logit-rtol", type=float, default=1e-5)
    parser.add_argument("--logit-atol", type=float, default=1e-6)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = recache_complete_causal_evidence(
        parent_model_directory=args.parent_model_directory,
        source_cache_directory=args.source_cache_directory,
        output_cache_root=args.output_cache_root,
        report_path=args.report,
        expected_source_cache_key=args.expected_source_cache_key,
        expected_source_manifest_sha256=(
            args.expected_source_manifest_sha256
        ),
        expected_parent_fingerprint=args.expected_parent_fingerprint,
        device=args.device,
        dtype=args.dtype,
        maximum_capture_bytes_per_batch=(
            args.maximum_capture_bytes_per_batch
        ),
        logit_rtol=args.logit_rtol,
        logit_atol=args.logit_atol,
    )
    print(_canonical_json(report).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
