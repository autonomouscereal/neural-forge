"""Extract a cached A12 derivative for fold-only causal support compilation.

This program reads only immutable A12 JSON and its already-captured parent
trace cache.  It never constructs a model, executes a forward, recaptures a
trace, or trains.  Each prompt becomes an authenticated, restartable batch
checkpoint containing prompt IDs and the minimal parent hidden trajectory.
The final feature pack is a 1024-wide C_in identity plus the stacked A12
parent-hidden C_out matrix.  Sparse position/context damage signatures are
published separately and are bounded to donors with observed causal damage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_causal_semantic_support import (  # noqa: E402
    Lfm2CausalContextDerivative,
    build_lfm2_causal_context_derivative,
    seal_lfm2_causal_support_feature_pack,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    Lfm2CircuitTraceCache,
)
from run_lfm2_causal_semantic_support import (  # noqa: E402
    authenticate_completed_trajectory_audit,
    build_fit_prompt_manifest,
)


EXTRACTION_REQUEST_SCHEMA = "neural-forge-lfm2-a12-causal-derivative-request-v1"
BATCH_RECEIPT_SCHEMA = "neural-forge-lfm2-a12-causal-derivative-batch-v1"
EXTRACTION_RECEIPT_SCHEMA = "neural-forge-lfm2-a12-causal-derivative-v1"
EXTRACTION_COMPLETION_SCHEMA = "neural-forge-lfm2-a12-causal-derivative-completion-v1"
MANUAL_REVIEW_SCHEMA = "neural-forge-lfm2-causal-support-future-manual-review-v1"
FEATURE_TENSOR_KEYS = (
    "input_feature_transform",
    "output_feature_transform",
)
CONTEXT_TENSOR_KEYS = (
    "input_context_signatures",
    "input_donor_ids",
    "output_context_signatures",
    "output_donor_ids",
)
BATCH_TENSOR_KEYS = (
    "forced_token_ids",
    "parent_hidden_states",
    "prompt_token_ids",
)
_TENSOR_DOMAIN = b"neural-forge-lfm2-a12-causal-derivative-tensor-v1\0"
_ZERO_OPERATION_AUDIT = {
    "training_steps": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "autograd_grad_calls": 0,
    "backward_calls": 0,
    "model_objects_constructed": 0,
    "model_forward_calls": 0,
    "model_generate_calls": 0,
    "parent_forward_calls": 0,
    "parent_generate_calls": 0,
    "parent_trace_recaptures": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
}


class Lfm2A12CausalDerivativeError(RuntimeError):
    """Raised when an immutable derivative input or checkpoint fails closed."""


def _plain_json(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2A12CausalDerivativeError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2A12CausalDerivativeError(f"{name} must be positive.")
    return int(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256(_TENSOR_DOMAIN)
    digest.update(str(tensor.dtype).removeprefix("torch.").encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2A12CausalDerivativeError(f"Cannot read {name} JSON.") from exc
    if not isinstance(value, Mapping):
        raise Lfm2A12CausalDerivativeError(f"{name} must be a JSON object.")
    return _plain_json(value)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _plain_json(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_save_tensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".safetensors", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        save_file(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in tensors.items()
            },
            str(temporary_path),
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _progress(path: Path, event: str, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema": "neural-forge-lfm2-a12-causal-derivative-progress-v1",
        "event": event,
        "unix_time": time.time(),
        **_plain_json(values),
    }
    with path.open("a", encoding="ascii", newline="\n") as handle:
        handle.write(
            json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":"))
        )
        handle.write("\n")
        handle.flush()


def authenticate_manual_review_inventory(
    value: Mapping[str, Any], *, expected_inventory_sha256: str
) -> dict[str, Any]:
    inventory = _plain_json(value)
    supplied = inventory.pop("inventory_sha256", None)
    prompts = inventory.get("prompts")
    contract = inventory.get("review_contract")
    proof = inventory.get("a12_disjointness_proof")
    if (
        supplied != _digest(expected_inventory_sha256, "manual inventory SHA-256")
        or supplied != canonical_sha256(inventory)
        or inventory.get("schema") != MANUAL_REVIEW_SCHEMA
        or not isinstance(prompts, list)
        or not prompts
        or not isinstance(contract, Mapping)
        or contract.get("manual_whole_response_review_only") is not True
        or contract.get("automatic_gate") is not False
        or contract.get("selection_or_fit_use_permitted") is not False
        or contract.get(
            "safety_behavior_is_scored_separately_from_intelligence_retention"
        )
        is not True
        or contract.get(
            "refusal_or_safety_policy_difference_counts_as_intelligence_failure"
        )
        is not False
        or contract.get(
            "output_truncation_at_an_evaluator_token_cap_counts_as_model_failure"
        )
        is not False
        or not isinstance(proof, Mapping)
        or proof.get("exact_prompt_id_intersection_count") != 0
        or proof.get("exact_prompt_text_intersection_count") != 0
    ):
        raise Lfm2A12CausalDerivativeError(
            "Future manual-review inventory authentication failed."
        )
    ids: list[str] = []
    texts: list[str] = []
    safety_count = 0
    for row in prompts:
        if not isinstance(row, Mapping):
            raise Lfm2A12CausalDerivativeError("Review prompt row is invalid.")
        prompt_id = row.get("id")
        text = row.get("text")
        safety = row.get("safety_review_only")
        if (
            not isinstance(prompt_id, str)
            or not prompt_id
            or not isinstance(text, str)
            or not text
            or not isinstance(safety, bool)
        ):
            raise Lfm2A12CausalDerivativeError("Review prompt fields are invalid.")
        ids.append(prompt_id)
        texts.append(text)
        safety_count += int(safety)
    if len(set(ids)) != len(ids) or len(set(texts)) != len(texts):
        raise Lfm2A12CausalDerivativeError(
            "Review prompt IDs and texts must be unique."
        )
    if len(prompts) != 18 or safety_count != 3:
        raise Lfm2A12CausalDerivativeError(
            "Frozen review inventory must contain 18 prompts and 3 safety rows."
        )
    return {
        **inventory,
        "inventory_sha256": supplied,
        "prompt_ids": ids,
        "prompt_ids_sha256": canonical_sha256(ids),
        "prompt_texts_sha256": canonical_sha256(texts),
        "safety_review_prompt_count": safety_count,
        "intelligence_review_prompt_count": len(prompts) - safety_count,
    }


def _trajectory_inputs(
    audit: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = audit.get("prompt_results")
    trace_receipt = audit.get("trace_selective_session_receipt")
    trace_sha = audit.get("trace_selective_session_sha256")
    if (
        not isinstance(results, list)
        or not results
        or not isinstance(trace_receipt, Mapping)
        or trace_receipt.get("session_sha256") != trace_sha
        or trace_receipt.get("batch_count") != len(results)
    ):
        raise Lfm2A12CausalDerivativeError(
            "A12 trace-session provenance is missing or inconsistent."
        )
    ordered = sorted(
        (_plain_json(row) for row in results), key=lambda row: row["batch_index"]
    )
    if [row["batch_index"] for row in ordered] != list(range(len(ordered))):
        raise Lfm2A12CausalDerivativeError(
            "A12 prompt results are not one canonical batch inventory."
        )
    if len({row["prompt_id"] for row in ordered}) != len(ordered):
        raise Lfm2A12CausalDerivativeError("A12 prompt IDs are not unique.")
    return ordered, _plain_json(trace_receipt)


def _phase_names(forced_count: int) -> tuple[str, ...]:
    return ("prefill", *(f"decode.{index:03d}" for index in range(forced_count)))


def _extract_batch(
    *,
    session: Any,
    reference: Any,
    prompt_result: Mapping[str, Any],
    hidden_size: int,
    request_cache_key: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    forced_count = _positive_int(
        prompt_result.get("forced_token_count"), "forced count"
    )
    state_count = _positive_int(prompt_result.get("state_count"), "state count")
    if state_count != forced_count + 1:
        raise Lfm2A12CausalDerivativeError(
            "A12 state count must equal forced-token count plus one."
        )
    phases = _phase_names(forced_count)
    inventory = set(reference.ordered_tensor_names)
    selected = {
        "input.attention_mask",
        "input.forced_decode_ids",
        "input.input_ids",
    }
    for phase in phases:
        selected.add(f"{phase}.output.lm_head_input")
        if phase != "prefill":
            selected.add(f"{phase}.forced_token_ids")
    names = tuple(sorted(selected))
    missing = sorted(name for name in names if name not in inventory)
    if missing:
        raise Lfm2A12CausalDerivativeError(
            f"Trace batch {reference.index} lacks derivative tensors: {missing}."
        )
    payload = session.load_selected_tensors(reference.index, names)
    try:
        tensors = payload.tensors
        prompt = tensors["input.input_ids"].detach().cpu().to(torch.int64)
        attention = tensors["input.attention_mask"].detach().cpu().to(torch.int64)
        forced = tensors["input.forced_decode_ids"].detach().cpu().to(torch.int64)
        if (
            prompt.ndim != 2
            or int(prompt.shape[0]) != 1
            or tuple(attention.shape) != tuple(prompt.shape)
            or not bool((attention == 1).all())
            or tuple(forced.shape) != (1, forced_count)
            or int(prompt.shape[1]) != int(prompt_result["prompt_token_count"])
        ):
            raise Lfm2A12CausalDerivativeError(
                "Trace prompt inputs are not canonical unpadded A12 rows."
            )
        expected_forced = torch.tensor(
            [
                int(row["next_forced_token_id"])
                for row in prompt_result["parent_reference_tokens"][:-1]
            ],
            dtype=torch.int64,
        )
        if not torch.equal(forced[0], expected_forced):
            raise Lfm2A12CausalDerivativeError(
                "Cached forced IDs differ from the authenticated A12 trajectory."
            )
        hidden_rows: list[torch.Tensor] = []
        for phase_index, phase in enumerate(phases):
            hidden = tensors[f"{phase}.output.lm_head_input"]
            if (
                hidden.ndim != 3
                or int(hidden.shape[0]) != 1
                or int(hidden.shape[-1]) != hidden_size
            ):
                raise Lfm2A12CausalDerivativeError(
                    f"Trace phase {phase!r} has invalid hidden geometry."
                )
            hidden_rows.append(hidden[0, -1, :].detach().cpu().to(torch.float32))
            if phase_index:
                observed = tensors[f"{phase}.forced_token_ids"].to(torch.int64)
                if not torch.equal(observed, forced[:, phase_index - 1 : phase_index]):
                    raise Lfm2A12CausalDerivativeError(
                        f"Trace phase {phase!r} lost forced-token identity."
                    )
        output = {
            "prompt_token_ids": prompt[0].contiguous(),
            "forced_token_ids": forced[0].contiguous(),
            "parent_hidden_states": torch.stack(hidden_rows).contiguous(),
        }
        body = {
            "schema": BATCH_RECEIPT_SCHEMA,
            "request_cache_key": request_cache_key,
            "batch_index": int(reference.index),
            "prompt_id": str(prompt_result["prompt_id"]),
            "trace_selective_session_sha256": session.evidence_sha256,
            "trace_input_sha256": reference.input_sha256,
            "trace_record_sha256": reference.record_sha256,
            "selected_payload_sha256": payload.selection_sha256,
            "a12_prompt_result_sha256": prompt_result["result_sha256"],
            "a12_parent_trajectory_receipt_sha256": prompt_result[
                "parent_trajectory_receipt_sha256"
            ],
            "prompt_token_ids_sha256": _tensor_sha256(output["prompt_token_ids"]),
            "forced_token_ids_sha256": _tensor_sha256(output["forced_token_ids"]),
            "parent_hidden_states_sha256": _tensor_sha256(
                output["parent_hidden_states"]
            ),
            "prompt_token_count": int(output["prompt_token_ids"].numel()),
            "forced_token_count": int(output["forced_token_ids"].numel()),
            "state_count": int(output["parent_hidden_states"].shape[0]),
            "hidden_size": int(output["parent_hidden_states"].shape[1]),
            "parent_hidden_publication_dtype": "float32",
            "only_selected_cached_tensors_loaded": True,
            "trace_logits_loaded": False,
            "operation_audit": dict(_ZERO_OPERATION_AUDIT),
        }
        return output, {**body, "artifact_sha256": canonical_sha256(body)}
    finally:
        session.release_selected_tensors(payload)


def _publish_batch_checkpoint(
    directory: Path,
    tensors: Mapping[str, torch.Tensor],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    index = int(receipt["batch_index"])
    tensor_path = directory / f"batch-{index:06d}.safetensors"
    receipt_path = directory / f"batch-{index:06d}.json"
    _atomic_save_tensors(tensor_path, tensors)
    published = {
        **_plain_json(receipt),
        "tensor_file_sha256": _file_sha256(tensor_path),
    }
    body = dict(published)
    body.pop("artifact_sha256")
    published["artifact_sha256"] = canonical_sha256(body)
    _atomic_write_json(receipt_path, published)
    return published


def _load_batch_checkpoint(
    directory: Path,
    *,
    batch_index: int,
    prompt_result: Mapping[str, Any],
    request_cache_key: str,
    trace_session_sha256: str,
    hidden_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]] | None:
    tensor_path = directory / f"batch-{batch_index:06d}.safetensors"
    receipt_path = directory / f"batch-{batch_index:06d}.json"
    observed = (tensor_path.exists(), receipt_path.exists())
    if not any(observed):
        return None
    if not all(observed) or tensor_path.is_symlink() or receipt_path.is_symlink():
        raise Lfm2A12CausalDerivativeError(
            f"Batch checkpoint {batch_index} is incomplete or unsafe."
        )
    receipt = _read_json(receipt_path, f"batch {batch_index} receipt")
    supplied = receipt.pop("artifact_sha256", None)
    if (
        supplied != canonical_sha256(receipt)
        or receipt.get("schema") != BATCH_RECEIPT_SCHEMA
        or receipt.get("request_cache_key") != request_cache_key
        or receipt.get("batch_index") != batch_index
        or receipt.get("prompt_id") != prompt_result["prompt_id"]
        or receipt.get("trace_selective_session_sha256") != trace_session_sha256
        or receipt.get("a12_prompt_result_sha256") != prompt_result["result_sha256"]
        or receipt.get("tensor_file_sha256") != _file_sha256(tensor_path)
        or receipt.get("operation_audit") != _ZERO_OPERATION_AUDIT
    ):
        raise Lfm2A12CausalDerivativeError(
            f"Batch checkpoint {batch_index} authentication failed."
        )
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(BATCH_TENSOR_KEYS):
            raise Lfm2A12CausalDerivativeError(
                f"Batch checkpoint {batch_index} tensor inventory changed."
            )
        tensors = {name: handle.get_tensor(name) for name in BATCH_TENSOR_KEYS}
    if (
        tensors["prompt_token_ids"].dtype != torch.int64
        or tensors["forced_token_ids"].dtype != torch.int64
        or tensors["parent_hidden_states"].dtype != torch.float32
        or tuple(tensors["parent_hidden_states"].shape)
        != (int(prompt_result["state_count"]), hidden_size)
        or receipt.get("prompt_token_ids_sha256")
        != _tensor_sha256(tensors["prompt_token_ids"])
        or receipt.get("forced_token_ids_sha256")
        != _tensor_sha256(tensors["forced_token_ids"])
        or receipt.get("parent_hidden_states_sha256")
        != _tensor_sha256(tensors["parent_hidden_states"])
    ):
        raise Lfm2A12CausalDerivativeError(
            f"Batch checkpoint {batch_index} tensor binding failed."
        )
    return tensors, {**receipt, "artifact_sha256": supplied}


def _context_tensors(value: Lfm2CausalContextDerivative) -> dict[str, torch.Tensor]:
    return {
        "input_donor_ids": value.input_donor_ids,
        "input_context_signatures": value.input_context_signatures,
        "output_donor_ids": value.output_donor_ids,
        "output_context_signatures": value.output_context_signatures,
    }


def _authenticate_cached_completion(
    directory: Path, request: Mapping[str, Any]
) -> dict[str, Any] | None:
    completion_path = directory / "complete.json"
    if not completion_path.exists():
        return None
    completion = _read_json(completion_path, "derivative completion")
    supplied = completion.pop("completion_sha256", None)
    paths = {
        "feature_tensor_file_sha256": directory / "features.safetensors",
        "feature_receipt_file_sha256": directory / "feature-receipt.json",
        "context_tensor_file_sha256": directory / "causal-context.safetensors",
        "context_receipt_file_sha256": directory / "causal-context-receipt.json",
        "fit_prompt_manifest_file_sha256": directory / "fit-prompt-manifest.json",
        "extraction_receipt_file_sha256": directory / "extraction-receipt.json",
    }
    if (
        supplied != canonical_sha256(completion)
        or completion.get("schema") != EXTRACTION_COMPLETION_SCHEMA
        or completion.get("request_cache_key") != request["request_cache_key"]
        or any(
            not path.is_file()
            or path.is_symlink()
            or completion.get(key) != _file_sha256(path)
            for key, path in paths.items()
        )
    ):
        raise Lfm2A12CausalDerivativeError(
            "Cached derivative completion authentication failed."
        )
    return {**completion, "completion_sha256": supplied, "cache_hit": True}


def run(args: argparse.Namespace) -> dict[str, Any]:
    audit_path = args.trajectory_audit_json.expanduser().resolve()
    completion_path = args.trajectory_completion_json.expanduser().resolve()
    inventory_path = args.manual_review_inventory_json.expanduser().resolve()
    trace_path = args.trace_cache_directory.expanduser().resolve()
    output_root = args.output_directory.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    audit, completion = authenticate_completed_trajectory_audit(
        audit_path=audit_path,
        completion_path=completion_path,
        expected_audit_file_sha256=args.expected_trajectory_audit_file_sha256,
        expected_audit_sha256=args.expected_trajectory_audit_sha256,
        expected_completion_sha256=args.expected_trajectory_completion_sha256,
    )
    inventory = authenticate_manual_review_inventory(
        _read_json(inventory_path, "manual review inventory"),
        expected_inventory_sha256=args.expected_manual_review_inventory_sha256,
    )
    prompt_results, trace_receipt = _trajectory_inputs(audit)
    fit_ids = [str(row["prompt_id"]) for row in prompt_results]
    review_ids = list(inventory["prompt_ids"])
    if set(fit_ids).intersection(review_ids):
        raise Lfm2A12CausalDerivativeError(
            "A12 fit prompts overlap the frozen future review inventory."
        )
    trace_session_sha = _digest(
        audit["trace_selective_session_sha256"], "A12 trace session SHA-256"
    )
    if trace_session_sha != _digest(
        args.expected_trace_selective_session_sha256,
        "expected trace session SHA-256",
    ):
        raise Lfm2A12CausalDerivativeError("A12 trace session pin differs.")
    geometry = audit["request"]["geometry"]
    hidden_size = _positive_int(geometry["hidden_size"], "hidden size")
    donor_count = _positive_int(geometry["donor_row_count"], "donor row count")
    source_hashes = {
        "extractor": _file_sha256(Path(__file__).resolve()),
        "causal_support_module": _file_sha256(
            SRC / "neural_forge" / "hf_lfm2_causal_semantic_support.py"
        ),
    }
    request_body = {
        "schema": EXTRACTION_REQUEST_SCHEMA,
        "trajectory_audit_file_sha256": args.expected_trajectory_audit_file_sha256,
        "trajectory_audit_sha256": audit["audit_sha256"],
        "trajectory_completion_sha256": completion["completion_sha256"],
        "trace_cache_key": args.expected_trace_cache_key,
        "trace_parent_fingerprint": args.expected_trace_parent_fingerprint,
        "trace_selective_session_sha256": trace_session_sha,
        "trace_manifest_sha256": trace_receipt["manifest_sha256"],
        "trace_completion_sha256": trace_receipt["completion_sha256"],
        "fit_prompt_ids": fit_ids,
        "fit_prompt_ids_sha256": canonical_sha256(fit_ids),
        "manual_review_inventory_sha256": inventory["inventory_sha256"],
        "manual_review_prompt_ids": review_ids,
        "manual_review_prompt_ids_sha256": inventory["prompt_ids_sha256"],
        "selection_and_manual_review_prompt_ids_disjoint": True,
        "donor_row_count": donor_count,
        "hidden_size": hidden_size,
        "expected_output_state_count": sum(
            int(row["state_count"]) for row in prompt_results
        ),
        "position_bin_count": _positive_int(
            args.position_bin_count, "position bin count"
        ),
        "context_hash_bucket_count": _positive_int(
            args.context_hash_bucket_count, "context bucket count"
        ),
        "topology_sha256": _digest(args.topology_sha256, "topology SHA-256"),
        "numeric_contract_sha256": _digest(
            args.numeric_contract_sha256, "numeric contract SHA-256"
        ),
        "source_implementation_sha256s": source_hashes,
        "support_only_contract": {
            "direct_4096_anchors_preserved": True,
            "exact_carrier_replanting_incorporated": False,
            "carrier_replanting_claim_permitted": False,
        },
        "next_stage_compatibility": {
            "existing_runner": "experiments/run_lfm2_carrier_reclosure.py",
            "required_evidence": (
                "A12-fit-manifest-authenticated-carrier-demand-and-reclaim-policy"
            ),
            "required_reclosure": (
                "independent-C_in-C_out-displaced-and-nondirect-residual-OMP"
            ),
            "evaluation_inventory_sha256": inventory["inventory_sha256"],
        },
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    request = {
        **request_body,
        "request_cache_key": canonical_sha256(request_body),
    }
    directory = output_root / request["request_cache_key"]
    directory.mkdir(parents=True, exist_ok=True)
    progress_path = directory / "progress.jsonl"
    _atomic_write_json(directory / "request.json", request)
    cached = _authenticate_cached_completion(directory, request)
    if cached is not None:
        return cached
    _progress(
        progress_path,
        "request-authenticated",
        request_cache_key=request["request_cache_key"],
        fit_prompt_count=len(fit_ids),
        manual_review_prompt_count=len(review_ids),
    )
    checkpoint_directory = directory / "batch-checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    loaded: list[tuple[dict[str, torch.Tensor], dict[str, Any]] | None] = []
    for index, prompt_result in enumerate(prompt_results):
        loaded.append(
            _load_batch_checkpoint(
                checkpoint_directory,
                batch_index=index,
                prompt_result=prompt_result,
                request_cache_key=request["request_cache_key"],
                trace_session_sha256=trace_session_sha,
                hidden_size=hidden_size,
            )
        )
    missing = [index for index, value in enumerate(loaded) if value is None]
    if missing:
        cache = Lfm2CircuitTraceCache.open(trace_path, verify=False)
        if (
            cache.cache_key != args.expected_trace_cache_key
            or cache.parent_fingerprint != args.expected_trace_parent_fingerprint
            or cache.batch_count != len(prompt_results)
        ):
            raise Lfm2A12CausalDerivativeError(
                "Trace cache identity or geometry differs from A12."
            )
        _progress(
            progress_path,
            "trace-authentication-started",
            missing_batch_indices=missing,
        )
        with cache.selective_authenticated_session() as session:
            if session.evidence_sha256 != trace_session_sha:
                raise Lfm2A12CausalDerivativeError(
                    "Live trace session differs from the session consumed by A12."
                )
            for index in missing:
                tensors, receipt = _extract_batch(
                    session=session,
                    reference=session.batch_references[index],
                    prompt_result=prompt_results[index],
                    hidden_size=hidden_size,
                    request_cache_key=request["request_cache_key"],
                )
                published = _publish_batch_checkpoint(
                    checkpoint_directory, tensors, receipt
                )
                loaded[index] = (tensors, published)
                _progress(
                    progress_path,
                    "batch-checkpoint-published",
                    batch_index=index,
                    prompt_id=prompt_results[index]["prompt_id"],
                    state_count=int(tensors["parent_hidden_states"].shape[0]),
                    artifact_sha256=published["artifact_sha256"],
                )
            session.verify()
        _progress(progress_path, "trace-authentication-complete")
    else:
        _progress(progress_path, "all-batch-checkpoints-reused")
    if any(value is None for value in loaded):  # pragma: no cover - defensive
        raise Lfm2A12CausalDerivativeError("A derivative batch remained absent.")
    batches = [value for value in loaded if value is not None]
    prompt_tokens = {
        fit_ids[index]: batch[0]["prompt_token_ids"].tolist()
        for index, batch in enumerate(batches)
    }
    batch_receipts = {
        fit_ids[index]: str(batch[1]["artifact_sha256"])
        for index, batch in enumerate(batches)
    }
    output_features = torch.cat(
        [batch[0]["parent_hidden_states"] for batch in batches], dim=0
    ).contiguous()
    if tuple(output_features.shape) != (
        request["expected_output_state_count"],
        hidden_size,
    ):
        raise Lfm2A12CausalDerivativeError(
            "Stacked cached-parent hidden feature geometry is not A12-aligned."
        )
    input_features = torch.eye(hidden_size, dtype=torch.float32)
    context = build_lfm2_causal_context_derivative(
        trajectory_audit=audit,
        prompt_token_ids_by_prompt=prompt_tokens,
        fit_prompt_ids=fit_ids,
        manual_review_prompt_ids=review_ids,
        donor_count=donor_count,
        position_bin_count=request["position_bin_count"],
        context_hash_bucket_count=request["context_hash_bucket_count"],
        topology_sha256=request["topology_sha256"],
        numeric_contract_sha256=request["numeric_contract_sha256"],
    )
    context_tensor_path = directory / "causal-context.safetensors"
    context_receipt_path = directory / "causal-context-receipt.json"
    _atomic_save_tensors(context_tensor_path, _context_tensors(context))
    _atomic_write_json(context_receipt_path, context.as_dict())
    feature_pack = seal_lfm2_causal_support_feature_pack(
        input_feature_transform=input_features,
        output_feature_transform=output_features,
        trajectory_audit_sha256=audit["audit_sha256"],
        fit_prompt_ids=fit_ids,
        manual_review_prompt_ids=review_ids,
        topology_sha256=request["topology_sha256"],
        numeric_contract_sha256=request["numeric_contract_sha256"],
        source_artifact_sha256s={
            "a12_completion": completion["completion_sha256"],
            "trace_selective_session": trace_session_sha,
            "batch_checkpoint_inventory": canonical_sha256(batch_receipts),
            "causal_context_derivative": context.artifact_sha256,
            "manual_review_inventory": inventory["inventory_sha256"],
        },
        input_feature_kind="explicit-c-in-hidden-identity-v1",
        output_feature_kind="a12-cached-parent-hidden-c-out-features-v1",
    )
    feature_tensor_path = directory / "features.safetensors"
    feature_receipt_path = directory / "feature-receipt.json"
    _atomic_save_tensors(
        feature_tensor_path,
        {
            "input_feature_transform": input_features,
            "output_feature_transform": output_features,
        },
    )
    _atomic_write_json(feature_receipt_path, feature_pack.as_dict())
    fit_manifest = build_fit_prompt_manifest(
        trajectory_audit_sha256=audit["audit_sha256"],
        feature_pack_artifact_sha256=feature_pack.artifact_sha256,
        fit_prompt_ids=fit_ids,
        manual_review_prompt_ids=review_ids,
        prompt_token_ids_by_prompt=prompt_tokens,
        source_trace_batch_receipt_sha256s=batch_receipts,
    )
    fit_manifest_path = directory / "fit-prompt-manifest.json"
    _atomic_write_json(fit_manifest_path, fit_manifest)
    extraction_body = {
        "schema": EXTRACTION_RECEIPT_SCHEMA,
        "status": "complete-cached-a12-causal-support-derivative",
        "request": request,
        "feature_artifact_sha256": feature_pack.artifact_sha256,
        "feature_tensor_file_sha256": _file_sha256(feature_tensor_path),
        "causal_context_artifact_sha256": context.artifact_sha256,
        "causal_context_tensor_file_sha256": _file_sha256(context_tensor_path),
        "fit_prompt_manifest_sha256": fit_manifest["manifest_sha256"],
        "manual_review_inventory_sha256": inventory["inventory_sha256"],
        "batch_checkpoint_receipt_sha256s": batch_receipts,
        "batch_checkpoint_inventory_sha256": canonical_sha256(batch_receipts),
        "input_feature_shape": list(input_features.shape),
        "output_feature_shape": list(output_features.shape),
        "output_state_count": int(output_features.shape[0]),
        "hidden_size": hidden_size,
        "support_only_contract": request["support_only_contract"],
        "next_stage_compatibility": request["next_stage_compatibility"],
        "manual_review_prompts_consumed": 0,
        "network_access_performed": False,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }
    extraction_receipt = {
        **extraction_body,
        "artifact_sha256": canonical_sha256(extraction_body),
    }
    extraction_path = directory / "extraction-receipt.json"
    _atomic_write_json(extraction_path, extraction_receipt)
    completion_body = {
        "schema": EXTRACTION_COMPLETION_SCHEMA,
        "request_cache_key": request["request_cache_key"],
        "artifact_sha256": extraction_receipt["artifact_sha256"],
        "feature_artifact_sha256": feature_pack.artifact_sha256,
        "context_artifact_sha256": context.artifact_sha256,
        "fit_prompt_manifest_sha256": fit_manifest["manifest_sha256"],
        "feature_tensor_file_sha256": _file_sha256(feature_tensor_path),
        "feature_receipt_file_sha256": _file_sha256(feature_receipt_path),
        "context_tensor_file_sha256": _file_sha256(context_tensor_path),
        "context_receipt_file_sha256": _file_sha256(context_receipt_path),
        "fit_prompt_manifest_file_sha256": _file_sha256(fit_manifest_path),
        "extraction_receipt_file_sha256": _file_sha256(extraction_path),
        "atomic_batch_checkpoints": True,
        "reusable_identical_math": True,
    }
    published_completion = {
        **completion_body,
        "completion_sha256": canonical_sha256(completion_body),
    }
    _atomic_write_json(directory / "complete.json", published_completion)
    _progress(
        progress_path,
        "publication-complete",
        artifact_sha256=extraction_receipt["artifact_sha256"],
        feature_artifact_sha256=feature_pack.artifact_sha256,
    )
    return {**published_completion, "cache_hit": False}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-audit-json", type=Path, required=True)
    parser.add_argument("--trajectory-completion-json", type=Path, required=True)
    parser.add_argument("--expected-trajectory-audit-file-sha256", required=True)
    parser.add_argument("--expected-trajectory-audit-sha256", required=True)
    parser.add_argument("--expected-trajectory-completion-sha256", required=True)
    parser.add_argument("--trace-cache-directory", type=Path, required=True)
    parser.add_argument("--expected-trace-cache-key", required=True)
    parser.add_argument("--expected-trace-parent-fingerprint", required=True)
    parser.add_argument("--expected-trace-selective-session-sha256", required=True)
    parser.add_argument("--manual-review-inventory-json", type=Path, required=True)
    parser.add_argument("--expected-manual-review-inventory-sha256", required=True)
    parser.add_argument("--topology-sha256", required=True)
    parser.add_argument("--numeric-contract-sha256", required=True)
    parser.add_argument("--position-bin-count", type=int, default=4)
    parser.add_argument("--context-hash-bucket-count", type=int, default=256)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "schema": EXTRACTION_RECEIPT_SCHEMA,
                    "status": "failed",
                    "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "message": str(exc),
                },
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )
        raise
    print(
        json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
