"""Evaluate one completed carrier bridge on a frozen disjoint prompt inventory.

This is an evaluation-only runtime. It authenticates a previously compiled
standalone carrier bridge, installs its already-materialized target table into
the existing same-body child, and binds its packaged C_in/C_out routes. It does
not load a parent model, compile or solve a route, capture a trace, train, prune,
quantize, share parameters, or create a runtime sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from safetensors.torch import load_file

from experiments.run_lfm2_carrier_reclosure import (
    _materialization_tensor_sha256,
    carrier_bridge_allocations_from_publication,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_direct_local import (
    Lfm2DirectLocalCheckpointIO,
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_semantic_step_model_runtime import (
    run_lfm2_semantic_step_model_review,
)
from neural_forge.hf_lfm2_semantic_step_token_fold import (
    bind_lfm2_separate_sparse_semantic_code_runtime,
)
from neural_forge.hf_tokenizer_bridge import load_strict_tokenizer_bridge_artifact
from neural_forge.lfm2_decoded_review import response_termination_evidence


DOSSIER_SCHEMA = "neural-forge-lfm2-carrier-frozen-review-dossier-v1"
ATTEMPT_SCHEMA = "neural-forge-lfm2-carrier-frozen-review-attempt-v1"
COMPLETION_SCHEMA = "neural-forge-lfm2-carrier-frozen-review-completion-v1"
_INITIAL_CAP = 1024
_ADAPTIVE_CAP = 2048
_FORBIDDEN_EVALUATION_AUDIT = {
    "training_steps": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "autograd_grad_calls": 0,
    "backward_calls": 0,
    "parent_forward_calls": 0,
    "parent_generate_calls": 0,
    "child_generate_calls": 0,
    "child_lm_head_forward_calls": 0,
    "registered_parameter_count_delta": 0,
    "registered_module_count_delta": 0,
    "quantized_parameters": 0,
    "pruned_parameters": 0,
    "network_access": False,
}


class Lfm2CarrierEvaluationError(RuntimeError):
    """The authenticated reuse-only carrier evaluation contract was violated."""


def _plain(value: Any) -> Any:
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _plain(as_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise Lfm2CarrierEvaluationError(
        f"Evidence contains unsupported value {type(value).__name__}."
    )


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CarrierEvaluationError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CarrierEvaluationError(f"{label} must be a positive integer.")
    return int(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _file(path: str | Path, label: str) -> Path:
    lexical = Path(path).expanduser().absolute()
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise Lfm2CarrierEvaluationError(f"{label} is missing.") from error
    if lexical.is_symlink() or resolved != lexical or not resolved.is_file():
        raise Lfm2CarrierEvaluationError(f"{label} must be one canonical file.")
    return resolved


def _directory(path: str | Path, label: str) -> Path:
    lexical = Path(path).expanduser().absolute()
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise Lfm2CarrierEvaluationError(f"{label} is missing.") from error
    if lexical.is_symlink() or resolved != lexical or not resolved.is_dir():
        raise Lfm2CarrierEvaluationError(
            f"{label} must be one canonical directory."
        )
    return resolved


def _output_directory(path: str | Path) -> Path:
    lexical = Path(path).expanduser().absolute()
    if lexical.exists() or lexical.is_symlink():
        if lexical.is_symlink() or not lexical.is_dir():
            raise Lfm2CarrierEvaluationError(
                "Output path exists but is not a real directory."
            )
    else:
        lexical.mkdir(parents=True)
    return lexical.resolve(strict=True)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Lfm2CarrierEvaluationError(f"{label} is not readable JSON.") from error
    if not isinstance(value, dict):
        raise Lfm2CarrierEvaluationError(f"{label} must be a JSON object.")
    return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(_plain(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_seal(
    value: Mapping[str, Any],
    *,
    field: str,
    expected: str,
    label: str,
) -> dict[str, Any]:
    body = dict(value)
    supplied = body.pop(field, None)
    if supplied != _sha256(expected, f"expected {label} seal"):
        raise Lfm2CarrierEvaluationError(f"{label} differs from its expected seal.")
    if supplied != canonical_sha256(body):
        raise Lfm2CarrierEvaluationError(f"{label} self-seal does not verify.")
    return {**body, field: supplied}


def _load_frozen_prompts(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_inventory_sha256: str,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if _file_sha256(path) != _sha256(
        expected_file_sha256, "expected frozen prompt file SHA-256"
    ):
        raise Lfm2CarrierEvaluationError("Frozen prompt file bytes changed.")
    inventory = _verify_seal(
        _read_json(path, "frozen prompt inventory"),
        field="inventory_sha256",
        expected=expected_inventory_sha256,
        label="frozen prompt inventory",
    )
    contract = inventory.get("review_contract")
    disjoint = inventory.get("a12_disjointness_proof")
    rows = inventory.get("prompts")
    if (
        inventory.get("schema")
        != "neural-forge-lfm2-causal-support-future-manual-review-v1"
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
        or not isinstance(disjoint, Mapping)
        or disjoint.get("exact_prompt_id_intersection_count") != 0
        or disjoint.get("exact_prompt_text_intersection_count") != 0
        or not isinstance(rows, list)
        or len(rows) != 18
    ):
        raise Lfm2CarrierEvaluationError(
            "Frozen prompt inventory does not satisfy the disjoint review contract."
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise Lfm2CarrierEvaluationError("Every frozen prompt must be an object.")
        prompt_id = row.get("id")
        text = row.get("text")
        safety = row.get("safety_review_only")
        if (
            not isinstance(prompt_id, str)
            or not prompt_id
            or prompt_id in seen
            or not isinstance(text, str)
            or not text.strip()
            or not isinstance(safety, bool)
        ):
            raise Lfm2CarrierEvaluationError("Frozen prompt row is invalid.")
        seen.add(prompt_id)
        normalized.append(
            {
                "id": prompt_id,
                "category": str(row.get("category", "general")),
                "safety_review_only": safety,
                "prompt": text,
            }
        )
    if sum(bool(row["safety_review_only"]) for row in normalized) != 3:
        raise Lfm2CarrierEvaluationError(
            "Frozen prompt inventory must contain 15 intelligence and 3 safety rows."
        )
    return inventory, tuple(normalized)


def _load_span_evidence(path: Path, expected_sha256: str) -> dict[str, Any]:
    return _verify_seal(
        _read_json(path, "cached tokenizer span evidence"),
        field="evidence_sha256",
        expected=expected_sha256,
        label="cached tokenizer span evidence",
    )


def _attempt_path(root: Path, prompt_id: str, cap: int) -> Path:
    safe = hashlib.sha256(prompt_id.encode("utf-8")).hexdigest()[:16]
    return root / "attempts" / f"{safe}-cap{cap}.json"


def _progress_callback(path: Path, *, prompt_id: str, cap: int):
    def callback(record: Mapping[str, Any]) -> None:
        stage = record.get("stage")
        step = record.get("step_index")
        if stage == "semantic-step-complete" and isinstance(step, int):
            if (step + 1) % 32 != 0 and not record.get("eos_selected") and not record.get(
                "generation_limit_selected"
            ):
                return
        line = {
            "prompt_id": prompt_id,
            "maximum_new_tokens": cap,
            "runtime_progress": _plain(record),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")
            handle.flush()
    return callback


def _load_or_run_attempt(
    *,
    root: Path,
    prompt: Mapping[str, Any],
    cap: int,
    child: Any,
    routing: Any,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_id = str(prompt["id"])
    path = _attempt_path(root, prompt_id, cap)
    expected_identity = {
        **dict(identity),
        "prompt_id": prompt_id,
        "prompt_sha256": canonical_sha256(prompt),
        "maximum_new_tokens": cap,
    }
    if path.is_file() and not path.is_symlink():
        value = _read_json(path, "cached evaluation attempt")
        body = dict(value)
        supplied = body.pop("attempt_sha256", None)
        if (
            body.get("schema") == ATTEMPT_SCHEMA
            and supplied == canonical_sha256(body)
            and body.get("identity") == expected_identity
            and body.get("runtime_error") is None
        ):
            print(
                json.dumps(
                    {"stage": "attempt-cache-hit", "prompt_id": prompt_id, "cap": cap},
                    sort_keys=True,
                ),
                flush=True,
            )
            return {**body, "attempt_sha256": supplied}
        if body.get("identity") != expected_identity:
            raise Lfm2CarrierEvaluationError(
                f"Cached attempt for {prompt_id!r} does not match this evaluation."
            )
        print(
            json.dumps(
                {"stage": "attempt-runtime-error-retry", "prompt_id": prompt_id, "cap": cap},
                sort_keys=True,
            ),
            flush=True,
        )
    progress_path = root / "progress" / f"{prompt_id}-cap{cap}.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        runtime = run_lfm2_semantic_step_model_review(
            child,
            routing,
            donor_tokenizer,
            target_tokenizer,
            [prompt],
            maximum_prompt_tokens=8192,
            maximum_new_tokens=cap,
            maximum_embedding_rows_per_chunk=1024,
            maximum_donor_rows_per_score_chunk=4096,
            progress_callback=_progress_callback(
                progress_path, prompt_id=prompt_id, cap=cap
            ),
        )
        response = _plain(runtime["responses"][0])
        termination = _plain(response["termination_evidence"])
        runtime_error = None
    except Exception as error:  # preserve one prompt failure and continue
        runtime = None
        runtime_error = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        response = {
            **dict(prompt),
            "response": "",
            "response_token_ids": [],
            "termination_evidence": response_termination_evidence(
                ended_with_eos=False,
                generation_limit_reached=False,
                runtime_error=runtime_error,
            ),
            "diagnostics": {"runtime_error": runtime_error},
        }
        termination = _plain(response["termination_evidence"])
    body = {
        "schema": ATTEMPT_SCHEMA,
        "identity": expected_identity,
        "response": response,
        "runtime": _plain(runtime),
        "runtime_error": runtime_error,
        "termination_evidence": termination,
        "automatic_intelligence_decision": None,
        "output_cap_hit_counts_as_model_failure": False,
        "safety_behavior_counts_as_intelligence_failure": False,
    }
    result = {**body, "attempt_sha256": canonical_sha256(body)}
    _atomic_write(path, result)
    return result


def _validate_evaluation_audit(attempt: Mapping[str, Any]) -> None:
    runtime = attempt.get("runtime")
    if runtime is None:
        return
    if not isinstance(runtime, Mapping) or not isinstance(
        runtime.get("operation_audit"), Mapping
    ):
        raise Lfm2CarrierEvaluationError("Completed attempt has no operation audit.")
    audit = runtime["operation_audit"]
    failures = [
        key for key, expected in _FORBIDDEN_EVALUATION_AUDIT.items()
        if audit.get(key) != expected
    ]
    if failures:
        raise Lfm2CarrierEvaluationError(
            "Evaluation performed a forbidden operation: " + ", ".join(failures)
        )


def _manual_markdown(dossier: Mapping[str, Any]) -> str:
    lines = [
        "# A12 carrier bridge — frozen disjoint-18 manual review",
        "",
        "This is a human review sheet, not an automatic gate. Output-cap hits are "
        "censored observations, not model failures. The three safety prompts are "
        "reviewed separately and never lower intelligence retention because of a "
        "refusal or policy difference.",
        "",
        f"- Carrier artifact: `{dossier['carrier_bridge_artifact_sha256']}`",
        f"- Frozen inventory: `{dossier['prompt_inventory_sha256']}`",
        f"- Effective child parameters: {dossier['effective_parameter_accounting']['effective_child_parameter_count']:,}",
        f"- Parent-to-child ratio: {dossier['effective_parameter_accounting']['effective_parameter_compaction_ratio']:.6f}x",
        "",
    ]
    rows = dossier["responses"]
    for safety, heading in ((False, "Intelligence prompts"), (True, "Safety-only prompts")):
        lines.extend([f"## {heading}", ""])
        for row in rows:
            if bool(row["safety_review_only"]) != safety:
                continue
            termination = row["termination_evidence"]
            lines.extend(
                [
                    f"### {row['id']}",
                    "",
                    f"Category: `{row['category']}`  ",
                    f"Selected cap: `{row['selected_maximum_new_tokens']}`  ",
                    f"Termination: `{termination['completion_status']}` / `{termination['finish_reason']}`  ",
                    f"Censored by cap: `{termination['censored_by_output_cap']}`",
                    "",
                    "Prompt:",
                    "",
                    *[f"    {line}" for line in str(row["prompt"]).splitlines()],
                    "",
                    "Response:",
                    "",
                    *[
                        f"    {line}"
                        for line in str(row["response"]).splitlines()
                    ],
                    "",
                ]
            )
            if safety:
                lines.extend(
                    [
                        "Manual safety review (separate from intelligence): `pending`",
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        "Manual coherence/correctness/instruction-following review: `pending`",
                        "",
                    ]
                )
    return "\n".join(lines).rstrip() + "\n"


# Scoped, not process-global: a bare ``torch.set_grad_enabled(False)`` here
# leaked into the calling thread for the rest of the interpreter and silently
# disabled autograd for everything that ran afterwards.
@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    prompt_path = _file(args.prompts_json, "frozen prompt inventory")
    inventory, prompts = _load_frozen_prompts(
        prompt_path,
        expected_file_sha256=args.expected_prompts_file_sha256,
        expected_inventory_sha256=args.expected_prompt_inventory_sha256,
    )
    carrier_directory = _directory(args.carrier_bridge_directory, "carrier bridge")
    allocations = carrier_bridge_allocations_from_publication(
        carrier_directory,
        expected_artifact_sha256=args.expected_carrier_bridge_artifact_sha256,
    )
    carrier_receipt = _read_json(
        carrier_directory / "bridge-receipt.json", "carrier bridge receipt"
    )
    carrier_tensors = load_file(
        str(carrier_directory / "bridge.safetensors"), device="cpu"
    )
    target_table = carrier_tensors["target_table"]
    if target_table.dtype is not torch.bfloat16:
        raise Lfm2CarrierEvaluationError(
            "Carrier evaluation requires the packaged BF16 target table unchanged."
        )

    donor_directory = _directory(args.donor_tokenizer_directory, "donor tokenizer")
    target_directory = _directory(args.target_tokenizer_directory, "target tokenizer")
    strict_bridge_directory = _directory(
        args.strict_tokenizer_bridge_directory, "strict tokenizer bridge"
    )
    checkpoint_directory = _directory(args.checkpoint_directory, "same-body checkpoint")
    span_path = _file(args.span_evidence_json, "cached tokenizer span evidence")
    span_evidence = _load_span_evidence(
        span_path, args.expected_span_evidence_sha256
    )
    donor_tokenizer = load_lfm2_direct_local_tokenizer(donor_directory)
    target_tokenizer = load_lfm2_direct_local_tokenizer(target_directory)
    strict_bridge = load_strict_tokenizer_bridge_artifact(
        strict_bridge_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=args.expected_strict_tokenizer_bridge_artifact_sha256,
    )
    device = torch.device(args.runtime_device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise Lfm2CarrierEvaluationError("A CUDA evaluation device is required.")
    if target_table.dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise Lfm2CarrierEvaluationError(
            "Selected GPU cannot execute the packaged BF16 carrier table."
        )
    checkpoint_io = Lfm2DirectLocalCheckpointIO(
        tokenizer_bridge=strict_bridge,
        device=torch.device("cpu"),
        dtype=target_table.dtype,
    )
    child, load_receipt = checkpoint_io.load_authenticated_portable(
        checkpoint_directory,
        runtime_device=device,
        runtime_dtype=target_table.dtype,
    )
    child.requires_grad_(False)
    child.eval()
    physical_parameters = sum(int(parameter.numel()) for parameter in child.parameters())
    if physical_parameters != _positive_int(
        args.expected_physical_child_parameters, "physical child parameters"
    ):
        raise Lfm2CarrierEvaluationError("Loaded child parameter count changed.")
    live_table = child.get_input_embeddings().weight
    output_table = child.get_output_embeddings().weight
    if live_table.data_ptr() != output_table.data_ptr():
        raise Lfm2CarrierEvaluationError("Same-body child input/output table is not tied.")
    if tuple(live_table.shape) != tuple(target_table.shape):
        raise Lfm2CarrierEvaluationError("Carrier target table shape changed.")
    with torch.no_grad():
        live_table.copy_(target_table.to(device=device, dtype=live_table.dtype))
    expected_target_sha256 = carrier_receipt["materialization_receipt"][
        "target_table_sha256"
    ]
    observed_target_sha256 = _materialization_tensor_sha256(live_table)
    if observed_target_sha256 != expected_target_sha256:
        raise Lfm2CarrierEvaluationError(
            "Live child table differs from the compiled carrier target table."
        )
    child_state_sha256 = fingerprint_lfm2_direct_local_state(child)
    routing = bind_lfm2_separate_sparse_semantic_code_runtime(
        span_evidence,
        live_table,
        input_semantic_code_offsets=carrier_tensors["input_offsets"],
        input_semantic_code_target_ids=carrier_tensors["input_target_ids"],
        input_semantic_code_coefficients=carrier_tensors["input_coefficients"],
        output_semantic_code_offsets=carrier_tensors["output_offsets"],
        output_semantic_code_target_ids=carrier_tensors["output_target_ids"],
        output_semantic_code_coefficients=carrier_tensors["output_coefficients"],
        target_to_donor=carrier_tensors["target_to_donor"],
    )
    routing_receipt = _plain(routing.as_dict())
    routing_sha256 = canonical_sha256(routing_receipt)
    output_root = _output_directory(args.output_directory)
    identity = {
        "carrier_bridge_artifact_sha256": _sha256(
            args.expected_carrier_bridge_artifact_sha256,
            "carrier bridge artifact SHA-256",
        ),
        "child_state_sha256": child_state_sha256,
        "routing_contract_sha256": routing_sha256,
        "prompt_inventory_sha256": inventory["inventory_sha256"],
        "span_evidence_sha256": span_evidence["evidence_sha256"],
        "strict_tokenizer_bridge_artifact_sha256": strict_bridge.artifact_sha256,
    }
    responses = []
    selected_attempts = []
    for index, prompt in enumerate(prompts):
        print(
            json.dumps(
                {
                    "stage": "prompt-start",
                    "prompt_index": index,
                    "prompt_count": len(prompts),
                    "prompt_id": prompt["id"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        attempts = []
        for cap in (_INITIAL_CAP, _ADAPTIVE_CAP):
            if attempts and attempts[-1]["termination_evidence"].get(
                "censored_by_output_cap"
            ) is not True:
                break
            attempt = _load_or_run_attempt(
                root=output_root,
                prompt=prompt,
                cap=cap,
                child=child,
                routing=routing,
                donor_tokenizer=donor_tokenizer,
                target_tokenizer=target_tokenizer,
                identity=identity,
            )
            _validate_evaluation_audit(attempt)
            attempts.append(attempt)
            if attempt.get("runtime_error") is not None:
                break
        selected = attempts[-1]
        selected_attempts.append(selected)
        response = selected["response"]
        responses.append(
            {
                "id": prompt["id"],
                "category": prompt["category"],
                "safety_review_only": prompt["safety_review_only"],
                "prompt": prompt["prompt"],
                "response": response.get("response", ""),
                "response_token_ids": response.get("response_token_ids", []),
                "selected_maximum_new_tokens": selected["identity"][
                    "maximum_new_tokens"
                ],
                "attempt_sha256s": [row["attempt_sha256"] for row in attempts],
                "termination_evidence": selected["termination_evidence"],
                "diagnostics": response.get("diagnostics"),
                "runtime_error": selected.get("runtime_error"),
                "manual_intelligence_review": (
                    None if prompt["safety_review_only"] is False else "not-applicable"
                ),
                "manual_safety_review": (
                    None if prompt["safety_review_only"] is True else "not-applicable"
                ),
                "automatic_intelligence_decision": None,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "prompt-complete",
                    "prompt_id": prompt["id"],
                    "completion_status": selected["termination_evidence"][
                        "completion_status"
                    ],
                    "selected_cap": selected["identity"]["maximum_new_tokens"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    parent_parameters = _positive_int(args.parent_parameters, "parent parameters")
    route_parameters = sum(value.parameter_elements for value in allocations)
    effective_parameters = physical_parameters + route_parameters
    accounting = {
        "parent_parameter_count": parent_parameters,
        "physical_child_parameter_count": physical_parameters,
        "continuous_carrier_route_parameter_count": route_parameters,
        "effective_child_parameter_count": effective_parameters,
        "effective_parameter_compaction_ratio": parent_parameters / effective_parameters,
        "effective_parameter_reduction_fraction": 1.0
        - effective_parameters / parent_parameters,
        "quantization_credit": 0,
        "integer_routing_metadata_counted_as_parameters": 0,
        "all_C_in_and_C_out_coefficients_counted_exactly_once": True,
    }
    capped = sum(
        row["termination_evidence"].get("censored_by_output_cap") is True
        for row in responses
    )
    runtime_errors = sum(row["runtime_error"] is not None for row in responses)
    body = {
        "schema": DOSSIER_SCHEMA,
        "status": "complete-awaiting-manual-review",
        "carrier_bridge_artifact_sha256": identity[
            "carrier_bridge_artifact_sha256"
        ],
        "carrier_bridge_tensor_file_sha256": carrier_receipt[
            "bridge_tensor_file_sha256"
        ],
        "carrier_route_allocations": [
            {
                "state_id": value.state_id,
                "parameter_elements": value.parameter_elements,
                "tensor_sha256": value.tensor_sha256,
                "artifact_sha256": value.artifact_sha256,
                "dtype": value.dtype,
            }
            for value in allocations
        ],
        "compiled_target_table_sha256": expected_target_sha256,
        "live_target_table_sha256": observed_target_sha256,
        "child_state_sha256": child_state_sha256,
        "checkpoint_load_receipt": _plain(load_receipt),
        "routing_contract": routing_receipt,
        "routing_contract_sha256": routing_sha256,
        "strict_tokenizer_bridge_artifact_sha256": strict_bridge.artifact_sha256,
        "span_evidence_sha256": span_evidence["evidence_sha256"],
        "prompt_inventory_sha256": inventory["inventory_sha256"],
        "prompt_file_sha256": _file_sha256(prompt_path),
        "a12_disjointness_proof": _plain(inventory["a12_disjointness_proof"]),
        "review_contract": {
            **_plain(inventory["review_contract"]),
            "output_cap_hits_are_censored_not_failures": True,
            "safety_rows_are_excluded_from_intelligence_retention": True,
            "manual_review_is_authoritative": True,
            "automatic_acceptance_or_rejection_performed": False,
        },
        "prompt_count": len(responses),
        "intelligence_prompt_count": sum(
            row["safety_review_only"] is False for row in responses
        ),
        "safety_prompt_count": sum(
            row["safety_review_only"] is True for row in responses
        ),
        "selected_output_cap_censored_count": capped,
        "runtime_error_count": runtime_errors,
        "responses": responses,
        "effective_parameter_accounting": accounting,
        "compile_operation_audit": _plain(carrier_receipt["operation_audit"]),
        "evaluation_operation_audits": [
            attempt["runtime"]["operation_audit"]
            for attempt in selected_attempts
            if attempt.get("runtime") is not None
        ],
        "parent_model_loaded_for_evaluation": False,
        "parent_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameter_pruning_calls": 0,
        "parameter_quantization_calls": 0,
        "parameter_sharing_calls": 0,
        "parameter_sparsification_calls": 0,
        "runtime_sidecar_parameters": 0,
    }
    dossier = {**body, "dossier_sha256": canonical_sha256(body)}
    dossier_path = output_root / "whole-response-dossier.json"
    manual_path = output_root / "manual-review.md"
    _atomic_write(dossier_path, dossier)
    _atomic_write_text(manual_path, _manual_markdown(dossier))
    completion_body = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "dossier_file": str(dossier_path),
        "dossier_file_sha256": _file_sha256(dossier_path),
        "dossier_sha256": dossier["dossier_sha256"],
        "manual_review_file": str(manual_path),
        "manual_review_file_sha256": _file_sha256(manual_path),
        "prompt_count": len(responses),
        "selected_output_cap_censored_count": capped,
        "runtime_error_count": runtime_errors,
        "parent_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "training_steps": 0,
    }
    completion = {
        **completion_body,
        "completion_sha256": canonical_sha256(completion_body),
    }
    _atomic_write(output_root / "complete.json", completion)
    print(json.dumps(completion, sort_keys=True), flush=True)
    return completion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-directory", required=True, type=Path)
    parser.add_argument("--donor-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--target-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--strict-tokenizer-bridge-directory", required=True, type=Path)
    parser.add_argument(
        "--expected-strict-tokenizer-bridge-artifact-sha256", required=True
    )
    parser.add_argument("--span-evidence-json", required=True, type=Path)
    parser.add_argument("--expected-span-evidence-sha256", required=True)
    parser.add_argument("--carrier-bridge-directory", required=True, type=Path)
    parser.add_argument("--expected-carrier-bridge-artifact-sha256", required=True)
    parser.add_argument("--prompts-json", required=True, type=Path)
    parser.add_argument("--expected-prompts-file-sha256", required=True)
    parser.add_argument("--expected-prompt-inventory-sha256", required=True)
    parser.add_argument("--runtime-device", default="cuda:0")
    parser.add_argument(
        "--expected-physical-child-parameters", required=True, type=int
    )
    parser.add_argument("--parent-parameters", required=True, type=int)
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
