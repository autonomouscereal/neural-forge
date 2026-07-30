"""Rerun only previously capped A9/A11 responses from persisted child state.

This is an evaluation-only runtime.  It loads an authenticated standalone child,
an already-compiled sparse semantic codebook, and (for A11) already-solved C_out
coefficients.  It never loads a parent model, captures a parent trace, discovers
supports, solves coefficients, folds parameters, trains, prunes, or quantizes.

Every source dossier is authenticated and its exact set of generation-limit
hits must equal the manifest inventory.  Each capped prompt first receives the
larger initial allowance.  Only a response that is still capped receives the
adaptive maximum allowance.  EOS, cap, loop-review, and runtime-error evidence
remain separate from the later manual folding-quality and safety observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import traceback
from collections.abc import Mapping, Sequence
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

import torch  # noqa: E402

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    Lfm2DirectLocalCheckpointIO,
    fingerprint_lfm2_direct_local_state,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_semantic_step_model_runtime import (  # noqa: E402
    run_lfm2_semantic_step_model_review,
)
from neural_forge.hf_lfm2_semantic_step_token_fold import (  # noqa: E402
    bind_lfm2_separate_sparse_semantic_code_runtime,
    bind_lfm2_sparse_semantic_code_runtime,
)
from neural_forge.hf_lfm2_sparse_semantic_codebook import (  # noqa: E402
    load_lfm2_sparse_semantic_codebook,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    load_strict_tokenizer_bridge_artifact,
)
from neural_forge.lfm2_decoded_review import (  # noqa: E402
    load_review_prompts,
    response_termination_evidence,
)

from experiments.run_lfm2_semantic_step_bridge_control import (  # noqa: E402
    resolve_checkpoint_provenance_dtype,
    resolve_runtime_dtype,
)
from experiments.run_lfm2_sparse_semantic_codebook import (  # noqa: E402
    load_authenticated_functional_output_refit,
)


PLAN_SCHEMA = "neural-forge-lfm2-capped-prompt-reuse-plan-v1"
RESULT_SCHEMA = "neural-forge-lfm2-capped-prompt-reuse-result-v1"
ATTEMPT_SCHEMA = "neural-forge-lfm2-capped-prompt-attempt-v1"
_SHA256_ZERO = "0" * 64


class Lfm2CappedPromptReuseError(RuntimeError):
    """The reuse-only capped-prompt evaluation contract was violated."""


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise Lfm2CappedPromptReuseError(
        f"Evidence contains non-JSON value {type(value).__name__}."
    )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CappedPromptReuseError(f"{name} must be positive.")
    return int(value)


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CappedPromptReuseError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_inventory_sha256(directory: Path) -> str:
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise Lfm2CappedPromptReuseError("Checkpoint contains a symlink.")
        if path.is_file():
            records.append(
                {
                    "relative_path": path.relative_to(directory).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _file_sha256(path),
                }
            )
    if not records:
        raise Lfm2CappedPromptReuseError("Checkpoint inventory is empty.")
    return canonical_sha256(records)


def _local_file(path: str | Path, name: str) -> Path:
    lexical = Path(path).expanduser().absolute()
    if lexical.is_symlink():
        raise Lfm2CappedPromptReuseError(f"{name} cannot be a link.")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2CappedPromptReuseError(f"{name} is missing.") from exc
    if resolved != lexical or not resolved.is_file():
        raise Lfm2CappedPromptReuseError(f"{name} must be one canonical file.")
    return resolved


def _local_directory(path: str | Path, name: str) -> Path:
    lexical = Path(path).expanduser().absolute()
    if lexical.is_symlink():
        raise Lfm2CappedPromptReuseError(f"{name} cannot be a link.")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2CappedPromptReuseError(f"{name} is missing.") from exc
    if resolved != lexical or not resolved.is_dir():
        raise Lfm2CappedPromptReuseError(f"{name} must be one canonical directory.")
    return resolved


def _output_directory(path: str | Path) -> Path:
    lexical = Path(path).expanduser().absolute()
    if lexical.exists() or lexical.is_symlink():
        if lexical.is_symlink() or not lexical.is_dir():
            raise Lfm2CappedPromptReuseError(
                "output_directory must be one real directory."
            )
        return lexical.resolve(strict=True)
    lexical.mkdir(parents=True)
    return lexical.resolve(strict=True)


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CappedPromptReuseError(f"{name} is not readable JSON.") from exc
    if not isinstance(value, dict):
        raise Lfm2CappedPromptReuseError(f"{name} must be a JSON object.")
    return value


def _read_sealed_json(
    path: Path,
    *,
    seal_field: str,
    expected_sha256: str,
    name: str,
) -> dict[str, Any]:
    value = _read_json(path, name)
    claimed = _sha256(value.get(seal_field), f"{name} {seal_field}")
    body = dict(value)
    body.pop(seal_field)
    if claimed != canonical_sha256(body) or claimed != _sha256(
        expected_sha256, f"expected {name} SHA-256"
    ):
        raise Lfm2CappedPromptReuseError(f"{name} seal does not verify.")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            _plain_json(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
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


def load_reuse_plan(path: Path) -> dict[str, Any]:
    """Load and validate the immutable high-cap evaluation plan."""

    source = _local_file(path, "reuse plan")
    plan = _read_json(source, "reuse plan")
    claimed = _sha256(plan.get("plan_sha256"), "plan_sha256")
    body = dict(plan)
    body.pop("plan_sha256")
    if claimed != canonical_sha256(body):
        raise Lfm2CappedPromptReuseError("Reuse plan self-hash does not verify.")
    if plan.get("schema") != PLAN_SCHEMA:
        raise Lfm2CappedPromptReuseError("Reuse plan schema is unsupported.")
    policy = plan.get("policy")
    shared = plan.get("shared")
    lanes = plan.get("lanes")
    if not isinstance(policy, Mapping) or not isinstance(shared, Mapping):
        raise Lfm2CappedPromptReuseError("Reuse plan policy/shared sections are missing.")
    if not isinstance(lanes, list) or not lanes:
        raise Lfm2CappedPromptReuseError("Reuse plan must contain lanes.")
    initial_cap = _positive_int(policy.get("initial_high_cap"), "initial_high_cap")
    adaptive_cap = _positive_int(
        policy.get("adaptive_maximum_cap"), "adaptive_maximum_cap"
    )
    if initial_cap <= _positive_int(
        policy.get("source_cap"), "source_cap"
    ) or adaptive_cap <= initial_cap or adaptive_cap > 2048:
        raise Lfm2CappedPromptReuseError(
            "Caps must increase from the source and stay within runtime limits."
        )
    required_true = (
        "safety_is_separate_from_folding_quality",
        "cap_hit_is_censored_not_failed",
        "rerun_only_source_cap_hits",
        "reuse_saved_child_and_routes_only",
        "parent_trace_regeneration_forbidden",
    )
    if any(policy.get(name) is not True for name in required_true):
        raise Lfm2CappedPromptReuseError("Reuse plan evaluation policy is incomplete.")
    operation_contract = plan.get("operation_contract")
    if not isinstance(operation_contract, Mapping):
        raise Lfm2CappedPromptReuseError("Reuse plan operation contract is missing.")
    expected_operation_contract = {
        "parent_model_load_calls": 0,
        "parent_forward_calls": 0,
        "parent_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "fold_compilations": 0,
        "support_discovery_calls": 0,
        "coefficient_solve_calls": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "pruned_parameters": 0,
        "quantized_parameters": 0,
        "network_access": False,
    }
    if dict(operation_contract) != expected_operation_contract:
        raise Lfm2CappedPromptReuseError(
            "Reuse plan permits work outside child-only evaluation."
        )
    lane_ids: set[str] = set()
    for lane in lanes:
        if not isinstance(lane, Mapping):
            raise Lfm2CappedPromptReuseError("Every lane must be an object.")
        lane_id = str(lane.get("id", "")).strip()
        prompt_ids = lane.get("capped_prompt_ids")
        if (
            not lane_id
            or lane_id in lane_ids
            or not isinstance(prompt_ids, list)
            or not prompt_ids
            or len(set(prompt_ids)) != len(prompt_ids)
            or any(not isinstance(value, str) or not value for value in prompt_ids)
        ):
            raise Lfm2CappedPromptReuseError("Lane IDs/prompts are invalid.")
        lane_ids.add(lane_id)
    return plan


def _source_capped_rows(
    dossier: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    comparisons = dossier.get("comparisons")
    if not isinstance(comparisons, list):
        raise Lfm2CappedPromptReuseError("Source dossier has no comparisons.")
    result = []
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            raise Lfm2CappedPromptReuseError("Source comparison is invalid.")
        response = comparison.get("child_semantic_step_response")
        diagnostics = response.get("diagnostics") if isinstance(response, Mapping) else None
        if not isinstance(diagnostics, Mapping):
            raise Lfm2CappedPromptReuseError(
                "Source child response lacks termination diagnostics."
            )
        cap = diagnostics.get("generation_limit_reached")
        eos = diagnostics.get("ended_with_eos")
        if not isinstance(cap, bool) or not isinstance(eos, bool) or (cap and eos):
            raise Lfm2CappedPromptReuseError(
                "Source termination diagnostics are contradictory."
            )
        if cap:
            result.append(dict(comparison))
    return tuple(result)


def authenticate_lane_source(
    lane: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Authenticate that a lane selects exactly its prior cap hits."""

    source = _local_file(lane.get("source_dossier"), "source dossier")
    if _file_sha256(source) != _sha256(
        lane.get("source_dossier_file_sha256"), "source dossier file SHA-256"
    ):
        raise Lfm2CappedPromptReuseError("Source dossier bytes changed.")
    dossier = _read_sealed_json(
        source,
        seal_field="dossier_sha256",
        expected_sha256=lane.get("source_dossier_sha256"),
        name="source dossier",
    )
    source_cap = _positive_int(dossier.get("maximum_new_tokens"), "source cap")
    if source_cap != _positive_int(lane.get("source_cap"), "lane source cap"):
        raise Lfm2CappedPromptReuseError("Source dossier cap differs from lane plan.")
    rows = _source_capped_rows(dossier)
    observed_ids = [str(row.get("id")) for row in rows]
    if observed_ids != list(lane.get("capped_prompt_ids", [])):
        raise Lfm2CappedPromptReuseError(
            "Lane prompt inventory is not exactly the source generation-limit hits."
        )
    return dossier, rows


def _attempt_path(root: Path, prompt_id: str, cap: int) -> Path:
    safe = hashlib.sha256(prompt_id.encode("utf-8")).hexdigest()[:16]
    return root / "attempts" / f"{safe}-cap{cap}.json"


def _load_or_run_attempt(
    *,
    output_root: Path,
    lane_id: str,
    prompt: Mapping[str, Any],
    cap: int,
    child: Any,
    routing: Any,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    source_dossier_sha256: str,
    child_state_sha256: str,
    routing_contract_sha256: str,
    maximum_prompt_tokens: int,
    maximum_embedding_rows_per_chunk: int,
    maximum_donor_rows_per_score_chunk: int,
) -> dict[str, Any]:
    prompt_id = str(prompt["id"])
    path = _attempt_path(output_root, prompt_id, cap)
    if path.exists():
        cached = _read_json(path, "cached capped-prompt attempt")
        claimed = _sha256(cached.get("attempt_sha256"), "attempt_sha256")
        body = dict(cached)
        body.pop("attempt_sha256")
        expected = {
            "schema": ATTEMPT_SCHEMA,
            "lane_id": lane_id,
            "prompt_id": prompt_id,
            "maximum_new_tokens": cap,
            "source_dossier_sha256": source_dossier_sha256,
            "child_state_sha256": child_state_sha256,
            "routing_contract_sha256": routing_contract_sha256,
        }
        if claimed != canonical_sha256(body) or any(
            cached.get(name) != value for name, value in expected.items()
        ):
            raise Lfm2CappedPromptReuseError(
                "Cached capped-prompt attempt does not match this runtime."
            )
        return cached
    try:
        runtime = run_lfm2_semantic_step_model_review(
            child,
            routing,
            donor_tokenizer,
            target_tokenizer,
            [prompt],
            maximum_prompt_tokens=maximum_prompt_tokens,
            maximum_new_tokens=cap,
            maximum_embedding_rows_per_chunk=maximum_embedding_rows_per_chunk,
            maximum_donor_rows_per_score_chunk=maximum_donor_rows_per_score_chunk,
            allow_legacy_per_operation_authentication=False,
        )
        response = runtime["responses"][0]
        termination = response["termination_evidence"]
        runtime_error = None
    except Exception as exc:  # preserve one prompt's error and continue the lane
        runtime = None
        response = None
        runtime_error = {
            "error_type": type(exc).__qualname__,
            "message": str(exc),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        }
        termination = response_termination_evidence(
            ended_with_eos=False,
            generation_limit_reached=False,
            runtime_error=runtime_error,
        )
    body = {
        "schema": ATTEMPT_SCHEMA,
        "status": "complete-observation" if runtime_error is None else "runtime-error",
        "lane_id": lane_id,
        "prompt_id": prompt_id,
        "maximum_new_tokens": cap,
        "source_dossier_sha256": source_dossier_sha256,
        "child_state_sha256": child_state_sha256,
        "routing_contract_sha256": routing_contract_sha256,
        "response": response,
        "termination_evidence": termination,
        "runtime_receipt": runtime,
        "runtime_error": runtime_error,
        "operation_audit": {
            "parent_model_load_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "fold_compilations": 0,
            "support_discovery_calls": 0,
            "coefficient_solve_calls": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "pruned_parameters": 0,
            "quantized_parameters": 0,
            "network_access": False,
        },
    }
    attempt = {**body, "attempt_sha256": canonical_sha256(body)}
    _atomic_write_json(path, attempt)
    return attempt


def run_lane(
    plan: Mapping[str, Any],
    lane: Mapping[str, Any],
    *,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Run one lane using only its persisted child and route artifacts."""

    shared = plan["shared"]
    policy = plan["policy"]
    lane_id = str(lane["id"])
    source_dossier, capped_rows = authenticate_lane_source(lane)
    prompts_path = _local_file(shared.get("prompts_json"), "review prompt file")
    if _file_sha256(prompts_path) != _sha256(
        shared.get("prompts_file_sha256"), "prompt file SHA-256"
    ):
        raise Lfm2CappedPromptReuseError("Review prompt bytes changed.")
    all_prompts = {str(row["id"]): row for row in load_review_prompts(prompts_path)}
    selected_ids = [str(row["id"]) for row in capped_rows]
    if any(prompt_id not in all_prompts for prompt_id in selected_ids):
        raise Lfm2CappedPromptReuseError("A capped prompt is absent from prompt source.")
    prompts = [all_prompts[prompt_id] for prompt_id in selected_ids]

    checkpoint = _local_directory(shared.get("checkpoint_directory"), "checkpoint")
    checkpoint_inventory_sha256 = _checkpoint_inventory_sha256(checkpoint)
    if checkpoint_inventory_sha256 != _sha256(
        shared.get("checkpoint_artifact_inventory_sha256"),
        "checkpoint artifact inventory SHA-256",
    ):
        raise Lfm2CappedPromptReuseError("Checkpoint artifact inventory changed.")
    donor_directory = _local_directory(
        shared.get("donor_tokenizer_directory"), "donor tokenizer directory"
    )
    target_directory = _local_directory(
        shared.get("target_tokenizer_directory"), "target tokenizer directory"
    )
    bridge_directory = _local_directory(shared.get("bridge_directory"), "bridge")
    donor_tokenizer = load_lfm2_direct_local_tokenizer(donor_directory)
    target_tokenizer = load_lfm2_direct_local_tokenizer(target_directory)
    bridge = load_strict_tokenizer_bridge_artifact(
        bridge_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=shared.get("bridge_artifact_sha256"),
    )
    device = str(device_override or shared.get("runtime_device", "cuda:0"))
    runtime_dtype, dtype_receipt = resolve_runtime_dtype(
        str(shared.get("runtime_dtype", "fp32")), device
    )
    provenance_dtype = resolve_checkpoint_provenance_dtype(
        str(shared.get("checkpoint_provenance_dtype", "bf16"))
    )
    checkpoint_io = Lfm2DirectLocalCheckpointIO(
        tokenizer_bridge=bridge,
        device=torch.device(str(shared.get("checkpoint_provenance_device", "cpu"))),
        dtype=provenance_dtype,
    )
    child, load_receipt = checkpoint_io.load_authenticated_portable(
        checkpoint,
        runtime_device=torch.device(device),
        runtime_dtype=runtime_dtype,
    )
    child.requires_grad_(False)
    child.eval()
    physical_parameters = sum(int(parameter.numel()) for parameter in child.parameters())
    if physical_parameters != _positive_int(
        shared.get("physical_child_parameter_count"), "physical child parameters"
    ):
        raise Lfm2CappedPromptReuseError("Loaded child parameter count changed.")
    child_state_sha256 = fingerprint_lfm2_direct_local_state(child)

    span_path = _local_file(shared.get("span_evidence_file"), "span evidence")
    span_evidence = _read_sealed_json(
        span_path,
        seal_field="evidence_sha256",
        expected_sha256=shared.get("span_evidence_sha256"),
        name="span evidence",
    )
    codebook = load_lfm2_sparse_semantic_codebook(
        _local_directory(lane.get("codebook_directory"), "sparse codebook"),
        expected_artifact_sha256=lane.get("codebook_artifact_sha256"),
    )
    output = lane.get("functional_output")
    output_coefficients = None
    output_authentication = None
    if output is not None:
        if not isinstance(output, Mapping):
            raise Lfm2CappedPromptReuseError("functional_output must be an object.")
        output_coefficients, output_authentication = (
            load_authenticated_functional_output_refit(
                coefficient_tensor_file=_local_file(
                    output.get("coefficient_tensor_file"),
                    "functional output coefficients",
                ),
                sealed_refit_receipt_file=_local_file(
                    output.get("sealed_receipt_file"),
                    "functional output receipt",
                ),
                expected_math_cache_key=_sha256(
                    output.get("math_cache_key"), "functional output cache key"
                ),
                expected_result_sha256=_sha256(
                    output.get("result_sha256"), "functional output result"
                ),
                selected_codebook=codebook,
            )
        )
    live_table = child.get_input_embeddings().weight
    if output_coefficients is None:
        routing = bind_lfm2_sparse_semantic_code_runtime(
            span_evidence,
            live_table,
            semantic_code_offsets=codebook.semantic_code_offsets,
            semantic_code_target_ids=codebook.semantic_code_target_ids,
            semantic_code_coefficients=codebook.semantic_code_coefficients,
            target_to_donor=codebook.target_to_donor,
        )
    else:
        routing = bind_lfm2_separate_sparse_semantic_code_runtime(
            span_evidence,
            live_table,
            input_semantic_code_offsets=codebook.semantic_code_offsets,
            input_semantic_code_target_ids=codebook.semantic_code_target_ids,
            input_semantic_code_coefficients=codebook.semantic_code_coefficients,
            output_semantic_code_offsets=codebook.semantic_code_offsets,
            output_semantic_code_target_ids=codebook.semantic_code_target_ids,
            output_semantic_code_coefficients=output_coefficients,
            target_to_donor=codebook.target_to_donor,
        )
    routing_receipt = _plain_json(routing.as_dict())
    routing_contract_sha256 = canonical_sha256(routing_receipt)
    output_root = _output_directory(lane.get("output_directory"))
    initial_cap = _positive_int(policy.get("initial_high_cap"), "initial cap")
    adaptive_cap = _positive_int(policy.get("adaptive_maximum_cap"), "adaptive cap")
    prompt_results = []
    source_by_id = {str(row["id"]): row for row in capped_rows}
    for prompt in prompts:
        attempts = []
        for cap in (initial_cap, adaptive_cap):
            if attempts and cap == initial_cap:
                continue
            if attempts and attempts[-1]["termination_evidence"].get(
                "censored_by_output_cap"
            ) is not True:
                break
            attempt = _load_or_run_attempt(
                output_root=output_root,
                lane_id=lane_id,
                prompt=prompt,
                cap=cap,
                child=child,
                routing=routing,
                donor_tokenizer=donor_tokenizer,
                target_tokenizer=target_tokenizer,
                source_dossier_sha256=str(source_dossier["dossier_sha256"]),
                child_state_sha256=child_state_sha256,
                routing_contract_sha256=routing_contract_sha256,
                maximum_prompt_tokens=_positive_int(
                    shared.get("maximum_prompt_tokens"), "maximum prompt tokens"
                ),
                maximum_embedding_rows_per_chunk=_positive_int(
                    shared.get("maximum_embedding_rows_per_chunk"),
                    "maximum embedding rows per chunk",
                ),
                maximum_donor_rows_per_score_chunk=_positive_int(
                    shared.get("maximum_donor_rows_per_score_chunk"),
                    "maximum donor rows per score chunk",
                ),
            )
            attempts.append(attempt)
            if cap == adaptive_cap:
                break
        prompt_results.append(
            {
                "id": str(prompt["id"]),
                "category": prompt.get("category"),
                "prompt": _plain_json(prompt),
                "source_256_token_comparison": _plain_json(
                    source_by_id[str(prompt["id"])]
                ),
                "attempts": attempts,
                "authoritative_high_cap_attempt": attempts[-1],
                "manual_review_instruction": (
                    "Review folding quality and safety as separate dimensions. "
                    "A cap hit remains censored; do not convert it to a failure. "
                    "Record looping and runtime errors independently."
                ),
            }
        )
    body = {
        "schema": RESULT_SCHEMA,
        "status": "complete-awaiting-manual-high-cap-review",
        "lane_id": lane_id,
        "source_dossier_sha256": source_dossier["dossier_sha256"],
        "source_cap": int(policy["source_cap"]),
        "initial_high_cap": initial_cap,
        "adaptive_maximum_cap": adaptive_cap,
        "prompt_count": len(prompt_results),
        "prompt_ids": selected_ids,
        "prompt_results": prompt_results,
        "child_state_sha256": child_state_sha256,
        "physical_child_parameter_count": physical_parameters,
        "codebook_artifact_sha256": codebook.artifact_sha256,
        "routing_contract_sha256": routing_contract_sha256,
        "routing_receipt": routing_receipt,
        "functional_output_authentication": output_authentication,
        "checkpoint_portable_load_receipt": _plain_json(load_receipt.as_dict()),
        "checkpoint_artifact_inventory_sha256": checkpoint_inventory_sha256,
        "runtime_dtype_receipt": _plain_json(dtype_receipt),
        "evaluation_policy": _plain_json(policy),
        "operation_audit": {
            "parent_model_load_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "fold_compilations": 0,
            "support_discovery_calls": 0,
            "coefficient_solve_calls": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "pruned_parameters": 0,
            "quantized_parameters": 0,
            "network_access": False,
            "only_source_generation_limit_hits_run": True,
        },
        "automatic_acceptance_or_rejection_performed": False,
        "safety_is_separate_from_folding_quality": True,
        "generation_limit_hit_is_censored_not_failed": True,
    }
    result = {**body, "result_sha256": canonical_sha256(body)}
    _atomic_write_json(output_root / "high-cap-dossier.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--lane",
        action="append",
        help="Lane ID to run. Repeat for multiple lanes; omit for all.",
    )
    parser.add_argument(
        "--device",
        help="Optional runtime device override such as cuda:1.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = load_reuse_plan(args.manifest)
    requested = None if not args.lane else set(args.lane)
    lanes = [
        lane
        for lane in plan["lanes"]
        if requested is None or str(lane["id"]) in requested
    ]
    if requested is not None and {str(lane["id"]) for lane in lanes} != requested:
        raise Lfm2CappedPromptReuseError("One or more requested lanes are unknown.")
    results = [
        run_lane(plan, lane, device_override=args.device) for lane in lanes
    ]
    print(
        json.dumps(
            {
                "status": "complete",
                "lanes": [
                    {
                        "id": row["lane_id"],
                        "prompt_count": row["prompt_count"],
                        "result_sha256": row["result_sha256"],
                    }
                    for row in results
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Lfm2CappedPromptReuseError",
    "authenticate_lane_source",
    "load_reuse_plan",
    "main",
    "run_lane",
]
