"""Cached, fold-only teacher-forced semantic-trajectory factorial audit.

The audit removes autoregressive branching from token-bridge diagnosis.  Every
lane consumes the same authenticated parent donor-token trajectory.  Only the
source of the prompt/continuation embeddings changes:

``exact_exact``
    Exact parent table for prompt and continuation.
``folded_exact``
    Folded ``C_in`` for the prompt, exact continuation.
``exact_folded``
    Exact prompt, folded ``C_in`` for the continuation.
``folded_folded``
    Folded ``C_in`` for prompt and continuation.

Each resulting hidden state is scored twice, once with the exact parent table
and once with the authenticated folded ``C_out`` route.  There is no fitting,
training, backward pass, parent forward, global residual map, pruning, or
quantization in this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

from .dynamic_circuit_ir import canonical_sha256


LFM2_TEACHER_FORCED_TRAJECTORY_BATCH_SCHEMA = (
    "neural-forge-lfm2-teacher-forced-parent-trajectory-batch-v1"
)
LFM2_TEACHER_FORCED_TRAJECTORY_REQUEST_SCHEMA = (
    "neural-forge-lfm2-teacher-forced-trajectory-audit-request-v1"
)
LFM2_TEACHER_FORCED_TRAJECTORY_PROMPT_RESULT_SCHEMA = (
    "neural-forge-lfm2-teacher-forced-trajectory-prompt-result-v1"
)
LFM2_TEACHER_FORCED_TRAJECTORY_AUDIT_SCHEMA = (
    "neural-forge-lfm2-teacher-forced-trajectory-factorial-audit-v1"
)

LANE_CONTRACT: Mapping[str, Mapping[str, str]] = {
    "exact_exact": {"prompt_embeddings": "exact", "continuation_embeddings": "exact"},
    "folded_exact": {"prompt_embeddings": "folded-c-in", "continuation_embeddings": "exact"},
    "exact_folded": {"prompt_embeddings": "exact", "continuation_embeddings": "folded-c-in"},
    "folded_folded": {
        "prompt_embeddings": "folded-c-in",
        "continuation_embeddings": "folded-c-in",
    },
}
READOUT_CONTRACT: Mapping[str, str] = {
    "exact_parent_table": "exact-defined-parent-tied-table",
    "folded_c_out": "authenticated-separate-folded-output-route",
}
_OPERATION_AUDIT: Mapping[str, Any] = {
    "parent_forward_calls": 0,
    "parent_trace_recapture_calls": 0,
    "training_steps": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "global_residual_maps_fit": 0,
    "pruned_parameters": 0,
    "quantized_parameters": 0,
    "teacher_forced_donor_trajectory": True,
    "argmax_branching_used_as_body_input": False,
    "target_token_spans_used_as_body_steps": False,
}


class Lfm2TeacherForcedTrajectoryAuditError(RuntimeError):
    """Raised when trajectory evidence or execution fails closed."""


ProgressSink = Callable[[Mapping[str, Any]], None]


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
        raise Lfm2TeacherForcedTrajectoryAuditError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2TeacherForcedTrajectoryAuditError(f"{name} must be positive.")
    return int(value)


def _dtype_name(value: torch.dtype) -> str:
    return str(value).removeprefix("torch.")


def resolve_floating_dtype(
    value: str | torch.dtype,
    *,
    role: str,
    allow_low_precision: bool,
) -> torch.dtype:
    """Resolve an explicit floating dtype without silently inheriting defaults."""

    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
    }
    dtype = value if isinstance(value, torch.dtype) else aliases.get(str(value).lower())
    allowed = {torch.float32, torch.float64}
    if allow_low_precision:
        allowed.update({torch.float16, torch.bfloat16})
    if dtype not in allowed:
        precision = "FP16/BF16/FP32/FP64" if allow_low_precision else "FP32/FP64"
        raise Lfm2TeacherForcedTrajectoryAuditError(
            f"{role} dtype must be one of {precision}."
        )
    assert isinstance(dtype, torch.dtype)
    return dtype


def resolve_runtime_dtype_policy(
    requested_dtype: str | torch.dtype,
    device: str | torch.device,
    *,
    cuda_capability: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Resolve body dtype and make hardware-driven widening explicit.

    BF16 body execution is widened to FP32 on pre-Ampere CUDA devices.  This is
    a runtime placement decision, not a change to folded coefficients or cache
    math, and prevents repeated P40 BF16 failures.
    """

    requested = resolve_floating_dtype(
        requested_dtype, role="runtime", allow_low_precision=True
    )
    placement = torch.device(device)
    effective = requested
    fallback = None
    capability = cuda_capability
    if placement.type == "cpu" and requested in {torch.float16, torch.bfloat16}:
        effective = torch.float32
        fallback = "cpu-low-precision-body-kernel-support-not-assumed;used-float32"
    elif placement.type == "cuda" and requested == torch.bfloat16:
        if capability is None:
            try:
                capability = torch.cuda.get_device_capability(placement)
            except (AssertionError, RuntimeError):
                capability = None
        if capability is not None and capability[0] < 8:
            effective = torch.float32
            fallback = "cuda-device-lacks-native-bfloat16-body-execution;used-float32"
    body = {
        "schema": "neural-forge-explicit-runtime-dtype-policy-v1",
        "device": str(placement),
        "requested_dtype": _dtype_name(requested),
        "effective_dtype": _dtype_name(effective),
        "cuda_capability": None if capability is None else list(capability),
        "typed_fallback": fallback,
        "silent_dtype_inference": False,
    }
    return {**body, "policy_sha256": canonical_sha256(body)}


def tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256(b"neural-forge-teacher-forced-trajectory-tensor-v1\0")
    digest.update(_dtype_name(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape], separators=(",", ":")
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validated_integer_vector(value: Tensor, name: str) -> Tensor:
    result = value.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if result.ndim != 1 or result.numel() == 0 or bool((result < 0).any()):
        raise Lfm2TeacherForcedTrajectoryAuditError(
            f"{name} must be one nonempty nonnegative integer vector."
        )
    return result


def _validated_float_matrix(value: Tensor, name: str) -> Tensor:
    result = value.detach().to(device="cpu").contiguous()
    if (
        result.ndim != 2
        or not torch.is_floating_point(result)
        or result.is_complex()
        or result.requires_grad
        or not bool(torch.isfinite(result).all())
        or min(int(size) for size in result.shape) <= 0
    ):
        raise Lfm2TeacherForcedTrajectoryAuditError(
            f"{name} must be one finite detached floating matrix."
        )
    return result


@dataclass(frozen=True, slots=True)
class AuthenticatedParentTrajectoryBatch:
    """Minimal selected trace payload needed by the factorial audit."""

    prompt_token_ids: Tensor
    forced_token_ids: Tensor
    parent_hidden_states: Tensor
    parent_logits: Tensor
    receipt: Mapping[str, Any]
    position_ids: Tensor | None = None

    def __post_init__(self) -> None:
        prompt = _validated_integer_vector(self.prompt_token_ids, "prompt_token_ids")
        forced = _validated_integer_vector(self.forced_token_ids, "forced_token_ids")
        hidden = _validated_float_matrix(
            self.parent_hidden_states, "parent_hidden_states"
        )
        logits = _validated_float_matrix(self.parent_logits, "parent_logits")
        positions = None
        if self.position_ids is not None:
            positions = _validated_integer_vector(self.position_ids, "position_ids")
            if int(positions.numel()) != int(prompt.numel()):
                raise Lfm2TeacherForcedTrajectoryAuditError(
                    "Position IDs must align with the prompt."
                )
            expected = positions[:1] + torch.arange(
                int(positions.numel()), dtype=torch.int64
            )
            if not torch.equal(positions, expected):
                raise Lfm2TeacherForcedTrajectoryAuditError(
                    "Position IDs must be contiguous."
                )
        state_count = int(forced.numel()) + 1
        if (
            int(hidden.shape[0]) != state_count
            or int(logits.shape[0]) != state_count
        ):
            raise Lfm2TeacherForcedTrajectoryAuditError(
                "Parent hidden/logit rows must cover prefill plus every forced prefix."
            )
        value = _plain_json(self.receipt)
        supplied = value.pop("receipt_sha256", None)
        if (
            supplied != canonical_sha256(value)
            or value.get("schema") != LFM2_TEACHER_FORCED_TRAJECTORY_BATCH_SCHEMA
            or value.get("prompt_token_count") != int(prompt.numel())
            or value.get("forced_token_count") != int(forced.numel())
            or value.get("state_count") != state_count
            or value.get("hidden_size") != int(hidden.shape[1])
            or value.get("donor_row_count") != int(logits.shape[1])
            or value.get("prompt_token_ids_sha256") != tensor_sha256(prompt)
            or value.get("forced_token_ids_sha256") != tensor_sha256(forced)
            or value.get("parent_hidden_states_sha256") != tensor_sha256(hidden)
            or value.get("parent_logits_sha256") != tensor_sha256(logits)
            or value.get("position_ids_sha256")
            != (None if positions is None else tensor_sha256(positions))
            or value.get("parent_forward_calls") != 0
            or value.get("parent_trace_recapture_calls") != 0
        ):
            raise Lfm2TeacherForcedTrajectoryAuditError(
                "Parent trajectory batch receipt binding is invalid."
            )
        object.__setattr__(self, "prompt_token_ids", prompt.clone())
        object.__setattr__(self, "forced_token_ids", forced.clone())
        object.__setattr__(self, "parent_hidden_states", hidden.clone())
        object.__setattr__(self, "parent_logits", logits.clone())
        object.__setattr__(self, "position_ids", None if positions is None else positions.clone())
        object.__setattr__(self, "receipt", {**value, "receipt_sha256": supplied})

    @property
    def batch_index(self) -> int:
        return int(self.receipt["batch_index"])

    @property
    def prompt_id(self) -> str:
        return str(self.receipt["prompt_id"])


def seal_parent_trajectory_batch(
    *,
    batch_index: int,
    prompt_id: str,
    trace_cache_key: str,
    trace_parent_fingerprint: str,
    trace_selective_session_sha256: str,
    trace_input_sha256: str,
    trace_record_sha256: str,
    selected_payload_sha256: str,
    prompt_token_ids: Tensor,
    forced_token_ids: Tensor,
    parent_hidden_states: Tensor,
    parent_logits: Tensor,
    position_ids: Tensor | None = None,
) -> AuthenticatedParentTrajectoryBatch:
    prompt = _validated_integer_vector(prompt_token_ids, "prompt_token_ids")
    forced = _validated_integer_vector(forced_token_ids, "forced_token_ids")
    hidden = _validated_float_matrix(parent_hidden_states, "parent_hidden_states")
    logits = _validated_float_matrix(parent_logits, "parent_logits")
    positions = (
        None
        if position_ids is None
        else _validated_integer_vector(position_ids, "position_ids")
    )
    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 0:
        raise Lfm2TeacherForcedTrajectoryAuditError("batch_index must be nonnegative.")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise Lfm2TeacherForcedTrajectoryAuditError("prompt_id must be nonempty.")
    body = {
        "schema": LFM2_TEACHER_FORCED_TRAJECTORY_BATCH_SCHEMA,
        "batch_index": batch_index,
        "prompt_id": prompt_id,
        "trace_cache_key": _digest(trace_cache_key, "trace_cache_key"),
        "trace_parent_fingerprint": _digest(
            trace_parent_fingerprint, "trace_parent_fingerprint"
        ),
        "trace_selective_session_sha256": _digest(
            trace_selective_session_sha256, "trace_selective_session_sha256"
        ),
        "trace_input_sha256": _digest(trace_input_sha256, "trace_input_sha256"),
        "trace_record_sha256": _digest(trace_record_sha256, "trace_record_sha256"),
        "selected_payload_sha256": _digest(
            selected_payload_sha256, "selected_payload_sha256"
        ),
        "prompt_token_count": int(prompt.numel()),
        "forced_token_count": int(forced.numel()),
        "state_count": int(forced.numel()) + 1,
        "hidden_size": int(hidden.shape[1]),
        "donor_row_count": int(logits.shape[1]),
        "prompt_token_ids_sha256": tensor_sha256(prompt),
        "forced_token_ids_sha256": tensor_sha256(forced),
        "parent_hidden_states_sha256": tensor_sha256(hidden),
        "parent_logits_sha256": tensor_sha256(logits),
        "position_ids_sha256": None if positions is None else tensor_sha256(positions),
        "parent_forward_calls": 0,
        "parent_trace_recapture_calls": 0,
    }
    return AuthenticatedParentTrajectoryBatch(
        prompt,
        forced,
        hidden,
        logits,
        {**body, "receipt_sha256": canonical_sha256(body)},
        positions,
    )


def build_teacher_forced_trajectory_audit_request(
    *,
    artifact_bindings: Mapping[str, Any],
    runtime_dtype_policy: Mapping[str, Any],
    metric_dtype: str | torch.dtype,
    score_dtype: str | torch.dtype,
    donor_row_count: int,
    target_row_count: int,
    hidden_size: int,
    execution_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a content identity that excludes chunk/log orchestration choices."""

    metric = resolve_floating_dtype(metric_dtype, role="metric", allow_low_precision=False)
    score = resolve_floating_dtype(score_dtype, role="score", allow_low_precision=False)
    bindings = _plain_json(artifact_bindings)
    if not bindings or any(value is None for value in bindings.values()):
        raise Lfm2TeacherForcedTrajectoryAuditError(
            "artifact_bindings must be nonempty and fully explicit."
        )
    policy = _plain_json(runtime_dtype_policy)
    if policy.get("policy_sha256") != canonical_sha256(
        {key: value for key, value in policy.items() if key != "policy_sha256"}
    ):
        raise Lfm2TeacherForcedTrajectoryAuditError(
            "Runtime dtype policy is not sealed."
        )
    math_body = {
        "schema": LFM2_TEACHER_FORCED_TRAJECTORY_REQUEST_SCHEMA,
        "artifact_bindings": bindings,
        "geometry": {
            "donor_row_count": _positive_int(donor_row_count, "donor_row_count"),
            "target_row_count": _positive_int(target_row_count, "target_row_count"),
            "hidden_size": _positive_int(hidden_size, "hidden_size"),
        },
        "lane_contract": _plain_json(LANE_CONTRACT),
        "readout_contract": _plain_json(READOUT_CONTRACT),
        "runtime_dtype_policy": policy,
        "metric_dtype": _dtype_name(metric),
        "score_dtype": _dtype_name(score),
        "body_execution": "one-causal-full-donor-trajectory-forward-per-lane",
        "teacher_forced_parent_tokens": True,
        "target_token_spans_used_as_body_steps": False,
        "operation_audit": dict(_OPERATION_AUDIT),
    }
    execution = _plain_json({} if execution_contract is None else execution_contract)
    return {
        **math_body,
        "cache_key": canonical_sha256(math_body),
        "execution_contract": execution,
        "execution_contract_changes_cache_identity": False,
    }


def _decode_exact(tokenizer: Any, token_ids: Sequence[int], name: str) -> str:
    decoder = getattr(tokenizer, "decode", None)
    if not callable(decoder):
        raise Lfm2TeacherForcedTrajectoryAuditError(
            f"{name} tokenizer must expose decode()."
        )
    try:
        result = decoder(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        result = decoder(list(token_ids))
    if not isinstance(result, str):
        raise Lfm2TeacherForcedTrajectoryAuditError(
            f"{name} tokenizer decode() did not return text."
        )
    return result


def _encode_exact(tokenizer: Any, text: str, name: str) -> tuple[int, ...]:
    encoder = getattr(tokenizer, "encode", None)
    if not callable(encoder):
        raise Lfm2TeacherForcedTrajectoryAuditError(
            f"{name} tokenizer must expose encode()."
        )
    value = encoder(text, add_special_tokens=False)
    if isinstance(value, Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Lfm2TeacherForcedTrajectoryAuditError(
            f"{name} tokenizer returned invalid token IDs."
        )
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise Lfm2TeacherForcedTrajectoryAuditError(
                f"{name} tokenizer returned invalid token IDs."
            )
        result.append(int(item))
    if not result:
        raise Lfm2TeacherForcedTrajectoryAuditError(
            f"{name} tokenizer returned an empty semantic span."
        )
    return tuple(result)


def assert_text_span_roundtrip_only(
    *,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    prompt_token_ids: Sequence[int] | Tensor,
    forced_token_ids: Sequence[int] | Tensor,
) -> dict[str, Any]:
    """Authenticate exact full-prefix text roundtrips without returning body IDs."""

    prompt = (
        prompt_token_ids.detach().cpu().reshape(-1).tolist()
        if isinstance(prompt_token_ids, Tensor)
        else list(prompt_token_ids)
    )
    forced = (
        forced_token_ids.detach().cpu().reshape(-1).tolist()
        if isinstance(forced_token_ids, Tensor)
        else list(forced_token_ids)
    )
    rows = []
    for prefix_index in range(len(forced) + 1):
        donor_ids = [int(value) for value in (*prompt, *forced[:prefix_index])]
        semantic_text = _decode_exact(donor_tokenizer, donor_ids, "donor")
        target_span = _encode_exact(target_tokenizer, semantic_text, "target")
        roundtrip = _decode_exact(target_tokenizer, target_span, "target")
        if roundtrip != semantic_text or _encode_exact(
            target_tokenizer, roundtrip, "target"
        ) != target_span:
            raise Lfm2TeacherForcedTrajectoryAuditError(
                f"Target text-span roundtrip failed at prefix {prefix_index}."
            )
        row = {
            "prefix_index": prefix_index,
            "donor_prefix_token_count": len(donor_ids),
            "target_span_token_count": len(target_span),
            "semantic_text_sha256": hashlib.sha256(
                semantic_text.encode("utf-8")
            ).hexdigest(),
            "target_span_ids_sha256": canonical_sha256(list(target_span)),
        }
        rows.append({**row, "row_sha256": canonical_sha256(row)})
    body = {
        "schema": "neural-forge-lfm2-trajectory-text-span-roundtrip-v1",
        "prefix_count": len(rows),
        "rows_sha256": canonical_sha256(rows),
        "exact_text_roundtrip": True,
        "target_span_ids_returned_to_body": False,
        "body_step_domain": "authenticated-donor-token-trajectory-only",
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _extract_body_hidden(output: Any, *, sequence_length: int, hidden_size: int) -> Tensor:
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, (tuple, list)) and output:
        hidden = output[0]
    if (
        not isinstance(hidden, Tensor)
        or hidden.ndim != 3
        or tuple(int(size) for size in hidden.shape)
        != (1, sequence_length, hidden_size)
        or not torch.is_floating_point(hidden)
        or hidden.is_complex()
        or not bool(torch.isfinite(hidden.detach()).all())
    ):
        raise Lfm2TeacherForcedTrajectoryAuditError(
            "Child body returned invalid full-trajectory hidden states."
        )
    return hidden.detach()


def _relative_hidden_metrics(
    candidate: Tensor, reference: Tensor, *, metric_dtype: torch.dtype
) -> dict[str, float]:
    left = candidate.detach().to(device="cpu", dtype=metric_dtype)
    right = reference.detach().to(device="cpu", dtype=metric_dtype)
    delta = left - right
    l2 = torch.linalg.vector_norm(delta)
    reference_norm = torch.linalg.vector_norm(right)
    denominator = reference_norm.clamp_min(torch.finfo(metric_dtype).tiny)
    cosine = torch.nn.functional.cosine_similarity(
        left.reshape(1, -1), right.reshape(1, -1), dim=-1
    )[0]
    return {
        "cosine": float(cosine.item()),
        "cosine_distance": float((1.0 - cosine).item()),
        "l2": float(l2.item()),
        "relative_l2": float((l2 / denominator).item()),
        "candidate_norm": float(torch.linalg.vector_norm(left).item()),
        "parent_norm": float(reference_norm.item()),
    }


def parent_winner_rank_margin(scores: Tensor, parent_winner: int) -> dict[str, Any]:
    """Return exact rank and signed margin for one authenticated parent winner."""

    values = scores.detach().reshape(-1)
    if values.dtype not in {torch.float32, torch.float64}:
        values = values.to(dtype=torch.float32)
    if (
        values.numel() <= 1
        or not bool(torch.isfinite(values).all())
        or isinstance(parent_winner, bool)
        or not isinstance(parent_winner, int)
        or parent_winner < 0
        or parent_winner >= int(values.numel())
    ):
        raise Lfm2TeacherForcedTrajectoryAuditError(
            "Readout scores or parent winner are invalid."
        )
    winner_score = values[parent_winner]
    top_values, top_ids = torch.topk(values, k=2, largest=True, sorted=True)
    top_id = int(top_ids[0].item())
    competing_index = 1 if top_id == parent_winner else 0
    best_competing_score = top_values[competing_index]
    best_competing_id = top_ids[competing_index]
    return {
        "top1_token_id": top_id,
        "parent_winner_token_id": parent_winner,
        "parent_winner_rank": int((values > winner_score).sum().item()) + 1,
        "parent_winner_score": float(winner_score.item()),
        "best_competing_token_id": int(best_competing_id.item()),
        "best_competing_score": float(best_competing_score.item()),
        "parent_winner_margin": float((winner_score - best_competing_score).item()),
        "top1_matches_parent": top_id == parent_winner,
    }


def _slope(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    y = torch.tensor(tuple(values), dtype=torch.float64)
    x = torch.arange(len(values), dtype=torch.float64)
    centered_x = x - x.mean()
    denominator = torch.dot(centered_x, centered_x)
    return float((torch.dot(centered_x, y - y.mean()) / denominator).item())


def _first_divergence(records: Sequence[Mapping[str, Any]], readout: str) -> int | None:
    for record in records:
        if not bool(record["readouts"][readout]["top1_matches_parent"]):
            return int(record["prefix_index"])
    return None


def _emit(progress: ProgressSink | None, **value: Any) -> None:
    if progress is not None:
        progress(_plain_json(value))


def run_teacher_forced_trajectory_batch(
    *,
    child_body: Any,
    routing: Any,
    routing_session: Any,
    live_target_table: Tensor,
    exact_parent_table: Tensor,
    batch: AuthenticatedParentTrajectoryBatch,
    metric_dtype: str | torch.dtype = torch.float32,
    score_dtype: str | torch.dtype = torch.float32,
    maximum_embedding_rows_per_chunk: int = 1024,
    maximum_donor_rows_per_score_chunk: int = 4096,
    text_span_roundtrip_receipt: Mapping[str, Any] | None = None,
    progress: ProgressSink | None = None,
) -> dict[str, Any]:
    """Run all four teacher-forced lanes for one authenticated cache batch."""

    if not callable(child_body):
        raise Lfm2TeacherForcedTrajectoryAuditError("child_body must be callable.")
    metric = resolve_floating_dtype(metric_dtype, role="metric", allow_low_precision=False)
    score = resolve_floating_dtype(score_dtype, role="score", allow_low_precision=False)
    embedding_chunk = _positive_int(
        maximum_embedding_rows_per_chunk, "maximum_embedding_rows_per_chunk"
    )
    score_chunk = _positive_int(
        maximum_donor_rows_per_score_chunk, "maximum_donor_rows_per_score_chunk"
    )
    exact_table = _validated_float_matrix(exact_parent_table, "exact_parent_table")
    live_table = live_target_table.detach()
    hidden_size = int(exact_table.shape[1])
    donor_count = int(exact_table.shape[0])
    if (
        live_table.ndim != 2
        or int(live_table.shape[1]) != hidden_size
        or int(batch.parent_hidden_states.shape[1]) != hidden_size
        or int(batch.parent_logits.shape[1]) != donor_count
    ):
        raise Lfm2TeacherForcedTrajectoryAuditError(
            "Trajectory, parent table, and live target table geometry differ."
        )
    if max(
        int(batch.prompt_token_ids.max().item()),
        int(batch.forced_token_ids.max().item()),
    ) >= donor_count:
        raise Lfm2TeacherForcedTrajectoryAuditError(
            "Teacher-forced trajectory leaves the defined donor vocabulary."
        )
    if not bool(getattr(routing, "has_separate_output_semantic_code", False)):
        raise Lfm2TeacherForcedTrajectoryAuditError(
            "Factorial audit requires an authenticated separate C_in/C_out route."
        )
    routing_receipt = routing.as_dict()
    if (
        routing_receipt.get("input_output_coefficient_storage_shared") is not False
        or routing_receipt.get("donor_row_count") != donor_count
        or routing_receipt.get("target_row_count") != int(live_table.shape[0])
        or routing_receipt.get("hidden_size") != hidden_size
    ):
        raise Lfm2TeacherForcedTrajectoryAuditError(
            "Separate semantic route receipt geometry is invalid."
        )
    if text_span_roundtrip_receipt is not None:
        span_receipt = _plain_json(text_span_roundtrip_receipt)
        supplied = span_receipt.pop("receipt_sha256", None)
        if (
            supplied != canonical_sha256(span_receipt)
            or span_receipt.get("exact_text_roundtrip") is not True
            or span_receipt.get("target_span_ids_returned_to_body") is not False
            or span_receipt.get("prefix_count") != int(batch.forced_token_ids.numel()) + 1
        ):
            raise Lfm2TeacherForcedTrajectoryAuditError(
                "Text-span roundtrip receipt is invalid."
            )
        span_receipt["receipt_sha256"] = supplied
    else:
        span_receipt = None

    device = live_table.device
    runtime_dtype = live_table.dtype
    prompt_ids = batch.prompt_token_ids.to(device=device)
    forced_ids = batch.forced_token_ids.to(device=device)
    exact_runtime_table = exact_table.to(device=device, dtype=runtime_dtype)
    exact_prompt = exact_runtime_table.index_select(0, prompt_ids)
    exact_continuation = exact_runtime_table.index_select(0, forced_ids)
    folded_prompt = routing.reconstruct_selected_donor_embeddings(
        routing_session,
        batch.prompt_token_ids,
        maximum_rows_per_chunk=embedding_chunk,
    )
    folded_continuation = routing.reconstruct_selected_donor_embeddings(
        routing_session,
        batch.forced_token_ids,
        maximum_rows_per_chunk=embedding_chunk,
    )
    for name, value, expected_rows in (
        ("exact_prompt", exact_prompt, int(prompt_ids.numel())),
        ("folded_prompt", folded_prompt, int(prompt_ids.numel())),
        ("exact_continuation", exact_continuation, int(forced_ids.numel())),
        ("folded_continuation", folded_continuation, int(forced_ids.numel())),
    ):
        if (
            tuple(int(size) for size in value.shape) != (expected_rows, hidden_size)
            or value.device != device
            or value.dtype != runtime_dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise Lfm2TeacherForcedTrajectoryAuditError(
                f"{name} has the wrong runtime dtype, placement, or geometry."
            )

    prompt_count = int(prompt_ids.numel())
    forced_count = int(forced_ids.numel())
    sequence_count = prompt_count + forced_count
    state_positions = torch.arange(
        prompt_count - 1,
        sequence_count,
        device=device,
        dtype=torch.int64,
    )
    attention_mask = torch.ones((1, sequence_count), device=device, dtype=torch.int64)
    position_ids = None
    if batch.position_ids is not None:
        prompt_positions = batch.position_ids.to(device=device, dtype=torch.int64)
        continuation_positions = prompt_positions[-1] + torch.arange(
            1, forced_count + 1, device=device, dtype=torch.int64
        )
        position_ids = torch.cat((prompt_positions, continuation_positions)).unsqueeze(0)

    cached_parent_winners = batch.parent_logits.argmax(dim=-1).to(torch.int64)
    forced_match_count = int(
        cached_parent_winners[:-1].eq(batch.forced_token_ids).sum().item()
    )
    parent_reference_rows = []
    exact_table_score = exact_table.to(dtype=score)
    exact_score_table_device = exact_table.to(device=device, dtype=score)
    with torch.inference_mode():
        parent_hidden_score = batch.parent_hidden_states.to(dtype=score)
        parent_exact_scores = parent_hidden_score @ exact_table_score.transpose(0, 1)
        parent_target_scores = routing.score_target_tokens(
            routing_session,
            batch.parent_hidden_states.to(device=device, dtype=runtime_dtype),
        )
        parent_folded_scores = routing.compute_all_donor_logits_from_target_scores(
            routing_session,
            parent_target_scores,
            maximum_donor_rows_per_chunk=score_chunk,
        )
        for index in range(forced_count + 1):
            parent_winner = int(cached_parent_winners[index].item())
            parent_reference_rows.append(
                {
                    "prefix_index": index,
                    "parent_winner_token_id": parent_winner,
                    "next_forced_token_id": (
                        None if index == forced_count else int(batch.forced_token_ids[index])
                    ),
                    "next_forced_token_matches_parent_winner": (
                        None
                        if index == forced_count
                        else int(batch.forced_token_ids[index]) == parent_winner
                    ),
                    "cached_parent_logit_margin": parent_winner_rank_margin(
                        batch.parent_logits[index], parent_winner
                    )["parent_winner_margin"],
                    "parent_hidden_exact_parent_table": parent_winner_rank_margin(
                        parent_exact_scores[index], parent_winner
                    ),
                    "parent_hidden_folded_c_out": parent_winner_rank_margin(
                        parent_folded_scores[index], parent_winner
                    ),
                }
            )

    lane_embeddings = {
        "exact_exact": torch.cat((exact_prompt, exact_continuation), dim=0),
        "folded_exact": torch.cat((folded_prompt, exact_continuation), dim=0),
        "exact_folded": torch.cat((exact_prompt, folded_continuation), dim=0),
        "folded_folded": torch.cat((folded_prompt, folded_continuation), dim=0),
    }
    lane_results: dict[str, Any] = {}
    _emit(
        progress,
        event="prompt-start",
        stage="trajectory",
        batch_index=batch.batch_index,
        prompt_id=batch.prompt_id,
        forced_token_count=forced_count,
        body_sequence_token_count=sequence_count,
    )
    with torch.inference_mode():
        for lane_name in LANE_CONTRACT:
            embeddings = lane_embeddings[lane_name]
            _emit(
                progress,
                event="lane-forward-start",
                stage="child-body",
                batch_index=batch.batch_index,
                prompt_id=batch.prompt_id,
                lane=lane_name,
                body_sequence_token_count=sequence_count,
                body_step_domain="donor-token-trajectory",
                runtime_device=str(device),
                runtime_dtype=_dtype_name(runtime_dtype),
            )
            kwargs: dict[str, Any] = {
                "inputs_embeds": embeddings.unsqueeze(0),
                "attention_mask": attention_mask,
                "use_cache": False,
            }
            if position_ids is not None:
                kwargs["position_ids"] = position_ids
            output = child_body(**kwargs)
            full_hidden = _extract_body_hidden(
                output, sequence_length=sequence_count, hidden_size=hidden_size
            )
            states = full_hidden[0].index_select(0, state_positions)
            lane_exact_scores = states.to(dtype=score) @ exact_score_table_device.transpose(
                0, 1
            )
            lane_target_scores = routing.score_target_tokens(routing_session, states)
            lane_folded_scores = routing.compute_all_donor_logits_from_target_scores(
                routing_session,
                lane_target_scores,
                maximum_donor_rows_per_chunk=score_chunk,
            )
            records = []
            for prefix_index in range(forced_count + 1):
                candidate = states[prefix_index]
                parent = batch.parent_hidden_states[prefix_index]
                hidden_metrics = _relative_hidden_metrics(
                    candidate, parent, metric_dtype=metric
                )
                parent_winner = int(cached_parent_winners[prefix_index].item())
                row = {
                    "prefix_index": prefix_index,
                    "consumed_forced_token_count": prefix_index,
                    "next_forced_token_id": (
                        None
                        if prefix_index == forced_count
                        else int(batch.forced_token_ids[prefix_index])
                    ),
                    "parent_winner_token_id": parent_winner,
                    "hidden_vs_cached_parent": hidden_metrics,
                    "readouts": {
                        "exact_parent_table": parent_winner_rank_margin(
                            lane_exact_scores[prefix_index], parent_winner
                        ),
                        "folded_c_out": parent_winner_rank_margin(
                            lane_folded_scores[prefix_index], parent_winner
                        ),
                    },
                }
                records.append(row)
                _emit(
                    progress,
                    event="trajectory-token-complete",
                    stage="diagnostic",
                    batch_index=batch.batch_index,
                    prompt_id=batch.prompt_id,
                    lane=lane_name,
                    **row,
                )
            lane_summary = {
                "state_count": len(records),
                "mean_hidden_cosine": sum(
                    row["hidden_vs_cached_parent"]["cosine"] for row in records
                )
                / len(records),
                "mean_hidden_relative_l2": sum(
                    row["hidden_vs_cached_parent"]["relative_l2"] for row in records
                )
                / len(records),
                "hidden_cosine_distance_drift_slope": _slope(
                    [
                        row["hidden_vs_cached_parent"]["cosine_distance"]
                        for row in records
                    ]
                ),
                "hidden_relative_l2_drift_slope": _slope(
                    [row["hidden_vs_cached_parent"]["relative_l2"] for row in records]
                ),
                "first_exact_parent_table_divergence": _first_divergence(
                    records, "exact_parent_table"
                ),
                "first_folded_c_out_divergence": _first_divergence(
                    records, "folded_c_out"
                ),
                "exact_parent_table_match_count": sum(
                    bool(row["readouts"]["exact_parent_table"]["top1_matches_parent"])
                    for row in records
                ),
                "folded_c_out_match_count": sum(
                    bool(row["readouts"]["folded_c_out"]["top1_matches_parent"])
                    for row in records
                ),
            }
            lane_results[lane_name] = {"summary": lane_summary, "tokens": records}
            _emit(
                progress,
                event="lane-forward-complete",
                stage="child-body",
                batch_index=batch.batch_index,
                prompt_id=batch.prompt_id,
                lane=lane_name,
                summary=lane_summary,
            )

    body = {
        "schema": LFM2_TEACHER_FORCED_TRAJECTORY_PROMPT_RESULT_SCHEMA,
        "batch_index": batch.batch_index,
        "prompt_id": batch.prompt_id,
        "parent_trajectory_receipt_sha256": batch.receipt["receipt_sha256"],
        "routing_runtime_sha256": routing_receipt["runtime_sha256"],
        "routing_source_artifact_sha256": routing_receipt["source_artifact_sha256"],
        "prompt_token_count": prompt_count,
        "forced_token_count": forced_count,
        "state_count": forced_count + 1,
        "parent_forced_token_match_count": forced_match_count,
        "parent_forced_token_match_rate": forced_match_count / forced_count,
        "parent_reference_tokens": parent_reference_rows,
        "lanes": lane_results,
        "lane_contract": _plain_json(LANE_CONTRACT),
        "readout_contract": _plain_json(READOUT_CONTRACT),
        "runtime_dtype": _dtype_name(runtime_dtype),
        "metric_dtype": _dtype_name(metric),
        "score_dtype": _dtype_name(score),
        "body_forward_count": len(LANE_CONTRACT),
        "body_step_domain": "authenticated-donor-token-trajectory-only",
        "text_span_roundtrip_receipt": span_receipt,
        "target_token_spans_used_as_body_steps": False,
        "operation_audit": dict(_OPERATION_AUDIT),
    }
    result = {**body, "result_sha256": canonical_sha256(body)}
    _emit(
        progress,
        event="prompt-complete",
        stage="trajectory",
        batch_index=batch.batch_index,
        prompt_id=batch.prompt_id,
        result_sha256=result["result_sha256"],
    )
    return result


def authenticate_prompt_result(
    value: Mapping[str, Any], *, expected_batch_index: int | None = None
) -> dict[str, Any]:
    result = _plain_json(value)
    supplied = result.pop("result_sha256", None)
    if (
        supplied != canonical_sha256(result)
        or result.get("schema")
        != LFM2_TEACHER_FORCED_TRAJECTORY_PROMPT_RESULT_SCHEMA
        or (expected_batch_index is not None and result.get("batch_index") != expected_batch_index)
        or set(result.get("lanes", {})) != set(LANE_CONTRACT)
        or result.get("target_token_spans_used_as_body_steps") is not False
        or result.get("operation_audit") != _OPERATION_AUDIT
    ):
        raise Lfm2TeacherForcedTrajectoryAuditError(
            "Cached trajectory prompt result authentication failed."
        )
    return {**result, "result_sha256": supplied}


def aggregate_teacher_forced_trajectory_audit(
    *, request: Mapping[str, Any], prompt_results: Sequence[Mapping[str, Any]], trace_session_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    request_value = _plain_json(request)
    cache_key = _digest(request_value.get("cache_key"), "request cache key")
    math_body = {
        key: value
        for key, value in request_value.items()
        if key not in {"cache_key", "execution_contract", "execution_contract_changes_cache_identity"}
    }
    if canonical_sha256(math_body) != cache_key:
        raise Lfm2TeacherForcedTrajectoryAuditError("Audit request seal is invalid.")
    results = [authenticate_prompt_result(value) for value in prompt_results]
    if not results or [row["batch_index"] for row in results] != list(range(len(results))):
        raise Lfm2TeacherForcedTrajectoryAuditError(
            "Prompt results must be complete and in canonical batch order."
        )
    session = _plain_json(trace_session_receipt)
    session_sha = _digest(session.get("session_sha256"), "trace session SHA-256")
    summaries = {}
    for lane in LANE_CONTRACT:
        lane_rows = [row["lanes"][lane]["summary"] for row in results]
        state_count = sum(int(row["state_count"]) for row in lane_rows)
        summaries[lane] = {
            "prompt_count": len(lane_rows),
            "state_count": state_count,
            "mean_prompt_mean_hidden_cosine": sum(
                float(row["mean_hidden_cosine"]) for row in lane_rows
            )
            / len(lane_rows),
            "mean_prompt_mean_hidden_relative_l2": sum(
                float(row["mean_hidden_relative_l2"]) for row in lane_rows
            )
            / len(lane_rows),
            "exact_parent_table_match_count": sum(
                int(row["exact_parent_table_match_count"]) for row in lane_rows
            ),
            "folded_c_out_match_count": sum(
                int(row["folded_c_out_match_count"]) for row in lane_rows
            ),
            "exact_parent_table_match_rate": sum(
                int(row["exact_parent_table_match_count"]) for row in lane_rows
            )
            / state_count,
            "folded_c_out_match_rate": sum(
                int(row["folded_c_out_match_count"]) for row in lane_rows
            )
            / state_count,
        }
    body = {
        "schema": LFM2_TEACHER_FORCED_TRAJECTORY_AUDIT_SCHEMA,
        "status": "complete-cached-fold-only-teacher-forced-factorial-audit",
        "request_cache_key": cache_key,
        "request": request_value,
        "trace_selective_session_sha256": session_sha,
        "trace_selective_session_receipt": session,
        "prompt_count": len(results),
        "prompt_result_sha256s": [row["result_sha256"] for row in results],
        "lane_summaries": summaries,
        "prompt_results": results,
        "metrics_are_diagnostic_not_an_intelligence_gate": True,
        "manual_whole_output_intelligence_review_remains_authoritative": True,
        "operation_audit": dict(_OPERATION_AUDIT),
    }
    return {**body, "audit_sha256": canonical_sha256(body)}


__all__ = [
    "AuthenticatedParentTrajectoryBatch",
    "LANE_CONTRACT",
    "LFM2_TEACHER_FORCED_TRAJECTORY_AUDIT_SCHEMA",
    "LFM2_TEACHER_FORCED_TRAJECTORY_BATCH_SCHEMA",
    "LFM2_TEACHER_FORCED_TRAJECTORY_PROMPT_RESULT_SCHEMA",
    "LFM2_TEACHER_FORCED_TRAJECTORY_REQUEST_SCHEMA",
    "Lfm2TeacherForcedTrajectoryAuditError",
    "READOUT_CONTRACT",
    "aggregate_teacher_forced_trajectory_audit",
    "assert_text_span_roundtrip_only",
    "authenticate_prompt_result",
    "build_teacher_forced_trajectory_audit_request",
    "parent_winner_rank_margin",
    "resolve_floating_dtype",
    "resolve_runtime_dtype_policy",
    "run_teacher_forced_trajectory_batch",
    "seal_parent_trajectory_batch",
    "tensor_sha256",
]
