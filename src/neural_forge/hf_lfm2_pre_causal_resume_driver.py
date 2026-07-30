"""Parent-free causal-head continuation for an authenticated LFM2 body.

The expensive whole-body fold ends at
``hf_lfm2_pre_causal_body_resume_capsule``.  This module resumes only the
causal output boundary from that immutable checkpoint.  It restores the
typed causal trie from its already-sealed receipt, replays the *child* body
through those fixed prefixes, analytically compiles one requested residual
head policy, and publishes a freshly reloaded standalone candidate plus all
decoded responses required for manual review.

No parent model is accepted by this API.  It performs no parent load,
forward, generation, or trace recapture and it has no training, optimizer,
autograd, pruning, quantization, sparsity, sharing, or sidecar path.  Numeric
diagnostics are retained as evidence and never decide intelligence.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import (
    fingerprint_model_runtime_state,
    fingerprint_model_state,
)
from neural_forge.dynamic_fold_phase_cache import (
    DynamicFoldPhaseCache,
    DynamicFoldPhaseCacheError,
    FoldPhaseEventLog,
    FoldPhaseKey,
    VerifiedFoldPhaseCacheEntry,
)
from neural_forge.hf_lfm2_bridge_only_control import (
    _tensor_sha256 as _interface_tensor_sha256,
)
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_direct_local import Lfm2DirectLocalCheckpointIO
from neural_forge.hf_lfm2_exact4x_composite_interface import (
    LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA,
    Lfm2IndependentStructuralInterfaceInstall,
    _parameter_topology,
    install_lfm2_independent_causal_output_head,
)
from neural_forge.hf_lfm2_interface_frontier import (
    LFM2_CAUSAL_RESIDUAL_TOKEN_BRIDGE_SCHEMA,
    LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA,
    LFM2_CHILD_CAUSAL_RESIDUAL_EVIDENCE_SCHEMA,
    LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA,
    LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA,
    LFM2_UNSUPPORTED_LOGIT_POLICIES,
    Lfm2CausalTrieEvidence,
    Lfm2ChildCausalResidualEvidence,
    collect_lfm2_child_causal_residual_evidence,
    compile_lfm2_causal_residual_token_bridge,
    compile_lfm2_dense_mapped_root_hybrid_causal_evidence,
    expand_lfm2_causal_trie_internal_prefixes,
    fingerprint_lfm2_causal_body_inventory,
)
from neural_forge.hf_lfm2_mapped_causal_target_capsule import (
    Lfm2MappedCausalTargetReference,
    load_lfm2_mapped_causal_target_capsule,
)
from neural_forge.hf_lfm2_pre_causal_body_resume_capsule import (
    Lfm2PreCausalBodyResumeReference,
    load_lfm2_pre_causal_body_resume_capsule,
)
from neural_forge.lfm2_decoded_review import generate_review_rows
from neural_forge.hf_tokenizer_bridge import fingerprint_hf_tokenizer_semantics
from neural_forge.lfm2_exact4x_contract import EXPECTED_PARENT_UNIQUE_PARAMETERS


LFM2_PRE_CAUSAL_RESUME_RUN_SCHEMA = (
    "neural-forge-lfm2-parent-free-pre-causal-resume-run-v1"
)
LFM2_PRE_CAUSAL_RESUME_DOSSIER_SCHEMA = (
    "neural-forge-lfm2-parent-free-pre-causal-resume-dossier-v1"
)
LFM2_PINNED_CAUSAL_RESUME_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-pinned-causal-resume-evidence-v1"
)
_EXPANDED_CAUSAL_PLAN_PHASE = "lfm2-expanded-causal-plan"
_CHILD_CAUSAL_REPLAY_PHASE = "lfm2-child-causal-replay-evidence"
_EXPANDED_CAUSAL_PLAN_CACHE_ABI = (
    "neural-forge-lfm2-expanded-causal-plan-cache-abi-v1"
)
_CHILD_CAUSAL_REPLAY_CACHE_ABI = (
    "neural-forge-lfm2-child-causal-replay-cache-abi-v1"
)
_EXPANDED_CAUSAL_PLAN_FILENAME = "expanded-causal-plan.json"
_CHILD_CAUSAL_REPLAY_RECEIPT_FILENAME = "child-causal-replay.json"
_CHILD_CAUSAL_REPLAY_TENSOR_FILENAME = "child-causal-replay.safetensors"
_CHILD_CAUSAL_REPLAY_TENSOR_KEY = "hidden_rows"


class Lfm2PreCausalResumeError(RuntimeError):
    """A pinned body, cached causal source, or continuation failed closed."""


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise Lfm2PreCausalResumeError(
            f"{label} must be one nonzero lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2PreCausalResumeError(f"{label} must be a positive integer.")
    return value


def _finite_positive(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise Lfm2PreCausalResumeError(f"{label} must be finite and positive.")
    return float(value)


def _verify_seal(
    value: Any,
    *,
    field: str,
    schema: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2PreCausalResumeError(f"{label} must be one mapping.")
    result = dict(value)
    body = dict(result)
    supplied = _digest(body.pop(field, None), f"{label} {field}")
    if body.get("schema") != schema or canonical_sha256(body) != supplied:
        raise Lfm2PreCausalResumeError(f"{label} seal or schema is invalid.")
    return result


def _sealed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise Lfm2PreCausalResumeError(
            f"Cannot seal a document which already contains {field!r}."
        )
    value = dict(body)
    return {**value, field: canonical_sha256(value)}


def _logsumexp(values: Sequence[float], *, label: str) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required LFM2 dependency
        raise RuntimeError("PyTorch is required for causal evidence restore.") from exc
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise Lfm2PreCausalResumeError(f"{label} has no finite donor logits.")
    return torch.logsumexp(torch.tensor(values, dtype=torch.float32), dim=0)


def restore_lfm2_causal_trie_evidence_from_receipt(
    receipt: Mapping[str, Any],
) -> Lfm2CausalTrieEvidence:
    """Restore exact typed trie tensors from one immutable sealed receipt.

    The causal-trie receipt already retains every bounded append candidate,
    its source logit, the selected internal prefixes, and their order.  Those
    values are sufficient to replay the original deterministic float32
    log-sum-exp reductions.  The typed evidence constructor then authenticates
    the reconstructed tensor hashes against the original receipt.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required LFM2 dependency
        raise RuntimeError("PyTorch is required for causal evidence restore.") from exc
    value = _verify_seal(
        receipt,
        field="result_sha256",
        schema=LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA,
        label="causal trie evidence",
    )
    transition = value.get("contextual_transition_plan")
    if not isinstance(transition, Mapping):
        raise Lfm2PreCausalResumeError(
            "Causal trie receipt has no contextual transition plan."
        )
    transition = dict(transition)
    transition_body = dict(transition)
    transition_sha = _digest(
        transition_body.pop("plan_sha256", None),
        "contextual transition plan",
    )
    root_plans = transition.get("root_transition_plans")
    target_rows = _positive_int(
        transition.get("target_row_count"),
        "causal trie target row count",
    )
    if (
        canonical_sha256(transition_body) != transition_sha
        or transition_sha != value.get("contextual_transition_plan_sha256")
        or not isinstance(root_plans, list)
        or not root_plans
        or value.get("root_row_count") != len(root_plans)
    ):
        raise Lfm2PreCausalResumeError(
            "Causal trie transition plan is stale or incomplete."
        )

    canonical_roots: list[tuple[int, ...]] = []
    candidates_by_root: list[tuple[dict[str, Any], ...]] = []
    logits: list[Any] = []
    supports: list[Any] = []
    for expected_root_index, raw_plan in enumerate(root_plans):
        if not isinstance(raw_plan, Mapping):
            raise Lfm2PreCausalResumeError("A causal root plan is not a mapping.")
        plan = dict(raw_plan)
        plan_body = dict(plan)
        plan_sha = _digest(
            plan_body.pop("root_transition_plan_sha256", None),
            "root transition plan",
        )
        root = plan.get("target_root_token_ids")
        raw_candidates = plan.get("append_only_candidates")
        if (
            canonical_sha256(plan_body) != plan_sha
            or plan.get("root_index") != expected_root_index
            or not isinstance(root, list)
            or not root
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                or token_id >= target_rows
                for token_id in root
            )
            or not isinstance(raw_candidates, list)
        ):
            raise Lfm2PreCausalResumeError(
                "A causal root plan has invalid ancestry, prefix, or candidates."
            )
        candidates: list[dict[str, Any]] = []
        grouped: dict[int, list[float]] = {}
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, Mapping):
                raise Lfm2PreCausalResumeError(
                    "A causal append candidate is not a mapping."
                )
            candidate = dict(raw_candidate)
            append = candidate.get("target_append_token_ids")
            donor_logit = candidate.get("donor_logit")
            if (
                not isinstance(append, list)
                or not append
                or any(
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    or token_id >= target_rows
                    for token_id in append
                )
                or isinstance(donor_logit, bool)
                or not isinstance(donor_logit, (int, float))
                or not math.isfinite(float(donor_logit))
            ):
                raise Lfm2PreCausalResumeError(
                    "A causal append candidate has invalid tokens or logit."
                )
            grouped.setdefault(int(append[0]), []).append(float(donor_logit))
            candidates.append(candidate)
        row = torch.zeros(target_rows, dtype=torch.float32)
        support = torch.zeros(target_rows, dtype=torch.bool)
        for target_id, donor_values in sorted(grouped.items()):
            row[target_id] = _logsumexp(
                donor_values,
                label=f"causal root {expected_root_index} target {target_id}",
            )
            support[target_id] = True
        canonical_roots.append(tuple(int(token_id) for token_id in root))
        candidates_by_root.append(tuple(candidates))
        logits.append(row)
        supports.append(support)

    node_records = value.get("node_records")
    if not isinstance(node_records, list):
        raise Lfm2PreCausalResumeError("Causal trie node records are missing.")
    prefixes = list(canonical_roots)
    for raw_node in node_records:
        if not isinstance(raw_node, Mapping):
            raise Lfm2PreCausalResumeError("A causal trie node is not a mapping.")
        node = dict(raw_node)
        root_index = node.get("root_index")
        trie_prefix = node.get("trie_prefix")
        if (
            isinstance(root_index, bool)
            or not isinstance(root_index, int)
            or not 0 <= root_index < len(canonical_roots)
            or not isinstance(trie_prefix, list)
            or not trie_prefix
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                or token_id >= target_rows
                for token_id in trie_prefix
            )
            or node.get("root_transition_plan_sha256")
            != root_plans[root_index].get("root_transition_plan_sha256")
        ):
            raise Lfm2PreCausalResumeError(
                "A causal trie node has invalid root or prefix ancestry."
            )
        prefix = tuple(int(token_id) for token_id in trie_prefix)
        grouped: dict[int, list[float]] = {}
        for candidate in candidates_by_root[root_index]:
            append = tuple(int(token_id) for token_id in candidate["target_append_token_ids"])
            if len(append) > len(prefix) and append[: len(prefix)] == prefix:
                grouped.setdefault(append[len(prefix)], []).append(
                    float(candidate["donor_logit"])
                )
        continuation_ids = sorted(grouped)
        if not grouped or continuation_ids != node.get("continuation_target_ids"):
            raise Lfm2PreCausalResumeError(
                "A causal trie node no longer matches its retained candidates."
            )
        row = torch.zeros(target_rows, dtype=torch.float32)
        support = torch.zeros(target_rows, dtype=torch.bool)
        for target_id, donor_values in sorted(grouped.items()):
            row[target_id] = _logsumexp(
                donor_values,
                label=f"causal trie node target {target_id}",
            )
            support[target_id] = True
        prefixes.append((*canonical_roots[root_index], *prefix))
        logits.append(row)
        supports.append(support)

    if (
        value.get("internal_prefix_row_count") != len(node_records)
        or value.get("total_row_count") != len(prefixes)
    ):
        raise Lfm2PreCausalResumeError(
            "Causal trie row counts differ from its retained node inventory."
        )
    return Lfm2CausalTrieEvidence(
        target_prefix_token_ids=tuple(prefixes),
        target_logits=torch.stack(logits).contiguous(),
        target_logit_support_mask=torch.stack(supports).contiguous(),
        receipt=value,
    )


@dataclass(frozen=True, slots=True)
class Lfm2PinnedCausalResumeEvidence:
    """Immutable parent-cache lineage and restored target-native causal rows."""

    terminal_receipt: Mapping[str, Any]
    causal_trie: Lfm2CausalTrieEvidence
    sequence_plan: Mapping[str, Any]
    structural_interface_receipt: Mapping[str, Any]
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        terminal = _verify_seal(
            self.terminal_receipt,
            field="evidence_sha256",
            schema=LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA,
            label="terminal cache evidence",
        )
        if not isinstance(self.causal_trie, Lfm2CausalTrieEvidence):
            raise TypeError("causal_trie must be Lfm2CausalTrieEvidence.")
        trie = self.causal_trie.as_dict()
        sequence = _verify_seal(
            self.sequence_plan,
            field="plan_sha256",
            schema=LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA,
            label="interface sequence plan",
        )
        structural = _verify_seal(
            self.structural_interface_receipt,
            field="result_sha256",
            schema=LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA,
            label="structural interface install",
        )
        common = (
            "cache_key",
            "parent_identity_sha256",
            "selective_session_sha256",
            "manifest_sha256",
            "completion_sha256",
            "semantic_alignment_sha256",
            "tokenizer_live_session_sha256",
            "causal_root_rows_sha256",
            "bridge_artifact_sha256",
            "bridge_profile_sha256",
            "donor_tokenizer_semantics_sha256",
            "target_tokenizer_semantics_sha256",
        )
        if (
            trie.get("parent_terminal_evidence_sha256")
            != terminal["evidence_sha256"]
            or any(terminal.get(name) != trie.get(name) for name in common)
            or structural.get("sequence_plan_sha256") != sequence["plan_sha256"]
            or sequence.get("bridge_artifact_sha256")
            != trie.get("bridge_artifact_sha256")
            or sequence.get("bridge_profile_sha256")
            != trie.get("bridge_profile_sha256")
            or sequence.get("donor_tokenizer_semantics_sha256")
            != trie.get("donor_tokenizer_semantics_sha256")
            or sequence.get("target_tokenizer_semantics_sha256")
            != trie.get("target_tokenizer_semantics_sha256")
        ):
            raise Lfm2PreCausalResumeError(
                "Pinned terminal, trie, sequence, and structural evidence diverged."
            )
        audit = terminal.get("operation_audit")
        if (
            not isinstance(audit, Mapping)
            or audit.get("network_access") is not False
            or any(
                audit.get(name) != 0
                for name in (
                    "training_steps",
                    "optimizer_constructions",
                    "optimizer_steps",
                    "autograd_grad_calls",
                    "backward_calls",
                    "parent_forward_calls",
                    "parent_generate_calls",
                    "parent_trace_recaptures",
                    "child_forward_calls",
                    "child_generate_calls",
                )
            )
        ):
            raise Lfm2PreCausalResumeError(
                "Terminal cache receipt does not prove a cache-only source."
            )
        body = {
            "schema": LFM2_PINNED_CAUSAL_RESUME_EVIDENCE_SCHEMA,
            "terminal_evidence_sha256": terminal["evidence_sha256"],
            "causal_trie_evidence_sha256": trie["result_sha256"],
            "sequence_plan_sha256": sequence["plan_sha256"],
            "structural_interface_install_sha256": structural["result_sha256"],
            "parent_identity_sha256": terminal["parent_identity_sha256"],
            "bridge_artifact_sha256": terminal["bridge_artifact_sha256"],
            "target_tokenizer_semantics_sha256": terminal[
                "target_tokenizer_semantics_sha256"
            ],
            "typed_trie_restored_without_parent_cache_or_model": True,
            "parent_trace_regeneration_performed": False,
        }
        seal = canonical_sha256(body)
        if self.evidence_sha256 and self.evidence_sha256 != seal:
            raise Lfm2PreCausalResumeError(
                "Pinned causal resume evidence self-seal is invalid."
            )
        object.__setattr__(self, "terminal_receipt", MappingProxyType(terminal))
        object.__setattr__(self, "sequence_plan", MappingProxyType(sequence))
        object.__setattr__(
            self,
            "structural_interface_receipt",
            MappingProxyType(structural),
        )
        object.__setattr__(self, "evidence_sha256", seal)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": LFM2_PINNED_CAUSAL_RESUME_EVIDENCE_SCHEMA,
            "terminal_evidence_sha256": self.terminal_receipt["evidence_sha256"],
            "causal_trie_evidence_sha256": self.causal_trie.receipt["result_sha256"],
            "sequence_plan_sha256": self.sequence_plan["plan_sha256"],
            "structural_interface_install_sha256": (
                self.structural_interface_receipt["result_sha256"]
            ),
            "parent_identity_sha256": self.terminal_receipt[
                "parent_identity_sha256"
            ],
            "bridge_artifact_sha256": self.terminal_receipt[
                "bridge_artifact_sha256"
            ],
            "target_tokenizer_semantics_sha256": self.terminal_receipt[
                "target_tokenizer_semantics_sha256"
            ],
            "typed_trie_restored_without_parent_cache_or_model": True,
            "parent_trace_regeneration_performed": False,
            "evidence_sha256": self.evidence_sha256,
        }


def _reauthenticate_pinned_causal_resume_evidence(
    evidence: Lfm2PinnedCausalResumeEvidence,
) -> Lfm2PinnedCausalResumeEvidence:
    """Rebuild every mutable trie payload and reject live-object drift."""

    if not isinstance(evidence, Lfm2PinnedCausalResumeEvidence):
        raise TypeError("evidence must be Lfm2PinnedCausalResumeEvidence.")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required LFM2 dependency
        raise RuntimeError("PyTorch is required for causal evidence restore.") from exc
    live_trie = evidence.causal_trie
    restored_trie = restore_lfm2_causal_trie_evidence_from_receipt(
        live_trie.as_dict()
    )
    if (
        live_trie.target_prefix_token_ids
        != restored_trie.target_prefix_token_ids
        or not torch.equal(live_trie.target_logits, restored_trie.target_logits)
        or not torch.equal(
            live_trie.target_logit_support_mask,
            restored_trie.target_logit_support_mask,
        )
    ):
        raise Lfm2PreCausalResumeError(
            "Live causal trie tensors or prefixes changed after construction."
        )
    restored = Lfm2PinnedCausalResumeEvidence(
        terminal_receipt=dict(evidence.terminal_receipt),
        causal_trie=restored_trie,
        sequence_plan=dict(evidence.sequence_plan),
        structural_interface_receipt=dict(
            evidence.structural_interface_receipt
        ),
    )
    if restored.evidence_sha256 != evidence.evidence_sha256:
        raise Lfm2PreCausalResumeError(
            "Pinned causal evidence lineage changed after construction."
        )
    return restored


def _new_directory(value: str | os.PathLike[str]) -> Path:
    output = Path(value).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise Lfm2PreCausalResumeError(
            f"Refusing to replace existing resume output {output}."
        )
    parent = output.parent
    if not parent.is_dir() or parent.is_symlink():
        raise Lfm2PreCausalResumeError(
            "Resume output parent must be one existing real directory."
        )
    output.mkdir()
    return output


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Lfm2PreCausalResumeError(f"Refusing to replace evidence {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


def _read_phase_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2PreCausalResumeError(
            f"Cached {label} is not readable ASCII JSON."
        ) from exc
    if not isinstance(value, Mapping):
        raise Lfm2PreCausalResumeError(f"Cached {label} must be one JSON object.")
    return dict(value)


def _callable_source_sha256(value: Any, label: str) -> str:
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError) as exc:
        raise Lfm2PreCausalResumeError(
            f"Cannot bind {label} implementation source to a phase key."
        ) from exc
    if not source:
        raise Lfm2PreCausalResumeError(
            f"Cannot bind empty {label} implementation source."
        )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _phase_event_log(
    path: str | os.PathLike[str] | None,
    *,
    body_reference: Lfm2PreCausalBodyResumeReference,
    evidence: Lfm2PinnedCausalResumeEvidence,
    candidate_id: str,
) -> FoldPhaseEventLog | None:
    if path is None:
        return None
    run_sha256 = canonical_sha256(
        {
            "schema": "neural-forge-lfm2-pre-causal-phase-event-run-v1",
            "body_capsule_receipt_sha256": body_reference.receipt_sha256,
            "body_capsule_bindings_sha256": body_reference.bindings_sha256,
            "pinned_causal_evidence_sha256": evidence.evidence_sha256,
            "candidate_id": candidate_id,
        }
    )
    return FoldPhaseEventLog(
        path,
        run_id=f"lfm2-pre-causal-{run_sha256[:24]}",
    )


def _process_resident_bytes() -> int | None:
    statm = Path("/proc/self/statm")
    if not statm.is_file():
        return None
    try:
        fields = statm.read_text(encoding="ascii").split()
        if len(fields) < 2:
            return None
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, TypeError, ValueError):
        return None


def _phase_memory_observation(device: Any) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required LFM2 dependency
        raise RuntimeError("PyTorch is required for phase observations.") from exc
    requested = torch.device(device)
    result: dict[str, Any] = {
        "process_resident_bytes": _process_resident_bytes(),
        "requested_device": str(requested),
        "cuda_memory_available": False,
        "cuda_memory_allocated_bytes": None,
        "cuda_memory_reserved_bytes": None,
        "cuda_max_memory_allocated_bytes": None,
    }
    if requested.type == "cuda" and torch.cuda.is_available():
        result.update(
            {
                "cuda_memory_available": True,
                "cuda_memory_allocated_bytes": int(
                    torch.cuda.memory_allocated(requested)
                ),
                "cuda_memory_reserved_bytes": int(
                    torch.cuda.memory_reserved(requested)
                ),
                "cuda_max_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(requested)
                ),
            }
        )
    return result


def _phase_work_event(
    *,
    stage: str,
    completed_work: int,
    total_work: int,
    work_unit: str,
    started_at: float,
    device: Any,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "completed_work": completed_work,
        "total_work": total_work,
        "work_unit": work_unit,
        "elapsed_seconds": max(0.0, float(time.monotonic() - started_at)),
        "memory": _phase_memory_observation(device),
    }


def _expanded_causal_plan_phase_key(
    evidence: Lfm2PinnedCausalResumeEvidence,
    source: Lfm2CausalTrieEvidence,
) -> FoldPhaseKey:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required LFM2 dependency
        raise RuntimeError("PyTorch is required for causal expansion.") from exc
    receipt = source.as_dict()
    return FoldPhaseKey(
        phase_name=_EXPANDED_CAUSAL_PLAN_PHASE,
        dependency_sha256s={
            "pinned-causal-evidence": evidence.evidence_sha256,
            "terminal-receipt": evidence.terminal_receipt["evidence_sha256"],
            "source-causal-trie": receipt["result_sha256"],
            "sequence-plan": evidence.sequence_plan["plan_sha256"],
            "structural-interface": evidence.structural_interface_receipt[
                "result_sha256"
            ],
            "expansion-implementation": _callable_source_sha256(
                expand_lfm2_causal_trie_internal_prefixes,
                "causal-trie expansion",
            ),
            "receipt-restoration-implementation": _callable_source_sha256(
                restore_lfm2_causal_trie_evidence_from_receipt,
                "causal-trie restoration",
            ),
        },
        numeric_contract={
            "execution_device": "cpu",
            "reduction_dtype": "torch.float32",
            "source_logits_dtype": str(source.target_logits.dtype),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": (
                None if torch.version.cuda is None else str(torch.version.cuda)
            ),
            "sealed_donor_candidates_only": True,
            "parent_execution": False,
        },
        abi={
            "schema": _EXPANDED_CAUSAL_PLAN_CACHE_ABI,
            "evidence_schema": LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA,
            "payload_filename": _EXPANDED_CAUSAL_PLAN_FILENAME,
            "payload_kind": "sealed-causal-trie-receipt",
            "restoration_api": "restore_lfm2_causal_trie_evidence_from_receipt",
        },
    )


def _child_replay_numeric_contract(
    model: Any,
    *,
    model_device: Any,
    maximum_batch_rows: int,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required LFM2 dependency
        raise RuntimeError("PyTorch is required for child causal replay.") from exc
    try:
        model_dtype = str(next(model.parameters()).dtype)
    except StopIteration as exc:
        raise Lfm2PreCausalResumeError(
            "Child causal replay model has no parameters."
        ) from exc
    return {
        "requested_device": str(torch.device(model_device)),
        "model_parameter_dtype": model_dtype,
        "maximum_batch_rows": maximum_batch_rows,
        "inference_mode": True,
        "full_prefix_use_cache": False,
        "position_origin": 0,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
    }


def _child_causal_replay_phase_key(
    *,
    model: Any,
    active_causal_evidence: Lfm2CausalTrieEvidence,
    evidence: Lfm2PinnedCausalResumeEvidence,
    body_reference: Lfm2PreCausalBodyResumeReference,
    bindings: Any,
    model_device: Any,
    maximum_batch_rows: int,
) -> FoldPhaseKey:
    active_receipt = active_causal_evidence.as_dict()
    return FoldPhaseKey(
        phase_name=_CHILD_CAUSAL_REPLAY_PHASE,
        dependency_sha256s={
            "active-causal-evidence": active_receipt["result_sha256"],
            "pinned-causal-evidence": evidence.evidence_sha256,
            "terminal-receipt": evidence.terminal_receipt["evidence_sha256"],
            "sequence-plan": evidence.sequence_plan["plan_sha256"],
            "structural-interface": evidence.structural_interface_receipt[
                "result_sha256"
            ],
            "body-capsule-receipt": body_reference.receipt_sha256,
            "body-capsule-manifest": body_reference.manifest_sha256,
            "body-checkpoint-inventory": (
                body_reference.checkpoint_inventory_sha256
            ),
            "body-bindings": bindings.bindings_sha256,
            "body-model-state": bindings.body_model_state_sha256,
            "body-runtime-state": bindings.body_runtime_state_sha256,
            "causal-input-table": bindings.installed_input_table_sha256,
            "causal-body-inventory": bindings.causal_body_inventory_sha256,
            "bridge-artifact": bindings.bridge_artifact_sha256,
            "bridge-profile": bindings.bridge_profile_sha256,
            "target-tokenizer-semantics": (
                bindings.target_tokenizer_semantics_sha256
            ),
            "replay-implementation": _callable_source_sha256(
                collect_lfm2_child_causal_residual_evidence,
                "child causal replay",
            ),
            "evidence-restoration-implementation": _callable_source_sha256(
                _restore_cached_child_causal_evidence,
                "cached child causal evidence restoration",
            ),
        },
        numeric_contract=_child_replay_numeric_contract(
            model,
            model_device=model_device,
            maximum_batch_rows=maximum_batch_rows,
        ),
        abi={
            "schema": _CHILD_CAUSAL_REPLAY_CACHE_ABI,
            "evidence_schema": LFM2_CHILD_CAUSAL_RESIDUAL_EVIDENCE_SCHEMA,
            "consumer_schema": LFM2_CAUSAL_RESIDUAL_TOKEN_BRIDGE_SCHEMA,
            "receipt_filename": _CHILD_CAUSAL_REPLAY_RECEIPT_FILENAME,
            "tensor_filename": _CHILD_CAUSAL_REPLAY_TENSOR_FILENAME,
            "tensor_key": _CHILD_CAUSAL_REPLAY_TENSOR_KEY,
            "tensor_format": "safetensors",
            "evidence_role": "parent-free-pre-causal-body-resume",
            "row_count": len(active_causal_evidence.target_prefix_token_ids),
            "hidden_size": int(model.config.hidden_size),
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        },
    )


def _require_phase_payload_files(
    entry: VerifiedFoldPhaseCacheEntry,
    expected: set[str],
) -> None:
    inventory = entry.payload_inventory
    files = inventory.get("files")
    directories = inventory.get("directories")
    actual = (
        {str(record.get("path")) for record in files}
        if isinstance(files, list)
        else set()
    )
    if actual != expected or directories != []:
        raise Lfm2PreCausalResumeError(
            "Cached phase payload has missing or foreign semantic files."
        )


def _restore_cached_expanded_causal_plan(
    entry: VerifiedFoldPhaseCacheEntry,
) -> Lfm2CausalTrieEvidence:
    _require_phase_payload_files(entry, {_EXPANDED_CAUSAL_PLAN_FILENAME})
    receipt = _read_phase_json(
        entry.payload_path / _EXPANDED_CAUSAL_PLAN_FILENAME,
        "expanded causal plan",
    )
    restored = restore_lfm2_causal_trie_evidence_from_receipt(receipt)
    if entry.result.get("evidence_sha256") != receipt.get("result_sha256"):
        raise Lfm2PreCausalResumeError(
            "Cached expanded causal plan differs from its phase result binding."
        )
    return restored


def _write_cached_child_causal_evidence(
    payload_path: Path,
    evidence: Lfm2ChildCausalResidualEvidence,
) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - required cache dependency
        raise Lfm2PreCausalResumeError(
            "safetensors is required when child replay phase caching is enabled."
        ) from exc
    receipt = evidence.as_dict()
    _write_json(
        payload_path / _CHILD_CAUSAL_REPLAY_RECEIPT_FILENAME,
        receipt,
    )
    tensor_path = payload_path / _CHILD_CAUSAL_REPLAY_TENSOR_FILENAME
    if tensor_path.exists() or tensor_path.is_symlink():
        raise Lfm2PreCausalResumeError(
            "Refusing to replace cached child causal replay tensors."
        )
    save_file(
        {
            _CHILD_CAUSAL_REPLAY_TENSOR_KEY: (
                evidence.hidden_rows.detach().cpu().contiguous()
            )
        },
        str(tensor_path),
        metadata={
            "schema": _CHILD_CAUSAL_REPLAY_CACHE_ABI,
            "evidence_sha256": str(receipt["evidence_sha256"]),
        },
    )


def _restore_cached_child_causal_evidence(
    entry: VerifiedFoldPhaseCacheEntry,
    *,
    expected_trie_sha256: str,
    expected_input_table_sha256: str,
    expected_body_inventory_sha256: str,
    expected_model_state_sha256: str,
) -> Lfm2ChildCausalResidualEvidence:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - required cache dependency
        raise Lfm2PreCausalResumeError(
            "safetensors is required to restore cached child causal replay."
        ) from exc
    _require_phase_payload_files(
        entry,
        {
            _CHILD_CAUSAL_REPLAY_RECEIPT_FILENAME,
            _CHILD_CAUSAL_REPLAY_TENSOR_FILENAME,
        },
    )
    receipt = _read_phase_json(
        entry.payload_path / _CHILD_CAUSAL_REPLAY_RECEIPT_FILENAME,
        "child causal replay receipt",
    )
    tensors = load_file(
        str(entry.payload_path / _CHILD_CAUSAL_REPLAY_TENSOR_FILENAME),
        device="cpu",
    )
    if set(tensors) != {_CHILD_CAUSAL_REPLAY_TENSOR_KEY}:
        raise Lfm2PreCausalResumeError(
            "Cached child causal replay tensor keys are incomplete or foreign."
        )
    restored = Lfm2ChildCausalResidualEvidence(
        hidden_rows=tensors[_CHILD_CAUSAL_REPLAY_TENSOR_KEY],
        receipt=receipt,
    )
    if (
        entry.result.get("evidence_sha256") != receipt.get("evidence_sha256")
        or receipt.get("trie_evidence_sha256") != expected_trie_sha256
        or receipt.get("causal_input_table_sha256")
        != expected_input_table_sha256
        or receipt.get("causal_body_inventory_sha256")
        != expected_body_inventory_sha256
        or receipt.get("child_model_state_before_sha256")
        != expected_model_state_sha256
    ):
        raise Lfm2PreCausalResumeError(
            "Cached child causal replay is stale against the live body lineage."
        )
    return restored


def _phase_cache_audit(
    *,
    key: FoldPhaseKey,
    cache_hit: bool,
    entry: VerifiedFoldPhaseCacheEntry,
) -> dict[str, Any]:
    return {
        "phase_name": key.phase_name,
        "phase_key_sha256": key.phase_key_sha256,
        "cache_hit": cache_hit,
        "completion_sha256": entry.completion_sha256,
        "payload_tree_sha256": entry.payload_inventory["payload_tree_sha256"],
        "payload_total_bytes": entry.payload_inventory["total_bytes"],
    }


def _expanded_causal_plan_with_optional_cache(
    source: Lfm2CausalTrieEvidence,
    *,
    evidence: Lfm2PinnedCausalResumeEvidence,
    cache: DynamicFoldPhaseCache | None,
    events: FoldPhaseEventLog | None,
) -> tuple[Lfm2CausalTrieEvidence, dict[str, Any] | None]:
    key = _expanded_causal_plan_phase_key(evidence, source)
    if cache is not None:
        try:
            entry = cache.lookup(key, events=events)
            if entry is not None:
                restored = _restore_cached_expanded_causal_plan(entry)
                if events is not None:
                    events.progress(
                        key,
                        _phase_work_event(
                            stage=_EXPANDED_CAUSAL_PLAN_PHASE,
                            completed_work=len(restored.target_prefix_token_ids),
                            total_work=len(restored.target_prefix_token_ids),
                            work_unit="causal-prefix-rows",
                            started_at=time.monotonic(),
                            device="cpu",
                        ),
                    )
                return restored, _phase_cache_audit(
                    key=key,
                    cache_hit=True,
                    entry=entry,
                )
        except DynamicFoldPhaseCacheError as exc:
            raise Lfm2PreCausalResumeError(
                "Expanded causal-plan cache validation failed closed."
            ) from exc

    source_receipt = source.as_dict()
    total_work = max(
        1,
        int(source_receipt.get("root_row_count", 0))
        + int(source_receipt.get("internal_prefix_row_count", 0))
        + int(source_receipt.get("omitted_internal_prefix_count", 0)),
    )
    started_at = time.monotonic()
    if cache is not None:
        try:
            with cache.begin(key, events=events) as transaction:
                transaction.heartbeat(
                    _phase_work_event(
                        stage=_EXPANDED_CAUSAL_PLAN_PHASE,
                        completed_work=0,
                        total_work=total_work,
                        work_unit="causal-prefix-rows",
                        started_at=started_at,
                        device="cpu",
                    )
                )
                expanded = expand_lfm2_causal_trie_internal_prefixes(source)
                transaction.progress(
                    _phase_work_event(
                        stage=_EXPANDED_CAUSAL_PLAN_PHASE,
                        completed_work=len(expanded.target_prefix_token_ids),
                        total_work=len(expanded.target_prefix_token_ids),
                        work_unit="causal-prefix-rows",
                        started_at=started_at,
                        device="cpu",
                    )
                )
                receipt = expanded.as_dict()
                _write_json(
                    transaction.payload_path / _EXPANDED_CAUSAL_PLAN_FILENAME,
                    receipt,
                )
                entry = transaction.complete(
                    result={
                        "evidence_sha256": receipt["result_sha256"],
                        "stage": _EXPANDED_CAUSAL_PLAN_PHASE,
                    }
                )
            return expanded, _phase_cache_audit(
                key=key,
                cache_hit=False,
                entry=entry,
            )
        except DynamicFoldPhaseCacheError as exc:
            raise Lfm2PreCausalResumeError(
                "Expanded causal-plan cache publication failed closed."
            ) from exc

    if events is not None:
        events.begin(key, {"stage": _EXPANDED_CAUSAL_PLAN_PHASE})
        events.heartbeat(
            key,
            _phase_work_event(
                stage=_EXPANDED_CAUSAL_PLAN_PHASE,
                completed_work=0,
                total_work=total_work,
                work_unit="causal-prefix-rows",
                started_at=started_at,
                device="cpu",
            ),
        )
    try:
        expanded = expand_lfm2_causal_trie_internal_prefixes(source)
    except BaseException as exc:
        if events is not None:
            events.fail(
                key,
                {
                    "stage": _EXPANDED_CAUSAL_PLAN_PHASE,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        raise
    if events is not None:
        completed = _phase_work_event(
            stage=_EXPANDED_CAUSAL_PLAN_PHASE,
            completed_work=len(expanded.target_prefix_token_ids),
            total_work=len(expanded.target_prefix_token_ids),
            work_unit="causal-prefix-rows",
            started_at=started_at,
            device="cpu",
        )
        events.progress(key, completed)
        events.complete(
            key,
            {
                **completed,
                "evidence_sha256": expanded.as_dict()["result_sha256"],
            },
        )
    return expanded, None


def _child_causal_replay_with_optional_cache(
    model: Any,
    active_causal_evidence: Lfm2CausalTrieEvidence,
    *,
    evidence: Lfm2PinnedCausalResumeEvidence,
    body_reference: Lfm2PreCausalBodyResumeReference,
    bindings: Any,
    model_device: Any,
    maximum_batch_rows: int,
    cache: DynamicFoldPhaseCache | None,
    events: FoldPhaseEventLog | None,
) -> tuple[Lfm2ChildCausalResidualEvidence, dict[str, Any] | None]:
    key = _child_causal_replay_phase_key(
        model=model,
        active_causal_evidence=active_causal_evidence,
        evidence=evidence,
        body_reference=body_reference,
        bindings=bindings,
        model_device=model_device,
        maximum_batch_rows=maximum_batch_rows,
    )
    active_sha256 = active_causal_evidence.as_dict()["result_sha256"]
    total_work = len(active_causal_evidence.target_prefix_token_ids)
    if cache is not None:
        try:
            entry = cache.lookup(key, events=events)
            if entry is not None:
                restored = _restore_cached_child_causal_evidence(
                    entry,
                    expected_trie_sha256=active_sha256,
                    expected_input_table_sha256=(
                        bindings.installed_input_table_sha256
                    ),
                    expected_body_inventory_sha256=(
                        bindings.causal_body_inventory_sha256
                    ),
                    expected_model_state_sha256=bindings.body_model_state_sha256,
                )
                if events is not None:
                    events.progress(
                        key,
                        _phase_work_event(
                            stage=_CHILD_CAUSAL_REPLAY_PHASE,
                            completed_work=total_work,
                            total_work=total_work,
                            work_unit="child-prefix-rows",
                            started_at=time.monotonic(),
                            device=model_device,
                        ),
                    )
                return restored, _phase_cache_audit(
                    key=key,
                    cache_hit=True,
                    entry=entry,
                )
        except DynamicFoldPhaseCacheError as exc:
            raise Lfm2PreCausalResumeError(
                "Child causal replay cache validation failed closed."
            ) from exc

    started_at = time.monotonic()
    if cache is not None:
        try:
            with cache.begin(key, events=events) as transaction:
                transaction.heartbeat(
                    _phase_work_event(
                        stage=_CHILD_CAUSAL_REPLAY_PHASE,
                        completed_work=0,
                        total_work=total_work,
                        work_unit="child-prefix-rows",
                        started_at=started_at,
                        device=model_device,
                    )
                )
                child_evidence = collect_lfm2_child_causal_residual_evidence(
                    model,
                    active_causal_evidence,
                    expected_input_table_sha256=(
                        bindings.installed_input_table_sha256
                    ),
                    expected_body_inventory_sha256=(
                        bindings.causal_body_inventory_sha256
                    ),
                    device=model_device,
                    evidence_role="parent-free-pre-causal-body-resume",
                    maximum_batch_rows=maximum_batch_rows,
                )
                transaction.progress(
                    _phase_work_event(
                        stage=_CHILD_CAUSAL_REPLAY_PHASE,
                        completed_work=total_work,
                        total_work=total_work,
                        work_unit="child-prefix-rows",
                        started_at=started_at,
                        device=model_device,
                    )
                )
                _write_cached_child_causal_evidence(
                    transaction.payload_path,
                    child_evidence,
                )
                receipt = child_evidence.as_dict()
                entry = transaction.complete(
                    result={
                        "evidence_sha256": receipt["evidence_sha256"],
                        "stage": _CHILD_CAUSAL_REPLAY_PHASE,
                    }
                )
            return child_evidence, _phase_cache_audit(
                key=key,
                cache_hit=False,
                entry=entry,
            )
        except DynamicFoldPhaseCacheError as exc:
            raise Lfm2PreCausalResumeError(
                "Child causal replay cache publication failed closed."
            ) from exc

    if events is not None:
        events.begin(key, {"stage": _CHILD_CAUSAL_REPLAY_PHASE})
        events.heartbeat(
            key,
            _phase_work_event(
                stage=_CHILD_CAUSAL_REPLAY_PHASE,
                completed_work=0,
                total_work=total_work,
                work_unit="child-prefix-rows",
                started_at=started_at,
                device=model_device,
            ),
        )
    try:
        child_evidence = collect_lfm2_child_causal_residual_evidence(
            model,
            active_causal_evidence,
            expected_input_table_sha256=bindings.installed_input_table_sha256,
            expected_body_inventory_sha256=bindings.causal_body_inventory_sha256,
            device=model_device,
            evidence_role="parent-free-pre-causal-body-resume",
            maximum_batch_rows=maximum_batch_rows,
        )
    except BaseException as exc:
        if events is not None:
            events.fail(
                key,
                {
                    "stage": _CHILD_CAUSAL_REPLAY_PHASE,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        raise
    if events is not None:
        completed = _phase_work_event(
            stage=_CHILD_CAUSAL_REPLAY_PHASE,
            completed_work=total_work,
            total_work=total_work,
            work_unit="child-prefix-rows",
            started_at=started_at,
            device=model_device,
        )
        events.progress(key, completed)
        events.complete(
            key,
            {
                **completed,
                "evidence_sha256": child_evidence.as_dict()["evidence_sha256"],
            },
        )
    return child_evidence, None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_inventory(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise Lfm2PreCausalResumeError("Candidate checkpoint is not a real directory.")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise Lfm2PreCausalResumeError(
                "Standalone candidate contains a symbolic link."
            )
        if path.is_dir():
            continue
        if not path.is_file() or path.suffix in {".pt", ".pth", ".pkl", ".pickle"}:
            raise Lfm2PreCausalResumeError(
                "Standalone candidate contains an unsafe or pickle payload."
            )
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not rows:
        raise Lfm2PreCausalResumeError("Standalone candidate inventory is empty.")
    return rows


def _unique_parameter_count(model: Any) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += int(parameter.numel())
    return total


def _storage_signature(model: Any) -> tuple[tuple[Any, ...], ...]:
    rows = []
    for name, value in tuple(model.named_parameters()) + tuple(model.named_buffers()):
        rows.append(
            (
                str(name),
                id(value),
                str(value.device),
                int(value.untyped_storage().data_ptr()),
                int(value.storage_offset()),
                tuple(int(size) for size in value.shape),
                tuple(int(stride) for stride in value.stride()),
            )
        )
    return tuple(rows)


def _storage_addresses(model: Any) -> set[tuple[Any, ...]]:
    return {
        (
            str(value.device),
            int(value.untyped_storage().data_ptr()),
            int(value.storage_offset()),
            tuple(int(size) for size in value.shape),
        )
        for value in tuple(model.parameters()) + tuple(model.buffers())
    }


def _install_from_frozen_resume(
    *,
    model: Any,
    capsule_manifest: Mapping[str, Any],
    structural_receipt: Mapping[str, Any],
    causal_bridge: Any,
    child_evidence: Any,
    expected_input_table_sha256: str,
    expected_body_inventory_sha256: str,
) -> tuple[Any, dict[str, Any]]:
    """Bridge the historical trainability-only topology seal, if necessary.

    The structural receipt was issued before the body capsule's authenticated
    inference-policy transition.  Its topology digest includes
    ``requires_grad`` even though that flag is not model state.  Recreate only
    the all-trainable historical flag set long enough for the existing output
    installer to authenticate its old receipt, under no-grad/no-autograd, then
    restore the required frozen policy.  Every object, storage, tensor byte,
    and gradient absence is checked around both transitions.
    """

    parameters = tuple(model.parameters())
    if not parameters or any(parameter.grad is not None for parameter in parameters):
        raise Lfm2PreCausalResumeError(
            "Resumed body must be materialized and gradient-free."
        )
    before_signature = _storage_signature(model)
    before_state = fingerprint_model_state(model)
    frozen_topology_sha = canonical_sha256(_parameter_topology(model))
    expected_topology_sha = structural_receipt.get("target_topology_sha256")
    transition = "none-frozen-topology-already-matches"
    transitioned_count = 0
    if frozen_topology_sha != expected_topology_sha:
        runtime = capsule_manifest.get("runtime_boundary_audit")
        incoming_count = (
            runtime.get("incoming_parameter_requires_grad_count")
            if isinstance(runtime, Mapping)
            else None
        )
        if incoming_count != len(parameters):
            raise Lfm2PreCausalResumeError(
                "Historical structural topology used a mixed trainability policy "
                "which this capsule did not retain path-by-path."
            )
        model.requires_grad_(True)
        transitioned_count = len(parameters)
        transition = "all-trainable-structural-receipt-compatibility"
        if canonical_sha256(_parameter_topology(model)) != expected_topology_sha:
            raise Lfm2PreCausalResumeError(
                "Trainability compatibility did not reproduce structural topology."
            )
    if (
        fingerprint_model_state(model) != before_state
        or _storage_signature(model) != before_signature
        or any(parameter.grad is not None for parameter in parameters)
    ):
        raise Lfm2PreCausalResumeError(
            "Trainability compatibility changed tensor state, storage, or gradients."
        )
    structural = Lfm2IndependentStructuralInterfaceInstall(
        model=model,
        receipt=structural_receipt,
    )
    installed = None
    try:
        installed = install_lfm2_independent_causal_output_head(
            structural,
            causal_bridge,
            child_evidence,
            expected_input_table_sha256=expected_input_table_sha256,
            expected_body_inventory_sha256=expected_body_inventory_sha256,
        )
        installed_state = fingerprint_model_state(model)
    finally:
        model.requires_grad_(False)
        model.eval()
    if installed is None:  # pragma: no cover - installer exception is re-raised
        raise AssertionError("Causal installer returned no result.")
    if (
        fingerprint_model_state(model) != installed_state
        or _storage_signature(model) != before_signature
        or any(parameter.requires_grad for parameter in model.parameters())
        or any(parameter.grad is not None for parameter in model.parameters())
        or any(module.training for module in model.modules())
    ):
        raise Lfm2PreCausalResumeError(
            "Restoring frozen inference policy changed the installed child."
        )
    audit_body = {
        "schema": "neural-forge-lfm2-resumed-structural-topology-compatibility-v1",
        "transition": transition,
        "transitioned_parameter_count": transitioned_count,
        "historical_target_topology_sha256": expected_topology_sha,
        "frozen_target_topology_sha256": frozen_topology_sha,
        "state_before_sha256": before_state,
        "state_after_install_sha256": installed_state,
        "object_and_storage_bindings_preserved": True,
        "tensor_bytes_unchanged_by_trainability_transitions": True,
        "parameter_gradients_absent_throughout": True,
        "autograd_execution_performed": False,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "final_parameters_frozen": True,
        "final_model_eval": True,
    }
    return installed, _sealed(audit_body, "result_sha256")


def resume_lfm2_pre_causal_body(
    *,
    body_reference: Lfm2PreCausalBodyResumeReference,
    checkpoint_io: Lfm2DirectLocalCheckpointIO,
    evidence: Lfm2PinnedCausalResumeEvidence,
    target_tokenizer: Any,
    prompts: Sequence[Mapping[str, Any]],
    output_directory: str | os.PathLike[str],
    unsupported_logit_policy: str,
    model_device: Any,
    solve_device: Any,
    ridge_fraction: float,
    residual_scale_ppm: int,
    row_norm_calibration_ppm: int,
    protect_special_tokens: bool = True,
    maximum_prompt_tokens: int = 512,
    maximum_new_tokens: int = 256,
    minimum_parameter_compaction_ratio: float = 4.0,
    candidate_id: str = "parent-free-causal-resume",
    mapped_target_reference: Lfm2MappedCausalTargetReference | None = None,
    expand_all_sealed_internal_prefixes: bool = True,
    child_replay_maximum_batch_rows: int = 32,
    phase_cache_root: str | os.PathLike[str] | None = None,
    phase_event_log_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Compile, install, reload, and expose one resumed causal-head candidate."""

    if not isinstance(body_reference, Lfm2PreCausalBodyResumeReference):
        raise TypeError("body_reference must be Lfm2PreCausalBodyResumeReference.")
    if not isinstance(checkpoint_io, Lfm2DirectLocalCheckpointIO):
        raise TypeError("checkpoint_io must be Lfm2DirectLocalCheckpointIO.")
    if not isinstance(evidence, Lfm2PinnedCausalResumeEvidence):
        raise TypeError("evidence must be Lfm2PinnedCausalResumeEvidence.")
    if mapped_target_reference is not None and not isinstance(
        mapped_target_reference,
        Lfm2MappedCausalTargetReference,
    ):
        raise TypeError(
            "mapped_target_reference must be Lfm2MappedCausalTargetReference "
            "or None."
        )
    if not isinstance(expand_all_sealed_internal_prefixes, bool):
        raise TypeError("expand_all_sealed_internal_prefixes must be bool.")
    child_replay_maximum_batch_rows = _positive_int(
        child_replay_maximum_batch_rows,
        "child_replay_maximum_batch_rows",
    )
    evidence = _reauthenticate_pinned_causal_resume_evidence(evidence)
    parent_count = EXPECTED_PARENT_UNIQUE_PARAMETERS
    maximum_prompt_tokens = _positive_int(
        maximum_prompt_tokens,
        "maximum_prompt_tokens",
    )
    maximum_new_tokens = _positive_int(maximum_new_tokens, "maximum_new_tokens")
    ridge_fraction = _finite_positive(ridge_fraction, "ridge_fraction")
    minimum_ratio = _finite_positive(
        minimum_parameter_compaction_ratio,
        "minimum_parameter_compaction_ratio",
    )
    if unsupported_logit_policy not in LFM2_UNSUPPORTED_LOGIT_POLICIES:
        raise Lfm2PreCausalResumeError(
            "unsupported_logit_policy is not an implemented causal residual policy."
        )
    for value, label in (
        (residual_scale_ppm, "residual_scale_ppm"),
        (row_norm_calibration_ppm, "row_norm_calibration_ppm"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 1_000_000
        ):
            raise Lfm2PreCausalResumeError(
                f"{label} must be an integer from zero through one million."
            )
    prompts = tuple(dict(row) for row in prompts)
    if not prompts:
        raise Lfm2PreCausalResumeError("Manual review prompts cannot be empty.")
    if (
        not isinstance(candidate_id, str)
        or not candidate_id.strip()
        or Path(candidate_id).name != candidate_id
        or candidate_id in {".", ".."}
    ):
        raise Lfm2PreCausalResumeError("candidate_id must be one safe path segment.")

    phase_cache = (
        None
        if phase_cache_root is None
        else DynamicFoldPhaseCache(phase_cache_root)
    )
    phase_events = _phase_event_log(
        phase_event_log_path,
        body_reference=body_reference,
        evidence=evidence,
        candidate_id=candidate_id,
    )
    output_path = Path(output_directory).expanduser().absolute()
    if phase_cache is not None and (
        phase_cache.root == output_path or output_path in phase_cache.root.parents
    ):
        raise Lfm2PreCausalResumeError(
            "Reusable phase cache root cannot be inside the one-shot run directory."
        )
    run_root: Path | None = None
    if phase_events is not None and (
        phase_events.path == output_path or output_path in phase_events.path.parents
    ):
        if phase_events.path.parent != output_path:
            raise Lfm2PreCausalResumeError(
                "A phase event log inside the run must be one direct run-root file."
            )
        run_root = _new_directory(output_path)

    capsule = load_lfm2_pre_causal_body_resume_capsule(
        body_reference,
        checkpoint_io=checkpoint_io,
    )
    model = capsule.model
    bindings = capsule.bindings
    bridge = checkpoint_io.tokenizer_bridge
    terminal = evidence.terminal_receipt
    trie_receipt = evidence.causal_trie.as_dict()
    structural_receipt = evidence.structural_interface_receipt
    if (
        bindings.parent_identity_sha256 != terminal["parent_identity_sha256"]
        or bindings.bridge_artifact_sha256 != bridge.artifact_sha256
        or bindings.bridge_artifact_sha256 != terminal["bridge_artifact_sha256"]
        or bindings.bridge_profile_sha256 != bridge.profile_fingerprint
        or bindings.target_tokenizer_semantics_sha256
        != terminal["target_tokenizer_semantics_sha256"]
        or fingerprint_hf_tokenizer_semantics(target_tokenizer)
        != bindings.target_tokenizer_semantics_sha256
        or bindings.structural_interface_install_sha256
        != structural_receipt["result_sha256"]
        or bindings.installed_input_table_sha256
        != structural_receipt["installed_input_table_sha256"]
        or bindings.installed_output_baseline_sha256
        != structural_receipt["installed_output_baseline_sha256"]
        or structural_receipt["sequence_plan_sha256"]
        != evidence.sequence_plan["plan_sha256"]
        or trie_receipt["bridge_artifact_sha256"] != bridge.artifact_sha256
        or trie_receipt["bridge_profile_sha256"] != bridge.profile_fingerprint
    ):
        raise Lfm2PreCausalResumeError(
            "Body capsule, bridge, and cached causal evidence have different lineage."
        )
    child_count = _unique_parameter_count(model)
    ratio = parent_count / child_count
    if (
        child_count != bindings.target_unique_parameters
        or ratio < minimum_ratio
        or model.get_input_embeddings().weight
        is model.get_output_embeddings().weight
        or _interface_tensor_sha256(model.get_input_embeddings().weight)
        != bindings.installed_input_table_sha256
        or _interface_tensor_sha256(model.get_output_embeddings().weight)
        != bindings.installed_output_baseline_sha256
    ):
        raise Lfm2PreCausalResumeError(
            "Resumed body is not the pinned dense untied aggressive-compaction target."
        )

    active_causal_evidence: Lfm2CausalTrieEvidence = evidence.causal_trie
    expanded_phase_cache_audit = None
    if expand_all_sealed_internal_prefixes:
        if phase_cache is None and phase_events is None:
            active_causal_evidence = expand_lfm2_causal_trie_internal_prefixes(
                active_causal_evidence
            )
        else:
            (
                active_causal_evidence,
                expanded_phase_cache_audit,
            ) = _expanded_causal_plan_with_optional_cache(
                active_causal_evidence,
                evidence=evidence,
                cache=phase_cache,
                events=phase_events,
            )
    expanded_trie_receipt = active_causal_evidence.as_dict()
    mapped_target_capsule = None
    mapped_hybrid_receipt = None
    if mapped_target_reference is not None:
        mapped_target_capsule = load_lfm2_mapped_causal_target_capsule(
            mapped_target_reference
        )
        active_causal_evidence = (
            compile_lfm2_dense_mapped_root_hybrid_causal_evidence(
                active_causal_evidence,
                mapped_target_capsule.evidence,
            )
        )
        mapped_hybrid_receipt = active_causal_evidence.as_dict()

    if run_root is None:
        run_root = _new_directory(output_directory)
    _write_json(run_root / "pinned-causal-evidence.json", evidence.as_dict())
    if expanded_trie_receipt["result_sha256"] != trie_receipt["result_sha256"]:
        _write_json(
            run_root / "expanded-causal-trie-evidence.json",
            expanded_trie_receipt,
        )
    if mapped_hybrid_receipt is not None:
        _write_json(
            run_root / "dense-mapped-root-hybrid-causal-evidence.json",
            mapped_hybrid_receipt,
        )
    child_phase_cache_audit = None
    if phase_cache is None and phase_events is None:
        child_evidence = collect_lfm2_child_causal_residual_evidence(
            model,
            active_causal_evidence,
            expected_input_table_sha256=bindings.installed_input_table_sha256,
            expected_body_inventory_sha256=bindings.causal_body_inventory_sha256,
            device=model_device,
            evidence_role="parent-free-pre-causal-body-resume",
            maximum_batch_rows=child_replay_maximum_batch_rows,
        )
    else:
        child_evidence, child_phase_cache_audit = (
            _child_causal_replay_with_optional_cache(
                model,
                active_causal_evidence,
                evidence=evidence,
                body_reference=body_reference,
                bindings=bindings,
                model_device=model_device,
                maximum_batch_rows=child_replay_maximum_batch_rows,
                cache=phase_cache,
                events=phase_events,
            )
        )
    child_evidence_receipt = child_evidence.as_dict()
    _write_json(
        run_root / "child-causal-residual-evidence.json",
        child_evidence_receipt,
    )
    feature_rows = child_evidence.hidden_rows
    gram_diagonal_mean = float(feature_rows.float().square().sum(dim=1).mean())
    if not math.isfinite(gram_diagonal_mean) or gram_diagonal_mean <= 0.0:
        raise Lfm2PreCausalResumeError(
            "Resumed child causal features have zero or nonfinite ridge scale."
        )
    raw_special_ids = getattr(target_tokenizer, "all_special_ids", ())
    protected_ids = (
        tuple(
            sorted(
                {
                    int(token_id)
                    for token_id in raw_special_ids
                    if isinstance(token_id, int)
                    and not isinstance(token_id, bool)
                    and 0 <= token_id < int(model.config.vocab_size)
                }
            )
        )
        if protect_special_tokens
        else ()
    )
    structural_table = (
        model.get_input_embeddings().weight.detach().cpu().contiguous().clone()
    )
    causal_bridge = compile_lfm2_causal_residual_token_bridge(
        structural_table,
        child_evidence,
        active_causal_evidence,
        parent_terminal_evidence=terminal,
        sequence_plan_sha256=evidence.sequence_plan["plan_sha256"],
        ridge=gram_diagonal_mean * ridge_fraction,
        residual_scale_ppm=residual_scale_ppm,
        protected_target_ids=protected_ids,
        row_norm_calibration_ppm=row_norm_calibration_ppm,
        unsupported_logit_policy=unsupported_logit_policy,
        solve_device=solve_device,
    )
    bridge_receipt = causal_bridge.as_dict()
    if bridge_receipt.get("unsupported_logit_policy") != unsupported_logit_policy:
        raise Lfm2PreCausalResumeError(
            "Compiled causal bridge did not retain the requested policy."
        )
    _write_json(run_root / "causal-output-bridge.json", bridge_receipt)
    installed, compatibility = _install_from_frozen_resume(
        model=model,
        capsule_manifest=capsule.manifest,
        structural_receipt=structural_receipt,
        causal_bridge=causal_bridge,
        child_evidence=child_evidence,
        expected_input_table_sha256=bindings.installed_input_table_sha256,
        expected_body_inventory_sha256=bindings.causal_body_inventory_sha256,
    )
    install_receipt = installed.as_dict()
    _write_json(run_root / "causal-output-install.json", install_receipt)
    _write_json(run_root / "structural-topology-compatibility.json", compatibility)

    installed_state = fingerprint_model_state(model)
    installed_runtime = fingerprint_model_runtime_state(model)
    installed_identity = fingerprint_lfm2_parent(model)
    installed_body = fingerprint_lfm2_causal_body_inventory(model)
    live_storage = _storage_addresses(model)
    checkpoint = run_root / "candidates" / candidate_id / "checkpoint"
    checkpoint.mkdir(parents=True)
    checkpoint_io.save(model, checkpoint)
    checkpoint_inventory = _checkpoint_inventory(checkpoint)
    checkpoint_inventory_sha256 = canonical_sha256(checkpoint_inventory)
    reloaded = checkpoint_io.load(checkpoint)
    reload_state = fingerprint_model_state(reloaded)
    reload_runtime = fingerprint_model_runtime_state(reloaded)
    reload_identity = fingerprint_lfm2_parent(reloaded)
    reload_body = fingerprint_lfm2_causal_body_inventory(reloaded)
    if (
        reload_state != installed_state
        or reload_identity.as_dict() != installed_identity.as_dict()
        or reload_body != installed_body
        or reload_body != bindings.causal_body_inventory_sha256
        or _unique_parameter_count(reloaded) != child_count
        or reloaded.get_input_embeddings().weight
        is reloaded.get_output_embeddings().weight
        or not _storage_addresses(reloaded).isdisjoint(live_storage)
        or any(parameter.requires_grad for parameter in reloaded.parameters())
        or any(parameter.grad is not None for parameter in reloaded.parameters())
        or any(module.training for module in reloaded.modules())
        or _interface_tensor_sha256(reloaded.get_input_embeddings().weight)
        != bindings.installed_input_table_sha256
        or _interface_tensor_sha256(reloaded.get_output_embeddings().weight)
        != install_receipt["installed_output_table_sha256"]
    ):
        raise Lfm2PreCausalResumeError(
            "Fresh standalone reload changed child state, body, interface, or policy."
        )

    review_state_before = fingerprint_model_state(reloaded)
    responses = generate_review_rows(
        reloaded,
        target_tokenizer,
        prompts,
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
    )
    if (
        len(responses) != len(prompts)
        or fingerprint_model_state(reloaded) != review_state_before
    ):
        raise Lfm2PreCausalResumeError(
            "Standalone manual-review generation was incomplete or mutated the child."
        )
    prompt_set_sha256 = canonical_sha256(list(prompts))
    dossier_body = {
        "schema": LFM2_PRE_CAUSAL_RESUME_DOSSIER_SCHEMA,
        "status": "standalone-candidate-ready-for-manual-intelligence-review",
        "candidate_id": candidate_id,
        "candidate_model_state_sha256": reload_state,
        "prompt_set_sha256": prompt_set_sha256,
        "response_count": len(responses),
        "responses": responses,
        "complete_decoded_responses_preserved": True,
        "decoded_whole_responses_are_primary_evidence": True,
        "numeric_diagnostics_are_advisory_only": True,
        "automatic_intelligence_verdict_performed": False,
        "manual_intelligence_review_required": True,
        "intelligence_success_claimed": False,
    }
    dossier = _sealed(dossier_body, "dossier_sha256")
    dossier_path = run_root / "whole-response-dossier.json"
    _write_json(dossier_path, dossier)

    replay_forward_calls = child_evidence_receipt["operation_audit"][
        "child_forward_calls"
    ]
    replay_cache_hit = bool(
        child_phase_cache_audit is not None
        and child_phase_cache_audit["cache_hit"] is True
    )
    operation_audit = {
        "parent_model_objects_constructed": 0,
        "parent_model_loads": 0,
        "parent_weight_tensor_loads": 0,
        "parent_model_forward_calls": 0,
        "parent_model_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "semantic_parent_cache_sessions": 0,
        "typed_causal_trie_restores_from_sealed_receipt": 1,
        "child_body_checkpoint_loads": 1,
        "child_causal_prefix_forward_calls": (
            0 if replay_cache_hit else replay_forward_calls
        ),
        "child_standalone_reload_materializations": 1,
        "child_generate_requests_for_manual_review": len(responses),
        "framework_internal_generation_forward_count_claimed": False,
        "training_steps": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "parameters_sparsified": 0,
        "parameter_sharing_used": False,
        "parent_runtime_sidecar_used": False,
        "network_access": False,
    }
    if mapped_target_capsule is not None:
        operation_audit["mapped_target_capsule_loads"] = 1
        operation_audit["mapped_target_parent_cache_sessions"] = 0
        operation_audit["mapped_target_parent_model_operations"] = 0
    if phase_cache is not None:
        operation_audit.update(
            {
                "phase_cache_enabled": True,
                "expanded_causal_plan_cache_hit": bool(
                    expanded_phase_cache_audit is not None
                    and expanded_phase_cache_audit["cache_hit"] is True
                ),
                "child_causal_replay_cache_hit": replay_cache_hit,
                "child_causal_prefix_forward_calls_avoided": (
                    replay_forward_calls if replay_cache_hit else 0
                ),
            }
        )
    body = {
        "schema": LFM2_PRE_CAUSAL_RESUME_RUN_SCHEMA,
        "status": "standalone-parent-free-causal-resume-ready-for-manual-review",
        "run_directory": str(run_root),
        "candidate_id": candidate_id,
        "candidate_checkpoint": str(checkpoint),
        "parent_identity_sha256": bindings.parent_identity_sha256,
        "parent_unique_parameters": parent_count,
        "parent_unique_parameter_count_basis": (
            "immutable-neural-forge-lfm2-exact4x-parent-contract"
        ),
        "parent_unique_parameter_count_recomputed_in_this_resume": False,
        "child_unique_parameters": child_count,
        "parameter_compaction_ratio": ratio,
        "minimum_parameter_compaction_ratio": minimum_ratio,
        "aggressive_parameter_compaction_preserved": True,
        "dense_child": True,
        "quantization_used": False,
        "pruning_used": False,
        "body_capsule_reference": body_reference.as_dict(),
        "body_capsule_bindings_sha256": bindings.bindings_sha256,
        "pinned_causal_evidence": evidence.as_dict(),
        "target_tokenizer_semantics_sha256": (
            bindings.target_tokenizer_semantics_sha256
        ),
        "live_target_tokenizer_semantics_authenticated": True,
        "unsupported_logit_policy": unsupported_logit_policy,
        "ridge_fraction": ridge_fraction,
        "analytic_ridge": gram_diagonal_mean * ridge_fraction,
        "residual_scale_ppm": residual_scale_ppm,
        "row_norm_calibration_ppm": row_norm_calibration_ppm,
        "expand_all_sealed_internal_prefixes": (
            expand_all_sealed_internal_prefixes
        ),
        "expanded_causal_trie_evidence_sha256": expanded_trie_receipt[
            "result_sha256"
        ],
        "sealed_internal_prefix_row_count": expanded_trie_receipt[
            "internal_prefix_row_count"
        ],
        "sealed_internal_prefix_omission_count": expanded_trie_receipt[
            "omitted_internal_prefix_count"
        ],
        "child_replay_maximum_batch_rows": child_replay_maximum_batch_rows,
        "protected_target_ids": list(protected_ids),
        "child_causal_residual_evidence_sha256": child_evidence_receipt[
            "evidence_sha256"
        ],
        "causal_residual_bridge_sha256": bridge_receipt["result_sha256"],
        "causal_output_install_sha256": install_receipt["result_sha256"],
        "structural_topology_compatibility_sha256": compatibility[
            "result_sha256"
        ],
        "installed_model_state_sha256": installed_state,
        "installed_runtime_state_sha256": installed_runtime,
        "fresh_reload_model_state_sha256": reload_state,
        "fresh_reload_runtime_state_sha256": reload_runtime,
        "fresh_reload_lfm2_identity_sha256": reload_identity.identity_sha256,
        "fresh_reload_causal_body_inventory_sha256": reload_body,
        "checkpoint_inventory": checkpoint_inventory,
        "checkpoint_inventory_sha256": checkpoint_inventory_sha256,
        "standalone_fresh_reload_verified": True,
        "parent_required_for_runtime": False,
        "parent_sidecar_parameters": 0,
        "runtime_shared_parent_parameters": 0,
        "dossier_path": str(dossier_path),
        "dossier_sha256": dossier["dossier_sha256"],
        "decoded_response_count": len(responses),
        "decoded_whole_responses_retained": True,
        "manual_intelligence_review_required": True,
        "automatic_intelligence_verdict_performed": False,
        "numeric_diagnostics_have_acceptance_authority": False,
        "intelligence_success_claimed": False,
        "operation_audit": operation_audit,
    }
    if mapped_target_capsule is not None and mapped_hybrid_receipt is not None:
        body.update(
            {
                "mapped_target_capsule_reference": (
                    mapped_target_reference.as_dict()
                ),
                "mapped_target_evidence_sha256": (
                    mapped_target_capsule.evidence.receipt["evidence_sha256"]
                ),
                "active_causal_evidence_mode": (
                    "dense-strict-mapped-roots-with-contextual-trie-overrides"
                ),
                "active_causal_evidence_sha256": mapped_hybrid_receipt[
                    "result_sha256"
                ],
                "all_root_target_cells_supported_before_child_solve": True,
                "mapped_target_capsule_loaded_without_parent_cache": True,
            }
        )
    if phase_cache is not None:
        body["phase_cache"] = {
            "enabled": True,
            "expanded_causal_plan": expanded_phase_cache_audit,
            "child_causal_replay": child_phase_cache_audit,
            "silent_fallback_performed": False,
        }
    if phase_events is not None:
        phase_event_records = phase_events.read()
        body["phase_event_log"] = {
            "enabled": True,
            "path": str(phase_events.path),
            "run_id": phase_events.run_id,
            "event_count": len(phase_event_records),
            "head_event_sha256": (
                None
                if not phase_event_records
                else phase_event_records[-1].event_sha256
            ),
        }
    result = _sealed(body, "result_sha256")
    _write_json(run_root / "run-receipt.json", result)
    return result


__all__ = [
    "LFM2_PINNED_CAUSAL_RESUME_EVIDENCE_SCHEMA",
    "LFM2_PRE_CAUSAL_RESUME_DOSSIER_SCHEMA",
    "LFM2_PRE_CAUSAL_RESUME_RUN_SCHEMA",
    "Lfm2PinnedCausalResumeEvidence",
    "Lfm2PreCausalResumeError",
    "restore_lfm2_causal_trie_evidence_from_receipt",
    "resume_lfm2_pre_causal_body",
]
