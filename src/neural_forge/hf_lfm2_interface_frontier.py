"""Automatic no-training LFM2 token-interface folding frontier.

The strict tokenizer bridge proves that every target row has one donor anchor,
but that anchor is not the whole tokenization story.  A retained target token
can correspond to different donor tokens (or donor-token sequences) at lexical
boundaries.  This module deterministically recovers that context envelope from
the two sealed local tokenizers and folds it into dense target input/output
operators.  It also supports a compiler-owned untied output blank so input
composition and next-token decoding can use different analytic maps.

No parent forward, training, autograd, optimizer, pruning, quantization,
sparsity, network access, parent sidecar, or runtime parent sharing is used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.hf_lfm2_bridge_only_control import (
    _all_named_tensors,
    _body_records,
    _config_topology,
    _storage_identity,
    _tensor_sha256,
    _unique_parameter_count,
)
from neural_forge.hf_tokenizer_bridge import StrictTokenizerBridgeArtifact


LFM2_INTERFACE_FRONTIER_SCHEMA = "neural-forge-lfm2-interface-frontier-v1"
LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA = (
    "neural-forge-lfm2-interface-context-sequence-plan-v1"
)
LFM2_INTERFACE_INVERSE_OUTPUT_PLAN_SCHEMA = (
    "neural-forge-lfm2-interface-inverse-output-plan-v2"
)
LFM2_CACHED_LOGIT_RESIDUAL_HEAD_SCHEMA = (
    "neural-forge-lfm2-cached-logit-residual-head-v1"
)
LFM2_CAUSAL_RESIDUAL_TOKEN_BRIDGE_SCHEMA = (
    "neural-forge-lfm2-causal-residual-token-bridge-v2"
)
LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-causal-token-trie-evidence-v2"
)
LFM2_CAUSAL_CONTEXT_TRANSITION_PLAN_SCHEMA = (
    "neural-forge-lfm2-causal-context-transition-plan-v2"
)
LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-interface-terminal-cache-evidence-v1"
)
LFM2_CHILD_CAUSAL_RESIDUAL_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-child-causal-residual-evidence-v2"
)
LFM2_UNTIED_INTERFACE_BODY_REBALANCE_SCHEMA = (
    "neural-forge-lfm2-untied-interface-body-rebalance-v1"
)
DEFAULT_INTERFACE_CONTEXTS = ("", " ", "A", "The ", "\n", "0")


class Lfm2InterfaceFrontierError(RuntimeError):
    """An automatic interface candidate could not be authenticated."""


@dataclass(frozen=True, slots=True)
class Lfm2InterfaceRecipe:
    """One deterministic dense interface transport recipe."""

    candidate_id: str
    input_strict_anchor_ppm: int
    input_preserve_anchor_norm: bool
    output_mode: str
    output_strict_anchor_ppm: int = 0
    output_preserve_anchor_norm: bool = True
    output_bucket_weighting: str = "uniform"

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise Lfm2InterfaceFrontierError("candidate_id must be nonempty text.")
        object.__setattr__(self, "candidate_id", self.candidate_id.strip())
        for name in ("input_strict_anchor_ppm", "output_strict_anchor_ppm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 1_000_000:
                raise Lfm2InterfaceFrontierError(
                    f"{name} must be an integer from zero through one million."
                )
        for name in (
            "input_preserve_anchor_norm",
            "output_preserve_anchor_norm",
        ):
            if not isinstance(getattr(self, name), bool):
                raise Lfm2InterfaceFrontierError(f"{name} must be boolean.")
        if self.output_mode not in {
            "tied-to-input",
            "strict-row",
            "context-envelope",
            "inverse-bucket-centroid",
            "prefix-transition-centroid",
            "cached-logit-residual",
            "causal-residual-token-bridge",
            "causal-residual-token-bridge-tied",
        }:
            raise Lfm2InterfaceFrontierError("Unsupported output_mode.")
        if self.output_bucket_weighting not in {"uniform", "row-norm"}:
            raise Lfm2InterfaceFrontierError(
                "output_bucket_weighting must be uniform or row-norm."
            )

    @property
    def untied_output(self) -> bool:
        return self.output_mode not in {
            "tied-to-input",
            "causal-residual-token-bridge-tied",
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "input_transport": "context-envelope-sequence-mean",
            "input_strict_anchor_ppm": self.input_strict_anchor_ppm,
            "input_preserve_anchor_norm": self.input_preserve_anchor_norm,
            "output_mode": self.output_mode,
            "output_strict_anchor_ppm": self.output_strict_anchor_ppm,
            "output_preserve_anchor_norm": self.output_preserve_anchor_norm,
            "output_bucket_weighting": self.output_bucket_weighting,
            "untied_output": self.untied_output,
        }


DEFAULT_INTERFACE_RECIPES = (
    Lfm2InterfaceRecipe(
        candidate_id="context-mean-norm-tied",
        input_strict_anchor_ppm=0,
        input_preserve_anchor_norm=True,
        output_mode="tied-to-input",
    ),
    Lfm2InterfaceRecipe(
        candidate_id="context-anchor500-norm-tied",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="tied-to-input",
    ),
    Lfm2InterfaceRecipe(
        candidate_id="context-anchor500-input-strict-output-untied",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="strict-row",
        output_strict_anchor_ppm=1_000_000,
        output_preserve_anchor_norm=True,
    ),
    Lfm2InterfaceRecipe(
        candidate_id="context-mean-input-anchor500-output-untied",
        input_strict_anchor_ppm=0,
        input_preserve_anchor_norm=True,
        output_mode="context-envelope",
        output_strict_anchor_ppm=500_000,
        output_preserve_anchor_norm=True,
    ),
    Lfm2InterfaceRecipe(
        candidate_id="context-anchor500-input-inverse-bucket-output-untied",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="inverse-bucket-centroid",
        output_strict_anchor_ppm=250_000,
        output_preserve_anchor_norm=True,
        output_bucket_weighting="row-norm",
    ),
)


@dataclass(frozen=True, slots=True)
class Lfm2InterfaceFrontierResult:
    """One installed standalone candidate and its sealed structural receipt."""

    model: Any
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.model is None:
            raise TypeError("model must be a materialized LFM2 child.")
        body = dict(self.receipt)
        supplied = body.pop("result_sha256", None)
        if supplied != canonical_sha256(body):
            raise Lfm2InterfaceFrontierError(
                "Interface-frontier receipt seal is invalid."
            )
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.receipt)


@dataclass(frozen=True, slots=True)
class Lfm2CachedLogitResidualHead:
    """One dense output table compiled analytically from sealed cache rows."""

    table: Any
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        if (
            not isinstance(self.table, torch.Tensor)
            or self.table.ndim != 2
            or not torch.is_floating_point(self.table)
            or torch.is_complex(self.table)
            or self.table.layout != torch.strided
            or not bool(torch.isfinite(self.table).all())
        ):
            raise Lfm2InterfaceFrontierError(
                "Cached-logit residual head must be one finite dense matrix."
            )
        body = dict(self.receipt)
        supplied = body.pop("result_sha256", None)
        if (
            supplied != canonical_sha256(body)
            or body.get("schema") != LFM2_CACHED_LOGIT_RESIDUAL_HEAD_SCHEMA
            or body.get("output_table_sha256") != _tensor_sha256(self.table)
        ):
            raise Lfm2InterfaceFrontierError(
                "Cached-logit residual-head receipt seal is invalid."
            )
        object.__setattr__(self, "table", self.table.detach().cpu().contiguous())
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.receipt)


@dataclass(frozen=True, slots=True)
class Lfm2TerminalCacheEvidence:
    """Typed cache-only parent terminal rows and exact causal root ancestry."""

    hidden_rows: Any
    donor_logits: Any
    causal_root_records: tuple[Mapping[str, Any], ...]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        if any(
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or not torch.is_floating_point(value)
            or torch.is_complex(value)
            or value.layout != torch.strided
            or value.device.type == "meta"
            or value.requires_grad
            or not bool(torch.isfinite(value).all())
            for value in (self.hidden_rows, self.donor_logits)
        ):
            raise Lfm2InterfaceFrontierError(
                "Terminal cache evidence must contain detached finite matrices."
            )
        if (
            int(self.hidden_rows.shape[0]) <= 0
            or int(self.hidden_rows.shape[1]) <= 0
            or int(self.donor_logits.shape[1]) <= 0
            or int(self.hidden_rows.shape[0]) != int(self.donor_logits.shape[0])
        ):
            raise Lfm2InterfaceFrontierError(
                "Terminal hidden rows and donor logits are not row-aligned."
            )
        value = dict(self.receipt)
        body = dict(value)
        supplied = body.pop("evidence_sha256", None)
        target_vocabulary_size = value.get("target_vocabulary_size")
        if (
            isinstance(target_vocabulary_size, bool)
            or not isinstance(target_vocabulary_size, int)
            or target_vocabulary_size <= 0
        ):
            raise Lfm2InterfaceFrontierError(
                "Terminal cache evidence lacks its target vocabulary capacity."
            )
        canonical_roots = _causal_root_evidence_rows(
            self.causal_root_records,
            donor_row_count=int(self.donor_logits.shape[1]),
            target_row_count=target_vocabulary_size,
            terminal_row_count=int(self.donor_logits.shape[0]),
        )
        audit = value.get("operation_audit")
        selected_batches = value.get("selected_batches")
        ancestry_fields = (
            "cache_key",
            "parent_identity_sha256",
            "selective_session_sha256",
            "manifest_sha256",
            "completion_sha256",
            "bridge_artifact_sha256",
            "bridge_profile_sha256",
            "donor_tokenizer_semantics_sha256",
            "target_tokenizer_semantics_sha256",
            "semantic_alignment_sha256",
            "tokenizer_live_session_sha256",
        )
        ancestry_valid = True
        for field in ancestry_fields:
            try:
                _digest_text(value.get(field), field)
            except Lfm2InterfaceFrontierError:
                ancestry_valid = False
                break
        if (
            supplied != canonical_sha256(body)
            or value.get("schema") != LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA
            or value.get("status")
            != "complete-authenticated-terminal-cache-selection"
            or value.get("terminal_row_count") != int(self.hidden_rows.shape[0])
            or value.get("hidden_size") != int(self.hidden_rows.shape[1])
            or value.get("donor_vocabulary_size")
            != int(self.donor_logits.shape[1])
            or value.get("hidden_rows_sha256") != _tensor_sha256(self.hidden_rows)
            or value.get("donor_logits_sha256") != _tensor_sha256(self.donor_logits)
            or value.get("causal_root_row_count") != len(canonical_roots)
            or value.get("causal_root_rows_sha256")
            != canonical_sha256(list(canonical_roots))
            or not isinstance(selected_batches, list)
            or value.get("batch_count") != len(selected_batches)
            or not selected_batches
            or any(not isinstance(row, Mapping) for row in selected_batches)
            or sum(int(row.get("terminal_rows", -1)) for row in selected_batches)
            != int(self.hidden_rows.shape[0])
            or value.get("sealed_parent_cache_reused_without_recapture") is not True
            or value.get("all_terminal_rows_retained_in_diagnostic_tensors")
            is not True
            or value.get(
                "all_eligible_exact_boundary_rows_retained_for_causal_replay"
            )
            is not True
            or value.get("unpaired_cached_prefill_logits_excluded") is not True
            or not ancestry_valid
            or not isinstance(audit, Mapping)
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
            raise Lfm2InterfaceFrontierError(
                "Terminal cache evidence receipt, tensors, roots, or ancestry differ."
            )
        object.__setattr__(
            self,
            "hidden_rows",
            self.hidden_rows.detach().cpu().contiguous(),
        )
        object.__setattr__(
            self,
            "donor_logits",
            self.donor_logits.detach().cpu().contiguous(),
        )
        object.__setattr__(
            self,
            "causal_root_records",
            tuple(MappingProxyType(dict(record)) for record in canonical_roots),
        )
        object.__setattr__(self, "receipt", MappingProxyType(value))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.receipt)


@dataclass(frozen=True, slots=True)
class Lfm2CausalResidualTokenBridge:
    """Dense token bridge compiled against causal child residual features."""

    table: Any
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        if (
            not isinstance(self.table, torch.Tensor)
            or self.table.ndim != 2
            or not torch.is_floating_point(self.table)
            or torch.is_complex(self.table)
            or self.table.layout != torch.strided
            or not bool(torch.isfinite(self.table).all())
        ):
            raise Lfm2InterfaceFrontierError(
                "Causal residual token bridge must be one finite dense matrix."
            )
        body = dict(self.receipt)
        supplied = body.pop("result_sha256", None)
        if (
            supplied != canonical_sha256(body)
            or body.get("schema") != LFM2_CAUSAL_RESIDUAL_TOKEN_BRIDGE_SCHEMA
            or body.get("output_table_sha256") != _tensor_sha256(self.table)
        ):
            raise Lfm2InterfaceFrontierError(
                "Causal residual token-bridge receipt seal is invalid."
            )
        object.__setattr__(self, "table", self.table.detach().cpu().contiguous())
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.receipt)


@dataclass(frozen=True, slots=True)
class Lfm2CausalTrieEvidence:
    """Authenticated root and internal-prefix target conditionals."""

    target_prefix_token_ids: tuple[tuple[int, ...], ...]
    target_logits: Any
    target_logit_support_mask: Any
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        prefixes = tuple(tuple(value) for value in self.target_prefix_token_ids)
        if (
            not prefixes
            or any(
                not value
                or any(
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    for token_id in value
                )
                for value in prefixes
            )
            or not isinstance(self.target_logits, torch.Tensor)
            or self.target_logits.ndim != 2
            or not torch.is_floating_point(self.target_logits)
            or not bool(torch.isfinite(self.target_logits).all())
            or not isinstance(self.target_logit_support_mask, torch.Tensor)
            or self.target_logit_support_mask.dtype != torch.bool
            or tuple(self.target_logit_support_mask.shape)
            != tuple(self.target_logits.shape)
            or int(self.target_logits.shape[0]) != len(prefixes)
        ):
            raise Lfm2InterfaceFrontierError(
                "Causal trie evidence tensors or target prefixes are invalid."
            )
        body = dict(self.receipt)
        supplied = body.pop("result_sha256", None)
        transition_plan = body.get("contextual_transition_plan")
        transition_plan_valid = isinstance(transition_plan, Mapping)
        root_plans_valid = False
        if transition_plan_valid:
            transition_body = dict(transition_plan)
            transition_sha256 = transition_body.pop("plan_sha256", None)
            root_plans = transition_plan.get("root_transition_plans")
            transition_plan_valid = bool(
                transition_sha256 == canonical_sha256(transition_body)
                and transition_sha256
                == body.get("contextual_transition_plan_sha256")
                and transition_plan.get("schema")
                == LFM2_CAUSAL_CONTEXT_TRANSITION_PLAN_SCHEMA
                and isinstance(root_plans, list)
                and root_plans
            )
            if transition_plan_valid:
                root_plans_valid = all(
                    isinstance(root_plan, Mapping)
                    and root_plan.get("root_transition_plan_sha256")
                    == canonical_sha256(
                        {
                            key: value
                            for key, value in root_plan.items()
                            if key != "root_transition_plan_sha256"
                        }
                    )
                    for root_plan in root_plans
                )
        if (
            supplied != canonical_sha256(body)
            or body.get("schema") != LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA
            or not transition_plan_valid
            or not root_plans_valid
            or body.get("target_prefix_rows_sha256")
            != canonical_sha256([list(value) for value in prefixes])
            or body.get("target_logits_sha256")
            != _tensor_sha256(self.target_logits)
            or body.get("target_logit_support_mask_sha256")
            != _tensor_sha256(self.target_logit_support_mask)
        ):
            raise Lfm2InterfaceFrontierError(
                "Causal trie evidence receipt seal is invalid."
            )
        object.__setattr__(self, "target_prefix_token_ids", prefixes)
        object.__setattr__(
            self,
            "target_logits",
            self.target_logits.detach().cpu().contiguous(),
        )
        object.__setattr__(
            self,
            "target_logit_support_mask",
            self.target_logit_support_mask.detach().cpu().contiguous(),
        )
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.receipt)


@dataclass(frozen=True, slots=True)
class Lfm2ChildCausalResidualEvidence:
    """Typed child-coordinate residual rows and their replay ancestry."""

    hidden_rows: Any
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        torch = _torch()
        if (
            not isinstance(self.hidden_rows, torch.Tensor)
            or self.hidden_rows.ndim != 2
            or not torch.is_floating_point(self.hidden_rows)
            or torch.is_complex(self.hidden_rows)
            or self.hidden_rows.layout != torch.strided
            or self.hidden_rows.device.type == "meta"
            or self.hidden_rows.requires_grad
            or not bool(torch.isfinite(self.hidden_rows).all())
        ):
            raise Lfm2InterfaceFrontierError(
                "Child causal residual evidence must be one detached finite matrix."
            )
        body = dict(self.receipt)
        supplied = body.pop("evidence_sha256", None)
        audit = body.get("operation_audit")
        if (
            supplied != canonical_sha256(body)
            or body.get("schema")
            != LFM2_CHILD_CAUSAL_RESIDUAL_EVIDENCE_SCHEMA
            or body.get("child_hidden_rows_sha256")
            != _tensor_sha256(self.hidden_rows)
            or body.get("row_count") != int(self.hidden_rows.shape[0])
            or body.get("hidden_size") != int(self.hidden_rows.shape[1])
            or body.get("child_model_state_before_sha256")
            != body.get("child_model_state_after_sha256")
            or body.get("child_state_preserved") is not True
            or body.get("causal_input_table_after_sha256")
            != body.get("causal_input_table_sha256")
            or body.get("causal_body_inventory_after_sha256")
            != body.get("causal_body_inventory_sha256")
            or body.get("input_and_body_coordinate_preserved") is not True
            or body.get("batch_size") != 1
            or body.get("full_prefix_use_cache_false") is not True
            or body.get("position_origin") != 0
            or not isinstance(audit, Mapping)
            or audit.get("child_forward_calls")
            != int(self.hidden_rows.shape[0])
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
                    "child_generate_calls",
                )
            )
        ):
            raise Lfm2InterfaceFrontierError(
                "Child causal residual evidence receipt or tensor seal is invalid."
            )
        object.__setattr__(
            self,
            "hidden_rows",
            self.hidden_rows.detach().cpu().contiguous(),
        )
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.receipt)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime boundary
        raise RuntimeError("LFM2 interface folding requires PyTorch.") from exc
    return torch


def _body_topology(topology: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(topology)
    result.pop("vocabulary_size", None)
    result.pop("tie_word_embeddings", None)
    return result


def _verify_base_blank(receipt: Mapping[str, Any]) -> str:
    body = dict(receipt)
    supplied = body.pop("report_sha256", None)
    if (
        supplied != canonical_sha256(body)
        or receipt.get("compiler_created_blank") is not True
        or receipt.get("all_parameters_zero") is not True
        or receipt.get("dense_materialized_parameters") is not True
        or receipt.get("training_used") is not False
        or receipt.get("pruning_used") is not False
        or receipt.get("quantization_used") is not False
        or receipt.get("weights_installed") is not False
    ):
        raise Lfm2InterfaceFrontierError(
            "Target is not an authenticated compiler-created dense zero blank."
        )
    assert isinstance(supplied, str)
    return supplied


def _interface_storage_ids(model: Any) -> set[tuple[int, int, tuple[int, ...]]]:
    input_weight = getattr(model.get_input_embeddings(), "weight", None)
    output_weight = getattr(model.get_output_embeddings(), "weight", None)
    if input_weight is None or output_weight is None:
        raise Lfm2InterfaceFrontierError(
            "LFM2 model lacks materialized input/output weights."
        )
    return {_storage_identity(input_weight), _storage_identity(output_weight)}


def fingerprint_lfm2_causal_body_inventory(model: Any) -> str:
    """Fingerprint the collector's exact body coordinate without mutation."""

    return canonical_sha256(
        list(
            _body_records(
                model,
                interface_storage_ids=_interface_storage_ids(model),
            )
        )
    )


def _terminal_phase_pairs(reference: Any) -> tuple[tuple[str, str, str], ...]:
    names = set(reference.ordered_tensor_names)
    suffix = ".output.lm_head_input"
    pairs = []
    for hidden_name in sorted(name for name in names if name.endswith(suffix)):
        phase = hidden_name[: -len(suffix)]
        logits_name = f"{phase}.logits"
        if logits_name not in names:
            raise Lfm2InterfaceFrontierError(
                f"Cache phase {phase!r} lacks matching full logits."
            )
        pairs.append((phase, hidden_name, logits_name))
    if not pairs:
        raise Lfm2InterfaceFrontierError(
            "Cache batch contains no aligned terminal feature/logit rows."
        )
    return tuple(pairs)


def collect_lfm2_terminal_cache_evidence(
    cache: Any,
    *,
    expected_cache_key: str,
    expected_parent_identity_sha256: str,
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
) -> Lfm2TerminalCacheEvidence:
    """Read and seal reusable terminal evidence without executing the parent."""

    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    expected_cache_key = _digest_text(expected_cache_key, "expected_cache_key")
    expected_parent_identity_sha256 = _digest_text(
        expected_parent_identity_sha256,
        "expected_parent_identity_sha256",
    )
    if (
        cache.cache_key != expected_cache_key
        or cache.parent_fingerprint != expected_parent_identity_sha256
    ):
        raise Lfm2InterfaceFrontierError(
            "Trace cache ancestry differs from its external pins."
        )
    # Kept lazy so ordinary interface folding does not import the semantic
    # postfold implementation unless authenticated cache extraction is used.
    from neural_forge.hf_lfm2_semantic_postfold import (
        _semantic_alignments_from_session,
    )

    torch = _torch()
    hidden_parts = []
    logits_parts = []
    causal_roots = []
    selected_batches = []
    with cache.selective_authenticated_session() as session:
        semantic_alignments, tokenizer_session_receipt = (
            _semantic_alignments_from_session(
                cache=cache,
                session=session,
                bridge=bridge,
                donor_tokenizer=donor_tokenizer,
                target_tokenizer=target_tokenizer,
            )
        )
        alignments_by_batch = {
            alignment.source_batch_index: alignment
            for alignment in semantic_alignments
        }
        if len(alignments_by_batch) != len(session.batch_references):
            raise Lfm2InterfaceFrontierError(
                "Semantic target-prefix alignments do not cover every cache batch."
            )
        terminal_row_cursor = 0
        for reference in session.batch_references:
            pairs = _terminal_phase_pairs(reference)
            alignment = alignments_by_batch.get(reference.index)
            if alignment is None:
                raise Lfm2InterfaceFrontierError(
                    f"Cache batch {reference.index} has no semantic alignment."
                )
            names = tuple(
                sorted(
                    name
                    for _phase, hidden_name, logits_name in pairs
                    for name in (hidden_name, logits_name)
                )
            )
            payload = session.load_selected_tensors(reference.index, names)
            try:
                batch_rows = 0
                phases = []
                for phase, hidden_name, logits_name in pairs:
                    hidden = payload.tensors[hidden_name]
                    logits = payload.tensors[logits_name]
                    if (
                        hidden.ndim != 3
                        or logits.ndim != 3
                        or tuple(hidden.shape[:-1]) != tuple(logits.shape[:-1])
                        or int(hidden.shape[0]) != 1
                        or int(hidden.shape[-1]) <= 0
                        or int(logits.shape[-1]) <= 0
                    ):
                        raise Lfm2InterfaceFrontierError(
                            f"Terminal phase {phase!r} is not one-example "
                            "row-aligned."
                        )
                    rows = int(hidden.shape[1])
                    hidden_parts.append(
                        hidden.detach().reshape(rows, int(hidden.shape[-1])).clone()
                    )
                    logits_parts.append(
                        logits.detach().reshape(rows, int(logits.shape[-1])).clone()
                    )
                    phase_alignment = alignment.get_phase(phase)
                    if rows != phase_alignment.parent_sampled_suffix_tokens:
                        raise Lfm2InterfaceFrontierError(
                            f"Terminal phase {phase!r} sequence length differs "
                            "from its authenticated sampled suffix."
                        )
                    if phase_alignment.target_position_origin != 0:
                        raise Lfm2InterfaceFrontierError(
                            "Causal replay requires the authenticated unpadded "
                            "zero position origin."
                        )
                    donor_tokens = phase_alignment.donor_full_prefix_tokens
                    if rows > donor_tokens:
                        raise Lfm2InterfaceFrontierError(
                            f"Terminal phase {phase!r} samples more rows than its "
                            "authenticated donor prefix."
                        )
                    donor_start = donor_tokens - rows
                    missing_exact_boundaries = 0
                    for row_index, donor_position in enumerate(
                        range(donor_start, donor_tokens)
                    ):
                        target_positions = tuple(
                            target_position
                            for target_position, observed_donor_position in enumerate(
                                phase_alignment.target_to_donor_boundary_positions
                            )
                            if observed_donor_position == donor_position
                        )
                        if len(target_positions) != 1:
                            missing_exact_boundaries += 1
                            continue
                        target_position = target_positions[0]
                        target_prefix = tuple(
                            phase_alignment.target_full_prefix_token_ids[
                                : target_position + 1
                            ]
                        )
                        donor_prefix = tuple(
                            phase_alignment.donor_full_prefix_token_ids[
                                : donor_position + 1
                            ]
                        )
                        causal_roots.append(
                            {
                                "terminal_row_index": (
                                    terminal_row_cursor + row_index
                                ),
                                "batch_index": reference.index,
                                "phase": phase,
                                "phase_row_index": row_index,
                                "donor_boundary_position": donor_position,
                                "target_boundary_position": target_position,
                                "donor_prefix_token_ids": donor_prefix,
                                "target_prefix_token_ids": target_prefix,
                                "semantic_alignment_sha256": (
                                    alignment.alignment_sha256
                                ),
                            }
                        )
                    batch_rows += rows
                    terminal_row_cursor += rows
                    phases.append(
                        {
                            "phase": phase,
                            "rows": rows,
                            "hidden_name": hidden_name,
                            "logits_name": logits_name,
                            "causal_root_rows": rows - missing_exact_boundaries,
                            "rows_without_exact_target_boundary": (
                                missing_exact_boundaries
                            ),
                        }
                    )
                selected_batches.append(
                    {
                        "batch_index": reference.index,
                        "input_sha256": reference.input_sha256,
                        "selection_sha256": payload.selection_sha256,
                        "logical_bytes": payload.logical_bytes,
                        "terminal_rows": batch_rows,
                        "phases": phases,
                    }
                )
            finally:
                session.release_selected_tensors(payload)
        session.verify()
        session_receipt = session.as_dict()
    if not hidden_parts or not logits_parts:
        raise Lfm2InterfaceFrontierError(
            "Authenticated cache exposes no terminal evidence rows."
        )
    hidden_rows = torch.cat(hidden_parts, dim=0).contiguous()
    donor_logits = torch.cat(logits_parts, dim=0).contiguous()
    if (
        int(hidden_rows.shape[0]) != int(donor_logits.shape[0])
        or int(hidden_rows.shape[0])
        != sum(int(row["terminal_rows"]) for row in selected_batches)
        or not bool(torch.isfinite(hidden_rows).all())
        or not bool(torch.isfinite(donor_logits).all())
    ):
        raise Lfm2InterfaceFrontierError(
            "Selected terminal evidence is incomplete or nonfinite."
        )
    canonical_causal_roots = _causal_root_evidence_rows(
        causal_roots,
        donor_row_count=int(donor_logits.shape[1]),
        target_row_count=bridge.transport_profile.target_row_capacity,
        terminal_row_count=int(donor_logits.shape[0]),
    )
    body = {
        "schema": LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA,
        "status": "complete-authenticated-terminal-cache-selection",
        "cache_key": expected_cache_key,
        "parent_identity_sha256": expected_parent_identity_sha256,
        "selective_session_sha256": session_receipt["session_sha256"],
        "manifest_sha256": session_receipt["manifest_sha256"],
        "completion_sha256": session_receipt["completion_sha256"],
        "batch_count": len(selected_batches),
        "selected_batches": selected_batches,
        "terminal_row_count": int(hidden_rows.shape[0]),
        "hidden_size": int(hidden_rows.shape[1]),
        "donor_vocabulary_size": int(donor_logits.shape[1]),
        "target_vocabulary_size": bridge.transport_profile.target_row_capacity,
        "hidden_rows_sha256": _tensor_sha256(hidden_rows),
        "donor_logits_sha256": _tensor_sha256(donor_logits),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "donor_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "semantic_alignment_sha256": canonical_sha256(
            [alignment.to_evidence_dict() for alignment in semantic_alignments]
        ),
        "semantic_alignment_count": len(semantic_alignments),
        "tokenizer_live_session_sha256": canonical_sha256(
            dict(tokenizer_session_receipt)
        ),
        "causal_root_row_count": len(canonical_causal_roots),
        "causal_root_rows_sha256": canonical_sha256(
            list(canonical_causal_roots)
        ),
        "rows_without_exact_target_boundary_excluded_from_causal_replay": (
            int(hidden_rows.shape[0]) - len(canonical_causal_roots)
        ),
        "all_terminal_rows_retained_in_diagnostic_tensors": True,
        "all_eligible_exact_boundary_rows_retained_for_causal_replay": True,
        "unpaired_cached_prefill_logits_excluded": True,
        "sealed_parent_cache_reused_without_recapture": True,
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "child_forward_calls": 0,
            "child_generate_calls": 0,
            "network_access": False,
        },
    }
    return Lfm2TerminalCacheEvidence(
        hidden_rows=hidden_rows,
        donor_logits=donor_logits,
        causal_root_records=canonical_causal_roots,
        receipt={**body, "evidence_sha256": canonical_sha256(body)},
    )


def collect_lfm2_child_causal_residual_evidence(
    model: Any,
    trie_evidence: Lfm2CausalTrieEvidence,
    *,
    expected_input_table_sha256: str,
    expected_body_inventory_sha256: str,
    device: Any,
    evidence_role: str,
) -> Lfm2ChildCausalResidualEvidence:
    """Replay typed target prefixes through one standalone child coordinate."""

    if not isinstance(trie_evidence, Lfm2CausalTrieEvidence):
        raise TypeError("trie_evidence must be Lfm2CausalTrieEvidence.")
    expected_input_table_sha256 = _digest_text(
        expected_input_table_sha256,
        "expected_input_table_sha256",
    )
    expected_body_inventory_sha256 = _digest_text(
        expected_body_inventory_sha256,
        "expected_body_inventory_sha256",
    )
    if not isinstance(evidence_role, str) or not evidence_role.strip():
        raise Lfm2InterfaceFrontierError(
            "evidence_role must be nonempty text."
        )
    torch = _torch()
    prefixes = trie_evidence.target_prefix_token_ids
    requested_device = torch.device(device)
    if requested_device.type not in {"cpu", "cuda"} or (
        requested_device.type == "cuda" and not torch.cuda.is_available()
    ):
        raise Lfm2InterfaceFrontierError(
            "Child causal replay device must be an available CPU or CUDA device."
        )
    input_weight = getattr(model.get_input_embeddings(), "weight", None)
    output_module = model.get_output_embeddings()
    if input_weight is None or output_module is None:
        raise Lfm2InterfaceFrontierError(
            "Child causal replay requires concrete input/output embeddings."
        )
    vocabulary = int(input_weight.shape[0])
    if any(
        not prefix
        or min(prefix) < 0
        or max(prefix) >= vocabulary
        for prefix in prefixes
    ):
        raise Lfm2InterfaceFrontierError(
            "Child causal replay prefix exceeds the target vocabulary."
        )

    def coordinate_hashes() -> tuple[str, str]:
        return (
            _tensor_sha256(model.get_input_embeddings().weight),
            fingerprint_lfm2_causal_body_inventory(model),
        )

    input_before, body_before = coordinate_hashes()
    if (
        input_before != expected_input_table_sha256
        or body_before != expected_body_inventory_sha256
    ):
        raise Lfm2InterfaceFrontierError(
            "Child causal replay input/body coordinate differs from its pins."
        )
    state_before = fingerprint_model_state(model)
    try:
        original_device = next(model.parameters()).device
    except StopIteration as exc:
        raise Lfm2InterfaceFrontierError(
            "Child causal replay model has no parameters."
        ) from exc
    captured: list[Any] = []

    def capture_terminal(_module: Any, args: tuple[Any, ...]) -> None:
        if (
            len(args) != 1
            or not torch.is_tensor(args[0])
            or args[0].ndim != 3
            or int(args[0].shape[0]) != 1
        ):
            raise Lfm2InterfaceFrontierError(
                "Child LM head did not receive one rank-3 batch-one residual stream."
            )
        captured.append(args[0][:, -1:, :].detach().cpu().clone())

    rows = []
    model.requires_grad_(False)
    model.eval()
    model.to(requested_device)
    handle = output_module.register_forward_pre_hook(capture_terminal)
    try:
        with torch.inference_mode():
            for prefix in prefixes:
                captured.clear()
                input_ids = torch.tensor(
                    [prefix],
                    dtype=torch.long,
                    device=requested_device,
                )
                if tuple(input_ids.shape) != (1, len(prefix)):
                    raise Lfm2InterfaceFrontierError(
                        "Child causal replay did not construct one exact full prefix."
                    )
                model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    position_ids=torch.arange(
                        len(prefix),
                        dtype=torch.long,
                        device=requested_device,
                    ).unsqueeze(0),
                    use_cache=False,
                )
                if len(captured) != 1:
                    raise Lfm2InterfaceFrontierError(
                        "Child causal replay did not expose one terminal residual."
                    )
                rows.append(captured[0].reshape(1, -1))
    finally:
        handle.remove()
        model.to(original_device)
    features = torch.cat(rows, dim=0).contiguous()
    state_after = fingerprint_model_state(model)
    input_after, body_after = coordinate_hashes()
    if (
        int(features.shape[0]) != len(prefixes)
        or not bool(torch.isfinite(features).all())
        or state_after != state_before
        or input_after != input_before
        or body_after != body_before
    ):
        raise Lfm2InterfaceFrontierError(
            "Child causal replay is incomplete, nonfinite, mutating, or changed "
            "its input/body coordinate."
        )
    trie_receipt = trie_evidence.as_dict()
    body = {
        "schema": LFM2_CHILD_CAUSAL_RESIDUAL_EVIDENCE_SCHEMA,
        "status": "complete-target-native-child-residual-replay",
        "evidence_role": evidence_role.strip(),
        "trie_evidence_sha256": trie_receipt["result_sha256"],
        "parent_terminal_evidence_sha256": trie_receipt[
            "parent_terminal_evidence_sha256"
        ],
        "cache_key": trie_receipt["cache_key"],
        "parent_identity_sha256": trie_receipt["parent_identity_sha256"],
        "selective_session_sha256": trie_receipt["selective_session_sha256"],
        "manifest_sha256": trie_receipt["manifest_sha256"],
        "completion_sha256": trie_receipt["completion_sha256"],
        "semantic_alignment_sha256": trie_receipt[
            "semantic_alignment_sha256"
        ],
        "tokenizer_live_session_sha256": trie_receipt[
            "tokenizer_live_session_sha256"
        ],
        "causal_root_rows_sha256": trie_receipt["causal_root_rows_sha256"],
        "contextual_transition_plan_sha256": trie_receipt[
            "contextual_transition_plan_sha256"
        ],
        "bridge_artifact_sha256": trie_receipt["bridge_artifact_sha256"],
        "bridge_profile_sha256": trie_receipt["bridge_profile_sha256"],
        "donor_tokenizer_semantics_sha256": trie_receipt[
            "donor_tokenizer_semantics_sha256"
        ],
        "target_tokenizer_semantics_sha256": trie_receipt[
            "target_tokenizer_semantics_sha256"
        ],
        "target_prefix_rows_sha256": trie_receipt["target_prefix_rows_sha256"],
        "child_hidden_rows_sha256": _tensor_sha256(features),
        "causal_input_table_sha256": input_before,
        "causal_body_inventory_sha256": body_before,
        "causal_input_table_after_sha256": input_after,
        "causal_body_inventory_after_sha256": body_after,
        "row_count": int(features.shape[0]),
        "hidden_size": int(features.shape[1]),
        "child_model_state_before_sha256": state_before,
        "child_model_state_after_sha256": state_after,
        "child_state_preserved": True,
        "input_and_body_coordinate_preserved": True,
        "batch_size": 1,
        "full_prefix_use_cache_false": True,
        "position_origin": 0,
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "child_forward_calls": len(prefixes),
            "child_generate_calls": 0,
            "network_access": False,
        },
    }
    return Lfm2ChildCausalResidualEvidence(
        hidden_rows=features,
        receipt={**body, "evidence_sha256": canonical_sha256(body)},
    )


def _target_token_surface(tokenizer: Any, token_id: int) -> str:
    try:
        value = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:  # pragma: no cover - legacy tokenizer boundary
        value = tokenizer.decode([token_id])
    if not isinstance(value, str):
        raise Lfm2InterfaceFrontierError("Tokenizer decode must return text.")
    return value


def _tokenize_with_offsets(tokenizer: Any, value: str) -> tuple[tuple[int, int, int], ...]:
    try:
        encoded = tokenizer(
            value,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2InterfaceFrontierError(
            "Donor tokenizer must expose exact local offset tokenization."
        ) from exc
    if not isinstance(encoded, Mapping):
        raise Lfm2InterfaceFrontierError("Tokenizer output must be a mapping.")
    raw_ids = encoded.get("input_ids")
    raw_offsets = encoded.get("offset_mapping")
    if (
        not isinstance(raw_ids, Sequence)
        or isinstance(raw_ids, (str, bytes))
        or not isinstance(raw_offsets, Sequence)
        or isinstance(raw_offsets, (str, bytes))
        or len(raw_ids) != len(raw_offsets)
    ):
        raise Lfm2InterfaceFrontierError(
            "Tokenizer IDs and offsets must be aligned sequences."
        )
    rows = []
    for raw_id, raw_offset in zip(raw_ids, raw_offsets, strict=True):
        if (
            isinstance(raw_id, bool)
            or not isinstance(raw_id, int)
            or not isinstance(raw_offset, Sequence)
            or len(raw_offset) != 2
        ):
            raise Lfm2InterfaceFrontierError("Tokenizer offset row is invalid.")
        start, end = raw_offset
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            raise Lfm2InterfaceFrontierError("Tokenizer offsets are invalid.")
        rows.append((int(raw_id), start, end))
    return tuple(rows)


def compile_lfm2_interface_sequence_plan(
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    *,
    contexts: Sequence[str] = DEFAULT_INTERFACE_CONTEXTS,
) -> dict[str, Any]:
    """Compile all target rows into authenticated donor context sequences."""

    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    bridge.verify_live_tokenizers(donor_tokenizer, target_tokenizer)
    normalized_contexts = tuple(contexts)
    if (
        not normalized_contexts
        or any(not isinstance(value, str) for value in normalized_contexts)
        or len(set(normalized_contexts)) != len(normalized_contexts)
    ):
        raise Lfm2InterfaceFrontierError(
            "contexts must be a nonempty unique text sequence."
        )
    profile = bridge.transport_profile
    special_ids = {
        int(value) for value in getattr(target_tokenizer, "all_special_ids", ())
    }
    records: list[dict[str, Any]] = []
    sequence_length_histogram: dict[int, int] = {}
    variant_count_histogram: dict[int, int] = {}
    rows_with_context_variants = 0
    for target_id, raw_anchor in enumerate(
        profile.target_to_donor_row_indices
    ):
        anchor = int(raw_anchor)
        variants: set[tuple[int, ...]] = {(anchor,)}
        surface = _target_token_surface(target_tokenizer, target_id)
        if target_id not in special_ids and surface:
            for prefix in normalized_contexts:
                selected = tuple(
                    token_id
                    for token_id, _start, end in _tokenize_with_offsets(
                        donor_tokenizer,
                        prefix + surface,
                    )
                    if end > len(prefix)
                )
                if selected:
                    variants.add(selected)
        ordered = tuple(sorted(variants))
        if any(
            not sequence
            or min(sequence) < 0
            or max(sequence) >= profile.donor_row_capacity
            for sequence in ordered
        ):
            raise Lfm2InterfaceFrontierError(
                "Context sequence addresses an invalid donor row."
            )
        if ordered != ((anchor,),):
            rows_with_context_variants += 1
        variant_count_histogram[len(ordered)] = (
            variant_count_histogram.get(len(ordered), 0) + 1
        )
        for sequence in ordered:
            sequence_length_histogram[len(sequence)] = (
                sequence_length_histogram.get(len(sequence), 0) + 1
            )
        records.append(
            {
                "target_id": target_id,
                "strict_donor_id": anchor,
                "donor_sequences": [list(value) for value in ordered],
            }
        )
    body = {
        "schema": LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "donor_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "contexts": list(normalized_contexts),
        "target_row_count": profile.target_row_capacity,
        "rows_with_context_variants": rows_with_context_variants,
        "variant_count_histogram": {
            str(key): value for key, value in sorted(variant_count_histogram.items())
        },
        "sequence_length_histogram": {
            str(key): value for key, value in sorted(sequence_length_histogram.items())
        },
        "records": records,
        "parent_forward_calls": 0,
        "training_steps": 0,
        "network_access": False,
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def _encode_without_specials(tokenizer: Any, value: str) -> tuple[int, ...]:
    encoder = getattr(tokenizer, "encode", None)
    if not callable(encoder):
        raise Lfm2InterfaceFrontierError("Tokenizer must expose encode().")
    try:
        raw = encoder(value, add_special_tokens=False)
    except TypeError:  # pragma: no cover - legacy tokenizer boundary
        raw = encoder(value)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise Lfm2InterfaceFrontierError("Tokenizer encode must return IDs.")
    values = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise Lfm2InterfaceFrontierError(
                "Tokenizer encode returned an invalid token ID."
            )
        values.append(int(item))
    return tuple(values)


def compile_lfm2_inverse_output_plan(
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    *,
    fallback_target_id: int | None = None,
) -> dict[str, Any]:
    """Assign every donor output row to its first target-token decomposition."""

    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    bridge.verify_live_tokenizers(donor_tokenizer, target_tokenizer)
    profile = bridge.transport_profile
    reverse_anchor = {
        int(donor_id): target_id
        for target_id, donor_id in enumerate(
            profile.target_to_donor_row_indices
        )
    }
    special_assignments: dict[int, int] = {}
    for role in ("bos", "eos", "pad", "unk"):
        donor_id = getattr(donor_tokenizer, f"{role}_token_id", None)
        target_id = getattr(target_tokenizer, f"{role}_token_id", None)
        if donor_id is None or target_id is None:
            continue
        if (
            isinstance(donor_id, bool)
            or not isinstance(donor_id, int)
            or isinstance(target_id, bool)
            or not isinstance(target_id, int)
            or not 0 <= donor_id < profile.donor_row_capacity
            or not 0 <= target_id < profile.target_row_capacity
        ):
            raise Lfm2InterfaceFrontierError(
                "Tokenizer special-token roles are outside bridge capacities."
            )
        special_assignments[int(donor_id)] = int(target_id)
    target_tokenizer_rows = len(target_tokenizer)
    donor_tokenizer_rows = len(donor_tokenizer)
    if (
        isinstance(target_tokenizer_rows, bool)
        or not isinstance(target_tokenizer_rows, int)
        or target_tokenizer_rows <= 0
        or isinstance(donor_tokenizer_rows, bool)
        or not isinstance(donor_tokenizer_rows, int)
        or donor_tokenizer_rows <= 0
        or target_tokenizer_rows > profile.target_row_capacity
        or donor_tokenizer_rows > profile.donor_row_capacity
    ):
        raise Lfm2InterfaceFrontierError(
            "Tokenizer row inventories exceed their authenticated bridge capacities."
        )
    target_unk = getattr(target_tokenizer, "unk_token_id", None)
    if target_unk is not None and (
        isinstance(target_unk, bool)
        or not isinstance(target_unk, int)
        or not 0 <= target_unk < target_tokenizer_rows
    ):
        raise Lfm2InterfaceFrontierError(
            "Target unknown-token row is outside the live tokenizer."
        )
    if fallback_target_id is None:
        effective_fallback_target_id = target_unk
        fallback_policy = "target-unknown-token"
    else:
        if (
            isinstance(fallback_target_id, bool)
            or not isinstance(fallback_target_id, int)
            or not 0 <= fallback_target_id < target_tokenizer_rows
        ):
            raise Lfm2InterfaceFrontierError(
                "fallback_target_id must address one defined target-tokenizer row."
            )
        effective_fallback_target_id = int(fallback_target_id)
        fallback_policy = "explicit-authenticated-target-row"
    if effective_fallback_target_id is None:
        raise Lfm2InterfaceFrontierError(
            "Inverse output compilation requires an explicit fallback target row "
            "when the target tokenizer has no unknown token."
        )
    effective_fallback_target_id = int(effective_fallback_target_id)
    fallback_target_surface = _target_token_surface(
        target_tokenizer,
        effective_fallback_target_id,
    )

    buckets: list[list[int]] = [
        [] for _target_id in range(profile.target_row_capacity)
    ]
    empty_decomposition_count = 0
    valid_empty_decomposition_count = 0
    invalid_donor_capacity_row_count = 0
    reverse_anchor_empty_assignment_count = 0
    fallback_assignment_count = 0
    special_assignment_count = 0
    encoded_assignment_count = 0
    donor_target_decompositions: list[list[int]] = []
    prefix_transition_buckets: list[list[int]] = [
        [] for _target_id in range(profile.target_row_capacity)
    ]
    for donor_id in range(profile.donor_row_capacity):
        if donor_id >= donor_tokenizer_rows:
            target_id = effective_fallback_target_id
            decomposition = (target_id,)
            empty_decomposition_count += 1
            invalid_donor_capacity_row_count += 1
            fallback_assignment_count += 1
        elif donor_id in special_assignments:
            target_id = special_assignments[donor_id]
            decomposition = (target_id,)
            special_assignment_count += 1
        else:
            surface = _target_token_surface(donor_tokenizer, donor_id)
            encoded = _encode_without_specials(target_tokenizer, surface)
            if encoded:
                target_id = encoded[0]
                decomposition = encoded
                encoded_assignment_count += 1
            else:
                valid_empty_decomposition_count += 1
                if donor_id in reverse_anchor:
                    target_id = reverse_anchor[donor_id]
                    reverse_anchor_empty_assignment_count += 1
                else:
                    target_id = effective_fallback_target_id
                    fallback_assignment_count += 1
                decomposition = (target_id,)
                empty_decomposition_count += 1
        if not 0 <= target_id < profile.target_row_capacity:
            raise Lfm2InterfaceFrontierError(
                "Inverse output assignment left the target vocabulary."
            )
        buckets[target_id].append(donor_id)
        donor_target_decompositions.append(list(decomposition))
        for transition_target_id in decomposition:
            if not 0 <= transition_target_id < profile.target_row_capacity:
                raise Lfm2InterfaceFrontierError(
                    "Donor decomposition left the target vocabulary."
                )
            prefix_transition_buckets[transition_target_id].append(donor_id)
    flattened = [donor_id for bucket in buckets for donor_id in bucket]
    if sorted(flattened) != list(range(profile.donor_row_capacity)):
        raise Lfm2InterfaceFrontierError(
            "Inverse output plan does not cover every donor row exactly once."
        )
    empty_target_rows = [
        target_id for target_id, bucket in enumerate(buckets) if not bucket
    ]
    counts = [len(bucket) for bucket in buckets]
    decomposition_lengths = [
        len(decomposition) for decomposition in donor_target_decompositions
    ]
    body = {
        "schema": LFM2_INTERFACE_INVERSE_OUTPUT_PLAN_SCHEMA,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "donor_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "assignment": "decoded-donor-surface-to-first-target-subtoken",
        "donor_row_count": profile.donor_row_capacity,
        "donor_tokenizer_defined_row_count": donor_tokenizer_rows,
        "target_tokenizer_defined_row_count": target_tokenizer_rows,
        "target_row_count": profile.target_row_capacity,
        "assigned_donor_row_count": len(flattened),
        "donor_rows_assigned_exactly_once": True,
        "special_assignment_count": special_assignment_count,
        "encoded_assignment_count": encoded_assignment_count,
        "empty_decomposition_count": empty_decomposition_count,
        "valid_empty_decomposition_count": valid_empty_decomposition_count,
        "invalid_donor_capacity_row_count": invalid_donor_capacity_row_count,
        "reverse_anchor_empty_assignment_count": (
            reverse_anchor_empty_assignment_count
        ),
        "fallback_assignment_count": fallback_assignment_count,
        "fallback_policy": fallback_policy,
        "fallback_target_id": effective_fallback_target_id,
        "fallback_target_surface": fallback_target_surface,
        "target_unknown_token_id": target_unk,
        "empty_target_row_count": len(empty_target_rows),
        "empty_target_rows": empty_target_rows,
        "minimum_bucket_size": min(counts),
        "maximum_bucket_size": max(counts),
        "buckets": buckets,
        "donor_target_decompositions": donor_target_decompositions,
        "prefix_transition_buckets": prefix_transition_buckets,
        "multi_subtoken_donor_count": sum(
            length > 1 for length in decomposition_lengths
        ),
        "maximum_decomposition_length": max(decomposition_lengths),
        "exact_autoregressive_tokenizer_bridge": False,
        "first_subtoken_mass_is_partial_bridge_only": True,
        "cached_child_features_after_inserted_target_subtokens_available": False,
        "exact_prefix_conditionals_compilable_from_this_plan": False,
        "prefix_transition_buckets_are_structural_proxies": True,
        "parent_forward_calls": 0,
        "training_steps": 0,
        "network_access": False,
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def _verify_precompiled_plan(
    plan: Mapping[str, Any],
    *,
    schema: str,
    bridge: StrictTokenizerBridgeArtifact,
) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise Lfm2InterfaceFrontierError("Precompiled plan must be a mapping.")
    value = dict(plan)
    body = dict(value)
    supplied = body.pop("plan_sha256", None)
    if (
        supplied != canonical_sha256(body)
        or value.get("schema") != schema
        or value.get("bridge_artifact_sha256") != bridge.artifact_sha256
        or value.get("bridge_profile_sha256") != bridge.profile_fingerprint
        or value.get("donor_tokenizer_semantics_sha256")
        != bridge.donor_tokenizer_semantics_sha256
        or value.get("target_tokenizer_semantics_sha256")
        != bridge.target_tokenizer_semantics_sha256
    ):
        raise Lfm2InterfaceFrontierError(
            "Precompiled interface plan differs from the live bridge."
        )
    return value


def compile_lfm2_contextual_input_table(
    source_rows: Any,
    sequence_plan: Mapping[str, Any],
    *,
    strict_anchor_ppm: int,
    preserve_anchor_norm: bool,
) -> Any:
    """Compile a sealed context table from already-projected donor rows.

    The donor hidden width is unconstrained, so callers can first transport a
    parent H1024 interface into the final H776 child coordinate and only then
    perform token/context composition.
    """

    torch = _torch()
    if (
        not isinstance(source_rows, torch.Tensor)
        or source_rows.ndim != 2
        or not torch.is_floating_point(source_rows)
        or torch.is_complex(source_rows)
        or source_rows.layout != torch.strided
        or source_rows.device.type == "meta"
        or not bool(torch.isfinite(source_rows).all())
    ):
        raise Lfm2InterfaceFrontierError(
            "Projected source interface must be one finite dense matrix."
        )
    if not isinstance(sequence_plan, Mapping):
        raise Lfm2InterfaceFrontierError("Sequence plan must be one sealed mapping.")
    plan = dict(sequence_plan)
    plan_body = dict(plan)
    plan_sha256 = plan_body.pop("plan_sha256", None)
    records = plan.get("records")
    if (
        plan_sha256 != canonical_sha256(plan_body)
        or plan.get("schema") != LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA
        or plan.get("target_row_count")
        != (len(records) if isinstance(records, list) else None)
        or not isinstance(records, list)
        or not records
    ):
        raise Lfm2InterfaceFrontierError(
            "Contextual input sequence plan is unsealed or incomplete."
        )
    _digest_text(
        plan.get("bridge_artifact_sha256"),
        "sequence-plan bridge_artifact_sha256",
    )
    _digest_text(
        plan.get("bridge_profile_sha256"),
        "sequence-plan bridge_profile_sha256",
    )
    _digest_text(
        plan.get("donor_tokenizer_semantics_sha256"),
        "sequence-plan donor_tokenizer_semantics_sha256",
    )
    _digest_text(
        plan.get("target_tokenizer_semantics_sha256"),
        "sequence-plan target_tokenizer_semantics_sha256",
    )
    if (
        isinstance(strict_anchor_ppm, bool)
        or not isinstance(strict_anchor_ppm, int)
        or not 0 <= strict_anchor_ppm <= 1_000_000
        or not isinstance(preserve_anchor_norm, bool)
    ):
        raise Lfm2InterfaceFrontierError(
            "Contextual input blend controls are invalid."
        )
    alpha = strict_anchor_ppm / 1_000_000.0
    source = source_rows.detach().to(device="cpu", dtype=torch.float64)
    compiled = []
    with torch.no_grad():
        for expected_target_id, record in enumerate(records):
            if (
                not isinstance(record, Mapping)
                or record.get("target_id") != expected_target_id
                or not isinstance(record.get("strict_donor_id"), int)
                or not isinstance(record.get("donor_sequences"), list)
            ):
                raise Lfm2InterfaceFrontierError(
                    "Sequence plan row ordering is invalid."
                )
            strict_id = int(record["strict_donor_id"])
            if not 0 <= strict_id < int(source.shape[0]):
                raise Lfm2InterfaceFrontierError(
                    "Sequence-plan strict donor anchor leaves projected source rows."
                )
            sequence_vectors = []
            for raw_sequence in record["donor_sequences"]:
                if (
                    not isinstance(raw_sequence, list)
                    or not raw_sequence
                    or any(
                        isinstance(donor_id, bool)
                        or not isinstance(donor_id, int)
                        or not 0 <= donor_id < int(source.shape[0])
                        for donor_id in raw_sequence
                    )
                ):
                    raise Lfm2InterfaceFrontierError(
                        "Sequence plan contains an empty donor sequence."
                    )
                indices = torch.tensor(raw_sequence, dtype=torch.long)
                sequence_vectors.append(source.index_select(0, indices).mean(dim=0))
            envelope = torch.stack(sequence_vectors, dim=0).mean(dim=0)
            anchor = source[strict_id]
            value = alpha * anchor + (1.0 - alpha) * envelope
            if preserve_anchor_norm:
                anchor_norm = torch.linalg.vector_norm(anchor)
                value_norm = torch.linalg.vector_norm(value)
                if float(value_norm) > 0.0:
                    value = value * (anchor_norm / value_norm)
                elif float(anchor_norm) > 0.0:
                    value = anchor
            if not bool(torch.isfinite(value).all()):
                raise Lfm2InterfaceFrontierError(
                    "Compiled interface row contains NaN or infinity."
                )
            compiled.append(value)
    result = torch.stack(compiled, dim=0).to(dtype=source_rows.dtype).contiguous()
    return result


def _compile_context_table(
    source_rows: Any,
    plan: Mapping[str, Any],
    *,
    strict_anchor_ppm: int,
    preserve_anchor_norm: bool,
) -> Any:
    """Compatibility wrapper for the authenticated public compiler."""

    return compile_lfm2_contextual_input_table(
        source_rows,
        plan,
        strict_anchor_ppm=strict_anchor_ppm,
        preserve_anchor_norm=preserve_anchor_norm,
    )


def compile_lfm2_inverse_bucket_output_table(
    source_rows: Any,
    inverse_plan: Mapping[str, Any],
    strict_mapping: Sequence[int],
    *,
    strict_anchor_ppm: int,
    preserve_anchor_norm: bool,
    weighting: str,
) -> Any:
    torch = _torch()
    if (
        not isinstance(source_rows, torch.Tensor)
        or source_rows.ndim != 2
        or not torch.is_floating_point(source_rows)
        or torch.is_complex(source_rows)
        or source_rows.layout != torch.strided
    ):
        raise Lfm2InterfaceFrontierError("Source output head must be dense.")
    buckets = inverse_plan.get("buckets")
    if (
        not isinstance(buckets, list)
        or len(buckets) != len(strict_mapping)
        or weighting not in {"uniform", "row-norm"}
    ):
        raise Lfm2InterfaceFrontierError(
            "Inverse output buckets or weighting are invalid."
        )
    alpha = strict_anchor_ppm / 1_000_000.0
    source = source_rows.detach().to(device="cpu", dtype=torch.float64)
    compiled = []
    with torch.no_grad():
        for target_id, raw_bucket in enumerate(buckets):
            if not isinstance(raw_bucket, list) or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < int(source.shape[0])
                for value in raw_bucket
            ):
                raise Lfm2InterfaceFrontierError(
                    "Inverse output bucket contains an invalid donor row."
                )
            anchor = source[int(strict_mapping[target_id])]
            if raw_bucket:
                rows = source.index_select(
                    0,
                    torch.tensor(raw_bucket, dtype=torch.long),
                )
                if weighting == "row-norm":
                    weights = torch.linalg.vector_norm(rows, dim=1)
                    denominator = weights.sum()
                    centroid = (
                        rows.mean(dim=0)
                        if float(denominator) <= 0.0
                        else (rows * weights[:, None]).sum(dim=0) / denominator
                    )
                else:
                    centroid = rows.mean(dim=0)
            else:
                centroid = anchor
            value = alpha * anchor + (1.0 - alpha) * centroid
            if preserve_anchor_norm:
                anchor_norm = torch.linalg.vector_norm(anchor)
                value_norm = torch.linalg.vector_norm(value)
                if float(value_norm) > 0.0:
                    value = value * (anchor_norm / value_norm)
                elif float(anchor_norm) > 0.0:
                    value = anchor
            if not bool(torch.isfinite(value).all()):
                raise Lfm2InterfaceFrontierError(
                    "Inverse output row contains NaN or infinity."
                )
            compiled.append(value)
    return torch.stack(compiled, dim=0).to(dtype=source_rows.dtype).contiguous()


def compile_lfm2_prefix_transition_output_table(
    source_rows: Any,
    inverse_plan: Mapping[str, Any],
    strict_mapping: Sequence[int],
    *,
    strict_anchor_ppm: int,
    preserve_anchor_norm: bool,
) -> Any:
    """Compile a later-subtoken-aware structural proxy over decompositions.

    A donor row contributes total weight one across every target subtoken in
    its decomposition.  This retains later pieces that first-subtoken buckets
    discard, but is deliberately not claimed as an exact prefix-conditional
    autoregressive bridge because it has no per-prefix child feature row.
    """

    torch = _torch()
    decompositions = inverse_plan.get("donor_target_decompositions")
    if (
        not isinstance(source_rows, torch.Tensor)
        or source_rows.ndim != 2
        or not torch.is_floating_point(source_rows)
        or torch.is_complex(source_rows)
        or source_rows.layout != torch.strided
        or not isinstance(decompositions, list)
        or len(decompositions) != int(source_rows.shape[0])
        or len(strict_mapping) <= 0
    ):
        raise Lfm2InterfaceFrontierError(
            "Prefix-transition source rows or decomposition plan are invalid."
        )
    if (
        isinstance(strict_anchor_ppm, bool)
        or not isinstance(strict_anchor_ppm, int)
        or not 0 <= strict_anchor_ppm <= 1_000_000
    ):
        raise Lfm2InterfaceFrontierError(
            "strict_anchor_ppm must be between zero and one million."
        )
    target_count = len(strict_mapping)
    weighted_rows: list[list[tuple[int, float]]] = [
        [] for _target_id in range(target_count)
    ]
    for donor_id, decomposition in enumerate(decompositions):
        if not isinstance(decomposition, list) or not decomposition or any(
            isinstance(target_id, bool)
            or not isinstance(target_id, int)
            or not 0 <= target_id < target_count
            for target_id in decomposition
        ):
            raise Lfm2InterfaceFrontierError(
                "Prefix-transition decomposition contains an invalid target row."
            )
        weight = 1.0 / len(decomposition)
        for target_id in decomposition:
            weighted_rows[target_id].append((donor_id, weight))
    source = source_rows.detach().to(device="cpu", dtype=torch.float64)
    alpha = strict_anchor_ppm / 1_000_000.0
    compiled = []
    with torch.no_grad():
        for target_id, entries in enumerate(weighted_rows):
            anchor = source[int(strict_mapping[target_id])]
            if entries:
                indices = torch.tensor(
                    [donor_id for donor_id, _weight in entries],
                    dtype=torch.long,
                )
                weights = torch.tensor(
                    [weight for _donor_id, weight in entries],
                    dtype=torch.float64,
                )
                rows = source.index_select(0, indices)
                proxy = (rows * weights[:, None]).sum(dim=0) / weights.sum()
            else:
                proxy = anchor
            value = alpha * anchor + (1.0 - alpha) * proxy
            if preserve_anchor_norm:
                anchor_norm = torch.linalg.vector_norm(anchor)
                value_norm = torch.linalg.vector_norm(value)
                if float(value_norm) > 0.0:
                    value = value * (anchor_norm / value_norm)
                elif float(anchor_norm) > 0.0:
                    value = anchor
            if not bool(torch.isfinite(value).all()):
                raise Lfm2InterfaceFrontierError(
                    "Prefix-transition structural row contains NaN or infinity."
                )
            compiled.append(value)
    return torch.stack(compiled, dim=0).to(dtype=source_rows.dtype).contiguous()


def _relative_centered_error(actual: Any, target: Any) -> float:
    torch = _torch()
    residual = actual - target
    denominator = torch.linalg.vector_norm(target).clamp_min(1e-12)
    return float((torch.linalg.vector_norm(residual) / denominator).detach().cpu())


def _aggregate_donor_logit_mass(
    donor_logits: Any,
    donor_to_target: Any,
    *,
    target_row_count: int,
    maximum_rows_per_chunk: int,
) -> Any:
    """Compute exact logsumexp mass for a total donor-to-target assignment."""

    torch = _torch()
    row_count, donor_row_count = (
        int(donor_logits.shape[0]),
        int(donor_logits.shape[1]),
    )
    if tuple(donor_to_target.shape) != (donor_row_count,):
        raise Lfm2InterfaceFrontierError(
            "Donor-to-target assignment does not cover the logits vocabulary."
        )
    chunks = []
    for start in range(0, row_count, maximum_rows_per_chunk):
        values = donor_logits[start : start + maximum_rows_per_chunk]
        rows = int(values.shape[0])
        indices = donor_to_target.view(1, -1).expand(rows, -1)
        maxima = torch.full(
            (rows, target_row_count),
            -torch.inf,
            dtype=values.dtype,
            device=values.device,
        )
        maxima.scatter_reduce_(
            1,
            indices,
            values,
            reduce="amax",
            include_self=True,
        )
        gathered_maxima = maxima.gather(1, indices)
        contributions = torch.exp(values - gathered_maxima)
        sums = torch.zeros_like(maxima)
        sums.scatter_add_(1, indices, contributions)
        chunks.append(maxima + torch.log(sums))
    return torch.cat(chunks, dim=0)


def _digest_text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2InterfaceFrontierError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return value


def _decode_without_cleanup(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        value = tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:  # pragma: no cover - legacy tokenizer boundary
        value = tokenizer.decode(list(token_ids))
    if not isinstance(value, str):
        raise Lfm2InterfaceFrontierError("Tokenizer decode must return text.")
    return value


def _causal_root_evidence_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    donor_row_count: int,
    target_row_count: int,
    terminal_row_count: int,
) -> tuple[dict[str, Any], ...]:
    """Canonicalize the exact root rows sealed by terminal cache evidence."""

    result = []
    seen_terminal_rows = set()
    required_integer_fields = (
        "terminal_row_index",
        "batch_index",
        "phase_row_index",
        "donor_boundary_position",
        "target_boundary_position",
    )
    for record in records:
        if not isinstance(record, Mapping):
            raise Lfm2InterfaceFrontierError("Causal root record must be a mapping.")
        normalized: dict[str, Any] = {}
        for field in required_integer_fields:
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Lfm2InterfaceFrontierError(
                    f"Causal root field {field!r} must be a nonnegative integer."
                )
            normalized[field] = int(value)
        if normalized["terminal_row_index"] >= terminal_row_count:
            raise Lfm2InterfaceFrontierError(
                "Causal root addresses a terminal evidence row that does not exist."
            )
        if normalized["terminal_row_index"] in seen_terminal_rows:
            raise Lfm2InterfaceFrontierError(
                "Causal roots repeat one terminal evidence row."
            )
        seen_terminal_rows.add(normalized["terminal_row_index"])
        phase = record.get("phase")
        if not isinstance(phase, str) or not phase:
            raise Lfm2InterfaceFrontierError("Causal root phase must be nonempty text.")
        normalized["phase"] = phase
        donor_prefix = tuple(record.get("donor_prefix_token_ids", ()))
        target_prefix = tuple(record.get("target_prefix_token_ids", ()))
        if (
            not donor_prefix
            or not target_prefix
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < donor_row_count
                for token_id in donor_prefix
            )
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < target_row_count
                for token_id in target_prefix
            )
            or normalized["donor_boundary_position"] != len(donor_prefix) - 1
            or normalized["target_boundary_position"] != len(target_prefix) - 1
        ):
            raise Lfm2InterfaceFrontierError(
                "Causal root donor/target prefixes are invalid or not terminal."
            )
        normalized["donor_prefix_token_ids"] = list(donor_prefix)
        normalized["target_prefix_token_ids"] = list(target_prefix)
        normalized["semantic_alignment_sha256"] = _digest_text(
            record.get("semantic_alignment_sha256"),
            "semantic_alignment_sha256",
        )
        result.append(normalized)
    if not result:
        raise Lfm2InterfaceFrontierError("Causal root evidence must be nonempty.")
    return tuple(result)


def _verified_terminal_cache_evidence(
    receipt: Mapping[str, Any],
    *,
    donor_logits: Any,
    root_records: tuple[dict[str, Any], ...],
    bridge: StrictTokenizerBridgeArtifact,
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise Lfm2InterfaceFrontierError(
            "Parent terminal evidence must be one sealed mapping."
        )
    value = dict(receipt)
    body = dict(value)
    supplied = body.pop("evidence_sha256", None)
    audit = value.get("operation_audit")
    ancestry_fields = (
        "cache_key",
        "parent_identity_sha256",
        "selective_session_sha256",
        "manifest_sha256",
        "completion_sha256",
        "semantic_alignment_sha256",
        "tokenizer_live_session_sha256",
    )
    ancestry_valid = True
    for field in ancestry_fields:
        try:
            _digest_text(value.get(field), field)
        except Lfm2InterfaceFrontierError:
            ancestry_valid = False
            break
    if (
        supplied != canonical_sha256(body)
        or value.get("schema") != LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA
        or value.get("terminal_row_count") != int(donor_logits.shape[0])
        or value.get("donor_vocabulary_size") != int(donor_logits.shape[1])
        or value.get("donor_logits_sha256") != _tensor_sha256(donor_logits)
        or value.get("causal_root_row_count") != len(root_records)
        or value.get("causal_root_rows_sha256")
        != canonical_sha256(list(root_records))
        or value.get("bridge_artifact_sha256") != bridge.artifact_sha256
        or value.get("bridge_profile_sha256") != bridge.profile_fingerprint
        or value.get("donor_tokenizer_semantics_sha256")
        != bridge.donor_tokenizer_semantics_sha256
        or value.get("target_tokenizer_semantics_sha256")
        != bridge.target_tokenizer_semantics_sha256
        or not ancestry_valid
        or not isinstance(audit, Mapping)
        or any(
            audit.get(name) != 0
            for name in (
                "training_steps",
                "autograd_grad_calls",
                "backward_calls",
                "parent_forward_calls",
                "parent_generate_calls",
                "parent_trace_recaptures",
            )
        )
    ):
        raise Lfm2InterfaceFrontierError(
            "Parent terminal cache receipt, tensors, roots, or tokenizer ancestry differ."
        )
    return value


def _special_transition_map(
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
) -> tuple[dict[int, int], set[int], tuple[dict[str, Any], ...]]:
    """Map only authenticated donor control rows to exact target controls."""

    profile = bridge.transport_profile
    donor_specials = {
        int(value) for value in getattr(donor_tokenizer, "all_special_ids", ())
    }
    target_specials = {
        int(value) for value in getattr(target_tokenizer, "all_special_ids", ())
    }
    candidates: dict[int, set[int]] = {}
    lineage = []
    for role in ("bos", "eos", "pad", "unk"):
        donor_id = getattr(donor_tokenizer, f"{role}_token_id", None)
        target_id = getattr(target_tokenizer, f"{role}_token_id", None)
        if isinstance(donor_id, int) and not isinstance(donor_id, bool):
            donor_specials.add(int(donor_id))
        if (
            isinstance(donor_id, int)
            and not isinstance(donor_id, bool)
            and isinstance(target_id, int)
            and not isinstance(target_id, bool)
            and 0 <= donor_id < profile.donor_row_capacity
            and 0 <= target_id < profile.target_row_capacity
        ):
            candidates.setdefault(int(donor_id), set()).add(int(target_id))
            lineage.append(
                {"source": "special-role", "role": role, "donor_id": donor_id, "target_id": target_id}
            )
    mapping = tuple(int(value) for value in profile.target_to_donor_row_indices)
    for target_id in sorted(target_specials):
        if not 0 <= target_id < len(mapping):
            continue
        donor_id = mapping[target_id]
        candidates.setdefault(donor_id, set()).add(target_id)
        lineage.append(
            {
                "source": "strict-bridge-special-anchor",
                "donor_id": donor_id,
                "target_id": target_id,
            }
        )
    exact = {
        donor_id: next(iter(target_ids))
        for donor_id, target_ids in candidates.items()
        if len(target_ids) == 1
    }
    return exact, donor_specials, tuple(lineage)


def _log_mass(values: Any, indices: Sequence[int], denominator: Any) -> float | None:
    torch = _torch()
    if not indices:
        return None
    selected = values.index_select(0, torch.tensor(indices, dtype=torch.long))
    return float((torch.logsumexp(selected, dim=0) - denominator).item())


def compile_lfm2_causal_trie_evidence(
    donor_logits: Any,
    causal_root_records: Sequence[Mapping[str, Any]],
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    *,
    parent_terminal_evidence: Mapping[str, Any],
    maximum_donor_candidates_per_root: int = 64,
    maximum_internal_prefixes_per_root: int = 8,
    maximum_total_internal_prefixes: int = 512,
) -> Lfm2CausalTrieEvidence:
    """Compile bounded, root-contextual target transitions from cached logits.

    Only each root's bounded top-K *defined* donor rows are tokenized. A donor
    candidate contributes evidence only when retokenizing the complete donor
    prefix plus that candidate canonically appends to the authenticated target
    root. Suffix rewrites, zero-width transitions, unmapped controls, undefined
    vocabulary rows, and unexamined rows remain structural-head territory.
    """

    torch = _torch()
    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        if isinstance(bridge, Mapping) and bridge.get("schema") == (
            LFM2_INTERFACE_INVERSE_OUTPUT_PLAN_SCHEMA
        ):
            raise Lfm2InterfaceFrontierError(
                "Causal-v2 rejects resealed context-free inverse output plans."
            )
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    bridge.verify_live_tokenizers(donor_tokenizer, target_tokenizer)
    if (
        not isinstance(donor_logits, torch.Tensor)
        or donor_logits.ndim != 2
        or not torch.is_floating_point(donor_logits)
        or torch.is_complex(donor_logits)
        or donor_logits.layout != torch.strided
        or donor_logits.device.type == "meta"
        or donor_logits.requires_grad
        or not bool(torch.isfinite(donor_logits).all())
    ):
        raise Lfm2InterfaceFrontierError(
            "Causal trie donor logits must be one detached finite dense matrix."
        )
    for value, label in (
        (maximum_donor_candidates_per_root, "maximum_donor_candidates_per_root"),
        (maximum_internal_prefixes_per_root, "maximum_internal_prefixes_per_root"),
        (maximum_total_internal_prefixes, "maximum_total_internal_prefixes"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Lfm2InterfaceFrontierError(f"{label} must be positive.")
    donor_row_count = int(donor_logits.shape[1])
    target_row_count = bridge.transport_profile.target_row_capacity
    terminal_row_count = int(donor_logits.shape[0])
    if donor_row_count != bridge.transport_profile.donor_row_capacity:
        raise Lfm2InterfaceFrontierError(
            "Donor logit width differs from the authenticated bridge capacity."
        )
    donor_defined_rows = len(donor_tokenizer)
    target_defined_rows = len(target_tokenizer)
    if (
        isinstance(donor_defined_rows, bool)
        or not isinstance(donor_defined_rows, int)
        or not 0 < donor_defined_rows <= donor_row_count
        or isinstance(target_defined_rows, bool)
        or not isinstance(target_defined_rows, int)
        or not 0 < target_defined_rows <= target_row_count
    ):
        raise Lfm2InterfaceFrontierError(
            "Live tokenizer inventories exceed authenticated bridge capacities."
        )
    roots = _causal_root_evidence_rows(
        causal_root_records,
        donor_row_count=donor_row_count,
        target_row_count=target_row_count,
        terminal_row_count=terminal_row_count,
    )
    terminal_receipt = _verified_terminal_cache_evidence(
        parent_terminal_evidence,
        donor_logits=donor_logits,
        root_records=roots,
        bridge=bridge,
    )
    logits = donor_logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    valid_donor_mask = torch.arange(donor_row_count) < donor_defined_rows
    valid_donor_mask_sha256 = _tensor_sha256(valid_donor_mask)
    special_map, donor_special_ids, special_lineage = _special_transition_map(
        bridge,
        donor_tokenizer,
        target_tokenizer,
    )
    candidate_count = min(maximum_donor_candidates_per_root, donor_defined_rows)
    root_plans = []
    append_by_root: list[tuple[dict[str, Any], ...]] = []
    quarantined_total = 0
    suffix_rewrite_total = 0
    empty_transition_total = 0
    for root_index, root in enumerate(roots):
        terminal_index = int(root["terminal_row_index"])
        root_values = logits[terminal_index]
        valid_values = root_values[:donor_defined_rows]
        valid_maximum = valid_values.max()
        valid_log_normalizer = torch.logsumexp(
            valid_values - valid_maximum,
            dim=0,
        )
        valid_denominator = valid_maximum + valid_log_normalizer
        top_values, top_ids = torch.topk(
            valid_values,
            k=candidate_count,
            largest=True,
            sorted=True,
        )
        ranked = sorted(
            (
                (float(score.item()), int(donor_id.item()))
                for score, donor_id in zip(top_values, top_ids, strict=True)
            ),
            key=lambda item: (-item[0], item[1]),
        )
        donor_prefix = tuple(root["donor_prefix_token_ids"])
        target_root = tuple(root["target_prefix_token_ids"])
        donor_root_text = _decode_without_cleanup(donor_tokenizer, donor_prefix)
        target_root_text = _decode_without_cleanup(target_tokenizer, target_root)
        canonical_target_root = _encode_without_specials(
            target_tokenizer,
            donor_root_text,
        )
        if donor_root_text != target_root_text or canonical_target_root != target_root:
            raise Lfm2InterfaceFrontierError(
                "Authenticated causal root is not the canonical target encoding "
                "of its donor prefix."
            )
        append_only = []
        quarantined = []
        for rank, (score, donor_id) in enumerate(ranked):
            base = {
                "rank": rank,
                "donor_id": donor_id,
                "donor_logit": score,
                "donor_log_probability": (
                    score
                    - float(valid_maximum.item())
                    - float(valid_log_normalizer.item())
                ),
            }
            if donor_id in donor_special_ids:
                target_special_id = special_map.get(donor_id)
                if target_special_id is None:
                    quarantined.append(
                        {**base, "reason": "unmapped-or-ambiguous-donor-control"}
                    )
                    continue
                append_only.append(
                    {
                        **base,
                        "transition_kind": "authenticated-special-control",
                        "target_append_token_ids": [target_special_id],
                    }
                )
                continue
            candidate_text = _decode_without_cleanup(
                donor_tokenizer,
                (*donor_prefix, donor_id),
            )
            if not candidate_text.startswith(donor_root_text):
                quarantined.append(
                    {
                        **base,
                        "reason": "donor-text-nonappend-only",
                        "candidate_text_sha256": canonical_sha256(candidate_text),
                    }
                )
                continue
            candidate_target = _encode_without_specials(
                target_tokenizer,
                candidate_text,
            )
            roundtrip = _decode_without_cleanup(target_tokenizer, candidate_target)
            common = 0
            for left, right in zip(target_root, candidate_target, strict=False):
                if left != right:
                    break
                common += 1
            if roundtrip != candidate_text:
                reason = "target-roundtrip-mismatch"
            elif candidate_target[: len(target_root)] != target_root:
                reason = "target-suffix-rewrite"
                suffix_rewrite_total += 1
            elif len(candidate_target) == len(target_root):
                reason = "zero-width-target-transition"
                empty_transition_total += 1
            else:
                append_only.append(
                    {
                        **base,
                        "transition_kind": "canonical-contextual-append",
                        "target_append_token_ids": list(
                            candidate_target[len(target_root) :]
                        ),
                        "candidate_text_sha256": canonical_sha256(candidate_text),
                    }
                )
                continue
            quarantined.append(
                {
                    **base,
                    "reason": reason,
                    "candidate_target_token_ids": list(candidate_target),
                    "target_common_prefix_tokens": common,
                    "target_replaced_root_suffix_tokens": len(target_root) - common,
                    "candidate_text_sha256": canonical_sha256(candidate_text),
                }
            )
        inspected_ids = [donor_id for _score, donor_id in ranked]
        append_ids = [int(value["donor_id"]) for value in append_only]
        quarantined_ids = [int(value["donor_id"]) for value in quarantined]
        unexamined_mask = torch.ones(donor_defined_rows, dtype=torch.bool)
        unexamined_mask[
            torch.tensor(inspected_ids, dtype=torch.long)
        ] = False
        unexamined_count = donor_defined_rows - len(inspected_ids)
        unexamined_log_mass = (
            None
            if unexamined_count == 0
            else float(
                (
                    torch.logsumexp(valid_values[unexamined_mask], dim=0)
                    - valid_denominator
                ).item()
            )
        )
        undefined_count = donor_row_count - donor_defined_rows
        undefined_log_mass = (
            None
            if undefined_count == 0
            else float(
                (
                    torch.logsumexp(root_values[donor_defined_rows:], dim=0)
                    - torch.logsumexp(root_values, dim=0)
                ).item()
            )
        )
        root_body = {
            "schema": LFM2_CAUSAL_CONTEXT_TRANSITION_PLAN_SCHEMA,
            "root_index": root_index,
            "terminal_row_index": terminal_index,
            "semantic_alignment_sha256": root["semantic_alignment_sha256"],
            "donor_prefix_token_ids": list(donor_prefix),
            "target_root_token_ids": list(target_root),
            "donor_prefix_sha256": canonical_sha256(list(donor_prefix)),
            "target_root_sha256": canonical_sha256(list(target_root)),
            "canonical_root_text_sha256": canonical_sha256(donor_root_text),
            "inspected_candidate_count": len(inspected_ids),
            "inspected_donor_candidate_ids": inspected_ids,
            "append_only_candidate_count": len(append_only),
            "append_only_candidates": append_only,
            "append_only_log_probability_mass": _log_mass(
                root_values,
                append_ids,
                valid_denominator,
            ),
            "quarantined_candidate_count": len(quarantined),
            "quarantined_donor_candidate_ids": quarantined_ids,
            "quarantined_candidates": quarantined,
            "quarantined_log_probability_mass": _log_mass(
                root_values,
                quarantined_ids,
                valid_denominator,
            ),
            "unexamined_defined_candidate_count": unexamined_count,
            "unexamined_defined_log_probability_mass": unexamined_log_mass,
            "undefined_donor_row_count": undefined_count,
            "undefined_donor_log_probability_mass_over_full_head": (
                undefined_log_mass
            ),
            "ordinary_replay_uses_only_append_only_candidates": True,
            "unexamined_and_quarantined_cells_supply_no_target_evidence": True,
            "unsupported_cells_start_from_structural_baseline": True,
        }
        root_plan = {
            **root_body,
            "root_transition_plan_sha256": canonical_sha256(root_body),
        }
        root_plans.append(root_plan)
        append_by_root.append(tuple(append_only))
        quarantined_total += len(quarantined)

    transition_body = {
        "schema": LFM2_CAUSAL_CONTEXT_TRANSITION_PLAN_SCHEMA,
        "status": "bounded-root-contextual-append-only-transitions-compiled",
        "parent_terminal_evidence_sha256": terminal_receipt["evidence_sha256"],
        "cache_key": terminal_receipt["cache_key"],
        "parent_identity_sha256": terminal_receipt["parent_identity_sha256"],
        "selective_session_sha256": terminal_receipt["selective_session_sha256"],
        "manifest_sha256": terminal_receipt["manifest_sha256"],
        "completion_sha256": terminal_receipt["completion_sha256"],
        "semantic_alignment_sha256": terminal_receipt[
            "semantic_alignment_sha256"
        ],
        "tokenizer_live_session_sha256": terminal_receipt[
            "tokenizer_live_session_sha256"
        ],
        "causal_root_rows_sha256": terminal_receipt["causal_root_rows_sha256"],
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "donor_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "donor_logits_sha256": _tensor_sha256(donor_logits),
        "valid_donor_mask_sha256": valid_donor_mask_sha256,
        "donor_head_row_count": donor_row_count,
        "donor_tokenizer_defined_row_count": donor_defined_rows,
        "undefined_donor_row_count": donor_row_count - donor_defined_rows,
        "target_row_count": target_row_count,
        "target_tokenizer_defined_row_count": target_defined_rows,
        "maximum_donor_candidates_per_root": maximum_donor_candidates_per_root,
        "candidate_contextual_retokenization_count": sum(
            int(value["inspected_candidate_count"]) for value in root_plans
        ),
        "full_donor_vocabulary_retokenized_per_root": False,
        "root_transition_plans": root_plans,
        "special_transition_lineage": list(special_lineage),
        "context_free_decomposition_plan_used": False,
        "all_undefined_donor_rows_excluded_before_topk": True,
        "diagnostic_metrics_are_not_acceptance_gates": True,
    }
    transition_plan_sha256 = canonical_sha256(transition_body)

    root_prefixes = [tuple(root["target_prefix_token_ids"]) for root in roots]
    root_target_rows = []
    root_support_rows = []
    proposals: dict[tuple[int, tuple[int, ...]], tuple[float, int]] = {}
    for root_index, candidates in enumerate(append_by_root):
        row = torch.zeros(target_row_count, dtype=torch.float32)
        support = torch.zeros(target_row_count, dtype=torch.bool)
        first_buckets: dict[int, list[int]] = {}
        for candidate_index, candidate in enumerate(candidates):
            decomposition = tuple(candidate["target_append_token_ids"])
            first_buckets.setdefault(decomposition[0], []).append(
                int(candidate["donor_id"])
            )
            for stop in range(1, len(decomposition)):
                key = (root_index, decomposition[:stop])
                priority = (
                    float(candidate["donor_log_probability"]),
                    int(candidate["rank"]),
                )
                incumbent = proposals.get(key)
                if incumbent is None or priority[0] > incumbent[0] or (
                    priority[0] == incumbent[0] and priority[1] < incumbent[1]
                ):
                    proposals[key] = priority
        terminal_index = int(roots[root_index]["terminal_row_index"])
        for target_id, donor_ids in sorted(first_buckets.items()):
            row[target_id] = torch.logsumexp(
                logits[
                    terminal_index,
                    torch.tensor(donor_ids, dtype=torch.long),
                ],
                dim=0,
            )
            support[target_id] = True
        root_target_rows.append(row)
        root_support_rows.append(support)
    ordered_proposals = sorted(
        proposals,
        key=lambda key: (
            -proposals[key][0],
            proposals[key][1],
            key[0],
            len(key[1]),
            key[1],
        ),
    )
    selected = []
    selected_per_root = [0 for _root in roots]
    for key in ordered_proposals:
        root_index, _prefix = key
        if selected_per_root[root_index] >= maximum_internal_prefixes_per_root:
            continue
        selected.append(key)
        selected_per_root[root_index] += 1
        if len(selected) >= maximum_total_internal_prefixes:
            break
    selected_set = set(selected)
    omitted_prefix_records = []
    for root_index, prefix in ordered_proposals:
        if (root_index, prefix) in selected_set:
            continue
        candidate_ids = [
            int(candidate["donor_id"])
            for candidate in append_by_root[root_index]
            if tuple(candidate["target_append_token_ids"])[: len(prefix)] == prefix
        ]
        terminal_index = int(roots[root_index]["terminal_row_index"])
        valid_denominator = torch.logsumexp(
            logits[terminal_index, :donor_defined_rows],
            dim=0,
        )
        omitted_prefix_records.append(
            {
                "root_index": root_index,
                "trie_prefix": list(prefix),
                "observed_candidate_count": len(candidate_ids),
                "observed_log_probability_mass": _log_mass(
                    logits[terminal_index],
                    candidate_ids,
                    valid_denominator,
                ),
            }
        )

    prefix_rows = list(root_prefixes)
    target_rows = list(root_target_rows)
    support_rows = list(root_support_rows)
    node_records = []
    ambiguous_count = 0
    for root_index, prefix in selected:
        continuations: dict[int, list[int]] = {}
        terminal_ids = []
        for candidate in append_by_root[root_index]:
            decomposition = tuple(candidate["target_append_token_ids"])
            if decomposition == prefix:
                terminal_ids.append(int(candidate["donor_id"]))
            elif decomposition[: len(prefix)] == prefix:
                continuations.setdefault(decomposition[len(prefix)], []).append(
                    int(candidate["donor_id"])
                )
        if not continuations:
            raise Lfm2InterfaceFrontierError(
                "Selected contextual trie prefix has no observed continuation."
            )
        row = torch.zeros(target_row_count, dtype=torch.float32)
        support = torch.zeros(target_row_count, dtype=torch.bool)
        terminal_row_index = int(roots[root_index]["terminal_row_index"])
        for target_id, donor_ids in sorted(continuations.items()):
            row[target_id] = torch.logsumexp(
                logits[
                    terminal_row_index,
                    torch.tensor(donor_ids, dtype=torch.long),
                ],
                dim=0,
            )
            support[target_id] = True
        ambiguous = bool(terminal_ids)
        ambiguous_count += int(ambiguous)
        valid_denominator = torch.logsumexp(
            logits[terminal_row_index, :donor_defined_rows],
            dim=0,
        )
        prefix_rows.append((*root_prefixes[root_index], *prefix))
        target_rows.append(row)
        support_rows.append(support)
        node_records.append(
            {
                "root_index": root_index,
                "root_transition_plan_sha256": root_plans[root_index][
                    "root_transition_plan_sha256"
                ],
                "trie_prefix": list(prefix),
                "continuation_target_ids": sorted(continuations),
                "continuation_donor_row_count": sum(
                    len(value) for value in continuations.values()
                ),
                "continuation_log_probability_mass": _log_mass(
                    logits[terminal_row_index],
                    [value for values in continuations.values() for value in values],
                    valid_denominator,
                ),
                "terminal_donor_row_count": len(terminal_ids),
                "terminal_log_probability_mass": _log_mass(
                    logits[terminal_row_index],
                    terminal_ids,
                    valid_denominator,
                ),
                "next_donor_conditional_ambiguity": ambiguous,
                "unsupported_cells_supply_no_contextual_target": True,
            }
        )
    result_logits = torch.stack(target_rows).contiguous()
    result_support = torch.stack(support_rows).contiguous()
    prefix_tuple = tuple(prefix_rows)
    root_supported_cells = sum(int(value.sum().item()) for value in root_support_rows)
    body = {
        "schema": LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA,
        "status": "cache-only-bounded-contextual-causal-conditionals-compiled",
        "parent_terminal_evidence_sha256": terminal_receipt["evidence_sha256"],
        "cache_key": terminal_receipt["cache_key"],
        "parent_identity_sha256": terminal_receipt["parent_identity_sha256"],
        "selective_session_sha256": terminal_receipt["selective_session_sha256"],
        "manifest_sha256": terminal_receipt["manifest_sha256"],
        "completion_sha256": terminal_receipt["completion_sha256"],
        "semantic_alignment_sha256": terminal_receipt[
            "semantic_alignment_sha256"
        ],
        "tokenizer_live_session_sha256": terminal_receipt[
            "tokenizer_live_session_sha256"
        ],
        "causal_root_rows_sha256": terminal_receipt["causal_root_rows_sha256"],
        "contextual_transition_plan_sha256": transition_plan_sha256,
        "contextual_transition_plan": {
            **transition_body,
            "plan_sha256": transition_plan_sha256,
        },
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "donor_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "donor_logits_sha256": _tensor_sha256(donor_logits),
        "valid_donor_mask_sha256": valid_donor_mask_sha256,
        "target_prefix_rows_sha256": canonical_sha256(
            [list(value) for value in prefix_tuple]
        ),
        "target_logits_sha256": _tensor_sha256(result_logits),
        "target_logit_support_mask_sha256": _tensor_sha256(result_support),
        "root_row_count": len(roots),
        "all_root_rows_retained": True,
        "root_rows_without_observed_append_support": sum(
            not bool(value.any()) for value in root_support_rows
        ),
        "root_supported_logit_cell_count": root_supported_cells,
        "root_support_is_bounded_partial_observation": True,
        "root_unknown_cells_receive_finite_floor": False,
        "root_unexamined_and_quarantined_cells_remain_unsupported": True,
        "quarantined_candidate_count": quarantined_total,
        "suffix_rewrite_candidate_count": suffix_rewrite_total,
        "zero_width_transition_candidate_count": empty_transition_total,
        "internal_prefix_proposal_count": len(proposals),
        "internal_prefix_row_count": len(selected),
        "omitted_internal_prefix_count": len(omitted_prefix_records),
        "omitted_internal_prefix_records": omitted_prefix_records,
        "total_row_count": len(prefix_tuple),
        "maximum_donor_candidates_per_root": maximum_donor_candidates_per_root,
        "maximum_internal_prefixes_per_root": maximum_internal_prefixes_per_root,
        "maximum_total_internal_prefixes": maximum_total_internal_prefixes,
        "bounds_prioritize_replay_cost_not_candidate_acceptance": True,
        "cross_root_prefix_priority_uses_normalized_log_probability": True,
        "ambiguous_terminal_prefix_node_count": ambiguous_count,
        "next_donor_conditionals_available": False,
        "node_records": node_records,
        "context_free_first_bucket_plan_used": False,
        "exact_all_branch_autoregressive_equivalence_claimed": False,
        "bounded_partial_conditionals_only": True,
        "missing_cache_statistics": [
            "unexamined donor candidates outside each bounded top-K",
            "next-donor conditionals after a contextual decomposition terminates",
            "ordinary autoregressive realization of suffix-rewrite candidates",
        ],
        "diagnostic_metrics_are_not_acceptance_gates": True,
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "child_forward_calls": 0,
            "network_access": False,
        },
    }
    return Lfm2CausalTrieEvidence(
        target_prefix_token_ids=prefix_tuple,
        target_logits=result_logits,
        target_logit_support_mask=result_support,
        receipt={**body, "result_sha256": canonical_sha256(body)},
    )


def compile_lfm2_cached_logit_residual_head(
    structural_output_table: Any,
    hidden_rows: Any,
    donor_logits: Any,
    inverse_output_plan: Mapping[str, Any],
    *,
    source_evidence_sha256: str,
    ridge: float,
    residual_scale_ppm: int,
    protected_target_ids: Sequence[int] = (),
    row_norm_calibration_ppm: int = 0,
    solve_device: Any = "cpu",
    maximum_rows_per_aggregation_chunk: int = 32,
) -> Lfm2CachedLogitResidualHead:
    """Compile a dense target head by one closed-form cache-only solve.

    Every donor token contributes to exactly one first-target-subtoken bucket.
    The sealed parent logits are reduced by exact log-sum-exp mass, then a
    dual ridge residual around the structural output table is solved without
    autograd.  This is analytic parameter transport, not iterative fitting.
    """

    torch = _torch()
    if (
        not isinstance(source_evidence_sha256, str)
        or len(source_evidence_sha256) != 64
        or source_evidence_sha256 != source_evidence_sha256.lower()
        or any(value not in "0123456789abcdef" for value in source_evidence_sha256)
    ):
        raise Lfm2InterfaceFrontierError(
            "source_evidence_sha256 must be a lowercase SHA-256 digest."
        )
    for value, label in (
        (structural_output_table, "structural output table"),
        (hidden_rows, "hidden rows"),
        (donor_logits, "donor logits"),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or not torch.is_floating_point(value)
            or torch.is_complex(value)
            or value.layout != torch.strided
            or not bool(torch.isfinite(value).all())
        ):
            raise Lfm2InterfaceFrontierError(
                f"Cached {label} must be one finite dense matrix."
            )
    if (
        int(hidden_rows.shape[0]) != int(donor_logits.shape[0])
        or int(hidden_rows.shape[0]) <= 0
        or int(hidden_rows.shape[1]) != int(structural_output_table.shape[1])
    ):
        raise Lfm2InterfaceFrontierError(
            "Cached terminal hidden/logit rows do not align with the head."
        )
    if (
        isinstance(ridge, bool)
        or not isinstance(ridge, (int, float))
        or not float(ridge) > 0.0
        or not float(ridge) < float("inf")
    ):
        raise Lfm2InterfaceFrontierError("ridge must be finite and positive.")
    for value, label in (
        (residual_scale_ppm, "residual_scale_ppm"),
        (row_norm_calibration_ppm, "row_norm_calibration_ppm"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 1_000_000
        ):
            raise Lfm2InterfaceFrontierError(
                f"{label} must be an integer from zero through one million."
            )
    if (
        isinstance(maximum_rows_per_aggregation_chunk, bool)
        or not isinstance(maximum_rows_per_aggregation_chunk, int)
        or maximum_rows_per_aggregation_chunk <= 0
    ):
        raise Lfm2InterfaceFrontierError(
            "maximum_rows_per_aggregation_chunk must be positive."
        )
    plan_body = dict(inverse_output_plan)
    supplied_plan_sha256 = plan_body.pop("plan_sha256", None)
    buckets = inverse_output_plan.get("buckets")
    target_row_count = int(structural_output_table.shape[0])
    donor_row_count = int(donor_logits.shape[1])
    if (
        supplied_plan_sha256 != canonical_sha256(plan_body)
        or inverse_output_plan.get("schema")
        != LFM2_INTERFACE_INVERSE_OUTPUT_PLAN_SCHEMA
        or inverse_output_plan.get("donor_row_count") != donor_row_count
        or inverse_output_plan.get("target_row_count") != target_row_count
        or inverse_output_plan.get("donor_rows_assigned_exactly_once") is not True
        or not isinstance(buckets, list)
        or len(buckets) != target_row_count
    ):
        raise Lfm2InterfaceFrontierError(
            "Inverse output plan is unsealed or incompatible with cached logits."
        )
    donor_to_target = torch.full((donor_row_count,), -1, dtype=torch.long)
    seen: list[int] = []
    empty_target_ids: list[int] = []
    for target_id, bucket in enumerate(buckets):
        if not isinstance(bucket, list) or any(
            isinstance(donor_id, bool)
            or not isinstance(donor_id, int)
            or not 0 <= donor_id < donor_row_count
            for donor_id in bucket
        ):
            raise Lfm2InterfaceFrontierError(
                "Inverse output plan contains an invalid donor bucket."
            )
        if not bucket:
            empty_target_ids.append(target_id)
            continue
        seen.extend(bucket)
        donor_to_target[torch.tensor(bucket, dtype=torch.long)] = target_id
    if sorted(seen) != list(range(donor_row_count)) or bool(
        (donor_to_target < 0).any()
    ):
        raise Lfm2InterfaceFrontierError(
            "Inverse output plan does not assign every donor row exactly once."
        )
    protected = tuple(sorted(set(protected_target_ids)))
    if any(
        isinstance(target_id, bool)
        or not isinstance(target_id, int)
        or not 0 <= target_id < target_row_count
        for target_id in protected
    ):
        raise Lfm2InterfaceFrontierError(
            "protected_target_ids contain a row outside the target head."
        )

    requested_device = torch.device(solve_device)
    if requested_device.type not in {"cpu", "cuda"} or (
        requested_device.type == "cuda" and not torch.cuda.is_available()
    ):
        raise Lfm2InterfaceFrontierError(
            "solve_device must be an available CPU or CUDA device."
        )
    structural_sha256 = _tensor_sha256(structural_output_table)
    hidden_sha256 = _tensor_sha256(hidden_rows)
    donor_logits_sha256 = _tensor_sha256(donor_logits)
    scale = residual_scale_ppm / 1_000_000.0
    norm_scale = row_norm_calibration_ppm / 1_000_000.0
    with torch.no_grad():
        hidden = hidden_rows.detach().to(
            device=requested_device,
            dtype=torch.float32,
        )
        logits = donor_logits.detach().to(
            device=requested_device,
            dtype=torch.float32,
        )
        structural = structural_output_table.detach().to(
            device=requested_device,
            dtype=torch.float32,
        )
        assignment = donor_to_target.to(device=requested_device)
        target_logits = _aggregate_donor_logit_mass(
            logits,
            assignment,
            target_row_count=target_row_count,
            maximum_rows_per_chunk=maximum_rows_per_aggregation_chunk,
        )
        structural_prediction = hidden.matmul(structural.t())
        if empty_target_ids:
            empty_index = torch.tensor(
                empty_target_ids,
                dtype=torch.long,
                device=requested_device,
            )
            target_logits.index_copy_(
                1,
                empty_index,
                structural_prediction.index_select(1, empty_index),
            )
        if not bool(torch.isfinite(target_logits).all()):
            raise Lfm2InterfaceFrontierError(
                "Aggregated target logit mass contains NaN or infinity."
            )
        centered_target = target_logits - target_logits.mean(dim=1, keepdim=True)
        residual = target_logits - structural_prediction
        residual = residual - residual.mean(dim=1, keepdim=True)
        gram = hidden.matmul(hidden.t())
        gram = gram + float(ridge) * torch.eye(
            int(gram.shape[0]),
            dtype=gram.dtype,
            device=gram.device,
        )
        dual = torch.linalg.solve(gram, hidden)
        delta = residual.t().matmul(dual)
        compiled = structural + scale * delta
        if protected:
            protected_index = torch.tensor(
                protected,
                dtype=torch.long,
                device=requested_device,
            )
            compiled.index_copy_(
                0,
                protected_index,
                structural.index_select(0, protected_index),
            )
        if norm_scale > 0.0:
            structural_norm = torch.linalg.vector_norm(structural, dim=1)
            compiled_norm = torch.linalg.vector_norm(compiled, dim=1)
            calibrated_norm = (
                (1.0 - norm_scale) * compiled_norm
                + norm_scale * structural_norm
            )
            safe_scale = torch.where(
                compiled_norm > 0.0,
                calibrated_norm / compiled_norm.clamp_min(1e-12),
                torch.ones_like(compiled_norm),
            )
            compiled = compiled * safe_scale[:, None]
            zero_rows = (compiled_norm <= 0.0) & (structural_norm > 0.0)
            if bool(zero_rows.any()):
                compiled[zero_rows] = structural[zero_rows]
        if protected:
            compiled.index_copy_(
                0,
                protected_index,
                structural.index_select(0, protected_index),
            )
        result_table = compiled.to(
            device="cpu",
            dtype=structural_output_table.dtype,
        ).contiguous()
        projected_compiled = result_table.to(
            device=requested_device,
            dtype=torch.float32,
        )
        final_prediction = hidden.matmul(projected_compiled.t())
        centered_structural_prediction = (
            structural_prediction
            - structural_prediction.mean(dim=1, keepdim=True)
        )
        centered_final_prediction = (
            final_prediction - final_prediction.mean(dim=1, keepdim=True)
        )
        before_error = _relative_centered_error(
            centered_structural_prediction,
            centered_target,
        )
        after_error = _relative_centered_error(
            centered_final_prediction,
            centered_target,
        )
    if not bool(torch.isfinite(result_table).all()):
        raise Lfm2InterfaceFrontierError(
            "Closed-form residual solve produced NaN or infinity."
        )
    output_sha256 = _tensor_sha256(result_table)
    body = {
        "schema": LFM2_CACHED_LOGIT_RESIDUAL_HEAD_SCHEMA,
        "status": "dense-cache-only-dual-ridge-residual-head-compiled",
        "source_evidence_sha256": source_evidence_sha256,
        "inverse_output_plan_sha256": supplied_plan_sha256,
        "assignment": "full-donor-logit-mass-to-first-target-subtoken",
        "aggregation": "exact-bucket-logsumexp",
        "structural_output_table_sha256": structural_sha256,
        "hidden_rows_sha256": hidden_sha256,
        "donor_logits_sha256": donor_logits_sha256,
        "output_table_sha256": output_sha256,
        "evidence_row_count": int(hidden_rows.shape[0]),
        "hidden_size": int(hidden_rows.shape[1]),
        "donor_row_count": donor_row_count,
        "target_row_count": target_row_count,
        "empty_target_row_count": len(empty_target_ids),
        "empty_target_rows_use_structural_prediction": True,
        "ridge": float(ridge),
        "residual_scale_ppm": residual_scale_ppm,
        "row_norm_calibration_ppm": row_norm_calibration_ppm,
        "protected_target_ids": list(protected),
        "relative_centered_error_before": before_error,
        "relative_centered_error_after": after_error,
        "diagnostic_metrics_are_not_acceptance_gates": True,
        "solve_device": str(requested_device),
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "network_access": False,
        },
    }
    return Lfm2CachedLogitResidualHead(
        table=result_table,
        receipt={**body, "result_sha256": canonical_sha256(body)},
    )


def _verify_child_causal_feature_evidence(
    receipt: Mapping[str, Any],
    *,
    child_hidden_rows: Any,
    trie: Lfm2CausalTrieEvidence,
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise Lfm2InterfaceFrontierError(
            "Child causal feature evidence must be a sealed mapping."
        )
    value = dict(receipt)
    body = dict(value)
    supplied = body.pop("evidence_sha256", None)
    trie_receipt = trie.as_dict()
    audit = value.get("operation_audit")
    if (
        supplied != canonical_sha256(body)
        or value.get("schema") != LFM2_CHILD_CAUSAL_RESIDUAL_EVIDENCE_SCHEMA
        or value.get("trie_evidence_sha256") != trie_receipt["result_sha256"]
        or value.get("parent_terminal_evidence_sha256")
        != trie_receipt["parent_terminal_evidence_sha256"]
        or any(
            value.get(field) != trie_receipt.get(field)
            for field in (
                "cache_key",
                "parent_identity_sha256",
                "selective_session_sha256",
                "manifest_sha256",
                "completion_sha256",
                "semantic_alignment_sha256",
                "tokenizer_live_session_sha256",
            )
        )
        or value.get("causal_root_rows_sha256")
        != trie_receipt["causal_root_rows_sha256"]
        or value.get("contextual_transition_plan_sha256")
        != trie_receipt["contextual_transition_plan_sha256"]
        or value.get("bridge_artifact_sha256")
        != trie_receipt["bridge_artifact_sha256"]
        or value.get("bridge_profile_sha256")
        != trie_receipt["bridge_profile_sha256"]
        or value.get("donor_tokenizer_semantics_sha256")
        != trie_receipt["donor_tokenizer_semantics_sha256"]
        or value.get("target_tokenizer_semantics_sha256")
        != trie_receipt["target_tokenizer_semantics_sha256"]
        or value.get("target_prefix_rows_sha256")
        != trie_receipt["target_prefix_rows_sha256"]
        or value.get("child_hidden_rows_sha256")
        != _tensor_sha256(child_hidden_rows)
        or value.get("row_count") != int(child_hidden_rows.shape[0])
        or value.get("hidden_size") != int(child_hidden_rows.shape[1])
        or value.get("child_model_state_before_sha256")
        != value.get("child_model_state_after_sha256")
        or value.get("child_state_preserved") is not True
        or not isinstance(value.get("causal_input_table_sha256"), str)
        or not isinstance(value.get("causal_body_inventory_sha256"), str)
        or not isinstance(audit, Mapping)
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
                "child_generate_calls",
            )
        )
    ):
        raise Lfm2InterfaceFrontierError(
            "Child causal feature receipt, tensors, prefix order, or ancestry differ."
        )
    _digest_text(value["causal_input_table_sha256"], "causal_input_table_sha256")
    _digest_text(
        value["causal_body_inventory_sha256"],
        "causal_body_inventory_sha256",
    )
    return value


def compile_lfm2_causal_residual_token_bridge(
    structural_output_table: Any,
    child_feature_evidence: Lfm2ChildCausalResidualEvidence,
    trie_evidence: Lfm2CausalTrieEvidence,
    *,
    parent_terminal_evidence: Mapping[str, Any],
    sequence_plan_sha256: str,
    ridge: float,
    residual_scale_ppm: int,
    protected_target_ids: Sequence[int] = (),
    row_norm_calibration_ppm: int = 0,
    unsupported_logit_policy: str = "measure-drift",
    solve_device: Any = "cpu",
) -> Lfm2CausalResidualTokenBridge:
    """Compile bounded contextual evidence in the authenticated child coordinate.

    Each output row is fit only on feature rows where that target token has
    contextual support; identical masks share one factorization.  The default
    ``measure-drift`` policy reports the otherwise unconstrained background
    drift.  ``sealed-background-nullspace`` restricts each correction to the
    nullspace of its unsupported child features and reverts any row whose
    target-dtype projection does not preserve those logits exactly.
    """

    if not isinstance(trie_evidence, Lfm2CausalTrieEvidence):
        raise TypeError("trie_evidence must be Lfm2CausalTrieEvidence.")
    if not isinstance(
        child_feature_evidence,
        Lfm2ChildCausalResidualEvidence,
    ):
        raise TypeError(
            "child_feature_evidence must be Lfm2ChildCausalResidualEvidence."
        )
    child_hidden_rows = child_feature_evidence.hidden_rows
    sequence_plan_sha256 = _digest_text(
        sequence_plan_sha256,
        "sequence_plan_sha256",
    )
    torch = _torch()
    target_logits = trie_evidence.target_logits
    target_logit_support_mask = trie_evidence.target_logit_support_mask
    for value, label in (
        (structural_output_table, "structural output table"),
        (child_hidden_rows, "child hidden rows"),
        (target_logits, "target logits"),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or not torch.is_floating_point(value)
            or torch.is_complex(value)
            or value.layout != torch.strided
            or value.device.type == "meta"
            or value.requires_grad
            or not bool(torch.isfinite(value).all())
        ):
            raise Lfm2InterfaceFrontierError(
                f"Causal {label} must be one detached finite dense matrix."
            )
    rows = int(child_hidden_rows.shape[0])
    target_row_count = int(structural_output_table.shape[0])
    hidden_size = int(structural_output_table.shape[1])
    if (
        rows <= 0
        or tuple(target_logits.shape) != (rows, target_row_count)
        or tuple(target_logit_support_mask.shape) != (rows, target_row_count)
        or int(child_hidden_rows.shape[1]) != hidden_size
    ):
        raise Lfm2InterfaceFrontierError(
            "Causal features, typed trie tensors, and dense head do not align."
        )
    parent_value = dict(parent_terminal_evidence)
    parent_body = dict(parent_value)
    parent_sha256 = parent_body.pop("evidence_sha256", None)
    trie_receipt = trie_evidence.as_dict()
    if (
        parent_sha256 != canonical_sha256(parent_body)
        or parent_value.get("schema") != LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA
        or parent_sha256 != trie_receipt["parent_terminal_evidence_sha256"]
        or parent_value.get("causal_root_rows_sha256")
        != trie_receipt["causal_root_rows_sha256"]
        or any(
            parent_value.get(field) != trie_receipt.get(field)
            for field in (
                "cache_key",
                "parent_identity_sha256",
                "selective_session_sha256",
                "manifest_sha256",
                "completion_sha256",
                "semantic_alignment_sha256",
                "tokenizer_live_session_sha256",
            )
        )
        or parent_value.get("bridge_artifact_sha256")
        != trie_receipt["bridge_artifact_sha256"]
        or parent_value.get("bridge_profile_sha256")
        != trie_receipt["bridge_profile_sha256"]
    ):
        raise Lfm2InterfaceFrontierError(
            "Typed trie no longer descends from the supplied terminal cache receipt."
        )
    child_value = _verify_child_causal_feature_evidence(
        child_feature_evidence.as_dict(),
        child_hidden_rows=child_hidden_rows,
        trie=trie_evidence,
    )
    if (
        isinstance(ridge, bool)
        or not isinstance(ridge, (int, float))
        or not 0.0 < float(ridge) < float("inf")
    ):
        raise Lfm2InterfaceFrontierError("ridge must be finite and positive.")
    if unsupported_logit_policy not in {
        "measure-drift",
        "sealed-background-nullspace",
    }:
        raise Lfm2InterfaceFrontierError(
            "unsupported_logit_policy must be 'measure-drift' or "
            "'sealed-background-nullspace'."
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
            raise Lfm2InterfaceFrontierError(
                f"{label} must be an integer from zero through one million."
            )
    protected = tuple(sorted(set(protected_target_ids)))
    if any(
        isinstance(target_id, bool)
        or not isinstance(target_id, int)
        or not 0 <= target_id < target_row_count
        for target_id in protected
    ):
        raise Lfm2InterfaceFrontierError(
            "protected_target_ids contain a row outside the target head."
        )
    requested_device = torch.device(solve_device)
    if requested_device.type not in {"cpu", "cuda"} or (
        requested_device.type == "cuda" and not torch.cuda.is_available()
    ):
        raise Lfm2InterfaceFrontierError(
            "solve_device must be an available CPU or CUDA device."
        )
    structural_sha256 = _tensor_sha256(structural_output_table)
    if child_value["causal_input_table_sha256"] != structural_sha256:
        raise Lfm2InterfaceFrontierError(
            "Causal-v2 structural head must begin in the replayed input coordinate."
        )
    sealed_background = (
        unsupported_logit_policy == "sealed-background-nullspace"
    )
    child_hidden_sha256 = _tensor_sha256(child_hidden_rows)
    original_support_sha256 = _tensor_sha256(target_logit_support_mask)
    scale = residual_scale_ppm / 1_000_000.0
    norm_scale = row_norm_calibration_ppm / 1_000_000.0
    with torch.no_grad():
        hidden = child_hidden_rows.detach().to(
            device=requested_device,
            dtype=torch.float32,
        )
        evidence_logits = target_logits.detach().to(
            device=requested_device,
            dtype=torch.float32,
        )
        support = target_logit_support_mask.detach().to(
            device=requested_device
        ).clone()
        protected_support_cells_removed = 0
        protected_index = None
        if protected:
            protected_index = torch.tensor(
                protected,
                dtype=torch.long,
                device=requested_device,
            )
            protected_support_cells_removed = int(
                support.index_select(1, protected_index).sum().item()
            )
            support.index_fill_(1, protected_index, False)
        effective_support_cpu = support.detach().cpu().contiguous()
        effective_support_sha256 = _tensor_sha256(effective_support_cpu)
        structural = structural_output_table.detach().to(
            device=requested_device,
            dtype=torch.float32,
        )
        structural_prediction = hidden.matmul(structural.t())
        effective_target = torch.where(
            support,
            evidence_logits,
            structural_prediction,
        )
        residual = effective_target - structural_prediction
        support_count = support.sum(dim=1, keepdim=True).clamp_min(1)
        supported_residual_mean = (
            (residual * support).sum(dim=1, keepdim=True) / support_count
        )
        residual = torch.where(
            support,
            residual - supported_residual_mean,
            torch.zeros_like(residual),
        )
        # Each output row is fit only on feature rows where that target token
        # has contextual evidence. Columns with identical masks share one
        # factorization; columns with no observations retain delta=0.
        support_groups: dict[tuple[int, ...], list[int]] = {}
        support_cpu = support.detach().cpu()
        for target_id in range(target_row_count):
            observed_rows = tuple(
                int(value)
                for value in torch.nonzero(
                    support_cpu[:, target_id],
                    as_tuple=False,
                ).flatten()
            )
            support_groups.setdefault(observed_rows, []).append(target_id)
        delta = torch.zeros_like(structural)
        dual_group_count = 0
        primal_group_count = 0
        empty_output_row_count = 0
        maximum_group_gram_dimension = 0
        background_constraint_group_count = 0
        background_constraint_output_rows_checked = 0
        background_constraint_feasible_output_row_count = 0
        background_constraint_blocked_output_row_count = 0
        background_constraint_ranks: list[int] = []
        background_nullspace_dimensions: list[int] = []
        background_rank_tolerances: list[float] = []
        for observed_rows, target_ids in sorted(
            support_groups.items(),
            key=lambda item: (len(item[0]), item[0], item[1][0]),
        ):
            target_index = torch.tensor(
                target_ids,
                dtype=torch.long,
                device=requested_device,
            )
            if not observed_rows:
                empty_output_row_count += len(target_ids)
                continue
            observed_index = torch.tensor(
                observed_rows,
                dtype=torch.long,
                device=requested_device,
            )
            selected_hidden = hidden.index_select(0, observed_index)
            selected_residual = residual.index_select(
                0,
                observed_index,
            ).index_select(1, target_index)
            solve_hidden = selected_hidden
            nullspace_basis = None
            if sealed_background:
                group_support = support[:, target_ids[0]]
                unsupported_hidden = hidden[~group_support]
                background_constraint_group_count += 1
                background_constraint_output_rows_checked += len(target_ids)
                if int(unsupported_hidden.shape[0]) == 0:
                    constraint_rank = 0
                    rank_tolerance = 0.0
                    nullspace_basis = torch.eye(
                        hidden_size,
                        dtype=hidden.dtype,
                        device=hidden.device,
                    )
                else:
                    _left, singular_values, right_vectors = torch.linalg.svd(
                        unsupported_hidden,
                        full_matrices=True,
                    )
                    rank_tolerance = float(
                        max(int(unsupported_hidden.shape[0]), hidden_size)
                        * torch.finfo(unsupported_hidden.dtype).eps
                        * float(singular_values.max().item())
                    )
                    constraint_rank = int(
                        (singular_values > rank_tolerance).sum().item()
                    )
                    nullspace_basis = right_vectors[constraint_rank:].t()
                nullspace_dimension = int(nullspace_basis.shape[1])
                background_constraint_ranks.append(constraint_rank)
                background_nullspace_dimensions.append(nullspace_dimension)
                background_rank_tolerances.append(rank_tolerance)
                if nullspace_dimension == 0:
                    background_constraint_blocked_output_row_count += len(
                        target_ids
                    )
                    continue
                background_constraint_feasible_output_row_count += len(
                    target_ids
                )
                solve_hidden = selected_hidden.matmul(nullspace_basis)
            observed_count = len(observed_rows)
            solve_feature_count = int(solve_hidden.shape[1])
            if observed_count <= solve_feature_count:
                dual_group_count += 1
                maximum_group_gram_dimension = max(
                    maximum_group_gram_dimension,
                    observed_count,
                )
                gram = solve_hidden.matmul(solve_hidden.t())
                gram = gram + float(ridge) * torch.eye(
                    observed_count,
                    dtype=gram.dtype,
                    device=gram.device,
                )
                dual = torch.linalg.solve(gram, solve_hidden)
                selected_delta = selected_residual.t().matmul(dual)
            else:
                primal_group_count += 1
                maximum_group_gram_dimension = max(
                    maximum_group_gram_dimension,
                    solve_feature_count,
                )
                gram = solve_hidden.t().matmul(solve_hidden)
                gram = gram + float(ridge) * torch.eye(
                    solve_feature_count,
                    dtype=gram.dtype,
                    device=gram.device,
                )
                solved = torch.linalg.solve(
                    gram,
                    solve_hidden.t().matmul(selected_residual),
                )
                selected_delta = solved.t()
            if nullspace_basis is not None:
                selected_delta = selected_delta.matmul(nullspace_basis.t())
            delta.index_copy_(0, target_index, selected_delta)
        solve_formulation = (
            "grouped-masked-background-nullspace-ridge"
            if sealed_background
            else "grouped-masked-per-output-ridge"
        )
        compiled = structural + scale * delta
        if protected_index is not None:
            compiled.index_copy_(
                0,
                protected_index,
                structural.index_select(0, protected_index),
            )
        if norm_scale > 0.0:
            if sealed_background:
                # Scaling the whole compiled row would add a structural-table
                # component to the correction and break the nullspace seal.
                # Shrinking only the delta keeps every background equality.
                compiled = structural + (1.0 - norm_scale) * (
                    compiled - structural
                )
                row_norm_calibration_formulation = (
                    "background-preserving-delta-contraction"
                )
            else:
                structural_norm = torch.linalg.vector_norm(structural, dim=1)
                compiled_norm = torch.linalg.vector_norm(compiled, dim=1)
                calibrated_norm = (
                    (1.0 - norm_scale) * compiled_norm
                    + norm_scale * structural_norm
                )
                safe_scale = torch.where(
                    compiled_norm > 0.0,
                    calibrated_norm / compiled_norm.clamp_min(1e-12),
                    torch.ones_like(compiled_norm),
                )
                compiled = compiled * safe_scale[:, None]
                zero_rows = (compiled_norm <= 0.0) & (structural_norm > 0.0)
                if bool(zero_rows.any()):
                    compiled[zero_rows] = structural[zero_rows]
                row_norm_calibration_formulation = "whole-row-norm-interpolation"
        else:
            row_norm_calibration_formulation = "disabled"
        if protected_index is not None:
            compiled.index_copy_(
                0,
                protected_index,
                structural.index_select(0, protected_index),
            )
        structural_target_table = structural_output_table.detach().to(
            device="cpu",
        ).contiguous()
        result_table = compiled.to(
            device="cpu",
            dtype=structural_output_table.dtype,
        ).contiguous()
        target_dtype_background_reverted_output_row_count = 0
        if sealed_background:
            projected_structural = structural_target_table.to(
                device=requested_device,
                dtype=torch.float32,
            )
            projected_compiled = result_table.to(
                device=requested_device,
                dtype=torch.float32,
            )
            for observed_rows, target_ids in sorted(
                support_groups.items(),
                key=lambda item: (len(item[0]), item[0], item[1][0]),
            ):
                if not observed_rows:
                    continue
                group_support = support[:, target_ids[0]]
                unsupported_hidden = hidden[~group_support]
                if int(unsupported_hidden.shape[0]) == 0:
                    continue
                target_index = torch.tensor(
                    target_ids,
                    dtype=torch.long,
                    device=requested_device,
                )
                baseline_logits = unsupported_hidden.matmul(
                    projected_structural.index_select(0, target_index).t()
                )
                compiled_logits = unsupported_hidden.matmul(
                    projected_compiled.index_select(0, target_index).t()
                )
                violating = (compiled_logits != baseline_logits).any(dim=0)
                if bool(violating.any()):
                    violating_index = target_index[violating]
                    violating_index_cpu = violating_index.to(device="cpu")
                    result_table.index_copy_(
                        0,
                        violating_index_cpu,
                        structural_target_table.index_select(
                            0,
                            violating_index_cpu,
                        ),
                    )
                    projected_compiled.index_copy_(
                        0,
                        violating_index,
                        projected_structural.index_select(0, violating_index),
                    )
                    target_dtype_background_reverted_output_row_count += int(
                        violating.sum().item()
                    )
        projected_result = result_table.to(
            device=requested_device,
            dtype=torch.float32,
        )
        final_prediction = hidden.matmul(projected_result.t())
        before_supported = torch.where(
            support,
            structural_prediction,
            effective_target,
        )
        after_supported = torch.where(
            support,
            final_prediction,
            effective_target,
        )
        before_error = _relative_centered_error(
            before_supported,
            effective_target,
        )
        after_error = _relative_centered_error(
            after_supported,
            effective_target,
        )
        unsupported_drift_values = (
            final_prediction - structural_prediction
        )[~support]
        unsupported_drift_rms = (
            0.0
            if unsupported_drift_values.numel() == 0
            else float(unsupported_drift_values.square().mean().sqrt().item())
        )
        unsupported_drift_max_abs = (
            0.0
            if unsupported_drift_values.numel() == 0
            else float(unsupported_drift_values.abs().max().item())
        )
        if sealed_background and bool(
            torch.count_nonzero(unsupported_drift_values).item()
        ):
            raise Lfm2InterfaceFrontierError(
                "Target-dtype causal head violated its sealed background logits."
            )
        output_row_has_support = effective_support_cpu.any(dim=0)
        output_row_equals_structural = (
            result_table == structural_target_table
        ).all(dim=1)
        fully_unsupported_output_rows_equal_structural_baseline = bool(
            output_row_equals_structural[~output_row_has_support].all().item()
        )
        if not fully_unsupported_output_rows_equal_structural_baseline:
            raise Lfm2InterfaceFrontierError(
                "A vocabulary row without contextual support lost its structural baseline."
            )
        contextually_supported_output_row_count = int(
            output_row_has_support.sum().item()
        )
        contextual_correction_output_row_count = int(
            (output_row_has_support & ~output_row_equals_structural).sum().item()
        )
        structural_baseline_output_row_count = int(
            output_row_equals_structural.sum().item()
        )
    if not bool(torch.isfinite(result_table).all()):
        raise Lfm2InterfaceFrontierError(
            "Causal residual solve produced NaN or infinity."
        )
    output_sha256 = _tensor_sha256(result_table)
    background_constraint_sha256 = None
    if sealed_background:
        background_constraint_sha256 = canonical_sha256(
            {
                "policy": unsupported_logit_policy,
                "child_hidden_rows_sha256": child_hidden_sha256,
                "effective_target_logit_support_mask_sha256": (
                    effective_support_sha256
                ),
                "structural_output_table_sha256": structural_sha256,
                "output_table_sha256": output_sha256,
                "unsupported_logit_drift_rms": unsupported_drift_rms,
                "unsupported_logit_drift_max_abs": unsupported_drift_max_abs,
            }
        )
    digests = {
        "parent_terminal_evidence_sha256": parent_sha256,
        "cache_key": trie_receipt["cache_key"],
        "parent_identity_sha256": trie_receipt["parent_identity_sha256"],
        "selective_session_sha256": trie_receipt["selective_session_sha256"],
        "manifest_sha256": trie_receipt["manifest_sha256"],
        "completion_sha256": trie_receipt["completion_sha256"],
        "semantic_alignment_sha256": trie_receipt[
            "semantic_alignment_sha256"
        ],
        "tokenizer_live_session_sha256": trie_receipt[
            "tokenizer_live_session_sha256"
        ],
        "child_feature_evidence_sha256": child_value["evidence_sha256"],
        "trie_conditional_evidence_sha256": trie_receipt["result_sha256"],
        "contextual_transition_plan_sha256": trie_receipt[
            "contextual_transition_plan_sha256"
        ],
        "causal_root_rows_sha256": trie_receipt["causal_root_rows_sha256"],
        "bridge_artifact_sha256": trie_receipt["bridge_artifact_sha256"],
        "bridge_profile_sha256": trie_receipt["bridge_profile_sha256"],
        "donor_tokenizer_semantics_sha256": trie_receipt[
            "donor_tokenizer_semantics_sha256"
        ],
        "target_tokenizer_semantics_sha256": trie_receipt[
            "target_tokenizer_semantics_sha256"
        ],
        "causal_input_table_sha256": child_value[
            "causal_input_table_sha256"
        ],
        "causal_body_inventory_sha256": child_value[
            "causal_body_inventory_sha256"
        ],
        "child_model_state_sha256": child_value[
            "child_model_state_before_sha256"
        ],
        "sequence_plan_sha256": sequence_plan_sha256,
    }
    source_body = {
        "schema": "neural-forge-lfm2-causal-residual-source-binding-v2",
        **digests,
        "target_prefix_rows_sha256": trie_receipt["target_prefix_rows_sha256"],
        "child_hidden_rows_sha256": child_hidden_sha256,
    }
    source_sha256 = canonical_sha256(source_body)
    body = {
        "schema": LFM2_CAUSAL_RESIDUAL_TOKEN_BRIDGE_SCHEMA,
        "status": "dense-cache-only-contextual-causal-residual-bridge-compiled",
        **digests,
        "source_binding_sha256": source_sha256,
        "context_free_inverse_output_plan_used": False,
        "structural_output_table_sha256": structural_sha256,
        "child_hidden_rows_sha256": child_hidden_sha256,
        "target_prefix_rows_sha256": trie_receipt["target_prefix_rows_sha256"],
        "target_logits_sha256": trie_receipt["target_logits_sha256"],
        "target_logit_support_mask_sha256": original_support_sha256,
        "effective_target_logit_support_mask_sha256": effective_support_sha256,
        "output_table_sha256": output_sha256,
        "evidence_row_count": rows,
        "hidden_size": hidden_size,
        "target_row_count": target_row_count,
        "vocabulary_row_count": target_row_count,
        "vocabulary_row_coverage_complete": True,
        "all_vocabulary_rows_classified": True,
        "all_vocabulary_rows_have_structural_or_contextual_authority": True,
        "contextually_supported_output_row_count": (
            contextually_supported_output_row_count
        ),
        "fully_unsupported_output_row_count": empty_output_row_count,
        "contextual_correction_output_row_count": (
            contextual_correction_output_row_count
        ),
        "supported_output_rows_retaining_structural_baseline_count": (
            contextually_supported_output_row_count
            - contextual_correction_output_row_count
        ),
        "structural_baseline_output_row_count_after_target_dtype_projection": (
            structural_baseline_output_row_count
        ),
        "fully_unsupported_output_rows_equal_structural_baseline": (
            fully_unsupported_output_rows_equal_structural_baseline
        ),
        "input_supported_logit_cell_count": int(
            target_logit_support_mask.sum().item()
        ),
        "effective_supported_logit_cell_count": int(
            effective_support_cpu.sum().item()
        ),
        "rows_without_effective_support": int(
            (~effective_support_cpu.any(dim=1)).sum().item()
        ),
        "protected_support_cells_removed_before_centering": (
            protected_support_cells_removed
        ),
        "solve_formulation": solve_formulation,
        "support_mask_group_count": len(support_groups),
        "nonempty_support_mask_group_count": (
            len(support_groups) - int(() in support_groups)
        ),
        "empty_observation_output_row_count": empty_output_row_count,
        "dual_support_mask_group_count": dual_group_count,
        "primal_support_mask_group_count": primal_group_count,
        "maximum_group_gram_dimension": maximum_group_gram_dimension,
        "group_formulation_selected_by_observed_rows_vs_hidden_size": True,
        "unsupported_logit_policy": unsupported_logit_policy,
        "background_constraint_kind": (
            "exact-unsupported-child-logits-via-nullspace-and-target-dtype-seal"
            if sealed_background
            else "none"
        ),
        "background_constraint_applied": sealed_background,
        "background_constraint_sha256": background_constraint_sha256,
        "background_constraint_group_count": background_constraint_group_count,
        "background_constraint_output_rows_checked": (
            background_constraint_output_rows_checked
        ),
        "background_constraint_feasible_output_row_count": (
            background_constraint_feasible_output_row_count
        ),
        "background_constraint_blocked_output_row_count": (
            background_constraint_blocked_output_row_count
        ),
        "background_constraint_rank_min": (
            min(background_constraint_ranks)
            if background_constraint_ranks
            else None
        ),
        "background_constraint_rank_max": (
            max(background_constraint_ranks)
            if background_constraint_ranks
            else None
        ),
        "background_nullspace_dimension_min": (
            min(background_nullspace_dimensions)
            if background_nullspace_dimensions
            else None
        ),
        "background_nullspace_dimension_max": (
            max(background_nullspace_dimensions)
            if background_nullspace_dimensions
            else None
        ),
        "background_rank_tolerance_max": (
            max(background_rank_tolerances)
            if background_rank_tolerances
            else None
        ),
        "target_dtype_background_reverted_output_row_count": (
            target_dtype_background_reverted_output_row_count
        ),
        "row_norm_calibration_formulation": (
            row_norm_calibration_formulation
        ),
        "unsupported_cells_included_as_zero_residual_observations": False,
        "each_output_row_fit_only_on_supported_feature_rows": True,
        "unsupported_cells_exactly_preserved_claimed": sealed_background,
        "unsupported_drift_measured_after_target_dtype_projection": True,
        "unsupported_logit_drift_rms": unsupported_drift_rms,
        "unsupported_logit_drift_max_abs": unsupported_drift_max_abs,
        "ridge": float(ridge),
        "residual_scale_ppm": residual_scale_ppm,
        "row_norm_calibration_ppm": row_norm_calibration_ppm,
        "protected_target_ids": list(protected),
        "relative_centered_error_before": before_error,
        "relative_centered_error_after": after_error,
        "diagnostic_metrics_are_not_acceptance_gates": True,
        "diagnostic_metrics_used_as_compilation_gate": False,
        "manual_behavior_scores_used_as_compilation_gate": False,
        "behavior_retention_goal_is_not_a_structural_cutoff": True,
        "causal_child_residual_features_used": True,
        "compiled_as_independent_token_lookup": False,
        "compiled_against_parent_hidden_features": False,
        "materialized_runtime_operator": "ordinary-dense-output-table",
        "runtime_parent_or_adapter_required": False,
        "bounded_partial_contextual_conditionals_transported": True,
        "exact_all_branch_autoregressive_equivalence_claimed": False,
        "missing_cache_statistics": trie_receipt["missing_cache_statistics"],
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "network_access": False,
        },
    }
    return Lfm2CausalResidualTokenBridge(
        table=result_table,
        receipt={**body, "result_sha256": canonical_sha256(body)},
    )


def _untie_zero_blank(model: Any) -> dict[str, Any]:
    torch = _torch()
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    if input_weight is not output_weight or not bool(
        getattr(model.config, "tie_word_embeddings", False)
    ):
        raise Lfm2InterfaceFrontierError(
            "Untied output conversion requires a native tied zero blank."
        )
    if int(torch.count_nonzero(input_weight.detach()).item()) != 0:
        raise Lfm2InterfaceFrontierError("Blank input/output table is not zero.")
    before = fingerprint_model_state(model)
    vocabulary, hidden = (int(value) for value in input_weight.shape)
    head = torch.nn.Linear(
        hidden,
        vocabulary,
        bias=False,
        device=input_weight.device,
        dtype=input_weight.dtype,
    )
    with torch.no_grad():
        head.weight.zero_()
    model.config.tie_word_embeddings = False
    setter = getattr(model, "set_output_embeddings", None)
    if not callable(setter):
        raise Lfm2InterfaceFrontierError(
            "LFM2 model cannot install an independent output operator."
        )
    setter(head)
    installed_output = model.get_output_embeddings().weight
    if (
        installed_output is input_weight
        or _storage_identity(installed_output) == _storage_identity(input_weight)
        or int(torch.count_nonzero(installed_output.detach()).item()) != 0
        or bool(getattr(model.config, "tie_word_embeddings", True))
    ):
        raise Lfm2InterfaceFrontierError(
            "Compiler-owned untied output blank was not materialized."
        )
    return {
        "operation": "compiler-owned-zero-output-untie",
        "base_blank_state_sha256": before,
        "untied_blank_state_sha256": fingerprint_model_state(model),
        "added_dense_parameters": vocabulary * hidden,
        "input_output_storage_distinct": True,
        "all_added_parameters_zero": True,
    }


def fold_lfm2_interface_candidate(
    parent_model: Any,
    blank_target_model: Any,
    *,
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    blank_target_receipt: Mapping[str, Any],
    recipe: Lfm2InterfaceRecipe,
    maximum_target_unique_parameters: int,
    contexts: Sequence[str] = DEFAULT_INTERFACE_CONTEXTS,
    cached_logit_residual_head: Lfm2CachedLogitResidualHead | None = None,
    causal_residual_token_bridge: Lfm2CausalResidualTokenBridge | None = None,
    precompiled_sequence_plan: Mapping[str, Any] | None = None,
    precompiled_inverse_output_plan: Mapping[str, Any] | None = None,
    inverse_output_fallback_target_id: int | None = None,
) -> Lfm2InterfaceFrontierResult:
    """Install one exact-body compositional/asymmetric interface candidate."""

    torch = _torch()
    if parent_model is blank_target_model:
        raise Lfm2InterfaceFrontierError(
            "Parent and blank child must be distinct model objects."
        )
    if not isinstance(recipe, Lfm2InterfaceRecipe):
        raise TypeError("recipe must be Lfm2InterfaceRecipe.")
    if (recipe.output_mode == "cached-logit-residual") != isinstance(
        cached_logit_residual_head,
        Lfm2CachedLogitResidualHead,
    ):
        raise Lfm2InterfaceFrontierError(
            "Cached-logit output mode requires exactly one sealed residual head."
        )
    causal_output_mode = recipe.output_mode in {
        "causal-residual-token-bridge",
        "causal-residual-token-bridge-tied",
    }
    if causal_output_mode != isinstance(
        causal_residual_token_bridge,
        Lfm2CausalResidualTokenBridge,
    ):
        raise Lfm2InterfaceFrontierError(
            "Causal-residual output mode requires exactly one sealed token bridge."
        )
    if (
        isinstance(maximum_target_unique_parameters, bool)
        or not isinstance(maximum_target_unique_parameters, int)
        or maximum_target_unique_parameters <= 0
    ):
        raise Lfm2InterfaceFrontierError(
            "maximum_target_unique_parameters must be positive."
        )
    base_blank_sha = _verify_base_blank(blank_target_receipt)
    bridge.verify_live_tokenizers(donor_tokenizer, target_tokenizer)
    parent_topology = _config_topology(parent_model.config)
    initial_target_topology = _config_topology(blank_target_model.config)
    if (
        parent_topology["model_type"] != "lfm2"
        or initial_target_topology["model_type"] != "lfm2"
        or _body_topology(parent_topology) != _body_topology(initial_target_topology)
        or parent_topology["tie_word_embeddings"] is not True
        or initial_target_topology["tie_word_embeddings"] is not True
    ):
        raise Lfm2InterfaceFrontierError(
            "Interface frontier requires an exact-body native-tied source blank."
        )
    if any(
        value.layout != torch.strided
        or int(torch.count_nonzero(value.detach()).item()) != 0
        for value in blank_target_model.parameters()
    ):
        raise Lfm2InterfaceFrontierError(
            "Compiler-authenticated blank contains nonzero or non-dense parameters."
        )

    base_blank_state = fingerprint_model_state(blank_target_model)
    untie_receipt = None
    if recipe.untied_output:
        untie_receipt = _untie_zero_blank(blank_target_model)
    target_topology = _config_topology(blank_target_model.config)
    if target_topology["tie_word_embeddings"] == recipe.untied_output:
        raise Lfm2InterfaceFrontierError(
            "Candidate output tie differs from its requested recipe."
        )

    profile = bridge.transport_profile
    parent_input = parent_model.get_input_embeddings().weight
    parent_output = parent_model.get_output_embeddings().weight
    target_input = blank_target_model.get_input_embeddings().weight
    target_output = blank_target_model.get_output_embeddings().weight
    if (
        int(parent_input.shape[0]) != profile.donor_row_capacity
        or int(parent_output.shape[0]) != profile.donor_row_capacity
        or int(target_input.shape[0]) != profile.target_row_capacity
        or int(target_output.shape[0]) != profile.target_row_capacity
        or int(parent_input.shape[1]) != int(target_input.shape[1])
        or int(parent_output.shape[1]) != int(target_output.shape[1])
    ):
        raise Lfm2InterfaceFrontierError(
            "Bridge capacities do not address the concrete interface matrices."
        )

    parent_parameters = _unique_parameter_count(parent_model)
    target_parameters = _unique_parameter_count(blank_target_model)
    hidden = int(target_input.shape[1])
    body_parameters = parent_parameters - profile.donor_row_capacity * hidden
    expected_base = body_parameters + profile.target_row_capacity * hidden
    expected_target = expected_base + (
        profile.target_row_capacity * hidden if recipe.untied_output else 0
    )
    if (
        blank_target_receipt.get("parent_unique_parameters") != parent_parameters
        or blank_target_receipt.get("target_unique_parameters") != expected_base
        or target_parameters != expected_target
        or target_parameters > maximum_target_unique_parameters
        or target_parameters >= parent_parameters
    ):
        raise Lfm2InterfaceFrontierError(
            "Candidate does not satisfy its authenticated target-size envelope "
            f"(receipt_parent={blank_target_receipt.get('parent_unique_parameters')}, "
            f"parent={parent_parameters}, "
            f"receipt_base={blank_target_receipt.get('target_unique_parameters')}, "
            f"expected_base={expected_base}, observed={target_parameters}, "
            f"expected={expected_target}, maximum={maximum_target_unique_parameters})."
        )

    parent_interface_ids = _interface_storage_ids(parent_model)
    target_interface_ids = _interface_storage_ids(blank_target_model)
    parent_named = dict(_all_named_tensors(parent_model))
    target_named = dict(_all_named_tensors(blank_target_model))
    parent_body_names = {
        name
        for name, value in parent_named.items()
        if _storage_identity(value) not in parent_interface_ids
    }
    target_body_names = {
        name
        for name, value in target_named.items()
        if _storage_identity(value) not in target_interface_ids
    }
    if parent_body_names != target_body_names:
        raise Lfm2InterfaceFrontierError(
            "Parent and target non-interface inventories differ."
        )
    for name in sorted(parent_body_names):
        if tuple(parent_named[name].shape) != tuple(target_named[name].shape):
            raise Lfm2InterfaceFrontierError(
                f"Body tensor shape differs for {name!r}."
            )
    source_body_records = _body_records(
        parent_model,
        interface_storage_ids=parent_interface_ids,
    )
    source_body_inventory_sha256 = canonical_sha256(list(source_body_records))

    plan = (
        compile_lfm2_interface_sequence_plan(
            bridge,
            donor_tokenizer,
            target_tokenizer,
            contexts=contexts,
        )
        if precompiled_sequence_plan is None
        else _verify_precompiled_plan(
            precompiled_sequence_plan,
            schema=LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA,
            bridge=bridge,
        )
    )
    input_table = _compile_context_table(
        parent_input,
        plan,
        strict_anchor_ppm=recipe.input_strict_anchor_ppm,
        preserve_anchor_norm=recipe.input_preserve_anchor_norm,
    ).to(device=target_input.device, dtype=target_input.dtype)
    mapping = tuple(int(value) for value in profile.target_to_donor_row_indices)
    inverse_plan = None
    if recipe.output_mode == "tied-to-input":
        output_table = input_table
    elif recipe.output_mode == "strict-row":
        output_table = parent_output.detach().index_select(
            0,
            torch.tensor(mapping, dtype=torch.long, device=parent_output.device),
        ).to(device=target_output.device, dtype=target_output.dtype)
    elif recipe.output_mode == "context-envelope":
        output_table = _compile_context_table(
            parent_output,
            plan,
            strict_anchor_ppm=recipe.output_strict_anchor_ppm,
            preserve_anchor_norm=recipe.output_preserve_anchor_norm,
        ).to(device=target_output.device, dtype=target_output.dtype)
    elif recipe.output_mode == "inverse-bucket-centroid":
        inverse_plan = (
            compile_lfm2_inverse_output_plan(
                bridge,
                donor_tokenizer,
                target_tokenizer,
                fallback_target_id=inverse_output_fallback_target_id,
            )
            if precompiled_inverse_output_plan is None
            else _verify_precompiled_plan(
                precompiled_inverse_output_plan,
                schema=LFM2_INTERFACE_INVERSE_OUTPUT_PLAN_SCHEMA,
                bridge=bridge,
            )
        )
        output_table = compile_lfm2_inverse_bucket_output_table(
            parent_output,
            inverse_plan,
            mapping,
            strict_anchor_ppm=recipe.output_strict_anchor_ppm,
            preserve_anchor_norm=recipe.output_preserve_anchor_norm,
            weighting=recipe.output_bucket_weighting,
        ).to(device=target_output.device, dtype=target_output.dtype)
    elif recipe.output_mode == "prefix-transition-centroid":
        inverse_plan = (
            compile_lfm2_inverse_output_plan(
                bridge,
                donor_tokenizer,
                target_tokenizer,
                fallback_target_id=inverse_output_fallback_target_id,
            )
            if precompiled_inverse_output_plan is None
            else _verify_precompiled_plan(
                precompiled_inverse_output_plan,
                schema=LFM2_INTERFACE_INVERSE_OUTPUT_PLAN_SCHEMA,
                bridge=bridge,
            )
        )
        output_table = compile_lfm2_prefix_transition_output_table(
            parent_output,
            inverse_plan,
            mapping,
            strict_anchor_ppm=recipe.output_strict_anchor_ppm,
            preserve_anchor_norm=recipe.output_preserve_anchor_norm,
        ).to(device=target_output.device, dtype=target_output.dtype)
    elif recipe.output_mode == "cached-logit-residual":
        assert cached_logit_residual_head is not None
        inverse_plan = (
            compile_lfm2_inverse_output_plan(
                bridge,
                donor_tokenizer,
                target_tokenizer,
                fallback_target_id=inverse_output_fallback_target_id,
            )
            if precompiled_inverse_output_plan is None
            else _verify_precompiled_plan(
                precompiled_inverse_output_plan,
                schema=LFM2_INTERFACE_INVERSE_OUTPUT_PLAN_SCHEMA,
                bridge=bridge,
            )
        )
        structural_output_table = compile_lfm2_inverse_bucket_output_table(
            parent_output,
            inverse_plan,
            mapping,
            strict_anchor_ppm=recipe.output_strict_anchor_ppm,
            preserve_anchor_norm=recipe.output_preserve_anchor_norm,
            weighting=recipe.output_bucket_weighting,
        )
        residual_receipt = cached_logit_residual_head.as_dict()
        if (
            residual_receipt.get("inverse_output_plan_sha256")
            != inverse_plan["plan_sha256"]
            or residual_receipt.get("structural_output_table_sha256")
            != _tensor_sha256(structural_output_table)
            or tuple(cached_logit_residual_head.table.shape)
            != tuple(target_output.shape)
        ):
            raise Lfm2InterfaceFrontierError(
                "Cached-logit residual head differs from this bridge or source."
            )
        output_table = cached_logit_residual_head.table.to(
            device=target_output.device,
            dtype=target_output.dtype,
        )
    else:
        assert causal_residual_token_bridge is not None
        # Causal-v2 starts from the same contextual interface coordinate that
        # produced the child features. The context-free inverse first-bucket
        # plan remains available only to the historical ablations above.
        structural_output_table = input_table.detach().to(
            device="cpu"
        ).contiguous()
        causal_receipt = causal_residual_token_bridge.as_dict()
        if (
            causal_receipt.get("context_free_inverse_output_plan_used") is not False
            or causal_receipt.get("bridge_artifact_sha256")
            != bridge.artifact_sha256
            or causal_receipt.get("bridge_profile_sha256")
            != bridge.profile_fingerprint
            or causal_receipt.get("donor_tokenizer_semantics_sha256")
            != bridge.donor_tokenizer_semantics_sha256
            or causal_receipt.get("target_tokenizer_semantics_sha256")
            != bridge.target_tokenizer_semantics_sha256
            or causal_receipt.get("causal_input_table_sha256")
            != _tensor_sha256(input_table)
            or causal_receipt.get("causal_body_inventory_sha256")
            != source_body_inventory_sha256
            or causal_receipt.get("sequence_plan_sha256")
            != plan["plan_sha256"]
            or causal_receipt.get("structural_output_table_sha256")
            != _tensor_sha256(structural_output_table)
            or tuple(causal_residual_token_bridge.table.shape)
            != tuple(target_output.shape)
        ):
            raise Lfm2InterfaceFrontierError(
                "Causal residual token bridge differs from this bridge or source."
            )
        output_table = causal_residual_token_bridge.table.to(
            device=target_output.device,
            dtype=target_output.dtype,
        )
        if recipe.output_mode == "causal-residual-token-bridge-tied":
            input_table = output_table
    if (
        inverse_plan is not None
        and inverse_output_fallback_target_id is not None
        and inverse_plan.get("fallback_target_id")
        != inverse_output_fallback_target_id
    ):
        raise Lfm2InterfaceFrontierError(
            "Precompiled inverse-output fallback differs from the request."
        )
    if tuple(input_table.shape) != tuple(target_input.shape) or tuple(
        output_table.shape
    ) != tuple(target_output.shape):
        raise Lfm2InterfaceFrontierError(
            "Compiled interface table shape differs from the target."
        )

    parent_state_before = fingerprint_model_state(parent_model)
    input_sha256 = _tensor_sha256(input_table)
    output_sha256 = _tensor_sha256(output_table)
    with torch.no_grad():
        for name in sorted(target_body_names):
            target_named[name].copy_(
                parent_named[name].detach().to(
                    device=target_named[name].device,
                    dtype=target_named[name].dtype,
                )
            )
        target_input.copy_(input_table)
        if recipe.untied_output:
            target_output.copy_(output_table)
    parent_state_after = fingerprint_model_state(parent_model)
    installed_state = fingerprint_model_state(blank_target_model)
    source_body = source_body_records
    installed_body = _body_records(
        blank_target_model,
        interface_storage_ids=target_interface_ids,
    )
    if (
        source_body != installed_body
        or parent_state_after != parent_state_before
        or installed_state == base_blank_state
        or _tensor_sha256(target_input) != input_sha256
        or _tensor_sha256(target_output) != output_sha256
        or (target_input is target_output) == recipe.untied_output
        or _unique_parameter_count(blank_target_model) != target_parameters
    ):
        raise Lfm2InterfaceFrontierError(
            "Installed interface candidate failed structural replay."
        )

    recipe_document = recipe.as_dict()
    plan_summary = {
        key: value
        for key, value in plan.items()
        if key != "records"
    }
    inverse_plan_summary = (
        None
        if inverse_plan is None
        else {
            key: value
            for key, value in inverse_plan.items()
            if key
            not in {
                "buckets",
                "donor_target_decompositions",
                "prefix_transition_buckets",
            }
        }
    )
    body = {
        "schema": LFM2_INTERFACE_FRONTIER_SCHEMA,
        "status": "standalone-exact-body-interface-candidate-installed",
        "candidate_id": recipe.candidate_id,
        "recipe": recipe_document,
        "recipe_sha256": canonical_sha256(recipe_document),
        "diagnostic_role": (
            "automatic-context-compositional-and-asymmetric-interface-frontier"
        ),
        "interface_isolation_control": True,
        "final_four_x_topology": False,
        "reason_not_final_four_x_topology": (
            "exact parent body is retained to isolate the token interface"
        ),
        "parent_topology": parent_topology,
        "target_topology": target_topology,
        "exact_parent_body_depth_width_preserved": True,
        "body_transport": "exact-name-and-shape-identity-transport",
        "body_tensor_records": list(installed_body),
        "body_tensor_inventory_sha256": canonical_sha256(list(installed_body)),
        "body_tensor_count": len(installed_body),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "sequence_plan": plan_summary,
        "sequence_plan_sha256": plan["plan_sha256"],
        "inverse_output_plan": inverse_plan_summary,
        "inverse_output_plan_sha256": (
            None if inverse_plan is None else inverse_plan["plan_sha256"]
        ),
        "cached_logit_residual_head": (
            None
            if cached_logit_residual_head is None
            else cached_logit_residual_head.as_dict()
        ),
        "causal_residual_token_bridge": (
            None
            if causal_residual_token_bridge is None
            else causal_residual_token_bridge.as_dict()
        ),
        "exact_autoregressive_tokenizer_bridge_claimed": False,
        "prefix_conditional_child_feature_rows_used": (
            causal_residual_token_bridge is not None
        ),
        "causal_feature_coordinate_mode": (
            None
            if causal_residual_token_bridge is None
            else (
                "tied-shared-table-perturbative-ablation"
                if recipe.output_mode == "causal-residual-token-bridge-tied"
                else "untied-output-exact-causal-coordinate"
            )
        ),
        "one_shot_tied_causal_self_consistency_claimed": False,
        "postinstall_tied_feature_drift_audit_required": (
            recipe.output_mode == "causal-residual-token-bridge-tied"
        ),
        "input_table_sha256": input_sha256,
        "output_table_sha256": output_sha256,
        "native_input_output_tying_preserved": not recipe.untied_output,
        "compiler_owned_untied_blank": untie_receipt,
        "parent_unique_parameters": parent_parameters,
        "base_tied_target_unique_parameters": expected_base,
        "target_unique_parameters": target_parameters,
        "maximum_target_unique_parameters": maximum_target_unique_parameters,
        "target_size_envelope_satisfied": True,
        "realized_parameter_compaction_ratio": parent_parameters / target_parameters,
        "base_blank_report_sha256": base_blank_sha,
        "base_blank_model_state_sha256": base_blank_state,
        "installed_model_state_sha256": installed_state,
        "parent_model_state_before_sha256": parent_state_before,
        "parent_model_state_after_sha256": parent_state_after,
        "parent_state_preserved": True,
        "standalone_save_reload_pending": True,
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "child_forward_calls": 0,
            "child_generate_calls": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "external_adapter_parameters": 0,
            "parent_sidecar_parameters": 0,
            "runtime_shared_parent_parameters": 0,
            "network_access": False,
        },
    }
    return Lfm2InterfaceFrontierResult(
        model=blank_target_model,
        receipt={**body, "result_sha256": canonical_sha256(body)},
    )


def target_unique_parameters_for_recipe(
    *,
    base_tied_target_unique_parameters: int,
    target_vocabulary_size: int,
    hidden_size: int,
    recipe: Lfm2InterfaceRecipe,
) -> int:
    """Return the exact standalone dense count before materialization."""

    for value, name in (
        (base_tied_target_unique_parameters, "base_tied_target_unique_parameters"),
        (target_vocabulary_size, "target_vocabulary_size"),
        (hidden_size, "hidden_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Lfm2InterfaceFrontierError(f"{name} must be positive.")
    if not isinstance(recipe, Lfm2InterfaceRecipe):
        raise TypeError("recipe must be Lfm2InterfaceRecipe.")
    return base_tied_target_unique_parameters + (
        target_vocabulary_size * hidden_size if recipe.untied_output else 0
    )


def base_tied_target_budget_for_recipe(
    *,
    maximum_total_target_unique_parameters: int,
    target_vocabulary_size: int,
    hidden_size: int,
    recipe: Lfm2InterfaceRecipe,
) -> int:
    """Reserve interface parameters before a dynamic body planner runs."""

    for value, name in (
        (
            maximum_total_target_unique_parameters,
            "maximum_total_target_unique_parameters",
        ),
        (target_vocabulary_size, "target_vocabulary_size"),
        (hidden_size, "hidden_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Lfm2InterfaceFrontierError(f"{name} must be positive.")
    if not isinstance(recipe, Lfm2InterfaceRecipe):
        raise TypeError("recipe must be Lfm2InterfaceRecipe.")
    reserved = (
        target_vocabulary_size * hidden_size if recipe.untied_output else 0
    )
    available = maximum_total_target_unique_parameters - reserved
    if available <= 0:
        raise Lfm2InterfaceFrontierError(
            "Target budget cannot materialize the requested dense interface."
        )
    return available


def plan_lfm2_dense_untied_target_rebalance(
    *,
    parent_unique_parameters: int,
    base_tied_target_unique_parameters: int,
    target_vocabulary_size: int,
    hidden_size: int,
    layer_count: int,
    current_intermediate_size: int,
    target_compaction_denominator: int,
    minimum_intermediate_size: int = 1,
) -> dict[str, Any]:
    """Reserve an untied head and reduce FFN width to a total-size ceiling.

    LFM2 has three dense hidden/intermediate matrices per block.  This hook
    leaves residual width, attention geometry, vocabulary, and depth fixed,
    and computes the smallest integer FFN-width reduction that makes the
    independently materialized output head fit the requested parameter ratio.
    """

    values = (
        (parent_unique_parameters, "parent_unique_parameters"),
        (
            base_tied_target_unique_parameters,
            "base_tied_target_unique_parameters",
        ),
        (target_vocabulary_size, "target_vocabulary_size"),
        (hidden_size, "hidden_size"),
        (layer_count, "layer_count"),
        (current_intermediate_size, "current_intermediate_size"),
        (target_compaction_denominator, "target_compaction_denominator"),
        (minimum_intermediate_size, "minimum_intermediate_size"),
    )
    for value, name in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Lfm2InterfaceFrontierError(f"{name} must be positive.")
    if minimum_intermediate_size > current_intermediate_size:
        raise Lfm2InterfaceFrontierError(
            "minimum_intermediate_size exceeds current_intermediate_size."
        )
    target_ceiling = parent_unique_parameters // target_compaction_denominator
    added_interface_parameters = target_vocabulary_size * hidden_size
    unrebalanced_parameters = (
        base_tied_target_unique_parameters + added_interface_parameters
    )
    overage = max(0, unrebalanced_parameters - target_ceiling)
    parameters_per_intermediate_step = 3 * hidden_size * layer_count
    reduction = (
        0
        if overage == 0
        else (
            overage + parameters_per_intermediate_step - 1
        )
        // parameters_per_intermediate_step
    )
    target_intermediate_size = current_intermediate_size - reduction
    if target_intermediate_size < minimum_intermediate_size:
        raise Lfm2InterfaceFrontierError(
            "FFN width cannot absorb the dense untied interface within budget."
        )
    removed_body_parameters = reduction * parameters_per_intermediate_step
    target_unique_parameters = unrebalanced_parameters - removed_body_parameters
    if target_unique_parameters > target_ceiling:
        raise Lfm2InterfaceFrontierError(
            "Computed FFN rebalance still exceeds its parameter ceiling."
        )
    body = {
        "schema": LFM2_UNTIED_INTERFACE_BODY_REBALANCE_SCHEMA,
        "parent_unique_parameters": parent_unique_parameters,
        "target_compaction_denominator": target_compaction_denominator,
        "target_unique_parameter_ceiling": target_ceiling,
        "base_tied_target_unique_parameters": base_tied_target_unique_parameters,
        "added_dense_untied_output_parameters": added_interface_parameters,
        "unrebalanced_target_unique_parameters": unrebalanced_parameters,
        "hidden_size_preserved": hidden_size,
        "target_vocabulary_size_preserved": target_vocabulary_size,
        "layer_count_preserved": layer_count,
        "current_intermediate_size": current_intermediate_size,
        "target_intermediate_size": target_intermediate_size,
        "intermediate_size_reduction": reduction,
        "parameters_removed_per_intermediate_step": (
            parameters_per_intermediate_step
        ),
        "removed_body_parameters": removed_body_parameters,
        "target_unique_parameters": target_unique_parameters,
        "parameter_ceiling_spare": target_ceiling - target_unique_parameters,
        "realized_parameter_compaction_ratio": (
            parent_unique_parameters / target_unique_parameters
        ),
        "residual_width_preserved": True,
        "attention_geometry_preserved": True,
        "vocabulary_preserved": True,
        "depth_preserved": True,
        "automatic_target_budget_accounting": True,
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


__all__ = [
    "DEFAULT_INTERFACE_CONTEXTS",
    "DEFAULT_INTERFACE_RECIPES",
    "LFM2_CAUSAL_CONTEXT_TRANSITION_PLAN_SCHEMA",
    "LFM2_CAUSAL_RESIDUAL_TOKEN_BRIDGE_SCHEMA",
    "LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA",
    "LFM2_CHILD_CAUSAL_RESIDUAL_EVIDENCE_SCHEMA",
    "LFM2_CACHED_LOGIT_RESIDUAL_HEAD_SCHEMA",
    "LFM2_INTERFACE_FRONTIER_SCHEMA",
    "LFM2_INTERFACE_INVERSE_OUTPUT_PLAN_SCHEMA",
    "LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA",
    "LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA",
    "LFM2_UNTIED_INTERFACE_BODY_REBALANCE_SCHEMA",
    "Lfm2InterfaceFrontierError",
    "Lfm2InterfaceFrontierResult",
    "Lfm2InterfaceRecipe",
    "base_tied_target_budget_for_recipe",
    "Lfm2CachedLogitResidualHead",
    "Lfm2ChildCausalResidualEvidence",
    "Lfm2CausalResidualTokenBridge",
    "Lfm2CausalTrieEvidence",
    "Lfm2TerminalCacheEvidence",
    "collect_lfm2_terminal_cache_evidence",
    "compile_lfm2_causal_residual_token_bridge",
    "compile_lfm2_causal_trie_evidence",
    "collect_lfm2_child_causal_residual_evidence",
    "compile_lfm2_cached_logit_residual_head",
    "compile_lfm2_contextual_input_table",
    "compile_lfm2_interface_sequence_plan",
    "compile_lfm2_inverse_bucket_output_table",
    "compile_lfm2_inverse_output_plan",
    "compile_lfm2_prefix_transition_output_table",
    "fold_lfm2_interface_candidate",
    "fingerprint_lfm2_causal_body_inventory",
    "plan_lfm2_dense_untied_target_rebalance",
    "target_unique_parameters_for_recipe",
]
