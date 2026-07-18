from __future__ import annotations

import copy
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import torch
from torch import nn

import neural_forge.dynamic_lfm2_cpcf_reviewed_gate as reviewed_gate_module
from neural_forge.dynamic_behavior_execution import (
    BehaviorInferenceExecutionContract,
)
from neural_forge.dynamic_behavior_gate import resolve_behavior_gate_binding
from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.dynamic_lfm2_cpcf_reviewed_gate import (
    AUTOMATIC_RUNNER_REPORT_SCHEMA,
    COMPOSITE_PROMOTION_SCHEMA,
    CPCF_CAPTURE_SCHEMA,
    CPCF_RUN_SCHEMA,
    CanonicalJsonArtifact,
    Lfm2CpcfReviewedPromotionArtifacts,
    Lfm2CpcfReviewedPromotionError,
    Lfm2CpcfReviewedPromotionEvidence,
    Lfm2CpcfReviewedPromotionGate,
    authenticate_lfm2_cpcf_reviewed_gate_implementation,
    authenticate_lfm2_cpcf_reviewed_promotion_evidence,
    authenticate_lfm2_cpcf_reviewed_promotion_gate,
    build_cpcf_reviewed_promotion_report,
    fingerprint_cpcf_candidate_state,
    fingerprint_local_tokenizer_package,
    load_lfm2_cpcf_reviewed_promotion_evidence,
)
from neural_forge.hf_lfm2_intelligence_gate import (
    IntelligenceGatePolicy,
    build_lfm2_intelligence_holdout_manifest,
    build_manual_review_packet,
    evaluate_cached_intelligence,
    load_lfm2_intelligence_holdout_manifest,
)
from neural_forge.hf_lfm2_intelligence_holdout import (
    baseline_from_lfm2_intelligence_holdout_envelope,
    load_lfm2_intelligence_holdout_candidate_outputs,
    load_lfm2_intelligence_holdout_parent_envelope,
    seal_lfm2_intelligence_holdout_candidate_outputs,
    seal_lfm2_intelligence_holdout_parent_envelope,
)
from neural_forge.hf_lfm2_cpcf_model_fold import (
    LFM2_CPCF_MODEL_FOLD_SCHEMA,
    LFM2_CPCF_PARAMETER_RECEIPT_SCHEMA,
    Lfm2CpcfModelFoldPolicy,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
    Lfm2CpcfShortConvV4SolverPolicy,
)
from neural_forge.hf_lfm2_parent_relative_manual_gate import (
    bind_parent_relative_manual_ratings,
    build_parent_relative_manual_packet,
    evaluate_parent_relative_manual_retention,
)
from neural_forge.hf_tokenizer_bridge import fingerprint_hf_tokenizer_semantics


def _word_count(value: str) -> int:
    return len(value.replace(",", " ").replace(".", " ").split())


_CODE_BODIES = {
    "holdout_v1_python_00": "def holdout_is_even(n):\n    return n % 2 == 0",
    "holdout_v1_python_01": (
        "def holdout_clamp(x, low, high):\n"
        "    return low if x < low else high if x > high else x"
    ),
    "holdout_v1_python_02": "def holdout_last(items):\n    return items[-1]",
    "holdout_v1_python_03": "def holdout_reverse(text):\n    return text[::-1]",
    "holdout_v1_python_04": (
        "def holdout_between(x, low, high):\n    return low <= x and x <= high"
    ),
    "holdout_v1_python_05": (
        "def holdout_choose(flag, yes, no):\n    return yes if flag else no"
    ),
    "holdout_v1_python_06": ("def holdout_difference(a, b):\n    return a - b"),
    "holdout_v1_python_07": ("def holdout_join(left, right):\n    return left + right"),
}


def _passing_output(spec: Any, *, candidate: bool) -> str:
    validator = spec.validator
    if validator is None:
        assert spec.capability is not None
        terms = [
            group[-1] if candidate and len(group) > 1 else group[0]
            for group in spec.capability.required_concept_groups
        ]
        filler = "a coherent useful answer with clear practical context".split()
        output = " ".join(terms)
        index = 0
        while _word_count(output) < spec.capability.minimum_words:
            output = f"{output} {filler[index % len(filler)]}".strip()
            index += 1
        return output
    if validator.kind == "arithmetic_exact":
        return f"The checked result is {validator.expected_numeric}"
    if validator.kind == "python_pure_function":
        source = _CODE_BODIES[spec.case_id]
        return f"```python\n{source}\n```" if candidate else source
    if validator.exact_word is not None:
        return validator.exact_word
    terms = [
        group[-1] if candidate and len(group) > 1 else group[0]
        for group in validator.required_groups
    ]
    filler = "this concise response explains the requested result safely".split()
    output = " ".join(terms)
    index = 0
    while _word_count(output) < validator.minimum_words:
        output = f"{output} {filler[index % len(filler)]}".strip()
        index += 1
    return output


class _Tokenizer:
    def __init__(self, manifest: Any) -> None:
        self._prompt_ids = {
            spec.prompt: index + 1 for index, spec in enumerate(manifest.cases)
        }
        self._outputs = {
            index + 1: _passing_output(spec, candidate=True)
            for index, spec in enumerate(manifest.cases)
        }
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.bos_token_id = None
        self.unk_token_id = None
        self.all_special_ids = [0]

    def __len__(self) -> int:
        return 65

    def get_vocab(self) -> dict[str, int]:
        return {f"token_{index}": index for index in range(len(self))}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"token_{token_id}"

    def __call__(
        self,
        prompt: str,
        *,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert add_special_tokens is True
        token_id = self._prompt_ids[prompt]
        return {
            "input_ids": torch.tensor([[token_id]], dtype=torch.long),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool | None = None,
    ) -> str:
        del clean_up_tokenization_spaces
        retained = [
            token_id
            for token_id in token_ids
            if not (skip_special_tokens and token_id == self.eos_token_id)
        ]
        if len(retained) == 1 and retained[0] in self._outputs:
            return self._outputs[retained[0]]
        return " ".join(f"token_{value}" for value in retained)


class _ReplayModel(nn.Module):
    def __init__(self, *, output_offset: int = 0, execute_forward: bool = True):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor([1.25]))
        self.output_offset = output_offset
        self.execute_forward = execute_forward

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(dtype=self.anchor.dtype) * self.anchor

    def generate(self, **kwargs: Any) -> torch.Tensor:
        input_ids = kwargs["input_ids"]
        if self.execute_forward:
            self(input_ids)
        token_id = int(input_ids[0, -1]) + self.output_offset
        continuation = torch.tensor(
            [[token_id, 0]],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        return torch.cat((input_ids, continuation), dim=1)


def _artifact(value: dict[str, Any]) -> CanonicalJsonArtifact:
    return CanonicalJsonArtifact(
        source=value,
        document_sha256=canonical_sha256(value),
    )


def _seal(body: dict[str, Any], field: str) -> dict[str, Any]:
    return {**body, field: canonical_sha256(body)}


def _parameter_inventory(unique_parameters: int) -> dict[str, Any]:
    body = {
        "method": ("unique_materialized_parameter_storage_offset_numel_identity"),
        "unique_parameters": unique_parameters,
        "raw_named_parameter_occurrences": 1,
        "raw_named_parameter_numel": unique_parameters,
        "parameter_iterator_unique_numel": unique_parameters,
        "independent_methods_agree": True,
        "alias_group_count": 0,
        "alias_groups": [],
        "checkpoint_byte_size_used": False,
        "quantization_adjustment_used": False,
        "pruning_mask_adjustment_used": False,
    }
    return _seal(body, "inventory_sha256")


def _parameter_receipt(
    source_inventory: dict[str, Any],
    target_inventory: dict[str, Any],
    *,
    promotion_compression_ratio: float = 100.0,
) -> dict[str, Any]:
    source_parameters = int(source_inventory["unique_parameters"])
    target_parameters = int(target_inventory["unique_parameters"])
    realized_ratio = source_parameters / target_parameters
    body = {
        "schema": LFM2_CPCF_PARAMETER_RECEIPT_SCHEMA,
        "source_inventory": source_inventory,
        "target_inventory": target_inventory,
        "source_unique_parameters": source_parameters,
        "target_unique_parameters": target_parameters,
        "planned_target_unique_parameters": target_parameters,
        "parameters_removed_from_topology": (source_parameters - target_parameters),
        "realized_compression_ratio": realized_ratio,
        "promotion_compression_ratio": promotion_compression_ratio,
        "promotion_ratio_met": (realized_ratio >= promotion_compression_ratio),
        "valid_lower_ratio_diagnostic_fold": (
            realized_ratio < promotion_compression_ratio
        ),
        "parameter_count_compaction_only": True,
        "checkpoint_byte_ratio_used": False,
        "quantization_used": False,
        "pruning_used": False,
    }
    return _seal(body, "receipt_sha256")


def _rebuild_cpcf_compiler_chain(
    cpcf_body: dict[str, Any],
    *,
    policy: Lfm2CpcfModelFoldPolicy | None = None,
    phase_sha256: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Rebuild every nested compiler seal after synthetic provenance edits."""

    resolved_policy = (
        Lfm2CpcfModelFoldPolicy(
            promotion_compression_ratio=100.0,
            require_promotion_ratio=True,
        )
        if policy is None
        else policy
    )
    source_parameters = int(
        cpcf_body["parent_parameter_inventory"]["unique_parameters"]
    )
    target_parameters = int(cpcf_body["blank_target"]["target_unique_parameters"])
    parent_inventory = _parameter_inventory(source_parameters)
    blank_inventory = _parameter_inventory(target_parameters)
    compiled_inventory = _parameter_inventory(target_parameters)
    reloaded_inventory = _parameter_inventory(target_parameters)
    parameter_receipt = _parameter_receipt(
        parent_inventory,
        blank_inventory,
        promotion_compression_ratio=(resolved_policy.promotion_compression_ratio),
    )

    blank_body = copy.deepcopy(cpcf_body["blank_target"])
    blank_body.pop("report_sha256", None)
    blank_body.update(
        {
            "schema": "neural-forge-lfm2-cpcf-blank-target-v1",
            "parent_unique_parameters": source_parameters,
            "target_unique_parameters": target_parameters,
            "realized_compression_ratio": (source_parameters / target_parameters),
            "compiler_created_blank": True,
            "all_parameters_zero": True,
            "dense_materialized_parameters": True,
            "topology_provenance_verified": True,
            "parent_state_preserved": True,
            "quantization_used": False,
            "pruning_used": False,
            "training_used": False,
            "weights_installed": False,
        }
    )
    blank_report = _seal(blank_body, "report_sha256")

    prior_compiled = cpcf_body.get("compiled_model_fold")
    if not isinstance(prior_compiled, dict):
        prior_compiled = {}
    ordered_phases = (
        tuple(prior_compiled.get("phase_sha256", ()))
        if phase_sha256 is None
        else tuple(phase_sha256)
    )
    if not ordered_phases:
        ordered_phases = (
            canonical_sha256({"compiler_phase": 0}),
            canonical_sha256({"compiler_phase": 1}),
        )
    digest_defaults = {
        "boundary_bundle_sha256": canonical_sha256({"compiled": "boundary-bundle"}),
        "interface_install_report_sha256": canonical_sha256(
            {"compiled": "interface-install"}
        ),
        "neutral_norm_install_report_sha256": canonical_sha256(
            {"compiled": "neutral-norm-install"}
        ),
        "final_state_bank_sha256": canonical_sha256({"compiled": "final-state-bank"}),
        "final_norm_report_sha256": canonical_sha256({"compiled": "final-norm"}),
    }
    compiled_body = {
        "schema": LFM2_CPCF_MODEL_FOLD_SCHEMA,
        "blank_target_report_sha256": blank_report["report_sha256"],
        "policy": resolved_policy.as_dict(),
        "phase_count": len(ordered_phases),
        "phase_sha256": list(ordered_phases),
        **{
            name: prior_compiled.get(name, value)
            for name, value in digest_defaults.items()
        },
        "ordered_layer_result_sha256": prior_compiled.get(
            "ordered_layer_result_sha256",
            [canonical_sha256({"compiled": "layer-result-0"})],
        ),
        "parameter_receipt": parameter_receipt,
        "target_parameter_inventory_before": blank_inventory,
        "target_parameter_inventory_after": compiled_inventory,
        "source_state_preserved": True,
        "target_parameter_count_preserved": True,
        "all_layers_compiled_in_order": True,
        "actual_folded_upstream_replay_used": True,
        "operation_audit": {
            "parent_model_forward_calls": 0,
            "target_model_forward_calls": 0,
            "generation_calls": 0,
            "parent_cache_generation_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "fallback_paths": 0,
        },
    }
    compiled = _seal(compiled_body, "result_sha256")
    reload_receipt = _parameter_receipt(
        parent_inventory,
        reloaded_inventory,
        promotion_compression_ratio=(resolved_policy.promotion_compression_ratio),
    )
    cpcf_body.update(
        {
            "blank_target": blank_report,
            "parent_parameter_inventory": parent_inventory,
            "blank_parameter_inventory": blank_inventory,
            "compiled_parameter_inventory": compiled_inventory,
            "reloaded_parameter_inventory": reloaded_inventory,
            "compression": reload_receipt,
            "compiled_model_fold": compiled,
        }
    )
    ordered_phase_binding = canonical_sha256(
        {
            "schema": ("neural-forge-lfm2-cpcf-ordered-phase-binding-v1"),
            "phase_count": len(ordered_phases),
            "ordered_phase_sha256": list(ordered_phases),
        }
    )
    return {
        "compiled_model_fold_result_sha256": compiled["result_sha256"],
        "compiled_model_fold_policy": resolved_policy.as_dict(),
        "compiled_model_fold_policy_sha256": canonical_sha256(
            resolved_policy.as_dict()
        ),
        "compiled_model_fold_phase_count": len(ordered_phases),
        "compiled_model_fold_phase_sha256": list(ordered_phases),
        "compiled_model_fold_phase_binding_sha256": ordered_phase_binding,
        "compiled_blank_target_report_sha256": (blank_report["report_sha256"]),
        "compiled_parameter_receipt_sha256": (parameter_receipt["receipt_sha256"]),
    }


def _candidate_file_sha(document: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_bytes(document) + b"\n").hexdigest()


def _write_tokenizer_package(path: Path) -> str:
    path.mkdir()
    (path / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [{"id": 0, "content": "<eos>", "special": True}],
                "model": {
                    "type": "BPE",
                    "vocab": {f"token_{index}": index for index in range(65)},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {"eos_token": "<eos>", "pad_token": "<eos>"},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return fingerprint_local_tokenizer_package(path)[0]


def _fixture(
    tmp_path: Path,
    *,
    bridge_artifact_sha256: str | None = None,
    bridge_profile_sha256: str | None = None,
    tokenizer: Any | None = None,
    tokenizer_directory: Path | None = None,
) -> dict[str, Any]:
    manifest_document = build_lfm2_intelligence_holdout_manifest()
    manifest = load_lfm2_intelligence_holdout_manifest(manifest_document)
    tokenizer = _Tokenizer(manifest) if tokenizer is None else tokenizer
    if tokenizer_directory is None:
        tokenizer_directory = tmp_path / "tokenizer"
        tokenizer_package = _write_tokenizer_package(tokenizer_directory)
    else:
        tokenizer_package = fingerprint_local_tokenizer_package(tokenizer_directory)[0]
    tokenizer_semantics = fingerprint_hf_tokenizer_semantics(tokenizer)
    eos = [0]

    parent_rows = [
        {
            "case_id": spec.case_id,
            "input_token_count": index + 1,
            "output": _passing_output(spec, candidate=False),
            "output_token_ids": [index + 1, 0],
        }
        for index, spec in enumerate(manifest.cases)
    ]
    parent_state = canonical_sha256({"parent": "state"})
    parent_model = canonical_sha256({"parent": "model"})
    parent = seal_lfm2_intelligence_holdout_parent_envelope(
        manifest,
        parent_rows,
        parent_state_fingerprint=parent_state,
        parent_model_fingerprint=parent_model,
        tokenizer_fingerprint=tokenizer_package,
        eos_token_ids=eos,
    )
    loaded_parent = load_lfm2_intelligence_holdout_parent_envelope(
        parent,
        expected_envelope_sha256=parent["envelope_sha256"],
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_parent_state_fingerprint=parent_state,
        expected_parent_model_fingerprint=parent_model,
        expected_tokenizer_fingerprint=tokenizer_package,
    )
    baseline = baseline_from_lfm2_intelligence_holdout_envelope(loaded_parent)

    model = _ReplayModel()
    candidate_state = fingerprint_cpcf_candidate_state(model)
    candidate_rows = [
        {
            "case_id": spec.case_id,
            "output": _passing_output(spec, candidate=True),
            "output_token_ids": [index + 1, 0],
            "generation_limit": spec.generation_limit,
            "topical_relevance": True,
        }
        for index, spec in enumerate(manifest.cases)
    ]
    candidate = seal_lfm2_intelligence_holdout_candidate_outputs(
        manifest,
        candidate_rows,
        candidate_state_fingerprint=candidate_state,
        tokenizer_fingerprint=tokenizer_package,
        eos_token_ids=eos,
    )
    loaded_candidate = load_lfm2_intelligence_holdout_candidate_outputs(
        candidate,
        expected_candidate_outputs_sha256=candidate["candidate_outputs_sha256"],
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_candidate_state_fingerprint=candidate_state,
        expected_tokenizer_fingerprint=tokenizer_package,
        eos_token_ids=eos,
    )

    bridge_artifact = (
        canonical_sha256({"bridge": "artifact"})
        if bridge_artifact_sha256 is None
        else bridge_artifact_sha256
    )
    bridge_profile = (
        canonical_sha256({"bridge": "profile"})
        if bridge_profile_sha256 is None
        else bridge_profile_sha256
    )
    cache_key = canonical_sha256({"cache": "parent-traces"})
    trace_adapter = canonical_sha256({"trace": "adapter"})
    trajectory_corpus = canonical_sha256({"trajectory": "corpus"})
    parent_graph = canonical_sha256({"parent": "graph"})
    live_parent_state = canonical_sha256({"parent": "live-state"})
    requested_budget = {
        "schema": "neural-forge-dense-parameter-budget-v1",
        "kind": "budget_request",
        "target_family": "lfm2",
        "target_modalities": ["text"],
        "target_unique_parameters": 1,
        "compression_ratio": None,
        "target_vocabulary_size": 65,
        "match_policy": "at_or_below",
        "maximum_candidates": 8,
    }
    frontier = canonical_sha256({"target": "frontier"})
    target_certificate = canonical_sha256({"target": "certificate"})
    selected_candidate = canonical_sha256({"target": "candidate"})
    target_topology = canonical_sha256({"target": "topology"})
    blank_state = canonical_sha256({"target": "zero-blank"})
    parent_inventory = {"unique_parameters": 100}
    inventory = {"unique_parameters": 1}
    cpcf_body = {
        "schema": CPCF_RUN_SCHEMA,
        "status": "folded_child_requires_manual_intelligence_review",
        "inputs": {
            "bridge_artifact_sha256": bridge_artifact,
            "bridge_profile_sha256": bridge_profile,
            "cache_key": cache_key,
            "parent_fingerprint": parent_model,
            "trace_adapter_sha256": trace_adapter,
            "trajectory_corpus_sha256": trajectory_corpus,
        },
        "blank_target": {
            "parent_graph_fingerprint": parent_graph,
            "parent_state_fingerprint": live_parent_state,
            "target_unique_parameters": 1,
            "requested_budget": requested_budget,
            "frontier_fingerprint": frontier,
            "target_certificate_sha256": target_certificate,
            "selected_candidate_fingerprint": selected_candidate,
            "selected_candidate_rank": 0,
            "target_topology_fingerprint": target_topology,
            "blank_state_fingerprint": blank_state,
        },
        "parent_parameter_inventory": parent_inventory,
        "blank_parameter_inventory": inventory,
        "compiled_parameter_inventory": inventory,
        "reloaded_parameter_inventory": inventory,
        "compression": {
            "source_unique_parameters": 100,
            "target_unique_parameters": 1,
            "planned_target_unique_parameters": 1,
            "parameter_count_compaction_only": True,
            "checkpoint_byte_ratio_used": False,
            "promotion_ratio_met": True,
            "quantization_used": False,
            "pruning_used": False,
        },
        "fresh_reload_verified": True,
        "structural_parameter_compaction_proven": True,
        "parent_state_preserved": True,
        "project_promotion_ratio_met": True,
        "diagnostic_control": False,
        "behavioral_intelligence_retention_proven": False,
        "manual_intelligence_review_required": True,
        "operation_audit": {
            "parent_forward_calls": 0,
            "parent_generation_calls": 0,
            "parent_cache_generation_calls": 0,
            "parent_cache_regeneration_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "quantized_parameters": 0,
            "pruned_parameters": 0,
            "fallback_model_calls": 0,
            "network_access": False,
        },
        "claims": {
            "folding_only": True,
            "blank_target_filled": True,
            "training_used": False,
            "quantization_used": False,
            "pruning_used": False,
            "parent_backed_fallback_used": False,
            "exact_token_match_used_as_principal_objective": False,
            "marginal_compaction_reported_as_success": False,
        },
    }
    compiled_provenance = _rebuild_cpcf_compiler_chain(cpcf_body)
    parent_inventory = cpcf_body["parent_parameter_inventory"]
    inventory = cpcf_body["reloaded_parameter_inventory"]
    cpcf = _seal(cpcf_body, "receipt_sha256")

    capture_body = {
        "schema": CPCF_CAPTURE_SCHEMA,
        "status": "captured_frozen_child_holdout",
        "manifest": {
            "manifest_sha256": manifest.manifest_sha256,
            "manifest_document_sha256": manifest.document_sha256,
            "case_count": 64,
        },
        "cpcf_provenance": {
            "receipt_sha256": cpcf["receipt_sha256"],
            "external_receipt_file_sha256": _candidate_file_sha(cpcf),
            "internal_receipt_file_sha256": _candidate_file_sha(cpcf),
            "external_internal_receipts_identical": True,
            "saved_checkpoint_files_match_receipt": True,
            "reloaded_parameter_inventory": inventory,
            "reloaded_parameter_inventory_matches_receipt": True,
            "parameter_inventory_unchanged_after_generation": True,
        },
        "strict_tokenizer_bridge": {
            "artifact_sha256": bridge_artifact,
            "profile_sha256": bridge_profile,
            "files_verified": True,
        },
        "target_tokenizer": {
            "semantics_sha256": tokenizer_semantics,
            "source_package_fingerprint": tokenizer_package,
            "checkpoint_package_fingerprint": tokenizer_package,
            "source_and_checkpoint_packages_identical": True,
            "source_and_checkpoint_semantics_identical": True,
            "eos_token_ids": eos,
        },
        "candidate": {
            "output_file_sha256": _candidate_file_sha(candidate),
            "candidate_outputs_sha256": candidate["candidate_outputs_sha256"],
            "candidate_state_fingerprint": candidate_state,
            "state_before_sha256": candidate_state,
            "state_after_sha256": candidate_state,
            "state_unchanged": True,
            "case_count": 64,
            "all_output_token_ids_are_child_domain_only": True,
            "decoded_text_sealed": True,
        },
        "generation_protocol": {
            "mode": "greedy_whole_response",
            "do_sample": False,
            "use_cache": True,
            "one_generate_invocation_per_case": True,
            "per_case_manifest_generation_limits_used": True,
            "global_64_token_cap_used": False,
            "eos_token_ids": eos,
            "exact_parent_wording_used": False,
            "exact_parent_token_matching_used": False,
        },
        "operation_audit": {
            "candidate_generate_invocations": 64,
            "parent_model_loaded": False,
            "parent_forward_passes": 0,
            "parent_generate_invocations": 0,
            "trace_cache_opened": False,
            "training_used": False,
            "training_steps": 0,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "gradients_used": False,
            "quantization_used": False,
            "pruning_used": False,
            "network_access": False,
            "hub_loader_used": False,
        },
        "claims": {
            "fresh_local_frozen_child_loaded": True,
            "candidate_state_preserved": True,
            "folding_receipt_authenticated": True,
            "strict_bridge_authenticated": True,
            "strict_bridge_artifact_and_target_domain_authenticated": True,
            "target_tokenizer_package_authenticated": True,
            "target_tokenizer_semantics_authenticated": True,
            "candidate_outputs_are_not_training": True,
            "candidate_outputs_are_not_exact_token_matching": True,
            "intelligence_retention_scored_here": False,
        },
    }
    capture = _seal(capture_body, "capture_receipt_sha256")

    policy = IntelligenceGatePolicy()
    automatic = evaluate_cached_intelligence(
        baseline,
        loaded_candidate.candidates,
        policy=policy,
        manual_ratings=None,
    ).as_dict()
    manual = build_manual_review_packet(
        baseline,
        loaded_candidate.candidates,
        policy,
    )
    parent_packet = build_parent_relative_manual_packet(manual)
    ratings = bind_parent_relative_manual_ratings(
        parent_packet,
        [
            {
                "case_id": value["case_id"],
                "parent": {
                    "correctness": 4,
                    "coherence": 4,
                    "instruction_following": 4,
                    "capability_retention": 4,
                },
                "candidate": {
                    "correctness": 4,
                    "coherence": 4,
                    "instruction_following": 4,
                    "capability_retention": 4,
                },
                "reason": "Both responses are correct, coherent, and useful.",
            }
            for value in parent_packet["cases"]
        ],
        reviewer_id="reviewer-1",
    )
    parent_report = evaluate_parent_relative_manual_retention(
        parent_packet,
        ratings,
    )
    documents = {
        "manifest": manifest_document,
        "parent": parent,
        "cpcf": cpcf,
        "candidate": candidate,
        "capture": capture,
        "automatic": automatic,
        "manual": manual,
        "parent_packet": parent_packet,
        "ratings": ratings,
        "parent_report": parent_report,
    }
    evidence_manifest = {
        "schema": ("neural-forge-lfm2-cpcf-reviewed-promotion-evidence-v1"),
        "case_count": 64,
        "manifest_sha256": manifest.manifest_sha256,
        "manifest_document_sha256": canonical_sha256(manifest_document),
        "parent_envelope_sha256": parent["envelope_sha256"],
        "parent_envelope_document_sha256": canonical_sha256(parent),
        "cpcf_run_receipt_sha256": cpcf["receipt_sha256"],
        "cpcf_run_receipt_document_sha256": canonical_sha256(cpcf),
        "parent_cache_key": cache_key,
        "cache_live_parent_fingerprint": parent_model,
        "live_parent_state_fingerprint": live_parent_state,
        "trace_adapter_sha256": trace_adapter,
        "trajectory_corpus_sha256": trajectory_corpus,
        "blank_target_parent_graph_fingerprint": parent_graph,
        "exact_target_unique_parameters": 1,
        "target_requested_budget": requested_budget,
        "target_frontier_fingerprint": frontier,
        "target_certificate_sha256": target_certificate,
        "selected_target_candidate_fingerprint": selected_candidate,
        "selected_target_candidate_rank": 0,
        "target_topology_fingerprint": target_topology,
        "blank_target_state_fingerprint": blank_state,
        "target_plan_binding_sha256": canonical_sha256(
            {
                "schema": "neural-forge-lfm2-cpcf-target-plan-binding-v1",
                "parent_graph_fingerprint": parent_graph,
                "exact_target_unique_parameters": 1,
                "requested_budget": requested_budget,
                "frontier_fingerprint": frontier,
                "target_certificate_sha256": target_certificate,
                "selected_candidate_fingerprint": selected_candidate,
                "selected_candidate_rank": 0,
                "target_topology_fingerprint": target_topology,
                "blank_state_fingerprint": blank_state,
            }
        ),
        **compiled_provenance,
        "bridge_artifact_sha256": bridge_artifact,
        "bridge_profile_sha256": bridge_profile,
        "candidate_state_fingerprint": candidate_state,
        "candidate_outputs_sha256": candidate["candidate_outputs_sha256"],
        "candidate_outputs_document_sha256": canonical_sha256(candidate),
        "capture_receipt_sha256": capture["capture_receipt_sha256"],
        "capture_receipt_document_sha256": canonical_sha256(capture),
        "automatic_report_sha256": automatic["report_sha256"],
        "automatic_report_document_sha256": canonical_sha256(automatic),
        "automatic_core_report_sha256": automatic["report_sha256"],
        "automatic_gate_ready_for_manual_completion": True,
        "manual_packet_sha256": manual["packet_sha256"],
        "manual_packet_document_sha256": canonical_sha256(manual),
        "parent_relative_manual_packet_sha256": parent_packet["packet_sha256"],
        "parent_relative_manual_packet_document_sha256": canonical_sha256(
            parent_packet
        ),
        "parent_relative_ratings_sha256": ratings["ratings_sha256"],
        "parent_relative_ratings_document_sha256": canonical_sha256(ratings),
        "parent_relative_manual_report_sha256": parent_report["report_sha256"],
        "parent_relative_manual_report_document_sha256": canonical_sha256(
            parent_report
        ),
        "parent_relative_manual_report_passed": True,
        "parent_relative_manual_retention": parent_report["retention"],
        "candidate_tokenizer_fingerprint": tokenizer_package,
        "target_tokenizer_semantics_sha256": tokenizer_semantics,
        "eos_token_ids": eos,
        "parent_model_loaded": False,
        "training_used": False,
        "pruning_used": False,
        "quantization_used": False,
        "network_access": False,
    }
    composite = build_cpcf_reviewed_promotion_report(
        reviewed_evidence=evidence_manifest,
        inference_execution_contract=(
            BehaviorInferenceExecutionContract.top_level_forward()
        ),
    )
    documents["composite"] = composite
    artifacts = Lfm2CpcfReviewedPromotionArtifacts(
        manifest=_artifact(manifest_document),
        parent_envelope=_artifact(parent),
        cpcf_run_receipt=_artifact(cpcf),
        candidate_outputs=_artifact(candidate),
        capture_receipt=_artifact(capture),
        automatic_report=_artifact(automatic),
        manual_packet=_artifact(manual),
        parent_relative_manual_packet=_artifact(parent_packet),
        parent_relative_ratings=_artifact(ratings),
        parent_relative_report=_artifact(parent_report),
        composite_promotion_report=_artifact(composite),
    )
    return {
        "artifacts": artifacts,
        "documents": documents,
        "model": model,
        "tokenizer": tokenizer,
        "tokenizer_directory": tokenizer_directory,
    }


def _replace_artifact(
    artifacts: Lfm2CpcfReviewedPromotionArtifacts,
    name: str,
    replacement: CanonicalJsonArtifact,
) -> Lfm2CpcfReviewedPromotionArtifacts:
    values = {
        field: getattr(artifacts, field) for field in artifacts.__dataclass_fields__
    }
    values[name] = replacement
    return Lfm2CpcfReviewedPromotionArtifacts(**values)


def test_reviewed_gate_accepts_live_and_fresh_reload_and_binds_all_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )

    live = gate(fixture["model"])
    fresh_reload = gate(_ReplayModel())
    binding = resolve_behavior_gate_binding(gate)

    assert live.accepted is True
    assert fresh_reload.accepted is True
    assert binding.baseline_fingerprint == (evidence.full_reviewed_evidence_fingerprint)
    assert binding.manifest["inference_execution"] == (
        evidence.inference_execution_contract.as_dict()
    )
    assert (
        binding.manifest["components"][0]["full_reviewed_evidence_fingerprint"]
        == evidence.full_reviewed_evidence_fingerprint
    )
    assert (
        evidence.bridge_artifact_sha256
        == fixture["documents"]["cpcf"]["inputs"]["bridge_artifact_sha256"]
    )
    assert (
        evidence.bridge_profile_sha256
        == fixture["documents"]["cpcf"]["inputs"]["bridge_profile_sha256"]
    )
    receipt = fixture["documents"]["cpcf"]
    blank = receipt["blank_target"]
    component = binding.manifest["components"][0]
    assert evidence.cpcf_run_receipt_sha256 == receipt["receipt_sha256"]
    assert evidence.cpcf_run_receipt_document_sha256 == canonical_sha256(receipt)
    assert evidence.parent_cache_key == receipt["inputs"]["cache_key"]
    assert (
        evidence.cache_live_parent_fingerprint
        == receipt["inputs"]["parent_fingerprint"]
    )
    assert evidence.live_parent_state_fingerprint == blank["parent_state_fingerprint"]
    assert evidence.trace_adapter_sha256 == (receipt["inputs"]["trace_adapter_sha256"])
    assert (
        evidence.trajectory_corpus_sha256
        == (receipt["inputs"]["trajectory_corpus_sha256"])
    )
    assert (
        evidence.blank_target_parent_graph_fingerprint
        == blank["parent_graph_fingerprint"]
    )
    assert evidence.exact_target_unique_parameters == 1
    assert dict(evidence.target_requested_budget) == blank["requested_budget"]
    assert evidence.target_frontier_fingerprint == blank["frontier_fingerprint"]
    assert evidence.target_certificate_sha256 == blank["target_certificate_sha256"]
    assert (
        evidence.selected_target_candidate_fingerprint
        == blank["selected_candidate_fingerprint"]
    )
    assert evidence.selected_target_candidate_rank == 0
    assert evidence.target_topology_fingerprint == blank["target_topology_fingerprint"]
    assert evidence.blank_target_state_fingerprint == blank["blank_state_fingerprint"]
    assert (
        component["target_plan_binding_sha256"] == evidence.target_plan_binding_sha256
    )
    assert component["cpcf_run_receipt_sha256"] == receipt["receipt_sha256"]
    assert component["parent_cache_key"] == receipt["inputs"]["cache_key"]
    assert component["exact_target_unique_parameters"] == 1
    compiled = receipt["compiled_model_fold"]
    assert evidence.compiled_model_fold_result_sha256 == compiled["result_sha256"]
    assert dict(evidence.compiled_model_fold_policy) == compiled["policy"]
    assert evidence.compiled_model_fold_policy_sha256 == canonical_sha256(
        compiled["policy"]
    )
    assert evidence.compiled_model_fold_phase_count == compiled["phase_count"]
    assert list(evidence.compiled_model_fold_phase_sha256) == compiled["phase_sha256"]
    assert evidence.compiled_blank_target_report_sha256 == blank["report_sha256"]
    assert (
        evidence.compiled_parameter_receipt_sha256
        == compiled["parameter_receipt"]["receipt_sha256"]
    )
    assert component["compiled_model_fold_result_sha256"] == compiled["result_sha256"]
    assert component["compiled_model_fold_policy"] == compiled["policy"]
    assert component["compiled_model_fold_phase_sha256"] == compiled["phase_sha256"]


def _evidence_constructor_fields(
    evidence: Lfm2CpcfReviewedPromotionEvidence,
) -> dict[str, Any]:
    return {
        name: getattr(evidence, name)
        for name in (
            "manifest",
            "cpcf_run_receipt_sha256",
            "cpcf_run_receipt_document_sha256",
            "parent_cache_key",
            "cache_live_parent_fingerprint",
            "live_parent_state_fingerprint",
            "trace_adapter_sha256",
            "trajectory_corpus_sha256",
            "blank_target_parent_graph_fingerprint",
            "exact_target_unique_parameters",
            "target_requested_budget",
            "target_frontier_fingerprint",
            "target_certificate_sha256",
            "selected_target_candidate_fingerprint",
            "selected_target_candidate_rank",
            "target_topology_fingerprint",
            "blank_target_state_fingerprint",
            "target_plan_binding_sha256",
            "compiled_model_fold_result_sha256",
            "compiled_model_fold_policy",
            "compiled_model_fold_policy_sha256",
            "compiled_model_fold_phase_count",
            "compiled_model_fold_phase_sha256",
            "compiled_model_fold_phase_binding_sha256",
            "compiled_blank_target_report_sha256",
            "compiled_parameter_receipt_sha256",
            "candidate_rows",
            "candidate_state_fingerprint",
            "candidate_outputs_sha256",
            "candidate_tokenizer_fingerprint",
            "target_tokenizer_semantics_sha256",
            "eos_token_ids",
            "reviewed_evidence",
            "reviewed_evidence_fingerprint",
            "composite_report_sha256",
            "composite_report_document_sha256",
            "full_reviewed_evidence_fingerprint",
            "inference_execution_contract",
        )
    }


def test_reviewed_evidence_is_loader_only(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])

    with pytest.raises(TypeError, match="loader-only"):
        Lfm2CpcfReviewedPromotionEvidence(**_evidence_constructor_fields(evidence))
    forged = object.__new__(Lfm2CpcfReviewedPromotionEvidence)
    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="lacks loader authentication",
    ):
        authenticate_lfm2_cpcf_reviewed_promotion_evidence(forged)
    assert authenticate_lfm2_cpcf_reviewed_promotion_evidence(evidence) is evidence


@pytest.mark.parametrize(
    ("field", "replacement_value"),
    (
        (
            "cpcf_run_receipt_sha256",
            canonical_sha256({"spoof": "run-receipt"}),
        ),
        (
            "cpcf_run_receipt_document_sha256",
            canonical_sha256({"spoof": "run-document"}),
        ),
        (
            "parent_cache_key",
            canonical_sha256({"spoof": "cache"}),
        ),
        (
            "cache_live_parent_fingerprint",
            canonical_sha256({"spoof": "parent"}),
        ),
        (
            "trace_adapter_sha256",
            canonical_sha256({"spoof": "trace-adapter"}),
        ),
        (
            "trajectory_corpus_sha256",
            canonical_sha256({"spoof": "trajectory"}),
        ),
        (
            "blank_target_parent_graph_fingerprint",
            canonical_sha256({"spoof": "parent-graph"}),
        ),
        ("exact_target_unique_parameters", 2),
        (
            "target_requested_budget",
            MappingProxyType({"spoof": "budget"}),
        ),
        (
            "selected_target_candidate_fingerprint",
            canonical_sha256({"spoof": "candidate"}),
        ),
        ("selected_target_candidate_rank", 1),
        (
            "target_plan_binding_sha256",
            canonical_sha256({"spoof": "target-plan"}),
        ),
        (
            "compiled_model_fold_result_sha256",
            canonical_sha256({"spoof": "compiled-result"}),
        ),
        (
            "compiled_model_fold_policy_sha256",
            canonical_sha256({"spoof": "compiled-policy"}),
        ),
        ("compiled_model_fold_phase_count", 1),
        (
            "compiled_model_fold_phase_binding_sha256",
            canonical_sha256({"spoof": "compiled-phases"}),
        ),
        (
            "candidate_state_fingerprint",
            canonical_sha256({"spoof": "state"}),
        ),
        (
            "candidate_outputs_sha256",
            canonical_sha256({"spoof": "outputs"}),
        ),
        (
            "candidate_tokenizer_fingerprint",
            canonical_sha256({"spoof": "tokenizer"}),
        ),
        (
            "target_tokenizer_semantics_sha256",
            canonical_sha256({"spoof": "semantics"}),
        ),
        ("eos_token_ids", (1,)),
    ),
)
def test_loader_authenticated_evidence_rejects_typed_field_mutation(
    tmp_path: Path,
    field: str,
    replacement_value: Any,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    object.__setattr__(evidence, field, replacement_value)

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="self-signed, or mutated",
    ):
        authenticate_lfm2_cpcf_reviewed_promotion_evidence(evidence)


@pytest.mark.parametrize(
    ("field", "replacement_value"),
    (
        ("schema", "spoofed-reviewed-schema-v1"),
        ("bridge_artifact_sha256", "not-a-digest"),
        ("bridge_profile_sha256", "not-a-digest"),
        (
            "parent_cache_key",
            canonical_sha256({"spoof": "reviewed-cache"}),
        ),
        ("exact_target_unique_parameters", 2),
        (
            "target_plan_binding_sha256",
            canonical_sha256({"spoof": "reviewed-plan"}),
        ),
        (
            "compiled_model_fold_result_sha256",
            canonical_sha256({"spoof": "reviewed-compiled-result"}),
        ),
        (
            "compiled_model_fold_phase_binding_sha256",
            canonical_sha256({"spoof": "reviewed-compiled-phases"}),
        ),
        (
            "candidate_state_fingerprint",
            canonical_sha256({"spoof": "reviewed-state"}),
        ),
    ),
)
def test_loader_authenticated_evidence_rejects_reviewed_map_mutation(
    tmp_path: Path,
    field: str,
    replacement_value: Any,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    reviewed = dict(evidence.reviewed_evidence)
    reviewed[field] = replacement_value
    object.__setattr__(
        evidence,
        "reviewed_evidence",
        MappingProxyType(reviewed),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="self-signed, or mutated",
    ):
        authenticate_lfm2_cpcf_reviewed_promotion_evidence(evidence)


def test_spoofed_trusted_document_digest_rejects(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    spoofed = _replace_artifact(
        fixture["artifacts"],
        "candidate_outputs",
        CanonicalJsonArtifact(
            source=fixture["documents"]["candidate"],
            document_sha256="0" * 64,
        ),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="trusted document",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(spoofed)


def test_run_receipt_parent_must_match_reviewed_holdout_parent(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    body["inputs"]["parent_fingerprint"] = canonical_sha256({"different": "parent"})
    modified = _seal(body, "receipt_sha256")
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "cpcf_run_receipt",
        _artifact(modified),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="differs from the reviewed holdout parent",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_run_receipt_exact_target_count_must_cross_bind_every_inventory(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    body["reloaded_parameter_inventory"] = _parameter_inventory(2)
    modified = _seal(body, "receipt_sha256")
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "cpcf_run_receipt",
        _artifact(modified),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="one exact target parameter count",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_run_receipt_requires_compiled_model_fold(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    body.pop("compiled_model_fold")
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "cpcf_run_receipt",
        _artifact(_seal(body, "receipt_sha256")),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="no compiled_model_fold report",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_compiled_model_fold_result_seal_is_recomputed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    body["compiled_model_fold"]["policy"]["ridge"] = 0.125
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "cpcf_run_receipt",
        _artifact(_seal(body, "receipt_sha256")),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="compiled model-fold report schema or canonical seal",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_compiled_model_fold_requires_zero_operation_audit(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    compiled = body["compiled_model_fold"]
    compiled.pop("result_sha256")
    compiled["operation_audit"]["training_steps"] = 1
    body["compiled_model_fold"] = _seal(compiled, "result_sha256")
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "cpcf_run_receipt",
        _artifact(_seal(body, "receipt_sha256")),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="zero-operation invariants",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_compiled_model_fold_binds_every_ordered_phase(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    compiled = body["compiled_model_fold"]
    compiled.pop("result_sha256")
    compiled["phase_count"] += 1
    body["compiled_model_fold"] = _seal(compiled, "result_sha256")
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "cpcf_run_receipt",
        _artifact(_seal(body, "receipt_sha256")),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="phase seals do not cover every phase",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_compiled_model_fold_cross_binds_compiled_inventory(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    compiled = body["compiled_model_fold"]
    compiled.pop("result_sha256")
    compiled["target_parameter_inventory_after"] = _parameter_inventory(2)
    body["compiled_model_fold"] = _seal(compiled, "result_sha256")
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "cpcf_run_receipt",
        _artifact(_seal(body, "receipt_sha256")),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="do not share exact inventories",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_compiled_model_fold_policy_must_be_complete_and_typed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    compiled = body["compiled_model_fold"]
    compiled.pop("result_sha256")
    compiled["policy"].pop("short_conv_gate_policy")
    body["compiled_model_fold"] = _seal(compiled, "result_sha256")
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "cpcf_run_receipt",
        _artifact(_seal(body, "receipt_sha256")),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="no short-conv gate policy",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_compiled_model_fold_v4_policy_round_trips_typed() -> None:
    expected = Lfm2CpcfModelFoldPolicy(
        short_conv_v4_solver_policy=Lfm2CpcfShortConvV4SolverPolicy(
            ridge_alphas=(1e-8, 1e-4),
            variable_projection_passes=2,
            context_objective_weight=0.075,
        )
    )

    observed = reviewed_gate_module._normalize_model_fold_policy(expected.as_dict())

    assert observed == expected.as_dict()


def test_compiled_model_fold_v4_policy_rejects_authentication_drift() -> None:
    policy = Lfm2CpcfModelFoldPolicy(
        short_conv_v4_solver_policy=Lfm2CpcfShortConvV4SolverPolicy(
            ridge_alphas=(1e-8, 1e-4),
            variable_projection_passes=2,
        )
    ).as_dict()
    policy["short_conv_v4_solver_policy"]["authentication_sha256"] = "0" * 64

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="invalid",
    ):
        reviewed_gate_module._normalize_model_fold_policy(policy)


@pytest.mark.parametrize("status,promoted", [("pending", False), ("rejected", False)])
def test_pending_or_rejected_composite_report_rejects(
    tmp_path: Path,
    status: str,
    promoted: bool,
) -> None:
    fixture = _fixture(tmp_path)
    document = copy.deepcopy(fixture["documents"]["composite"])
    body = dict(document)
    body.pop("promotion_report_sha256")
    body["status"] = status
    body["promoted"] = promoted
    modified = _seal(body, "promotion_report_sha256")
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "composite_promotion_report",
        _artifact(modified),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="pending, rejected, spoofed, or stale",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_rejected_parent_relative_report_rejects(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    packet = fixture["documents"]["parent_packet"]
    ratings = bind_parent_relative_manual_ratings(
        packet,
        [
            {
                "case_id": value["case_id"],
                "parent": {
                    "correctness": 4,
                    "coherence": 4,
                    "instruction_following": 4,
                    "capability_retention": 4,
                },
                "candidate": {
                    "correctness": 0,
                    "coherence": 0,
                    "instruction_following": 0,
                    "capability_retention": 0,
                },
                "reason": "The candidate clearly regressed on every dimension.",
            }
            for value in packet["cases"]
        ],
        reviewer_id="reviewer-2",
    )
    rejected = evaluate_parent_relative_manual_retention(packet, ratings)
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "parent_relative_ratings",
        _artifact(ratings),
    )
    artifacts = _replace_artifact(
        artifacts,
        "parent_relative_report",
        _artifact(rejected),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="pending, rejected, or regressed",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_exact_candidate_state_mismatch_rejects(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )
    mismatched = _ReplayModel()
    with torch.no_grad():
        mismatched.anchor.add_(1.0)

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="model state differs",
    ):
        gate(mismatched)


def test_reviewed_output_replay_mismatch_rejects(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="reviewed candidate output digest",
    ):
        gate(_ReplayModel(output_offset=1))


def test_generate_that_skips_model_forward_rejects(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="without executing",
    ):
        gate(_ReplayModel(execute_forward=False))


def test_reviewed_gate_instance_is_immutable_and_cannot_shadow_evaluation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )

    assert not hasattr(gate, "__dict__")
    with pytest.raises(AttributeError, match="immutable"):
        gate.evaluate_model = lambda model: model  # type: ignore[method-assign]
    with pytest.raises(AttributeError):
        object.__setattr__(gate, "evaluate_model", lambda model: model)
    with pytest.raises(AttributeError, match="immutable"):
        gate.evidence = evidence
    with pytest.raises(AttributeError, match="immutable"):
        del gate.tokenizer
    assert authenticate_lfm2_cpcf_reviewed_promotion_gate(gate) is gate


def test_object_new_gate_cannot_impersonate_constructor_built_gate() -> None:
    forged = object.__new__(Lfm2CpcfReviewedPromotionGate)

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="constructor-authenticated",
    ):
        authenticate_lfm2_cpcf_reviewed_promotion_gate(forged)
    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="constructor-authenticated",
    ):
        forged(_ReplayModel())


def test_replaced_replay_callable_rejects_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )
    monkeypatch.setattr(
        reviewed_gate_module,
        "_greedy_replay_rows",
        lambda model, tokenizer, reviewed: (),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="implementation object was replaced",
    ):
        gate(fixture["model"])


def test_replaced_v4_policy_parser_invalidates_implementation_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reviewed_gate_module,
        "parse_lfm2_cpcf_short_conv_v4_solver_policy",
        lambda value: value,
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="implementation object was replaced",
    ):
        authenticate_lfm2_cpcf_reviewed_gate_implementation()


def test_replay_callable_code_and_resources_are_reauthenticated() -> None:
    replay = reviewed_gate_module._greedy_replay_rows
    original_code = replay.__code__
    original_keyword_defaults = replay.__kwdefaults__

    def spoofed_replay(model: Any, tokenizer: Any, evidence: Any) -> tuple:
        del model, tokenizer, evidence
        return ()

    try:
        replay.__code__ = spoofed_replay.__code__
        with pytest.raises(
            Lfm2CpcfReviewedPromotionError,
            match="callable code/resources changed",
        ):
            authenticate_lfm2_cpcf_reviewed_gate_implementation()
    finally:
        replay.__code__ = original_code

    try:
        replay.__kwdefaults__ = {"tampered_resource": True}
        with pytest.raises(
            Lfm2CpcfReviewedPromotionError,
            match="callable code/resources changed",
        ):
            authenticate_lfm2_cpcf_reviewed_gate_implementation()
    finally:
        replay.__kwdefaults__ = original_keyword_defaults

    authenticate_lfm2_cpcf_reviewed_gate_implementation()


def test_gate_class_replacement_is_blocked_and_bypass_is_detected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )
    original = vars(Lfm2CpcfReviewedPromotionGate)["evaluate_model"]

    with pytest.raises(TypeError, match="implementation-sealed"):
        setattr(
            Lfm2CpcfReviewedPromotionGate,
            "evaluate_model",
            lambda self, model: model,
        )

    type.__setattr__(
        Lfm2CpcfReviewedPromotionGate,
        "evaluate_model",
        lambda self, model: model,
    )
    try:
        with pytest.raises(
            Lfm2CpcfReviewedPromotionError,
            match="implementation object was replaced",
        ):
            gate(fixture["model"])
        with pytest.raises(
            Lfm2CpcfReviewedPromotionError,
            match="implementation object was replaced",
        ):
            authenticate_lfm2_cpcf_reviewed_promotion_gate(gate)
    finally:
        type.__setattr__(
            Lfm2CpcfReviewedPromotionGate,
            "evaluate_model",
            original,
        )

    assert gate(fixture["model"]).accepted is True


def test_in_place_gate_method_code_mutation_is_detected_before_dispatch(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )
    method = vars(Lfm2CpcfReviewedPromotionGate)["evaluate_model"]
    original_code = method.__code__

    def spoofed_evaluate_model(self: Any, model: Any) -> Any:
        del self
        return model

    try:
        method.__code__ = spoofed_evaluate_model.__code__
        with pytest.raises(
            Lfm2CpcfReviewedPromotionError,
            match="callable code/resources changed",
        ):
            gate(fixture["model"])
        with pytest.raises(
            Lfm2CpcfReviewedPromotionError,
            match="callable code/resources changed",
        ):
            authenticate_lfm2_cpcf_reviewed_promotion_gate(gate)
    finally:
        method.__code__ = original_code

    assert gate(fixture["model"]).accepted is True


def test_constants_are_canonical_production_schemas() -> None:
    assert AUTOMATIC_RUNNER_REPORT_SCHEMA.endswith("-v1")
    assert COMPOSITE_PROMOTION_SCHEMA.endswith("-v1")
