from __future__ import annotations

import hashlib
import json

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import (
    fingerprint_model_runtime_state,
    fingerprint_model_state,
)
from neural_forge.dynamic_model_inspector import default_inspector_registry
from neural_forge.dynamic_target_synthesis import serialized_inference_state_inventory
from neural_forge.hf_lfm2_exact4x_composite_interface import (
    LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA,
    _parameter_topology,
)
from neural_forge.hf_lfm2_interface_frontier import (
    LFM2_CAUSAL_CONTEXT_TRANSITION_PLAN_SCHEMA,
    LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA,
    LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA,
    LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA,
    LFM2_UNSUPPORTED_LOGIT_POLICY_ADAPTIVE_STRONGEST_SURVIVING_BACKGROUND_SEAL,
    compile_lfm2_causal_trie_evidence,
)
import neural_forge.hf_lfm2_interface_frontier as frontier
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_direct_local import (
    Lfm2DirectLocalCheckpointIO,
    load_lfm2_direct_local_tokenizer,
)
import neural_forge.hf_lfm2_pre_causal_body_resume_capsule as body_capsule
from neural_forge.hf_lfm2_pre_causal_resume_driver import (
    Lfm2PinnedCausalResumeEvidence,
    Lfm2PreCausalResumeError,
    restore_lfm2_causal_trie_evidence_from_receipt,
    resume_lfm2_pre_causal_body,
)
import neural_forge.hf_lfm2_pre_causal_resume_driver as driver
from neural_forge.hf_tokenizer_bridge import (
    seal_strict_tokenizer_bridge_artifact,
    strict_tokenizer_bridge_target_package_fingerprint,
)
from neural_forge.lfm2_exact4x_contract import EXPECTED_PARENT_UNIQUE_PARAMETERS


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sealed(body: dict, field: str) -> dict:
    return {**body, field: canonical_sha256(body)}


def _trie_receipt() -> tuple[dict, torch.Tensor, torch.Tensor]:
    shared = {
        "cache_key": _sha("cache"),
        "parent_identity_sha256": _sha("parent"),
        "selective_session_sha256": _sha("session"),
        "manifest_sha256": _sha("manifest"),
        "completion_sha256": _sha("completion"),
        "semantic_alignment_sha256": _sha("alignment"),
        "tokenizer_live_session_sha256": _sha("tokenizer-session"),
        "causal_root_rows_sha256": _sha("roots"),
        "bridge_artifact_sha256": _sha("bridge"),
        "bridge_profile_sha256": _sha("bridge-profile"),
        "donor_tokenizer_semantics_sha256": _sha("donor-tokenizer"),
        "target_tokenizer_semantics_sha256": _sha("target-tokenizer"),
    }
    candidates = [
        {
            "rank": 0,
            "donor_id": 10,
            "donor_logit": 1.25,
            "target_append_token_ids": [4],
        },
        {
            "rank": 1,
            "donor_id": 11,
            "donor_logit": 0.5,
            "target_append_token_ids": [4, 5],
        },
        {
            "rank": 2,
            "donor_id": 12,
            "donor_logit": -0.25,
            "target_append_token_ids": [7, 8],
        },
    ]
    root_body = {
        "schema": LFM2_CAUSAL_CONTEXT_TRANSITION_PLAN_SCHEMA,
        "root_index": 0,
        "target_root_token_ids": [3],
        "append_only_candidates": candidates,
    }
    root = _sealed(root_body, "root_transition_plan_sha256")
    transition_body = {
        "schema": LFM2_CAUSAL_CONTEXT_TRANSITION_PLAN_SCHEMA,
        "target_row_count": 12,
        "root_transition_plans": [root],
    }
    transition = _sealed(transition_body, "plan_sha256")
    root_row = torch.zeros(12, dtype=torch.float32)
    root_row[4] = torch.logsumexp(torch.tensor([1.25, 0.5]), dim=0)
    root_row[7] = -0.25
    node_row = torch.zeros(12, dtype=torch.float32)
    node_row[5] = 0.5
    logits = torch.stack((root_row, node_row)).contiguous()
    support = logits != 0
    prefixes = [[3], [3, 4]]
    body = {
        "schema": LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA,
        "parent_terminal_evidence_sha256": _sha("terminal"),
        **shared,
        "contextual_transition_plan_sha256": transition["plan_sha256"],
        "contextual_transition_plan": transition,
        "target_prefix_rows_sha256": canonical_sha256(prefixes),
        "target_logits_sha256": frontier._tensor_sha256(logits),
        "target_logit_support_mask_sha256": frontier._tensor_sha256(support),
        "root_row_count": 1,
        "internal_prefix_row_count": 1,
        "total_row_count": 2,
        "node_records": [
            {
                "root_index": 0,
                "root_transition_plan_sha256": root[
                    "root_transition_plan_sha256"
                ],
                "trie_prefix": [4],
                "continuation_target_ids": [5],
            }
        ],
    }
    return _sealed(body, "result_sha256"), logits, support


def test_restore_causal_trie_replays_sealed_float32_reductions() -> None:
    receipt, logits, support = _trie_receipt()

    restored = restore_lfm2_causal_trie_evidence_from_receipt(receipt)

    assert restored.as_dict() == receipt
    assert restored.target_prefix_token_ids == ((3,), (3, 4))
    torch.testing.assert_close(restored.target_logits, logits, rtol=0, atol=0)
    assert torch.equal(restored.target_logit_support_mask, support)


def test_restore_causal_trie_rejects_candidate_tampering() -> None:
    receipt, _logits, _support = _trie_receipt()
    receipt["contextual_transition_plan"]["root_transition_plans"][0][
        "append_only_candidates"
    ][0]["donor_logit"] = 99.0

    with pytest.raises(Lfm2PreCausalResumeError, match="seal or schema"):
        restore_lfm2_causal_trie_evidence_from_receipt(receipt)


def test_pinned_resume_evidence_binds_terminal_trie_sequence_and_structure() -> None:
    trie_receipt, _logits, _support = _trie_receipt()
    shared = {
        name: trie_receipt[name]
        for name in (
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
    }
    terminal_body = {
        "schema": LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA,
        **shared,
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
    terminal = _sealed(terminal_body, "evidence_sha256")
    trie_body = dict(trie_receipt)
    trie_body.pop("result_sha256")
    trie_body["parent_terminal_evidence_sha256"] = terminal["evidence_sha256"]
    trie_receipt = _sealed(trie_body, "result_sha256")
    trie = restore_lfm2_causal_trie_evidence_from_receipt(trie_receipt)
    sequence_body = {
        "schema": LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA,
        "bridge_artifact_sha256": shared["bridge_artifact_sha256"],
        "bridge_profile_sha256": shared["bridge_profile_sha256"],
        "donor_tokenizer_semantics_sha256": shared[
            "donor_tokenizer_semantics_sha256"
        ],
        "target_tokenizer_semantics_sha256": shared[
            "target_tokenizer_semantics_sha256"
        ],
    }
    sequence = _sealed(sequence_body, "plan_sha256")
    structural_body = {
        "schema": LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA,
        "sequence_plan_sha256": sequence["plan_sha256"],
    }
    structural = _sealed(structural_body, "result_sha256")

    evidence = Lfm2PinnedCausalResumeEvidence(
        terminal_receipt=terminal,
        causal_trie=trie,
        sequence_plan=sequence,
        structural_interface_receipt=structural,
    )

    assert evidence.as_dict()["typed_trie_restored_without_parent_cache_or_model"]
    assert evidence.as_dict()["parent_trace_regeneration_performed"] is False
    assert len(evidence.evidence_sha256) == 64


def _tokenizer_package(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    path.mkdir()
    vocabulary = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "hello": 4,
        "world": 5,
        "small": 6,
        "model": 7,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }
        ),
        encoding="utf-8",
    )
    return load_lfm2_direct_local_tokenizer(path)


def _tiny_untied_model():
    config = transformers.Lfm2Config(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    torch.manual_seed(90210)
    model = transformers.Lfm2ForCausalLM(config).eval()
    with torch.no_grad():
        model.get_output_embeddings().weight.copy_(
            model.get_input_embeddings().weight
        )
    return model


def test_restore_matches_actual_causal_trie_compiler_receipt(tmp_path) -> None:
    tokenizer_root = tmp_path / "restore-tokenizer"
    tokenizer = _tokenizer_package(tokenizer_root)
    bridge = seal_strict_tokenizer_bridge_artifact(
        tokenizer,
        tokenizer,
        tokenizer_root,
        donor_row_capacity=8,
        target_row_capacity=8,
    )
    donor_logits = torch.full((1, 8), -8.0, dtype=torch.float32)
    donor_logits[0, 5] = 3.0
    donor_logits[0, 3] = 2.0
    roots = [
        {
            "terminal_row_index": 0,
            "batch_index": 0,
            "phase": "decode.000",
            "phase_row_index": 0,
            "donor_boundary_position": 0,
            "target_boundary_position": 0,
            "donor_prefix_token_ids": [4],
            "target_prefix_token_ids": [4],
            "semantic_alignment_sha256": _sha("actual-alignment"),
        }
    ]
    canonical_roots = frontier._causal_root_evidence_rows(
        roots,
        donor_row_count=8,
        target_row_count=8,
        terminal_row_count=1,
    )
    terminal_body = {
        "schema": LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA,
        "cache_key": _sha("actual-cache"),
        "parent_identity_sha256": _sha("actual-parent"),
        "selective_session_sha256": _sha("actual-session"),
        "manifest_sha256": _sha("actual-manifest"),
        "completion_sha256": _sha("actual-completion"),
        "semantic_alignment_sha256": _sha("actual-alignments"),
        "tokenizer_live_session_sha256": _sha("actual-tokenizer-session"),
        "terminal_row_count": 1,
        "donor_vocabulary_size": 8,
        "donor_logits_sha256": frontier._tensor_sha256(donor_logits),
        "causal_root_row_count": 1,
        "causal_root_rows_sha256": canonical_sha256(list(canonical_roots)),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "donor_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "operation_audit": {
            "training_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
        },
    }
    terminal = _sealed(terminal_body, "evidence_sha256")
    compiled = compile_lfm2_causal_trie_evidence(
        donor_logits,
        roots,
        bridge,
        tokenizer,
        tokenizer,
        parent_terminal_evidence=terminal,
        maximum_donor_candidates_per_root=2,
        maximum_internal_prefixes_per_root=2,
        maximum_total_internal_prefixes=2,
    )

    restored = restore_lfm2_causal_trie_evidence_from_receipt(compiled.as_dict())

    assert restored.as_dict() == compiled.as_dict()
    assert restored.target_prefix_token_ids == compiled.target_prefix_token_ids
    torch.testing.assert_close(
        restored.target_logits,
        compiled.target_logits,
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        restored.target_logit_support_mask,
        compiled.target_logit_support_mask,
    )


def _actual_resume_problem(tmp_path):
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer = _tokenizer_package(tokenizer_root)
    bridge = seal_strict_tokenizer_bridge_artifact(
        tokenizer,
        tokenizer,
        tokenizer_root,
        donor_row_capacity=8,
        target_row_capacity=8,
    )
    checkpoint_io = Lfm2DirectLocalCheckpointIO(
        tokenizer_bridge=bridge,
        device="cpu",
        dtype=torch.float32,
    )
    model = _tiny_untied_model()
    child_name = "tiny-exact4x-resume"
    graph = default_inspector_registry().inspect(
        model,
        model_name=child_name,
        tokenizer_fingerprint=bridge.target_tokenizer_semantics_sha256,
    )
    sequence_body = {
        "schema": LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "donor_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
    }
    sequence = _sealed(sequence_body, "plan_sha256")
    input_sha = body_capsule._interface_tensor_sha256(
        model.get_input_embeddings().weight
    )
    structural_body = {
        "schema": LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA,
        "target_topology_sha256": canonical_sha256(_parameter_topology(model)),
        "exact_target_unique_parameters": graph.unique_parameter_count,
        "sequence_plan_sha256": sequence["plan_sha256"],
        "installed_input_table_sha256": input_sha,
        "installed_output_baseline_sha256": input_sha,
    }
    structural = _sealed(structural_body, "result_sha256")

    donor_logits = torch.full((1, 8), -8.0, dtype=torch.float32)
    donor_logits[0, 5] = 3.0
    donor_logits[0, 3] = 2.0
    roots = [
        {
            "terminal_row_index": 0,
            "batch_index": 0,
            "phase": "decode.000",
            "phase_row_index": 0,
            "donor_boundary_position": 0,
            "target_boundary_position": 0,
            "donor_prefix_token_ids": [4],
            "target_prefix_token_ids": [4],
            "semantic_alignment_sha256": _sha("real-root-alignment"),
        }
    ]
    canonical_roots = frontier._causal_root_evidence_rows(
        roots,
        donor_row_count=8,
        target_row_count=8,
        terminal_row_count=1,
    )
    shared = {
        "cache_key": _sha("real-cache"),
        "parent_identity_sha256": _sha("real-parent"),
        "selective_session_sha256": _sha("real-session"),
        "manifest_sha256": _sha("real-manifest"),
        "completion_sha256": _sha("real-completion"),
        "semantic_alignment_sha256": _sha("real-alignment"),
        "tokenizer_live_session_sha256": _sha("real-tokenizer-session"),
        "causal_root_rows_sha256": canonical_sha256(list(canonical_roots)),
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "donor_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
    }
    terminal_body = {
        "schema": LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA,
        **shared,
        "terminal_row_count": 1,
        "donor_vocabulary_size": 8,
        "donor_logits_sha256": frontier._tensor_sha256(donor_logits),
        "causal_root_row_count": 1,
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
    terminal = _sealed(terminal_body, "evidence_sha256")
    compiled_trie = compile_lfm2_causal_trie_evidence(
        donor_logits,
        roots,
        bridge,
        tokenizer,
        tokenizer,
        parent_terminal_evidence=terminal,
        maximum_donor_candidates_per_root=2,
        maximum_internal_prefixes_per_root=2,
        maximum_total_internal_prefixes=2,
    )
    trie = restore_lfm2_causal_trie_evidence_from_receipt(
        compiled_trie.as_dict()
    )
    evidence = Lfm2PinnedCausalResumeEvidence(
        terminal_receipt=terminal,
        causal_trie=trie,
        sequence_plan=sequence,
        structural_interface_receipt=structural,
    )
    inventory = serialized_inference_state_inventory(model)
    bindings = body_capsule.Lfm2PreCausalBodyResumeBindings(
        child_name=child_name,
        target_unique_parameters=graph.unique_parameter_count,
        target_logical_parameters=graph.logical_parameter_count,
        parent_identity_sha256=shared["parent_identity_sha256"],
        target_synthesis_fingerprint=_sha("target-synthesis"),
        target_topology_fingerprint=_sha("target-topology"),
        target_configuration_sha256=_sha("target-config"),
        target_state_inventory_sha256=canonical_sha256(inventory),
        native_target_contract_sha256=_sha("native-contract"),
        child_graph_fingerprint=graph.fingerprint,
        child_structural_topology_fingerprint=(
            graph.structural_topology_fingerprint
        ),
        bridge_artifact_sha256=bridge.artifact_sha256,
        bridge_profile_sha256=bridge.profile_fingerprint,
        target_tokenizer_semantics_sha256=(
            bridge.target_tokenizer_semantics_sha256
        ),
        target_tokenizer_package_fingerprint=(
            strict_tokenizer_bridge_target_package_fingerprint(bridge)
        ),
        body_fold_realization_sha256=_sha("body-realization"),
        global_residual_closure_selection_sha256=_sha("global-closure"),
        structural_interface_install_sha256=structural["result_sha256"],
        semantic_layer_stack_sha256=_sha("semantic-stack"),
        structural_attention_candidate_archive_manifest_sha256=_sha(
            "attention-archive"
        ),
        postfold_reclosure_seed_archive_manifest_sha256=_sha("seed-archive"),
        postfold_reclosure_seed_publication_sha256=_sha("seed-publication"),
        deferred_final_w2_install_sha256=_sha("final-w2"),
        moving_gauge_post_w2_finalization_sha256=_sha("moving-gauge"),
        body_model_state_sha256=fingerprint_model_state(model),
        body_runtime_state_sha256=fingerprint_model_runtime_state(model),
        body_lfm2_identity_sha256=fingerprint_lfm2_parent(model).identity_sha256,
        causal_body_inventory_sha256=(
            frontier.fingerprint_lfm2_causal_body_inventory(model)
        ),
        installed_input_table_sha256=input_sha,
        installed_output_baseline_sha256=input_sha,
    )
    reference = body_capsule.publish_lfm2_pre_causal_body_resume_capsule(
        model=model,
        checkpoint_io=checkpoint_io,
        bindings=bindings,
        output_directory=tmp_path / "body-capsule",
    )
    return tokenizer, checkpoint_io, reference, evidence, bindings


def test_parent_free_resume_publishes_reload_and_complete_dossier(
    tmp_path,
) -> None:
    tokenizer, checkpoint_io, reference, evidence, bindings = _actual_resume_problem(
        tmp_path
    )
    child_count = bindings.target_unique_parameters
    result = resume_lfm2_pre_causal_body(
        body_reference=reference,
        checkpoint_io=checkpoint_io,
        evidence=evidence,
        target_tokenizer=tokenizer,
        prompts=({"id": "hello", "prompt": "hello"},),
        output_directory=tmp_path / "resume-run",
        unsupported_logit_policy=(
            LFM2_UNSUPPORTED_LOGIT_POLICY_ADAPTIVE_STRONGEST_SURVIVING_BACKGROUND_SEAL
        ),
        model_device="cpu",
        solve_device="cpu",
        ridge_fraction=1e-2,
        residual_scale_ppm=1_000_000,
        row_norm_calibration_ppm=250_000,
        maximum_prompt_tokens=32,
        maximum_new_tokens=4,
    )

    assert result["parent_unique_parameters"] == EXPECTED_PARENT_UNIQUE_PARAMETERS
    assert result["parameter_compaction_ratio"] == pytest.approx(
        EXPECTED_PARENT_UNIQUE_PARAMETERS / child_count
    )
    assert result["unsupported_logit_policy"] == (
        LFM2_UNSUPPORTED_LOGIT_POLICY_ADAPTIVE_STRONGEST_SURVIVING_BACKGROUND_SEAL
    )
    assert result["standalone_fresh_reload_verified"] is True
    assert result["decoded_whole_responses_retained"] is True
    assert result["automatic_intelligence_verdict_performed"] is False
    assert result["operation_audit"]["parent_model_objects_constructed"] == 0
    assert result["operation_audit"]["parent_trace_recaptures"] == 0
    assert result["operation_audit"]["training_steps"] == 0
    dossier = json.loads(
        (tmp_path / "resume-run" / "whole-response-dossier.json").read_text()
    )
    assert dossier["response_count"] == 1
    assert isinstance(dossier["responses"][0]["response"], str)
    bridge_receipt = json.loads(
        (tmp_path / "resume-run" / "causal-output-bridge.json").read_text()
    )
    assert bridge_receipt["adaptive_background_policy_selected"] is True
    compatibility = json.loads(
        (tmp_path / "resume-run" / "structural-topology-compatibility.json").read_text()
    )
    assert compatibility["transition"] == (
        "all-trainable-structural-receipt-compatibility"
    )
    assert compatibility["final_parameters_frozen"] is True


def test_resume_boundary_rejects_post_construction_trie_tensor_tamper(
    tmp_path,
) -> None:
    _tokenizer, _io, _reference, evidence, _bindings = _actual_resume_problem(
        tmp_path
    )
    evidence.causal_trie.target_logits[0, 3] += 1.0

    with pytest.raises(
        Lfm2PreCausalResumeError,
        match="tensors or prefixes changed after construction",
    ):
        driver._reauthenticate_pinned_causal_resume_evidence(evidence)
