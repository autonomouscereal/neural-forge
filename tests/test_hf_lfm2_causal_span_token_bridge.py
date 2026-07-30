from __future__ import annotations

import math
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_causal_span_token_bridge import (
    LFM2_CAUSAL_SPAN_CONTEXTUAL_PROGRESS_SCHEMA,
    Lfm2CausalSpanHead,
    Lfm2CausalSpanTokenBridgeError,
    build_lfm2_full_donor_span_transition_evidence,
    compile_lfm2_causal_span_token_bridge_plan,
    compile_lfm2_causal_span_token_bridge_plan_from_local_tokenizers,
    compile_lfm2_full_donor_contextual_transition_evidence,
    seal_lfm2_causal_span_root_logits_evidence,
)
from neural_forge.hf_tokenizer_bridge import (
    seal_strict_tokenizer_bridge_artifact,
)


def _bpe_tokenizer(
    path: Path,
    vocabulary: dict[str, int],
    merges: list[tuple[str, str]],
):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    path.mkdir()
    backend = tokenizers.Tokenizer(
        tokenizers.models.BPE(
            vocab=vocabulary,
            merges=merges,
            unk_token="<unk>",
        )
    )
    backend.decoder = tokenizers.decoders.Fuse()
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
    return transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )


def _contextual_bridge_case(tmp_path: Path):
    vocabulary = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "a": 4,
        "b": 5,
        "ab": 6,
        "c": 7,
        "é": 8,
    }
    donor = _bpe_tokenizer(tmp_path / "donor", vocabulary, [])
    target_path = tmp_path / "target"
    target = _bpe_tokenizer(target_path, vocabulary, [("a", "b")])
    bridge = seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        target_path,
        donor_row_capacity=len(vocabulary),
        target_row_capacity=len(vocabulary),
    )
    return bridge, donor, target, vocabulary


def _compile(
    logits: torch.Tensor,
    transitions,
    *,
    target_rows: int = 4,
    chunk: int = 1,
):
    parent = seal_lfm2_causal_span_root_logits_evidence(
        logits,
        lineage={"cache_key": "a" * 64},
    )
    evidence = build_lfm2_full_donor_span_transition_evidence(
        logits,
        parent_root_logits_evidence=parent,
        root_terminal_row_indices=[0],
        root_target_prefix_token_ids=[[3]],
        target_row_count=target_rows,
        contextual_transitions=[transitions],
    )
    return compile_lfm2_causal_span_token_bridge_plan(
        logits,
        evidence,
        maximum_rows_per_chunk=chunk,
    )


def test_full_donor_root_and_internal_prefix_mass_is_exact_and_deterministic() -> None:
    logits = torch.tensor(
        [[math.log(0.60), math.log(0.30), math.log(0.10)]],
        dtype=torch.float32,
    )
    transitions = [
        {"donor_id": 0, "kind": "append", "target_append_token_ids": [1, 2]},
        {"donor_id": 1, "kind": "append", "target_append_token_ids": [1, 3]},
        {"donor_id": 2, "kind": "append", "target_append_token_ids": [0]},
    ]
    first = _compile(logits, transitions, chunk=1)
    second = _compile(logits, transitions, chunk=1)

    assert first.as_dict() == second.as_dict()
    assert torch.equal(first.root_target_log_mass, second.root_target_log_mass)
    assert first.receipt["source_dtype"] == "float32"
    assert first.receipt["accumulation_dtype"] == "float64"
    assert first.receipt["maximum_rows_per_chunk"] == 1
    assert first.receipt["limitation_transition_count"] == 0
    assert first.receipt["maximum_root_log_mass_conservation_error"] < 1e-12
    assert first.receipt["maximum_internal_log_mass_conservation_error"] < 1e-12

    root = first.root_state_id(0)
    assert first.state_prefix_token_ids[root] == ()
    assert torch.allclose(
        first.root_target_log_mass[0, torch.tensor([0, 1])],
        torch.tensor([math.log(0.10), math.log(0.90)], dtype=torch.float64),
        atol=1e-7,
        rtol=0,
    )
    internal = first.state_prefix_token_ids.index((1,))
    probabilities, support = first.state_conditional_row(internal)
    assert support.tolist() == [False, False, True, True]
    assert torch.allclose(
        probabilities[torch.tensor([2, 3])].exp(),
        torch.tensor([2.0 / 3.0, 1.0 / 3.0], dtype=torch.float64),
        atol=1e-7,
        rtol=0,
    )
    assert torch.allclose(
        torch.logsumexp(
            first.root_target_log_mass[0][first.root_target_support_mask[0]],
            dim=0,
        ),
        first.root_full_donor_log_mass[0],
        atol=1e-12,
        rtol=0,
    )
    assert first.receipt["operation_audit"] == {
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
    }


def test_suffix_rewrite_mass_is_conserved_but_not_misreported_as_an_edge() -> None:
    logits = torch.tensor([[math.log(0.7), math.log(0.3)]], dtype=torch.float64)
    transitions = [
        {"donor_id": 0, "kind": "append", "target_append_token_ids": [1]},
        {
            "donor_id": 1,
            "kind": "suffix-rewrite",
            "reason": "target-suffix-rewrite",
            "candidate_target_token_ids": [2, 3],
            "target_common_prefix_tokens": 0,
            "target_replaced_root_suffix_tokens": 1,
        },
    ]
    plan = _compile(logits, transitions)
    receipt = plan.as_dict()
    assert receipt["limitation_kind_counts"] == {"suffix-rewrite": 1}
    assert receipt[
        "suffix_rewrite_transitions_are_not_installed_as_append_edges"
    ] is True
    assert receipt["exact_all_branch_autoregressive_equivalence_claimed"] is False
    assert bool(plan.state_unresolved_mask[plan.root_state_id(0)])
    representable = torch.logsumexp(
        plan.root_target_log_mass[0][plan.root_target_support_mask[0]], dim=0
    )
    reconstructed = torch.logaddexp(representable, plan.root_unresolved_log_mass[0])
    assert torch.allclose(
        reconstructed,
        plan.root_full_donor_log_mass[0],
        atol=1e-12,
        rtol=0,
    )
    with pytest.raises(Lfm2CausalSpanTokenBridgeError, match="unresolved"):
        Lfm2CausalSpanHead(plan)
    partial = Lfm2CausalSpanHead(plan, allow_unresolved_roots=True)
    assert partial.runtime_receipt["runtime_parent_or_sidecar_required"] is False


def test_stateful_head_represents_root_and_post_prefix_conditionals_global_head_cannot() -> None:
    logits = torch.tensor([[math.log(0.75), math.log(0.25)]], dtype=torch.float32)
    transitions = [
        {"donor_id": 0, "kind": "append", "target_append_token_ids": [1, 2]},
        {"donor_id": 1, "kind": "append", "target_append_token_ids": [1, 3]},
    ]
    plan = _compile(logits, transitions)
    head = Lfm2CausalSpanHead(plan)
    assert sum(parameter.numel() for parameter in head.parameters()) == 0

    hidden = torch.tensor([[1.0, -0.5]], dtype=torch.bfloat16)
    global_head = torch.nn.Linear(2, 4, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        global_head.weight.copy_(
            torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                dtype=torch.bfloat16,
            )
        )
    # A global head sees the same hidden vector and must emit the same row.
    assert torch.equal(global_head(hidden), global_head(hidden.clone()))

    root_state = head.start(0)
    root_logits = head.forward_state(hidden, root_state)
    assert root_logits.dtype == torch.bfloat16
    assert int(root_logits.argmax(dim=-1).item()) == 1

    after_first = head.advance(root_state, 1)
    assert not after_first.completed
    internal_logits = head.forward_state(hidden, after_first)
    assert int(internal_logits.argmax(dim=-1).item()) == 2
    assert not torch.equal(root_logits, internal_logits)
    probabilities = torch.softmax(internal_logits.float(), dim=-1)
    assert torch.allclose(
        probabilities[0, torch.tensor([2, 3])],
        torch.tensor([0.75, 0.25]),
        atol=2e-3,
        rtol=0,
    )
    complete = head.advance(after_first, 2)
    assert complete.completed and complete.terminal_available
    with pytest.raises(Lfm2CausalSpanTokenBridgeError, match="next-donor"):
        head.commit_terminal(complete)
    reset = head.commit_terminal(complete, next_root_index=0)
    assert reset == root_state
    assert head.runtime_receipt["state_is_child_owned"] is True
    assert head.runtime_receipt["learned_parameter_count"] == 0


def test_terminal_continuation_ambiguity_is_explicit_and_fails_closed() -> None:
    logits = torch.tensor([[math.log(0.4), math.log(0.6)]], dtype=torch.float32)
    transitions = [
        {"donor_id": 0, "kind": "append", "target_append_token_ids": [1]},
        {"donor_id": 1, "kind": "append", "target_append_token_ids": [1, 2]},
    ]
    plan = _compile(logits, transitions)
    assert plan.receipt["ambiguous_terminal_state_count"] == 1
    head = Lfm2CausalSpanHead(plan)
    ambiguous = head.advance(head.start(0), 1)
    assert ambiguous.terminal_available and ambiguous.terminal_ambiguous
    with pytest.raises(Lfm2CausalSpanTokenBridgeError, match="next-donor"):
        head.forward_state(torch.ones((1, 2)), ambiguous)

    acknowledged = Lfm2CausalSpanHead(plan, allow_terminal_ambiguity=True)
    continuation = acknowledged.forward_state(torch.ones((1, 2)), ambiguous)
    assert int(continuation.argmax(dim=-1).item()) == 2
    committed = acknowledged.commit_terminal(ambiguous, next_root_index=0)
    assert committed.state_id == plan.root_state_id(0)


def test_tampered_transition_and_plan_receipts_fail_closed() -> None:
    logits = torch.tensor([[0.0, -1.0]], dtype=torch.float32)
    parent = seal_lfm2_causal_span_root_logits_evidence(logits)
    evidence = build_lfm2_full_donor_span_transition_evidence(
        logits,
        parent_root_logits_evidence=parent,
        root_terminal_row_indices=[0],
        target_row_count=3,
        donor_target_decompositions=[[1], [1, 2]],
    )
    tampered = deepcopy(evidence)
    tampered["donor_target_decompositions"][0] = [2]
    with pytest.raises(Lfm2CausalSpanTokenBridgeError, match="seal"):
        compile_lfm2_causal_span_token_bridge_plan(logits, tampered)
    tampered["evidence_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "evidence_sha256"}
    )
    with pytest.raises(Lfm2CausalSpanTokenBridgeError, match="inventory is unsealed"):
        compile_lfm2_causal_span_token_bridge_plan(logits, tampered)

    plan = compile_lfm2_causal_span_token_bridge_plan(logits, evidence)
    plan_receipt = plan.as_dict()
    plan_receipt["edge_count"] += 1
    plan_receipt["result_sha256"] = canonical_sha256(
        {key: value for key, value in plan_receipt.items() if key != "result_sha256"}
    )
    with pytest.raises(Lfm2CausalSpanTokenBridgeError, match="authenticate"):
        type(plan)(
            root_state_ids=plan.root_state_ids,
            state_root_indices=plan.state_root_indices,
            state_prefix_token_ids=plan.state_prefix_token_ids,
            state_edge_offsets=plan.state_edge_offsets,
            edge_target_ids=plan.edge_target_ids,
            edge_next_state_ids=plan.edge_next_state_ids,
            edge_log_mass=plan.edge_log_mass,
            edge_log_probability=plan.edge_log_probability,
            state_log_mass=plan.state_log_mass,
            state_terminal_log_mass=plan.state_terminal_log_mass,
            state_terminal_mask=plan.state_terminal_mask,
            state_unresolved_log_mass=plan.state_unresolved_log_mass,
            state_unresolved_mask=plan.state_unresolved_mask,
            root_target_log_mass=plan.root_target_log_mass,
            root_target_support_mask=plan.root_target_support_mask,
            root_full_donor_log_mass=plan.root_full_donor_log_mass,
            root_unresolved_log_mass=plan.root_unresolved_log_mass,
            receipt=plan_receipt,
        )


def test_local_tokenizer_adapter_covers_all_rows_unicode_specials_and_suffix_mass(
    tmp_path: Path,
) -> None:
    bridge, donor, target, vocabulary = _contextual_bridge_case(tmp_path)
    logits = torch.tensor(
        [[-0.1 * index for index in range(len(vocabulary))]],
        dtype=torch.float64,
    )
    parent = seal_lfm2_causal_span_root_logits_evidence(
        logits,
        lineage={"cache_key": "a" * 64},
    )
    root = {
        "terminal_row_index": 0,
        "batch_index": 0,
        "phase": "prefill",
        "phase_row_index": 0,
        "donor_boundary_position": 0,
        "target_boundary_position": 0,
        "donor_prefix_token_ids": [vocabulary["a"]],
        "target_prefix_token_ids": [vocabulary["a"]],
        "semantic_alignment_sha256": "b" * 64,
    }
    progress = []
    evidence = compile_lfm2_full_donor_contextual_transition_evidence(
        logits,
        [root],
        bridge,
        donor,
        target,
        parent_root_logits_evidence=parent,
        maximum_donor_rows_per_chunk=2,
        progress_callback=progress.append,
    )

    assert evidence["donor_defined_row_count"] == len(vocabulary)
    assert evidence["roots"][0]["transition_count"] == len(vocabulary)
    assert evidence["every_defined_donor_row_classified_once_per_root"] is True
    assert evidence["classification_is_exhaustive_not_topk"] is True
    assert evidence["classification_counts"] == {
        "append": 4,
        "authenticated-special-control": 4,
        "nonappend-only": 0,
        "roundtrip-mismatch": 0,
        "suffix-rewrite": 1,
        "unmapped-control": 0,
        "unsupported": 0,
        "zero-width": 0,
    }
    transitions = evidence["roots"][0]["transitions"]
    assert [row["donor_id"] for row in transitions] == list(
        range(len(vocabulary))
    )
    assert transitions[vocabulary["<eos>"]]["kind"] == "append"
    assert transitions[vocabulary["<eos>"]]["append_classification"] == (
        "authenticated-special-control"
    )
    assert transitions[vocabulary["b"]]["kind"] == "suffix-rewrite"
    assert transitions[vocabulary["b"]]["candidate_target_token_ids"] == [
        vocabulary["ab"]
    ]
    assert transitions[vocabulary["b"]]["target_replaced_root_suffix_tokens"] == 1
    assert transitions[vocabulary["é"]]["kind"] == "append"
    assert transitions[vocabulary["é"]]["target_append_token_ids"] == [
        vocabulary["é"]
    ]
    assert transitions[vocabulary["é"]]["candidate_text_sha256"] == (
        canonical_sha256("aé")
    )

    assert [record["stage"] for record in progress] == [
        "start",
        "root-start",
        "chunk-complete",
        "chunk-complete",
        "chunk-complete",
        "chunk-complete",
        "chunk-complete",
        "root-complete",
        "classification-complete",
    ]
    previous = None
    for ordinal, immutable_record in enumerate(progress):
        record = dict(immutable_record)
        assert record["schema"] == LFM2_CAUSAL_SPAN_CONTEXTUAL_PROGRESS_SCHEMA
        assert record["ordinal"] == ordinal
        assert record["previous_progress_sha256"] == previous
        supplied = record.pop("progress_sha256")
        assert supplied == canonical_sha256(record)
        previous = supplied
        with pytest.raises(TypeError):
            immutable_record["stage"] = "tampered"
    assert evidence["progress_record_count"] == len(progress)
    assert evidence["progress_callback_invocation_count"] == len(progress)
    assert evidence["progress_final_sha256"] == previous
    assert "donor.batch_decode" in evidence["tokenizer_methods"]
    assert "target.__call__" in evidence["tokenizer_methods"]
    assert "target.batch_decode" in evidence["tokenizer_methods"]
    assert evidence["tokenizer_operation_audit"]["parent_forward_calls"] == 0
    assert evidence["tokenizer_operation_audit"]["parent_trace_recaptures"] == 0
    assert evidence["tokenizer_operation_audit"]["network_access"] is False

    plan = compile_lfm2_causal_span_token_bridge_plan(logits, evidence)
    representable = torch.logsumexp(
        plan.root_target_log_mass[0][plan.root_target_support_mask[0]],
        dim=0,
    )
    reconstructed = torch.logaddexp(representable, plan.root_unresolved_log_mass[0])
    assert torch.allclose(
        reconstructed,
        plan.root_full_donor_log_mass[0],
        atol=1e-12,
        rtol=0,
    )
    assert plan.receipt["limitation_kind_counts"] == {"suffix-rewrite": 1}
    assert plan.receipt["maximum_root_log_mass_conservation_error"] < 1e-12

    direct_plan = compile_lfm2_causal_span_token_bridge_plan_from_local_tokenizers(
        logits,
        [root],
        bridge,
        donor,
        target,
        parent_root_logits_evidence=parent,
        maximum_donor_rows_per_chunk=3,
        maximum_plan_rows_per_chunk=2,
    )
    assert direct_plan.receipt["maximum_rows_per_chunk"] == 2
    assert direct_plan.receipt["limitation_kind_counts"] == {"suffix-rewrite": 1}
    assert direct_plan.receipt["operation_audit"]["parent_forward_calls"] == 0
    assert direct_plan.state_prefix_token_ids == plan.state_prefix_token_ids
    for name in (
        "root_state_ids",
        "state_edge_offsets",
        "edge_target_ids",
        "edge_next_state_ids",
        "edge_log_mass",
        "root_target_log_mass",
        "root_unresolved_log_mass",
    ):
        assert torch.equal(getattr(direct_plan, name), getattr(plan, name))


def test_local_tokenizer_adapter_rejects_noncanonical_causal_root(
    tmp_path: Path,
) -> None:
    bridge, donor, target, vocabulary = _contextual_bridge_case(tmp_path)
    logits = torch.zeros((1, len(vocabulary)), dtype=torch.bfloat16)
    parent = seal_lfm2_causal_span_root_logits_evidence(logits)
    bad_root = {
        "terminal_row_index": 0,
        "batch_index": 0,
        "phase": "prefill",
        "phase_row_index": 0,
        "donor_boundary_position": 0,
        "target_boundary_position": 0,
        "donor_prefix_token_ids": [vocabulary["a"]],
        "target_prefix_token_ids": [vocabulary["ab"]],
        "semantic_alignment_sha256": "c" * 64,
    }
    with pytest.raises(Lfm2CausalSpanTokenBridgeError, match="canonical target"):
        compile_lfm2_full_donor_contextual_transition_evidence(
            logits,
            [bad_root],
            bridge,
            donor,
            target,
            parent_root_logits_evidence=parent,
            maximum_donor_rows_per_chunk=2,
        )
