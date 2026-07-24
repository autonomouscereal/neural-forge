from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from neural_forge.dynamic_fold_executor import fingerprint_model_state  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_cpcf_blank_target import (  # noqa: E402
    plan_and_synthesize_lfm2_cpcf_blank,
)
from neural_forge.hf_lfm2_interface_frontier import (  # noqa: E402
    Lfm2InterfaceFrontierError,
    Lfm2InterfaceRecipe,
    base_tied_target_budget_for_recipe,
    compile_lfm2_causal_residual_token_bridge,
    compile_lfm2_causal_trie_evidence,
    compile_lfm2_cached_logit_residual_head,
    compile_lfm2_interface_sequence_plan,
    compile_lfm2_inverse_bucket_output_table,
    compile_lfm2_inverse_output_plan,
    fold_lfm2_interface_candidate,
    plan_lfm2_dense_untied_target_rebalance,
    target_unique_parameters_for_recipe,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    seal_strict_tokenizer_bridge_artifact,
)
from experiments import run_lfm2_interface_frontier as interface_runner  # noqa: E402


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tokenizer(path: Path, vocabulary: dict[str, int]):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    path.mkdir()
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
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
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )


def _no_unk_tokenizer(path: Path, vocabulary: dict[str, int]):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    path.mkdir()
    backend = Tokenizer(WordLevel(vocabulary))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }
        ),
        encoding="utf-8",
    )
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"),
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )


def _parent():
    config = transformers.Lfm2Config(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    torch.manual_seed(91)
    return transformers.Lfm2ForCausalLM(config).eval()


def _case(tmp_path: Path):
    donor_tokens = [
        "<unk>",
        "<pad>",
        "<bos>",
        "<eos>",
        "a",
        "!",
        *[f"t{index}" for index in range(6, 16)],
        "a!",
        *[f"t{index}" for index in range(17, 40)],
    ]
    donor_vocab = {token: index for index, token in enumerate(donor_tokens)}
    target_vocab = {
        token: index for index, token in enumerate(donor_tokens[:16])
    }
    donor = _tokenizer(tmp_path / "donor", donor_vocab)
    target_path = tmp_path / "target"
    target = _tokenizer(target_path, target_vocab)
    bridge = seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        target_path,
        donor_row_capacity=40,
        target_row_capacity=16,
    )
    parent = _parent()
    body_parameters = (
        sum(parameter.numel() for parameter in parent.parameters()) - 40 * 16
    )
    base_parameters = body_parameters + 16 * 16
    blank = plan_and_synthesize_lfm2_cpcf_blank(
        parent,
        parent_name="parent",
        parent_tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
        evidence_fingerprint=_sha("sealed-cache"),
        child_name="interface-frontier-child",
        target_unique_parameters=base_parameters,
        target_vocabulary_size=16,
        child_tokenizer_fingerprint=bridge.target_tokenizer_semantics_sha256,
        tokenizer_bridge_artifact=bridge,
        minimum_hidden_size=16,
        maximum_candidates=64,
        blank_seed=3,
        device="cpu",
        dtype=torch.float32,
    )
    return parent, blank, bridge, donor, target, base_parameters


def test_context_plan_captures_non_strict_boundary_token(tmp_path: Path) -> None:
    _parent_model, _blank, bridge, donor, target, _base = _case(tmp_path)
    plan = compile_lfm2_interface_sequence_plan(
        bridge,
        donor,
        target,
        contexts=("", "a"),
    )
    exclamation = plan["records"][5]
    assert [5] in exclamation["donor_sequences"]
    assert [16] in exclamation["donor_sequences"]
    assert plan["rows_with_context_variants"] >= 1
    assert plan["plan_sha256"]


def test_inverse_output_plan_covers_every_donor_row_once(tmp_path: Path) -> None:
    _parent_model, _blank, bridge, donor, target, _base = _case(tmp_path)
    plan = compile_lfm2_inverse_output_plan(bridge, donor, target)
    flattened = [value for bucket in plan["buckets"] for value in bucket]
    assert sorted(flattened) == list(range(40))
    assert plan["assigned_donor_row_count"] == 40
    assert plan["donor_rows_assigned_exactly_once"] is True


def test_inverse_output_plan_explicitly_sinks_no_unk_capacity_rows(
    tmp_path: Path,
) -> None:
    vocabulary = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        **{f"t{index}": index for index in range(3, 16)},
    }
    donor = _no_unk_tokenizer(tmp_path / "donor-no-unk", vocabulary)
    target_path = tmp_path / "target-no-unk"
    target = _no_unk_tokenizer(target_path, vocabulary)
    bridge = seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        target_path,
        donor_row_capacity=40,
        target_row_capacity=16,
    )
    assert donor.unk_token_id is None
    assert target.unk_token_id is None
    with pytest.raises(
        Lfm2InterfaceFrontierError,
        match="explicit fallback target row",
    ):
        compile_lfm2_inverse_output_plan(bridge, donor, target)
    plan = compile_lfm2_inverse_output_plan(
        bridge,
        donor,
        target,
        fallback_target_id=target.pad_token_id,
    )
    flattened = [value for bucket in plan["buckets"] for value in bucket]
    assert sorted(flattened) == list(range(40))
    assert plan["fallback_policy"] == "explicit-authenticated-target-row"
    assert plan["fallback_target_id"] == 0
    assert plan["donor_tokenizer_defined_row_count"] == 16
    assert plan["invalid_donor_capacity_row_count"] == 24
    assert plan["valid_empty_decomposition_count"] == 0
    assert plan["fallback_assignment_count"] == 24
    assert plan["empty_decomposition_count"] == 24
    assert set(range(16, 40)).issubset(plan["buckets"][0])


def test_context_tied_candidate_preserves_body_and_composes_rows(
    tmp_path: Path,
) -> None:
    parent, blank, bridge, donor, target, base_parameters = _case(tmp_path)
    parent_before = fingerprint_model_state(parent)
    recipe = Lfm2InterfaceRecipe(
        candidate_id="context-mean-norm-tied",
        input_strict_anchor_ppm=0,
        input_preserve_anchor_norm=True,
        output_mode="tied-to-input",
    )
    result = fold_lfm2_interface_candidate(
        parent,
        blank.model,
        bridge=bridge,
        donor_tokenizer=donor,
        target_tokenizer=target,
        blank_target_receipt=blank.as_dict(),
        recipe=recipe,
        maximum_target_unique_parameters=base_parameters,
        contexts=("", "a"),
    )
    receipt = result.as_dict()
    assert receipt["exact_parent_body_depth_width_preserved"] is True
    assert receipt["native_input_output_tying_preserved"] is True
    assert receipt["target_unique_parameters"] == base_parameters
    assert result.model.get_input_embeddings().weight is result.model.get_output_embeddings().weight
    assert not torch.equal(
        result.model.get_input_embeddings().weight[5],
        parent.get_input_embeddings().weight[5],
    )
    assert fingerprint_model_state(parent) == parent_before
    assert receipt["operation_audit"]["training_steps"] == 0


def test_asymmetric_candidate_materializes_dense_untied_output(
    tmp_path: Path,
) -> None:
    parent, blank, bridge, donor, target, base_parameters = _case(tmp_path)
    recipe = Lfm2InterfaceRecipe(
        candidate_id="context-input-strict-output-untied",
        input_strict_anchor_ppm=0,
        input_preserve_anchor_norm=True,
        output_mode="strict-row",
        output_strict_anchor_ppm=1_000_000,
    )
    expected = target_unique_parameters_for_recipe(
        base_tied_target_unique_parameters=base_parameters,
        target_vocabulary_size=16,
        hidden_size=16,
        recipe=recipe,
    )
    result = fold_lfm2_interface_candidate(
        parent,
        blank.model,
        bridge=bridge,
        donor_tokenizer=donor,
        target_tokenizer=target,
        blank_target_receipt=blank.as_dict(),
        recipe=recipe,
        maximum_target_unique_parameters=expected,
        contexts=("", "a"),
    )
    receipt = result.as_dict()
    input_weight = result.model.get_input_embeddings().weight
    output_weight = result.model.get_output_embeddings().weight
    mapping = torch.tensor(
        bridge.transport_profile.target_to_donor_row_indices,
        dtype=torch.long,
    )
    assert input_weight is not output_weight
    assert receipt["native_input_output_tying_preserved"] is False
    assert receipt["compiler_owned_untied_blank"]["added_dense_parameters"] == 256
    assert receipt["target_unique_parameters"] == expected
    assert torch.equal(
        output_weight,
        parent.get_output_embeddings().weight.index_select(0, mapping),
    )
    assert receipt["operation_audit"]["runtime_shared_parent_parameters"] == 0


def test_inverse_bucket_candidate_uses_all_parent_output_rows(
    tmp_path: Path,
) -> None:
    parent, blank, bridge, donor, target, base_parameters = _case(tmp_path)
    recipe = Lfm2InterfaceRecipe(
        candidate_id="inverse-bucket-output",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="inverse-bucket-centroid",
        output_strict_anchor_ppm=0,
        output_preserve_anchor_norm=True,
        output_bucket_weighting="row-norm",
    )
    expected = base_parameters + 16 * 16
    result = fold_lfm2_interface_candidate(
        parent,
        blank.model,
        bridge=bridge,
        donor_tokenizer=donor,
        target_tokenizer=target,
        blank_target_receipt=blank.as_dict(),
        recipe=recipe,
        maximum_target_unique_parameters=expected,
        contexts=("", "a"),
    )
    receipt = result.as_dict()
    assert receipt["inverse_output_plan"]["assigned_donor_row_count"] == 40
    assert receipt["inverse_output_plan"]["donor_rows_assigned_exactly_once"] is True
    assert receipt["inverse_output_plan_sha256"]
    assert result.model.get_input_embeddings().weight is not result.model.get_output_embeddings().weight


def test_cached_logit_residual_head_is_closed_form_and_preserves_specials(
    tmp_path: Path,
) -> None:
    parent, _blank, bridge, donor, target, _base_parameters = _case(tmp_path)
    inverse_plan = compile_lfm2_inverse_output_plan(bridge, donor, target)
    mapping = tuple(bridge.transport_profile.target_to_donor_row_indices)
    structural = compile_lfm2_inverse_bucket_output_table(
        parent.get_output_embeddings().weight,
        inverse_plan,
        mapping,
        strict_anchor_ppm=250_000,
        preserve_anchor_norm=True,
        weighting="row-norm",
    )
    torch.manual_seed(18)
    hidden = torch.randn(8, 16)
    donor_logits = hidden.matmul(parent.get_output_embeddings().weight.t())
    result = compile_lfm2_cached_logit_residual_head(
        structural,
        hidden,
        donor_logits,
        inverse_plan,
        source_evidence_sha256=_sha("sealed-terminal-evidence"),
        ridge=1e-3,
        residual_scale_ppm=1_000_000,
        protected_target_ids=(0, 1, 2, 3),
        solve_device="cpu",
    )
    receipt = result.as_dict()
    assert receipt["relative_centered_error_after"] < receipt[
        "relative_centered_error_before"
    ]
    assert receipt["diagnostic_metrics_are_not_acceptance_gates"] is True
    assert receipt["operation_audit"]["training_steps"] == 0
    assert receipt["operation_audit"]["autograd_grad_calls"] == 0
    torch.testing.assert_close(result.table[:4], structural[:4], rtol=0, atol=0)


def test_cached_logit_residual_candidate_installs_as_dense_untied_head(
    tmp_path: Path,
) -> None:
    parent, blank, bridge, donor, target, base_parameters = _case(tmp_path)
    inverse_plan = compile_lfm2_inverse_output_plan(bridge, donor, target)
    mapping = tuple(bridge.transport_profile.target_to_donor_row_indices)
    structural = compile_lfm2_inverse_bucket_output_table(
        parent.get_output_embeddings().weight,
        inverse_plan,
        mapping,
        strict_anchor_ppm=250_000,
        preserve_anchor_norm=True,
        weighting="row-norm",
    )
    torch.manual_seed(19)
    hidden = torch.randn(8, 16)
    donor_logits = hidden.matmul(parent.get_output_embeddings().weight.t())
    residual_head = compile_lfm2_cached_logit_residual_head(
        structural,
        hidden,
        donor_logits,
        inverse_plan,
        source_evidence_sha256=_sha("sealed-terminal-evidence"),
        ridge=1e-2,
        residual_scale_ppm=500_000,
        solve_device="cpu",
    )
    recipe = Lfm2InterfaceRecipe(
        candidate_id="cached-logit-residual",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="cached-logit-residual",
        output_strict_anchor_ppm=250_000,
        output_preserve_anchor_norm=True,
        output_bucket_weighting="row-norm",
    )
    expected = base_parameters + 16 * 16
    folded = fold_lfm2_interface_candidate(
        parent,
        blank.model,
        bridge=bridge,
        donor_tokenizer=donor,
        target_tokenizer=target,
        blank_target_receipt=blank.as_dict(),
        recipe=recipe,
        maximum_target_unique_parameters=expected,
        contexts=("", "a"),
        cached_logit_residual_head=residual_head,
    )
    receipt = folded.as_dict()
    assert receipt["cached_logit_residual_head"]["result_sha256"] == (
        residual_head.as_dict()["result_sha256"]
    )
    assert folded.model.get_input_embeddings().weight is not folded.model.get_output_embeddings().weight
    torch.testing.assert_close(
        folded.model.get_output_embeddings().weight,
        residual_head.table,
        rtol=0,
        atol=0,
    )


def _plan_with_internal_target_prefixes(plan: dict) -> dict:
    body = dict(plan)
    body.pop("plan_sha256")
    donor_rows = int(body["donor_row_count"])
    target_rows = int(body["target_row_count"])
    decompositions = [[donor_id % target_rows] for donor_id in range(donor_rows)]
    for donor_id in range(donor_rows):
        if decompositions[donor_id][0] in {4, 7}:
            decompositions[donor_id] = [9]
    decompositions[0] = [4]
    decompositions[1] = [4, 5]
    decompositions[2] = [4, 6]
    decompositions[3] = [7, 8]
    buckets = [[] for _target_id in range(target_rows)]
    prefix_buckets = [[] for _target_id in range(target_rows)]
    for donor_id, decomposition in enumerate(decompositions):
        buckets[decomposition[0]].append(donor_id)
        prefix_buckets[decomposition[-1]].append(donor_id)
    body["donor_target_decompositions"] = decompositions
    body["buckets"] = buckets
    body["prefix_transition_buckets"] = prefix_buckets
    return {**body, "plan_sha256": canonical_sha256(body)}


def test_causal_trie_retains_roots_and_marks_terminal_prefix_ambiguity(
    tmp_path: Path,
) -> None:
    _parent_model, _blank, bridge, donor, target, _base = _case(tmp_path)
    inverse_plan = _plan_with_internal_target_prefixes(
        compile_lfm2_inverse_output_plan(bridge, donor, target)
    )
    donor_logits = torch.full((2, 40), -8.0)
    donor_logits[0, :4] = torch.tensor([9.0, 8.0, 7.0, 6.0])
    donor_logits[1, :4] = torch.tensor([8.0, 9.0, 6.0, 7.0])
    result = compile_lfm2_causal_trie_evidence(
        donor_logits,
        ((1, 2), (3, 4)),
        inverse_plan,
        source_evidence_sha256=_sha("sealed-terminal-evidence"),
        maximum_donor_candidates_per_root=4,
        maximum_internal_prefixes_per_root=4,
        maximum_total_internal_prefixes=8,
        log_zero_floor_delta=40.0,
    )
    receipt = result.as_dict()
    assert receipt["root_row_count"] == 2
    assert receipt["all_root_rows_retained"] is True
    assert receipt["internal_prefix_row_count"] >= 2
    assert receipt["ambiguous_terminal_prefix_node_count"] >= 1
    assert receipt["zero_mass_target_tokens_receive_finite_authenticated_floor"] is True
    assert bool(result.target_logit_support_mask[:2].all()) is True
    assert any(
        row["trie_prefix"] == [4]
        and row["next_donor_conditional_ambiguity"] is True
        for row in receipt["node_records"]
    )
    assert any(
        row["trie_prefix"] == [7]
        and row["next_donor_conditional_ambiguity"] is False
        for row in receipt["node_records"]
    )
    assert bool(torch.isfinite(result.target_logits).all()) is True


@pytest.mark.parametrize(
    ("evidence_rows", "expected_formulation"),
    ((8, "dual-evidence-row-gram"), (20, "primal-hidden-coordinate-gram")),
)
def test_causal_residual_bridge_selects_bounded_closed_form_solve(
    tmp_path: Path,
    evidence_rows: int,
    expected_formulation: str,
) -> None:
    parent, _blank, bridge, donor, target, _base = _case(tmp_path)
    inverse_plan = compile_lfm2_inverse_output_plan(bridge, donor, target)
    mapping = tuple(bridge.transport_profile.target_to_donor_row_indices)
    structural = compile_lfm2_inverse_bucket_output_table(
        parent.get_output_embeddings().weight,
        inverse_plan,
        mapping,
        strict_anchor_ppm=250_000,
        preserve_anchor_norm=True,
        weighting="row-norm",
    )
    torch.manual_seed(721 + evidence_rows)
    child_hidden = torch.randn(evidence_rows, 16)
    target_logits = torch.randn(evidence_rows, 16)
    support = torch.ones((evidence_rows, 16), dtype=torch.bool)
    support[0, -2:] = False
    result = compile_lfm2_causal_residual_token_bridge(
        structural,
        child_hidden,
        target_logits,
        support,
        inverse_plan,
        parent_terminal_evidence_sha256=_sha("parent-terminal"),
        child_feature_evidence_sha256=_sha("child-features"),
        prefix_alignment_sha256=_sha("semantic-prefixes"),
        trie_conditional_evidence_sha256=_sha("target-trie"),
        causal_input_table_sha256=_sha("causal-input-table"),
        causal_body_inventory_sha256=_sha("causal-body"),
        sequence_plan_sha256=_sha("sequence-plan"),
        ridge=1e-2,
        residual_scale_ppm=1_000_000,
        solve_device="cpu",
    )
    receipt = result.as_dict()
    assert receipt["solve_formulation"] == expected_formulation
    assert receipt["solve_gram_dimension"] == min(evidence_rows, 16)
    assert receipt["causal_child_residual_features_used"] is True
    assert receipt["compiled_against_parent_hidden_features"] is False
    assert receipt["diagnostic_metrics_are_not_acceptance_gates"] is True
    assert receipt["operation_audit"]["training_steps"] == 0


def test_tied_causal_bridge_materializes_as_explicit_perturbative_ablation(
    tmp_path: Path,
) -> None:
    parent, control_blank, bridge, donor, target, base_parameters = _case(tmp_path)
    control_recipe = Lfm2InterfaceRecipe(
        candidate_id="tied-structural-control",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="tied-to-input",
    )
    control = fold_lfm2_interface_candidate(
        parent,
        control_blank.model,
        bridge=bridge,
        donor_tokenizer=donor,
        target_tokenizer=target,
        blank_target_receipt=control_blank.as_dict(),
        recipe=control_recipe,
        maximum_target_unique_parameters=base_parameters,
        contexts=("", "a"),
    )
    tied_structural = control.model.get_input_embeddings().weight.detach().clone()
    inverse_plan = compile_lfm2_inverse_output_plan(bridge, donor, target)
    torch.manual_seed(827)
    child_hidden = torch.randn(8, 16)
    target_logits = torch.randn(8, 16)
    support = torch.ones((8, 16), dtype=torch.bool)
    causal = compile_lfm2_causal_residual_token_bridge(
        tied_structural,
        child_hidden,
        target_logits,
        support,
        inverse_plan,
        parent_terminal_evidence_sha256=_sha("parent-terminal"),
        child_feature_evidence_sha256=_sha("child-features"),
        prefix_alignment_sha256=_sha("semantic-prefixes"),
        trie_conditional_evidence_sha256=_sha("target-trie"),
        causal_input_table_sha256=control.as_dict()["input_table_sha256"],
        causal_body_inventory_sha256=control.as_dict()[
            "body_tensor_inventory_sha256"
        ],
        sequence_plan_sha256=control.as_dict()["sequence_plan_sha256"],
        ridge=1e-2,
        residual_scale_ppm=100_000,
        solve_device="cpu",
    )
    blank = plan_and_synthesize_lfm2_cpcf_blank(
        parent,
        parent_name="parent",
        parent_tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
        evidence_fingerprint=_sha("sealed-cache"),
        child_name="tied-causal-child",
        target_unique_parameters=base_parameters,
        target_vocabulary_size=16,
        child_tokenizer_fingerprint=bridge.target_tokenizer_semantics_sha256,
        tokenizer_bridge_artifact=bridge,
        minimum_hidden_size=16,
        maximum_candidates=64,
        blank_seed=3,
        device="cpu",
        dtype=torch.float32,
    )
    recipe = Lfm2InterfaceRecipe(
        candidate_id="tied-causal",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="causal-residual-token-bridge-tied",
        output_strict_anchor_ppm=250_000,
        output_bucket_weighting="row-norm",
    )
    with pytest.raises(Lfm2InterfaceFrontierError, match="differs"):
        fold_lfm2_interface_candidate(
            parent,
            blank.model,
            bridge=bridge,
            donor_tokenizer=donor,
            target_tokenizer=target,
            blank_target_receipt=blank.as_dict(),
            recipe=recipe,
            maximum_target_unique_parameters=base_parameters,
            contexts=("", "changed-context"),
            causal_residual_token_bridge=causal,
            precompiled_inverse_output_plan=inverse_plan,
        )
    altered_recipe = Lfm2InterfaceRecipe(
        candidate_id="tied-causal-altered-input",
        input_strict_anchor_ppm=0,
        input_preserve_anchor_norm=True,
        output_mode="causal-residual-token-bridge-tied",
        output_strict_anchor_ppm=250_000,
        output_bucket_weighting="row-norm",
    )
    with pytest.raises(Lfm2InterfaceFrontierError, match="differs"):
        fold_lfm2_interface_candidate(
            parent,
            blank.model,
            bridge=bridge,
            donor_tokenizer=donor,
            target_tokenizer=target,
            blank_target_receipt=blank.as_dict(),
            recipe=altered_recipe,
            maximum_target_unique_parameters=base_parameters,
            contexts=("", "a"),
            causal_residual_token_bridge=causal,
            precompiled_inverse_output_plan=inverse_plan,
        )
    folded = fold_lfm2_interface_candidate(
        parent,
        blank.model,
        bridge=bridge,
        donor_tokenizer=donor,
        target_tokenizer=target,
        blank_target_receipt=blank.as_dict(),
        recipe=recipe,
        maximum_target_unique_parameters=base_parameters,
        contexts=("", "a"),
        causal_residual_token_bridge=causal,
        precompiled_inverse_output_plan=inverse_plan,
    )
    receipt = folded.as_dict()
    assert folded.model.get_input_embeddings().weight is (
        folded.model.get_output_embeddings().weight
    )
    torch.testing.assert_close(
        folded.model.get_input_embeddings().weight,
        causal.table,
        rtol=0,
        atol=0,
    )
    assert receipt["causal_feature_coordinate_mode"] == (
        "tied-shared-table-perturbative-ablation"
    )
    assert receipt["one_shot_tied_causal_self_consistency_claimed"] is False
    assert receipt["postinstall_tied_feature_drift_audit_required"] is True


def test_untied_causal_bridge_preserves_feature_replay_input_coordinate(
    tmp_path: Path,
) -> None:
    parent, control_blank, bridge, donor, target, base_parameters = _case(tmp_path)
    control_recipe = Lfm2InterfaceRecipe(
        candidate_id="tied-structural-control",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="tied-to-input",
    )
    control = fold_lfm2_interface_candidate(
        parent,
        control_blank.model,
        bridge=bridge,
        donor_tokenizer=donor,
        target_tokenizer=target,
        blank_target_receipt=control_blank.as_dict(),
        recipe=control_recipe,
        maximum_target_unique_parameters=base_parameters,
        contexts=("", "a"),
    )
    inverse_plan = compile_lfm2_inverse_output_plan(bridge, donor, target)
    mapping = tuple(bridge.transport_profile.target_to_donor_row_indices)
    structural_output = compile_lfm2_inverse_bucket_output_table(
        parent.get_output_embeddings().weight,
        inverse_plan,
        mapping,
        strict_anchor_ppm=250_000,
        preserve_anchor_norm=True,
        weighting="row-norm",
    )
    torch.manual_seed(911)
    causal = compile_lfm2_causal_residual_token_bridge(
        structural_output,
        torch.randn(8, 16),
        torch.randn(8, 16),
        torch.ones((8, 16), dtype=torch.bool),
        inverse_plan,
        parent_terminal_evidence_sha256=_sha("parent-terminal"),
        child_feature_evidence_sha256=_sha("child-features"),
        prefix_alignment_sha256=_sha("semantic-prefixes"),
        trie_conditional_evidence_sha256=_sha("target-trie"),
        causal_input_table_sha256=control.as_dict()["input_table_sha256"],
        causal_body_inventory_sha256=control.as_dict()[
            "body_tensor_inventory_sha256"
        ],
        sequence_plan_sha256=control.as_dict()["sequence_plan_sha256"],
        ridge=1e-2,
        residual_scale_ppm=1_000_000,
        solve_device="cpu",
    )
    blank = plan_and_synthesize_lfm2_cpcf_blank(
        parent,
        parent_name="parent",
        parent_tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
        evidence_fingerprint=_sha("sealed-cache"),
        child_name="untied-causal-child",
        target_unique_parameters=base_parameters,
        target_vocabulary_size=16,
        child_tokenizer_fingerprint=bridge.target_tokenizer_semantics_sha256,
        tokenizer_bridge_artifact=bridge,
        minimum_hidden_size=16,
        maximum_candidates=64,
        blank_seed=3,
        device="cpu",
        dtype=torch.float32,
    )
    recipe = Lfm2InterfaceRecipe(
        candidate_id="untied-causal",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="causal-residual-token-bridge",
        output_strict_anchor_ppm=250_000,
        output_bucket_weighting="row-norm",
    )
    expected_parameters = base_parameters + 16 * 16
    folded = fold_lfm2_interface_candidate(
        parent,
        blank.model,
        bridge=bridge,
        donor_tokenizer=donor,
        target_tokenizer=target,
        blank_target_receipt=blank.as_dict(),
        recipe=recipe,
        maximum_target_unique_parameters=expected_parameters,
        contexts=("", "a"),
        causal_residual_token_bridge=causal,
        precompiled_inverse_output_plan=inverse_plan,
    )
    receipt = folded.as_dict()
    assert folded.model.get_input_embeddings().weight is not (
        folded.model.get_output_embeddings().weight
    )
    torch.testing.assert_close(
        folded.model.get_input_embeddings().weight,
        control.model.get_input_embeddings().weight,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        folded.model.get_output_embeddings().weight,
        causal.table,
        rtol=0,
        atol=0,
    )
    assert receipt["input_table_sha256"] == control.as_dict()["input_table_sha256"]
    assert receipt["body_tensor_inventory_sha256"] == control.as_dict()[
        "body_tensor_inventory_sha256"
    ]
    assert receipt["causal_feature_coordinate_mode"] == (
        "untied-output-exact-causal-coordinate"
    )
    assert receipt["postinstall_tied_feature_drift_audit_required"] is False


def test_runner_causal_child_replay_is_full_prefix_and_nonmutating() -> None:
    model = _parent()
    before = fingerprint_model_state(model)
    features, receipt = interface_runner._collect_child_terminal_features(
        model,
        ((2, 4), (2, 4, 5), (2, 6, 7, 8)),
        trie_evidence_sha256=_sha("target-trie"),
        semantic_alignment_sha256=_sha("semantic-prefixes"),
        device="cpu",
        evidence_role="unit-test-causal-replay",
    )
    assert tuple(features.shape) == (3, 16)
    assert bool(torch.isfinite(features).all()) is True
    assert fingerprint_model_state(model) == before
    assert receipt["full_prefix_use_cache_false"] is True
    assert receipt["operation_audit"]["child_forward_calls"] == 3
    assert receipt["operation_audit"]["training_steps"] == 0
    assert receipt["operation_audit"]["parent_forward_calls"] == 0


def test_asymmetric_candidate_fails_closed_above_target_size(
    tmp_path: Path,
) -> None:
    parent, blank, bridge, donor, target, base_parameters = _case(tmp_path)
    recipe = Lfm2InterfaceRecipe(
        candidate_id="untied",
        input_strict_anchor_ppm=0,
        input_preserve_anchor_norm=True,
        output_mode="strict-row",
    )
    with pytest.raises(Lfm2InterfaceFrontierError, match="target-size"):
        fold_lfm2_interface_candidate(
            parent,
            blank.model,
            bridge=bridge,
            donor_tokenizer=donor,
            target_tokenizer=target,
            blank_target_receipt=blank.as_dict(),
            recipe=recipe,
            maximum_target_unique_parameters=base_parameters,
            contexts=("", "a"),
        )


def test_exact4x_untied_interface_budget_rebalances_only_ffn_width() -> None:
    recipe = Lfm2InterfaceRecipe(
        candidate_id="exact4x-untied",
        input_strict_anchor_ppm=500_000,
        input_preserve_anchor_norm=True,
        output_mode="cached-logit-residual",
    )
    assert base_tied_target_budget_for_recipe(
        maximum_total_target_unique_parameters=57_423_296,
        target_vocabulary_size=4_096,
        hidden_size=776,
        recipe=recipe,
    ) == 54_244_800
    plan = plan_lfm2_dense_untied_target_rebalance(
        parent_unique_parameters=229_693_184,
        base_tied_target_unique_parameters=57_405_368,
        target_vocabulary_size=4_096,
        hidden_size=776,
        layer_count=14,
        current_intermediate_size=797,
        target_compaction_denominator=4,
    )
    assert plan["target_unique_parameter_ceiling"] == 57_423_296
    assert plan["added_dense_untied_output_parameters"] == 3_178_496
    assert plan["intermediate_size_reduction"] == 97
    assert plan["target_intermediate_size"] == 700
    assert plan["removed_body_parameters"] == 3_161_424
    assert plan["target_unique_parameters"] == 57_422_440
    assert plan["parameter_ceiling_spare"] == 856
    assert plan["residual_width_preserved"] is True
    assert plan["attention_geometry_preserved"] is True
