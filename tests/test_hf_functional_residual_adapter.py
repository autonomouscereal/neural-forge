from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import neural_forge.hf_functional_residual_adapter as adapter_module
from neural_forge.hf_attention_codebook import (
    GlobalAttentionCodebook,
    RaggedAttentionCodebookRegistry,
)
from neural_forge.hf_functional_residual_adapter import (
    P35_L19G4_TO_L21G6,
    FunctionalActionBatch,
    FunctionalActionEvaluation,
    FunctionalResidualAdapterAccounting,
    FunctionalResidualAdapterError,
    FunctionalResidualAdapterTopologyError,
    FunctionalResidualAdapterTrainingError,
    RedirectedAttentionSite,
    SparseFunctionalResidualAdapterRegistry,
    audit_redirected_attention_site,
    evaluate_functional_action_batches,
    install_functional_residual_adapter,
    train_functional_residual_adapter,
    train_functional_residual_adapter_disjoint,
)


def test_qwen_one_site_rank8_accounting_is_exact_and_not_the_prior_26m_result() -> None:
    report = FunctionalResidualAdapterAccounting.qwen3_4b_one_site(rank=8)

    assert report.removed_site_parameters == 3_276_800
    assert report.redirect_role_scale_parameters == 36 * 8 * 4 == 1_152
    assert report.adapter_factor_parameters == 8 * (2560 + 512) == 24_576
    assert report.adapter_gate_parameters == 0
    assert report.net_parameters_saved_vs_incumbent == 3_251_072
    assert report.learned_payload_bytes_saved == 6_502_144
    assert report.index_adjusted_payload_bytes_saved == 6_502_140
    assert report.tensor_payload_bytes_saved == 6_502_132
    assert report.flops_per_site_token == 49_152
    payload = report.as_dict()
    assert payload["prior_p36_to_p35_observed_saving_included"] is False
    assert payload["full_p35_to_p34_step_hardcoded"] is False
    assert payload["artifact_plan_required_before_checkpoint_claim"] is True


def test_optional_gate_is_counted_and_can_never_be_free() -> None:
    without = FunctionalResidualAdapterAccounting.qwen3_4b_one_site(rank=8)
    with_gate = FunctionalResidualAdapterAccounting.qwen3_4b_one_site(
        rank=8, added_scalar_gate=True
    )

    assert with_gate.adapter_gate_parameters == 1
    assert with_gate.adapter_parameters == without.adapter_parameters + 1
    assert (
        with_gate.net_parameters_saved_vs_incumbent
        == without.net_parameters_saved_vs_incumbent - 1
    )


def test_multisite_accounting_counts_each_removed_index_and_optional_qkv_bias() -> None:
    report = FunctionalResidualAdapterAccounting.build(
        hidden_size=10,
        group_context_width=4,
        head_dim=2,
        logical_layers=3,
        physical_groups=2,
        rank=1,
        site_count=2,
        q_prototype_bias=True,
        k_prototype_bias=True,
        v_prototype_bias=True,
        role_scale_table_new_for_redirect=False,
    )

    assert report.removed_site_weight_parameters == 240
    assert report.removed_site_bias_parameters == 16
    assert report.removed_site_parameters == 256
    assert report.adapter_parameters == 28
    assert report.net_parameters_saved_vs_incumbent == 228
    assert report.removed_prototype_index_bytes == 8
    assert report.adapter_site_index_bytes == 16
    assert report.tensor_payload_bytes_saved == 440

def test_sparse_registry_uses_the_physical_group_slice_not_canonical_order() -> None:
    left = torch.tensor([[[1.0], [2.0]]])
    right = torch.tensor([[[3.0, -1.0]]])
    registry = SparseFunctionalResidualAdapterRegistry(
        [[1, 2]],
        logical_layers=2,
        physical_groups=3,
        hidden_size=2,
        group_context_width=2,
        rank=1,
        left=left,
        right=right,
    )
    contexts = torch.tensor([[100.0, 100.0, 50.0, 50.0, 4.0, 1.0]])
    base = torch.tensor([[0.5, -0.5]])
    # The permutation is deliberately non-identity.  O inputs have already
    # been materialized in physical order, so physical group 2 is the last slice.
    result = registry.apply_to_output(
        contexts,
        base,
        layer_index=1,
        group_permutation=torch.tensor([2, 0, 1]),
    )

    assert torch.equal(result, torch.tensor([[11.5, 21.5]]))
    assert torch.equal(
        registry.apply_to_output(
            contexts,
            base,
            layer_index=0,
            group_permutation=torch.tensor([0, 1, 2]),
        ),
        base,
    )


@pytest.mark.parametrize(
    "indices",
    [
        [[1, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[2, 0]],
        [[0, 3]],
        [[0.5, 0]],
    ],
)
def test_sparse_registry_rejects_unsorted_duplicate_or_invalid_sites(indices) -> None:
    with pytest.raises(FunctionalResidualAdapterTopologyError):
        SparseFunctionalResidualAdapterRegistry(
            indices,
            logical_layers=2,
            physical_groups=3,
            hidden_size=2,
            group_context_width=2,
            rank=1,
        )


def test_sparse_registry_zero_init_roundtrips_as_first_class_state() -> None:
    first = SparseFunctionalResidualAdapterRegistry(
        [[0, 1], [2, 0]],
        logical_layers=3,
        physical_groups=2,
        hidden_size=4,
        group_context_width=3,
        rank=2,
        initialization_seed=17,
    )
    second = SparseFunctionalResidualAdapterRegistry(
        [[0, 1], [2, 0]],
        logical_layers=3,
        physical_groups=2,
        hidden_size=4,
        group_context_width=3,
        rank=2,
        initialization_seed=999,
    )
    second.load_state_dict(first.state_dict(), strict=True)

    assert first.starts_as_exact_zero_correction
    assert first.learned_parameter_count == 2 * 2 * (4 + 3)
    assert first.index_buffer_bytes == 16
    assert first.schema_buffer_bytes == 8
    assert set(first.state_dict()) == {
        "left",
        "right",
        "site_indices",
        "_registry_schema_version",
        "_registry_layout",
    }
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )


def test_sparse_registry_rejects_bad_permutation_and_nonfinite_factors() -> None:
    registry = SparseFunctionalResidualAdapterRegistry(
        [[0, 0]],
        logical_layers=1,
        physical_groups=2,
        hidden_size=2,
        group_context_width=1,
        rank=1,
    )
    with pytest.raises(FunctionalResidualAdapterTopologyError):
        registry.apply_to_output(
            torch.ones(1, 2),
            torch.zeros(1, 2),
            layer_index=0,
            group_permutation=torch.tensor([0, 0]),
        )
    with torch.no_grad():
        registry.left[0, 0, 0] = float("nan")
    with pytest.raises(FunctionalResidualAdapterError):
        registry.apply_to_output(
            torch.ones(1, 2),
            torch.zeros(1, 2),
            layer_index=0,
            group_permutation=torch.tensor([0, 1]),
        )


class _ToySelfAttention(nn.Module):
    def __init__(self, o_proj: nn.Module) -> None:
        super().__init__()
        self.o_proj = o_proj


class _ToyLayer(nn.Module):
    def __init__(self, o_proj: nn.Module) -> None:
        super().__init__()
        self.self_attn = _ToySelfAttention(o_proj)


class _ToyBackbone(nn.Module):
    def __init__(self, attention: GlobalAttentionCodebook) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _ToyLayer(attention[layer].o_proj)
            for layer in range(attention.logical_layers)
        )


class _ToyFunctionalLM(nn.Module):
    def __init__(self, attention: GlobalAttentionCodebook) -> None:
        super().__init__()
        self.attention_codebook = attention
        self.model = _ToyBackbone(attention)
        self.embedding = nn.Embedding(4, attention.registry.query_width)
        self.lm_head = nn.Linear(attention.registry.output_width, 3, bias=False)
        with torch.no_grad():
            self.embedding.weight.zero_()
            # Physical group 4 occupies flattened O-input columns [16:20].
            self.embedding.weight[0, 16:20] = 1.0
            self.embedding.weight[1, 16:20] = 2.0
            self.embedding.weight[2, 16:20] = 3.0
            self.lm_head.weight.zero_()
            self.lm_head.weight[0, 0] = 1.0
            self.lm_head.weight[1, 1] = 1.0
            self.lm_head.weight[2, 2] = 1.0

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> SimpleNamespace:
        del attention_mask, kwargs
        contexts = self.embedding(input_ids)
        hidden = self.model.layers[19].self_attn.o_proj(contexts)
        return SimpleNamespace(logits=self.lm_head(hidden))


def _tiny_redirected_runtime() -> tuple[_ToyFunctionalLM, RaggedAttentionCodebookRegistry]:
    layers = 22
    groups = 8
    group_width = 4
    head_dim = 1
    hidden = 128
    prototypes = 2
    q = torch.zeros(prototypes, group_width, hidden)
    k = torch.zeros(prototypes, head_dim, hidden)
    v = torch.zeros(prototypes, head_dim, hidden)
    o = torch.zeros(prototypes, hidden, group_width)
    assignments = torch.zeros(layers, groups, dtype=torch.int32)
    permutations = torch.arange(groups, dtype=torch.int32).repeat(layers, 1)
    # Exercise physical->canonical mapping at both sites.
    permutations[19, 4], permutations[19, 5] = (
        permutations[19, 5].clone(),
        permutations[19, 4].clone(),
    )
    permutations[21, 6], permutations[21, 7] = (
        permutations[21, 7].clone(),
        permutations[21, 6].clone(),
    )
    assignments[19, 5] = 1  # canonical 5 -> physical 4
    assignments[21, 7] = 1  # canonical 7 -> physical 6
    registry = RaggedAttentionCodebookRegistry(
        q,
        k,
        v,
        o,
        assignments,
        torch.tensor([0, P35_L19G4_TO_L21G6.survivor_source_flat_index]),
        permutations,
        role_scales=torch.ones(layers, groups, 4),
    )
    return _ToyFunctionalLM(GlobalAttentionCodebook(registry)), registry


def test_redirect_audit_follows_physical_permutation_and_rejects_false_redirect() -> None:
    _model, registry = _tiny_redirected_runtime()
    report = audit_redirected_attention_site(registry, P35_L19G4_TO_L21G6)

    assert report.removed_canonical_group == 5
    assert report.survivor_canonical_group == 7
    assert report.shared_prototype_slot == 1
    assert report.shared_prototype_source == 174
    assert report.removed_source_absent

    with torch.no_grad():
        registry.assignments[19, 5] = 0
    with pytest.raises(FunctionalResidualAdapterTopologyError):
        audit_redirected_attention_site(registry, P35_L19G4_TO_L21G6)


def test_redirect_audit_rejects_source_ids_that_do_not_encode_physical_sites() -> None:
    _model, registry = _tiny_redirected_runtime()
    false_provenance = RedirectedAttentionSite(
        layer_index=19,
        physical_group=4,
        removed_source_flat_index=0,
        survivor_layer_index=21,
        survivor_physical_group=6,
        survivor_source_flat_index=174,
    )

    with pytest.raises(
        FunctionalResidualAdapterTopologyError,
        match="do not encode their declared physical sites",
    ):
        audit_redirected_attention_site(registry, false_provenance)


def test_install_is_zero_exact_sparse_counted_and_rolls_back_even_after_adapter_tamper() -> None:
    model, registry = _tiny_redirected_runtime()
    original = model.model.layers[19].self_attn.o_proj
    contexts = torch.randn(2, 3, registry.query_width)
    baseline = original(contexts).detach().clone()
    before_count = sum(parameter.numel() for parameter in model.parameters())

    handle = install_functional_residual_adapter(
        model,
        registry,
        site=P35_L19G4_TO_L21G6,
        rank=1,
        initialization_seed=3,
    )
    installed = model.model.layers[19].self_attn.o_proj

    assert torch.equal(installed(contexts), baseline)
    assert handle.report.adapter_parameter_count == 128 + 4
    assert handle.report.model_parameter_count_before == before_count
    assert (
        handle.report.model_parameter_count_installed
        == before_count + handle.report.adapter_parameter_count
    )
    assert "functional_residual_adapter_registry.left" in model.state_dict()
    assert handle.report.accounting.removed_site_parameters == 1_280
    assert handle.report.accounting.redirect_role_scale_parameters == 704
    assert handle.report.accounting.net_parameters_saved_vs_incumbent == 444

    with torch.no_grad():
        handle.adapter_registry.left.fill_(1.0)
    assert not torch.equal(installed(contexts), baseline)
    with torch.no_grad():
        handle.adapter_registry.left[0, 0, 0] = float("nan")
    with pytest.raises(FunctionalResidualAdapterError):
        handle.assert_live()

    handle.rollback()
    assert model.model.layers[19].self_attn.o_proj is original
    assert not hasattr(model, "functional_residual_adapter_registry")
    assert sum(parameter.numel() for parameter in model.parameters()) == before_count
    assert torch.equal(original(contexts), baseline)


def test_live_transaction_hashes_and_rejects_any_base_registry_mutation() -> None:
    model, registry = _tiny_redirected_runtime()
    handle = install_functional_residual_adapter(
        model,
        registry,
        site=P35_L19G4_TO_L21G6,
        rank=1,
    )
    original = registry.prototype_o_weight.detach().clone()
    with torch.no_grad():
        registry.prototype_o_weight[0, 0, 0] = 1.0
    with pytest.raises(FunctionalResidualAdapterTopologyError):
        handle.assert_live()

    # The adapter transaction never owns or silently repairs base weights.
    # Restore the independently tampered base, then rejection is exact.
    with torch.no_grad():
        registry.prototype_o_weight.copy_(original)
    handle.rollback()


def test_rollback_fails_before_overwriting_a_replaced_projection() -> None:
    model, registry = _tiny_redirected_runtime()
    handle = install_functional_residual_adapter(
        model,
        registry,
        site=P35_L19G4_TO_L21G6,
        rank=1,
    )
    installed = model.model.layers[19].self_attn.o_proj
    replacement = nn.Identity()
    model.model.layers[19].self_attn.o_proj = replacement

    with pytest.raises(
        FunctionalResidualAdapterTopologyError,
        match="output view was replaced",
    ):
        handle.rollback()
    assert model.model.layers[19].self_attn.o_proj is replacement
    assert getattr(model, handle.report.owner_name) is handle.adapter_registry

    model.model.layers[19].self_attn.o_proj = installed
    handle.rollback()


def test_live_handle_rejects_duplicate_adapter_registration_before_rollback() -> None:
    model, registry = _tiny_redirected_runtime()
    handle = install_functional_residual_adapter(
        model,
        registry,
        site=P35_L19G4_TO_L21G6,
        rank=1,
    )
    model.add_module("duplicate_adapter_owner", handle.adapter_registry)

    with pytest.raises(
        FunctionalResidualAdapterTopologyError,
        match="exactly one registered owner",
    ):
        handle.assert_live()
    with pytest.raises(
        FunctionalResidualAdapterTopologyError,
        match="exactly one registered owner",
    ):
        handle.rollback()

    delattr(model, "duplicate_adapter_owner")
    handle.rollback()


def _action_batch(
    sequence_id: str,
    token_id: int,
    *,
    split: str,
    prefix_source: str,
) -> FunctionalActionBatch:
    return FunctionalActionBatch(
        sequence_id=sequence_id,
        split=split,
        prefix_source=prefix_source,
        input_ids=torch.tensor([[token_id]]),
        attention_mask=torch.tensor([[1]]),
        parent_action_ids=torch.tensor([1]),
        parent_action_logits=torch.tensor([[0.0, 4.0, 0.0]]),
    )


def test_functional_training_uses_both_rollout_prefixes_and_improves_actions_only() -> None:
    model, registry = _tiny_redirected_runtime()
    handle = install_functional_residual_adapter(
        model,
        registry,
        site=P35_L19G4_TO_L21G6,
        rank=1,
        initialization_seed=4,
    )
    calibration = (
        _action_batch(
            "parent-prefix",
            0,
            split="calibration-selection",
            prefix_source="parent-greedy",
        ),
        _action_batch(
            "student-prefix",
            1,
            split="calibration-selection",
            prefix_source="student-greedy",
        ),
    )
    base_state_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith(f"{handle.report.owner_name}.")
    }
    base_parameters = tuple(
        parameter
        for parameter in model.parameters()
        if all(parameter is not adapter for adapter in handle.adapter_registry.parameters())
    )
    assert all(parameter.grad is None for parameter in base_parameters)
    before = evaluate_functional_action_batches(
        model,
        calibration,
        expected_split="calibration-selection",
        distribution_weight=0.0,
        used_for_selection=True,
    )
    report = train_functional_residual_adapter(
        handle,
        calibration,
        steps=24,
        learning_rate=0.2,
        distribution_weight=0.0,
    )
    after = evaluate_functional_action_batches(
        model,
        calibration,
        expected_split="calibration-selection",
        distribution_weight=0.0,
        used_for_selection=True,
    )

    assert report.optimizer_constructed is False
    assert report.heldout_evaluation_calls == 0
    assert report.parent_greedy_examples == 1
    assert report.student_greedy_examples == 1
    assert report.best_calibration_loss < report.initial_calibration_loss
    assert report.final_calibration_loss == pytest.approx(
        report.best_calibration_loss
    )
    assert after.total_loss < before.total_loss
    assert after.parent_action_top1_rate == 1.0
    assert all(parameter.grad is None for parameter in base_parameters)
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in base_state_before.items()
    )
    training_payload = report.as_dict()
    assert training_payload["base_parameters_updated"] == 0
    assert training_payload["learned_compaction_not_fold_only"] is True
    assert set(training_payload["excluded_objectives"]) == {
        "weight-similarity",
        "activation-similarity",
        "attention-similarity",
        "residual-similarity",
        "cache-similarity",
    }

    heldout = (
        _action_batch(
            "heldout",
            2,
            split="heldout-evaluation",
            prefix_source="student-greedy",
        ),
    )
    evaluation = evaluate_functional_action_batches(
        model,
        heldout,
        expected_split="heldout-evaluation",
        distribution_weight=0.0,
    )
    assert evaluation.used_for_selection is False
    assert evaluation.parent_action_top1_rate == 1.0
    assert evaluation.as_dict()["attention_residual_cache_weight_similarity_used"] is False
    handle.rollback()


def test_disjoint_training_uses_fit_gradients_and_calibration_selection_only() -> None:
    model, registry = _tiny_redirected_runtime()
    handle = install_functional_residual_adapter(
        model,
        registry,
        site=P35_L19G4_TO_L21G6,
        rank=1,
        initialization_seed=7,
    )
    fit = (
        _action_batch("fit-parent", 0, split="fit", prefix_source="parent-greedy"),
        _action_batch("fit-student", 1, split="fit", prefix_source="student-greedy"),
    )
    calibration = (
        _action_batch(
            "cal-parent",
            2,
            split="calibration-selection",
            prefix_source="parent-greedy",
        ),
        _action_batch(
            "cal-student",
            3,
            split="calibration-selection",
            prefix_source="student-greedy",
        ),
    )
    base_versions = {
        name: int(value._version)
        for name, value in model.named_parameters()
        if not name.startswith(f"{handle.report.owner_name}.")
    }

    report = train_functional_residual_adapter_disjoint(
        handle,
        fit,
        calibration,
        steps=3,
        learning_rate=0.1,
        distribution_weight=0.0,
    )

    payload = report.as_dict()
    assert report.fit_gradient_calls == 3
    assert report.calibration_evaluations == 5
    assert report.heldout_evaluation_calls == 0
    assert report.fit_parent_greedy_examples == 1
    assert report.fit_student_greedy_examples == 1
    assert report.calibration_parent_greedy_examples == 1
    assert report.calibration_student_greedy_examples == 1
    assert payload["gradient_split"] == "fit"
    assert payload["selection_split"] == "calibration-selection"
    assert payload["fit_and_calibration_prefixes_disjoint"] is True
    assert payload["adapter_parameters_only"] is True
    assert payload["heldout_used_for_fitting"] is False
    assert payload["heldout_used_for_selection"] is False
    assert base_versions == {
        name: int(value._version)
        for name, value in model.named_parameters()
        if not name.startswith(f"{handle.report.owner_name}.")
    }
    handle.rollback()


def test_disjoint_training_does_not_promote_continuous_only_improvement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, registry = _tiny_redirected_runtime()
    handle = install_functional_residual_adapter(
        model,
        registry,
        site=P35_L19G4_TO_L21G6,
        rank=1,
        initialization_seed=9,
    )
    fit = (
        _action_batch("fit-parent", 0, split="fit", prefix_source="parent-greedy"),
        _action_batch("fit-student", 1, split="fit", prefix_source="student-greedy"),
    )
    calibration = (
        _action_batch(
            "cal-parent",
            2,
            split="calibration-selection",
            prefix_source="parent-greedy",
        ),
        _action_batch(
            "cal-student",
            3,
            split="calibration-selection",
            prefix_source="student-greedy",
        ),
    )
    initial = tuple(
        value.detach().clone() for value in handle.adapter_registry.parameters()
    )

    def discrete_tie_with_better_continuous_metrics(
        _model,
        batches,
        *,
        expected_split,
        used_for_selection,
        **_kwargs,
    ) -> FunctionalActionEvaluation:
        changed = any(
            not torch.equal(value.detach(), before)
            for value, before in zip(
                handle.adapter_registry.parameters(), initial, strict=True
            )
        )
        return FunctionalActionEvaluation(
            split=expected_split,
            example_count=sum(value.batch_size for value in batches),
            total_loss=1.0 if changed else 2.0,
            hard_action_loss=1.0 if changed else 2.0,
            distribution_loss=0.0,
            parent_action_top1_rate=0.5,
            mean_parent_action_log_probability=-0.5 if changed else -1.0,
            mean_parent_action_rank=2.0,
            used_for_selection=used_for_selection,
            prefix_sources=("parent-greedy", "student-greedy"),
        )

    monkeypatch.setattr(
        adapter_module,
        "evaluate_functional_action_batches",
        discrete_tie_with_better_continuous_metrics,
    )
    report = train_functional_residual_adapter_disjoint(
        handle,
        fit,
        calibration,
        steps=2,
        learning_rate=0.1,
        distribution_weight=0.0,
        calibration_steps=(0, 1, 2),
    )

    assert report.best_step == 0
    assert all(
        torch.equal(value.detach(), before)
        for value, before in zip(
            handle.adapter_registry.parameters(), initial, strict=True
        )
    )
    payload = report.as_dict()
    assert payload["continuous_log_probability_or_loss_can_promote"] is False
    assert payload["selection_key_order"] == [
        "maximize-parent-action-top1-rate",
        "minimize-mean-parent-action-rank",
    ]
    handle.rollback()


def test_disjoint_training_rejects_split_relabeling_overlap_and_heldout_selection() -> None:
    model, registry = _tiny_redirected_runtime()
    handle = install_functional_residual_adapter(
        model,
        registry,
        site=P35_L19G4_TO_L21G6,
        rank=1,
    )
    fit = (
        _action_batch("fit-parent", 0, split="fit", prefix_source="parent-greedy"),
        _action_batch("fit-student", 1, split="fit", prefix_source="student-greedy"),
    )
    calibration_with_overlap = (
        _action_batch(
            "cal-parent",
            0,
            split="calibration-selection",
            prefix_source="parent-greedy",
        ),
        _action_batch(
            "cal-student",
            2,
            split="calibration-selection",
            prefix_source="student-greedy",
        ),
    )
    with pytest.raises(FunctionalResidualAdapterTrainingError, match="overlapping"):
        train_functional_residual_adapter_disjoint(
            handle,
            fit,
            calibration_with_overlap,
            steps=1,
        )
    with pytest.raises(FunctionalResidualAdapterTrainingError, match="Gradient batches"):
        train_functional_residual_adapter_disjoint(
            handle,
            calibration_with_overlap,
            calibration_with_overlap,
            steps=1,
        )
    heldout = (
        _action_batch(
            "heldout",
            2,
            split="heldout-evaluation",
            prefix_source="student-greedy",
        ),
    )
    with pytest.raises(FunctionalResidualAdapterTrainingError, match="Only calibration"):
        evaluate_functional_action_batches(
            model,
            heldout,
            expected_split="heldout-evaluation",
            used_for_selection=True,
        )
    handle.rollback()


def test_training_fails_closed_on_heldout_or_missing_student_rollout() -> None:
    model, registry = _tiny_redirected_runtime()
    handle = install_functional_residual_adapter(
        model,
        registry,
        site=P35_L19G4_TO_L21G6,
        rank=1,
    )
    parent_only = (
        _action_batch(
            "only-parent",
            0,
            split="calibration-selection",
            prefix_source="parent-greedy",
        ),
    )
    with pytest.raises(FunctionalResidualAdapterTrainingError):
        train_functional_residual_adapter(handle, parent_only, steps=1)

    heldout = (
        _action_batch(
            "heldout",
            0,
            split="heldout-evaluation",
            prefix_source="parent-greedy",
        ),
    )
    with pytest.raises(FunctionalResidualAdapterTrainingError):
        train_functional_residual_adapter(handle, heldout, steps=1)
    with pytest.raises(FunctionalResidualAdapterTrainingError):
        evaluate_functional_action_batches(
            model,
            heldout,
            expected_split="heldout-evaluation",
            used_for_selection=True,
        )
    handle.rollback()


def test_fixed_parent_action_anchor_cannot_disagree_with_parent_logits() -> None:
    with pytest.raises(FunctionalResidualAdapterTrainingError):
        FunctionalActionBatch(
            sequence_id="bad-anchor",
            split="calibration-selection",
            prefix_source="parent-greedy",
            input_ids=torch.tensor([[0]]),
            attention_mask=torch.tensor([[1]]),
            parent_action_ids=torch.tensor([0]),
            parent_action_logits=torch.tensor([[0.0, 4.0, 0.0]]),
        )


def test_action_batch_accepts_real_multitoken_right_padded_prefixes() -> None:
    batch = FunctionalActionBatch(
        sequence_id="multitoken",
        split="calibration-selection",
        prefix_source="parent-greedy",
        input_ids=torch.tensor([[1, 2, 0], [2, 0, 0]]),
        attention_mask=torch.tensor([[1, 1, 0], [1, 0, 0]]),
        parent_action_ids=torch.tensor([1, 0]),
        parent_action_logits=torch.tensor([[0.0, 4.0], [4.0, 0.0]]),
    )

    assert batch.batch_size == 2
