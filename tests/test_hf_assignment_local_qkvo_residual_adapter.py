from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import neural_forge.hf_assignment_local_qkvo_residual_adapter as qkvo_module
from neural_forge.hf_assignment_local_qkvo_residual_adapter import (
    AssignmentLocalQKVOResidualAccounting,
    AssignmentLocalQKVOResidualError,
    AssignmentLocalQKVOResidualProjection,
    AssignmentLocalQKVOResidualTopologyError,
    SparseAssignmentLocalQKVOResidualRegistry,
    install_assignment_local_qkvo_residual_adapter,
    train_assignment_local_qkvo_residual_disjoint,
)
from neural_forge.hf_attention_codebook import (
    GlobalAttentionCodebook,
    RaggedAttentionCodebookRegistry,
)
from neural_forge.hf_functional_residual_adapter import (
    P35_L19G4_TO_L21G6,
    FunctionalActionBatch,
    FunctionalActionEvaluation,
)


def test_exact_qwen_rank8_accounting_is_92160_and_net_3183488() -> None:
    report = AssignmentLocalQKVOResidualAccounting.qwen3_4b_one_site(rank=8)

    assert report.removed_site_parameters == 3_276_800
    assert report.redirect_role_scale_parameters == 1_152
    assert report.q_adapter_parameters == 24_576
    assert report.k_adapter_parameters == 21_504
    assert report.v_adapter_parameters == 21_504
    assert report.o_adapter_parameters == 24_576
    assert report.adapter_parameters == 92_160
    assert report.net_parameters_saved_vs_incumbent == 3_183_488
    assert report.learned_payload_bytes_saved == 6_366_976
    assert report.tensor_payload_bytes_saved == 6_366_964
    assert report.flops_per_site_token == 184_320
    assert report.as_dict()["checkpoint_claimed"] is False


def test_sparse_registry_supports_multiple_sites_and_roundtrips_state() -> None:
    first = SparseAssignmentLocalQKVOResidualRegistry(
        [[0, 1], [2, 0]],
        logical_layers=3,
        physical_groups=2,
        hidden_size=8,
        query_group_width=4,
        head_dim=1,
        rank=2,
        dtype=torch.float32,
        device="cpu",
        initialization_seed=17,
    )
    second = SparseAssignmentLocalQKVOResidualRegistry(
        [[0, 1], [2, 0]],
        logical_layers=3,
        physical_groups=2,
        hidden_size=8,
        query_group_width=4,
        head_dim=1,
        rank=2,
        dtype=torch.float32,
        device="cpu",
        initialization_seed=999,
    )
    second.load_state_dict(first.state_dict(), strict=True)

    assert first.site_count == 2
    assert first.starts_as_exact_zero_correction
    assert first.input_factors_are_nonzero
    assert first.learned_parameter_count == 2 * 2 * (4 * 8 + 2 * 4 + 2 * 1)
    assert first.index_buffer_bytes == 16
    assert first.schema_buffer_bytes == 8
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )


@pytest.mark.parametrize(
    "indices",
    (
        [[1, 0], [0, 0]],
        [[0, 0], [0, 0]],
        [[3, 0]],
        [[0, 2]],
        [[0.5, 0]],
    ),
)
def test_sparse_registry_rejects_invalid_site_tables(indices) -> None:
    with pytest.raises(AssignmentLocalQKVOResidualTopologyError):
        SparseAssignmentLocalQKVOResidualRegistry(
            indices,
            logical_layers=3,
            physical_groups=2,
            hidden_size=8,
            query_group_width=4,
            head_dim=1,
            rank=1,
            dtype=torch.float32,
            device="cpu",
        )


def _active_direct_registry() -> SparseAssignmentLocalQKVOResidualRegistry:
    registry = SparseAssignmentLocalQKVOResidualRegistry(
        [[1, 2]],
        logical_layers=2,
        physical_groups=3,
        hidden_size=2,
        query_group_width=2,
        head_dim=1,
        rank=1,
        dtype=torch.float32,
        device="cpu",
        initialization_seed=3,
    )
    with torch.no_grad():
        for role in ("q", "k", "v", "o"):
            getattr(registry, f"{role}_left").fill_(1.0)
            getattr(registry, f"{role}_right").fill_(1.0)
    return registry


def test_qkv_and_o_corrections_touch_only_the_declared_physical_group() -> None:
    registry = _active_direct_registry()
    hidden = torch.tensor([[2.0, 3.0]])
    q_base = torch.zeros(1, 6)
    k_base = torch.zeros(1, 3)
    v_base = torch.zeros(1, 3)
    contexts = torch.tensor([[50.0, 50.0, 70.0, 70.0, 4.0, 1.0]])
    o_base = torch.zeros(1, 2)

    q = registry.apply_to_projection("q", hidden, q_base, layer_index=1)
    k = registry.apply_to_projection("k", hidden, k_base, layer_index=1)
    v = registry.apply_to_projection("v", hidden, v_base, layer_index=1)
    o = registry.apply_to_output(contexts, o_base, layer_index=1)

    assert torch.equal(q, torch.tensor([[0.0, 0.0, 0.0, 0.0, 5.0, 5.0]]))
    assert torch.equal(k, torch.tensor([[0.0, 0.0, 5.0]]))
    assert torch.equal(v, torch.tensor([[0.0, 0.0, 5.0]]))
    assert torch.equal(o, torch.tensor([[5.0, 5.0]]))
    assert torch.equal(
        registry.apply_to_projection("q", hidden, q_base, layer_index=0),
        q_base,
    )


def test_adapter_is_strictly_pointwise_and_preserves_causal_prefixes() -> None:
    registry = _active_direct_registry()
    hidden = torch.randn(1, 4, 2)
    q_base = torch.randn(1, 4, 6)
    contexts = torch.randn(1, 4, 6)
    o_base = torch.randn(1, 4, 2)

    q_full = registry.apply_to_projection("q", hidden, q_base, layer_index=1)
    q_prefix = registry.apply_to_projection(
        "q", hidden[:, :2], q_base[:, :2], layer_index=1
    )
    o_full = registry.apply_to_output(contexts, o_base, layer_index=1)
    o_prefix = registry.apply_to_output(
        contexts[:, :2], o_base[:, :2], layer_index=1
    )

    assert torch.equal(q_full[:, :2], q_prefix)
    assert torch.equal(o_full[:, :2], o_prefix)
    assert registry.as_dict()["cross_token_mixing_inside_adapter"] is False


class _RecordingNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input: Tensor | None = None

    def forward(self, value: Tensor) -> Tensor:
        self.last_input = value.detach().clone()
        return value


class _RecordingCache:
    def __init__(self) -> None:
        self.key: Tensor | None = None
        self.value: Tensor | None = None

    def update(self, key: Tensor, value: Tensor, layer_index: int):
        del layer_index
        self.key = key.detach().clone()
        self.value = value.detach().clone()
        return key, value


class _ToySelfAttention(nn.Module):
    def __init__(self, view, layer_index: int) -> None:
        super().__init__()
        self.q_proj = view.q_proj
        self.k_proj = view.k_proj
        self.v_proj = view.v_proj
        self.o_proj = view.o_proj
        self.q_norm = _RecordingNorm()
        self.k_norm = _RecordingNorm()
        self.layer_idx = layer_index

    def forward(self, hidden: Tensor, cache: _RecordingCache | None = None) -> Tensor:
        q = self.q_norm(self.q_proj(hidden))
        k = self.k_norm(self.k_proj(hidden))
        v = self.v_proj(hidden)
        if cache is not None:
            k, v = cache.update(k, v, self.layer_idx)
        context = q + k.repeat_interleave(4, dim=-1) + v.repeat_interleave(4, dim=-1)
        return self.o_proj(context)


class _ToyLayer(nn.Module):
    def __init__(self, view, layer_index: int) -> None:
        super().__init__()
        self.self_attn = _ToySelfAttention(view, layer_index)


class _ToyBackbone(nn.Module):
    def __init__(self, attention: GlobalAttentionCodebook) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _ToyLayer(attention[layer], layer)
            for layer in range(attention.logical_layers)
        )


class _ToyQKVOLM(nn.Module):
    def __init__(self, attention: GlobalAttentionCodebook) -> None:
        super().__init__()
        self.attention_codebook = attention
        self.model = _ToyBackbone(attention)
        self.embedding = nn.Embedding(8, attention.registry.input_width)
        self.lm_head = nn.Linear(attention.registry.output_width, 5, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> SimpleNamespace:
        del attention_mask, kwargs
        hidden = self.embedding(input_ids)
        output = self.model.layers[19].self_attn(hidden)
        return SimpleNamespace(logits=self.lm_head(output))


def _redirected_runtime() -> tuple[_ToyQKVOLM, RaggedAttentionCodebookRegistry]:
    layers = 22
    groups = 8
    query_group_width = 4
    head_dim = 1
    hidden = 8
    prototypes = 2
    torch.manual_seed(44)
    q = torch.randn(prototypes, query_group_width, hidden)
    k = torch.randn(prototypes, head_dim, hidden)
    v = torch.randn(prototypes, head_dim, hidden)
    o = torch.randn(prototypes, hidden, query_group_width)
    assignments = torch.zeros(layers, groups, dtype=torch.int32)
    permutations = torch.arange(groups, dtype=torch.int32).repeat(layers, 1)
    permutations[19, 4], permutations[19, 5] = (
        permutations[19, 5].clone(),
        permutations[19, 4].clone(),
    )
    permutations[21, 6], permutations[21, 7] = (
        permutations[21, 7].clone(),
        permutations[21, 6].clone(),
    )
    assignments[19, 5] = 1
    assignments[21, 7] = 1
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
    return _ToyQKVOLM(GlobalAttentionCodebook(registry)), registry


def test_install_is_zero_exact_and_qkv_execute_before_norm_and_cache() -> None:
    model, registry = _redirected_runtime()
    attention = model.model.layers[19].self_attn
    originals = {
        role: getattr(attention, f"{role}_proj")
        for role in ("q", "k", "v", "o")
    }
    hidden = torch.randn(2, 3, registry.input_width)
    baseline_cache = _RecordingCache()
    baseline = attention(hidden, baseline_cache).detach().clone()

    handle = install_assignment_local_qkvo_residual_adapter(
        model,
        registry,
        sites=(P35_L19G4_TO_L21G6,),
        rank=1,
        initialization_seed=5,
        role_scale_table_new_for_redirect=False,
    )
    installed_attention = model.model.layers[19].self_attn
    installed_cache = _RecordingCache()
    installed = installed_attention(hidden, installed_cache)

    assert torch.equal(installed, baseline)
    assert torch.equal(installed_cache.key, baseline_cache.key)
    assert torch.equal(installed_cache.value, baseline_cache.value)
    assert handle.report.adapter_parameter_count == 42
    assert handle.report.accounting.net_parameters_saved_vs_incumbent == 38
    assert all(
        isinstance(
            getattr(installed_attention, f"{role}_proj"),
            AssignmentLocalQKVOResidualProjection,
        )
        for role in ("q", "k", "v", "o")
    )

    with torch.no_grad():
        handle.adapter_registry.q_left.fill_(0.25)
        handle.adapter_registry.k_left.fill_(0.5)
        handle.adapter_registry.v_left.fill_(0.75)
    q_delta = installed_attention.q_proj(hidden) - originals["q"](hidden)
    k_delta = installed_attention.k_proj(hidden) - originals["k"](hidden)
    v_delta = installed_attention.v_proj(hidden) - originals["v"](hidden)
    assert torch.count_nonzero(q_delta[..., :16]) == 0
    assert torch.count_nonzero(q_delta[..., 20:]) == 0
    assert torch.count_nonzero(q_delta[..., 16:20]) > 0
    assert torch.count_nonzero(k_delta[..., :4]) == 0
    assert torch.count_nonzero(k_delta[..., 5:]) == 0
    assert torch.count_nonzero(k_delta[..., 4:5]) > 0
    assert torch.count_nonzero(v_delta[..., :4]) == 0
    assert torch.count_nonzero(v_delta[..., 5:]) == 0
    assert torch.count_nonzero(v_delta[..., 4:5]) > 0
    changed_cache = _RecordingCache()
    installed_attention(hidden, changed_cache)
    assert torch.equal(
        installed_attention.q_norm.last_input,
        installed_attention.q_proj(hidden),
    )
    assert torch.equal(
        installed_attention.k_norm.last_input,
        installed_attention.k_proj(hidden),
    )
    assert torch.equal(changed_cache.key, installed_attention.k_proj(hidden))
    assert torch.equal(changed_cache.value, installed_attention.v_proj(hidden))

    payload = handle.report.as_dict()
    assert payload["q_correction_boundary"].endswith("before-q-norm-and-rope")
    assert payload["k_correction_boundary"].endswith(
        "before-k-norm-rope-and-cache-append"
    )
    assert payload["v_correction_boundary"].endswith("before-cache-append")
    handle.rollback()
    assert all(
        getattr(attention, f"{role}_proj") is originals[role]
        for role in ("q", "k", "v", "o")
    )


def test_output_training_scope_freezes_base_and_exposes_only_adapter_factors() -> None:
    model, registry = _redirected_runtime()
    handle = install_assignment_local_qkvo_residual_adapter(
        model,
        registry,
        sites=(P35_L19G4_TO_L21G6,),
        rank=1,
        role_scale_table_new_for_redirect=False,
    )
    adapter_ids = {id(value) for value in handle.adapter_registry.parameters()}
    base = tuple(value for value in model.parameters() if id(value) not in adapter_ids)
    base_versions = tuple(int(value._version) for value in base)

    with handle.output_training_scope() as trainable:
        assert {id(value) for value in trainable} == adapter_ids
        assert all(value.requires_grad for value in trainable)
        assert all(not value.requires_grad for value in base)
        logits = model(torch.tensor([[1, 2]])).logits
        gradients = torch.autograd.grad(logits.sum(), trainable)
        assert len(gradients) == 8
        assert all(value is not None for value in gradients)
        with torch.no_grad():
            for parameter, gradient in zip(trainable, gradients, strict=True):
                parameter.add_(gradient, alpha=-1e-3)

    assert tuple(int(value._version) for value in base) == base_versions
    assert all(value.requires_grad for value in base)
    scope_payload = handle.output_training_scope().as_dict()
    assert scope_payload["accepted_target_space"] == (
        "causal-model-output-actions-or-logits"
    )
    assert scope_payload["internal_similarity_target_api_exposed"] is False
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
        parent_action_logits=torch.tensor([[0.0, 4.0, 0.0, 0.0, 0.0]]),
    )


def test_disjoint_trainer_uses_per_branch_rates_and_never_reads_heldout() -> None:
    model, registry = _redirected_runtime()
    handle = install_assignment_local_qkvo_residual_adapter(
        model,
        registry,
        sites=(P35_L19G4_TO_L21G6,),
        rank=1,
        role_scale_table_new_for_redirect=False,
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
    report = train_assignment_local_qkvo_residual_disjoint(
        handle,
        fit,
        calibration,
        steps=2,
        q_learning_rate=1e-3,
        k_learning_rate=1e-3,
        v_learning_rate=2e-3,
        o_learning_rate=2e-3,
        calibration_steps=(0, 1, 2),
    )

    payload = report.as_dict()
    assert report.q_learning_rate == report.k_learning_rate == 1e-3
    assert report.v_learning_rate == report.o_learning_rate == 2e-3
    assert report.fit_gradient_calls == 2
    assert report.calibration_selection_steps == (0, 1, 2)
    assert report.heldout_evaluation_calls == 0
    assert report.branch_parameter_counts == {"q": 12, "k": 9, "v": 9, "o": 12}
    assert [value["step"] for value in report.calibration_history] == [0, 1, 2]
    assert len(report.fit_step_history) == 2
    assert set(report.fit_loss_summary) == {
        "total_loss",
        "hard_action_loss",
        "distribution_loss",
    }
    assert set(report.branch_gradient_summary) == {"q", "k", "v", "o"}
    assert set(report.branch_parameter_delta_summary) == {"q", "k", "v", "o"}
    assert set(report.parameter_gradient_activation) == {
        "q_left",
        "q_right",
        "k_left",
        "k_right",
        "v_left",
        "v_right",
        "o_left",
        "o_right",
    }
    assert isinstance(report.all_branches_received_nonzero_gradient, bool)
    assert isinstance(report.all_parameter_tensors_received_nonzero_gradient, bool)
    assert payload["gradient_split"] == "fit"
    assert payload["selection_split"] == "calibration-selection"
    assert payload["continuous_log_probability_or_loss_can_promote"] is False
    assert payload["base_parameters_updated"] == 0
    assert payload["diagnostics_can_select_or_promote"] is False
    handle.rollback()


def test_disjoint_trainer_does_not_promote_continuous_only_movement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, registry = _redirected_runtime()
    handle = install_assignment_local_qkvo_residual_adapter(
        model,
        registry,
        sites=(P35_L19G4_TO_L21G6,),
        rank=1,
        role_scale_table_new_for_redirect=False,
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

    def continuous_only(
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
        qkvo_module,
        "evaluate_functional_action_batches",
        continuous_only,
    )
    report = train_assignment_local_qkvo_residual_disjoint(
        handle,
        fit,
        calibration,
        steps=1,
        calibration_steps=(0, 1),
    )

    assert report.best_step == 0
    assert all(
        torch.equal(value.detach(), before)
        for value, before in zip(
            handle.adapter_registry.parameters(), initial, strict=True
        )
    )
    handle.rollback()


def test_base_mutation_is_rejected_and_finite_adapter_tamper_rolls_back() -> None:
    model, registry = _redirected_runtime()
    handle = install_assignment_local_qkvo_residual_adapter(
        model,
        registry,
        sites=(P35_L19G4_TO_L21G6,),
        rank=1,
        role_scale_table_new_for_redirect=False,
    )
    original = registry.prototype_q_weight.detach().clone()
    with torch.no_grad():
        registry.prototype_q_weight[0, 0, 0].add_(1.0)
    with pytest.raises(AssignmentLocalQKVOResidualTopologyError):
        handle.assert_live()
    with torch.no_grad():
        registry.prototype_q_weight.copy_(original)
        handle.adapter_registry.o_left.fill_(2.0)
    handle.rollback()


def test_nonfinite_adapter_fails_forward() -> None:
    registry = _active_direct_registry()
    with torch.no_grad():
        registry.q_left[0, 0, 0] = float("nan")
    with pytest.raises(AssignmentLocalQKVOResidualError):
        registry.apply_to_projection(
            "q",
            torch.ones(1, 2),
            torch.zeros(1, 6),
            layer_index=1,
        )


def test_nonfinite_installed_adapter_is_still_exactly_removable() -> None:
    model, registry = _redirected_runtime()
    attention = model.model.layers[19].self_attn
    originals = {
        role: getattr(attention, f"{role}_proj")
        for role in ("q", "k", "v", "o")
    }
    handle = install_assignment_local_qkvo_residual_adapter(
        model,
        registry,
        sites=(P35_L19G4_TO_L21G6,),
        rank=1,
        role_scale_table_new_for_redirect=False,
    )
    with torch.no_grad():
        handle.adapter_registry.k_left[0, 0, 0] = float("nan")
    with pytest.raises(AssignmentLocalQKVOResidualError):
        handle.assert_live()

    handle.rollback()
    assert all(
        getattr(attention, f"{role}_proj") is originals[role]
        for role in ("q", "k", "v", "o")
    )


def test_rollback_fails_closed_if_one_projection_was_replaced() -> None:
    model, registry = _redirected_runtime()
    handle = install_assignment_local_qkvo_residual_adapter(
        model,
        registry,
        sites=(P35_L19G4_TO_L21G6,),
        rank=1,
        role_scale_table_new_for_redirect=False,
    )
    attention = model.model.layers[19].self_attn
    installed = attention.k_proj
    replacement = nn.Identity()
    attention.k_proj = replacement

    with pytest.raises(
        AssignmentLocalQKVOResidualTopologyError,
        match="projection view was replaced",
    ):
        handle.rollback()
    assert attention.k_proj is replacement
    assert getattr(model, handle.report.owner_name) is handle.adapter_registry

    attention.k_proj = installed
    handle.rollback()
