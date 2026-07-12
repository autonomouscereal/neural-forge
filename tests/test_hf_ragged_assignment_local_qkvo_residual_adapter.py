from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from neural_forge.hf_assignment_local_qkvo_residual_adapter import (
    train_assignment_local_qkvo_residual_disjoint,
)
from neural_forge.hf_attention_codebook import (
    GlobalAttentionCodebook,
    RaggedAttentionCodebookRegistry,
)
from neural_forge.hf_functional_residual_adapter import (
    FunctionalActionBatch,
    RedirectedAttentionSite,
)
from neural_forge.hf_ragged_assignment_local_qkvo_residual_adapter import (
    RaggedAssignmentLocalQKVOAccounting,
    RaggedAssignmentLocalQKVOResidualError,
    RaggedAssignmentLocalQKVOResidualRegistry,
    install_ragged_assignment_local_qkvo_residual_adapter,
)


SITE_A = RedirectedAttentionSite(19, 4, 156, 21, 6, 174)
SITE_B = RedirectedAttentionSite(15, 2, 122, 19, 5, 157)


def test_exact_p35_to_p34_uneven_rank_accounting_is_real_not_padded() -> None:
    accounting = RaggedAssignmentLocalQKVOAccounting.build(
        site_ranks=(8, 4, 4, 4, 4, 4, 4, 4),
        hidden_size=2560,
        query_group_width=512,
        head_dim=128,
        logical_layers=36,
        physical_groups=8,
        checkpoint_element_size=2,
        role_scale_table_new_for_redirects=True,
    )

    assert accounting.rank_sum == 36
    assert accounting.removed_parameters == 26_214_400
    assert accounting.role_scale_parameters == 1_152
    assert accounting.adapter_parameters == 414_720
    assert accounting.adapter_tensor_bytes == 829_440
    assert accounting.net_parameters_saved == 25_798_528
    assert accounting.net_tensor_bytes_saved == 51_597_056
    assert accounting.as_dict()["padded_to_max_rank"] is False


def test_ragged_registry_allocates_exact_per_site_shapes_and_local_corrections() -> (
    None
):
    registry = RaggedAssignmentLocalQKVOResidualRegistry(
        [[1, 0], [2, 1]],
        [2, 1],
        logical_layers=3,
        physical_groups=2,
        hidden_size=8,
        query_group_width=4,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
        initialization_seed=7,
    )

    assert registry.ranks == (2, 1)
    assert tuple(registry.q_left[0].shape) == (4, 2)
    assert tuple(registry.q_left[1].shape) == (4, 1)
    assert registry.learned_parameter_count == 126
    assert registry.starts_as_exact_zero_correction
    assert registry.input_factors_are_nonzero

    with torch.no_grad():
        for role in registry._ROLES:
            for value in getattr(registry, f"{role}_left"):
                value.fill_(1.0)
            for value in getattr(registry, f"{role}_right"):
                value.fill_(1.0)
    hidden = torch.ones(1, 8)
    q = registry.apply_to_projection("q", hidden, torch.zeros(1, 8), layer_index=2)
    assert torch.equal(q[:, :4], torch.zeros(1, 4))
    assert torch.equal(q[:, 4:], torch.full((1, 4), 8.0))


class _ToyAttention(nn.Module):
    def __init__(self, view, layer_index: int) -> None:
        super().__init__()
        self.q_proj = view.q_proj
        self.k_proj = view.k_proj
        self.v_proj = view.v_proj
        self.o_proj = view.o_proj
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.layer_idx = layer_index

    def forward(self, hidden: Tensor) -> Tensor:
        q = self.q_norm(self.q_proj(hidden))
        k = self.k_norm(self.k_proj(hidden)).repeat_interleave(4, dim=-1)
        v = self.v_proj(hidden).repeat_interleave(4, dim=-1)
        return self.o_proj(q + k + v)


class _ToyLayer(nn.Module):
    def __init__(self, view, layer_index: int) -> None:
        super().__init__()
        self.self_attn = _ToyAttention(view, layer_index)


class _ToyModel(nn.Module):
    def __init__(self, attention: GlobalAttentionCodebook) -> None:
        super().__init__()
        self.attention_codebook = attention
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            _ToyLayer(attention[index], index)
            for index in range(attention.logical_layers)
        )
        self.embedding = nn.Embedding(8, attention.registry.input_width)
        self.lm_head = nn.Linear(attention.registry.output_width, 5, bias=False)

    def forward(self, input_ids: Tensor, **_kwargs) -> SimpleNamespace:
        hidden = self.embedding(input_ids)
        hidden = self.model.layers[15].self_attn(hidden)
        hidden = self.model.layers[19].self_attn(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def _runtime() -> tuple[_ToyModel, RaggedAttentionCodebookRegistry]:
    layers = 22
    groups = 8
    hidden = 8
    query_group_width = 4
    head_dim = 1
    torch.manual_seed(91)
    assignments = torch.zeros(layers, groups, dtype=torch.int32)
    permutations = torch.arange(groups, dtype=torch.int32).repeat(layers, 1)
    # Physical L19/G4 and L19/G5 map to canonical rows 5 and 4.
    permutations[19, 4] = 5
    permutations[19, 5] = 4
    permutations[21, 6] = 7
    permutations[21, 7] = 6
    assignments[19, 5] = 1
    assignments[21, 7] = 1
    assignments[15, 2] = 2
    assignments[19, 4] = 2
    registry = RaggedAttentionCodebookRegistry(
        torch.randn(3, query_group_width, hidden),
        torch.randn(3, head_dim, hidden),
        torch.randn(3, head_dim, hidden),
        torch.randn(3, hidden, query_group_width),
        assignments,
        torch.tensor([0, 174, 157], dtype=torch.int64),
        permutations,
        role_scales=torch.ones(layers, groups, 4),
    )
    attention = GlobalAttentionCodebook(registry)
    return _ToyModel(attention), registry


def _batch(name: str, token: int, split: str, prefix: str) -> FunctionalActionBatch:
    return FunctionalActionBatch(
        sequence_id=name,
        split=split,
        prefix_source=prefix,
        input_ids=torch.tensor([[token]]),
        attention_mask=torch.tensor([[1]]),
        parent_action_ids=torch.tensor([1]),
        parent_action_logits=torch.tensor([[0.0, 4.0, 0.0, 0.0, 0.0]]),
    )


def test_ragged_install_trains_adapter_only_and_rolls_back_exactly() -> None:
    model, attention_registry = _runtime()
    originals = {
        (layer, role): getattr(model.model.layers[layer].self_attn, f"{role}_proj")
        for layer in (15, 19)
        for role in ("q", "k", "v", "o")
    }
    handle = install_ragged_assignment_local_qkvo_residual_adapter(
        model,
        attention_registry,
        sites=(SITE_B, SITE_A),
        site_ranks=(1, 2),
        initialization_seed=3,
        role_scale_table_new_for_redirects=False,
    )

    assert handle.adapter_registry.ranks == (1, 2)
    assert handle.report.adapter_parameter_count == 126
    assert handle.report.accounting.net_parameters_saved == 34
    fit = (
        _batch("fit-parent", 0, "fit", "parent-greedy"),
        _batch("fit-student", 1, "fit", "student-greedy"),
    )
    calibration = (
        _batch("cal-parent", 2, "calibration-selection", "parent-greedy"),
        _batch("cal-student", 3, "calibration-selection", "student-greedy"),
    )
    report = train_assignment_local_qkvo_residual_disjoint(
        handle,
        fit,
        calibration,
        steps=1,
        calibration_steps=(0, 1),
    )
    assert report.branch_parameter_counts == {"q": 36, "k": 27, "v": 27, "o": 36}
    assert set(report.parameter_gradient_activation) == {
        f"{role}_{side}.{site}"
        for role in ("q", "k", "v", "o")
        for side in ("left", "right")
        for site in (0, 1)
    }
    handle.rollback()
    assert all(
        getattr(model.model.layers[layer].self_attn, f"{role}_proj") is projection
        for (layer, role), projection in originals.items()
    )


def test_rank_map_tamper_blocks_rollback_until_restored() -> None:
    model, attention_registry = _runtime()
    handle = install_ragged_assignment_local_qkvo_residual_adapter(
        model,
        attention_registry,
        sites=(SITE_B, SITE_A),
        site_ranks=(1, 2),
        role_scale_table_new_for_redirects=False,
    )
    with torch.no_grad():
        handle.adapter_registry.site_ranks[0] = 2
    with pytest.raises(RaggedAssignmentLocalQKVOResidualError, match="ranks changed"):
        handle.rollback()
    with torch.no_grad():
        handle.adapter_registry.site_ranks.copy_(torch.tensor([1, 2]))
    handle.rollback()
