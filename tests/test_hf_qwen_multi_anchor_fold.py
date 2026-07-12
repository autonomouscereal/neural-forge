from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neural_forge.hf_qwen_compensated_neuron_fold import (  # noqa: E402
    Qwen3CompensatedFoldShapeError,
    Qwen3CompensatedFoldStateError,
    fit_compensated_neuron_pairs,
    install_qwen3_compensated_fold,
)
from neural_forge.hf_qwen_multi_anchor_fold import (  # noqa: E402
    fit_multi_anchor_neuron_fold,
)


def _weights(width: int, hidden: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(9)
    return (
        torch.randn(width, hidden, generator=generator),
        torch.randn(width, hidden, generator=generator),
        torch.randn(hidden, width, generator=generator),
    )


def _exact_traces() -> tuple[torch.Tensor, torch.Tensor]:
    calibration_base = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [2.0, 1.0, 0.5], [1.0, -1.0, 3.0]]
    )
    validation_base = torch.tensor(
        [[-1.0, 2.0, 1.0], [3.0, 0.5, -2.0], [0.25, -1.5, 4.0]]
    )
    calibration = torch.column_stack(
        (calibration_base, 0.25 * calibration_base[:, 0] - 0.75 * calibration_base[:, 2])
    )
    validation = torch.column_stack(
        (validation_base, 0.25 * validation_base[:, 0] - 0.75 * validation_base[:, 2])
    )
    return calibration, validation


def test_multi_anchor_fold_exactly_absorbs_nonlocal_path() -> None:
    gate, up, down = _weights(4, 3)
    calibration, validation = _exact_traces()
    down[:, 1].mul_(100.0)
    down[:, 3].mul_(0.01)
    fold = fit_multi_anchor_neuron_fold(
        gate,
        up,
        down,
        calibration,
        validation_activations=validation,
        target_width=3,
        calibration_provenance="cal-prompts-v1",
        validation_provenance="validation-prompts-v1",
        candidate_pool_size=1,
        max_anchors=2,
        similarity_chunk_size=1,
    )

    assert fold.merges[0].dropped == 3
    assert set(fold.merges[0].anchors) == {0, 2}
    assert fold.diagnostics.aggregate_calibration_relative_l2 < 1e-6
    assert fold.diagnostics.aggregate_validation_relative_l2 < 1e-6
    assert fold.diagnostics.materialized_full_pairwise_matrix is False
    assert fold.parameter_savings == 9


def test_multi_anchor_projection_beats_single_pair_on_composed_path() -> None:
    gate, up, down = _weights(4, 3)
    calibration, validation = _exact_traces()
    down[:, 1].mul_(100.0)
    down[:, 3].mul_(0.01)
    pair = fit_compensated_neuron_pairs(
        gate,
        up,
        down,
        calibration,
        target_width=3,
        validation_activations=validation,
        projection_count=8,
        neighbor_window=3,
        max_pair_candidates=12,
        max_generated_pairs=12,
        score_chunk_size=4,
    )
    multi = fit_multi_anchor_neuron_fold(
        gate,
        up,
        down,
        calibration,
        validation_activations=validation,
        target_width=3,
        calibration_provenance="cal",
        validation_provenance="val",
        candidate_pool_size=1,
        max_anchors=2,
        similarity_chunk_size=1,
    )

    assert multi.diagnostics.aggregate_validation_relative_l2 < (
        pair.diagnostics.aggregate_validation_relative_l2 * 1e-3
    )


def test_multi_anchor_fold_keeps_protected_paths_bitwise_unchanged() -> None:
    gate, up, down = _weights(6, 4)
    generator = torch.Generator().manual_seed(17)
    calibration = torch.randn(16, 6, generator=generator)
    validation = torch.randn(13, 6, generator=generator)
    protected_gate = gate[1].clone()
    protected_up = up[1].clone()
    protected_down = down[:, 1].clone()
    fold = fit_multi_anchor_neuron_fold(
        gate,
        up,
        down,
        calibration,
        validation_activations=validation,
        target_width=5,
        calibration_provenance="cal-a",
        validation_provenance="val-b",
        protected_indices=(1,),
        candidate_pool_size=2,
        max_anchors=2,
    )

    target_index = fold.kept_source_indices.index(1)
    assert torch.equal(fold.gate_weight[target_index], protected_gate)
    assert torch.equal(fold.up_weight[target_index], protected_up)
    assert torch.equal(fold.down_weight[:, target_index], protected_down)
    assert all(1 not in merge.anchors for merge in fold.merges)
    assert all(merge.dropped != 1 for merge in fold.merges)


def test_multi_anchor_fold_enforces_provenance_and_work_caps() -> None:
    gate, up, down = _weights(8, 3)
    calibration = torch.randn(10, 8)
    validation = torch.randn(9, 8)
    with pytest.raises(Qwen3CompensatedFoldShapeError, match="provenance"):
        fit_multi_anchor_neuron_fold(
            gate,
            up,
            down,
            calibration,
            validation_activations=validation,
            target_width=7,
            calibration_provenance="same",
            validation_provenance="same",
        )
    with pytest.raises(Qwen3CompensatedFoldShapeError, match="similarity work"):
        fit_multi_anchor_neuron_fold(
            gate,
            up,
            down,
            calibration,
            validation_activations=validation,
            target_width=7,
            calibration_provenance="cal",
            validation_provenance="val",
            candidate_pool_size=3,
            max_similarity_products=2,
        )
    with pytest.raises(Qwen3CompensatedFoldShapeError, match="projection work"):
        fit_multi_anchor_neuron_fold(
            gate,
            up,
            down,
            calibration,
            validation_activations=validation,
            target_width=7,
            calibration_provenance="cal",
            validation_provenance="val",
            candidate_pool_size=3,
            max_projection_fits=2,
        )


def test_validation_trace_rejects_calibration_only_projection() -> None:
    gate, up, down = _weights(5, 3)
    generator = torch.Generator().manual_seed(71)
    cal_base = torch.randn(18, 3, generator=generator)
    val_base = torch.randn(15, 3, generator=generator)
    calibration = torch.column_stack(
        (cal_base, cal_base[:, 0], 0.5 * cal_base[:, 1])
    )
    validation = torch.column_stack(
        (val_base, torch.randn(15, generator=generator), 0.5 * val_base[:, 1])
    )
    down[:, :3].mul_(100.0)
    down[:, 3:].mul_(0.01)
    fold = fit_multi_anchor_neuron_fold(
        gate,
        up,
        down,
        calibration,
        validation_activations=validation,
        target_width=4,
        calibration_provenance="cal-only",
        validation_provenance="untouched-validation",
        candidate_pool_size=2,
        max_anchors=2,
    )

    assert fold.merges[0].dropped == 4
    assert fold.merges[0].validation_path_relative_l2 < 1e-5


def test_multi_anchor_fold_integrity_rejects_mutation() -> None:
    gate, up, down = _weights(5, 3)
    calibration = torch.randn(12, 5)
    validation = torch.randn(11, 5)
    fold = fit_multi_anchor_neuron_fold(
        gate,
        up,
        down,
        calibration,
        validation_activations=validation,
        target_width=4,
        calibration_provenance="cal",
        validation_provenance="val",
        candidate_pool_size=2,
    )
    fold.down_weight[0, 0].add_(1.0)
    with pytest.raises(Qwen3CompensatedFoldStateError, match="changed"):
        fold.assert_integrity()


class _MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(3, 4, bias=False)
        self.up_proj = torch.nn.Linear(3, 4, bias=False)
        self.down_proj = torch.nn.Linear(4, 3, bias=False)
        self.act_fn = torch.nn.SiLU()


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _MLP()


class _Inner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer()])


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()


def test_existing_physical_installer_accepts_multi_anchor_fold() -> None:
    model = _Model()
    mlp = model.model.layers[0].mlp
    calibration, validation = _exact_traces()
    with torch.inference_mode():
        mlp.down_proj.weight[:, 1].mul_(100.0)
        mlp.down_proj.weight[:, 3].mul_(0.01)
    fold = fit_multi_anchor_neuron_fold(
        mlp.gate_proj.weight,
        mlp.up_proj.weight,
        mlp.down_proj.weight,
        calibration,
        validation_activations=validation,
        target_width=3,
        calibration_provenance="cal",
        validation_provenance="val",
        candidate_pool_size=1,
        max_anchors=2,
        similarity_chunk_size=1,
    )
    handle = install_qwen3_compensated_fold(model, 0, fold)
    assert model.model.layers[0].mlp.gate_proj.out_features == 3
    handle.rollback()
    assert model.model.layers[0].mlp is mlp
