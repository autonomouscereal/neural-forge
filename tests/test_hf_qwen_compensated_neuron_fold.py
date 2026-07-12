from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)

from neural_forge.hf_qwen_compensated_neuron_fold import (  # noqa: E402
    Qwen3CompensatedFoldShapeError,
    Qwen3CompensatedFoldStateError,
    fit_compensated_neuron_pairs,
    install_qwen3_compensated_fold,
    synchronize_qwen3_intermediate_size,
)


def _weights(seed: int = 73):
    generator = torch.Generator().manual_seed(seed)
    gate = torch.randn(6, 4, generator=generator)
    up = torch.randn(6, 4, generator=generator)
    down = torch.randn(4, 6, generator=generator)
    return gate, up, down


def test_closed_form_consumer_exactly_merges_proportional_nonadjacent_paths() -> None:
    gate, up, down = _weights()
    activations = torch.randn(24, 6, generator=torch.Generator().manual_seed(91))
    activations[:, 5] = -2.5 * activations[:, 0]
    validation = 1.25 * activations.roll(1, dims=0)
    fold = fit_compensated_neuron_pairs(
        gate,
        up,
        down,
        activations,
        target_width=5,
        validation_activations=validation,
        projection_count=12,
        neighbor_window=5,
        max_pair_candidates=100,
        score_chunk_size=2,
        seed=11,
    )

    assert len(fold.merges) == 1
    merge = fold.merges[0]
    assert {merge.left, merge.right} == {0, 5}
    assert merge.calibration_relative_error < 1e-6
    folded_activations = activations[:, list(fold.kept_source_indices)]
    expected = activations @ down.T
    actual = folded_activations @ fold.down_weight.T
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert fold.parameter_savings == 12
    assert fold.diagnostics.materialized_full_pairwise_matrix is False
    assert fold.diagnostics.score_chunk_size == 2
    assert fold.diagnostics.retained_scored_pair_count <= (
        fold.diagnostics.scored_pair_count
    )
    assert fold.diagnostics.aggregate_calibration_relative_l2 < 1e-6
    assert fold.diagnostics.aggregate_validation_relative_l2 < 1e-6
    assert fold.diagnostics.optimizer_steps == 0
    assert fold.diagnostics.backward_called is False


def test_high_damage_anchor_is_never_merged_or_modified() -> None:
    gate, up, down = _weights(101)
    activations = torch.randn(20, 6, generator=torch.Generator().manual_seed(102))
    activations[:, 5] = activations[:, 0]
    fold = fit_compensated_neuron_pairs(
        gate,
        up,
        down,
        activations,
        target_width=5,
        validation_activations=activations.roll(1, dims=0),
        protected_indices=(0,),
        projection_count=16,
        neighbor_window=5,
        max_pair_candidates=100,
    )

    assert 0 in fold.kept_source_indices
    assert all(0 not in (item.left, item.right) for item in fold.merges)
    kept_position = fold.kept_source_indices.index(0)
    torch.testing.assert_close(fold.gate_weight[kept_position], gate[0])
    torch.testing.assert_close(fold.up_weight[kept_position], up[0])
    torch.testing.assert_close(fold.down_weight[:, kept_position], down[:, 0])


def test_closed_form_consumer_is_never_worse_than_dropping_the_other_path() -> None:
    gate, up, down = _weights(121)
    activations = torch.randn(32, 6, generator=torch.Generator().manual_seed(122))
    activations[:, 4] = 0.8 * activations[:, 1] + 0.02 * activations[:, 2]
    fold = fit_compensated_neuron_pairs(
        gate,
        up,
        down,
        activations,
        target_width=5,
        validation_activations=activations.roll(1, dims=0),
        projection_count=16,
        neighbor_window=5,
        max_pair_candidates=100,
    )
    merge = fold.merges[0]
    source_pair = (
        activations[:, merge.left, None] * down[:, merge.left]
        + activations[:, merge.right, None] * down[:, merge.right]
    )
    representative = merge.representative
    kept_position = fold.kept_source_indices.index(representative)
    compensated = activations[:, representative, None] * fold.down_weight[:, kept_position]
    naive_drop = activations[:, representative, None] * down[:, representative]

    assert torch.linalg.vector_norm(source_pair - compensated) <= (
        torch.linalg.vector_norm(source_pair - naive_drop) + 1e-6
    )


def test_disjoint_pair_pass_rejects_an_unreachable_width() -> None:
    gate, up, down = _weights(131)
    activations = torch.randn(12, 6, generator=torch.Generator().manual_seed(132))
    with pytest.raises(Qwen3CompensatedFoldShapeError, match="disjoint-pair"):
        fit_compensated_neuron_pairs(
            gate,
            up,
            down,
            activations,
            target_width=2,
            validation_activations=activations.roll(1, dims=0),
            protected_indices=(0,),
        )


def test_validation_split_and_search_work_caps_fail_closed() -> None:
    gate, up, down = _weights(141)
    activations = torch.randn(12, 6, generator=torch.Generator().manual_seed(142))
    with pytest.raises(Qwen3CompensatedFoldShapeError, match="provenance-disjoint"):
        fit_compensated_neuron_pairs(
            gate,
            up,
            down,
            activations,
            target_width=5,
            validation_activations=activations.clone(),
        )
    with pytest.raises(Qwen3CompensatedFoldShapeError, match="hard cap 64"):
        fit_compensated_neuron_pairs(
            gate,
            up,
            down,
            activations,
            target_width=5,
            validation_activations=activations.roll(1, dims=0),
            projection_count=65,
        )
    with pytest.raises(Qwen3CompensatedFoldShapeError, match="cannot exceed"):
        fit_compensated_neuron_pairs(
            gate,
            up,
            down,
            activations,
            target_width=5,
            validation_activations=activations.roll(1, dims=0),
            max_pair_candidates=101,
            max_generated_pairs=100,
        )


def test_realized_error_uses_destination_dtype_and_fold_tensors_are_sealed() -> None:
    gate, up, down = (value.half() for value in _weights(151))
    activations = torch.randn(18, 6, generator=torch.Generator().manual_seed(152))
    activations[:, 5] = 0.75 * activations[:, 0]
    fold = fit_compensated_neuron_pairs(
        gate,
        up,
        down,
        activations,
        target_width=5,
        validation_activations=activations.roll(1, dims=0),
        projection_count=12,
        neighbor_window=5,
        max_pair_candidates=100,
    )
    assert fold.down_weight.dtype == torch.float16
    assert fold.diagnostics.realized_dtype == "torch.float16"
    assert fold.diagnostics.aggregate_calibration_relative_l2 >= 0.0
    fold.down_weight[0, 0] += 1.0
    with pytest.raises(Qwen3CompensatedFoldStateError, match="changed"):
        fold.assert_integrity()


def _tiny_qwen(seed: int = 181):
    torch.manual_seed(seed)
    return transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=17,
            hidden_size=8,
            intermediate_size=6,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            attention_dropout=0.0,
            use_sliding_window=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    ).eval()


def _layer_activations(model, layer_index: int, hidden: torch.Tensor) -> torch.Tensor:
    mlp = model.model.layers[layer_index].mlp
    with torch.no_grad():
        return (mlp.act_fn(mlp.gate_proj(hidden)) * mlp.up_proj(hidden)).reshape(
            -1, mlp.gate_proj.weight.shape[0]
        )


def test_qwen_layer_install_rollback_and_commit_are_explicit() -> None:
    model = _tiny_qwen()
    hidden = torch.randn(3, 4, 8, generator=torch.Generator().manual_seed(190))
    original = model.model.layers[0].mlp
    activations = _layer_activations(model, 0, hidden)
    fold = fit_compensated_neuron_pairs(
        original.gate_proj.weight,
        original.up_proj.weight,
        original.down_proj.weight,
        activations,
        target_width=5,
        validation_activations=activations.roll(1, dims=0),
        projection_count=12,
        neighbor_window=5,
        max_pair_candidates=100,
    )

    handle = install_qwen3_compensated_fold(model, 0, fold)
    assert model.model.layers[0].mlp is handle.installed_mlp
    assert model.model.layers[0].mlp.gate_proj.out_features == 5
    assert synchronize_qwen3_intermediate_size(model) is None
    with torch.no_grad():
        output = model.model.layers[0].mlp(hidden)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    handle.rollback()
    assert model.model.layers[0].mlp is original
    assert handle.reverted
    with pytest.raises(Qwen3CompensatedFoldStateError, match="reverted"):
        handle.commit()

    second = install_qwen3_compensated_fold(model, 0, fold)
    second.commit()
    assert second.committed
    with pytest.raises(Qwen3CompensatedFoldStateError, match="committed"):
        second.rollback()


def test_exact_source_hash_rejects_any_post_fit_weight_change() -> None:
    model = _tiny_qwen(201)
    hidden = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(202))
    original = model.model.layers[0].mlp
    activations = _layer_activations(model, 0, hidden)
    fold = fit_compensated_neuron_pairs(
        original.gate_proj.weight,
        original.up_proj.weight,
        original.down_proj.weight,
        activations,
        target_width=5,
        validation_activations=activations.roll(1, dims=0),
        projection_count=12,
        neighbor_window=5,
        max_pair_candidates=100,
    )
    with torch.no_grad():
        original.gate_proj.weight[-1, -1].add_(0.125)
    with pytest.raises(Qwen3CompensatedFoldStateError, match="do not match"):
        install_qwen3_compensated_fold(model, 0, fold)


def test_uniform_committed_layers_update_export_config() -> None:
    model = _tiny_qwen(211)
    hidden = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(212))
    handles = []
    for layer_index in range(2):
        mlp = model.model.layers[layer_index].mlp
        activations = _layer_activations(model, layer_index, hidden)
        fold = fit_compensated_neuron_pairs(
            mlp.gate_proj.weight,
            mlp.up_proj.weight,
            mlp.down_proj.weight,
            activations,
            target_width=5,
            validation_activations=activations.roll(1, dims=0),
            projection_count=12,
            neighbor_window=5,
            max_pair_candidates=100,
            seed=layer_index,
        )
        handle = install_qwen3_compensated_fold(model, layer_index, fold)
        handle.commit()
        handles.append(handle)

    assert synchronize_qwen3_intermediate_size(model) == 5
    assert model.config.intermediate_size == 5
    assert all(item.original_mlp is None for item in handles)
    assert all(parameter.grad is None for parameter in model.parameters())
