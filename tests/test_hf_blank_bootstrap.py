from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_blank_bootstrap as bootstrap_module  # noqa: E402
from neural_forge.hf_arch_layout import NormLayout  # noqa: E402
from neural_forge.hf_blank_bootstrap import (  # noqa: E402
    BlankBootstrapError,
    bootstrap_blank_decoder_models,
)
from neural_forge.hf_transport_manifest import (  # noqa: E402
    build_decoder_transport_manifest,
)
from neural_forge.hf_token_align import (  # noqa: E402
    build_strict_token_transport_profile,
)


def _gpt2_source():
    torch.manual_seed(101)
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_embd=12,
            n_layer=3,
            n_head=3,
            n_inner=20,
            activation_function="gelu_new",
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    )


def _gpt2_target():
    torch.manual_seed(151)
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_embd=8,
            n_layer=2,
            n_head=2,
            n_inner=12,
            activation_function="gelu_new",
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    )


def _llama_source():
    torch.manual_seed(202)
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=64,
            hidden_size=12,
            intermediate_size=20,
            num_hidden_layers=3,
            num_attention_heads=3,
            num_key_value_heads=1,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            mlp_bias=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
    )


def _qwen3_source():
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("Transformers build has no Qwen3 support")
    torch.manual_seed(252)
    return transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=64,
            hidden_size=12,
            intermediate_size=20,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            attention_dropout=0.0,
            use_sliding_window=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
    )


def _llama_target(*, hidden_size: int = 8, vocab_size: int = 64):
    torch.manual_seed(303)
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            mlp_bias=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
    )


def _qwen3_target(*, head_dim: int = 2):
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("Transformers build has no Qwen3 support")
    torch.manual_seed(353)
    return transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=64,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=head_dim,
            max_position_embeddings=32,
            hidden_act="silu",
            attention_bias=False,
            attention_dropout=0.0,
            use_sliding_window=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
    )


def _batch(seed: int = 404, *, batch_size: int = 8, sequence_length: int = 8):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        3,
        64,
        (batch_size, sequence_length),
        generator=generator,
    )
    input_ids[:, 0] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }


def _state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _right_pad(batch, padded_length: int):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    padding = padded_length - int(input_ids.shape[1])
    assert padding > 0
    return {
        "input_ids": torch.cat(
            [input_ids, torch.zeros((input_ids.shape[0], padding), dtype=torch.long)],
            dim=1,
        ),
        "attention_mask": torch.cat(
            [
                attention_mask,
                torch.zeros((input_ids.shape[0], padding), dtype=torch.long),
            ],
            dim=1,
        ),
    }


class _FakeTokenizer:
    def __init__(self, surfaces):
        self.surfaces = list(surfaces)
        self.all_special_ids = []

    def __len__(self):
        return len(self.surfaces)

    def decode(self, token_ids, **_kwargs):
        return "".join(self.surfaces[int(token_id)] for token_id in token_ids)

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [self.surfaces.index(text)]

    def get_vocab(self):
        return {surface: index for index, surface in enumerate(self.surfaces)}


def test_output_only_gate_ignores_non_authoritative_hidden_coordinate_scale() -> None:
    values = {
        "final_gain": -0.36,
        "logit_gain": 0.37,
        "top1_gain": 0.16,
        "min_final_gain": 0.0,
        "min_logit_gain": 0.0,
        "min_top1_gain": 0.0,
    }

    assert bootstrap_module._heldout_gate_accepts("all", **values) is False
    assert bootstrap_module._heldout_gate_accepts("output_only", **values) is True
    with pytest.raises(BlankBootstrapError, match="policy is invalid"):
        bootstrap_module._heldout_gate_accepts("invalid", **values)


@pytest.mark.parametrize(
    "source_factory",
    [_gpt2_source, _llama_source, _qwen3_source],
)
def test_blank_bootstrap_improves_source_anchors_and_covers_every_target_role(
    source_factory,
) -> None:
    source = source_factory()
    target = _llama_target()
    calibration = _batch()
    validation = _batch(405)

    result = bootstrap_blank_decoder_models(
        source,
        target,
        calibration,
        validation_inputs=validation,
        max_coordinate_rows=24,
        vocab_projection_chunk_size=7,
    )

    assert result.transaction.accepted is True, result.transaction.report["gate"]
    assert result.transaction.report["coverage"]["complete"] is True
    manifest = build_decoder_transport_manifest(target)
    assert result.transaction.report["coverage"]["evidenced_role_count"] == len(
        manifest.plan.required_roles
    )
    assert result.donor_layer_spans == ((0,), (1, 2))
    assert result.candidate_gate_metrics.final_state.relative_rmse <= (
        result.baseline_gate_metrics.final_state.relative_rmse
    )
    assert result.candidate_gate_metrics.logits.relative_rmse <= (
        result.baseline_gate_metrics.logits.relative_rmse
    )
    assert result.candidate_gate_metrics.top1_agreement >= (
        result.baseline_gate_metrics.top1_agreement
    )
    assert all(torch.isfinite(parameter).all() for parameter in target.parameters())
    assert all(parameter.grad is None for parameter in source.parameters())
    assert all(parameter.grad is None for parameter in target.parameters())
    if source_factory is _qwen3_source:
        fold_report = result.fit_reports["block.0.attention.feature_folds"]
        assert fold_report["context"]["full_source_coverage"] is True
        assert fold_report["source_query_head_norms"] == [
            "model.layers.0.self_attn.q_norm"
        ]
        assert fold_report["source_key_head_norms"] == [
            "model.layers.0.self_attn.k_norm"
        ]
        assert (
            result.fit_reports["block.0.attention.structural_prior"]["applied"] is True
        )
        assert result.fit_reports["block.0.mlp.structural_prior"]["applied"] is True
        assert (
            result.fit_reports["block.0.mlp.feature_fold"]["negative_coefficient_count"]
            == 0
        )
        assert result.fit_reports["block.0.mlp.product_target"]["applied"] is True
        assert result.fit_reports["block.0.mlp.product_fit"]["finite"] is True
        assert result.fit_reports["block.0.attention.query"]["method"] == (
            "trace-scaled-prior-preserving-parent-anchor-fit"
        )


def test_rejected_bootstrap_restores_the_exact_blank_target() -> None:
    source = _gpt2_source()
    target = _llama_target()
    before = _state(target)
    batch = _batch()

    result = bootstrap_blank_decoder_models(
        source,
        target,
        batch,
        validation_inputs=batch,
        min_final_state_relative_improvement=10.0,
        min_logit_relative_improvement=10.0,
        min_top1_agreement_gain=1.0,
    )

    assert result.transaction.accepted is False
    assert result.transaction.report["rollback"] == {
        "performed": True,
        "exact": True,
    }
    for name, value in target.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_prompt_pooled_bootstrap_accepts_different_token_sequence_lengths() -> None:
    source = _gpt2_source()
    target = _llama_target()
    target_calibration = _batch(501, batch_size=16, sequence_length=5)
    target_validation = _batch(502, batch_size=16, sequence_length=4)

    result = bootstrap_blank_decoder_models(
        source,
        target,
        _right_pad(target_calibration, 9),
        validation_inputs=_right_pad(target_validation, 7),
        target_calibration_inputs=target_calibration,
        target_validation_inputs=target_validation,
        trace_pooling="mean",
        fit_full_lm_head=False,
    )

    assert result.transaction.accepted is True, result.transaction.report["gate"]
    assert result.transaction.report["coverage"]["complete"] is True
    assert result.candidate_gate_metrics.final_state.relative_rmse <= (
        result.baseline_gate_metrics.final_state.relative_rmse
    )
    assert result.candidate_gate_metrics.logits.relative_rmse <= (
        result.baseline_gate_metrics.logits.relative_rmse
    )


def test_cross_tokenizer_bootstrap_reorders_vocab_and_projects_teacher_columns() -> (
    None
):
    source = _gpt2_source()
    target = _llama_target(vocab_size=16)
    donor_surfaces = [f"token-{index}" for index in range(64)]
    mapping = (0, 1, 2, 13, 8, 21, 5, 34, 3, 55, 9, 12, 7, 19, 4, 31)
    target_surfaces = [donor_surfaces[index] for index in mapping]
    profile = build_strict_token_transport_profile(
        _FakeTokenizer(donor_surfaces),
        _FakeTokenizer(target_surfaces),
        donor_row_capacity=64,
        target_row_capacity=16,
    )
    generator = torch.Generator().manual_seed(550)
    target_ids = torch.randint(3, 16, (16, 6), generator=generator)
    donor_index = torch.tensor(mapping, dtype=torch.long)
    source_ids = donor_index[target_ids]
    source_inputs = {
        "input_ids": source_ids,
        "attention_mask": torch.ones_like(source_ids),
    }
    target_inputs = {
        "input_ids": target_ids,
        "attention_mask": torch.ones_like(target_ids),
    }

    result = bootstrap_blank_decoder_models(
        source,
        target,
        source_inputs,
        validation_inputs=source_inputs,
        target_calibration_inputs=target_inputs,
        target_validation_inputs=target_inputs,
        token_transport_profile=profile,
        cross_tokenizer_mode=True,
        trace_pooling="mean",
        fit_full_lm_head=False,
        max_coordinate_rows=24,
        vocab_projection_chunk_size=5,
        vocab_row_norm_repair="source_l2",
    )

    assert result.transaction.accepted is True, result.transaction.report["gate"]
    assert result.fit_reports["embeddings.token"]["row_transport"] == (
        "strict-index-select"
    )
    contract = result.fit_reports["teacher_logits.target_space"]
    assert contract["space"] == "target-tokenizer-columns"
    assert contract["source_shape"] == [16, 64]
    assert contract["target_shape"] == [16, 16]
    assert contract["transport_profile_fingerprint"] == profile.fingerprint()
    selected_source = source.get_input_embeddings().weight.index_select(0, donor_index)
    assert torch.allclose(
        torch.linalg.vector_norm(target.get_input_embeddings().weight, dim=1),
        torch.linalg.vector_norm(selected_source, dim=1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_pca_operator_prior_uses_coordinate_transport_formula_and_bias() -> None:
    generator = torch.Generator().manual_seed(575)
    input_coordinates = bootstrap_module.build_parent_residual_basis(
        torch.randn(32, 5, generator=generator, dtype=torch.float64),
        3,
        center=False,
    )
    output_coordinates = bootstrap_module.build_parent_residual_basis(
        torch.randn(32, 7, generator=generator, dtype=torch.float64),
        4,
        center=False,
    )
    source = bootstrap_module.LinearOperator(
        path="source",
        weight=torch.randn(7, 5, generator=generator, dtype=torch.float64),
        bias=torch.randn(7, generator=generator, dtype=torch.float64),
    )
    target = bootstrap_module.LinearOperator(
        path="target",
        weight=torch.zeros(4, 3, dtype=torch.float64),
        bias=torch.zeros(4, dtype=torch.float64),
    )

    actual = bootstrap_module._transport_operator_prior(
        torch,
        source,
        target,
        output_coordinates,
        input_coordinates,
    )
    expected_weight = (
        output_coordinates.encode @ source.weight @ input_coordinates.decode
    )
    expected_bias = output_coordinates.encode @ source.bias

    assert torch.allclose(actual.weight, expected_weight, atol=1e-12, rtol=1e-12)
    assert torch.allclose(actual.bias, expected_bias, atol=1e-12, rtol=1e-12)


def test_operator_prior_supports_pca_residual_with_sparse_nonlinear_axis() -> None:
    generator = torch.Generator().manual_seed(576)
    input_coordinates = bootstrap_module.build_parent_residual_basis(
        torch.randn(32, 5, generator=generator, dtype=torch.float64),
        3,
        center=False,
    )
    output_coordinates = bootstrap_module.build_feature_fold_map(
        torch.randn(32, 7, generator=generator, dtype=torch.float64),
        4,
    )
    source = bootstrap_module.LinearOperator(
        path="source",
        weight=torch.randn(7, 5, generator=generator, dtype=torch.float64),
        bias=torch.randn(7, generator=generator, dtype=torch.float64),
    )
    target = bootstrap_module.LinearOperator(
        path="target",
        weight=torch.zeros(4, 3, dtype=torch.float64),
        bias=torch.zeros(4, dtype=torch.float64),
    )

    actual = bootstrap_module._transport_operator_prior(
        torch,
        source,
        target,
        output_coordinates,
        input_coordinates,
    )
    dense_output = output_coordinates.materialize_dense(max_elements=28)
    expected_weight = dense_output.T @ source.weight @ input_coordinates.decode
    expected_bias = dense_output.T @ source.bias

    assert torch.allclose(actual.weight, expected_weight, atol=1e-12, rtol=1e-12)
    assert torch.allclose(actual.bias, expected_bias, atol=1e-12, rtol=1e-12)


def test_pca_bootstrap_transports_all_priors_and_composes_depth_spans() -> None:
    result = bootstrap_blank_decoder_models(
        _qwen3_source(),
        _qwen3_target(),
        _batch(580, batch_size=16),
        validation_inputs=_batch(581, batch_size=16),
        residual_map_method="pca",
        max_coordinate_rows=24,
        fit_full_lm_head=False,
    )

    assert result.transaction.accepted is True, result.transaction.report["gate"]
    for block_index in range(2):
        attention = result.fit_reports[
            f"block.{block_index}.attention.structural_prior"
        ]
        mlp = result.fit_reports[f"block.{block_index}.mlp.structural_prior"]
        assert attention["applied"] is True
        assert attention["coordinate_method"] == "pca-coordinate-formula"
        assert attention["producer_aggregation"] == "mean"
        assert attention["residual_consumer_aggregation"] == "sum"
        assert mlp["applied"] is True
        assert mlp["coordinate_method"] == "pca-coordinate-formula"
        assert mlp["residual_consumer_aggregation"] == "sum"
        assert (
            result.fit_reports[f"block.{block_index}.attention.norm"]["finite"] is True
        )
        assert result.fit_reports[f"block.{block_index}.mlp.norm"]["finite"] is True
        assert (
            result.fit_reports[f"block.{block_index}.attention.query_norm"]["method"]
            == "deterministic-full-coverage-effective-rms-scale-mean"
        )
        assert (
            result.fit_reports[f"block.{block_index}.attention.key_norm"]["method"]
            == "deterministic-full-coverage-effective-rms-scale-mean"
        )
        feature_report = result.fit_reports[
            f"block.{block_index}.attention.feature_folds"
        ]
        assert feature_report["query"]["method"] == (
            "gqa-head-rope-pair-constrained-fold"
        )
        assert feature_report["query"]["gqa_routing_preserved"] is True
        assert feature_report["query"]["rotary_pairs_preserved"] is True
        mlp_report = result.fit_reports[f"block.{block_index}.mlp.feature_fold"]
        assert mlp_report["full_source_coverage"] is True
        assert mlp_report["sign_flips_allowed"] is False

    span = result.fit_reports["block.1.span"]
    assert span["source_layers"] == [1, 2]
    assert span["residual_composition"] == "ordered-source-span-endpoint"
    assert span["attention_delta_target"] == ("sum-encoded-source-attention-deltas")
    assert span["structural_prior_residual_consumers"] == "sum"


def test_real_medoid_bootstrap_copies_parent_circuits_with_full_coverage() -> None:
    result = bootstrap_blank_decoder_models(
        _qwen3_source(),
        _qwen3_target(head_dim=4),
        _batch(584, batch_size=16),
        validation_inputs=_batch(585, batch_size=16),
        residual_map_method="pca",
        nonlinear_merge_method="real_medoid",
        max_coordinate_rows=24,
        fit_full_lm_head=False,
    )

    for block_index, source_layer_count, representative_layer in (
        (0, 1, 0),
        (1, 2, 2),
    ):
        attention = result.fit_reports[
            f"block.{block_index}.attention.activation_cluster"
        ]
        assert attention["source_query_heads"] == 4 * source_layer_count
        assert attention["source_key_value_heads"] == 2 * source_layer_count
        assert attention["target_query_heads"] == 4
        assert attention["target_key_value_heads"] == 2
        assert attention["full_source_coverage"] is True
        assert attention["producer_aggregation"] == "none"
        assert attention["invariants"]["real_source_query_head_medoids"] is True
        assert attention["invariants"]["real_source_group_medoids"] is True
        assert attention["invariants"]["no_producer_averaging"] is True
        assert set(attention["group_medoids"]) <= set(
            attention["group_medoid_candidate_indices"]
        )
        assert (
            result.fit_reports[f"block.{block_index}.attention.structural_prior"][
                "norm_and_rope_coherent_source_layer"
            ]
            == representative_layer
        )
        assert result.fit_reports[f"block.{block_index}.attention.query_norm"][
            "source_layers"
        ] == [representative_layer]

        mlp = result.fit_reports[f"block.{block_index}.mlp.activation_cluster"]
        assert mlp["source_neurons"] == 20 * source_layer_count
        assert mlp["target_neurons"] == 12
        assert mlp["full_source_coverage"] is True
        assert mlp["producer_aggregation"] == "none"
        assert mlp["consumer_coefficient_gate_division"] is False
        assert mlp["invariants"]["real_source_neuron_medoids"] is True
        assert mlp["invariants"]["shared_gate_up_down_assignment"] is True
        assert mlp["invariants"]["no_producer_averaging"] is True

        span = result.fit_reports[f"block.{block_index}.span"]
        assert span["nonlinear_merge_method"] == "real_medoid"
        assert span["structural_prior_feature_producers"] == "real-medoid-copy"

    assert all(
        metric.finite
        for metric in (
            result.candidate_gate_metrics.final_state,
            result.candidate_gate_metrics.logits,
        )
    )


def test_real_medoid_final_model_keeps_selected_producer_weights(monkeypatch) -> None:
    source = _qwen3_source()
    target = _qwen3_target(head_dim=4)
    attention_snapshots = []
    mlp_snapshots = []
    original_attention = bootstrap_module._initialize_attention_real_medoid_prior
    original_mlp = bootstrap_module._initialize_gated_mlp_real_medoid_prior

    def capture_attention(*args, **kwargs):
        result = original_attention(*args, **kwargs)
        operators = bootstrap_module.read_attention_operators(
            args[2], args[5].attention
        )
        attention_snapshots.append(
            tuple(
                operator.weight.clone()
                for operator in (operators.query, operators.key, operators.value)
            )
        )
        return result

    def capture_mlp(*args, **kwargs):
        result = original_mlp(*args, **kwargs)
        operators = bootstrap_module.read_mlp_operators(args[2], args[4])
        mlp_snapshots.append(
            (operators.gate.weight.clone(), operators.up.weight.clone())
        )
        return result

    monkeypatch.setattr(
        bootstrap_module,
        "_initialize_attention_real_medoid_prior",
        capture_attention,
    )
    monkeypatch.setattr(
        bootstrap_module,
        "_initialize_gated_mlp_real_medoid_prior",
        capture_mlp,
    )
    result = bootstrap_blank_decoder_models(
        source,
        target,
        _batch(586, batch_size=16),
        validation_inputs=_batch(587, batch_size=16),
        residual_map_method="pca",
        nonlinear_merge_method="real_medoid",
        max_coordinate_rows=24,
        fit_full_lm_head=False,
    )

    assert result.transaction.accepted is True, result.transaction.report["gate"]
    layout = bootstrap_module.build_decoder_layout(target)
    for index, block in enumerate(layout.blocks):
        attention = bootstrap_module.read_attention_operators(target, block.attention)
        for actual, expected in zip(
            (attention.query.weight, attention.key.weight, attention.value.weight),
            attention_snapshots[index],
            strict=True,
        ):
            assert torch.equal(actual, expected)
        mlp = bootstrap_module.read_mlp_operators(target, block)
        assert torch.equal(mlp.gate.weight, mlp_snapshots[index][0])
        assert torch.equal(mlp.up.weight, mlp_snapshots[index][1])


def test_activation_subset_bootstrap_uses_full_residual_neuron_support() -> None:
    result = bootstrap_blank_decoder_models(
        _qwen3_source(),
        _qwen3_target(head_dim=4),
        _batch(591, batch_size=8, sequence_length=8),
        validation_inputs=_batch(592, batch_size=8, sequence_length=8),
        trace_pooling="token_rows",
        residual_map_method="pca",
        nonlinear_merge_method="activation_subset",
        max_coordinate_rows=64,
        fit_full_lm_head=False,
    )

    for block_index, source_layer_count in ((0, 1), (1, 2)):
        subset = result.fit_reports[f"block.{block_index}.mlp.activation_subset"]
        assert subset["source_neurons"] == 20 * source_layer_count
        assert subset["target_neurons"] == 12
        assert subset["observation_rows"] == 64
        assert len(subset["selected_source_neurons"]) == 12
        assert subset["joint_consumer_fit"] is True
        assert subset["coupled_gate_up_down_identities"] is True
        assert subset["classification"] == "approximate"
        assert subset["zero_gradient"] is True
        span = result.fit_reports[f"block.{block_index}.span"]
        assert span["nonlinear_merge_method"] == "activation_subset"
        assert span["structural_prior_feature_producers"] == (
            "real-leverage-subset-copy"
        )
    assert result.candidate_gate_metrics.final_state.finite is True
    assert result.candidate_gate_metrics.logits.finite is True


def test_activation_subset_rejects_underdetermined_last_valid_rows() -> None:
    with pytest.raises(BlankBootstrapError, match="sampled trajectory rows"):
        bootstrap_blank_decoder_models(
            _qwen3_source(),
            _qwen3_target(head_dim=4),
            _batch(593, batch_size=8, sequence_length=8),
            validation_inputs=_batch(594, batch_size=8, sequence_length=8),
            trace_pooling="last_valid",
            residual_map_method="pca",
            nonlinear_merge_method="activation_subset",
            max_coordinate_rows=64,
            fit_full_lm_head=False,
        )


def test_dual_orthonormal_bootstrap_reports_margin_oracle() -> None:
    result = bootstrap_blank_decoder_models(
        _qwen3_source(),
        _qwen3_target(),
        _batch(582, batch_size=16),
        validation_inputs=_batch(583, batch_size=16),
        residual_map_method="dual_orthonormal",
        max_coordinate_rows=24,
        dual_alpha=1.0,
        dual_beta=1.0,
        dual_gamma=2.0,
        dual_margin_topk=3,
        dual_oracle_vocab_rows=32,
        fit_full_lm_head=False,
    )

    assert result.transaction.accepted is True, result.transaction.report["gate"]
    report = result.fit_reports["residual.map"]
    assert report["method"] == "dual-aware-orthonormal-residual-basis"
    assert report["basis"]["covariance_weights"] == {
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 2.0,
    }
    assert report["basis"]["orthonormality_max_abs_error"] < 1e-12
    assert report["teacher_margins"]["topk"] == 3
    assert report["teacher_margins"]["direction_rows"] > 0
    oracle = report["heldout_oracle"]
    assert oracle["kind"] == "dual-basis-no-model-oracle"
    assert 0.0 <= oracle["top1_agreement"] <= 1.0
    assert 0.0 <= oracle["teacher_margin_sign_retention"] <= 1.0
    assert oracle["vocabulary_rows"] == 32
    assert result.fit_reports["block.0.attention.structural_prior"]["applied"] is True
    assert result.fit_reports["block.0.mlp.structural_prior"]["applied"] is True


def test_external_canonical_boundaries_drive_block_endpoints_and_gate() -> None:
    source = _qwen3_source()
    target = _qwen3_target(head_dim=4)
    calibration = _batch(589, batch_size=8)
    validation = _batch(590, batch_size=8)
    source_calibration = bootstrap_module._trace_rows(
        source,
        calibration,
        "token_rows",
    )
    source_validation = bootstrap_module._trace_rows(
        source,
        validation,
        "token_rows",
    )

    def boundaries(trace):
        return (
            trace.blocks[0].block_input[:, :8],
            trace.blocks[0].block_output[:, :8],
            trace.blocks[-1].block_output[:, :8],
        )

    result = bootstrap_blank_decoder_models(
        source,
        target,
        calibration,
        validation_inputs=validation,
        canonical_calibration_boundaries=boundaries(source_calibration),
        canonical_validation_boundaries=boundaries(source_validation),
        canonical_calibration_target_log_probs=torch.log_softmax(
            source_calibration.logits,
            dim=-1,
        ),
        canonical_validation_target_log_probs=torch.log_softmax(
            source_validation.logits,
            dim=-1,
        ),
        canonical_execution_graph_fingerprint="a" * 64,
        residual_map_method="pca",
        max_coordinate_rows=24,
        fit_full_lm_head=False,
    )

    contract = result.fit_reports["canonical.boundaries"]
    assert contract["boundary_count"] == 3
    assert contract["hidden_size"] == 8
    assert contract["used_for_ordered_block_endpoints"] is True
    assert contract["used_for_heldout_final_state_gate"] is True
    assert contract["exactness_class"] == "finite-trace-approximate"
    assert contract["algebraically_exact"] is False
    teacher = result.fit_reports["teacher_logits.target_space"]
    assert teacher["projection"] == "simultaneous-parent-probability-mixture"
    assert teacher["canonical_execution_graph_fingerprint"] == "a" * 64
    assert teacher["single_authoritative_parent"] is False
    assert len(contract["calibration_sha256"]) == 64
    assert len(contract["validation_sha256"]) == 64
    for block_index in range(2):
        assert result.fit_reports[f"block.{block_index}.span"]["endpoint_target"] == (
            "architecture-neutral-canonical-parent-kernel"
        )
    assert result.candidate_gate_metrics.final_state.finite is True
    assert result.candidate_gate_metrics.logits.finite is True


def test_blank_bootstrap_rejects_a_wider_target_before_mutation() -> None:
    source = _gpt2_source()
    target = _llama_target(hidden_size=16)
    before = _state(target)

    with pytest.raises(BlankBootstrapError, match="hidden size"):
        bootstrap_blank_decoder_models(source, target, _batch())

    for name, value in target.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_blank_bootstrap_rejects_an_undersized_coordinate_sample() -> None:
    with pytest.raises(BlankBootstrapError, match="max_coordinate_rows"):
        bootstrap_blank_decoder_models(
            _gpt2_source(),
            _llama_target(),
            _batch(),
            max_coordinate_rows=7,
        )


def test_qwen_target_transfers_head_norms_without_baking_them_twice(
    monkeypatch,
) -> None:
    source = _qwen3_source()
    target = _qwen3_target()
    applied_head_norms = []
    original_apply = bootstrap_module._apply_head_norm

    def record_apply(*args, **kwargs):
        applied_head_norms.append(args[3].path)
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(bootstrap_module, "_apply_head_norm", record_apply)
    result = bootstrap_blank_decoder_models(
        source,
        target,
        _batch(601),
        validation_inputs=_batch(602),
        max_coordinate_rows=24,
        vocab_projection_chunk_size=7,
    )

    assert applied_head_norms == []
    assert result.transaction.accepted is True, result.transaction.report["gate"]
    assert result.transaction.report["coverage"]["complete"] is True
    manifest = build_decoder_transport_manifest(target)
    assert result.transaction.report["coverage"]["evidenced_role_count"] == len(
        manifest.plan.required_roles
    )
    for block_index, expected_layers in ((0, [0]), (1, [1, 2])):
        for role in ("query_norm", "key_norm"):
            report = result.fit_reports[f"block.{block_index}.attention.{role}"]
            assert report["method"] == (
                "deterministic-full-coverage-effective-rms-scale-mean"
            )
            assert report["source_layers"] == expected_layers
            assert report["full_source_coverage"] is True
            assert report["target_head_dim"] == 2
            assert report["dimension_group_sizes"] == [[2, 2]] * len(expected_layers)


class _HeadNormCarrier(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.norms = torch.nn.ModuleList()
        for weight in weights:
            module = torch.nn.Module()
            module.weight = torch.nn.Parameter(
                torch.tensor(weight, dtype=torch.float32)
            )
            self.norms.append(module)
        self.target_norm = torch.nn.Module()
        self.target_norm.weight = torch.nn.Parameter(torch.zeros(2))


def _head_norm_layout(path: str, *, weight_kind: str) -> NormLayout:
    return NormLayout(
        path=path,
        kind="rms_norm",
        placement="query_head",
        epsilon=1e-6,
        has_bias=False,
        weight_kind=weight_kind,
    )


def test_head_norm_transfer_uses_effective_scales_span_mean_and_identity() -> None:
    model = _HeadNormCarrier(
        (
            [0.0, 2.0, 4.0, 6.0],
            [3.0, 5.0, 7.0, 9.0],
        )
    )
    source_layout = SimpleNamespace(
        blocks=(
            SimpleNamespace(
                attention=SimpleNamespace(
                    query_norm=_head_norm_layout(
                        "norms.0",
                        weight_kind="unit_offset",
                    ),
                    head_dim=4,
                )
            ),
            SimpleNamespace(
                attention=SimpleNamespace(
                    query_norm=_head_norm_layout(
                        "norms.1",
                        weight_kind="direct_scale",
                    ),
                    head_dim=4,
                )
            ),
            SimpleNamespace(attention=SimpleNamespace(query_norm=None, head_dim=4)),
        )
    )
    target_norm = _head_norm_layout("target_norm", weight_kind="unit_offset")

    report = bootstrap_module._transport_target_head_norm(
        torch,
        model,
        model,
        source_layout,
        (0, 1, 2),
        target_norm,
        2,
        role="query",
    )

    # Effective folds are [2, 6], [4, 8], and identity [1, 1].
    expected_effective = torch.tensor([7.0 / 3.0, 5.0])
    assert torch.allclose(model.target_norm.weight + 1.0, expected_effective)
    assert report["source_layers"] == [0, 1, 2]
    assert report["source_norm_paths"] == ["norms.0", "norms.1"]
    assert report["identity_source_layers"] == [2]
    assert report["dimension_group_sizes"] == [[2, 2], [2, 2], [2, 2]]
    assert report["full_source_coverage"] is True


def test_head_norm_baking_uses_unit_offset_effective_scale() -> None:
    model = _HeadNormCarrier(([0.0, 1.0, 2.0, 3.0],))
    layout = _head_norm_layout("norms.0", weight_kind="unit_offset")
    rows = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    actual = bootstrap_module._apply_head_norm(
        torch,
        rows,
        model,
        layout,
        1,
        4,
    )

    normalized = rows * torch.rsqrt(rows.square().mean(dim=-1, keepdim=True) + 1e-6)
    expected = normalized * torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.allclose(actual, expected.to(torch.float64))


@pytest.mark.parametrize(
    ("kind", "has_bias", "message"),
    [
        ("layer_norm", False, "Only per-head RMSNorm"),
        ("rms_norm", True, "Biased per-head normalization"),
    ],
)
def test_head_norm_transfer_fails_closed_for_unsupported_topology(
    kind,
    has_bias,
    message,
) -> None:
    model = _HeadNormCarrier(([1.0, 1.0, 1.0, 1.0],))
    source_layout = SimpleNamespace(
        blocks=(
            SimpleNamespace(attention=SimpleNamespace(query_norm=None, head_dim=4)),
        )
    )
    target_norm = NormLayout(
        path="target_norm",
        kind=kind,
        placement="query_head",
        epsilon=1e-6,
        has_bias=has_bias,
        weight_kind="direct_scale",
    )

    with pytest.raises(BlankBootstrapError, match=message):
        bootstrap_module._transport_target_head_norm(
            torch,
            model,
            model,
            source_layout,
            (0,),
            target_norm,
            2,
            role="query",
        )


class _TiedVocabularyCarrier(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.embedding = torch.nn.Module()
        self.embedding.weight = torch.nn.Parameter(weight.detach().clone())
        self.lm_head = torch.nn.Module()
        self.lm_head.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head


def _tied_vocabulary_layout(*, vocab_size: int, hidden_size: int):
    return SimpleNamespace(
        tie_word_embeddings=True,
        hidden_size=hidden_size,
        embeddings=SimpleNamespace(
            token_path="embedding",
            vocab_size=vocab_size,
        ),
        lm_head_path="lm_head",
    )


def test_joint_sparse_fold_reports_and_preserves_structural_transport() -> None:
    source = _gpt2_source()
    target = _gpt2_target()

    result = bootstrap_blank_decoder_models(
        source,
        target,
        _batch(701),
        validation_inputs=_batch(702),
        residual_map_method="joint_sparse_fold",
        max_coordinate_rows=24,
        vocab_projection_chunk_size=7,
        vocab_row_norm_repair="source_l2",
    )

    report = result.fit_reports["residual.map"]
    assert report["method"] == "joint-logit-aware-sparse-fold"
    assert report["feature_fold"]["full_source_coverage"] is True
    assert report["feature_fold"]["deterministic"] is True
    assert report["normalization"] == {
        "method": "equal-total-frobenius-energy",
        "target_energy_per_input": 12.0,
        "prevents_row-count_or_raw-energy_dominance": True,
    }
    residual_input = report["inputs"]["parent_residual_activations"]
    vocabulary_input = report["inputs"]["tied_source_embedding_lm_head"]
    assert residual_input["sampled_rows"] == 24
    assert vocabulary_input["sampling"]["sampled_rows"] == 24
    assert vocabulary_input["sampling"]["available_rows"] == 64
    assert vocabulary_input["full_vocabulary_copy_materialized"] is False
    assert vocabulary_input["provenance"] == {
        "embedding_path": "transformer.wte",
        "lm_head_path": "lm_head",
        "weight_shape": [64, 12],
        "tied_by_storage": True,
    }
    for basis_input in (residual_input, vocabulary_input):
        assert basis_input["normalized_total_energy"] == pytest.approx(12.0)
    assert result.fit_reports["block.0.attention.structural_prior"]["applied"] is True
    assert result.fit_reports["block.0.mlp.structural_prior"]["applied"] is True
    assert result.transaction.report["coverage"]["complete"] is True
    embedding_repair = result.fit_reports["embeddings.token"]["row_norm_repair"]
    head_repair = result.fit_reports["lm_head"]["row_norm_repair"]
    for repair in (embedding_repair, head_repair):
        assert repair["method"] == "source-l2-row-norm-repair"
        assert repair["total_rows"] == 64
        assert repair["repaired_rows"] == 64
        assert repair["zero_source_rows"] == 0
        assert repair["zero_projected_rows"] == 0
        assert repair["finite"] is True
        assert repair["applied_source_to_projected_ratio"]["count"] == 64
        assert repair["post_copy_to_source_ratio"]["count"] == 64
    assert result.fit_reports["embeddings.token"]["target_tied_storage"] is True
    assert result.fit_reports["lm_head"]["target_tied_storage"] is True


def test_joint_sparse_fold_vocabulary_changes_assignments_deterministically() -> None:
    activation_generator = torch.Generator().manual_seed(1)
    vocabulary_generator_a = torch.Generator().manual_seed(100)
    vocabulary_generator_b = torch.Generator().manual_seed(101)
    residual_rows = torch.randn(
        8,
        4,
        generator=activation_generator,
        dtype=torch.float64,
    )
    vocabulary_a = torch.randn(
        8,
        4,
        generator=vocabulary_generator_a,
        dtype=torch.float64,
    )
    vocabulary_b = torch.randn(
        8,
        4,
        generator=vocabulary_generator_b,
        dtype=torch.float64,
    )
    model = _TiedVocabularyCarrier(vocabulary_a)
    layout = _tied_vocabulary_layout(vocab_size=8, hidden_size=4)

    geometry_a = bootstrap_module._require_tied_source_vocabulary_geometry(
        torch,
        model,
        layout,
    )
    fold_a, report_a = bootstrap_module._build_joint_sparse_residual_map(
        torch,
        residual_rows,
        geometry_a,
        2,
        8,
        1,
    )
    repeat_a, repeat_report_a = bootstrap_module._build_joint_sparse_residual_map(
        torch,
        residual_rows,
        geometry_a,
        2,
        8,
        1,
    )
    with torch.no_grad():
        model.embedding.weight.copy_(vocabulary_b)
    geometry_b = bootstrap_module._require_tied_source_vocabulary_geometry(
        torch,
        model,
        layout,
    )
    fold_b, report_b = bootstrap_module._build_joint_sparse_residual_map(
        torch,
        residual_rows,
        geometry_b,
        2,
        8,
        1,
    )

    assert torch.equal(fold_a.assignment, repeat_a.assignment)
    assert torch.equal(fold_a.coefficients, repeat_a.coefficients)
    assert report_a == repeat_report_a
    assert not torch.equal(fold_a.assignment, fold_b.assignment)
    assert report_a != report_b
    for report in (report_a, report_b):
        assert report["feature_fold"]["full_source_coverage"] is True
        assert report["feature_fold"]["source_channels_covered"] == 4
        assert (
            report["inputs"]["tied_source_embedding_lm_head"]["sampling"][
                "deterministic"
            ]
            is True
        )


def test_joint_sparse_fold_rejects_untied_source_before_target_mutation() -> None:
    source = _llama_source()
    target = _llama_target()
    before = _state(target)

    with pytest.raises(BlankBootstrapError, match="genuinely tied source"):
        bootstrap_blank_decoder_models(
            source,
            target,
            _batch(703),
            residual_map_method="joint_sparse_fold",
        )

    for name, value in target.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_source_l2_vocab_repair_preserves_row_norms_and_tied_storage() -> None:
    coordinates = bootstrap_module.FeatureFoldMap(
        torch.tensor([0, 0, 1, 1]),
        torch.ones(4, dtype=torch.float64),
        2,
    )
    source_weight = torch.tensor(
        [
            [3.0, 4.0, 1.0, 2.0],
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    target = _TiedVocabularyCarrier(torch.zeros(3, 2, dtype=torch.float64))
    original_pointer = int(target.embedding.weight.data_ptr())

    embedding_report = bootstrap_module._copy_projected_rows(
        torch,
        source_weight,
        target.embedding.weight,
        coordinates,
        2,
        row_norm_repair="source_l2",
    )
    head_report = bootstrap_module._copy_projected_rows(
        torch,
        source_weight,
        target.lm_head.weight,
        coordinates,
        2,
        row_norm_repair="source_l2",
    )

    assert target.embedding.weight is target.lm_head.weight
    assert int(target.embedding.weight.data_ptr()) == original_pointer
    assert torch.allclose(
        torch.linalg.vector_norm(target.embedding.weight, dim=1),
        torch.linalg.vector_norm(source_weight, dim=1),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.equal(target.embedding.weight[2], torch.zeros(2, dtype=torch.float64))
    for report in (embedding_report, head_report):
        assert report["repaired_rows"] == 2
        assert report["zero_source_rows"] == 1
        assert report["zero_projected_rows"] == 1
        assert report["zeroed_projected_rows"] == 0
        assert report["finite"] is True
        assert report["post_copy_to_source_ratio"] == {
            "count": 2,
            "minimum": pytest.approx(1.0),
            "maximum": pytest.approx(1.0),
            "mean": pytest.approx(1.0),
        }


def test_source_l2_vocab_repair_fails_if_nonzero_row_projects_to_zero() -> None:
    coordinates = bootstrap_module.FeatureFoldMap(
        torch.tensor([0, 0, 1, 1]),
        torch.ones(4, dtype=torch.float64),
        2,
    )
    source_weight = torch.tensor(
        [[1.0, -1.0, 2.0, -2.0]],
        dtype=torch.float64,
    )
    target_weight = torch.full((1, 2), 7.0, dtype=torch.float64)

    with pytest.raises(BlankBootstrapError, match="projected to zero"):
        bootstrap_module._copy_projected_rows(
            torch,
            source_weight,
            target_weight,
            coordinates,
            1,
            row_norm_repair="source_l2",
        )

    assert torch.equal(target_weight, torch.full((1, 2), 7.0, dtype=torch.float64))
