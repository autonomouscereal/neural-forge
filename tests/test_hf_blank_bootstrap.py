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


def _llama_target(*, hidden_size: int = 8):
    torch.manual_seed(303)
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=64,
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
        assert result.fit_reports["block.0.attention.structural_prior"][
            "applied"
        ] is True
        assert result.fit_reports["block.0.mlp.structural_prior"]["applied"] is True
        assert result.fit_reports["block.0.mlp.feature_fold"][
            "negative_coefficient_count"
        ] == 0
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
            assert report["dimension_group_sizes"] == [[2, 2]] * len(
                expected_layers
            )


class _HeadNormCarrier(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.norms = torch.nn.ModuleList()
        for weight in weights:
            module = torch.nn.Module()
            module.weight = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float32))
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
            SimpleNamespace(
                attention=SimpleNamespace(query_norm=None, head_dim=4)
            ),
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
            SimpleNamespace(
                attention=SimpleNamespace(query_norm=None, head_dim=4)
            ),
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
        assert report["inputs"]["tied_source_embedding_lm_head"]["sampling"][
            "deterministic"
        ] is True


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
