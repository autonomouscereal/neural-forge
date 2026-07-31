from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_lfm2_dynamic_structural_compression.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_dynamic_structural_compression",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


_LFM2_MODULE = "transformers.models.lfm2.modeling_lfm2"


class Config(SimpleNamespace):
    def to_dict(self):
        return copy.deepcopy(vars(self))


class Lfm2RMSNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))


class Lfm2MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.w1 = nn.Linear(hidden, intermediate, bias=False)
        self.w3 = nn.Linear(hidden, intermediate, bias=False)
        self.w2 = nn.Linear(intermediate, hidden, bias=False)


class Lfm2Attention(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_layernorm = Lfm2RMSNorm(config.head_dim)
        self.k_layernorm = Lfm2RMSNorm(config.head_dim)


class Lfm2ShortConv(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias
        hidden = config.hidden_size
        self.in_proj = nn.Linear(hidden, 3 * hidden, bias=config.conv_bias)
        self.conv = nn.Conv1d(
            hidden,
            hidden,
            config.conv_L_cache,
            groups=hidden,
            padding=config.conv_L_cache - 1,
            bias=config.conv_bias,
        )
        self.out_proj = nn.Linear(hidden, hidden, bias=config.conv_bias)


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, config: Config, layer_idx: int):
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(config, layer_idx)
        else:
            self.conv = Lfm2ShortConv(config, layer_idx)
        self.feed_forward = Lfm2MLP(
            config.hidden_size,
            config.intermediate_size,
        )
        self.operator_norm = Lfm2RMSNorm(config.hidden_size)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size)


for _cls in (
    Lfm2RMSNorm,
    Lfm2MLP,
    Lfm2Attention,
    Lfm2ShortConv,
    Lfm2DecoderLayer,
):
    _cls.__module__ = _LFM2_MODULE


class _Body(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = nn.Identity()
        self.layers = nn.ModuleList(
            [Lfm2DecoderLayer(config, index) for index in range(14)]
        )
        self.embedding_norm = Lfm2RMSNorm(config.hidden_size)


class TinyLfm2(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = _Body(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _config(
    *,
    hidden: int = 8,
    query: int = 4,
    key_value: int = 2,
    head_dim: int = 2,
    intermediate: int = 6,
) -> Config:
    layer_types = [
        "conv",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
    ]
    return Config(
        model_type="lfm2",
        vocab_size=19,
        hidden_size=hidden,
        block_dim=hidden,
        conv_dim=hidden,
        intermediate_size=intermediate,
        block_ff_dim=intermediate,
        block_auto_adjust_ff_dim=False,
        num_hidden_layers=14,
        num_layers=14,
        num_attention_heads=query,
        num_heads=query,
        num_key_value_heads=key_value,
        head_dim=head_dim,
        max_position_embeddings=128_000,
        rope_parameters={
            "rope_theta": 1_000_000.0,
            "rope_type": "default",
        },
        layer_types=layer_types,
        conv_L_cache=3,
        conv_bias=False,
        tie_word_embeddings=True,
        _name_or_path="local-test",
    )


def _small_candidate() -> MODULE.StructuralCandidate:
    target_config = _config(
        hidden=4,
        query=2,
        key_value=1,
        head_dim=2,
        intermediate=3,
    )
    exact = MODULE.dense_logical_parameter_audit(TinyLfm2(target_config))[
        "unique_logical_parameters"
    ]
    return MODULE.StructuralCandidate(
        "tiny-h4",
        4,
        2,
        1,
        2,
        3,
        exact,
        1.0 - exact / MODULE.EXPECTED_PARENT_UNIQUE_PARAMETERS,
    )


def _evidence(parent: TinyLfm2, rows: int = 32):
    generator = torch.Generator().manual_seed(77)
    hidden = parent.config.hidden_size
    retained_rank = hidden // 2
    evidence_basis, _ = torch.linalg.qr(
        torch.randn(hidden, retained_rank, generator=generator)
    )

    def high_retention_rows(count: int) -> torch.Tensor:
        latent = torch.randn(count, retained_rank, generator=generator)
        noise = torch.randn(count, hidden, generator=generator) * 1e-3
        return latent @ evidence_basis.T + noise

    residual = high_retention_rows(rows * 2)
    observable = high_retention_rows(rows)
    margins = high_retention_rows(rows)
    fields = []
    for layer in parent.model.layers:
        gate = torch.randn(
            rows,
            parent.config.intermediate_size,
            generator=generator,
        )
        up = torch.randn(
            rows,
            parent.config.intermediate_size,
            generator=generator,
        )
        layer_fields = {
            "mlp.gate": gate,
            "mlp.up": up,
            "mlp.post_swiglu": torch.nn.functional.silu(gate) * up,
            "residual.pre": torch.randn(rows, hidden, generator=generator),
            "residual.post_operator": torch.randn(
                rows, hidden, generator=generator
            ),
            "residual.post_layer": torch.randn(rows, hidden, generator=generator),
        }
        if layer.is_attention_layer:
            layer_fields.update(
                {
                    "attention.q": torch.randn(
                        rows,
                        parent.config.num_attention_heads
                        * parent.config.head_dim,
                        generator=generator,
                    ),
                    "attention.k": torch.randn(
                        rows,
                        parent.config.num_key_value_heads
                        * parent.config.head_dim,
                        generator=generator,
                    ),
                    "attention.v": torch.randn(
                        rows,
                        parent.config.num_key_value_heads
                        * parent.config.head_dim,
                        generator=generator,
                    ),
                    "attention.context": torch.randn(
                        rows,
                        parent.config.num_attention_heads
                        * parent.config.head_dim,
                        generator=generator,
                    ),
                }
            )
        fields.append(layer_fields)
    return MODULE.CachedFoldEvidence(
        residual_rows=residual,
        observable_rows=observable,
        margin_rows=margins,
        margin_weights=torch.ones(rows),
        layer_fields=tuple(fields),
        cache_key="a" * 64,
        corpus_sha256="b" * 64,
        evidence_sha256="c" * 64,
    )


class _WholeWeightMap:
    def __init__(self, source_width: int, target_width: int) -> None:
        generator = torch.Generator().manual_seed(811)
        raw = torch.randn(
            source_width,
            target_width,
            generator=generator,
            dtype=torch.float64,
        )
        self.basis = torch.linalg.qr(raw, mode="reduced").Q.contiguous()
        self.map_sha256 = "d" * 64

    @property
    def source_width(self) -> int:
        return int(self.basis.shape[0])

    @property
    def target_width(self) -> int:
        return int(self.basis.shape[1])

    def encode(self, value):
        return value.to(torch.float64) @ self.basis

    def decode(self, value):
        return value.to(torch.float64) @ self.basis.T

    def fold_consumer_weight(self, value):
        return self.encode(value)

    def fold_producer_weight(self, value):
        return self.basis.T @ value.to(torch.float64)

    def materialize_dense(self, *, max_elements, dtype=None, device="cpu"):
        assert self.basis.numel() <= max_elements
        return self.basis.to(
            device=device,
            dtype=self.basis.dtype if dtype is None else dtype,
        )


def test_verified_large_candidate_ladder_uses_true_dense_counts() -> None:
    observed = {
        candidate.name: candidate.verified_unique_parameters
        for candidate in MODULE.STRUCTURAL_CANDIDATES
    }
    assert observed == {
        "h640_q10_kv5_ff1920": 114_067_328,
        "h640_q10_kv5_ff1088": 91_703_168,
        "h512_q8_kv4_ff1568": 80_407_808,
        "h512_q8_kv4_ff1024": 68_709_632,
    }
    assert MODULE.STRUCTURAL_CANDIDATES[0].verified_parent_reduction_fraction > 0.50
    assert MODULE.STRUCTURAL_CANDIDATES[-1].verified_parent_reduction_fraction > 0.70


def test_config_updates_every_lfm2_width_alias_and_keeps_depth() -> None:
    candidate = _small_candidate()
    config = MODULE.build_structural_lfm2_config(_config(), candidate)
    payload = config.to_dict()
    assert payload["hidden_size"] == payload["block_dim"] == payload["conv_dim"] == 4
    assert payload["num_attention_heads"] == payload["num_heads"] == 2
    assert payload["num_key_value_heads"] == 1
    assert payload["head_dim"] == 2
    assert payload["intermediate_size"] == payload["block_ff_dim"] == 3
    assert payload["block_auto_adjust_ff_dim"] is False
    assert payload["num_hidden_layers"] == payload["num_layers"] == 14
    assert payload["layer_types"] == _config().layer_types
    assert payload["max_position_embeddings"] == 128_000
    assert payload["rope_parameters"] == {
        "rope_theta": 1_000_000.0,
        "rope_type": "default",
    }


def test_config_rejects_short_context_or_rope_leakage() -> None:
    candidate = _small_candidate()
    parent = _config()
    child = MODULE.build_structural_lfm2_config(parent, candidate)
    child.max_position_embeddings = 64
    with pytest.raises(
        Exception,
        match="changed context capacity",
    ):
        MODULE.assert_model_context_preserved(parent, child)


def test_parser_exposes_prompt_independent_weight_basis_without_activation() -> None:
    args = MODULE.build_parser().parse_args(
        [
            "--parent-model",
            "parent",
            "--trace-cache",
            "cache",
            "--output",
            "child",
            "--residual-basis",
            "weight-complete",
        ]
    )
    assert args.residual_basis == "weight-complete"
    assert args.weight_basis_activation_weight == 0.0
    assert args.weight_basis_maximum_activation_weight == 0.25
    assert args.weight_basis_minimum_retained_energy == 0.0


def test_parser_exposes_opt_in_circuit_balanced_basis_and_preserves_default() -> None:
    parser = MODULE.build_parser()
    required = [
        "--parent-model",
        "parent",
        "--trace-cache",
        "cache",
        "--output",
        "child",
    ]
    default_args = parser.parse_args(required)
    circuit_args = parser.parse_args(
        required + ["--residual-basis", "circuit-balanced"]
    )
    margin_args = parser.parse_args(
        required
        + [
            "--residual-basis",
            "circuit-balanced",
            "--circuit-basis-vocabulary-margin-observability",
            "--circuit-basis-vocabulary-margin-top-k",
            "5",
            "--circuit-basis-vocabulary-margin-floor-fraction",
            "0.1",
        ]
    )

    assert default_args.residual_basis == "cached-balanced"
    assert circuit_args.residual_basis == "circuit-balanced"
    assert circuit_args.circuit_basis_validation_fraction == 0.25
    assert (
        circuit_args.circuit_basis_split_seed
        == "lfm2-circuit-balanced-v1"
    )
    assert circuit_args.circuit_basis_max_rows_per_block == 512
    assert circuit_args.circuit_basis_structural_weight == 0.35
    assert circuit_args.circuit_basis_cached_weight == 0.65
    assert circuit_args.circuit_basis_minimum_heldout_retained_energy == 0.0
    assert (
        circuit_args.circuit_basis_vocabulary_margin_observability
        is False
    )
    assert circuit_args.circuit_basis_vocabulary_margin_top_k == 8
    assert circuit_args.circuit_basis_vocabulary_margin_floor_fraction == 0.05
    assert margin_args.circuit_basis_vocabulary_margin_observability is True
    assert margin_args.circuit_basis_vocabulary_margin_top_k == 5
    assert margin_args.circuit_basis_vocabulary_margin_floor_fraction == 0.1


def test_whole_weight_map_override_replaces_trace_selected_coordinates() -> None:
    parent = TinyLfm2(_config())
    candidate = _small_candidate()
    evidence = _evidence(parent)
    mapping = _WholeWeightMap(8, 4)
    resolved, report = MODULE.resolve_global_residual_map(
        evidence,
        candidate,
        global_map_override=mapping,
        global_map_report_override={
            "schema": "test-whole-weight-basis-v1",
            "prompt_independent_weight_objective": True,
        },
    )
    assert resolved is mapping
    assert report["map_sha256"] == "d" * 64
    assert report["source_width"] == 8
    assert report["target_width"] == 4
    assert report["source_axes_with_nonzero_loading"] == 8
    assert report["prompt_independent_weight_basis_override"] is True

    with pytest.raises(
        MODULE.Lfm2StructuralCompressionError,
        match="requires its complete audit report",
    ):
        MODULE.resolve_global_residual_map(
            evidence,
            candidate,
            global_map_override=mapping,
        )


def test_circuit_balanced_artifact_is_fresh_reload_verified(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(9091)
    basis = torch.linalg.qr(
        torch.randn(8, 4, generator=generator, dtype=torch.float64),
        mode="reduced",
    ).Q.contiguous()
    identity = torch.eye(8, dtype=torch.float64)
    balanced = SimpleNamespace(
        basis=basis,
        balanced_encoder=basis.T.contiguous(),
        balanced_decoder=basis,
        full_weight_gram=identity,
        controllability_gram=identity * 2.0,
        observability_gram=identity * 3.0,
        heldout_gram=identity * 4.0,
        map_sha256="a" * 64,
        source_width=8,
        target_width=4,
    )
    report = MODULE.save_circuit_balanced_basis_artifact(
        tmp_path,
        balanced,
    )

    assert report["file"] == "residual-coordinate-basis.safetensors"
    assert report["basis_sha256"] == "a" * 64
    assert report["source_width"] == 8
    assert report["target_width"] == 4
    assert report["fresh_tensor_reload_verified"] is True
    assert report["contains_balanced_encoder_decoder"] is True
    assert set(report["tensor_names"]) == {
        "basis",
        "balanced_encoder",
        "balanced_decoder",
        "full_weight_gram",
        "controllability_gram",
        "observability_gram",
        "heldout_gram",
    }


def test_dense_audit_counts_tied_parameter_once_and_rejects_packing() -> None:
    model = TinyLfm2(_config())
    audit = MODULE.dense_logical_parameter_audit(model, require_any_nonzero=True)
    naive = sum(parameter.numel() for _, parameter in model.named_parameters(remove_duplicate=False))
    assert audit["unique_logical_parameters"] < naive
    assert any(
        set(group) == {"lm_head.weight", "model.embed_tokens.weight"}
        for group in audit["alias_groups"]
    )

    class PackedProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 2))

    with pytest.raises(
        MODULE.Lfm2StructuralCompressionError,
        match="Packed/quantized",
    ):
        MODULE.dense_logical_parameter_audit(PackedProjection())


def test_compiler_blank_is_zero_alias_free_and_seed_erased_before_fold() -> None:
    torch.manual_seed(1)
    parent = TinyLfm2(_config())
    candidate = _small_candidate()
    first = MODULE.instantiate_compiler_blank(
        parent,
        candidate,
        seed=11,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second = MODULE.instantiate_compiler_blank(
        parent,
        candidate,
        seed=29,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert first.random_state_sha256 != second.random_state_sha256
    assert first.blank_state_sha256 == second.blank_state_sha256
    assert first.provenance["compiler_created_blank"] is True
    assert first.provenance["all_parameters_zero"] is True
    assert all(
        torch.count_nonzero(parameter).item() == 0
        for parameter in first.model.parameters()
    )
    parent_views = {
        MODULE._storage_view(parameter)
        for parameter in parent.parameters()
    }
    child_views = {
        MODULE._storage_view(parameter)
        for parameter in first.model.parameters()
    }
    assert parent_views.isdisjoint(child_views)


def test_global_residual_and_attention_maps_cover_every_source_unit() -> None:
    parent = TinyLfm2(_config())
    candidate = _small_candidate()
    evidence = _evidence(parent)
    residual, report = MODULE.build_global_residual_map(evidence, candidate)
    assert residual.source_width == 8
    assert residual.target_width == 4
    assert isinstance(residual, MODULE.DenseCoordinateFoldMap)
    assert torch.allclose(
        residual.basis.T @ residual.basis,
        torch.eye(4, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )
    repeated, repeated_report = MODULE.build_global_residual_map(
        evidence,
        candidate,
    )
    assert torch.equal(residual.basis, repeated.basis)
    assert residual.map_sha256 == repeated.map_sha256
    assert report == repeated_report
    assert report["kind"].endswith("dense-orthonormal-residual-basis")
    assert report["sparse_assignment_used"] is False
    assert report["centered_for_fit"] is False
    assert report["source_coverage_fraction"] == 1.0
    assert report["source_axes_with_nonzero_loading"] == 8
    assert report["dropped_source_units"] == 0
    assert (
        report["weighted_combined_exact_retained_energy_fraction"]
        >= 0.94
    )
    assert report["weighted_combined_retained_energy_gate_passed"] is True
    objectives = report["objective_weighting"]["objectives"]
    assert objectives["residual"]["raw_retained_energy_fraction"] >= 0.95
    assert objectives["observable"]["raw_retained_energy_fraction"] >= 0.85
    assert objectives["teacher_margin"]["raw_retained_energy_fraction"] >= 0.98
    assert objectives["teacher_margin"]["role_priority"] == 1.0
    assert (
        objectives["teacher_margin"][
            "teacher_margin_row_weighting_preserved"
        ]
        is True
    )
    assert (
        report["raw_unweighted_combined_diagnostic"][
            "used_for_acceptance_gate"
        ]
        is False
    )

    maps, attention = MODULE.build_attention_maps(
        evidence.layer_fields[2],
        source_query_heads=4,
        source_key_value_heads=2,
        target_query_heads=2,
        target_key_value_heads=1,
        head_dim=2,
    )
    assert maps["query"].source_width == 8
    assert maps["key"].source_width == 4
    assert sum(maps["query"].group_sizes) == 8
    assert sum(maps["key"].group_sizes) == 4
    assert attention["source_coverage_fraction"] == 1.0
    assert attention["dropped_source_heads"] == 0


def test_dense_residual_basis_is_byte_stable_across_caller_thread_counts() -> None:
    parent = TinyLfm2(_config())
    evidence = _evidence(parent)
    candidate = _small_candidate()
    original_threads = torch.get_num_threads()
    alternate_threads = 2 if original_threads != 2 else 3
    try:
        torch.set_num_threads(alternate_threads)
        first, first_report = MODULE.build_global_residual_map(
            evidence,
            candidate,
        )
        assert torch.get_num_threads() == alternate_threads

        torch.set_num_threads(1)
        second, second_report = MODULE.build_global_residual_map(
            evidence,
            candidate,
        )
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(original_threads)

    assert torch.equal(first.basis, second.basis)
    assert first.map_sha256 == second.map_sha256
    assert first_report == second_report


def test_global_dense_residual_energy_gate_rejects_low_retention_basis() -> None:
    parent = TinyLfm2(_config())
    evidence = _evidence(parent)
    isotropic = torch.eye(parent.config.hidden_size)
    low_retention = MODULE.CachedFoldEvidence(
        residual_rows=isotropic,
        observable_rows=isotropic,
        margin_rows=isotropic,
        margin_weights=torch.ones(parent.config.hidden_size),
        layer_fields=evidence.layer_fields,
        cache_key=evidence.cache_key,
        corpus_sha256=evidence.corpus_sha256,
        evidence_sha256=evidence.evidence_sha256,
    )
    with pytest.raises(
        MODULE.Lfm2StructuralCompressionError,
        match="below required",
    ):
        MODULE.build_global_residual_map(
            low_retention,
            _small_candidate(),
        )


def test_global_objectives_are_frobenius_balanced_before_role_priority() -> None:
    parent = TinyLfm2(_config())
    evidence = _evidence(parent)
    scaled = MODULE.CachedFoldEvidence(
        residual_rows=evidence.residual_rows,
        observable_rows=evidence.observable_rows * 11.0,
        margin_rows=evidence.margin_rows * 0.125,
        margin_weights=evidence.margin_weights,
        layer_fields=evidence.layer_fields,
        cache_key=evidence.cache_key,
        corpus_sha256=evidence.corpus_sha256,
        evidence_sha256=evidence.evidence_sha256,
    )
    _, report = MODULE.build_global_residual_map(scaled, _small_candidate())
    objectives = report["objective_weighting"]["objectives"]
    residual_energy = objectives["residual"]["weighted_input_energy"]
    observable_energy = objectives["observable"]["weighted_input_energy"]
    margin_energy = objectives["teacher_margin"]["weighted_input_energy"]

    assert observable_energy == pytest.approx(residual_energy, rel=1e-12)
    assert margin_energy == pytest.approx(residual_energy, rel=1e-12)


def test_dense_linear_bias_and_depthwise_transports_match_basis_formulas() -> None:
    parent = TinyLfm2(_config())
    mapping, _ = MODULE.build_global_residual_map(
        _evidence(parent),
        _small_candidate(),
    )
    generator = torch.Generator().manual_seed(903)
    weight = torch.randn(8, 8, generator=generator)
    bias = torch.randn(8, generator=generator)
    kernel = torch.randn(8, 3, generator=generator)
    basis = mapping.basis.to(dtype=torch.float32)

    transported_weight = MODULE._transport_square(mapping, weight)
    transported_bias = MODULE._map_vector_producer(mapping, bias)
    transported_kernel = MODULE._transport_depthwise_kernel(mapping, kernel)

    assert torch.allclose(
        transported_weight,
        basis.T @ weight @ basis,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        transported_bias,
        basis.T @ bias,
        atol=1e-6,
        rtol=1e-6,
    )
    expected_taps = torch.stack(
        tuple(
            torch.diagonal(basis.T @ torch.diag(kernel[:, tap]) @ basis)
            for tap in range(kernel.shape[1])
        ),
        dim=1,
    )
    assert torch.allclose(
        transported_kernel,
        expected_taps,
        atol=1e-6,
        rtol=1e-6,
    )


def test_end_to_end_fold_only_overwrites_prebuilt_child_without_topology_change() -> None:
    torch.manual_seed(101)
    parent = TinyLfm2(_config())
    parent_before = MODULE.model_state_sha256(parent)
    candidate = _small_candidate()
    evidence = _evidence(parent)
    first = MODULE.instantiate_compiler_blank(
        parent,
        candidate,
        seed=121,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second = MODULE.instantiate_compiler_blank(
        parent,
        candidate,
        seed=909,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    first_report = MODULE.fold_lfm2_into_blank(
        parent,
        first,
        evidence,
        candidate,
        global_guidance={"causal_anchor_neurons_by_layer": {}},
    )
    second_report = MODULE.fold_lfm2_into_blank(
        parent,
        second,
        evidence,
        candidate,
        global_guidance={"causal_anchor_neurons_by_layer": {}},
    )
    assert first_report["folded_state_sha256"] == second_report["folded_state_sha256"]
    assert first_report["complete_target_parameter_overwrite"] is True
    assert first_report["parameter_objects_shapes_dtypes_unchanged"] is True
    assert first_report["source_coverage"]["source_coverage_fraction"] == 1.0
    assert first_report["source_coverage"]["dropped_source_units"] == 0
    assert all(
        layer["mlp"]["report"]["negative_coefficient_count"] == 0
        for layer in first_report["layers"]
    )
    assert MODULE.model_state_sha256(parent) == parent_before
    assert MODULE.dense_logical_parameter_audit(first.model)[
        "unique_logical_parameters"
    ] == candidate.verified_unique_parameters


def test_runner_has_no_training_hub_or_dtype_compression_lane() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.append(node.func.id)
    assert "from_pretrained" not in calls
    assert "backward" not in calls
    assert "step" not in calls
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "bitsandbytes" not in source
    assert "optimizer_steps\": 0" in source
    parser = MODULE.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


class _FakeTraceCache:
    def __init__(
        self,
        parent: TinyLfm2,
        *,
        batch_count: int = 4,
        decode_phase_counts=None,
        include_logits: bool = False,
    ) -> None:
        from neural_forge.hf_lfm2_circuit_trace_cache import (
            fingerprint_lfm2_parent,
        )

        self.cache_key = "1" * 64
        self.corpus_fingerprint = "2" * 64
        self.path = None
        self.manifest = {
            "cache_key": self.cache_key,
            "manifest_sha256": "3" * 64,
        }
        self.parent_fingerprint = fingerprint_lfm2_parent(
            parent
        ).identity_sha256
        self.batch_count = batch_count
        self.verify_calls = 0
        if decode_phase_counts is None:
            decode_phase_counts = (1,) * batch_count
        decode_phase_counts = tuple(decode_phase_counts)
        if len(decode_phase_counts) != batch_count:
            raise ValueError("decode_phase_counts must match batch_count")
        generator = torch.Generator().manual_seed(20260717)
        batches = []
        for batch_index in range(batch_count):
            tensors = {}
            phases = ("prefill",) + tuple(
                f"decode.{index:03d}"
                for index in range(decode_phase_counts[batch_index])
            )
            for phase in phases:
                for layer_index in range(14):
                    for endpoint in (
                        "pre",
                        "post_operator",
                        "post_layer",
                    ):
                        name = f"{phase}.layer.{layer_index:03d}.residual.{endpoint}"
                        tensors[name] = torch.randn(
                            1,
                            3,
                            parent.config.hidden_size,
                            generator=generator,
                        )
                if include_logits:
                    final_residual = tensors[
                        f"{phase}.layer.013.residual.post_layer"
                    ]
                    normalized = (
                        final_residual.to(dtype=torch.float64)
                        * parent.model.embedding_norm.weight.detach()
                        .to(dtype=torch.float64)
                    )
                    tensors[f"{phase}.logits"] = (
                        normalized
                        @ parent.lm_head.weight.detach()
                        .to(dtype=torch.float64)
                        .T
                    ).contiguous()
            batches.append(
                SimpleNamespace(
                    index=batch_index,
                    input_sha256=hashlib.sha256(
                        f"balanced-cache-batch-{batch_index}".encode("ascii")
                    ).hexdigest(),
                    tensors=MappingProxyType(tensors),
                )
            )
        self._batches = tuple(batches)

    def verify(self):
        self.verify_calls += 1
        return self

    def iter_batches(self):
        return iter(self._batches)


class _StreamingPayload:
    def __init__(
        self,
        *,
        index,
        record_id,
        tensors,
        session_sha256=None,
        row_indices=None,
        source_logical_bytes=None,
        maximum_materialized_source_rows=0,
        safetensors_get_slice_calls=0,
        coalesced_slice_count=0,
    ):
        self.batch_index = index
        self.input_sha256 = record_id
        self.ordered_tensor_names = tuple(sorted(tensors))
        self.logical_bytes = sum(
            int(value.numel() * value.element_size())
            for value in tensors.values()
        )
        self.row_indices = MappingProxyType(dict(row_indices or {}))
        self.source_logical_bytes = (
            self.logical_bytes
            if source_logical_bytes is None
            else int(source_logical_bytes)
        )
        self.maximum_materialized_source_rows = int(
            maximum_materialized_source_rows
        )
        self.safetensors_get_slice_calls = int(safetensors_get_slice_calls)
        self.coalesced_slice_count = int(coalesced_slice_count)
        self.session_sha256 = session_sha256
        self.selection_sha256 = hashlib.sha256(
            (
                str(index)
                + "\0"
                + record_id
                + "\0"
                + "\0".join(self.ordered_tensor_names)
            ).encode("ascii")
        ).hexdigest()
        self.tensors = MappingProxyType(dict(tensors))
        self.released = False


class _FakeSelectiveSession:
    def __init__(self, batches, *, trace_cache=None):
        from neural_forge.dynamic_circuit_ir import canonical_sha256
        from neural_forge.hf_lfm2_circuit_trace_cache import (
            SELECTIVE_AUTHENTICATED_SESSION_SCHEMA,
        )
        from neural_forge.hf_lfm2_streaming_vocabulary_margin import (
            _authenticated_trace_tensor_sha256,
        )

        self._batches = tuple(batches)
        references = []
        for batch in self._batches:
            inventory = {
                name: {
                    "dtype": str(value.dtype).replace("torch.", ""),
                    "shape": [int(size) for size in value.shape],
                    "numel": int(value.numel()),
                    "size_bytes": int(value.numel() * value.element_size()),
                    "sha256": _authenticated_trace_tensor_sha256(value),
                }
                for name, value in batch.tensors.items()
            }
            phases = tuple(
                sorted(
                    {
                        name.split(".layer.", 1)[0]
                        for name in batch.tensors
                        if ".layer." in name
                        and not name.startswith("cached_prefill.")
                    },
                    key=lambda value: (
                        0 if value == "prefill" else 1,
                        value,
                    ),
                )
            )
            record_sha256 = hashlib.sha256(
                f"record-{batch.index}".encode("ascii")
            ).hexdigest()
            logical_bytes = sum(
                int(value.numel() * value.element_size())
                for value in batch.tensors.values()
            )
            references.append(
                SimpleNamespace(
                    index=batch.index,
                    input_sha256=batch.input_sha256,
                    ordered_tensor_names=tuple(sorted(batch.tensors)),
                    record={
                        "tensors": inventory,
                        "capture": {
                            "phase_diagnostics": {
                                phase: {} for phase in phases
                            }
                        },
                    },
                    record_sha256=record_sha256,
                    tensor_count=len(inventory),
                    logical_bytes=logical_bytes,
                )
            )
        self.batch_references = tuple(references)
        self.load_calls = 0
        self.whole_tensor_load_calls = 0
        self.row_load_calls = 0
        self.release_calls = 0
        self.row_release_calls = 0
        self.maximum_loaded_payload_rows = 0
        self.maximum_loaded_payload_bytes = 0
        self.path = getattr(trace_cache, "path", None)
        self.cache_key = getattr(trace_cache, "cache_key", None)
        self.batch_count = len(self._batches)
        self.manifest = getattr(trace_cache, "manifest", None)
        self.verify_calls = 0
        receipt_body = {
            "schema": SELECTIVE_AUTHENTICATED_SESSION_SCHEMA,
            "cache_key": self.cache_key,
            "manifest_sha256": (
                None
                if self.manifest is None
                else self.manifest["manifest_sha256"]
            ),
            "completion_sha256": "4" * 64,
            "batch_count": self.batch_count,
            "ordered_authenticated_batches": [
                {
                    "batch_index": reference.index,
                    "input_sha256": reference.input_sha256,
                    "trace_file_sha256": hashlib.sha256(
                        f"trace-{reference.index}".encode("ascii")
                    ).hexdigest(),
                    "record_file_sha256": hashlib.sha256(
                        f"record-file-{reference.index}".encode("ascii")
                    ).hexdigest(),
                    "record_sha256": hashlib.sha256(
                        f"record-{reference.index}".encode("ascii")
                    ).hexdigest(),
                    "tensor_count": reference.tensor_count,
                    "logical_bytes": reference.logical_bytes,
                }
                for reference in references
            ],
            "authentication_contract": {
                "manifest_completion_and_layout_verified": True,
                "trace_file_sha256_passes_per_session": 1,
                "record_file_sha256_passes_per_session": 1,
                "record_self_hash_verified": True,
                "safetensors_metadata_verified": True,
                "complete_tensor_key_inventory_verified": True,
                "selected_tensor_descriptors_verified_per_load": True,
                "unselected_tensor_payloads_loaded": False,
                "parent_forward_calls": 0,
                "training_steps": 0,
            },
        }
        self._receipt = {
            **receipt_body,
            "session_sha256": canonical_sha256(receipt_body),
        }

    @property
    def evidence_sha256(self):
        return self._receipt["session_sha256"]

    def verify(self):
        self.verify_calls += 1
        return self

    def as_dict(self):
        return copy.deepcopy(self._receipt)

    def load_selected_tensors(self, index, tensor_names):
        self.load_calls += 1
        self.whole_tensor_load_calls += 1
        batch = self._batches[index]
        tensors = {
            name: batch.tensors[name].clone()
            for name in tensor_names
        }
        self.maximum_loaded_payload_rows = max(
            self.maximum_loaded_payload_rows,
            sum(
                int(value.numel() // value.shape[-1])
                for value in tensors.values()
            ),
        )
        return _StreamingPayload(
            index=index,
            record_id=batch.input_sha256,
            tensors=tensors,
            session_sha256=self.evidence_sha256,
        )

    def load_selected_tensor_rows(self, index, row_indices):
        self.load_calls += 1
        self.row_load_calls += 1
        batch = self._batches[index]
        tensors = {}
        source_logical_bytes = 0
        maximum_materialized_source_rows = 0
        for name in sorted(row_indices):
            source = batch.tensors[name]
            indices = tuple(row_indices[name])
            rows = source.reshape(-1, source.shape[-1]).index_select(
                0,
                torch.tensor(indices, dtype=torch.long),
            ).contiguous()
            tensors[name] = rows
            source_logical_bytes += int(source.numel() * source.element_size())
            maximum_materialized_source_rows = max(
                maximum_materialized_source_rows,
                len(indices),
            )
        payload_rows = sum(
            int(value.numel() // value.shape[-1])
            for value in tensors.values()
        )
        payload_bytes = sum(
            int(value.numel() * value.element_size())
            for value in tensors.values()
        )
        self.maximum_loaded_payload_rows = max(
            self.maximum_loaded_payload_rows,
            payload_rows,
        )
        self.maximum_loaded_payload_bytes = max(
            self.maximum_loaded_payload_bytes,
            payload_bytes,
        )
        return _StreamingPayload(
            index=index,
            record_id=batch.input_sha256,
            tensors=tensors,
            session_sha256=self.evidence_sha256,
            row_indices=row_indices,
            source_logical_bytes=source_logical_bytes,
            maximum_materialized_source_rows=(
                maximum_materialized_source_rows
            ),
            safetensors_get_slice_calls=len(tensors),
            coalesced_slice_count=sum(len(indices) for indices in row_indices.values()),
        )

    def release_selected_tensors(self, payload):
        payload.tensors = MappingProxyType({})
        payload.released = True
        self.release_calls += 1

    def release_selected_tensor_rows(self, payload):
        payload.tensors = MappingProxyType({})
        payload.released = True
        self.release_calls += 1
        self.row_release_calls += 1


def _streaming_cache_fixture(
    *,
    row_scale: int = 1,
    decode_phase_counts=(2, 1, 3, 2),
    source_dtype=torch.float32,
):
    width = 4
    layer_count = 2
    batches = []
    for batch_index, decode_count in enumerate(decode_phase_counts):
        tensors = {}
        phases = ("prefill",) + tuple(
            f"decode.{index:03d}" for index in range(decode_count)
        )
        for phase_index, phase in enumerate(phases):
            row_count = row_scale * (7 + batch_index + phase_index)
            for layer_index in range(layer_count):
                for endpoint_index, endpoint in enumerate(
                    ("pre", "post_operator", "post_layer")
                ):
                    name = (
                        f"{phase}.layer.{layer_index:03d}."
                        f"residual.{endpoint}"
                    )
                    offset = (
                        batch_index * 1_000
                        + phase_index * 100
                        + layer_index * 10
                        + endpoint_index
                    )
                    tensors[name] = (
                        torch.arange(
                            row_count * width,
                            dtype=torch.float32,
                        ).reshape(1, row_count, width).to(source_dtype)
                        + offset
                    )
        batches.append(
            SimpleNamespace(
                index=batch_index,
                input_sha256=hashlib.sha256(
                    f"streaming-record-{batch_index}".encode("ascii")
                ).hexdigest(),
                tensors=MappingProxyType(tensors),
            )
        )
    cache = SimpleNamespace(iter_batches=lambda: iter(tuple(batches)))
    return cache, _FakeSelectiveSession(batches), width, layer_count


@pytest.mark.parametrize(
    "source_dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
def test_streaming_global_row_selection_is_byte_identical_to_legacy(
    source_dtype,
    monkeypatch,
) -> None:
    from neural_forge import hf_lfm2_balanced_circuit_basis as balanced
    from neural_forge.dynamic_balanced_circuit_basis import _snapshot_gram

    cache, session, width, layer_count = _streaming_cache_fixture(
        source_dtype=source_dtype,
    )
    cache_key = "1" * 64
    validation_fraction = 0.25
    split_seed = "streaming-equivalence-v1"
    maximum = 5

    legacy_records = balanced._load_cache_records(
        cache,
        source_width=width,
        layer_count=layer_count,
        vocabulary_size=19,
        vocabulary_margin_top_k=None,
    )
    legacy_fit, legacy_heldout, legacy_split = balanced._split_records(
        legacy_records,
        cache_key=cache_key,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
    )
    legacy_fit_snapshots = balanced._snapshot_blocks(
        legacy_fit,
        split=balanced.EvidenceSplit.FIT,
        max_rows_per_block=maximum,
    )
    legacy_heldout_snapshots = balanced._snapshot_blocks(
        legacy_heldout,
        split=balanced.EvidenceSplit.HELDOUT,
        max_rows_per_block=maximum,
    )

    progress = []
    monkeypatch.setattr(
        balanced,
        "_emit_streaming_row_selection_progress",
        lambda event: progress.append(dict(event)),
    )
    streamed = balanced._stream_selective_cache_snapshot_grams(
        session,
        source_width=width,
        layer_count=layer_count,
        cache_key=cache_key,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        max_rows_per_block=maximum,
    )

    assert dict(streamed.split_report) == legacy_split
    assert session.load_calls > 0
    assert session.release_calls == session.load_calls
    assert session.row_load_calls == session.load_calls
    assert session.row_release_calls == session.row_load_calls
    assert session.whole_tensor_load_calls == 0
    assert len(progress) >= session.load_calls
    assert all(
        event["authenticated_row_slice_path"] is True
        and event["whole_tensor_cpu_clone_preceded_row_selection"] is False
        and event["payload_released"] is True
        and event["selected_logical_bytes"] <= event["source_logical_bytes"]
        and event["retained_fp64_logical_bytes"]
        >= event["selected_row_count"] * width * 8
        and event["safetensors_get_slice_calls"] == event["selected_tensor_count"]
        and event["coalesced_slice_count"] >= event["selected_tensor_count"]
        and event["telemetry_excluded_from_basis_evidence_hashes"] is True
        and event["phase_windowed_fp64_retention"] is True
        and event["all_snapshot_blocks_retained"] is False
        for event in progress
    )
    for split, expected, actual in (
        (
            balanced.EvidenceSplit.FIT,
            legacy_fit_snapshots,
            streamed.fit_snapshot_gram,
        ),
        (
            balanced.EvidenceSplit.HELDOUT,
            legacy_heldout_snapshots,
            streamed.heldout_snapshot_gram,
        ),
    ):
        expected_gram = _snapshot_gram(expected, split=split)
        assert torch.equal(actual.raw, expected_gram.raw)
        assert torch.equal(actual.balanced, expected_gram.balanced)
        assert actual.as_dict()["report"] == dict(expected_gram.report)


def test_lfm2_balanced_builder_streaming_result_is_exact_legacy_result(
    monkeypatch,
) -> None:
    from neural_forge import hf_lfm2_balanced_circuit_basis as balanced

    torch.manual_seed(321)
    parent = TinyLfm2(_config())
    parent_forward_calls = 0

    def forbidden_parent_forward(*_args, **_kwargs):
        nonlocal parent_forward_calls
        parent_forward_calls += 1
        raise AssertionError("balanced row slicing invoked the parent model")

    parent.forward = forbidden_parent_forward
    cache = _FakeTraceCache(
        parent,
        batch_count=4,
        decode_phase_counts=(1, 3, 2, 4),
    )
    legacy = balanced.build_lfm2_circuit_balanced_basis(
        parent,
        cache,
        4,
        validation_fraction=0.25,
        split_seed="builder-streaming-equivalence-v1",
        max_rows_per_block=5,
    )
    session = _FakeSelectiveSession(cache._batches, trace_cache=cache)
    monkeypatch.setattr(
        balanced,
        "Lfm2CircuitTraceSelectiveAuthenticatedSession",
        _FakeSelectiveSession,
    )

    def forbidden_legacy_loader(*_args, **_kwargs):
        raise AssertionError("streaming branch invoked the legacy full loader")

    monkeypatch.setattr(
        balanced,
        "_load_cache_records",
        forbidden_legacy_loader,
    )
    streamed = balanced.build_lfm2_circuit_balanced_basis(
        parent,
        cache,
        4,
        selective_session=session,
        validation_fraction=0.25,
        split_seed="builder-streaming-equivalence-v1",
        max_rows_per_block=5,
    )

    assert session.verify_calls == 2
    assert session.release_calls == session.load_calls
    assert session.row_load_calls > 0
    assert session.whole_tensor_load_calls == 0
    assert parent_forward_calls == 0
    assert streamed.as_dict() == legacy.as_dict()
    streamed_report = streamed.as_dict()
    assert streamed_report["model_forward_calls"] == 0
    assert streamed_report["parent_trace_regeneration_calls"] == 0
    assert streamed_report["training_steps"] == 0
    assert streamed_report["backward_calls"] == 0
    assert streamed_report["optimizer_steps"] == 0
    assert streamed_report["autograd_used"] is False
    for name in (
        "basis",
        "balanced_encoder",
        "balanced_decoder",
        "full_weight_gram",
        "controllability_gram",
        "observability_gram",
        "heldout_gram",
    ):
        assert torch.equal(getattr(streamed, name), getattr(legacy, name))


def test_streaming_retained_rows_scale_with_cap_not_source_rows() -> None:
    from neural_forge import hf_lfm2_balanced_circuit_basis as balanced

    cache_key = "3" * 64
    common = {
        "source_width": 4,
        "layer_count": 2,
        "cache_key": cache_key,
        "validation_fraction": 0.25,
        "split_seed": "streaming-memory-bound-v1",
        "max_rows_per_block": 7,
    }
    _small_cache, small_session, _width, _layers = (
        _streaming_cache_fixture(
            row_scale=1,
            decode_phase_counts=(2, 2, 2, 2),
        )
    )
    _large_cache, large_session, _width, _layers = (
        _streaming_cache_fixture(
            row_scale=10,
            decode_phase_counts=(2, 2, 2, 2),
        )
    )

    small = balanced._stream_selective_cache_snapshot_grams(
        small_session,
        **common,
    )
    large = balanced._stream_selective_cache_snapshot_grams(
        large_session,
        **common,
    )

    expected_blocks = 2 * 3 * 2 * 3
    expected_retained_rows = expected_blocks * common["max_rows_per_block"]
    assert small.selected_snapshot_row_count == expected_retained_rows
    assert large.selected_snapshot_row_count == expected_retained_rows
    assert large.source_row_count == 10 * small.source_row_count
    assert all(
        not hasattr(record, "locations")
        for record in (*small.records, *large.records)
    )
    assert large.selected_snapshot_row_count < large.source_row_count
    maximum_phase_blocks = 2 * 3
    assert small.maximum_resident_fp64_snapshot_rows <= (
        (maximum_phase_blocks + 3) * common["max_rows_per_block"]
    )
    assert large.maximum_resident_fp64_snapshot_rows == (
        small.maximum_resident_fp64_snapshot_rows
    )
    assert large_session.maximum_loaded_payload_rows < large.source_row_count
    assert small_session.maximum_loaded_payload_rows == (
        large_session.maximum_loaded_payload_rows
    )
    assert small_session.maximum_loaded_payload_bytes == (
        large_session.maximum_loaded_payload_bytes
    )
    assert small_session.whole_tensor_load_calls == 0
    assert large_session.whole_tensor_load_calls == 0
    memory = dict(large.memory_bound_report)
    assert memory["residual_endpoint_count_per_layer"] == 3
    assert memory["residual_endpoints"] == [
        "residual.pre",
        "residual.post_operator",
        "residual.post_layer",
    ]
    assert memory["block_owned_clone_count"] == 1
    assert memory["block_hash_clone_count"] == 1
    assert memory["block_hash_tobytes_copy_count"] == 1
    assert memory["phase_and_source_read_bound_bytes"] == (
        memory["maximum_phase_buffer_fp64_bytes"]
        + memory["maximum_selected_source_payload_bytes"]
        + memory["maximum_source_slice_or_hash_overhead_bytes"]
    )
    assert memory["phase_and_conversion_bound_bytes"] == (
        memory["maximum_phase_buffer_fp64_bytes"]
        + memory["maximum_selected_source_payload_bytes"]
        + memory["maximum_fp64_conversion_bytes"]
        + memory["maximum_fp64_finiteness_scratch_bytes"]
    )
    assert memory["phase_and_block_seal_bound_bytes"] == (
        memory["maximum_phase_buffer_fp64_bytes"]
        + memory["block_owned_clone_bytes"]
        + memory["block_hash_clone_bytes"]
        + memory["block_hash_tobytes_copy_bytes"]
    )
    assert memory["maximum_row_path_bound_bytes"] == max(
        memory["phase_and_source_read_bound_bytes"],
        memory["phase_and_conversion_bound_bytes"],
        memory["phase_and_block_seal_bound_bytes"],
    )
    assert memory["maximum_concurrent_gram_matrix_count"] == 8
    assert memory["gram_allocation_bound_bytes"] == (
        memory["gram_matrix_fp64_bytes"]
        * memory["maximum_concurrent_gram_matrix_count"]
    )
    assert memory["known_python_overhead_bound_bytes"] == (
        memory["python_plan_overhead_bound_bytes"]
        + memory["python_metadata_overhead_bound_bytes"]
        + memory["python_report_overhead_bound_bytes"]
    )
    assert memory["python_plan_overhead_bound_bytes"] == (
        memory["python_plan_single_copy_overhead_bytes"]
        * memory["python_plan_live_copy_count_bound"]
    )
    assert memory["python_report_overhead_bound_bytes"] == (
        memory["python_report_single_copy_overhead_bytes"]
        * memory["python_report_live_copy_count_bound"]
    )
    assert memory["maximum_resident_snapshot_logical_bytes"] == (
        memory["maximum_row_path_bound_bytes"]
        + memory["gram_allocation_bound_bytes"]
        + memory["known_python_overhead_bound_bytes"]
    )
    assert large.maximum_resident_snapshot_logical_bytes == memory[
        "maximum_resident_snapshot_logical_bytes"
    ]
    assert memory["source_payload_included"] is True
    assert memory["block_clone_hash_and_tobytes_copies_included"] is True
    assert memory["precomputed_fit_gram_residency_during_heldout_included"] is True
    assert memory["known_python_container_overhead_included"] is True
    assert memory["allocator_fragmentation_and_native_runtime_overhead_included"] is False
    assert memory["telemetry_excluded_from_basis_evidence_hashes"] is True


def test_streaming_row_payload_releases_when_conversion_fails(
    monkeypatch,
) -> None:
    from neural_forge import hf_lfm2_balanced_circuit_basis as balanced

    _cache, session, width, layer_count = _streaming_cache_fixture()

    def fail_conversion(*_args, **_kwargs):
        raise RuntimeError("injected selected-row conversion failure")

    monkeypatch.setattr(
        balanced,
        "_preselected_residual_rows",
        fail_conversion,
    )
    with pytest.raises(
        RuntimeError,
        match="injected selected-row conversion failure",
    ):
        balanced._stream_selective_cache_snapshot_grams(
            session,
            source_width=width,
            layer_count=layer_count,
            cache_key="8" * 64,
            validation_fraction=0.25,
            split_seed="streaming-release-on-error-v1",
            max_rows_per_block=5,
        )
    assert session.row_load_calls == 1
    assert session.row_release_calls == 1
    assert session.release_calls == session.load_calls == 1
    assert session.whole_tensor_load_calls == 0


def test_lfm2_circuit_balanced_adapter_uses_cache_weights_and_ir_without_forward() -> None:
    from neural_forge.hf_lfm2_balanced_circuit_basis import (
        build_lfm2_circuit_balanced_basis,
    )

    torch.manual_seed(321)
    parent = TinyLfm2(_config())
    cache = _FakeTraceCache(parent)
    forward_calls = 0

    def forbidden_forward(*_args, **_kwargs):
        nonlocal forward_calls
        forward_calls += 1
        raise AssertionError("the circuit-balanced adapter invoked the parent")

    parent.forward = forbidden_forward
    result = build_lfm2_circuit_balanced_basis(
        parent,
        cache,
        4,
        validation_fraction=0.25,
        max_rows_per_block=8,
    )

    assert forward_calls == 0
    assert cache.verify_calls == 1
    assert result.source_width == 8
    assert result.target_width == 4
    assert torch.allclose(
        result.basis.T @ result.basis,
        torch.eye(4, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-11,
    )
    report = result.as_dict()
    assert report["split"]["fit_record_count"] == 3
    assert report["split"]["heldout_record_count"] == 1
    assert report["split"]["record_ids_disjoint"] is True
    assert report["snapshot_adapter"]["fit_block_count"] == 84
    assert report["snapshot_adapter"]["heldout_block_count"] == 84
    assert (
        report["balanced_basis"]["circuit_ir"][
            "all_writer_nodes_covered"
        ]
        is True
    )
    assert (
        report["balanced_basis"]["circuit_ir"][
            "all_output_interfaces_covered"
        ]
        is True
    )
    assert report["model_forward_calls"] == 0
    assert report["parent_trace_regeneration_calls"] == 0
    assert "vocabulary_margin_observability" not in report
    assert result.map_sha256 == report["hashes"]["basis_sha256"]


def test_lfm2_vocabulary_margin_observability_is_explicit_and_cache_only() -> None:
    from neural_forge.hf_lfm2_balanced_circuit_basis import (
        build_lfm2_circuit_balanced_basis,
    )

    torch.manual_seed(921)
    parent = TinyLfm2(_config())
    cache = _FakeTraceCache(parent, include_logits=True)
    forward_calls = 0

    def forbidden_forward(*_args, **_kwargs):
        nonlocal forward_calls
        forward_calls += 1
        raise AssertionError("vocabulary margins invoked the parent")

    parent.forward = forbidden_forward
    result = build_lfm2_circuit_balanced_basis(
        parent,
        cache,
        4,
        validation_fraction=0.25,
        max_rows_per_block=8,
        vocabulary_margin_observability=True,
        vocabulary_margin_top_k=3,
    )

    assert forward_calls == 0
    report = result.as_dict()
    margin = report["vocabulary_margin_observability"]
    diagnostics = margin["heldout_diagnostics"]
    observable = report["balanced_basis"]["downstream_observability"]

    assert margin["enabled"] is True
    assert margin["observability_role"] == "vocabulary_margin_geometry"
    assert margin["fit_used_for_selection"] is True
    assert margin["heldout_used_for_selection"] is False
    assert margin["full_parent_logits_read_from_sealed_cache"] is True
    assert margin["parent_forward_calls"] == 0
    assert margin["fit"]["phase_count"] == 6
    assert margin["heldout"]["phase_count"] == 2
    assert margin["fit"]["embedding_row_differences_used"] is True
    assert margin["fit"]["weighted_centering"] is True
    assert margin["fit"]["weighted_mean_reinserted"] is True
    assert margin["fit"]["gram_trace_energy"] == pytest.approx(
        1.0,
        abs=1e-12,
    )
    assert "vocabulary_margin_geometry" in observable["role_order"]
    assert "vocabulary_margin_geometry" in report["balanced_basis"][
        "retention"
    ]["observability_roles"]
    assert 0.0 <= diagnostics[
        "retained_top_k_margin_direction_energy_fraction"
    ] <= 1.0 + 1e-12
    assert diagnostics[
        "closed_form_unprojected_vs_cached_sign_agreement_fraction"
    ] == pytest.approx(1.0, abs=1e-12)
    assert 0.0 <= diagnostics[
        "projected_top1_vs_top_competitor_nonnegative_fraction"
    ] <= 1.0
    assert 0.0 <= diagnostics[
        "projected_top1_vs_all_top_k_nonnegative_fraction"
    ] <= 1.0
    assert diagnostics[
        "projection_diagnostics_require_parent_execution"
    ] is False
    projection_gate = result.evaluate_vocabulary_margin_projection_gate(
        result.basis,
        minimum_retained_energy_fraction=0.0,
        minimum_sign_agreement_fraction=0.0,
        minimum_top_competitor_nonnegative_fraction=0.0,
        minimum_all_top_k_nonnegative_fraction=0.0,
    )
    assert projection_gate["gate_passed"] is True
    assert projection_gate["candidate_forward_calls"] == 0
    candidate_gate = result.evaluate_vocabulary_margin_candidate_gate(
        cache,
        minimum_sign_agreement_fraction=1.0,
        minimum_top_competitor_nonnegative_fraction=1.0,
        minimum_all_top_k_nonnegative_fraction=1.0,
        minimum_exact_top1_agreement_fraction=1.0,
    )
    assert candidate_gate["gate_passed"] is True
    assert candidate_gate["metrics"][
        "candidate_exact_parent_top1_agreement_fraction"
    ] == pytest.approx(1.0, abs=1e-12)
    assert candidate_gate["parent_forward_calls"] == 0
    assert candidate_gate["candidate_forward_calls"] == 0
    assert report["model_forward_calls"] == 0
    assert report["training_steps"] == 0


def test_streaming_vocabulary_margin_v2_matches_legacy_decisions_without_rows(
    monkeypatch,
) -> None:
    from neural_forge import hf_lfm2_balanced_circuit_basis as balanced

    torch.manual_seed(921)
    parent = TinyLfm2(_config())
    cache = _FakeTraceCache(
        parent,
        include_logits=True,
        decode_phase_counts=(2, 3, 1, 4),
    )
    legacy = balanced.build_lfm2_circuit_balanced_basis(
        parent,
        cache,
        4,
        validation_fraction=0.25,
        split_seed="streaming-vocabulary-margin-v2-equivalence",
        max_rows_per_block=8,
        vocabulary_margin_observability=True,
        vocabulary_margin_top_k=3,
    )
    session = _FakeSelectiveSession(cache._batches, trace_cache=cache)
    monkeypatch.setattr(
        balanced,
        "Lfm2CircuitTraceSelectiveAuthenticatedSession",
        _FakeSelectiveSession,
    )
    streamed = balanced.build_lfm2_circuit_balanced_basis(
        parent,
        cache,
        4,
        selective_session=session,
        validation_fraction=0.25,
        split_seed="streaming-vocabulary-margin-v2-equivalence",
        max_rows_per_block=8,
        vocabulary_margin_observability=True,
        vocabulary_margin_top_k=3,
        streaming_vocabulary_margin_v2=True,
        streaming_vocabulary_margin_pair_chunk_size=5,
    )

    assert torch.allclose(
        streamed.observability_gram,
        legacy.observability_gram,
        rtol=2e-11,
        atol=2e-12,
    )
    assert torch.allclose(
        streamed.basis @ streamed.basis.T,
        legacy.basis @ legacy.basis.T,
        rtol=2e-9,
        atol=2e-10,
    )
    probe = torch.linalg.qr(
        torch.randn(
            8,
            4,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(601),
        )
    ).Q
    legacy_projection = legacy.evaluate_vocabulary_margin_projection_gate(
        probe,
        minimum_retained_energy_fraction=0.0,
        minimum_sign_agreement_fraction=0.0,
        minimum_top_competitor_nonnegative_fraction=0.0,
        minimum_all_top_k_nonnegative_fraction=0.0,
    )
    streamed_projection = streamed.evaluate_vocabulary_margin_projection_gate(
        probe,
        minimum_retained_energy_fraction=0.0,
        minimum_sign_agreement_fraction=0.0,
        minimum_top_competitor_nonnegative_fraction=0.0,
        minimum_all_top_k_nonnegative_fraction=0.0,
    )
    for name in (
        "retained_top_k_margin_direction_energy_fraction",
        "cached_top1_margin_nonnegative_fraction",
        "closed_form_unprojected_vs_cached_sign_agreement_fraction",
        "projected_vs_cached_sign_agreement_fraction",
        "projected_top1_vs_top_competitor_nonnegative_fraction",
        "projected_top1_vs_all_top_k_nonnegative_fraction",
        "projected_margin_minimum",
        "projected_margin_maximum",
    ):
        assert streamed_projection["metrics"][name] == pytest.approx(
            legacy_projection["metrics"][name],
            abs=2e-12,
        )
    assert (
        streamed_projection["gate_passed"] is legacy_projection["gate_passed"]
    )

    legacy_candidate = legacy.evaluate_vocabulary_margin_candidate_gate(
        cache,
        minimum_sign_agreement_fraction=1.0,
        minimum_top_competitor_nonnegative_fraction=1.0,
        minimum_all_top_k_nonnegative_fraction=1.0,
        minimum_exact_top1_agreement_fraction=1.0,
    )
    streamed_candidate = streamed.evaluate_vocabulary_margin_candidate_gate(
        cache,
        minimum_sign_agreement_fraction=1.0,
        minimum_top_competitor_nonnegative_fraction=1.0,
        minimum_all_top_k_nonnegative_fraction=1.0,
        minimum_exact_top1_agreement_fraction=1.0,
    )
    for name in (
        "candidate_vs_cached_sign_agreement_fraction",
        "candidate_top1_vs_top_competitor_nonnegative_fraction",
        "candidate_top1_vs_all_top_k_nonnegative_fraction",
        "candidate_exact_parent_top1_agreement_fraction",
        "candidate_margin_minimum",
        "candidate_margin_maximum",
    ):
        assert streamed_candidate["metrics"][name] == pytest.approx(
            legacy_candidate["metrics"][name],
            abs=1e-15,
        )
    assert streamed_candidate["gate_passed"] is legacy_candidate["gate_passed"]
    assert (
        streamed_candidate["candidate_selected_logits_sha256"]
        == (legacy_candidate["candidate_selected_logits_sha256"])
    )

    report = streamed.as_dict()["vocabulary_margin_observability"]
    adapter = report["adapter"]
    assert report["streaming_v2"] is True
    assert report["all_declared_fit_rows_consumed"] is True
    assert report["row_cap_or_subsample_applied"] is False
    assert (
        adapter["coverage_claims"]["full_sealed_cache_integrity_authenticated"]
        is True
    )
    assert (
        adapter["coverage_claims"][
            "vocabulary_margin_selected_tensor_subset_only"
        ]
        is True
    )
    assert (
        adapter["coverage_claims"][
            "selected_subset_is_not_claimed_as_all_cache_tensor_semantics"
        ]
        is True
    )
    assert adapter["retention"]["retained_full_direction_rows"] == 0
    assert adapter["retention"]["retained_full_residual_rows"] == 0
    assert session.release_calls == session.load_calls


def test_streaming_v2_projects_every_cached_row_into_strict_target_vocabulary(
    monkeypatch,
    tmp_path,
) -> None:
    from neural_forge import hf_lfm2_balanced_circuit_basis as balanced
    from neural_forge.dynamic_circuit_ir import canonical_sha256
    from neural_forge.hf_lfm2_streaming_vocabulary_margin_adapter import (
        Lfm2TargetTokenizerVocabularyMarginBinding,
    )
    from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (
        Lfm2TargetTokenizedRecord,
        build_lfm2_strict_tokenizer_bridge_trace_adapter,
    )
    from neural_forge.hf_token_align import (
        StrictTokenRowMapping,
        StrictTokenTransportProfile,
    )
    from neural_forge.hf_tokenizer_bridge import (
        STRICT_TOKENIZER_BRIDGE_SCHEMA,
        StrictTokenizerBridgeArtifact,
        TokenizerArtifactFile,
    )

    torch.manual_seed(923)
    parent = TinyLfm2(_config())
    cache = _FakeTraceCache(
        parent,
        include_logits=True,
        decode_phase_counts=(2, 3, 1, 4),
    )
    mapping = (7, 2, 7, 4, 0)
    bridge_root = tmp_path / "target-tokenizer"
    bridge_root.mkdir()
    tokenizer_bytes = b"{}\n"
    (bridge_root / "tokenizer.json").write_bytes(tokenizer_bytes)
    file_record = TokenizerArtifactFile(
        relative_path="tokenizer.json",
        size_bytes=len(tokenizer_bytes),
        sha256=hashlib.sha256(tokenizer_bytes).hexdigest(),
    )
    profile = StrictTokenTransportProfile(
        donor_tokenizer_class="TinyDonor",
        target_tokenizer_class="TinyTarget",
        donor_vocab_size=19,
        target_vocab_size=5,
        donor_row_capacity=19,
        target_row_capacity=5,
        alignment_fingerprint="7" * 64,
        row_mappings=tuple(
            StrictTokenRowMapping(
                target_row_index=index,
                donor_row_index=donor_index,
                resolution="exact-token",
            )
            for index, donor_index in enumerate(mapping)
        ),
    )
    bridge_body = {
        "schema": STRICT_TOKENIZER_BRIDGE_SCHEMA,
        "kind": "sealed-strict-target-tokenizer",
        "profile_fingerprint": profile.fingerprint(),
        "transport_profile": profile.to_dict(),
        "donor_tokenizer_semantics_sha256": "8" * 64,
        "target_tokenizer_semantics_sha256": "9" * 64,
        "target_files": [file_record.to_dict()],
        "model_parameter_operations": {
            "training_steps": 0,
            "backward_calls": 0,
            "pruned_parameters": 0,
            "quantized_parameters": 0,
            "random_target_rows": 0,
        },
    }
    bridge = StrictTokenizerBridgeArtifact(
        root_dir=bridge_root,
        transport_profile=profile,
        donor_tokenizer_semantics_sha256="8" * 64,
        target_tokenizer_semantics_sha256="9" * 64,
        target_files=(file_record,),
        artifact_sha256=canonical_sha256(bridge_body),
    )
    records = tuple(
        Lfm2TargetTokenizedRecord(
            record_id=f"batch-{index:06d}",
            split="fit" if index < 3 else "heldout",
            input_ids=torch.tensor(
                [[0, 1, (index + 1) % 5]],
                dtype=torch.long,
            ),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            forced_decode_ids=torch.tensor([[1]], dtype=torch.long),
        )
        for index in range(cache.batch_count)
    )
    trace_adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        bridge,
        records,
        expected_artifact_sha256=bridge.artifact_sha256,
    )
    for batch, record in zip(
        cache._batches,
        trace_adapter.records,
        strict=True,
    ):
        batch.input_sha256 = record.donor_cache_input_sha256
    session = _FakeSelectiveSession(cache._batches, trace_cache=cache)
    runtime_body = {
        "schema": "neural-forge-production-lfm2-tokenizer-bridge-runtime-v1",
        "enabled": True,
        "artifact_sha256": bridge.artifact_sha256,
        "transport_profile_sha256": bridge.profile_fingerprint,
        "parent_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "target_tokenizer_artifact_fingerprint": "a" * 64,
        "target_calibration_sha256": "b" * 64,
        "target_row_capacity": 5,
        "donor_row_capacity": 19,
        "trace_adapter_evidence_sha256": trace_adapter.evidence_sha256,
        "canonical_target_tokenization_evidence_sha256": "c" * 64,
        "paired_input_evidence_sha256": "d" * 64,
        "paired_record_count": cache.batch_count,
        "target_to_donor_row_indices_sha256": canonical_sha256(list(mapping)),
        "all_physical_target_rows_mapped": True,
        "mapping_injective_required": False,
        "repeated_donor_rows_preserve_distinct_target_rows": True,
        "numeric_token_id_equality_required": False,
        "donor_canonical_sequence_equivalence_claimed": False,
        "parent_cache_input_mode": (
            "target-tokenizer-canonical-then-one-row-mapped-to-donor"
        ),
        "parent_model_forwards": 0,
        "training_steps": 0,
    }
    binding = Lfm2TargetTokenizerVocabularyMarginBinding(
        trace_adapter=trace_adapter,
        runtime_receipt={
            **runtime_body,
            "runtime_receipt_sha256": canonical_sha256(runtime_body),
        },
    )
    monkeypatch.setattr(
        balanced,
        "Lfm2CircuitTraceSelectiveAuthenticatedSession",
        _FakeSelectiveSession,
    )

    result = balanced.build_lfm2_circuit_balanced_basis(
        parent,
        cache,
        4,
        selective_session=session,
        validation_fraction=0.25,
        split_seed="streaming-target-vocabulary-v2",
        max_rows_per_block=8,
        vocabulary_margin_observability=True,
        vocabulary_margin_top_k=3,
        streaming_vocabulary_margin_v2=True,
        streaming_vocabulary_margin_pair_chunk_size=5,
        target_tokenizer_vocabulary_margin_binding=binding,
    )

    adapter_report = result.as_dict()["vocabulary_margin_observability"][
        "adapter"
    ]
    domain = adapter_report["vocabulary_domain"]
    assert domain["mode"] == "strict-bridge-target-physical-vocabulary"
    assert domain["source_vocabulary_size"] == 19
    assert domain["target_vocabulary_size"] == 5
    assert domain["target_logit_projection"][
        "target_to_source_row_indices_sha256"
    ] == canonical_sha256(list(mapping))
    assert adapter_report["fit"]["all_declared_rows_consumed"] is True
    assert adapter_report["fit"]["statistics"][
        "cached_source_vocabulary_size"
    ] == 19
    assert adapter_report["fit"]["statistics"]["vocabulary_size"] == 5
    assert domain["candidate_numerical_gate_supported"] is False
    assert domain["manual_whole_response_review_remains_verdict"] is True
    with pytest.raises(ValueError, match="candidate replay gating remains disabled"):
        result.evaluate_vocabulary_margin_candidate_gate(
            cache,
            minimum_sign_agreement_fraction=0.0,
            minimum_top_competitor_nonnegative_fraction=0.0,
            minimum_all_top_k_nonnegative_fraction=0.0,
            minimum_exact_top1_agreement_fraction=0.0,
        )
    assert session.release_calls == session.load_calls


def test_streaming_vocabulary_margin_v2_fails_closed_on_descriptor_tamper(
    monkeypatch,
) -> None:
    from neural_forge import hf_lfm2_balanced_circuit_basis as balanced

    parent = TinyLfm2(_config())
    cache = _FakeTraceCache(parent, include_logits=True)
    session = _FakeSelectiveSession(cache._batches, trace_cache=cache)
    first = session.batch_references[0]
    record = copy.deepcopy(first.record)
    record["tensors"]["prefill.logits"]["sha256"] = "f" * 64
    first.record = record
    monkeypatch.setattr(
        balanced,
        "Lfm2CircuitTraceSelectiveAuthenticatedSession",
        _FakeSelectiveSession,
    )

    with pytest.raises(ValueError, match="not aligned"):
        balanced.build_lfm2_circuit_balanced_basis(
            parent,
            cache,
            4,
            selective_session=session,
            vocabulary_margin_observability=True,
            vocabulary_margin_top_k=3,
            streaming_vocabulary_margin_v2=True,
        )
    assert session.release_calls == session.load_calls


def test_streaming_vocabulary_margin_v2_fails_closed_on_session_tamper(
    monkeypatch,
) -> None:
    from neural_forge import hf_lfm2_balanced_circuit_basis as balanced

    parent = TinyLfm2(_config())
    cache = _FakeTraceCache(parent, include_logits=True)
    session = _FakeSelectiveSession(cache._batches, trace_cache=cache)
    session._receipt["completion_sha256"] = "e" * 64
    monkeypatch.setattr(
        balanced,
        "Lfm2CircuitTraceSelectiveAuthenticatedSession",
        _FakeSelectiveSession,
    )

    with pytest.raises(ValueError, match="session seal is invalid"):
        balanced.build_lfm2_circuit_balanced_basis(
            parent,
            cache,
            4,
            selective_session=session,
            vocabulary_margin_observability=True,
            vocabulary_margin_top_k=3,
            streaming_vocabulary_margin_v2=True,
        )


def test_streaming_vocabulary_margin_v2_rejects_resealed_reference_mismatch(
    monkeypatch,
) -> None:
    from neural_forge import hf_lfm2_balanced_circuit_basis as balanced
    from neural_forge.dynamic_circuit_ir import canonical_sha256

    parent = TinyLfm2(_config())
    cache = _FakeTraceCache(parent, include_logits=True)
    session = _FakeSelectiveSession(cache._batches, trace_cache=cache)
    body = copy.deepcopy(session._receipt)
    body.pop("session_sha256")
    body["ordered_authenticated_batches"][0]["input_sha256"] = "e" * 64
    session._receipt = {
        **body,
        "session_sha256": canonical_sha256(body),
    }
    monkeypatch.setattr(
        balanced,
        "Lfm2CircuitTraceSelectiveAuthenticatedSession",
        _FakeSelectiveSession,
    )

    with pytest.raises(ValueError, match="differs from its authenticated"):
        balanced.build_lfm2_circuit_balanced_basis(
            parent,
            cache,
            4,
            selective_session=session,
            vocabulary_margin_observability=True,
            vocabulary_margin_top_k=3,
            streaming_vocabulary_margin_v2=True,
        )


def test_lfm2_balanced_coordinate_recipe_binds_isometry_and_full_vocabulary(
    tmp_path,
) -> None:
    from neural_forge.dynamic_circuit_ir import canonical_sha256
    from neural_forge.hf_tied_causal_interface_fold import (
        TiedCausalInterfaceFold,
        _tensor_sha256 as interface_tensor_sha256,
    )
    from neural_forge.hf_token_align import (
        StrictTokenRowMapping,
        StrictTokenTransportProfile,
    )
    from neural_forge.hf_tokenizer_bridge import (
        STRICT_TOKENIZER_BRIDGE_SCHEMA,
        StrictTokenizerBridgeArtifact,
        TokenizerArtifactFile,
    )
    from neural_forge.lfm2_circuit_balanced_coordinate_recipe import (
        LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE,
        build_lfm2_circuit_balanced_coordinate_evidence,
    )

    torch.manual_seed(922)
    parent = TinyLfm2(_config())
    cache = _FakeTraceCache(parent, include_logits=True)
    target_width = 4
    encoder = torch.linalg.qr(
        torch.randn(8, target_width, dtype=torch.float64)
    ).Q
    target_table = torch.randn(19, target_width, dtype=torch.float64)
    interface_body = {
        "schema": "test-foundation-interface-v1",
        "source_width": 8,
        "target_width": target_width,
        "vocabulary_size": 19,
        "target_weight_sha256": interface_tensor_sha256(target_table),
        "producer_encoder_sha256": interface_tensor_sha256(encoder),
    }
    interface = TiedCausalInterfaceFold(
        target_weight=target_table,
        producer_encoder=encoder,
        report={
            **interface_body,
            "report_sha256": canonical_sha256(interface_body),
        },
    )
    bridge_root = tmp_path / "tokenizer"
    bridge_root.mkdir()
    tokenizer_bytes = b"{}\n"
    (bridge_root / "tokenizer.json").write_bytes(tokenizer_bytes)
    file_record = TokenizerArtifactFile(
        relative_path="tokenizer.json",
        size_bytes=len(tokenizer_bytes),
        sha256=hashlib.sha256(tokenizer_bytes).hexdigest(),
    )
    profile = StrictTokenTransportProfile(
        donor_tokenizer_class="TinyDonor",
        target_tokenizer_class="TinyTarget",
        donor_vocab_size=19,
        target_vocab_size=19,
        donor_row_capacity=19,
        target_row_capacity=19,
        alignment_fingerprint="7" * 64,
        row_mappings=tuple(
            StrictTokenRowMapping(
                target_row_index=index,
                donor_row_index=index,
                resolution="test-direct-row",
            )
            for index in range(19)
        ),
    )
    bridge_body = {
        "schema": STRICT_TOKENIZER_BRIDGE_SCHEMA,
        "kind": "sealed-strict-target-tokenizer",
        "profile_fingerprint": profile.fingerprint(),
        "transport_profile": profile.to_dict(),
        "donor_tokenizer_semantics_sha256": "8" * 64,
        "target_tokenizer_semantics_sha256": "9" * 64,
        "target_files": [file_record.to_dict()],
        "model_parameter_operations": {
            "training_steps": 0,
            "backward_calls": 0,
            "pruned_parameters": 0,
            "quantized_parameters": 0,
            "random_target_rows": 0,
        },
    }
    bridge = StrictTokenizerBridgeArtifact(
        root_dir=bridge_root,
        transport_profile=profile,
        donor_tokenizer_semantics_sha256="8" * 64,
        target_tokenizer_semantics_sha256="9" * 64,
        target_files=(file_record,),
        artifact_sha256=canonical_sha256(bridge_body),
    )
    forward_calls = 0

    def forbidden_forward(*_args, **_kwargs):
        nonlocal forward_calls
        forward_calls += 1
        raise AssertionError("balanced coordinate evidence invoked the parent")

    parent.forward = forbidden_forward
    evidence = build_lfm2_circuit_balanced_coordinate_evidence(
        parent,
        cache,
        target_width,
        foundation_interface=interface,
        tokenizer_bridge=bridge,
        max_rows_per_block=8,
        vocabulary_margin_top_k=3,
    )
    receipt = evidence.as_dict()

    assert forward_calls == 0
    assert receipt["recipe"] == LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE
    assert receipt["foundation_interface"][
        "structural_target_table_preserved"
    ] is True
    assert receipt["token_bridge"]["full_target_vocabulary_row_coverage"] is True
    assert receipt["rmsnorm_isometry_contract"][
        "orthonormal_basis_isometry"
    ] is True
    assert receipt["rmsnorm_isometry_contract"][
        "nonorthogonal_balanced_factors_cross_rmsnorm"
    ] is False
    assert receipt["causal_output_contract"][
        "partial_causal_trie_may_replace_structural_table"
    ] is False
    assert receipt["causal_output_contract"][
        "unsupported_logit_policy_required"
    ] == "adaptive-strongest-surviving-background-seal"
    assert receipt["causal_output_contract"][
        "fully_unsupported_rows_must_equal_structural_baseline"
    ] is True
    assert receipt["evidence_policy"]["parent_trace_regeneration_calls"] == 0


def test_lfm2_vocabulary_margin_heldout_logits_cannot_select_basis() -> None:
    from neural_forge.hf_lfm2_balanced_circuit_basis import (
        build_lfm2_circuit_balanced_basis,
    )

    torch.manual_seed(811)
    parent = TinyLfm2(_config())
    first_cache = _FakeTraceCache(parent, include_logits=True)
    first = build_lfm2_circuit_balanced_basis(
        parent,
        first_cache,
        4,
        validation_fraction=0.25,
        max_rows_per_block=8,
        vocabulary_margin_observability=True,
        vocabulary_margin_top_k=3,
    )
    first_report = first.as_dict()
    heldout_ids = {
        value["record_id"]
        for value in first_report["vocabulary_margin_observability"][
            "heldout"
        ]["phases"]
    }

    second_cache = _FakeTraceCache(parent, include_logits=True)
    for batch in second_cache._batches:
        if batch.input_sha256 not in heldout_ids:
            continue
        changed = dict(batch.tensors)
        for name, value in tuple(changed.items()):
            if name.endswith(".logits"):
                changed[name] = torch.roll(value, shifts=1, dims=-1)
        batch.tensors = MappingProxyType(changed)
    second = build_lfm2_circuit_balanced_basis(
        parent,
        second_cache,
        4,
        validation_fraction=0.25,
        max_rows_per_block=8,
        vocabulary_margin_observability=True,
        vocabulary_margin_top_k=3,
    )
    second_report = second.as_dict()

    assert torch.equal(first.basis, second.basis)
    assert first.map_sha256 == second.map_sha256
    assert first_report["balanced_basis"]["hashes"][
        "selection_sha256"
    ] == second_report["balanced_basis"]["hashes"]["selection_sha256"]
    assert first_report["vocabulary_margin_observability"][
        "heldout_diagnostics"
    ]["diagnostics_sha256"] != second_report[
        "vocabulary_margin_observability"
    ]["heldout_diagnostics"]["diagnostics_sha256"]
    rejected = first.evaluate_vocabulary_margin_candidate_gate(
        second_cache,
        minimum_sign_agreement_fraction=1.0,
        minimum_top_competitor_nonnegative_fraction=1.0,
        minimum_all_top_k_nonnegative_fraction=1.0,
        minimum_exact_top1_agreement_fraction=1.0,
    )
    assert rejected["gate_passed"] is False


def test_lfm2_vocabulary_margin_observability_fails_closed_without_logits() -> None:
    from neural_forge.hf_lfm2_balanced_circuit_basis import (
        Lfm2CircuitBalancedBasisError,
        build_lfm2_circuit_balanced_basis,
    )

    parent = TinyLfm2(_config())
    cache = _FakeTraceCache(parent, include_logits=False)
    with pytest.raises(
        Lfm2CircuitBalancedBasisError,
        match="requires full parent logits",
    ):
        build_lfm2_circuit_balanced_basis(
            parent,
            cache,
            4,
            vocabulary_margin_observability=True,
            vocabulary_margin_top_k=3,
        )


def test_lfm2_vocabulary_margin_top_tokens_break_ties_by_token_id() -> None:
    from neural_forge import hf_lfm2_balanced_circuit_basis as balanced

    evidence = balanced._deterministic_top_tokens(
        torch.tensor([[[2.0, 2.0, 2.0, 1.0]]]),
        vocabulary_size=4,
        top_k=2,
        phase="prefill",
    )

    assert evidence.token_ids.tolist() == [[0, 1, 2]]
    assert evidence.margins.tolist() == [[0.0, 0.0]]


def test_lfm2_circuit_balanced_adapter_preserves_variable_decode_horizons() -> None:
    from neural_forge.hf_lfm2_balanced_circuit_basis import (
        build_lfm2_circuit_balanced_basis,
    )

    torch.manual_seed(654)
    parent = TinyLfm2(_config())
    decode_phase_counts = (2, 5, 3, 7, 4)
    cache = _FakeTraceCache(
        parent,
        batch_count=len(decode_phase_counts),
        decode_phase_counts=decode_phase_counts,
    )
    result = build_lfm2_circuit_balanced_basis(
        parent,
        cache,
        4,
        validation_fraction=0.2,
        max_rows_per_block=8,
    )
    report = result.as_dict()
    cache_report = report["cache"]
    snapshot_report = report["snapshot_adapter"]
    records = cache_report["record_horizons"]

    assert cache_report["variable_decode_horizons_supported"] is True
    assert cache_report["phase_count_minimum"] == 3
    assert cache_report["phase_count_maximum"] == 8
    assert cache_report["decode_phase_count_minimum"] == 2
    assert cache_report["decode_phase_count_maximum"] == 7
    assert cache_report["residual_location_count_minimum"] == 3 * 14 * 3
    assert cache_report["residual_location_count_maximum"] == 8 * 14 * 3
    assert [value["decode_phase_count"] for value in records] == list(
        decode_phase_counts
    )
    assert [value["residual_location_count"] for value in records] == [
        (decode_count + 1) * 14 * 3
        for decode_count in decode_phase_counts
    ]
    fit_max = max(
        value["phase_count"]
        for value in records
        if value["split"] == "fit"
    )
    heldout_max = max(
        value["phase_count"]
        for value in records
        if value["split"] == "heldout"
    )
    assert snapshot_report["fit_union_phase_count"] == fit_max
    assert snapshot_report["heldout_union_phase_count"] == heldout_max
    assert snapshot_report["fit_block_count"] == fit_max * 14 * 3
    assert snapshot_report["heldout_block_count"] == heldout_max * 14 * 3
    assert (
        snapshot_report["all_record_locations_consumed_by_own_split"] is True
    )
    assert snapshot_report["common_horizon_truncation_used"] is False


def test_lfm2_circuit_balanced_adapter_source_has_no_execution_or_training_lane() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "neural_forge"
        / "hf_lfm2_balanced_circuit_basis.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.append(node.func.id)
    assert "forward" not in calls
    assert "backward" not in calls
    assert "step" not in calls
    assert "from_pretrained" not in calls
