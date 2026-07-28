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
                        name = (
                            f"{phase}.layer.{layer_index:03d}."
                            f"residual.{endpoint}"
                        )
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
