from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "refine_lfm2_dynamic_structural_residual_attention.py"
)
SPEC = importlib.util.spec_from_file_location(
    "refine_lfm2_dynamic_structural_residual_attention",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    Lfm2CachedAttentionFoldCase,
    Lfm2CachedMlpFoldCase,
)


class _DenseTestCoordinateMap:
    """Protocol-compatible dense basis with deliberately no assignment."""

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(19)
        raw = torch.randn(
            8, 4, generator=generator, dtype=torch.float64
        )
        self.basis = torch.linalg.qr(raw, mode="reduced").Q.contiguous()
        self.map_sha256 = MODULE.baseline._tensor_sha256(self.basis)

    @property
    def source_width(self) -> int:
        return int(self.basis.shape[0])

    @property
    def target_width(self) -> int:
        return int(self.basis.shape[1])

    def _basis_for(self, value: torch.Tensor) -> torch.Tensor:
        return self.basis.to(device=value.device, dtype=value.dtype)

    def encode(self, source_rows: torch.Tensor) -> torch.Tensor:
        return source_rows @ self._basis_for(source_rows)

    def decode(self, target_rows: torch.Tensor) -> torch.Tensor:
        return target_rows @ self._basis_for(target_rows).T

    def fold_consumer_weight(
        self, source_weight: torch.Tensor
    ) -> torch.Tensor:
        return source_weight @ self._basis_for(source_weight)

    def fold_producer_weight(
        self, source_weight: torch.Tensor
    ) -> torch.Tensor:
        return self._basis_for(source_weight).T @ source_weight


def _global_map() -> _DenseTestCoordinateMap:
    return _DenseTestCoordinateMap()


def _phase_tensors(
    *,
    generator: torch.Generator,
    sequence: int,
    source_hidden: int = 8,
    query_heads: int = 4,
    key_value_heads: int = 2,
    head_dim: int = 2,
    intermediate: int = 6,
) -> dict[str, torch.Tensor]:
    operator_input = torch.randn(
        1, sequence, source_hidden, generator=generator
    )
    query = torch.randn(
        1, query_heads, sequence, head_dim, generator=generator
    )
    key = torch.randn(
        1, key_value_heads, sequence, head_dim, generator=generator
    )
    value = torch.randn(
        1, key_value_heads, sequence, head_dim, generator=generator
    )
    gate = torch.randn(1, sequence, intermediate, generator=generator)
    up = torch.randn(1, sequence, intermediate, generator=generator)
    return {
        "operator.input": operator_input,
        "attention.q_linear": query.clone(),
        "attention.k_linear": key.clone(),
        "attention.q": query,
        "attention.k": key,
        "attention.v": value,
        "attention.context": torch.randn(
            1, query_heads, sequence, head_dim, generator=generator
        ),
        "mlp.input": torch.randn(
            1, sequence, source_hidden, generator=generator
        ),
        "mlp.gate": gate,
        "mlp.up": up,
        "mlp.post_swiglu": torch.nn.functional.silu(gate) * up,
        "residual.pre": torch.randn(
            1, sequence, source_hidden, generator=generator
        ),
        "residual.post_operator": torch.randn(
            1, sequence, source_hidden, generator=generator
        ),
        "residual.post_layer": torch.randn(
            1, sequence, source_hidden, generator=generator
        ),
    }


def _two_phase_attention_trace() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(91)
    prefill = _phase_tensors(generator=generator, sequence=2)
    decode = _phase_tensors(generator=generator, sequence=1)
    result: dict[str, torch.Tensor] = {}
    for phase, values in (("prefill", prefill), ("decode.000", decode)):
        base = f"{phase}.layer.000."
        result.update({base + name: value for name, value in values.items()})
    result[
        "decode.000.layer.000.attention.key_cache"
    ] = torch.cat(
        (prefill["attention.k"], decode["attention.k"]), dim=2
    )
    result[
        "decode.000.layer.000.attention.value_cache"
    ] = torch.cat(
        (prefill["attention.v"], decode["attention.v"]), dim=2
    )
    return result


def _single_phase_short_conv_trace() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(92)
    sequence = 12
    hidden = 8
    intermediate = 6
    gate = torch.randn(
        1, sequence, intermediate, generator=generator
    )
    up = torch.randn(
        1, sequence, intermediate, generator=generator
    )
    values = {
        "operator.input": torch.randn(
            1, sequence, hidden, generator=generator
        ),
        "short_conv.b": torch.randn(
            1, sequence, hidden, generator=generator
        ),
        "short_conv.c": torch.randn(
            1, sequence, hidden, generator=generator
        ),
        "short_conv.x": torch.randn(
            1, sequence, hidden, generator=generator
        ),
        "mlp.input": torch.randn(
            1, sequence, hidden, generator=generator
        ),
        "mlp.gate": gate,
        "mlp.up": up,
        "mlp.post_swiglu": torch.nn.functional.silu(gate) * up,
        "residual.pre": torch.randn(
            1, sequence, hidden, generator=generator
        ),
        "residual.post_operator": torch.randn(
            1, sequence, hidden, generator=generator
        ),
        "residual.post_layer": torch.randn(
            1, sequence, hidden, generator=generator
        ),
    }
    return {
        "prefill.layer.000." + name: value
        for name, value in values.items()
    }


class _ToyAttention(nn.Module):
    def __init__(
        self,
        hidden: int,
        query_heads: int,
        key_value_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(
            hidden, query_heads * head_dim, bias=False, dtype=torch.float64
        )
        self.k_proj = nn.Linear(
            hidden,
            key_value_heads * head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.v_proj = nn.Linear(
            hidden,
            key_value_heads * head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.out_proj = nn.Linear(
            query_heads * head_dim,
            hidden,
            bias=False,
            dtype=torch.float64,
        )


class _ToyMlp(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(
            hidden, intermediate, bias=False, dtype=torch.float64
        )
        self.w3 = nn.Linear(
            hidden, intermediate, bias=False, dtype=torch.float64
        )
        self.w2 = nn.Linear(
            intermediate, hidden, bias=False, dtype=torch.float64
        )


class _ToyRmsNorm(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden, dtype=torch.float64))
        self.variance_epsilon = 1e-5

    def forward(self, value):
        normalized = value.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True)
            + self.variance_epsilon
        )
        return self.weight * normalized.to(dtype=value.dtype)


class _ToyAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        query_heads: int,
        key_value_heads: int,
        head_dim: int,
        intermediate: int,
    ) -> None:
        super().__init__()
        self.is_attention_layer = True
        self.self_attn = _ToyAttention(
            hidden, query_heads, key_value_heads, head_dim
        )
        self.operator_norm = _ToyRmsNorm(hidden)
        self.ffn_norm = _ToyRmsNorm(hidden)
        self.feed_forward = _ToyMlp(hidden, intermediate)


class _ToyShortConv(nn.Module):
    def __init__(self, hidden: int, fir_length: int = 3) -> None:
        super().__init__()
        self.in_proj = nn.Linear(
            hidden, 3 * hidden, bias=False, dtype=torch.float64
        )
        self.conv = nn.Conv1d(
            hidden,
            hidden,
            fir_length,
            groups=hidden,
            bias=False,
            dtype=torch.float64,
        )
        self.out_proj = nn.Linear(
            hidden, hidden, bias=False, dtype=torch.float64
        )


class _ToyShortConvLayer(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.is_attention_layer = False
        self.conv = _ToyShortConv(hidden)
        self.operator_norm = _ToyRmsNorm(hidden)
        self.ffn_norm = _ToyRmsNorm(hidden)
        self.feed_forward = _ToyMlp(hidden, intermediate)


class _ToyFinalNormModel(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [nn.Identity() for _ in range(14)]
        )
        self.model.embedding_norm = _ToyRmsNorm(hidden)


def _random_refinement_cases() -> MODULE.SealedLayerCases:
    generator = torch.Generator().manual_seed(121)
    target_input = torch.randn(
        1, 7, 4, generator=generator, dtype=torch.float64
    )
    query = torch.randn(
        1, 4, 7, 2, generator=generator, dtype=torch.float64
    )
    key = torch.randn(
        1, 2, 7, 2, generator=generator, dtype=torch.float64
    )
    value = torch.randn(
        1, 2, 7, 2, generator=generator, dtype=torch.float64
    )
    attention = Lfm2CachedAttentionFoldCase(
        target_query_input=target_input,
        target_key_value_input=target_input,
        parent_query_linear=query,
        parent_key_linear=key,
        parent_value=value,
        parent_query=query,
        parent_key=key,
        parent_context=torch.randn(
            1, 4, 7, 2, generator=generator, dtype=torch.float64
        ),
        parent_post_attention_residual_target_space=torch.randn(
            1, 7, 4, generator=generator, dtype=torch.float64
        ),
        target_pre_attention_residual=torch.randn(
            1, 7, 4, generator=generator, dtype=torch.float64
        ),
    )
    rows = 11
    gate = torch.randn(
        rows, 6, generator=generator, dtype=torch.float64
    )
    up = torch.randn(rows, 6, generator=generator, dtype=torch.float64)
    mlp = Lfm2CachedMlpFoldCase(
        target_input=torch.randn(
            rows, 4, generator=generator, dtype=torch.float64
        ),
        parent_gate=gate,
        parent_up=up,
        parent_post_swiglu=torch.nn.functional.silu(gate) * up,
        parent_post_mlp_residual_target_space=torch.randn(
            rows, 4, generator=generator, dtype=torch.float64
        ),
        target_pre_mlp_residual=torch.randn(
            rows, 4, generator=generator, dtype=torch.float64
        ),
    )
    return MODULE.SealedLayerCases(
        layer_index=0,
        operator_kind="full_attention",
        phase_sequences=(("prefill",),),
        attention=(attention,),
        mlp=(mlp,),
        short_conv=(),
        cases_sha256="1" * 64,
    )


def test_dense_global_map_protocol_projects_without_sparse_assignment() -> None:
    mapping = _global_map()
    source = torch.arange(24, dtype=torch.float32).reshape(3, 8)

    projected = MODULE.project_with_global_residual_map(mapping, source)

    assert not hasattr(mapping, "assignment")
    assert MODULE._global_coordinate_map_contract(mapping) == (
        8,
        4,
        mapping.map_sha256,
    )
    assert torch.equal(projected, mapping.encode(source))


def test_public_adapters_assemble_prefill_and_decode_without_trace_mutation() -> None:
    tensors = _two_phase_attention_trace()
    target = _ToyAttentionLayer(4, 2, 1, 2, 3)
    before = {
        name: value.clone()
        for name, value in tensors.items()
    }

    cases = MODULE.assemble_sealed_layer_cases(
        (tensors,),
        layer_index=0,
        is_attention_layer=True,
        mapping=_global_map(),
        target_layer=target,
    )
    repeated = MODULE.assemble_sealed_layer_cases(
        (tensors,),
        layer_index=0,
        is_attention_layer=True,
        mapping=_global_map(),
        target_layer=target,
    )

    assert cases.operator_kind == "full_attention"
    assert cases.phase_sequences == (("prefill", "decode.000"),)
    assert len(cases.attention) == 2
    assert len(cases.mlp) == 2
    assert cases.short_conv == ()
    assert cases.full_multi_phase is True
    assert cases.cases_sha256 == repeated.cases_sha256
    assert cases.attention[0].target_query_input.shape[-1] == 4
    assert cases.attention[1].target_key_value_input.shape[1] == 3
    assert cases.target_norms is not None
    projected_residual = MODULE.project_with_global_residual_map(
        _global_map(),
        tensors["prefill.layer.000.residual.pre"],
    )
    deployed_operator_input = MODULE._materialized_rms_norm(
        projected_residual,
        cases.target_norms.operator.weight,
        epsilon=cases.target_norms.operator.epsilon,
    )
    assert torch.equal(
        cases.attention[0].target_query_input,
        deployed_operator_input,
    )
    assert not torch.equal(
        cases.attention[0].target_query_input,
        MODULE.project_with_global_residual_map(
            _global_map(),
            tensors["prefill.layer.000.operator.input"],
        ),
    )
    assert cases.target_norms.operator.as_dict()["model_forward_calls"] == 0
    assert cases.as_dict()["parent_forward_calls"] == 0
    for name, value in tensors.items():
        assert torch.equal(value, before[name])


def test_real_core_refines_qkvo_and_swiglu_in_existing_parameters() -> None:
    torch.manual_seed(301)
    source = _ToyAttentionLayer(8, 4, 2, 2, 6)
    target = _ToyAttentionLayer(4, 2, 1, 2, 3)
    source_before = MODULE._module_parameter_state_sha256(source)
    target_before = MODULE._module_parameter_state_sha256(target)
    cases = MODULE.assemble_sealed_layer_cases(
        (_two_phase_attention_trace(),),
        layer_index=0,
        is_attention_layer=True,
        mapping=_global_map(),
        target_layer=target,
    )
    identities = {
        name: id(parameter)
        for name, parameter in target.named_parameters()
    }

    report = MODULE.refine_layer_from_sealed_cases(
        source,
        target,
        cases,
        global_residual_map_sha256=_global_map().map_sha256,
        ridge=1e-6,
        max_prior_delta_ratio=None,
        cluster_ridge=1e-8,
        cluster_max_iterations=4,
        cluster_distance_chunk_size=16,
    )

    assert report["operator"]["kind"] == "full_attention"
    assert report["operator"]["atomic_qkvo_and_swiglu"] is True
    assert report["operator"]["install"]["accepted"] is True
    assert (
        report["operator"]["global_residual_map_sha256"]
        == _global_map().map_sha256
    )
    assert report["operator"]["short_conv_channel_assignment_sha256"] is None
    assert report["mlp"]["full_source_coverage"] is True
    assert report["full_source_coverage"] is True
    assert report["source_units_dropped"] == 0
    assert report["target_parameter_topology_preserved"] is True
    assert report["target_rms_norms"]["install"]["accepted"] is True
    assert MODULE.baseline._tensor_sha256(
        target.operator_norm.weight
    ) == cases.target_norms.operator.fitted_weight_sha256
    assert MODULE.baseline._tensor_sha256(
        target.ffn_norm.weight
    ) == cases.target_norms.ffn.fitted_weight_sha256
    assert report["parent_forward_calls"] == 0
    assert MODULE._module_parameter_state_sha256(source) == source_before
    assert MODULE._module_parameter_state_sha256(target) != target_before
    assert {
        name: id(parameter)
        for name, parameter in target.named_parameters()
    } == identities


def test_short_conv_uses_separate_bcx_assignment_from_dense_global_map() -> None:
    torch.manual_seed(302)
    mapping = _global_map()
    source = _ToyShortConvLayer(8, 6)
    target = _ToyShortConvLayer(4, 3)
    cases = MODULE.assemble_sealed_layer_cases(
        (_single_phase_short_conv_trace(),),
        layer_index=0,
        is_attention_layer=False,
        mapping=mapping,
        target_layer=target,
    )
    first_assignment, first_report = (
        MODULE.build_short_conv_channel_assignment(
            cases.short_conv,
            layer_index=0,
            target_width=4,
            global_residual_map_sha256=mapping.map_sha256,
        )
    )
    second_assignment, second_report = (
        MODULE.build_short_conv_channel_assignment(
            cases.short_conv,
            layer_index=0,
            target_width=4,
            global_residual_map_sha256=mapping.map_sha256,
        )
    )

    assert not hasattr(mapping, "assignment")
    assert first_assignment.assignment_sha256 == (
        second_assignment.assignment_sha256
    )
    assert first_report == second_report
    assert first_assignment.full_source_coverage is True
    assert first_assignment.source_width == 8
    assert first_assignment.target_width == 4
    assert first_report["dense_global_basis_interpreted_as_sparse_assignment"] is False
    assert first_report["global_residual_map_sha256"] == mapping.map_sha256

    report = MODULE.refine_layer_from_sealed_cases(
        source,
        target,
        cases,
        global_residual_map_sha256=mapping.map_sha256,
        short_conv_channel_assignment=first_assignment,
        short_conv_channel_assignment_report=first_report,
        ridge=1e-6,
        max_prior_delta_ratio=None,
        cluster_ridge=1e-8,
        cluster_max_iterations=4,
        cluster_distance_chunk_size=16,
    )

    assert report["operator"]["kind"] == "short_conv"
    assert report["global_residual_map_sha256"] == mapping.map_sha256
    assert report["mlp"]["residual_assignment_sha256"] == (
        mapping.map_sha256
    )
    assert report["short_conv_channel_assignment_sha256"] == (
        first_assignment.assignment_sha256
    )
    assert report["operator"]["compiled"]["residual_assignment"][
        "assignment_sha256"
    ] == first_assignment.assignment_sha256
    assert report["operator"][
        "short_conv_channel_assignment"
    ]["independent_of_global_residual_coordinate_map"] is True


def test_final_embedding_norm_is_derived_from_sealed_last_residual() -> None:
    generator = torch.Generator().manual_seed(777)
    parent = _ToyFinalNormModel(8)
    child = _ToyFinalNormModel(4)
    with torch.no_grad():
        parent.model.embedding_norm.weight.copy_(
            torch.linspace(0.5, 1.5, 8, dtype=torch.float64)
        )
        child.model.embedding_norm.weight.fill_(0.25)
    residual = torch.randn(
        1, 6, 8, generator=generator, dtype=torch.float64
    )
    tensors = {
        "prefill.layer.013.residual.post_layer": residual,
        "prefill.layer.013.residual.pre": residual,
        "prefill.logits": torch.randn(
            1, 6, 11, generator=generator, dtype=torch.float64
        ),
    }
    parent_before = MODULE._module_parameter_state_sha256(
        parent.model.embedding_norm
    )
    identity = id(child.model.embedding_norm.weight)

    report = MODULE.refine_final_embedding_norm(
        parent,
        child,
        (tensors,),
        _global_map(),
        max_phases_per_batch=0,
        ridge=1e-8,
    )

    assert report["install"]["accepted"] is True
    assert report["global_residual_map_sha256"] == (
        _global_map().map_sha256
    )
    assert report["explicit_final_normalized_cache_tensor_used"] is False
    assert report["cache_non_logit_global_tensor_inventory"] == []
    assert report["parent_forward_calls"] == 0
    assert id(child.model.embedding_norm.weight) == identity
    assert MODULE._module_parameter_state_sha256(
        parent.model.embedding_norm
    ) == parent_before
    assert (
        MODULE.baseline._tensor_sha256(child.model.embedding_norm.weight)
        == report["fit"]["fitted_weight_sha256"]
    )


def test_save_helper_constructs_a_fresh_independent_model(tmp_path: Path) -> None:
    def build_model() -> nn.Module:
        return nn.Sequential(
            nn.Linear(4, 3, bias=False),
            nn.Linear(3, 2, bias=False),
        )

    torch.manual_seed(401)
    model = build_model()
    count = MODULE.baseline.dense_logical_parameter_audit(model)[
        "unique_logical_parameters"
    ]

    def save_function(model, _source, destination, report):
        destination.mkdir()
        torch.save(model.state_dict(), destination / "state.pt")
        (destination / "report.json").write_text(
            str(report), encoding="utf-8"
        )

    def load_function(destination, *, device, dtype):
        result = build_model().to(device=device, dtype=dtype)
        result.load_state_dict(
            torch.load(destination / "state.pt", weights_only=True)
        )
        return result

    reloaded, report = MODULE.save_and_fresh_reload(
        model,
        source_directory=tmp_path,
        destination=tmp_path / "child",
        report={"schema": "toy"},
        device=torch.device("cpu"),
        dtype=torch.float32,
        expected_unique_parameters=count,
        save_function=save_function,
        load_function=load_function,
    )

    assert reloaded is not model
    assert report["verified"] is True
    assert report["independent_model_object"] is True
    assert report["exact_unique_logical_parameters"] == count
    assert MODULE.baseline.model_state_sha256(reloaded) == (
        MODULE.baseline.model_state_sha256(model)
    )


def test_zero_cache_batch_limit_loads_every_verified_batch() -> None:
    class Batch:
        def __init__(self, index: int) -> None:
            self.tensors = {
                f"prefill.layer.{index:03d}.residual.pre": torch.tensor(
                    [float(index)]
                )
            }

    class Cache:
        def __init__(self, count: int) -> None:
            self._batches = tuple(Batch(index) for index in range(count))

        def iter_batches(self):
            yield from self._batches

    cache = Cache(5)
    all_batches = MODULE.load_sealed_trace_batches(cache, maximum=0)
    bounded = MODULE.load_sealed_trace_batches(cache, maximum=2)

    assert len(all_batches) == 5
    assert len(bounded) == 2
    assert tuple(all_batches[0]) == (
        "prefill.layer.000.residual.pre",
    )
    with pytest.raises(
        MODULE.Lfm2StructuralRefinementError,
        match="non-negative",
    ):
        MODULE.load_sealed_trace_batches(cache, maximum=-1)
    with pytest.raises(
        MODULE.Lfm2StructuralRefinementError,
        match="non-negative",
    ):
        MODULE.load_sealed_trace_batches(cache, maximum=True)


def test_parser_defaults_to_all_sealed_cache_batches() -> None:
    args = MODULE.build_parser().parse_args(
        [
            "--parent-model",
            "parent",
            "--baseline-child",
            "child",
            "--trace-cache",
            "cache",
            "--output",
            "output",
        ]
    )
    assert args.max_cache_batches == 0


def test_driver_has_no_training_parent_execution_hub_or_quantization_lane() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert "from_pretrained" not in called_attributes
    assert "forward" not in called_attributes
    assert "generate" not in called_attributes
    assert "backward" not in called_attributes
    assert "step" not in called_attributes
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "bitsandbytes" not in source
    assert "dtype_or_packing_reduction_claimed" in source
    assert MODULE.parse_layer_selection("0,2-4", 6) == (0, 2, 3, 4)
    assert MODULE.parse_layer_selection("all", 3) == (0, 1, 2)
