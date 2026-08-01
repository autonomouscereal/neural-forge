from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import os
import socket
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_lfm2_residual_attention_fold.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_residual_attention_fold", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


_LFM2_MODULE = "transformers.models.lfm2.modeling_lfm2"


class Lfm2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float64))


class Lfm2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=torch.float64
        )
        self.w3 = nn.Linear(
            hidden_size, intermediate_size, bias=False, dtype=torch.float64
        )
        self.w2 = nn.Linear(
            intermediate_size, hidden_size, bias=False, dtype=torch.float64
        )


class Lfm2Attention(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.out_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
            dtype=torch.float64,
        )
        self.q_layernorm = Lfm2RMSNorm(config.head_dim)
        self.k_layernorm = Lfm2RMSNorm(config.head_dim)


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int = 0):
        super().__init__()
        self.is_attention_layer = True
        self.self_attn = Lfm2Attention(config, layer_idx)
        self.feed_forward = Lfm2MLP(config.hidden_size, config.intermediate_size)
        self.operator_norm = Lfm2RMSNorm(config.hidden_size)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size)


for _class in (Lfm2RMSNorm, Lfm2MLP, Lfm2Attention, Lfm2DecoderLayer):
    _class.__module__ = _LFM2_MODULE


class _Body(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=torch.float64
        )
        self.layers = nn.ModuleList([Lfm2DecoderLayer(config)])
        self.norm = Lfm2RMSNorm(config.hidden_size)


class _TinyLfmCausal(nn.Module):
    def __init__(self, config: SimpleNamespace, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = _Body(config)
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                dtype=torch.float64,
            )
        self.lm_head.weight = self.model.embed_tokens.weight
        self.config = config

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _layer_config(*, intermediate: int, query: int, key_value: int):
    return SimpleNamespace(
        model_type="lfm2",
        hidden_size=8,
        intermediate_size=intermediate,
        block_ff_dim=intermediate,
        block_auto_adjust_ff_dim=False,
        num_attention_heads=query,
        num_key_value_heads=key_value,
        num_heads=query,
        head_dim=2,
        layer_types=["full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        vocab_size=13,
    )


def _trace_rows(seed: int = 41):
    generator = torch.Generator().manual_seed(seed)
    row_count = 12
    fields = {
        "attention.q": torch.randn(
            row_count, 16, 2, generator=generator, dtype=torch.float64
        ),
        "attention.k": torch.randn(
            row_count, 8, 2, generator=generator, dtype=torch.float64
        ),
        "attention.v": torch.randn(
            row_count, 8, 2, generator=generator, dtype=torch.float64
        ),
        "attention.context": torch.randn(
            row_count, 16, 2, generator=generator, dtype=torch.float64
        ),
        "mlp.gate": torch.randn(row_count, 6, generator=generator, dtype=torch.float64),
        "mlp.up": torch.randn(row_count, 6, generator=generator, dtype=torch.float64),
        "mlp.post_swiglu": torch.randn(
            row_count, 6, generator=generator, dtype=torch.float64
        ),
        "residual.pre": torch.randn(
            row_count, 8, generator=generator, dtype=torch.float64
        ),
        "residual.post_operator": torch.randn(
            row_count, 8, generator=generator, dtype=torch.float64
        ),
        "residual.post_layer": torch.randn(
            row_count, 8, generator=generator, dtype=torch.float64
        ),
    }
    paths = tuple(
        f"batch.{index // 6:06d}/prefill/row.{index % 6:04d}"
        for index in range(row_count)
    )
    return MODULE.TraceRowIndex(
        paths=paths,
        path_splits=tuple(
            "selection" if index < 6 else "fit" for index in range(row_count)
        ),
        layer_kinds=("full_attention",),
        layer_fields=MappingProxyType({0: MappingProxyType(fields)}),
        cache_key="a" * 64,
        corpus_fingerprint="b" * 64,
        trace_rows_sha256="c" * 64,
    )


class _Config:
    def __init__(self):
        self.model_type = "lfm2"
        self.hidden_size = 1024
        self.intermediate_size = 2560
        self.block_ff_dim = 2560
        self.block_auto_adjust_ff_dim = True
        self.num_attention_heads = 16
        self.num_key_value_heads = 8
        self.num_heads = 16
        self.head_dim = 64

    def to_dict(self):
        return copy.deepcopy(vars(self))


def test_fixed_target_config_sets_all_aliases_without_mutating_parent() -> None:
    parent = _Config()
    before = parent.to_dict()
    target = MODULE.build_fixed_lfm2_target_config(parent)

    assert parent.to_dict() == before
    assert MODULE.validate_fixed_target_aliases(target)["exact"] is True
    assert target.num_attention_heads == 14
    assert target.num_key_value_heads == 7
    assert target.num_heads == 14
    assert target.head_dim == 64
    assert target.intermediate_size == 2528
    assert target.block_ff_dim == 2528
    assert target.block_auto_adjust_ff_dim is False
    accounting = MODULE.fixed_contract_accounting()
    assert accounting["target_unique_parameters"] == 225_957_632
    assert accounting["learned_parameter_savings"] == 3_735_552


def test_dynamic_target_can_keep_attention_exact_and_remove_one_mlp_channel() -> None:
    parent = _Config()
    target_spec = MODULE.Lfm2TargetSpec(
        query_heads=16,
        key_value_heads=8,
        intermediate_size=2559,
    )

    target = MODULE.build_lfm2_target_config(parent, target_spec)

    assert target.num_attention_heads == 16
    assert target.num_key_value_heads == 8
    assert target.intermediate_size == 2559
    assert target.block_ff_dim == 2559
    assert MODULE.validate_target_aliases(target, target_spec)["exact"] is True


def test_dynamic_target_rejects_changed_gqa_group_ratio() -> None:
    with pytest.raises(
        MODULE.Lfm2ResidualAttentionRunnerError,
        match="query-heads-per-KV-group",
    ):
        MODULE.build_lfm2_target_config(
            _Config(),
            MODULE.Lfm2TargetSpec(
                query_heads=15,
                key_value_heads=5,
                intermediate_size=2559,
            ),
        )


def test_path_partition_holds_out_whole_batch_before_selection_and_fit() -> None:
    paths = tuple(
        f"batch.{batch:06d}/decode.{step:03d}/row.0000"
        for batch in range(3)
        for step in range(4)
    )
    partition = MODULE.deterministic_path_partition(
        paths,
        path_splits=tuple(
            "selection" if batch == 0 else "fit"
            for batch in range(3)
            for _step in range(4)
        ),
        max_selection_rows=6,
        max_fit_rows=7,
        max_validation_rows=4,
        validation_fraction=0.25,
    )
    report = partition.as_dict()

    assert report["validation_disjoint_from_selection"] is True
    assert report["validation_disjoint_from_fit"] is True
    assert report["selection_disjoint_from_fit"] is True
    validation_batches = {
        paths[index].split("/", 1)[0] for index in partition.validation_indices
    }
    discovery_batches = {
        paths[index].split("/", 1)[0]
        for index in (
            *partition.selection_indices,
            *partition.fit_indices,
        )
    }
    assert validation_batches.isdisjoint(discovery_batches)


class _TraceBatch:
    def __init__(self, tensors):
        self.index = 0
        self.tensors = tensors


class _TraceCache:
    cache_key = "d" * 64
    corpus_fingerprint = "e" * 64
    manifest = {
        "layer_topology": {"layers": [{"contract": {"kind": "full_attention"}}]}
    }

    def __init__(self, tensors):
        self._batch = _TraceBatch(tensors)

    def iter_batches(self):
        yield self._batch

    def __len__(self):
        return 1


class _SelectedPayload:
    def __init__(self, tensors):
        self.tensors = MappingProxyType(dict(tensors))
        self.released = False


class _SelectiveSession:
    def __init__(self, cache, tensors):
        self.cache_key = cache.cache_key
        self._tensors = tensors
        inventory = {
            name: {
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "numel": tensor.numel(),
                "size_bytes": tensor.numel() * tensor.element_size(),
                "sha256": "0" * 64,
            }
            for name, tensor in tensors.items()
        }
        self.batch_references = (
            SimpleNamespace(
                index=0,
                ordered_tensor_names=tuple(sorted(tensors)),
                record={"tensors": inventory},
            ),
        )
        self.payloads = []

    def __len__(self):
        return 1

    def load_selected_tensors(self, index, names):
        assert index == 0
        assert tuple(names) == tuple(sorted(set(names)))
        payload = _SelectedPayload({name: self._tensors[name] for name in names})
        self.payloads.append(payload)
        return payload

    @staticmethod
    def release_selected_tensors(payload):
        payload.released = True


def test_trace_extraction_permutes_attention_and_evidence_uses_residual_endpoints() -> (
    None
):
    q = torch.arange(1 * 2 * 3 * 2, dtype=torch.float64).reshape(1, 2, 3, 2)
    optional_q = torch.arange(1 * 2 * 4 * 2, dtype=torch.float64).reshape(1, 2, 4, 2)
    tensors = {
        "prefill.layer.000.attention.q": q,
        "prefill.layer.000.attention.q_linear": optional_q + 10.0,
        "prefill.layer.000.attention.q_normalized_pre_rope": optional_q + 20.0,
        "prefill.layer.000.attention.k": q[:, :1],
        "prefill.layer.000.attention.k_linear": optional_q[:, :1] + 30.0,
        "prefill.layer.000.attention.k_normalized_pre_rope": (optional_q[:, :1] + 40.0),
        "prefill.layer.000.attention.v": q[:, :1],
        "prefill.layer.000.attention.context": q,
        "prefill.layer.000.mlp.gate": torch.ones(1, 3, 6),
        "prefill.layer.000.mlp.up": torch.full((1, 3, 6), 2.0),
        "prefill.layer.000.mlp.post_swiglu": torch.full((1, 3, 6), 3.0),
        "prefill.layer.000.residual.pre": torch.full((1, 3, 8), 4.0),
        "prefill.layer.000.residual.post_operator": torch.full((1, 3, 8), 5.0),
        "prefill.layer.000.residual.post_layer": torch.full((1, 3, 8), 6.0),
    }
    extracted = MODULE.extract_trace_rows(_TraceCache(tensors))
    expected_q = q.permute(0, 2, 1, 3).reshape(3, 2, 2)
    expected_optional_q = optional_q.permute(0, 2, 1, 3).reshape(4, 2, 2)[-3:]
    torch.testing.assert_close(extracted.field(0, "attention.q"), expected_q)
    torch.testing.assert_close(
        extracted.field(0, "attention.q_normalized_pre_rope"),
        expected_optional_q + 20.0,
    )
    torch.testing.assert_close(
        extracted.field(0, "attention.k_normalized_pre_rope"),
        expected_optional_q[:, :1] + 40.0,
    )

    parent = _trace_rows(51)
    child = _trace_rows(52)
    child = MODULE.TraceRowIndex(
        paths=parent.paths,
        path_splits=parent.path_splits,
        layer_kinds=parent.layer_kinds,
        layer_fields=child.layer_fields,
        cache_key=child.cache_key,
        corpus_fingerprint=parent.corpus_fingerprint,
        trace_rows_sha256=child.trace_rows_sha256,
    )
    producer = SimpleNamespace(
        layer_index=0,
        selection_indices=(0, 1, 2, 3),
        fit_indices=(0, 1, 2),
        validation_indices=(6, 7),
    )
    source = Lfm2DecoderLayer(_layer_config(intermediate=6, query=16, key_value=8))
    evidence, split = MODULE.build_layer_evidence(
        source,
        parent,
        child,
        producer,
        max_fit_rows=99,
        max_validation_rows=99,
        validation_fraction=0.5,
    )

    assert split.fit_indices == (0, 1, 2)
    assert split.validation_indices == (6, 7)
    torch.testing.assert_close(
        evidence.attention.parent_branch_output_rows,
        parent.field(0, "residual.post_operator")[:3],
    )
    torch.testing.assert_close(
        evidence.attention.child_pre_branch_residual_rows,
        child.field(0, "residual.pre")[:3],
    )
    torch.testing.assert_close(
        evidence.mlp.parent_branch_output_rows,
        parent.field(0, "residual.post_layer")[:3],
    )
    torch.testing.assert_close(
        evidence.mlp.child_pre_branch_residual_rows,
        child.field(0, "residual.post_operator")[:3],
    )


def test_selective_cache_inputs_and_trace_rows_match_legacy_without_full_loads(
    monkeypatch,
) -> None:
    q = torch.arange(1 * 2 * 3 * 2, dtype=torch.float64).reshape(1, 2, 3, 2)
    tensors = {
        "input.attention_mask": torch.ones((1, 3), dtype=torch.long),
        "input.forced_decode_ids": torch.tensor([[4, 5]], dtype=torch.long),
        "input.input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "input.position_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "prefill.layer.000.attention.context": q,
        "prefill.layer.000.attention.k": q[:, :1],
        "prefill.layer.000.attention.q": q,
        "prefill.layer.000.attention.v": q[:, :1],
        "prefill.layer.000.mlp.gate": torch.ones(1, 3, 6),
        "prefill.layer.000.mlp.post_swiglu": torch.full((1, 3, 6), 3.0),
        "prefill.layer.000.mlp.up": torch.full((1, 3, 6), 2.0),
        "prefill.layer.000.residual.post_layer": torch.full((1, 3, 8), 6.0),
        "prefill.layer.000.residual.post_operator": torch.full((1, 3, 8), 5.0),
        "prefill.layer.000.residual.pre": torch.full((1, 3, 8), 4.0),
    }
    cache = _TraceCache(tensors)
    cache._batch.input_ids = tensors["input.input_ids"]
    cache._batch.attention_mask = tensors["input.attention_mask"]
    cache._batch.forced_decode_ids = tensors["input.forced_decode_ids"]
    legacy_inputs = MODULE.calibration_batches_from_cache(cache)
    legacy_rows = MODULE.extract_trace_rows(cache)
    session = _SelectiveSession(cache, tensors)

    selective_inputs = MODULE.calibration_batches_from_cache(
        cache,
        selective_session=session,
    )

    def reject_concatenation(*_args, **_kwargs):
        raise AssertionError("selective extraction must fill preallocated storage")

    monkeypatch.setattr(MODULE.torch, "cat", reject_concatenation)
    selective_rows = MODULE.extract_trace_rows(
        cache,
        selective_session=session,
    )

    assert len(session.payloads) == 2
    assert all(payload.released for payload in session.payloads)
    assert selective_rows.trace_rows_sha256 == legacy_rows.trace_rows_sha256
    for field in legacy_rows.layer_fields[0]:
        torch.testing.assert_close(
            selective_rows.field(0, field),
            legacy_rows.field(0, field),
        )
    for selective, legacy in zip(selective_inputs, legacy_inputs, strict=True):
        assert selective.keys() == legacy.keys()
        for name in legacy:
            torch.testing.assert_close(selective[name], legacy[name])


def test_preallocated_trace_extraction_releases_payload_when_fill_fails() -> None:
    q = torch.arange(1 * 2 * 3 * 2, dtype=torch.float64).reshape(1, 2, 3, 2)
    tensors = {
        "prefill.layer.000.attention.context": q,
        "prefill.layer.000.attention.k": q[:, :1],
        "prefill.layer.000.attention.q": q,
        "prefill.layer.000.attention.v": q[:, :1],
        "prefill.layer.000.mlp.gate": torch.ones(1, 3, 6),
        "prefill.layer.000.mlp.post_swiglu": torch.full((1, 3, 6), 3.0),
        "prefill.layer.000.mlp.up": torch.full((1, 3, 6), 2.0),
        "prefill.layer.000.residual.post_layer": torch.full((1, 3, 8), 6.0),
        "prefill.layer.000.residual.post_operator": torch.full((1, 3, 8), 5.0),
        "prefill.layer.000.residual.pre": torch.full((1, 3, 8), 4.0),
    }
    cache = _TraceCache(tensors)
    session = _SelectiveSession(cache, tensors)
    session._tensors["prefill.layer.000.mlp.gate"] = torch.ones(1, 2, 6)

    with pytest.raises(
        MODULE.Lfm2ResidualAttentionRunnerError,
        match="changed shape or dtype",
    ):
        MODULE.extract_trace_rows(cache, selective_session=session)

    assert len(session.payloads) == 1
    assert session.payloads[0].released is True


def test_real_medoid_provisional_compiler_erases_two_blank_seeds() -> None:
    parent = _TinyLfmCausal(
        _layer_config(intermediate=6, query=16, key_value=8), seed=100
    )
    child_a = _TinyLfmCausal(
        _layer_config(intermediate=4, query=14, key_value=7), seed=101
    )
    child_b = _TinyLfmCausal(
        _layer_config(intermediate=4, query=14, key_value=7), seed=202
    )
    rows = _trace_rows()
    plans = MODULE.derive_all_producer_plans(
        parent,
        child_a,
        rows,
        fold_config=MODULE.Lfm2ClosedLoopFoldConfig(
            max_endpoint_relative_rmse=10.0,
            min_endpoint_cosine_similarity=-1.0,
        ),
        max_selection_rows=6,
        max_fit_rows=5,
        max_validation_rows=3,
        validation_fraction=0.25,
    )
    plan = plans[0]
    before_a = MODULE.model_state_sha256(child_a)
    before_b = MODULE.model_state_sha256(child_b)
    assert before_a != before_b
    assert plan.attention is not None
    assert len(set(plan.attention.group_medoids)) == 7
    assert len(set(plan.attention.query_head_medoids)) == 14
    assert len(set(plan.mlp.medoid_source_neurons)) == 4
    assert set(plan.validation_indices).isdisjoint(plan.selection_indices)
    assert set(plan.validation_indices).isdisjoint(plan.fit_indices)

    report_a = MODULE.install_complete_provisional_state(parent, child_a, plans)
    report_b = MODULE.install_complete_provisional_state(parent, child_b, plans)
    assert report_a["blank_values_read"] is False
    assert report_a["state_sha256"] == report_b["state_sha256"]
    assert (
        MODULE.audit_global_transplant(parent, child_a)["tie_topology_preserved"]
        is True
    )
    assert all(parameter.grad is None for parameter in child_a.parameters())


class _GateBatch:
    def __init__(self, tensors):
        self.tensors = tensors
        self.input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        self.attention_mask = torch.ones((1, 3), dtype=torch.long)
        self.forced_decode_ids = torch.tensor([[5, 6]], dtype=torch.long)


class _GateCache:
    cache_key = "f" * 64
    parent_fingerprint = "1" * 64
    corpus_fingerprint = "2" * 64

    def __init__(self, tensors):
        self._batch = _GateBatch(tensors)

    def __len__(self):
        return 1

    def load_batch(self, index):
        assert index == 0
        return self._batch


def _logits(token: int, *, alternative: int | None = None):
    value = torch.zeros((1, 1, 8), dtype=torch.float64)
    value[..., token] = 1.0
    if alternative is not None:
        value[..., alternative] = 0.9
    return value


class _GreedyChild(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        position_ids=None,
        use_cache,
        return_dict,
        past_key_values=None,
    ):
        del attention_mask, position_ids, return_dict
        prefix_length = int(input_ids.shape[1])
        if past_key_values is None:
            token = 5
            step = 0
        else:
            token = 6
            step = int(past_key_values) + 1
        logits = torch.zeros((1, prefix_length, 8), device=input_ids.device)
        logits[..., token] = 1.0
        if not use_cache:
            alternative = 6 if token == 5 else 5
            logits[..., alternative] = 1.1
            logits[..., token] = 1.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=(step if use_cache else None),
        )


def test_cross_width_residual_metrics_use_representation_geometry() -> None:
    latent = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=torch.float64,
    )
    parent = torch.cat((latent, latent), dim=1)
    preserved = MODULE._residual_metrics(parent, latent)
    collapsed = MODULE._residual_metrics(
        parent,
        torch.zeros_like(latent),
    )

    assert preserved["comparison_mode"] == "cross_width_centered_linear_cka"
    assert preserved["parent_width"] == 4
    assert preserved["child_width"] == 2
    assert preserved["centered_linear_cka"] == pytest.approx(1.0)
    assert preserved["normalized_gram_distance"] == pytest.approx(0.0)
    assert collapsed["mean_cosine"] == 0.0
    assert collapsed["relative_rmse"] == pytest.approx(2.0**0.5)


def test_gate_uses_cached_parent_trajectory_and_full_top1_is_diagnostic() -> None:
    parent_tensors = {
        # Deliberately disagree with the externally cached parent generation
        # token. Generation processors can choose a valid trajectory that is
        # not the raw-logit argmax; this remains diagnostic, not a veto.
        "cached_prefill.logits": _logits(4),
        "prefill.logits": _logits(0),
        "decode.000.logits": _logits(6),
        "decode.001.logits": _logits(7),
    }
    child_tensors = {
        "cached_prefill.logits": _logits(5),
        "prefill.logits": _logits(1),
        "decode.000.logits": _logits(6, alternative=5),
        "decode.000.full_prefix_logits": _logits(5, alternative=6),
        "decode.001.logits": _logits(7, alternative=6),
        "decode.001.full_prefix_logits": _logits(6, alternative=7),
    }
    rows = _trace_rows(61)
    child_rows = MODULE.TraceRowIndex(
        paths=rows.paths,
        path_splits=rows.path_splits,
        layer_kinds=rows.layer_kinds,
        layer_fields=rows.layer_fields,
        cache_key="3" * 64,
        corpus_fingerprint=rows.corpus_fingerprint,
        trace_rows_sha256="4" * 64,
    )
    report = MODULE.evaluate_trajectory_gate(
        _GateCache(parent_tensors),
        _GateCache(child_tensors),
        _GreedyChild(),
        rows,
        child_rows,
        config=MODULE.TrajectoryGateConfig(
            min_teacher_forced_top1=1.0,
            min_free_run_token_match=1.0,
            min_mean_residual_cosine=1.0,
            logit_rtol=1.0,
            logit_atol=1.0,
        ),
    )

    assert report["passed"] is True
    assert report["teacher_forced"]["top1_agreement"] < 1.0
    assert report["free_run"]["token_match_fraction"] == 1.0
    assert report["cached_parent_trajectory_is_canonical"] is True
    assert report["parent_greedy_reference_verified"] is False
    assert report["parent_reference_token_check_is_diagnostic_only"] is True
    assert report["full_prefix_top1_is_diagnostic_only"] is True
    assert report["cached_full_top1_diagnostic_fraction"] < 1.0
    assert report["exact_token_and_lcp_metrics_are_diagnostic_only"] is True


def test_gate_accepts_only_canonical_full_prefix_child_trace_logits() -> None:
    parent_tensors = {
        "cached_prefill.logits": _logits(5),
        "prefill.logits": _logits(0),
        "decode.000.logits": _logits(6),
        "decode.001.logits": _logits(7),
    }
    child_tensors = {
        "prefill.logits": _logits(5),
        "decode.000.logits": _logits(6),
        "decode.001.logits": _logits(7),
    }

    class CanonicalCache(_GateCache):
        manifest = {
            "capture": {
                "decode_trace_mode": "full_prefix_canonical",
                "forced_decode_use_cache": False,
                "cached_logits_accepted": False,
            }
        }

    rows = _trace_rows(62)
    child_rows = MODULE.TraceRowIndex(
        paths=rows.paths,
        path_splits=rows.path_splits,
        layer_kinds=rows.layer_kinds,
        layer_fields=rows.layer_fields,
        cache_key="5" * 64,
        corpus_fingerprint=rows.corpus_fingerprint,
        trace_rows_sha256="6" * 64,
    )
    report = MODULE.evaluate_trajectory_gate(
        _GateCache(parent_tensors),
        CanonicalCache(child_tensors),
        _GreedyChild(),
        rows,
        child_rows,
        config=MODULE.TrajectoryGateConfig(
            min_mean_residual_cosine=1.0,
            logit_rtol=1.0,
            logit_atol=1.0,
        ),
    )

    assert report["passed"] is True
    assert report["child_trace_mode"] == "full_prefix_canonical"
    assert report["child_trace_logits_source"] == "use_cache_false_full_prefix"
    assert report["child_cached_logits_accepted"] is False
    assert report["teacher_forced"]["top1_agreement"] == 1.0


def test_manual_review_packet_promotes_semantics_not_parent_wording() -> None:
    prompts = (
        MODULE.ReviewPrompt("p1", "Explain why the sky is blue."),
        MODULE.ReviewPrompt("p2", "Give two safe troubleshooting steps."),
    )
    identity = MODULE.review_prompt_identity(prompts)

    def responses(prefix):
        return tuple(
            {
                "id": prompt.prompt_id,
                "prompt": prompt.prompt,
                "context": None,
                "response": f"{prefix} semantically useful response {index}",
                "diagnostics": {
                    "characters": 40,
                    "tokens": 12,
                    "unique_token_ratio": 0.8,
                    "maximum_repeated_bigram_share": 0.1,
                    "empty": False,
                    "fragment_like": False,
                },
            }
            for index, prompt in enumerate(prompts)
        )

    packet = MODULE.build_manual_review_packet(
        layer_index=3,
        candidate_state_sha256="a" * 64,
        parent_fingerprint="b" * 64,
        prompt_identity=identity,
        parent_responses=responses("parent wording"),
        incumbent_responses=responses("incumbent wording"),
        candidate_responses=responses("different candidate wording"),
        internal_diagnostics={
            "teacher_forced": {"top1_agreement": 0.0},
            "free_run": {"token_match_fraction": 0.0},
        },
    )
    prompt_scores = []
    for prompt in prompts:
        prompt_scores.append(
            {
                "id": prompt.prompt_id,
                "scores": {name: 3.0 for name in MODULE.MANUAL_RUBRIC},
                "critical_fails": {
                    name: False for name in MODULE.MANUAL_CRITICAL_FAILS
                },
                "incumbent_semantic_pass": True,
                "candidate_semantic_pass": True,
                "notes": "Different wording, same or better task performance.",
            }
        )
    score = {
        "packet_sha256": packet["packet_sha256"],
        "reviewer": "human-reviewer",
        "decision": "commit",
        "prompt_scores": prompt_scores,
    }
    evaluation = MODULE.evaluate_manual_score(packet, score)

    assert packet["exact_token_match_is_not_a_review_objective"] is True
    assert all(
        row["parent_wording_is_not_a_reference_string"] for row in packet["comparisons"]
    )
    assert evaluation["promotion_suggested"] is True
    assert evaluation["effective_commit"] is True
    assert evaluation["mean_score"] == 3.0
    assert evaluation["median_score"] == 3.0


@pytest.mark.parametrize("interruption", (KeyboardInterrupt, SystemExit))
def test_layer_transaction_base_exception_restores_exact_state_and_objects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    interruption: type[BaseException],
) -> None:
    parent = _TinyLfmCausal(
        _layer_config(intermediate=6, query=16, key_value=8),
        seed=701,
    )
    child = _TinyLfmCausal(
        _layer_config(intermediate=4, query=14, key_value=7),
        seed=702,
    )
    before_state = MODULE.model_state_sha256(child)
    before_parameters = tuple(
        (name, id(parameter))
        for name, parameter in child.named_parameters(remove_duplicate=False)
    )
    producer = SimpleNamespace(layer_index=0)
    split = SimpleNamespace(validation_indices=(0,))

    monkeypatch.setattr(
        MODULE,
        "build_layer_evidence",
        lambda *_args, **_kwargs: (object(), split),
    )
    monkeypatch.setattr(
        MODULE,
        "build_lfm2_closed_loop_fold_plan",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        MODULE,
        "assert_closed_loop_preserves_producer_plan",
        lambda *_args, **_kwargs: None,
    )

    def mutate_then_interrupt(layer, _plan):
        with torch.no_grad():
            next(layer.parameters()).add_(0.25)
        raise interruption("injected layer interruption")

    monkeypatch.setattr(MODULE, "_copy_plan_state", mutate_then_interrupt)

    with pytest.raises(interruption, match="injected layer interruption"):
        MODULE.fold_one_layer_transaction(
            parent,
            child,
            producer,
            object(),
            object(),
            object(),
            object(),
            {},
            (),
            (),
            child_cache_root=tmp_path / "cache",
            capture_config=object(),
            fold_config=object(),
            runtime_config=MODULE.LayerFoldRuntimeConfig(endpoint_refinement_passes=2),
            gate_config=object(),
        )

    after_parameters = tuple(
        (name, id(parameter))
        for name, parameter in child.named_parameters(remove_duplicate=False)
    )
    assert MODULE.model_state_sha256(child) == before_state
    assert after_parameters == before_parameters


def test_runner_source_has_no_training_execution_path() -> None:
    source = inspect.getsource(MODULE)
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "loss.backward" not in source
    parser = MODULE.build_parser()
    help_text = parser.format_help()
    assert "zero-gradient" in help_text.lower()
    assert "225,957,632" in help_text


def test_direct_local_model_and_tokenizer_loaders_forbid_from_pretrained(
    tmp_path, monkeypatch
) -> None:
    import huggingface_hub
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    config = transformers.Lfm2Config(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    torch.manual_seed(707)
    source = transformers.Lfm2ForCausalLM(config).eval()
    (tmp_path / "config.json").write_text(
        json.dumps(config.to_dict()), encoding="utf-8"
    )
    state = source.state_dict()
    stored = {}
    seen = set()
    for name, value in state.items():
        identity = (
            value.untyped_storage().data_ptr(),
            value.storage_offset(),
            tuple(value.shape),
        )
        if identity in seen:
            continue
        seen.add(identity)
        stored[name] = value.detach().cpu().contiguous()
    save_file(stored, str(tmp_path / "model.safetensors"))

    backend = Tokenizer(
        WordLevel(
            {
                "<unk>": 0,
                "<pad>": 1,
                "<bos>": 2,
                "<eos>": 3,
                "hello": 4,
                "world": 5,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    backend.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }
        ),
        encoding="utf-8",
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("from_pretrained must never be called")

    monkeypatch.setattr(
        transformers.PreTrainedModel,
        "from_pretrained",
        classmethod(forbidden),
    )
    monkeypatch.setattr(
        transformers.PreTrainedTokenizerBase,
        "from_pretrained",
        classmethod(forbidden),
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)

    loaded = MODULE.load_local_lfm2_model(tmp_path, device="cpu", dtype=torch.float32)
    tokenizer = MODULE.load_local_tokenizer(tmp_path)
    output_root = tmp_path / "output"
    checkpoint = output_root / "checkpoint"
    saved = MODULE.atomic_save_local_checkpoint(
        loaded,
        checkpoint,
        output_root=output_root,
    )
    reloaded = MODULE.load_local_lfm2_model(
        checkpoint, device="cpu", dtype=torch.float32
    )
    io_audit = MODULE.direct_local_io_audit(tmp_path)

    assert MODULE.model_state_sha256(loaded) == MODULE.model_state_sha256(source)
    assert MODULE.model_state_sha256(reloaded) == MODULE.model_state_sha256(source)
    assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight
    assert tokenizer.encode("hello world", add_special_tokens=False) == [4, 5]
    assert saved["no_parameter_sidecars"] is True
    assert io_audit["network_allowed"] is False
    assert io_audit["hub_download_used"] is False
    assert io_audit["from_pretrained_used"] is False
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert ".from_pretrained" not in inspect.getsource(MODULE)
    assert ".save_pretrained" not in inspect.getsource(MODULE)


# --- streaming sealed-evidence merge ---------------------------------------


def _merge_capture_config():
    from neural_forge.hf_lfm2_circuit_trace_cache import (
        Lfm2CircuitTraceCaptureConfig,
    )

    return Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=4,
        max_decode_steps=8,
        max_capture_bytes=64 * 1024 * 1024,
        logit_rtol=0.2,
        logit_atol=1.0,
        require_top1_agreement=False,
    )


def _merge_batch(prompt, decode):
    return {
        "input_ids": torch.tensor([prompt], dtype=torch.long),
        "attention_mask": torch.ones((1, len(prompt)), dtype=torch.long),
        "forced_decode_ids": torch.tensor([decode], dtype=torch.long),
    }


@pytest.fixture(scope="module")
def merge_lane(tmp_path_factory):
    """One sealed base corpus plus one sealed extension corpus, real caches."""

    from neural_forge.hf_lfm2_circuit_trace_cache import (
        build_lfm2_circuit_trace_cache,
    )

    root = tmp_path_factory.mktemp("lfm2-merge-into")
    config = transformers.Lfm2Config(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4321)
        parent = transformers.Lfm2ForCausalLM(config).eval()
    parent.requires_grad_(False)
    capture = _merge_capture_config()
    base = build_lfm2_circuit_trace_cache(
        root / "base",
        parent,
        (
            _merge_batch([2, 4, 5], [6, 7]),
            _merge_batch([2, 5, 6], [7, 8]),
            _merge_batch([2, 6, 7], [8, 4]),
        ),
        config=capture,
        input_device="cpu",
    )
    extension = build_lfm2_circuit_trace_cache(
        root / "extension",
        parent,
        (_merge_batch([4, 5, 6], [7, 8]), _merge_batch([5, 6, 7], [8, 4])),
        config=capture,
        input_device="cpu",
    )
    return SimpleNamespace(root=root, parent=parent, base=base, extension=extension)


def _merge_members(lane, base_splits, extension_splits, sessions):
    return (
        MODULE.SealedEvidenceMergeMember(
            cache=lane.base,
            selective_session=sessions[0],
            batch_splits=base_splits,
            batch_offset=0,
        ),
        MODULE.SealedEvidenceMergeMember(
            cache=lane.extension,
            selective_session=sessions[1],
            batch_splits=extension_splits,
            batch_offset=len(lane.base),
        ),
    )


def test_merge_trace_rows_into_matches_a_separate_join(merge_lane) -> None:
    """The streamed merge is the concatenation, tensor for tensor."""

    lane = merge_lane
    base_splits = ("selection", "fit", "validation")
    extension_splits = ("fit", "fit")
    sessions = [
        lane.base.selective_authenticated_session(),
        lane.extension.selective_authenticated_session(),
    ]
    try:
        base_rows = MODULE.extract_trace_rows(
            lane.base,
            batch_splits=base_splits,
            selective_session=sessions[0],
        )
        extension_rows = MODULE.extract_trace_rows(
            lane.extension,
            batch_splits=extension_splits,
            selective_session=sessions[1],
        )
        merged = MODULE.merge_trace_rows_into(
            _merge_members(lane, base_splits, extension_splits, sessions),
            cache_key="a" * 64,
            corpus_fingerprint="b" * 64,
            merged_trace_rows_sha256=lambda members, paths, splits: "c" * 64,
        )
    finally:
        for session in sessions:
            session.close()

    assert merged.rows.cache_key == "a" * 64
    assert merged.rows.corpus_fingerprint == "b" * 64
    assert merged.rows.trace_rows_sha256 == "c" * 64
    assert merged.rows.layer_kinds == base_rows.layer_kinds
    assert merged.rows.path_splits == (
        base_rows.path_splits + extension_rows.path_splits
    )
    # Extension paths are renumbered into merged batch order; the base prefix
    # keeps its own numbering because its offset is zero.
    assert merged.rows.paths[: len(base_rows.paths)] == base_rows.paths
    renumbered = []
    for path in extension_rows.paths:
        head, _separator, tail = path.partition("/")
        index = int(head.removeprefix("batch.")) + len(lane.base)
        renumbered.append(f"batch.{index:06d}/{tail}")
    assert merged.rows.paths[len(base_rows.paths) :] == tuple(renumbered)
    for layer_index in sorted(base_rows.layer_fields):
        names = sorted(base_rows.layer_fields[layer_index])
        assert sorted(merged.rows.layer_fields[layer_index]) == names
        for name in names:
            joined = merged.rows.field(layer_index, name)
            assert torch.equal(
                joined,
                torch.cat(
                    (
                        base_rows.field(layer_index, name),
                        extension_rows.field(layer_index, name),
                    ),
                    dim=0,
                ),
            )

    # Every member view is the member's own standalone identity, computed over
    # its own slice, so a merged index can name what produced each row.
    assert len(merged.member_rows) == 2
    assert merged.base_rows is merged.member_rows[0]
    assert merged.member_rows[0].trace_rows_sha256 == (
        base_rows.trace_rows_sha256
    )
    assert merged.member_rows[1].trace_rows_sha256 == (
        extension_rows.trace_rows_sha256
    )
    assert merged.member_rows[0].cache_key == lane.base.cache_key
    assert merged.member_rows[1].cache_key == lane.extension.cache_key
    assert merged.member_rows[0].paths == base_rows.paths
    assert merged.member_rows[1].paths == extension_rows.paths


def test_merge_trace_rows_into_never_holds_two_copies(merge_lane) -> None:
    """Preallocate once, fill from members, release each source as consumed."""

    lane = merge_lane
    base_splits = ("selection", "fit", "validation")
    extension_splits = ("fit", "fit")
    sessions = [
        lane.base.selective_authenticated_session(),
        lane.extension.selective_authenticated_session(),
    ]
    # The discarded implementation joined per-layer fields with ``torch.cat``
    # while every member index was still referenced.  Making that call fail is
    # the direct proof that the streaming merge never performs it.
    real_cat = torch.cat

    def refuse_cat(*args, **kwargs):
        raise AssertionError("merge_trace_rows_into must not concatenate rows")

    try:
        torch.cat = refuse_cat
        try:
            merged = MODULE.merge_trace_rows_into(
                _merge_members(lane, base_splits, extension_splits, sessions),
                cache_key="a" * 64,
                corpus_fingerprint="b" * 64,
                merged_trace_rows_sha256=lambda members, paths, splits: "c" * 64,
            )
        finally:
            torch.cat = real_cat
    finally:
        for session in sessions:
            session.close()

    total_rows = len(merged.rows.paths)
    base_rows = len(merged.member_rows[0].paths)
    for layer_index, fields in merged.rows.layer_fields.items():
        for name, tensor in fields.items():
            # One allocation per layer field, sized for the merged corpus.
            assert int(tensor.shape[0]) == total_rows
            for ordinal, member in enumerate(merged.member_rows):
                view = member.field(layer_index, name)
                # Member views are windows into that single allocation, so no
                # member index is a second retained copy of the rows.
                assert (
                    view.untyped_storage().data_ptr()
                    == tensor.untyped_storage().data_ptr()
                )
                assert int(view.shape[0]) == len(member.paths)
                assert view.storage_offset() == (
                    0
                    if ordinal == 0
                    else base_rows * tensor.stride(0)
                )

    accounting = merged.accounting
    assert accounting["schema"] == MODULE.MERGED_TRACE_ROWS_ACCOUNTING_SCHEMA
    assert accounting["preallocated_once"] is True
    assert accounting["member_sources_released_as_consumed"] is True
    assert accounting["member_rows_are_zero_copy_views"] is True
    assert accounting["parent_forward_calls"] == 0
    assert accounting["merged_row_count"] == total_rows
    assert accounting["member_row_counts"] == [
        len(value.paths) for value in merged.member_rows
    ]
    retained = sum(
        int(tensor.numel()) * int(tensor.element_size())
        for fields in merged.rows.layer_fields.values()
        for tensor in fields.values()
    )
    assert accounting["retained_bytes"] == retained
    assert accounting["separate_materialization_peak_bytes"] == 2 * retained
    assert accounting["streaming_merge_peak_bytes"] == (
        retained + accounting["maximum_transient_selection_bytes"]
    )
    assert accounting["streaming_merge_peak_bytes"] < (
        accounting["separate_materialization_peak_bytes"]
    )
    assert accounting["peak_bytes_avoided"] == (
        retained - accounting["maximum_transient_selection_bytes"]
    )
    assert 1.0 < accounting["peak_bytes_ratio_to_retained"] < 2.0


def test_merge_trace_rows_into_fails_closed_on_a_broken_member(
    merge_lane,
) -> None:
    lane = merge_lane
    sessions = [
        lane.base.selective_authenticated_session(),
        lane.extension.selective_authenticated_session(),
    ]
    try:
        with pytest.raises(
            MODULE.Lfm2ResidualAttentionRunnerError,
            match="cumulative merged batch order",
        ):
            MODULE.merge_trace_rows_into(
                (
                    MODULE.SealedEvidenceMergeMember(
                        cache=lane.base,
                        selective_session=sessions[0],
                        batch_splits=("fit", "fit", "fit"),
                        batch_offset=0,
                    ),
                    MODULE.SealedEvidenceMergeMember(
                        cache=lane.extension,
                        selective_session=sessions[1],
                        batch_splits=("fit", "fit"),
                        batch_offset=99,
                    ),
                ),
                cache_key="a" * 64,
                corpus_fingerprint="b" * 64,
                merged_trace_rows_sha256=lambda members, paths, splits: "c" * 64,
            )
        with pytest.raises(
            MODULE.Lfm2ResidualAttentionRunnerError,
            match="split assignment does not cover its cache",
        ):
            MODULE.SealedEvidenceMergeMember(
                cache=lane.base,
                selective_session=sessions[0],
                batch_splits=("fit",),
                batch_offset=0,
            )
        with pytest.raises(
            MODULE.Lfm2ResidualAttentionRunnerError,
            match="requires its authenticated selective session",
        ):
            MODULE.SealedEvidenceMergeMember(
                cache=lane.base,
                selective_session=None,
                batch_splits=("fit", "fit", "fit"),
                batch_offset=0,
            )
        with pytest.raises(
            MODULE.Lfm2ResidualAttentionRunnerError,
            match="lowercase sha256 digest",
        ):
            MODULE.merge_trace_rows_into(
                _merge_members(
                    lane,
                    ("fit", "fit", "fit"),
                    ("fit", "fit"),
                    sessions,
                ),
                cache_key="a" * 64,
                corpus_fingerprint="b" * 64,
                merged_trace_rows_sha256=lambda members, paths, splits: "nope",
            )
    finally:
        for session in sessions:
            session.close()
