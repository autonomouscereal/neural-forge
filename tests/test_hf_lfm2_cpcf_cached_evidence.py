from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_cpcf_cached_evidence import (  # noqa: E402
    Lfm2CpcfCachedEvidenceAdapter,
    Lfm2CpcfCachedEvidenceError,
    Lfm2CpcfEvidenceConfig,
    audit_lfm2_cpcf_predictive_rank,
    build_lfm2_cpcf_boundary_evidence,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class _FakeBatch:
    index: int
    input_sha256: str
    tensors: dict[str, Any]
    record: dict[str, Any]


class _FakeSealedCache:
    def __init__(self, batches: list[_FakeBatch]) -> None:
        self.cache_key = _digest("sealed-cache")
        self.batch_count = len(batches)
        self._batches = tuple(batches)
        self.verify_count = 0
        self.load_count = 0
        self.parent_forward_count = 0
        self.manifest = {
            "corpus": {
                "ordered_batch_input_sha256": [
                    batch.input_sha256 for batch in batches
                ],
            },
            "layer_topology": {
                "layer_topology_sha256": _digest("topology"),
                "layers": [
                    {
                        "contract": {
                            "layer_index": 0,
                            "kind": "short_conv",
                            "hidden_size": 4,
                            "ffn_intermediate_size": 6,
                            "operator": {
                                "fir_length": 3,
                                "input_projection_chunk_order": [
                                    "b",
                                    "c",
                                    "x",
                                ],
                                "not_attention_qkv": True,
                            },
                        },
                    },
                    {
                        "contract": {
                            "layer_index": 1,
                            "kind": "full_attention",
                            "hidden_size": 4,
                            "ffn_intermediate_size": 6,
                            "operator": {
                                "query_heads": 2,
                                "key_value_heads": 1,
                                "head_dim": 2,
                                "qk_rmsnorm": True,
                            },
                        },
                    },
                ],
            },
        }

    def verify(self) -> "_FakeSealedCache":
        self.verify_count += 1
        return self

    def load_batch(self, index: int) -> _FakeBatch:
        self.load_count += 1
        return self._batches[index]

    def forward(self, *_args, **_kwargs):
        self.parent_forward_count += 1
        raise AssertionError("The CPCF cache adapter executed a parent.")


def _pattern(sequence: int, *, scale: float) -> Any:
    base = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    repeats = (sequence + 3) // 4
    return (base.repeat(repeats, 1)[:sequence] * scale).unsqueeze(0)


def _mlp_rows(state: Any) -> tuple[Any, Any, Any, Any]:
    source = state[..., :2]
    gate = torch.cat((source, -source, source), dim=2)
    up = torch.ones_like(gate)
    post = gate.clone()
    down = state * 0.10
    return gate, up, post, down


def _logits(state: Any) -> Any:
    a = state[..., 0]
    b = state[..., 1]
    return torch.stack(
        (
            5.0 + 0.10 * a,
            3.5 + 0.10 * b,
            2.0 - 0.05 * a,
            1.0 - 0.05 * b,
            -1.0 + 0.02 * a,
            -2.0 + 0.02 * b,
        ),
        dim=2,
    )


def _attention_paths(state: Any) -> dict[str, Any]:
    a = state[..., 0]
    b = state[..., 1]
    query_zero = torch.stack((a, b), dim=2)
    query_one = torch.stack((b, -a), dim=2)
    query = torch.stack((query_zero, query_one), dim=1)
    key = torch.zeros(
        (int(state.shape[0]), 1, int(state.shape[1]), 2),
        dtype=state.dtype,
    )
    key[:, :, :, 0] = 1.0
    value = query_zero.unsqueeze(1)
    return {
        "attention.q_linear": query.clone(),
        "attention.q": query,
        "attention.k_linear": key.clone(),
        "attention.k": key,
        "attention.v": value,
        "attention.context": query.clone(),
    }


def _phase_tensors(phase: str, state: Any) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    layer_zero = state
    gate, up, post, down = _mlp_rows(layer_zero)
    operator = layer_zero * 0.20
    post_operator = layer_zero + operator
    post_layer = post_operator + down
    prefix = f"{phase}.layer.000."
    tensors.update(
        {
            prefix + "residual.pre": layer_zero,
            prefix + "operator.input": layer_zero,
            prefix + "operator.output": operator,
            prefix + "residual.post_operator": post_operator,
            prefix + "mlp.input": post_operator,
            prefix + "mlp.gate": gate,
            prefix + "mlp.up": up,
            prefix + "mlp.post_swiglu": post,
            prefix + "mlp.down": down,
            prefix + "residual.post_layer": post_layer,
            prefix + "short_conv.b": layer_zero,
            prefix + "short_conv.c": layer_zero * 0.5,
            prefix + "short_conv.x": torch.ones_like(layer_zero),
            prefix + "short_conv.bx": layer_zero,
            prefix + "short_conv.context": layer_zero * 0.25,
            prefix + "short_conv.o": operator,
        }
    )

    layer_one = post_layer
    gate, up, post, down = _mlp_rows(layer_one)
    operator = layer_one * 0.15
    post_operator = layer_one + operator
    post_layer = post_operator + down
    prefix = f"{phase}.layer.001."
    tensors.update(
        {
            prefix + "residual.pre": layer_one,
            prefix + "operator.input": layer_one,
            prefix + "operator.output": operator,
            prefix + "residual.post_operator": post_operator,
            prefix + "mlp.input": post_operator,
            prefix + "mlp.gate": gate,
            prefix + "mlp.up": up,
            prefix + "mlp.post_swiglu": post,
            prefix + "mlp.down": down,
            prefix + "residual.post_layer": post_layer,
            prefix + "attention.o": operator,
        }
    )
    for name, value in _attention_paths(layer_one).items():
        tensors[prefix + name] = value
    tensors[f"{phase}.logits"] = _logits(post_layer)
    return tensors


def _cache(*, batch_count: int = 6) -> _FakeSealedCache:
    batches = []
    for batch_index in range(batch_count):
        tensors: dict[str, Any] = {}
        tensors.update(
            _phase_tensors(
                "prefill",
                _pattern(4, scale=1.0 + batch_index * 0.05),
            )
        )
        for step in range(4):
            tensors.update(
                _phase_tensors(
                    f"decode.{step:03d}",
                    _pattern(
                        4,
                        scale=(1.0 + batch_index * 0.05)
                        * (1.0 if step % 2 == 0 else -1.0),
                    )[:, step : step + 1, :],
                )
            )
        batches.append(
            _FakeBatch(
                index=batch_index,
                input_sha256=_digest(f"batch-{batch_index}"),
                tensors=tensors,
                record={"capture": {"decode_steps": 4}},
            )
        )
    return _FakeSealedCache(batches)


def _config() -> Lfm2CpcfEvidenceConfig:
    return Lfm2CpcfEvidenceConfig(
        validation_fraction=1 / 3,
        ridge=1e-8,
        producer_sketch_width=4,
        logit_topk=4,
        logit_identity_sketch_width=4,
    )


def test_adapter_builds_whole_batch_disjoint_evidence_for_each_layer() -> None:
    cache = _cache()
    adapter = Lfm2CpcfCachedEvidenceAdapter(cache, config=_config())
    boundaries = tuple(adapter.iter_boundaries())

    assert [value.layer_index for value in boundaries] == [0, 1]
    assert [value.operator_kind for value in boundaries] == [
        "short_conv",
        "full_attention",
    ]
    assert set(adapter.split.fit_batch_indices).isdisjoint(
        adapter.split.validation_batch_indices
    )
    for boundary in boundaries:
        assert set(boundary.fit_row_ids).isdisjoint(
            boundary.validation_row_ids
        )
        assert boundary.fit_rows == 32
        assert boundary.validation_rows == 16
        assert boundary.fit_weights.sum().item() == pytest.approx(32.0)
        assert boundary.validation_weights.sum().item() == pytest.approx(
            16.0
        )
        assert "action.logit_margins" in {
            block.name for block in boundary.fit_features
        }
    assert "action.short_conv_bcx_path" in {
        block.name for block in boundaries[0].fit_features
    }
    assert "action.attention_qk_scores" in {
        block.name for block in boundaries[1].fit_features
    }
    attention_contract = boundaries[1].feature_contracts[
        "action.attention_qk_scores"
    ]
    assert attention_contract["post_rope_observability_only"] is True
    assert (
        attention_contract["installable_pre_rope_qk_factor_claimed"]
        is False
    )
    assert cache.parent_forward_count == 0


def test_predictive_rank_audit_reports_minimum_width_and_validation() -> None:
    cache = _cache()
    result = audit_lfm2_cpcf_predictive_rank(
        cache,
        config=_config(),
        candidate_widths=(1, 2, 4),
    )

    assert len(result.boundaries) == 2
    assert all(
        1 <= boundary.minimum_width_at_90_predictive_energy <= 2
        for boundary in result.boundaries
    )
    assert result.minimum_uniform_width_at_90_predictive_energy <= 2
    assert result.candidate_widths[2][
        "feasible_at_predictive_energy_floor"
    ] is True
    assert result.candidate_widths[4][
        "feasible_at_predictive_energy_floor"
    ] is True
    for boundary in result.boundaries:
        assert math_is_finite(
            boundary.validation_state_retained_energy_fraction
        )
        assert math_is_finite(
            boundary.minimum_validation_feature_retention
        )
        assert boundary.validation_blocks
    assert result.full_compilation_ready is False
    assert result.as_dict()["measurement_only"] is True
    assert result.as_dict()["installability_contract_evaluated"] is False
    assert result.as_dict()["parent_side_compiler_inputs_ready"] is False
    assert result.as_dict()["operation_audit"] == {
        "parent_forward_count": 0,
        "parent_generation_count": 0,
        "parent_cache_regeneration_count": 0,
        "training_step_count": 0,
        "backward_call_count": 0,
        "optimizer_step_count": 0,
    }
    assert cache.verify_count == 2
    assert cache.load_count == cache.batch_count * 2
    assert cache.parent_forward_count == 0


def math_is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def test_full_compilation_gap_names_rope_and_fir_evidence() -> None:
    adapter = Lfm2CpcfCachedEvidenceAdapter(_cache(), config=_config())
    gaps = adapter.missing_cache_fields_for_full_compilation
    missing = {
        field
        for gap in gaps
        for field in gap["missing_cache_fields"]
    }

    assert "attention.q_normalized_pre_rope" in missing
    assert "attention.rope_cos" in missing
    assert "attention.rope_sin" in missing
    assert "attention.full_causal_score_matrix" in missing
    assert "short_conv.kernel_weights" in missing
    assert "model.final_norm.output" in missing
    attention_gap = next(
        gap for gap in gaps if gap["scope"] == "attention_qk_installability"
    )
    assert "shared" in attention_gap["reason"]
    assert "pre-normalization/pre-RoPE" in attention_gap["reason"]


def test_missing_consumed_cache_field_fails_closed() -> None:
    cache = _cache()
    del cache._batches[0].tensors[
        "prefill.layer.001.attention.context"
    ]
    adapter = Lfm2CpcfCachedEvidenceAdapter(cache, config=_config())

    with pytest.raises(
        Lfm2CpcfCachedEvidenceError,
        match="attention.context",
    ):
        adapter.boundary(1)
    assert cache.parent_forward_count == 0


def test_one_cache_batch_cannot_create_disjoint_validation() -> None:
    with pytest.raises(
        Lfm2CpcfCachedEvidenceError,
        match="at least two complete cache batches",
    ):
        Lfm2CpcfCachedEvidenceAdapter(_cache(batch_count=1), config=_config())


def test_selectable_single_boundary_entrypoint_does_not_audit_other_layers() -> None:
    cache = _cache()
    result = build_lfm2_cpcf_boundary_evidence(
        cache,
        layer_index=0,
        config=_config(),
    )

    assert result.layer_index == 0
    assert result.operator_kind == "short_conv"
    assert cache.load_count == cache.batch_count
    assert cache.parent_forward_count == 0
