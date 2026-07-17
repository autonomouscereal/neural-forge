from __future__ import annotations

import hashlib
from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn
pytest.importorskip("transformers")

from neural_forge.hf_lfm2_closed_loop_fold import (  # noqa: E402
    Lfm2ClosedLoopFoldConfig,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_producer_first_model_fold import (  # noqa: E402
    Lfm2ProducerFirstCapture,
    Lfm2ProducerFirstEvidence,
    Lfm2ProducerFirstFoldError,
    Lfm2ProducerFirstFoldConfig,
    install_lfm2_producer_first_model_fold,
)


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
        self.feed_forward = Lfm2MLP(
            config.hidden_size, config.intermediate_size
        )
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


class _Config(SimpleNamespace):
    def to_dict(self):
        return dict(vars(self))


def _config(*, intermediate: int, query: int, key_value: int):
    return _Config(
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


class _Rows:
    def __init__(
        self,
        fields,
        *,
        trace: str,
        query: int,
        key_value: int,
        intermediate: int,
    ):
        del query, key_value, intermediate
        self.paths = tuple(f"row.{index:03d}" for index in range(18))
        self.path_splits = (
            *("selection" for _ in range(6)),
            *("fit" for _ in range(6)),
            *("validation" for _ in range(6)),
        )
        self.layer_kinds = ("full_attention",)
        self.layer_fields = MappingProxyType(
            {0: MappingProxyType(dict(fields))}
        )
        self.cache_key = "a" * 64
        self.corpus_fingerprint = "b" * 64
        self.trace_rows_sha256 = trace

    def field(self, layer_index: int, name: str):
        return self.layer_fields[layer_index][name]


def _rows(
    seed: int,
    *,
    query: int,
    key_value: int,
    intermediate: int,
    trace: str,
):
    generator = torch.Generator().manual_seed(seed)
    random = lambda *shape: torch.randn(  # noqa: E731
        *shape, generator=generator, dtype=torch.float64
    )
    return _Rows(
        {
            "attention.q": random(18, query, 2),
            "attention.k": random(18, key_value, 2),
            "attention.v": random(18, key_value, 2),
            "attention.context": random(18, query, 2),
            "mlp.gate": random(18, intermediate),
            "mlp.up": random(18, intermediate),
            "mlp.post_swiglu": random(18, intermediate),
            "residual.pre": random(18, 8),
            "residual.post_operator": random(18, 8),
            "residual.post_layer": random(18, 8),
        },
        trace=trace,
        query=query,
        key_value=key_value,
        intermediate=intermediate,
    )


def _state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _assert_state(model, expected):
    observed = model.state_dict()
    assert set(observed) == set(expected)
    for name, value in expected.items():
        assert torch.equal(observed[name].detach().cpu(), value), name


def _evidence(parent, parent_rows, captures, blank_state, *, fail_at=None):
    def capture(child):
        index = len(captures)
        assert any(
            not torch.equal(value.detach().cpu(), blank_state[name])
            for name, value in child.state_dict().items()
        )
        captures.append(_state(child))
        if fail_at is not None and index == fail_at:
            raise RuntimeError("synthetic capture failure")
        rows = _rows(
            200 + index,
            query=14,
            key_value=7,
            intermediate=4,
            trace=hashlib.sha256(f"capture-{index}".encode()).hexdigest(),
        )
        return Lfm2ProducerFirstCapture(
            rows=rows,
            cache_key=hashlib.sha256(f"cache-{index}".encode()).hexdigest(),
            child_forward_count=1,
        )

    return Lfm2ProducerFirstEvidence(
        parent_rows=parent_rows,
        selection_indices=tuple(range(6)),
        fit_indices=tuple(range(6, 12)),
        validation_indices=tuple(range(12, 18)),
        capture_child=capture,
        parent_cache_fingerprint="a" * 64,
        parent_identity_sha256=(
            fingerprint_lfm2_parent(parent).identity_sha256
        ),
        capture_builder_id="test.synthetic.capture.v1",
    )


def _fold_config():
    return Lfm2ProducerFirstFoldConfig(
        layer_config=Lfm2ClosedLoopFoldConfig(
            max_endpoint_relative_rmse=1e6,
            min_endpoint_cosine_similarity=-1.0,
        ),
        endpoint_refinement_passes=2,
    )


def test_blank_features_are_never_used_and_candidates_are_recaptured() -> None:
    parent = _TinyLfmCausal(
        _config(intermediate=6, query=16, key_value=8), seed=10
    )
    child = _TinyLfmCausal(
        _config(intermediate=4, query=14, key_value=7), seed=11
    )
    blank = _state(child)
    captures = []
    transaction = install_lfm2_producer_first_model_fold(
        parent,
        child,
        _evidence(
            parent,
            _rows(
                100,
                query=16,
                key_value=8,
                intermediate=6,
                trace="c" * 64,
            ),
            captures,
            blank,
        ),
        config=_fold_config(),
    )

    assert transaction.active is True
    assert len(captures) == 3
    report = transaction.plan.as_dict()
    assert report["blank_features_used"] is False
    assert report["producer_install_report"]["blank_values_read"] is False
    assert all(
        row["kind"] in {
            "complete_producer_state",
            "accepted_layer_refinement",
        }
        for row in report["capture_reports"]
    )
    assert all(
        fold["candidate_features_recaptured"]
        for layer in report["layer_reports"]
        for fold in layer["passes"]
    )
    transaction.rollback()
    _assert_state(child, blank)


def test_capture_failure_restores_the_entire_blank_model() -> None:
    parent = _TinyLfmCausal(
        _config(intermediate=6, query=16, key_value=8), seed=20
    )
    child = _TinyLfmCausal(
        _config(intermediate=4, query=14, key_value=7), seed=21
    )
    blank = _state(child)
    with pytest.raises(RuntimeError, match="synthetic capture failure"):
        install_lfm2_producer_first_model_fold(
            parent,
            child,
            _evidence(
                parent,
                _rows(
                    101,
                    query=16,
                    key_value=8,
                    intermediate=6,
                    trace="d" * 64,
                ),
                [],
                blank,
                fail_at=1,
            ),
            config=_fold_config(),
        )
    _assert_state(child, blank)


def test_evidence_rejects_parent_rows_cache_key_mismatch() -> None:
    parent = _TinyLfmCausal(
        _config(intermediate=6, query=16, key_value=8), seed=30
    )
    child = _TinyLfmCausal(
        _config(intermediate=4, query=14, key_value=7), seed=31
    )
    blank = _state(child)
    parent_rows = _rows(
        102,
        query=16,
        key_value=8,
        intermediate=6,
        trace="e" * 64,
    )
    parent_rows.cache_key = "f" * 64

    with pytest.raises(
        Lfm2ProducerFirstFoldError,
        match=r"parent_rows\.cache_key must exactly match",
    ):
        _evidence(parent, parent_rows, [], blank)

    _assert_state(child, blank)
    assert not hasattr(child, "_neural_forge_producer_plan_fingerprint")


def test_wrong_live_parent_with_identical_topology_is_rejected_without_mutation() -> None:
    sealed_parent = _TinyLfmCausal(
        _config(intermediate=6, query=16, key_value=8), seed=40
    )
    wrong_parent = _TinyLfmCausal(
        _config(intermediate=6, query=16, key_value=8), seed=41
    )
    child = _TinyLfmCausal(
        _config(intermediate=4, query=14, key_value=7), seed=42
    )
    blank = _state(child)
    captures = []
    evidence = _evidence(
        sealed_parent,
        _rows(
            103,
            query=16,
            key_value=8,
            intermediate=6,
            trace="f" * 64,
        ),
        captures,
        blank,
    )

    with pytest.raises(
        Lfm2ProducerFirstFoldError,
        match="different exact loaded LFM2 parent",
    ):
        install_lfm2_producer_first_model_fold(
            wrong_parent,
            child,
            evidence,
            config=_fold_config(),
        )

    assert captures == []
    _assert_state(child, blank)
    assert not hasattr(child, "_neural_forge_producer_plan_fingerprint")


@pytest.mark.parametrize(
    "tamper",
    ("rows_cache_key", "row_tensor", "evidence_parent_identity"),
)
def test_post_construction_tamper_is_rejected_without_mutation(
    tamper: str,
) -> None:
    parent = _TinyLfmCausal(
        _config(intermediate=6, query=16, key_value=8), seed=50
    )
    child = _TinyLfmCausal(
        _config(intermediate=4, query=14, key_value=7), seed=51
    )
    blank = _state(child)
    captures = []
    parent_rows = _rows(
        104,
        query=16,
        key_value=8,
        intermediate=6,
        trace="1" * 64,
    )
    evidence = _evidence(parent, parent_rows, captures, blank)
    if tamper == "rows_cache_key":
        parent_rows.cache_key = "2" * 64
    elif tamper == "row_tensor":
        with torch.no_grad():
            parent_rows.layer_fields[0]["mlp.gate"][0, 0].add_(1.0)
    else:
        object.__setattr__(evidence, "parent_identity_sha256", "3" * 64)

    with pytest.raises(Lfm2ProducerFirstFoldError):
        install_lfm2_producer_first_model_fold(
            parent,
            child,
            evidence,
            config=_fold_config(),
        )

    assert captures == []
    _assert_state(child, blank)
    assert not hasattr(child, "_neural_forge_producer_plan_fingerprint")
