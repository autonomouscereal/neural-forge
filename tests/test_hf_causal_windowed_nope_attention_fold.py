"""Windowed / NoPE policy flags on the coupled GQA attention kernel.

Two things have to hold at once, exactly as they did when the per-head Q/K
normalization lane was added.  A chimera ``local`` branch -- softmax GQA with a
hard sliding window and no rotation at all
(``chimera-omni/src/chimera_omni/modeling/layers.py:26-57``) -- must compile
through this kernel with the *target architecture's own* semantics.  And a
target that declares neither flag must keep the exact bytes it had before the
lane existed: no new receipt keys, no new topology keys, no changed solve.

Stage 0 of ``docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md`` section 5.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_causal_gqa_attention_fold import (  # noqa: E402
    HFCausalGQAAttentionFoldError,
    HFCausalGQATopology,
    apply_rope,
    causal_attention_probabilities,
    compile_hf_causal_gqa_attention_fold,
    invert_rope,
)


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_chimera_omni_toy_fold.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_chimera_omni_toy_fold", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()

try:
    _REPO = runner.resolve_chimera_omni_repo()
    _CHIMERA = runner.import_chimera_omni(_REPO)
except runner.ChimeraOmniFoldError as exc:  # pragma: no cover - environment gate
    pytest.skip(f"chimera-omni checkout unavailable: {exc}", allow_module_level=True)

chimera_config = _CHIMERA["config"]
chimera_layers = sys.modules["chimera_omni.modeling.layers"]


HIDDEN = 32
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
WINDOW = 6
BATCH = 8
SEQUENCE = 24


def _config(**changes):
    body = {
        "vocab_size": 32,
        "hidden_size": HIDDEN,
        "intermediate_size": HIDDEN * 2,
        "num_layers": 2,
        "num_heads": HEADS,
        "num_kv_heads": KV_HEADS,
        "max_sequence_length": 64,
        "block_pattern": ("local",),
        "residual_mode": "none",
        "local_window": WINDOW,
    }
    body.update(changes)
    return chimera_config.ChimeraConfig(**body)


def _local_attention(seed: int):
    torch.manual_seed(seed)
    module = chimera_layers.LocalCausalAttention(_config())
    module.eval()
    return module


def _rows(seed: int):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, SEQUENCE, HIDDEN, generator=generator)


def _nope_topology(**changes) -> HFCausalGQATopology:
    body = {
        "query_heads": HEADS,
        "key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "rope_theta": 10000.0,
        "attention_scaling": 1.0 / math.sqrt(HEAD_DIM),
        "attention_window": WINDOW,
        "position_encoding": "nope",
    }
    body.update(changes)
    return HFCausalGQATopology(**body)


def _heads(flat, count):
    return (
        flat.detach()
        .to(dtype=torch.float64)
        .reshape(BATCH, SEQUENCE, count, HEAD_DIM)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _relative_rmse(prediction, target) -> float:
    numerator = (prediction - target).square().mean().sqrt()
    denominator = target.square().mean().sqrt().clamp_min(1e-30)
    return float((numerator / denominator).item())


# ---------------------------------------------------------------------------
# The defaults are unchanged, byte for byte
# ---------------------------------------------------------------------------


def test_a_default_topology_serializes_to_its_pre_lane_bytes() -> None:
    topology = HFCausalGQATopology(
        query_heads=4,
        key_value_heads=2,
        head_dim=8,
        rope_theta=10000.0,
        attention_scaling=0.125,
    )
    pre_lane = {
        "query_heads": 4,
        "key_value_heads": 2,
        "head_dim": 8,
        "rope_theta": 10000.0,
        "attention_scaling": 0.125,
        "query_key_norm": "identity",
        "rms_norm_epsilon": 1e-6,
    }

    assert topology.attention_window is None
    assert topology.position_encoding == "rope"
    assert topology.as_dict() == pre_lane
    assert canonical_sha256(topology.as_dict()) == canonical_sha256(pre_lane)


def test_a_default_receipt_carries_no_window_or_encoding_keys() -> None:
    generator = torch.Generator().manual_seed(3)
    child = torch.randn(2, 12, 16, generator=generator, dtype=torch.float64)
    topology = HFCausalGQATopology(
        query_heads=2,
        key_value_heads=1,
        head_dim=4,
        rope_theta=10000.0,
        attention_scaling=0.5,
    )
    mask = torch.ones(2, 12, dtype=torch.bool)
    fit = torch.zeros(2, 12, dtype=torch.bool)
    fit[:, :8] = True
    validation = ~fit
    fold = compile_hf_causal_gqa_attention_fold(
        child,
        source_query_sequences=torch.randn(2, 12, 8, generator=generator, dtype=torch.float64),
        source_key_sequences=torch.randn(2, 12, 4, generator=generator, dtype=torch.float64),
        source_value_sequences=torch.randn(2, 12, 4, generator=generator, dtype=torch.float64),
        target_residual_delta_sequences=torch.randn(
            2, 12, 16, generator=generator, dtype=torch.float64
        ),
        attention_mask=mask,
        position_ids=torch.arange(12).reshape(1, -1).expand(2, 12),
        valid_token_mask=mask,
        fit_token_mask=fit,
        validation_token_mask=validation,
        source_topology=topology,
        target_topology=topology,
    )

    for key in ("attention_window", "position_encoding", "rope_applied"):
        assert key not in fold.report
    assert "attention_window" not in fold.report["target_topology"]


def test_nope_makes_the_rotations_identity_passthroughs() -> None:
    generator = torch.Generator().manual_seed(9)
    value = torch.randn(2, 3, 5, 8, generator=generator, dtype=torch.float64)
    positions = torch.arange(5).reshape(1, -1).expand(2, 5)

    forward = apply_rope(value, positions, theta=10000.0, position_encoding="nope")
    inverse = invert_rope(value, positions, theta=10000.0, position_encoding="nope")

    assert torch.equal(forward, value)
    assert torch.equal(inverse, value)
    assert not torch.equal(
        apply_rope(value, positions, theta=10000.0), value
    )


def test_nope_admits_an_odd_head_dimension_and_rope_does_not() -> None:
    topology = _nope_topology(head_dim=5, attention_scaling=0.4)

    assert topology.head_dim == 5

    with pytest.raises(HFCausalGQAAttentionFoldError) as error:
        _nope_topology(head_dim=5, attention_scaling=0.4, position_encoding="rope")
    assert "head_dim must be even" in str(error.value)


def test_an_invalid_position_encoding_fails_closed() -> None:
    with pytest.raises(HFCausalGQAAttentionFoldError) as error:
        _nope_topology(position_encoding="alibi")
    assert "position_encoding must be" in str(error.value)


# ---------------------------------------------------------------------------
# The window is the chimera window
# ---------------------------------------------------------------------------


def test_the_window_restricts_probabilities_exactly_as_chimera_does() -> None:
    module = _local_attention(17)
    rows = _rows(18)
    query = _heads(module.q_proj(rows), HEADS)
    key = _heads(module.k_proj(rows), KV_HEADS)
    mask = torch.ones(BATCH, SEQUENCE, dtype=torch.bool)

    _scores, probabilities = causal_attention_probabilities(
        query,
        key,
        attention_mask=mask,
        scaling=1.0 / math.sqrt(HEAD_DIM),
        attention_window=WINDOW,
    )

    offsets = torch.arange(SEQUENCE).reshape(-1, 1) - torch.arange(SEQUENCE).reshape(
        1, -1
    )
    band = (offsets >= 0) & (offsets < WINDOW)
    assert float(probabilities[:, :, ~band].abs().max().item()) == 0.0
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones(BATCH, HEADS, SEQUENCE, dtype=torch.float64),
    )

    with torch.no_grad():
        parent = module(rows)
    repeats = HEADS // KV_HEADS
    value = _heads(module.v_proj(rows), KV_HEADS).repeat_interleave(repeats, dim=1)
    context = torch.matmul(probabilities, value)
    replayed = context.permute(0, 2, 1, 3).reshape(
        BATCH, SEQUENCE, HEADS * HEAD_DIM
    ) @ module.out.weight.detach().to(dtype=torch.float64).T

    assert _relative_rmse(replayed, parent.detach().to(dtype=torch.float64)) < 1e-6


def test_an_unwindowed_replay_keeps_the_full_causal_mask() -> None:
    module = _local_attention(21)
    rows = _rows(22)
    query = _heads(module.q_proj(rows), HEADS)
    key = _heads(module.k_proj(rows), KV_HEADS)
    mask = torch.ones(BATCH, SEQUENCE, dtype=torch.bool)

    _scores, windowed = causal_attention_probabilities(
        query, key, attention_mask=mask, scaling=0.35, attention_window=SEQUENCE
    )
    _scores, plain = causal_attention_probabilities(
        query, key, attention_mask=mask, scaling=0.35
    )

    assert torch.equal(windowed, plain)


# ---------------------------------------------------------------------------
# Folding a real chimera local branch
# ---------------------------------------------------------------------------


def _compile_local(module, rows, *, target_topology=None):
    mask = torch.ones(BATCH, SEQUENCE, dtype=torch.bool)
    fit = torch.zeros(BATCH, SEQUENCE, dtype=torch.bool)
    fit[:, : SEQUENCE - 6] = True
    validation = ~fit
    with torch.no_grad():
        delta = module(rows)
    return compile_hf_causal_gqa_attention_fold(
        rows.to(dtype=torch.float64),
        source_query_sequences=module.q_proj(rows),
        source_key_sequences=module.k_proj(rows),
        source_value_sequences=module.v_proj(rows),
        target_residual_delta_sequences=delta,
        attention_mask=mask,
        position_ids=torch.arange(SEQUENCE).reshape(1, -1).expand(BATCH, SEQUENCE),
        valid_token_mask=mask,
        fit_token_mask=fit,
        validation_token_mask=validation,
        source_topology=_nope_topology(),
        target_topology=target_topology or _nope_topology(),
    )


def test_a_windowed_nope_fold_reproduces_the_chimera_local_branch() -> None:
    module = _local_attention(31)
    rows = _rows(32)

    fold = _compile_local(module, rows)

    assert fold.report["validation_residual_relative_rmse"] < 1e-6
    assert fold.report["position_encoding"] == "nope"
    assert fold.report["attention_window"] == WINDOW
    assert fold.report["head_dimension_selection"] == "channels"
    assert fold.report["rope_applied"] is False
    assert fold.report["training_steps"] == 0
    assert fold.report["optimizer_steps"] == 0
    assert fold.report["backward_calls"] == 0


def test_a_contracted_windowed_nope_fold_beats_the_no_branch_baseline() -> None:
    module = _local_attention(41)
    rows = _rows(42)
    target = _nope_topology(query_heads=2, key_value_heads=1, head_dim=5)

    fold = _compile_local(module, rows, target_topology=target)

    mask = torch.zeros(BATCH, SEQUENCE, dtype=torch.bool)
    mask[:, SEQUENCE - 6 :] = True
    with torch.no_grad():
        delta = module(rows).to(dtype=torch.float64)
    baseline = _relative_rmse(torch.zeros_like(delta[mask]), delta[mask])

    assert fold.report["validation_residual_relative_rmse"] < baseline
    # An odd head dimension is only admissible because no rotation is paired.
    assert all(
        len(group) == 5 for group in fold.assignment.selected_rope_pairs_by_target_kv
    )


def test_changing_the_window_or_the_encoding_fails_closed() -> None:
    module = _local_attention(51)
    rows = _rows(52)

    with pytest.raises(HFCausalGQAAttentionFoldError) as narrowed:
        _compile_local(module, rows, target_topology=_nope_topology(attention_window=3))
    assert "kind translation" in str(narrowed.value)

    with pytest.raises(HFCausalGQAAttentionFoldError) as rotated:
        _compile_local(
            module,
            rows,
            target_topology=_nope_topology(position_encoding="rope", head_dim=8),
        )
    assert "kind translation" in str(rotated.value)
