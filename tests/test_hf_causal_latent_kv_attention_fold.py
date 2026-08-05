"""Chimera-omni gated-MLA branch folding -- stage 0 and stage 1 acceptance.

Every parent here is a **real** ``chimera_omni.modeling.layers``
``GatedMultiHeadLatentAttention`` built through chimera-omni's own
configuration API.  Nothing is vendored; when the checkout is unavailable the
suite skips and says so.

Stage 0/1 of ``docs/GATED_MLA_MOE_FOLD_KERNEL_DESIGN_2026-08-05.md`` section 1:
the compressed key/value readout identity is exact, the shared key block really
is head-independent, an identity fold lands at numerical noise, a contracted
fold beats the no-branch baseline, the compiled child is byte-for-byte
installable into the real module, and every fail-closed condition fires.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_causal_latent_kv_attention_fold import (  # noqa: E402
    ChimeraLatentKVFoldPolicy,
    ChimeraLatentKVGeometry,
    HFCausalLatentKVAttentionFoldError,
    assemble_latent_keys,
    broadcast_shared_key,
    capture_chimera_latent_kv_layer_evidence,
    chimera_latent_kv_geometry,
    chimera_latent_kv_parameters,
    compile_hf_causal_latent_kv_attention_fold,
    latent_attention_probabilities,
    latent_kv_consistency,
    latent_pod_basis,
)


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_chimera_omni_toy_fold.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_chimera_omni_toy_fold_latent_kv", RUNNER
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


HIDDEN = 48
BATCH = 8
SEQUENCE = 24
HEADS = 4
KEY_HEAD_DIM = 8
VALUE_HEAD_DIM = 8
EXTRA_HEAD_DIM = 4
LATENT = 16
QUERY_LATENT = 16


def _config(**changes):
    body = {
        "vocab_size": 32,
        "hidden_size": HIDDEN,
        "intermediate_size": HIDDEN * 2,
        "num_layers": 2,
        "num_heads": HEADS,
        "num_kv_heads": HEADS,
        "max_sequence_length": 64,
        "block_pattern": ("gated_mla",),
        "residual_mode": "none",
        "attention_head_dim": KEY_HEAD_DIM,
        "mla_latent_size": LATENT,
        "mla_query_latent_size": QUERY_LATENT,
        "mla_key_head_dim": KEY_HEAD_DIM,
        "mla_extra_head_dim": EXTRA_HEAD_DIM,
        "mla_value_head_dim": VALUE_HEAD_DIM,
    }
    body.update(changes)
    return chimera_config.ChimeraConfig(**body)


def _mixer(seed: int, **changes):
    torch.manual_seed(seed)
    module = chimera_layers.GatedMultiHeadLatentAttention(_config(**changes))
    module.eval()
    return module.double()


def _rows(seed: int, *, hidden: int = HIDDEN):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, SEQUENCE, hidden, generator=generator, dtype=torch.float64)


def _masks():
    fit = torch.zeros(BATCH, SEQUENCE, dtype=torch.bool)
    fit[:, : SEQUENCE - 8] = True
    return fit, ~fit


def _evidence(seed: int, **changes):
    module = _mixer(seed, **changes)
    hidden = int(_config(**changes).hidden_size)
    evidence = capture_chimera_latent_kv_layer_evidence(
        module, _rows(seed + 1, hidden=hidden), layer_index=3
    )
    return module, evidence


def _relative_rmse(prediction, target) -> float:
    numerator = (prediction - target).square().mean().sqrt()
    denominator = target.square().mean().sqrt().clamp_min(1e-30)
    return float((numerator / denominator).item())


def _compile(evidence, parameters, geometry, **changes):
    fit, validation = _masks()
    body = {
        "attention_mask": torch.ones(BATCH, SEQUENCE, dtype=torch.bool),
        "fit_token_mask": fit,
        "validation_token_mask": validation,
    }
    body.update(changes)
    return compile_hf_causal_latent_kv_attention_fold(
        evidence, parameters, geometry, **body
    )


# ---------------------------------------------------------------------------
# Capture and the two evidence-consistency gates
# ---------------------------------------------------------------------------


def test_the_capture_is_a_trace_of_the_parent() -> None:
    module, evidence = _evidence(101)

    assert evidence.metadata["parent_forward_calls"] == 1
    assert evidence.geometry == chimera_latent_kv_geometry(module)
    assert evidence.geometry.query_head_dim == KEY_HEAD_DIM + EXTRA_HEAD_DIM
    assert evidence.geometry.compressed_query is True
    assert tuple(evidence.compressed_kv_rows.shape) == (
        BATCH,
        SEQUENCE,
        LATENT + EXTRA_HEAD_DIM,
    )


def test_the_key_value_readout_identity_is_exact() -> None:
    """Everything the key side contributes passes through the latent.

    The parent's ``kv_up`` and ``kv_norm`` run in the module's own dtype, so on
    a float64 parent this identity is exact linear algebra rather than a
    rounding claim: it holds to machine epsilon.
    """

    module, evidence = _evidence(103)
    parameters = chimera_latent_kv_parameters(module)
    gates = latent_kv_consistency(evidence, parameters)

    assert gates.readout_relative_rmse < 1e-12
    assert gates.diagnostics["shared_key_channels"] == EXTRA_HEAD_DIM
    assert 0.0 < gates.diagnostics["shared_key_channel_score_energy_fraction"] < 1.0


def test_the_delta_reconstruction_rebuilds_the_captured_branch_write() -> None:
    """The derived split, broadcast, mask and gate are checked, not asserted.

    ``GatedMultiHeadLatentAttention.forward`` computes its scores and context in
    float32 regardless of the module dtype (``layers.py:301``, ``:304``), so the
    measured number here is that downcast, not this kernel's arithmetic.
    """

    module, evidence = _evidence(105)
    parameters = chimera_latent_kv_parameters(module)
    gates = latent_kv_consistency(evidence, parameters)

    assert gates.delta_relative_rmse < 1e-6
    assert gates.diagnostics["parent_forward_calls"] == 0


def test_a_broken_shared_key_broadcast_fails_the_reconstruction_closed() -> None:
    module, evidence = _evidence(107)
    parameters = chimera_latent_kv_parameters(module)
    corrupted = evidence.compressed_kv_rows.clone()
    corrupted[..., LATENT:] = corrupted[..., LATENT:].flip(-1)
    broken = type(evidence)(
        index=evidence.index,
        block_input=evidence.block_input,
        attention_norm_output=evidence.attention_norm_output,
        query_rows=evidence.query_rows,
        compressed_kv_rows=corrupted,
        normalized_latent_rows=evidence.normalized_latent_rows,
        kv_rows=evidence.kv_rows,
        gate_rows=evidence.gate_rows,
        attention_delta=evidence.attention_delta,
        block_output=evidence.block_output,
        geometry=evidence.geometry,
        metadata=dict(evidence.metadata),
        query_latent_rows=evidence.query_latent_rows,
        normalized_query_latent_rows=evidence.normalized_query_latent_rows,
    )

    with pytest.raises(
        HFCausalLatentKVAttentionFoldError, match="does not reconstruct"
    ):
        latent_kv_consistency(broken, parameters)


def test_the_shared_key_block_is_head_independent() -> None:
    _module, evidence = _evidence(109)
    shared = evidence.shared_key_rows()
    broadcast = broadcast_shared_key(shared, HEADS)

    assert tuple(broadcast.shape) == (BATCH, HEADS, SEQUENCE, EXTRA_HEAD_DIM)
    for head in range(1, HEADS):
        assert torch.equal(broadcast[:, 0], broadcast[:, head])
    key_base, _value = evidence.head_key_value()
    assembled = assemble_latent_keys(key_base, shared)
    assert tuple(assembled.shape) == (
        BATCH,
        HEADS,
        SEQUENCE,
        KEY_HEAD_DIM + EXTRA_HEAD_DIM,
    )


def test_the_probability_replay_is_the_parents_causal_softmax() -> None:
    _module, evidence = _evidence(111)
    key_base, _value = evidence.head_key_value()
    key = assemble_latent_keys(key_base, evidence.shared_key_rows())
    _scores, probabilities = latent_attention_probabilities(
        evidence.head_query(),
        key,
        attention_mask=torch.ones(BATCH, SEQUENCE, dtype=torch.bool),
        scaling=evidence.geometry.attention_scaling,
    )

    upper = torch.triu(torch.ones(SEQUENCE, SEQUENCE, dtype=torch.bool), diagonal=1)
    assert float(probabilities[:, :, upper].abs().max().item()) == 0.0
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones(BATCH, HEADS, SEQUENCE, dtype=torch.float64),
        atol=1e-12,
    )


def test_an_unchanged_latent_width_takes_the_identity_basis() -> None:
    _module, evidence = _evidence(113)
    basis = latent_pod_basis(evidence.latent_rows(), LATENT)

    assert basis.identity is True
    assert torch.equal(basis.basis, torch.eye(LATENT, dtype=torch.float64))
    assert basis.retained_energy_fraction == 1.0
    truncated = latent_pod_basis(evidence.latent_rows(), LATENT // 2)
    assert truncated.identity is False
    assert 0.0 < truncated.retained_energy_fraction < 1.0


# ---------------------------------------------------------------------------
# Stage 1: the identity fold and the installable child
# ---------------------------------------------------------------------------


def test_an_identity_fold_reproduces_the_parent_branch() -> None:
    module, evidence = _evidence(201)
    parameters = chimera_latent_kv_parameters(module)
    fold = _compile(evidence, parameters, evidence.geometry)
    report = dict(fold.report)

    assert report["validation_residual_relative_rmse"] < 5e-2
    assert report["validation_beats_no_branch_baseline"] is True
    assert report["kv_latent_basis"] == "identity"
    assert report["query_latent_basis"] == "identity"
    assert report["attention_probability_overlap"] > 0.9
    assert report["parent_forward_calls"] == 0
    assert report["training_steps"] == 0
    assert report["backward_calls"] == 0
    assert report["score_chart_used"] is False


def test_the_compiled_child_is_installable_into_the_real_module() -> None:
    """The kernel's replay is the module's forward, not an approximation of it.

    Every emitted tensor is loaded into a fresh chimera module and the module's
    own forward is compared against the kernel's FP64 replay.  The parent class
    downcasts its scores to float32, so the two agree to that downcast and no
    further -- which is exactly the claim being made.
    """

    module, evidence = _evidence(203)
    parameters = chimera_latent_kv_parameters(module)
    fold = _compile(evidence, parameters, evidence.geometry)
    child = _mixer(999)
    state = fold.child_state_dict()

    assert set(state) == set(child.state_dict())
    with torch.no_grad():
        for name, value in state.items():
            child.state_dict()[name].copy_(value)
        installed = child(evidence.attention_norm_output)
    replayed, _probabilities = fold.replay(
        evidence.attention_norm_output,
        attention_mask=torch.ones(BATCH, SEQUENCE, dtype=torch.bool),
    )

    assert _relative_rmse(installed.to(dtype=torch.float64), replayed) < 1e-6


def test_a_contracted_fold_beats_the_no_branch_baseline() -> None:
    module, evidence = _evidence(205)
    parameters = chimera_latent_kv_parameters(module)
    geometry = ChimeraLatentKVGeometry(
        hidden_size=HIDDEN,
        heads=HEADS // 2,
        key_head_dim=KEY_HEAD_DIM // 2,
        value_head_dim=VALUE_HEAD_DIM // 2,
        extra_head_dim=EXTRA_HEAD_DIM,
        kv_latent_size=LATENT // 2,
        query_latent_size=QUERY_LATENT // 2,
    )
    fold = _compile(evidence, parameters, geometry)
    report = dict(fold.report)

    assert report["validation_beats_no_branch_baseline"] is True
    assert report["kv_latent_basis"] == "observed_row_pod"
    assert 0.0 < report["kv_latent_retained_energy_fraction"] < 1.0
    assert report["source_head_groups"] == [[0, 1], [2, 3]]
    assert len(report["selected_key_channels"]) == KEY_HEAD_DIM // 2
    assert len(report["selected_value_channels"]) == VALUE_HEAD_DIM // 2
    assert tuple(fold.kv_up_weight.shape) == (
        geometry.key_value_width,
        geometry.kv_latent_size,
    )


def test_a_dense_query_parent_folds_through_the_same_executable() -> None:
    """One executable, two query variants -- the latent size is a contract shape."""

    module, evidence = _evidence(207, mla_query_latent_size=0)
    parameters = chimera_latent_kv_parameters(module)
    fold = _compile(evidence, parameters, evidence.geometry)
    state = fold.child_state_dict()

    assert evidence.geometry.compressed_query is False
    assert "q_proj.weight" in state
    assert "q_down.weight" not in state
    assert dict(fold.report)["compressed_query_path"] is False
    assert dict(fold.report)["validation_beats_no_branch_baseline"] is True


# ---------------------------------------------------------------------------
# Fail-closed conditions
# ---------------------------------------------------------------------------


def test_a_shared_key_channel_change_is_refused_as_a_kind_translation() -> None:
    module, evidence = _evidence(301)
    parameters = chimera_latent_kv_parameters(module)
    geometry = ChimeraLatentKVGeometry(
        hidden_size=HIDDEN,
        heads=HEADS,
        key_head_dim=KEY_HEAD_DIM,
        value_head_dim=VALUE_HEAD_DIM,
        extra_head_dim=EXTRA_HEAD_DIM // 2,
        kv_latent_size=LATENT,
        query_latent_size=QUERY_LATENT,
    )

    with pytest.raises(
        HFCausalLatentKVAttentionFoldError, match="contract constant"
    ):
        _compile(evidence, parameters, geometry)


def test_a_query_variant_change_is_refused() -> None:
    module, evidence = _evidence(303)
    parameters = chimera_latent_kv_parameters(module)
    geometry = ChimeraLatentKVGeometry(
        hidden_size=HIDDEN,
        heads=HEADS,
        key_head_dim=KEY_HEAD_DIM,
        value_head_dim=VALUE_HEAD_DIM,
        extra_head_dim=EXTRA_HEAD_DIM,
        kv_latent_size=LATENT,
        query_latent_size=0,
    )

    with pytest.raises(
        HFCausalLatentKVAttentionFoldError, match="contract shape"
    ):
        _compile(evidence, parameters, geometry)


def test_head_or_latent_expansion_is_refused() -> None:
    module, evidence = _evidence(305)
    parameters = chimera_latent_kv_parameters(module)

    with pytest.raises(HFCausalLatentKVAttentionFoldError, match="expansion"):
        _compile(
            evidence,
            parameters,
            ChimeraLatentKVGeometry(
                hidden_size=HIDDEN,
                heads=HEADS * 2,
                key_head_dim=KEY_HEAD_DIM,
                value_head_dim=VALUE_HEAD_DIM,
                extra_head_dim=EXTRA_HEAD_DIM,
                kv_latent_size=LATENT,
                query_latent_size=QUERY_LATENT,
            ),
        )
    with pytest.raises(HFCausalLatentKVAttentionFoldError, match="expansion"):
        _compile(
            evidence,
            parameters,
            ChimeraLatentKVGeometry(
                hidden_size=HIDDEN,
                heads=HEADS,
                key_head_dim=KEY_HEAD_DIM,
                value_head_dim=VALUE_HEAD_DIM,
                extra_head_dim=EXTRA_HEAD_DIM,
                kv_latent_size=LATENT * 2,
                query_latent_size=QUERY_LATENT,
            ),
        )


def test_a_tightened_reconstruction_tolerance_refuses_before_any_solve() -> None:
    module, evidence = _evidence(307)
    parameters = chimera_latent_kv_parameters(module)

    with pytest.raises(
        HFCausalLatentKVAttentionFoldError, match="does not reconstruct"
    ):
        _compile(
            evidence,
            parameters,
            evidence.geometry,
            policy=ChimeraLatentKVFoldPolicy(
                delta_reconstruction_tolerance=1e-15
            ),
        )


def test_a_probability_overlap_floor_refuses_a_collapsed_score_path() -> None:
    """A collapsed query/key path is refused before any writer is fitted.

    A ridge large enough to crush ``q``/``k`` to nothing leaves the child
    replaying a near-uniform causal distribution.  That still keeps a
    surprising amount of the parent's mass -- which is why the floor is a
    policy constant and not a hard-coded number -- so the case is driven with
    a policy that asks for the parent's selection rather than for any mass at
    all, and the refusal names the measurement.
    """

    module, evidence = _evidence(309)
    parameters = chimera_latent_kv_parameters(module)

    with pytest.raises(
        HFCausalLatentKVAttentionFoldError, match="attention mass"
    ):
        _compile(
            evidence,
            parameters,
            evidence.geometry,
            policy=ChimeraLatentKVFoldPolicy(
                ridge=1.0e12, minimum_attention_probability_overlap=0.9
            ),
        )


def test_a_branch_that_loses_to_writing_nothing_is_refused() -> None:
    """The design's 1.0 no-branch floor, exercised through the ridge axis."""

    module, evidence = _evidence(311)
    parameters = chimera_latent_kv_parameters(module)

    with pytest.raises(
        HFCausalLatentKVAttentionFoldError, match="no-branch baseline"
    ):
        _compile(
            evidence,
            parameters,
            evidence.geometry,
            policy=ChimeraLatentKVFoldPolicy(ridge=1.0e12),
        )


def test_a_mismatched_carried_weight_is_refused() -> None:
    module, evidence = _evidence(313)
    other = _mixer(4242)
    parameters = chimera_latent_kv_parameters(other)

    with pytest.raises(
        HFCausalLatentKVAttentionFoldError, match="captured key/value readout"
    ):
        latent_kv_consistency(evidence, parameters)
