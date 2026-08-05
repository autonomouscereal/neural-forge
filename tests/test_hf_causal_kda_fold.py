"""Chimera-omni KDA branch folding -- stage 0 and stage 1 acceptance.

Every parent here is a **real** ``chimera_omni.modeling.layers``
``KimiDeltaAttention`` built through chimera-omni's own configuration API and
imported from the target repository's checkout.  Nothing about that repository
is vendored, copied, or re-implemented; when the checkout is unavailable the
suite skips and says so.

The acceptance ladder of
``docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md`` section 5 is what these
tests assert: the exact effective-weight unroll reproduces the slow recurrence,
a constant-key head collapses to the closed-form scalar IIR, an ``L``-tap FIR
is matched by ``L`` poles through the Vandermonde solve, an identity-geometry
fold lands at numerical noise, a contracted fold beats the no-branch baseline,
every fail-closed condition fires, and nothing trains.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_causal_kda_fold import (  # noqa: E402
    ChimeraKDAFoldPolicy,
    ChimeraKDAGeometry,
    ChimeraKDALayerEvidence,
    HFCausalKDAFoldError,
    capture_chimera_kda_layer_evidence,
    chimera_kda_geometry,
    chimera_kda_parameters,
    compile_gqa_to_kda_translation,
    compile_hf_causal_kda_fold,
    compile_kda_to_gqa_translation,
    compile_short_conv_to_kda_translation,
    constant_key_kda_pole,
    fir_taps_to_kda_poles,
    kda_effective_weights,
    replay_kda_branch,
    select_kda_channels,
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
BATCH = 8
SEQUENCE = 24
SNAPSHOT_STRIDE = 8


def _config(**changes):
    body = {
        "vocab_size": 32,
        "hidden_size": HIDDEN,
        "intermediate_size": HIDDEN * 2,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "max_sequence_length": 64,
        "block_pattern": ("kda",),
        "residual_mode": "none",
        "kda_head_dim": 8,
        "kda_value_head_dim": 8,
        "kda_decay_rank": 4,
        "local_window": 8,
    }
    body.update(changes)
    return chimera_config.ChimeraConfig(**body)


def _kda(seed: int, *, double: bool = False, **changes):
    torch.manual_seed(seed)
    module = chimera_layers.KimiDeltaAttention(_config(**changes))
    module.eval()
    if double:
        module = module.double()
    return module


def _rows(seed: int, *, dtype=torch.float32, hidden: int = HIDDEN):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, SEQUENCE, hidden, generator=generator, dtype=dtype)


def _masks():
    fit = torch.zeros(BATCH, SEQUENCE, dtype=torch.bool)
    fit[:, : SEQUENCE - 6] = True
    return fit, ~fit


def _evidence(seed: int = 101, *, double: bool = False, module=None):
    parent = module or _kda(seed, double=double)
    rows = _rows(seed + 1, dtype=torch.float64 if double else torch.float32)
    return parent, capture_chimera_kda_layer_evidence(
        parent, rows, layer_index=0, state_snapshot_stride=SNAPSHOT_STRIDE
    )


def _state_digest(module) -> str:
    return canonical_sha256(
        {
            name: hashlib.sha256(
                value.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            for name, value in sorted(module.state_dict().items())
        }
    )


def _relative_rmse(prediction, target) -> float:
    numerator = (prediction - target).square().mean().sqrt()
    denominator = target.square().mean().sqrt().clamp_min(1e-30)
    return float((numerator / denominator).item())


# ---------------------------------------------------------------------------
# Evidence capture and the exact unroll
# ---------------------------------------------------------------------------


def test_the_capture_is_a_trace_of_the_parent_not_a_reimplementation() -> None:
    parent, evidence = _evidence(11)

    assert evidence.metadata["parent_forward_calls"] == 1
    assert evidence.metadata["capture_reconstruction_relative_rmse"] == 0.0
    assert evidence.geometry == chimera_kda_geometry(parent)
    assert evidence.state_snapshot_positions == (7, 15, 23)
    assert tuple(evidence.state_snapshots.shape) == (3, BATCH, 4, 8, 8)


def test_the_unroll_equals_the_slow_recurrence_in_double_precision() -> None:
    _parent, evidence = _evidence(13, double=True)

    unroll = kda_effective_weights(evidence, tolerance=1e-12)

    assert unroll.relative_rmse < 1e-12
    assert unroll.diagnostics["parent_forward_calls"] == 0
    assert unroll.diagnostics["acausal_weight_maximum_absolute_value"] == 0.0
    assert unroll.diagnostics["within_tolerance"] is True
    # Signed, unnormalized rows: this is the property that makes KDA -> softmax
    # a measured-partial translation rather than an exact one.
    assert unroll.diagnostics["row_mass_standard_deviation"] > 0.0


def test_an_inconsistent_recurrent_cache_fails_the_unroll_closed() -> None:
    _parent, evidence = _evidence(15)
    tampered = ChimeraKDALayerEvidence(
        **{
            **{
                descriptor: getattr(evidence, descriptor)
                for descriptor in (
                    "index",
                    "block_input",
                    "attention_norm_output",
                    "pre_conv_query",
                    "pre_conv_key",
                    "pre_conv_value",
                    "query_rows",
                    "key_rows",
                    "value_rows",
                    "retention_rows",
                    "decay_logit_rows",
                    "beta_rows",
                    "gate_rows",
                    "attention_delta",
                    "block_output",
                    "state_snapshots",
                    "state_snapshot_positions",
                    "geometry",
                    "metadata",
                )
            },
            "recurrent_output_rows": evidence.recurrent_output_rows * 1.5,
        }
    )

    with pytest.raises(HFCausalKDAFoldError) as error:
        kda_effective_weights(tampered, tolerance=1e-6)
    assert "evidence" in str(error.value)
    assert "fails closed" in str(error.value)


# ---------------------------------------------------------------------------
# Stage 0 -- the closed-form primitives of the shortconv -> KDA lane
# ---------------------------------------------------------------------------


def _constant_key_module(seed: int):
    """One KDA head family with a constant key, constant beta, uniform decay."""

    module = _kda(seed, double=True)
    with torch.no_grad():
        module.q_proj.weight.zero_()
        module.k_proj.weight.zero_()
        module.q_conv.conv.weight.copy_(module.k_conv.conv.weight)
        module.q_conv.conv.bias.copy_(module.k_conv.conv.bias)
        module.write_strength.weight.zero_()
        module.decay_up.weight.zero_()
        module.decay_bias.fill_(0.25)
    return module


def test_a_constant_key_head_equals_the_closed_form_scalar_iir() -> None:
    module = _constant_key_module(21)
    _parent, evidence = _evidence(21, double=True, module=module)

    for head in range(evidence.geometry.heads):
        key = evidence.key_rows[0, 0, head]
        retention = evidence.retention_rows[0, 0, head]
        beta = float(evidence.beta_rows[0, 0, head].item())
        pole = constant_key_kda_pole(
            key=key, retention=retention, write_strength=beta
        )

        assert pole.exact is True
        assert pole.write_strength == pytest.approx(0.5, abs=1e-12)

        state = torch.zeros(BATCH, evidence.geometry.value_head_dim, dtype=torch.float64)
        expected = []
        for position in range(SEQUENCE):
            state = pole.pole * state + beta * evidence.value_rows[:, position, head]
            expected.append(state.clone())
        # The collapse is algebraically exact for a unit key.  The parent
        # normalizes q/k in float32 (``layers.py:139-140``) even in a float64
        # module, so ``k.k`` differs from one at float32 epsilon and the
        # residue lands there -- not at float64 noise.
        assert _relative_rmse(
            torch.stack(expected, dim=1), evidence.recurrent_output_rows[:, :, head]
        ) < 1e-6


def test_a_nonuniform_decay_reports_the_single_pole_reading_as_inexact() -> None:
    module = _constant_key_module(23)
    with torch.no_grad():
        module.decay_bias[:, 0] = -2.0
    _parent, evidence = _evidence(23, double=True, module=module)

    pole = constant_key_kda_pole(
        key=evidence.key_rows[0, 0, 0],
        retention=evidence.retention_rows[0, 0, 0],
        write_strength=float(evidence.beta_rows[0, 0, 0].item()),
    )

    assert pole.exact is False
    assert pole.diagnostics["retention_spread_on_key_support"] > 0.0


def test_a_vandermonde_pole_set_matches_an_l_tap_fir_at_every_lag() -> None:
    generator = torch.Generator().manual_seed(29)
    taps = torch.randn(3, generator=generator, dtype=torch.float64)

    fit = fir_taps_to_kda_poles(taps, heads=4)

    assert fit.report["matched_lag_maximum_absolute_error"] < 1e-9
    assert len(set(fit.poles)) == 3
    assert all(pole <= 0.9 for pole in fit.poles)
    assert fit.report["vandermonde_condition_number"] < 1e4
    # The tail beyond the matched lags is measured, never assumed zero.
    assert fit.report["tail_energy_fraction"] >= 0.0
    assert fit.report["parent_forward_calls"] == 0


def test_too_few_heads_for_the_required_poles_fails_closed() -> None:
    taps = torch.ones(4, dtype=torch.float64)

    with pytest.raises(HFCausalKDAFoldError) as error:
        fir_taps_to_kda_poles(taps, heads=3)
    assert "at least 4 KDA heads" in str(error.value)


def test_an_over_tight_condition_cap_fails_the_pole_fit_closed() -> None:
    taps = torch.ones(3, dtype=torch.float64)

    with pytest.raises(HFCausalKDAFoldError) as error:
        fir_taps_to_kda_poles(taps, heads=4, condition_cap=1.5)
    assert "ill-conditioned" in str(error.value)


# ---------------------------------------------------------------------------
# Stage 1 -- identity geometry
# ---------------------------------------------------------------------------


def test_an_identity_geometry_fold_lands_at_numerical_noise() -> None:
    parent, evidence = _evidence(31)
    fit, validation = _masks()
    before = _state_digest(parent)

    fold = compile_hf_causal_kda_fold(
        evidence,
        chimera_kda_parameters(parent),
        chimera_kda_geometry(parent),
        fit_token_mask=fit,
        validation_token_mask=validation,
    )

    assert fold.report["validation_residual_relative_rmse"] < 1e-5
    assert fold.report["validation_no_branch_relative_rmse"] == pytest.approx(1.0)
    assert fold.report["validation_beats_no_branch_baseline"] is True
    assert fold.report["conv_tap_carry_relative_rmse"] < 1e-5
    assert fold.report["decay_inverse_chain_relative_rmse"] < 1e-5
    assert fold.report["decay_rank_truncation_relative_error"] < 1e-6
    assert fold.report["recurrent_output_relative_rmse"] < 1e-5
    assert max(fold.report["state_snapshot_relative_rmse"]) < 1e-5
    assert fold.report["rotation_applied"] is False
    assert fold.report["recurrence_installed_before_writer_fit"] is True
    # The compile never touches the parent.
    assert _state_digest(parent) == before


def test_the_receipt_seals_zero_training_counters_and_reseals() -> None:
    parent, evidence = _evidence(33)
    fit, validation = _masks()

    fold = compile_hf_causal_kda_fold(
        evidence,
        chimera_kda_parameters(parent),
        chimera_kda_geometry(parent),
        fit_token_mask=fit,
        validation_token_mask=validation,
    )

    for counter in (
        "training_steps",
        "optimizer_steps",
        "backward_calls",
        "parent_forward_calls",
    ):
        assert fold.report[counter] == 0
    assert fold.report["numeric_weight_matching_objective"] is False
    body = {
        key: value for key, value in fold.report.items() if key != "report_sha256"
    }
    assert canonical_sha256(body) == fold.report["report_sha256"]


def test_the_fp64_replay_reproduces_the_compiled_endpoint() -> None:
    parent, evidence = _evidence(35)
    fit, validation = _masks()

    fold = compile_hf_causal_kda_fold(
        evidence,
        chimera_kda_parameters(parent),
        chimera_kda_geometry(parent),
        fit_token_mask=fit,
        validation_token_mask=validation,
    )
    delta = replay_kda_branch(fold, evidence.attention_norm_output)
    bf16 = replay_kda_branch(
        fold, evidence.attention_norm_output, dtype=torch.bfloat16
    )

    assert _relative_rmse(delta, evidence.attention_delta) < 1e-5
    assert fold.report["validation_residual_relative_rmse_bf16"] < 1.0
    assert _relative_rmse(bf16, delta) < 0.1


# ---------------------------------------------------------------------------
# Stage 2 -- contraction
# ---------------------------------------------------------------------------


def _contracted_geometry(**changes) -> ChimeraKDAGeometry:
    body = {
        "hidden_size": HIDDEN,
        "heads": 3,
        "key_head_dim": 6,
        "value_head_dim": 6,
        "decay_rank": 3,
        "conv_kernel_size": 4,
    }
    body.update(changes)
    return ChimeraKDAGeometry(**body)


def test_a_contracted_fold_beats_the_no_branch_baseline() -> None:
    parent, evidence = _evidence(41)
    fit, validation = _masks()

    fold = compile_hf_causal_kda_fold(
        evidence,
        chimera_kda_parameters(parent),
        _contracted_geometry(),
        fit_token_mask=fit,
        validation_token_mask=validation,
    )

    report = fold.report
    assert report["validation_residual_relative_rmse"] < report[
        "validation_no_branch_relative_rmse"
    ]
    assert report["validation_residual_relative_rmse_bf16"] < report[
        "validation_no_branch_relative_rmse"
    ]
    assert report["validation_residual_retained_energy_fraction"] > 0.1
    assert report["target_geometry"]["heads"] == 3
    assert report["assignment"]["source_head_groups"] == [[0, 1], [2], [3]]
    assert sum(len(group) for group in fold.assignment.source_head_groups) == 4


def test_channel_selection_is_channel_aligned_and_never_rotates() -> None:
    parent, evidence = _evidence(43)
    fit, _validation = _masks()
    parameters = chimera_kda_parameters(parent)
    geometry = _contracted_geometry()
    assignment = compile_hf_causal_kda_fold(
        evidence,
        parameters,
        geometry,
        fit_token_mask=fit,
        validation_token_mask=~fit,
    ).assignment

    selection = select_kda_channels(
        evidence,
        parameters,
        geometry,
        assignment=assignment,
        row_mask=fit,
        policy=ChimeraKDAFoldPolicy(),
    )

    assert selection.report["rotation_applied"] is False
    assert selection.report["selection_is_channel_aligned"] is True
    for channels in (*selection.key_channels, *selection.value_channels):
        assert list(channels) == sorted(set(channels))
        assert len(channels) == 6
    for row in selection.report["per_head"]:
        assert row["key_selection_method"] == "balanced_krylov_pivot_qdeim"
        assert row["balanced_hankel_singular_values"]
        assert row["retained_key_write_energy_fraction"] > 0.0


def test_the_selected_channels_carry_their_conv_taps_exactly() -> None:
    parent, evidence = _evidence(45)
    fit, validation = _masks()
    parameters = chimera_kda_parameters(parent)
    geometry = _contracted_geometry(heads=4)

    fold = compile_hf_causal_kda_fold(
        evidence,
        parameters,
        geometry,
        fit_token_mask=fit,
        validation_token_mask=validation,
    )

    # Singleton head groups: the depthwise rows are channel-aligned, so the tap
    # carry is exact rather than barycentric.
    assert fold.report["merged_query_conv_tap_dispersion"] == 0.0
    taps = parameters.value_conv_weight.reshape(4, 8, 4)
    for head, channels in enumerate(fold.selection.value_channels):
        expected = taps[head, list(channels)]
        installed = fold.value_conv_weight.reshape(4, 6, 4)[head]
        assert torch.equal(installed, expected)


def test_a_head_merge_is_measured_at_the_endpoint_not_assumed_exact() -> None:
    """Merged recurrences do not commute; the cost shows up on validation.

    An identity fold keeps essentially all of the branch energy.  Merging one
    pair of heads on a random-initialization parent keeps a fifth of it, and
    merging every pair keeps under a tenth -- the honest reading of the design's
    own warning that a merged head is *refit*, never conjugated.
    """

    parent, evidence = _evidence(47)
    fit, validation = _masks()
    parameters = chimera_kda_parameters(parent)

    def energy(heads: int) -> float:
        fold = compile_hf_causal_kda_fold(
            evidence,
            parameters,
            _contracted_geometry(
                heads=heads, key_head_dim=8, value_head_dim=8, decay_rank=4
            ),
            fit_token_mask=fit,
            validation_token_mask=validation,
        )
        return float(fold.report["validation_residual_retained_energy_fraction"])

    identity = energy(4)
    partial = energy(3)
    full = energy(2)

    assert identity > 0.999
    assert partial < 0.5
    assert full < partial


def test_an_unrepresentable_endpoint_target_fails_closed() -> None:
    parent, evidence = _evidence(48)
    fit, validation = _masks()
    generator = torch.Generator().manual_seed(480)

    with pytest.raises(HFCausalKDAFoldError) as error:
        compile_hf_causal_kda_fold(
            evidence,
            chimera_kda_parameters(parent),
            _contracted_geometry(),
            fit_token_mask=fit,
            validation_token_mask=validation,
            endpoint_target=torch.randn(
                BATCH, SEQUENCE, HIDDEN, generator=generator, dtype=torch.float64
            ),
        )
    assert "no-branch baseline" in str(error.value)


def test_merged_head_dispersion_thresholds_fail_closed() -> None:
    parent, evidence = _evidence(47)
    fit, validation = _masks()
    parameters = chimera_kda_parameters(parent)

    def compile_with(policy: ChimeraKDAFoldPolicy):
        return compile_hf_causal_kda_fold(
            evidence,
            parameters,
            _contracted_geometry(),
            fit_token_mask=fit,
            validation_token_mask=validation,
            policy=policy,
        )

    with pytest.raises(HFCausalKDAFoldError) as decay:
        compile_with(ChimeraKDAFoldPolicy(maximum_merged_decay_dispersion=0.0))
    assert "decay log-scale dispersion" in str(decay.value)

    with pytest.raises(HFCausalKDAFoldError) as beta:
        compile_with(ChimeraKDAFoldPolicy(maximum_merged_beta_dispersion=0.0))
    assert "beta dispersion" in str(beta.value)


def test_a_head_expansion_and_a_kernel_change_both_fail_closed() -> None:
    parent, evidence = _evidence(49)
    fit, validation = _masks()
    parameters = chimera_kda_parameters(parent)

    with pytest.raises(HFCausalKDAFoldError) as expansion:
        compile_hf_causal_kda_fold(
            evidence,
            parameters,
            _contracted_geometry(heads=8),
            fit_token_mask=fit,
            validation_token_mask=validation,
        )
    assert "contraction-only" in str(expansion.value)

    with pytest.raises(HFCausalKDAFoldError) as kernel:
        compile_hf_causal_kda_fold(
            evidence,
            parameters,
            _contracted_geometry(conv_kernel_size=3),
            fit_token_mask=fit,
            validation_token_mask=validation,
        )
    assert "kind translation" in str(kernel.value)


def test_overlapping_partitions_fail_closed() -> None:
    parent, evidence = _evidence(53)
    everything = torch.ones(BATCH, SEQUENCE, dtype=torch.bool)

    with pytest.raises(HFCausalKDAFoldError) as error:
        compile_hf_causal_kda_fold(
            evidence,
            chimera_kda_parameters(parent),
            chimera_kda_geometry(parent),
            fit_token_mask=everything,
            validation_token_mask=everything,
        )
    assert "disjoint" in str(error.value)


# ---------------------------------------------------------------------------
# Emission and rollback
# ---------------------------------------------------------------------------


def test_the_child_state_dict_installs_and_rolls_back_exactly() -> None:
    parent, evidence = _evidence(61)
    fit, validation = _masks()
    geometry = _contracted_geometry()
    fold = compile_hf_causal_kda_fold(
        evidence,
        chimera_kda_parameters(parent),
        geometry,
        fit_token_mask=fit,
        validation_token_mask=validation,
    )
    child = _kda(
        71,
        num_heads=3,
        num_kv_heads=3,
        kda_head_dim=6,
        kda_value_head_dim=6,
        kda_decay_rank=3,
    )
    snapshot = {
        name: value.detach().clone() for name, value in child.state_dict().items()
    }
    before = _state_digest(child)

    emitted = fold.child_state_dict()
    assert set(emitted) == set(child.state_dict())
    child.load_state_dict(
        {name: value.to(dtype=torch.float32) for name, value in emitted.items()}
    )
    assert _state_digest(child) != before

    child.load_state_dict(snapshot)
    assert _state_digest(child) == before

    # The installed child computes the branch the fold reported.
    child.load_state_dict(
        {name: value.to(dtype=torch.float32) for name, value in emitted.items()}
    )
    with torch.no_grad():
        installed = child(evidence.attention_norm_output.to(dtype=torch.float32))
    replayed = replay_kda_branch(fold, evidence.attention_norm_output)
    assert _relative_rmse(installed.to(dtype=torch.float64), replayed) < 1e-4


def test_every_translation_entry_point_fails_closed_with_its_citation() -> None:
    for entry in (
        compile_short_conv_to_kda_translation,
        compile_gqa_to_kda_translation,
        compile_kda_to_gqa_translation,
    ):
        with pytest.raises(HFCausalKDAFoldError) as error:
            entry()
        message = str(error.value)
        assert "KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md" in message
        assert "fails closed" in message
