"""Chimera-omni CSA/HCA branch folding -- stage 0 and stage 1 acceptance.

Every parent here is a **real** ``chimera_omni.modeling.layers``
``SparseLatentAttention`` built through chimera-omni's own configuration API.
CSA and HCA are the same module at two strides, so the same tests run at both
and the stride is asserted to be a contract constant rather than a fold
coordinate.

Stage 0/1 of ``docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md`` section 5:
the sub-branch split is exact, the pooling operator is the parent's and
commutes with a chart, an identity fold lands at numerical noise on *both*
sub-branches, a contracted fold beats the no-branch baseline, the alternation
is bounded and monotone, the three-way translation split is exact and shows the
HCA coverage gap, and every fail-closed condition fires.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_causal_gqa_attention_fold import (  # noqa: E402
    causal_attention_probabilities,
)
from neural_forge.hf_causal_sparse_latent_fold import (  # noqa: E402
    ChimeraSparseLatentFoldPolicy,
    ChimeraSparseLatentGeometry,
    HFCausalSparseLatentFoldError,
    capture_chimera_sparse_latent_layer_evidence,
    chimera_sparse_latent_geometry,
    compile_gqa_to_summary_attention_translation,
    compile_hf_causal_sparse_latent_fold,
    gqa_to_summary_three_way_split,
    split_summary_branch_delta,
    summary_block_mask,
    summary_pooling_matrix,
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
BATCH = 12
SEQUENCE = 32
WINDOW = 8
CSA_STRIDE = 4
HCA_STRIDE = 16


def _config(**changes):
    body = {
        "vocab_size": 32,
        "hidden_size": HIDDEN,
        "intermediate_size": HIDDEN * 2,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "max_sequence_length": 64,
        "block_pattern": ("csa",),
        "residual_mode": "none",
        "local_window": WINDOW,
        "csa_compress_stride": CSA_STRIDE,
        "hca_compress_stride": HCA_STRIDE,
    }
    body.update(changes)
    return chimera_config.ChimeraConfig(**body)


def _sparse_latent(seed: int, *, stride: int = CSA_STRIDE, **changes):
    torch.manual_seed(seed)
    module = chimera_layers.SparseLatentAttention(_config(**changes), stride=stride)
    module.eval()
    return module


def _rows(seed: int, *, hidden: int = HIDDEN):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, SEQUENCE, hidden, generator=generator)


def _masks():
    fit = torch.zeros(BATCH, SEQUENCE, dtype=torch.bool)
    fit[:, : SEQUENCE - 10] = True
    return fit, ~fit


def _evidence(seed: int, *, stride: int = CSA_STRIDE, family: str = "csa"):
    module = _sparse_latent(seed, stride=stride)
    evidence = capture_chimera_sparse_latent_layer_evidence(
        module, _rows(seed + 1), layer_index=2, family=family
    )
    return module, evidence


def _state_digest(module) -> str:
    return canonical_sha256(
        {
            name: hashlib.sha256(
                value.detach()
                .to(device="cpu")
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            ).hexdigest()
            for name, value in sorted(module.state_dict().items())
        }
    )


def _relative_rmse(prediction, target) -> float:
    numerator = (prediction - target).square().mean().sqrt()
    denominator = target.square().mean().sqrt().clamp_min(1e-30)
    return float((numerator / denominator).item())


def _compile(evidence, geometry, **changes):
    fit, validation = _masks()
    body = {
        "attention_mask": torch.ones(BATCH, SEQUENCE, dtype=torch.bool),
        "fit_token_mask": fit,
        "validation_token_mask": validation,
    }
    body.update(changes)
    return compile_hf_causal_sparse_latent_fold(evidence, geometry, **body)


# ---------------------------------------------------------------------------
# Capture, pooling, and the exact split
# ---------------------------------------------------------------------------


def test_the_capture_is_a_trace_of_the_parent() -> None:
    module, evidence = _evidence(101)

    assert evidence.metadata["parent_forward_calls"] == 1
    assert evidence.metadata["capture_reconstruction_relative_rmse"] < 1e-6
    assert evidence.geometry == chimera_sparse_latent_geometry(module)
    assert evidence.summaries == SEQUENCE // CSA_STRIDE + 1
    assert evidence.block_ids[0] == 0
    assert evidence.block_ids[-1] == (SEQUENCE - 1) // CSA_STRIDE


def test_the_pooling_operator_is_the_parents_and_commutes_with_a_chart() -> None:
    module, evidence = _evidence(103)
    rows = evidence.attention_norm_output
    pooling = summary_pooling_matrix(SEQUENCE, CSA_STRIDE)

    with torch.no_grad():
        parent = module._summaries(rows.to(dtype=torch.float32))
    pooled = torch.matmul(pooling.reshape(1, *pooling.shape), rows)

    assert float(pooled[:, 0].abs().max().item()) == 0.0
    assert _relative_rmse(pooled[:, 1:], parent.to(dtype=torch.float64)) < 1e-6

    generator = torch.Generator().manual_seed(1030)
    chart, _residual = torch.linalg.qr(
        torch.randn(HIDDEN, HIDDEN // 2, generator=generator, dtype=torch.float64)
    )
    # mean(h) E == mean(h E): the same boundary chart serves raw and pooled rows.
    assert torch.allclose(
        torch.matmul(pooling.reshape(1, *pooling.shape), rows @ chart),
        pooled @ chart,
        atol=1e-12,
    )


def test_the_block_mask_only_admits_completed_blocks() -> None:
    mask = summary_block_mask(SEQUENCE, CSA_STRIDE)

    assert bool(mask[:, 0].all().item())
    for position in range(SEQUENCE):
        allowed = [index for index in range(int(mask.shape[1])) if bool(mask[position, index])]
        assert allowed == list(range(position // CSA_STRIDE + 1))
    # The newest summarized token is up to 2*stride - 1 positions old.
    assert not bool(mask[SEQUENCE - 1, SEQUENCE // CSA_STRIDE].item())


def test_the_sub_branch_split_reconstructs_the_sealed_write() -> None:
    _module, evidence = _evidence(105)

    split = split_summary_branch_delta(evidence)

    # The parent adds the two sub-branches in float32 (``layers.py:228``), so
    # the identity closes at float32 epsilon, not at float64 noise.
    assert split.diagnostics["split_reconstruction_relative_rmse"] < 1e-7
    assert split.diagnostics["within_tolerance"] is True
    assert split.diagnostics["parent_forward_calls"] == 0
    assert split.diagnostics["local_energy_fraction"] > 0.0
    assert split.diagnostics["summary_energy_fraction"] > 0.0


def test_a_tampered_split_fails_closed() -> None:
    _module, evidence = _evidence(107)
    object.__setattr__(evidence, "summary_delta", evidence.summary_delta * 1.25)

    with pytest.raises(HFCausalSparseLatentFoldError) as error:
        split_summary_branch_delta(evidence)
    assert "do not reconstruct the sealed branch write" in str(error.value)


# ---------------------------------------------------------------------------
# Stage 1 -- identity geometry
# ---------------------------------------------------------------------------


def test_an_identity_fold_lands_at_numerical_noise_on_both_sub_branches() -> None:
    module, evidence = _evidence(111)
    before = _state_digest(module)

    fold = _compile(evidence, chimera_sparse_latent_geometry(module))

    report = fold.report
    assert report["validation_residual_relative_rmse"] < 1e-4
    assert report["local_sub_branch_validation_relative_rmse"] < 1e-4
    assert report["summary_sub_branch_validation_relative_rmse"] < 1e-3
    assert report["summary_probability_overlap"] > 0.999
    assert report["validation_beats_no_branch_baseline"] is True
    assert report["local_position_encoding"] == "nope"
    assert report["zero_summary_preserved_exactly"] is True
    assert _state_digest(module) == before


def test_the_receipt_seals_zero_training_counters_and_reseals() -> None:
    module, evidence = _evidence(113)

    fold = _compile(evidence, chimera_sparse_latent_geometry(module))

    for counter in (
        "training_steps",
        "optimizer_steps",
        "backward_calls",
        "parent_forward_calls",
    ):
        assert fold.report[counter] == 0
    body = {
        key: value for key, value in fold.report.items() if key != "report_sha256"
    }
    assert canonical_sha256(body) == fold.report["report_sha256"]
    assert fold.report["local_fold_report_sha256"] == fold.local_fold.report[
        "report_sha256"
    ]


def test_the_replay_reproduces_the_compiled_endpoint() -> None:
    module, evidence = _evidence(115)
    fold = _compile(evidence, chimera_sparse_latent_geometry(module))

    total, local, summary = fold.replay(
        evidence.attention_norm_output,
        attention_mask=torch.ones(BATCH, SEQUENCE, dtype=torch.bool),
        position_ids=torch.arange(SEQUENCE).reshape(1, -1).expand(BATCH, SEQUENCE),
    )

    assert _relative_rmse(total, evidence.attention_delta) < 1e-4
    assert _relative_rmse(local, evidence.local_delta) < 1e-4
    assert _relative_rmse(summary, evidence.summary_delta) < 1e-3
    assert torch.allclose(total, local + summary, atol=1e-12)


def test_hca_is_the_same_module_at_a_longer_stride() -> None:
    module, evidence = _evidence(117, stride=HCA_STRIDE, family="hca")

    fold = _compile(evidence, chimera_sparse_latent_geometry(module, family="hca"))

    assert fold.report["family"] == "hca"
    assert fold.report["summary_stride"] == HCA_STRIDE
    assert fold.report["validation_residual_relative_rmse"] < 1e-3
    assert evidence.summaries == SEQUENCE // HCA_STRIDE + 1


# ---------------------------------------------------------------------------
# Stage 2 -- contraction and the alternation
# ---------------------------------------------------------------------------


def _contracted(**changes) -> ChimeraSparseLatentGeometry:
    body = {
        "hidden_size": HIDDEN,
        "query_heads": 2,
        "key_value_heads": 1,
        "head_dim": 6,
        "local_window": WINDOW,
        "summary_stride": CSA_STRIDE,
        "family": "csa",
    }
    body.update(changes)
    return ChimeraSparseLatentGeometry(**body)


def test_a_contracted_fold_beats_the_no_branch_baseline() -> None:
    _module, evidence = _evidence(121)

    fold = _compile(evidence, _contracted())

    report = fold.report
    assert report["validation_residual_relative_rmse"] < report[
        "validation_no_branch_relative_rmse"
    ]
    assert report["validation_residual_relative_rmse_bf16"] < report[
        "validation_no_branch_relative_rmse"
    ]
    assert report["validation_residual_retained_energy_fraction"] > 0.2
    # Per-sub-branch energy is reported separately so a summary collapse cannot
    # hide inside a good local fit.
    assert "local_sub_branch_retained_energy_fraction" in report
    assert "summary_sub_branch_retained_energy_fraction" in report


def test_the_alternation_is_bounded_and_monotone_on_fit_rows() -> None:
    _module, evidence = _evidence(123)

    fold = _compile(
        evidence,
        _contracted(),
        policy=ChimeraSparseLatentFoldPolicy(alternation_rounds=3),
    )

    trace = fold.report["alternation_trace"]
    assert [row["round"] for row in trace] == [0, 1, 1, 2, 2]
    assert fold.report["alternation_objective_monotone"] is True
    assert all(
        trace[index]["fit_objective"] <= trace[index - 1]["fit_objective"] + 1e-12
        for index in range(1, len(trace))
    )


def test_a_single_round_skips_the_alternation_entirely() -> None:
    _module, evidence = _evidence(125)

    fold = _compile(
        evidence,
        _contracted(),
        policy=ChimeraSparseLatentFoldPolicy(alternation_rounds=1),
    )

    assert len(fold.report["alternation_trace"]) == 1
    assert fold.report["alternation_rounds"] == 1


def test_the_child_state_dict_installs_and_rolls_back_exactly() -> None:
    _module, evidence = _evidence(127)
    geometry = _contracted()
    fold = _compile(evidence, geometry)
    child = _sparse_latent(
        137, num_heads=2, num_kv_heads=1, attention_head_dim=6
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

    with torch.no_grad():
        installed = child(evidence.attention_norm_output.to(dtype=torch.float32))
    replayed, _local, _summary = fold.replay(
        evidence.attention_norm_output,
        attention_mask=torch.ones(BATCH, SEQUENCE, dtype=torch.bool),
        position_ids=torch.arange(SEQUENCE).reshape(1, -1).expand(BATCH, SEQUENCE),
    )
    assert _relative_rmse(installed.to(dtype=torch.float64), replayed) < 1e-4

    child.load_state_dict(snapshot)
    assert _state_digest(child) == before


# ---------------------------------------------------------------------------
# Fail-closed conditions
# ---------------------------------------------------------------------------


def test_a_stride_change_fails_closed_as_a_kind_translation() -> None:
    _module, evidence = _evidence(131)

    with pytest.raises(HFCausalSparseLatentFoldError) as error:
        _compile(evidence, _contracted(summary_stride=8))
    assert "contract constant" in str(error.value)
    assert "kind translation" in str(error.value)


def test_a_window_change_fails_closed_as_a_band_restriction() -> None:
    _module, evidence = _evidence(133)

    with pytest.raises(HFCausalSparseLatentFoldError) as error:
        _compile(evidence, _contracted(local_window=4))
    assert "band" in str(error.value)


def test_a_width_change_without_a_chart_fails_closed() -> None:
    _module, evidence = _evidence(135)
    narrow = _rows(1350, hidden=HIDDEN // 2).to(dtype=torch.float64)

    with pytest.raises(HFCausalSparseLatentFoldError) as error:
        _compile(
            evidence,
            _contracted(hidden_size=HIDDEN // 2),
            child_normalized_rows=narrow,
        )
    assert "orthonormal score_chart" in str(error.value)


def test_a_cross_width_fold_under_an_orthonormal_chart_compiles() -> None:
    _module, evidence = _evidence(139)
    generator = torch.Generator().manual_seed(1390)
    chart, _residual = torch.linalg.qr(
        torch.randn(HIDDEN, HIDDEN // 2, generator=generator, dtype=torch.float64)
    )
    child_rows = evidence.attention_norm_output @ chart

    fold = _compile(
        evidence,
        _contracted(hidden_size=HIDDEN // 2),
        child_normalized_rows=child_rows,
        score_chart=chart,
    )

    assert fold.report["score_chart_used"] is True
    assert fold.report["score_chart_orthonormality_max_abs_error"] < 1e-8
    assert fold.report["validation_residual_relative_rmse"] < fold.report[
        "validation_no_branch_relative_rmse"
    ]


def test_a_non_orthonormal_chart_fails_closed() -> None:
    _module, evidence = _evidence(141)
    generator = torch.Generator().manual_seed(1410)
    chart = torch.randn(HIDDEN, HIDDEN // 2, generator=generator, dtype=torch.float64)

    with pytest.raises(HFCausalSparseLatentFoldError) as error:
        _compile(
            evidence,
            _contracted(hidden_size=HIDDEN // 2),
            child_normalized_rows=evidence.attention_norm_output @ chart,
            score_chart=chart,
        )
    assert "orthonormal columns" in str(error.value)


def test_the_translation_entry_point_fails_closed_with_its_citation() -> None:
    with pytest.raises(HFCausalSparseLatentFoldError) as error:
        compile_gqa_to_summary_attention_translation()
    message = str(error.value)
    assert "KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md" in message
    assert "fails closed" in message


# ---------------------------------------------------------------------------
# Stage-2 comparability hook -- the exact three-way translation split
# ---------------------------------------------------------------------------


def _three_way(probabilities, values, writer, *, window: int, stride: int):
    return gqa_to_summary_three_way_split(
        attention_probabilities=probabilities,
        attention_value_sequences=values,
        attention_output_weight=writer,
        query_heads=1,
        key_value_heads=1,
        head_dim=int(values.shape[2]),
        local_window=window,
        summary_stride=stride,
        row_mask=torch.ones(
            int(values.shape[0]), int(values.shape[1]), dtype=torch.bool
        ),
    )


def _uniform_causal_probabilities(sequence: int) -> torch.Tensor:
    allowed = torch.ones(sequence, sequence, dtype=torch.float64).tril()
    return (allowed / allowed.sum(dim=-1, keepdim=True)).reshape(1, 1, sequence, sequence)


def test_block_constant_values_leave_nothing_in_the_deviation_term() -> None:
    sequence, stride, width = 16, 4, 3
    generator = torch.Generator().manual_seed(211)
    block_values = torch.randn(sequence // stride, width, generator=generator, dtype=torch.float64)
    values = block_values.repeat_interleave(stride, dim=0).reshape(1, sequence, width)
    writer = torch.eye(width, dtype=torch.float64)

    split = _three_way(
        _uniform_causal_probabilities(sequence),
        values,
        writer,
        window=stride,
        stride=stride,
    )

    assert split.report["rho_far_deviation"] < 1e-24
    assert float(split.deviation_delta.abs().max().item()) < 1e-12
    assert split.report["representable_energy_fraction"] > 0.999


def test_within_block_variation_lands_entirely_in_the_deviation_term() -> None:
    sequence, stride, width = 16, 4, 3
    generator = torch.Generator().manual_seed(213)
    values = torch.randn(1, sequence, width, generator=generator, dtype=torch.float64)
    writer = torch.eye(width, dtype=torch.float64)

    split = _three_way(
        _uniform_causal_probabilities(sequence),
        values,
        writer,
        window=stride,
        stride=stride,
    )

    assert split.report["rho_far_deviation"] > 0.0
    total = (
        split.window_delta
        + split.block_mean_delta
        + split.deviation_delta
        + split.coverage_gap_delta
    )
    contexts = torch.matmul(
        _uniform_causal_probabilities(sequence), values.reshape(1, 1, sequence, width)
    ).reshape(1, sequence, width)
    assert _relative_rmse(total, contexts @ writer.T) < 1e-12


def test_an_hca_stride_reports_its_coverage_gap_as_its_own_fraction() -> None:
    sequence, stride, window, width = 32, 8, 4, 3
    generator = torch.Generator().manual_seed(215)
    values = torch.randn(1, sequence, width, generator=generator, dtype=torch.float64)
    writer = torch.eye(width, dtype=torch.float64)

    split = _three_way(
        _uniform_causal_probabilities(sequence),
        values,
        writer,
        window=window,
        stride=stride,
    )

    assert split.report["rho_coverage_gap"] > 0.0
    assert split.report["split_is_exact"] is True
    # Content outside the window and inside an incomplete block is
    # unrepresentable by construction, not by defect.
    assert split.report["rho_window"] + split.report["rho_far_block_mean"] == pytest.approx(
        split.report["representable_energy_fraction"], rel=0.5
    )


def _global_probabilities(evidence):
    """The unwindowed softmax a *global* GQA parent would produce here.

    The translation source is a global attention, so splitting the branch's own
    already-windowed probabilities would put all mass in ``rho_window`` by
    construction and prove nothing.
    """

    def heads(flat, count):
        return (
            flat.reshape(BATCH, SEQUENCE, count, 8).permute(0, 2, 1, 3).contiguous()
        )

    _scores, probabilities = causal_attention_probabilities(
        heads(evidence.local_query_rows, 4),
        heads(evidence.local_key_rows, 2),
        attention_mask=torch.ones(BATCH, SEQUENCE, dtype=torch.bool),
        scaling=1.0 / math.sqrt(8.0),
    )
    return probabilities


def test_the_three_way_split_of_a_global_parent_is_exact() -> None:
    module, evidence = _evidence(217)
    probabilities = _global_probabilities(evidence)

    split = gqa_to_summary_three_way_split(
        attention_probabilities=probabilities,
        attention_value_sequences=evidence.local_value_rows,
        attention_output_weight=module.local.out.weight,
        query_heads=4,
        key_value_heads=2,
        head_dim=8,
        local_window=WINDOW,
        summary_stride=CSA_STRIDE,
        row_mask=torch.ones(BATCH, SEQUENCE, dtype=torch.bool),
    )

    report = split.report
    assert report["parent_forward_calls"] == 0
    assert 0.0 < report["rho_window"] < 1.0
    assert report["rho_far_block_mean"] > 0.0
    assert report["rho_far_deviation"] > 0.0
    # A CSA stride no larger than the window leaves no coverage gap.
    assert report["rho_coverage_gap"] == pytest.approx(0.0, abs=1e-24)

    contexts = torch.matmul(
        probabilities,
        evidence.local_value_rows.reshape(BATCH, SEQUENCE, 2, 8)
        .permute(0, 2, 1, 3)
        .repeat_interleave(2, dim=1),
    )
    reference = contexts.permute(0, 2, 1, 3).reshape(
        BATCH, SEQUENCE, 32
    ) @ module.local.out.weight.detach().to(dtype=torch.float64).T
    total = (
        split.window_delta
        + split.block_mean_delta
        + split.deviation_delta
        + split.coverage_gap_delta
    )
    assert _relative_rmse(total, reference) < 1e-12


def test_an_hca_stride_wider_than_the_window_opens_a_real_coverage_gap() -> None:
    module, evidence = _evidence(219, stride=HCA_STRIDE, family="hca")
    probabilities = _global_probabilities(evidence)

    def split_at(stride: int):
        return gqa_to_summary_three_way_split(
            attention_probabilities=probabilities,
            attention_value_sequences=evidence.local_value_rows,
            attention_output_weight=module.local.out.weight,
            query_heads=4,
            key_value_heads=2,
            head_dim=8,
            local_window=WINDOW,
            summary_stride=stride,
            row_mask=torch.ones(BATCH, SEQUENCE, dtype=torch.bool),
        )

    csa = split_at(CSA_STRIDE)
    hca = split_at(HCA_STRIDE)

    assert csa.report["rho_coverage_gap"] == pytest.approx(0.0, abs=1e-24)
    assert hca.report["rho_coverage_gap"] > 0.0
    assert hca.report["representable_energy_fraction"] < csa.report[
        "representable_energy_fraction"
    ]
