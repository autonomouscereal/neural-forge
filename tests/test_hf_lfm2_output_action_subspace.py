from __future__ import annotations

import inspect

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_output_action_subspace as output_pod  # noqa: E402
from neural_forge.hf_lfm2_output_action_subspace import (  # noqa: E402
    Lfm2OutputActionRankDeficitError,
    Lfm2OutputActionSubspaceError,
    compile_lfm2_output_action_subspace,
    seal_lfm2_output_action_evidence,
    seal_lfm2_output_action_rows,
)


_PARENT = "a" * 64
_CACHE = "b" * 64
_COMPONENT = "model.layers.2.self_attn.output_action"


def _fit_rows(
    actions: torch.Tensor,
    *,
    weights=None,
    ids: tuple[str, ...] | None = None,
):
    if ids is None:
        ids = tuple(f"fit-{index:03d}" for index in range(actions.shape[0]))
    pre = (
        torch.arange(
            actions.numel(),
            dtype=actions.dtype,
        ).reshape_as(actions)
        / 17.0
    )
    post = pre + actions
    return seal_lfm2_output_action_rows(
        role="fit",
        row_ids=ids,
        pre_rows=pre,
        post_rows=post,
        case_weights=weights,
    )


def _heldout_rows(
    actions: torch.Tensor,
    *,
    suffix: str = "a",
    weights=None,
):
    return seal_lfm2_output_action_rows(
        role="heldout",
        row_ids=tuple(
            f"heldout-{suffix}-{index:03d}" for index in range(actions.shape[0])
        ),
        branch_action_rows=actions,
        case_weights=weights,
    )


def _evidence(fit, heldout=None):
    return seal_lfm2_output_action_evidence(
        parent_state_sha256=_PARENT,
        cache_sha256=_CACHE,
        component_id=_COMPONENT,
        fit=fit,
        heldout=heldout,
    )


def test_pre_post_actions_compile_canonical_degenerate_basis() -> None:
    actions = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    # Deliberately reverse the IDs and values.  Sealing canonicalizes them as
    # aligned pairs before deriving the covariance.
    fit = _fit_rows(
        actions.flip(0),
        ids=("fit-003", "fit-002", "fit-001", "fit-000"),
    )
    result = compile_lfm2_output_action_subspace(
        _evidence(fit),
        target_rank=2,
    )

    expected = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result.output_basis, expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        result.projector,
        expected @ expected.T,
        atol=1e-12,
        rtol=0,
    )
    assert result.report["compute_device"] == "cpu"
    assert result.report["compute_dtype"] == "float64"
    assert all(
        group["canonicalized_from_projector"]
        for group in result.report["canonical_eigenspace_groups"]
    )


def test_direct_action_and_pre_post_sources_have_same_geometry() -> None:
    generator = torch.Generator().manual_seed(101)
    actions = torch.randn(12, 5, generator=generator, dtype=torch.float64)
    paired = _fit_rows(actions)
    direct = seal_lfm2_output_action_rows(
        role="fit",
        row_ids=paired.row_ids,
        branch_action_rows=actions,
    )

    paired_result = compile_lfm2_output_action_subspace(
        _evidence(paired), target_rank=3
    )
    direct_result = compile_lfm2_output_action_subspace(
        _evidence(direct), target_rank=3
    )

    torch.testing.assert_close(
        paired_result.output_basis,
        direct_result.output_basis,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        paired_result.projector,
        direct_result.projector,
        atol=1e-12,
        rtol=1e-12,
    )
    assert paired.source_mode == "post_minus_pre"
    assert direct.source_mode == "sealed_branch_action"


def test_heldout_rows_are_diagnostic_only_and_cannot_change_basis() -> None:
    generator = torch.Generator().manual_seed(202)
    fit = _fit_rows(torch.randn(16, 6, generator=generator, dtype=torch.float64))
    heldout_a = _heldout_rows(
        torch.randn(7, 6, generator=generator, dtype=torch.float64),
        suffix="a",
    )
    heldout_b = _heldout_rows(
        torch.randn(7, 6, generator=generator, dtype=torch.float64) * 1000.0,
        suffix="b",
    )

    result_a = compile_lfm2_output_action_subspace(
        _evidence(fit, heldout_a), target_rank=3
    )
    result_b = compile_lfm2_output_action_subspace(
        _evidence(fit, heldout_b), target_rank=3
    )

    assert torch.equal(result_a.output_basis, result_b.output_basis)
    assert torch.equal(result_a.projector, result_b.projector)
    assert result_a.basis_receipt_sha256 == result_b.basis_receipt_sha256
    assert result_a.result_sha256 != result_b.result_sha256
    assert result_a.report["heldout_influences_basis"] is False
    assert result_a.report["heldout_policy"] == "diagnostic-only-no-basis-influence"
    assert (
        result_a.report["heldout_diagnostics"] != result_b.report["heldout_diagnostics"]
    )


def test_positive_case_weights_control_fit_energy_without_an_optimizer() -> None:
    actions = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=torch.float64,
    )
    first = compile_lfm2_output_action_subspace(
        _evidence(_fit_rows(actions, weights=[10.0, 10.0, 1.0, 1.0])),
        target_rank=1,
    )
    second = compile_lfm2_output_action_subspace(
        _evidence(_fit_rows(actions, weights=[1.0, 1.0, 10.0, 10.0])),
        target_rank=1,
    )

    torch.testing.assert_close(
        first.output_basis,
        torch.tensor([[1.0], [0.0]], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(
        second.output_basis,
        torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    assert first.report["fit_case_weights_supplied"] is True


def test_encode_decode_project_and_materialize_are_consistent() -> None:
    generator = torch.Generator().manual_seed(303)
    actions = torch.randn(20, 7, generator=generator, dtype=torch.float64)
    result = compile_lfm2_output_action_subspace(
        _evidence(_fit_rows(actions)),
        target_rank=4,
    )
    probes = torch.randn(5, 7, generator=generator, dtype=torch.float64)

    encoded = result.encode(probes)
    decoded = result.decode(encoded)
    projected = result.project(probes)
    torch.testing.assert_close(encoded, probes @ result.output_basis)
    torch.testing.assert_close(decoded, projected)
    torch.testing.assert_close(projected, probes @ result.projector)

    materialized = result.materialize()
    assert materialized.encoder_weight.dtype == torch.float32
    assert materialized.decoder_weight.dtype == torch.float32
    assert materialized.encoder_weight.shape == (4, 7)
    assert materialized.decoder_weight.shape == (7, 4)
    assert (
        materialized.encoder_weight.untyped_storage().data_ptr()
        != materialized.decoder_weight.untyped_storage().data_ptr()
    )
    state = materialized.state_dict()
    assert tuple(sorted(state)) == ("decoder.weight", "encoder.weight")
    state["encoder.weight"].zero_()
    assert bool((materialized.encoder_weight != 0).any().item())
    report = materialized.report
    assert report["dense_projector_retained_as_parameter"] is False
    assert report["runtime_parameter_sharing_used"] is False
    assert report["materialized_parameter_count"] == 2 * 7 * 4


def test_rank_deficit_fails_closed_with_authenticated_report() -> None:
    actions = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [2.0, 4.0, 0.0],
            [-1.0, -2.0, 0.0],
        ],
        dtype=torch.float64,
    )
    with pytest.raises(Lfm2OutputActionRankDeficitError) as captured:
        compile_lfm2_output_action_subspace(
            _evidence(_fit_rows(actions)),
            target_rank=2,
        )

    report = captured.value.report
    assert report["status"] == "fit-action-rank-deficit"
    assert report["observed_rank"] == 1
    assert report["requested_rank"] == 2
    assert len(report["report_sha256"]) == 64


def test_fit_heldout_overlap_and_width_mismatch_fail_closed() -> None:
    fit = _fit_rows(
        torch.eye(3, dtype=torch.float64),
        ids=("same-0", "same-1", "same-2"),
    )
    overlapping = seal_lfm2_output_action_rows(
        role="heldout",
        row_ids=("same-0", "heldout-1", "heldout-2"),
        branch_action_rows=torch.eye(3, dtype=torch.float64),
    )
    with pytest.raises(Lfm2OutputActionSubspaceError, match="overlap"):
        _evidence(fit, overlapping)

    wrong_width = _heldout_rows(torch.eye(4, dtype=torch.float64))
    with pytest.raises(Lfm2OutputActionSubspaceError, match="different widths"):
        _evidence(fit, wrong_width)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"post_rows": None}, "exactly one complete"),
        (
            {"branch_action_rows": torch.ones(2, 2, dtype=torch.float64)},
            "exactly one complete",
        ),
    ],
)
def test_source_form_must_be_exactly_one_complete_mode(changes, message) -> None:
    arguments = {
        "role": "fit",
        "row_ids": ("a", "b"),
        "pre_rows": torch.zeros(2, 2, dtype=torch.float64),
        "post_rows": torch.ones(2, 2, dtype=torch.float64),
    }
    arguments.update(changes)
    with pytest.raises(Lfm2OutputActionSubspaceError, match=message):
        seal_lfm2_output_action_rows(**arguments)


@pytest.mark.parametrize(
    "weights",
    (
        [1.0, 0.0],
        [1.0, -1.0],
        [1.0, float("nan")],
        [1.0],
    ),
)
def test_nonpositive_nonfinite_or_misaligned_weights_fail_closed(weights) -> None:
    with pytest.raises(Lfm2OutputActionSubspaceError):
        seal_lfm2_output_action_rows(
            role="fit",
            row_ids=("a", "b"),
            branch_action_rows=torch.eye(2, dtype=torch.float64),
            case_weights=weights,
        )


def test_nonfinite_grad_bearing_and_tensor_subclasses_fail_closed() -> None:
    nonfinite = torch.eye(2, dtype=torch.float64)
    nonfinite[0, 0] = torch.inf
    with pytest.raises(Lfm2OutputActionSubspaceError, match="NaN or infinity"):
        seal_lfm2_output_action_rows(
            role="fit",
            row_ids=("a", "b"),
            branch_action_rows=nonfinite,
        )

    parameter = torch.nn.Parameter(torch.eye(2, dtype=torch.float64))
    with pytest.raises(Lfm2OutputActionSubspaceError, match="exact, detached"):
        seal_lfm2_output_action_rows(
            role="fit",
            row_ids=("a", "b"),
            branch_action_rows=parameter,
        )

    class _TensorSubclass(torch.Tensor):
        pass

    subclass = torch.Tensor._make_subclass(
        _TensorSubclass,
        torch.eye(2, dtype=torch.float64),
        False,
    )
    with pytest.raises(Lfm2OutputActionSubspaceError, match="exact, detached"):
        seal_lfm2_output_action_rows(
            role="fit",
            row_ids=("a", "b"),
            branch_action_rows=subclass,
        )


def test_mutated_evidence_and_compiled_tensors_are_rejected() -> None:
    fit = _fit_rows(torch.eye(3, dtype=torch.float64))
    evidence = _evidence(fit)
    fit.action_rows[0, 0] += 1.0
    with pytest.raises(Lfm2OutputActionSubspaceError, match="authentication"):
        compile_lfm2_output_action_subspace(evidence, target_rank=2)

    clean = compile_lfm2_output_action_subspace(
        _evidence(_fit_rows(torch.eye(3, dtype=torch.float64))),
        target_rank=2,
    )
    clean.output_basis[0, 0] += 0.25
    with pytest.raises(Lfm2OutputActionSubspaceError):
        clean.project(torch.eye(3, dtype=torch.float64))

    materialized = compile_lfm2_output_action_subspace(
        _evidence(_fit_rows(torch.eye(3, dtype=torch.float64))),
        target_rank=2,
    ).materialize()
    materialized.encoder_weight[0, 0] += 0.25
    with pytest.raises(Lfm2OutputActionSubspaceError):
        materialized.state_dict()


def test_repeated_compilation_and_row_permutation_are_bit_identical() -> None:
    generator = torch.Generator().manual_seed(404)
    actions = torch.randn(18, 5, generator=generator, dtype=torch.float64)
    ids = tuple(f"fit-{index:03d}" for index in range(18))
    order = torch.randperm(18, generator=generator)
    permuted_ids = tuple(ids[int(index)] for index in order)
    first_fit = seal_lfm2_output_action_rows(
        role="fit",
        row_ids=ids,
        branch_action_rows=actions,
    )
    second_fit = seal_lfm2_output_action_rows(
        role="fit",
        row_ids=permuted_ids,
        branch_action_rows=actions.index_select(0, order),
    )
    first = compile_lfm2_output_action_subspace(_evidence(first_fit), target_rank=3)
    second = compile_lfm2_output_action_subspace(_evidence(second_fit), target_rank=3)

    assert first_fit.seal_sha256 == second_fit.seal_sha256
    assert torch.equal(first.output_basis, second.output_basis)
    assert torch.equal(first.projector, second.projector)
    assert first.result_sha256 == second.result_sha256


def test_materialization_rejects_low_precision_or_quantized_shortcuts() -> None:
    result = compile_lfm2_output_action_subspace(
        _evidence(_fit_rows(torch.eye(4, dtype=torch.float64))),
        target_rank=2,
    )
    with pytest.raises(Lfm2OutputActionSubspaceError, match="FP32 or FP64"):
        result.materialize(dtype=torch.float16)
    with pytest.raises(Lfm2OutputActionSubspaceError, match="FP32 or FP64"):
        result.materialize(dtype=torch.bfloat16)


def test_operation_receipts_and_api_prove_fold_only_surface() -> None:
    result = compile_lfm2_output_action_subspace(
        _evidence(
            _fit_rows(torch.eye(4, dtype=torch.float64)),
            _heldout_rows(torch.eye(4, dtype=torch.float64)),
        ),
        target_rank=3,
    )
    materialized = result.materialize(dtype=torch.float64)

    for audit in (
        result.report["operation_audit"],
        materialized.report["operation_audit"],
    ):
        assert audit["model_objects_accepted"] == 0
        assert audit["parent_model_forward_calls"] == 0
        assert audit["candidate_model_forward_calls"] == 0
        assert audit["parent_trace_recaptures"] == 0
        assert audit["training_steps"] == 0
        assert audit["optimizer_constructions"] == 0
        assert audit["optimizer_steps"] == 0
        assert audit["autograd_grad_calls"] == 0
        assert audit["backward_calls"] == 0
        assert audit["parameters_pruned"] == 0
        assert audit["parameters_quantized"] == 0
        assert audit["parameters_sparsified"] == 0
        assert audit["filesystem_reads"] == 0
        assert audit["filesystem_writes"] == 0
        assert audit["network_access"] is False

    signature = inspect.signature(compile_lfm2_output_action_subspace)
    assert "model" not in signature.parameters
    source = inspect.getsource(output_pod)
    for forbidden in (
        "torch.optim",
        ".backward(",
        "autograd.grad",
        "open(",
        "Path(",
        "requests.",
        "urllib.",
        "socket.",
    ):
        assert forbidden not in source
    assert result.report["training_or_optimization_used"] is False
    assert materialized.report["training_or_optimization_used"] is False
