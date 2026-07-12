import hashlib
import json

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_output_jacobian_orthogonal_basis import (  # noqa: E402
    MINIMUM_GUIDED_RANDOM_CAPTURE_RATIO,
    MINIMUM_NORMAL_ENERGY_FRACTION,
    OutputJacobianDegenerateError,
    OutputJacobianFiniteError,
    OutputJacobianShapeError,
    build_output_jacobian_guided_orthogonal_basis,
    build_output_jacobian_orthogonal_basis,
)


def _problem():
    # The inherited state occupies output/input coordinate zero.  The normal
    # Jacobian is diagonal on coordinates 1: with a unique leading direction.
    jacobian = torch.diag(
        torch.tensor([1.0, 8.0, 2.0, 0.5], dtype=torch.float64)
    )
    left = torch.tensor([[2.0], [0.0], [0.0], [0.0]], dtype=torch.float64)
    right = torch.tensor([[3.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    return jacobian, left, right


def _identity_error(columns):
    gram = columns.T @ columns
    target = torch.zeros_like(gram)
    target.diagonal().fill_(1.0)
    return torch.max(torch.abs(gram - target)).item()


def test_builds_two_sided_normal_jacobian_and_guided_right_factor_rows() -> None:
    jacobian, left, right = _problem()

    result = build_output_jacobian_orthogonal_basis(
        jacobian,
        left,
        right,
        1,
        random_seed=71012,
    )

    expected_normal = jacobian.clone()
    expected_normal[0, :] = 0.0
    expected_normal[:, 0] = 0.0
    torch.testing.assert_close(result.normal_jacobian, expected_normal)
    assert tuple(result.guided_basis.shape) == (4, 1)
    assert tuple(result.right_factor_rows.shape) == (1, 4)
    torch.testing.assert_close(
        result.right_factor_rows,
        result.guided_basis.T,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.guided_basis[:, 0],
        torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64),
        rtol=0.0,
        atol=1e-15,
    )
    assert _identity_error(result.guided_basis) < 1e-14
    assert _identity_error(result.random_control_basis) < 1e-14
    assert torch.max(
        torch.abs(result.inherited_row_basis.T @ result.guided_basis)
    ).item() < 1e-14
    assert torch.max(
        torch.abs(result.inherited_row_basis.T @ result.random_control_basis)
    ).item() < 1e-14


def test_metrics_use_exact_normal_energy_capture_denominator() -> None:
    jacobian, left, right = _problem()
    result = build_output_jacobian_orthogonal_basis(
        jacobian,
        left,
        right,
        1,
        random_seed=31,
    )
    report = result.report

    total = float(torch.sum(jacobian.square()).item())
    normal = float(torch.sum(result.normal_jacobian.square()).item())
    guided = float(
        torch.sum((result.normal_jacobian @ result.guided_basis).square()).item()
    )
    random = float(
        torch.sum(
            (result.normal_jacobian @ result.random_control_basis).square()
        ).item()
    )
    assert report.total_energy == total
    assert report.normal_energy == normal
    assert report.normal_energy_fraction == normal / total
    assert report.guided_capture_energy == guided
    assert report.guided_capture_fraction == guided / normal
    assert report.random_control_capture_energy == random
    assert report.random_control_capture_fraction == random / normal
    assert report.guided_random_capture_ratio == guided / random
    assert report.guided_capture_fraction == pytest.approx(64.0 / 68.25)
    assert report.selected_singular_values == pytest.approx((8.0,))


def test_preregistered_thresholds_are_fixed_and_conjunctive() -> None:
    jacobian, left, right = _problem()
    result = build_output_jacobian_orthogonal_basis(
        jacobian,
        left,
        right,
        1,
        random_seed=71012,
    )
    report = result.report

    assert report.minimum_normal_energy_fraction == 0.20
    assert report.minimum_guided_random_capture_ratio == 1.50
    assert report.minimum_normal_energy_fraction == (
        MINIMUM_NORMAL_ENERGY_FRACTION
    )
    assert report.minimum_guided_random_capture_ratio == (
        MINIMUM_GUIDED_RANDOM_CAPTURE_RATIO
    )
    assert report.normal_energy_threshold_passed is (
        report.normal_energy_fraction >= 0.20
    )
    assert report.guided_random_ratio_threshold_passed is (
        report.guided_random_capture_ratio >= 1.50
    )
    assert report.preregistration_passed is (
        report.normal_energy_threshold_passed
        and report.guided_random_ratio_threshold_passed
    )

    mostly_inherited = jacobian.clone()
    mostly_inherited[0, 0] = 100.0
    failed = build_output_jacobian_orthogonal_basis(
        mostly_inherited,
        left,
        right,
        1,
        random_seed=71012,
    ).report
    assert failed.normal_energy_threshold_passed is False
    assert failed.preregistration_passed is False


def test_seed_binds_equal_rank_random_control_and_report() -> None:
    jacobian, left, right = _problem()
    first = build_output_jacobian_orthogonal_basis(
        jacobian, left, right, 2, random_seed=47
    )
    repeated = build_output_jacobian_orthogonal_basis(
        jacobian, left, right, 2, random_seed=47
    )
    alternate = build_output_jacobian_orthogonal_basis(
        jacobian, left, right, 2, random_seed=48
    )

    torch.testing.assert_close(first.guided_basis, repeated.guided_basis)
    torch.testing.assert_close(
        first.random_control_basis,
        repeated.random_control_basis,
        rtol=0.0,
        atol=0.0,
    )
    assert first.report.random_control_basis_sha256 == (
        repeated.report.random_control_basis_sha256
    )
    assert first.report.report_sha256 == repeated.report.report_sha256
    assert first.report.canonical_json() == repeated.report.canonical_json()
    assert first.report.random_control_basis_sha256 != (
        alternate.report.random_control_basis_sha256
    )
    assert first.report.report_sha256 != alternate.report.report_sha256
    assert first.report.guided_basis_sha256 == alternate.report.guided_basis_sha256
    assert first.random_control_basis.shape == first.guided_basis.shape == (4, 2)


def test_report_json_is_canonical_hash_bound_and_orientation_explicit() -> None:
    jacobian, left, right = _problem()
    report = build_output_jacobian_orthogonal_basis(
        jacobian,
        left,
        right,
        1,
        random_seed=5,
    ).report

    payload = report.as_dict()
    assert json.loads(report.canonical_json()) == payload
    assert report.canonical_json() == json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert report.report_sha256 == hashlib.sha256(
        report.canonical_payload_json().encode("utf-8")
    ).hexdigest()
    assert payload["report_sha256"] == report.report_sha256
    assert payload["orientation"]["guided_basis"].startswith(
        "[input_columns, requested_rank]"
    )
    assert payload["orientation"]["right_factor_rows"] == (
        "[requested_rank, input_columns]"
    )
    assert payload["energy"]["capture_fraction_denominator"] == "normal_energy"


def test_inputs_are_not_mutated_or_attached_to_autograd() -> None:
    jacobian, left, right = _problem()
    jacobian.requires_grad_(True)
    left.requires_grad_(True)
    right.requires_grad_(True)
    originals = tuple(value.detach().clone() for value in (jacobian, left, right))

    result = build_output_jacobian_orthogonal_basis(
        jacobian,
        left,
        right,
        1,
        random_seed=9,
    )

    for value, original in zip((jacobian, left, right), originals, strict=True):
        torch.testing.assert_close(value.detach(), original, rtol=0.0, atol=0.0)
        assert value.grad is None
    for value in (
        result.inherited_column_basis,
        result.inherited_row_basis,
        result.normal_jacobian,
        result.guided_basis,
        result.right_factor_rows,
        result.random_control_basis,
    ):
        assert value.device.type == "cpu"
        assert value.dtype == torch.float64
        assert value.requires_grad is False
        assert value.grad_fn is None
    assert result.report.input_mutation is False
    assert result.report.backward_used is False
    assert result.report.optimizer_used is False


def test_canonical_pivots_are_positive_and_ties_use_lowest_index() -> None:
    # Coordinate 1 and 2 have equal magnitude in the unique leading right
    # singular vector.  argmax must choose coordinate 1 and canonicalize it +.
    jacobian = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -3.0, -3.0, 0.0],
            [0.0, 0.2, -0.2, 0.0],
            [0.0, 0.0, 0.0, 0.1],
        ],
        dtype=torch.float64,
    )
    left = torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float64)
    right = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

    result = build_output_jacobian_orthogonal_basis(
        jacobian,
        left,
        right,
        1,
        random_seed=7,
    )

    assert result.report.guided_canonical_pivot_indices == (1,)
    assert result.guided_basis[1, 0].item() > 0.0
    for column, pivot in enumerate(
        result.report.random_canonical_pivot_indices
    ):
        assert result.random_control_basis[pivot, column].item() > 0.0


def test_alias_has_identical_behavior() -> None:
    jacobian, left, right = _problem()
    direct = build_output_jacobian_orthogonal_basis(
        jacobian, left, right, 1, random_seed=11
    )
    alias = build_output_jacobian_guided_orthogonal_basis(
        jacobian, left, right, 1, random_seed=11
    )
    assert direct.report.canonical_json() == alias.report.canonical_json()
    torch.testing.assert_close(direct.guided_basis, alias.guided_basis)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ((torch.ones(4), None, None), "G must be two-dimensional"),
        ((None, torch.ones(4), None), "L0 must be two-dimensional"),
        ((None, None, torch.ones(4)), "R0 must be two-dimensional"),
        ((torch.ones(3, 4), None, None), "L0 rows must equal"),
        ((None, torch.ones(4, 2), None), "L0 columns must equal"),
        ((None, None, torch.ones(1, 3)), "R0 columns must equal"),
    ],
)
def test_shape_mismatches_fail_closed(replacement, message) -> None:
    jacobian, left, right = _problem()
    new_jacobian, new_left, new_right = replacement
    with pytest.raises(OutputJacobianShapeError, match=message):
        build_output_jacobian_orthogonal_basis(
            jacobian if new_jacobian is None else new_jacobian,
            left if new_left is None else new_left,
            right if new_right is None else new_right,
            1,
        )


@pytest.mark.parametrize("rank", [True, False, 0, -1, 1.5, "1"])
def test_invalid_rank_fails_closed(rank) -> None:
    jacobian, left, right = _problem()
    with pytest.raises(OutputJacobianShapeError, match="rank must"):
        build_output_jacobian_orthogonal_basis(jacobian, left, right, rank)


@pytest.mark.parametrize("seed", [True, -1, 2**63, 1.5, "7"])
def test_invalid_random_seed_fails_closed(seed) -> None:
    jacobian, left, right = _problem()
    with pytest.raises(OutputJacobianShapeError, match="random_seed"):
        build_output_jacobian_orthogonal_basis(
            jacobian, left, right, 1, random_seed=seed
        )


@pytest.mark.parametrize("name", ["jacobian", "left", "right"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs_fail_closed(name, bad) -> None:
    jacobian, left, right = _problem()
    values = {"jacobian": jacobian, "left": left, "right": right}
    values[name] = values[name].clone()
    values[name].reshape(-1)[0] = bad
    with pytest.raises(OutputJacobianFiniteError, match="NaN or infinity"):
        build_output_jacobian_orthogonal_basis(
            values["jacobian"], values["left"], values["right"], 1
        )


@pytest.mark.parametrize(
    "bad",
    [
        torch.ones(4, 4, dtype=torch.int64),
        torch.ones(4, 4, dtype=torch.complex128),
        torch.ones(4, 4).to_sparse(),
    ],
)
def test_nondense_or_nonreal_floating_jacobian_fails_closed(bad) -> None:
    _jacobian, left, right = _problem()
    with pytest.raises(OutputJacobianShapeError):
        build_output_jacobian_orthogonal_basis(bad, left, right, 1)


def test_rank_deficient_inherited_factors_fail_closed() -> None:
    jacobian = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    left = torch.tensor(
        [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    right = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    with pytest.raises(OutputJacobianDegenerateError, match="effective rank"):
        build_output_jacobian_orthogonal_basis(jacobian, left, right, 1)


def test_zero_total_or_zero_normal_energy_fails_closed() -> None:
    jacobian, left, right = _problem()
    with pytest.raises(OutputJacobianDegenerateError, match="zero total energy"):
        build_output_jacobian_orthogonal_basis(
            torch.zeros_like(jacobian), left, right, 1
        )
    inherited_only = torch.zeros_like(jacobian)
    inherited_only[0, 0] = 4.0
    with pytest.raises(OutputJacobianDegenerateError, match="zero energy outside"):
        build_output_jacobian_orthogonal_basis(inherited_only, left, right, 1)


def test_insufficient_normal_rank_and_complement_dimension_fail_closed() -> None:
    jacobian, left, right = _problem()
    rank_one_normal = torch.zeros_like(jacobian)
    rank_one_normal[1, 2] = 1.0
    with pytest.raises(OutputJacobianDegenerateError, match="below requested rank"):
        build_output_jacobian_orthogonal_basis(
            rank_one_normal, left, right, 2
        )
    with pytest.raises(OutputJacobianShapeError, match="cannot exceed"):
        build_output_jacobian_orthogonal_basis(jacobian, left, right, 4)


def test_rank_cutting_degenerate_singular_block_fails_closed() -> None:
    jacobian = torch.diag(torch.tensor([1.0, 5.0, 5.0, 2.0]))
    left = torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float64)
    right = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    with pytest.raises(OutputJacobianDegenerateError, match="degenerate"):
        build_output_jacobian_orthogonal_basis(jacobian, left, right, 1)


def test_complete_degenerate_selected_block_is_coordinate_canonical() -> None:
    jacobian = torch.diag(torch.tensor([1.0, 5.0, 5.0, 2.0]))
    left = torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float64)
    right = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    result = build_output_jacobian_orthogonal_basis(
        jacobian,
        left,
        right,
        2,
        random_seed=3,
    )
    expected = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result.guided_basis, expected, atol=1e-14, rtol=0.0)
    assert result.report.guided_canonical_pivot_indices == (1, 2)

