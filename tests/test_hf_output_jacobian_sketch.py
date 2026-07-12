from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

import neural_forge.hf_output_jacobian_sketch as sketch_module
from neural_forge.hf_output_jacobian_sketch import (
    OutputJacobianBlockIdentity,
    OutputJacobianSketchFiniteError,
    OutputJacobianSketchLimitError,
    OutputJacobianSketchPhaseError,
    OutputJacobianSketchRankError,
    OutputJacobianSketchShapeError,
    OutputJacobianSketchSpec,
    TwoPassOutputJacobianSketch,
    sketch_output_jacobian,
)


def _identity(index: int, fingerprint: str | None = None) -> OutputJacobianBlockIdentity:
    return OutputJacobianBlockIdentity(
        semantic_identity=f"test:block:{index}",
        numeric_fingerprint_sha256=fingerprint or f"{index + 1:064x}",
    )


def _problem() -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(417)
    delta = torch.randn((19, 7), generator=generator)
    inputs = torch.randn((19, 8), generator=generator)
    left = torch.randn((7, 2), generator=generator)
    right = torch.randn((2, 8), generator=generator)
    blocks = [(delta[:5], inputs[:5]), (delta[5:13], inputs[5:13]), (delta[13:], inputs[13:])]
    return blocks, left, right


def _explicit_normal(
    blocks: list[tuple[torch.Tensor, torch.Tensor]],
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u = torch.linalg.svd(left.to(torch.float64), full_matrices=False).U
    v = torch.linalg.svd(right.T.to(torch.float64), full_matrices=False).U
    normal = torch.zeros((left.shape[0], right.shape[1]), dtype=torch.float64)
    raw = torch.zeros_like(normal)
    for delta, inputs in blocks:
        d = delta.to(torch.float64)
        x = inputs.to(torch.float64)
        d_bar = d - (d @ u) @ u.T
        x_bar = x - (x @ v) @ v.T
        raw.add_(d.T @ x)
        normal.add_(d_bar.T @ x_bar)
    return raw, normal, v


def _spec(**overrides: object) -> OutputJacobianSketchSpec:
    values: dict[str, object] = {
        "out_features": 7,
        "in_features": 8,
        "rank": 2,
        "sketch_width": 6,
        "seed": 31,
        "max_bytes": 8 * 1024 * 1024,
    }
    values.update(overrides)
    return OutputJacobianSketchSpec(**values)  # type: ignore[arg-type]


def test_two_pass_core_matches_explicit_normal_jacobian_math() -> None:
    blocks, left, right = _problem()
    accumulator = TwoPassOutputJacobianSketch(_spec(), left, right)
    probe = accumulator._omega.clone()  # noqa: SLF001 - exact core contract check
    for index, (delta, inputs) in enumerate(blocks):
        accumulator.add_pass1(delta, inputs, block_identity=_identity(index))

    raw, normal, _ = _explicit_normal(blocks, left, right)
    expected_raw_y = raw.to(torch.float32).T @ probe
    expected_normal_y = normal.to(torch.float32).T @ probe
    assert torch.allclose(
        accumulator._raw_y, expected_raw_y, atol=2e-5, rtol=2e-5  # noqa: SLF001
    )
    assert torch.allclose(
        accumulator._normal_y,  # noqa: SLF001
        expected_normal_y,
        atol=2e-5,
        rtol=2e-5,
    )

    accumulator.finish_pass1()
    for index, (delta, inputs) in enumerate(blocks):
        accumulator.add_pass2(delta, inputs, block_identity=_identity(index))
    result = accumulator.finalize()

    guided_energy = torch.linalg.vector_norm(
        normal @ result.guided_right_rows.to(torch.float64).T
    ).square()
    random_energy = torch.linalg.vector_norm(
        normal @ result.random_right_rows.to(torch.float64).T
    ).square()
    raw_estimate = expected_raw_y.to(torch.float64).square().sum() / probe.shape[1]
    normal_estimate = (
        expected_normal_y.to(torch.float64).square().sum() / probe.shape[1]
    )
    assert result.report.guided_capture_energy == pytest.approx(
        float(guided_energy), rel=2e-5, abs=2e-5
    )
    assert result.report.random_capture_energy == pytest.approx(
        float(random_energy), rel=2e-5, abs=2e-5
    )
    assert result.report.raw_frobenius_sq_estimate == pytest.approx(
        float(raw_estimate), rel=2e-6
    )
    assert result.report.normal_frobenius_sq_estimate == pytest.approx(
        float(normal_estimate), rel=2e-6
    )
    assert result.report.sample_count == 19
    assert result.report.block_count == 3


def test_full_width_sketch_recovers_explicit_top_right_subspace() -> None:
    generator = torch.Generator(device="cpu").manual_seed(83)
    delta = torch.randn((21, 5), generator=generator)
    inputs = torch.randn((21, 6), generator=generator)
    result = sketch_output_jacobian(
        [(delta[:9], inputs[:9]), (delta[9:], inputs[9:])],
        OutputJacobianSketchSpec(
            out_features=5,
            in_features=6,
            rank=2,
            sketch_width=6,
            seed=8,
            max_bytes=2 * 1024 * 1024,
        ),
    )
    jacobian = delta.to(torch.float64).T @ inputs.to(torch.float64)
    _, singular, vh = torch.linalg.svd(jacobian, full_matrices=False)
    explicit_projector = vh[:2].T @ vh[:2]
    guided = result.guided_right_rows.to(torch.float64)
    assert torch.allclose(guided.T @ guided, explicit_projector, atol=2e-5, rtol=2e-5)
    assert result.report.guided_capture_energy == pytest.approx(
        float(singular[:2].square().sum()), rel=2e-5
    )


def test_determinism_canonical_signs_hashes_and_no_input_mutation() -> None:
    blocks, left, right = _problem()
    snapshots = [(delta.clone(), inputs.clone()) for delta, inputs in blocks]
    left_snapshot = left.clone()
    right_snapshot = right.clone()
    first = sketch_output_jacobian(blocks, _spec(), left, right)
    second = sketch_output_jacobian(blocks, _spec(), left, right)

    assert torch.equal(first.guided_right_rows, second.guided_right_rows)
    assert torch.equal(first.random_right_rows, second.random_right_rows)
    assert first.report == second.report
    assert first.report.report_sha256 == second.report.report_sha256
    for row, pivot in zip(
        first.guided_right_rows, first.report.guided_sign_pivots, strict=True
    ):
        assert row[pivot] >= 0.0
    for row, pivot in zip(
        first.random_right_rows, first.report.random_sign_pivots, strict=True
    ):
        assert row[pivot] >= 0.0
    for (delta, inputs), (expected_delta, expected_inputs) in zip(
        blocks, snapshots, strict=True
    ):
        assert torch.equal(delta, expected_delta)
        assert torch.equal(inputs, expected_inputs)
    assert torch.equal(left, left_snapshot)
    assert torch.equal(right, right_snapshot)

    payload = first.report.to_dict()
    report_hash = payload.pop("report_sha256")
    encoded = json.dumps(
        payload, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    assert report_hash == hashlib.sha256(encoded).hexdigest()


def test_accumulator_device_is_explicit_and_outputs_stay_on_it() -> None:
    blocks, left, right = _problem()
    accumulator = TwoPassOutputJacobianSketch(
        _spec(accumulator_device="cpu"), left, right
    )
    assert accumulator.accumulator_device == torch.device("cpu")
    for index, (delta, inputs) in enumerate(blocks):
        accumulator.add_pass1(delta, inputs, block_identity=_identity(index))
    accumulator.finish_pass1()
    assert accumulator._q.device.type == "cpu"  # noqa: SLF001
    assert accumulator._control.device.type == "cpu"  # noqa: SLF001
    for index, (delta, inputs) in enumerate(blocks):
        accumulator.add_pass2(delta, inputs, block_identity=_identity(index))
    result = accumulator.finalize()
    assert result.guided_right_rows.device.type == "cpu"
    assert result.random_right_rows.device.type == "cpu"
    assert result.report.accumulator_device == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_accumulation_avoids_cpu_block_transfer() -> None:
    blocks, left, right = _problem()
    cuda_blocks = [(delta.cuda(), inputs.cuda()) for delta, inputs in blocks]
    accumulator = TwoPassOutputJacobianSketch(
        _spec(accumulator_device="cuda"), left.cuda(), right.cuda()
    )
    assert accumulator.accumulator_device.type == "cuda"
    for index, (delta, inputs) in enumerate(cuda_blocks):
        prepared_delta, prepared_inputs = accumulator._prepare_block(  # noqa: SLF001
            delta, inputs, pass_number=1
        )
        assert prepared_delta.data_ptr() == delta.data_ptr()
        assert prepared_inputs.data_ptr() == inputs.data_ptr()
        accumulator.add_pass1(delta, inputs, block_identity=_identity(index))
    accumulator.finish_pass1()
    assert accumulator._q.is_cuda  # noqa: SLF001
    assert accumulator._control.is_cuda  # noqa: SLF001
    for index, (delta, inputs) in enumerate(cuda_blocks):
        accumulator.add_pass2(delta, inputs, block_identity=_identity(index))
    result = accumulator.finalize()
    assert result.guided_right_rows.is_cuda
    assert result.random_right_rows.is_cuda
    assert result.report.accumulator_device.startswith("cuda")


def test_rows_are_orthonormal_and_normal_to_inherited_input_space() -> None:
    blocks, left, right = _problem()
    result = sketch_output_jacobian(blocks, _spec(), left, right)
    v = torch.linalg.svd(right.T.to(torch.float64), full_matrices=False).U
    identity = torch.eye(2, dtype=torch.float64)
    for rows in (result.guided_right_rows, result.random_right_rows):
        rows64 = rows.to(torch.float64)
        assert torch.allclose(rows64 @ rows64.T, identity, atol=2e-6, rtol=2e-6)
        assert torch.linalg.vector_norm(rows64 @ v) < 2e-6


def test_numerically_zero_inherited_product_has_empty_occupied_spaces() -> None:
    blocks, _, _ = _problem()
    generator = torch.Generator(device="cpu").manual_seed(12)
    left = torch.zeros((7, 2))
    right = torch.zeros((2, 8))
    left[:, 0] = torch.randn((7,), generator=generator)
    right[1] = torch.randn((8,), generator=generator)
    assert torch.count_nonzero(left @ right) == 0

    with_factors = sketch_output_jacobian(blocks, _spec(), left, right)
    without_factors = sketch_output_jacobian(blocks, _spec())
    assert with_factors.report.inherited_product_numerically_zero is True
    assert with_factors.report.occupied_output_rank == 0
    assert with_factors.report.occupied_input_rank == 0
    assert torch.equal(
        with_factors.guided_right_rows, without_factors.guided_right_rows
    )
    assert torch.equal(
        with_factors.random_right_rows, without_factors.random_right_rows
    )


def test_occupied_spaces_follow_actual_product_under_latent_cancellation() -> None:
    blocks, _, _ = _problem()
    left = torch.zeros((7, 3))
    right = torch.zeros((3, 8))
    left[0, 0] = 1.0
    left[1, 1:] = 1.0
    right[0, 0] = 2.0
    right[1, 1] = 3.0
    right[2, 1] = -3.0
    product = left @ right
    assert torch.linalg.matrix_rank(left) == 2
    assert torch.linalg.matrix_rank(right) == 2
    assert torch.linalg.matrix_rank(product) == 1

    accumulator = TwoPassOutputJacobianSketch(_spec(), left, right)
    assert accumulator._u.shape == (7, 1)  # noqa: SLF001
    assert accumulator._v.shape == (8, 1)  # noqa: SLF001
    for index, (delta, inputs) in enumerate(blocks):
        accumulator.add_pass1(delta, inputs, block_identity=_identity(index))
    accumulator.finish_pass1()
    for index, (delta, inputs) in enumerate(blocks):
        accumulator.add_pass2(delta, inputs, block_identity=_identity(index))
    result = accumulator.finalize()
    assert result.report.occupied_output_rank == 1
    assert result.report.occupied_input_rank == 1
    assert result.report.inherited_product_numerically_zero is False
    expected_u = torch.zeros((7, 1))
    expected_v = torch.zeros((8, 1))
    expected_u[0, 0] = 1.0
    expected_v[0, 0] = 1.0
    assert torch.allclose(accumulator._u.abs(), expected_u)  # noqa: SLF001
    assert torch.allclose(accumulator._v.abs(), expected_v)  # noqa: SLF001


def test_gate_thresholds_are_individually_audited() -> None:
    blocks, left, right = _problem()
    result = sketch_output_jacobian(
        blocks,
        _spec(
            min_normal_fraction=1e6,
            min_guided_capture_fraction=1e6,
            min_guided_over_random_ratio=1e6,
        ),
        left,
        right,
    )
    assert result.report.normal_fraction_gate_passed is False
    assert result.report.guided_capture_gate_passed is False
    assert result.report.guided_over_random_gate_passed is False
    assert result.report.gate_passed is False

    permissive = sketch_output_jacobian(blocks, _spec(), left, right)
    assert permissive.report.normal_fraction_gate_passed is True
    assert permissive.report.guided_capture_gate_passed is True
    assert permissive.report.guided_over_random_gate_passed is True
    assert permissive.report.gate_passed is True


def test_memory_ceiling_rejects_setup_and_batches_before_accumulation() -> None:
    with pytest.raises(OutputJacobianSketchLimitError, match="max_bytes=1"):
        TwoPassOutputJacobianSketch(_spec(max_bytes=1))

    accumulator = TwoPassOutputJacobianSketch(_spec(max_bytes=4096))
    before = accumulator._raw_y.clone()  # noqa: SLF001
    with pytest.raises(OutputJacobianSketchLimitError, match="pass-one batch"):
        accumulator.add_pass1(
            torch.zeros((100, 7)),
            torch.zeros((100, 8)),
            block_identity=_identity(0),
        )
    assert torch.equal(accumulator._raw_y, before)  # noqa: SLF001
    assert accumulator.phase == "pass1"


def test_public_per_site_persistent_and_concurrent_transient_accounting() -> None:
    blocks, left, right = _problem()
    accumulator = TwoPassOutputJacobianSketch(_spec(), left, right)

    def actual_owned_tensor_bytes() -> int:
        names = (
            "_u",
            "_v",
            "_omega",
            "_raw_y",
            "_normal_y",
            "_q",
            "_control",
            "_b",
            "_random_b",
        )
        return sum(
            int(value.numel() * value.element_size())
            for name in names
            if isinstance((value := getattr(accumulator, name, None)), torch.Tensor)
        )

    assert accumulator.accounted_persistent_bytes == actual_owned_tensor_bytes()
    pass1_small = accumulator.accounted_batch_transient_bytes(3, 1)
    pass1_large = accumulator.accounted_batch_transient_bytes(9, 1)
    pass2_small = accumulator.accounted_batch_transient_bytes(3, 2)
    assert 0 < pass1_small < pass1_large
    assert pass2_small > 0

    accumulator.add_pass1(*blocks[0], block_identity=_identity(0))
    accumulator.finish_pass1()
    assert accumulator.accounted_persistent_bytes == actual_owned_tensor_bytes()
    assert accumulator.accounted_persistent_bytes <= accumulator.accounted_peak_bytes
    with pytest.raises(OutputJacobianSketchShapeError, match="rows must be positive"):
        accumulator.accounted_batch_transient_bytes(0, 1)
    with pytest.raises(OutputJacobianSketchShapeError, match="must be 1 or 2"):
        accumulator.accounted_batch_transient_bytes(1, 3)


def test_nonfinite_shape_and_factor_errors_are_fail_closed() -> None:
    blocks, left, right = _problem()
    bad_left = left.clone()
    bad_left[0, 0] = torch.nan
    with pytest.raises(OutputJacobianSketchFiniteError, match="inherited_left"):
        TwoPassOutputJacobianSketch(_spec(), bad_left, right)
    with pytest.raises(OutputJacobianSketchShapeError, match="supplied together"):
        TwoPassOutputJacobianSketch(_spec(), left, None)
    with pytest.raises(OutputJacobianSketchShapeError, match="wrong output width"):
        TwoPassOutputJacobianSketch(_spec()).add_pass1(
            torch.zeros((2, 6)),
            torch.zeros((2, 8)),
            block_identity=_identity(0),
        )

    accumulator = TwoPassOutputJacobianSketch(_spec())
    delta, inputs = blocks[0]
    bad_delta = delta.clone()
    bad_delta[0, 0] = torch.inf
    with pytest.raises(OutputJacobianSketchFiniteError, match="D is non-finite"):
        accumulator.add_pass1(bad_delta, inputs, block_identity=_identity(0))
    assert accumulator.phase == "pass1"
    assert accumulator._pass1_blocks == 0  # noqa: SLF001


def test_metadata_and_reservation_precede_value_scans_and_conversion() -> None:
    with pytest.raises(OutputJacobianSketchShapeError, match="meta tensor"):
        TwoPassOutputJacobianSketch(_spec()).add_pass1(
            torch.empty((2, 7), device="meta"),
            torch.empty((2, 8), device="meta"),
            block_identity=_identity(0),
        )
    sparse_delta = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]]),
        torch.tensor([1.0]),
        (2, 7),
        check_invariants=True,
    )
    with pytest.raises(OutputJacobianSketchShapeError, match="dense strided"):
        TwoPassOutputJacobianSketch(_spec()).add_pass1(
            sparse_delta,
            torch.zeros((2, 8)),
            block_identity=_identity(0),
        )
    with pytest.raises(OutputJacobianSketchShapeError, match="real floating-point"):
        TwoPassOutputJacobianSketch(_spec()).add_pass1(
            torch.zeros((2, 7), dtype=torch.int64),
            torch.zeros((2, 8)),
            block_identity=_identity(0),
        )

    nonfinite_left = torch.full((7, 2), torch.nan)
    right = torch.ones((2, 8))
    with pytest.raises(OutputJacobianSketchLimitError):
        TwoPassOutputJacobianSketch(
            _spec(max_bytes=1), nonfinite_left, right
        )

    accumulator = TwoPassOutputJacobianSketch(_spec(max_bytes=4096))
    huge_nonfinite = torch.full((100, 7), torch.inf)
    with pytest.raises(OutputJacobianSketchLimitError, match="pass-one batch"):
        accumulator.add_pass1(
            huge_nonfinite,
            torch.zeros((100, 8)),
            block_identity=_identity(0),
        )


def test_invalid_accumulator_device_is_rejected_before_allocation() -> None:
    if torch.cuda.is_available():
        invalid = f"cuda:{torch.cuda.device_count()}"
        match = "invalid CUDA device index"
    else:
        invalid = "cuda:0"
        match = "CUDA accumulator_device was requested"
    with pytest.raises(OutputJacobianSketchShapeError, match=match):
        TwoPassOutputJacobianSketch(_spec(accumulator_device=invalid))


def test_tensor_hash_flattens_elements_before_byte_reinterpretation() -> None:
    tensor = torch.arange(24, dtype=torch.float64).reshape(4, 6).T
    contiguous = tensor.contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode())
    digest.update(memoryview(contiguous.reshape(-1).view(torch.uint8).numpy()))
    assert sketch_module._tensor_sha256(tensor) == digest.hexdigest()  # noqa: SLF001


def test_phase_and_replay_count_errors() -> None:
    blocks, _, _ = _problem()
    accumulator = TwoPassOutputJacobianSketch(_spec())
    with pytest.raises(OutputJacobianSketchPhaseError, match="phase pass2"):
        accumulator.add_pass2(*blocks[0], block_identity=_identity(0))
    with pytest.raises(OutputJacobianSketchPhaseError, match="At least one"):
        accumulator.finish_pass1()

    accumulator.add_pass1(*blocks[0], block_identity=_identity(0))
    accumulator.finish_pass1()
    with pytest.raises(OutputJacobianSketchPhaseError, match="phase pass1"):
        accumulator.add_pass1(*blocks[0], block_identity=_identity(0))
    with pytest.raises(OutputJacobianSketchPhaseError, match="same row and block"):
        accumulator.finalize()

    accumulator.add_pass2(*blocks[0], block_identity=_identity(0))
    result = accumulator.finalize()
    assert result.report.block_count == 1
    with pytest.raises(OutputJacobianSketchPhaseError, match="not finalized"):
        accumulator.finalize()


def test_block_identity_authenticates_content_and_order_without_tensor_hashing() -> None:
    blocks, _, _ = _problem()
    accumulator = TwoPassOutputJacobianSketch(_spec())
    with pytest.raises(TypeError, match="block_identity"):
        accumulator.add_pass1(*blocks[0])  # type: ignore[call-arg]
    for index, (delta, inputs) in enumerate(blocks[:2]):
        accumulator.add_pass1(delta, inputs, block_identity=_identity(index))
    accumulator.finish_pass1()
    for index, (delta, inputs) in enumerate(blocks[:2]):
        wrong_fingerprint = "f" * 64 if index == 1 else None
        accumulator.add_pass2(
            delta,
            inputs,
            block_identity=_identity(index, wrong_fingerprint),
        )
    with pytest.raises(OutputJacobianSketchPhaseError, match="does not authenticate"):
        accumulator.finalize()

    reordered = TwoPassOutputJacobianSketch(_spec())
    for index, (delta, inputs) in enumerate(blocks[:2]):
        reordered.add_pass1(delta, inputs, block_identity=_identity(index))
    reordered.finish_pass1()
    for index, (delta, inputs) in enumerate(blocks[:2]):
        reordered.add_pass2(
            delta,
            inputs,
            block_identity=_identity(1 - index),
        )
    with pytest.raises(OutputJacobianSketchPhaseError, match="does not authenticate"):
        reordered.finalize()


def test_exact_identity_hashes_canonical_tensor_content() -> None:
    blocks, _, _ = _problem()
    accumulator = TwoPassOutputJacobianSketch(_spec())
    delta, inputs = blocks[0]
    identity = accumulator.make_exact_block_identity(
        delta,
        inputs,
        semantic_identity="exact:test:block:0",
    )
    changed = delta.clone()
    changed[0, 0] += 1.0
    changed_identity = accumulator.make_exact_block_identity(
        changed,
        inputs,
        semantic_identity="exact:test:block:0",
    )
    assert identity.delta_sha256 is not None
    assert identity.inputs_sha256 is not None
    assert identity.delta_sha256 != changed_identity.delta_sha256
    assert (
        identity.numeric_fingerprint_sha256
        != changed_identity.numeric_fingerprint_sha256
    )
    with pytest.raises(OutputJacobianSketchShapeError, match="lowercase SHA-256"):
        OutputJacobianBlockIdentity("site", "A" * 64)


def test_one_shot_report_certifies_exact_identity_replay() -> None:
    blocks, left, right = _problem()
    result = sketch_output_jacobian(blocks, _spec(), left, right)
    assert result.report.block_identity_replay_authenticated is True
    assert (
        result.report.pass1_block_identity_chain_sha256
        == result.report.pass2_block_identity_chain_sha256
    )


def test_one_shot_rehashes_exact_content_on_the_second_pass() -> None:
    delta = torch.randn((4, 3), generator=torch.Generator().manual_seed(4))
    inputs = torch.randn((4, 3), generator=torch.Generator().manual_seed(5))

    class MutatingReplay(Sequence[tuple[torch.Tensor, torch.Tensor]]):
        def __init__(self) -> None:
            self.iterations = 0

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            if index != 0:
                raise IndexError(index)
            self.iterations += 1
            current = delta.clone()
            if self.iterations >= 2:
                current[0, 0] += 1.0
            return current, inputs

    with pytest.raises(OutputJacobianSketchPhaseError, match="does not authenticate"):
        sketch_output_jacobian(
            MutatingReplay(),
            OutputJacobianSketchSpec(
                out_features=3,
                in_features=3,
                rank=1,
                sketch_width=3,
                max_bytes=1024 * 1024,
            ),
        )


def test_finalization_rejects_insufficient_effective_rank() -> None:
    delta = torch.zeros((1, 4))
    inputs = torch.zeros((1, 4))
    delta[0, 0] = 1.0
    inputs[0, 0] = 3.0
    accumulator = TwoPassOutputJacobianSketch(
        OutputJacobianSketchSpec(
            out_features=4,
            in_features=4,
            rank=2,
            sketch_width=4,
            max_bytes=1024 * 1024,
        )
    )
    accumulator.add_pass1(delta, inputs, block_identity=_identity(0))
    accumulator.finish_pass1()
    accumulator.add_pass2(delta, inputs, block_identity=_identity(0))
    with pytest.raises(OutputJacobianSketchRankError, match="effective rank"):
        accumulator.finalize()


def test_finalization_rejects_degenerate_requested_rank_boundary() -> None:
    delta = torch.eye(3)
    inputs = torch.diag(torch.tensor([5.0, 5.0, 1.0]))
    accumulator = TwoPassOutputJacobianSketch(
        OutputJacobianSketchSpec(
            out_features=3,
            in_features=3,
            rank=1,
            sketch_width=3,
            max_bytes=1024 * 1024,
        )
    )
    accumulator.add_pass1(delta, inputs, block_identity=_identity(0))
    accumulator.finish_pass1()
    accumulator.add_pass2(delta, inputs, block_identity=_identity(0))
    with pytest.raises(OutputJacobianSketchRankError, match="boundary"):
        accumulator.finalize()


def test_selected_degenerate_cluster_is_coordinate_canonical() -> None:
    delta = torch.eye(3)
    inputs = torch.diag(torch.tensor([5.0, 5.0, 1.0]))
    result = sketch_output_jacobian(
        [(delta, inputs)],
        OutputJacobianSketchSpec(
            out_features=3,
            in_features=3,
            rank=2,
            sketch_width=3,
            seed=71,
            max_bytes=1024 * 1024,
        ),
    )
    expected = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    assert torch.allclose(result.guided_right_rows, expected, atol=2e-6, rtol=0.0)
    assert result.report.normal_effective_rank == 3
    assert result.report.selected_singular_values == pytest.approx((5.0, 5.0))


def test_coordinate_canonicalization_is_invariant_to_subspace_rotation() -> None:
    generator = torch.Generator().manual_seed(99)
    basis, _ = torch.linalg.qr(
        torch.randn((7, 2), dtype=torch.float64, generator=generator),
        mode="reduced",
    )
    angle = torch.tensor(0.731, dtype=torch.float64)
    rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle))),
            torch.stack((torch.sin(angle), torch.cos(angle))),
        )
    )
    first = sketch_module._coordinate_canonical_subspace(basis)  # noqa: SLF001
    second = sketch_module._coordinate_canonical_subspace(  # noqa: SLF001
        basis @ rotation
    )
    assert torch.allclose(first, second, atol=2e-14, rtol=2e-14)


def test_source_never_constructs_identity_or_full_jacobian() -> None:
    source = Path(
        "src/neural_forge/hf_output_jacobian_sketch.py"
    ).read_text(encoding="utf-8")
    assert "torch.eye" not in source
    assert "delta.T @ inputs" not in source
    assert "d.T @ x" not in source
