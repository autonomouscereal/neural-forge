from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import torch


pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "stage2_cpu_toy_validation",
    REPOSITORY_ROOT / "experiments" / "stage2_cpu_toy_validation.py",
)
assert SPEC is not None and SPEC.loader is not None
TOY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOY
SPEC.loader.exec_module(TOY)

from neural_forge.lfm2_carrier_bridge_publication import (  # noqa: E402
    bind_packaged_lfm2_carrier_routes,
)
from neural_forge.stage2_carrier_student import (  # noqa: E402
    STAGE2_FOLD_INITIALIZATION,
    STAGE2_RANDOM_INITIALIZATION,
    Stage2CarrierStudentError,
    assert_route_surface_matches_frozen_runtime,
    build_coalesced_route_layout,
    load_stage2_routed_student,
)
from neural_forge.stage2_kd_objective import (  # noqa: E402
    Stage2DonorVocabulary,
    forward_kl_logit_distillation,
)


@pytest.fixture(scope="module")
def published_child(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return TOY.build_toy_published_child(
        tmp_path_factory.mktemp("stage2-student-fixture")
    )


def _load(published: dict[str, Any], **overrides: Any) -> tuple[Any, Any, Any]:
    arguments: dict[str, Any] = {
        "base": TOY._ToyBaseCheckpointIO(None),
        "mode": STAGE2_FOLD_INITIALIZATION,
        "train_routes": True,
    }
    arguments.update(overrides)
    return load_stage2_routed_student(published["child_directory"], **arguments)


def test_coalesced_layout_reproduces_torch_coalesce_exactly() -> None:
    """The trainer replays the runtime's edge order without re-sorting it."""

    offsets = torch.tensor([0, 2, 3, 6], dtype=torch.int64)
    target_ids = torch.tensor([2, 0, 1, 3, 1, 3], dtype=torch.int64)
    coefficients = torch.tensor(
        [0.5, 0.25, 1.0, 0.125, 2.0, 0.75],
        dtype=torch.float32,
    )
    layout = build_coalesced_route_layout(
        offsets,
        target_ids,
        donor_row_count=3,
        target_row_count=4,
    )
    donor_ids = torch.repeat_interleave(
        torch.arange(3, dtype=torch.int64),
        offsets[1:] - offsets[:-1],
    )
    expected = torch.sparse_coo_tensor(
        torch.stack((donor_ids, target_ids), dim=0),
        coefficients,
        size=(3, 4),
        check_invariants=True,
    ).coalesce()
    assert torch.equal(layout.coalesced_donor_ids, expected.indices()[0])
    assert torch.equal(layout.coalesced_target_ids, expected.indices()[1])
    assert torch.equal(
        layout.coalesced_values(coefficients, dtype=torch.float32),
        expected.values(),
    )


def test_coalesced_layout_sums_duplicate_edges_like_coalesce() -> None:
    offsets = torch.tensor([0, 3], dtype=torch.int64)
    target_ids = torch.tensor([1, 1, 0], dtype=torch.int64)
    coefficients = torch.tensor([0.25, 0.75, 2.0], dtype=torch.float32)
    layout = build_coalesced_route_layout(
        offsets,
        target_ids,
        donor_row_count=1,
        target_row_count=2,
    )
    expected = torch.sparse_coo_tensor(
        torch.stack((torch.zeros(3, dtype=torch.int64), target_ids), dim=0),
        coefficients,
        size=(1, 2),
        check_invariants=True,
    ).coalesce()
    assert layout.duplicate_free is False
    assert torch.equal(
        layout.coalesced_values(coefficients, dtype=torch.float32),
        expected.values(),
    )


def test_coalesced_values_carry_gradient_to_the_route_parameter() -> None:
    offsets = torch.tensor([0, 2, 3], dtype=torch.int64)
    target_ids = torch.tensor([1, 0, 1], dtype=torch.int64)
    coefficients = torch.tensor([0.5, 0.5, 1.0], requires_grad=True)
    layout = build_coalesced_route_layout(
        offsets,
        target_ids,
        donor_row_count=2,
        target_row_count=2,
    )
    layout.coalesced_values(coefficients, dtype=torch.float32).sum().backward()
    assert coefficients.grad is not None
    assert torch.all(coefficients.grad == 1.0)


def test_trainable_route_surface_is_bitwise_identical_to_the_decode_runtime(
    published_child: dict[str, Any],
) -> None:
    """The load-bearing claim: training and review use the same function."""

    student, decode_model, package = _load(published_child)
    runtime = bind_packaged_lfm2_carrier_routes(
        decode_model,
        package.routes,
        expected_target_rows=TOY.TOY_TARGET_ROWS,
        expected_hidden_size=TOY.TOY_HIDDEN_SIZE,
    )
    report = assert_route_surface_matches_frozen_runtime(
        student,
        runtime,
        donor_ids=tuple(range(TOY.TOY_DONOR_ROWS)),
        target_scores=torch.randn(7, TOY.TOY_TARGET_ROWS),
    )
    assert report["input_route_bitwise_identical_to_frozen_runtime"] is True
    assert report["output_route_bitwise_identical_to_frozen_runtime"] is True


def test_a_moved_route_coefficient_breaks_the_equivalence_claim(
    published_child: dict[str, Any],
) -> None:
    """The equivalence check is a real check, not a formality that passes."""

    student, decode_model, package = _load(published_child)
    runtime = bind_packaged_lfm2_carrier_routes(
        decode_model,
        package.routes,
        expected_target_rows=TOY.TOY_TARGET_ROWS,
        expected_hidden_size=TOY.TOY_HIDDEN_SIZE,
    )
    with torch.no_grad():
        student.output_coefficients[-1] += 0.5
    with pytest.raises(Stage2CarrierStudentError, match="differs from the frozen"):
        assert_route_surface_matches_frozen_runtime(
            student,
            runtime,
            donor_ids=(0, 1, 2),
            target_scores=torch.randn(3, TOY.TOY_TARGET_ROWS),
        )


def test_protected_anchor_coefficients_are_located_and_pinned(
    published_child: dict[str, Any],
) -> None:
    """The 4,096 injective anchors are structure; stage 2 must not train them."""

    student, _model, _package = _load(published_child)
    mask = student.route_training_mask()
    assert mask["protected_anchor_coefficient_elements"] == 2 * TOY.TOY_TARGET_ROWS
    assert mask["protected_anchors_intact"] is True
    with torch.no_grad():
        student.input_coefficients[student.input_protected_edges[0]] = 0.5
    assert student.protected_anchors_are_intact() is False
    student.restore_protected_anchor_coefficients()
    assert student.protected_anchors_are_intact() is True


def test_anchor_gradients_are_masked_out(
    published_child: dict[str, Any],
) -> None:
    student, _model, _package = _load(published_child)
    vocabulary = Stage2DonorVocabulary(
        defined_row_count=TOY.TOY_DONOR_ROWS,
        physical_row_count=TOY.TOY_DONOR_ROWS,
    )
    donor_ids = torch.arange(TOY.TOY_DONOR_ROWS, dtype=torch.int64).reshape(2, -1)
    logits = student(donor_ids)
    forward_kl_logit_distillation(
        logits,
        torch.zeros_like(logits),
        vocabulary=vocabulary,
    ).backward()
    for parameter, positions in (
        (student.input_coefficients, student.input_protected_edges),
        (student.output_coefficients, student.output_protected_edges),
    ):
        assert parameter.grad is not None
        assert torch.all(parameter.grad.index_select(0, positions) == 0.0)
        assert torch.any(parameter.grad != 0.0)


def test_frozen_routes_receive_no_gradient_at_all(
    published_child: dict[str, Any],
) -> None:
    student, _model, _package = _load(published_child, train_routes=False)
    assert student.input_coefficients.requires_grad is False
    assert student.output_coefficients.requires_grad is False
    assert student.route_training_mask()["route_coefficients_are_trainable"] is False


def test_fold_and_random_arms_load_through_one_path_and_differ_only_in_numbers(
    published_child: dict[str, Any],
) -> None:
    """Design section 1: byte-identical config, different initialization."""

    fold, _fold_model, _fold_package = _load(published_child)
    control, _control_model, _control_package = _load(
        published_child,
        mode=STAGE2_RANDOM_INITIALIZATION,
        seed=99,
    )
    assert fold.initialization.config_sha256 == control.initialization.config_sha256
    assert (
        fold.initialization.child_package_sha256
        == control.initialization.child_package_sha256
    )
    assert fold.counted_parameter_elements() == control.counted_parameter_elements()
    assert fold.route_layouts() == control.route_layouts()
    assert (
        fold.initialization.init_source_sha256
        != control.initialization.init_source_sha256
    )
    assert control.initialization.seed == 99
    assert control.initialization.scheme is not None
    assert control.protected_anchors_are_intact()
    assert not torch.equal(
        fold.child_tables()[0].detach(),
        control.child_tables()[0].detach(),
    )
    assert not torch.equal(
        fold.output_coefficients.detach(),
        control.output_coefficients.detach(),
    )


def test_random_initialization_is_reproducible_from_its_recorded_seed(
    published_child: dict[str, Any],
) -> None:
    first, _model, _package = _load(
        published_child,
        mode=STAGE2_RANDOM_INITIALIZATION,
        seed=1234,
    )
    second, _model2, _package2 = _load(
        published_child,
        mode=STAGE2_RANDOM_INITIALIZATION,
        seed=1234,
    )
    assert torch.equal(
        first.child_tables()[0].detach(),
        second.child_tables()[0].detach(),
    )
    assert torch.equal(
        first.input_coefficients.detach(),
        second.input_coefficients.detach(),
    )
    assert (
        first.initialization.init_source_sha256
        == second.initialization.init_source_sha256
    )


def test_a_random_control_needs_an_explicit_seed(
    published_child: dict[str, Any],
) -> None:
    with pytest.raises(Stage2CarrierStudentError, match="explicit recorded seed"):
        _load(published_child, mode=STAGE2_RANDOM_INITIALIZATION)


def test_forward_produces_donor_logits_and_trains_body_and_routes(
    published_child: dict[str, Any],
) -> None:
    student, _model, _package = _load(published_child)
    donor_ids = torch.randint(0, TOY.TOY_DONOR_ROWS, (2, 5), dtype=torch.int64)
    logits = student(donor_ids)
    assert tuple(logits.shape) == (2, 5, TOY.TOY_DONOR_ROWS)
    logits.sum().backward()
    assert student.input_coefficients.grad is not None
    assert student.output_coefficients.grad is not None
    body_gradients = [
        parameter.grad
        for name, parameter in student.child.named_parameters()
        if name.startswith("model.")
    ]
    assert body_gradients and all(value is not None for value in body_gradients)


def test_a_declared_vocabulary_that_disagrees_with_the_routes_is_refused(
    published_child: dict[str, Any],
) -> None:
    with pytest.raises(Stage2CarrierStudentError, match="different donor row count"):
        _load(
            published_child,
            donor_vocabulary=Stage2DonorVocabulary(
                defined_row_count=TOY.TOY_DONOR_ROWS + 1,
                physical_row_count=TOY.TOY_DONOR_ROWS + 4,
            ),
        )
