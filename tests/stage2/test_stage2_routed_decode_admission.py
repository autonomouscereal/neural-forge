from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest
import torch


pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "stage2_cpu_toy_validation_admission",
    REPOSITORY_ROOT / "experiments" / "stage2_cpu_toy_validation.py",
)
assert SPEC is not None and SPEC.loader is not None
TOY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOY
SPEC.loader.exec_module(TOY)

from neural_forge.lfm2_carrier_bridge_publication import (  # noqa: E402
    Lfm2CarrierBridgePublicationError,
    load_packaged_lfm2_carrier_routes,
)
from neural_forge.lfm2_carrier_child_package import (  # noqa: E402
    ADMITTED_LFM2_STAGE2_CHECKPOINT_LABELS,
    LFM2_CARRIER_EXACT_2X_COUNTED_ABI,
    LFM2_CARRIER_EXACT_4X_COUNTED_ABI,
    LFM2_STAGE2_S2A_FOLD_LABEL,
    LFM2_STAGE2_S2B_FOLD_LABEL,
    LFM2_STAGE2_S2C_CONTROL_LABEL,
    LFM2_STAGE2_S2D_CONTROL_LABEL,
    Lfm2CarrierChildPackageError,
    Lfm2CarrierChildPackageRegistration,
    Lfm2Stage2CheckpointLabel,
    admitted_lfm2_stage2_checkpoint_label,
    is_admitted_lfm2_stage2_checkpoint_label,
)
from neural_forge.stage2_carrier_student import (  # noqa: E402
    STAGE2_FOLD_INITIALIZATION,
    STAGE2_INITIALIZATION_MODES,
    STAGE2_RANDOM_INITIALIZATION,
    load_stage2_routed_student,
)
from neural_forge.stage2_checkpoint_layout import (  # noqa: E402
    STAGE1_TRAINED_ROUTE_REFUSAL,
    round_trip_stage2_rung_checkpoint,
    stage2_route_provenance,
    write_stage2_rung_checkpoint,
)
from neural_forge.stage2_routed_decode_admission import (  # noqa: E402
    STAGE2_ROUTED_DECODE_ADMISSION_SCHEMA,
    Stage2CheckpointIdentity,
    Stage2RoutedDecodeAdmissionError,
    admit_stage2_rung_checkpoint,
    authenticate_stage2_packaged_carrier_routes,
)


ARM_LABELS = {
    "S2-A": LFM2_STAGE2_S2A_FOLD_LABEL,
    "S2-B": LFM2_STAGE2_S2B_FOLD_LABEL,
    "S2-C": LFM2_STAGE2_S2C_CONTROL_LABEL,
    "S2-D": LFM2_STAGE2_S2D_CONTROL_LABEL,
}
ROUTE_RECEIPT = Path("semantic-carrier-routes") / "carrier-routes-receipt.json"
ROUTE_TENSORS = Path("semantic-carrier-routes") / "carrier-routes.safetensors"
_UNSET = object()


@pytest.fixture(scope="module")
def published_child(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return TOY.build_toy_published_child(
        tmp_path_factory.mktemp("stage2-admission-fixture")
    )


def _write_arm_checkpoint(
    published: dict[str, Any],
    destination: Path,
    *,
    label: Lfm2Stage2CheckpointLabel,
    routes_were_trained: bool = True,
    optimizer_steps: int = 7,
    backward_calls: int = 28,
    arm_id: str | None = None,
    initialization_mode: str | None = None,
    init_artifact_sha256: Any = _UNSET,
) -> dict[str, Any]:
    """One rung checkpoint written the way the arm the label names writes it."""

    mode = label.initialization_mode
    student, _decode_model, package = load_stage2_routed_student(
        published["child_directory"],
        base=TOY._ToyBaseCheckpointIO(None),
        mode=mode,
        train_routes=routes_were_trained,
        seed=None if mode == STAGE2_FOLD_INITIALIZATION else 11,
    )
    provenance = stage2_route_provenance(
        package,
        coefficients_from_source_artifact=(mode == STAGE2_FOLD_INITIALIZATION),
        initial_input_coefficients=student.input_coefficients.detach(),
        initial_output_coefficients=student.output_coefficients.detach(),
    )
    return write_stage2_rung_checkpoint(
        destination,
        student=student,
        base=TOY._ToyBaseCheckpointIO(published["target_tokenizer_directory"]),
        provenance=provenance,
        structural_route_tensors=package.routes.tensors,
        initialization_child_directory=published["child_directory"],
        semantic_source_tokenizer_sha256=published[
            "semantic_source_tokenizer_sha256"
        ],
        lexicon_package_sha256=published["lexicon_package_sha256"],
        arm_id=label.arm_id if arm_id is None else arm_id,
        rung_tokens=10_000,
        tokens_consumed=10_240,
        optimizer_steps=optimizer_steps,
        backward_calls=backward_calls,
        init_source_sha256=student.initialization.init_source_sha256,
        routes_were_trained=routes_were_trained,
        initialization_mode=(
            label.initialization_mode
            if initialization_mode is None
            else initialization_mode
        ),
        init_artifact_sha256=(
            label.init_artifact_sha256
            if init_artifact_sha256 is _UNSET
            else init_artifact_sha256
        ),
    )


def _identity(manifest: dict[str, Any]) -> Stage2CheckpointIdentity:
    return Stage2CheckpointIdentity(
        manifest_sha256=str(manifest["manifest_sha256"]),
        child_package_sha256=str(manifest["child_package_sha256"]),
        init_source_sha256=str(manifest["init_source_sha256"]),
    )


# ---------------------------------------------------------------------------
# The sealed label list itself
# ---------------------------------------------------------------------------


def test_the_label_list_is_sealed_coherent_and_identity_matched() -> None:
    assert len(ADMITTED_LFM2_STAGE2_CHECKPOINT_LABELS) == 4
    assert {label.arm_id for label in ADMITTED_LFM2_STAGE2_CHECKPOINT_LABELS} == {
        "S2-A",
        "S2-B",
        "S2-C",
        "S2-D",
    }
    for label in ADMITTED_LFM2_STAGE2_CHECKPOINT_LABELS:
        assert label.initialization_mode in STAGE2_INITIALIZATION_MODES
        assert label.route_coefficients_trained is True
        if label.fold_weights_used:
            assert isinstance(label.init_artifact_sha256, str)
        else:
            assert label.init_artifact_sha256 is None
        assert is_admitted_lfm2_stage2_checkpoint_label(label) is True
    # An equal-valued label assembled at a call site is *not* the sealed one.
    forged = Lfm2Stage2CheckpointLabel(
        label_id=LFM2_STAGE2_S2B_FOLD_LABEL.label_id,
        arm_id=LFM2_STAGE2_S2B_FOLD_LABEL.arm_id,
        initialization_mode=LFM2_STAGE2_S2B_FOLD_LABEL.initialization_mode,
        init_artifact_sha256=LFM2_STAGE2_S2B_FOLD_LABEL.init_artifact_sha256,
        route_coefficients_trained=True,
        counted_abi_profile_id=LFM2_STAGE2_S2B_FOLD_LABEL.counted_abi_profile_id,
    )
    assert forged == LFM2_STAGE2_S2B_FOLD_LABEL
    assert is_admitted_lfm2_stage2_checkpoint_label(forged) is False


def test_the_label_lookup_is_coherent_not_per_field() -> None:
    assert (
        admitted_lfm2_stage2_checkpoint_label(
            arm_id="S2-B",
            initialization_mode=STAGE2_FOLD_INITIALIZATION,
            init_artifact_sha256=LFM2_STAGE2_S2B_FOLD_LABEL.init_artifact_sha256,
            route_coefficients_trained=True,
        )
        is LFM2_STAGE2_S2B_FOLD_LABEL
    )
    refused = (
        # S2-B's arm with S2-A's initialization artifact
        {
            "arm_id": "S2-B",
            "initialization_mode": STAGE2_FOLD_INITIALIZATION,
            "init_artifact_sha256": (
                LFM2_STAGE2_S2A_FOLD_LABEL.init_artifact_sha256
            ),
            "route_coefficients_trained": True,
        },
        # a fold arm claiming to be a control
        {
            "arm_id": "S2-B",
            "initialization_mode": STAGE2_RANDOM_INITIALIZATION,
            "init_artifact_sha256": None,
            "route_coefficients_trained": True,
        },
        # a control claiming a source artifact it does not have
        {
            "arm_id": "S2-D",
            "initialization_mode": STAGE2_RANDOM_INITIALIZATION,
            "init_artifact_sha256": (
                LFM2_STAGE2_S2B_FOLD_LABEL.init_artifact_sha256
            ),
            "route_coefficients_trained": True,
        },
        # frozen routes: nothing to admit, so nothing admits it
        {
            "arm_id": "S2-D",
            "initialization_mode": STAGE2_RANDOM_INITIALIZATION,
            "init_artifact_sha256": None,
            "route_coefficients_trained": False,
        },
        # an arm that is not in the measurement at all
        {
            "arm_id": "S2-TOY",
            "initialization_mode": STAGE2_FOLD_INITIALIZATION,
            "init_artifact_sha256": (
                LFM2_STAGE2_S2B_FOLD_LABEL.init_artifact_sha256
            ),
            "route_coefficients_trained": True,
        },
    )
    for request in refused:
        assert admitted_lfm2_stage2_checkpoint_label(**request) is None


def test_the_label_modes_equal_the_stage2_students_own_constants() -> None:
    from neural_forge.lfm2_carrier_child_package import (
        STAGE2_CHECKPOINT_LABEL_INITIALIZATION_MODES,
    )

    assert set(STAGE2_CHECKPOINT_LABEL_INITIALIZATION_MODES) == set(
        STAGE2_INITIALIZATION_MODES
    )


# ---------------------------------------------------------------------------
# The parallel counted-plan lane
# ---------------------------------------------------------------------------


class _Registration(Lfm2CarrierChildPackageRegistration):
    """A registration whose counted plan is stubbed; only the gate is tested."""

    def __init__(self) -> None:  # noqa: D107 - deliberately not the real ctor
        self.calls: list[Any] = []

    def admitted_counted_plan(self, profile: Any) -> Any:
        self.calls.append(profile)
        return f"plan-for-{profile.profile_id}"


def test_a_stage2_label_cannot_admit_through_the_stage1_entrypoint() -> None:
    registration = _Registration()
    # The stage-1 entrypoint takes profiles.  A label is not a profile, and the
    # membership test is by identity, so it cannot be smuggled in.
    with pytest.raises(
        Lfm2CarrierChildPackageError,
        match="sealed to the admitted carrier ABI",
    ):
        Lfm2CarrierChildPackageRegistration.admitted_counted_plan(
            registration,
            LFM2_STAGE2_S2B_FOLD_LABEL,
        )
    # And the stage-2 entrypoint takes both, in that order, by identity.
    with pytest.raises(
        Lfm2CarrierChildPackageError,
        match="sealed stage-2 checkpoint label",
    ):
        registration.admitted_stage2_checkpoint_plan(
            LFM2_CARRIER_EXACT_2X_COUNTED_ABI,
            LFM2_CARRIER_EXACT_2X_COUNTED_ABI,
        )
    assert registration.calls == []


def test_the_stage2_counted_plan_requires_the_label_to_name_that_abi() -> None:
    registration = _Registration()
    assert (
        registration.admitted_stage2_checkpoint_plan(
            LFM2_CARRIER_EXACT_2X_COUNTED_ABI,
            LFM2_STAGE2_S2B_FOLD_LABEL,
        )
        == "plan-for-exact-2x-h816"
    )
    # S2-B is the 2x rung; it does not become a 4x run by asking.
    with pytest.raises(Lfm2CarrierChildPackageError, match="does not run at the"):
        registration.admitted_stage2_checkpoint_plan(
            LFM2_CARRIER_EXACT_4X_COUNTED_ABI,
            LFM2_STAGE2_S2B_FOLD_LABEL,
        )
    # S2-A runs at the A11 publication topology, which is not a sealed
    # production ABI tuple at all, so it reaches no counted plan.
    with pytest.raises(Lfm2CarrierChildPackageError, match="does not run at the"):
        registration.admitted_stage2_checkpoint_plan(
            LFM2_CARRIER_EXACT_2X_COUNTED_ABI,
            LFM2_STAGE2_S2A_FOLD_LABEL,
        )
    assert registration.calls == [LFM2_CARRIER_EXACT_2X_COUNTED_ABI]


# ---------------------------------------------------------------------------
# The decode lane
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm_id", sorted(ARM_LABELS))
def test_labelled_admission_accepts_each_of_the_four_arms(
    published_child: dict[str, Any],
    tmp_path: Path,
    arm_id: str,
) -> None:
    """The four design arms, each admitted only under its own sealed label."""

    label = ARM_LABELS[arm_id]
    manifest = _write_arm_checkpoint(
        published_child,
        tmp_path / f"rung-{arm_id}",
        label=label,
    )
    record = admit_stage2_rung_checkpoint(
        tmp_path / f"rung-{arm_id}",
        label=label,
        identity=_identity(manifest),
        base=TOY._ToyBaseCheckpointIO(None),
    )
    assert record["schema"] == STAGE2_ROUTED_DECODE_ADMISSION_SCHEMA
    assert record["status"] == "admitted"
    assert record["arm_id"] == arm_id
    assert record["admitted_by_label"]["label_id"] == label.label_id
    assert record["route_operation_audit"]["optimizer_steps"] == 7
    assert record["route_operation_audit"]["backward_calls"] == 28
    assert record["stage1_guard_untouched"] is True
    assert (
        record["stage1_refusal_this_checkpoint_still_meets"]
        == STAGE1_TRAINED_ROUTE_REFUSAL
    )
    assert record["routed_decode_bound"] is True
    assert record["semantic_step_generate_installed"] is True
    assert callable(record["model"].semantic_step_generate)
    assert len(record["admission_sha256"]) == 64


def test_the_same_checkpoint_is_still_refused_at_the_stage1_message(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    """The parallel-lane proof: admission opens nothing the default lane opens."""

    label = LFM2_STAGE2_S2B_FOLD_LABEL
    manifest = _write_arm_checkpoint(
        published_child,
        tmp_path / "rung-parallel",
        label=label,
    )
    admitted = admit_stage2_rung_checkpoint(
        tmp_path / "rung-parallel",
        label=label,
        identity=_identity(manifest),
        base=TOY._ToyBaseCheckpointIO(None),
    )
    assert admitted["routed_decode_bound"] is True
    # Same directory, same bytes, no label: the stage-1 guard is untouched.
    unlabelled = round_trip_stage2_rung_checkpoint(
        tmp_path / "rung-parallel",
        base=TOY._ToyBaseCheckpointIO(None),
    )
    assert unlabelled["loaded_through_review_loader"] is False
    assert unlabelled["refused_by_stage1_guard"] is True
    assert unlabelled["refusal_message"] == STAGE1_TRAINED_ROUTE_REFUSAL
    with pytest.raises(
        Lfm2CarrierBridgePublicationError,
        match="zero-operation fold audit",
    ) as refusal:
        load_packaged_lfm2_carrier_routes(
            tmp_path / "rung-parallel" / "child",
            expected_source_artifact_sha256=(
                published_child["source_artifact_sha256"]
            ),
        )
    assert str(refusal.value) == STAGE1_TRAINED_ROUTE_REFUSAL


def test_a_label_from_the_wrong_arm_is_refused(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    manifest = _write_arm_checkpoint(
        published_child,
        tmp_path / "rung-wrong-arm",
        label=LFM2_STAGE2_S2B_FOLD_LABEL,
    )
    with pytest.raises(
        Stage2RoutedDecodeAdmissionError,
        match="does not describe this checkpoint",
    ):
        admit_stage2_rung_checkpoint(
            tmp_path / "rung-wrong-arm",
            label=LFM2_STAGE2_S2A_FOLD_LABEL,
            identity=_identity(manifest),
            base=TOY._ToyBaseCheckpointIO(None),
        )


def test_a_fold_checkpoint_cannot_be_admitted_as_its_own_control(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    """The finding-3 case: only the label separates arm from control."""

    manifest = _write_arm_checkpoint(
        published_child,
        tmp_path / "rung-arm-as-control",
        label=LFM2_STAGE2_S2B_FOLD_LABEL,
        arm_id="S2-D",
    )
    with pytest.raises(
        Stage2RoutedDecodeAdmissionError,
        match="does not describe this checkpoint",
    ):
        admit_stage2_rung_checkpoint(
            tmp_path / "rung-arm-as-control",
            label=LFM2_STAGE2_S2D_CONTROL_LABEL,
            identity=_identity(manifest),
            base=TOY._ToyBaseCheckpointIO(None),
        )


def test_a_checkpoint_with_no_initialization_mode_admits_nowhere(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    manifest = _write_arm_checkpoint(
        published_child,
        tmp_path / "rung-unlabelled-init",
        label=LFM2_STAGE2_S2D_CONTROL_LABEL,
        initialization_mode="",
    )
    # ``initialization_mode=""`` is recorded truthfully and matches nothing.
    with pytest.raises(
        Stage2RoutedDecodeAdmissionError,
        match="does not describe this checkpoint",
    ):
        admit_stage2_rung_checkpoint(
            tmp_path / "rung-unlabelled-init",
            label=LFM2_STAGE2_S2D_CONTROL_LABEL,
            identity=_identity(manifest),
            base=TOY._ToyBaseCheckpointIO(None),
        )


def test_a_control_that_claims_a_fold_artifact_is_refused(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    manifest = _write_arm_checkpoint(
        published_child,
        tmp_path / "rung-fake-provenance",
        label=LFM2_STAGE2_S2D_CONTROL_LABEL,
        init_artifact_sha256=LFM2_STAGE2_S2B_FOLD_LABEL.init_artifact_sha256,
    )
    with pytest.raises(
        Stage2RoutedDecodeAdmissionError,
        match="does not describe this checkpoint",
    ):
        admit_stage2_rung_checkpoint(
            tmp_path / "rung-fake-provenance",
            label=LFM2_STAGE2_S2D_CONTROL_LABEL,
            identity=_identity(manifest),
            base=TOY._ToyBaseCheckpointIO(None),
        )


def test_a_frozen_route_checkpoint_is_not_admitted_by_a_trained_label(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Every sealed label admits trained routes; a frozen rung needs no lane."""

    manifest = _write_arm_checkpoint(
        published_child,
        tmp_path / "rung-frozen",
        label=LFM2_STAGE2_S2D_CONTROL_LABEL,
        routes_were_trained=False,
    )
    with pytest.raises(
        Stage2RoutedDecodeAdmissionError,
        match="does not describe this checkpoint",
    ):
        admit_stage2_rung_checkpoint(
            tmp_path / "rung-frozen",
            label=LFM2_STAGE2_S2D_CONTROL_LABEL,
            identity=_identity(manifest),
            base=TOY._ToyBaseCheckpointIO(None),
        )
    # It does not need one: the untouched stage-1 loader already opens it.
    result = round_trip_stage2_rung_checkpoint(
        tmp_path / "rung-frozen",
        base=TOY._ToyBaseCheckpointIO(None),
    )
    assert result["loaded_through_review_loader"] is True


def test_a_forged_label_object_is_refused(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    manifest = _write_arm_checkpoint(
        published_child,
        tmp_path / "rung-forged-label",
        label=LFM2_STAGE2_S2D_CONTROL_LABEL,
    )
    forged = Lfm2Stage2CheckpointLabel(
        label_id=LFM2_STAGE2_S2D_CONTROL_LABEL.label_id,
        arm_id="S2-D",
        initialization_mode=STAGE2_RANDOM_INITIALIZATION,
        init_artifact_sha256=None,
        route_coefficients_trained=True,
        counted_abi_profile_id="exact-2x-h816",
    )
    with pytest.raises(
        Stage2RoutedDecodeAdmissionError,
        match="sealed stage-2 checkpoint label",
    ):
        admit_stage2_rung_checkpoint(
            tmp_path / "rung-forged-label",
            label=forged,
            identity=_identity(manifest),
            base=TOY._ToyBaseCheckpointIO(None),
        )


@pytest.mark.parametrize(
    "field",
    ["manifest_sha256", "child_package_sha256", "init_source_sha256"],
)
def test_a_declared_identity_that_is_not_this_checkpoint_is_refused(
    published_child: dict[str, Any],
    tmp_path: Path,
    field: str,
) -> None:
    manifest = _write_arm_checkpoint(
        published_child,
        tmp_path / f"rung-identity-{field}",
        label=LFM2_STAGE2_S2C_CONTROL_LABEL,
    )
    values = _identity(manifest).as_dict()
    values[field] = "0" * 64
    expected = {
        "manifest_sha256": "belongs to a different checkpoint",
        "child_package_sha256": "not this checkpoint",
        "init_source_sha256": "not this checkpoint",
    }[field]
    with pytest.raises(Exception, match=expected):  # noqa: PT011 - two lanes
        admit_stage2_rung_checkpoint(
            tmp_path / f"rung-identity-{field}",
            label=LFM2_STAGE2_S2C_CONTROL_LABEL,
            identity=Stage2CheckpointIdentity(**values),
            base=TOY._ToyBaseCheckpointIO(None),
        )


def test_admission_without_a_base_refuses_rather_than_half_binding(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    manifest = _write_arm_checkpoint(
        published_child,
        tmp_path / "rung-no-base",
        label=LFM2_STAGE2_S2C_CONTROL_LABEL,
    )
    with pytest.raises(
        Stage2RoutedDecodeAdmissionError,
        match="needs a base checkpoint IO",
    ):
        admit_stage2_rung_checkpoint(
            tmp_path / "rung-no-base",
            label=LFM2_STAGE2_S2C_CONTROL_LABEL,
            identity=_identity(manifest),
        )
    record = admit_stage2_rung_checkpoint(
        tmp_path / "rung-no-base",
        label=LFM2_STAGE2_S2C_CONTROL_LABEL,
        identity=_identity(manifest),
        bind_decode_runtime=False,
    )
    assert record["routed_decode_bound"] is False
    assert "model" not in record


def test_the_stage2_lane_refuses_an_ordinary_stage1_published_child(
    published_child: dict[str, Any],
) -> None:
    """The lane is not a relaxed loader; it cannot open a stage-1 child."""

    with pytest.raises(
        Stage2RoutedDecodeAdmissionError,
        match="not a stage-2 checkpoint",
    ):
        authenticate_stage2_packaged_carrier_routes(
            published_child["child_directory"],
            expected_source_artifact_sha256=(
                published_child["source_artifact_sha256"]
            ),
            label=LFM2_STAGE2_S2B_FOLD_LABEL,
            optimizer_steps=7,
            backward_calls=28,
        )


def test_a_pruning_or_quantization_credit_claim_is_never_admitted(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Stage 2 relaxes the training counters only; the credits stay at zero."""

    _manifest = _write_arm_checkpoint(
        published_child,
        tmp_path / "rung-credit",
        label=LFM2_STAGE2_S2B_FOLD_LABEL,
    )
    receipt_path = tmp_path / "rung-credit" / "child" / ROUTE_RECEIPT
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "package_sha256"}
    body["operation_audit"]["parameter_pruning_calls"] = 1
    from neural_forge.dynamic_circuit_ir import canonical_sha256

    resealed = {**body, "package_sha256": canonical_sha256(body)}
    receipt_path.write_text(json.dumps(resealed, sort_keys=True), encoding="ascii")
    with pytest.raises(
        Stage2RoutedDecodeAdmissionError,
        match="zero pruning, quantization",
    ):
        authenticate_stage2_packaged_carrier_routes(
            tmp_path / "rung-credit" / "child",
            expected_source_artifact_sha256=(
                published_child["source_artifact_sha256"]
            ),
            label=LFM2_STAGE2_S2B_FOLD_LABEL,
            optimizer_steps=7,
            backward_calls=28,
        )


def test_an_audit_that_disagrees_with_the_checkpoint_record_is_refused(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    _manifest = _write_arm_checkpoint(
        published_child,
        tmp_path / "rung-audit-drift",
        label=LFM2_STAGE2_S2B_FOLD_LABEL,
        optimizer_steps=7,
        backward_calls=28,
    )
    with pytest.raises(
        Stage2RoutedDecodeAdmissionError,
        match="own training record",
    ):
        authenticate_stage2_packaged_carrier_routes(
            tmp_path / "rung-audit-drift" / "child",
            expected_source_artifact_sha256=(
                published_child["source_artifact_sha256"]
            ),
            label=LFM2_STAGE2_S2B_FOLD_LABEL,
            optimizer_steps=8,
            backward_calls=28,
        )


# ---------------------------------------------------------------------------
# The anti-drift pin: the stage-2 twin against the stage-1 verifier
# ---------------------------------------------------------------------------


def _sibling_checkpoints(
    published: dict[str, Any],
    tmp_path: Path,
    name: str,
) -> tuple[Path, Path]:
    """One student, written twice: routes declared trained, and frozen.

    The route tensors are bit-identical (nothing moved them), so the two trees
    differ only in the operation audit and the stage-2 label block.  That makes
    the frozen sibling a surface the *stage-1* loader will run on, and the
    trained sibling a surface only the stage-2 twin will run on, with every
    other byte the same.
    """

    student, _decode_model, package = load_stage2_routed_student(
        published["child_directory"],
        base=TOY._ToyBaseCheckpointIO(None),
        mode=STAGE2_FOLD_INITIALIZATION,
        train_routes=True,
    )
    provenance = stage2_route_provenance(
        package,
        coefficients_from_source_artifact=True,
        initial_input_coefficients=student.input_coefficients.detach(),
        initial_output_coefficients=student.output_coefficients.detach(),
    )
    paths: list[Path] = []
    for trained in (True, False):
        destination = tmp_path / f"{name}-{'trained' if trained else 'frozen'}"
        write_stage2_rung_checkpoint(
            destination,
            student=student,
            base=TOY._ToyBaseCheckpointIO(published["target_tokenizer_directory"]),
            provenance=provenance,
            structural_route_tensors=package.routes.tensors,
            initialization_child_directory=published["child_directory"],
            semantic_source_tokenizer_sha256=published[
                "semantic_source_tokenizer_sha256"
            ],
            lexicon_package_sha256=published["lexicon_package_sha256"],
            arm_id="S2-B",
            rung_tokens=10_000,
            tokens_consumed=10_240,
            optimizer_steps=7,
            backward_calls=28,
            init_source_sha256=student.initialization.init_source_sha256,
            routes_were_trained=trained,
            initialization_mode=STAGE2_FOLD_INITIALIZATION,
            init_artifact_sha256=LFM2_STAGE2_S2B_FOLD_LABEL.init_artifact_sha256,
        )
        paths.append(destination / "child")
    return paths[0], paths[1]


def _stage1(child: Path, published: dict[str, Any]) -> Any:
    return load_packaged_lfm2_carrier_routes(
        child,
        expected_source_artifact_sha256=published["source_artifact_sha256"],
    )


def _stage2(child: Path, published: dict[str, Any]) -> Any:
    return authenticate_stage2_packaged_carrier_routes(
        child,
        expected_source_artifact_sha256=published["source_artifact_sha256"],
        label=LFM2_STAGE2_S2B_FOLD_LABEL,
        optimizer_steps=7,
        backward_calls=28,
    )


def test_the_stage2_twin_matches_the_stage1_loader_where_stage1_will_run(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Accepted case: identical tensors, allocations and receipt policy."""

    trained, frozen = _sibling_checkpoints(published_child, tmp_path, "equiv")
    stage1 = _stage1(frozen, published_child)
    stage2 = _stage2(trained, published_child)
    assert set(stage1.tensors) == set(stage2.tensors)
    for name in stage1.tensors:
        assert torch.equal(stage1.tensors[name], stage2.tensors[name])
    assert stage1.allocations == stage2.allocations
    ignored = {"operation_audit", "package_sha256", "stage2_route_provenance"}
    assert {
        key: value for key, value in stage1.receipt.items() if key not in ignored
    } == {key: value for key, value in stage2.receipt.items() if key not in ignored}


CORRUPTIONS: dict[str, Callable[[Path], None]] = {
    "tensor-file-bytes": lambda child: (child / ROUTE_TENSORS).write_bytes(
        (child / ROUTE_TENSORS).read_bytes() + b"x"
    ),
    "receipt-seal": lambda child: (child / ROUTE_RECEIPT).write_text(
        json.dumps(
            {
                **json.loads((child / ROUTE_RECEIPT).read_text(encoding="utf-8")),
                "span_evidence_sha256": "0" * 64,
            },
            sort_keys=True,
        ),
        encoding="ascii",
    ),
    "extra-file": lambda child: (
        child / "semantic-carrier-routes" / "stowaway.json"
    ).write_text("{}", encoding="ascii"),
    "missing-tensor-file": lambda child: (child / ROUTE_TENSORS).unlink(),
}


@pytest.mark.parametrize("corruption", sorted(CORRUPTIONS))
def test_the_stage2_twin_refuses_exactly_what_the_stage1_loader_refuses(
    published_child: dict[str, Any],
    tmp_path: Path,
    corruption: str,
) -> None:
    """Refused cases: same corruption, same refusal string, both lanes."""

    trained, frozen = _sibling_checkpoints(published_child, tmp_path, corruption)
    CORRUPTIONS[corruption](trained)
    CORRUPTIONS[corruption](frozen)
    with pytest.raises(Exception) as stage1_refusal:  # noqa: PT011 - see below
        _stage1(frozen, published_child)
    with pytest.raises(Exception) as stage2_refusal:  # noqa: PT011
        _stage2(trained, published_child)
    # Not just "both refused": the same refusal, type and message.  That is the
    # anti-drift pin -- the shared steps really are the stage-1 verifier's.
    assert type(stage2_refusal.value) is type(stage1_refusal.value)
    assert str(stage2_refusal.value) == str(stage1_refusal.value)
