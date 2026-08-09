from __future__ import annotations

import importlib.util
import json
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
    "stage2_cpu_toy_validation_checkpoints",
    REPOSITORY_ROOT / "experiments" / "stage2_cpu_toy_validation.py",
)
assert SPEC is not None and SPEC.loader is not None
TOY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOY
SPEC.loader.exec_module(TOY)

from neural_forge.lfm2_carrier_child_package import (  # noqa: E402
    CARRIER_CHILD_PACKAGE_RECEIPT,
)
from neural_forge.stage2_carrier_student import (  # noqa: E402
    STAGE2_FOLD_INITIALIZATION,
    STAGE2_RANDOM_INITIALIZATION,
    load_stage2_routed_student,
)
from neural_forge.stage2_checkpoint_layout import (  # noqa: E402
    STAGE1_TRAINED_ROUTE_REFUSAL,
    STAGE2_CHECKPOINT_MANIFEST_FILENAME,
    STAGE2_FROZEN_ROUTE_PROVENANCE,
    STAGE2_ROUTED_DECODE_ADMISSION_HANDOFF,
    STAGE2_TRAINED_ROUTE_PROVENANCE,
    Stage2CheckpointLayoutError,
    load_stage2_rung_checkpoint_manifest,
    round_trip_stage2_rung_checkpoint,
    stage2_route_provenance,
    write_stage2_rung_checkpoint,
)


@pytest.fixture(scope="module")
def published_child(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return TOY.build_toy_published_child(
        tmp_path_factory.mktemp("stage2-checkpoint-fixture")
    )


def _write_checkpoint(
    published: dict[str, Any],
    destination: Path,
    *,
    routes_were_trained: bool,
    mode: str = STAGE2_FOLD_INITIALIZATION,
    seed: int | None = None,
    mutate: Any = None,
) -> tuple[dict[str, Any], Any]:
    student, _decode_model, package = load_stage2_routed_student(
        published["child_directory"],
        base=TOY._ToyBaseCheckpointIO(None),
        mode=mode,
        train_routes=routes_were_trained,
        seed=seed,
    )
    provenance = stage2_route_provenance(
        package,
        coefficients_from_source_artifact=(mode == STAGE2_FOLD_INITIALIZATION),
        initial_input_coefficients=student.input_coefficients.detach(),
        initial_output_coefficients=student.output_coefficients.detach(),
    )
    if mutate is not None:
        mutate(student)
    manifest = write_stage2_rung_checkpoint(
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
        arm_id="S2-TOY",
        rung_tokens=10_000,
        tokens_consumed=10_240,
        optimizer_steps=5,
        backward_calls=20,
        init_source_sha256=student.initialization.init_source_sha256,
        routes_were_trained=routes_were_trained,
    )
    return manifest, student


def test_a_frozen_route_checkpoint_round_trips_through_the_review_loader(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    """The load-bearing layout claim, measured on the real loader."""

    manifest, student = _write_checkpoint(
        published_child,
        tmp_path / "rung-10000",
        routes_were_trained=False,
        mutate=lambda value: _nudge_body(value),
    )
    assert manifest["route_provenance_mode"] == STAGE2_FROZEN_ROUTE_PROVENANCE
    assert manifest["stage2_checkpoint"] is True
    result = round_trip_stage2_rung_checkpoint(
        tmp_path / "rung-10000",
        base=TOY._ToyBaseCheckpointIO(None),
    )
    assert result["loaded_through_review_loader"] is True
    assert result["refused_by_stage1_guard"] is False
    assert result["semantic_step_generate_installed"] is True
    assert result["child_package_sha256"] == manifest["child_package_sha256"]
    reloaded = load_stage2_rung_checkpoint_manifest(
        tmp_path / "rung-10000",
        expected_manifest_sha256=str(manifest["manifest_sha256"]),
    )
    assert reloaded["arm_id"] == "S2-TOY"
    del student


def _nudge_body(student: Any) -> None:
    with torch.no_grad():
        for name, parameter in student.child.named_parameters():
            if name.startswith("model."):
                parameter.add_(0.01)


def test_a_trained_route_checkpoint_is_refused_at_the_pinned_stage1_message(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    """The truthful audit meets the guard; this is the lane-2 handoff."""

    manifest, _student = _write_checkpoint(
        published_child,
        tmp_path / "rung-trained",
        routes_were_trained=True,
    )
    assert manifest["route_provenance_mode"] == STAGE2_TRAINED_ROUTE_PROVENANCE
    assert manifest["route_operation_audit"]["optimizer_steps"] == 5
    assert manifest["route_operation_audit"]["backward_calls"] == 20
    result = round_trip_stage2_rung_checkpoint(
        tmp_path / "rung-trained",
        base=TOY._ToyBaseCheckpointIO(None),
    )
    assert result["loaded_through_review_loader"] is False
    assert result["refused_by_stage1_guard"] is True
    assert result["refusal_message"] == STAGE1_TRAINED_ROUTE_REFUSAL
    assert result["handoff"]["needs_cyril_approval"] is True
    assert (
        STAGE2_ROUTED_DECODE_ADMISSION_HANDOFF["stage1_refusal_message"]
        == STAGE1_TRAINED_ROUTE_REFUSAL
    )


def test_a_frozen_declaration_with_moved_coefficients_is_refused(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    """A zero-operation audit is never written over coefficients that moved."""

    def move_routes(student: Any) -> None:
        with torch.no_grad():
            student.output_coefficients[-1] += 0.25

    with pytest.raises(Stage2CheckpointLayoutError, match="would be false"):
        _write_checkpoint(
            published_child,
            tmp_path / "rung-lying",
            routes_were_trained=False,
            mutate=move_routes,
        )


def test_the_checkpoint_manifest_refuses_a_tampered_tree(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    _manifest, _student = _write_checkpoint(
        published_child,
        tmp_path / "rung-tamper",
        routes_were_trained=False,
    )
    weights = tmp_path / "rung-tamper" / "child" / "model.safetensors"
    with weights.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(Stage2CheckpointLayoutError, match="differs from its sealed"):
        load_stage2_rung_checkpoint_manifest(tmp_path / "rung-tamper")


def test_the_checkpoint_manifest_refuses_a_tampered_seal(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    _manifest, _student = _write_checkpoint(
        published_child,
        tmp_path / "rung-seal",
        routes_were_trained=False,
    )
    path = tmp_path / "rung-seal" / STAGE2_CHECKPOINT_MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["arm_id"] = "S2-OTHER"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Stage2CheckpointLayoutError, match="self-seal does not verify"):
        load_stage2_rung_checkpoint_manifest(tmp_path / "rung-seal")


def test_every_stage2_checkpoint_labels_itself_in_both_sealed_receipts(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    """A control must never read as a stage-1 publication to any consumer."""

    manifest, _student = _write_checkpoint(
        published_child,
        tmp_path / "rung-control",
        routes_were_trained=False,
        mode=STAGE2_RANDOM_INITIALIZATION,
        seed=5,
    )
    child = tmp_path / "rung-control" / "child"
    outer = json.loads(
        (child / CARRIER_CHILD_PACKAGE_RECEIPT).read_text(encoding="utf-8")
    )
    routes = json.loads(
        (child / "semantic-carrier-routes" / "carrier-routes-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert outer["stage2_checkpoint"]["stage2_checkpoint"] is True
    assert outer["stage2_checkpoint"]["arm_id"] == "S2-TOY"
    assert routes["stage2_route_provenance"]["stage2_checkpoint"] is True
    assert (
        routes["stage2_route_provenance"]["route_coefficients_from_source_artifact"]
        is False
    )
    assert manifest["route_provenance"][
        "route_coefficients_from_source_artifact"
    ] is False
    # The stage-1 child-local loader authenticates internal consistency, not
    # source-artifact identity, so this control *does* open -- which is exactly
    # why the explicit stage-2 admission list is the handoff, not the loader.
    result = round_trip_stage2_rung_checkpoint(
        tmp_path / "rung-control",
        base=TOY._ToyBaseCheckpointIO(None),
    )
    assert result["loaded_through_review_loader"] is True
    assert "second_finding" in STAGE2_ROUTED_DECODE_ADMISSION_HANDOFF


def test_writing_over_an_existing_rung_directory_is_refused(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "rung-existing"
    destination.mkdir()
    with pytest.raises(Stage2CheckpointLayoutError, match="already exists"):
        _write_checkpoint(
            published_child,
            destination,
            routes_were_trained=False,
        )
