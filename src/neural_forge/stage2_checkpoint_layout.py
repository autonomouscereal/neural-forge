"""Stage-2 rung checkpoints written in the published-child layout.

Design section 5.4 is the fairness mechanism of the whole measurement: fold
arms, controls, and the parent baseline must reach the reviewer through **one**
decode surface, or the comparison is void.  A rung checkpoint is therefore not
a ``state_dict`` -- it is a complete carrier child directory
(``config.json`` + weights + target tokenizer + ``semantic-carrier-routes/`` +
``semantic-source-lexicon/`` + the sealed outer receipt), laid out exactly as a
stage-1 publication, so the review runtime's own loader opens it unchanged.

Where this module deliberately stops
------------------------------------

``load_packaged_lfm2_carrier_routes`` requires the packaged route receipt to
carry a **zero-operation audit** (``training_steps``/``backward_calls``/
``optimizer_steps`` all zero).  That guard is correct and stays untouched.  It
also means the two cases split:

* **routes frozen** (``train_routes=False``, coefficients bit-identical to the
  initialization): the audit is truthfully zero, and the checkpoint loads
  end-to-end through the unmodified stage-1 loader today.  This is the case the
  toy round-trip proves, and it is what makes "published-child-compatible" a
  measured claim rather than a layout opinion.
* **routes trained** (the design's default, section 5.4 item 1): the audit is
  truthfully non-zero, and the stage-1 loader refuses with
  :data:`STAGE1_TRAINED_ROUTE_REFUSAL`.  Writing zeros there would be a lie
  about trained state, so this module writes the truth and refuses to launder
  it.  Admitting such a checkpoint is exactly the stage-2 admission path the
  design assigns to lane 2 (section 5.4 item 3,
  :data:`STAGE2_ROUTED_DECODE_ADMISSION_HANDOFF`), gated on Cyril's approval.

Every checkpoint this module writes is loudly labelled ``stage2_checkpoint:
true`` with its arm, rung, step and init-source digest inside both sealed
receipts, so a stage-2 artifact can never be mistaken for a stage-1 publication
by any consumer, including the future admission list.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .dynamic_circuit_ir import canonical_sha256
from .dynamic_counted_child_state import ContinuousChildStateAllocation
from .lfm2_carrier_bridge_publication import (
    PACKAGED_ROUTE_DIRECTORY,
    PACKAGED_ROUTE_RECEIPT_FILENAME,
    PACKAGED_ROUTE_SCHEMA,
    PACKAGED_ROUTE_TENSOR_FILENAME,
    PACKAGED_ROUTE_TENSOR_KEYS,
    carrier_materialization_tensor_sha256,
    child_table_tensor_sha256,
)
from .lfm2_carrier_child_package import (
    CARRIER_CHILD_PACKAGE_RECEIPT,
    CARRIER_CHILD_PACKAGE_SCHEMA,
    Lfm2CarrierChildCheckpointIO,
    load_authenticated_lfm2_carrier_child_package,
)
# The outer receipt's inventory is verified by the stage-1 loader against
# whatever this module writes.  Computing it with a second implementation could
# drift from the verifier, so the sealed inventory is built by the verifier's
# own helper rather than by a look-alike.
from .lfm2_carrier_child_package import _child_inventory
from .lfm2_semantic_source_lexicon import SEMANTIC_SOURCE_LEXICON_DIRECTORY
from .stage2_training_receipt import file_sha256


STAGE2_CHECKPOINT_MANIFEST_SCHEMA = "neural-forge-stage2-checkpoint-manifest-v1"
STAGE2_CHECKPOINT_MANIFEST_FILENAME = "stage2-checkpoint-manifest.json"
STAGE2_CHECKPOINT_CHILD_DIRECTORY = "child"
STAGE2_OPTIMIZER_STATE_FILENAME = "optimizer-state.pt"
STAGE2_FROZEN_ROUTE_PROVENANCE = "sealed-zero-operation-routes"
STAGE2_TRAINED_ROUTE_PROVENANCE = "stage2-trained-routes"
# The byte-exact stage-1 refusal a trained-route checkpoint currently meets.
# It is pinned so the handoff to lane 2 names a real message rather than a
# paraphrase, and so a change to that guard shows up as a test failure here.
STAGE1_TRAINED_ROUTE_REFUSAL = (
    "Packaged carrier routes do not have a zero-operation fold audit."
)
_PACKAGED_AUDIT_KEYS = (
    "training_steps",
    "backward_calls",
    "optimizer_steps",
    "parameter_pruning_calls",
    "parameter_quantization_calls",
    "parameter_sparsification_calls",
    "parameter_sharing_calls",
)
STAGE2_ROUTED_DECODE_ADMISSION_HANDOFF = {
    "owner": "lane-2 routed_decode_admission.py (design section 5.4 item 3)",
    "needs_cyril_approval": True,
    # Landed as ``neural_forge.stage2_routed_decode_admission``: an explicit,
    # separately-invoked lane keyed on the sealed
    # ``ADMITTED_LFM2_STAGE2_CHECKPOINT_LABELS``.  The record below still
    # describes the stage-1 guard exactly as it was, because the stage-1 guard
    # is exactly as it was -- an unlabelled caller still meets
    # ``stage1_refusal_message`` byte for byte.
    "stage2_admission_module": "neural_forge.stage2_routed_decode_admission",
    "stage2_admission_entrypoint": "admit_stage2_rung_checkpoint",
    "stage2_admission_label_list": (
        "neural_forge.lfm2_carrier_child_package."
        "ADMITTED_LFM2_STAGE2_CHECKPOINT_LABELS"
    ),
    "reason": (
        "A stage-2 checkpoint whose k32 route coefficients were trained "
        "carries a truthful non-zero packaged-route operation audit, which "
        "the stage-1 guard refuses by design."
    ),
    "stage1_refusal_message": STAGE1_TRAINED_ROUTE_REFUSAL,
    "stage1_guard": (
        "neural_forge.lfm2_carrier_bridge_publication."
        "load_packaged_lfm2_carrier_routes"
    ),
    "admission_key_material": (
        "stage2_checkpoint_manifest_sha256",
        "child_package_sha256",
        "config_sha256",
        "arm_id",
        "rung_tokens",
    ),
    "must_not_change": (
        "the stage-1 admitted-tuple guard, its refusal messages, or its tests"
    ),
    "second_finding": (
        "the child-local fresh-process loader authenticates internal "
        "consistency, not source-artifact identity of the coefficients, so a "
        "random-init control with frozen routes would otherwise load as if it "
        "were a published fold child; every stage-2 checkpoint is therefore "
        "labelled stage2_checkpoint=true in both sealed receipts for the "
        "admission list to key on"
    ),
}


class Stage2CheckpointLayoutError(RuntimeError):
    """A stage-2 rung checkpoint could not be written or reloaded honestly."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Stage2CheckpointLayoutError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Stage2CheckpointLayoutError(f"{name} must be nonnegative.")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage2CheckpointLayoutError(f"{name} must be a lowercase SHA-256.")
    return value


def _save_route_tensors(tensors: Mapping[str, Any], path: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise Stage2CheckpointLayoutError(
            "safetensors is required to write a stage-2 rung checkpoint."
        ) from exc
    save_file(
        {
            str(name): tensor.detach().to(device="cpu").contiguous()
            for name, tensor in tensors.items()
        },
        str(path),
    )


@dataclass(frozen=True, slots=True)
class Stage2RouteProvenance:
    """The initialization's route identity, carried into every checkpoint."""

    source_bridge_artifact_sha256: str
    source_bridge_tensor_file_sha256: str
    span_evidence_sha256: str
    initial_input_coefficients_sha256: str
    initial_output_coefficients_sha256: str
    coefficients_from_source_artifact: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_bridge_artifact_sha256": (
                self.source_bridge_artifact_sha256
            ),
            "span_evidence_sha256": self.span_evidence_sha256,
            "initial_input_coefficients_sha256": (
                self.initial_input_coefficients_sha256
            ),
            "initial_output_coefficients_sha256": (
                self.initial_output_coefficients_sha256
            ),
            "route_coefficients_from_source_artifact": (
                self.coefficients_from_source_artifact
            ),
        }


def stage2_route_provenance(
    package: Any,
    *,
    coefficients_from_source_artifact: bool,
    initial_input_coefficients: Any = None,
    initial_output_coefficients: Any = None,
) -> Stage2RouteProvenance:
    """Capture the initialization's route identity from its sealed package."""

    receipt = package.routes.receipt
    allocations = package.routes.allocations
    input_sha256 = (
        allocations[0].tensor_sha256
        if initial_input_coefficients is None
        else carrier_materialization_tensor_sha256(initial_input_coefficients)
    )
    output_sha256 = (
        allocations[1].tensor_sha256
        if initial_output_coefficients is None
        else carrier_materialization_tensor_sha256(initial_output_coefficients)
    )
    return Stage2RouteProvenance(
        source_bridge_artifact_sha256=_digest(
            receipt["source_bridge_artifact_sha256"],
            "source_bridge_artifact_sha256",
        ),
        source_bridge_tensor_file_sha256=_digest(
            receipt["source_bridge_tensor_file_sha256"],
            "source_bridge_tensor_file_sha256",
        ),
        span_evidence_sha256=_digest(
            receipt["span_evidence_sha256"],
            "span_evidence_sha256",
        ),
        initial_input_coefficients_sha256=input_sha256,
        initial_output_coefficients_sha256=output_sha256,
        coefficients_from_source_artifact=bool(coefficients_from_source_artifact),
    )


def _route_allocations(
    *,
    artifact_sha256: str,
    input_coefficients: Any,
    output_coefficients: Any,
) -> tuple[ContinuousChildStateAllocation, ContinuousChildStateAllocation]:
    return (
        ContinuousChildStateAllocation(
            state_id="semantic-carrier-route.c_in",
            parameter_elements=int(input_coefficients.numel()),
            tensor_sha256=carrier_materialization_tensor_sha256(
                input_coefficients
            ),
            artifact_sha256=artifact_sha256,
            dtype="float32",
        ),
        ContinuousChildStateAllocation(
            state_id="semantic-carrier-route.c_out",
            parameter_elements=int(output_coefficients.numel()),
            tensor_sha256=carrier_materialization_tensor_sha256(
                output_coefficients
            ),
            artifact_sha256=artifact_sha256,
            dtype="float32",
        ),
    )


def _write_route_package(
    child_directory: Path,
    *,
    student: Any,
    provenance: Stage2RouteProvenance,
    structural_tensors: Mapping[str, Any],
    operation_audit: Mapping[str, int],
    stage2_label: Mapping[str, Any],
) -> tuple[str, tuple[ContinuousChildStateAllocation, ...]]:
    """Write ``semantic-carrier-routes/`` for the checkpoint's live state."""

    package_root = child_directory / PACKAGED_ROUTE_DIRECTORY
    if package_root.exists() or package_root.is_symlink():
        raise Stage2CheckpointLayoutError(
            "Stage-2 checkpoint child already contains carrier-route state."
        )
    input_table, output_table = student.child_tables()
    input_coefficients = student.input_coefficients.detach().to(device="cpu")
    output_coefficients = student.output_coefficients.detach().to(device="cpu")
    tensors = {
        name: structural_tensors[name].detach().to(device="cpu")
        for name in PACKAGED_ROUTE_TENSOR_KEYS
        if name not in {"input_coefficients", "output_coefficients"}
    }
    tensors["input_coefficients"] = input_coefficients
    tensors["output_coefficients"] = output_coefficients
    if set(tensors) != set(PACKAGED_ROUTE_TENSOR_KEYS):
        raise Stage2CheckpointLayoutError(
            "Stage-2 route package is missing sealed structural tensors."
        )
    allocations = _route_allocations(
        artifact_sha256=provenance.source_bridge_artifact_sha256,
        input_coefficients=input_coefficients,
        output_coefficients=output_coefficients,
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=".stage2-routes.", dir=str(child_directory))
    ).resolve(strict=True)
    try:
        tensor_path = temporary / PACKAGED_ROUTE_TENSOR_FILENAME
        _save_route_tensors(tensors, tensor_path)
        body = {
            "schema": PACKAGED_ROUTE_SCHEMA,
            "status": "complete",
            "source_bridge_artifact_sha256": (
                provenance.source_bridge_artifact_sha256
            ),
            "source_bridge_tensor_file_sha256": (
                provenance.source_bridge_tensor_file_sha256
            ),
            "route_tensor_file": PACKAGED_ROUTE_TENSOR_FILENAME,
            "route_tensor_file_sha256": file_sha256(tensor_path),
            "route_tensor_keys": list(PACKAGED_ROUTE_TENSOR_KEYS),
            "input_route_allocation": allocations[0].as_dict(),
            "output_route_allocation": allocations[1].as_dict(),
            "span_evidence_sha256": provenance.span_evidence_sha256,
            "input_table_binding": {
                "parameter_path": "model.embed_tokens.weight",
                "route_state_id": "semantic-carrier-route.c_in",
                "shape": [int(size) for size in input_table.shape],
                "dtype": str(input_table.dtype).removeprefix("torch."),
                "tensor_sha256": child_table_tensor_sha256(input_table),
            },
            "output_table_binding": {
                "parameter_path": "lm_head.weight",
                "route_state_id": "semantic-carrier-route.c_out",
                "shape": [int(size) for size in output_table.shape],
                "dtype": str(output_table.dtype).removeprefix("torch."),
                "tensor_sha256": child_table_tensor_sha256(output_table),
            },
            "input_output_tables_are_untied": True,
            "input_output_table_storage_is_independent": True,
            "source_target_table_packaged": False,
            "source_target_table_used_as_child_weight": False,
            "integer_csr_routes_are_structural_metadata": True,
            "continuous_route_state_counted_exactly_once": True,
            "continuous_semantic_coefficient_count": sum(
                value.parameter_elements for value in allocations
            ),
            "standalone_runtime": True,
            "external_runtime_state_required": False,
            "parameter_sharing_credit": False,
            "quantization_credit": False,
            "pruning_credit": False,
            "operation_audit": dict(operation_audit),
            "stage2_route_provenance": {
                **provenance.as_dict(),
                **dict(stage2_label),
            },
        }
        receipt = {**body, "package_sha256": canonical_sha256(body)}
        (temporary / PACKAGED_ROUTE_RECEIPT_FILENAME).write_text(
            json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        os.rename(temporary, package_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return str(receipt["package_sha256"]), allocations


def _seal_stage2_child_package(
    child_directory: Path,
    *,
    route_package_sha256: str,
    allocations: tuple[ContinuousChildStateAllocation, ...],
    semantic_source_tokenizer_sha256: str,
    source_bridge_artifact_sha256: str,
    lexicon_package_sha256: str,
    target_rows: int,
    hidden_size: int,
    stage2_label: Mapping[str, Any],
) -> str:
    """Seal the outer child receipt for a stage-2 checkpoint.

    The body is the stage-1 body plus one ``stage2_checkpoint`` block.  The
    stage-1 loader validates named fields and re-derives the self-seal over the
    whole object, so the extra block is authenticated rather than ignored.
    """

    destination = child_directory / CARRIER_CHILD_PACKAGE_RECEIPT
    if destination.exists() or destination.is_symlink():
        raise Stage2CheckpointLayoutError(
            "Stage-2 checkpoint child already has an outer receipt."
        )
    records, directories = _child_inventory(child_directory)
    base_records = [
        value for value in records if "/" not in str(value["relative_path"])
    ]
    body = {
        "schema": CARRIER_CHILD_PACKAGE_SCHEMA,
        "status": "complete",
        "standalone_child_local_runtime": True,
        "external_source_publication_required_at_load": False,
        "external_source_tokenizer_required_at_load": False,
        "source_bridge_artifact_sha256": source_bridge_artifact_sha256,
        "semantic_source_tokenizer_sha256": semantic_source_tokenizer_sha256,
        "carrier_route_package_sha256": route_package_sha256,
        "semantic_source_lexicon_package_sha256": lexicon_package_sha256,
        "target_rows": _positive_int(target_rows, "target_rows"),
        "hidden_size": _positive_int(hidden_size, "hidden_size"),
        "route_allocations": [value.as_dict() for value in allocations],
        "continuous_carrier_parameter_elements": sum(
            value.parameter_elements for value in allocations
        ),
        "continuous_parameter_count_credit_from_lexicon": False,
        "base_root_file_records": base_records,
        "child_file_records": list(records),
        "child_directories": list(directories),
        "stage2_checkpoint": dict(stage2_label),
    }
    receipt = {**body, "package_sha256": canonical_sha256(body)}
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".stage2-child-receipt.",
        dir=str(child_directory),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return str(receipt["package_sha256"])


def _tree_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_dir():
            continue
        if path.is_symlink() or not path.is_file():
            raise Stage2CheckpointLayoutError(
                f"Stage-2 checkpoint contains an unsafe path {path.name!r}."
            )
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    return records


def write_stage2_rung_checkpoint(
    rung_directory: Path,
    *,
    student: Any,
    base: Any,
    provenance: Stage2RouteProvenance,
    structural_route_tensors: Mapping[str, Any],
    initialization_child_directory: Path,
    semantic_source_tokenizer_sha256: str,
    lexicon_package_sha256: str,
    arm_id: str,
    rung_tokens: int,
    tokens_consumed: int,
    optimizer_steps: int,
    backward_calls: int,
    init_source_sha256: str,
    routes_were_trained: bool,
    initialization_mode: str | None = None,
    init_artifact_sha256: str | None = None,
    optimizer: Any = None,
    save_optimizer: Callable[[Any, Path], None] | None = None,
) -> dict[str, Any]:
    """Write one ladder rung as a complete, sealed, labelled carrier child.

    ``initialization_mode`` and ``init_artifact_sha256`` are what the stage-2
    admission list keys on (``stage2_routed_decode_admission``).  They default
    to ``None`` so the writer keeps working for a checkpoint nobody intends to
    admit, and admission is fail-closed on the absence: a checkpoint that never
    stated its arm's initialization mode cannot match any sealed label.
    """

    rung_root = Path(rung_directory).expanduser().absolute()
    if rung_root.exists() or rung_root.is_symlink():
        raise Stage2CheckpointLayoutError(
            "Stage-2 rung directory already exists."
        )
    initialization_root = Path(initialization_child_directory).absolute()
    lexicon_source = initialization_root / SEMANTIC_SOURCE_LEXICON_DIRECTORY
    if not lexicon_source.is_dir() or lexicon_source.is_symlink():
        raise Stage2CheckpointLayoutError(
            "Initialization child has no packaged semantic-source lexicon."
        )
    rung_root.mkdir(parents=True)
    child_directory = rung_root / STAGE2_CHECKPOINT_CHILD_DIRECTORY
    # The base checkpoint IO owns destination creation, exactly as
    # ``Lfm2CarrierChildCheckpointIO.save`` lets it.
    base.save(student.child, child_directory)
    if not child_directory.is_dir():
        raise Stage2CheckpointLayoutError(
            "Base checkpoint IO did not produce a child directory."
        )
    # The lexicon is a tokenizer package: training cannot change it, so it is
    # copied verbatim rather than rebuilt from a source tokenizer the trainer
    # host is not required to have.
    shutil.copytree(
        lexicon_source,
        child_directory / SEMANTIC_SOURCE_LEXICON_DIRECTORY,
    )
    stage2_label = {
        "stage": "stage2",
        "stage2_checkpoint": True,
        "arm_id": str(arm_id),
        "rung_tokens": _positive_int(rung_tokens, "rung_tokens"),
        "tokens_consumed": _nonnegative_int(tokens_consumed, "tokens_consumed"),
        "optimizer_steps": _nonnegative_int(optimizer_steps, "optimizer_steps"),
        "backward_calls": _nonnegative_int(backward_calls, "backward_calls"),
        "init_source_sha256": _digest(init_source_sha256, "init_source_sha256"),
        "initialization_mode": (
            None if initialization_mode is None else str(initialization_mode)
        ),
        "init_artifact_sha256": (
            None
            if init_artifact_sha256 is None
            else _digest(init_artifact_sha256, "init_artifact_sha256")
        ),
        "route_coefficients_trained": bool(routes_were_trained),
        "route_provenance_mode": (
            STAGE2_TRAINED_ROUTE_PROVENANCE
            if routes_were_trained
            else STAGE2_FROZEN_ROUTE_PROVENANCE
        ),
    }
    if routes_were_trained:
        # Truthful: these coefficients moved under the optimizer.
        operation_audit = {
            "training_steps": int(optimizer_steps),
            "backward_calls": int(backward_calls),
            "optimizer_steps": int(optimizer_steps),
            "parameter_pruning_calls": 0,
            "parameter_quantization_calls": 0,
            "parameter_sparsification_calls": 0,
            "parameter_sharing_calls": 0,
        }
    else:
        observed = carrier_materialization_tensor_sha256(
            student.input_coefficients.detach().to(device="cpu")
        )
        observed_output = carrier_materialization_tensor_sha256(
            student.output_coefficients.detach().to(device="cpu")
        )
        if (
            observed != provenance.initial_input_coefficients_sha256
            or observed_output != provenance.initial_output_coefficients_sha256
        ):
            raise Stage2CheckpointLayoutError(
                "Route coefficients moved in a run that declared frozen "
                "routes; a zero-operation route audit would be false."
            )
        operation_audit = dict.fromkeys(_PACKAGED_AUDIT_KEYS, 0)
    route_package_sha256, allocations = _write_route_package(
        child_directory,
        student=student,
        provenance=provenance,
        structural_tensors=structural_route_tensors,
        operation_audit=operation_audit,
        stage2_label=stage2_label,
    )
    input_table, _output_table = student.child_tables()
    child_package_sha256 = _seal_stage2_child_package(
        child_directory,
        route_package_sha256=route_package_sha256,
        allocations=allocations,
        semantic_source_tokenizer_sha256=_digest(
            semantic_source_tokenizer_sha256,
            "semantic_source_tokenizer_sha256",
        ),
        source_bridge_artifact_sha256=(
            provenance.source_bridge_artifact_sha256
        ),
        lexicon_package_sha256=_digest(
            lexicon_package_sha256,
            "lexicon_package_sha256",
        ),
        target_rows=int(input_table.shape[0]),
        hidden_size=int(input_table.shape[1]),
        stage2_label=stage2_label,
    )
    optimizer_state_path: str | None = None
    if optimizer is not None:
        if save_optimizer is None:
            raise Stage2CheckpointLayoutError(
                "Saving optimizer state requires an explicit save_optimizer."
            )
        target = rung_root / STAGE2_OPTIMIZER_STATE_FILENAME
        save_optimizer(optimizer, target)
        if not target.is_file():
            raise Stage2CheckpointLayoutError(
                "save_optimizer did not write the optimizer state file."
            )
        optimizer_state_path = STAGE2_OPTIMIZER_STATE_FILENAME
    manifest_body = {
        "schema": STAGE2_CHECKPOINT_MANIFEST_SCHEMA,
        "status": "complete",
        **stage2_label,
        "child_directory": STAGE2_CHECKPOINT_CHILD_DIRECTORY,
        "child_package_sha256": child_package_sha256,
        "carrier_route_package_sha256": route_package_sha256,
        "route_operation_audit": dict(operation_audit),
        "route_provenance": provenance.as_dict(),
        "initialization_child_directory": str(initialization_root),
        "optimizer_state_file": optimizer_state_path,
        "published_child_layout": True,
        "stage1_artifacts_modified": False,
        "files": _tree_records(rung_root),
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": canonical_sha256(manifest_body),
    }
    (rung_root / STAGE2_CHECKPOINT_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return manifest


def load_stage2_rung_checkpoint_manifest(
    rung_directory: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Reload and re-verify a rung manifest, including every file digest."""

    rung_root = Path(rung_directory).expanduser().absolute()
    path = rung_root / STAGE2_CHECKPOINT_MANIFEST_FILENAME
    if path.is_symlink() or not path.is_file():
        raise Stage2CheckpointLayoutError(
            "Stage-2 rung manifest is missing or not a real file."
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage2CheckpointLayoutError(
                    f"Stage-2 rung manifest repeats key {key!r}."
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except Stage2CheckpointLayoutError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2CheckpointLayoutError(
            "Stage-2 rung manifest is not strict UTF-8 JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise Stage2CheckpointLayoutError("Stage-2 rung manifest must be an object.")
    body = dict(payload)
    seal = body.pop("manifest_sha256", None)
    if (
        body.get("schema") != STAGE2_CHECKPOINT_MANIFEST_SCHEMA
        or body.get("stage2_checkpoint") is not True
        or not isinstance(seal, str)
        or seal != canonical_sha256(body)
    ):
        raise Stage2CheckpointLayoutError(
            "Stage-2 rung manifest self-seal does not verify."
        )
    if expected_manifest_sha256 is not None and seal != _digest(
        expected_manifest_sha256,
        "expected manifest sha256",
    ):
        raise Stage2CheckpointLayoutError(
            "Stage-2 rung manifest belongs to a different checkpoint."
        )
    observed = _tree_records(rung_root)
    if [
        record
        for record in observed
        if record["relative_path"] != STAGE2_CHECKPOINT_MANIFEST_FILENAME
    ] != [
        dict(record)
        for record in body["files"]
        if record["relative_path"] != STAGE2_CHECKPOINT_MANIFEST_FILENAME
    ]:
        raise Stage2CheckpointLayoutError(
            "Stage-2 rung checkpoint tree differs from its sealed manifest."
        )
    return payload


def round_trip_stage2_rung_checkpoint(
    rung_directory: Path,
    *,
    base: Any,
    target_tokenizer_loader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Open a rung checkpoint with the loader the frozen review itself uses.

    Returns a structured result instead of raising when the stage-1 guard is
    the thing that refuses: a trained-route checkpoint *should* be refused
    today, and turning that into a receipt line is the point.  Any other
    failure still raises.
    """

    manifest = load_stage2_rung_checkpoint_manifest(rung_directory)
    child_directory = (
        Path(rung_directory).expanduser().absolute()
        / str(manifest["child_directory"])
    )
    loader_arguments: dict[str, Any] = {
        "base": base,
        "child_directory": child_directory,
    }
    if target_tokenizer_loader is not None:
        loader_arguments["target_tokenizer_loader"] = target_tokenizer_loader
    try:
        loader = Lfm2CarrierChildCheckpointIO.from_child_directory(
            **loader_arguments
        )
        model = loader.load(child_directory)
        package = load_authenticated_lfm2_carrier_child_package(
            child_directory,
            child_model=model,
            expected_package_sha256=loader.expected_child_package_sha256,
        )
    except Exception as exc:  # noqa: BLE001 - the refusal itself is the result
        if STAGE1_TRAINED_ROUTE_REFUSAL in str(exc):
            return {
                "loaded_through_review_loader": False,
                "refused_by_stage1_guard": True,
                "refusal_message": str(exc),
                "manifest_sha256": str(manifest["manifest_sha256"]),
                "route_provenance_mode": str(manifest["route_provenance_mode"]),
                "handoff": dict(STAGE2_ROUTED_DECODE_ADMISSION_HANDOFF),
            }
        raise
    return {
        "loaded_through_review_loader": True,
        "refused_by_stage1_guard": False,
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "route_provenance_mode": str(manifest["route_provenance_mode"]),
        "child_package_sha256": package.package_sha256,
        "semantic_step_generate_installed": callable(
            getattr(model, "semantic_step_generate", None)
        ),
        "model": model,
        "package": package,
    }


__all__ = [
    "STAGE1_TRAINED_ROUTE_REFUSAL",
    "STAGE2_CHECKPOINT_CHILD_DIRECTORY",
    "STAGE2_CHECKPOINT_MANIFEST_FILENAME",
    "STAGE2_CHECKPOINT_MANIFEST_SCHEMA",
    "STAGE2_FROZEN_ROUTE_PROVENANCE",
    "STAGE2_OPTIMIZER_STATE_FILENAME",
    "STAGE2_ROUTED_DECODE_ADMISSION_HANDOFF",
    "STAGE2_TRAINED_ROUTE_PROVENANCE",
    "Stage2CheckpointLayoutError",
    "Stage2RouteProvenance",
    "load_stage2_rung_checkpoint_manifest",
    "round_trip_stage2_rung_checkpoint",
    "stage2_route_provenance",
    "write_stage2_rung_checkpoint",
]
