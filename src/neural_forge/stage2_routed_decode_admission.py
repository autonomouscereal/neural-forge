"""The stage-2 routed-decode admission lane (design section 5.4 item 3).

What this is, and what it deliberately is not
---------------------------------------------

A stage-2 checkpoint whose k32 route coefficients were trained carries a
**truthful non-zero** packaged-route operation audit.  The stage-1 loader
refuses it at

    Packaged carrier routes do not have a zero-operation fold audit.

That refusal is correct, it is the honest thing for a stage-1 lane to say, and
nothing in this module changes it.  ``load_packaged_lfm2_carrier_routes``, its
message, and its tests are untouched; an unlabelled caller opening a trained
stage-2 checkpoint through
``Lfm2CarrierChildCheckpointIO.from_child_directory(...).load(...)`` still meets
that byte-exact string.  ``tests/stage2/test_stage2_routed_decode_admission.py``
pins exactly that, on the same checkpoint this module admits.

This module is a **parallel lane you have to ask for by name**.  It admits one
checkpoint only when *both* of these hold:

1. the caller presents one of the sealed stage-2 labels from
   ``ADMITTED_LFM2_STAGE2_CHECKPOINT_LABELS`` -- by identity, so an equal-valued
   label assembled at a call site is refused, exactly as an equal-valued carrier
   ABI tuple is refused by ``admitted_counted_plan``; and
2. the checkpoint's own sealed identity matches: the caller-declared manifest
   digest, child-package digest and init-source digest are the ones on disk, and
   the checkpoint's sealed stage-2 label block agrees with the presented label
   on arm id, initialization mode, initialization artifact and trained-routes
   flag.

Finding 3 of NOTES-S2B is why (1) exists at all: the child-local loader
authenticates *internal consistency*, not source-artifact identity, so a
random-init control opens as if it were a published fold child.  No loader can
tell the arms apart on its own.  An explicit label can.

Why the route authentication is restated here
---------------------------------------------

The stage-1 route loader refuses at the audit *before* it authenticates any
tensor, so there is no way to reach its structural checks for a trained
checkpoint without editing it.  Editing it is out of scope by design.  Writing a
zeroed audit into a trained checkpoint -- or into a temporary copy of one -- is
worse: it manufactures an artifact that lies about trained state, which is the
exact failure the stage-1 guard exists to prevent.

So this module restates the authentication, and pays the drift risk down two
ways.  Every shared step is executed by **the stage-1 verifier's own helpers**
(``_sealed_json``, ``_load_authenticated_tensors``, ``_route_allocations``,
``_child_tables``, ``_child_inventory``), not by look-alikes, so a refusal
raised inside a shared helper is the stage-1 refusal verbatim, type and message.
And ``test_the_stage2_twin_matches_the_stage1_loader_where_stage1_will_run``
runs both loaders over the same frozen-route checkpoint and a battery of
corruptions of it, and requires identical results and identical refusal strings.

The stage-2 rule differs from the stage-1 rule in exactly one place: the three
*training* counters of the operation audit may be non-zero when they equal the
checkpoint's own sealed training record.  The four credit counters
(pruning, quantization, sparsification, sharing) must still be exactly zero --
stage 2 trains, it does not prune, quantize, sparsify or share, and no label
admits a checkpoint that claims otherwise.  A package with no
``stage2_route_provenance`` block is refused outright, so this is not a relaxed
general-purpose loader: it cannot open a stage-1 published child at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

import torch

from .dynamic_circuit_ir import canonical_sha256
from .hf_lfm2_direct_local import load_lfm2_direct_local_tokenizer
from .lfm2_carrier_bridge_publication import (
    PACKAGED_ROUTE_DIRECTORY,
    PACKAGED_ROUTE_RECEIPT_FILENAME,
    PACKAGED_ROUTE_SCHEMA,
    PACKAGED_ROUTE_TENSOR_FILENAME,
    PACKAGED_ROUTE_TENSOR_KEYS,
    PACKAGED_TARGET_ROUTE_KEY,
    AuthenticatedPackagedLfm2CarrierRoutes,
    Lfm2CarrierBridgePublicationError,
)
# Deliberate: these are the stage-1 verifier's own steps.  Re-implementing any
# of them here would put a second, drifting copy of a security check in the
# tree; calling them means a shared refusal is byte-identical to stage 1's.
from .lfm2_carrier_bridge_publication import (  # noqa: PLC2701
    _STRUCTURAL_ROUTE_KEYS,
    _bind_lfm2_carrier_runtime,
    _child_tables,
    _load_authenticated_tensors,
    _ordinary_directory,
    _ordinary_file,
    _route_allocations,
    _sealed_json,
    _stable_file_identity,
    _strict_json_bytes,
)
from .lfm2_carrier_child_package import (
    CARRIER_CHILD_PACKAGE_RECEIPT,
    CARRIER_CHILD_PACKAGE_SCHEMA,
    AuthenticatedLfm2CarrierChildPackage,
    Lfm2Stage2CheckpointLabel,
    describe_admitted_lfm2_stage2_checkpoint_labels,
    is_admitted_lfm2_stage2_checkpoint_label,
)
from .lfm2_carrier_child_package import (  # noqa: PLC2701
    _child_inventory,
    _ordinary_child_directory,
    _strict_receipt,
)
from .lfm2_carrier_route_runtime import install_lfm2_semantic_step_generate
from .lfm2_semantic_source_lexicon import (
    load_packaged_lfm2_semantic_source_lexicon,
)
from .stage2_checkpoint_layout import (
    STAGE1_TRAINED_ROUTE_REFUSAL,
    load_stage2_rung_checkpoint_manifest,
)


STAGE2_ROUTED_DECODE_ADMISSION_SCHEMA = (
    "neural-forge-stage2-routed-decode-admission-v1"
)
# The three counters stage 2 legitimately moves, and the four it must not.
STAGE2_ADMITTED_TRAINING_AUDIT_KEYS = (
    "backward_calls",
    "optimizer_steps",
    "training_steps",
)
STAGE2_FORBIDDEN_AUDIT_CREDIT_KEYS = (
    "parameter_pruning_calls",
    "parameter_quantization_calls",
    "parameter_sharing_calls",
    "parameter_sparsification_calls",
)


class Stage2RoutedDecodeAdmissionError(RuntimeError):
    """A stage-2 checkpoint was not admitted to the routed decode."""


@dataclass(frozen=True, slots=True)
class Stage2CheckpointIdentity:
    """The run-time half of the key: which checkpoint the caller means.

    The label is sealed at design time and names an *arm*; these three digests
    are produced by the run and name one *checkpoint*.  Admission needs both,
    which is why neither can be defaulted.
    """

    manifest_sha256: str
    child_package_sha256: str
    init_source_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "child_package_sha256": self.child_package_sha256,
            "init_source_sha256": self.init_source_sha256,
        }


def _refuse(message: str) -> None:
    raise Stage2RoutedDecodeAdmissionError(message)


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _refuse(f"{name} must be a lowercase SHA-256.")
    return str(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _refuse(f"{name} must be a positive integer.")
    return int(value)


def admitted_stage2_checkpoint_label(
    label: Any,
) -> Lfm2Stage2CheckpointLabel:
    """Return ``label`` only if it *is* one of the sealed label objects."""

    if not isinstance(
        label, Lfm2Stage2CheckpointLabel
    ) or not is_admitted_lfm2_stage2_checkpoint_label(label):
        _refuse(
            "Stage-2 routed decode requires one sealed stage-2 checkpoint "
            f"label ({describe_admitted_lfm2_stage2_checkpoint_labels()})."
        )
    return label


def _stage2_operation_audit(
    audit: Any,
    *,
    label: Lfm2Stage2CheckpointLabel,
    optimizer_steps: int,
    backward_calls: int,
) -> dict[str, int]:
    """The one rule that differs from stage 1, stated in full.

    Stage 1 requires every counter to be zero.  Stage 2 requires the four
    credit counters to be zero *and* the three training counters to equal this
    checkpoint's own sealed training record -- so the audit still cannot say
    anything the receipt does not already say, it just no longer has to say
    nothing.
    """

    expected = {
        "backward_calls": int(backward_calls),
        "optimizer_steps": int(optimizer_steps),
        "training_steps": int(optimizer_steps),
        **dict.fromkeys(STAGE2_FORBIDDEN_AUDIT_CREDIT_KEYS, 0),
    }
    if not isinstance(audit, Mapping) or dict(audit) != expected:
        _refuse(
            "Stage-2 packaged route audit must be this checkpoint's own "
            "training record with zero pruning, quantization, sparsification "
            "and sharing credit."
        )
    if label.route_coefficients_trained and any(
        expected[name] <= 0 for name in STAGE2_ADMITTED_TRAINING_AUDIT_KEYS
    ):
        _refuse(
            f"Stage-2 label {label.label_id!r} admits trained routes, but this "
            "checkpoint recorded no training at all."
        )
    return expected


def _stage2_route_provenance(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = receipt.get("stage2_route_provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("stage") != "stage2"
        or provenance.get("stage2_checkpoint") is not True
    ):
        _refuse(
            "Stage-2 routed decode admits only packages labelled "
            "stage2_checkpoint=true; this is not a stage-2 checkpoint."
        )
    return provenance


def authenticate_stage2_packaged_carrier_routes(
    package_or_child_directory: str | Path,
    *,
    expected_source_artifact_sha256: str,
    label: Lfm2Stage2CheckpointLabel,
    optimizer_steps: int,
    backward_calls: int,
    child_model: Any | None = None,
    expected_target_rows: int | None = None,
    expected_hidden_size: int | None = None,
) -> AuthenticatedPackagedLfm2CarrierRoutes:
    """Authenticate a *labelled stage-2* packaged route directory.

    Structurally identical to ``load_packaged_lfm2_carrier_routes`` -- same
    helpers, same order, same refusals -- except that the operation audit is
    checked by :func:`_stage2_operation_audit` and the package must carry a
    stage-2 provenance block.
    """

    supplied = _ordinary_directory(
        package_or_child_directory,
        "packaged carrier-route path",
    )
    direct_payload = (
        (supplied / PACKAGED_ROUTE_RECEIPT_FILENAME).is_file()
        and not (supplied / PACKAGED_ROUTE_RECEIPT_FILENAME).is_symlink()
        and (supplied / PACKAGED_ROUTE_TENSOR_FILENAME).is_file()
        and not (supplied / PACKAGED_ROUTE_TENSOR_FILENAME).is_symlink()
    )
    root = (
        supplied
        if supplied.name == PACKAGED_ROUTE_DIRECTORY or direct_payload
        else _ordinary_directory(
            supplied / PACKAGED_ROUTE_DIRECTORY,
            "packaged carrier-route directory",
        )
    )
    expected_inventory = {
        PACKAGED_ROUTE_RECEIPT_FILENAME,
        PACKAGED_ROUTE_TENSOR_FILENAME,
    }
    entries = tuple(root.iterdir())
    if {path.name for path in entries} != expected_inventory or any(
        path.is_symlink()
        or not path.is_file()
        or getattr(path.stat(), "st_nlink", 1) != 1
        for path in entries
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier-route directory contains unsealed state."
        )
    receipt_path = _ordinary_file(
        root / PACKAGED_ROUTE_RECEIPT_FILENAME,
        "packaged carrier-route receipt",
    )
    before = _stable_file_identity(receipt_path)
    raw_receipt = _strict_json_bytes(
        receipt_path.read_bytes(),
        "packaged carrier-route receipt",
    )
    if _stable_file_identity(receipt_path) != before:
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier-route receipt changed while being read."
        )
    if not isinstance(raw_receipt, Mapping):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier-route receipt must be an object."
        )
    receipt = _sealed_json(
        raw_receipt,
        schema=PACKAGED_ROUTE_SCHEMA,
        seal_field="package_sha256",
        label="packaged carrier-route receipt",
    )
    source_sha256 = str(receipt.get("source_bridge_artifact_sha256"))
    if source_sha256 != _digest(
        expected_source_artifact_sha256,
        "expected packaged source bridge artifact SHA-256",
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier routes belong to a different source publication."
        )
    if (
        receipt.get("status") != "complete"
        or receipt.get("route_tensor_file") != PACKAGED_ROUTE_TENSOR_FILENAME
        or receipt.get("route_tensor_keys") != list(PACKAGED_ROUTE_TENSOR_KEYS)
        or not isinstance(receipt.get("span_evidence_sha256"), str)
        or receipt.get("input_output_tables_are_untied") is not True
        or receipt.get("input_output_table_storage_is_independent") is not True
        or receipt.get("source_target_table_packaged") is not False
        or receipt.get("source_target_table_used_as_child_weight") is not False
        or receipt.get("integer_csr_routes_are_structural_metadata") is not True
        or receipt.get("continuous_route_state_counted_exactly_once") is not True
        or receipt.get("standalone_runtime") is not True
        or receipt.get("external_runtime_state_required") is not False
        or receipt.get("parameter_sharing_credit") is not False
        or receipt.get("quantization_credit") is not False
        or receipt.get("pruning_credit") is not False
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier-route policy is incomplete or unsafe."
        )
    # -- the one stage-2 difference, and its stage-2-only precondition --------
    _stage2_route_provenance(receipt)
    _stage2_operation_audit(
        receipt.get("operation_audit"),
        label=label,
        optimizer_steps=optimizer_steps,
        backward_calls=backward_calls,
    )
    # ------------------------------------------------------------------------
    tensors, _observed_sha256 = _load_authenticated_tensors(
        root / PACKAGED_ROUTE_TENSOR_FILENAME,
        expected_sha256=str(receipt.get("route_tensor_file_sha256")),
        label="packaged carrier-route tensors",
    )
    if set(tensors) != set(PACKAGED_ROUTE_TENSOR_KEYS):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier routes have an unexpected tensor key set."
        )
    if "target_table" in tensors:
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier routes must not contain the source target table."
        )
    if any(
        not isinstance(tensors[name], torch.Tensor)
        or tensors[name].dtype is not torch.int64
        or tensors[name].ndim != 1
        for name in _STRUCTURAL_ROUTE_KEYS
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier CSR metadata must be one-dimensional int64 tensors."
        )
    input_coefficients = tensors["input_coefficients"]
    output_coefficients = tensors["output_coefficients"]
    if any(
        not isinstance(value, torch.Tensor)
        or value.dtype is not torch.float32
        or value.ndim != 1
        or not bool(torch.isfinite(value).all())
        for value in (input_coefficients, output_coefficients)
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged C_in/C_out must be finite independent FP32 vectors."
        )
    if (
        int(tensors["input_offsets"][0].item()) != 0
        or int(tensors["input_offsets"][-1].item())
        != int(input_coefficients.numel())
        or int(tensors["output_offsets"][0].item()) != 0
        or int(tensors["output_offsets"][-1].item())
        != int(output_coefficients.numel())
        or int(tensors["input_target_ids"].numel())
        != int(input_coefficients.numel())
        or int(tensors["output_target_ids"].numel())
        != int(output_coefficients.numel())
        or int(tensors["span_offsets"][0].item()) != 0
        or int(tensors["span_offsets"][-1].item())
        != int(tensors["span_target_ids"].numel())
        or int(tensors["span_offsets"].numel())
        != int(tensors["input_offsets"].numel())
        or int(tensors["span_offsets"].numel())
        != int(tensors["output_offsets"].numel())
        or bool((tensors["span_offsets"][1:] <= tensors["span_offsets"][:-1]).any())
        or bool((tensors["input_offsets"][1:] < tensors["input_offsets"][:-1]).any())
        or bool((tensors["output_offsets"][1:] < tensors["output_offsets"][:-1]).any())
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier CSR offsets, indices, and coefficients disagree."
        )
    target_row_count = int(tensors[PACKAGED_TARGET_ROUTE_KEY].numel())
    if target_row_count <= 0 or any(
        bool((tensors[name] < 0).any())
        or bool((tensors[name] >= target_row_count).any())
        for name in ("input_target_ids", "output_target_ids", "span_target_ids")
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier route or span IDs leave the target vocabulary."
        )
    allocations = _route_allocations(
        artifact_sha256=source_sha256,
        input_coefficients=input_coefficients,
        output_coefficients=output_coefficients,
    )
    serialized_allocations = (
        receipt.get("input_route_allocation"),
        receipt.get("output_route_allocation"),
    )
    if any(
        not isinstance(serialized, Mapping)
        or serialized.get("state_id") != allocation.state_id
        or serialized.get("parameter_elements") != allocation.parameter_elements
        or serialized.get("tensor_sha256") != allocation.tensor_sha256
        or serialized.get("artifact_sha256") != allocation.artifact_sha256
        or serialized.get("dtype") != allocation.dtype
        for serialized, allocation in zip(
            serialized_allocations,
            allocations,
            strict=True,
        )
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier-route allocations differ from loaded tensors."
        )
    if receipt.get("continuous_semantic_coefficient_count") != sum(
        value.parameter_elements for value in allocations
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier-route count differs from its tensor payload."
        )
    input_binding = receipt.get("input_table_binding")
    output_binding = receipt.get("output_table_binding")
    if (
        not isinstance(input_binding, Mapping)
        or not isinstance(output_binding, Mapping)
        or input_binding.get("parameter_path") != "model.embed_tokens.weight"
        or output_binding.get("parameter_path") != "lm_head.weight"
        or input_binding.get("route_state_id") != allocations[0].state_id
        or output_binding.get("route_state_id") != allocations[1].state_id
        or input_binding.get("shape") != output_binding.get("shape")
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Packaged routes are not bound to separate input/readout tables."
        )
    if child_model is not None:
        _input, _output, observed_input, observed_output = _child_tables(
            child_model,
            expected_target_rows=expected_target_rows,
            expected_hidden_size=expected_hidden_size,
        )
        if (
            dict(input_binding) != observed_input
            or dict(output_binding) != observed_output
        ):
            raise Lfm2CarrierBridgePublicationError(
                "Fresh child input/readout tables differ from carrier bindings."
            )
    elif expected_target_rows is not None or expected_hidden_size is not None:
        if input_binding.get("shape") != [expected_target_rows, expected_hidden_size]:
            raise Lfm2CarrierBridgePublicationError(
                "Packaged carrier table binding has the wrong exact-body shape."
            )
    return AuthenticatedPackagedLfm2CarrierRoutes(
        root=root,
        receipt=MappingProxyType(dict(receipt)),
        tensors=MappingProxyType(dict(tensors)),
        allocations=allocations,
    )


def authenticate_stage2_child_package(
    child_directory: str | Path,
    *,
    label: Lfm2Stage2CheckpointLabel,
    optimizer_steps: int,
    backward_calls: int,
    child_model: Any | None = None,
    expected_package_sha256: str | None = None,
) -> AuthenticatedLfm2CarrierChildPackage:
    """``load_authenticated_lfm2_carrier_child_package`` on the stage-2 lane.

    Same inventory, same self-seal, same nested-identity cross-checks, same
    lexicon loader.  Only the route authentication is the stage-2 twin, and the
    outer receipt must additionally carry its ``stage2_checkpoint`` block.
    """

    root = _ordinary_child_directory(child_directory)
    raw = _strict_receipt(root / CARRIER_CHILD_PACKAGE_RECEIPT)
    seal = raw.pop("package_sha256", None)
    if (
        raw.get("schema") != CARRIER_CHILD_PACKAGE_SCHEMA
        or raw.get("status") != "complete"
        or raw.get("standalone_child_local_runtime") is not True
        or raw.get("external_source_publication_required_at_load") is not False
        or raw.get("external_source_tokenizer_required_at_load") is not False
        or raw.get("continuous_parameter_count_credit_from_lexicon") is not False
        or not isinstance(seal, str)
        or seal != canonical_sha256(raw)
    ):
        _refuse("Carrier child package receipt or self-seal is invalid.")
    if expected_package_sha256 is not None and seal != _digest(
        expected_package_sha256,
        "expected carrier child package SHA-256",
    ):
        _refuse("Carrier child package belongs to a different sealed child.")
    records, directories = _child_inventory(root)
    if raw.get("child_file_records") != list(records) or raw.get(
        "child_directories"
    ) != list(directories):
        _refuse("Carrier child package tree differs from its sealed inventory.")
    base_records = [
        value for value in records if "/" not in str(value["relative_path"])
    ]
    if raw.get("base_root_file_records") != base_records:
        _refuse("Carrier child base checkpoint inventory is not authentic.")
    stage2_label = raw.get("stage2_checkpoint")
    if (
        not isinstance(stage2_label, Mapping)
        or stage2_label.get("stage2_checkpoint") is not True
    ):
        _refuse(
            "Stage-2 routed decode admits only children labelled "
            "stage2_checkpoint=true in their outer sealed receipt."
        )
    packaged = authenticate_stage2_packaged_carrier_routes(
        root,
        expected_source_artifact_sha256=_digest(
            raw.get("source_bridge_artifact_sha256"),
            "sealed source bridge artifact SHA-256",
        ),
        label=label,
        optimizer_steps=optimizer_steps,
        backward_calls=backward_calls,
        child_model=child_model,
        expected_target_rows=_positive_int(
            raw.get("target_rows"),
            "sealed target_rows",
        ),
        expected_hidden_size=_positive_int(
            raw.get("hidden_size"),
            "sealed hidden_size",
        ),
    )
    lexicon = load_packaged_lfm2_semantic_source_lexicon(
        root,
        expected_tokenizer_semantics_sha256=_digest(
            raw.get("semantic_source_tokenizer_sha256"),
            "sealed semantic-source tokenizer SHA-256",
        ),
    )
    if (
        raw.get("carrier_route_package_sha256") != packaged.package_sha256
        or raw.get("semantic_source_lexicon_package_sha256")
        != lexicon.package_sha256
        or raw.get("route_allocations")
        != [value.as_dict() for value in packaged.allocations]
        or raw.get("continuous_carrier_parameter_elements")
        != sum(value.parameter_elements for value in packaged.allocations)
    ):
        _refuse("Carrier child nested package identities or counts changed.")
    return AuthenticatedLfm2CarrierChildPackage(
        root=root,
        receipt=MappingProxyType({**raw, "package_sha256": seal}),
        routes=packaged,
        semantic_source_lexicon=lexicon,
    )


def bind_stage2_packaged_carrier_routes(
    child_model: Any,
    packaged: AuthenticatedPackagedLfm2CarrierRoutes,
    *,
    label: Lfm2Stage2CheckpointLabel,
    optimizer_steps: int,
    backward_calls: int,
    expected_target_rows: int | None = None,
    expected_hidden_size: int | None = None,
) -> Any:
    """``bind_packaged_lfm2_carrier_routes`` on the stage-2 lane.

    The stage-1 binder cannot serve a trained checkpoint even after admission:
    it re-opens the package with ``load_packaged_lfm2_carrier_routes`` at bind
    time -- deliberately, so a retained mutable tensor mapping is never trusted
    -- and that re-open meets the audit guard again.  This keeps the re-open
    discipline verbatim and only swaps in the stage-2 twin, then hands the
    identical tensors to the stage-1 runtime binder ``_bind_lfm2_carrier_runtime``.
    """

    if not isinstance(packaged, AuthenticatedPackagedLfm2CarrierRoutes):
        raise Lfm2CarrierBridgePublicationError(
            "packaged must be an authenticated carrier-route bundle."
        )
    authenticated = authenticate_stage2_packaged_carrier_routes(
        packaged.root,
        expected_source_artifact_sha256=str(
            packaged.receipt["source_bridge_artifact_sha256"]
        ),
        label=label,
        optimizer_steps=optimizer_steps,
        backward_calls=backward_calls,
        child_model=child_model,
        expected_target_rows=expected_target_rows,
        expected_hidden_size=expected_hidden_size,
    )
    if (
        authenticated.package_sha256 != packaged.package_sha256
        or authenticated.allocations != packaged.allocations
        or set(authenticated.tensors) != set(packaged.tensors)
        or any(
            not isinstance(packaged.tensors[name], torch.Tensor)
            or packaged.tensors[name].dtype != authenticated.tensors[name].dtype
            or tuple(packaged.tensors[name].shape)
            != tuple(authenticated.tensors[name].shape)
            or not bool(
                torch.equal(
                    packaged.tensors[name].detach().to(device="cpu"),
                    authenticated.tensors[name],
                )
            )
            for name in authenticated.tensors
        )
    ):
        raise Lfm2CarrierBridgePublicationError(
            "Carrier routes changed between authentication and live binding."
        )
    tensors = authenticated.tensors
    offsets = tensors["span_offsets"]
    target_ids = tensors["span_target_ids"]
    spans = []
    for donor_id in range(int(offsets.numel()) - 1):
        start = int(offsets[donor_id].item())
        stop = int(offsets[donor_id + 1].item())
        spans.append([int(value) for value in target_ids[start:stop].tolist()])
    try:
        from .hf_lfm2_semantic_step_token_fold import (
            seal_lfm2_semantic_step_span_evidence,
        )

        span_evidence = seal_lfm2_semantic_step_span_evidence(
            spans,
            target_row_count=int(tensors[PACKAGED_TARGET_ROUTE_KEY].numel()),
            lineage={
                "packaged_span_evidence_sha256": authenticated.receipt[
                    "span_evidence_sha256"
                ],
                "carrier_route_package_sha256": authenticated.package_sha256,
                "derived_only_from_packaged_integer_span_tensors": True,
            },
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CarrierBridgePublicationError(
            "Packaged carrier routes could not bind to fresh child tables."
        ) from exc
    return _bind_lfm2_carrier_runtime(
        child_model,
        span_evidence=span_evidence,
        input_offsets=tensors["input_offsets"],
        input_target_ids=tensors["input_target_ids"],
        input_coefficients=tensors["input_coefficients"],
        output_offsets=tensors["output_offsets"],
        output_target_ids=tensors["output_target_ids"],
        output_coefficients=tensors["output_coefficients"],
        direct_target_row_to_semantic_id=tensors[PACKAGED_TARGET_ROUTE_KEY],
        expected_allocations=authenticated.allocations,
        expected_target_rows=expected_target_rows,
        expected_hidden_size=expected_hidden_size,
    )


def _match_label_to_checkpoint(
    stage2_label: Mapping[str, Any],
    *,
    label: Lfm2Stage2CheckpointLabel,
) -> None:
    """The label half of the key: this arm, this provenance, these routes."""

    arm_id = stage2_label.get("arm_id")
    mode = stage2_label.get("initialization_mode")
    artifact = stage2_label.get("init_artifact_sha256")
    trained = stage2_label.get("route_coefficients_trained")
    if mode is None:
        _refuse(
            "This checkpoint never recorded its initialization mode, so no "
            "sealed stage-2 label can match it."
        )
    if not label.matches(
        arm_id=arm_id,
        initialization_mode=mode,
        init_artifact_sha256=artifact,
        route_coefficients_trained=trained,
    ):
        _refuse(
            f"Stage-2 label {label.label_id!r} does not describe this "
            f"checkpoint ({label.describe()}); the checkpoint declares arm "
            f"{arm_id!r} / {mode!r} / init {artifact!r} / trained routes "
            f"{trained!r}."
        )


def admit_stage2_rung_checkpoint(
    rung_directory: str | Path,
    *,
    label: Lfm2Stage2CheckpointLabel,
    identity: Stage2CheckpointIdentity,
    base: Any = None,
    target_tokenizer_loader: Callable[[Path], Any] = (
        load_lfm2_direct_local_tokenizer
    ),
    bind_decode_runtime: bool = True,
) -> dict[str, Any]:
    """Admit one labelled stage-2 rung checkpoint to the routed decode.

    ``label`` and ``identity`` are both required: a sealed arm label alone
    admits nothing, and a checkpoint's own digests alone admit nothing.  With
    ``bind_decode_runtime`` and a ``base`` checkpoint IO the result carries a
    live model with ``semantic_step_generate`` installed -- the same decode
    surface the frozen review uses -- so fold arms, controls and the parent
    baseline still reach the reviewer through one path (design section 5.4
    item 4).

    Returns a sealed admission record.  It raises rather than returning a
    refusal, because unlike the stage-1 round trip (where a refusal is the
    expected outcome and therefore evidence), an admission that did not happen
    has nothing to record.
    """

    admitted = admitted_stage2_checkpoint_label(label)
    if not isinstance(identity, Stage2CheckpointIdentity):
        _refuse(
            "Stage-2 routed decode requires an explicit Stage2CheckpointIdentity."
        )
    rung_root = Path(rung_directory).expanduser().absolute()
    manifest = load_stage2_rung_checkpoint_manifest(
        rung_root,
        expected_manifest_sha256=_digest(
            identity.manifest_sha256,
            "declared manifest sha256",
        ),
    )
    if (
        manifest.get("stage") != "stage2"
        or manifest.get("stage2_checkpoint") is not True
    ):
        _refuse("Stage-2 routed decode admits only stage-2 rung checkpoints.")
    if manifest.get("child_package_sha256") != _digest(
        identity.child_package_sha256,
        "declared child package sha256",
    ):
        _refuse("Declared child package digest is not this checkpoint's.")
    if manifest.get("init_source_sha256") != _digest(
        identity.init_source_sha256,
        "declared init source sha256",
    ):
        _refuse("Declared init source digest is not this checkpoint's.")
    _match_label_to_checkpoint(manifest, label=admitted)
    optimizer_steps = int(manifest["optimizer_steps"])
    backward_calls = int(manifest["backward_calls"])
    child_directory = rung_root / str(manifest["child_directory"])
    package = authenticate_stage2_child_package(
        child_directory,
        label=admitted,
        optimizer_steps=optimizer_steps,
        backward_calls=backward_calls,
        expected_package_sha256=identity.child_package_sha256,
    )
    outer_label = package.receipt["stage2_checkpoint"]
    if dict(outer_label) != {
        key: value
        for key, value in manifest.items()
        if key in set(outer_label)
    }:
        _refuse(
            "The outer child receipt's stage-2 label disagrees with the rung "
            "manifest's."
        )
    _match_label_to_checkpoint(outer_label, label=admitted)
    model: Any | None = None
    if bind_decode_runtime:
        if base is None or not callable(getattr(base, "load", None)):
            _refuse(
                "Binding the stage-2 routed decode needs a base checkpoint IO "
                "with load(); pass bind_decode_runtime=False to admit only."
            )
        model = base.load(child_directory)
        # Second pass with the live child: the route bindings must match the
        # tables that were actually loaded, exactly as the stage-1 loader
        # re-authenticates after ``base.load``.
        package = authenticate_stage2_child_package(
            child_directory,
            label=admitted,
            optimizer_steps=optimizer_steps,
            backward_calls=backward_calls,
            child_model=model,
            expected_package_sha256=package.package_sha256,
        )
        try:
            runtime = bind_stage2_packaged_carrier_routes(
                model,
                package.routes,
                label=admitted,
                optimizer_steps=optimizer_steps,
                backward_calls=backward_calls,
                expected_target_rows=int(package.receipt["target_rows"]),
                expected_hidden_size=int(package.receipt["hidden_size"]),
            )
        except Lfm2CarrierBridgePublicationError as exc:
            raise Stage2RoutedDecodeAdmissionError(
                "Admitted stage-2 carrier routes failed live binding."
            ) from exc
        model = install_lfm2_semantic_step_generate(
            model,
            semantic_source_tokenizer=package.semantic_source_lexicon.tokenizer,
            target_tokenizer=target_tokenizer_loader(Path(child_directory)),
            runtime=runtime,
        )
    body = {
        "schema": STAGE2_ROUTED_DECODE_ADMISSION_SCHEMA,
        "status": "admitted",
        "stage": "stage2",
        "admitted_by_label": admitted.as_dict(),
        "checkpoint_identity": identity.as_dict(),
        "arm_id": str(manifest["arm_id"]),
        "rung_tokens": int(manifest["rung_tokens"]),
        "tokens_consumed": int(manifest["tokens_consumed"]),
        "optimizer_steps": optimizer_steps,
        "backward_calls": backward_calls,
        "route_provenance_mode": str(manifest["route_provenance_mode"]),
        "route_operation_audit": dict(manifest["route_operation_audit"]),
        "carrier_route_package_sha256": package.routes.package_sha256,
        "stage1_guard_untouched": True,
        "stage1_refusal_this_checkpoint_still_meets": (
            STAGE1_TRAINED_ROUTE_REFUSAL
            if bool(manifest["route_coefficients_trained"])
            else None
        ),
        "routed_decode_bound": model is not None,
        "semantic_step_generate_installed": callable(
            getattr(model, "semantic_step_generate", None)
        ),
    }
    record = {**body, "admission_sha256": canonical_sha256(body)}
    if model is not None:
        record["model"] = model
        record["package"] = package
    return record


__all__ = [
    "STAGE2_ADMITTED_TRAINING_AUDIT_KEYS",
    "STAGE2_FORBIDDEN_AUDIT_CREDIT_KEYS",
    "STAGE2_ROUTED_DECODE_ADMISSION_SCHEMA",
    "Stage2CheckpointIdentity",
    "Stage2RoutedDecodeAdmissionError",
    "admit_stage2_rung_checkpoint",
    "admitted_stage2_checkpoint_label",
    "authenticate_stage2_child_package",
    "authenticate_stage2_packaged_carrier_routes",
    "bind_stage2_packaged_carrier_routes",
]
