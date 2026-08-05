"""Authenticate and materialize one exact-4x moving-gauge V5 candidate.

The V5 fold stages six immutable two-tensor packs without mutating its source
child.  This module is the deliberately small publication bridge from one of
those packs to a fresh, standalone carrier child.  It never reopens the
original carrier publication, loads a parent, performs a forward pass, or
optimizes parameters.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .dynamic_counted_child_state import build_counted_child_state_plan
from .hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from .hf_lfm2_direct_local import (
    count_lfm2_direct_local_unique_parameter_elements,
    load_lfm2_direct_local_model,
    load_lfm2_direct_local_tokenizer,
)
from .hf_lfm2_moving_gauge_residual_reclosure import (
    fingerprint_lfm2_moving_gauge_tensor,
)
from .hf_tokenizer_bridge import (
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
)
from .lfm2_carrier_child_package import (
    CARRIER_CHILD_PACKAGE_RECEIPT,
    CURRENT_LFM2_CARRIER_DENSE_PARAMETER_ELEMENTS,
    CURRENT_LFM2_CARRIER_HIDDEN_SIZE,
    CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS,
    CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS,
    CURRENT_LFM2_CARRIER_TARGET_ROWS,
    CURRENT_LFM2_PARENT_PARAMETER_ELEMENTS,
    Lfm2CarrierChildCheckpointIO,
    load_authenticated_lfm2_carrier_child_package,
    seal_lfm2_carrier_child_package,
)


CANDIDATE_PACK_SCHEMA = (
    "neural-forge-lfm2-exact4x-moving-gauge-v5-candidate-pack-v1"
)
MATERIALIZATION_SCHEMA = (
    "neural-forge-lfm2-exact4x-moving-gauge-v5-candidate-materialization-v1"
)
V5_LIVE_SCHEMA = "neural-forge-lfm2-exact4x-moving-gauge-v5-live-v1"
MATERIALIZATION_RECEIPT = "candidate-materialization-receipt.json"
REPLACEMENT_TENSOR_FILE = "replacement-tensors.safetensors"
REPLACEMENT_TENSOR_NAMES = (
    "model.layers.0.conv.out_proj.weight",
    "model.layers.0.feed_forward.w2.weight",
)
CHART_MODES = frozenset({"E0", "E0R", "E0A"})
TARGET_ARMS = frozenset({"joint", "sequential_conditioned"})
UPSTREAM_RECEIPT_NAMES = frozenset(
    {
        "chart",
        "heldout_comparison",
        "lineage",
        "mlp_solution",
        "operator_solution",
        "target_plan",
    }
)
_CANDIDATE_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "request_sha256",
        "counted_child_state_plan_fingerprint",
        "chart_mode",
        "target_arm",
        "replacement_tensor_names",
        "replacement_tensor_count",
        "tensor_descriptors",
        "tensor_file",
        "upstream_receipt_sha256s",
        "child_state_before_sha256",
        "child_state_after_sha256",
        "source_child_state_preserved",
        "installed",
        "ready_for_atomic_apply",
        "contains_complete_child_checkpoint",
        "contains_only_two_replacement_tensors_and_receipt",
        "parent_runtime_sidecar",
        "quantization_credit",
        "pruning_credit",
        "parameter_sharing_credit",
        "training_steps",
        "numeric_metrics_are_diagnostic_only",
        "manual_semantic_review_status",
        "manual_semantic_review_required_downstream",
        "candidate_pack_sha256",
    }
)
_V5_PHASE8_EVIDENCE_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "packs",
        "candidate_pack_count",
        "replacement_tensor_names",
        "replacement_dtype",
        "installed",
        "ready_for_atomic_apply",
        "manual_semantic_review_status",
        "source_child_state_preserved",
        "source_child_identity_sha256",
        "source_child_state_dict_sha256",
        "source_route_state_sha256",
    }
)
class Lfm2Exact4xV5CandidateMaterializationError(RuntimeError):
    """A candidate, source child, or materialized child failed authentication."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            f"{name} must be a lowercase SHA-256."
        )
    return value


def _ordinary_directory(value: Path, name: str) -> Path:
    path = Path(value).expanduser().absolute()
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            f"{name} is missing."
        ) from exc
    if path.is_symlink() or not resolved.is_dir():
        raise Lfm2Exact4xV5CandidateMaterializationError(
            f"{name} must be one real local directory."
        )
    return resolved


def _ordinary_file(path: Path, name: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            f"{name} is missing."
        ) from exc
    if (
        path.is_symlink()
        or not resolved.is_file()
        or getattr(resolved.stat(), "st_nlink", 1) != 1
    ):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            f"{name} must be one ordinary, unlinked file."
        )
    return resolved


def _is_link_like(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(callable(is_junction) and is_junction())


def _validated_disjoint_output_root(
    value: Path,
    *,
    immutable_roots: Sequence[Path],
) -> Path:
    """Resolve a future output root without mutating or overlapping inputs."""

    if not isinstance(value, (str, os.PathLike)) or "://" in str(value):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Materialized-child output root must be one ordinary local path."
        )
    supplied = Path(value).expanduser().absolute()
    for ancestor in (supplied, *supplied.parents):
        if _is_link_like(ancestor):
            raise Lfm2Exact4xV5CandidateMaterializationError(
                "Materialized-child output root has a link-like ancestor."
            )
        if ancestor.exists() and not ancestor.is_dir():
            raise Lfm2Exact4xV5CandidateMaterializationError(
                "Materialized-child output root has a non-directory ancestor."
            )
    output = supplied.resolve(strict=False)
    for raw in immutable_roots:
        immutable = Path(raw).resolve(strict=True)
        if (
            output == immutable
            or output in immutable.parents
            or immutable in output.parents
        ):
            raise Lfm2Exact4xV5CandidateMaterializationError(
                "Materialized-child output root overlaps an immutable input."
            )
    return output


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _strict_canonical_json(path: Path, name: str) -> dict[str, Any]:
    source = _ordinary_file(path, name)
    raw = source.read_bytes()

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Lfm2Exact4xV5CandidateMaterializationError(
                    f"{name} repeats key {key!r}."
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            f"{name} contains non-finite JSON constant {value!r}."
        )

    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except Lfm2Exact4xV5CandidateMaterializationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            f"{name} is not strict ASCII JSON."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            f"{name} must contain one JSON object."
        )
    if raw != canonical_json_bytes(value) + b"\n":
        raise Lfm2Exact4xV5CandidateMaterializationError(
            f"{name} does not use the canonical encoding."
        )
    return value


def _canonical_relative_path(value: Any, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            f"{name} is not a canonical child-relative path."
        )
    path = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            f"{name} is not a canonical child-relative path."
        )
    return path


@dataclass(frozen=True, slots=True)
class AuthenticatedLfm2Exact4xV5CandidatePack:
    """One immutable, authenticated two-tensor candidate pack."""

    root: Path
    receipt: Mapping[str, Any]
    tensors: Mapping[str, Any]

    @property
    def candidate_pack_sha256(self) -> str:
        return str(self.receipt["candidate_pack_sha256"])


def load_authenticated_lfm2_exact4x_v5_candidate_pack(
    candidate_directory: Path,
    *,
    expected_candidate_pack_sha256: str,
    expected_request_sha256: str,
    expected_chart_mode: str,
    expected_target_arm: str,
    expected_counted_plan_fingerprint: str,
) -> AuthenticatedLfm2Exact4xV5CandidatePack:
    """Authenticate a V5 pack's complete two-file and two-tensor contract."""

    root = _ordinary_directory(candidate_directory, "candidate pack directory")
    expected_pack = _digest(
        expected_candidate_pack_sha256,
        "expected candidate pack SHA-256",
    )
    expected_request = _digest(expected_request_sha256, "expected request SHA-256")
    expected_plan = _digest(
        expected_counted_plan_fingerprint,
        "expected counted-plan fingerprint",
    )
    if expected_chart_mode not in CHART_MODES:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Expected chart mode is not a publishable V5 chart."
        )
    if expected_target_arm not in TARGET_ARMS:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Expected target arm is not a publishable V5 arm."
        )
    try:
        entries = tuple(root.iterdir())
    except OSError as exc:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Candidate pack could not be enumerated."
        ) from exc
    if {path.name for path in entries} != {
        "receipt.json",
        REPLACEMENT_TENSOR_FILE,
    } or any(
        path.is_symlink()
        or not path.is_file()
        or getattr(path.stat(), "st_nlink", 1) != 1
        for path in entries
    ):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Candidate pack must contain exactly two ordinary files."
        )
    receipt = _strict_canonical_json(root / "receipt.json", "candidate receipt")
    if set(receipt) != _CANDIDATE_RECEIPT_FIELDS:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Candidate receipt field inventory changed."
        )
    body = dict(receipt)
    seal = body.pop("candidate_pack_sha256", None)
    if (
        seal != canonical_sha256(body)
        or seal != expected_pack
        or root.name != expected_pack
    ):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Candidate receipt, directory name, or expected content address differs."
        )
    upstream = receipt.get("upstream_receipt_sha256s")
    descriptors = receipt.get("tensor_descriptors")
    tensor_file = receipt.get("tensor_file")
    flags_match = (
        receipt.get("schema") == CANDIDATE_PACK_SCHEMA
        and receipt.get("request_sha256") == expected_request
        and receipt.get("counted_child_state_plan_fingerprint") == expected_plan
        and receipt.get("chart_mode") == expected_chart_mode
        and receipt.get("target_arm") == expected_target_arm
        and receipt.get("replacement_tensor_names")
        == list(REPLACEMENT_TENSOR_NAMES)
        and receipt.get("replacement_tensor_count") == 2
        and receipt.get("child_state_before_sha256")
        == receipt.get("child_state_after_sha256")
        and receipt.get("source_child_state_preserved") is True
        and receipt.get("installed") is False
        and receipt.get("ready_for_atomic_apply") is True
        and receipt.get("contains_complete_child_checkpoint") is False
        and receipt.get("contains_only_two_replacement_tensors_and_receipt")
        is True
        and receipt.get("parent_runtime_sidecar") is False
        and receipt.get("quantization_credit") is False
        and receipt.get("pruning_credit") is False
        and receipt.get("parameter_sharing_credit") is False
        and receipt.get("training_steps") == 0
        and receipt.get("numeric_metrics_are_diagnostic_only") is True
        and receipt.get("manual_semantic_review_status") == "pending"
        and receipt.get("manual_semantic_review_required_downstream") is True
    )
    if not flags_match:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Candidate receipt policy or lineage contract changed."
        )
    _digest(receipt.get("child_state_before_sha256"), "candidate child state")
    if not isinstance(upstream, Mapping) or set(upstream) != UPSTREAM_RECEIPT_NAMES:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Candidate upstream receipt inventory is incomplete."
        )
    for name, value in upstream.items():
        _digest(value, f"candidate upstream receipt {name}")
    if not isinstance(descriptors, Mapping) or set(descriptors) != set(
        REPLACEMENT_TENSOR_NAMES
    ):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Candidate tensor descriptor inventory changed."
        )
    tensor_path = _ordinary_file(
        root / REPLACEMENT_TENSOR_FILE,
        "candidate replacement tensor file",
    )
    if (
        not isinstance(tensor_file, Mapping)
        or set(tensor_file) != {"filename", "sha256", "size_bytes"}
        or tensor_file.get("filename") != REPLACEMENT_TENSOR_FILE
        or tensor_file.get("sha256") != _file_sha256(tensor_path)
        or isinstance(tensor_file.get("size_bytes"), bool)
        or not isinstance(tensor_file.get("size_bytes"), int)
        or tensor_file.get("size_bytes") != tensor_path.stat().st_size
    ):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Candidate tensor file differs from its receipt."
        )
    _digest(tensor_file.get("sha256"), "candidate tensor file SHA-256")
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - production dependency
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Candidate authentication requires torch and safetensors."
        ) from exc
    expected_metadata = {
        "neural_forge": canonical_json_bytes(
            {
                "schema": CANDIDATE_PACK_SCHEMA,
                "request_sha256": expected_request,
                "chart_mode": expected_chart_mode,
                "target_arm": expected_target_arm,
                "installed": False,
            }
        ).decode("ascii")
    }
    loaded: dict[str, Any] = {}
    try:
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            if dict(handle.metadata() or {}) != expected_metadata or set(
                handle.keys()
            ) != set(REPLACEMENT_TENSOR_NAMES):
                raise Lfm2Exact4xV5CandidateMaterializationError(
                    "Candidate safetensor metadata or key inventory changed."
                )
            for name in REPLACEMENT_TENSOR_NAMES:
                value = handle.get_tensor(name)
                if (
                    value.layout != torch.strided
                    or value.device.type != "cpu"
                    or value.dtype != torch.float32
                    or value.ndim != 2
                    or value.is_complex()
                    or not bool(torch.isfinite(value).all().item())
                ):
                    raise Lfm2Exact4xV5CandidateMaterializationError(
                        f"Candidate tensor {name!r} is not one finite CPU FP32 matrix."
                    )
                tensor = value.detach().contiguous().clone()
                descriptor = descriptors[name]
                expected_descriptor = {
                    "shape": [int(size) for size in tensor.shape],
                    "dtype": "float32",
                    "numel": int(tensor.numel()),
                    "tensor_sha256": fingerprint_lfm2_moving_gauge_tensor(tensor),
                }
                if descriptor != expected_descriptor:
                    raise Lfm2Exact4xV5CandidateMaterializationError(
                        f"Candidate tensor {name!r} differs from its descriptor."
                    )
                loaded[name] = tensor
    except Lfm2Exact4xV5CandidateMaterializationError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Candidate safetensor could not be authenticated."
        ) from exc
    return AuthenticatedLfm2Exact4xV5CandidatePack(
        root=root,
        receipt=MappingProxyType(dict(receipt)),
        tensors=MappingProxyType(loaded),
    )


@dataclass(frozen=True, slots=True)
class AuthenticatedLfm2Exact4xV5MaterializationProvenance:
    """The sealed launch, completed run, and exact six-pack phase-8 bank."""

    launch_manifest: Mapping[str, Any]
    final_receipt: Mapping[str, Any]
    phase8_outcome: Mapping[str, Any]
    request: Any
    source_model: Any
    source_package: Any
    counted_plan: Any
    source_fingerprint: Any
    source_tokenizer_semantics_sha256: str
    source_topology_sha256: str
    candidates: Mapping[str, AuthenticatedLfm2Exact4xV5CandidatePack]
    selected_candidate: AuthenticatedLfm2Exact4xV5CandidatePack
    tokenizer_bridge_reference: Mapping[str, Any]
    output_root: Path


def _expected_candidate_upstream_receipts(
    outcomes: Mapping[str, Any],
    *,
    chart_mode: str,
    target_arm: str,
) -> Mapping[str, str]:
    """Derive all six candidate subreceipt identities from signed outcomes."""

    try:
        chart_hash = outcomes["freeze_fit_charts"].evidence[
            "chart_bundle_sha256s"
        ][chart_mode]
        baseline = outcomes["build_baseline_target_arms"].evidence["charts"][
            chart_mode
        ]
        conditioned = outcomes[
            "freeze_conditioned_targets_and_solutions"
        ].evidence["charts"][chart_mode]
        if target_arm == "joint":
            split = baseline["joint_split_receipt_sha256s"]
            operator_solution = split["operator"]
            mlp_solution = split["mlp"]
            target_plan = baseline["joint_target_sha256"]
        elif target_arm == "sequential_conditioned":
            operator_solution = conditioned["operator_solution_sha256"]
            mlp_solution = conditioned["mlp_solution_sha256"]
            target_plan = conditioned["target_sha256"]
        else:  # pragma: no cover - caller validation rejects this first.
            raise KeyError(target_arm)
        expected = {
            "chart": chart_hash,
            "heldout_comparison": outcomes[
                "compare_frozen_candidates_on_heldout"
            ].artifact_sha256,
            "lineage": outcomes["authenticate_lineage"].artifact_sha256,
            "mlp_solution": mlp_solution,
            "operator_solution": operator_solution,
            "target_plan": target_plan,
        }
    except (AttributeError, KeyError, TypeError) as exc:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Authenticated V5 outcomes lack candidate subreceipt lineage."
        ) from exc
    if set(expected) != UPSTREAM_RECEIPT_NAMES:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Candidate subreceipt lineage inventory changed."
        )
    for name, value in expected.items():
        _digest(value, f"expected candidate upstream {name}")
    return MappingProxyType(expected)


def _authenticated_phase8_candidate_bank_evidence(
    phase8: Mapping[str, Any],
    *,
    source_fingerprint: Any,
) -> Mapping[str, Any]:
    """Validate materialization-specific claims in an authenticated outcome."""

    evidence = phase8.get("evidence")
    if (
        phase8.get("child_forward_count") != 0
        or phase8.get("parent_forward_count") != 0
        or phase8.get("heldout_rows_read") is not False
        or phase8.get("temporary_child_mutation_performed") is not False
        or phase8.get("child_state_restored") is not False
        or phase8.get("child_state_before_sha256") is not None
        or phase8.get("child_state_after_sha256") is not None
        or phase8.get("installed") is not False
        or phase8.get("ready_for_atomic_apply") is not True
        or phase8.get("authenticated_checkpoint_reuse") is not False
        or not isinstance(evidence, Mapping)
        or set(evidence) != _V5_PHASE8_EVIDENCE_FIELDS
        or evidence.get("schema") != V5_LIVE_SCHEMA
        or evidence.get("kind") != "six_atomic_fp32_candidate_packs"
        or evidence.get("candidate_pack_count") != 6
        or evidence.get("replacement_tensor_names")
        != list(REPLACEMENT_TENSOR_NAMES)
        or evidence.get("replacement_dtype") != "float32"
        or evidence.get("installed") is not False
        or evidence.get("ready_for_atomic_apply") is not True
        or evidence.get("manual_semantic_review_status") != "pending"
        or evidence.get("source_child_state_preserved") is not True
        or evidence.get("source_child_identity_sha256")
        != source_fingerprint.identity_sha256
        or evidence.get("source_child_state_dict_sha256")
        != source_fingerprint.state_dict_sha256
    ):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Phase-8 outcome does not match the completed V5 run."
        )
    # This is producer-HMAC-attested evidence, not a fresh materializer
    # observation. Recomputing it requires repeating the V3 implementation-
    # identity-bound runtime relocation. The materializer instead freshly
    # authenticates the packaged route artifact and preserves this distinct
    # owned-tensor-state hash as signed upstream provenance.
    _digest(evidence.get("source_route_state_sha256"), "phase-8 route state")
    return MappingProxyType(dict(evidence))


def _authenticate_strict_bridge_reference_for_materialization(
    *,
    launch: Any,
    manifest: Mapping[str, Any],
    source_package: Any,
    source_tokenizer_semantics_sha256: str,
) -> Mapping[str, Any]:
    """Retain the producer's live-bridge/package cross-authentication result."""

    provenance = manifest["provenance_sha256s"]
    try:
        bridge = load_strict_tokenizer_bridge_artifact(
            Path(manifest["tokenizer_bridge_root"]),
            expected_artifact_sha256=provenance["bridge_artifact"],
            packaged=False,
        )
        reference = launch.authenticate_packaged_tokenizer_bridge_reference(
            source_package,
            bridge,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Strict tokenizer-bridge provenance authentication failed."
        ) from exc
    if (
        not isinstance(reference, Mapping)
        or reference.get("target_tokenizer_semantics_sha256")
        != source_tokenizer_semantics_sha256
        or reference.get("bridge_artifact_sha256")
        != provenance["bridge_artifact"]
        or reference.get("bridge_profile_sha256")
        != provenance["bridge_profile"]
    ):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Strict tokenizer bridge differs from the child or completed launch."
        )
    _digest(
        reference.get("target_tokenizer_package_fingerprint"),
        "strict tokenizer package fingerprint",
    )
    return reference


def load_authenticated_lfm2_exact4x_v5_materialization_provenance(
    *,
    launch_manifest_json: Path,
    authority_key_path: Path,
    chart_mode: str,
    target_arm: str,
) -> AuthenticatedLfm2Exact4xV5MaterializationProvenance:
    """Authenticate one completed HMAC-bound launch and its six-pack bank."""

    try:
        from experiments import run_lfm2_exact4x_moving_gauge_v5 as v5
        from experiments import run_lfm2_exact4x_moving_gauge_v5_r12_live as launch
    except ImportError as exc:  # pragma: no cover - deployment packaging error
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "The exact V5 launch/request authenticators are unavailable."
        ) from exc
    if chart_mode not in CHART_MODES or target_arm not in TARGET_ARMS:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Candidate selection must name one publishable V5 chart and arm."
        )
    manifest_path = _ordinary_file(
        Path(launch_manifest_json).expanduser().absolute(),
        "R12 V5 launch manifest",
    )
    try:
        manifest = launch.read_r12_v5_launch_manifest(manifest_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "R12 V5 launch manifest authentication failed."
        ) from exc
    lineage = manifest["lineage_sha256s"]
    provenance = manifest["provenance_sha256s"]
    source_model, source_package, counted_plan, source_tokenizer, source_topology = (
        _authenticate_source_child(
            Path(manifest["local_paths"]["published_child"]),
            expected_package_sha256=lineage["r12_child_package"],
        )
    )
    source_fingerprint = fingerprint_lfm2_parent(source_model)
    if (
        lineage["r12_child_package"] != source_package.package_sha256
        or provenance["child_package"] != source_package.package_sha256
        or lineage["r12_child_identity"] != source_fingerprint.identity_sha256
        or provenance["child_identity"] != source_fingerprint.identity_sha256
        or lineage["r12_state_dict"] != source_fingerprint.state_dict_sha256
        or provenance["child_state_dict"] != source_fingerprint.state_dict_sha256
        or lineage["counted_child_state_plan"] != counted_plan.fingerprint
        or lineage["tokenizer_bridge"] != provenance["bridge_artifact"]
        or provenance["route_package"] != source_package.routes.package_sha256
        or Path(manifest["local_paths"]["published_child"]).resolve(strict=True)
        != source_package.root
    ):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Launch lineage differs from the authenticated R12 child."
        )
    request = v5.Lfm2Exact4xMovingGaugeV5Request(
        local_paths=manifest["local_paths"],
        lineage_sha256s=lineage,
        counted_child_state_plan=counted_plan,
        replay_devices=tuple(manifest["replay"]["devices"]),
        recapture_checkpoint_batch_count=manifest["replay"][
            "recapture_checkpoint_batch_count"
        ],
    )
    try:
        receipt = launch.read_r12_v5_launch_receipt(
            manifest,
            authority_key_path=authority_key_path,
            request=request,
        )
        outcomes = v5.authenticate_v5_phase_outcomes(
            request,
            receipt["phase_outcomes"],
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "HMAC-bound R12 V5 receipt authentication failed."
        ) from exc
    if (
        receipt.get("r12_child_package_sha256") != source_package.package_sha256
        or receipt.get("counted_child_state_plan_sha256")
        != counted_plan.fingerprint
        or receipt.get("v5_request_sha256") != request.request_sha256
    ):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Authenticated receipt differs from the reconstructed source child."
        )
    bridge_reference = _authenticate_strict_bridge_reference_for_materialization(
        launch=launch,
        manifest=manifest,
        source_package=source_package,
        source_tokenizer_semantics_sha256=source_tokenizer,
    )
    phase8 = outcomes["stage_atomic_candidate_packs"].as_dict()
    evidence = _authenticated_phase8_candidate_bank_evidence(
        phase8,
        source_fingerprint=source_fingerprint,
    )
    packs = evidence.get("packs")
    expected_pack_keys = {
        f"{chart}:{arm}" for chart in CHART_MODES for arm in TARGET_ARMS
    }
    candidate_root = Path(manifest["local_paths"]["output_root"]).resolve(strict=True)
    if not isinstance(packs, Mapping) or set(packs) != expected_pack_keys:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Phase-8 does not contain exactly the six publishable candidates."
        )
    authenticated: dict[str, AuthenticatedLfm2Exact4xV5CandidatePack] = {}
    for key in sorted(expected_pack_keys):
        row = packs[key]
        chart, arm = key.split(":", 1)
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "path",
                "candidate_pack_sha256",
                "chart_mode",
                "target_arm",
            }
            or row.get("chart_mode") != chart
            or row.get("target_arm") != arm
        ):
            raise Lfm2Exact4xV5CandidateMaterializationError(
                f"Phase-8 candidate row {key!r} is malformed."
            )
        pack_path = _ordinary_directory(Path(row["path"]), f"phase-8 pack {key}")
        if pack_path.parent != candidate_root:
            raise Lfm2Exact4xV5CandidateMaterializationError(
                f"Phase-8 candidate {key!r} escaped its sealed output root."
            )
        candidate = load_authenticated_lfm2_exact4x_v5_candidate_pack(
            pack_path,
            expected_candidate_pack_sha256=row["candidate_pack_sha256"],
            expected_request_sha256=request.request_sha256,
            expected_chart_mode=chart,
            expected_target_arm=arm,
            expected_counted_plan_fingerprint=counted_plan.fingerprint,
        )
        upstream = candidate.receipt["upstream_receipt_sha256s"]
        expected_upstream = _expected_candidate_upstream_receipts(
            outcomes,
            chart_mode=chart,
            target_arm=arm,
        )
        if (
            candidate.receipt["child_state_before_sha256"]
            != source_fingerprint.state_dict_sha256
            or dict(upstream) != dict(expected_upstream)
        ):
            raise Lfm2Exact4xV5CandidateMaterializationError(
                f"Phase-8 candidate {key!r} is not bound to this V5 run."
            )
        authenticated[key] = candidate
    selected_key = f"{chart_mode}:{target_arm}"
    selected = authenticated.get(selected_key)
    if selected is None:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Selected candidate is not a member of the authenticated phase-8 bank."
        )
    output_root = _validated_disjoint_output_root(
        Path(manifest["materialized_child_output_root"]),
        immutable_roots=(
            source_package.root,
            *(candidate.root for candidate in authenticated.values()),
        ),
    )
    return AuthenticatedLfm2Exact4xV5MaterializationProvenance(
        launch_manifest=MappingProxyType(dict(manifest)),
        final_receipt=MappingProxyType(dict(receipt)),
        phase8_outcome=MappingProxyType(dict(phase8)),
        request=request,
        source_model=source_model,
        source_package=source_package,
        counted_plan=counted_plan,
        source_fingerprint=source_fingerprint,
        source_tokenizer_semantics_sha256=source_tokenizer,
        source_topology_sha256=source_topology,
        candidates=MappingProxyType(authenticated),
        selected_candidate=selected,
        tokenizer_bridge_reference=bridge_reference,
        output_root=output_root,
    )


@dataclass(frozen=True, slots=True)
class _DirectLocalReadOnlyIO:
    loader_id: str = "neural-forge.lfm2.v5-materializer-direct-local-v1"
    resource_fingerprint: str = "exact4x-v5-materialized-child-only"

    def load(self, source: Path) -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - production dependency
            raise Lfm2Exact4xV5CandidateMaterializationError(
                "Materialization requires torch."
            ) from exc
        return load_lfm2_direct_local_model(
            source,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def _state_topology_sha256(model: Any) -> str:
    try:
        state = model.state_dict()
        named_parameters = tuple(model.named_parameters(remove_duplicate=False))
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Child model does not expose a stable state topology."
        ) from exc
    storage_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    rows = []
    for name, value in sorted(state.items()):
        rows.append(
            {
                "name": str(name),
                "shape": [int(size) for size in value.shape],
                "stride": [int(size) for size in value.stride()],
                "dtype": str(value.dtype),
            }
        )
    for name, value in named_parameters:
        if int(value.numel()) <= 0:
            continue
        storage = value.untyped_storage()
        key = (int(storage.data_ptr()), int(storage.nbytes()))
        storage_groups.setdefault(key, []).append(
            {
                "name": str(name),
                "shape": [int(size) for size in value.shape],
                "stride": [int(size) for size in value.stride()],
                "storage_offset": int(value.storage_offset()),
            }
        )
    aliases = sorted(
        (
            sorted(members, key=lambda row: row["name"])
            for members in storage_groups.values()
            if len(members) > 1
        ),
        key=lambda members: [row["name"] for row in members],
    )
    return canonical_sha256({"state": rows, "parameter_alias_groups": aliases})


def _authenticate_source_child(
    source_child_directory: Path,
    *,
    expected_package_sha256: str,
) -> tuple[Any, Any, Any, str, str]:
    source = _ordinary_directory(source_child_directory, "source carrier child")
    expected = _digest(
        expected_package_sha256,
        "expected source child package SHA-256",
    )
    package_without_model = load_authenticated_lfm2_carrier_child_package(
        source,
        expected_package_sha256=expected,
    )
    model = _DirectLocalReadOnlyIO().load(source)
    package = load_authenticated_lfm2_carrier_child_package(
        source,
        child_model=model,
        expected_package_sha256=package_without_model.package_sha256,
    )
    dense_count = count_lfm2_direct_local_unique_parameter_elements(model)
    if (
        dense_count != CURRENT_LFM2_CARRIER_DENSE_PARAMETER_ELEMENTS
        or int(package.receipt["target_rows"])
        != CURRENT_LFM2_CARRIER_TARGET_ROWS
        or int(package.receipt["hidden_size"])
        != CURRENT_LFM2_CARRIER_HIDDEN_SIZE
        or len(package.routes.allocations) != 2
        or any(
            allocation.parameter_elements
            != CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS
            for allocation in package.routes.allocations
        )
    ):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Source child does not have the sealed exact-4x dense/route geometry."
        )
    plan = build_counted_child_state_plan(
        parent_unique_parameters=CURRENT_LFM2_PARENT_PARAMETER_ELEMENTS,
        target_total_parameter_elements=CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS,
        continuous_allocations=package.routes.allocations,
    )
    if (
        plan.dense_model_parameter_elements
        != CURRENT_LFM2_CARRIER_DENSE_PARAMETER_ELEMENTS
        or plan.target_total_parameter_elements
        != CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS
    ):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Source child counted-state plan is not the exact-4x target."
        )
    tokenizer_sha256 = fingerprint_hf_tokenizer_semantics(
        load_lfm2_direct_local_tokenizer(source)
    )
    return model, package, plan, tokenizer_sha256, _state_topology_sha256(model)


def _copy_authenticated_child_tree(source: Any, destination: Path) -> None:
    records = source.receipt.get("child_file_records")
    directories = source.receipt.get("child_directories")
    if not isinstance(records, list) or not isinstance(directories, list):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Source child has no sealed copy inventory."
        )
    seen_directories: set[str] = set()
    for raw in sorted(directories, key=lambda value: (str(value).count("/"), str(value))):
        relative = _canonical_relative_path(raw, "sealed child directory")
        text = relative.as_posix()
        if text in seen_directories:
            raise Lfm2Exact4xV5CandidateMaterializationError(
                "Source child directory inventory repeats a path."
            )
        seen_directories.add(text)
        (destination / Path(*relative.parts)).mkdir()
    seen_files: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise Lfm2Exact4xV5CandidateMaterializationError(
                "Source child file record is malformed."
            )
        relative = _canonical_relative_path(
            record.get("relative_path"),
            "sealed child file",
        )
        text = relative.as_posix()
        if text == CARRIER_CHILD_PACKAGE_RECEIPT or text in seen_files:
            raise Lfm2Exact4xV5CandidateMaterializationError(
                "Source copy inventory contains a stale receipt or duplicate file."
            )
        seen_files.add(text)
        source_file = _ordinary_file(
            source.root / Path(*relative.parts),
            f"source child file {text!r}",
        )
        destination_file = destination / Path(*relative.parts)
        if not destination_file.parent.is_dir() or destination_file.exists():
            raise Lfm2Exact4xV5CandidateMaterializationError(
                f"Staged destination for {text!r} is unsafe."
            )
        shutil.copyfile(source_file, destination_file, follow_symlinks=False)
        if (
            destination_file.is_symlink()
            or not destination_file.is_file()
            or getattr(destination_file.stat(), "st_nlink", 1) != 1
            or destination_file.stat().st_size != record.get("size_bytes")
            or _file_sha256(destination_file) != record.get("sha256")
        ):
            raise Lfm2Exact4xV5CandidateMaterializationError(
                f"Copied child file {text!r} differs from its sealed source."
            )


def _replace_candidate_tensors(
    child_directory: Path,
    candidate: AuthenticatedLfm2Exact4xV5CandidatePack,
) -> None:
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - production dependency
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Candidate installation requires torch and safetensors."
        ) from exc
    model_path = _ordinary_file(
        child_directory / "model.safetensors",
        "unsharded child model safetensor",
    )
    if (child_directory / "model.safetensors.index.json").exists() or tuple(
        child_directory.glob("model-*.safetensors")
    ):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "V5 materialization currently requires one root model.safetensors."
        )
    stored: dict[str, Any] = {}
    try:
        with safe_open(str(model_path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            names = tuple(handle.keys())
            if not set(REPLACEMENT_TENSOR_NAMES).issubset(names):
                raise Lfm2Exact4xV5CandidateMaterializationError(
                    "Source checkpoint lacks one of the two replacement tensors."
                )
            for name in names:
                stored[name] = handle.get_tensor(name).detach().contiguous().clone()
    except Lfm2Exact4xV5CandidateMaterializationError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Source model safetensor could not be read."
        ) from exc
    original = {name: value for name, value in stored.items()}
    for name in REPLACEMENT_TENSOR_NAMES:
        replacement = candidate.tensors[name]
        if tuple(stored[name].shape) != tuple(replacement.shape):
            raise Lfm2Exact4xV5CandidateMaterializationError(
                f"Candidate tensor {name!r} has the wrong checkpoint shape."
            )
        stored[name] = replacement.detach().to(device="cpu").contiguous().clone()
    temporary = child_directory / f".model.{uuid.uuid4().hex}.safetensors.tmp"
    try:
        save_file(stored, str(temporary), metadata=metadata or None)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, model_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    try:
        with safe_open(str(model_path), framework="pt", device="cpu") as handle:
            if dict(handle.metadata() or {}) != metadata or set(handle.keys()) != set(
                original
            ):
                raise Lfm2Exact4xV5CandidateMaterializationError(
                    "Materialized model safetensor changed metadata or key inventory."
                )
            for name in original:
                observed = handle.get_tensor(name)
                expected = (
                    candidate.tensors[name]
                    if name in REPLACEMENT_TENSOR_NAMES
                    else original[name]
                )
                if (
                    observed.dtype != expected.dtype
                    or tuple(observed.shape) != tuple(expected.shape)
                    or fingerprint_lfm2_moving_gauge_tensor(observed)
                    != fingerprint_lfm2_moving_gauge_tensor(expected)
                ):
                    raise Lfm2Exact4xV5CandidateMaterializationError(
                        f"Stored model tensor {name!r} differs after installation."
                    )
    except Lfm2Exact4xV5CandidateMaterializationError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Materialized model safetensor could not be reauthenticated."
        ) from exc


def _compare_fresh_child_state(
    source_model: Any,
    materialized_model: Any,
    candidate: AuthenticatedLfm2Exact4xV5CandidatePack,
) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production dependency
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "State verification requires torch."
        ) from exc
    source_state = source_model.state_dict()
    result_state = materialized_model.state_dict()
    if set(source_state) != set(result_state):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Materialization changed the child state-key set."
        )
    for name in sorted(source_state):
        source = source_state[name].detach().to(device="cpu")
        result = result_state[name].detach().to(device="cpu")
        if tuple(source.shape) != tuple(result.shape) or source.dtype != result.dtype:
            raise Lfm2Exact4xV5CandidateMaterializationError(
                f"Materialization changed state topology at {name!r}."
            )
        expected = candidate.tensors.get(name)
        if expected is not None:
            if (
                result.dtype != torch.float32
                or tuple(result.shape) != tuple(expected.shape)
                or fingerprint_lfm2_moving_gauge_tensor(result)
                != fingerprint_lfm2_moving_gauge_tensor(expected)
            ):
                raise Lfm2Exact4xV5CandidateMaterializationError(
                    f"Candidate tensor {name!r} was not installed bit-for-bit."
                )
        elif fingerprint_lfm2_moving_gauge_tensor(
            source
        ) != fingerprint_lfm2_moving_gauge_tensor(result):
            raise Lfm2Exact4xV5CandidateMaterializationError(
                f"Non-candidate state tensor {name!r} changed."
            )


def _write_materialization_receipt(
    child_directory: Path,
    body: Mapping[str, Any],
) -> Mapping[str, Any]:
    destination = child_directory / MATERIALIZATION_RECEIPT
    if destination.exists() or destination.is_symlink():
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Source child already contains a candidate materialization receipt."
        )
    receipt = {
        **dict(body),
        "candidate_materialization_sha256": canonical_sha256(dict(body)),
    }
    payload = canonical_json_bytes(receipt) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".candidate-materialization.",
        suffix=".tmp",
        dir=child_directory,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return MappingProxyType(receipt)


def _load_materialization_receipt(
    child_directory: Path,
    *,
    expected_materialization_sha256: str | None = None,
) -> Mapping[str, Any]:
    receipt = _strict_canonical_json(
        child_directory / MATERIALIZATION_RECEIPT,
        "candidate materialization receipt",
    )
    body = dict(receipt)
    seal = body.pop("candidate_materialization_sha256", None)
    if (
        receipt.get("schema") != MATERIALIZATION_SCHEMA
        or receipt.get("status") != "complete-pending-manual-semantic-review"
        or seal != canonical_sha256(body)
        or (
            expected_materialization_sha256 is not None
            and seal
            != _digest(
                expected_materialization_sha256,
                "expected materialization SHA-256",
            )
        )
    ):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Candidate materialization receipt or self-seal is invalid."
        )
    return MappingProxyType(receipt)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_file():
            # Windows' CRT rejects ``fsync`` on a read-only descriptor even
            # though the file is ordinary and locally owned.  The staging
            # tree is private and writable, so use a non-truncating writable
            # descriptor on every platform.
            with path.open("r+b") as handle:
                os.fsync(handle.fileno())
    if os.name != "nt":
        for path in sorted(
            (value for value in root.rglob("*") if value.is_dir()),
            key=lambda value: value.as_posix().count("/"),
            reverse=True,
        ):
            _fsync_directory(path)
        _fsync_directory(root)


def _rename_noreplace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "This POSIX runtime lacks atomic rename-without-replacement support."
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def _validate_existing_destination(
    destination: Path,
    *,
    expected_package_sha256: str,
    expected_source_package_sha256: str,
    expected_candidate_pack_sha256: str,
    expected_materialization_sha256: str,
    expected_launch_manifest_sha256: str,
    expected_launch_receipt_sha256: str,
    expected_launch_receipt_hmac_sha256: str,
    expected_phase8_artifact_sha256: str,
) -> Mapping[str, Any]:
    package = load_authenticated_lfm2_carrier_child_package(
        destination,
        expected_package_sha256=expected_package_sha256,
    )
    receipt = _load_materialization_receipt(
        destination,
        expected_materialization_sha256=expected_materialization_sha256,
    )
    if (
        receipt.get("source_child_package_sha256")
        != expected_source_package_sha256
        or receipt.get("candidate_pack_sha256")
        != expected_candidate_pack_sha256
        or receipt.get("v5_launch_manifest_sha256")
        != expected_launch_manifest_sha256
        or receipt.get("v5_launch_receipt_sha256")
        != expected_launch_receipt_sha256
        or receipt.get("v5_launch_receipt_hmac_sha256")
        != expected_launch_receipt_hmac_sha256
        or receipt.get("v5_phase8_artifact_sha256")
        != expected_phase8_artifact_sha256
        or receipt.get("installed") is not True
        or receipt.get("publication_authorized") is not False
        or receipt.get("accepted_artifact") is not False
    ):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Existing destination belongs to another materialization."
        )
    loader = Lfm2CarrierChildCheckpointIO.from_child_directory(
        base=_DirectLocalReadOnlyIO(),
        child_directory=destination,
    )
    model = loader.load(destination)
    load_authenticated_lfm2_carrier_child_package(
        destination,
        child_model=model,
        expected_package_sha256=package.package_sha256,
    )
    return receipt


def materialize_lfm2_exact4x_v5_candidate(
    *,
    launch_manifest_json: Path,
    authority_key_path: Path,
    chart_mode: str,
    target_arm: str,
) -> Mapping[str, Any]:
    """Install one HMAC-authenticated V5 bank member into a standalone child."""

    authenticated = load_authenticated_lfm2_exact4x_v5_materialization_provenance(
        launch_manifest_json=launch_manifest_json,
        authority_key_path=authority_key_path,
        chart_mode=chart_mode,
        target_arm=target_arm,
    )
    source_model = authenticated.source_model
    source_package = authenticated.source_package
    plan = authenticated.counted_plan
    source_tokenizer_sha256 = authenticated.source_tokenizer_semantics_sha256
    source_topology = authenticated.source_topology_sha256
    source_fingerprint = authenticated.source_fingerprint
    candidate = authenticated.selected_candidate
    if (
        candidate.receipt["child_state_before_sha256"]
        != source_fingerprint.state_dict_sha256
    ):
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Candidate was staged from a different source child state."
        )
    output = _validated_disjoint_output_root(
        authenticated.output_root,
        immutable_roots=(
            source_package.root,
            *(value.root for value in authenticated.candidates.values()),
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or not output.is_dir():
        raise Lfm2Exact4xV5CandidateMaterializationError(
            "Output root must be one real local directory."
        )
    staging = output / f".v5-materialization-{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    lock_path: Path | None = None
    lock_owned = False
    try:
        _copy_authenticated_child_tree(source_package, staging)
        _replace_candidate_tensors(staging, candidate)
        materialized_model = _DirectLocalReadOnlyIO().load(staging)
        _compare_fresh_child_state(source_model, materialized_model, candidate)
        result_fingerprint = fingerprint_lfm2_parent(materialized_model)
        result_topology = _state_topology_sha256(materialized_model)
        result_dense_count = count_lfm2_direct_local_unique_parameter_elements(
            materialized_model
        )
        result_tokenizer_sha256 = fingerprint_hf_tokenizer_semantics(
            load_lfm2_direct_local_tokenizer(staging)
        )
        if (
            result_fingerprint.config_sha256 != source_fingerprint.config_sha256
            or result_fingerprint.model_class != source_fingerprint.model_class
            or result_fingerprint.state_tensor_count
            != source_fingerprint.state_tensor_count
            or result_fingerprint.runtime_buffer_sha256
            != source_fingerprint.runtime_buffer_sha256
            or result_fingerprint.runtime_buffer_count
            != source_fingerprint.runtime_buffer_count
            or result_topology != source_topology
            or result_dense_count != CURRENT_LFM2_CARRIER_DENSE_PARAMETER_ELEMENTS
            or result_tokenizer_sha256 != source_tokenizer_sha256
        ):
            raise Lfm2Exact4xV5CandidateMaterializationError(
                "Candidate installation changed topology, config, count, or tokenizer."
            )
        body = {
            "schema": MATERIALIZATION_SCHEMA,
            "status": "complete-pending-manual-semantic-review",
            "v5_launch_manifest_sha256": authenticated.launch_manifest[
                "launch_manifest_sha256"
            ],
            "v5_launch_receipt_sha256": authenticated.final_receipt[
                "receipt_sha256"
            ],
            "v5_launch_receipt_hmac_sha256": authenticated.final_receipt[
                "receipt_hmac_sha256"
            ],
            "v5_launch_receipt_hmac_key_id": authenticated.final_receipt[
                "receipt_hmac_key_id"
            ],
            "v5_phase8_artifact_sha256": authenticated.phase8_outcome[
                "artifact_sha256"
            ],
            "source_child_package_sha256": source_package.package_sha256,
            "source_child_state_dict_sha256": source_fingerprint.state_dict_sha256,
            "source_child_identity_sha256": source_fingerprint.identity_sha256,
            "child_model_class": source_fingerprint.model_class,
            "source_state_tensor_count": source_fingerprint.state_tensor_count,
            "result_state_tensor_count": result_fingerprint.state_tensor_count,
            "source_runtime_buffer_sha256": (
                source_fingerprint.runtime_buffer_sha256
            ),
            "result_runtime_buffer_sha256": (
                result_fingerprint.runtime_buffer_sha256
            ),
            "source_runtime_buffer_count": source_fingerprint.runtime_buffer_count,
            "result_runtime_buffer_count": result_fingerprint.runtime_buffer_count,
            "candidate_pack_sha256": candidate.candidate_pack_sha256,
            "request_sha256": candidate.receipt["request_sha256"],
            "chart_mode": candidate.receipt["chart_mode"],
            "target_arm": candidate.receipt["target_arm"],
            "counted_child_state_plan_fingerprint": plan.fingerprint,
            "upstream_receipt_sha256s": dict(
                candidate.receipt["upstream_receipt_sha256s"]
            ),
            "replacement_tensor_names": list(REPLACEMENT_TENSOR_NAMES),
            "replacement_tensor_descriptors": dict(
                candidate.receipt["tensor_descriptors"]
            ),
            "result_child_state_dict_sha256": result_fingerprint.state_dict_sha256,
            "result_child_identity_sha256": result_fingerprint.identity_sha256,
            "source_config_sha256": source_fingerprint.config_sha256,
            "result_config_sha256": result_fingerprint.config_sha256,
            "source_topology_sha256": source_topology,
            "result_topology_sha256": result_topology,
            "carrier_route_package_sha256": source_package.routes.package_sha256,
            "v5_source_route_state_sha256": authenticated.phase8_outcome[
                "evidence"
            ]["source_route_state_sha256"],
            "v5_source_route_state_authentication": "producer-hmac-attested",
            "v5_source_route_state_freshly_recomputed": False,
            "carrier_route_package_freshly_reauthenticated": True,
            "source_bridge_artifact_sha256": source_package.receipt[
                "source_bridge_artifact_sha256"
            ],
            "route_allocations": [
                value.as_dict() for value in source_package.routes.allocations
            ],
            "input_table_binding": dict(
                source_package.routes.receipt["input_table_binding"]
            ),
            "output_table_binding": dict(
                source_package.routes.receipt["output_table_binding"]
            ),
            "semantic_source_lexicon_package_sha256": (
                source_package.semantic_source_lexicon.package_sha256
            ),
            "semantic_source_tokenizer_sha256": source_package.receipt[
                "semantic_source_tokenizer_sha256"
            ],
            "target_tokenizer_semantics_sha256": source_tokenizer_sha256,
            "strict_tokenizer_bridge_artifact_sha256": (
                authenticated.tokenizer_bridge_reference[
                    "bridge_artifact_sha256"
                ]
            ),
            "strict_tokenizer_bridge_profile_sha256": (
                authenticated.tokenizer_bridge_reference[
                    "bridge_profile_sha256"
                ]
            ),
            "strict_tokenizer_package_fingerprint": (
                authenticated.tokenizer_bridge_reference[
                    "target_tokenizer_package_fingerprint"
                ]
            ),
            "dense_model_parameter_elements": result_dense_count,
            "continuous_parameter_elements": plan.continuous_parameter_elements,
            "total_child_parameter_elements": plan.target_total_parameter_elements,
            "installed": True,
            "exactly_two_tensors_replaced": True,
            "noncandidate_state_bit_identical": True,
            "config_topology_and_tokenizers_unchanged": True,
            "carrier_routes_and_lexicon_unchanged": True,
            "standalone_child_local_runtime": True,
            "manual_semantic_review_status": "pending",
            "publication_authorized": False,
            "accepted_artifact": False,
            "parent_model_load_count": 0,
            "parent_forward_count": 0,
            "child_forward_count": 0,
            "child_generation_count": 0,
            "training_steps": 0,
            "gradient_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "quantization_calls": 0,
            "pruning_calls": 0,
            "parameter_sharing_credit": False,
            "runtime_sidecar_parameter_elements": 0,
        }
        materialization = _write_materialization_receipt(staging, body)
        sealed = seal_lfm2_carrier_child_package(
            staging,
            child_model=materialized_model,
            expected_source_artifact_sha256=str(
                source_package.receipt["source_bridge_artifact_sha256"]
            ),
            expected_tokenizer_semantics_sha256=str(
                source_package.receipt["semantic_source_tokenizer_sha256"]
            ),
        )
        if (
            sealed.routes.package_sha256 != source_package.routes.package_sha256
            or sealed.routes.allocations != source_package.routes.allocations
            or dict(sealed.routes.receipt) != dict(source_package.routes.receipt)
            or sealed.semantic_source_lexicon.package_sha256
            != source_package.semantic_source_lexicon.package_sha256
            or result_dense_count + sum(
                value.parameter_elements for value in sealed.routes.allocations
            )
            != CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS
        ):
            raise Lfm2Exact4xV5CandidateMaterializationError(
                "Resealing changed carrier routes, lexicon, or exact count."
            )
        del materialized_model
        gc.collect()
        review_loader = Lfm2CarrierChildCheckpointIO.from_child_directory(
            base=_DirectLocalReadOnlyIO(),
            child_directory=staging,
        )
        review_model = review_loader.load(staging)
        reviewed = load_authenticated_lfm2_carrier_child_package(
            staging,
            child_model=review_model,
            expected_package_sha256=sealed.package_sha256,
        )
        if reviewed.package_sha256 != sealed.package_sha256:
            raise Lfm2Exact4xV5CandidateMaterializationError(
                "Fresh child-local review loader changed the sealed package."
            )
        _load_materialization_receipt(
            staging,
            expected_materialization_sha256=str(
                materialization["candidate_materialization_sha256"]
            ),
        )
        load_authenticated_lfm2_carrier_child_package(
            source_package.root,
            expected_package_sha256=source_package.package_sha256,
        )
        load_authenticated_lfm2_exact4x_v5_candidate_pack(
            candidate.root,
            expected_candidate_pack_sha256=candidate.candidate_pack_sha256,
            expected_request_sha256=str(candidate.receipt["request_sha256"]),
            expected_chart_mode=str(candidate.receipt["chart_mode"]),
            expected_target_arm=str(candidate.receipt["target_arm"]),
            expected_counted_plan_fingerprint=plan.fingerprint,
        )
        _fsync_tree(staging)
        destination = output / sealed.package_sha256
        lock_path = output / f".{sealed.package_sha256}.publish.lock"
        lock_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        lock_owned = True
        os.close(lock_descriptor)
        if destination.exists() or destination.is_symlink():
            existing = _validate_existing_destination(
                destination,
                expected_package_sha256=sealed.package_sha256,
                expected_source_package_sha256=source_package.package_sha256,
                expected_candidate_pack_sha256=candidate.candidate_pack_sha256,
                expected_materialization_sha256=str(
                    materialization["candidate_materialization_sha256"]
                ),
                expected_launch_manifest_sha256=str(
                    authenticated.launch_manifest["launch_manifest_sha256"]
                ),
                expected_launch_receipt_sha256=str(
                    authenticated.final_receipt["receipt_sha256"]
                ),
                expected_launch_receipt_hmac_sha256=str(
                    authenticated.final_receipt["receipt_hmac_sha256"]
                ),
                expected_phase8_artifact_sha256=str(
                    authenticated.phase8_outcome["artifact_sha256"]
                ),
            )
            shutil.rmtree(staging)
            materialization = existing
        else:
            _rename_noreplace(staging, destination)
            _fsync_directory(output)
        return MappingProxyType(
            {
                "child_directory": str(destination.absolute()),
                "child_package_sha256": sealed.package_sha256,
                "published_child_artifact_sha256": sealed.package_sha256,
                "candidate_materialization_sha256": materialization[
                    "candidate_materialization_sha256"
                ],
                "candidate_pack_sha256": candidate.candidate_pack_sha256,
            }
        )
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        if lock_owned and lock_path is not None and lock_path.exists():
            lock_path.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize one authenticated exact-4x LFM2 V5 candidate."
    )
    parser.add_argument("--launch-manifest-json", type=Path, required=True)
    parser.add_argument("--authority-key-file", type=Path, required=True)
    parser.add_argument("--chart-mode", choices=sorted(CHART_MODES), required=True)
    parser.add_argument("--target-arm", choices=sorted(TARGET_ARMS), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = materialize_lfm2_exact4x_v5_candidate(
            launch_manifest_json=args.launch_manifest_json,
            authority_key_path=args.authority_key_file,
            chart_mode=args.chart_mode,
            target_arm=args.target_arm,
        )
    except (RuntimeError, TypeError, ValueError, OSError) as exc:
        print(f"V5 candidate materialization failed: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(dict(result), allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())


__all__ = [
    "AuthenticatedLfm2Exact4xV5CandidatePack",
    "AuthenticatedLfm2Exact4xV5MaterializationProvenance",
    "CANDIDATE_PACK_SCHEMA",
    "Lfm2Exact4xV5CandidateMaterializationError",
    "MATERIALIZATION_RECEIPT",
    "MATERIALIZATION_SCHEMA",
    "REPLACEMENT_TENSOR_NAMES",
    "load_authenticated_lfm2_exact4x_v5_candidate_pack",
    "load_authenticated_lfm2_exact4x_v5_materialization_provenance",
    "main",
    "materialize_lfm2_exact4x_v5_candidate",
]
