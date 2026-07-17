"""Device-independent canonical physical payloads for LFM2 FFN proposals.

Medoid search is not assumed to be reproducible across devices or concurrent
numerical runtimes.  A canonical producer publishes the exact changed-layer
tensors and clustering metadata once.  Workers authenticate that compact pack
and rebuild the full isolated bundle without clustering, medoid selection,
coefficient fitting, or consumer aggregation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_forge.hf_lfm2_global_causal_allocator import Lfm2CausalFfnProposal
from neural_forge.hf_lfm2_variable_width_fold import (
    Lfm2VariableWidthEvidence,
    Lfm2VariableWidthFixedPlanBundle,
    Lfm2VariableWidthLayerPlan,
    rebuild_lfm2_variable_width_fixed_plan_bundle_from_layer_payload,
    verify_lfm2_variable_width_fixed_plan_bundle,
)


CANONICAL_PROPOSAL_PACK_SCHEMA = "neural-forge-lfm2-canonical-ffn-proposal-pack-v1"
MANIFEST_FILENAME = "manifest.json"
TENSOR_FILENAME = "changed-layer.safetensors"
_OPTION_ID = re.compile(r"^ffn-l\d{2}-w\d{4}-[0-9a-f]{16}$")
_TENSOR_KEYS = ("gate_weight", "up_weight", "down_weight")
_ZERO_FIELDS = (
    "parent_forward_calls",
    "parent_cache_build_calls",
    "parent_generation_calls",
    "training_steps",
    "optimizer_steps",
    "backward_calls",
)


class Lfm2CanonicalProposalPackError(RuntimeError):
    """A canonical proposal payload is unsafe, stale, or mismatched."""


@dataclass(frozen=True, slots=True)
class Lfm2CanonicalProposalPack:
    root: Path
    manifest_path: Path
    tensor_path: Path
    manifest: Mapping[str, Any]
    proposal: Lfm2CausalFfnProposal
    bundle: Lfm2VariableWidthFixedPlanBundle


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2CanonicalProposalPackError(
            "Canonical proposal metadata must be finite JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CanonicalProposalPackError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _option_id(value: Any) -> str:
    if not isinstance(value, str) or _OPTION_ID.fullmatch(value) is None:
        raise Lfm2CanonicalProposalPackError(
            "Canonical proposal option ID is malformed."
        )
    return value


def _safe_root(
    value: str | os.PathLike[str],
    *,
    create: bool,
) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2CanonicalProposalPackError("Canonical proposal root must be local.")
    root = Path(raw).expanduser().absolute()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or root.is_symlink():
        raise Lfm2CanonicalProposalPackError(
            "Canonical proposal root is missing or unsafe."
        )
    return root.resolve(strict=True)


def _safe_pack_directory(root: Path, option_id: str) -> Path:
    pack = root / _option_id(option_id)
    if not pack.is_dir() or pack.is_symlink():
        raise Lfm2CanonicalProposalPackError(
            "Canonical proposal pack is missing or unsafe."
        )
    resolved = pack.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Lfm2CanonicalProposalPackError(
            "Canonical proposal pack escaped its root."
        ) from exc
    return resolved


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise Lfm2CanonicalProposalPackError(
            "Canonical proposal manifest is missing or unsafe."
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CanonicalProposalPackError(
            "Canonical proposal manifest is unreadable."
        ) from exc
    if not isinstance(value, dict) or raw != _canonical_json(value) + b"\n":
        raise Lfm2CanonicalProposalPackError(
            "Canonical proposal manifest is not canonical JSON."
        )
    return value


def _proposal_from_dict(value: Mapping[str, Any]) -> Lfm2CausalFfnProposal:
    try:
        proposal = Lfm2CausalFfnProposal(
            schema=value["schema"],
            option_id=value["option_id"],
            source_evidence_sha256=value["source_evidence_sha256"],
            layer_index=value["layer_index"],
            source_width=value["source_width"],
            target_width=value["target_width"],
            parameters_saved=value["parameters_saved"],
            medoid_source_neurons=tuple(value["medoid_source_neurons"]),
            layer_binding_sha256=value["layer_binding_sha256"],
            isolated_plan_sha256=value["isolated_plan_sha256"],
            proposal_sha256=value["proposal_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Lfm2CanonicalProposalPackError(
            "Canonical proposal metadata is malformed."
        ) from exc
    body = proposal.as_dict()
    claimed = body.pop("proposal_sha256")
    if (
        proposal.as_dict() != dict(value)
        or _json_sha256(body) != claimed
        or _option_id(proposal.option_id) != proposal.option_id
    ):
        raise Lfm2CanonicalProposalPackError(
            "Canonical physical proposal seal is invalid."
        )
    return proposal


def _layer_plan_without_tensors(
    layer_plan: Lfm2VariableWidthLayerPlan,
) -> dict[str, Any]:
    record = layer_plan.as_dict()
    for name in _TENSOR_KEYS:
        record.pop(name)
    return record


def _validate_proposal_bundle(
    proposal: Lfm2CausalFfnProposal,
    bundle: Lfm2VariableWidthFixedPlanBundle,
    *,
    parent: Any,
) -> Lfm2VariableWidthLayerPlan:
    verify_lfm2_variable_width_fixed_plan_bundle(bundle, parent=parent)
    contracted = tuple(
        layer_plan
        for layer_plan in bundle.plan.layer_plans
        if layer_plan.target_width != layer_plan.source_width
    )
    if len(contracted) != 1:
        raise Lfm2CanonicalProposalPackError(
            "Canonical packs require one isolated contracted layer."
        )
    layer_plan = contracted[0]
    if (
        proposal.source_evidence_sha256 != bundle.source_evidence_sha256
        or proposal.layer_index != layer_plan.layer_index
        or proposal.source_width != layer_plan.source_width
        or proposal.target_width != layer_plan.target_width
        or proposal.parameters_saved != bundle.plan.parameters_saved
        or proposal.medoid_source_neurons != layer_plan.medoid_source_neurons
        or proposal.layer_binding_sha256
        != bundle.layer_binding_sha256[layer_plan.layer_index]
        or proposal.isolated_plan_sha256 != bundle.plan.plan_sha256
    ):
        raise Lfm2CanonicalProposalPackError(
            "Canonical proposal and physical bundle differ."
        )
    return layer_plan


def write_lfm2_canonical_proposal_pack(
    root: str | os.PathLike[str],
    proposal: Lfm2CausalFfnProposal,
    bundle: Lfm2VariableWidthFixedPlanBundle,
    *,
    parent: Any,
    evidence: Lfm2VariableWidthEvidence,
) -> Lfm2CanonicalProposalPack:
    """Publish one immutable changed-layer pack and reauthenticate it."""

    if not isinstance(proposal, Lfm2CausalFfnProposal):
        raise Lfm2CanonicalProposalPackError("proposal must be Lfm2CausalFfnProposal.")
    root_path = _safe_root(root, create=True)
    option_id = _option_id(proposal.option_id)
    destination = root_path / option_id
    expected = {
        "option_id": proposal.option_id,
        "layer_index": proposal.layer_index,
        "target_width": proposal.target_width,
        "proposal_sha256": proposal.proposal_sha256,
    }
    if destination.exists():
        return load_lfm2_canonical_proposal_pack(
            root_path,
            expected,
            parent=parent,
            evidence=evidence,
            expected_cluster_ridge=bundle.plan.cluster_ridge,
            expected_cluster_max_iterations=(bundle.plan.cluster_max_iterations),
            expected_cluster_distance_chunk_size=(
                bundle.plan.cluster_distance_chunk_size
            ),
        )
    layer_plan = _validate_proposal_bundle(
        proposal,
        bundle,
        parent=parent,
    )
    if evidence.evidence_sha256 != bundle.source_evidence_sha256:
        raise Lfm2CanonicalProposalPackError(
            "Canonical pack received another source evidence object."
        )
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - required HF dependency
        raise Lfm2CanonicalProposalPackError(
            "safetensors is required for canonical proposal packs."
        ) from exc

    stage = Path(
        tempfile.mkdtemp(
            dir=root_path,
            prefix=f".{option_id}.",
        )
    )
    try:
        tensor_path = stage / TENSOR_FILENAME
        tensors = {
            "gate_weight": layer_plan.gate_weight.detach()
            .to(device="cpu")
            .contiguous(),
            "up_weight": layer_plan.up_weight.detach().to(device="cpu").contiguous(),
            "down_weight": layer_plan.down_weight.detach()
            .to(device="cpu")
            .contiguous(),
        }
        save_file(tensors, str(tensor_path))
        layer_record = layer_plan.as_dict()
        tensor_records = {name: layer_record[name] for name in _TENSOR_KEYS}
        body = {
            "schema": CANONICAL_PROPOSAL_PACK_SCHEMA,
            "proposal": proposal.as_dict(),
            "source_evidence_sha256": bundle.source_evidence_sha256,
            "fixed_evidence_sha256": bundle.fixed_evidence.evidence_sha256,
            "target_widths": list(bundle.plan.target_widths),
            "plan_sha256": bundle.plan.plan_sha256,
            "bundle_sha256": bundle.bundle_sha256,
            "contracted_layer_index": layer_plan.layer_index,
            "contracted_layer_binding_sha256": (
                bundle.layer_binding_sha256[layer_plan.layer_index]
            ),
            "layer_plan": _layer_plan_without_tensors(layer_plan),
            "tensor_records": tensor_records,
            "cluster_ridge": bundle.plan.cluster_ridge,
            "cluster_max_iterations": bundle.plan.cluster_max_iterations,
            "cluster_distance_chunk_size": (bundle.plan.cluster_distance_chunk_size),
            "tensor_filename": TENSOR_FILENAME,
            "tensor_file_sha256": _file_sha256(tensor_path),
            **{field: 0 for field in _ZERO_FIELDS},
        }
        manifest = {
            **body,
            "manifest_sha256": _json_sha256(body),
        }
        manifest_path = stage / MANIFEST_FILENAME
        with manifest_path.open("xb") as handle:
            handle.write(_canonical_json(manifest) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return load_lfm2_canonical_proposal_pack(
        root_path,
        expected,
        parent=parent,
        evidence=evidence,
        expected_cluster_ridge=bundle.plan.cluster_ridge,
        expected_cluster_max_iterations=bundle.plan.cluster_max_iterations,
        expected_cluster_distance_chunk_size=(bundle.plan.cluster_distance_chunk_size),
    )


def load_lfm2_canonical_proposal_pack(
    root: str | os.PathLike[str],
    expected_inventory: Mapping[str, Any],
    *,
    parent: Any,
    evidence: Lfm2VariableWidthEvidence,
    expected_cluster_ridge: float,
    expected_cluster_max_iterations: int,
    expected_cluster_distance_chunk_size: int,
) -> Lfm2CanonicalProposalPack:
    """Authenticate and rebuild one exact proposal without reclustering."""

    root_path = _safe_root(root, create=False)
    option_id = _option_id(expected_inventory.get("option_id"))
    layer_index = expected_inventory.get("layer_index")
    target_width = expected_inventory.get("target_width")
    proposal_sha256 = _digest(
        expected_inventory.get("proposal_sha256"),
        "expected proposal_sha256",
    )
    pack_root = _safe_pack_directory(root_path, option_id)
    manifest_path = pack_root / MANIFEST_FILENAME
    manifest = _read_manifest(manifest_path)
    body = dict(manifest)
    manifest_sha256 = body.pop("manifest_sha256", None)
    if (
        body.get("schema") != CANONICAL_PROPOSAL_PACK_SCHEMA
        or _digest(manifest_sha256, "manifest_sha256") != _json_sha256(body)
        or any(body.get(field) != 0 for field in _ZERO_FIELDS)
    ):
        raise Lfm2CanonicalProposalPackError(
            "Canonical proposal manifest seal is invalid."
        )
    raw_proposal = body.get("proposal")
    if not isinstance(raw_proposal, Mapping):
        raise Lfm2CanonicalProposalPackError(
            "Canonical proposal manifest has no proposal."
        )
    proposal = _proposal_from_dict(raw_proposal)
    if (
        proposal.option_id != option_id
        or proposal.proposal_sha256 != proposal_sha256
        or proposal.layer_index != layer_index
        or proposal.target_width != target_width
    ):
        raise Lfm2CanonicalProposalPackError(
            "Canonical proposal differs from the common inventory."
        )
    if (
        body.get("source_evidence_sha256") != evidence.evidence_sha256
        or proposal.source_evidence_sha256 != evidence.evidence_sha256
        or body.get("contracted_layer_index") != layer_index
        or body.get("contracted_layer_binding_sha256") != proposal.layer_binding_sha256
        or body.get("plan_sha256") != proposal.isolated_plan_sha256
        or body.get("cluster_ridge") != expected_cluster_ridge
        or body.get("cluster_max_iterations") != expected_cluster_max_iterations
        or body.get("cluster_distance_chunk_size")
        != expected_cluster_distance_chunk_size
    ):
        raise Lfm2CanonicalProposalPackError(
            "Canonical proposal evidence, plan, or settings are stale."
        )
    target_widths = body.get("target_widths")
    if (
        isinstance(target_widths, (str, bytes))
        or not isinstance(target_widths, Sequence)
        or len(target_widths) != 14
    ):
        raise Lfm2CanonicalProposalPackError(
            "Canonical proposal target widths are malformed."
        )
    tensor_path = pack_root / TENSOR_FILENAME
    if (
        body.get("tensor_filename") != TENSOR_FILENAME
        or not tensor_path.is_file()
        or tensor_path.is_symlink()
        or _file_sha256(tensor_path)
        != _digest(body.get("tensor_file_sha256"), "tensor_file_sha256")
    ):
        raise Lfm2CanonicalProposalPackError(
            "Canonical proposal tensor file is missing or changed."
        )
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise Lfm2CanonicalProposalPackError(
            "safetensors is required for canonical proposal packs."
        ) from exc
    tensors = load_file(str(tensor_path), device="cpu")
    if set(tensors) != set(_TENSOR_KEYS):
        raise Lfm2CanonicalProposalPackError(
            "Canonical proposal tensor keys are incomplete."
        )
    layer_record = body.get("layer_plan")
    tensor_records = body.get("tensor_records")
    if not isinstance(layer_record, Mapping) or not isinstance(
        tensor_records,
        Mapping,
    ):
        raise Lfm2CanonicalProposalPackError(
            "Canonical changed-layer metadata is missing."
        )
    try:
        layer_plan = Lfm2VariableWidthLayerPlan(
            layer_index=layer_record["layer_index"],
            source_width=layer_record["source_width"],
            target_width=layer_record["target_width"],
            medoid_source_neurons=tuple(layer_record["medoid_source_neurons"]),
            source_to_target=tuple(layer_record["source_to_target"]),
            consumer_coefficients=tuple(layer_record["consumer_coefficients"]),
            gate_weight=tensors["gate_weight"],
            up_weight=tensors["up_weight"],
            down_weight=tensors["down_weight"],
            requires_grad=tuple(layer_record["requires_grad"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Lfm2CanonicalProposalPackError(
            "Canonical changed-layer metadata is malformed."
        ) from exc
    rebuilt_record = layer_plan.as_dict()
    rebuilt_without_tensors = dict(rebuilt_record)
    rebuilt_tensors = {name: rebuilt_without_tensors.pop(name) for name in _TENSOR_KEYS}
    if (
        rebuilt_without_tensors != dict(layer_record)
        or rebuilt_tensors != dict(tensor_records)
        or layer_plan.medoid_source_neurons != proposal.medoid_source_neurons
    ):
        raise Lfm2CanonicalProposalPackError(
            "Canonical changed-layer tensor manifest is invalid."
        )
    bundle = rebuild_lfm2_variable_width_fixed_plan_bundle_from_layer_payload(
        parent,
        evidence,
        tuple(int(value) for value in target_widths),
        layer_plan,
        cluster_ridge=expected_cluster_ridge,
        cluster_max_iterations=expected_cluster_max_iterations,
        cluster_distance_chunk_size=(expected_cluster_distance_chunk_size),
        expected_plan_sha256=proposal.isolated_plan_sha256,
        expected_layer_binding_sha256=proposal.layer_binding_sha256,
        expected_bundle_sha256=_digest(
            body.get("bundle_sha256"),
            "bundle_sha256",
        ),
    )
    if bundle.fixed_evidence.evidence_sha256 != body.get("fixed_evidence_sha256"):
        raise Lfm2CanonicalProposalPackError(
            "Canonical fixed evidence seal changed during reconstruction."
        )
    _validate_proposal_bundle(proposal, bundle, parent=parent)
    return Lfm2CanonicalProposalPack(
        root=pack_root,
        manifest_path=manifest_path,
        tensor_path=tensor_path,
        manifest=manifest,
        proposal=proposal,
        bundle=bundle,
    )


__all__ = [
    "CANONICAL_PROPOSAL_PACK_SCHEMA",
    "Lfm2CanonicalProposalPack",
    "Lfm2CanonicalProposalPackError",
    "MANIFEST_FILENAME",
    "TENSOR_FILENAME",
    "load_lfm2_canonical_proposal_pack",
    "write_lfm2_canonical_proposal_pack",
]
