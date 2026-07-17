#!/usr/bin/env python3
"""Canonical-pack merge entrypoint for the sharded LFM2 causal frontier.

The live producer contract intentionally remains byte-frozen.  Its original
merge finalizer predates canonical proposal packs and re-runs clustering while
sealing evidence and cumulative stages.  Cross-device medoids are not assumed
reproducible, so this wrapper replaces only that future merge callback with the
separately hashed, zero-reclustering canonical finalizer.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments import build_lfm2_cached_causal_evidence as producer
from neural_forge import (
    hf_lfm2_interaction_beam_allocator as canonical_finalizer,
    hf_lfm2_variable_width_fold as variable_width_fold_module,
)
from neural_forge.dynamic_fold_executor import run_fold_only_callback
from neural_forge.hf_lfm2_cached_causal_sharding import (
    verify_lfm2_cached_causal_run_contract,
)
from neural_forge.hf_lfm2_canonical_proposal_pack import (
    load_lfm2_canonical_proposal_pack,
)
from neural_forge.hf_lfm2_global_causal_allocator import (
    Lfm2CausalDamageWeights,
    audit_lfm2_parent_forward_calls,
)


FINALIZER_SCHEMA = "neural-forge-lfm2-canonical-merge-finalizer-v1"
_ROOT_ENV = "NEURAL_FORGE_CANONICAL_PROPOSAL_ROOT"
_CONTRACT_ENV = "NEURAL_FORGE_CAUSAL_RUN_CONTRACT"
_REPORT_ENV = "NEURAL_FORGE_CANONICAL_FINALIZER_REPORT"


class CanonicalMergeFinalizerError(RuntimeError):
    """The separately bound canonical merge finalizer failed closed."""


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
        raise CanonicalMergeFinalizerError(
            "Canonical finalizer provenance must be finite JSON."
        ) from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _safe_file(path: Path, name: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise CanonicalMergeFinalizerError(
            f"{name} is missing, unsafe, or symlinked: {path}."
        )
    return path.resolve(strict=True)


def _safe_directory(path: Path, name: str) -> Path:
    if not path.is_dir() or path.is_symlink():
        raise CanonicalMergeFinalizerError(
            f"{name} is missing, unsafe, or symlinked: {path}."
        )
    return path.resolve(strict=True)


def _required_environment() -> tuple[Path, Path, Path]:
    values = []
    for variable, kind in (
        (_ROOT_ENV, "directory"),
        (_CONTRACT_ENV, "file"),
        (_REPORT_ENV, "output"),
    ):
        raw = os.environ.get(variable, "")
        if not raw:
            raise CanonicalMergeFinalizerError(
                f"Required environment variable {variable} is missing."
            )
        path = Path(raw).expanduser().absolute()
        if kind == "directory":
            path = _safe_directory(path, variable)
        elif kind == "file":
            path = _safe_file(path, variable)
        elif path.exists() and (not path.is_file() or path.is_symlink()):
            raise CanonicalMergeFinalizerError(
                f"{variable} output path is unsafe: {path}."
            )
        values.append(path)
    return values[0], values[1], values[2]


def _load_contract(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as handle:
        contract = json.load(handle)
    verify_lfm2_cached_causal_run_contract(contract)
    return dict(contract)


def _guard_is_zero(report: Mapping[str, Any]) -> bool:
    return all(
        int(report[name]) == 0
        for name in (
            "optimizer_constructions",
            "optimizer_steps",
            "backward_calls",
            "autograd_grad_calls",
        )
    )


def _canonical_replacement(
    parent: Any,
    variable_width_evidence: Any,
    candidates: Sequence[Any],
    split: Any,
    widths_by_layer: Sequence[Sequence[int]],
    measurement_reports: Sequence[Mapping[str, Any]],
    *,
    parent_output_envelope_sha256: str,
    parent_output_file_sha256: str,
    task_anchor_manifest_sha256: str,
    task_anchor_manifest: Mapping[str, Any],
    tokenizer_fingerprint: str,
    damage_weights: Lfm2CausalDamageWeights | None = None,
    cluster_ridge: float = 1e-8,
    cluster_max_iterations: int = 16,
    cluster_distance_chunk_size: int = 512,
    maximum_stage_fraction: float = 0.10,
    max_states: int = 100_000,
) -> tuple[Any, Any, dict[str, Any]]:
    canonical_root, contract_path, report_path = _required_environment()
    contract = _load_contract(contract_path)
    expected_inventory = {
        value["option_id"]: value
        for value in contract["frontier"]["candidate_inventory"]
    }
    candidate_ids = tuple(sorted(candidate.option_id for candidate in candidates))
    if candidate_ids != tuple(sorted(expected_inventory)):
        raise CanonicalMergeFinalizerError(
            "Merged candidates differ from the sealed run-contract inventory."
        )
    clustering_calls = 0
    original_clustering = (
        variable_width_fold_module.build_gated_mlp_activation_cluster_plan
    )

    def forbidden_clustering(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal clustering_calls
        clustering_calls += 1
        raise CanonicalMergeFinalizerError(
            "Canonical merge finalization is forbidden from reclustering."
        )

    def load_and_finalize() -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
        variable_width_fold_module.build_gated_mlp_activation_cluster_plan = (
            forbidden_clustering
        )
        try:
            packs = {
                option_id: load_lfm2_canonical_proposal_pack(
                    canonical_root,
                    expected,
                    parent=parent,
                    evidence=variable_width_evidence,
                    expected_cluster_ridge=cluster_ridge,
                    expected_cluster_max_iterations=cluster_max_iterations,
                    expected_cluster_distance_chunk_size=(
                        cluster_distance_chunk_size
                    ),
                )
                for option_id, expected in sorted(expected_inventory.items())
            }
            evidence, plan, document = (
                canonical_finalizer
                .build_and_seal_lfm2_cached_causal_frontier_from_canonical_packs(
                    parent,
                    variable_width_evidence,
                    candidates,
                    split,
                    widths_by_layer,
                    measurement_reports,
                    canonical_packs=packs,
                    parent_output_envelope_sha256=(
                        parent_output_envelope_sha256
                    ),
                    parent_output_file_sha256=parent_output_file_sha256,
                    task_anchor_manifest_sha256=(
                        task_anchor_manifest_sha256
                    ),
                    task_anchor_manifest=task_anchor_manifest,
                    tokenizer_fingerprint=tokenizer_fingerprint,
                    damage_weights=damage_weights,
                    cluster_ridge=cluster_ridge,
                    cluster_max_iterations=cluster_max_iterations,
                    cluster_distance_chunk_size=(
                        cluster_distance_chunk_size
                    ),
                    maximum_stage_fraction=maximum_stage_fraction,
                    max_states=max_states,
                )
            )
            pack_binding = {
                option_id: {
                    "manifest_sha256": pack.manifest["manifest_sha256"],
                    "tensor_file_sha256": pack.manifest[
                        "tensor_file_sha256"
                    ],
                    "proposal_sha256": pack.proposal.proposal_sha256,
                    "layer_binding_sha256": (
                        pack.proposal.layer_binding_sha256
                    ),
                }
                for option_id, pack in sorted(packs.items())
            }
            return evidence, plan, document, pack_binding
        finally:
            variable_width_fold_module.build_gated_mlp_activation_cluster_plan = (
                original_clustering
            )

    parent_events: list[Mapping[str, Any]] = []
    ((result, parent_calls), guard) = run_fold_only_callback(
        lambda: audit_lfm2_parent_forward_calls(
            parent,
            load_and_finalize,
            phase="lfm2-canonical-merge-finalization",
            event_sink=parent_events,
        ),
        phase="lfm2-canonical-merge-finalization",
        protected_models=(parent,),
    )
    if parent_calls or clustering_calls or not _guard_is_zero(guard.as_dict()):
        raise CanonicalMergeFinalizerError(
            "Canonical merge used a forbidden parent forward, clustering, "
            "or training operation."
        )
    evidence, plan, document, pack_binding = result
    finalizer_path = Path(__file__).resolve(strict=True)
    module_path = Path(canonical_finalizer.__file__).resolve(strict=True)
    body = {
        "schema": FINALIZER_SCHEMA,
        "run_contract_sha256": contract["run_contract_sha256"],
        "frontier_sha256": document["frontier_sha256"],
        "evidence_sha256": evidence.evidence_sha256,
        "plan_sha256": plan.plan_sha256,
        "finalizer_implementation_sha256": _file_sha256(finalizer_path),
        "canonical_allocator_implementation_sha256": _file_sha256(
            module_path
        ),
        "canonical_pack_binding_sha256": _sha256_json(pack_binding),
        "canonical_pack_count": len(pack_binding),
        "guard_report_sha256": _sha256_json(guard.as_dict()),
        "clustering_calls": 0,
        "parent_forward_calls": 0,
        "parent_cache_build_calls": 0,
        "parent_generation_calls": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "exact_token_match_objective_used": False,
    }
    report = {
        **body,
        "finalizer_report_sha256": _sha256_json(body),
    }
    producer._atomic_write_json(report_path, report)
    return evidence, plan, document


def main(argv: Sequence[str] | None = None) -> int:
    selected = list(sys.argv[1:] if argv is None else argv)
    if "--mode" not in selected:
        raise CanonicalMergeFinalizerError(
            "Canonical finalizer requires explicit --mode merge."
        )
    mode_index = selected.index("--mode")
    if mode_index + 1 >= len(selected) or selected[mode_index + 1] != "merge":
        raise CanonicalMergeFinalizerError(
            "Canonical finalizer may only execute merge mode."
        )
    producer.build_and_seal_lfm2_cached_causal_frontier = (
        _canonical_replacement
    )
    return producer.main(selected)


if __name__ == "__main__":
    raise SystemExit(main())
