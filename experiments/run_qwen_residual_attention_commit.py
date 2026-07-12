#!/usr/bin/env python3
"""Commit one *experimental, uncertified* compensated Qwen attention redirect.

This runner is deliberately narrower than a general surgery loop.  It loads an
existing parent-free Qwen codebook checkpoint, previews explicit atomic
redirects with the same transactional machinery as the p35 frontier, fits only
the selected survivor O blocks, gates calibration and held-out cached lanes
separately, and publishes a new canonical four-file checkpoint only after a
parent-free reload audit succeeds.

The incumbent directory is read-only input.  The output directory must not
exist.  No option bypasses the calibration/held-out gate, and successful local
gates do not certify compaction: every output is permanently marked
``experimental-uncertified`` pending broad behavior/generation evaluation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

import neural_forge.hf_qwen_attention_surgery as attention_surgery
from neural_forge.hf_cached_attention_trace import HFCachedAttentionTraceLimits
from neural_forge.hf_qwen_attention_surgery import (
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import (
    load_qwen3_codebook_checkpoint,
    qwen3_codebook_parameter_report,
    save_qwen3_codebook_checkpoint,
)
from neural_forge.hf_residual_attention_fold import (
    ResidualAttentionGatePolicy,
    ResidualAttentionScoringConfig,
)

# The measurement runner owns the already-tested Qwen/cache adapters.  Keeping
# this orchestration beside it avoids cloning those model-version-sensitive
# internals into a second implementation.
import run_qwen_residual_attention_probe as probe


COMMIT_SCHEMA = "neural-forge-qwen-residual-attention-experimental-commit-v1"
CONFIG_MARKER = "neural_forge_residual_attention_experiment"
STATUS = "experimental-uncertified"
CompensationKind = Literal["scalar", "ridge"]
FaultInjector = Callable[[str], None]


class QwenResidualAttentionCommitError(RuntimeError):
    """An experimental redirect failed a mutation, gate, or publication proof."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        default=probe._json_default,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_inventory(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_dir():
        raise QwenResidualAttentionCommitError(
            f"Checkpoint directory does not exist: {path}"
        )
    entries = tuple(sorted(path.iterdir(), key=lambda value: value.name))
    if not entries or any(not value.is_file() for value in entries):
        raise QwenResidualAttentionCommitError(
            "Checkpoint inventory must contain files only."
        )
    return {
        value.name: {
            "bytes": value.stat().st_size,
            "sha256": _file_sha256(value),
        }
        for value in entries
    }


def _resolved_distinct_paths(source: Path, output: Path) -> tuple[Path, Path]:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise QwenResidualAttentionCommitError(
            "Output checkpoint cannot be the incumbent checkpoint."
        )
    if output.exists():
        raise QwenResidualAttentionCommitError(
            "Output checkpoint must not already exist; incumbents are never overwritten."
        )
    if output in source.parents or source in output.parents:
        raise QwenResidualAttentionCommitError(
            "Source and output checkpoints cannot contain one another."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    return source, output


def _prompt_sha256(value: Tensor) -> str:
    prompt = torch.as_tensor(value, dtype=torch.long).detach().cpu().contiguous()
    return probe._tensor_sha256(prompt)


def _validated_split(
    calibration: Sequence[Tensor], heldout: Sequence[Tensor]
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], dict[str, Any]]:
    calibration_values = tuple(calibration)
    heldout_values = tuple(heldout)
    if not calibration_values or not heldout_values:
        raise QwenResidualAttentionCommitError(
            "Calibration and held-out prompt splits must both be non-empty."
        )
    calibration_hashes = tuple(_prompt_sha256(value) for value in calibration_values)
    heldout_hashes = tuple(_prompt_sha256(value) for value in heldout_values)
    if len(set(calibration_hashes)) != len(calibration_hashes):
        raise QwenResidualAttentionCommitError(
            "Calibration prompt tensors must be unique."
        )
    if len(set(heldout_hashes)) != len(heldout_hashes):
        raise QwenResidualAttentionCommitError(
            "Held-out prompt tensors must be unique."
        )
    overlap = sorted(set(calibration_hashes) & set(heldout_hashes))
    if overlap:
        raise QwenResidualAttentionCommitError(
            "Calibration and held-out prompt tensors must be disjoint."
        )
    split = {
        "calibration": {
            "count": len(calibration_values),
            "prompt_sha256s": list(calibration_hashes),
            "used_for_fit": True,
            "used_for_gate": True,
        },
        "heldout": {
            "count": len(heldout_values),
            "prompt_sha256s": list(heldout_hashes),
            "used_for_fit": False,
            "used_for_gate": True,
        },
        "disjoint": True,
    }
    split["split_sha256"] = _sha256(split)
    return calibration_values, heldout_values, split


def _fits_payload(fits: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        str(source): {
            **{
                key: value
                for key, value in fit.items()
                if key not in {"scalar", "ridge"}
            },
            "scalar": fit["scalar"].as_dict(),
            "ridge": fit["ridge"].report.as_dict(),
        }
        for source, fit in fits.items()
    }


def _state_hashes(module: nn.Module) -> dict[str, str]:
    """Hash model state without retaining a second full tensor copy."""

    result: dict[str, str] = {}
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest = hashlib.sha256(
            _canonical_json(
                {
                    "dtype": str(detached.dtype),
                    "shape": [int(item) for item in detached.shape],
                }
            )
        )
        digest.update(
            detached.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
        result[name] = digest.hexdigest()
    return result


def _reload_audit(
    live: nn.Module,
    checkpoint: Path,
    *,
    reload_device: str | torch.device,
) -> tuple[dict[str, Any], nn.Module]:
    live_snapshot = qwen3_attention_runtime_snapshot(live)
    live_report = qwen3_codebook_parameter_report(live)
    live_hashes = _state_hashes(live)
    reloaded = load_qwen3_codebook_checkpoint(checkpoint, device=reload_device).eval()
    reloaded_snapshot = qwen3_attention_runtime_snapshot(reloaded)
    reloaded_report = qwen3_codebook_parameter_report(reloaded)
    reloaded_hashes = _state_hashes(reloaded)
    marker = getattr(reloaded.config, CONFIG_MARKER, None)
    problems = []
    if reloaded_snapshot != live_snapshot:
        problems.append("attention runtime snapshot")
    if reloaded_report != live_report:
        problems.append("parameter accounting")
    if reloaded_hashes != live_hashes:
        problems.append("full runtime tensor state")
    if reloaded_report.parent_runtime_reference_count != 0:
        problems.append("parent runtime reference count")
    if not isinstance(marker, Mapping) or marker.get("status") != STATUS:
        problems.append("embedded experimental status marker")
    if problems:
        raise QwenResidualAttentionCommitError(
            "Reload audit disagreed on: " + ", ".join(problems)
        )
    return (
        {
            "reload_succeeded": True,
            "attention_snapshot_exact": True,
            "parameter_report_exact": True,
            "full_tensor_state_exact": True,
            "parent_runtime_reference_count": 0,
            "embedded_status": marker["status"],
            "reload_device": str(reload_device),
            "state_tensor_count": len(live_hashes),
            "state_manifest_sha256": _sha256(live_hashes),
        },
        reloaded,
    )


def _default_gate_policy() -> ResidualAttentionGatePolicy:
    return ResidualAttentionGatePolicy(
        quantile_limits={
            "attention_js_divergence": 0.10,
            "residual_transition_error": 0.20,
            "cache_transition_error": 0.20,
        },
        worst_limits={
            "attention_js_divergence": 0.20,
            "residual_transition_error": 0.50,
            "cache_transition_error": 0.50,
            "action_mismatch": 0.10,
        },
        max_risk_score=0.25,
        minimum_horizon=2,
    )


def commit_qwen_residual_attention_redirect(
    runtime: nn.Module,
    calibration_input_ids: Sequence[Tensor],
    heldout_input_ids: Sequence[Tensor],
    redirects: Sequence[probe.AttentionRedirect],
    *,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    compensation: CompensationKind,
    horizon: int = 8,
    ridge: float = 1e-6,
    trace_limits: HFCachedAttentionTraceLimits | None = None,
    max_affected_sites: int = 16,
    scoring: ResidualAttentionScoringConfig | None = None,
    gate_policy: ResidualAttentionGatePolicy | None = None,
    tokenizer_contract: Mapping[str, Any] | None = None,
    reload_device: str | torch.device = "cpu",
    fault_injector: FaultInjector | None = None,
) -> dict[str, Any]:
    """Fit, gate, save, and reload one atomic experimental redirect batch.

    ``fault_injector`` is a test hook invoked at named pre-publication stages.
    Any exception restores the incumbent registry/config exactly, deletes the
    staging directory, and leaves ``output_checkpoint`` absent.
    """

    if compensation not in {"scalar", "ridge"}:
        raise QwenResidualAttentionCommitError(
            "compensation must be exactly scalar or ridge."
        )
    redirect_values = tuple(redirects)
    if not redirect_values:
        raise QwenResidualAttentionCommitError(
            "At least one explicit attention redirect is required."
        )
    source, output = _resolved_distinct_paths(
        Path(source_checkpoint), Path(output_checkpoint)
    )
    source_before = _checkpoint_inventory(source)
    calibration, heldout, split = _validated_split(
        calibration_input_ids, heldout_input_ids
    )
    limits = trace_limits or HFCachedAttentionTraceLimits()
    if max(len(calibration), len(heldout)) > limits.max_prompts:
        raise QwenResidualAttentionCommitError(
            f"A prompt split exceeds max_prompts={limits.max_prompts}."
        )
    if horizon > limits.max_horizon:
        raise QwenResidualAttentionCommitError(
            f"horizon exceeds max_horizon={limits.max_horizon}."
        )
    controls = scoring or ResidualAttentionScoringConfig()
    policy = gate_policy or _default_gate_policy()
    parent_registry = probe._registry(runtime)
    parent_snapshot = qwen3_attention_runtime_snapshot(runtime)
    parent_report = qwen3_codebook_parameter_report(runtime)
    parent_installation = attention_surgery._capture_installed(runtime)
    marker_existed = hasattr(runtime.config, CONFIG_MARKER)
    old_marker = getattr(runtime.config, CONFIG_MARKER, None)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    )
    published = False
    proposed_snapshot = None
    edit_audit: dict[str, Any] | None = None
    reloaded: nn.Module | None = None

    def inject(stage: str) -> None:
        if fault_injector is not None:
            fault_injector(stage)

    try:
        bridge, proposal, candidates, tentative = probe._tentative_codebook(
            runtime, redirect_values
        )
        del bridge, proposal, candidates
        codebook, used_projections = tentative
        affected_sites, sites_by_source = probe._affected_sites(
            parent_registry, codebook.registry, redirect_values
        )
        if len(affected_sites) > int(max_affected_sites):
            raise QwenResidualAttentionCommitError(
                "Affected assignment count exceeds max_affected_sites."
            )
        if len({layer for layer, _group in affected_sites}) > limits.max_layers:
            raise QwenResidualAttentionCommitError(
                "Affected layer count exceeds the trace limit."
            )
        parent_o = probe._parent_o_blocks(parent_registry, affected_sites)
        calibration_parent, calibration_drivers = probe._collect_parent_traces(
            runtime,
            calibration,
            affected_sites,
            horizon=horizon,
            limits=limits,
        )
        heldout_parent, heldout_drivers = probe._collect_parent_traces(
            runtime,
            heldout,
            affected_sites,
            horizon=horizon,
            limits=limits,
        )
        inject("after_parent_traces")
        with torch.inference_mode():
            attention_surgery._install_attention_codebook(runtime, codebook)
        proposed_snapshot = qwen3_attention_runtime_snapshot(runtime)
        calibration_uncompensated = probe._collect_candidate_traces(
            runtime,
            calibration,
            affected_sites,
            calibration_drivers,
            horizon=horizon,
            limits=limits,
        )
        fits = probe._fit_survivor_blocks(
            calibration_parent,
            calibration_uncompensated,
            parent_o,
            codebook.registry,
            sites_by_source,
            prompt_count=len(calibration),
            ridge=ridge,
        )
        edit_audit = probe._apply_compensation(
            codebook.registry, fits, kind=compensation
        )
        inject("after_compensation")
        calibration_candidate = probe._collect_candidate_traces(
            runtime,
            calibration,
            affected_sites,
            calibration_drivers,
            horizon=horizon,
            limits=limits,
        )
        heldout_candidate = probe._collect_candidate_traces(
            runtime,
            heldout,
            affected_sites,
            heldout_drivers,
            horizon=horizon,
            limits=limits,
        )
        candidate_ids = tuple(value.candidate_id for value in redirect_values)
        calibration_gate = probe._variant_report(
            calibration_parent,
            calibration_candidate,
            candidate_ids=candidate_ids,
            scoring=controls,
            policy=policy,
        )
        heldout_gate = probe._variant_report(
            heldout_parent,
            heldout_candidate,
            candidate_ids=candidate_ids,
            scoring=controls,
            policy=policy,
        )
        if not calibration_gate["promotion_recommended"]:
            raise QwenResidualAttentionCommitError(
                "Compensated redirect failed the calibration cached-lane gate."
            )
        if not heldout_gate["promotion_recommended"]:
            raise QwenResidualAttentionCommitError(
                "Compensated redirect failed the held-out cached-lane gate."
            )
        compensated_snapshot = qwen3_attention_runtime_snapshot(runtime)
        compensated_report = qwen3_codebook_parameter_report(runtime)
        affected_payload = [
            {
                "layer_index": layer,
                "materialized_group": group,
                "parent_source_flat_index": (
                    probe._materialized_site_bindings(parent_registry)[
                        (layer, group)
                    ].source_flat_index
                ),
                "proposed_source_flat_index": (
                    probe._materialized_site_bindings(codebook.registry)[
                        (layer, group)
                    ].source_flat_index
                ),
            }
            for layer, group in affected_sites
        ]
        marker = {
            "schema": COMMIT_SCHEMA,
            "status": STATUS,
            "compaction_certified": False,
            "broad_behavior_generation_evaluation_performed": False,
            "source_checkpoint_manifest_sha256": source_before["manifest.json"][
                "sha256"
            ],
            "redirects": [value.as_dict() for value in redirect_values],
            "compensation": compensation,
            "ridge": float(ridge),
            "horizon": int(horizon),
            "data_split": split,
            "affected_assignment_sha256": _sha256(affected_payload),
            "fit_contract": {
                "calibration_only": True,
                "heldout_excluded_from_fit": True,
                "all_survivor_assignments_included": True,
                "design": "candidate-runtime-A-child-V-child",
                "target": "isolated-parent-selected-group-O-contribution",
                "edited_tensor": "existing-survivor-prototype-o-block-only",
            },
        }
        marker["transaction_sha256"] = _sha256(marker)
        setattr(runtime.config, CONFIG_MARKER, marker)
        inject("before_save")
        contract = tokenizer_contract
        if contract is None:
            contract = getattr(runtime, "neural_forge_tokenizer_contract", None)
        if not isinstance(contract, Mapping) or not contract:
            raise QwenResidualAttentionCommitError(
                "A resolved tokenizer contract is required for canonical saving."
            )
        manifest = save_qwen3_codebook_checkpoint(
            runtime,
            staging,
            tokenizer_contract=contract,
        )
        reload_audit, reloaded = _reload_audit(
            runtime, staging, reload_device=reload_device
        )
        inject("after_reload")
        source_after_stage = _checkpoint_inventory(source)
        if source_after_stage != source_before:
            raise QwenResidualAttentionCommitError(
                "Incumbent checkpoint bytes changed during the transaction."
            )
        if output.exists():
            raise QwenResidualAttentionCommitError(
                "Output checkpoint appeared during staging; refusing overwrite."
            )
        # Finish every fallible audit/report operation against the staging
        # directory.  A successful directory rename is the final operation, so
        # no later exception can turn a reported failure into a published child.
        output_inventory = _checkpoint_inventory(staging)
        result = {
            "schema": COMMIT_SCHEMA,
            "status": STATUS,
            "fold_committed": True,
            "checkpoint_saved": True,
            "checkpoint_reload_verified": True,
            "compaction_certified": False,
            "broad_behavior_generation_evaluation_performed": False,
            "warning": (
                "This child passed bounded local calibration/held-out cache gates "
                "only; it is not a coherent-compaction result until broad behavior "
                "and free-generation evaluation passes."
            ),
            "source_checkpoint": str(source),
            "output_checkpoint": str(output),
            "incumbent_overwritten": False,
            "incumbent_inventory_before": source_before,
            "incumbent_inventory_after": source_after_stage,
            "incumbent_byte_exact": True,
            "output_inventory": output_inventory,
            "redirects": [value.as_dict() for value in redirect_values],
            "used_projections": [value.as_dict() for value in used_projections],
            "compensation": compensation,
            "ridge": ridge,
            "horizon": horizon,
            "data_split": split,
            "fit_data_access": {
                "calibration_prompt_sha256s": split["calibration"][
                    "prompt_sha256s"
                ],
                "heldout_prompt_sha256s": [],
                "heldout_rows_used": 0,
            },
            "fits": _fits_payload(fits),
            "edit_audit": edit_audit,
            "affected_assignments": affected_payload,
            "affected_assignment_count": len(affected_payload),
            "affected_assignment_sha256": _sha256(affected_payload),
            "survivor_assignment_sets": {
                str(source_index): [list(site) for site in sites]
                for source_index, sites in sites_by_source.items()
            },
            "calibration_gate": calibration_gate,
            "heldout_gate": heldout_gate,
            "both_splits_passed": True,
            "parent_attention_snapshot": parent_snapshot.as_dict(),
            "uncompensated_proposed_attention_snapshot": (
                proposed_snapshot.as_dict()
            ),
            "committed_attention_snapshot": compensated_snapshot.as_dict(),
            "physical_attention_bytes_saved": (
                parent_snapshot.tensor_bytes - compensated_snapshot.tensor_bytes
            ),
            "parameter_accounting": {
                "parent": parent_report.as_dict(),
                "child": compensated_report.as_dict(),
                "unique_parameter_delta": (
                    compensated_report.unique_parameter_count
                    - parent_report.unique_parameter_count
                ),
                "parent_runtime_reference_count": (
                    compensated_report.parent_runtime_reference_count
                ),
            },
            "artifact_accounting": manifest["artifact_plan"],
            "saved_manifest": manifest,
            "reload_audit": reload_audit,
            "embedded_checkpoint_marker": marker,
            "rollback_on_failure_enabled": True,
            "gradient_free": True,
            "training_used": False,
        }
        result["report_sha256"] = _sha256(result)
        os.rename(staging, output)
        published = True
        return result
    except BaseException as error:
        rollback_problems: list[str] = []
        try:
            if marker_existed:
                setattr(runtime.config, CONFIG_MARKER, old_marker)
            elif hasattr(runtime.config, CONFIG_MARKER):
                delattr(runtime.config, CONFIG_MARKER)
        except BaseException as rollback_error:  # pragma: no cover - defensive
            rollback_problems.append(f"config marker: {rollback_error}")
        try:
            if not attention_surgery._restore_installed(runtime, parent_installation):
                rollback_problems.append("attention registry identity/snapshot/report")
        except BaseException as rollback_error:
            rollback_problems.append(f"attention registry: {rollback_error}")
        try:
            if qwen3_attention_runtime_snapshot(runtime) != parent_snapshot:
                rollback_problems.append("post-rollback attention snapshot")
            if qwen3_codebook_parameter_report(runtime) != parent_report:
                rollback_problems.append("post-rollback parameter accounting")
        except BaseException as rollback_error:
            rollback_problems.append(f"post-rollback audit: {rollback_error}")
        try:
            if _checkpoint_inventory(source) != source_before:
                rollback_problems.append("incumbent checkpoint bytes")
        except BaseException as rollback_error:
            rollback_problems.append(f"incumbent inventory: {rollback_error}")
        if not published and output.exists():
            rollback_problems.append("unexpected output checkpoint exists")
        if rollback_problems:
            raise QwenResidualAttentionCommitError(
                "Transaction failed and rollback was not exact: "
                + "; ".join(rollback_problems)
            ) from error
        raise
    finally:
        if reloaded is not None:
            del reloaded
        gc.collect()
        if staging.exists():
            shutil.rmtree(staging)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise QwenResidualAttentionCommitError(
            "Report destination already exists; refusing overwrite."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise QwenResidualAttentionCommitError(
                "Report destination appeared during staging; refusing overwrite."
            )
        os.rename(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_texts(direct: Sequence[str], json_path: Path | None, label: str) -> tuple[str, ...]:
    values = [str(value) for value in direct]
    if json_path is not None:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or any(
            not isinstance(value, str) for value in payload
        ):
            raise QwenResidualAttentionCommitError(
                f"--{label}-prompts-json must contain a JSON string list."
            )
        values.extend(payload)
    if not values:
        raise QwenResidualAttentionCommitError(
            f"Provide at least one --{label}-prompt or --{label}-prompts-json."
        )
    return tuple(values)


def _dtype(value: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=probe.parse_attention_redirect,
        action="append",
        required=True,
        help=(
            "SOURCE_LAYER:SOURCE_PHYSICAL_GROUP="
            "TARGET_LAYER:TARGET_PHYSICAL_GROUP; repeat for one atomic batch"
        ),
    )
    parser.add_argument("--calibration-prompt", action="append", default=[])
    parser.add_argument("--calibration-prompts-json", type=Path)
    parser.add_argument("--heldout-prompt", action="append", default=[])
    parser.add_argument("--heldout-prompts-json", type=Path)
    parser.add_argument("--compensation", choices=("scalar", "ridge"), required=True)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--reload-device", default="cpu")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--max-prompts-per-split", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-affected-sites", type=int, default=16)
    parser.add_argument("--max-affected-layers", type=int, default=8)
    parser.add_argument("--max-capture-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--logit-rtol", type=float, default=5e-3)
    parser.add_argument("--logit-atol", type=float, default=5e-3)
    parser.add_argument("--context-rtol", type=float, default=8e-3)
    parser.add_argument("--context-atol", type=float, default=8e-3)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source, output = _resolved_distinct_paths(
        args.codebook_checkpoint, args.output_checkpoint
    )
    report_path = args.report.resolve()
    if report_path.exists() or report_path == source or report_path == output:
        raise QwenResidualAttentionCommitError(
            "Report path must be new and outside both checkpoint directories."
        )
    try:
        import transformers
    except ImportError as error:
        raise QwenResidualAttentionCommitError("transformers is required.") from error
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    runtime = load_qwen3_codebook_checkpoint(source, device="cpu")
    runtime.to(dtype=_dtype(args.dtype))
    runtime.to(args.device)
    runtime.eval()
    calibration_texts = _load_texts(
        args.calibration_prompt,
        args.calibration_prompts_json,
        "calibration",
    )
    heldout_texts = _load_texts(
        args.heldout_prompt,
        args.heldout_prompts_json,
        "heldout",
    )
    if max(len(calibration_texts), len(heldout_texts)) > args.max_prompts_per_split:
        raise QwenResidualAttentionCommitError(
            "A prompt split exceeds --max-prompts-per-split."
        )

    def encode(texts: Sequence[str]) -> tuple[Tensor, ...]:
        return tuple(
            tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_tokens,
            )["input_ids"]
            for text in texts
        )

    limits = HFCachedAttentionTraceLimits(
        max_prompts=args.max_prompts_per_split,
        max_horizon=args.horizon,
        max_prompt_tokens=args.max_prompt_tokens,
        max_layers=args.max_affected_layers,
        max_groups=1,
        max_model_devices=1,
        max_capture_bytes=args.max_capture_bytes,
        logit_rtol=args.logit_rtol,
        logit_atol=args.logit_atol,
        context_rtol=args.context_rtol,
        context_atol=args.context_atol,
    )
    report = commit_qwen_residual_attention_redirect(
        runtime,
        encode(calibration_texts),
        encode(heldout_texts),
        args.candidate,
        source_checkpoint=source,
        output_checkpoint=output,
        compensation=args.compensation,
        horizon=args.horizon,
        ridge=args.ridge,
        trace_limits=limits,
        max_affected_sites=args.max_affected_sites,
        reload_device=args.reload_device,
    )
    report["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "reload_device": args.reload_device,
        "dtype": args.dtype,
        "numeric_tolerances": {
            "logit_rtol": args.logit_rtol,
            "logit_atol": args.logit_atol,
            "context_rtol": args.context_rtol,
            "context_atol": args.context_atol,
        },
    }
    report["report_sha256"] = _sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    _atomic_write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "checkpoint": str(output),
                "report": str(report_path),
                "report_sha256": report["report_sha256"],
                "physical_attention_bytes_saved": report[
                    "physical_attention_bytes_saved"
                ],
                "compaction_certified": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
