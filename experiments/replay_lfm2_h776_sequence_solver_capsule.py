#!/usr/bin/env python3
"""Replay the immutable H776 solver capsule through the production sweep.

This entry point is deliberately narrow.  It does not implement a solver.  It
authenticates the captured iteration-1 sufficient statistics, invokes the
live production ``_structured_value_pcg_sweep`` exactly once, and emits a
create-new receipt only after all five production group and PCG reports pass
their exact residual and self-seal gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import torch  # noqa: E402

from neural_forge import (  # noqa: E402
    hf_lfm2_sequence_effective_attention_fold as sequence_fold,
)


H776_SOLVER_CAPSULE_SCHEMA = "neural-forge-h776-sequence-effective-solver-capsule-v1"
H776_PRODUCTION_REPLAY_SCHEMA = (
    "neural-forge-h776-sequence-effective-production-sweep-replay-v1"
)
H776_CAPSULE_SHA256 = "31f3618b77bda5b4bfaae34a2bcc43936d36f63647c92f55979ed4706b6f5c12"
H776_CAPSULE_PAYLOAD_SHA256 = (
    "1a47b393781d35f1b719fc57fec41c5907da6d191b9cfa3d8d489dd76faae56b"
)
H776_CAPSULE_PAYLOAD_BYTES = 535_878_879
H776_REFERENCE_FINAL_VALUE_FACTOR_SHA256 = (
    "e157e6b4accf7bbac0691123f7a9b11ec893d83e406bde3a24b7348e1e895a1d"
)
H776_TOPOLOGY = {
    "target_query_heads": 10,
    "target_key_value_heads": 5,
    "target_query_heads_per_group": 2,
    "target_hidden_size": 776,
    "target_output_size": 776,
    "head_dim": 64,
}
_CAPSULE_TENSOR_NAMES = (
    "cross_head_gram",
    "endpoint_right",
    "value_factor",
    "output_blocks",
)
_CAPSULE_MANIFEST_FIELDS = {
    "schema",
    "diagnostic_only",
    "payload_file",
    "payload_file_sha256",
    "payload_file_bytes",
    "tensor_sha256",
    "tensor_shapes",
    "endpoint_square_sum",
    "row_count",
    "topology",
    "policy",
    "iteration",
    "source_sequence_file_sha256",
    "solver_executed",
    "training_steps",
    "backward_calls",
    "optimizer_steps",
    "capsule_sha256",
}
_CAPSULE_PAYLOAD_FIELDS = {
    *_CAPSULE_TENSOR_NAMES,
    "endpoint_square_sum",
    "row_count",
    "topology",
    "policy",
    "iteration",
}
_REQUIRED_EXACT_RESIDUAL = 1e-8
_EXPECTED_PCG_SOLVER = (
    "deterministic_continuous_hidden_partial_trace_cholesky_pcg_single_tensor"
)
_ZERO_MUTATION_EXECUTION = {
    "parent_model_forward_calls": 0,
    "target_model_forward_calls": 0,
    "target_layer_forward_calls": 0,
    "generation_calls": 0,
    "training_steps": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "autograd_graphs": 0,
}


class H776ProductionCapsuleReplayError(RuntimeError):
    """The immutable capsule or production replay failed authentication."""


@dataclass(frozen=True, slots=True)
class H776ProductionCapsuleReplayConfig:
    capsule_directory: Path
    destination_json: Path
    expected_final_value_factor_sha256: str = H776_REFERENCE_FINAL_VALUE_FACTOR_SHA256


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
        raise H776ProductionCapsuleReplayError(
            "Replay evidence must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise H776ProductionCapsuleReplayError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise H776ProductionCapsuleReplayError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise H776ProductionCapsuleReplayError(f"{name} must be finite.") from exc
    if not math.isfinite(result):
        raise H776ProductionCapsuleReplayError(f"{name} must be finite.")
    return result


def _plain(value: Any) -> Any:
    return json.loads(_canonical_json(value).decode("ascii"))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise H776ProductionCapsuleReplayError(
                f"Canonical JSON contains duplicate key {key!r}."
            )
        result[key] = value
    return result


def _read_canonical_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise H776ProductionCapsuleReplayError(
            f"{name} must be one non-symlink regular file."
        )
    raw = path.read_bytes()
    try:
        document = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                H776ProductionCapsuleReplayError(
                    f"{name} contains non-finite value {value!r}."
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise H776ProductionCapsuleReplayError(
            f"{name} is not canonical ASCII JSON."
        ) from exc
    if not isinstance(document, dict) or raw != _canonical_json(document) + b"\n":
        raise H776ProductionCapsuleReplayError(
            f"{name} is not an exact newline-terminated canonical document."
        )
    return document


def _authenticate_seal(
    value: Any,
    *,
    seal_field: str,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise H776ProductionCapsuleReplayError(f"{name} must be a mapping.")
    report = _plain(dict(value))
    supplied = _digest(report.get(seal_field), f"{name}.{seal_field}")
    body = dict(report)
    body.pop(seal_field)
    if _sha256(body) != supplied:
        raise H776ProductionCapsuleReplayError(
            f"{name} failed self-seal authentication."
        )
    return report


def _validated_paths(config: H776ProductionCapsuleReplayConfig) -> tuple[Path, Path]:
    if not isinstance(config, H776ProductionCapsuleReplayConfig):
        raise TypeError("config must be H776ProductionCapsuleReplayConfig.")
    capsule = config.capsule_directory.expanduser().resolve(strict=True)
    if not capsule.is_dir() or config.capsule_directory.is_symlink():
        raise H776ProductionCapsuleReplayError(
            "capsule_directory must be one non-symlink local directory."
        )
    raw_destination = config.destination_json.expanduser().absolute()
    if raw_destination.exists() or raw_destination.is_symlink():
        raise H776ProductionCapsuleReplayError(
            "destination_json already exists; refusing to overwrite it."
        )
    try:
        destination_parent = raw_destination.parent.resolve(strict=True)
    except FileNotFoundError as exc:
        raise H776ProductionCapsuleReplayError(
            "destination_json parent must already exist."
        ) from exc
    if not destination_parent.is_dir():
        raise H776ProductionCapsuleReplayError(
            "destination_json parent must be a directory."
        )
    destination = destination_parent / raw_destination.name
    if (
        destination == capsule
        or destination in capsule.parents
        or capsule in destination.parents
    ):
        raise H776ProductionCapsuleReplayError(
            "destination_json must be disjoint from the immutable capsule."
        )
    return capsule, destination


def _authenticate_capsule(
    capsule_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    manifest_path = capsule_directory / "manifest.json"
    manifest = _read_canonical_json(manifest_path, "capsule manifest")
    if set(manifest) != _CAPSULE_MANIFEST_FIELDS:
        raise H776ProductionCapsuleReplayError(
            "Capsule manifest fields differ from the immutable schema."
        )
    manifest = _authenticate_seal(
        manifest,
        seal_field="capsule_sha256",
        name="capsule manifest",
    )
    if (
        manifest["schema"] != H776_SOLVER_CAPSULE_SCHEMA
        or manifest["capsule_sha256"] != H776_CAPSULE_SHA256
        or manifest["payload_file_sha256"] != H776_CAPSULE_PAYLOAD_SHA256
        or manifest["payload_file_bytes"] != H776_CAPSULE_PAYLOAD_BYTES
        or manifest["diagnostic_only"] is not True
        or manifest["solver_executed"] is not False
        or manifest["iteration"] != 1
        or manifest["training_steps"] != 0
        or manifest["backward_calls"] != 0
        or manifest["optimizer_steps"] != 0
        or manifest["topology"] != H776_TOPOLOGY
    ):
        raise H776ProductionCapsuleReplayError(
            "Capsule manifest differs from the pinned pre-solver H776 evidence."
        )
    _digest(
        manifest["source_sequence_file_sha256"],
        "manifest.source_sequence_file_sha256",
    )
    payload_name = manifest["payload_file"]
    if (
        not isinstance(payload_name, str)
        or not payload_name
        or Path(payload_name).name != payload_name
        or payload_name in {".", ".."}
    ):
        raise H776ProductionCapsuleReplayError(
            "Capsule payload_file must be one local basename."
        )
    payload_path = capsule_directory / payload_name
    if not payload_path.is_file() or payload_path.is_symlink():
        raise H776ProductionCapsuleReplayError(
            "Capsule payload must be one non-symlink regular file."
        )
    payload_path = payload_path.resolve(strict=True)
    if payload_path.parent != capsule_directory:
        raise H776ProductionCapsuleReplayError(
            "Capsule payload escaped its immutable directory."
        )
    if (
        payload_path.stat().st_size != H776_CAPSULE_PAYLOAD_BYTES
        or _file_sha256(payload_path) != H776_CAPSULE_PAYLOAD_SHA256
    ):
        raise H776ProductionCapsuleReplayError(
            "Capsule payload bytes differ from the pinned manifest."
        )
    try:
        payload = torch.load(
            payload_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise H776ProductionCapsuleReplayError(
            "Capsule payload could not be loaded safely."
        ) from exc
    if not isinstance(payload, dict) or set(payload) != _CAPSULE_PAYLOAD_FIELDS:
        raise H776ProductionCapsuleReplayError(
            "Capsule payload fields differ from the immutable schema."
        )
    if (
        payload["iteration"] != 1
        or payload["topology"] != H776_TOPOLOGY
        or payload["topology"] != manifest["topology"]
        or payload["policy"] != manifest["policy"]
        or payload["endpoint_square_sum"] != manifest["endpoint_square_sum"]
        or payload["row_count"] != manifest["row_count"]
    ):
        raise H776ProductionCapsuleReplayError(
            "Capsule payload metadata differs from its sealed manifest."
        )
    endpoint_square_sum = _finite_float(
        payload["endpoint_square_sum"],
        "payload.endpoint_square_sum",
    )
    if (
        endpoint_square_sum < 0.0
        or isinstance(payload["row_count"], bool)
        or not isinstance(payload["row_count"], int)
        or payload["row_count"] <= 0
    ):
        raise H776ProductionCapsuleReplayError(
            "Capsule sufficient-statistic scalar metadata is invalid."
        )

    expected_shapes = {
        "cross_head_gram": [
            H776_TOPOLOGY["target_query_heads"],
            H776_TOPOLOGY["target_query_heads"],
            H776_TOPOLOGY["target_hidden_size"],
            H776_TOPOLOGY["target_hidden_size"],
        ],
        "endpoint_right": [
            H776_TOPOLOGY["target_query_heads"],
            H776_TOPOLOGY["target_hidden_size"],
            H776_TOPOLOGY["target_output_size"],
        ],
        "value_factor": [
            H776_TOPOLOGY["target_key_value_heads"],
            H776_TOPOLOGY["target_hidden_size"],
            H776_TOPOLOGY["head_dim"],
        ],
        "output_blocks": [
            H776_TOPOLOGY["target_query_heads"],
            H776_TOPOLOGY["head_dim"],
            H776_TOPOLOGY["target_output_size"],
        ],
    }
    if (
        set(manifest["tensor_sha256"]) != set(_CAPSULE_TENSOR_NAMES)
        or set(manifest["tensor_shapes"]) != set(_CAPSULE_TENSOR_NAMES)
        or manifest["tensor_shapes"] != expected_shapes
    ):
        raise H776ProductionCapsuleReplayError(
            "Capsule tensor inventory differs from the H776 topology."
        )
    for name in _CAPSULE_TENSOR_NAMES:
        tensor = payload[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.layout != torch.strided
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float64
            or tensor.requires_grad
            or list(tensor.shape) != expected_shapes[name]
            or not bool(torch.isfinite(tensor).all().item())
            or sequence_fold._tensor_sha256(tensor)
            != _digest(
                manifest["tensor_sha256"][name],
                f"manifest.tensor_sha256.{name}",
            )
        ):
            raise H776ProductionCapsuleReplayError(
                f"Capsule tensor {name!r} failed authentication."
            )

    live_policy = sequence_fold.Lfm2FixedQkSequenceEffectiveVoAlsPolicy()
    if (
        not live_policy.verify_authentication()
        or payload["policy"] != live_policy.as_dict()
        or payload["policy"].get("authentication_sha256")
        != sequence_fold.FIXED_QK_SEQUENCE_EFFECTIVE_VO_ALS_POLICY_SHA256
    ):
        raise H776ProductionCapsuleReplayError(
            "Capsule policy differs from the authenticated live policy."
        )
    return manifest, payload, payload_path


def _authenticate_production_reports(
    reports: Any,
    *,
    updated_value_factor: Any,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    if not isinstance(reports, list) or len(reports) != 5:
        raise H776ProductionCapsuleReplayError(
            "Production sweep must return exactly five group reports."
        )
    authenticated: list[dict[str, Any]] = []
    pcg_hashes: list[str] = []
    factor_hashes: list[str] = []
    for group, raw_report in enumerate(reports):
        report = _authenticate_seal(
            raw_report,
            seal_field="report_sha256",
            name=f"group_reports[{group}]",
        )
        role = f"sequence_effective_structured_value_iteration_1_group_{group}"
        heads = [
            group * H776_TOPOLOGY["target_query_heads_per_group"] + offset
            for offset in range(H776_TOPOLOGY["target_query_heads_per_group"])
        ]
        solution_sha256 = sequence_fold._tensor_sha256(updated_value_factor[group])
        if (
            report.get("role") != role
            or report.get("group") != group
            or report.get("heads") != heads
            or report.get("features")
            != H776_TOPOLOGY["target_hidden_size"] * H776_TOPOLOGY["head_dim"]
            or report.get("solution_sha256") != solution_sha256
            or report.get("explicit_coefficient_tensor_materialized") is not False
            or report.get("hidden_head_dim_gram_materialized") is not False
            or report.get("hidden_partial_trace_supplied_to_restart_lane") is not True
            or report.get("hidden_partial_trace_pcg_execution_profile_sha256")
            != sequence_fold.LFM2_SEQUENCE_EFFECTIVE_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE_SHA256
        ):
            raise H776ProductionCapsuleReplayError(
                f"Production group {group} report is not bound to its result."
            )
        construction = _authenticate_seal(
            report.get("hidden_partial_trace_construction_report"),
            seal_field="report_sha256",
            name=f"group_reports[{group}].hidden_partial_trace",
        )
        if (
            construction.get("schema")
            != sequence_fold.LFM2_SEQUENCE_EFFECTIVE_HIDDEN_PARTIAL_TRACE_CHOLESKY_SCHEMA
            or construction.get("role") != role
            or construction.get("hidden_size") != H776_TOPOLOGY["target_hidden_size"]
            or construction.get("head_dim") != H776_TOPOLOGY["head_dim"]
            or construction.get("full_operator_materialized") is not False
        ):
            raise H776ProductionCapsuleReplayError(
                f"Production group {group} preconditioner report is invalid."
            )

        pcg = _authenticate_seal(
            report.get("pcg_report"),
            seal_field="report_sha256",
            name=f"group_reports[{group}].pcg_report",
        )
        final_residual = _finite_float(
            pcg.get("final_relative_normal_residual"),
            f"group_reports[{group}].pcg_report.final_relative_normal_residual",
        )
        if (
            pcg.get("role") != role
            or pcg.get("solver") != _EXPECTED_PCG_SOLVER
            or pcg.get("maximum_iterations") != 4096
            or pcg.get("hard_cap_includes_original_lane_iterations") is not True
            or pcg.get("original_lane_maximum_iterations") != 64
            or pcg.get("original_lane_iterations_executed") != 64
            or isinstance(pcg.get("iterations_executed"), bool)
            or not isinstance(pcg.get("iterations_executed"), int)
            or not 64 < pcg["iterations_executed"] <= 4096
            or pcg.get("continuous_hidden_partial_trace_lane_iterations_executed")
            != pcg["iterations_executed"] - 64
            or pcg.get("single_restart_after_original_lane_exact_authentication_miss")
            is not True
            or pcg.get("required_relative_normal_residual") != _REQUIRED_EXACT_RESIDUAL
            or pcg.get("exact_final_residual_authenticated") is not True
            or final_residual > _REQUIRED_EXACT_RESIDUAL
            or final_residual < 0.0
            or pcg.get("solution_finite") is not True
            or pcg.get("solution_sha256") != solution_sha256
            or pcg.get("periodic_restart_count") != 0
            or pcg.get("extended_after_original_lane_exact_authentication_miss")
            is not True
            or pcg.get("exact_residual_used_to_initialize_continuous_lane_once")
            is not True
            or pcg.get("exact_diagnostic_matvecs_modify_recurrence_state_or_direction")
            is not False
        ):
            raise H776ProductionCapsuleReplayError(
                f"Production group {group} failed its exact PCG result gate."
            )
        execution_profile = _authenticate_seal(
            pcg.get("execution_profile"),
            seal_field="report_sha256",
            name=f"group_reports[{group}].execution_profile",
        )
        if (
            execution_profile
            != dict(
                sequence_fold.LFM2_SEQUENCE_EFFECTIVE_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE
            )
            or pcg.get("execution_profile_sha256") != execution_profile["report_sha256"]
        ):
            raise H776ProductionCapsuleReplayError(
                f"Production group {group} execution profile is invalid."
            )
        restart = pcg.get("restart_preconditioner")
        if not isinstance(restart, Mapping):
            raise H776ProductionCapsuleReplayError(
                f"Production group {group} restart evidence is missing."
            )
        restart = _plain(dict(restart))
        if (
            restart.get("kind") != "deterministic_hidden_partial_trace_cholesky"
            or restart.get("supplied") is not True
            or restart.get("authentication_attempted") is not True
            or restart.get("authenticated") is not True
            or restart.get("applied") is not True
            or restart.get("authentication_failure_reason") is not None
            or isinstance(restart.get("application_attempt_count"), bool)
            or not isinstance(restart.get("application_attempt_count"), int)
            or restart["application_attempt_count"] <= 0
            or restart.get("application_count") != restart["application_attempt_count"]
        ):
            raise H776ProductionCapsuleReplayError(
                f"Production group {group} restart preconditioner was not authenticated."
            )
        receipt = _authenticate_seal(
            restart.get("receipt"),
            seal_field="report_sha256",
            name=f"group_reports[{group}].restart_receipt",
        )
        receipt_construction = _authenticate_seal(
            receipt.get("construction_report"),
            seal_field="report_sha256",
            name=f"group_reports[{group}].receipt_construction",
        )
        factor_sha256 = _digest(
            receipt.get("factor_sha256"),
            f"group_reports[{group}].factor_sha256",
        )
        partial_trace_sha256 = _digest(
            receipt.get("partial_trace_sha256"),
            f"group_reports[{group}].partial_trace_sha256",
        )
        inverse_probe_residual = _finite_float(
            receipt.get("inverse_orientation_probe_relative_residual"),
            f"group_reports[{group}].inverse_orientation_probe_relative_residual",
        )
        if (
            receipt.get("kind") != "deterministic_hidden_partial_trace_cholesky"
            or receipt.get("construction_report_sha256")
            != receipt_construction["report_sha256"]
            or receipt_construction != construction
            or partial_trace_sha256 != construction.get("partial_trace_sha256")
            or receipt.get("factorization") != "lower_cholesky_ex"
            or receipt.get("factorization_info") != 0
            or receipt.get("factor_shape")
            != [
                H776_TOPOLOGY["target_hidden_size"],
                H776_TOPOLOGY["target_hidden_size"],
            ]
            or receipt.get("factor_dtype") != "torch.float64"
            or receipt.get("factor_device") != "cpu"
            or receipt.get("hidden_size") != H776_TOPOLOGY["target_hidden_size"]
            or receipt.get("head_dim") != H776_TOPOLOGY["head_dim"]
            or receipt.get("partial_trace_spd") is not True
            or receipt.get("positive_finite_factor_diagonal") is not True
            or receipt.get("inverse_orientation_probe_tolerance") != 1e-10
            or inverse_probe_residual < 0.0
            or inverse_probe_residual > 1e-10
            or receipt.get(
                "factorization_materialized_only_after_original_lane_exact_miss"
            )
            is not True
            or receipt.get("full_operator_materialized") is not False
        ):
            raise H776ProductionCapsuleReplayError(
                f"Production group {group} factor receipt is invalid."
            )
        authenticated.append(report)
        pcg_hashes.append(pcg["report_sha256"])
        factor_hashes.append(factor_sha256)
    return authenticated, pcg_hashes, factor_hashes


def replay_h776_production_capsule(
    config: H776ProductionCapsuleReplayConfig,
) -> dict[str, Any]:
    """Authenticate, replay once through production, and seal the result."""

    capsule_directory, destination = _validated_paths(config)
    expected_final_sha256 = _digest(
        config.expected_final_value_factor_sha256,
        "expected_final_value_factor_sha256",
    )
    manifest, payload, payload_path = _authenticate_capsule(capsule_directory)
    topology = SimpleNamespace(**H776_TOPOLOGY)
    policy = sequence_fold.Lfm2FixedQkSequenceEffectiveVoAlsPolicy()
    live_source = Path(sequence_fold.__file__).resolve(strict=True)
    replay_source = Path(__file__).resolve(strict=True)
    live_source_sha256_before = _file_sha256(live_source)
    replay_source_sha256_before = _file_sha256(replay_source)
    statistics = {
        "cross_head_gram": payload["cross_head_gram"],
        "endpoint_right": payload["endpoint_right"],
        "endpoint_square_sum": payload["endpoint_square_sum"],
        "row_count": payload["row_count"],
    }

    # This is intentionally the only production-sweep call in this module.
    with torch.no_grad():
        updated_value_factor, raw_reports = sequence_fold._structured_value_pcg_sweep(
            statistics=statistics,
            topology=topology,
            value_factor=payload["value_factor"],
            output_blocks=payload["output_blocks"],
            policy=policy,
            iteration=1,
        )
    if (
        any(
            sequence_fold._tensor_sha256(payload[name])
            != manifest["tensor_sha256"][name]
            for name in _CAPSULE_TENSOR_NAMES
        )
        or policy.as_dict() != payload["policy"]
    ):
        raise H776ProductionCapsuleReplayError(
            "Production sweep mutated its immutable capsule inputs or policy."
        )
    live_source_sha256_after = _file_sha256(live_source)
    replay_source_sha256_after = _file_sha256(replay_source)
    if (
        live_source_sha256_after != live_source_sha256_before
        or replay_source_sha256_after != replay_source_sha256_before
    ):
        raise H776ProductionCapsuleReplayError(
            "Implementation source changed during the production replay."
        )
    expected_shape = (
        H776_TOPOLOGY["target_key_value_heads"],
        H776_TOPOLOGY["target_hidden_size"],
        H776_TOPOLOGY["head_dim"],
    )
    if (
        not isinstance(updated_value_factor, torch.Tensor)
        or updated_value_factor.layout != torch.strided
        or tuple(updated_value_factor.shape) != expected_shape
        or updated_value_factor.dtype != torch.float64
        or updated_value_factor.requires_grad
        or not bool(torch.isfinite(updated_value_factor).all().item())
    ):
        raise H776ProductionCapsuleReplayError(
            "Production sweep returned an invalid final value factor."
        )
    final_value_factor_sha256 = sequence_fold._tensor_sha256(updated_value_factor)
    if final_value_factor_sha256 != expected_final_sha256:
        raise H776ProductionCapsuleReplayError(
            "Production final value factor differs from the preregistered reference."
        )
    reports, pcg_hashes, factor_hashes = _authenticate_production_reports(
        raw_reports,
        updated_value_factor=updated_value_factor,
    )

    implementation = {
        "production_module": sequence_fold.__name__,
        "production_sweep": "_structured_value_pcg_sweep",
        "production_sequence_source_sha256_before": (live_source_sha256_before),
        "production_sequence_source_sha256_after": live_source_sha256_after,
        "replay_source_sha256_before": replay_source_sha256_before,
        "replay_source_sha256_after": replay_source_sha256_after,
        "implementation_sources_unchanged_during_replay": True,
    }
    implementation["implementation_binding_sha256"] = _sha256(implementation)
    body = {
        "schema": H776_PRODUCTION_REPLAY_SCHEMA,
        "status": "authenticated_converged",
        "capsule_schema": manifest["schema"],
        "capsule_sha256": manifest["capsule_sha256"],
        "capsule_payload_file": payload_path.name,
        "capsule_payload_file_sha256": manifest["payload_file_sha256"],
        "capsule_payload_file_bytes": manifest["payload_file_bytes"],
        "captured_sequence_source_sha256": manifest["source_sequence_file_sha256"],
        "implementation": implementation,
        "topology": dict(H776_TOPOLOGY),
        "policy_sha256": policy.authentication_sha256,
        "iteration": 1,
        "production_sweep_calls": 1,
        "production_group_count": len(reports),
        "production_group_reports": reports,
        "production_group_reports_sha256": _sha256(reports),
        "production_pcg_report_sha256": pcg_hashes,
        "production_pcg_reports_sha256": _sha256(pcg_hashes),
        "hidden_partial_trace_factor_sha256": factor_hashes,
        "hidden_partial_trace_factors_sha256": _sha256(factor_hashes),
        "initial_value_factor_sha256": manifest["tensor_sha256"]["value_factor"],
        "expected_final_value_factor_sha256": expected_final_sha256,
        "final_value_factor_sha256": final_value_factor_sha256,
        "final_value_factor_group_sha256": [
            sequence_fold._tensor_sha256(updated_value_factor[group])
            for group in range(H776_TOPOLOGY["target_key_value_heads"])
        ],
        "all_groups_exact_residual_at_or_below_1e_8": True,
        "all_group_and_pcg_reports_self_seal_authenticated": True,
        "capsule_inputs_unchanged_after_production_sweep": True,
        "validation_inputs_accessible": False,
        "selection_inputs_accessible": False,
        "execution": {
            **_ZERO_MUTATION_EXECUTION,
            "production_sweep_calls": 1,
            "closed_form_linear_system_group_solves": 5,
            "cached_parent_sufficient_statistics_only": True,
            "autograd_enabled_during_sweep": False,
        },
    }
    document = {**body, "replay_sha256": _sha256(body)}
    encoded = _canonical_json(document) + b"\n"
    try:
        with destination.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise H776ProductionCapsuleReplayError(
            "destination_json appeared during replay; refusing to overwrite it."
        ) from exc
    return document


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule-directory", type=Path, required=True)
    parser.add_argument("--destination-json", type=Path, required=True)
    parser.add_argument(
        "--expected-final-value-factor-sha256",
        default=H776_REFERENCE_FINAL_VALUE_FACTOR_SHA256,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    document = replay_h776_production_capsule(
        H776ProductionCapsuleReplayConfig(
            capsule_directory=args.capsule_directory,
            destination_json=args.destination_json,
            expected_final_value_factor_sha256=(
                args.expected_final_value_factor_sha256
            ),
        )
    )
    print(
        json.dumps(
            {
                "status": document["status"],
                "production_group_count": document["production_group_count"],
                "final_value_factor_sha256": document["final_value_factor_sha256"],
                "destination_json": os.fspath(
                    args.destination_json.expanduser().absolute()
                ),
                "replay_sha256": document["replay_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
