#!/usr/bin/env python3
"""Validate and export one successful LFM2 CPCF-v5 quarantine attempt.

The validator is intentionally post-success and read-only with respect to the
immutable quarantine.  It authenticates the sealed supervisor chain, performs
one fresh standalone local reload, reproduces the captured 64-case behavior,
and writes a create-new sibling evidence directory.  Parent behavior comes
only from the already-sealed parent envelope; the parent model is never loaded
or executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any


sys.dont_write_bytecode = True
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from experiments import (  # noqa: E402
    evaluate_lfm2_intelligence_holdout as holdout_runner,
)
from experiments import run_lfm2_cpcf_v5_frontier as supervisor  # noqa: E402
from experiments import (  # noqa: E402
    run_lfm2_cpcf_v5_selective_blank_fold as fold_runner,
)
from neural_forge.dynamic_candidate_capture import (  # noqa: E402
    QuarantineCaptureRequest,
    QuarantineEvaluation,
    QuarantineZeroOperationAudit,
    _assert_dense_unpruned_model,
    fingerprint_quarantine_callable,
    load_quarantined_candidate,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    run_fold_only_callback,
)
from neural_forge.hf_lfm2_cpcf_model_fold import (  # noqa: E402
    independent_parameter_inventory,
)
from neural_forge.hf_lfm2_cpcf_v5_quarantine import (  # noqa: E402
    LFM2_CPCF_V5_EXPECTED_HOLDOUT_CASES,
    Lfm2CpcfV5HoldoutEvaluator,
    Lfm2CpcfV5StrictLocalLoader,
    _artifact_parameter_inventory,
    _default_tokenizer_loader,
    _normalized_eos_ids,
    _read_artifact_metadata,
)
from neural_forge.hf_lfm2_intelligence_gate import (  # noqa: E402
    load_manual_review_packet,
)
from neural_forge.hf_lfm2_intelligence_holdout import (  # noqa: E402
    load_lfm2_intelligence_holdout_parent_envelope,
    seal_lfm2_intelligence_holdout_candidate_outputs,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
)


VALIDATION_SCHEMA = "neural-forge-lfm2-cpcf-v5-post-success-validation-v1"
WHOLE_RESPONSE_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-whole-response-comparison-v1"
)
DEFAULT_OUTPUT_PREFIX = "post-success-validation"
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_REQUEST_DIGEST_FIELDS = (
    "plan_sha256",
    "parent_sha256",
    "cache_sha256",
    "bridge_sha256",
    "corpus_sha256",
    "compiler_sha256",
    "source_sha256",
    "saver_sha256",
    "loader_sha256",
    "evaluator_sha256",
)
_OPTIONAL_REQUEST_DIGEST_FIELDS = (
    "expected_model_state_sha256",
    "expected_output_sha256",
    "pre_reviewed_candidate_fingerprint_sha256",
)
_GUARD_ZERO_FIELDS = (
    "optimizer_constructions",
    "optimizer_steps",
    "backward_calls",
    "autograd_grad_calls",
    "pruning_operations",
    "quantization_operations",
)
_CAPTURE_ZERO_FIELDS = (
    *_GUARD_ZERO_FIELDS,
    "training_steps",
    "parent_forward_calls",
    "parent_inference_calls",
)
_EXECUTION_ZERO_FIELDS = (
    "training_steps",
    "backward_calls",
    "optimizer_steps",
    "parameters_pruned",
    "parameters_quantized",
)
_STABLE_RELOAD_FIELDS = (
    "schema",
    "capture_receipt_sha256",
    "destination_path_sha256",
    "model_state_sha256",
    "output_sha256",
    "evaluation_sha256",
    "artifact_inventory_sha256",
    "loader_sha256",
    "evaluator_sha256",
    "loader_path_challenge_performed",
    "loader_path_challenge_exception_type",
    "fresh_final_destination_reload_performed",
    "fresh_reload_execution",
    "promotable",
    "publication_authorized",
    "requires_reviewed_rerun",
)
_SELECTIVE_CATALOG_AUTHENTICATION_FIELDS = (
    "preparation_receipt_sha256",
    "binding_sha256",
    "artifact_manifest_sha256",
    "artifact_catalog_sha256",
    "selective_catalog_sha256",
    "selective_source_sha256",
    "selective_provider_sha256",
    "codec_sha256",
    "provider_implementation_sha256",
    "artifact_store_source_sha256",
    "selective_provider_source_sha256",
)


class Lfm2CpcfV5PostSuccessValidationError(RuntimeError):
    """A successful quarantine attempt could not be authenticated or replayed."""


def _require(condition: Any, message: str) -> None:
    if condition is not True:
        raise Lfm2CpcfV5PostSuccessValidationError(message)


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _identifier(value: Any, name: str) -> str:
    text = str(value).strip()
    if (
        not text
        or text in {".", ".."}
        or "/" in text
        or "\\" in text
        or any(character not in supervisor._IDENTIFIER_CHARACTERS for character in text)
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            f"{name} must be one ordinary path identifier."
        )
    return text


def _ordinary_file(value: Path, name: str) -> Path:
    lexical = Path(value).expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise Lfm2CpcfV5PostSuccessValidationError(
            f"{name} is missing or unsafe: {lexical}."
        )
    try:
        return lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2CpcfV5PostSuccessValidationError(
            f"{name} could not be resolved."
        ) from exc


def _ordinary_directory(value: Path, name: str) -> Path:
    lexical = Path(value).expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_dir():
        raise Lfm2CpcfV5PostSuccessValidationError(
            f"{name} is missing or unsafe: {lexical}."
        )
    try:
        return lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2CpcfV5PostSuccessValidationError(
            f"{name} could not be resolved."
        ) from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, name: str) -> dict[str, Any]:
    source = _ordinary_file(path, name)
    try:
        value = json.loads(source.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfV5PostSuccessValidationError(
            f"{name} is not a JSON object."
        ) from exc
    if not isinstance(value, Mapping):
        raise Lfm2CpcfV5PostSuccessValidationError(
            f"{name} is not a JSON object."
        )
    return dict(value)


def _sealed_inline(
    value: Any,
    *,
    schema: str,
    seal_field: str,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfV5PostSuccessValidationError(
            f"{name} is not a mapping."
        )
    body = dict(value)
    claimed = _digest(body.pop(seal_field, None), f"{name}.{seal_field}")
    if body.get("schema") != schema or canonical_sha256(body) != claimed:
        raise Lfm2CpcfV5PostSuccessValidationError(
            f"{name} schema or self-hash is invalid."
        )
    return {**body, seal_field: claimed}


def _catalog_authentication_summary(
    value: Any,
    *,
    name: str,
) -> dict[str, str]:
    """Authenticate the exact catalog identity folded into source provenance."""

    if not isinstance(value, Mapping):
        raise Lfm2CpcfV5PostSuccessValidationError(
            f"{name} is not an authenticated catalog preparation."
        )
    preparation = dict(value)
    try:
        summary = {
            field: _digest(preparation.get(field), f"{name}.{field}")
            for field in _SELECTIVE_CATALOG_AUTHENTICATION_FIELDS
        }
    except Lfm2CpcfV5PostSuccessValidationError:
        raise
    if (
        preparation.get("status")
        != "prepared_authenticated_tensor_free_catalog"
        or preparation.get("warm_hit_required_for_production_fold") is not True
        or preparation.get("cold_rebuild_permitted_in_production_fold")
        is not False
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            f"{name} has an invalid warm-only production lifecycle."
        )
    return summary


def _option(command: Any, flag: str) -> str:
    if (
        not isinstance(command, list)
        or command.count(flag) != 1
        or command.index(flag) + 1 >= len(command)
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            f"Worker command has no unique {flag} binding."
        )
    value = command[command.index(flag) + 1]
    if not isinstance(value, str) or not value:
        raise Lfm2CpcfV5PostSuccessValidationError(
            f"Worker command {flag} binding is invalid."
        )
    return value


@dataclass(frozen=True, slots=True)
class PostSuccessValidationConfig:
    """Externally rooted inputs for exactly one completed capture attempt."""

    plan_path: Path
    ratio: int
    attempt_id: str
    expected_validator_source_sha256: str
    output_directory: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_path",
            _ordinary_file(Path(self.plan_path), "supervisor plan"),
        )
        if isinstance(self.ratio, bool) or not isinstance(self.ratio, int):
            raise Lfm2CpcfV5PostSuccessValidationError(
                "ratio must be a positive integer."
            )
        if self.ratio <= 0:
            raise Lfm2CpcfV5PostSuccessValidationError(
                "ratio must be a positive integer."
            )
        object.__setattr__(
            self,
            "attempt_id",
            _identifier(self.attempt_id, "attempt_id"),
        )
        _digest(
            self.expected_validator_source_sha256,
            "expected_validator_source_sha256",
        )
        if self.output_directory is not None:
            output = Path(self.output_directory).expanduser().absolute()
            if output.exists() or output.is_symlink():
                raise Lfm2CpcfV5PostSuccessValidationError(
                    "Output directory already exists; refusing overwrite."
                )
            object.__setattr__(self, "output_directory", output)


@dataclass(frozen=True, slots=True)
class AuthenticatedRun:
    """The exact sealed attempt and immutable retained candidate."""

    plan: Mapping[str, Any]
    lane: Mapping[str, Any]
    manifest: Mapping[str, Any]
    execution: Mapping[str, Any]
    diagnostic: Mapping[str, Any]
    quarantine_receipt: Mapping[str, Any]
    artifact_metadata: Mapping[str, Any]
    compiler_guard: Mapping[str, Any]
    consumer_guard: Mapping[str, Any]
    run_directory: Path
    quarantine_directory: Path
    candidate_directory: Path
    output_directory: Path


@dataclass(frozen=True, slots=True)
class ReplayRuntime:
    """Exact bound local loader/evaluator resources for a fresh replay."""

    request: QuarantineCaptureRequest
    loader: Lfm2CpcfV5StrictLocalLoader
    evaluator: Lfm2CpcfV5HoldoutEvaluator
    parent_envelope: Any
    target_tokenizer: Any
    target_tokenizer_semantics_sha256: str
    child_eos_token_ids: tuple[int, ...]


def _output_directory(
    config: PostSuccessValidationConfig,
    run_directory: Path,
) -> Path:
    expected_parent = _ordinary_directory(
        run_directory.parent,
        "attempt plan directory",
    )
    output = (
        expected_parent
        / f"{DEFAULT_OUTPUT_PREFIX}-{config.attempt_id}"
        if config.output_directory is None
        else config.output_directory
    )
    output = Path(output).expanduser().absolute()
    if output.parent != expected_parent:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Validation output must be a direct sibling of the attempt directory."
        )
    if (
        not output.name.startswith(f"{DEFAULT_OUTPUT_PREFIX}-")
        or output.name.startswith("attempt-")
        or output.name == "capture_quarantine"
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Validation output must use the reserved post-success-validation-* "
            "namespace and cannot claim an attempt or capture namespace."
        )
    if output == run_directory or output.exists() or output.is_symlink():
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Validation output is unsafe or already exists."
        )
    return output


def _assert_output_disjoint_from_protected_paths(
    output: Path,
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    run_directory: Path,
    quarantine_directory: Path,
    catalog_preparation: Mapping[str, Any],
) -> None:
    """Reject output paths that claim any sealed input or lifecycle artifact."""

    lanes = plan.get("lanes")
    local_inputs = plan.get("local_inputs")
    if not isinstance(lanes, list) or not isinstance(local_inputs, Mapping):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Supervisor plan lacks sealed lane or input paths."
        )
    protected: dict[str, Path] = {
        "supervisor plan": plan_path,
        "attempt run": run_directory,
        "immutable quarantine": quarantine_directory,
    }
    for name, value in local_inputs.items():
        protected[f"local input {name}"] = Path(str(value))
    for lane_index, lane_value in enumerate(lanes):
        lane_paths = (
            lane_value.get("paths")
            if isinstance(lane_value, Mapping)
            else None
        )
        if not isinstance(lane_paths, Mapping):
            raise Lfm2CpcfV5PostSuccessValidationError(
                "Supervisor plan has a lane without sealed paths."
            )
        for name, value in lane_paths.items():
            if name == "resume_reopen_allowed":
                continue
            protected[f"lane {lane_index} {name}"] = Path(str(value))
    artifact_path = catalog_preparation.get("artifact_path")
    if artifact_path is not None:
        protected["selective catalog artifact"] = Path(str(artifact_path))

    output_value = output.expanduser().absolute().resolve(strict=False)
    for name, value in protected.items():
        candidate = value.expanduser().absolute().resolve(strict=False)
        if supervisor._paths_overlap(output_value, candidate):
            raise Lfm2CpcfV5PostSuccessValidationError(
                f"Validation output overlaps protected {name}: {candidate}."
            )

    lane_manifest = next(
        (
            lane.get("command_manifest")
            for lane in lanes
            if isinstance(lane, Mapping)
            and lane.get("paths", {}).get("run_directory")
            == str(run_directory)
        ),
        None,
    )
    worker_config = (
        lane_manifest.get("worker_config")
        if isinstance(lane_manifest, Mapping)
        else None
    )
    if not isinstance(worker_config, Mapping):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Selected lane has no sealed worker root."
        )
    work = Path(str(worker_config.get("work_directory"))).expanduser().absolute()
    work = work.resolve(strict=False)
    capture_root = work / "capture_quarantine"
    if (
        output_value == work
        or output_value == capture_root
        or work not in output_value.parents
        or capture_root not in output_value.parents
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Validation output does not remain a strict evidence descendant "
            "of the sealed work/capture root."
        )


def _plan_lane(
    plan: Mapping[str, Any],
    *,
    ratio: int,
    attempt_id: str,
) -> tuple[dict[str, Any], Path, Path]:
    lanes = plan.get("lanes")
    if not isinstance(lanes, list):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Supervisor plan contains no lanes."
        )
    matches = [
        value
        for value in lanes
        if isinstance(value, Mapping) and value.get("ratio") == ratio
    ]
    if len(matches) != 1:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Requested ratio is not one unique sealed lane."
        )
    lane = _sealed_inline(
        matches[0],
        schema=supervisor.LANE_SCHEMA,
        seal_field="lane_sha256",
        name="supervisor lane",
    )
    paths = lane.get("paths")
    if not isinstance(paths, Mapping):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Supervisor lane paths are missing."
        )
    planned_manifest = lane.get("command_manifest")
    worker_config = (
        planned_manifest.get("worker_config")
        if isinstance(planned_manifest, Mapping)
        else None
    )
    if not isinstance(worker_config, Mapping):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Supervisor lane has no sealed worker configuration."
        )
    work_directory = _ordinary_directory(
        Path(str(worker_config.get("work_directory"))),
        "sealed supervisor work directory",
    )
    run = _ordinary_directory(Path(str(paths.get("run_directory"))), "attempt")
    quarantine = _ordinary_directory(
        Path(str(paths.get("quarantine_directory"))),
        "quarantine",
    )
    try:
        relative = run.relative_to(work_directory)
    except ValueError as exc:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Attempt lies outside its sealed work directory."
        ) from exc
    if (
        len(relative.parts) != 3
        or relative.parts[0] != "capture_quarantine"
        or not relative.parts[1].startswith(f"{ratio:03d}x-")
        or relative.parts[2] != attempt_id
        or run.name != attempt_id
        or quarantine.parent != run
        or not quarantine.name.startswith("quarantine-")
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Attempt or quarantine path differs from the sealed lifecycle."
        )
    return lane, run, quarantine


def _authenticate_run(
    config: PostSuccessValidationConfig,
) -> AuthenticatedRun:
    plan = supervisor._read_sealed_json(
        config.plan_path,
        schema=supervisor.SUPERVISOR_SCHEMA,
        seal_field="plan_sha256",
        name="post-success supervisor plan",
    )
    if (
        plan.get("status") != "capture_quarantine_processes_ready"
        or config.ratio not in plan.get("requested_ratios", ())
        or plan.get("offline") is not True
        or plan.get("cached_parent_evidence_only") is not True
        or any(plan.get(name) != 0 for name in _EXECUTION_ZERO_FIELDS)
        or plan.get("publication_authorized") is not False
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Supervisor plan is not a clean capture-quarantine plan."
        )
    lane, run, quarantine = _plan_lane(
        plan,
        ratio=config.ratio,
        attempt_id=config.attempt_id,
    )
    output = _output_directory(config, run)
    paths = lane["paths"]
    expected_paths = {
        "diagnostic_json": run / "diagnostic.json",
        "process_manifest_json": run / "process-manifest.json",
        "execution_receipt_json": run / "execution-receipt.json",
    }
    for name, expected in expected_paths.items():
        actual = _ordinary_file(Path(str(paths.get(name))), name)
        if actual != expected:
            raise Lfm2CpcfV5PostSuccessValidationError(
                f"Sealed {name} path differs from the attempt."
            )

    manifest = supervisor._read_sealed_json(
        expected_paths["process_manifest_json"],
        schema=supervisor.COMMAND_SCHEMA,
        seal_field="command_sha256",
        name="post-success process manifest",
    )
    planned_manifest = lane.get("command_manifest")
    if (
        not isinstance(planned_manifest, Mapping)
        or manifest != dict(planned_manifest)
        or manifest.get("kind")
        != "fixed_quarantine_worker_subprocess_manifest"
        or manifest.get("ratio") != config.ratio
        or manifest.get("target_plan_fingerprint")
        != lane.get("target_plan_fingerprint")
        or manifest.get("target_unique_parameters")
        != lane.get("target_unique_parameters")
        or manifest.get("selected_candidate_fingerprint")
        != lane.get("selected_candidate_fingerprint")
        or manifest.get("selected_candidate_rank")
        != lane.get("selected_candidate_rank")
        or manifest.get("paths") != paths
        or manifest.get("one_ratio_only") is not True
        or manifest.get("new_process_required") is not True
        or manifest.get("fixed_programmatic_consumer_seam_used") is not True
        or manifest.get("offline") is not True
        or manifest.get("parent_trace_generation_permitted") is not False
        or any(
            manifest.get(name) is not False
            for name in (
                "training_permitted",
                "backward_permitted",
                "optimizer_permitted",
                "pruning_permitted",
                "quantization_permitted",
            )
        )
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Process manifest differs from the exact sealed capture lane."
        )
    environment = manifest.get("environment_overrides")
    if not isinstance(environment, Mapping):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Process manifest environment bindings are missing."
        )
    changed_environment = {
        str(name): {
            "current": os.environ.get(str(name)),
            "expected": value,
        }
        for name, value in environment.items()
        if os.environ.get(str(name)) != value
    }
    if changed_environment:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Post-success process environment differs from the sealed worker: "
            + json.dumps(changed_environment, sort_keys=True)
        )
    current_sources = supervisor._source_identities()
    if (
        plan.get("source_identities") != current_sources
        or manifest.get("source_identities") != current_sources
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Current launch sources differ from the sealed plan."
        )

    execution = supervisor._read_sealed_json(
        expected_paths["execution_receipt_json"],
        schema=supervisor.EXECUTION_SCHEMA,
        seal_field="receipt_sha256",
        name="post-success execution receipt",
    )
    diagnostic = supervisor._read_sealed_json(
        expected_paths["diagnostic_json"],
        schema=fold_runner.RUN_SCHEMA,
        seal_field="receipt_sha256",
        name="post-success compile diagnostic",
    )
    input_bindings = manifest.get("input_bindings")
    worker_config = manifest.get("worker_config")
    plan_input_bindings = plan.get("input_bindings")
    if (
        not isinstance(input_bindings, Mapping)
        or not isinstance(worker_config, Mapping)
        or not isinstance(plan_input_bindings, Mapping)
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Worker input/config bindings are missing."
        )
    try:
        catalog_preparation = dict(
            fold_runner.authenticate_prepared_selective_catalog(
                trace_cache_directory=Path(
                    str(worker_config["trace_cache_directory"])
                ),
                selective_catalog_store_directory=Path(
                    str(
                        worker_config[
                            "selective_catalog_store_directory"
                        ]
                    )
                ),
                selective_catalog_receipt_json=Path(
                    str(worker_config["selective_catalog_receipt_json"])
                ),
                expected_selective_catalog_receipt_sha256=str(
                    worker_config[
                        "expected_selective_catalog_receipt_sha256"
                    ]
                ),
            )
        )
    except Exception as exc:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Prepared selective catalog failed fresh metadata-only "
            "authentication."
        ) from exc
    catalog_summary = _catalog_authentication_summary(
        catalog_preparation,
        name="prepared selective catalog",
    )
    catalog_records = {
        "supervisor plan": plan_input_bindings.get(
            "selective_catalog_preparation"
        ),
        "worker input bindings": input_bindings.get(
            "selective_catalog_preparation"
        ),
        "worker manifest": manifest.get("selective_catalog_artifact"),
        "compile diagnostic": diagnostic.get("selective_catalog_artifact"),
        "execution receipt": execution.get("selective_catalog_artifact"),
    }
    if any(
        not isinstance(value, Mapping)
        or dict(value) != catalog_preparation
        for value in catalog_records.values()
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Prepared selective catalog differs across sealed plan, worker, "
            "diagnostic, or execution provenance."
        )
    for name, value in catalog_records.items():
        if (
            _catalog_authentication_summary(value, name=name)
            != catalog_summary
        ):
            raise Lfm2CpcfV5PostSuccessValidationError(
                f"{name} has a different selective catalog composite."
            )
    _assert_output_disjoint_from_protected_paths(
        output,
        plan_path=config.plan_path,
        plan=plan,
        run_directory=run,
        quarantine_directory=quarantine,
        catalog_preparation=catalog_preparation,
    )
    diagnostic_command = manifest.get("diagnostic_runner_command")
    if (
        plan.get(
            "shared_selective_catalog_preparation_receipt_sha256"
        )
        != catalog_summary["preparation_receipt_sha256"]
        or worker_config.get(
            "expected_selective_catalog_receipt_sha256"
        )
        != catalog_summary["preparation_receipt_sha256"]
        or _option(
            diagnostic_command,
            "--selective-catalog-store-directory",
        )
        != str(worker_config["selective_catalog_store_directory"])
        or _option(
            diagnostic_command,
            "--selective-catalog-receipt-json",
        )
        != str(worker_config["selective_catalog_receipt_json"])
        or _option(
            diagnostic_command,
            "--expected-selective-catalog-receipt-sha256",
        )
        != catalog_summary["preparation_receipt_sha256"]
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Catalog CLI, config, plan, and receipt pins are not exact."
        )
    plan_stub = SimpleNamespace(
        evidence=SimpleNamespace(
            parent_model_identity_sha256=input_bindings.get(
                "parent_model_identity_sha256"
            ),
            cache_sha256=input_bindings.get("cache_sha256"),
            corpus_sha256=input_bindings.get("corpus_sha256"),
        ),
        config=SimpleNamespace(
            expected_bridge_artifact_sha256=worker_config.get(
                "expected_bridge_artifact_sha256"
            )
        ),
    )
    lane_stub = SimpleNamespace(
        paths=SimpleNamespace(quarantine_directory=quarantine),
        catalog_lane=SimpleNamespace(
            target_plan_fingerprint=manifest.get("target_plan_fingerprint"),
            exact_compression=SimpleNamespace(
                target_unique_parameters=manifest.get(
                    "target_unique_parameters"
                )
            ),
        ),
    )
    try:
        quarantine_receipt = dict(
            supervisor._authenticate_quarantine_receipt(
                plan_stub,
                lane_stub,
                execution,
            )
        )
    except Exception as exc:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Generic quarantine receipt or retained inventory failed authentication."
        ) from exc

    guard_source = current_sources["fold_only_runtime_guard"]
    compiler_guard = dict(
        supervisor._authenticate_runtime_guard_receipt(
            diagnostic.get("compiler_runtime_fold_only_guard"),
            schema=(
                "neural-forge-lfm2-cpcf-v5-compiler-runtime-guard-v1"
            ),
            guard_source_sha256=guard_source,
            name="post-success compiler runtime guard",
        )
    )
    consumer_guard = dict(
        supervisor._authenticate_runtime_guard_receipt(
            execution.get("consumer_runtime_guard_receipt"),
            schema=(
                "neural-forge-lfm2-cpcf-v5-consumer-runtime-fold-only-guard-v1"
            ),
            guard_source_sha256=guard_source,
            name="post-success consumer runtime guard",
        )
    )
    final_reload = execution.get("final_reload_verification")
    compiled = diagnostic.get("compiled_model_fold")
    diagnostic_catalog_reverification = diagnostic.get(
        "selective_catalog_post_run_reverification"
    )
    execution_catalog_reverification = execution.get(
        "selective_catalog_post_run_reverification"
    )
    compiler_implementation = (
        compiled.get("compiler_implementation")
        if isinstance(compiled, Mapping)
        else None
    )
    guard_implementation = compiler_guard.get("guard_implementation")
    try:
        source_binding = fold_runner._source_binding_sha256(
            selective_source_sha256=execution.get(
                "selective_source_sha256"
            ),
            compiler_implementation_sha256=execution.get(
                "compiler_implementation_sha256"
            ),
            runner_source_sha256=execution.get("runner_source_sha256"),
            fold_only_runtime_guard_identity_sha256=(
                guard_implementation.get("identity_sha256")
                if isinstance(guard_implementation, Mapping)
                else None
            ),
            catalog_preparation=catalog_preparation,
        )
        fold_runner.authenticate_selective_catalog_post_run_reverification(
            diagnostic_catalog_reverification,
            catalog_preparation=catalog_preparation,
        )
        fold_runner.authenticate_selective_catalog_post_run_reverification(
            execution_catalog_reverification,
            catalog_preparation=catalog_preparation,
        )
    except Exception as exc:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Catalog-bound source provenance or exact post-run "
            "reverification record could not be authenticated."
        ) from exc
    diagnostic_inputs = diagnostic.get("inputs")
    capture_bindings = quarantine_receipt.get("bindings")
    capture_request = quarantine_receipt.get("capture_request")
    if (
        execution.get("status")
        != "captured_quarantine_requires_reviewed_rerun"
        or execution.get("ratio") != config.ratio
        or execution.get("target_plan_fingerprint")
        != manifest.get("target_plan_fingerprint")
        or execution.get("target_unique_parameters")
        != manifest.get("target_unique_parameters")
        or execution.get("selected_candidate_fingerprint")
        != manifest.get("selected_candidate_fingerprint")
        or execution.get("selected_candidate_rank")
        != manifest.get("selected_candidate_rank")
        or execution.get("parent_sha256")
        != input_bindings.get("parent_model_identity_sha256")
        or execution.get("cache_sha256") != input_bindings.get("cache_sha256")
        or execution.get("bridge_sha256")
        != worker_config.get("expected_bridge_artifact_sha256")
        or execution.get("corpus_sha256")
        != input_bindings.get("corpus_sha256")
        or execution.get("selective_source_sha256")
        != catalog_summary["selective_source_sha256"]
        or execution.get("source_binding_sha256") != source_binding
        or execution.get("post_capture_source_identities") != current_sources
        or execution.get("diagnostic_json")
        != str(expected_paths["diagnostic_json"])
        or execution.get("quarantine_directory") != str(quarantine)
        or any(execution.get(name) != 0 for name in _EXECUTION_ZERO_FIELDS)
        or execution.get("standalone_candidate") is not True
        or execution.get("standalone_reload_verified") is not True
        or execution.get("whole_output_evaluation_captured") is not True
        or execution.get("manual_review_status") != "pending"
        or execution.get("promotable") is not False
        or execution.get("publication_authorized") is not False
        or execution.get("requires_reviewed_rerun") is not True
        or not isinstance(final_reload, Mapping)
        or consumer_guard.get("compiler_runtime_guard_sha256")
        != compiler_guard.get("report_sha256")
        or consumer_guard.get("capture_verification_sha256")
        != final_reload.get("verification_sha256")
        or not isinstance(compiled, Mapping)
        or compiled.get("result_sha256")
        != execution.get("compiler_receipt_sha256")
        or compiled.get("selective_source_sha256")
        != execution.get("selective_source_sha256")
        or not isinstance(compiler_implementation, Mapping)
        or compiler_implementation.get("implementation_sha256")
        != execution.get("compiler_implementation_sha256")
        or compiled.get("selective_catalog_sha256")
        != catalog_summary["selective_catalog_sha256"]
        or not isinstance(diagnostic_inputs, Mapping)
        or diagnostic_inputs.get("parent_fingerprint")
        != execution.get("parent_sha256")
        or diagnostic_inputs.get("cache_key") != execution.get("cache_sha256")
        or diagnostic_inputs.get("bridge_artifact_sha256")
        != execution.get("bridge_sha256")
        or diagnostic_inputs.get("trajectory_corpus_sha256")
        != execution.get("corpus_sha256")
        or diagnostic_inputs.get(
            "selective_catalog_preparation_receipt_sha256"
        )
        != catalog_summary["preparation_receipt_sha256"]
        or diagnostic_inputs.get("selective_catalog_binding_sha256")
        != catalog_summary["binding_sha256"]
        or diagnostic_inputs.get(
            "selective_catalog_artifact_manifest_sha256"
        )
        != catalog_summary["artifact_manifest_sha256"]
        or diagnostic_inputs.get("selective_catalog_artifact_sha256")
        != catalog_summary["artifact_catalog_sha256"]
        or diagnostic_inputs.get("selective_catalog_codec_sha256")
        != catalog_summary["codec_sha256"]
        or diagnostic_inputs.get(
            "selective_catalog_provider_implementation_sha256"
        )
        != catalog_summary["provider_implementation_sha256"]
        or not isinstance(capture_bindings, Mapping)
        or capture_bindings.get("source_sha256") != source_binding
        or not isinstance(capture_request, Mapping)
        or capture_request.get("source_sha256") != source_binding
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Execution receipt differs from its sealed lane or diagnostic."
        )
    resume_progress = diagnostic.get("resume_progress")
    if (
        paths.get("resume_reopen_allowed") is not False
        or not isinstance(resume_progress, Mapping)
        or resume_progress.get("resume_requested") is not False
        or resume_progress.get("invocation_head") is not None
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "This validator requires the non-resumed capture attempt contract."
        )

    candidate = _ordinary_directory(
        quarantine / "candidate",
        "standalone candidate",
    )
    metadata = _read_artifact_metadata(candidate)
    if (
        metadata.get("model_state_sha256")
        != execution.get("target_state_sha256")
        or metadata.get("dense_safetensors_only") is not True
        or metadata.get("target_tokenizer_saved_with_child") is not True
        or metadata.get("local_reload_only") is not True
        or metadata.get("network_access") is not False
        or metadata.get("hub_loader_used") is not False
        or any(
            metadata.get(name) != 0
            for name in (
                "training_steps",
                "backward_calls",
                "optimizer_steps",
                "parameters_pruned",
                "parameters_quantized",
            )
        )
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Standalone artifact metadata is not exact dense local-only state."
        )
    return AuthenticatedRun(
        plan=MappingProxyType(dict(plan)),
        lane=MappingProxyType(dict(lane)),
        manifest=MappingProxyType(dict(manifest)),
        execution=MappingProxyType(dict(execution)),
        diagnostic=MappingProxyType(dict(diagnostic)),
        quarantine_receipt=MappingProxyType(dict(quarantine_receipt)),
        artifact_metadata=MappingProxyType(dict(metadata)),
        compiler_guard=MappingProxyType(dict(compiler_guard)),
        consumer_guard=MappingProxyType(dict(consumer_guard)),
        run_directory=run,
        quarantine_directory=quarantine,
        candidate_directory=candidate,
        output_directory=output,
    )


def _reconstruct_request(
    quarantine_receipt: Mapping[str, Any],
) -> QuarantineCaptureRequest:
    value = quarantine_receipt.get("capture_request")
    if not isinstance(value, Mapping):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Quarantine capture request is missing."
        )
    kwargs = {name: value.get(name) for name in _REQUEST_DIGEST_FIELDS}
    kwargs.update(
        {name: value.get(name) for name in _OPTIONAL_REQUEST_DIGEST_FIELDS}
    )
    try:
        request = QuarantineCaptureRequest(
            **kwargs,
            prior_audit=QuarantineZeroOperationAudit(),
        )
    except Exception as exc:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Quarantine capture request could not be reconstructed."
        ) from exc
    if request.as_dict() != dict(value):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Reconstructed capture request differs from the typed receipt."
        )
    return request


def _build_replay_runtime(run: AuthenticatedRun) -> ReplayRuntime:
    manifest = run.manifest
    worker_config = manifest["worker_config"]
    bindings = manifest["input_bindings"]
    parent_path = _ordinary_file(
        Path(str(worker_config.get("parent_envelope"))),
        "cached parent envelope",
    )
    if _file_sha256(parent_path) != bindings.get(
        "parent_envelope_file_sha256"
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Cached parent-envelope file differs from the sealed input."
        )
    parent_document = _json_object(parent_path, "cached parent envelope")
    try:
        envelope = load_lfm2_intelligence_holdout_parent_envelope(
            parent_document,
            expected_envelope_sha256=bindings.get("parent_envelope_sha256"),
            expected_manifest_sha256=bindings.get(
                "holdout_manifest_sha256"
            ),
            expected_parent_state_fingerprint=parent_document.get(
                "parent_state_fingerprint"
            ),
            expected_parent_model_fingerprint=bindings.get(
                "parent_model_identity_sha256"
            ),
            expected_tokenizer_fingerprint=parent_document.get(
                "tokenizer_fingerprint"
            ),
        )
    except Exception as exc:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Cached parent envelope failed typed authentication."
        ) from exc
    if (
        len(envelope.manifest.cases) != LFM2_CPCF_V5_EXPECTED_HOLDOUT_CASES
        or envelope.parent_model_fingerprint
        != run.execution.get("parent_sha256")
        or envelope.envelope_sha256
        != bindings.get("parent_envelope_sha256")
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Cached parent envelope differs from the exact 64-case lane."
        )

    tokenizer = _default_tokenizer_loader(run.candidate_directory)
    tokenizer_semantics = fingerprint_hf_tokenizer_semantics(tokenizer)
    metadata = run.artifact_metadata
    schedule = manifest.get("schedule_bindings")
    if (
        tokenizer_semantics != metadata.get("tokenizer_semantics_sha256")
        or not isinstance(schedule, Mapping)
        or tokenizer_semantics
        != schedule.get("target_tokenizer_semantics_sha256")
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Fresh child tokenizer differs from the sealed target semantics."
        )
    command = manifest.get("diagnostic_runner_command")
    device = _option(command, "--model-device")
    dtype = _option(command, "--dtype")
    request = _reconstruct_request(run.quarantine_receipt)
    loader = Lfm2CpcfV5StrictLocalLoader(
        context_binding_sha256=metadata["context_binding_sha256"],
        device=device,
        dtype=dtype,
        expected_ffn_width=metadata["materialized_ffn_width"],
        expected_model_state_sha256=metadata["model_state_sha256"],
        expected_parameter_inventory_sha256=metadata[
            "parameter_inventory_sha256"
        ],
        expected_tokenizer_semantics_sha256=tokenizer_semantics,
    )
    evaluator = Lfm2CpcfV5HoldoutEvaluator(
        target_tokenizer=tokenizer,
        parent_envelope=envelope,
        context_binding_sha256=metadata["context_binding_sha256"],
        expected_target_tokenizer_semantics_sha256=tokenizer_semantics,
        expected_case_count=LFM2_CPCF_V5_EXPECTED_HOLDOUT_CASES,
    )
    if (
        fingerprint_quarantine_callable(loader, role="loader")
        != request.loader_sha256
        or fingerprint_quarantine_callable(evaluator, role="evaluator")
        != request.evaluator_sha256
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Fresh local loader/evaluator resources differ from capture."
        )
    return ReplayRuntime(
        request=request,
        loader=loader,
        evaluator=evaluator,
        parent_envelope=envelope,
        target_tokenizer=tokenizer,
        target_tokenizer_semantics_sha256=tokenizer_semantics,
        child_eos_token_ids=_normalized_eos_ids(tokenizer),
    )


def _zero_mapping(
    value: Any,
    fields: Sequence[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfV5PostSuccessValidationError(
            f"{name} is not a mapping."
        )
    result = dict(value)
    if (
        result.get("clean") is not True
        or any(type(result.get(field)) is not int for field in fields)
        or any(result.get(field) != 0 for field in fields)
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            f"{name} does not prove zero forbidden operations."
        )
    return result


def _fresh_replay(
    run: AuthenticatedRun,
    runtime: ReplayRuntime,
) -> tuple[QuarantineEvaluation, Mapping[str, Any], Mapping[str, Any]]:
    try:
        loaded = load_quarantined_candidate(
            run.quarantine_directory,
            request=runtime.request,
            expected_receipt_sha256=run.quarantine_receipt[
                "receipt_sha256"
            ],
            loader=runtime.loader,
            evaluator=runtime.evaluator,
            forbidden_parent_models=(),
        )
    except Exception as exc:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Fresh retained-destination standalone reload failed."
        ) from exc
    if (
        loaded.path != run.quarantine_directory
        or loaded.artifact_path != run.candidate_directory
        or loaded.receipt.as_dict() != dict(run.quarantine_receipt)
        or loaded.promotable is not False
        or loaded.publication_authorized is not False
        or loaded.requires_reviewed_rerun is not True
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Fresh standalone result differs from immutable quarantine."
        )
    fresh_verification = dict(loaded.verification_receipt)
    original_verification = run.execution["final_reload_verification"]
    if any(
        fresh_verification.get(name) != original_verification.get(name)
        for name in _STABLE_RELOAD_FIELDS
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Fresh standalone reload differs from the original stable evidence."
        )
    _zero_mapping(
        fresh_verification.get("capture_observed_audit"),
        _CAPTURE_ZERO_FIELDS,
        "fresh standalone reload audit",
    )
    try:
        _assert_dense_unpruned_model(loaded.model)
        inventory = dict(independent_parameter_inventory(loaded.model))
    except Exception as exc:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Fresh candidate is not exact dense materialized state."
        ) from exc
    metadata = run.artifact_metadata
    if (
        inventory.get("unique_parameters")
        != run.execution.get("target_unique_parameters")
        or inventory.get("inventory_sha256")
        != metadata.get("parameter_inventory_sha256")
        or _artifact_parameter_inventory(inventory)
        != metadata.get("parameter_inventory")
        or inventory.get("independent_methods_agree") is not True
        or inventory.get("checkpoint_byte_size_used") is not False
        or inventory.get("quantization_adjustment_used") is not False
        or inventory.get("pruning_mask_adjustment_used") is not False
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Fresh dense parameter inventory differs from the exact target."
        )

    try:
        evidence, export_guard = run_fold_only_callback(
            lambda: runtime.evaluator(loaded.model),
            phase="lfm2-cpcf-v5-post-success-output-export",
            protected_models=(loaded.model,),
            freeze_parameters=True,
        )
    except Exception as exc:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Guarded post-success whole-output replay failed."
        ) from exc
    if type(evidence) is not QuarantineEvaluation:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Post-success evaluator returned the wrong exact evidence type."
        )
    guard = _zero_mapping(
        export_guard.as_dict(),
        _GUARD_ZERO_FIELDS,
        "post-success output export guard",
    )
    evaluations = run.quarantine_receipt.get("evaluations")
    bindings = run.quarantine_receipt.get("bindings")
    output = evidence.output
    evaluation = evidence.evaluation
    if (
        not isinstance(evaluations, Mapping)
        or not isinstance(bindings, Mapping)
        or evidence.output_sha256 != bindings.get("output_sha256")
        or evidence.output_sha256
        != evaluations.get("fresh_reload_output_sha256")
        or evidence.evaluation_sha256
        != evaluations.get("fresh_reload_evaluation_sha256")
        or output.get("case_count") != LFM2_CPCF_V5_EXPECTED_HOLDOUT_CASES
        or len(output.get("rows", ()))
        != LFM2_CPCF_V5_EXPECTED_HOLDOUT_CASES
        or output.get("full_decoded_outputs_retained") is not True
        or output.get("parent_model_loaded") is not False
        or output.get("parent_forward_calls") != 0
        or evaluation.get("manual_review_status") != "pending"
        or evaluation.get("parent_forward_calls") != 0
        or evaluation.get("optimizer_steps") != 0
        or evaluation.get("backward_calls") != 0
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Guarded whole-output replay differs from captured behavior."
        )
    return evidence, inventory, {
        "fresh_verification": fresh_verification,
        "export_guard": guard,
    }


def _whole_response_comparison(
    run: AuthenticatedRun,
    runtime: ReplayRuntime,
    evidence: QuarantineEvaluation,
) -> dict[str, Any]:
    parent_rows = {
        str(row["case_id"]): dict(row)
        for row in runtime.parent_envelope.rows
    }
    child_rows = {
        str(row["case_id"]): dict(row)
        for row in evidence.output["rows"]
    }
    manifest = runtime.parent_envelope.manifest
    if (
        set(parent_rows) != set(manifest.case_map)
        or set(child_rows) != set(manifest.case_map)
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Whole-response rows do not cover the exact holdout."
        )
    rows = []
    for spec in manifest.cases:
        parent = parent_rows[spec.case_id]
        child = child_rows[spec.case_id]
        rows.append(
            {
                "case_id": spec.case_id,
                "category": spec.category,
                "split": spec.split,
                "prompt": spec.prompt,
                "generation_limit": spec.generation_limit,
                "parent_response": parent["parent_output"],
                "parent_output_token_ids": list(
                    parent["parent_output_token_ids"]
                ),
                "child_response": child["output"],
                "child_output_token_ids": list(child["output_token_ids"]),
                "validator": parent["validator"],
                "parent_effective_capability": parent[
                    "effective_capability"
                ],
                "parent_demonstrated_capability": parent[
                    "capability_demonstrated_by_parent"
                ],
                "manual_review_required": spec.manual_review_required,
            }
        )
    body = {
        "schema": WHOLE_RESPONSE_SCHEMA,
        "plan_sha256": run.plan["plan_sha256"],
        "execution_receipt_sha256": run.execution["receipt_sha256"],
        "capture_receipt_sha256": run.quarantine_receipt["receipt_sha256"],
        "target_state_sha256": run.execution["target_state_sha256"],
        "manifest_sha256": manifest.manifest_sha256,
        "parent_envelope_sha256": runtime.parent_envelope.envelope_sha256,
        "candidate_output_sha256": evidence.output_sha256,
        "case_count": len(rows),
        "rows": rows,
        "semantic_whole_response_scoring_required": True,
        "exact_token_matching_used": False,
        "exact_parent_wording_criterion_used": False,
        "parent_model_loaded": False,
        "parent_forward_calls": 0,
    }
    return {**body, "comparison_sha256": canonical_sha256(body)}


def _new_output_directory(path: Path) -> Path:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Validation output appeared; refusing overwrite."
        ) from exc
    if path.is_symlink() or not path.is_dir():
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Validation output is not an ordinary create-new directory."
        )
    return path.resolve(strict=True)


def _write_output(root: Path, name: str, value: Mapping[str, Any]) -> Path:
    path = root / _identifier(name, "output filename")
    if path.parent != root:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Output path escaped the validation directory."
        )
    supervisor._atomic_new_json(path, value)
    return _ordinary_file(path, f"validation output {name}")


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": _file_sha256(path),
    }


def _publish_validation(
    config: PostSuccessValidationConfig,
    run: AuthenticatedRun,
    runtime: ReplayRuntime,
    evidence: QuarantineEvaluation,
    inventory: Mapping[str, Any],
    replay_audit: Mapping[str, Any],
) -> Mapping[str, Any]:
    candidate_outputs = seal_lfm2_intelligence_holdout_candidate_outputs(
        runtime.parent_envelope.manifest,
        evidence.output["rows"],
        candidate_state_fingerprint=run.execution["target_state_sha256"],
        tokenizer_fingerprint=runtime.target_tokenizer_semantics_sha256,
        eos_token_ids=runtime.child_eos_token_ids,
    )
    comparison = _whole_response_comparison(run, runtime, evidence)
    root = _new_output_directory(run.output_directory)
    output_paths = {
        "behavioral_output": _write_output(
            root,
            "behavioral-output.json",
            evidence.output,
        ),
        "behavioral_evaluation": _write_output(
            root,
            "behavioral-evaluation.json",
            evidence.evaluation,
        ),
        "whole_response_comparison": _write_output(
            root,
            "whole-response-comparison.json",
            comparison,
        ),
        "candidate_outputs": _write_output(
            root,
            "candidate-outputs.json",
            candidate_outputs,
        ),
        "fresh_parameter_inventory": _write_output(
            root,
            "fresh-parameter-inventory.json",
            inventory,
        ),
        "standalone_reload_verification": _write_output(
            root,
            "standalone-reload-verification.json",
            replay_audit["fresh_verification"],
        ),
    }
    automatic_path = root / "automatic-intelligence-report.json"
    manual_path = root / "manual-review-packet.json"
    worker_config = run.manifest["worker_config"]
    parent_envelope_path = _ordinary_file(
        Path(str(worker_config["parent_envelope"])),
        "cached parent envelope",
    )
    try:
        automatic = holdout_runner.evaluate_lfm2_intelligence_holdout(
            holdout_runner.HoldoutEvaluationConfig(
                parent_envelope=parent_envelope_path,
                expected_parent_envelope_file_sha256=run.manifest[
                    "input_bindings"
                ]["parent_envelope_file_sha256"],
                expected_parent_envelope_sha256=(
                    runtime.parent_envelope.envelope_sha256
                ),
                expected_manifest_sha256=(
                    runtime.parent_envelope.manifest.manifest_sha256
                ),
                expected_parent_state_fingerprint=(
                    runtime.parent_envelope.parent_state_fingerprint
                ),
                expected_parent_model_fingerprint=(
                    runtime.parent_envelope.parent_model_fingerprint
                ),
                tokenizer_directory=Path(
                    str(worker_config["donor_tokenizer_directory"])
                ),
                expected_tokenizer_fingerprint=(
                    runtime.parent_envelope.tokenizer_fingerprint
                ),
                child_tokenizer_directory=run.candidate_directory,
                expected_child_tokenizer_fingerprint=(
                    runtime.target_tokenizer_semantics_sha256
                ),
                candidate_report=output_paths["candidate_outputs"],
                expected_candidate_report_sha256=_file_sha256(
                    output_paths["candidate_outputs"]
                ),
                expected_candidate_outputs_sha256=candidate_outputs[
                    "candidate_outputs_sha256"
                ],
                expected_candidate_state_fingerprint=run.execution[
                    "target_state_sha256"
                ],
                report_path=automatic_path,
                manual_review_packet_path=manual_path,
            )
        )
    except Exception as exc:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Cached-only automatic holdout evaluation failed."
        ) from exc
    automatic_auth = supervisor._read_sealed_json(
        automatic_path,
        schema=holdout_runner.RUNNER_SCHEMA,
        seal_field="report_sha256",
        name="post-success automatic intelligence report",
    )
    if automatic_auth != automatic:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Automatic intelligence report changed after publication."
        )
    manual = load_manual_review_packet(
        manual_path,
        expected_packet_sha256=automatic["manual_review"]["packet_sha256"],
    )
    if (
        automatic.get("manual_review", {}).get("status") != "pending"
        or automatic.get("intelligence_retention_claim", {}).get("status")
        != "withheld_pending_manual_review"
        or len(manual.get("cases", ())) != 6
        or manual.get("rubric", {}).get(
            "exact_wording_comparison_permitted"
        )
        is not False
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Automatic report or six-case semantic manual packet is invalid."
        )
    output_paths["automatic_intelligence_report"] = _ordinary_file(
        automatic_path,
        "automatic intelligence report",
    )
    output_paths["manual_review_packet"] = _ordinary_file(
        manual_path,
        "manual review packet",
    )

    retained_inventory = run.quarantine_receipt.get(
        "candidate_artifact",
        {},
    ).get("inventory")
    if (
        not isinstance(retained_inventory, Mapping)
        or not supervisor._retained_inventory_matches(
            run.quarantine_directory,
            retained_inventory,
        )
    ):
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Immutable quarantine changed during post-success validation."
        )
    validator_source = _ordinary_file(
        Path(__file__),
        "post-success validator source",
    )
    validator_sha = _file_sha256(validator_source)
    if validator_sha != config.expected_validator_source_sha256:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Validator source differs from its external expected SHA-256."
        )
    files = {
        name: _file_record(path, root)
        for name, path in sorted(output_paths.items())
    }
    body = {
        "schema": VALIDATION_SCHEMA,
        "status": "validated_pending_manual_review",
        "ratio": config.ratio,
        "attempt_id": config.attempt_id,
        "validator_source_sha256": validator_sha,
        "plan_sha256": run.plan["plan_sha256"],
        "lane_sha256": run.lane["lane_sha256"],
        "command_sha256": run.manifest["command_sha256"],
        "environment_overrides_sha256": canonical_sha256(
            run.manifest["environment_overrides"]
        ),
        "execution_receipt_sha256": run.execution["receipt_sha256"],
        "compile_diagnostic_sha256": run.diagnostic["receipt_sha256"],
        "capture_receipt_sha256": run.quarantine_receipt["receipt_sha256"],
        "original_final_reload_verification_sha256": run.execution[
            "final_reload_verification"
        ]["verification_sha256"],
        "fresh_standalone_reload_verification_sha256": replay_audit[
            "fresh_verification"
        ]["verification_sha256"],
        "compiler_runtime_guard_sha256": run.compiler_guard["report_sha256"],
        "consumer_runtime_guard_sha256": run.consumer_guard["report_sha256"],
        "selective_catalog_authentication": (
            _catalog_authentication_summary(
                run.execution["selective_catalog_artifact"],
                name="validated selective catalog",
            )
        ),
        "target_plan_fingerprint": run.execution["target_plan_fingerprint"],
        "target_state_sha256": run.execution["target_state_sha256"],
        "target_unique_parameters": run.execution[
            "target_unique_parameters"
        ],
        "parameter_inventory_sha256": inventory["inventory_sha256"],
        "candidate_output_sha256": evidence.output_sha256,
        "candidate_evaluation_sha256": evidence.evaluation_sha256,
        "sealed_candidate_outputs_sha256": candidate_outputs[
            "candidate_outputs_sha256"
        ],
        "whole_response_comparison_sha256": comparison[
            "comparison_sha256"
        ],
        "automatic_intelligence_report_sha256": automatic["report_sha256"],
        "manual_review_packet_sha256": manual["packet_sha256"],
        "quarantine_directory": str(run.quarantine_directory),
        "candidate_directory": str(run.candidate_directory),
        "output_directory": str(root),
        "files": files,
        "case_count": LFM2_CPCF_V5_EXPECTED_HOLDOUT_CASES,
        "manual_review_case_count": 6,
        "full_decoded_outputs_saved": True,
        "cached_parent_responses_saved": True,
        "fresh_standalone_reload_verified": True,
        "exact_dense_parameter_target_verified": True,
        "independent_parameter_methods_agree": True,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "post_success_export_guard": dict(
            replay_audit["export_guard"]
        ),
        "parent_model_loaded": False,
        "parent_forward_calls": 0,
        "parent_generate_calls": 0,
        "network_access": False,
        "hf_auto_loader_used": False,
        "from_pretrained_used": False,
        "hub_loader_used": False,
        "exact_parent_token_matching_used": False,
        "exact_parent_wording_criterion_used": False,
        "manual_review_status": "pending",
        "promotable": False,
        "publication_authorized": False,
        "requires_reviewed_rerun": True,
        "completion_marker_written_last": True,
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    receipt_path = _write_output(
        root,
        "post-success-validation-receipt.json",
        receipt,
    )
    authenticated = supervisor._read_sealed_json(
        receipt_path,
        schema=VALIDATION_SCHEMA,
        seal_field="receipt_sha256",
        name="post-success validation receipt",
    )
    if authenticated != receipt:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Final post-success receipt changed after publication."
        )
    return MappingProxyType(receipt)


def validate_lfm2_cpcf_v5_quarantine(
    config: PostSuccessValidationConfig,
) -> Mapping[str, Any]:
    """Authenticate, replay, export, and score one exact quarantine attempt."""

    if type(config) is not PostSuccessValidationConfig:
        raise TypeError("config must be an exact PostSuccessValidationConfig.")
    source_sha = _file_sha256(_ordinary_file(Path(__file__), "validator source"))
    if source_sha != config.expected_validator_source_sha256:
        raise Lfm2CpcfV5PostSuccessValidationError(
            "Validator source differs before validation begins."
        )
    run = _authenticate_run(config)
    runtime = _build_replay_runtime(run)
    evidence, inventory, replay_audit = _fresh_replay(run, runtime)
    return _publish_validation(
        config,
        run,
        runtime,
        evidence,
        inventory,
        replay_audit,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate, freshly reload, and export one completed LFM2 "
            "CPCF-v5 quarantine attempt without parent execution or training."
        )
    )
    parser.add_argument("--plan", dest="plan_path", type=Path, required=True)
    parser.add_argument("--ratio", type=int, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument(
        "--expected-validator-source-sha256",
        required=True,
    )
    parser.add_argument("--output-directory", type=Path)
    return parser


def config_from_args(args: argparse.Namespace) -> PostSuccessValidationConfig:
    return PostSuccessValidationConfig(**vars(args))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = config_from_args(build_parser().parse_args(argv))
        receipt = validate_lfm2_cpcf_v5_quarantine(config)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "target_unique_parameters": receipt[
                        "target_unique_parameters"
                    ],
                    "output_directory": receipt["output_directory"],
                    "manual_review_status": receipt[
                        "manual_review_status"
                    ],
                    "receipt_sha256": receipt["receipt_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
