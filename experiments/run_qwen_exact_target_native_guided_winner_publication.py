#!/usr/bin/env python3
"""Publish exactly one tournament winner after a one-shot sealed evaluation.

This stage consumes the candidate-only tournament's authenticated winner.  It
does not retrain or rerank candidates.  The sealed suite is evaluated exactly
once, behind an exclusive durable consumption marker; post-bake validation uses
only fit and candidate.  Existing target-native restore, composite bake, and
fresh-reload certification primitives own all model mutation and publication.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib
import importlib.util
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

import retain_qwen_exact_target_candidates as retention
import run_qwen_exact_target_native_guided_candidate_tournament as tournament
import run_qwen_exact_target_native_guided_continuation as guided
import run_qwen_exact_target_native_stage as stage
from neural_forge.hf_guided_checkpoint_recovery import (
    GuidedCheckpointRecoverySink,
    canonical_json_bytes,
    canonical_sha256,
    portable_v3_factor_state_sha256,
    rebind_recovered_v3_factor_payload,
)
from neural_forge.hf_progressive_target_bridge import ExternalBehaviorGateReceipt
from neural_forge.hf_target_native_behavior_distillation import (
    BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
    authenticate_guided_factor_sources,
    capture_target_native_factor_state,
)


PREREGISTRATION_SCHEMA = (
    "neural-forge-qwen-guided-winner-publication-preregistration-v1"
)
RUN_SCHEMA = "neural-forge-qwen-guided-winner-publication-v1"
SEALED_CONSUMPTION_SCHEMA = "neural-forge-guided-sealed-consumption-v1"
BUILD_FILENAME = "prepare_qwen_exact_target_native_guided_winner_publication.py"
EXPECTED_SPLIT_COUNTS = {"fit": 168, "candidate": 24, "sealed": 32}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GuidedWinnerPublicationError(RuntimeError):
    """Winner provenance, sealed lifecycle, bake, or publication failed."""


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise GuidedWinnerPublicationError(f"{name} must be a lowercase SHA-256.")
    return value


def _strict(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise GuidedWinnerPublicationError(
            f"{name} fields changed; expected={sorted(fields)!r}, actual={actual!r}."
        )
    return value


def _resolve(value: Any, *, base: Path, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise GuidedWinnerPublicationError(f"{name} must be a nonempty path.")
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else base / raw
    cursor = candidate.absolute()
    while True:
        if cursor.is_symlink():
            raise GuidedWinnerPublicationError(f"{name} cannot traverse a symlink.")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    return candidate.resolve()


def _bound_file(
    path: Path | str,
    expected_sha256: str,
    name: str,
) -> Path:
    source = _resolve(path, base=Path.cwd(), name=name)
    expected = _sha256(expected_sha256, f"{name}.sha256")
    if not source.is_file() or stage._file_sha256(source) != expected:
        raise GuidedWinnerPublicationError(f"{name} is absent or changed.")
    return source


def _bound_json(
    path: Path | str,
    expected_sha256: str,
    name: str,
) -> dict[str, Any]:
    source = _bound_file(path, expected_sha256, name)
    payload = stage._strict_json(source)
    if not isinstance(payload, dict):
        raise GuidedWinnerPublicationError(f"{name} must contain one JSON object.")
    canonical_json_bytes(payload)
    return payload


def _module_file(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise GuidedWinnerPublicationError(
            f"Required module {module_name!r} is absent."
        )
    return Path(spec.origin).resolve()


def _implementation_hashes(*, builder_path: Path) -> dict[str, str]:
    builder = _resolve(builder_path, base=Path.cwd(), name="builder implementation")
    if not builder.is_file():
        raise GuidedWinnerPublicationError("Builder implementation is absent.")
    return {
        "runner_sha256": stage._file_sha256(Path(__file__).resolve()),
        "builder_sha256": stage._file_sha256(builder),
        "tournament_runner_sha256": stage._file_sha256(
            Path(tournament.__file__).resolve()
        ),
        "guided_runner_sha256": stage._file_sha256(Path(guided.__file__).resolve()),
        "exact_stage_runner_sha256": stage._file_sha256(Path(stage.__file__).resolve()),
        "trainer_sha256": stage._file_sha256(
            _module_file(guided.GUIDED_TRAINER_MODULE)
        ),
        "checkpoint_recovery_sha256": stage._file_sha256(
            _module_file("neural_forge.hf_guided_checkpoint_recovery")
        ),
        "progressive_bridge_sha256": stage._file_sha256(
            _module_file("neural_forge.hf_progressive_target_bridge")
        ),
        "guided_adapter_sha256": stage._file_sha256(
            _module_file(guided.GUIDED_ADAPTER_MODULE)
        ),
        "independent_authenticator_sha256": stage._file_sha256(
            Path(retention.__file__).resolve()
        ),
    }


def _split_semantics(binding: stage.SplitBinding) -> dict[str, Any]:
    return {
        "path": str(binding.path),
        "sha256": binding.sha256,
        "expected_count": binding.expected_count,
        "row_ids_sha256": canonical_sha256(list(binding.row_ids)),
        "prompt_sha256s_sha256": canonical_sha256(list(binding.prompt_sha256s)),
        "payload_sha256": canonical_sha256(binding.payload),
    }


def _winner_payload(
    contract: tournament.AuthenticatedCandidateTournament,
    run_report: Mapping[str, Any],
    winner_selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        run_report.get("schema") != tournament.RUN_SCHEMA
        or run_report.get("status") != "candidate-winner-selected"
        or run_report.get("evaluation_contract_sha256")
        != contract.evaluation_contract_sha256
        or run_report.get("sealed_forward_count") != 0
        or run_report.get("sealed_evaluation_accessed") is not False
        or run_report.get("model_checkpoint_written") is not False
        or run_report.get("publication_attempted") is not False
        or run_report.get("stop_after_candidate_winner") is not True
        or canonical_json_bytes(run_report.get("winner_selection"))
        != canonical_json_bytes(winner_selection)
    ):
        raise GuidedWinnerPublicationError(
            "Tournament terminal report is not the sealed-free winner boundary."
        )
    winner = winner_selection.get("winner")
    required_winner = {
        "arm_id",
        "arm_index",
        "step",
        "rank",
        "runtime_factor_state_sha256",
        "portable_factor_state_sha256",
        "checkpoint_sha256",
        "metrics",
        "factor_pack",
        "factor_pack_sha256",
    }
    if (
        winner_selection.get("schema") != tournament.WINNER_SCHEMA
        or winner_selection.get("evaluation_contract_sha256")
        != contract.evaluation_contract_sha256
        or winner_selection.get("candidate_only") is not True
        or winner_selection.get("sealed_evaluation_accessed") is not False
        or winner_selection.get("model_checkpoint_written") is not False
        or winner_selection.get("publication_attempted") is not False
        or not isinstance(winner, Mapping)
        or set(winner) != required_winner
    ):
        raise GuidedWinnerPublicationError("Winner selection is malformed or unlocked.")
    factor_path = _resolve(
        winner["factor_pack"],
        base=contract.path.parent,
        name="winner.factor_pack",
    )
    factor_sha = _sha256(winner["factor_pack_sha256"], "winner.factor_pack_sha256")
    _bound_file(factor_path, factor_sha, "winner.factor_pack")
    try:
        loaded = torch.load(factor_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise GuidedWinnerPublicationError(
            "Winner factor pack cannot load safely."
        ) from error
    if not isinstance(loaded, Mapping):
        raise GuidedWinnerPublicationError("Winner factor pack is not a mapping.")
    factor_state_sha = _sha256(
        winner["runtime_factor_state_sha256"],
        "winner.runtime_factor_state_sha256",
    )
    try:
        GuidedCheckpointRecoverySink._validate_v3_factor_payload(
            loaded,
            expected_factor_state_sha256=factor_state_sha,
        )
    except Exception as error:
        raise GuidedWinnerPublicationError(
            "Winner factor pack failed v3 authentication."
        ) from error
    portable_sha = _sha256(
        winner["portable_factor_state_sha256"],
        "winner.portable_factor_state_sha256",
    )
    if portable_v3_factor_state_sha256(loaded) != portable_sha:
        raise GuidedWinnerPublicationError("Winner portable factor identity changed.")
    metrics = winner.get("metrics")
    if not isinstance(metrics, Mapping):
        raise GuidedWinnerPublicationError("Winner candidate metrics are absent.")
    return (
        {
            "arm_id": str(winner["arm_id"]),
            "arm_index": int(winner["arm_index"]),
            "step": int(winner["step"]),
            "rank": list(winner["rank"]),
            "runtime_factor_state_sha256": factor_state_sha,
            "portable_factor_state_sha256": portable_sha,
            "checkpoint_sha256": _sha256(
                winner["checkpoint_sha256"], "winner.checkpoint_sha256"
            ),
            "metrics": dict(metrics),
            "metrics_sha256": canonical_sha256(metrics),
            "factor_pack": str(factor_path),
            "factor_pack_sha256": factor_sha,
        },
        dict(loaded),
    )


def build_winner_publication_preregistration_payload(
    *,
    tournament_preregistration: Path,
    tournament_preregistration_sha256: str,
    tournament_run_report: Path,
    tournament_run_report_sha256: str,
    winner_selection: Path,
    winner_selection_sha256: str,
    sealed_consumption_marker: Path,
    dtype: str,
    parent_device_map: str,
    target_device_map: str,
    builder_path: Path,
) -> tuple[
    dict[str, Any],
    tournament.AuthenticatedCandidateTournament,
    dict[str, Any],
    dict[str, Any],
]:
    tournament_path = _bound_file(
        tournament_preregistration,
        tournament_preregistration_sha256,
        "tournament preregistration",
    )
    tournament_sha = _sha256(
        tournament_preregistration_sha256,
        "tournament_preregistration_sha256",
    )
    source = tournament.authenticate_candidate_tournament_preregistration(
        tournament_path,
        tournament_sha,
    )
    run_path = _bound_file(
        tournament_run_report,
        tournament_run_report_sha256,
        "tournament run report",
    )
    run_sha = _sha256(
        tournament_run_report_sha256,
        "tournament_run_report_sha256",
    )
    run_report = _bound_json(run_path, run_sha, "tournament run report")
    winner_path = _bound_file(
        winner_selection,
        winner_selection_sha256,
        "winner selection",
    )
    winner_sha = _sha256(winner_selection_sha256, "winner_selection_sha256")
    winner_selection_payload = _bound_json(winner_path, winner_sha, "winner selection")
    winner, factor_payload = _winner_payload(
        source,
        run_report,
        winner_selection_payload,
    )
    source_runtime = source.runtime
    if (
        dtype != source_runtime["dtype"]
        or parent_device_map != source_runtime["parent_device_map"]
        or target_device_map != source_runtime["target_device_map"]
    ):
        raise GuidedWinnerPublicationError(
            "Winner publication runtime must exactly match the authenticated "
            "tournament runtime."
        )
    marker = _resolve(
        sealed_consumption_marker,
        base=Path.cwd(),
        name="sealed consumption marker",
    )
    if marker.name != "sealed_consumption.json":
        raise GuidedWinnerPublicationError(
            "Sealed marker must be a real path named sealed_consumption.json."
        )
    sealed = source.splits["sealed"]
    if sealed.expected_count != 32:
        raise GuidedWinnerPublicationError(
            "Winner publication requires exact sealed32."
        )
    splits = {
        name: _split_semantics(source.splits[name])
        for name in ("fit", "candidate", "sealed")
    }
    implementation = _implementation_hashes(builder_path=builder_path)
    runtime = {
        "dtype": dtype,
        "attention_implementation": stage.CANONICAL_ATTENTION_IMPLEMENTATION,
        "parent_device_map": parent_device_map,
        "target_device_map": target_device_map,
        "explicit_execute_required": True,
    }
    semantic_contract = {
        "tournament_evaluation_contract_sha256": source.evaluation_contract_sha256,
        "source_guided_evaluation_contract_sha256": (
            source.source.evaluation_contract_sha256
        ),
        "winner_portable_factor_state_sha256": winner["portable_factor_state_sha256"],
        "winner_candidate_metrics_sha256": winner["metrics_sha256"],
        "splits": splits,
        "sealed_forward_limit": 1,
        "post_bake_split_order": ["fit", "candidate"],
        "post_bake_sealed_forbidden": True,
        "runtime": runtime,
        "implementation": implementation,
    }
    payload = {
        "schema": PREREGISTRATION_SCHEMA,
        "tournament": {
            "preregistration": str(tournament_path),
            "preregistration_sha256": tournament_sha,
            "run_report": str(run_path),
            "run_report_sha256": run_sha,
            "winner_selection": str(winner_path),
            "winner_selection_sha256": winner_sha,
        },
        "source_guided": {
            "preregistration": str(source.source.path),
            "preregistration_sha256": source.source.file_sha256,
            "evaluation_contract_sha256": source.source.evaluation_contract_sha256,
        },
        "winner": winner,
        "splits": splits,
        "sealed_consumption": {
            "marker": str(marker),
            "schema": SEALED_CONSUMPTION_SCHEMA,
            "must_be_absent_at_preflight": True,
            "exclusive_create_immediately_before_forward": True,
            "maximum_forward_count": 1,
        },
        "runtime": runtime,
        "implementation": implementation,
        "semantic_contract": semantic_contract,
        "semantic_contract_sha256": canonical_sha256(semantic_contract),
        "model_free_preregistration": True,
        "model_weights_loaded": False,
        "gpu_work_launched": False,
        "publication_authorized_before_sealed_pass": False,
    }
    return payload, source, winner, factor_payload


@dataclass(frozen=True, slots=True)
class AuthenticatedWinnerPublication:
    path: Path
    file_sha256: str
    tournament: tournament.AuthenticatedCandidateTournament
    winner: Mapping[str, Any]
    factor_payload: Mapping[str, Any]
    sealed_marker: Path
    runtime: Mapping[str, Any]
    raw: Mapping[str, Any]

    def preflight_report(self) -> dict[str, Any]:
        return {
            "schema": RUN_SCHEMA,
            "status": "preflight-passed",
            "preregistration": str(self.path),
            "preregistration_sha256": self.file_sha256,
            "semantic_contract_sha256": self.raw["semantic_contract_sha256"],
            "evaluation_contract_sha256": self.raw["semantic_contract_sha256"],
            "winner": dict(self.winner),
            "sealed": dict(self.raw["splits"]["sealed"]),
            "sealed_consumption_marker": str(self.sealed_marker),
            "sealed_marker_absent": not self.sealed_marker.exists(),
            "runtime": dict(self.runtime),
            "models_loaded": False,
            "gpu_work_launched": False,
            "sealed_forward_count": 0,
            "publication_attempted": False,
        }


def authenticate_winner_publication_preregistration(
    path: Path | str,
    expected_sha256: str,
) -> AuthenticatedWinnerPublication:
    source = _bound_file(path, expected_sha256, "winner publication preregistration")
    expected = _sha256(expected_sha256, "preregistration_sha256")
    raw = _bound_json(source, expected, "winner publication preregistration")
    _strict(
        raw,
        {
            "schema",
            "tournament",
            "source_guided",
            "winner",
            "splits",
            "sealed_consumption",
            "runtime",
            "implementation",
            "semantic_contract",
            "semantic_contract_sha256",
            "model_free_preregistration",
            "model_weights_loaded",
            "gpu_work_launched",
            "publication_authorized_before_sealed_pass",
        },
        "winner publication preregistration",
    )
    if raw.get("schema") != PREREGISTRATION_SCHEMA:
        raise GuidedWinnerPublicationError("Winner publication schema changed.")
    tournament_row = _strict(
        raw["tournament"],
        {
            "preregistration",
            "preregistration_sha256",
            "run_report",
            "run_report_sha256",
            "winner_selection",
            "winner_selection_sha256",
        },
        "tournament",
    )
    marker_row = _strict(
        raw["sealed_consumption"],
        {
            "marker",
            "schema",
            "must_be_absent_at_preflight",
            "exclusive_create_immediately_before_forward",
            "maximum_forward_count",
        },
        "sealed_consumption",
    )
    builder_path = Path(__file__).resolve().with_name(BUILD_FILENAME)
    expected_payload, tournament_contract, winner, factor_payload = (
        build_winner_publication_preregistration_payload(
            tournament_preregistration=Path(str(tournament_row["preregistration"])),
            tournament_preregistration_sha256=str(
                tournament_row["preregistration_sha256"]
            ),
            tournament_run_report=Path(str(tournament_row["run_report"])),
            tournament_run_report_sha256=str(tournament_row["run_report_sha256"]),
            winner_selection=Path(str(tournament_row["winner_selection"])),
            winner_selection_sha256=str(tournament_row["winner_selection_sha256"]),
            sealed_consumption_marker=Path(str(marker_row["marker"])),
            dtype=str(raw["runtime"]["dtype"]),
            parent_device_map=str(raw["runtime"]["parent_device_map"]),
            target_device_map=str(raw["runtime"]["target_device_map"]),
            builder_path=builder_path,
        )
    )
    if canonical_json_bytes(raw) != canonical_json_bytes(expected_payload):
        raise GuidedWinnerPublicationError(
            "Winner publication preregistration differs from authenticated sources."
        )
    marker = _resolve(
        marker_row["marker"],
        base=source.parent,
        name="sealed consumption marker",
    )
    return AuthenticatedWinnerPublication(
        path=source,
        file_sha256=expected,
        tournament=tournament_contract,
        winner=MappingProxyType(dict(winner)),
        factor_payload=MappingProxyType(dict(factor_payload)),
        sealed_marker=marker,
        runtime=MappingProxyType(dict(raw["runtime"])),
        raw=MappingProxyType(dict(raw)),
    )


def _consume_sealed_marker(
    contract: AuthenticatedWinnerPublication,
    output_root: Path,
) -> dict[str, Any]:
    marker = contract.sealed_marker
    if (
        marker.parent != contract.path.parent
        or marker.name != "sealed_consumption.json"
    ):
        raise GuidedWinnerPublicationError(
            "Sealed marker escaped the preregistration evidence directory."
        )
    payload = {
        "schema": SEALED_CONSUMPTION_SCHEMA,
        "status": "consumed-before-forward",
        "preregistration_sha256": contract.file_sha256,
        "semantic_contract_sha256": contract.raw["semantic_contract_sha256"],
        "winner_portable_factor_state_sha256": contract.winner[
            "portable_factor_state_sha256"
        ],
        "sealed_split_sha256": contract.tournament.splits["sealed"].sha256,
        "sealed_forward_limit": 1,
        "output_root": str(output_root),
    }
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise GuidedWinnerPublicationError(
            "Sealed suite was already consumed for this preregistration."
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            parent_descriptor = os.open(marker.parent, os.O_RDONLY)
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        except OSError:
            pass
    except BaseException:
        # Once exclusive creation succeeds, never remove the marker: a crash or
        # partial write burns sealed access fail-closed.
        raise
    return {
        **payload,
        "marker": str(marker),
        "marker_file_sha256": stage._file_sha256(marker),
        "exclusive_create_succeeded": True,
    }


def _method_gate(
    metrics: Any,
    *,
    winner_step: int,
    tournament_pareto_admission: bool,
) -> tuple[bool, dict[str, Any]]:
    payload = stage._metrics_dict(metrics)
    checks = {
        "winner_is_post_step_zero": winner_step > 0,
        "candidate_has_one_lcp_at_least_4_rollout": int(payload["lcp_at_least_4_count"])
        >= 1,
        "tournament_later_state_required_pareto_safety_vs_step_zero": (
            tournament_pareto_admission
        ),
    }
    decision = {
        "passed": all(checks.values()),
        "checks": checks,
        "winner_step": winner_step,
        "candidate_lcp_at_least_4_count": payload["lcp_at_least_4_count"],
        "internal_numeric_matching_consulted": False,
    }
    decision["decision_sha256"] = canonical_sha256(decision)
    return bool(decision["passed"]), decision


class ProductionWinnerPublicationRuntime(guided.ProductionGuidedRuntime):
    """Production heavy boundary with an auditable sealed forward counter."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sealed_forward_count = 0

    def evaluate_publication_split(
        self,
        model: Any,
        parent: Any,
        contract: AuthenticatedWinnerPublication,
        split_name: str,
    ) -> Any:
        if split_name not in {"fit", "candidate", "sealed"}:
            raise GuidedWinnerPublicationError("Unknown publication split.")
        if split_name == "sealed":
            self.sealed_forward_count += 1
            if self.sealed_forward_count != 1:
                raise GuidedWinnerPublicationError(
                    "Sealed evaluation exceeded one call."
                )
        module = importlib.import_module(guided.GUIDED_TRAINER_MODULE)
        loader = getattr(module, "load_behavior_domain_suite")
        evaluator = getattr(module, "evaluate_behavior_domain")
        binding = contract.tournament.splits[split_name]
        suite = loader(
            split_name,
            binding.payload,
            artifact_sha256=binding.sha256,
        )
        schedule = contract.tournament.schedule
        return evaluator(
            parent,
            model,
            suite,
            horizon=schedule.evaluation_horizon,
            top_k=schedule.top_k,
            split={
                "fit": "calibration-exploratory",
                "candidate": "calibration-selection",
                "sealed": "heldout-evaluation",
            }[split_name],
        )

    def independent_reload_certify(
        self,
        child_path: Path,
        exact_unique_parameters: int,
    ) -> Mapping[str, Any]:
        return retention._independent_accepted_child_authenticator(
            child_path,
            exact_unique_parameters,
        )


def execute_winner_publication(
    contract: AuthenticatedWinnerPublication,
    output_root: Path | str,
    *,
    dtype: str,
    parent_device_map: str,
    target_device_map: str,
    runtime: Any | None = None,
) -> dict[str, Any]:
    """Execute the exact winner once; sealed access is permanently one-shot."""

    if type(contract) is not AuthenticatedWinnerPublication:
        raise GuidedWinnerPublicationError(
            "Execution requires the canonical authenticated winner contract."
        )
    refreshed = authenticate_winner_publication_preregistration(
        contract.path,
        contract.file_sha256,
    )
    if canonical_json_bytes(refreshed.raw) != canonical_json_bytes(contract.raw):
        raise GuidedWinnerPublicationError("Winner sources changed before execution.")
    expected_runtime = refreshed.runtime
    if (
        dtype != expected_runtime["dtype"]
        or parent_device_map != expected_runtime["parent_device_map"]
        or target_device_map != expected_runtime["target_device_map"]
    ):
        raise GuidedWinnerPublicationError(
            "Execution runtime differs from preregistration."
        )
    if refreshed.sealed_marker.exists():
        raise GuidedWinnerPublicationError(
            "Sealed suite was already consumed; refusing model access."
        )
    cuda_preflight = tournament.validate_candidate_tournament_cuda_preflight(
        refreshed.tournament,
        dtype=dtype,
        parent_device_map=parent_device_map,
        target_device_map=target_device_map,
    )
    runner = (
        ProductionWinnerPublicationRuntime(
            dtype=dtype,
            parent_device_map=parent_device_map,
            target_device_map=target_device_map,
        )
        if runtime is None
        else runtime
    )
    base = dataclasses.replace(
        refreshed.tournament.source.execution_contract,
        splits=refreshed.tournament.splits,
        training=refreshed.tournament.schedule.as_dict(),
    )
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise GuidedWinnerPublicationError("output_root must not already exist.")
    output.mkdir(parents=True)
    parent: Any = None
    target: Any = None
    tokenizer: Any = None
    compiled: Any = None
    transaction: Any = None
    compiled_sha: str | None = None
    published = False
    marker_receipt: Mapping[str, Any] | None = None
    report: dict[str, Any] = {
        **refreshed.preflight_report(),
        "status": "running",
        "output_root": str(output),
        "cuda_preflight": dict(cuda_preflight),
    }
    try:
        prepare_cache = getattr(runner, "prepare_compiler_cache", None)
        if callable(prepare_cache):
            cache = prepare_cache(base)
            if cache is not None:
                report["compiled_child_cache"] = dict(cache)
        parent, target, tokenizer = runner.load_models(base)
        report.update({"models_loaded": True, "gpu_work_launched": True})
        compiled = runner.compile(parent, target, tokenizer, base)
        compiler_report = stage._compiled_report_dict(compiled)
        compiled_sha = runner.state_sha256(compiled.child_model)
        stage._validate_continuation_compilation(base, compiler_report, compiled_sha)
        finalize = getattr(runner, "finalize_compiled_base", None)
        if callable(finalize):
            compiled = finalize(compiled, base)
            if runner.state_sha256(compiled.child_model) != compiled_sha:
                raise GuidedWinnerPublicationError(
                    "Compiler finalization changed the compiled child."
                )
        transaction = runner.install_transaction(compiled)
        inherited_install = transaction.installation_report()
        source_restore = runner.restore_source_v2(
            transaction, refreshed.tournament.source
        )
        guided_install = runner.install_guided_transaction(
            transaction,
            refreshed.tournament.source,
        )
        trainer = importlib.import_module(guided.GUIDED_TRAINER_MODULE)
        source_v2_state = capture_target_native_factor_state(transaction)
        live_binding = authenticate_guided_factor_sources(
            transaction,
            runner._guided_tranche,
            source_v2_state,
        )
        del source_v2_state
        gc.collect()
        rebound = rebind_recovered_v3_factor_payload(
            refreshed.factor_payload,
            live_binding,
        )
        set_stage = getattr(runner._guided_tranche, "set_training_stage", None)
        if not callable(set_stage):
            raise GuidedWinnerPublicationError("Guided tranche has no joint stage.")
        set_stage("joint")
        restore = getattr(trainer, "restore_target_native_factor_payload", None)
        if not callable(restore):
            raise GuidedWinnerPublicationError("Public full-v3 restore is unavailable.")
        restored = restore(
            transaction,
            rebound,
            guided_tranche=runner._guided_tranche,
            expected_guided_source_binding=live_binding,
        )
        if (
            getattr(restored, "schema", None) != BEHAVIOR_FACTOR_PACK_V3_SCHEMA
            or getattr(restored, "factor_state_sha256", None)
            != rebound["factor_state_sha256"]
            or portable_v3_factor_state_sha256(rebound, live_binding)
            != refreshed.winner["portable_factor_state_sha256"]
        ):
            raise GuidedWinnerPublicationError(
                "Live winner restore changed factor state."
            )
        report.update(
            {
                "compiler_report": compiler_report,
                "inherited_installation": inherited_install,
                "source_v2_restore": dict(source_restore),
                "guided_installation": dict(guided_install),
                "runtime_rebound_factor_state_sha256": rebound["factor_state_sha256"],
                "winner_portable_factor_state_sha256": refreshed.winner[
                    "portable_factor_state_sha256"
                ],
            }
        )

        candidate_metrics = runner.evaluate_publication_split(
            compiled.child_model,
            parent,
            refreshed,
            "candidate",
        )
        candidate_payload = stage._metrics_dict(candidate_metrics)
        if canonical_json_bytes(candidate_payload) != canonical_json_bytes(
            refreshed.winner["metrics"]
        ):
            raise GuidedWinnerPublicationError(
                "Restored winner candidate behavior differs from tournament evidence."
            )
        candidate_passed, candidate_decision = stage._behavior_gate(
            candidate_metrics,
            base.thresholds["candidate"],
        )
        method_passed, method_decision = _method_gate(
            candidate_metrics,
            winner_step=int(refreshed.winner["step"]),
            tournament_pareto_admission=True,
        )
        report["pre_sealed_candidate_decision"] = candidate_decision
        report["guided_method_decision"] = method_decision
        if not candidate_passed or not method_passed:
            runner.rollback_guided_installation(transaction)
            transaction.rollback()
            rollback_sha = runner.state_sha256(compiled.child_model)
            if rollback_sha != compiled_sha:
                raise GuidedWinnerPublicationError(
                    "Candidate rejection rollback failed."
                )
            report.update(
                {
                    "status": "rejected",
                    "reason": "winner-candidate-recheck-rejected",
                    "sealed_forward_count": 0,
                    "sealed_evaluation_accessed": False,
                    "sealed_marker_consumed": False,
                    "rollback_exact": True,
                    "rollback_state_sha256": rollback_sha,
                    "standalone_compaction_passed": False,
                    "model_checkpoint_written": False,
                }
            )
            stage._atomic_json(output / "run_report.json", report)
            return report

        if getattr(runner, "sealed_forward_count", 0) != 0:
            raise GuidedWinnerPublicationError("Sealed forward occurred before marker.")
        marker_receipt = _consume_sealed_marker(refreshed, output)
        sealed_metrics = runner.evaluate_publication_split(
            compiled.child_model,
            parent,
            refreshed,
            "sealed",
        )
        if getattr(runner, "sealed_forward_count", None) != 1:
            raise GuidedWinnerPublicationError(
                "Sealed forward count is not exactly one."
            )
        sealed_passed, sealed_decision = stage._behavior_gate(
            sealed_metrics,
            base.thresholds["sealed"],
        )
        pre_bake_decisions = {
            "fit": None,
            "candidate": candidate_decision,
            "sealed": sealed_decision,
        }
        report.update(
            {
                "sealed_evaluation": {
                    "forward_count": 1,
                    "one_shot_marker_consumed": True,
                    "marker_receipt": dict(marker_receipt),
                    "post_bake_sealed_forward_count": 0,
                },
                "pre_bake_behavior_decisions": pre_bake_decisions,
                "sealed_evaluation_accessed": True,
            }
        )
        if not sealed_passed:
            runner.rollback_guided_installation(transaction)
            transaction.rollback()
            rollback_sha = runner.state_sha256(compiled.child_model)
            if rollback_sha != compiled_sha:
                raise GuidedWinnerPublicationError("Sealed rejection rollback failed.")
            report.update(
                {
                    "status": "rejected",
                    "reason": "one-shot-sealed-gate-rejected",
                    "sealed_forward_count": 1,
                    "sealed_marker_consumed": True,
                    "rollback_exact": True,
                    "rollback_state_sha256": rollback_sha,
                    "standalone_compaction_passed": False,
                    "model_checkpoint_written": False,
                }
            )
            stage._atomic_json(output / "run_report.json", report)
            return report

        sealed_payload = stage._metrics_dict(sealed_metrics)
        gate_evidence = {
            "preregistration_sha256": refreshed.file_sha256,
            "winner_portable_factor_state_sha256": refreshed.winner[
                "portable_factor_state_sha256"
            ],
            "candidate_decision_sha256": candidate_decision["decision_sha256"],
            "method_decision_sha256": method_decision["decision_sha256"],
            "sealed_decision_sha256": sealed_decision["decision_sha256"],
            "sealed_marker_file_sha256": marker_receipt["marker_file_sha256"],
        }
        gate_evidence_sha = canonical_sha256(gate_evidence)
        receipt = ExternalBehaviorGateReceipt(
            evidence_sha256=gate_evidence_sha,
            evaluation_contract_sha256=refreshed.raw["semantic_contract_sha256"],
            heldout_action_count=int(sealed_payload["action_count"]),
            free_running_rollout_count=int(sealed_payload["rollout_count"]),
            passed=True,
            internal_similarity_only=False,
        )
        handoff = runner.handoff_guided_for_publication(
            transaction,
            refreshed.tournament.source,
        )
        publication = transaction.bake_for_publication(receipt)
        report.update(
            {
                "external_behavior_gate_evidence": gate_evidence,
                "external_behavior_gate_evidence_sha256": gate_evidence_sha,
                "guided_publication_handoff": dict(handoff),
                "publication": publication,
            }
        )

        post_decisions: dict[str, Any] = {
            "fit": None,
            "candidate": None,
            "sealed": None,
        }
        post_candidate_metrics: Any = None
        for name in ("fit", "candidate"):
            metrics = runner.evaluate_publication_split(
                compiled.child_model,
                parent,
                refreshed,
                name,
            )
            passed, decision = stage._behavior_gate(metrics, base.thresholds[name])
            post_decisions[name] = decision
            if not passed:
                transaction.restore_prepublication_native()
                rollback_sha = runner.state_sha256(compiled.child_model)
                if rollback_sha != compiled_sha:
                    raise GuidedWinnerPublicationError("Post-bake rollback failed.")
                report.update(
                    {
                        "status": "rejected",
                        "reason": f"post-bake-{name}-gate-rejected",
                        "post_bake_behavior_decisions": post_decisions,
                        "sealed_forward_count": 1,
                        "sealed_marker_consumed": True,
                        "rollback_exact": True,
                        "rollback_state_sha256": rollback_sha,
                        "standalone_compaction_passed": False,
                        "model_checkpoint_written": False,
                    }
                )
                stage._atomic_json(output / "run_report.json", report)
                return report
            if name == "candidate":
                post_candidate_metrics = metrics
        post_method_passed, post_method_decision = _method_gate(
            post_candidate_metrics,
            winner_step=int(refreshed.winner["step"]),
            tournament_pareto_admission=True,
        )
        report["post_bake_behavior_decisions"] = post_decisions
        report["post_bake_guided_method_decision"] = post_method_decision
        if not post_method_passed or getattr(runner, "sealed_forward_count", None) != 1:
            transaction.restore_prepublication_native()
            rollback_sha = runner.state_sha256(compiled.child_model)
            if rollback_sha != compiled_sha:
                raise GuidedWinnerPublicationError("Post-bake method rollback failed.")
            report.update(
                {
                    "status": "rejected",
                    "reason": "post-bake-guided-method-gate-rejected",
                    "sealed_forward_count": 1,
                    "sealed_marker_consumed": True,
                    "rollback_exact": True,
                    "rollback_state_sha256": rollback_sha,
                    "standalone_compaction_passed": False,
                    "model_checkpoint_written": False,
                }
            )
            stage._atomic_json(output / "run_report.json", report)
            return report

        certificate = runner.save_reload_certify(
            compiled.child_model,
            tokenizer,
            output,
        )
        published = True
        child_path = Path(str(certificate.get("path", ""))).expanduser().resolve()
        if child_path.parent != output or child_path.name != "accepted_child":
            raise GuidedWinnerPublicationError(
                "Fresh reload certificate does not bind the direct accepted child."
            )
        independent = runner.independent_reload_certify(
            child_path,
            stage.QWEN3_1P4B_EXACT_PARAMETERS,
        )
        independent_tree = independent.get("artifact_tree")
        certificate_tree = certificate.get("artifact_tree")
        if (
            certificate.get("fresh_reload_performed") is not True
            or certificate.get("exact_budget_match") is not True
            or certificate.get("exact_unique_parameters")
            != stage.QWEN3_1P4B_EXACT_PARAMETERS
            or certificate.get("state_keys_match_pre_save") is not True
            or certificate.get("state_bytes_match_pre_save") is not True
            or certificate.get("published_adapter_parameter_count") != 0
            or certificate.get("published_parent_runtime_reference_count") != 0
            or certificate.get("published_source_sized_sidecar_parameter_count") != 0
            or independent.get("fresh_reload_performed") is not True
            or independent.get("loaded_from_candidate_path") is not True
            or independent.get("exact_unique_parameters")
            != stage.QWEN3_1P4B_EXACT_PARAMETERS
            or independent.get("state_dict_sha256")
            != certificate.get("state_dict_sha256")
            or independent.get("published_adapter_parameter_count") != 0
            or independent.get("parent_runtime_reference_count") != 0
            or independent.get("source_sized_sidecar_parameter_count") != 0
            or not isinstance(independent_tree, Mapping)
            or not isinstance(certificate_tree, Mapping)
            or any(
                independent_tree.get(key) != certificate_tree.get(key)
                for key in ("file_count", "total_bytes", "tree_sha256")
            )
        ):
            raise GuidedWinnerPublicationError(
                "Fresh/independent reload did not prove a standalone exact child."
            )
        report.update(
            {
                "status": "accepted",
                "reason": None,
                "sealed_forward_count": 1,
                "sealed_marker_consumed": True,
                "post_bake_sealed_forward_count": 0,
                "standalone_compaction_passed": True,
                "model_checkpoint_written": True,
                "model_save_count": 1,
                "fresh_reload_certificate": certificate,
                "independent_reload_certificate": dict(independent),
                "fresh_reload_guided_method_gate": {
                    "passed": True,
                    "post_bake_decision_sha256": post_method_decision[
                        "decision_sha256"
                    ],
                    "state_bytes_match_pre_save": certificate[
                        "state_bytes_match_pre_save"
                    ],
                    "candidate_behavior_certified_before_save": True,
                    "serialized_bytes_equal_behavior_certified_bytes": True,
                    "internal_numeric_matching_consulted": False,
                },
                "rollback_exact": None,
                "retained_artifacts": ["accepted_child", "run_report.json"],
            }
        )
        stage._atomic_json(output / "run_report.json", report)
        return report
    except BaseException as error:
        rollback_error: str | None = None
        restored = False
        if transaction is not None:
            try:
                rollback_guided = getattr(runner, "rollback_guided_installation", None)
                if callable(rollback_guided):
                    rollback_guided(transaction)
                restore = getattr(transaction, "restore_prepublication_native", None)
                rollback = getattr(transaction, "rollback", None)
                if callable(restore):
                    restore()
                    restored = True
                elif callable(rollback) and getattr(transaction, "is_live", False):
                    rollback()
                    restored = True
                if (
                    restored
                    and compiled is not None
                    and compiled_sha is not None
                    and runner.state_sha256(compiled.child_model) != compiled_sha
                ):
                    raise GuidedWinnerPublicationError(
                        "Failure rollback changed child."
                    )
            except BaseException as rollback_failure:
                rollback_error = repr(rollback_failure)
        if published or (output / "accepted_child").exists():
            try:
                stage._remove_published_child_tree(output / "accepted_child", output)
            except BaseException as cleanup_failure:
                rollback_error = f"{rollback_error or ''}; {cleanup_failure!r}"
        failure = {
            **report,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "sealed_forward_count": getattr(runner, "sealed_forward_count", 0),
            "sealed_marker_consumed": marker_receipt is not None,
            "post_bake_sealed_forward_count": 0,
            "rollback_error": rollback_error,
            "prepublication_native_restored": restored,
            "standalone_compaction_passed": False,
            "model_checkpoint_written": False,
        }
        try:
            if not (output / "run_report.json").exists():
                stage._atomic_json(output / "run_report.json", failure)
        except BaseException:
            pass
        raise
    finally:
        parent = None
        target = None
        tokenizer = None
        compiled = None
        gc.collect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output-root", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--dtype", choices=("float16",), required=True)
    parser.add_argument("--parent-device-map", required=True)
    parser.add_argument("--target-device-map", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = authenticate_winner_publication_preregistration(
        args.preregistration,
        args.preregistration_sha256,
    )
    if args.preflight:
        if args.output_root is not None:
            raise SystemExit("--output-root is only valid with --execute.")
        cuda_preflight = tournament.validate_candidate_tournament_cuda_preflight(
            contract.tournament,
            dtype=args.dtype,
            parent_device_map=args.parent_device_map,
            target_device_map=args.target_device_map,
        )
        report = contract.preflight_report()
        report["cuda_preflight"] = cuda_preflight
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.output_root is None:
        raise SystemExit("--execute requires --output-root.")
    result = execute_winner_publication(
        contract,
        args.output_root,
        dtype=args.dtype,
        parent_device_map=args.parent_device_map,
        target_device_map=args.target_device_map,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
