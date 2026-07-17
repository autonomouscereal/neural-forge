#!/usr/bin/env python3
"""Run the real LFM2 residual-coordinate child-closed-loop fold.

This is a continuation of an already accepted blank-first structural fold.  It
does not create another target, rebuild the parent cache, execute the parent,
train, prune, quantize, or change the child topology.  The runner:

1. authenticates the dense smaller child and its persisted weight-complete or
   circuit-balanced residual basis;
2. replays every sealed token trajectory through the current child;
3. compiles each selected layer producer, recaptures the actual child context,
   fits its consumer, recaptures again, and compiles the downstream MLP;
4. accepts each stage only on held-out whole trajectories; and
5. writes a new ordinary dense checkpoint only after the whole transaction
   commits and a fresh local reload preserves parameter count and content.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import MethodType
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments import (  # noqa: E402
    refine_lfm2_dynamic_structural_residual_attention as refinement,
)
from experiments import run_lfm2_dynamic_structural_compression as baseline  # noqa: E402
from neural_forge.dynamic_closed_loop_fold import (  # noqa: E402
    ChildClosedLoopCompiler,
    ClosedLoopCompilerConfig,
)
from neural_forge.dynamic_balanced_circuit_basis import (  # noqa: E402
    BalancedCircuitBasisResult,
    _tensor_sha256 as _balanced_tensor_sha256,
)
from neural_forge.dynamic_weight_complete_basis import (  # noqa: E402
    WeightCompleteBasisResult,
    _tensor_sha256 as _weight_tensor_sha256,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_dynamic_closed_loop_adapter import (  # noqa: E402
    Lfm2DynamicClosedLoopAdapter,
    Lfm2DynamicClosedLoopAdapterConfig,
)


RUN_SCHEMA = "neural-forge-lfm2-weight-complete-closed-loop-run-v1"
STRUCTURAL_CONTINUATION_SCHEMA = (
    "neural-forge-lfm2-weight-complete-closed-loop-continuation-v1"
)


class Lfm2WeightCompleteClosedLoopError(RuntimeError):
    """An input, folding, transaction, or reload contract failed."""


class _ProgressAdapter:
    """Emit compact stage timing while preserving the adapter protocol."""

    _OBSERVED = frozenset(
        {
            "begin_child_transaction",
            "layer_indices",
            "parent_trajectory_records",
            "bridge_parent_endpoints",
            "replay_and_capture_child",
            "compile_producer",
            "install_producer",
            "fit_consumer",
            "install_consumer",
            "compile_downstream",
            "install_downstream",
            "validate_layer",
            "compile_final_normalization",
            "install_final_normalization",
            "validate_final_normalization",
            "compile_output_interface",
            "install_output_interface",
            "validate_output_interface",
        }
    )

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._delegate, name)
        if name not in self._OBSERVED or not callable(value):
            return value

        def observed(*args: Any, **kwargs: Any) -> Any:
            scope = next(
                (
                    getattr(argument, "scope_id", None)
                    for argument in args
                    if getattr(argument, "scope_id", None) is not None
                ),
                None,
            )
            point = next(
                (
                    getattr(argument, "value", None)
                    for argument in args
                    if type(argument).__name__ == "CapturePoint"
                ),
                None,
            )
            started = time.monotonic()
            print(
                json.dumps(
                    {
                        "event": "closed_loop_stage_started",
                        "method": name,
                        "scope": scope,
                        "capture_point": point,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )
            try:
                result = value(*args, **kwargs)
            except BaseException as exc:
                print(
                    json.dumps(
                        {
                            "event": "closed_loop_stage_failed",
                            "method": name,
                            "scope": scope,
                            "capture_point": point,
                            "elapsed_seconds": (
                                time.monotonic() - started
                            ),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    flush=True,
                )
                raise
            print(
                json.dumps(
                    {
                        "event": "closed_loop_stage_completed",
                        "method": name,
                        "scope": scope,
                        "capture_point": point,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )
            return result

        return observed


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Lfm2WeightCompleteClosedLoopError(
            f"Could not read JSON object {path}."
        ) from exc
    if not isinstance(value, Mapping):
        raise Lfm2WeightCompleteClosedLoopError(
            f"{path} must contain one JSON object."
        )
    return dict(value)


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
        raise Lfm2WeightCompleteClosedLoopError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _verified_baseline_contract(
    report: Mapping[str, Any],
) -> tuple[baseline.StructuralCandidate, str, str]:
    actual_run_sha = _digest(report.get("run_sha256"), "baseline run_sha256")
    expected_run_sha = baseline._json_sha256(
        {key: value for key, value in report.items() if key != "run_sha256"}
    )
    if actual_run_sha != expected_run_sha:
        raise Lfm2WeightCompleteClosedLoopError(
            "Baseline structural report seal is invalid."
        )
    candidate_payload = report.get("candidate")
    fold = report.get("fold")
    selection = report.get("residual_basis_selection")
    fresh = report.get("fresh_local_reload")
    if (
        not isinstance(candidate_payload, Mapping)
        or not isinstance(fold, Mapping)
        or not isinstance(selection, Mapping)
        or not isinstance(fresh, Mapping)
    ):
        raise Lfm2WeightCompleteClosedLoopError(
            "Baseline report lacks candidate, fold, basis, or reload contracts."
        )
    name = candidate_payload.get("name")
    candidate = baseline._CANDIDATES_BY_NAME.get(name)
    if candidate is None:
        raise Lfm2WeightCompleteClosedLoopError(
            "Baseline candidate is not in the verified structural ladder."
        )
    global_map = fold.get("global_residual_map")
    basis_mode = selection.get("mode")
    if basis_mode not in {"weight-complete", "circuit-balanced"}:
        raise Lfm2WeightCompleteClosedLoopError(
            "Baseline residual basis mode is unsupported."
        )
    if (
        selection.get("prompt_independent_whole_weight_basis") is not True
        or not isinstance(global_map, Mapping)
        or global_map.get("prompt_independent_weight_basis_override") is not True
        or global_map.get("dense_basis") is not True
        or global_map.get("sparse_assignment_used") is not False
        or report.get("dtype_or_packing_reduction_claimed") is not False
        or report.get("training_steps") != 0
        or report.get("optimizer_steps") != 0
        or report.get("backward_calls") != 0
        or report.get("hub_loader_used") is not False
        or fresh.get("verified") is not True
    ):
        raise Lfm2WeightCompleteClosedLoopError(
            "Input is not an accepted local dense residual-coordinate fold."
        )
    if basis_mode == "circuit-balanced" and (
        selection.get("cached_residual_controllability") is not True
        or selection.get("downstream_observability_balancing") is not True
        or not isinstance(selection.get("circuit_balanced_report"), Mapping)
    ):
        raise Lfm2WeightCompleteClosedLoopError(
            "Circuit-balanced baseline lacks its controllability/observability "
            "contract."
        )
    map_sha256 = _digest(
        global_map.get("map_sha256"), "global residual map_sha256"
    )
    state_sha256 = _digest(
        fold.get("folded_state_sha256"), "folded state_sha256"
    )
    if fresh.get("state_sha256") != state_sha256:
        raise Lfm2WeightCompleteClosedLoopError(
            "Baseline fresh reload does not bind its folded state."
        )
    return candidate, map_sha256, state_sha256


def _load_weight_complete_basis(
    child_directory: Path,
    report: Mapping[str, Any],
    *,
    expected_map_sha256: str,
) -> tuple[WeightCompleteBasisResult, dict[str, Any]]:
    from safetensors import safe_open

    selection = report.get("residual_basis_selection")
    if not isinstance(selection, Mapping):
        raise Lfm2WeightCompleteClosedLoopError(
            "Baseline has no residual basis selection."
        )
    artifact = selection.get("artifact")
    wrapper = selection.get("weight_complete_report")
    if not isinstance(artifact, Mapping) or not isinstance(wrapper, Mapping):
        raise Lfm2WeightCompleteClosedLoopError(
            "Baseline has no persisted weight-complete basis artifact."
        )
    basis_report = wrapper.get("basis")
    if not isinstance(basis_report, Mapping):
        raise Lfm2WeightCompleteClosedLoopError(
            "Weight-complete report has no generic basis contract."
        )
    filename = artifact.get("file")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
    ):
        raise Lfm2WeightCompleteClosedLoopError(
            "Basis artifact must be one local checkpoint filename."
        )
    path = child_directory / filename
    if not path.is_file() or path.is_symlink():
        raise Lfm2WeightCompleteClosedLoopError(
            "Persisted residual-coordinate basis is missing or unsafe."
        )
    file_sha256 = _file_sha256(path)
    if file_sha256 != _digest(
        artifact.get("file_sha256"), "basis artifact file_sha256"
    ):
        raise Lfm2WeightCompleteClosedLoopError(
            "Persisted residual-coordinate basis bytes changed."
        )
    expected_keys = {
        "basis",
        "full_weight_gram",
        "balanced_structural_gram",
        "combined_gram",
    }
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != expected_keys:
            raise Lfm2WeightCompleteClosedLoopError(
                "Basis artifact tensor inventory changed."
            )
        tensors = {name: handle.get_tensor(name) for name in expected_keys}
    hashes = basis_report.get("hashes")
    if not isinstance(hashes, Mapping):
        raise Lfm2WeightCompleteClosedLoopError(
            "Basis report has no tensor hashes."
        )
    expected_hashes = {
        "basis": "basis_sha256",
        "full_weight_gram": "full_weight_gram_sha256",
        "balanced_structural_gram": "balanced_structural_gram_sha256",
        "combined_gram": "combined_gram_sha256",
    }
    for tensor_name, report_name in expected_hashes.items():
        if _weight_tensor_sha256(tensors[tensor_name]) != _digest(
            hashes.get(report_name), f"basis {report_name}"
        ):
            raise Lfm2WeightCompleteClosedLoopError(
                f"Basis tensor {tensor_name!r} does not match its report."
            )
    result = WeightCompleteBasisResult(
        basis=tensors["basis"],
        full_weight_gram=tensors["full_weight_gram"],
        balanced_structural_gram=tensors["balanced_structural_gram"],
        combined_gram=tensors["combined_gram"],
        report=basis_report,
    )
    if (
        result.map_sha256 != expected_map_sha256
        or artifact.get("basis_sha256") != expected_map_sha256
        or artifact.get("source_width") != result.source_width
        or artifact.get("target_width") != result.target_width
        or artifact.get("contains_complete_objective_grams") is not True
    ):
        raise Lfm2WeightCompleteClosedLoopError(
            "Basis artifact, report, and folded child map bindings disagree."
        )
    return result, {
        "mode": "weight-complete",
        "file": filename,
        "file_sha256": file_sha256,
        "basis_sha256": result.map_sha256,
        "source_width": result.source_width,
        "target_width": result.target_width,
        "contains_complete_objective_grams": True,
    }


def _load_circuit_balanced_basis(
    child_directory: Path,
    report: Mapping[str, Any],
    *,
    expected_map_sha256: str,
) -> tuple[BalancedCircuitBasisResult, dict[str, Any]]:
    """Reauthenticate the complete persisted balanced realization."""

    from safetensors import safe_open

    selection = report.get("residual_basis_selection")
    if not isinstance(selection, Mapping):
        raise Lfm2WeightCompleteClosedLoopError(
            "Baseline has no residual basis selection."
        )
    artifact = selection.get("artifact")
    wrapper = selection.get("circuit_balanced_report")
    if not isinstance(artifact, Mapping) or not isinstance(wrapper, Mapping):
        raise Lfm2WeightCompleteClosedLoopError(
            "Baseline has no persisted circuit-balanced basis artifact."
        )
    basis_report = wrapper.get("balanced_basis")
    if not isinstance(basis_report, Mapping):
        raise Lfm2WeightCompleteClosedLoopError(
            "Circuit-balanced report has no generic basis contract."
        )
    filename = artifact.get("file")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
    ):
        raise Lfm2WeightCompleteClosedLoopError(
            "Basis artifact must be one local checkpoint filename."
        )
    path = child_directory / filename
    if not path.is_file() or path.is_symlink():
        raise Lfm2WeightCompleteClosedLoopError(
            "Persisted residual-coordinate basis is missing or unsafe."
        )
    file_sha256 = _file_sha256(path)
    if file_sha256 != _digest(
        artifact.get("file_sha256"), "basis artifact file_sha256"
    ):
        raise Lfm2WeightCompleteClosedLoopError(
            "Persisted residual-coordinate basis bytes changed."
        )
    expected_keys = {
        "basis",
        "balanced_encoder",
        "balanced_decoder",
        "full_weight_gram",
        "controllability_gram",
        "observability_gram",
        "heldout_gram",
    }
    if (
        artifact.get("tensor_names") != sorted(expected_keys)
        or artifact.get("contains_complete_objective_grams") is not True
        or artifact.get("contains_balanced_encoder_decoder") is not True
        or artifact.get("fresh_tensor_reload_verified") is not True
    ):
        raise Lfm2WeightCompleteClosedLoopError(
            "Circuit-balanced artifact inventory contract changed."
        )
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != expected_keys:
            raise Lfm2WeightCompleteClosedLoopError(
                "Circuit-balanced artifact tensor inventory changed."
            )
        tensors = {name: handle.get_tensor(name) for name in expected_keys}
    artifact_hashes = artifact.get("tensor_sha256")
    basis_hashes = basis_report.get("hashes")
    if not isinstance(artifact_hashes, Mapping) or not isinstance(
        basis_hashes, Mapping
    ):
        raise Lfm2WeightCompleteClosedLoopError(
            "Circuit-balanced artifact has no tensor-hash contracts."
        )
    report_hash_names = {
        "basis": "basis_sha256",
        "balanced_encoder": "balanced_encoder_sha256",
        "balanced_decoder": "balanced_decoder_sha256",
        "full_weight_gram": "full_weight_gram_sha256",
        "controllability_gram": "controllability_gram_sha256",
        "observability_gram": "observability_gram_sha256",
        "heldout_gram": "heldout_gram_sha256",
    }
    for tensor_name, report_name in report_hash_names.items():
        if baseline._tensor_sha256(tensors[tensor_name]) != _digest(
            artifact_hashes.get(tensor_name),
            f"artifact tensor {tensor_name} sha256",
        ):
            raise Lfm2WeightCompleteClosedLoopError(
                f"Artifact tensor {tensor_name!r} changed after publication."
            )
        if _balanced_tensor_sha256(tensors[tensor_name]) != _digest(
            basis_hashes.get(report_name), f"basis {report_name}"
        ):
            raise Lfm2WeightCompleteClosedLoopError(
                f"Balanced tensor {tensor_name!r} does not match its report."
            )
    result = BalancedCircuitBasisResult(
        basis=tensors["basis"],
        balanced_encoder=tensors["balanced_encoder"],
        balanced_decoder=tensors["balanced_decoder"],
        full_weight_gram=tensors["full_weight_gram"],
        controllability_gram=tensors["controllability_gram"],
        observability_gram=tensors["observability_gram"],
        heldout_gram=tensors["heldout_gram"],
        report=basis_report,
    )
    if (
        result.map_sha256 != expected_map_sha256
        or artifact.get("basis_sha256") != expected_map_sha256
        or artifact.get("source_width") != result.source_width
        or artifact.get("target_width") != result.target_width
        or artifact.get("reloaded_basis_tensor_sha256")
        != artifact_hashes["basis"]
    ):
        raise Lfm2WeightCompleteClosedLoopError(
            "Balanced artifact, report, and folded child map bindings disagree."
        )
    return result, {
        "mode": "circuit-balanced",
        "file": filename,
        "file_sha256": file_sha256,
        "basis_sha256": result.map_sha256,
        "source_width": result.source_width,
        "target_width": result.target_width,
        "tensor_names": sorted(expected_keys),
        "contains_complete_objective_grams": True,
        "contains_balanced_encoder_decoder": True,
    }


def _load_residual_coordinate_map(
    child_directory: Path,
    report: Mapping[str, Any],
    *,
    expected_map_sha256: str,
) -> tuple[
    WeightCompleteBasisResult | BalancedCircuitBasisResult,
    dict[str, Any],
]:
    selection = report.get("residual_basis_selection")
    mode = selection.get("mode") if isinstance(selection, Mapping) else None
    if mode == "weight-complete":
        return _load_weight_complete_basis(
            child_directory,
            report,
            expected_map_sha256=expected_map_sha256,
        )
    if mode == "circuit-balanced":
        return _load_circuit_balanced_basis(
            child_directory,
            report,
            expected_map_sha256=expected_map_sha256,
        )
    raise Lfm2WeightCompleteClosedLoopError(
        "Baseline residual basis mode is unsupported."
    )


def _parse_dtype(value: str) -> torch.dtype:
    values = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return values[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "dtype must be float16, bfloat16, or float32"
        ) from exc


def execute(args: argparse.Namespace) -> dict[str, Any]:
    parent_directory = baseline._local_model_directory(args.parent_model)
    child_directory = baseline._local_model_directory(args.baseline_child)
    output_directory = Path(args.output).expanduser().absolute()
    if output_directory.exists():
        raise Lfm2WeightCompleteClosedLoopError(
            f"Output checkpoint already exists: {output_directory}."
        )
    structural_path = child_directory / "structural-fold-report.json"
    baseline_report = _read_json(structural_path)
    candidate, expected_map_sha, expected_child_state = (
        _verified_baseline_contract(baseline_report)
    )
    coordinate_map, basis_artifact = _load_residual_coordinate_map(
        child_directory,
        baseline_report,
        expected_map_sha256=expected_map_sha,
    )

    device = torch.device(args.device)
    parent = baseline.load_local_lfm2(
        parent_directory, device=device, dtype=args.dtype
    )
    child = baseline.load_local_lfm2(
        child_directory, device=device, dtype=args.dtype
    )
    parent_fingerprint = fingerprint_lfm2_parent(parent)
    baseline_parent = baseline_report.get("parent")
    if (
        not isinstance(baseline_parent, Mapping)
        or baseline_parent.get("identity_sha256")
        != parent_fingerprint.identity_sha256
    ):
        raise Lfm2WeightCompleteClosedLoopError(
            "Loaded parent is not the parent bound to the structural child."
        )
    parent_state_before = baseline.model_state_sha256(parent)
    child_state_before = baseline.model_state_sha256(child)
    tied_output_parameter_before = child.get_input_embeddings().weight
    if (
        tied_output_parameter_before
        is not child.get_output_embeddings().weight
    ):
        raise Lfm2WeightCompleteClosedLoopError(
            "The child input embedding and output head are not tied."
        )
    tied_output_sha256_before = _balanced_tensor_sha256(
        tied_output_parameter_before.detach()
    )
    if child_state_before != expected_child_state:
        raise Lfm2WeightCompleteClosedLoopError(
            "Loaded child bytes differ from the accepted structural fold."
        )
    baseline.validate_live_structural_topology(parent, child, candidate)
    child_dense_before = baseline.dense_logical_parameter_audit(
        child, require_any_nonzero=True
    )
    if (
        child_dense_before["unique_logical_parameters"]
        != candidate.verified_unique_parameters
    ):
        raise Lfm2WeightCompleteClosedLoopError(
            "Loaded child does not have the verified dense target size."
        )
    cache = open_lfm2_circuit_trace_cache(args.trace_cache, verify=True)
    baseline_cache = baseline_report.get("cache")
    if (
        cache.parent_fingerprint != parent_fingerprint.identity_sha256
        or not isinstance(baseline_cache, Mapping)
        or baseline_cache.get("cache_key") != cache.cache_key
        or baseline_cache.get("corpus_sha256") != cache.corpus_fingerprint
    ):
        raise Lfm2WeightCompleteClosedLoopError(
            "Sealed cache is not the exact parent/corpus bound to the child."
        )
    selected_layers = refinement.parse_layer_selection(
        args.layers, len(baseline._lfm2_layers(parent))
    )

    adapter = Lfm2DynamicClosedLoopAdapter(
        source_model=parent,
        child_model=child,
        parent_cache=cache,
        coordinate_map=coordinate_map,
        initial_child_basis_sha256=expected_map_sha,
        selected_layer_indices=selected_layers,
        input_device=device,
        config=Lfm2DynamicClosedLoopAdapterConfig(
            ridge=args.ridge,
            norm_ridge=args.norm_ridge,
            max_prior_delta_ratio=args.max_prior_delta_ratio,
            cluster_ridge=args.cluster_ridge,
            cluster_max_iterations=args.cluster_max_iterations,
            cluster_distance_chunk_size=args.cluster_distance_chunk_size,
            minimum_layer_relative_rmse_improvement=(
                args.minimum_layer_relative_rmse_improvement
            ),
            minimum_final_norm_relative_rmse_improvement=(
                args.minimum_final_norm_relative_rmse_improvement
            ),
            output_ridge=args.output_ridge,
            output_max_prior_delta_ratio=(
                args.output_max_prior_delta_ratio
            ),
            output_max_transform_condition_number=(
                args.output_max_transform_condition_number
            ),
            output_min_transform_singular_value=(
                args.output_min_transform_singular_value
            ),
            minimum_output_kl_improvement=(
                args.minimum_output_kl_improvement
            ),
            minimum_output_known_sequence_nll_improvement=(
                args.minimum_output_known_sequence_nll_improvement
            ),
            maximum_output_eos_margin_rmse_regression=(
                args.maximum_output_eos_margin_rmse_regression
            ),
            output_validation_tolerance=(
                args.output_validation_tolerance
            ),
            output_metric_chunk_rows=args.output_metric_chunk_rows,
            validation_tolerance=args.validation_tolerance,
            replay_capture_mode=args.replay_capture_mode,
            attention_consumer_mode=args.attention_consumer_mode,
            attention_value_output_iterations=(
                args.attention_value_output_iterations
            ),
            attention_value_output_context_ridge=(
                args.attention_value_output_context_ridge
            ),
            attention_value_output_fit_tolerance=(
                args.attention_value_output_fit_tolerance
            ),
            attention_value_output_context_equivalence_tolerance=(
                args.attention_value_output_context_equivalence_tolerance
            ),
            output_interface_mode=(
                "identity-skip"
                if args.preserve_output_interface
                else "fit-transform"
            ),
        ),
    )
    parent_forward_calls = 0
    original_parent_forward = parent.forward

    def forbidden_parent_forward(
        _self: Any, *_positional: Any, **_keyword: Any
    ) -> Any:
        nonlocal parent_forward_calls
        parent_forward_calls += 1
        raise Lfm2WeightCompleteClosedLoopError(
            "Parent forward is forbidden; use the sealed cache."
        )

    parent.forward = MethodType(forbidden_parent_forward, parent)
    try:
        compiler_report = ChildClosedLoopCompiler(
            adapter if args.quiet_progress else _ProgressAdapter(adapter),
            ClosedLoopCompilerConfig(
                validation_fraction=args.validation_fraction,
                split_seed=args.split_seed,
                fold_output_interface=True,
            ),
        ).run()
    finally:
        parent.forward = original_parent_forward
    if parent_forward_calls:
        raise Lfm2WeightCompleteClosedLoopError(
            "Closed-loop folding attempted a parent forward."
        )
    parent_state_after = baseline.model_state_sha256(parent)
    if parent_state_after != parent_state_before:
        raise Lfm2WeightCompleteClosedLoopError(
            "Closed-loop folding mutated the parent weights or buffers."
        )
    child_state_after = baseline.model_state_sha256(child)
    tied_output_parameter_after = child.get_input_embeddings().weight
    tied_output_sha256_after = _balanced_tensor_sha256(
        tied_output_parameter_after.detach()
    )
    if args.preserve_output_interface and (
        tied_output_parameter_after
        is not child.get_output_embeddings().weight
        or tied_output_sha256_after != tied_output_sha256_before
    ):
        raise Lfm2WeightCompleteClosedLoopError(
            "Explicit output-interface preservation changed the tied "
            "embedding/head Parameter."
        )
    child_dense_after = baseline.dense_logical_parameter_audit(
        child, require_any_nonzero=True
    )
    if child_dense_after != child_dense_before:
        raise Lfm2WeightCompleteClosedLoopError(
            "Closed-loop folding changed the dense parameter-count contract."
        )
    audit = compiler_report.get("audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("parent_forward_calls") != 0
        or audit.get("parent_trace_regeneration_calls") != 0
        or audit.get("training_steps") != 0
        or audit.get("backward_calls") != 0
        or audit.get("optimizer_steps") != 0
        or audit.get("torch_fold_only_guard_passed") is not True
        or compiler_report.get("transaction_committed") is not True
        or compiler_report.get("child_topology_preserved") is not True
    ):
        raise Lfm2WeightCompleteClosedLoopError(
            "Committed compiler report violates fold-only invariants."
        )

    structural_report = copy.deepcopy(baseline_report)
    structural_report.pop("run_sha256", None)
    structural_fold = structural_report.get("fold")
    if not isinstance(structural_fold, dict):
        raise Lfm2WeightCompleteClosedLoopError(
            "Structural fold report cannot record the continuation."
        )
    structural_fold["folded_state_sha256"] = child_state_after
    continuation = {
        "schema": STRUCTURAL_CONTINUATION_SCHEMA,
        "baseline_residual_basis_mode": basis_artifact["mode"],
        "baseline_structural_run_sha256": baseline_report["run_sha256"],
        "baseline_child_state_sha256": child_state_before,
        "result_child_state_sha256": child_state_after,
        "coordinate_map_sha256": expected_map_sha,
        "selected_layers": list(selected_layers),
        "all_model_layers_selected": len(selected_layers)
        == len(baseline._lfm2_layers(parent)),
        "compiler_run_sha256": compiler_report["run_sha256"],
        "cache_key": cache.cache_key,
        "sealed_batches_used": len(cache),
        "heldout_whole_trajectory_validation": True,
        "actual_child_recapture_between_stages": True,
        "replay_capture_mode": args.replay_capture_mode,
        "parent_forward_calls": 0,
        "parent_trace_regeneration_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "semantic_promotion_status": (
            "unpromoted_pending_external_whole_output_gate"
        ),
    }
    structural_fold["residual_coordinate_child_closed_loop"] = continuation
    if basis_artifact["mode"] == "weight-complete":
        structural_fold["weight_complete_child_closed_loop"] = continuation
    structural_report["fresh_local_reload"] = {
        "verified": False,
        "pending_closed_loop_checkpoint_reload": True,
    }
    structural_report["run_sha256"] = baseline._json_sha256(
        structural_report
    )
    reloaded, reload_report = refinement.save_and_fresh_reload(
        child,
        source_directory=child_directory,
        destination=output_directory,
        report=structural_report,
        device=device,
        dtype=args.dtype,
        expected_unique_parameters=candidate.verified_unique_parameters,
    )
    destination_basis = output_directory / basis_artifact["file"]
    shutil.copy2(
        child_directory / basis_artifact["file"], destination_basis
    )
    if _file_sha256(destination_basis) != basis_artifact["file_sha256"]:
        raise Lfm2WeightCompleteClosedLoopError(
            "Copied residual-coordinate basis bytes changed."
        )
    reload_topology = baseline.validate_live_structural_topology(
        parent, reloaded, candidate
    )
    reload_report["topology"] = reload_topology
    reload_report["basis_artifact_file_sha256"] = basis_artifact[
        "file_sha256"
    ]
    structural_report["fresh_local_reload"] = reload_report
    structural_report["run_sha256"] = baseline._json_sha256(
        {
            key: value
            for key, value in structural_report.items()
            if key != "run_sha256"
        }
    )
    (output_directory / "structural-fold-report.json").write_bytes(
        baseline._canonical_json(structural_report) + b"\n"
    )

    report: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "baseline_checkpoint": {
            "directory": str(child_directory),
            "structural_report_file_sha256": _file_sha256(
                structural_path
            ),
            "structural_run_sha256": baseline_report["run_sha256"],
            "child_state_sha256": child_state_before,
            "complete_blank_first_fold": True,
        },
        "parent": {
            **parent_fingerprint.as_dict(),
            "state_sha256_before": parent_state_before,
            "state_sha256_after": parent_state_after,
            "immutable": True,
            "forward_calls": parent_forward_calls,
        },
        "cache": {
            "directory": str(cache.path),
            "cache_key": cache.cache_key,
            "corpus_sha256": cache.corpus_fingerprint,
            "sealed_batches_used": len(cache),
            "parent_forward_calls": 0,
            "parent_trace_regeneration_calls": 0,
        },
        "coordinate_map": basis_artifact,
        "selection": {
            "selected_layers": list(selected_layers),
            "all_model_layers_selected": len(selected_layers)
            == len(baseline._lfm2_layers(parent)),
            "validation_fraction": args.validation_fraction,
            "split_seed": args.split_seed,
            "preserve_output_interface": args.preserve_output_interface,
        },
        "adapter_config": {
            "ridge": args.ridge,
            "norm_ridge": args.norm_ridge,
            "max_prior_delta_ratio": args.max_prior_delta_ratio,
            "cluster_ridge": args.cluster_ridge,
            "cluster_max_iterations": args.cluster_max_iterations,
            "cluster_distance_chunk_size": (
                args.cluster_distance_chunk_size
            ),
            "minimum_layer_relative_rmse_improvement": (
                args.minimum_layer_relative_rmse_improvement
            ),
            "minimum_final_norm_relative_rmse_improvement": (
                args.minimum_final_norm_relative_rmse_improvement
            ),
            "output_ridge": args.output_ridge,
            "output_max_prior_delta_ratio": (
                args.output_max_prior_delta_ratio
            ),
            "output_max_transform_condition_number": (
                args.output_max_transform_condition_number
            ),
            "output_min_transform_singular_value": (
                args.output_min_transform_singular_value
            ),
            "minimum_output_kl_improvement": (
                args.minimum_output_kl_improvement
            ),
            "minimum_output_known_sequence_nll_improvement": (
                args.minimum_output_known_sequence_nll_improvement
            ),
            "maximum_output_eos_margin_rmse_regression": (
                args.maximum_output_eos_margin_rmse_regression
            ),
            "output_validation_tolerance": (
                args.output_validation_tolerance
            ),
            "output_metric_chunk_rows": args.output_metric_chunk_rows,
            "validation_tolerance": args.validation_tolerance,
            "replay_capture_mode": args.replay_capture_mode,
            "attention_consumer_mode": args.attention_consumer_mode,
            "attention_value_output_iterations": (
                args.attention_value_output_iterations
            ),
            "attention_value_output_context_ridge": (
                args.attention_value_output_context_ridge
            ),
            "attention_value_output_fit_tolerance": (
                args.attention_value_output_fit_tolerance
            ),
            "attention_value_output_context_equivalence_tolerance": (
                args.attention_value_output_context_equivalence_tolerance
            ),
            "output_interface_mode": (
                "identity-skip"
                if args.preserve_output_interface
                else "fit-transform"
            ),
        },
        "compiler": compiler_report,
        "child": {
            "state_sha256_before": child_state_before,
            "state_sha256_after": child_state_after,
            "content_changed": child_state_after != child_state_before,
            "dense_parameter_audit": child_dense_after,
            "exact_unique_logical_parameters": candidate.verified_unique_parameters,
            "parent_unique_logical_parameters": (
                baseline.EXPECTED_PARENT_UNIQUE_PARAMETERS
            ),
            "realized_logical_parameter_reduction_fraction": (
                1.0
                - candidate.verified_unique_parameters
                / baseline.EXPECTED_PARENT_UNIQUE_PARAMETERS
            ),
            "dtype_or_packing_reduction_claimed": False,
            "topology_changed_during_closed_loop": False,
            "target_architecture_differs_from_parent": True,
            "tied_output_interface": {
                "preserved_by_explicit_nonexecution": (
                    args.preserve_output_interface
                ),
                "parameter_identity_preserved": (
                    tied_output_parameter_after
                    is child.get_output_embeddings().weight
                ),
                "sha256_before": tied_output_sha256_before,
                "sha256_after": tied_output_sha256_after,
                "bytes_preserved": (
                    tied_output_sha256_after == tied_output_sha256_before
                ),
            },
        },
        "fresh_local_reload": reload_report,
        "promotion": {
            "status": "unpromoted_pending_external_whole_output_gate",
            "transaction_commit_meaning": (
                "candidate_checkpoint_construction_only"
            ),
            "external_whole_output_behavior_gate_required": True,
            "manual_coherence_review_required": True,
        },
        "output_checkpoint": str(output_directory),
        "resulting_structural_fold_run_sha256": structural_report[
            "run_sha256"
        ],
        "parent_forward_calls": 0,
        "parent_trace_regeneration_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "hub_loader_used": False,
        "network_allowed": False,
        "dtype_or_packing_reduction_claimed": False,
    }
    report["run_sha256"] = baseline._json_sha256(report)
    (output_directory / "closed-loop-fold-report.json").write_bytes(
        baseline._canonical_json(report) + b"\n"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model", required=True)
    parser.add_argument("--baseline-child", required=True)
    parser.add_argument("--trace-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--layers",
        default="all",
        help="all, a comma list (0,2,4), or inclusive ranges (0-4)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", type=_parse_dtype, default=torch.bfloat16
    )
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument(
        "--split-seed",
        default="lfm2-weight-complete-closed-loop-v1",
    )
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--norm-ridge", type=float, default=1e-6)
    parser.add_argument(
        "--max-prior-delta-ratio", type=float, default=1.0
    )
    parser.add_argument("--cluster-ridge", type=float, default=1e-8)
    parser.add_argument(
        "--cluster-max-iterations", type=int, default=16
    )
    parser.add_argument(
        "--cluster-distance-chunk-size", type=int, default=512
    )
    parser.add_argument(
        "--minimum-layer-relative-rmse-improvement",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--minimum-final-norm-relative-rmse-improvement",
        type=float,
        default=0.0,
    )
    parser.add_argument("--output-ridge", type=float, default=1e-3)
    parser.add_argument(
        "--output-max-prior-delta-ratio", type=float, default=0.25
    )
    parser.add_argument(
        "--preserve-output-interface",
        action="store_true",
        help=(
            "Keep the tied input embedding/output head byte-identical and "
            "defer acceptance to the external whole-output behavior gate."
        ),
    )
    parser.add_argument(
        "--output-max-transform-condition-number",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--output-min-transform-singular-value",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--minimum-output-kl-improvement", type=float, default=0.0
    )
    parser.add_argument(
        "--minimum-output-known-sequence-nll-improvement",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--maximum-output-eos-margin-rmse-regression",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--output-validation-tolerance", type=float, default=1e-8
    )
    parser.add_argument(
        "--output-metric-chunk-rows", type=int, default=16
    )
    parser.add_argument(
        "--validation-tolerance", type=float, default=1e-8
    )
    parser.add_argument(
        "--attention-consumer-mode",
        choices=("output-only", "coupled-value-output"),
        default="output-only",
    )
    parser.add_argument(
        "--attention-value-output-iterations",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--attention-value-output-context-ridge",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--attention-value-output-fit-tolerance",
        type=float,
        default=1e-10,
    )
    parser.add_argument(
        "--attention-value-output-context-equivalence-tolerance",
        type=float,
        default=5e-2,
    )
    parser.add_argument(
        "--replay-capture-mode",
        choices=(
            CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
            CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        ),
        default=CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
        help=(
            "Prefix-by-prefix is the canonical fallback. single_pass_causal "
            "is an explicit, runtime-contract-gated acceleration."
        ),
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Suppress compact JSON stage timing on stdout.",
    )
    return parser


def main() -> None:
    report = execute(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "run_sha256": report["run_sha256"],
                "selected_layers": report["selection"]["selected_layers"],
                "child": report["child"],
                "fresh_local_reload": report["fresh_local_reload"],
                "output": report["output_checkpoint"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
