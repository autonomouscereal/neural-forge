#!/usr/bin/env python3
"""Run one non-promotable real H776 CPCF-v5 layer-2 attention probe.

This entry point opens an independently pinned k=2 prefix produced by the
older compiler runtime, reconstructs its actual-child causal state bank, runs
the current attention compiler exactly once for layer 2, and stops before k=3.
It cannot publish or save a child and does not weaken ordinary resume gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

# These must precede Transformers and CUDA imports.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


PROBE_RUN_SCHEMA = "neural-forge-lfm2-cpcf-v5-real-attention-probe-run-v1"
PROBE_FAILURE_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-real-attention-probe-failure-v1"
)
_GIB = 1024**3


class Lfm2CpcfV5RealAttentionProbeRunnerError(RuntimeError):
    """The strict local real-attention probe failed closed."""


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
        raise Lfm2CpcfV5RealAttentionProbeRunnerError(
            "Probe diagnostics must be finite canonical JSON."
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CpcfV5RealAttentionProbeRunnerError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _plain(value: Any) -> Any:
    return json.loads(_canonical_json(value).decode("ascii"))


def _local_lexical_path(value: Path, name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be pathlib.Path.")
    raw = os.fspath(value.expanduser())
    if not raw or "://" in raw or raw.startswith(("\\\\", "//")):
        raise Lfm2CpcfV5RealAttentionProbeRunnerError(
            f"{name} must be an explicit local path."
        )
    return value.expanduser().absolute().resolve(strict=False)


def _validate_diagnostic_destination(
    diagnostic: Path,
    *,
    protected: Mapping[str, Path | None],
) -> Path:
    """Authenticate a create-new destination without creating any parent."""

    destination = _local_lexical_path(diagnostic, "diagnostic_json")
    overlaps = []
    for name, raw in protected.items():
        if raw is None:
            continue
        path = _local_lexical_path(raw, name)
        if (
            destination == path
            or destination in path.parents
            or path in destination.parents
        ):
            overlaps.append(name)
    if overlaps:
        raise Lfm2CpcfV5RealAttentionProbeRunnerError(
            "diagnostic_json must be disjoint from every input and immutable "
            "store: " + ", ".join(sorted(overlaps)) + "."
        )
    if destination.exists() or destination.is_symlink():
        raise Lfm2CpcfV5RealAttentionProbeRunnerError(
            "diagnostic_json already exists; refusing to overwrite it."
        )
    return destination


def validate_diagnostic_destination_args(
    args: argparse.Namespace,
) -> Path:
    """Validate the failure destination before any catch block may persist."""

    return _validate_diagnostic_destination(
        args.diagnostic_json,
        protected={
            "parent_model_directory": args.parent_model_directory,
            "donor_tokenizer_directory": args.donor_tokenizer_directory,
            "target_tokenizer_directory": args.target_tokenizer_directory,
            "bridge_directory": args.bridge_directory,
            "trace_cache_directory": args.trace_cache_directory,
            "trajectory_corpus": args.trajectory_corpus,
            "checkpoint_store_directory": args.resume_store_directory,
            "selective_catalog_store_directory": (
                args.selective_catalog_store_directory
            ),
            "selective_catalog_receipt_json": (
                args.selective_catalog_receipt_json
            ),
        },
    )


def _v5_runner() -> Any:
    from experiments import run_lfm2_cpcf_v5_selective_blank_fold

    return run_lfm2_cpcf_v5_selective_blank_fold


def _base_runner() -> Any:
    return _v5_runner()._base_runner()


@dataclass(frozen=True, slots=True)
class RealAttentionProbeConfig:
    """The ordinary local setup plus independent cross-runtime artifact pins."""

    fold: Any
    checkpoint_key: str
    checkpoint_manifest_sha256: str
    expected_trace_sha256: str
    expected_runtime_compiler_implementation_sha256: str
    expected_sequence_source_sha256: str

    def __post_init__(self) -> None:
        from neural_forge.hf_lfm2_cpcf_v5_real_attention_probe import (
            H776_MAXIMUM_CUDA_BYTES_PER_DEVICE,
            H776_MAXIMUM_PROCESS_RSS_BYTES,
            H776_ORIGIN_COMPILER_SHA256,
            H776_TARGET_PLAN_FINGERPRINT,
        )

        fold = self.fold
        compression = fold.compression
        changed = []
        if compression.compression_ratio != "4":
            changed.append("compression_ratio")
        if compression.target_unique_parameters is not None:
            changed.append("target_unique_parameters")
        if fold.selected_candidate_rank is not None:
            changed.append("selected_candidate_rank")
        if (
            fold.expected_target_plan_fingerprint
            != H776_TARGET_PLAN_FINGERPRINT
        ):
            changed.append("expected_target_plan_fingerprint")
        if fold.maximum_process_rss_bytes != H776_MAXIMUM_PROCESS_RSS_BYTES:
            changed.append("maximum_process_rss_gib")
        if (
            fold.maximum_cuda_bytes_per_device
            != H776_MAXIMUM_CUDA_BYTES_PER_DEVICE
        ):
            changed.append("maximum_cuda_gib_per_device")
        if fold.model_device != "cuda:0":
            changed.append("model_device")
        if fold.dtype != "bfloat16":
            changed.append("dtype")
        if fold.resume_store_directory is None:
            changed.append("resume_store_directory")
        if fold.resume_admission_sha256 is not None:
            changed.append("resume_admission_sha256")
        if (
            fold.selective_catalog_store_directory is None
            or fold.selective_catalog_receipt_json is None
            or fold.expected_selective_catalog_receipt_sha256 is None
        ):
            changed.append("prepared_selective_catalog")
        if fold.prepare_selective_catalog_only:
            changed.append("prepare_selective_catalog_only")
        if changed:
            raise Lfm2CpcfV5RealAttentionProbeRunnerError(
                "The probe is pinned to the exact H776 4x layer-2 lane; "
                "invalid fields: " + ", ".join(changed) + "."
            )
        for name in (
            "checkpoint_key",
            "checkpoint_manifest_sha256",
            "expected_trace_sha256",
            "expected_runtime_compiler_implementation_sha256",
            "expected_sequence_source_sha256",
        ):
            _digest(getattr(self, name), name)
        if (
            self.expected_runtime_compiler_implementation_sha256
            == H776_ORIGIN_COMPILER_SHA256
        ):
            raise Lfm2CpcfV5RealAttentionProbeRunnerError(
                "The real probe requires an explicitly pinned new compiler "
                "runtime, not ordinary same-runtime resume."
            )
        _validate_diagnostic_destination(
            fold.diagnostic_json,
            protected={
                "parent_model_directory": fold.parent_model_directory,
                "donor_tokenizer_directory": fold.donor_tokenizer_directory,
                "target_tokenizer_directory": fold.target_tokenizer_directory,
                "bridge_directory": fold.bridge_directory,
                "trace_cache_directory": fold.trace_cache_directory,
                "trajectory_corpus": fold.trajectory_corpus,
                "checkpoint_store_directory": fold.resume_store_directory,
                "selective_catalog_store_directory": (
                    fold.selective_catalog_store_directory
                ),
                "selective_catalog_receipt_json": (
                    fold.selective_catalog_receipt_json
                ),
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = _v5_runner().build_parser()
    parser.description = __doc__
    parser.add_argument("--checkpoint-key", required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--expected-trace-sha256", required=True)
    parser.add_argument(
        "--expected-runtime-compiler-implementation-sha256",
        required=True,
    )
    parser.add_argument("--expected-sequence-source-sha256", required=True)
    return parser


def config_from_args(args: argparse.Namespace) -> RealAttentionProbeConfig:
    fold = _v5_runner().config_from_args(args)
    return RealAttentionProbeConfig(
        fold=fold,
        checkpoint_key=args.checkpoint_key,
        checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
        expected_trace_sha256=args.expected_trace_sha256,
        expected_runtime_compiler_implementation_sha256=(
            args.expected_runtime_compiler_implementation_sha256
        ),
        expected_sequence_source_sha256=args.expected_sequence_source_sha256,
    )


def _cuda_peak_sampler(torch: Any) -> Mapping[str, int]:
    return {
        f"cuda:{index}": int(torch.cuda.max_memory_reserved(index))
        for index in (0, 1)
    }


def _runtime_guard_report(
    guard: Any,
    *,
    parent_execution: Mapping[str, int],
    target_execution: Mapping[str, int],
    allocator_policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    body = {
        "schema": (
            "neural-forge-lfm2-cpcf-v5-real-attention-runtime-guard-v1"
        ),
        "cuda_allocator_policy": _plain(allocator_policy),
        "optimizer_constructions": int(guard.optimizer_constructions),
        "optimizer_steps": int(guard.optimizer_steps),
        "backward_calls": int(guard.backward_calls),
        "autograd_grad_calls": int(guard.autograd_grad_calls),
        "pruning_operations": int(guard.pruning_operations),
        "quantization_operations": int(guard.quantization_operations),
        "parent_module_forward_calls": int(
            parent_execution.get("module_forward_calls", 0)
        ),
        "parent_root_forward_calls": int(
            parent_execution.get("root_forward_calls", 0)
        ),
        "parent_inference_entrypoint_calls": int(
            parent_execution.get("inference_entrypoint_calls", 0)
        ),
        "target_module_forward_calls": int(
            target_execution.get("module_forward_calls", 0)
        ),
        "target_root_forward_calls": int(
            target_execution.get("root_forward_calls", 0)
        ),
        "target_inference_entrypoint_calls": int(
            target_execution.get("inference_entrypoint_calls", 0)
        ),
        "training_steps": 0,
        "ordinary_bfloat16_and_float32_casts_blocked": False,
        "scope_started": "immediately_after_parent_model_load",
        "parent_model_load_guarded": False,
        "parent_model_observer_started_immediately_after_load": True,
        "blank_target_synthesis_guarded": True,
        "target_model_observer_started_immediately_after_synthesis": True,
        "selective_provider_open_and_close_guarded": True,
        "source_tied_table_mapping_guarded": True,
        "layer_2_probe_guarded": True,
        "all_model_touching_operations_after_parent_load_guarded": True,
    }
    forbidden = (
        "optimizer_constructions",
        "optimizer_steps",
        "backward_calls",
        "autograd_grad_calls",
        "pruning_operations",
        "quantization_operations",
        "parent_module_forward_calls",
        "parent_root_forward_calls",
        "parent_inference_entrypoint_calls",
        "target_root_forward_calls",
        "target_inference_entrypoint_calls",
    )
    body["clean"] = all(body[name] == 0 for name in forbidden)
    if body["clean"] is not True:
        raise Lfm2CpcfV5RealAttentionProbeRunnerError(
            "The layer probe attempted a forbidden learning, mutation, "
            "parent-execution, or target-root API."
        )
    return {**body, "report_sha256": _sha256(body)}


def run(config: RealAttentionProbeConfig) -> Mapping[str, Any]:
    """Build the exact local runtime, execute layer 2 once, and close it."""

    if not isinstance(config, RealAttentionProbeConfig):
        raise TypeError("config must be RealAttentionProbeConfig.")
    fold = config.fold
    v5_runner = _v5_runner()
    base = _base_runner()

    parent_directory = base._local_directory(
        fold.parent_model_directory,
        "parent_model_directory",
    )
    donor_tokenizer_directory = base._local_directory(
        fold.donor_tokenizer_directory,
        "donor_tokenizer_directory",
    )
    target_tokenizer_directory = base._local_directory(
        fold.target_tokenizer_directory,
        "target_tokenizer_directory",
    )
    bridge_directory = base._local_directory(
        fold.bridge_directory,
        "bridge_directory",
    )
    cache_directory = base._local_directory(
        fold.trace_cache_directory,
        "trace_cache_directory",
    )
    trajectory_path = base._local_file(
        fold.trajectory_corpus,
        "trajectory_corpus",
    )
    checkpoint_store = base._local_directory(
        fold.resume_store_directory,
        "resume_store_directory",
    )
    _validate_diagnostic_destination(
        fold.diagnostic_json,
        protected={
            "parent_model_directory": fold.parent_model_directory,
            "donor_tokenizer_directory": fold.donor_tokenizer_directory,
            "target_tokenizer_directory": fold.target_tokenizer_directory,
            "bridge_directory": fold.bridge_directory,
            "trace_cache_directory": fold.trace_cache_directory,
            "trajectory_corpus": fold.trajectory_corpus,
            "checkpoint_store_directory": fold.resume_store_directory,
            "selective_catalog_store_directory": (
                fold.selective_catalog_store_directory
            ),
            "selective_catalog_receipt_json": (
                fold.selective_catalog_receipt_json
            ),
        },
    )

    catalog_preparation = v5_runner._read_selective_catalog_preparation(fold)
    catalog_authentication = v5_runner._catalog_authentication_summary(
        catalog_preparation
    )
    torch = v5_runner._configure_torch()
    allocator_policy = v5_runner._enforce_cuda_allocator_caps(
        torch,
        maximum_cuda_bytes_per_device=(
            fold.maximum_cuda_bytes_per_device
        ),
    )

    from experiments.run_lfm2_residual_attention_fold import (
        direct_local_io_audit,
        load_local_lfm2_model,
        load_local_tokenizer,
    )
    from neural_forge import dynamic_fold_executor
    from neural_forge import hf_lfm2_cpcf_v5_model_compiler as v5
    from neural_forge.dynamic_behavior_execution import observe_model_execution
    from neural_forge.hf_lfm2_circuit_trace_cache import (
        fingerprint_lfm2_parent,
        open_lfm2_circuit_trace_cache,
    )
    from neural_forge.hf_lfm2_cpcf_blank_target import (
        plan_and_synthesize_lfm2_cpcf_blank,
    )
    from neural_forge.hf_lfm2_cpcf_model_fold import (
        Lfm2CpcfModelFoldPolicy,
        audit_lfm2_cpcf_parameter_compaction,
        independent_parameter_inventory,
    )
    from neural_forge.hf_lfm2_cpcf_selective_evidence import (
        open_lfm2_cpcf_selective_evidence_provider_from_catalog,
    )
    from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
        Lfm2CpcfShortConvV4SolverPolicy,
    )
    from neural_forge.hf_lfm2_cpcf_v5_real_attention_probe import (
        H776_TARGET_UNIQUE_PARAMETERS,
        Lfm2CpcfV5RealAttentionProbePins,
        Lfm2CpcfV5RealAttentionProbeRuntime,
        Lfm2CpcfV5RealAttentionResourceMonitor,
        execute_lfm2_cpcf_v5_real_attention_probe,
    )
    from neural_forge.hf_tokenizer_bridge import (
        fingerprint_hf_tokenizer_semantics,
        load_strict_tokenizer_bridge_artifact,
    )

    dtype = base._torch_dtype(torch, fold.dtype)
    device = torch.device(fold.model_device)
    donor_tokenizer = load_local_tokenizer(donor_tokenizer_directory)
    target_tokenizer = load_local_tokenizer(target_tokenizer_directory)
    bridge = load_strict_tokenizer_bridge_artifact(
        bridge_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    bridge.verify_files()
    donor_fingerprint = fingerprint_hf_tokenizer_semantics(donor_tokenizer)
    target_fingerprint = fingerprint_hf_tokenizer_semantics(target_tokenizer)
    if (
        donor_fingerprint != bridge.donor_tokenizer_semantics_sha256
        or target_fingerprint != bridge.target_tokenizer_semantics_sha256
    ):
        raise Lfm2CpcfV5RealAttentionProbeRunnerError(
            "Live local tokenizers differ from the sealed token bridge."
        )
    rows = base._read_trajectory_rows(trajectory_path)
    cache = open_lfm2_circuit_trace_cache(cache_directory, verify=False)
    trace_adapter = base._trace_adapter(
        rows=rows,
        target_tokenizer=target_tokenizer,
        bridge=bridge,
        cache_manifest=cache.manifest,
    )
    trace_adapter.verify()
    if trace_adapter.evidence_sha256 != config.expected_trace_sha256:
        raise Lfm2CpcfV5RealAttentionProbeRunnerError(
            "Live trace adapter differs from its invocation pin."
        )
    parent = load_local_lfm2_model(
        parent_directory,
        device=device,
        dtype=dtype,
    )
    guard = dynamic_fold_executor._FoldOnlyTorchGuard()
    parent_execution: Mapping[str, int] = {}
    target_execution: Mapping[str, int] = {}
    provider = None
    with torch.no_grad(), guard:
        with ExitStack() as execution_observers:
            parent_execution = execution_observers.enter_context(
                observe_model_execution(parent)
            )
            local_io_audit = direct_local_io_audit(parent_directory)
            parent.requires_grad_(False)
            parent.eval()
            parent_fingerprint = fingerprint_lfm2_parent(parent)
            if parent_fingerprint.identity_sha256 != cache.parent_fingerprint:
                raise Lfm2CpcfV5RealAttentionProbeRunnerError(
                    "Explicit local parent differs from the sealed trace cache."
                )
            parent_inventory = independent_parameter_inventory(parent)
            blank = plan_and_synthesize_lfm2_cpcf_blank(
                parent,
                parent_name=fold.parent_name,
                parent_tokenizer_fingerprint=donor_fingerprint,
                evidence_fingerprint=cache.cache_key,
                child_name=fold.child_name,
                target_vocabulary_size=(
                    bridge.transport_profile.target_row_capacity
                ),
                child_tokenizer_fingerprint=target_fingerprint,
                tokenizer_bridge_artifact=bridge,
                minimum_hidden_size=fold.minimum_hidden_size,
                maximum_candidates=fold.maximum_candidates,
                selected_candidate_rank=fold.selected_candidate_rank,
                blank_seed=fold.blank_seed,
                device=device,
                dtype=dtype,
                **fold.compression.as_kwargs(),
            )
            target = blank.model
            target_execution = execution_observers.enter_context(
                observe_model_execution(target)
            )
            target.requires_grad_(False)
            target.eval()
            target_inventory = independent_parameter_inventory(target)
            if (
                blank.plan.fingerprint
                != fold.expected_target_plan_fingerprint
                or blank.unique_parameters != H776_TARGET_UNIQUE_PARAMETERS
                or int(target_inventory["unique_parameters"])
                != H776_TARGET_UNIQUE_PARAMETERS
            ):
                raise Lfm2CpcfV5RealAttentionProbeRunnerError(
                    "Synthesized target differs from the exact H776 plan."
                )
            structural_compression_floor = (
                v5_runner._request_derived_structural_compression_floor(
                    fold.compression
                )
            )
            structural = audit_lfm2_cpcf_parameter_compaction(
                parent,
                target,
                planned_target_unique_parameters=(
                    H776_TARGET_UNIQUE_PARAMETERS
                ),
                promotion_compression_ratio=structural_compression_floor,
            )
            try:
                provider = (
                    open_lfm2_cpcf_selective_evidence_provider_from_catalog(
                        cache,
                        artifact_store=fold.selective_catalog_store_directory,
                        expected_binding_sha256=(
                            catalog_authentication["binding_sha256"]
                        ),
                        expected_manifest_sha256=(
                            catalog_authentication[
                                "artifact_manifest_sha256"
                            ]
                        ),
                        expected_catalog_sha256=(
                            catalog_authentication[
                                "artifact_catalog_sha256"
                            ]
                        ),
                    )
                )
                catalog = provider.catalog
                logical_budget, _required_budget = (
                    v5_runner._resolved_logical_resident_bytes(
                        references=catalog.dynamic_catalog.references,
                        final_norm_logical_bytes=(
                            catalog.final_norm_logical_bytes
                        ),
                        requested_gib=fold.maximum_logical_resident_gib,
                    )
                )
                full_phase_tree_bytes = int(
                    catalog.final_norm_logical_bytes
                ) + sum(
                    int(reference.logical_bytes)
                    for reference in catalog.dynamic_catalog.references
                )
                effective_reduction, _maximum_increase = (
                    v5_runner._derive_effective_rss_reduction_fraction(
                        full_phase_tree_logical_bytes=(
                            full_phase_tree_bytes
                        ),
                        compiler_baseline_rss_bytes=int(
                            v5._process_rss_bytes()
                        ),
                        maximum_process_rss_bytes=(
                            fold.maximum_process_rss_bytes
                        ),
                        requested_minimum_reduction_fraction=(
                            fold.minimum_physical_rss_reduction_fraction
                        ),
                    )
                )
                short_conv_policy = Lfm2CpcfShortConvV4SolverPolicy(
                    ridge_alphas=fold.short_conv_v4_ridge_alphas,
                    variable_projection_passes=(
                        fold.short_conv_v4_variable_projection_passes
                    ),
                    writer_trust_radius=(
                        fold.short_conv_v4_writer_trust_radius
                    ),
                    fir_trust_radius=fold.short_conv_v4_fir_trust_radius,
                    bcx_trust_radius=fold.short_conv_v4_bcx_trust_radius,
                    correction_rms_floor=(
                        fold.short_conv_v4_correction_rms_floor
                    ),
                    context_objective_weight=(
                        fold.short_conv_v4_context_objective_weight
                    ),
                    target_dtype=fold.dtype,
                )
                model_policy = Lfm2CpcfModelFoldPolicy(
                    ridge=fold.ridge,
                    state_reconstruction_weight=(
                        fold.boundary_state_reconstruction_weight
                    ),
                    minimum_predictive_energy_fraction=(
                        fold.boundary_predictive_energy_floor
                    ),
                    minimum_validation_block_retention=(
                        fold.boundary_validation_retention_floor
                    ),
                    enforce_validation_block_floor=True,
                    short_conv_v4_solver_policy=short_conv_policy,
                    promotion_compression_ratio=(
                        structural_compression_floor
                    ),
                    require_promotion_ratio=True,
                )
                runtime_policy = v5.Lfm2CpcfV5RuntimePolicy(
                    model_fold_policy=model_policy,
                    maximum_resident_evidence_bytes=logical_budget,
                    full_phase_tree_logical_bytes=full_phase_tree_bytes,
                    maximum_process_rss_bytes=(
                        fold.maximum_process_rss_bytes
                    ),
                    minimum_physical_rss_reduction_fraction=(
                        effective_reduction
                    ),
                    rss_sample_interval_seconds=0.05,
                    require_short_conv_v5_endpoint_first_selector=True,
                    require_short_conv_v5_authenticated_whole_macrocell_selector=(
                        True
                    ),
                    require_short_conv_v5_real_numerical_frontier_adapter=(
                        True
                    ),
                )
                source_tied_table = base._mapped_source_tied_table(
                    parent,
                    bridge,
                )
                pins = Lfm2CpcfV5RealAttentionProbePins(
                    checkpoint_store_directory=checkpoint_store,
                    checkpoint_key=config.checkpoint_key,
                    checkpoint_manifest_sha256=(
                        config.checkpoint_manifest_sha256
                    ),
                    expected_trace_sha256=config.expected_trace_sha256,
                    expected_runtime_compiler_implementation_sha256=(
                        config.expected_runtime_compiler_implementation_sha256
                    ),
                    expected_sequence_source_sha256=(
                        config.expected_sequence_source_sha256
                    ),
                )
                runtime = Lfm2CpcfV5RealAttentionProbeRuntime(
                    source_model=parent,
                    target_model=target,
                    trace_adapter=trace_adapter,
                    evidence_provider=provider,
                    source_tied_table=source_tied_table,
                    blank_target_report=blank.as_dict(),
                    target_plan_fingerprint=blank.plan.fingerprint,
                    runtime_policy=runtime_policy,
                )
                for index in (0, 1):
                    torch.cuda.reset_peak_memory_stats(index)
                monitor = Lfm2CpcfV5RealAttentionResourceMonitor(
                    cuda_sampler=lambda: _cuda_peak_sampler(torch),
                    sample_interval_seconds=0.05,
                )
                probe = execute_lfm2_cpcf_v5_real_attention_probe(
                    runtime,
                    pins,
                    resource_monitor=monitor,
                )
            finally:
                if provider is not None:
                    provider.close()
    guard_report = _runtime_guard_report(
        guard,
        parent_execution=parent_execution,
        target_execution=target_execution,
        allocator_policy=allocator_policy,
    )

    post_catalog = v5_runner._read_selective_catalog_preparation(fold)
    if dict(post_catalog) != dict(catalog_preparation):
        raise Lfm2CpcfV5RealAttentionProbeRunnerError(
            "Prepared selective catalog changed during the probe."
        )
    body = {
        "schema": PROBE_RUN_SCHEMA,
        "status": probe["status"],
        "probe": _plain(probe),
        "runtime_guard": _plain(guard_report),
        "structural_parameter_compaction": _plain(structural),
        "parent_parameter_inventory": _plain(parent_inventory),
        "target_parameter_inventory": _plain(target_inventory),
        "local_parent_io_audit": _plain(local_io_audit),
        "catalog_preparation_receipt_sha256": (
            catalog_authentication["preparation_receipt_sha256"]
        ),
        "catalog_reauthenticated_after_provider_close": True,
        "diagnostic_only": True,
        "child_checkpoint_saved": False,
        "publication_attempted": False,
        "promotion_authorized": False,
    }
    return {**body, "receipt_sha256": _sha256(body)}


def _failure(
    error: BaseException,
    *,
    config: RealAttentionProbeConfig | None,
) -> Mapping[str, Any]:
    chain = []
    seen = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(
            {
                "type": f"{type(current).__module__}.{type(current).__qualname__}",
                "message": str(current),
            }
        )
        current = current.__cause__ or current.__context__
    body = {
        "schema": PROBE_FAILURE_SCHEMA,
        "status": "failed_closed",
        "error_chain": chain,
        "checkpoint_key": (
            None if config is None else config.checkpoint_key
        ),
        "layer_index": 2,
        "non_promotable_probe_only": True,
        "child_checkpoint_saved": False,
        "publication_attempted": False,
        "promotion_authorized": False,
    }
    return {**body, "failure_sha256": _sha256(body)}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        safe_diagnostic = validate_diagnostic_destination_args(args)
    except Exception as exc:
        print(
            "error: unsafe diagnostic destination; no diagnostic was written: "
            f"{exc}",
            file=sys.stderr,
        )
        return 2
    config: RealAttentionProbeConfig | None = None
    try:
        config = config_from_args(args)
        receipt = run(config)
    except Exception as exc:
        failure = _failure(exc, config=config)
        try:
            _base_runner()._atomic_json(safe_diagnostic, failure)
        except Exception as write_error:
            print(
                "error: probe failed and diagnostic persistence also failed: "
                f"{write_error}",
                file=sys.stderr,
            )
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        _base_runner()._atomic_json(safe_diagnostic, receipt)
    except Exception as exc:
        print(
            "error: in-memory probe completed but diagnostic persistence "
            f"failed: {exc}",
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_sha256": receipt["receipt_sha256"],
                "diagnostic_json": str(safe_diagnostic),
                "child_checkpoint_saved": False,
                "publication_attempted": False,
                "promotion_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
