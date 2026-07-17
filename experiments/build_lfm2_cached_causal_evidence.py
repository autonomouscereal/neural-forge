#!/usr/bin/env python3
"""Build the real cached causal frontier for a 35% LFM2 structural fold.

The parent is loaded only to authenticate weights and build physical medoid
payloads.  Its already completed sixteen-recovery-case circuit cache is opened
directly; no parent trace, response, generation, or forward is permitted.
Every isolated child proposal is traced and rolled out with uncached
full-prefix forwards, scored on row-disjoint validation evidence, and rolled
back byte-exactly.  Resumable candidate records avoid repeating completed GPU
work while temporary child caches are pruned by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import secrets
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

from experiments import run_lfm2_long_horizon_repair as horizon
from experiments import run_lfm2_residual_attention_fold as residual
from experiments.run_lfm2_dynamic_catalog import fingerprint_local_tokenizer
from neural_forge import (
    hf_lfm2_cached_causal_evidence as cached_causal_evidence_module,
)
from neural_forge import (
    hf_lfm2_cached_causal_sharding as cached_causal_sharding_module,
)
from neural_forge import (
    hf_lfm2_canonical_proposal_pack as canonical_proposal_pack_module,
)
from neural_forge import (
    hf_lfm2_global_causal_allocator as global_causal_allocator_module,
)
from neural_forge import hf_lfm2_variable_width_fold as variable_width_fold_module
from neural_forge.dynamic_fold_executor import (
    fingerprint_model_runtime_state,
    fingerprint_model_state,
    run_fold_only_callback,
)
from neural_forge.dynamic_lfm2_behavior import (
    load_lfm2_cached_whole_output_baseline,
)
from neural_forge.hf_lfm2_cached_causal_evidence import (
    build_and_seal_lfm2_cached_causal_frontier,
    build_lfm2_cached_causal_split,
    build_lfm2_causal_physical_source_evidence,
    candidate_from_cached_measurement,
    generate_lfm2_full_prefix_rollouts,
    lfm2_large_target_width_frontier,
    load_lfm2_cached_causal_candidate_record,
    measure_lfm2_cached_causal_damage,
    seal_lfm2_cached_causal_candidate_record,
    verify_lfm2_cached_causal_frontier_document,
    verify_lfm2_full_prefix_child_trace_cache,
)
from neural_forge.hf_lfm2_cached_causal_sharding import (
    seal_lfm2_cached_causal_run_contract,
    seal_lfm2_cached_causal_shard_manifest,
    seal_lfm2_cached_causal_shard_spec,
    validate_lfm2_cached_causal_merge_inventory,
    verify_lfm2_cached_causal_run_contract,
)
from neural_forge.hf_lfm2_canonical_proposal_pack import (
    load_lfm2_canonical_proposal_pack,
    write_lfm2_canonical_proposal_pack,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_global_causal_allocator import (
    Lfm2CausalFfnProposal,
    Lfm2CausalDamageWeights,
    audit_lfm2_parent_forward_calls,
    prepare_lfm2_causal_ffn_proposal_bundle,
)
from neural_forge.hf_lfm2_long_horizon_anchors import (
    tag_lfm2_long_horizon_rows,
)
from neural_forge.hf_lfm2_variable_width_fold import (
    install_lfm2_variable_width_fold_plan,
)


RUN_SCHEMA = "neural-forge-lfm2-cached-causal-evidence-producer-v1"
TASK_ANCHOR_MANIFEST_SCHEMA = "neural-forge-lfm2-cached-causal-task-anchor-manifest-v1"
EXPECTED_PARENT_PARAMETERS = 229_693_184
PRODUCER_MODES = ("full", "contract", "probe", "shard", "merge")


class Lfm2CachedCausalProducerError(RuntimeError):
    """The cached causal producer failed a local binding or invariant."""


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
        raise Lfm2CachedCausalProducerError(
            "Producer provenance must be finite canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _implementation_binding_sha256() -> str:
    files = {
        "producer": Path(__file__),
        "cached_causal_evidence": Path(cached_causal_evidence_module.__file__),
        "cached_causal_sharding": Path(cached_causal_sharding_module.__file__),
        "canonical_proposal_pack": Path(canonical_proposal_pack_module.__file__),
        "global_causal_allocator": Path(global_causal_allocator_module.__file__),
        "variable_width_fold": Path(variable_width_fold_module.__file__),
    }
    return _json_sha256(
        {
            name: _file_sha256(path.resolve(strict=True))
            for name, path in sorted(files.items())
        }
    )


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CachedCausalProducerError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _safe_file(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2CachedCausalProducerError(f"{name} must be a local file.")
    path = Path(raw).expanduser()
    if not path.is_file() or path.is_symlink():
        raise Lfm2CachedCausalProducerError(f"{name} is missing or unsafe: {path}.")
    return path.resolve(strict=True)


def _safe_directory(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2CachedCausalProducerError(f"{name} must be a local directory.")
    path = Path(raw).expanduser()
    if not path.is_dir() or path.is_symlink():
        raise Lfm2CachedCausalProducerError(f"{name} is missing or unsafe: {path}.")
    return path.resolve(strict=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(dict(value)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CachedCausalProducerError(
            f"Could not read cached candidate record: {path}."
        ) from exc
    if not isinstance(value, Mapping):
        raise Lfm2CachedCausalProducerError(
            f"Cached candidate record is not an object: {path}."
        )
    return value


def _progress(event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {"event": event, **fields},
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _parse_widths(value: str) -> tuple[int, ...]:
    try:
        widths = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "coarse widths must be comma-separated integers"
        ) from exc
    if (
        not widths
        or len(set(widths)) != len(widths)
        or any(width <= 0 for width in widths)
    ):
        raise argparse.ArgumentTypeError(
            "coarse widths must be unique positive integers"
        )
    return widths


def _parse_positive_integers(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "values must be comma-separated integers"
        ) from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(
            "values must be comma-separated positive integers"
        )
    return result


@contextmanager
def _exclusive_writer_lock(root: Path, *, role: str):
    """Prevent two producers from mutating one work root concurrently."""

    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".cached-causal-writer.lock"
    token = secrets.token_hex(16)
    document = {
        "schema": "neural-forge-cached-causal-writer-lock-v1",
        "role": role,
        "pid": os.getpid(),
        "host": platform.node(),
        "token": token,
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise Lfm2CachedCausalProducerError(
            f"Another cached-causal writer owns {root}."
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(document) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            current = _read_json(lock_path)
        except Lfm2CachedCausalProducerError:
            current = {}
        if current.get("token") == token:
            lock_path.unlink(missing_ok=True)


def _proposal_from_candidate_record(
    document: Mapping[str, Any],
    expected_inventory: Mapping[str, Any],
) -> Lfm2CausalFfnProposal:
    """Restore the proposal sealed by the contract builder and shard worker."""

    raw = document.get("proposal")
    if not isinstance(raw, Mapping):
        raise Lfm2CachedCausalProducerError(
            "Merged candidate record has no physical proposal."
        )
    try:
        proposal = Lfm2CausalFfnProposal(
            schema=raw["schema"],
            option_id=raw["option_id"],
            source_evidence_sha256=raw["source_evidence_sha256"],
            layer_index=raw["layer_index"],
            source_width=raw["source_width"],
            target_width=raw["target_width"],
            parameters_saved=raw["parameters_saved"],
            medoid_source_neurons=tuple(raw["medoid_source_neurons"]),
            layer_binding_sha256=raw["layer_binding_sha256"],
            isolated_plan_sha256=raw["isolated_plan_sha256"],
            proposal_sha256=raw["proposal_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Lfm2CachedCausalProducerError(
            "Merged candidate proposal is malformed."
        ) from exc
    if (
        proposal.as_dict() != dict(raw)
        or proposal.option_id != expected_inventory.get("option_id")
        or proposal.layer_index != expected_inventory.get("layer_index")
        or proposal.target_width != expected_inventory.get("target_width")
        or proposal.proposal_sha256 != expected_inventory.get("proposal_sha256")
    ):
        raise Lfm2CachedCausalProducerError(
            "Merged candidate proposal differs from the common run contract."
        )
    return proposal


def _tensor_batches_from_parent_cache(parent_cache: Any) -> tuple[dict[str, Any], ...]:
    """Clone exact cached inputs without executing the parent."""

    batches = []
    for source in parent_cache.iter_batches():
        value = {
            "input_ids": source.input_ids.detach().to(device="cpu").clone(),
            "attention_mask": (source.attention_mask.detach().to(device="cpu").clone()),
            "forced_decode_ids": (
                source.forced_decode_ids.detach().to(device="cpu").clone()
            ),
        }
        position_ids = source.tensors.get("input.position_ids")
        if position_ids is not None:
            value["position_ids"] = position_ids.detach().to(device="cpu").clone()
        batches.append(value)
    return tuple(batches)


def _task_anchor_manifest(
    *,
    envelope: Any,
    anchor_envelope: Mapping[str, Any],
    tagging: Any,
    anchor_plan: Any,
    profile: Any,
    split: Any,
) -> dict[str, Any]:
    return {
        "schema": TASK_ANCHOR_MANIFEST_SCHEMA,
        "parent_output_envelope_sha256": envelope.envelope_sha256,
        "parent_output_file_sha256": envelope.source_file_sha256,
        "envelope_split_identity_sha256": envelope.split_identity_sha256,
        "recovery_case_ids": list(envelope.recovery_case_ids),
        "manual_case_ids_excluded": list(envelope.manual_case_ids),
        "anchor_case_envelope": dict(anchor_envelope),
        "tagging": tagging.as_dict(),
        "anchor_plan": anchor_plan.as_dict(),
        "profile": profile.as_dict(),
        "causal_split": split.as_dict(),
        "selection_fit_validation_rows_disjoint": True,
        "manual_cases_used_for_candidate_evidence": False,
        "independent_64_case_holdout_used_for_candidate_evidence": False,
    }


def _safe_prune_candidate_cache(candidate_root: Path, cache_root: Path) -> None:
    try:
        candidate = candidate_root.resolve(strict=True)
        root = cache_root.resolve(strict=True)
    except OSError:
        return
    if (
        candidate.parent != root
        or not candidate.name.startswith("ffn-l")
        or not candidate.is_dir()
        or candidate.is_symlink()
    ):
        raise Lfm2CachedCausalProducerError(
            "Refusing to prune an unrecognized candidate cache path."
        )
    shutil.rmtree(candidate)


def execute_cached_causal_evidence_producer(args: argparse.Namespace) -> dict[str, Any]:
    mode = args.mode
    if mode not in PRODUCER_MODES:
        raise Lfm2CachedCausalProducerError(f"Unknown producer mode {mode!r}.")
    if mode in ("full", "merge") and not args.frontier_document:
        raise Lfm2CachedCausalProducerError(
            f"{mode} mode requires --frontier-document."
        )
    if mode in ("contract", "probe", "shard", "merge") and not (
        args.run_contract_document
    ):
        raise Lfm2CachedCausalProducerError(
            f"{mode} mode requires --run-contract-document."
        )
    if mode in ("contract", "probe", "shard") and not (args.canonical_proposal_root):
        raise Lfm2CachedCausalProducerError(
            f"{mode} mode requires --canonical-proposal-root."
        )
    if mode == "probe" and (
        args.probe_layer_index is None
        or args.probe_target_width is None
        or not args.probe_report
    ):
        raise Lfm2CachedCausalProducerError(
            "probe mode requires layer, target width, and report output."
        )
    if mode == "shard" and args.shard_index is None:
        raise Lfm2CachedCausalProducerError("shard mode requires --shard-index.")
    if (
        mode == "shard"
        and args.shard_weights is not None
        and len(args.shard_weights) != args.shard_count
    ):
        raise Lfm2CachedCausalProducerError(
            "--shard-weights needs one positive integer per shard."
        )
    if mode == "merge" and not args.merge_shard_root:
        raise Lfm2CachedCausalProducerError(
            "merge mode requires every --merge-shard-root."
        )
    parent_directory = _safe_directory(args.parent_model, "parent_model")
    parent_cache_directory = _safe_directory(
        args.parent_cache,
        "parent_cache",
    )
    envelope_path = _safe_file(
        args.parent_output_envelope,
        "parent_output_envelope",
    )
    output_path = (
        None
        if not args.frontier_document
        else Path(args.frontier_document).expanduser().absolute()
    )
    contract_path = (
        None
        if not args.run_contract_document
        else Path(args.run_contract_document).expanduser().absolute()
    )
    canonical_proposal_root = (
        None
        if not args.canonical_proposal_root
        else Path(args.canonical_proposal_root).expanduser().absolute()
    )
    work_root = Path(args.work_root).expanduser().absolute()
    if mode in ("full", "merge") and output_path is not None and output_path.exists():
        existing = _read_json(output_path)
        verify_lfm2_cached_causal_frontier_document(existing)
        _progress(
            "frontier_already_complete",
            frontier_sha256=existing["frontier_sha256"],
            path=str(output_path),
        )
        return dict(existing)
    if work_root.exists() and (not work_root.is_dir() or work_root.is_symlink()):
        raise Lfm2CachedCausalProducerError(
            "work_root must be a regular local directory."
        )
    work_root.mkdir(parents=True, exist_ok=True)
    candidate_record_root = work_root / "candidate-records"
    candidate_cache_root = work_root / "candidate-caches"
    candidate_record_root.mkdir(exist_ok=True)
    candidate_cache_root.mkdir(exist_ok=True)

    envelope_file_sha = _file_sha256(envelope_path)
    if envelope_file_sha != _digest(
        args.expected_parent_output_file_sha256,
        "expected_parent_output_file_sha256",
    ):
        raise Lfm2CachedCausalProducerError("Parent output envelope file changed.")
    tokenizer = residual.load_local_tokenizer(parent_directory)
    tokenizer_fingerprint = fingerprint_local_tokenizer(parent_directory)
    if tokenizer_fingerprint != _digest(
        args.expected_tokenizer_fingerprint,
        "expected_tokenizer_fingerprint",
    ):
        raise Lfm2CachedCausalProducerError(
            "Local tokenizer differs from the launch binding."
        )
    envelope = horizon.audit_sealed_horizon_envelope(
        envelope_path,
        expected_sha256=envelope_file_sha,
        token_text_decoder=lambda token_ids: horizon._decode_token_pieces(
            tokenizer,
            token_ids,
        ),
        tokenizer_eos_token_ids=horizon._tokenizer_eos_ids(tokenizer),
    )
    if not envelope.ready:
        raise Lfm2CachedCausalProducerError(
            "The sealed sixteen-recovery-case envelope is not evidence-ready."
        )
    if envelope.tokenizer_fingerprint != tokenizer_fingerprint:
        raise Lfm2CachedCausalProducerError(
            "Envelope tokenizer binding differs from the local tokenizer."
        )

    parent_cache = open_lfm2_circuit_trace_cache(
        parent_cache_directory,
        verify=True,
    )
    expected_parent_cache_key = _digest(
        args.expected_parent_cache_key,
        "expected_parent_cache_key",
    )
    expected_parent_identity = _digest(
        args.expected_parent_identity,
        "expected_parent_identity",
    )
    if (
        parent_cache.cache_key != expected_parent_cache_key
        or parent_cache.parent_fingerprint != expected_parent_identity
        or parent_cache.parent_fingerprint != envelope.parent_state_fingerprint
        or len(parent_cache) != 16
    ):
        raise Lfm2CachedCausalProducerError(
            "Opened circuit cache differs from the sealed 16-case parent."
        )
    parent_rows = residual.extract_trace_rows(parent_cache)
    descriptors = horizon.trace_path_descriptors_from_paths(
        parent_rows.paths,
        envelope,
    )
    anchor_envelope = horizon.build_anchor_case_envelope(envelope)
    tagging = tag_lfm2_long_horizon_rows(anchor_envelope, descriptors)
    anchor_plan = horizon.build_repair_anchor_plan(
        anchor_envelope,
        descriptors,
        tagging=tagging,
    )
    profiles = horizon.build_evidence_profiles(anchor_plan, tagging)
    profile = next(
        (value for value in profiles if value.name == args.profile),
        None,
    )
    if profile is None:
        raise Lfm2CachedCausalProducerError(
            f"Unknown evidence profile {args.profile!r}."
        )
    split = build_lfm2_cached_causal_split(
        profile,
        descriptors,
        parent_trace_rows_sha256=parent_rows.trace_rows_sha256,
    )
    task_anchor_manifest = _task_anchor_manifest(
        envelope=envelope,
        anchor_envelope=anchor_envelope,
        tagging=tagging,
        anchor_plan=anchor_plan,
        profile=profile,
        split=split,
    )
    task_anchor_manifest_sha256 = _json_sha256(task_anchor_manifest)
    baseline = load_lfm2_cached_whole_output_baseline(
        envelope_path,
        expected_sha256=envelope_file_sha,
        expected_parent_cache_key=envelope.parent_cache_key,
        expected_parent_state_fingerprint=(envelope.parent_state_fingerprint),
        expected_tokenizer_fingerprint=tokenizer_fingerprint,
        expected_generation_limit=envelope.generation_limit,
    )
    _progress(
        "sealed_parent_evidence_ready",
        parent_cache_key=parent_cache.cache_key,
        recovery_case_count=len(envelope.recovery_case_ids),
        validation_case_count=len(split.validation_case_ids),
        split_sha256=split.split_sha256,
    )

    device = torch.device(args.device)
    dtype = residual._torch_dtype(args.dtype)
    parent = residual.load_local_lfm2_model(
        parent_directory,
        device=device,
        dtype=dtype,
    )
    parent_identity = fingerprint_lfm2_parent(parent).identity_sha256
    parent_parameters = residual.unique_parameter_count(parent)
    if (
        parent_identity != expected_parent_identity
        or parent_parameters != args.expected_parent_unique_parameters
    ):
        raise Lfm2CachedCausalProducerError(
            "Loaded local parent differs from the launch binding."
        )
    source_width = int(parent.model.layers[0].feed_forward.w1.out_features)
    hidden_size = int(parent.model.layers[0].feed_forward.w1.in_features)
    widths_by_layer = lfm2_large_target_width_frontier(
        parent_unique_parameters=parent_parameters,
        hidden_size=hidden_size,
        source_width=source_width,
        target_fraction=0.35,
        coarse_widths=args.coarse_widths,
    )
    variable_evidence = build_lfm2_causal_physical_source_evidence(
        parent,
        parent_rows,
        split,
        widths_by_layer,
        cache_key=parent_cache.cache_key,
        heldout_corpus_sha256=parent_cache.corpus_fingerprint,
    )
    parent_capture = parent_cache.manifest["capture"]
    child_capture = Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=int(parent_capture["sample_last_tokens"]),
        max_decode_steps=int(parent_capture["max_decode_steps"]),
        max_capture_bytes=int(args.max_capture_bytes),
        logit_rtol=float(parent_capture["logit_rtol"]),
        logit_atol=float(parent_capture["logit_atol"]),
        require_top1_agreement=False,
        decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
    )
    runtime_compatibility = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    contract_arguments = {
        "parent_binding": {
            "parent_identity_sha256": parent_identity,
            "parent_unique_parameters": parent_parameters,
            "parent_trace_cache_key": parent_cache.cache_key,
            "parent_corpus_sha256": parent_cache.corpus_fingerprint,
            "parent_trace_rows_sha256": parent_rows.trace_rows_sha256,
            "parent_output_envelope_sha256": envelope.envelope_sha256,
            "parent_output_file_sha256": envelope_file_sha,
            "tokenizer_fingerprint": tokenizer_fingerprint,
        },
        "evidence_binding": {
            "profile": profile.name,
            "split_sha256": split.split_sha256,
            "task_anchor_manifest_sha256": task_anchor_manifest_sha256,
            "variable_width_evidence_sha256": variable_evidence.evidence_sha256,
        },
        "target_fraction": 0.35,
        "widths_by_layer": widths_by_layer,
        "measurement_contract": {
            "dtype": args.dtype,
            "decode_trace_mode": FULL_PREFIX_CANONICAL_TRACE_MODE,
            "forced_decode_use_cache": False,
            "rollout_batch_size": args.rollout_batch_size,
            "cluster_ridge": args.cluster_ridge,
            "cluster_max_iterations": args.cluster_max_iterations,
            "cluster_distance_chunk_size": args.cluster_distance_chunk_size,
            "max_capture_bytes": args.max_capture_bytes,
            "maximum_stage_fraction": args.maximum_stage_fraction,
            "max_states": args.max_states,
            "child_capture_selection_sha256": (
                child_capture.as_dict()["selection_sha256"]
            ),
            "runtime_compatibility": runtime_compatibility,
            "runtime_compatibility_sha256": _json_sha256(runtime_compatibility),
            "damage_weights": Lfm2CausalDamageWeights().as_dict(),
        },
        "producer_implementation_sha256": _implementation_binding_sha256(),
    }

    def prepare_bundle(layer_index: int, target_width: int) -> tuple[Any, Any, Any]:
        prepare_phase = f"lfm2-cached-causal-prepare-l{layer_index}-w{target_width}"
        prepare_parent_events: list[Mapping[str, Any]] = []

        def prepare() -> Any:
            return prepare_lfm2_causal_ffn_proposal_bundle(
                parent,
                variable_evidence,
                layer_index=layer_index,
                target_width=target_width,
                cluster_ridge=args.cluster_ridge,
                cluster_max_iterations=args.cluster_max_iterations,
                cluster_distance_chunk_size=args.cluster_distance_chunk_size,
            )

        (prepared, parent_calls), prepare_guard = run_fold_only_callback(
            lambda: audit_lfm2_parent_forward_calls(
                parent,
                prepare,
                phase=prepare_phase,
                event_sink=prepare_parent_events,
            ),
            phase=prepare_phase,
            protected_models=(parent,),
        )
        if parent_calls:
            _progress(
                "live_parent_forward_violation",
                phase=prepare_phase,
                parent_forward_calls=parent_calls,
                audit_events=prepare_parent_events,
            )
            raise Lfm2CachedCausalProducerError(
                "Physical proposal preparation called the live parent; "
                f"phase={prepare_phase!r}, module_calls={parent_calls}, "
                f"first_event={prepare_parent_events[:1]!r}."
            )
        proposal, bundle = prepared
        return proposal, bundle, prepare_guard

    def load_canonical_bundle(
        expected_inventory: Mapping[str, Any],
        *,
        phase: str,
    ) -> tuple[Any, Any]:
        if canonical_proposal_root is None:
            raise Lfm2CachedCausalProducerError(
                "Canonical proposal root is unavailable."
            )
        parent_events: list[Mapping[str, Any]] = []
        clustering_calls = 0
        original_clustering = (
            variable_width_fold_module.build_gated_mlp_activation_cluster_plan
        )

        def forbidden_clustering(*_args: Any, **_kwargs: Any) -> Any:
            nonlocal clustering_calls
            clustering_calls += 1
            raise Lfm2CachedCausalProducerError(
                "Canonical workers are forbidden from reclustering."
            )

        def load() -> Any:
            variable_width_fold_module.build_gated_mlp_activation_cluster_plan = (
                forbidden_clustering
            )
            try:
                return load_lfm2_canonical_proposal_pack(
                    canonical_proposal_root,
                    expected_inventory,
                    parent=parent,
                    evidence=variable_evidence,
                    expected_cluster_ridge=args.cluster_ridge,
                    expected_cluster_max_iterations=(args.cluster_max_iterations),
                    expected_cluster_distance_chunk_size=(
                        args.cluster_distance_chunk_size
                    ),
                )
            finally:
                variable_width_fold_module.build_gated_mlp_activation_cluster_plan = (
                    original_clustering
                )

        (canonical_pack, parent_calls), load_guard = run_fold_only_callback(
            lambda: audit_lfm2_parent_forward_calls(
                parent,
                load,
                phase=phase,
                event_sink=parent_events,
            ),
            phase=phase,
            protected_models=(parent,),
        )
        if parent_calls or clustering_calls:
            raise Lfm2CachedCausalProducerError(
                "Canonical payload loading invoked a forbidden parent forward "
                "or clustering operation."
            )
        if (
            canonical_pack.manifest["manifest_sha256"]
            != expected_inventory["canonical_pack_manifest_sha256"]
            or canonical_pack.manifest["tensor_file_sha256"]
            != expected_inventory["canonical_pack_tensor_file_sha256"]
            or canonical_pack.bundle.bundle_sha256
            != expected_inventory["canonical_bundle_sha256"]
            or canonical_pack.proposal.layer_binding_sha256
            != expected_inventory["canonical_layer_binding_sha256"]
            or expected_inventory.get("canonical_payload_required") is not True
        ):
            raise Lfm2CachedCausalProducerError(
                "Canonical payload differs from the common run contract."
            )
        return canonical_pack, load_guard

    run_contract = None
    if mode in ("probe", "shard", "merge") or (
        mode == "contract" and contract_path is not None and contract_path.exists()
    ):
        if contract_path is None or not contract_path.exists():
            raise Lfm2CachedCausalProducerError(
                "The sealed common run contract is missing."
            )
        run_contract = dict(_read_json(_safe_file(contract_path, "run_contract")))
        verify_lfm2_cached_causal_run_contract(run_contract)
        expected_contract = seal_lfm2_cached_causal_run_contract(
            **contract_arguments,
            candidate_inventory=run_contract["frontier"]["candidate_inventory"],
        )
        if expected_contract != run_contract:
            raise Lfm2CachedCausalProducerError(
                "Common run contract differs from this parent, evidence, "
                "runtime, or producer implementation."
            )

    if mode == "contract":
        if run_contract is not None:
            _progress(
                "run_contract_already_complete",
                path=str(contract_path),
                run_contract_sha256=run_contract["run_contract_sha256"],
                candidate_count=run_contract["frontier"]["candidate_count"],
            )
            return run_contract
        if contract_path is None:
            raise Lfm2CachedCausalProducerError(
                "contract mode requires a run contract output path."
            )
        candidate_inventory = []
        guard_reports = []
        total_options = sum(len(widths) - 1 for widths in widths_by_layer)
        completed = 0
        with _exclusive_writer_lock(work_root, role="contract"):
            for layer_index, layer_widths in enumerate(widths_by_layer):
                for target_width in layer_widths[1:]:
                    proposal, bundle, prepare_guard = prepare_bundle(
                        layer_index,
                        target_width,
                    )
                    if canonical_proposal_root is None:
                        raise Lfm2CachedCausalProducerError(
                            "Contract lost its canonical proposal root."
                        )
                    publish_phase = (
                        "lfm2-cached-causal-canonical-publish-"
                        f"l{layer_index}-w{target_width}"
                    )
                    publish_events: list[Mapping[str, Any]] = []

                    def publish() -> Any:
                        return write_lfm2_canonical_proposal_pack(
                            canonical_proposal_root,
                            proposal,
                            bundle,
                            parent=parent,
                            evidence=variable_evidence,
                        )

                    (
                        (canonical_pack, publish_parent_calls),
                        publish_guard,
                    ) = run_fold_only_callback(
                        lambda: audit_lfm2_parent_forward_calls(
                            parent,
                            publish,
                            phase=publish_phase,
                            event_sink=publish_events,
                        ),
                        phase=publish_phase,
                        protected_models=(parent,),
                    )
                    if publish_parent_calls:
                        raise Lfm2CachedCausalProducerError(
                            "Canonical publication called the live parent."
                        )
                    if (
                        canonical_pack.proposal != proposal
                        or canonical_pack.bundle.bundle_sha256 != bundle.bundle_sha256
                    ):
                        raise Lfm2CachedCausalProducerError(
                            "Published canonical proposal did not round-trip."
                        )
                    candidate_inventory.append(
                        {
                            "option_id": proposal.option_id,
                            "layer_index": proposal.layer_index,
                            "target_width": proposal.target_width,
                            "proposal_sha256": proposal.proposal_sha256,
                            "canonical_pack_manifest_sha256": (
                                canonical_pack.manifest["manifest_sha256"]
                            ),
                            "canonical_pack_tensor_file_sha256": (
                                canonical_pack.manifest["tensor_file_sha256"]
                            ),
                            "canonical_bundle_sha256": (
                                canonical_pack.bundle.bundle_sha256
                            ),
                            "canonical_layer_binding_sha256": (
                                proposal.layer_binding_sha256
                            ),
                            "canonical_payload_required": True,
                        }
                    )
                    guard_reports.append(prepare_guard.as_dict())
                    guard_reports.append(publish_guard.as_dict())
                    completed += 1
                    _progress(
                        "contract_proposal_prepared",
                        completed=completed,
                        total=total_options,
                        option_id=proposal.option_id,
                        canonical_pack_manifest_sha256=(
                            canonical_pack.manifest["manifest_sha256"]
                        ),
                    )
                    del bundle
            run_contract = seal_lfm2_cached_causal_run_contract(
                **contract_arguments,
                candidate_inventory=candidate_inventory,
            )
            contract_run = {
                "schema": RUN_SCHEMA,
                "mode": "contract",
                "run_contract_sha256": run_contract["run_contract_sha256"],
                "candidate_count": len(candidate_inventory),
                "guard_reports_sha256": _json_sha256(guard_reports),
                "parent_forward_calls": 0,
                "parent_cache_build_calls": 0,
                "parent_generation_calls": 0,
                "optimizer_steps": 0,
                "backward_calls": 0,
            }
            _atomic_write_json(contract_path, run_contract)
            _atomic_write_json(work_root / "contract-run.json", contract_run)
        _progress(
            "run_contract_ready",
            path=str(contract_path),
            run_contract_sha256=run_contract["run_contract_sha256"],
            candidate_count=len(candidate_inventory),
        )
        return run_contract

    if mode == "probe":
        if run_contract is None:
            raise Lfm2CachedCausalProducerError(
                "Probe mode lacks its authenticated run contract."
            )
        expected = next(
            (
                value
                for value in run_contract["frontier"]["candidate_inventory"]
                if value["layer_index"] == args.probe_layer_index
                and value["target_width"] == args.probe_target_width
            ),
            None,
        )
        if expected is None:
            raise Lfm2CachedCausalProducerError(
                "Probe coordinate is absent from the common inventory."
            )
        canonical_pack, load_guard = load_canonical_bundle(
            expected,
            phase=(
                "lfm2-cached-causal-canonical-probe-load-"
                f"l{args.probe_layer_index}-w{args.probe_target_width}"
            ),
        )
        child = residual.load_local_lfm2_model(
            parent_directory,
            device=device,
            dtype=dtype,
        )
        blank_state = fingerprint_model_state(child)
        blank_runtime = fingerprint_model_runtime_state(child)
        install_events: list[Mapping[str, Any]] = []
        install_phase = (
            "lfm2-cached-causal-canonical-probe-install-"
            f"l{args.probe_layer_index}-w{args.probe_target_width}"
        )

        def install_probe() -> Mapping[str, Any]:
            transaction = install_lfm2_variable_width_fold_plan(
                parent,
                child,
                canonical_pack.bundle.fixed_evidence,
                canonical_pack.bundle.plan,
            )
            try:
                return {
                    "installed_child_identity": fingerprint_lfm2_parent(
                        child
                    ).identity_sha256,
                    "installed_child_state_sha256": fingerprint_model_state(child),
                    "fold_report": transaction.report.as_dict(),
                }
            finally:
                if transaction.active:
                    transaction.rollback()

        (installed, parent_calls), install_guard = run_fold_only_callback(
            lambda: audit_lfm2_parent_forward_calls(
                parent,
                install_probe,
                phase=install_phase,
                event_sink=install_events,
            ),
            phase=install_phase,
            protected_models=(parent,),
        )
        if (
            parent_calls
            or fingerprint_model_state(child) != blank_state
            or fingerprint_model_runtime_state(child) != blank_runtime
        ):
            raise Lfm2CachedCausalProducerError(
                "Canonical cross-device probe called the parent or failed "
                "exact child rollback."
            )
        probe_body = {
            "schema": "neural-forge-lfm2-canonical-cross-device-probe-v1",
            "run_contract_sha256": run_contract["run_contract_sha256"],
            "option_id": canonical_pack.proposal.option_id,
            "proposal_sha256": canonical_pack.proposal.proposal_sha256,
            "canonical_pack_manifest_sha256": canonical_pack.manifest[
                "manifest_sha256"
            ],
            "canonical_pack_tensor_file_sha256": canonical_pack.manifest[
                "tensor_file_sha256"
            ],
            "canonical_bundle_sha256": (canonical_pack.bundle.bundle_sha256),
            "canonical_plan_sha256": (canonical_pack.bundle.plan.plan_sha256),
            "canonical_layer_binding_sha256": (
                canonical_pack.proposal.layer_binding_sha256
            ),
            "canonical_medoids_sha256": _json_sha256(
                list(canonical_pack.proposal.medoid_source_neurons)
            ),
            **dict(installed),
            "device_provenance": {
                "requested_device": args.device,
                "device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else str(device)
                ),
                "runtime_compatibility": runtime_compatibility,
            },
            "load_guard": load_guard.as_dict(),
            "install_guard": install_guard.as_dict(),
            "clustering_calls": 0,
            "parent_forward_calls": 0,
            "parent_cache_build_calls": 0,
            "parent_generation_calls": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }
        probe_report = {
            **probe_body,
            "probe_sha256": _json_sha256(probe_body),
        }
        probe_path = Path(args.probe_report).expanduser().absolute()
        _atomic_write_json(probe_path, probe_report)
        _progress(
            "canonical_cross_device_probe_ready",
            path=str(probe_path),
            probe_sha256=probe_report["probe_sha256"],
            option_id=canonical_pack.proposal.option_id,
            installed_child_identity=installed["installed_child_identity"],
        )
        return probe_report

    if mode == "merge":
        if run_contract is None or output_path is None:
            raise Lfm2CachedCausalProducerError(
                "merge mode lacks its run contract or frontier output."
            )
        shard_inputs = []
        shard_roots_by_index: dict[int, Path] = {}
        for raw_root in args.merge_shard_root:
            shard_root = _safe_directory(raw_root, "merge_shard_root")
            manifest_path = _safe_file(
                shard_root / "shard-manifest.json",
                "shard_manifest",
            )
            manifest = dict(_read_json(manifest_path))
            shard_index = int(manifest.get("shard_spec", {}).get("shard_index", -1))
            if shard_index in shard_roots_by_index:
                raise Lfm2CachedCausalProducerError(
                    "Merge received a duplicate shard index."
                )
            shard_roots_by_index[shard_index] = shard_root
            shard_inputs.append((shard_root, manifest))
        merge_inventory = validate_lfm2_cached_causal_merge_inventory(
            run_contract,
            shard_inputs,
        )
        expected_by_option = {
            value["option_id"]: value
            for value in run_contract["frontier"]["candidate_inventory"]
        }
        candidates = []
        reports = []
        for record_entry in merge_inventory["records"]:
            option_id = record_entry["option_id"]
            shard_root = shard_roots_by_index[int(record_entry["source_shard_index"])]
            record_path = _safe_file(
                shard_root / record_entry["relative_path"],
                "merged_candidate_record",
            )
            document = _read_json(record_path)
            proposal = _proposal_from_candidate_record(
                document,
                expected_by_option[option_id],
            )
            candidate, report = load_lfm2_cached_causal_candidate_record(
                document,
                expected_proposal=proposal,
                split=split,
            )
            candidates.append(candidate)
            reports.append(report)
        evidence, plan, document = build_and_seal_lfm2_cached_causal_frontier(
            parent,
            variable_evidence,
            candidates,
            split,
            widths_by_layer,
            reports,
            parent_output_envelope_sha256=envelope.envelope_sha256,
            parent_output_file_sha256=envelope_file_sha,
            task_anchor_manifest_sha256=task_anchor_manifest_sha256,
            task_anchor_manifest=task_anchor_manifest,
            tokenizer_fingerprint=tokenizer_fingerprint,
            damage_weights=Lfm2CausalDamageWeights(),
            cluster_ridge=args.cluster_ridge,
            cluster_max_iterations=args.cluster_max_iterations,
            cluster_distance_chunk_size=args.cluster_distance_chunk_size,
            maximum_stage_fraction=args.maximum_stage_fraction,
            max_states=args.max_states,
        )
        run_manifest = {
            "schema": RUN_SCHEMA,
            "mode": "merge",
            "run_contract_sha256": run_contract["run_contract_sha256"],
            "merge_inventory_sha256": merge_inventory["merge_inventory_sha256"],
            "frontier_sha256": document["frontier_sha256"],
            "evidence_sha256": evidence.evidence_sha256,
            "plan_sha256": plan.plan_sha256,
            "candidate_count": len(candidates),
            "parent_forward_calls": 0,
            "parent_cache_build_calls": 0,
            "parent_generation_calls": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }
        with _exclusive_writer_lock(work_root, role="merge"):
            verify_lfm2_cached_causal_frontier_document(document)
            _atomic_write_json(work_root / "merge-inventory.json", merge_inventory)
            _atomic_write_json(work_root / "producer-run.json", run_manifest)
            _atomic_write_json(output_path, document)
        _progress(
            "frontier_ready",
            path=str(output_path),
            frontier_sha256=document["frontier_sha256"],
            evidence_sha256=evidence.evidence_sha256,
            plan_sha256=plan.plan_sha256,
            target_unique_parameters=plan.budget.target_unique_parameters,
            realized_fraction=plan.budget.realized_fraction,
        )
        return document

    batches = _tensor_batches_from_parent_cache(parent_cache)
    child = residual.load_local_lfm2_model(
        parent_directory,
        device=device,
        dtype=dtype,
    )
    blank_state = fingerprint_model_state(child)
    blank_runtime = fingerprint_model_runtime_state(child)

    shard_spec = None
    owned_layers = set(range(len(widths_by_layer)))
    expected_by_coordinate: dict[tuple[int, int], Mapping[str, Any]] = {}
    if mode == "shard":
        if run_contract is None:
            raise Lfm2CachedCausalProducerError(
                "Shard mode lacks its authenticated run contract."
            )
        shard_spec = seal_lfm2_cached_causal_shard_spec(
            run_contract,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            shard_weights=args.shard_weights,
        )
        owned_layers = set(shard_spec["owned_layer_indices"])
        expected_by_coordinate = {
            (value["layer_index"], value["target_width"]): value
            for value in shard_spec["expected_records"]
        }
        spec_path = work_root / "shard-spec.json"
        if spec_path.exists():
            if _read_json(spec_path) != shard_spec:
                raise Lfm2CachedCausalProducerError(
                    "Existing shard root belongs to another ownership contract."
                )
        else:
            _atomic_write_json(spec_path, shard_spec)

    candidates = []
    reports = []
    guard_reports = []
    total_options = (
        sum(len(widths) - 1 for widths in widths_by_layer)
        if shard_spec is None
        else int(shard_spec["expected_candidate_count"])
    )
    completed = 0
    for layer_index, layer_widths in enumerate(widths_by_layer):
        if layer_index not in owned_layers:
            continue
        for target_width in layer_widths[1:]:
            if mode == "shard":
                expected = expected_by_coordinate[(layer_index, target_width)]
                canonical_pack, load_guard = load_canonical_bundle(
                    expected,
                    phase=(
                        "lfm2-cached-causal-canonical-worker-load-"
                        f"l{layer_index}-w{target_width}"
                    ),
                )
                proposal = canonical_pack.proposal
                bundle = canonical_pack.bundle
                guard_reports.append(load_guard.as_dict())
                if (
                    proposal.option_id != expected["option_id"]
                    or proposal.proposal_sha256 != expected["proposal_sha256"]
                ):
                    raise Lfm2CachedCausalProducerError(
                        "Shard proposal differs from the sealed common inventory."
                    )
            else:
                proposal, bundle, prepare_guard = prepare_bundle(
                    layer_index,
                    target_width,
                )
                guard_reports.append(prepare_guard.as_dict())
            record_path = candidate_record_root / f"{proposal.option_id}.json"
            if not record_path.exists() and mode == "shard":
                for raw_root in args.adopt_candidate_record_root:
                    adoption_root = _safe_directory(
                        raw_root,
                        "adopt_candidate_record_root",
                    )
                    source_path = (
                        adoption_root
                        / "candidate-records"
                        / f"{proposal.option_id}.json"
                    )
                    if not source_path.exists():
                        continue
                    source_document = _read_json(
                        _safe_file(source_path, "adopted_candidate_record")
                    )
                    load_lfm2_cached_causal_candidate_record(
                        source_document,
                        expected_proposal=proposal,
                        split=split,
                    )
                    _atomic_write_json(record_path, source_document)
                    _progress(
                        "candidate_adopted",
                        option_id=proposal.option_id,
                        source_root=str(adoption_root),
                    )
                    break
            if record_path.exists():
                candidate, report = load_lfm2_cached_causal_candidate_record(
                    _read_json(record_path),
                    expected_proposal=proposal,
                    split=split,
                )
                candidates.append(candidate)
                reports.append(report)
                completed += 1
                _progress(
                    "candidate_reused",
                    completed=completed,
                    total=total_options,
                    option_id=proposal.option_id,
                )
                continue

            candidate_cache_directory = candidate_cache_root / proposal.option_id
            if candidate_cache_directory.exists():
                if (
                    not candidate_cache_directory.is_dir()
                    or candidate_cache_directory.is_symlink()
                ):
                    raise Lfm2CachedCausalProducerError(
                        "Candidate cache path is not a safe directory."
                    )
            else:
                candidate_cache_directory.mkdir()
            cache_preexisting = any(candidate_cache_directory.iterdir())

            def measure_candidate() -> Any:
                transaction = install_lfm2_variable_width_fold_plan(
                    parent,
                    child,
                    bundle.fixed_evidence,
                    bundle.plan,
                )
                try:
                    installed_child_identity = fingerprint_lfm2_parent(
                        child
                    ).identity_sha256
                    child_cache, child_rows = residual.build_child_trace_cache(
                        candidate_cache_directory,
                        child,
                        batches,
                        capture_config=child_capture,
                    )
                    verify_lfm2_full_prefix_child_trace_cache(child_cache)
                    if (
                        child_cache.parent_fingerprint != installed_child_identity
                        or child_cache.corpus_fingerprint
                        != parent_cache.corpus_fingerprint
                        or len(child_cache) != len(parent_cache)
                        or child_cache.path.parent.resolve(strict=True)
                        != candidate_cache_directory.resolve(strict=True)
                    ):
                        raise Lfm2CachedCausalProducerError(
                            "Candidate cache does not bind the freshly "
                            "installed physical child and sealed corpus."
                        )
                    if cache_preexisting:
                        _progress(
                            "candidate_cache_authenticated",
                            option_id=proposal.option_id,
                            cache_key=child_cache.cache_key,
                            installed_child_identity=installed_child_identity,
                            batch_count=len(child_cache),
                        )
                    rollouts = generate_lfm2_full_prefix_rollouts(
                        child,
                        envelope.recovery_rows,
                        validation_case_ids=split.validation_case_ids,
                        tokenizer=tokenizer,
                        baseline=baseline,
                        eos_token_ids=envelope.eos_token_ids,
                        generation_limit=envelope.generation_limit,
                        rollout_batch_size=args.rollout_batch_size,
                    )
                    measurement = measure_lfm2_cached_causal_damage(
                        parent_cache,
                        child_cache,
                        parent_rows,
                        child_rows,
                        descriptors,
                        split,
                        rollouts,
                        eos_token_ids=envelope.eos_token_ids,
                        parent_recovery_rows=envelope.recovery_rows,
                        option_id=proposal.option_id,
                        proposal_layer_index=proposal.layer_index,
                        generation_limit=envelope.generation_limit,
                    )
                    return measurement, transaction.report.as_dict()
                finally:
                    if transaction.active:
                        transaction.rollback()

            measure_phase = f"lfm2-cached-causal-measure-l{layer_index}-w{target_width}"
            measure_parent_events: list[Mapping[str, Any]] = []
            try:
                (measured, parent_calls), measure_guard = run_fold_only_callback(
                    lambda: audit_lfm2_parent_forward_calls(
                        parent,
                        measure_candidate,
                        phase=measure_phase,
                        event_sink=measure_parent_events,
                    ),
                    phase=measure_phase,
                    protected_models=(parent,),
                )
            finally:
                # The exact-parent audit uses a temporary global torch hook.
                # Runtime fingerprints intentionally include global hooks, so
                # authenticate the fully restored child only after the audit
                # callback has removed that instrumentation.
                if (
                    fingerprint_model_state(child) != blank_state
                    or fingerprint_model_runtime_state(child) != blank_runtime
                ):
                    raise Lfm2CachedCausalProducerError(
                        "Isolated candidate did not roll back exactly."
                    )
            if parent_calls:
                _progress(
                    "live_parent_forward_violation",
                    phase=measure_phase,
                    parent_forward_calls=parent_calls,
                    audit_events=measure_parent_events,
                )
                raise Lfm2CachedCausalProducerError(
                    "Candidate measurement called the live parent; "
                    f"phase={measure_phase!r}, module_calls={parent_calls}, "
                    f"first_event={measure_parent_events[:1]!r}."
                )
            measurement, _fold_report = measured
            guard_reports.append(measure_guard.as_dict())
            candidate = candidate_from_cached_measurement(
                proposal,
                measurement,
            )
            record = seal_lfm2_cached_causal_candidate_record(
                proposal,
                candidate,
                measurement.report,
                split,
            )
            _atomic_write_json(record_path, record)
            candidates.append(candidate)
            reports.append(measurement.report)
            completed += 1
            _progress(
                "candidate_measured",
                completed=completed,
                total=total_options,
                option_id=proposal.option_id,
                target_width=proposal.target_width,
                damage=candidate.damage.as_dict(),
            )
            if not args.keep_candidate_caches:
                _safe_prune_candidate_cache(
                    candidate_cache_directory,
                    candidate_cache_root,
                )

    if mode == "shard":
        if run_contract is None or shard_spec is None:
            raise Lfm2CachedCausalProducerError(
                "Completed shard lost its run contract or ownership spec."
            )
        device_name = (
            torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        )
        worker_provenance = {
            "host": platform.node(),
            "device": args.device,
            "device_name": device_name,
            "runtime_compatibility": runtime_compatibility,
            "runtime_compatibility_sha256": _json_sha256(runtime_compatibility),
            "producer_implementation_sha256": contract_arguments[
                "producer_implementation_sha256"
            ],
        }
        shard_manifest = seal_lfm2_cached_causal_shard_manifest(
            work_root,
            run_contract,
            shard_spec,
            worker_provenance=worker_provenance,
        )
        run_manifest = {
            "schema": RUN_SCHEMA,
            "mode": "shard",
            "run_contract_sha256": run_contract["run_contract_sha256"],
            "shard_spec_sha256": shard_spec["shard_spec_sha256"],
            "shard_manifest_sha256": shard_manifest["shard_manifest_sha256"],
            "candidate_count": len(candidates),
            "guard_reports_sha256": _json_sha256(guard_reports),
            "parent_forward_calls": 0,
            "parent_cache_build_calls": 0,
            "parent_generation_calls": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }
        with _exclusive_writer_lock(work_root, role="shard-manifest"):
            _atomic_write_json(
                work_root / "shard-manifest.json",
                shard_manifest,
            )
            _atomic_write_json(work_root / "producer-run.json", run_manifest)
        _progress(
            "shard_ready",
            shard_index=shard_spec["shard_index"],
            shard_count=shard_spec["shard_count"],
            candidate_count=len(candidates),
            shard_manifest_sha256=shard_manifest["shard_manifest_sha256"],
        )
        return shard_manifest

    if output_path is None:
        raise Lfm2CachedCausalProducerError(
            "Full producer lost its frontier output path."
        )
    evidence, plan, document = build_and_seal_lfm2_cached_causal_frontier(
        parent,
        variable_evidence,
        candidates,
        split,
        widths_by_layer,
        reports,
        parent_output_envelope_sha256=envelope.envelope_sha256,
        parent_output_file_sha256=envelope_file_sha,
        task_anchor_manifest_sha256=task_anchor_manifest_sha256,
        task_anchor_manifest=task_anchor_manifest,
        tokenizer_fingerprint=tokenizer_fingerprint,
        damage_weights=Lfm2CausalDamageWeights(),
        cluster_ridge=args.cluster_ridge,
        cluster_max_iterations=args.cluster_max_iterations,
        cluster_distance_chunk_size=args.cluster_distance_chunk_size,
        maximum_stage_fraction=args.maximum_stage_fraction,
        max_states=args.max_states,
    )
    run_manifest = {
        "schema": RUN_SCHEMA,
        "frontier_sha256": document["frontier_sha256"],
        "evidence_sha256": evidence.evidence_sha256,
        "plan_sha256": plan.plan_sha256,
        "candidate_count": len(candidates),
        "guard_reports_sha256": _json_sha256(guard_reports),
        "parent_forward_calls": 0,
        "parent_cache_build_calls": 0,
        "parent_generation_calls": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    verify_lfm2_cached_causal_frontier_document(document)
    _atomic_write_json(work_root / "producer-run.json", run_manifest)
    _atomic_write_json(output_path, document)
    _progress(
        "frontier_ready",
        path=str(output_path),
        frontier_sha256=document["frontier_sha256"],
        evidence_sha256=evidence.evidence_sha256,
        plan_sha256=plan.plan_sha256,
        target_unique_parameters=plan.budget.target_unique_parameters,
        realized_fraction=plan.budget.realized_fraction,
    )
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build real validation-only causal evidence from an existing "
            "sealed LFM2 recovery cache; no parent forward or training."
        )
    )
    parser.add_argument("--parent-model", required=True)
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--expected-parent-cache-key", required=True)
    parser.add_argument("--parent-output-envelope", required=True)
    parser.add_argument(
        "--expected-parent-output-file-sha256",
        required=True,
    )
    parser.add_argument("--expected-parent-identity", required=True)
    parser.add_argument("--expected-tokenizer-fingerprint", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument(
        "--mode",
        choices=PRODUCER_MODES,
        default="full",
        help=(
            "full preserves the serial producer; contract seals the common "
            "inventory; shard measures only one owned layer range; merge is "
            "the sole sharded-frontier writer."
        ),
    )
    parser.add_argument("--frontier-document")
    parser.add_argument("--run-contract-document")
    parser.add_argument(
        "--canonical-proposal-root",
        help=(
            "Immutable canonical changed-layer JSON+safetensors packs. "
            "Contract mode publishes them; probe/shard modes only load them."
        ),
    )
    parser.add_argument("--probe-layer-index", type=int)
    parser.add_argument("--probe-target-width", type=int)
    parser.add_argument("--probe-report")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument(
        "--shard-weights",
        type=_parse_positive_integers,
        help=(
            "Optional deterministic whole-layer weights, for example 3,3,2 "
            "for two RTX 3090 workers and one slower RTX 8000 worker."
        ),
    )
    parser.add_argument(
        "--adopt-candidate-record-root",
        action="append",
        default=[],
        help=(
            "Read-only prior work root whose authenticated owned records may "
            "be copied into this shard."
        ),
    )
    parser.add_argument(
        "--merge-shard-root",
        action="append",
        default=[],
        help=(
            "Complete shard root containing shard-manifest.json; repeat once "
            "for every shard."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--expected-parent-unique-parameters",
        type=int,
        default=EXPECTED_PARENT_PARAMETERS,
    )
    parser.add_argument(
        "--coarse-widths",
        type=_parse_widths,
        default=(1792, 1280, 896, 512),
    )
    parser.add_argument(
        "--profile",
        choices=("balanced", "termination_heavy"),
        default="balanced",
    )
    parser.add_argument("--cluster-ridge", type=float, default=1e-8)
    parser.add_argument(
        "--cluster-max-iterations",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--cluster-distance-chunk-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--max-capture-bytes",
        type=int,
        default=1024 * 1024 * 1024,
    )
    parser.add_argument(
        "--maximum-stage-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument("--max-states", type=int, default=100_000)
    parser.add_argument(
        "--keep-candidate-caches",
        action="store_true",
        help=(
            "Retain every very large isolated child trace cache. By default "
            "only sealed candidate records are kept."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    execute_cached_causal_evidence_producer(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
