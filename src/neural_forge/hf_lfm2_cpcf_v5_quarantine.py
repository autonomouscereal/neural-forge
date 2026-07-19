"""Standalone quarantine backend for one compiled LFM2 CPCF-v5 child.

The v5 compiler owns folding.  This module starts only after that compiler has
returned an authenticated, dense in-memory target.  It gives the generic
candidate-quarantine lifecycle three LFM2-specific callables:

* a dense safetensors/tokenizer saver,
* a direct-local JSON+safetensors loader, and
* a cached-parent, whole-output intelligence evaluator.

No function here folds a layer, changes parameter topology, executes a parent,
trains, prunes, quantizes, or authorizes publication.  A successful result is
still an unreviewed quarantine that requires a separately authenticated
reviewed rerun.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dynamic_candidate_capture import (
    QuarantineCaptureRequest,
    QuarantineEvaluation,
    QuarantineZeroOperationAudit,
    QuarantinedDynamicFoldCandidate,
    capture_quarantined_candidate,
    fingerprint_quarantine_callable,
)
from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .dynamic_fold_executor import fingerprint_model_state
from .hf_lfm2_cpcf_model_fold import independent_parameter_inventory
from .hf_lfm2_intelligence_gate import (
    IntelligenceGatePolicy,
    evaluate_cached_intelligence,
)
from .hf_lfm2_intelligence_holdout import (
    Lfm2IntelligenceHoldoutParentEnvelope,
    baseline_from_lfm2_intelligence_holdout_envelope,
    candidate_outputs_from_lfm2_holdout_rows,
)
from .hf_tokenizer_bridge import fingerprint_hf_tokenizer_semantics


LFM2_CPCF_V5_QUARANTINE_BACKEND_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-quarantine-backend-v1"
)
LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-standalone-candidate-v1"
)
LFM2_CPCF_V5_QUARANTINE_OUTPUT_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-holdout-output-v1"
)
LFM2_CPCF_V5_QUARANTINE_EVALUATION_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-holdout-evaluation-v1"
)
LFM2_CPCF_V5_CONTEXT_BINDING_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-consumer-context-binding-v1"
)
LFM2_CPCF_V5_ARTIFACT_METADATA = "neural_forge_lfm2_artifact.json"
LFM2_CPCF_V5_EXPECTED_HOLDOUT_CASES = 64

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACT_BODY_FIELDS = frozenset(
    {
        "schema",
        "backend_schema",
        "context_binding_sha256",
        "model_state_sha256",
        "parameter_inventory",
        "parameter_inventory_sha256",
        "materialized_ffn_width",
        "serialized_ffn_aliases",
        "tokenizer_package_sha256",
        "tokenizer_package_files",
        "tokenizer_semantics_sha256",
        "tokenizer_special_token_ids",
        "dense_safetensor_files",
        "dense_safetensors_only",
        "target_tokenizer_saved_with_child",
        "local_reload_only",
        "network_access",
        "hub_loader_used",
        "training_steps",
        "backward_calls",
        "optimizer_steps",
        "parameters_pruned",
        "parameters_quantized",
    }
)
_COMPILED_AUDIT_ZERO_FIELDS = frozenset(
    {
        "parent_model_forward_calls",
        "target_model_forward_calls",
        "generation_calls",
        "parent_cache_generation_calls",
        "training_steps",
        "backward_calls",
        "optimizer_steps",
        "parameters_pruned",
        "parameters_quantized",
        "fallback_paths",
    }
)
_CONTEXT_FIELDS = (
    "target_model",
    "target_tokenizer",
    "forbidden_parent_model",
    "target_plan_fingerprint",
    "blank_target_report",
    "compiled_model_fold_report",
    "compiler_receipt_sha256",
    "parent_sha256",
    "cache_sha256",
    "bridge_sha256",
    "corpus_sha256",
    "source_sha256",
    "compiler_sha256",
    "runner_source_sha256",
    "source_binding_sha256",
    "parent_tokenizer_semantics_sha256",
    "target_tokenizer_semantics_sha256",
    "target_tokenizer_package_sha256",
    "parent_parameter_inventory",
    "blank_parameter_inventory",
    "compiled_parameter_inventory",
    "zero_operation_audit",
    "target_state_sha256",
    "target_runtime_sha256",
    "parent_state_sha256",
    "parent_runtime_sha256",
    "compile_diagnostic",
)
_CONTEXT_ZERO_COUNTER_FIELDS = (
    "optimizer_constructions",
    "optimizer_steps",
    "backward_calls",
    "autograd_grad_calls",
    "training_steps",
    "pruning_operations",
    "quantization_operations",
    "parent_forward_calls",
    "parent_inference_calls",
    "target_forward_calls",
    "cache_generation_calls",
)


class Lfm2CpcfV5QuarantineError(RuntimeError):
    """The compiled child could not enter strict standalone quarantine."""


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfV5QuarantineError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfV5QuarantineError(
            f"{name} must be a positive integer."
        )
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfV5QuarantineError(f"{name} must be a mapping.")
    try:
        normalized = json.loads(canonical_json_bytes(dict(value)).decode("ascii"))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise Lfm2CpcfV5QuarantineError(
            f"{name} is not canonical JSON data."
        ) from error
    if not isinstance(normalized, dict):
        raise Lfm2CpcfV5QuarantineError(f"{name} must be a JSON object.")
    return normalized


def _sealed_mapping(
    value: Any,
    *,
    name: str,
    seal_field: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    normalized = _mapping(value, name)
    body = dict(normalized)
    supplied = body.pop(seal_field, None)
    supplied = _digest(supplied, f"{name} {seal_field}")
    if supplied != canonical_sha256(body):
        raise Lfm2CpcfV5QuarantineError(
            f"{name} self-hash does not authenticate."
        )
    if expected_sha256 is not None and supplied != _digest(
        expected_sha256,
        f"expected {name} SHA-256",
    ):
        raise Lfm2CpcfV5QuarantineError(
            f"{name} differs from its externally rooted receipt."
        )
    return normalized


def _ordinary_empty_directory(value: Path) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2CpcfV5QuarantineError(
            "Candidate destination must be an explicit local directory."
        )
    try:
        path = Path(raw).resolve(strict=True)
    except OSError as error:
        raise Lfm2CpcfV5QuarantineError(
            "Candidate destination does not exist."
        ) from error
    if path.is_symlink() or not path.is_dir() or any(path.iterdir()):
        raise Lfm2CpcfV5QuarantineError(
            "Candidate destination must be an ordinary empty directory."
        )
    return path


def _ordinary_existing_directory(value: Path) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2CpcfV5QuarantineError(
            "Candidate source must be an explicit local directory."
        )
    try:
        path = Path(raw).resolve(strict=True)
    except OSError as error:
        raise Lfm2CpcfV5QuarantineError(
            "Candidate source does not exist."
        ) from error
    if path.is_symlink() or not path.is_dir():
        raise Lfm2CpcfV5QuarantineError(
            "Candidate source must be an ordinary local directory."
        )
    return path


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(dict(value)) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _read_artifact_metadata(directory: Path) -> dict[str, Any]:
    path = directory / LFM2_CPCF_V5_ARTIFACT_METADATA
    if path.is_symlink() or not path.is_file():
        raise Lfm2CpcfV5QuarantineError(
            "Standalone LFM2 artifact metadata is missing or unsafe."
        )
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Lfm2CpcfV5QuarantineError(
            "Standalone LFM2 artifact metadata is invalid."
        ) from error
    normalized = _sealed_mapping(
        document,
        name="standalone LFM2 artifact metadata",
        seal_field="artifact_sha256",
    )
    body = dict(normalized)
    body.pop("artifact_sha256")
    if set(body) != _ARTIFACT_BODY_FIELDS:
        raise Lfm2CpcfV5QuarantineError(
            "Standalone LFM2 artifact metadata fields are not exact."
        )
    return normalized


def _runtime_modules() -> tuple[Any, Any, Any]:
    """Load only the already-established local LFM2 runtime helpers."""

    try:
        from experiments import (
            capture_lfm2_cpcf_holdout_candidate as capture_runtime,
        )
        from experiments import run_lfm2_cpcf_blank_fold as cpcf_runner
        from experiments import (
            run_lfm2_residual_attention_fold as local_runtime,
        )
    except ImportError as error:  # pragma: no cover - packaging boundary
        raise Lfm2CpcfV5QuarantineError(
            "The canonical local LFM2 runtime helpers are unavailable."
        ) from error
    return capture_runtime, cpcf_runner, local_runtime


def _posthoc_runtime() -> Any:
    try:
        from experiments import evaluate_lfm2_intelligence_retention
    except ImportError as error:  # pragma: no cover - packaging boundary
        raise Lfm2CpcfV5QuarantineError(
            "The canonical tokenizer-package authenticator is unavailable."
        ) from error
    return evaluate_lfm2_intelligence_retention


def _default_model_loader(
    directory: Path,
    *,
    device: str,
    dtype: str,
) -> Any:
    _capture, _runner, local_runtime = _runtime_modules()
    return local_runtime.load_local_lfm2_model(
        directory,
        device=device,
        dtype=local_runtime._torch_dtype(dtype),
    )


def _default_tokenizer_loader(directory: Path) -> Any:
    _capture, _runner, local_runtime = _runtime_modules()
    return local_runtime.load_local_tokenizer(directory)


def _default_candidate_generator(
    model: Any,
    tokenizer: Any,
    manifest: Any,
) -> tuple[dict[str, Any], ...]:
    capture_runtime, _runner, _local = _runtime_modules()
    return capture_runtime.generate_candidate_rows_once(
        model,
        tokenizer,
        manifest,
    )


def _callable_fingerprint(value: Callable[..., Any], name: str) -> str:
    from .dynamic_behavior_gate import identify_behavior_callable

    try:
        return identify_behavior_callable(value).fingerprint
    except BaseException as error:
        raise Lfm2CpcfV5QuarantineError(
            f"Could not authenticate {name} implementation."
        ) from error


def _model_ffn_width(model: Any) -> int:
    config = getattr(model, "config", None)
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if config is None or layers is None:
        raise Lfm2CpcfV5QuarantineError(
            "Candidate is not a concrete materialized LFM2 model."
        )
    layers = tuple(layers)
    if not layers:
        raise Lfm2CpcfV5QuarantineError("Candidate LFM2 stack is empty.")
    widths = set()
    for index, layer in enumerate(layers):
        feed_forward = getattr(layer, "feed_forward", None)
        w1 = getattr(getattr(feed_forward, "w1", None), "weight", None)
        w2 = getattr(getattr(feed_forward, "w2", None), "weight", None)
        w3 = getattr(getattr(feed_forward, "w3", None), "weight", None)
        shapes = (
            getattr(w1, "shape", ()),
            getattr(w2, "shape", ()),
            getattr(w3, "shape", ()),
        )
        if (
            len(shapes[0]) != 2
            or len(shapes[1]) != 2
            or len(shapes[2]) != 2
            or int(shapes[0][0]) != int(shapes[2][0])
            or int(shapes[0][0]) != int(shapes[1][1])
        ):
            raise Lfm2CpcfV5QuarantineError(
                f"Candidate LFM2 layer {index} has inconsistent FFN shapes."
            )
        widths.add(int(shapes[0][0]))
    if len(widths) != 1:
        raise Lfm2CpcfV5QuarantineError(
            "Candidate LFM2 layers do not share one FFN width."
        )
    width = next(iter(widths))
    if (
        getattr(config, "intermediate_size", None) != width
        or getattr(config, "block_auto_adjust_ff_dim", None) is not False
    ):
        raise Lfm2CpcfV5QuarantineError(
            "Candidate LFM2 FFN shape is not sealed by its config."
        )
    return width


def _restore_ffn_alias(model: Any, *, expected_width: int, scope: str) -> dict[str, Any]:
    _capture, cpcf_runner, _local = _runtime_modules()
    try:
        audit = cpcf_runner._restore_lfm2_ffn_config_alias(model, scope)
    except BaseException as error:
        raise Lfm2CpcfV5QuarantineError(
            "Could not restore the shape-stable LFM2 FFN config alias."
        ) from error
    normalized = _mapping(audit, "LFM2 FFN alias audit")
    if (
        normalized.get("materialized_ffn_width") != expected_width
        or normalized.get("shape_stable") is not True
        or normalized.get("serialized_aliases")
        != {
            "intermediate_size": expected_width,
            "block_ff_dim": expected_width,
            "block_auto_adjust_ff_dim": False,
        }
    ):
        raise Lfm2CpcfV5QuarantineError(
            "Restored LFM2 FFN aliases differ from materialized state."
        )
    return normalized


def _snapshot_ffn_config_aliases(config: Any) -> tuple[tuple[str, bool, Any], ...]:
    return tuple(
        (name, hasattr(config, name), getattr(config, name, None))
        for name in (
            "intermediate_size",
            "block_ff_dim",
            "block_auto_adjust_ff_dim",
            "_name_or_path",
        )
    )


def _restore_ffn_config_aliases(
    config: Any,
    snapshot: Sequence[tuple[str, bool, Any]],
) -> None:
    for name, existed, value in snapshot:
        if existed:
            setattr(config, name, value)
        elif hasattr(config, name):
            delattr(config, name)


def _verify_saved_ffn_config(directory: Path, *, expected_width: int) -> dict[str, Any]:
    _capture, cpcf_runner, _local = _runtime_modules()
    try:
        result = cpcf_runner._verify_saved_lfm2_ffn_config(
            directory,
            expected_width=expected_width,
        )
    except BaseException as error:
        raise Lfm2CpcfV5QuarantineError(
            "Saved LFM2 config cannot rebuild the exact FFN shape."
        ) from error
    return _mapping(result, "saved LFM2 FFN config audit")


def _dense_safetensor_files(directory: Path) -> tuple[dict[str, Any], ...]:
    forbidden = tuple(directory.glob("*.bin")) + tuple(directory.glob("*.pt"))
    if forbidden:
        raise Lfm2CpcfV5QuarantineError(
            "Standalone child must use dense safetensors only."
        )
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Lfm2CpcfV5QuarantineError(
                "Safetensors index is invalid."
            ) from error
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise Lfm2CpcfV5QuarantineError(
                "Safetensors index contains no weight map."
            )
        paths = tuple(
            directory / str(name) for name in sorted(set(weight_map.values()))
        )
    else:
        single = directory / "model.safetensors"
        paths = (
            (single,)
            if single.is_file()
            else tuple(sorted(directory.glob("model-*.safetensors")))
        )
    if not paths or any(
        path.is_symlink() or not path.is_file() for path in paths
    ):
        raise Lfm2CpcfV5QuarantineError(
            "Dense safetensors are missing or unsafe."
        )
    rows = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        rows.append(
            {
                "path": path.name,
                "size_bytes": int(path.stat().st_size),
                "sha256": digest.hexdigest(),
            }
        )
    return tuple(rows)


def _tokenizer_package(directory: Path) -> tuple[str, tuple[dict[str, Any], ...], dict[str, Any]]:
    posthoc = _posthoc_runtime()
    try:
        package_sha, files = posthoc.fingerprint_bound_local_tokenizer(
            directory
        )
        special = posthoc.load_bound_special_token_ids(
            directory,
            expected_tokenizer_fingerprint=package_sha,
        )
    except BaseException as error:
        raise Lfm2CpcfV5QuarantineError(
            "Saved target tokenizer package could not be authenticated."
        ) from error
    return (
        _digest(package_sha, "target tokenizer package SHA-256"),
        tuple(_mapping(item, "target tokenizer package file") for item in files),
        _mapping(special, "target tokenizer special-token identity"),
    )


def _artifact_parameter_inventory(value: Mapping[str, Any]) -> dict[str, Any]:
    inventory = _mapping(value, "parameter inventory")
    if inventory.get("checkpoint_byte_size_used") is not False:
        raise Lfm2CpcfV5QuarantineError(
            "Parameter inventory improperly relies on checkpoint byte size."
        )
    # The canonical standalone-artifact scanner correctly rejects any JSON
    # key containing ``checkpoint``.  The externally authenticated inventory
    # seal remains in full; only that redundant negative claim is omitted from
    # the child-owned metadata document.
    inventory.pop("checkpoint_byte_size_used")
    return inventory


def _normalized_eos_ids(tokenizer: Any) -> tuple[int, ...]:
    value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
    else:
        values = (value,)
    result = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise Lfm2CpcfV5QuarantineError(
                "Target tokenizer has no valid EOS identity."
            )
        if item not in result:
            result.append(item)
    if not result:
        raise Lfm2CpcfV5QuarantineError(
            "Target tokenizer has no valid EOS identity."
        )
    return tuple(result)


class Lfm2CpcfV5QuarantineSaver:
    """Authenticated dense saver with an opaque live tokenizer resource."""

    __slots__ = (
        "_target_tokenizer",
        "_tokenizer_loader",
        "context_binding_sha256",
        "expected_ffn_width",
        "expected_model_state_sha256",
        "expected_parameter_inventory_sha256",
        "expected_tokenizer_semantics_sha256",
        "resource_fingerprint",
    )

    def __init__(
        self,
        *,
        target_tokenizer: Any,
        context_binding_sha256: str,
        expected_ffn_width: int,
        expected_model_state_sha256: str,
        expected_parameter_inventory_sha256: str,
        expected_tokenizer_semantics_sha256: str,
        tokenizer_loader: Callable[[Path], Any] = _default_tokenizer_loader,
    ) -> None:
        if not callable(tokenizer_loader):
            raise Lfm2CpcfV5QuarantineError(
                "Direct local tokenizer loader must be callable."
            )
        self._target_tokenizer = target_tokenizer
        self._tokenizer_loader = tokenizer_loader
        self.context_binding_sha256 = _digest(
            context_binding_sha256,
            "context_binding_sha256",
        )
        self.expected_ffn_width = _positive_int(
            expected_ffn_width,
            "expected_ffn_width",
        )
        self.expected_model_state_sha256 = _digest(
            expected_model_state_sha256,
            "expected_model_state_sha256",
        )
        self.expected_parameter_inventory_sha256 = _digest(
            expected_parameter_inventory_sha256,
            "expected_parameter_inventory_sha256",
        )
        self.expected_tokenizer_semantics_sha256 = _digest(
            expected_tokenizer_semantics_sha256,
            "expected_tokenizer_semantics_sha256",
        )
        body = {
            "schema": LFM2_CPCF_V5_QUARANTINE_BACKEND_SCHEMA,
            "role": "dense-saver",
            "version": 1,
            "context_binding_sha256": self.context_binding_sha256,
            "expected_ffn_width": self.expected_ffn_width,
            "expected_model_state_sha256": self.expected_model_state_sha256,
            "expected_parameter_inventory_sha256": (
                self.expected_parameter_inventory_sha256
            ),
            "expected_tokenizer_semantics_sha256": (
                self.expected_tokenizer_semantics_sha256
            ),
            "serialization": "hf-save-pretrained-dense-safetensors",
            "tokenizer_serialization": "target-tokenizer-save-pretrained",
            "tokenizer_loader_sha256": _callable_fingerprint(
                tokenizer_loader,
                "saver tokenizer reload",
            ),
        }
        self.resource_fingerprint = canonical_sha256(body)

    def __call__(self, model: Any, destination: Path) -> None:
        directory = _ordinary_empty_directory(destination)
        if fingerprint_model_state(model) != self.expected_model_state_sha256:
            raise Lfm2CpcfV5QuarantineError(
                "Saver received a different compiled child state."
            )
        inventory = _mapping(
            independent_parameter_inventory(model),
            "compiled child parameter inventory",
        )
        if (
            inventory.get("inventory_sha256")
            != self.expected_parameter_inventory_sha256
        ):
            raise Lfm2CpcfV5QuarantineError(
                "Saver received a different compiled child topology."
            )
        width = _model_ffn_width(model)
        if width != self.expected_ffn_width:
            raise Lfm2CpcfV5QuarantineError(
                "Saver received a different materialized FFN width."
            )
        semantics = fingerprint_hf_tokenizer_semantics(
            self._target_tokenizer
        )
        if semantics != self.expected_tokenizer_semantics_sha256:
            raise Lfm2CpcfV5QuarantineError(
                "Saver target tokenizer semantics changed."
            )
        save_model = getattr(model, "save_pretrained", None)
        save_tokenizer = getattr(
            self._target_tokenizer,
            "save_pretrained",
            None,
        )
        if not callable(save_model) or not callable(save_tokenizer):
            raise Lfm2CpcfV5QuarantineError(
                "Child model or target tokenizer lacks save_pretrained()."
            )
        config_snapshot = _snapshot_ffn_config_aliases(model.config)
        try:
            # A cloned local parent config normally retains its source path.
            # That path is neither needed for standalone reconstruction nor
            # allowed in a child artifact.
            if hasattr(model.config, "_name_or_path"):
                model.config._name_or_path = ""
            alias = _restore_ffn_alias(
                model,
                expected_width=width,
                scope="lfm2_cpcf_v5_quarantine_saver",
            )
            save_model(directory, safe_serialization=True)
        finally:
            _restore_ffn_config_aliases(model.config, config_snapshot)
        save_tokenizer(directory)
        saved_alias = _verify_saved_ffn_config(
            directory,
            expected_width=width,
        )
        safetensors = _dense_safetensor_files(directory)
        package_sha, package_files, special = _tokenizer_package(directory)
        loaded_tokenizer = self._tokenizer_loader(directory)
        loaded_semantics = fingerprint_hf_tokenizer_semantics(
            loaded_tokenizer
        )
        if loaded_semantics != self.expected_tokenizer_semantics_sha256:
            raise Lfm2CpcfV5QuarantineError(
                "Serialized target tokenizer changed row/token semantics."
            )
        body = {
            "schema": LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA,
            "backend_schema": LFM2_CPCF_V5_QUARANTINE_BACKEND_SCHEMA,
            "context_binding_sha256": self.context_binding_sha256,
            "model_state_sha256": self.expected_model_state_sha256,
            "parameter_inventory": _artifact_parameter_inventory(inventory),
            "parameter_inventory_sha256": (
                self.expected_parameter_inventory_sha256
            ),
            "materialized_ffn_width": width,
            "serialized_ffn_aliases": saved_alias,
            "tokenizer_package_sha256": package_sha,
            "tokenizer_package_files": list(package_files),
            "tokenizer_semantics_sha256": (
                self.expected_tokenizer_semantics_sha256
            ),
            "tokenizer_special_token_ids": special,
            "dense_safetensor_files": list(safetensors),
            "dense_safetensors_only": True,
            "target_tokenizer_saved_with_child": True,
            "local_reload_only": True,
            "network_access": False,
            "hub_loader_used": False,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        }
        document = {**body, "artifact_sha256": canonical_sha256(body)}
        _atomic_create_json(
            directory / LFM2_CPCF_V5_ARTIFACT_METADATA,
            document,
        )
        if alias.get("shape_stable") is not True:
            raise Lfm2CpcfV5QuarantineError(
                "In-memory LFM2 FFN aliases were not shape stable."
            )


class Lfm2CpcfV5StrictLocalLoader:
    """Authenticated direct-local loader; no Auto/HF-hub path is available."""

    __slots__ = (
        "_model_loader",
        "_tokenizer_loader",
        "context_binding_sha256",
        "device",
        "dtype",
        "expected_ffn_width",
        "expected_model_state_sha256",
        "expected_parameter_inventory_sha256",
        "expected_tokenizer_semantics_sha256",
        "resource_fingerprint",
    )

    def __init__(
        self,
        *,
        context_binding_sha256: str,
        device: str,
        dtype: str,
        expected_ffn_width: int,
        expected_model_state_sha256: str,
        expected_parameter_inventory_sha256: str,
        expected_tokenizer_semantics_sha256: str,
        model_loader: Callable[..., Any] = _default_model_loader,
        tokenizer_loader: Callable[[Path], Any] = _default_tokenizer_loader,
    ) -> None:
        if not isinstance(device, str) or not device.strip():
            raise Lfm2CpcfV5QuarantineError("device must be nonempty.")
        if not isinstance(dtype, str) or not dtype.strip():
            raise Lfm2CpcfV5QuarantineError("dtype must be nonempty.")
        if not callable(model_loader) or not callable(tokenizer_loader):
            raise Lfm2CpcfV5QuarantineError(
                "Strict local loader implementations must be callable."
            )
        self._model_loader = model_loader
        self._tokenizer_loader = tokenizer_loader
        self.context_binding_sha256 = _digest(
            context_binding_sha256,
            "context_binding_sha256",
        )
        self.device = device.strip()
        self.dtype = dtype.strip().lower()
        self.expected_ffn_width = _positive_int(
            expected_ffn_width,
            "expected_ffn_width",
        )
        self.expected_model_state_sha256 = _digest(
            expected_model_state_sha256,
            "expected_model_state_sha256",
        )
        self.expected_parameter_inventory_sha256 = _digest(
            expected_parameter_inventory_sha256,
            "expected_parameter_inventory_sha256",
        )
        self.expected_tokenizer_semantics_sha256 = _digest(
            expected_tokenizer_semantics_sha256,
            "expected_tokenizer_semantics_sha256",
        )
        body = {
            "schema": LFM2_CPCF_V5_QUARANTINE_BACKEND_SCHEMA,
            "role": "strict-local-loader",
            "version": 1,
            "context_binding_sha256": self.context_binding_sha256,
            "device": self.device,
            "dtype": self.dtype,
            "expected_ffn_width": self.expected_ffn_width,
            "expected_model_state_sha256": self.expected_model_state_sha256,
            "expected_parameter_inventory_sha256": (
                self.expected_parameter_inventory_sha256
            ),
            "expected_tokenizer_semantics_sha256": (
                self.expected_tokenizer_semantics_sha256
            ),
            "model_loader_sha256": _callable_fingerprint(
                model_loader,
                "direct local model loader",
            ),
            "tokenizer_loader_sha256": _callable_fingerprint(
                tokenizer_loader,
                "direct local tokenizer loader",
            ),
            "auto_loader_used": False,
            "from_pretrained_used": False,
            "hub_loader_used": False,
            "network_access": False,
        }
        self.resource_fingerprint = canonical_sha256(body)

    def __call__(self, source: Path) -> Any:
        directory = _ordinary_existing_directory(source)
        metadata = _read_artifact_metadata(directory)
        if (
            metadata["schema"] != LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA
            or metadata["backend_schema"]
            != LFM2_CPCF_V5_QUARANTINE_BACKEND_SCHEMA
            or metadata["context_binding_sha256"]
            != self.context_binding_sha256
            or metadata["model_state_sha256"]
            != self.expected_model_state_sha256
            or metadata["parameter_inventory_sha256"]
            != self.expected_parameter_inventory_sha256
            or metadata["materialized_ffn_width"] != self.expected_ffn_width
            or metadata["tokenizer_semantics_sha256"]
            != self.expected_tokenizer_semantics_sha256
            or metadata["dense_safetensors_only"] is not True
            or metadata["target_tokenizer_saved_with_child"] is not True
            or metadata["local_reload_only"] is not True
            or metadata["network_access"] is not False
            or metadata["hub_loader_used"] is not False
        ):
            raise Lfm2CpcfV5QuarantineError(
                "Standalone LFM2 artifact identities differ from the loader."
            )
        _verify_saved_ffn_config(
            directory,
            expected_width=self.expected_ffn_width,
        )
        if list(_dense_safetensor_files(directory)) != metadata[
            "dense_safetensor_files"
        ]:
            raise Lfm2CpcfV5QuarantineError(
                "Standalone LFM2 safetensor identities changed."
            )
        package_sha, package_files, special = _tokenizer_package(directory)
        if (
            package_sha != metadata["tokenizer_package_sha256"]
            or list(package_files) != metadata["tokenizer_package_files"]
            or special != metadata["tokenizer_special_token_ids"]
        ):
            raise Lfm2CpcfV5QuarantineError(
                "Standalone target tokenizer package changed."
            )
        try:
            model = self._model_loader(
                directory,
                device=self.device,
                dtype=self.dtype,
            )
            tokenizer = self._tokenizer_loader(directory)
        except BaseException as error:
            raise Lfm2CpcfV5QuarantineError(
                "Direct-local standalone reload failed."
            ) from error
        requires_grad = getattr(model, "requires_grad_", None)
        evaluate = getattr(model, "eval", None)
        if not callable(requires_grad) or not callable(evaluate):
            raise Lfm2CpcfV5QuarantineError(
                "Fresh child does not expose inference-only model controls."
            )
        requires_grad(False)
        evaluate()
        if fingerprint_hf_tokenizer_semantics(
            tokenizer
        ) != self.expected_tokenizer_semantics_sha256:
            raise Lfm2CpcfV5QuarantineError(
                "Fresh target tokenizer semantics changed."
            )
        alias = _restore_ffn_alias(
            model,
            expected_width=self.expected_ffn_width,
            scope="lfm2_cpcf_v5_quarantine_fresh_reload",
        )
        if (
            _model_ffn_width(model) != self.expected_ffn_width
            or fingerprint_model_state(model)
            != self.expected_model_state_sha256
        ):
            raise Lfm2CpcfV5QuarantineError(
                "Fresh LFM2 child state or FFN shape differs from capture."
            )
        inventory = _mapping(
            independent_parameter_inventory(model),
            "fresh child parameter inventory",
        )
        if (
            inventory.get("inventory_sha256")
            != self.expected_parameter_inventory_sha256
            or _artifact_parameter_inventory(inventory)
            != metadata["parameter_inventory"]
        ):
            raise Lfm2CpcfV5QuarantineError(
                "Fresh LFM2 child parameter inventory differs from capture."
            )
        setattr(model, "_neural_forge_quarantine_tokenizer", tokenizer)
        setattr(
            model,
            "_neural_forge_quarantine_load_audit",
            {
                "schema": LFM2_CPCF_V5_QUARANTINE_BACKEND_SCHEMA,
                "context_binding_sha256": self.context_binding_sha256,
                "artifact_sha256": metadata["artifact_sha256"],
                "model_state_sha256": self.expected_model_state_sha256,
                "parameter_inventory_sha256": (
                    self.expected_parameter_inventory_sha256
                ),
                "ffn_alias_report_sha256": alias["report_sha256"],
                "tokenizer_package_sha256": package_sha,
                "tokenizer_semantics_sha256": (
                    self.expected_tokenizer_semantics_sha256
                ),
                "direct_local_json_and_safetensors": True,
                "from_pretrained_used": False,
                "hub_loader_used": False,
                "network_access": False,
            },
        )
        return model


class Lfm2CpcfV5HoldoutEvaluator:
    """Cached-parent 64-case whole-output evaluator with manual review pending."""

    __slots__ = (
        "_candidate_generator",
        "_parent_envelope",
        "_policy",
        "_target_tokenizer",
        "context_binding_sha256",
        "expected_case_count",
        "expected_target_tokenizer_semantics_sha256",
        "resource_fingerprint",
    )

    def __init__(
        self,
        *,
        target_tokenizer: Any,
        parent_envelope: Lfm2IntelligenceHoldoutParentEnvelope,
        context_binding_sha256: str,
        expected_target_tokenizer_semantics_sha256: str,
        expected_case_count: int = LFM2_CPCF_V5_EXPECTED_HOLDOUT_CASES,
        policy: IntelligenceGatePolicy | None = None,
        candidate_generator: Callable[
            [Any, Any, Any],
            Sequence[Mapping[str, Any]],
        ] = _default_candidate_generator,
    ) -> None:
        if type(parent_envelope) is not Lfm2IntelligenceHoldoutParentEnvelope:
            raise Lfm2CpcfV5QuarantineError(
                "parent_envelope must be an exact authenticated LFM2 envelope."
            )
        selected_policy = IntelligenceGatePolicy() if policy is None else policy
        if type(selected_policy) is not IntelligenceGatePolicy:
            raise Lfm2CpcfV5QuarantineError(
                "policy must be an exact IntelligenceGatePolicy."
            )
        if not callable(candidate_generator):
            raise Lfm2CpcfV5QuarantineError(
                "candidate_generator must be callable."
            )
        self._target_tokenizer = target_tokenizer
        self._parent_envelope = parent_envelope
        self._policy = selected_policy
        self._candidate_generator = candidate_generator
        self.context_binding_sha256 = _digest(
            context_binding_sha256,
            "context_binding_sha256",
        )
        self.expected_target_tokenizer_semantics_sha256 = _digest(
            expected_target_tokenizer_semantics_sha256,
            "expected_target_tokenizer_semantics_sha256",
        )
        self.expected_case_count = _positive_int(
            expected_case_count,
            "expected_case_count",
        )
        if len(parent_envelope.manifest.cases) != self.expected_case_count:
            raise Lfm2CpcfV5QuarantineError(
                "Cached parent envelope does not cover the exact holdout size."
            )
        if fingerprint_hf_tokenizer_semantics(
            target_tokenizer
        ) != self.expected_target_tokenizer_semantics_sha256:
            raise Lfm2CpcfV5QuarantineError(
                "Live target tokenizer differs from its bridge semantics."
            )
        body = {
            "schema": LFM2_CPCF_V5_QUARANTINE_BACKEND_SCHEMA,
            "role": "cached-whole-output-evaluator",
            "version": 1,
            "context_binding_sha256": self.context_binding_sha256,
            "parent_envelope_sha256": parent_envelope.envelope_sha256,
            "parent_envelope_document_sha256": (
                parent_envelope.document_sha256
            ),
            "manifest_sha256": parent_envelope.manifest.manifest_sha256,
            "expected_case_count": self.expected_case_count,
            "parent_tokenizer_semantics_sha256": (
                parent_envelope.tokenizer_fingerprint
            ),
            "target_tokenizer_semantics_sha256": (
                self.expected_target_tokenizer_semantics_sha256
            ),
            "policy_sha256": selected_policy.fingerprint,
            "candidate_generator_sha256": _callable_fingerprint(
                candidate_generator,
                "candidate whole-output generator",
            ),
            "parent_cache_only": True,
            "exact_token_criterion": False,
            "exact_wording_criterion": False,
            "manual_cases_pending": True,
        }
        self.resource_fingerprint = canonical_sha256(body)

    def __call__(self, model: Any) -> QuarantineEvaluation:
        tokenizer = getattr(
            model,
            "_neural_forge_quarantine_tokenizer",
            self._target_tokenizer,
        )
        target_semantics = fingerprint_hf_tokenizer_semantics(tokenizer)
        if target_semantics != self.expected_target_tokenizer_semantics_sha256:
            raise Lfm2CpcfV5QuarantineError(
                "Evaluator target tokenizer semantics changed."
            )
        eos_token_ids = _normalized_eos_ids(tokenizer)
        rows = tuple(
            self._candidate_generator(
                model,
                tokenizer,
                self._parent_envelope.manifest,
            )
        )
        if len(rows) != self.expected_case_count:
            raise Lfm2CpcfV5QuarantineError(
                "Candidate generation did not cover all holdout cases."
            )
        candidates = candidate_outputs_from_lfm2_holdout_rows(
            self._parent_envelope.manifest,
            rows,
            eos_token_ids=eos_token_ids,
        )
        baseline = baseline_from_lfm2_intelligence_holdout_envelope(
            self._parent_envelope
        )
        report = evaluate_cached_intelligence(
            baseline,
            candidates,
            policy=self._policy,
            manual_ratings=None,
        )
        report_document = report.as_dict()
        if (
            report_document.get("baseline_document_sha256")
            != baseline.document_sha256
            or report_document.get("candidate_outputs_sha256")
            != report.candidate_outputs_sha256
            or report_document.get("manual_review", {}).get("status")
            != "pending"
            or report_document.get("manual_review", {}).get(
                "exact_wording_comparison_used"
            )
            is not False
        ):
            raise Lfm2CpcfV5QuarantineError(
                "Cached intelligence report is not pending exact manual review."
            )
        normalized_rows = [_mapping(row, "candidate holdout row") for row in rows]
        output = {
            "schema": LFM2_CPCF_V5_QUARANTINE_OUTPUT_SCHEMA,
            "context_binding_sha256": self.context_binding_sha256,
            "manifest_sha256": (
                self._parent_envelope.manifest.manifest_sha256
            ),
            "parent_envelope_sha256": self._parent_envelope.envelope_sha256,
            "case_count": len(normalized_rows),
            "rows": normalized_rows,
            "candidate_outputs": [item.as_dict() for item in candidates],
            "full_decoded_outputs_retained": True,
            "per_case_manifest_generation_limits_used": True,
            "target_tokenizer_semantics_sha256": target_semantics,
            "parent_tokenizer_semantics_sha256": (
                self._parent_envelope.tokenizer_fingerprint
            ),
            "cross_tokenizer_decoded_semantics_supported": True,
            "cross_tokenizer_semantics_active": (
                target_semantics
                != self._parent_envelope.tokenizer_fingerprint
            ),
            "exact_parent_token_matching_used": False,
            "exact_parent_wording_matching_used": False,
            "parent_model_loaded": False,
            "parent_forward_calls": 0,
        }
        evaluation = {
            "schema": LFM2_CPCF_V5_QUARANTINE_EVALUATION_SCHEMA,
            "context_binding_sha256": self.context_binding_sha256,
            "gate_report": report_document,
            "gate_report_sha256": report.report_sha256,
            "candidate_outputs_sha256": report.candidate_outputs_sha256,
            "manual_review_status": "pending",
            "manual_review_packet_sha256": report.manual_review[
                "packet_sha256"
            ],
            "automatic_promotion_authorized": False,
            "quarantine_only": True,
            "semantic_whole_output_criterion": True,
            "exact_token_criterion": False,
            "exact_wording_criterion": False,
            "parent_forward_calls": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }
        return QuarantineEvaluation(output=output, evaluation=evaluation)


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5QuarantineConfig:
    """Immutable external inputs owned by the quarantine consumer."""

    destination: Path
    parent_envelope: Lfm2IntelligenceHoldoutParentEnvelope
    target_tokenizer_semantics_sha256: str
    device: str
    dtype: str
    expected_output_sha256: str | None = None
    expected_case_count: int = LFM2_CPCF_V5_EXPECTED_HOLDOUT_CASES
    policy: IntelligenceGatePolicy | None = None
    model_loader: Callable[..., Any] = _default_model_loader
    tokenizer_loader: Callable[[Path], Any] = _default_tokenizer_loader
    candidate_generator: Callable[
        [Any, Any, Any],
        Sequence[Mapping[str, Any]],
    ] = _default_candidate_generator

    def __post_init__(self) -> None:
        if not isinstance(self.destination, Path):
            object.__setattr__(self, "destination", Path(self.destination))
        if type(self.parent_envelope) is not Lfm2IntelligenceHoldoutParentEnvelope:
            raise Lfm2CpcfV5QuarantineError(
                "parent_envelope must be an exact authenticated envelope."
            )
        _digest(
            self.target_tokenizer_semantics_sha256,
            "target_tokenizer_semantics_sha256",
        )
        if self.expected_output_sha256 is not None:
            _digest(self.expected_output_sha256, "expected_output_sha256")
        _positive_int(self.expected_case_count, "expected_case_count")
        if not isinstance(self.device, str) or not self.device.strip():
            raise Lfm2CpcfV5QuarantineError("device must be nonempty.")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise Lfm2CpcfV5QuarantineError("dtype must be nonempty.")
        if self.policy is not None and type(self.policy) is not IntelligenceGatePolicy:
            raise Lfm2CpcfV5QuarantineError(
                "policy must be an exact IntelligenceGatePolicy."
            )
        if not all(
            callable(value)
            for value in (
                self.model_loader,
                self.tokenizer_loader,
                self.candidate_generator,
            )
        ):
            raise Lfm2CpcfV5QuarantineError(
                "Configured local loader and evaluator seams must be callable."
            )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5QuarantineRuntime:
    """The exact generic request and callbacks used for one capture."""

    request: QuarantineCaptureRequest
    saver: Lfm2CpcfV5QuarantineSaver
    loader: Lfm2CpcfV5StrictLocalLoader
    evaluator: Lfm2CpcfV5HoldoutEvaluator
    context_binding_sha256: str
    expected_ffn_width: int


class Lfm2CpcfV5QuarantineConsumer:
    """Fixed runner-seam consumer that returns the runner's exact result type."""

    __slots__ = ("config", "resource_fingerprint")

    def __init__(self, config: Lfm2CpcfV5QuarantineConfig) -> None:
        if type(config) is not Lfm2CpcfV5QuarantineConfig:
            raise Lfm2CpcfV5QuarantineError(
                "config must be an exact Lfm2CpcfV5QuarantineConfig."
            )
        self.config = config
        body = {
            "schema": LFM2_CPCF_V5_QUARANTINE_BACKEND_SCHEMA,
            "role": "compiled-candidate-consumer",
            "version": 1,
            "destination_path_sha256": canonical_sha256(
                os.fspath(config.destination.absolute())
            ),
            "parent_envelope_sha256": config.parent_envelope.envelope_sha256,
            "parent_envelope_document_sha256": (
                config.parent_envelope.document_sha256
            ),
            "manifest_sha256": config.parent_envelope.manifest.manifest_sha256,
            "target_tokenizer_semantics_sha256": (
                config.target_tokenizer_semantics_sha256
            ),
            "device": config.device,
            "dtype": config.dtype,
            "expected_output_sha256": config.expected_output_sha256,
            "expected_case_count": config.expected_case_count,
            "policy_sha256": (
                IntelligenceGatePolicy().fingerprint
                if config.policy is None
                else config.policy.fingerprint
            ),
            "model_loader_sha256": _callable_fingerprint(
                config.model_loader,
                "consumer model loader",
            ),
            "tokenizer_loader_sha256": _callable_fingerprint(
                config.tokenizer_loader,
                "consumer tokenizer loader",
            ),
            "candidate_generator_sha256": _callable_fingerprint(
                config.candidate_generator,
                "consumer candidate generator",
            ),
            "quarantine_only": True,
            "publication_authorized": False,
        }
        self.resource_fingerprint = canonical_sha256(body)

    def __call__(self, context: Any) -> Any:
        candidate = capture_lfm2_cpcf_v5_quarantine(context, self.config)
        try:
            from experiments.run_lfm2_cpcf_v5_selective_blank_fold import (
                V5CompiledCandidateConsumerResult,
            )
        except ImportError as error:  # pragma: no cover - packaging boundary
            raise Lfm2CpcfV5QuarantineError(
                "The exact v5 compiled-candidate result type is unavailable."
            ) from error
        receipt = candidate.receipt.as_dict()
        return V5CompiledCandidateConsumerResult(
            candidate=candidate,
            capture_receipt=receipt,
            capture_receipt_sha256=candidate.receipt.receipt_sha256,
            compiler_receipt_sha256=context.compiler_receipt_sha256,
            target_plan_fingerprint=context.target_plan_fingerprint,
            target_state_sha256=context.target_state_sha256,
            parent_sha256=context.parent_sha256,
            cache_sha256=context.cache_sha256,
            bridge_sha256=context.bridge_sha256,
            corpus_sha256=context.corpus_sha256,
            source_sha256=context.source_sha256,
            compiler_sha256=context.compiler_sha256,
            runner_source_sha256=context.runner_source_sha256,
            source_binding_sha256=context.source_binding_sha256,
            parent_tokenizer_semantics_sha256=(
                context.parent_tokenizer_semantics_sha256
            ),
            target_tokenizer_semantics_sha256=(
                context.target_tokenizer_semantics_sha256
            ),
            compile_diagnostic=context.compile_diagnostic,
            compiler_child_checkpoint_saved=False,
            compiler_publication_attempted=False,
            quarantine_candidate_saved=True,
            quarantine_publication_nonpromotable=True,
        )


def _require_context(context: Any) -> None:
    try:
        from experiments.run_lfm2_cpcf_v5_selective_blank_fold import (
            V5CompiledCandidateConsumerContext,
        )
    except ImportError as error:  # pragma: no cover - packaging boundary
        raise Lfm2CpcfV5QuarantineError(
            "The canonical v5 compiled-candidate context type is unavailable."
        ) from error
    if type(context) is not V5CompiledCandidateConsumerContext:
        raise Lfm2CpcfV5QuarantineError(
            "Context is not the exact canonical v5 compiled-candidate "
            "consumer context."
        )
    missing = [name for name in _CONTEXT_FIELDS if not hasattr(context, name)]
    if missing:
        raise Lfm2CpcfV5QuarantineError(
            f"V5 compiled-candidate context lacks fields: {missing}."
        )


def _authenticated_prior_audit(context: Any) -> QuarantineZeroOperationAudit:
    report = _sealed_mapping(
        context.compiled_model_fold_report,
        name="compiled v5 model-fold receipt",
        seal_field="result_sha256",
        expected_sha256=context.compiler_receipt_sha256,
    )
    operation = _mapping(
        report.get("operation_audit"),
        "compiled v5 operation audit",
    )
    compiler_implementation = _mapping(
        report.get("compiler_implementation"),
        "compiled v5 compiler implementation",
    )
    declared = _mapping(
        context.zero_operation_audit,
        "consumer context zero-operation audit",
    )
    compiler_runtime_guard = _sealed_mapping(
        declared.get("compiler_runtime_guard"),
        name="compiler runtime fold-only guard",
        seal_field="report_sha256",
    )
    guard_implementation = _mapping(
        compiler_runtime_guard.get("guard_implementation"),
        "compiler runtime fold-only guard implementation",
    )
    guard_identity_sha256 = _digest(
        guard_implementation.get("identity_sha256"),
        "fold-only runtime guard identity SHA-256",
    )
    source_binding = canonical_sha256(
        {
            "selective_source_sha256": _digest(
                context.source_sha256,
                "selective source SHA-256",
            ),
            "compiler_implementation_sha256": _digest(
                context.compiler_sha256,
                "compiler implementation SHA-256",
            ),
            "runner_source_sha256": _digest(
                context.runner_source_sha256,
                "runner source SHA-256",
            ),
            "fold_only_runtime_guard_identity_sha256": (
                guard_identity_sha256
            ),
        }
    )
    if (
        report.get("selective_source_sha256") != context.source_sha256
        or compiler_implementation.get("implementation_sha256")
        != context.compiler_sha256
        or source_binding != context.source_binding_sha256
    ):
        raise Lfm2CpcfV5QuarantineError(
            "Compiled v5 source, implementation, or runner composite binding "
            "is not authenticated."
        )
    if set(operation) != _COMPILED_AUDIT_ZERO_FIELDS:
        raise Lfm2CpcfV5QuarantineError(
            "Compiled v5 receipt operation-audit fields are not exact."
        )
    for name in _COMPILED_AUDIT_ZERO_FIELDS:
        if type(operation[name]) is not int or operation[name] != 0:
            raise Lfm2CpcfV5QuarantineError(
                f"Compiled v5 receipt has nonzero {name}."
            )
    if report.get("source_state_preserved") is not True:
        raise Lfm2CpcfV5QuarantineError(
            "Compiled v5 receipt does not preserve the parent state."
        )
    expected_schema = getattr(
        __import__(
            "experiments.run_lfm2_cpcf_v5_selective_blank_fold",
            fromlist=["COMPILED_CANDIDATE_ZERO_AUDIT_SCHEMA"],
        ),
        "COMPILED_CANDIDATE_ZERO_AUDIT_SCHEMA",
    )
    if (
        declared.get("schema") != expected_schema
        or declared.get("clean") is not True
        or any(
            type(declared.get(name)) is not int
            or declared.get(name) != 0
            for name in _CONTEXT_ZERO_COUNTER_FIELDS
        )
        or declared.get("training_used") is not False
        or declared.get("pruning_used") is not False
        or declared.get("quantization_used") is not False
        or declared.get("live_parent_calls_used") is not False
    ):
        raise Lfm2CpcfV5QuarantineError(
            "Consumer context conflicts with the authenticated zero audit."
        )
    compiler_declared = declared.get("compiler_operation_audit")
    if compiler_declared is not None:
        compiler_declared = _mapping(
            compiler_declared,
            "context compiler operation audit",
        )
        if compiler_declared != operation:
            raise Lfm2CpcfV5QuarantineError(
                "Consumer context conflicts with the authenticated zero audit."
            )
    # These values are derived only from the compiler receipt authenticated
    # above; the context audit is a cross-check and never grants authority.
    return QuarantineZeroOperationAudit(
        optimizer_constructions=0,
        optimizer_steps=operation["optimizer_steps"],
        backward_calls=operation["backward_calls"],
        autograd_grad_calls=0,
        training_steps=operation["training_steps"],
        pruning_operations=operation["parameters_pruned"],
        quantization_operations=operation["parameters_quantized"],
        parent_forward_calls=operation["parent_model_forward_calls"],
        parent_inference_calls=operation["generation_calls"],
    )


def _context_binding(context: Any) -> str:
    blank_report = _mapping(
        context.blank_target_report,
        "blank target report",
    )
    compiled_report = _mapping(
        context.compiled_model_fold_report,
        "compiled model-fold report",
    )
    compile_diagnostic = _mapping(
        context.compile_diagnostic,
        "runner compile diagnostic",
    )
    inventories = {
        "parent": _mapping(
            context.parent_parameter_inventory,
            "parent parameter inventory",
        ),
        "blank": _mapping(
            context.blank_parameter_inventory,
            "blank parameter inventory",
        ),
        "compiled": _mapping(
            context.compiled_parameter_inventory,
            "compiled parameter inventory",
        ),
    }
    body = {
        "schema": LFM2_CPCF_V5_CONTEXT_BINDING_SCHEMA,
        "context_type": (
            f"{type(context).__module__}.{type(context).__qualname__}"
        ),
        "target_plan_fingerprint": _digest(
            context.target_plan_fingerprint,
            "target_plan_fingerprint",
        ),
        "blank_target_report_sha256": canonical_sha256(blank_report),
        "compiler_receipt_sha256": _digest(
            context.compiler_receipt_sha256,
            "compiler_receipt_sha256",
        ),
        "compiled_model_fold_report_sha256": canonical_sha256(
            compiled_report
        ),
        "parent_sha256": _digest(context.parent_sha256, "parent_sha256"),
        "cache_sha256": _digest(context.cache_sha256, "cache_sha256"),
        "bridge_sha256": _digest(context.bridge_sha256, "bridge_sha256"),
        "corpus_sha256": _digest(context.corpus_sha256, "corpus_sha256"),
        "source_sha256": _digest(context.source_sha256, "source_sha256"),
        "compiler_sha256": _digest(
            context.compiler_sha256,
            "compiler_sha256",
        ),
        "runner_source_sha256": _digest(
            context.runner_source_sha256,
            "runner_source_sha256",
        ),
        "source_binding_sha256": _digest(
            context.source_binding_sha256,
            "source_binding_sha256",
        ),
        "parent_tokenizer_semantics_sha256": _digest(
            context.parent_tokenizer_semantics_sha256,
            "parent_tokenizer_semantics_sha256",
        ),
        "target_tokenizer_semantics_sha256": _digest(
            context.target_tokenizer_semantics_sha256,
            "target_tokenizer_semantics_sha256",
        ),
        "target_tokenizer_package_sha256": (
            None
            if context.target_tokenizer_package_sha256 is None
            else _digest(
                context.target_tokenizer_package_sha256,
                "target_tokenizer_package_sha256",
            )
        ),
        "parameter_inventories_sha256": canonical_sha256(inventories),
        "zero_operation_audit_sha256": canonical_sha256(
            _mapping(
                context.zero_operation_audit,
                "context zero-operation audit",
            )
        ),
        "target_state_sha256": _digest(
            context.target_state_sha256,
            "target_state_sha256",
        ),
        "target_runtime_sha256": _digest(
            context.target_runtime_sha256,
            "target_runtime_sha256",
        ),
        "parent_state_sha256": _digest(
            context.parent_state_sha256,
            "parent_state_sha256",
        ),
        "parent_runtime_sha256": _digest(
            context.parent_runtime_sha256,
            "parent_runtime_sha256",
        ),
        "compile_diagnostic_sha256": canonical_sha256(compile_diagnostic),
    }
    return canonical_sha256(body)


def build_lfm2_cpcf_v5_quarantine_runtime(
    context: Any,
    config: Lfm2CpcfV5QuarantineConfig,
) -> Lfm2CpcfV5QuarantineRuntime:
    """Authenticate context and construct the exact generic capture request."""

    _require_context(context)
    if type(config) is not Lfm2CpcfV5QuarantineConfig:
        raise Lfm2CpcfV5QuarantineError(
            "config must be an exact Lfm2CpcfV5QuarantineConfig."
        )
    live_target_state = fingerprint_model_state(context.target_model)
    if live_target_state != context.target_state_sha256:
        raise Lfm2CpcfV5QuarantineError(
            "Consumer context target state is not the generic capture state "
            "fingerprint."
        )
    if (
        fingerprint_model_state(context.forbidden_parent_model)
        != context.parent_state_sha256
    ):
        raise Lfm2CpcfV5QuarantineError(
            "Consumer context parent state changed before quarantine."
        )
    tokenizer_semantics = fingerprint_hf_tokenizer_semantics(
        context.target_tokenizer
    )
    if (
        tokenizer_semantics != config.target_tokenizer_semantics_sha256
        or tokenizer_semantics != context.target_tokenizer_semantics_sha256
    ):
        raise Lfm2CpcfV5QuarantineError(
            "Consumer target tokenizer differs from configured bridge semantics."
        )
    if (
        config.parent_envelope.parent_model_fingerprint
        != context.parent_sha256
        or config.parent_envelope.tokenizer_fingerprint
        != context.parent_tokenizer_semantics_sha256
    ):
        raise Lfm2CpcfV5QuarantineError(
            "Cached parent envelope belongs to a different parent or donor "
            "tokenizer than the authenticated blank-plan/bridge context."
        )
    compiled_inventory = _mapping(
        independent_parameter_inventory(context.target_model),
        "live compiled parameter inventory",
    )
    expected_inventory = _mapping(
        context.compiled_parameter_inventory,
        "context compiled parameter inventory",
    )
    if compiled_inventory != expected_inventory:
        raise Lfm2CpcfV5QuarantineError(
            "Live compiled parameter inventory differs from context."
        )
    inventory_sha = _digest(
        compiled_inventory.get("inventory_sha256"),
        "compiled parameter inventory SHA-256",
    )
    width = _model_ffn_width(context.target_model)
    context_sha = _context_binding(context)
    saver = Lfm2CpcfV5QuarantineSaver(
        target_tokenizer=context.target_tokenizer,
        context_binding_sha256=context_sha,
        expected_ffn_width=width,
        expected_model_state_sha256=live_target_state,
        expected_parameter_inventory_sha256=inventory_sha,
        expected_tokenizer_semantics_sha256=tokenizer_semantics,
        tokenizer_loader=config.tokenizer_loader,
    )
    loader = Lfm2CpcfV5StrictLocalLoader(
        context_binding_sha256=context_sha,
        device=config.device,
        dtype=config.dtype,
        expected_ffn_width=width,
        expected_model_state_sha256=live_target_state,
        expected_parameter_inventory_sha256=inventory_sha,
        expected_tokenizer_semantics_sha256=tokenizer_semantics,
        model_loader=config.model_loader,
        tokenizer_loader=config.tokenizer_loader,
    )
    evaluator = Lfm2CpcfV5HoldoutEvaluator(
        target_tokenizer=context.target_tokenizer,
        parent_envelope=config.parent_envelope,
        context_binding_sha256=context_sha,
        expected_target_tokenizer_semantics_sha256=tokenizer_semantics,
        expected_case_count=config.expected_case_count,
        policy=config.policy,
        candidate_generator=config.candidate_generator,
    )
    request = QuarantineCaptureRequest(
        plan_sha256=context.target_plan_fingerprint,
        parent_sha256=context.parent_sha256,
        cache_sha256=context.cache_sha256,
        bridge_sha256=context.bridge_sha256,
        corpus_sha256=context.corpus_sha256,
        compiler_sha256=context.compiler_receipt_sha256,
        source_sha256=context.source_binding_sha256,
        saver_sha256=fingerprint_quarantine_callable(saver, role="saver"),
        loader_sha256=fingerprint_quarantine_callable(loader, role="loader"),
        evaluator_sha256=fingerprint_quarantine_callable(
            evaluator,
            role="evaluator",
        ),
        expected_model_state_sha256=live_target_state,
        expected_output_sha256=config.expected_output_sha256,
        pre_reviewed_candidate_fingerprint_sha256=None,
        prior_audit=_authenticated_prior_audit(context),
    )
    return Lfm2CpcfV5QuarantineRuntime(
        request=request,
        saver=saver,
        loader=loader,
        evaluator=evaluator,
        context_binding_sha256=context_sha,
        expected_ffn_width=width,
    )


def build_lfm2_cpcf_v5_quarantine_request(
    context: Any,
    config: Lfm2CpcfV5QuarantineConfig,
) -> QuarantineCaptureRequest:
    """Return the authenticated generic request for external inspection."""

    return build_lfm2_cpcf_v5_quarantine_runtime(context, config).request


def capture_lfm2_cpcf_v5_quarantine(
    context: Any,
    config: Lfm2CpcfV5QuarantineConfig,
) -> QuarantinedDynamicFoldCandidate:
    """Save, reload, evaluate, and retain one non-promotable v5 child."""

    runtime = build_lfm2_cpcf_v5_quarantine_runtime(context, config)
    result = capture_quarantined_candidate(
        config.destination,
        candidate=context.target_model,
        request=runtime.request,
        saver=runtime.saver,
        loader=runtime.loader,
        evaluator=runtime.evaluator,
        forbidden_parent_models=(context.forbidden_parent_model,),
    )
    if (
        result.promotable is not False
        or result.publication_authorized is not False
        or result.requires_reviewed_rerun is not True
    ):
        raise Lfm2CpcfV5QuarantineError(
            "Generic capture returned a promotable or publication-authorized "
            "candidate."
        )
    return result


__all__ = [
    "LFM2_CPCF_V5_ARTIFACT_METADATA",
    "LFM2_CPCF_V5_CONTEXT_BINDING_SCHEMA",
    "LFM2_CPCF_V5_EXPECTED_HOLDOUT_CASES",
    "LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA",
    "LFM2_CPCF_V5_QUARANTINE_BACKEND_SCHEMA",
    "LFM2_CPCF_V5_QUARANTINE_EVALUATION_SCHEMA",
    "LFM2_CPCF_V5_QUARANTINE_OUTPUT_SCHEMA",
    "Lfm2CpcfV5HoldoutEvaluator",
    "Lfm2CpcfV5QuarantineConfig",
    "Lfm2CpcfV5QuarantineConsumer",
    "Lfm2CpcfV5QuarantineError",
    "Lfm2CpcfV5QuarantineRuntime",
    "Lfm2CpcfV5QuarantineSaver",
    "Lfm2CpcfV5StrictLocalLoader",
    "build_lfm2_cpcf_v5_quarantine_request",
    "build_lfm2_cpcf_v5_quarantine_runtime",
    "capture_lfm2_cpcf_v5_quarantine",
]
