"""Executable, transactional semantic refold of the complete LFM2 layer stack.

The adapter is deliberately narrower than the model-family-independent planner:
it executes the authenticated fourteen-layer LFM2 child transaction whose
inputs were already normalized by :mod:`hf_lfm2_semantic_postfold`.  Parent
modules are never constructed or called.  Parent activations come only from
the complete sealed cache and parent weights come only from local safetensors.

Every layer is handled in causal order.  Its operator is compiled from the
actual child residual produced by every previously accepted layer, then the
same macrocell is replayed again before its MLP is compiled.  A third replay
propagates the accepted layer to the next layer.  The last MLP's W2 is shared
with the terminal/lexical writer, so it is deferred and returned as four
authenticated terminal-action evidence banks.  Any failure restores the
entire model to its entry bytes before raising.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_cpcf_macrocell_replay import (
    Lfm2CpcfMacrocellReplay,
    replay_lfm2_cpcf_macrocell,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import _rope_from_target
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (
    seal_lfm2_dense_qkvo_source_inventory,
)
from neural_forge.hf_lfm2_semantic_attention_refold_adapter import (
    Lfm2SemanticAttentionReplayEndpoint,
)
from neural_forge.hf_lfm2_semantic_postfold import (
    LFM2_SEMANTIC_ATTENTION_INSTALLATION_SCHEMA,
    Lfm2SemanticAttentionTransportEvidence,
    Lfm2SemanticCompilerLayerLineage,
    Lfm2SemanticLayerStackInstallation,
    _ZERO_OPERATIONS,
    _additive_causal_mask,
    _attention_tensor_names,
    _changed_state_keys,
    _interface_tensor_sha256,
    _lfm2_dense_producer_chain_coverage,
    _local_safetensor_shards,
    _parameter_count,
    _semantic_alignments_from_session,
    _state_tensor_hashes,
    _target_rope,
    _tensor_sha256,
    _validate_attention_installation_receipt,
)
from neural_forge.hf_lfm2_semantic_span_cpcf_alignment import (
    Lfm2SemanticSpanCpcfAlignment,
    audit_lfm2_semantic_span_cpcf_refinement,
)
from neural_forge.hf_semantic_attention_lattice import (
    Lfm2SemanticAttentionCacheCapability,
    SemanticAttentionPreRopeHistory,
    assemble_lfm2_explicit_pre_rope_history,
    inspect_lfm2_semantic_attention_cache_capability,
    transport_lfm2_phase_probability_preserving_attention,
)
from neural_forge.hf_tied_causal_interface_anchor_fold import (
    TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
)
from neural_forge.hf_tied_causal_interface_fold import TiedCausalInterfaceFold
from neural_forge.hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    fingerprint_hf_tokenizer_semantics,
)


LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA = (
    "neural-forge-lfm2-semantic-layer-stack-refold-v1"
)
_TIED_CAUSAL_INTERFACE_FOLD_SCHEMA = "neural-forge-tied-causal-interface-fold-v1"
_EXPECTED_ATTENTION_INDICES = (2, 4, 6, 8, 10, 12)
_EXPECTED_SHORT_CONV_INDICES = (0, 1, 3, 5, 7, 9, 11, 13)
_PARTITIONS = ("fit", "fit_check", "selection", "validation")
_SHORT_CONV_INSTALLATION_SCHEMA = (
    "neural-forge-lfm2-semantic-short-conv-installation-v1"
)
_MLP_INSTALLATION_SCHEMA = "neural-forge-lfm2-semantic-mlp-installation-v1"


class Lfm2SemanticLayerStackRefoldError(RuntimeError):
    """The complete causal transaction failed and was rolled back."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("Semantic layer-stack refolding requires PyTorch.") from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _sealed(body: Mapping[str, Any]) -> Mapping[str, Any]:
    value = dict(body)
    value["result_sha256"] = canonical_sha256(value)
    return MappingProxyType(value)


def _verify_sealed(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    result = dict(value)
    supplied = result.pop("result_sha256", None)
    if supplied != canonical_sha256(result):
        raise Lfm2SemanticLayerStackRefoldError(f"{label} receipt seal is invalid.")
    return {**result, "result_sha256": supplied}


def _validate_producer_coverage_decision(
    *,
    layer_index: int,
    producer_kind: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    installation: Mapping[str, Any],
    defer_down_weight: bool = False,
) -> Mapping[str, Any]:
    """Require cold starts to plant a chain and authenticate refinement no-ops."""

    before_receipt = _verify_sealed(before, "producer coverage before")
    after_receipt = _verify_sealed(after, "producer coverage after")
    expected_deferred = ["feed_forward.w2.weight"] if defer_down_weight else []
    common_invalid = bool(
        before_receipt.get("layer_index") != layer_index
        or after_receipt.get("layer_index") != layer_index
        or before_receipt.get("producer_kind") != producer_kind
        or after_receipt.get("producer_kind") != producer_kind
        or before_receipt.get("deferred_roles") != expected_deferred
        or after_receipt.get("deferred_roles") != expected_deferred
        or before_receipt.get("dense_existing_topology_only") is not True
        or after_receipt.get("dense_existing_topology_only") is not True
        or before_receipt.get("operation_audit") != dict(_ZERO_OPERATIONS)
        or after_receipt.get("operation_audit") != dict(_ZERO_OPERATIONS)
        or after_receipt.get("all_required_roles_structurally_nonzero") is not True
        or after_receipt.get("blank_or_partially_blank_required_role_count") != 0
    )
    installed = installation.get("installed") is True
    model_state_changed = bool(
        installation.get("model_state_before_sha256")
        != installation.get("model_state_after_sha256")
    )
    role_state_changed = bool(
        before_receipt.get("required_role_state_sha256")
        != after_receipt.get("required_role_state_sha256")
    )
    cold_start = bool(
        before_receipt.get("all_required_roles_structurally_nonzero") is not True
    )
    retained_nonzero_seed = bool(
        not installed
        and not cold_start
        and not model_state_changed
        and not role_state_changed
    )
    installed_complete_chain = bool(installed and model_state_changed)
    if common_invalid or not (installed_complete_chain or retained_nonzero_seed):
        if cold_start and not installed_complete_chain:
            reason = "blank-or-partial-producer-chain-was-not-densely-planted"
        elif not installed and not retained_nonzero_seed:
            reason = "producer-noop-did-not-retain-an-authenticated-nonzero-seed"
        else:
            reason = "producer-installation-did-not-prove-complete-coverage"
        raise Lfm2SemanticLayerStackRefoldError(
            f"Layer {layer_index} {producer_kind} coverage rejected: {reason}."
        )
    body = {
        "schema": "neural-forge-lfm2-producer-coverage-decision-v1",
        "layer_index": layer_index,
        "producer_kind": producer_kind,
        "deferred_roles": expected_deferred,
        "coverage_before_sha256": before_receipt["result_sha256"],
        "coverage_after_sha256": after_receipt["result_sha256"],
        "coverage_before": before_receipt,
        "coverage_after": after_receipt,
        "cold_start_or_partial_seed_at_entry": cold_start,
        "installed_complete_structurally_nonzero_chain": installed_complete_chain,
        "authenticated_nonzero_folded_seed_retained_by_noop": retained_nonzero_seed,
        "all_required_roles_structurally_nonzero_after_decision": True,
        "blank_or_partially_blank_required_role_count_after_decision": 0,
        "numeric_parent_child_weight_match_objective": False,
        "training_steps": 0,
        "operation_audit": dict(_ZERO_OPERATIONS),
    }
    return _sealed(body)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _parameter_topology(model: Any) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            str(name),
            id(value),
            int(value.data_ptr()),
            tuple(int(size) for size in value.shape),
            str(value.dtype),
            str(value.device),
            bool(value.requires_grad),
        )
        for name, value in model.named_parameters(remove_duplicate=False)
    )


@dataclass(slots=True)
class _WholeModelSnapshot:
    state: Mapping[str, Any] = field(repr=False)
    gradients: Mapping[str, Any | None] = field(repr=False)
    fingerprint: str
    parameter_count: int
    topology: tuple[tuple[Any, ...], ...]
    modes: tuple[tuple[str, bool], ...]

    @classmethod
    def capture(cls, model: Any) -> _WholeModelSnapshot:
        state = {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in model.state_dict().items()
        }
        gradients = {
            name: (
                None
                if value.grad is None
                else value.grad.detach().to(device="cpu").contiguous().clone()
            )
            for name, value in model.named_parameters()
        }
        return cls(
            state=MappingProxyType(state),
            gradients=MappingProxyType(gradients),
            fingerprint=fingerprint_lfm2_parent(model).identity_sha256,
            parameter_count=_parameter_count(model),
            topology=_parameter_topology(model),
            modes=tuple(
                (str(name), bool(module.training))
                for name, module in model.named_modules()
            ),
        )

    def restore(self, model: Any) -> None:
        torch = _torch()
        current = model.state_dict(keep_vars=True)
        if set(current) != set(self.state):
            raise Lfm2SemanticLayerStackRefoldError(
                "Whole-stage rollback cannot restore a changed state topology."
            )
        with torch.no_grad():
            for name, value in current.items():
                value.copy_(self.state[name].to(device=value.device, dtype=value.dtype))
            parameters = dict(model.named_parameters())
            if set(parameters) != set(self.gradients):
                raise Lfm2SemanticLayerStackRefoldError(
                    "Whole-stage rollback cannot restore changed Parameters."
                )
            for name, parameter in parameters.items():
                gradient = self.gradients[name]
                if gradient is None:
                    parameter.grad = None
                elif parameter.grad is None:
                    parameter.grad = gradient.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    ).clone()
                else:
                    parameter.grad.copy_(
                        gradient.to(
                            device=parameter.grad.device,
                            dtype=parameter.grad.dtype,
                        )
                    )
        modules = dict(model.named_modules())
        if set(modules) != {name for name, _mode in self.modes}:
            raise Lfm2SemanticLayerStackRefoldError(
                "Whole-stage rollback cannot restore changed module topology."
            )
        for name, mode in self.modes:
            modules[name].training = mode
        if (
            fingerprint_lfm2_parent(model).identity_sha256 != self.fingerprint
            or _parameter_count(model) != self.parameter_count
            or _parameter_topology(model) != self.topology
            or tuple(
                (str(name), bool(module.training))
                for name, module in model.named_modules()
            )
            != self.modes
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Whole-stage rollback did not restore exact entry state."
            )


def _gradient_state_matches_snapshot(
    model: Any,
    snapshot: _WholeModelSnapshot,
) -> bool:
    parameters = dict(model.named_parameters())
    if set(parameters) != set(snapshot.gradients):
        return False
    for name, parameter in parameters.items():
        expected = snapshot.gradients[name]
        observed = parameter.grad
        if (expected is None) != (observed is None):
            return False
        if expected is not None and _tensor_sha256(expected) != _tensor_sha256(
            observed
        ):
            return False
    return True


@dataclass(frozen=True, slots=True)
class _LayerReplayBank:
    captured_child_model_state_sha256: str
    layer_index: int
    pre_residual_by_batch: Mapping[int, Any] = field(repr=False, compare=False)
    replay_by_batch: Mapping[int, Lfm2CpcfMacrocellReplay] = field(
        repr=False,
        compare=False,
    )
    bank_sha256: str


def _layer_runtime(layer: Any) -> tuple[Any, Any]:
    torch = _torch()
    parameters = tuple(layer.parameters())
    if (
        not parameters
        or any(
            value.device.type == "meta"
            or value.dtype != torch.bfloat16
            or not torch.is_floating_point(value)
            for value in parameters
        )
        or len({value.device for value in parameters}) != 1
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Every child macrocell must use one materialized BF16 device."
        )
    return parameters[0].device, parameters[0].dtype


def _initial_residual_bank(
    target_model: Any,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
) -> Mapping[int, Any]:
    torch = _torch()
    embedding = getattr(target_model.get_input_embeddings(), "weight", None)
    if (
        not isinstance(embedding, torch.nn.Parameter)
        or embedding.dtype != torch.bfloat16
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Saved child has no materialized BF16 input table."
        )
    result = {}
    with torch.no_grad():
        for alignment in alignments:
            ids = alignment.phases[-1].canonical_target_input_ids
            if bool((ids < 0).any().item()) or bool(
                (ids >= int(embedding.shape[0])).any().item()
            ):
                raise Lfm2SemanticLayerStackRefoldError(
                    "Target-native replay IDs exceed the saved vocabulary."
                )
            device_ids = ids.to(device=embedding.device, dtype=torch.long)
            result[alignment.source_batch_index] = (
                embedding.index_select(0, device_ids.reshape(-1))
                .reshape(1, int(ids.shape[1]), int(embedding.shape[1]))
                .detach()
                .to(device="cpu")
                .contiguous()
                .clone()
            )
    return MappingProxyType(result)


def _replay_current_layer(
    target_model: Any,
    *,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    layer_index: int,
    pre_residual_by_batch: Mapping[int, Any],
) -> _LayerReplayBank:
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if layer_index < 0 or layer_index >= len(layers):
        raise Lfm2SemanticLayerStackRefoldError(
            f"Child has no replayable layer {layer_index}."
        )
    state = fingerprint_lfm2_parent(target_model).identity_sha256
    layer = layers[layer_index]
    device, dtype = _layer_runtime(layer)
    replays = {}
    receipts = []
    for alignment in alignments:
        batch = alignment.source_batch_index
        residual = pre_residual_by_batch[batch].to(
            device=device,
            dtype=dtype,
            non_blocking=False,
        )
        final = alignment.phases[-1]
        mask = final.canonical_target_attention_mask
        positions = final.canonical_target_position_ids
        if bool(getattr(layer, "is_attention_layer", False)):
            replay = replay_lfm2_cpcf_macrocell(
                layer,
                residual,
                position_embeddings=_rope_from_target(
                    target_model,
                    residual,
                    positions,
                ),
                additive_causal_mask=_additive_causal_mask(mask, residual),
            )
        else:
            replay = replay_lfm2_cpcf_macrocell(
                layer,
                residual,
                padding_mask=mask.to(device=device, dtype=dtype),
            )
        replays[batch] = replay
        receipts.append(
            {
                "batch_index": batch,
                "alignment_sha256": alignment.alignment_sha256,
                "replay_sha256": replay.replay_sha256,
            }
        )
    if fingerprint_lfm2_parent(target_model).identity_sha256 != state:
        raise Lfm2SemanticLayerStackRefoldError(
            "Analytical macrocell replay mutated the child."
        )
    body = {
        "schema": "neural-forge-lfm2-semantic-layer-replay-bank-v1",
        "layer_index": layer_index,
        "captured_child_model_state_sha256": state,
        "batch_replays": receipts,
        "model_forward_calls": 0,
        "layer_forward_calls": 0,
        "training_steps": 0,
    }
    return _LayerReplayBank(
        captured_child_model_state_sha256=state,
        layer_index=layer_index,
        pre_residual_by_batch=MappingProxyType(dict(pre_residual_by_batch)),
        replay_by_batch=MappingProxyType(replays),
        bank_sha256=canonical_sha256(body),
    )


def _next_residual_bank(bank: _LayerReplayBank) -> Mapping[int, Any]:
    return MappingProxyType(
        {
            batch: replay.post_layer_residual
            for batch, replay in bank.replay_by_batch.items()
        }
    )


class _DirectParentSafetensorsSession:
    """One file-hash/open pass serving all parent tensors for the transaction."""

    def __init__(
        self,
        model_directory: str | os.PathLike[str],
        *,
        expected_parent_identity_sha256: str,
    ) -> None:
        self.directory = Path(model_directory).expanduser().resolve()
        self.parent_identity_sha256 = _digest(
            expected_parent_identity_sha256,
            "parent model identity",
        )
        self._stack: ExitStack | None = None
        self._handles: tuple[Any, ...] = ()
        self._keys: Mapping[str, int] = MappingProxyType({})
        self._loaded: dict[str, str] = {}
        self.config_file_sha256 = ""
        self.norm_epsilon = 0.0
        self.manifest_sha256 = ""
        self.open_receipt_sha256 = ""

    def __enter__(self) -> _DirectParentSafetensorsSession:
        if (
            not self.directory.is_dir()
            or self.directory.is_symlink()
            or self._stack is not None
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Parent model directory is missing, unsafe, or already open."
            )
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Direct parent loading requires safetensors.") from exc
        config = self.directory / "config.json"
        if not config.is_file() or config.is_symlink():
            raise Lfm2SemanticLayerStackRefoldError(
                "Parent model directory has no safe config.json."
            )
        try:
            config_value = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Lfm2SemanticLayerStackRefoldError(
                "Parent config.json is invalid."
            ) from exc
        norm_epsilon = (
            config_value.get("norm_eps") if isinstance(config_value, Mapping) else None
        )
        if (
            not isinstance(config_value, Mapping)
            or config_value.get("model_type") != "lfm2"
            or isinstance(norm_epsilon, bool)
            or not isinstance(norm_epsilon, (int, float))
            or not math.isfinite(float(norm_epsilon))
            or float(norm_epsilon) <= 0.0
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Direct parent checkpoint is not LFM2 or has no finite norm_eps."
            )
        self.config_file_sha256 = _file_sha256(config)
        self.norm_epsilon = float(norm_epsilon)
        shards = _local_safetensor_shards(self.directory)
        shard_records = [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "file_sha256": _file_sha256(path),
            }
            for path in shards
        ]
        index = self.directory / "model.safetensors.index.json"
        index_record = None
        if index.exists():
            if not index.is_file() or index.is_symlink():
                raise Lfm2SemanticLayerStackRefoldError(
                    "Parent safetensors index is unsafe."
                )
            index_record = {
                "name": index.name,
                "size_bytes": index.stat().st_size,
                "file_sha256": _file_sha256(index),
            }
        manifest = {
            "schema": "neural-forge-direct-parent-checkpoint-manifest-v1",
            "parent_model_identity_sha256": self.parent_identity_sha256,
            "config_file_sha256": self.config_file_sha256,
            "norm_epsilon": self.norm_epsilon,
            "index": index_record,
            "shards": shard_records,
        }
        self.manifest_sha256 = canonical_sha256(manifest)
        stack = ExitStack()
        handles = []
        keys: dict[str, int] = {}
        try:
            for shard_index, path in enumerate(shards):
                handle = stack.enter_context(
                    safe_open(str(path), framework="pt", device="cpu")
                )
                handles.append(handle)
                for key in handle.keys():
                    if key in keys:
                        raise Lfm2SemanticLayerStackRefoldError(
                            f"Parent tensor {key!r} appears in multiple shards."
                        )
                    keys[key] = shard_index
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        self._handles = tuple(handles)
        self._keys = MappingProxyType(keys)
        self.open_receipt_sha256 = canonical_sha256(
            {
                **manifest,
                "checkpoint_manifest_sha256": self.manifest_sha256,
                "tensor_key_inventory_sha256": canonical_sha256(sorted(keys)),
                "shards_hashed_once": True,
                "shards_opened_once": True,
                "parent_model_objects_constructed": 0,
            }
        )
        return self

    def tensor(self, key: str) -> Any:
        if self._stack is None or key not in self._keys:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Direct parent checkpoint lacks tensor {key!r}."
            )
        value = self._handles[self._keys[key]].get_tensor(key)
        digest = _tensor_sha256(value)
        previous = self._loaded.setdefault(key, digest)
        if previous != digest:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Parent tensor {key!r} changed during direct loading."
            )
        return value.detach().to(device="cpu").contiguous().clone()

    def optional_tensor(self, key: str) -> Any | None:
        return None if key not in self._keys else self.tensor(key)

    def receipt(self) -> Mapping[str, Any]:
        if self._stack is None:
            raise Lfm2SemanticLayerStackRefoldError(
                "Direct parent session is not open."
            )
        body = {
            "schema": "neural-forge-direct-parent-session-v1",
            "parent_model_identity_sha256": self.parent_identity_sha256,
            "checkpoint_manifest_sha256": self.manifest_sha256,
            "open_receipt_sha256": self.open_receipt_sha256,
            "config_file_sha256": self.config_file_sha256,
            "norm_epsilon": self.norm_epsilon,
            "loaded_tensor_sha256": dict(sorted(self._loaded.items())),
            "shards_hashed_once": True,
            "shards_opened_once": True,
            "parent_model_objects_constructed": 0,
            "parent_forward_calls": 0,
            "network_access_performed": False,
        }
        return MappingProxyType({**body, "session_sha256": canonical_sha256(body)})

    def __exit__(self, *_args: Any) -> None:
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._handles = ()
        self._keys = MappingProxyType({})


_COMMON_PARENT_FIELDS = (
    "residual.pre",
    "residual.post_operator",
    "mlp.input",
    "mlp.gate",
    "mlp.up",
    "mlp.post_swiglu",
    "residual.post_layer",
)
_SHORT_CONV_HISTORY_FIELDS = (
    "short_conv.b_history",
    "short_conv.c_history",
    "short_conv.x_history",
)


@dataclass(frozen=True, slots=True)
class _ParentLayerTrajectory:
    batch_index: int
    layer_index: int
    alignment_sha256: str
    phase_rows: Mapping[str, Mapping[str, Any]] = field(
        repr=False,
        compare=False,
    )
    phase_histories: Mapping[str, Mapping[str, Any]] = field(
        repr=False,
        compare=False,
    )
    final_histories: Mapping[str, Any] = field(repr=False, compare=False)
    evidence_sha256: str


def _extend_parent_history(
    previous: Any | None,
    current: Any,
    *,
    expected_full_tokens: int,
    field_name: str,
) -> Any:
    torch = _torch()
    value = current.detach().to(device="cpu").contiguous().clone()
    if value.ndim != 3 or int(value.shape[0]) != 1:
        raise Lfm2SemanticLayerStackRefoldError(
            f"Parent field {field_name!r} is not one rank-3 trajectory."
        )
    if previous is None:
        if int(value.shape[1]) != expected_full_tokens:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Parent prefill field {field_name!r} is not a full prefix."
            )
        return value
    previous_tokens = int(previous.shape[1])
    suffix_tokens = expected_full_tokens - previous_tokens
    if suffix_tokens <= 0:
        raise Lfm2SemanticLayerStackRefoldError(
            f"Parent field {field_name!r} did not advance causally."
        )
    if int(value.shape[1]) == suffix_tokens:
        return torch.cat((previous, value), dim=1).contiguous()
    if int(value.shape[1]) == expected_full_tokens:
        if not torch.equal(value[:, :previous_tokens], previous):
            raise Lfm2SemanticLayerStackRefoldError(
                f"Parent field {field_name!r} changed its causal prefix."
            )
        return value
    raise Lfm2SemanticLayerStackRefoldError(
        f"Parent field {field_name!r} is neither a suffix nor full history."
    )


def _parent_layer_tensor_names(
    reference: Any,
    *,
    alignment: Lfm2SemanticSpanCpcfAlignment,
    layer_index: int,
    operator_kind: str,
    include_final_norm: bool,
) -> tuple[str, ...]:
    fields = list(_COMMON_PARENT_FIELDS)
    if operator_kind == "short_conv":
        fields.extend(_SHORT_CONV_HISTORY_FIELDS)
        fields.extend(
            (
                "short_conv.fir_kernel",
                "short_conv.fir_bias",
                "short_conv.fir_has_bias",
            )
        )
    names_set = {
        f"{phase.phase}.layer.{layer_index:03d}.{field}"
        for phase in alignment.phases
        for field in fields
    }
    if include_final_norm:
        names_set.update(
            f"{phase.phase}.output.final_norm_output" for phase in alignment.phases
        )
    names = tuple(sorted(names_set))
    missing = tuple(sorted(set(names).difference(reference.ordered_tensor_names)))
    if missing:
        raise Lfm2SemanticLayerStackRefoldError(
            f"Complete cache lacks layer {layer_index} fields {list(missing[:6])}."
        )
    return names


def _load_parent_layer_trajectories(
    session: Any,
    *,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    layer_index: int,
    operator_kind: str,
    include_final_norm: bool = False,
) -> Mapping[int, _ParentLayerTrajectory]:
    result = {}
    for alignment, reference in zip(
        alignments,
        session.batch_references,
        strict=True,
    ):
        if reference.index != alignment.source_batch_index:
            raise Lfm2SemanticLayerStackRefoldError(
                "Cache references and semantic alignments are reordered."
            )
        payload = session.load_selected_tensors(
            reference.index,
            _parent_layer_tensor_names(
                reference,
                alignment=alignment,
                layer_index=layer_index,
                operator_kind=operator_kind,
                include_final_norm=include_final_norm,
            ),
        )
        try:
            raw = dict(payload.tensors)
            fields = list(_COMMON_PARENT_FIELDS)
            if operator_kind == "short_conv":
                fields.extend(_SHORT_CONV_HISTORY_FIELDS)
            histories: dict[str, Any] = {}
            phase_rows = {}
            phase_histories = {}
            fir_reference = None
            for phase in alignment.phases:
                row_values = {}
                history_values = {}
                for field in fields:
                    key = f"{phase.phase}.layer.{layer_index:03d}.{field}"
                    value = raw[key]
                    history = _extend_parent_history(
                        histories.get(field),
                        value,
                        expected_full_tokens=phase.donor_full_prefix_tokens,
                        field_name=key,
                    )
                    histories[field] = history
                    history_values[field] = history
                    row_values[field] = value.contiguous().clone()
                if operator_kind == "short_conv":
                    prefix = f"{phase.phase}.layer.{layer_index:03d}.short_conv."
                    fir = (
                        raw[prefix + "fir_kernel"],
                        raw[prefix + "fir_bias"],
                        raw[prefix + "fir_has_bias"],
                    )
                    if fir_reference is None:
                        fir_reference = tuple(value.clone() for value in fir)
                    elif any(
                        not _torch().equal(left, right)
                        for left, right in zip(fir_reference, fir, strict=True)
                    ):
                        raise Lfm2SemanticLayerStackRefoldError(
                            f"Cached parent FIR changed across layer {layer_index} phases."
                        )
                    row_values["short_conv.fir_kernel"] = fir[0]
                    row_values["short_conv.fir_bias"] = fir[1]
                    row_values["short_conv.fir_has_bias"] = fir[2]
                if include_final_norm:
                    output_key = f"{phase.phase}.output.final_norm_output"
                    final_norm = raw[output_key]
                    final_norm_history = _extend_parent_history(
                        histories.get("output.final_norm_output"),
                        final_norm,
                        expected_full_tokens=phase.donor_full_prefix_tokens,
                        field_name=output_key,
                    )
                    histories["output.final_norm_output"] = final_norm_history
                    history_values["output.final_norm_output"] = final_norm_history
                    row_values["output.final_norm_output"] = (
                        final_norm.contiguous().clone()
                    )
                phase_rows[phase.phase] = MappingProxyType(row_values)
                phase_histories[phase.phase] = MappingProxyType(history_values)
            body = {
                "schema": "neural-forge-lfm2-parent-layer-trajectory-v1",
                "batch_index": reference.index,
                "layer_index": layer_index,
                "alignment_sha256": alignment.alignment_sha256,
                "operator_kind": operator_kind,
                "phase_row_sha256": {
                    phase: {
                        name: _tensor_sha256(value)
                        for name, value in sorted(values.items())
                    }
                    for phase, values in phase_rows.items()
                },
                "final_history_sha256": {
                    name: _tensor_sha256(value)
                    for name, value in sorted(histories.items())
                },
            }
            result[reference.index] = _ParentLayerTrajectory(
                batch_index=reference.index,
                layer_index=layer_index,
                alignment_sha256=alignment.alignment_sha256,
                phase_rows=MappingProxyType(phase_rows),
                phase_histories=MappingProxyType(phase_histories),
                final_histories=MappingProxyType(histories),
                evidence_sha256=canonical_sha256(body),
            )
        finally:
            session.release_selected_tensors(payload)
    return MappingProxyType(result)


def _fit_residual_chart(
    *,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent: Mapping[int, _ParentLayerTrajectory],
    child_bank: _LayerReplayBank,
    producer_encoder: Any,
    layer_index: int,
) -> tuple[Any, Mapping[str, Any]]:
    torch = _torch()
    ordered = tuple(sorted(alignments, key=lambda value: value.source_batch_index))
    if len(ordered) < 4:
        raise Lfm2SemanticLayerStackRefoldError(
            "Residual chart requires four isolated trajectories."
        )
    encoder = producer_encoder.detach().to(device="cpu", dtype=torch.float64)
    if encoder.ndim != 2 or not bool(torch.isfinite(encoder).all().item()):
        raise Lfm2SemanticLayerStackRefoldError(
            "The base producer encoder is not one finite residual chart."
        )
    count = len(ordered)
    counts = [max(1, count // 2), max(1, (3 * count) // 16), max(1, count // 8), 1]
    while sum(counts) > count:
        for index in (0, 1, 2):
            if counts[index] > 1 and sum(counts) > count:
                counts[index] -= 1
    counts[3] += count - sum(counts)
    partition_batches: dict[str, tuple[int, ...]] = {}
    cursor = 0
    for partition, size in zip(_PARTITIONS, counts, strict=True):
        partition_batches[partition] = tuple(
            value.source_batch_index for value in ordered[cursor : cursor + size]
        )
        cursor += size
    if cursor != count or any(not value for value in partition_batches.values()):
        raise AssertionError("Internal residual-chart partitioning failed.")
    batch_partition = {
        batch: partition
        for partition, batches in partition_batches.items()
        for batch in batches
    }
    banks: dict[str, dict[str, list[Any]]] = {
        partition: {"left": [], "right": []} for partition in _PARTITIONS
    }
    for alignment in ordered:
        batch = alignment.source_batch_index
        source = parent[batch].final_histories["residual.pre"].to(torch.float64)
        target = child_bank.pre_residual_by_batch[batch].to(torch.float64)
        boundary = alignment.phases[-1].target_to_donor_boundary_positions
        target_positions = tuple(
            index for index, donor in enumerate(boundary) if donor is not None
        )
        donor_positions = tuple(int(value) for value in boundary if value is not None)
        if donor_positions != tuple(range(int(source.shape[1]))):
            raise Lfm2SemanticLayerStackRefoldError(
                "Semantic endpoints do not cover every parent residual row."
            )
        selected = target.index_select(
            1,
            torch.tensor(target_positions, dtype=torch.long),
        )[0]
        left = (source[0] @ encoder) / math.sqrt(int(source.shape[1]))
        right = selected / math.sqrt(int(selected.shape[0]))
        partition = batch_partition[batch]
        banks[partition]["left"].append(left)
        banks[partition]["right"].append(right)
    materialized = {
        partition: (
            torch.cat(tuple(banks[partition]["left"]), dim=0),
            torch.cat(tuple(banks[partition]["right"]), dim=0),
        )
        for partition in _PARTITIONS
    }
    left_fit, right_fit = materialized["fit"]
    identity = torch.eye(int(encoder.shape[1]), dtype=torch.float64)
    design_left, singular, design_right_h = torch.linalg.svd(
        left_fit,
        full_matrices=False,
    )
    largest = float(singular.max().item()) if singular.numel() else 0.0
    threshold = max(
        largest * 1e-8,
        largest
        * torch.finfo(torch.float64).eps
        * max(int(left_fit.shape[0]), int(left_fit.shape[1])),
    )
    supported = singular > threshold
    supported_rank = int(supported.sum().item())
    candidates: dict[str, Any] = {"base_identity": identity}
    if supported_rank:
        supported_right = design_right_h[supported]
        support_projector = supported_right.T @ supported_right
        cross_left, _cross_singular, cross_right_h = torch.linalg.svd(
            left_fit.T @ right_fit,
            full_matrices=False,
        )
        full_procrustes = cross_left @ cross_right_h
        candidates["rank_supported_procrustes"] = (
            identity + support_projector @ (full_procrustes - identity)
        ).contiguous()
        supported_left = design_left[:, supported]
        supported_singular = singular[supported]
        fit_residual = right_fit - left_fit
        projected_residual = supported_left.T @ fit_residual
        scale = max(largest * largest, 1e-30)
        for ratio in (1e-6, 1e-4, 1e-2, 1e-1, 1.0):
            ridge = ratio * scale
            coefficients = supported_singular / (supported_singular.square() + ridge)
            delta = supported_right.T @ (coefficients.unsqueeze(1) * projected_residual)
            candidates[f"rank_supported_ridge_{ratio:g}"] = (
                identity + delta
            ).contiguous()
    candidates = {
        name: value
        for name, value in candidates.items()
        if bool(torch.isfinite(value).all().item())
    }

    def metrics(rotation: Any, partition: str) -> Mapping[str, Any]:
        left, right = materialized[partition]
        prediction = left @ rotation
        difference = prediction - right
        denominator = right.square().mean().sqrt().clamp_min(1e-30)
        prediction_norm = torch.linalg.vector_norm(prediction, dim=1)
        target_norm = torch.linalg.vector_norm(right, dim=1)
        cosine_denominator = prediction_norm * target_norm
        cosine = torch.where(
            cosine_denominator > 0,
            (prediction * right).sum(dim=1) / cosine_denominator.clamp_min(1e-300),
            torch.where(
                torch.linalg.vector_norm(difference, dim=1) == 0,
                torch.ones_like(cosine_denominator),
                torch.zeros_like(cosine_denominator),
            ),
        ).clamp(-1.0, 1.0)
        return {
            "rows": int(prediction.shape[0]),
            "rmse": float(difference.square().mean().sqrt().item()),
            "relative_rmse": float(
                (difference.square().mean().sqrt() / denominator).item()
            ),
            "maximum_absolute_error": float(difference.abs().max().item()),
            "target_maximum_absolute_value": float(right.abs().max().item()),
            "mean_cosine": float(cosine.mean().item()),
            "minimum_cosine": float(cosine.min().item()),
        }

    candidate_metrics = {
        name: {partition: metrics(rotation, partition) for partition in _PARTITIONS}
        for name, rotation in candidates.items()
    }
    baseline = candidate_metrics["base_identity"]

    def nonregressing(
        candidate: Mapping[str, Any], incumbent: Mapping[str, Any]
    ) -> bool:
        rmse_tolerance = max(1e-12, 1e-6 * max(incumbent["rmse"], 1e-12))
        maximum_tolerance = max(
            1e-12,
            1e-6 * max(incumbent["target_maximum_absolute_value"], 1e-12),
        )
        return bool(
            candidate["rmse"] <= incumbent["rmse"] + rmse_tolerance
            and candidate["maximum_absolute_error"]
            <= incumbent["maximum_absolute_error"] + maximum_tolerance
            and candidate["mean_cosine"] + 1e-8 >= incumbent["mean_cosine"]
            and candidate["minimum_cosine"] + 1e-8 >= incumbent["minimum_cosine"]
        )

    fit_check_eligible = {
        name: nonregressing(values["fit_check"], baseline["fit_check"])
        for name, values in candidate_metrics.items()
    }
    selectable = tuple(name for name in candidates if fit_check_eligible[name])
    selected_before_validation = min(
        selectable,
        key=lambda name: (
            candidate_metrics[name]["selection"]["relative_rmse"],
            candidate_metrics[name]["selection"]["maximum_absolute_error"],
            -candidate_metrics[name]["selection"]["mean_cosine"],
            name != "base_identity",
            name,
        ),
    )
    validation_passed = nonregressing(
        candidate_metrics[selected_before_validation]["validation"],
        baseline["validation"],
    )
    selected = selected_before_validation if validation_passed else "base_identity"
    rotation = candidates[selected]
    chart = (encoder @ rotation).contiguous()
    if not bool(torch.isfinite(chart).all().item()):
        raise Lfm2SemanticLayerStackRefoldError("Selected residual chart is nonfinite.")
    unsupported_identity_error = 0.0
    if supported_rank < int(design_right_h.shape[0]):
        unsupported = design_right_h[~supported]
        if unsupported.numel():
            unsupported_identity_error = float(
                (unsupported @ (rotation - identity)).abs().max().item()
            )
    if unsupported_identity_error > 1e-10:
        raise Lfm2SemanticLayerStackRefoldError(
            "Selected residual chart changed an unsupported FIT dimension."
        )

    body = {
        "schema": "neural-forge-lfm2-layer-stack-residual-chart-v1",
        "layer_index": layer_index,
        "captured_child_model_state_sha256": (
            child_bank.captured_child_model_state_sha256
        ),
        "partition_batch_indices": {
            name: list(partition_batches[name]) for name in _PARTITIONS
        },
        "base_producer_encoder_sha256": _tensor_sha256(encoder),
        "rotation_sha256": _tensor_sha256(rotation),
        "chart_sha256": _tensor_sha256(chart),
        "singular_values_sha256": _tensor_sha256(singular),
        "supported_rank": supported_rank,
        "fit_design_width": int(left_fit.shape[1]),
        "support_singular_value_threshold": threshold,
        "unsupported_subspace_identity_maximum_absolute_error": (
            unsupported_identity_error
        ),
        "candidate_rotation_sha256": {
            name: _tensor_sha256(value) for name, value in sorted(candidates.items())
        },
        "candidate_partition_metrics": candidate_metrics,
        "fit_check_nonregression": fit_check_eligible,
        "selected_before_validation": selected_before_validation,
        "selected_candidate": selected,
        "validation_nonregression_passed": validation_passed,
        "validation_rejection_fell_back_to_base": (
            not validation_passed and selected == "base_identity"
        ),
        "candidate_family_constructed_from_fit_only": True,
        "fit_check_used_for_eligibility_only": True,
        "selection_used_for_frozen_choice_only": True,
        "candidate_frozen_before_validation": True,
        "validation_reject_only": True,
        "validation_triggered_refit": False,
        "unsupported_fit_subspace_preserved_as_identity": True,
        "trajectory_balanced_rows": True,
        "fit_only": True,
        "validation_rows_used_for_fit": 0,
        "numeric_parent_child_weight_match_objective": False,
        "training_steps": 0,
    }
    report = {**body, "result_sha256": canonical_sha256(body)}
    return chart, MappingProxyType(report)


def _build_attention_evidence(
    *,
    cache: Lfm2CircuitTraceCache,
    session: Any,
    bridge: StrictTokenizerBridgeArtifact,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent: Mapping[int, _ParentLayerTrajectory],
    child_bank: _LayerReplayBank,
    chart: Any,
    chart_receipt: Mapping[str, Any],
    layer_index: int,
    lineage: Lfm2SemanticCompilerLayerLineage,
    target_rope_inv_freq: Any,
    target_rope_attention_scaling: float,
) -> tuple[Lfm2SemanticAttentionTransportEvidence, ...]:
    torch = _torch()
    result = []
    for alignment, reference in zip(
        alignments,
        session.batch_references,
        strict=True,
    ):
        phases = alignment.phase_order
        payload = session.load_selected_tensors(
            reference.index,
            _attention_tensor_names(
                reference,
                layer_index=layer_index,
                phase_order=phases,
            ),
        )
        try:
            tensors = payload.tensors
            prompt = alignment.phases[0].donor_full_prefix_tokens
            capability = inspect_lfm2_semantic_attention_cache_capability(
                tensors,
                phase_order=phases,
                layer_index=layer_index,
                donor_prompt_token_count=prompt,
            )
            capability.require_exact_history()
            history = assemble_lfm2_explicit_pre_rope_history(
                tensors,
                phase_order=phases,
                layer_index=layer_index,
                donor_prompt_token_count=prompt,
            )
            parent_query = torch.cat(
                tuple(
                    tensors[f"{phase}.layer.{layer_index:03d}.attention.q"]
                    for phase in phases
                ),
                dim=2,
            ).contiguous()
            parent_key = torch.cat(
                tuple(
                    tensors[f"{phase}.layer.{layer_index:03d}.attention.k"]
                    for phase in phases
                ),
                dim=2,
            ).contiguous()
            replay = child_bank.replay_by_batch[reference.index]
            pre_residual = child_bank.pre_residual_by_batch[reference.index]
            for phase_index, phase_alignment in enumerate(alignment.phases):
                donor_tokens = phase_alignment.donor_full_prefix_tokens
                target_tokens = phase_alignment.target_full_prefix_tokens
                phase_order = phases[: phase_index + 1]
                phase_capability = Lfm2SemanticAttentionCacheCapability(
                    layer_index=layer_index,
                    phase_order=phase_order,
                    donor_prompt_token_count=prompt,
                    selected_route="explicit_pre_rope",
                    prefill_captured_token_count=prompt,
                    assembled_current_row_count=donor_tokens,
                    missing_enhanced_tensor_names=(),
                    missing_legacy_tensor_names=(),
                    missing_donor_prefix_positions=(),
                    blocking_reasons=(),
                    exact_history_supported=True,
                )
                phase_history = SemanticAttentionPreRopeHistory(
                    query_normalized_pre_rope=(
                        history.query_normalized_pre_rope[:, :, :donor_tokens]
                    ),
                    key_normalized_pre_rope=(
                        history.key_normalized_pre_rope[:, :, :donor_tokens]
                    ),
                    value=history.value[:, :, :donor_tokens],
                    source_capability_sha256=phase_capability.capability_sha256,
                    source_tensor_names=tuple(
                        name
                        for name in history.source_tensor_names
                        if any(name.startswith(item + ".") for item in phase_order)
                    ),
                )
                transport = transport_lfm2_phase_probability_preserving_attention(
                    phase_alignment,
                    parent_query_post_rope=parent_query[:, :, :donor_tokens],
                    parent_key_post_rope=parent_key[:, :, :donor_tokens],
                    parent_value=history.value[:, :, :donor_tokens],
                    target_rope_inv_freq=target_rope_inv_freq,
                    target_rope_attention_scaling=(target_rope_attention_scaling),
                    query_mode="semantic_endpoints",
                )
                query_positions = torch.tensor(
                    transport.query_target_positions,
                    dtype=torch.long,
                )
                parent_history = parent[reference.index].phase_histories[
                    phase_alignment.phase
                ]["residual.post_operator"]
                donor_positions = torch.tensor(
                    tuple(int(value) for value in transport.query_donor_positions),
                    dtype=torch.long,
                )
                if bool((donor_positions < 0).any().item()) or bool(
                    (donor_positions >= int(parent_history.shape[1])).any().item()
                ):
                    raise Lfm2SemanticLayerStackRefoldError(
                        "Attention query endpoints exceed reconstructed parent history."
                    )
                parent_target = (
                    parent_history.index_select(1, donor_positions).to(torch.float64)
                    @ chart
                ).to(torch.float32)
                child_query = replay.operator_input[:, :target_tokens].index_select(
                    1,
                    query_positions,
                )
                child_key_value = replay.operator_input[:, :target_tokens]
                actual_pre = pre_residual[:, :target_tokens].index_select(
                    1,
                    query_positions,
                )
                weights = torch.full(
                    (1, len(transport.query_target_positions)),
                    1.0 / math.sqrt(len(alignment.phases)),
                    dtype=torch.float64,
                )
                endpoint = Lfm2SemanticAttentionReplayEndpoint(
                    batch_index=reference.index,
                    layer_index=layer_index,
                    alignment_sha256=phase_alignment.alignment_sha256,
                    cache_key=cache.cache_key,
                    bridge_artifact_sha256=bridge.artifact_sha256,
                    captured_child_model_state_sha256=(
                        child_bank.captured_child_model_state_sha256
                    ),
                    target_query_inputs=child_query,
                    target_key_value_inputs=child_key_value,
                    actual_child_pre_attention_residual=actual_pre,
                    parent_post_attention_residual_target_space=parent_target,
                    query_row_sqrt_weights=weights,
                )
                result.append(
                    Lfm2SemanticAttentionTransportEvidence(
                        batch_index=reference.index,
                        layer_index=layer_index,
                        alignment_sha256=phase_alignment.alignment_sha256,
                        alignment=phase_alignment,
                        capability=phase_capability.to_evidence_dict(),
                        history=phase_history.to_evidence_dict(),
                        actual_child_replay=endpoint,
                        source_inventory=lineage.source_inventory,
                        query_groups=lineage.query_groups,
                        query_group_weights=lineage.query_group_weights,
                        compiler_lineage_sha256=canonical_sha256(
                            {
                                "compiler_lineage_sha256": (
                                    lineage.compiler_lineage_sha256
                                ),
                                "residual_chart_sha256": chart_receipt["result_sha256"],
                                "replay_bank_sha256": child_bank.bank_sha256,
                            }
                        ),
                        transport=transport,
                    )
                )
        finally:
            session.release_selected_tensors(payload)
    return tuple(result)


def _is_zero_operation_audit(value: Any) -> bool:
    """Accept adapter-specific zero-operation supersets, never missing/nonzero data."""

    return bool(
        isinstance(value, Mapping)
        and all(
            value.get(name) == expected for name, expected in _ZERO_OPERATIONS.items()
        )
        and all(
            isinstance(name, str)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count == 0
            for name, count in value.items()
        )
    )


def _authenticated_heldout_incumbent_noop(
    receipt: Mapping[str, Any],
    *,
    before_sha256: str,
    after_sha256: str,
    changed_state_keys: Sequence[str],
) -> bool:
    """Recognize a frozen heldout-selected incumbent without requiring mutation."""

    return bool(
        receipt.get("status") == "heldout-selected-incumbent-noop"
        and receipt.get("installed") is False
        and receipt.get("incumbent_selected_by_heldout_gates") is True
        and receipt.get("fit_check_nonregression_passed") is True
        and receipt.get("selection_nonregression_passed") is True
        and receipt.get("validation_nonregression_passed") is True
        and receipt.get("candidate_frozen_before_validation") is True
        and receipt.get("validation_triggered_refit") is False
        and receipt.get("model_state_before_sha256") == before_sha256
        and receipt.get("model_state_after_sha256") == after_sha256
        and before_sha256 == after_sha256
        and receipt.get(
            "changed_state_keys",
            receipt.get("changed_parameter_roles"),
        )
        == []
        and not changed_state_keys
        and receipt.get("parameter_count_changed") is False
        and receipt.get("topology_changed") is False
        and receipt.get("parameter_identities_preserved") is True
        and receipt.get("parameter_storage_preserved") is True
        and receipt.get("gradient_state_preserved") is True
        and receipt.get("exact_noop_state_proven") is True
        and (
            receipt.get("analytic_cached_evidence_only") is True
            or receipt.get("analytic_linear_and_cluster_solves_only") is True
        )
        and _is_zero_operation_audit(receipt.get("operation_audit"))
    )


def _install_attention_operator(
    *,
    target_model: Any,
    layer_index: int,
    evidence: Sequence[Lfm2SemanticAttentionTransportEvidence],
    cache_key: str,
    bridge_artifact_sha256: str,
    installer: Callable[..., Mapping[str, Any]],
    parameter_count: int,
) -> Mapping[str, Any]:
    before_hashes = _state_tensor_hashes(target_model)
    before = fingerprint_lfm2_parent(target_model).identity_sha256
    if any(
        item.actual_child_replay.captured_child_model_state_sha256 != before
        for item in evidence
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Attention evidence is stale before installation."
        )
    candidate = installer(
        target_model=target_model,
        layer_index=layer_index,
        evidence=tuple(evidence),
        cache_key=cache_key,
        bridge_artifact_sha256=bridge_artifact_sha256,
    )
    after_hashes = _state_tensor_hashes(target_model)
    after = fingerprint_lfm2_parent(target_model).identity_sha256
    changed = _changed_state_keys(before_hashes, after_hashes)
    raw_receipt = _verify_sealed(candidate, "semantic attention")
    if _authenticated_heldout_incumbent_noop(
        raw_receipt,
        before_sha256=before,
        after_sha256=after,
        changed_state_keys=changed,
    ):
        analytic = raw_receipt.get("analytic_refold", {})
        if (
            raw_receipt.get("schema") != LFM2_SEMANTIC_ATTENTION_INSTALLATION_SCHEMA
            or raw_receipt.get("layer_index") != layer_index
            or raw_receipt.get("install_count") != 0
            or raw_receipt.get("transport_evidence_sha256")
            != [item.evidence_sha256 for item in evidence]
            or raw_receipt.get("changed_state_keys") != []
            or raw_receipt.get("parameter_element_count_before") != parameter_count
            or raw_receipt.get("parameter_element_count_after") != parameter_count
            or raw_receipt.get("target_topology_sha256_before")
            != raw_receipt.get("target_topology_sha256_after")
            or analytic.get("cache_key") != cache_key
            or analytic.get("bridge_artifact_sha256") != bridge_artifact_sha256
            or analytic.get("analytically_refit_projection_roles") != []
            or analytic.get("fixed_inherited_projection_roles")
            != ["q_proj", "k_proj", "v_proj", "out_proj"]
            or analytic.get("complete_qkvo_bundle_installed_transactionally")
            is not False
            or analytic.get("complete_qkvo_bundle_transaction_verified") is not True
            or analytic.get(
                "authenticated_incumbent_qkvo_retained_without_installation"
            )
            is not True
            or analytic.get("projection_parameter_copy_calls") != 0
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Attention incumbent no-op receipt is stale or incomplete."
            )
        receipt = raw_receipt
    else:
        receipt = _validate_attention_installation_receipt(
            candidate,
            layer_index=layer_index,
            evidence=evidence,
            before_sha256=before,
            after_sha256=after,
            changed_keys=changed,
            parameter_count=parameter_count,
        )
    if receipt.get("schema") != LFM2_SEMANTIC_ATTENTION_INSTALLATION_SCHEMA:
        raise Lfm2SemanticLayerStackRefoldError(
            "Attention operator returned the wrong receipt schema."
        )
    return MappingProxyType(receipt)


def _semantic_terminal_target_position(phase: Any) -> int:
    donor_terminal = phase.donor_full_prefix_tokens - 1
    positions = tuple(
        index
        for index, value in enumerate(phase.target_to_donor_boundary_positions)
        if value == donor_terminal
    )
    if len(positions) != 1 or positions[0] != phase.target_full_prefix_tokens - 1:
        raise Lfm2SemanticLayerStackRefoldError(
            "A semantic phase has no unique terminal parent-exact target boundary."
        )
    return positions[0]


def _validate_cached_fir(
    *,
    parent: Mapping[int, _ParentLayerTrajectory],
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    source_weight: Any,
    source_bias: Any | None,
    layer_index: int,
) -> Mapping[str, Any]:
    torch = _torch()
    raw_weight = source_weight.detach().to(device="cpu").contiguous()
    weight = raw_weight[:, 0, :] if raw_weight.ndim == 3 else raw_weight
    if weight.ndim != 2:
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent FIR is not depthwise rank-2/3."
        )
    comparisons = []
    for alignment in alignments:
        trajectory = parent[alignment.source_batch_index]
        for phase in alignment.phases:
            rows = trajectory.phase_rows[phase.phase]
            cached_weight = rows["short_conv.fir_kernel"]
            cached_bias = rows["short_conv.fir_bias"]
            has_bias = bool(int(rows["short_conv.fir_has_bias"].reshape(-1)[0].item()))
            expected_bias = (
                torch.zeros_like(cached_bias)
                if source_bias is None
                else source_bias.detach().to(device="cpu").contiguous()
            )
            if (
                not torch.equal(cached_weight, weight)
                or has_bias != (source_bias is not None)
                or not torch.equal(cached_bias, expected_bias)
            ):
                raise Lfm2SemanticLayerStackRefoldError(
                    f"Cached and direct-local parent FIR differ at layer {layer_index}."
                )
            comparisons.append(
                canonical_sha256(
                    {
                        "batch_index": alignment.source_batch_index,
                        "phase": phase.phase,
                        "weight_sha256": _tensor_sha256(cached_weight),
                        "bias_sha256": _tensor_sha256(cached_bias),
                        "has_bias": has_bias,
                    }
                )
            )
    body = {
        "schema": "neural-forge-lfm2-direct-cache-fir-equivalence-v1",
        "layer_index": layer_index,
        "direct_weight_sha256": _tensor_sha256(raw_weight),
        "direct_bias_sha256": (
            None if source_bias is None else _tensor_sha256(source_bias)
        ),
        "phase_comparison_sha256": comparisons,
        "all_cached_phases_exact": True,
    }
    return MappingProxyType({**body, "result_sha256": canonical_sha256(body)})


def _resolve_short_conv_adapter() -> tuple[type[Any], Callable[..., Mapping[str, Any]]]:
    from neural_forge.hf_lfm2_semantic_short_conv_refold_adapter import (
        Lfm2SemanticShortConvReplayEvidence,
        compile_install_lfm2_semantic_short_conv_postfold,
    )

    return (
        Lfm2SemanticShortConvReplayEvidence,
        compile_install_lfm2_semantic_short_conv_postfold,
    )


def _build_short_conv_evidence(
    *,
    evidence_type: type[Any],
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent: Mapping[int, _ParentLayerTrajectory],
    child_bank: _LayerReplayBank,
    chart: Any,
    chart_receipt: Mapping[str, Any],
    layer_index: int,
) -> tuple[Any, ...]:
    torch = _torch()
    lineage = canonical_sha256(
        {
            "schema": "neural-forge-lfm2-short-conv-semantic-lineage-v1",
            "cache_key": cache.cache_key,
            "cache_parent_fingerprint": cache.parent_fingerprint,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "layer_index": layer_index,
            "residual_chart_sha256": chart_receipt["result_sha256"],
            "replay_bank_sha256": child_bank.bank_sha256,
        }
    )
    result = []
    for alignment in alignments:
        batch = alignment.source_batch_index
        trajectory = parent[batch]
        replay = child_bank.replay_by_batch[batch]
        pre = child_bank.pre_residual_by_batch[batch]
        for phase_index, phase in enumerate(alignment.phases):
            target_tokens = phase.target_full_prefix_tokens
            terminal = _semantic_terminal_target_position(phase)
            rows = trajectory.phase_rows[phase.phase]
            histories = trajectory.phase_histories[phase.phase]
            parent_post = (
                rows["residual.post_operator"][:, -1:, :].to(torch.float64) @ chart
            ).to(torch.bfloat16)
            history_receipt = canonical_sha256(
                {
                    "batch_index": batch,
                    "phase_index": phase_index,
                    "b_sha256": _tensor_sha256(histories["short_conv.b_history"]),
                    "c_sha256": _tensor_sha256(histories["short_conv.c_history"]),
                    "x_sha256": _tensor_sha256(histories["short_conv.x_history"]),
                    "owner_mapping_sha256": canonical_sha256(
                        list(phase.target_to_donor_span_owner_positions)
                    ),
                    "inserted_rows_parent_exact": False,
                    "semantic_boundary_endpoint_parent_exact": True,
                }
            )
            capability = canonical_sha256(
                {
                    "cache_manifest_sha256": cache.manifest["manifest_sha256"],
                    "parent_trajectory_sha256": trajectory.evidence_sha256,
                    "complete_parent_bcx_history": True,
                    "pure_expansion": (
                        phase.probability_preserving_attention_transport_supported
                    ),
                    "contractions_supported": False,
                }
            )
            result.append(
                evidence_type(
                    batch_index=batch,
                    phase=phase_index,
                    layer_index=layer_index,
                    alignment_sha256=phase.alignment_sha256,
                    cache_key=cache.cache_key,
                    bridge_artifact_sha256=bridge.artifact_sha256,
                    source_capability_sha256=capability,
                    source_history_sha256=history_receipt,
                    compiler_lineage_sha256=lineage,
                    captured_child_model_state_sha256=(
                        child_bank.captured_child_model_state_sha256
                    ),
                    target_to_parent_span_owner_positions=(
                        phase.target_to_donor_span_owner_positions
                    ),
                    endpoint_target_positions=(terminal,),
                    actual_child_operator_input_history=(
                        replay.operator_input[:, :target_tokens]
                    ),
                    actual_child_pre_operator_residual_history=(pre[:, :target_tokens]),
                    parent_b_history=histories["short_conv.b_history"],
                    parent_c_history=histories["short_conv.c_history"],
                    parent_x_history=histories["short_conv.x_history"],
                    parent_post_operator_residual_target_space=parent_post,
                    endpoint_row_sqrt_weights=torch.full(
                        (1, 1),
                        1.0 / math.sqrt(len(alignment.phases)),
                        dtype=torch.float64,
                    ),
                )
            )
    return tuple(result)


def _install_short_conv_operator(
    *,
    target_model: Any,
    layer_index: int,
    evidence: Sequence[Any],
    source_fir_weight: Any,
    source_fir_bias: Any | None,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    installer: Callable[..., Mapping[str, Any]],
    parameter_count: int,
) -> Mapping[str, Any]:
    before_hashes = _state_tensor_hashes(target_model)
    before = fingerprint_lfm2_parent(target_model).identity_sha256
    if any(item.captured_child_model_state_sha256 != before for item in evidence):
        raise Lfm2SemanticLayerStackRefoldError(
            "Short-conv evidence is stale before installation."
        )
    candidate = installer(
        target_model=target_model,
        layer_index=layer_index,
        evidence=tuple(evidence),
        source_fir_weight=source_fir_weight,
        source_fir_bias=source_fir_bias,
        cache_key=cache.cache_key,
        bridge_artifact_sha256=bridge.artifact_sha256,
        source_model_identity_sha256=cache.parent_fingerprint,
    )
    receipt = _verify_sealed(candidate, "short-conv operator")
    after_hashes = _state_tensor_hashes(target_model)
    after = fingerprint_lfm2_parent(target_model).identity_sha256
    changed = _changed_state_keys(before_hashes, after_hashes)
    prefix = f"model.layers.{layer_index}.conv."
    if _authenticated_heldout_incumbent_noop(
        receipt,
        before_sha256=before,
        after_sha256=after,
        changed_state_keys=changed,
    ):
        if (
            receipt.get("schema") != _SHORT_CONV_INSTALLATION_SCHEMA
            or receipt.get("layer_index") != layer_index
            or receipt.get("cache_key") != cache.cache_key
            or receipt.get("bridge_artifact_sha256") != bridge.artifact_sha256
            or receipt.get("source_model_identity_sha256") != cache.parent_fingerprint
            or receipt.get("transport_evidence_sha256")
            != [item.evidence_sha256 for item in evidence]
            or receipt.get("changed_state_keys") != []
            or receipt.get("parameter_element_count") != parameter_count
            or _parameter_count(target_model) != parameter_count
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Short-conv incumbent no-op receipt is stale or incomplete."
            )
        return MappingProxyType(receipt)
    if (
        receipt.get("schema") != _SHORT_CONV_INSTALLATION_SCHEMA
        or receipt.get("installed") is not True
        or receipt.get("layer_index") != layer_index
        or receipt.get("cache_key") != cache.cache_key
        or receipt.get("bridge_artifact_sha256") != bridge.artifact_sha256
        or receipt.get("source_model_identity_sha256") != cache.parent_fingerprint
        or receipt.get("model_state_before_sha256") != before
        or receipt.get("model_state_after_sha256") != after
        or before == after
        or receipt.get("parameter_element_count") != parameter_count
        or receipt.get("parameter_count_changed") is not False
        or receipt.get("changed_state_keys") != list(changed)
        or not changed
        or any(
            not str(name).startswith(prefix)
            for name in receipt.get("changed_state_keys", ())
        )
        or receipt.get("parent_bcx_exact_only_at_semantic_span_boundaries") is not True
        or receipt.get("inserted_target_bcx_rows_are_owner_expanded_proxies")
        is not True
        or receipt.get("inserted_target_rows_parent_exact") is not False
        or receipt.get("inserted_target_rows_used_for_endpoint_fit") is not False
        or receipt.get("analytic_cached_evidence_only") is not True
        or _parameter_count(target_model) != parameter_count
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Short-conv installer returned an incomplete or over-broad receipt."
        )
    return MappingProxyType(receipt)


def _resolve_mlp_adapter() -> tuple[type[Any], type[Any], Callable[..., Any]]:
    from neural_forge.hf_lfm2_semantic_mlp_refold_adapter import (
        Lfm2DirectLocalMlpSourceInventory,
        Lfm2SemanticMlpEndpointEvidence,
        compile_install_lfm2_semantic_mlp_postfold,
    )

    return (
        Lfm2DirectLocalMlpSourceInventory,
        Lfm2SemanticMlpEndpointEvidence,
        compile_install_lfm2_semantic_mlp_postfold,
    )


def _mlp_lineage_sha256(
    *,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    layer_index: int,
    operator_kind: str,
    chart_receipt: Mapping[str, Any],
    attention_lineage: Lfm2SemanticCompilerLayerLineage | None,
) -> str:
    return canonical_sha256(
        {
            "schema": "neural-forge-lfm2-semantic-mlp-layer-lineage-v2",
            "cache_key": cache.cache_key,
            "cache_parent_fingerprint": cache.parent_fingerprint,
            "cache_layer_topology_sha256": cache.layer_topology_fingerprint,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "layer_index": layer_index,
            "operator_kind": operator_kind,
            "attention_compiler_lineage_sha256": (
                None
                if attention_lineage is None
                else attention_lineage.compiler_lineage_sha256
            ),
            "fit_only_residual_assignment_sha256": chart_receipt["result_sha256"],
            "selected_residual_chart_sha256": chart_receipt["chart_sha256"],
        }
    )


def _strict_parent_bf16_matrix(value: Any, name: str, *, columns: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or int(value.shape[0]) <= 0
        or int(value.shape[1]) != columns
        or value.dtype != torch.bfloat16
        or value.requires_grad
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            f"{name} must be one detached finite BF16 matrix with width {columns}."
        )
    return value.detach().to(device="cpu").contiguous().clone()


def _reconstruct_parent_normalized_mlp_input(
    residual: Any,
    *,
    ffn_norm_weight: Any,
    norm_epsilon: float,
    execution_device: Any = "cpu",
) -> Any:
    """Reconstruct parent RMSNorm output from cached residual bytes, no forward."""

    torch = _torch()
    if (
        not isinstance(residual, torch.Tensor)
        or residual.ndim != 2
        or int(residual.shape[0]) <= 0
        or int(residual.shape[1]) <= 0
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Cached parent post-operator residual must be one nonempty matrix."
        )
    source_hidden = int(residual.shape[1])
    replay_residual = _strict_parent_bf16_matrix(
        residual,
        "cached parent post-operator residual",
        columns=source_hidden,
    )
    replay_norm = _strict_parent_bf16_matrix(
        ffn_norm_weight.reshape(1, -1),
        "direct parent FFN RMSNorm weight",
        columns=source_hidden,
    )[0]
    if (
        isinstance(norm_epsilon, bool)
        or not isinstance(norm_epsilon, (int, float))
        or not math.isfinite(float(norm_epsilon))
        or float(norm_epsilon) <= 0.0
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent FFN normalization has no authenticated positive epsilon."
        )
    device = torch.device(execution_device)
    with torch.no_grad():
        residual_bf16 = replay_residual.to(device=device, dtype=torch.bfloat16)
        norm_bf16 = replay_norm.to(device=device, dtype=torch.bfloat16)
        residual_float = residual_bf16.to(torch.float32)
        variance = residual_float.square().mean(dim=-1, keepdim=True)
        normalized = norm_bf16 * (
            residual_float * torch.rsqrt(variance + float(norm_epsilon))
        ).to(torch.bfloat16)
    return normalized.detach().to(device="cpu").contiguous()


def _bf16_cross_kernel_equivalence_policy() -> Mapping[str, Any]:
    """Return the fixed arithmetic envelope used only for provenance replay."""

    torch = _torch()
    machine_epsilon = float(torch.finfo(torch.bfloat16).eps)
    unit_roundoff = machine_epsilon / 2.0
    body = {
        "schema": "neural-forge-bf16-cross-kernel-equivalence-policy-v1",
        "arithmetic_dtype": "bfloat16",
        "machine_epsilon": machine_epsilon,
        "unit_roundoff": unit_roundoff,
        "elementwise_rtol": machine_epsilon,
        "elementwise_atol": unit_roundoff,
        "maximum_relative_rmse": machine_epsilon,
        "relative_rmse_reference": "cached_endpoint_rms",
        "near_zero_ulp_distance_used": False,
        "authentication_only": True,
        "fold_selection_or_quality_objective": False,
    }
    return MappingProxyType(
        {
            **body,
            "policy_sha256": canonical_sha256(body),
        }
    )


def _bf16_cross_kernel_equivalence_metrics(
    *,
    replayed: Any,
    cached: Any,
    policy: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    """Measure backend drift without treating numeric equality as folding."""

    torch = _torch()
    if (
        not isinstance(replayed, torch.Tensor)
        or not isinstance(cached, torch.Tensor)
        or replayed.layout != torch.strided
        or cached.layout != torch.strided
        or replayed.device.type != "cpu"
        or cached.device.type != "cpu"
        or replayed.dtype != torch.bfloat16
        or cached.dtype != torch.bfloat16
        or replayed.shape != cached.shape
        or int(replayed.numel()) <= 0
        or not bool(torch.isfinite(replayed).all().item())
        or not bool(torch.isfinite(cached).all().item())
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            f"{field_name} cross-kernel provenance tensors must be compatible "
            "finite CPU BF16 tensors."
        )
    replayed_float = replayed.to(torch.float32)
    cached_float = cached.to(torch.float32)
    residual = replayed_float - cached_float
    absolute = residual.abs()
    error_rms = float(torch.sqrt(residual.square().mean()).item())
    cached_rms = float(torch.sqrt(cached_float.square().mean()).item())
    relative_rmse = error_rms / max(
        cached_rms,
        float(torch.finfo(torch.float32).tiny),
    )
    elementwise_close = bool(
        torch.allclose(
            replayed_float,
            cached_float,
            rtol=float(policy["elementwise_rtol"]),
            atol=float(policy["elementwise_atol"]),
        )
    )
    relative_rmse_close = bool(
        math.isfinite(relative_rmse)
        and relative_rmse <= float(policy["maximum_relative_rmse"])
    )
    exact_element_count = int(torch.eq(replayed, cached).sum().item())
    element_count = int(replayed.numel())
    cached_maximum_absolute_value = float(cached_float.abs().max().item())
    body = {
        "field": field_name,
        "shape": [int(size) for size in replayed.shape],
        "element_count": element_count,
        "exact_element_count": exact_element_count,
        "mismatched_element_count": element_count - exact_element_count,
        "exact_element_fraction": exact_element_count / element_count,
        "maximum_absolute_error": float(absolute.max().item()),
        "mean_absolute_error": float(absolute.mean().item()),
        "error_rms": error_rms,
        "cached_endpoint_rms": cached_rms,
        "relative_rmse": relative_rmse,
        "cached_maximum_absolute_value": cached_maximum_absolute_value,
        "maximum_elementwise_tolerance_at_cached_maximum": (
            float(policy["elementwise_atol"])
            + float(policy["elementwise_rtol"]) * cached_maximum_absolute_value
        ),
        "elementwise_envelope_passed": elementwise_close,
        "relative_rmse_envelope_passed": relative_rmse_close,
        "bit_exact": exact_element_count == element_count,
        "accepted": bool(elementwise_close and relative_rmse_close),
    }
    return MappingProxyType(body)


def _replay_parent_mlp_functional_provenance(
    *,
    cache: Lfm2CircuitTraceCache,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent: Mapping[int, _ParentLayerTrajectory],
    parent_session: _DirectParentSafetensorsSession,
    layer_index: int,
    gate_weight: Any,
    up_weight: Any,
    down_weight: Any,
    ffn_norm_weight: Any,
    execution_device: Any,
) -> Mapping[str, Any]:
    """Check direct FFN consistency on cached rows across equivalent BF16 kernels."""

    torch = _torch()
    source_intermediate, source_hidden = map(int, gate_weight.shape)
    gate = _strict_parent_bf16_matrix(
        gate_weight,
        "direct parent W1",
        columns=source_hidden,
    )
    up = _strict_parent_bf16_matrix(
        up_weight,
        "direct parent W3",
        columns=source_hidden,
    )
    down = _strict_parent_bf16_matrix(
        down_weight,
        "direct parent W2",
        columns=source_intermediate,
    )
    norm = _strict_parent_bf16_matrix(
        ffn_norm_weight.reshape(1, -1),
        "direct parent FFN RMSNorm weight",
        columns=source_hidden,
    )[0]
    if (
        tuple(up.shape) != tuple(gate.shape)
        or tuple(down.shape) != (source_hidden, source_intermediate)
        or tuple(norm.shape) != (source_hidden,)
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent FFN tensors do not form one compatible LFM2 layer."
        )
    epsilon = parent_session.norm_epsilon
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0.0
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent FFN provenance has no authenticated norm epsilon."
        )

    residual_rows = []
    expected: dict[str, list[Any]] = {
        "mlp.input": [],
        "mlp.gate": [],
        "mlp.up": [],
        "mlp.post_swiglu": [],
        "residual.post_layer": [],
    }
    row_ids = []
    for alignment in sorted(
        alignments,
        key=lambda value: value.source_batch_index,
    ):
        trajectory = parent[alignment.source_batch_index]
        for phase_index, phase in enumerate(alignment.phases):
            rows = trajectory.phase_rows[phase.phase]
            residual = rows["residual.post_operator"][:, -1:, :].reshape(
                1,
                -1,
            )
            residual_rows.append(
                _strict_parent_bf16_matrix(
                    residual,
                    "cached parent post-operator residual endpoint",
                    columns=source_hidden,
                )
            )
            for field_name, columns in (
                ("mlp.input", source_hidden),
                ("mlp.gate", source_intermediate),
                ("mlp.up", source_intermediate),
                ("mlp.post_swiglu", source_intermediate),
                ("residual.post_layer", source_hidden),
            ):
                expected[field_name].append(
                    _strict_parent_bf16_matrix(
                        rows[field_name][:, -1:, :].reshape(1, -1),
                        f"cached parent {field_name} endpoint",
                        columns=columns,
                    )
                )
            row_ids.append(
                {
                    "batch_index": alignment.source_batch_index,
                    "phase_index": phase_index,
                    "phase": phase.phase,
                    "semantic_phase_sha256": phase.semantic_phase_sha256,
                    "parent_trajectory_sha256": trajectory.evidence_sha256,
                }
            )
    if not residual_rows:
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent FFN provenance has no semantic endpoint rows."
        )
    residual = torch.cat(tuple(residual_rows), dim=0)
    cached = {
        field: torch.cat(tuple(values), dim=0) for field, values in expected.items()
    }
    device = torch.device(execution_device)
    normalized_mlp_input = _reconstruct_parent_normalized_mlp_input(
        residual,
        ffn_norm_weight=norm,
        norm_epsilon=float(epsilon),
        execution_device=device,
    )
    with torch.no_grad():
        replay_residual = residual.to(device=device, dtype=torch.bfloat16)
        mlp_input = normalized_mlp_input.to(device=device, dtype=torch.bfloat16)
        replay_gate = torch.nn.functional.linear(
            mlp_input,
            gate.to(device=device, dtype=torch.bfloat16),
        )
        replay_up = torch.nn.functional.linear(
            mlp_input,
            up.to(device=device, dtype=torch.bfloat16),
        )
        replay_post = torch.nn.functional.silu(replay_gate) * replay_up
        replay_down = torch.nn.functional.linear(
            replay_post,
            down.to(device=device, dtype=torch.bfloat16),
        )
        replay_layer = replay_residual + replay_down
    replayed = {
        "mlp.input": normalized_mlp_input,
        "mlp.gate": replay_gate.detach().to(device="cpu").contiguous(),
        "mlp.up": replay_up.detach().to(device="cpu").contiguous(),
        "mlp.post_swiglu": replay_post.detach().to(device="cpu").contiguous(),
        "residual.post_layer": replay_layer.detach().to(device="cpu").contiguous(),
    }
    equivalence_policy = _bf16_cross_kernel_equivalence_policy()
    endpoint_metrics = {
        field: _bf16_cross_kernel_equivalence_metrics(
            replayed=replayed[field],
            cached=cached[field],
            policy=equivalence_policy,
            field_name=field,
        )
        for field in replayed
    }
    rejected = {
        field: {
            "maximum_absolute_error": metrics["maximum_absolute_error"],
            "relative_rmse": metrics["relative_rmse"],
            "elementwise_envelope_passed": metrics["elementwise_envelope_passed"],
            "relative_rmse_envelope_passed": metrics["relative_rmse_envelope_passed"],
        }
        for field, metrics in endpoint_metrics.items()
        if metrics["accepted"] is not True
    }
    if rejected:
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct parent FFN tensors fall outside the authenticated BF16 "
            f"cross-kernel numerical envelope at layer {layer_index}: {rejected}."
        )
    body = {
        "schema": "neural-forge-lfm2-direct-parent-mlp-functional-provenance-v2",
        "cache_key": cache.cache_key,
        "cache_parent_fingerprint": cache.parent_fingerprint,
        "cache_manifest_sha256": cache.manifest["manifest_sha256"],
        "layer_index": layer_index,
        "parent_checkpoint_manifest_sha256": parent_session.manifest_sha256,
        "parent_session_open_receipt_sha256": parent_session.open_receipt_sha256,
        "parent_config_file_sha256": parent_session.config_file_sha256,
        "parent_norm_epsilon": float(epsilon),
        "direct_tensor_sha256": {
            "ffn_norm.weight": _tensor_sha256(norm),
            "feed_forward.w1.weight": _tensor_sha256(gate),
            "feed_forward.w3.weight": _tensor_sha256(up),
            "feed_forward.w2.weight": _tensor_sha256(down),
        },
        "cached_endpoint_sha256": {
            field: _tensor_sha256(value) for field, value in sorted(cached.items())
        },
        "replayed_endpoint_sha256": {
            field: _tensor_sha256(value) for field, value in sorted(replayed.items())
        },
        "reconstructed_normalized_mlp_input_sha256": _tensor_sha256(
            normalized_mlp_input
        ),
        "cached_normalized_mlp_input_sha256": _tensor_sha256(cached["mlp.input"]),
        "semantic_endpoint_rows": len(row_ids),
        "semantic_endpoint_row_ids_sha256": canonical_sha256(row_ids),
        "execution_device": str(device),
        "rms_norm_formula": (
            "weight*(x.float()*rsqrt(mean(x.float()^2)+norm_eps)).to(bfloat16)"
        ),
        "mlp_formula": "W2*(silu(W1*rmsnorm(x))*(W3*rmsnorm(x)))+x",
        "cross_kernel_equivalence_policy": dict(equivalence_policy),
        "endpoint_equivalence_metrics": {
            field: dict(metrics) for field, metrics in sorted(endpoint_metrics.items())
        },
        "all_cached_endpoints_bit_exact": all(
            metrics["bit_exact"] is True for metrics in endpoint_metrics.values()
        ),
        "all_cached_endpoints_within_authenticated_bf16_envelope": True,
        "cached_mlp_input_is_normalized_evidence_source_of_truth": True,
        "reconstructed_mlp_input_is_authentication_check_only": True,
        "direct_parent_tensors_numerically_consistent_on_cached_endpoints": True,
        "semantic_endpoint_scope_is_complete_tensor_identity_proof": False,
        "cryptographic_parent_tensor_identity_proven_by_this_receipt": False,
        "parent_checkpoint_files_hashed_by_session": True,
        "direct_tensor_digests_recorded": True,
        "expected_per_tensor_digest_comparison_performed": False,
        "numerical_equivalence_is_provenance_only": True,
        "numeric_parent_child_weight_match_objective": False,
        "parent_model_forward_calls": 0,
        "training_steps": 0,
    }
    return _sealed(body)


def _direct_mlp_inventory(
    *,
    inventory_type: type[Any],
    parent_session: _DirectParentSafetensorsSession,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    layer_index: int,
    operator_kind: str,
    residual_chart: Any,
    chart_receipt: Mapping[str, Any],
    attention_lineage: Lfm2SemanticCompilerLayerLineage | None,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent: Mapping[int, _ParentLayerTrajectory],
    target_model: Any,
) -> tuple[Any, Mapping[str, Any]]:
    prefix = f"model.layers.{layer_index}.feed_forward."
    gate = parent_session.tensor(prefix + "w1.weight")
    up = parent_session.tensor(prefix + "w3.weight")
    down = parent_session.tensor(prefix + "w2.weight")
    ffn_norm = parent_session.tensor(f"model.layers.{layer_index}.ffn_norm.weight")
    torch = _torch()
    if (
        not isinstance(residual_chart, torch.Tensor)
        or residual_chart.layout != torch.strided
        or residual_chart.device.type == "meta"
        or residual_chart.ndim != 2
        or residual_chart.numel() == 0
        or not torch.is_floating_point(residual_chart)
        or torch.is_complex(residual_chart)
        or residual_chart.requires_grad
        or not bool(torch.isfinite(residual_chart).all().item())
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Selected MLP residual chart must be one detached finite dense matrix."
        )
    selected_chart = residual_chart.detach().to(device="cpu").contiguous().clone()
    verified_chart_receipt = _verify_sealed(chart_receipt, "residual chart")
    if (
        verified_chart_receipt.get("schema")
        != "neural-forge-lfm2-layer-stack-residual-chart-v1"
        or verified_chart_receipt.get("layer_index") != layer_index
        or verified_chart_receipt.get("chart_sha256") != _tensor_sha256(selected_chart)
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Residual-chart receipt does not authenticate the selected MLP chart."
        )
    lineage = _mlp_lineage_sha256(
        cache=cache,
        bridge=bridge,
        layer_index=layer_index,
        operator_kind=operator_kind,
        chart_receipt=verified_chart_receipt,
        attention_lineage=attention_lineage,
    )
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if layer_index >= len(layers):
        raise Lfm2SemanticLayerStackRefoldError(
            f"Child has no layer {layer_index} for parent FFN provenance replay."
        )
    execution_device, _execution_dtype = _layer_runtime(layers[layer_index])
    functional_provenance = _replay_parent_mlp_functional_provenance(
        cache=cache,
        alignments=alignments,
        parent=parent,
        parent_session=parent_session,
        layer_index=layer_index,
        gate_weight=gate,
        up_weight=up,
        down_weight=down,
        ffn_norm_weight=ffn_norm,
        execution_device=execution_device,
    )
    inventory = inventory_type(
        layer_index=layer_index,
        parent_model_identity_sha256=cache.parent_fingerprint,
        parent_checkpoint_manifest_sha256=parent_session.manifest_sha256,
        direct_local_loader_receipt_sha256=functional_provenance["result_sha256"],
        residual_assignment_sha256=verified_chart_receipt["result_sha256"],
        compiler_lineage_sha256=lineage,
        residual_chart=selected_chart,
        ffn_norm_weight=ffn_norm,
        norm_epsilon=parent_session.norm_epsilon,
        gate_weight=gate,
        up_weight=up,
        down_weight=down,
    )
    inventory_body = inventory.authentication_body()
    if (
        not inventory.verify()
        or inventory.residual_assignment_sha256
        != verified_chart_receipt["result_sha256"]
        or inventory.compiler_lineage_sha256 != lineage
        or inventory_body.get("residual_chart_sha256")
        != verified_chart_receipt["chart_sha256"]
        or inventory_body.get("residual_chart_receipt_sha256")
        != verified_chart_receipt["result_sha256"]
        or inventory_body.get("ffn_norm_weight_sha256") != _tensor_sha256(ffn_norm)
        or inventory_body.get("norm_epsilon") != parent_session.norm_epsilon
        or inventory_body.get("normalization_and_loader_lineage_jointly_hashed")
        is not True
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Direct-local MLP inventory lost residual-chart or normalization "
            "lineage binding."
        )
    return inventory, functional_provenance


def _build_mlp_evidence(
    *,
    evidence_type: type[Any],
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    interface: TiedCausalInterfaceFold,
    alignments: Sequence[Lfm2SemanticSpanCpcfAlignment],
    parent: Mapping[int, _ParentLayerTrajectory],
    child_bank: _LayerReplayBank,
    chart: Any,
    source_inventory: Any,
    layer_index: int,
    final_layer: bool,
    execution_device: Any,
) -> tuple[Any, ...]:
    torch = _torch()
    manifest_sha = _digest(
        cache.manifest.get("manifest_sha256"),
        "cache manifest SHA-256",
    )
    base_encoder = interface.producer_encoder.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    equivalence_policy = _bf16_cross_kernel_equivalence_policy()
    result = []
    for alignment in alignments:
        batch = alignment.source_batch_index
        replay = child_bank.replay_by_batch[batch]
        trajectory = parent[batch]
        prior_target_tokens = 0
        prior_donor_tokens = 0
        evidence_ordinal = 0
        phase_count = len(alignment.phases)
        if phase_count <= 0:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Semantic MLP trajectory {batch} has no causal phases."
            )
        for source_phase_index, phase in enumerate(alignment.phases):
            target_tokens = int(phase.target_full_prefix_tokens)
            donor_tokens = int(phase.donor_full_prefix_tokens)
            boundary = tuple(phase.target_to_donor_boundary_positions)
            prior_boundary_changed = bool(
                source_phase_index > 0
                and boundary[:prior_target_tokens]
                != tuple(
                    alignment.phases[
                        source_phase_index - 1
                    ].target_to_donor_boundary_positions
                )
            )
            if (
                target_tokens <= prior_target_tokens
                or donor_tokens <= prior_donor_tokens
                or len(boundary) != target_tokens
                or prior_boundary_changed
            ):
                raise Lfm2SemanticLayerStackRefoldError(
                    f"Semantic MLP phase {batch}:{phase.phase} is not one "
                    "append-only causal prefix."
                )
            target_positions = tuple(
                position
                for position in range(prior_target_tokens, target_tokens)
                if boundary[position] is not None
            )
            donor_positions = tuple(
                int(boundary[position]) for position in target_positions
            )
            expected_donor_positions = tuple(range(prior_donor_tokens, donor_tokens))
            if donor_positions != expected_donor_positions:
                raise Lfm2SemanticLayerStackRefoldError(
                    f"Semantic MLP phase {batch}:{phase.phase} does not expose "
                    "every newly introduced parent-exact causal boundary once."
                )
            # The terminal check independently proves that the newest cached
            # parent state and the newest child prefix end at the same exact
            # byte boundary.  Inserted target rows never enter the evidence.
            if target_positions[-1] != _semantic_terminal_target_position(phase):
                raise Lfm2SemanticLayerStackRefoldError(
                    f"Semantic MLP phase {batch}:{phase.phase} lacks an exact "
                    "terminal row in its newly introduced causal suffix."
                )

            histories = trajectory.phase_histories[phase.phase]
            required_histories = (
                "residual.post_operator",
                "mlp.input",
                "mlp.gate",
                "mlp.up",
                "mlp.post_swiglu",
                "residual.post_layer",
            )
            if any(name not in histories for name in required_histories):
                raise Lfm2SemanticLayerStackRefoldError(
                    f"Semantic MLP phase {batch}:{phase.phase} lacks complete "
                    "cached causal histories."
                )
            donor_index = torch.tensor(donor_positions, dtype=torch.long)
            target_index = torch.tensor(target_positions, dtype=torch.long)
            parent_post_operator = histories["residual.post_operator"].index_select(
                1,
                donor_index,
            )[0]
            parent_normalized_input = _reconstruct_parent_normalized_mlp_input(
                parent_post_operator,
                ffn_norm_weight=source_inventory.ffn_norm_weight,
                norm_epsilon=source_inventory.norm_epsilon,
                execution_device=execution_device,
            )
            cached_parent_normalized_input = (
                histories["mlp.input"]
                .index_select(
                    1,
                    donor_index,
                )[0]
                .detach()
                .to(device="cpu")
                .contiguous()
                .clone()
            )
            normalized_input_metrics = _bf16_cross_kernel_equivalence_metrics(
                replayed=parent_normalized_input,
                cached=cached_parent_normalized_input,
                policy=equivalence_policy,
                field_name="mlp.input",
            )
            if normalized_input_metrics["accepted"] is not True:
                raise Lfm2SemanticLayerStackRefoldError(
                    "Cached parent mlp.input falls outside the authenticated "
                    "BF16 reconstruction envelope at layer "
                    f"{layer_index}, batch {batch}, phase {phase.phase}."
                )
            parent_gate = histories["mlp.gate"].index_select(1, donor_index)[0]
            parent_up = histories["mlp.up"].index_select(1, donor_index)[0]
            parent_post = histories["mlp.post_swiglu"].index_select(1, donor_index)[0]
            parent_post_layer = (
                histories["residual.post_layer"]
                .index_select(1, donor_index)
                .to(torch.float64)
                @ chart
            ).to(torch.bfloat16)[0]
            child_input = replay.mlp_input.index_select(1, target_index)[0]
            child_pre = replay.post_operator_residual.index_select(1, target_index)[0]
            terminal_rows = None
            if final_layer:
                parent_final_norm = histories.get("output.final_norm_output")
                if parent_final_norm is None:
                    raise Lfm2SemanticLayerStackRefoldError(
                        "Final-layer MLP evidence lacks cached parent final-norm "
                        "history."
                    )
                terminal_rows = (
                    parent_final_norm.index_select(1, donor_index).to(torch.float64)
                    @ base_encoder
                ).to(torch.bfloat16)[0]

            row_count = len(target_positions)
            row_weight = 1.0 / (phase_count * row_count)
            for row_offset, (target_position, donor_position) in enumerate(
                zip(target_positions, donor_positions, strict=True)
            ):
                row_phase_sha256 = canonical_sha256(
                    {
                        "schema": (
                            "neural-forge-lfm2-semantic-mlp-causal-boundary-row-v1"
                        ),
                        "source_semantic_phase_sha256": (phase.semantic_phase_sha256),
                        "source_phase_index": source_phase_index,
                        "target_position": target_position,
                        "donor_position": donor_position,
                    }
                )
                row = slice(row_offset, row_offset + 1)
                result.append(
                    evidence_type(
                        batch_index=batch,
                        trajectory_sha256=alignment.alignment_sha256,
                        phase_index=evidence_ordinal,
                        phase_sha256=row_phase_sha256,
                        layer_index=layer_index,
                        alignment_sha256=phase.alignment_sha256,
                        cache_key=cache.cache_key,
                        cache_manifest_sha256=manifest_sha,
                        bridge_artifact_sha256=bridge.artifact_sha256,
                        captured_child_model_state_sha256=(
                            child_bank.captured_child_model_state_sha256
                        ),
                        source_inventory=source_inventory,
                        parent_gate=parent_gate[row],
                        parent_up=parent_up[row],
                        parent_post_swiglu=parent_post[row],
                        parent_pre_mlp_residual=parent_post_operator[row],
                        parent_normalized_mlp_input=(
                            cached_parent_normalized_input[row]
                        ),
                        actual_child_mlp_input=child_input[row],
                        actual_child_pre_mlp_residual=child_pre[row],
                        parent_post_mlp_residual_target_space=(parent_post_layer[row]),
                        parent_final_norm_output_target_space=(
                            None if terminal_rows is None else terminal_rows[row]
                        ),
                        sample_weight=row_weight,
                        parent_exact_semantic_endpoint=True,
                        inserted_target_row=False,
                    )
                )
                evidence_ordinal += 1
            prior_target_tokens = target_tokens
            prior_donor_tokens = donor_tokens
        if evidence_ordinal != prior_donor_tokens:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Semantic MLP trajectory {batch} emitted {evidence_ordinal} exact "
                f"rows for {prior_donor_tokens} cached parent states."
            )
    return tuple(result)


def _install_mlp(
    *,
    target_model: Any,
    layer_index: int,
    evidence: Sequence[Any],
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    installer: Callable[..., Any],
    final_layer: bool,
    parameter_count: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    before_hashes = _state_tensor_hashes(target_model)
    before = fingerprint_lfm2_parent(target_model).identity_sha256
    if any(item.captured_child_model_state_sha256 != before for item in evidence):
        raise Lfm2SemanticLayerStackRefoldError(
            "MLP evidence is stale before installation."
        )
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    w2 = getattr(
        getattr(getattr(layers[layer_index], "feed_forward", None), "w2", None),
        "weight",
        None,
    )
    w2_before = _tensor_sha256(w2)
    result = installer(
        target_model=target_model,
        layer_index=layer_index,
        evidence=tuple(evidence),
        cache_key=cache.cache_key,
        cache_manifest_sha256=cache.manifest["manifest_sha256"],
        bridge_artifact_sha256=bridge.artifact_sha256,
        defer_down_weight=final_layer,
    )
    receipt_value = getattr(result, "receipt", None)
    terminal_evidence = dict(getattr(result, "final_layer_mlp_evidence", {}))
    if not isinstance(receipt_value, Mapping):
        raise Lfm2SemanticLayerStackRefoldError(
            "MLP installer returned no authenticated receipt."
        )
    receipt = _verify_sealed(receipt_value, "semantic MLP")
    after_hashes = _state_tensor_hashes(target_model)
    after = fingerprint_lfm2_parent(target_model).identity_sha256
    changed = _changed_state_keys(before_hashes, after_hashes)
    w2_after = _tensor_sha256(w2)
    common_invalid = (
        receipt.get("schema") != _MLP_INSTALLATION_SCHEMA
        or receipt.get("layer_index") != layer_index
        or receipt.get("model_state_before_sha256") != before
        or receipt.get("model_state_after_sha256") != after
        or receipt.get("parameter_count_before") != parameter_count
        or receipt.get("parameter_count_after") != parameter_count
        or receipt.get("parameter_count_changed") is not False
        or receipt.get("topology_changed") is not False
        or _parameter_count(target_model) != parameter_count
    )
    if common_invalid:
        raise Lfm2SemanticLayerStackRefoldError(
            "Semantic MLP receipt is stale or changed model topology."
        )
    if final_layer:
        evidence_sha = {
            split: value.authentication_sha256
            for split, value in sorted(terminal_evidence.items())
        }
        if (
            receipt.get("final_layer_mlp_w2_deferred_to_fused_terminal") is not True
            or receipt.get("install_count") not in (0, 1)
            or set(terminal_evidence) != set(_PARTITIONS)
            or any(
                value.split != split or not value.verify_authentication()
                for split, value in terminal_evidence.items()
            )
            or receipt.get("terminal_action_evidence_sha256") != evidence_sha
            or w2_after != w2_before
            or (receipt.get("install_count") == 0 and after != before)
            or (
                receipt.get("install_count") == 1
                and (receipt.get("installed") is not True or after == before)
            )
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Final MLP did not defer W2 with four authenticated banks."
            )
    elif _authenticated_heldout_incumbent_noop(
        receipt,
        before_sha256=before,
        after_sha256=after,
        changed_state_keys=changed,
    ):
        if (
            receipt.get("install_count") != 0
            or receipt.get("final_layer_mlp_w2_deferred_to_fused_terminal") is not False
            or receipt.get("cache_key") != cache.cache_key
            or receipt.get("bridge_artifact_sha256") != bridge.artifact_sha256
            or receipt.get("evidence_endpoint_count") != len(evidence)
            or terminal_evidence
            or w2_after != w2_before
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Nonterminal MLP incumbent no-op receipt is stale or incomplete."
            )
    elif (
        receipt.get("installed") is not True
        or receipt.get("install_count") != 1
        or receipt.get("final_layer_mlp_w2_deferred_to_fused_terminal") is not False
        or after == before
        or terminal_evidence
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Nonterminal MLP did not install one accepted complete writer."
        )
    return MappingProxyType(receipt), MappingProxyType(terminal_evidence)


def _validated_operator_topology(
    cache: Lfm2CircuitTraceCache,
    target_model: Any,
    compiler_lineage: Mapping[int, Lfm2SemanticCompilerLayerLineage],
) -> tuple[str, ...]:
    topology = cache.manifest.get("layer_topology", {}).get("layers")
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    if not isinstance(topology, list) or len(topology) != 14 or len(layers) != 14:
        raise Lfm2SemanticLayerStackRefoldError(
            "The executable LFM2 transaction requires exactly fourteen layers."
        )
    kinds = []
    for index, value in enumerate(topology):
        contract = value.get("contract") if isinstance(value, Mapping) else None
        kind = contract.get("kind") if isinstance(contract, Mapping) else None
        if kind not in {"full_attention", "short_conv"}:
            raise Lfm2SemanticLayerStackRefoldError(
                f"Cache layer {index} has no supported LFM2 operator contract."
            )
        is_attention = getattr(layers[index], "is_attention_layer", None)
        if not isinstance(is_attention, bool) or is_attention != (
            kind == "full_attention"
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                f"Child layer {index} differs from the cached operator topology."
            )
        kinds.append(str(kind))
    attention = tuple(
        index for index, kind in enumerate(kinds) if kind == "full_attention"
    )
    short_conv = tuple(
        index for index, kind in enumerate(kinds) if kind == "short_conv"
    )
    if (
        attention != _EXPECTED_ATTENTION_INDICES
        or short_conv != _EXPECTED_SHORT_CONV_INDICES
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Cache does not have the authenticated 6-attention/8-short-conv "
            "LFM2 topology."
        )
    child_types = tuple(
        getattr(getattr(target_model, "config", None), "layer_types", ())
    )
    if child_types:
        normalized = tuple(
            "short_conv" if value in {"conv", "short_conv"} else str(value)
            for value in child_types
        )
        if normalized != tuple(kinds):
            raise Lfm2SemanticLayerStackRefoldError(
                "Child config layer types differ from its cached topology."
            )
    if set(compiler_lineage) != set(_EXPECTED_ATTENTION_INDICES) or any(
        not isinstance(compiler_lineage[index], Lfm2SemanticCompilerLayerLineage)
        or compiler_lineage[index].layer_index != index
        for index in _EXPECTED_ATTENTION_INDICES
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Attention compiler lineage must cover the exact six attention layers."
        )
    return tuple(kinds)


def _validated_interface_lineage(
    target_model: Any,
    interface: TiedCausalInterfaceFold,
) -> Mapping[str, Any]:
    torch = _torch()
    if not isinstance(interface, TiedCausalInterfaceFold):
        raise TypeError("interface must be TiedCausalInterfaceFold.")
    input_weight = getattr(target_model.get_input_embeddings(), "weight", None)
    output_weight = getattr(target_model.get_output_embeddings(), "weight", None)
    report = dict(interface.report)
    report_body = dict(report)
    report_sha = report_body.pop("report_sha256", None)
    report_schema = report.get("schema")
    if report_schema == _TIED_CAUSAL_INTERFACE_FOLD_SCHEMA:
        tensor_sha256 = _interface_tensor_sha256
        tensor_seal_schema = "dtype-nul-shape-nul-raw-v1"
    elif report_schema == TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA:
        tensor_sha256 = _anchor_interface_tensor_sha256
        tensor_seal_schema = "canonical-descriptor-prefix-raw-v1"
    else:
        raise Lfm2SemanticLayerStackRefoldError(
            "Passed tied-interface coordinates use an unsupported report schema."
        )
    producer_sha = tensor_sha256(interface.producer_encoder)
    target_sha = tensor_sha256(interface.target_weight)
    if (
        not isinstance(input_weight, torch.nn.Parameter)
        or output_weight is not input_weight
        or tuple(interface.target_weight.shape) != tuple(input_weight.shape)
        or not torch.equal(
            interface.target_weight.to(device="cpu", dtype=input_weight.dtype),
            input_weight.detach().to(device="cpu"),
        )
        or report_sha != canonical_sha256(report_body)
        or report.get("producer_encoder_sha256") != producer_sha
        or report.get("target_weight_sha256") != target_sha
    ):
        raise Lfm2SemanticLayerStackRefoldError(
            "Passed tied-interface coordinates lack exact authenticated child lineage."
        )
    return MappingProxyType(
        {
            "interface_report_sha256": report_sha,
            "interface_report_schema": report_schema,
            "interface_tensor_seal_schema": tensor_seal_schema,
            "producer_encoder_sha256": producer_sha,
            "target_weight_sha256": target_sha,
            "target_weight_matches_live_tied_child": True,
            "producer_encoder_taken_from_authenticated_compiler_lineage": True,
            "producer_encoder_inferred_from_repaired_target_table": False,
        }
    )


def _anchor_interface_tensor_sha256(value: Any) -> str:
    """Match the tensor seal used by the causal-anchor residual report."""

    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(
        json.dumps(
            descriptor,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _final_w2_parameter(target_model: Any) -> Any:
    torch = _torch()
    layers = tuple(getattr(getattr(target_model, "model", None), "layers", ()))
    parameter = None
    if layers:
        parameter = getattr(
            getattr(getattr(layers[-1], "feed_forward", None), "w2", None),
            "weight",
            None,
        )
    if not isinstance(parameter, torch.nn.Parameter):
        raise Lfm2SemanticLayerStackRefoldError(
            "The final child MLP has no ordinary dense W2 Parameter."
        )
    named = dict(target_model.named_parameters(remove_duplicate=False))
    if named.get("model.layers.13.feed_forward.w2.weight") is not parameter:
        raise Lfm2SemanticLayerStackRefoldError(
            "The final child W2 is renamed, missing, or not independently addressable."
        )
    return parameter


def _validate_direct_attention_sources(
    parent_session: _DirectParentSafetensorsSession,
    compiler_lineage: Mapping[int, Lfm2SemanticCompilerLayerLineage],
) -> Mapping[int, Mapping[str, Any]]:
    receipts = {}
    for layer_index in _EXPECTED_ATTENTION_INDICES:
        prefix = f"model.layers.{layer_index}.self_attn."
        tensors = {
            "q": parent_session.tensor(prefix + "q_proj.weight"),
            "k": parent_session.tensor(prefix + "k_proj.weight"),
            "v": parent_session.tensor(prefix + "v_proj.weight"),
            "o": parent_session.tensor(prefix + "out_proj.weight"),
        }
        observed = seal_lfm2_dense_qkvo_source_inventory(
            source_query_weight=tensors["q"],
            source_key_weight=tensors["k"],
            source_value_weight=tensors["v"],
            source_output_weight=tensors["o"],
        )
        expected = compiler_lineage[layer_index].source_inventory
        if observed.as_dict() != expected.as_dict():
            raise Lfm2SemanticLayerStackRefoldError(
                f"Direct-local parent Q/K/V/O changed at layer {layer_index}."
            )
        receipts[layer_index] = _sealed(
            {
                "schema": "neural-forge-lfm2-direct-parent-qkvo-session-load-v1",
                "layer_index": layer_index,
                "parent_session_open_receipt_sha256": (
                    parent_session.open_receipt_sha256
                ),
                "source_inventory_sha256": observed.inventory_sha256,
                "compiler_lineage_sha256": (
                    compiler_lineage[layer_index].compiler_lineage_sha256
                ),
                "direct_local_safetensors": True,
                "parent_model_objects_constructed": 0,
            }
        )
    return MappingProxyType(receipts)


def _layer_receipt(
    *,
    layer_index: int,
    operator_kind: str,
    before_operator_replay: _LayerReplayBank,
    operator_receipt: Mapping[str, Any],
    operator_coverage_decision: Mapping[str, Any],
    after_operator_replay: _LayerReplayBank,
    mlp_receipt: Mapping[str, Any],
    mlp_coverage_decision: Mapping[str, Any],
    mlp_source_provenance: Mapping[str, Any],
    propagation_replay: _LayerReplayBank,
    chart_receipt: Mapping[str, Any],
    parent: Mapping[int, _ParentLayerTrajectory],
    operator_evidence_sha256: Sequence[str],
    mlp_evidence_sha256: Sequence[str],
    fir_receipt: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    body = {
        "schema": "neural-forge-lfm2-semantic-layer-transaction-v1",
        "layer_index": layer_index,
        "operator_kind": operator_kind,
        "model_state_before_operator_sha256": (
            before_operator_replay.captured_child_model_state_sha256
        ),
        "before_operator_replay_bank_sha256": before_operator_replay.bank_sha256,
        "operator_decision_sha256": operator_receipt["result_sha256"],
        "operator_installation_sha256": operator_receipt["result_sha256"],
        "operator_producer_coverage_decision_sha256": (
            operator_coverage_decision["result_sha256"]
        ),
        "operator_producer_coverage_decision": dict(operator_coverage_decision),
        "operator_decision_count": 1,
        "operator_installed": operator_receipt.get("installed") is True,
        "operator_install_count": int(operator_receipt.get("installed") is True),
        "model_state_after_operator_sha256": (
            after_operator_replay.captured_child_model_state_sha256
        ),
        "operator_parameter_bytes_changed": (
            before_operator_replay.captured_child_model_state_sha256
            != after_operator_replay.captured_child_model_state_sha256
        ),
        "operator_incumbent_noop_selected": (
            operator_receipt.get("status") == "heldout-selected-incumbent-noop"
        ),
        "after_operator_replay_bank_sha256": after_operator_replay.bank_sha256,
        "mlp_decision_sha256": mlp_receipt["result_sha256"],
        "mlp_installation_sha256": mlp_receipt["result_sha256"],
        "mlp_producer_coverage_decision_sha256": (
            mlp_coverage_decision["result_sha256"]
        ),
        "mlp_producer_coverage_decision": dict(mlp_coverage_decision),
        "mlp_decision_count": 1,
        "mlp_installed": mlp_receipt.get("installed") is True,
        "mlp_install_count": int(mlp_receipt.get("installed") is True),
        "mlp_source_functional_provenance_sha256": (
            mlp_source_provenance["result_sha256"]
        ),
        "mlp_source_direct_tensors_functionally_bound_to_cache": (
            mlp_source_provenance["direct_parent_tensors_functionally_bound_to_cache"]
        ),
        "model_state_after_mlp_sha256": (
            propagation_replay.captured_child_model_state_sha256
        ),
        "mlp_parameter_bytes_changed": (
            after_operator_replay.captured_child_model_state_sha256
            != propagation_replay.captured_child_model_state_sha256
        ),
        "mlp_incumbent_noop_selected": (
            mlp_receipt.get("status") == "heldout-selected-incumbent-noop"
        ),
        "propagation_replay_bank_sha256": propagation_replay.bank_sha256,
        "fit_only_residual_chart_sha256": chart_receipt["result_sha256"],
        "parent_trajectory_sha256": [
            parent[index].evidence_sha256 for index in sorted(parent)
        ],
        "operator_evidence_sha256": list(operator_evidence_sha256),
        "mlp_evidence_sha256": list(mlp_evidence_sha256),
        "fir_equivalence_sha256": (
            None if fir_receipt is None else fir_receipt["result_sha256"]
        ),
        "operator_then_replay_fresh_mlp_then_propagate": True,
        "operator_and_mlp_producer_chains_structurally_nonzero": True,
        "numeric_parent_child_weight_match_objective": False,
        "training_steps": 0,
    }
    return _sealed(body)


def compile_install_lfm2_semantic_layer_stack_postfold(
    *,
    cache: Lfm2CircuitTraceCache,
    bridge: StrictTokenizerBridgeArtifact,
    target_model: Any,
    interface: TiedCausalInterfaceFold,
    compiler_lineage: Mapping[int, Lfm2SemanticCompilerLayerLineage],
    parent_model_directory: str | os.PathLike[str],
    donor_tokenizer: Any,
    target_tokenizer: Any,
    attention_installer: Callable[..., Mapping[str, Any]] | None,
    short_conv_installer: Callable[..., Mapping[str, Any]] | None = None,
    mlp_installer: Callable[..., Any] | None = None,
) -> Lfm2SemanticLayerStackInstallation:
    """Compile and install the complete fourteen-layer child transaction.

    This is a no-training compiler.  It consumes an immutable parent trace
    cache and direct local parent tensors, then replays only individual child
    macrocells.  It never constructs or forwards the parent model and never
    invokes the child model's public ``forward`` method.
    """

    if not isinstance(cache, Lfm2CircuitTraceCache):
        raise TypeError("cache must be Lfm2CircuitTraceCache.")
    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    if not isinstance(compiler_lineage, Mapping):
        raise TypeError("compiler_lineage must be a mapping.")
    if not callable(attention_installer):
        raise Lfm2SemanticLayerStackRefoldError(
            "The complete stack requires the analytic attention installer."
        )
    saved_tokenizer = getattr(
        target_model,
        "_neural_forge_quarantine_tokenizer",
        target_tokenizer,
    )
    if saved_tokenizer is not target_tokenizer and fingerprint_hf_tokenizer_semantics(
        saved_tokenizer
    ) != fingerprint_hf_tokenizer_semantics(target_tokenizer):
        raise Lfm2SemanticLayerStackRefoldError(
            "Saved child tokenizer differs from the semantic target tokenizer."
        )
    kinds = _validated_operator_topology(cache, target_model, compiler_lineage)
    interface_lineage = _validated_interface_lineage(target_model, interface)
    short_evidence_type, default_short_installer = _resolve_short_conv_adapter()
    (
        mlp_inventory_type,
        mlp_evidence_type,
        default_mlp_installer,
    ) = _resolve_mlp_adapter()
    selected_short_installer = (
        default_short_installer
        if short_conv_installer is None
        else short_conv_installer
    )
    selected_mlp_installer = (
        default_mlp_installer if mlp_installer is None else mlp_installer
    )
    if not callable(selected_short_installer) or not callable(selected_mlp_installer):
        raise TypeError("Short-conv and MLP installers must be callable.")
    parameter_count = _parameter_count(target_model)
    if parameter_count <= 0:
        raise Lfm2SemanticLayerStackRefoldError("Child has no dense Parameters.")
    final_w2 = _final_w2_parameter(target_model)
    final_w2_before = _tensor_sha256(final_w2)
    snapshot = _WholeModelSnapshot.capture(target_model)
    entry_state_hashes = {
        name: _tensor_sha256(value) for name, value in snapshot.state.items()
    }
    result: Lfm2SemanticLayerStackInstallation | None = None
    try:
        target_rope_inv_freq, target_rope_attention_scaling = _target_rope(target_model)
        with (
            cache.selective_authenticated_session() as cache_session,
            _DirectParentSafetensorsSession(
                parent_model_directory,
                expected_parent_identity_sha256=cache.parent_fingerprint,
            ) as parent_session,
        ):
            alignments, tokenizer_session_receipt = _semantic_alignments_from_session(
                cache=cache,
                session=cache_session,
                bridge=bridge,
                donor_tokenizer=donor_tokenizer,
                target_tokenizer=target_tokenizer,
            )
            refinement = dict(
                audit_lfm2_semantic_span_cpcf_refinement(
                    alignments,
                    expected_batch_count=cache.batch_count,
                )
            )
            if not refinement.get(
                "probability_preserving_attention_transport_go",
                False,
            ):
                raise Lfm2SemanticLayerStackRefoldError(
                    "Semantic span refinement is not exact: "
                    + ", ".join(refinement.get("blocking_reasons", ()))
                )
            direct_attention_receipts = _validate_direct_attention_sources(
                parent_session,
                compiler_lineage,
            )
            residual_bank = _initial_residual_bank(target_model, alignments)
            layer_receipts = []
            attention_decision_count = 0
            attention_install_count = 0
            attention_incumbent_noop_count = 0
            short_conv_decision_count = 0
            short_conv_install_count = 0
            short_conv_incumbent_noop_count = 0
            operator_parameter_mutation_count = 0
            mlp_evidence_compile_count = 0
            nonterminal_mlp_decision_count = 0
            nonterminal_mlp_install_count = 0
            nonterminal_mlp_incumbent_noop_count = 0
            final_mlp_producer_install_count = 0
            final_mlp_producer_incumbent_noop_count = 0
            mlp_parameter_mutation_count = 0
            attention_producer_coverage_count = 0
            short_conv_producer_coverage_count = 0
            nonterminal_mlp_producer_coverage_count = 0
            final_mlp_producer_coverage_count = 0
            cold_start_producer_chain_count = 0
            installed_complete_producer_chain_count = 0
            retained_nonzero_seed_noop_count = 0
            blank_or_partially_blank_producer_chain_count = 0
            final_terminal_evidence: Mapping[str, Any] = MappingProxyType({})
            for layer_index, operator_kind in enumerate(kinds):
                before_operator_replay = _replay_current_layer(
                    target_model,
                    alignments=alignments,
                    layer_index=layer_index,
                    pre_residual_by_batch=residual_bank,
                )
                parent = _load_parent_layer_trajectories(
                    cache_session,
                    alignments=alignments,
                    layer_index=layer_index,
                    operator_kind=operator_kind,
                    include_final_norm=layer_index == 13,
                )
                chart, chart_receipt = _fit_residual_chart(
                    alignments=alignments,
                    parent=parent,
                    child_bank=before_operator_replay,
                    producer_encoder=interface.producer_encoder,
                    layer_index=layer_index,
                )
                fir_receipt = None
                operator_coverage_before = _lfm2_dense_producer_chain_coverage(
                    target_model,
                    layer_index=layer_index,
                    producer_kind=operator_kind,
                )
                if operator_kind == "full_attention":
                    lineage = compiler_lineage[layer_index]
                    operator_evidence = _build_attention_evidence(
                        cache=cache,
                        session=cache_session,
                        bridge=bridge,
                        alignments=alignments,
                        parent=parent,
                        child_bank=before_operator_replay,
                        chart=chart,
                        chart_receipt=chart_receipt,
                        layer_index=layer_index,
                        lineage=lineage,
                        target_rope_inv_freq=target_rope_inv_freq,
                        target_rope_attention_scaling=(target_rope_attention_scaling),
                    )
                    operator_receipt = _install_attention_operator(
                        target_model=target_model,
                        layer_index=layer_index,
                        evidence=operator_evidence,
                        cache_key=cache.cache_key,
                        bridge_artifact_sha256=bridge.artifact_sha256,
                        installer=attention_installer,
                        parameter_count=parameter_count,
                    )
                    attention_decision_count += 1
                    attention_install_count += int(
                        operator_receipt.get("installed") is True
                    )
                    attention_incumbent_noop_count += int(
                        operator_receipt.get("installed") is False
                    )
                    attention_lineage = lineage
                else:
                    attention_lineage = None
                    prefix = f"model.layers.{layer_index}.conv.conv."
                    source_fir_weight = parent_session.tensor(prefix + "weight")
                    source_fir_bias = parent_session.optional_tensor(prefix + "bias")
                    fir_receipt = _validate_cached_fir(
                        parent=parent,
                        alignments=alignments,
                        source_weight=source_fir_weight,
                        source_bias=source_fir_bias,
                        layer_index=layer_index,
                    )
                    operator_evidence = _build_short_conv_evidence(
                        evidence_type=short_evidence_type,
                        cache=cache,
                        bridge=bridge,
                        alignments=alignments,
                        parent=parent,
                        child_bank=before_operator_replay,
                        chart=chart,
                        chart_receipt=chart_receipt,
                        layer_index=layer_index,
                    )
                    operator_receipt = _install_short_conv_operator(
                        target_model=target_model,
                        layer_index=layer_index,
                        evidence=operator_evidence,
                        source_fir_weight=source_fir_weight,
                        source_fir_bias=source_fir_bias,
                        cache=cache,
                        bridge=bridge,
                        installer=selected_short_installer,
                        parameter_count=parameter_count,
                    )
                    short_conv_decision_count += 1
                    short_conv_install_count += int(
                        operator_receipt.get("installed") is True
                    )
                    short_conv_incumbent_noop_count += int(
                        operator_receipt.get("installed") is False
                    )
                operator_coverage_after = _lfm2_dense_producer_chain_coverage(
                    target_model,
                    layer_index=layer_index,
                    producer_kind=operator_kind,
                )
                operator_coverage_decision = _validate_producer_coverage_decision(
                    layer_index=layer_index,
                    producer_kind=operator_kind,
                    before=operator_coverage_before,
                    after=operator_coverage_after,
                    installation=operator_receipt,
                )
                attention_producer_coverage_count += int(
                    operator_kind == "full_attention"
                )
                short_conv_producer_coverage_count += int(operator_kind == "short_conv")
                cold_start_producer_chain_count += int(
                    operator_coverage_decision["cold_start_or_partial_seed_at_entry"]
                )
                installed_complete_producer_chain_count += int(
                    operator_coverage_decision[
                        "installed_complete_structurally_nonzero_chain"
                    ]
                )
                retained_nonzero_seed_noop_count += int(
                    operator_coverage_decision[
                        "authenticated_nonzero_folded_seed_retained_by_noop"
                    ]
                )
                blank_or_partially_blank_producer_chain_count += int(
                    operator_coverage_decision[
                        "blank_or_partially_blank_required_role_count_after_decision"
                    ]
                    > 0
                )
                operator_parameter_mutation_count += int(
                    operator_receipt["model_state_before_sha256"]
                    != operator_receipt["model_state_after_sha256"]
                )
                after_operator_replay = _replay_current_layer(
                    target_model,
                    alignments=alignments,
                    layer_index=layer_index,
                    pre_residual_by_batch=residual_bank,
                )
                if (
                    after_operator_replay.captured_child_model_state_sha256
                    != operator_receipt["model_state_after_sha256"]
                ):
                    raise Lfm2SemanticLayerStackRefoldError(
                        f"Layer {layer_index} operator replay is not fresh."
                    )
                source_inventory, mlp_source_provenance = _direct_mlp_inventory(
                    inventory_type=mlp_inventory_type,
                    parent_session=parent_session,
                    cache=cache,
                    bridge=bridge,
                    layer_index=layer_index,
                    operator_kind=operator_kind,
                    residual_chart=chart,
                    chart_receipt=chart_receipt,
                    attention_lineage=attention_lineage,
                    alignments=alignments,
                    parent=parent,
                    target_model=target_model,
                )
                mlp_evidence = _build_mlp_evidence(
                    evidence_type=mlp_evidence_type,
                    cache=cache,
                    bridge=bridge,
                    interface=interface,
                    alignments=alignments,
                    parent=parent,
                    child_bank=after_operator_replay,
                    chart=chart,
                    source_inventory=source_inventory,
                    layer_index=layer_index,
                    final_layer=layer_index == 13,
                    execution_device=mlp_source_provenance.get(
                        "execution_device",
                        "cpu",
                    ),
                )
                mlp_evidence_compile_count += 1
                mlp_coverage_before = _lfm2_dense_producer_chain_coverage(
                    target_model,
                    layer_index=layer_index,
                    producer_kind="mlp",
                    defer_down_weight=layer_index == 13,
                )
                mlp_receipt, terminal_evidence = _install_mlp(
                    target_model=target_model,
                    layer_index=layer_index,
                    evidence=mlp_evidence,
                    cache=cache,
                    bridge=bridge,
                    installer=selected_mlp_installer,
                    final_layer=layer_index == 13,
                    parameter_count=parameter_count,
                )
                mlp_coverage_after = _lfm2_dense_producer_chain_coverage(
                    target_model,
                    layer_index=layer_index,
                    producer_kind="mlp",
                    defer_down_weight=layer_index == 13,
                )
                mlp_coverage_decision = _validate_producer_coverage_decision(
                    layer_index=layer_index,
                    producer_kind="mlp",
                    before=mlp_coverage_before,
                    after=mlp_coverage_after,
                    installation=mlp_receipt,
                    defer_down_weight=layer_index == 13,
                )
                cold_start_producer_chain_count += int(
                    mlp_coverage_decision["cold_start_or_partial_seed_at_entry"]
                )
                installed_complete_producer_chain_count += int(
                    mlp_coverage_decision[
                        "installed_complete_structurally_nonzero_chain"
                    ]
                )
                retained_nonzero_seed_noop_count += int(
                    mlp_coverage_decision[
                        "authenticated_nonzero_folded_seed_retained_by_noop"
                    ]
                )
                blank_or_partially_blank_producer_chain_count += int(
                    mlp_coverage_decision[
                        "blank_or_partially_blank_required_role_count_after_decision"
                    ]
                    > 0
                )
                if layer_index < 13:
                    nonterminal_mlp_producer_coverage_count += 1
                    nonterminal_mlp_decision_count += 1
                    nonterminal_mlp_install_count += int(
                        mlp_receipt.get("installed") is True
                    )
                    nonterminal_mlp_incumbent_noop_count += int(
                        mlp_receipt.get("installed") is False
                    )
                else:
                    final_mlp_producer_coverage_count = 1
                    final_terminal_evidence = terminal_evidence
                    final_mlp_producer_install_count = int(
                        mlp_receipt.get("installed") is True
                    )
                    final_mlp_producer_incumbent_noop_count = int(
                        mlp_receipt.get("installed") is False
                    )
                mlp_parameter_mutation_count += int(
                    mlp_receipt["model_state_before_sha256"]
                    != mlp_receipt["model_state_after_sha256"]
                )
                propagation_replay = _replay_current_layer(
                    target_model,
                    alignments=alignments,
                    layer_index=layer_index,
                    pre_residual_by_batch=residual_bank,
                )
                if (
                    propagation_replay.captured_child_model_state_sha256
                    != mlp_receipt["model_state_after_sha256"]
                ):
                    raise Lfm2SemanticLayerStackRefoldError(
                        f"Layer {layer_index} propagation replay is not fresh."
                    )
                layer_receipts.append(
                    _layer_receipt(
                        layer_index=layer_index,
                        operator_kind=operator_kind,
                        before_operator_replay=before_operator_replay,
                        operator_receipt=operator_receipt,
                        operator_coverage_decision=operator_coverage_decision,
                        after_operator_replay=after_operator_replay,
                        mlp_receipt=mlp_receipt,
                        mlp_coverage_decision=mlp_coverage_decision,
                        mlp_source_provenance=mlp_source_provenance,
                        propagation_replay=propagation_replay,
                        chart_receipt=chart_receipt,
                        parent=parent,
                        operator_evidence_sha256=tuple(
                            item.evidence_sha256 for item in operator_evidence
                        ),
                        mlp_evidence_sha256=tuple(
                            item.evidence_sha256 for item in mlp_evidence
                        ),
                        fir_receipt=fir_receipt,
                    )
                )
                residual_bank = _next_residual_bank(propagation_replay)
            cache_session.verify()
            cache_session_receipt = cache_session.as_dict()
            parent_session_receipt = dict(parent_session.receipt())
        final_state = fingerprint_lfm2_parent(target_model).identity_sha256
        final_state_hashes = _state_tensor_hashes(target_model)
        changed_keys = _changed_state_keys(entry_state_hashes, final_state_hashes)
        final_w2_after = _tensor_sha256(final_w2)
        terminal_evidence_sha = {
            split: value.authentication_sha256
            for split, value in sorted(final_terminal_evidence.items())
        }
        if (
            attention_decision_count != 6
            or attention_install_count + attention_incumbent_noop_count != 6
            or short_conv_decision_count != 8
            or short_conv_install_count + short_conv_incumbent_noop_count != 8
            or mlp_evidence_compile_count != 14
            or nonterminal_mlp_decision_count != 13
            or (
                nonterminal_mlp_install_count + nonterminal_mlp_incumbent_noop_count
                != 13
            )
            or final_mlp_producer_install_count
            + final_mlp_producer_incumbent_noop_count
            != 1
            or attention_producer_coverage_count != 6
            or short_conv_producer_coverage_count != 8
            or nonterminal_mlp_producer_coverage_count != 13
            or final_mlp_producer_coverage_count != 1
            or blank_or_partially_blank_producer_chain_count != 0
            or installed_complete_producer_chain_count
            + retained_nonzero_seed_noop_count
            != 28
            or retained_nonzero_seed_noop_count
            != (
                attention_incumbent_noop_count
                + short_conv_incumbent_noop_count
                + nonterminal_mlp_incumbent_noop_count
                + final_mlp_producer_incumbent_noop_count
            )
            or cold_start_producer_chain_count > installed_complete_producer_chain_count
            or operator_parameter_mutation_count
            != attention_install_count + short_conv_install_count
            or mlp_parameter_mutation_count
            != nonterminal_mlp_install_count + final_mlp_producer_install_count
            or set(final_terminal_evidence) != set(_PARTITIONS)
            or operator_parameter_mutation_count + mlp_parameter_mutation_count <= 0
            or not changed_keys
            or final_state == snapshot.fingerprint
            or _parameter_count(target_model) != parameter_count
            or _parameter_topology(target_model) != snapshot.topology
            or not _gradient_state_matches_snapshot(target_model, snapshot)
            or set(final_state_hashes) != set(snapshot.state)
            or final_w2_after != final_w2_before
            or tuple(
                (str(name), bool(module.training))
                for name, module in target_model.named_modules()
            )
            != snapshot.modes
        ):
            raise Lfm2SemanticLayerStackRefoldError(
                "Complete layer-stack postconditions were not satisfied."
            )
        body = {
            "schema": LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA,
            "status": "complete-causal-layer-stack-adjudicated-final-w2-deferred",
            "cache_key": cache.cache_key,
            "cache_parent_fingerprint": cache.parent_fingerprint,
            "cache_layer_topology_sha256": cache.layer_topology_fingerprint,
            "cache_selective_session_sha256": cache_session_receipt["session_sha256"],
            "tokenizer_session_receipt_sha256": tokenizer_session_receipt[
                "receipt_sha256"
            ],
            "tokenizer_session": dict(tokenizer_session_receipt),
            "parent_checkpoint_session_sha256": parent_session_receipt[
                "session_sha256"
            ],
            "parent_checkpoint_session": parent_session_receipt,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "tied_interface_lineage": dict(interface_lineage),
            "semantic_refinement_audit_sha256": refinement["report_sha256"],
            "semantic_alignment_sha256": [
                value.alignment_sha256 for value in alignments
            ],
            "direct_attention_source_receipt_sha256": {
                str(index): direct_attention_receipts[index]["result_sha256"]
                for index in _EXPECTED_ATTENTION_INDICES
            },
            "layer_installation_sha256": [
                value["result_sha256"] for value in layer_receipts
            ],
            "layer_installations": [dict(value) for value in layer_receipts],
            "layer_count": 14,
            "attention_operator_decision_count": attention_decision_count,
            "attention_operator_install_count": attention_install_count,
            "attention_operator_incumbent_noop_count": (attention_incumbent_noop_count),
            "short_conv_operator_decision_count": short_conv_decision_count,
            "short_conv_operator_install_count": short_conv_install_count,
            "short_conv_operator_incumbent_noop_count": (
                short_conv_incumbent_noop_count
            ),
            "operator_parameter_mutation_count": (operator_parameter_mutation_count),
            "replay_fresh_mlp_evidence_compile_count": (mlp_evidence_compile_count),
            "nonterminal_mlp_writer_decision_count": nonterminal_mlp_decision_count,
            "nonterminal_mlp_writer_install_count": (nonterminal_mlp_install_count),
            "nonterminal_mlp_writer_incumbent_noop_count": (
                nonterminal_mlp_incumbent_noop_count
            ),
            "final_mlp_producer_install_count": final_mlp_producer_install_count,
            "final_mlp_producer_incumbent_noop_count": (
                final_mlp_producer_incumbent_noop_count
            ),
            "attention_producer_structural_coverage_count": (
                attention_producer_coverage_count
            ),
            "short_conv_producer_structural_coverage_count": (
                short_conv_producer_coverage_count
            ),
            "nonterminal_mlp_producer_structural_coverage_count": (
                nonterminal_mlp_producer_coverage_count
            ),
            "final_mlp_producer_structural_coverage_count": (
                final_mlp_producer_coverage_count
            ),
            "cold_start_or_partial_producer_chain_count": (
                cold_start_producer_chain_count
            ),
            "installed_complete_producer_chain_count": (
                installed_complete_producer_chain_count
            ),
            "authenticated_nonzero_folded_seed_retained_noop_count": (
                retained_nonzero_seed_noop_count
            ),
            "blank_or_partially_blank_producer_chain_count": (
                blank_or_partially_blank_producer_chain_count
            ),
            "mlp_parameter_mutation_count": mlp_parameter_mutation_count,
            "heldout_selected_incumbent_noop_count": sum(
                int(value["operator_incumbent_noop_selected"])
                + int(value["mlp_incumbent_noop_selected"])
                for value in layer_receipts
            ),
            "complete_causal_layer_stack_adjudicated": True,
            "complete_nonzero_structural_producer_coverage": True,
            "all_producer_chains_installed_or_retained_nonzero_seed": True,
            "complete_causal_layer_stack_installed": (
                attention_install_count == attention_decision_count
                and short_conv_install_count == short_conv_decision_count
                and nonterminal_mlp_install_count == nonterminal_mlp_decision_count
                and final_mlp_producer_install_count == 1
            ),
            "all_attention_operator_decisions_completed": True,
            "all_attention_operators_installed": (
                attention_install_count == attention_decision_count
            ),
            "all_short_conv_operator_decisions_completed": True,
            "all_short_conv_operators_installed": (
                short_conv_install_count == short_conv_decision_count
            ),
            "all_replay_fresh_mlp_evidence_compiled": True,
            "all_nonterminal_mlp_writer_decisions_completed": True,
            "all_nonterminal_mlp_writers_installed": (
                nonterminal_mlp_install_count == nonterminal_mlp_decision_count
            ),
            "authenticated_incumbent_noop_layer_decisions_permitted": True,
            "whole_stage_contains_meaningful_parameter_changes": True,
            "final_layer_mlp_w2_deferred_to_fused_terminal": True,
            "final_layer_mlp_w2_install_count": 0,
            "final_layer_mlp_evidence_sha256": terminal_evidence_sha,
            "causal_install_order": ("operator_then_replay_fresh_mlp_then_next_layer"),
            "model_state_before_sha256": snapshot.fingerprint,
            "model_state_after_sha256": final_state,
            "changed_state_keys": list(changed_keys),
            "parameter_element_count": parameter_count,
            "parameter_count_changed": False,
            "parameter_topology_changed": False,
            "deferred_final_w2_sha256": final_w2_before,
            "one_selective_cache_session_for_complete_stack": True,
            "one_parent_checkpoint_hash_and_open_session": True,
            "parent_model_objects_constructed": 0,
            "parent_model_forward_calls": 0,
            "child_model_forward_calls": 0,
            "child_macrocell_replay_only": True,
            "stack_compute_device_router_present": False,
            "attention_solver_device_policy_delegated_to_attention_adapter": True,
            "final_norm_target_uses_unrotated_authenticated_interface": True,
            "short_conv_inserted_owner_rows_are_proxies_only": True,
            "numeric_parent_child_weight_match_objective": False,
            "operation_audit": dict(_ZERO_OPERATIONS),
        }
        receipt = _sealed(body)
        result = Lfm2SemanticLayerStackInstallation(
            receipt=receipt,
            final_layer_mlp_evidence=final_terminal_evidence,
        )
    except BaseException as exc:
        try:
            snapshot.restore(target_model)
        except BaseException as rollback_exc:
            raise Lfm2SemanticLayerStackRefoldError(
                "Layer-stack transaction failed and exact whole-model rollback "
                "could not be proven."
            ) from rollback_exc
        raise Lfm2SemanticLayerStackRefoldError(
            "Layer-stack transaction failed; exact entry bytes were restored."
        ) from exc
    if result is None:  # pragma: no cover - defensive totality guard
        raise AssertionError("Layer-stack transaction produced no typed result.")
    return result


__all__ = [
    "LFM2_SEMANTIC_LAYER_STACK_REFOLD_SCHEMA",
    "Lfm2SemanticLayerStackRefoldError",
    "compile_install_lfm2_semantic_layer_stack_postfold",
]
