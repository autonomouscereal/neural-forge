"""Dynamic target synthesis for standalone Qwen3 codebook children.

The adapter constructs only a seeded blank runtime with the requested global
attention/neuron registry topology. Parent-derived medoid fitting and folding
remain the responsibility of a registered fold operator and cached evidence
stack.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .dynamic_behavior_gate import (
    DynamicBehaviorGateError,
    resolve_behavior_gate_binding,
)
from .dynamic_circuit_ir import (
    ModelCircuitGraph,
    canonical_json_bytes,
    canonical_sha256,
)
from .dynamic_fold_compiler import DynamicFoldSystemStack
from .dynamic_fold_executor import (
    FoldDecision,
    FoldEvaluationContext,
    FoldEvaluator,
    FoldEvidence,
    FoldExecutionContext,
    FoldOperatorRegistry,
    FoldTransaction,
    fingerprint_model_state,
)
from .dynamic_fold_planner import DynamicFoldPlan, FoldAction
from .dynamic_target_synthesis import (
    AdapterMaterialization,
    DynamicTargetRequest,
    DynamicTargetSynthesisResult,
)
from .hf_attention_codebook import (
    DEFAULT_SKETCH_DIM,
    fit_global_attention_codebook,
)
from .hf_neuron_codebook import (
    DEFAULT_WEIGHT_SKETCH_SIZE,
    fit_global_neuron_codebook,
)
from .hf_qwen_codebook_model import (
    _validate_dense_source,
    canonicalize_qwen3_tokenizer_contract,
    load_qwen3_codebook_checkpoint,
    prepare_qwen3_codebook_model,
    qwen3_codebook_parameter_report,
    save_qwen3_codebook_checkpoint,
)
from .hf_tied_depth import exact_token_row_mapping


GLOBAL_CODEBOOK_METHOD = "qwen3_global_codebook_fold"
ADAPTER_SCHEMA = "neural-forge-dynamic-qwen3-codebook-v1"


class DynamicQwenAdapterError(RuntimeError):
    """The requested Qwen codebook blank cannot be proven."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicQwenAdapterError(f"{name} must be a positive integer.")
    return value


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor) or value.device.type == "meta":
        raise DynamicQwenAdapterError(
            "Cached Qwen evidence requires materialized tensors."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "bytes_sha256": hashlib.sha256(
                    tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
        )
    ).hexdigest()


def _tensor_mapping_sha256(
    values: Sequence[tuple[str, Tensor]],
) -> str:
    return canonical_sha256(
        {
            name: _tensor_sha256(value)
            for name, value in sorted(values, key=lambda item: item[0])
        }
    )


def _model_state_sha256(model: nn.Module) -> str:
    values = [
        *model.named_parameters(recurse=True, remove_duplicate=False),
        *model.named_buffers(recurse=True, remove_duplicate=False),
    ]
    if not values:
        raise DynamicQwenAdapterError("Qwen model state is empty.")
    return _tensor_mapping_sha256(tuple(values))


def _cached_tensor(
    value: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    return value.detach().to(device=device).contiguous().clone()


def _module_parameter_values(
    prefix: str,
    module: nn.Module,
    *,
    device: torch.device,
) -> list[tuple[str, Tensor]]:
    return [
        (
            f"{prefix}.{name}" if name else prefix,
            _cached_tensor(parameter, device=device),
        )
        for name, parameter in module.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    ]


@dataclass(frozen=True, slots=True)
class Qwen3GlobalCodebookFoldCache:
    """Immutable parent-derived state for one global p35/p36 transaction."""

    source_parent: str
    parent_state_fingerprint: str
    parameter_values: tuple[tuple[str, Tensor], ...]
    topology_buffers: tuple[tuple[str, Tensor], ...]
    attention_prototypes_per_group: int
    neuron_prototypes_per_neuron: int
    target_vocab_size: int
    expected_unique_parameters: int
    expected_parameter_bytes: int
    parameter_fingerprint: str
    topology_fingerprint: str
    fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_parent, str) or not self.source_parent:
            raise DynamicQwenAdapterError("source_parent must be nonempty.")
        for name in (
            "parent_state_fingerprint",
            "parameter_fingerprint",
            "topology_fingerprint",
            "fingerprint",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise DynamicQwenAdapterError(
                    f"{name} must be a lowercase SHA-256 digest."
                )
        for name in (
            "attention_prototypes_per_group",
            "neuron_prototypes_per_neuron",
            "target_vocab_size",
            "expected_unique_parameters",
            "expected_parameter_bytes",
        ):
            _positive_int(getattr(self, name), name)
        parameter_values = tuple(
            sorted(self.parameter_values, key=lambda item: item[0])
        )
        topology_buffers = tuple(
            sorted(self.topology_buffers, key=lambda item: item[0])
        )
        for label, values in (
            ("parameter_values", parameter_values),
            ("topology_buffers", topology_buffers),
        ):
            names = [name for name, _value in values]
            if not names or len(set(names)) != len(names):
                raise DynamicQwenAdapterError(
                    f"{label} must contain unique named tensors."
                )
            if any(
                not isinstance(name, str)
                or not name
                or not isinstance(value, Tensor)
                or value.device.type == "meta"
                or value.requires_grad
                for name, value in values
            ):
                raise DynamicQwenAdapterError(
                    f"{label} contains an invalid cached tensor."
                )
        object.__setattr__(self, "parameter_values", parameter_values)
        object.__setattr__(self, "topology_buffers", topology_buffers)
        self.verify()

    @property
    def parameter_paths(self) -> tuple[str, ...]:
        return tuple(name for name, _value in self.parameter_values)

    @property
    def topology_buffer_paths(self) -> tuple[str, ...]:
        return tuple(name for name, _value in self.topology_buffers)

    def verify(self) -> None:
        if _tensor_mapping_sha256(self.parameter_values) != (
            self.parameter_fingerprint
        ):
            raise DynamicQwenAdapterError(
                "Cached Qwen parameter evidence failed integrity verification."
            )
        if _tensor_mapping_sha256(self.topology_buffers) != (self.topology_fingerprint):
            raise DynamicQwenAdapterError(
                "Cached Qwen topology evidence failed integrity verification."
            )
        expected = canonical_sha256(
            {
                "schema": ADAPTER_SCHEMA,
                "kind": "global_codebook_fold_cache",
                "source_parent": self.source_parent,
                "parent_state_fingerprint": self.parent_state_fingerprint,
                "attention_prototypes_per_group": (self.attention_prototypes_per_group),
                "neuron_prototypes_per_neuron": (self.neuron_prototypes_per_neuron),
                "target_vocab_size": self.target_vocab_size,
                "expected_unique_parameters": self.expected_unique_parameters,
                "expected_parameter_bytes": self.expected_parameter_bytes,
                "parameter_fingerprint": self.parameter_fingerprint,
                "topology_fingerprint": self.topology_fingerprint,
            }
        )
        if self.fingerprint != expected:
            raise DynamicQwenAdapterError(
                "Cached Qwen evidence fingerprint is inconsistent."
            )

    def validate_parent(self, parent_name: str, parent_model: Any) -> None:
        self.verify()
        if parent_name != self.source_parent:
            raise DynamicQwenAdapterError(
                "The selected Qwen parent differs from the cached source."
            )
        if fingerprint_model_state(parent_model) != self.parent_state_fingerprint:
            raise DynamicQwenAdapterError(
                "The loaded Qwen parent differs from the cached checkpoint."
            )

    def apply_topology(self, model: nn.Module) -> None:
        self.verify()
        expected_paths = set(self.topology_buffer_paths)
        actual_paths = {
            name
            for name, _value in model.named_buffers(
                recurse=True,
                remove_duplicate=False,
            )
            if name == "parent_token_rows"
            or name.startswith("attention_codebook.registry.")
            or name.startswith("neuron_codebook.registry.")
        }
        if actual_paths != expected_paths:
            raise DynamicQwenAdapterError(
                "Blank Qwen codebook topology-buffer paths differ from cache."
            )
        with torch.no_grad():
            for path, cached in self.topology_buffers:
                target = model.get_buffer(path)
                if (
                    tuple(target.shape) != tuple(cached.shape)
                    or target.dtype != cached.dtype
                ):
                    raise DynamicQwenAdapterError(
                        f"Topology buffer {path!r} differs from cached shape/dtype."
                    )
                target.copy_(cached.to(device=target.device))
        self.assert_topology(model)

    def assert_topology(self, model: nn.Module) -> None:
        for path, cached in self.topology_buffers:
            target = model.get_buffer(path)
            if not torch.equal(
                target.detach().to(device="cpu"),
                cached.detach().to(device="cpu"),
            ):
                raise DynamicQwenAdapterError(
                    f"Qwen topology buffer {path!r} differs from cached evidence."
                )

    def assert_parameter_contract(self, model: nn.Module) -> None:
        actual_paths = {
            name
            for name, _parameter in model.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
        }
        if actual_paths != set(self.parameter_paths):
            raise DynamicQwenAdapterError(
                "Qwen target parameter paths differ from cached evidence."
            )

    def matches_parameters(self, model: nn.Module) -> bool:
        self.assert_parameter_contract(model)
        return all(
            torch.equal(
                model.get_parameter(path).detach().to(device="cpu"),
                cached.detach().to(device="cpu"),
            )
            for path, cached in self.parameter_values
        )


def build_qwen3_global_codebook_fold_cache(
    *,
    source_parent: str,
    parent_model: nn.Module,
    parent_tokenizer: Any,
    target_tokenizer: Any,
    attention_prototypes_per_group: int,
    neuron_prototypes_per_neuron: int,
    group_permutations: Tensor | Sequence[Sequence[int]] | None = None,
    neuron_permutations: Tensor | Sequence[Tensor | Sequence[int]] | None = None,
    attention_use_scales: bool = False,
    neuron_use_scales: bool = False,
    attention_sketch_dim: int = DEFAULT_SKETCH_DIM,
    neuron_weight_sketch_size: int = DEFAULT_WEIGHT_SKETCH_SIZE,
    cache_device: str | torch.device = "cpu",
) -> Qwen3GlobalCodebookFoldCache:
    """Fit existing global codebook math once and cache the complete child state."""

    if not isinstance(source_parent, str) or not source_parent:
        raise DynamicQwenAdapterError("source_parent must be nonempty.")
    attention_depth = _positive_int(
        attention_prototypes_per_group,
        "attention_prototypes_per_group",
    )
    neuron_depth = _positive_int(
        neuron_prototypes_per_neuron,
        "neuron_prototypes_per_neuron",
    )
    attention_sketch = _positive_int(
        attention_sketch_dim,
        "attention_sketch_dim",
    )
    neuron_sketch = _positive_int(
        neuron_weight_sketch_size,
        "neuron_weight_sketch_size",
    )
    if any(parameter.grad is not None for parameter in parent_model.parameters()):
        raise DynamicQwenAdapterError(
            "Qwen fold-cache construction requires zero-gradient parent weights."
        )
    cache_target = torch.device(cache_device)
    source = _validate_dense_source(parent_model)
    try:
        parent_rows = exact_token_row_mapping(
            parent_tokenizer,
            target_tokenizer,
        )
    except ValueError as exc:
        raise DynamicQwenAdapterError("Exact target-token row mapping failed.") from exc
    row_indices = torch.tensor(
        parent_rows,
        device=source.embedding.weight.device,
        dtype=torch.long,
    )
    with torch.no_grad():
        attention = fit_global_attention_codebook(
            source.attention_bundles,
            prototypes_per_group=attention_depth,
            query_heads=int(parent_model.config.num_attention_heads),
            key_value_heads=int(parent_model.config.num_key_value_heads),
            head_dim=int(parent_model.config.head_dim),
            group_permutations=group_permutations,
            use_scales=bool(attention_use_scales),
            sketch_dim=attention_sketch,
        )
        neurons = fit_global_neuron_codebook(
            source.neuron_bundles,
            prototypes_per_neuron=neuron_depth,
            activation=str(parent_model.config.hidden_act),
            use_scales=bool(neuron_use_scales),
            weight_sketch_size=neuron_sketch,
            assignment_strategy="aligned_index",
            neuron_permutations=neuron_permutations,
        )
        selected_embedding = source.embedding.weight.detach().index_select(
            0,
            row_indices,
        )

    parameters: list[tuple[str, Tensor]] = [
        (
            "model.embed_tokens.weight",
            _cached_tensor(selected_embedding, device=cache_target),
        ),
        (
            "lm_head.weight",
            _cached_tensor(selected_embedding, device=cache_target),
        ),
    ]
    for layer_index, layer in enumerate(source.layers):
        parameters.extend(
            _module_parameter_values(
                f"model.layers.{layer_index}.input_layernorm",
                layer.input_layernorm,
                device=cache_target,
            )
        )
        parameters.extend(
            _module_parameter_values(
                f"model.layers.{layer_index}.post_attention_layernorm",
                layer.post_attention_layernorm,
                device=cache_target,
            )
        )
        parameters.extend(
            _module_parameter_values(
                f"model.layers.{layer_index}.self_attn.q_norm",
                layer.self_attn.q_norm,
                device=cache_target,
            )
        )
        parameters.extend(
            _module_parameter_values(
                f"model.layers.{layer_index}.self_attn.k_norm",
                layer.self_attn.k_norm,
                device=cache_target,
            )
        )
    parameters.extend(
        _module_parameter_values(
            "model.norm",
            parent_model.model.norm,
            device=cache_target,
        )
    )
    parameters.extend(
        (
            f"attention_codebook.registry.{name}",
            _cached_tensor(parameter, device=cache_target),
        )
        for name, parameter in attention.registry.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    )
    parameters.extend(
        (
            f"neuron_codebook.registry.{name}",
            _cached_tensor(parameter, device=cache_target),
        )
        for name, parameter in neurons.registry.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    )
    buffers: list[tuple[str, Tensor]] = [
        (
            "parent_token_rows",
            torch.tensor(parent_rows, dtype=torch.int32, device=cache_target),
        )
    ]
    buffers.extend(
        (
            f"attention_codebook.registry.{name}",
            _cached_tensor(value, device=cache_target),
        )
        for name, value in attention.registry.named_buffers(
            recurse=True,
            remove_duplicate=False,
        )
    )
    buffers.extend(
        (
            f"neuron_codebook.registry.{name}",
            _cached_tensor(value, device=cache_target),
        )
        for name, value in neurons.registry.named_buffers(
            recurse=True,
            remove_duplicate=False,
        )
    )
    parameters.sort(key=lambda item: item[0])
    buffers.sort(key=lambda item: item[0])
    names = [name for name, _value in parameters]
    if len(names) != len(set(names)):
        raise DynamicQwenAdapterError(
            "Qwen fold cache produced duplicate target parameter paths."
        )
    unique_parameters = sum(
        int(value.numel()) for name, value in parameters if name != "lm_head.weight"
    )
    parameter_bytes = sum(
        int(value.numel()) * int(value.element_size())
        for name, value in parameters
        if name != "lm_head.weight"
    )
    parameter_fingerprint = _tensor_mapping_sha256(parameters)
    topology_fingerprint = _tensor_mapping_sha256(buffers)
    parent_state = fingerprint_model_state(parent_model)
    cache_body = {
        "schema": ADAPTER_SCHEMA,
        "kind": "global_codebook_fold_cache",
        "source_parent": source_parent,
        "parent_state_fingerprint": parent_state,
        "attention_prototypes_per_group": attention_depth,
        "neuron_prototypes_per_neuron": neuron_depth,
        "target_vocab_size": len(parent_rows),
        "expected_unique_parameters": unique_parameters,
        "expected_parameter_bytes": parameter_bytes,
        "parameter_fingerprint": parameter_fingerprint,
        "topology_fingerprint": topology_fingerprint,
    }
    if any(parameter.grad is not None for parameter in parent_model.parameters()):
        raise DynamicQwenAdapterError(
            "Qwen codebook fitting unexpectedly populated parent gradients."
        )
    return Qwen3GlobalCodebookFoldCache(
        source_parent=source_parent,
        parent_state_fingerprint=parent_state,
        parameter_values=tuple(parameters),
        topology_buffers=tuple(buffers),
        attention_prototypes_per_group=attention_depth,
        neuron_prototypes_per_neuron=neuron_depth,
        target_vocab_size=len(parent_rows),
        expected_unique_parameters=unique_parameters,
        expected_parameter_bytes=parameter_bytes,
        parameter_fingerprint=parameter_fingerprint,
        topology_fingerprint=topology_fingerprint,
        fingerprint=canonical_sha256(cache_body),
    )


def build_qwen3_global_codebook_fold_cache_from_checkpoint(
    *,
    checkpoint: str | Path,
    source_parent: str,
    live_parent_model: nn.Module,
    attention_prototypes_per_group: int,
    neuron_prototypes_per_neuron: int,
    tokenizer_contract: Mapping[str, Any] | None = None,
    expected_manifest_sha256: str | None = None,
    cache_device: str | torch.device = "cpu",
) -> Qwen3GlobalCodebookFoldCache:
    """Reuse one authenticated standalone p35/p36 checkpoint as fold evidence."""

    attention_depth = _positive_int(
        attention_prototypes_per_group,
        "attention_prototypes_per_group",
    )
    neuron_depth = _positive_int(
        neuron_prototypes_per_neuron,
        "neuron_prototypes_per_neuron",
    )
    path = Path(checkpoint)
    manifest_path = path / "manifest.json"
    if expected_manifest_sha256 is not None:
        if (
            not isinstance(expected_manifest_sha256, str)
            or len(expected_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_manifest_sha256
            )
        ):
            raise DynamicQwenAdapterError(
                "expected_manifest_sha256 must be a lowercase SHA-256 digest."
            )
        if (
            not manifest_path.is_file()
            or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            != expected_manifest_sha256
        ):
            raise DynamicQwenAdapterError(
                "Qwen codebook checkpoint manifest hash differs from expectation."
            )
    runtime = load_qwen3_codebook_checkpoint(path, device=cache_device)
    report = qwen3_codebook_parameter_report(runtime)
    source = _validate_dense_source(live_parent_model)
    if (
        report.attention_prototypes_per_group != attention_depth
        or report.neuron_prototypes_per_neuron != neuron_depth
        or not report.full_logical_depth
        or not report.exact_parent_token_rows
        or not report.tied_input_output_embeddings
        or not report.parameterless_layer_views
        or not report.single_registry_owner_per_codebook
        or report.parent_runtime_reference_count != 0
    ):
        raise DynamicQwenAdapterError(
            "Standalone Qwen checkpoint is not the requested p35/p36 runtime."
        )
    if (
        report.logical_layers != len(source.layers)
        or report.hidden_size != int(live_parent_model.config.hidden_size)
        or report.intermediate_size != int(live_parent_model.config.intermediate_size)
        or report.query_heads != int(live_parent_model.config.num_attention_heads)
        or report.key_value_heads != int(live_parent_model.config.num_key_value_heads)
        or report.head_dim != int(live_parent_model.config.head_dim)
        or report.source_vocab_size != source.embedding.num_embeddings
        or report.source_unique_parameter_count != source.source_unique_parameters
    ):
        raise DynamicQwenAdapterError(
            "Standalone Qwen checkpoint source topology differs from the live parent."
        )
    stored_contract = getattr(
        runtime,
        "neural_forge_tokenizer_contract",
        None,
    )
    if not isinstance(stored_contract, Mapping):
        raise DynamicQwenAdapterError(
            "Standalone Qwen checkpoint lacks a resolved tokenizer contract."
        )
    if tokenizer_contract is not None:
        expected_contract = canonicalize_qwen3_tokenizer_contract(
            runtime,
            tokenizer_contract,
        )
        if dict(stored_contract) != expected_contract:
            raise DynamicQwenAdapterError(
                "Standalone Qwen checkpoint tokenizer contract differs."
            )
    cache_target = torch.device(cache_device)
    parameters = tuple(
        sorted(
            (
                name,
                _cached_tensor(parameter, device=cache_target),
            )
            for name, parameter in runtime.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
        )
    )
    buffers = tuple(
        sorted(
            (
                name,
                _cached_tensor(value, device=cache_target),
            )
            for name, value in runtime.named_buffers(
                recurse=True,
                remove_duplicate=False,
            )
            if name == "parent_token_rows"
            or name.startswith("attention_codebook.registry.")
            or name.startswith("neuron_codebook.registry.")
        )
    )
    parameter_paths = {name for name, _value in parameters}
    if not parameter_paths.issubset(runtime.state_dict()):
        raise DynamicQwenAdapterError(
            "Standalone Qwen checkpoint parameter extraction is inconsistent."
        )
    parameter_fingerprint = _tensor_mapping_sha256(parameters)
    topology_fingerprint = _tensor_mapping_sha256(buffers)
    parent_state = fingerprint_model_state(live_parent_model)
    parameter_bytes = sum(
        int(parameter.numel()) * int(parameter.element_size())
        for parameter in runtime.parameters()
    )
    cache_body = {
        "schema": ADAPTER_SCHEMA,
        "kind": "global_codebook_fold_cache",
        "source_parent": source_parent,
        "parent_state_fingerprint": parent_state,
        "attention_prototypes_per_group": attention_depth,
        "neuron_prototypes_per_neuron": neuron_depth,
        "target_vocab_size": report.target_vocab_size,
        "expected_unique_parameters": report.unique_parameter_count,
        "expected_parameter_bytes": parameter_bytes,
        "parameter_fingerprint": parameter_fingerprint,
        "topology_fingerprint": topology_fingerprint,
    }
    return Qwen3GlobalCodebookFoldCache(
        source_parent=source_parent,
        parent_state_fingerprint=parent_state,
        parameter_values=parameters,
        topology_buffers=buffers,
        attention_prototypes_per_group=attention_depth,
        neuron_prototypes_per_neuron=neuron_depth,
        target_vocab_size=report.target_vocab_size,
        expected_unique_parameters=report.unique_parameter_count,
        expected_parameter_bytes=parameter_bytes,
        parameter_fingerprint=parameter_fingerprint,
        topology_fingerprint=topology_fingerprint,
        fingerprint=canonical_sha256(cache_body),
    )


class Qwen3CodebookTargetSynthesisAdapter:
    """Materialize an exact seeded codebook-shaped Qwen3 blank locally."""

    adapter_id = "neural_forge.qwen3.codebook"
    version = 1
    priority = 200
    target_families = ("qwen3_codebook",)
    target_modalities = ("text",)

    def __init__(
        self,
        *,
        parent_tokenizer: Any,
        target_tokenizer: Any,
        attention_prototypes_per_group: int,
        neuron_prototypes_per_neuron: int,
        attention_use_scales: bool = False,
        neuron_use_scales: bool = False,
        attention_sketch_dim: int = DEFAULT_SKETCH_DIM,
        neuron_weight_sketch_size: int = DEFAULT_WEIGHT_SKETCH_SIZE,
        fold_cache: Qwen3GlobalCodebookFoldCache | None = None,
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> None:
        self.parent_tokenizer = parent_tokenizer
        self.target_tokenizer = target_tokenizer
        self.attention_prototypes_per_group = _positive_int(
            attention_prototypes_per_group,
            "attention_prototypes_per_group",
        )
        self.neuron_prototypes_per_neuron = _positive_int(
            neuron_prototypes_per_neuron,
            "neuron_prototypes_per_neuron",
        )
        self.attention_use_scales = bool(attention_use_scales)
        self.neuron_use_scales = bool(neuron_use_scales)
        self.attention_sketch_dim = _positive_int(
            attention_sketch_dim,
            "attention_sketch_dim",
        )
        self.neuron_weight_sketch_size = _positive_int(
            neuron_weight_sketch_size,
            "neuron_weight_sketch_size",
        )
        if fold_cache is not None and not isinstance(
            fold_cache,
            Qwen3GlobalCodebookFoldCache,
        ):
            raise DynamicQwenAdapterError(
                "fold_cache must be Qwen3GlobalCodebookFoldCache or None."
            )
        if fold_cache is not None and (
            fold_cache.attention_prototypes_per_group
            != self.attention_prototypes_per_group
            or fold_cache.neuron_prototypes_per_neuron
            != self.neuron_prototypes_per_neuron
        ):
            raise DynamicQwenAdapterError(
                "Target topology settings differ from the cached Qwen fold."
            )
        self.fold_cache = fold_cache
        self.device = device
        self.dtype = dtype

    def supports(
        self,
        request: DynamicTargetRequest,
        parent_graphs: Mapping[str, ModelCircuitGraph],
    ) -> bool:
        return (
            request.target_family in {"auto", "qwen3_codebook"}
            and request.target_modalities == ("text",)
            and any(graph.family == "qwen3" for graph in parent_graphs.values())
        )

    @staticmethod
    def _source_name(
        request: DynamicTargetRequest,
        parent_graphs: Mapping[str, ModelCircuitGraph],
    ) -> str:
        configured = dict(request.configuration).get("source_parent")
        if configured is not None:
            if not isinstance(configured, str) or configured not in parent_graphs:
                raise DynamicQwenAdapterError(
                    "Configured source_parent is not a loaded parent."
                )
            if parent_graphs[configured].family != "qwen3":
                raise DynamicQwenAdapterError(
                    "Configured source_parent is not a dense Qwen3 model."
                )
            return configured
        candidates = [
            (name, graph)
            for name, graph in parent_graphs.items()
            if graph.family == "qwen3"
        ]
        if not candidates:
            raise DynamicQwenAdapterError(
                "No dense Qwen3 parent can define the codebook target."
            )
        candidates.sort(key=lambda item: (-item[1].unique_parameter_count, item[0]))
        return candidates[0][0]

    def materialize(
        self,
        request: DynamicTargetRequest,
        parent_models: Mapping[str, Any],
        parent_graphs: Mapping[str, ModelCircuitGraph],
        inspectors: Any,
    ) -> AdapterMaterialization:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional HF boundary
            raise DynamicQwenAdapterError(
                "PyTorch is required for Qwen3 target synthesis."
            ) from exc
        source_name = self._source_name(request, parent_graphs)
        parent = parent_models[source_name]
        if self.fold_cache is not None:
            self.fold_cache.validate_parent(source_name, parent)
        parameters = tuple(parent.parameters())
        if not parameters:
            raise DynamicQwenAdapterError("Loaded Qwen3 parent has no parameters.")
        parent_devices = {value.device for value in parameters}
        parent_dtypes = {value.dtype for value in parameters}
        if self.device is None:
            if len(parent_devices) != 1:
                raise DynamicQwenAdapterError(
                    "A sharded parent requires an explicit child device."
                )
            device = next(iter(parent_devices))
        else:
            device = torch.device(self.device)
        if self.dtype is None:
            if len(parent_dtypes) != 1:
                raise DynamicQwenAdapterError(
                    "A mixed-dtype parent requires an explicit child dtype."
                )
            dtype = next(iter(parent_dtypes))
        else:
            dtype = self.dtype
        config = copy.deepcopy(getattr(parent, "config", None))
        if config is None or getattr(config, "model_type", None) != "qwen3":
            raise DynamicQwenAdapterError(
                "Codebook target synthesis requires a Qwen3 parent config."
            )
        cuda_devices: list[int] = []
        if device.type == "cuda":
            index = (
                torch.cuda.current_device() if device.index is None else device.index
            )
            cuda_devices = [index]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(request.blank_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(request.blank_seed)
            blank = type(parent)(config)
        blank.to(device=device, dtype=dtype)
        prepare_qwen3_codebook_model(
            blank,
            self.parent_tokenizer,
            self.target_tokenizer,
            attention_prototypes_per_group=(self.attention_prototypes_per_group),
            neuron_prototypes_per_neuron=(self.neuron_prototypes_per_neuron),
            attention_use_scales=self.attention_use_scales,
            neuron_use_scales=self.neuron_use_scales,
            attention_sketch_dim=self.attention_sketch_dim,
            neuron_weight_sketch_size=self.neuron_weight_sketch_size,
        )
        if self.fold_cache is not None:
            self.fold_cache.apply_topology(blank)
            self.fold_cache.assert_parameter_contract(blank)
        blank.eval()
        report = qwen3_codebook_parameter_report(blank)
        blank_state_fingerprint = _model_state_sha256(blank)
        if self.fold_cache is not None and (
            report.unique_parameter_count != self.fold_cache.expected_unique_parameters
            or report.target_vocab_size != self.fold_cache.target_vocab_size
        ):
            raise DynamicQwenAdapterError(
                "Materialized Qwen blank differs from the cached exact budget."
            )
        if self.fold_cache is not None and self.fold_cache.matches_parameters(blank):
            raise DynamicQwenAdapterError(
                "Target synthesis produced an already-folded child instead of a blank."
            )
        graph = inspectors.inspect(
            blank,
            model_name=request.child_name,
            tokenizer_fingerprint=request.tokenizer_fingerprint,
        )
        if graph.family != "qwen3_codebook":
            raise DynamicQwenAdapterError(
                "The independent inspector did not identify a Qwen3 codebook."
            )
        topology = {
            "family": graph.family,
            "logical_layers": report.logical_layers,
            "hidden_size": report.hidden_size,
            "intermediate_size": report.intermediate_size,
            "query_heads": report.query_heads,
            "key_value_heads": report.key_value_heads,
            "head_dim": report.head_dim,
            "attention_prototypes_per_group": (report.attention_prototypes_per_group),
            "neuron_prototypes_per_neuron": (report.neuron_prototypes_per_neuron),
            "target_vocab_size": report.target_vocab_size,
            "token_row_fingerprint": report.token_row_fingerprint,
            "cached_topology_fingerprint": (
                None
                if self.fold_cache is None
                else self.fold_cache.topology_fingerprint
            ),
        }
        return AdapterMaterialization(
            model=blank,
            graph=graph,
            topology_fingerprint=canonical_sha256(topology),
            report={
                "source_parent": source_name,
                "logical_layers": report.logical_layers,
                "attention_prototypes_per_group": (
                    report.attention_prototypes_per_group
                ),
                "neuron_prototypes_per_neuron": (report.neuron_prototypes_per_neuron),
                "target_vocab_size": report.target_vocab_size,
                "blank_seed": request.blank_seed,
                "blank_state_fingerprint": blank_state_fingerprint,
                "cached_fold_fingerprint": (
                    None if self.fold_cache is None else self.fold_cache.fingerprint
                ),
                "cached_topology_applied": self.fold_cache is not None,
                "blank_learned_parameters_parent_derived": False,
                "local_parent_class_instantiation": True,
                "parent_runtime_reference_count": (
                    report.parent_runtime_reference_count
                ),
                "hub_loader_used": False,
            },
            adapter_payload=report,
        )


class Qwen3CodebookCachedEvidenceProvider:
    """Expose one immutable fitted codebook cache without parent forwards."""

    def __init__(self, cache: Qwen3GlobalCodebookFoldCache) -> None:
        if not isinstance(cache, Qwen3GlobalCodebookFoldCache):
            raise DynamicQwenAdapterError("cache must be Qwen3GlobalCodebookFoldCache.")
        cache.verify()
        self.cache = cache
        self.prepare_count = 0
        self.refresh_count = 0
        self.restore_count = 0

    @property
    def base_fingerprint(self) -> str:
        return self.cache.fingerprint

    def _validate(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
    ) -> None:
        self.cache.verify()
        if action.method_id != GLOBAL_CODEBOOK_METHOD:
            raise DynamicQwenAdapterError(
                "Qwen evidence provider received a different fold method."
            )
        if len(action.sources) != 1 or (
            action.sources[0].parent_name != self.cache.source_parent
        ):
            raise DynamicQwenAdapterError(
                "Qwen action source differs from the cached parent."
            )
        if (
            context.parent_state_fingerprints[self.cache.source_parent]
            != self.cache.parent_state_fingerprint
        ):
            raise DynamicQwenAdapterError(
                "Qwen parent changed after fold-cache construction."
            )
        if set(action.target_parameter_paths) != set(self.cache.parameter_paths):
            raise DynamicQwenAdapterError(
                "Qwen action does not own every cached target parameter."
            )
        if set(action.target_node_ids) != {
            node.node_id for node in context.child_graph.nodes
        }:
            raise DynamicQwenAdapterError(
                "Qwen p35/p36 must be one complete model-level action."
            )
        self.cache.assert_parameter_contract(context.child_model)
        self.cache.assert_topology(context.child_model)

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
    ) -> FoldEvidence:
        self.prepare_count += 1
        self._validate(action, context)
        return FoldEvidence(
            fingerprint=self.cache.fingerprint,
            payload=self.cache,
            channels=action.evidence_requirements,
            parent_forward_count=0,
            child_forward_count=0,
        )

    def refresh(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        transaction_report: Mapping[str, Any],
    ) -> None:
        del transaction_report
        self._validate(action, context)
        if not self.cache.matches_parameters(context.child_model):
            raise DynamicQwenAdapterError(
                "Accepted Qwen child differs from the cached folded state."
            )
        self.refresh_count += 1

    def restore(
        self,
        context: FoldExecutionContext,
        state_fingerprint: str,
        action_ids: Sequence[str],
    ) -> None:
        del action_ids
        if context.child_state_fingerprint != state_fingerprint:
            raise DynamicQwenAdapterError(
                "Qwen evidence restore observed an inexact global rollback."
            )
        self.cache.assert_topology(context.child_model)
        self.restore_count += 1


class Qwen3CodebookFoldOnlyInitializer:
    """Prove the seeded blank and cached topology before the global action."""

    def __init__(self, cache: Qwen3GlobalCodebookFoldCache) -> None:
        self.cache = cache

    def initialize(
        self,
        plan: DynamicFoldPlan,
        context: FoldExecutionContext,
    ) -> Mapping[str, Any]:
        self.cache.verify()
        if (
            len(plan.actions) != 1
            or plan.actions[0].method_id != GLOBAL_CODEBOOK_METHOD
        ):
            raise DynamicQwenAdapterError(
                "Qwen codebook execution requires one global model-level action."
            )
        action = plan.actions[0]
        if not action.signature.zero_gradient:
            raise DynamicQwenAdapterError(
                "Qwen global codebook action is not declared zero-gradient."
            )
        if plan.evidence_fingerprint != self.cache.fingerprint:
            raise DynamicQwenAdapterError(
                "Qwen plan is not bound to the fitted codebook cache."
            )
        if plan.exact_target_unique_parameters != self.cache.expected_unique_parameters:
            raise DynamicQwenAdapterError(
                "Qwen plan budget differs from cached target storage."
            )
        if set(action.target_parameter_paths) != set(self.cache.parameter_paths):
            raise DynamicQwenAdapterError(
                "Qwen global action does not own the complete target state."
            )
        if context.child_graph.family != "qwen3_codebook":
            raise DynamicQwenAdapterError(
                "Qwen initializer received a non-codebook child."
            )
        self.cache.assert_parameter_contract(context.child_model)
        self.cache.assert_topology(context.child_model)
        if self.cache.matches_parameters(context.child_model):
            raise DynamicQwenAdapterError(
                "Qwen initializer received an already-folded child, not a blank."
            )
        state = context.child_state_fingerprint
        return {
            "kind": "qwen3_global_codebook_cached_parent",
            "child_state_fingerprint": state,
            "complete_target_coverage": False,
            "cache_fingerprint": self.cache.fingerprint,
            "cached_parent_forward_count": 0,
            "blank_seed_erasure_pending": True,
            "zero_gradient": True,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }


class Qwen3GlobalCodebookTransaction:
    method_id = GLOBAL_CODEBOOK_METHOD

    def __init__(
        self,
        *,
        context: FoldExecutionContext,
        cache: Qwen3GlobalCodebookFoldCache,
        incumbent_state_fingerprint: str,
        candidate_state_fingerprint: str,
        snapshots: tuple[tuple[nn.Parameter, Tensor], ...],
        overwritten_parameter_paths: tuple[str, ...],
    ) -> None:
        self.context = context
        self.cache = cache
        self.incumbent_state_fingerprint = incumbent_state_fingerprint
        self.candidate_state_fingerprint = candidate_state_fingerprint
        self.snapshots = snapshots
        self.overwritten_parameter_paths = overwritten_parameter_paths
        self._active = True
        self.verified_noop = False

    @property
    def active(self) -> bool:
        return self._active

    def _report(self, *, rollback_exact: bool | None) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "cache_fingerprint": self.cache.fingerprint,
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "overwritten_parameter_path_count": len(self.overwritten_parameter_paths),
            "overwritten_unique_storage_count": len(self.snapshots),
            "seed_erasure_by_construction": True,
            "cached_parent_forward_count": 0,
            "zero_gradient": True,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "rollback_exact": rollback_exact,
        }

    def commit(self) -> Mapping[str, Any]:
        if not self._active:
            raise DynamicQwenAdapterError(
                "Qwen global transaction is no longer active."
            )
        if self.context.child_state_fingerprint != (self.candidate_state_fingerprint):
            raise DynamicQwenAdapterError(
                "Qwen candidate changed before transaction commit."
            )
        self.cache.assert_topology(self.context.child_model)
        if not self.cache.matches_parameters(self.context.child_model):
            raise DynamicQwenAdapterError(
                "Qwen candidate differs from cached folded parameters."
            )
        self._active = False
        return self._report(rollback_exact=None)

    def rollback(self) -> Mapping[str, Any]:
        if not self._active:
            raise DynamicQwenAdapterError(
                "Qwen global transaction is no longer active."
            )
        with torch.no_grad():
            for parameter, value in self.snapshots:
                parameter.copy_(
                    value.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
        self._active = False
        rollback_exact = self.context.child_state_fingerprint == (
            self.incumbent_state_fingerprint
        )
        return self._report(rollback_exact=rollback_exact)


class Qwen3GlobalCodebookFoldOperator:
    """Install the cached p35/p36 state into the exact blank in one action."""

    method_id = GLOBAL_CODEBOOK_METHOD

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> FoldTransaction:
        if action.method_id != self.method_id:
            raise DynamicQwenAdapterError("Qwen operator received a different method.")
        if not isinstance(evidence.payload, Qwen3GlobalCodebookFoldCache):
            raise DynamicQwenAdapterError(
                "Qwen operator requires cached global codebook evidence."
            )
        cache = evidence.payload
        cache.verify()
        if evidence.fingerprint != cache.fingerprint:
            raise DynamicQwenAdapterError(
                "Qwen evidence fingerprint differs from its cache."
            )
        if action.compression_delta <= 0:
            raise DynamicQwenAdapterError(
                "Qwen global codebook action does not compact the parent."
            )
        if set(action.target_parameter_paths) != set(cache.parameter_paths):
            raise DynamicQwenAdapterError(
                "Qwen action target ownership differs from cached parameters."
            )
        cache.assert_parameter_contract(context.child_model)
        cache.assert_topology(context.child_model)
        incumbent = context.child_state_fingerprint
        snapshots: dict[int, tuple[nn.Parameter, Tensor]] = {}
        try:
            with torch.no_grad():
                for path, cached in cache.parameter_values:
                    parameter = context.child_model.get_parameter(path)
                    snapshots.setdefault(
                        id(parameter),
                        (
                            parameter,
                            parameter.detach().to(device="cpu").contiguous().clone(),
                        ),
                    )
                    if (
                        tuple(parameter.shape) != tuple(cached.shape)
                        or parameter.dtype != cached.dtype
                    ):
                        raise DynamicQwenAdapterError(
                            f"Cached Qwen parameter {path!r} has wrong shape/dtype."
                        )
                    parameter.copy_(cached.to(device=parameter.device))
            cache.assert_topology(context.child_model)
            if not cache.matches_parameters(context.child_model):
                raise DynamicQwenAdapterError(
                    "Qwen operator did not install the complete cached state."
                )
            candidate = context.child_state_fingerprint
            if candidate == incumbent:
                raise DynamicQwenAdapterError(
                    "Qwen global fold did not erase the blank initialization."
                )
        except BaseException:
            with torch.no_grad():
                for parameter, value in snapshots.values():
                    parameter.copy_(
                        value.to(
                            device=parameter.device,
                            dtype=parameter.dtype,
                        )
                    )
            raise
        return Qwen3GlobalCodebookTransaction(
            context=context,
            cache=cache,
            incumbent_state_fingerprint=incumbent,
            candidate_state_fingerprint=candidate,
            snapshots=tuple(snapshots.values()),
            overwritten_parameter_paths=tuple(sorted(action.target_parameter_paths)),
        )


class Qwen3CodebookFoldAdmissionEvaluator:
    """Local structural gate; whole-output behavior remains independently injected."""

    def evaluate(self, context: FoldEvaluationContext) -> FoldDecision:
        transaction = context.transaction
        if not isinstance(transaction, Qwen3GlobalCodebookTransaction):
            return FoldDecision.reject(
                "Qwen global action lacks its typed codebook transaction."
            )
        metrics = {
            "local_gate_only": True,
            "seed_erasure_by_construction": True,
            "cached_parent_forward_count": 0,
            "target_unique_parameters": (
                context.execution_context.child_graph.unique_parameter_count
            ),
            "compression_delta": context.action.compression_delta,
        }
        if context.action.method_id != GLOBAL_CODEBOOK_METHOD:
            return FoldDecision.reject("Unexpected Qwen fold method.", metrics)
        if context.action.compression_delta <= 0:
            return FoldDecision.reject(
                "Qwen codebook candidate does not compact the parent.",
                metrics,
            )
        if not transaction.cache.matches_parameters(
            context.execution_context.child_model
        ):
            return FoldDecision.reject(
                "Qwen candidate differs from cached folded parameters.",
                metrics,
            )
        transaction.cache.assert_topology(context.execution_context.child_model)
        return FoldDecision.accept(
            "Global Qwen p35/p36 codebook state passed exact local invariants; "
            "whole-output behavior still requires the final evaluator.",
            metrics,
        )


def build_qwen3_codebook_tokenizer_contract(
    *,
    target_tokenizer: Any,
    cache: Qwen3GlobalCodebookFoldCache,
    tokenizer_name_or_path: str,
) -> dict[str, Any]:
    if not isinstance(tokenizer_name_or_path, str) or not tokenizer_name_or_path:
        raise DynamicQwenAdapterError("tokenizer_name_or_path must be nonempty.")
    get_vocab = getattr(target_tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise DynamicQwenAdapterError("target_tokenizer must expose get_vocab().")
    vocabulary = get_vocab()
    if not isinstance(vocabulary, Mapping) or len(vocabulary) != (
        cache.target_vocab_size
    ):
        raise DynamicQwenAdapterError(
            "Target tokenizer vocabulary differs from the cached child."
        )
    parent_rows = dict(cache.topology_buffers)["parent_token_rows"]
    return {
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "vocab_size": cache.target_vocab_size,
        "vocab_sha256": hashlib.sha256(
            canonical_json_bytes(dict(vocabulary))
        ).hexdigest(),
        "special_token_ids": {
            name: getattr(target_tokenizer, name, None)
            for name in ("bos_token_id", "eos_token_id", "pad_token_id")
        },
        "parent_token_rows_sha256": hashlib.sha256(
            canonical_json_bytes(
                [
                    int(value)
                    for value in parent_rows.detach()
                    .to(device="cpu", dtype=torch.int64)
                    .tolist()
                ]
            )
        ).hexdigest(),
    }


class Qwen3CodebookCheckpointIO:
    """Canonical standalone checkpoint callbacks for the dynamic publisher."""

    saver_id = "neural_forge.qwen3.codebook_checkpoint.v3"
    loader_id = "neural_forge.qwen3.codebook_checkpoint_loader.v3"

    def __init__(
        self,
        *,
        tokenizer_contract: Mapping[str, Any],
        load_device: str | torch.device = "cpu",
    ) -> None:
        if not isinstance(tokenizer_contract, Mapping) or not tokenizer_contract:
            raise DynamicQwenAdapterError(
                "tokenizer_contract must be a nonempty mapping."
            )
        self.tokenizer_contract = copy.deepcopy(dict(tokenizer_contract))
        self.load_device = torch.device(load_device)

    def save(self, model: nn.Module, path: Path) -> None:
        save_qwen3_codebook_checkpoint(
            model,
            path,
            tokenizer_contract=self.tokenizer_contract,
        )

    def load(self, path: Path) -> nn.Module:
        return load_qwen3_codebook_checkpoint(
            path,
            device=self.load_device,
        )


class Qwen3CodebookStackFactory:
    """Build the existing compiler stack for one cached p35/p36 target."""

    adapter_id = "neural_forge.qwen3.codebook.stack"
    version = 1
    priority = 200
    target_families = ("qwen3_codebook",)
    target_modalities = ("text",)
    method_ids = (GLOBAL_CODEBOOK_METHOD,)

    def __init__(
        self,
        *,
        cache: Qwen3GlobalCodebookFoldCache,
        final_evaluator: FoldEvaluator,
        tokenizer_contract: Mapping[str, Any],
        fresh_reload_evaluator: Callable[[nn.Module], FoldDecision | bool],
        load_device: str | torch.device = "cpu",
    ) -> None:
        if not isinstance(cache, Qwen3GlobalCodebookFoldCache):
            raise DynamicQwenAdapterError("cache must be Qwen3GlobalCodebookFoldCache.")
        if final_evaluator is None or not callable(
            getattr(final_evaluator, "evaluate", None)
        ):
            raise DynamicQwenAdapterError("A whole-output final evaluator is required.")
        if not callable(fresh_reload_evaluator):
            raise DynamicQwenAdapterError("fresh_reload_evaluator must be callable.")
        self.cache = cache
        self.final_evaluator = final_evaluator
        self.fresh_reload_evaluator = fresh_reload_evaluator
        try:
            final_binding = resolve_behavior_gate_binding(final_evaluator)
        except DynamicBehaviorGateError:
            final_binding = None
        try:
            fresh_binding = resolve_behavior_gate_binding(fresh_reload_evaluator)
        except DynamicBehaviorGateError:
            fresh_binding = None
        if (final_binding is None) != (fresh_binding is None):
            raise DynamicQwenAdapterError(
                "Final and fresh-reload evaluators must expose the same canonical "
                "behavior gate manifest."
            )
        if (
            final_binding is not None
            and fresh_binding is not None
            and final_binding.manifest_fingerprint
            != fresh_binding.manifest_fingerprint
        ):
            raise DynamicQwenAdapterError(
                "Final and fresh-reload behavior gate manifests differ."
            )
        self.behavior_gate_binding = final_binding
        self.checkpoint_io = Qwen3CodebookCheckpointIO(
            tokenizer_contract=tokenizer_contract,
            load_device=load_device,
        )

    def supports(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> bool:
        return bool(
            isinstance(target, DynamicTargetSynthesisResult)
            and target.adapter_id == Qwen3CodebookTargetSynthesisAdapter.adapter_id
            and target.adapter_version == Qwen3CodebookTargetSynthesisAdapter.version
            and target.resource_profile.family == "qwen3_codebook"
            and target.resource_profile.modalities == ("text",)
            and target.adapter_report.get("cached_fold_fingerprint")
            == self.cache.fingerprint
        )

    def build(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> DynamicFoldSystemStack:
        self.cache.verify()
        if not self.supports(target):
            raise DynamicQwenAdapterError(
                "Synthesized target is incompatible with this Qwen stack adapter."
            )
        if (
            target.graph.family != "qwen3_codebook"
            or target.resource_profile.unique_parameters
            != self.cache.expected_unique_parameters
            or target.request.exact_unique_parameters
            != self.cache.expected_unique_parameters
        ):
            raise DynamicQwenAdapterError(
                "Synthesized Qwen target differs from the cached exact budget."
            )
        if target.adapter_report.get("cached_fold_fingerprint") != (
            self.cache.fingerprint
        ):
            raise DynamicQwenAdapterError(
                "Qwen target was not materialized with this cached topology."
            )
        self.cache.assert_parameter_contract(target.model)
        self.cache.assert_topology(target.model)
        behavior_binding = self.behavior_gate_binding
        return DynamicFoldSystemStack(
            evidence_fingerprint=self.cache.fingerprint,
            operators=FoldOperatorRegistry((Qwen3GlobalCodebookFoldOperator(),)),
            evidence_provider=Qwen3CodebookCachedEvidenceProvider(self.cache),
            evaluator=Qwen3CodebookFoldAdmissionEvaluator(),
            final_evaluator=self.final_evaluator,
            model_saver=self.checkpoint_io.save,
            model_loader=self.checkpoint_io.load,
            fresh_reload_evaluator=self.fresh_reload_evaluator,
            saver_id=self.checkpoint_io.saver_id,
            loader_id=self.checkpoint_io.loader_id,
            behavior_gate_id=(
                "neural_forge.qwen3_codebook.whole_output_behavior"
                if behavior_binding is None
                else behavior_binding.gate_id
            ),
            behavior_gate_version=(
                1 if behavior_binding is None else behavior_binding.gate_version
            ),
            behavior_baseline_fingerprint=(
                self.cache.fingerprint
                if behavior_binding is None
                else behavior_binding.baseline_fingerprint
            ),
            initializer=Qwen3CodebookFoldOnlyInitializer(self.cache),
        )

    def __call__(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> DynamicFoldSystemStack:
        return self.build(target)


__all__ = [
    "ADAPTER_SCHEMA",
    "DynamicQwenAdapterError",
    "GLOBAL_CODEBOOK_METHOD",
    "Qwen3CodebookCachedEvidenceProvider",
    "Qwen3CodebookCheckpointIO",
    "Qwen3CodebookFoldAdmissionEvaluator",
    "Qwen3CodebookFoldOnlyInitializer",
    "Qwen3CodebookStackFactory",
    "Qwen3CodebookTargetSynthesisAdapter",
    "Qwen3GlobalCodebookFoldCache",
    "Qwen3GlobalCodebookFoldOperator",
    "Qwen3GlobalCodebookTransaction",
    "build_qwen3_codebook_tokenizer_contract",
    "build_qwen3_global_codebook_fold_cache",
    "build_qwen3_global_codebook_fold_cache_from_checkpoint",
]
