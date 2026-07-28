"""Progressive, non-final LFM2 component-fold ablation hybrids.

This module is a diagnostic seam, not a competing model compiler.  It answers
one bounded question: when an already-folded compact operator is placed inside
the original LFM2 residual stream, which folded subsystem preserves whole
decoded behavior and which subsystem breaks composition?

Each replacement is an actual physical parameter substitution.  A parent
attention, MLP, short-convolution, or terminal-readout operator is removed and
replaced by the corresponding compact-child operator plus authenticated dense
parent-to-child and child-to-parent interface maps.  The maps are counted as
parameters.  No parameter delta is attributed to quantization, sparsity,
pruning, sharing, or an uncounted sidecar.

The resulting checkpoints deliberately retain the parent shell and explicit
bridges.  Every receipt therefore labels them ``NON-FINAL`` and forbids using
their parameter ratio as evidence of standalone child compression.  Their
purpose is fault localization: isolated replacements test one component,
topology-derived whole-subsystem groups test distributed attention/MLP/conv
folds, subsystem-composition groups test their interactions, and ordered
cumulative prefixes expose the first interaction that destroys useful behavior.

Only the concrete LFM2 runtime seams proven below are materialized.  A future
family adapter can consume the generic manifest, but unsupported module layouts
fail closed instead of being guessed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    lfm2_authenticated_tensor_sha256,
)


LFM2_PROGRESSIVE_COMPONENT_CAPSULE_SCHEMA = (
    "neural-forge-lfm2-progressive-component-capsule-v1"
)
LFM2_PROGRESSIVE_COMPONENT_INSTALL_SCHEMA = (
    "neural-forge-lfm2-progressive-component-install-v1"
)
LFM2_PROGRESSIVE_COMPONENT_SCHEDULE_SCHEMA = (
    "neural-forge-lfm2-progressive-component-schedule-v1"
)
LFM2_PROGRESSIVE_HYBRID_MANIFEST_SCHEMA = (
    "neural-forge-lfm2-progressive-hybrid-checkpoint-v1"
)
HYBRID_MANIFEST_FILENAME = "neural-forge-non-final-hybrid-manifest.json"
HYBRID_TENSOR_FILENAME = "model.safetensors"

Lfm2ComponentKind = Literal[
    "attention",
    "mlp",
    "short_conv",
    "terminal_readout",
]
Lfm2AblationMode = Literal[
    "isolated",
    "subsystem_group",
    "subsystem_cumulative",
    "cumulative",
]


class Lfm2ProgressiveComponentAblationError(RuntimeError):
    """A component, bridge, receipt, artifact, or runtime seam is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required runtime
        raise RuntimeError("Progressive LFM2 ablation requires PyTorch.") from exc
    return torch


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ProgressiveComponentAblationError(
            f"{label} must be one lowercase SHA-256 digest."
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _ordinary_directory(
    value: str | os.PathLike[str],
    *,
    label: str,
    must_exist: bool,
) -> Path:
    if "://" in os.fspath(value):
        raise Lfm2ProgressiveComponentAblationError(f"{label} must be local.")
    path = Path(value).expanduser().absolute()
    if must_exist:
        if path.is_symlink() or not path.is_dir():
            raise Lfm2ProgressiveComponentAblationError(
                f"{label} must be one ordinary existing directory."
            )
    elif path.exists() or path.is_symlink():
        raise Lfm2ProgressiveComponentAblationError(f"{label} must not already exist.")
    elif not path.parent.is_dir() or path.parent.is_symlink():
        raise Lfm2ProgressiveComponentAblationError(
            f"{label} parent must be one ordinary directory."
        )
    return path


def _plain_matrix(value: Any, *, label: str) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != 2
        or not value.dtype.is_floating_point
        or value.requires_grad
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Lfm2ProgressiveComponentAblationError(
            f"{label} must be one finite, dense, detached floating matrix."
        )
    return value.detach().to(device="cpu").contiguous()


def _module_state_sha256(module: Any) -> str:
    state_dict = getattr(module, "state_dict", None)
    if not callable(state_dict):
        raise Lfm2ProgressiveComponentAblationError("A component has no state_dict().")
    rows = []
    for name, tensor in sorted(state_dict().items()):
        rows.append(
            {
                "name": str(name),
                "shape": [int(size) for size in tensor.shape],
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "sha256": lfm2_authenticated_tensor_sha256(tensor),
            }
        )
    if not rows:
        raise Lfm2ProgressiveComponentAblationError(
            "A compactable component must contain state tensors."
        )
    return canonical_sha256(rows)


def _model_state_sha256(model: Any) -> str:
    return _module_state_sha256(model)


def _unique_parameter_elements(module: Any) -> int:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        raise Lfm2ProgressiveComponentAblationError("A component has no parameters().")
    return sum(int(parameter.numel()) for parameter in parameters())


def _unique_parameter_bytes(module: Any) -> int:
    return sum(
        int(parameter.numel()) * int(parameter.element_size())
        for parameter in module.parameters()
    )


def _stored_tensor_elements(module: Any) -> int:
    seen: set[tuple[str, str, int, int]] = set()
    total = 0
    for tensor in module.state_dict().values():
        storage_nbytes = int(tensor.untyped_storage().nbytes())
        identity = (
            str(tensor.device),
            str(tensor.dtype),
            int(tensor.untyped_storage().data_ptr()),
            storage_nbytes,
        )
        if identity in seen:
            continue
        seen.add(identity)
        element_size = int(tensor.element_size())
        if storage_nbytes % element_size:
            raise Lfm2ProgressiveComponentAblationError(
                "Parameter storage byte width is not divisible by its element size."
            )
        total += storage_nbytes // element_size
    return total


def _parameter_alias_topology(
    module: Any,
) -> tuple[
    tuple[
        str,
        int,
        tuple[tuple[str, int, tuple[int, ...], tuple[int, ...]], ...],
    ],
    ...,
]:
    """Return exact runtime parameter-storage alias groups, excluding singletons."""

    groups: dict[tuple[str, str, int, int], list[tuple[Any, ...]]] = {}
    try:
        parameters = module.named_parameters(remove_duplicate=False)
    except TypeError as exc:  # pragma: no cover - requires modern supported PyTorch
        raise Lfm2ProgressiveComponentAblationError(
            "Alias authentication requires named_parameters(remove_duplicate=False)."
        ) from exc
    for name, parameter in parameters:
        identity = (
            str(parameter.device),
            str(parameter.dtype),
            int(parameter.untyped_storage().data_ptr()),
            int(parameter.untyped_storage().nbytes()),
        )
        groups.setdefault(identity, []).append(
            (
                str(name),
                int(parameter.storage_offset()),
                tuple(int(size) for size in parameter.shape),
                tuple(int(stride) for stride in parameter.stride()),
            )
        )
    return tuple(
        sorted(
            (
                identity[1],
                identity[3],
                tuple(sorted(members)),
            )
            for identity, members in groups.items()
            if len(members) > 1
        )
    )


def _parameter_alias_receipt(module: Any) -> dict[str, Any]:
    groups = _parameter_alias_topology(module)
    plain_groups = [
        {
            "storage_dtype": storage_dtype,
            "storage_nbytes": storage_nbytes,
            "members": [
                {
                    "name": name,
                    "storage_offset": storage_offset,
                    "shape": list(shape),
                    "stride": list(stride),
                }
                for name, storage_offset, shape, stride in members
            ],
        }
        for storage_dtype, storage_nbytes, members in groups
    ]
    return {
        "alias_groups": plain_groups,
        "alias_group_count": len(groups),
        "native_runtime_alias_link_count": sum(
            len(members) - 1 for _, _, members in groups
        ),
        "alias_topology_sha256": canonical_sha256(plain_groups),
    }


def _authenticated_parameter_alias_topology(
    value: Any,
    *,
    label: str,
) -> tuple[Any, ...]:
    if not isinstance(value, Mapping):
        raise Lfm2ProgressiveComponentAblationError(
            f"{label} must be one parameter-alias receipt."
        )
    raw_groups = value.get("alias_groups")
    if not isinstance(raw_groups, list):
        raise Lfm2ProgressiveComponentAblationError(f"{label} has no alias groups.")
    groups = []
    names_seen: set[str] = set()
    for raw_group in raw_groups:
        if not isinstance(raw_group, Mapping):
            raise Lfm2ProgressiveComponentAblationError(
                f"{label} contains a malformed alias group."
            )
        storage_dtype = raw_group.get("storage_dtype")
        storage_nbytes = raw_group.get("storage_nbytes")
        raw_members = raw_group.get("members")
        members = []
        if (
            not isinstance(storage_dtype, str)
            or not storage_dtype
            or isinstance(storage_nbytes, bool)
            or not isinstance(storage_nbytes, int)
            or storage_nbytes <= 0
            or not isinstance(raw_members, list)
            or len(raw_members) < 2
        ):
            raise Lfm2ProgressiveComponentAblationError(
                f"{label} contains malformed storage metadata."
            )
        for raw_member in raw_members:
            if not isinstance(raw_member, Mapping):
                raise Lfm2ProgressiveComponentAblationError(
                    f"{label} contains a malformed alias member."
                )
            name = raw_member.get("name")
            offset = raw_member.get("storage_offset")
            shape = raw_member.get("shape")
            stride = raw_member.get("stride")
            if (
                not isinstance(name, str)
                or not name
                or name in names_seen
                or isinstance(offset, bool)
                or not isinstance(offset, int)
                or offset < 0
                or not isinstance(shape, list)
                or not shape
                or not isinstance(stride, list)
                or len(stride) != len(shape)
                or any(
                    isinstance(size, bool) or not isinstance(size, int) or size <= 0
                    for size in shape
                )
                or any(
                    isinstance(step, bool) or not isinstance(step, int)
                    for step in stride
                )
            ):
                raise Lfm2ProgressiveComponentAblationError(
                    f"{label} contains a malformed alias member."
                )
            names_seen.add(name)
            members.append((name, offset, tuple(shape), tuple(stride)))
        if members != sorted(members):
            raise Lfm2ProgressiveComponentAblationError(
                f"{label} alias members are not canonical."
            )
        groups.append((storage_dtype, storage_nbytes, tuple(members)))
    topology = tuple(groups)
    expected_plain = [
        {
            "storage_dtype": storage_dtype,
            "storage_nbytes": storage_nbytes,
            "members": [
                {
                    "name": name,
                    "storage_offset": offset,
                    "shape": list(shape),
                    "stride": list(stride),
                }
                for name, offset, shape, stride in members
            ],
        }
        for storage_dtype, storage_nbytes, members in sorted(topology)
    ]
    if (
        raw_groups != expected_plain
        or value.get("alias_group_count") != len(topology)
        or value.get("native_runtime_alias_link_count")
        != sum(len(members) - 1 for _, _, members in topology)
        or value.get("alias_topology_sha256") != canonical_sha256(expected_plain)
    ):
        raise Lfm2ProgressiveComponentAblationError(
            f"{label} seal or counts are invalid."
        )
    return topology


def _module_at(root: Any, path: str) -> Any:
    if (
        not isinstance(path, str)
        or not path
        or path.startswith(".")
        or path.endswith(".")
        or ".." in path
    ):
        raise Lfm2ProgressiveComponentAblationError(
            "Component module paths must be canonical dotted paths."
        )
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            raise Lfm2ProgressiveComponentAblationError(
                f"Component module path does not exist: {path!r}."
            )
        current = getattr(current, part)
    return current


def _replace_module(root: Any, path: str, replacement: Any) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        if not hasattr(parent, part):
            raise Lfm2ProgressiveComponentAblationError(
                f"Component parent path does not exist: {path!r}."
            )
        parent = getattr(parent, part)
    leaf = parts[-1]
    if not hasattr(parent, leaf):
        raise Lfm2ProgressiveComponentAblationError(
            f"Component leaf path does not exist: {path!r}."
        )
    setattr(parent, leaf, replacement)


def _linear_shape(module: Any, *, label: str) -> tuple[int, int, bool]:
    torch = _torch()
    weight = getattr(module, "weight", None)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.ndim != 2
        or weight.device.type == "meta"
    ):
        raise Lfm2ProgressiveComponentAblationError(
            f"{label} must expose one materialized rank-2 weight."
        )
    bias = getattr(module, "bias", None)
    if bias is not None and (
        not isinstance(bias, torch.Tensor)
        or bias.ndim != 1
        or int(bias.numel()) != int(weight.shape[0])
    ):
        raise Lfm2ProgressiveComponentAblationError(
            f"{label} bias does not match its output width."
        )
    return int(weight.shape[1]), int(weight.shape[0]), bias is not None


def _component_geometry(
    module: Any,
    kind: Lfm2ComponentKind,
    *,
    label: str,
) -> dict[str, Any]:
    if kind == "attention":
        required = ("q_proj", "k_proj", "v_proj", "out_proj", "head_dim")
        if any(not hasattr(module, name) for name in required):
            raise Lfm2ProgressiveComponentAblationError(
                f"{label} is not a structurally supported LFM2 attention operator."
            )
        q_in, q_out, _ = _linear_shape(module.q_proj, label=f"{label}.q_proj")
        k_in, k_out, _ = _linear_shape(module.k_proj, label=f"{label}.k_proj")
        v_in, v_out, _ = _linear_shape(module.v_proj, label=f"{label}.v_proj")
        o_in, o_out, _ = _linear_shape(module.out_proj, label=f"{label}.out_proj")
        head_dim = int(module.head_dim)
        if (
            q_in != k_in
            or q_in != v_in
            or q_out != o_in
            or k_out != v_out
            or min(q_in, q_out, k_out, o_out, head_dim) <= 0
            or q_out % head_dim
            or k_out % head_dim
        ):
            raise Lfm2ProgressiveComponentAblationError(
                f"{label} attention geometry is internally inconsistent."
            )
        return {
            "external_input_width": q_in,
            "external_output_width": o_out,
            "head_dim": head_dim,
            "query_head_count": q_out // head_dim,
            "key_value_head_count": k_out // head_dim,
        }
    if kind == "mlp":
        if any(not hasattr(module, name) for name in ("w1", "w2", "w3")):
            raise Lfm2ProgressiveComponentAblationError(
                f"{label} is not a structurally supported LFM2 MLP."
            )
        w1_in, w1_out, _ = _linear_shape(module.w1, label=f"{label}.w1")
        w3_in, w3_out, _ = _linear_shape(module.w3, label=f"{label}.w3")
        w2_in, w2_out, _ = _linear_shape(module.w2, label=f"{label}.w2")
        if w1_in != w3_in or w1_out != w3_out or w1_out != w2_in:
            raise Lfm2ProgressiveComponentAblationError(
                f"{label} MLP geometry is internally inconsistent."
            )
        return {
            "external_input_width": w1_in,
            "external_output_width": w2_out,
            "intermediate_width": w1_out,
        }
    if kind == "short_conv":
        if any(
            not hasattr(module, name)
            for name in ("in_proj", "out_proj", "conv", "L_cache")
        ):
            raise Lfm2ProgressiveComponentAblationError(
                f"{label} is not a structurally supported LFM2 short convolution."
            )
        in_in, in_out, _ = _linear_shape(
            module.in_proj,
            label=f"{label}.in_proj",
        )
        out_in, out_out, _ = _linear_shape(
            module.out_proj,
            label=f"{label}.out_proj",
        )
        conv_weight = getattr(module.conv, "weight", None)
        if (
            in_out != 3 * in_in
            or out_in != in_in
            or not hasattr(conv_weight, "shape")
            or tuple(int(size) for size in conv_weight.shape[:2]) != (in_in, 1)
            or int(conv_weight.shape[2]) != int(module.L_cache)
        ):
            raise Lfm2ProgressiveComponentAblationError(
                f"{label} short-convolution geometry is internally inconsistent."
            )
        return {
            "external_input_width": in_in,
            "external_output_width": out_out,
            "conv_channel_width": in_in,
            "conv_kernel_width": int(module.L_cache),
        }
    if kind == "terminal_readout":
        input_width, output_width, has_bias = _linear_shape(module, label=label)
        return {
            "external_input_width": input_width,
            "external_output_width": output_width,
            "has_bias": has_bias,
        }
    raise Lfm2ProgressiveComponentAblationError(
        f"Unsupported LFM2 component kind: {kind!r}."
    )


def _frozen_linear_from_row_map(
    row_map: Any,
    *,
    source_width: int,
    target_width: int,
    dtype: Any,
    device: Any,
    label: str,
) -> Any:
    """Build ``x @ row_map`` as a counted, frozen ``nn.Linear``."""

    torch = _torch()
    matrix = _plain_matrix(row_map, label=label)
    if tuple(int(size) for size in matrix.shape) != (source_width, target_width):
        raise Lfm2ProgressiveComponentAblationError(
            f"{label} must have shape [{source_width}, {target_width}]."
        )
    layer = torch.nn.Linear(source_width, target_width, bias=False)
    layer.to(device=device, dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(matrix.T.to(device=device, dtype=dtype))
    layer.requires_grad_(False)
    return layer


class Lfm2BridgedCompactAttention:  # instantiated through __new__ below
    """Typing marker for the concrete ``torch.nn.Module`` implementation."""


class Lfm2BridgedCompactMlp:  # instantiated through __new__ below
    """Typing marker for the concrete ``torch.nn.Module`` implementation."""


class Lfm2BridgedCompactShortConv:  # instantiated through __new__ below
    """Typing marker for the concrete ``torch.nn.Module`` implementation."""


class Lfm2BridgedCompactReadout:  # instantiated through __new__ below
    """Typing marker for the concrete ``torch.nn.Module`` implementation."""


def _wrapper_classes() -> Mapping[Lfm2ComponentKind, type[Any]]:
    """Create real ``nn.Module`` classes lazily so importing stays torch-optional."""

    torch = _torch()

    class BridgedAttention(torch.nn.Module):
        def __init__(self, input_bridge: Any, operator: Any, output_bridge: Any):
            super().__init__()
            self.input_bridge = input_bridge
            self.compact_operator = operator
            self.output_bridge = output_bridge

        def forward(
            self,
            hidden_states: Any,
            position_embeddings: Any,
            attention_mask: Any,
            past_key_values: Any = None,
            **kwargs: Any,
        ) -> tuple[Any, Any]:
            compact_hidden = self.input_bridge(hidden_states)
            compact_output, attention_weights = self.compact_operator(
                hidden_states=compact_hidden,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            return self.output_bridge(compact_output), attention_weights

    class BridgedMlp(torch.nn.Module):
        def __init__(self, input_bridge: Any, operator: Any, output_bridge: Any):
            super().__init__()
            self.input_bridge = input_bridge
            self.compact_operator = operator
            self.output_bridge = output_bridge

        def forward(self, hidden_states: Any) -> Any:
            return self.output_bridge(
                self.compact_operator(self.input_bridge(hidden_states))
            )

    class BridgedShortConv(torch.nn.Module):
        def __init__(self, input_bridge: Any, operator: Any, output_bridge: Any):
            super().__init__()
            self.input_bridge = input_bridge
            self.compact_operator = operator
            self.output_bridge = output_bridge

        def forward(
            self,
            hidden_states: Any,
            past_key_values: Any = None,
            attention_mask: Any = None,
            seq_idx: Any = None,
        ) -> Any:
            compact_output = self.compact_operator(
                hidden_states=self.input_bridge(hidden_states),
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                seq_idx=seq_idx,
            )
            return self.output_bridge(compact_output)

    class BridgedReadout(torch.nn.Module):
        def __init__(
            self,
            input_bridge: Any,
            operator: Any,
            output_domain_bridge: Any | None,
        ):
            super().__init__()
            self.input_bridge = input_bridge
            self.compact_operator = operator
            self.output_domain_bridge = output_domain_bridge

        def forward(self, hidden_states: Any) -> Any:
            logits = self.compact_operator(self.input_bridge(hidden_states))
            if self.output_domain_bridge is not None:
                logits = self.output_domain_bridge(logits)
            return logits

    return MappingProxyType(
        {
            "attention": BridgedAttention,
            "mlp": BridgedMlp,
            "short_conv": BridgedShortConv,
            "terminal_readout": BridgedReadout,
        }
    )


@dataclass(frozen=True, slots=True)
class Lfm2ProgressiveComponentCapsule:
    """One independently pinned compact operator and its interface maps."""

    component_id: str
    kind: Lfm2ComponentKind
    parent_module_path: str
    compact_module_path: str
    input_encoder: Any = field(repr=False, compare=False)
    output_decoder: Any | None = field(default=None, repr=False, compare=False)
    output_domain_decoder: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    expected_parent_module_sha256: str = ""
    expected_compact_module_sha256: str = ""
    expected_input_encoder_sha256: str = ""
    expected_output_decoder_sha256: str | None = None
    expected_output_domain_decoder_sha256: str | None = None
    source_fold_receipt_sha256: str = ""
    source_compact_checkpoint_sha256: str = ""
    require_parameter_reduction: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.component_id, str)
            or not self.component_id.strip()
            or self.component_id != self.component_id.strip()
        ):
            raise Lfm2ProgressiveComponentAblationError(
                "component_id must be nonempty and canonical."
            )
        if self.kind not in {
            "attention",
            "mlp",
            "short_conv",
            "terminal_readout",
        }:
            raise Lfm2ProgressiveComponentAblationError(
                f"Unsupported component kind: {self.kind!r}."
            )
        _module_path_text(self.parent_module_path, label="parent_module_path")
        _module_path_text(self.compact_module_path, label="compact_module_path")
        for value, label in (
            (self.expected_parent_module_sha256, "expected parent module"),
            (self.expected_compact_module_sha256, "expected compact module"),
            (self.expected_input_encoder_sha256, "expected input encoder"),
            (self.source_fold_receipt_sha256, "source fold receipt"),
            (self.source_compact_checkpoint_sha256, "compact checkpoint"),
        ):
            _digest(value, label=label)
        input_encoder = _plain_matrix(self.input_encoder, label="input_encoder")
        if (
            lfm2_authenticated_tensor_sha256(input_encoder)
            != self.expected_input_encoder_sha256
        ):
            raise Lfm2ProgressiveComponentAblationError(
                "Input encoder differs from its independent pin."
            )
        object.__setattr__(self, "input_encoder", input_encoder)
        _validate_optional_pinned_matrix(
            self,
            value_name="output_decoder",
            digest_name="expected_output_decoder_sha256",
        )
        _validate_optional_pinned_matrix(
            self,
            value_name="output_domain_decoder",
            digest_name="expected_output_domain_decoder_sha256",
        )
        if self.kind == "terminal_readout":
            if self.output_decoder is not None:
                raise Lfm2ProgressiveComponentAblationError(
                    "Terminal readout uses output_domain_decoder, not residual output_decoder."
                )
        elif self.output_decoder is None:
            raise Lfm2ProgressiveComponentAblationError(
                f"{self.kind} requires an authenticated residual output decoder."
            )
        elif self.output_domain_decoder is not None:
            raise Lfm2ProgressiveComponentAblationError(
                "Only terminal readout can use an output-domain decoder."
            )
        if not isinstance(self.require_parameter_reduction, bool):
            raise Lfm2ProgressiveComponentAblationError(
                "require_parameter_reduction must be boolean."
            )

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": LFM2_PROGRESSIVE_COMPONENT_CAPSULE_SCHEMA,
            "component_id": self.component_id,
            "family_adapter": "lfm2-concrete-runtime-v1",
            "kind": self.kind,
            "parent_module_path": self.parent_module_path,
            "compact_module_path": self.compact_module_path,
            "input_encoder_sha256": self.expected_input_encoder_sha256,
            "output_decoder_sha256": self.expected_output_decoder_sha256,
            "output_domain_decoder_sha256": (
                self.expected_output_domain_decoder_sha256
            ),
            "parent_module_sha256": self.expected_parent_module_sha256,
            "compact_module_sha256": self.expected_compact_module_sha256,
            "source_fold_receipt_sha256": self.source_fold_receipt_sha256,
            "source_compact_checkpoint_sha256": (self.source_compact_checkpoint_sha256),
            "require_parameter_reduction": self.require_parameter_reduction,
            "parent_shell_retained": True,
            "diagnostic_hybrid_only": True,
        }
        return {**body, "capsule_sha256": canonical_sha256(body)}


def _module_path_text(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith(".")
        or value.endswith(".")
        or ".." in value
        or any(not part for part in value.split("."))
    ):
        raise Lfm2ProgressiveComponentAblationError(
            f"{label} must be one canonical dotted module path."
        )
    return value


def _validate_optional_pinned_matrix(
    capsule: Lfm2ProgressiveComponentCapsule,
    *,
    value_name: str,
    digest_name: str,
) -> None:
    value = getattr(capsule, value_name)
    expected = getattr(capsule, digest_name)
    if (value is None) != (expected is None):
        raise Lfm2ProgressiveComponentAblationError(
            f"{value_name} and {digest_name} must be paired."
        )
    if value is None:
        return
    expected = _digest(expected, label=f"expected {value_name}")
    matrix = _plain_matrix(value, label=value_name)
    if lfm2_authenticated_tensor_sha256(matrix) != expected:
        raise Lfm2ProgressiveComponentAblationError(
            f"{value_name} differs from its independent pin."
        )
    object.__setattr__(capsule, value_name, matrix)


@dataclass(frozen=True, slots=True)
class Lfm2ProgressiveComponentInstallation:
    """The installed wrapper and its sealed non-final diagnostic receipt."""

    wrapper: Any = field(repr=False, compare=False)
    receipt: Mapping[str, Any]
    reconstruction: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Lfm2ComponentAblationCandidate:
    """One component, subsystem group, or ordered composition candidate."""

    candidate_id: str
    mode: Lfm2AblationMode
    capsules: tuple[Lfm2ProgressiveComponentCapsule, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "mode": self.mode,
            "component_ids": [value.component_id for value in self.capsules],
            "component_count": len(self.capsules),
        }


def _uniform_parameter_dtype_device(module: Any) -> tuple[Any, Any]:
    parameters = tuple(module.parameters())
    if not parameters:
        raise Lfm2ProgressiveComponentAblationError(
            "A replaceable operator must contain parameters."
        )
    dtypes = {value.dtype for value in parameters}
    devices = {value.device for value in parameters}
    if len(dtypes) != 1 or len(devices) != 1:
        raise Lfm2ProgressiveComponentAblationError(
            "A replaceable operator must have one materialization dtype/device."
        )
    return next(iter(dtypes)), next(iter(devices))


def _copy_compact_operator(
    compact_operator: Any,
    *,
    dtype: Any,
    device: Any,
    parent_layer_index: int | None,
) -> Any:
    operator = copy.deepcopy(compact_operator)
    operator.to(device=device, dtype=dtype)
    operator.requires_grad_(False)
    operator.eval()
    if parent_layer_index is not None and hasattr(operator, "layer_idx"):
        operator.layer_idx = parent_layer_index
    return operator


def _layer_index_from_path(path: str) -> int | None:
    parts = path.split(".")
    for index, part in enumerate(parts[:-1]):
        if part == "layers" and parts[index + 1].isdigit():
            return int(parts[index + 1])
    return None


def install_lfm2_progressive_component(
    *,
    parent_model: Any,
    compact_model: Any,
    capsule: Lfm2ProgressiveComponentCapsule,
    ablation_mode: Lfm2AblationMode,
    step_index: int,
    prior_installation_sha256: str | None = None,
) -> Lfm2ProgressiveComponentInstallation:
    """Atomically replace one parent operator by a bridged compact operator.

    The compact module must already have been produced by the folding compiler;
    this function performs no fitting and accepts no trace tensors.  It merely
    authenticates, copies, bridges, installs, counts, and seals the replacement.
    """

    if ablation_mode not in {
        "isolated",
        "subsystem_group",
        "subsystem_cumulative",
        "cumulative",
    }:
        raise Lfm2ProgressiveComponentAblationError(
            "ablation_mode must name an isolated, grouped-subsystem, "
            "subsystem-cumulative, or ordered component-prefix experiment."
        )
    if (
        isinstance(step_index, bool)
        or not isinstance(step_index, int)
        or step_index < 0
    ):
        raise Lfm2ProgressiveComponentAblationError(
            "step_index must be a nonnegative integer."
        )
    if prior_installation_sha256 is not None:
        _digest(prior_installation_sha256, label="prior installation")
    parent_operator = _module_at(parent_model, capsule.parent_module_path)
    compact_operator = _module_at(compact_model, capsule.compact_module_path)
    parent_state_sha256 = _module_state_sha256(parent_operator)
    compact_state_sha256 = _module_state_sha256(compact_operator)
    if (
        parent_state_sha256 != capsule.expected_parent_module_sha256
        or compact_state_sha256 != capsule.expected_compact_module_sha256
    ):
        raise Lfm2ProgressiveComponentAblationError(
            "Parent or compact component differs from its independent pin."
        )
    parent_geometry = _component_geometry(
        parent_operator,
        capsule.kind,
        label="parent component",
    )
    compact_geometry = _component_geometry(
        compact_operator,
        capsule.kind,
        label="compact component",
    )
    parent_input_width = int(parent_geometry["external_input_width"])
    parent_output_width = int(parent_geometry["external_output_width"])
    compact_input_width = int(compact_geometry["external_input_width"])
    compact_output_width = int(compact_geometry["external_output_width"])
    if capsule.kind == "attention" and (
        parent_geometry["head_dim"] != compact_geometry["head_dim"]
    ):
        raise Lfm2ProgressiveComponentAblationError(
            "Bridged LFM2 attention currently requires identical rotary head_dim."
        )
    if tuple(capsule.input_encoder.shape) != (
        parent_input_width,
        compact_input_width,
    ):
        raise Lfm2ProgressiveComponentAblationError(
            "Input encoder does not map the parent residual width into the compact operator."
        )
    if capsule.kind == "terminal_readout":
        if parent_output_width != compact_output_width:
            if capsule.output_domain_decoder is None or tuple(
                capsule.output_domain_decoder.shape
            ) != (compact_output_width, parent_output_width):
                raise Lfm2ProgressiveComponentAblationError(
                    "Cross-tokenizer terminal replacement requires an authenticated "
                    "compact-vocabulary-to-parent-vocabulary output-domain decoder."
                )
        elif capsule.output_domain_decoder is not None and tuple(
            capsule.output_domain_decoder.shape
        ) != (compact_output_width, parent_output_width):
            raise Lfm2ProgressiveComponentAblationError(
                "Terminal output-domain decoder has the wrong vocabulary geometry."
            )
    elif tuple(capsule.output_decoder.shape) != (
        compact_output_width,
        parent_output_width,
    ):
        raise Lfm2ProgressiveComponentAblationError(
            "Output decoder does not map the compact residual width into the parent shell."
        )

    parent_dtype, parent_device = _uniform_parameter_dtype_device(parent_operator)
    compact_dtype, _ = _uniform_parameter_dtype_device(compact_operator)
    if compact_dtype != parent_dtype:
        raise Lfm2ProgressiveComponentAblationError(
            "Diagnostic component replacement cannot hide a dtype/quantization change."
        )
    parameter_elements_before = _unique_parameter_elements(parent_model)
    parameter_bytes_before = _unique_parameter_bytes(parent_model)
    stored_elements_before = _stored_tensor_elements(parent_model)
    parameter_aliases_before = _parameter_alias_topology(parent_model)
    parameter_alias_receipt_before = _parameter_alias_receipt(parent_model)
    original_parameter_elements = _unique_parameter_elements(parent_operator)
    compact_parameter_elements = _unique_parameter_elements(compact_operator)
    original = parent_operator
    input_bridge = _frozen_linear_from_row_map(
        capsule.input_encoder,
        source_width=parent_input_width,
        target_width=compact_input_width,
        dtype=parent_dtype,
        device=parent_device,
        label="input_encoder",
    )
    copied_operator = _copy_compact_operator(
        compact_operator,
        dtype=parent_dtype,
        device=parent_device,
        parent_layer_index=_layer_index_from_path(capsule.parent_module_path),
    )
    output_bridge = None
    output_domain_bridge = None
    if capsule.kind != "terminal_readout":
        output_bridge = _frozen_linear_from_row_map(
            capsule.output_decoder,
            source_width=compact_output_width,
            target_width=parent_output_width,
            dtype=parent_dtype,
            device=parent_device,
            label="output_decoder",
        )
    elif capsule.output_domain_decoder is not None:
        output_domain_bridge = _frozen_linear_from_row_map(
            capsule.output_domain_decoder,
            source_width=compact_output_width,
            target_width=parent_output_width,
            dtype=parent_dtype,
            device=parent_device,
            label="output_domain_decoder",
        )
    wrappers = _wrapper_classes()
    if capsule.kind == "terminal_readout":
        wrapper = wrappers[capsule.kind](
            input_bridge,
            copied_operator,
            output_domain_bridge,
        )
    else:
        wrapper = wrappers[capsule.kind](
            input_bridge,
            copied_operator,
            output_bridge,
        )
    wrapper.requires_grad_(False)
    wrapper.eval()
    try:
        _replace_module(parent_model, capsule.parent_module_path, wrapper)
        installed = _module_at(parent_model, capsule.parent_module_path)
        if installed is not wrapper:
            raise Lfm2ProgressiveComponentAblationError(
                "Component wrapper was not installed at the requested path."
            )
        parameter_elements_after = _unique_parameter_elements(parent_model)
        parameter_bytes_after = _unique_parameter_bytes(parent_model)
        stored_elements_after = _stored_tensor_elements(parent_model)
        parameter_aliases_after = _parameter_alias_topology(parent_model)
        installed_component_elements = _unique_parameter_elements(wrapper)
        parameter_delta = parameter_elements_after - parameter_elements_before
        if (
            installed_component_elements
            != compact_parameter_elements
            + _unique_parameter_elements(input_bridge)
            + (
                0
                if output_bridge is None
                else _unique_parameter_elements(output_bridge)
            )
            + (
                0
                if output_domain_bridge is None
                else _unique_parameter_elements(output_domain_bridge)
            )
            or parameter_delta
            != installed_component_elements - original_parameter_elements
            or stored_elements_after - stored_elements_before != parameter_delta
        ):
            raise Lfm2ProgressiveComponentAblationError(
                "Physical component/bridge parameter accounting is inconsistent."
            )
        if parameter_aliases_after != parameter_aliases_before:
            raise Lfm2ProgressiveComponentAblationError(
                "Component installation changed native parameter-alias topology."
            )
        if capsule.require_parameter_reduction and parameter_delta >= 0:
            raise Lfm2ProgressiveComponentAblationError(
                "The requested component plus counted bridges is not physically smaller."
            )
        if _module_state_sha256(copied_operator) != compact_state_sha256:
            raise Lfm2ProgressiveComponentAblationError(
                "Compact operator changed while being copied into the parent shell."
            )
    except Exception:
        _replace_module(parent_model, capsule.parent_module_path, original)
        if (
            _module_at(parent_model, capsule.parent_module_path) is not original
            or _parameter_alias_topology(parent_model) != parameter_aliases_before
        ):
            raise Lfm2ProgressiveComponentAblationError(
                "Failed component installation could not prove exact rollback and "
                "native parameter-alias restoration."
            )
        raise

    compact_config = getattr(compact_model, "config", None)
    to_dict = getattr(compact_config, "to_dict", None)
    if not callable(to_dict):
        # In-memory synthetic tests can still exercise installation; persistence
        # deliberately fails later unless this concrete LFM2 runtime descriptor exists.
        compact_config_dict = None
    else:
        compact_config_dict = to_dict()
    compact_attention_implementation = None
    compact_operator_config = getattr(compact_operator, "config", None)
    if compact_operator_config is not None:
        observed_attention_implementation = getattr(
            compact_operator_config,
            "_attn_implementation",
            None,
        )
        if isinstance(observed_attention_implementation, str):
            compact_attention_implementation = observed_attention_implementation
    reconstruction = {
        "family_adapter": "lfm2-concrete-runtime-v1",
        "kind": capsule.kind,
        "parent_module_path": capsule.parent_module_path,
        "parent_layer_index": _layer_index_from_path(capsule.parent_module_path),
        "compact_layer_index": _layer_index_from_path(capsule.compact_module_path),
        "compact_config": compact_config_dict,
        "compact_attention_implementation": compact_attention_implementation,
        "parent_geometry": parent_geometry,
        "compact_geometry": compact_geometry,
        "input_bridge_shape": [
            parent_input_width,
            compact_input_width,
        ],
        "output_bridge_shape": (
            None
            if output_bridge is None
            else [compact_output_width, parent_output_width]
        ),
        "output_domain_bridge_shape": (
            None
            if output_domain_bridge is None
            else [compact_output_width, parent_output_width]
        ),
        "compact_operator_state_sha256": compact_state_sha256,
        "input_encoder_sha256": capsule.expected_input_encoder_sha256,
        "output_decoder_sha256": capsule.expected_output_decoder_sha256,
        "output_domain_decoder_sha256": (capsule.expected_output_domain_decoder_sha256),
    }
    receipt_body = {
        "schema": LFM2_PROGRESSIVE_COMPONENT_INSTALL_SCHEMA,
        "status": "NON-FINAL-DIAGNOSTIC-HYBRID-COMPONENT-INSTALLED",
        "capsule": capsule.as_dict(),
        "capsule_sha256": capsule.as_dict()["capsule_sha256"],
        "component_id": capsule.component_id,
        "kind": capsule.kind,
        "ablation_mode": ablation_mode,
        "step_index": step_index,
        "prior_installation_sha256": prior_installation_sha256,
        "parent_module_path": capsule.parent_module_path,
        "compact_module_path": capsule.compact_module_path,
        "parent_geometry": parent_geometry,
        "compact_geometry": compact_geometry,
        "parent_module_state_sha256_before": parent_state_sha256,
        "compact_module_state_sha256_source": compact_state_sha256,
        "compact_module_state_sha256_installed": _module_state_sha256(copied_operator),
        "installed_wrapper_state_sha256": _module_state_sha256(wrapper),
        "physical_parameter_accounting": {
            "whole_model_parameter_elements_before": parameter_elements_before,
            "whole_model_parameter_elements_after": parameter_elements_after,
            "whole_model_parameter_element_delta": parameter_delta,
            "whole_model_parameter_bytes_before": parameter_bytes_before,
            "whole_model_parameter_bytes_after": parameter_bytes_after,
            "whole_model_parameter_byte_delta": (
                parameter_bytes_after - parameter_bytes_before
            ),
            "whole_model_stored_tensor_elements_before": stored_elements_before,
            "whole_model_stored_tensor_elements_after": stored_elements_after,
            "original_component_parameter_elements": original_parameter_elements,
            "compact_operator_parameter_elements": compact_parameter_elements,
            "input_bridge_parameter_elements": _unique_parameter_elements(input_bridge),
            "output_bridge_parameter_elements": (
                0
                if output_bridge is None
                else _unique_parameter_elements(output_bridge)
            ),
            "output_domain_bridge_parameter_elements": (
                0
                if output_domain_bridge is None
                else _unique_parameter_elements(output_domain_bridge)
            ),
            "installed_component_parameter_elements": installed_component_elements,
            "component_parameter_reduction": (
                original_parameter_elements - installed_component_elements
            ),
            "counted_bridges_are_parameters": True,
            "parameter_alias_topology_before": parameter_alias_receipt_before,
            "parameter_alias_topology_after": _parameter_alias_receipt(parent_model),
            "new_runtime_shared_parameters": 0,
            "native_parent_aliases_retained": True,
            "parameter_reduction_attributed_to_sharing": False,
        },
        "reconstruction": reconstruction,
        "operation_audit": {
            "training_steps": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "parent_trace_recaptures": 0,
            "new_parent_trace_rows": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameters_sparsified": 0,
            "new_runtime_shared_parameters": 0,
            "native_parent_aliases_retained": True,
            "parameter_reduction_attributed_to_sharing": False,
            "network_access": False,
        },
        "interpretation": {
            "parent_shell_retained": True,
            "explicit_interface_bridges_retained": True,
            "bridges_counted_in_physical_parameters": True,
            "whole_decoded_behavior_required": True,
            "numeric_metrics_have_decision_authority": False,
            "diagnostic_hybrid_only": True,
            "standalone_child_compression_claimed": False,
            "success_localizes_one_folded_component_or_composition_prefix": True,
        },
    }
    receipt = {
        **receipt_body,
        "installation_sha256": canonical_sha256(receipt_body),
    }
    return Lfm2ProgressiveComponentInstallation(
        wrapper=wrapper,
        receipt=MappingProxyType(receipt),
        reconstruction=MappingProxyType(reconstruction),
    )


def build_lfm2_progressive_component_ablation_schedule(
    capsules: Sequence[Lfm2ProgressiveComponentCapsule],
) -> tuple[Lfm2ComponentAblationCandidate, ...]:
    """Return component, whole-subsystem, and composition-localization candidates."""

    ordered = tuple(capsules)
    if not ordered:
        raise Lfm2ProgressiveComponentAblationError(
            "At least one component capsule is required."
        )
    ids = [value.component_id for value in ordered]
    paths = [value.parent_module_path for value in ordered]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise Lfm2ProgressiveComponentAblationError(
            "Component IDs and parent module paths must be unique."
        )
    isolated = tuple(
        Lfm2ComponentAblationCandidate(
            candidate_id=f"isolated--{capsule.component_id}",
            mode="isolated",
            capsules=(capsule,),
        )
        for capsule in ordered
    )
    subsystem_kinds = []
    for capsule in ordered:
        if (
            capsule.kind in {"attention", "mlp", "short_conv"}
            and capsule.kind not in subsystem_kinds
        ):
            subsystem_kinds.append(capsule.kind)
    subsystem_labels = {
        "attention": "all-attention",
        "mlp": "all-mlp",
        "short_conv": "all-short-conv",
    }
    cumulative = tuple(
        Lfm2ComponentAblationCandidate(
            candidate_id=("cumulative--" + "--".join(ids[:prefix_length])),
            mode="cumulative",
            capsules=ordered[:prefix_length],
        )
        for prefix_length in range(2, len(ordered) + 1)
    )
    protected_path_sets = {
        frozenset(value.parent_module_path for value in candidate.capsules)
        for candidate in (*isolated, *cumulative)
    }
    subsystem_groups_list = []
    supplemental_path_sets: set[frozenset[str]] = set()
    for kind in subsystem_kinds:
        members = tuple(value for value in ordered if value.kind == kind)
        path_set = frozenset(value.parent_module_path for value in members)
        if len(members) <= 1 or path_set in protected_path_sets:
            continue
        subsystem_groups_list.append(
            Lfm2ComponentAblationCandidate(
                candidate_id=f"subsystem--{subsystem_labels[kind]}",
                mode="subsystem_group",
                capsules=members,
            )
        )
        supplemental_path_sets.add(path_set)
    subsystem_cumulative_list = []
    for prefix_length in range(2, len(subsystem_kinds) + 1):
        members = tuple(
            value for value in ordered if value.kind in subsystem_kinds[:prefix_length]
        )
        path_set = frozenset(value.parent_module_path for value in members)
        if path_set in protected_path_sets or path_set in supplemental_path_sets:
            continue
        subsystem_cumulative_list.append(
            Lfm2ComponentAblationCandidate(
                candidate_id=(
                    "subsystem-cumulative--"
                    + "--".join(
                        subsystem_labels[kind]
                        for kind in subsystem_kinds[:prefix_length]
                    )
                ),
                mode="subsystem_cumulative",
                capsules=members,
            )
        )
        supplemental_path_sets.add(path_set)
    result = (
        *isolated,
        *subsystem_groups_list,
        *subsystem_cumulative_list,
        *cumulative,
    )
    candidate_ids = [value.candidate_id for value in result]
    candidate_path_sets = [
        frozenset(value.parent_module_path for value in candidate.capsules)
        for candidate in result
    ]
    if len(candidate_ids) != len(set(candidate_ids)) or len(candidate_path_sets) != len(
        set(candidate_path_sets)
    ):
        raise Lfm2ProgressiveComponentAblationError(
            "Ablation schedule candidates must have unique IDs and component-path sets."
        )
    return result


def schedule_receipt(
    candidates: Sequence[Lfm2ComponentAblationCandidate],
) -> Mapping[str, Any]:
    rows = [value.as_dict() for value in candidates]
    if not rows:
        raise Lfm2ProgressiveComponentAblationError(
            "A schedule receipt cannot be empty."
        )
    mode_counts = {
        mode: sum(value.mode == mode for value in candidates)
        for mode in (
            "isolated",
            "subsystem_group",
            "subsystem_cumulative",
            "cumulative",
        )
    }
    body = {
        "schema": LFM2_PROGRESSIVE_COMPONENT_SCHEDULE_SCHEMA,
        "status": "NON-FINAL-DIAGNOSTIC-ABLATION-SCHEDULE",
        "candidate_count": len(rows),
        "candidates": rows,
        "individual_isolated_candidates_preserved": True,
        "candidate_mode_counts": mode_counts,
        "topology_derived_grouping_capability_enabled": True,
        "topology_derived_grouping_applied": mode_counts["subsystem_group"] > 0,
        "cumulative_subsystem_grouping_applied": (
            mode_counts["subsystem_cumulative"] > 0
        ),
        "duplicate_component_path_sets_emitted": False,
        "ordered_component_prefixes_preserved": True,
        "schedule_order": [
            "individual-isolated",
            "whole-subsystem-groups",
            "cumulative-subsystem-groups",
            "ordered-component-prefixes",
        ],
        "whole_decoded_behavior_required_for_every_candidate": True,
        "automatic_acceptance_or_rejection_performed": False,
        "standalone_child_compression_claimed": False,
    }
    return MappingProxyType({**body, "schedule_sha256": canonical_sha256(body)})


def apply_lfm2_component_ablation_candidate(
    *,
    parent_model: Any,
    compact_model: Any,
    candidate: Lfm2ComponentAblationCandidate,
) -> tuple[Lfm2ProgressiveComponentInstallation, ...]:
    """Atomically install one component, subsystem group, or composition candidate.

    Each individual installer already rolls back its own path.  This outer
    transaction additionally owns every earlier path in the candidate:
    if a later component fails, all prior replacements are reverse-restored and
    exact whole-model state, topology, parameter accounting, training modes,
    and gradient flags are proved identical to the entry state before the
    original exception is re-raised.
    """

    baseline_state = _model_state_sha256(parent_model)
    baseline_parameters = _unique_parameter_elements(parent_model)
    baseline_parameter_bytes = _unique_parameter_bytes(parent_model)
    baseline_stored_elements = _stored_tensor_elements(parent_model)
    baseline_parameter_aliases = _parameter_alias_topology(parent_model)
    baseline_topology = _module_topology_identity(parent_model)
    baseline_runtime = _module_runtime_identity(parent_model)
    originals = {
        capsule.parent_module_path: _module_at(
            parent_model,
            capsule.parent_module_path,
        )
        for capsule in candidate.capsules
    }
    installations = []
    installed_paths: list[str] = []
    prior = None
    try:
        for step_index, capsule in enumerate(candidate.capsules):
            installation = install_lfm2_progressive_component(
                parent_model=parent_model,
                compact_model=compact_model,
                capsule=capsule,
                ablation_mode=candidate.mode,
                step_index=step_index,
                prior_installation_sha256=prior,
            )
            installations.append(installation)
            installed_paths.append(capsule.parent_module_path)
            prior = str(installation.receipt["installation_sha256"])
    except Exception as original_error:
        try:
            for path in reversed(installed_paths):
                _replace_module(parent_model, path, originals[path])
            restored_identities = all(
                _module_at(parent_model, path) is original
                for path, original in originals.items()
            )
            if (
                not restored_identities
                or _model_state_sha256(parent_model) != baseline_state
                or _unique_parameter_elements(parent_model) != baseline_parameters
                or _unique_parameter_bytes(parent_model) != baseline_parameter_bytes
                or _stored_tensor_elements(parent_model) != baseline_stored_elements
                or _parameter_alias_topology(parent_model) != baseline_parameter_aliases
                or _module_topology_identity(parent_model) != baseline_topology
                or _module_runtime_identity(parent_model) != baseline_runtime
            ):
                raise Lfm2ProgressiveComponentAblationError(
                    "Failed cumulative component candidate could not prove exact "
                    "whole-parent rollback."
                )
        except Exception as rollback_error:
            if rollback_error is original_error:
                raise
            raise Lfm2ProgressiveComponentAblationError(
                "Failed cumulative component candidate could not prove exact "
                "whole-parent rollback."
            ) from rollback_error
        raise
    if _parameter_alias_topology(parent_model) != baseline_parameter_aliases:
        raise Lfm2ProgressiveComponentAblationError(
            "Successful cumulative component candidate changed native parameter aliases."
        )
    return tuple(installations)


def _module_topology_identity(module: Any) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            str(name),
            type(value).__module__,
            type(value).__qualname__,
        )
        for name, value in module.named_modules()
    )


def _module_runtime_identity(
    module: Any,
) -> tuple[bool, tuple[tuple[str, bool], ...], tuple[tuple[str, bool], ...]]:
    return (
        bool(module.training),
        tuple(
            (str(name), bool(value.training)) for name, value in module.named_modules()
        ),
        tuple(
            (str(name), bool(parameter.requires_grad))
            for name, parameter in module.named_parameters()
        ),
    )


def _checkpoint_state(model: Any) -> dict[str, Any]:
    stored: dict[str, Any] = {}
    seen: set[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = set()
    for name, tensor in model.state_dict().items():
        identity = (
            int(tensor.untyped_storage().data_ptr()),
            int(tensor.storage_offset()),
            tuple(int(size) for size in tensor.shape),
            tuple(int(stride) for stride in tensor.stride()),
        )
        if identity in seen:
            continue
        seen.add(identity)
        stored[str(name)] = tensor.detach().to(device="cpu").contiguous()
    if not stored:
        raise Lfm2ProgressiveComponentAblationError(
            "Cannot persist an empty hybrid state."
        )
    return stored


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Lfm2ProgressiveComponentAblationError(
            f"Refusing to overwrite hybrid evidence: {path.name}."
        )
    path.write_bytes(canonical_json_bytes(dict(value)) + b"\n")


def _json_plain(value: Any) -> Any:
    """Normalize immutable mappings/tuples to their persisted JSON form."""

    if isinstance(value, Mapping):
        return {str(key): _json_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_plain(item) for item in value]
    return value


def _copy_local_tokenizer_package(source: Path, destination: Path) -> list[str]:
    try:
        from neural_forge.hf_tokenizer_bridge import (
            LOCAL_TOKENIZER_PACKAGE_FILENAMES,
        )
    except ImportError:  # pragma: no cover - package-local import
        LOCAL_TOKENIZER_PACKAGE_FILENAMES = (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
        )
    copied = []
    for relative in LOCAL_TOKENIZER_PACKAGE_FILENAMES:
        source_path = source / relative
        if not source_path.exists():
            continue
        if source_path.is_symlink() or not source_path.is_file():
            raise Lfm2ProgressiveComponentAblationError(
                f"Tokenizer package file is unsafe: {relative!r}."
            )
        target_path = destination / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)
        copied.append(relative)
    if "tokenizer.json" not in copied:
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid checkpoint requires a complete local tokenizer package."
        )
    return copied


def _artifact_inventory(
    directory: Path,
    *,
    exclude: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise Lfm2ProgressiveComponentAblationError(
                "Hybrid checkpoint contains a symlink."
            )
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in exclude:
            continue
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        )
    return rows


def _hybrid_temporary_prefix(target_name: str) -> str:
    if not isinstance(target_name, str) or not target_name:
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid checkpoint target name must be nonempty."
        )
    return (
        ".nf-hybrid-"
        + canonical_sha256(
            {
                "schema": "neural-forge-lfm2-progressive-hybrid-temp-v1",
                "target_name": target_name,
            }
        )[:20]
        + "-"
    )


def write_lfm2_progressive_hybrid_checkpoint(
    *,
    model: Any,
    destination: str | os.PathLike[str],
    tokenizer_source_directory: str | os.PathLike[str],
    candidate: Lfm2ComponentAblationCandidate,
    installations: Sequence[Lfm2ProgressiveComponentInstallation],
) -> Mapping[str, Any]:
    """Atomically persist one non-final hybrid for independent fresh reload."""

    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid persistence requires safetensors."
        ) from exc
    target = _ordinary_directory(destination, label="destination", must_exist=False)
    tokenizer_source = _ordinary_directory(
        tokenizer_source_directory,
        label="tokenizer source directory",
        must_exist=True,
    )
    installation_rows = [dict(value.receipt) for value in installations]
    reconstruction_rows = [dict(value.reconstruction) for value in installations]
    parameter_aliases_before_save = _parameter_alias_topology(model)
    parameter_alias_receipt = _parameter_alias_receipt(model)
    if (
        len(installation_rows) != len(candidate.capsules)
        or [row["component_id"] for row in installation_rows]
        != [value.component_id for value in candidate.capsules]
        or any(row.get("compact_config") is None for row in reconstruction_rows)
    ):
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid persistence requires the complete concrete LFM2 installation chain."
        )
    if any(
        row.get("physical_parameter_accounting", {}).get(
            "parameter_alias_topology_after"
        )
        != parameter_alias_receipt
        or row.get("physical_parameter_accounting", {}).get(
            "new_runtime_shared_parameters"
        )
        != 0
        or row.get("physical_parameter_accounting", {}).get(
            "native_parent_aliases_retained"
        )
        is not True
        or row.get("physical_parameter_accounting", {}).get(
            "parameter_reduction_attributed_to_sharing"
        )
        is not False
        for row in installation_rows
    ):
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid installations do not preserve the authenticated native alias topology."
        )
    config = getattr(model, "config", None)
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict) or getattr(config, "model_type", None) != "lfm2":
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid persistence supports concrete LFM2 models only."
        )
    parent = target.parent
    temporary_prefix = _hybrid_temporary_prefix(target.name)
    temporary = Path(tempfile.mkdtemp(prefix=temporary_prefix, dir=str(parent)))
    try:
        _write_json(temporary / "config.json", to_dict())
        generation = getattr(model, "generation_config", None)
        if generation is not None:
            generation_dict = getattr(generation, "to_diff_dict", None)
            if not callable(generation_dict):
                generation_dict = getattr(generation, "to_dict", None)
            if not callable(generation_dict):
                raise Lfm2ProgressiveComponentAblationError(
                    "Generation configuration is not serializable."
                )
            _write_json(
                temporary / "generation_config.json",
                generation_dict(),
            )
        _copy_local_tokenizer_package(tokenizer_source, temporary)
        save_file(
            _checkpoint_state(model),
            str(temporary / HYBRID_TENSOR_FILENAME),
            metadata={"format": "pt", "neural_forge_status": "NON-FINAL"},
        )
        if _parameter_alias_topology(model) != parameter_aliases_before_save:
            raise Lfm2ProgressiveComponentAblationError(
                "Hybrid save changed native parameter-alias topology."
            )
        parameter_dtype_names = sorted(
            {str(value.dtype).removeprefix("torch.") for value in model.parameters()}
        )
        body = _json_plain(
            {
                "schema": LFM2_PROGRESSIVE_HYBRID_MANIFEST_SCHEMA,
                "status": "NON-FINAL-DIAGNOSTIC-HYBRID-SAVED",
                "family_adapter": "lfm2-concrete-runtime-v1",
                "candidate": candidate.as_dict(),
                "candidate_id": candidate.candidate_id,
                "ablation_mode": candidate.mode,
                "component_ids": [value.component_id for value in candidate.capsules],
                "component_installations": installation_rows,
                "component_reconstruction": reconstruction_rows,
                "model_state_sha256": _model_state_sha256(model),
                "unique_parameter_elements": _unique_parameter_elements(model),
                "unique_parameter_bytes": _unique_parameter_bytes(model),
                "stored_tensor_elements": _stored_tensor_elements(model),
                "parameter_alias_topology": parameter_alias_receipt,
                "new_runtime_shared_parameters": 0,
                "native_parent_aliases_retained": True,
                "parameter_reduction_attributed_to_sharing": False,
                "parameter_dtype_names": parameter_dtype_names,
                "artifact_inventory": _artifact_inventory(temporary),
                "fresh_reload_required_before_behavior_review": True,
                "parent_shell_retained": True,
                "interface_bridges_retained_and_counted": True,
                "diagnostic_hybrid_only": True,
                "standalone_child_compression_claimed": False,
                "operation_audit": {
                    "training_steps": 0,
                    "optimizer_constructions": 0,
                    "optimizer_steps": 0,
                    "autograd_grad_calls": 0,
                    "backward_calls": 0,
                    "parent_trace_recaptures": 0,
                    "parameters_pruned": 0,
                    "parameters_quantized": 0,
                    "parameters_sparsified": 0,
                    "new_runtime_shared_parameters": 0,
                    "native_parent_aliases_retained": True,
                    "parameter_reduction_attributed_to_sharing": False,
                    "network_access": False,
                },
            }
        )
        manifest = {**body, "manifest_sha256": canonical_sha256(body)}
        _write_json(temporary / HYBRID_MANIFEST_FILENAME, manifest)
        if _parameter_alias_topology(model) != parameter_aliases_before_save:
            raise Lfm2ProgressiveComponentAblationError(
                "Hybrid publication changed native parameter-alias topology."
            )
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return MappingProxyType(manifest)


def _load_manifest(
    directory: Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    path = directory / HYBRID_MANIFEST_FILENAME
    if path.is_symlink() or not path.is_file():
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid manifest is missing or unsafe."
        )
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid manifest is not canonical JSON."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid manifest must be one JSON object."
        )
    body = dict(value)
    supplied = body.pop("manifest_sha256", None)
    if (
        value.get("schema") != LFM2_PROGRESSIVE_HYBRID_MANIFEST_SCHEMA
        or value.get("status") != "NON-FINAL-DIAGNOSTIC-HYBRID-SAVED"
        or supplied != canonical_sha256(body)
        or supplied
        != _digest(expected_manifest_sha256, label="expected hybrid manifest")
        or value.get("diagnostic_hybrid_only") is not True
        or value.get("standalone_child_compression_claimed") is not False
        or value.get("new_runtime_shared_parameters") != 0
        or value.get("native_parent_aliases_retained") is not True
        or value.get("parameter_reduction_attributed_to_sharing") is not False
    ):
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid manifest seal, status, or external pin is invalid."
        )
    _authenticated_parameter_alias_topology(
        value.get("parameter_alias_topology"),
        label="hybrid manifest parameter aliases",
    )
    expected_inventory = value.get("artifact_inventory")
    observed_inventory = _artifact_inventory(
        directory,
        exclude=frozenset({HYBRID_MANIFEST_FILENAME}),
    )
    if expected_inventory != observed_inventory:
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid artifact inventory differs from its sealed manifest."
        )
    return value


def _reconstruct_lfm2_wrapper(
    *,
    parent_model: Any,
    descriptor: Mapping[str, Any],
    dtype: Any,
    device: Any,
) -> None:
    torch = _torch()
    try:
        from transformers import Lfm2Config
        from transformers.models.lfm2.modeling_lfm2 import (
            Lfm2Attention,
            Lfm2MLP,
            Lfm2ShortConv,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Lfm2ProgressiveComponentAblationError(
            "Concrete Transformers LFM2 classes are required for hybrid reload."
        ) from exc
    if descriptor.get("family_adapter") != "lfm2-concrete-runtime-v1":
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid descriptor names an unsupported family adapter."
        )
    kind = descriptor.get("kind")
    if kind not in {"attention", "mlp", "short_conv", "terminal_readout"}:
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid descriptor names an unsupported component kind."
        )
    path = _module_path_text(
        descriptor.get("parent_module_path"),
        label="reconstructed parent module path",
    )
    compact_config_value = descriptor.get("compact_config")
    if not isinstance(compact_config_value, Mapping):
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid descriptor lacks its compact LFM2 config."
        )
    compact_config = Lfm2Config(**dict(compact_config_value))
    attention_implementation = descriptor.get("compact_attention_implementation")
    if attention_implementation is not None:
        if (
            not isinstance(attention_implementation, str)
            or not attention_implementation
        ):
            raise Lfm2ProgressiveComponentAblationError(
                "Hybrid descriptor has an invalid compact attention implementation."
            )
        compact_config._attn_implementation = attention_implementation
    layer_index = descriptor.get("parent_layer_index")
    if layer_index is not None and (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
    ):
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid descriptor has an invalid parent layer index."
        )
    compact_layer_index = descriptor.get("compact_layer_index")
    if compact_layer_index is not None and (
        isinstance(compact_layer_index, bool)
        or not isinstance(compact_layer_index, int)
        or compact_layer_index < 0
    ):
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid descriptor has an invalid compact layer index."
        )
    if kind == "attention":
        if layer_index is None or compact_layer_index is None:
            raise Lfm2ProgressiveComponentAblationError(
                "Attention reconstruction requires parent and compact layer indices."
            )
        operator = Lfm2Attention(compact_config, compact_layer_index)
        operator.layer_idx = layer_index
    elif kind == "mlp":
        operator = Lfm2MLP(compact_config)
    elif kind == "short_conv":
        if layer_index is None or compact_layer_index is None:
            raise Lfm2ProgressiveComponentAblationError(
                "Short-conv reconstruction requires parent and compact layer indices."
            )
        operator = Lfm2ShortConv(compact_config, compact_layer_index)
        operator.layer_idx = layer_index
    else:
        compact_geometry = descriptor.get("compact_geometry")
        if not isinstance(compact_geometry, Mapping):
            raise Lfm2ProgressiveComponentAblationError(
                "Readout reconstruction lacks compact geometry."
            )
        operator = torch.nn.Linear(
            int(compact_geometry["external_input_width"]),
            int(compact_geometry["external_output_width"]),
            bias=bool(compact_geometry.get("has_bias", False)),
        )
    operator.to(device=device, dtype=dtype)
    operator.requires_grad_(False)
    parent_geometry = descriptor.get("parent_geometry")
    compact_geometry = descriptor.get("compact_geometry")
    if not isinstance(parent_geometry, Mapping) or not isinstance(
        compact_geometry,
        Mapping,
    ):
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid descriptor lacks parent/compact geometry."
        )
    parent_input = int(parent_geometry["external_input_width"])
    parent_output = int(parent_geometry["external_output_width"])
    compact_input = int(compact_geometry["external_input_width"])
    compact_output = int(compact_geometry["external_output_width"])
    # Values are populated by strict state loading immediately after topology
    # reconstruction.  Zero initialization prevents accidental use beforehand.
    input_bridge = torch.nn.Linear(parent_input, compact_input, bias=False)
    input_bridge.to(device=device, dtype=dtype)
    input_bridge.requires_grad_(False)
    wrappers = _wrapper_classes()
    if kind == "terminal_readout":
        domain_shape = descriptor.get("output_domain_bridge_shape")
        domain_bridge = None
        if domain_shape is not None:
            if domain_shape != [compact_output, parent_output]:
                raise Lfm2ProgressiveComponentAblationError(
                    "Readout output-domain bridge descriptor has stale geometry."
                )
            domain_bridge = torch.nn.Linear(
                compact_output,
                parent_output,
                bias=False,
            )
            domain_bridge.to(device=device, dtype=dtype)
            domain_bridge.requires_grad_(False)
        wrapper = wrappers[kind](input_bridge, operator, domain_bridge)
    else:
        if descriptor.get("output_bridge_shape") != [
            compact_output,
            parent_output,
        ]:
            raise Lfm2ProgressiveComponentAblationError(
                "Residual output bridge descriptor has stale geometry."
            )
        output_bridge = torch.nn.Linear(compact_output, parent_output, bias=False)
        output_bridge.to(device=device, dtype=dtype)
        output_bridge.requires_grad_(False)
        wrapper = wrappers[kind](input_bridge, operator, output_bridge)
    wrapper.to(device=device, dtype=dtype)
    wrapper.requires_grad_(False)
    wrapper.eval()
    _replace_module(parent_model, path, wrapper)


def _copy_safetensor_state(model: Any, path: Path) -> None:
    torch = _torch()
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid reload requires safetensors."
        ) from exc
    target_state = model.state_dict()
    loaded: set[str] = set()
    unexpected = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        with torch.no_grad():
            for name in handle.keys():
                if name not in target_state:
                    unexpected.append(name)
                    continue
                source = handle.get_tensor(name)
                target = target_state[name]
                if tuple(source.shape) != tuple(target.shape):
                    raise Lfm2ProgressiveComponentAblationError(
                        f"Hybrid tensor shape mismatch for {name!r}."
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype))
                loaded.add(name)
    if unexpected:
        raise Lfm2ProgressiveComponentAblationError(
            f"Hybrid tensors contain unexpected state names: {unexpected!r}."
        )
    loaded_storage = {
        (
            int(target_state[name].untyped_storage().data_ptr()),
            int(target_state[name].storage_offset()),
            tuple(int(size) for size in target_state[name].shape),
        )
        for name in loaded
    }
    missing = sorted(
        name
        for name, tensor in target_state.items()
        if name not in loaded
        and (
            int(tensor.untyped_storage().data_ptr()),
            int(tensor.storage_offset()),
            tuple(int(size) for size in tensor.shape),
        )
        not in loaded_storage
    )
    if missing:
        raise Lfm2ProgressiveComponentAblationError(
            f"Hybrid tensors are missing non-aliased state names: {missing!r}."
        )


def load_lfm2_progressive_hybrid_checkpoint(
    source: str | os.PathLike[str],
    *,
    expected_manifest_sha256: str,
    device: Any,
    dtype: Any | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Freshly reconstruct and authenticate one persisted non-final hybrid."""

    torch = _torch()
    try:
        from transformers import GenerationConfig, Lfm2ForCausalLM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Lfm2ProgressiveComponentAblationError(
            "Concrete Transformers LFM2 classes are required for hybrid reload."
        ) from exc
    from neural_forge.hf_lfm2_direct_local import load_lfm2_direct_local_config

    directory = _ordinary_directory(source, label="hybrid checkpoint", must_exist=True)
    manifest = _load_manifest(
        directory,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    dtype_names = manifest.get("parameter_dtype_names")
    if (
        not isinstance(dtype_names, list)
        or len(dtype_names) != 1
        or not isinstance(dtype_names[0], str)
        or not hasattr(torch, dtype_names[0])
    ):
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid manifest must declare one concrete parameter dtype."
        )
    saved_dtype = getattr(torch, dtype_names[0])
    if dtype is not None and dtype != saved_dtype:
        raise Lfm2ProgressiveComponentAblationError(
            "Fresh reload dtype cannot disguise a quantization/materialization change."
        )
    resolved_device = torch.device(device)
    config = load_lfm2_direct_local_config(directory)
    model = Lfm2ForCausalLM(config)
    model.to(device=resolved_device, dtype=saved_dtype)
    expected_parameter_aliases = _authenticated_parameter_alias_topology(
        manifest.get("parameter_alias_topology"),
        label="hybrid manifest parameter aliases",
    )
    if _parameter_alias_topology(model) != expected_parameter_aliases:
        raise Lfm2ProgressiveComponentAblationError(
            "Fresh parent shell does not reproduce the native parameter aliases."
        )
    for descriptor in manifest.get("component_reconstruction", []):
        if not isinstance(descriptor, Mapping):
            raise Lfm2ProgressiveComponentAblationError(
                "Hybrid reconstruction descriptor is malformed."
            )
        _reconstruct_lfm2_wrapper(
            parent_model=model,
            descriptor=descriptor,
            dtype=saved_dtype,
            device=resolved_device,
        )
        if _parameter_alias_topology(model) != expected_parameter_aliases:
            raise Lfm2ProgressiveComponentAblationError(
                "Hybrid reconstruction changed native parameter-alias topology."
            )
    tensor_path = directory / HYBRID_TENSOR_FILENAME
    if tensor_path.is_symlink() or not tensor_path.is_file():
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid safetensors payload is missing or unsafe."
        )
    _copy_safetensor_state(model, tensor_path)
    if _parameter_alias_topology(model) != expected_parameter_aliases:
        raise Lfm2ProgressiveComponentAblationError(
            "Hybrid state reload changed native parameter-alias topology."
        )
    generation_path = directory / "generation_config.json"
    if generation_path.is_file() and not generation_path.is_symlink():
        generation = json.loads(generation_path.read_text(encoding="ascii"))
        if not isinstance(generation, Mapping):
            raise Lfm2ProgressiveComponentAblationError(
                "Hybrid generation config must be one JSON object."
            )
        model.generation_config = GenerationConfig(**dict(generation))
    model.requires_grad_(False)
    model.eval()
    if (
        _model_state_sha256(model) != manifest.get("model_state_sha256")
        or _unique_parameter_elements(model)
        != manifest.get("unique_parameter_elements")
        or _unique_parameter_bytes(model) != manifest.get("unique_parameter_bytes")
        or _stored_tensor_elements(model) != manifest.get("stored_tensor_elements")
        or _parameter_alias_topology(model) != expected_parameter_aliases
    ):
        raise Lfm2ProgressiveComponentAblationError(
            "Freshly reloaded hybrid differs from its sealed physical state."
        )
    return model, MappingProxyType(manifest)


__all__ = [
    "HYBRID_MANIFEST_FILENAME",
    "HYBRID_TENSOR_FILENAME",
    "LFM2_PROGRESSIVE_COMPONENT_CAPSULE_SCHEMA",
    "LFM2_PROGRESSIVE_COMPONENT_INSTALL_SCHEMA",
    "LFM2_PROGRESSIVE_COMPONENT_SCHEDULE_SCHEMA",
    "LFM2_PROGRESSIVE_HYBRID_MANIFEST_SCHEMA",
    "Lfm2ComponentAblationCandidate",
    "Lfm2ProgressiveComponentAblationError",
    "Lfm2ProgressiveComponentCapsule",
    "Lfm2ProgressiveComponentInstallation",
    "apply_lfm2_component_ablation_candidate",
    "build_lfm2_progressive_component_ablation_schedule",
    "install_lfm2_progressive_component",
    "load_lfm2_progressive_hybrid_checkpoint",
    "schedule_receipt",
    "write_lfm2_progressive_hybrid_checkpoint",
]
