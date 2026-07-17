"""Install a packed CREVF runtime into an already authenticated LFM2 skeleton.

This module deliberately owns no exporter, model loader, or quantization
policy.  A checkpoint loader first constructs the model on ``meta`` from a
sealed local configuration, authenticates its logical-to-canonical parameter
aliases, decodes the CREVF artifact into :class:`CrevfPackedRuntime`, and then
calls :func:`install_lfm2_crevf_runtime`.

Every dense learned module is replaced by a wrapper that materializes its
BF16 tensor only for the operation that consumes it.  The packed stores are
registered exactly once under the runtime registry.  Wrappers keep
non-registering references to those stores so tied embedding/head storage
remains one physical record.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .hf_lfm2_causal_residual_entropy_value_fold import (
    CrevfPackedRuntime,
    CrevfPackedTensor,
)


_REGISTRY_NAME = "neural_forge_crevf_runtime"
_SENTINEL_NAME = "neural_forge_device_sentinel"


class Lfm2CrevfRuntimeInstallError(ValueError):
    """The authenticated skeleton and packed state cannot be installed."""


@dataclass(frozen=True, slots=True)
class Lfm2CrevfInstallReport:
    canonical_tensor_count: int
    logical_parameter_count: int
    replaced_linear_count: int
    replaced_embedding_count: int
    replaced_conv1d_count: int
    replaced_norm_count: int
    persistent_encoded_bytes: int
    decoded_full_state_bytes: int
    maximum_tensor_decode_transient_bytes: int
    maximum_layer_decode_transient_bytes: int
    tied_vocabulary_stored_once: bool
    persistent_dense_parameter_elements: int
    zero_element_device_sentinel: bool


class CrevfPackedLinear(nn.Module):
    """Linear operation backed by transient CREVF dequantization."""

    def __init__(
        self,
        weight: CrevfPackedTensor,
        bias: CrevfPackedTensor | None,
    ) -> None:
        super().__init__()
        if len(weight.shape) != 2:
            raise Lfm2CrevfRuntimeInstallError(
                "A CREVF linear weight must be two-dimensional."
            )
        if bias is not None and bias.shape != (weight.shape[0],):
            raise Lfm2CrevfRuntimeInstallError(
                "A CREVF linear bias shape differs from its output width."
            )
        object.__setattr__(self, "_packed_weight", weight)
        object.__setattr__(self, "_packed_bias", bias)
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1]

    @property
    def weight(self) -> Tensor:
        return self._packed_weight.dequantize()

    @property
    def bias(self) -> Tensor | None:
        return None if self._packed_bias is None else self._packed_bias.dequantize()

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs, self.weight, self.bias)


class CrevfPackedEmbedding(nn.Module):
    """Embedding operation backed by one canonical CREVF store."""

    def __init__(
        self,
        weight: CrevfPackedTensor,
        *,
        padding_idx: int | None,
        max_norm: float | None,
        norm_type: float,
        scale_grad_by_freq: bool,
        sparse: bool,
    ) -> None:
        super().__init__()
        if len(weight.shape) != 2:
            raise Lfm2CrevfRuntimeInstallError(
                "A CREVF embedding weight must be two-dimensional."
            )
        object.__setattr__(self, "_packed_weight", weight)
        self.num_embeddings = weight.shape[0]
        self.embedding_dim = weight.shape[1]
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = float(norm_type)
        self.scale_grad_by_freq = bool(scale_grad_by_freq)
        self.sparse = bool(sparse)

    @property
    def weight(self) -> Tensor:
        return self._packed_weight.dequantize()

    def forward(self, input_ids: Tensor) -> Tensor:
        return F.embedding(
            input_ids,
            self.weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )


class CrevfPackedConv1d(nn.Module):
    """Conv1d operation backed by transient CREVF dequantization."""

    def __init__(
        self,
        weight: CrevfPackedTensor,
        bias: CrevfPackedTensor | None,
        source: nn.Conv1d,
    ) -> None:
        super().__init__()
        expected = (
            source.out_channels,
            source.in_channels // source.groups,
            source.kernel_size[0],
        )
        if weight.shape != expected:
            raise Lfm2CrevfRuntimeInstallError(
                "A CREVF Conv1d weight shape differs from its module geometry."
            )
        if bias is not None and bias.shape != (source.out_channels,):
            raise Lfm2CrevfRuntimeInstallError(
                "A CREVF Conv1d bias shape differs from its output width."
            )
        object.__setattr__(self, "_packed_weight", weight)
        object.__setattr__(self, "_packed_bias", bias)
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.padding_mode = source.padding_mode

    @property
    def weight(self) -> Tensor:
        return self._packed_weight.dequantize()

    @property
    def bias(self) -> Tensor | None:
        return None if self._packed_bias is None else self._packed_bias.dequantize()

    def forward(self, inputs: Tensor) -> Tensor:
        if self.padding_mode != "zeros":
            raise Lfm2CrevfRuntimeInstallError(
                "Packed CREVF Conv1d supports only zero padding."
            )
        return F.conv1d(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class CrevfPackedLfm2RMSNorm(nn.Module):
    """LFM2 RMSNorm backed by an exact CREVF control tensor."""

    def __init__(
        self,
        weight: CrevfPackedTensor,
        *,
        variance_epsilon: float,
    ) -> None:
        super().__init__()
        if len(weight.shape) != 1 or not weight.exact_only:
            raise Lfm2CrevfRuntimeInstallError(
                "A packed LFM2 norm must use one exact-only CREVF vector."
            )
        object.__setattr__(self, "_packed_weight", weight)
        self.variance_epsilon = float(variance_epsilon)

    @property
    def weight(self) -> Tensor:
        return self._packed_weight.dequantize()

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.to(torch.float32)
        variance = normalized.square().mean(-1, keepdim=True)
        normalized *= torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


def _resolve_owner(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    owner_path, separator, child_name = path.rpartition(".")
    if not separator:
        return model, path
    owner: nn.Module = model
    try:
        for component in owner_path.split("."):
            owner = getattr(owner, component)
    except (AttributeError, KeyError, IndexError) as exc:
        raise Lfm2CrevfRuntimeInstallError(
            f"Module path {path!r} does not exist in the authenticated skeleton."
        ) from exc
    return owner, child_name


def _store_for_logical(
    stores: Mapping[str, CrevfPackedTensor],
    aliases: Mapping[str, str],
    logical_name: str,
) -> CrevfPackedTensor:
    try:
        canonical = aliases[logical_name]
        return stores[canonical]
    except KeyError as exc:
        raise Lfm2CrevfRuntimeInstallError(
            f"Logical tensor {logical_name!r} lacks a canonical CREVF store."
        ) from exc


def _validate_aliases(
    model: nn.Module,
    runtime: CrevfPackedRuntime,
    aliases: Mapping[str, str],
) -> tuple[dict[str, nn.Parameter], dict[str, CrevfPackedTensor]]:
    try:
        logical = dict(model.named_parameters(recurse=True, remove_duplicate=False))
    except TypeError as exc:  # pragma: no cover - supported torch is required
        raise Lfm2CrevfRuntimeInstallError(
            "PyTorch must support remove_duplicate=False."
        ) from exc
    if not logical or set(aliases) != set(logical):
        raise Lfm2CrevfRuntimeInstallError(
            "The logical alias inventory differs from the skeleton Parameters."
        )
    canonical_names = set(aliases.values())
    if canonical_names != set(runtime.tensor_names):
        raise Lfm2CrevfRuntimeInstallError(
            "The canonical alias inventory differs from the CREVF artifact."
        )
    stores = {name: runtime.tensor(name) for name in runtime.tensor_names}
    identity_to_canonical: dict[int, str] = {}
    canonical_to_identity: dict[str, int] = {}
    for logical_name, parameter in logical.items():
        canonical = aliases[logical_name]
        store = stores[canonical]
        if tuple(parameter.shape) != store.shape:
            raise Lfm2CrevfRuntimeInstallError(
                f"CREVF tensor {canonical!r} differs from skeleton shape."
            )
        prior = identity_to_canonical.setdefault(id(parameter), canonical)
        if prior != canonical:
            raise Lfm2CrevfRuntimeInstallError(
                "One tied skeleton Parameter maps to multiple CREVF records."
            )
        prior_identity = canonical_to_identity.setdefault(canonical, id(parameter))
        if prior_identity != id(parameter):
            raise Lfm2CrevfRuntimeInstallError(
                "One canonical CREVF record maps to multiple untied Parameters."
            )
    return logical, stores


def install_lfm2_crevf_runtime(
    model: nn.Module,
    runtime: CrevfPackedRuntime,
    logical_to_canonical: Mapping[str, str],
    *,
    tied_embedding_name: str = "model.embed_tokens.weight",
    tied_head_name: str = "lm_head.weight",
) -> Lfm2CrevfInstallReport:
    """Replace all learned LFM2 modules with one authenticated packed runtime."""

    if not isinstance(model, nn.Module) or not isinstance(runtime, CrevfPackedRuntime):
        raise Lfm2CrevfRuntimeInstallError(
            "A torch model and CrevfPackedRuntime are required."
        )
    if hasattr(model, _REGISTRY_NAME) or hasattr(model, _SENTINEL_NAME):
        raise Lfm2CrevfRuntimeInstallError(
            "The model already owns a CREVF runtime or device sentinel."
        )
    aliases = dict(logical_to_canonical)
    logical, stores = _validate_aliases(model, runtime, aliases)
    if (
        tied_embedding_name not in logical
        or tied_head_name not in logical
        or logical[tied_embedding_name] is not logical[tied_head_name]
        or aliases[tied_embedding_name] != aliases[tied_head_name]
    ):
        raise Lfm2CrevfRuntimeInstallError(
            "The authenticated input embedding and output head are not tied."
        )

    counts = {
        "linear": 0,
        "embedding": 0,
        "conv1d": 0,
        "norm": 0,
    }
    covered_logical_names: set[str] = set()
    replacements: list[tuple[nn.Module, str, nn.Module, nn.Module]] = []
    original_modules = tuple(model.named_modules())
    for module_name, module in reversed(original_modules):
        if not module_name:
            continue
        replacement: nn.Module | None = None
        if isinstance(module, nn.Linear):
            weight_name = f"{module_name}.weight"
            weight = _store_for_logical(
                stores,
                aliases,
                weight_name,
            )
            covered_logical_names.add(weight_name)
            bias = (
                _store_for_logical(
                    stores,
                    aliases,
                    f"{module_name}.bias",
                )
                if module.bias is not None
                else None
            )
            if module.bias is not None:
                covered_logical_names.add(f"{module_name}.bias")
            replacement = CrevfPackedLinear(weight, bias)
            counts["linear"] += 1
        elif isinstance(module, nn.Embedding):
            weight_name = f"{module_name}.weight"
            weight = _store_for_logical(
                stores,
                aliases,
                weight_name,
            )
            covered_logical_names.add(weight_name)
            replacement = CrevfPackedEmbedding(
                weight,
                padding_idx=module.padding_idx,
                max_norm=module.max_norm,
                norm_type=module.norm_type,
                scale_grad_by_freq=module.scale_grad_by_freq,
                sparse=module.sparse,
            )
            counts["embedding"] += 1
        elif isinstance(module, nn.Conv1d):
            weight_name = f"{module_name}.weight"
            weight = _store_for_logical(
                stores,
                aliases,
                weight_name,
            )
            covered_logical_names.add(weight_name)
            bias = (
                _store_for_logical(
                    stores,
                    aliases,
                    f"{module_name}.bias",
                )
                if module.bias is not None
                else None
            )
            if module.bias is not None:
                covered_logical_names.add(f"{module_name}.bias")
            replacement = CrevfPackedConv1d(weight, bias, module)
            counts["conv1d"] += 1
        elif type(module).__name__ == "Lfm2RMSNorm":
            weight_name = f"{module_name}.weight"
            weight = _store_for_logical(
                stores,
                aliases,
                weight_name,
            )
            covered_logical_names.add(weight_name)
            epsilon = getattr(module, "variance_epsilon", None)
            if not isinstance(epsilon, float):
                raise Lfm2CrevfRuntimeInstallError(
                    f"LFM2 norm {module_name!r} lacks variance_epsilon."
                )
            replacement = CrevfPackedLfm2RMSNorm(
                weight,
                variance_epsilon=epsilon,
            )
            counts["norm"] += 1
        if replacement is not None:
            owner, child_name = _resolve_owner(model, module_name)
            replacements.append((owner, child_name, module, replacement))

    unsupported = sorted(set(logical).difference(covered_logical_names))
    if unsupported:
        names = ", ".join(unsupported[:8])
        raise Lfm2CrevfRuntimeInstallError(
            "The LFM2 skeleton contains Parameters outside supported packed "
            "modules: "
            f"{names}."
        )
    devices = {store.device for store in stores.values()}
    if len(devices) != 1:
        raise Lfm2CrevfRuntimeInstallError(
            "All canonical CREVF stores must share one runtime device."
        )
    target_device = next(iter(devices))
    applied: list[tuple[nn.Module, str, nn.Module]] = []
    registry_added = False
    sentinel_added = False
    original_training = model.training
    try:
        for owner, child_name, original, replacement in replacements:
            setattr(owner, child_name, replacement)
            applied.append((owner, child_name, original))
        remaining = tuple(model.named_parameters(recurse=True))
        if remaining:
            names = ", ".join(name for name, _parameter in remaining[:8])
            raise Lfm2CrevfRuntimeInstallError(
                "Dense learned Parameters remain after CREVF installation: "
                f"{names}."
            )
        model.add_module(_REGISTRY_NAME, runtime)
        registry_added = True
        model.register_parameter(
            _SENTINEL_NAME,
            nn.Parameter(
                torch.empty(0, dtype=torch.bfloat16, device=target_device),
                requires_grad=False,
            ),
        )
        sentinel_added = True

        embedding_owner, embedding_child = _resolve_owner(
            model,
            tied_embedding_name.removesuffix(".weight"),
        )
        head_owner, head_child = _resolve_owner(
            model,
            tied_head_name.removesuffix(".weight"),
        )
        embedding_module = getattr(embedding_owner, embedding_child)
        head_module = getattr(head_owner, head_child)
        tied_store = stores[aliases[tied_embedding_name]]
        tied_once = (
            getattr(embedding_module, "_packed_weight", None) is tied_store
            and getattr(head_module, "_packed_weight", None) is tied_store
        )
        if not tied_once:
            raise Lfm2CrevfRuntimeInstallError(
                "The installed embedding and output head do not share one store."
            )
        parameter_elements = sum(parameter.numel() for parameter in model.parameters())
        if parameter_elements:
            raise Lfm2CrevfRuntimeInstallError(
                "The installed CREVF model retained nonempty Parameters."
            )
        report = Lfm2CrevfInstallReport(
            canonical_tensor_count=len(stores),
            logical_parameter_count=len(logical),
            replaced_linear_count=counts["linear"],
            replaced_embedding_count=counts["embedding"],
            replaced_conv1d_count=counts["conv1d"],
            replaced_norm_count=counts["norm"],
            persistent_encoded_bytes=runtime.persistent_encoded_bytes,
            decoded_full_state_bytes=runtime.decoded_full_state_bytes,
            maximum_tensor_decode_transient_bytes=(
                runtime.maximum_tensor_decode_transient_bytes
            ),
            maximum_layer_decode_transient_bytes=(
                runtime.maximum_layer_decode_transient_bytes
            ),
            tied_vocabulary_stored_once=True,
            persistent_dense_parameter_elements=0,
            zero_element_device_sentinel=True,
        )
        model.eval()
        setattr(model, "neural_forge_crevf_install_report", report)
        return report
    except BaseException:
        if hasattr(model, "neural_forge_crevf_install_report"):
            delattr(model, "neural_forge_crevf_install_report")
        if sentinel_added:
            model._parameters.pop(_SENTINEL_NAME, None)
        if registry_added:
            model._modules.pop(_REGISTRY_NAME, None)
        for owner, child_name, original in reversed(applied):
            setattr(owner, child_name, original)
        model.train(original_training)
        raise


__all__ = [
    "CrevfPackedConv1d",
    "CrevfPackedEmbedding",
    "CrevfPackedLfm2RMSNorm",
    "CrevfPackedLinear",
    "Lfm2CrevfInstallReport",
    "Lfm2CrevfRuntimeInstallError",
    "install_lfm2_crevf_runtime",
]
