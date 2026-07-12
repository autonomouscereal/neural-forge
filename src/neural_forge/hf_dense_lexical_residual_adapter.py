"""Bakeable target-native residuals for an untied lexical boundary.

The temporary operator corrects the two ordinary dense matrices at a causal
language model's lexical boundary::

    embedding[token] += embedding_left[token] @ embedding_right
    logits           += lm_head_left @ (lm_head_right @ hidden)

Both ``left`` factors start at exact zero while the corresponding ``right``
factors use deterministic nonzero initialization.  Installation is therefore
an exact functional identity.  All four factors are FP32 even when the dense
child is FP16/BF16, and adapter-only output training freezes every native
child parameter.

The adapter owns no parent model, source-width basis, activation cache, or
publication sidecar.  A successful bake adds the two materialized low-rank
deltas to fresh native ``nn.Embedding`` and ``nn.Linear`` modules, removes all
adapter scaffolding, and restores the child's exact original parameter count
and state-key topology.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


LexicalRole = Literal["embedding", "lm_head"]

DENSE_LEXICAL_RESIDUAL_ACCOUNTING_SCHEMA = (
    "neural-forge-dense-lexical-residual-accounting-v1"
)
DENSE_LEXICAL_RESIDUAL_REGISTRY_SCHEMA = (
    "neural-forge-dense-lexical-residual-registry-v1"
)
DENSE_LEXICAL_RESIDUAL_INSTALLATION_SCHEMA = (
    "neural-forge-dense-lexical-residual-installation-v1"
)
DENSE_LEXICAL_RESIDUAL_TRAINING_SCOPE_SCHEMA = (
    "neural-forge-dense-lexical-factor-only-output-training-v1"
)
DENSE_LEXICAL_RESIDUAL_BAKE_SCHEMA = "neural-forge-dense-lexical-residual-bake-v1"


class DenseLexicalResidualError(RuntimeError):
    """A target-native lexical residual transaction is invalid."""


class DenseLexicalResidualTopologyError(DenseLexicalResidualError):
    """The native lexical modules or transaction ownership changed."""


class DenseLexicalResidualAccountingError(DenseLexicalResidualError):
    """Closed-form and live lexical parameter accounting disagree."""


class DenseLexicalResidualTrainingError(DenseLexicalResidualError):
    """The factor-only output-training boundary was violated."""


def _index(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise DenseLexicalResidualTopologyError(f"{name} must be an integer.")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise DenseLexicalResidualTopologyError(
            f"{name} must be an integer."
        ) from error
    if (positive and result <= 0) or (not positive and result < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise DenseLexicalResidualTopologyError(
            f"{name} must be a {qualifier} integer."
        )
    return int(result)


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise DenseLexicalResidualError(
            "Lexical residual provenance must be finite canonical JSON."
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise DenseLexicalResidualTopologyError("Expected a tensor to hash.")
    contiguous = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(
        header + b"\0" + contiguous.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _unique_parameter_count(module: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in module.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += int(parameter.numel())
    return total


def _state_key_sha256(keys: tuple[str, ...]) -> str:
    return _canonical_sha256(list(keys))


@dataclass(frozen=True, slots=True)
class DenseLexicalResidualAccounting:
    """Exact transient and published accounting for both lexical matrices."""

    vocab_size: int
    hidden_size: int
    rank: int
    factor_element_size: int
    embedding_left_parameters: int
    embedding_right_parameters: int
    lm_head_left_parameters: int
    lm_head_right_parameters: int
    embedding_adapter_parameters: int
    lm_head_adapter_parameters: int
    adapter_parameters: int
    installed_parameter_delta: int
    baked_publication_parameter_delta: int
    installed_factor_tensor_count: int
    installed_factor_payload_bytes: int
    baked_publication_adapter_tensor_count: int
    baked_publication_adapter_payload_bytes: int
    embedding_residual_multiply_accumulates_per_token: int
    lm_head_residual_multiply_accumulates_per_token: int
    total_residual_multiply_accumulates_per_token: int

    @classmethod
    def build(
        cls,
        *,
        vocab_size: int,
        hidden_size: int,
        rank: int,
    ) -> DenseLexicalResidualAccounting:
        vocab = _index(vocab_size, "vocab_size", positive=True)
        hidden = _index(hidden_size, "hidden_size", positive=True)
        rank_value = _index(rank, "rank", positive=True)
        left = vocab * rank_value
        right = rank_value * hidden
        per_matrix = left + right
        total = 2 * per_matrix
        embedding_macs = rank_value * hidden
        head_macs = rank_value * (hidden + vocab)
        return cls(
            vocab_size=vocab,
            hidden_size=hidden,
            rank=rank_value,
            factor_element_size=torch.tensor([], dtype=torch.float32).element_size(),
            embedding_left_parameters=left,
            embedding_right_parameters=right,
            lm_head_left_parameters=left,
            lm_head_right_parameters=right,
            embedding_adapter_parameters=per_matrix,
            lm_head_adapter_parameters=per_matrix,
            adapter_parameters=total,
            installed_parameter_delta=total,
            baked_publication_parameter_delta=0,
            installed_factor_tensor_count=4,
            installed_factor_payload_bytes=total * 4,
            baked_publication_adapter_tensor_count=0,
            baked_publication_adapter_payload_bytes=0,
            embedding_residual_multiply_accumulates_per_token=embedding_macs,
            lm_head_residual_multiply_accumulates_per_token=head_macs,
            total_residual_multiply_accumulates_per_token=(embedding_macs + head_macs),
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": DENSE_LEXICAL_RESIDUAL_ACCOUNTING_SCHEMA,
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
            "factor_dtype": "torch.float32",
            "formula": "2*rank*(vocab_size+hidden_size)",
            "embedding_bake_rule": (
                "embedding.weight += embedding_left @ embedding_right"
            ),
            "lm_head_bake_rule": "lm_head.weight += lm_head_left @ lm_head_right",
            "published_adapter_parameter_count": 0,
            "published_dense_parameter_topology_unchanged": True,
            "parent_runtime_reference_count": 0,
            "source_sized_basis_parameters": 0,
            "source_sized_sidecar_retained": False,
        }
        payload["accounting_sha256"] = _canonical_sha256(payload)
        return payload


class DenseLexicalResidualRegistry(nn.Module):
    """Four independent FP32 low-rank factors for the lexical boundary."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        rank: int,
        embedding_device: torch.device | str,
        lm_head_device: torch.device | str,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = _index(vocab_size, "vocab_size", positive=True)
        self.hidden_size = _index(hidden_size, "hidden_size", positive=True)
        self.rank = _index(rank, "rank", positive=True)
        seed = _index(initialization_seed, "initialization_seed")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        def nonzero_right(device: torch.device | str) -> Tensor:
            value = torch.empty(
                (self.rank, self.hidden_size),
                dtype=torch.float32,
                device="cpu",
            )
            value.uniform_(
                -1.0 / math.sqrt(self.hidden_size),
                1.0 / math.sqrt(self.hidden_size),
                generator=generator,
            )
            return value.to(device=device)

        self.embedding_left = nn.Parameter(
            torch.zeros(
                (self.vocab_size, self.rank),
                dtype=torch.float32,
                device=embedding_device,
            )
        )
        self.embedding_right = nn.Parameter(nonzero_right(embedding_device))
        self.lm_head_left = nn.Parameter(
            torch.zeros(
                (self.vocab_size, self.rank),
                dtype=torch.float32,
                device=lm_head_device,
            )
        )
        self.lm_head_right = nn.Parameter(nonzero_right(lm_head_device))
        self.assert_finite()

    @property
    def learned_parameter_count(self) -> int:
        return sum(int(value.numel()) for value in self.parameters())

    @property
    def starts_as_exact_zero_correction(self) -> bool:
        return (
            int(torch.count_nonzero(self.embedding_left.detach()).item()) == 0
            and int(torch.count_nonzero(self.lm_head_left.detach()).item()) == 0
        )

    @property
    def input_factors_are_nonzero(self) -> bool:
        return (
            int(torch.count_nonzero(self.embedding_right.detach()).item()) > 0
            and int(torch.count_nonzero(self.lm_head_right.detach()).item()) > 0
        )

    def output_trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return (
            self.embedding_left,
            self.embedding_right,
            self.lm_head_left,
            self.lm_head_right,
        )

    def assert_finite(self) -> None:
        for name, value in self.named_parameters():
            if not bool(torch.isfinite(value.detach()).all()):
                raise DenseLexicalResidualError(
                    f"Lexical residual factor {name!r} contains NaN or infinity."
                )

    def embedding_correction(
        self,
        input_ids: Tensor,
        *,
        padding_idx: int | None,
    ) -> Tensor:
        self.assert_finite()
        selected = F.embedding(
            input_ids,
            self.embedding_left,
            padding_idx=padding_idx,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
        )
        result = selected @ self.embedding_right
        if not bool(torch.isfinite(result.detach()).all()):
            raise DenseLexicalResidualError(
                "Embedding residual correction became non-finite."
            )
        return result

    def lm_head_correction(self, hidden_states: Tensor) -> Tensor:
        self.assert_finite()
        projected = F.linear(hidden_states.to(dtype=torch.float32), self.lm_head_right)
        result = F.linear(projected, self.lm_head_left)
        if not bool(torch.isfinite(result.detach()).all()):
            raise DenseLexicalResidualError(
                "LM-head residual correction became non-finite."
            )
        return result

    def dense_delta(self, role: LexicalRole) -> Tensor:
        if role == "embedding":
            result = self.embedding_left @ self.embedding_right
        elif role == "lm_head":
            result = self.lm_head_left @ self.lm_head_right
        else:
            raise DenseLexicalResidualTopologyError(
                "Lexical role must be embedding or lm_head."
            )
        if not bool(torch.isfinite(result.detach()).all()):
            raise DenseLexicalResidualError(
                f"Materialized {role} residual correction is non-finite."
            )
        return result

    def as_dict(self) -> dict[str, Any]:
        manifest = [
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "sha256": _tensor_sha256(value),
            }
            for name, value in self.named_parameters()
        ]
        payload = {
            "schema": DENSE_LEXICAL_RESIDUAL_REGISTRY_SCHEMA,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "rank": self.rank,
            "factor_dtype": "torch.float32",
            "learned_parameter_count": self.learned_parameter_count,
            "factor_manifest": manifest,
            "exact_zero_output_factor_initialization": (
                self.starts_as_exact_zero_correction
            ),
            "input_factors_nonzero": self.input_factors_are_nonzero,
            "parent_runtime_reference_count": 0,
            "source_tensor_reference_count": 0,
            "activation_cache_reference_count": 0,
        }
        payload["registry_sha256"] = _canonical_sha256(payload)
        return payload


class DenseLexicalResidualEmbedding(nn.Module):
    """An ``nn.Embedding``-compatible view plus one low-rank correction."""

    def __init__(
        self,
        base_embedding: nn.Embedding,
        registry: DenseLexicalResidualRegistry,
    ) -> None:
        super().__init__()
        if type(base_embedding) is not nn.Embedding:
            raise DenseLexicalResidualTopologyError(
                "base_embedding must be a plain torch.nn.Embedding."
            )
        object.__setattr__(self, "_registry", registry)
        self.base_embedding = base_embedding
        self.num_embeddings = base_embedding.num_embeddings
        self.embedding_dim = base_embedding.embedding_dim
        self.padding_idx = base_embedding.padding_idx
        self.max_norm = base_embedding.max_norm
        self.norm_type = base_embedding.norm_type
        self.scale_grad_by_freq = base_embedding.scale_grad_by_freq
        self.sparse = base_embedding.sparse
        self.train(base_embedding.training)

    @property
    def registry(self) -> DenseLexicalResidualRegistry:
        return self._registry

    @property
    def weight(self) -> Tensor:
        return self.base_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        base = self.base_embedding(input_ids)
        correction = self.registry.embedding_correction(
            input_ids, padding_idx=self.padding_idx
        ).to(dtype=base.dtype)
        if tuple(correction.shape) != tuple(base.shape):
            raise DenseLexicalResidualTopologyError(
                "Embedding residual output shape does not match the native embedding."
            )
        return base + correction


class DenseLexicalResidualLMHead(nn.Module):
    """An ``nn.Linear``-compatible LM head plus one low-rank correction."""

    def __init__(
        self,
        base_lm_head: nn.Linear,
        registry: DenseLexicalResidualRegistry,
    ) -> None:
        super().__init__()
        if type(base_lm_head) is not nn.Linear:
            raise DenseLexicalResidualTopologyError(
                "base_lm_head must be a plain torch.nn.Linear."
            )
        object.__setattr__(self, "_registry", registry)
        self.base_lm_head = base_lm_head
        self.in_features = base_lm_head.in_features
        self.out_features = base_lm_head.out_features
        self.train(base_lm_head.training)

    @property
    def registry(self) -> DenseLexicalResidualRegistry:
        return self._registry

    @property
    def weight(self) -> Tensor:
        return self.base_lm_head.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base_lm_head.bias

    def forward(self, hidden_states: Tensor) -> Tensor:
        base = self.base_lm_head(hidden_states)
        correction = self.registry.lm_head_correction(hidden_states).to(
            dtype=base.dtype
        )
        if tuple(correction.shape) != tuple(base.shape):
            raise DenseLexicalResidualTopologyError(
                "LM-head residual output shape does not match the native head."
            )
        return base + correction


@dataclass(frozen=True, slots=True)
class DenseLexicalResidualInstallationReport:
    vocab_size: int
    hidden_size: int
    rank: int
    factor_dtype: str
    owner_name: str
    embedding_path: str
    lm_head_path: str
    adapter_parameter_count: int
    model_parameter_count_before: int
    model_parameter_count_installed: int
    original_state_key_count: int
    original_state_key_sha256: str
    base_lexical_state_sha256: str
    accounting: DenseLexicalResidualAccounting

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": DENSE_LEXICAL_RESIDUAL_INSTALLATION_SCHEMA,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "rank": self.rank,
            "factor_dtype": self.factor_dtype,
            "owner_name": self.owner_name,
            "embedding_path": self.embedding_path,
            "lm_head_path": self.lm_head_path,
            "adapter_parameter_count": self.adapter_parameter_count,
            "model_parameter_count_before": self.model_parameter_count_before,
            "model_parameter_count_installed": self.model_parameter_count_installed,
            "original_state_key_count": self.original_state_key_count,
            "original_state_key_sha256": self.original_state_key_sha256,
            "base_lexical_state_sha256": self.base_lexical_state_sha256,
            "accounting": self.accounting.as_dict(),
            "untied_embedding_and_lm_head_required": True,
            "zero_step_identity": True,
            "factor_only_output_training": True,
            "native_dense_bake_supported": True,
            "checkpoint_publication_allowed_while_installed": False,
            "parent_runtime_reference_count": 0,
            "source_tensor_reference_count": 0,
            "sidecar_reference_count": 0,
            "internal_numeric_matching_used": False,
        }
        payload["installation_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class DenseLexicalResidualBakeReport:
    vocab_size: int
    hidden_size: int
    rank: int
    adapter_parameters_removed: int
    model_parameter_count_before_installation: int
    model_parameter_count_before_bake: int
    model_parameter_count_after_bake: int
    original_state_key_sha256: str
    published_state_key_sha256: str
    dense_delta_sha256: str
    affected_dense_matrix_count: int

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": DENSE_LEXICAL_RESIDUAL_BAKE_SCHEMA,
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
            "published_embedding_type": "torch.nn.Embedding",
            "published_lm_head_type": "torch.nn.Linear",
            "published_adapter_parameter_count": 0,
            "published_parameter_delta_vs_original_dense_target": 0,
            "published_state_key_topology_exact": (
                self.original_state_key_sha256 == self.published_state_key_sha256
            ),
            "source_sized_sidecar_retained": False,
            "parent_runtime_reference_count": 0,
            "functional_publication_gate_still_required": True,
            "internal_numeric_matching_used": False,
        }
        payload["bake_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class _TensorSnapshot:
    tensor: Tensor
    data_pointer: int
    version: int
    sha256: str


def _lexical_state(
    embedding: nn.Embedding,
    lm_head: nn.Linear,
) -> dict[str, _TensorSnapshot]:
    tensors: dict[str, Tensor] = {
        "embedding.weight": embedding.weight,
        "lm_head.weight": lm_head.weight,
    }
    if lm_head.bias is not None:
        tensors["lm_head.bias"] = lm_head.bias
    return {
        name: _TensorSnapshot(
            tensor=value,
            data_pointer=int(value.data_ptr()),
            version=int(value._version),
            sha256=_tensor_sha256(value),
        )
        for name, value in tensors.items()
    }


def _lexical_state_sha256(state: dict[str, _TensorSnapshot]) -> str:
    return _canonical_sha256(
        [
            {"name": name, "sha256": snapshot.sha256}
            for name, snapshot in sorted(state.items())
        ]
    )


@dataclass(frozen=True, slots=True)
class _ModuleSlot:
    owner: nn.Module
    name: str
    path: str

    def current(self) -> nn.Module | None:
        return getattr(self.owner, self.name, None)

    def replace(self, module: nn.Module) -> None:
        setattr(self.owner, self.name, module)


def _named_modules_with_duplicates(model: nn.Module):
    try:
        return model.named_modules(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with old torch
        return model.named_modules()


def _locate_unique_module_slot(
    model: nn.Module,
    target: nn.Module,
    *,
    label: str,
) -> _ModuleSlot:
    matches: dict[tuple[int, str], _ModuleSlot] = {}
    for owner_path, owner in _named_modules_with_duplicates(model):
        for name, child in owner._modules.items():
            if child is not target:
                continue
            path = f"{owner_path}.{name}" if owner_path else name
            matches.setdefault(
                (id(owner), name),
                _ModuleSlot(owner=owner, name=name, path=path),
            )
    if len(matches) != 1:
        raise DenseLexicalResidualTopologyError(
            f"{label} must occupy exactly one registered module slot in the target."
        )
    return next(iter(matches.values()))


def _named_parameters_with_duplicates(model: nn.Module):
    try:
        return model.named_parameters(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with old torch
        return model.named_parameters()


def _assert_parameter_unaliased(
    model: nn.Module,
    parameter: nn.Parameter,
    *,
    label: str,
) -> None:
    aliases = [
        name
        for name, value in _named_parameters_with_duplicates(model)
        if value is parameter
    ]
    if len(aliases) != 1:
        raise DenseLexicalResidualTopologyError(
            f"{label} is aliased; exact native bake is unsafe."
        )


def _storage_pointer(value: Tensor) -> int:
    try:
        return int(value.untyped_storage().data_ptr())
    except AttributeError:  # pragma: no cover - compatibility with old torch
        return int(value.storage().data_ptr())


def _resolve_default_module(
    model: nn.Module,
    supplied: nn.Module | None,
    *,
    getter_name: str,
    label: str,
) -> nn.Module:
    if supplied is not None:
        return supplied
    getter = getattr(model, getter_name, None)
    if not callable(getter):
        raise DenseLexicalResidualTopologyError(
            f"{label} was not supplied and model.{getter_name} is unavailable."
        )
    result = getter()
    if not isinstance(result, nn.Module):
        raise DenseLexicalResidualTopologyError(
            f"model.{getter_name} did not return an nn.Module."
        )
    return result


class InstalledDenseLexicalResidualAdapter:
    """Live lexical adapter with exact rollback and native dense bake."""

    def __init__(
        self,
        *,
        model: nn.Module,
        registry: DenseLexicalResidualRegistry,
        report: DenseLexicalResidualInstallationReport,
        embedding_slot: _ModuleSlot,
        lm_head_slot: _ModuleSlot,
        original_embedding: nn.Embedding,
        original_lm_head: nn.Linear,
        installed_embedding: DenseLexicalResidualEmbedding,
        installed_lm_head: DenseLexicalResidualLMHead,
        base_state_before: dict[str, _TensorSnapshot],
        original_state_keys: tuple[str, ...],
    ) -> None:
        self.model = model
        self.registry = registry
        self.report = report
        self._embedding_slot = embedding_slot
        self._lm_head_slot = lm_head_slot
        self._original_embedding = original_embedding
        self._original_lm_head = original_lm_head
        self._installed_embedding = installed_embedding
        self._installed_lm_head = installed_lm_head
        self._base_state_before = dict(base_state_before)
        self._original_state_keys = original_state_keys
        self._published_embedding: nn.Embedding | None = None
        self._published_lm_head: nn.Linear | None = None
        self._state: Literal["live", "rolled-back", "baked"] = "live"
        self._bake_report: DenseLexicalResidualBakeReport | None = None

    @property
    def adapter_registry(self) -> DenseLexicalResidualRegistry:
        return self.registry

    @property
    def is_applied(self) -> bool:
        return self._state == "live"

    @property
    def is_baked(self) -> bool:
        return self._state == "baked"

    @property
    def bake_report(self) -> DenseLexicalResidualBakeReport | None:
        return self._bake_report

    def _assert_base_unchanged(self) -> None:
        current = _lexical_state(self._original_embedding, self._original_lm_head)
        if set(current) != set(self._base_state_before):
            raise DenseLexicalResidualTopologyError(
                "Native lexical base tensor topology changed."
            )
        for name, before in self._base_state_before.items():
            after = current[name]
            if (
                after.tensor is not before.tensor
                or after.data_pointer != before.data_pointer
                or after.version != before.version
                or after.sha256 != before.sha256
            ):
                raise DenseLexicalResidualTopologyError(
                    f"Native lexical base tensor {name!r} changed during the transaction."
                )

    def _assert_original_state_keys(self) -> None:
        if tuple(self.model.state_dict()) != self._original_state_keys:
            raise DenseLexicalResidualTopologyError(
                "Native lexical state-key topology was not restored exactly."
            )

    def assert_structurally_live(
        self,
    ) -> DenseLexicalResidualInstallationReport:
        """Validate ownership and immutable native tensors, ignoring factors."""

        if self._state != "live":
            raise DenseLexicalResidualTopologyError(
                "Dense lexical residual adapter is no longer live."
            )
        if getattr(self.model, self.report.owner_name, None) is not self.registry:
            raise DenseLexicalResidualTopologyError(
                "Model lost the lexical residual registry owner."
            )
        if self._embedding_slot.current() is not self._installed_embedding:
            raise DenseLexicalResidualTopologyError(
                "The live lexical embedding wrapper was replaced."
            )
        if self._lm_head_slot.current() is not self._installed_lm_head:
            raise DenseLexicalResidualTopologyError(
                "The live lexical LM-head wrapper was replaced."
            )
        self._assert_base_unchanged()
        return self.report

    def assert_live(self) -> DenseLexicalResidualInstallationReport:
        self.assert_structurally_live()
        self.registry.assert_finite()
        return self.report

    def output_trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        self.assert_live()
        return self.registry.output_trainable_parameters()

    def output_training_scope(self) -> DenseLexicalFactorOnlyOutputTrainingScope:
        return DenseLexicalFactorOnlyOutputTrainingScope(self)

    def rollback(self) -> None:
        if self._state == "rolled-back":
            return
        if self._state == "baked":
            raise DenseLexicalResidualTopologyError(
                "Use restore_prepublication_native after a lexical bake."
            )
        # The four factors are disposable transaction state.  They may contain
        # NaN/Inf after a failed optimizer step, so rollback deliberately checks
        # only wrapper ownership and the untouched native dense tensors.
        self.assert_structurally_live()
        self._embedding_slot.replace(self._original_embedding)
        self._lm_head_slot.replace(self._original_lm_head)
        delattr(self.model, self.report.owner_name)
        self._assert_base_unchanged()
        if (
            _unique_parameter_count(self.model)
            != self.report.model_parameter_count_before
        ):
            raise DenseLexicalResidualTopologyError(
                "Lexical rollback parameter accounting failed."
            )
        self._assert_original_state_keys()
        if (
            self._embedding_slot.current() is not self._original_embedding
            or self._lm_head_slot.current() is not self._original_lm_head
        ):
            raise DenseLexicalResidualTopologyError(
                "Native lexical module rollback failed."
            )
        self._state = "rolled-back"

    def _new_baked_embedding(self, delta: Tensor) -> nn.Embedding:
        original = self._original_embedding
        replacement = nn.Embedding(
            original.num_embeddings,
            original.embedding_dim,
            padding_idx=original.padding_idx,
            max_norm=original.max_norm,
            norm_type=original.norm_type,
            scale_grad_by_freq=original.scale_grad_by_freq,
            sparse=original.sparse,
            device=original.weight.device,
            dtype=original.weight.dtype,
        )
        with torch.no_grad():
            replacement.weight.copy_(original.weight + delta)
        replacement.weight.requires_grad_(original.weight.requires_grad)
        replacement.train(original.training)
        return replacement

    def _new_baked_lm_head(self, delta: Tensor) -> nn.Linear:
        original = self._original_lm_head
        replacement = nn.Linear(
            original.in_features,
            original.out_features,
            bias=original.bias is not None,
            device=original.weight.device,
            dtype=original.weight.dtype,
        )
        with torch.no_grad():
            replacement.weight.copy_(original.weight + delta)
            if original.bias is not None:
                assert replacement.bias is not None
                replacement.bias.copy_(original.bias)
        replacement.weight.requires_grad_(original.weight.requires_grad)
        if original.bias is not None:
            assert replacement.bias is not None
            replacement.bias.requires_grad_(original.bias.requires_grad)
        replacement.train(original.training)
        return replacement

    def bake_for_publication(self) -> DenseLexicalResidualBakeReport:
        self.assert_live()
        before_bake = _unique_parameter_count(self.model)
        manifest: list[dict[str, Any]] = []
        published_embedding: nn.Embedding | None = None
        published_lm_head: nn.Linear | None = None
        try:
            embedding_delta = (
                self.registry.dense_delta("embedding")
                .detach()
                .to(
                    device=self._original_embedding.weight.device,
                    dtype=self._original_embedding.weight.dtype,
                )
            )
            if tuple(embedding_delta.shape) != tuple(
                self._original_embedding.weight.shape
            ):
                raise DenseLexicalResidualTopologyError(
                    "Materialized embedding delta has the wrong dense shape."
                )
            manifest.append(
                {
                    "role": "embedding",
                    "shape": list(embedding_delta.shape),
                    "dtype": str(embedding_delta.dtype),
                    "sha256": _tensor_sha256(embedding_delta),
                }
            )
            published_embedding = self._new_baked_embedding(embedding_delta)
            del embedding_delta

            lm_head_delta = (
                self.registry.dense_delta("lm_head")
                .detach()
                .to(
                    device=self._original_lm_head.weight.device,
                    dtype=self._original_lm_head.weight.dtype,
                )
            )
            if tuple(lm_head_delta.shape) != tuple(self._original_lm_head.weight.shape):
                raise DenseLexicalResidualTopologyError(
                    "Materialized LM-head delta has the wrong dense shape."
                )
            manifest.append(
                {
                    "role": "lm_head",
                    "shape": list(lm_head_delta.shape),
                    "dtype": str(lm_head_delta.dtype),
                    "sha256": _tensor_sha256(lm_head_delta),
                }
            )
            published_lm_head = self._new_baked_lm_head(lm_head_delta)
            del lm_head_delta

            if not bool(torch.isfinite(published_embedding.weight.detach()).all()):
                raise DenseLexicalResidualError(
                    "Publication bake produced a non-finite embedding."
                )
            if any(
                not bool(torch.isfinite(value.detach()).all())
                for value in published_lm_head.parameters()
            ):
                raise DenseLexicalResidualError(
                    "Publication bake produced a non-finite LM head."
                )

            self._embedding_slot.replace(published_embedding)
            self._lm_head_slot.replace(published_lm_head)
            delattr(self.model, self.report.owner_name)
            after = _unique_parameter_count(self.model)
            if after != self.report.model_parameter_count_before:
                raise DenseLexicalResidualAccountingError(
                    "Baked lexical target did not return to its native parameter count."
                )
            self._assert_original_state_keys()
        except Exception:
            if getattr(self.model, self.report.owner_name, None) is not self.registry:
                if hasattr(self.model, self.report.owner_name):
                    delattr(self.model, self.report.owner_name)
                self.model.add_module(self.report.owner_name, self.registry)
            self._embedding_slot.replace(self._installed_embedding)
            self._lm_head_slot.replace(self._installed_lm_head)
            raise

        published_keys = tuple(self.model.state_dict())
        report = DenseLexicalResidualBakeReport(
            vocab_size=self.report.vocab_size,
            hidden_size=self.report.hidden_size,
            rank=self.report.rank,
            adapter_parameters_removed=self.report.adapter_parameter_count,
            model_parameter_count_before_installation=(
                self.report.model_parameter_count_before
            ),
            model_parameter_count_before_bake=before_bake,
            model_parameter_count_after_bake=after,
            original_state_key_sha256=self.report.original_state_key_sha256,
            published_state_key_sha256=_state_key_sha256(published_keys),
            dense_delta_sha256=_canonical_sha256(manifest),
            affected_dense_matrix_count=2,
        )
        self._published_embedding = published_embedding
        self._published_lm_head = published_lm_head
        self._state = "baked"
        self._bake_report = report
        return report

    def restore_prepublication_native(self) -> None:
        """Undo a bake while a surrounding publication transaction rejects."""

        if self._state == "live":
            self.rollback()
            return
        if self._state == "rolled-back":
            return
        if (
            self._published_embedding is None
            or self._published_lm_head is None
            or self._embedding_slot.current() is not self._published_embedding
            or self._lm_head_slot.current() is not self._published_lm_head
        ):
            raise DenseLexicalResidualTopologyError(
                "Published lexical ownership changed; restoration is unsafe."
            )
        self._embedding_slot.replace(self._original_embedding)
        self._lm_head_slot.replace(self._original_lm_head)
        self._assert_base_unchanged()
        if (
            _unique_parameter_count(self.model)
            != self.report.model_parameter_count_before
        ):
            raise DenseLexicalResidualTopologyError(
                "Prepublication lexical restoration changed the native budget."
            )
        self._assert_original_state_keys()
        self._state = "rolled-back"

    def __enter__(self) -> InstalledDenseLexicalResidualAdapter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self._state == "live":
            self.rollback()


class DenseLexicalFactorOnlyOutputTrainingScope:
    """Freeze the dense child and expose exactly four lexical factors."""

    def __init__(self, handle: InstalledDenseLexicalResidualAdapter) -> None:
        if not isinstance(handle, InstalledDenseLexicalResidualAdapter):
            raise DenseLexicalResidualTrainingError(
                "Training scope requires a live dense lexical handle."
            )
        self.handle = handle
        self._base_state: tuple[
            tuple[nn.Parameter, bool, Tensor | None, int, int], ...
        ] = ()
        self._module_modes: tuple[tuple[nn.Module, bool], ...] = ()
        self._entered = False

    def __enter__(self) -> tuple[nn.Parameter, ...]:
        if self._entered:
            raise DenseLexicalResidualTrainingError(
                "Factor-only output training cannot be re-entered."
            )
        self.handle.assert_live()
        trainable = self.handle.output_trainable_parameters()
        factor_ids = {id(value) for value in trainable}
        if len(trainable) != 4 or any(not value.requires_grad for value in trainable):
            raise DenseLexicalResidualTrainingError(
                "Exactly four trainable FP32 lexical factors are required."
            )
        self._base_state = tuple(
            (
                value,
                bool(value.requires_grad),
                value.grad,
                int(value._version),
                int(value.data_ptr()),
            )
            for value in self.handle.model.parameters()
            if id(value) not in factor_ids
        )
        self._module_modes = tuple(
            (module, bool(module.training)) for module in self.handle.model.modules()
        )
        for value, _requires_grad, _gradient, _version, _pointer in self._base_state:
            value.requires_grad_(False)
        self.handle.model.eval()
        self._entered = True
        return trainable

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if not self._entered:
            return
        for value, requires_grad, gradient, _version, _pointer in self._base_state:
            value.requires_grad_(requires_grad)
            if value.grad is not gradient:
                value.grad = gradient
        for module, training in self._module_modes:
            module.training = training
        self._entered = False
        if any(
            int(value._version) != version
            or int(value.data_ptr()) != pointer
            or value.grad is not gradient
            or bool(value.requires_grad) != requires_grad
            for value, requires_grad, gradient, version, pointer in self._base_state
        ):
            raise DenseLexicalResidualTrainingError(
                "A native child parameter changed during factor-only output training."
            )
        self.handle.assert_live()

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": DENSE_LEXICAL_RESIDUAL_TRAINING_SCOPE_SCHEMA,
            "factor_parameters_only": True,
            "factor_dtype": "torch.float32",
            "base_parameters_frozen": True,
            "accepted_target_space": "causal-model-output-actions-or-logits",
            "internal_similarity_target_api_exposed": False,
            "parent_model_api_exposed": False,
            "optimizer_constructed": False,
            "optimizer_state_saved": False,
            "checkpoint_saved": False,
            "candidate_committed": False,
        }
        payload["training_scope_sha256"] = _canonical_sha256(payload)
        return payload


def install_dense_lexical_residual_adapter(
    model: nn.Module,
    input_embedding: nn.Embedding | None = None,
    lm_head: nn.Linear | None = None,
    *,
    rank: int,
    owner_name: str = "dense_lexical_residual_registry",
    initialization_seed: int = 0,
) -> InstalledDenseLexicalResidualAdapter:
    """Install independent target-native corrections on embedding and LM head.

    ``input_embedding`` and ``lm_head`` may be omitted for Hugging Face models
    implementing ``get_input_embeddings`` and ``get_output_embeddings``.
    Both modules must be plain, dense, untied, uniquely owned native modules.
    """

    if not isinstance(model, nn.Module):
        raise DenseLexicalResidualTopologyError("model must be an nn.Module.")
    if (
        not isinstance(owner_name, str)
        or not owner_name.isidentifier()
        or hasattr(model, owner_name)
    ):
        raise DenseLexicalResidualTopologyError(
            "owner_name must be an unused model attribute identifier."
        )
    resolved_embedding = _resolve_default_module(
        model,
        input_embedding,
        getter_name="get_input_embeddings",
        label="input_embedding",
    )
    resolved_lm_head = _resolve_default_module(
        model,
        lm_head,
        getter_name="get_output_embeddings",
        label="lm_head",
    )
    if type(resolved_embedding) is not nn.Embedding:
        raise DenseLexicalResidualTopologyError(
            "input_embedding must be a plain torch.nn.Embedding."
        )
    if type(resolved_lm_head) is not nn.Linear:
        raise DenseLexicalResidualTopologyError(
            "lm_head must be a plain torch.nn.Linear."
        )
    embedding = resolved_embedding
    head = resolved_lm_head
    if embedding is head:
        raise DenseLexicalResidualTopologyError(
            "Embedding and LM head must be separate native modules."
        )
    if embedding.max_norm is not None:
        raise DenseLexicalResidualTopologyError(
            "Embedding max_norm mutates native weights during forward and is unsafe."
        )
    if (
        embedding.num_embeddings != head.out_features
        or embedding.embedding_dim != head.in_features
        or tuple(embedding.weight.shape) != tuple(head.weight.shape)
    ):
        raise DenseLexicalResidualTopologyError(
            "Embedding and LM head must share target vocab and hidden dimensions."
        )
    for label, weight in (
        ("embedding.weight", embedding.weight),
        ("lm_head.weight", head.weight),
    ):
        if (
            not weight.dtype.is_floating_point
            or weight.device.type == "meta"
            or not weight.is_contiguous()
        ):
            raise DenseLexicalResidualTopologyError(
                f"{label} must be a materialized contiguous floating tensor."
            )
        if not bool(torch.isfinite(weight.detach()).all()):
            raise DenseLexicalResidualTopologyError(f"{label} is non-finite.")
    if head.bias is not None:
        if (
            head.bias.device != head.weight.device
            or head.bias.dtype != head.weight.dtype
            or not bool(torch.isfinite(head.bias.detach()).all())
        ):
            raise DenseLexicalResidualTopologyError(
                "LM-head bias must be finite and colocated with its weight."
            )
    if embedding.weight is head.weight or _storage_pointer(
        embedding.weight
    ) == _storage_pointer(head.weight):
        raise DenseLexicalResidualTopologyError(
            "Embedding and LM-head weights must be untied and storage-disjoint."
        )

    embedding_slot = _locate_unique_module_slot(
        model, embedding, label="input_embedding"
    )
    head_slot = _locate_unique_module_slot(model, head, label="lm_head")
    _assert_parameter_unaliased(model, embedding.weight, label="embedding.weight")
    _assert_parameter_unaliased(model, head.weight, label="lm_head.weight")
    if head.bias is not None:
        _assert_parameter_unaliased(model, head.bias, label="lm_head.bias")

    vocab = embedding.num_embeddings
    hidden = embedding.embedding_dim
    rank_value = _index(rank, "rank", positive=True)
    original_state_keys = tuple(model.state_dict())
    before = _unique_parameter_count(model)
    base_state = _lexical_state(embedding, head)
    registry = DenseLexicalResidualRegistry(
        vocab_size=vocab,
        hidden_size=hidden,
        rank=rank_value,
        embedding_device=embedding.weight.device,
        lm_head_device=head.weight.device,
        initialization_seed=initialization_seed,
    )
    installed_embedding = DenseLexicalResidualEmbedding(embedding, registry)
    installed_lm_head = DenseLexicalResidualLMHead(head, registry)
    try:
        model.add_module(owner_name, registry)
        embedding_slot.replace(installed_embedding)
        head_slot.replace(installed_lm_head)
        after = _unique_parameter_count(model)
        accounting = DenseLexicalResidualAccounting.build(
            vocab_size=vocab,
            hidden_size=hidden,
            rank=rank_value,
        )
        if (
            not registry.starts_as_exact_zero_correction
            or not registry.input_factors_are_nonzero
            or any(value.dtype is not torch.float32 for value in registry.parameters())
            or accounting.adapter_parameters != registry.learned_parameter_count
            or after != before + registry.learned_parameter_count
        ):
            raise DenseLexicalResidualAccountingError(
                "Installed lexical factor accounting or exact-zero identity failed."
            )
        report = DenseLexicalResidualInstallationReport(
            vocab_size=vocab,
            hidden_size=hidden,
            rank=rank_value,
            factor_dtype="torch.float32",
            owner_name=owner_name,
            embedding_path=embedding_slot.path,
            lm_head_path=head_slot.path,
            adapter_parameter_count=registry.learned_parameter_count,
            model_parameter_count_before=before,
            model_parameter_count_installed=after,
            original_state_key_count=len(original_state_keys),
            original_state_key_sha256=_state_key_sha256(original_state_keys),
            base_lexical_state_sha256=_lexical_state_sha256(base_state),
            accounting=accounting,
        )
        handle = InstalledDenseLexicalResidualAdapter(
            model=model,
            registry=registry,
            report=report,
            embedding_slot=embedding_slot,
            lm_head_slot=head_slot,
            original_embedding=embedding,
            original_lm_head=head,
            installed_embedding=installed_embedding,
            installed_lm_head=installed_lm_head,
            base_state_before=base_state,
            original_state_keys=original_state_keys,
        )
        handle.assert_live()
        return handle
    except Exception:
        embedding_slot.replace(embedding)
        head_slot.replace(head)
        if getattr(model, owner_name, None) is registry:
            delattr(model, owner_name)
        raise


__all__ = [
    "DENSE_LEXICAL_RESIDUAL_ACCOUNTING_SCHEMA",
    "DENSE_LEXICAL_RESIDUAL_BAKE_SCHEMA",
    "DENSE_LEXICAL_RESIDUAL_INSTALLATION_SCHEMA",
    "DENSE_LEXICAL_RESIDUAL_REGISTRY_SCHEMA",
    "DENSE_LEXICAL_RESIDUAL_TRAINING_SCOPE_SCHEMA",
    "DenseLexicalFactorOnlyOutputTrainingScope",
    "DenseLexicalResidualAccounting",
    "DenseLexicalResidualAccountingError",
    "DenseLexicalResidualBakeReport",
    "DenseLexicalResidualEmbedding",
    "DenseLexicalResidualError",
    "DenseLexicalResidualInstallationReport",
    "DenseLexicalResidualLMHead",
    "DenseLexicalResidualRegistry",
    "DenseLexicalResidualTopologyError",
    "DenseLexicalResidualTrainingError",
    "InstalledDenseLexicalResidualAdapter",
    "LexicalRole",
    "install_dense_lexical_residual_adapter",
]
