"""Exact-budget progressive targets and target-native QKVO publication.

This module connects three already-existing Neural Forge boundaries:

* an explicit, target-sized dense GQA child;
* the exact-size blank-child parent compiler; and
* temporary output-trained Q/K/V/O residual factors that can be baked into
  ordinary target ``nn.Linear`` weights.

The bridge does not train factors and does not score behavior.  It requires an
external behavior-gate receipt before publication, then proves the structural
facts it can establish locally: the adapter registry is gone, every Q/K/V/O
projection is native dense, the state-key topology is restored, and the unique
parameter count is exactly the requested child budget.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from typing import Any, Literal

from torch import Tensor, nn

from neural_forge.hf_arch_layout import (
    DecoderLayout,
    build_decoder_layout,
    resolve_module,
)
from neural_forge.hf_dense_gqa_qkvo_residual_adapter import (
    DenseGQAAdapterOnlyOutputTrainingScope,
    DenseGQAGeometry,
    DenseGQAQKVOResidualProjection,
    DenseGQAQKVOResidualTopologyError,
    DenseGQASite,
    InstalledDenseGQAQKVOResidualAdapter,
    SparseDenseGQAQKVOResidualRegistry,
    install_dense_gqa_qkvo_residual_adapter,
)
from neural_forge.hf_dense_lexical_residual_adapter import (
    DenseLexicalResidualError,
    InstalledDenseLexicalResidualAdapter,
    install_dense_lexical_residual_adapter,
)
from neural_forge.hf_factorized_vocab import exact_parameter_report
from neural_forge.hf_dense_swiglu_projection_residual_adapter import (
    DenseSwiGLUProjectionResidualTopologyError,
    DenseSwiGLUResidualProjection,
    InstalledDenseSwiGLUProjectionResidualAdapter,
    SparseDenseSwiGLUProjectionResidualRegistry,
    install_dense_swiglu_projection_residual_adapter,
)
from neural_forge.hf_parent_compiler import (
    HFParentCompilerResult,
    fingerprint_hf_child_structure,
)


PROGRESSIVE_TARGET_PLAN_SCHEMA = "neural-forge-progressive-dense-gqa-plan-v1"
NATIVE_TARGET_CERTIFICATE_SCHEMA = "neural-forge-native-dense-gqa-certificate-v1"
EXACT_TARGET_QKVO_INSTALLATION_SCHEMA = "neural-forge-exact-target-qkvo-installation-v1"
EXACT_TARGET_QKVO_PUBLICATION_SCHEMA = "neural-forge-exact-target-qkvo-publication-v1"
EXACT_TARGET_COMPILER_TRANSACTION_INSTALLATION_SCHEMA = (
    "neural-forge-exact-target-compiler-transaction-installation-v1"
)
EXACT_TARGET_COMPILER_TRANSACTION_PUBLICATION_SCHEMA = (
    "neural-forge-exact-target-compiler-transaction-publication-v1"
)
EXACT_TARGET_COMPILER_OUTPUT_TRAINING_SCOPE_SCHEMA = (
    "neural-forge-exact-target-compiler-output-training-scope-v1"
)
EXTERNAL_BEHAVIOR_GATE_SCHEMA = "neural-forge-external-behavior-gate-receipt-v1"

QWEN3_4B_EXACT_PARAMETERS = 4_022_468_096
QWEN3_1P4B_EXACT_PARAMETERS = 1_410_567_168
QWEN3_600M_EXACT_PARAMETERS = 600_628_224
SMOLLM2_135M_EXACT_PARAMETERS = 134_515_008

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProgressiveTargetBridgeError(ValueError):
    """The exact target or publication contract is invalid."""


class ProgressiveTargetPublicationError(ProgressiveTargetBridgeError):
    """A temporary QKVO correction cannot be published safely."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProgressiveTargetBridgeError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProgressiveTargetBridgeError(f"{name} must be a nonnegative integer.")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProgressiveTargetBridgeError(f"{name} must be bool.")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ProgressiveTargetBridgeError(f"{name} must be a SHA-256 string.")
    normalized = value.removeprefix("sha256:").lower()
    if not _HEX_SHA256.fullmatch(normalized):
        raise ProgressiveTargetBridgeError(f"{name} must be a SHA-256 string.")
    return normalized


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
        raise ProgressiveTargetBridgeError(
            "Target provenance must be finite canonical JSON."
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _unique_parameter_bytes(model: nn.Module) -> int:
    unique: dict[tuple[Any, ...], int] = {}
    for _, parameter in model.named_parameters(recurse=True, remove_duplicate=False):
        key = (
            parameter.device.type,
            parameter.device.index,
            str(parameter.dtype),
            parameter.untyped_storage().data_ptr(),
            parameter.storage_offset(),
            tuple(parameter.shape),
            tuple(parameter.stride()),
        )
        unique[key] = int(parameter.numel()) * int(parameter.element_size())
    return sum(unique.values())


@dataclass(frozen=True, slots=True)
class DenseGQATargetShape:
    """Closed-form storage shape for a native dense GQA/SwiGLU target.

    The formula covers the current Qwen3 and SmolLM2/Llama progressive targets.
    ``query_key_head_norms`` is explicit because Qwen3 stores one Q and K
    RMSNorm vector of ``head_dim`` elements per layer, while SmolLM2 does not.
    """

    family: Literal["qwen3", "llama"]
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    tie_word_embeddings: bool
    query_key_head_norms: bool
    attention_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self) -> None:
        if self.family not in {"qwen3", "llama"}:
            raise ProgressiveTargetBridgeError(
                "DenseGQATargetShape family must be qwen3 or llama."
            )
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in (
            "tie_word_embeddings",
            "query_key_head_norms",
            "attention_bias",
            "mlp_bias",
        ):
            object.__setattr__(self, name, _strict_bool(getattr(self, name), name))
        if self.num_attention_heads % self.num_key_value_heads:
            raise ProgressiveTargetBridgeError(
                "num_attention_heads must be divisible by num_key_value_heads."
            )

    @property
    def geometry(self) -> DenseGQAGeometry:
        return DenseGQAGeometry(
            logical_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            query_heads=self.num_attention_heads,
            key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
        )

    def parameter_breakdown(self) -> dict[str, Any]:
        hidden = self.hidden_size
        intermediate = self.intermediate_size
        query_width = self.num_attention_heads * self.head_dim
        key_value_width = self.num_key_value_heads * self.head_dim
        attention_weights = 2 * hidden * (query_width + key_value_width)
        attention_biases = (
            query_width + 2 * key_value_width + hidden if self.attention_bias else 0
        )
        query_key_norms = 2 * self.head_dim if self.query_key_head_norms else 0
        mlp_weights = 3 * hidden * intermediate
        mlp_biases = 2 * intermediate + hidden if self.mlp_bias else 0
        decoder_norms = 2 * hidden
        per_layer = (
            attention_weights
            + attention_biases
            + query_key_norms
            + mlp_weights
            + mlp_biases
            + decoder_norms
        )
        embedding = self.vocab_size * hidden
        output_head_unique = 0 if self.tie_word_embeddings else embedding
        decoder_layers = self.num_hidden_layers * per_layer
        final_norm = hidden
        total = embedding + output_head_unique + decoder_layers + final_norm
        return {
            "family": self.family,
            "shape": self.as_dict(),
            "derived": {
                "query_width": query_width,
                "key_value_width": key_value_width,
                "queries_per_key_value_head": (
                    self.num_attention_heads // self.num_key_value_heads
                ),
            },
            "per_layer": {
                "attention_weights": attention_weights,
                "attention_biases": attention_biases,
                "query_key_head_norms": query_key_norms,
                "gated_mlp_weights": mlp_weights,
                "mlp_biases": mlp_biases,
                "decoder_norms": decoder_norms,
                "total": per_layer,
            },
            "components": {
                "token_embedding": embedding,
                "output_head_unique": output_head_unique,
                "decoder_layers": decoder_layers,
                "final_norm": final_norm,
            },
            "total": total,
        }

    @property
    def unique_parameter_count(self) -> int:
        return int(self.parameter_breakdown()["total"])

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProgressiveDenseGQAStage:
    """One exact native target in the progressive compiler chain."""

    name: str
    shape: DenseGQATargetShape
    exact_unique_parameters: int
    tokenizer_transition: Literal["preserve", "canonical-byte-transition"]
    purpose: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or not self.name.replace("_", "").replace("-", "").isalnum()
        ):
            raise ProgressiveTargetBridgeError("Stage name is not a safe identifier.")
        if not isinstance(self.shape, DenseGQATargetShape):
            raise ProgressiveTargetBridgeError(
                "stage.shape must be DenseGQATargetShape."
            )
        object.__setattr__(
            self,
            "exact_unique_parameters",
            _positive_int(self.exact_unique_parameters, "exact_unique_parameters"),
        )
        if self.tokenizer_transition not in {
            "preserve",
            "canonical-byte-transition",
        }:
            raise ProgressiveTargetBridgeError(
                "Unsupported tokenizer transition policy."
            )
        if not isinstance(self.purpose, str) or not self.purpose.strip():
            raise ProgressiveTargetBridgeError("Stage purpose cannot be empty.")
        calculated = self.shape.unique_parameter_count
        if calculated != self.exact_unique_parameters:
            raise ProgressiveTargetBridgeError(
                f"Stage {self.name!r} declares {self.exact_unique_parameters} "
                f"parameters, but its native shape contains {calculated}."
            )

    @property
    def geometry(self) -> DenseGQAGeometry:
        return self.shape.geometry

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "exact_unique_parameters": self.exact_unique_parameters,
            "tokenizer_transition": self.tokenizer_transition,
            "purpose": self.purpose,
            "shape": self.shape.as_dict(),
            "parameter_breakdown": self.shape.parameter_breakdown(),
            "native_dense_target": True,
            "adapter_parameters_at_publication": 0,
        }
        payload["stage_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class ProgressiveDenseGQAPlan:
    """Exact monotone stage plan; behavior promotion remains external."""

    source_name: str
    source_unique_parameters: int
    stages: tuple[ProgressiveDenseGQAStage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_name, str) or not self.source_name.strip():
            raise ProgressiveTargetBridgeError("source_name cannot be empty.")
        object.__setattr__(
            self,
            "source_unique_parameters",
            _positive_int(self.source_unique_parameters, "source_unique_parameters"),
        )
        values = tuple(self.stages)
        if not values or any(
            not isinstance(stage, ProgressiveDenseGQAStage) for stage in values
        ):
            raise ProgressiveTargetBridgeError(
                "stages must contain ProgressiveDenseGQAStage values."
            )
        if len({stage.name for stage in values}) != len(values):
            raise ProgressiveTargetBridgeError(
                "Progressive stage names must be unique."
            )
        previous = self.source_unique_parameters
        for stage in values:
            if stage.exact_unique_parameters >= previous:
                raise ProgressiveTargetBridgeError(
                    "Progressive target counts must be strictly descending."
                )
            previous = stage.exact_unique_parameters
        object.__setattr__(self, "stages", values)

    @property
    def final_unique_parameters(self) -> int:
        return self.stages[-1].exact_unique_parameters

    @property
    def overall_compression_ratio(self) -> float:
        return self.source_unique_parameters / self.final_unique_parameters

    def as_dict(self) -> dict[str, Any]:
        previous_name = self.source_name
        previous_count = self.source_unique_parameters
        stage_records = []
        for stage in self.stages:
            stage_records.append(
                {
                    **stage.as_dict(),
                    "input_name": previous_name,
                    "input_unique_parameters": previous_count,
                    "single_stage_compression_ratio": (
                        previous_count / stage.exact_unique_parameters
                    ),
                    "cumulative_compression_ratio": (
                        self.source_unique_parameters / stage.exact_unique_parameters
                    ),
                    "behavior_gate_required_before_next_stage": True,
                }
            )
            previous_name = stage.name
            previous_count = stage.exact_unique_parameters
        payload = {
            "schema": PROGRESSIVE_TARGET_PLAN_SCHEMA,
            "source": {
                "name": self.source_name,
                "unique_parameters": self.source_unique_parameters,
            },
            "stages": stage_records,
            "final_unique_parameters": self.final_unique_parameters,
            "overall_compression_ratio": self.overall_compression_ratio,
            "minimum_25x_endpoint": self.overall_compression_ratio >= 25.0,
            "every_stage_is_native_target_sized": True,
            "every_stage_requires_parent_relative_behavior_promotion": True,
            "capability_proven_by_plan": False,
        }
        payload["plan_sha256"] = _canonical_sha256(payload)
        return payload


def default_qwen4b_progressive_target_plan() -> ProgressiveDenseGQAPlan:
    """Return the exact 4B -> 1.410B -> 600.628M -> 134.515M plan."""

    return ProgressiveDenseGQAPlan(
        source_name="Qwen3-4B-Base",
        source_unique_parameters=QWEN3_4B_EXACT_PARAMETERS,
        stages=(
            ProgressiveDenseGQAStage(
                name="qwen3_1p410b",
                shape=DenseGQATargetShape(
                    family="qwen3",
                    vocab_size=151_936,
                    hidden_size=1_536,
                    intermediate_size=5_120,
                    num_hidden_layers=30,
                    num_attention_heads=16,
                    num_key_value_heads=4,
                    head_dim=128,
                    tie_word_embeddings=False,
                    query_key_head_norms=True,
                ),
                exact_unique_parameters=QWEN3_1P4B_EXACT_PARAMETERS,
                tokenizer_transition="preserve",
                purpose="first bounded same-tokenizer width/depth fold",
            ),
            ProgressiveDenseGQAStage(
                name="qwen3_600m",
                shape=DenseGQATargetShape(
                    family="qwen3",
                    vocab_size=151_936,
                    hidden_size=1_024,
                    intermediate_size=3_072,
                    num_hidden_layers=24,
                    num_attention_heads=8,
                    num_key_value_heads=2,
                    head_dim=128,
                    tie_word_embeddings=False,
                    query_key_head_norms=True,
                ),
                exact_unique_parameters=QWEN3_600M_EXACT_PARAMETERS,
                tokenizer_transition="preserve",
                purpose="second bounded same-tokenizer width/depth fold",
            ),
            ProgressiveDenseGQAStage(
                name="smollm2_134m",
                shape=DenseGQATargetShape(
                    family="llama",
                    vocab_size=49_152,
                    hidden_size=576,
                    intermediate_size=1_536,
                    num_hidden_layers=30,
                    num_attention_heads=9,
                    num_key_value_heads=3,
                    head_dim=64,
                    tie_word_embeddings=True,
                    query_key_head_norms=False,
                ),
                exact_unique_parameters=SMOLLM2_135M_EXACT_PARAMETERS,
                tokenizer_transition="canonical-byte-transition",
                purpose="29.9x standalone product-size endpoint",
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class ExternalBehaviorGateReceipt:
    """External output/rollout evidence required to authorize dense baking.

    The bridge validates the receipt's shape, not the referenced evaluation
    file.  Consequently its reports say that behavior was externally attested,
    never that this structural module independently proved capability.
    """

    evidence_sha256: str
    evaluation_contract_sha256: str
    heldout_action_count: int
    free_running_rollout_count: int
    passed: bool
    internal_similarity_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_sha256",
            _sha256(self.evidence_sha256, "evidence_sha256"),
        )
        object.__setattr__(
            self,
            "evaluation_contract_sha256",
            _sha256(
                self.evaluation_contract_sha256,
                "evaluation_contract_sha256",
            ),
        )
        object.__setattr__(
            self,
            "heldout_action_count",
            _nonnegative_int(self.heldout_action_count, "heldout_action_count"),
        )
        object.__setattr__(
            self,
            "free_running_rollout_count",
            _nonnegative_int(
                self.free_running_rollout_count,
                "free_running_rollout_count",
            ),
        )
        object.__setattr__(self, "passed", _strict_bool(self.passed, "passed"))
        object.__setattr__(
            self,
            "internal_similarity_only",
            _strict_bool(self.internal_similarity_only, "internal_similarity_only"),
        )

    def require_publication_approval(self) -> None:
        if not self.passed:
            raise ProgressiveTargetPublicationError(
                "The external behavior gate did not approve publication."
            )
        if self.internal_similarity_only:
            raise ProgressiveTargetPublicationError(
                "Internal numeric similarity cannot authorize publication."
            )
        if self.heldout_action_count <= 0 or self.free_running_rollout_count <= 0:
            raise ProgressiveTargetPublicationError(
                "Publication needs held-out actions and free-running rollouts."
            )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": EXTERNAL_BEHAVIOR_GATE_SCHEMA,
            **asdict(self),
            "behavior_evidence_externally_attested": True,
            "evidence_payload_revalidated_by_bridge": False,
        }
        payload["receipt_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class NativeDenseGQACertificate:
    exact_unique_parameters: int
    logical_parameters: int
    unique_learned_weight_bytes: int
    structure_sha256: str
    state_key_sha256: str
    state_key_count: int
    dense_qkvo_projection_count: int
    dense_swiglu_projection_count: int
    geometry: DenseGQAGeometry
    projection_names: Mapping[str, str]
    swiglu_projection_names: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": NATIVE_TARGET_CERTIFICATE_SCHEMA,
            "exact_unique_parameters": self.exact_unique_parameters,
            "logical_parameters": self.logical_parameters,
            "unique_learned_weight_bytes": self.unique_learned_weight_bytes,
            "structure_sha256": self.structure_sha256,
            "state_key_sha256": self.state_key_sha256,
            "state_key_count": self.state_key_count,
            "dense_qkvo_projection_count": self.dense_qkvo_projection_count,
            "dense_swiglu_projection_count": self.dense_swiglu_projection_count,
            "geometry": self.geometry.as_dict(),
            "projection_names": dict(self.projection_names),
            "swiglu_projection_names": dict(self.swiglu_projection_names),
            "all_qkvo_projections_are_exact_nn_linear": True,
            "all_swiglu_projections_are_exact_nn_linear": True,
            "adapter_registry_count": 0,
            "adapter_projection_wrapper_count": 0,
            "adapter_parameter_count": 0,
            "parent_runtime_reference_count": 0,
            "source_sized_sidecar_parameter_count": 0,
            "capability_proven": False,
        }
        payload["certificate_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class _DenseGQATopology:
    layout: DecoderLayout
    geometry: DenseGQAGeometry
    attention_modules: Mapping[int, nn.Module]
    projection_names: Mapping[str, str]
    projection_paths: tuple[str, ...]
    mlp_modules: Mapping[int, nn.Module]
    swiglu_projection_names: Mapping[str, str]
    swiglu_projection_paths: tuple[str, ...]


def _direct_projection_name(module_path: str, projection_path: str) -> str:
    prefix = f"{module_path}."
    if not projection_path.startswith(prefix):
        raise ProgressiveTargetBridgeError(
            "Projection path is not owned by its declared module."
        )
    name = projection_path[len(prefix) :]
    if not name.isidentifier():
        raise ProgressiveTargetBridgeError(
            "Native publication requires direct module projection fields."
        )
    return name


def _resolve_dense_gqa_topology(
    model: nn.Module,
    layout: DecoderLayout | None,
    *,
    require_dense_swiglu: bool = False,
) -> _DenseGQATopology:
    resolved = build_decoder_layout(model) if layout is None else layout
    if not isinstance(resolved, DecoderLayout):
        raise ProgressiveTargetBridgeError("layout must be DecoderLayout.")
    if not resolved.blocks:
        raise ProgressiveTargetBridgeError("Target decoder has no layers.")
    if tuple(block.index for block in resolved.blocks) != tuple(
        range(resolved.layer_count)
    ):
        raise ProgressiveTargetBridgeError(
            "Target decoder block indices must be contiguous from zero."
        )
    first = resolved.blocks[0].attention
    if first.projection_style != "split_qkv":
        raise ProgressiveTargetBridgeError(
            "Target-native QKVO baking requires split dense Q/K/V projections."
        )
    geometry = DenseGQAGeometry(
        logical_layers=resolved.layer_count,
        hidden_size=resolved.hidden_size,
        query_heads=first.query_heads,
        key_value_heads=first.key_value_heads,
        head_dim=first.head_dim,
    )
    modules: dict[int, nn.Module] = {}
    mlp_modules: dict[int, nn.Module] = {}
    paths: list[str] = []
    mlp_paths: list[str] = []
    names: dict[str, str] | None = None
    mlp_names: dict[str, str] | None = None
    for block in resolved.blocks:
        attention = block.attention
        if (
            attention.projection_style != "split_qkv"
            or attention.hidden_size != geometry.hidden_size
            or attention.query_heads != geometry.query_heads
            or attention.key_value_heads != geometry.key_value_heads
            or attention.head_dim != geometry.head_dim
            or attention.projection_weight_layout != "output_input"
            or attention.query_path is None
            or attention.key_path is None
            or attention.value_path is None
        ):
            raise ProgressiveTargetBridgeError(
                "Every target layer must share one dense split-GQA geometry."
            )
        module = resolve_module(model, attention.module_path, family=resolved.family)
        modules[block.index] = module
        layer_paths = {
            "q": attention.query_path,
            "k": attention.key_path,
            "v": attention.value_path,
            "o": attention.output_path,
        }
        layer_names = {
            role: _direct_projection_name(attention.module_path, path)
            for role, path in layer_paths.items()
        }
        if names is None:
            names = layer_names
        elif names != layer_names:
            raise ProgressiveTargetBridgeError(
                "Target layers use inconsistent QKVO projection attribute names."
            )
        for path in layer_paths.values():
            projection = resolve_module(model, path, family=resolved.family)
            if type(projection) is not nn.Linear:
                raise ProgressiveTargetBridgeError(
                    f"Target projection {path!r} is not an ordinary nn.Linear."
                )
            paths.append(path)
        if require_dense_swiglu:
            mlp = block.mlp
            if (
                mlp.kind != "gated"
                or mlp.gate_projection_path is None
                or mlp.up_projection_path is None
                or mlp.down_projection_path is None
                or mlp.projection_weight_layout != "output_input"
                or mlp.hidden_size != resolved.hidden_size
                or mlp.intermediate_size != resolved.intermediate_size
            ):
                raise ProgressiveTargetBridgeError(
                    "Every target layer must expose one native dense gated MLP."
                )
            mlp_module = resolve_module(model, mlp.module_path, family=resolved.family)
            mlp_modules[block.index] = mlp_module
            layer_mlp_paths = {
                "gate": mlp.gate_projection_path,
                "up": mlp.up_projection_path,
                "down": mlp.down_projection_path,
            }
            layer_mlp_names = {
                role: _direct_projection_name(mlp.module_path, path)
                for role, path in layer_mlp_paths.items()
            }
            if mlp_names is None:
                mlp_names = layer_mlp_names
            elif mlp_names != layer_mlp_names:
                raise ProgressiveTargetBridgeError(
                    "Target layers use inconsistent Gate/Up/Down attribute names."
                )
            for path in layer_mlp_paths.values():
                projection = resolve_module(model, path, family=resolved.family)
                if type(projection) is not nn.Linear:
                    raise ProgressiveTargetBridgeError(
                        f"Target projection {path!r} is not an ordinary nn.Linear."
                    )
                mlp_paths.append(path)
    assert names is not None
    if require_dense_swiglu:
        assert mlp_names is not None
    return _DenseGQATopology(
        layout=resolved,
        geometry=geometry,
        attention_modules=modules,
        projection_names=names,
        projection_paths=tuple(paths),
        mlp_modules=mlp_modules,
        swiglu_projection_names={} if mlp_names is None else mlp_names,
        swiglu_projection_paths=tuple(mlp_paths),
    )


def _validate_stage_matches_layout(
    stage: ProgressiveDenseGQAStage,
    topology: _DenseGQATopology,
) -> None:
    shape = stage.shape
    layout = topology.layout
    first_attention = layout.blocks[0].attention
    query_norm_present = first_attention.query_norm is not None
    key_norm_present = first_attention.key_norm is not None
    if query_norm_present != key_norm_present:
        raise ProgressiveTargetBridgeError(
            "Target must expose both Q/K head norms or neither."
        )
    checks = {
        "family": (layout.family, shape.family),
        "vocab_size": (layout.embeddings.vocab_size, shape.vocab_size),
        "hidden_size": (layout.hidden_size, shape.hidden_size),
        "intermediate_size": (layout.intermediate_size, shape.intermediate_size),
        "num_hidden_layers": (layout.layer_count, shape.num_hidden_layers),
        "num_attention_heads": (
            first_attention.query_heads,
            shape.num_attention_heads,
        ),
        "num_key_value_heads": (
            first_attention.key_value_heads,
            shape.num_key_value_heads,
        ),
        "head_dim": (first_attention.head_dim, shape.head_dim),
        "tie_word_embeddings": (
            layout.tie_word_embeddings,
            shape.tie_word_embeddings,
        ),
        "attention_bias": (
            first_attention.projection_bias,
            shape.attention_bias,
        ),
        "mlp_bias": (layout.blocks[0].mlp.projection_bias, shape.mlp_bias),
        "query_key_head_norms": (
            query_norm_present,
            shape.query_key_head_norms,
        ),
    }
    mismatches = [
        f"{name}: live={actual!r}, planned={planned!r}"
        for name, (actual, planned) in checks.items()
        if actual != planned
    ]
    if mismatches:
        raise ProgressiveTargetBridgeError(
            "Live child does not match its progressive stage: " + "; ".join(mismatches)
        )


def certify_native_dense_gqa_target(
    model: nn.Module,
    *,
    exact_unique_parameters: int,
    layout: DecoderLayout | None = None,
    stage: ProgressiveDenseGQAStage | None = None,
    require_dense_swiglu: bool = False,
) -> NativeDenseGQACertificate:
    """Certify a loaded target has exact storage and no QKVO sidecar modules."""

    if not isinstance(model, nn.Module):
        raise ProgressiveTargetBridgeError("model must be torch.nn.Module.")
    expected = _positive_int(exact_unique_parameters, "exact_unique_parameters")
    topology = _resolve_dense_gqa_topology(
        model,
        layout,
        require_dense_swiglu=require_dense_swiglu,
    )
    if stage is not None:
        if not isinstance(stage, ProgressiveDenseGQAStage):
            raise ProgressiveTargetBridgeError(
                "stage must be ProgressiveDenseGQAStage."
            )
        if stage.exact_unique_parameters != expected:
            raise ProgressiveTargetBridgeError(
                "Stage count and requested exact count disagree."
            )
        _validate_stage_matches_layout(stage, topology)
    forbidden_registry = sum(
        isinstance(module, SparseDenseGQAQKVOResidualRegistry)
        for module in model.modules()
    )
    forbidden_wrappers = sum(
        isinstance(module, DenseGQAQKVOResidualProjection) for module in model.modules()
    )
    forbidden_mlp_registry = sum(
        isinstance(module, SparseDenseSwiGLUProjectionResidualRegistry)
        for module in model.modules()
    )
    forbidden_mlp_wrappers = sum(
        isinstance(module, DenseSwiGLUResidualProjection) for module in model.modules()
    )
    if (
        forbidden_registry
        or forbidden_wrappers
        or forbidden_mlp_registry
        or forbidden_mlp_wrappers
    ):
        raise ProgressiveTargetBridgeError(
            "Target still contains an adapter registry or projection wrapper."
        )
    counts = exact_parameter_report(model)
    actual = int(counts["unique_parameter_count"])
    if actual != expected:
        raise ProgressiveTargetBridgeError(
            f"Native target contains {actual} unique parameters, expected {expected}."
        )
    keys = tuple(sorted(model.state_dict()))
    return NativeDenseGQACertificate(
        exact_unique_parameters=actual,
        logical_parameters=int(counts["logical_parameter_count"]),
        unique_learned_weight_bytes=_unique_parameter_bytes(model),
        structure_sha256=fingerprint_hf_child_structure(
            model,
            layout=topology.layout,
        ),
        state_key_sha256=_canonical_sha256(keys),
        state_key_count=len(keys),
        dense_qkvo_projection_count=len(topology.projection_paths),
        dense_swiglu_projection_count=len(topology.swiglu_projection_paths),
        geometry=topology.geometry,
        projection_names=topology.projection_names,
        swiglu_projection_names=topology.swiglu_projection_names,
    )


class ExactTargetQKVOPublicationSession:
    """One temporary adapter transaction around an exact native child."""

    def __init__(
        self,
        *,
        model: nn.Module,
        exact_unique_parameters: int,
        topology: _DenseGQATopology,
        native_before: NativeDenseGQACertificate,
        handle: InstalledDenseGQAQKVOResidualAdapter,
        stage: ProgressiveDenseGQAStage | None,
        parent_compiler_report_sha256: str | None,
    ) -> None:
        self.model = model
        self.exact_unique_parameters = exact_unique_parameters
        self.topology = topology
        self.native_before = native_before
        self.handle = handle
        self.stage = stage
        self.parent_compiler_report_sha256 = parent_compiler_report_sha256
        self.publication_report: dict[str, Any] | None = None

    @property
    def is_live(self) -> bool:
        return self.handle.is_applied

    def output_training_scope(self) -> DenseGQAAdapterOnlyOutputTrainingScope:
        return self.handle.output_training_scope()

    def rollback(self) -> NativeDenseGQACertificate:
        self.handle.rollback()
        restored = certify_native_dense_gqa_target(
            self.model,
            exact_unique_parameters=self.exact_unique_parameters,
            layout=self.topology.layout,
            stage=self.stage,
        )
        if restored.as_dict() != self.native_before.as_dict():
            raise ProgressiveTargetPublicationError(
                "QKVO rejection did not restore the exact native target."
            )
        return restored

    def installation_report(self) -> dict[str, Any]:
        payload = {
            "schema": EXACT_TARGET_QKVO_INSTALLATION_SCHEMA,
            "exact_unique_parameters_at_publication": self.exact_unique_parameters,
            "temporary_installed_parameters": (
                self.handle.report.model_parameter_count_installed
            ),
            "temporary_adapter_parameters": (
                self.handle.report.adapter_parameter_count
            ),
            "native_before": self.native_before.as_dict(),
            "dense_qkvo_installation": self.handle.report.as_dict(),
            "stage": None if self.stage is None else self.stage.as_dict(),
            "parent_compiler_report_sha256": self.parent_compiler_report_sha256,
            "temporary_factors_may_be_output_trained": True,
            "internal_numeric_matching_required": False,
            "checkpoint_publication_allowed_while_installed": False,
            "capability_proven": False,
        }
        payload["installation_bridge_sha256"] = _canonical_sha256(payload)
        return payload

    def bake_for_publication(
        self,
        behavior_gate: ExternalBehaviorGateReceipt,
    ) -> dict[str, Any]:
        if not isinstance(behavior_gate, ExternalBehaviorGateReceipt):
            raise ProgressiveTargetPublicationError(
                "behavior_gate must be ExternalBehaviorGateReceipt."
            )
        behavior_gate.require_publication_approval()
        bake = self.handle.bake_for_publication()
        native_after = certify_native_dense_gqa_target(
            self.model,
            exact_unique_parameters=self.exact_unique_parameters,
            layout=self.topology.layout,
            stage=self.stage,
        )
        if (
            native_after.structure_sha256 != self.native_before.structure_sha256
            or native_after.state_key_sha256 != self.native_before.state_key_sha256
            or native_after.state_key_count != self.native_before.state_key_count
        ):
            raise ProgressiveTargetPublicationError(
                "Dense QKVO bake did not restore the native target topology."
            )
        payload = {
            "schema": EXACT_TARGET_QKVO_PUBLICATION_SCHEMA,
            "exact_unique_parameters": self.exact_unique_parameters,
            "exact_budget_match": True,
            "native_before": self.native_before.as_dict(),
            "native_after": native_after.as_dict(),
            "dense_bake": bake.as_dict(),
            "external_behavior_gate": behavior_gate.as_dict(),
            "stage": None if self.stage is None else self.stage.as_dict(),
            "parent_compiler_report_sha256": self.parent_compiler_report_sha256,
            "published_adapter_parameter_count": 0,
            "published_adapter_registry_count": 0,
            "published_projection_wrapper_count": 0,
            "published_adapter_flops_per_token": 0,
            "published_parent_runtime_reference_count": 0,
            "published_source_sized_sidecar_parameter_count": 0,
            "native_state_key_topology_restored": True,
            "behavior_gate_externally_attested": True,
            "behavior_evidence_revalidated_by_this_module": False,
            "capability_proven_by_this_module": False,
            "save_reload_artifact_gate_still_required": True,
        }
        payload["publication_sha256"] = _canonical_sha256(payload)
        self.publication_report = payload
        return payload

    def __enter__(self) -> ExactTargetQKVOPublicationSession:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self.handle.is_applied:
            self.rollback()


class ExactTargetCompilerOutputTrainingScope:
    """Expose only combined QKVO, Gate/Up/Down, and lexical factors."""

    def __init__(self, transaction: ExactTargetCompilerTransaction) -> None:
        self.transaction = transaction
        self._base: tuple[tuple[nn.Parameter, bool, Tensor | None, int, int], ...] = ()
        self._trainable: tuple[tuple[nn.Parameter, bool, Tensor | None], ...] = ()
        self._modes: tuple[tuple[nn.Module, bool], ...] = ()
        self._entered = False

    def __enter__(self) -> tuple[nn.Parameter, ...]:
        if self._entered:
            raise ProgressiveTargetBridgeError(
                "Compiler output-training scope cannot be re-entered."
            )
        self.transaction.assert_live()
        trainable = self.transaction.output_trainable_parameters()
        trainable_ids = {id(value) for value in trainable}
        if len(trainable_ids) != len(trainable):
            raise ProgressiveTargetBridgeError(
                "Temporary compiler factors unexpectedly share parameters."
            )
        self._base = tuple(
            (
                value,
                bool(value.requires_grad),
                value.grad,
                int(value._version),
                int(value.data_ptr()),
            )
            for value in self.transaction.model.parameters()
            if id(value) not in trainable_ids
        )
        self._trainable = tuple(
            (value, bool(value.requires_grad), value.grad) for value in trainable
        )
        self._modes = tuple(
            (module, bool(module.training))
            for module in self.transaction.model.modules()
        )
        for value, *_rest in self._base:
            value.requires_grad_(False)
        for value, _requires_grad, _gradient in self._trainable:
            value.requires_grad_(True)
        self.transaction.model.eval()
        self._entered = True
        return trainable

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        failed = exc_type is not None
        del exc, traceback
        if not self._entered:
            return
        for value, requires_grad, gradient, _version, _pointer in self._base:
            value.requires_grad_(requires_grad)
            value.grad = gradient
        for value, requires_grad, _gradient in self._trainable:
            value.requires_grad_(requires_grad)
        for module, training in self._modes:
            module.training = training
        self._entered = False
        if any(
            int(value._version) != version
            or int(value.data_ptr()) != pointer
            or value.grad is not gradient
            for value, _requires_grad, gradient, version, pointer in self._base
        ):
            raise ProgressiveTargetBridgeError(
                "A native child parameter changed during factor-only training."
            )
        # Preserve the original training exception even when a disposable
        # factor was poisoned.  The trainer can now restore its exact captured
        # factor state before the usual full finite/live assertion.  A clean
        # scope exit still enforces the complete invariant here.
        if not failed:
            self.transaction.assert_live()

    def as_dict(self) -> dict[str, Any]:
        self.transaction.assert_live()
        payload = {
            "schema": EXACT_TARGET_COMPILER_OUTPUT_TRAINING_SCOPE_SCHEMA,
            "trainable_qkvo_factor_parameters": (
                self.transaction.qkvo_handle.report.adapter_parameter_count
            ),
            "trainable_gate_up_down_factor_parameters": (
                self.transaction.swiglu_handle.report.adapter_parameter_count
            ),
            "base_parameter_updates_allowed": False,
            "accepted_objective_domain": (
                "causal-model-output-actions-logits-and-free-rollouts"
            ),
            "weight_matching_allowed": False,
            "activation_matching_allowed": False,
            "hidden_state_matching_allowed": False,
            "parent_runtime_reference_count": 0,
            "optimizer_state_is_temporary": True,
            "optimizer_state_published": False,
        }
        if self.transaction.lexical_handle is not None:
            payload.update(
                {
                    "trainable_lexical_factor_parameters": (
                        self.transaction.lexical_handle.report.adapter_parameter_count
                    ),
                    "trainable_lexical_factor_tensor_count": 4,
                    "role_complete_lexical_boundary": True,
                }
            )
        payload["training_scope_sha256"] = _canonical_sha256(payload)
        return payload


class ExactTargetCompilerTransaction:
    """Atomic target-native QKVO, Gate/Up/Down, and optional lexical transaction."""

    def __init__(
        self,
        *,
        model: nn.Module,
        exact_unique_parameters: int,
        topology: _DenseGQATopology,
        native_before: NativeDenseGQACertificate,
        qkvo_handle: InstalledDenseGQAQKVOResidualAdapter,
        swiglu_handle: InstalledDenseSwiGLUProjectionResidualAdapter,
        stage: ProgressiveDenseGQAStage | None,
        parent_compiler_report_sha256: str | None,
        lexical_handle: InstalledDenseLexicalResidualAdapter | None = None,
    ) -> None:
        self.model = model
        self.exact_unique_parameters = exact_unique_parameters
        self.topology = topology
        self.native_before = native_before
        self.qkvo_handle = qkvo_handle
        self.swiglu_handle = swiglu_handle
        self.lexical_handle = lexical_handle
        self.stage = stage
        self.parent_compiler_report_sha256 = parent_compiler_report_sha256
        self.publication_report: dict[str, Any] | None = None
        self._zero_step_identity_at_installation = (
            qkvo_handle.adapter_registry.starts_as_exact_zero_correction
            and swiglu_handle.registry.starts_as_exact_zero_correction
            and (
                lexical_handle is None
                or lexical_handle.registry.starts_as_exact_zero_correction
            )
        )
        if not self._zero_step_identity_at_installation:
            raise ProgressiveTargetBridgeError(
                "Compiler transaction did not start as an exact identity."
            )

    @property
    def is_live(self) -> bool:
        return (
            self.qkvo_handle.is_applied
            and self.swiglu_handle.is_applied
            and (self.lexical_handle is None or self.lexical_handle.is_applied)
        )

    @property
    def starts_as_zero_step_identity(self) -> bool:
        return self._zero_step_identity_at_installation

    def assert_live(self) -> None:
        self.qkvo_handle.assert_live()
        self.swiglu_handle.assert_live()
        if self.lexical_handle is not None:
            self.lexical_handle.assert_live()

    def assert_structurally_live(self) -> None:
        """Validate transaction ownership while permitting poisoned factors."""

        self.qkvo_handle.assert_structurally_live()
        self.swiglu_handle.assert_structurally_live()
        if self.lexical_handle is not None:
            self.lexical_handle.assert_structurally_live()

    def output_trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        self.assert_live()
        values = (
            *self.qkvo_handle.output_trainable_parameters(),
            *self.swiglu_handle.output_trainable_parameters(),
        )
        if self.lexical_handle is not None:
            values = (*values, *self.lexical_handle.output_trainable_parameters())
        return values

    def output_training_scope(self) -> ExactTargetCompilerOutputTrainingScope:
        return ExactTargetCompilerOutputTrainingScope(self)

    def installation_report(self) -> dict[str, Any]:
        self.assert_live()
        qkvo = self.qkvo_handle.report
        swiglu = self.swiglu_handle.report
        lexical = None if self.lexical_handle is None else self.lexical_handle.report
        temporary = (
            qkvo.adapter_parameter_count
            + swiglu.adapter_parameter_count
            + (0 if lexical is None else lexical.adapter_parameter_count)
        )
        installed = exact_parameter_report(self.model)["unique_parameter_count"]
        if installed != self.exact_unique_parameters + temporary:
            raise ProgressiveTargetBridgeError(
                "Combined temporary factor accounting does not match the live model."
            )
        payload = {
            "schema": EXACT_TARGET_COMPILER_TRANSACTION_INSTALLATION_SCHEMA,
            "exact_native_parameters": self.exact_unique_parameters,
            "temporary_qkvo_factor_parameters": qkvo.adapter_parameter_count,
            "temporary_gate_up_down_factor_parameters": (
                swiglu.adapter_parameter_count
            ),
            "temporary_factor_parameters": temporary,
            "temporary_installed_parameters": installed,
            "published_parameters_if_promoted": self.exact_unique_parameters,
            "published_temporary_factor_parameters": 0,
            "zero_step_identity": self.starts_as_zero_step_identity,
            "native_before": self.native_before.as_dict(),
            "dense_qkvo_installation": qkvo.as_dict(),
            "dense_gate_up_down_installation": swiglu.as_dict(),
            "stage": None if self.stage is None else self.stage.as_dict(),
            "parent_compiler_report_sha256": self.parent_compiler_report_sha256,
            "role_complete_attention_qkvo": True,
            "role_complete_mlp_gate_up_down": True,
            "target_native_factors_only": True,
            "output_and_free_rollout_training_only": True,
            "internal_numeric_matching_required": False,
            "checkpoint_publication_allowed_while_installed": False,
            "capability_proven": False,
        }
        if lexical is not None:
            payload.update(
                {
                    "temporary_lexical_factor_parameters": (
                        lexical.adapter_parameter_count
                    ),
                    "dense_lexical_installation": lexical.as_dict(),
                    "role_complete_lexical_boundary": True,
                    "lexical_embedding_and_lm_head_untied": True,
                }
            )
        payload["installation_transaction_sha256"] = _canonical_sha256(payload)
        return payload

    @staticmethod
    def _enter_guided_owner_suspensions(
        stack: ExitStack,
        handles: Sequence[tuple[str, Any]],
    ) -> tuple[str, ...]:
        """Hide later guided owners while an older transaction restores its baseline."""

        entered: list[str] = []
        identities: set[int] = set()
        for name, handle in handles:
            if id(handle) in identities or not bool(
                getattr(handle, "is_applied", False)
            ):
                continue
            identities.add(id(handle))
            suspend = getattr(
                handle,
                "suspend_guided_registry_owner_for_nested_publication",
                None,
            )
            if callable(suspend):
                stack.enter_context(suspend())
                entered.append(name)
        return tuple(entered)

    def rollback(self) -> NativeDenseGQACertificate:
        if self.lexical_handle is not None and self.lexical_handle.is_applied:
            with ExitStack() as stack:
                self._enter_guided_owner_suspensions(
                    stack,
                    (
                        ("qkvo", self.qkvo_handle),
                        ("gate-up-down", self.swiglu_handle),
                    ),
                )
                self.lexical_handle.rollback()
        if self.qkvo_handle.is_applied:
            with ExitStack() as stack:
                self._enter_guided_owner_suspensions(
                    stack,
                    (("gate-up-down", self.swiglu_handle),),
                )
                self.qkvo_handle.rollback()
        if self.swiglu_handle.is_applied:
            self.swiglu_handle.rollback()
        restored = certify_native_dense_gqa_target(
            self.model,
            exact_unique_parameters=self.exact_unique_parameters,
            layout=self.topology.layout,
            stage=self.stage,
            require_dense_swiglu=True,
        )
        if restored.as_dict() != self.native_before.as_dict():
            raise ProgressiveTargetPublicationError(
                "Compiler rollback did not restore the exact native child."
            )
        return restored

    def _restore_failed_publication(self) -> NativeDenseGQACertificate:
        # A lexical bake is first on the success path.  If it never completed,
        # reject its live wrapper before the older transactions whose state-key
        # topology formed its installation baseline.
        if self.lexical_handle is not None and self.lexical_handle.is_applied:
            with ExitStack() as stack:
                self._enter_guided_owner_suspensions(
                    stack,
                    (
                        ("qkvo", self.qkvo_handle),
                        ("gate-up-down", self.swiglu_handle),
                    ),
                )
                self.lexical_handle.rollback()
        # QKVO is baked first.  If it is still live, reject it before removing
        # the MLP factors because its installation baseline includes them.
        if self.qkvo_handle.is_applied:
            with ExitStack() as stack:
                self._enter_guided_owner_suspensions(
                    stack,
                    (("gate-up-down", self.swiglu_handle),),
                )
                self.qkvo_handle.rollback()
            if self.swiglu_handle.is_applied:
                self.swiglu_handle.rollback()
        else:
            if self.swiglu_handle.is_baked:
                self.swiglu_handle.restore_prepublication_native()
            elif self.swiglu_handle.is_applied:
                self.swiglu_handle.rollback()
            if self.qkvo_handle.is_baked:
                self.qkvo_handle.restore_prepublication_native(
                    expected_model_parameter_count=self.exact_unique_parameters
                )
        # If lexical publication completed, restore it only after the older
        # temporary operators have returned the model to its final native
        # budget.  Its own installation baseline included both of them.
        if self.lexical_handle is not None and self.lexical_handle.is_baked:
            self.lexical_handle.restore_prepublication_native(
                expected_model_parameter_count=self.exact_unique_parameters
            )
        restored = certify_native_dense_gqa_target(
            self.model,
            exact_unique_parameters=self.exact_unique_parameters,
            layout=self.topology.layout,
            stage=self.stage,
            require_dense_swiglu=True,
        )
        if restored.as_dict() != self.native_before.as_dict():
            raise ProgressiveTargetPublicationError(
                "Failed publication could not restore the exact native child."
            )
        return restored

    def restore_prepublication_native(self) -> NativeDenseGQACertificate:
        """Undo live or already-baked factors after a later publication failure."""

        restored = self._restore_failed_publication()
        self.publication_report = None
        return restored

    def bake_for_publication(
        self,
        behavior_gate: ExternalBehaviorGateReceipt,
    ) -> dict[str, Any]:
        if not isinstance(behavior_gate, ExternalBehaviorGateReceipt):
            raise ProgressiveTargetPublicationError(
                "behavior_gate must be ExternalBehaviorGateReceipt."
            )
        behavior_gate.require_publication_approval()
        self.assert_live()
        try:
            if self.lexical_handle is None:
                lexical_bake = None
                lexical_suspensions: tuple[str, ...] = ()
            else:
                with ExitStack() as stack:
                    lexical_suspensions = self._enter_guided_owner_suspensions(
                        stack,
                        (
                            ("qkvo", self.qkvo_handle),
                            ("gate-up-down", self.swiglu_handle),
                        ),
                    )
                    lexical_bake = self.lexical_handle.bake_for_publication()
            with ExitStack() as stack:
                qkvo_suspensions = self._enter_guided_owner_suspensions(
                    stack,
                    (("gate-up-down", self.swiglu_handle),),
                )
                qkvo_bake = self.qkvo_handle.bake_for_publication()
            swiglu_bake = self.swiglu_handle.bake_for_publication()
            native_after = certify_native_dense_gqa_target(
                self.model,
                exact_unique_parameters=self.exact_unique_parameters,
                layout=self.topology.layout,
                stage=self.stage,
                require_dense_swiglu=True,
            )
            if (
                native_after.structure_sha256 != self.native_before.structure_sha256
                or native_after.state_key_sha256 != self.native_before.state_key_sha256
                or native_after.state_key_count != self.native_before.state_key_count
            ):
                raise ProgressiveTargetPublicationError(
                    "Combined bake did not restore native state-key topology."
                )
        except Exception:
            self._restore_failed_publication()
            raise
        try:
            payload = {
                "schema": EXACT_TARGET_COMPILER_TRANSACTION_PUBLICATION_SCHEMA,
                "exact_unique_parameters": self.exact_unique_parameters,
                "exact_budget_match": True,
                "native_before": self.native_before.as_dict(),
                "native_after": native_after.as_dict(),
                "dense_qkvo_bake": qkvo_bake.as_dict(),
                "dense_gate_up_down_bake": swiglu_bake.as_dict(),
                "external_behavior_gate": behavior_gate.as_dict(),
                "stage": None if self.stage is None else self.stage.as_dict(),
                "parent_compiler_report_sha256": self.parent_compiler_report_sha256,
                "published_adapter_parameter_count": 0,
                "published_optimizer_parameter_count": 0,
                "published_parent_runtime_reference_count": 0,
                "published_source_sized_sidecar_parameter_count": 0,
                "native_state_key_topology_restored": True,
                "output_action_gate_required": True,
                "free_running_rollout_gate_required": True,
                "weight_or_activation_similarity_can_promote": False,
                "behavior_gate_externally_attested": True,
                "behavior_evidence_revalidated_by_this_module": False,
                "capability_proven_by_this_module": False,
                "save_reload_artifact_gate_still_required": True,
            }
            if lexical_bake is not None:
                payload.update(
                    {
                        "dense_lexical_bake": lexical_bake.as_dict(),
                        "published_lexical_adapter_parameter_count": 0,
                        "native_lexical_state_key_topology_restored": True,
                        "role_complete_lexical_boundary": True,
                        "guided_registry_suspended_for_lexical_bake": (
                            "qkvo" in lexical_suspensions
                        ),
                    }
                )
            if "gate-up-down" in (*lexical_suspensions, *qkvo_suspensions):
                payload.update(
                    {
                        "guided_gate_up_down_registry_suspended_for_lexical_bake": (
                            "gate-up-down" in lexical_suspensions
                        ),
                        "guided_gate_up_down_registry_suspended_for_qkvo_bake": (
                            "gate-up-down" in qkvo_suspensions
                        ),
                        "nested_publication_bake_order": [
                            "lexical",
                            "qkvo-inherited-then-guided",
                            "gate-up-down-inherited-then-guided",
                        ],
                    }
                )
            payload["publication_transaction_sha256"] = _canonical_sha256(payload)
        except Exception:
            self._restore_failed_publication()
            raise
        self.publication_report = payload
        return payload

    def __enter__(self) -> ExactTargetCompilerTransaction:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        failed = exc_type is not None
        del exc_type, exc, traceback
        if self.is_live:
            self.rollback()
        elif failed and (
            self.qkvo_handle.is_baked
            or self.swiglu_handle.is_baked
            or (self.lexical_handle is not None and self.lexical_handle.is_baked)
        ):
            self.restore_prepublication_native()


def install_exact_target_qkvo_bridge(
    model: nn.Module,
    *,
    exact_unique_parameters: int,
    sites: Sequence[DenseGQASite],
    rank: int,
    layout: DecoderLayout | None = None,
    stage: ProgressiveDenseGQAStage | None = None,
    owner_name: str = "dense_gqa_qkvo_residual_registry",
    initialization_seed: int = 0,
    parent_compiler_report_sha256: str | None = None,
) -> ExactTargetQKVOPublicationSession:
    """Install temporary QKVO factors only after exact target certification."""

    expected = _positive_int(exact_unique_parameters, "exact_unique_parameters")
    if parent_compiler_report_sha256 is not None:
        parent_compiler_report_sha256 = _sha256(
            parent_compiler_report_sha256,
            "parent_compiler_report_sha256",
        )
    topology = _resolve_dense_gqa_topology(model, layout)
    native = certify_native_dense_gqa_target(
        model,
        exact_unique_parameters=expected,
        layout=topology.layout,
        stage=stage,
    )
    try:
        handle = install_dense_gqa_qkvo_residual_adapter(
            model,
            topology.attention_modules,
            geometry=topology.geometry,
            sites=sites,
            rank=rank,
            projection_names=topology.projection_names,
            owner_name=owner_name,
            initialization_seed=initialization_seed,
        )
    except DenseGQAQKVOResidualTopologyError as error:
        raise ProgressiveTargetBridgeError(str(error)) from error
    if handle.report.model_parameter_count_before != expected:
        handle.rollback()
        raise ProgressiveTargetBridgeError(
            "Dense adapter installation disagrees with exact target accounting."
        )
    return ExactTargetQKVOPublicationSession(
        model=model,
        exact_unique_parameters=expected,
        topology=topology,
        native_before=native,
        handle=handle,
        stage=stage,
        parent_compiler_report_sha256=parent_compiler_report_sha256,
    )


def install_exact_target_compiler_transaction(
    model: nn.Module,
    *,
    exact_unique_parameters: int,
    qkvo_sites: Sequence[DenseGQASite],
    qkvo_rank: int,
    swiglu_layer_indices: Sequence[int],
    swiglu_rank: int,
    lexical_rank: int | None = None,
    layout: DecoderLayout | None = None,
    stage: ProgressiveDenseGQAStage | None = None,
    qkvo_owner_name: str = "dense_gqa_qkvo_residual_registry",
    swiglu_owner_name: str = "dense_swiglu_projection_residual_registry",
    lexical_owner_name: str = "dense_lexical_residual_registry",
    initialization_seed: int = 0,
    parent_compiler_report_sha256: str | None = None,
) -> ExactTargetCompilerTransaction:
    """Open one exact-budget target-native compiler-factor transaction."""

    expected = _positive_int(exact_unique_parameters, "exact_unique_parameters")
    if parent_compiler_report_sha256 is not None:
        parent_compiler_report_sha256 = _sha256(
            parent_compiler_report_sha256,
            "parent_compiler_report_sha256",
        )
    topology = _resolve_dense_gqa_topology(
        model,
        layout,
        require_dense_swiglu=True,
    )
    native = certify_native_dense_gqa_target(
        model,
        exact_unique_parameters=expected,
        layout=topology.layout,
        stage=stage,
        require_dense_swiglu=True,
    )
    try:
        swiglu_handle = install_dense_swiglu_projection_residual_adapter(
            model,
            topology.mlp_modules,
            layer_indices=swiglu_layer_indices,
            hidden_size=topology.layout.hidden_size,
            intermediate_size=topology.layout.intermediate_size,
            rank=swiglu_rank,
            projection_names=topology.swiglu_projection_names,
            owner_name=swiglu_owner_name,
            initialization_seed=initialization_seed,
        )
    except DenseSwiGLUProjectionResidualTopologyError as error:
        raise ProgressiveTargetBridgeError(str(error)) from error
    try:
        qkvo_handle = install_dense_gqa_qkvo_residual_adapter(
            model,
            topology.attention_modules,
            geometry=topology.geometry,
            sites=qkvo_sites,
            rank=qkvo_rank,
            projection_names=topology.projection_names,
            owner_name=qkvo_owner_name,
            initialization_seed=initialization_seed + 1,
        )
        if (
            swiglu_handle.report.model_parameter_count_before != expected
            or qkvo_handle.report.model_parameter_count_before
            != swiglu_handle.report.model_parameter_count_installed
        ):
            raise ProgressiveTargetBridgeError(
                "Combined adapter installation disagrees with exact target accounting."
            )
        lexical_handle = None
        if lexical_rank is not None:
            lexical_handle = install_dense_lexical_residual_adapter(
                model,
                rank=lexical_rank,
                owner_name=lexical_owner_name,
                initialization_seed=initialization_seed + 2,
            )
            if (
                lexical_handle.report.model_parameter_count_before
                != qkvo_handle.report.model_parameter_count_installed
                or lexical_handle.report.model_parameter_count_installed
                != (
                    qkvo_handle.report.model_parameter_count_installed
                    + lexical_handle.report.adapter_parameter_count
                )
            ):
                raise ProgressiveTargetBridgeError(
                    "Lexical adapter installation disagrees with combined accounting."
                )
    except DenseLexicalResidualError as error:
        if "lexical_handle" in locals() and lexical_handle is not None:
            if lexical_handle.is_applied:
                lexical_handle.rollback()
        if "qkvo_handle" in locals() and qkvo_handle.is_applied:
            qkvo_handle.rollback()
        if swiglu_handle.is_applied:
            swiglu_handle.rollback()
        raise ProgressiveTargetBridgeError(str(error)) from error
    except Exception:
        if (
            "lexical_handle" in locals()
            and lexical_handle is not None
            and lexical_handle.is_applied
        ):
            lexical_handle.rollback()
        if "qkvo_handle" in locals() and qkvo_handle.is_applied:
            qkvo_handle.rollback()
        if swiglu_handle.is_applied:
            swiglu_handle.rollback()
        raise
    transaction = ExactTargetCompilerTransaction(
        model=model,
        exact_unique_parameters=expected,
        topology=topology,
        native_before=native,
        qkvo_handle=qkvo_handle,
        swiglu_handle=swiglu_handle,
        lexical_handle=lexical_handle,
        stage=stage,
        parent_compiler_report_sha256=parent_compiler_report_sha256,
    )
    try:
        transaction.installation_report()
        return transaction
    except Exception:
        if transaction.is_live:
            transaction.rollback()
        raise


def install_parent_compiler_result_qkvo_bridge(
    result: HFParentCompilerResult,
    *,
    sites: Sequence[DenseGQASite],
    rank: int,
    layout: DecoderLayout | None = None,
    stage: ProgressiveDenseGQAStage | None = None,
    owner_name: str = "dense_gqa_qkvo_residual_registry",
    initialization_seed: int = 0,
) -> ExactTargetQKVOPublicationSession:
    """Attach the native QKVO transaction to an accepted parent compilation."""

    if not isinstance(result, HFParentCompilerResult):
        raise ProgressiveTargetBridgeError("result must be HFParentCompilerResult.")
    report = result.report
    if (
        not report.target_budget_exact
        or not report.bootstrap_accepted
        or report.target_unique_parameter_count != report.target_parameter_budget
    ):
        raise ProgressiveTargetBridgeError(
            "Parent compiler result is not an accepted exact-budget child."
        )
    report_payload = report.as_dict()
    return install_exact_target_qkvo_bridge(
        result.child_model,
        exact_unique_parameters=report.target_unique_parameter_count,
        sites=sites,
        rank=rank,
        layout=layout,
        stage=stage,
        owner_name=owner_name,
        initialization_seed=initialization_seed,
        parent_compiler_report_sha256=_canonical_sha256(report_payload),
    )


def install_parent_compiler_result_target_native_transaction(
    result: HFParentCompilerResult,
    *,
    qkvo_sites: Sequence[DenseGQASite],
    qkvo_rank: int,
    swiglu_layer_indices: Sequence[int],
    swiglu_rank: int,
    lexical_rank: int | None = None,
    layout: DecoderLayout | None = None,
    stage: ProgressiveDenseGQAStage | None = None,
    qkvo_owner_name: str = "dense_gqa_qkvo_residual_registry",
    swiglu_owner_name: str = "dense_swiglu_projection_residual_registry",
    lexical_owner_name: str = "dense_lexical_residual_registry",
    initialization_seed: int = 0,
) -> ExactTargetCompilerTransaction:
    """Attach the full target-native transaction to an accepted compilation."""

    if not isinstance(result, HFParentCompilerResult):
        raise ProgressiveTargetBridgeError("result must be HFParentCompilerResult.")
    report = result.report
    if (
        not report.target_budget_exact
        or not report.bootstrap_accepted
        or report.target_unique_parameter_count != report.target_parameter_budget
    ):
        raise ProgressiveTargetBridgeError(
            "Parent compiler result is not an accepted exact-budget child."
        )
    return install_exact_target_compiler_transaction(
        result.child_model,
        exact_unique_parameters=report.target_unique_parameter_count,
        qkvo_sites=qkvo_sites,
        qkvo_rank=qkvo_rank,
        swiglu_layer_indices=swiglu_layer_indices,
        swiglu_rank=swiglu_rank,
        lexical_rank=lexical_rank,
        layout=layout,
        stage=stage,
        qkvo_owner_name=qkvo_owner_name,
        swiglu_owner_name=swiglu_owner_name,
        lexical_owner_name=lexical_owner_name,
        initialization_seed=initialization_seed,
        parent_compiler_report_sha256=_canonical_sha256(report.as_dict()),
    )


__all__ = [
    "EXACT_TARGET_QKVO_INSTALLATION_SCHEMA",
    "EXACT_TARGET_QKVO_PUBLICATION_SCHEMA",
    "EXACT_TARGET_COMPILER_OUTPUT_TRAINING_SCOPE_SCHEMA",
    "EXACT_TARGET_COMPILER_TRANSACTION_INSTALLATION_SCHEMA",
    "EXACT_TARGET_COMPILER_TRANSACTION_PUBLICATION_SCHEMA",
    "EXTERNAL_BEHAVIOR_GATE_SCHEMA",
    "NATIVE_TARGET_CERTIFICATE_SCHEMA",
    "PROGRESSIVE_TARGET_PLAN_SCHEMA",
    "QWEN3_1P4B_EXACT_PARAMETERS",
    "QWEN3_4B_EXACT_PARAMETERS",
    "QWEN3_600M_EXACT_PARAMETERS",
    "SMOLLM2_135M_EXACT_PARAMETERS",
    "DenseGQATargetShape",
    "ExactTargetCompilerOutputTrainingScope",
    "ExactTargetCompilerTransaction",
    "ExactTargetQKVOPublicationSession",
    "ExternalBehaviorGateReceipt",
    "NativeDenseGQACertificate",
    "ProgressiveDenseGQAPlan",
    "ProgressiveDenseGQAStage",
    "ProgressiveTargetBridgeError",
    "ProgressiveTargetPublicationError",
    "certify_native_dense_gqa_target",
    "default_qwen4b_progressive_target_plan",
    "install_exact_target_qkvo_bridge",
    "install_exact_target_compiler_transaction",
    "install_parent_compiler_result_qkvo_bridge",
    "install_parent_compiler_result_target_native_transaction",
]
