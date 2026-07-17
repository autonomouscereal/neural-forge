"""Activation-aware, zero-training low-rank factorization for LFM2 FFNs.

This module is deliberately different from cross-depth parameter sharing.  It
does not assume that two decoder layers use the same basis.  Each selected
``w1``, ``w3``, or ``w2`` projection is reconstructed in its own activation
geometry:

``argmin_rank(W_hat)<=r ||X (W - W_hat)^T||_F``.

``w1`` and ``w3`` use sealed layer-input rows.  ``w2`` uses sealed post-SwiGLU
rows.  A disjoint validation split chooses between the activation-weighted
solution and a raw-weight SVD guard, and allocates a global parameter budget by
held-out reconstruction benefit per additional physical parameter.

The installed runtime contains two ordinary bias-free Linear modules.  There
is no retained dense weight, fake sharing, optimizer, backward pass, gradient,
or training step.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
import torch.nn.functional as F


EVIDENCE_SCHEMA = "neural-forge-lfm2-activation-factorization-evidence-v1"
PLAN_SCHEMA = "neural-forge-lfm2-activation-factorization-plan-v1"
REPORT_SCHEMA = "neural-forge-lfm2-activation-factorization-report-v1"
METHOD = "per-module-activation-weighted-low-rank-factorization"
_ROLES = ("w1", "w3", "w2")
_TRANSACTION_LOCK = threading.RLock()


class Lfm2ActivationFactorizationError(RuntimeError):
    """Base error for activation-aware LFM2 factorization."""


class Lfm2ActivationFactorizationEvidenceError(
    Lfm2ActivationFactorizationError
):
    """Activation evidence is malformed, changed, or stale."""


class Lfm2ActivationFactorizationTopologyError(
    Lfm2ActivationFactorizationError
):
    """The model or requested factor topology cannot be proven safe."""


class Lfm2ActivationFactorizationTransactionError(
    Lfm2ActivationFactorizationError
):
    """Planning, installation, accounting, or rollback failed closed."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Lfm2ActivationFactorizationTransactionError(
            "Factorization metadata is not canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Lfm2ActivationFactorizationTransactionError(
                "Model configuration contains a non-finite float."
            )
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _tensor_sha256(value: torch.Tensor, *, domain: bytes) -> str:
    if not isinstance(value, torch.Tensor):
        raise Lfm2ActivationFactorizationTransactionError(
            "A sealed tensor payload is not a Tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(b"\0")
    digest.update(str(tensor.dtype).replace("torch.", "").encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json([int(size) for size in tensor.shape]))
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tensor_record(
    value: torch.Tensor,
    name: str,
    *,
    domain: bytes,
) -> dict[str, Any]:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.numel() == 0
        or not bool(torch.isfinite(value).all())
    ):
        raise Lfm2ActivationFactorizationEvidenceError(
            f"{name} must be a nonempty finite rank-2 Tensor."
        )
    return {
        "name": name,
        "shape": [int(size) for size in value.shape],
        "dtype": str(value.dtype).replace("torch.", ""),
        "sha256": _tensor_sha256(value, domain=domain),
    }


def _clone_cpu(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise Lfm2ActivationFactorizationEvidenceError(
            f"{name} must be a Tensor."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    _tensor_record(result, name, domain=b"lfm2-activation-evidence-v1")
    return result


def _model_config(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "lfm2":
        raise Lfm2ActivationFactorizationTopologyError(
            "The model must expose config.model_type == 'lfm2'."
        )
    if hasattr(config, "to_dict"):
        raw = config.to_dict()
    else:
        raw = vars(config)
    if not isinstance(raw, Mapping):
        raise Lfm2ActivationFactorizationTopologyError(
            "The LFM2 configuration is not serializable."
        )
    return _json_safe(raw)


def _model_identity(model: Any) -> str:
    config = _model_config(model)
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-exact-model-v1\0")
    digest.update(
        f"{type(model).__module__}.{type(model).__qualname__}".encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(_canonical_json(config))
    for namespace, values in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, value in values:
            digest.update(b"\0")
            digest.update(namespace.encode("ascii"))
            digest.update(b"\0")
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(
                _tensor_sha256(
                    value,
                    domain=b"lfm2-exact-model-tensor-v1",
                ).encode("ascii")
            )
    return digest.hexdigest()


def _locate_layers(model: Any) -> tuple[Any, ...]:
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
    )
    for candidate in candidates:
        if isinstance(candidate, (torch.nn.ModuleList, list, tuple)) and candidate:
            layers = tuple(candidate)
            if all(isinstance(layer, torch.nn.Module) for layer in layers):
                return layers
    raise Lfm2ActivationFactorizationTopologyError(
        "Could not locate a nonempty LFM2 decoder layer sequence."
    )


@dataclass(frozen=True, slots=True)
class _DenseRole:
    layer_index: int
    role: str
    owner: Any
    module: torch.nn.Linear

    @property
    def input_features(self) -> int:
        return int(self.module.in_features)

    @property
    def output_features(self) -> int:
        return int(self.module.out_features)

    @property
    def dense_parameters(self) -> int:
        return int(self.module.weight.numel())

    @property
    def increment_cost(self) -> int:
        return self.input_features + self.output_features

    @property
    def key(self) -> tuple[int, int]:
        return self.layer_index, _ROLES.index(self.role)


def _inspect_dense_roles(model: Any) -> tuple[tuple[Any, ...], tuple[_DenseRole, ...]]:
    _model_config(model)
    layers = _locate_layers(model)
    roles: list[_DenseRole] = []
    seen_ffns: set[int] = set()
    for layer_index, layer in enumerate(layers):
        ffn = getattr(layer, "feed_forward", None)
        if not isinstance(ffn, torch.nn.Module) or id(ffn) in seen_ffns:
            raise Lfm2ActivationFactorizationTopologyError(
                "Every decoder layer must have an independent feed_forward module."
            )
        seen_ffns.add(id(ffn))
        modules = {role: getattr(ffn, role, None) for role in _ROLES}
        if any(
            not isinstance(module, torch.nn.Linear)
            or module.bias is not None
            for module in modules.values()
        ):
            raise Lfm2ActivationFactorizationTopologyError(
                "LFM2 w1/w3/w2 must be independent bias-free dense Linear modules."
            )
        w1 = modules["w1"]
        w3 = modules["w3"]
        w2 = modules["w2"]
        assert isinstance(w1, torch.nn.Linear)
        assert isinstance(w3, torch.nn.Linear)
        assert isinstance(w2, torch.nn.Linear)
        hidden = int(w1.in_features)
        intermediate = int(w1.out_features)
        if (
            w3.in_features != hidden
            or w3.out_features != intermediate
            or w2.in_features != intermediate
            or w2.out_features != hidden
        ):
            raise Lfm2ActivationFactorizationTopologyError(
                "LFM2 SwiGLU projection dimensions are inconsistent."
            )
        for role in _ROLES:
            module = modules[role]
            assert isinstance(module, torch.nn.Linear)
            if (
                module.weight.ndim != 2
                or not bool(torch.isfinite(module.weight).all())
            ):
                raise Lfm2ActivationFactorizationTopologyError(
                    f"Layer {layer_index} {role} weight is not finite rank-2."
                )
            roles.append(
                _DenseRole(
                    layer_index=layer_index,
                    role=role,
                    owner=ffn,
                    module=module,
                )
            )
    return layers, tuple(roles)


def _parameter_counts(model: Any) -> tuple[int, int]:
    named = tuple(model.named_parameters(remove_duplicate=False))
    logical = sum(int(parameter.numel()) for _, parameter in named)
    unique = {id(parameter): parameter for _, parameter in named}
    return logical, sum(int(parameter.numel()) for parameter in unique.values())


@dataclass(frozen=True, slots=True)
class Lfm2ActivationFactorizationLayerEvidence:
    """Disjoint fit/validation activation rows for one LFM2 layer."""

    layer_index: int
    fit_layer_inputs: Any
    validation_layer_inputs: Any
    fit_post_swiglu: Any
    validation_post_swiglu: Any
    fit_layer_inputs_sha256: str
    validation_layer_inputs_sha256: str
    fit_post_swiglu_sha256: str
    validation_post_swiglu_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "fit_layer_inputs": _tensor_record(
                self.fit_layer_inputs,
                f"layer[{self.layer_index}].fit_layer_inputs",
                domain=b"lfm2-activation-evidence-v1",
            ),
            "validation_layer_inputs": _tensor_record(
                self.validation_layer_inputs,
                f"layer[{self.layer_index}].validation_layer_inputs",
                domain=b"lfm2-activation-evidence-v1",
            ),
            "fit_post_swiglu": _tensor_record(
                self.fit_post_swiglu,
                f"layer[{self.layer_index}].fit_post_swiglu",
                domain=b"lfm2-activation-evidence-v1",
            ),
            "validation_post_swiglu": _tensor_record(
                self.validation_post_swiglu,
                f"layer[{self.layer_index}].validation_post_swiglu",
                domain=b"lfm2-activation-evidence-v1",
            ),
        }


@dataclass(frozen=True, slots=True)
class Lfm2ActivationFactorizationEvidence:
    """Exact-parent-bound calibration evidence with an untouched validation split."""

    schema: str
    parent_identity_sha256: str
    cache_key: str
    heldout_corpus_sha256: str
    fit_row_ids: tuple[str, ...]
    validation_row_ids: tuple[str, ...]
    layers: tuple[Lfm2ActivationFactorizationLayerEvidence, ...]
    evidence_sha256: str

    def layer(self, layer_index: int) -> Lfm2ActivationFactorizationLayerEvidence:
        for value in self.layers:
            if value.layer_index == layer_index:
                return value
        raise Lfm2ActivationFactorizationEvidenceError(
            f"Evidence has no layer {layer_index}."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "parent_identity_sha256": self.parent_identity_sha256,
            "cache_key": self.cache_key,
            "heldout_corpus_sha256": self.heldout_corpus_sha256,
            "fit_row_ids": list(self.fit_row_ids),
            "validation_row_ids": list(self.validation_row_ids),
            "layers": [layer.as_dict() for layer in self.layers],
            "evidence_sha256": self.evidence_sha256,
        }


def _evidence_body(
    evidence: Lfm2ActivationFactorizationEvidence,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in evidence.as_dict().items()
        if key != "evidence_sha256"
    }


def _normalize_row_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise Lfm2ActivationFactorizationEvidenceError(
            f"{name} must be a sequence of row IDs."
        )
    result = tuple(values)
    if (
        not result
        or any(not isinstance(value, str) or not value for value in result)
        or len(set(result)) != len(result)
    ):
        raise Lfm2ActivationFactorizationEvidenceError(
            f"{name} must contain unique nonempty strings."
        )
    return result


def seal_lfm2_activation_factorization_evidence(
    model: Any,
    fit_layer_inputs: Mapping[int, Any],
    validation_layer_inputs: Mapping[int, Any],
    fit_post_swiglu: Mapping[int, Any],
    validation_post_swiglu: Mapping[int, Any],
    *,
    fit_row_ids: Sequence[str],
    validation_row_ids: Sequence[str],
    cache_key: str,
    heldout_corpus_sha256: str,
) -> Lfm2ActivationFactorizationEvidence:
    """Clone and seal disjoint activation rows against the exact live parent."""

    with _TRANSACTION_LOCK, torch.no_grad():
        _layers, dense_roles = _inspect_dense_roles(model)
        if not _valid_sha256(cache_key) or not _valid_sha256(
            heldout_corpus_sha256
        ):
            raise Lfm2ActivationFactorizationEvidenceError(
                "cache_key and heldout_corpus_sha256 must be SHA-256 strings."
            )
        fit_ids = _normalize_row_ids(fit_row_ids, "fit_row_ids")
        validation_ids = _normalize_row_ids(
            validation_row_ids,
            "validation_row_ids",
        )
        if set(fit_ids).intersection(validation_ids):
            raise Lfm2ActivationFactorizationEvidenceError(
                "Fit and validation row IDs must be disjoint."
            )
        mappings = (
            fit_layer_inputs,
            validation_layer_inputs,
            fit_post_swiglu,
            validation_post_swiglu,
        )
        if any(not isinstance(value, Mapping) for value in mappings):
            raise Lfm2ActivationFactorizationEvidenceError(
                "Activation collections must be layer-index mappings."
            )
        key_sets = tuple(set(value) for value in mappings)
        if (
            not key_sets[0]
            or any(keys != key_sets[0] for keys in key_sets[1:])
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in key_sets[0]
            )
        ):
            raise Lfm2ActivationFactorizationEvidenceError(
                "All activation mappings must contain the same valid layer indices."
            )
        role_lookup = {
            (role.layer_index, role.role): role for role in dense_roles
        }
        layer_values: list[Lfm2ActivationFactorizationLayerEvidence] = []
        for layer_index in sorted(key_sets[0]):
            if (layer_index, "w1") not in role_lookup:
                raise Lfm2ActivationFactorizationEvidenceError(
                    f"Activation evidence references absent layer {layer_index}."
                )
            hidden = role_lookup[(layer_index, "w1")].input_features
            intermediate = role_lookup[(layer_index, "w2")].input_features
            fit_inputs = _clone_cpu(
                fit_layer_inputs[layer_index],
                f"layer[{layer_index}].fit_layer_inputs",
            )
            validation_inputs = _clone_cpu(
                validation_layer_inputs[layer_index],
                f"layer[{layer_index}].validation_layer_inputs",
            )
            fit_swiglu = _clone_cpu(
                fit_post_swiglu[layer_index],
                f"layer[{layer_index}].fit_post_swiglu",
            )
            validation_swiglu = _clone_cpu(
                validation_post_swiglu[layer_index],
                f"layer[{layer_index}].validation_post_swiglu",
            )
            if (
                fit_inputs.shape != (len(fit_ids), hidden)
                or validation_inputs.shape
                != (len(validation_ids), hidden)
                or fit_swiglu.shape != (len(fit_ids), intermediate)
                or validation_swiglu.shape
                != (len(validation_ids), intermediate)
            ):
                raise Lfm2ActivationFactorizationEvidenceError(
                    f"Layer {layer_index} activation shapes do not match the "
                    "declared row splits and FFN dimensions."
                )
            hashes = tuple(
                _tensor_sha256(value, domain=b"lfm2-activation-evidence-v1")
                for value in (
                    fit_inputs,
                    validation_inputs,
                    fit_swiglu,
                    validation_swiglu,
                )
            )
            layer_values.append(
                Lfm2ActivationFactorizationLayerEvidence(
                    layer_index=layer_index,
                    fit_layer_inputs=fit_inputs,
                    validation_layer_inputs=validation_inputs,
                    fit_post_swiglu=fit_swiglu,
                    validation_post_swiglu=validation_swiglu,
                    fit_layer_inputs_sha256=hashes[0],
                    validation_layer_inputs_sha256=hashes[1],
                    fit_post_swiglu_sha256=hashes[2],
                    validation_post_swiglu_sha256=hashes[3],
                )
            )
        evidence = Lfm2ActivationFactorizationEvidence(
            schema=EVIDENCE_SCHEMA,
            parent_identity_sha256=_model_identity(model),
            cache_key=cache_key,
            heldout_corpus_sha256=heldout_corpus_sha256,
            fit_row_ids=fit_ids,
            validation_row_ids=validation_ids,
            layers=tuple(layer_values),
            evidence_sha256="",
        )
        return replace(
            evidence,
            evidence_sha256=_json_sha256(_evidence_body(evidence)),
        )


def _validate_evidence(
    model: Any,
    evidence: Lfm2ActivationFactorizationEvidence,
) -> tuple[tuple[Any, ...], tuple[_DenseRole, ...]]:
    if (
        not isinstance(evidence, Lfm2ActivationFactorizationEvidence)
        or evidence.schema != EVIDENCE_SCHEMA
        or not _valid_sha256(evidence.evidence_sha256)
        or not _valid_sha256(evidence.parent_identity_sha256)
        or not _valid_sha256(evidence.cache_key)
        or not _valid_sha256(evidence.heldout_corpus_sha256)
    ):
        raise Lfm2ActivationFactorizationEvidenceError(
            "A sealed activation-factorization evidence object is required."
        )
    fit_ids = _normalize_row_ids(evidence.fit_row_ids, "fit_row_ids")
    validation_ids = _normalize_row_ids(
        evidence.validation_row_ids,
        "validation_row_ids",
    )
    if set(fit_ids).intersection(validation_ids):
        raise Lfm2ActivationFactorizationEvidenceError(
            "Fit and validation evidence is no longer disjoint."
        )
    if _json_sha256(_evidence_body(evidence)) != evidence.evidence_sha256:
        raise Lfm2ActivationFactorizationEvidenceError(
            "Sealed activation evidence changed after capture."
        )
    for layer in evidence.layers:
        values_and_hashes = (
            (layer.fit_layer_inputs, layer.fit_layer_inputs_sha256),
            (
                layer.validation_layer_inputs,
                layer.validation_layer_inputs_sha256,
            ),
            (layer.fit_post_swiglu, layer.fit_post_swiglu_sha256),
            (
                layer.validation_post_swiglu,
                layer.validation_post_swiglu_sha256,
            ),
        )
        if any(
            _tensor_sha256(value, domain=b"lfm2-activation-evidence-v1")
            != expected
            for value, expected in values_and_hashes
        ):
            raise Lfm2ActivationFactorizationEvidenceError(
                "A sealed activation tensor changed after capture."
            )
    layers, roles = _inspect_dense_roles(model)
    if _model_identity(model) != evidence.parent_identity_sha256:
        raise Lfm2ActivationFactorizationEvidenceError(
            "Activation evidence is stale for the exact live parent."
        )
    return layers, roles


def _canonicalize_svd(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    components = int(right.shape[0])
    if components == 0:
        return left, right
    maxima = right.abs().argmax(dim=1)
    rows = torch.arange(components, device=right.device)
    pivots = right[rows, maxima]
    signs = torch.where(
        pivots < 0,
        -torch.ones_like(pivots),
        torch.ones_like(pivots),
    )
    return left * signs.unsqueeze(0), right * signs.unsqueeze(1)


def _relative_squared_error(target: torch.Tensor, observed: torch.Tensor) -> float:
    residual = float((target - observed).square().sum().item())
    denominator = float(target.square().sum().item())
    value = residual / max(denominator, torch.finfo(torch.float64).tiny)
    if not math.isfinite(value) or value < 0:
        raise Lfm2ActivationFactorizationTransactionError(
            "Reconstruction error is invalid."
        )
    return value


def _factor_compute_dtype(value: torch.Tensor) -> torch.dtype:
    # Float64 keeps the exactness proofs honest for float64 toy/reference
    # matrices.  Real LFM2 checkpoints are normally float32/bfloat16/float16;
    # solving those in float32 avoids a 2x storage and large CPU-SVD penalty
    # without claiming precision that the installed factors cannot retain.
    return torch.float64 if value.dtype == torch.float64 else torch.float32


def _raw_svd_factors(
    weight: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    left, singular, right = torch.linalg.svd(weight, full_matrices=False)
    left, right = _canonicalize_svd(left, right)
    roots = singular[:rank].clamp_min(0).sqrt()
    output_factor = left[:, :rank] * roots.unsqueeze(0)
    input_factor = roots.unsqueeze(1) * right[:rank, :]
    return input_factor.contiguous(), output_factor.contiguous()


def _activation_factors(
    weight: torch.Tensor,
    fit_inputs: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fit_outputs = fit_inputs @ weight.transpose(0, 1)
    left, singular, right = torch.linalg.svd(
        fit_outputs,
        full_matrices=False,
    )
    left, right = _canonicalize_svd(left, right)
    components = min(rank, int(singular.numel()))
    roots = singular[:components].clamp_min(0).sqrt()
    target_left = left[:, :components] * roots.unsqueeze(0)
    input_transpose = torch.linalg.pinv(fit_inputs) @ target_left
    input_factor = input_transpose.transpose(0, 1).contiguous()
    output_factor = (
        right[:components, :].transpose(0, 1) * roots.unsqueeze(0)
    ).contiguous()
    if components < rank:
        input_factor = torch.cat(
            (
                input_factor,
                torch.zeros(
                    rank - components,
                    int(weight.shape[1]),
                    dtype=weight.dtype,
                ),
            ),
            dim=0,
        )
        output_factor = torch.cat(
            (
                output_factor,
                torch.zeros(
                    int(weight.shape[0]),
                    rank - components,
                    dtype=weight.dtype,
                ),
            ),
            dim=1,
        )
    return input_factor, output_factor


@dataclass(frozen=True, slots=True)
class Lfm2LinearFactorization:
    """Sealed two-factor approximation for one bias-free dense matrix."""

    matrix_shape: tuple[int, int]
    rank: int
    method: Literal["activation_weighted", "raw_svd_validation_guard"]
    input_factor: Any
    output_factor: Any
    input_factor_sha256: str
    output_factor_sha256: str
    fit_relative_rmse: float
    validation_relative_rmse: float
    raw_svd_validation_relative_rmse: float

    @property
    def dense_parameters(self) -> int:
        return self.matrix_shape[0] * self.matrix_shape[1]

    @property
    def factor_parameters(self) -> int:
        return self.rank * sum(self.matrix_shape)

    @property
    def parameters_saved(self) -> int:
        return self.dense_parameters - self.factor_parameters

    def effective_weight(self) -> torch.Tensor:
        return self.output_factor @ self.input_factor

    def as_dict(self) -> dict[str, Any]:
        return {
            "matrix_shape": list(self.matrix_shape),
            "rank": self.rank,
            "method": self.method,
            "dense_parameters": self.dense_parameters,
            "factor_parameters": self.factor_parameters,
            "parameters_saved": self.parameters_saved,
            "fit_relative_rmse": self.fit_relative_rmse,
            "validation_relative_rmse": self.validation_relative_rmse,
            "raw_svd_validation_relative_rmse": (
                self.raw_svd_validation_relative_rmse
            ),
            "input_factor": {
                "shape": [int(size) for size in self.input_factor.shape],
                "dtype": str(self.input_factor.dtype).replace("torch.", ""),
                "sha256": self.input_factor_sha256,
            },
            "output_factor": {
                "shape": [int(size) for size in self.output_factor.shape],
                "dtype": str(self.output_factor.dtype).replace("torch.", ""),
                "sha256": self.output_factor_sha256,
            },
        }


def factorize_activation_weighted_linear(
    weight: Any,
    fit_inputs: Any,
    validation_inputs: Any,
    *,
    rank: int,
) -> Lfm2LinearFactorization:
    """Solve a deterministic activation-weighted rank-r linear approximation.

    The fit split constructs the reduced-rank regression.  The validation split
    fail-safes it against raw truncated SVD; therefore the returned held-out
    reconstruction error cannot exceed the raw-SVD baseline on the same rows.
    """

    with torch.no_grad():
        if (
            not isinstance(weight, torch.Tensor)
            or weight.ndim != 2
            or not bool(torch.isfinite(weight).all())
        ):
            raise Lfm2ActivationFactorizationTopologyError(
                "weight must be a finite rank-2 Tensor."
            )
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank <= 0
            or rank > min(int(weight.shape[0]), int(weight.shape[1]))
        ):
            raise Lfm2ActivationFactorizationTopologyError(
                "rank must be within the full matrix rank."
            )
        compute_dtype = _factor_compute_dtype(weight)
        fit = _clone_cpu(fit_inputs, "fit_inputs").to(dtype=compute_dtype)
        validation = _clone_cpu(
            validation_inputs,
            "validation_inputs",
        ).to(dtype=compute_dtype)
        if (
            int(fit.shape[1]) != int(weight.shape[1])
            or int(validation.shape[1]) != int(weight.shape[1])
        ):
            raise Lfm2ActivationFactorizationTopologyError(
                "Activation feature dimension does not match the matrix input."
            )
        compute_weight = (
            weight.detach().to(device="cpu", dtype=compute_dtype).contiguous()
        )
        activation_input, activation_output = _activation_factors(
            compute_weight,
            fit,
            rank,
        )
        raw_input, raw_output = _raw_svd_factors(compute_weight, rank)
        original_dtype = weight.dtype

        def converted(
            input_factor: torch.Tensor,
            output_factor: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            first = input_factor.to(dtype=original_dtype).contiguous()
            second = output_factor.to(dtype=original_dtype).contiguous()
            if not bool(torch.isfinite(first).all()) or not bool(
                torch.isfinite(second).all()
            ):
                raise Lfm2ActivationFactorizationTransactionError(
                    "Factor dtype conversion produced a non-finite Tensor."
                )
            return first, second

        activation_input, activation_output = converted(
            activation_input,
            activation_output,
        )
        raw_input, raw_output = converted(raw_input, raw_output)
        target_fit = fit @ compute_weight.transpose(0, 1)
        target_validation = validation @ compute_weight.transpose(0, 1)

        def errors(
            first: torch.Tensor,
            second: torch.Tensor,
        ) -> tuple[float, float]:
            effective = second.to(dtype=compute_dtype) @ first.to(
                dtype=compute_dtype
            )
            fit_error = _relative_squared_error(
                target_fit,
                fit @ effective.transpose(0, 1),
            )
            validation_error = _relative_squared_error(
                target_validation,
                validation @ effective.transpose(0, 1),
            )
            return math.sqrt(fit_error), math.sqrt(validation_error)

        activation_fit_error, activation_validation_error = errors(
            activation_input,
            activation_output,
        )
        raw_fit_error, raw_validation_error = errors(raw_input, raw_output)
        tolerance = 16.0 * torch.finfo(torch.float64).eps
        if activation_validation_error <= raw_validation_error + tolerance:
            method: Literal[
                "activation_weighted",
                "raw_svd_validation_guard",
            ] = "activation_weighted"
            selected_input = activation_input
            selected_output = activation_output
            fit_error = activation_fit_error
            validation_error = activation_validation_error
        else:
            method = "raw_svd_validation_guard"
            selected_input = raw_input
            selected_output = raw_output
            fit_error = raw_fit_error
            validation_error = raw_validation_error
        result = Lfm2LinearFactorization(
            matrix_shape=(int(weight.shape[0]), int(weight.shape[1])),
            rank=rank,
            method=method,
            input_factor=selected_input,
            output_factor=selected_output,
            input_factor_sha256=_tensor_sha256(
                selected_input,
                domain=b"lfm2-activation-factor-v1",
            ),
            output_factor_sha256=_tensor_sha256(
                selected_output,
                domain=b"lfm2-activation-factor-v1",
            ),
            fit_relative_rmse=fit_error,
            validation_relative_rmse=validation_error,
            raw_svd_validation_relative_rmse=raw_validation_error,
        )
        return result


def _activation_validation_curve(
    weight: torch.Tensor,
    fit_inputs: torch.Tensor,
    validation_inputs: torch.Tensor,
    max_rank: int,
) -> tuple[float, ...]:
    compute_dtype = _factor_compute_dtype(weight)
    compute_weight = (
        weight.detach().to(device="cpu", dtype=compute_dtype).contiguous()
    )
    fit = fit_inputs.detach().to(device="cpu", dtype=compute_dtype)
    validation = validation_inputs.detach().to(
        device="cpu",
        dtype=compute_dtype,
    )
    input_factor, output_factor = _activation_factors(
        compute_weight,
        fit,
        max_rank,
    )
    target = validation @ compute_weight.transpose(0, 1)
    prediction = torch.zeros_like(target)
    denominator = max(
        float(target.square().sum().item()),
        torch.finfo(torch.float64).tiny,
    )
    values = [float("inf")]
    previous = float("inf")
    projected = validation @ input_factor.transpose(0, 1)
    for component in range(max_rank):
        prediction.add_(
            projected[:, component : component + 1]
            * output_factor[:, component].unsqueeze(0)
        )
        error = float((target - prediction).square().sum().item()) / denominator
        if not math.isfinite(error) or error < 0:
            raise Lfm2ActivationFactorizationTransactionError(
                "Activation-weighted validation curve is invalid."
            )
        # Rounding can make an otherwise nested low-rank curve move upward by
        # a few ulps.  Rank allocation uses the monotone held-out envelope.
        previous = min(previous, error)
        values.append(previous)
    return tuple(values)


@dataclass(frozen=True, slots=True)
class Lfm2RankAllocationStep:
    """One deterministic validation-benefit-per-parameter rank increment."""

    layer_index: int
    role: str
    rank_before: int
    rank_after: int
    parameter_cost: int
    validation_error_reduction: float
    benefit_per_parameter: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "role": self.role,
            "rank_before": self.rank_before,
            "rank_after": self.rank_after,
            "parameter_cost": self.parameter_cost,
            "validation_error_reduction": self.validation_error_reduction,
            "benefit_per_parameter": self.benefit_per_parameter,
        }


@dataclass(frozen=True, slots=True)
class Lfm2ActivationRolePlan:
    """Selected rank and sealed factor payload for one FFN projection."""

    layer_index: int
    role: str
    factorization: Lfm2LinearFactorization
    source_weight_sha256: str
    requires_grad: bool

    @property
    def rank(self) -> int:
        return self.factorization.rank

    @property
    def dense_parameters(self) -> int:
        return self.factorization.dense_parameters

    @property
    def factor_parameters(self) -> int:
        return self.factorization.factor_parameters

    @property
    def parameters_saved(self) -> int:
        return self.factorization.parameters_saved

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "role": self.role,
            "source_weight_sha256": self.source_weight_sha256,
            "requires_grad": self.requires_grad,
            "factorization": self.factorization.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class Lfm2ActivationFactorizationPlan:
    """Sealed exact-budget plan for per-module activation factorization."""

    schema: str
    parent_identity_sha256: str
    evidence_sha256: str
    target_unique_parameter_budget: int
    logical_parameters_before: int
    unique_parameters_before: int
    projected_logical_parameters_after: int
    projected_unique_parameters_after: int
    projected_parameters_saved: int
    role_plans: tuple[Lfm2ActivationRolePlan, ...]
    allocation_steps: tuple[Lfm2RankAllocationStep, ...]
    integer_metadata_values: int
    integer_tensor_count: int
    integer_tensor_bytes: int
    plan_sha256: str
    training_steps: int = 0
    gradients_used: bool = False

    @property
    def selected_roles(self) -> tuple[tuple[int, str], ...]:
        return tuple(
            (value.layer_index, value.role) for value in self.role_plans
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "method": METHOD,
            "parent_identity_sha256": self.parent_identity_sha256,
            "evidence_sha256": self.evidence_sha256,
            "target_unique_parameter_budget": (
                self.target_unique_parameter_budget
            ),
            "logical_parameters_before": self.logical_parameters_before,
            "unique_parameters_before": self.unique_parameters_before,
            "projected_logical_parameters_after": (
                self.projected_logical_parameters_after
            ),
            "projected_unique_parameters_after": (
                self.projected_unique_parameters_after
            ),
            "projected_parameters_saved": self.projected_parameters_saved,
            "selected_roles": [
                [layer_index, role]
                for layer_index, role in self.selected_roles
            ],
            "role_plans": [value.as_dict() for value in self.role_plans],
            "allocation_steps": [
                value.as_dict() for value in self.allocation_steps
            ],
            "integer_metadata_values": self.integer_metadata_values,
            "integer_tensor_count": self.integer_tensor_count,
            "integer_tensor_bytes": self.integer_tensor_bytes,
            "training_steps": self.training_steps,
            "gradients_used": self.gradients_used,
            "plan_sha256": self.plan_sha256,
        }


def _plan_body(plan: Lfm2ActivationFactorizationPlan) -> dict[str, Any]:
    return {
        key: value
        for key, value in plan.as_dict().items()
        if key != "plan_sha256"
    }


def _seal_plan(
    plan: Lfm2ActivationFactorizationPlan,
) -> Lfm2ActivationFactorizationPlan:
    return replace(plan, plan_sha256=_json_sha256(_plan_body(plan)))


def _validate_plan_seal(plan: Lfm2ActivationFactorizationPlan) -> None:
    for value in plan.role_plans:
        factorization = value.factorization
        if (
            _tensor_sha256(
                factorization.input_factor,
                domain=b"lfm2-activation-factor-v1",
            )
            != factorization.input_factor_sha256
            or _tensor_sha256(
                factorization.output_factor,
                domain=b"lfm2-activation-factor-v1",
            )
            != factorization.output_factor_sha256
        ):
            raise Lfm2ActivationFactorizationTransactionError(
                "A sealed activation factor payload changed after planning."
            )
    if _json_sha256(_plan_body(plan)) != plan.plan_sha256:
        raise Lfm2ActivationFactorizationTransactionError(
            "The sealed activation-factorization plan changed after planning."
        )


def _normalize_selection(
    evidence: Lfm2ActivationFactorizationEvidence,
    dense_roles: tuple[_DenseRole, ...],
    layer_indices: Sequence[int] | None,
    roles: Sequence[str],
) -> tuple[_DenseRole, ...]:
    available_layers = {value.layer_index for value in evidence.layers}
    if layer_indices is None:
        selected_layers = tuple(sorted(available_layers))
    elif isinstance(layer_indices, (str, bytes)) or not isinstance(
        layer_indices,
        Sequence,
    ):
        raise Lfm2ActivationFactorizationTopologyError(
            "layer_indices must be a sequence of integers."
        )
    else:
        selected_layers = tuple(layer_indices)
    if (
        not selected_layers
        or len(set(selected_layers)) != len(selected_layers)
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in available_layers
            for index in selected_layers
        )
    ):
        raise Lfm2ActivationFactorizationTopologyError(
            "Selected layers must be unique and present in the evidence."
        )
    if isinstance(roles, (str, bytes)) or not isinstance(roles, Sequence):
        raise Lfm2ActivationFactorizationTopologyError(
            "roles must be a sequence drawn from w1/w3/w2."
        )
    selected_role_names = tuple(roles)
    if (
        not selected_role_names
        or len(set(selected_role_names)) != len(selected_role_names)
        or any(role not in _ROLES for role in selected_role_names)
    ):
        raise Lfm2ActivationFactorizationTopologyError(
            "roles must be unique values drawn from w1/w3/w2."
        )
    role_set = set(selected_role_names)
    layer_set = set(selected_layers)
    selected = tuple(
        value
        for value in dense_roles
        if value.layer_index in layer_set and value.role in role_set
    )
    selected = tuple(sorted(selected, key=lambda value: value.key))
    if len(selected) != len(layer_set) * len(role_set):
        raise Lfm2ActivationFactorizationTopologyError(
            "The requested dense LFM2 projections are incomplete."
        )
    return selected


def _activation_rows_for_role(
    evidence: Lfm2ActivationFactorizationEvidence,
    role: _DenseRole,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = evidence.layer(role.layer_index)
    if role.role == "w2":
        return layer.fit_post_swiglu, layer.validation_post_swiglu
    return layer.fit_layer_inputs, layer.validation_layer_inputs


def _validate_selected_parameter_independence(
    model: Any,
    selected: Sequence[_DenseRole],
) -> None:
    named = tuple(model.named_parameters(remove_duplicate=False))
    occurrences: dict[int, list[str]] = {}
    storage: dict[tuple[str, int], list[str]] = {}
    for name, parameter in named:
        occurrences.setdefault(id(parameter), []).append(name)
        key = (str(parameter.device), int(parameter.untyped_storage().data_ptr()))
        storage.setdefault(key, []).append(name)
    for value in selected:
        weight = value.module.weight
        names = occurrences.get(id(weight), ())
        key = (str(weight.device), int(weight.untyped_storage().data_ptr()))
        if len(names) != 1 or len(storage.get(key, ())) != 1:
            raise Lfm2ActivationFactorizationTopologyError(
                f"Layer {value.layer_index} {value.role} weight is aliased; "
                "exact physical savings cannot be proven."
            )


@dataclass(frozen=True, slots=True)
class _RankCurve:
    role: _DenseRole
    max_rank: int
    validation_errors: tuple[float, ...]


def plan_lfm2_activation_factorization(
    model: Any,
    evidence: Lfm2ActivationFactorizationEvidence,
    *,
    target_unique_parameter_budget: int,
    layer_indices: Sequence[int] | None = None,
    roles: Sequence[str] = _ROLES,
) -> Lfm2ActivationFactorizationPlan:
    """Allocate and seal physical ranks under a global unique-parameter budget."""

    with _TRANSACTION_LOCK, torch.no_grad():
        _layers, dense_roles = _validate_evidence(model, evidence)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise Lfm2ActivationFactorizationTransactionError(
                "Existing gradients must be cleared before zero-training planning."
            )
        logical_before, unique_before = _parameter_counts(model)
        if (
            isinstance(target_unique_parameter_budget, bool)
            or not isinstance(target_unique_parameter_budget, int)
            or target_unique_parameter_budget <= 0
            or target_unique_parameter_budget >= unique_before
        ):
            raise Lfm2ActivationFactorizationTopologyError(
                "target_unique_parameter_budget must be positive and smaller "
                "than the exact parent unique-Parameter count."
            )
        selected = _normalize_selection(
            evidence,
            dense_roles,
            layer_indices,
            roles,
        )
        _validate_selected_parameter_independence(model, selected)
        selected_dense = sum(value.dense_parameters for value in selected)
        constant_parameters = unique_before - selected_dense
        curves: list[_RankCurve] = []
        minimum_factor_parameters = 0
        for value in selected:
            maximum = min(
                value.input_features,
                value.output_features,
                (value.dense_parameters - 1) // value.increment_cost,
            )
            if maximum <= 0:
                raise Lfm2ActivationFactorizationTopologyError(
                    f"Layer {value.layer_index} {value.role} cannot be "
                    "physically reduced by a two-factor runtime."
                )
            fit, validation = _activation_rows_for_role(evidence, value)
            errors = _activation_validation_curve(
                value.module.weight,
                fit,
                validation,
                maximum,
            )
            curves.append(
                _RankCurve(
                    role=value,
                    max_rank=maximum,
                    validation_errors=errors,
                )
            )
            minimum_factor_parameters += value.increment_cost
        minimum_total = constant_parameters + minimum_factor_parameters
        if target_unique_parameter_budget < minimum_total:
            raise Lfm2ActivationFactorizationTopologyError(
                "The global target is below the rank-1 physical minimum for "
                "the selected projections."
            )

        ranks = {
            (curve.role.layer_index, curve.role.role): 1 for curve in curves
        }
        projected = minimum_total
        steps: list[Lfm2RankAllocationStep] = []
        while True:
            candidates: list[
                tuple[
                    float,
                    int,
                    int,
                    str,
                    _RankCurve,
                    float,
                ]
            ] = []
            for curve in curves:
                key = (curve.role.layer_index, curve.role.role)
                current = ranks[key]
                cost = curve.role.increment_cost
                if (
                    current >= curve.max_rank
                    or projected + cost > target_unique_parameter_budget
                ):
                    continue
                benefit = max(
                    0.0,
                    curve.validation_errors[current]
                    - curve.validation_errors[current + 1],
                )
                score = benefit / cost
                candidates.append(
                    (
                        -score,
                        curve.role.layer_index,
                        _ROLES.index(curve.role.role),
                        curve.role.role,
                        curve,
                        benefit,
                    )
                )
            if not candidates:
                break
            (
                _negative_score,
                layer_index,
                _role_order,
                role_name,
                chosen,
                benefit,
            ) = min(candidates)
            key = (layer_index, role_name)
            rank_before = ranks[key]
            ranks[key] += 1
            projected += chosen.role.increment_cost
            steps.append(
                Lfm2RankAllocationStep(
                    layer_index=layer_index,
                    role=role_name,
                    rank_before=rank_before,
                    rank_after=rank_before + 1,
                    parameter_cost=chosen.role.increment_cost,
                    validation_error_reduction=benefit,
                    benefit_per_parameter=(
                        benefit / chosen.role.increment_cost
                    ),
                )
            )

        role_plans: list[Lfm2ActivationRolePlan] = []
        for value in selected:
            fit, validation = _activation_rows_for_role(evidence, value)
            rank = ranks[(value.layer_index, value.role)]
            factorization = factorize_activation_weighted_linear(
                value.module.weight,
                fit,
                validation,
                rank=rank,
            )
            role_plans.append(
                Lfm2ActivationRolePlan(
                    layer_index=value.layer_index,
                    role=value.role,
                    factorization=factorization,
                    source_weight_sha256=_tensor_sha256(
                        value.module.weight,
                        domain=b"lfm2-source-dense-weight-v1",
                    ),
                    requires_grad=bool(value.module.weight.requires_grad),
                )
            )
        factor_parameters = sum(
            value.factor_parameters for value in role_plans
        )
        projected_exact = constant_parameters + factor_parameters
        if (
            projected_exact != projected
            or projected_exact > target_unique_parameter_budget
            or projected_exact >= unique_before
        ):
            raise Lfm2ActivationFactorizationTransactionError(
                "Rank allocation did not satisfy exact physical accounting."
            )
        saved = unique_before - projected_exact
        plan = Lfm2ActivationFactorizationPlan(
            schema=PLAN_SCHEMA,
            parent_identity_sha256=evidence.parent_identity_sha256,
            evidence_sha256=evidence.evidence_sha256,
            target_unique_parameter_budget=target_unique_parameter_budget,
            logical_parameters_before=logical_before,
            unique_parameters_before=unique_before,
            projected_logical_parameters_after=logical_before - saved,
            projected_unique_parameters_after=projected_exact,
            projected_parameters_saved=saved,
            role_plans=tuple(role_plans),
            allocation_steps=tuple(steps),
            # Each runtime records in_features, out_features, and rank as
            # Python integer attributes.  It adds no integer Tensor/buffer.
            integer_metadata_values=3 * len(role_plans),
            integer_tensor_count=0,
            integer_tensor_bytes=0,
            plan_sha256="",
        )
        return _seal_plan(plan)


class Lfm2ActivationFactorizedLinear(torch.nn.Module):
    """Physical two-factor bias-free runtime with no retained dense weight."""

    def __init__(
        self,
        input_factor: torch.Tensor,
        output_factor: torch.Tensor,
        *,
        requires_grad: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if (
            not isinstance(input_factor, torch.Tensor)
            or not isinstance(output_factor, torch.Tensor)
            or input_factor.ndim != 2
            or output_factor.ndim != 2
            or input_factor.shape[0] != output_factor.shape[1]
            or input_factor.shape[0] <= 0
        ):
            raise Lfm2ActivationFactorizationTopologyError(
                "Factor tensors do not form a valid two-stage Linear."
            )
        self.in_features = int(input_factor.shape[1])
        self.out_features = int(output_factor.shape[0])
        self.rank = int(input_factor.shape[0])
        self.input_factor = torch.nn.Parameter(
            input_factor.detach().to(device=device, dtype=dtype).clone(),
            requires_grad=requires_grad,
        )
        self.output_factor = torch.nn.Parameter(
            output_factor.detach().to(device=device, dtype=dtype).clone(),
            requires_grad=requires_grad,
        )
        self.register_parameter("bias", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(
            F.linear(inputs, self.input_factor),
            self.output_factor,
        )

    def effective_weight(self) -> torch.Tensor:
        return self.output_factor @ self.input_factor

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias=False"
        )


@dataclass(frozen=True, slots=True)
class Lfm2ActivationFactorizationReport:
    """Authenticated physical accounting for one installed plan."""

    schema: str
    parent_identity_before: str
    model_identity_after: str
    evidence_sha256: str
    selected_roles: tuple[tuple[int, str], ...]
    ranks: tuple[int, ...]
    methods: tuple[str, ...]
    logical_parameters_before: int
    logical_parameters_after: int
    unique_parameters_before: int
    unique_parameters_after: int
    parameters_saved: int
    dense_parameters_removed: int
    factor_parameters_added: int
    integer_metadata_values: int
    integer_tensor_count: int
    integer_tensor_bytes: int
    unselected_modules_preserved_exactly: bool
    attention_and_recurrent_operators_preserved: bool
    buffers_preserved: bool
    training_steps: int = 0
    gradients_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "method": METHOD,
            "parent_identity_before": self.parent_identity_before,
            "model_identity_after": self.model_identity_after,
            "evidence_sha256": self.evidence_sha256,
            "selected_roles": [
                [layer_index, role]
                for layer_index, role in self.selected_roles
            ],
            "ranks": list(self.ranks),
            "methods": list(self.methods),
            "logical_parameters_before": self.logical_parameters_before,
            "logical_parameters_after": self.logical_parameters_after,
            "unique_parameters_before": self.unique_parameters_before,
            "unique_parameters_after": self.unique_parameters_after,
            "parameters_saved": self.parameters_saved,
            "dense_parameters_removed": self.dense_parameters_removed,
            "factor_parameters_added": self.factor_parameters_added,
            "integer_metadata_values": self.integer_metadata_values,
            "integer_tensor_count": self.integer_tensor_count,
            "integer_tensor_bytes": self.integer_tensor_bytes,
            "unselected_modules_preserved_exactly": (
                self.unselected_modules_preserved_exactly
            ),
            "attention_and_recurrent_operators_preserved": (
                self.attention_and_recurrent_operators_preserved
            ),
            "buffers_preserved": self.buffers_preserved,
            "training_steps": self.training_steps,
            "gradients_used": self.gradients_used,
        }


@dataclass(frozen=True, slots=True)
class _ProjectionBinding:
    owner: Any
    layer_index: int
    role: str
    original: torch.nn.Linear
    installed: Lfm2ActivationFactorizedLinear


def _assign_projection(owner: Any, role: str, module: Any) -> None:
    setattr(owner, role, module)


def _buffer_snapshot(model: Any) -> dict[str, tuple[Any, str]]:
    return {
        name: (
            value,
            _tensor_sha256(value, domain=b"lfm2-buffer-snapshot-v1"),
        )
        for name, value in model.named_buffers(remove_duplicate=False)
    }


def _verify_buffers(
    model: Any,
    before: Mapping[str, tuple[Any, str]],
) -> None:
    after = dict(model.named_buffers(remove_duplicate=False))
    if set(after) != set(before):
        raise Lfm2ActivationFactorizationTransactionError(
            "Factorization changed the model buffer namespace."
        )
    for name, (original, digest) in before.items():
        if (
            after[name] is not original
            or _tensor_sha256(
                after[name],
                domain=b"lfm2-buffer-snapshot-v1",
            )
            != digest
        ):
            raise Lfm2ActivationFactorizationTransactionError(
                f"Factorization changed unselected buffer {name!r}."
            )


def _module_snapshot(
    layers: Sequence[Any],
) -> tuple[tuple[int, tuple[tuple[str, int], ...]], ...]:
    values = []
    for layer in layers:
        ffn = layer.feed_forward
        children = tuple(
            (name, id(module))
            for name, module in layer.named_children()
            if name != "feed_forward"
        )
        values.append((id(ffn), children))
    return tuple(values)


class InstalledLfm2ActivationFactorization:
    """Live transaction handle for commit or exact module/Parameter rollback."""

    def __init__(
        self,
        *,
        model: Any,
        report: Lfm2ActivationFactorizationReport,
        bindings: tuple[_ProjectionBinding, ...],
        parent_parameters: dict[str, Any],
        parent_identity: str,
        buffers: dict[str, tuple[Any, str]],
    ) -> None:
        self.model = model
        self.report = report
        self._bindings = bindings
        self._parent_parameters = parent_parameters
        self._parent_identity = parent_identity
        self._buffers = buffers
        self.active = True
        self.committed = False

    def rollback(self) -> bool:
        with _TRANSACTION_LOCK, torch.no_grad():
            if not self.active:
                return False
            for value in self._bindings:
                _assign_projection(value.owner, value.role, value.original)
            _verify_buffers(self.model, self._buffers)
            restored = dict(
                self.model.named_parameters(remove_duplicate=False)
            )
            exact = (
                set(restored) == set(self._parent_parameters)
                and all(
                    restored[name] is parameter
                    for name, parameter in self._parent_parameters.items()
                )
                and _model_identity(self.model) == self._parent_identity
            )
            if not exact:
                raise Lfm2ActivationFactorizationTransactionError(
                    "Exact activation-factorization rollback failed."
                )
            self.active = False
            return True

    def commit(self) -> Lfm2ActivationFactorizationReport:
        with _TRANSACTION_LOCK:
            if not self.active:
                raise Lfm2ActivationFactorizationTransactionError(
                    "The factorization transaction is no longer active."
                )
            self.active = False
            self.committed = True
            self._bindings = ()
            self._parent_parameters = {}
            self._buffers = {}
            return self.report

    def __enter__(self) -> InstalledLfm2ActivationFactorization:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()


def install_lfm2_activation_factorization_plan(
    model: Any,
    evidence: Lfm2ActivationFactorizationEvidence,
    plan: Lfm2ActivationFactorizationPlan,
) -> InstalledLfm2ActivationFactorization:
    """Verify, then atomically install a sealed two-factor LFM2 plan."""

    with _TRANSACTION_LOCK, torch.no_grad():
        layers, dense_roles = _validate_evidence(model, evidence)
        if (
            not isinstance(plan, Lfm2ActivationFactorizationPlan)
            or plan.schema != PLAN_SCHEMA
        ):
            raise Lfm2ActivationFactorizationTransactionError(
                "A sealed activation-factorization plan is required."
            )
        # Validate every tensor seal before constructing or assigning a module.
        _validate_plan_seal(plan)
        if (
            plan.parent_identity_sha256 != evidence.parent_identity_sha256
            or plan.evidence_sha256 != evidence.evidence_sha256
        ):
            raise Lfm2ActivationFactorizationTransactionError(
                "The plan does not match the exact model/evidence pair."
            )
        logical_before, unique_before = _parameter_counts(model)
        if (
            logical_before != plan.logical_parameters_before
            or unique_before != plan.unique_parameters_before
            or _model_identity(model) != plan.parent_identity_sha256
        ):
            raise Lfm2ActivationFactorizationTransactionError(
                "The live parent changed after planning."
            )
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise Lfm2ActivationFactorizationTransactionError(
                "Existing gradients must be cleared before zero-training install."
            )
        lookup = {
            (value.layer_index, value.role): value for value in dense_roles
        }
        if (
            not plan.role_plans
            or len(set(plan.selected_roles)) != len(plan.role_plans)
        ):
            raise Lfm2ActivationFactorizationTransactionError(
                "The plan has duplicate or absent selected roles."
            )
        _validate_selected_parameter_independence(
            model,
            tuple(lookup[key] for key in plan.selected_roles),
        )
        parent_parameters = dict(
            model.named_parameters(remove_duplicate=False)
        )
        buffers = _buffer_snapshot(model)
        structure = _module_snapshot(layers)
        bindings: list[_ProjectionBinding] = []
        selected_original_ids: set[int] = set()
        for role_plan in plan.role_plans:
            key = (role_plan.layer_index, role_plan.role)
            dense = lookup.get(key)
            if dense is None:
                raise Lfm2ActivationFactorizationTransactionError(
                    f"Planned role {key!r} is absent from the live parent."
                )
            factorization = role_plan.factorization
            if (
                _tensor_sha256(
                    dense.module.weight,
                    domain=b"lfm2-source-dense-weight-v1",
                )
                != role_plan.source_weight_sha256
                or tuple(dense.module.weight.shape)
                != factorization.matrix_shape
                or bool(dense.module.weight.requires_grad)
                != role_plan.requires_grad
            ):
                raise Lfm2ActivationFactorizationTransactionError(
                    f"Planned source role {key!r} changed."
                )
            installed = Lfm2ActivationFactorizedLinear(
                factorization.input_factor,
                factorization.output_factor,
                requires_grad=role_plan.requires_grad,
                device=dense.module.weight.device,
                dtype=dense.module.weight.dtype,
            )
            bindings.append(
                _ProjectionBinding(
                    owner=dense.owner,
                    layer_index=dense.layer_index,
                    role=dense.role,
                    original=dense.module,
                    installed=installed,
                )
            )
            selected_original_ids.add(id(dense.module.weight))

        try:
            for value in bindings:
                _assign_projection(
                    value.owner,
                    value.role,
                    value.installed,
                )
            if _module_snapshot(layers) != structure:
                raise Lfm2ActivationFactorizationTransactionError(
                    "Factorization changed a decoder shell or non-FFN operator."
                )
            for value in bindings:
                if getattr(value.owner, value.role) is not value.installed:
                    raise Lfm2ActivationFactorizationTransactionError(
                        "A planned factorized projection was not installed."
                    )
            _verify_buffers(model, buffers)
            parameters_after = dict(
                model.named_parameters(remove_duplicate=False)
            )
            for name, parameter in parent_parameters.items():
                if id(parameter) not in selected_original_ids:
                    if (
                        name not in parameters_after
                        or parameters_after[name] is not parameter
                    ):
                        raise Lfm2ActivationFactorizationTransactionError(
                            f"Unselected Parameter {name!r} changed."
                        )
            factor_ids = {
                id(parameter)
                for value in bindings
                for parameter in (
                    value.installed.input_factor,
                    value.installed.output_factor,
                )
            }
            new_ids = {
                id(parameter)
                for parameter in parameters_after.values()
                if id(parameter) not in {
                    id(value) for value in parent_parameters.values()
                }
            }
            if factor_ids != new_ids or len(factor_ids) != 2 * len(bindings):
                raise Lfm2ActivationFactorizationTransactionError(
                    "Installed factor Parameter namespace is not exact."
                )
            logical_after, unique_after = _parameter_counts(model)
            if (
                logical_after != plan.projected_logical_parameters_after
                or unique_after != plan.projected_unique_parameters_after
                or unique_before - unique_after
                != plan.projected_parameters_saved
            ):
                raise Lfm2ActivationFactorizationTransactionError(
                    "Installed physical Parameter accounting differs from plan."
                )
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise Lfm2ActivationFactorizationTransactionError(
                    "A gradient appeared during zero-training factorization."
                )
            report = Lfm2ActivationFactorizationReport(
                schema=REPORT_SCHEMA,
                parent_identity_before=plan.parent_identity_sha256,
                model_identity_after=_model_identity(model),
                evidence_sha256=evidence.evidence_sha256,
                selected_roles=plan.selected_roles,
                ranks=tuple(value.rank for value in plan.role_plans),
                methods=tuple(
                    value.factorization.method for value in plan.role_plans
                ),
                logical_parameters_before=logical_before,
                logical_parameters_after=logical_after,
                unique_parameters_before=unique_before,
                unique_parameters_after=unique_after,
                parameters_saved=unique_before - unique_after,
                dense_parameters_removed=sum(
                    value.dense_parameters for value in plan.role_plans
                ),
                factor_parameters_added=sum(
                    value.factor_parameters for value in plan.role_plans
                ),
                integer_metadata_values=plan.integer_metadata_values,
                integer_tensor_count=plan.integer_tensor_count,
                integer_tensor_bytes=plan.integer_tensor_bytes,
                unselected_modules_preserved_exactly=True,
                attention_and_recurrent_operators_preserved=True,
                buffers_preserved=True,
            )
            return InstalledLfm2ActivationFactorization(
                model=model,
                report=report,
                bindings=tuple(bindings),
                parent_parameters=parent_parameters,
                parent_identity=plan.parent_identity_sha256,
                buffers=buffers,
            )
        except BaseException as exc:
            for value in bindings:
                _assign_projection(
                    value.owner,
                    value.role,
                    value.original,
                )
            try:
                _verify_buffers(model, buffers)
                restored = dict(
                    model.named_parameters(remove_duplicate=False)
                )
                exact = (
                    set(restored) == set(parent_parameters)
                    and all(
                        restored[name] is parameter
                        for name, parameter in parent_parameters.items()
                    )
                    and _model_identity(model) == plan.parent_identity_sha256
                )
            except BaseException as rollback_exc:
                raise Lfm2ActivationFactorizationTransactionError(
                    "Factorization failed and rollback could not be authenticated."
                ) from rollback_exc
            if not exact:
                raise Lfm2ActivationFactorizationTransactionError(
                    "Factorization failed and exact rollback failed."
                ) from exc
            raise


def install_lfm2_activation_factorization(
    model: Any,
    evidence: Lfm2ActivationFactorizationEvidence,
    *,
    target_unique_parameter_budget: int,
    layer_indices: Sequence[int] | None = None,
    roles: Sequence[str] = _ROLES,
) -> InstalledLfm2ActivationFactorization:
    """Plan and atomically install an activation-aware LFM2 factorization."""

    plan = plan_lfm2_activation_factorization(
        model,
        evidence,
        target_unique_parameter_budget=target_unique_parameter_budget,
        layer_indices=layer_indices,
        roles=roles,
    )
    return install_lfm2_activation_factorization_plan(model, evidence, plan)


build_lfm2_activation_factorization_plan = (
    plan_lfm2_activation_factorization
)


__all__ = [
    "EVIDENCE_SCHEMA",
    "METHOD",
    "PLAN_SCHEMA",
    "REPORT_SCHEMA",
    "InstalledLfm2ActivationFactorization",
    "Lfm2ActivationFactorizationError",
    "Lfm2ActivationFactorizationEvidence",
    "Lfm2ActivationFactorizationEvidenceError",
    "Lfm2ActivationFactorizationLayerEvidence",
    "Lfm2ActivationFactorizationPlan",
    "Lfm2ActivationFactorizationReport",
    "Lfm2ActivationFactorizationTopologyError",
    "Lfm2ActivationFactorizationTransactionError",
    "Lfm2ActivationFactorizedLinear",
    "Lfm2ActivationRolePlan",
    "Lfm2LinearFactorization",
    "Lfm2RankAllocationStep",
    "build_lfm2_activation_factorization_plan",
    "factorize_activation_weighted_linear",
    "install_lfm2_activation_factorization",
    "install_lfm2_activation_factorization_plan",
    "plan_lfm2_activation_factorization",
    "seal_lfm2_activation_factorization_evidence",
]
