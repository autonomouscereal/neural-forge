"""T4 — factored (low-rank) writers, exposed as a matrix-campaign technique.

The estate already carries this path.  ``CURRENT_FOLDING_GOAL`` §evidence names
the feasibility witness — "writer-rank 31 with layer-5 MLP width 765 reaches
the exact dense target while retaining all six attention layers at Q16/KV8/D64"
— and records that *layout accounting and blank materialization already support
factored writers*; what was missing was a compiler that produces the two factor
tensors and a caller allowed to install them.

So this module wires, it does not reinvent:

* the child geometry is the one
  :mod:`neural_forge.hf_lfm2_dynamic_blank_materializer` already materializes —
  ``out_proj`` becomes ``Sequential(Linear(internal, rank), Linear(rank,
  hidden))``, published under ``conv.out_proj.0.weight`` /
  ``conv.out_proj.1.weight`` and bridged from fold-evidence names
  ``conv.out_encoder.weight`` / ``conv.out_decoder.weight`` by
  ``lfm2_factorized_short_conv_evidence_state_key_mapping``;
* the parameter algebra is the one ``dynamic_parameter_frontier`` already
  counts: ``writer_rank * (hidden + internal)`` against a dense
  ``hidden * internal``;
* what is new is the *fit*.

Two fits, both closed form
--------------------------
``weighting="none"`` is the plain truncated SVD of the writer.  It is the
weight-space answer and it is what a layout study wants when it asks whether a
rank exists at all.

``weighting="evidence"`` is the answer a fold wants: minimize
``‖X W − X E D‖_F`` over rank-``r`` factorizations, which is *not* the plain
SVD.  With ``M = XᵀX`` the sealed-evidence Gram, the optimum is
``E D = M^{-1/2} · trunc_r(M^{1/2} W) `` — one eigendecomposition and one SVD,
no iteration.  Directions the evidence never excites fall in the null space of
``M`` and are dropped rather than spent, which is exactly the capacity a
rank-limited writer should not be wasting; the receipt seals how much weight
energy that removed so the decision is visible instead of silent.

Fold-only discipline
--------------------
SVD and eigendecomposition on already-materialized parent weights and sealed
evidence rows.  No gradient, no optimizer, no parent forward.  Tensors carrying
autograd state are refused.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_sha256


MATRIX_TECHNIQUE_ENTRY_SCHEMA = "neural-forge-matrix-technique-entry-v1"
T4_LOWRANK_WRITER_RECORD_SCHEMA = "neural-forge-matrix-t4-lowrank-writer-record-v1"

T4_TECHNIQUE_ID = "T4"
#: Must match ``matrix_cell_receipt.TECHNIQUES["T4"]``.
T4_TECHNIQUE_NAME = "low_rank_factor_writers"

T4_EVIDENCE_ALGORITHM = (
    "evidence-gram-whitened-truncated-svd-writer-factorization-v1"
)
T4_WEIGHT_ALGORITHM = "plain-truncated-svd-writer-factorization-v1"

#: The materializer's factorized ``out_proj`` state keys.  Kept as literals
#: with the owning module named so a rename there fails a test here rather than
#: silently producing an uninstallable child.
SHORT_CONV_FACTOR_ENCODER_SUFFIX = "conv.out_proj.0.weight"
SHORT_CONV_FACTOR_DECODER_SUFFIX = "conv.out_proj.1.weight"
SHORT_CONV_EVIDENCE_ENCODER_SUFFIX = "conv.out_encoder.weight"
SHORT_CONV_EVIDENCE_DECODER_SUFFIX = "conv.out_decoder.weight"
SHORT_CONV_FACTOR_KEY_OWNER = (
    "neural_forge.hf_lfm2_dynamic_blank_materializer."
    "lfm2_factorized_short_conv_evidence_state_key_mapping"
)

#: Writers the campaign names.  ``C05`` is the short-conv output writer — the
#: only one with materializer support today; ``C09``/``C13`` factor correctly
#: here but need the harness to install a two-factor module itself.
_COMPONENTS: Mapping[str, str] = MappingProxyType(
    {
        "C05": "conv_out_proj",
        "C09": "attn_o",
        "C13": "mlp_down",
    }
)
_COMPONENT_IDS_BY_NAME: Mapping[str, str] = MappingProxyType(
    {value: key for key, value in _COMPONENTS.items()}
)
_MATERIALIZER_BACKED_COMPONENTS = ("C05",)

_LEVEL_KEEP_FRACTION: Mapping[str, tuple[int, int]] = MappingProxyType(
    {"M16": (15, 16), "M8": (7, 8), "M4": (3, 4), "M1": (1, 1)}
)

_WEIGHTINGS = ("evidence", "none")

_PROHIBITED_OPERATION_AUDIT: Mapping[str, Any] = MappingProxyType(
    {
        "training_steps": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "parent_forward_calls": 0,
        "parent_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "model_forward_calls": 0,
        "pruned_parameters": 0,
        "quantized_parameters": 0,
        "network_access": False,
    }
)


class LowRankWriterError(ValueError):
    """A T4 factored-writer request is outside the technique's envelope."""


class LowRankWriterGeometryError(LowRankWriterError):
    """Writer geometry, level, or rank cannot be honoured."""


class LowRankWriterEvidenceError(LowRankWriterError):
    """Sealed evidence is missing, malformed, or carries no usable energy."""


class LowRankWriterConditioningError(LowRankWriterError):
    """The evidence Gram or the factorization is degenerate."""


class LowRankWriterBudgetError(LowRankWriterError):
    """The requested rank costs more parameters than the dense writer."""


class LowRankWriterDisciplineError(LowRankWriterError):
    """A fold-only discipline rule was violated by the request."""


class LowRankWriterAuthenticationError(LowRankWriterError):
    """A sealed technique record failed its seal or schema check."""


# ---------------------------------------------------------------------------
# Validated primitives
# ---------------------------------------------------------------------------


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise LowRankWriterError(
            "PyTorch is required for T4 factored writers."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LowRankWriterGeometryError(f"{name} must be a positive integer.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise LowRankWriterGeometryError(f"{name} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LowRankWriterGeometryError(
            f"{name} must be finite and non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise LowRankWriterGeometryError(f"{name} must be finite and non-negative.")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LowRankWriterError(f"{name} must be a nonempty string.")
    return value.strip()


def _no_autograd(value: Any, name: str) -> None:
    torch = _torch()
    if isinstance(value, torch.Tensor) and (
        value.requires_grad or value.grad_fn is not None
    ):
        raise LowRankWriterDisciplineError(
            f"{name} carries autograd state; T4 is a fold-only technique."
        )


def _matrix(value: Any, name: str, *, dtype: Any, device: Any) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
    ):
        raise LowRankWriterEvidenceError(f"{name} must be a nonempty real matrix.")
    _no_autograd(value, name)
    result = value.detach().to(device=device, dtype=dtype).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise LowRankWriterEvidenceError(f"{name} is not finite.")
    return result


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype),
            "shape": [int(size) for size in tensor.shape],
            "bytes": tensor.reshape(-1).view(torch.uint8).numpy().tobytes().hex(),
        }
    )


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LowRankWriterError("Sealed records cannot carry non-finite numbers.")
        return value
    raise LowRankWriterError(
        f"Sealed record contains non-JSON value {type(value).__name__}."
    )


def _sealed(body: Mapping[str, Any], field: str = "record_sha256") -> dict[str, Any]:
    value = _plain_json(body)
    return {**value, field: canonical_sha256(value)}


def verify_sealed_lowrank_writer_record(
    value: Mapping[str, Any],
    *,
    field: str = "record_sha256",
) -> dict[str, Any]:
    """Re-check a sealed technique record before the harness embeds it."""

    if not isinstance(value, Mapping):
        raise LowRankWriterAuthenticationError("A technique record must be a mapping.")
    result = _plain_json(value)
    supplied = result.pop(field, None)
    if supplied != canonical_sha256(result):
        raise LowRankWriterAuthenticationError("Technique record seal is invalid.")
    if result.get("schema") != T4_LOWRANK_WRITER_RECORD_SCHEMA:
        raise LowRankWriterAuthenticationError(
            "Technique record schema is unsupported."
        )
    return {**result, field: supplied}


def resolve_component(component: Any) -> tuple[str, str]:
    """Return ``(component_id, component_name)`` for a writer cell."""

    name = _text(component, "component")
    if name in _COMPONENTS:
        return name, _COMPONENTS[name]
    if name in _COMPONENT_IDS_BY_NAME:
        return _COMPONENT_IDS_BY_NAME[name], name
    if name == "custom":
        return "custom", "custom"
    raise LowRankWriterGeometryError(
        f"{name!r} is not a campaign writer for T4.  Supported: "
        + ", ".join(sorted(_COMPONENTS))
        + " (or component='custom')."
    )


def maximum_shrinking_rank(in_features: int, out_features: int) -> int:
    """Highest rank at which a factored writer still costs fewer parameters.

    A factored writer costs ``r*(in+out)`` against a dense ``in*out``, so it
    only shrinks below ``r = in*out/(in+out)``.  For a square writer that
    ceiling is just under half the full rank, which means the campaign's
    marginal levels (M16/M8/M4 keep 15/16, 7/8, 3/4 of the full rank) **never**
    shrink ``conv_out_proj`` or ``attn_o``.  T4's usable regime on those
    components is a deep explicit rank — the goal doc's witness is rank 31 on a
    1024-channel writer — not a marginal level.
    """

    inputs = _positive_int(in_features, "in_features")
    outputs = _positive_int(out_features, "out_features")
    ceiling = (inputs * outputs - 1) // (inputs + outputs)
    return max(ceiling, 0)


def resolve_writer_rank(
    *,
    level: Any,
    full_rank: int,
    rank: Any = None,
) -> int:
    """Resolve a campaign level into a writer rank."""

    width = _positive_int(full_rank, "full_rank")
    if rank is not None:
        resolved = _positive_int(rank, "rank")
    else:
        name = _text(level, "level").upper()
        if name in _LEVEL_KEEP_FRACTION:
            numerator, denominator = _LEVEL_KEEP_FRACTION[name]
            resolved = (width * numerator) // denominator
        elif name == "X4":
            raise LowRankWriterGeometryError(
                "level='X4' has no closed-form rank; supply the option bank's "
                "writer rank (the goal doc's witness is rank 31 at L13)."
            )
        else:
            raise LowRankWriterGeometryError(
                f"level={name!r} is not a campaign level. Supported: "
                + ", ".join(sorted(_LEVEL_KEEP_FRACTION))
                + ", X4 (with rank)."
            )
    if resolved < 1 or resolved > width:
        raise LowRankWriterGeometryError(
            f"writer rank {resolved} is outside 1..{width}."
        )
    return resolved


# ---------------------------------------------------------------------------
# The factorization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LowRankWriterFactorization:
    """One writer factored into ``decoder @ encoder`` at a fixed rank."""

    component_id: str
    component: str
    site: str
    level: str
    rank: int
    in_features: int
    out_features: int
    weighting: str
    encoder: Any
    decoder: Any
    singular_values: Any
    calibration: Mapping[str, Any]
    solve: Mapping[str, Any]
    record: Mapping[str, Any]

    @property
    def record_sha256(self) -> str:
        return str(self.record["record_sha256"])

    @property
    def dense_equivalent(self) -> Any:
        """``decoder @ encoder`` — the dense writer the child is equal to."""

        return self.decoder @ self.encoder

    def apply(self, rows: Any) -> Any:
        """Child writer output for ``(rows, in_features)`` input rows."""

        torch = _torch()
        if not isinstance(rows, torch.Tensor) or rows.ndim != 2:
            raise LowRankWriterEvidenceError("apply expects a 2-D row matrix.")
        _no_autograd(rows, "apply rows")
        if int(rows.shape[1]) != self.in_features:
            raise LowRankWriterGeometryError(
                f"apply expects {self.in_features} input features."
            )
        with torch.no_grad():
            values = rows.detach().to(
                dtype=self.encoder.dtype, device=self.encoder.device
            )
            return (values @ self.encoder.transpose(0, 1)) @ self.decoder.transpose(
                0, 1
            )

    def state_entries(self, prefix: str) -> Mapping[str, Any]:
        """Two ``nn.Sequential`` state entries under ``prefix``.

        ``prefix`` names the ``Sequential`` module, so the emitted keys are
        ``{prefix}.0.weight`` (encoder) and ``{prefix}.1.weight`` (decoder) —
        the layout the blank materializer builds.
        """

        name = _text(prefix, "prefix").rstrip(".")
        return MappingProxyType(
            {f"{name}.0.weight": self.encoder, f"{name}.1.weight": self.decoder}
        )


def factor_writer(
    *,
    parent_weight: Any,
    level: str | None = None,
    rank: int | None = None,
    evidence_rows: Any = None,
    weighting: str = "evidence",
    component: str = "custom",
    site: str = "global",
    ridge_relative: float = 0.0,
    allow_parameter_increase: bool = False,
    heldout_rows: Any = None,
    solve_dtype: Any = None,
    solve_device: Any = "cpu",
) -> LowRankWriterFactorization:
    """Factor a writer into ``decoder @ encoder`` at the level's rank.

    ``parent_weight`` is in ``torch.nn.Linear`` orientation
    ``(out_features, in_features)``; the returned ``encoder`` is
    ``(rank, in_features)`` and ``decoder`` is ``(out_features, rank)``, so
    ``decoder @ encoder`` is directly comparable with the parent weight and the
    pair drops into a two-``Linear`` ``Sequential`` unchanged.
    """

    torch = _torch()
    dtype = torch.float64 if solve_dtype is None else solve_dtype
    component_id, component_name = resolve_component(component)
    site_name = _text(site, "site")
    level_name = _text(level, "level").upper() if level is not None else "EXPLICIT"
    if weighting not in _WEIGHTINGS:
        raise LowRankWriterGeometryError(
            "weighting must be one of " + ", ".join(_WEIGHTINGS)
        )
    weight = _matrix(parent_weight, "parent_weight", dtype=dtype, device=solve_device)
    out_features, in_features = map(int, weight.shape)
    full_rank = min(out_features, in_features)
    resolved_rank = resolve_writer_rank(
        level=level_name if level is not None else "M1",
        full_rank=full_rank,
        rank=rank,
    )
    ridge_ratio = _nonnegative_float(ridge_relative, "ridge_relative")

    dense_parameters = out_features * in_features
    factored_parameters = resolved_rank * (out_features + in_features)
    ceiling = maximum_shrinking_rank(in_features, out_features)
    if factored_parameters >= dense_parameters and not allow_parameter_increase:
        raise LowRankWriterBudgetError(
            f"writer rank {resolved_rank} costs {factored_parameters} "
            f"parameters against a dense {dense_parameters}; this writer only "
            f"shrinks at rank <= {ceiling} (of a full rank {full_rank}). "
            "A factored writer that does not shrink the writer is refused "
            "unless allow_parameter_increase is set."
        )

    if weighting == "evidence" and evidence_rows is None:
        raise LowRankWriterEvidenceError(
            "weighting='evidence' requires evidence_rows; pass "
            "weighting='none' for the weight-space factorization."
        )

    with torch.no_grad():
        math_weight = weight.transpose(0, 1).contiguous()
        evidence_rank = in_features
        null_space_energy = 0.0
        gram_floor = 0.0
        if weighting == "none":
            algorithm = T4_WEIGHT_ALGORITHM
            target = math_weight
            forward = None
            backward = None
        else:
            algorithm = T4_EVIDENCE_ALGORITHM
            rows = _matrix(
                evidence_rows, "evidence_rows", dtype=dtype, device=solve_device
            )
            if int(rows.shape[1]) != in_features:
                raise LowRankWriterEvidenceError(
                    f"evidence_rows width {int(rows.shape[1])} does not match "
                    f"the writer's {in_features} input features."
                )
            gram = rows.transpose(0, 1) @ rows
            gram = (gram + gram.transpose(0, 1)) * 0.5
            if ridge_ratio > 0.0:
                gram = gram + ridge_ratio * float(
                    torch.diagonal(gram).mean().item()
                ) * torch.eye(in_features, dtype=dtype, device=gram.device)
            try:
                values, vectors = torch.linalg.eigh(gram)
            except RuntimeError as exc:  # pragma: no cover - LAPACK boundary
                raise LowRankWriterConditioningError(
                    "The evidence Gram eigendecomposition failed."
                ) from exc
            values = values.clamp_min(0.0)
            largest = float(values.max().item())
            if not math.isfinite(largest) or largest <= 0.0:
                raise LowRankWriterEvidenceError(
                    "The evidence Gram carries no energy."
                )
            gram_floor = float(torch.finfo(dtype).eps) * largest * in_features
            usable = values > gram_floor
            evidence_rank = int(usable.sum().item())
            if evidence_rank < resolved_rank:
                raise LowRankWriterConditioningError(
                    f"The evidence Gram has rank {evidence_rank}; a rank-"
                    f"{resolved_rank} writer is not identifiable from it."
                )
            roots = torch.where(usable, torch.sqrt(values), torch.zeros_like(values))
            inverse = torch.where(
                usable,
                1.0 / torch.sqrt(values.clamp_min(gram_floor)),
                torch.zeros_like(values),
            )
            forward = vectors @ torch.diag(roots) @ vectors.transpose(0, 1)
            backward = vectors @ torch.diag(inverse) @ vectors.transpose(0, 1)
            projected = backward @ forward @ math_weight
            null_space_energy = float(
                torch.linalg.matrix_norm(math_weight - projected, ord="fro").item()
            ) / max(
                float(torch.linalg.matrix_norm(math_weight, ord="fro").item()),
                float(torch.finfo(dtype).tiny),
            )
            target = forward @ math_weight

        left, singular, right_h = torch.linalg.svd(target, full_matrices=False)
        if int(singular.numel()) < resolved_rank:
            raise LowRankWriterGeometryError(
                "The writer has fewer singular directions than the requested "
                "rank."
            )
        roots = torch.sqrt(singular[:resolved_rank].clamp_min(0.0))
        left_factor = left[:, :resolved_rank] * roots.unsqueeze(0)
        right_factor = right_h[:resolved_rank, :] * roots.unsqueeze(1)
        if weighting == "evidence":
            left_factor = backward @ left_factor
        encoder = left_factor.transpose(0, 1).contiguous()
        decoder = right_factor.transpose(0, 1).contiguous()
        reconstruction = decoder @ encoder
        if not bool(torch.isfinite(reconstruction).all().item()):
            raise LowRankWriterConditioningError(
                "The factored writer contains non-finite values."
            )

        weight_norm = float(torch.linalg.matrix_norm(weight, ord="fro").item())
        weight_relative = float(
            torch.linalg.matrix_norm(reconstruction - weight, ord="fro").item()
        ) / max(weight_norm, float(torch.finfo(dtype).tiny))
        tail = float((singular[resolved_rank:] ** 2).sum().item())
        total = float((singular**2).sum().item())
        spectral_tail = math.sqrt(tail / total) if total > 0.0 else 0.0

        evidence_report: dict[str, Any] | None = None
        if evidence_rows is not None:
            rows = _matrix(
                evidence_rows, "evidence_rows", dtype=dtype, device=solve_device
            )
            if int(rows.shape[1]) != in_features:
                raise LowRankWriterEvidenceError(
                    "evidence_rows width does not match the writer input."
                )
            parent_output = rows @ math_weight
            child_output = (rows @ encoder.transpose(0, 1)) @ decoder.transpose(0, 1)
            denominator = float(
                torch.linalg.matrix_norm(parent_output, ord="fro").item()
            )
            evidence_report = {
                "rows": int(rows.shape[0]),
                "write_relative_l2": float(
                    torch.linalg.matrix_norm(
                        child_output - parent_output, ord="fro"
                    ).item()
                )
                / max(denominator, float(torch.finfo(dtype).tiny)),
                "write_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        child_output.reshape(1, -1),
                        parent_output.reshape(1, -1),
                    ).item()
                ),
            }

        heldout_report: dict[str, Any] | None = None
        if heldout_rows is not None:
            held = _matrix(
                heldout_rows, "heldout_rows", dtype=dtype, device=solve_device
            )
            if int(held.shape[1]) != in_features:
                raise LowRankWriterEvidenceError(
                    "heldout_rows width does not match the writer input."
                )
            parent_output = held @ math_weight
            child_output = (held @ encoder.transpose(0, 1)) @ decoder.transpose(0, 1)
            denominator = float(
                torch.linalg.matrix_norm(parent_output, ord="fro").item()
            )
            heldout_report = {
                "rows": int(held.shape[0]),
                "write_relative_l2": float(
                    torch.linalg.matrix_norm(
                        child_output - parent_output, ord="fro"
                    ).item()
                )
                / max(denominator, float(torch.finfo(dtype).tiny)),
                "write_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        child_output.reshape(1, -1),
                        parent_output.reshape(1, -1),
                    ).item()
                ),
            }

        published_singular = singular.to(device="cpu").contiguous()

    calibration = {
        "full_rank": full_rank,
        "at_full_rank": resolved_rank == full_rank,
        "weight_relative_l2": weight_relative,
        "spectral_tail_fraction": spectral_tail,
        "evidence": evidence_report,
        "heldout": heldout_report,
    }
    solve = {
        "solver_path": f"{__name__}.factor_writer",
        "algorithm": algorithm,
        "weighting": weighting,
        "ridge_relative": ridge_ratio,
        # Reported as a bounded ratio: an exactly singular writer would
        # otherwise seal an infinity and fail the record's JSON contract.
        "condition_number": float(
            published_singular.max().item()
            / max(
                float(published_singular.min().item()),
                float(published_singular.max().item())
                * float(torch.finfo(dtype).eps),
            )
        ),
        "condition_cap_saturated": False,
        "evidence_rank": evidence_rank,
        "evidence_gram_floor": gram_floor,
        "weight_energy_outside_evidence_range": null_space_energy,
        "prior_norm": float(torch.linalg.matrix_norm(weight, ord="fro").item()),
        "correction_norm": float(
            torch.linalg.matrix_norm(reconstruction - weight, ord="fro").item()
        ),
        "requested_clamp_ratio": 0.0,
        "allowed_clamp_ratio": 0.0,
        "clamp_engaged": False,
    }
    body = {
        "schema": T4_LOWRANK_WRITER_RECORD_SCHEMA,
        "technique_id": T4_TECHNIQUE_ID,
        "technique": T4_TECHNIQUE_NAME,
        "component_id": component_id,
        "component": component_name,
        "site": site_name,
        "level": level_name,
        "writer_rank": resolved_rank,
        "writer_topology": "factorized",
        "writer_factor_count": 2,
        "in_features": in_features,
        "out_features": out_features,
        "parent_parameters": dense_parameters,
        "child_parameters": factored_parameters,
        "parameter_ratio": factored_parameters / dense_parameters,
        "maximum_shrinking_rank": ceiling,
        "shrinks_the_writer": factored_parameters < dense_parameters,
        "encoder_sha256": _tensor_sha256(encoder),
        "decoder_sha256": _tensor_sha256(decoder),
        "parent_weight_sha256": _tensor_sha256(weight),
        "materializer_state_key_owner": SHORT_CONV_FACTOR_KEY_OWNER,
        "materializer_backed_components": list(_MATERIALIZER_BACKED_COMPONENTS),
        "materializer_backed": component_id in _MATERIALIZER_BACKED_COMPONENTS,
        "solve_placement": {
            "device": str(weight.device),
            "dtype": str(weight.dtype).replace("torch.", ""),
        },
        "calibration": calibration,
        "solve": solve,
        "operation_audit": dict(_PROHIBITED_OPERATION_AUDIT),
    }
    return LowRankWriterFactorization(
        component_id=component_id,
        component=component_name,
        site=site_name,
        level=level_name,
        rank=resolved_rank,
        in_features=in_features,
        out_features=out_features,
        weighting=weighting,
        encoder=encoder,
        decoder=decoder,
        singular_values=published_singular,
        calibration=MappingProxyType(calibration),
        solve=MappingProxyType(solve),
        record=MappingProxyType(_sealed(body)),
    )


def lfm2_short_conv_writer_state_entries(
    factorization: LowRankWriterFactorization,
    *,
    layer_index: int,
    internal_channel_width: int | None = None,
    hidden_size: int | None = None,
) -> Mapping[str, Any]:
    """Publish a factored short-conv writer under the materializer's keys.

    The emitted keys are exactly the ones
    :mod:`neural_forge.hf_lfm2_dynamic_blank_materializer` builds for a
    ``short_conv_writer_rank`` layer, so a strict state-dict load accepts them
    without an alias, a sidecar, or a runtime recomputation.
    """

    if not isinstance(factorization, LowRankWriterFactorization):
        raise LowRankWriterError(
            "lfm2_short_conv_writer_state_entries needs a "
            "LowRankWriterFactorization."
        )
    index = int(layer_index)
    if isinstance(layer_index, bool) or index < 0:
        raise LowRankWriterGeometryError("layer_index must be a non-negative int.")
    if internal_channel_width is not None and (
        _positive_int(internal_channel_width, "internal_channel_width")
        != factorization.in_features
    ):
        raise LowRankWriterGeometryError(
            "internal_channel_width disagrees with the factored writer input."
        )
    if hidden_size is not None and (
        _positive_int(hidden_size, "hidden_size") != factorization.out_features
    ):
        raise LowRankWriterGeometryError(
            "hidden_size disagrees with the factored writer output."
        )
    prefix = f"model.layers.{index}."
    return MappingProxyType(
        {
            prefix + SHORT_CONV_FACTOR_ENCODER_SUFFIX: factorization.encoder,
            prefix + SHORT_CONV_FACTOR_DECODER_SUFFIX: factorization.decoder,
        }
    )


# ---------------------------------------------------------------------------
# Registry entry (matrix-campaign design §4.3 dispatch table)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixTechniqueEntry:
    """One dispatch-table row for ``matrix_technique_registry``."""

    schema: str
    technique_id: str
    technique: str
    module: str
    components: tuple[str, ...]
    levels: tuple[str, ...]
    fit: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    describe: Callable[[], Mapping[str, Any]]


_REQUIRED = ("parent_weight",)
_OPTIONAL = (
    "level",
    "rank",
    "evidence_rows",
    "weighting",
    "component",
    "site",
    "ridge_relative",
    "allow_parameter_increase",
    "heldout_rows",
    "solve_dtype",
    "solve_device",
)


def _describe() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "schema": MATRIX_TECHNIQUE_ENTRY_SCHEMA,
            "technique_id": T4_TECHNIQUE_ID,
            "technique": T4_TECHNIQUE_NAME,
            "module": __name__,
            "record_schema": T4_LOWRANK_WRITER_RECORD_SCHEMA,
            "components": tuple(sorted(_COMPONENTS)),
            "component_names": MappingProxyType(dict(_COMPONENTS)),
            "materializer_backed_components": _MATERIALIZER_BACKED_COMPONENTS,
            "levels": ("M16", "M8", "M4", "X4"),
            "required_request_keys": _REQUIRED,
            "optional_request_keys": _OPTIONAL,
            "marginal_levels_shrink_square_writers": False,
            "evidence_algorithm": T4_EVIDENCE_ALGORITHM,
            "weight_algorithm": T4_WEIGHT_ALGORITHM,
            "state_key_owner": SHORT_CONV_FACTOR_KEY_OWNER,
            "fold_only": True,
            "gradient_free": True,
        }
    )


def fit_from_request(request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Registry-facing entry point: mapping in, mapping out."""

    if not isinstance(request, Mapping):
        raise LowRankWriterError("A technique request must be a mapping.")
    unknown = sorted(set(request) - frozenset(_REQUIRED + _OPTIONAL))
    if unknown:
        raise LowRankWriterError("Unsupported T4 request keys: " + ", ".join(unknown))
    missing = sorted(set(_REQUIRED) - set(request))
    if missing:
        raise LowRankWriterError("Missing T4 request keys: " + ", ".join(missing))
    factorization = factor_writer(**dict(request))
    return MappingProxyType(
        {
            "schema": T4_LOWRANK_WRITER_RECORD_SCHEMA,
            "technique_id": T4_TECHNIQUE_ID,
            "technique": T4_TECHNIQUE_NAME,
            "component": factorization.component,
            "component_id": factorization.component_id,
            "site": factorization.site,
            "level": factorization.level,
            "child_parameters": MappingProxyType(
                {
                    "encoder": factorization.encoder,
                    "decoder": factorization.decoder,
                }
            ),
            "route": MappingProxyType(
                {
                    "writer_rank": factorization.rank,
                    "writer_topology": "factorized",
                    "writer_factor_count": 2,
                    "singular_values": factorization.singular_values,
                }
            ),
            "apply": factorization.apply,
            "calibration": factorization.calibration,
            "solve": factorization.solve,
            "technique_record": factorization.record,
            "factorization": factorization,
        }
    )


def matrix_technique_registry_entry() -> MatrixTechniqueEntry:
    """Build the registry entry the matrix-cell harness dispatches through."""

    description = _describe()
    return MatrixTechniqueEntry(
        schema=MATRIX_TECHNIQUE_ENTRY_SCHEMA,
        technique_id=T4_TECHNIQUE_ID,
        technique=T4_TECHNIQUE_NAME,
        module=__name__,
        components=tuple(description["components"]),
        levels=tuple(description["levels"]),
        fit=fit_from_request,
        describe=_describe,
    )


# ---------------------------------------------------------------------------
# Harness adapter: neural_forge.matrix_technique_registry contract
# ---------------------------------------------------------------------------


def t4_lowrank_writer_technique(request: Any) -> Any:
    """T4 kernel for the matrix-cell harness: one writer, one site.

    The materialized tensor is ``decoder @ encoder`` — parent-shaped by
    construction — and ``stored_parameter_count`` is ``rank * (in + out)``, the
    same algebra ``dynamic_parameter_frontier`` already counts.

    One hard edge is surfaced rather than smoothed: a factored writer only
    costs less than a dense one below ``in*out/(in+out)``, which for a square
    writer is just under half the full rank.  The campaign's marginal levels
    keep 15/16, 7/8 and 3/4, so **M16/M8/M4 do not contract ``conv_out_proj``
    or ``attn_o`` at all** — they cost more.  That cell fails closed with the
    usable rank ceiling in the message instead of sealing a receipt for a child
    that grew.  ``t4_allow_parameter_increase`` is the deliberate override for
    a probe that wants the numeric answer anyway.
    """

    from .matrix_technique_registry import CellTechniqueError, CellTechniqueResult

    spec = request.component
    if spec.contraction_axis not in ("input", "output"):
        raise CellTechniqueError(
            f"T4 factors a linear writer; {spec.component_id} has axis "
            f"{spec.contraction_axis!r} and needs its own kernel."
        )
    if spec.input_endpoint is None:
        raise CellTechniqueError(
            f"{spec.component_id} has no sealed input endpoint to weight by."
        )
    key = spec.weight_keys[0]
    parent = request.parent_weights[key].detach()
    shape = tuple(int(value) for value in parent.shape)
    full_rank = min(shape[0], shape[1])
    # A level is a fraction of the component's *own width*, which on a
    # rectangular writer (mlp_down: 2560 in, 1024 out) can exceed the writer's
    # full rank.  Clamping and declaring beats refusing a well-posed cell for
    # an accounting reason; ``writer_rank`` lets the option bank drive the
    # goal doc's witness rank directly.
    requested_rank = int(request.knob("writer_rank", int(request.level_width)))
    resolved_rank = max(min(requested_rank, full_rank), 1)
    weighting = str(request.knob("weighting", "evidence"))
    rows = None
    if weighting == "evidence":
        rows = request.evidence.fit_rows(spec.input_endpoint)
    heldout = (
        request.evidence.heldout_rows(spec.input_endpoint)
        if int(getattr(request.evidence, "heldout_row_count", 0)) > 0
        else None
    )
    try:
        factorization = factor_writer(
            parent_weight=parent,
            rank=resolved_rank,
            evidence_rows=rows,
            weighting=weighting,
            component=spec.component_id,
            site=spec.site,
            ridge_relative=float(request.knob("ridge_relative", 0.0)),
            allow_parameter_increase=bool(
                request.knob("t4_allow_parameter_increase", False)
            ),
            heldout_rows=heldout,
        )
    except LowRankWriterError as exc:
        raise CellTechniqueError(f"{request.cell.text}: {exc}") from exc

    request.log(
        f"T4 {request.cell.text}: rank={factorization.rank} of "
        f"{factorization.calibration['full_rank']}, weighting={weighting}, "
        f"parameter ratio {factorization.record['parameter_ratio']:.4f}"
    )
    installed = factorization.dense_equivalent.contiguous()
    solve = dict(factorization.solve)
    solve["technique_record_sha256"] = factorization.record_sha256
    solve["calibration"] = dict(factorization.calibration)
    solve["writer_rank"] = factorization.rank
    solve["requested_writer_rank"] = requested_rank
    solve["writer_rank_clamped"] = resolved_rank != requested_rank
    solve["maximum_shrinking_rank"] = int(
        factorization.record["maximum_shrinking_rank"]
    )
    solve["shrinks_the_writer"] = bool(factorization.record["shrinks_the_writer"])
    solve["omp"] = None

    def predict(values: Any) -> Any:
        return factorization.apply(values)

    return CellTechniqueResult(
        installed_weights={key: installed},
        stored_parameter_count=int(factorization.record["child_parameters"]),
        solve=solve,
        predict=predict,
        artifact_identities={
            "t4_lowrank_writer_record_sha256": factorization.record_sha256
        },
        notes=(
            f"T4 factored writer at rank {factorization.rank}, "
            f"{factorization.record['child_parameters']} stored against "
            f"{factorization.record['parent_parameters']} dense; state keys "
            f"{SHORT_CONV_FACTOR_ENCODER_SUFFIX} / "
            f"{SHORT_CONV_FACTOR_DECODER_SUFFIX} for a C05 install."
        ),
    )


def _register_lowrank_writer_technique() -> None:
    """Wire T4 into the harness dispatch table at import time."""

    try:
        from .matrix_technique_registry import register
    except ImportError:  # pragma: no cover - the harness lane may not be present
        return
    for component in _COMPONENTS:
        register(
            "T4", t4_lowrank_writer_technique, component=component, replace=True
        )


_register_lowrank_writer_technique()


__all__ = [
    "MATRIX_TECHNIQUE_ENTRY_SCHEMA",
    "SHORT_CONV_EVIDENCE_DECODER_SUFFIX",
    "SHORT_CONV_EVIDENCE_ENCODER_SUFFIX",
    "SHORT_CONV_FACTOR_DECODER_SUFFIX",
    "SHORT_CONV_FACTOR_ENCODER_SUFFIX",
    "SHORT_CONV_FACTOR_KEY_OWNER",
    "T4_EVIDENCE_ALGORITHM",
    "T4_LOWRANK_WRITER_RECORD_SCHEMA",
    "T4_TECHNIQUE_ID",
    "T4_TECHNIQUE_NAME",
    "T4_WEIGHT_ALGORITHM",
    "LowRankWriterAuthenticationError",
    "LowRankWriterBudgetError",
    "LowRankWriterConditioningError",
    "LowRankWriterDisciplineError",
    "LowRankWriterError",
    "LowRankWriterEvidenceError",
    "LowRankWriterFactorization",
    "LowRankWriterGeometryError",
    "MatrixTechniqueEntry",
    "factor_writer",
    "fit_from_request",
    "maximum_shrinking_rank",
    "lfm2_short_conv_writer_state_entries",
    "matrix_technique_registry_entry",
    "resolve_component",
    "resolve_writer_rank",
    "t4_lowrank_writer_technique",
    "verify_sealed_lowrank_writer_record",
]
