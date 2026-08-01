"""Attention <-> ShortConv operator-KIND translation.

This module implements stage 1 of
``docs/ATTENTION_SHORTCONV_TRANSLATION_DESIGN_2026-08-01.md``: both closed-form
translations between the two operator kinds, at toy scale, with every quantity
the design requires measured and sealed.  There is no optimizer, gradient,
parent forward, runtime sidecar, or database anywhere in this file.

Direction B (ShortConv -> attention) is exact under gate collapse: a kernel-L
depthwise FIR with frozen ``B``/``C`` is a banded linear attention with per-lag
value maps ``M_l``.  With live gates the branch is expanded bilinearly around
evidence means into ``L`` lag streams plus one self-gate stream, and the
quadratic covariance remainder ``rho_2`` is measured, never fitted.  The softmax
compilation reuses ``compile_hf_causal_gqa_attention_fold`` **unchanged** by
synthesizing desired attention evidence for it.

Direction A (attention -> ShortConv) splits the cached attention delta
row-exactly into a band (lags ``0..L-1``) and a far remainder, fits the
lag-separable per-channel rank-one conv to the band with bounded deterministic
variable projection over condition-capped trace-normalized ridge sub-solves, and
routes the unrepresentable part to a downstream carrier as an endpoint-target
addition.

Implementation deviations from the design document, with reasons
----------------------------------------------------------------

1. **Direction B needs ``key_value_heads >= stream count``, not just
   ``query_heads >= L``** (design B.3 step 1).  A GQA group shares one key
   sequence, so every query head in a group sees the same relative-position
   score profile and therefore the same lag.  The per-lag head assignment is
   made over *key/value* heads; ``query_heads >= key_value_heads`` then follows
   from the topology contract.  Compiling with fewer KV heads than streams would
   silently collapse distinct lags onto one profile, so it fails closed.

2. **Per-stream value rank is bounded by the head geometry.**  The value written
   by a KV head is ``head_dim``-wide, so a stream owning ``m`` KV heads can only
   carry a rank-``m * head_dim`` image of its ``H x H`` map ``M_s``.  The design
   does not mention this bound.  Each ``M_s`` is therefore factorized by a
   sign-canonicalized truncated SVD at the available rank, and
   ``stream_rank_truncation_relative_error`` is a first-class sealed diagnostic.
   Total representable rank across all streams is exactly the target's
   ``key_value_heads * head_dim``.

3. **Desired Q/K are emitted pre-RoPE.**  The design says to inverse-rotate the
   constant vectors with ``invert_rope`` before handing them to the kernel.  The
   kernel applies RoPE to the source Q/K it is given and then inverse-rotates
   with the target theta (``hf_causal_gqa_attention_fold.py:621-666``), so
   passing the constants directly — with a source topology whose theta, scaling,
   head counts and head dim equal the target's — reproduces exactly the object
   the design asks for, with one fewer rotation round trip.

4. **The additive constant of the linearized conv is not representable** by the
   bias-free attention class; it can only ride the child rows' DC component
   through the fitted ``W_o``.  ``linearized_constant_relative_energy`` measures
   it instead of pretending it away (design A.3 item 4 for the mirror case).

5. **Direction A's ALS is implemented here rather than imported from the v5
   frontier.**  ``condition_capped_equilibrated_prior_residual_ridge_frontier``
   (``hf_lfm2_cpcf_short_conv_v5.py:1577``) requires authenticated
   ``Lfm2CpcfV5RidgeCase`` bundles and a *prior-residual* parameterization keyed
   to a transported source operator; a kind translation has no source operator
   of the target kind to transport.  The numerical recipe is reproduced exactly:
   column equilibration, ridge from the condition cap
   ``max(0, (lmax - kappa*lmin)/(kappa - 1))`` over the deployed candidate set
   ``(1e3, 1e4)``, trace-normalized ``alpha * trace(G)/d``, unpenalized
   intercept, and identifiability thresholding.  The two ridges are combined by
   ``max`` so both constraints hold.

6. **Monotonicity is guarded, not assumed.**  Design A.4 stage 2 claims the fit
   objective is monotone nonincreasing because each sub-step is an exact
   weighted ridge given the others.  That is true of the *regularized* objective
   only.  Each sub-step is therefore applied and reverted if the unregularized
   fit objective rose, which makes the claim hold for the reported objective.

7. **Bias columns inside the gates are penalized.**  Only ``b_out`` is a plain
   intercept of its design and is fitted unpenalized by centering.  ``b_B``,
   ``b_C``, ``b_x`` and ``conv_bias`` enter their sub-solves multiplied by
   data-dependent factors, so they are ordinary penalized columns.

8. **Direction A's endpoint gate is scored against the band target**, which is
   what the operator claims to represent; the full-delta number is reported
   beside it.  Gating on the full delta would fail exactly when ``rho_far`` is
   large and correctly routed.

9. **Direction B has an unrepresentable FIR prefix.**  A zero-padded FIR writes
   nothing for a lag that does not exist yet, but every causal softmax row sums
   to one, so the lag-``l`` head at position ``t < l`` unavoidably writes some
   other key's value.  Positions ``0..L-2`` are therefore outside the operator
   class in exactly the sense ``Delta_far`` is in direction A.  They are dropped
   from the *endpoint solve* under the design's own rule ("not fittable, and
   must not be fitted", A.3 item 1), counted in
   ``fit_rows_excluded_as_unrepresentable_prefix``, scored separately as
   ``validation_prefix_endpoint_relative_rmse``, and carried by the routed
   remainder.  Validation is never narrowed.  Leaving them in the solve is not
   neutral: on the toy lane it moves the fitted writer enough to take the
   collapsed-gate validation endpoint error from ``0.0239`` to ``0.2488``,
   because the writer ridge spends its budget on rows that no setting of the
   weights can serve.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_sha256
from .hf_causal_gqa_attention_fold import (
    HFCausalCoupledAttentionFold,
    HFCausalGQATopology,
    apply_rope,
    causal_attention_probabilities,
    compile_hf_causal_gqa_attention_fold,
)


SHORT_CONV_TO_ATTENTION_SCHEMA = (
    "neural-forge-causal-short-conv-to-attention-fold-v1"
)
ATTENTION_TO_SHORT_CONV_SCHEMA = (
    "neural-forge-causal-attention-to-short-conv-fold-v1"
)
KIND_TRANSLATION_REMAINDER_SCHEMA = (
    "neural-forge-causal-kind-translation-remainder-v1"
)

# The deployed v5 candidate set (hf_lfm2_cpcf_short_conv_v5.py:35).
CONDITION_CAP_CANDIDATES = (1_000.0, 10_000.0)


class HFCausalKindTranslationError(ValueError):
    """A kind translation is unsupported or its evidence is invalid."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise HFCausalKindTranslationError(
            "PyTorch is required for operator-kind translation."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HFCausalKindTranslationError(f"{name} must be a positive integer.")
    return value


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFCausalKindTranslationError(f"{name} must be positive.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise HFCausalKindTranslationError(f"{name} must be positive and finite.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFCausalKindTranslationError(f"{name} must be non-negative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise HFCausalKindTranslationError(f"{name} must be finite and non-negative.")
    return result


def _tensor(value: Any, name: str, *, ndim: int | None = None) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise HFCausalKindTranslationError(f"{name} must be a tensor.")
    if ndim is not None and value.ndim != ndim:
        raise HFCausalKindTranslationError(
            f"{name} must be rank {ndim}, got rank {value.ndim}."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise HFCausalKindTranslationError(f"{name} contains non-finite values.")
    return result


def _mask(value: Any, name: str, shape: tuple[int, int]) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise HFCausalKindTranslationError(f"{name} must have shape {shape}.")
    return value.detach().to(device="cpu", dtype=torch.bool).contiguous()


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _report(body: Mapping[str, Any]) -> Mapping[str, Any]:
    # Round-trip first so tensor/numpy scalars cannot enter receipts.
    decoded = json.loads(json.dumps(dict(body), allow_nan=False, sort_keys=True))
    decoded["report_sha256"] = canonical_sha256(decoded)
    return MappingProxyType(decoded)


def _relative_rmse(prediction: Any, target: Any) -> float:
    numerator = (prediction - target).square().mean().sqrt()
    denominator = target.square().mean().sqrt().clamp_min(1e-30)
    return float((numerator / denominator).item())


def _retained_energy(prediction: Any, target: Any) -> float:
    error = (prediction - target).square().sum()
    total = target.square().sum().clamp_min(1e-30)
    return float((1.0 - error / total).clamp(min=0.0, max=1.0).item())


def _energy_ratio(part: Any, total: Any) -> float:
    numerator = part.square().sum()
    denominator = total.square().sum().clamp_min(1e-30)
    return float((numerator / denominator).item())


def _masked_rows(value: Any, mask: Any) -> Any:
    trailing = tuple(int(size) for size in value.shape[2:])
    return value.reshape(int(mask.numel()), *trailing)[mask.reshape(-1)].contiguous()


def _balanced_partition(total: int, count: int) -> tuple[tuple[int, ...], ...]:
    """Deterministic contiguous balanced groups, front-loaded remainder."""

    total = _positive_int(total, "partition total")
    count = _positive_int(count, "partition count")
    if count > total:
        raise HFCausalKindTranslationError(
            "A balanced partition cannot have more groups than members."
        )
    quotient, remainder = divmod(total, count)
    groups: list[tuple[int, ...]] = []
    cursor = 0
    for index in range(count):
        width = quotient + (1 if index < remainder else 0)
        groups.append(tuple(range(cursor, cursor + width)))
        cursor += width
    return tuple(groups)


def _proportional_partition(
    total: int, weights: Sequence[float]
) -> tuple[tuple[int, ...], ...]:
    """Contiguous groups sized by ``weights`` with deterministic largest remainder.

    Ties break toward the earlier (shorter) lag, which is the order the FIR
    itself is written in, so the allocation is reproducible byte for byte.
    """

    total = _positive_int(total, "partition total")
    values = [max(0.0, float(value)) for value in weights]
    if not values:
        raise HFCausalKindTranslationError("A proportional partition needs weights.")
    mass = sum(values)
    if mass <= 0.0:
        return _balanced_partition(total, len(values))
    exact = [value * total / mass for value in values]
    counts = [int(math.floor(value)) for value in exact]
    remaining = total - sum(counts)
    order = sorted(
        range(len(values)),
        key=lambda index: (-(exact[index] - counts[index]), index),
    )
    for position in range(remaining):
        counts[order[position % len(order)]] += 1
    groups: list[tuple[int, ...]] = []
    cursor = 0
    for count in counts:
        groups.append(tuple(range(cursor, cursor + count)))
        cursor += count
    return tuple(groups)


def _lag_shift(value: Any, lag: int) -> Any:
    """Shift a ``[batch, sequence, width]`` stream ``lag`` positions later."""

    if lag == 0:
        return value
    torch = _torch()
    sequence = int(value.shape[1])
    if lag >= sequence:
        return torch.zeros_like(value)
    head = torch.zeros_like(value[:, :lag])
    return torch.cat((head, value[:, : sequence - lag]), dim=1)


def _disjoint_partitions(
    *,
    valid: Any,
    fit: Any,
    validation: Any,
    selection: Any | None,
) -> None:
    parts = {"fit": fit, "validation": validation}
    if selection is not None:
        parts["selection"] = selection
    for name, part in parts.items():
        if not bool(part.any().item()):
            raise HFCausalKindTranslationError(f"The {name} partition is empty.")
        if bool((part & ~valid).any().item()):
            raise HFCausalKindTranslationError(
                f"The {name} partition contains invalid tokens."
            )
    names = sorted(parts)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            if bool((parts[first] & parts[second]).any().item()):
                raise HFCausalKindTranslationError(
                    f"The {first} and {second} partitions overlap."
                )


# ---------------------------------------------------------------------------
# Condition-capped, equilibrated, trace-normalized ridge (v5 recipe)
# ---------------------------------------------------------------------------


def _condition_capped_solve(
    gram: Any,
    right: Any,
    *,
    alpha: float,
    condition_cap: float,
    rank_relative_tolerance: float = 1e-12,
) -> tuple[Any, dict[str, Any]]:
    """Solve one symmetric normal-equation system under the v5 numeric policy."""

    torch = _torch()
    width = int(gram.shape[0])
    diagonal = torch.diagonal(gram).clamp_min(0.0)
    scale = diagonal.sqrt()
    maximum_scale = float(scale.max().item())
    threshold = (
        maximum_scale * rank_relative_tolerance
        if maximum_scale > 0.0
        else rank_relative_tolerance
    )
    active = scale > threshold
    equilibration = torch.where(active, scale, torch.ones_like(scale))
    equilibrated = gram / (equilibration.reshape(-1, 1) * equilibration.reshape(1, -1))
    equilibrated = 0.5 * (equilibrated + equilibrated.T)
    equilibrated_right = right / equilibration.reshape(-1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(equilibrated)
    eigenvalues = eigenvalues.clamp(min=0.0)
    maximum_eigenvalue = float(eigenvalues.max().item())
    minimum_eigenvalue = float(eigenvalues.min().item())
    if maximum_eigenvalue == 0.0:
        raise HFCausalKindTranslationError(
            "A ridge sub-solve received an entirely degenerate design."
        )
    cap_ridge = max(
        0.0,
        (maximum_eigenvalue - condition_cap * minimum_eigenvalue)
        / (condition_cap - 1.0),
    )
    trace_ridge = alpha * float(eigenvalues.sum().item()) / float(width)
    ridge = max(cap_ridge, trace_ridge)
    rank_threshold = maximum_eigenvalue * rank_relative_tolerance
    identifiable = eigenvalues > rank_threshold
    inverse = torch.zeros_like(eigenvalues)
    inverse[identifiable] = 1.0 / (eigenvalues[identifiable] + ridge)
    solution = eigenvectors @ (inverse.unsqueeze(-1) * (eigenvectors.T @ equilibrated_right))
    solution = solution / equilibration.reshape(-1, 1)
    solution[~active, :] = 0.0
    regularized_condition = (maximum_eigenvalue + ridge) / (minimum_eigenvalue + ridge)
    diagnostics = {
        "feature_width": width,
        "active_columns": int(active.sum().item()),
        "effective_rank": int(identifiable.sum().item()),
        "unregularized_condition": (
            float("inf")
            if minimum_eigenvalue <= 0.0
            else maximum_eigenvalue / minimum_eigenvalue
        ),
        "regularized_condition": regularized_condition,
        "condition_cap": float(condition_cap),
        "condition_cap_passed": bool(
            regularized_condition <= condition_cap * (1.0 + 1e-9)
        ),
        "cap_ridge": cap_ridge,
        "trace_ridge": trace_ridge,
        "ridge": ridge,
        "coefficient_frobenius_norm": float(solution.norm().item()),
    }
    if not diagnostics["condition_cap_passed"]:
        raise HFCausalKindTranslationError(
            "The condition cap was exhausted for a ridge sub-solve."
        )
    return solution.contiguous(), diagnostics


def _ridge_fit(
    design: Any,
    target: Any,
    *,
    alpha: float,
    condition_cap: float,
    fit_intercept: bool,
) -> tuple[Any, Any | None, dict[str, Any]]:
    """Least squares from explicit rows, with an unpenalized intercept."""

    torch = _torch()
    if int(design.shape[0]) != int(target.shape[0]):
        raise HFCausalKindTranslationError("Ridge rows are incompatible.")
    rows = float(int(design.shape[0]))
    if fit_intercept:
        design_mean = design.mean(dim=0)
        target_mean = target.mean(dim=0)
        centered_design = design - design_mean
        centered_target = target - target_mean
    else:
        design_mean = None
        target_mean = None
        centered_design = design
        centered_target = target
    gram = (centered_design.T @ centered_design) / rows
    right = (centered_design.T @ centered_target) / rows
    weight, diagnostics = _condition_capped_solve(
        gram, right, alpha=alpha, condition_cap=condition_cap
    )
    bias = None
    if fit_intercept:
        assert design_mean is not None and target_mean is not None
        bias = target_mean - design_mean @ weight
    diagnostics["fit_intercept"] = bool(fit_intercept)
    diagnostics["rows"] = int(rows)
    return weight, bias, diagnostics


def _directional_ridge(
    design: Any,
    residual: Any,
    direction: Any,
    *,
    alpha: float,
    condition_cap: float,
) -> tuple[Any | None, dict[str, Any]]:
    """Exact sufficient-statistic solve for ``min || R - (D b) w^T ||_F^2``.

    This is the per-channel normal-equation accumulation the failure record
    mandates (``G = (||w||^2/O)(J^T J/n)``, ``q = J^T (R w)/(n O)``); the shared
    ``1/O`` cancels between the two sides and is omitted.  A channel whose
    writer column or design is identically zero contributes nothing to the
    endpoint and is reported as skipped rather than zeroed.
    """

    rows = float(int(design.shape[0]))
    weight_energy = float(direction.square().sum().item())
    design_energy = float(design.square().sum().item())
    if weight_energy <= 0.0 or design_energy <= 0.0:
        return None, {
            "feature_width": int(design.shape[1]),
            "skipped_degenerate_channel": True,
            "regularized_condition": 0.0,
            "direction_energy": weight_energy,
        }
    gram = weight_energy * (design.T @ design) / rows
    right = (design.T @ (residual @ direction)).reshape(-1, 1) / rows
    solution, diagnostics = _condition_capped_solve(
        gram, right, alpha=alpha, condition_cap=condition_cap
    )
    diagnostics["skipped_degenerate_channel"] = False
    diagnostics["direction_energy"] = weight_energy
    return solution.reshape(-1), diagnostics


# ---------------------------------------------------------------------------
# ShortConv branch semantics
# ---------------------------------------------------------------------------


def _short_conv_evaluate(
    *,
    input_projection_weight: Any,
    input_projection_bias: Any | None,
    lag_kernel: Any,
    conv_bias: Any | None,
    output_projection_weight: Any,
    output_projection_bias: Any | None,
    sequences: Any,
) -> dict[str, Any]:
    """Evaluate the contract-pinned ShortConv semantics in the input dtype.

    ``(B_t, C_t, x_t) = chunk_3(in_proj(h_t))``, ``r_t = B_t * x_t``,
    ``v_t = conv_bias + sum_lag k_lag * r_(t-lag)``,
    ``delta_t = out_proj(C_t * v_t)`` (``hf_lfm2_canonical.py:100-105``).
    """

    projected = sequences @ input_projection_weight.transpose(0, 1)
    if input_projection_bias is not None:
        projected = projected + input_projection_bias
    gate_b, gate_c, stream_x = projected.chunk(3, dim=-1)
    gated = gate_b * stream_x
    filtered = None
    for lag in range(int(lag_kernel.shape[1])):
        term = _lag_shift(gated, lag) * lag_kernel[:, lag]
        filtered = term if filtered is None else filtered + term
    assert filtered is not None
    if conv_bias is not None:
        filtered = filtered + conv_bias
    features = gate_c * filtered
    delta = features @ output_projection_weight.transpose(0, 1)
    if output_projection_bias is not None:
        delta = delta + output_projection_bias
    return {
        "b": gate_b,
        "c": gate_c,
        "x": stream_x,
        "r": gated,
        "v": filtered,
        "features": features,
        "delta": delta,
    }


@dataclass(frozen=True, slots=True)
class ShortConvBranchParameters:
    """One ShortConv branch in the exact LFM2 layout, possibly heterogeneous."""

    input_projection_weight: Any
    conv_weight: Any
    output_projection_weight: Any
    input_projection_bias: Any | None = None
    conv_bias: Any | None = None
    output_projection_bias: Any | None = None

    def __post_init__(self) -> None:
        weight = _tensor(
            self.input_projection_weight, "input_projection_weight", ndim=2
        ).clone()
        kernel = _tensor(self.conv_weight, "conv_weight", ndim=3).clone()
        writer = _tensor(
            self.output_projection_weight, "output_projection_weight", ndim=2
        ).clone()
        channel_width = int(writer.shape[1])
        hidden_size = int(writer.shape[0])
        if int(weight.shape[0]) != 3 * channel_width:
            raise HFCausalKindTranslationError(
                "input_projection_weight must have shape [3D, H]."
            )
        if int(weight.shape[1]) != hidden_size:
            raise HFCausalKindTranslationError(
                "input_projection_weight input width must equal the hidden size."
            )
        if tuple(kernel.shape[:2]) != (channel_width, 1) or int(kernel.shape[2]) < 1:
            raise HFCausalKindTranslationError("conv_weight must have shape [D, 1, L].")
        object.__setattr__(self, "input_projection_weight", weight)
        object.__setattr__(self, "conv_weight", kernel)
        object.__setattr__(self, "output_projection_weight", writer)
        biases = (
            self.input_projection_bias,
            self.conv_bias,
            self.output_projection_bias,
        )
        if any(value is None for value in biases) and not all(
            value is None for value in biases
        ):
            raise HFCausalKindTranslationError(
                "All ShortConv biases must be present or absent together."
            )
        if biases[0] is not None:
            widths = (3 * channel_width, channel_width, hidden_size)
            names = ("input_projection_bias", "conv_bias", "output_projection_bias")
            for name, width in zip(names, widths):
                vector = _tensor(getattr(self, name), name, ndim=1).clone()
                if int(vector.numel()) != width:
                    raise HFCausalKindTranslationError(f"{name} has the wrong width.")
                object.__setattr__(self, name, vector)

    @property
    def hidden_size(self) -> int:
        return int(self.output_projection_weight.shape[0])

    @property
    def channel_width(self) -> int:
        return int(self.output_projection_weight.shape[1])

    @property
    def fir_length(self) -> int:
        return int(self.conv_weight.shape[2])

    @property
    def bias(self) -> bool:
        return self.input_projection_bias is not None

    @property
    def lag_kernel(self) -> Any:
        """``k[:, lag] = conv.weight[:, 0, L-1-lag]`` (the pinned lag order)."""

        return self.conv_weight[:, 0, :].flip(-1).contiguous()

    def evaluate(self, sequences: Any, *, dtype: Any = None) -> dict[str, Any]:
        torch = _torch()
        rows = _tensor(sequences, "short_conv_input_sequences", ndim=3)
        if int(rows.shape[2]) != self.hidden_size:
            raise HFCausalKindTranslationError(
                "ShortConv input width differs from the branch hidden size."
            )
        target_dtype = torch.float64 if dtype is None else dtype

        def cast(value: Any) -> Any:
            return None if value is None else value.to(dtype=target_dtype)

        return _short_conv_evaluate(
            input_projection_weight=cast(self.input_projection_weight),
            input_projection_bias=cast(self.input_projection_bias),
            lag_kernel=cast(self.lag_kernel),
            conv_bias=cast(self.conv_bias),
            output_projection_weight=cast(self.output_projection_weight),
            output_projection_bias=cast(self.output_projection_bias),
            sequences=rows.to(dtype=target_dtype),
        )

    def forward(self, sequences: Any, *, dtype: Any = None) -> Any:
        return self.evaluate(sequences, dtype=dtype)["delta"]

    def tensor_fingerprints(self) -> dict[str, str]:
        values = {
            "input_projection_weight": self.input_projection_weight,
            "conv_weight": self.conv_weight,
            "output_projection_weight": self.output_projection_weight,
        }
        if self.bias:
            values["input_projection_bias"] = self.input_projection_bias
            values["conv_bias"] = self.conv_bias
            values["output_projection_bias"] = self.output_projection_bias
        return {name: _tensor_sha256(value) for name, value in sorted(values.items())}


def short_conv_parameters_from_solve(
    *,
    gate_b_weight: Any,
    gate_c_weight: Any,
    stream_x_weight: Any,
    lag_kernel: Any,
    output_weight: Any,
    gate_b_bias: Any | None = None,
    gate_c_bias: Any | None = None,
    stream_x_bias: Any | None = None,
    conv_bias: Any | None = None,
    output_bias: Any | None = None,
) -> ShortConvBranchParameters:
    """Assemble the LFM2 physical layout from the solved factor tensors."""

    torch = _torch()
    input_projection = torch.cat(
        (gate_b_weight, gate_c_weight, stream_x_weight), dim=0
    ).contiguous()
    input_bias = None
    if gate_b_bias is not None:
        input_bias = torch.cat((gate_b_bias, gate_c_bias, stream_x_bias), dim=0)
    kernel = lag_kernel.flip(-1).unsqueeze(1).contiguous()
    return ShortConvBranchParameters(
        input_projection_weight=input_projection,
        conv_weight=kernel,
        output_projection_weight=output_weight,
        input_projection_bias=input_bias,
        conv_bias=conv_bias,
        output_projection_bias=output_bias,
    )


# ---------------------------------------------------------------------------
# Direction B: ShortConv -> attention
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShortConvLinearization:
    """The banded-linear-attention image of one ShortConv branch."""

    lag_maps: tuple[Any, ...]
    self_gate_map: Any
    constant_rows: Any
    gate_b_mean: Any
    gate_c_mean: Any
    stream_x_mean: Any
    context_v_mean: Any
    banded_delta: Any
    quadratic_remainder: Any
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lag_maps", tuple(self.lag_maps))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


def linearize_short_conv_branch(
    parameters: ShortConvBranchParameters,
    sequences: Any,
    *,
    row_mask: Any,
) -> ShortConvLinearization:
    """Expand a ShortConv branch bilinearly around its evidence means.

    ``Delta_t = sum_l n_{t-l} M_l^T + n_t M_self^T + c_t + eps_t`` with

    ``M_l = W_out diag(Cbar * k_l) (diag(Bbar) W_x + diag(xbar) W_B)``,
    ``M_self = W_out diag(vbar) W_C`` and
    ``eps_t = W_out [Cbar * (sum_l k_l * eps_B,t-l) + eps_C,t]``.

    Boundary truncation is preserved exactly: at ``t < l`` the FIR has no
    ``r_{t-l}`` and neither does the expansion, which makes ``c_t`` depend on
    position for the first ``L-1`` rows only.  Under gate collapse
    (``W_B = W_C = 0``) ``eps`` is identically zero and the expansion reproduces
    the branch exactly.
    """

    torch = _torch()
    rows = _tensor(sequences, "short_conv_input_sequences", ndim=3)
    batch, sequence = int(rows.shape[0]), int(rows.shape[1])
    mask = _mask(row_mask, "row_mask", (batch, sequence))
    if not bool(mask.any().item()):
        raise HFCausalKindTranslationError(
            "Linearizing a ShortConv branch requires at least one evidence row."
        )
    evaluated = parameters.evaluate(rows)
    hidden = parameters.hidden_size
    channels = parameters.channel_width
    length = parameters.fir_length
    kernel = parameters.lag_kernel
    writer = parameters.output_projection_weight
    weight_b, weight_c, weight_x = parameters.input_projection_weight.chunk(3, dim=0)
    if parameters.bias:
        bias_b, bias_c, bias_x = parameters.input_projection_bias.chunk(3, dim=0)
        conv_bias = parameters.conv_bias
        output_bias = parameters.output_projection_bias
    else:
        zeros = torch.zeros(channels, dtype=torch.float64)
        bias_b, bias_c, bias_x = zeros, zeros, zeros
        conv_bias = zeros
        output_bias = torch.zeros(hidden, dtype=torch.float64)

    mean_b = _masked_rows(evaluated["b"], mask).mean(dim=0)
    mean_c = _masked_rows(evaluated["c"], mask).mean(dim=0)
    mean_x = _masked_rows(evaluated["x"], mask).mean(dim=0)
    mean_v = _masked_rows(evaluated["v"], mask).mean(dim=0)

    # r_lin_s = n_s A^T + a   (affine in the operator-norm row)
    affine_weight = mean_b.reshape(-1, 1) * weight_x + mean_x.reshape(-1, 1) * weight_b
    affine_bias = mean_b * bias_x + bias_b * mean_x - mean_b * mean_x

    lag_maps = tuple(
        (writer * (mean_c * kernel[:, lag]).reshape(1, -1)) @ affine_weight
        for lag in range(length)
    )
    self_gate_map = (writer * mean_v.reshape(1, -1)) @ weight_c

    constant_rows = torch.zeros(sequence, hidden, dtype=torch.float64)
    for position in range(sequence):
        accumulated = conv_bias.clone()
        for lag in range(min(length, position + 1)):
            accumulated = accumulated + kernel[:, lag] * affine_bias
        channelwise = mean_c * accumulated - mean_c * mean_v + bias_c * mean_v
        constant_rows[position] = channelwise @ writer.T + output_bias

    banded = constant_rows.reshape(1, sequence, hidden).expand(batch, -1, -1).clone()
    for lag in range(length):
        banded = banded + _lag_shift(rows, lag) @ lag_maps[lag].T
    banded = banded + rows @ self_gate_map.T

    delta = evaluated["delta"]
    remainder = delta - banded

    deviation_b = evaluated["b"] - mean_b
    deviation_c = evaluated["c"] - mean_c
    deviation_x = evaluated["x"] - mean_x
    fit_delta = _masked_rows(delta, mask)
    dispersion_b = float(
        (
            _masked_rows(deviation_b, mask).norm()
            / _masked_rows(evaluated["b"], mask).norm().clamp_min(1e-30)
        ).item()
    )
    dispersion_c = float(
        (
            _masked_rows(deviation_c, mask).norm()
            / _masked_rows(evaluated["c"], mask).norm().clamp_min(1e-30)
        ).item()
    )
    constant_energy = (
        constant_rows.reshape(1, sequence, hidden).expand(batch, -1, -1).contiguous()
    )
    diagnostics = {
        "hidden_size": hidden,
        "channel_width": channels,
        "fir_length": length,
        "bias": bool(parameters.bias),
        "evidence_rows": int(mask.sum().item()),
        "gate_b_variance_mean": float(
            _masked_rows(deviation_b, mask).square().mean().item()
        ),
        "gate_c_variance_mean": float(
            _masked_rows(deviation_c, mask).square().mean().item()
        ),
        "stream_x_variance_mean": float(
            _masked_rows(deviation_x, mask).square().mean().item()
        ),
        "gate_b_relative_dispersion": dispersion_b,
        "gate_c_relative_dispersion": dispersion_c,
        "gate_collapse_measure": max(dispersion_b, dispersion_c),
        "quadratic_remainder_relative_energy": _energy_ratio(
            _masked_rows(remainder, mask), fit_delta
        ),
        "quadratic_remainder_relative_rmse": _relative_rmse(
            _masked_rows(banded, mask), fit_delta
        ),
        "linearized_constant_relative_energy": _energy_ratio(
            _masked_rows(constant_energy, mask), fit_delta
        ),
        "lag_map_frobenius_norms": [float(value.norm().item()) for value in lag_maps],
        "self_gate_map_frobenius_norm": float(self_gate_map.norm().item()),
        "boundary_constant_rows": min(length - 1, sequence),
    }
    return ShortConvLinearization(
        lag_maps=lag_maps,
        self_gate_map=self_gate_map,
        constant_rows=constant_rows,
        gate_b_mean=mean_b,
        gate_c_mean=mean_c,
        stream_x_mean=mean_x,
        context_v_mean=mean_v,
        banded_delta=banded,
        quadratic_remainder=remainder,
        diagnostics=diagnostics,
    )


def _canonical_svd(matrix: Any) -> tuple[Any, Any, Any]:
    """SVD with a deterministic sign convention on every singular direction."""

    torch = _torch()
    left, values, right = torch.linalg.svd(matrix, full_matrices=False)
    count = int(values.numel())
    for index in range(count):
        column = left[:, index]
        pivot = int(torch.argmax(column.abs()).item())
        if float(column[pivot].item()) < 0.0:
            left[:, index] = -column
            right[index, :] = -right[index, :]
    return left, values, right


def _factorize_stream_map(matrix: Any, rank_budget: int) -> tuple[Any, Any, float]:
    """Split ``M`` into value rows and a writer block at the available rank."""

    torch = _torch()
    hidden = int(matrix.shape[0])
    left, values, right = _canonical_svd(matrix)
    rank = min(rank_budget, int(values.numel()))
    root = values[:rank].clamp_min(0.0).sqrt()
    value_rows = root.reshape(-1, 1) * right[:rank, :]
    writer_block = left[:, :rank] * root.reshape(1, -1)
    if rank < rank_budget:
        pad_rows = torch.zeros(rank_budget - rank, int(matrix.shape[1]), dtype=torch.float64)
        pad_writer = torch.zeros(hidden, rank_budget - rank, dtype=torch.float64)
        value_rows = torch.cat((value_rows, pad_rows), dim=0)
        writer_block = torch.cat((writer_block, pad_writer), dim=1)
    residual = matrix - writer_block @ value_rows
    total = float(matrix.norm().item())
    relative = 0.0 if total == 0.0 else float(residual.norm().item()) / total
    return value_rows.contiguous(), writer_block.contiguous(), relative


def _rope_band_preimage(
    *,
    head_dim: int,
    rope_theta: float,
    attention_scaling: float,
    window: int,
    lag: int,
    peak_logits: float,
    ridge_alpha_candidates: Sequence[float],
) -> tuple[Any, Any, dict[str, Any]]:
    """Constant pre-RoPE ``q0``/``k0`` whose relative score peaks at ``lag``.

    With ``q0`` supported on the first half of every rotation pair the post-RoPE
    score is the cosine series ``g(d) = sum_j A_j cos(w_j d) + B_j sin(w_j d)``
    with ``(A_j, B_j) = (k0_j, k0_{j+half})``, so the band selector is a small
    deterministic ridge least-squares problem in ``head_dim`` coefficients.

    The RoPE frequency ladder is geometric, so an unregularized delta target is
    badly ill-posed: the coefficient norm explodes and the *achieved* margin per
    unit of Q/K projection error collapses.  The score perturbation caused by a
    Q/K fit error ``d`` is bounded by ``scaling * ||d|| * ||k0||`` while the
    margin is bounded by ``scaling * ||q0|| * ||k0|| * c``, so the quantity that
    decides whether the compiled head keeps its band is the scale-invariant
    normalized margin ``c = margin / (||q0|| ||k0||)``.  The ridge is therefore
    selected from a frozen ladder by maximizing ``c``.  That criterion depends
    only on ``head_dim``, ``rope_theta``, ``window`` and ``lag`` — no evidence
    row of any partition enters it.
    """

    torch = _torch()
    half = head_dim // 2
    lags = torch.arange(0, window + 1, dtype=torch.float64).reshape(-1, 1)
    frequencies = rope_theta ** (
        -torch.arange(0, head_dim, 2, dtype=torch.float64) / float(head_dim)
    ).reshape(1, -1)
    angles = lags * frequencies
    design = torch.cat((angles.cos(), angles.sin()), dim=1)
    target = torch.zeros(window + 1, 1, dtype=torch.float64)
    target[lag, 0] = 1.0
    gram = design.T @ design
    right = design.T @ target
    trace = float(torch.diagonal(gram).sum().item())
    query_norm = math.sqrt(float(half))
    identity = torch.eye(2 * half, dtype=torch.float64)
    ladder: list[dict[str, Any]] = []
    best: tuple[float, float, Any] | None = None
    for candidate in ridge_alpha_candidates:
        alpha = _positive_float(candidate, "rope series ridge alpha")
        ridge = alpha * trace / float(2 * half)
        solution = torch.linalg.solve(gram + ridge * identity, right)
        values = (design @ solution).reshape(-1)
        peak = float(values[lag].item())
        norm = float(solution.norm().item())
        others = torch.cat((values[:lag], values[lag + 1 :]))
        competitor = float(others.max().item()) if int(others.numel()) else -math.inf
        margin = peak - competitor
        normalized = (
            -math.inf if norm <= 0.0 or peak <= 0.0 else margin / (norm * query_norm)
        )
        ladder.append(
            {
                "ridge_alpha": alpha,
                "coefficient_norm": norm,
                "normalized_margin": (
                    None if normalized == -math.inf else normalized
                ),
            }
        )
        if peak <= 0.0 or margin <= 0.0:
            continue
        if best is None or normalized > best[0]:
            best = (normalized, peak, solution)
    if best is None:
        raise HFCausalKindTranslationError(
            f"No RoPE band selector for lag {lag} dominates its window."
        )
    normalized_margin, peak, coefficients = best
    gain = (peak_logits / attention_scaling) / peak
    coefficients = coefficients * gain
    scaled_target = target.reshape(-1) * gain
    query = torch.zeros(head_dim, dtype=torch.float64)
    query[:half] = 1.0
    key = coefficients.reshape(-1).clone()

    positions = torch.arange(0, window + 1, dtype=torch.float64).reshape(1, -1)
    query_rows = query.reshape(1, 1, 1, -1).expand(1, 1, window + 1, head_dim).contiguous()
    key_rows = key.reshape(1, 1, 1, -1).expand(1, 1, window + 1, head_dim).contiguous()
    rotated_query = apply_rope(query_rows, positions, theta=rope_theta)
    rotated_key = apply_rope(key_rows, positions, theta=rope_theta)
    realized = attention_scaling * (rotated_query[0, 0] @ rotated_key[0, 0][0])
    predicted = attention_scaling * (design @ coefficients).reshape(-1)
    construction_error = float((realized - predicted).abs().max().item())
    if construction_error > 1e-6 * max(1.0, float(realized.abs().max().item())):
        raise HFCausalKindTranslationError(
            "The RoPE cosine-series preimage does not reproduce its own scores."
        )
    others = torch.cat((realized[:lag], realized[lag + 1 :]))
    competitor = float(others.max().item()) if int(others.numel()) else -math.inf
    margin = float(realized[lag].item()) - competitor
    if margin <= 0.0:
        raise HFCausalKindTranslationError(
            f"The RoPE band selector for lag {lag} does not dominate its window."
        )
    exponent = -margin
    leak = 1.0 if exponent > 700.0 else min(1.0, float(window) * math.exp(exponent))
    diagnostics = {
        "lag": lag,
        "window": window,
        "peak_logits": float(realized[lag].item()),
        "margin_logits": margin,
        "normalized_margin": normalized_margin,
        "selected_series_ridge_alpha": next(
            entry["ridge_alpha"]
            for entry in ladder
            if entry["normalized_margin"] is not None
            and entry["normalized_margin"] == normalized_margin
        ),
        "series_ridge_ladder": ladder,
        "key_preimage_norm": float(key.norm().item()),
        "query_preimage_norm": float(query.norm().item()),
        "predicted_band_leak_upper_bound": leak,
        "series_construction_max_absolute_error": construction_error,
        "series_relative_residual": float(
            ((design @ coefficients).reshape(-1) - scaled_target).norm().item()
        )
        / max(1e-30, float(scaled_target.norm().item())),
    }
    return query.contiguous(), key.contiguous(), diagnostics


@dataclass(frozen=True, slots=True)
class ShortConvToAttentionPolicy:
    """Frozen policy constants for the conv-to-attention translation."""

    band_logit_peak: float = 24.0
    self_gate_stream: str = "auto"
    self_gate_relative_floor: float = 1e-12
    rope_series_ridge_alpha_candidates: tuple[float, ...] = (
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        3e-1,
        1.0,
        3.0,
    )
    minimum_assigned_lag_probability_mass: float = 0.5
    gate_collapse_tolerance: float = 1e-12
    maximum_unrouted_quadratic_mass: float = 1e-9
    ridge: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "band_logit_peak", _positive_float(self.band_logit_peak, "band_logit_peak")
        )
        mode = str(self.self_gate_stream).strip().lower()
        if mode not in {"auto", "on", "off"}:
            raise HFCausalKindTranslationError(
                "self_gate_stream must be 'auto', 'on', or 'off'."
            )
        object.__setattr__(self, "self_gate_stream", mode)
        candidates = tuple(
            _positive_float(value, "rope series ridge alpha")
            for value in self.rope_series_ridge_alpha_candidates
        )
        if not candidates:
            raise HFCausalKindTranslationError(
                "The RoPE band-selector ridge ladder cannot be empty."
            )
        object.__setattr__(self, "rope_series_ridge_alpha_candidates", candidates)
        for name in (
            "self_gate_relative_floor",
            "minimum_assigned_lag_probability_mass",
            "gate_collapse_tolerance",
            "maximum_unrouted_quadratic_mass",
            "ridge",
        ):
            object.__setattr__(self, name, _nonnegative_float(getattr(self, name), name))

    def as_dict(self) -> dict[str, Any]:
        return {
            "band_logit_peak": self.band_logit_peak,
            "self_gate_stream": self.self_gate_stream,
            "self_gate_relative_floor": self.self_gate_relative_floor,
            "rope_series_ridge_alpha_candidates": list(
                self.rope_series_ridge_alpha_candidates
            ),
            "minimum_assigned_lag_probability_mass": (
                self.minimum_assigned_lag_probability_mass
            ),
            "gate_collapse_tolerance": self.gate_collapse_tolerance,
            "maximum_unrouted_quadratic_mass": self.maximum_unrouted_quadratic_mass,
            "ridge": self.ridge,
        }


@dataclass(frozen=True, slots=True)
class ShortConvToAttentionTranslation:
    """One compiled conv-to-attention branch with its sealed receipt."""

    fold: HFCausalCoupledAttentionFold
    linearization: ShortConvLinearization
    remainder: Any
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))


def _replay_attention_bf16(
    fold: HFCausalCoupledAttentionFold,
    child: Any,
    *,
    attention_mask: Any,
    position_ids: Any,
) -> Any:
    """Replay the compiled attention with every intermediate in BF16."""

    torch = _torch()
    topology = fold.target_topology
    batch, sequence = int(child.shape[0]), int(child.shape[1])
    rows = child.to(dtype=torch.bfloat16)

    def project(weight: Any, heads: int) -> Any:
        projected = rows @ weight.to(dtype=torch.bfloat16).transpose(0, 1)
        return (
            projected.reshape(batch, sequence, heads, topology.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    query = project(fold.query_weight, topology.query_heads)
    key = project(fold.key_weight, topology.key_value_heads)
    value = project(fold.value_weight, topology.key_value_heads)
    if topology.query_key_norm == "rms_norm":

        def head_norm(value_rows: Any, gain: Any) -> Any:
            scale = (
                value_rows.to(dtype=torch.float32).square().mean(dim=-1, keepdim=True)
                + topology.rms_norm_epsilon
            ).rsqrt().to(dtype=torch.bfloat16)
            return value_rows * scale * gain.to(dtype=torch.bfloat16).reshape(1, 1, 1, -1)

        query = head_norm(query, fold.query_norm_weight)
        key = head_norm(key, fold.key_norm_weight)
    positions = position_ids.detach().to(device="cpu")
    query = apply_rope(query.to(dtype=torch.float64), positions, theta=topology.rope_theta).to(
        dtype=torch.bfloat16
    )
    key = apply_rope(key.to(dtype=torch.float64), positions, theta=topology.rope_theta).to(
        dtype=torch.bfloat16
    )
    repeats = topology.query_heads // topology.key_value_heads
    keys = key.repeat_interleave(repeats, dim=1)
    scores = torch.matmul(query, keys.transpose(-1, -2)) * torch.tensor(
        topology.attention_scaling, dtype=torch.bfloat16
    )
    key_mask = attention_mask.reshape(batch, 1, 1, sequence).to(torch.bool)
    causal = torch.ones(sequence, sequence, dtype=torch.bool).tril().reshape(
        1, 1, sequence, sequence
    )
    allowed = key_mask & causal
    no_key = ~allowed.any(dim=-1, keepdim=True)
    if bool(no_key.any().item()):
        diagonal = torch.eye(sequence, dtype=torch.bool).reshape(1, 1, sequence, sequence)
        allowed = allowed | (no_key & diagonal)
    probabilities = torch.softmax(
        scores.masked_fill(~allowed, float("-inf")), dim=-1
    )
    probabilities = probabilities * attention_mask.reshape(batch, 1, sequence, 1).to(
        torch.bool
    )
    contexts = torch.matmul(probabilities, value.repeat_interleave(repeats, dim=1))
    concatenated = contexts.permute(0, 2, 1, 3).reshape(
        batch, sequence, topology.query_width
    )
    delta = concatenated @ fold.output_weight.to(dtype=torch.bfloat16).transpose(0, 1)
    return delta.to(dtype=torch.float64)


def compile_short_conv_to_attention_translation(
    parent_normalized_sequences: Any,
    *,
    short_conv_parameters: ShortConvBranchParameters,
    target_topology: HFCausalGQATopology,
    attention_mask: Any,
    position_ids: Any,
    valid_token_mask: Any,
    fit_token_mask: Any,
    validation_token_mask: Any,
    selection_token_mask: Any | None = None,
    child_normalized_sequences: Any | None = None,
    target_residual_delta_sequences: Any | None = None,
    source_layer_index: int = 0,
    carrier_layer_index: int | None = None,
    evidence_cache_key: str | None = None,
    policy: ShortConvToAttentionPolicy | None = None,
) -> ShortConvToAttentionTranslation:
    """Translate one ShortConv branch into a compiled softmax GQA/RoPE branch."""

    torch = _torch()
    policy = policy or ShortConvToAttentionPolicy()
    if not isinstance(short_conv_parameters, ShortConvBranchParameters):
        raise HFCausalKindTranslationError(
            "short_conv_parameters must be a ShortConvBranchParameters value."
        )
    if not isinstance(target_topology, HFCausalGQATopology):
        raise HFCausalKindTranslationError("target_topology is invalid.")
    parent = _tensor(parent_normalized_sequences, "parent_normalized_sequences", ndim=3)
    child = (
        parent
        if child_normalized_sequences is None
        else _tensor(child_normalized_sequences, "child_normalized_sequences", ndim=3)
    )
    batch, sequence = int(parent.shape[0]), int(parent.shape[1])
    if tuple(child.shape[:2]) != (batch, sequence):
        raise HFCausalKindTranslationError(
            "Child replay rows must share the parent's batch and sequence shape."
        )
    if int(parent.shape[2]) != short_conv_parameters.hidden_size:
        raise HFCausalKindTranslationError(
            "Parent rows do not match the ShortConv branch hidden size."
        )
    mask = _mask(attention_mask, "attention_mask", (batch, sequence))
    valid = _mask(valid_token_mask, "valid_token_mask", (batch, sequence))
    fit = _mask(fit_token_mask, "fit_token_mask", (batch, sequence))
    validation = _mask(validation_token_mask, "validation_token_mask", (batch, sequence))
    selection = (
        None
        if selection_token_mask is None
        else _mask(selection_token_mask, "selection_token_mask", (batch, sequence))
    )
    _disjoint_partitions(
        valid=valid, fit=fit, validation=validation, selection=selection
    )
    positions = position_ids.detach().to(device="cpu")
    if tuple(positions.shape) != (batch, sequence):
        raise HFCausalKindTranslationError("position_ids has the wrong shape.")

    branch_delta = short_conv_parameters.forward(parent)
    if target_residual_delta_sequences is None:
        delta_target = branch_delta
        cached_delta_agreement = 0.0
    else:
        delta_target = _tensor(
            target_residual_delta_sequences, "target_residual_delta_sequences", ndim=3
        )
        if tuple(delta_target.shape) != tuple(branch_delta.shape):
            raise HFCausalKindTranslationError(
                "The cached branch delta shape differs from the replayed branch."
            )
        cached_delta_agreement = _relative_rmse(
            _masked_rows(branch_delta, fit), _masked_rows(delta_target, fit)
        )

    linearization = linearize_short_conv_branch(
        short_conv_parameters, parent, row_mask=fit
    )
    gate_collapsed = bool(
        linearization.diagnostics["gate_collapse_measure"]
        <= policy.gate_collapse_tolerance
    )
    quadratic_mass = float(
        linearization.diagnostics["quadratic_remainder_relative_energy"]
    )
    if (
        not gate_collapsed
        and carrier_layer_index is None
        and quadratic_mass > policy.maximum_unrouted_quadratic_mass
    ):
        raise HFCausalKindTranslationError(
            "Gate linearization leaves unrepresentable quadratic mass "
            f"({quadratic_mass:.6e}) and no routing carrier was declared."
        )

    length = short_conv_parameters.fir_length
    streams: list[dict[str, Any]] = [
        {"kind": "lag", "lag": lag, "map": linearization.lag_maps[lag]}
        for lag in range(length)
    ]
    self_norm = float(linearization.self_gate_map.norm().item())
    lag_norm = max(float(value.norm().item()) for value in linearization.lag_maps)
    if policy.self_gate_stream == "on":
        include_self = True
    elif policy.self_gate_stream == "off":
        include_self = False
    else:
        include_self = self_norm > policy.self_gate_relative_floor * max(lag_norm, 1e-30)
    if include_self:
        streams.append({"kind": "self_gate", "lag": 0, "map": linearization.self_gate_map})

    stream_count = len(streams)
    if target_topology.key_value_heads < stream_count:
        raise HFCausalKindTranslationError(
            "The target has fewer key/value heads "
            f"({target_topology.key_value_heads}) than translation streams "
            f"({stream_count}); a GQA group shares one key sequence and cannot "
            "carry two different lags."
        )
    if target_topology.query_heads < stream_count:
        raise HFCausalKindTranslationError(
            "The target has fewer query heads than translation streams."
        )

    kv_groups = _balanced_partition(target_topology.key_value_heads, stream_count)
    queries_per_kv = target_topology.queries_per_key_value
    head_dim = target_topology.head_dim
    window = sequence - 1
    hidden = short_conv_parameters.hidden_size

    desired_query = torch.zeros(
        batch, sequence, target_topology.query_heads * head_dim, dtype=torch.float64
    )
    desired_key = torch.zeros(
        batch, sequence, target_topology.key_value_heads * head_dim, dtype=torch.float64
    )
    desired_value = torch.zeros(
        batch, sequence, target_topology.key_value_heads * head_dim, dtype=torch.float64
    )
    stream_receipts: list[dict[str, Any]] = []
    stream_query_heads: list[tuple[int, ...]] = []
    for index, stream in enumerate(streams):
        group = kv_groups[index]
        rank_budget = len(group) * head_dim
        value_rows, writer_block, truncation = _factorize_stream_map(
            stream["map"], rank_budget
        )
        query_vector, key_vector, band = _rope_band_preimage(
            head_dim=head_dim,
            rope_theta=target_topology.rope_theta,
            attention_scaling=target_topology.attention_scaling,
            window=window,
            lag=int(stream["lag"]),
            peak_logits=policy.band_logit_peak,
            ridge_alpha_candidates=policy.rope_series_ridge_alpha_candidates,
        )
        heads: list[int] = []
        for local, kv_head in enumerate(group):
            key_slice = slice(kv_head * head_dim, (kv_head + 1) * head_dim)
            desired_key[:, :, key_slice] = key_vector.reshape(1, 1, -1)
            block = value_rows[local * head_dim : (local + 1) * head_dim, :]
            desired_value[:, :, key_slice] = parent @ block.T
            for local_query in range(queries_per_kv):
                head = kv_head * queries_per_kv + local_query
                heads.append(head)
                desired_query[
                    :, :, head * head_dim : (head + 1) * head_dim
                ] = query_vector.reshape(1, 1, -1)
        stream_query_heads.append(tuple(heads))
        stream_receipts.append(
            {
                "stream_index": index,
                "kind": stream["kind"],
                "lag": int(stream["lag"]),
                "key_value_heads": list(group),
                "query_heads": list(heads),
                "rank_budget": rank_budget,
                "map_frobenius_norm": float(stream["map"].norm().item()),
                "stream_rank_truncation_relative_error": truncation,
                "value_rows_sha256": _tensor_sha256(value_rows),
                "writer_block_sha256": _tensor_sha256(writer_block),
                **band,
            }
        )

    # A zero-padded FIR contributes nothing for lags that do not exist yet, but
    # every causal softmax row sums to one, so a lag-l head at position t < l
    # necessarily writes some other key's value.  Prefix rows are therefore
    # outside the operator class exactly as far mass is in direction A: they are
    # excluded from the endpoint solve ("not fittable, and must not be fitted"),
    # measured, and carried by the routed remainder.  Validation is untouched.
    representable = torch.zeros(batch, sequence, dtype=torch.bool)
    representable[:, min(length - 1, sequence) :] = True
    solve_fit = fit & representable
    if not bool(solve_fit.any().item()):
        raise HFCausalKindTranslationError(
            "Every fit row lies in the unrepresentable FIR prefix "
            f"(positions 0..{length - 2})."
        )
    source_topology = HFCausalGQATopology(
        query_heads=target_topology.query_heads,
        key_value_heads=target_topology.key_value_heads,
        head_dim=head_dim,
        rope_theta=target_topology.rope_theta,
        attention_scaling=target_topology.attention_scaling,
        query_key_norm="identity",
        rms_norm_epsilon=target_topology.rms_norm_epsilon,
    )
    fold = compile_hf_causal_gqa_attention_fold(
        child,
        source_query_sequences=desired_query,
        source_key_sequences=desired_key,
        source_value_sequences=desired_value,
        target_residual_delta_sequences=delta_target,
        attention_mask=mask,
        position_ids=positions,
        valid_token_mask=valid,
        fit_token_mask=solve_fit,
        validation_token_mask=validation,
        source_topology=source_topology,
        target_topology=target_topology,
        ridge=policy.ridge,
    )
    predicted, _scores, probabilities = fold.replay(
        child, attention_mask=mask, position_ids=positions
    )
    predicted_bf16 = _replay_attention_bf16(
        fold, child, attention_mask=mask, position_ids=positions
    )

    def assigned_mass(part: Any) -> list[float]:
        results: list[float] = []
        for index, stream in enumerate(streams):
            lag = int(stream["lag"])
            rows: list[Any] = []
            for head in stream_query_heads[index]:
                for position in range(lag, sequence):
                    selected = part[:, position]
                    if not bool(selected.any().item()):
                        continue
                    rows.append(probabilities[:, head, position, position - lag][selected])
            if not rows:
                results.append(0.0)
                continue
            results.append(float(torch.cat(rows).mean().item()))
        return results

    fit_mass = assigned_mass(solve_fit)
    validation_mass = assigned_mass(validation)
    minimum_validation_mass = min(validation_mass)
    if minimum_validation_mass < policy.minimum_assigned_lag_probability_mass:
        raise HFCausalKindTranslationError(
            "The replayed probabilities place only "
            f"{minimum_validation_mass:.6f} of the mass on an assigned lag; the "
            f"policy floor is {policy.minimum_assigned_lag_probability_mass}."
        )

    interior_validation = validation & representable
    prefix_validation = validation & ~representable
    fit_target = _masked_rows(delta_target, solve_fit)
    validation_target = _masked_rows(delta_target, validation)
    zero_baseline_fit = _relative_rmse(torch.zeros_like(fit_target), fit_target)
    zero_baseline_validation = _relative_rmse(
        torch.zeros_like(validation_target), validation_target
    )
    frozen_gate_fit = _relative_rmse(
        _masked_rows(linearization.banded_delta, solve_fit), fit_target
    )
    frozen_gate_validation = _relative_rmse(
        _masked_rows(linearization.banded_delta, validation), validation_target
    )
    validation_prediction_bf16 = _masked_rows(predicted_bf16, validation)
    validation_relative_rmse_bf16 = _relative_rmse(
        validation_prediction_bf16, validation_target
    )
    if validation_relative_rmse_bf16 >= zero_baseline_validation:
        raise HFCausalKindTranslationError(
            "The BF16 validation endpoint is at or above the no-branch baseline "
            f"({validation_relative_rmse_bf16:.6f} >= {zero_baseline_validation:.6f})."
        )

    remainder = delta_target - predicted
    remainder_bf16 = delta_target - predicted_bf16
    body = {
        "schema": SHORT_CONV_TO_ATTENTION_SCHEMA,
        "component": "short_conv_to_softmax_gqa_rope_branch_translation",
        "direction": "short_conv_to_attention",
        "method": "gate_mean_bilinear_linearization_then_synthesized_band_evidence",
        "reused_attention_kernel": "compile_hf_causal_gqa_attention_fold",
        "attention_kernel_report_sha256": fold.report["report_sha256"],
        "source_layer_index": int(source_layer_index),
        "carrier_layer_index": (
            None if carrier_layer_index is None else int(carrier_layer_index)
        ),
        "evidence_cache_key": evidence_cache_key,
        "policy": policy.as_dict(),
        "target_topology": target_topology.as_dict(),
        "synthesized_source_topology": source_topology.as_dict(),
        "short_conv": {
            "hidden_size": hidden,
            "channel_width": short_conv_parameters.channel_width,
            "fir_length": length,
            "bias": bool(short_conv_parameters.bias),
            **short_conv_parameters.tensor_fingerprints(),
        },
        "linearization": dict(linearization.diagnostics),
        "gate_collapse_detected": gate_collapsed,
        "quadratic_remainder_relative_energy": quadratic_mass,
        "streams": stream_receipts,
        "stream_count": stream_count,
        "self_gate_stream_included": include_self,
        "total_representable_value_rank": (
            target_topology.key_value_heads * head_dim
        ),
        "cached_branch_delta_replay_relative_rmse": cached_delta_agreement,
        "fit_rows": int(solve_fit.sum().item()),
        "supplied_fit_rows": int(fit.sum().item()),
        "unrepresentable_prefix_positions": min(length - 1, sequence),
        "fit_rows_excluded_as_unrepresentable_prefix": int(
            (fit & ~representable).sum().item()
        ),
        "validation_rows": int(validation.sum().item()),
        "validation_interior_rows": int(interior_validation.sum().item()),
        "validation_prefix_rows": int(prefix_validation.sum().item()),
        "selection_rows": (0 if selection is None else int(selection.sum().item())),
        "fit_assigned_lag_probability_mass": fit_mass,
        "validation_assigned_lag_probability_mass": validation_mass,
        "minimum_validation_assigned_lag_probability_mass": minimum_validation_mass,
        "fit_endpoint_relative_rmse": _relative_rmse(
            _masked_rows(predicted, solve_fit), fit_target
        ),
        "fit_endpoint_retained_energy_fraction": _retained_energy(
            _masked_rows(predicted, solve_fit), fit_target
        ),
        "validation_interior_endpoint_relative_rmse": (
            _relative_rmse(
                _masked_rows(predicted, interior_validation),
                _masked_rows(delta_target, interior_validation),
            )
            if bool(interior_validation.any().item())
            else None
        ),
        "validation_prefix_endpoint_relative_rmse": (
            _relative_rmse(
                _masked_rows(predicted, prefix_validation),
                _masked_rows(delta_target, prefix_validation),
            )
            if bool(prefix_validation.any().item())
            else None
        ),
        "validation_endpoint_relative_rmse": _relative_rmse(
            _masked_rows(predicted, validation), validation_target
        ),
        "validation_endpoint_retained_energy_fraction": _retained_energy(
            _masked_rows(predicted, validation), validation_target
        ),
        "validation_endpoint_relative_rmse_bf16": validation_relative_rmse_bf16,
        "fit_endpoint_relative_rmse_bf16": _relative_rmse(
            _masked_rows(predicted_bf16, solve_fit), fit_target
        ),
        "bf16_versus_fp64_endpoint_relative_delta": _relative_rmse(
            _masked_rows(predicted_bf16, validation),
            _masked_rows(predicted, validation),
        ),
        "baseline_zero_fit_relative_rmse": zero_baseline_fit,
        "baseline_zero_validation_relative_rmse": zero_baseline_validation,
        "baseline_no_branch_fit_relative_rmse": zero_baseline_fit,
        "baseline_no_branch_validation_relative_rmse": zero_baseline_validation,
        "baseline_frozen_gate_linear_fir_fit_relative_rmse": frozen_gate_fit,
        "baseline_frozen_gate_linear_fir_validation_relative_rmse": (
            frozen_gate_validation
        ),
        "remainder_fit_relative_energy": _energy_ratio(
            _masked_rows(remainder, solve_fit), fit_target
        ),
        "remainder_validation_relative_energy": _energy_ratio(
            _masked_rows(remainder, validation), validation_target
        ),
        "remainder_bf16_validation_relative_energy": _energy_ratio(
            _masked_rows(remainder_bf16, validation), validation_target
        ),
        "remainder_sha256": _tensor_sha256(remainder),
        "remainder_bf16_sha256": _tensor_sha256(remainder_bf16),
        "numeric_weight_matching_objective": False,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parent_forward_calls": 0,
    }
    return ShortConvToAttentionTranslation(
        fold=fold,
        linearization=linearization,
        remainder=remainder,
        report=_report(body),
    )


# ---------------------------------------------------------------------------
# Direction A: attention -> ShortConv
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AttentionBandDecomposition:
    """The exact lag split of one cached attention branch delta."""

    band_delta: Any
    far_delta: Any
    band_mass_rows: Any
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


def attention_band_decomposition(
    *,
    attention_probabilities: Any,
    attention_value_sequences: Any,
    attention_output_weight: Any,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
    fir_length: int,
    row_mask: Any,
) -> AttentionBandDecomposition:
    """Split ``Delta`` into band (lags ``0..L-1``) and far mass, row-exactly.

    ``Delta_band_t = sum_h sum_{l<L} P^h_{t,t-l} V^h_{t-l} W_o[h]^T`` and
    ``Delta_far_t = sum_h sum_{s<=t-L} P^h_{t,s} V^h_s W_o[h]^T``; both are
    computed from cached probabilities and values with no parent forward.
    """

    torch = _torch()
    probabilities = _tensor(attention_probabilities, "attention_probabilities", ndim=4)
    values = _tensor(attention_value_sequences, "attention_value_sequences", ndim=3)
    writer = _tensor(attention_output_weight, "attention_output_weight", ndim=2)
    query_heads = _positive_int(query_heads, "query_heads")
    key_value_heads = _positive_int(key_value_heads, "key_value_heads")
    head_dim = _positive_int(head_dim, "head_dim")
    fir_length = _positive_int(fir_length, "fir_length")
    if query_heads % key_value_heads:
        raise HFCausalKindTranslationError(
            "query_heads must be divisible by key_value_heads."
        )
    batch, sequence = int(values.shape[0]), int(values.shape[1])
    if tuple(probabilities.shape) != (batch, query_heads, sequence, sequence):
        raise HFCausalKindTranslationError(
            "attention_probabilities must have shape [batch, query_heads, S, S]."
        )
    if int(values.shape[2]) != key_value_heads * head_dim:
        raise HFCausalKindTranslationError(
            "attention_value_sequences width differs from key_value_heads*head_dim."
        )
    if int(writer.shape[1]) != query_heads * head_dim:
        raise HFCausalKindTranslationError(
            "attention_output_weight input width differs from query_heads*head_dim."
        )
    mask = _mask(row_mask, "row_mask", (batch, sequence))
    heads = (
        values.reshape(batch, sequence, key_value_heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
        .repeat_interleave(query_heads // key_value_heads, dim=1)
    )
    offsets = torch.arange(sequence).reshape(-1, 1) - torch.arange(sequence).reshape(1, -1)
    band_mask = ((offsets >= 0) & (offsets < fir_length)).reshape(1, 1, sequence, sequence)

    def contract(weights: Any) -> Any:
        contexts = torch.matmul(weights, heads)
        concatenated = contexts.permute(0, 2, 1, 3).reshape(
            batch, sequence, query_heads * head_dim
        )
        return concatenated @ writer.T

    band_probabilities = probabilities * band_mask
    far_probabilities = probabilities - band_probabilities
    band_delta = contract(band_probabilities)
    far_delta = contract(far_probabilities)
    total_delta = band_delta + far_delta
    band_mass_rows = band_probabilities.sum(dim=-1).mean(dim=1)
    selected_total = _masked_rows(total_delta, mask)
    selected_mass = band_mass_rows.reshape(-1)[mask.reshape(-1)]
    diagnostics = {
        "fir_length": fir_length,
        "query_heads": query_heads,
        "key_value_heads": key_value_heads,
        "head_dim": head_dim,
        "evidence_rows": int(mask.sum().item()),
        "rho_band": _energy_ratio(_masked_rows(band_delta, mask), selected_total),
        "rho_far": _energy_ratio(_masked_rows(far_delta, mask), selected_total),
        "band_mass_mean": float(selected_mass.mean().item()),
        "band_mass_minimum": float(selected_mass.min().item()),
        "band_mass_maximum": float(selected_mass.max().item()),
        # The design asks for the per-row distribution, not just its extremes:
        # a layer can average a high band mass while a tenth of its rows reach
        # outside the window entirely.
        "band_mass_deciles": [
            float(value.item())
            for value in torch.quantile(
                selected_mass,
                torch.arange(0, 11, dtype=torch.float64) / 10.0,
            )
        ],
        "band_delta_sha256": _tensor_sha256(band_delta),
        "far_delta_sha256": _tensor_sha256(far_delta),
    }
    return AttentionBandDecomposition(
        band_delta=band_delta,
        far_delta=far_delta,
        band_mass_rows=band_mass_rows,
        diagnostics=diagnostics,
    )


def attention_evidence_from_projections(
    *,
    query_projection_sequences: Any,
    key_projection_sequences: Any,
    value_projection_sequences: Any,
    topology: HFCausalGQATopology,
    attention_mask: Any,
    position_ids: Any,
    query_norm_weight: Any | None = None,
    key_norm_weight: Any | None = None,
) -> tuple[Any, Any]:
    """Replay cached probabilities from sealed Q/K/V projection outputs.

    Family-neutral evidence seals projections rather than probabilities
    (``hf_causal_circuit_evidence.py:215-233``); this reproduces the probability
    tensor deterministically from them, with the same semantics the attention
    branch kernel uses (``hf_causal_gqa_attention_fold.py:302-346``).
    """

    if not isinstance(topology, HFCausalGQATopology):
        raise HFCausalKindTranslationError("topology is invalid.")
    query = _tensor(query_projection_sequences, "query_projection_sequences", ndim=3)
    key = _tensor(key_projection_sequences, "key_projection_sequences", ndim=3)
    value = _tensor(value_projection_sequences, "value_projection_sequences", ndim=3)
    batch, sequence = int(query.shape[0]), int(query.shape[1])
    mask = _mask(attention_mask, "attention_mask", (batch, sequence))
    positions = position_ids.detach().to(device="cpu")

    def heads(value_rows: Any, count: int, name: str) -> Any:
        if int(value_rows.shape[2]) != count * topology.head_dim:
            raise HFCausalKindTranslationError(f"{name} has the wrong projection width.")
        return (
            value_rows.reshape(batch, sequence, count, topology.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    query_heads = heads(query, topology.query_heads, "query_projection_sequences")
    key_heads = heads(key, topology.key_value_heads, "key_projection_sequences")
    if topology.query_key_norm == "rms_norm":
        if query_norm_weight is None or key_norm_weight is None:
            raise HFCausalKindTranslationError(
                "RMS-normalized Q/K evidence requires both norm weights."
            )

        def normalize(rows: Any, gain: Any) -> Any:
            vector = _tensor(gain, "query_key_norm_weight", ndim=1)
            scale = (
                rows.square().mean(dim=-1, keepdim=True) + topology.rms_norm_epsilon
            ).rsqrt()
            return rows * scale * vector.reshape(1, 1, 1, -1)

        query_heads = normalize(query_heads, query_norm_weight)
        key_heads = normalize(key_heads, key_norm_weight)
    elif query_norm_weight is not None or key_norm_weight is not None:
        raise HFCausalKindTranslationError(
            "Identity Q/K semantics forbid norm weights."
        )
    _scores, probabilities = causal_attention_probabilities(
        apply_rope(query_heads, positions, theta=topology.rope_theta),
        apply_rope(key_heads, positions, theta=topology.rope_theta),
        attention_mask=mask,
        scaling=topology.attention_scaling,
    )
    return probabilities, value


@dataclass(frozen=True, slots=True)
class AttentionToShortConvPolicy:
    """Frozen policy constants for the attention-to-conv translation."""

    fir_length: int = 3
    channel_width: int | None = None
    bias: bool = True
    initialization_candidates: tuple[str, ...] = (
        "value_transport",
        "lag_blocked",
        "mean_lag_profile",
    )
    round_candidates: tuple[int, ...] = (1, 3)
    ridge_alpha_candidates: tuple[float, ...] = (1e-6, 1e-4)
    condition_caps: tuple[float, ...] = CONDITION_CAP_CANDIDATES
    maximum_unrouted_far_mass: float = 1e-3
    band_reconstruction_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        object.__setattr__(self, "fir_length", _positive_int(self.fir_length, "fir_length"))
        initializations = tuple(str(value) for value in self.initialization_candidates)
        if not initializations or any(
            value not in {"value_transport", "lag_blocked", "mean_lag_profile"}
            for value in initializations
        ):
            raise HFCausalKindTranslationError(
                "initialization_candidates must be a nonempty subset of "
                "('value_transport', 'lag_blocked', 'mean_lag_profile')."
            )
        object.__setattr__(self, "initialization_candidates", initializations)
        if self.channel_width is not None:
            object.__setattr__(
                self, "channel_width", _positive_int(self.channel_width, "channel_width")
            )
        if not isinstance(self.bias, bool):
            raise HFCausalKindTranslationError("bias must be boolean.")
        rounds = tuple(_positive_int(value, "round candidate") for value in self.round_candidates)
        alphas = tuple(
            _positive_float(value, "ridge alpha candidate")
            for value in self.ridge_alpha_candidates
        )
        caps = tuple(_positive_float(value, "condition cap") for value in self.condition_caps)
        if not rounds or not alphas or not caps:
            raise HFCausalKindTranslationError("Policy candidate ladders cannot be empty.")
        if any(cap <= 1.0 for cap in caps):
            raise HFCausalKindTranslationError("Condition caps must exceed one.")
        object.__setattr__(self, "round_candidates", rounds)
        object.__setattr__(self, "ridge_alpha_candidates", alphas)
        object.__setattr__(self, "condition_caps", caps)
        object.__setattr__(
            self,
            "maximum_unrouted_far_mass",
            _nonnegative_float(self.maximum_unrouted_far_mass, "maximum_unrouted_far_mass"),
        )
        object.__setattr__(
            self,
            "band_reconstruction_tolerance",
            _nonnegative_float(
                self.band_reconstruction_tolerance, "band_reconstruction_tolerance"
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "fir_length": self.fir_length,
            "channel_width": self.channel_width,
            "bias": self.bias,
            "initialization_candidates": list(self.initialization_candidates),
            "round_candidates": list(self.round_candidates),
            "ridge_alpha_candidates": list(self.ridge_alpha_candidates),
            "condition_caps": list(self.condition_caps),
            "maximum_unrouted_far_mass": self.maximum_unrouted_far_mass,
            "band_reconstruction_tolerance": self.band_reconstruction_tolerance,
        }


class _ShortConvSolveState:
    """Mutable factor bundle for the bounded variable-projection solve."""

    __slots__ = (
        "gate_b_weight",
        "gate_b_bias",
        "gate_c_weight",
        "gate_c_bias",
        "stream_x_weight",
        "stream_x_bias",
        "lag_kernel",
        "conv_bias",
        "output_weight",
        "output_bias",
    )

    def __init__(self, **values: Any) -> None:
        for name in self.__slots__:
            setattr(self, name, values[name])

    def snapshot(self, names: Sequence[str]) -> dict[str, Any]:
        return {name: getattr(self, name).clone() for name in names}

    def restore(self, values: Mapping[str, Any]) -> None:
        for name, value in values.items():
            setattr(self, name, value)

    def streams(self, rows: Any) -> dict[str, Any]:
        gate_b = rows @ self.gate_b_weight.T + self.gate_b_bias
        gate_c = rows @ self.gate_c_weight.T + self.gate_c_bias
        stream_x = rows @ self.stream_x_weight.T + self.stream_x_bias
        gated = gate_b * stream_x
        context = self.conv_bias.reshape(1, 1, -1).expand_as(gated).clone()
        for lag in range(int(self.lag_kernel.shape[1])):
            context = context + _lag_shift(gated, lag) * self.lag_kernel[:, lag]
        features = gate_c * context
        prediction = features @ self.output_weight.T + self.output_bias
        return {
            "b": gate_b,
            "c": gate_c,
            "x": stream_x,
            "r": gated,
            "v": context,
            "features": features,
            "prediction": prediction,
        }


def _signed_channel_directions(
    *,
    source: Any,
    target: Any,
    count: int,
) -> Any:
    """Deterministic PLS/balanced-circuit directions, never positive averaging.

    Signed left singular directions of the fit-row cross-covariance, unit-RMS
    normalized on the projected stream so the writer solve sees an equilibrated
    design.  Directions beyond ``min(count, hidden)`` cannot be independent —
    ``x`` is one linear image of ``n`` — so the surplus is filled with distinct
    adjacent-pair combinations rather than zeros, which would make those
    channels permanently inert.
    """

    torch = _torch()
    hidden = int(source.shape[1])
    centered_source = source - source.mean(dim=0)
    centered_target = target - target.mean(dim=0)
    left, _values, _right = _canonical_svd(centered_source.T @ centered_target)
    rank = min(count, int(left.shape[1]))
    directions = torch.zeros(count, hidden, dtype=torch.float64)
    directions[:rank] = left[:, :rank].T
    for channel in range(rank, count):
        first = (channel - rank) % rank
        second = (first + 1) % rank
        directions[channel] = left[:, first] + left[:, second]
    projected = centered_source @ directions.T
    scale = projected.square().mean(dim=0).sqrt().clamp_min(1e-12)
    return (directions / scale.reshape(-1, 1)).contiguous()


def _initial_solve_state(
    *,
    rows: Any,
    target: Any,
    fit: Any,
    channels: int,
    length: int,
    bias: bool,
    lag_profile: Any,
    alpha: float,
    condition_cap: float,
    initialization: str,
    transported_prior: Mapping[str, Any] | None = None,
) -> _ShortConvSolveState:
    torch = _torch()
    hidden = int(rows.shape[2])
    target_fit = _masked_rows(target, fit)
    kernel = torch.zeros(channels, length, dtype=torch.float64)
    directions = torch.zeros(channels, hidden, dtype=torch.float64)
    if initialization == "value_transport":
        # Failure-record item 6: transport the parent's own projections and use
        # them as priors instead of inventing a basis.  Each channel block is
        # one (head, lag) cell of the band, carries that head's recovered value
        # directions, and starts with that cell's mean in-band probability.
        if transported_prior is None:  # pragma: no cover - guarded by the caller
            raise HFCausalKindTranslationError(
                "The value-transport initialization needs a transported prior."
            )
        directions = transported_prior["directions"].clone()
        kernel = transported_prior["lag_kernel"].clone()
        scale = (
            (_masked_rows(rows, fit) @ directions.T)
            .square()
            .mean(dim=0)
            .sqrt()
            .clamp_min(1e-12)
        )
        directions = directions / scale.reshape(-1, 1)
    elif initialization == "mean_lag_profile":
        # The design's initialization: the mean band lag profile broadcast over
        # every channel, with one signed direction set for the smeared stream.
        smeared = torch.zeros_like(rows)
        for lag in range(length):
            smeared = smeared + float(lag_profile[lag].item()) * _lag_shift(rows, lag)
        directions = _signed_channel_directions(
            source=_masked_rows(smeared, fit), target=target_fit, count=channels
        )
        kernel = lag_profile.reshape(1, -1).expand(channels, length).contiguous().clone()
    elif initialization == "lag_blocked":
        # A uniform lag profile makes every channel see the same lag mixture, so
        # alternation has to break a symmetry before it can express the CP
        # structure the band actually has.  Assigning contiguous channel blocks
        # to single lags seeds that structure directly: block l carries only
        # k_l and is initialized against the lag-l shifted rows.  Block sizes
        # follow the measured band lag profile, so a lag that carries no mass
        # spends no channel budget.
        allocation = _proportional_partition(
            channels, [float(value.item()) for value in lag_profile]
        )
        for lag, group in enumerate(allocation):
            if not group:
                continue
            block = _signed_channel_directions(
                source=_masked_rows(_lag_shift(rows, lag), fit),
                target=target_fit,
                count=len(group),
            )
            for local, channel in enumerate(group):
                directions[channel] = block[local]
                kernel[channel, lag] = lag_profile[lag]
    else:  # pragma: no cover - guarded by the policy
        raise HFCausalKindTranslationError(
            f"Unknown short-conv initialization {initialization!r}."
        )
    ones = torch.ones(channels, dtype=torch.float64)
    if bias:
        gate_b_weight = torch.zeros(channels, hidden, dtype=torch.float64)
        gate_c_weight = torch.zeros(channels, hidden, dtype=torch.float64)
        gate_b_bias = ones.clone()
        gate_c_bias = ones.clone()
        stream_x_bias = torch.zeros(channels, dtype=torch.float64)
        conv_bias = torch.zeros(channels, dtype=torch.float64)
        output_bias = torch.zeros(hidden, dtype=torch.float64)
    else:
        # No bias vector exists, so the unit gate is realized by the closest
        # constant-one projection of the evidence rows: the design's "otherwise
        # initialized by the first alternation pass".
        constant, _bias, _diagnostics = _ridge_fit(
            _masked_rows(rows, fit),
            torch.ones(int(fit.sum().item()), channels, dtype=torch.float64),
            alpha=alpha,
            condition_cap=condition_cap,
            fit_intercept=False,
        )
        gate_b_weight = constant.T.contiguous()
        gate_c_weight = constant.T.contiguous()
        zeros = torch.zeros(channels, dtype=torch.float64)
        gate_b_bias = zeros
        gate_c_bias = zeros
        stream_x_bias = zeros
        conv_bias = zeros
        output_bias = torch.zeros(hidden, dtype=torch.float64)
    return _ShortConvSolveState(
        gate_b_weight=gate_b_weight,
        gate_b_bias=gate_b_bias,
        gate_c_weight=gate_c_weight,
        gate_c_bias=gate_c_bias,
        stream_x_weight=directions.contiguous(),
        stream_x_bias=stream_x_bias,
        lag_kernel=kernel.contiguous(),
        conv_bias=conv_bias,
        output_weight=torch.zeros(hidden, channels, dtype=torch.float64),
        output_bias=output_bias,
    )


def _solve_short_conv_to_band(
    *,
    rows: Any,
    target: Any,
    fit: Any,
    channels: int,
    length: int,
    bias: bool,
    lag_profile: Any,
    alpha: float,
    condition_cap: float,
    rounds: int,
    initialization: str,
    transported_prior: Mapping[str, Any] | None = None,
) -> tuple[_ShortConvSolveState, dict[str, Any]]:
    """Bounded deterministic variable projection against the composed endpoint."""

    torch = _torch()
    state = _initial_solve_state(
        rows=rows,
        target=target,
        fit=fit,
        channels=channels,
        length=length,
        bias=bias,
        lag_profile=lag_profile,
        alpha=alpha,
        condition_cap=condition_cap,
        initialization=initialization,
        transported_prior=transported_prior,
    )
    target_fit = _masked_rows(target, fit)
    row_count = int(fit.sum().item())

    def objective() -> float:
        prediction = _masked_rows(state.streams(rows)["prediction"], fit)
        return float((prediction - target_fit).square().sum().item())

    trace: list[dict[str, Any]] = [{"step": "initial", "fit_objective": objective()}]
    condition_receipts: list[dict[str, Any]] = []

    def guarded(step: str, names: Sequence[str], update: Any) -> None:
        before = trace[-1]["fit_objective"]
        saved = state.snapshot(names)
        diagnostics: dict[str, Any] | None = update()
        after = objective()
        reverted = after > before * (1.0 + 1e-12) + 1e-300
        if reverted:
            state.restore(saved)
            after = before
        trace.append(
            {
                "step": step,
                "fit_objective": after,
                "reverted_for_monotonicity": bool(reverted),
            }
        )
        if diagnostics is not None:
            condition_receipts.append({"step": step, **diagnostics})

    def update_writer() -> dict[str, Any]:
        streams = state.streams(rows)
        weight, intercept, diagnostics = _ridge_fit(
            _masked_rows(streams["features"], fit),
            target_fit,
            alpha=alpha,
            condition_cap=condition_cap,
            fit_intercept=bias,
        )
        state.output_weight = weight.T.contiguous()
        if bias:
            assert intercept is not None
            state.output_bias = intercept.contiguous()
        return diagnostics

    def update_lag_kernel() -> dict[str, Any]:
        streams = state.streams(rows)
        residual = _masked_rows(target - streams["prediction"], fit)
        gate_c = _masked_rows(streams["c"], fit)
        lagged = [
            _masked_rows(_lag_shift(streams["r"], lag), fit) for lag in range(length)
        ]
        worst = None
        for channel in range(channels):
            direction = state.output_weight[:, channel]
            columns = [gate_c[:, channel] * lagged[lag][:, channel] for lag in range(length)]
            if bias:
                columns.append(gate_c[:, channel])
            design = torch.stack(columns, dim=1)
            previous = torch.cat(
                (
                    state.lag_kernel[channel],
                    state.conv_bias[channel].reshape(1) if bias else torch.zeros(0, dtype=torch.float64),
                )
            )
            local_residual = residual + (design @ previous).reshape(-1, 1) * direction.reshape(1, -1)
            solution, diagnostics = _directional_ridge(
                design,
                local_residual,
                direction,
                alpha=alpha,
                condition_cap=condition_cap,
            )
            if worst is None or (
                diagnostics["regularized_condition"] > worst["regularized_condition"]
            ):
                worst = diagnostics
            if solution is None:
                continue
            residual = local_residual - (design @ solution).reshape(-1, 1) * direction.reshape(
                1, -1
            )
            state.lag_kernel[channel] = solution[:length]
            if bias:
                state.conv_bias[channel] = solution[length]
        return worst

    def update_gate(kind: str) -> dict[str, Any]:
        streams = state.streams(rows)
        residual = _masked_rows(target - streams["prediction"], fit)
        gate_c = _masked_rows(streams["c"], fit)
        rows_fit = _masked_rows(rows, fit)
        worst = None
        for channel in range(channels):
            direction = state.output_weight[:, channel]
            if kind == "gate_c":
                context = _masked_rows(streams["v"], fit)[:, channel]
                columns = [rows_fit * context.reshape(-1, 1)]
                if bias:
                    columns.append(context.reshape(-1, 1))
                design = torch.cat(columns, dim=1)
                previous = torch.cat(
                    (
                        state.gate_c_weight[channel],
                        state.gate_c_bias[channel].reshape(1)
                        if bias
                        else torch.zeros(0, dtype=torch.float64),
                    )
                )
            else:
                partner = "x" if kind == "gate_b" else "b"
                weighted_rows = None
                weighted_scalar = None
                for lag in range(length):
                    coefficient = (
                        gate_c[:, channel]
                        * float(state.lag_kernel[channel, lag].item())
                        * _masked_rows(_lag_shift(streams[partner], lag), fit)[:, channel]
                    )
                    lagged_rows = _masked_rows(_lag_shift(rows, lag), fit)
                    term = coefficient.reshape(-1, 1) * lagged_rows
                    weighted_rows = term if weighted_rows is None else weighted_rows + term
                    weighted_scalar = (
                        coefficient
                        if weighted_scalar is None
                        else weighted_scalar + coefficient
                    )
                assert weighted_rows is not None and weighted_scalar is not None
                columns = [weighted_rows]
                if bias:
                    columns.append(weighted_scalar.reshape(-1, 1))
                design = torch.cat(columns, dim=1)
                if kind == "gate_b":
                    previous = torch.cat(
                        (
                            state.gate_b_weight[channel],
                            state.gate_b_bias[channel].reshape(1)
                            if bias
                            else torch.zeros(0, dtype=torch.float64),
                        )
                    )
                else:
                    previous = torch.cat(
                        (
                            state.stream_x_weight[channel],
                            state.stream_x_bias[channel].reshape(1)
                            if bias
                            else torch.zeros(0, dtype=torch.float64),
                        )
                    )
            local_residual = residual + (design @ previous).reshape(-1, 1) * direction.reshape(
                1, -1
            )
            solution, diagnostics = _directional_ridge(
                design,
                local_residual,
                direction,
                alpha=alpha,
                condition_cap=condition_cap,
            )
            if worst is None or (
                diagnostics["regularized_condition"] > worst["regularized_condition"]
            ):
                worst = diagnostics
            if solution is None:
                continue
            residual = local_residual - (design @ solution).reshape(-1, 1) * direction.reshape(
                1, -1
            )
            hidden = int(rows.shape[2])
            if kind == "gate_c":
                state.gate_c_weight[channel] = solution[:hidden]
                if bias:
                    state.gate_c_bias[channel] = solution[hidden]
            elif kind == "gate_b":
                state.gate_b_weight[channel] = solution[:hidden]
                if bias:
                    state.gate_b_bias[channel] = solution[hidden]
            else:
                state.stream_x_weight[channel] = solution[:hidden]
                if bias:
                    state.stream_x_bias[channel] = solution[hidden]
        return worst

    for _round in range(rounds):
        guarded("output_weight", ("output_weight", "output_bias"), update_writer)
        guarded("lag_kernel", ("lag_kernel", "conv_bias"), update_lag_kernel)
        guarded("gate_b", ("gate_b_weight", "gate_b_bias"), lambda: update_gate("gate_b"))
        guarded("gate_c", ("gate_c_weight", "gate_c_bias"), lambda: update_gate("gate_c"))
        guarded(
            "stream_x", ("stream_x_weight", "stream_x_bias"), lambda: update_gate("stream_x")
        )

    objectives = [entry["fit_objective"] for entry in trace]
    diagnostics = {
        "rounds": rounds,
        "initialization": initialization,
        "ridge_alpha": alpha,
        "condition_cap": condition_cap,
        "fit_rows": row_count,
        "objective_trace": trace,
        "fit_objective_monotone_nonincreasing": all(
            later <= earlier * (1.0 + 1e-12) + 1e-300
            for earlier, later in zip(objectives, objectives[1:])
        ),
        "reverted_substeps": sum(
            1 for entry in trace if entry.get("reverted_for_monotonicity")
        ),
        "worst_regularized_condition": (
            max(receipt["regularized_condition"] for receipt in condition_receipts)
            if condition_receipts
            else 0.0
        ),
        "condition_receipts": condition_receipts,
    }
    return state, diagnostics


def _transported_value_prior(
    *,
    rows: Any,
    values: Any,
    probabilities: Any,
    fit: Any,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
    channels: int,
    length: int,
    alpha: float,
    condition_cap: float,
) -> dict[str, Any]:
    """Recover the parent value directions and lay them out as channel blocks.

    Sealed evidence carries value *rows*, not the projection, so the linear map
    is recovered by one intercept-free ridge from the same operator-norm rows
    the conv will read; its relative RMSE is reported, because a layer whose
    cached values are not a linear image of the sealed rows has no transportable
    prior and the recovery quietly degrades into an arbitrary basis.
    """

    torch = _torch()
    hidden = int(rows.shape[2])
    sequence = int(rows.shape[1])
    recovered, _bias, _diagnostics = _ridge_fit(
        _masked_rows(rows, fit),
        _masked_rows(values, fit),
        alpha=alpha,
        condition_cap=condition_cap,
        fit_intercept=False,
    )
    value_weight = recovered.T.contiguous()
    recovery_error = _relative_rmse(
        _masked_rows(rows @ value_weight.T, fit), _masked_rows(values, fit)
    )
    queries_per_kv = query_heads // key_value_heads
    cells: list[tuple[int, int, float]] = []
    for head in range(query_heads):
        for lag in range(length):
            collected: list[Any] = []
            for position in range(lag, sequence):
                selected = fit[:, position]
                if not bool(selected.any().item()):
                    continue
                collected.append(probabilities[:, head, position, position - lag][selected])
            mass = (
                float(torch.cat(collected).mean().item()) if collected else 0.0
            )
            cells.append((head, lag, mass))
    allocation = _proportional_partition(channels, [cell[2] for cell in cells])
    directions = torch.zeros(channels, hidden, dtype=torch.float64)
    kernel = torch.zeros(channels, length, dtype=torch.float64)
    receipts: list[dict[str, Any]] = []
    for (head, lag, mass), group in zip(cells, allocation):
        if not group:
            continue
        kv_head = head // queries_per_kv
        block = value_weight[kv_head * head_dim : (kv_head + 1) * head_dim, :]
        for local, channel in enumerate(group):
            if local < head_dim:
                directions[channel] = block[local]
            else:
                first = (local - head_dim) % head_dim
                directions[channel] = block[first] + block[(first + 1) % head_dim]
            kernel[channel, lag] = mass
        receipts.append(
            {
                "query_head": head,
                "key_value_head": kv_head,
                "lag": lag,
                "mean_in_band_probability": mass,
                "channels": list(group),
            }
        )
    return {
        "directions": directions,
        "lag_kernel": kernel,
        "diagnostics": {
            "value_projection_recovery_relative_rmse": recovery_error,
            "channel_cells": receipts,
        },
    }


@dataclass(frozen=True, slots=True)
class AttentionToShortConvTranslation:
    """One compiled attention-to-conv branch with its sealed receipt."""

    parameters: ShortConvBranchParameters
    decomposition: AttentionBandDecomposition
    remainder: Any
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))


def compile_attention_to_short_conv_translation(
    normalized_sequences: Any,
    *,
    attention_probabilities: Any,
    attention_value_sequences: Any,
    attention_output_weight: Any,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
    target_residual_delta_sequences: Any,
    valid_token_mask: Any,
    fit_token_mask: Any,
    selection_token_mask: Any,
    validation_token_mask: Any,
    source_layer_index: int = 0,
    carrier_layer_index: int | None = None,
    evidence_cache_key: str | None = None,
    policy: AttentionToShortConvPolicy | None = None,
) -> AttentionToShortConvTranslation:
    """Translate one cached attention branch into a ShortConv branch."""

    torch = _torch()
    policy = policy or AttentionToShortConvPolicy()
    rows = _tensor(normalized_sequences, "normalized_sequences", ndim=3)
    delta = _tensor(
        target_residual_delta_sequences, "target_residual_delta_sequences", ndim=3
    )
    batch, sequence, hidden = (int(rows.shape[0]), int(rows.shape[1]), int(rows.shape[2]))
    if tuple(delta.shape) != (batch, sequence, hidden):
        raise HFCausalKindTranslationError(
            "The attention branch delta shape differs from the operator rows."
        )
    valid = _mask(valid_token_mask, "valid_token_mask", (batch, sequence))
    fit = _mask(fit_token_mask, "fit_token_mask", (batch, sequence))
    selection = _mask(selection_token_mask, "selection_token_mask", (batch, sequence))
    validation = _mask(validation_token_mask, "validation_token_mask", (batch, sequence))
    _disjoint_partitions(
        valid=valid, fit=fit, validation=validation, selection=selection
    )
    length = policy.fir_length
    channels = policy.channel_width or hidden
    decomposition = attention_band_decomposition(
        attention_probabilities=attention_probabilities,
        attention_value_sequences=attention_value_sequences,
        attention_output_weight=attention_output_weight,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        fir_length=length,
        row_mask=fit,
    )
    reconstruction = decomposition.band_delta + decomposition.far_delta
    reconstruction_error = _relative_rmse(
        _masked_rows(reconstruction, fit), _masked_rows(delta, fit)
    )
    if reconstruction_error > policy.band_reconstruction_tolerance:
        raise HFCausalKindTranslationError(
            "The cached probabilities and values do not reconstruct the sealed "
            f"attention delta (relative RMSE {reconstruction_error:.6e})."
        )
    rho_far = float(decomposition.diagnostics["rho_far"])
    if carrier_layer_index is None and rho_far > policy.maximum_unrouted_far_mass:
        raise HFCausalKindTranslationError(
            f"Far attention mass rho_far={rho_far:.6e} exceeds the unrouted bound "
            f"{policy.maximum_unrouted_far_mass} and no downstream carrier layer "
            "can represent global mixing."
        )

    probabilities = _tensor(attention_probabilities, "attention_probabilities", ndim=4)
    lag_profile = torch.zeros(length, dtype=torch.float64)
    for lag in range(length):
        collected: list[Any] = []
        for position in range(lag, sequence):
            selected = fit[:, position]
            if not bool(selected.any().item()):
                continue
            collected.append(probabilities[:, :, position, position - lag][selected])
        lag_profile[lag] = (
            torch.cat([value.reshape(-1) for value in collected]).mean()
            if collected
            else torch.zeros((), dtype=torch.float64)
        )

    band_target = decomposition.band_delta
    selection_target = _masked_rows(band_target, selection)
    # The best *unconstrained* lag-separable linear operator on the same rows.
    # Anything the conv cannot reach beyond this bound is the rank-one-per-
    # channel factorization loss (A.2); anything the bound itself cannot reach
    # is the softmax normalization coupling (A.3 item 3), which no depthwise FIR
    # of any width or gating can represent.
    lag_design = torch.cat(
        [_lag_shift(rows, lag) for lag in range(length)], dim=-1
    )
    floor_weight, floor_bias, _floor_diagnostics = _ridge_fit(
        _masked_rows(lag_design, fit),
        _masked_rows(band_target, fit),
        alpha=min(policy.ridge_alpha_candidates),
        condition_cap=max(policy.condition_caps),
        fit_intercept=policy.bias,
    )
    floor_prediction = lag_design @ floor_weight
    if floor_bias is not None:
        floor_prediction = floor_prediction + floor_bias
    transported_prior: dict[str, Any] | None = None
    if "value_transport" in policy.initialization_candidates:
        transported_prior = _transported_value_prior(
            rows=rows,
            values=_tensor(attention_value_sequences, "attention_value_sequences", ndim=3),
            probabilities=probabilities,
            fit=fit,
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            head_dim=head_dim,
            channels=channels,
            length=length,
            alpha=min(policy.ridge_alpha_candidates),
            condition_cap=max(policy.condition_caps),
        )
    candidates: list[dict[str, Any]] = []
    best: tuple[float, int] | None = None
    best_state: _ShortConvSolveState | None = None
    best_diagnostics: dict[str, Any] | None = None
    for initialization in policy.initialization_candidates:
        for rounds in policy.round_candidates:
            for alpha in policy.ridge_alpha_candidates:
                for cap in policy.condition_caps:
                    state, diagnostics = _solve_short_conv_to_band(
                        rows=rows,
                        target=band_target,
                        fit=fit,
                        channels=channels,
                        length=length,
                        bias=policy.bias,
                        lag_profile=lag_profile,
                        alpha=alpha,
                        condition_cap=cap,
                        rounds=rounds,
                        initialization=initialization,
                        transported_prior=transported_prior,
                    )
                    score = _relative_rmse(
                        _masked_rows(state.streams(rows)["prediction"], selection),
                        selection_target,
                    )
                    index = len(candidates)
                    candidates.append(
                        {
                            "candidate_index": index,
                            "initialization": initialization,
                            "rounds": rounds,
                            "ridge_alpha": alpha,
                            "condition_cap": cap,
                            "selection_band_relative_rmse": score,
                            "fit_objective_monotone_nonincreasing": diagnostics[
                                "fit_objective_monotone_nonincreasing"
                            ],
                            "reverted_substeps": diagnostics["reverted_substeps"],
                            "worst_regularized_condition": diagnostics[
                                "worst_regularized_condition"
                            ],
                        }
                    )
                    if best is None or score < best[0]:
                        best = (score, index)
                        best_state = state
                        best_diagnostics = diagnostics
    assert best is not None and best_state is not None and best_diagnostics is not None

    parameters = short_conv_parameters_from_solve(
        gate_b_weight=best_state.gate_b_weight,
        gate_c_weight=best_state.gate_c_weight,
        stream_x_weight=best_state.stream_x_weight,
        lag_kernel=best_state.lag_kernel,
        output_weight=best_state.output_weight,
        gate_b_bias=best_state.gate_b_bias if policy.bias else None,
        gate_c_bias=best_state.gate_c_bias if policy.bias else None,
        stream_x_bias=best_state.stream_x_bias if policy.bias else None,
        conv_bias=best_state.conv_bias if policy.bias else None,
        output_bias=best_state.output_bias if policy.bias else None,
    )
    predicted = parameters.forward(rows)
    predicted_bf16 = parameters.forward(rows, dtype=torch.bfloat16).to(
        dtype=torch.float64
    )
    # The conv and the attention truncate the same way at the start of a
    # sequence, but the softmax renormalizes there while a FIR does not, so the
    # in-band weights the conv must match are position-dependent exactly on the
    # first L-1 rows.  That is design A.3 item 3 (softmax normalization
    # coupling); it is reported split out rather than blended away.
    interior = torch.zeros(batch, sequence, dtype=torch.bool)
    interior[:, min(length - 1, sequence) :] = True
    interior_validation = validation & interior
    prefix_validation = validation & ~interior
    band_fit = _masked_rows(band_target, fit)
    band_validation = _masked_rows(band_target, validation)
    delta_fit = _masked_rows(delta, fit)
    delta_validation = _masked_rows(delta, validation)
    zero_validation = _relative_rmse(torch.zeros_like(band_validation), band_validation)
    validation_band_rmse_bf16 = _relative_rmse(
        _masked_rows(predicted_bf16, validation), band_validation
    )
    if validation_band_rmse_bf16 >= zero_validation:
        raise HFCausalKindTranslationError(
            "The BF16 validation band endpoint is at or above the no-branch "
            f"baseline ({validation_band_rmse_bf16:.6f} >= {zero_validation:.6f})."
        )
    remainder = delta - predicted
    remainder_bf16 = delta - predicted_bf16
    factorization_residual_fit = _energy_ratio(
        _masked_rows(band_target - predicted, fit), band_fit
    )

    body = {
        "schema": ATTENTION_TO_SHORT_CONV_SCHEMA,
        "component": "attention_to_short_conv_branch_translation",
        "direction": "attention_to_short_conv",
        "method": "exact_lag_split_then_bounded_variable_projection_rank_one_band_fit",
        "source_layer_index": int(source_layer_index),
        "carrier_layer_index": (
            None if carrier_layer_index is None else int(carrier_layer_index)
        ),
        "evidence_cache_key": evidence_cache_key,
        "policy": policy.as_dict(),
        "hidden_size": hidden,
        "channel_width": channels,
        "fir_length": length,
        "bias": bool(policy.bias),
        "decomposition": dict(decomposition.diagnostics),
        "band_decomposition_reconstruction_relative_rmse": reconstruction_error,
        "initial_lag_profile": [float(value.item()) for value in lag_profile],
        "transported_value_prior": (
            None if transported_prior is None else transported_prior["diagnostics"]
        ),
        "fit_rows": int(fit.sum().item()),
        "selection_rows": int(selection.sum().item()),
        "validation_rows": int(validation.sum().item()),
        "solve_candidates": candidates,
        "selected_candidate_index": best[1],
        "selected_selection_band_relative_rmse": best[0],
        "solve": {
            key: value
            for key, value in best_diagnostics.items()
            if key != "condition_receipts"
        },
        "worst_regularized_condition": best_diagnostics["worst_regularized_condition"],
        "fit_band_relative_rmse": _relative_rmse(_masked_rows(predicted, fit), band_fit),
        "fit_band_retained_energy_fraction": _retained_energy(
            _masked_rows(predicted, fit), band_fit
        ),
        "validation_band_relative_rmse": _relative_rmse(
            _masked_rows(predicted, validation), band_validation
        ),
        "validation_band_retained_energy_fraction": _retained_energy(
            _masked_rows(predicted, validation), band_validation
        ),
        "validation_band_relative_rmse_bf16": validation_band_rmse_bf16,
        "validation_interior_band_relative_rmse": (
            _relative_rmse(
                _masked_rows(predicted, interior_validation),
                _masked_rows(band_target, interior_validation),
            )
            if bool(interior_validation.any().item())
            else None
        ),
        "validation_prefix_band_relative_rmse": (
            _relative_rmse(
                _masked_rows(predicted, prefix_validation),
                _masked_rows(band_target, prefix_validation),
            )
            if bool(prefix_validation.any().item())
            else None
        ),
        "validation_interior_rows": int(interior_validation.sum().item()),
        "validation_prefix_rows": int(prefix_validation.sum().item()),
        "bf16_versus_fp64_endpoint_relative_delta": _relative_rmse(
            _masked_rows(predicted_bf16, validation), _masked_rows(predicted, validation)
        ),
        "fit_full_delta_relative_rmse": _relative_rmse(
            _masked_rows(predicted, fit), delta_fit
        ),
        "validation_full_delta_relative_rmse": _relative_rmse(
            _masked_rows(predicted, validation), delta_validation
        ),
        "band_factorization_residual_relative_energy": factorization_residual_fit,
        "baseline_zero_validation_relative_rmse": zero_validation,
        "baseline_no_branch_validation_relative_rmse": zero_validation,
        "baseline_band_only_validation_relative_rmse": _relative_rmse(
            band_validation, delta_validation
        ),
        "band_lag_separable_linear_floor_fit_relative_rmse": _relative_rmse(
            _masked_rows(floor_prediction, fit), band_fit
        ),
        "band_lag_separable_linear_floor_validation_relative_rmse": _relative_rmse(
            _masked_rows(floor_prediction, validation), band_validation
        ),
        "remainder_fit_relative_energy": _energy_ratio(
            _masked_rows(remainder, fit), delta_fit
        ),
        "remainder_validation_relative_energy": _energy_ratio(
            _masked_rows(remainder, validation), delta_validation
        ),
        "remainder_bf16_validation_relative_energy": _energy_ratio(
            _masked_rows(remainder_bf16, validation), delta_validation
        ),
        "remainder_sha256": _tensor_sha256(remainder),
        "remainder_bf16_sha256": _tensor_sha256(remainder_bf16),
        "short_conv": parameters.tensor_fingerprints(),
        "numeric_weight_matching_objective": False,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parent_forward_calls": 0,
    }
    return AttentionToShortConvTranslation(
        parameters=parameters,
        decomposition=decomposition,
        remainder=remainder,
        report=_report(body),
    )


# ---------------------------------------------------------------------------
# Remainder routing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KindTranslationRemainder:
    """One sealed routed-remainder bundle."""

    source_layer_index: int
    carrier_layer_index: int
    checkpoint_boundary_id: str
    remainder: Any
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))


def seal_kind_translation_remainder(
    *,
    direction: str,
    source_layer_index: int,
    carrier_layer_index: int | None,
    remainder: Any,
    source_report_sha256: str,
    evidence_cache_key: str | None = None,
    carrier_action_id: str | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    routing_checkpoint_metrics: Mapping[str, Any] | None = None,
) -> KindTranslationRemainder:
    """Seal the remainder a converted layer hands to its downstream carrier."""

    if direction not in {"attention_to_short_conv", "short_conv_to_attention"}:
        raise HFCausalKindTranslationError("direction is not a kind translation.")
    if carrier_layer_index is None:
        raise HFCausalKindTranslationError(
            "Routing a kind-translation remainder requires a carrier layer; a "
            "child with no downstream operator that can represent global mixing "
            "fails closed instead of dropping the remainder."
        )
    source_layer_index = int(source_layer_index)
    carrier_layer_index = int(carrier_layer_index)
    if carrier_layer_index <= source_layer_index:
        raise HFCausalKindTranslationError(
            "The carrier layer must be downstream of the converted layer."
        )
    tensor = _tensor(remainder, "remainder", ndim=3)
    boundary = canonical_sha256(
        {
            "schema": KIND_TRANSLATION_REMAINDER_SCHEMA,
            "direction": direction,
            "source_layer_index": source_layer_index,
            "carrier_layer_index": carrier_layer_index,
        }
    )
    body = {
        "schema": KIND_TRANSLATION_REMAINDER_SCHEMA,
        "direction": direction,
        "source_layer_index": source_layer_index,
        "carrier_layer_index": carrier_layer_index,
        "checkpoint_boundary_id": boundary,
        "source_report_sha256": str(source_report_sha256),
        "evidence_cache_key": evidence_cache_key,
        "carrier_action_id": carrier_action_id,
        "remainder_sha256": _tensor_sha256(tensor),
        "remainder_shape": [int(size) for size in tensor.shape],
        "diagnostics": dict(diagnostics or {}),
        "routing_checkpoint_metrics": (
            None if routing_checkpoint_metrics is None else dict(routing_checkpoint_metrics)
        ),
    }
    return KindTranslationRemainder(
        source_layer_index=source_layer_index,
        carrier_layer_index=carrier_layer_index,
        checkpoint_boundary_id=boundary,
        remainder=tensor,
        report=_report(body),
    )


def carrier_endpoint_target_with_remainder(
    carrier_delta: Any,
    remainders: Sequence[KindTranslationRemainder],
) -> Any:
    """``Y_j = Delta_j + sum_{i: carrier(i)=j} R_i`` (design A.3)."""

    target = _tensor(carrier_delta, "carrier_delta", ndim=3).clone()
    for bundle in remainders:
        if not isinstance(bundle, KindTranslationRemainder):
            raise HFCausalKindTranslationError(
                "Every routed remainder must be a sealed KindTranslationRemainder."
            )
        if tuple(bundle.remainder.shape) != tuple(target.shape):
            raise HFCausalKindTranslationError(
                "A routed remainder does not match the carrier endpoint shape."
            )
        target = target + bundle.remainder
    return target.contiguous()


def evaluate_carrier_routing_checkpoint(
    *,
    checkpoint_target: Any,
    routing_on_prediction: Any,
    routing_off_prediction: Any,
    validation_token_mask: Any,
    fail_closed: bool = True,
) -> Mapping[str, Any]:
    """Score the carrier checkpoint with routing on versus off (rule A.3).

    The checkpoint quantity is the cumulative residual the downstream stack
    consumes, so both candidate carriers are compared against the *same*
    parent-coordinate checkpoint target.  Routing that makes the checkpoint
    worse fails the conversion closed; the remainder is never silently dropped.
    """

    target = _tensor(checkpoint_target, "checkpoint_target", ndim=3)
    routing_on = _tensor(routing_on_prediction, "routing_on_prediction", ndim=3)
    routing_off = _tensor(routing_off_prediction, "routing_off_prediction", ndim=3)
    if tuple(routing_on.shape) != tuple(target.shape) or tuple(
        routing_off.shape
    ) != tuple(target.shape):
        raise HFCausalKindTranslationError(
            "Checkpoint predictions must match the checkpoint target shape."
        )
    mask = _mask(
        validation_token_mask,
        "validation_token_mask",
        (int(target.shape[0]), int(target.shape[1])),
    )
    if not bool(mask.any().item()):
        raise HFCausalKindTranslationError("The checkpoint validation mask is empty.")
    selected_target = _masked_rows(target, mask)
    on_error = _relative_rmse(_masked_rows(routing_on, mask), selected_target)
    off_error = _relative_rmse(_masked_rows(routing_off, mask), selected_target)
    improves = on_error <= off_error
    metrics = _report(
        {
            "schema": KIND_TRANSLATION_REMAINDER_SCHEMA,
            "component": "carrier_checkpoint_routing_comparison",
            "checkpoint_validation_rows": int(mask.sum().item()),
            "routing_on_checkpoint_relative_rmse": on_error,
            "routing_off_checkpoint_relative_rmse": off_error,
            "routing_improves_checkpoint": bool(improves),
        }
    )
    if fail_closed and not improves:
        raise HFCausalKindTranslationError(
            "Routing the kind-translation remainder made the carrier checkpoint "
            f"worse ({on_error:.6f} > {off_error:.6f}); the conversion fails closed."
        )
    return metrics


__all__ = [
    "ATTENTION_TO_SHORT_CONV_SCHEMA",
    "CONDITION_CAP_CANDIDATES",
    "KIND_TRANSLATION_REMAINDER_SCHEMA",
    "SHORT_CONV_TO_ATTENTION_SCHEMA",
    "AttentionBandDecomposition",
    "AttentionToShortConvPolicy",
    "AttentionToShortConvTranslation",
    "HFCausalKindTranslationError",
    "KindTranslationRemainder",
    "ShortConvBranchParameters",
    "ShortConvLinearization",
    "ShortConvToAttentionPolicy",
    "ShortConvToAttentionTranslation",
    "attention_band_decomposition",
    "attention_evidence_from_projections",
    "carrier_endpoint_target_with_remainder",
    "compile_attention_to_short_conv_translation",
    "compile_short_conv_to_attention_translation",
    "evaluate_carrier_routing_checkpoint",
    "linearize_short_conv_branch",
    "seal_kind_translation_remainder",
    "short_conv_parameters_from_solve",
]
