"""Zero-training LFM2 ``w2`` factorization with one shared residual basis.

For selected decoder layers this module replaces each dense residual-write
matrix ``W_l`` with

``W_l ~= B A_l``.

``B`` is one physical learned Parameter in the common residual coordinate
system.  Every layer retains a distinct ``A_l`` in its own post-SwiGLU input
coordinate system.  No layer operators are averaged or required to match
numerically.

The basis is fitted only from the sealed fit rows
``Y_l = Z_l W_l.T``.  A disjoint validation split chooses, independently per
layer, between an exact weight projection and a closed-form ridge solve.  Rank
comes directly from exact unique-Parameter accounting.  Installation is a
BaseException-safe transaction and never retains a dense ``w2`` fallback.
"""

from __future__ import annotations

import math
import threading
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
import torch.nn.functional as F

from neural_forge import hf_lfm2_activation_factorization as _activation
from neural_forge.hf_lfm2_activation_factorization import (
    Lfm2ActivationFactorizationEvidence,
    Lfm2ActivationFactorizationTopologyError,
    Lfm2ActivationFactorizationTransactionError,
)


PLAN_SCHEMA = "neural-forge-lfm2-shared-residual-basis-plan-v1"
REPORT_SCHEMA = "neural-forge-lfm2-shared-residual-basis-report-v1"
METHOD = "shared-residual-output-basis-distinct-layer-coefficients"
BASIS_ATTRIBUTE = "_neural_forge_lfm2_shared_residual_basis"
_FACTOR_DOMAIN = b"lfm2-shared-residual-factor-v1"
_SOURCE_DOMAIN = b"lfm2-source-dense-weight-v1"
_TRANSACTION_LOCK = threading.RLock()


class Lfm2SharedResidualBasisError(RuntimeError):
    """Base error for shared residual-output-basis folding."""


class Lfm2SharedResidualBasisTopologyError(
    Lfm2SharedResidualBasisError,
    Lfm2ActivationFactorizationTopologyError,
):
    """The selected LFM2 topology cannot support the requested fold."""


class Lfm2SharedResidualBasisTransactionError(
    Lfm2SharedResidualBasisError,
    Lfm2ActivationFactorizationTransactionError,
):
    """Sealing, accounting, installation, or rollback failed closed."""


def _json_sha256(value: Any) -> str:
    return _activation._json_sha256(value)


def _tensor_sha256(value: torch.Tensor, *, domain: bytes = _FACTOR_DOMAIN) -> str:
    return _activation._tensor_sha256(value, domain=domain)


def _model_identity(model: Any) -> str:
    return _activation._model_identity(model)


def _parameter_counts(model: Any) -> tuple[int, int]:
    return _activation._parameter_counts(model)


def _relative_squared_error(
    target: torch.Tensor,
    observed: torch.Tensor,
) -> float:
    return _activation._relative_squared_error(target, observed)


def _flattened_cosine(target: torch.Tensor, observed: torch.Tensor) -> float:
    numerator = float((target * observed).sum().item())
    denominator = math.sqrt(
        max(float(target.square().sum().item()), 0.0)
        * max(float(observed.square().sum().item()), 0.0)
    )
    if denominator <= torch.finfo(torch.float64).tiny:
        return 1.0 if bool(torch.equal(target, observed)) else 0.0
    value = numerator / denominator
    if not math.isfinite(value):
        raise Lfm2SharedResidualBasisTransactionError(
            "Reconstruction cosine is invalid."
        )
    return max(-1.0, min(1.0, value))


def _compute_dtype(value: torch.Tensor) -> torch.dtype:
    return torch.float64 if value.dtype == torch.float64 else torch.float32


def _canonicalize_basis(value: torch.Tensor) -> torch.Tensor:
    """Fix eigenvector sign without changing its represented subspace."""

    if value.ndim != 2 or value.shape[1] == 0:
        return value
    maxima = value.abs().argmax(dim=0)
    columns = torch.arange(int(value.shape[1]), device=value.device)
    pivots = value[maxima, columns]
    signs = torch.where(
        pivots < 0,
        -torch.ones_like(pivots),
        torch.ones_like(pivots),
    )
    return (value * signs.unsqueeze(0)).contiguous()


def _finite_matrix(value: Any, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.numel() == 0
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise Lfm2SharedResidualBasisTopologyError(
            f"{name} must be a nonempty finite rank-2 Tensor."
        )
    return value.detach().to(device="cpu").contiguous()


def _normalized_collections(
    weights: Sequence[Any],
    fit_post_swiglu: Sequence[Any],
    validation_post_swiglu: Sequence[Any],
) -> tuple[
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
]:
    if (
        isinstance(weights, (str, bytes))
        or isinstance(fit_post_swiglu, (str, bytes))
        or isinstance(validation_post_swiglu, (str, bytes))
        or not isinstance(weights, Sequence)
        or not isinstance(fit_post_swiglu, Sequence)
        or not isinstance(validation_post_swiglu, Sequence)
    ):
        raise Lfm2SharedResidualBasisTopologyError(
            "Weights and activation rows must be Tensor sequences."
        )
    dense = tuple(
        _finite_matrix(value, f"weights[{index}]")
        for index, value in enumerate(weights)
    )
    fit = tuple(
        _finite_matrix(value, f"fit_post_swiglu[{index}]")
        for index, value in enumerate(fit_post_swiglu)
    )
    validation = tuple(
        _finite_matrix(value, f"validation_post_swiglu[{index}]")
        for index, value in enumerate(validation_post_swiglu)
    )
    if not dense or len(dense) != len(fit) or len(dense) != len(validation):
        raise Lfm2SharedResidualBasisTopologyError(
            "Every selected weight needs fit and validation activation rows."
        )
    output_features = int(dense[0].shape[0])
    if any(
        int(value.shape[0]) != output_features
        or value.dtype != dense[0].dtype
        for value in dense
    ):
        raise Lfm2SharedResidualBasisTopologyError(
            "Selected w2 matrices do not share one residual width and dtype."
        )
    for index, (weight, fit_rows, validation_rows) in enumerate(
        zip(dense, fit, validation)
    ):
        if (
            int(fit_rows.shape[1]) != int(weight.shape[1])
            or int(validation_rows.shape[1]) != int(weight.shape[1])
        ):
            raise Lfm2SharedResidualBasisTopologyError(
                f"Layer {index} post-SwiGLU rows do not match w2 input width."
            )
    return dense, fit, validation


def _ridge_input_factor(
    fit: torch.Tensor,
    target_coefficients: torch.Tensor,
    *,
    ridge: float,
) -> torch.Tensor:
    """Solve ridge in the smaller primal or dual system, without gradients."""

    rows, features = (int(fit.shape[0]), int(fit.shape[1]))
    if rows <= features:
        gram = fit @ fit.transpose(0, 1)
        scale = max(
            float(gram.diagonal().mean().item()),
            torch.finfo(gram.dtype).eps,
        )
        gram = gram + torch.eye(
            rows,
            dtype=gram.dtype,
            device=gram.device,
        ) * (ridge * scale)
        solution = fit.transpose(0, 1) @ torch.linalg.solve(
            gram,
            target_coefficients,
        )
    else:
        gram = fit.transpose(0, 1) @ fit
        scale = max(
            float(gram.diagonal().mean().item()),
            torch.finfo(gram.dtype).eps,
        )
        gram = gram + torch.eye(
            features,
            dtype=gram.dtype,
            device=gram.device,
        ) * (ridge * scale)
        solution = torch.linalg.solve(
            gram,
            fit.transpose(0, 1) @ target_coefficients,
        )
    return solution.transpose(0, 1).contiguous()


@dataclass(frozen=True, slots=True)
class Lfm2SharedResidualLayerFactor:
    """One private layer coefficient factor against the shared basis."""

    layer_index: int
    matrix_shape: tuple[int, int]
    method: Literal["exact_weight_projection", "closed_form_ridge"]
    input_factor: Any
    input_factor_sha256: str
    source_weight_sha256: str
    requires_grad: bool
    fit_relative_rmse: float
    fit_cosine: float
    validation_relative_rmse: float
    validation_cosine: float
    projection_validation_relative_rmse: float

    @property
    def rank(self) -> int:
        return int(self.input_factor.shape[0])

    @property
    def dense_parameters(self) -> int:
        return math.prod(self.matrix_shape)

    @property
    def private_factor_parameters(self) -> int:
        return int(self.input_factor.numel())

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "role": "w2",
            "matrix_shape": list(self.matrix_shape),
            "rank": self.rank,
            "method": self.method,
            "source_weight_sha256": self.source_weight_sha256,
            "requires_grad": self.requires_grad,
            "dense_parameters": self.dense_parameters,
            "private_factor_parameters": self.private_factor_parameters,
            "fit_relative_rmse": self.fit_relative_rmse,
            "fit_cosine": self.fit_cosine,
            "validation_relative_rmse": self.validation_relative_rmse,
            "validation_cosine": self.validation_cosine,
            "projection_validation_relative_rmse": (
                self.projection_validation_relative_rmse
            ),
            "input_factor": {
                "shape": [int(size) for size in self.input_factor.shape],
                "dtype": str(self.input_factor.dtype).replace("torch.", ""),
                "sha256": self.input_factor_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class Lfm2SharedResidualFactorization:
    """A fitted global output basis plus private per-layer input factors."""

    residual_features: int
    rank: int
    output_basis: Any
    output_basis_sha256: str
    layers: tuple[Lfm2SharedResidualLayerFactor, ...]
    fit_row_count: int
    validation_row_count: int
    fit_stack_sha256: str
    basis_weighting: Literal["layer_balanced", "raw_stack"]
    fit_layer_output_scales: tuple[float, ...]
    fit_relative_rmse: float
    fit_worst_layer_relative_rmse: float
    validation_relative_rmse: float
    validation_worst_layer_relative_rmse: float
    validation_mean_layer_cosine: float
    validation_worst_layer_cosine: float
    fit_explained_energy_fraction: float
    ridge: float

    @property
    def shared_basis_parameters(self) -> int:
        return int(self.output_basis.numel())

    @property
    def private_factor_parameters(self) -> int:
        return sum(value.private_factor_parameters for value in self.layers)

    @property
    def factor_parameters(self) -> int:
        return self.shared_basis_parameters + self.private_factor_parameters

    @property
    def dense_parameters(self) -> int:
        return sum(value.dense_parameters for value in self.layers)

    @property
    def parameters_saved(self) -> int:
        return self.dense_parameters - self.factor_parameters

    def as_dict(self) -> dict[str, Any]:
        return {
            "residual_features": self.residual_features,
            "rank": self.rank,
            "fit_row_count": self.fit_row_count,
            "validation_row_count": self.validation_row_count,
            "fit_stack_sha256": self.fit_stack_sha256,
            "basis_weighting": self.basis_weighting,
            "fit_layer_output_scales": list(self.fit_layer_output_scales),
            "fit_relative_rmse": self.fit_relative_rmse,
            "fit_worst_layer_relative_rmse": (
                self.fit_worst_layer_relative_rmse
            ),
            "validation_relative_rmse": self.validation_relative_rmse,
            "validation_worst_layer_relative_rmse": (
                self.validation_worst_layer_relative_rmse
            ),
            "validation_mean_layer_cosine": (
                self.validation_mean_layer_cosine
            ),
            "validation_worst_layer_cosine": (
                self.validation_worst_layer_cosine
            ),
            "fit_explained_energy_fraction": (
                self.fit_explained_energy_fraction
            ),
            "ridge": self.ridge,
            "dense_parameters": self.dense_parameters,
            "shared_basis_parameters": self.shared_basis_parameters,
            "private_factor_parameters": self.private_factor_parameters,
            "factor_parameters": self.factor_parameters,
            "parameters_saved": self.parameters_saved,
            "output_basis": {
                "shape": [int(size) for size in self.output_basis.shape],
                "dtype": str(self.output_basis.dtype).replace("torch.", ""),
                "sha256": self.output_basis_sha256,
            },
            "layers": [value.as_dict() for value in self.layers],
        }


def factorize_shared_residual_output_basis(
    weights: Sequence[Any],
    fit_post_swiglu: Sequence[Any],
    validation_post_swiglu: Sequence[Any],
    *,
    rank: int,
    layer_indices: Sequence[int] | None = None,
    ridge: float = 1e-6,
    basis_weighting: Literal["layer_balanced", "raw_stack"] = (
        "layer_balanced"
    ),
) -> Lfm2SharedResidualFactorization:
    """Fit one residual output basis and distinct private layer factors."""

    with torch.no_grad():
        source_requires_grad = tuple(
            bool(value.requires_grad)
            if isinstance(value, torch.Tensor)
            else False
            for value in weights
        )
        dense, fit_rows, validation_rows = _normalized_collections(
            weights,
            fit_post_swiglu,
            validation_post_swiglu,
        )
        residual_features = int(dense[0].shape[0])
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank <= 0
            or rank > residual_features
        ):
            raise Lfm2SharedResidualBasisTopologyError(
                "rank must be within the shared residual output dimension."
            )
        if (
            isinstance(ridge, bool)
            or not isinstance(ridge, (int, float))
            or not math.isfinite(float(ridge))
            or float(ridge) <= 0.0
        ):
            raise Lfm2SharedResidualBasisTopologyError(
                "ridge must be a finite positive scalar."
            )
        if basis_weighting not in ("layer_balanced", "raw_stack"):
            raise Lfm2SharedResidualBasisTopologyError(
                "basis_weighting must be layer_balanced or raw_stack."
            )
        if layer_indices is None:
            indices = tuple(range(len(dense)))
        else:
            if (
                isinstance(layer_indices, (str, bytes))
                or not isinstance(layer_indices, Sequence)
            ):
                raise Lfm2SharedResidualBasisTopologyError(
                    "layer_indices must be a sequence of unique integers."
                )
            indices = tuple(layer_indices)
        if (
            len(indices) != len(dense)
            or len(set(indices)) != len(indices)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in indices
            )
        ):
            raise Lfm2SharedResidualBasisTopologyError(
                "layer_indices must identify every factor exactly once."
            )
        compute_dtype = _compute_dtype(dense[0])
        compute_weights = tuple(value.to(dtype=compute_dtype) for value in dense)
        compute_fit = tuple(value.to(dtype=compute_dtype) for value in fit_rows)
        compute_validation = tuple(
            value.to(dtype=compute_dtype) for value in validation_rows
        )
        fit_outputs = tuple(
            activations @ weight.transpose(0, 1)
            for activations, weight in zip(compute_fit, compute_weights)
        )
        if basis_weighting == "layer_balanced":
            scales = tuple(
                max(
                    float(value.norm().item()),
                    torch.finfo(value.dtype).eps,
                )
                for value in fit_outputs
            )
        else:
            scales = tuple(1.0 for _value in fit_outputs)
        stacked_fit_outputs = torch.cat(
            tuple(
                value / scale
                for value, scale in zip(fit_outputs, scales)
            ),
            dim=0,
        ).contiguous()
        # The Gram eigensystem is exactly the right-singular output basis of
        # the requested stacked Y, while avoiding a very large full U matrix.
        gram = (
            stacked_fit_outputs.transpose(0, 1) @ stacked_fit_outputs
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues.index_select(0, order)
        basis_compute = _canonicalize_basis(
            eigenvectors.index_select(1, order)[:, :rank]
        )
        output_dtype = dense[0].dtype
        output_basis = basis_compute.to(dtype=output_dtype).contiguous()
        if not bool(torch.isfinite(output_basis).all()):
            raise Lfm2SharedResidualBasisTransactionError(
                "Basis dtype conversion produced non-finite values."
            )
        # Evaluate the actual installed dtype, not an optimistic solve dtype.
        installed_basis = output_basis.to(dtype=compute_dtype)
        basis_gram = installed_basis.transpose(0, 1) @ installed_basis
        basis_left_inverse = torch.linalg.solve(
            basis_gram,
            installed_basis.transpose(0, 1),
        )
        layer_factors: list[Lfm2SharedResidualLayerFactor] = []
        fit_numerator = 0.0
        fit_denominator = 0.0
        validation_numerator = 0.0
        validation_denominator = 0.0
        fit_layer_errors: list[float] = []
        validation_layer_errors: list[float] = []
        validation_layer_cosines: list[float] = []
        for (
            layer_index,
            original_weight,
            weight,
            fit,
            validation,
            source_requires,
        ) in zip(
            indices,
            dense,
            compute_weights,
            compute_fit,
            compute_validation,
            source_requires_grad,
        ):
            target_fit = fit @ weight.transpose(0, 1)
            target_validation = validation @ weight.transpose(0, 1)
            projection_compute = basis_left_inverse @ weight
            projection = projection_compute.to(dtype=output_dtype).contiguous()
            projection_compute = projection.to(dtype=compute_dtype)

            ridge_factor_compute = _ridge_input_factor(
                fit,
                target_fit @ basis_left_inverse.transpose(0, 1),
                ridge=float(ridge),
            )
            ridge_factor = ridge_factor_compute.to(
                dtype=output_dtype
            ).contiguous()
            ridge_factor_compute = ridge_factor.to(dtype=compute_dtype)

            def prediction(
                rows: torch.Tensor,
                factor: torch.Tensor,
            ) -> torch.Tensor:
                return (rows @ factor.transpose(0, 1)) @ (
                    installed_basis.transpose(0, 1)
                )

            projection_validation_error = math.sqrt(
                _relative_squared_error(
                    target_validation,
                    prediction(validation, projection_compute),
                )
            )
            ridge_validation_error = math.sqrt(
                _relative_squared_error(
                    target_validation,
                    prediction(validation, ridge_factor_compute),
                )
            )
            tolerance = 16.0 * torch.finfo(torch.float64).eps
            if ridge_validation_error <= projection_validation_error + tolerance:
                method: Literal[
                    "exact_weight_projection",
                    "closed_form_ridge",
                ] = "closed_form_ridge"
                selected = ridge_factor
                selected_compute = ridge_factor_compute
                validation_error = ridge_validation_error
            else:
                method = "exact_weight_projection"
                selected = projection
                selected_compute = projection_compute
                validation_error = projection_validation_error
            observed_fit = prediction(fit, selected_compute)
            observed_validation = prediction(validation, selected_compute)
            fit_error = math.sqrt(
                _relative_squared_error(target_fit, observed_fit)
            )
            fit_cosine = _flattened_cosine(target_fit, observed_fit)
            validation_cosine = _flattened_cosine(
                target_validation,
                observed_validation,
            )
            fit_layer_errors.append(fit_error)
            validation_layer_errors.append(validation_error)
            validation_layer_cosines.append(validation_cosine)
            fit_numerator += float(
                (target_fit - observed_fit).square().sum().item()
            )
            fit_denominator += float(target_fit.square().sum().item())
            validation_numerator += float(
                (target_validation - observed_validation).square().sum().item()
            )
            validation_denominator += float(
                target_validation.square().sum().item()
            )
            layer_factors.append(
                Lfm2SharedResidualLayerFactor(
                    layer_index=layer_index,
                    matrix_shape=(
                        int(original_weight.shape[0]),
                        int(original_weight.shape[1]),
                    ),
                    method=method,
                    input_factor=selected,
                    input_factor_sha256=_tensor_sha256(selected),
                    source_weight_sha256=_tensor_sha256(
                        original_weight,
                        domain=_SOURCE_DOMAIN,
                    ),
                    requires_grad=source_requires,
                    fit_relative_rmse=fit_error,
                    fit_cosine=fit_cosine,
                    validation_relative_rmse=validation_error,
                    validation_cosine=validation_cosine,
                    projection_validation_relative_rmse=(
                        projection_validation_error
                    ),
                )
            )
        tiny = torch.finfo(torch.float64).tiny
        total_energy = max(float(eigenvalues.clamp_min(0).sum().item()), tiny)
        explained = float(eigenvalues[:rank].clamp_min(0).sum().item())
        result = Lfm2SharedResidualFactorization(
            residual_features=residual_features,
            rank=rank,
            output_basis=output_basis,
            output_basis_sha256=_tensor_sha256(output_basis),
            layers=tuple(layer_factors),
            fit_row_count=sum(int(value.shape[0]) for value in fit_rows),
            validation_row_count=sum(
                int(value.shape[0]) for value in validation_rows
            ),
            fit_stack_sha256=_tensor_sha256(
                stacked_fit_outputs,
                domain=b"lfm2-shared-residual-fit-output-stack-v1",
            ),
            basis_weighting=basis_weighting,
            fit_layer_output_scales=scales,
            fit_relative_rmse=math.sqrt(
                fit_numerator / max(fit_denominator, tiny)
            ),
            fit_worst_layer_relative_rmse=max(fit_layer_errors),
            validation_relative_rmse=math.sqrt(
                validation_numerator / max(validation_denominator, tiny)
            ),
            validation_worst_layer_relative_rmse=max(
                validation_layer_errors
            ),
            validation_mean_layer_cosine=(
                sum(validation_layer_cosines)
                / len(validation_layer_cosines)
            ),
            validation_worst_layer_cosine=min(
                validation_layer_cosines
            ),
            fit_explained_energy_fraction=explained / total_energy,
            ridge=float(ridge),
        )
        return result


@dataclass(frozen=True, slots=True)
class Lfm2SharedResidualBasisPlan:
    """Sealed exact-budget plan for one shared basis and private factors."""

    schema: str
    parent_identity_sha256: str
    evidence_sha256: str
    target_unique_parameter_budget: int
    logical_parameters_before: int
    unique_parameters_before: int
    projected_logical_parameters_after: int
    projected_unique_parameters_after: int
    projected_parameters_saved: int
    selected_layer_indices: tuple[int, ...]
    factorization: Lfm2SharedResidualFactorization
    independent_equal_rank_factor_parameters: int
    independent_uniform_rank_at_same_factor_budget: int
    integer_metadata_values: int
    integer_tensor_count: int
    integer_tensor_bytes: int
    plan_sha256: str
    training_steps: int = 0
    gradients_used: bool = False

    @property
    def selected_roles(self) -> tuple[tuple[int, str], ...]:
        return tuple((value, "w2") for value in self.selected_layer_indices)

    @property
    def role_plans(self) -> tuple[Lfm2SharedResidualLayerFactor, ...]:
        return self.factorization.layers

    @property
    def rank(self) -> int:
        return self.factorization.rank

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
            "selected_layer_indices": list(self.selected_layer_indices),
            "factorization": self.factorization.as_dict(),
            "independent_equal_rank_factor_parameters": (
                self.independent_equal_rank_factor_parameters
            ),
            "independent_uniform_rank_at_same_factor_budget": (
                self.independent_uniform_rank_at_same_factor_budget
            ),
            "shared_rank_advantage_over_independent": (
                self.rank
                - self.independent_uniform_rank_at_same_factor_budget
            ),
            "integer_metadata_values": self.integer_metadata_values,
            "integer_tensor_count": self.integer_tensor_count,
            "integer_tensor_bytes": self.integer_tensor_bytes,
            "training_steps": self.training_steps,
            "gradients_used": self.gradients_used,
            "plan_sha256": self.plan_sha256,
        }


def _plan_body(plan: Lfm2SharedResidualBasisPlan) -> dict[str, Any]:
    return {
        key: value
        for key, value in plan.as_dict().items()
        if key != "plan_sha256"
    }


def _seal_plan(
    plan: Lfm2SharedResidualBasisPlan,
) -> Lfm2SharedResidualBasisPlan:
    return replace(plan, plan_sha256=_json_sha256(_plan_body(plan)))


def _validate_plan_seal(plan: Lfm2SharedResidualBasisPlan) -> None:
    factorization = plan.factorization
    if (
        _tensor_sha256(factorization.output_basis)
        != factorization.output_basis_sha256
        or any(
            _tensor_sha256(value.input_factor) != value.input_factor_sha256
            for value in factorization.layers
        )
    ):
        raise Lfm2SharedResidualBasisTransactionError(
            "A sealed shared residual factor payload changed after planning."
        )
    if _json_sha256(_plan_body(plan)) != plan.plan_sha256:
        raise Lfm2SharedResidualBasisTransactionError(
            "The sealed shared-residual-basis plan changed after planning."
        )


def _selected_w2_roles(
    evidence: Lfm2ActivationFactorizationEvidence,
    dense_roles: Sequence[Any],
    layer_indices: Sequence[int] | None,
    roles: Sequence[str],
) -> tuple[Any, ...]:
    if tuple(roles) != ("w2",):
        raise Lfm2SharedResidualBasisTopologyError(
            "A shared residual-output basis is defined only for w2."
        )
    selected = _activation._normalize_selection(
        evidence,
        tuple(dense_roles),
        layer_indices,
        ("w2",),
    )
    if any(value.role != "w2" for value in selected):
        raise Lfm2SharedResidualBasisTopologyError(
            "Only residual-write w2 modules can share the output basis."
        )
    return selected


def plan_lfm2_shared_residual_basis(
    model: Any,
    evidence: Lfm2ActivationFactorizationEvidence,
    *,
    target_unique_parameter_budget: int,
    layer_indices: Sequence[int] | None = None,
    roles: Sequence[str] = ("w2",),
    ridge: float = 1e-6,
    basis_weighting: Literal["layer_balanced", "raw_stack"] = (
        "layer_balanced"
    ),
) -> Lfm2SharedResidualBasisPlan:
    """Choose the largest exact shared rank under the unique-Parameter target."""

    with _TRANSACTION_LOCK, torch.no_grad():
        _layers, dense_roles = _activation._validate_evidence(model, evidence)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise Lfm2SharedResidualBasisTransactionError(
                "Existing gradients must be cleared before zero-training planning."
            )
        logical_before, unique_before = _parameter_counts(model)
        if (
            isinstance(target_unique_parameter_budget, bool)
            or not isinstance(target_unique_parameter_budget, int)
            or target_unique_parameter_budget <= 0
            or target_unique_parameter_budget >= unique_before
        ):
            raise Lfm2SharedResidualBasisTopologyError(
                "target_unique_parameter_budget must be positive and smaller "
                "than the exact source-child unique-Parameter count."
            )
        selected = _selected_w2_roles(
            evidence,
            dense_roles,
            layer_indices,
            roles,
        )
        _activation._validate_selected_parameter_independence(model, selected)
        residual_features = int(selected[0].output_features)
        reference_weight = selected[0].module.weight
        if any(
            value.output_features != residual_features
            or value.module.weight.dtype != reference_weight.dtype
            or value.module.weight.device != reference_weight.device
            for value in selected
        ):
            raise Lfm2SharedResidualBasisTopologyError(
                "Selected w2 modules do not share one residual width, dtype, "
                "and device."
            )
        selected_dense = sum(value.dense_parameters for value in selected)
        constant_parameters = unique_before - selected_dense
        per_rank = residual_features + sum(
            int(value.input_features) for value in selected
        )
        available = target_unique_parameter_budget - constant_parameters
        rank = min(residual_features, available // per_rank)
        maximum_saving_rank = (selected_dense - 1) // per_rank
        rank = min(rank, maximum_saving_rank)
        if rank <= 0:
            raise Lfm2SharedResidualBasisTopologyError(
                "The target is below the rank-1 shared-basis physical minimum."
            )
        factorization = factorize_shared_residual_output_basis(
            tuple(value.module.weight for value in selected),
            tuple(evidence.layer(value.layer_index).fit_post_swiglu for value in selected),
            tuple(
                evidence.layer(value.layer_index).validation_post_swiglu
                for value in selected
            ),
            rank=rank,
            layer_indices=tuple(value.layer_index for value in selected),
            ridge=ridge,
            basis_weighting=basis_weighting,
        )
        projected = constant_parameters + factorization.factor_parameters
        if (
            projected > target_unique_parameter_budget
            or projected >= unique_before
            or factorization.dense_parameters != selected_dense
            or factorization.factor_parameters != rank * per_rank
        ):
            raise Lfm2SharedResidualBasisTransactionError(
                "Shared-rank allocation failed exact physical accounting."
            )
        independent_cost_per_rank = sum(
            value.output_features + value.input_features for value in selected
        )
        independent_rank = min(
            residual_features,
            available // independent_cost_per_rank,
        )
        plan = Lfm2SharedResidualBasisPlan(
            schema=PLAN_SCHEMA,
            parent_identity_sha256=evidence.parent_identity_sha256,
            evidence_sha256=evidence.evidence_sha256,
            target_unique_parameter_budget=target_unique_parameter_budget,
            logical_parameters_before=logical_before,
            unique_parameters_before=unique_before,
            projected_logical_parameters_after=(
                logical_before - (unique_before - projected)
            ),
            projected_unique_parameters_after=projected,
            projected_parameters_saved=unique_before - projected,
            selected_layer_indices=tuple(
                value.layer_index for value in selected
            ),
            factorization=factorization,
            independent_equal_rank_factor_parameters=(
                rank * independent_cost_per_rank
            ),
            independent_uniform_rank_at_same_factor_budget=independent_rank,
            # One rank and two feature sizes on each runtime plus a shared
            # residual/rank pair on the basis owner.  No integer Tensors exist.
            integer_metadata_values=(
                3 * len(selected) + 2
            ),
            integer_tensor_count=0,
            integer_tensor_bytes=0,
            plan_sha256="",
        )
        return _seal_plan(plan)


class Lfm2SharedResidualBasis(torch.nn.Module):
    """The sole physical owner of the global residual output basis."""

    def __init__(
        self,
        output_basis: torch.Tensor,
        *,
        requires_grad: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if (
            not isinstance(output_basis, torch.Tensor)
            or output_basis.ndim != 2
            or output_basis.numel() == 0
        ):
            raise Lfm2SharedResidualBasisTopologyError(
                "output_basis must be a nonempty rank-2 Tensor."
            )
        self.residual_features = int(output_basis.shape[0])
        self.rank = int(output_basis.shape[1])
        self.output_basis = torch.nn.Parameter(
            output_basis.detach().to(device=device, dtype=dtype).clone(),
            requires_grad=requires_grad,
        )

    def extra_repr(self) -> str:
        return (
            f"residual_features={self.residual_features}, "
            f"rank={self.rank}"
        )


class Lfm2SharedResidualFactorizedLinear(torch.nn.Module):
    """Private ``A_l`` runtime referencing the one non-duplicated basis owner."""

    def __init__(
        self,
        input_factor: torch.Tensor,
        basis_owner: Lfm2SharedResidualBasis,
        *,
        requires_grad: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if (
            not isinstance(input_factor, torch.Tensor)
            or input_factor.ndim != 2
            or not isinstance(basis_owner, Lfm2SharedResidualBasis)
            or int(input_factor.shape[0]) != basis_owner.rank
        ):
            raise Lfm2SharedResidualBasisTopologyError(
                "The private factor and shared basis dimensions do not compose."
            )
        self.in_features = int(input_factor.shape[1])
        self.out_features = basis_owner.residual_features
        self.rank = int(input_factor.shape[0])
        self.input_factor = torch.nn.Parameter(
            input_factor.detach().to(device=device, dtype=dtype).clone(),
            requires_grad=requires_grad,
        )
        self.register_parameter("bias", None)
        # Avoid registering the same basis Module or Parameter below every
        # layer.  The root model owns it once; this weak reference is runtime
        # routing metadata, not a learned scalar or dense-weight fallback.
        object.__setattr__(self, "_basis_owner_ref", weakref.ref(basis_owner))

    @property
    def basis_owner(self) -> Lfm2SharedResidualBasis:
        owner = self._basis_owner_ref()
        if owner is None:
            raise Lfm2SharedResidualBasisTransactionError(
                "The shared residual basis owner is no longer live."
            )
        return owner

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(
            F.linear(inputs, self.input_factor),
            self.basis_owner.output_basis,
        )

    def effective_weight(self) -> torch.Tensor:
        return self.basis_owner.output_basis @ self.input_factor

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias=False, shared_output_basis=True"
        )


@dataclass(frozen=True, slots=True)
class Lfm2SharedResidualBasisReport:
    """Authenticated installation and physical sharing/accounting evidence."""

    schema: str
    parent_identity_before: str
    model_identity_after: str
    evidence_sha256: str
    selected_roles: tuple[tuple[int, str], ...]
    rank: int
    logical_parameters_before: int
    logical_parameters_after: int
    unique_parameters_before: int
    unique_parameters_after: int
    parameters_saved: int
    dense_parameters_removed: int
    shared_basis_parameters_added: int
    private_factor_parameters_added: int
    shared_basis_parameter_occurrences: int
    shared_basis_unique_parameter_objects: int
    private_factor_unique_parameter_objects: int
    integer_metadata_values: int
    integer_tensor_count: int
    integer_tensor_bytes: int
    unselected_parameters_preserved_exactly: bool
    attention_and_recurrent_operators_preserved: bool
    buffers_preserved: bool
    dense_w2_weights_retained: bool
    dense_w2_weights_registered_in_runtime: bool
    rollback_escrow_retains_original_modules: bool
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
            "rank": self.rank,
            "logical_parameters_before": self.logical_parameters_before,
            "logical_parameters_after": self.logical_parameters_after,
            "unique_parameters_before": self.unique_parameters_before,
            "unique_parameters_after": self.unique_parameters_after,
            "parameters_saved": self.parameters_saved,
            "dense_parameters_removed": self.dense_parameters_removed,
            "shared_basis_parameters_added": (
                self.shared_basis_parameters_added
            ),
            "private_factor_parameters_added": (
                self.private_factor_parameters_added
            ),
            "shared_basis_parameter_occurrences": (
                self.shared_basis_parameter_occurrences
            ),
            "shared_basis_unique_parameter_objects": (
                self.shared_basis_unique_parameter_objects
            ),
            "private_factor_unique_parameter_objects": (
                self.private_factor_unique_parameter_objects
            ),
            "integer_metadata_values": self.integer_metadata_values,
            "integer_tensor_count": self.integer_tensor_count,
            "integer_tensor_bytes": self.integer_tensor_bytes,
            "unselected_parameters_preserved_exactly": (
                self.unselected_parameters_preserved_exactly
            ),
            "attention_and_recurrent_operators_preserved": (
                self.attention_and_recurrent_operators_preserved
            ),
            "buffers_preserved": self.buffers_preserved,
            "dense_w2_weights_retained": self.dense_w2_weights_retained,
            "dense_w2_weights_registered_in_runtime": (
                self.dense_w2_weights_registered_in_runtime
            ),
            "rollback_escrow_retains_original_modules": (
                self.rollback_escrow_retains_original_modules
            ),
            "training_steps": self.training_steps,
            "gradients_used": self.gradients_used,
        }


@dataclass(frozen=True, slots=True)
class _ProjectionBinding:
    owner: Any
    layer_index: int
    original: torch.nn.Linear
    installed: Lfm2SharedResidualFactorizedLinear


def _assign_projection(owner: Any, module: Any) -> None:
    setattr(owner, "w2", module)


def _attach_basis(model: Any, basis: Lfm2SharedResidualBasis) -> None:
    setattr(model, BASIS_ATTRIBUTE, basis)


def _detach_basis(model: Any) -> None:
    delattr(model, BASIS_ATTRIBUTE)


class InstalledLfm2SharedResidualBasis:
    """Live transaction handle supporting commit or authenticated rollback."""

    def __init__(
        self,
        *,
        model: Any,
        report: Lfm2SharedResidualBasisReport,
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
                _assign_projection(value.owner, value.original)
            if hasattr(self.model, BASIS_ATTRIBUTE):
                _detach_basis(self.model)
            _activation._verify_buffers(self.model, self._buffers)
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
                raise Lfm2SharedResidualBasisTransactionError(
                    "Exact shared-residual-basis rollback failed."
                )
            self.active = False
            return True

    def commit(self) -> Lfm2SharedResidualBasisReport:
        with _TRANSACTION_LOCK:
            if not self.active:
                raise Lfm2SharedResidualBasisTransactionError(
                    "The shared-basis transaction is no longer active."
                )
            self.active = False
            self.committed = True
            self._bindings = ()
            self._parent_parameters = {}
            self._buffers = {}
            return self.report

    def __enter__(self) -> InstalledLfm2SharedResidualBasis:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()


def install_lfm2_shared_residual_basis_plan(
    model: Any,
    evidence: Lfm2ActivationFactorizationEvidence,
    plan: Lfm2SharedResidualBasisPlan,
) -> InstalledLfm2SharedResidualBasis:
    """Validate every seal, then atomically install one shared-basis plan."""

    with _TRANSACTION_LOCK, torch.no_grad():
        layers, dense_roles = _activation._validate_evidence(model, evidence)
        if (
            not isinstance(plan, Lfm2SharedResidualBasisPlan)
            or plan.schema != PLAN_SCHEMA
        ):
            raise Lfm2SharedResidualBasisTransactionError(
                "A sealed shared-residual-basis plan is required."
            )
        _validate_plan_seal(plan)
        if (
            plan.parent_identity_sha256 != evidence.parent_identity_sha256
            or plan.evidence_sha256 != evidence.evidence_sha256
        ):
            raise Lfm2SharedResidualBasisTransactionError(
                "The plan does not match the exact model/evidence pair."
            )
        if hasattr(model, BASIS_ATTRIBUTE):
            raise Lfm2SharedResidualBasisTransactionError(
                "The model already owns a shared residual basis."
            )
        logical_before, unique_before = _parameter_counts(model)
        if (
            logical_before != plan.logical_parameters_before
            or unique_before != plan.unique_parameters_before
            or _model_identity(model) != plan.parent_identity_sha256
        ):
            raise Lfm2SharedResidualBasisTransactionError(
                "The live source child changed after planning."
            )
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise Lfm2SharedResidualBasisTransactionError(
                "Existing gradients must be cleared before zero-training install."
            )
        lookup = {
            (value.layer_index, value.role): value for value in dense_roles
        }
        selected = tuple(
            lookup.get((layer_index, "w2"))
            for layer_index in plan.selected_layer_indices
        )
        if any(value is None for value in selected):
            raise Lfm2SharedResidualBasisTransactionError(
                "A planned w2 role is absent from the live source child."
            )
        selected = tuple(value for value in selected if value is not None)
        _activation._validate_selected_parameter_independence(model, selected)
        selected_original_ids = {
            id(value.module.weight) for value in selected
        }
        parent_parameters = dict(
            model.named_parameters(remove_duplicate=False)
        )
        unselected_snapshot = {
            name: (
                parameter,
                _tensor_sha256(
                    parameter,
                    domain=b"lfm2-shared-unselected-parameter-v1",
                ),
            )
            for name, parameter in parent_parameters.items()
            if id(parameter) not in selected_original_ids
        }
        buffers = _activation._buffer_snapshot(model)
        structure = _activation._module_snapshot(layers)
        first_weight = selected[0].module.weight
        requires_grad_values = {
            value.requires_grad for value in plan.factorization.layers
        }
        if len(requires_grad_values) != 1:
            raise Lfm2SharedResidualBasisTransactionError(
                "Selected w2 weights disagree on requires_grad."
            )
        basis = Lfm2SharedResidualBasis(
            plan.factorization.output_basis,
            requires_grad=requires_grad_values.pop(),
            device=first_weight.device,
            dtype=first_weight.dtype,
        )
        layer_plan_lookup = {
            value.layer_index: value for value in plan.factorization.layers
        }
        bindings: list[_ProjectionBinding] = []
        for dense in selected:
            layer_plan = layer_plan_lookup.get(dense.layer_index)
            if (
                layer_plan is None
                or tuple(dense.module.weight.shape) != layer_plan.matrix_shape
                or _tensor_sha256(
                    dense.module.weight,
                    domain=_SOURCE_DOMAIN,
                )
                != layer_plan.source_weight_sha256
                or bool(dense.module.weight.requires_grad)
                != layer_plan.requires_grad
            ):
                raise Lfm2SharedResidualBasisTransactionError(
                    f"Planned source w2 layer {dense.layer_index} changed."
                )
            installed = Lfm2SharedResidualFactorizedLinear(
                layer_plan.input_factor,
                basis,
                requires_grad=layer_plan.requires_grad,
                device=dense.module.weight.device,
                dtype=dense.module.weight.dtype,
            )
            bindings.append(
                _ProjectionBinding(
                    owner=dense.owner,
                    layer_index=dense.layer_index,
                    original=dense.module,
                    installed=installed,
                )
            )
        try:
            _attach_basis(model, basis)
            for value in bindings:
                _assign_projection(value.owner, value.installed)
            if _activation._module_snapshot(layers) != structure:
                raise Lfm2SharedResidualBasisTransactionError(
                    "Installation changed a decoder shell or non-FFN operator."
                )
            _activation._verify_buffers(model, buffers)
            parameters_after = dict(
                model.named_parameters(remove_duplicate=False)
            )
            for name, (parameter, digest) in unselected_snapshot.items():
                if (
                    name not in parameters_after
                    or parameters_after[name] is not parameter
                    or _tensor_sha256(
                        parameters_after[name],
                        domain=b"lfm2-shared-unselected-parameter-v1",
                    )
                    != digest
                ):
                    raise Lfm2SharedResidualBasisTransactionError(
                        f"Unselected Parameter {name!r} changed."
                    )
            basis_occurrences = [
                name
                for name, parameter in parameters_after.items()
                if parameter is basis.output_basis
            ]
            private_parameters = tuple(
                value.installed.input_factor for value in bindings
            )
            new_ids = {
                id(parameter)
                for parameter in parameters_after.values()
                if id(parameter)
                not in {id(value) for value in parent_parameters.values()}
            }
            expected_new_ids = {
                id(basis.output_basis),
                *(id(value) for value in private_parameters),
            }
            if (
                len(basis_occurrences) != 1
                or new_ids != expected_new_ids
                or len(new_ids) != 1 + len(bindings)
                or len({id(value) for value in private_parameters})
                != len(bindings)
                or any(
                    value.installed.basis_owner is not basis
                    for value in bindings
                )
                or any(
                    hasattr(value.installed, "weight")
                    for value in bindings
                )
            ):
                raise Lfm2SharedResidualBasisTransactionError(
                    "The installed basis/private Parameter namespace is not exact."
                )
            logical_after, unique_after = _parameter_counts(model)
            if (
                logical_after != plan.projected_logical_parameters_after
                or unique_after != plan.projected_unique_parameters_after
                or unique_before - unique_after
                != plan.projected_parameters_saved
            ):
                raise Lfm2SharedResidualBasisTransactionError(
                    "Installed physical Parameter accounting differs from plan."
                )
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise Lfm2SharedResidualBasisTransactionError(
                    "A gradient appeared during zero-training installation."
                )
            report = Lfm2SharedResidualBasisReport(
                schema=REPORT_SCHEMA,
                parent_identity_before=plan.parent_identity_sha256,
                model_identity_after=_model_identity(model),
                evidence_sha256=evidence.evidence_sha256,
                selected_roles=plan.selected_roles,
                rank=plan.rank,
                logical_parameters_before=logical_before,
                logical_parameters_after=logical_after,
                unique_parameters_before=unique_before,
                unique_parameters_after=unique_after,
                parameters_saved=unique_before - unique_after,
                dense_parameters_removed=plan.factorization.dense_parameters,
                shared_basis_parameters_added=(
                    plan.factorization.shared_basis_parameters
                ),
                private_factor_parameters_added=(
                    plan.factorization.private_factor_parameters
                ),
                shared_basis_parameter_occurrences=len(basis_occurrences),
                shared_basis_unique_parameter_objects=1,
                private_factor_unique_parameter_objects=len(bindings),
                integer_metadata_values=plan.integer_metadata_values,
                integer_tensor_count=plan.integer_tensor_count,
                integer_tensor_bytes=plan.integer_tensor_bytes,
                unselected_parameters_preserved_exactly=True,
                attention_and_recurrent_operators_preserved=True,
                buffers_preserved=True,
                dense_w2_weights_retained=False,
                dense_w2_weights_registered_in_runtime=False,
                rollback_escrow_retains_original_modules=True,
            )
            return InstalledLfm2SharedResidualBasis(
                model=model,
                report=report,
                bindings=tuple(bindings),
                parent_parameters=parent_parameters,
                parent_identity=plan.parent_identity_sha256,
                buffers=buffers,
            )
        except BaseException as exc:
            for value in bindings:
                _assign_projection(value.owner, value.original)
            if hasattr(model, BASIS_ATTRIBUTE):
                _detach_basis(model)
            try:
                _activation._verify_buffers(model, buffers)
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
                raise Lfm2SharedResidualBasisTransactionError(
                    "Installation failed and rollback could not be authenticated."
                ) from rollback_exc
            if not exact:
                raise Lfm2SharedResidualBasisTransactionError(
                    "Installation failed and exact rollback failed."
                ) from exc
            raise


def install_lfm2_shared_residual_basis(
    model: Any,
    evidence: Lfm2ActivationFactorizationEvidence,
    *,
    target_unique_parameter_budget: int,
    layer_indices: Sequence[int] | None = None,
    ridge: float = 1e-6,
    basis_weighting: Literal["layer_balanced", "raw_stack"] = (
        "layer_balanced"
    ),
) -> InstalledLfm2SharedResidualBasis:
    """Plan and atomically install a shared residual-output-basis fold."""

    plan = plan_lfm2_shared_residual_basis(
        model,
        evidence,
        target_unique_parameter_budget=target_unique_parameter_budget,
        layer_indices=layer_indices,
        roles=("w2",),
        ridge=ridge,
        basis_weighting=basis_weighting,
    )
    return install_lfm2_shared_residual_basis_plan(model, evidence, plan)


build_lfm2_shared_residual_basis_plan = plan_lfm2_shared_residual_basis


__all__ = [
    "BASIS_ATTRIBUTE",
    "METHOD",
    "PLAN_SCHEMA",
    "REPORT_SCHEMA",
    "InstalledLfm2SharedResidualBasis",
    "Lfm2SharedResidualBasis",
    "Lfm2SharedResidualBasisError",
    "Lfm2SharedResidualBasisPlan",
    "Lfm2SharedResidualBasisReport",
    "Lfm2SharedResidualBasisTopologyError",
    "Lfm2SharedResidualBasisTransactionError",
    "Lfm2SharedResidualFactorization",
    "Lfm2SharedResidualFactorizedLinear",
    "Lfm2SharedResidualLayerFactor",
    "build_lfm2_shared_residual_basis_plan",
    "factorize_shared_residual_output_basis",
    "install_lfm2_shared_residual_basis",
    "install_lfm2_shared_residual_basis_plan",
    "plan_lfm2_shared_residual_basis",
]
