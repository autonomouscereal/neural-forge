"""Architecture-neutral multi-parent hidden-trajectory compilation.

Parents only need row-aligned hidden states from standard Hugging Face
``output_hidden_states`` calls. Their hidden widths, layer counts, model
families, and internal block implementations may differ. The compiler builds a
weighted row kernel at each target depth, factors it into a deterministic child
space, fixes the coordinate gauge across depth, and fits zero-gradient dual
ridge maps from every parent into that shared trajectory.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence


ExactnessClass = Literal[
    "algebraically-exact",
    "finite-trace-exact",
    "finite-trace-approximate",
]

_TRACE_EXACT_RELATIVE_RMSE_TOLERANCE = 1e-8
_KERNEL_EXACT_RETAINED_FRACTION_TOLERANCE = 1e-12
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CanonicalKernelError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CanonicalRowIdentity:
    """Architecture-neutral proof that parent rows describe the same examples."""

    split_name: str
    pooling: Literal["mean", "last_valid", "token_rows"]
    row_ids: tuple[str, ...]
    raw_prompt_sha256: tuple[str, ...]
    corpus_sha256: str
    tokenizer_fingerprint: str
    cache_fingerprint: str

    def __post_init__(self) -> None:
        split = _nonempty_text(self.split_name, "split_name")
        if self.pooling not in {"mean", "last_valid", "token_rows"}:
            raise CanonicalKernelError(
                "Canonical row pooling must be mean, last_valid, or token_rows."
            )
        row_ids = tuple(self.row_ids)
        prompt_hashes = tuple(self.raw_prompt_sha256)
        if not row_ids or len(row_ids) != len(prompt_hashes):
            raise CanonicalKernelError(
                "Canonical row IDs and raw-prompt hashes must be nonempty and aligned."
            )
        if any(not isinstance(value, str) or not value for value in row_ids):
            raise CanonicalKernelError("Canonical row IDs must be nonempty strings.")
        if len(set(row_ids)) != len(row_ids):
            raise CanonicalKernelError("Canonical row IDs must be unique and ordered.")
        for index, digest in enumerate(prompt_hashes):
            _sha256(digest, f"raw_prompt_sha256[{index}]")
        object.__setattr__(self, "split_name", split)
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "raw_prompt_sha256", prompt_hashes)
        object.__setattr__(
            self,
            "corpus_sha256",
            _sha256(self.corpus_sha256, "corpus_sha256"),
        )
        object.__setattr__(
            self,
            "tokenizer_fingerprint",
            _sha256(self.tokenizer_fingerprint, "tokenizer_fingerprint"),
        )
        object.__setattr__(
            self,
            "cache_fingerprint",
            _sha256(self.cache_fingerprint, "cache_fingerprint"),
        )

    @property
    def alignment_fingerprint(self) -> str:
        return _fingerprint(
            {
                "schema": "neural-forge-canonical-row-alignment-v1",
                "split_name": self.split_name,
                "pooling": self.pooling,
                "row_ids": list(self.row_ids),
                "raw_prompt_sha256": list(self.raw_prompt_sha256),
                "corpus_sha256": self.corpus_sha256,
            }
        )

    def report(self) -> dict[str, Any]:
        return {
            "split_name": self.split_name,
            "pooling": self.pooling,
            "row_count": len(self.row_ids),
            "row_ids_sha256": _fingerprint(list(self.row_ids)),
            "raw_prompt_inventory_sha256": _fingerprint(
                list(self.raw_prompt_sha256)
            ),
            "corpus_sha256": self.corpus_sha256,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "cache_fingerprint": self.cache_fingerprint,
            "alignment_fingerprint": self.alignment_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class CanonicalDepthCover:
    """Explicit monotone child boundaries covering every source layer once."""

    boundary_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        boundaries = tuple(self.boundary_indices)
        if len(boundaries) < 2:
            raise CanonicalKernelError(
                "A canonical depth cover needs at least two boundaries."
            )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in boundaries):
            raise CanonicalKernelError(
                "Canonical depth-cover boundaries must be integers."
            )
        if boundaries[0] != 0 or boundaries[-1] <= 0:
            raise CanonicalKernelError(
                "A canonical depth cover must start at zero and cover source layers."
            )
        if any(left > right for left, right in zip(boundaries, boundaries[1:])):
            raise CanonicalKernelError(
                "Canonical depth-cover boundaries must be monotone."
            )
        object.__setattr__(self, "boundary_indices", boundaries)

    @property
    def source_layer_count(self) -> int:
        return self.boundary_indices[-1]

    @property
    def target_layer_count(self) -> int:
        return len(self.boundary_indices) - 1

    @property
    def source_spans(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(range(start, stop))
            for start, stop in zip(
                self.boundary_indices[:-1],
                self.boundary_indices[1:],
                strict=True,
            )
        )

    def report(self) -> dict[str, Any]:
        return {
            "boundary_indices": list(self.boundary_indices),
            "source_spans": [list(span) for span in self.source_spans],
            "source_layer_count": self.source_layer_count,
            "target_layer_count": self.target_layer_count,
            "source_layers_covered_exactly_once": True,
            "empty_identity_spans": sum(not span for span in self.source_spans),
        }


def build_proportional_depth_cover(
    source_layer_count: int,
    target_layer_count: int,
) -> CanonicalDepthCover:
    source = _positive_int(source_layer_count, "source_layer_count")
    target = _positive_int(target_layer_count, "target_layer_count")
    return CanonicalDepthCover(
        tuple(round(boundary * source / target) for boundary in range(target + 1))
    )


@dataclass(frozen=True, slots=True)
class CanonicalParentTrace:
    name: str
    layer_states: tuple[Any, ...]
    row_identity: CanonicalRowIdentity
    depth_cover: CanonicalDepthCover
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class CanonicalParentMap:
    parent_name: str
    source_layer_index: int
    weight: Any
    bias: Any
    relative_rmse: float
    mean_cosine_similarity: float
    effective_ridge: float

    @property
    def exactness_class(self) -> ExactnessClass:
        if self.relative_rmse <= _TRACE_EXACT_RELATIVE_RMSE_TOLERANCE:
            return "finite-trace-exact"
        return "finite-trace-approximate"

    def report(self) -> dict[str, Any]:
        return {
            "parent_name": self.parent_name,
            "source_layer_index": self.source_layer_index,
            "source_hidden_size": int(self.weight.shape[0]),
            "target_hidden_size": int(self.weight.shape[1]),
            "relative_rmse": self.relative_rmse,
            "mean_cosine_similarity": self.mean_cosine_similarity,
            "effective_ridge": self.effective_ridge,
            "exactness_class": self.exactness_class,
            "exactness_scope": "fingerprinted calibration rows only",
            "trace_exact_relative_rmse_tolerance": (
                _TRACE_EXACT_RELATIVE_RMSE_TOLERANCE
            ),
            "algebraically_exact": False,
            "finite": True,
        }


@dataclass(frozen=True, slots=True)
class CanonicalKernelLayer:
    target_boundary_index: int
    canonical_rows: Any
    parent_maps: tuple[CanonicalParentMap, ...]
    effective_rank: int
    retained_kernel_fraction: float
    canonical_rms: float
    gauge_alignment_cosine: float | None
    fingerprint: str

    @property
    def exactness_class(self) -> ExactnessClass:
        kernel_exact = self.retained_kernel_fraction >= (
            1.0 - _KERNEL_EXACT_RETAINED_FRACTION_TOLERANCE
        )
        maps_exact = all(
            mapping.exactness_class == "finite-trace-exact"
            for mapping in self.parent_maps
        )
        if kernel_exact and maps_exact:
            return "finite-trace-exact"
        return "finite-trace-approximate"

    def report(self) -> dict[str, Any]:
        return {
            "target_boundary_index": self.target_boundary_index,
            "row_count": int(self.canonical_rows.shape[0]),
            "target_hidden_size": int(self.canonical_rows.shape[1]),
            "effective_rank": self.effective_rank,
            "retained_kernel_fraction": self.retained_kernel_fraction,
            "canonical_rms": self.canonical_rms,
            "gauge_alignment_cosine": self.gauge_alignment_cosine,
            "exactness_class": self.exactness_class,
            "exactness_scope": (
                "weighted parent row-kernel geometry and parent maps on "
                "fingerprinted calibration rows"
            ),
            "kernel_exact_retained_fraction_tolerance": (
                _KERNEL_EXACT_RETAINED_FRACTION_TOLERANCE
            ),
            "algebraically_exact": False,
            "parent_maps": [mapping.report() for mapping in self.parent_maps],
            "fingerprint": self.fingerprint,
            "finite": True,
        }


@dataclass(frozen=True, slots=True)
class CanonicalKernelPlan:
    parent_names: tuple[str, ...]
    normalized_parent_weights: tuple[float, ...]
    parent_row_identities: tuple[CanonicalRowIdentity, ...]
    source_depth_covers: tuple[CanonicalDepthCover, ...]
    row_alignment_fingerprint: str
    target_layer_count: int
    target_hidden_size: int
    row_count: int
    layers: tuple[CanonicalKernelLayer, ...]
    fingerprint: str

    @property
    def boundary_rows(self) -> tuple[Any, ...]:
        return tuple(layer.canonical_rows for layer in self.layers)

    @property
    def exactness_class(self) -> ExactnessClass:
        if all(layer.exactness_class == "finite-trace-exact" for layer in self.layers):
            return "finite-trace-exact"
        return "finite-trace-approximate"

    def report(self) -> dict[str, Any]:
        return {
            "schema": "neural-forge-canonical-parent-kernel-v1",
            "method": "weighted-row-kernel-depth-aligned-dual-ridge",
            "parent_names": list(self.parent_names),
            "normalized_parent_weights": list(self.normalized_parent_weights),
            "row_alignment_fingerprint": self.row_alignment_fingerprint,
            "parent_row_identities": [
                identity.report() for identity in self.parent_row_identities
            ],
            "source_depth_covers": [
                cover.report() for cover in self.source_depth_covers
            ],
            "target_layer_count": self.target_layer_count,
            "target_boundary_count": len(self.layers),
            "target_hidden_size": self.target_hidden_size,
            "row_count": self.row_count,
            "layers": [layer.report() for layer in self.layers],
            "fingerprint": self.fingerprint,
            "architecture_neutral_parent_interface": "output_hidden_states",
            "exactness_class": self.exactness_class,
            "exactness_scope": (
                "canonical target construction on fingerprinted calibration rows; "
                "not whole-model functional equivalence"
            ),
            "algebraically_exact": False,
            "training_steps": 0,
            "gradients_used": False,
        }


@dataclass(frozen=True, slots=True)
class CanonicalBoundarySplits:
    calibration_plan: CanonicalKernelPlan
    validation_rows: tuple[Any, ...]
    fingerprint: str

    @property
    def calibration_rows(self) -> tuple[Any, ...]:
        return self.calibration_plan.boundary_rows

    def report(self) -> dict[str, Any]:
        return {
            "schema": "neural-forge-canonical-boundary-splits-v1",
            "calibration_plan": self.calibration_plan.report(),
            "validation_boundary_count": len(self.validation_rows),
            "validation_row_count": int(self.validation_rows[0].shape[0]),
            "validation_sha256": _tensor_sequence_sha256(self.validation_rows),
            "exactness_class": "finite-trace-approximate",
            "exactness_scope": (
                "calibration-fitted parent maps applied to held-out validation rows"
            ),
            "algebraically_exact": False,
            "fingerprint": self.fingerprint,
            "training_steps": 0,
            "gradients_used": False,
        }


def build_canonical_kernel_plan(
    parents: Sequence[CanonicalParentTrace],
    *,
    target_layer_count: int,
    target_hidden_size: int,
    ridge: float = 1e-6,
    center_rows: bool = True,
) -> CanonicalKernelPlan:
    """Compile row-aligned parent trajectories into one child coordinate gauge."""

    torch = _require_torch()
    target_layer_count = _positive_int(target_layer_count, "target_layer_count")
    target_hidden_size = _positive_int(target_hidden_size, "target_hidden_size")
    ridge = _finite_nonnegative(ridge, "ridge")
    if not isinstance(center_rows, bool):
        raise CanonicalKernelError("center_rows must be boolean.")
    if not isinstance(parents, Sequence) or not parents:
        raise CanonicalKernelError("parents must contain at least one trace.")

    normalized_parents = _validated_sorted_parents(
        torch,
        parents,
        target_layer_count=target_layer_count,
    )
    names = tuple(parent.name for parent in normalized_parents)
    if len(set(names)) != len(names):
        raise CanonicalKernelError("Canonical parent names must be unique.")
    alignment_fingerprints = {
        parent.row_identity.alignment_fingerprint for parent in normalized_parents
    }
    if len(alignment_fingerprints) != 1:
        raise CanonicalKernelError(
            "Canonical parents do not describe the same ordered raw-prompt rows."
        )
    row_alignment_fingerprint = next(iter(alignment_fingerprints))
    raw_weights = torch.tensor(
        [parent.weight for parent in normalized_parents],
        dtype=torch.float64,
    )
    normalized_weights = raw_weights / raw_weights.sum()
    row_count = int(normalized_parents[0].layer_states[0].shape[0])
    if any(
        int(state.shape[0]) != row_count
        for parent in normalized_parents
        for state in parent.layer_states
    ):
        raise CanonicalKernelError(
            "Every parent hidden state must contain the same prompt-aligned rows."
        )

    source_indices = tuple(
        parent.depth_cover.boundary_indices for parent in normalized_parents
    )
    layers = []
    previous_rows = None
    for boundary in range(target_layer_count + 1):
        states = tuple(
            parent.layer_states[source_indices[parent_index][boundary]]
            for parent_index, parent in enumerate(normalized_parents)
        )
        canonical, effective_rank, retained_fraction = _canonical_rows(
            torch,
            states,
            normalized_weights,
            target_hidden_size,
            center_rows=center_rows,
        )
        gauge_cosine = None
        if previous_rows is not None:
            canonical, gauge_cosine = _align_gauge(
                torch,
                canonical,
                previous_rows,
            )
        parent_maps = tuple(
            _fit_parent_map(
                torch,
                parent.name,
                source_indices[parent_index][boundary],
                states[parent_index],
                canonical,
                ridge=ridge,
            )
            for parent_index, parent in enumerate(normalized_parents)
        )
        layer_fingerprint = _fingerprint(
            {
                "target_boundary_index": boundary,
                "effective_rank": effective_rank,
                "retained_kernel_fraction": retained_fraction,
                "canonical_rows_sha256": _tensor_sha256(canonical),
                "parent_maps": [
                    {
                        "parent_name": mapping.parent_name,
                        "source_layer_index": mapping.source_layer_index,
                        "weight_sha256": _tensor_sha256(mapping.weight),
                        "bias_sha256": _tensor_sha256(mapping.bias),
                    }
                    for mapping in parent_maps
                ],
            }
        )
        layers.append(
            CanonicalKernelLayer(
                target_boundary_index=boundary,
                canonical_rows=canonical,
                parent_maps=parent_maps,
                effective_rank=effective_rank,
                retained_kernel_fraction=retained_fraction,
                canonical_rms=float(torch.sqrt(canonical.square().mean())),
                gauge_alignment_cosine=gauge_cosine,
                fingerprint=layer_fingerprint,
            )
        )
        previous_rows = canonical

    plan_fingerprint = _fingerprint(
        {
            "schema": "neural-forge-canonical-parent-kernel-v1",
            "parent_names": names,
            "normalized_parent_weights": [float(value) for value in normalized_weights],
            "target_layer_count": target_layer_count,
            "target_hidden_size": target_hidden_size,
            "row_count": row_count,
            "row_alignment_fingerprint": row_alignment_fingerprint,
            "source_depth_indices": [list(indices) for indices in source_indices],
            "layer_fingerprints": [layer.fingerprint for layer in layers],
        }
    )
    return CanonicalKernelPlan(
        parent_names=names,
        normalized_parent_weights=tuple(float(value) for value in normalized_weights),
        parent_row_identities=tuple(
            parent.row_identity for parent in normalized_parents
        ),
        source_depth_covers=tuple(parent.depth_cover for parent in normalized_parents),
        row_alignment_fingerprint=row_alignment_fingerprint,
        target_layer_count=target_layer_count,
        target_hidden_size=target_hidden_size,
        row_count=row_count,
        layers=tuple(layers),
        fingerprint=plan_fingerprint,
    )


def build_canonical_boundary_splits(
    calibration_parents: Sequence[CanonicalParentTrace],
    validation_parents: Sequence[CanonicalParentTrace],
    *,
    target_layer_count: int,
    target_hidden_size: int,
    ridge: float = 1e-6,
    center_rows: bool = True,
) -> CanonicalBoundarySplits:
    torch = _require_torch()
    if len(calibration_parents) != len(validation_parents):
        raise CanonicalKernelError(
            "Calibration and validation parent inventories must have equal lengths."
        )
    ordered_calibration = _validated_sorted_parents(
        torch,
        calibration_parents,
        target_layer_count=target_layer_count,
    )
    ordered_validation = _validated_sorted_parents(
        torch,
        validation_parents,
        target_layer_count=target_layer_count,
    )
    calibration_names = tuple(parent.name for parent in ordered_calibration)
    validation_names = tuple(parent.name for parent in ordered_validation)
    if calibration_names != validation_names:
        raise CanonicalKernelError(
            "Calibration and validation parent names must have the same order."
        )
    calibration_weights = tuple(float(parent.weight) for parent in ordered_calibration)
    validation_weights = tuple(float(parent.weight) for parent in ordered_validation)
    if calibration_weights != validation_weights:
        raise CanonicalKernelError(
            "Calibration and validation parent weights must match exactly."
        )
    validation_alignments = {
        parent.row_identity.alignment_fingerprint for parent in ordered_validation
    }
    if len(validation_alignments) != 1:
        raise CanonicalKernelError(
            "Validation parents do not describe the same ordered raw-prompt rows."
        )
    for calibration, validation in zip(
        ordered_calibration,
        ordered_validation,
        strict=True,
    ):
        if (
            calibration.row_identity.tokenizer_fingerprint
            != validation.row_identity.tokenizer_fingerprint
        ):
            raise CanonicalKernelError(
                "A parent's calibration and validation tokenizer fingerprints differ."
            )
        if calibration.depth_cover != validation.depth_cover:
            raise CanonicalKernelError(
                "A parent's calibration and validation depth covers differ."
            )
    plan = build_canonical_kernel_plan(
        ordered_calibration,
        target_layer_count=target_layer_count,
        target_hidden_size=target_hidden_size,
        ridge=ridge,
        center_rows=center_rows,
    )
    validation_boundaries = []
    for layer in plan.layers:
        rows = []
        for parent, mapping in zip(ordered_validation, layer.parent_maps, strict=True):
            if mapping.source_layer_index >= len(parent.layer_states):
                raise CanonicalKernelError(
                    "A validation parent is missing a calibration-selected boundary."
                )
            rows.append(parent.layer_states[mapping.source_layer_index])
        validation_boundaries.append(
            fuse_parent_rows(rows, layer, plan.normalized_parent_weights)
        )
    validation_rows = tuple(validation_boundaries)
    fingerprint = _fingerprint(
        {
            "schema": "neural-forge-canonical-boundary-splits-v1",
            "calibration_plan_fingerprint": plan.fingerprint,
            "validation_sha256": _tensor_sequence_sha256(validation_rows),
        }
    )
    return CanonicalBoundarySplits(
        calibration_plan=plan,
        validation_rows=validation_rows,
        fingerprint=fingerprint,
    )


def canonical_parent_trace_from_cache(
    cache: Any,
    *,
    name: str,
    row_identity: CanonicalRowIdentity,
    depth_cover: CanonicalDepthCover,
    weight: float = 1.0,
    pooling: Literal["mean", "last_valid"] = "last_valid",
) -> CanonicalParentTrace:
    """Load prompt rows from an immutable generic parent trace cache."""

    torch = _require_torch()
    if pooling not in {"mean", "last_valid"}:
        raise CanonicalKernelError("pooling must be mean or last_valid.")
    if not isinstance(row_identity, CanonicalRowIdentity):
        raise CanonicalKernelError("row_identity must be a CanonicalRowIdentity.")
    if row_identity.pooling != pooling:
        raise CanonicalKernelError(
            "row_identity pooling does not match cache pooling."
        )
    if not isinstance(depth_cover, CanonicalDepthCover):
        raise CanonicalKernelError("depth_cover must be a CanonicalDepthCover.")
    cache_key = getattr(cache, "cache_key", None)
    if cache_key is not None and cache_key != row_identity.cache_fingerprint:
        raise CanonicalKernelError(
            "row_identity cache fingerprint does not match the opened cache."
        )
    iterator = getattr(cache, "iter_batches", None)
    if not callable(iterator):
        raise CanonicalKernelError("cache must expose iter_batches().")
    by_layer: dict[int, list[Any]] = {}
    expected_indices = None
    batch_count = 0
    for batch in iterator():
        hidden_states = getattr(batch, "hidden_states", None)
        mask = getattr(batch, "valid_token_mask", None)
        if hidden_states is None or mask is None:
            raise CanonicalKernelError(
                "A cached batch is missing hidden states or its valid-token mask."
            )
        indices = tuple(sorted(int(index) for index in hidden_states))
        if not indices or (
            expected_indices is not None and indices != expected_indices
        ):
            raise CanonicalKernelError(
                "Cached batches do not expose one consistent hidden-state inventory."
            )
        expected_indices = indices
        mask_rows = mask.detach().to(device="cpu", dtype=torch.bool)
        if mask_rows.ndim != 2 or bool((mask_rows.sum(dim=1) <= 0).any()):
            raise CanonicalKernelError("A cached valid-token mask is malformed.")
        for index in indices:
            values = (
                hidden_states[index]
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.float64,
                )
            )
            if values.ndim != 3 or tuple(values.shape[:2]) != tuple(mask_rows.shape):
                raise CanonicalKernelError(
                    "A cached hidden state does not match its batch/sequence mask."
                )
            if pooling == "mean":
                pooled = (values * mask_rows.unsqueeze(-1)).sum(dim=1) / mask_rows.sum(
                    dim=1,
                    keepdim=True,
                )
            else:
                positions = torch.arange(mask_rows.shape[1]).reshape(1, -1)
                last = (
                    torch.where(mask_rows, positions, -torch.ones_like(positions))
                    .max(dim=1)
                    .values
                )
                pooled = values[
                    torch.arange(values.shape[0]),
                    last,
                ]
            if not bool(torch.isfinite(pooled).all()):
                raise CanonicalKernelError("Cached pooled hidden rows are non-finite.")
            by_layer.setdefault(index, []).append(pooled.contiguous())
        batch_count += 1
    if batch_count == 0 or expected_indices is None:
        raise CanonicalKernelError("The parent trace cache contains no batches.")
    pooled_row_count = sum(int(rows.shape[0]) for rows in by_layer[expected_indices[0]])
    if pooled_row_count != len(row_identity.row_ids):
        raise CanonicalKernelError(
            "row_identity row count does not match pooled cache rows."
        )
    if len(expected_indices) != depth_cover.source_layer_count + 1:
        raise CanonicalKernelError(
            "depth_cover source layer count does not match cached boundaries."
        )
    return CanonicalParentTrace(
        name=name,
        layer_states=tuple(
            torch.cat(by_layer[index], dim=0).contiguous() for index in expected_indices
        ),
        row_identity=row_identity,
        depth_cover=depth_cover,
        weight=_finite_positive(weight, "weight"),
    )


def apply_parent_map(rows: Any, mapping: CanonicalParentMap) -> Any:
    torch = _require_torch()
    values = _matrix(torch, rows, "rows")
    if int(values.shape[1]) != int(mapping.weight.shape[0]):
        raise CanonicalKernelError("Parent-map input width does not match its rows.")
    result = values.matmul(mapping.weight) + mapping.bias
    if not bool(torch.isfinite(result).all()):
        raise CanonicalKernelError("Parent-map application produced non-finite rows.")
    return result.contiguous()


def fuse_parent_rows(
    parent_rows: Sequence[Any],
    layer: CanonicalKernelLayer,
    parent_weights: Sequence[float],
) -> Any:
    torch = _require_torch()
    if len(parent_rows) != len(layer.parent_maps) or len(parent_weights) != len(
        layer.parent_maps
    ):
        raise CanonicalKernelError(
            "Fused parent rows, maps, and weights must have equal lengths."
        )
    weights = torch.tensor(
        [_finite_positive(value, "parent_weights") for value in parent_weights],
        dtype=torch.float64,
    )
    weights = weights / weights.sum()
    encoded = [
        apply_parent_map(rows, mapping)
        for rows, mapping in zip(parent_rows, layer.parent_maps, strict=True)
    ]
    row_count = int(encoded[0].shape[0])
    if any(tuple(value.shape) != tuple(encoded[0].shape) for value in encoded):
        raise CanonicalKernelError("Fused parent-map outputs are not row aligned.")
    result = torch.zeros_like(encoded[0])
    for weight, value in zip(weights, encoded, strict=True):
        result.add_(value, alpha=float(weight))
    if int(result.shape[0]) != row_count or not bool(torch.isfinite(result).all()):
        raise CanonicalKernelError("Fused parent rows are invalid or non-finite.")
    return result.contiguous()


def _validated_sorted_parents(
    torch: Any,
    parents: Sequence[CanonicalParentTrace],
    *,
    target_layer_count: int,
) -> tuple[CanonicalParentTrace, ...]:
    normalized = tuple(
        _validate_parent(
            torch,
            parent,
            index,
            target_layer_count=target_layer_count,
        )
        for index, parent in enumerate(parents)
    )
    return tuple(sorted(normalized, key=lambda parent: parent.name))


def _validate_parent(
    torch: Any,
    value: Any,
    index: int,
    *,
    target_layer_count: int,
) -> CanonicalParentTrace:
    if not isinstance(value, CanonicalParentTrace):
        raise CanonicalKernelError(f"parents[{index}] must be a CanonicalParentTrace.")
    if not isinstance(value.name, str) or not value.name.strip():
        raise CanonicalKernelError(f"parents[{index}].name must be non-empty.")
    weight = _finite_positive(value.weight, f"parents[{index}].weight")
    if len(value.layer_states) < 2:
        raise CanonicalKernelError(
            f"parents[{index}] must include at least two layer boundaries."
        )
    states = tuple(
        _matrix(torch, state, f"parents[{index}].layer_states[{layer}]")
        for layer, state in enumerate(value.layer_states)
    )
    if not isinstance(value.row_identity, CanonicalRowIdentity):
        raise CanonicalKernelError(
            f"parents[{index}].row_identity must be a CanonicalRowIdentity."
        )
    if len(value.row_identity.row_ids) != int(states[0].shape[0]):
        raise CanonicalKernelError(
            f"parents[{index}] row identity does not match its hidden-state rows."
        )
    if not isinstance(value.depth_cover, CanonicalDepthCover):
        raise CanonicalKernelError(
            f"parents[{index}].depth_cover must be a CanonicalDepthCover."
        )
    if value.depth_cover.target_layer_count != target_layer_count:
        raise CanonicalKernelError(
            f"parents[{index}] depth cover does not match target_layer_count."
        )
    if value.depth_cover.source_layer_count != len(states) - 1:
        raise CanonicalKernelError(
            f"parents[{index}] depth cover does not match its source boundaries."
        )
    return CanonicalParentTrace(
        name=value.name.strip(),
        layer_states=states,
        row_identity=value.row_identity,
        depth_cover=value.depth_cover,
        weight=weight,
    )


def _canonical_rows(
    torch: Any,
    states: Sequence[Any],
    weights: Any,
    target_hidden_size: int,
    *,
    center_rows: bool,
) -> tuple[Any, int, float]:
    row_count = int(states[0].shape[0])
    kernel = torch.zeros((row_count, row_count), dtype=torch.float64)
    target_rms = 0.0
    for parent_weight, state in zip(weights, states, strict=True):
        values = state - state.mean(dim=0, keepdim=True) if center_rows else state
        norm = torch.linalg.vector_norm(values)
        if float(norm) <= torch.finfo(torch.float64).tiny:
            raise CanonicalKernelError("A parent layer has no usable row geometry.")
        normalized = values / norm
        kernel.add_(normalized.matmul(normalized.T), alpha=float(parent_weight))
        target_rms += float(parent_weight) * float(torch.sqrt(state.square().mean()))

    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    order = torch.argsort(eigenvalues, descending=True, stable=True)
    eigenvalues = eigenvalues.index_select(0, order).clamp_min(0.0)
    eigenvectors = eigenvectors.index_select(1, order)
    total_energy = float(eigenvalues.sum())
    floor = max(
        float(eigenvalues[0]) * torch.finfo(torch.float64).eps * row_count,
        torch.finfo(torch.float64).tiny,
    )
    available_rank = int((eigenvalues > floor).sum())
    retained_rank = min(target_hidden_size, available_rank)
    if retained_rank <= 0:
        raise CanonicalKernelError("The weighted parent kernel has zero rank.")
    basis = eigenvectors[:, :retained_rank]
    basis = _fix_column_signs(torch, basis)
    canonical = basis * eigenvalues[:retained_rank].sqrt().reshape(1, -1)
    if retained_rank < target_hidden_size:
        canonical = torch.cat(
            [
                canonical,
                torch.zeros(
                    (row_count, target_hidden_size - retained_rank),
                    dtype=torch.float64,
                ),
            ],
            dim=1,
        )
    canonical_rms = float(torch.sqrt(canonical.square().mean()))
    if canonical_rms <= torch.finfo(torch.float64).tiny:
        raise CanonicalKernelError("Canonical rows have zero scale.")
    canonical = canonical * (target_rms / canonical_rms)
    retained = float(eigenvalues[:retained_rank].sum()) / max(
        total_energy,
        torch.finfo(torch.float64).tiny,
    )
    return canonical.contiguous(), retained_rank, retained


def _align_gauge(torch: Any, rows: Any, reference: Any) -> tuple[Any, float]:
    cross = rows.T.matmul(reference)
    left, _, right = torch.linalg.svd(cross, full_matrices=False)
    rotation = left.matmul(right)
    aligned = rows.matmul(rotation)
    cosine = _mean_cosine(torch, aligned, reference)
    return aligned.contiguous(), cosine


def _fit_parent_map(
    torch: Any,
    parent_name: str,
    source_layer_index: int,
    source_rows: Any,
    target_rows: Any,
    *,
    ridge: float,
) -> CanonicalParentMap:
    ones = torch.ones((source_rows.shape[0], 1), dtype=torch.float64)
    design = torch.cat([source_rows, ones], dim=1)
    row_gram = design.matmul(design.T)
    trace_scale = float(torch.trace(row_gram)) / max(1, int(row_gram.shape[0]))
    effective_ridge = ridge * max(trace_scale, torch.finfo(torch.float64).tiny)
    system = row_gram + effective_ridge * torch.eye(
        row_gram.shape[0],
        dtype=torch.float64,
    )
    try:
        dual = torch.linalg.solve(system, target_rows)
    except RuntimeError as exc:
        raise CanonicalKernelError("Canonical parent-map ridge solve failed.") from exc
    parameters = design.T.matmul(dual)
    weight = parameters[:-1].contiguous()
    bias = parameters[-1].contiguous()
    prediction = source_rows.matmul(weight) + bias
    residual = prediction - target_rows
    relative_rmse = float(torch.sqrt(residual.square().mean())) / max(
        float(torch.sqrt(target_rows.square().mean())),
        torch.finfo(torch.float64).eps,
    )
    cosine = _mean_cosine(torch, prediction, target_rows)
    if not all(math.isfinite(value) for value in (relative_rmse, cosine)):
        raise CanonicalKernelError("Canonical parent-map metrics are non-finite.")
    return CanonicalParentMap(
        parent_name=parent_name,
        source_layer_index=source_layer_index,
        weight=weight,
        bias=bias,
        relative_rmse=relative_rmse,
        mean_cosine_similarity=cosine,
        effective_ridge=effective_ridge,
    )


def _fix_column_signs(torch: Any, values: Any) -> Any:
    result = values.clone()
    absolute = result.abs()
    pivot_rows = torch.argmax(absolute, dim=0)
    pivots = result[pivot_rows, torch.arange(result.shape[1])]
    signs = torch.where(pivots < 0.0, -torch.ones_like(pivots), torch.ones_like(pivots))
    return (result * signs.reshape(1, -1)).contiguous()


def _mean_cosine(torch: Any, left: Any, right: Any) -> float:
    numerator = (left * right).sum(dim=1)
    denominator = torch.linalg.vector_norm(left, dim=1) * torch.linalg.vector_norm(
        right,
        dim=1,
    )
    valid = denominator > torch.finfo(torch.float64).tiny
    if not bool(valid.any()):
        return 0.0
    return float((numerator[valid] / denominator[valid]).mean())


def _matrix(torch: Any, value: Any, name: str) -> Any:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise CanonicalKernelError(f"{name} must be a two-dimensional tensor.")
    if not torch.is_floating_point(value) or value.layout != torch.strided:
        raise CanonicalKernelError(f"{name} must be a dense floating-point tensor.")
    if any(int(size) <= 0 for size in value.shape):
        raise CanonicalKernelError(f"{name} cannot have an empty dimension.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise CanonicalKernelError(f"{name} contains NaN or infinity.")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CanonicalKernelError(f"{name} must be a positive integer.")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise CanonicalKernelError(f"{name} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CanonicalKernelError(f"{name} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise CanonicalKernelError(f"{name} must be finite and non-negative.")
    return result


def _finite_positive(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result == 0.0:
        raise CanonicalKernelError(f"{name} must be positive.")
    return result


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CanonicalKernelError(f"{name} must be a nonempty string.")
    return value.strip()


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CanonicalKernelError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _tensor_sha256(value: Any) -> str:
    contiguous = (
        value.detach().to(device="cpu", dtype=_require_torch().float64).contiguous()
    )
    return hashlib.sha256(contiguous.numpy().tobytes(order="C")).hexdigest()


def _tensor_sequence_sha256(values: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(values):
        digest.update(index.to_bytes(8, "little", signed=False))
        digest.update(_tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Canonical parent kernels require PyTorch.") from exc
    return torch


__all__ = [
    "CanonicalKernelError",
    "CanonicalBoundarySplits",
    "CanonicalKernelLayer",
    "CanonicalKernelPlan",
    "CanonicalDepthCover",
    "CanonicalParentMap",
    "CanonicalParentTrace",
    "CanonicalRowIdentity",
    "ExactnessClass",
    "apply_parent_map",
    "build_canonical_boundary_splits",
    "build_canonical_kernel_plan",
    "build_proportional_depth_cover",
    "canonical_parent_trace_from_cache",
    "fuse_parent_rows",
]
