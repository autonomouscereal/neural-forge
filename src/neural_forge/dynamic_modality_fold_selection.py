"""Modality-neutral fold-selection numerics shared by every native lane.

``NOTES-W3B.md`` item 7 and ``NOTES-W5C.md`` item 2 recorded the same debt: the
heldout-blind low-importance selector and its sparse anchor regression existed
as three byte-identical copies in :mod:`neural_forge.dynamic_neo_fold`,
:mod:`neural_forge.dynamic_audio_fold`, and
:mod:`neural_forge.dynamic_video_fold`.  This module is the single copy those
three lanes now delegate to.

The extraction is a numeric no-op **by construction**.  Every operation, its
argument order, its dtype promotion, and its iteration order are carried over
verbatim, because these functions feed sealed content-addressed fold caches
(``neural-forge-neo-native-fold-cache-v1``,
``neural-forge-native-audio-fold-cache-v1``,
``neural-forge-native-video-fold-cache-v1``) whose keys are the canonical
SHA-256 of a body that contains each selection.  Any drift -- a different
reduction order, a different clamp, a different tie-break -- silently
invalidates already-sealed artifacts, so it is not permitted here.

The only things a lane supplies are its vocabulary (its error type, its
message labels, and its own integer/float validators) and the dataclass that
holds a finished selection.  Nothing in this module names a modality.

Why the validators are injected rather than defined here: the three lanes'
``_finite`` helpers disagree on one message.  The NEO lane rejects a non-real
value with ``"<name> must be finite."``; the audio and video lanes say
``"<name> must be a real number."``.  That divergence predates this module and
is observable behavior on a sealed, fail-closed path, so it is preserved
exactly rather than unified.  See ``NOTES-W6B.md``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


__all__ = [
    "ModalityFoldSelectionVocabulary",
    "normalize_sparse_anchor_selection",
    "selection_from_statistics",
    "sparse_anchor_regression",
    "sparse_compensation_terms",
    "survivors_for",
]


@dataclass(frozen=True, slots=True)
class ModalityFoldSelectionVocabulary:
    """Everything lane-specific the shared selection numerics need.

    ``subject_label`` is the sentence-initial name a lane uses in its own
    messages (``"NEO"``, ``"Native audio"``, ``"Native video"``);
    ``selection_label`` is the mid-sentence form (``"NEO"``, ``"native
    audio"``, ``"native video"``).  Both are kept because the NEO lane
    capitalizes in both positions and the other two do not.
    """

    error_type: type[Exception]
    subject_label: str
    selection_label: str
    nonnegative_int: Callable[[Any, str], int]
    positive_int: Callable[[Any, str], int]
    finite: Callable[[Any, str], float]


def survivors_for(source_neurons: int, removed_neuron: int) -> tuple[int, ...]:
    """Return the target-order source indices left after one removal."""

    return tuple(index for index in range(source_neurons) if index != removed_neuron)


def sparse_compensation_terms(
    anchor_neurons: tuple[int, ...],
    anchor_target_indices: tuple[int, ...],
    compensation_coefficients: tuple[float, ...],
) -> tuple[tuple[int, int, float], ...]:
    """Zip one selection's sparse compensation into ``(source, target, beta)``."""

    return tuple(
        zip(
            anchor_neurons,
            anchor_target_indices,
            compensation_coefficients,
            strict=True,
        )
    )


def normalize_sparse_anchor_selection(
    selection: Any,
    *,
    vocabulary: ModalityFoldSelectionVocabulary,
) -> None:
    """Validate and freeze one lane selection's sparse compensation in place.

    This is the shared body of every lane's ``__post_init__``.  It mutates the
    frozen dataclass through :func:`object.__setattr__` exactly as the three
    inline copies did, so the normalized field values -- and therefore the
    sealed ``as_dict()`` payload -- are unchanged.
    """

    error_type = vocabulary.error_type
    subject = vocabulary.subject_label
    nonnegative_int = vocabulary.nonnegative_int
    positive_int = vocabulary.positive_int
    finite = vocabulary.finite

    object.__setattr__(
        selection,
        "layer_index",
        nonnegative_int(selection.layer_index, "layer_index"),
    )
    for name in (
        "source_neurons",
        "target_neurons",
        "trace_rows",
        "candidate_count",
    ):
        object.__setattr__(
            selection,
            name,
            positive_int(getattr(selection, name), name),
        )
    if selection.source_neurons != selection.target_neurons + 1:
        raise error_type(f"{subject} selection must remove exactly one source neuron.")
    for name in ("removed_neuron", "anchor_neuron"):
        value = nonnegative_int(getattr(selection, name), name)
        if value >= selection.source_neurons:
            raise error_type(f"{name} is out of range.")
        object.__setattr__(selection, name, value)
    if selection.removed_neuron == selection.anchor_neuron:
        raise error_type("Removed and anchor neurons must differ.")
    anchor_target_index = nonnegative_int(
        selection.anchor_target_index,
        "anchor_target_index",
    )
    if anchor_target_index >= selection.target_neurons:
        raise error_type("anchor_target_index is out of range.")
    object.__setattr__(selection, "anchor_target_index", anchor_target_index)
    for name in (
        "compensation_beta",
        "activation_correlation",
        "relative_reconstruction_error",
        "residual_endpoint_relative_rmse",
        "removed_importance",
    ):
        object.__setattr__(selection, name, finite(getattr(selection, name), name))
    anchor_neurons = tuple(selection.anchor_neurons) or (selection.anchor_neuron,)
    anchor_target_indices = tuple(selection.anchor_target_indices) or (
        selection.anchor_target_index,
    )
    compensation_coefficients = tuple(selection.compensation_coefficients) or (
        selection.compensation_beta,
    )
    if not (
        len(anchor_neurons)
        == len(anchor_target_indices)
        == len(compensation_coefficients)
    ):
        raise error_type(f"{subject} sparse anchor and compensation lengths differ.")
    if len(set(anchor_neurons)) != len(anchor_neurons):
        raise error_type(f"{subject} sparse anchor neurons repeat.")
    survivors = selection.survivors
    normalized_anchors = []
    normalized_target_indices = []
    normalized_coefficients = []
    for anchor, target_index, coefficient in zip(
        anchor_neurons,
        anchor_target_indices,
        compensation_coefficients,
        strict=True,
    ):
        normalized_anchor = nonnegative_int(anchor, "anchor_neurons")
        if (
            normalized_anchor >= selection.source_neurons
            or normalized_anchor == selection.removed_neuron
        ):
            raise error_type(f"{subject} sparse anchor neuron is out of range.")
        normalized_target = nonnegative_int(
            target_index,
            "anchor_target_indices",
        )
        if (
            normalized_target >= selection.target_neurons
            or survivors[normalized_target] != normalized_anchor
        ):
            raise error_type(f"{subject} sparse anchor target index is inconsistent.")
        normalized_anchors.append(normalized_anchor)
        normalized_target_indices.append(normalized_target)
        normalized_coefficients.append(finite(coefficient, "compensation_coefficients"))
    if (
        normalized_anchors[0] != selection.anchor_neuron
        or normalized_target_indices[0] != selection.anchor_target_index
        or normalized_coefficients[0] != selection.compensation_beta
    ):
        raise error_type(
            f"{subject} primary anchor fields differ from sparse compensation."
        )
    object.__setattr__(selection, "anchor_neurons", tuple(normalized_anchors))
    object.__setattr__(
        selection,
        "anchor_target_indices",
        tuple(normalized_target_indices),
    )
    object.__setattr__(
        selection,
        "compensation_coefficients",
        tuple(normalized_coefficients),
    )


def sparse_anchor_regression(
    sketch: Any,
    *,
    removed: int,
    anchor_count: int,
    column_norms: Any,
    epsilon: Any,
    vocabulary: ModalityFoldSelectionVocabulary,
) -> tuple[tuple[int, ...], tuple[float, ...], float, float]:
    """Fit one removed activation from a small survivor anchor set."""

    import torch

    removed_vector = sketch[:, removed]
    removed_norm = column_norms[removed]
    available = tuple(
        index for index in range(int(sketch.shape[1])) if index != removed
    )
    count = min(
        vocabulary.positive_int(anchor_count, "anchor_count"),
        len(available),
        int(sketch.shape[0]),
    )
    if float(removed_norm) <= 0.0:
        anchors = available[:count]
        return anchors, tuple(0.0 for _ in anchors), 0.0, 0.0

    correlations = (sketch.T @ removed_vector) / (
        column_norms * removed_norm
    ).clamp_min(epsilon)
    if count == 1:
        absolute = correlations.abs()
        absolute[removed] = -1.0
        anchor = int(torch.argmax(absolute).item())
        anchor_vector = sketch[:, anchor]
        denominator = torch.dot(anchor_vector, anchor_vector).clamp_min(epsilon)
        beta = max(
            -4.0,
            min(
                4.0,
                float((torch.dot(anchor_vector, removed_vector) / denominator).item()),
            ),
        )
        residual = removed_vector - beta * anchor_vector
        relative_error = float(
            (
                torch.linalg.vector_norm(residual) / removed_norm.clamp_min(epsilon)
            ).item()
        )
        return (anchor,), (beta,), float(correlations[anchor].item()), (relative_error)

    anchors: list[int] = []
    coefficients = torch.empty(0, dtype=sketch.dtype, device=sketch.device)
    residual = removed_vector.clone()
    for _index in range(count):
        residual_norm = torch.linalg.vector_norm(residual)
        residual_correlations = (sketch.T @ residual) / (
            column_norms * residual_norm
        ).clamp_min(epsilon)
        scores = residual_correlations.abs()
        scores[removed] = -1.0
        for selected in anchors:
            scores[selected] = -1.0
        anchor = int(torch.argmax(scores).item())
        anchors.append(anchor)
        design = sketch[:, anchors]
        gram = design.T @ design
        scale = torch.trace(gram) / len(anchors)
        ridge = scale.abs().clamp_min(epsilon) * 1.0e-6
        coefficients = torch.linalg.solve(
            gram
            + torch.eye(
                len(anchors),
                dtype=gram.dtype,
                device=gram.device,
            )
            * ridge,
            design.T @ removed_vector,
        ).clamp(min=-4.0, max=4.0)
        residual = removed_vector - design @ coefficients
    relative_error = float(
        (torch.linalg.vector_norm(residual) / removed_norm.clamp_min(epsilon)).item()
    )
    return (
        tuple(anchors),
        tuple(float(value.item()) for value in coefficients),
        float(correlations[anchors[0]].item()),
        relative_error,
    )


def selection_from_statistics(
    layer_index: int,
    statistics: Mapping[str, Any],
    *,
    trace_rows: int,
    candidate_count: int,
    anchor_count: int = 1,
    vocabulary: ModalityFoldSelectionVocabulary,
    selection_factory: Callable[..., Any],
) -> Any:
    """Choose the least-damaging removable neuron from sealed lane evidence."""

    error_type = vocabulary.error_type
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is an extra
        raise error_type(
            f"PyTorch is required for {vocabulary.selection_label} selection."
        ) from exc
    mean_abs = statistics["mean_abs"].to(dtype=torch.float64)
    mean_square = statistics["mean_square"].to(dtype=torch.float64)
    sketch = statistics["sketch"].to(dtype=torch.float64)
    down_norm = statistics["down_norm"].to(dtype=torch.float64)
    residual_endpoint_energy = statistics["residual_endpoint_mean_square_l2"].to(
        dtype=torch.float64
    )
    if (
        mean_abs.ndim != 1
        or mean_square.shape != mean_abs.shape
        or down_norm.shape != mean_abs.shape
        or sketch.ndim != 2
        or sketch.shape[1] != mean_abs.shape[0]
        or residual_endpoint_energy.numel() != 1
        or not all(
            bool(torch.isfinite(value).all())
            for value in (
                mean_abs,
                mean_square,
                sketch,
                down_norm,
                residual_endpoint_energy,
            )
        )
        or float(residual_endpoint_energy.item()) <= 0.0
    ):
        raise error_type(
            f"{vocabulary.subject_label} layer {layer_index} trace statistics "
            "are malformed."
        )
    source_neurons = int(mean_abs.shape[0])
    if source_neurons <= 1:
        raise error_type(
            f"{vocabulary.subject_label} trace width must exceed one neuron."
        )
    selected_candidates = min(
        vocabulary.positive_int(candidate_count, "candidate_count"),
        source_neurons - 1,
    )
    selected_anchors = min(
        vocabulary.positive_int(anchor_count, "anchor_count"),
        source_neurons - 1,
        int(sketch.shape[0]),
    )
    importance = mean_abs * down_norm
    candidates = torch.argsort(importance, stable=True)[:selected_candidates]
    column_norms = torch.linalg.vector_norm(sketch, dim=0)
    choices = []
    epsilon = torch.finfo(torch.float64).eps
    for candidate_tensor in candidates:
        removed = int(candidate_tensor.item())
        anchors, coefficients, correlation, relative_error = sparse_anchor_regression(
            sketch,
            removed=removed,
            anchor_count=selected_anchors,
            column_norms=column_norms,
            epsilon=epsilon,
            vocabulary=vocabulary,
        )
        endpoint_relative_rmse = (
            relative_error
            * math.sqrt(max(0.0, float(mean_square[removed].item())))
            * float(down_norm[removed].item())
            / math.sqrt(float(residual_endpoint_energy.item()))
        )
        score = (
            float(importance[removed].item())
            * (1.0 + relative_error)
            * (1.0 + endpoint_relative_rmse)
        )
        choices.append(
            (
                score,
                removed,
                anchors,
                coefficients,
                correlation,
                relative_error,
                endpoint_relative_rmse,
            )
        )
    (
        _score,
        removed,
        anchors,
        coefficients,
        correlation,
        relative_error,
        endpoint_relative_rmse,
    ) = min(choices)
    survivors = survivors_for(source_neurons, removed)
    anchor_target_indices = tuple(survivors.index(anchor) for anchor in anchors)
    return selection_factory(
        layer_index=layer_index,
        source_neurons=source_neurons,
        target_neurons=source_neurons - 1,
        removed_neuron=removed,
        anchor_neuron=anchors[0],
        anchor_target_index=anchor_target_indices[0],
        compensation_beta=coefficients[0],
        activation_correlation=correlation,
        relative_reconstruction_error=relative_error,
        residual_endpoint_relative_rmse=endpoint_relative_rmse,
        removed_importance=float(importance[removed].item()),
        trace_rows=trace_rows,
        candidate_count=selected_candidates,
        anchor_neurons=anchors,
        anchor_target_indices=anchor_target_indices,
        compensation_coefficients=coefficients,
    )
