"""Experimental FIT-only verbatim-channel selection; no production fitter wiring.

Fixed recipe: pivot the FP64 centered feature Gram weighted by column norms of
the native BF16 parent readout. Return source-order indices, never new features.
This heuristic is not evidence of an accepted QIMG fold or retained capability.
"""
from __future__ import annotations

import torch

from .linalg import pivoted_columns_gram


RECIPE_ID = "fit-centered-native-bf16-parent-output-norm-v1"
DEFAULT_CHUNK_ROWS = 512
_PIVOT_ABSOLUTE_FLOOR = 1.0e-24  # The existing pivot primitive's stopping floor.


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _cpu_matrix(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.device.type != "cpu" or value.layout != torch.strided:
        raise ValueError(f"{name} must be a strided CPU tensor")
    if value.ndim != 2 or min(value.shape) <= 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if value.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise ValueError(f"{name} must contain FP16, BF16, FP32 or FP64 values")


@torch.no_grad()
def select_fit_output_weighted_columns(
    fit_features: torch.Tensor,
    native_parent_w2: torch.Tensor,
    keep: int,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> torch.Tensor:
    """Return sorted, unique CPU-long indices into unchanged source channels.

    ``fit_features`` is [FIT rows, channels]. Its caller must establish authentic,
    replayed, FIT-only source activations; this function cannot establish split or
    provenance from a tensor. ``native_parent_w2`` is [outputs, channels] and must
    already be the source runtime's BF16 readout, not widened FP8 coefficients.

    ``chunk_rows`` bounds intermediate row matrices; it is not a recipe option.
    No normalization, target fitting, held-out scoring, channel mixing, gradient,
    model/capture loading, artifact writing, or numerical admission occurs here.
    """
    _positive_integer(keep, "keep")
    _positive_integer(chunk_rows, "chunk_rows")
    _cpu_matrix(fit_features, "fit_features")
    _cpu_matrix(native_parent_w2, "native_parent_w2")
    if native_parent_w2.dtype != torch.bfloat16:
        raise ValueError("native_parent_w2 must already have native BF16 dtype")
    rows, channels = fit_features.shape
    if native_parent_w2.shape[1] != channels:
        raise ValueError("parent readout channel count does not match FIT features")
    if keep > channels:
        raise ValueError("keep exceeds the source channel count")
    if keep > rows - 1:
        raise ValueError("insufficient centered FIT rank for keep")

    # Translate before computing the mean. Summing a large constant FP64 value
    # can otherwise round its mean and manufacture a nonzero centered column.
    # Two passes also avoid subtracting H.T @ H - N * mean * mean.T. Every block
    # subtraction allocates even for FP64 input, preserving caller-owned storage.
    origin = fit_features.detach()[0].to(torch.float64)
    if not bool(torch.isfinite(origin).all()):
        raise ValueError("non-finite FIT feature")
    total = torch.zeros(channels, dtype=torch.float64)
    for block in fit_features.detach().split(chunk_rows):
        if not bool(torch.isfinite(block).all()):
            raise ValueError("non-finite FIT feature")
        translated = block.to(torch.float64) - origin
        if not bool(torch.isfinite(translated).all()):
            raise ValueError("non-finite translated FIT feature")
        total.add_(translated.sum(dim=0))
    mean_offset = total / rows
    if not bool(torch.isfinite(mean_offset).all()):
        raise ValueError("non-finite FIT feature mean offset")
    gram = torch.zeros((channels, channels), dtype=torch.float64)
    for block in fit_features.detach().split(chunk_rows):
        centered = (block.to(torch.float64) - origin) - mean_offset
        gram.add_(centered.T @ centered)
    if not bool(torch.isfinite(gram).all()):
        raise ValueError("non-finite centered FIT Gram")

    norm_squared = torch.zeros(channels, dtype=torch.float64)
    for block in native_parent_w2.detach().split(chunk_rows):
        if not bool(torch.isfinite(block).all()):
            raise ValueError("non-finite native BF16 parent readout")
        norm_squared.add_(block.to(torch.float64).square().sum(dim=0))
    norms = norm_squared.sqrt()
    gram.mul_(norms[:, None]).mul_(norms[None, :])
    if not bool(torch.isfinite(gram).all()):
        raise ValueError("non-finite output-weighted FIT Gram")

    selection = pivoted_columns_gram(gram, keep)
    if selection.numel() != keep or selection.unique().numel() != keep:
        raise ValueError("insufficient output-weighted centered FIT rank for keep")

    # The primitive can continue on roundoff residues above its absolute floor.
    # Certify the chosen columns without changing their selection or mixing them.
    selected_gram = gram.index_select(0, selection).index_select(1, selection)
    factor, info = torch.linalg.cholesky_ex(selected_gram, check_errors=False)
    pivot_floor = max(
        _PIVOT_ABSOLUTE_FLOOR,
        torch.finfo(torch.float64).eps * max(rows, channels)
        * float(torch.diagonal(gram).max()),
    )
    if int(info) != 0 or not bool(torch.isfinite(factor).all()) or bool(
        (torch.diagonal(factor).square() <= pivot_floor).any()
    ):
        raise ValueError("insufficient numerically independent FIT rank for keep")
    return selection.sort().values
