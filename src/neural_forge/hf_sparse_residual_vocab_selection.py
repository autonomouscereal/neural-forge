"""Calibration-only token-row selection and logit-margin validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from neural_forge.hf_sparse_residual_vocab import (
    SparseResidualVocabularyCore,
)


class SparseResidualSelectionError(ValueError):
    """Calibration evidence, split, token priority, or metric is invalid."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).reshape(-1).numpy().tobytes())
    return digest.hexdigest()


def _token_tuple(
    value: Any,
    *,
    vocab_size: int,
    name: str,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SparseResidualSelectionError(f"{name} must be a token-ID sequence.")
    result = tuple(value)
    if any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or not 0 <= token < vocab_size
        for token in result
    ):
        raise SparseResidualSelectionError(
            f"{name} contains invalid vocabulary IDs."
        )
    return result


@dataclass(frozen=True, slots=True)
class VocabularyCalibrationCase:
    """One recovery-only case with cached LM-head inputs and parent logits."""

    case_id: str
    category: str
    prompt_token_ids: tuple[int, ...]
    parent_output_token_ids: tuple[int, ...]
    hidden_rows: Tensor
    parent_logits: Tensor

    @classmethod
    def create(
        cls,
        *,
        case_id: str,
        category: str,
        prompt_token_ids: Sequence[int],
        parent_output_token_ids: Sequence[int],
        hidden_rows: Tensor,
        parent_logits: Tensor,
        vocab_size: int,
        hidden_size: int,
    ) -> VocabularyCalibrationCase:
        if (
            not isinstance(case_id, str)
            or not case_id.startswith("recovery_")
            or not isinstance(category, str)
            or category not in {"math", "code"}
        ):
            raise SparseResidualSelectionError(
                "Calibration cases must be recovery math/code cases."
            )
        prompt = _token_tuple(
            prompt_token_ids,
            vocab_size=vocab_size,
            name=f"{case_id}.prompt_token_ids",
        )
        output = _token_tuple(
            parent_output_token_ids,
            vocab_size=vocab_size,
            name=f"{case_id}.parent_output_token_ids",
        )
        if (
            not isinstance(hidden_rows, Tensor)
            or not isinstance(parent_logits, Tensor)
            or hidden_rows.ndim != 2
            or parent_logits.ndim != 2
            or int(hidden_rows.shape[0]) != int(parent_logits.shape[0])
            or int(hidden_rows.shape[1]) != hidden_size
            or int(parent_logits.shape[1]) != vocab_size
            or hidden_rows.shape[0] <= 0
            or not torch.is_floating_point(hidden_rows)
            or not torch.is_floating_point(parent_logits)
            or not bool(torch.isfinite(hidden_rows).all())
            or not bool(torch.isfinite(parent_logits).all())
        ):
            raise SparseResidualSelectionError(
                "Calibration hidden/logit rows are non-finite or shape-incompatible."
            )
        if not prompt or len(output) != int(hidden_rows.shape[0]):
            raise SparseResidualSelectionError(
                "Calibration requires a nonempty prompt and one generated token "
                "ID for every hidden/logit row."
            )
        return cls(
            case_id=case_id,
            category=category,
            prompt_token_ids=prompt,
            parent_output_token_ids=output,
            hidden_rows=hidden_rows.detach().to(device="cpu").contiguous(),
            parent_logits=parent_logits.detach().to(device="cpu").contiguous(),
        )

    def descriptor(self) -> dict[str, Any]:
        parent_winners = tuple(
            int(value)
            for value in self.parent_logits.detach()
            .argmax(dim=-1)
            .to(device="cpu")
            .tolist()
        )
        agreement_count = sum(
            int(forced == winner)
            for forced, winner in zip(
                self.parent_output_token_ids,
                parent_winners,
                strict=True,
            )
        )
        return {
            "case_id": self.case_id,
            "category": self.category,
            "prompt_token_ids_sha256": _canonical_sha256(
                list(self.prompt_token_ids)
            ),
            "parent_output_token_ids_sha256": _canonical_sha256(
                list(self.parent_output_token_ids)
            ),
            "hidden_rows_shape": list(self.hidden_rows.shape),
            "hidden_rows_sha256": _tensor_sha256(self.hidden_rows),
            "parent_logits_shape": list(self.parent_logits.shape),
            "parent_logits_sha256": _tensor_sha256(self.parent_logits),
            "parent_top1_token_ids_sha256": _canonical_sha256(
                list(parent_winners)
            ),
            "forced_vs_parent_top1_agreement_fraction": (
                agreement_count / len(parent_winners)
            ),
            "teacher_forced_rows_allowed": True,
        }


@dataclass(frozen=True, slots=True)
class CalibrationCaseSplit:
    """Deterministic category-stratified fit/validation case IDs."""

    fit_case_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]
    split_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "fit_case_ids": list(self.fit_case_ids),
            "validation_case_ids": list(self.validation_case_ids),
            "fit_count": len(self.fit_case_ids),
            "validation_count": len(self.validation_case_ids),
            "case_disjoint": not bool(
                set(self.fit_case_ids).intersection(self.validation_case_ids)
            ),
            "split_sha256": self.split_sha256,
        }


def split_recovery_calibration_cases(
    cases: Sequence[VocabularyCalibrationCase],
    *,
    validation_per_category: int = 2,
) -> CalibrationCaseSplit:
    """Reserve exact math/code cases for validation before token selection."""

    if (
        isinstance(cases, (str, bytes))
        or not isinstance(cases, Sequence)
        or len(cases) != 16
        or any(not isinstance(case, VocabularyCalibrationCase) for case in cases)
    ):
        raise SparseResidualSelectionError(
            "Exactly 16 authenticated recovery calibration cases are required."
        )
    if (
        isinstance(validation_per_category, bool)
        or not isinstance(validation_per_category, int)
        or validation_per_category <= 0
    ):
        raise SparseResidualSelectionError(
            "validation_per_category must be a positive integer."
        )
    ids = tuple(case.case_id for case in cases)
    if len(set(ids)) != len(ids):
        raise SparseResidualSelectionError("Calibration case IDs must be unique.")
    by_category: dict[str, list[str]] = {"math": [], "code": []}
    for case in cases:
        by_category[case.category].append(case.case_id)
    if any(len(values) != 8 for values in by_category.values()):
        raise SparseResidualSelectionError(
            "Calibration requires exactly eight recovery math and eight code cases."
        )
    validation: list[str] = []
    for category, values in sorted(by_category.items()):
        ordered = sorted(
            values,
            key=lambda case_id: hashlib.sha256(
                f"lfm2-sparse-vocab-validation-v1\0{category}\0{case_id}".encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
        validation.extend(ordered[:validation_per_category])
    validation_set = set(validation)
    fit = tuple(case_id for case_id in ids if case_id not in validation_set)
    validation_ordered = tuple(case_id for case_id in ids if case_id in validation_set)
    body = {
        "fit_case_ids": list(fit),
        "validation_case_ids": list(validation_ordered),
        "fit_count": len(fit),
        "validation_count": len(validation_ordered),
        "case_disjoint": True,
    }
    return CalibrationCaseSplit(
        fit_case_ids=fit,
        validation_case_ids=validation_ordered,
        split_sha256=_canonical_sha256(body),
    )


def _case_subset(
    cases: Sequence[VocabularyCalibrationCase],
    ids: Sequence[str],
) -> tuple[VocabularyCalibrationCase, ...]:
    by_id = {case.case_id: case for case in cases}
    try:
        result = tuple(by_id[case_id] for case_id in ids)
    except KeyError as exc:
        raise SparseResidualSelectionError(
            "Split references an unknown calibration case."
        ) from exc
    return result


def _require_exact_split(
    cases: Sequence[VocabularyCalibrationCase],
    split: CalibrationCaseSplit,
) -> None:
    if not isinstance(split, CalibrationCaseSplit):
        raise SparseResidualSelectionError(
            "split must be a CalibrationCaseSplit."
        )
    expected = split_recovery_calibration_cases(
        cases,
        validation_per_category=2,
    )
    if split != expected:
        raise SparseResidualSelectionError(
            "Calibration split was modified or is not the exact 12-fit/4-validation "
            "category-stratified split."
        )


def residual_output_energy(
    *,
    source_weight: Tensor,
    vocab_factor: Tensor,
    hidden_factor: Tensor,
    hidden_rows: Tensor,
    chunk_rows: int = 2048,
) -> Tensor:
    """Compute actual mean squared logit residual energy for every token row."""

    if (
        not isinstance(source_weight, Tensor)
        or not isinstance(vocab_factor, Tensor)
        or not isinstance(hidden_factor, Tensor)
        or not isinstance(hidden_rows, Tensor)
        or source_weight.ndim != 2
        or vocab_factor.ndim != 2
        or hidden_factor.ndim != 2
        or hidden_rows.ndim != 2
        or tuple(source_weight.shape)
        != (int(vocab_factor.shape[0]), int(hidden_factor.shape[1]))
        or int(vocab_factor.shape[1]) != int(hidden_factor.shape[0])
        or int(hidden_rows.shape[1]) != int(source_weight.shape[1])
    ):
        raise SparseResidualSelectionError(
            "Residual-energy tensors are shape-incompatible."
        )
    if (
        isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_rows <= 0
    ):
        raise SparseResidualSelectionError("chunk_rows must be positive.")
    if (
        source_weight.device != vocab_factor.device
        or vocab_factor.device != hidden_factor.device
    ):
        raise SparseResidualSelectionError(
            "Source weight and base factors must share one compute device."
        )
    device = vocab_factor.device
    right = hidden_factor.detach().to(device=device, dtype=torch.float32)
    hidden = hidden_rows.detach().to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(right).all()) or not bool(
        torch.isfinite(hidden).all()
    ):
        raise SparseResidualSelectionError(
            "Residual-energy tensors contain NaN or infinity."
        )
    result = torch.empty(int(source_weight.shape[0]), dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, int(source_weight.shape[0]), chunk_rows):
            stop = min(int(source_weight.shape[0]), start + chunk_rows)
            source = source_weight[start:stop].detach().to(
                device=device,
                dtype=torch.float32,
            )
            left = vocab_factor[start:stop].detach().to(
                device=device,
                dtype=torch.float32,
            )
            if not bool(torch.isfinite(source).all()) or not bool(
                torch.isfinite(left).all()
            ):
                raise SparseResidualSelectionError(
                    "Residual-energy tensors contain NaN or infinity."
                )
            residual = source - left @ right
            logits = hidden @ residual.transpose(0, 1)
            result[start:stop] = (
                logits.square().mean(dim=0).to(device="cpu", dtype=torch.float64)
            )
    return result


@dataclass(frozen=True, slots=True)
class SelectedTokenRows:
    """Sorted physical IDs plus auditable priority order and fit-only evidence."""

    selected_token_ids: tuple[int, ...]
    priority_order: tuple[int, ...]
    mandatory_token_ids: tuple[int, ...]
    fit_case_ids: tuple[str, ...]
    token_diagnostics: tuple[dict[str, Any], ...]
    selection_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_token_ids": list(self.selected_token_ids),
            "selected_row_count": len(self.selected_token_ids),
            "priority_order": list(self.priority_order),
            "mandatory_token_ids": list(self.mandatory_token_ids),
            "fit_case_ids": list(self.fit_case_ids),
            "token_diagnostics": [dict(value) for value in self.token_diagnostics],
            "selection_sha256": self.selection_sha256,
        }


def select_sparse_residual_token_rows(
    cases: Sequence[VocabularyCalibrationCase],
    split: CalibrationCaseSplit,
    *,
    selected_row_budget: int,
    special_token_ids: Sequence[int],
    residual_energy: Tensor,
    top_k: int = 16,
) -> SelectedTokenRows:
    """Select rows from fit cases only; validation cases never influence S."""

    _require_exact_split(cases, split)
    if (
        isinstance(selected_row_budget, bool)
        or not isinstance(selected_row_budget, int)
        or selected_row_budget <= 0
    ):
        raise SparseResidualSelectionError(
            "selected_row_budget must be positive."
        )
    if (
        not isinstance(residual_energy, Tensor)
        or residual_energy.ndim != 1
        or residual_energy.numel() <= 0
        or not bool(torch.isfinite(residual_energy).all())
        or bool((residual_energy < 0).any())
    ):
        raise SparseResidualSelectionError(
            "residual_energy must be a finite nonnegative vocabulary vector."
        )
    vocab_size = int(residual_energy.numel())
    if selected_row_budget > vocab_size:
        raise SparseResidualSelectionError(
            "selected_row_budget exceeds the vocabulary."
        )
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 2 <= top_k <= vocab_size
    ):
        raise SparseResidualSelectionError("top_k is invalid.")
    special = _token_tuple(
        special_token_ids,
        vocab_size=vocab_size,
        name="special_token_ids",
    )
    fit_cases = _case_subset(cases, split.fit_case_ids)
    mandatory = set(special)
    prompt_count: Counter[int] = Counter()
    generated_count: Counter[int] = Counter()
    top1_count: Counter[int] = Counter()
    topk_count: Counter[int] = Counter()
    small_margin_score: Counter[int] = Counter()
    for case in fit_cases:
        mandatory.update(case.prompt_token_ids)
        prompt_count.update(case.prompt_token_ids)
        generated_count.update(case.parent_output_token_ids)
        logits = case.parent_logits.to(dtype=torch.float64)
        values, indices = torch.topk(logits, k=top_k, dim=-1)
        for row_values, row_indices in zip(values, indices):
            winner = int(row_indices[0].item())
            top1_count[winner] += 1
            margin = max(
                float(row_values[0].item() - row_values[1].item()),
                1e-12,
            )
            margin_weight = 1.0 / margin
            for position, token in enumerate(row_indices.tolist()):
                token_id = int(token)
                topk_count[token_id] += 1
                if position > 0:
                    small_margin_score[token_id] += margin_weight / position
    if len(mandatory) > selected_row_budget:
        raise SparseResidualSelectionError(
            "The residual-row budget cannot include every special and prompt token."
        )
    energy = residual_energy.detach().to(device="cpu", dtype=torch.float64)
    maximum_energy = max(float(energy.max().item()), 1e-30)
    maximum_margin = max(
        (float(value) for value in small_margin_score.values()),
        default=1.0,
    )
    maximum_occurrence = max(
        (
            prompt_count[token] + generated_count[token]
            for token in set(prompt_count) | set(generated_count)
        ),
        default=1,
    )
    diagnostics: dict[int, dict[str, Any]] = {}
    priority_keys: dict[int, tuple[Any, ...]] = {}
    for token_id in range(vocab_size):
        observed = prompt_count[token_id] + generated_count[token_id]
        energy_fraction = float(energy[token_id].item()) / maximum_energy
        score = (
            1000.0 * float(generated_count[token_id] > 0)
            + 100.0 * top1_count[token_id]
            + 10.0 * topk_count[token_id]
            + 5.0
            * (float(small_margin_score[token_id]) / maximum_margin)
            + 2.0 * (observed / maximum_occurrence)
            + energy_fraction
        )
        diagnostics[token_id] = {
            "token_id": token_id,
            "mandatory": token_id in mandatory,
            "prompt_occurrences": prompt_count[token_id],
            "parent_generated_occurrences": generated_count[token_id],
            "parent_top1_occurrences": top1_count[token_id],
            "parent_topk_occurrences": topk_count[token_id],
            "small_margin_competitor_score": float(
                small_margin_score[token_id]
            ),
            "residual_output_energy": float(energy[token_id].item()),
            "priority_score": score,
        }
        priority_keys[token_id] = (
            0 if token_id in mandatory else 1,
            -score,
            -energy_fraction,
            token_id,
        )
    ordered = tuple(sorted(range(vocab_size), key=priority_keys.__getitem__))
    priority = ordered[:selected_row_budget]
    selected = tuple(sorted(priority))
    selected_diagnostics = tuple(diagnostics[token] for token in priority)
    body = {
        "selected_token_ids": list(selected),
        "selected_row_count": len(selected),
        "priority_order": list(priority),
        "mandatory_token_ids": sorted(mandatory),
        "fit_case_ids": list(split.fit_case_ids),
        "token_diagnostics": list(selected_diagnostics),
    }
    return SelectedTokenRows(
        selected_token_ids=selected,
        priority_order=priority,
        mandatory_token_ids=tuple(sorted(mandatory)),
        fit_case_ids=split.fit_case_ids,
        token_diagnostics=selected_diagnostics,
        selection_sha256=_canonical_sha256(body),
    )


def _flatten_validation(
    cases: Sequence[VocabularyCalibrationCase],
    split: CalibrationCaseSplit,
) -> tuple[Tensor, Tensor]:
    _require_exact_split(cases, split)
    validation = _case_subset(cases, split.validation_case_ids)
    hidden = torch.cat([case.hidden_rows for case in validation], dim=0)
    logits = torch.cat([case.parent_logits for case in validation], dim=0)
    return hidden, logits


def evaluate_sparse_residual_logit_validation(
    core: SparseResidualVocabularyCore,
    cases: Sequence[VocabularyCalibrationCase],
    split: CalibrationCaseSplit,
    *,
    top_k: int = 16,
) -> dict[str, Any]:
    """Measure disjoint validation top-1, margins, top-k overlap, and worst case."""

    if not isinstance(core, SparseResidualVocabularyCore):
        raise SparseResidualSelectionError(
            "core must be a SparseResidualVocabularyCore."
        )
    hidden, parent = _flatten_validation(cases, split)
    if int(hidden.shape[1]) != core.hidden_size or int(parent.shape[1]) != (
        core.vocab_size
    ):
        raise SparseResidualSelectionError(
            "Validation rows differ from the sparse vocabulary core."
        )
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 2 <= top_k <= core.vocab_size
    ):
        raise SparseResidualSelectionError("top_k is invalid.")
    device = core.vocab_factor.device
    dtype = core.vocab_factor.dtype
    with torch.no_grad():
        candidate = core.logits(hidden.to(device=device, dtype=dtype)).to(
            device="cpu",
            dtype=torch.float64,
        )
    parent = parent.to(dtype=torch.float64)
    parent_values, parent_indices = torch.topk(parent, k=top_k, dim=-1)
    candidate_values, candidate_indices = torch.topk(
        candidate,
        k=top_k,
        dim=-1,
    )
    parent_winners = parent_indices[:, 0]
    candidate_winners = candidate_indices[:, 0]
    top1 = parent_winners.eq(candidate_winners)
    row_indices = torch.arange(int(parent.shape[0]))
    parent_winner_candidate_logits = candidate[row_indices, parent_winners]
    masked = candidate.clone()
    masked[row_indices, parent_winners] = -torch.inf
    best_competitor = masked.max(dim=-1).values
    candidate_parent_winner_margin = (
        parent_winner_candidate_logits - best_competitor
    )
    parent_margin = parent_values[:, 0] - parent_values[:, 1]
    margin_delta = candidate_parent_winner_margin - parent_margin
    overlap = []
    for left, right in zip(parent_indices.tolist(), candidate_indices.tolist()):
        overlap.append(len(set(left).intersection(right)) / top_k)
    parent_topk_candidate = candidate.gather(1, parent_indices)
    topk_error = (parent_topk_candidate - parent_values).abs()
    finite_values = (
        candidate_parent_winner_margin,
        margin_delta,
        topk_error,
    )
    if not all(bool(torch.isfinite(value).all()) for value in finite_values):
        raise SparseResidualSelectionError(
            "Validation metrics contain NaN or infinity."
        )
    body = {
        "fit_case_ids": list(split.fit_case_ids),
        "validation_case_ids": list(split.validation_case_ids),
        "case_disjoint": True,
        "validation_row_count": int(parent.shape[0]),
        "top1_agreement": float(top1.to(torch.float64).mean().item()),
        "parent_winner_margin_positive_fraction": float(
            (candidate_parent_winner_margin > 0)
            .to(torch.float64)
            .mean()
            .item()
        ),
        "mean_parent_margin": float(parent_margin.mean().item()),
        "mean_candidate_parent_winner_margin": float(
            candidate_parent_winner_margin.mean().item()
        ),
        "minimum_candidate_parent_winner_margin": float(
            candidate_parent_winner_margin.min().item()
        ),
        "worst_case_margin_delta": float(margin_delta.min().item()),
        "mean_margin_delta": float(margin_delta.mean().item()),
        "mean_topk_overlap": float(sum(overlap) / len(overlap)),
        "mean_parent_topk_absolute_logit_error": float(topk_error.mean().item()),
        "worst_parent_topk_absolute_logit_error": float(topk_error.max().item()),
        "selection_not_fit_on_validation": True,
    }
    return {**body, "validation_sha256": _canonical_sha256(body)}


def logit_validation_sort_key(report: dict[str, Any]) -> tuple[float, ...]:
    """Higher is better; use after every report is independently sealed."""

    required = (
        "top1_agreement",
        "parent_winner_margin_positive_fraction",
        "minimum_candidate_parent_winner_margin",
        "worst_case_margin_delta",
        "mean_topk_overlap",
        "worst_parent_topk_absolute_logit_error",
    )
    if not isinstance(report, dict) or any(name not in report for name in required):
        raise SparseResidualSelectionError(
            "Validation report is missing ranking metrics."
        )
    values = tuple(float(report[name]) for name in required)
    if not all(math.isfinite(value) for value in values):
        raise SparseResidualSelectionError(
            "Validation ranking metrics are non-finite."
        )
    return (
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        -values[5],
    )


__all__ = [
    "CalibrationCaseSplit",
    "SelectedTokenRows",
    "SparseResidualSelectionError",
    "VocabularyCalibrationCase",
    "evaluate_sparse_residual_logit_validation",
    "logit_validation_sort_key",
    "residual_output_energy",
    "select_sparse_residual_token_rows",
    "split_recovery_calibration_cases",
]
