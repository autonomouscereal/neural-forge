"""Closed-form teacher/child logit diagnostics for Hugging Face models.

The tensor functions in this module do not mutate their inputs.  The Hugging
Face collector runs under ``torch.inference_mode()`` and only retains bounded,
detached CPU logits; it performs no backward pass or optimizer step.
"""

from __future__ import annotations

import json
import math
from hashlib import sha256
from itertools import islice
from typing import Any, Iterable, Mapping, Sequence


REPORT_SCHEMA_VERSION = 1
BLANK_BASELINE_KIND = "neural_forge_blank_logit_baseline"

__all__ = [
    "aggregate_by_domain",
    "align_vocabulary_logits",
    "build_blank_baseline_cache",
    "build_logit_diagnostics_report",
    "calibration_hashes",
    "centered_logit_cosine",
    "collect_hf_calibration",
    "collect_hf_calibration_logits",
    "collect_hf_logit_diagnostics",
    "compute_logit_diagnostics",
    "kl_teacher_child",
    "normalized_kl_retention",
    "paired_bootstrap_lower_bound",
    "paired_bootstrap_report",
    "teacher_child_kl",
    "teacher_margin_sign_retention",
    "teacher_margin_values",
    "teacher_top1_in_child_topk",
    "top1_agreement",
    "top_k_agreement",
    "topk_agreement",
]


def _torch():
    import torch

    return torch


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _as_tensor(value: Any, name: str):
    torch = _torch()
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension.")
    return tensor


def _floating_pair(teacher_logits: Any, child_logits: Any):
    torch = _torch()
    teacher = _as_tensor(teacher_logits, "teacher_logits")
    child = _as_tensor(child_logits, "child_logits")
    if teacher.device != child.device:
        raise ValueError("teacher_logits and child_logits must be on the same device.")
    dtype = torch.promote_types(teacher.dtype, child.dtype)
    if not dtype.is_floating_point or dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    teacher = teacher.to(dtype=dtype)
    child = child.to(dtype=dtype)
    if not bool(torch.isfinite(teacher).all()) or not bool(torch.isfinite(child).all()):
        raise ValueError("teacher_logits and child_logits must contain only finite values.")
    return teacher, child


def _alignment_indices(
    torch,
    teacher_vocab_indices: Any,
    child_vocab_indices: Any,
    vocab_alignment: Any,
    *,
    teacher_device,
    child_device,
):
    if vocab_alignment is not None:
        if teacher_vocab_indices is not None or child_vocab_indices is not None:
            raise ValueError("Use vocab_alignment or the two vocab index arrays, not both.")
        pairs = vocab_alignment if torch.is_tensor(vocab_alignment) else list(vocab_alignment)
        pair_tensor = torch.as_tensor(pairs, dtype=torch.long)
        if pair_tensor.ndim != 2 or pair_tensor.shape[1] != 2:
            raise ValueError("vocab_alignment must have shape [aligned_vocab, 2].")
        teacher_vocab_indices = pair_tensor[:, 0]
        child_vocab_indices = pair_tensor[:, 1]
    if (teacher_vocab_indices is None) != (child_vocab_indices is None):
        raise ValueError("teacher_vocab_indices and child_vocab_indices must be provided together.")
    if teacher_vocab_indices is None:
        return None, None
    teacher_idx = torch.as_tensor(teacher_vocab_indices, dtype=torch.long, device=teacher_device).reshape(-1)
    child_idx = torch.as_tensor(child_vocab_indices, dtype=torch.long, device=child_device).reshape(-1)
    if teacher_idx.numel() == 0 or teacher_idx.numel() != child_idx.numel():
        raise ValueError("Aligned vocabulary index arrays must have the same nonzero length.")
    if torch.unique(teacher_idx).numel() != teacher_idx.numel():
        raise ValueError("teacher_vocab_indices must be unique for a probability alignment.")
    if torch.unique(child_idx).numel() != child_idx.numel():
        raise ValueError("child_vocab_indices must be unique for a probability alignment.")
    return teacher_idx, child_idx


def align_vocabulary_logits(
    teacher_logits: Any,
    child_logits: Any,
    *,
    teacher_vocab_indices: Any = None,
    child_vocab_indices: Any = None,
    vocab_alignment: Any = None,
):
    """Return teacher and child logits in a shared, one-to-one vocab order."""

    torch = _torch()
    teacher = _as_tensor(teacher_logits, "teacher_logits")
    child = _as_tensor(child_logits, "child_logits")
    if teacher.shape[:-1] != child.shape[:-1]:
        raise ValueError("Teacher and child logits must have identical leading dimensions.")
    teacher_idx, child_idx = _alignment_indices(
        torch,
        teacher_vocab_indices,
        child_vocab_indices,
        vocab_alignment,
        teacher_device=teacher.device,
        child_device=child.device,
    )
    if teacher_idx is None:
        if teacher.shape[-1] != child.shape[-1]:
            raise ValueError("Different vocabulary sizes require an explicit vocabulary alignment.")
        if teacher.shape[-1] == 0:
            raise ValueError("Logits must have a nonempty vocabulary dimension.")
        return teacher, child
    if bool((teacher_idx < 0).any()) or bool((teacher_idx >= teacher.shape[-1]).any()):
        raise ValueError("teacher_vocab_indices contains an out-of-range index.")
    if bool((child_idx < 0).any()) or bool((child_idx >= child.shape[-1]).any()):
        raise ValueError("child_vocab_indices contains an out-of-range index.")
    return teacher.index_select(-1, teacher_idx), child.index_select(-1, child_idx)


def centered_logit_cosine(
    teacher_logits: Any,
    child_logits: Any,
    *,
    eps: float = 1e-12,
    teacher_vocab_indices: Any = None,
    child_vocab_indices: Any = None,
    vocab_alignment: Any = None,
):
    """Cosine per observation after removing each logit row's vocab mean.

    A row with zero centered norm has no direction and receives a score of 0.
    """

    epsilon = _finite_positive(eps, "eps")
    teacher, child = align_vocabulary_logits(
        teacher_logits,
        child_logits,
        teacher_vocab_indices=teacher_vocab_indices,
        child_vocab_indices=child_vocab_indices,
        vocab_alignment=vocab_alignment,
    )
    teacher, child = _floating_pair(teacher, child)
    teacher = teacher - teacher.mean(dim=-1, keepdim=True)
    child = child - child.mean(dim=-1, keepdim=True)
    numerator = (teacher * child).sum(dim=-1)
    denominator = teacher.square().sum(dim=-1).sqrt() * child.square().sum(dim=-1).sqrt()
    score = numerator / denominator.clamp_min(epsilon)
    return score.clamp(-1.0, 1.0).where(denominator > epsilon, score.new_zeros(()))


def top1_agreement(
    teacher_logits: Any,
    child_logits: Any,
    *,
    teacher_vocab_indices: Any = None,
    child_vocab_indices: Any = None,
    vocab_alignment: Any = None,
):
    """Return a float indicator that the aligned argmax positions agree."""

    torch = _torch()
    teacher, child = align_vocabulary_logits(
        teacher_logits,
        child_logits,
        teacher_vocab_indices=teacher_vocab_indices,
        child_vocab_indices=child_vocab_indices,
        vocab_alignment=vocab_alignment,
    )
    return teacher.argmax(dim=-1).eq(child.argmax(dim=-1)).to(dtype=torch.float32)


def top_k_agreement(
    teacher_logits: Any,
    child_logits: Any,
    k: int = 5,
    *,
    mode: str = "overlap",
    teacher_vocab_indices: Any = None,
    child_vocab_indices: Any = None,
    vocab_alignment: Any = None,
):
    """Compare aligned top-k sets per observation.

    ``overlap`` returns intersection size divided by effective k, ``exact``
    returns set equality, and ``teacher-top1`` checks whether the teacher's
    winner appears in the child's top-k set.  Effective k is capped at vocab
    size.
    """

    torch = _torch()
    requested_k = _positive_int(k, "k")
    if mode not in {"overlap", "exact", "teacher-top1"}:
        raise ValueError("mode must be overlap, exact, or teacher-top1.")
    teacher, child = align_vocabulary_logits(
        teacher_logits,
        child_logits,
        teacher_vocab_indices=teacher_vocab_indices,
        child_vocab_indices=child_vocab_indices,
        vocab_alignment=vocab_alignment,
    )
    effective_k = min(requested_k, int(teacher.shape[-1]))
    teacher_top = torch.topk(teacher, k=effective_k, dim=-1).indices
    child_top = torch.topk(child, k=effective_k, dim=-1).indices
    if mode == "teacher-top1":
        return child_top.eq(teacher_top[..., :1]).any(dim=-1).to(dtype=torch.float32)
    overlap = teacher_top.unsqueeze(-1).eq(child_top.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
    if mode == "exact":
        return overlap.eq(effective_k).to(dtype=torch.float32)
    return overlap.to(dtype=torch.float32) / float(effective_k)


def teacher_top1_in_child_topk(
    teacher_logits: Any,
    child_logits: Any,
    k: int = 5,
    **alignment: Any,
):
    return top_k_agreement(teacher_logits, child_logits, k=k, mode="teacher-top1", **alignment)


topk_agreement = top_k_agreement


def teacher_child_kl(
    teacher_logits: Any,
    child_logits: Any,
    *,
    temperature: float = 1.0,
    teacher_vocab_indices: Any = None,
    child_vocab_indices: Any = None,
    vocab_alignment: Any = None,
):
    """Compute ``KL(teacher || child)`` per row over aligned vocabulary."""

    torch = _torch()
    scale = _finite_positive(temperature, "temperature")
    teacher, child = align_vocabulary_logits(
        teacher_logits,
        child_logits,
        teacher_vocab_indices=teacher_vocab_indices,
        child_vocab_indices=child_vocab_indices,
        vocab_alignment=vocab_alignment,
    )
    teacher, child = _floating_pair(teacher, child)
    teacher_logp = torch.nn.functional.log_softmax(teacher / scale, dim=-1)
    child_logp = torch.nn.functional.log_softmax(child / scale, dim=-1)
    kl = (teacher_logp.exp() * (teacher_logp - child_logp)).sum(dim=-1)
    return kl.clamp_min(0.0)


kl_teacher_child = teacher_child_kl


def normalized_kl_retention(
    kl_teacher_child_values: Any,
    blank_baseline_kl: Any,
    *,
    eps: float = 1e-12,
    reduction: str = "none",
):
    """Normalize child KL against blank KL as ``1 - child_kl / blank_kl``.

    Scores are intentionally not clipped: 1 is teacher-identical, 0 matches
    the blank baseline, and negative values are worse than blank.
    """

    torch = _torch()
    epsilon = _finite_positive(eps, "eps")
    child_kl = torch.as_tensor(kl_teacher_child_values)
    blank_kl = torch.as_tensor(blank_baseline_kl, device=child_kl.device)
    dtype = torch.promote_types(child_kl.dtype, blank_kl.dtype)
    if not dtype.is_floating_point or dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    child_kl = child_kl.to(dtype)
    blank_kl = blank_kl.to(dtype)
    if blank_kl.shape != child_kl.shape and blank_kl.numel() == child_kl.numel():
        blank_kl = blank_kl.reshape(child_kl.shape)
    try:
        child_kl, blank_kl = torch.broadcast_tensors(child_kl, blank_kl)
    except RuntimeError as exc:
        raise ValueError("blank_baseline_kl must be scalar or broadcast to child KL values.") from exc
    if not bool(torch.isfinite(child_kl).all()) or not bool(torch.isfinite(blank_kl).all()):
        raise ValueError("KL values must be finite.")
    if bool((child_kl < -epsilon).any()) or bool((blank_kl < -epsilon).any()):
        raise ValueError("KL values must be nonnegative.")
    if reduction == "none":
        return 1.0 - child_kl.clamp_min(0.0) / blank_kl.clamp_min(epsilon)
    if reduction == "mean":
        return (1.0 - child_kl.clamp_min(0.0) / blank_kl.clamp_min(epsilon)).mean()
    if reduction == "ratio-of-means":
        return 1.0 - child_kl.clamp_min(0.0).mean() / blank_kl.clamp_min(0.0).mean().clamp_min(epsilon)
    raise ValueError("reduction must be none, mean, or ratio-of-means.")


def _margin_index(index: Any, leading_shape: Sequence[int], vocab_size: int, device, name: str):
    torch = _torch()
    value = torch.as_tensor(index, dtype=torch.long, device=device)
    try:
        value = torch.broadcast_to(value, tuple(leading_shape))
    except RuntimeError as exc:
        raise ValueError(f"{name} must broadcast to the logits leading dimensions.") from exc
    if bool((value < 0).any()) or bool((value >= vocab_size).any()):
        raise ValueError(f"{name} contains an out-of-range index.")
    return value


def teacher_margin_values(
    teacher_logits: Any,
    child_logits: Any,
    *,
    positive_indices: Any = None,
    negative_indices: Any = None,
    teacher_vocab_indices: Any = None,
    child_vocab_indices: Any = None,
    vocab_alignment: Any = None,
):
    """Return teacher and child margins for the same aligned token pair.

    Without explicit pair indices, the pair is the teacher's top-1 and top-2.
    """

    torch = _torch()
    teacher, child = align_vocabulary_logits(
        teacher_logits,
        child_logits,
        teacher_vocab_indices=teacher_vocab_indices,
        child_vocab_indices=child_vocab_indices,
        vocab_alignment=vocab_alignment,
    )
    teacher, child = _floating_pair(teacher, child)
    if (positive_indices is None) != (negative_indices is None):
        raise ValueError("positive_indices and negative_indices must be provided together.")
    if positive_indices is None:
        if teacher.shape[-1] == 1:
            zeros = teacher[..., 0] - teacher[..., 0]
            return zeros, child[..., 0] - child[..., 0]
        pair = torch.topk(teacher, k=2, dim=-1).indices
        positive = pair[..., 0]
        negative = pair[..., 1]
    else:
        positive = _margin_index(
            positive_indices,
            teacher.shape[:-1],
            int(teacher.shape[-1]),
            teacher.device,
            "positive_indices",
        )
        negative = _margin_index(
            negative_indices,
            teacher.shape[:-1],
            int(teacher.shape[-1]),
            teacher.device,
            "negative_indices",
        )
    teacher_positive = teacher.gather(-1, positive.unsqueeze(-1)).squeeze(-1)
    teacher_negative = teacher.gather(-1, negative.unsqueeze(-1)).squeeze(-1)
    child_positive = child.gather(-1, positive.unsqueeze(-1)).squeeze(-1)
    child_negative = child.gather(-1, negative.unsqueeze(-1)).squeeze(-1)
    return teacher_positive - teacher_negative, child_positive - child_negative


def teacher_margin_sign_retention(
    teacher_logits: Any,
    child_logits: Any,
    *,
    tie_tolerance: float = 0.0,
    positive_indices: Any = None,
    negative_indices: Any = None,
    teacher_vocab_indices: Any = None,
    child_vocab_indices: Any = None,
    vocab_alignment: Any = None,
):
    """Return whether the child preserves the sign of each teacher margin."""

    torch = _torch()
    tolerance = float(tie_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tie_tolerance must be finite and nonnegative.")
    teacher_margin, child_margin = teacher_margin_values(
        teacher_logits,
        child_logits,
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        teacher_vocab_indices=teacher_vocab_indices,
        child_vocab_indices=child_vocab_indices,
        vocab_alignment=vocab_alignment,
    )
    teacher_sign = torch.sign(teacher_margin).where(teacher_margin.abs() > tolerance, teacher_margin.new_zeros(()))
    child_sign = torch.sign(child_margin).where(child_margin.abs() > tolerance, child_margin.new_zeros(()))
    return teacher_sign.eq(child_sign).to(dtype=torch.float32)


def compute_logit_diagnostics(
    teacher_logits: Any,
    child_logits: Any,
    *,
    top_k: int = 5,
    temperature: float = 1.0,
    blank_baseline_kl: Any = None,
    tie_tolerance: float = 0.0,
    teacher_vocab_indices: Any = None,
    child_vocab_indices: Any = None,
    vocab_alignment: Any = None,
) -> dict[str, Any]:
    """Compute all observation-level diagnostic tensors without mutation."""

    teacher, child = align_vocabulary_logits(
        teacher_logits,
        child_logits,
        teacher_vocab_indices=teacher_vocab_indices,
        child_vocab_indices=child_vocab_indices,
        vocab_alignment=vocab_alignment,
    )
    teacher_margin, child_margin = teacher_margin_values(teacher, child)
    metrics = {
        "centered_logit_cosine": centered_logit_cosine(teacher, child),
        "top1_agreement": top1_agreement(teacher, child),
        "topk_agreement": top_k_agreement(teacher, child, k=top_k),
        "topk_exact_agreement": top_k_agreement(teacher, child, k=top_k, mode="exact"),
        "teacher_top1_in_child_topk": teacher_top1_in_child_topk(teacher, child, k=top_k),
        "kl_teacher_child": teacher_child_kl(teacher, child, temperature=temperature),
        "teacher_margin": teacher_margin,
        "child_on_teacher_margin": child_margin,
        "teacher_margin_sign_retention": teacher_margin_sign_retention(
            teacher,
            child,
            tie_tolerance=tie_tolerance,
        ),
    }
    if blank_baseline_kl is not None:
        metrics["normalized_kl_retention"] = normalized_kl_retention(
            metrics["kl_teacher_child"],
            blank_baseline_kl,
        )
    return metrics


def _normalize_calibration_item(item: Any, index: int) -> dict[str, str]:
    if isinstance(item, str):
        text = item
        domain = "all"
        sample_id = str(index)
    elif isinstance(item, Mapping):
        text = item.get("text", item.get("prompt"))
        if not isinstance(text, str):
            raise ValueError("Each calibration mapping must contain a string text or prompt field.")
        domain = str(item.get("domain", "all"))
        sample_id = str(item.get("id", index))
    else:
        raise ValueError("Calibration items must be strings or mappings.")
    if not domain:
        domain = "all"
    return {"id": sample_id, "domain": domain, "text": text}


def _bounded_calibration_items(calibration: Iterable[Any], max_samples: int):
    if isinstance(calibration, (str, Mapping)):
        calibration = [calibration]
    bounded = list(islice(iter(calibration), max_samples + 1))
    truncated = len(bounded) > max_samples
    bounded = bounded[:max_samples]
    if not bounded:
        raise ValueError("Need at least one calibration item.")
    return [_normalize_calibration_item(item, index) for index, item in enumerate(bounded)], truncated


def calibration_hashes(calibration: Iterable[Any], *, max_samples: int = 4096) -> dict[str, Any]:
    """Hash normalized calibration text, IDs, domains, and order."""

    limit = _positive_int(max_samples, "max_samples")
    samples, truncated = _bounded_calibration_items(calibration, limit)
    sample_hashes = [_json_sha256(sample) for sample in samples]
    return {
        "algorithm": "sha256",
        "ordered_samples_sha256": _json_sha256(sample_hashes),
        "sample_sha256": sample_hashes,
        "sample_count": len(samples),
        "truncated": bool(truncated),
    }


def _alignment_hash(
    teacher_vocab_size: int,
    child_vocab_size: int,
    teacher_vocab_indices: Any = None,
    child_vocab_indices: Any = None,
    vocab_alignment: Any = None,
) -> str:
    torch = _torch()
    teacher_idx, child_idx = _alignment_indices(
        torch,
        teacher_vocab_indices,
        child_vocab_indices,
        vocab_alignment,
        teacher_device="cpu",
        child_device="cpu",
    )
    if teacher_idx is None:
        pairs: Any = "identity"
        aligned_size = teacher_vocab_size
    else:
        pairs = list(zip(teacher_idx.tolist(), child_idx.tolist()))
        aligned_size = int(teacher_idx.numel())
    return _json_sha256(
        {
            "teacher_vocab_size": int(teacher_vocab_size),
            "child_vocab_size": int(child_vocab_size),
            "aligned_vocab_size": int(aligned_size),
            "pairs": pairs,
        }
    )


def _default_calibration_hashes(row_count: int, domains: Sequence[str], pair_ids: Sequence[str]) -> dict[str, Any]:
    layout = {
        "row_count": int(row_count),
        "domains": [str(value) for value in domains],
        "pair_ids": [str(value) for value in pair_ids],
    }
    return {
        "algorithm": "sha256",
        "source": "tensor-layout",
        "calibration_layout_sha256": _json_sha256(layout),
    }


def _flatten_finite(values: Any, name: str):
    torch = _torch()
    tensor = torch.as_tensor(values).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values.")
    return tensor


def _collapse_pair_units(values, pair_ids: Sequence[Any] | None):
    torch = _torch()
    if pair_ids is None:
        return values
    if len(pair_ids) != int(values.numel()):
        raise ValueError("pair_ids length must match the number of observations.")
    grouped: dict[str, list[int]] = {}
    order: list[str] = []
    for index, pair_id in enumerate(pair_ids):
        key = str(pair_id)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(index)
    return torch.stack([values[grouped[key]].mean() for key in order])


def paired_bootstrap_report(
    values: Any,
    reference: Any = None,
    *,
    pair_ids: Sequence[Any] | None = None,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """One-sided percentile bound for a paired mean or paired mean delta."""

    torch = _torch()
    level = float(confidence)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    resamples = _positive_int(n_resamples, "n_resamples")
    chunk = _positive_int(chunk_size, "chunk_size")
    observed = _flatten_finite(values, "values")
    if reference is not None:
        baseline = torch.as_tensor(reference).detach().to(device="cpu", dtype=torch.float64)
        try:
            observed, baseline = torch.broadcast_tensors(observed, baseline.reshape(-1))
        except RuntimeError as exc:
            raise ValueError("reference must be scalar or match values.") from exc
        if not bool(torch.isfinite(baseline).all()):
            raise ValueError("reference must contain only finite values.")
        observed = observed - baseline
    units = _collapse_pair_units(observed.reshape(-1), pair_ids)
    unit_count = int(units.numel())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    bootstrap_means = torch.empty(resamples, dtype=torch.float64)
    offset = 0
    while offset < resamples:
        count = min(chunk, resamples - offset)
        indices = torch.randint(0, unit_count, (count, unit_count), generator=generator)
        bootstrap_means[offset : offset + count] = units[indices].mean(dim=1)
        offset += count
    lower = torch.quantile(bootstrap_means, 1.0 - level)
    return {
        "estimate": float(units.mean()),
        "lower_bound": float(lower),
        "confidence": level,
        "n_resamples": resamples,
        "seed": int(seed),
        "pair_count": unit_count,
        "method": "paired-percentile",
    }


def paired_bootstrap_lower_bound(
    values: Any,
    reference: Any = None,
    *,
    pair_ids: Sequence[Any] | None = None,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
    chunk_size: int = 512,
) -> float:
    return paired_bootstrap_report(
        values,
        reference,
        pair_ids=pair_ids,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
        chunk_size=chunk_size,
    )["lower_bound"]


def _metric_summary(
    values: Any,
    *,
    pair_ids: Sequence[Any] | None,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    units = _collapse_pair_units(_flatten_finite(values, "metric values"), pair_ids)
    bootstrap = paired_bootstrap_report(
        units,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    return {
        "mean": float(units.mean()),
        "std": float(units.std(unbiased=False)),
        "min": float(units.min()),
        "max": float(units.max()),
        "pair_count": int(units.numel()),
        "paired_bootstrap_lower_bound": bootstrap["lower_bound"],
        "bootstrap": bootstrap,
    }


def aggregate_by_domain(
    metrics: Mapping[str, Any],
    domains: Sequence[Any],
    *,
    pair_ids: Sequence[Any] | None = None,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Aggregate observation-level metric tensors independently by domain."""

    torch = _torch()
    if not metrics:
        raise ValueError("metrics must not be empty.")
    flattened = {name: _flatten_finite(value, name) for name, value in metrics.items()}
    row_counts = {int(value.numel()) for value in flattened.values()}
    if len(row_counts) != 1:
        raise ValueError("Every metric must contain the same number of observations.")
    row_count = row_counts.pop()
    if len(domains) != row_count:
        raise ValueError("domains length must match the number of observations.")
    if pair_ids is not None and len(pair_ids) != row_count:
        raise ValueError("pair_ids length must match the number of observations.")
    domain_names = [str(domain) for domain in domains]
    result: dict[str, Any] = {}
    for domain in sorted(set(domain_names)):
        indices = [index for index, value in enumerate(domain_names) if value == domain]
        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        domain_pairs = None if pair_ids is None else [pair_ids[index] for index in indices]
        result[domain] = {
            "row_count": len(indices),
            "metrics": {
                name: _metric_summary(
                    values.index_select(0, index_tensor),
                    pair_ids=domain_pairs,
                    confidence=confidence,
                    n_resamples=n_resamples,
                    seed=int(seed) + metric_index,
                )
                for metric_index, (name, values) in enumerate(flattened.items())
            },
        }
    return result


def _hashes_match(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    comparable = {
        "ordered_samples_sha256",
        "collected_samples_sha256",
        "tokenized_inputs_sha256",
        "calibration_layout_sha256",
    }
    shared = comparable.intersection(expected).intersection(actual)
    if shared:
        return all(expected[key] == actual[key] for key in shared)
    return _canonical_json(dict(expected)) == _canonical_json(dict(actual))


def build_blank_baseline_cache(
    teacher_logits: Any,
    blank_logits: Any,
    *,
    temperature: float = 1.0,
    calibration_hashes: Mapping[str, Any] | None = None,
    domains: Sequence[Any] | None = None,
    pair_ids: Sequence[Any] | None = None,
    teacher_vocab_indices: Any = None,
    blank_vocab_indices: Any = None,
    vocab_alignment: Any = None,
    vocabulary_alignment_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a JSON-serializable per-row blank KL cache."""

    raw_teacher = _as_tensor(teacher_logits, "teacher_logits")
    raw_blank = _as_tensor(blank_logits, "blank_logits")
    teacher, blank = align_vocabulary_logits(
        raw_teacher,
        raw_blank,
        teacher_vocab_indices=teacher_vocab_indices,
        child_vocab_indices=blank_vocab_indices,
        vocab_alignment=vocab_alignment,
    )
    values = teacher_child_kl(teacher, blank, temperature=temperature).detach().cpu().reshape(-1)
    row_count = int(values.numel())
    domain_values = ["all"] * row_count if domains is None else [str(value) for value in domains]
    pair_values = [str(index) for index in range(row_count)] if pair_ids is None else [str(value) for value in pair_ids]
    if len(domain_values) != row_count or len(pair_values) != row_count:
        raise ValueError("domains and pair_ids must match the number of logit observations.")
    hashes = dict(calibration_hashes or _default_calibration_hashes(row_count, domain_values, pair_values))
    alignment_hash = (
        vocabulary_alignment_sha256
        or hashes.get("vocabulary_alignment_sha256")
        or _alignment_hash(
            int(raw_teacher.shape[-1]),
            int(raw_blank.shape[-1]),
            teacher_vocab_indices,
            blank_vocab_indices,
            vocab_alignment,
        )
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": BLANK_BASELINE_KIND,
        "closed_form": True,
        "zero_gradient": True,
        "optimizer_steps": 0,
        "temperature": float(temperature),
        "row_count": row_count,
        "mean_kl_teacher_blank": float(values.mean()),
        "kl_teacher_blank": [float(value) for value in values],
        "calibration_hashes": hashes,
        "vocabulary_alignment_sha256": alignment_hash,
    }


def _baseline_values_from_cache(
    cache: Mapping[str, Any],
    *,
    row_count: int,
    temperature: float,
    calibration_hashes: Mapping[str, Any],
    alignment_hash: str,
    device,
):
    torch = _torch()
    if cache.get("kind") != BLANK_BASELINE_KIND:
        raise ValueError("blank_baseline_cache has an unsupported kind.")
    if cache.get("zero_gradient") is not True or int(cache.get("optimizer_steps", -1)) != 0:
        raise ValueError("Blank baseline cache must record a zero-gradient, zero-step collection.")
    if int(cache.get("row_count", -1)) != row_count:
        raise ValueError("Blank baseline cache row count does not match this calibration.")
    cache_temperature = float(cache.get("temperature", float("nan")))
    if not math.isclose(cache_temperature, float(temperature), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Blank baseline cache temperature does not match this diagnostic.")
    cache_hashes = cache.get("calibration_hashes")
    if not isinstance(cache_hashes, Mapping) or not _hashes_match(cache_hashes, calibration_hashes):
        raise ValueError("Blank baseline cache calibration hashes do not match.")
    if cache.get("vocabulary_alignment_sha256") != alignment_hash:
        raise ValueError("Blank baseline cache vocabulary alignment does not match.")
    values = torch.as_tensor(cache.get("kl_teacher_blank"), dtype=torch.float32, device=device).reshape(-1)
    if values.numel() != row_count or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("Blank baseline cache contains invalid KL values.")
    return values


def build_logit_diagnostics_report(
    teacher_logits: Any,
    child_logits: Any,
    *,
    domains: Sequence[Any] | None = None,
    pair_ids: Sequence[Any] | None = None,
    top_k: int = 5,
    temperature: float = 1.0,
    blank_baseline_kl: Any = None,
    blank_baseline_cache: Mapping[str, Any] | None = None,
    calibration_hashes: Mapping[str, Any] | None = None,
    bootstrap_confidence: float = 0.95,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 0,
    tie_tolerance: float = 0.0,
    teacher_vocab_indices: Any = None,
    child_vocab_indices: Any = None,
    vocab_alignment: Any = None,
    vocabulary_alignment_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe zero-gradient diagnostic report."""

    if blank_baseline_kl is not None and blank_baseline_cache is not None:
        raise ValueError("Provide blank_baseline_kl or blank_baseline_cache, not both.")
    raw_teacher = _as_tensor(teacher_logits, "teacher_logits")
    raw_child = _as_tensor(child_logits, "child_logits")
    teacher, child = align_vocabulary_logits(
        raw_teacher,
        raw_child,
        teacher_vocab_indices=teacher_vocab_indices,
        child_vocab_indices=child_vocab_indices,
        vocab_alignment=vocab_alignment,
    )
    row_count = int(teacher.numel() // teacher.shape[-1])
    domain_values = ["all"] * row_count if domains is None else [str(value) for value in domains]
    pair_values = [str(index) for index in range(row_count)] if pair_ids is None else [str(value) for value in pair_ids]
    if len(domain_values) != row_count or len(pair_values) != row_count:
        raise ValueError("domains and pair_ids must match the number of logit observations.")
    hashes = dict(calibration_hashes or _default_calibration_hashes(row_count, domain_values, pair_values))
    alignment_hash = (
        vocabulary_alignment_sha256
        or hashes.get("vocabulary_alignment_sha256")
        or _alignment_hash(
            int(raw_teacher.shape[-1]),
            int(raw_child.shape[-1]),
            teacher_vocab_indices,
            child_vocab_indices,
            vocab_alignment,
        )
    )
    baseline_values = blank_baseline_kl
    if blank_baseline_cache is not None:
        baseline_values = _baseline_values_from_cache(
            blank_baseline_cache,
            row_count=row_count,
            temperature=temperature,
            calibration_hashes=hashes,
            alignment_hash=alignment_hash,
            device=teacher.device,
        )
    metrics = compute_logit_diagnostics(
        teacher,
        child,
        top_k=top_k,
        temperature=temperature,
        blank_baseline_kl=baseline_values,
        tie_tolerance=tie_tolerance,
    )
    flattened = {name: value.reshape(-1) for name, value in metrics.items()}
    overall_metrics = {
        name: _metric_summary(
            values,
            pair_ids=pair_values,
            confidence=bootstrap_confidence,
            n_resamples=bootstrap_resamples,
            seed=int(bootstrap_seed) + metric_index,
        )
        for metric_index, (name, values) in enumerate(flattened.items())
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "operation": "closed_form_logit_diagnostics",
        "closed_form": True,
        "zero_gradient": True,
        "optimizer_steps": 0,
        "row_count": row_count,
        "pair_count": len(set(pair_values)),
        "aligned_vocab_size": int(teacher.shape[-1]),
        "top_k": int(top_k),
        "effective_top_k": min(int(top_k), int(teacher.shape[-1])),
        "temperature": float(temperature),
        "calibration_hashes": hashes,
        "vocabulary_alignment_sha256": alignment_hash,
        "blank_baseline_cached": blank_baseline_cache is not None,
        "overall": {"row_count": row_count, "pair_count": len(set(pair_values)), "metrics": overall_metrics},
        "per_domain": aggregate_by_domain(
            flattened,
            domain_values,
            pair_ids=pair_values,
            confidence=bootstrap_confidence,
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        "bootstrap": {
            "method": "paired-percentile",
            "confidence": float(bootstrap_confidence),
            "n_resamples": int(bootstrap_resamples),
            "seed": int(bootstrap_seed),
            "pairing_unit": "pair_ids",
        },
    }
    if baseline_values is not None:
        baseline_tensor = _flatten_finite(baseline_values, "blank_baseline_kl")
        if baseline_tensor.numel() == 1:
            baseline_mean = float(baseline_tensor[0])
        elif baseline_tensor.numel() == row_count:
            baseline_mean = float(baseline_tensor.mean())
        else:
            raise ValueError("blank_baseline_kl must be scalar or match the number of observations.")
        report["blank_baseline"] = {
            "kind": BLANK_BASELINE_KIND if blank_baseline_cache is not None else "direct",
            "cached": blank_baseline_cache is not None,
            "mean_kl_teacher_blank": baseline_mean,
        }
        report["normalized_kl_retention_ratio_of_means"] = float(
            normalized_kl_retention(
                flattened["kl_teacher_child"],
                baseline_values,
                reduction="ratio-of-means",
            )
        )
    return report


def _model_device(torch, model):
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        try:
            return next(model.buffers()).device
        except (StopIteration, AttributeError):
            return torch.device("cpu")


def _model_logits(output: Any, name: str):
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, Mapping):
        logits = output.get("logits")
    if logits is None and isinstance(output, (tuple, list)) and output:
        logits = output[0]
    if logits is None:
        raise ValueError(f"{name} output does not contain logits.")
    if logits.ndim != 3:
        raise ValueError(f"{name} logits must have shape [batch, sequence, vocabulary].")
    return logits


def collect_hf_calibration_logits(
    teacher_model: Any,
    child_model: Any,
    tokenizer: Any,
    calibration: Iterable[Any],
    *,
    max_samples: int = 32,
    batch_size: int = 1,
    max_length: int = 128,
    max_rows: int = 1024,
    max_logit_elements: int = 16_000_000,
    device: Any = None,
    teacher_device: Any = None,
    child_device: Any = None,
    teacher_vocab_indices: Any = None,
    child_vocab_indices: Any = None,
    vocab_alignment: Any = None,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect bounded aligned HF logits without creating or writing gradients."""

    torch = _torch()
    sample_limit = _positive_int(max_samples, "max_samples")
    batch_limit = _positive_int(batch_size, "batch_size")
    length_limit = _positive_int(max_length, "max_length")
    row_limit = _positive_int(max_rows, "max_rows")
    element_limit = _positive_int(max_logit_elements, "max_logit_elements")
    samples, sample_truncated = _bounded_calibration_items(calibration, sample_limit)
    extra_tokenizer_kwargs = dict(tokenizer_kwargs or {})
    reserved = {"return_tensors", "padding", "truncation", "max_length"}.intersection(extra_tokenizer_kwargs)
    if reserved:
        raise ValueError(f"tokenizer_kwargs cannot override bounded arguments: {sorted(reserved)}")

    teacher_input_device = torch.device(teacher_device or device or _model_device(torch, teacher_model))
    child_input_device = torch.device(child_device or device or _model_device(torch, child_model))
    original_modes: list[tuple[Any, bool]] = []
    seen_models: set[int] = set()
    for model in (teacher_model, child_model):
        if id(model) in seen_models:
            continue
        seen_models.add(id(model))
        original_modes.append((model, bool(getattr(model, "training", False))))
        model.eval()

    teacher_chunks = []
    child_chunks = []
    domains: list[str] = []
    pair_ids: list[str] = []
    used_positions: dict[int, list[int]] = {}
    tokenized_contexts: dict[int, list[int]] = {}
    used_sample_indices: set[int] = set()
    retained_elements = 0
    stopped_by_rows = False
    stopped_by_elements = False
    teacher_vocab_size = None
    child_vocab_size = None
    alignment_hash = None
    try:
        with torch.inference_mode():
            for batch_start in range(0, len(samples), batch_limit):
                if sum(chunk.shape[0] for chunk in teacher_chunks) >= row_limit:
                    stopped_by_rows = True
                    break
                batch_samples = samples[batch_start : batch_start + batch_limit]
                encoded = tokenizer(
                    [sample["text"] for sample in batch_samples],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=length_limit,
                    **extra_tokenizer_kwargs,
                )
                if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
                    raise ValueError("Tokenizer must return a mapping containing input_ids.")
                input_ids_cpu = torch.as_tensor(encoded["input_ids"]).detach().cpu()
                if input_ids_cpu.ndim != 2:
                    raise ValueError("Tokenizer input_ids must have shape [batch, sequence].")
                attention_cpu = torch.as_tensor(
                    encoded.get("attention_mask", torch.ones_like(input_ids_cpu))
                ).detach().cpu().bool()
                if attention_cpu.shape != input_ids_cpu.shape:
                    raise ValueError("Tokenizer attention_mask must match input_ids shape.")
                teacher_inputs = {
                    key: value.to(teacher_input_device) if torch.is_tensor(value) else value
                    for key, value in encoded.items()
                }
                child_inputs = {
                    key: value.to(child_input_device) if torch.is_tensor(value) else value
                    for key, value in encoded.items()
                }
                teacher_output = teacher_model(**teacher_inputs, use_cache=False)
                child_output = child_model(**child_inputs, use_cache=False)
                teacher_batch = _model_logits(teacher_output, "Teacher")
                child_batch = _model_logits(child_output, "Child")
                if teacher_batch.shape[:2] != child_batch.shape[:2] or teacher_batch.shape[:2] != input_ids_cpu.shape:
                    raise ValueError("Teacher, child, and tokenizer batch/sequence dimensions must match.")
                teacher_vocab_size = int(teacher_batch.shape[-1])
                child_vocab_size = int(child_batch.shape[-1])
                teacher_idx, child_idx = _alignment_indices(
                    torch,
                    teacher_vocab_indices,
                    child_vocab_indices,
                    vocab_alignment,
                    teacher_device=teacher_batch.device,
                    child_device=child_batch.device,
                )
                if teacher_idx is None:
                    if teacher_vocab_size != child_vocab_size:
                        raise ValueError("Different model vocabulary sizes require an explicit vocabulary alignment.")
                else:
                    if bool((teacher_idx < 0).any()) or bool((teacher_idx >= teacher_vocab_size).any()):
                        raise ValueError("teacher_vocab_indices contains an out-of-range index.")
                    if bool((child_idx < 0).any()) or bool((child_idx >= child_vocab_size).any()):
                        raise ValueError("child_vocab_indices contains an out-of-range index.")
                    teacher_batch = teacher_batch.index_select(-1, teacher_idx)
                    child_batch = child_batch.index_select(-1, child_idx)
                if alignment_hash is None:
                    alignment_hash = _alignment_hash(
                        teacher_vocab_size,
                        child_vocab_size,
                        teacher_vocab_indices,
                        child_vocab_indices,
                        vocab_alignment,
                    )
                flat_mask = attention_cpu.reshape(-1)
                flat_batch_indices = (
                    torch.arange(input_ids_cpu.shape[0]).unsqueeze(1).expand_as(input_ids_cpu).reshape(-1)[flat_mask]
                )
                flat_positions = (
                    torch.arange(input_ids_cpu.shape[1]).unsqueeze(0).expand_as(input_ids_cpu).reshape(-1)[flat_mask]
                )
                teacher_rows = teacher_batch.reshape(-1, teacher_batch.shape[-1])[
                    flat_mask.to(teacher_batch.device)
                ]
                child_rows = child_batch.reshape(-1, child_batch.shape[-1])[flat_mask.to(child_batch.device)]
                remaining_rows = row_limit - sum(chunk.shape[0] for chunk in teacher_chunks)
                row_width = int(teacher_rows.shape[-1] + child_rows.shape[-1])
                element_rows = (element_limit - retained_elements) // row_width
                take = min(int(teacher_rows.shape[0]), remaining_rows, element_rows)
                if take <= 0:
                    if not teacher_chunks:
                        raise ValueError("max_logit_elements is too small to retain one aligned logit row.")
                    stopped_by_elements = element_rows <= 0
                    stopped_by_rows = remaining_rows <= 0
                    break
                teacher_chunks.append(teacher_rows[:take].detach().to(device="cpu", dtype=torch.float32))
                child_chunks.append(child_rows[:take].detach().to(device="cpu", dtype=torch.float32))
                retained_elements += take * row_width
                for row_index in range(take):
                    local_sample = int(flat_batch_indices[row_index])
                    sample_index = batch_start + local_sample
                    sample = samples[sample_index]
                    used_sample_indices.add(sample_index)
                    pair_id = f"{sample_index}:{sample['id']}"
                    domains.append(sample["domain"])
                    pair_ids.append(pair_id)
                    used_positions.setdefault(sample_index, []).append(int(flat_positions[row_index]))
                for local_sample, sample in enumerate(batch_samples):
                    sample_index = batch_start + local_sample
                    valid_ids = input_ids_cpu[local_sample][attention_cpu[local_sample]].tolist()
                    tokenized_contexts[sample_index] = [int(value) for value in valid_ids]
                if take < int(teacher_rows.shape[0]):
                    stopped_by_rows = remaining_rows <= element_rows
                    stopped_by_elements = element_rows < remaining_rows
                    break
    finally:
        for model, was_training in original_modes:
            model.train(was_training)

    if not teacher_chunks or teacher_vocab_size is None or child_vocab_size is None or alignment_hash is None:
        raise ValueError("Calibration produced no valid logit rows.")
    teacher_logits = torch.cat(teacher_chunks, dim=0)
    child_logits = torch.cat(child_chunks, dim=0)
    used_indices = sorted(used_sample_indices)
    used_samples = [samples[index] for index in used_indices]
    sample_hash_values = [_json_sha256(sample) for sample in used_samples]
    tokenized_records = [
        {
            "sample_sha256": _json_sha256(samples[index]),
            "input_ids": tokenized_contexts[index],
            "retained_positions": used_positions[index],
        }
        for index in used_indices
    ]
    hashes = {
        "algorithm": "sha256",
        "ordered_samples_sha256": _json_sha256(sample_hash_values),
        "collected_samples_sha256": _json_sha256(sample_hash_values),
        "sample_sha256": sample_hash_values,
        "tokenized_inputs_sha256": _json_sha256(tokenized_records),
        "vocabulary_alignment_sha256": alignment_hash,
    }
    metadata = {
        "zero_gradient": True,
        "optimizer_steps": 0,
        "gradient_context": "torch.inference_mode",
        "parameter_gradients_written": False,
        "input_sample_count": len(samples),
        "collected_sample_count": len(used_samples),
        "row_count": int(teacher_logits.shape[0]),
        "teacher_vocab_size": teacher_vocab_size,
        "child_vocab_size": child_vocab_size,
        "aligned_vocab_size": int(teacher_logits.shape[-1]),
        "retained_logit_elements": retained_elements,
        "bounds": {
            "max_samples": sample_limit,
            "batch_size": batch_limit,
            "max_length": length_limit,
            "max_rows": row_limit,
            "max_logit_elements": element_limit,
        },
        "truncated": {
            "max_samples": bool(sample_truncated),
            "max_rows": bool(stopped_by_rows),
            "max_logit_elements": bool(stopped_by_elements),
        },
        "calibration_hashes": hashes,
        "vocabulary_alignment_sha256": alignment_hash,
    }
    return {
        "teacher_logits": teacher_logits,
        "child_logits": child_logits,
        "domains": domains,
        "pair_ids": pair_ids,
        "calibration_hashes": hashes,
        "metadata": metadata,
        "zero_gradient": True,
        "optimizer_steps": 0,
    }


collect_hf_calibration = collect_hf_calibration_logits


def collect_hf_logit_diagnostics(
    teacher_model: Any,
    child_model: Any,
    tokenizer: Any,
    calibration: Iterable[Any],
    *,
    top_k: int = 5,
    temperature: float = 1.0,
    blank_baseline_cache: Mapping[str, Any] | None = None,
    bootstrap_confidence: float = 0.95,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 0,
    **collector_kwargs: Any,
) -> dict[str, Any]:
    """Collect bounded logits and immediately return their JSON-safe report."""

    collected = collect_hf_calibration_logits(
        teacher_model,
        child_model,
        tokenizer,
        calibration,
        **collector_kwargs,
    )
    report = build_logit_diagnostics_report(
        collected["teacher_logits"],
        collected["child_logits"],
        domains=collected["domains"],
        pair_ids=collected["pair_ids"],
        top_k=top_k,
        temperature=temperature,
        blank_baseline_cache=blank_baseline_cache,
        calibration_hashes=collected["calibration_hashes"],
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        vocabulary_alignment_sha256=collected["metadata"]["vocabulary_alignment_sha256"],
    )
    report["calibration"] = collected["metadata"]
    return report
