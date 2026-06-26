"""Constrained cross-family Hugging Face model fusion utilities.

This module handles the first practical cross-family case: keep a host runtime
architecture, then transplant donor embedding/lm-head signal through shared
decoded-token anchors. It is not blind whole-checkpoint averaging.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from neural_forge.hf_tools import (
    _dtype_from_name,
    _ensure_unique_ratio_tags,
    _finite_float,
    _load_report,
    _ratio_tag,
    _torch_and_transformers,
    evaluate_text_model,
    gate_text_report,
)


def cross_family_vocab_merge(
    host_model: str,
    donor_model: str,
    output_dir: str | Path,
    alpha: float = 0.05,
    ridge: float = 1e-3,
    max_anchors: int = 50_000,
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Blend donor token embedding/head signal into a host architecture.

    The output remains a valid host model. Shared decoded tokens provide anchor
    pairs. A ridge map projects donor hidden rows into host hidden size, then
    matching host token rows are blended by `alpha`.
    """

    if not math.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1].")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if max_anchors <= 0:
        raise ValueError("max_anchors must be positive.")

    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    host = transformers.AutoModelForCausalLM.from_pretrained(
        host_model,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    donor = transformers.AutoModelForCausalLM.from_pretrained(
        donor_model,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    host_tokenizer = transformers.AutoTokenizer.from_pretrained(host_model, trust_remote_code=trust_remote_code)
    donor_tokenizer = transformers.AutoTokenizer.from_pretrained(donor_model, trust_remote_code=trust_remote_code)

    host_map = _canonical_token_map(host_tokenizer)
    donor_map = _canonical_token_map(donor_tokenizer)
    shared = sorted(set(host_map) & set(donor_map), key=lambda item: (len(item), item))
    if not shared:
        raise ValueError("No shared decoded-token anchors found.")
    anchors = shared[: min(max_anchors, len(shared))]

    host_emb = host.get_input_embeddings().weight.data.detach().cpu().to(torch.float32)
    donor_emb = donor.get_input_embeddings().weight.data.detach().cpu().to(torch.float32)
    h_idx = torch.tensor([host_map[token] for token in anchors], dtype=torch.long)
    d_idx = torch.tensor([donor_map[token] for token in anchors], dtype=torch.long)
    emb_map = _ridge_map(torch, donor_emb[d_idx], host_emb[h_idx], ridge)
    projected_emb = donor_emb[d_idx].matmul(emb_map)
    host_emb[h_idx] = (1.0 - float(alpha)) * host_emb[h_idx] + float(alpha) * projected_emb
    host.get_input_embeddings().weight.data.copy_(host_emb.to(host.get_input_embeddings().weight.dtype))

    host_lm = host.get_output_embeddings()
    donor_lm = donor.get_output_embeddings()
    head_updated = False
    if host_lm is not None and donor_lm is not None:
        host_head = host_lm.weight.data.detach().cpu().to(torch.float32)
        donor_head = donor_lm.weight.data.detach().cpu().to(torch.float32)
        head_map = _ridge_map(torch, donor_head[d_idx], host_head[h_idx], ridge)
        projected_head = donor_head[d_idx].matmul(head_map)
        host_head[h_idx] = (1.0 - float(alpha)) * host_head[h_idx] + float(alpha) * projected_head
        host_lm.weight.data.copy_(host_head.to(host_lm.weight.dtype))
        head_updated = True

    if getattr(host.config, "tie_word_embeddings", False):
        host.tie_weights()

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    host.save_pretrained(target, safe_serialization=True)
    host_tokenizer.save_pretrained(target)
    report = {
        "operation": "cross_family_vocab_merge",
        "host_model": host_model,
        "donor_model": donor_model,
        "output_dir": str(target),
        "alpha": float(alpha),
        "ridge": ridge,
        "torch_dtype": torch_dtype,
        "host_vocab_size": len(host_tokenizer),
        "donor_vocab_size": len(donor_tokenizer),
        "shared_decoded_tokens": len(shared),
        "anchors_used": len(anchors),
        "host_hidden_size": int(host_emb.shape[1]),
        "donor_hidden_size": int(donor_emb.shape[1]),
        "lm_head_updated": head_updated,
        "param_count": sum(param.numel() for param in host.parameters()),
    }
    _write_json(target / "neural_forge_xfamily_vocab_report.json", report)
    return report


def sweep_cross_family_vocab_merge(
    host_model: str,
    donor_model: str,
    output_dir: str | Path,
    alphas: list[float],
    baseline_report: str | Path | None = None,
    ridge: float = 1e-3,
    max_anchors: int = 50_000,
    torch_dtype: str = "float16",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    max_new_tokens: int = 80,
    max_ppl_ratio: float = 1.5,
    max_accuracy_drop: float = 0.25,
    min_choice_accuracy: float = 0.5,
) -> dict[str, Any]:
    """Sweep constrained cross-family token transplants and gate each result."""

    if not alphas:
        raise ValueError("At least one alpha is required.")
    alpha_values = [_finite_float(alpha, "alpha") for alpha in alphas]
    for alpha_value in alpha_values:
        if not 0.0 <= alpha_value <= 1.0:
            raise ValueError("alpha values must be in [0, 1].")
    _ensure_unique_ratio_tags(alpha_values, "alpha")

    root = Path(output_dir)
    model_root = root / "models"
    report_root = root / "reports"
    model_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    if baseline_report is None:
        baseline_path = report_root / "baseline_host.json"
        baseline = evaluate_text_model(
            host_model,
            baseline_path,
            torch_dtype=torch_dtype,
            device=device,
            trust_remote_code=trust_remote_code,
            max_new_tokens=max_new_tokens,
        )
    else:
        baseline_path = Path(baseline_report)
        baseline = _load_report(baseline_path)

    candidates = []
    for alpha_value in alpha_values:
        tag = _ratio_tag(alpha_value)
        candidate_dir = model_root / f"xfamily_alpha_{tag}"
        candidate_report_path = report_root / f"xfamily_alpha_{tag}.json"
        gate_path = report_root / f"xfamily_alpha_{tag}_gate.json"
        merge_report = cross_family_vocab_merge(
            host_model=host_model,
            donor_model=donor_model,
            output_dir=candidate_dir,
            alpha=alpha_value,
            ridge=ridge,
            max_anchors=max_anchors,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        candidate_report = evaluate_text_model(
            str(candidate_dir),
            candidate_report_path,
            torch_dtype=torch_dtype,
            device=device,
            trust_remote_code=trust_remote_code,
            max_new_tokens=max_new_tokens,
        )
        gate = gate_text_report(
            baseline,
            candidate_report,
            output_file=gate_path,
            max_ppl_ratio=max_ppl_ratio,
            max_accuracy_drop=max_accuracy_drop,
            min_choice_accuracy=min_choice_accuracy,
        )
        candidates.append(
            {
                "alpha": alpha_value,
                "tag": tag,
                "model_dir": str(candidate_dir),
                "report": str(candidate_report_path),
                "gate_report": str(gate_path),
                "passed": gate["passed"],
                "merge": merge_report,
                "metrics": gate["candidate"],
            }
        )

    summary = {
        "operation": "sweep_cross_family_vocab_merge",
        "host_model": host_model,
        "donor_model": donor_model,
        "baseline_report": str(baseline_path),
        "baseline_metrics": {
            "param_count": baseline.get("param_count"),
            "perplexity_mean": float(baseline["perplexity"]["mean"]),
            "choice_accuracy": float(baseline["choice_accuracy"]["accuracy"]),
        },
        "ridge": ridge,
        "max_anchors": max_anchors,
        "gate_thresholds": {
            "max_ppl_ratio": max_ppl_ratio,
            "max_accuracy_drop": max_accuracy_drop,
            "min_choice_accuracy": min_choice_accuracy,
        },
        "candidates": candidates,
        "passed_candidates": [item for item in candidates if item["passed"]],
    }
    _write_json(report_root / "xfamily_sweep_summary.json", summary)
    return summary


def _canonical_token_map(tokenizer) -> dict[str, int]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    mapping: dict[str, int] = {}
    for idx in range(len(tokenizer)):
        if idx in special_ids:
            continue
        try:
            decoded = tokenizer.decode([idx], clean_up_tokenization_spaces=False)
        except Exception:
            decoded = ""
        key = _canonical_token(decoded)
        if key and key not in mapping:
            mapping[key] = idx
    return mapping


def _canonical_token(text: str) -> str:
    text = text.replace("\u2581", " ").replace("Ġ", " ")
    text = text.strip()
    if not text:
        return ""
    if any(ord(ch) < 32 for ch in text):
        return ""
    return text


def _ridge_map(torch, source, target, ridge: float):
    source = source.to(torch.float32)
    target = target.to(torch.float32)
    xtx = source.T.matmul(source)
    if ridge > 0:
        xtx = xtx + torch.eye(xtx.shape[0], dtype=xtx.dtype) * float(ridge)
    xty = source.T.matmul(target)
    return torch.linalg.solve(xtx, xty)


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
