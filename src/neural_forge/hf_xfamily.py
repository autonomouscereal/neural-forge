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
    _decoder_layer_container,
    _dtype_from_name,
    _ensure_unique_ratio_tags,
    _finite_float,
    _load_report,
    _ratio_tag,
    _torch_and_transformers,
    evaluate_text_model,
    gate_text_report,
    search_merge_first_compression,
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


def cross_family_mlp_merge(
    host_model: str,
    donor_model: str,
    output_dir: str | Path,
    alpha: float = 0.01,
    vocab_alpha: float | None = None,
    ridge: float = 1e-3,
    max_anchors: int = 50_000,
    max_host_channels: int = 512,
    max_donor_channels: int = 512,
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Merge donor MLP channel signal into a host architecture.

    The output remains a normal host checkpoint. Shared decoded-token anchors
    solve a donor-hidden -> host-hidden projection. Donor FFN channels are then
    projected into host hidden space, assigned to similar host FFN channels, and
    blended into the host MLP weights. This is intentionally merge-first: no host
    channels are cut here.
    """

    alpha_value = _bounded_alpha(alpha, "alpha")
    vocab_alpha_value = alpha_value if vocab_alpha is None else _bounded_alpha(vocab_alpha, "vocab_alpha")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if max_anchors <= 0:
        raise ValueError("max_anchors must be positive.")
    if max_host_channels <= 0 or max_donor_channels <= 0:
        raise ValueError("max channel limits must be positive.")

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

    anchors, host_map, donor_map = _shared_anchor_tokens(host_tokenizer, donor_tokenizer, max_anchors)
    torch_anchor_idx_host = torch.tensor([host_map[token] for token in anchors], dtype=torch.long)
    torch_anchor_idx_donor = torch.tensor([donor_map[token] for token in anchors], dtype=torch.long)
    host_emb = host.get_input_embeddings().weight.data.detach().cpu().to(torch.float32)
    donor_emb = donor.get_input_embeddings().weight.data.detach().cpu().to(torch.float32)
    hidden_map = _ridge_map(torch, donor_emb[torch_anchor_idx_donor], host_emb[torch_anchor_idx_host], ridge)

    if vocab_alpha_value > 0:
        projected_emb = donor_emb[torch_anchor_idx_donor].matmul(hidden_map)
        host_emb[torch_anchor_idx_host] = (1.0 - vocab_alpha_value) * host_emb[torch_anchor_idx_host] + vocab_alpha_value * projected_emb
        host.get_input_embeddings().weight.data.copy_(host_emb.to(host.get_input_embeddings().weight.dtype))
        _merge_lm_head_rows(torch, host, donor, torch_anchor_idx_host, torch_anchor_idx_donor, hidden_map, vocab_alpha_value, ridge)

    host_layers = _compatible_mlp_layers(host)
    donor_layers = _compatible_mlp_layers(donor)
    if not host_layers:
        raise ValueError("Host model has no supported MLP layers.")
    if not donor_layers:
        raise ValueError("Donor model has no supported MLP layers.")

    layer_reports = []
    for host_idx, host_layer in enumerate(host_layers):
        donor_idx = _proportional_index(host_idx, len(host_layers), len(donor_layers))
        donor_layer = donor_layers[donor_idx]
        report = _merge_mlp_layer_channels(
            torch,
            host_layer,
            donor_layer,
            hidden_map,
            alpha_value,
            max_host_channels=max_host_channels,
            max_donor_channels=max_donor_channels,
        )
        report["host_layer"] = host_idx
        report["donor_layer"] = donor_idx
        layer_reports.append(report)

    if getattr(host.config, "tie_word_embeddings", False):
        host.tie_weights()

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    host.save_pretrained(target, safe_serialization=True)
    host_tokenizer.save_pretrained(target)
    report = {
        "operation": "cross_family_mlp_merge",
        "host_model": host_model,
        "donor_model": donor_model,
        "output_dir": str(target),
        "alpha": alpha_value,
        "vocab_alpha": vocab_alpha_value,
        "ridge": ridge,
        "host_vocab_size": len(host_tokenizer),
        "donor_vocab_size": len(donor_tokenizer),
        "shared_decoded_tokens": len(set(_canonical_token_map(host_tokenizer)) & set(_canonical_token_map(donor_tokenizer))),
        "anchors_used": len(anchors),
        "host_hidden_size": int(host_emb.shape[1]),
        "donor_hidden_size": int(donor_emb.shape[1]),
        "host_layers": len(host_layers),
        "donor_layers": len(donor_layers),
        "max_host_channels": max_host_channels,
        "max_donor_channels": max_donor_channels,
        "layers": layer_reports,
        "param_count": sum(param.numel() for param in host.parameters()),
    }
    _write_json(target / "neural_forge_xfamily_mlp_report.json", report)
    return report


def sweep_cross_family_mlp_merge(
    host_model: str,
    donor_model: str,
    output_dir: str | Path,
    alphas: list[float],
    baseline_report: str | Path | None = None,
    vocab_alpha: float | None = None,
    ridge: float = 1e-3,
    max_anchors: int = 50_000,
    max_host_channels: int = 512,
    max_donor_channels: int = 512,
    torch_dtype: str = "float16",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    max_new_tokens: int = 80,
    max_ppl_ratio: float = 1.5,
    max_accuracy_drop: float = 0.25,
    min_choice_accuracy: float = 0.5,
    recover_candidates: bool = False,
    recover_only_failed: bool = False,
    recovery_steps: int = 40,
    recovery_batch_size: int = 1,
    recovery_max_length: int = 96,
    recovery_lr: float = 5e-5,
    recovery_kd_weight: float = 0.5,
    recovery_lm_weight: float = 0.5,
    recovery_temperature: float = 2.0,
    recovery_train_scope: str = "last1",
    recovery_dtype: str = "float32",
    recovery_grad_clip: float = 1.0,
) -> dict[str, Any]:
    """Sweep cross-family MLP-channel transplants and gate each result."""

    if not alphas:
        raise ValueError("At least one alpha is required.")
    alpha_values = [_finite_float(alpha, "alpha") for alpha in alphas]
    for alpha_value in alpha_values:
        _bounded_alpha(alpha_value, "alpha")
    _ensure_unique_ratio_tags(alpha_values, "alpha")

    root = Path(output_dir)
    model_root = root / "models"
    recovered_root = root / "recovered_models"
    report_root = root / "reports"
    model_root.mkdir(parents=True, exist_ok=True)
    if recover_candidates:
        recovered_root.mkdir(parents=True, exist_ok=True)
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
        candidate_dir = model_root / f"xfamily_mlp_alpha_{tag}"
        candidate_report_path = report_root / f"xfamily_mlp_alpha_{tag}.json"
        gate_path = report_root / f"xfamily_mlp_alpha_{tag}_gate.json"
        merge_report = cross_family_mlp_merge(
            host_model=host_model,
            donor_model=donor_model,
            output_dir=candidate_dir,
            alpha=alpha_value,
            vocab_alpha=vocab_alpha,
            ridge=ridge,
            max_anchors=max_anchors,
            max_host_channels=max_host_channels,
            max_donor_channels=max_donor_channels,
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
        item = {
            "alpha": alpha_value,
            "tag": tag,
            "model_dir": str(candidate_dir),
            "report": str(candidate_report_path),
            "gate_report": str(gate_path),
            "passed": gate["passed"],
            "merge": merge_report,
            "metrics": gate["candidate"],
        }
        if recover_candidates and (not recover_only_failed or not gate["passed"]):
            recovery = _recover_and_gate_candidate(
                host_model=host_model,
                candidate_dir=candidate_dir,
                recovered_dir=recovered_root / f"xfamily_mlp_alpha_{tag}_recovered",
                recovered_report_path=report_root / f"xfamily_mlp_alpha_{tag}_recovered.json",
                recovered_gate_path=report_root / f"xfamily_mlp_alpha_{tag}_recovered_gate.json",
                baseline=baseline,
                torch_dtype=torch_dtype,
                device=device,
                trust_remote_code=trust_remote_code,
                max_new_tokens=max_new_tokens,
                max_ppl_ratio=max_ppl_ratio,
                max_accuracy_drop=max_accuracy_drop,
                min_choice_accuracy=min_choice_accuracy,
                recovery_steps=recovery_steps,
                recovery_batch_size=recovery_batch_size,
                recovery_max_length=recovery_max_length,
                recovery_lr=recovery_lr,
                recovery_kd_weight=recovery_kd_weight,
                recovery_lm_weight=recovery_lm_weight,
                recovery_temperature=recovery_temperature,
                recovery_train_scope=recovery_train_scope,
                recovery_dtype=recovery_dtype,
                recovery_grad_clip=recovery_grad_clip,
            )
            item["recovery"] = recovery
        candidates.append(item)

    summary = {
        "operation": "sweep_cross_family_mlp_merge",
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
        "max_host_channels": max_host_channels,
        "max_donor_channels": max_donor_channels,
        "recovery": {
            "enabled": recover_candidates,
            "recover_only_failed": recover_only_failed,
            "steps": recovery_steps,
            "batch_size": recovery_batch_size,
            "max_length": recovery_max_length,
            "lr": recovery_lr,
            "kd_weight": recovery_kd_weight,
            "lm_weight": recovery_lm_weight,
            "temperature": recovery_temperature,
            "train_scope": recovery_train_scope,
            "dtype": recovery_dtype,
            "grad_clip": recovery_grad_clip,
        },
        "gate_thresholds": {
            "max_ppl_ratio": max_ppl_ratio,
            "max_accuracy_drop": max_accuracy_drop,
            "min_choice_accuracy": min_choice_accuracy,
        },
        "candidates": candidates,
        "passed_candidates": [item for item in candidates if item["passed"]],
        "recovered_passed_candidates": [
            item for item in candidates if item.get("recovery", {}).get("passed")
        ],
    }
    _write_json(report_root / "xfamily_mlp_sweep_summary.json", summary)
    return summary


def cross_family_merge_compress_pipeline(
    host_model: str,
    donor_model: str,
    output_dir: str | Path,
    alpha: float = 0.01,
    vocab_alpha: float | None = None,
    ridge: float = 1e-3,
    max_anchors: int = 50_000,
    max_host_channels: int = 512,
    max_donor_channels: int = 512,
    width_ratios: list[float] | None = None,
    layer_group_specs: list[list[list[int]]] | None = None,
    depth_methods: list[str] | None = None,
    target_shrink: float | None = None,
    assignment: str = "combined-cosine",
    include_combinations: bool = True,
    baseline_report: str | Path | None = None,
    torch_dtype: str = "float16",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    max_new_tokens: int = 80,
    max_ppl_ratio: float = 1.5,
    max_accuracy_drop: float = 0.25,
    min_choice_accuracy: float = 0.75,
    recover_candidates: bool = False,
    recover_only_failed: bool = False,
    recover_top_k: int = 1,
    recovery_steps: int = 40,
    recovery_batch_size: int = 1,
    recovery_max_length: int = 96,
    recovery_lr: float = 5e-5,
    recovery_kd_weight: float = 0.5,
    recovery_lm_weight: float = 0.5,
    recovery_temperature: float = 2.0,
    recovery_train_scope: str = "last1",
    recovery_dtype: str = "float32",
    recovery_grad_clip: float = 1.0,
) -> dict[str, Any]:
    """Run cross-family MLP merge, merge-first compression, and optional recovery."""

    if recover_top_k <= 0:
        raise ValueError("recover_top_k must be positive.")

    root = Path(output_dir)
    model_root = root / "models"
    report_root = root / "reports"
    recovered_root = root / "recovered_models"
    compression_root = root / "compression"
    model_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    if recover_candidates:
        recovered_root.mkdir(parents=True, exist_ok=True)

    if baseline_report is None:
        host_baseline_path = report_root / "host_baseline.json"
        host_baseline = evaluate_text_model(
            host_model,
            host_baseline_path,
            torch_dtype=torch_dtype,
            device=device,
            trust_remote_code=trust_remote_code,
            max_new_tokens=max_new_tokens,
        )
    else:
        host_baseline_path = Path(baseline_report)
        host_baseline = _load_report(host_baseline_path)

    xfamily_dir = model_root / "xfamily_merged"
    xfamily_merge_report = cross_family_mlp_merge(
        host_model=host_model,
        donor_model=donor_model,
        output_dir=xfamily_dir,
        alpha=alpha,
        vocab_alpha=vocab_alpha,
        ridge=ridge,
        max_anchors=max_anchors,
        max_host_channels=max_host_channels,
        max_donor_channels=max_donor_channels,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    xfamily_report_path = report_root / "xfamily_merged.json"
    xfamily_gate_path = report_root / "xfamily_merged_host_gate.json"
    xfamily_eval = evaluate_text_model(
        str(xfamily_dir),
        xfamily_report_path,
        torch_dtype=torch_dtype,
        device=device,
        trust_remote_code=trust_remote_code,
        max_new_tokens=max_new_tokens,
    )
    xfamily_host_gate = gate_text_report(
        host_baseline,
        xfamily_eval,
        output_file=xfamily_gate_path,
        max_ppl_ratio=max_ppl_ratio,
        max_accuracy_drop=max_accuracy_drop,
        min_choice_accuracy=min_choice_accuracy,
    )

    compression_summary = search_merge_first_compression(
        source_model=str(xfamily_dir),
        output_dir=compression_root,
        width_ratios=width_ratios,
        layer_group_specs=layer_group_specs,
        depth_methods=depth_methods or ["average", "fold"],
        baseline_report=xfamily_report_path,
        torch_dtype=torch_dtype,
        device=device,
        trust_remote_code=trust_remote_code,
        max_new_tokens=max_new_tokens,
        max_ppl_ratio=max_ppl_ratio,
        max_accuracy_drop=max_accuracy_drop,
        min_choice_accuracy=min_choice_accuracy,
        assignment=assignment,
        include_combinations=include_combinations,
        target_shrink=target_shrink,
    )

    selected = _select_compression_candidates(compression_summary, target_shrink)
    final_candidates = []
    for candidate in selected:
        host_gate_path = report_root / f"{candidate['tag']}_host_gate.json"
        candidate_eval = _load_report(candidate["report"])
        host_gate = gate_text_report(
            host_baseline,
            candidate_eval,
            output_file=host_gate_path,
            max_ppl_ratio=max_ppl_ratio,
            max_accuracy_drop=max_accuracy_drop,
            min_choice_accuracy=min_choice_accuracy,
        )
        final_candidates.append(
            {
                **candidate,
                "host_gate_report": str(host_gate_path),
                "host_passed": host_gate["passed"],
                "host_metrics": host_gate["candidate"],
            }
        )

    recovered_candidates = []
    if recover_candidates:
        for candidate in final_candidates[:recover_top_k]:
            if recover_only_failed and candidate["host_passed"]:
                continue
            tag = str(candidate["tag"])
            recovery = _recover_and_gate_candidate(
                host_model=host_model,
                candidate_dir=Path(candidate["model_dir"]),
                recovered_dir=recovered_root / f"{tag}_recovered",
                recovered_report_path=report_root / f"{tag}_recovered.json",
                recovered_gate_path=report_root / f"{tag}_recovered_host_gate.json",
                baseline=host_baseline,
                torch_dtype=torch_dtype,
                device=device,
                trust_remote_code=trust_remote_code,
                max_new_tokens=max_new_tokens,
                max_ppl_ratio=max_ppl_ratio,
                max_accuracy_drop=max_accuracy_drop,
                min_choice_accuracy=min_choice_accuracy,
                recovery_steps=recovery_steps,
                recovery_batch_size=recovery_batch_size,
                recovery_max_length=recovery_max_length,
                recovery_lr=recovery_lr,
                recovery_kd_weight=recovery_kd_weight,
                recovery_lm_weight=recovery_lm_weight,
                recovery_temperature=recovery_temperature,
                recovery_train_scope=recovery_train_scope,
                recovery_dtype=recovery_dtype,
                recovery_grad_clip=recovery_grad_clip,
            )
            recovered_xfamily_gate_path = report_root / f"{tag}_recovered_xfamily_gate.json"
            recovered_xfamily_gate = gate_text_report(
                xfamily_eval,
                _load_report(recovery["report"]),
                output_file=recovered_xfamily_gate_path,
                max_ppl_ratio=max_ppl_ratio,
                max_accuracy_drop=max_accuracy_drop,
                min_choice_accuracy=min_choice_accuracy,
            )
            recovery["xfamily_gate_report"] = str(recovered_xfamily_gate_path)
            recovery["xfamily_passed"] = recovered_xfamily_gate["passed"]
            recovery["xfamily_metrics"] = recovered_xfamily_gate["candidate"]
            recovered_candidates.append(
                {
                    "source_candidate_tag": tag,
                    "source_candidate": candidate,
                    **recovery,
                }
            )

    usable_candidates = _usable_pipeline_candidates(final_candidates, recovered_candidates)

    summary = {
        "operation": "cross_family_merge_compress_pipeline",
        "host_model": host_model,
        "donor_model": donor_model,
        "output_dir": str(root),
        "host_baseline_report": str(host_baseline_path),
        "host_baseline_metrics": {
            "param_count": host_baseline.get("param_count"),
            "perplexity_mean": float(host_baseline["perplexity"]["mean"]),
            "choice_accuracy": float(host_baseline["choice_accuracy"]["accuracy"]),
        },
        "cross_family_merge": {
            "model_dir": str(xfamily_dir),
            "merge_report": xfamily_merge_report,
            "report": str(xfamily_report_path),
            "gate_report": str(xfamily_gate_path),
            "host_passed": xfamily_host_gate["passed"],
            "metrics": xfamily_host_gate["candidate"],
        },
        "compression": compression_summary,
        "selected_candidates": final_candidates,
        "recovered_candidates": recovered_candidates,
        "usable_candidates": usable_candidates,
        "best_usable_candidate": usable_candidates[0] if usable_candidates else None,
        "recovery": {
            "enabled": recover_candidates,
            "recover_only_failed": recover_only_failed,
            "top_k": recover_top_k,
            "steps": recovery_steps,
            "batch_size": recovery_batch_size,
            "max_length": recovery_max_length,
            "lr": recovery_lr,
            "kd_weight": recovery_kd_weight,
            "lm_weight": recovery_lm_weight,
            "temperature": recovery_temperature,
            "train_scope": recovery_train_scope,
            "dtype": recovery_dtype,
            "grad_clip": recovery_grad_clip,
        },
        "gate_thresholds": {
            "max_ppl_ratio": max_ppl_ratio,
            "max_accuracy_drop": max_accuracy_drop,
            "min_choice_accuracy": min_choice_accuracy,
        },
    }
    _write_json(report_root / "xfamily_compress_summary.json", summary)
    return summary


def _recover_and_gate_candidate(
    host_model: str,
    candidate_dir: Path,
    recovered_dir: Path,
    recovered_report_path: Path,
    recovered_gate_path: Path,
    baseline: dict[str, Any],
    torch_dtype: str,
    device: str,
    trust_remote_code: bool,
    max_new_tokens: int,
    max_ppl_ratio: float,
    max_accuracy_drop: float,
    min_choice_accuracy: float,
    recovery_steps: int,
    recovery_batch_size: int,
    recovery_max_length: int,
    recovery_lr: float,
    recovery_kd_weight: float,
    recovery_lm_weight: float,
    recovery_temperature: float,
    recovery_train_scope: str,
    recovery_dtype: str,
    recovery_grad_clip: float,
) -> dict[str, Any]:
    from neural_forge.hf_recovery import recover_causal_lm

    recovery_report = recover_causal_lm(
        source_model=host_model,
        candidate_model=str(candidate_dir),
        output_dir=recovered_dir,
        steps=recovery_steps,
        batch_size=recovery_batch_size,
        max_length=recovery_max_length,
        lr=recovery_lr,
        kd_weight=recovery_kd_weight,
        lm_weight=recovery_lm_weight,
        temperature=recovery_temperature,
        train_scope=recovery_train_scope,
        torch_dtype=recovery_dtype,
        device=device,
        trust_remote_code=trust_remote_code,
        grad_clip=recovery_grad_clip,
    )
    recovered_eval = evaluate_text_model(
        str(recovered_dir),
        recovered_report_path,
        torch_dtype=torch_dtype,
        device=device,
        trust_remote_code=trust_remote_code,
        max_new_tokens=max_new_tokens,
    )
    recovered_gate = gate_text_report(
        baseline,
        recovered_eval,
        output_file=recovered_gate_path,
        max_ppl_ratio=max_ppl_ratio,
        max_accuracy_drop=max_accuracy_drop,
        min_choice_accuracy=min_choice_accuracy,
    )
    return {
        "model_dir": str(recovered_dir),
        "recovery_report": recovery_report,
        "report": str(recovered_report_path),
        "gate_report": str(recovered_gate_path),
        "passed": recovered_gate["passed"],
        "metrics": recovered_gate["candidate"],
    }


def _select_compression_candidates(summary: dict[str, Any], target_shrink: float | None) -> list[dict[str, Any]]:
    if target_shrink is not None and summary.get("target_passed_candidates"):
        return list(summary["target_passed_candidates"])
    if target_shrink is not None and summary.get("best_passed_below_target"):
        return [summary["best_passed_below_target"]]
    if summary.get("best_passed_candidate"):
        return [summary["best_passed_candidate"]]
    return []


def _usable_pipeline_candidates(
    compressed_candidates: list[dict[str, Any]],
    recovered_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    usable = []
    for candidate in compressed_candidates:
        if candidate.get("host_passed"):
            usable.append(
                {
                    "source": "compressed",
                    "tag": candidate["tag"],
                    "model_dir": candidate["model_dir"],
                    "report": candidate["report"],
                    "gate_report": candidate["host_gate_report"],
                    "shrink": candidate["shrink"],
                    "metrics": candidate["host_metrics"],
                }
            )
    for candidate in recovered_candidates:
        if candidate.get("passed"):
            source_candidate = candidate["source_candidate"]
            usable.append(
                {
                    "source": "recovered",
                    "tag": f"{source_candidate['tag']}_recovered",
                    "model_dir": candidate["model_dir"],
                    "report": candidate["report"],
                    "gate_report": candidate["gate_report"],
                    "shrink": source_candidate["shrink"],
                    "metrics": candidate["metrics"],
                }
            )
    return sorted(
        usable,
        key=lambda item: (float(item["shrink"]), -float(item["metrics"]["perplexity_mean"])),
        reverse=True,
    )


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


def _bounded_alpha(value: float, name: str) -> float:
    alpha = _finite_float(value, name)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return alpha


def _shared_anchor_tokens(host_tokenizer, donor_tokenizer, max_anchors: int) -> tuple[list[str], dict[str, int], dict[str, int]]:
    host_map = _canonical_token_map(host_tokenizer)
    donor_map = _canonical_token_map(donor_tokenizer)
    shared = sorted(set(host_map) & set(donor_map), key=lambda item: (len(item), item))
    if not shared:
        raise ValueError("No shared decoded-token anchors found.")
    anchors = shared[: min(max_anchors, len(shared))]
    return anchors, host_map, donor_map


def _merge_lm_head_rows(torch, host, donor, host_idx, donor_idx, hidden_map, alpha: float, ridge: float) -> None:
    host_lm = host.get_output_embeddings()
    donor_lm = donor.get_output_embeddings()
    if host_lm is None or donor_lm is None:
        return
    host_head = host_lm.weight.data.detach().cpu().to(torch.float32)
    donor_head = donor_lm.weight.data.detach().cpu().to(torch.float32)
    head_map = _ridge_map(torch, donor_head[donor_idx], host_head[host_idx], ridge)
    projected_head = donor_head[donor_idx].matmul(head_map)
    host_head[host_idx] = (1.0 - alpha) * host_head[host_idx] + alpha * projected_head
    host_lm.weight.data.copy_(host_head.to(host_lm.weight.dtype))


def _compatible_mlp_layers(model) -> list[Any]:
    try:
        layers = list(_decoder_layer_container(model)[2])
    except ValueError:
        return []
    return [layer for layer in layers if _mlp_spec(layer) is not None]


def _mlp_spec(layer) -> dict[str, Any] | None:
    mlp = getattr(layer, "mlp", None)
    feed_forward = getattr(layer, "feed_forward", None)
    if feed_forward is not None and all(hasattr(feed_forward, name) for name in ("w1", "w3", "w2")):
        return {
            "kind": "gated",
            "mlp": feed_forward,
            "inputs": [feed_forward.w1, feed_forward.w3],
            "input_names": ["w1", "w3"],
            "output": feed_forward.w2,
            "output_name": "w2",
        }
    if mlp is None:
        return None
    if all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
        return {
            "kind": "gated",
            "mlp": mlp,
            "inputs": [mlp.gate_proj, mlp.up_proj],
            "input_names": ["gate_proj", "up_proj"],
            "output": mlp.down_proj,
            "output_name": "down_proj",
        }
    if all(hasattr(mlp, name) for name in ("dense_h_to_4h", "dense_4h_to_h")):
        return {
            "kind": "dense",
            "mlp": mlp,
            "inputs": [mlp.dense_h_to_4h],
            "input_names": ["dense_h_to_4h"],
            "output": mlp.dense_4h_to_h,
            "output_name": "dense_4h_to_h",
        }
    if all(hasattr(mlp, name) for name in ("fc1", "fc2")):
        return {
            "kind": "dense",
            "mlp": mlp,
            "inputs": [mlp.fc1],
            "input_names": ["fc1"],
            "output": mlp.fc2,
            "output_name": "fc2",
        }
    return None


def _merge_mlp_layer_channels(
    torch,
    host_layer,
    donor_layer,
    hidden_map,
    alpha: float,
    max_host_channels: int,
    max_donor_channels: int,
) -> dict[str, Any]:
    host_spec = _mlp_spec(host_layer)
    donor_spec = _mlp_spec(donor_layer)
    if host_spec is None or donor_spec is None:
        raise ValueError("Both layers must have supported MLP modules.")
    host_channels = int(host_spec["inputs"][0].weight.shape[0])
    donor_channels = int(donor_spec["inputs"][0].weight.shape[0])
    host_idx = _important_channels(torch, host_spec, min(max_host_channels, host_channels), None)
    donor_idx = _important_channels(torch, donor_spec, min(max_donor_channels, donor_channels), hidden_map)
    host_features = _mlp_channel_features(torch, host_spec, host_idx, None)
    donor_features = _mlp_channel_features(torch, donor_spec, donor_idx, hidden_map)
    assignment = _assign_donor_to_host(torch, donor_features, host_features)

    donor_input_branches = [_project_rows(torch, linear.weight.data.detach().cpu().to(torch.float32)[donor_idx], hidden_map) for linear in donor_spec["inputs"]]
    donor_input_summary = _average_rows(torch, donor_input_branches)
    donor_output_cols = _project_rows(
        torch,
        donor_spec["output"].weight.data.detach().cpu().to(torch.float32)[:, donor_idx].T,
        hidden_map,
    )

    host_positions = {int(channel): pos for pos, channel in enumerate(host_idx.tolist())}
    assigned_counts = torch.zeros(len(host_idx), dtype=torch.float32)
    input_sums = torch.zeros((len(host_idx), host_spec["inputs"][0].weight.shape[1]), dtype=torch.float32)
    output_sums = torch.zeros((len(host_idx), host_spec["output"].weight.shape[0]), dtype=torch.float32)
    for donor_pos, host_pos in enumerate(assignment.tolist()):
        input_sums[host_pos] += donor_input_summary[donor_pos]
        output_sums[host_pos] += donor_output_cols[donor_pos]
        assigned_counts[host_pos] += 1.0

    updated = 0
    for host_channel in host_idx.tolist():
        host_pos = host_positions[int(host_channel)]
        count = assigned_counts[host_pos]
        if count <= 0:
            continue
        donor_input = input_sums[host_pos] / count
        donor_output = output_sums[host_pos] / count
        for host_input in host_spec["inputs"]:
            weight = host_input.weight.data.detach().cpu().to(torch.float32)
            weight[int(host_channel)] = (1.0 - alpha) * weight[int(host_channel)] + alpha * donor_input
            host_input.weight.data.copy_(weight.to(host_input.weight.dtype))
        output_weight = host_spec["output"].weight.data.detach().cpu().to(torch.float32)
        output_weight[:, int(host_channel)] = (1.0 - alpha) * output_weight[:, int(host_channel)] + alpha * donor_output
        host_spec["output"].weight.data.copy_(output_weight.to(host_spec["output"].weight.dtype))
        updated += 1

    _merge_input_biases(torch, host_spec, donor_spec, host_idx, donor_idx, assignment, alpha)
    _merge_output_bias(torch, host_spec, donor_spec, hidden_map, alpha)
    return {
        "host_mlp_kind": host_spec["kind"],
        "donor_mlp_kind": donor_spec["kind"],
        "host_channels": host_channels,
        "donor_channels": donor_channels,
        "host_channels_considered": int(host_idx.numel()),
        "donor_channels_considered": int(donor_idx.numel()),
        "host_channels_updated": updated,
    }


def _important_channels(torch, spec: dict[str, Any], limit: int, hidden_map) -> Any:
    input_scores = sum(linear.weight.data.detach().cpu().to(torch.float32).norm(dim=1) for linear in spec["inputs"])
    output_scores = spec["output"].weight.data.detach().cpu().to(torch.float32).norm(dim=0)
    scores = input_scores + output_scores
    return torch.sort(torch.argsort(scores, descending=True)[:limit]).values


def _mlp_channel_features(torch, spec: dict[str, Any], indices, hidden_map) -> Any:
    input_rows = [_project_rows(torch, linear.weight.data.detach().cpu().to(torch.float32)[indices], hidden_map) for linear in spec["inputs"]]
    input_summary = _average_rows(torch, [_row_normalize(rows) for rows in input_rows])
    output_cols = _project_rows(
        torch,
        spec["output"].weight.data.detach().cpu().to(torch.float32)[:, indices].T,
        hidden_map,
    )
    return _row_normalize(torch.cat([_row_normalize(input_summary), _row_normalize(output_cols)], dim=1))


def _project_rows(torch, rows, hidden_map):
    rows = rows.to(torch.float32)
    if hidden_map is None:
        return rows
    return rows.matmul(hidden_map.to(torch.float32))


def _average_rows(torch, row_sets: list[Any]) -> Any:
    stacked = torch.stack(row_sets, dim=0)
    return stacked.mean(dim=0)


def _row_normalize(rows):
    return rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _assign_donor_to_host(torch, donor_features, host_features):
    assignments = []
    chunk = 128
    for start in range(0, donor_features.shape[0], chunk):
        sims = donor_features[start : start + chunk] @ host_features.T
        assignments.append(torch.argmax(sims, dim=1))
    return torch.cat(assignments, dim=0)


def _merge_input_biases(torch, host_spec, donor_spec, host_idx, donor_idx, assignment, alpha: float) -> None:
    donor_biases = [getattr(linear, "bias", None) for linear in donor_spec["inputs"]]
    host_biases = [getattr(linear, "bias", None) for linear in host_spec["inputs"]]
    if not any(bias is not None for bias in donor_biases) or not any(bias is not None for bias in host_biases):
        return
    donor_values = [bias.data.detach().cpu().to(torch.float32)[donor_idx] for bias in donor_biases if bias is not None]
    donor_summary = _average_rows(torch, [values.unsqueeze(1) for values in donor_values]).squeeze(1)
    sums = torch.zeros(len(host_idx), dtype=torch.float32)
    counts = torch.zeros(len(host_idx), dtype=torch.float32)
    for donor_pos, host_pos in enumerate(assignment.tolist()):
        sums[host_pos] += donor_summary[donor_pos]
        counts[host_pos] += 1.0
    for host_bias in host_biases:
        if host_bias is None:
            continue
        values = host_bias.data.detach().cpu().to(torch.float32)
        for host_pos, host_channel in enumerate(host_idx.tolist()):
            if counts[host_pos] > 0:
                donor_value = sums[host_pos] / counts[host_pos]
                values[int(host_channel)] = (1.0 - alpha) * values[int(host_channel)] + alpha * donor_value
        host_bias.data.copy_(values.to(host_bias.dtype))


def _merge_output_bias(torch, host_spec, donor_spec, hidden_map, alpha: float) -> None:
    host_bias = getattr(host_spec["output"], "bias", None)
    donor_bias = getattr(donor_spec["output"], "bias", None)
    if host_bias is None or donor_bias is None:
        return
    donor_projected = _project_rows(torch, donor_bias.data.detach().cpu().to(torch.float32).unsqueeze(0), hidden_map).squeeze(0)
    host_values = host_bias.data.detach().cpu().to(torch.float32)
    host_values = (1.0 - alpha) * host_values + alpha * donor_projected
    host_bias.data.copy_(host_values.to(host_bias.dtype))


def _proportional_index(index: int, host_count: int, donor_count: int) -> int:
    if donor_count <= 1 or host_count <= 1:
        return 0
    return int(round(index * (donor_count - 1) / (host_count - 1)))


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
