"""Constrained cross-family Hugging Face model fusion utilities.

This module handles the first practical cross-family case: keep a host runtime
architecture, then transplant donor embedding/lm-head signal through shared
decoded-token anchors. It is not blind whole-checkpoint averaging.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

from neural_forge.hf_tools import (
    _decoder_layer_container,
    _dtype_from_name,
    _ensure_unique_ratio_tags,
    _finite_float,
    _load_report,
    _merge_layer_state_dicts,
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


def donor_projected_init(
    target_config_model: str,
    donor_model: str,
    output_dir: str | Path,
    target_tokenizer_model: str | None = None,
    seed: int = 0,
    ridge: float = 1e-3,
    max_anchors: int = 50_000,
    fold_mode: str = "mean",
    activation_fold_assignment: str = "balanced",
    donor_layer_fold_mode: str = "representative",
    attention_fold_mode: str = "independent",
    token_init_mode: str = "shared",
    projection_scale: str = "none",
    activation_calibration_texts: list[str] | None = None,
    activation_max_length: int = 128,
    activation_device: str = "cuda:0",
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Initialize a target architecture by folding donor tensors into its shape.

    This is the blank-target counterpart to cross-family MLP merging. The target
    runtime comes from `target_config_model`, but layer tensors are initialized
    directly from proportionally matched donor layers. Each mismatched tensor
    axis is mean-folded or copied to the target size, which is a deterministic
    neuron/channel merge rather than prune-first deletion.
    """

    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if max_anchors <= 0:
        raise ValueError("max_anchors must be positive.")
    if fold_mode not in {"mean", "rms"}:
        raise ValueError("fold_mode must be mean or rms.")
    if activation_fold_assignment not in {"balanced", "similarity"}:
        raise ValueError("activation_fold_assignment must be balanced or similarity.")
    if donor_layer_fold_mode not in {"representative", "average", "fold"}:
        raise ValueError("donor_layer_fold_mode must be representative, average, or fold.")
    if attention_fold_mode not in {"independent", "coupled"}:
        raise ValueError("attention_fold_mode must be independent or coupled.")
    if token_init_mode not in {"shared", "donor-tokenized"}:
        raise ValueError("token_init_mode must be shared or donor-tokenized.")
    if projection_scale not in {"none", "target-rms"}:
        raise ValueError("projection_scale must be none or target-rms.")
    if activation_max_length <= 1:
        raise ValueError("activation_max_length must be greater than 1.")

    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    torch.manual_seed(int(seed))
    config = transformers.AutoConfig.from_pretrained(target_config_model, trust_remote_code=trust_remote_code)
    target = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    donor = transformers.AutoModelForCausalLM.from_pretrained(
        donor_model,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    tokenizer_source = target_tokenizer_model or target_config_model
    target_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)
    donor_tokenizer = transformers.AutoTokenizer.from_pretrained(donor_model, trust_remote_code=trust_remote_code)
    if target.get_input_embeddings() is not None and target.get_input_embeddings().num_embeddings != len(target_tokenizer):
        target.resize_token_embeddings(len(target_tokenizer))
    if dtype is not None:
        target.to(dtype=dtype)

    target_layers = list(_decoder_layer_container(target)[2])
    donor_layers = list(_decoder_layer_container(donor)[2])
    if not target_layers:
        raise ValueError("Target model has no decoder layers.")
    if not donor_layers:
        raise ValueError("Donor model has no decoder layers.")
    activation_profile = None
    if activation_calibration_texts:
        activation_profile = _collect_donor_axis_activation_scores(
            torch,
            donor,
            donor_tokenizer,
            donor_layers,
            activation_calibration_texts,
            max_length=activation_max_length,
            device=activation_device,
        )

    target_state = target.state_dict()
    donor_state = donor.state_dict()
    donor_layer_keys = _layer_tensor_index(donor_state)
    folded_donor_layer_cache: dict[int, dict[str, Any]] = {}
    layer_reports = []
    changed_keys = []
    skipped_layer_keys = []
    hidden_axis_weights = None
    if activation_profile is not None:
        hidden_axis_weights = {
            "weights": activation_profile["hidden_scores"],
            "features": activation_profile.get("hidden_features"),
        }
    for target_key, target_tensor in list(target_state.items()):
        parsed = _parse_layer_tensor_key(target_key)
        if parsed is None:
            continue
        target_layer_idx, suffix = parsed
        donor_layer_idx = _proportional_index(target_layer_idx, len(target_layers), len(donor_layers))
        donor_layer_group = (
            [donor_layer_idx]
            if donor_layer_fold_mode == "representative"
            else _donor_layer_group_for_target(target_layer_idx, len(target_layers), len(donor_layers))
        )
        if donor_layer_fold_mode == "representative":
            donor_key = donor_layer_keys.get((donor_layer_idx, suffix))
            donor_tensor = None if donor_key is None else donor_state[donor_key]
        else:
            donor_key = f"folded_donor_layers.{target_layer_idx}.{suffix}"
            folded_state = _folded_donor_layer_state(
                torch,
                donor_layers,
                target_layer_idx=target_layer_idx,
                target_layer_count=len(target_layers),
                donor_layer_fold_mode=donor_layer_fold_mode,
                cache=folded_donor_layer_cache,
            )
            donor_tensor = folded_state.get(suffix)
        if donor_tensor is None:
            skipped_layer_keys.append(
                {
                    "target_key": target_key,
                    "reason": "no_matching_donor_suffix",
                    "donor_layer_group": donor_layer_group,
                    "donor_layer_fold_mode": donor_layer_fold_mode,
                }
            )
            continue
        if not _projectable_tensor(torch, target_tensor, donor_tensor):
            skipped_layer_keys.append({"target_key": target_key, "donor_key": donor_key, "reason": "not_floating_tensor"})
            continue
        tensor_fold_mode = fold_mode if donor_tensor.dim() >= 2 else "mean"
        axis_weights = _activation_axis_weights_for_tensor(
            torch,
            donor_tensor,
            suffix,
            donor_layer_idx,
            donor_layer_group,
            activation_profile,
        )
        projected = None
        if attention_fold_mode == "coupled":
            projected = _resize_attention_tensor_to_shape(
                torch,
                donor_tensor.detach().cpu().to(torch.float32),
                tuple(target_tensor.shape),
                suffix,
                donor.config,
                target.config,
                fold_mode=tensor_fold_mode,
                axis_weights=axis_weights,
                activation_fold_assignment=activation_fold_assignment,
            )
        if projected is None:
            projected = _resize_tensor_to_shape(
                torch,
                donor_tensor.detach().cpu().to(torch.float32),
                tuple(target_tensor.shape),
                fold_mode=tensor_fold_mode,
                axis_weights=axis_weights,
                activation_fold_assignment=activation_fold_assignment,
            )
        projected = _scale_projection_to_target(torch, projected, target_tensor, projection_scale)
        target_state[target_key] = projected.to(dtype=target_tensor.dtype)
        changed_keys.append(target_key)
        layer_reports.append(
            {
                "target_key": target_key,
                "donor_key": donor_key,
                "target_layer": target_layer_idx,
                "donor_layer": donor_layer_idx,
                "donor_layer_group": donor_layer_group,
                "donor_layer_fold_mode": donor_layer_fold_mode,
                "attention_fold_mode": attention_fold_mode,
                "target_shape": list(target_tensor.shape),
                "donor_shape": list(donor_tensor.shape),
            }
        )

    non_layer_report = _initialize_non_layer_tensors_from_donor(
        torch,
        target_state,
        donor_state,
        changed_keys=set(changed_keys),
        fold_mode=fold_mode,
        hidden_axis_weights=hidden_axis_weights,
        activation_fold_assignment=activation_fold_assignment,
        projection_scale=projection_scale,
    )
    changed_keys.extend(item["target_key"] for item in non_layer_report["initialized"])
    load_result = target.load_state_dict(target_state, strict=False)
    token_report = _initialize_token_weights_from_donor(
        torch,
        target,
        donor,
        target_tokenizer,
        donor_tokenizer,
        max_anchors=max_anchors,
        token_init_mode=token_init_mode,
        fold_mode=fold_mode,
        hidden_axis_weights=hidden_axis_weights,
        activation_fold_assignment=activation_fold_assignment,
        projection_scale=projection_scale,
    )
    if getattr(target.config, "tie_word_embeddings", False):
        target.tie_weights()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target.save_pretrained(output, safe_serialization=True)
    target_tokenizer.save_pretrained(output)
    report = {
        "operation": "donor_projected_init",
        "target_config_model": target_config_model,
        "target_tokenizer_model": tokenizer_source,
        "donor_model": donor_model,
        "output_dir": str(output),
        "seed": int(seed),
        "ridge": float(ridge),
        "max_anchors": int(max_anchors),
        "fold_mode": fold_mode,
        "activation_fold_assignment": activation_fold_assignment,
        "donor_layer_fold_mode": donor_layer_fold_mode,
        "attention_fold_mode": attention_fold_mode,
        "token_init_mode": token_init_mode,
        "projection_scale": projection_scale,
        "activation_calibration_text_count": len(activation_calibration_texts or []),
        "activation_max_length": activation_max_length,
        "activation_device": activation_device if activation_calibration_texts else None,
        "torch_dtype": torch_dtype,
        "target_layers": len(target_layers),
        "donor_layers": len(donor_layers),
        "layer_tensors_initialized": len(layer_reports),
        "layer_tensors_skipped": skipped_layer_keys[:200],
        "non_layer_tensors_initialized": len(non_layer_report["initialized"]),
        "non_layer_tensors_skipped": non_layer_report["skipped"][:200],
        "non_layer_report": non_layer_report,
        "layer_reports": layer_reports,
        "changed_keys": changed_keys,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "token_report": token_report,
        "param_count": sum(param.numel() for param in target.parameters()),
    }
    _write_json(output / "neural_forge_donor_projected_init_report.json", report)
    return report


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
    channel_assignment: str = "weight-cosine",
    activation_calibration_texts: list[str] | None = None,
    activation_max_length: int = 128,
    activation_device: str = "cuda:0",
    layer_alpha_mode: str = "constant",
    layer_alpha_min_scale: float = 0.0,
    layer_alpha_max_scale: float = 2.0,
    layer_alpha_similarity_floor: float = 0.0,
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
    _validate_channel_assignment(channel_assignment)
    _validate_layer_alpha_settings(
        layer_alpha_mode,
        layer_alpha_min_scale,
        layer_alpha_max_scale,
        layer_alpha_similarity_floor,
    )
    if activation_max_length <= 1:
        raise ValueError("activation_max_length must be greater than 1.")

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

    host_activation_features = None
    donor_activation_features = None
    if _needs_activation_assignment(channel_assignment):
        if not activation_calibration_texts:
            raise ValueError("Activation channel assignment requires calibration texts.")
        host_activation_features = _collect_mlp_activation_features(
            torch,
            host,
            host_tokenizer,
            host_layers,
            activation_calibration_texts,
            max_length=activation_max_length,
            device=activation_device,
        )
        donor_activation_features = _collect_mlp_activation_features(
            torch,
            donor,
            donor_tokenizer,
            donor_layers,
            activation_calibration_texts,
            max_length=activation_max_length,
            device=activation_device,
        )

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
            channel_assignment=channel_assignment,
            host_activation_features=None if host_activation_features is None else host_activation_features[host_idx],
            donor_activation_features=None if donor_activation_features is None else donor_activation_features[donor_idx],
            layer_alpha_mode=layer_alpha_mode,
            layer_alpha_min_scale=layer_alpha_min_scale,
            layer_alpha_max_scale=layer_alpha_max_scale,
            layer_alpha_similarity_floor=layer_alpha_similarity_floor,
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
        "channel_assignment": channel_assignment,
        "activation_calibration_text_count": len(activation_calibration_texts or []),
        "activation_max_length": activation_max_length,
        "activation_device": activation_device if _needs_activation_assignment(channel_assignment) else None,
        "layer_alpha_mode": layer_alpha_mode,
        "layer_alpha_min_scale": layer_alpha_min_scale,
        "layer_alpha_max_scale": layer_alpha_max_scale,
        "layer_alpha_similarity_floor": layer_alpha_similarity_floor,
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
    channel_assignment: str = "weight-cosine",
    activation_calibration_texts: list[str] | None = None,
    activation_max_length: int = 128,
    activation_device: str = "cuda:0",
    layer_alpha_mode: str = "constant",
    layer_alpha_min_scale: float = 0.0,
    layer_alpha_max_scale: float = 2.0,
    layer_alpha_similarity_floor: float = 0.0,
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
    recovery_calibration_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Sweep cross-family MLP-channel transplants and gate each result."""

    if not alphas:
        raise ValueError("At least one alpha is required.")
    _validate_channel_assignment(channel_assignment)
    _validate_layer_alpha_settings(
        layer_alpha_mode,
        layer_alpha_min_scale,
        layer_alpha_max_scale,
        layer_alpha_similarity_floor,
    )
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
            channel_assignment=channel_assignment,
            activation_calibration_texts=activation_calibration_texts,
            activation_max_length=activation_max_length,
            activation_device=activation_device,
            layer_alpha_mode=layer_alpha_mode,
            layer_alpha_min_scale=layer_alpha_min_scale,
            layer_alpha_max_scale=layer_alpha_max_scale,
            layer_alpha_similarity_floor=layer_alpha_similarity_floor,
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
                recovery_calibration_texts=recovery_calibration_texts,
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
        "channel_assignment": channel_assignment,
        "activation_calibration_text_count": len(activation_calibration_texts or []),
        "activation_max_length": activation_max_length,
        "activation_device": activation_device if _needs_activation_assignment(channel_assignment) else None,
        "layer_alpha_mode": layer_alpha_mode,
        "layer_alpha_min_scale": layer_alpha_min_scale,
        "layer_alpha_max_scale": layer_alpha_max_scale,
        "layer_alpha_similarity_floor": layer_alpha_similarity_floor,
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
            "calibration_text_count": len(recovery_calibration_texts or []),
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
    channel_assignment: str = "weight-cosine",
    activation_calibration_texts: list[str] | None = None,
    activation_max_length: int = 128,
    activation_device: str = "cuda:0",
    layer_alpha_mode: str = "constant",
    layer_alpha_min_scale: float = 0.0,
    layer_alpha_max_scale: float = 2.0,
    layer_alpha_similarity_floor: float = 0.0,
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
    recovery_calibration_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Run cross-family MLP merge, merge-first compression, and optional recovery."""

    if recover_top_k <= 0:
        raise ValueError("recover_top_k must be positive.")
    _validate_channel_assignment(channel_assignment)
    _validate_layer_alpha_settings(
        layer_alpha_mode,
        layer_alpha_min_scale,
        layer_alpha_max_scale,
        layer_alpha_similarity_floor,
    )

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
        channel_assignment=channel_assignment,
        activation_calibration_texts=activation_calibration_texts,
        activation_max_length=activation_max_length,
        activation_device=activation_device,
        layer_alpha_mode=layer_alpha_mode,
        layer_alpha_min_scale=layer_alpha_min_scale,
        layer_alpha_max_scale=layer_alpha_max_scale,
        layer_alpha_similarity_floor=layer_alpha_similarity_floor,
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
                recovery_calibration_texts=recovery_calibration_texts,
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
        "channel_assignment": channel_assignment,
        "activation_calibration_text_count": len(activation_calibration_texts or []),
        "activation_max_length": activation_max_length,
        "activation_device": activation_device if _needs_activation_assignment(channel_assignment) else None,
        "layer_alpha_mode": layer_alpha_mode,
        "layer_alpha_min_scale": layer_alpha_min_scale,
        "layer_alpha_max_scale": layer_alpha_max_scale,
        "layer_alpha_similarity_floor": layer_alpha_similarity_floor,
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
            "calibration_text_count": len(recovery_calibration_texts or []),
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
    recovery_calibration_texts: list[str] | None = None,
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
        calibration_texts=recovery_calibration_texts,
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


def _layer_tensor_index(state_dict: dict[str, Any]) -> dict[tuple[int, str], str]:
    indexed = {}
    for key in state_dict:
        parsed = _parse_layer_tensor_key(key)
        if parsed is not None:
            indexed[parsed] = key
    return indexed


def _parse_layer_tensor_key(key: str) -> tuple[int, str] | None:
    match = re.match(r"^.+(?:layers|h)\.(\d+)\.(.+)$", key)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def _projectable_tensor(torch, target_tensor, donor_tensor) -> bool:
    return (
        hasattr(target_tensor, "is_floating_point")
        and hasattr(donor_tensor, "is_floating_point")
        and target_tensor.is_floating_point()
        and donor_tensor.is_floating_point()
        and target_tensor.dim() == donor_tensor.dim()
    )


def _initialize_non_layer_tensors_from_donor(
    torch,
    target_state: dict[str, Any],
    donor_state: dict[str, Any],
    changed_keys: set[str] | None = None,
    fold_mode: str = "mean",
    hidden_axis_weights=None,
    activation_fold_assignment: str = "balanced",
    projection_scale: str = "none",
) -> dict[str, Any]:
    donor_index = _non_layer_tensor_index(donor_state)
    already_changed = set(changed_keys or set())
    initialized = []
    skipped = []
    for target_key, target_tensor in list(target_state.items()):
        if target_key in already_changed or _is_excluded_non_layer_tensor_key(target_key):
            continue
        candidates = _non_layer_donor_candidates(target_key, donor_index)
        if not candidates:
            skipped.append({"target_key": target_key, "reason": "no_matching_donor_non_layer_tensor"})
            continue
        donor_key = _select_non_layer_donor_candidate(torch, target_tensor, donor_state, candidates)
        if donor_key is None:
            skipped.append(
                {
                    "target_key": target_key,
                    "candidate_donor_keys": candidates[:8],
                    "reason": "ambiguous_or_incompatible_donor_non_layer_tensor",
                }
            )
            continue
        donor_tensor = donor_state[donor_key]
        if not _projectable_tensor(torch, target_tensor, donor_tensor):
            skipped.append({"target_key": target_key, "donor_key": donor_key, "reason": "not_floating_tensor"})
            continue
        tensor_fold_mode = fold_mode if donor_tensor.dim() >= 2 else "mean"
        axis_weights = _hidden_axis_weights_for_non_layer_tensor(donor_tensor, hidden_axis_weights)
        projected = _resize_tensor_to_shape(
            torch,
            donor_tensor.detach().cpu().to(torch.float32),
            tuple(target_tensor.shape),
            fold_mode=tensor_fold_mode,
            axis_weights=axis_weights,
            activation_fold_assignment=activation_fold_assignment,
        )
        projected = _scale_projection_to_target(torch, projected, target_tensor, projection_scale)
        target_state[target_key] = projected.to(dtype=target_tensor.dtype)
        initialized.append(
            {
                "target_key": target_key,
                "donor_key": donor_key,
                "target_shape": list(target_tensor.shape),
                "donor_shape": list(donor_tensor.shape),
            }
        )
    return {
        "initialized": initialized,
        "skipped": skipped,
        "donor_index_keys": len(donor_index),
    }


def _non_layer_tensor_index(state_dict: dict[str, Any]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for key in state_dict:
        if _is_excluded_non_layer_tensor_key(key):
            continue
        for alias in _non_layer_tensor_aliases(key):
            index.setdefault(alias, []).append(key)
    return index


def _non_layer_donor_candidates(target_key: str, donor_index: dict[str, list[str]]) -> list[str]:
    candidates = []
    seen = set()
    for alias in _non_layer_tensor_aliases(target_key):
        for donor_key in donor_index.get(alias, []):
            if donor_key not in seen:
                candidates.append(donor_key)
                seen.add(donor_key)
    return candidates


def _select_non_layer_donor_candidate(torch, target_tensor, donor_state: dict[str, Any], candidates: Sequence[str]) -> str | None:
    compatible = [key for key in candidates if _projectable_tensor(torch, target_tensor, donor_state[key])]
    if len(compatible) == 1:
        return compatible[0]
    exact_shape = [key for key in compatible if tuple(donor_state[key].shape) == tuple(target_tensor.shape)]
    if len(exact_shape) == 1:
        return exact_shape[0]
    return None


def _is_excluded_non_layer_tensor_key(key: str) -> bool:
    if _parse_layer_tensor_key(key) is not None:
        return True
    lowered = key.lower()
    excluded_markers = (
        "embed_tokens",
        "embed_in",
        "embed_out",
        "word_embeddings",
        "wte",
        "lm_head",
        "output_projection",
        "rotary_emb",
        "rope",
        "inv_freq",
    )
    return any(marker in lowered for marker in excluded_markers)


def _non_layer_tensor_aliases(key: str) -> list[str]:
    aliases = [key]
    canonical = _canonical_non_layer_tensor_key(key)
    if canonical is not None and canonical not in aliases:
        aliases.append(canonical)
    return aliases


def _canonical_non_layer_tensor_key(key: str) -> str | None:
    parts = key.split(".")
    if not parts:
        return None
    leaf = parts[-1]
    if leaf not in {"weight", "bias"}:
        return ".".join(parts[-2:]) if len(parts) >= 2 else key
    lowered = key.lower()
    final_norm_markers = (
        ".norm.",
        ".ln_f.",
        ".norm_f.",
        ".final_norm.",
        ".final_layer_norm.",
    )
    padded = f".{lowered}"
    if any(marker in padded for marker in final_norm_markers):
        return f"final_norm.{leaf}"
    return ".".join(parts[-2:]) if len(parts) >= 2 else key


def _hidden_axis_weights_for_non_layer_tensor(donor_tensor, hidden_axis_weights):
    if hidden_axis_weights is None:
        return None
    weights = hidden_axis_weights.get("weights") if isinstance(hidden_axis_weights, dict) else None
    if weights is None:
        return None
    axis_weights = {}
    for axis, source_dim in enumerate(donor_tensor.shape):
        if int(source_dim) == int(weights.numel()):
            axis_weights[axis] = hidden_axis_weights
    return axis_weights or None


def _scale_projection_to_target(torch, projected, target_tensor, projection_scale: str):
    if projection_scale == "none":
        return projected
    if projection_scale != "target-rms":
        raise ValueError("projection_scale must be none or target-rms.")
    target_values = target_tensor.detach().cpu().to(torch.float32)
    projected_values = projected.to(torch.float32)
    target_rms = target_values.pow(2).mean().sqrt()
    projected_rms = projected_values.pow(2).mean().sqrt()
    return projected_values * (target_rms / projected_rms.clamp_min(1e-12))


def _donor_layer_group_for_target(target_layer_idx: int, target_layer_count: int, donor_layer_count: int) -> list[int]:
    if target_layer_count <= 0 or donor_layer_count <= 0:
        raise ValueError("Layer counts must be positive.")
    start = int(math.floor(int(target_layer_idx) * donor_layer_count / target_layer_count))
    end = int(math.floor((int(target_layer_idx) + 1) * donor_layer_count / target_layer_count))
    if end <= start:
        end = start + 1
    start = max(0, min(start, donor_layer_count - 1))
    end = max(start + 1, min(end, donor_layer_count))
    return list(range(start, end))


def _folded_donor_layer_state(
    torch,
    donor_layers: Sequence[Any],
    target_layer_idx: int,
    target_layer_count: int,
    donor_layer_fold_mode: str,
    cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if donor_layer_fold_mode not in {"average", "fold"}:
        raise ValueError("donor_layer_fold_mode must be average or fold for folded donor states.")
    if target_layer_idx not in cache:
        group = _donor_layer_group_for_target(target_layer_idx, target_layer_count, len(donor_layers))
        method = "average" if donor_layer_fold_mode == "average" else "fold"
        cache[target_layer_idx] = _merge_layer_state_dicts(
            torch,
            [donor_layers[layer_idx].state_dict() for layer_idx in group],
            method=method,
        )
    return cache[target_layer_idx]


def _resize_tensor_to_shape(
    torch,
    source,
    target_shape: tuple[int, ...],
    fold_mode: str = "mean",
    axis_weights: dict[int, Any] | None = None,
    activation_fold_assignment: str = "balanced",
):
    if fold_mode not in {"mean", "rms"}:
        raise ValueError("fold_mode must be mean or rms.")
    if activation_fold_assignment not in {"balanced", "similarity"}:
        raise ValueError("activation_fold_assignment must be balanced or similarity.")
    if tuple(source.shape) == tuple(target_shape):
        return source.clone()
    if source.dim() != len(target_shape):
        raise ValueError(f"Cannot resize rank-{source.dim()} tensor to shape {target_shape}.")
    resized = source.to(torch.float32)
    for axis, target_dim in enumerate(target_shape):
        if resized.shape[axis] != int(target_dim):
            profile = None if axis_weights is None else axis_weights.get(axis)
            weights, features = _axis_fold_profile(profile)
            resized = _resize_tensor_axis_mean_fold(
                torch,
                resized,
                axis,
                int(target_dim),
                fold_mode=fold_mode,
                weights=weights,
                features=features,
                assignment_mode=activation_fold_assignment,
            )
    return resized.contiguous()


def _resize_attention_tensor_to_shape(
    torch,
    source,
    target_shape: tuple[int, ...],
    suffix: str,
    donor_config,
    target_config,
    fold_mode: str = "mean",
    axis_weights: dict[int, Any] | None = None,
    activation_fold_assignment: str = "balanced",
):
    specs = _attention_partition_specs(suffix, donor_config, target_config)
    if not specs or source.dim() != len(target_shape):
        return None
    resized = source.to(torch.float32)
    handled_axes = set()
    used_coupled_axis = False
    for axis, source_heads, target_heads in specs:
        if axis >= resized.dim() or int(resized.shape[axis]) == int(target_shape[axis]):
            continue
        coupled = _resize_head_partitioned_axis(
            torch,
            resized,
            axis,
            int(target_shape[axis]),
            source_heads,
            target_heads,
            fold_mode=fold_mode,
        )
        if coupled is None:
            continue
        resized = coupled
        handled_axes.add(axis)
        used_coupled_axis = True
    if not used_coupled_axis:
        return None
    for axis, target_dim in enumerate(target_shape):
        if resized.shape[axis] != int(target_dim):
            profile = None if axis_weights is None or axis in handled_axes else axis_weights.get(axis)
            weights, features = _axis_fold_profile(profile)
            resized = _resize_tensor_axis_mean_fold(
                torch,
                resized,
                axis,
                int(target_dim),
                fold_mode=fold_mode,
                weights=weights,
                features=features,
                assignment_mode=activation_fold_assignment,
            )
    return resized.contiguous()


def _attention_partition_specs(suffix: str, donor_config, target_config) -> list[tuple[int, int, int]]:
    q_heads_source = _config_int(donor_config, "num_attention_heads", "n_head", "num_heads")
    q_heads_target = _config_int(target_config, "num_attention_heads", "n_head", "num_heads")
    kv_heads_source = _config_int(donor_config, "num_key_value_heads", default=q_heads_source)
    kv_heads_target = _config_int(target_config, "num_key_value_heads", default=q_heads_target)
    if q_heads_source is None or q_heads_target is None:
        return []
    marker = f".{suffix}"
    if marker.endswith(".q_proj.weight") or marker.endswith(".q_proj.bias"):
        return [(0, q_heads_source, q_heads_target)]
    if marker.endswith(".k_proj.weight") or marker.endswith(".k_proj.bias"):
        return [] if kv_heads_source is None or kv_heads_target is None else [(0, kv_heads_source, kv_heads_target)]
    if marker.endswith(".v_proj.weight") or marker.endswith(".v_proj.bias"):
        return [] if kv_heads_source is None or kv_heads_target is None else [(0, kv_heads_source, kv_heads_target)]
    if marker.endswith(".o_proj.weight"):
        return [(1, q_heads_source, q_heads_target)]
    return []


def _config_int(config, *names: str, default=None):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return None if default is None else int(default)


def _resize_head_partitioned_axis(
    torch,
    tensor,
    axis: int,
    target_dim: int,
    source_heads: int,
    target_heads: int,
    fold_mode: str,
):
    source_dim = int(tensor.shape[axis])
    if source_heads <= 0 or target_heads <= 0:
        return None
    if source_dim % int(source_heads) != 0 or target_dim % int(target_heads) != 0:
        return None
    source_head_dim = source_dim // int(source_heads)
    target_head_dim = target_dim // int(target_heads)
    moved = tensor.movedim(axis, 0).contiguous()
    rest_shape = tuple(moved.shape[1:])
    partitioned = moved.reshape(int(source_heads), source_head_dim, *rest_shape)
    partitioned = _resize_tensor_axis_mean_fold(
        torch,
        partitioned,
        axis=0,
        target_dim=int(target_heads),
        fold_mode=fold_mode,
    )
    partitioned = _resize_tensor_axis_mean_fold(
        torch,
        partitioned,
        axis=1,
        target_dim=target_head_dim,
        fold_mode=fold_mode,
    )
    resized = partitioned.reshape(target_dim, *rest_shape)
    return resized.movedim(0, axis).contiguous()


def _axis_fold_profile(profile):
    if isinstance(profile, dict):
        return profile.get("weights"), profile.get("features")
    return profile, None


def _resize_tensor_axis_mean_fold(
    torch,
    tensor,
    axis: int,
    target_dim: int,
    fold_mode: str = "mean",
    weights=None,
    features=None,
    assignment_mode: str = "balanced",
):
    if target_dim <= 0:
        raise ValueError("target_dim must be positive.")
    if fold_mode not in {"mean", "rms"}:
        raise ValueError("fold_mode must be mean or rms.")
    if assignment_mode not in {"balanced", "similarity"}:
        raise ValueError("assignment_mode must be balanced or similarity.")
    source_dim = int(tensor.shape[axis])
    if source_dim == target_dim:
        return tensor

    moved = tensor.movedim(axis, 0).contiguous()
    rest_shape = tuple(moved.shape[1:])
    flat = moved.reshape(source_dim, -1)
    if source_dim > target_dim:
        explicit_weights = weights is not None
        if weights is None and features is None:
            assignment = torch.floor(torch.arange(source_dim, dtype=torch.float32) * target_dim / source_dim)
            assignment = assignment.to(torch.long).clamp_max(target_dim - 1)
            fold_weights = torch.ones(source_dim, dtype=flat.dtype)
        else:
            if weights is None:
                fold_weights = torch.ones(source_dim, dtype=torch.float32)
            else:
                fold_weights = weights.detach().cpu().to(torch.float32).flatten()
            if int(fold_weights.numel()) != source_dim:
                raise ValueError("Axis weights must match the source dimension.")
            fold_weights = fold_weights.clamp_min(1e-12)
            if assignment_mode == "similarity" and features is not None:
                assignment = _activation_similarity_assignment(torch, features, target_dim, fold_weights)
            else:
                assignment = _balanced_activation_assignment(torch, fold_weights, target_dim)
        folded = torch.zeros((target_dim, flat.shape[1]), dtype=flat.dtype)
        counts = torch.zeros(target_dim, dtype=flat.dtype)
        weighted_flat = flat * fold_weights.to(flat.dtype).unsqueeze(1)
        folded.index_add_(0, assignment, weighted_flat)
        counts.index_add_(0, assignment, fold_weights.to(flat.dtype))
        divisor = counts.sqrt() if fold_mode == "rms" and not explicit_weights else counts
        folded = folded / divisor.clamp_min(1.0).unsqueeze(1)
    else:
        source_idx = torch.floor(torch.arange(target_dim, dtype=torch.float32) * source_dim / target_dim)
        source_idx = source_idx.to(torch.long).clamp_max(source_dim - 1)
        folded = flat[source_idx]
    return folded.reshape((target_dim, *rest_shape)).movedim(0, axis).contiguous()


def _balanced_activation_assignment(torch, weights, target_dim: int):
    order = torch.argsort(weights, descending=True)
    assignment = torch.empty(int(weights.numel()), dtype=torch.long)
    for rank, source_idx in enumerate(order.tolist()):
        assignment[int(source_idx)] = int(rank % target_dim)
    return assignment


def _activation_similarity_assignment(torch, features, target_dim: int, weights=None, iterations: int = 8):
    if target_dim <= 0:
        raise ValueError("target_dim must be positive.")
    feature_rows = features.detach().cpu().to(torch.float32)
    if feature_rows.dim() != 2:
        raise ValueError("Activation features must be a rank-2 tensor.")
    source_dim = int(feature_rows.shape[0])
    if source_dim == 0:
        raise ValueError("Activation features cannot be empty.")
    if source_dim < target_dim:
        raise ValueError("target_dim cannot exceed activation feature rows.")
    if weights is None:
        fold_weights = torch.ones(source_dim, dtype=torch.float32)
    else:
        fold_weights = weights.detach().cpu().to(torch.float32).flatten()
        if int(fold_weights.numel()) != source_dim:
            raise ValueError("Activation assignment weights must match feature rows.")
        fold_weights = fold_weights.clamp_min(1e-12)
    if target_dim == 1:
        return torch.zeros(source_dim, dtype=torch.long)

    feature_rows = _row_normalize(feature_rows)
    first = int(torch.argmax(fold_weights).item())
    centroids = [feature_rows[first]]
    min_dist = _squared_distance_to_centroid(torch, feature_rows, centroids[0])
    while len(centroids) < target_dim:
        weighted_dist = min_dist * fold_weights
        next_idx = int(torch.argmax(weighted_dist).item())
        centroids.append(feature_rows[next_idx])
        min_dist = torch.minimum(min_dist, _squared_distance_to_centroid(torch, feature_rows, centroids[-1]))
    centroid_tensor = torch.stack(centroids, dim=0)
    assignment = torch.zeros(source_dim, dtype=torch.long)
    for _ in range(max(1, iterations)):
        scores = feature_rows.matmul(centroid_tensor.t())
        assignment = torch.argmax(scores, dim=1)
        assignment = _fill_empty_activation_clusters(torch, feature_rows, assignment, centroid_tensor, fold_weights)
        updated = []
        for cluster_idx in range(target_dim):
            mask = assignment == cluster_idx
            cluster_weights = fold_weights[mask].unsqueeze(1)
            centroid = (feature_rows[mask] * cluster_weights).sum(dim=0) / cluster_weights.sum().clamp_min(1e-12)
            updated.append(centroid)
        centroid_tensor = _row_normalize(torch.stack(updated, dim=0))
    return assignment.to(torch.long)


def _squared_distance_to_centroid(torch, feature_rows, centroid):
    diff = feature_rows - centroid.unsqueeze(0)
    return diff.pow(2).sum(dim=1)


def _fill_empty_activation_clusters(torch, feature_rows, assignment, centroids, weights):
    target_dim = int(centroids.shape[0])
    present = set(int(item) for item in assignment.unique().tolist())
    missing = [idx for idx in range(target_dim) if idx not in present]
    if not missing:
        return assignment
    scores = 1.0 - feature_rows.matmul(centroids.t()).max(dim=1).values
    candidate_order = torch.argsort(scores * weights, descending=True).tolist()
    assignment = assignment.clone()
    used = set()
    for cluster_idx in missing:
        for source_idx in candidate_order:
            if int(source_idx) in used:
                continue
            current_cluster = int(assignment[int(source_idx)].item())
            if int((assignment == current_cluster).sum().item()) <= 1:
                continue
            assignment[int(source_idx)] = cluster_idx
            used.add(int(source_idx))
            break
    return assignment


def _collect_donor_axis_activation_scores(
    torch,
    donor,
    donor_tokenizer,
    donor_layers: list[Any],
    texts: list[str],
    max_length: int,
    device: str,
) -> dict[str, Any]:
    if not texts:
        raise ValueError("Need at least one activation calibration text.")
    hidden_scores = []
    hidden_features = []
    mlp_scores: dict[int, list[Any]] = {idx: [] for idx in range(len(donor_layers))}
    mlp_features: dict[int, list[Any]] = {idx: [] for idx in range(len(donor_layers))}
    handles = []

    def hidden_hook(_module, inputs):
        hidden = inputs[0].detach().to(torch.float32)
        hidden_scores.append(_activation_abs_score(torch, hidden).cpu())
        hidden_features.append(_activation_channel_summary(torch, _activation_2d(torch, hidden)).cpu())

    def mlp_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            values = tensor.detach().to(torch.float32)
            mlp_scores[layer_idx].append(_activation_abs_score(torch, values).cpu())
            mlp_features[layer_idx].append(_activation_channel_summary(torch, _activation_2d(torch, values)).cpu())

        return hook

    for layer_idx, layer in enumerate(donor_layers):
        handles.append(layer.register_forward_pre_hook(hidden_hook))
        spec = _mlp_spec(layer)
        if spec is None:
            continue
        for linear in spec["inputs"]:
            handles.append(linear.register_forward_hook(mlp_hook(layer_idx)))

    was_training = donor.training
    try:
        donor.eval()
        donor.to(device)
        with torch.no_grad():
            for text in texts:
                encoded = donor_tokenizer(str(text), return_tensors="pt", truncation=True, max_length=max_length)
                encoded = {key: value.to(device) for key, value in encoded.items()}
                donor(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        donor.to("cpu")
        if device.startswith("cuda") and hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
        if was_training:
            donor.train()

    if not hidden_scores:
        raise RuntimeError("Activation calibration did not capture hidden states.")
    folded_mlp_scores = {}
    folded_mlp_features = {}
    for layer_idx, scores in mlp_scores.items():
        if scores:
            folded_mlp_scores[layer_idx] = torch.stack(scores, dim=0).mean(dim=0).clamp_min(1e-12)
    for layer_idx, features in mlp_features.items():
        if features:
            folded_mlp_features[layer_idx] = torch.stack(features, dim=0).mean(dim=0)
    return {
        "hidden_scores": torch.stack(hidden_scores, dim=0).mean(dim=0).clamp_min(1e-12),
        "hidden_features": torch.stack(hidden_features, dim=0).mean(dim=0),
        "mlp_scores": folded_mlp_scores,
        "mlp_features": folded_mlp_features,
    }


def _activation_abs_score(torch, tensor):
    values = _activation_2d(torch, tensor)
    return values.abs().mean(dim=0)


def _activation_2d(torch, tensor):
    values = tensor
    if values.dim() == 3:
        values = values.reshape(-1, values.shape[-1])
    elif values.dim() == 1:
        values = values.unsqueeze(0)
    elif values.dim() != 2:
        values = values.reshape(-1, values.shape[-1])
    return values


def _activation_axis_weights_for_tensor(
    torch,
    donor_tensor,
    suffix: str,
    donor_layer_idx: int,
    donor_layer_group: Sequence[int] | None,
    activation_profile,
):
    if activation_profile is None:
        return None
    axis_weights = {}
    hidden_scores = activation_profile["hidden_scores"]
    hidden_features = activation_profile.get("hidden_features")
    layer_group = list(donor_layer_group or [donor_layer_idx])
    mlp_scores = _aggregate_layer_activation_profile(
        torch,
        activation_profile.get("mlp_scores", {}),
        layer_group,
        clamp_min=1e-12,
    )
    mlp_features = _aggregate_layer_activation_profile(
        torch,
        activation_profile.get("mlp_features", {}),
        layer_group,
    )
    for axis, source_dim in enumerate(donor_tensor.shape):
        if int(source_dim) == int(hidden_scores.numel()):
            axis_weights[axis] = {"weights": hidden_scores, "features": hidden_features}
    suffix_marker = f".{suffix}."
    if mlp_scores is not None and (".mlp." in suffix_marker or ".feed_forward." in suffix_marker):
        for axis, source_dim in enumerate(donor_tensor.shape):
            if int(source_dim) == int(mlp_scores.numel()):
                axis_weights[axis] = {"weights": mlp_scores, "features": mlp_features}
    return axis_weights or None


def _aggregate_layer_activation_profile(torch, profiles: dict[int, Any], layer_group: Sequence[int], clamp_min=None):
    rows = [profiles[int(layer_idx)] for layer_idx in layer_group if int(layer_idx) in profiles]
    if not rows:
        return None
    stacked = torch.stack([row.detach().cpu().to(torch.float32) for row in rows], dim=0)
    merged = stacked.mean(dim=0)
    if clamp_min is not None:
        merged = merged.clamp_min(float(clamp_min))
    return merged


def _initialize_token_weights_from_donor(
    torch,
    target,
    donor,
    target_tokenizer,
    donor_tokenizer,
    max_anchors: int,
    token_init_mode: str = "shared",
    fold_mode: str = "mean",
    hidden_axis_weights=None,
    activation_fold_assignment: str = "balanced",
    projection_scale: str = "none",
) -> dict[str, Any]:
    if token_init_mode not in {"shared", "donor-tokenized"}:
        raise ValueError("token_init_mode must be shared or donor-tokenized.")
    anchors, target_map, donor_map = _shared_anchor_tokens(target_tokenizer, donor_tokenizer, max_anchors)
    if token_init_mode == "shared":
        target_ids = [target_map[token] for token in anchors]
        donor_id_groups = [[donor_map[token]] for token in anchors]
        tokenized_rows = 0
    else:
        target_ids, donor_id_groups, tokenized_rows = _target_to_donor_token_groups(
            target_tokenizer,
            donor_tokenizer,
            target_map,
            donor_map,
        )

    target_embeddings = target.get_input_embeddings()
    donor_embeddings = donor.get_input_embeddings()
    target_emb_weight = target_embeddings.weight.data.detach().cpu().to(torch.float32)
    donor_emb_weight = donor_embeddings.weight.data.detach().cpu().to(torch.float32)
    embedding_rows_initialized = _project_token_weight_rows(
        torch,
        target_emb_weight,
        donor_emb_weight,
        target_ids,
        donor_id_groups,
        fold_mode=fold_mode,
        hidden_axis_weights=hidden_axis_weights,
        activation_fold_assignment=activation_fold_assignment,
        projection_scale=projection_scale,
        chunk_size=2048,
    )
    target_embeddings.weight.data.copy_(target_emb_weight.to(target_embeddings.weight.dtype))

    target_lm = target.get_output_embeddings()
    donor_lm = donor.get_output_embeddings()
    lm_head_updated = False
    lm_head_rows_initialized = 0
    if target_lm is not None and donor_lm is not None:
        target_head = target_lm.weight.data.detach().cpu().to(torch.float32)
        donor_head = donor_lm.weight.data.detach().cpu().to(torch.float32)
        lm_head_rows_initialized = _project_token_weight_rows(
            torch,
            target_head,
            donor_head,
            target_ids,
            donor_id_groups,
            fold_mode=fold_mode,
            hidden_axis_weights=hidden_axis_weights,
            activation_fold_assignment=activation_fold_assignment,
            projection_scale=projection_scale,
            chunk_size=2048,
        )
        target_lm.weight.data.copy_(target_head.to(target_lm.weight.dtype))
        lm_head_updated = True

    return {
        "mode": token_init_mode,
        "shared_decoded_tokens": len(set(_canonical_token_map(target_tokenizer)) & set(_canonical_token_map(donor_tokenizer))),
        "anchors_used": len(anchors),
        "tokenized_rows_initialized": int(tokenized_rows),
        "embedding_rows_initialized": int(embedding_rows_initialized),
        "lm_head_rows_initialized": int(lm_head_rows_initialized) if lm_head_updated else 0,
        "lm_head_updated": lm_head_updated,
    }


def _project_token_weight_rows(
    torch,
    target_weight,
    donor_weight,
    target_ids: Sequence[int],
    donor_id_groups: Sequence[Sequence[int]],
    fold_mode: str,
    hidden_axis_weights=None,
    activation_fold_assignment: str = "balanced",
    projection_scale: str = "none",
    chunk_size: int = 2048,
) -> int:
    if len(target_ids) != len(donor_id_groups):
        raise ValueError("target_ids and donor_id_groups must have the same length.")
    if not target_ids:
        return 0
    target_dim = int(target_weight.shape[1])
    for start in range(0, len(target_ids), int(chunk_size)):
        end = min(start + int(chunk_size), len(target_ids))
        chunk_target_ids = list(target_ids[start:end])
        donor_rows = torch.stack(
            [donor_weight[list(group)].mean(dim=0) for group in donor_id_groups[start:end]],
            dim=0,
        )
        projected = _resize_tensor_to_shape(
            torch,
            donor_rows,
            (len(chunk_target_ids), target_dim),
            fold_mode=fold_mode,
            axis_weights={} if hidden_axis_weights is None else {1: hidden_axis_weights},
            activation_fold_assignment=activation_fold_assignment,
        )
        projected = _scale_projection_to_target(
            torch,
            projected,
            target_weight[torch.tensor(chunk_target_ids, dtype=torch.long)],
            projection_scale,
        )
        target_weight[torch.tensor(chunk_target_ids, dtype=torch.long)] = projected
    return int(len(target_ids))


def _target_to_donor_token_groups(
    target_tokenizer,
    donor_tokenizer,
    target_map: dict[str, int],
    donor_map: dict[str, int],
) -> tuple[list[int], list[list[int]], int]:
    target_special_ids = set(getattr(target_tokenizer, "all_special_ids", []) or [])
    donor_special_ids = set(getattr(donor_tokenizer, "all_special_ids", []) or [])
    donor_vocab_size = len(donor_tokenizer)
    target_ids = []
    donor_id_groups = []
    tokenized_rows = 0
    seen_target_ids = set()
    for token, target_idx in sorted(target_map.items(), key=lambda item: item[1]):
        donor_idx = donor_map.get(token)
        if donor_idx is None:
            continue
        target_ids.append(int(target_idx))
        donor_id_groups.append([int(donor_idx)])
        seen_target_ids.add(int(target_idx))
    for target_idx in range(len(target_tokenizer)):
        if int(target_idx) in seen_target_ids or int(target_idx) in target_special_ids:
            continue
        decoded = _decode_single_token(target_tokenizer, int(target_idx))
        if not decoded or any(ord(ch) < 32 for ch in decoded):
            continue
        donor_ids = _encode_without_specials(donor_tokenizer, decoded)
        donor_ids = [
            int(item)
            for item in donor_ids
            if 0 <= int(item) < donor_vocab_size and int(item) not in donor_special_ids
        ]
        if not donor_ids:
            continue
        target_ids.append(int(target_idx))
        donor_id_groups.append(donor_ids)
        seen_target_ids.add(int(target_idx))
        tokenized_rows += 1
    return target_ids, donor_id_groups, tokenized_rows


def _decode_single_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return ""


def _encode_without_specials(tokenizer, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        try:
            encoded = tokenizer(text, add_special_tokens=False)
        except Exception:
            return []
        ids = encoded.get("input_ids", []) if isinstance(encoded, dict) else []
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)


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


def _validate_channel_assignment(channel_assignment: str) -> None:
    if channel_assignment not in {"weight-cosine", "activation-correlation", "hybrid"}:
        raise ValueError("channel_assignment must be weight-cosine, activation-correlation, or hybrid.")


def _needs_activation_assignment(channel_assignment: str) -> bool:
    return channel_assignment in {"activation-correlation", "hybrid"}


def _validate_layer_alpha_settings(
    layer_alpha_mode: str,
    min_scale: float,
    max_scale: float,
    similarity_floor: float,
) -> None:
    if layer_alpha_mode not in {"constant", "similarity"}:
        raise ValueError("layer_alpha_mode must be constant or similarity.")
    if min_scale < 0 or max_scale < 0:
        raise ValueError("layer alpha scales must be non-negative.")
    if min_scale > max_scale:
        raise ValueError("layer_alpha_min_scale must be <= layer_alpha_max_scale.")
    if not 0.0 <= similarity_floor < 1.0:
        raise ValueError("layer_alpha_similarity_floor must be in [0, 1).")


def _collect_mlp_activation_features(
    torch,
    model,
    tokenizer,
    layers: list[Any],
    texts: list[str],
    max_length: int,
    device: str,
) -> list[Any]:
    if not texts:
        raise ValueError("Need at least one activation calibration text.")
    specs = [_mlp_spec(layer) for layer in layers]
    if any(spec is None for spec in specs):
        raise ValueError("All layers must have supported MLP modules.")
    branch_stats: list[list[list[Any]]] = [
        [[] for _ in spec["inputs"]]  # type: ignore[index]
        for spec in specs
    ]
    handles = []

    def make_hook(layer_idx: int, branch_idx: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            tensor = tensor.detach().to(torch.float32)
            if tensor.dim() == 3:
                tensor = tensor.reshape(-1, tensor.shape[-1])
            elif tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() != 2:
                tensor = tensor.reshape(-1, tensor.shape[-1])
            branch_stats[layer_idx][branch_idx].append(_activation_channel_summary(torch, tensor).cpu())

        return hook

    for layer_idx, spec in enumerate(specs):
        for branch_idx, linear in enumerate(spec["inputs"]):  # type: ignore[index]
            handles.append(linear.register_forward_hook(make_hook(layer_idx, branch_idx)))

    was_training = model.training
    try:
        model.eval()
        model.to(device)
        with torch.no_grad():
            for text in texts:
                encoded = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=max_length)
                encoded = {key: value.to(device) for key, value in encoded.items()}
                model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        model.to("cpu")
        if device.startswith("cuda") and hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
        if was_training:
            model.train()

    features = []
    expected_count = len(texts)
    for layer_idx, spec in enumerate(specs):
        per_text = []
        for text_idx in range(expected_count):
            branch_items = []
            for branch_idx in range(len(spec["inputs"])):  # type: ignore[index]
                stats_for_branch = branch_stats[layer_idx][branch_idx]
                if len(stats_for_branch) != expected_count:
                    raise RuntimeError("Activation hook did not capture the expected number of batches.")
                branch_items.append(stats_for_branch[text_idx])
            per_text.append(_average_rows(torch, branch_items))
        features.append(_row_normalize(torch.cat(per_text, dim=1)))
    return features


def _activation_channel_summary(torch, values):
    values = values.to(torch.float32)
    mean = values.mean(dim=0)
    centered = values - mean.unsqueeze(0)
    std = centered.pow(2).mean(dim=0).sqrt()
    abs_mean = values.abs().mean(dim=0)
    positive_fraction = (values > 0).to(torch.float32).mean(dim=0)
    return torch.stack([mean, std, abs_mean, positive_fraction], dim=1)


def _merge_mlp_layer_channels(
    torch,
    host_layer,
    donor_layer,
    hidden_map,
    alpha: float,
    max_host_channels: int,
    max_donor_channels: int,
    channel_assignment: str = "weight-cosine",
    host_activation_features=None,
    donor_activation_features=None,
    layer_alpha_mode: str = "constant",
    layer_alpha_min_scale: float = 0.0,
    layer_alpha_max_scale: float = 2.0,
    layer_alpha_similarity_floor: float = 0.0,
) -> dict[str, Any]:
    host_spec = _mlp_spec(host_layer)
    donor_spec = _mlp_spec(donor_layer)
    if host_spec is None or donor_spec is None:
        raise ValueError("Both layers must have supported MLP modules.")
    host_channels = int(host_spec["inputs"][0].weight.shape[0])
    donor_channels = int(donor_spec["inputs"][0].weight.shape[0])
    host_idx = _important_channels(torch, host_spec, min(max_host_channels, host_channels), None)
    donor_idx = _important_channels(torch, donor_spec, min(max_donor_channels, donor_channels), hidden_map)
    host_weight_features = _mlp_channel_features(torch, host_spec, host_idx, None)
    donor_weight_features = _mlp_channel_features(torch, donor_spec, donor_idx, hidden_map)
    host_features, donor_features, activation_used = _channel_assignment_features(
        torch,
        channel_assignment,
        host_weight_features,
        donor_weight_features,
        None if host_activation_features is None else host_activation_features[host_idx],
        None if donor_activation_features is None else donor_activation_features[donor_idx],
    )
    assignment = _assign_donor_to_host(torch, donor_features, host_features)
    similarity = (donor_features * host_features[assignment]).sum(dim=1)
    layer_alpha, layer_alpha_scale = _effective_layer_alpha(
        alpha,
        float(similarity.mean().cpu()),
        layer_alpha_mode=layer_alpha_mode,
        min_scale=layer_alpha_min_scale,
        max_scale=layer_alpha_max_scale,
        similarity_floor=layer_alpha_similarity_floor,
    )

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
            weight[int(host_channel)] = (1.0 - layer_alpha) * weight[int(host_channel)] + layer_alpha * donor_input
            host_input.weight.data.copy_(weight.to(host_input.weight.dtype))
        output_weight = host_spec["output"].weight.data.detach().cpu().to(torch.float32)
        output_weight[:, int(host_channel)] = (1.0 - layer_alpha) * output_weight[:, int(host_channel)] + layer_alpha * donor_output
        host_spec["output"].weight.data.copy_(output_weight.to(host_spec["output"].weight.dtype))
        updated += 1

    _merge_input_biases(torch, host_spec, donor_spec, host_idx, donor_idx, assignment, layer_alpha)
    _merge_output_bias(torch, host_spec, donor_spec, hidden_map, layer_alpha)
    return {
        "host_mlp_kind": host_spec["kind"],
        "donor_mlp_kind": donor_spec["kind"],
        "host_channels": host_channels,
        "donor_channels": donor_channels,
        "host_channels_considered": int(host_idx.numel()),
        "donor_channels_considered": int(donor_idx.numel()),
        "host_channels_updated": updated,
        "channel_assignment": channel_assignment,
        "activation_features_used": activation_used,
        "assignment_similarity_mean": float(similarity.mean().cpu()),
        "assignment_similarity_min": float(similarity.min().cpu()),
        "assignment_similarity_max": float(similarity.max().cpu()),
        "base_alpha": float(alpha),
        "effective_alpha": float(layer_alpha),
        "layer_alpha_scale": float(layer_alpha_scale),
        "layer_alpha_mode": layer_alpha_mode,
    }


def _effective_layer_alpha(
    alpha: float,
    similarity_mean: float,
    layer_alpha_mode: str,
    min_scale: float,
    max_scale: float,
    similarity_floor: float,
) -> tuple[float, float]:
    _validate_layer_alpha_settings(layer_alpha_mode, min_scale, max_scale, similarity_floor)
    if layer_alpha_mode == "constant":
        return float(alpha), 1.0
    normalized = (float(similarity_mean) - similarity_floor) / max(1e-12, 1.0 - similarity_floor)
    normalized = min(1.0, max(0.0, normalized))
    scale = min_scale + (max_scale - min_scale) * normalized
    return min(1.0, max(0.0, float(alpha) * scale)), float(scale)


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


def _channel_assignment_features(
    torch,
    channel_assignment: str,
    host_weight_features,
    donor_weight_features,
    host_activation_features,
    donor_activation_features,
):
    _validate_channel_assignment(channel_assignment)
    if channel_assignment == "weight-cosine":
        return host_weight_features, donor_weight_features, False
    if host_activation_features is None or donor_activation_features is None:
        raise ValueError("Activation channel assignment requires activation features.")
    if host_activation_features.shape[1] != donor_activation_features.shape[1]:
        raise ValueError("Host and donor activation feature dimensions must match.")
    host_activation_features = _row_normalize(host_activation_features.to(torch.float32))
    donor_activation_features = _row_normalize(donor_activation_features.to(torch.float32))
    if channel_assignment == "activation-correlation":
        return host_activation_features, donor_activation_features, True
    return (
        _row_normalize(torch.cat([host_weight_features, host_activation_features], dim=1)),
        _row_normalize(torch.cat([donor_weight_features, donor_activation_features], dim=1)),
        True,
    )


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
