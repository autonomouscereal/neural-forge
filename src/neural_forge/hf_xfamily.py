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
    target_base_model: str | None = None,
    target_tokenizer_model: str | None = None,
    seed: int = 0,
    ridge: float = 1e-3,
    max_anchors: int = 50_000,
    fold_mode: str = "mean",
    hidden_basis_mode: str = "none",
    hidden_basis_scope: str = "global",
    hidden_basis_max_rows: int = 2048,
    hidden_basis_preprocess: str = "centered-rowrms",
    activation_fold_assignment: str = "balanced",
    transport_backend: str = "sparse",
    donor_layer_fold_mode: str = "representative",
    attention_fold_mode: str = "independent",
    attention_head_dim_transport: str = "none",
    token_init_mode: str = "shared",
    projection_scale: str = "none",
    projection_blend_alpha: float = 1.0,
    consumer_hidden_reduction: str = "mean",
    activation_fit_linears: bool = False,
    activation_fit_target: str = "all",
    activation_fit_max_rows: int = 0,
    activation_fit_alpha: float = 1.0,
    activation_fit_ridge: float = 1e-1,
    untie_output_head: bool = False,
    activation_calibration_texts: list[str] | None = None,
    activation_max_length: int = 128,
    activation_device: str = "cuda:0",
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Initialize a target architecture by folding donor tensors into its shape.

    This is the blank-target counterpart to cross-family MLP merging. The target
    runtime comes from `target_config_model`. If `target_base_model` is supplied,
    that checkpoint provides the target weights to blend into; otherwise the
    target starts from the config initializer. Layer tensors are initialized
    directly from proportionally matched donor layers. Each mismatched tensor
    axis is mean-folded or copied to the target size, which is a deterministic
    neuron/channel merge rather than prune-first deletion.
    """

    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if max_anchors <= 0:
        raise ValueError("max_anchors must be positive.")
    if fold_mode not in {"mean", "rms", "transport"}:
        raise ValueError("fold_mode must be mean, rms, or transport.")
    if hidden_basis_mode not in {"none", "activation-pca"}:
        raise ValueError("hidden_basis_mode must be none or activation-pca.")
    if hidden_basis_scope not in {"global", "layer"}:
        raise ValueError("hidden_basis_scope must be global or layer.")
    if hidden_basis_max_rows <= 0:
        raise ValueError("hidden_basis_max_rows must be positive.")
    if hidden_basis_preprocess not in {"centered-rowrms", "uncentered-rowrms", "uncentered"}:
        raise ValueError("hidden_basis_preprocess must be centered-rowrms, uncentered-rowrms, or uncentered.")
    if activation_fold_assignment not in {"balanced", "similarity"}:
        raise ValueError("activation_fold_assignment must be balanced or similarity.")
    if transport_backend not in {"sparse", "sinkhorn"}:
        raise ValueError("transport_backend must be sparse or sinkhorn.")
    if donor_layer_fold_mode not in {"representative", "average", "fold"}:
        raise ValueError("donor_layer_fold_mode must be representative, average, or fold.")
    if attention_fold_mode not in {"independent", "coupled", "activation-coupled"}:
        raise ValueError("attention_fold_mode must be independent, coupled, or activation-coupled.")
    if attention_head_dim_transport not in {"none", "qk-vo"}:
        raise ValueError("attention_head_dim_transport must be none or qk-vo.")
    if token_init_mode not in {"shared", "donor-tokenized", "donor-tokenized-weighted", "donor-omp"}:
        raise ValueError("token_init_mode must be shared, donor-tokenized, donor-tokenized-weighted, or donor-omp.")
    if projection_scale not in {"none", "target-rms"}:
        raise ValueError("projection_scale must be none or target-rms.")
    if consumer_hidden_reduction not in {"mean", "sum", "sqrt-sum"}:
        raise ValueError("consumer_hidden_reduction must be mean, sum, or sqrt-sum.")
    if activation_fit_target not in {"all", "attention", "attention-output", "mlp", "mlp-output", "input-projections"}:
        raise ValueError(
            "activation_fit_target must be all, attention, attention-output, mlp, mlp-output, or input-projections."
        )
    if activation_fit_max_rows < 0:
        raise ValueError("activation_fit_max_rows must be non-negative.")
    if activation_fit_ridge < 0:
        raise ValueError("activation_fit_ridge must be non-negative.")
    projection_blend_alpha = _bounded_alpha(projection_blend_alpha, "projection_blend_alpha")
    activation_fit_alpha = _bounded_alpha(activation_fit_alpha, "activation_fit_alpha")
    if activation_max_length <= 1:
        raise ValueError("activation_max_length must be greater than 1.")

    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    torch.manual_seed(int(seed))
    config = transformers.AutoConfig.from_pretrained(target_config_model, trust_remote_code=trust_remote_code)
    if untie_output_head and hasattr(config, "tie_word_embeddings"):
        config.tie_word_embeddings = False
    if target_base_model:
        target = transformers.AutoModelForCausalLM.from_pretrained(
            target_base_model,
            torch_dtype=dtype,
            device_map="cpu",
            trust_remote_code=trust_remote_code,
        )
    else:
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
    if untie_output_head:
        _untie_output_head(torch, target)
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
            hidden_basis_max_rows=hidden_basis_max_rows if hidden_basis_mode == "activation-pca" else 0,
            linear_fit_max_rows=activation_fit_max_rows if activation_fit_linears else 0,
        )
    if attention_fold_mode == "activation-coupled" and activation_profile is None:
        raise ValueError("activation-coupled attention folding requires activation calibration texts.")
    if attention_head_dim_transport != "none" and activation_profile is None:
        raise ValueError("attention head-dimension transport requires activation calibration texts.")
    if activation_fit_linears and activation_profile is None:
        raise ValueError("activation linear fitting requires activation calibration texts.")
    hidden_basis = None
    hidden_basis_report = None
    layer_hidden_bases = None
    layer_hidden_basis_report = None
    if hidden_basis_mode == "activation-pca":
        if activation_profile is None or activation_profile.get("hidden_basis_rows") is None:
            raise ValueError("activation-pca hidden basis requires activation calibration texts.")
        donor_hidden_size = int(donor.get_input_embeddings().weight.shape[1])
        target_hidden_size = int(target.get_input_embeddings().weight.shape[1])
        hidden_basis, hidden_basis_report = _build_activation_hidden_basis(
            torch,
            activation_profile["hidden_basis_rows"],
            source_dim=donor_hidden_size,
            target_dim=target_hidden_size,
            preprocess=hidden_basis_preprocess,
        )
        if hidden_basis_scope == "layer":
            layer_hidden_bases, layer_hidden_basis_report = _build_layer_activation_hidden_bases(
                torch,
                activation_profile,
                target_layer_count=len(target_layers),
                donor_layer_count=len(donor_layers),
                source_dim=donor_hidden_size,
                target_dim=target_hidden_size,
                preprocess=hidden_basis_preprocess,
            )

    target_state = target.state_dict()
    donor_state = donor.state_dict()
    donor_layer_keys = _layer_tensor_index(donor_state)
    folded_donor_layer_cache: dict[int, dict[str, Any]] = {}
    axis_assignment_cache: dict[Any, dict[str, Any]] = {}
    layer_reports = []
    changed_keys = []
    skipped_layer_keys = []
    activation_fit_reports = []
    hidden_axis_weights = None
    if activation_profile is not None:
        hidden_axis_weights = {
            "weights": activation_profile["hidden_scores"],
            "features": activation_profile.get("hidden_features"),
            "assignment_key": ("hidden", "residual"),
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
        axis_reductions = _axis_reductions_for_layer_tensor(suffix)
        axis_reductions = _with_consumer_hidden_reduction(axis_reductions, consumer_hidden_reduction)
        projected = None
        if attention_fold_mode in {"coupled", "activation-coupled"}:
            layer_axis_bases = _hidden_basis_axes_for_layer_tensor(
                torch,
                donor_tensor,
                suffix,
                None if layer_hidden_bases is None else layer_hidden_bases.get(int(target_layer_idx)),
            )
            attention_axis_weights = None
            if attention_fold_mode == "activation-coupled":
                attention_axis_weights = _attention_axis_weights_for_tensor(
                    torch,
                    donor_tensor,
                    suffix,
                    donor_layer_idx,
                    donor_layer_group,
                    activation_profile,
                )
            projected = _resize_attention_tensor_to_shape(
                torch,
                donor_tensor.detach().cpu().to(torch.float32),
                tuple(target_tensor.shape),
                suffix,
                donor.config,
                target.config,
                fold_mode=tensor_fold_mode,
                axis_bases=layer_axis_bases,
                hidden_basis=hidden_basis if donor_tensor.dim() >= 2 else None,
                attention_axis_weights=attention_axis_weights,
                attention_head_fold_mode="transport" if attention_fold_mode == "activation-coupled" else None,
                attention_head_dim_transport=attention_head_dim_transport,
                axis_weights=axis_weights,
                axis_reductions=axis_reductions,
                activation_fold_assignment=activation_fold_assignment,
                transport_backend=transport_backend,
                assignment_cache=axis_assignment_cache,
                semantic_prefix=("layer", int(target_layer_idx)),
            )
        if projected is None:
            layer_axis_bases = _hidden_basis_axes_for_layer_tensor(
                torch,
                donor_tensor,
                suffix,
                None if layer_hidden_bases is None else layer_hidden_bases.get(int(target_layer_idx)),
            )
            projected = _resize_tensor_to_shape(
                torch,
                donor_tensor.detach().cpu().to(torch.float32),
                tuple(target_tensor.shape),
                fold_mode=tensor_fold_mode,
                axis_bases=layer_axis_bases,
                hidden_basis=hidden_basis if donor_tensor.dim() >= 2 else None,
                axis_weights=axis_weights,
                axis_reductions=axis_reductions,
                activation_fold_assignment=activation_fold_assignment,
                transport_backend=transport_backend,
                assignment_cache=axis_assignment_cache,
            )
        projected = _scale_projection_to_target(torch, projected, target_tensor, projection_scale)
        projected = _blend_projection_with_target(torch, projected, target_tensor, projection_blend_alpha)
        activation_fit_report = None
        if (
            activation_fit_linears
            and target_tensor.dim() == 2
            and suffix.endswith(".weight")
            and _activation_fit_suffix_enabled(suffix, activation_fit_target)
        ):
            fitted = _fit_projected_linear_from_donor_activations(
                torch,
                activation_profile,
                suffix,
                tuple(target_tensor.shape),
                donor.config,
                target.config,
                donor_layer_idx,
                donor_layer_group,
                ridge=activation_fit_ridge,
                fold_mode=tensor_fold_mode,
                hidden_basis=hidden_basis if donor_tensor.dim() >= 2 else None,
                axis_weights=axis_weights,
                activation_fold_assignment=activation_fold_assignment,
                transport_backend=transport_backend,
                assignment_cache=axis_assignment_cache,
                reference_weight=projected,
            )
            if fitted is not None:
                projected = (1.0 - activation_fit_alpha) * projected + activation_fit_alpha * fitted
                activation_fit_report = {
                    "target_key": target_key,
                    "suffix": suffix,
                    "target_shape": list(target_tensor.shape),
                    "alpha": float(activation_fit_alpha),
                    "ridge": float(activation_fit_ridge),
                }
                activation_fit_reports.append(activation_fit_report)
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
                "activation_fit": activation_fit_report,
            }
        )

    non_layer_report = _initialize_non_layer_tensors_from_donor(
        torch,
        target_state,
        donor_state,
        changed_keys=set(changed_keys),
        fold_mode=fold_mode,
        hidden_basis=hidden_basis,
        hidden_axis_weights=hidden_axis_weights,
        activation_fold_assignment=activation_fold_assignment,
        transport_backend=transport_backend,
        projection_scale=projection_scale,
        projection_blend_alpha=projection_blend_alpha,
        consumer_hidden_reduction=consumer_hidden_reduction,
        assignment_cache=axis_assignment_cache,
    )
    changed_keys.extend(item["target_key"] for item in non_layer_report["initialized"])
    load_result = target.load_state_dict(target_state, strict=False)
    embedding_hidden_basis = hidden_basis
    lm_head_hidden_basis = hidden_basis
    if layer_hidden_bases:
        first_basis = layer_hidden_bases.get(0, {})
        last_basis = layer_hidden_bases.get(len(target_layers) - 1, {})
        embedding_hidden_basis = first_basis.get("input")
        if embedding_hidden_basis is None:
            embedding_hidden_basis = hidden_basis
        lm_head_hidden_basis = last_basis.get("output")
        if lm_head_hidden_basis is None:
            lm_head_hidden_basis = hidden_basis
    token_report = _initialize_token_weights_from_donor(
        torch,
        target,
        donor,
        target_tokenizer,
        donor_tokenizer,
        max_anchors=max_anchors,
        token_init_mode=token_init_mode,
        fold_mode=fold_mode,
        hidden_basis=embedding_hidden_basis,
        lm_head_hidden_basis=lm_head_hidden_basis,
        hidden_axis_weights=hidden_axis_weights,
        activation_fold_assignment=activation_fold_assignment,
        projection_scale=projection_scale,
        projection_blend_alpha=projection_blend_alpha,
        consumer_hidden_reduction=consumer_hidden_reduction,
        assignment_cache=axis_assignment_cache,
    )
    if getattr(target.config, "tie_word_embeddings", False) and not untie_output_head:
        target.tie_weights()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target.save_pretrained(output, safe_serialization=True)
    target_tokenizer.save_pretrained(output)
    report = {
        "operation": "donor_projected_init",
        "target_config_model": target_config_model,
        "target_base_model": target_base_model,
        "target_tokenizer_model": tokenizer_source,
        "donor_model": donor_model,
        "output_dir": str(output),
        "seed": int(seed),
        "ridge": float(ridge),
        "max_anchors": int(max_anchors),
        "fold_mode": fold_mode,
        "hidden_basis_mode": hidden_basis_mode,
        "hidden_basis_scope": hidden_basis_scope,
        "hidden_basis_max_rows": int(hidden_basis_max_rows),
        "hidden_basis_preprocess": hidden_basis_preprocess,
        "hidden_basis": hidden_basis_report,
        "layer_hidden_basis": layer_hidden_basis_report,
        "activation_fold_assignment": activation_fold_assignment,
        "transport_backend": transport_backend,
        "donor_layer_fold_mode": donor_layer_fold_mode,
        "attention_fold_mode": attention_fold_mode,
        "attention_head_dim_transport": attention_head_dim_transport,
        "token_init_mode": token_init_mode,
        "projection_scale": projection_scale,
        "projection_blend_alpha": projection_blend_alpha,
        "consumer_hidden_reduction": consumer_hidden_reduction,
        "activation_fit_linears": bool(activation_fit_linears),
        "activation_fit_target": activation_fit_target,
        "activation_fit_max_rows": int(activation_fit_max_rows),
        "activation_fit_alpha": float(activation_fit_alpha),
        "activation_fit_ridge": float(activation_fit_ridge),
        "activation_fit_reports": activation_fit_reports,
        "untie_output_head": bool(untie_output_head),
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
        "axis_assignment_cache_entries": _axis_assignment_cache_report(axis_assignment_cache),
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
    hidden_basis=None,
    hidden_axis_weights=None,
    activation_fold_assignment: str = "balanced",
    transport_backend: str = "sparse",
    projection_scale: str = "none",
    projection_blend_alpha: float = 1.0,
    consumer_hidden_reduction: str = "mean",
    assignment_cache: dict[Any, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    projection_blend_alpha = _bounded_alpha(projection_blend_alpha, "projection_blend_alpha")
    if consumer_hidden_reduction not in {"mean", "sum", "sqrt-sum"}:
        raise ValueError("consumer_hidden_reduction must be mean, sum, or sqrt-sum.")
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
            hidden_basis=hidden_basis if donor_tensor.dim() >= 2 else None,
            axis_weights=axis_weights,
            axis_reductions=_with_consumer_hidden_reduction(
                _axis_reductions_for_non_layer_tensor_key(target_key),
                consumer_hidden_reduction,
            ),
            activation_fold_assignment=activation_fold_assignment,
            transport_backend=transport_backend,
            assignment_cache=assignment_cache,
        )
        projected = _scale_projection_to_target(torch, projected, target_tensor, projection_scale)
        projected = _blend_projection_with_target(torch, projected, target_tensor, projection_blend_alpha)
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


def _blend_projection_with_target(torch, projected, target_tensor, alpha: float):
    alpha = _bounded_alpha(alpha, "projection_blend_alpha")
    if alpha == 1.0:
        return projected
    target_values = target_tensor.detach().cpu().to(torch.float32)
    return (float(alpha) * projected.to(torch.float32)) + ((1.0 - float(alpha)) * target_values)


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
    axis_bases: dict[int, Any] | None = None,
    hidden_basis=None,
    axis_weights: dict[int, Any] | None = None,
    axis_reductions: dict[int, str] | None = None,
    activation_fold_assignment: str = "balanced",
    transport_backend: str = "sparse",
    assignment_cache: dict[Any, dict[str, Any]] | None = None,
):
    if fold_mode not in {"mean", "rms", "transport"}:
        raise ValueError("fold_mode must be mean, rms, or transport.")
    if activation_fold_assignment not in {"balanced", "similarity"}:
        raise ValueError("activation_fold_assignment must be balanced or similarity.")
    if transport_backend not in {"sparse", "sinkhorn"}:
        raise ValueError("transport_backend must be sparse or sinkhorn.")
    if tuple(source.shape) == tuple(target_shape):
        return source.clone()
    if source.dim() != len(target_shape):
        raise ValueError(f"Cannot resize rank-{source.dim()} tensor to shape {target_shape}.")
    resized = source.to(torch.float32)
    for axis, target_dim in enumerate(target_shape):
        if resized.shape[axis] != int(target_dim):
            basis = None if axis_bases is None else axis_bases.get(axis)
            basis = _hidden_basis_for_axis(torch, basis, int(resized.shape[axis]), int(target_dim))
            if basis is None:
                basis = _hidden_basis_for_axis(torch, hidden_basis, int(resized.shape[axis]), int(target_dim))
            if basis is not None:
                resized = _resize_tensor_axis_hidden_basis(torch, resized, axis, basis)
                continue
            profile = None if axis_weights is None else axis_weights.get(axis)
            weights, features, semantic_key = _axis_fold_profile(profile)
            resized = _resize_tensor_axis_mean_fold(
                torch,
                resized,
                axis,
                int(target_dim),
                fold_mode=fold_mode,
                weights=weights,
                features=features,
                reduction="mean" if axis_reductions is None else axis_reductions.get(axis, "mean"),
                assignment_mode=activation_fold_assignment,
                transport_backend=transport_backend,
                assignment_cache=assignment_cache,
                assignment_cache_key=semantic_key,
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
    axis_bases: dict[int, Any] | None = None,
    hidden_basis=None,
    attention_axis_weights: dict[int, Any] | None = None,
    attention_head_fold_mode: str | None = None,
    attention_head_dim_transport: str = "none",
    axis_weights: dict[int, Any] | None = None,
    axis_reductions: dict[int, str] | None = None,
    activation_fold_assignment: str = "balanced",
    transport_backend: str = "sparse",
    assignment_cache: dict[Any, dict[str, Any]] | None = None,
    semantic_prefix=None,
):
    if attention_head_dim_transport not in {"none", "qk-vo"}:
        raise ValueError("attention_head_dim_transport must be none or qk-vo.")
    specs = _attention_partition_specs(suffix, donor_config, target_config)
    if not specs or source.dim() != len(target_shape):
        return None
    if axis_reductions is None:
        axis_reductions = _axis_reductions_for_layer_tensor(suffix)
    resized = source.to(torch.float32)
    handled_axes = set()
    used_coupled_axis = False
    for axis, source_heads, target_heads in specs:
        if axis >= resized.dim() or int(resized.shape[axis]) == int(target_shape[axis]):
            continue
        head_profile = None if attention_axis_weights is None else attention_axis_weights.get(axis)
        head_weights, head_features, head_semantic_key = _axis_fold_profile(head_profile)
        head_dim_weights, head_dim_features, head_dim_semantic_key = (None, None, None)
        if attention_head_dim_transport == "qk-vo":
            head_dim_weights, head_dim_features, head_dim_semantic_key = _axis_head_dim_fold_profile(head_profile)
        head_dim_has_profile = head_dim_weights is not None or head_dim_features is not None
        coupled = _resize_head_partitioned_axis(
            torch,
            resized,
            axis,
            int(target_shape[axis]),
            source_heads,
            target_heads,
            fold_mode=fold_mode,
            reduction="mean" if axis_reductions is None else axis_reductions.get(axis, "mean"),
            transport_backend=transport_backend,
            head_weights=head_weights,
            head_features=head_features,
            head_fold_mode=attention_head_fold_mode,
            head_assignment_mode=activation_fold_assignment,
            head_dim_weights=head_dim_weights,
            head_dim_features=head_dim_features,
            head_dim_fold_mode="transport" if attention_head_dim_transport == "qk-vo" and head_dim_has_profile else None,
            head_dim_assignment_mode=activation_fold_assignment,
            assignment_cache=assignment_cache,
            head_assignment_key=head_semantic_key or _attention_axis_assignment_key(suffix, semantic_prefix, "heads"),
            head_dim_assignment_key=head_dim_semantic_key
            or _attention_axis_assignment_key(suffix, semantic_prefix, "head_dim"),
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
            basis = None if axis_bases is None else axis_bases.get(axis)
            basis = _hidden_basis_for_axis(torch, basis, int(resized.shape[axis]), int(target_dim))
            if basis is None:
                basis = _hidden_basis_for_axis(torch, hidden_basis, int(resized.shape[axis]), int(target_dim))
            if basis is not None:
                resized = _resize_tensor_axis_hidden_basis(torch, resized, axis, basis)
                continue
            profile = None if axis_weights is None or axis in handled_axes else axis_weights.get(axis)
            weights, features, semantic_key = _axis_fold_profile(profile)
            resized = _resize_tensor_axis_mean_fold(
                torch,
                resized,
                axis,
                int(target_dim),
                fold_mode=fold_mode,
                weights=weights,
                features=features,
                reduction="mean" if axis_reductions is None else axis_reductions.get(axis, "mean"),
                assignment_mode=activation_fold_assignment,
                transport_backend=transport_backend,
                assignment_cache=assignment_cache,
                assignment_cache_key=semantic_key,
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
    if marker.endswith(".o_proj.weight") or marker.endswith(".out_proj.weight") or marker.endswith(".attn.c_proj.weight"):
        return [(1, q_heads_source, q_heads_target)]
    return []


def _axis_reductions_for_layer_tensor(suffix: str) -> dict[int, str]:
    lowered = f".{suffix.lower()}"
    if (
        lowered.endswith(".q_proj.weight")
        or lowered.endswith(".k_proj.weight")
        or lowered.endswith(".v_proj.weight")
        or lowered.endswith(".query_key_value.weight")
        or lowered.endswith(".c_attn.weight")
        or lowered.endswith(".gate_proj.weight")
        or lowered.endswith(".up_proj.weight")
        or lowered.endswith(".w1.weight")
        or lowered.endswith(".w3.weight")
        or lowered.endswith(".dense_h_to_4h.weight")
        or lowered.endswith(".fc1.weight")
    ):
        return {1: "sum"}
    if lowered.endswith(".down_proj.weight") or lowered.endswith(".fc2.weight") or lowered.endswith(".c_proj.weight"):
        return {1: "sum"}
    if lowered.endswith(".o_proj.weight") or lowered.endswith(".out_proj.weight") or lowered.endswith(".attn.c_proj.weight"):
        return {1: "sum"}
    return {}


def _axis_reductions_for_non_layer_tensor_key(key: str) -> dict[int, str]:
    lowered = key.lower()
    if lowered.endswith("lm_head.weight") or lowered.endswith("embed_out.weight"):
        return {1: "sum"}
    return {}


def _with_consumer_hidden_reduction(axis_reductions: dict[int, str], consumer_hidden_reduction: str) -> dict[int, str]:
    if consumer_hidden_reduction not in {"mean", "sum", "sqrt-sum"}:
        raise ValueError("consumer_hidden_reduction must be mean, sum, or sqrt-sum.")
    if consumer_hidden_reduction == "sum":
        return dict(axis_reductions)
    converted = dict(axis_reductions)
    for axis, reduction in list(converted.items()):
        if reduction == "sum":
            converted[axis] = consumer_hidden_reduction
    return converted


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
    reduction: str = "mean",
    transport_backend: str = "sparse",
    head_weights=None,
    head_features=None,
    head_fold_mode: str | None = None,
    head_assignment_mode: str = "balanced",
    head_dim_weights=None,
    head_dim_features=None,
    head_dim_fold_mode: str | None = None,
    head_dim_assignment_mode: str = "balanced",
    assignment_cache: dict[Any, dict[str, Any]] | None = None,
    head_assignment_key=None,
    head_dim_assignment_key=None,
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
        fold_mode=head_fold_mode or fold_mode,
        weights=head_weights,
        features=head_features,
        reduction=reduction,
        assignment_mode=head_assignment_mode,
        transport_backend=transport_backend,
        assignment_cache=assignment_cache,
        assignment_cache_key=head_assignment_key,
    )
    partitioned = _resize_tensor_axis_mean_fold(
        torch,
        partitioned,
        axis=1,
        target_dim=target_head_dim,
        fold_mode=head_dim_fold_mode or fold_mode,
        weights=head_dim_weights,
        features=head_dim_features,
        reduction=reduction,
        assignment_mode=head_dim_assignment_mode,
        transport_backend=transport_backend,
        assignment_cache=assignment_cache,
        assignment_cache_key=head_dim_assignment_key,
    )
    resized = partitioned.reshape(target_dim, *rest_shape)
    return resized.movedim(0, axis).contiguous()


def _axis_fold_profile(profile):
    if isinstance(profile, dict):
        return profile.get("weights"), profile.get("features"), profile.get("assignment_key")
    return profile, None, None


def _axis_head_dim_fold_profile(profile):
    if isinstance(profile, dict):
        return (
            profile.get("head_dim_weights"),
            profile.get("head_dim_features"),
            profile.get("head_dim_assignment_key"),
        )
    return None, None, None


def _hidden_basis_for_axis(torch, hidden_basis, source_dim: int, target_dim: int):
    if hidden_basis is None:
        return None
    basis = hidden_basis
    if isinstance(hidden_basis, dict):
        basis = hidden_basis.get("basis")
    if basis is None:
        return None
    basis = basis.detach().cpu().to(torch.float32)
    if basis.dim() != 2:
        raise ValueError("hidden basis must be rank-2.")
    if int(basis.shape[0]) == int(source_dim) and int(basis.shape[1]) == int(target_dim):
        return basis
    return None


def _resize_tensor_axis_hidden_basis(torch, tensor, axis: int, basis):
    source_dim = int(tensor.shape[axis])
    if int(basis.shape[0]) != source_dim:
        raise ValueError("Hidden basis source dimension does not match tensor axis.")
    target_dim = int(basis.shape[1])
    moved = tensor.movedim(axis, 0).contiguous()
    rest_shape = tuple(moved.shape[1:])
    flat = moved.reshape(source_dim, -1).to(torch.float32)
    projected = basis.to(flat.dtype).t().matmul(flat)
    return projected.reshape((target_dim, *rest_shape)).movedim(0, axis).contiguous()


def _build_activation_hidden_basis(
    torch,
    activation_rows,
    source_dim: int,
    target_dim: int,
    preprocess: str = "centered-rowrms",
):
    if source_dim <= 0 or target_dim <= 0:
        raise ValueError("source_dim and target_dim must be positive.")
    if preprocess not in {"centered-rowrms", "uncentered-rowrms", "uncentered"}:
        raise ValueError("preprocess must be centered-rowrms, uncentered-rowrms, or uncentered.")
    rows = torch.nan_to_num(activation_rows.detach().cpu().to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    if rows.dim() != 2:
        raise ValueError("activation rows must be rank-2.")
    if int(rows.shape[1]) != int(source_dim):
        raise ValueError(f"activation row width {rows.shape[1]} does not match source_dim {source_dim}.")
    if int(target_dim) >= int(source_dim):
        basis = torch.eye(int(source_dim), dtype=torch.float32)[:, : int(target_dim)]
        return basis.contiguous(), {
            "mode": "identity",
            "source_dim": int(source_dim),
            "target_dim": int(target_dim),
            "preprocess": preprocess,
            "rows": int(rows.shape[0]),
            "retained_energy": 1.0,
        }
    if int(rows.shape[0]) < int(target_dim):
        raise ValueError("Need at least target_dim activation rows to fit activation-pca basis.")
    if preprocess == "centered-rowrms":
        rows = rows - rows.mean(dim=0, keepdim=True)
    if preprocess in {"centered-rowrms", "uncentered-rowrms"}:
        row_rms = rows.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        rows = rows / row_rms
    _u, singular_values, vh = torch.linalg.svd(rows, full_matrices=False)
    keep = min(int(target_dim), int(vh.shape[0]))
    if keep < int(target_dim):
        raise ValueError("SVD did not return enough basis vectors for requested target_dim.")
    basis = vh[:keep].t().contiguous()
    total_energy = singular_values.pow(2).sum().clamp_min(1e-12)
    retained_energy = singular_values[:keep].pow(2).sum() / total_energy
    return basis, {
        "mode": "activation-pca",
        "source_dim": int(source_dim),
        "target_dim": int(target_dim),
        "preprocess": preprocess,
        "rows": int(rows.shape[0]),
        "retained_energy": float(retained_energy.detach().cpu()),
        "top_singular_value": float(singular_values[0].detach().cpu()) if int(singular_values.numel()) else 0.0,
        "kept_singular_value": float(singular_values[keep - 1].detach().cpu()),
    }


def _build_layer_activation_hidden_bases(
    torch,
    activation_profile: dict[str, Any],
    target_layer_count: int,
    donor_layer_count: int,
    source_dim: int,
    target_dim: int,
    preprocess: str = "centered-rowrms",
):
    input_rows_by_layer = activation_profile.get("hidden_basis_input_rows") or {}
    output_rows_by_layer = activation_profile.get("hidden_basis_output_rows") or {}
    layer_bases: dict[int, dict[str, Any]] = {}
    reports = []
    for target_layer_idx in range(int(target_layer_count)):
        group = _donor_layer_group_for_target(target_layer_idx, target_layer_count, donor_layer_count)
        input_rows = input_rows_by_layer.get(int(group[0]))
        output_rows = output_rows_by_layer.get(int(group[-1]))
        item: dict[str, Any] = {}
        report = {
            "target_layer": int(target_layer_idx),
            "donor_layer_group": [int(layer_idx) for layer_idx in group],
        }
        if input_rows is not None:
            input_basis, input_report = _build_activation_hidden_basis(
                torch,
                input_rows,
                source_dim=source_dim,
                target_dim=target_dim,
                preprocess=preprocess,
            )
            item["input"] = input_basis
            report["input"] = input_report
        else:
            report["input"] = {"skipped": True, "reason": "missing_input_boundary_rows"}
        if output_rows is not None:
            output_basis, output_report = _build_activation_hidden_basis(
                torch,
                output_rows,
                source_dim=source_dim,
                target_dim=target_dim,
                preprocess=preprocess,
            )
            item["output"] = output_basis
            report["output"] = output_report
        else:
            report["output"] = {"skipped": True, "reason": "missing_output_boundary_rows"}
        if item:
            layer_bases[int(target_layer_idx)] = item
        reports.append(report)
    return layer_bases, {
        "scope": "layer",
        "target_layer_count": int(target_layer_count),
        "donor_layer_count": int(donor_layer_count),
        "source_dim": int(source_dim),
        "target_dim": int(target_dim),
        "preprocess": preprocess,
        "layers": reports,
    }


def _hidden_basis_axes_for_layer_tensor(torch, donor_tensor, suffix: str, layer_basis):
    if not layer_basis:
        return None
    input_basis = layer_basis.get("input")
    output_basis = layer_basis.get("output")
    lowered = f".{suffix.lower()}"
    axis_bases = {}
    for axis, source_dim in enumerate(donor_tensor.shape):
        if input_basis is not None and _hidden_basis_for_axis(torch, input_basis, int(source_dim), int(input_basis.shape[1])) is not None:
            if axis == 1 and (
                lowered.endswith(".q_proj.weight")
                or lowered.endswith(".k_proj.weight")
                or lowered.endswith(".v_proj.weight")
                or lowered.endswith(".gate_proj.weight")
                or lowered.endswith(".up_proj.weight")
                or lowered.endswith(".w1.weight")
                or lowered.endswith(".w3.weight")
                or lowered.endswith(".dense_h_to_4h.weight")
                or lowered.endswith(".fc1.weight")
            ):
                axis_bases[axis] = input_basis
        if output_basis is not None and _hidden_basis_for_axis(torch, output_basis, int(source_dim), int(output_basis.shape[1])) is not None:
            if axis == 0 and (
                lowered.endswith(".o_proj.weight")
                or lowered.endswith(".out_proj.weight")
                or lowered.endswith(".attn.c_proj.weight")
                or lowered.endswith(".down_proj.weight")
                or lowered.endswith(".w2.weight")
                or lowered.endswith(".dense_4h_to_h.weight")
                or lowered.endswith(".fc2.weight")
                or lowered.endswith(".c_proj.weight")
            ):
                axis_bases[axis] = output_basis
    return axis_bases or None


def _axis_assignment_cache_key(base_key, source_dim: int, target_dim: int, assignment_mode: str, fold_mode: str = "mean"):
    if base_key is None:
        return None
    return (_freeze_assignment_key(base_key), int(source_dim), int(target_dim), str(assignment_mode), str(fold_mode))


def _freeze_assignment_key(value):
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_assignment_key(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_assignment_key(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_assignment_key(item) for item in value))
    return value


def _attention_axis_assignment_key(suffix: str, semantic_prefix, axis_kind: str):
    marker = f".{suffix.lower()}"
    if axis_kind == "head_dim":
        return (semantic_prefix, "attention", "shared", "head_dim")
    if marker.endswith(".q_proj.weight") or marker.endswith(".q_proj.bias") or marker.endswith(".o_proj.weight"):
        stream = "q"
    elif marker.endswith(".k_proj.weight") or marker.endswith(".k_proj.bias"):
        stream = "kv"
    elif marker.endswith(".v_proj.weight") or marker.endswith(".v_proj.bias"):
        stream = "kv"
    else:
        return None
    return (semantic_prefix, "attention", stream, axis_kind)


def _axis_assignment_cache_report(assignment_cache: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
    report = []
    for key, item in assignment_cache.items():
        map_type = "transport" if "transport_indices" in item or "transport_matrix" in item else "assignment"
        transport_matrix = item.get("transport_matrix")
        report.append(
            {
                "key": repr(key),
                "map_type": map_type,
                "transport_backend": item.get("transport_backend"),
                "transport_matrix_shape": list(transport_matrix.shape) if transport_matrix is not None else None,
                "source_dim": int(item.get("source_dim", 0)),
                "target_dim": int(item.get("target_dim", 0)),
                "assignment_mode": str(item.get("assignment_mode", "")),
            }
        )
    return report


def _build_sparse_activation_transport(
    torch,
    features,
    target_dim: int,
    weights=None,
    top_k: int = 4,
    balance_iterations: int = 32,
    temperature: float = 0.08,
    dense_limit: int = 16_000_000,
    transport_backend: str = "sparse",
):
    if transport_backend not in {"sparse", "sinkhorn"}:
        raise ValueError("transport_backend must be sparse or sinkhorn.")
    feature_rows = features.detach().cpu().to(torch.float32)
    if feature_rows.dim() != 2:
        raise ValueError("Activation features must be a rank-2 tensor.")
    source_dim = int(feature_rows.shape[0])
    if source_dim <= target_dim:
        raise ValueError("Sparse transport is only used for source_dim > target_dim.")
    if weights is None:
        fold_weights = torch.ones(source_dim, dtype=torch.float32)
    else:
        fold_weights = weights.detach().cpu().to(torch.float32).flatten().clamp_min(1e-12)
        if int(fold_weights.numel()) != source_dim:
            raise ValueError("Activation transport weights must match feature rows.")
    feature_rows = _row_normalize(feature_rows)
    seed_assignment = _activation_similarity_assignment(torch, feature_rows, target_dim, fold_weights, iterations=4)
    centroids = []
    for target_idx in range(target_dim):
        mask = seed_assignment == target_idx
        if bool(mask.any()):
            cluster_weights = fold_weights[mask].unsqueeze(1)
            centroid = (feature_rows[mask] * cluster_weights).sum(dim=0) / cluster_weights.sum().clamp_min(1e-12)
        else:
            centroid = feature_rows[int(torch.argmax(fold_weights).item())]
        centroids.append(centroid)
    centroid_rows = _row_normalize(torch.stack(centroids, dim=0))
    top_k = max(1, min(int(top_k), target_dim))
    scores = feature_rows.matmul(centroid_rows.t())
    if transport_backend == "sinkhorn" and int(source_dim) * int(target_dim) <= int(dense_limit):
        source_mass = fold_weights / fold_weights.sum().clamp_min(1e-12)
        target_mass = torch.full((target_dim,), 1.0 / float(target_dim), dtype=torch.float32)
        logits = scores / max(float(temperature), 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values
        kernel = torch.exp(logits).clamp_min(1e-30)
        row_scale = torch.ones_like(source_mass)
        col_scale = torch.ones_like(target_mass)
        for _ in range(max(1, int(balance_iterations))):
            row_scale = source_mass / kernel.matmul(col_scale).clamp_min(1e-30)
            col_scale = target_mass / kernel.t().matmul(row_scale).clamp_min(1e-30)
        plan = row_scale.unsqueeze(1) * kernel * col_scale.unsqueeze(0)
        values = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-30)
        top_values, top_indices = torch.topk(values, k=top_k, dim=1)
        return {
            "transport_matrix": values.detach().cpu().to(torch.float32),
            "transport_indices": top_indices.detach().cpu().to(torch.long),
            "transport_values": top_values.detach().cpu().to(torch.float32),
            "source_dim": int(source_dim),
            "target_dim": int(target_dim),
            "assignment_mode": "transport",
            "transport_backend": "sinkhorn_dense",
            "sinkhorn_iterations": int(balance_iterations),
        }
    top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)
    values = torch.softmax(top_scores / max(float(temperature), 1e-6), dim=1)
    for _ in range(max(0, int(balance_iterations))):
        loads = torch.zeros(target_dim, dtype=torch.float32)
        loads.index_add_(0, top_indices.reshape(-1), (values * fold_weights.unsqueeze(1)).reshape(-1))
        balance = (loads.mean().clamp_min(1e-12) / loads.clamp_min(1e-12)).clamp(0.25, 4.0)
        values = values * balance[top_indices]
        values = values / values.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return {
        "transport_indices": top_indices.detach().cpu().to(torch.long),
        "transport_values": values.detach().cpu().to(torch.float32),
        "source_dim": int(source_dim),
        "target_dim": int(target_dim),
        "assignment_mode": "transport",
    }


def _apply_sparse_axis_transport(
    torch,
    flat,
    transport: dict[str, Any],
    fold_weights,
    reduction: str,
    fold_mode: str,
):
    if "transport_matrix" in transport:
        matrix = transport["transport_matrix"].to(flat.dtype)
        source_dim = int(flat.shape[0])
        target_dim = int(transport["target_dim"])
        if int(matrix.shape[0]) != source_dim or int(matrix.shape[1]) != target_dim:
            raise ValueError("Dense transport matrix has the wrong shape.")
        if reduction in {"sum", "sqrt-sum"}:
            coeffs = matrix
            count_coeffs = matrix
        else:
            coeffs = matrix * fold_weights.to(flat.dtype).unsqueeze(1)
            count_coeffs = coeffs
        folded = coeffs.t().matmul(flat)
        counts = count_coeffs.sum(dim=0)
        if reduction == "mean":
            divisor = counts.sqrt() if fold_mode == "rms" else counts
            folded = folded / divisor.clamp_min(1e-12).unsqueeze(1)
        elif reduction == "sqrt-sum":
            folded = folded / counts.clamp_min(1e-12).sqrt().unsqueeze(1)
        return folded

    indices = transport["transport_indices"].to(torch.long)
    values = transport["transport_values"].to(flat.dtype)
    source_dim = int(flat.shape[0])
    target_dim = int(transport["target_dim"])
    if int(indices.shape[0]) != source_dim or int(values.shape[0]) != source_dim:
        raise ValueError("Sparse transport has the wrong source dimension.")
    folded = torch.zeros((target_dim, flat.shape[1]), dtype=flat.dtype)
    counts = torch.zeros(target_dim, dtype=flat.dtype)
    if reduction in {"sum", "sqrt-sum"}:
        coeffs = values
        count_coeffs = values
    else:
        coeffs = values * fold_weights.to(flat.dtype).unsqueeze(1)
        count_coeffs = coeffs
    for slot in range(int(indices.shape[1])):
        target_idx = indices[:, slot].contiguous()
        coeff = coeffs[:, slot].contiguous()
        folded.index_add_(0, target_idx, flat * coeff.unsqueeze(1))
        counts.index_add_(0, target_idx, count_coeffs[:, slot].to(flat.dtype))
    if reduction == "mean":
        divisor = counts.sqrt() if fold_mode == "rms" else counts
        folded = folded / divisor.clamp_min(1e-12).unsqueeze(1)
    elif reduction == "sqrt-sum":
        folded = folded / counts.clamp_min(1e-12).sqrt().unsqueeze(1)
    return folded


def _resize_tensor_axis_mean_fold(
    torch,
    tensor,
    axis: int,
    target_dim: int,
    fold_mode: str = "mean",
    weights=None,
    features=None,
    reduction: str = "mean",
    assignment_mode: str = "balanced",
    transport_backend: str = "sparse",
    assignment_cache: dict[Any, dict[str, Any]] | None = None,
    assignment_cache_key=None,
):
    if target_dim <= 0:
        raise ValueError("target_dim must be positive.")
    if fold_mode not in {"mean", "rms", "transport"}:
        raise ValueError("fold_mode must be mean, rms, or transport.")
    if reduction not in {"mean", "sum", "sqrt-sum"}:
        raise ValueError("reduction must be mean, sum, or sqrt-sum.")
    if assignment_mode not in {"balanced", "similarity"}:
        raise ValueError("assignment_mode must be balanced or similarity.")
    if transport_backend not in {"sparse", "sinkhorn"}:
        raise ValueError("transport_backend must be sparse or sinkhorn.")
    source_dim = int(tensor.shape[axis])
    if source_dim == target_dim:
        return tensor

    moved = tensor.movedim(axis, 0).contiguous()
    rest_shape = tuple(moved.shape[1:])
    flat = moved.reshape(source_dim, -1)
    if source_dim > target_dim:
        explicit_weights = weights is not None
        cache_key = _axis_assignment_cache_key(
            assignment_cache_key,
            source_dim=source_dim,
            target_dim=target_dim,
            assignment_mode=assignment_mode,
            fold_mode=fold_mode,
        )
        if weights is None and features is None:
            fold_weights = torch.ones(source_dim, dtype=flat.dtype)
        else:
            if weights is None:
                fold_weights = torch.ones(source_dim, dtype=torch.float32)
            else:
                fold_weights = weights.detach().cpu().to(torch.float32).flatten()
            if int(fold_weights.numel()) != source_dim:
                raise ValueError("Axis weights must match the source dimension.")
            fold_weights = fold_weights.clamp_min(1e-12)
        cached = None if cache_key is None or assignment_cache is None else assignment_cache.get(cache_key)
        if cached is not None and "transport_indices" in cached:
            folded = _apply_sparse_axis_transport(torch, flat, cached, fold_weights, reduction, fold_mode)
            return folded.reshape((target_dim, *rest_shape)).movedim(0, axis).contiguous()
        if fold_mode == "transport" and features is not None:
            if cached is None:
                cached = _build_sparse_activation_transport(
                    torch,
                    features,
                    target_dim,
                    weights=fold_weights,
                    transport_backend=transport_backend,
                )
                if cache_key is not None and assignment_cache is not None:
                    assignment_cache[cache_key] = cached
            folded = _apply_sparse_axis_transport(torch, flat, cached, fold_weights, reduction, fold_mode)
            return folded.reshape((target_dim, *rest_shape)).movedim(0, axis).contiguous()
        assignment = None
        if cached is not None and "assignment" in cached:
            assignment = cached["assignment"].to(torch.long)
        if assignment is None:
            if weights is None and features is None:
                assignment = torch.floor(torch.arange(source_dim, dtype=torch.float32) * target_dim / source_dim)
                assignment = assignment.to(torch.long).clamp_max(target_dim - 1)
            elif assignment_mode == "similarity" and features is not None:
                assignment = _activation_similarity_assignment(torch, features, target_dim, fold_weights)
            else:
                assignment = _balanced_activation_assignment(torch, fold_weights, target_dim)
            if cache_key is not None and assignment_cache is not None:
                assignment_cache[cache_key] = {
                    "assignment": assignment.detach().cpu().to(torch.long),
                    "source_dim": int(source_dim),
                    "target_dim": int(target_dim),
                    "assignment_mode": str(assignment_mode),
                }
        assignment = assignment.to(torch.long).cpu()
        if int(assignment.numel()) != source_dim:
            raise ValueError("Cached axis assignment has the wrong source dimension.")
        if int(assignment.min().item()) < 0 or int(assignment.max().item()) >= target_dim:
            raise ValueError("Cached axis assignment has out-of-range target bins.")
        folded = torch.zeros((target_dim, flat.shape[1]), dtype=flat.dtype)
        counts = torch.zeros(target_dim, dtype=flat.dtype)
        value_weights = torch.ones_like(fold_weights) if reduction in {"sum", "sqrt-sum"} else fold_weights
        weighted_flat = flat * value_weights.to(flat.dtype).unsqueeze(1)
        folded.index_add_(0, assignment, weighted_flat)
        counts.index_add_(0, assignment, fold_weights.to(flat.dtype))
        if reduction == "mean":
            divisor = counts.sqrt() if fold_mode == "rms" and not explicit_weights else counts
            folded = folded / divisor.clamp_min(1.0).unsqueeze(1)
        elif reduction == "sqrt-sum":
            folded = folded / counts.clamp_min(1.0).sqrt().unsqueeze(1)
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
    hidden_basis_max_rows: int = 0,
    linear_fit_max_rows: int = 0,
) -> dict[str, Any]:
    if not texts:
        raise ValueError("Need at least one activation calibration text.")
    hidden_scores = []
    hidden_features = []
    hidden_basis_pieces = []
    hidden_basis_rows = 0
    hidden_basis_input_pieces: dict[int, list[Any]] = {idx: [] for idx in range(len(donor_layers))}
    hidden_basis_output_pieces: dict[int, list[Any]] = {idx: [] for idx in range(len(donor_layers))}
    hidden_basis_input_rows: dict[int, int] = {idx: 0 for idx in range(len(donor_layers))}
    hidden_basis_output_rows: dict[int, int] = {idx: 0 for idx in range(len(donor_layers))}
    mlp_scores: dict[int, list[Any]] = {idx: [] for idx in range(len(donor_layers))}
    mlp_features: dict[int, list[Any]] = {idx: [] for idx in range(len(donor_layers))}
    attention_head_profiles: dict[int, dict[str, dict[str, list[Any]]]] = {
        idx: {
            "q": {"scores": [], "features": []},
            "kv": {"scores": [], "features": []},
        }
        for idx in range(len(donor_layers))
    }
    attention_head_dim_profiles: dict[int, dict[str, dict[str, list[Any]]]] = {
        idx: {
            "qk": {"scores": [], "features": []},
            "vo": {"scores": [], "features": []},
        }
        for idx in range(len(donor_layers))
    }
    linear_fit_rows: dict[int, dict[str, dict[str, list[Any]]]] = {idx: {} for idx in range(len(donor_layers))}
    linear_fit_counts: dict[tuple[int, str], int] = {}
    handles = []
    q_heads = _config_int(donor.config, "num_attention_heads", "n_head", "num_heads")
    kv_heads = _config_int(donor.config, "num_key_value_heads", default=q_heads)

    def append_basis_rows(store, counts, layer_idx: int, tensor) -> None:
        if int(hidden_basis_max_rows) <= 0 or counts[int(layer_idx)] >= int(hidden_basis_max_rows):
            return
        rows = _activation_2d(torch, tensor).detach().cpu().to(torch.float32)
        remaining = int(hidden_basis_max_rows) - counts[int(layer_idx)]
        take = min(int(rows.shape[0]), remaining)
        if take > 0:
            store[int(layer_idx)].append(rows[:take])
            counts[int(layer_idx)] += int(take)

    def hidden_hook(layer_idx: int):
        def hook(_module, inputs):
            nonlocal hidden_basis_rows
            hidden = inputs[0].detach().to(torch.float32)
            hidden_scores.append(_activation_abs_score(torch, hidden).cpu())
            hidden_features.append(_activation_channel_summary(torch, _activation_2d(torch, hidden)).cpu())
            if int(hidden_basis_max_rows) > 0 and hidden_basis_rows < int(hidden_basis_max_rows):
                rows = _activation_2d(torch, hidden).detach().cpu().to(torch.float32)
                remaining = int(hidden_basis_max_rows) - hidden_basis_rows
                take = min(int(rows.shape[0]), remaining)
                if take > 0:
                    hidden_basis_pieces.append(rows[:take])
                    hidden_basis_rows += int(take)
            append_basis_rows(hidden_basis_input_pieces, hidden_basis_input_rows, layer_idx, hidden)

        return hook

    def hidden_output_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            append_basis_rows(hidden_basis_output_pieces, hidden_basis_output_rows, layer_idx, tensor.detach().to(torch.float32))

        return hook

    def mlp_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            values = tensor.detach().to(torch.float32)
            mlp_scores[layer_idx].append(_activation_abs_score(torch, values).cpu())
            mlp_features[layer_idx].append(_activation_channel_summary(torch, _activation_2d(torch, values)).cpu())

        return hook

    def append_attention_profile(layer_idx: int, stream: str | None, dim_stream: str | None, tensor, heads: int | None) -> None:
        if heads is None:
            return
        profile = _attention_head_activation_profile(torch, tensor.detach().to(torch.float32), int(heads))
        if profile is None:
            return
        if stream is not None:
            attention_head_profiles[int(layer_idx)][stream]["scores"].append(profile["scores"].cpu())
            attention_head_profiles[int(layer_idx)][stream]["features"].append(profile["features"].cpu())
        if dim_stream is not None:
            attention_head_dim_profiles[int(layer_idx)][dim_stream]["scores"].append(
                profile["head_dim_scores"].cpu()
            )
            attention_head_dim_profiles[int(layer_idx)][dim_stream]["features"].append(
                profile["head_dim_features"].cpu()
            )

    def attention_output_hook(layer_idx: int, stream: str, dim_stream: str | None, heads: int | None):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            append_attention_profile(layer_idx, stream, dim_stream, tensor, heads)

        return hook

    def attention_input_hook(layer_idx: int, dim_stream: str, heads: int | None):
        def hook(_module, inputs):
            if not inputs:
                return
            append_attention_profile(layer_idx, None, dim_stream, inputs[0], heads)

        return hook

    def append_linear_fit_rows(layer_idx: int, suffix: str, inputs, output) -> None:
        if int(linear_fit_max_rows) <= 0:
            return
        key = (int(layer_idx), str(suffix))
        used = int(linear_fit_counts.get(key, 0))
        if used >= int(linear_fit_max_rows):
            return
        if not inputs:
            return
        input_rows = _activation_2d(torch, inputs[0].detach().to(torch.float32)).detach().cpu()
        output_tensor = output[0] if isinstance(output, tuple) else output
        output_rows = _activation_2d(torch, output_tensor.detach().to(torch.float32)).detach().cpu()
        row_count = min(int(input_rows.shape[0]), int(output_rows.shape[0]), int(linear_fit_max_rows) - used)
        if row_count <= 0:
            return
        slot = linear_fit_rows[int(layer_idx)].setdefault(str(suffix), {"input": [], "output": []})
        slot["input"].append(input_rows[:row_count])
        slot["output"].append(output_rows[:row_count])
        linear_fit_counts[key] = used + row_count

    def linear_fit_hook(layer_idx: int, suffix: str):
        def hook(_module, inputs, output):
            append_linear_fit_rows(layer_idx, suffix, inputs, output)

        return hook

    for layer_idx, layer in enumerate(donor_layers):
        handles.append(layer.register_forward_pre_hook(hidden_hook(layer_idx)))
        handles.append(layer.register_forward_hook(hidden_output_hook(layer_idx)))
        attention_modules = _attention_projection_modules(layer)
        if attention_modules.get("q") is not None:
            handles.append(attention_modules["q"].register_forward_hook(attention_output_hook(layer_idx, "q", "qk", q_heads)))
            handles.append(attention_modules["q"].register_forward_hook(linear_fit_hook(layer_idx, "self_attn.q_proj.weight")))
        if attention_modules.get("k") is not None:
            handles.append(attention_modules["k"].register_forward_hook(attention_output_hook(layer_idx, "kv", "qk", kv_heads)))
            handles.append(attention_modules["k"].register_forward_hook(linear_fit_hook(layer_idx, "self_attn.k_proj.weight")))
        if attention_modules.get("v") is not None:
            handles.append(attention_modules["v"].register_forward_hook(attention_output_hook(layer_idx, "kv", "vo", kv_heads)))
            handles.append(attention_modules["v"].register_forward_hook(linear_fit_hook(layer_idx, "self_attn.v_proj.weight")))
        if attention_modules.get("o") is not None:
            handles.append(attention_modules["o"].register_forward_pre_hook(attention_input_hook(layer_idx, "vo", q_heads)))
            handles.append(attention_modules["o"].register_forward_hook(linear_fit_hook(layer_idx, "self_attn.o_proj.weight")))
        spec = _mlp_spec(layer)
        if spec is None:
            continue
        for name, linear in zip(spec["input_names"], spec["inputs"]):
            handles.append(linear.register_forward_hook(mlp_hook(layer_idx)))
            handles.append(linear.register_forward_hook(linear_fit_hook(layer_idx, f"{_mlp_prefix(layer, spec)}.{name}.weight")))
        handles.append(
            spec["output"].register_forward_hook(linear_fit_hook(layer_idx, f"{_mlp_prefix(layer, spec)}.{spec['output_name']}.weight"))
        )

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
    folded_attention_profiles = {}
    for layer_idx, by_stream in attention_head_profiles.items():
        stream_profiles = {}
        for stream, values in by_stream.items():
            if values["scores"] and values["features"]:
                stream_profiles[stream] = {
                    "scores": torch.stack(values["scores"], dim=0).mean(dim=0).clamp_min(1e-12),
                    "features": torch.stack(values["features"], dim=0).mean(dim=0),
                }
        if stream_profiles:
            folded_attention_profiles[int(layer_idx)] = stream_profiles
    folded_attention_head_dim_profiles = {}
    for layer_idx, by_stream in attention_head_dim_profiles.items():
        stream_profiles = {}
        for stream, values in by_stream.items():
            if values["scores"] and values["features"]:
                stream_profiles[stream] = {
                    "scores": torch.stack(values["scores"], dim=0).mean(dim=0).clamp_min(1e-12),
                    "features": torch.stack(values["features"], dim=0).mean(dim=0),
                }
        if stream_profiles:
            folded_attention_head_dim_profiles[int(layer_idx)] = stream_profiles
    folded_linear_fit_rows = {}
    for layer_idx, by_suffix in linear_fit_rows.items():
        suffix_rows = {}
        for suffix, values in by_suffix.items():
            if values["input"] and values["output"]:
                suffix_rows[str(suffix)] = {
                    "input": torch.cat(values["input"], dim=0),
                    "output": torch.cat(values["output"], dim=0),
                }
        if suffix_rows:
            folded_linear_fit_rows[int(layer_idx)] = suffix_rows
    return {
        "hidden_scores": torch.stack(hidden_scores, dim=0).mean(dim=0).clamp_min(1e-12),
        "hidden_features": torch.stack(hidden_features, dim=0).mean(dim=0),
        "hidden_basis_rows": torch.cat(hidden_basis_pieces, dim=0) if hidden_basis_pieces else None,
        "hidden_basis_input_rows": {
            int(layer_idx): torch.cat(pieces, dim=0) for layer_idx, pieces in hidden_basis_input_pieces.items() if pieces
        },
        "hidden_basis_output_rows": {
            int(layer_idx): torch.cat(pieces, dim=0) for layer_idx, pieces in hidden_basis_output_pieces.items() if pieces
        },
        "mlp_scores": folded_mlp_scores,
        "mlp_features": folded_mlp_features,
        "attention_head_profiles": folded_attention_profiles,
        "attention_head_dim_profiles": folded_attention_head_dim_profiles,
        "linear_fit_rows": folded_linear_fit_rows,
    }


def _activation_abs_score(torch, tensor):
    values = _activation_2d(torch, tensor)
    return values.abs().mean(dim=0)


def _untie_output_head(torch, model) -> bool:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        return False
    if getattr(output_embeddings, "weight", None) is None:
        return False
    if output_embeddings.weight.data_ptr() != input_embeddings.weight.data_ptr():
        if hasattr(model.config, "tie_word_embeddings"):
            model.config.tie_word_embeddings = False
        return False
    bias = getattr(output_embeddings, "bias", None)
    new_head = torch.nn.Linear(
        int(output_embeddings.weight.shape[1]),
        int(output_embeddings.weight.shape[0]),
        bias=bias is not None,
        device=output_embeddings.weight.device,
        dtype=output_embeddings.weight.dtype,
    )
    new_head.weight.data.copy_(output_embeddings.weight.data.detach().clone())
    if bias is not None:
        new_head.bias.data.copy_(bias.data.detach().clone())
    model.set_output_embeddings(new_head)
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False
    return True


def _activation_2d(torch, tensor):
    values = tensor
    if values.dim() == 3:
        values = values.reshape(-1, values.shape[-1])
    elif values.dim() == 1:
        values = values.unsqueeze(0)
    elif values.dim() != 2:
        values = values.reshape(-1, values.shape[-1])
    return values


def _attention_projection_modules(layer) -> dict[str, Any]:
    attention = None
    for name in ("self_attn", "attn", "attention"):
        candidate = getattr(layer, name, None)
        if candidate is not None:
            attention = candidate
            break
    if attention is None:
        return {}
    return {
        "q": _first_existing_attr(attention, ("q_proj", "query", "q")),
        "k": _first_existing_attr(attention, ("k_proj", "key", "k")),
        "v": _first_existing_attr(attention, ("v_proj", "value", "v")),
        "o": _first_existing_attr(attention, ("o_proj", "out_proj", "c_proj", "dense")),
    }


def _first_existing_attr(obj, names: Sequence[str]):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _attention_head_activation_profile(torch, tensor, heads: int) -> dict[str, Any] | None:
    if heads <= 0:
        return None
    values = _activation_2d(torch, tensor)
    if values.dim() != 2:
        return None
    source_dim = int(values.shape[1])
    if source_dim <= 0 or source_dim % int(heads) != 0:
        return None
    head_dim = source_dim // int(heads)
    if head_dim <= 0:
        return None
    partitioned = values.reshape(-1, int(heads), head_dim).to(torch.float32)
    signed_mean = partitioned.mean(dim=0)
    abs_mean = partitioned.abs().mean(dim=0)
    rms = partitioned.pow(2).mean(dim=0).sqrt()
    features = torch.cat([signed_mean, abs_mean, rms], dim=1).contiguous()
    scores = abs_mean.mean(dim=1).clamp_min(1e-12)
    head_dim_features = torch.stack(
        [
            signed_mean.mean(dim=0),
            abs_mean.mean(dim=0),
            rms.mean(dim=0),
        ],
        dim=1,
    ).contiguous()
    head_dim_scores = abs_mean.mean(dim=0).clamp_min(1e-12)
    return {
        "scores": scores,
        "features": features,
        "head_dim_scores": head_dim_scores,
        "head_dim_features": head_dim_features,
    }


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
    layer_group_key = tuple(int(item) for item in layer_group)
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
            axis_weights[axis] = {
                "weights": hidden_scores,
                "features": hidden_features,
                "assignment_key": ("hidden", "residual"),
            }
    suffix_marker = f".{suffix}."
    if mlp_scores is not None and (".mlp." in suffix_marker or ".feed_forward." in suffix_marker):
        for axis, source_dim in enumerate(donor_tensor.shape):
            if int(source_dim) == int(mlp_scores.numel()):
                axis_weights[axis] = {
                    "weights": mlp_scores,
                    "features": mlp_features,
                    "assignment_key": ("mlp", layer_group_key),
                }
    return axis_weights or None


def _attention_axis_weights_for_tensor(
    torch,
    donor_tensor,
    suffix: str,
    donor_layer_idx: int,
    donor_layer_group: Sequence[int] | None,
    activation_profile,
):
    if activation_profile is None:
        return None
    stream_info = _attention_streams_and_partition_axis(suffix)
    if stream_info is None:
        return None
    stream, dim_stream, axis = stream_info
    head_profiles = activation_profile.get("attention_head_profiles") or {}
    head_dim_profiles = activation_profile.get("attention_head_dim_profiles") or {}
    layer_group = list(donor_layer_group or [donor_layer_idx])
    layer_group_key = tuple(int(item) for item in layer_group)
    scores = _aggregate_layer_attention_profile(torch, head_profiles, layer_group, stream, "scores", clamp_min=1e-12)
    features = _aggregate_layer_attention_profile(torch, head_profiles, layer_group, stream, "features")
    head_dim_scores = _aggregate_layer_attention_profile(
        torch,
        head_dim_profiles,
        layer_group,
        dim_stream,
        "scores",
        clamp_min=1e-12,
    )
    head_dim_features = _aggregate_layer_attention_profile(
        torch,
        head_dim_profiles,
        layer_group,
        dim_stream,
        "features",
    )
    if scores is None:
        return None
    if axis >= len(donor_tensor.shape):
        return None
    source_dim = int(donor_tensor.shape[axis])
    if int(scores.numel()) <= 0 or source_dim % int(scores.numel()) != 0:
        return None
    source_head_dim = source_dim // int(scores.numel())
    if head_dim_scores is not None and int(head_dim_scores.numel()) != source_head_dim:
        head_dim_scores = None
        head_dim_features = None
    return {
        axis: {
            "weights": scores,
            "features": features,
            "assignment_key": ("attention-heads", stream, layer_group_key),
            "head_dim_weights": head_dim_scores,
            "head_dim_features": head_dim_features,
            "head_dim_assignment_key": ("attention-head-dim", dim_stream, layer_group_key),
        }
    }


def _attention_stream_and_partition_axis(suffix: str) -> tuple[str, int] | None:
    stream_info = _attention_streams_and_partition_axis(suffix)
    if stream_info is None:
        return None
    stream, _dim_stream, axis = stream_info
    return stream, axis


def _attention_streams_and_partition_axis(suffix: str) -> tuple[str, str, int] | None:
    marker = f".{suffix.lower()}"
    if marker.endswith(".q_proj.weight") or marker.endswith(".q_proj.bias"):
        return "q", "qk", 0
    if marker.endswith(".o_proj.weight") or marker.endswith(".out_proj.weight") or marker.endswith(".attn.c_proj.weight"):
        return "q", "vo", 1
    if marker.endswith(".k_proj.weight") or marker.endswith(".k_proj.bias"):
        return "kv", "qk", 0
    if marker.endswith(".v_proj.weight") or marker.endswith(".v_proj.bias"):
        return "kv", "vo", 0
    return None


def _aggregate_layer_attention_profile(
    torch,
    profiles: dict[int, Any],
    layer_group: Sequence[int],
    stream: str,
    field: str,
    clamp_min=None,
):
    rows = []
    for layer_idx in layer_group:
        layer_profile = profiles.get(int(layer_idx), {})
        stream_profile = layer_profile.get(stream, {})
        value = stream_profile.get(field)
        if value is not None:
            rows.append(value.detach().cpu().to(torch.float32))
    if not rows:
        return None
    stacked = torch.stack(rows, dim=0)
    merged = stacked.mean(dim=0)
    if clamp_min is not None:
        merged = merged.clamp_min(float(clamp_min))
    return merged


def _fit_projected_linear_from_donor_activations(
    torch,
    activation_profile,
    suffix: str,
    target_shape: tuple[int, int],
    donor_config,
    target_config,
    donor_layer_idx: int,
    donor_layer_group: Sequence[int] | None,
    ridge: float,
    fold_mode: str,
    hidden_basis=None,
    axis_weights: dict[int, Any] | None = None,
    activation_fold_assignment: str = "balanced",
    transport_backend: str = "sparse",
    assignment_cache: dict[Any, dict[str, Any]] | None = None,
    reference_weight=None,
):
    rows = _linear_fit_rows_for_suffix(torch, activation_profile, donor_layer_group or [donor_layer_idx], suffix)
    if rows is None:
        return None
    source_input = rows["input"]
    source_output = rows["output"]
    if source_input.dim() != 2 or source_output.dim() != 2:
        return None
    if int(source_input.shape[0]) < 2 or int(source_output.shape[0]) < 2:
        return None
    target_out, target_in = int(target_shape[0]), int(target_shape[1])
    folded_input = _resize_linear_activation_rows(
        torch,
        source_input,
        target_in,
        suffix,
        side="input",
        donor_config=donor_config,
        target_config=target_config,
        donor_layer_idx=donor_layer_idx,
        donor_layer_group=donor_layer_group,
        activation_profile=activation_profile,
        fold_mode=fold_mode,
        hidden_basis=hidden_basis,
        axis_weights=axis_weights,
        activation_fold_assignment=activation_fold_assignment,
        transport_backend=transport_backend,
        assignment_cache=assignment_cache,
    )
    folded_output = _resize_linear_activation_rows(
        torch,
        source_output,
        target_out,
        suffix,
        side="output",
        donor_config=donor_config,
        target_config=target_config,
        donor_layer_idx=donor_layer_idx,
        donor_layer_group=donor_layer_group,
        activation_profile=activation_profile,
        fold_mode=fold_mode,
        hidden_basis=hidden_basis,
        axis_weights=axis_weights,
        activation_fold_assignment=activation_fold_assignment,
        transport_backend=transport_backend,
        assignment_cache=assignment_cache,
    )
    if folded_input is None or folded_output is None:
        return None
    row_count = min(int(folded_input.shape[0]), int(folded_output.shape[0]))
    if row_count < 2:
        return None
    folded_input = folded_input[:row_count]
    folded_output = folded_output[:row_count]
    validation_rows = int(max(0, min(row_count // 5, row_count - 2)))
    if validation_rows >= 8 and reference_weight is not None:
        train_input = folded_input[:-validation_rows]
        train_output = folded_output[:-validation_rows]
        val_input = folded_input[-validation_rows:]
        val_output = folded_output[-validation_rows:]
    else:
        train_input = folded_input
        train_output = folded_output
        val_input = None
        val_output = None
    fitted = _fit_linear_weight_from_activation_rows(torch, train_input, train_output, ridge)
    fitted = torch.nan_to_num(fitted, nan=0.0, posinf=0.0, neginf=0.0)
    if reference_weight is not None:
        reference = reference_weight.detach().cpu().to(torch.float32)
        fitted = _scale_projection_to_target(torch, fitted, reference, "target-rms")
    if val_input is not None and val_output is not None and reference_weight is not None:
        reference = reference_weight.detach().cpu().to(torch.float32)
        fit_error = val_input.matmul(fitted.t()).sub(val_output).pow(2).mean()
        reference_error = val_input.matmul(reference.t()).sub(val_output).pow(2).mean()
        if not bool(torch.isfinite(fit_error)) or float(fit_error) >= float(reference_error):
            return None
    return fitted


def _linear_fit_rows_for_suffix(torch, activation_profile, layer_group: Sequence[int], suffix: str):
    if activation_profile is None:
        return None
    rows_by_layer = activation_profile.get("linear_fit_rows") or {}
    input_rows = []
    output_rows = []
    for layer_idx in layer_group:
        layer_rows = rows_by_layer.get(int(layer_idx), {})
        values = layer_rows.get(str(suffix))
        if values is None:
            continue
        input_rows.append(values["input"].detach().cpu().to(torch.float32))
        output_rows.append(values["output"].detach().cpu().to(torch.float32))
    if not input_rows or not output_rows:
        return None
    return {"input": torch.cat(input_rows, dim=0), "output": torch.cat(output_rows, dim=0)}


def _resize_linear_activation_rows(
    torch,
    rows,
    target_dim: int,
    suffix: str,
    side: str,
    donor_config,
    target_config,
    donor_layer_idx: int,
    donor_layer_group: Sequence[int] | None,
    activation_profile,
    fold_mode: str,
    hidden_basis=None,
    axis_weights: dict[int, Any] | None = None,
    activation_fold_assignment: str = "balanced",
    transport_backend: str = "sparse",
    assignment_cache: dict[Any, dict[str, Any]] | None = None,
):
    if side not in {"input", "output"}:
        raise ValueError("side must be input or output.")
    values = rows.detach().cpu().to(torch.float32)
    if values.dim() != 2:
        return None
    if int(values.shape[1]) == int(target_dim):
        return values.contiguous()
    attention = _linear_activation_attention_partition(suffix, side, donor_config, target_config)
    if attention is not None:
        source_heads, target_heads, stream = attention
        profile = _attention_head_profile_for_stream(
            torch,
            activation_profile,
            donor_layer_group or [donor_layer_idx],
            stream,
        )
        head_weights, head_features, head_key = _axis_fold_profile(profile)
        return _resize_head_partitioned_axis(
            torch,
            values,
            axis=1,
            target_dim=int(target_dim),
            source_heads=int(source_heads),
            target_heads=int(target_heads),
            fold_mode="transport" if head_features is not None else fold_mode,
            reduction="mean",
            transport_backend=transport_backend,
            head_weights=head_weights,
            head_features=head_features,
            head_assignment_mode=activation_fold_assignment,
            assignment_cache=assignment_cache,
            head_assignment_key=head_key,
            head_dim_assignment_key=("linear-fit", "attention", stream, "head_dim", tuple(int(i) for i in donor_layer_group or [donor_layer_idx])),
        )
    dummy = torch.zeros((1, int(values.shape[1])), dtype=torch.float32)
    row_axis_weights = _activation_axis_weights_for_tensor(
        torch,
        dummy,
        suffix,
        donor_layer_idx,
        donor_layer_group,
        activation_profile,
    )
    return _resize_tensor_to_shape(
        torch,
        values,
        (int(values.shape[0]), int(target_dim)),
        fold_mode=fold_mode,
        hidden_basis=hidden_basis,
        axis_weights=row_axis_weights,
        axis_reductions={1: "mean"},
        activation_fold_assignment=activation_fold_assignment,
        transport_backend=transport_backend,
        assignment_cache=assignment_cache,
    )


def _linear_activation_attention_partition(suffix: str, side: str, donor_config, target_config):
    marker = f".{suffix.lower()}"
    q_heads_source = _config_int(donor_config, "num_attention_heads", "n_head", "num_heads")
    q_heads_target = _config_int(target_config, "num_attention_heads", "n_head", "num_heads")
    kv_heads_source = _config_int(donor_config, "num_key_value_heads", default=q_heads_source)
    kv_heads_target = _config_int(target_config, "num_key_value_heads", default=q_heads_target)
    if q_heads_source is None or q_heads_target is None:
        return None
    if side == "output" and (marker.endswith(".q_proj.weight") or marker.endswith(".q_proj.bias")):
        return q_heads_source, q_heads_target, "q"
    if side == "output" and (marker.endswith(".k_proj.weight") or marker.endswith(".k_proj.bias")):
        return None if kv_heads_source is None or kv_heads_target is None else (kv_heads_source, kv_heads_target, "kv")
    if side == "output" and (marker.endswith(".v_proj.weight") or marker.endswith(".v_proj.bias")):
        return None if kv_heads_source is None or kv_heads_target is None else (kv_heads_source, kv_heads_target, "kv")
    if side == "input" and (
        marker.endswith(".o_proj.weight") or marker.endswith(".out_proj.weight") or marker.endswith(".attn.c_proj.weight")
    ):
        return q_heads_source, q_heads_target, "q"
    return None


def _attention_head_profile_for_stream(torch, activation_profile, layer_group: Sequence[int], stream: str):
    head_profiles = (activation_profile or {}).get("attention_head_profiles") or {}
    scores = _aggregate_layer_attention_profile(torch, head_profiles, layer_group, stream, "scores", clamp_min=1e-12)
    features = _aggregate_layer_attention_profile(torch, head_profiles, layer_group, stream, "features")
    if scores is None:
        return None
    return {
        "weights": scores,
        "features": features,
        "assignment_key": ("linear-fit-attention-heads", stream, tuple(int(item) for item in layer_group)),
    }


def _activation_fit_suffix_enabled(suffix: str, activation_fit_target: str) -> bool:
    if activation_fit_target not in {"all", "attention", "attention-output", "mlp", "mlp-output", "input-projections"}:
        raise ValueError(
            "activation_fit_target must be all, attention, attention-output, mlp, mlp-output, or input-projections."
        )
    marker = f".{suffix.lower()}"
    attention_input = (
        marker.endswith(".self_attn.q_proj.weight")
        or marker.endswith(".self_attn.k_proj.weight")
        or marker.endswith(".self_attn.v_proj.weight")
        or marker.endswith(".attention.q_proj.weight")
        or marker.endswith(".attention.k_proj.weight")
        or marker.endswith(".attention.v_proj.weight")
        or marker.endswith(".attn.q_proj.weight")
        or marker.endswith(".attn.k_proj.weight")
        or marker.endswith(".attn.v_proj.weight")
        or marker.endswith(".query_key_value.weight")
        or marker.endswith(".c_attn.weight")
    )
    attention_output = (
        marker.endswith(".self_attn.o_proj.weight")
        or marker.endswith(".self_attn.out_proj.weight")
        or marker.endswith(".self_attn.dense.weight")
        or marker.endswith(".attention.o_proj.weight")
        or marker.endswith(".attention.out_proj.weight")
        or marker.endswith(".attention.dense.weight")
        or marker.endswith(".attn.o_proj.weight")
        or marker.endswith(".attn.out_proj.weight")
        or marker.endswith(".attn.c_proj.weight")
    )
    mlp_input = (
        marker.endswith(".mlp.gate_proj.weight")
        or marker.endswith(".mlp.up_proj.weight")
        or marker.endswith(".feed_forward.w1.weight")
        or marker.endswith(".feed_forward.w3.weight")
        or marker.endswith(".mlp.dense_h_to_4h.weight")
        or marker.endswith(".mlp.fc1.weight")
    )
    mlp_output = (
        marker.endswith(".mlp.down_proj.weight")
        or marker.endswith(".feed_forward.w2.weight")
        or marker.endswith(".mlp.dense_4h_to_h.weight")
        or marker.endswith(".mlp.fc2.weight")
    )
    if activation_fit_target == "all":
        return attention_input or attention_output or mlp_input or mlp_output
    if activation_fit_target == "attention":
        return attention_input or attention_output
    if activation_fit_target == "attention-output":
        return attention_output
    if activation_fit_target == "mlp":
        return mlp_input or mlp_output
    if activation_fit_target == "mlp-output":
        return mlp_output
    return attention_input or mlp_input


def _fit_linear_weight_from_activation_rows(torch, input_rows, output_rows, ridge: float):
    source = input_rows.detach().cpu().to(torch.float32)
    target = output_rows.detach().cpu().to(torch.float32)
    if source.dim() != 2 or target.dim() != 2:
        raise ValueError("input_rows and output_rows must be rank-2.")
    if int(source.shape[0]) != int(target.shape[0]):
        raise ValueError("input_rows and output_rows must have the same row count.")
    mapping = _ridge_map(torch, source, target, ridge)
    return mapping.t().contiguous()


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
    hidden_basis=None,
    lm_head_hidden_basis=None,
    hidden_axis_weights=None,
    activation_fold_assignment: str = "balanced",
    transport_backend: str = "sparse",
    projection_scale: str = "none",
    projection_blend_alpha: float = 1.0,
    consumer_hidden_reduction: str = "mean",
    assignment_cache: dict[Any, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if token_init_mode not in {"shared", "donor-tokenized", "donor-tokenized-weighted", "donor-omp"}:
        raise ValueError("token_init_mode must be shared, donor-tokenized, donor-tokenized-weighted, or donor-omp.")
    if consumer_hidden_reduction not in {"mean", "sum", "sqrt-sum"}:
        raise ValueError("consumer_hidden_reduction must be mean, sum, or sqrt-sum.")
    anchors, target_map, donor_map = _shared_anchor_tokens(target_tokenizer, donor_tokenizer, max_anchors)
    if token_init_mode == "shared":
        target_ids = [target_map[token] for token in anchors]
        donor_id_groups = [[donor_map[token]] for token in anchors]
        donor_group_weights = [[1.0] for _token in anchors]
        tokenized_rows = 0
        weighted_rows = 0
    elif token_init_mode == "donor-omp":
        target_ids, donor_id_groups, donor_group_weights, tokenized_rows, weighted_rows = _target_to_donor_omp_group_specs(
            torch,
            target_tokenizer,
            donor_tokenizer,
            target_map,
            donor_map,
        )
    else:
        target_ids, donor_id_groups, donor_group_weights, tokenized_rows, weighted_rows = _target_to_donor_token_group_specs(
            target_tokenizer,
            donor_tokenizer,
            target_map,
            donor_map,
            weighted=token_init_mode == "donor-tokenized-weighted",
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
        donor_id_group_weights=donor_group_weights,
        fold_mode=fold_mode,
        hidden_basis=hidden_basis,
        hidden_axis_weights=hidden_axis_weights,
        activation_fold_assignment=activation_fold_assignment,
        transport_backend=transport_backend,
        projection_scale=projection_scale,
        projection_blend_alpha=projection_blend_alpha,
        assignment_cache=assignment_cache,
        hidden_axis_reduction="mean",
        chunk_size=2048,
    )
    target_embeddings.weight.data.copy_(target_emb_weight.to(target_embeddings.weight.dtype))

    target_lm = target.get_output_embeddings()
    donor_lm = donor.get_output_embeddings()
    lm_head_updated = False
    lm_head_rows_initialized = 0
    tied_target_head = (
        target_lm is not None
        and target_embeddings is not None
        and target_lm.weight.data_ptr() == target_embeddings.weight.data_ptr()
    )
    if target_lm is not None and donor_lm is not None:
        if tied_target_head:
            lm_head_updated = False
        else:
            target_head = target_lm.weight.data.detach().cpu().to(torch.float32)
            donor_head = donor_lm.weight.data.detach().cpu().to(torch.float32)
            lm_head_rows_initialized = _project_token_weight_rows(
                torch,
                target_head,
                donor_head,
                target_ids,
                donor_id_groups,
                donor_id_group_weights=donor_group_weights,
                fold_mode=fold_mode,
                hidden_basis=hidden_basis if lm_head_hidden_basis is None else lm_head_hidden_basis,
                hidden_axis_weights=hidden_axis_weights,
                activation_fold_assignment=activation_fold_assignment,
                transport_backend=transport_backend,
                projection_scale=projection_scale,
                projection_blend_alpha=projection_blend_alpha,
                assignment_cache=assignment_cache,
                hidden_axis_reduction=consumer_hidden_reduction,
                chunk_size=2048,
            )
            target_lm.weight.data.copy_(target_head.to(target_lm.weight.dtype))
            lm_head_updated = True

    return {
        "mode": token_init_mode,
        "shared_decoded_tokens": len(set(_canonical_token_map(target_tokenizer)) & set(_canonical_token_map(donor_tokenizer))),
        "anchors_used": len(anchors),
        "tokenized_rows_initialized": int(tokenized_rows),
        "weighted_tokenized_rows_initialized": int(weighted_rows),
        "embedding_rows_initialized": int(embedding_rows_initialized),
        "lm_head_rows_initialized": int(lm_head_rows_initialized) if lm_head_updated else 0,
        "lm_head_updated": lm_head_updated,
        "lm_head_tied_to_embeddings": bool(tied_target_head),
        "consumer_hidden_reduction": consumer_hidden_reduction,
    }


def _project_token_weight_rows(
    torch,
    target_weight,
    donor_weight,
    target_ids: Sequence[int],
    donor_id_groups: Sequence[Sequence[int]],
    fold_mode: str,
    donor_id_group_weights: Sequence[Sequence[float]] | None = None,
    hidden_basis=None,
    hidden_axis_weights=None,
    activation_fold_assignment: str = "balanced",
    transport_backend: str = "sparse",
    projection_scale: str = "none",
    projection_blend_alpha: float = 1.0,
    assignment_cache: dict[Any, dict[str, Any]] | None = None,
    hidden_axis_reduction: str = "mean",
    chunk_size: int = 2048,
) -> int:
    projection_blend_alpha = _bounded_alpha(projection_blend_alpha, "projection_blend_alpha")
    if hidden_axis_reduction not in {"mean", "sum", "sqrt-sum"}:
        raise ValueError("hidden_axis_reduction must be mean, sum, or sqrt-sum.")
    if len(target_ids) != len(donor_id_groups):
        raise ValueError("target_ids and donor_id_groups must have the same length.")
    if donor_id_group_weights is not None and len(target_ids) != len(donor_id_group_weights):
        raise ValueError("target_ids and donor_id_group_weights must have the same length.")
    if not target_ids:
        return 0
    target_dim = int(target_weight.shape[1])
    for start in range(0, len(target_ids), int(chunk_size)):
        end = min(start + int(chunk_size), len(target_ids))
        chunk_target_ids = list(target_ids[start:end])
        donor_rows = torch.stack(
            [
                _weighted_donor_token_row(
                    torch,
                    donor_weight,
                    group,
                    None if donor_id_group_weights is None else donor_id_group_weights[row_idx],
                )
                for row_idx, group in enumerate(donor_id_groups[start:end], start=start)
            ],
            dim=0,
        )
        projected = _resize_tensor_to_shape(
            torch,
            donor_rows,
            (len(chunk_target_ids), target_dim),
            fold_mode=fold_mode,
            hidden_basis=hidden_basis,
            axis_weights={} if hidden_axis_weights is None else {1: hidden_axis_weights},
            axis_reductions={1: hidden_axis_reduction},
            activation_fold_assignment=activation_fold_assignment,
            transport_backend=transport_backend,
            assignment_cache=assignment_cache,
        )
        projected = _scale_projection_to_target(
            torch,
            projected,
            target_weight[torch.tensor(chunk_target_ids, dtype=torch.long)],
            projection_scale,
        )
        projected = _blend_projection_with_target(
            torch,
            projected,
            target_weight[torch.tensor(chunk_target_ids, dtype=torch.long)],
            projection_blend_alpha,
        )
        target_weight[torch.tensor(chunk_target_ids, dtype=torch.long)] = projected
    return int(len(target_ids))


def _weighted_donor_token_row(torch, donor_weight, donor_group: Sequence[int], donor_weights: Sequence[float] | None):
    group = [int(item) for item in donor_group]
    if not group:
        raise ValueError("donor token group cannot be empty.")
    rows = donor_weight[group]
    if donor_weights is None:
        return rows.mean(dim=0)
    weights = torch.tensor([float(item) for item in donor_weights], dtype=torch.float32)
    if int(weights.numel()) != len(group):
        raise ValueError("donor token weights must match donor token group length.")
    weights = weights.clamp_min(0.0)
    if float(weights.sum().item()) <= 0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum().clamp_min(1e-12)
    return (rows * weights.to(rows.dtype).unsqueeze(1)).sum(dim=0)


def _target_to_donor_token_groups(
    target_tokenizer,
    donor_tokenizer,
    target_map: dict[str, int],
    donor_map: dict[str, int],
) -> tuple[list[int], list[list[int]], int]:
    target_ids, donor_id_groups, _weights, tokenized_rows, _weighted_rows = _target_to_donor_token_group_specs(
        target_tokenizer,
        donor_tokenizer,
        target_map,
        donor_map,
        weighted=False,
    )
    return target_ids, donor_id_groups, tokenized_rows


def _target_to_donor_token_group_specs(
    target_tokenizer,
    donor_tokenizer,
    target_map: dict[str, int],
    donor_map: dict[str, int],
    weighted: bool = False,
) -> tuple[list[int], list[list[int]], list[list[float]], int, int]:
    target_special_ids = set(getattr(target_tokenizer, "all_special_ids", []) or [])
    donor_special_ids = set(getattr(donor_tokenizer, "all_special_ids", []) or [])
    donor_vocab_size = len(donor_tokenizer)
    target_ids = []
    donor_id_groups = []
    donor_group_weights = []
    tokenized_rows = 0
    weighted_rows = 0
    seen_target_ids = set()
    for token, target_idx in sorted(target_map.items(), key=lambda item: item[1]):
        donor_idx = donor_map.get(token)
        if donor_idx is None:
            continue
        target_ids.append(int(target_idx))
        donor_id_groups.append([int(donor_idx)])
        donor_group_weights.append([1.0])
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
        weights = _donor_tokenized_group_weights(donor_tokenizer, donor_ids) if weighted else [1.0] * len(donor_ids)
        donor_group_weights.append(weights)
        seen_target_ids.add(int(target_idx))
        tokenized_rows += 1
        if weighted and len(donor_ids) > 1:
            weighted_rows += 1
    return target_ids, donor_id_groups, donor_group_weights, tokenized_rows, weighted_rows


def _target_to_donor_omp_group_specs(
    torch,
    target_tokenizer,
    donor_tokenizer,
    target_map: dict[str, int],
    donor_map: dict[str, int],
    max_candidates: int = 32,
    max_selected: int = 4,
) -> tuple[list[int], list[list[int]], list[list[float]], int, int]:
    target_special_ids = set(getattr(target_tokenizer, "all_special_ids", []) or [])
    donor_special_ids = set(getattr(donor_tokenizer, "all_special_ids", []) or [])
    donor_vocab_size = len(donor_tokenizer)
    target_ids = []
    donor_id_groups = []
    donor_group_weights = []
    omp_rows = 0
    weighted_rows = 0
    seen_target_ids = set()
    for token, target_idx in sorted(target_map.items(), key=lambda item: item[1]):
        donor_idx = donor_map.get(token)
        if donor_idx is None:
            continue
        target_ids.append(int(target_idx))
        donor_id_groups.append([int(donor_idx)])
        donor_group_weights.append([1.0])
        seen_target_ids.add(int(target_idx))
    for target_idx in range(len(target_tokenizer)):
        if int(target_idx) in seen_target_ids or int(target_idx) in target_special_ids:
            continue
        decoded = _decode_single_token(target_tokenizer, int(target_idx))
        if not decoded or any(ord(ch) < 32 for ch in decoded):
            continue
        candidate_ids = _donor_omp_candidate_ids(
            donor_tokenizer,
            donor_map,
            decoded,
            donor_vocab_size,
            donor_special_ids,
            max_candidates=max_candidates,
        )
        if not candidate_ids:
            continue
        selected_ids, selected_weights = _omp_token_piece_weights(
            torch,
            decoded,
            donor_tokenizer,
            candidate_ids,
            max_selected=max_selected,
        )
        if not selected_ids:
            continue
        target_ids.append(int(target_idx))
        donor_id_groups.append(selected_ids)
        donor_group_weights.append(selected_weights)
        seen_target_ids.add(int(target_idx))
        omp_rows += 1
        if len(selected_ids) > 1:
            weighted_rows += 1
    return target_ids, donor_id_groups, donor_group_weights, omp_rows, weighted_rows


def _donor_omp_candidate_ids(
    donor_tokenizer,
    donor_map: dict[str, int],
    decoded: str,
    donor_vocab_size: int,
    donor_special_ids: set[int],
    max_candidates: int,
) -> list[int]:
    candidates = []

    def add_id(token_id: int | None) -> None:
        if token_id is None:
            return
        token_id = int(token_id)
        if token_id < 0 or token_id >= int(donor_vocab_size) or token_id in donor_special_ids:
            return
        if token_id not in candidates:
            candidates.append(token_id)

    for text in _omp_text_variants(decoded):
        for donor_id in _encode_without_specials(donor_tokenizer, text):
            add_id(int(donor_id))
        canonical = _canonical_token(text)
        add_id(donor_map.get(canonical))
        for piece in _omp_surface_pieces(canonical):
            add_id(donor_map.get(piece))
            for donor_id in _encode_without_specials(donor_tokenizer, piece):
                add_id(int(donor_id))
        if len(candidates) >= int(max_candidates):
            break
    return candidates[: int(max_candidates)]


def _omp_text_variants(decoded: str) -> list[str]:
    variants = []
    for item in (decoded, decoded.strip(), _canonical_token(decoded), _canonical_token(decoded).lower()):
        if item and item not in variants:
            variants.append(item)
    return variants


def _omp_surface_pieces(text: str) -> list[str]:
    pieces = []
    if not text:
        return pieces
    for match in re.finditer(r"[A-Za-z]+|\d+|[^\w\s]", text):
        value = match.group(0)
        if value and value not in pieces:
            pieces.append(value)
    for ch in text:
        if ch and not ch.isspace() and ch not in pieces:
            pieces.append(ch)
    return pieces


def _omp_token_piece_weights(
    torch,
    target_decoded: str,
    donor_tokenizer,
    candidate_ids: Sequence[int],
    max_selected: int = 4,
) -> tuple[list[int], list[float]]:
    target_features = _token_surface_features(target_decoded)
    if not target_features:
        return [], []
    feature_by_id = {}
    for donor_id in candidate_ids:
        decoded = _decode_single_token(donor_tokenizer, int(donor_id))
        features = _token_surface_features(decoded)
        if features:
            feature_by_id[int(donor_id)] = features
    if not feature_by_id:
        return [], []
    selected: list[int] = []
    residual = dict(target_features)
    for _ in range(max(1, int(max_selected))):
        best_id = None
        best_score = 0.0
        for donor_id, features in feature_by_id.items():
            if donor_id in selected:
                continue
            score = _sparse_cosine(residual, features)
            if score > best_score:
                best_id = donor_id
                best_score = score
        if best_id is None or best_score <= 0.0:
            break
        selected.append(best_id)
        weights = _solve_omp_surface_weights(torch, target_features, [feature_by_id[item] for item in selected])
        residual = _sparse_residual(target_features, [feature_by_id[item] for item in selected], weights)
        if _sparse_norm(residual) <= 1e-6:
            break
    if not selected:
        best_id = max(feature_by_id, key=lambda item: _sparse_cosine(target_features, feature_by_id[item]))
        selected = [best_id]
    weights = _solve_omp_surface_weights(torch, target_features, [feature_by_id[item] for item in selected])
    if not weights or sum(weights) <= 0:
        weights = [1.0 / len(selected)] * len(selected)
    return selected, weights


def _token_surface_features(decoded: str) -> dict[str, float]:
    text = _canonical_token(decoded).lower()
    text = "".join(ch for ch in text if ord(ch) >= 32)
    if not text:
        return {}
    features: dict[str, float] = {}

    def add(key: str, value: float = 1.0) -> None:
        features[key] = features.get(key, 0.0) + float(value)

    add(f"whole:{text}", 2.0)
    if text.isdigit():
        add("class:digit", 1.5)
    elif text.isalpha():
        add("class:alpha", 1.0)
    elif any(ch.isdigit() for ch in text):
        add("class:mixed_digit", 1.0)
    for ch in text:
        if not ch.isspace():
            add(f"char:{ch}", 1.0)
    for idx in range(max(0, len(text) - 1)):
        add(f"bigram:{text[idx:idx + 2]}", 1.25)
    for piece in _omp_surface_pieces(text):
        add(f"piece:{piece}", 1.5)
    norm = math.sqrt(sum(value * value for value in features.values()))
    if norm <= 0:
        return {}
    return {key: value / norm for key, value in features.items()}


def _solve_omp_surface_weights(torch, target_features: dict[str, float], selected_features: Sequence[dict[str, float]]) -> list[float]:
    if not selected_features:
        return []
    keys = sorted(set(target_features) | {key for features in selected_features for key in features})
    target = torch.tensor([target_features.get(key, 0.0) for key in keys], dtype=torch.float32)
    columns = [
        torch.tensor([features.get(key, 0.0) for key in keys], dtype=torch.float32)
        for features in selected_features
    ]
    matrix = torch.stack(columns, dim=1)
    try:
        solution = torch.linalg.lstsq(matrix, target).solution
    except RuntimeError:
        solution = torch.linalg.pinv(matrix).matmul(target)
    solution = torch.nan_to_num(solution.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if float(solution.sum().item()) <= 0:
        scores = torch.tensor(
            [_sparse_cosine(target_features, features) for features in selected_features],
            dtype=torch.float32,
        ).clamp_min(0.0)
        solution = scores if float(scores.sum().item()) > 0 else torch.ones(len(selected_features), dtype=torch.float32)
    solution = solution / solution.sum().clamp_min(1e-12)
    return [float(item) for item in solution.tolist()]


def _sparse_residual(
    target_features: dict[str, float],
    selected_features: Sequence[dict[str, float]],
    weights: Sequence[float],
) -> dict[str, float]:
    residual = dict(target_features)
    for weight, features in zip(weights, selected_features):
        for key, value in features.items():
            residual[key] = residual.get(key, 0.0) - float(weight) * float(value)
    return residual


def _sparse_dot(first: dict[str, float], second: dict[str, float]) -> float:
    if len(first) > len(second):
        first, second = second, first
    return float(sum(value * second.get(key, 0.0) for key, value in first.items()))


def _sparse_norm(features: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in features.values()))


def _sparse_cosine(first: dict[str, float], second: dict[str, float]) -> float:
    denom = _sparse_norm(first) * _sparse_norm(second)
    if denom <= 0:
        return 0.0
    return _sparse_dot(first, second) / denom


def _donor_tokenized_group_weights(donor_tokenizer, donor_ids: Sequence[int]) -> list[float]:
    weights = []
    for donor_id in donor_ids:
        decoded = _decode_single_token(donor_tokenizer, int(donor_id))
        cleaned = "".join(ch for ch in decoded if ord(ch) >= 32)
        canonical = _canonical_token(cleaned)
        weights.append(float(max(len(cleaned), len(canonical), 1)))
    return weights


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


def _mlp_prefix(layer, spec: dict[str, Any]) -> str:
    if getattr(layer, "feed_forward", None) is spec.get("mlp"):
        return "feed_forward"
    return "mlp"


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
