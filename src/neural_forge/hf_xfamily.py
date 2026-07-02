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
    _linear_cka,
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
    generation_config_report = _sanitize_generation_config_for_save(host, host_tokenizer)
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
        "generation_config_sanitized": generation_config_report,
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
            ppl_gate_mode="advisory",
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
    transport_support: str = "centroid",
    donor_layer_fold_mode: str = "representative",
    donor_layer_group_strategy: str = "depth",
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
    activation_fit_rank: int = 0,
    activation_fit_rank_mode: str = "absolute",
    activation_fit_method: str = "ridge",
    activation_fit_factor_steps: int = 0,
    activation_fit_factor_lr: float = 3e-2,
    activation_fit_factor_l2: float = 0.0,
    activation_fit_gate: str = "validation",
    activation_fit_validation_fraction: float = 0.20,
    activation_fit_min_validation_rows: int = 8,
    activation_fit_min_relative_improvement: float = 0.0,
    activation_fit_max_cosine_drop: float = 0.01,
    activation_feature_mode: str = "stats",
    activation_subspace_rank: int = 8,
    activation_domain_balance: str = "none",
    untie_output_head: bool = False,
    activation_calibration_texts: list[str] | None = None,
    activation_max_length: int = 128,
    activation_device: str = "cuda:0",
    neutralize_unmatched_operators: bool = False,
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
    if hidden_basis_mode not in {"none", "activation-pca", "shared-token-ridge"}:
        raise ValueError("hidden_basis_mode must be none, activation-pca, or shared-token-ridge.")
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
    if transport_support not in {"centroid", "importance", "quantile", "medoid", "quantile-medoid"}:
        raise ValueError("transport_support must be centroid, importance, quantile, medoid, or quantile-medoid.")
    if donor_layer_fold_mode not in {"representative", "average", "fold"}:
        raise ValueError("donor_layer_fold_mode must be representative, average, or fold.")
    if donor_layer_group_strategy not in {"depth", "activation-energy", "activation-geometry"}:
        raise ValueError("donor_layer_group_strategy must be depth, activation-energy, or activation-geometry.")
    if attention_fold_mode not in {"independent", "coupled", "activation-coupled"}:
        raise ValueError("attention_fold_mode must be independent, coupled, or activation-coupled.")
    if attention_head_dim_transport not in {"none", "qk-vo"}:
        raise ValueError("attention_head_dim_transport must be none or qk-vo.")
    if token_init_mode not in {"shared", "donor-tokenized", "donor-tokenized-weighted", "donor-omp"}:
        raise ValueError("token_init_mode must be shared, donor-tokenized, donor-tokenized-weighted, or donor-omp.")
    if projection_scale not in {"none", "target-rms", "source-rms", "geomean-rms"}:
        raise ValueError("projection_scale must be none, target-rms, source-rms, or geomean-rms.")
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
    if activation_fit_rank < 0:
        raise ValueError("activation_fit_rank must be non-negative.")
    if activation_fit_rank_mode not in {"absolute", "delta"}:
        raise ValueError("activation_fit_rank_mode must be absolute or delta.")
    if activation_fit_method not in {"ridge", "lrc"}:
        raise ValueError("activation_fit_method must be ridge or lrc.")
    if activation_fit_factor_steps < 0:
        raise ValueError("activation_fit_factor_steps must be non-negative.")
    if activation_fit_factor_steps > 0 and activation_fit_factor_lr <= 0:
        raise ValueError("activation_fit_factor_lr must be positive when factor steps are enabled.")
    if activation_fit_factor_l2 < 0:
        raise ValueError("activation_fit_factor_l2 must be non-negative.")
    if activation_fit_gate not in {"none", "validation"}:
        raise ValueError("activation_fit_gate must be none or validation.")
    if not 0 <= float(activation_fit_validation_fraction) < 1:
        raise ValueError("activation_fit_validation_fraction must be in [0, 1).")
    if activation_fit_min_validation_rows < 0:
        raise ValueError("activation_fit_min_validation_rows must be non-negative.")
    if activation_fit_min_relative_improvement < 0:
        raise ValueError("activation_fit_min_relative_improvement must be non-negative.")
    if activation_fit_max_cosine_drop < 0:
        raise ValueError("activation_fit_max_cosine_drop must be non-negative.")
    if activation_feature_mode not in {"stats", "subspace", "covariance"}:
        raise ValueError("activation_feature_mode must be stats, subspace, or covariance.")
    if activation_subspace_rank < 0:
        raise ValueError("activation_subspace_rank must be non-negative.")
    if activation_domain_balance not in {"none", "keyword"}:
        raise ValueError("activation_domain_balance must be none or keyword.")
    projection_blend_alpha = _bounded_alpha(projection_blend_alpha, "projection_blend_alpha")
    activation_fit_alpha = _bounded_alpha(activation_fit_alpha, "activation_fit_alpha")
    if activation_max_length <= 1:
        raise ValueError("activation_max_length must be greater than 1.")
    effective_activation_fit_max_rows = _effective_activation_fit_max_rows(
        activation_fit_linears,
        activation_fit_max_rows,
        hidden_basis_max_rows,
    )

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
    if hidden_basis_mode == "activation-pca":
        target_hidden_size = int(target.get_input_embeddings().weight.shape[1])
        if int(hidden_basis_max_rows) < target_hidden_size:
            raise ValueError(
                "activation-pca hidden basis needs hidden_basis_max_rows >= target hidden size "
                f"({hidden_basis_max_rows} < {target_hidden_size})."
            )
    activation_profile = None
    if activation_calibration_texts:
        needs_layer_activation_rows = hidden_basis_mode == "activation-pca" or donor_layer_group_strategy in {
            "activation-energy",
            "activation-geometry",
        }
        activation_profile = _collect_donor_axis_activation_scores(
            torch,
            donor,
            donor_tokenizer,
            donor_layers,
            activation_calibration_texts,
            max_length=activation_max_length,
            device=activation_device,
            hidden_basis_max_rows=hidden_basis_max_rows if needs_layer_activation_rows else 0,
            linear_fit_max_rows=effective_activation_fit_max_rows if activation_fit_linears else 0,
            activation_feature_mode=activation_feature_mode,
            activation_subspace_rank=activation_subspace_rank,
            activation_domain_balance=activation_domain_balance,
        )
    if attention_fold_mode == "activation-coupled" and activation_profile is None:
        raise ValueError("activation-coupled attention folding requires activation calibration texts.")
    if attention_head_dim_transport != "none" and activation_profile is None:
        raise ValueError("attention head-dimension transport requires activation calibration texts.")
    if activation_fit_linears and activation_profile is None:
        raise ValueError("activation linear fitting requires activation calibration texts.")
    if donor_layer_group_strategy in {"activation-energy", "activation-geometry"} and activation_profile is None:
        raise ValueError(
            f"{donor_layer_group_strategy} donor layer grouping requires activation calibration texts."
        )
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
    elif hidden_basis_mode == "shared-token-ridge":
        hidden_basis, hidden_basis_report = _build_shared_token_hidden_basis(
            torch,
            target,
            donor,
            target_tokenizer,
            donor_tokenizer,
            max_anchors=max_anchors,
            ridge=ridge,
        )

    target_state = target.state_dict()
    donor_state = donor.state_dict()
    donor_layer_keys = _layer_tensor_index(donor_state)
    donor_layer_groups, donor_layer_group_report = _donor_layer_groups_for_target_count(
        torch,
        target_layer_count=len(target_layers),
        donor_layer_count=len(donor_layers),
        strategy=donor_layer_group_strategy,
        activation_profile=activation_profile,
    )
    folded_donor_layer_cache: dict[int, dict[str, Any]] = {}
    axis_assignment_cache: dict[Any, dict[str, Any]] = {}
    layer_reports = []
    changed_keys = []
    skipped_layer_keys = []
    neutralized_layer_keys = []
    activation_fit_reports = []
    activation_fit_attempt_reports = []
    donor_layer_state_cache: dict[int, dict[str, Any]] = {}
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
        donor_layer_group = donor_layer_groups[int(target_layer_idx)]
        representative_scores = donor_layer_group_report.get("representative_scores")
        if representative_scores is None:
            representative_scores = donor_layer_group_report.get("delta_scores")
        donor_layer_idx = _representative_donor_layer_for_group(donor_layer_group, representative_scores)
        donor_layer_group = (
            [donor_layer_idx]
            if donor_layer_fold_mode == "representative"
            else donor_layer_group
        )
        if donor_layer_fold_mode == "representative":
            donor_key = _lookup_layer_tensor_key(donor_layer_keys, donor_layer_idx, suffix)
            donor_tensor = None if donor_key is None else donor_state[donor_key]
            synthetic_layer_report = None
            if donor_tensor is None:
                donor_layer_state = _donor_layer_state_from_index(
                    donor_state,
                    donor_layer_keys,
                    donor_layer_idx,
                    donor_layer_state_cache,
                )
                donor_tensor, synthetic_layer_report = _synthesize_cross_arch_layer_tensor(
                    torch,
                    donor_layer_state,
                    suffix,
                )
                if donor_tensor is not None:
                    donor_key = f"synthesized_donor_layer.{donor_layer_idx}.{suffix}"
        else:
            donor_key = f"folded_donor_layers.{target_layer_idx}.{suffix}"
            folded_state = _folded_donor_layer_state(
                torch,
                donor_layers,
                target_layer_idx=target_layer_idx,
                target_layer_count=len(target_layers),
                donor_layer_fold_mode=donor_layer_fold_mode,
                donor_layer_group=donor_layer_group,
                cache=folded_donor_layer_cache,
            )
            donor_tensor = _lookup_layer_tensor_value(folded_state, suffix)
            synthetic_layer_report = None
            if donor_tensor is None:
                donor_tensor, synthetic_layer_report = _synthesize_cross_arch_layer_tensor(
                    torch,
                    folded_state,
                    suffix,
                )
                if donor_tensor is not None:
                    donor_key = f"synthesized_folded_donor_layers.{target_layer_idx}.{suffix}"
        if donor_tensor is None:
            if neutralize_unmatched_operators:
                neutralized = _neutralized_unmatched_operator_tensor(torch, target_tensor, suffix)
                if neutralized is not None:
                    target_state[target_key] = neutralized.to(dtype=target_tensor.dtype)
                    changed_keys.append(target_key)
                    neutralized_layer_keys.append(
                        {
                            "target_key": target_key,
                            "reason": "neutralized_unmatched_operator",
                            "method": "zero_output_projection",
                            "donor_layer_group": donor_layer_group,
                            "donor_layer_fold_mode": donor_layer_fold_mode,
                        }
                    )
                    continue
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
                transport_support=transport_support,
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
                transport_support=transport_support,
                assignment_cache=axis_assignment_cache,
            )
        projected = _scale_projection_to_target(
            torch,
            projected,
            target_tensor,
            projection_scale,
            source_tensor=donor_tensor,
        )
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
                rank=activation_fit_rank,
                rank_mode=activation_fit_rank_mode,
                method=activation_fit_method,
                alpha=activation_fit_alpha,
                factor_steps=activation_fit_factor_steps,
                factor_lr=activation_fit_factor_lr,
                factor_l2=activation_fit_factor_l2,
                fold_mode=tensor_fold_mode,
                hidden_basis=hidden_basis if donor_tensor.dim() >= 2 else None,
                axis_weights=axis_weights,
                activation_fold_assignment=activation_fold_assignment,
                transport_backend=transport_backend,
                transport_support=transport_support,
                assignment_cache=axis_assignment_cache,
                reference_weight=projected,
                fit_gate=activation_fit_gate,
                validation_fraction=activation_fit_validation_fraction,
                min_validation_rows=activation_fit_min_validation_rows,
                min_relative_improvement=activation_fit_min_relative_improvement,
                max_cosine_drop=activation_fit_max_cosine_drop,
                return_report=True,
            )
            fit_report = dict(fitted["report"])
            activation_fit_report = {
                "target_key": target_key,
                "suffix": suffix,
                "target_shape": list(target_tensor.shape),
                "alpha": float(activation_fit_alpha),
                "ridge": float(activation_fit_ridge),
                "rank": int(activation_fit_rank),
                "rank_mode": activation_fit_rank_mode,
                **fit_report,
            }
            activation_fit_attempt_reports.append(activation_fit_report)
            fit_weight = fitted["weight"]
            if fit_weight is not None and bool(fit_report.get("accepted")):
                if activation_fit_method == "lrc":
                    projected = fit_weight
                    activation_fit_report["alpha_already_materialized"] = True
                else:
                    projected = (1.0 - activation_fit_alpha) * projected + activation_fit_alpha * fit_weight
                activation_fit_report["applied"] = True
                activation_fit_reports.append(activation_fit_report)
            else:
                activation_fit_report["applied"] = False
        projection_diagnostic = _tensor_projection_diagnostic(torch, donor_tensor, projected, target_tensor)
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
                "projection_diagnostic": projection_diagnostic,
                "synthetic_cross_arch": synthetic_layer_report,
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
        transport_support=transport_support,
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
        transport_backend=transport_backend,
        transport_support=transport_support,
        projection_scale=projection_scale,
        projection_blend_alpha=projection_blend_alpha,
        consumer_hidden_reduction=consumer_hidden_reduction,
        assignment_cache=axis_assignment_cache,
    )
    if getattr(target.config, "tie_word_embeddings", False) and not untie_output_head:
        target.tie_weights()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generation_config_report = _sanitize_generation_config_for_save(target, target_tokenizer)
    target.save_pretrained(output, safe_serialization=True)
    target_tokenizer.save_pretrained(output)
    projection_diagnostic_summary = _summarize_projection_diagnostics(
        item["projection_diagnostic"] for item in layer_reports if item.get("projection_diagnostic") is not None
    )
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
        "transport_support": transport_support,
        "donor_layer_fold_mode": donor_layer_fold_mode,
        "donor_layer_group_strategy": donor_layer_group_strategy,
        "donor_layer_groups": donor_layer_group_report,
        "attention_fold_mode": attention_fold_mode,
        "attention_head_dim_transport": attention_head_dim_transport,
        "token_init_mode": token_init_mode,
        "projection_scale": projection_scale,
        "projection_blend_alpha": projection_blend_alpha,
        "consumer_hidden_reduction": consumer_hidden_reduction,
        "activation_fit_linears": bool(activation_fit_linears),
        "activation_fit_target": activation_fit_target,
        "activation_fit_max_rows": int(effective_activation_fit_max_rows),
        "activation_fit_requested_max_rows": int(activation_fit_max_rows),
        "activation_fit_alpha": float(activation_fit_alpha),
        "activation_fit_ridge": float(activation_fit_ridge),
        "activation_fit_rank": int(activation_fit_rank),
        "activation_fit_rank_mode": activation_fit_rank_mode,
        "activation_fit_method": activation_fit_method,
        "activation_fit_factor_steps": int(activation_fit_factor_steps),
        "activation_fit_factor_lr": float(activation_fit_factor_lr),
        "activation_fit_factor_l2": float(activation_fit_factor_l2),
        "activation_fit_gate": activation_fit_gate,
        "activation_fit_validation_fraction": float(activation_fit_validation_fraction),
        "activation_fit_min_validation_rows": int(activation_fit_min_validation_rows),
        "activation_fit_min_relative_improvement": float(activation_fit_min_relative_improvement),
        "activation_fit_max_cosine_drop": float(activation_fit_max_cosine_drop),
        "activation_feature_mode": activation_feature_mode,
        "activation_subspace_rank": int(activation_subspace_rank),
        "activation_domain_balance": activation_domain_balance,
        "activation_domain_counts": {}
        if activation_profile is None
        else activation_profile.get("activation_domain_counts", {}),
        "activation_fit_reports": activation_fit_reports,
        "activation_fit_attempt_reports": activation_fit_attempt_reports,
        "activation_fit_attempt_count": len(activation_fit_attempt_reports),
        "activation_fit_applied_count": len(activation_fit_reports),
        "activation_fit_rejected_count": sum(
            1 for item in activation_fit_attempt_reports if not bool(item.get("applied"))
        ),
        "untie_output_head": bool(untie_output_head),
        "activation_calibration_text_count": len(activation_calibration_texts or []),
        "activation_max_length": activation_max_length,
        "activation_device": activation_device if activation_calibration_texts else None,
        "neutralize_unmatched_operators": bool(neutralize_unmatched_operators),
        "torch_dtype": torch_dtype,
        "target_layers": len(target_layers),
        "donor_layers": len(donor_layers),
        "layer_tensors_initialized": len(layer_reports),
        "projection_diagnostic_summary": projection_diagnostic_summary,
        "synthetic_cross_arch_layer_tensors_initialized": sum(
            1 for item in layer_reports if item.get("synthetic_cross_arch") is not None
        ),
        "layer_tensors_neutralized": len(neutralized_layer_keys),
        "neutralized_layer_tensors": neutralized_layer_keys[:200],
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
        "generation_config_sanitized": generation_config_report,
    }
    _write_json(output / "neural_forge_donor_projected_init_report.json", report)
    return report


def multi_donor_projected_init(
    target_config_model: str,
    donor_models: Sequence[str],
    output_dir: str | Path,
    donor_weights: Sequence[float] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Fold multiple donor families into one target shape via weighted stages.

    Each donor is projected into the target architecture with `donor_projected_init`.
    For donor `i`, the effective projection blend is `weight_i / cumulative_weight`
    times the requested projection blend alpha. With the default alpha of 1.0,
    the final checkpoint is a sequential weighted average of donor-projected
    target-shaped tensors rather than a same-shape checkpoint average.
    """

    donors = [str(item).strip() for item in donor_models if str(item).strip()]
    if not donors:
        raise ValueError("donor_models must contain at least one donor.")
    if donor_weights is None:
        weights = [1.0 for _ in donors]
    else:
        weights = [float(item) for item in donor_weights]
        if len(weights) != len(donors):
            raise ValueError("donor_weights must match donor_models length.")
    if any((not math.isfinite(weight)) or weight <= 0 for weight in weights):
        raise ValueError("donor_weights must be positive finite values.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stage_root = output / "multi_donor_stages"
    stage_root.mkdir(parents=True, exist_ok=True)
    requested_blend_alpha = _bounded_alpha(float(kwargs.get("projection_blend_alpha", 1.0)), "projection_blend_alpha")
    base_model = kwargs.get("target_base_model")
    cumulative_weight = 0.0
    stage_reports = []
    current_base = base_model
    for index, (donor_model, weight) in enumerate(zip(donors, weights)):
        cumulative_weight += float(weight)
        stage_alpha = requested_blend_alpha * (float(weight) / cumulative_weight)
        stage_output = output if index == len(donors) - 1 else stage_root / f"{index:02d}_{_model_path_tag(donor_model)}"
        stage_kwargs = dict(kwargs)
        stage_kwargs["target_base_model"] = current_base
        stage_kwargs["projection_blend_alpha"] = stage_alpha
        stage_kwargs["seed"] = int(stage_kwargs.get("seed", 0)) + index
        report = donor_projected_init(
            target_config_model=target_config_model,
            donor_model=donor_model,
            output_dir=stage_output,
            **stage_kwargs,
        )
        current_base = str(stage_output)
        stage_reports.append(
            {
                "stage": int(index),
                "donor_model": donor_model,
                "donor_weight": float(weight),
                "cumulative_weight": float(cumulative_weight),
                "effective_projection_blend_alpha": float(stage_alpha),
                "output_dir": str(stage_output),
                "param_count": report.get("param_count"),
                "layer_tensors_initialized": report.get("layer_tensors_initialized"),
                "projection_scale": report.get("projection_scale"),
                "projection_diagnostic_summary": report.get("projection_diagnostic_summary"),
                "token_report": {
                    key: report.get("token_report", {}).get(key)
                    for key in (
                        "mode",
                        "anchors_used",
                        "shared_decoded_tokens",
                        "embedding_rows_initialized",
                        "special_rows_initialized",
                        "tokenized_rows_initialized",
                        "weighted_tokenized_rows_initialized",
                        "lm_head_rows_initialized",
                        "lm_head_tied_to_embeddings",
                    )
                },
                "report_file": str(Path(stage_output) / "neural_forge_donor_projected_init_report.json"),
            }
        )

    summary = {
        "operation": "multi_donor_projected_init",
        "target_config_model": target_config_model,
        "target_base_model": base_model,
        "target_tokenizer_model": kwargs.get("target_tokenizer_model"),
        "donor_models": donors,
        "donor_weights": [float(weight) for weight in weights],
        "requested_projection_blend_alpha": float(requested_blend_alpha),
        "output_dir": str(output),
        "stage_root": str(stage_root),
        "stage_reports": stage_reports,
        "final_output_dir": str(output),
        "final_report_file": str(output / "neural_forge_donor_projected_init_report.json"),
    }
    _write_json(output / "neural_forge_multi_donor_projected_init_report.json", summary)
    return summary


def _model_path_tag(model: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model).strip()).strip("._-")
    return tag[:80] or "donor"


def _effective_activation_fit_max_rows(
    activation_fit_linears: bool,
    activation_fit_max_rows: int,
    hidden_basis_max_rows: int,
) -> int:
    requested = int(activation_fit_max_rows)
    if activation_fit_linears and requested == 0:
        return int(hidden_basis_max_rows)
    return requested


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
    generation_config_report = _sanitize_generation_config_for_save(host, host_tokenizer)
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
        "generation_config_sanitized": generation_config_report,
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
    eval_suite: str | Path | dict[str, Any] | None = None,
    max_text_evals: int | None = None,
    max_choice_probes: int | None = None,
    max_generation_prompts: int | None = None,
    max_behavior_probes: int | None = None,
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
    eval_kwargs = {
        "eval_suite": eval_suite,
        "max_text_evals": max_text_evals,
        "max_choice_probes": max_choice_probes,
        "max_generation_prompts": max_generation_prompts,
        "max_behavior_probes": max_behavior_probes,
    }

    if baseline_report is None:
        baseline_path = report_root / "baseline_host.json"
        baseline = evaluate_text_model(
            host_model,
            baseline_path,
            torch_dtype=torch_dtype,
            device=device,
            trust_remote_code=trust_remote_code,
            max_new_tokens=max_new_tokens,
            **eval_kwargs,
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
            **eval_kwargs,
        )
        gate = gate_text_report(
            baseline,
            candidate_report,
            output_file=gate_path,
            max_ppl_ratio=max_ppl_ratio,
            ppl_gate_mode="advisory",
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
                eval_suite=eval_suite,
                max_text_evals=max_text_evals,
                max_choice_probes=max_choice_probes,
                max_generation_prompts=max_generation_prompts,
                max_behavior_probes=max_behavior_probes,
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
        "eval_limits": {
            "eval_suite": str(eval_suite) if isinstance(eval_suite, Path) else eval_suite,
            "max_text_evals": max_text_evals,
            "max_choice_probes": max_choice_probes,
            "max_generation_prompts": max_generation_prompts,
            "max_behavior_probes": max_behavior_probes,
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
    eval_suite: str | Path | dict[str, Any] | None = None,
    max_text_evals: int | None = None,
    max_choice_probes: int | None = None,
    max_generation_prompts: int | None = None,
    max_behavior_probes: int | None = None,
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
    eval_kwargs = {
        "eval_suite": eval_suite,
        "max_text_evals": max_text_evals,
        "max_choice_probes": max_choice_probes,
        "max_generation_prompts": max_generation_prompts,
        "max_behavior_probes": max_behavior_probes,
    }

    if baseline_report is None:
        host_baseline_path = report_root / "host_baseline.json"
        host_baseline = evaluate_text_model(
            host_model,
            host_baseline_path,
            torch_dtype=torch_dtype,
            device=device,
            trust_remote_code=trust_remote_code,
            max_new_tokens=max_new_tokens,
            **eval_kwargs,
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
        **eval_kwargs,
    )
    xfamily_host_gate = gate_text_report(
        host_baseline,
        xfamily_eval,
        output_file=xfamily_gate_path,
        max_ppl_ratio=max_ppl_ratio,
        ppl_gate_mode="advisory",
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
        **eval_kwargs,
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
            ppl_gate_mode="advisory",
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
                eval_suite=eval_suite,
                max_text_evals=max_text_evals,
                max_choice_probes=max_choice_probes,
                max_generation_prompts=max_generation_prompts,
                max_behavior_probes=max_behavior_probes,
            )
            recovered_xfamily_gate_path = report_root / f"{tag}_recovered_xfamily_gate.json"
            recovered_xfamily_gate = gate_text_report(
                xfamily_eval,
                _load_report(recovery["report"]),
                output_file=recovered_xfamily_gate_path,
                max_ppl_ratio=max_ppl_ratio,
                ppl_gate_mode="advisory",
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
        "eval_limits": {
            "eval_suite": str(eval_suite) if isinstance(eval_suite, Path) else eval_suite,
            "max_text_evals": max_text_evals,
            "max_choice_probes": max_choice_probes,
            "max_generation_prompts": max_generation_prompts,
            "max_behavior_probes": max_behavior_probes,
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
    eval_suite: str | Path | dict[str, Any] | None = None,
    max_text_evals: int | None = None,
    max_choice_probes: int | None = None,
    max_generation_prompts: int | None = None,
    max_behavior_probes: int | None = None,
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
        behavior_eval_suite=eval_suite,
        behavior_max_probes=6 if max_behavior_probes is None else max_behavior_probes,
        behavior_max_generation_prompts=max_generation_prompts,
    )
    recovered_eval = evaluate_text_model(
        str(recovered_dir),
        recovered_report_path,
        torch_dtype=torch_dtype,
        device=device,
        trust_remote_code=trust_remote_code,
        max_new_tokens=max_new_tokens,
        eval_suite=eval_suite,
        max_text_evals=max_text_evals,
        max_choice_probes=max_choice_probes,
        max_generation_prompts=max_generation_prompts,
        max_behavior_probes=max_behavior_probes,
    )
    recovered_gate = gate_text_report(
        baseline,
        recovered_eval,
        output_file=recovered_gate_path,
        max_ppl_ratio=max_ppl_ratio,
        ppl_gate_mode="advisory",
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


def chain_anchor_staged_compress(
    source_model: str,
    anchor_model: str,
    output_dir: str | Path,
    stages: str | Sequence[str] = ("block", "alm"),
    calibration_texts: Sequence[str] | None = None,
    eval_suite: str | Path | dict[str, Any] | None = None,
    max_text_evals: int | None = None,
    max_choice_probes: int | None = None,
    max_generation_prompts: int | None = None,
    max_behavior_probes: int | None = None,
    max_new_tokens: int = 96,
    stage_max_ppl_ratio: float = 1.25,
    stage_ppl_gate_mode: str = "advisory",
    stage_max_accuracy_drop: float = 0.0,
    stage_max_behavior_drop: float = 0.0,
    stage_min_choice_accuracy: float = 0.0,
    stage_min_behavior_pass_rate: float = 0.0,
    promote_policy: str = "gate-and-rank",
    stage_input_policy: str = "incumbent",
    init_seed: int = 0,
    init_ridge: float = 1e-3,
    init_max_anchors: int = 50_000,
    init_fold_mode: str = "mean",
    init_hidden_basis_mode: str = "none",
    init_hidden_basis_scope: str = "global",
    init_hidden_basis_max_rows: int = 2048,
    init_hidden_basis_preprocess: str = "centered-rowrms",
    init_activation_fold_assignment: str = "balanced",
    init_transport_backend: str = "sparse",
    init_transport_support: str = "centroid",
    init_donor_layer_fold_mode: str = "fold",
    init_donor_layer_group_strategy: str = "depth",
    init_attention_fold_mode: str = "independent",
    init_attention_head_dim_transport: str = "none",
    init_token_init_mode: str = "shared",
    init_projection_scale: str = "none",
    init_projection_blend_alpha: float = 1.0,
    init_consumer_hidden_reduction: str = "mean",
    init_activation_fit_linears: bool = False,
    init_activation_fit_target: str = "all",
    init_activation_fit_max_rows: int = 0,
    init_activation_fit_alpha: float = 1.0,
    init_activation_fit_ridge: float = 1e-1,
    init_activation_fit_rank: int = 0,
    init_activation_fit_rank_mode: str = "absolute",
    init_activation_fit_method: str = "ridge",
    init_activation_fit_factor_steps: int = 0,
    init_activation_fit_factor_lr: float = 3e-2,
    init_activation_fit_factor_l2: float = 0.0,
    init_activation_fit_gate: str = "validation",
    init_activation_fit_validation_fraction: float = 0.20,
    init_activation_fit_min_validation_rows: int = 8,
    init_activation_fit_min_relative_improvement: float = 0.0,
    init_activation_fit_max_cosine_drop: float = 0.01,
    init_activation_feature_mode: str = "stats",
    init_activation_subspace_rank: int = 8,
    init_activation_domain_balance: str = "none",
    init_untie_output_head: bool = False,
    init_activation_max_length: int = 128,
    block_steps_per_layer: int = 4,
    block_batch_size: int = 1,
    block_max_length: int = 112,
    block_lr: float = 2e-6,
    block_lm_weight: float = 0.0,
    block_response_lm_weight: float = 0.0,
    block_first_token_lm_weight: float = 0.0,
    block_response_end_lm_weight: float = 0.0,
    block_response_loss_mode: str = "none",
    block_hidden_ridge: float = 1e-3,
    block_hidden_projection_scope: str = "layer-input",
    block_hidden_metric: str = "projected-rsd",
    block_span_pair_mode: str = "exact",
    block_target: str = "delta",
    block_residual_magnitude_compensation: str = "lowrank-pre",
    block_residual_compensation_min_scale: float = 0.50,
    block_residual_compensation_max_scale: float = 1.50,
    block_residual_compensation_linear_ridge: float = 1e-3,
    block_residual_compensation_alpha: float = 0.01,
    block_residual_compensation_linear_rank: int = 16,
    block_pre_reconstruction_target: str = "none",
    block_pre_reconstruction_ridge: float = 1e-3,
    block_pre_reconstruction_alpha: float = 0.05,
    block_pre_reconstruction_objective: str = "ridge",
    block_pre_reconstruction_cosine_steps: int = 40,
    block_pre_reconstruction_cosine_lr: float = 1e-2,
    block_pre_reconstruction_cosine_l2: float = 1e-4,
    block_pre_reconstruction_fit_gate: str = "validation",
    block_pre_reconstruction_validation_fraction: float = 0.25,
    block_pre_reconstruction_min_validation_rows: int = 32,
    block_pre_reconstruction_max_tokens_per_layer: int = 512,
    block_pre_reconstruction_min_relative_improvement: float = 0.0,
    block_pre_reconstruction_max_cosine_drop: float = 0.01,
    block_max_anchors: int = 2048,
    block_span_max_pairs: int = 512,
    block_donor_layer_fold_mode: str = "fold",
    block_donor_layer_group_strategy: str = "activation-energy",
    block_layer_limit: int | None = None,
    block_layer_gate_metric: str = "behavior",
    block_layer_gate_max_texts: int = 200,
    block_layer_gate_max_generation_prompts: int | None = None,
    block_layer_gate_max_behavior_drop: float = 0.0,
    block_layer_gate_behavior_max_new_tokens: int = 96,
    block_alm_kd_weight: float = 0.0,
    block_projected_vocab_kd_weight: float = 0.0,
    block_byte_kd_weight: float = 0.0,
    block_block_io_kd_weight: float = 0.0,
    block_on_policy_anchor_kd_weight: float = 0.0,
    block_on_policy_unlikelihood_weight: float = 0.0,
    block_on_policy_hidden_kd_weight: float = 0.0,
    block_on_policy_max_new_tokens: int = 24,
    block_on_policy_interval: int = 10,
    block_on_policy_response_loss_mode: str = "auto",
    block_on_policy_unlikelihood_window: int = 16,
    block_stepwise_donor_kd_weight: float = 0.0,
    block_stepwise_donor_max_new_tokens: int = 8,
    block_stepwise_donor_interval: int = 4,
    block_teacher_generate_lm_weight: float = 0.0,
    block_teacher_generate_hidden_kd_weight: float = 0.0,
    block_teacher_generate_projected_vocab_kd_weight: float = 0.0,
    block_teacher_generate_byte_kd_weight: float = 0.0,
    block_teacher_generate_max_new_tokens: int = 64,
    block_alm_max_chunks: int = 64,
    block_alm_min_chars: int = 2,
    block_focused_block_hidden_kd_weight: float = 0.0,
    block_focused_projected_vocab_kd_weight: float = 0.0,
    block_focused_lm_weight: float = 0.0,
    block_token_focus: str = "auto",
    block_projected_vocab_kd_mode: str = "teacher-to-student",
    block_byte_hidden_kd_weight: float = 0.0,
    block_block_io_ridge: float = 1e-3,
    block_block_io_loss: str = "both",
    block_train_output_head_with_kd: bool = False,
    block_untie_output_head_for_kd: bool = False,
    block_allow_tied_output_head_training: bool = False,
    block_module_output_kd_weight: float = 0.0,
    block_module_output_targets: str = "both",
    block_module_output_transport: str = "paired",
    block_module_output_temperature: float = 1.0,
    block_module_output_sinkhorn_iterations: int = 12,
    block_projected_vocab_max_source_tokens: int = 8,
    block_temperature: float = 2.0,
    block_grad_clip: float = 1.0,
    block_report_interval: int | None = None,
    block_save_best_layer_loss: bool = False,
    block_save_best_layer_behavior: bool = False,
    block_layer_behavior_eval_interval: int | None = None,
    block_layer_checkpoint_interval: int = 0,
    block_calibration_order: str = "original",
    alm_steps: int = 80,
    alm_batch_size: int = 1,
    alm_max_length: int = 112,
    alm_lr: float = 2e-6,
    alm_anchor_kd_weight: float = 0.0,
    alm_lm_weight: float = 0.0,
    alm_first_token_lm_weight: float = 0.0,
    alm_response_end_lm_weight: float = 0.0,
    alm_focused_lm_weight: float = 0.0,
    alm_dual_space_kd_weight: float = 0.0,
    alm_attention_influence_weight: float = 0.0,
    alm_attention_influence_layer: str = "last",
    alm_kd_weight: float = 0.05,
    alm_max_chunks: int = 64,
    alm_min_chars: int = 2,
    alm_projected_vocab_kd_weight: float = 0.0,
    alm_focused_projected_vocab_kd_weight: float = 0.0,
    alm_span_hidden_kd_weight: float = 0.0,
    alm_token_kd_weight: float = 0.0,
    alm_token_focus: str = "auto",
    alm_token_max_pairs: int = 512,
    alm_token_teacher_confidence_floor: float = 0.0,
    alm_answer_contrastive_weight: float = 0.0,
    alm_answer_contrastive_max_negatives: int = 4,
    alm_teacher_rank_weight: float = 0.0,
    alm_teacher_rank_temperature: float = 1.0,
    alm_teacher_rank_max_negatives: int = 4,
    alm_teacher_generate_lm_weight: float = 0.0,
    alm_teacher_generate_max_new_tokens: int = 64,
    alm_stepwise_donor_kd_weight: float = 0.0,
    alm_stepwise_donor_max_new_tokens: int = 16,
    alm_on_policy_anchor_kd_weight: float = 0.0,
    alm_on_policy_unlikelihood_weight: float = 0.0,
    alm_on_policy_hidden_kd_weight: float = 0.0,
    alm_on_policy_max_new_tokens: int = 24,
    alm_on_policy_interval: int = 10,
    alm_hidden_kd_metric: str = "projected-rsd",
    alm_train_scope: str = "last4",
    alm_response_loss_mode: str = "none",
    alm_append_eos: bool = False,
    alm_grad_clip: float = 1.0,
    alm_report_interval: int | None = None,
    alm_save_best_behavior: bool = False,
    alm_behavior_eval_interval: int | None = None,
    alm_calibration_order: str = "original",
    torch_dtype: str = "float16",
    candidate_torch_dtype: str = "float32",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    source_weights: Sequence[float] | str | None = None,
) -> dict[str, Any]:
    """Run staged donor-to-child updates while preserving the best incumbent.

    The command is an orchestration harness: each stage writes a normal model,
    evaluates it with the behavior-first suite, and promotes it only when it
    passes the gate against the current incumbent.
    """

    stage_list = _parse_chain_stages(stages)
    source_models, source_weight_values = _parse_source_models_and_weights(source_model, source_weights)
    primary_source_model = source_models[0]
    if promote_policy not in {"gate-and-rank", "gate-only", "always"}:
        raise ValueError("promote_policy must be gate-and-rank, gate-only, or always.")
    if stage_input_policy not in {"incumbent", "sequential"}:
        raise ValueError("stage_input_policy must be incumbent or sequential.")
    if stage_ppl_gate_mode not in {"hard", "advisory"}:
        raise ValueError("stage_ppl_gate_mode must be hard or advisory.")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    output_root = Path(output_dir)
    model_root = output_root / "models"
    report_root = output_root / "reports"
    model_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    anchor_report_path = report_root / "anchor_eval.json"
    anchor_gate_path = report_root / "anchor_self_gate.json"
    anchor_eval = evaluate_text_model(
        anchor_model,
        anchor_report_path,
        torch_dtype=torch_dtype,
        device=device,
        trust_remote_code=trust_remote_code,
        max_new_tokens=max_new_tokens,
        eval_suite=eval_suite,
        max_text_evals=max_text_evals,
        max_choice_probes=max_choice_probes,
        max_generation_prompts=max_generation_prompts,
        max_behavior_probes=max_behavior_probes,
    )
    anchor_gate = gate_text_report(
        anchor_eval,
        anchor_eval,
        output_file=anchor_gate_path,
        max_ppl_ratio=stage_max_ppl_ratio,
        ppl_gate_mode=stage_ppl_gate_mode,
        max_accuracy_drop=stage_max_accuracy_drop,
        max_behavior_drop=stage_max_behavior_drop,
        min_choice_accuracy=stage_min_choice_accuracy,
        min_behavior_pass_rate=stage_min_behavior_pass_rate,
    )
    incumbent = _chain_candidate_record(
        tag="anchor",
        model_dir=str(anchor_model),
        report_path=anchor_report_path,
        gate_path=anchor_gate_path,
        gate=anchor_gate,
        stage="anchor",
        promoted=True,
    )
    working_candidate = incumbent

    stage_records: list[dict[str, Any]] = []
    for index, stage in enumerate(stage_list, start=1):
        stage_tag = f"{index:02d}_{stage}"
        stage_model_dir = model_root / stage_tag
        stage_report_path = report_root / f"{stage_tag}_eval.json"
        stage_gate_path = report_root / f"{stage_tag}_gate.json"
        stage_operation: dict[str, Any]
        input_candidate = working_candidate if stage_input_policy == "sequential" else incumbent
        input_model_dir = str(input_candidate["model_dir"])

        if stage == "init":
            if len(source_models) > 1 or source_weight_values is not None:
                init_project = multi_donor_projected_init
                init_source_kwargs = {
                    "donor_models": source_models,
                    "donor_weights": source_weight_values,
                }
            else:
                init_project = donor_projected_init
                init_source_kwargs = {"donor_model": primary_source_model}
            stage_operation = init_project(
                target_config_model=anchor_model,
                output_dir=stage_model_dir,
                target_base_model=input_model_dir,
                target_tokenizer_model=anchor_model,
                seed=init_seed,
                ridge=init_ridge,
                max_anchors=init_max_anchors,
                fold_mode=init_fold_mode,
                hidden_basis_mode=init_hidden_basis_mode,
                hidden_basis_scope=init_hidden_basis_scope,
                hidden_basis_max_rows=init_hidden_basis_max_rows,
                hidden_basis_preprocess=init_hidden_basis_preprocess,
                activation_fold_assignment=init_activation_fold_assignment,
                transport_backend=init_transport_backend,
                transport_support=init_transport_support,
                donor_layer_fold_mode=init_donor_layer_fold_mode,
                donor_layer_group_strategy=init_donor_layer_group_strategy,
                attention_fold_mode=init_attention_fold_mode,
                attention_head_dim_transport=init_attention_head_dim_transport,
                token_init_mode=init_token_init_mode,
                projection_scale=init_projection_scale,
                projection_blend_alpha=init_projection_blend_alpha,
                consumer_hidden_reduction=init_consumer_hidden_reduction,
                activation_fit_linears=init_activation_fit_linears,
                activation_fit_target=init_activation_fit_target,
                activation_fit_max_rows=init_activation_fit_max_rows,
                activation_fit_alpha=init_activation_fit_alpha,
                activation_fit_ridge=init_activation_fit_ridge,
                activation_fit_rank=init_activation_fit_rank,
                activation_fit_rank_mode=init_activation_fit_rank_mode,
                activation_fit_method=init_activation_fit_method,
                activation_fit_factor_steps=init_activation_fit_factor_steps,
                activation_fit_factor_lr=init_activation_fit_factor_lr,
                activation_fit_factor_l2=init_activation_fit_factor_l2,
                activation_fit_gate=init_activation_fit_gate,
                activation_fit_validation_fraction=init_activation_fit_validation_fraction,
                activation_fit_min_validation_rows=init_activation_fit_min_validation_rows,
                activation_fit_min_relative_improvement=init_activation_fit_min_relative_improvement,
                activation_fit_max_cosine_drop=init_activation_fit_max_cosine_drop,
                activation_feature_mode=init_activation_feature_mode,
                activation_subspace_rank=init_activation_subspace_rank,
                activation_domain_balance=init_activation_domain_balance,
                untie_output_head=init_untie_output_head,
                activation_calibration_texts=list(calibration_texts or []),
                activation_max_length=init_activation_max_length,
                activation_device=device,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                **init_source_kwargs,
            )
        elif stage == "block":
            from neural_forge.hf_recovery import blockwise_reconstruct_cross_tokenizer_causal_lm

            stage_operation = blockwise_reconstruct_cross_tokenizer_causal_lm(
                source_model=primary_source_model,
                candidate_model=input_model_dir,
                output_dir=stage_model_dir,
                steps_per_layer=block_steps_per_layer,
                batch_size=block_batch_size,
                max_length=block_max_length,
                lr=block_lr,
                lm_weight=block_lm_weight,
                response_lm_weight=block_response_lm_weight,
                first_token_lm_weight=block_first_token_lm_weight,
                response_end_lm_weight=block_response_end_lm_weight,
                response_loss_mode=block_response_loss_mode,
                hidden_ridge=block_hidden_ridge,
                hidden_projection_scope=block_hidden_projection_scope,
                hidden_metric=block_hidden_metric,
                span_pair_mode=block_span_pair_mode,
                block_target=block_target,
                residual_magnitude_compensation=block_residual_magnitude_compensation,
                residual_compensation_min_scale=block_residual_compensation_min_scale,
                residual_compensation_max_scale=block_residual_compensation_max_scale,
                residual_compensation_linear_ridge=block_residual_compensation_linear_ridge,
                residual_compensation_alpha=block_residual_compensation_alpha,
                residual_compensation_linear_rank=block_residual_compensation_linear_rank,
                pre_reconstruction_target=block_pre_reconstruction_target,
                pre_reconstruction_ridge=block_pre_reconstruction_ridge,
                pre_reconstruction_alpha=block_pre_reconstruction_alpha,
                pre_reconstruction_objective=block_pre_reconstruction_objective,
                pre_reconstruction_cosine_steps=block_pre_reconstruction_cosine_steps,
                pre_reconstruction_cosine_lr=block_pre_reconstruction_cosine_lr,
                pre_reconstruction_cosine_l2=block_pre_reconstruction_cosine_l2,
                pre_reconstruction_fit_gate=block_pre_reconstruction_fit_gate,
                pre_reconstruction_validation_fraction=block_pre_reconstruction_validation_fraction,
                pre_reconstruction_min_validation_rows=block_pre_reconstruction_min_validation_rows,
                pre_reconstruction_max_tokens_per_layer=block_pre_reconstruction_max_tokens_per_layer,
                pre_reconstruction_min_relative_improvement=block_pre_reconstruction_min_relative_improvement,
                pre_reconstruction_max_cosine_drop=block_pre_reconstruction_max_cosine_drop,
                max_anchors=block_max_anchors,
                span_max_pairs=block_span_max_pairs,
                donor_layer_fold_mode=block_donor_layer_fold_mode,
                donor_layer_group_strategy=block_donor_layer_group_strategy,
                layer_limit=block_layer_limit,
                layer_gate_metric=block_layer_gate_metric,
                layer_gate_max_texts=block_layer_gate_max_texts,
                layer_gate_max_behavior_drop=block_layer_gate_max_behavior_drop,
                layer_gate_behavior_max_new_tokens=block_layer_gate_behavior_max_new_tokens,
                layer_gate_eval_suite=eval_suite,
                layer_gate_max_generation_prompts=block_layer_gate_max_generation_prompts,
                alm_kd_weight=block_alm_kd_weight,
                alm_max_chunks=block_alm_max_chunks,
                alm_min_chars=block_alm_min_chars,
                projected_vocab_kd_weight=block_projected_vocab_kd_weight,
                focused_block_hidden_kd_weight=block_focused_block_hidden_kd_weight,
                focused_projected_vocab_kd_weight=block_focused_projected_vocab_kd_weight,
                focused_lm_weight=block_focused_lm_weight,
                token_focus=block_token_focus,
                projected_vocab_kd_mode=block_projected_vocab_kd_mode,
                byte_kd_weight=block_byte_kd_weight,
                byte_hidden_kd_weight=block_byte_hidden_kd_weight,
                block_io_kd_weight=block_block_io_kd_weight,
                block_io_ridge=block_block_io_ridge,
                block_io_loss=block_block_io_loss,
                on_policy_anchor_kd_weight=block_on_policy_anchor_kd_weight,
                on_policy_unlikelihood_weight=block_on_policy_unlikelihood_weight,
                on_policy_hidden_kd_weight=block_on_policy_hidden_kd_weight,
                on_policy_max_new_tokens=block_on_policy_max_new_tokens,
                on_policy_interval=block_on_policy_interval,
                on_policy_response_loss_mode=block_on_policy_response_loss_mode,
                on_policy_unlikelihood_window=block_on_policy_unlikelihood_window,
                stepwise_donor_kd_weight=block_stepwise_donor_kd_weight,
                stepwise_donor_max_new_tokens=block_stepwise_donor_max_new_tokens,
                stepwise_donor_interval=block_stepwise_donor_interval,
                teacher_generate_lm_weight=block_teacher_generate_lm_weight,
                teacher_generate_hidden_kd_weight=block_teacher_generate_hidden_kd_weight,
                teacher_generate_projected_vocab_kd_weight=block_teacher_generate_projected_vocab_kd_weight,
                teacher_generate_byte_kd_weight=block_teacher_generate_byte_kd_weight,
                teacher_generate_max_new_tokens=block_teacher_generate_max_new_tokens,
                train_output_head_with_kd=block_train_output_head_with_kd,
                untie_output_head_for_kd=block_untie_output_head_for_kd,
                allow_tied_output_head_training=block_allow_tied_output_head_training,
                module_output_kd_weight=block_module_output_kd_weight,
                module_output_targets=block_module_output_targets,
                module_output_transport=block_module_output_transport,
                module_output_temperature=block_module_output_temperature,
                module_output_sinkhorn_iterations=block_module_output_sinkhorn_iterations,
                projected_vocab_max_source_tokens=block_projected_vocab_max_source_tokens,
                temperature=block_temperature,
                torch_dtype=torch_dtype,
                candidate_torch_dtype=candidate_torch_dtype,
                device=device,
                trust_remote_code=trust_remote_code,
                calibration_texts=list(calibration_texts or []),
                calibration_order=block_calibration_order,
                grad_clip=block_grad_clip,
                report_interval=block_report_interval,
                save_best_layer_loss=block_save_best_layer_loss,
                save_best_layer_behavior=block_save_best_layer_behavior,
                layer_behavior_eval_interval=block_layer_behavior_eval_interval,
                layer_checkpoint_interval=block_layer_checkpoint_interval,
            )
        elif stage == "alm":
            from neural_forge.hf_recovery import recover_cross_tokenizer_causal_lm

            stage_operation = recover_cross_tokenizer_causal_lm(
                source_model=primary_source_model,
                candidate_model=input_model_dir,
                output_dir=stage_model_dir,
                steps=alm_steps,
                batch_size=alm_batch_size,
                max_length=alm_max_length,
                lr=alm_lr,
                anchor_kd_weight=alm_anchor_kd_weight,
                lm_weight=alm_lm_weight,
                first_token_lm_weight=alm_first_token_lm_weight,
                response_end_lm_weight=alm_response_end_lm_weight,
                focused_lm_weight=alm_focused_lm_weight,
                dual_space_kd_weight=alm_dual_space_kd_weight,
                attention_influence_weight=alm_attention_influence_weight,
                attention_influence_layer=alm_attention_influence_layer,
                projected_vocab_kd_weight=alm_projected_vocab_kd_weight,
                focused_projected_vocab_kd_weight=alm_focused_projected_vocab_kd_weight,
                span_hidden_kd_weight=alm_span_hidden_kd_weight,
                token_kd_weight=alm_token_kd_weight,
                token_focus=alm_token_focus,
                token_max_pairs=alm_token_max_pairs,
                token_teacher_confidence_floor=alm_token_teacher_confidence_floor,
                answer_contrastive_weight=alm_answer_contrastive_weight,
                answer_contrastive_max_negatives=alm_answer_contrastive_max_negatives,
                teacher_rank_weight=alm_teacher_rank_weight,
                teacher_rank_temperature=alm_teacher_rank_temperature,
                teacher_rank_max_negatives=alm_teacher_rank_max_negatives,
                teacher_generate_lm_weight=alm_teacher_generate_lm_weight,
                teacher_generate_max_new_tokens=alm_teacher_generate_max_new_tokens,
                stepwise_donor_kd_weight=alm_stepwise_donor_kd_weight,
                stepwise_donor_max_new_tokens=alm_stepwise_donor_max_new_tokens,
                on_policy_anchor_kd_weight=alm_on_policy_anchor_kd_weight,
                on_policy_unlikelihood_weight=alm_on_policy_unlikelihood_weight,
                on_policy_hidden_kd_weight=alm_on_policy_hidden_kd_weight,
                on_policy_max_new_tokens=alm_on_policy_max_new_tokens,
                on_policy_interval=alm_on_policy_interval,
                hidden_kd_metric=alm_hidden_kd_metric,
                alm_kd_weight=alm_kd_weight,
                alm_max_chunks=alm_max_chunks,
                alm_min_chars=alm_min_chars,
                response_loss_mode=alm_response_loss_mode,
                append_eos=alm_append_eos,
                train_scope=alm_train_scope,
                torch_dtype=torch_dtype,
                candidate_torch_dtype=candidate_torch_dtype,
                device=device,
                trust_remote_code=trust_remote_code,
                calibration_texts=list(calibration_texts or []),
                calibration_order=alm_calibration_order,
                grad_clip=alm_grad_clip,
                report_interval=alm_report_interval,
                save_best_behavior=alm_save_best_behavior,
                behavior_eval_interval=alm_behavior_eval_interval,
                behavior_eval_suite=eval_suite,
                behavior_max_probes=6 if max_behavior_probes is None else max_behavior_probes,
                behavior_max_generation_prompts=max_generation_prompts,
                behavior_max_new_tokens=max_new_tokens,
            )
        else:
            raise ValueError(f"Unsupported chain stage: {stage}.")

        stage_eval = evaluate_text_model(
            str(stage_model_dir),
            stage_report_path,
            torch_dtype=torch_dtype,
            device=device,
            trust_remote_code=trust_remote_code,
            max_new_tokens=max_new_tokens,
            eval_suite=eval_suite,
            max_text_evals=max_text_evals,
            max_choice_probes=max_choice_probes,
            max_generation_prompts=max_generation_prompts,
            max_behavior_probes=max_behavior_probes,
        )
        stage_gate = gate_text_report(
            incumbent["report"],
            stage_eval,
            output_file=stage_gate_path,
            max_ppl_ratio=stage_max_ppl_ratio,
            ppl_gate_mode=stage_ppl_gate_mode,
            max_accuracy_drop=stage_max_accuracy_drop,
            max_behavior_drop=stage_max_behavior_drop,
            min_choice_accuracy=stage_min_choice_accuracy,
            min_behavior_pass_rate=stage_min_behavior_pass_rate,
        )
        record = _chain_candidate_record(
            tag=stage_tag,
            model_dir=stage_model_dir,
            report_path=stage_report_path,
            gate_path=stage_gate_path,
            gate=stage_gate,
            stage=stage,
            operation_report=stage_operation,
        )
        promoted = _chain_stage_promotes(incumbent, record, stage_gate, promote_policy)
        record["promoted"] = bool(promoted)
        record["incumbent_before"] = incumbent["tag"]
        record["input_tag"] = input_candidate["tag"]
        record["input_model_dir"] = input_model_dir
        if promoted:
            incumbent = record
        record["incumbent_after"] = incumbent["tag"]
        if stage_input_policy == "sequential":
            working_candidate = record
        else:
            working_candidate = incumbent
        record["next_input_after"] = working_candidate["tag"]
        stage_records.append(record)
        _write_chain_anchor_summary(
            output_root,
            source_model,
            anchor_model,
            stage_list,
            incumbent,
            stage_records,
            stage_input_policy=stage_input_policy,
            final_candidate=working_candidate,
            source_models=source_models,
            source_weights=source_weight_values,
            primary_source_model=primary_source_model,
        )

    return _write_chain_anchor_summary(
        output_root,
        source_model,
        anchor_model,
        stage_list,
        incumbent,
        stage_records,
        stage_input_policy=stage_input_policy,
        final_candidate=working_candidate,
        source_models=source_models,
        source_weights=source_weight_values,
        primary_source_model=primary_source_model,
    )


def _parse_chain_stages(stages: str | Sequence[str]) -> list[str]:
    if isinstance(stages, str):
        parts = [part.strip() for part in stages.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in stages if str(part).strip()]
    if not parts:
        raise ValueError("At least one chain stage is required.")
    allowed = {"init", "block", "alm"}
    invalid = [part for part in parts if part not in allowed]
    if invalid:
        raise ValueError(f"Unsupported chain stages: {', '.join(invalid)}.")
    return parts


def _parse_source_models_and_weights(
    source_model: str | Sequence[str],
    source_weights: Sequence[float] | str | None = None,
) -> tuple[list[str], list[float] | None]:
    if isinstance(source_model, str):
        sources = [part.strip() for part in source_model.split(",") if part.strip()]
    else:
        sources = [str(part).strip() for part in source_model if str(part).strip()]
    if not sources:
        raise ValueError("source_model must include at least one donor.")
    if source_weights is None:
        return sources, None
    if isinstance(source_weights, str):
        weights = [float(part.strip()) for part in source_weights.split(",") if part.strip()]
    else:
        weights = [float(item) for item in source_weights]
    if len(weights) != len(sources):
        raise ValueError("source_weights must match source_model donor count.")
    if any((not math.isfinite(weight)) or weight <= 0 for weight in weights):
        raise ValueError("source_weights must be positive finite values.")
    return sources, weights


def _chain_candidate_record(
    tag: str,
    model_dir: str | Path,
    report_path: str | Path,
    gate_path: str | Path,
    gate: dict[str, Any],
    stage: str,
    operation_report: dict[str, Any] | None = None,
    promoted: bool = False,
) -> dict[str, Any]:
    return {
        "tag": str(tag),
        "stage": str(stage),
        "model_dir": str(model_dir),
        "report": str(report_path),
        "gate_report": str(gate_path),
        "passed": bool(gate.get("passed", False)),
        "promoted": bool(promoted),
        "metrics": dict(gate.get("candidate") or {}),
        "gate": gate,
        "operation_report": operation_report,
    }


def _chain_stage_promotes(
    incumbent: dict[str, Any],
    candidate: dict[str, Any],
    gate: dict[str, Any],
    promote_policy: str,
) -> bool:
    if promote_policy == "always":
        return True
    if not bool(gate.get("passed", False)):
        return False
    if promote_policy == "gate-only":
        return True
    if promote_policy != "gate-and-rank":
        raise ValueError("promote_policy must be gate-and-rank, gate-only, or always.")
    return _usable_candidate_sort_key(candidate) <= _usable_candidate_sort_key(incumbent)


def _write_chain_anchor_summary(
    output_root: Path,
    source_model: str,
    anchor_model: str,
    stages: Sequence[str],
    incumbent: dict[str, Any],
    stage_records: Sequence[dict[str, Any]],
    stage_input_policy: str = "incumbent",
    final_candidate: dict[str, Any] | None = None,
    source_models: Sequence[str] | None = None,
    source_weights: Sequence[float] | None = None,
    primary_source_model: str | None = None,
) -> dict[str, Any]:
    final = final_candidate or incumbent
    summary = {
        "operation": "chain_anchor_staged_compress",
        "source_model": source_model,
        "source_models": list(source_models or [source_model]),
        "source_weights": None if source_weights is None else [float(weight) for weight in source_weights],
        "primary_source_model": primary_source_model or source_model,
        "anchor_model": anchor_model,
        "stages": list(stages),
        "stage_input_policy": stage_input_policy,
        "best_model_dir": incumbent["model_dir"],
        "best_tag": incumbent["tag"],
        "best_metrics": incumbent.get("metrics", {}),
        "final_model_dir": final["model_dir"],
        "final_tag": final["tag"],
        "final_metrics": final.get("metrics", {}),
        "stage_count": len(stage_records),
        "promoted_stage_count": sum(1 for item in stage_records if item.get("promoted")),
        "stage_records": list(stage_records),
    }
    report_root = output_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    _write_json(report_root / "chain_anchor_summary.json", summary)
    return summary


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
    return sorted(usable, key=_usable_candidate_sort_key)


def _usable_candidate_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float, float, str]:
    metrics = item.get("metrics") or {}
    behavior = metrics.get("behavior_pass_rate")
    behavior_score = 0.0 if behavior is None else float(behavior)
    ppl = float(metrics.get("perplexity_mean", math.inf))
    if not math.isfinite(ppl):
        ppl = math.inf
    return (
        -behavior_score,
        -float(metrics.get("choice_accuracy", 0.0)),
        -float(metrics.get("generation_health_rate", 0.0)),
        -float(item.get("shrink", 0.0)),
        ppl,
        str(item.get("tag", "")),
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


def _lookup_layer_tensor_key(index: dict[tuple[int, str], str], layer_idx: int, suffix: str) -> str | None:
    for candidate in _layer_suffix_candidates(suffix):
        key = index.get((int(layer_idx), candidate))
        if key is not None:
            return key
    return None


def _lookup_layer_tensor_value(layer_state: dict[str, Any], suffix: str):
    for candidate in _layer_suffix_candidates(suffix):
        value = layer_state.get(candidate)
        if value is not None:
            return value
    return None


def _donor_layer_state_from_index(
    donor_state: dict[str, Any],
    donor_layer_keys: dict[tuple[int, str], str],
    layer_idx: int,
    cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    layer_idx = int(layer_idx)
    cached = cache.get(layer_idx)
    if cached is not None:
        return cached
    layer_state: dict[str, Any] = {}
    for (idx, suffix), key in donor_layer_keys.items():
        if int(idx) == layer_idx:
            layer_state[str(suffix)] = donor_state[key]
    cache[layer_idx] = layer_state
    return layer_state


def _state_value(layer_state: dict[str, Any], *suffixes: str):
    for suffix in suffixes:
        value = _lookup_layer_tensor_value(layer_state, suffix)
        if value is not None:
            return value
    return None


def _zero_bias_from_weight(torch, weight, dim: int):
    return torch.zeros(int(dim), dtype=weight.detach().cpu().to(torch.float32).dtype)


def _synthesize_cross_arch_layer_tensor(torch, layer_state: dict[str, Any], suffix: str):
    """Synthesize common fused/Conv1D target tensors from split decoder donors."""

    marker = f".{str(suffix).lower()}"
    if marker.endswith(".ln_1.weight"):
        value = _state_value(layer_state, "input_layernorm.weight", "attention_norm.weight", "operator_norm.weight")
        if value is not None:
            return value.detach().cpu(), {"method": "layernorm_alias", "source_suffixes": ["input_layernorm.weight"]}
    if marker.endswith(".ln_2.weight"):
        value = _state_value(layer_state, "post_attention_layernorm.weight", "ffn_norm.weight")
        if value is not None:
            return value.detach().cpu(), {"method": "layernorm_alias", "source_suffixes": ["post_attention_layernorm.weight"]}
    if marker.endswith(".ln_1.bias"):
        value = _state_value(layer_state, "input_layernorm.bias", "attention_norm.bias", "operator_norm.bias")
        if value is not None:
            return value.detach().cpu(), {"method": "layernorm_bias_alias", "source_suffixes": ["input_layernorm.bias"]}
        weight = _state_value(layer_state, "input_layernorm.weight", "attention_norm.weight", "operator_norm.weight")
        if weight is not None:
            return _zero_bias_from_weight(torch, weight, int(weight.numel())), {
                "method": "zero_layernorm_bias",
                "source_suffixes": ["input_layernorm.weight"],
            }
    if marker.endswith(".ln_2.bias"):
        value = _state_value(layer_state, "post_attention_layernorm.bias", "ffn_norm.bias")
        if value is not None:
            return value.detach().cpu(), {"method": "layernorm_bias_alias", "source_suffixes": ["post_attention_layernorm.bias"]}
        weight = _state_value(layer_state, "post_attention_layernorm.weight", "ffn_norm.weight")
        if weight is not None:
            return _zero_bias_from_weight(torch, weight, int(weight.numel())), {
                "method": "zero_layernorm_bias",
                "source_suffixes": ["post_attention_layernorm.weight"],
            }
    if marker.endswith(".input_layernorm.weight") or marker.endswith(".attention_norm.weight"):
        value = _state_value(layer_state, "ln_1.weight")
        if value is not None:
            return value.detach().cpu(), {"method": "layernorm_alias", "source_suffixes": ["ln_1.weight"]}
    if marker.endswith(".post_attention_layernorm.weight") or marker.endswith(".ffn_norm.weight"):
        value = _state_value(layer_state, "ln_2.weight")
        if value is not None:
            return value.detach().cpu(), {"method": "layernorm_alias", "source_suffixes": ["ln_2.weight"]}
    if marker.endswith(".input_layernorm.bias") or marker.endswith(".attention_norm.bias"):
        value = _state_value(layer_state, "ln_1.bias")
        if value is not None:
            return value.detach().cpu(), {"method": "layernorm_bias_alias", "source_suffixes": ["ln_1.bias"]}
        weight = _state_value(layer_state, "ln_1.weight")
        if weight is not None:
            return _zero_bias_from_weight(torch, weight, int(weight.numel())), {
                "method": "zero_layernorm_bias",
                "source_suffixes": ["ln_1.weight"],
            }
    if marker.endswith(".post_attention_layernorm.bias") or marker.endswith(".ffn_norm.bias"):
        value = _state_value(layer_state, "ln_2.bias")
        if value is not None:
            return value.detach().cpu(), {"method": "layernorm_bias_alias", "source_suffixes": ["ln_2.bias"]}
        weight = _state_value(layer_state, "ln_2.weight")
        if weight is not None:
            return _zero_bias_from_weight(torch, weight, int(weight.numel())), {
                "method": "zero_layernorm_bias",
                "source_suffixes": ["ln_2.weight"],
            }

    q = _state_value(layer_state, "self_attn.q_proj.weight", "attention.q_proj.weight", "operator.q_proj.weight")
    k = _state_value(layer_state, "self_attn.k_proj.weight", "attention.k_proj.weight", "operator.k_proj.weight")
    v = _state_value(layer_state, "self_attn.v_proj.weight", "attention.v_proj.weight", "operator.v_proj.weight")
    o = _state_value(layer_state, "self_attn.o_proj.weight", "self_attn.out_proj.weight", "attention.o_proj.weight", "operator.out_proj.weight")
    conv_in = _state_value(layer_state, "conv.in_proj.weight", "operator.in_proj.weight")
    conv_out = _state_value(layer_state, "conv.out_proj.weight", "operator.out_proj.weight")
    c_attn = _state_value(layer_state, "attn.c_attn.weight", "self_attn.c_attn.weight")
    c_proj = _state_value(layer_state, "attn.c_proj.weight", "self_attn.c_proj.weight")
    if conv_in is not None and (
        marker.endswith(".self_attn.q_proj.weight")
        or marker.endswith(".attention.q_proj.weight")
        or marker.endswith(".operator.q_proj.weight")
        or marker.endswith(".self_attn.k_proj.weight")
        or marker.endswith(".attention.k_proj.weight")
        or marker.endswith(".operator.k_proj.weight")
        or marker.endswith(".self_attn.v_proj.weight")
        or marker.endswith(".attention.v_proj.weight")
        or marker.endswith(".operator.v_proj.weight")
    ):
        chunks = torch.chunk(conv_in.detach().cpu().to(torch.float32), 3, dim=0)
        if marker.endswith(".q_proj.weight"):
            index = 0
            stream = "q"
        elif marker.endswith(".k_proj.weight"):
            index = 1
            stream = "k"
        else:
            index = 2
            stream = "v"
        return chunks[index].contiguous(), {
            "method": f"split_lfm_conv_in_proj_for_{stream}_proj",
            "source_suffixes": ["conv.in_proj.weight"],
        }
    if conv_out is not None and (
        marker.endswith(".self_attn.o_proj.weight")
        or marker.endswith(".self_attn.out_proj.weight")
        or marker.endswith(".attention.o_proj.weight")
        or marker.endswith(".operator.out_proj.weight")
    ):
        return conv_out.detach().cpu().to(torch.float32).contiguous(), {
            "method": "lfm_conv_out_proj_for_attention_output",
            "source_suffixes": ["conv.out_proj.weight"],
        }
    if marker.endswith(".attn.c_attn.weight") and conv_in is not None:
        return conv_in.detach().cpu().to(torch.float32).t().contiguous(), {
            "method": "transpose_lfm_conv_in_proj_for_gpt2_c_attn",
            "source_suffixes": ["conv.in_proj.weight"],
        }
    if marker.endswith(".attn.c_proj.weight") and conv_out is not None:
        return conv_out.detach().cpu().to(torch.float32).t().contiguous(), {
            "method": "transpose_lfm_conv_out_proj_for_gpt2_c_proj",
            "source_suffixes": ["conv.out_proj.weight"],
        }
    if c_attn is not None and (
        marker.endswith(".self_attn.q_proj.weight")
        or marker.endswith(".attention.q_proj.weight")
        or marker.endswith(".operator.q_proj.weight")
        or marker.endswith(".self_attn.k_proj.weight")
        or marker.endswith(".attention.k_proj.weight")
        or marker.endswith(".operator.k_proj.weight")
        or marker.endswith(".self_attn.v_proj.weight")
        or marker.endswith(".attention.v_proj.weight")
        or marker.endswith(".operator.v_proj.weight")
    ):
        chunks = torch.chunk(c_attn.detach().cpu().to(torch.float32), 3, dim=1)
        if marker.endswith(".q_proj.weight"):
            index = 0
            stream = "q"
        elif marker.endswith(".k_proj.weight"):
            index = 1
            stream = "k"
        else:
            index = 2
            stream = "v"
        return chunks[index].t().contiguous(), {
            "method": f"split_gpt2_c_attn_for_{stream}_proj",
            "source_suffixes": ["attn.c_attn.weight"],
        }
    if c_attn is not None and (
        marker.endswith(".self_attn.q_proj.bias")
        or marker.endswith(".attention.q_proj.bias")
        or marker.endswith(".operator.q_proj.bias")
        or marker.endswith(".self_attn.k_proj.bias")
        or marker.endswith(".attention.k_proj.bias")
        or marker.endswith(".operator.k_proj.bias")
        or marker.endswith(".self_attn.v_proj.bias")
        or marker.endswith(".attention.v_proj.bias")
        or marker.endswith(".operator.v_proj.bias")
    ):
        c_attn_bias = _state_value(layer_state, "attn.c_attn.bias", "self_attn.c_attn.bias")
        width = int(c_attn.shape[1] // 3)
        if c_attn_bias is None:
            chunks = [torch.zeros(width, dtype=torch.float32) for _ in range(3)]
            zeroed = True
        else:
            chunks = list(torch.chunk(c_attn_bias.detach().cpu().to(torch.float32), 3, dim=0))
            zeroed = False
        if marker.endswith(".q_proj.bias"):
            index = 0
            stream = "q"
        elif marker.endswith(".k_proj.bias"):
            index = 1
            stream = "k"
        else:
            index = 2
            stream = "v"
        return chunks[index].contiguous(), {
            "method": f"split_gpt2_c_attn_bias_for_{stream}_proj",
            "source_suffixes": ["attn.c_attn.bias"],
            "missing_biases_zeroed": zeroed,
        }
    if c_proj is not None and (
        marker.endswith(".self_attn.o_proj.weight")
        or marker.endswith(".self_attn.out_proj.weight")
        or marker.endswith(".attention.o_proj.weight")
        or marker.endswith(".operator.out_proj.weight")
    ):
        return c_proj.detach().cpu().to(torch.float32).t().contiguous(), {
            "method": "transpose_gpt2_c_proj_for_o_proj",
            "source_suffixes": ["attn.c_proj.weight"],
        }
    if c_proj is not None and (
        marker.endswith(".self_attn.o_proj.bias")
        or marker.endswith(".self_attn.out_proj.bias")
        or marker.endswith(".attention.o_proj.bias")
        or marker.endswith(".operator.out_proj.bias")
    ):
        bias = _state_value(layer_state, "attn.c_proj.bias", "self_attn.c_proj.bias")
        if bias is not None:
            return bias.detach().cpu(), {"method": "gpt2_c_proj_bias_alias", "source_suffixes": ["attn.c_proj.bias"]}
        return torch.zeros(int(c_proj.shape[1]), dtype=torch.float32), {
            "method": "zero_gpt2_c_proj_bias_for_o_proj",
            "source_suffixes": ["attn.c_proj.weight"],
        }
    if marker.endswith(".attn.c_attn.weight") and q is not None and k is not None and v is not None:
        fused = torch.cat(
            [q.detach().cpu().to(torch.float32), k.detach().cpu().to(torch.float32), v.detach().cpu().to(torch.float32)],
            dim=0,
        ).t().contiguous()
        return fused, {
            "method": "fuse_qkv_transposed_for_gpt2_c_attn",
            "source_suffixes": ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"],
        }
    if marker.endswith(".attn.c_attn.bias") and q is not None and k is not None and v is not None:
        q_bias = _state_value(layer_state, "self_attn.q_proj.bias", "attention.q_proj.bias", "operator.q_proj.bias")
        k_bias = _state_value(layer_state, "self_attn.k_proj.bias", "attention.k_proj.bias", "operator.k_proj.bias")
        v_bias = _state_value(layer_state, "self_attn.v_proj.bias", "attention.v_proj.bias", "operator.v_proj.bias")
        parts = []
        for weight, bias in ((q, q_bias), (k, k_bias), (v, v_bias)):
            if bias is None:
                parts.append(torch.zeros(int(weight.shape[0]), dtype=torch.float32))
            else:
                parts.append(bias.detach().cpu().to(torch.float32))
        return torch.cat(parts, dim=0).contiguous(), {
            "method": "fuse_qkv_bias_for_gpt2_c_attn",
            "source_suffixes": ["self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"],
            "missing_biases_zeroed": True,
        }
    if marker.endswith(".attn.c_proj.weight") and o is not None:
        return o.detach().cpu().to(torch.float32).t().contiguous(), {
            "method": "transpose_o_proj_for_gpt2_c_proj",
            "source_suffixes": ["self_attn.o_proj.weight"],
        }
    if marker.endswith(".attn.c_proj.bias") and o is not None:
        bias = _state_value(layer_state, "self_attn.o_proj.bias", "self_attn.out_proj.bias", "attention.o_proj.bias", "operator.out_proj.bias")
        if bias is not None:
            return bias.detach().cpu(), {"method": "o_proj_bias_alias", "source_suffixes": ["self_attn.o_proj.bias"]}
        return torch.zeros(int(o.shape[0]), dtype=torch.float32), {
            "method": "zero_o_proj_bias_for_gpt2_c_proj",
            "source_suffixes": ["self_attn.o_proj.weight"],
        }

    gate = _state_value(layer_state, "mlp.gate_proj.weight", "mlp.w1.weight", "feed_forward.w1.weight")
    up = _state_value(layer_state, "mlp.up_proj.weight", "mlp.w3.weight", "feed_forward.w3.weight")
    down = _state_value(layer_state, "mlp.down_proj.weight", "mlp.w2.weight", "feed_forward.w2.weight")
    c_fc = _state_value(layer_state, "mlp.c_fc.weight", "feed_forward.c_fc.weight")
    mlp_c_proj = _state_value(layer_state, "mlp.c_proj.weight", "feed_forward.c_proj.weight")
    if c_fc is not None and (
        marker.endswith(".mlp.gate_proj.weight")
        or marker.endswith(".mlp.up_proj.weight")
        or marker.endswith(".feed_forward.w1.weight")
        or marker.endswith(".feed_forward.w3.weight")
    ):
        stream = "gate" if marker.endswith(".gate_proj.weight") or marker.endswith(".w1.weight") else "up"
        return c_fc.detach().cpu().to(torch.float32).t().contiguous(), {
            "method": f"transpose_gpt2_c_fc_for_{stream}_proj",
            "source_suffixes": ["mlp.c_fc.weight"],
        }
    if c_fc is not None and (
        marker.endswith(".mlp.gate_proj.bias")
        or marker.endswith(".mlp.up_proj.bias")
        or marker.endswith(".feed_forward.w1.bias")
        or marker.endswith(".feed_forward.w3.bias")
    ):
        bias = _state_value(layer_state, "mlp.c_fc.bias", "feed_forward.c_fc.bias")
        stream = "gate" if marker.endswith(".gate_proj.bias") or marker.endswith(".w1.bias") else "up"
        if bias is not None:
            return bias.detach().cpu(), {"method": f"gpt2_c_fc_bias_for_{stream}_proj", "source_suffixes": ["mlp.c_fc.bias"]}
        return torch.zeros(int(c_fc.shape[1]), dtype=torch.float32), {
            "method": f"zero_gpt2_c_fc_bias_for_{stream}_proj",
            "source_suffixes": ["mlp.c_fc.weight"],
        }
    if mlp_c_proj is not None and (
        marker.endswith(".mlp.down_proj.weight")
        or marker.endswith(".feed_forward.w2.weight")
    ):
        return mlp_c_proj.detach().cpu().to(torch.float32).t().contiguous(), {
            "method": "transpose_gpt2_mlp_c_proj_for_down_proj",
            "source_suffixes": ["mlp.c_proj.weight"],
        }
    if mlp_c_proj is not None and (
        marker.endswith(".mlp.down_proj.bias")
        or marker.endswith(".feed_forward.w2.bias")
    ):
        bias = _state_value(layer_state, "mlp.c_proj.bias", "feed_forward.c_proj.bias")
        if bias is not None:
            return bias.detach().cpu(), {"method": "gpt2_mlp_c_proj_bias_for_down_proj", "source_suffixes": ["mlp.c_proj.bias"]}
        return torch.zeros(int(mlp_c_proj.shape[1]), dtype=torch.float32), {
            "method": "zero_gpt2_mlp_c_proj_bias_for_down_proj",
            "source_suffixes": ["mlp.c_proj.weight"],
        }
    if marker.endswith(".mlp.c_fc.weight") and (gate is not None or up is not None):
        sources = [item.detach().cpu().to(torch.float32) for item in (gate, up) if item is not None]
        fused = torch.stack(sources, dim=0).mean(dim=0).t().contiguous()
        return fused, {
            "method": "average_gated_mlp_inputs_transposed_for_gpt2_c_fc",
            "source_suffixes": ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
        }
    if marker.endswith(".mlp.c_fc.bias") and (gate is not None or up is not None):
        source_weight = gate if gate is not None else up
        gate_bias = _state_value(layer_state, "mlp.gate_proj.bias", "mlp.w1.bias", "feed_forward.w1.bias")
        up_bias = _state_value(layer_state, "mlp.up_proj.bias", "mlp.w3.bias", "feed_forward.w3.bias")
        biases = [item.detach().cpu().to(torch.float32) for item in (gate_bias, up_bias) if item is not None]
        if biases:
            return torch.stack(biases, dim=0).mean(dim=0).contiguous(), {
                "method": "average_gated_mlp_bias_for_gpt2_c_fc",
                "source_suffixes": ["mlp.gate_proj.bias", "mlp.up_proj.bias"],
            }
        return torch.zeros(int(source_weight.shape[0]), dtype=torch.float32), {
            "method": "zero_mlp_input_bias_for_gpt2_c_fc",
            "source_suffixes": ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
        }
    if marker.endswith(".mlp.c_proj.weight") and down is not None:
        return down.detach().cpu().to(torch.float32).t().contiguous(), {
            "method": "transpose_down_proj_for_gpt2_c_proj",
            "source_suffixes": ["mlp.down_proj.weight"],
        }
    if marker.endswith(".mlp.c_proj.bias") and down is not None:
        bias = _state_value(layer_state, "mlp.down_proj.bias", "mlp.w2.bias", "feed_forward.w2.bias")
        if bias is not None:
            return bias.detach().cpu(), {"method": "down_proj_bias_alias", "source_suffixes": ["mlp.down_proj.bias"]}
        return torch.zeros(int(down.shape[0]), dtype=torch.float32), {
            "method": "zero_down_proj_bias_for_gpt2_c_proj",
            "source_suffixes": ["mlp.down_proj.weight"],
        }
    return None, None


def _layer_suffix_candidates(suffix: str) -> list[str]:
    """Return exact suffix first, then common cross-family decoder aliases."""

    normalized = str(suffix)
    aliases = {
        "feed_forward.w1.weight": ["mlp.gate_proj.weight", "mlp.w1.weight"],
        "feed_forward.w3.weight": ["mlp.up_proj.weight", "mlp.w3.weight"],
        "feed_forward.w2.weight": ["mlp.down_proj.weight", "mlp.w2.weight"],
        "feed_forward.w1.bias": ["mlp.gate_proj.bias", "mlp.w1.bias"],
        "feed_forward.w3.bias": ["mlp.up_proj.bias", "mlp.w3.bias"],
        "feed_forward.w2.bias": ["mlp.down_proj.bias", "mlp.w2.bias"],
        "operator.q_proj.weight": ["self_attn.q_proj.weight", "attention.q_proj.weight"],
        "operator.k_proj.weight": ["self_attn.k_proj.weight", "attention.k_proj.weight"],
        "operator.v_proj.weight": ["self_attn.v_proj.weight", "attention.v_proj.weight"],
        "operator.out_proj.weight": ["self_attn.o_proj.weight", "attention.o_proj.weight"],
        "operator.q_proj.bias": ["self_attn.q_proj.bias", "attention.q_proj.bias"],
        "operator.k_proj.bias": ["self_attn.k_proj.bias", "attention.k_proj.bias"],
        "operator.v_proj.bias": ["self_attn.v_proj.bias", "attention.v_proj.bias"],
        "operator.out_proj.bias": ["self_attn.o_proj.bias", "attention.o_proj.bias"],
        "attention.q_proj.weight": ["self_attn.q_proj.weight", "operator.q_proj.weight"],
        "attention.k_proj.weight": ["self_attn.k_proj.weight", "operator.k_proj.weight"],
        "attention.v_proj.weight": ["self_attn.v_proj.weight", "operator.v_proj.weight"],
        "attention.out_proj.weight": ["self_attn.o_proj.weight", "operator.out_proj.weight"],
        "attention.o_proj.weight": ["self_attn.o_proj.weight", "operator.out_proj.weight"],
        "self_attn.out_proj.weight": ["self_attn.o_proj.weight", "attention.o_proj.weight", "operator.out_proj.weight"],
        "attention.q_proj.bias": ["self_attn.q_proj.bias", "operator.q_proj.bias"],
        "attention.k_proj.bias": ["self_attn.k_proj.bias", "operator.k_proj.bias"],
        "attention.v_proj.bias": ["self_attn.v_proj.bias", "operator.v_proj.bias"],
        "attention.out_proj.bias": ["self_attn.o_proj.bias", "operator.out_proj.bias"],
        "attention.o_proj.bias": ["self_attn.o_proj.bias", "operator.out_proj.bias"],
        "self_attn.out_proj.bias": ["self_attn.o_proj.bias", "attention.o_proj.bias", "operator.out_proj.bias"],
        "self_attn.q_layernorm.weight": ["self_attn.q_norm.weight", "attention.q_norm.weight", "operator.q_norm.weight"],
        "self_attn.k_layernorm.weight": ["self_attn.k_norm.weight", "attention.k_norm.weight", "operator.k_norm.weight"],
        "attention.q_layernorm.weight": ["self_attn.q_norm.weight", "attention.q_norm.weight", "operator.q_norm.weight"],
        "attention.k_layernorm.weight": ["self_attn.k_norm.weight", "attention.k_norm.weight", "operator.k_norm.weight"],
        "operator.q_layernorm.weight": ["self_attn.q_norm.weight", "attention.q_norm.weight", "operator.q_norm.weight"],
        "operator.k_layernorm.weight": ["self_attn.k_norm.weight", "attention.k_norm.weight", "operator.k_norm.weight"],
        "operator_norm.weight": ["input_layernorm.weight"],
        "operator_norm.bias": ["input_layernorm.bias"],
        "attention_norm.weight": ["input_layernorm.weight", "operator_norm.weight"],
        "attention_norm.bias": ["input_layernorm.bias", "operator_norm.bias"],
        "ffn_norm.weight": ["post_attention_layernorm.weight"],
        "ffn_norm.bias": ["post_attention_layernorm.bias"],
    }
    candidates = []
    queue = [normalized]
    while queue:
        candidate = queue.pop(0)
        if candidate in candidates:
            continue
        candidates.append(candidate)
        queue.extend(item for item in aliases.get(candidate, []) if item not in candidates)
        for source, targets in aliases.items():
            if candidate in targets and source not in candidates:
                queue.append(source)
    return candidates


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


def _neutralized_unmatched_operator_tensor(torch, target_tensor, suffix: str):
    if not hasattr(target_tensor, "is_floating_point") or not target_tensor.is_floating_point():
        return None
    if suffix.endswith("conv.out_proj.weight") or suffix.endswith("conv.out_proj.bias"):
        return torch.zeros_like(target_tensor.detach().cpu())
    return None


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
    transport_support: str = "centroid",
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
            transport_support=transport_support,
            assignment_cache=assignment_cache,
        )
        projected = _scale_projection_to_target(
            torch,
            projected,
            target_tensor,
            projection_scale,
            source_tensor=donor_tensor,
        )
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


def _scale_projection_to_target(torch, projected, target_tensor, projection_scale: str, source_tensor=None):
    if projection_scale == "none":
        return projected
    if projection_scale not in {"target-rms", "source-rms", "geomean-rms"}:
        raise ValueError("projection_scale must be none, target-rms, source-rms, or geomean-rms.")
    if projection_scale in {"source-rms", "geomean-rms"}:
        if source_tensor is None:
            raise ValueError(f"{projection_scale} projection scale requires source_tensor.")
        source_values = source_tensor.detach().cpu().to(torch.float32)
        if projection_scale == "source-rms":
            reference_rms = source_values.pow(2).mean().sqrt()
        else:
            target_values = target_tensor.detach().cpu().to(torch.float32)
            source_rms = source_values.pow(2).mean().sqrt()
            target_rms = target_values.pow(2).mean().sqrt()
            reference_rms = (source_rms * target_rms).sqrt()
    else:
        target_values = target_tensor.detach().cpu().to(torch.float32)
        reference_rms = target_values.pow(2).mean().sqrt()
    projected_values = projected.to(torch.float32)
    projected_rms = projected_values.pow(2).mean().sqrt()
    return projected_values * (reference_rms / projected_rms.clamp_min(1e-12))


def _blend_projection_with_target(torch, projected, target_tensor, alpha: float):
    alpha = _bounded_alpha(alpha, "projection_blend_alpha")
    if alpha == 1.0:
        return projected
    target_values = target_tensor.detach().cpu().to(torch.float32)
    return (float(alpha) * projected.to(torch.float32)) + ((1.0 - float(alpha)) * target_values)


def _sample_tensor_values(torch, tensor, max_values: int = 262_144):
    values = tensor.detach().flatten()
    if values.numel() > int(max_values):
        indices = torch.linspace(0, values.numel() - 1, steps=int(max_values), dtype=torch.long, device=values.device)
        values = values.index_select(0, indices)
    return values.cpu().to(torch.float32)


def _sample_paired_tensor_values(torch, left, right, max_values: int = 262_144):
    left_values = left.detach().flatten()
    right_values = right.detach().flatten()
    if left_values.numel() != right_values.numel():
        return None, None
    if left_values.numel() > int(max_values):
        indices = torch.linspace(0, left_values.numel() - 1, steps=int(max_values), dtype=torch.long)
        left_values = left_values.index_select(0, indices.to(left_values.device))
        right_values = right_values.index_select(0, indices.to(right_values.device))
    return left_values.cpu().to(torch.float32), right_values.cpu().to(torch.float32)


def _tensor_stats(torch, tensor, max_values: int = 262_144) -> dict[str, Any]:
    values = _sample_tensor_values(torch, tensor, max_values=max_values)
    numel = int(tensor.numel())
    if numel == 0:
        return {
            "numel": 0,
            "sampled_numel": 0,
            "finite_fraction": 1.0,
            "rms": None,
            "mean_abs": None,
            "max_abs": None,
        }
    finite_mask = torch.isfinite(values)
    finite_fraction = float(finite_mask.to(torch.float32).mean().item())
    if not bool(finite_mask.any().item()):
        return {
            "numel": numel,
            "sampled_numel": int(values.numel()),
            "finite_fraction": finite_fraction,
            "rms": None,
            "mean_abs": None,
            "max_abs": None,
        }
    finite_values = values if finite_fraction == 1.0 else values[finite_mask]
    abs_values = finite_values.abs()
    return {
        "numel": numel,
        "sampled_numel": int(values.numel()),
        "finite_fraction": finite_fraction,
        "rms": float(finite_values.pow(2).mean().sqrt().item()),
        "mean_abs": float(abs_values.mean().item()),
        "max_abs": float(abs_values.max().item()),
    }


def _safe_numeric_ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None:
        return None
    numerator = float(numerator)
    denominator = float(denominator)
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) < 1e-12:
        return None
    return float(numerator / denominator)


def _same_shape_cosine(torch, left, right) -> float | None:
    left_values, right_values = _sample_paired_tensor_values(torch, left, right)
    if left_values is None or right_values is None or left_values.numel() == 0:
        return None
    finite = torch.isfinite(left_values) & torch.isfinite(right_values)
    if not bool(finite.any().item()):
        return None
    left_values = left_values[finite]
    right_values = right_values[finite]
    denominator = left_values.norm() * right_values.norm()
    if float(denominator.item()) < 1e-12:
        return None
    return float(torch.dot(left_values, right_values).div(denominator).item())


def _tensor_projection_diagnostic(torch, source_tensor, projected_tensor, target_tensor) -> dict[str, Any]:
    source_stats = _tensor_stats(torch, source_tensor)
    projected_stats = _tensor_stats(torch, projected_tensor)
    target_stats = _tensor_stats(torch, target_tensor)
    projected_values, target_values = _sample_paired_tensor_values(torch, projected_tensor, target_tensor)
    update_rms = None
    if projected_values is not None and target_values is not None and projected_values.numel() > 0:
        delta = projected_values - target_values
        finite_delta = delta[torch.isfinite(delta)]
        if finite_delta.numel() > 0:
            update_rms = float(finite_delta.pow(2).mean().sqrt().item())
    finite_fractions = [
        float(source_stats["finite_fraction"]),
        float(projected_stats["finite_fraction"]),
        float(target_stats["finite_fraction"]),
    ]
    return {
        "source": source_stats,
        "projected": projected_stats,
        "target_before": target_stats,
        "projected_to_source_rms_ratio": _safe_numeric_ratio(projected_stats["rms"], source_stats["rms"]),
        "projected_to_target_rms_ratio": _safe_numeric_ratio(projected_stats["rms"], target_stats["rms"]),
        "target_update_rms": update_rms,
        "target_update_relative_rms": _safe_numeric_ratio(update_rms, target_stats["rms"]),
        "target_projection_cosine": _same_shape_cosine(torch, projected_tensor, target_tensor),
        "nonfinite_detected": any(item < 1.0 for item in finite_fractions),
    }


def _numeric_report_values(reports: Sequence[dict[str, Any]], key: str) -> list[float]:
    values = []
    for report in reports:
        value = report.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _series_summary(values: Sequence[float]) -> dict[str, float] | None:
    values = [float(item) for item in values if math.isfinite(float(item))]
    if not values:
        return None
    ordered = sorted(values)
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "p95": float(ordered[min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)]),
    }


def _summarize_projection_diagnostics(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reports = list(reports)
    return {
        "count": len(reports),
        "nonfinite_detected_count": sum(1 for item in reports if bool(item.get("nonfinite_detected"))),
        "projected_to_source_rms_ratio": _series_summary(
            _numeric_report_values(reports, "projected_to_source_rms_ratio")
        ),
        "projected_to_target_rms_ratio": _series_summary(
            _numeric_report_values(reports, "projected_to_target_rms_ratio")
        ),
        "target_update_relative_rms": _series_summary(
            _numeric_report_values(reports, "target_update_relative_rms")
        ),
        "target_projection_cosine": _series_summary(_numeric_report_values(reports, "target_projection_cosine")),
    }


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


def _donor_layer_groups_for_target_count(
    torch,
    target_layer_count: int,
    donor_layer_count: int,
    strategy: str = "depth",
    activation_profile: dict[str, Any] | None = None,
) -> tuple[list[list[int]], dict[str, Any]]:
    if target_layer_count <= 0 or donor_layer_count <= 0:
        raise ValueError("Layer counts must be positive.")
    if strategy not in {"depth", "activation-energy", "activation-geometry"}:
        raise ValueError("donor layer group strategy must be depth, activation-energy, or activation-geometry.")
    if strategy == "depth":
        groups = [
            _donor_layer_group_for_target(idx, int(target_layer_count), int(donor_layer_count))
            for idx in range(int(target_layer_count))
        ]
        return groups, {"strategy": "depth", "groups": groups, "delta_scores": None}
    if activation_profile is None:
        raise ValueError(f"{strategy} donor layer grouping requires an activation profile.")
    if strategy == "activation-geometry":
        geometry = _donor_layer_geometry_scores(torch, activation_profile, int(donor_layer_count))
        scores = geometry["geometry_scores"]
        groups = _donor_layer_groups_by_energy(scores, int(target_layer_count))
        return groups, {
            "strategy": "activation-geometry",
            "groups": [[int(item) for item in group] for group in groups],
            "delta_scores": [float(item) for item in geometry["delta_scores"]],
            "representative_scores": [float(item) for item in scores],
            "geometry_scores": [float(item) for item in scores],
            "total_geometry_score": float(sum(scores)),
            "layer_geometry": geometry["layer_geometry"],
        }
    scores = _donor_layer_delta_scores(torch, activation_profile, int(donor_layer_count))
    groups = _donor_layer_groups_by_energy(scores, int(target_layer_count))
    return groups, {
        "strategy": "activation-energy",
        "groups": groups,
        "delta_scores": [float(item) for item in scores],
        "total_delta_score": float(sum(scores)),
    }


def _donor_layer_delta_scores(torch, activation_profile: dict[str, Any], donor_layer_count: int) -> list[float]:
    input_rows_by_layer = activation_profile.get("hidden_basis_input_rows") or {}
    output_rows_by_layer = activation_profile.get("hidden_basis_output_rows") or {}
    scores: list[float] = []
    for layer_idx in range(int(donor_layer_count)):
        input_rows = input_rows_by_layer.get(int(layer_idx))
        output_rows = output_rows_by_layer.get(int(layer_idx))
        if input_rows is None or output_rows is None:
            scores.append(1.0)
            continue
        rows = min(int(input_rows.shape[0]), int(output_rows.shape[0]))
        dims = min(int(input_rows.shape[1]), int(output_rows.shape[1]))
        if rows <= 0 or dims <= 0:
            scores.append(1.0)
            continue
        delta = output_rows[:rows, :dims].to(torch.float32) - input_rows[:rows, :dims].to(torch.float32)
        score = float(delta.pow(2).mean().sqrt().detach().cpu())
        scores.append(max(score, 1e-12))
    if not any(score > 1e-8 for score in scores):
        return [1.0 for _ in scores]
    return scores


def _donor_layer_geometry_scores(
    torch,
    activation_profile: dict[str, Any],
    donor_layer_count: int,
) -> dict[str, Any]:
    return _activation_layer_geometry_scores(
        torch,
        activation_profile.get("hidden_basis_input_rows") or {},
        activation_profile.get("hidden_basis_output_rows") or {},
        donor_layer_count,
    )


def _activation_layer_geometry_scores(
    torch,
    input_rows_by_layer: dict[int, Any],
    output_rows_by_layer: dict[int, Any],
    layer_count: int,
) -> dict[str, Any]:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive.")
    delta_scores: list[float] = []
    geometry_scores: list[float] = []
    layer_geometry: list[dict[str, Any]] = []
    previous_delta = None
    for layer_idx in range(int(layer_count)):
        input_rows = input_rows_by_layer.get(int(layer_idx))
        output_rows = output_rows_by_layer.get(int(layer_idx))
        if input_rows is None or output_rows is None:
            delta_scores.append(1.0)
            geometry_scores.append(1.0)
            layer_geometry.append(
                {
                    "layer": int(layer_idx),
                    "delta_score": 1.0,
                    "geometry_score": 1.0,
                    "linear_cka": None,
                    "mean_row_cosine": None,
                    "delta_novelty": None,
                    "fallback": "missing_activation_rows",
                }
            )
            previous_delta = None
            continue
        input_matrix, output_matrix = _paired_activation_matrices(torch, input_rows, output_rows)
        if input_matrix is None or output_matrix is None:
            delta_scores.append(1.0)
            geometry_scores.append(1.0)
            layer_geometry.append(
                {
                    "layer": int(layer_idx),
                    "delta_score": 1.0,
                    "geometry_score": 1.0,
                    "linear_cka": None,
                    "mean_row_cosine": None,
                    "delta_novelty": None,
                    "fallback": "empty_activation_rows",
                }
            )
            previous_delta = None
            continue
        delta = output_matrix - input_matrix
        delta_score = float(delta.pow(2).mean().sqrt().detach().cpu())
        cka = _safe_linear_cka(torch, input_matrix, output_matrix)
        row_cosine = _safe_mean_row_cosine(torch, input_matrix, output_matrix)
        novelty = 1.0 if previous_delta is None else _delta_novelty(torch, delta, previous_delta)
        previous_delta = delta.detach().cpu()
        cka_shift = 1.0 - max(0.0, min(1.0, cka))
        row_shift = 0.5 * (1.0 - max(-1.0, min(1.0, row_cosine)))
        geometry_multiplier = 1.0 + cka_shift + row_shift + (0.5 * max(0.0, min(2.0, novelty)))
        geometry_score = max(delta_score, 1e-12) * max(geometry_multiplier, 1e-6)
        delta_scores.append(max(delta_score, 1e-12))
        geometry_scores.append(max(float(geometry_score), 1e-12))
        layer_geometry.append(
            {
                "layer": int(layer_idx),
                "delta_score": float(max(delta_score, 1e-12)),
                "geometry_score": float(max(geometry_score, 1e-12)),
                "linear_cka": float(cka),
                "mean_row_cosine": float(row_cosine),
                "delta_novelty": float(novelty),
                "geometry_multiplier": float(geometry_multiplier),
            }
        )
    if not any(score > 1e-8 for score in geometry_scores):
        geometry_scores = [1.0 for _ in geometry_scores]
    if not any(score > 1e-8 for score in delta_scores):
        delta_scores = [1.0 for _ in delta_scores]
    return {
        "delta_scores": [float(max(score, 1e-12)) for score in delta_scores],
        "geometry_scores": [float(max(score, 1e-12)) for score in geometry_scores],
        "layer_geometry": layer_geometry,
    }


def _paired_activation_matrices(torch, first, second):
    first_matrix = first.detach().cpu().to(torch.float32)
    second_matrix = second.detach().cpu().to(torch.float32)
    if first_matrix.dim() > 2:
        first_matrix = first_matrix.reshape(-1, first_matrix.shape[-1])
    if second_matrix.dim() > 2:
        second_matrix = second_matrix.reshape(-1, second_matrix.shape[-1])
    if first_matrix.dim() != 2 or second_matrix.dim() != 2:
        return None, None
    rows = min(int(first_matrix.shape[0]), int(second_matrix.shape[0]))
    dims = min(int(first_matrix.shape[1]), int(second_matrix.shape[1]))
    if rows <= 0 or dims <= 0:
        return None, None
    return first_matrix[:rows, :dims], second_matrix[:rows, :dims]


def _safe_linear_cka(torch, first, second) -> float:
    if int(first.shape[0]) < 2 or int(second.shape[0]) < 2:
        return 1.0
    try:
        value = float(_linear_cka(torch, first, second))
    except Exception:
        return 1.0
    if not math.isfinite(value):
        return 1.0
    return value


def _safe_mean_row_cosine(torch, first, second) -> float:
    try:
        value = float(
            torch.nn.functional.cosine_similarity(
                torch.nan_to_num(first.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4),
                torch.nan_to_num(second.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4),
                dim=1,
                eps=1e-8,
            )
            .mean()
            .detach()
            .cpu()
        )
    except Exception:
        return 1.0
    if not math.isfinite(value):
        return 1.0
    return value


def _delta_novelty(torch, current_delta, previous_delta) -> float:
    current = current_delta.detach().cpu().to(torch.float32).flatten()
    previous = previous_delta.detach().cpu().to(torch.float32).flatten()
    size = min(int(current.numel()), int(previous.numel()))
    if size <= 0:
        return 1.0
    current = torch.nan_to_num(current[:size], nan=0.0, posinf=1e4, neginf=-1e4)
    previous = torch.nan_to_num(previous[:size], nan=0.0, posinf=1e4, neginf=-1e4)
    cosine = torch.nn.functional.cosine_similarity(current.unsqueeze(0), previous.unsqueeze(0), dim=1, eps=1e-8)
    value = float((1.0 - cosine.abs()).detach().cpu())
    if not math.isfinite(value):
        return 1.0
    return max(0.0, min(2.0, value))


def _donor_layer_groups_by_energy(delta_scores: Sequence[float], target_layer_count: int) -> list[list[int]]:
    scores = [max(float(score), 1e-12) for score in delta_scores]
    donor_layer_count = len(scores)
    if target_layer_count <= 0 or donor_layer_count <= 0:
        raise ValueError("Layer counts must be positive.")
    if target_layer_count >= donor_layer_count:
        return [
            _donor_layer_group_for_target(idx, int(target_layer_count), int(donor_layer_count))
            for idx in range(int(target_layer_count))
        ]
    total = sum(scores)
    target_energy = total / float(target_layer_count)
    ratio = donor_layer_count / float(target_layer_count)
    max_group_size = max(1, int(math.ceil(ratio)) + (0 if ratio < 2.0 else 1))
    prefix = [0.0]
    for score in scores:
        prefix.append(prefix[-1] + float(score))
    inf = float("inf")
    dp = [[inf for _ in range(donor_layer_count + 1)] for _ in range(int(target_layer_count) + 1)]
    back = [[0 for _ in range(donor_layer_count + 1)] for _ in range(int(target_layer_count) + 1)]
    dp[0][0] = 0.0
    for group_idx in range(1, int(target_layer_count) + 1):
        min_layers = group_idx
        max_layers = min(donor_layer_count, group_idx * max_group_size)
        for end in range(min_layers, max_layers + 1):
            best_cost = inf
            best_size = 1
            for size in range(1, min(max_group_size, end) + 1):
                start = end - size
                if dp[group_idx - 1][start] == inf:
                    continue
                group_energy = prefix[end] - prefix[start]
                cost = dp[group_idx - 1][start] + abs(group_energy - target_energy)
                if cost < best_cost:
                    best_cost = cost
                    best_size = size
            dp[group_idx][end] = best_cost
            back[group_idx][end] = best_size
    if dp[int(target_layer_count)][donor_layer_count] == inf:
        return [
            _donor_layer_group_for_target(idx, int(target_layer_count), int(donor_layer_count))
            for idx in range(int(target_layer_count))
        ]
    groups: list[list[int]] = []
    end = donor_layer_count
    for group_idx in range(int(target_layer_count), 0, -1):
        size = int(back[group_idx][end])
        start = end - size
        groups.append(list(range(start, end)))
        end = start
    return list(reversed(groups))


def _representative_donor_layer_for_group(group: Sequence[int], delta_scores: Sequence[float] | None = None) -> int:
    if not group:
        raise ValueError("Donor layer group cannot be empty.")
    if delta_scores:
        return max((int(item) for item in group), key=lambda idx: float(delta_scores[idx]))
    return int(group[len(group) // 2])


def _folded_donor_layer_state(
    torch,
    donor_layers: Sequence[Any],
    target_layer_idx: int,
    target_layer_count: int,
    donor_layer_fold_mode: str,
    donor_layer_group: Sequence[int] | None,
    cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if donor_layer_fold_mode not in {"average", "fold"}:
        raise ValueError("donor_layer_fold_mode must be average or fold for folded donor states.")
    if target_layer_idx not in cache:
        group = (
            [int(item) for item in donor_layer_group]
            if donor_layer_group is not None
            else _donor_layer_group_for_target(target_layer_idx, target_layer_count, len(donor_layers))
        )
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
    transport_support: str = "centroid",
    assignment_cache: dict[Any, dict[str, Any]] | None = None,
):
    if fold_mode not in {"mean", "rms", "transport"}:
        raise ValueError("fold_mode must be mean, rms, or transport.")
    if activation_fold_assignment not in {"balanced", "similarity"}:
        raise ValueError("activation_fold_assignment must be balanced or similarity.")
    if transport_backend not in {"sparse", "sinkhorn"}:
        raise ValueError("transport_backend must be sparse or sinkhorn.")
    if transport_support not in {"centroid", "importance", "quantile", "medoid", "quantile-medoid"}:
        raise ValueError("transport_support must be centroid, importance, quantile, medoid, or quantile-medoid.")
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
                transport_support=transport_support,
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
    transport_support: str = "centroid",
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
            transport_support=transport_support,
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
                transport_support=transport_support,
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
    transport_support: str = "centroid",
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
        transport_support=transport_support,
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
        transport_support=transport_support,
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


def _build_shared_token_hidden_basis(
    torch,
    target,
    donor,
    target_tokenizer,
    donor_tokenizer,
    max_anchors: int,
    ridge: float,
):
    if max_anchors <= 0:
        raise ValueError("max_anchors must be positive.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    anchors, target_map, donor_map = _shared_anchor_tokens(target_tokenizer, donor_tokenizer, max_anchors)
    target_embeddings = target.get_input_embeddings()
    donor_embeddings = donor.get_input_embeddings()
    if target_embeddings is None or donor_embeddings is None:
        raise ValueError("Both target and donor need input embeddings for shared-token hidden basis.")
    target_idx = torch.tensor([target_map[token] for token in anchors], dtype=torch.long)
    donor_idx = torch.tensor([donor_map[token] for token in anchors], dtype=torch.long)
    with torch.no_grad():
        target_rows = target_embeddings.weight.index_select(0, target_idx).detach().cpu().to(torch.float32)
        donor_rows = donor_embeddings.weight.index_select(0, donor_idx).detach().cpu().to(torch.float32)
    basis = _ridge_map(torch, donor_rows, target_rows, ridge).detach().cpu().to(torch.float32)
    predicted = donor_rows.matmul(basis)
    return basis.contiguous(), {
        "mode": "shared-token-ridge",
        "source_dim": int(donor_rows.shape[1]),
        "target_dim": int(target_rows.shape[1]),
        "anchors": int(len(anchors)),
        "available_shared_tokens": int(len(set(target_map) & set(donor_map))),
        "ridge": float(ridge),
        "relative_error": _relative_reconstruction_error(torch, predicted, target_rows),
        "mean_cosine": _mean_row_cosine(torch, predicted, target_rows),
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


def _axis_assignment_cache_key(
    base_key,
    source_dim: int,
    target_dim: int,
    assignment_mode: str,
    fold_mode: str = "mean",
    transport_backend: str | None = None,
    transport_support: str | None = None,
):
    if base_key is None:
        return None
    key = (_freeze_assignment_key(base_key), int(source_dim), int(target_dim), str(assignment_mode), str(fold_mode))
    if str(fold_mode) == "transport":
        key = (*key, str(transport_backend or ""), str(transport_support or ""))
    return key


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
        support_indices = item.get("target_support_indices")
        report.append(
            {
                "key": repr(key),
                "map_type": map_type,
                "transport_backend": item.get("transport_backend"),
                "transport_support": item.get("transport_support"),
                "target_masses": None
                if item.get("target_masses") is None
                else item["target_masses"].tolist(),
                "target_support_indices": support_indices.tolist() if support_indices is not None else None,
                "transport_matrix_shape": list(transport_matrix.shape) if transport_matrix is not None else None,
                "source_dim": int(item.get("source_dim", 0)),
                "target_dim": int(item.get("target_dim", 0)),
                "assignment_mode": str(item.get("assignment_mode", "")),
            }
        )
    return report


def _activation_quantile_order(torch, feature_rows, fold_weights):
    source_dim = int(feature_rows.shape[0])
    weights = fold_weights.detach().cpu().to(torch.float32).flatten()
    if int(weights.numel()) != source_dim:
        raise ValueError("Quantile support weights must match feature rows.")
    keys = weights.clone()
    if float((keys.max() - keys.min()).item()) <= 1e-12:
        rows = feature_rows.detach().cpu().to(torch.float32)
        centered = rows - rows.mean(dim=0, keepdim=True)
        if int(centered.shape[1]) > 0 and float(centered.pow(2).sum().item()) > 1e-12:
            try:
                _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
                keys = centered.matmul(vh[0])
            except Exception:
                keys = rows.norm(dim=1)
        else:
            keys = torch.arange(source_dim, dtype=torch.float32)
    return torch.argsort(keys, descending=False)


def _activation_quantile_support_indices(torch, feature_rows, fold_weights, target_dim: int):
    source_dim = int(feature_rows.shape[0])
    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError("target_dim must be positive.")
    if source_dim <= target_dim:
        return torch.arange(source_dim, dtype=torch.long)
    order = _activation_quantile_order(torch, feature_rows, fold_weights)
    positions = torch.linspace(0, source_dim - 1, steps=target_dim)
    selected = [int(order[int(round(float(pos.item())))].item()) for pos in positions]
    if len(set(selected)) < target_dim:
        selected_set = set(selected)
        for idx in order.tolist():
            idx = int(idx)
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            if len(selected_set) == target_dim:
                break
    return torch.tensor(selected[:target_dim], dtype=torch.long).sort().values


def _activation_quantile_medoid_support_indices(torch, feature_rows, fold_weights, target_dim: int):
    source_dim = int(feature_rows.shape[0])
    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError("target_dim must be positive.")
    if source_dim <= target_dim:
        return torch.arange(source_dim, dtype=torch.long)
    rows = _row_normalize(feature_rows.detach().cpu().to(torch.float32))
    weights = fold_weights.detach().cpu().to(torch.float32).flatten().clamp_min(1e-12)
    if int(weights.numel()) != source_dim:
        raise ValueError("Quantile-medoid support weights must match feature rows.")
    order = _activation_quantile_order(torch, feature_rows, weights)
    selected: list[int] = []
    selected_set: set[int] = set()
    for target_idx in range(target_dim):
        start = int(round(float(target_idx) * float(source_dim) / float(target_dim)))
        end = int(round(float(target_idx + 1) * float(source_dim) / float(target_dim)))
        start = min(max(start, 0), source_dim - 1)
        end = min(max(end, start + 1), source_dim)
        bin_indices = order[start:end]
        if int(bin_indices.numel()) == 0:
            pos = int(round(float(target_idx) * float(source_dim - 1) / float(max(1, target_dim - 1))))
            bin_indices = order[pos : pos + 1]
        bin_rows = rows.index_select(0, bin_indices)
        bin_weights = weights.index_select(0, bin_indices)
        centroid = (bin_rows * bin_weights.unsqueeze(1)).sum(dim=0) / bin_weights.sum().clamp_min(1e-12)
        centroid = _row_normalize(centroid.view(1, -1))[0]
        similarity = bin_rows.matmul(centroid)
        score = similarity + 0.01 * (bin_weights / bin_weights.max().clamp_min(1e-12))
        for local_idx in torch.argsort(score, descending=True).tolist():
            source_idx = int(bin_indices[int(local_idx)].item())
            if source_idx not in selected_set:
                selected.append(source_idx)
                selected_set.add(source_idx)
                break
    if len(selected) < target_dim:
        for source_idx in _activation_quantile_support_indices(torch, feature_rows, weights, target_dim).tolist():
            source_idx = int(source_idx)
            if source_idx in selected_set:
                continue
            selected.append(source_idx)
            selected_set.add(source_idx)
            if len(selected) == target_dim:
                break
    if len(selected) < target_dim:
        for source_idx in torch.argsort(weights, descending=True).tolist():
            source_idx = int(source_idx)
            if source_idx in selected_set:
                continue
            selected.append(source_idx)
            selected_set.add(source_idx)
            if len(selected) == target_dim:
                break
    return torch.tensor(selected[:target_dim], dtype=torch.long).sort().values


def _activation_medoid_support_indices(torch, feature_rows, fold_weights, assignment, target_dim: int):
    source_dim = int(feature_rows.shape[0])
    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError("target_dim must be positive.")
    if source_dim <= target_dim:
        return torch.arange(source_dim, dtype=torch.long)
    rows = _row_normalize(feature_rows.detach().cpu().to(torch.float32))
    weights = fold_weights.detach().cpu().to(torch.float32).flatten().clamp_min(1e-12)
    groups = assignment.detach().cpu().to(torch.long).flatten()
    if int(weights.numel()) != source_dim:
        raise ValueError("Medoid support weights must match feature rows.")
    if int(groups.numel()) != source_dim:
        raise ValueError("Medoid support assignment must match feature rows.")
    selected: list[int] = []
    selected_set: set[int] = set()
    for target_idx in range(target_dim):
        mask = groups == int(target_idx)
        if not bool(mask.any()):
            continue
        cluster_indices = mask.nonzero(as_tuple=False).flatten()
        cluster_rows = rows.index_select(0, cluster_indices)
        cluster_weights = weights.index_select(0, cluster_indices)
        centroid = (cluster_rows * cluster_weights.unsqueeze(1)).sum(dim=0) / cluster_weights.sum().clamp_min(1e-12)
        centroid = _row_normalize(centroid.view(1, -1))[0]
        similarity = cluster_rows.matmul(centroid)
        # Small weight term breaks near ties toward high-signal donor axes without letting weights dominate geometry.
        score = similarity + 0.01 * (cluster_weights / cluster_weights.max().clamp_min(1e-12))
        for local_idx in torch.argsort(score, descending=True).tolist():
            source_idx = int(cluster_indices[int(local_idx)].item())
            if source_idx not in selected_set:
                selected.append(source_idx)
                selected_set.add(source_idx)
                break
    if len(selected) < target_dim:
        fallback_order = torch.argsort(weights, descending=True).tolist()
        for source_idx in fallback_order:
            source_idx = int(source_idx)
            if source_idx in selected_set:
                continue
            selected.append(source_idx)
            selected_set.add(source_idx)
            if len(selected) == target_dim:
                break
    return torch.tensor(selected[:target_dim], dtype=torch.long).sort().values


def _activation_transport_target_mass(torch, scores, fold_weights, target_dim: int):
    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError("target_dim must be positive.")
    weights = fold_weights.detach().cpu().to(torch.float32).flatten().clamp_min(1e-12)
    if scores.dim() != 2 or int(scores.shape[0]) != int(weights.numel()) or int(scores.shape[1]) != target_dim:
        raise ValueError("Transport scores must be source-by-target and match weights.")
    nearest = torch.argmax(scores.detach().cpu().to(torch.float32), dim=1)
    mass = torch.zeros(target_dim, dtype=torch.float32)
    mass.index_add_(0, nearest.to(torch.long), weights)
    floor = weights.sum().clamp_min(1e-12) / float(target_dim) * 1e-3
    mass = mass.clamp_min(floor)
    return mass / mass.sum().clamp_min(1e-12)


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
    transport_support: str = "centroid",
):
    if transport_backend not in {"sparse", "sinkhorn"}:
        raise ValueError("transport_backend must be sparse or sinkhorn.")
    if transport_support not in {"centroid", "importance", "quantile", "medoid", "quantile-medoid"}:
        raise ValueError("transport_support must be centroid, importance, quantile, medoid, or quantile-medoid.")
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
    raw_feature_rows = feature_rows
    feature_rows = _row_normalize(feature_rows)
    support_indices = None
    if transport_support == "importance":
        support_indices = torch.topk(fold_weights, k=target_dim, largest=True).indices.sort().values
        centroid_rows = feature_rows[support_indices]
    elif transport_support == "quantile":
        support_indices = _activation_quantile_support_indices(
            torch,
            raw_feature_rows,
            fold_weights,
            target_dim,
        )
        centroid_rows = feature_rows[support_indices]
    elif transport_support == "quantile-medoid":
        support_indices = _activation_quantile_medoid_support_indices(
            torch,
            raw_feature_rows,
            fold_weights,
            target_dim,
        )
        centroid_rows = feature_rows[support_indices]
    elif transport_support == "medoid":
        seed_assignment = _activation_similarity_assignment(torch, feature_rows, target_dim, fold_weights, iterations=4)
        support_indices = _activation_medoid_support_indices(
            torch,
            feature_rows,
            fold_weights,
            seed_assignment,
            target_dim,
        )
        centroid_rows = feature_rows[support_indices]
    else:
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
        centroid_rows = torch.stack(centroids, dim=0)
    centroid_rows = _row_normalize(centroid_rows)
    top_k = max(1, min(int(top_k), target_dim))
    scores = feature_rows.matmul(centroid_rows.t())
    if transport_support in {"importance", "quantile", "medoid", "quantile-medoid"}:
        target_mass = _activation_transport_target_mass(torch, scores, fold_weights, target_dim)
    else:
        target_mass = torch.full((target_dim,), 1.0 / float(target_dim), dtype=torch.float32)
    if transport_backend == "sinkhorn" and int(source_dim) * int(target_dim) <= int(dense_limit):
        source_mass = fold_weights / fold_weights.sum().clamp_min(1e-12)
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
            "transport_support": transport_support,
            "target_masses": target_mass.detach().cpu().to(torch.float32),
            "target_support_indices": None
            if support_indices is None
            else support_indices.detach().cpu().to(torch.long),
            "sinkhorn_iterations": int(balance_iterations),
        }
    top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)
    values = torch.softmax(top_scores / max(float(temperature), 1e-6), dim=1)
    for _ in range(max(0, int(balance_iterations))):
        loads = torch.zeros(target_dim, dtype=torch.float32)
        loads.index_add_(0, top_indices.reshape(-1), (values * fold_weights.unsqueeze(1)).reshape(-1))
        target_loads = target_mass * fold_weights.sum().clamp_min(1e-12)
        balance = (target_loads.clamp_min(1e-12) / loads.clamp_min(1e-12)).clamp(0.25, 4.0)
        values = values * balance[top_indices]
        values = values / values.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return {
        "transport_indices": top_indices.detach().cpu().to(torch.long),
        "transport_values": values.detach().cpu().to(torch.float32),
        "source_dim": int(source_dim),
        "target_dim": int(target_dim),
        "assignment_mode": "transport",
        "transport_support": transport_support,
        "target_masses": target_mass.detach().cpu().to(torch.float32),
        "target_support_indices": None if support_indices is None else support_indices.detach().cpu().to(torch.long),
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
    transport_support: str = "centroid",
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
    if transport_support not in {"centroid", "importance", "quantile", "medoid", "quantile-medoid"}:
        raise ValueError("transport_support must be centroid, importance, quantile, medoid, or quantile-medoid.")
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
            transport_backend=transport_backend,
            transport_support=transport_support,
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
                    transport_support=transport_support,
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
    activation_feature_mode: str = "stats",
    activation_subspace_rank: int = 8,
    activation_domain_balance: str = "none",
) -> dict[str, Any]:
    if not texts:
        raise ValueError("Need at least one activation calibration text.")
    if activation_feature_mode not in {"stats", "subspace", "covariance"}:
        raise ValueError("activation_feature_mode must be stats, subspace, or covariance.")
    if activation_subspace_rank < 0:
        raise ValueError("activation_subspace_rank must be non-negative.")
    if activation_domain_balance not in {"none", "keyword"}:
        raise ValueError("activation_domain_balance must be none or keyword.")
    text_domains = (
        [_activation_calibration_domain(text) for text in texts]
        if activation_domain_balance == "keyword"
        else None
    )
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
            hidden_features.append(
                _activation_channel_summary(
                    torch,
                    _activation_2d(torch, hidden),
                    mode=activation_feature_mode,
                    subspace_rank=activation_subspace_rank,
                ).cpu()
            )
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

    def mlp_neuron_hook(layer_idx: int, spec: dict[str, Any]):
        def hook(_module, inputs):
            values = _mlp_neuron_activation_values(torch, spec, inputs)
            if values is None:
                return
            values = values.detach().to(torch.float32)
            mlp_scores[layer_idx].append(_activation_abs_score(torch, values).cpu())
            mlp_features[layer_idx].append(
                _activation_channel_summary(
                    torch,
                    _activation_2d(torch, values),
                    mode=activation_feature_mode,
                    subspace_rank=activation_subspace_rank,
                ).cpu()
            )

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
        handles.append(spec["mlp"].register_forward_pre_hook(mlp_neuron_hook(layer_idx, spec)))
        for name, linear in zip(spec["input_names"], spec["inputs"]):
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
    hidden_domains = None
    if text_domains is not None and len(hidden_scores) == len(texts) * len(donor_layers):
        hidden_domains = [domain for domain in text_domains for _ in donor_layers]
    folded_mlp_scores = {}
    folded_mlp_features = {}
    for layer_idx, scores in mlp_scores.items():
        if scores:
            folded_mlp_scores[layer_idx] = _domain_balanced_mean(torch, scores, text_domains).clamp_min(1e-12)
    for layer_idx, features in mlp_features.items():
        if features:
            folded_mlp_features[layer_idx] = _domain_balanced_mean(torch, features, text_domains)
    folded_attention_profiles = {}
    for layer_idx, by_stream in attention_head_profiles.items():
        stream_profiles = {}
        for stream, values in by_stream.items():
            if values["scores"] and values["features"]:
                stream_profiles[stream] = {
                    "scores": _domain_balanced_mean(torch, values["scores"], text_domains).clamp_min(1e-12),
                    "features": _domain_balanced_mean(torch, values["features"], text_domains),
                }
        if stream_profiles:
            folded_attention_profiles[int(layer_idx)] = stream_profiles
    folded_attention_head_dim_profiles = {}
    for layer_idx, by_stream in attention_head_dim_profiles.items():
        stream_profiles = {}
        for stream, values in by_stream.items():
            if values["scores"] and values["features"]:
                stream_profiles[stream] = {
                    "scores": _domain_balanced_mean(torch, values["scores"], text_domains).clamp_min(1e-12),
                    "features": _domain_balanced_mean(torch, values["features"], text_domains),
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
        "hidden_scores": _domain_balanced_mean(torch, hidden_scores, hidden_domains).clamp_min(1e-12),
        "hidden_features": _domain_balanced_mean(torch, hidden_features, hidden_domains),
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
        "activation_feature_mode": activation_feature_mode,
        "activation_subspace_rank": int(activation_subspace_rank),
        "activation_domain_balance": activation_domain_balance,
        "activation_domain_counts": _domain_counts(text_domains),
    }


def _domain_balanced_mean(torch, tensors: Sequence[Any], domains: Sequence[str] | None = None):
    if not tensors:
        raise ValueError("Need at least one tensor to average.")
    if domains is None or len(domains) != len(tensors):
        return torch.stack(list(tensors), dim=0).mean(dim=0)
    grouped: dict[str, list[Any]] = {}
    order = []
    for domain, tensor in zip(domains, tensors):
        key = str(domain)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(tensor)
    domain_means = [torch.stack(grouped[key], dim=0).mean(dim=0) for key in order]
    return torch.stack(domain_means, dim=0).mean(dim=0)


def _domain_counts(domains: Sequence[str] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for domain in domains or []:
        key = str(domain)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _activation_calibration_domain(text: str) -> str:
    value = str(text).lower()
    if re.search(r"\b(def|python|function|return|syntax|code|lambda|indent)\b", value):
        return "code"
    if re.search(r"\b(tokeni[sz]er|vocab|embedding|logit|token id|token ids|byte)\b", value):
        return "tokenizer"
    if re.search(r"\b(prun\w*|merg\w*|fold\w*|neuron\w*|compress\w*|model surgery|stitch\w*)\b", value):
        return "model_surgery"
    if re.search(r"\d+\s*[\+\-\*/]\s*\d+", value) or re.search(
        r"\b(arithmetic|math|number|add|subtract|multiply|divide|sum|product)\b",
        value,
    ):
        return "math"
    if re.search(r"\b(driver|rain|thunder|lightning|safety|road|cautious)\b", value):
        return "safety"
    return "general"


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
    for name in ("self_attn", "attn", "attention", "operator"):
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


def _mlp_neuron_activation_values(torch, spec: dict[str, Any], inputs):
    if not inputs:
        return None
    module_inputs = inputs[0]
    if module_inputs is None:
        return None
    if spec.get("kind") == "gated" and len(spec.get("inputs", [])) >= 2:
        gate = _linear_module_forward(torch, spec["inputs"][0], module_inputs)
        up = _linear_module_forward(torch, spec["inputs"][1], module_inputs)
        return _mlp_activation_fn(torch, spec.get("mlp"))(gate) * up
    if spec.get("inputs"):
        projected = _linear_module_forward(torch, spec["inputs"][0], module_inputs)
        return _mlp_activation_fn(torch, spec.get("mlp"))(projected)
    return None


def _linear_module_forward(torch, module, inputs):
    weight = module.weight
    bias = getattr(module, "bias", None)
    values = inputs.to(device=weight.device, dtype=weight.dtype)
    return torch.nn.functional.linear(values, weight, bias)


def _mlp_activation_fn(torch, mlp):
    for attr in ("act_fn", "activation_fn", "act"):
        activation = getattr(mlp, attr, None)
        if callable(activation):
            return activation
        if isinstance(activation, str):
            return _named_activation_fn(torch, activation)
    return torch.nn.functional.silu


def _named_activation_fn(torch, name: str):
    lowered = str(name).lower()
    if lowered in {"silu", "swish"}:
        return torch.nn.functional.silu
    if lowered == "gelu":
        return torch.nn.functional.gelu
    if lowered == "relu":
        return torch.nn.functional.relu
    if lowered == "tanh":
        return torch.tanh
    return torch.nn.functional.silu


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
    mlp_axes = _mlp_intermediate_axes_for_tensor_suffix(suffix, rank=len(tuple(donor_tensor.shape)))
    if mlp_scores is not None and mlp_axes:
        for axis, source_dim in enumerate(donor_tensor.shape):
            if int(axis) in mlp_axes and int(source_dim) == int(mlp_scores.numel()):
                axis_weights[axis] = {
                    "weights": mlp_scores,
                    "features": mlp_features,
                    "assignment_key": ("mlp", layer_group_key),
                }
    return axis_weights or None


def _mlp_intermediate_axes_for_tensor_suffix(suffix: str, rank: int) -> set[int]:
    marker = f".{suffix.lower()}"
    if rank <= 0:
        return set()
    if marker.endswith(
        (
            ".gate_proj.bias",
            ".up_proj.bias",
            ".w1.bias",
            ".w3.bias",
            ".dense_h_to_4h.bias",
            ".fc1.bias",
            ".c_fc.bias",
        )
    ):
        return {0}
    if rank < 2:
        return set()
    if marker.endswith(
        (
            ".gate_proj.weight",
            ".up_proj.weight",
            ".w1.weight",
            ".w3.weight",
            ".dense_h_to_4h.weight",
            ".fc1.weight",
        )
    ):
        return {0}
    if marker.endswith((".down_proj.weight", ".w2.weight", ".dense_4h_to_h.weight", ".fc2.weight")):
        return {1}
    if ".mlp." in marker or ".feed_forward." in marker:
        if marker.endswith(".c_fc.weight"):
            return {1}
        if marker.endswith(".c_proj.weight"):
            return {0}
    return set()


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
    rank: int = 0,
    rank_mode: str = "absolute",
    method: str = "ridge",
    alpha: float = 1.0,
    factor_steps: int = 0,
    factor_lr: float = 3e-2,
    factor_l2: float = 0.0,
    hidden_basis=None,
    axis_weights: dict[int, Any] | None = None,
    activation_fold_assignment: str = "balanced",
    transport_backend: str = "sparse",
    transport_support: str = "centroid",
    assignment_cache: dict[Any, dict[str, Any]] | None = None,
    reference_weight=None,
    fit_gate: str = "validation",
    validation_fraction: float = 0.20,
    min_validation_rows: int = 8,
    min_relative_improvement: float = 0.0,
    max_cosine_drop: float = 0.01,
    return_report: bool = False,
):
    if fit_gate not in {"none", "validation"}:
        raise ValueError("fit_gate must be none or validation.")
    if not 0 <= float(validation_fraction) < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    if min_validation_rows < 0:
        raise ValueError("min_validation_rows must be non-negative.")
    if min_relative_improvement < 0:
        raise ValueError("min_relative_improvement must be non-negative.")
    if max_cosine_drop < 0:
        raise ValueError("max_cosine_drop must be non-negative.")
    if rank < 0:
        raise ValueError("rank must be non-negative.")
    if rank_mode not in {"absolute", "delta"}:
        raise ValueError("rank_mode must be absolute or delta.")
    if method not in {"ridge", "lrc"}:
        raise ValueError("method must be ridge or lrc.")
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be positive.")
    if int(factor_steps) < 0:
        raise ValueError("factor_steps must be non-negative.")
    if int(factor_steps) > 0 and float(factor_lr) <= 0.0:
        raise ValueError("factor_lr must be positive when factor_steps is positive.")
    if float(factor_l2) < 0.0:
        raise ValueError("factor_l2 must be non-negative.")

    def finish(weight, report: dict[str, Any]):
        report = {
            "fit_gate": fit_gate,
            "validation_fraction": float(validation_fraction),
            "min_validation_rows": int(min_validation_rows),
            "min_relative_improvement": float(min_relative_improvement),
            "max_cosine_drop": float(max_cosine_drop),
            "rank": int(rank),
            "rank_mode": rank_mode,
            "method": method,
            "alpha": float(alpha),
            "factor_steps": int(factor_steps),
            "factor_lr": float(factor_lr),
            "factor_l2": float(factor_l2),
            **report,
        }
        if return_report:
            return {"weight": weight, "report": report}
        return weight

    def reject(reject_reason: str, **extra):
        return finish(None, {**extra, "accepted": False, "reason": reject_reason})

    rows = _linear_fit_rows_for_suffix(torch, activation_profile, donor_layer_group or [donor_layer_idx], suffix)
    if rows is None:
        return reject("missing_activation_rows")
    source_input = rows["input"]
    source_output = rows["output"]
    if source_input.dim() != 2 or source_output.dim() != 2:
        return reject(
            "invalid_activation_rank",
            source_input_rank=int(source_input.dim()),
            source_output_rank=int(source_output.dim()),
        )
    if int(source_input.shape[0]) < 2 or int(source_output.shape[0]) < 2:
        return reject(
            "too_few_source_rows",
            source_input_rows=int(source_input.shape[0]),
            source_output_rows=int(source_output.shape[0]),
        )
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
        transport_support=transport_support,
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
        transport_support=transport_support,
        assignment_cache=assignment_cache,
    )
    if folded_input is None or folded_output is None:
        return reject("activation_row_resize_failed")
    row_count = min(int(folded_input.shape[0]), int(folded_output.shape[0]))
    if row_count < 2:
        return reject("too_few_folded_rows", row_count=int(row_count))
    folded_input = folded_input[:row_count]
    folded_output = folded_output[:row_count]
    if method == "lrc":
        if reference_weight is None:
            return reject("missing_reference_weight_for_lrc", row_count=int(row_count))
        from neural_forge.lrc import fit_materialized_linear_lrc_delta

        reference = reference_weight.detach().cpu().to(torch.float32)
        lrc_rank = int(rank) if int(rank) > 0 else min(16, int(reference.shape[0]), int(reference.shape[1]))
        fitted, lrc_report = fit_materialized_linear_lrc_delta(
            reference,
            folded_input,
            folded_output,
            rank=lrc_rank,
            alpha=float(alpha),
            ridge=float(ridge),
            factor_steps=int(factor_steps),
            factor_lr=float(factor_lr),
            factor_l2=float(factor_l2),
            validation_fraction=float(validation_fraction),
            min_validation_rows=int(min_validation_rows),
            min_relative_improvement=float(min_relative_improvement),
            max_cosine_drop=float(max_cosine_drop),
            require_validation=fit_gate == "validation",
            max_rows=None,
        )
        report = {
            **lrc_report,
            "accepted": bool(lrc_report.get("accepted")),
            "reason": str(lrc_report.get("reason", "unknown")),
            "row_count": int(row_count),
            "target_shape": [int(target_out), int(target_in)],
            "requested_rank": int(rank),
            "effective_lrc_rank": int(lrc_rank),
            "rank_defaulted": int(rank) <= 0,
        }
        return finish(fitted if bool(lrc_report.get("accepted")) else None, report)
    validation_rows = int(max(0, min(int(row_count * float(validation_fraction)), row_count - 2)))
    if validation_rows > 0 and validation_rows >= int(min_validation_rows) and reference_weight is not None:
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
        if rank_mode == "delta":
            fitted = _scale_projection_to_target(torch, fitted, reference, "target-rms")
            fitted, rank_report = _truncate_linear_weight_delta_rank(torch, fitted, reference, int(rank))
        else:
            fitted, rank_report = _truncate_linear_weight_rank(torch, fitted, int(rank))
            fitted = _scale_projection_to_target(torch, fitted, reference, "target-rms")
    else:
        fitted, rank_report = _truncate_linear_weight_rank(torch, fitted, int(rank))
    report = {
        "accepted": True,
        "reason": "accepted",
        "row_count": int(row_count),
        "train_rows": int(train_input.shape[0]),
        "validation_rows": int(0 if val_input is None else val_input.shape[0]),
        "target_shape": [int(target_out), int(target_in)],
        **rank_report,
    }
    if val_input is not None and val_output is not None and reference_weight is not None:
        reference = reference_weight.detach().cpu().to(torch.float32)
        fit_prediction = val_input.matmul(fitted.t())
        reference_prediction = val_input.matmul(reference.t())
        fit_error = fit_prediction.sub(val_output).pow(2).mean()
        reference_error = reference_prediction.sub(val_output).pow(2).mean()
        fit_cosine = torch.nn.functional.cosine_similarity(fit_prediction, val_output, dim=1).mean()
        reference_cosine = torch.nn.functional.cosine_similarity(reference_prediction, val_output, dim=1).mean()
        relative_improvement = (
            float(reference_error) - float(fit_error)
        ) / max(abs(float(reference_error)), 1e-12)
        cosine_delta = float(fit_cosine) - float(reference_cosine)
        report.update(
            {
                "validation_reference_mse": float(reference_error),
                "validation_fit_mse": float(fit_error),
                "validation_relative_improvement": float(relative_improvement),
                "validation_reference_cosine": float(reference_cosine),
                "validation_fit_cosine": float(fit_cosine),
                "validation_cosine_delta": float(cosine_delta),
            }
        )
        if fit_gate == "validation":
            if not bool(torch.isfinite(fit_error)) or not bool(torch.isfinite(reference_error)):
                return reject("nonfinite_validation_mse", **report)
            if float(fit_error) >= float(reference_error):
                return reject("validation_mse_not_improved", **report)
            if relative_improvement < float(min_relative_improvement):
                return reject("validation_improvement_below_threshold", **report)
            if bool(torch.isfinite(fit_cosine)) and bool(torch.isfinite(reference_cosine)):
                if float(fit_cosine) + float(max_cosine_drop) < float(reference_cosine):
                    return reject("validation_cosine_drop", **report)
    elif fit_gate == "validation" and reference_weight is not None:
        report["validation_gate_skipped"] = True
        return reject("insufficient_validation_rows", **report)
    elif fit_gate == "validation":
        report["reason"] = "accepted_without_validation_split"
        report["validation_gate_skipped"] = True
    return finish(fitted, report)


def _linear_fit_rows_for_suffix(torch, activation_profile, layer_group: Sequence[int], suffix: str):
    if activation_profile is None:
        return None
    rows_by_layer = activation_profile.get("linear_fit_rows") or {}
    suffix_candidates = _layer_suffix_candidates(suffix)
    input_rows = []
    output_rows = []
    for layer_idx in layer_group:
        layer_rows = rows_by_layer.get(int(layer_idx), {})
        values = None
        for candidate in suffix_candidates:
            values = layer_rows.get(str(candidate))
            if values is not None:
                break
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
    transport_support: str = "centroid",
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
            transport_support=transport_support,
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
        transport_support=transport_support,
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
        or marker.endswith(".operator.q_proj.weight")
        or marker.endswith(".operator.k_proj.weight")
        or marker.endswith(".operator.v_proj.weight")
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
        or marker.endswith(".operator.out_proj.weight")
        or marker.endswith(".operator.o_proj.weight")
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


def _truncate_linear_weight_rank(torch, weight, rank: int):
    target = torch.nan_to_num(weight.detach().cpu().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if target.dim() != 2:
        return target, {
            "rank_truncated": False,
            "effective_rank": 0,
            "requested_rank": int(rank),
        }
    full_rank = min(int(target.shape[0]), int(target.shape[1]))
    if int(rank) <= 0 or int(rank) >= full_rank:
        return target, {
            "rank_truncated": False,
            "effective_rank": int(full_rank),
            "requested_rank": int(rank),
        }
    try:
        u, singular_values, vh = torch.linalg.svd(target, full_matrices=False)
    except RuntimeError:
        return target, {
            "rank_truncated": False,
            "effective_rank": int(full_rank),
            "requested_rank": int(rank),
            "rank_truncation_error": "svd_failed",
        }
    keep = max(1, int(rank))
    truncated = (u[:, :keep] * singular_values[:keep].unsqueeze(0)).matmul(vh[:keep, :])
    total_energy = singular_values.pow(2).sum().clamp_min(1e-12)
    kept_energy = singular_values[:keep].pow(2).sum()
    return truncated.contiguous(), {
        "rank_truncated": True,
        "effective_rank": int(keep),
        "requested_rank": int(rank),
        "full_rank": int(full_rank),
        "rank_energy_ratio": float(kept_energy / total_energy),
    }


def _truncate_linear_weight_delta_rank(torch, weight, reference_weight, rank: int):
    target = torch.nan_to_num(weight.detach().cpu().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    reference = torch.nan_to_num(
        reference_weight.detach().cpu().to(torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if target.shape != reference.shape:
        raise ValueError("weight and reference_weight must have the same shape.")
    delta = target - reference
    truncated_delta, report = _truncate_linear_weight_rank(torch, delta, int(rank))
    if int(rank) <= 0:
        return target, {
            **report,
            "rank_mode": "delta",
            "delta_rank_truncated": False,
            "delta_norm": float(delta.norm().detach().cpu()),
            "truncated_delta_norm": float(delta.norm().detach().cpu()),
        }
    return (reference + truncated_delta).contiguous(), {
        **report,
        "rank_mode": "delta",
        "delta_rank_truncated": bool(report.get("rank_truncated", False)),
        "delta_norm": float(delta.norm().detach().cpu()),
        "truncated_delta_norm": float(truncated_delta.norm().detach().cpu()),
    }


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
    transport_support: str = "centroid",
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
    special_target_ids: list[int] = []
    special_donor_groups: list[list[int]] = []
    special_group_weights: list[list[float]] = []
    if token_init_mode != "shared":
        special_target_ids, special_donor_groups, special_group_weights = _target_to_donor_special_token_group_specs(
            target_tokenizer,
            donor_tokenizer,
            seen_target_ids=set(target_ids),
        )
    if special_target_ids:
        target_ids.extend(special_target_ids)
        donor_id_groups.extend(special_donor_groups)
        donor_group_weights.extend(special_group_weights)

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
        transport_support=transport_support,
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
                transport_support=transport_support,
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
        "special_rows_initialized": int(len(special_target_ids)),
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
    transport_support: str = "centroid",
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
            transport_support=transport_support,
            assignment_cache=assignment_cache,
        )
        projected = _scale_projection_to_target(
            torch,
            projected,
            target_weight[torch.tensor(chunk_target_ids, dtype=torch.long)],
            projection_scale,
            source_tensor=donor_rows,
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


def _target_to_donor_special_token_group_specs(
    target_tokenizer,
    donor_tokenizer,
    seen_target_ids: set[int] | None = None,
) -> tuple[list[int], list[list[int]], list[list[float]]]:
    seen = {int(item) for item in (seen_target_ids or set())}
    target_roles = _special_token_ids_by_role(target_tokenizer)
    donor_roles = _special_token_ids_by_role(donor_tokenizer)
    donor_decoded = _special_token_ids_by_decoded_surface(donor_tokenizer)
    donor_vocab_size = len(donor_tokenizer)
    target_ids: list[int] = []
    donor_id_groups: list[list[int]] = []
    donor_group_weights: list[list[float]] = []
    for role, role_target_ids in target_roles.items():
        for target_id in role_target_ids:
            target_id = int(target_id)
            if target_id in seen:
                continue
            donor_ids = _donor_special_ids_for_target_role(
                role,
                target_id,
                target_tokenizer,
                donor_roles,
                donor_decoded,
            )
            donor_ids = [int(item) for item in donor_ids if 0 <= int(item) < donor_vocab_size]
            if not donor_ids:
                continue
            target_ids.append(target_id)
            donor_id_groups.append(donor_ids)
            donor_group_weights.append([1.0] * len(donor_ids))
            seen.add(target_id)
    return target_ids, donor_id_groups, donor_group_weights


def _special_token_ids_by_role(tokenizer) -> dict[str, list[int]]:
    roles: dict[str, list[int]] = {}

    def add(role: str, value) -> None:
        if value is None:
            return
        values = value if isinstance(value, (list, tuple, set)) else [value]
        for item in values:
            try:
                token_id = int(item)
            except (TypeError, ValueError):
                continue
            if token_id < 0:
                continue
            bucket = roles.setdefault(role, [])
            if token_id not in bucket:
                bucket.append(token_id)

    for role in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
    ):
        add(role, getattr(tokenizer, role, None))
    for token_id in getattr(tokenizer, "additional_special_tokens_ids", []) or []:
        add("additional_special_tokens_ids", token_id)
    for token_id in getattr(tokenizer, "all_special_ids", []) or []:
        add("all_special_ids", token_id)
    return roles


def _special_token_ids_by_decoded_surface(tokenizer) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    for token_id in getattr(tokenizer, "all_special_ids", []) or []:
        try:
            token_id = int(token_id)
        except (TypeError, ValueError):
            continue
        decoded = _decode_single_token(tokenizer, token_id)
        candidates = [decoded, decoded.strip(), _canonical_token(decoded)]
        for candidate in candidates:
            if not candidate:
                continue
            bucket = mapping.setdefault(candidate, [])
            if token_id not in bucket:
                bucket.append(token_id)
    return mapping


def _donor_special_ids_for_target_role(
    role: str,
    target_id: int,
    target_tokenizer,
    donor_roles: dict[str, list[int]],
    donor_decoded: dict[str, list[int]],
) -> list[int]:
    role_preferences = {
        "bos_token_id": ("bos_token_id", "eos_token_id", "all_special_ids"),
        "eos_token_id": ("eos_token_id", "bos_token_id", "all_special_ids"),
        "pad_token_id": ("pad_token_id", "eos_token_id", "all_special_ids"),
        "unk_token_id": ("unk_token_id", "all_special_ids"),
        "sep_token_id": ("sep_token_id", "eos_token_id", "all_special_ids"),
        "cls_token_id": ("cls_token_id", "bos_token_id", "all_special_ids"),
        "mask_token_id": ("mask_token_id", "unk_token_id", "all_special_ids"),
        "additional_special_tokens_ids": ("additional_special_tokens_ids", "all_special_ids"),
        "all_special_ids": ("all_special_ids",),
    }
    donor_ids: list[int] = []

    def add(items: Sequence[int] | None) -> None:
        if not items:
            return
        for item in items:
            item = int(item)
            if item not in donor_ids:
                donor_ids.append(item)

    decoded = _decode_single_token(target_tokenizer, int(target_id))
    if role in {"additional_special_tokens_ids", "all_special_ids"}:
        for candidate in (decoded, decoded.strip(), _canonical_token(decoded)):
            add(donor_decoded.get(candidate))
            if donor_ids:
                return donor_ids
    for preferred_role in role_preferences.get(role, (role, "all_special_ids")):
        add(donor_roles.get(preferred_role))
        if donor_ids:
            return donor_ids
    for candidate in (decoded, decoded.strip(), _canonical_token(decoded)):
        add(donor_decoded.get(candidate))
        if donor_ids:
            return donor_ids
    return donor_ids


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


def _activation_channel_summary(torch, values, mode: str = "stats", subspace_rank: int = 8):
    if mode not in {"stats", "subspace", "covariance"}:
        raise ValueError("mode must be stats, subspace, or covariance.")
    if subspace_rank < 0:
        raise ValueError("subspace_rank must be non-negative.")
    values = values.to(torch.float32)
    mean = values.mean(dim=0)
    centered = values - mean.unsqueeze(0)
    std = centered.pow(2).mean(dim=0).sqrt()
    abs_mean = values.abs().mean(dim=0)
    positive_fraction = (values > 0).to(torch.float32).mean(dim=0)
    stats = torch.stack([mean, std, abs_mean, positive_fraction], dim=1)
    if mode == "stats" or int(subspace_rank) == 0:
        return stats
    if mode == "covariance":
        return torch.cat(
            [stats, _activation_channel_covariance_signature(torch, centered, std, int(subspace_rank))],
            dim=1,
        )
    return torch.cat([stats, _activation_channel_subspace_signature(torch, centered, int(subspace_rank))], dim=1)


def _activation_channel_subspace_signature(torch, centered, rank: int):
    channels = int(centered.shape[1])
    signature = torch.zeros((channels, int(rank)), dtype=torch.float32, device=centered.device)
    if rank <= 0 or channels <= 0 or int(centered.shape[0]) < 2:
        return signature
    row_rms = centered.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    normalized = centered / row_rms
    try:
        _u, singular_values, vh = torch.linalg.svd(normalized, full_matrices=False)
    except RuntimeError:
        return signature
    keep = min(int(rank), int(vh.shape[0]), channels)
    if keep <= 0:
        return signature
    components = vh[:keep].t().contiguous()
    # SVD signs are arbitrary. Canonicalize each component by its largest entry
    # so repeated calibration batches aggregate consistently.
    for idx in range(keep):
        column = components[:, idx]
        pivot = int(torch.argmax(column.abs()).item())
        if float(column[pivot].item()) < 0.0:
            components[:, idx] = -column
    scale = singular_values[:keep] / singular_values[:keep].max().clamp_min(1e-12)
    signature[:, :keep] = components * scale.unsqueeze(0)
    return signature


def _activation_channel_covariance_signature(torch, centered, std, rank: int):
    channels = int(centered.shape[1])
    signature = torch.zeros((channels, int(rank)), dtype=torch.float32, device=centered.device)
    if rank <= 0 or channels <= 0 or int(centered.shape[0]) < 2:
        return signature
    standardized = centered / std.unsqueeze(0).clamp_min(1e-6)
    standardized = torch.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        _u, singular_values, vh = torch.linalg.svd(standardized, full_matrices=False)
    except RuntimeError:
        return signature
    keep = min(int(rank), int(vh.shape[0]), channels)
    if keep <= 0:
        return signature
    components = vh[:keep].t().contiguous()
    for idx in range(keep):
        column = components[:, idx]
        pivot = int(torch.argmax(column.abs()).item())
        if float(column[pivot].item()) < 0.0:
            components[:, idx] = -column
    scale = singular_values[:keep] / singular_values[:keep].max().clamp_min(1e-12)
    signature[:, :keep] = components * scale.unsqueeze(0)
    return signature


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


def _relative_reconstruction_error(torch, predicted, target) -> float:
    error = predicted.detach().to(torch.float32).sub(target.detach().to(torch.float32)).pow(2).mean().sqrt()
    scale = target.detach().to(torch.float32).pow(2).mean().sqrt().clamp_min(1e-12)
    return float((error / scale).detach().cpu())


def _mean_row_cosine(torch, predicted, target) -> float:
    predicted = predicted.detach().to(torch.float32)
    target = target.detach().to(torch.float32)
    numerator = (predicted * target).sum(dim=1)
    denominator = predicted.pow(2).sum(dim=1).sqrt().mul(target.pow(2).sum(dim=1).sqrt()).clamp_min(1e-12)
    return float((numerator / denominator).mean().detach().cpu())


def _sanitize_generation_config_for_save(model, tokenizer=None) -> dict[str, Any]:
    generation_config = getattr(model, "generation_config", None)
    model_config = getattr(model, "config", None)
    vocab_size = None
    try:
        embedding = model.get_input_embeddings()
        weight = getattr(embedding, "weight", None)
        shape = getattr(weight, "shape", None)
        if shape:
            vocab_size = int(shape[0])
    except Exception:
        vocab_size = None

    def valid_token_id(value: Any) -> bool:
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        if value < 0:
            return False
        return vocab_size is None or value < vocab_size

    def iter_candidate_ids(value: Any):
        if isinstance(value, (list, tuple)):
            for item in value:
                yield item
        else:
            yield value

    candidate_ids: list[int] = []
    for owner in (tokenizer, model_config, generation_config):
        if owner is None:
            continue
        for attr in ("pad_token_id", "eos_token_id", "bos_token_id", "decoder_start_token_id"):
            for candidate in iter_candidate_ids(getattr(owner, attr, None)):
                if valid_token_id(candidate):
                    candidate_ids.append(int(candidate))
    fallback = next((item for item in candidate_ids if item > 0), None)
    if fallback is None:
        fallback = candidate_ids[0] if candidate_ids else 0

    changes: list[dict[str, Any]] = []

    def sanitize_owner(owner, owner_name: str) -> None:
        if owner is None:
            return
        for attr in ("pad_token_id", "bos_token_id", "decoder_start_token_id"):
            if not hasattr(owner, attr):
                continue
            before = getattr(owner, attr, None)
            if before is not None and not valid_token_id(before):
                setattr(owner, attr, int(fallback))
                changes.append({"owner": owner_name, "field": attr, "before": before, "after": int(fallback)})
        if hasattr(owner, "eos_token_id"):
            before = getattr(owner, "eos_token_id", None)
            if isinstance(before, (list, tuple)):
                after = [int(item) for item in before if valid_token_id(item)]
                if len(after) != len(before):
                    if not after:
                        after = [int(fallback)]
                    setattr(owner, "eos_token_id", after)
                    changes.append({"owner": owner_name, "field": "eos_token_id", "before": list(before), "after": after})
            elif before is not None and not valid_token_id(before):
                setattr(owner, "eos_token_id", int(fallback))
                changes.append({"owner": owner_name, "field": "eos_token_id", "before": before, "after": int(fallback)})

    sanitize_owner(model_config, "config")
    sanitize_owner(generation_config, "generation_config")
    return {
        "changed": bool(changes),
        "vocab_size": vocab_size,
        "fallback_token_id": int(fallback),
        "changes": changes,
    }


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
