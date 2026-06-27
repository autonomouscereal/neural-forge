"""Lightweight recovery training for Hugging Face causal LM surgery outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from neural_forge.hf_tools import (
    TEXT_EVALS,
    _decoder_layer_container,
    _dtype_from_name,
    _ridge_solve_features,
    _torch_and_transformers,
)


RECOVERY_TEXTS = [
    "A compressed assistant should keep short factual answers stable after its internal layers are edited.",
    "Model surgery changes the geometry of hidden states, so recovery needs held-out behavior checks.",
    "A useful pruning run keeps arithmetic, code syntax, and common-sense answers readable.",
    "Python code blocks usually contain a function name, parameters, indentation, and a return statement.",
    "A cautious merge workflow searches several small coefficients and rejects repetitive decoding.",
    "Calibration text should teach the repaired model local language structure without copying eval prompts.",
    "When a checkpoint is folded too hard, perplexity and generation samples can reveal the damage.",
    "Short recovery training can reduce surgery shock, but it must be validated on unseen prompts.",
]


def recover_causal_lm(
    source_model: str | None,
    candidate_model: str,
    output_dir: str | Path,
    steps: int = 80,
    batch_size: int = 1,
    max_length: int = 96,
    lr: float = 5e-5,
    kd_weight: float = 0.5,
    lm_weight: float = 0.5,
    temperature: float = 2.0,
    train_scope: str = "all",
    torch_dtype: str = "float32",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    calibration_texts: Sequence[str] | None = None,
    grad_clip: float = 1.0,
) -> dict[str, Any]:
    """Run a short recovery pass after surgery.

    When `kd_weight` is positive, `source_model` and `candidate_model` must use
    compatible tokenizers because KL is computed over the same vocabulary. When
    `kd_weight` is zero, recovery is teacherless LM repair on the candidate
    tokenizer, which is useful after cross-family/tokenizer folding.
    """

    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1.")
    if lr <= 0:
        raise ValueError("lr must be positive.")
    if kd_weight < 0 or lm_weight < 0:
        raise ValueError("loss weights must be non-negative.")
    if kd_weight == 0 and lm_weight == 0:
        raise ValueError("At least one loss weight must be positive.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if grad_clip < 0:
        raise ValueError("grad_clip must be non-negative.")

    if kd_weight > 0 and not source_model:
        raise ValueError("source_model is required when kd_weight is positive.")

    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    candidate_tokenizer = transformers.AutoTokenizer.from_pretrained(candidate_model, trust_remote_code=trust_remote_code)
    tokenizer = candidate_tokenizer
    source = None
    if source_model:
        source_tokenizer = transformers.AutoTokenizer.from_pretrained(source_model, trust_remote_code=trust_remote_code)
        if kd_weight > 0:
            _assert_same_vocab(source_tokenizer, candidate_tokenizer)
        tokenizer = candidate_tokenizer
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if source_model:
        source = transformers.AutoModelForCausalLM.from_pretrained(
            source_model,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=trust_remote_code,
        )
    candidate = transformers.AutoModelForCausalLM.from_pretrained(
        candidate_model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    if source is not None:
        source.to(device)
        source.eval()
    candidate.to(device)
    candidate.train()
    if hasattr(candidate.config, "use_cache"):
        candidate.config.use_cache = False

    trainable_params = _set_train_scope(candidate, train_scope)
    optimizer = torch.optim.AdamW([p for p in candidate.parameters() if p.requires_grad], lr=lr)
    texts = list(calibration_texts or RECOVERY_TEXTS)
    if not texts:
        raise ValueError("Need at least one recovery text.")
    eval_overlap_count = len(set(texts) & set(TEXT_EVALS))
    batches = _make_batches(tokenizer, texts, batch_size=batch_size, max_length=max_length, device=device)

    losses = []
    for step in range(1, steps + 1):
        batch = batches[(step - 1) % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        teacher = None
        if source is not None:
            with torch.no_grad():
                teacher = source(**batch, use_cache=False)
        student = candidate(**batch, use_cache=False)
        loss = _recovery_loss(
            torch,
            student,
            teacher,
            batch["input_ids"],
            batch.get("attention_mask"),
            kd_weight,
            lm_weight,
            temperature,
        )
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise RuntimeError(f"Non-finite recovery loss at step {step}; aborting before save.")
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in candidate.parameters() if p.requires_grad], grad_clip)
        optimizer.step()
        if step == 1 or step % max(1, steps // 10) == 0:
            losses.append({"step": step, "loss": float(loss.detach().cpu())})

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    candidate.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    report = {
        "operation": "recover_causal_lm",
        "source_model": source_model,
        "candidate_model": candidate_model,
        "output_dir": str(target),
        "steps": steps,
        "batch_size": batch_size,
        "max_length": max_length,
        "lr": lr,
        "kd_weight": kd_weight,
        "lm_weight": lm_weight,
        "temperature": temperature,
        "teacher_used": source_model is not None and kd_weight > 0,
        "train_scope": train_scope,
        "trainable_param_count": trainable_params,
        "calibration_text_count": len(texts),
        "eval_overlap_count": eval_overlap_count,
        "losses": losses,
        "param_count": sum(param.numel() for param in candidate.parameters()),
    }
    _write_json(target / "neural_forge_recovery_report.json", report)
    return report


def reconstruct_folded_causal_lm(
    source_model: str,
    candidate_model: str,
    output_dir: str | Path,
    layer_merge_report: str | Path | dict[str, Any] | None = None,
    ridge: float = 1e-3,
    reconstruction_alpha: float = 1.0,
    reconstruction_target: str = "consumer-input",
    reconstruction_objective: str = "ridge",
    cosine_steps: int = 80,
    cosine_lr: float = 1e-2,
    cosine_l2: float = 1e-4,
    max_length: int = 128,
    max_tokens_per_layer: int = 512,
    torch_dtype: str = "float16",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    calibration_texts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Apply GRAIL-style linear reconstruction after residual layer folding.

    This is a no-backprop compensation pass for same-tokenizer depth folds. For
    each folded layer group, it fits calibration maps from the folded candidate's
    downstream layernorm activations to the source model's corresponding
    activations, then absorbs those maps into the downstream attention and MLP
    input projections.
    """

    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if not 0.0 <= float(reconstruction_alpha) <= 1.0:
        raise ValueError("reconstruction_alpha must be in [0, 1].")
    if reconstruction_target not in {"consumer-input", "survivor-mlp", "both"}:
        raise ValueError("reconstruction_target must be consumer-input, survivor-mlp, or both.")
    if reconstruction_objective not in {"ridge", "cosine"}:
        raise ValueError("reconstruction_objective must be ridge or cosine.")
    if cosine_steps < 0:
        raise ValueError("cosine_steps must be non-negative.")
    if cosine_lr <= 0:
        raise ValueError("cosine_lr must be positive.")
    if cosine_l2 < 0:
        raise ValueError("cosine_l2 must be non-negative.")
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1.")
    if max_tokens_per_layer <= 0:
        raise ValueError("max_tokens_per_layer must be positive.")

    report = _load_layer_merge_report(candidate_model, layer_merge_report)
    fold_groups = _fold_groups_from_report(report)
    if not fold_groups:
        raise ValueError("Fold reconstruction requires at least one fold-method layer group.")

    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source_model, trust_remote_code=trust_remote_code)
    candidate_tokenizer = transformers.AutoTokenizer.from_pretrained(candidate_model, trust_remote_code=trust_remote_code)
    _assert_same_vocab(tokenizer, candidate_tokenizer)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    source = transformers.AutoModelForCausalLM.from_pretrained(
        source_model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    candidate = transformers.AutoModelForCausalLM.from_pretrained(
        candidate_model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    source.to(device)
    candidate.to(device)
    source.eval()
    candidate.eval()
    if hasattr(source.config, "use_cache"):
        source.config.use_cache = False
    if hasattr(candidate.config, "use_cache"):
        candidate.config.use_cache = False

    source_layers = _recovery_decoder_layers(source)
    candidate_layers = _recovery_decoder_layers(candidate)
    if not source_layers or not candidate_layers:
        raise ValueError("Both source and candidate need decoder layers.")

    texts = list(calibration_texts or RECOVERY_TEXTS)
    if not texts:
        raise ValueError("Need at least one reconstruction text.")
    eval_overlap_count = len(set(texts) & set(TEXT_EVALS))
    batches = _make_batches(tokenizer, texts, batch_size=1, max_length=max_length, device=device)

    repairs = []
    with torch.no_grad():
        for group in fold_groups:
            repair = _reconstruct_fold_group(
                torch,
                source,
                candidate,
                source_layers,
                candidate_layers,
                group,
                batches,
                ridge=ridge,
                reconstruction_alpha=float(reconstruction_alpha),
                reconstruction_target=reconstruction_target,
                reconstruction_objective=reconstruction_objective,
                cosine_steps=cosine_steps,
                cosine_lr=cosine_lr,
                cosine_l2=cosine_l2,
                max_tokens_per_layer=max_tokens_per_layer,
            )
            repairs.append(repair)
            if device.startswith("cuda") and hasattr(torch, "cuda"):
                torch.cuda.empty_cache()

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    candidate.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    reconstruction_report = {
        "operation": "reconstruct_folded_causal_lm",
        "source_model": source_model,
        "candidate_model": candidate_model,
        "output_dir": str(target),
        "layer_merge_report": report,
        "ridge": ridge,
        "reconstruction_alpha": float(reconstruction_alpha),
        "reconstruction_target": reconstruction_target,
        "reconstruction_objective": reconstruction_objective,
        "cosine_steps": int(cosine_steps),
        "cosine_lr": float(cosine_lr),
        "cosine_l2": float(cosine_l2),
        "max_length": max_length,
        "max_tokens_per_layer": max_tokens_per_layer,
        "torch_dtype": torch_dtype,
        "calibration_text_count": len(texts),
        "eval_overlap_count": eval_overlap_count,
        "repairs": repairs,
        "param_count": sum(param.numel() for param in candidate.parameters()),
    }
    _write_json(target / "neural_forge_fold_reconstruction_report.json", reconstruction_report)
    return reconstruction_report


def load_calibration_texts(
    calibration_file: str | Path,
    max_texts: int | None = None,
    min_chars: int = 32,
    max_chars: int = 4096,
) -> list[str]:
    """Load recovery calibration texts from plain text or JSONL rows."""

    if max_texts is not None and max_texts <= 0:
        raise ValueError("max_texts must be positive when provided.")
    if min_chars < 0:
        raise ValueError("min_chars must be non-negative.")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    path = Path(calibration_file)
    if not path.exists():
        raise FileNotFoundError(path)

    texts = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = _calibration_text_from_line(line)
            if text is None:
                continue
            text = " ".join(text.split())
            if len(text) < min_chars:
                continue
            texts.append(text[:max_chars])
            if max_texts is not None and len(texts) >= max_texts:
                break
    if not texts:
        raise ValueError(f"No usable calibration texts found in {path}.")
    return texts


def _calibration_text_from_line(line: str) -> str | None:
    raw = line.strip()
    if not raw:
        return None
    if not raw.startswith("{"):
        return raw
    try:
        item = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return _calibration_text_from_obj(item)


def _calibration_text_from_obj(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return None
    messages = item.get("messages")
    if isinstance(messages, list):
        return _messages_to_text(messages)
    input_json = item.get("input_json")
    if isinstance(input_json, dict) and isinstance(input_json.get("messages"), list):
        pieces = [_messages_to_text(input_json["messages"])]
        target = item.get("target")
        if isinstance(target, str) and target.strip():
            pieces.append(target)
        target_json = item.get("target_json")
        if isinstance(target_json, dict):
            pieces.append(json.dumps(target_json, sort_keys=True))
        return "\n".join(piece for piece in pieces if piece)
    prompt = item.get("prompt")
    target = item.get("target")
    if isinstance(prompt, str) and isinstance(target, str):
        return f"{prompt}\n{target}"
    for key in ("text", "content", "completion", "response"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _messages_to_text(messages: Sequence[Any]) -> str:
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        role = message.get("role")
        if isinstance(role, str) and role.strip():
            parts.append(f"{role.strip()}: {content.strip()}")
        else:
            parts.append(content.strip())
    return "\n".join(parts)


def _assert_same_vocab(left, right) -> None:
    if len(left) != len(right):
        raise ValueError(f"Tokenizer length mismatch: {len(left)} != {len(right)}")
    if hasattr(left, "get_vocab") and hasattr(right, "get_vocab") and left.get_vocab() != right.get_vocab():
        raise ValueError("Recovery KD requires matching tokenizers/vocabularies.")


def _set_train_scope(model, train_scope: str) -> int:
    scope = train_scope.lower()
    for param in model.parameters():
        param.requires_grad_(False)
    if scope == "all":
        for param in model.parameters():
            param.requires_grad_(True)
    elif scope == "ffn":
        for layer in _recovery_decoder_layers(model):
            for module_name in ("mlp", "feed_forward"):
                module = getattr(layer, module_name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad_(True)
    elif scope.startswith("last"):
        count_text = scope.replace("last", "").strip("-_") or "1"
        count = int(count_text)
        layers = _recovery_decoder_layers(model)
        if count <= 0 or not layers:
            raise ValueError("last-N train scope requires N > 0 and decoder layers.")
        for layer in layers[-count:]:
            for param in layer.parameters():
                param.requires_grad_(True)
        output = model.get_output_embeddings()
        if output is not None:
            for param in output.parameters():
                param.requires_grad_(True)
    else:
        raise ValueError("train_scope must be all, ffn, or lastN, e.g. last1.")
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _recovery_decoder_layers(model) -> list[Any]:
    try:
        return list(_decoder_layer_container(model)[2])
    except ValueError:
        return []


def _make_batches(tokenizer, texts: Sequence[str], batch_size: int, max_length: int, device: str) -> list[dict[str, Any]]:
    batches = []
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        batches.append({key: value.to(device) for key, value in encoded.items()})
    return batches


def _load_layer_merge_report(candidate_model: str, layer_merge_report: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(layer_merge_report, dict):
        return layer_merge_report
    if layer_merge_report is None:
        path = Path(candidate_model) / "neural_forge_layer_merge_report.json"
    else:
        path = Path(layer_merge_report)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _fold_groups_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("operation") != "merge_decoder_layers":
        raise ValueError("Layer merge report must come from merge_decoder_layers.")
    if report.get("method") != "fold":
        return []
    groups = []
    folded_indices = sorted({int(idx) for item in report.get("groups", []) for idx in item.get("folded", [])})
    for item in report.get("groups", []):
        group = [int(idx) for idx in item.get("group", [])]
        folded = [int(idx) for idx in item.get("folded", [])]
        survivor = int(item.get("survivor", group[0] if group else -1))
        if len(group) < 2 or not folded:
            continue
        groups.append(
            {
                "group": group,
                "survivor": survivor,
                "folded": folded,
                "all_folded_indices": folded_indices,
            }
        )
    return groups


def _candidate_index_after_depth_fold(original_index: int, folded_indices: Sequence[int]) -> int:
    return int(original_index) - sum(1 for idx in folded_indices if int(idx) < int(original_index))


def _reconstruct_fold_group(
    torch,
    source,
    candidate,
    source_layers: Sequence[Any],
    candidate_layers: Sequence[Any],
    group: dict[str, Any],
    batches: Sequence[dict[str, Any]],
    ridge: float,
    reconstruction_alpha: float,
    reconstruction_target: str,
    reconstruction_objective: str,
    cosine_steps: int,
    cosine_lr: float,
    cosine_l2: float,
    max_tokens_per_layer: int,
) -> dict[str, Any]:
    source_consumer_idx = max(group["group"]) + 1
    candidate_consumer_idx = _candidate_index_after_depth_fold(source_consumer_idx, group["all_folded_indices"])
    base_report = {
        "group": group["group"],
        "survivor": group["survivor"],
        "folded": group["folded"],
        "source_consumer_layer": source_consumer_idx,
        "candidate_consumer_layer": candidate_consumer_idx,
    }
    if source_consumer_idx >= len(source_layers) or candidate_consumer_idx >= len(candidate_layers):
        return {**base_report, "skipped": True, "reason": "no_downstream_consumer_layer"}

    source_layer = source_layers[source_consumer_idx]
    candidate_layer = candidate_layers[candidate_consumer_idx]
    repair_report: dict[str, Any] = {**base_report, "skipped": False}

    if reconstruction_target in {"survivor-mlp", "both"}:
        source_survivor_idx = max(group["group"])
        candidate_survivor_idx = _candidate_index_after_depth_fold(group["survivor"], group["all_folded_indices"])
        if source_survivor_idx < len(source_layers) and candidate_survivor_idx < len(candidate_layers):
            repair_report["survivor_mlp_reconstruction"] = _reconstruct_survivor_mlp_output(
                torch,
                source,
                candidate,
                source_layers[source_survivor_idx],
                candidate_layers[candidate_survivor_idx],
                batches,
                ridge=ridge,
                reconstruction_alpha=reconstruction_alpha,
                reconstruction_objective=reconstruction_objective,
                cosine_steps=cosine_steps,
                cosine_lr=cosine_lr,
                cosine_l2=cosine_l2,
                max_tokens_per_layer=max_tokens_per_layer,
            )
        else:
            repair_report["survivor_mlp_reconstruction"] = {
                "skipped": True,
                "reason": "survivor_layer_out_of_range",
            }

    if reconstruction_target == "survivor-mlp":
        return repair_report

    source_attn_norm = _get_layer_module(source_layer, ("input_layernorm", "ln_1"))
    candidate_attn_norm = _get_layer_module(candidate_layer, ("input_layernorm", "ln_1"))
    attn_linears = _attention_input_linears(candidate_layer)
    if source_attn_norm is not None and candidate_attn_norm is not None and attn_linears:
        source_features = _collect_module_activations(torch, source, source_attn_norm, batches, max_tokens_per_layer)
        candidate_features = _collect_module_activations(torch, candidate, candidate_attn_norm, batches, max_tokens_per_layer)
        fitted = _fit_activation_map(
            torch,
            candidate_features,
            source_features,
            ridge,
            reconstruction_alpha,
            reconstruction_objective,
            cosine_steps,
            cosine_lr,
            cosine_l2,
        )
        _absorb_input_map_into_linears(torch, attn_linears, fitted["map"])
        repair_report["attention_reconstruction"] = {
            key: value for key, value in fitted.items() if key != "map"
        } | {"linears_updated": [name for name, _module in attn_linears]}
    else:
        repair_report["attention_reconstruction"] = {
            "skipped": True,
            "reason": "missing_layernorm_or_attention_input_linears",
        }

    source_mlp_norm = _get_layer_module(source_layer, ("post_attention_layernorm", "ln_2"))
    candidate_mlp_norm = _get_layer_module(candidate_layer, ("post_attention_layernorm", "ln_2"))
    mlp_linears = _mlp_input_linears(candidate_layer)
    if source_mlp_norm is not None and candidate_mlp_norm is not None and mlp_linears:
        source_features = _collect_module_activations(torch, source, source_mlp_norm, batches, max_tokens_per_layer)
        candidate_features = _collect_module_activations(torch, candidate, candidate_mlp_norm, batches, max_tokens_per_layer)
        fitted = _fit_activation_map(
            torch,
            candidate_features,
            source_features,
            ridge,
            reconstruction_alpha,
            reconstruction_objective,
            cosine_steps,
            cosine_lr,
            cosine_l2,
        )
        _absorb_input_map_into_linears(torch, mlp_linears, fitted["map"])
        repair_report["mlp_reconstruction"] = {
            key: value for key, value in fitted.items() if key != "map"
        } | {"linears_updated": [name for name, _module in mlp_linears]}
    else:
        repair_report["mlp_reconstruction"] = {
            "skipped": True,
            "reason": "missing_layernorm_or_mlp_input_linears",
        }

    return repair_report


def _get_layer_module(layer, names: Sequence[str]):
    for name in names:
        module = getattr(layer, name, None)
        if module is not None:
            return module
    return None


def _attention_input_linears(layer) -> list[tuple[str, Any]]:
    attention = getattr(layer, "self_attn", None) or getattr(layer, "attention", None) or getattr(layer, "attn", None)
    if attention is None:
        return []
    linears = []
    for name in ("q_proj", "k_proj", "v_proj", "query_key_value", "c_attn"):
        module = getattr(attention, name, None)
        if _is_linear_like(module):
            linears.append((f"self_attn.{name}", module))
    return linears


def _mlp_input_linears(layer) -> list[tuple[str, Any]]:
    mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
    if mlp is None:
        return []
    linears = []
    for name in ("gate_proj", "up_proj", "w1", "w3", "dense_h_to_4h", "fc1"):
        module = getattr(mlp, name, None)
        if _is_linear_like(module):
            linears.append((f"mlp.{name}", module))
    return linears


def _reconstruct_survivor_mlp_output(
    torch,
    source,
    candidate,
    source_last_layer,
    candidate_survivor_layer,
    batches: Sequence[dict[str, Any]],
    ridge: float,
    reconstruction_alpha: float,
    reconstruction_objective: str,
    cosine_steps: int,
    cosine_lr: float,
    cosine_l2: float,
    max_tokens_per_layer: int,
) -> dict[str, Any]:
    spec = _mlp_projection_spec(candidate_survivor_layer)
    if spec is None:
        return {"skipped": True, "reason": "unsupported_mlp_shape"}

    source_output = _collect_layer_output_activations(torch, source, source_last_layer, batches, max_tokens_per_layer)
    hidden, residual, candidate_output = _collect_survivor_mlp_calibration(
        torch,
        candidate,
        candidate_survivor_layer,
        batches,
        max_tokens_per_layer,
    )
    rows = min(int(source_output.shape[0]), int(hidden.shape[0]), int(residual.shape[0]))
    source_rows = source_output[:rows].to(torch.float32)
    hidden_rows = hidden[:rows].to(torch.float32)
    residual_rows = residual[:rows].to(torch.float32)
    candidate_rows = candidate_output[:rows].to(torch.float32)
    target_mlp_output = source_rows - residual_rows
    solved_report = _solve_linear_reconstruction(
        torch,
        hidden_rows,
        target_mlp_output,
        ridge,
        reconstruction_objective,
        cosine_steps,
        cosine_lr,
        cosine_l2,
    )
    solved = solved_report["map"]
    old_weight = spec["output"].weight.data.detach().cpu().to(torch.float32)
    solved_weight = solved.T.contiguous()
    if tuple(old_weight.shape) != tuple(solved_weight.shape):
        return {
            "skipped": True,
            "reason": "solved_weight_shape_mismatch",
            "old_weight_shape": list(old_weight.shape),
            "solved_weight_shape": list(solved_weight.shape),
        }
    blended_weight = (1.0 - float(reconstruction_alpha)) * old_weight + float(reconstruction_alpha) * solved_weight
    before = _relative_reconstruction_error(torch, candidate_rows, source_rows)
    solved_output = residual_rows + hidden_rows.matmul(solved)
    solved_error = _relative_reconstruction_error(torch, solved_output, source_rows)
    blended_output = residual_rows + hidden_rows.matmul(blended_weight.T)
    after = _relative_reconstruction_error(torch, blended_output, source_rows)
    before_cosine = _mean_row_cosine(torch, candidate_rows, source_rows)
    solved_cosine = _mean_row_cosine(torch, solved_output, source_rows)
    after_cosine = _mean_row_cosine(torch, blended_output, source_rows)
    spec["output"].weight.data.copy_(blended_weight.to(device=spec["output"].weight.device, dtype=spec["output"].weight.dtype))
    return {
        "skipped": False,
        "tokens": rows,
        "feature_size": int(hidden_rows.shape[1]),
        "hidden_size": int(source_rows.shape[1]),
        "output_projection": spec["output_name"],
        "reconstruction_alpha": float(reconstruction_alpha),
        "reconstruction_objective": reconstruction_objective,
        "cosine_steps": int(cosine_steps),
        "cosine_lr": float(cosine_lr),
        "cosine_l2": float(cosine_l2),
        "relative_error_before": before,
        "relative_error_solved": solved_error,
        "relative_error_after": after,
        "mean_cosine_before": before_cosine,
        "mean_cosine_solved": solved_cosine,
        "mean_cosine_after": after_cosine,
        "solver": {key: value for key, value in solved_report.items() if key != "map"},
    }


def _mlp_projection_spec(layer) -> dict[str, Any] | None:
    mlp = getattr(layer, "mlp", None)
    feed_forward = getattr(layer, "feed_forward", None)
    if feed_forward is not None and all(hasattr(feed_forward, name) for name in ("w1", "w3", "w2")):
        return {
            "kind": "gated",
            "mlp": feed_forward,
            "gate": feed_forward.w1,
            "up": feed_forward.w3,
            "output": feed_forward.w2,
            "output_name": "feed_forward.w2",
        }
    if mlp is None:
        return None
    if all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
        return {
            "kind": "gated",
            "mlp": mlp,
            "gate": mlp.gate_proj,
            "up": mlp.up_proj,
            "output": mlp.down_proj,
            "output_name": "mlp.down_proj",
        }
    if all(hasattr(mlp, name) for name in ("dense_h_to_4h", "dense_4h_to_h")):
        return {
            "kind": "dense",
            "mlp": mlp,
            "input": mlp.dense_h_to_4h,
            "output": mlp.dense_4h_to_h,
            "output_name": "mlp.dense_4h_to_h",
        }
    if all(hasattr(mlp, name) for name in ("fc1", "fc2")):
        return {
            "kind": "dense",
            "mlp": mlp,
            "input": mlp.fc1,
            "output": mlp.fc2,
            "output_name": "mlp.fc2",
        }
    return None


def _mlp_hidden_features(torch, spec: dict[str, Any], inputs):
    module_inputs = inputs.to(device=spec["output"].weight.device, dtype=spec["output"].weight.dtype)
    if spec["kind"] == "gated":
        activation = getattr(spec["mlp"], "act_fn", None) or torch.nn.functional.silu
        return activation(spec["gate"](module_inputs)) * spec["up"](module_inputs)
    activation = getattr(spec["mlp"], "act_fn", None) or getattr(spec["mlp"], "act", None) or torch.nn.functional.gelu
    return activation(spec["input"](module_inputs))


def _collect_layer_output_activations(torch, model, layer, batches: Sequence[dict[str, Any]], max_tokens: int):
    pieces = []
    token_count = 0
    current_mask = None

    def hook(_module, _inputs, output):
        nonlocal token_count
        tensor = output[0] if isinstance(output, tuple) else output
        rows = _select_token_rows(torch, tensor.detach().to(torch.float32), current_mask)
        remaining = max_tokens - token_count
        if remaining > 0:
            pieces.append(rows[:remaining].detach().cpu())
            token_count += int(min(rows.shape[0], remaining))

    handle = layer.register_forward_hook(hook)
    was_training = model.training
    try:
        model.eval()
        for batch in batches:
            if token_count >= max_tokens:
                break
            current_mask = batch.get("attention_mask")
            model(**batch, use_cache=False)
    finally:
        handle.remove()
        if was_training:
            model.train()
    if not pieces:
        raise RuntimeError("No layer output activations were captured.")
    return torch.cat(pieces, dim=0)


def _collect_survivor_mlp_calibration(
    torch,
    model,
    layer,
    batches: Sequence[dict[str, Any]],
    max_tokens: int,
):
    spec = _mlp_projection_spec(layer)
    if spec is None:
        raise ValueError("Unsupported MLP shape.")
    hidden_pieces = []
    mlp_output_pieces = []
    layer_output_pieces = []
    token_count = 0
    current_mask = None

    def mlp_hook(_module, inputs, output):
        if token_count >= max_tokens:
            return
        hidden_rows = _select_token_rows(
            torch,
            _mlp_hidden_features(torch, spec, inputs[0]).detach().to(torch.float32),
            current_mask,
        )
        output_rows = _select_token_rows(torch, output.detach().to(torch.float32), current_mask)
        hidden_pieces.append(hidden_rows.detach().cpu())
        mlp_output_pieces.append(output_rows.detach().cpu())

    def layer_hook(_module, _inputs, output):
        nonlocal token_count
        tensor = output[0] if isinstance(output, tuple) else output
        rows = _select_token_rows(torch, tensor.detach().to(torch.float32), current_mask)
        remaining = max_tokens - token_count
        if remaining > 0:
            layer_output_pieces.append(rows[:remaining].detach().cpu())
            token_count += int(min(rows.shape[0], remaining))

    mlp_handle = spec["mlp"].register_forward_hook(mlp_hook)
    layer_handle = layer.register_forward_hook(layer_hook)
    was_training = model.training
    try:
        model.eval()
        for batch in batches:
            if token_count >= max_tokens:
                break
            current_mask = batch.get("attention_mask")
            model(**batch, use_cache=False)
    finally:
        mlp_handle.remove()
        layer_handle.remove()
        if was_training:
            model.train()
    if not hidden_pieces or not mlp_output_pieces or not layer_output_pieces:
        raise RuntimeError("No survivor MLP calibration activations were captured.")
    hidden = torch.cat(hidden_pieces, dim=0)[:max_tokens]
    mlp_output = torch.cat(mlp_output_pieces, dim=0)[:max_tokens]
    layer_output = torch.cat(layer_output_pieces, dim=0)[:max_tokens]
    rows = min(int(hidden.shape[0]), int(mlp_output.shape[0]), int(layer_output.shape[0]))
    hidden = hidden[:rows]
    mlp_output = mlp_output[:rows]
    layer_output = layer_output[:rows]
    residual = layer_output - mlp_output
    return hidden, residual, layer_output


def _is_linear_like(module) -> bool:
    weight = getattr(module, "weight", None)
    return weight is not None and getattr(weight, "dim", lambda: 0)() == 2


def _collect_module_activations(torch, model, module, batches: Sequence[dict[str, Any]], max_tokens: int):
    pieces = []
    token_count = 0
    current_mask = None

    def hook(_module, _inputs, output):
        nonlocal token_count
        tensor = output[0] if isinstance(output, tuple) else output
        rows = _select_token_rows(torch, tensor.detach().to(torch.float32), current_mask)
        remaining = max_tokens - token_count
        if remaining > 0:
            pieces.append(rows[:remaining].detach().cpu())
            token_count += int(min(rows.shape[0], remaining))

    handle = module.register_forward_hook(hook)
    was_training = model.training
    try:
        model.eval()
        for batch in batches:
            if token_count >= max_tokens:
                break
            current_mask = batch.get("attention_mask")
            model(**batch, use_cache=False)
    finally:
        handle.remove()
        if was_training:
            model.train()
    if not pieces:
        raise RuntimeError("No calibration activations were captured.")
    return torch.cat(pieces, dim=0)


def _select_token_rows(torch, tensor, attention_mask):
    values = tensor
    if values.dim() == 3:
        values = values.reshape(-1, values.shape[-1])
        if attention_mask is not None:
            mask = attention_mask.reshape(-1).detach().to(torch.bool).cpu()
            values = values.cpu()[mask]
            return values
    elif values.dim() == 1:
        values = values.unsqueeze(0)
    elif values.dim() != 2:
        values = values.reshape(-1, values.shape[-1])
    return values.cpu()


def _fit_activation_map(
    torch,
    candidate_features,
    source_features,
    ridge: float,
    reconstruction_alpha: float = 1.0,
    reconstruction_objective: str = "ridge",
    cosine_steps: int = 80,
    cosine_lr: float = 1e-2,
    cosine_l2: float = 1e-4,
) -> dict[str, Any]:
    rows = min(int(candidate_features.shape[0]), int(source_features.shape[0]))
    if rows <= 0:
        raise RuntimeError("Need at least one activation row to fit reconstruction.")
    candidate_rows = candidate_features[:rows].to(torch.float32)
    source_rows = source_features[:rows].to(torch.float32)
    if candidate_rows.shape[1] != source_rows.shape[1]:
        raise ValueError(
            f"Activation hidden-size mismatch: {candidate_rows.shape[1]} != {source_rows.shape[1]}"
        )
    before = _relative_reconstruction_error(torch, candidate_rows, source_rows)
    before_cosine = _mean_row_cosine(torch, candidate_rows, source_rows)
    solved_report = _solve_linear_reconstruction(
        torch,
        candidate_rows,
        source_rows,
        ridge,
        reconstruction_objective,
        cosine_steps,
        cosine_lr,
        cosine_l2,
    )
    solved = solved_report["map"]
    solved_error = _relative_reconstruction_error(torch, candidate_rows.matmul(solved), source_rows)
    solved_cosine = _mean_row_cosine(torch, candidate_rows.matmul(solved), source_rows)
    if reconstruction_alpha < 1.0:
        identity = torch.eye(solved.shape[0], dtype=solved.dtype, device=solved.device)
        solved = (1.0 - float(reconstruction_alpha)) * identity + float(reconstruction_alpha) * solved
    after = _relative_reconstruction_error(torch, candidate_rows.matmul(solved), source_rows)
    after_cosine = _mean_row_cosine(torch, candidate_rows.matmul(solved), source_rows)
    return {
        "map": solved,
        "tokens": rows,
        "hidden_size": int(candidate_rows.shape[1]),
        "reconstruction_alpha": float(reconstruction_alpha),
        "reconstruction_objective": reconstruction_objective,
        "relative_error_before": before,
        "relative_error_solved": solved_error,
        "relative_error_after": after,
        "mean_cosine_before": before_cosine,
        "mean_cosine_solved": solved_cosine,
        "mean_cosine_after": after_cosine,
        "solver": {key: value for key, value in solved_report.items() if key != "map"},
    }


def _solve_linear_reconstruction(
    torch,
    features,
    targets,
    ridge: float,
    reconstruction_objective: str,
    cosine_steps: int,
    cosine_lr: float,
    cosine_l2: float,
) -> dict[str, Any]:
    ridge_map = _ridge_solve_features(torch, features, targets, ridge)
    if reconstruction_objective == "ridge" or cosine_steps == 0:
        return {
            "map": ridge_map,
            "objective": "ridge",
            "ridge": float(ridge),
            "cosine_refined": False,
        }
    if reconstruction_objective != "cosine":
        raise ValueError("reconstruction_objective must be ridge or cosine.")
    refined, refine_report = _refine_linear_map_by_cosine(
        torch,
        features,
        targets,
        ridge_map,
        steps=cosine_steps,
        lr=cosine_lr,
        l2=cosine_l2,
    )
    return {
        "map": refined,
        "objective": "cosine",
        "ridge": float(ridge),
        "cosine_refined": True,
        **refine_report,
    }


def _refine_linear_map_by_cosine(torch, features, targets, initial_map, steps: int, lr: float, l2: float):
    features = torch.nan_to_num(features.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    targets = torch.nan_to_num(targets.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    matrix = torch.nn.Parameter(initial_map.detach().clone().to(torch.float32))
    anchor = initial_map.detach().clone().to(torch.float32)
    optimizer = torch.optim.AdamW([matrix], lr=float(lr), weight_decay=0.0)
    start_cosine = _mean_row_cosine(torch, features.matmul(matrix.detach()), targets)
    start_error = _relative_reconstruction_error(torch, features.matmul(matrix.detach()), targets)
    final_loss = None
    with torch.enable_grad():
        for _step in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            predicted = features.matmul(matrix)
            cosine_loss = 1.0 - torch.nn.functional.cosine_similarity(predicted, targets, dim=1, eps=1e-8).mean()
            if l2 > 0:
                cosine_loss = cosine_loss + float(l2) * torch.mean((matrix - anchor) ** 2)
            if not torch.isfinite(cosine_loss):
                break
            cosine_loss.backward()
            optimizer.step()
            final_loss = float(cosine_loss.detach().cpu())
    refined = matrix.detach()
    return refined, {
        "cosine_steps": int(steps),
        "cosine_lr": float(lr),
        "cosine_l2": float(l2),
        "cosine_loss_final": final_loss,
        "mean_cosine_initial": start_cosine,
        "mean_cosine_final": _mean_row_cosine(torch, features.matmul(refined), targets),
        "relative_error_initial": start_error,
        "relative_error_final": _relative_reconstruction_error(torch, features.matmul(refined), targets),
    }


def _relative_reconstruction_error(torch, predicted, target) -> float:
    return float((predicted - target).norm().detach().cpu() / target.norm().detach().cpu().clamp_min(1e-12))


def _mean_row_cosine(torch, predicted, target) -> float:
    predicted = torch.nan_to_num(predicted.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    target = torch.nan_to_num(target.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    return float(torch.nn.functional.cosine_similarity(predicted, target, dim=1, eps=1e-8).mean().detach().cpu())


def _absorb_input_map_into_linears(torch, linears: Sequence[tuple[str, Any]], mapping) -> None:
    map_cpu = mapping.detach().cpu().to(torch.float32)
    for _name, linear in linears:
        weight = linear.weight.data.detach().cpu().to(torch.float32)
        if int(weight.shape[1]) != int(map_cpu.shape[0]):
            raise ValueError(
                f"Cannot absorb reconstruction map with input {map_cpu.shape[0]} into weight {tuple(weight.shape)}."
            )
        updated = weight.matmul(map_cpu.T)
        linear.weight.data.copy_(updated.to(device=linear.weight.device, dtype=linear.weight.dtype))


def _recovery_loss(torch, student, teacher, input_ids, attention_mask, kd_weight: float, lm_weight: float, temperature: float):
    shift_student = student.logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    if attention_mask is None:
        mask = torch.ones_like(shift_labels, dtype=torch.bool)
    else:
        mask = attention_mask[:, 1:].contiguous().bool()
    shift_labels = shift_labels.masked_fill(~mask, -100)
    total = torch.zeros((), dtype=shift_student.dtype, device=shift_student.device)
    if kd_weight > 0:
        if teacher is None:
            raise ValueError("teacher output is required when kd_weight is positive.")
        shift_teacher = teacher.logits[:, :-1, :].contiguous()
        student_logp = torch.nn.functional.log_softmax(shift_student / temperature, dim=-1)
        teacher_p = torch.nn.functional.softmax(shift_teacher / temperature, dim=-1)
        token_kl = torch.nn.functional.kl_div(student_logp, teacher_p, reduction="none").sum(dim=-1)
        kd = (token_kl * mask).sum() / mask.sum().clamp_min(1)
        kd = kd * (temperature**2)
        total = total + float(kd_weight) * kd
    if lm_weight > 0:
        lm = torch.nn.functional.cross_entropy(
            shift_student.view(-1, shift_student.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        total = total + float(lm_weight) * lm
    return total


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
