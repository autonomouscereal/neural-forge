"""Hugging Face checkpoint surgery and lightweight behavior probes.

This module is imported lazily by the CLI so the NumPy-only dense engine remains
usable without installing torch/transformers.
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Sequence


TEXT_EVALS = [
    "In 2026, compact language models are useful when they preserve reasoning while reducing latency.",
    "A careful model merge should be evaluated with perplexity, answer probes, and generation samples.",
    "Neural pruning removes or folds redundant channels, but aggressive compression often needs recovery training.",
    "Vision-language models can describe simple objects, read short text, and ground answers in pixels.",
]

CHOICE_PROBES = [
    {
        "id": "math_addition",
        "prompt": "Question: What is 17 + 28?\nAnswer:",
        "choices": [" 45", " 44", " 46", " 39"],
        "answer": 0,
    },
    {
        "id": "paper_tool",
        "prompt": "Question: Which tool is usually used to cut paper?\nA. spoon\nB. scissors\nC. pillow\nD. cloud\nAnswer:",
        "choices": [" A", " B", " C", " D"],
        "answer": 1,
    },
    {
        "id": "python_len",
        "prompt": "Question: In Python, which expression returns the length of list x?\nA. size(x)\nB. count(x)\nC. len(x)\nD. width(x)\nAnswer:",
        "choices": [" A", " B", " C", " D"],
        "answer": 2,
    },
    {
        "id": "causal_reasoning",
        "prompt": "Question: A glass fell from a table and shattered. What most likely caused the shattered pieces?\nA. gravity pulled it down\nB. it became colder\nC. the table grew taller\nD. the room got darker\nAnswer:",
        "choices": [" A", " B", " C", " D"],
        "answer": 0,
    },
]

GENERATION_PROMPTS = [
    "Explain neural pruning in two concise sentences.",
    "Write a tiny Python function that adds two numbers.",
    "Give one risk when merging two language models.",
]

GENERATION_RELEVANCE_RULES = {
    GENERATION_PROMPTS[0]: [
        [r"\bprun\w*", r"\bremove\w*", r"\beliminat\w*", r"\bcompress\w*", r"\bfold\w*"],
        [r"\bneuron\w*", r"\bweight\w*", r"\bchannel\w*", r"\bparameter\w*", r"\bconnection\w*"],
    ],
    GENERATION_PROMPTS[1]: [
        [r"\bdef\b", r"\blambda\b"],
        [r"\+", r"\badd\w*\b", r"\bsum\b"],
        [r"\breturn\b", r":"],
    ],
    GENERATION_PROMPTS[2]: [
        [r"\bmerge\w*", r"\bstitch\w*", r"\bcombin\w*", r"\bfus\w*"],
        [
            r"\brisk\w*",
            r"\bforget\w*",
            r"\bdegrad\w*",
            r"\bcollapse\w*",
            r"\bmisalign\w*",
            r"\bmismatch\w*",
            r"\bincompat\w*",
            r"\binterfer\w*",
            r"\bloss\b",
            r"\bhurt\w*",
            r"\bbias\w*",
        ],
        [
            r"\bmodel\w*",
            r"\btokenizer\w*",
            r"\bcapabilit\w*",
            r"\baccuracy\b",
            r"\bbehavior\w*",
            r"\bweights?\b",
        ],
    ],
}

CRITICAL_EVAL_CATEGORIES = ("math", "code", "tokenizer", "model_surgery")

BEHAVIOR_PROBES = [
    {
        "id": "answer_only_multiply",
        "prompt": "Question: What is 12 * 13? Answer with only the number.\nAnswer:",
        "kind": "first_number",
        "answer": "156",
    },
    {
        "id": "answer_only_addition",
        "prompt": "Question: What is 19 + 26? Answer with only the number.\nAnswer:",
        "kind": "first_number",
        "answer": "45",
    },
    {
        "id": "tokenizer_mismatch",
        "prompt": "In one sentence, why can tokenizer mismatch hurt model stitching?\nAnswer:",
        "kind": "pattern_groups",
        "pattern_groups": [
            [r"\btokeni[sz]er\w*", r"\bvocab\w*", r"\btokens?\b"],
            [r"\bmismatch\w*", r"\bdiffer\w*", r"\balign\w*", r"\bincompat\w*"],
            [r"\bembedding\w*", r"\blogits?\b", r"\btoken ids?\b", r"\brows?\b"],
        ],
    },
    {
        "id": "rain_common_sense",
        "prompt": "A driver sees heavy rain on the road. What should they do?\nAnswer:",
        "kind": "pattern_groups",
        "pattern_groups": [
            [r"\bslow\w*", r"\bcaution\w*", r"\bsafe\w*", r"\breduce\w*"],
            [r"\bdistance\b", r"\bfollow\w*", r"\bheadlights?\b", r"\bwipers?\b", r"\bpull over\b"],
        ],
    },
    {
        "id": "python_add_function",
        "prompt": "Write a tiny Python function named add that returns the sum of two inputs.\n```python\n",
        "kind": "python_function",
        "function_name": "add",
        "tests": [
            {"args": [2, 3], "expected": 5},
            {"args": [-1, 5], "expected": 4},
            {"args": [0, 0], "expected": 0},
        ],
    },
    {
        "id": "neuron_merge_definition",
        "prompt": "Briefly define neuron merging for model compression.\nAnswer:",
        "kind": "pattern_groups",
        "pattern_groups": [
            [r"\bmerg\w*", r"\bfold\w*", r"\bcombin\w*"],
            [r"\bneurons?\b", r"\bchannels?\b", r"\bfilters?\b"],
            [r"\bsimilar\w*", r"\bredundan\w*", r"\baverage\w*", r"\bsum\w*"],
        ],
    },
]


def random_init_causal_lm(
    config_model: str,
    output_dir: str | Path,
    tokenizer_model: str | None = None,
    seed: int = 0,
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Create an untrained HF causal LM from an existing config/tokenizer."""

    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    torch.manual_seed(int(seed))
    config = transformers.AutoConfig.from_pretrained(config_model, trust_remote_code=trust_remote_code)
    model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    tokenizer_source = tokenizer_model or config_model
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)
    vocab_capacity = _ensure_tokenizer_vocab_capacity(model, tokenizer)
    if dtype is not None:
        model.to(dtype=dtype)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    report = {
        "operation": "random_init_causal_lm",
        "config_model": config_model,
        "tokenizer_model": tokenizer_source,
        "output_dir": str(target),
        "seed": int(seed),
        "torch_dtype": torch_dtype,
        "vocab_size": vocab_capacity["model_vocab_size"],
        "tokenizer_vocab_size": vocab_capacity["tokenizer_vocab_size"],
        "vocab_capacity": vocab_capacity,
        "param_count": sum(param.numel() for param in model.parameters()),
    }
    _write_json(target / "neural_forge_random_init_report.json", report)
    return report


def _set_config_shape_value(config, names: Sequence[str], value: int) -> list[str]:
    updated = []
    targets = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None and text_config is not config:
        targets.append(text_config)
    for target in targets:
        matched = False
        for name in names:
            if hasattr(target, name):
                setattr(target, name, int(value))
                updated.append(name)
                matched = True
        if not matched and target is targets[-1]:
            setattr(target, names[0], int(value))
            updated.append(names[0])
    return list(dict.fromkeys(updated))


def _ensure_tokenizer_vocab_capacity(model, tokenizer) -> dict[str, Any]:
    """Grow an embedding table for a tokenizer without deleting reserved rows."""

    tokenizer_vocab_size = int(len(tokenizer))
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        return {
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "original_model_vocab_size": None,
            "model_vocab_size": None,
            "resized": False,
        }
    original_model_vocab_size = int(input_embeddings.num_embeddings)
    resized = tokenizer_vocab_size > original_model_vocab_size
    if resized:
        model.resize_token_embeddings(tokenizer_vocab_size)
    model_vocab_size = int(model.get_input_embeddings().num_embeddings)
    return {
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "original_model_vocab_size": original_model_vocab_size,
        "model_vocab_size": model_vocab_size,
        "resized": bool(resized),
    }


def shape_random_init_causal_lm(
    config_model: str,
    output_dir: str | Path,
    tokenizer_model: str | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    num_hidden_layers: int | None = None,
    num_attention_heads: int | None = None,
    num_key_value_heads: int | None = None,
    head_dim: int | None = None,
    tie_word_embeddings: bool | None = None,
    seed: int = 0,
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Create a blank HF causal LM after explicit architecture-shape overrides."""

    requested = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
    }
    for name, value in requested.items():
        if value is not None and int(value) <= 0:
            raise ValueError(f"{name} must be positive when provided.")
    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    torch.manual_seed(int(seed))
    config = transformers.AutoConfig.from_pretrained(config_model, trust_remote_code=trust_remote_code)
    aliases = {
        "hidden_size": ("hidden_size", "n_embd", "d_model"),
        "intermediate_size": ("intermediate_size", "ffn_dim", "n_inner"),
        "num_hidden_layers": ("num_hidden_layers", "n_layer", "num_layers"),
        "num_attention_heads": ("num_attention_heads", "n_head", "num_heads"),
        "num_key_value_heads": ("num_key_value_heads", "num_kv_heads"),
        "head_dim": ("head_dim",),
    }
    updated_fields = {}
    for name, value in requested.items():
        if value is not None:
            updated_fields[name] = _set_config_shape_value(config, aliases[name], int(value))
    if tie_word_embeddings is not None:
        config.tie_word_embeddings = bool(tie_word_embeddings)
        if getattr(config, "text_config", None) is not None:
            config.text_config.tie_word_embeddings = bool(tie_word_embeddings)

    effective_heads = int(
        num_attention_heads
        or getattr(config, "num_attention_heads", 0)
        or getattr(getattr(config, "text_config", None), "num_attention_heads", 0)
    )
    effective_kv_heads = int(
        num_key_value_heads
        or getattr(config, "num_key_value_heads", effective_heads)
        or getattr(getattr(config, "text_config", None), "num_key_value_heads", effective_heads)
    )
    if effective_heads <= 0 or effective_kv_heads <= 0 or effective_heads % effective_kv_heads != 0:
        raise ValueError("num_attention_heads must be a positive multiple of num_key_value_heads.")

    model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    tokenizer_source = tokenizer_model or config_model
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)
    vocab_capacity = _ensure_tokenizer_vocab_capacity(model, tokenizer)
    if tie_word_embeddings is True and hasattr(model, "tie_weights"):
        model.tie_weights()
    if dtype is not None:
        model.to(dtype=dtype)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    report = {
        "operation": "shape_random_init_causal_lm",
        "config_model": config_model,
        "tokenizer_model": tokenizer_source,
        "output_dir": str(target),
        "seed": int(seed),
        "torch_dtype": torch_dtype,
        "requested_shape": requested,
        "updated_config_fields": updated_fields,
        "tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False)),
        "vocab_size": vocab_capacity["model_vocab_size"],
        "tokenizer_vocab_size": vocab_capacity["tokenizer_vocab_size"],
        "vocab_capacity": vocab_capacity,
        "param_count": sum(param.numel() for param in model.parameters()),
    }
    _write_json(target / "neural_forge_shape_init_report.json", report)
    return report


def same_shape_merge(
    model_ids: Sequence[str],
    output_dir: str | Path,
    weights: Sequence[float] | None = None,
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    torch, transformers = _torch_and_transformers()
    if len(model_ids) < 2:
        raise ValueError("Need at least two models to merge.")
    merge_weights = _normalized_weights(len(model_ids), weights)
    dtype = _dtype_from_name(torch, torch_dtype)

    base_id = model_ids[0]
    base = transformers.AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_id, trust_remote_code=trust_remote_code)
    merged_state = {name: tensor.detach().cpu().to(torch.float32).mul_(merge_weights[0]) for name, tensor in base.state_dict().items()}
    shapes = {name: tuple(tensor.shape) for name, tensor in merged_state.items()}
    source_dtypes = {name: tensor.dtype for name, tensor in base.state_dict().items()}

    for weight, model_id in zip(merge_weights[1:], model_ids[1:]):
        other = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="cpu",
            trust_remote_code=trust_remote_code,
        )
        other_state = other.state_dict()
        if set(other_state) != set(merged_state):
            missing = sorted(set(merged_state) - set(other_state))[:8]
            extra = sorted(set(other_state) - set(merged_state))[:8]
            raise ValueError(f"State dict key mismatch for {model_id}; missing={missing}; extra={extra}")
        for name, tensor in other_state.items():
            if tuple(tensor.shape) != shapes[name]:
                raise ValueError(f"Shape mismatch for {name}: {tuple(tensor.shape)} != {shapes[name]}")
            merged_state[name].add_(tensor.detach().cpu().to(torch.float32), alpha=weight)
        del other

    cast_state = {
        name: tensor.to(source_dtypes[name] if dtype is None else dtype) if tensor.is_floating_point() else tensor
        for name, tensor in merged_state.items()
    }
    base.load_state_dict(cast_state, strict=True)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    report = {
        "operation": "same_shape_merge",
        "models": list(model_ids),
        "weights": merge_weights,
        "output_dir": str(target),
        "param_count": sum(p.numel() for p in base.parameters()),
    }
    _write_json(target / "neural_forge_merge_report.json", report)
    return report


def task_arithmetic_merge(
    base_model: str,
    tuned_model: str,
    output_dir: str | Path,
    alpha: float = 1.0,
    torch_dtype: str = "auto",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Create `base + alpha * (tuned - base)` for same-shape HF causal LMs."""

    alpha_value = _finite_float(alpha, "alpha")
    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    base = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    tuned = transformers.AutoModelForCausalLM.from_pretrained(
        tuned_model,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    tuned_tokenizer = transformers.AutoTokenizer.from_pretrained(tuned_model, trust_remote_code=trust_remote_code)
    tokenizer_report = _assert_tokenizers_compatible(tokenizer, tuned_tokenizer)
    base_state = base.state_dict()
    tuned_state = tuned.state_dict()
    if set(base_state) != set(tuned_state):
        missing = sorted(set(base_state) - set(tuned_state))[:8]
        extra = sorted(set(tuned_state) - set(base_state))[:8]
        raise ValueError(f"State dict key mismatch; missing={missing}; extra={extra}")

    merged_state = {}
    for name, base_tensor in base_state.items():
        tuned_tensor = tuned_state[name]
        if tuple(base_tensor.shape) != tuple(tuned_tensor.shape):
            raise ValueError(f"Shape mismatch for {name}: {tuple(base_tensor.shape)} != {tuple(tuned_tensor.shape)}")
        if base_tensor.is_floating_point():
            merged = base_tensor.detach().cpu().to(torch.float32)
            merged.add_(tuned_tensor.detach().cpu().to(torch.float32) - merged, alpha=alpha_value)
            merged_state[name] = merged.to(base_tensor.dtype if dtype is None else dtype)
        else:
            base_cpu = base_tensor.detach().cpu()
            tuned_cpu = tuned_tensor.detach().cpu()
            if not torch.equal(base_cpu, tuned_cpu):
                raise ValueError(f"Non-floating tensor mismatch for {name}; task merge requires matching buffers.")
            merged_state[name] = base_cpu

    base.load_state_dict(merged_state, strict=True)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    report = {
        "operation": "task_arithmetic_merge",
        "base_model": base_model,
        "tuned_model": tuned_model,
        "alpha": alpha_value,
        "torch_dtype": torch_dtype,
        "tokenizer_source": base_model,
        "tokenizer_compatibility": tokenizer_report,
        "output_dir": str(target),
        "param_count": sum(p.numel() for p in base.parameters()),
    }
    _write_json(target / "neural_forge_task_merge_report.json", report)
    return report


def resize_gated_ffn(
    model_id_or_path: str,
    output_dir: str | Path,
    ratio: float | None = None,
    target_intermediate_size: int | None = None,
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
    assignment: str = "combined-cosine",
) -> dict[str, Any]:
    torch, transformers = _torch_and_transformers()
    if assignment not in {"combined-cosine", "down-cosine", "round-robin", "topk"}:
        raise ValueError("assignment must be combined-cosine, down-cosine, round-robin, or topk.")
    if ratio is None and target_intermediate_size is None:
        raise ValueError("Provide ratio or target_intermediate_size.")
    if ratio is not None and ratio <= 0:
        raise ValueError("ratio must be positive.")
    dtype = _dtype_from_name(torch, torch_dtype)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
    layers = _decoder_layers(model)
    if not layers:
        raise ValueError("Could not find decoder layers with gated MLP modules.")
    old_size = int(layers[0].mlp.gate_proj.weight.shape[0])
    new_size = int(target_intermediate_size or round(old_size * float(ratio)))
    if new_size <= 0:
        raise ValueError("target intermediate size must be positive.")

    changes = []
    for idx, layer in enumerate(layers):
        before = int(layer.mlp.gate_proj.weight.shape[0])
        if new_size < before:
            _shrink_gated_mlp_layer(torch, layer, new_size, assignment=assignment)
        elif new_size > before:
            _expand_gated_mlp_layer(torch, layer, new_size)
        after = int(layer.mlp.gate_proj.weight.shape[0])
        changes.append({"layer": idx, "before": before, "after": after})

    _update_intermediate_size(model.config, new_size)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    report = {
        "operation": "resize_gated_ffn",
        "source": model_id_or_path,
        "output_dir": str(target),
        "old_intermediate_size": old_size,
        "new_intermediate_size": new_size,
        "ratio": new_size / old_size,
        "assignment": assignment,
        "layers": changes,
        "param_count": sum(p.numel() for p in model.parameters()),
    }
    _write_json(target / "neural_forge_resize_report.json", report)
    return report


def repair_gated_ffn_with_calibration(
    source_model: str,
    resized_model: str,
    output_dir: str | Path,
    calibration_texts: Sequence[str] | None = None,
    torch_dtype: str = "float16",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    ridge: float = 1e-3,
    max_length: int = 128,
    max_tokens_per_layer: int = 512,
) -> dict[str, Any]:
    """Refit resized gated-FFN down projections against source FFN outputs.

    This is a lightweight GRAIL/RegMean-style compensation pass: no gradient
    training, no new parameters, just ridge least squares on calibration tokens.
    It assumes source and resized models share depth and hidden size, while FFN
    intermediate widths may differ.
    """

    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if max_tokens_per_layer <= 0:
        raise ValueError("max_tokens_per_layer must be positive.")
    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    texts = list(calibration_texts or _default_calibration_texts())
    if not texts:
        raise ValueError("Need at least one calibration text.")

    source = transformers.AutoModelForCausalLM.from_pretrained(
        source_model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    candidate = transformers.AutoModelForCausalLM.from_pretrained(
        resized_model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(source_model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    source.to(device)
    candidate.to(device)
    source.eval()
    candidate.eval()

    source_layers = _decoder_layers(source)
    candidate_layers = _decoder_layers(candidate)
    if len(source_layers) != len(candidate_layers):
        raise ValueError(f"Layer count mismatch: source={len(source_layers)} candidate={len(candidate_layers)}")

    encoded_batches = _encode_calibration_batches(tokenizer, texts, device, max_length=max_length)
    repairs = []
    with torch.no_grad():
        for layer_idx, (source_layer, candidate_layer) in enumerate(zip(source_layers, candidate_layers)):
            inputs, targets = _collect_mlp_calibration(torch, source, source_layer.mlp, encoded_batches, max_tokens_per_layer)
            hidden = _gated_mlp_hidden(torch, candidate_layer.mlp, inputs.to(device)).to(torch.float32)
            target = targets.to(device=device, dtype=torch.float32)
            solved = _ridge_solve_features(torch, hidden, target, ridge)
            _replace_linear(torch, candidate_layer.mlp.down_proj, solved.T, axis="in")
            err_before = _relative_mlp_error(torch, candidate_layer.mlp, inputs.to(device), target)
            repairs.append(
                {
                    "layer": layer_idx,
                    "tokens": int(hidden.shape[0]),
                    "intermediate_size": int(hidden.shape[1]),
                    "relative_error_after": err_before,
                }
            )
            del inputs, targets, hidden, target, solved
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    candidate.save_pretrained(target_dir, safe_serialization=True)
    tokenizer.save_pretrained(target_dir)
    report = {
        "operation": "repair_gated_ffn_with_calibration",
        "source_model": source_model,
        "resized_model": resized_model,
        "output_dir": str(target_dir),
        "ridge": ridge,
        "max_length": max_length,
        "max_tokens_per_layer": max_tokens_per_layer,
        "calibration_text_count": len(texts),
        "repairs": repairs,
        "param_count": sum(p.numel() for p in candidate.parameters()),
    }
    _write_json(target_dir / "neural_forge_repair_report.json", report)
    return report


def merge_decoder_layers(
    model_id_or_path: str,
    output_dir: str | Path,
    groups: Sequence[Sequence[int]],
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
    method: str = "fold",
) -> dict[str, Any]:
    """Fold groups of residual decoder blocks into one block per group.

    This is a merge-first depth reduction. The output keeps the source runtime
    architecture and replaces each group with a merged copy of the first block.
    """

    if method not in {"fold", "average"}:
        raise ValueError("method must be fold or average.")
    normalized_groups = _normalize_layer_groups(groups)
    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
    parent, attr, layers = _decoder_layer_container(model)
    layer_list = list(layers)
    old_count = len(layer_list)
    flat = [idx for group in normalized_groups for idx in group]
    bad = sorted(idx for idx in flat if idx < 0 or idx >= old_count)
    if bad:
        raise ValueError(f"Layer indices out of range for {old_count} layers: {bad}")

    group_by_start = {group[0]: group for group in normalized_groups}
    folded_indices = {idx for group in normalized_groups for idx in group[1:]}
    new_layers = []
    merge_reports = []
    for idx, layer in enumerate(layer_list):
        if idx in folded_indices:
            continue
        group = group_by_start.get(idx)
        if group is None:
            new_layers.append(layer)
            continue
        merged_state = _merge_layer_state_dicts(torch, [layer_list[item].state_dict() for item in group], method=method)
        layer.load_state_dict(merged_state, strict=True)
        new_layers.append(layer)
        merge_reports.append({"group": group, "survivor": idx, "folded": group[1:]})

    setattr(parent, attr, torch.nn.ModuleList(new_layers))
    _merge_config_layer_metadata(model.config, normalized_groups, old_count)
    _update_layer_count(model.config, len(new_layers))

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    report = {
        "operation": "merge_decoder_layers",
        "source": model_id_or_path,
        "output_dir": str(target),
        "old_layer_count": old_count,
        "new_layer_count": len(new_layers),
        "groups": merge_reports,
        "method": method,
        "torch_dtype": torch_dtype,
        "param_count": sum(p.numel() for p in model.parameters()),
    }
    _write_json(target / "neural_forge_layer_merge_report.json", report)
    return report


def drop_decoder_layers(
    model_id_or_path: str,
    output_dir: str | Path,
    drop_layers: Sequence[int],
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Remove selected residual decoder blocks from an HF causal LM."""

    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
    parent, attr, layers = _decoder_layer_container(model)
    old_count = len(layers)
    drop_set = {int(idx) for idx in drop_layers}
    if not drop_set:
        raise ValueError("drop_layers must contain at least one layer index.")
    bad = sorted(idx for idx in drop_set if idx < 0 or idx >= old_count)
    if bad:
        raise ValueError(f"Layer indices out of range for {old_count} layers: {bad}")
    if len(drop_set) >= old_count:
        raise ValueError("Cannot drop every decoder layer.")

    kept_layers = [layer for idx, layer in enumerate(layers) if idx not in drop_set]
    setattr(parent, attr, torch.nn.ModuleList(kept_layers))
    _drop_config_layer_metadata(model.config, drop_set, old_count)
    _update_layer_count(model.config, len(kept_layers))

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    report = {
        "operation": "drop_decoder_layers",
        "source": model_id_or_path,
        "output_dir": str(target),
        "old_layer_count": old_count,
        "new_layer_count": len(kept_layers),
        "dropped_layers": sorted(drop_set),
        "torch_dtype": torch_dtype,
        "param_count": sum(p.numel() for p in model.parameters()),
    }
    _write_json(target / "neural_forge_layer_drop_report.json", report)
    return report


def rank_layer_fold_candidates(
    model_id_or_path: str,
    output_file: str | Path,
    window: int = 2,
    top_k: int = 10,
    max_length: int = 128,
    max_tokens_per_layer: int = 512,
    torch_dtype: str = "float16",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    calibration_texts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Rank adjacent decoder layer groups by activation redundancy.

    The score favors high linear CKA and row-wise cosine between the first and
    last layer outputs in a candidate group, while penalizing large residual
    stream deltas. This is a fold-first selection heuristic, not a pruning
    criterion.
    """

    if window < 2:
        raise ValueError("window must be at least 2.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1.")
    if max_tokens_per_layer <= 0:
        raise ValueError("max_tokens_per_layer must be positive.")

    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    layers = _decoder_layers(model)
    if not layers:
        raise ValueError("Model needs decoder layers.")
    if window > len(layers):
        raise ValueError(f"window {window} exceeds layer count {len(layers)}.")

    texts = list(calibration_texts or TEXT_EVALS)
    if not texts:
        raise ValueError("Need at least one calibration text.")
    with torch.no_grad():
        layer_rows = _collect_all_layer_output_rows(
            torch,
            model,
            tokenizer,
            layers,
            texts,
            max_length=max_length,
            max_tokens_per_layer=max_tokens_per_layer,
            device=device,
        )

    candidates = []
    for start in range(0, len(layers) - window + 1):
        end = start + window - 1
        first = layer_rows[start]
        last = layer_rows[end]
        rows = min(int(first.shape[0]), int(last.shape[0]))
        first = first[:rows].to(torch.float32)
        last = last[:rows].to(torch.float32)
        cka = _linear_cka(torch, first, last)
        cosine = _mean_row_cosine(torch, first, last)
        delta = _relative_delta(torch, first, last)
        variance_ratio = _variance_ratio(torch, first, last)
        score = float(cka + cosine - delta)
        candidates.append(
            {
                "group": list(range(start, start + window)),
                "survivor": start,
                "folded": list(range(start + 1, start + window)),
                "score": score,
                "linear_cka": cka,
                "mean_cosine": cosine,
                "relative_delta": delta,
                "variance_ratio": variance_ratio,
                "tokens": rows,
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    report = {
        "operation": "rank_layer_fold_candidates",
        "source": model_id_or_path,
        "window": int(window),
        "top_k": int(top_k),
        "old_layer_count": len(layers),
        "max_length": int(max_length),
        "max_tokens_per_layer": int(max_tokens_per_layer),
        "torch_dtype": torch_dtype,
        "device": device,
        "calibration_text_count": len(texts),
        "candidates": candidates[:top_k],
    }
    _write_json(output_file, report)
    return report


def evaluate_text_model(
    model_id_or_path: str,
    output_file: str | Path,
    torch_dtype: str = "float16",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    max_new_tokens: int = 96,
    eval_suite: str | Path | dict[str, Any] | None = None,
    max_text_evals: int | None = None,
    max_choice_probes: int | None = None,
    max_generation_prompts: int | None = None,
    max_behavior_probes: int | None = None,
) -> dict[str, Any]:
    torch, transformers = _torch_and_transformers()
    suite = _load_eval_suite(eval_suite)
    text_evals = _limit_sequence(suite["text_evals"], max_text_evals)
    choice_probes = _limit_sequence(suite["choice_probes"], max_choice_probes)
    generation_specs = _limit_sequence(suite["generation_prompts"], max_generation_prompts)
    behavior_probes = _limit_sequence(suite["behavior_probes"], max_behavior_probes)
    dtype = _dtype_from_name(torch, torch_dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    requested_prompts = []
    seen_prompts = set()
    for spec in generation_specs:
        prompt = str(spec["prompt"])
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            requested_prompts.append(prompt)
    for probe in behavior_probes:
        prompt = str(probe["prompt"])
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            requested_prompts.append(prompt)

    with torch.no_grad():
        ppls = [_perplexity(torch, model, tokenizer, text, device) for text in text_evals]
        probes = [_score_choice_probe(torch, model, tokenizer, probe, device) for probe in choice_probes]
        generation_cache = {
            item["prompt"]: item
            for item in _generate_text_batch(torch, model, tokenizer, requested_prompts, device, max_new_tokens)
        }
        generations = [
            _generation_from_cached_spec(dict(generation_cache[str(spec["prompt"])]), spec)
            for spec in generation_specs
        ]
        behavior_items = [
            _score_behavior_probe(
                probe,
                dict(generation_cache[str(probe["prompt"])]),
            )
            for probe in behavior_probes
        ]

    generation_diagnostics = [_generation_health(item, 0, 0.0) for item in generations]
    behavior_correct = sum(1 for item in behavior_items if item["passed"])
    choice_correct = sum(1 for item in probes if item["correct"])
    report = {
        "model": model_id_or_path,
        "param_count": sum(p.numel() for p in model.parameters()),
        "eval_suite": {
            "source": suite["source"],
            "text_eval_count": len(text_evals),
            "choice_probe_count": len(choice_probes),
            "generation_prompt_count": len(generation_specs),
            "behavior_probe_count": len(behavior_probes),
        },
        "perplexity": {
            "mean": float(sum(ppls) / max(1, len(ppls))),
            "items": ppls,
        },
        "choice_accuracy": {
            "correct": choice_correct,
            "total": len(probes),
            "accuracy": choice_correct / max(1, len(probes)),
            "items": probes,
        },
        "generations": generations,
        "generation_diagnostics": generation_diagnostics,
        "generation_health": {
            "correct": sum(1 for item in generation_diagnostics if item["passed"]),
            "total": len(generation_diagnostics),
            "pass_rate": (
                sum(1 for item in generation_diagnostics if item["passed"]) / max(1, len(generation_diagnostics))
            ),
        },
        "behavior_suite": {
            "correct": behavior_correct,
            "total": len(behavior_items),
            "pass_rate": behavior_correct / max(1, len(behavior_items)),
            "items": behavior_items,
        },
    }
    _write_json(output_file, report)
    return report


def build_eval_suite(
    output_file: str | Path,
    behavior_count: int = 1000,
    generation_count: int = 1000,
    choice_count: int = 200,
    text_count: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Write a deterministic broad text eval suite for model-surgery gates."""

    if behavior_count < 0 or generation_count < 0 or choice_count < 0 or text_count < 0:
        raise ValueError("suite counts must be non-negative.")
    rng = random.Random(int(seed))
    suite = {
        "text_evals": _synthetic_text_evals(int(text_count)),
        "choice_probes": _synthetic_choice_probes(int(choice_count), rng),
        "generation_prompts": _synthetic_generation_prompts(int(generation_count)),
        "behavior_probes": _synthetic_behavior_probes(int(behavior_count), rng),
    }
    report = {
        "operation": "build_eval_suite",
        "output_file": str(output_file),
        "seed": int(seed),
        "counts": {
            "text_evals": len(suite["text_evals"]),
            "choice_probes": len(suite["choice_probes"]),
            "generation_prompts": len(suite["generation_prompts"]),
            "behavior_probes": len(suite["behavior_probes"]),
        },
        "suite": suite,
    }
    _write_json(output_file, report)
    return {key: value for key, value in report.items() if key != "suite"}


def build_recovery_calibration(
    output_file: str | Path,
    math_count: int = 512,
    code_count: int = 128,
    tokenizer_count: int = 64,
    model_surgery_count: int = 64,
    commonsense_count: int = 64,
    seed: int = 0,
    exclude_eval_suite: str | Path | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write synthetic non-eval recovery prompt/response rows as JSONL."""

    counts = {
        "math": int(math_count),
        "code": int(code_count),
        "tokenizer": int(tokenizer_count),
        "model_surgery": int(model_surgery_count),
        "commonsense": int(commonsense_count),
    }
    if any(value < 0 for value in counts.values()):
        raise ValueError("recovery calibration counts must be non-negative.")
    rng = random.Random(int(seed))
    rows: list[dict[str, Any]] = []
    rows.extend(_synthetic_recovery_math_rows(counts["math"], rng))
    rows.extend(_synthetic_recovery_code_rows(counts["code"], rng))
    rows.extend(_synthetic_recovery_template_rows("tokenizer", counts["tokenizer"], _RECOVERY_TOKENIZER_TEMPLATES))
    rows.extend(_synthetic_recovery_template_rows("model_surgery", counts["model_surgery"], _RECOVERY_MODEL_SURGERY_TEMPLATES))
    rows.extend(_synthetic_recovery_template_rows("commonsense", counts["commonsense"], _RECOVERY_COMMONSENSE_TEMPLATES))
    rng.shuffle(rows)
    pre_exclusion_counts = _category_counts(rows)
    excluded_prompts = _eval_suite_prompt_set(exclude_eval_suite) if exclude_eval_suite is not None else set()
    excluded_rows = [row for row in rows if str(row.get("prompt", "")) in excluded_prompts]
    if excluded_prompts:
        rows = [row for row in rows if str(row.get("prompt", "")) not in excluded_prompts]

    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    return {
        "operation": "build_recovery_calibration",
        "output_file": str(target),
        "seed": int(seed),
        "counts": counts,
        "pre_exclusion_counts": pre_exclusion_counts,
        "excluded_eval_suite": str(exclude_eval_suite) if exclude_eval_suite is not None else None,
        "excluded_prompt_count": len(excluded_prompts),
        "excluded_row_count": len(excluded_rows),
        "excluded_counts": _category_counts(excluded_rows),
        "total_rows": len(rows),
        "final_counts": _category_counts(rows),
        "sample_ids": [str(row["id"]) for row in rows[:10]],
    }


def _eval_suite_prompt_set(eval_suite: str | Path | dict[str, Any] | None) -> set[str]:
    if eval_suite is None:
        return set()
    suite = _load_eval_suite(eval_suite)
    prompts: set[str] = set()
    for collection in ("generation_prompts", "behavior_probes", "choice_probes"):
        for item in suite.get(collection, []):
            if isinstance(item, dict) and item.get("prompt") is not None:
                prompts.add(str(item["prompt"]))
    return prompts


def _category_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row.get("category") or "unknown")
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def _synthetic_recovery_math_rows(count: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for index in range(int(count)):
        mode = index % 3
        if mode == 0:
            a = rng.randint(3, 99)
            b = rng.randint(2, 99)
            op = "+"
            answer = a + b
        elif mode == 1:
            a = rng.randint(6, 34)
            b = rng.randint(3, 24)
            op = "*"
            answer = a * b
        else:
            b = rng.randint(2, 24)
            answer = rng.randint(2, 64)
            a = b * answer
            op = "/"
        rows.append(
            {
                "id": f"recovery_math_{index:05d}",
                "category": "math",
                "prompt": f"Question: What is {a} {op} {b}? Answer with only the number.\nAnswer:",
                "response": str(answer),
            }
        )
    return rows


def _synthetic_recovery_code_rows(count: int, rng: random.Random) -> list[dict[str, Any]]:
    specs = [
        ("add", "+", "return a + b"),
        ("subtract", "-", "return a - b"),
        ("multiply", "*", "return a * b"),
        ("maximum", "larger of a and b", "return a if a >= b else b"),
    ]
    rows = []
    for index in range(int(count)):
        name, description, body = specs[index % len(specs)]
        suffix = "" if index < len(specs) else f"_{index:03d}"
        function_name = f"{name}{suffix}"
        rows.append(
            {
                "id": f"recovery_code_{index:05d}",
                "category": "code",
                "prompt": (
                    f"Write a tiny Python function named {function_name} that returns the {description}.\n"
                    "```python\n"
                ),
                "response": f"def {function_name}(a, b):\n    {body}\n```",
            }
        )
    return rows


def _synthetic_recovery_template_rows(
    category: str,
    count: int,
    templates: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for index in range(int(count)):
        prompt, response = templates[index % len(templates)]
        rows.append(
            {
                "id": f"recovery_{category}_{index:05d}",
                "category": category,
                "prompt": prompt,
                "response": response,
            }
        )
    return rows


_RECOVERY_TOKENIZER_TEMPLATES = [
    (
        "In one sentence, why can tokenizer mismatch hurt model stitching?\nAnswer:",
        "Tokenizer mismatch can map the same text to different token IDs, so copied embeddings and logits can point at the wrong rows.",
    ),
    (
        "Name one safe way to reduce tokenizer mismatch during cross-family recovery.\nAnswer:",
        "Use aligned text spans, sparse token projections, and byte-level checks so both tokenizers supervise the same surface text.",
    ),
]

_RECOVERY_MODEL_SURGERY_TEMPLATES = [
    (
        "Briefly define neuron merging for model compression.\nAnswer:",
        "Neuron merging combines similar or redundant channels before pruning so the smaller model preserves more of the original function.",
    ),
    (
        "Why should folding happen before pruning in model surgery?\nAnswer:",
        "Folding preserves redundant pathways by merging their contributions, while pruning discards capacity outright.",
    ),
]

_RECOVERY_COMMONSENSE_TEMPLATES = [
    (
        "A driver sees heavy rain on the road. What should they do?\nAnswer:",
        "They should slow down, increase following distance, use headlights and wipers, and pull over if conditions become unsafe.",
    ),
    (
        "A glass falls from a table and shatters. What caused it to break?\nAnswer:",
        "Gravity pulled the glass down, and the impact with the floor broke it into pieces.",
    ),
]


def tie_output_head_checkpoint(
    model_id_or_path: str,
    output_dir: str | Path,
    torch_dtype: str = "float32",
    device: str = "cpu",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Tie a causal LM output head to its input embeddings and save the smaller checkpoint."""

    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    if model.get_input_embeddings() is None or model.get_output_embeddings() is None:
        raise ValueError("Model needs input and output embeddings to tie the output head.")
    before_param_count = sum(p.numel() for p in model.parameters())
    model.config.tie_word_embeddings = True
    model.tie_weights()
    after_param_count = sum(p.numel() for p in model.parameters())
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    report = {
        "operation": "tie_output_head_checkpoint",
        "source_model": model_id_or_path,
        "output_dir": str(target),
        "torch_dtype": torch_dtype,
        "device": device,
        "before_param_count": before_param_count,
        "after_param_count": after_param_count,
        "removed_param_count": before_param_count - after_param_count,
        "tied": bool(model.get_input_embeddings().weight is model.get_output_embeddings().weight),
        "input_embedding_shape": list(model.get_input_embeddings().weight.shape),
        "output_embedding_shape": list(model.get_output_embeddings().weight.shape),
    }
    _write_json(target / "neural_forge_tie_head_report.json", report)
    return report


def gate_text_report(
    baseline_report: str | Path | dict[str, Any],
    candidate_report: str | Path | dict[str, Any],
    output_file: str | Path | None = None,
    max_ppl_ratio: float = 1.25,
    ppl_gate_mode: str = "hard",
    max_accuracy_drop: float = 0.0,
    max_behavior_drop: float = 0.0,
    min_choice_accuracy: float = 0.75,
    min_behavior_pass_rate: float = 0.0,
    min_continuation_chars: int = 24,
    min_unique_token_ratio: float = 0.20,
) -> dict[str, Any]:
    """Compare candidate behavior to a baseline report and produce a gate verdict."""

    if max_ppl_ratio <= 0:
        raise ValueError("max_ppl_ratio must be positive.")
    if ppl_gate_mode not in {"hard", "advisory"}:
        raise ValueError("ppl_gate_mode must be hard or advisory.")
    if max_accuracy_drop < 0:
        raise ValueError("max_accuracy_drop must be non-negative.")
    if max_behavior_drop < 0:
        raise ValueError("max_behavior_drop must be non-negative.")
    if not 0 <= min_behavior_pass_rate <= 1:
        raise ValueError("min_behavior_pass_rate must be between 0 and 1.")
    baseline = _load_report(baseline_report)
    candidate = _load_report(candidate_report)

    base_ppl = float(baseline["perplexity"]["mean"])
    cand_ppl = float(candidate["perplexity"]["mean"])
    base_acc = float(baseline["choice_accuracy"]["accuracy"])
    cand_acc = float(candidate["choice_accuracy"]["accuracy"])
    ppl_ratio = cand_ppl / max(base_ppl, 1e-12)
    generation_checks = [
        _generation_health(item, min_continuation_chars, min_unique_token_ratio)
        for item in candidate.get("generations", [])
    ]
    baseline_generation_checks = [
        _generation_health(item, min_continuation_chars, min_unique_token_ratio)
        for item in baseline.get("generations", [])
    ]

    checks = [
        {
            "name": "perplexity_ratio",
            "passed": ppl_ratio <= max_ppl_ratio,
            "hard": ppl_gate_mode == "hard",
            "value": ppl_ratio,
            "threshold": max_ppl_ratio,
        },
        {
            "name": "choice_accuracy_floor",
            "passed": cand_acc >= min_choice_accuracy,
            "value": cand_acc,
            "threshold": min_choice_accuracy,
        },
        {
            "name": "choice_accuracy_drop",
            "passed": cand_acc >= base_acc - max_accuracy_drop,
            "value": base_acc - cand_acc,
            "threshold": max_accuracy_drop,
        },
        {
            "name": "generation_health",
            "passed": all(item["passed"] for item in generation_checks),
            "value": generation_checks,
            "threshold": {
                "min_continuation_chars": min_continuation_chars,
                "min_unique_token_ratio": min_unique_token_ratio,
            },
        },
    ]
    behavior_suite = candidate.get("behavior_suite")
    baseline_behavior_suite = baseline.get("behavior_suite")
    lost_behavior_ids: list[str] = []
    if behavior_suite is not None or baseline_behavior_suite is not None or min_behavior_pass_rate > 0:
        baseline_behavior_pass_rate = float((baseline_behavior_suite or {}).get("pass_rate", 0.0))
        behavior_pass_rate = float((behavior_suite or {}).get("pass_rate", 0.0))
        baseline_passed_ids = {
            str(item.get("id"))
            for item in (baseline_behavior_suite or {}).get("items", [])
            if item.get("passed") and item.get("id") is not None
        }
        candidate_passed_ids = {
            str(item.get("id"))
            for item in (behavior_suite or {}).get("items", [])
            if item.get("passed") and item.get("id") is not None
        }
        lost_behavior_ids = sorted(baseline_passed_ids - candidate_passed_ids)
        checks.append(
            {
                "name": "behavior_suite",
                "passed": behavior_pass_rate >= min_behavior_pass_rate,
                "value": behavior_pass_rate,
                "threshold": min_behavior_pass_rate,
            }
        )
        checks.append(
            {
                "name": "behavior_suite_drop",
                "passed": behavior_pass_rate >= baseline_behavior_pass_rate - float(max_behavior_drop),
                "value": baseline_behavior_pass_rate - behavior_pass_rate,
                "threshold": float(max_behavior_drop),
                "baseline_pass_rate": baseline_behavior_pass_rate,
                "candidate_pass_rate": behavior_pass_rate,
                "lost_passed_ids": lost_behavior_ids[:50],
                "lost_passed_id_count": len(lost_behavior_ids),
            }
        )
    generation_passed_ids = {
        str(item.get("id") or item.get("prompt"))
        for item in generation_checks
        if item.get("passed") and (item.get("id") is not None or item.get("prompt"))
    }
    baseline_generation_passed_ids = {
        str(item.get("id") or item.get("prompt"))
        for item in baseline_generation_checks
        if item.get("passed") and (item.get("id") is not None or item.get("prompt"))
    }
    lost_generation_ids = sorted(baseline_generation_passed_ids - generation_passed_ids)
    verdict = {
        "operation": "gate_text_report",
        "passed": all(check["passed"] for check in checks if check.get("hard", True)),
        "ppl_gate_mode": ppl_gate_mode,
        "baseline_model": baseline.get("model"),
        "candidate_model": candidate.get("model"),
        "baseline": {
            "param_count": baseline.get("param_count"),
            "perplexity_mean": base_ppl,
            "choice_accuracy": base_acc,
            "generation_health_rate": (
                sum(1 for check in baseline_generation_checks if check["passed"]) / len(baseline_generation_checks)
                if baseline_generation_checks
                else 0.0
            ),
            "behavior_pass_rate": None
            if baseline.get("behavior_suite") is None
            else float(baseline["behavior_suite"].get("pass_rate", 0.0)),
        },
        "candidate": {
            "param_count": candidate.get("param_count"),
            "perplexity_mean": cand_ppl,
            "perplexity_ratio": ppl_ratio,
            "choice_accuracy": cand_acc,
            "generation_health_correct": sum(1 for check in generation_checks if check["passed"]),
            "generation_health_total": len(generation_checks),
            "generation_health_rate": (
                sum(1 for check in generation_checks if check["passed"]) / len(generation_checks)
                if generation_checks
                else 0.0
            ),
            "lost_generation_ids": lost_generation_ids[:50],
            "lost_generation_id_count": len(lost_generation_ids),
            "behavior_pass_rate": None
            if candidate.get("behavior_suite") is None
            else float(candidate["behavior_suite"].get("pass_rate", 0.0)),
            "behavior_correct": None
            if candidate.get("behavior_suite") is None
            else int(candidate["behavior_suite"].get("correct", 0)),
            "behavior_total": None
            if candidate.get("behavior_suite") is None
            else int(candidate["behavior_suite"].get("total", 0)),
            "lost_behavior_ids": lost_behavior_ids[:50],
            "lost_behavior_id_count": len(lost_behavior_ids),
        },
        "checks": checks,
    }
    if output_file is not None:
        _write_json(output_file, verdict)
    return verdict


def _behavior_first_candidate_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float, float, str]:
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


def _eval_item_category(item: dict[str, Any], kind: str) -> str:
    explicit = item.get("category")
    if explicit:
        return _normalize_eval_category(explicit)
    identifier = str(item.get("id") or "").lower()
    prompt = str(item.get("prompt") or "").lower()
    if "python" in identifier or "python" in prompt or "function" in prompt:
        return "code"
    if "math" in identifier or "addition" in identifier or "multiply" in identifier:
        return "math"
    if "tokenizer" in identifier or "tokenizer" in prompt or "vocab" in prompt:
        return "tokenizer"
    if "rain" in identifier or "driver" in prompt or "heavy rain" in prompt:
        return "commonsense"
    if "merge" in identifier or "merging" in prompt or "stitch" in prompt or "compression" in prompt:
        return "model_surgery"
    if "pruning" in identifier or "pruning" in prompt:
        return "model_surgery"
    if kind == "choice":
        return "choice"
    return "other"


def _passed_category_breakdown(items: Sequence[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    breakdown: dict[str, dict[str, Any]] = {}
    for item in items:
        category = _eval_item_category(item, kind)
        row = breakdown.setdefault(
            category,
            {"correct": 0, "total": 0, "pass_rate": 0.0, "failed_ids": [], "passed_ids": []},
        )
        row["total"] += 1
        item_id = item.get("id") or item.get("prompt")
        passed = bool(item.get("passed", item.get("correct", False)))
        if passed:
            row["correct"] += 1
            if item_id is not None and len(row["passed_ids"]) < 20:
                row["passed_ids"].append(str(item_id))
        elif item_id is not None and len(row["failed_ids"]) < 20:
            row["failed_ids"].append(str(item_id))
    for row in breakdown.values():
        row["pass_rate"] = row["correct"] / max(1, row["total"])
    return dict(sorted(breakdown.items()))


def _normalize_eval_category(category: Any) -> str:
    return str(category).strip().lower().replace("-", "_") or "other"


def _category_breakdown_summary(
    breakdown: dict[str, dict[str, Any]],
    hard_categories: Sequence[str] = CRITICAL_EVAL_CATEGORIES,
) -> dict[str, Any]:
    totals = [int(row.get("total", 0) or 0) for row in breakdown.values()]
    rates = [
        float(row.get("pass_rate", 0.0))
        for row in breakdown.values()
        if int(row.get("total", 0) or 0) > 0
    ]
    hard_correct = 0
    hard_total = 0
    hard_breakdown: dict[str, dict[str, Any]] = {}
    for category in hard_categories:
        normalized = _normalize_eval_category(category)
        row = breakdown.get(normalized)
        if row is None:
            continue
        hard_correct += int(row.get("correct", 0) or 0)
        hard_total += int(row.get("total", 0) or 0)
        hard_breakdown[normalized] = {
            "correct": int(row.get("correct", 0) or 0),
            "total": int(row.get("total", 0) or 0),
            "pass_rate": float(row.get("pass_rate", 0.0)),
        }
    return {
        "category_count": len(breakdown),
        "covered_category_count": sum(1 for total in totals if total > 0),
        "min_pass_rate": min(rates) if rates else 0.0,
        "mean_pass_rate": sum(rates) / max(1, len(rates)),
        "hard_categories": [_normalize_eval_category(category) for category in hard_categories],
        "hard_category_correct": int(hard_correct),
        "hard_category_total": int(hard_total),
        "hard_category_pass_rate": hard_correct / max(1, hard_total),
        "hard_category_breakdown": hard_breakdown,
    }


def _eval_suite_signature(report: dict[str, Any]) -> dict[str, Any]:
    suite = report.get("eval_suite") or {}
    return {
        "source": suite.get("source"),
        "text_eval_count": int(suite.get("text_eval_count", 0) or 0),
        "choice_probe_count": int(suite.get("choice_probe_count", 0) or 0),
        "generation_prompt_count": int(suite.get("generation_prompt_count", 0) or 0),
        "behavior_probe_count": int(suite.get("behavior_probe_count", 0) or 0),
    }


def _rank_suite_warnings(signatures: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if not signatures:
        return warnings
    count_keys = ("text_eval_count", "choice_probe_count", "generation_prompt_count", "behavior_probe_count")
    count_sets = {key: sorted({int(sig.get(key, 0) or 0) for sig in signatures}) for key in count_keys}
    mismatched_counts = {key: values for key, values in count_sets.items() if len(values) > 1}
    if mismatched_counts:
        warnings.append({"type": "mixed_eval_suite_counts", "counts": mismatched_counts})
    sources = sorted({str(sig.get("source")) for sig in signatures if sig.get("source") is not None})
    if len(sources) > 1:
        warnings.append({"type": "mixed_eval_suite_sources", "sources": sources[:20], "source_count": len(sources)})
    return warnings


def rank_evaluation_reports(
    reports: Sequence[str | Path | dict[str, Any]],
    output_file: str | Path | None = None,
    min_continuation_chars: int = 24,
    min_unique_token_ratio: float = 0.20,
) -> dict[str, Any]:
    """Rank eval reports with behavior and coherent generations ahead of PPL.

    This is intentionally behavior-first. In high-ratio model surgery sweeps,
    PPL often improves while greedy decoding collapses; ranking by PPL alone
    promotes the wrong checkpoint.
    """

    if not reports:
        raise ValueError("Need at least one evaluation report to rank.")
    if min_continuation_chars < 0:
        raise ValueError("min_continuation_chars must be non-negative.")
    if not 0 <= float(min_unique_token_ratio) <= 1:
        raise ValueError("min_unique_token_ratio must be in [0, 1].")

    entries = []
    suite_signatures = []
    for index, item in enumerate(reports):
        report = _load_report(item)
        suite_signature = _eval_suite_signature(report)
        suite_signatures.append(suite_signature)
        ppl = float(report.get("perplexity", {}).get("mean", 1e300))
        if not math.isfinite(ppl):
            ppl = 1e300
        choice = report.get("choice_accuracy", {})
        choice_accuracy = float(choice.get("accuracy", 0.0))
        behavior = report.get("behavior_suite") or {}
        behavior_pass_rate = float(behavior.get("pass_rate", 0.0))
        generation_checks = [
            _generation_health(gen, min_continuation_chars, min_unique_token_ratio)
            for gen in report.get("generations", [])
        ]
        generation_health_rate = (
            sum(1 for check in generation_checks if check["passed"]) / len(generation_checks)
            if generation_checks
            else 0.0
        )
        generation_health_correct = sum(1 for check in generation_checks if check["passed"])
        behavior_items = list(behavior.get("items", []))
        behavior_breakdown = _passed_category_breakdown(behavior_items, "behavior")
        generation_breakdown = _passed_category_breakdown(generation_checks, "generation")
        behavior_category_summary = _category_breakdown_summary(behavior_breakdown)
        generation_category_summary = _category_breakdown_summary(generation_breakdown)
        param_count = int(report.get("param_count") or 0)
        source = str(item) if not isinstance(item, dict) else f"<dict:{index}>"
        entry = {
            "rank": 0,
            "source": source,
            "model": report.get("model"),
            "param_count": param_count,
            "perplexity_mean": ppl,
            "choice_accuracy": choice_accuracy,
            "choice_correct": int(choice.get("correct", 0)),
            "choice_total": int(choice.get("total", 0)),
            "behavior_pass_rate": behavior_pass_rate,
            "behavior_correct": int(behavior.get("correct", 0)),
            "behavior_total": int(behavior.get("total", 0)),
            "generation_health_rate": generation_health_rate,
            "generation_health_correct": int(generation_health_correct),
            "generation_health_total": len(generation_checks),
            "generation_checks": generation_checks,
            "behavior_category_breakdown": behavior_breakdown,
            "generation_category_breakdown": generation_breakdown,
            "behavior_category_summary": behavior_category_summary,
            "generation_category_summary": generation_category_summary,
            "eval_suite": suite_signature,
            "sort_key": [
                behavior_pass_rate,
                int(behavior.get("correct", 0)),
                int(behavior_category_summary["hard_category_correct"]),
                float(behavior_category_summary["min_pass_rate"]),
                generation_health_rate,
                int(generation_health_correct),
                int(generation_category_summary["hard_category_correct"]),
                float(generation_category_summary["min_pass_rate"]),
                choice_accuracy,
                -param_count,
                -ppl,
            ],
        }
        entries.append(entry)

    entries.sort(
        key=lambda entry: (
            -entry["behavior_pass_rate"],
            -entry["behavior_correct"],
            -entry["behavior_category_summary"]["hard_category_correct"],
            -entry["behavior_category_summary"]["min_pass_rate"],
            -entry["generation_health_rate"],
            -entry["generation_health_correct"],
            -entry["generation_category_summary"]["hard_category_correct"],
            -entry["generation_category_summary"]["min_pass_rate"],
            -entry["choice_accuracy"],
            entry["param_count"],
            entry["perplexity_mean"],
            entry["source"],
        )
    )
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank

    summary_rows = [
        {
            "rank": entry["rank"],
            "source": entry["source"],
            "model": entry["model"],
            "param_count": entry["param_count"],
            "perplexity_mean": entry["perplexity_mean"],
            "choice": f"{entry['choice_correct']}/{entry['choice_total']}",
            "behavior": f"{entry['behavior_correct']}/{entry['behavior_total']}",
            "generation": f"{entry['generation_health_correct']}/{entry['generation_health_total']}",
            "choice_accuracy": entry["choice_accuracy"],
            "behavior_pass_rate": entry["behavior_pass_rate"],
            "generation_health_rate": entry["generation_health_rate"],
            "behavior_category_breakdown": entry["behavior_category_breakdown"],
            "generation_category_breakdown": entry["generation_category_breakdown"],
            "behavior_category_summary": entry["behavior_category_summary"],
            "generation_category_summary": entry["generation_category_summary"],
        }
        for entry in entries
    ]
    report = {
        "operation": "rank_evaluation_reports",
        "ranking_policy": [
            "behavior_pass_rate_desc",
            "behavior_correct_desc",
            "behavior_hard_category_correct_desc",
            "behavior_min_category_pass_rate_desc",
            "generation_health_rate_desc",
            "generation_health_correct_desc",
            "generation_hard_category_correct_desc",
            "generation_min_category_pass_rate_desc",
            "choice_accuracy_desc",
            "param_count_asc",
            "perplexity_mean_asc",
        ],
        "min_continuation_chars": int(min_continuation_chars),
        "min_unique_token_ratio": float(min_unique_token_ratio),
        "suite_warnings": _rank_suite_warnings(suite_signatures),
        "summary_rows": summary_rows,
        "best": entries[0],
        "candidates": entries,
    }
    if output_file is not None:
        _write_json(output_file, report)
    return report


def sweep_gated_ffn_resize(
    source_model: str,
    output_dir: str | Path,
    ratios: Sequence[float],
    baseline_report: str | Path | None = None,
    torch_dtype: str = "float16",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    max_new_tokens: int = 80,
    max_ppl_ratio: float = 1.25,
    max_accuracy_drop: float = 0.0,
    min_choice_accuracy: float = 0.75,
    repair_shrinks: bool = False,
    repair_ridge: float = 1e-3,
    repair_max_tokens: int = 512,
    assignment: str = "combined-cosine",
) -> dict[str, Any]:
    """Resize a source model across ratios, evaluate each candidate, and gate it."""

    if not ratios:
        raise ValueError("At least one ratio is required.")
    root = Path(output_dir)
    model_root = root / "models"
    report_root = root / "reports"
    model_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    if baseline_report is None:
        baseline_path = report_root / "baseline.json"
        baseline = evaluate_text_model(
            source_model,
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
    for ratio in ratios:
        ratio_value = float(ratio)
        tag = _ratio_tag(ratio_value)
        candidate_dir = model_root / f"ffn_{tag}"
        candidate_report_path = report_root / f"ffn_{tag}.json"
        gate_path = report_root / f"ffn_{tag}_gate.json"
        resize_report = resize_gated_ffn(
            source_model,
            candidate_dir,
            ratio=ratio_value,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            assignment=assignment,
        )
        repair_report = None
        eval_dir = candidate_dir
        if repair_shrinks and ratio_value < 1.0:
            repaired_dir = model_root / f"ffn_{tag}_repaired"
            repair_report = repair_gated_ffn_with_calibration(
                source_model=source_model,
                resized_model=str(candidate_dir),
                output_dir=repaired_dir,
                torch_dtype=torch_dtype,
                device=device,
                trust_remote_code=trust_remote_code,
                ridge=repair_ridge,
                max_tokens_per_layer=repair_max_tokens,
            )
            eval_dir = repaired_dir
        candidate_report = evaluate_text_model(
            str(eval_dir),
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
                "ratio": ratio_value,
                "tag": tag,
                "model_dir": str(eval_dir),
                "report": str(candidate_report_path),
                "gate_report": str(gate_path),
                "passed": gate["passed"],
                "resize": resize_report,
                "repair": repair_report,
                "metrics": gate["candidate"],
            }
        )

    summary = {
        "operation": "sweep_gated_ffn_resize",
        "source_model": source_model,
        "baseline_report": str(baseline_path),
        "baseline_metrics": {
            "param_count": baseline.get("param_count"),
            "perplexity_mean": float(baseline["perplexity"]["mean"]),
            "choice_accuracy": float(baseline["choice_accuracy"]["accuracy"]),
        },
        "gate_thresholds": {
            "max_ppl_ratio": max_ppl_ratio,
            "max_accuracy_drop": max_accuracy_drop,
            "min_choice_accuracy": min_choice_accuracy,
        },
        "repair_shrinks": repair_shrinks,
        "repair_ridge": repair_ridge,
        "repair_max_tokens": repair_max_tokens,
        "assignment": assignment,
        "candidates": candidates,
        "passed_candidates": [item for item in candidates if item["passed"]],
    }
    _write_json(report_root / "sweep_summary.json", summary)
    return summary


def search_merge_first_compression(
    source_model: str,
    output_dir: str | Path,
    width_ratios: Sequence[float] | None = None,
    layer_group_specs: Sequence[Sequence[Sequence[int]]] | None = None,
    depth_methods: Sequence[str] = ("average", "fold"),
    baseline_report: str | Path | None = None,
    torch_dtype: str = "float16",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    max_new_tokens: int = 80,
    max_ppl_ratio: float = 1.5,
    max_accuracy_drop: float = 0.25,
    min_choice_accuracy: float = 0.75,
    assignment: str = "combined-cosine",
    include_combinations: bool = True,
    target_shrink: float | None = None,
    eval_suite: str | Path | dict[str, Any] | None = None,
    max_text_evals: int | None = None,
    max_choice_probes: int | None = None,
    max_generation_prompts: int | None = None,
    max_behavior_probes: int | None = None,
) -> dict[str, Any]:
    """Search merge-first width/depth compression candidates and gate each one."""

    ratios = [float(ratio) for ratio in (width_ratios or [])]
    group_specs = [_normalize_layer_groups(spec) for spec in (layer_group_specs or [])]
    methods = [str(method).lower() for method in depth_methods]
    if not ratios and not group_specs:
        raise ValueError("Provide at least one width ratio or one layer group spec.")
    for ratio in ratios:
        if ratio <= 0:
            raise ValueError("width ratios must be positive.")
    for method in methods:
        if method not in {"average", "fold"}:
            raise ValueError("depth methods must be average or fold.")
    if target_shrink is not None and target_shrink < 0:
        raise ValueError("target_shrink must be non-negative.")

    root = Path(output_dir)
    model_root = root / "models"
    report_root = root / "reports"
    model_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    if baseline_report is None:
        baseline_path = report_root / "baseline.json"
        baseline = evaluate_text_model(
            source_model,
            baseline_path,
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
    else:
        baseline_path = Path(baseline_report)
        baseline = _load_report(baseline_path)
    baseline_params = int(baseline.get("param_count") or 0)

    candidates = []
    width_sources: list[dict[str, Any]] = []

    def evaluate_candidate(kind: str, tag: str, model_dir: Path, surgery_report: dict[str, Any]) -> dict[str, Any]:
        report_path = report_root / f"{tag}.json"
        gate_path = report_root / f"{tag}_gate.json"
        candidate_report = evaluate_text_model(
            str(model_dir),
            report_path,
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
        gate = gate_text_report(
            baseline,
            candidate_report,
            output_file=gate_path,
            max_ppl_ratio=max_ppl_ratio,
            ppl_gate_mode="advisory",
            max_accuracy_drop=max_accuracy_drop,
            min_choice_accuracy=min_choice_accuracy,
        )
        candidate_params = int(candidate_report.get("param_count") or 0)
        shrink = (baseline_params - candidate_params) / baseline_params if baseline_params else 0.0
        item = {
            "kind": kind,
            "tag": tag,
            "model_dir": str(model_dir),
            "report": str(report_path),
            "gate_report": str(gate_path),
            "passed": gate["passed"],
            "surgery": surgery_report,
            "metrics": gate["candidate"],
            "shrink": shrink,
        }
        candidates.append(item)
        return item

    for ratio in ratios:
        ratio_tag = _ratio_tag(ratio)
        tag = f"width_{ratio_tag}"
        candidate_dir = model_root / tag
        surgery_report = resize_gated_ffn(
            source_model,
            candidate_dir,
            ratio=ratio,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            assignment=assignment,
        )
        item = evaluate_candidate("width", tag, candidate_dir, surgery_report)
        width_sources.append({"ratio": ratio, "tag": tag, "model_dir": candidate_dir, "candidate": item})

    for spec in group_specs:
        groups_tag = _layer_groups_tag(spec)
        for method in methods:
            tag = f"depth_{method}_{groups_tag}"
            candidate_dir = model_root / tag
            surgery_report = merge_decoder_layers(
                source_model,
                candidate_dir,
                groups=spec,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                method=method,
            )
            evaluate_candidate("depth", tag, candidate_dir, surgery_report)

    if include_combinations:
        for width in width_sources:
            for spec in group_specs:
                groups_tag = _layer_groups_tag(spec)
                for method in methods:
                    tag = f"{width['tag']}_depth_{method}_{groups_tag}"
                    candidate_dir = model_root / tag
                    surgery_report = merge_decoder_layers(
                        str(width["model_dir"]),
                        candidate_dir,
                        groups=spec,
                        torch_dtype=torch_dtype,
                        trust_remote_code=trust_remote_code,
                        method=method,
                    )
                    combo_report = {
                        "operation": "width_then_depth_merge",
                        "width": width["candidate"]["surgery"],
                        "depth": surgery_report,
                    }
                    evaluate_candidate("width_depth", tag, candidate_dir, combo_report)

    passed = sorted([item for item in candidates if item["passed"]], key=_behavior_first_candidate_sort_key)
    if target_shrink is None:
        target_passed = []
        best_at_or_below_target = None
    else:
        target_passed = [item for item in passed if item["shrink"] >= target_shrink]
        below = [item for item in passed if item["shrink"] < target_shrink]
        best_at_or_below_target = below[0] if below else None

    summary = {
        "operation": "search_merge_first_compression",
        "source_model": source_model,
        "baseline_report": str(baseline_path),
        "baseline_metrics": {
            "param_count": baseline.get("param_count"),
            "perplexity_mean": float(baseline["perplexity"]["mean"]),
            "choice_accuracy": float(baseline["choice_accuracy"]["accuracy"]),
        },
        "width_ratios": ratios,
        "layer_group_specs": [_layer_groups_for_report(spec) for spec in group_specs],
        "depth_methods": methods,
        "assignment": assignment,
        "include_combinations": include_combinations,
        "target_shrink": target_shrink,
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
        "passed_candidates": passed,
        "target_passed_candidates": target_passed,
        "best_passed_candidate": passed[0] if passed else None,
        "best_passed_below_target": best_at_or_below_target,
    }
    _write_json(report_root / "merge_first_search_summary.json", summary)
    return summary


def sweep_task_arithmetic(
    base_model: str,
    tuned_model: str,
    output_dir: str | Path,
    alphas: Sequence[float],
    baseline_report: str | Path | None = None,
    torch_dtype: str = "auto",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    max_new_tokens: int = 80,
    max_ppl_ratio: float = 2.0,
    max_accuracy_drop: float = 0.25,
    min_choice_accuracy: float = 0.75,
) -> dict[str, Any]:
    """Sweep task-arithmetic alphas, evaluate each candidate, and gate results."""

    if not alphas:
        raise ValueError("At least one alpha is required.")
    alpha_values = [_finite_float(alpha, "alpha") for alpha in alphas]
    _ensure_unique_ratio_tags(alpha_values, "alpha")
    root = Path(output_dir)
    model_root = root / "models"
    report_root = root / "reports"
    model_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    if baseline_report is None:
        baseline_path = report_root / "baseline_base.json"
        baseline = evaluate_text_model(
            base_model,
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
        candidate_dir = model_root / f"task_alpha_{tag}"
        candidate_report_path = report_root / f"task_alpha_{tag}.json"
        gate_path = report_root / f"task_alpha_{tag}_gate.json"
        merge_report = task_arithmetic_merge(
            base_model=base_model,
            tuned_model=tuned_model,
            output_dir=candidate_dir,
            alpha=alpha_value,
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
        "operation": "sweep_task_arithmetic",
        "base_model": base_model,
        "tuned_model": tuned_model,
        "baseline_report": str(baseline_path),
        "baseline_metrics": {
            "param_count": baseline.get("param_count"),
            "perplexity_mean": float(baseline["perplexity"]["mean"]),
            "choice_accuracy": float(baseline["choice_accuracy"]["accuracy"]),
        },
        "gate_thresholds": {
            "max_ppl_ratio": max_ppl_ratio,
            "max_accuracy_drop": max_accuracy_drop,
            "min_choice_accuracy": min_choice_accuracy,
        },
        "candidates": candidates,
        "passed_candidates": [item for item in candidates if item["passed"]],
    }
    _write_json(report_root / "task_sweep_summary.json", summary)
    return summary


def vision_probe(
    model_id_or_path: str,
    output_file: str | Path,
    image_dir: str | Path,
    torch_dtype: str = "float16",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    max_new_tokens: int = 80,
) -> dict[str, Any]:
    torch, transformers = _torch_and_transformers()
    from PIL import Image, ImageDraw

    image_root = Path(image_dir)
    image_root.mkdir(parents=True, exist_ok=True)
    images = _make_probe_images(Image, ImageDraw, image_root)
    processor = transformers.AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
    model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForVision2Seq")
    model = model_cls.from_pretrained(
        model_id_or_path,
        torch_dtype=_dtype_from_name(torch, torch_dtype),
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()

    prompts = [
        ("shapes", images["shapes"], "Describe the image in one sentence."),
        ("ocr", images["ocr"], "What text or number is visible in this image?"),
    ]
    outputs = []
    with torch.no_grad():
        for probe_id, image_path, question in prompts:
            image = Image.open(image_path).convert("RGB")
            text = _apply_vision_template(processor, question)
            inputs = processor(text=text, images=image, return_tensors="pt")
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
            outputs.append({"id": probe_id, "image": str(image_path), "question": question, "output": decoded})

    report = {
        "model": model_id_or_path,
        "param_count": sum(p.numel() for p in model.parameters()),
        "outputs": outputs,
    }
    _write_json(output_file, report)
    return report


def vision_text_chain(
    vision_model_id_or_path: str,
    text_model_id_or_path: str,
    output_file: str | Path,
    image_dir: str | Path,
    torch_dtype: str = "float16",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    max_new_tokens: int = 80,
) -> dict[str, Any]:
    torch, transformers = _torch_and_transformers()
    from PIL import Image, ImageDraw

    image_root = Path(image_dir)
    image_root.mkdir(parents=True, exist_ok=True)
    images = _make_probe_images(Image, ImageDraw, image_root)

    processor = transformers.AutoProcessor.from_pretrained(vision_model_id_or_path, trust_remote_code=trust_remote_code)
    model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForVision2Seq")
    vision_model = model_cls.from_pretrained(
        vision_model_id_or_path,
        torch_dtype=_dtype_from_name(torch, torch_dtype),
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    vision_model.to(device)
    vision_model.eval()
    vision_param_count = sum(p.numel() for p in vision_model.parameters())

    observations = []
    with torch.no_grad():
        for probe_id, image_path, question in [
            ("shapes", images["shapes"], "Describe the image in one sentence."),
            ("ocr", images["ocr"], "What text or number is visible in this image?"),
        ]:
            image = Image.open(image_path).convert("RGB")
            text = _apply_vision_template(processor, question)
            inputs = processor(text=text, images=image, return_tensors="pt")
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            generated = vision_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
            observations.append({"id": probe_id, "image": str(image_path), "question": question, "observation": decoded})

    del vision_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = transformers.AutoTokenizer.from_pretrained(text_model_id_or_path, trust_remote_code=trust_remote_code)
    text_model = transformers.AutoModelForCausalLM.from_pretrained(
        text_model_id_or_path,
        torch_dtype=_dtype_from_name(torch, torch_dtype),
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    text_model.to(device)
    text_model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    answers = []
    with torch.no_grad():
        for item in observations:
            prompt = (
                "You are validating a stitched vision-language system.\n"
                f"Vision observation: {item['observation']}\n"
                f"Original question: {item['question']}\n"
                "Answer concisely:"
            )
            answers.append(
                {
                    **item,
                    "text_prompt": prompt,
                    "text_answer": _generate_text(torch, text_model, tokenizer, prompt, device, max_new_tokens)["output"],
                }
            )

    report = {
        "operation": "vision_text_chain",
        "vision_model": vision_model_id_or_path,
        "text_model": text_model_id_or_path,
        "vision_param_count": vision_param_count,
        "text_param_count": sum(p.numel() for p in text_model.parameters()),
        "outputs": answers,
    }
    _write_json(output_file, report)
    return report


def _torch_and_transformers():
    import torch
    import transformers

    return torch, transformers


def _dtype_from_name(torch, name: str):
    lowered = name.lower()
    if lowered in {"auto", "source"}:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if lowered not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[lowered]


def _normalized_weights(count: int, weights: Sequence[float] | None) -> list[float]:
    if weights is None:
        return [1.0 / count] * count
    if len(weights) != count:
        raise ValueError(f"Expected {count} weights, got {len(weights)}.")
    values = [float(weight) for weight in weights]
    total = sum(values)
    if total == 0:
        raise ValueError("Merge weights must not sum to zero.")
    return [weight / total for weight in values]


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _ensure_unique_ratio_tags(values: Sequence[float], name: str) -> None:
    seen: dict[str, float] = {}
    for value in values:
        tag = _ratio_tag(value)
        if tag in seen:
            raise ValueError(f"Duplicate {name} output tag {tag!r} for values {seen[tag]} and {value}.")
        seen[tag] = value


def _assert_tokenizers_compatible(base_tokenizer, tuned_tokenizer) -> dict[str, Any]:
    try:
        base_len = len(base_tokenizer)
        tuned_len = len(tuned_tokenizer)
    except TypeError:
        base_len = tuned_len = None
    if base_len is not None and base_len != tuned_len:
        raise ValueError(f"Tokenizer length mismatch: base={base_len} tuned={tuned_len}")

    special_tokens_match = getattr(base_tokenizer, "special_tokens_map", None) == getattr(tuned_tokenizer, "special_tokens_map", None)

    vocab_match = None
    if hasattr(base_tokenizer, "get_vocab") and hasattr(tuned_tokenizer, "get_vocab"):
        vocab_match = base_tokenizer.get_vocab() == tuned_tokenizer.get_vocab()
        if not vocab_match:
            raise ValueError("Tokenizer vocabularies differ; task merge requires compatible tokenization.")

    return {
        "length": base_len,
        "vocab_match": vocab_match,
        "special_tokens_match": special_tokens_match,
    }


def _load_report(report: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(report, dict):
        return report
    return json.loads(Path(report).read_text(encoding="utf-8"))


def _load_eval_suite(suite: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if suite is None:
        return _normalize_eval_suite(
            {
                "text_evals": list(TEXT_EVALS),
                "choice_probes": list(CHOICE_PROBES),
                "generation_prompts": [
                    {
                        "id": f"builtin_generation_{index}",
                        "prompt": prompt,
                        "relevance_rules": GENERATION_RELEVANCE_RULES.get(prompt, []),
                    }
                    for index, prompt in enumerate(GENERATION_PROMPTS)
                ],
                "behavior_probes": list(BEHAVIOR_PROBES),
            },
            source="builtin",
        )
    if isinstance(suite, dict):
        source = str(suite.get("output_file") or suite.get("source") or "<dict>")
        return _normalize_eval_suite(suite.get("suite", suite), source=source)

    path = Path(suite)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        data: dict[str, Any] = {"text_evals": [], "choice_probes": [], "generation_prompts": [], "behavior_probes": []}
        for row in rows:
            kind = str(row.get("type") or row.get("kind") or "").lower()
            if kind in {"text", "text_eval", "ppl"} or ("text" in row and not row.get("prompt")):
                data["text_evals"].append(row.get("text", row.get("value", "")))
            elif kind in {"choice", "choice_probe", "multiple_choice"} or ("choices" in row and "answer" in row):
                data["choice_probes"].append(row)
            elif kind in {"generation", "generation_prompt"} or ("relevance_rules" in row and "prompt" in row):
                data["generation_prompts"].append(row)
            elif "prompt" in row and "kind" in row:
                data["behavior_probes"].append(row)
            else:
                raise ValueError(f"Cannot infer eval-suite row type for {row!r}.")
        return _normalize_eval_suite(data, source=str(path))

    loaded = json.loads(text)
    if isinstance(loaded, list):
        return _load_eval_suite({"suite": _jsonl_rows_to_suite(loaded), "source": str(path)})
    if not isinstance(loaded, dict):
        raise ValueError("Eval suite JSON must be an object or list.")
    return _normalize_eval_suite(loaded.get("suite", loaded), source=str(path))


def _jsonl_rows_to_suite(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {"text_evals": [], "choice_probes": [], "generation_prompts": [], "behavior_probes": []}
    for row in rows:
        kind = str(row.get("type") or row.get("kind") or "").lower()
        if kind in {"text", "text_eval", "ppl"} or ("text" in row and not row.get("prompt")):
            data["text_evals"].append(row.get("text", row.get("value", "")))
        elif kind in {"choice", "choice_probe", "multiple_choice"} or ("choices" in row and "answer" in row):
            data["choice_probes"].append(row)
        elif kind in {"generation", "generation_prompt"} or ("relevance_rules" in row and "prompt" in row):
            data["generation_prompts"].append(row)
        elif "prompt" in row and "kind" in row:
            data["behavior_probes"].append(row)
        else:
            raise ValueError(f"Cannot infer eval-suite row type for {row!r}.")
    return data


def _normalize_eval_suite(data: dict[str, Any], source: str) -> dict[str, Any]:
    raw_text_evals = data.get("text_evals")
    raw_choice_probes = data.get("choice_probes")
    raw_generation_prompts = data.get("generation_prompts")
    raw_behavior_probes = data.get("behavior_probes")
    text_evals = [
        str(item.get("text", item)) if isinstance(item, dict) else str(item)
        for item in (raw_text_evals or [])
    ]
    choice_probes = [
        _normalize_choice_probe(item, index)
        for index, item in enumerate(raw_choice_probes or [])
    ]
    generation_prompts = [
        _normalize_generation_prompt(item, index)
        for index, item in enumerate(raw_generation_prompts or [])
    ]
    behavior_probes = [
        _normalize_behavior_probe(item, index)
        for index, item in enumerate(raw_behavior_probes or [])
    ]
    if raw_text_evals is None:
        text_evals = list(TEXT_EVALS)
    if raw_choice_probes is None:
        choice_probes = list(CHOICE_PROBES)
    if raw_generation_prompts is None:
        generation_prompts = [
            {
                "id": f"builtin_generation_{index}",
                "prompt": prompt,
                "relevance_rules": GENERATION_RELEVANCE_RULES.get(prompt, []),
            }
            for index, prompt in enumerate(GENERATION_PROMPTS)
        ]
    if raw_behavior_probes is None:
        behavior_probes = list(BEHAVIOR_PROBES)
    return {
        "source": source,
        "text_evals": text_evals,
        "choice_probes": choice_probes,
        "generation_prompts": generation_prompts,
        "behavior_probes": behavior_probes,
    }


def _normalize_choice_probe(item: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Choice probes must be objects.")
    probe = dict(item)
    probe.setdefault("id", f"choice_{index:04d}")
    if "prompt" not in probe or "choices" not in probe or "answer" not in probe:
        raise ValueError("Choice probe requires prompt, choices, and answer.")
    probe["choices"] = [str(choice) for choice in probe["choices"]]
    probe["answer"] = int(probe["answer"])
    return probe


def _normalize_generation_prompt(item: str | dict[str, Any], index: int) -> dict[str, Any]:
    if isinstance(item, str):
        return {
            "id": f"generation_{index:04d}",
            "prompt": item,
            "relevance_rules": GENERATION_RELEVANCE_RULES.get(item, []),
        }
    if not isinstance(item, dict):
        raise ValueError("Generation prompts must be strings or objects.")
    prompt = str(item.get("prompt", ""))
    if not prompt:
        raise ValueError("Generation prompt requires prompt text.")
    normalized = {
        "id": str(item.get("id") or f"generation_{index:04d}"),
        "prompt": prompt,
        "relevance_rules": item.get("relevance_rules", item.get("pattern_groups", GENERATION_RELEVANCE_RULES.get(prompt, []))),
    }
    if item.get("category") is not None:
        normalized["category"] = str(item.get("category"))
    for key in ("kind", "answer", "function_name", "tests", "pattern_groups"):
        if key in item:
            normalized[key] = item[key]
    return normalized


def _normalize_behavior_probe(item: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Behavior probes must be objects.")
    probe = dict(item)
    probe.setdefault("id", f"behavior_{index:04d}")
    if "prompt" not in probe or "kind" not in probe:
        raise ValueError("Behavior probe requires prompt and kind.")
    return probe


def _limit_sequence(items: Sequence[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return list(items)
    if limit < 0:
        raise ValueError("suite limits must be non-negative.")
    return list(items)[: int(limit)]


def _generation_health(item: dict[str, Any], min_continuation_chars: int, min_unique_token_ratio: float) -> dict[str, Any]:
    prompt = str(item.get("prompt", ""))
    output = str(item.get("output", ""))
    continuation = output[len(prompt) :] if prompt and output.startswith(prompt) else output
    continuation = continuation.strip()
    diagnostics = item.get("diagnostics") or _generation_diagnostics(continuation)
    tokens = re.findall(r"[A-Za-z0-9_]+", continuation.lower())
    unique_ratio = float(diagnostics.get("distinct_1", 0.0))
    long_enough = len(continuation) >= min_continuation_chars
    diverse_enough = (
        len(tokens) < 20
        or (
            unique_ratio >= min_unique_token_ratio
            and float(diagnostics.get("top_token_share", 1.0)) <= 0.60
            and int(diagnostics.get("max_token_run", 999)) <= 12
        )
    )
    relevance = _generation_relevance(prompt, continuation, item.get("relevance_rules"))
    expected = _generation_expected_behavior(item)
    expectation_passed = True if expected is None else bool(expected["passed"])
    return {
        "id": item.get("id"),
        "category": item.get("category"),
        "prompt": prompt,
        "passed": long_enough and diverse_enough and relevance["passed"] and expectation_passed,
        "continuation_chars": len(continuation),
        "token_count": len(tokens),
        "unique_token_ratio": unique_ratio,
        "diagnostics": diagnostics,
        "long_enough": long_enough,
        "diverse_enough": diverse_enough,
        "relevant_enough": relevance["passed"],
        "expected_enough": expectation_passed,
        "expected_behavior": expected,
        "relevance": relevance,
    }


def _generation_expected_behavior(item: dict[str, Any]) -> dict[str, Any] | None:
    if item.get("kind") is None:
        return None
    probe = {
        "id": str(item.get("id") or item.get("prompt") or "generation_expected"),
        "category": item.get("category"),
        "prompt": str(item.get("prompt", "")),
        "kind": str(item.get("kind")),
    }
    for key in ("answer", "function_name", "tests", "pattern_groups"):
        if key in item:
            probe[key] = item[key]
    try:
        scored = _score_behavior_probe(probe, item)
    except ValueError as exc:
        return {"checked": True, "passed": False, "reason": str(exc)}
    return {
        "checked": True,
        "passed": bool(scored.get("passed")),
        "kind": probe["kind"],
        "details": scored.get("details", {}),
    }


def _generation_relevance(prompt: str, continuation: str, relevance_rules: Sequence[Sequence[str]] | None = None) -> dict[str, Any]:
    pattern_groups = relevance_rules if relevance_rules is not None else GENERATION_RELEVANCE_RULES.get(prompt)
    if not pattern_groups:
        return {"checked": False, "passed": True, "matched_groups": [], "missing_groups": []}

    matched_groups = []
    missing_groups = []
    text = continuation.lower()
    for group_index, patterns in enumerate(pattern_groups):
        matched = [pattern for pattern in patterns if re.search(pattern, text)]
        group_result = {
            "group": group_index,
            "matched_patterns": matched,
            "patterns": patterns,
        }
        if matched:
            matched_groups.append(group_result)
        else:
            missing_groups.append(group_result)

    return {
        "checked": True,
        "passed": not missing_groups,
        "matched_groups": matched_groups,
        "missing_groups": missing_groups,
    }


def _generation_diagnostics(continuation: str, generated_token_ids: Sequence[int] | None = None) -> dict[str, Any]:
    tokens = re.findall(r"[A-Za-z0-9_]+", str(continuation).lower())
    bigrams = list(zip(tokens, tokens[1:]))
    repeated_bigrams = len(bigrams) - len(set(bigrams)) if bigrams else 0
    token_counts = {token: tokens.count(token) for token in set(tokens)}
    top_count = max(token_counts.values()) if token_counts else 0
    token_ids = [int(item) for item in (generated_token_ids or [])]
    return {
        "word_token_count": len(tokens),
        "distinct_1": (len(set(tokens)) / len(tokens)) if tokens else 0.0,
        "distinct_2": (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0,
        "top_token_share": (top_count / len(tokens)) if tokens else 0.0,
        "max_token_run": _max_run_length(tokens),
        "repeated_bigram_share": (repeated_bigrams / len(bigrams)) if bigrams else 0.0,
        "generated_token_count": len(token_ids),
        "generated_distinct_token_ratio": (len(set(token_ids)) / len(token_ids)) if token_ids else 0.0,
        "generated_max_token_id_run": _max_run_length(token_ids),
    }


def _max_run_length(items: Sequence[Any]) -> int:
    max_run = 0
    current_run = 0
    previous = object()
    for item in items:
        if item == previous:
            current_run += 1
        else:
            previous = item
            current_run = 1
        max_run = max(max_run, current_run)
    return max_run


def _ratio_tag(ratio: float) -> str:
    return f"{ratio:.4f}".replace(".", "p").replace("-", "m")


def _synthetic_text_evals(count: int) -> list[str]:
    templates = [
        "Neural model compression should preserve useful behavior, not only lower perplexity.",
        "A folded child model needs calibration data that covers arithmetic, code, and instruction following.",
        "Tokenizer mismatch can make copied embedding rows point at the wrong symbols.",
        "Merge-first pruning combines redundant channels before deleting capacity.",
        "A safe driving answer should mention slowing down and increasing following distance.",
        "Small language models are useful when they answer concisely and avoid repetitive loops.",
        "Cross-family model surgery needs behavior gates because local activation losses can be misleading.",
        "A Python function should use valid syntax and return the requested value.",
    ]
    return [templates[index % len(templates)] for index in range(int(count))]


def _synthetic_choice_probes(count: int, rng: random.Random) -> list[dict[str, Any]]:
    probes = []
    for index in range(int(count)):
        mode = index % 4
        if mode == 0:
            a = 7 + (index * 3) % 43
            b = 5 + (index * 5) % 37
            answer = a + b
            prompt = f"Question: What is {a} + {b}?\nAnswer:"
            choices = [f" {answer}", f" {answer + 1}", f" {answer - 1}", f" {a * b}"]
            answer_index = 0
        elif mode == 1:
            a = 3 + (index * 2) % 19
            b = 2 + (index * 7) % 13
            answer = a * b
            prompt = f"Question: What is {a} * {b}?\nAnswer:"
            choices = [f" {answer}", f" {answer + a}", f" {answer - b}", f" {a + b}"]
            answer_index = 0
        elif mode == 2:
            prompt = (
                "Question: Which Python expression returns the sum of a and b?\n"
                "A. a + b\nB. a - b\nC. len(a)\nD. print(a)\nAnswer:"
            )
            choices = [" A", " B", " C", " D"]
            answer_index = 0
        else:
            prompt = (
                "Question: A road is covered in heavy rain. What should a driver do?\n"
                "A. speed up\nB. slow down and keep distance\nC. close their eyes\nD. turn off headlights\nAnswer:"
            )
            choices = [" A", " B", " C", " D"]
            answer_index = 1
        indexed = list(enumerate(choices))
        rng.shuffle(indexed)
        shuffled_choices = [choice for _, choice in indexed]
        shuffled_answer = next(pos for pos, (old_index, _) in enumerate(indexed) if old_index == answer_index)
        probes.append(
            {
                "id": f"synthetic_choice_{index:04d}",
                "prompt": prompt,
                "choices": shuffled_choices,
                "answer": shuffled_answer,
            }
        )
    return probes


def _synthetic_generation_prompts(count: int) -> list[dict[str, Any]]:
    templates = [
        (
            "generation_pruning",
            "Explain neural pruning in two concise sentences.",
            "model_surgery",
            [
                [r"\bprun\w*", r"\bremove\w*", r"\bcompress\w*", r"\bfold\w*"],
                [r"\bneuron\w*", r"\bweight\w*", r"\bchannel\w*", r"\bparameter\w*"],
            ],
        ),
        (
            "generation_model_merge_risk",
            "Give one risk when merging two language models.",
            "model_surgery",
            [
                [r"\bmerge\w*", r"\bstitch\w*", r"\bcombin\w*", r"\bfus\w*"],
                [r"\brisk\w*", r"\bmismatch\w*", r"\bforget\w*", r"\bdegrad\w*", r"\bcollapse\w*"],
                [r"\bmodel\w*", r"\btokeni[sz]er\w*", r"\bweight\w*", r"\bcapabilit\w*"],
            ],
        ),
        (
            "generation_python_add",
            "Write a tiny Python function that adds two numbers.",
            "code",
            [[r"\bdef\b", r"\blambda\b"], [r"\+", r"\badd\w*\b", r"\bsum\b"], [r"\breturn\b", r":"]],
        ),
        (
            "generation_tokenizer_mismatch",
            "In one sentence, why can tokenizer mismatch hurt model stitching?",
            "tokenizer",
            [
                [r"\btokeni[sz]er\w*", r"\bvocab\w*", r"\btokens?\b"],
                [r"\bmismatch\w*", r"\bdiffer\w*", r"\balign\w*", r"\bincompat\w*"],
                [r"\bembedding\w*", r"\blogits?\b", r"\brows?\b", r"\bids?\b"],
            ],
        ),
        (
            "generation_rain_safety",
            "A driver sees heavy rain on the road. What should they do?",
            "commonsense",
            [
                [r"\bslow\w*", r"\bcaution\w*", r"\bsafe\w*", r"\breduce\w*"],
                [r"\bdistance\b", r"\bheadlights?\b", r"\bwipers?\b", r"\bpull over\b"],
            ],
        ),
    ]
    prompts = []
    for index in range(int(count)):
        mode = index % 7
        if mode == 5:
            a = 6 + (index * 7) % 41
            b = 4 + (index * 11) % 29
            op = "+" if index % 14 == 5 else "*"
            answer = a + b if op == "+" else a * b
            prompts.append(
                {
                    "id": f"generation_math_answer_{index:04d}",
                    "category": "math",
                    "prompt": f"Question: What is {a} {op} {b}? Answer with only the number.\nAnswer:",
                    "kind": "first_number",
                    "answer": str(answer),
                    "relevance_rules": [],
                }
            )
            continue
        if mode == 6:
            prompts.append(
                {
                    "id": f"generation_python_add_exact_{index:04d}",
                    "category": "code",
                    "prompt": (
                        "Write a tiny Python function named add that returns a + b.\n"
                        "```python\n"
                    ),
                    "kind": "python_function",
                    "function_name": "add",
                    "tests": [
                        {"args": [2, 3], "expected": 5},
                        {"args": [-1, 5], "expected": 4},
                        {"args": [0, 0], "expected": 0},
                    ],
                    "relevance_rules": [[r"\bdef\b", r"\blambda\b"], [r"\+", r"\badd\w*\b"], [r"\breturn\b", r":"]],
                }
            )
            continue
        base_id, prompt, category, rules = templates[mode % len(templates)]
        prompts.append(
            {
                "id": f"{base_id}_{index:04d}",
                "category": category,
                "prompt": f"{prompt}\nVariation id: {index:04d}.",
                "relevance_rules": rules,
            }
        )
    return prompts


def _synthetic_behavior_probes(count: int, rng: random.Random) -> list[dict[str, Any]]:
    probes = []
    for index in range(int(count)):
        mode = index % 5
        if mode == 0:
            a = 6 + (index * 7) % 41
            b = 4 + (index * 11) % 29
            op = "+" if index % 10 == 0 else "*"
            answer = a + b if op == "+" else a * b
            probes.append(
                {
                    "id": f"synthetic_math_{index:04d}",
                    "category": "math",
                    "prompt": f"Question: What is {a} {op} {b}? Answer with only the number.\nAnswer:",
                    "kind": "first_number",
                    "answer": str(answer),
                }
            )
        elif mode == 1:
            probes.append(
                {
                    "id": f"synthetic_tokenizer_{index:04d}",
                    "category": "tokenizer",
                    "prompt": "In one sentence, why can tokenizer mismatch hurt model stitching?\nAnswer:",
                    "kind": "pattern_groups",
                    "pattern_groups": [
                        [r"\btokeni[sz]er\w*", r"\bvocab\w*", r"\btokens?\b"],
                        [r"\bmismatch\w*", r"\bdiffer\w*", r"\balign\w*", r"\bincompat\w*"],
                        [r"\bembedding\w*", r"\blogits?\b", r"\btoken ids?\b", r"\brows?\b"],
                    ],
                }
            )
        elif mode == 2:
            probes.append(
                {
                    "id": f"synthetic_rain_{index:04d}",
                    "category": "commonsense",
                    "prompt": "A driver sees heavy rain on the road. What should they do?\nAnswer:",
                    "kind": "pattern_groups",
                    "pattern_groups": [
                        [r"\bslow\w*", r"\bcaution\w*", r"\bsafe\w*", r"\breduce\w*"],
                        [r"\bdistance\b", r"\bfollow\w*", r"\bheadlights?\b", r"\bwipers?\b", r"\bpull over\b"],
                    ],
                }
            )
        elif mode == 3:
            probes.append(
                {
                    "id": f"synthetic_merge_definition_{index:04d}",
                    "category": "model_surgery",
                    "prompt": "Briefly define neuron merging for model compression.\nAnswer:",
                    "kind": "pattern_groups",
                    "pattern_groups": [
                        [r"\bmerg\w*", r"\bfold\w*", r"\bcombin\w*"],
                        [r"\bneurons?\b", r"\bchannels?\b", r"\bfilters?\b"],
                        [r"\bsimilar\w*", r"\bredundan\w*", r"\baverage\w*", r"\bsum\w*"],
                    ],
                }
            )
        else:
            fn_specs = [
                ("add", "+", lambda a, b: a + b),
                ("subtract", "-", lambda a, b: a - b),
                ("multiply", "*", lambda a, b: a * b),
            ]
            function_name, symbol, func = fn_specs[(index // 5) % len(fn_specs)]
            tests = []
            for case in range(3):
                a = rng.randint(-9, 12)
                b = rng.randint(-7, 11)
                tests.append({"args": [a, b], "expected": func(a, b)})
            probes.append(
                {
                    "id": f"synthetic_python_{function_name}_{index:04d}",
                    "category": "code",
                    "prompt": (
                        f"Write a tiny Python function named {function_name} that returns a {symbol} b.\n"
                        "```python\n"
                    ),
                    "kind": "python_function",
                    "function_name": function_name,
                    "tests": tests,
                }
            )
    return probes


def _default_calibration_texts() -> list[str]:
    texts = list(TEXT_EVALS)
    texts.extend(GENERATION_PROMPTS)
    texts.extend(probe["prompt"] + probe["choices"][probe["answer"]] for probe in CHOICE_PROBES)
    texts.extend(
        [
            "A small repaired model should keep the parent model's local activations close.",
            "Calibration repair can reduce the damage from folding redundant feed-forward channels.",
        ]
    )
    return texts


def _encode_calibration_batches(tokenizer, texts: Sequence[str], device: str, max_length: int) -> list[dict[str, Any]]:
    batches = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        batches.append({key: value.to(device) for key, value in encoded.items()})
    return batches


def _collect_mlp_calibration(torch, model, source_mlp, encoded_batches: Sequence[dict[str, Any]], max_tokens: int):
    collected_inputs = []
    collected_outputs = []

    def hook(_module, inputs, output):
        collected_inputs.append(inputs[0].detach().cpu().to(torch.float32))
        collected_outputs.append(output.detach().cpu().to(torch.float32))

    handle = source_mlp.register_forward_hook(hook)
    try:
        for batch in encoded_batches:
            model(**batch, use_cache=False)
            if sum(tensor.numel() // tensor.shape[-1] for tensor in collected_inputs) >= max_tokens:
                break
    finally:
        handle.remove()

    if not collected_inputs:
        raise RuntimeError("No calibration activations were collected.")
    inputs = torch.cat([item.reshape(-1, item.shape[-1]) for item in collected_inputs], dim=0)
    outputs = torch.cat([item.reshape(-1, item.shape[-1]) for item in collected_outputs], dim=0)
    if inputs.shape[0] > max_tokens:
        inputs = inputs[:max_tokens]
        outputs = outputs[:max_tokens]
    return inputs, outputs


def _gated_mlp_hidden(torch, mlp, inputs):
    activation = getattr(mlp, "act_fn", None)
    if activation is None:
        activation = torch.nn.functional.silu
    module_inputs = inputs.to(device=mlp.gate_proj.weight.device, dtype=mlp.gate_proj.weight.dtype)
    return activation(mlp.gate_proj(module_inputs)) * mlp.up_proj(module_inputs)


def _ridge_solve_features(torch, features, targets, ridge: float):
    features = torch.nan_to_num(features.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    targets = torch.nan_to_num(targets.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    token_count = features.shape[0]
    eye = torch.eye(token_count, dtype=torch.float32, device=features.device)
    gram = features @ features.T
    scale = float((torch.trace(gram) / max(1, token_count)).detach().cpu())
    regularization = max(float(ridge) * max(scale, 1e-6), 1e-6)
    gram = gram + regularization * eye
    try:
        dual = torch.linalg.solve(gram, targets)
    except RuntimeError:
        dual = torch.linalg.pinv(gram) @ targets
    return features.T @ dual


def _relative_mlp_error(torch, mlp, inputs, targets) -> float:
    module_inputs = inputs.to(device=mlp.gate_proj.weight.device, dtype=mlp.gate_proj.weight.dtype)
    outputs = mlp(module_inputs).to(torch.float32)
    targets = targets.to(device=outputs.device, dtype=torch.float32)
    error = torch.linalg.norm(outputs - targets)
    scale = torch.linalg.norm(targets).clamp_min(1e-12)
    return float((error / scale).detach().cpu())


def _collect_all_layer_output_rows(
    torch,
    model,
    tokenizer,
    layers: Sequence[Any],
    texts: Sequence[str],
    max_length: int,
    max_tokens_per_layer: int,
    device: str,
) -> list[Any]:
    pieces = [[] for _layer in layers]
    token_counts = [0 for _layer in layers]
    current_mask = None

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            nonlocal token_counts
            if token_counts[index] >= max_tokens_per_layer:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            rows = _select_token_rows(torch, tensor.detach().to(torch.float32), current_mask)
            remaining = max_tokens_per_layer - token_counts[index]
            if remaining > 0:
                pieces[index].append(rows[:remaining].detach().cpu())
                token_counts[index] += int(min(rows.shape[0], remaining))

        return hook

    handles = [layer.register_forward_hook(make_hook(idx)) for idx, layer in enumerate(layers)]
    try:
        for text in texts:
            if all(count >= max_tokens_per_layer for count in token_counts):
                break
            batch = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            current_mask = batch.get("attention_mask")
            model(**batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if any(not item for item in pieces):
        missing = [idx for idx, item in enumerate(pieces) if not item]
        raise RuntimeError(f"No activation rows captured for layers: {missing}")
    return [torch.cat(item, dim=0) for item in pieces]


def _select_token_rows(torch, tensor, attention_mask):
    values = tensor
    if values.dim() == 3:
        values = values.reshape(-1, values.shape[-1])
        if attention_mask is not None:
            mask = attention_mask.reshape(-1).detach().to(torch.bool).cpu()
            return values.cpu()[mask]
    elif values.dim() == 1:
        values = values.unsqueeze(0)
    elif values.dim() != 2:
        values = values.reshape(-1, values.shape[-1])
    return values.cpu()


def _linear_cka(torch, first, second) -> float:
    first = torch.nan_to_num(first.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    second = torch.nan_to_num(second.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    first = first - first.mean(dim=0, keepdim=True)
    second = second - second.mean(dim=0, keepdim=True)
    cross = torch.linalg.norm(first.T.matmul(second)) ** 2
    first_norm = torch.linalg.norm(first.T.matmul(first))
    second_norm = torch.linalg.norm(second.T.matmul(second))
    return float((cross / (first_norm * second_norm).clamp_min(1e-12)).detach().cpu())


def _mean_row_cosine(torch, first, second) -> float:
    first = torch.nan_to_num(first.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    second = torch.nan_to_num(second.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    return float(torch.nn.functional.cosine_similarity(first, second, dim=1, eps=1e-8).mean().detach().cpu())


def _relative_delta(torch, first, second) -> float:
    first = torch.nan_to_num(first.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    second = torch.nan_to_num(second.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    return float(((second - first).norm() / first.norm().clamp_min(1e-12)).detach().cpu())


def _variance_ratio(torch, first, second) -> float:
    first_var = torch.var(first.to(torch.float32), dim=0, unbiased=False).mean().clamp_min(1e-12)
    second_var = torch.var(second.to(torch.float32), dim=0, unbiased=False).mean()
    ratio = second_var / first_var
    if ratio < 1:
        ratio = 1 / ratio.clamp_min(1e-12)
    return float(ratio.detach().cpu())


def _decoder_layers(model) -> list[Any]:
    try:
        layers = list(_decoder_layer_container(model)[2])
    except ValueError:
        return []
    if layers and hasattr(layers[0], "mlp") and hasattr(layers[0].mlp, "gate_proj"):
        return layers
    return []


def _decoder_layer_container(model):
    candidates = [
        (getattr(model, "model", None), "layers"),
        (getattr(getattr(model, "language_model", None), "model", None), "layers"),
        (getattr(model, "gpt_neox", None), "layers"),
        (getattr(model, "transformer", None), "h"),
    ]
    for parent, attr in candidates:
        if parent is None:
            continue
        layers = getattr(parent, attr, None)
        if layers is None:
            continue
        layer_list = list(layers)
        if layer_list:
            return parent, attr, layers
    raise ValueError("Could not find decoder layer container on model.")


def _update_intermediate_size(config, new_size: int) -> None:
    for attr in ("intermediate_size", "ffn_dim"):
        if hasattr(config, attr):
            setattr(config, attr, int(new_size))
    if hasattr(config, "text_config"):
        _update_intermediate_size(config.text_config, new_size)


def _update_layer_count(config, new_count: int) -> None:
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(config, attr):
            setattr(config, attr, int(new_count))
    if hasattr(config, "text_config"):
        _update_layer_count(config.text_config, new_count)


def _normalize_layer_groups(groups: Sequence[Sequence[int]]) -> list[list[int]]:
    normalized = [sorted({int(idx) for idx in group}) for group in groups]
    normalized = [group for group in normalized if group]
    if not normalized:
        raise ValueError("Need at least one layer group.")
    for group in normalized:
        if len(group) < 2:
            raise ValueError("Each merge group must contain at least two layers.")
    seen: set[int] = set()
    for group in normalized:
        overlap = sorted(seen & set(group))
        if overlap:
            raise ValueError(f"Layer merge groups overlap at indices: {overlap}")
        seen.update(group)
    return sorted(normalized, key=lambda group: group[0])


def _layer_groups_tag(groups: Sequence[Sequence[int]]) -> str:
    return "_".join("-".join(str(idx) for idx in group) for group in groups)


def _layer_groups_for_report(groups: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(group) for group in groups]


def _merge_layer_state_dicts(
    torch,
    states: Sequence[dict[str, Any]],
    method: str,
    weights: Sequence[float] | None = None,
    output_scale: str = "group-count",
) -> dict[str, Any]:
    if not states:
        raise ValueError("Need at least one layer state.")
    if output_scale not in {"group-count", "effective-count", "sqrt-effective-count", "unit"}:
        raise ValueError("output_scale must be group-count, effective-count, sqrt-effective-count, or unit.")
    keys = set(states[0])
    for state in states[1:]:
        if set(state) != keys:
            raise ValueError("Layer state dict keys differ; cannot merge these layers.")

    merged = {}
    count = float(len(states))
    weight_values = None
    if weights is not None:
        if len(weights) != len(states):
            raise ValueError("Layer merge weights must match state count.")
        raw_weights = [max(float(item), 0.0) for item in weights]
        total_weight = sum(raw_weights)
        if not math.isfinite(total_weight) or total_weight <= 0.0:
            raw_weights = [1.0 for _state in states]
            total_weight = float(len(states))
        weight_values = [float(item / total_weight) for item in raw_weights]
    if weight_values is None and output_scale != "group-count":
        weight_values = [1.0 / count for _state in states]
    if weight_values is None:
        output_multiplier = count
    else:
        effective_count = 1.0 / max(sum(float(weight) ** 2 for weight in weight_values), 1e-12)
        if output_scale == "group-count":
            output_multiplier = count
        elif output_scale == "effective-count":
            output_multiplier = float(effective_count)
        elif output_scale == "sqrt-effective-count":
            output_multiplier = math.sqrt(float(effective_count))
        else:
            output_multiplier = 1.0
    for name in states[0]:
        first = states[0][name]
        if not first.is_floating_point():
            merged[name] = first.detach().clone()
            continue
        tensors = [state[name].detach().cpu().to(torch.float32) for state in states]
        shape = tuple(tensors[0].shape)
        if any(tuple(tensor.shape) != shape for tensor in tensors):
            raise ValueError(f"Layer parameter shape mismatch for {name}.")
        if weight_values is None:
            if method == "fold" and _is_residual_output_param(name):
                merged_tensor = sum(tensors)
            else:
                merged_tensor = sum(tensors) / count
        elif method == "fold" and _is_residual_output_param(name):
            merged_tensor = sum(tensor * weight for tensor, weight in zip(tensors, weight_values)) * output_multiplier
        else:
            merged_tensor = sum(tensor * weight for tensor, weight in zip(tensors, weight_values))
        merged[name] = merged_tensor.to(first.dtype)
    return merged


def _is_residual_output_param(name: str) -> bool:
    output_markers = (
        "o_proj.",
        "out_proj.",
        "down_proj.",
        "c_proj.",
        "w2.",
        "attention.dense.",
        "self_attention.dense.",
        "dense_4h_to_h.",
    )
    return any(marker in name for marker in output_markers)


def _drop_config_layer_metadata(config, drop_set: set[int], old_count: int) -> None:
    layer_types = getattr(config, "layer_types", None)
    if isinstance(layer_types, list) and len(layer_types) == old_count:
        config.layer_types = [value for idx, value in enumerate(layer_types) if idx not in drop_set]
    if hasattr(config, "text_config"):
        _drop_config_layer_metadata(config.text_config, drop_set, old_count)


def _merge_config_layer_metadata(config, groups: Sequence[Sequence[int]], old_count: int) -> None:
    group_by_start = {group[0]: group for group in groups}
    folded_indices = {idx for group in groups for idx in group[1:]}
    layer_types = getattr(config, "layer_types", None)
    if isinstance(layer_types, list) and len(layer_types) == old_count:
        merged_types = []
        for idx, value in enumerate(layer_types):
            if idx in folded_indices:
                continue
            group = group_by_start.get(idx)
            if group is None:
                merged_types.append(value)
            else:
                merged_types.append(value)
        config.layer_types = merged_types
    if hasattr(config, "text_config"):
        _merge_config_layer_metadata(config.text_config, groups, old_count)


def _shrink_gated_mlp_layer(torch, layer, new_size: int, assignment: str = "combined-cosine") -> None:
    gate = layer.mlp.gate_proj.weight.data.detach().cpu().to(torch.float32)
    up = layer.mlp.up_proj.weight.data.detach().cpu().to(torch.float32)
    down = layer.mlp.down_proj.weight.data.detach().cpu().to(torch.float32)
    old_size = gate.shape[0]
    if new_size >= old_size:
        return

    importance = gate.norm(dim=1) + up.norm(dim=1) + down.norm(dim=0)
    survivor_idx = torch.argsort(importance, descending=True)[:new_size]
    dropped_idx = torch.argsort(importance, descending=True)[new_size:]
    survivor_idx = torch.sort(survivor_idx).values
    survivor_set = {int(idx) for idx in survivor_idx.tolist()}
    dropped_idx = torch.tensor([idx for idx in range(old_size) if idx not in survivor_set], dtype=torch.long)

    new_gate = gate[survivor_idx].clone()
    new_up = up[survivor_idx].clone()
    new_down = down[:, survivor_idx].clone()
    counts = torch.ones(new_size, dtype=torch.float32)

    if dropped_idx.numel() > 0 and assignment != "topk":
        if assignment == "round-robin":
            assigned = torch.arange(dropped_idx.numel()) % new_size
        elif assignment in {"combined-cosine", "down-cosine"}:
            if assignment == "combined-cosine":
                surv_feat = _gated_channel_features(torch, gate, up, down, survivor_idx)
                drop_feat = _gated_channel_features(torch, gate, up, down, dropped_idx)
            else:
                surv_feat = _row_normalize(down[:, survivor_idx].T)
                drop_feat = _row_normalize(down[:, dropped_idx].T)
            assigned = []
            chunk = 128
            for start in range(0, drop_feat.shape[0], chunk):
                sims = drop_feat[start : start + chunk] @ surv_feat.T
                assigned.append(torch.argmax(sims, dim=1))
            assigned = torch.cat(assigned, dim=0)
        else:
            raise ValueError("assignment must be combined-cosine, down-cosine, round-robin, or topk.")
        for source_pos, target_pos in zip(dropped_idx.tolist(), assigned.tolist()):
            count = counts[target_pos]
            new_gate[target_pos] = (new_gate[target_pos] * count + gate[source_pos]) / (count + 1.0)
            new_up[target_pos] = (new_up[target_pos] * count + up[source_pos]) / (count + 1.0)
            new_down[:, target_pos] += down[:, source_pos]
            counts[target_pos] += 1.0

    _replace_linear(torch, layer.mlp.gate_proj, new_gate, axis="out")
    _replace_linear(torch, layer.mlp.up_proj, new_up, axis="out")
    _replace_linear(torch, layer.mlp.down_proj, new_down, axis="in")


def _expand_gated_mlp_layer(torch, layer, new_size: int) -> None:
    gate = layer.mlp.gate_proj.weight.data.detach().cpu().to(torch.float32)
    up = layer.mlp.up_proj.weight.data.detach().cpu().to(torch.float32)
    down = layer.mlp.down_proj.weight.data.detach().cpu().to(torch.float32)
    old_size = gate.shape[0]
    if new_size <= old_size:
        return
    importance = gate.norm(dim=1) + up.norm(dim=1) + down.norm(dim=0)
    order = torch.argsort(importance, descending=True).tolist()
    counts = torch.ones(old_size, dtype=torch.long)
    for extra_idx in range(new_size - old_size):
        counts[order[extra_idx % old_size]] += 1

    gate_rows = []
    up_rows = []
    down_cols = []
    for idx, copies in enumerate(counts.tolist()):
        for _ in range(copies):
            gate_rows.append(gate[idx])
            up_rows.append(up[idx])
            down_cols.append(down[:, idx] / float(copies))
    new_gate = torch.stack(gate_rows, dim=0)
    new_up = torch.stack(up_rows, dim=0)
    new_down = torch.stack(down_cols, dim=1)
    _replace_linear(torch, layer.mlp.gate_proj, new_gate, axis="out")
    _replace_linear(torch, layer.mlp.up_proj, new_up, axis="out")
    _replace_linear(torch, layer.mlp.down_proj, new_down, axis="in")


def _replace_linear(torch, linear, new_weight, axis: str) -> None:
    dtype = linear.weight.dtype
    device = linear.weight.device
    linear.out_features = int(new_weight.shape[0])
    linear.in_features = int(new_weight.shape[1])
    linear.weight = torch.nn.Parameter(new_weight.to(device=device, dtype=dtype), requires_grad=linear.weight.requires_grad)
    if linear.bias is not None:
        if axis == "out":
            linear.bias = torch.nn.Parameter(linear.bias.data[: new_weight.shape[0]].clone(), requires_grad=linear.bias.requires_grad)
        elif linear.bias.shape[0] != new_weight.shape[0]:
            raise ValueError("Cannot resize an input-axis linear layer with unexpected bias shape.")


def _row_normalize(torch_tensor):
    return torch_tensor / torch_tensor.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _gated_channel_features(torch, gate, up, down, indices):
    parts = [
        _row_normalize(gate[indices]),
        _row_normalize(up[indices]),
        _row_normalize(down[:, indices].T),
    ]
    return _row_normalize(torch.cat(parts, dim=1))


def _perplexity(torch, model, tokenizer, text: str, device: str) -> float:
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    outputs = model(input_ids=input_ids, labels=input_ids)
    return float(math.exp(min(20.0, float(outputs.loss.detach().cpu()))))


def _score_choice_probe(torch, model, tokenizer, probe: dict[str, Any], device: str) -> dict[str, Any]:
    scores = []
    prompt_ids = tokenizer(probe["prompt"], return_tensors="pt")["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]
    for choice in probe["choices"]:
        full_ids = tokenizer(probe["prompt"] + choice, return_tensors="pt")["input_ids"].to(device)
        labels = full_ids.clone()
        labels[:, :prompt_len] = -100
        outputs = model(input_ids=full_ids, labels=labels)
        token_count = int((labels != -100).sum().item())
        scores.append(float(-outputs.loss.detach().cpu() * max(1, token_count)))
    pred = int(max(range(len(scores)), key=lambda idx: scores[idx]))
    return {
        "id": probe["id"],
        "prediction": pred,
        "answer": int(probe["answer"]),
        "correct": pred == int(probe["answer"]),
        "scores": scores,
        "choices": probe["choices"],
    }


def _generate_text(torch, model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
    )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    continuation = decoded[len(prompt) :] if decoded.startswith(prompt) else decoded
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :].detach().cpu().tolist()
    return {
        "prompt": prompt,
        "output": decoded,
        "diagnostics": _generation_diagnostics(continuation, generated_ids),
    }


def _generate_text_batch(
    torch,
    model,
    tokenizer,
    prompts: Sequence[str],
    device: str,
    max_new_tokens: int,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    prompt_list = [str(prompt) for prompt in prompts]
    if not prompt_list:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    old_padding_side = getattr(tokenizer, "padding_side", None)
    if old_padding_side is not None:
        tokenizer.padding_side = "left"
    try:
        results = []
        for start in range(0, len(prompt_list), int(batch_size)):
            batch_prompts = prompt_list[start : start + int(batch_size)]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
            prompt_width = int(inputs["input_ids"].shape[1])
            for index, prompt in enumerate(batch_prompts):
                decoded = tokenizer.decode(output_ids[index], skip_special_tokens=True)
                continuation = decoded[len(prompt) :] if decoded.startswith(prompt) else decoded
                generated_ids = output_ids[index, prompt_width:].detach().cpu().tolist()
                results.append(
                    {
                        "prompt": prompt,
                        "output": decoded,
                        "diagnostics": _generation_diagnostics(continuation, generated_ids),
                    }
                )
        return results
    finally:
        if old_padding_side is not None:
            tokenizer.padding_side = old_padding_side


def _generate_text_for_spec(torch, model, tokenizer, spec: dict[str, Any], device: str, max_new_tokens: int) -> dict[str, Any]:
    item = _generate_text(torch, model, tokenizer, str(spec["prompt"]), device, max_new_tokens)
    return _generation_from_cached_spec(item, spec)


def _generation_from_cached_spec(generation: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    item = dict(generation)
    item["id"] = spec.get("id")
    if spec.get("category") is not None:
        item["category"] = spec.get("category")
    if spec.get("relevance_rules") is not None:
        item["relevance_rules"] = spec.get("relevance_rules")
    for key in ("kind", "answer", "function_name", "tests", "pattern_groups"):
        if key in spec:
            item[key] = spec[key]
    return item


def _score_behavior_probe(probe: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    prompt = str(probe["prompt"])
    output = str(generation.get("output", ""))
    continuation = output[len(prompt) :] if output.startswith(prompt) else output
    continuation = continuation.strip()
    kind = str(probe["kind"])
    if kind == "first_number":
        numbers = re.findall(r"-?\d+", continuation)
        prediction = numbers[0] if numbers else ""
        passed = prediction == str(probe["answer"])
        details = {"prediction": prediction, "answer": str(probe["answer"]), "numbers": numbers[:5]}
    elif kind == "pattern_groups":
        details = _score_pattern_groups(continuation, probe.get("pattern_groups", []))
        passed = bool(details["passed"])
    elif kind == "python_function":
        details = _score_python_function(
            continuation,
            function_name=str(probe.get("function_name", "")),
            tests=probe.get("tests", []),
        )
        passed = bool(details["passed"])
    else:
        raise ValueError(f"Unknown behavior probe kind: {kind}")
    return {
        "id": str(probe["id"]),
        "category": probe.get("category"),
        "prompt": prompt,
        "kind": kind,
        "passed": bool(passed),
        "output": output,
        "continuation": continuation,
        "diagnostics": generation.get("diagnostics", _generation_diagnostics(continuation)),
        "details": details,
    }


def _score_python_function(text: str, function_name: str, tests: Sequence[dict[str, Any]]) -> dict[str, Any]:
    source = _extract_python_source(text, function_name)
    details: dict[str, Any] = {
        "function_name": function_name,
        "source": source,
        "passed": False,
        "parse_error": None,
        "function_found": False,
        "test_results": [],
    }
    if not function_name:
        details["reason"] = "missing_function_name"
        return details
    if not tests:
        details["reason"] = "missing_tests"
        return details
    try:
        tree = _parse_python_source_prefix(source)
    except SyntaxError as exc:
        details["parse_error"] = str(exc)
        details["reason"] = "syntax_error"
        return details
    target = _find_python_function_expr(tree, function_name)
    if target is None:
        details["reason"] = "function_not_found"
        return details
    arg_names, expr = target
    details["function_found"] = True
    if len(arg_names) != 2:
        details["reason"] = "wrong_arity"
        details["arity"] = len(arg_names)
        return details
    passed = True
    for item in tests:
        args = list(item.get("args", []))
        expected = item.get("expected")
        result_item: dict[str, Any] = {"args": args, "expected": expected, "passed": False}
        if len(args) != len(arg_names):
            result_item["reason"] = "wrong_test_arity"
            passed = False
            details["test_results"].append(result_item)
            continue
        env = {name: value for name, value in zip(arg_names, args)}
        try:
            observed = _eval_safe_python_expr(expr, env)
        except ValueError as exc:
            result_item["reason"] = str(exc)
            passed = False
            details["test_results"].append(result_item)
            continue
        result_item["observed"] = observed
        result_item["passed"] = observed == expected
        if observed != expected:
            passed = False
        details["test_results"].append(result_item)
    details["passed"] = bool(passed)
    if not passed:
        details["reason"] = "test_failed"
    return details


def _extract_python_source(text: str, function_name: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:python)?\s*", "", cleaned, count=1, flags=re.IGNORECASE)
        cleaned = cleaned.split("```", 1)[0].strip()
    else:
        cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
    if cleaned.lower().startswith("python\n"):
        cleaned = cleaned.split("\n", 1)[1].strip()
    patterns = [
        rf"(?m)^\s*def\s+{re.escape(function_name)}\b",
        rf"(?m)^\s*{re.escape(function_name)}\s*=\s*lambda\b",
    ]
    starts = []
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            starts.append(match.start())
    if starts:
        cleaned = cleaned[min(starts):].strip()
    return cleaned


def _parse_python_source_prefix(source: str):
    try:
        return ast.parse(source)
    except SyntaxError as original:
        lines = source.splitlines()
        for end in range(len(lines) - 1, 0, -1):
            candidate = "\n".join(lines[:end]).strip()
            if not candidate:
                continue
            try:
                return ast.parse(candidate)
            except SyntaxError:
                continue
        raise original


def _find_python_function_expr(tree, function_name: str) -> tuple[list[str], ast.AST] | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            arg_names = [arg.arg for arg in node.args.args]
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                if isinstance(body[0].value.value, str):
                    body = body[1:]
            for statement in body:
                if isinstance(statement, ast.Return) and statement.value is not None:
                    return arg_names, statement.value
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == function_name:
                if isinstance(node.value, ast.Lambda):
                    arg_names = [arg.arg for arg in node.value.args.args]
                    return arg_names, node.value.body
    return None


def _eval_safe_python_expr(node, env: dict[str, Any]):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"unknown_name:{node.id}")
        value = env[node.id]
        if not isinstance(value, (int, float)):
            raise ValueError(f"non_numeric_name:{node.id}")
        return value
    if isinstance(node, ast.UnaryOp):
        value = _eval_safe_python_expr(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError("unsupported_unary_op")
    if isinstance(node, ast.BinOp):
        left = _eval_safe_python_expr(node.left, env)
        right = _eval_safe_python_expr(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        raise ValueError("unsupported_binary_op")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id != "sum" or len(node.args) != 1 or node.keywords:
            raise ValueError("unsupported_call")
        values = _eval_safe_iterable_expr(node.args[0], env)
        return sum(values)
    raise ValueError(f"unsupported_expr:{type(node).__name__}")


def _eval_safe_iterable_expr(node, env: dict[str, Any]) -> list[Any]:
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_safe_python_expr(item, env) for item in node.elts]
    raise ValueError("unsupported_iterable")


def _score_pattern_groups(text: str, pattern_groups: Sequence[Sequence[str]]) -> dict[str, Any]:
    lowered = text.lower()
    matched_groups = []
    missing_groups = []
    for group_index, patterns in enumerate(pattern_groups):
        matched = [pattern for pattern in patterns if re.search(pattern, lowered)]
        group = {"group": group_index, "matched_patterns": matched, "patterns": list(patterns)}
        if matched:
            matched_groups.append(group)
        else:
            missing_groups.append(group)
    return {
        "passed": not missing_groups,
        "matched_groups": matched_groups,
        "missing_groups": missing_groups,
    }


def _make_probe_images(Image, ImageDraw, image_root: Path) -> dict[str, Path]:
    shapes = image_root / "red_square_blue_circle.png"
    img = Image.new("RGB", (320, 220), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((35, 50, 135, 150), fill="red")
    draw.ellipse((190, 55, 290, 155), fill="blue")
    img.save(shapes)

    ocr = image_root / "number_42.png"
    img2 = Image.new("RGB", (320, 180), "white")
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle((0, 0, 319, 179), outline="black", width=4)
    draw2.text((105, 55), "42", fill="black")
    img2.save(ocr)
    return {"shapes": shapes, "ocr": ocr}


def _apply_vision_template(processor, question: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, add_generation_prompt=True)
    return f"<image>\n{question}\nAnswer:"


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
