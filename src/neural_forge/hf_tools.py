"""Hugging Face checkpoint surgery and lightweight behavior probes.

This module is imported lazily by the CLI so the NumPy-only dense engine remains
usable without installing torch/transformers.
"""

from __future__ import annotations

import json
import math
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

    cast_state = {name: tensor.to(dtype) if tensor.is_floating_point() else tensor for name, tensor in merged_state.items()}
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


def resize_gated_ffn(
    model_id_or_path: str,
    output_dir: str | Path,
    ratio: float | None = None,
    target_intermediate_size: int | None = None,
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
    assignment: str = "down-cosine",
) -> dict[str, Any]:
    torch, transformers = _torch_and_transformers()
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


def evaluate_text_model(
    model_id_or_path: str,
    output_file: str | Path,
    torch_dtype: str = "float16",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    max_new_tokens: int = 96,
) -> dict[str, Any]:
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
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        ppls = [_perplexity(torch, model, tokenizer, text, device) for text in TEXT_EVALS]
        probes = [_score_choice_probe(torch, model, tokenizer, probe, device) for probe in CHOICE_PROBES]
        generations = [_generate_text(torch, model, tokenizer, prompt, device, max_new_tokens) for prompt in GENERATION_PROMPTS]

    report = {
        "model": model_id_or_path,
        "param_count": sum(p.numel() for p in model.parameters()),
        "perplexity": {
            "mean": float(sum(ppls) / len(ppls)),
            "items": ppls,
        },
        "choice_accuracy": {
            "correct": sum(1 for item in probes if item["correct"]),
            "total": len(probes),
            "accuracy": sum(1 for item in probes if item["correct"]) / len(probes),
            "items": probes,
        },
        "generations": generations,
    }
    _write_json(output_file, report)
    return report


def gate_text_report(
    baseline_report: str | Path | dict[str, Any],
    candidate_report: str | Path | dict[str, Any],
    output_file: str | Path | None = None,
    max_ppl_ratio: float = 1.25,
    max_accuracy_drop: float = 0.0,
    min_choice_accuracy: float = 0.75,
    min_continuation_chars: int = 24,
    min_unique_token_ratio: float = 0.20,
) -> dict[str, Any]:
    """Compare candidate behavior to a baseline report and produce a gate verdict."""

    if max_ppl_ratio <= 0:
        raise ValueError("max_ppl_ratio must be positive.")
    if max_accuracy_drop < 0:
        raise ValueError("max_accuracy_drop must be non-negative.")
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

    checks = [
        {
            "name": "perplexity_ratio",
            "passed": ppl_ratio <= max_ppl_ratio,
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
    verdict = {
        "operation": "gate_text_report",
        "passed": all(check["passed"] for check in checks),
        "baseline_model": baseline.get("model"),
        "candidate_model": candidate.get("model"),
        "baseline": {
            "param_count": baseline.get("param_count"),
            "perplexity_mean": base_ppl,
            "choice_accuracy": base_acc,
        },
        "candidate": {
            "param_count": candidate.get("param_count"),
            "perplexity_mean": cand_ppl,
            "choice_accuracy": cand_acc,
        },
        "checks": checks,
    }
    if output_file is not None:
        _write_json(output_file, verdict)
    return verdict


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
        "candidates": candidates,
        "passed_candidates": [item for item in candidates if item["passed"]],
    }
    _write_json(report_root / "sweep_summary.json", summary)
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


def _load_report(report: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(report, dict):
        return report
    return json.loads(Path(report).read_text(encoding="utf-8"))


def _generation_health(item: dict[str, Any], min_continuation_chars: int, min_unique_token_ratio: float) -> dict[str, Any]:
    prompt = str(item.get("prompt", ""))
    output = str(item.get("output", ""))
    continuation = output[len(prompt) :] if prompt and output.startswith(prompt) else output
    continuation = continuation.strip()
    tokens = re.findall(r"[A-Za-z0-9_]+", continuation.lower())
    unique_ratio = (len(set(tokens)) / len(tokens)) if tokens else 0.0
    long_enough = len(continuation) >= min_continuation_chars
    diverse_enough = len(tokens) < 20 or unique_ratio >= min_unique_token_ratio
    return {
        "prompt": prompt,
        "passed": long_enough and diverse_enough,
        "continuation_chars": len(continuation),
        "token_count": len(tokens),
        "unique_token_ratio": unique_ratio,
        "long_enough": long_enough,
        "diverse_enough": diverse_enough,
    }


def _ratio_tag(ratio: float) -> str:
    return f"{ratio:.4f}".replace(".", "p").replace("-", "m")


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


def _decoder_layers(model) -> list[Any]:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "language_model", None), "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for layers in candidates:
        if layers is None:
            continue
        layer_list = list(layers)
        if layer_list and hasattr(layer_list[0], "mlp") and hasattr(layer_list[0].mlp, "gate_proj"):
            return layer_list
    return []


def _update_intermediate_size(config, new_size: int) -> None:
    for attr in ("intermediate_size", "ffn_dim"):
        if hasattr(config, attr):
            setattr(config, attr, int(new_size))
    if hasattr(config, "text_config"):
        _update_intermediate_size(config.text_config, new_size)


def _shrink_gated_mlp_layer(torch, layer, new_size: int, assignment: str = "down-cosine") -> None:
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

    if dropped_idx.numel() > 0:
        if assignment == "round-robin":
            assigned = torch.arange(dropped_idx.numel()) % new_size
        else:
            surv_feat = _row_normalize(down[:, survivor_idx].T)
            drop_feat = _row_normalize(down[:, dropped_idx].T)
            assigned = []
            chunk = 128
            for start in range(0, drop_feat.shape[0], chunk):
                sims = drop_feat[start : start + chunk] @ surv_feat.T
                assigned.append(torch.argmax(sims, dim=1))
            assigned = torch.cat(assigned, dim=0)
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


def _generate_text(torch, model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> dict[str, str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"prompt": prompt, "output": decoded}


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
