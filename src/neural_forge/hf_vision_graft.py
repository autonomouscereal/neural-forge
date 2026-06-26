"""Raw-patch vision graft training for Hugging Face causal LMs.

This is a deliberately small native multimodal experiment: raw image patches are
projected into the same embedding stream as text tokens and trained against a
real causal LM parent. No external vision encoder is used.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from neural_forge.hf_tools import _decoder_layers, _dtype_from_name, _torch_and_transformers
from neural_forge.native_multimodal import COLORS, DIGITS, _patchify, _seed_everything, _synthetic_image


GRAFT_QUESTIONS = [
    ("What color is the square?", "square_color"),
    ("What color is the circle?", "circle_color"),
    ("What digit is shown?", "digit"),
]


def train_hf_native_vision_graft(
    model_id_or_path: str,
    output_dir: str | Path,
    steps: int = 300,
    batch_size: int = 8,
    eval_batches: int = 8,
    seed: int = 11,
    device: str = "cuda:0",
    torch_dtype: str = "float16",
    lr: float = 3e-3,
    train_last_layers: int = 0,
    trust_remote_code: bool = False,
    max_new_tokens: int = 6,
    task_keys: Sequence[str] | None = None,
    grad_clip: float = 1.0,
) -> dict[str, Any]:
    """Train/evaluate a raw-patch prefix graft on a real HF causal LM parent."""

    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if eval_batches <= 0:
        raise ValueError("eval_batches must be positive.")
    if train_last_layers < 0:
        raise ValueError("train_last_layers must be non-negative.")
    if grad_clip < 0:
        raise ValueError("grad_clip must be non-negative.")
    if train_last_layers > 0 and torch_dtype.lower() in {"auto", "source", "float16", "fp16"}:
        raise ValueError("Training LM layers requires an explicit non-fp16 dtype; use --dtype float32 or keep --train-last-layers 0.")
    active_task_keys = _normalize_task_keys(task_keys)

    torch, transformers = _torch_and_transformers()
    _seed_everything(torch, seed)
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

    hidden_size = int(model.get_input_embeddings().embedding_dim)
    graft = _build_patch_graft(torch, hidden_size=hidden_size)
    graft.to(device)

    trainable_model_params = _set_trainable_model_window(model, train_last_layers)
    optimizer_params = list(graft.parameters()) + [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(optimizer_params, lr=lr)

    baseline_metrics = _evaluate_graft(
        torch,
        model,
        graft,
        tokenizer,
        device=device,
        eval_batches=eval_batches,
        seed=seed + 10_000,
        max_new_tokens=max_new_tokens,
        task_keys=active_task_keys,
    )

    losses = []
    for step in range(1, steps + 1):
        model.train(train_last_layers > 0)
        graft.train()
        batch = _make_hf_vision_batch(
            torch,
            tokenizer,
            batch_size=batch_size,
            seed=seed * 100_000 + step,
            device=device,
            task_keys=active_task_keys,
        )
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward_with_graft(torch, model, graft, batch, labels=batch["labels"])
        loss = outputs.loss
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise RuntimeError(f"Non-finite loss at step {step}; aborting before saving a graft.")
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(optimizer_params, grad_clip)
        optimizer.step()
        if step == 1 or step % max(1, steps // 10) == 0:
            losses.append({"step": step, "loss": float(loss.detach().cpu())})

    metrics = _evaluate_graft(
        torch,
        model,
        graft,
        tokenizer,
        device=device,
        eval_batches=eval_batches,
        seed=seed + 20_000,
        max_new_tokens=max_new_tokens,
        task_keys=active_task_keys,
    )

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "graft_state": graft.state_dict(),
            "config": {"hidden_size": hidden_size, "patch_size": 4, "image_size": 16, "patch_count": 16},
            "model_id_or_path": model_id_or_path,
            "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", model_id_or_path),
        },
        target / "hf_native_vision_graft.pt",
    )
    if train_last_layers > 0:
        model.save_pretrained(target / "text_model", safe_serialization=True)
        tokenizer.save_pretrained(target / "text_model")

    report = {
        "operation": "train_hf_native_vision_graft",
        "model": model_id_or_path,
        "output_dir": str(target),
        "steps": steps,
        "batch_size": batch_size,
        "eval_batches": eval_batches,
        "seed": seed,
        "device": device,
        "torch_dtype": torch_dtype,
        "lr": lr,
        "train_last_layers": train_last_layers,
        "grad_clip": grad_clip,
        "task_keys": active_task_keys,
        "losses": losses,
        "baseline_metrics": baseline_metrics,
        "metrics": metrics,
        "parent_param_count": sum(param.numel() for param in model.parameters()),
        "graft_param_count": sum(param.numel() for param in graft.parameters()),
        "trainable_model_param_count": trainable_model_params,
    }
    _write_json(target / "hf_native_vision_graft_report.json", report)
    return report


def graft_prompt(question: str) -> str:
    return f"Image patches are provided before this text. Question: {question} Answer:"


def graft_answer_options(answer_key: str) -> list[str]:
    if answer_key in {"square_color", "circle_color"}:
        return list(COLORS)
    if answer_key == "digit":
        return list(DIGITS)
    raise ValueError(f"Unknown answer key: {answer_key}")


def _normalize_task_keys(task_keys: Sequence[str] | None) -> list[str]:
    available = {key for _, key in GRAFT_QUESTIONS}
    if task_keys is None:
        return [key for _, key in GRAFT_QUESTIONS]
    values = [str(key).strip() for key in task_keys if str(key).strip()]
    if not values:
        raise ValueError("At least one task key is required.")
    unknown = sorted(set(values) - available)
    if unknown:
        raise ValueError(f"Unknown task key(s): {unknown}; available={sorted(available)}")
    return values


def _build_patch_graft(torch, hidden_size: int):
    class PatchGraft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = torch.nn.Linear(4 * 4 * 3, hidden_size)
            self.pos = torch.nn.Parameter(torch.zeros(1, 16, hidden_size))
            torch.nn.init.normal_(self.patch.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(self.patch.bias)

        def forward(self, images):
            patches = _patchify(torch, images)
            return self.patch(patches) + self.pos[:, : patches.shape[1]]

    return PatchGraft()


def _set_trainable_model_window(model, train_last_layers: int) -> int:
    for param in model.parameters():
        param.requires_grad_(False)
    if train_last_layers == 0:
        return 0

    layers = _decoder_layers(model)
    if not layers:
        raise ValueError("Could not find decoder layers for train_last_layers.")
    for layer in layers[-train_last_layers:]:
        for param in layer.parameters():
            param.requires_grad_(True)
    for name in ("lm_head", "norm"):
        module = getattr(model, name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad_(True)
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _make_hf_vision_batch(torch, tokenizer, batch_size: int, seed: int, device: str, task_keys: Sequence[str] | None = None) -> dict[str, Any]:
    rng = random.Random(seed)
    examples = [_make_hf_vision_example(rng, seed + idx, task_keys=task_keys) for idx in range(batch_size)]
    encoded = [_encode_prompt_answer(tokenizer, item["question"], item["answer"]) for item in examples]
    max_len = max(len(item["input_ids"]) for item in encoded)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    input_ids = []
    labels = []
    attention = []
    for item in encoded:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
        attention.append([1] * len(item["input_ids"]) + [0] * pad_len)

    return {
        "images": torch.tensor(np.stack([item["image"] for item in examples]), dtype=torch.float32, device=device),
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
        "examples": examples,
    }


def _make_hf_vision_example(rng: random.Random, seed: int, task_keys: Sequence[str] | None = None) -> dict[str, Any]:
    square_color = rng.choice(list(COLORS))
    circle_color = rng.choice(list(COLORS))
    digit = rng.choice(DIGITS)
    active_keys = set(_normalize_task_keys(task_keys))
    question, answer_key = rng.choice([item for item in GRAFT_QUESTIONS if item[1] in active_keys])
    if answer_key == "square_color":
        answer = square_color
    elif answer_key == "circle_color":
        answer = circle_color
    else:
        answer = digit
    image = _synthetic_image(square_color, circle_color, digit, seed=seed)
    return {
        "image": image,
        "question": question,
        "answer_key": answer_key,
        "answer": answer,
        "square_color": square_color,
        "circle_color": circle_color,
        "digit": digit,
    }


def _encode_prompt_answer(tokenizer, question: str, answer: str) -> dict[str, list[int]]:
    prompt_ids = tokenizer(graft_prompt(question), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
        "prompt_ids": prompt_ids,
        "answer_ids": answer_ids,
    }


def _forward_with_graft(torch, model, graft, batch: dict[str, Any], labels=None):
    token_emb = model.get_input_embeddings()(batch["input_ids"])
    patch_emb = graft(batch["images"]).to(dtype=token_emb.dtype)
    inputs_embeds = torch.cat([patch_emb, token_emb], dim=1)
    patch_attention = torch.ones((token_emb.shape[0], patch_emb.shape[1]), dtype=batch["attention_mask"].dtype, device=token_emb.device)
    attention_mask = torch.cat([patch_attention, batch["attention_mask"]], dim=1)
    if labels is not None:
        patch_labels = torch.full((token_emb.shape[0], patch_emb.shape[1]), -100, dtype=labels.dtype, device=labels.device)
        labels = torch.cat([patch_labels, labels], dim=1)
    return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, use_cache=False)


def _evaluate_graft(
    torch,
    model,
    graft,
    tokenizer,
    device: str,
    eval_batches: int,
    seed: int,
    max_new_tokens: int,
    task_keys: Sequence[str] | None,
) -> dict[str, Any]:
    model.eval()
    graft.eval()
    total = 0
    correct = 0
    samples = []
    losses = []
    with torch.no_grad():
        for batch_idx in range(eval_batches):
            rng = random.Random(seed * 100_000 + batch_idx)
            for row in range(8):
                example = _make_hf_vision_example(rng, seed + batch_idx * 100 + row, task_keys=task_keys)
                prediction, candidate_losses = _predict_by_candidate_loss(torch, model, graft, tokenizer, example, device)
                total += 1
                correct += int(prediction == example["answer"])
                losses.append(candidate_losses[example["answer"]])
                if len(samples) < 8:
                    generated = _greedy_generate_answer(torch, model, graft, tokenizer, example, device, max_new_tokens=max_new_tokens)
                    samples.append(
                        {
                            "question": example["question"],
                            "expected": example["answer"],
                            "predicted": prediction,
                            "generated": generated,
                            "candidate_losses": candidate_losses,
                        }
                    )
    return {
        "answer_accuracy": correct / max(1, total),
        "correct": correct,
        "total": total,
        "mean_gold_loss": float(sum(losses) / max(1, len(losses))),
        "samples": samples,
    }


def _predict_by_candidate_loss(torch, model, graft, tokenizer, example: dict[str, Any], device: str) -> tuple[str, dict[str, float]]:
    options = graft_answer_options(example["answer_key"])
    losses: dict[str, float] = {}
    for option in options:
        encoded = _encode_prompt_answer(tokenizer, example["question"], option)
        batch = {
            "images": torch.tensor(example["image"][None], dtype=torch.float32, device=device),
            "input_ids": torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device),
            "labels": torch.tensor([encoded["labels"]], dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, len(encoded["input_ids"])), dtype=torch.long, device=device),
        }
        outputs = _forward_with_graft(torch, model, graft, batch, labels=None)
        losses[option] = _masked_next_token_loss(torch, outputs.logits, batch["labels"], prefix_len=16)
    prediction = min(losses, key=losses.get)
    return prediction, losses


def _masked_next_token_loss(torch, logits, token_labels, prefix_len: int) -> float:
    patch_labels = torch.full((token_labels.shape[0], prefix_len), -100, dtype=token_labels.dtype, device=token_labels.device)
    labels = torch.cat([patch_labels, token_labels], dim=1)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    mask = shift_labels != -100
    return float((flat_loss * mask).sum().detach().cpu() / mask.sum().clamp_min(1).detach().cpu())


def _greedy_generate_answer(torch, model, graft, tokenizer, example: dict[str, Any], device: str, max_new_tokens: int) -> str:
    prompt_ids = tokenizer(graft_prompt(example["question"]), add_special_tokens=False)["input_ids"]
    generated: list[int] = []
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([prompt_ids + generated], dtype=torch.long, device=device)
        batch = {
            "images": torch.tensor(example["image"][None], dtype=torch.float32, device=device),
            "input_ids": input_ids,
            "labels": torch.full_like(input_ids, -100),
            "attention_mask": torch.ones_like(input_ids),
        }
        outputs = _forward_with_graft(torch, model, graft, batch, labels=None)
        next_id = int(outputs.logits[0, -1].argmax().detach().cpu())
        if tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
            break
        generated.append(next_id)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
