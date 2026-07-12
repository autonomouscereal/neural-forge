"""Tiny native early-fusion vision/text sandbox.

This module trains a single causal transformer over raw image patch embeddings
and text token embeddings. It is deliberately small: the point is to validate a
native multimodal pathway end-to-end, not to compete with real VLMs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


COLORS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 0.8, 0.0),
    "blue": (0.0, 0.1, 1.0),
    "yellow": (1.0, 0.9, 0.0),
}

DIGITS = ["zero", "one", "two", "three"]

QUESTIONS = [
    ("what color is square ?", "square_color"),
    ("what color is circle ?", "circle_color"),
    ("what number ?", "digit"),
]

SPECIAL_TOKENS = ["<pad>", "<bos>", "<sep>", "<eos>"]


def native_vocab() -> dict[str, int]:
    words = set(SPECIAL_TOKENS)
    for question, _ in QUESTIONS:
        words.update(question.split())
    words.update(COLORS)
    words.update(DIGITS)
    return {word: idx for idx, word in enumerate(sorted(words))}


def train_native_vlm_sandbox(
    output_dir: str | Path,
    steps: int = 600,
    batch_size: int = 64,
    eval_batches: int = 16,
    seed: int = 7,
    device: str = "cpu",
    d_model: int = 96,
    layers: int = 3,
    heads: int = 4,
    lr: float = 3e-4,
) -> dict[str, Any]:
    torch = _import_torch()
    _seed_everything(torch, seed)
    vocab = native_vocab()
    id_to_token = {idx: token for token, idx in vocab.items()}
    model = _build_native_model(torch, vocab_size=len(vocab), d_model=d_model, layers=layers, heads=heads)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    model.train()
    for step in range(1, steps + 1):
        batch = _make_batch(torch, vocab, batch_size=batch_size, seed=seed * 100_000 + step, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["images"], batch["input_ids"])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch["labels"].reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, steps // 10) == 0:
            losses.append({"step": step, "loss": float(loss.detach().cpu())})

    metrics = _evaluate_native_model(torch, model, vocab, id_to_token, device, eval_batches=eval_batches, seed=seed + 99)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "vocab": vocab,
        "config": {"d_model": d_model, "layers": layers, "heads": heads, "image_size": 16, "patch_size": 4},
    }
    torch.save(checkpoint, target / "native_vlm_sandbox.pt")
    report = {
        "operation": "train_native_vlm_sandbox",
        "output_dir": str(target),
        "steps": steps,
        "batch_size": batch_size,
        "eval_batches": eval_batches,
        "seed": seed,
        "device": device,
        "losses": losses,
        "metrics": metrics,
        "param_count": sum(param.numel() for param in model.parameters()),
        "vocab": vocab,
    }
    _write_json(target / "native_vlm_report.json", report)
    return report


def _import_torch():
    import torch

    return torch


def _seed_everything(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_native_model(torch, vocab_size: int, d_model: int, layers: int, heads: int):
    class NativeVisionTextModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            patch_dim = 4 * 4 * 3
            self.patch = torch.nn.Linear(patch_dim, d_model)
            self.token = torch.nn.Embedding(vocab_size, d_model)
            self.pos = torch.nn.Parameter(torch.zeros(1, 64, d_model))
            block = torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=d_model * 4,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.blocks = torch.nn.TransformerEncoder(block, num_layers=layers)
            self.norm = torch.nn.LayerNorm(d_model)
            self.head = torch.nn.Linear(d_model, vocab_size)

        def forward(self, images, input_ids):
            patches = _patchify(torch, images)
            patch_emb = self.patch(patches)
            token_emb = self.token(input_ids)
            hidden = torch.cat([patch_emb, token_emb], dim=1)
            hidden = hidden + self.pos[:, : hidden.shape[1]]
            mask = torch.nn.Transformer.generate_square_subsequent_mask(hidden.shape[1], device=hidden.device)
            hidden = self.blocks(hidden, mask=mask, is_causal=True)
            return self.head(self.norm(hidden))

    return NativeVisionTextModel()


def _patchify(torch, images):
    batch, channels, height, width = images.shape
    patches = images.unfold(2, 4, 4).unfold(3, 4, 4)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch, (height // 4) * (width // 4), channels * 4 * 4)
    return patches


def _make_batch(torch, vocab: dict[str, int], batch_size: int, seed: int, device: str):
    rng = random.Random(seed)
    images = []
    input_ids = []
    labels = []
    max_text_len = 0
    examples = []
    for _ in range(batch_size):
        square_color = rng.choice(list(COLORS))
        circle_color = rng.choice(list(COLORS))
        digit = rng.choice(DIGITS)
        question, answer_key = rng.choice(QUESTIONS)
        if answer_key == "square_color":
            answer = square_color
        elif answer_key == "circle_color":
            answer = circle_color
        else:
            answer = digit
        image = _synthetic_image(square_color, circle_color, digit, seed=seed + len(examples))
        text_ids, label_ids = _encode_qa(vocab, question, answer)
        max_text_len = max(max_text_len, len(text_ids))
        examples.append((image, text_ids, label_ids))

    pad = vocab["<pad>"]
    for image, text_ids, label_ids in examples:
        pad_len = max_text_len - len(text_ids)
        images.append(image)
        input_ids.append(text_ids + [pad] * pad_len)
        labels.append([-100] * 16 + label_ids + [-100] * pad_len)

    return {
        "images": torch.tensor(np.stack(images), dtype=torch.float32, device=device),
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def _synthetic_image(square_color: str, circle_color: str, digit: str, seed: int) -> np.ndarray:
    image = np.ones((3, 16, 16), dtype=np.float32)
    square = np.array(COLORS[square_color], dtype=np.float32)[:, None, None]
    circle = np.array(COLORS[circle_color], dtype=np.float32)[:, None]
    image[:, 2:7, 2:7] = square
    yy, xx = np.ogrid[:16, :16]
    mask = (yy - 10) ** 2 + (xx - 11) ** 2 <= 3**2
    image[:, mask] = circle
    _draw_digit(image, DIGITS.index(digit))
    image += np.random.default_rng(seed).normal(0.0, 0.015, size=image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0)


def _draw_digit(image: np.ndarray, digit: int) -> None:
    segments_by_digit = {
        0: ("top", "upper_left", "upper_right", "lower_left", "lower_right", "bottom"),
        1: ("upper_right", "lower_right"),
        2: ("top", "upper_right", "middle", "lower_left", "bottom"),
        3: ("top", "upper_right", "middle", "lower_right", "bottom"),
    }
    slices = {
        "top": (slice(1, 2), slice(10, 15)),
        "middle": (slice(5, 6), slice(10, 15)),
        "bottom": (slice(9, 10), slice(10, 15)),
        "upper_left": (slice(1, 6), slice(10, 11)),
        "upper_right": (slice(1, 6), slice(14, 15)),
        "lower_left": (slice(5, 10), slice(10, 11)),
        "lower_right": (slice(5, 10), slice(14, 15)),
    }
    for segment in segments_by_digit[digit]:
        rows, cols = slices[segment]
        image[:, rows, cols] = 0.0


def _encode_qa(vocab: dict[str, int], question: str, answer: str) -> tuple[list[int], list[int]]:
    prompt = ["<bos>"] + question.split() + ["<sep>"]
    answer_tokens = [answer, "<eos>"]
    input_tokens = prompt + answer_tokens[:-1]
    labels = [-100] * (len(prompt) - 1) + [vocab[token] for token in answer_tokens]
    return [vocab[token] for token in input_tokens], labels


def _evaluate_native_model(torch, model, vocab: dict[str, int], id_to_token: dict[int, str], device: str, eval_batches: int, seed: int):
    model.eval()
    total = 0
    correct = 0
    samples = []
    with torch.no_grad():
        for batch_idx in range(eval_batches):
            batch = _make_batch(torch, vocab, batch_size=32, seed=seed * 10_000 + batch_idx, device=device)
            logits = model(batch["images"], batch["input_ids"])
            label_positions = batch["labels"] != -100
            first_label_positions = label_positions.float().argmax(dim=1)
            predicted = logits[torch.arange(logits.shape[0], device=device), first_label_positions].argmax(dim=-1)
            expected = batch["labels"][torch.arange(logits.shape[0], device=device), first_label_positions]
            correct += int((predicted == expected).sum().item())
            total += int(expected.numel())
            if len(samples) < 8:
                for row in range(min(4, expected.numel())):
                    samples.append(
                        {
                            "expected": id_to_token[int(expected[row].detach().cpu())],
                            "predicted": id_to_token[int(predicted[row].detach().cpu())],
                        }
                    )
    return {"answer_accuracy": correct / max(1, total), "correct": correct, "total": total, "samples": samples}


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
