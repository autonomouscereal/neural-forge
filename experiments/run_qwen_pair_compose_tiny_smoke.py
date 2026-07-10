#!/usr/bin/env python3
"""Build local tiny Qwen checkpoints and smoke the pair-composition command offline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from experiments.launch_qwen_pair_compose_run import ComposeSettings
    from experiments.launch_qwen_pair_compose_run import build_compose_command
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.launch_qwen_pair_compose_run import ComposeSettings
    from experiments.launch_qwen_pair_compose_run import build_compose_command


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TinyFixture:
    root: Path
    source: Path
    blank_candidate: Path
    calibration_file: Path
    output: Path
    architecture: str


def offline_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": str(REPO_ROOT / "src"),
        }
    )
    return environment


def _tiny_vocabulary() -> dict[str, int]:
    tokens = [
        "<unk>",
        "<pad>",
        "<bos>",
        "<eos>",
        "Q",
        "A",
        "What",
        "is",
        "two",
        "plus",
        "three",
        "Write",
        "a",
        "small",
        "function",
        "return",
        "Explain",
        "model",
        "compression",
        "layer",
        "pair",
        "attention",
        "token",
        "safe",
        "answer",
        "+",
        "-",
        "=",
        "(",
        ")",
        ":",
        ".",
    ]
    return {token: index for index, token in enumerate(tokens)}


def _build_local_tokenizer(targets: tuple[Path, Path]) -> int:
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise RuntimeError(
            "The offline tiny smoke requires the neural-forge 'hf' extra."
        ) from exc

    vocabulary = _tiny_vocabulary()
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        model_max_length=128,
    )
    for target in targets:
        tokenizer.save_pretrained(target)
    return len(vocabulary)


def _qwen_config_class():
    try:
        from transformers import Qwen3Config

        return Qwen3Config
    except ImportError:
        try:
            from transformers import Qwen2Config

            return Qwen2Config
        except ImportError as exc:
            raise RuntimeError(
                "Installed transformers does not provide a Qwen causal-LM config."
            ) from exc


def _save_tiny_model(
    target: Path,
    *,
    layers: int,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    seed: int,
    vocab_size: int,
) -> str:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "The offline tiny smoke requires the neural-forge 'hf' extra."
        ) from exc

    config_class = _qwen_config_class()
    config = config_class(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=8,
        max_position_embeddings=128,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=1,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        use_cache=False,
    )
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(target, safe_serialization=True)
    return config.model_type


def prepare_tiny_fixture(root: Path, *, seed: int = 0) -> TinyFixture:
    """Create a 4-layer source and 2-layer blank child without hub access."""

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source_qwen_4l"
    blank_candidate = root / "blank_qwen_2l"
    output = root / "composed_qwen_2l"
    source.mkdir()
    blank_candidate.mkdir()
    vocabulary_size = len(_tiny_vocabulary())
    architecture = _save_tiny_model(
        source,
        layers=4,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        seed=seed,
        vocab_size=vocabulary_size,
    )
    blank_architecture = _save_tiny_model(
        blank_candidate,
        layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        seed=seed + 1,
        vocab_size=vocabulary_size,
    )
    if blank_architecture != architecture:
        raise RuntimeError(
            "source and blank tiny checkpoints selected different Qwen config families."
        )
    _build_local_tokenizer((source, blank_candidate))

    calibration_file = root / "calibration.jsonl"
    calibration_rows = [
        {
            "text": "Q What is two plus three A",
            "prompt": "Q: What is two plus three? A:",
            "response": "five",
        },
        {
            "text": "Write a small function return",
            "prompt": "Write a small function.",
            "response": "return",
        },
        {
            "text": "Explain model compression",
            "prompt": "Explain model compression.",
            "response": "safe answer",
        },
        {
            "text": "layer pair attention token",
            "prompt": "Describe a layer pair.",
            "response": "attention",
        },
        {
            "text": "Q two plus two A",
            "prompt": "Q: two plus two? A:",
            "response": "answer",
        },
        {
            "text": "model layer pair safe",
            "prompt": "Give a safe answer.",
            "response": "safe",
        },
    ]
    calibration_file.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in calibration_rows),
        encoding="utf-8",
    )
    return TinyFixture(
        root=root,
        source=source,
        blank_candidate=blank_candidate,
        calibration_file=calibration_file,
        output=output,
        architecture=architecture,
    )


def build_tiny_command(
    fixture: TinyFixture,
    *,
    python: str | Path = sys.executable,
    device: str = "cpu",
) -> list[str]:
    settings = ComposeSettings(
        ridge=1e-2,
        max_length=32,
        max_tokens_per_layer=64,
        max_anchors=64,
        max_calibration_texts=6,
        fit_gate_validation_fraction=0.5,
        fit_gate_min_validation_rows=4,
        fit_gate_min_relative_improvement=0.0,
        fit_gate_max_cosine_drop=0.05,
        dtype="float32",
        candidate_dtype="float32",
        device=device,
    )
    return build_compose_command(
        python=python,
        source=fixture.source,
        blank_candidate=fixture.blank_candidate,
        output=fixture.output,
        calibration_file=fixture.calibration_file,
        settings=settings,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/qwen_pair_compose_tiny_smoke")
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.update(offline_environment())
    fixture = prepare_tiny_fixture(args.output_dir, seed=args.seed)
    command = build_tiny_command(fixture, python=args.python, device=args.device)
    manifest = {
        "schema": "neural-forge-qwen-pair-compose-tiny-smoke-v1",
        "offline": True,
        "architecture": fixture.architecture,
        "source": str(fixture.source),
        "blank_candidate": str(fixture.blank_candidate),
        "calibration_file": str(fixture.calibration_file),
        "output": str(fixture.output),
        "command": command,
        "prepare_only": bool(args.prepare_only),
    }
    manifest_path = fixture.root / "smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.prepare_only:
        print(json.dumps(manifest, sort_keys=True))
        return 0

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=offline_environment(),
        stdin=subprocess.DEVNULL,
        close_fds=True,
        check=False,
    )
    if completed.returncode != 0:
        return int(completed.returncode)
    if not (fixture.output / "config.json").is_file():
        raise RuntimeError(
            "pair-composition smoke completed without an output checkpoint."
        )
    print(json.dumps({**manifest, "returncode": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
