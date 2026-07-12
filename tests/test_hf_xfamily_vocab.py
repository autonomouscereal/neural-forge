from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from neural_forge.hf_xfamily import cross_family_vocab_merge


def test_cross_family_vocab_merge_executes_on_tiny_distinct_models(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")

    host = tmp_path / "host"
    donor = tmp_path / "donor"
    output = tmp_path / "merged"
    _save_tiny_checkpoint(
        torch,
        transformers,
        tokenizers,
        host,
        hidden_size=16,
        seed=17,
    )
    _save_tiny_checkpoint(
        torch,
        transformers,
        tokenizers,
        donor,
        hidden_size=24,
        seed=29,
    )

    report = cross_family_vocab_merge(
        host_model=str(host),
        donor_model=str(donor),
        output_dir=output,
        alpha=0.25,
        ridge=1e-3,
        torch_dtype="float32",
    )

    assert report["operation"] == "cross_family_vocab_merge"
    assert report["host_hidden_size"] == 16
    assert report["donor_hidden_size"] == 24
    assert report["anchors_used"] >= 4
    assert report["lm_head_updated"] is True
    assert (output / "model.safetensors").is_file()
    assert (output / "neural_forge_xfamily_vocab_report.json").is_file()
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(output)
    assert reloaded.config.n_embd == 16


def _save_tiny_checkpoint(
    torch: Any,
    transformers: Any,
    tokenizers: Any,
    output: Path,
    *,
    hidden_size: int,
    seed: int,
) -> None:
    vocabulary = {
        "<unk>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<pad>": 3,
        "hello": 4,
        "world": 5,
        " math": 6,
        " code": 7,
    }
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocabulary, unk_token="<unk>")
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    torch.manual_seed(seed)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=32,
            n_ctx=32,
            n_embd=hidden_size,
            n_layer=1,
            n_head=2,
            n_inner=hidden_size * 2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
            tie_word_embeddings=False,
        )
    )
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
