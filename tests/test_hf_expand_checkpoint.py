from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_forge.cli import build_parser
from neural_forge.hf_expand import count_unique_parameters
from neural_forge.hf_expand_checkpoint import (
    REPORT_FILENAME,
    HFExpandCheckpointError,
    expand_hf_checkpoint,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")


def _tokenizer():
    words = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] + [f"w{index}" for index in range(28)]
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {word: index for index, word in enumerate(words)},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )


def _model():
    torch.manual_seed(101)
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=32,
            n_positions=32,
            n_embd=12,
            n_layer=2,
            n_head=3,
            n_inner=20,
            activation_function="gelu_new",
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    )


def _source(tmp_path: Path) -> tuple[Path, object]:
    source = tmp_path / "source"
    model = _model()
    model.save_pretrained(source, safe_serialization=True)
    _tokenizer().save_pretrained(source)
    return source, model


@pytest.mark.parametrize("operation", ["mlp_width", "identity_depth"])
def test_local_checkpoint_expands_and_stock_reload_is_exact(
    tmp_path: Path, operation: str
) -> None:
    source, model = _source(tmp_path)
    source_count = count_unique_parameters(model)
    if operation == "mlp_width":
        target = source_count + model.config.n_layer * (2 * model.config.n_embd + 1)
    else:
        block_count = sum(
            parameter.numel() for parameter in model.transformer.h[0].parameters()
        )
        target = source_count + block_count
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(json.dumps(["w4 w5 w6", "w7 w8"]), encoding="utf-8")
    output = tmp_path / f"expanded-{operation}"

    report = expand_hf_checkpoint(
        source,
        output,
        target,
        dtype="float32",
        device="cpu",
        local_files_only=True,
        validation_prompts=["w1 w2 w3"],
        validation_prompts_file=prompt_file,
        max_equivalence_error=2e-6,
    )

    assert report["plan"]["operations"][0]["kind"] == operation
    assert report["plan"]["target_parameter_count"] == target
    assert report["validation"]["prompt_count"] == 3
    assert report["validation"]["function_equivalence_max_error"] <= 2e-6
    assert report["checkpoint_verification"]["exact_state_equality"] is True
    assert (
        report["checkpoint_verification"]["candidate_state_dict_sha256"]
        == report["checkpoint_verification"]["reloaded_state_dict_sha256"]
    )
    assert report["execution"]["training_performed"] is False
    assert list(output.glob("*.safetensors"))
    assert not list(output.glob("*.bin"))
    assert json.loads((output / REPORT_FILENAME).read_text(encoding="utf-8")) == report

    reloaded = transformers.AutoModelForCausalLM.from_pretrained(
        output, local_files_only=True
    )
    assert count_unique_parameters(reloaded) == target
    assert reloaded.config.n_layer == report["plan"]["target_layers"]
    assert reloaded.config.n_inner == report["plan"]["target_intermediate_size"]
    if operation == "identity_depth":
        assert report["provenance"][0]["donor_index"] == 0
        assert report["provenance"][0]["inserted_index"] == 2


def test_unreachable_budget_fails_without_publication(tmp_path: Path) -> None:
    source, model = _source(tmp_path)
    output = tmp_path / "unreachable"

    with pytest.raises(HFExpandCheckpointError, match="unreachable"):
        expand_hf_checkpoint(
            source,
            output,
            count_unique_parameters(model) + 1,
            local_files_only=True,
            validation_prompts=["w1 w2"],
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".unreachable.stage-*"))


def test_failed_independent_reload_removes_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, model = _source(tmp_path)
    output = tmp_path / "corrupt-reload"
    target = count_unique_parameters(model) + model.config.n_layer * (
        2 * model.config.n_embd + 1
    )
    original = transformers.AutoModelForCausalLM.from_pretrained

    def corrupt_stage_reload(model_path, *args, **kwargs):
        loaded = original(model_path, *args, **kwargs)
        if ".corrupt-reload.stage-" in str(model_path):
            with torch.no_grad():
                next(loaded.parameters()).view(-1)[0].add_(1)
        return loaded

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        corrupt_stage_reload,
    )

    with pytest.raises(HFExpandCheckpointError, match="tensor values differ"):
        expand_hf_checkpoint(
            source,
            output,
            target,
            local_files_only=True,
            validation_prompts=["w1 w2"],
            max_equivalence_error=2e-6,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".corrupt-reload.stage-*"))


def test_hf_expand_cli_wires_all_checkpoint_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import neural_forge.hf_expand_checkpoint as checkpoint_module

    captured = {}

    def fake_expand(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(checkpoint_module, "expand_hf_checkpoint", fake_expand)
    prompt_file = tmp_path / "prompts.txt"
    args = build_parser().parse_args(
        [
            "hf-expand",
            "local-source",
            "expanded-output",
            "--target-params",
            "123456",
            "--dtype",
            "bfloat16",
            "--device",
            "cuda:1",
            "--trust-remote-code",
            "--local-files-only",
            "--validation-prompt",
            "first prompt",
            "--validation-prompts",
            "second prompt",
            "--validation-prompts-file",
            str(prompt_file),
            "--max-equivalence-error",
            "0.002",
        ]
    )

    assert args.func(args) == {"ok": True}
    assert captured == {
        "source_model": "local-source",
        "output_dir": "expanded-output",
        "target_parameter_count": 123456,
        "dtype": "bfloat16",
        "device": "cuda:1",
        "trust_remote_code": True,
        "local_files_only": True,
        "validation_prompts": ["first prompt", "second prompt"],
        "validation_prompts_file": str(prompt_file),
        "max_equivalence_error": 0.002,
    }
