from __future__ import annotations

import json

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from neural_forge.hf_blank_checkpoint import (  # noqa: E402
    REPORT_FILENAME,
    BlankCheckpointError,
    bootstrap_blank_decoder_checkpoint,
)


def _tokenizer():
    words = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] + [f"w{i}" for i in range(60)]
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


def _gpt2_source():
    torch.manual_seed(101)
    return transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_embd=12,
            n_layer=3,
            n_head=3,
            n_inner=20,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    )


def _qwen3_source():
    if not hasattr(transformers, "Qwen3ForCausalLM"):
        pytest.skip("Transformers build has no Qwen3 support")
    torch.manual_seed(202)
    return transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=64,
            hidden_size=12,
            intermediate_size=20,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            rope_parameters={"rope_theta": 1_000_000.0, "rope_type": "default"},
            max_position_embeddings=32,
            use_sliding_window=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
    )


def _prompts(offset: int) -> list[str]:
    return [
        " ".join(f"w{(offset + row + column) % 50}" for column in range(8))
        for row in range(8)
    ]


def _run(source, output, **overrides):
    values = {
        "target_family": "llama",
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "max_position_embeddings": 32,
        "output_dir": output,
        "calibration_prompts": _prompts(0),
        "validation_prompts": _prompts(7),
        "tokenizer": _tokenizer(),
        "seed": 303,
        "local_files_only": True,
        "max_coordinate_rows": 24,
        "vocab_projection_chunk_size": 7,
    }
    values.update(overrides)
    return bootstrap_blank_decoder_checkpoint(source, **values)


@pytest.mark.parametrize("source_factory", [_qwen3_source, _gpt2_source])
def test_cross_family_checkpoint_is_accepted_and_reload_exact(
    tmp_path, source_factory
) -> None:
    output = tmp_path / source_factory.__name__
    report = _run(source_factory(), output)

    assert report["accepted"] is True, report["bootstrap"]["gate"]
    if source_factory is _qwen3_source:
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        assert config["rope_parameters"]["rope_theta"] == pytest.approx(1_000_000.0)
    assert report["blank_target"]["constructor"].startswith("LlamaForCausalLM(")
    assert report["blank_target"]["pretrained_child_weights_loaded"] is False
    assert report["compression"]["vocabulary_body"]["body_source_to_child_ratio"] > 1.0
    assert report["execution"]["training_steps"] == 0
    assert report["execution"]["zero_gradients"] is True
    assert report["bootstrap"]["knobs"]["residual_map_method"] == "sparse_fold"
    assert report["bootstrap"]["residual_coordinates"]["full_source_coverage"] is True
    assert report["bootstrap"]["residual_coordinates"]["dropped_channels"] == 0
    assert report["bootstrap"]["fit_reports"]
    assert report["checkpoint_verification"]["exact_output_equality"] is True
    assert (
        report["checkpoint_verification"]["validation_logits_sha256_before_save"]
        == report["checkpoint_verification"]["validation_logits_sha256_after_reload"]
    )
    assert list(output.glob("*.safetensors"))
    assert not list(output.glob("*.bin"))

    reloaded = transformers.LlamaForCausalLM.from_pretrained(
        output, local_files_only=True
    )
    assert (
        report["candidate"]["state_dict_sha256"]
        == report["checkpoint_verification"]["state_dict_sha256_after_reload"]
    )
    assert (
        sum(parameter.numel() for parameter in reloaded.parameters())
        == report["candidate"]["parameter_counts"]["unique_parameter_count"]
    )


def test_rejected_candidate_is_not_saved(tmp_path) -> None:
    donor = tmp_path / "donor"
    _gpt2_source().save_pretrained(donor, safe_serialization=True)
    _tokenizer().save_pretrained(donor)
    output = tmp_path / "rejected"
    report = _run(
        donor,
        output,
        tokenizer=None,
        min_final_state_relative_improvement=10.0,
        min_logit_relative_improvement=10.0,
        min_top1_agreement_gain=1.0,
    )

    assert report["accepted"] is False
    assert report["candidate"] == {
        "saved": False,
        "state_restored_to_blank": True,
    }
    assert report["bootstrap"]["rollback"] == {"performed": True, "exact": True}
    assert report["source"]["pretrained_load_calls"] == 1
    assert report["tokenizer"]["defaulted_to_donor"] is True
    assert [path.name for path in output.iterdir()] == [REPORT_FILENAME]
    saved_report = json.loads((output / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert saved_report == report


def test_shape_and_exact_parameter_budget_checks_fail_before_publication(
    tmp_path,
) -> None:
    with pytest.raises(BlankCheckpointError, match="Qwen3 requires"):
        _run(
            _gpt2_source(),
            tmp_path / "invalid-qwen3",
            target_family="qwen3",
        )
    assert not (tmp_path / "invalid-qwen3").exists()

    with pytest.raises(BlankCheckpointError, match="exact_parameter_budget"):
        _run(
            _gpt2_source(),
            tmp_path / "wrong-budget",
            exact_parameter_budget=1,
        )
    assert not (tmp_path / "wrong-budget").exists()
