from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from neural_forge.hf_blank_checkpoint import (  # noqa: E402
    REPORT_FILENAME,
    BlankCheckpointError,
    bootstrap_blank_decoder_checkpoint,
)
import neural_forge.hf_blank_checkpoint as checkpoint_module  # noqa: E402


def _tokenizer(words=None, *, whitespace_split=False):
    if words is None:
        words = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] + [f"w{i}" for i in range(60)]
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {word: index for index, word in enumerate(words)},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = (
        tokenizers.pre_tokenizers.WhitespaceSplit()
        if whitespace_split
        else tokenizers.pre_tokenizers.Whitespace()
    )
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


def _cross_prompts(offset: int) -> list[str]:
    return [
        " ".join(
            f"w{(offset + row + column) % 16}-w{(offset + row + column + 1) % 16}"
            for column in range(4)
        )
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
    assert report["bootstrap"]["knobs"]["nonlinear_merge_method"] == "activation_fold"
    assert report["bootstrap"]["knobs"]["vocab_row_norm_repair"] == "none"
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


def test_shared_tokenizer_preserves_donor_embedding_padding_rows(tmp_path) -> None:
    short_tokenizer = _tokenizer(
        ["[PAD]", "[BOS]", "[EOS]", "[UNK]", *[f"w{i}" for i in range(16)]]
    )
    report = _run(
        _gpt2_source(),
        tmp_path / "shared-tokenizer-padding",
        tokenizer=short_tokenizer,
    )

    assert report["tokenizers"]["mode"] == "shared-tokenizer"
    assert report["tokenizers"]["target"]["length"] == 20
    assert report["blank_target"]["architecture"]["vocab_size"] == 64
    assert report["token_transport"]["profile"] == "shared-token-id-identity"


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


def test_real_medoid_is_forwarded_to_model_bootstrap_and_reported(
    tmp_path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_bootstrap(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            transaction=SimpleNamespace(
                accepted=False,
                report={
                    "gate": {"accepted": False},
                    "coverage": {"complete": True},
                    "rollback": {"performed": True, "exact": True},
                },
            ),
            donor_layer_spans=(),
            fit_reports={"teacher_logits.target_space": {"space": "target"}},
            residual_coordinates=SimpleNamespace(report=None),
        )

    monkeypatch.setattr(
        checkpoint_module,
        "bootstrap_blank_decoder_models",
        fake_bootstrap,
    )
    report = _run(
        _gpt2_source(),
        tmp_path / "real-medoid",
        nonlinear_merge_method="real_medoid",
    )

    assert captured["nonlinear_merge_method"] == "real_medoid"
    assert report["bootstrap"]["knobs"]["nonlinear_merge_method"] == "real_medoid"


def test_smaller_reordered_cross_tokenizer_checkpoint_is_exact_after_reload(
    tmp_path,
) -> None:
    target_words = [
        "w7",
        "[EOS]",
        "w1",
        "[PAD]",
        "[UNK]",
        "[BOS]",
        *[f"w{index}" for index in range(16) if index not in {1, 7}],
    ]
    target_tokenizer = _tokenizer(target_words, whitespace_split=True)
    output = tmp_path / "cross-tokenizer"

    report = _run(
        _qwen3_source(),
        output,
        donor_tokenizer=_tokenizer(),
        target_tokenizer=target_tokenizer,
        tokenizer=None,
        calibration_prompts=_cross_prompts(0),
        validation_prompts=_cross_prompts(3),
        seed=304,
        trace_pooling="mean",
        residual_map_method="pca",
        fit_full_lm_head=True,
        vocab_row_norm_repair="source_l2",
    )

    assert report["schema"] == "neural-forge-blank-checkpoint-v2"
    assert report["accepted"] is True, report["bootstrap"]["gate"]
    assert report["tokenizers"]["mode"] == "cross-tokenizer"
    assert report["tokenizers"]["donor"]["length"] == 64
    assert report["tokenizers"]["target"]["length"] == len(target_words)
    assert report["blank_target"]["architecture"]["vocab_size"] == len(target_words)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["bos_token_id"] == 5
    assert config["eos_token_id"] == 1
    assert config["pad_token_id"] == 3
    assert config["unk_token_id"] == 4
    transport = report["token_transport"]
    assert transport["profile"] == "strict-cross-tokenizer-one-row-v1"
    assert len(transport["row_mappings"]) == len(target_words)
    assert transport["target_to_donor_row_indices"][:6] == [11, 2, 5, 0, 3, 1]
    assert transport["alignment_fingerprint"]
    assert transport["profile_fingerprint"]
    teacher_contract = report["bootstrap"]["target_space_teacher_logits"]
    assert teacher_contract["space"] == "target-tokenizer-columns"
    assert teacher_contract["source_shape"][1] == 64
    assert teacher_contract["target_shape"][1] == len(target_words)
    attention_prior = report["bootstrap"]["fit_reports"][
        "block.0.attention.structural_prior"
    ]
    mlp_prior = report["bootstrap"]["fit_reports"]["block.0.mlp.structural_prior"]
    assert attention_prior["applied"] is True
    assert attention_prior["coordinate_method"] == "pca-coordinate-formula"
    assert mlp_prior["applied"] is True
    assert mlp_prior["coordinate_method"] == "pca-coordinate-formula"
    donor_shape = report["tokenization"]["donor"]["validation"]["input_shape"]
    target_shape = report["tokenization"]["target"]["validation"]["input_shape"]
    assert donor_shape[0] == target_shape[0] == 8
    assert donor_shape[1] != target_shape[1]
    verification = report["checkpoint_verification"]
    assert verification["target_tokenizer_identity_exact"] is True
    assert verification["validation_retokenization_exact"] is True
    assert (
        verification["validation_retokenization_hashes_before_save"]
        == verification["validation_retokenization_hashes_after_reload"]
    )

    reloaded_tokenizer = transformers.AutoTokenizer.from_pretrained(
        output, local_files_only=True
    )
    assert reloaded_tokenizer.get_vocab() == target_tokenizer.get_vocab()


def test_cross_tokenizer_alignment_failure_prevents_publication(tmp_path) -> None:
    target_words = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "missing-token"]
    output = tmp_path / "unalignable"

    with pytest.raises(BlankCheckpointError, match="Strict token transport"):
        _run(
            _gpt2_source(),
            output,
            donor_tokenizer=_tokenizer(),
            target_tokenizer=_tokenizer(target_words),
            tokenizer=None,
            trace_pooling="mean",
        )

    assert not output.exists()


def test_shape_and_exact_parameter_budget_checks_fail_before_publication(
    tmp_path,
) -> None:
    with pytest.raises(BlankCheckpointError, match="nonlinear_merge_method"):
        _run(
            _gpt2_source(),
            tmp_path / "invalid-nonlinear-merge",
            nonlinear_merge_method="centroid",
        )
    assert not (tmp_path / "invalid-nonlinear-merge").exists()

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
