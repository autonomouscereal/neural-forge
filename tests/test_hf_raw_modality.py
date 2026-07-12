from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neural_forge.cli import build_parser, cmd_hf_raw_token_expand
from neural_forge.hf_raw_modality import (
    RAW_MODALITY_REPORT_FILENAME,
    expand_hf_raw_token_checkpoint,
    expand_hf_raw_token_model_in_memory,
)
from neural_forge.raw_modality_tokens import (
    RAW_TOKEN_STRINGS,
    codec_from_hf_tokenizer,
)


def test_hf_raw_token_expand_cli_parser_defaults_and_options() -> None:
    parser = build_parser()
    defaults = parser.parse_args(["hf-raw-token-expand", "model-a", "output-a"])

    assert defaults.model == "model-a"
    assert defaults.output == "output-a"
    assert defaults.tokenizer_model is None
    assert defaults.dtype == "auto"
    assert defaults.device == "cpu"
    assert defaults.seed == 0
    assert defaults.trust_remote_code is False
    assert defaults.func is cmd_hf_raw_token_expand

    explicit = parser.parse_args(
        [
            "hf-raw-token-expand",
            "model-b",
            "output-b",
            "--tokenizer-model",
            "tokenizer-b",
            "--dtype",
            "bfloat16",
            "--device",
            "cuda:1",
            "--seed",
            "41",
            "--trust-remote-code",
        ]
    )
    assert explicit.tokenizer_model == "tokenizer-b"
    assert explicit.dtype == "bfloat16"
    assert explicit.device == "cuda:1"
    assert explicit.seed == 41
    assert explicit.trust_remote_code is True


def test_in_memory_expansion_is_seeded_deterministic_and_adds_no_modules() -> None:
    torch, transformers, tokenizers = _hf_stack()
    tokenizer_a = _tiny_tokenizer(transformers, tokenizers)
    tokenizer_b = _tiny_tokenizer(transformers, tokenizers)
    model_a = _tiny_model(torch, transformers, len(tokenizer_a), seed=17)
    model_b = _tiny_model(torch, transformers, len(tokenizer_b), seed=17)
    original_rows = len(tokenizer_a)
    original_input_rows = model_a.get_input_embeddings().weight.detach().clone()
    original_output_rows = model_a.get_output_embeddings().weight.detach().clone()

    report_a = expand_hf_raw_token_model_in_memory(
        model_a,
        tokenizer_a,
        seed=1234,
    )
    report_b = expand_hf_raw_token_model_in_memory(
        model_b,
        tokenizer_b,
        seed=1234,
    )

    assert report_a["row_initialization"]["seeded_deterministic"] is True
    assert (
        report_a["row_initialization"]["combined_raw_rows_sha256"]
        == report_b["row_initialization"]["combined_raw_rows_sha256"]
    )
    assert report_a["raw_token_ids"] == report_b["raw_token_ids"]
    assert torch.equal(
        model_a.get_input_embeddings().weight,
        model_b.get_input_embeddings().weight,
    )
    assert torch.equal(
        model_a.get_output_embeddings().weight,
        model_b.get_output_embeddings().weight,
    )
    assert torch.equal(
        model_a.get_input_embeddings().weight[:original_rows],
        original_input_rows,
    )
    assert torch.equal(
        model_a.get_output_embeddings().weight[:original_rows],
        original_output_rows,
    )
    assert (
        model_a.get_input_embeddings().weight.data_ptr()
        != model_a.get_output_embeddings().weight.data_ptr()
    )
    assert report_a["output_head"]["is_untied_after"] is True
    assert (
        report_a["expected_vocabulary_only_parameter_delta"]
        == report_a["actual_vocabulary_only_parameter_delta"]
    )
    assert report_a["parameter_count_after"] == (
        report_a["parameter_count_after_untie"]
        + report_a["expected_vocabulary_only_parameter_delta"]
    )
    inventory = report_a["architecture_inventory"]
    assert inventory["module_names_unchanged"] is True
    assert inventory["module_types_unchanged"] is True
    assert inventory["added_modules"] == []
    assert inventory["removed_modules"] == []
    assert inventory["module_type_changes"] == []
    assert inventory["only_embedding_and_head_rows_changed"] is True
    assert report_a["zero_gradients"] is True
    assert report_a["gradients_after"]["nonzero_gradient_element_count"] == 0
    assert report_a["operation_bounds"] == {
        "learned_modality_modules_added": [],
        "encoder_added": False,
        "decoder_added": False,
        "projector_added": False,
        "adapter_added": False,
        "bridge_added": False,
        "modality_specific_tower_added": False,
        "optimizer_created": False,
        "backward_called": False,
        "training_performed": False,
        "forward_called": False,
        "only_head_untie_tokenizer_and_vocabulary_rows_mutated": True,
    }


def test_tiny_hf_checkpoint_save_reload_and_all_raw_codec_roundtrips(
    tmp_path: Path,
) -> None:
    torch, transformers, tokenizers = _hf_stack()
    source = tmp_path / "source"
    output = tmp_path / "expanded"
    tokenizer = _tiny_tokenizer(transformers, tokenizers)
    source_model = _tiny_model(torch, transformers, len(tokenizer), seed=29)
    source_modules = _module_name_types(source_model)
    source_model.save_pretrained(source, safe_serialization=True)
    tokenizer.save_pretrained(source)

    report = expand_hf_raw_token_checkpoint(
        source,
        output,
        torch_dtype="float32",
        device="cpu",
        seed=907,
    )

    report_path = output / RAW_MODALITY_REPORT_FILENAME
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert RAW_MODALITY_REPORT_FILENAME in report["saved_artifacts"]
    assert report["parameter_count_before"] == sum(
        parameter.numel() for parameter in source_model.parameters()
    )
    assert (
        report["actual_vocabulary_only_parameter_delta"]
        == report["expected_vocabulary_only_parameter_delta"]
    )
    assert len(report["raw_token_ids"]) == len(RAW_TOKEN_STRINGS)
    assert len(set(report["raw_token_ids"].values())) == len(RAW_TOKEN_STRINGS)
    assert len(report["raw_token_ids_sha256"]) == 64
    assert report["architecture_inventory"]["added_modules"] == []
    assert report["architecture_inventory"]["removed_modules"] == []
    assert report["architecture_inventory"]["module_type_changes"] == []

    reloaded_model = transformers.AutoModelForCausalLM.from_pretrained(output)
    reloaded_tokenizer = transformers.AutoTokenizer.from_pretrained(output)
    assert _module_name_types(reloaded_model) == source_modules
    assert (
        sum(parameter.numel() for parameter in reloaded_model.parameters())
        == report["parameter_count_after"]
    )
    assert (
        reloaded_model.get_input_embeddings().weight.data_ptr()
        != reloaded_model.get_output_embeddings().weight.data_ptr()
    )
    assert reloaded_model.config.tie_word_embeddings is False
    assert reloaded_tokenizer.get_vocab() == {
        **reloaded_tokenizer.get_vocab(),
        **report["raw_token_ids"],
    }

    codec = codec_from_hf_tokenizer(reloaded_tokenizer)
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)[::-1]
    decoded_image = codec.decode_image(codec.encode_image(image))
    assert decoded_image.shape == image.shape
    assert decoded_image.dtype == image.dtype
    assert decoded_image.tobytes(order="C") == np.ascontiguousarray(image).tobytes()

    audio = np.array([[-32768, -7], [0, 7], [32767, 19]], dtype=np.int16)
    decoded_audio, sample_rate = codec.decode_audio(
        codec.encode_audio(audio, sample_rate=48_000)
    )
    assert sample_rate == 48_000
    assert decoded_audio.dtype == audio.dtype
    assert decoded_audio.tobytes() == audio.tobytes()

    video = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
    decoded_video = codec.decode_video(codec.encode_video(video, frame_rate=24.0))
    assert decoded_video.shape == video.shape
    assert decoded_video.dtype == video.dtype
    assert decoded_video.tobytes() == video.tobytes()

    ocr = "invoice\x00OCR\nline 2: caf\u00e9"
    decoded_ocr = codec.decode_ocr(codec.encode_ocr(ocr))
    assert decoded_ocr.encode("utf-8") == ocr.encode("utf-8")

    points = np.array(
        [[-(2**31), -7, 0], [3, 19, 2**31 - 1]],
        dtype=np.int32,
    )
    decoded_points, quantization = codec.decode_points3d(
        codec.encode_points3d(
            points,
            scale=(0.25, 0.5, 2.0),
            zero_point=(-3, 0, 9),
        )
    )
    assert decoded_points.dtype == points.dtype
    assert decoded_points.tobytes() == points.tobytes()
    assert quantization.scale == (0.25, 0.5, 2.0)
    assert quantization.zero_point == (-3, 0, 9)


def _hf_stack() -> tuple[Any, Any, Any]:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    return torch, transformers, tokenizers


def _tiny_tokenizer(transformers: Any, tokenizers: Any) -> Any:
    vocabulary = {
        "<unk>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<pad>": 3,
        "hello": 4,
        "world": 5,
    }
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocabulary, unk_token="<unk>")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )


def _tiny_model(
    torch: Any,
    transformers: Any,
    vocabulary_size: int,
    *,
    seed: int,
) -> Any:
    torch.manual_seed(seed)
    config = transformers.GPT2Config(
        vocab_size=vocabulary_size,
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_inner=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        tie_word_embeddings=True,
    )
    model = transformers.GPT2LMHeadModel(config)
    assert (
        model.get_input_embeddings().weight.data_ptr()
        == model.get_output_embeddings().weight.data_ptr()
    )
    return model


def _module_name_types(model: Any) -> list[tuple[str, str]]:
    return [
        (name, f"{type(module).__module__}.{type(module).__qualname__}")
        for name, module in model.named_modules()
    ]
