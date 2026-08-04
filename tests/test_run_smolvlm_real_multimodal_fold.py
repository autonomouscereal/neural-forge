from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_smolvlm_real_multimodal_fold.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_smolvlm_real_multimodal_fold",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


VISION_CONFIG = {
    "model_type": "smolvlm_vision",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "patch_size": 8,
    "image_size": 16,
    "num_channels": 3,
    "hidden_act": "gelu_pytorch_tanh",
    "layer_norm_eps": 1e-6,
}
TEXT_CONFIG = {
    "model_type": "llama",
    "hidden_size": 24,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 3,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "vocab_size": 40,
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 64,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "mlp_bias": False,
    "hidden_act": "silu",
    "rope_theta": 10000.0,
}


def _tiny_smolvlm():
    config = transformers.SmolVLMConfig(
        vision_config=dict(VISION_CONFIG),
        text_config=dict(TEXT_CONFIG),
        scale_factor=2,
        image_token_id=39,
        vocab_size=TEXT_CONFIG["vocab_size"],
        tie_word_embeddings=False,
    )
    model = transformers.SmolVLMForConditionalGeneration(config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Seal schema
# ---------------------------------------------------------------------------


def test_receipt_seal_round_trips_and_detects_tampering() -> None:
    body = {"schema": runner.RUNNER_SCHEMA, "stage": "configuration", "layers": 3}
    sealed = runner.seal_receipt(body)

    assert runner.verify_receipt(sealed)
    assert sealed["receipt_sha256"] != ""
    assert runner.seal_receipt(sealed) == sealed

    tampered = {**sealed, "layers": 4}
    assert not runner.verify_receipt(tampered)


def test_receipt_files_are_canonical_and_reread_under_their_seal(tmp_path) -> None:
    path = tmp_path / "configuration.json"
    sealed = runner._write_receipt(path, {"schema": runner.RUNNER_SCHEMA, "value": 1})

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document == sealed
    assert path.read_bytes().endswith(b"\n")
    assert runner._read_receipt(path) == sealed

    path.write_text(
        json.dumps({**sealed, "value": 2}, indent=2, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(runner.SmolVlmFoldError, match="Receipt seal is broken"):
        runner._read_receipt(path)


def test_verify_report_checks_every_manifest_digest_and_stage_seal(tmp_path) -> None:
    runner._write_receipt(
        tmp_path / "configuration.json",
        {"schema": runner.RUNNER_SCHEMA, "stage": "configuration"},
    )
    runner._write_receipt(
        tmp_path / "behavior.json",
        {"schema": runner.RUNNER_SCHEMA, "stage": "behavior"},
    )
    runner._write_receipt(
        tmp_path / "manifest.json",
        {
            "schema": runner.RUNNER_SCHEMA,
            "stage": "manifest",
            "files": runner._tree_manifest(tmp_path, skip=("manifest.json",)),
        },
    )

    result = runner.verify_report(tmp_path)
    assert result["verified"] is True
    assert result["problems"] == []
    assert result["files"] == 2

    (tmp_path / "behavior.json").write_text("{}", encoding="utf-8")
    broken = runner.verify_report(tmp_path)
    assert broken["verified"] is False
    assert any("behavior.json" in problem for problem in broken["problems"])


def test_verify_report_refuses_a_run_that_never_sealed_a_manifest(tmp_path) -> None:
    with pytest.raises(runner.SmolVlmFoldError, match="No sealed manifest"):
        runner.verify_report(tmp_path)


def test_stage_names_cover_every_receipt_the_runner_writes() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for stage in runner.STAGES:
        assert f'root / "{stage}.json"' in source
    assert "manifest" not in runner.STAGES


# ---------------------------------------------------------------------------
# Deterministic probes
# ---------------------------------------------------------------------------


def test_probe_records_are_deterministic_and_integer_valued() -> None:
    first = runner.build_probe_records(16)
    second = runner.build_probe_records(16)

    assert [row["image_sha256"] for row in first] == [
        row["image_sha256"] for row in second
    ]
    assert [row["probe_id"] for row in first] == [row["probe_id"] for row in second]
    assert len({row["probe_id"] for row in first}) == 16
    assert {row["category"] for row in first} == {
        "color",
        "count",
        "spatial",
        "split",
    }
    for row in first:
        image = row["image"]
        assert image.mode == "RGB"
        assert image.size == (runner.PROBE_EDGE, runner.PROBE_EDGE)
        assert all(
            isinstance(value, int)
            for pixel in image.convert("RGB").tobytes()
            for value in (pixel,)
        )
        assert row["prompt"].strip()
        assert row["expected_answer"].strip()


def test_probe_records_refuse_a_nonpositive_count() -> None:
    with pytest.raises(runner.SmolVlmFoldError, match="positive integer"):
        runner.build_probe_records(0)


def test_probe_prefix_is_stable_as_the_probe_count_grows() -> None:
    small = runner.build_probe_records(8)
    large = runner.build_probe_records(16)

    assert [row["image_sha256"] for row in small] == [
        row["image_sha256"] for row in large[:8]
    ]


# ---------------------------------------------------------------------------
# Checkpoint identity
# ---------------------------------------------------------------------------


def test_parent_identity_refuses_a_directory_without_the_required_files(tmp_path):
    with pytest.raises(runner.SmolVlmFoldError, match="needs config.json"):
        runner.pretrained_parent_identity(tmp_path)


def test_parent_identity_refuses_a_checkpoint_that_is_not_the_declared_one(tmp_path):
    for name in (
        runner.PRETRAINED_WEIGHT_FILE,
        runner.PRETRAINED_TOKENIZER_FILE,
        runner.PRETRAINED_PREPROCESSOR_FILE,
        runner.PRETRAINED_CHAT_TEMPLATE_FILE,
    ):
        (tmp_path / name).write_bytes(b"{}")
    payload = {
        **runner.DECLARED_COMPOSITE_CONFIG,
        "text_config": {**runner.DECLARED_TEXT_CONFIG, "hidden_size": 512},
        "vision_config": dict(runner.DECLARED_VISION_CONFIG),
    }
    (tmp_path / runner.PRETRAINED_CONFIG_FILE).write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(runner.SmolVlmFoldError, match="text_config.hidden_size"):
        runner.pretrained_parent_identity(tmp_path)


# ---------------------------------------------------------------------------
# Decoder view and blank child
# ---------------------------------------------------------------------------


def test_decoder_view_shares_storage_and_proves_the_causal_layout() -> None:
    from neural_forge.hf_arch_layout import build_decoder_layout

    model = _tiny_smolvlm()
    view = runner.SmolVlmTextDecoderModule(model)

    assert view.model is model.model.text_model
    assert view.lm_head is model.lm_head
    assert (
        view.get_parameter("model.embed_tokens.weight")
        is model.model.text_model.embed_tokens.weight
    )
    layout = build_decoder_layout(view)
    assert layout.family == "llama"
    assert layout.hidden_size == TEXT_CONFIG["hidden_size"]
    assert layout.lm_head_path == "lm_head"
    assert layout.layer_count == TEXT_CONFIG["num_hidden_layers"]

    ids = torch.zeros((1, 4), dtype=torch.long)
    with torch.no_grad():
        logits = view(input_ids=ids).logits
    assert tuple(logits.shape) == (1, 4, TEXT_CONFIG["vocab_size"])


def test_blank_child_is_narrower_untied_and_keeps_the_vision_geometry() -> None:
    parent = _tiny_smolvlm()
    child = runner.build_child(
        parent,
        hidden_size=16,
        intermediate_size=24,
        query_heads=2,
        key_value_heads=1,
        layers=2,
        seed=7,
    )

    assert int(child.config.text_config.hidden_size) == 16
    assert int(child.config.text_config.num_hidden_layers) == 2
    assert child.model.text_model.embed_tokens.weight is not child.lm_head.weight
    assert child.config.vision_config.to_dict() == (
        parent.config.vision_config.to_dict()
    )
    assert {name for name, _value in child.model.vision_model.named_parameters()} == {
        name for name, _value in parent.model.vision_model.named_parameters()
    }
    assert tuple(child.model.connector.modality_projection.proj.weight.shape) == (
        16,
        int(parent.config.vision_config.hidden_size) * parent.config.scale_factor**2,
    )


def test_image_merge_hook_fails_closed_on_an_unsealed_batch() -> None:
    model = _tiny_smolvlm()
    sealed = torch.zeros((2, 5), dtype=torch.long)
    features = torch.zeros((1, 1, TEXT_CONFIG["hidden_size"]))
    hook = runner.ImageTokenMergeHook(
        model.model, input_ids=sealed, image_features=features
    )

    other = torch.ones((2, 5), dtype=torch.long)
    with pytest.raises(runner.SmolVlmFoldError, match="not sealed against"):
        hook(None, (other,), torch.zeros((2, 5, TEXT_CONFIG["hidden_size"])))
    assert hook.calls == 0


# ---------------------------------------------------------------------------
# Command surface
# ---------------------------------------------------------------------------


def test_parser_defaults_declare_the_bounded_prefix_and_child_geometry() -> None:
    args = runner.build_parser().parse_args(["--report-root", "reports/x"])

    assert args.layers == 3
    assert args.child_hidden_size == 432
    assert args.child_intermediate_size == 1152
    assert args.child_query_heads == 6
    assert args.child_key_value_heads == 3
    assert args.image_seq_len == 64
    assert args.calibration_probe_images < args.probe_images
    assert args.verify_only is False


def test_verify_only_path_reports_a_missing_report_as_a_closed_failure(tmp_path):
    assert runner.main(["--report-root", str(tmp_path), "--verify-only"]) == 1
