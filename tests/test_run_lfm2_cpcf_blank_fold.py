from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_lfm2_cpcf_blank_fold.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_cpcf_blank_fold",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--parent-model-directory",
        str(tmp_path / "parent"),
        "--donor-tokenizer-directory",
        str(tmp_path / "donor-tokenizer"),
        "--target-tokenizer-directory",
        str(tmp_path / "target-tokenizer"),
        "--bridge-directory",
        str(tmp_path / "bridge"),
        "--trace-cache-directory",
        str(tmp_path / "cache"),
        "--trajectory-corpus",
        str(tmp_path / "trajectories.json"),
        "--output-directory",
        str(tmp_path / "child"),
    ]


def test_parser_requires_one_severe_parameter_target(tmp_path: Path) -> None:
    parser = MODULE.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_required_args(tmp_path))
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--compression-ratio",
                "100",
                "--target-unique-parameters",
                "1000",
            ]
        )
    args = parser.parse_args(
        [*_required_args(tmp_path), "--compression-ratio", "100"]
    )
    config = MODULE.config_from_args(args)
    assert config.compression.compression_ratio == "100"
    assert config.compression.target_unique_parameters is None


@pytest.mark.parametrize("ratio", ["0.5", "1", "NaN"])
def test_noncompacting_ratio_is_refused(ratio: str) -> None:
    with pytest.raises(
        MODULE.Lfm2CpcfBlankFoldRunnerError,
        match="greater than one",
    ):
        MODULE.CompressionRequest(
            compression_ratio=ratio,
            target_unique_parameters=None,
        )


def test_exact_parameter_budget_remains_generic_until_promotion_audit() -> None:
    request = MODULE.CompressionRequest(
        compression_ratio=None,
        target_unique_parameters=1_000_000,
    )
    assert request.as_kwargs() == {
        "compression_ratio": None,
        "target_unique_parameters": 1_000_000,
    }
    control = MODULE.CompressionRequest(
        compression_ratio="10",
        target_unique_parameters=None,
    )
    assert control.compression_ratio == "10"
    assert MODULE.MINIMUM_CLAIMABLE_COMPRESSION_RATIO == 100


def test_trajectory_reader_preserves_whole_record_splits(
    tmp_path: Path,
) -> None:
    source = tmp_path / "trajectories.json"
    rows = [
        {
            "schema": MODULE.TRAJECTORY_SCHEMA,
            "id": "selection-1",
            "split": "selection",
            "input_text": "Pick one.",
            "forced_decode_text": "One.",
        },
        {
            "schema": MODULE.TRAJECTORY_SCHEMA,
            "id": "fit-1",
            "split": "fit",
            "input_text": "Add two numbers.",
            "forced_decode_text": "Four.",
        },
        {
            "schema": MODULE.TRAJECTORY_SCHEMA,
            "id": "validation-1",
            "split": "validation",
            "input_text": "Explain rain.",
            "forced_decode_text": "Water condenses.",
        },
    ]
    source.write_text(json.dumps(rows), encoding="utf-8")
    loaded = MODULE._read_trajectory_rows(source)
    assert tuple(row["split"] for row in loaded) == (
        "selection",
        "fit",
        "validation",
    )
    assert loaded[2]["forced_decode_text"] == "Water condenses."


def test_source_has_no_remote_loader_training_or_fallback_path() -> None:
    source = inspect.getsource(MODULE)
    assert ".from_pretrained(" not in source
    assert "AutoModel" not in source
    assert "AutoTokenizer" not in source
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "build_lfm2_circuit_trace_cache" not in source
    assert "capture_lfm2" not in source
    assert "parent.generate(" not in source
    assert "structural_dry_plan_only" in source
    assert "folded_child_created\": False" in source
    assert "marginal_compaction_reported_as_success" in source
    assert "--diagnostic-control" in source
    assert "compile_lfm2_cpcf_model_fold" in source


def test_direct_script_entry_resolves_sibling_experiment_imports(
    tmp_path: Path,
) -> None:
    probe = (
        "import importlib.util,sys;"
        f"path={str(SCRIPT)!r};"
        "spec=importlib.util.spec_from_file_location('cpcf_direct',path);"
        "module=importlib.util.module_from_spec(spec);"
        "sys.modules[spec.name]=module;"
        "spec.loader.exec_module(module);"
        "assert importlib.util.find_spec("
        "'experiments.run_lfm2_residual_attention_fold') is not None"
    )
    command = [
        sys.executable,
        "-c",
        probe,
    ]
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_repetition_proxy_is_diagnostic_not_intelligence_claim() -> None:
    healthy = MODULE._repetition_proxy([1, 2, 3, 2, 4, 5])
    collapsed = MODULE._repetition_proxy([7] * 12)
    assert healthy["passed"] is True
    assert collapsed["passed"] is False
    source = inspect.getsource(MODULE._standalone_whole_response_evaluation)
    assert "manual_intelligence_retention_proven" in source
    assert "False" in source


def test_local_lfm2_save_reload_preserves_and_restores_ffn_alias(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from experiments.run_lfm2_residual_attention_fold import (
        load_local_lfm2_model,
    )

    config = transformers.Lfm2Config(
        vocab_size=24,
        hidden_size=32,
        intermediate_size=47,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    model = transformers.Lfm2ForCausalLM(config)
    before = MODULE._restore_lfm2_ffn_config_alias(
        model,
        "test_target_before_save",
    )
    checkpoint = tmp_path / "child"
    model.save_pretrained(checkpoint, safe_serialization=True)
    saved = MODULE._verify_saved_lfm2_ffn_config(
        checkpoint,
        expected_width=47,
    )
    reloaded = load_local_lfm2_model(
        checkpoint,
        device="cpu",
        dtype=torch.float32,
    )
    consumed_alias = getattr(reloaded.config, "block_ff_dim", None)
    after = MODULE._restore_lfm2_ffn_config_alias(
        reloaded,
        "test_target_after_reload",
    )

    assert before["materialized_ffn_width"] == 47
    assert saved["serialized_aliases"]["block_ff_dim"] == 47
    assert consumed_alias is None
    assert after["previous_block_ff_dim"] is None
    assert reloaded.config.block_ff_dim == 47
    assert reloaded.config.to_dict()["block_ff_dim"] == 47
    assert {
        int(layer.feed_forward.w1.weight.shape[0])
        for layer in reloaded.model.layers
    } == {47}


def test_ffn_alias_audit_rejects_conflicting_serialized_width() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.Lfm2Config(
        vocab_size=12,
        hidden_size=8,
        intermediate_size=13,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["conv"],
        block_auto_adjust_ff_dim=False,
    )
    model = transformers.Lfm2ForCausalLM(config)
    model.config.block_ff_dim = 12

    with pytest.raises(
        MODULE.Lfm2CpcfBlankFoldRunnerError,
        match="conflicts",
    ):
        MODULE._restore_lfm2_ffn_config_alias(
            model,
            "conflicting_target",
        )
