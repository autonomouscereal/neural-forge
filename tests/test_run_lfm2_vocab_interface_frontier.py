from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest


pytest.importorskip("torch")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_lfm2_vocab_interface_frontier.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_lfm2_vocab_interface_frontier",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _paths(tmp_path: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in (
        "strong_source_model",
        "strong_source_cache",
        "original_parent_model",
        "parent_trajectory_cache",
        "whole_output_parent_cache",
        "tokenizer_directory",
    ):
        path = tmp_path / name
        path.mkdir(parents=True)
        result[name] = path
    envelope = tmp_path / "parent-envelope.json"
    envelope.write_text("{}\n", encoding="ascii")
    result.update(
        {
            "parent_output_envelope": envelope,
            "expected_parent_output_envelope_sha256": "a" * 64,
            "expected_parent_cache_key": "b" * 64,
            "expected_parent_state_fingerprint": "c" * 64,
            "expected_tokenizer_fingerprint": "d" * 64,
            "expected_strong_source_state_sha256": "e" * 64,
            "expected_strong_source_cache_key": "f" * 64,
            "work_root": tmp_path / "work",
            "checkpoint_root": tmp_path / "checkpoint",
            "report_path": tmp_path / "report.json",
        }
    )
    return result


def test_defaults_are_large_four_point_frontier_and_allocation_control(
    tmp_path: Path,
) -> None:
    config = MODULE.VocabularyFrontierConfig(**_paths(tmp_path))

    assert config.strong_compression_fractions == (0.15, 0.20, 0.25, 0.35)
    assert config.direct_parent_compression_fractions == (0.15, 0.20)
    assert config.expected_strong_source_unique_parameters == 206_726_912
    assert config.expected_original_parent_unique_parameters == 229_693_184
    assert config.cached_parent_outputs == config.parent_output_envelope
    assert (
        config.cached_parent_outputs_sha256
        == config.expected_parent_output_envelope_sha256
    )
    serialized = config.as_dict()
    assert serialized["factorization"] == (
        "raw_weight_svd_float64_single_decomposition"
    )
    assert serialized["training_allowed"] is False
    assert serialized["new_parent_trace_generation_allowed"] is False
    assert serialized["new_parent_response_generation_allowed"] is False
    assert (
        serialized["checkpoint_publication_requires_strict_and_loader_pass"]
        is True
    )


def test_checkpoint_or_report_preexistence_fails_closed(tmp_path: Path) -> None:
    paths = _paths(tmp_path / "checkpoint-case")
    checkpoint = paths["checkpoint_root"]
    assert isinstance(checkpoint, Path)
    checkpoint.mkdir()
    with pytest.raises(
        MODULE.Lfm2VocabularyFrontierError,
        match="checkpoint_root",
    ):
        MODULE.VocabularyFrontierConfig(**paths)

    paths = _paths(tmp_path / "report-case")
    report = paths["report_path"]
    assert isinstance(report, Path)
    report.write_text("{}\n", encoding="ascii")
    with pytest.raises(
        MODULE.Lfm2VocabularyFrontierError,
        match="report_path",
    ):
        MODULE.VocabularyFrontierConfig(**paths)


def test_runner_uses_one_raw_svd_and_current_strict_posthoc_gate() -> None:
    source = inspect.getsource(MODULE)

    assert source.count("analyze_tied_vocabulary_modules(") == 1
    assert "strict_runner.evaluate_posthoc(" in source
    assert "strict_runner.EvaluationConfig(" in source
    assert "capture_live_candidate_trajectory(" in source
    assert "evaluate_live_trajectory(" in source
    assert "install_lfm2_tied_vocabulary_pair(" in source
    assert "except BaseException" in source
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in source


def test_runner_has_no_training_hub_loader_or_checkpoint_write_path() -> None:
    source = inspect.getsource(MODULE)

    assert "AutoModel" not in source
    assert "AutoTokenizer" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert "save_pretrained" not in source
    assert "pickle" not in source
    assert "safetensors.torch.save" not in source
    assert "checkpoint_root.mkdir" not in source
    assert '"checkpoint_published": False' in source


def test_parser_requires_authenticated_local_inputs() -> None:
    parser = MODULE.build_parser()
    actions = {action.dest: action for action in parser._actions}

    required = {
        "strong_source_model",
        "strong_source_cache",
        "original_parent_model",
        "parent_trajectory_cache",
        "whole_output_parent_cache",
        "parent_output_envelope",
        "expected_parent_output_envelope_sha256",
        "expected_parent_cache_key",
        "expected_parent_state_fingerprint",
        "tokenizer_directory",
        "expected_tokenizer_fingerprint",
        "expected_strong_source_state_sha256",
        "expected_strong_source_cache_key",
        "work_root",
        "checkpoint_root",
        "report_path",
    }
    assert required.issubset(actions)
    assert all(actions[name].required for name in required)
    assert actions["strong_compression_fractions"].default == (
        0.15,
        0.20,
        0.25,
        0.35,
    )
