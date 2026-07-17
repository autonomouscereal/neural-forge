from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from experiments import (
    finalize_lfm2_protected_weight_error_allocator as finalizer,
)
from experiments import run_lfm2_causal_value_allocator as causal_runner
from experiments import (
    run_lfm2_protected_weight_error_allocator as runner,
)


def test_config_needs_only_local_parent_and_independent_envelope(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    tokenizer = tmp_path / "tokenizer"
    parent.mkdir()
    tokenizer.mkdir()
    envelope = tmp_path / "independent64.json"
    envelope.write_text("{}", encoding="utf-8")
    digest = "0" * 64
    config = runner.ProtectedWeightErrorRunConfig(
        parent_model_directory=parent,
        tokenizer_directory=tokenizer,
        output_root=tmp_path / "output",
        parent_envelope=envelope,
        expected_parent_envelope_file_sha256=digest,
        expected_parent_envelope_sha256=digest,
        expected_manifest_sha256=digest,
        expected_parent_state_fingerprint=digest,
        expected_parent_model_fingerprint=digest,
        expected_tokenizer_fingerprint=digest,
    )
    assert config.parent_envelope == envelope.absolute()
    assert not hasattr(config, "cached_causal_frontier")
    assert not hasattr(config, "calibration_token_selection")


def test_optional_ratings_path_and_digest_are_paired(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    tokenizer = tmp_path / "tokenizer"
    parent.mkdir()
    tokenizer.mkdir()
    envelope = tmp_path / "independent64.json"
    envelope.write_text("{}", encoding="utf-8")
    digest = "0" * 64
    with pytest.raises(
        runner.Lfm2ProtectedWeightErrorRunnerError,
        match="must be supplied together",
    ):
        runner.ProtectedWeightErrorRunConfig(
            parent_model_directory=parent,
            tokenizer_directory=tokenizer,
            output_root=tmp_path / "output",
            parent_envelope=envelope,
            expected_parent_envelope_file_sha256=digest,
            expected_parent_envelope_sha256=digest,
            expected_manifest_sha256=digest,
            expected_parent_state_fingerprint=digest,
            expected_parent_model_fingerprint=digest,
            expected_tokenizer_fingerprint=digest,
            parent_relative_manual_ratings=envelope,
        )


def test_cli_has_no_frontier_or_causal_selection_argument() -> None:
    destinations = {action.dest for action in runner.build_parser()._actions}
    assert "cached_causal_frontier" not in destinations
    assert "expected_cached_causal_frontier_file_sha256" not in destinations
    assert "calibration_token_selection" not in destinations
    assert {
        "parent_envelope",
        "expected_parent_envelope_file_sha256",
        "expected_parent_envelope_sha256",
        "expected_manifest_sha256",
        "expected_parent_state_fingerprint",
        "expected_parent_model_fingerprint",
        "expected_tokenizer_fingerprint",
    } <= destinations
    assert (
        runner.shared_runner._parent_relative_decision
        is causal_runner._parent_relative_decision
    )


@pytest.mark.parametrize("module", [runner, finalizer])
def test_ablation_runners_have_no_hub_training_optimizer_or_backward_call(
    module,
) -> None:
    tree = ast.parse(inspect.getsource(module))
    forbidden = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"from_pretrained", "backward", "step", "train"}
    }
    assert forbidden == set()
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("optim" in value.lower() for value in imports)
