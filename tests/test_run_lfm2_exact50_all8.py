from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

from experiments import run_lfm2_exact50_all8 as runner
from experiments import finalize_lfm2_exact50_all8 as finalizer
from neural_forge.hf_lfm2_intelligence_gate import (
    load_lfm2_intelligence_holdout_manifest,
)


class _PackedStore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scales", torch.ones(1, dtype=torch.float16))


class _CandidateOnlyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.neural_forge_packed_registry = torch.nn.ModuleList([_PackedStore()])
        self.generate_calls = 0

    def generate(self, *, input_ids, **_kwargs):
        self.generate_calls += 1
        continuation = torch.tensor([[7, 2]], dtype=torch.long, device=input_ids.device)
        return torch.cat((input_ids, continuation), dim=1)


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, _text, **_kwargs):
        return {
            "input_ids": torch.tensor([[1, 4]], dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "candidate " + " ".join(str(value) for value in token_ids)


def test_all8_candidate_generator_covers_independent64_without_parent() -> None:
    manifest = load_lfm2_intelligence_holdout_manifest()
    assert len(manifest.cases) == 64
    model = _CandidateOnlyModel()
    rows = runner._generate_candidate_rows(model, _Tokenizer(), manifest)
    assert len(rows) == 64
    assert model.generate_calls == 64
    assert [row["case_id"] for row in rows] == [spec.case_id for spec in manifest.cases]
    assert all(row["output_token_ids"] == [7, 2] for row in rows)


def test_all8_runner_has_no_hub_loader_optimizer_or_training_calls() -> None:
    source = inspect.getsource(runner)
    tree = ast.parse(source)
    forbidden_attributes = {
        "from_pretrained",
        "backward",
        "step",
    }
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden_attributes
    ]
    assert calls == []
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("optim" in name.lower() for name in imported_names)
    assert "load_lfm2_intelligence_holdout_parent_envelope" in source
    assert "seal_lfm2_intelligence_holdout_candidate_outputs" in source


def test_all8_run_config_is_exact50_local_and_immutable(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    tokenizer = tmp_path / "tokenizer"
    parent.mkdir()
    tokenizer.mkdir()
    envelope = tmp_path / "parent.json"
    envelope.write_text("{}", encoding="utf-8")
    digest = "0" * 64
    config = runner.Exact50All8RunConfig(
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
    assert config.group_size == 256
    assert config.analysis_chunk_groups == 4096
    assert not config.output_root.exists()
    with pytest.raises(runner.Lfm2Exact50All8RunnerError, match="multiple of eight"):
        runner.Exact50All8RunConfig(
            parent_model_directory=parent,
            tokenizer_directory=tokenizer,
            output_root=tmp_path / "bad-output",
            parent_envelope=envelope,
            expected_parent_envelope_file_sha256=digest,
            expected_parent_envelope_sha256=digest,
            expected_manifest_sha256=digest,
            expected_parent_state_fingerprint=digest,
            expected_parent_model_fingerprint=digest,
            expected_tokenizer_fingerprint=digest,
            group_size=255,
        )


def test_all8_finalizer_is_cached_output_only() -> None:
    source = inspect.getsource(finalizer)
    tree = ast.parse(source)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not {
        "load_lfm2_exact50_all8",
        "export_lfm2_exact50_all8",
    }.intersection(called_names)
    assert not {"generate", "forward", "backward"}.intersection(called_attributes)
    assert "evaluate_parent_relative_manual_retention" in called_names
    assert "postflight_cached_outputs_only" in source


def test_all8_review_transition_is_exact_all_case_and_versioned() -> None:
    run_source = inspect.getsource(runner)
    finalizer_source = inspect.getsource(finalizer)
    assert "_build_exact_all_case_parent_relative_packet" in run_source
    assert "build_parent_relative_manual_packet(" not in run_source
    assert "_authenticate_intelligence_report" in finalizer_source
    assert runner.LEGACY_TARGET_REPORT_SCHEMA.endswith("-v1")
    assert runner.TARGET_REPORT_SCHEMA.endswith("-v2")
    assert finalizer.LEGACY_FINAL_SCHEMA.endswith("-v1")
    assert finalizer.FINAL_SCHEMA.endswith("-v2")
