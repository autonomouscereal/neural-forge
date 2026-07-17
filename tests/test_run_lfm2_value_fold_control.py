from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

from experiments import run_lfm2_value_fold_control as runner
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
        continuation = torch.tensor(
            [[7, 2]],
            dtype=torch.long,
            device=input_ids.device,
        )
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


def test_candidate_generator_covers_independent_64_without_a_parent() -> None:
    manifest = load_lfm2_intelligence_holdout_manifest()
    assert len(manifest.cases) == 64
    model = _CandidateOnlyModel()
    rows = runner._generate_candidate_rows(model, _Tokenizer(), manifest)
    assert len(rows) == 64
    assert model.generate_calls == 64
    assert [row["case_id"] for row in rows] == [spec.case_id for spec in manifest.cases]
    assert all(row["output_token_ids"] == [7, 2] for row in rows)
    assert all(row["topical_relevance"] is None for row in rows)


def test_runner_has_no_hub_model_loader_optimizer_or_backward_call() -> None:
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


def test_run_config_requires_both_large_targets_and_immutable_inputs(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    tokenizer = tmp_path / "tokenizer"
    parent.mkdir()
    tokenizer.mkdir()
    envelope = tmp_path / "parent.json"
    envelope.write_text("{}", encoding="utf-8")
    digest = "0" * 64
    with pytest.raises(
        runner.Lfm2ValueFoldControlRunnerError,
        match="exactly the 50% and 75%",
    ):
        runner.ValueFoldControlRunConfig(
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
            target_reductions=(0.50,),
        )
