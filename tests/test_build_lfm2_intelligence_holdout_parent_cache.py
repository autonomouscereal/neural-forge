from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from experiments import build_lfm2_intelligence_holdout_parent_cache as runner
from neural_forge.hf_lfm2_intelligence_gate import (
    load_lfm2_intelligence_holdout_manifest,
)
from neural_forge.hf_lfm2_intelligence_holdout import (
    load_lfm2_intelligence_holdout_parent_envelope,
)


PARENT_STATE = hashlib.sha256(b"builder-parent-state").hexdigest()
PARENT_MODEL = hashlib.sha256(b"builder-parent-model").hexdigest()
TOKENIZER = hashlib.sha256(b"builder-tokenizer").hexdigest()
EOS_ID = 7


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        prompt: str,
        *,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> dict[str, torch.Tensor]:
        assert isinstance(prompt, str) and prompt
        assert return_tensors == "pt"
        assert add_special_tokens is True
        case_number = self.calls
        self.calls += 1
        return {
            "input_ids": torch.tensor([[1, 100 + case_number]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }

    def decode(
        self,
        output_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        assert skip_special_tokens is True
        return (
            f"Cached local parent response for synthetic token {output_ids[0]}."
        )


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor([1.0]))
        self.generate_calls = 0

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generate_calls += 1
        assert kwargs["do_sample"] is False
        assert kwargs["use_cache"] is True
        assert kwargs["max_new_tokens"] >= 16
        input_ids = kwargs["input_ids"]
        case_number = int(input_ids[0, -1].item())
        continuation = torch.tensor(
            [[20000 + case_number, EOS_ID]],
            device=input_ids.device,
            dtype=torch.long,
        )
        return torch.cat((input_ids, continuation), dim=1)


def test_generate_calls_parent_once_per_case_and_restores_mode() -> None:
    manifest = load_lfm2_intelligence_holdout_manifest()
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    model.train()

    rows = runner.generate_lfm2_holdout_rows_once(
        model,
        tokenizer,
        manifest,
    )

    assert len(rows) == 64
    assert model.generate_calls == 64
    assert tokenizer.calls == 64
    assert model.training is True
    assert all(row["output_token_ids"][-1] == EOS_ID for row in rows)
    assert [row["case_id"] for row in rows] == [
        spec.case_id for spec in manifest.cases
    ]
    assert [row["generation_limit"] for row in rows] == [
        spec.generation_limit for spec in manifest.cases
    ]


def test_build_parent_cache_is_local_sealed_atomic_and_never_overwrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_directory = tmp_path / "parent"
    tokenizer_directory = tmp_path / "tokenizer"
    parent_directory.mkdir()
    tokenizer_directory.mkdir()
    output_path = tmp_path / "artifacts" / "parent-envelope.json"
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    loader_calls = 0

    def load_model(*args: Any, **kwargs: Any) -> _FakeModel:
        nonlocal loader_calls
        loader_calls += 1
        assert args[0] == parent_directory.resolve()
        assert kwargs["device"] == "cpu"
        assert kwargs["dtype"] == torch.float32
        return model

    monkeypatch.setattr(
        runner.posthoc,
        "fingerprint_bound_local_tokenizer",
        lambda _path: (TOKENIZER, ()),
    )
    monkeypatch.setattr(
        runner.posthoc,
        "load_bound_special_token_ids",
        lambda _path, **_kwargs: {
            "special_token_ids": {"eos": [EOS_ID]}
        },
    )
    monkeypatch.setattr(
        runner.fold_runtime,
        "load_local_lfm2_model",
        load_model,
    )
    monkeypatch.setattr(
        runner.fold_runtime,
        "load_local_tokenizer",
        lambda _path: tokenizer,
    )
    monkeypatch.setattr(
        runner.fold_runtime,
        "model_state_sha256",
        lambda _model: PARENT_STATE,
    )
    monkeypatch.setattr(
        runner,
        "fingerprint_lfm2_parent",
        lambda _model: SimpleNamespace(identity_sha256=PARENT_MODEL),
    )
    config = runner.ParentCacheConfig(
        parent_model_directory=parent_directory,
        tokenizer_directory=tokenizer_directory,
        output_path=output_path,
        device="cpu",
        dtype="float32",
        expected_manifest_sha256=(
            load_lfm2_intelligence_holdout_manifest().manifest_sha256
        ),
        expected_parent_state_fingerprint=PARENT_STATE,
        expected_parent_model_fingerprint=PARENT_MODEL,
        expected_tokenizer_fingerprint=TOKENIZER,
    )

    result = runner.build_parent_cache(config)

    assert loader_calls == 1
    assert model.generate_calls == 64
    assert output_path.is_file()
    assert result["case_count"] == 64
    assert result["parent_generate_invocations"] == 64
    assert result["parent_outputs_generated_once"] is True
    assert result["parent_forward_passes_after_seal"] == 0
    assert result["network_access"] is False
    assert result["hub_loader_used"] is False
    assert result["output_file_sha256"] == hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()
    loaded = load_lfm2_intelligence_holdout_parent_envelope(
        output_path,
        expected_envelope_sha256=result["envelope_sha256"],
        expected_manifest_sha256=result["manifest_sha256"],
        expected_parent_state_fingerprint=PARENT_STATE,
        expected_parent_model_fingerprint=PARENT_MODEL,
        expected_tokenizer_fingerprint=TOKENIZER,
    )
    assert len(loaded.rows) == 64

    with pytest.raises(
        runner.Lfm2HoldoutParentCacheError,
        match="replace existing",
    ):
        runner.ParentCacheConfig(
            parent_model_directory=parent_directory,
            tokenizer_directory=tokenizer_directory,
            output_path=output_path,
            device="cpu",
            dtype="float32",
        )
    assert output_path.is_file()
    assert loader_calls == 1


def test_parent_cache_runner_has_no_hub_loader_training_or_network_path() -> None:
    source = inspect.getsource(runner)

    for forbidden in (
        "AutoModel",
        "AutoTokenizer",
        "from_pretrained",
        "requests.",
        "urllib.",
        ".backward(",
        "torch.optim",
    ):
        assert forbidden not in source
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in source
