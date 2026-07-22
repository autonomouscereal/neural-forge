from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import cache_lfm2_whole_response_parent as cache


class _Parameter:
    grad = None


class _Model:
    def parameters(self):
        return iter((_Parameter(),))


def test_parent_responses_are_generated_once_and_sealed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    tokenizer = tmp_path / "tokenizer"
    parent.mkdir()
    tokenizer.mkdir()
    prompts = tmp_path / "prompts.json"
    prompts.write_text(
        json.dumps(
            {
                "prompts": [
                    {"id": "a", "prompt": "Explain five plus seven."},
                    {
                        "id": "b",
                        "messages": [{"role": "user", "content": "Say hello."}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    model = _Model()
    monkeypatch.setattr(
        cache.residual,
        "load_local_lfm2_model",
        lambda *_args, **_kwargs: model,
    )
    monkeypatch.setattr(cache.residual, "load_local_tokenizer", lambda _path: object())
    monkeypatch.setattr(cache.residual, "_torch_dtype", lambda _value: object())
    monkeypatch.setattr(
        cache.residual,
        "model_state_sha256",
        lambda _model: "a" * 64,
    )
    monkeypatch.setattr(
        cache.residual,
        "unique_parameter_count",
        lambda _model: 229_000_000,
    )
    monkeypatch.setattr(cache, "fingerprint_local_tokenizer", lambda _path: "b" * 64)
    monkeypatch.setattr(
        cache,
        "fingerprint_lfm2_parent",
        lambda _model: SimpleNamespace(identity_sha256="c" * 64),
    )
    calls = []

    def generate(
        _model: object,
        _tokenizer: object,
        rows: object,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        calls.append((len(rows), kwargs["maximum_new_tokens"]))
        return [
            {
                "id": row["id"],
                "category": row["category"],
                "response": f"complete parent response {row['id']}",
                "response_token_ids": [1, 2, 3],
                "diagnostics": {"interpretation": "diagnostic-only"},
            }
            for row in rows
        ]

    monkeypatch.setattr(cache, "_generate_rows", generate)
    output = tmp_path / "parent-envelope.json"
    document = cache.cache_parent_responses(
        parent_model_directory=parent,
        tokenizer_directory=tokenizer,
        prompts_json=prompts,
        output_json=output,
        device="cpu",
        dtype="bfloat16",
        maximum_prompt_tokens=1024,
        maximum_new_tokens=256,
        expected_parent_state_sha256="a" * 64,
        expected_parent_identity_sha256="c" * 64,
        expected_tokenizer_sha256="b" * 64,
    )

    assert calls == [(2, 256)]
    assert document["prompt_count"] == 2
    assert document["maximum_new_tokens"] == 256
    assert document["execution_audit"]["parent_model_loads"] == 1
    assert document["execution_audit"]["parent_generate_invocations"] == 2
    assert document["execution_audit"]["training_steps"] == 0
    assert document["execution_audit"]["hub_loader_used"] is False
    assert document["execution_audit"]["parent_state_unchanged"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == document


def test_parent_cache_refuses_to_replace_existing_output(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    prompts = tmp_path / "prompts.json"
    prompts.write_text('[{"id":"x","prompt":"hello"}]', encoding="utf-8")
    output = tmp_path / "existing.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(Exception, match="replace existing"):
        cache.cache_parent_responses(
            parent_model_directory=parent,
            tokenizer_directory=parent,
            prompts_json=prompts,
            output_json=output,
            device="cpu",
            dtype="bfloat16",
            maximum_prompt_tokens=1024,
            maximum_new_tokens=256,
        )
