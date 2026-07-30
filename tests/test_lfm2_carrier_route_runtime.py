from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from torch import nn

import neural_forge.lfm2_carrier_route_runtime as route_runtime
from neural_forge.dynamic_target_synthesis import (
    validate_registered_inference_state,
)
from neural_forge.hf_lfm2_semantic_step_model_runtime import (
    LFM2_SEMANTIC_STEP_MODEL_RUNTIME_SCHEMA,
)


class _Tokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {"<unk>": 0, "hello": 1}

    def decode(self, _ids: Sequence[int], **_kwargs: Any) -> str:
        return "hello"


class _Child(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(2, 2, bias=False)


def _routed_result(
    prompts: Sequence[Mapping[str, Any]],
    *,
    maximum_new_tokens: int,
) -> dict[str, Any]:
    responses = []
    for prompt in prompts:
        capped = prompt["id"] == "ordinary-capped" and maximum_new_tokens == 1_024
        responses.append(
            {
                **dict(prompt),
                "response": (
                    "continued coherent answer" if not capped else "partial answer"
                ),
                "diagnostics": {
                    "censored_by_output_cap": capped,
                    "maximum_new_tokens": maximum_new_tokens,
                },
            }
        )
    return {
        "schema": LFM2_SEMANTIC_STEP_MODEL_RUNTIME_SCHEMA,
        "status": "complete-standalone-semantic-step-whole-response-review",
        "responses": responses,
        "input_output_tables_are_untied": True,
        "routing_session_receipt": {
            "input": {"route_role": "input", "routing_cache_use_count": 2},
            "output": {"route_role": "output", "routing_cache_use_count": 2},
        },
        "operation_audit": {
            "child_body_forward_calls": max(1, len(responses)),
            "child_generate_calls": 0,
            "child_lm_head_forward_calls": 0,
            "routing_input_output_coefficient_storage_shared": False,
        },
        "runtime_sha256": f"runtime-{maximum_new_tokens}",
    }


def _install(monkeypatch: pytest.MonkeyPatch) -> tuple[_Child, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def fake_review(
        _model: Any,
        _runtime: Any,
        _semantic_source_tokenizer: Any,
        _target_tokenizer: Any,
        prompts: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append(
            {
                "ids": [str(value["id"]) for value in prompts],
                "maximum_new_tokens": kwargs["maximum_new_tokens"],
            }
        )
        return _routed_result(
            prompts,
            maximum_new_tokens=kwargs["maximum_new_tokens"],
        )

    monkeypatch.setattr(
        route_runtime,
        "run_lfm2_semantic_step_model_review",
        fake_review,
    )
    child = _Child()
    route_runtime.install_lfm2_semantic_step_generate(
        child,
        semantic_source_tokenizer=_Tokenizer(),
        target_tokenizer=_Tokenizer(),
        runtime=object(),
    )
    return child, calls


def test_public_entrypoint_is_class_level_and_reachable_state_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child, calls = _install(monkeypatch)

    assert "semantic_step_generate" not in vars(child)
    assert callable(child.semantic_step_generate)
    validate_registered_inference_state(child)

    result = route_runtime.run_lfm2_carrier_route_smoke(
        child,
        ({"id": "smoke", "category": "general", "prompt": "hello"},),
        maximum_prompt_tokens=32,
    )

    assert result["operation_audit"]["child_generate_calls"] == 0
    assert calls == [{"ids": ["smoke"], "maximum_new_tokens": 2}]


def test_adaptive_review_retries_only_capped_and_excludes_all_safety_case_forms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child, calls = _install(monkeypatch)
    prompts = (
        {
            "id": "safety-row",
            "category": "  SAFETY / adversarial  ",
            "prompt": "unsafe request",
        },
        {
            "id": "ordinary-capped",
            "category": "reasoning",
            "prompt": "long answer",
        },
    )

    dossier = route_runtime.run_lfm2_adaptive_carrier_manual_review(
        child,
        prompts,
    )

    assert calls == [
        {
            "ids": ["safety-row", "ordinary-capped"],
            "maximum_new_tokens": 1_024,
        },
        {"ids": ["ordinary-capped"], "maximum_new_tokens": 2_048},
    ]
    assert dossier["safety_observation_count"] == 1
    assert dossier["intelligence_review_response_count"] == 1
    assert dossier["initial_cap_censored_response_count"] == 1
    assert dossier["adaptive_retry_response_count"] == 1
    assert dossier["remaining_cap_censored_response_count"] == 0
    assert dossier["responses"][1]["response"] == "continued coherent answer"
    assert dossier["safety_excluded_from_intelligence_aggregate"] is True
    assert dossier["output_cap_censoring_is_not_automatic_folding_failure"] is True
    assert dossier["automatic_intelligence_verdict_performed"] is False


def test_adaptive_review_rejects_duplicate_prompt_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child, calls = _install(monkeypatch)
    prompts = (
        {"id": "duplicate", "category": "general", "prompt": "one"},
        {"id": "duplicate", "category": "general", "prompt": "two"},
    )

    with pytest.raises(
        route_runtime.Lfm2CarrierRouteRuntimeError,
        match="present and unique",
    ):
        route_runtime.run_lfm2_adaptive_carrier_manual_review(child, prompts)

    assert calls == []


def test_review_verifier_rejects_ordinary_generate_use() -> None:
    result = _routed_result(
        ({"id": "probe", "category": "general", "prompt": "hello"},),
        maximum_new_tokens=2,
    )
    result["operation_audit"]["child_generate_calls"] = 1

    with pytest.raises(
        route_runtime.Lfm2CarrierRouteRuntimeError,
        match="ordinary generate/readout",
    ):
        route_runtime.verify_lfm2_carrier_routed_review(result)


def test_resumable_review_reuses_completed_rows_after_interruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    child, calls = _install(monkeypatch)
    working_review = route_runtime.run_lfm2_semantic_step_model_review
    prompts = (
        {"id": "completed-first", "category": "general", "prompt": "one"},
        {
            "id": "ordinary-capped",
            "category": "reasoning",
            "prompt": "two",
        },
    )

    def interrupted_review(
        model: Any,
        runtime: Any,
        source_tokenizer: Any,
        target_tokenizer: Any,
        rows: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        if rows[0]["id"] == "ordinary-capped":
            raise RuntimeError("simulated process interruption")
        return working_review(
            model,
            runtime,
            source_tokenizer,
            target_tokenizer,
            rows,
            **kwargs,
        )

    monkeypatch.setattr(
        route_runtime,
        "run_lfm2_semantic_step_model_review",
        interrupted_review,
    )
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        route_runtime.run_lfm2_resumable_adaptive_carrier_manual_review(
            child,
            prompts,
            checkpoint_directory=tmp_path / "checkpoints",
            published_child_artifact_sha256="a" * 64,
        )

    assert calls == [
        {"ids": ["completed-first"], "maximum_new_tokens": 1_024}
    ]
    monkeypatch.setattr(
        route_runtime,
        "run_lfm2_semantic_step_model_review",
        working_review,
    )
    resumed = route_runtime.run_lfm2_resumable_adaptive_carrier_manual_review(
        child,
        prompts,
        checkpoint_directory=tmp_path / "checkpoints",
        published_child_artifact_sha256="a" * 64,
    )

    assert calls == [
        {"ids": ["completed-first"], "maximum_new_tokens": 1_024},
        {"ids": ["ordinary-capped"], "maximum_new_tokens": 1_024},
        {"ids": ["ordinary-capped"], "maximum_new_tokens": 2_048},
    ]
    assert resumed["checkpoint_rows_reused"] == 1
    assert resumed["checkpoint_rows_executed"] == 2
    assert resumed["checkpoint_row_count"] == 3
    assert resumed["adaptive_retry_response_count"] == 1
    assert resumed["remaining_cap_censored_response_count"] == 0
