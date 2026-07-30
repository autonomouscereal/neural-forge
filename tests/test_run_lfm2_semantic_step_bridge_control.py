from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import run_lfm2_semantic_step_bridge_control as runner
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_phase_cache import (
    DynamicFoldPhaseCache,
    FoldPhaseEventLog,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sealed(body: dict[str, object], field: str) -> dict[str, object]:
    return {**body, field: canonical_sha256(body)}


def _span_contract() -> dict[str, object]:
    body = {
        "dependencies": {
            "bridge_artifact_sha256": _sha("bridge"),
            "bridge_profile_sha256": _sha("profile"),
            "donor_tokenizer_semantics_sha256": _sha("donor"),
            "target_tokenizer_semantics_sha256": _sha("target"),
        },
        "numeric_contract": {
            "floating_point_operations": 0,
            "comparison": "exact-raw-token-concatenation",
        },
        "abi": {
            "schema": "test-span-cache-v1",
            "algorithm": "direct-then-trie-dp",
        },
    }
    return {**body, "contract_sha256": canonical_sha256(body)}


def test_checkpoint_inventory_is_content_addressed_and_rejects_links(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")

    inventory = runner._checkpoint_artifact_inventory(checkpoint)
    assert [row["relative_path"] for row in inventory] == [
        "config.json",
        "model.safetensors",
    ]
    first_sha256 = canonical_sha256(inventory)
    (checkpoint / "model.safetensors").write_bytes(b"changed")
    assert canonical_sha256(
        runner._checkpoint_artifact_inventory(checkpoint)
    ) != first_sha256

    link = checkpoint / "linked.json"
    try:
        link.symlink_to(checkpoint / "config.json")
    except OSError:
        pytest.skip("This Windows account cannot create symlinks.")
    with pytest.raises(
        runner.Lfm2SemanticStepBridgeControlError,
        match="symlink",
    ):
        runner._checkpoint_artifact_inventory(checkpoint)


def _span_result() -> SimpleNamespace:
    evidence = _sealed(
        {
            "schema": "test-span-evidence-v1",
            "donor_row_count": 4,
            "target_row_count": 3,
            "spans": [
                {"donor_id": 0, "target_token_ids": [0]},
                {"donor_id": 1, "target_token_ids": [1]},
                {"donor_id": 2, "target_token_ids": [2]},
                {"donor_id": 3, "target_token_ids": [1, 2]},
            ],
        },
        "evidence_sha256",
    )
    receipt = _sealed(
        {
            "schema": "test-span-receipt-v1",
            "span_evidence_sha256": evidence["evidence_sha256"],
            "operation_audit": {
                "model_load_calls": 0,
                "model_forward_calls": 0,
                "training_steps": 0,
            },
        },
        "receipt_sha256",
    )
    return SimpleNamespace(span_evidence=evidence, receipt=receipt)


def _runtime(prompt_id: str, response: str = "A coherent answer.") -> dict[str, object]:
    body = {
        "schema": "test-runtime-v1",
        "status": "complete",
        "prompt_count": 1,
        "responses": [
            {
                "id": prompt_id,
                "category": "general",
                "response": response,
                "donor_response_text": response,
                "raw_donor_response_token_ids": [4, 2],
                "final_target_response_token_ids": [1, 0, 2],
                "final_target_response_text": response,
                "target_roundtrip_exact": True,
            }
        ],
        "operation_audit": {
            "child_body_forward_calls": 3,
            "parent_forward_calls": 0,
            "training_steps": 0,
            "network_access": False,
        },
    }
    return _sealed(body, "runtime_sha256")


def _prompt_key(prompt: dict[str, object]) -> object:
    return runner.build_prompt_phase_key(
        child_state_sha256=_sha("child"),
        routing_contract_sha256=_sha("routing"),
        bridge_artifact_sha256=_sha("bridge"),
        donor_tokenizer_semantics_sha256=_sha("donor"),
        target_tokenizer_semantics_sha256=_sha("target"),
        prompt=prompt,
        device="cpu",
        dtype_receipt={
            "requested_dtype": "auto",
            "resolved_dtype": "float32",
            "device": "cpu",
            "resolution_reason": "automatic-cpu-safe-selection",
            "silent_dtype_coercion": False,
            "fallback_recorded": False,
        },
        maximum_prompt_tokens=1024,
        maximum_new_tokens=256,
        maximum_embedding_rows_per_chunk=128,
        maximum_donor_rows_per_score_chunk=512,
    )


def test_phase_keys_bind_contract_prompt_numeric_and_routing() -> None:
    span = runner.span_phase_key_from_contract(_span_contract())
    assert span.phase_name == runner.SPAN_PHASE_NAME

    tampered = _span_contract()
    tampered["numeric_contract"]["floating_point_operations"] = 1
    with pytest.raises(runner.Lfm2SemanticStepBridgeControlError, match="self-hash"):
        runner.span_phase_key_from_contract(tampered)

    first = _prompt_key(
        {"id": "p1", "prompt": "Hello", "category": "general"}
    )
    changed = _prompt_key(
        {"id": "p1", "prompt": "Explain gravity", "category": "general"}
    )
    assert first.phase_name == runner.PROMPT_PHASE_NAME
    assert first.phase_key_sha256 != changed.phase_key_sha256
    assert "console" not in json.dumps(first.as_dict(), sort_keys=True)
    assert "console" not in json.dumps(span.as_dict(), sort_keys=True)


def test_span_phase_compiles_once_then_reuses_immutable_payload(tmp_path: Path) -> None:
    cache = DynamicFoldPhaseCache(tmp_path / "cache")
    events = FoldPhaseEventLog(
        tmp_path / "events.jsonl",
        run_id="span-test",
        clock=lambda: "2026-07-30T12:00:00.000000Z",
    )
    key = runner.span_phase_key_from_contract(_span_contract())
    calls = []

    def compile_call(progress):
        calls.append("compile")
        progress({"stage": "chunk", "donor_rows_processed": 4})
        return _span_result()

    first = runner.cache_or_compute_span_phase(
        cache=cache,
        events=events,
        phase_key=key,
        compile_call=compile_call,
        telemetry_call=lambda: {
            "elapsed_seconds": 0.5,
            "rss_bytes": 100,
            "device": "cpu",
            "resolved_dtype": "integer-exact",
        },
    )
    second = runner.cache_or_compute_span_phase(
        cache=cache,
        events=events,
        phase_key=key,
        compile_call=lambda _progress: pytest.fail("span phase recomputed"),
    )

    assert calls == ["compile"]
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.completion_sha256 == second.completion_sha256
    assert first.span_evidence == second.span_evidence
    assert [event.kind for event in events.read()] == [
        "miss",
        "begin",
        "progress",
        "complete",
        "hit",
    ]
    progress = events.read()[2].details
    assert progress["compiler_progress"]["donor_rows_processed"] == 4
    assert progress["telemetry"]["rss_bytes"] == 100


def test_each_prompt_is_independently_resumable_with_step_logging(
    tmp_path: Path,
) -> None:
    cache = DynamicFoldPhaseCache(tmp_path / "cache")
    prompt = {"id": "reasoning", "prompt": "Why?", "category": "reasoning"}
    key = _prompt_key(prompt)
    events = FoldPhaseEventLog(
        tmp_path / "prompt-events.jsonl",
        run_id="prompt-test",
        clock=lambda: "2026-07-30T12:00:00.000000Z",
    )
    calls = []

    def review_call(progress):
        calls.append("review")
        progress(
            {
                "stage": "semantic-step-complete",
                "step_index": 0,
                "donor_token_id": 4,
            }
        )
        return _runtime("reasoning")

    first = runner.cache_or_compute_prompt_phase(
        cache=cache,
        events=events,
        phase_key=key,
        prompt_id="reasoning",
        review_call=review_call,
        telemetry_call=lambda: {
            "elapsed_seconds": 0.25,
            "rss_bytes": 200,
            "device": "cpu",
            "resolved_dtype": "float32",
        },
    )
    second = runner.cache_or_compute_prompt_phase(
        cache=cache,
        events=events,
        phase_key=key,
        prompt_id="reasoning",
        review_call=lambda _progress: pytest.fail("prompt phase recomputed"),
    )

    assert calls == ["review"]
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.response["response"] == "A coherent answer."
    progress_events = [event for event in events.read() if event.kind == "progress"]
    assert len(progress_events) == 1
    assert (
        progress_events[0].details["runtime_progress"]["stage"]
        == "semantic-step-complete"
    )
    assert progress_events[0].details["telemetry"]["rss_bytes"] == 200


def test_console_progress_is_canonical_bounded_and_keeps_final_step() -> None:
    stream = io.StringIO()
    console = runner.ConsoleProgressReporter(
        semantic_step_every=16,
        stream=stream,
    )
    for stage in (
        "runtime-start",
        "prompt-start",
        "prompt-prefill-complete",
    ):
        console.runtime(
            {
                "runtime_progress": {"stage": stage, "prompt_id": "p1"},
                "telemetry": {"rss_bytes": 100},
            }
        )
    for step_index in range(33):
        console.runtime(
            {
                "runtime_progress": {
                    "stage": "semantic-step-complete",
                    "prompt_id": "p1",
                    "step_index": step_index,
                    "eos_selected": False,
                },
                "telemetry": {"rss_bytes": 100 + step_index},
            }
        )
    console.runtime(
        {
            "runtime_progress": {"stage": "prompt-complete", "prompt_id": "p1"},
            "telemetry": {"rss_bytes": 150},
        }
    )
    console.runtime(
        {
            "runtime_progress": {"stage": "runtime-complete"},
            "telemetry": {"rss_bytes": 150},
        }
    )
    console.span(
        {
            "compiler_progress": {"stage": "chunk", "donor_rows_processed": 4},
            "telemetry": {"rss_bytes": 75},
        }
    )

    lines = stream.getvalue().splitlines()
    documents = [json.loads(line) for line in lines]
    for line, document in zip(lines, documents, strict=True):
        assert line == json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    semantic = [
        document
        for document in documents
        if document["details"].get("runtime_progress", {}).get("stage")
        == "semantic-step-complete"
    ]
    assert [
        row["details"]["runtime_progress"]["step_index"] for row in semantic
    ] == [15, 31, 32]
    assert [row["reason"] for row in semantic] == [
        "periodic-semantic-step",
        "periodic-semantic-step",
        "final-semantic-step",
    ]
    lifecycle = [
        row["details"]["runtime_progress"]["stage"]
        for row in documents
        if row["reason"].startswith("lifecycle-")
    ]
    assert lifecycle == [
        "runtime-start",
        "prompt-start",
        "prompt-prefill-complete",
        "prompt-complete",
        "runtime-complete",
    ]
    assert documents[-1]["reason"] == "span-progress"


def test_console_throttling_does_not_drop_durable_step_events(
    tmp_path: Path,
) -> None:
    cache = DynamicFoldPhaseCache(tmp_path / "cache")
    prompt = {"id": "p1", "prompt": "Hello", "category": "general"}
    key = _prompt_key(prompt)
    events = FoldPhaseEventLog(
        tmp_path / "events.jsonl",
        run_id="console-durable-test",
        clock=lambda: "2026-07-30T12:00:00.000000Z",
    )
    stream = io.StringIO()
    console = runner.ConsoleProgressReporter(
        semantic_step_every=16,
        stream=stream,
    )

    def review_call(progress):
        for step_index in range(18):
            progress(
                {
                    "stage": "semantic-step-complete",
                    "prompt_id": "p1",
                    "step_index": step_index,
                    "eos_selected": False,
                }
            )
        progress({"stage": "prompt-complete", "prompt_id": "p1"})
        return _runtime("p1")

    runner.cache_or_compute_prompt_phase(
        cache=cache,
        events=events,
        phase_key=key,
        prompt_id="p1",
        review_call=review_call,
        console_progress_call=console.runtime,
    )

    durable_progress = [event for event in events.read() if event.kind == "progress"]
    assert len(durable_progress) == 19
    console_documents = [
        json.loads(line) for line in stream.getvalue().splitlines()
    ]
    semantic_steps = [
        row["details"]["runtime_progress"]["step_index"]
        for row in console_documents
        if row["details"]["runtime_progress"]["stage"]
        == "semantic-step-complete"
    ]
    assert semantic_steps == [15, 17]


def test_eos_step_is_always_printed_even_between_intervals() -> None:
    stream = io.StringIO()
    console = runner.ConsoleProgressReporter(
        semantic_step_every=16,
        stream=stream,
    )
    console.runtime(
        {
            "runtime_progress": {
                "stage": "semantic-step-complete",
                "prompt_id": "p1",
                "step_index": 6,
                "eos_selected": True,
            }
        }
    )
    console.runtime(
        {"runtime_progress": {"stage": "prompt-complete", "prompt_id": "p1"}}
    )

    documents = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert documents[0]["reason"] == "eos-semantic-step"
    assert documents[0]["details"]["runtime_progress"]["step_index"] == 6
    assert sum(row["reason"] == "final-semantic-step" for row in documents) == 0


def test_manual_dossier_attaches_parent_text_without_a_numeric_gate() -> None:
    prompt = {"id": "p1", "prompt": "Hello", "category": "general"}
    runtime = _runtime("p1")
    prompt_phase = runner.CachedPromptPhase(
        prompt_id="p1",
        response=runtime["responses"][0],
        runtime_receipt=runtime,
        phase_key=_prompt_key(prompt),
        completion_sha256=_sha("prompt-complete"),
        cache_hit=False,
    )
    span_key = runner.span_phase_key_from_contract(_span_contract())
    span_result = _span_result()
    span_phase = runner.CachedSpanPhase(
        span_evidence=span_result.span_evidence,
        adapter_receipt=span_result.receipt,
        phase_key=span_key,
        completion_sha256=_sha("span-complete"),
        cache_hit=False,
    )
    parent_responses = [
        {
            "id": "p1",
            "category": "general",
            "response": "Hello! How can I help?",
            "response_token_ids": [1, 2, 3],
        }
    ]
    envelope_body = {
        "responses": parent_responses,
        "response_bank_sha256": canonical_sha256(parent_responses),
    }
    envelope = _sealed(envelope_body, "envelope_sha256")

    dossier = runner.assemble_manual_review_dossier(
        prompts=[prompt],
        prompt_phases=[prompt_phase],
        parent_envelope=envelope,
        child_state_sha256=_sha("child"),
        child_unique_parameters=100,
        parent_unique_parameters=400,
        span_phase=span_phase,
        routing_contract_sha256=_sha("routing"),
        maximum_new_tokens=256,
    )

    assert dossier["realized_parameter_compaction_ratio"] == 4.0
    assert dossier["numeric_or_binary_behavior_gate_used"] is False
    assert dossier["exact_token_match_gate_used"] is False
    assert dossier["automatic_acceptance_or_rejection_performed"] is False
    comparison = dossier["comparisons"][0]
    assert comparison["parent_cached_response"]["response"].startswith("Hello")
    assert comparison["child_semantic_step_response"]["response"].startswith("A ")
    assert comparison["termination_evidence"]["completion_status"] == "unknown"
    assert (
        comparison["manual_review_dimensions"]["safety"].startswith(
            "separate observation"
        )
    )
    assert dossier["evaluation_policy"] == {
        "safety_is_separate_from_folding_quality": True,
        "generation_limit_hit_is_censored_not_failed": True,
        "loop_and_runtime_error_metadata_preserved_separately": True,
        "manual_whole_response_review_is_authoritative": True,
    }
    assert "score" not in comparison
    assert "accepted" not in dossier


def test_runtime_operation_audit_sums_events_but_not_state_gauges() -> None:
    phases = [
        SimpleNamespace(
            runtime_receipt={
                "operation_audit": {
                    "child_body_forward_calls": calls,
                    "owned_learned_parameter_count": 0,
                    "routing_fixed_continuous_state_count": 968_992,
                    "effective_child_parameter_count": 167_747_616,
                    "routing_input_output_coefficient_storage_shared": False,
                    "semantic_code_coefficients_counted_exactly_once": True,
                    "network_access": network_access,
                }
            }
        )
        for calls, network_access in ((2, False), (3, True), (5, False))
    ]

    audit = runner._sum_runtime_operation_audits(phases)

    assert audit["child_body_forward_calls"] == 10
    assert audit["owned_learned_parameter_count"] == 0
    assert audit["routing_fixed_continuous_state_count"] == 968_992
    assert audit["effective_child_parameter_count"] == 167_747_616
    assert audit["routing_input_output_coefficient_storage_shared"] is False
    assert audit["semantic_code_coefficients_counted_exactly_once"] is True
    assert audit["network_access"] is True


@pytest.mark.parametrize("failure", ["changed", "missing"])
def test_runtime_operation_audit_rejects_inconsistent_state_gauges(
    failure: str,
) -> None:
    first = {
        "operation_audit": {
            "child_body_forward_calls": 1,
            "effective_child_parameter_count": 100,
        }
    }
    second_audit = {
        "child_body_forward_calls": 1,
        "effective_child_parameter_count": 101,
    }
    if failure == "missing":
        second_audit.pop("effective_child_parameter_count")

    with pytest.raises(
        runner.Lfm2SemanticStepBridgeControlError,
        match="gauge (changed|is missing)",
    ):
        runner._sum_runtime_operation_audits(
            [
                SimpleNamespace(runtime_receipt=first),
                SimpleNamespace(runtime_receipt={"operation_audit": second_audit}),
            ]
        )


@pytest.mark.parametrize("failure", ["changed", "missing", "not-counted"])
def test_runtime_operation_audit_rejects_invalid_boolean_gauges(
    failure: str,
) -> None:
    first = {
        "operation_audit": {
            "routing_input_output_coefficient_storage_shared": False,
            "semantic_code_coefficients_counted_exactly_once": True,
        }
    }
    second_audit = {
        "routing_input_output_coefficient_storage_shared": False,
        "semantic_code_coefficients_counted_exactly_once": True,
    }
    if failure == "changed":
        second_audit["routing_input_output_coefficient_storage_shared"] = True
    elif failure == "missing":
        second_audit.pop("routing_input_output_coefficient_storage_shared")
    else:
        first["operation_audit"][
            "semantic_code_coefficients_counted_exactly_once"
        ] = False
        second_audit["semantic_code_coefficients_counted_exactly_once"] = False

    with pytest.raises(
        runner.Lfm2SemanticStepBridgeControlError,
        match="boolean gauge|counted exactly once",
    ):
        runner._sum_runtime_operation_audits(
            [
                SimpleNamespace(runtime_receipt=first),
                SimpleNamespace(runtime_receipt={"operation_audit": second_audit}),
            ]
        )


def test_parent_envelope_is_authenticated_without_a_model(tmp_path: Path) -> None:
    prompt = {"id": "p1", "prompt": "Hello", "category": "general"}
    prompt_set_sha = canonical_sha256([prompt])
    responses = [{"id": "p1", "response": "Hi", "response_token_ids": [1]}]
    body = {
        "schema": "neural-forge-lfm2-whole-response-parent-envelope-v1",
        "status": "complete-one-time-parent-response-cache",
        "prompt_set_sha256": prompt_set_sha,
        "prompt_count": 1,
        "responses": responses,
        "response_bank_sha256": canonical_sha256(responses),
        "parent_identity_sha256": _sha("parent-identity"),
        "parent_unique_parameters": 400,
        "maximum_new_tokens": 256,
    }
    envelope = _sealed(body, "envelope_sha256")
    path = tmp_path / "parent-envelope.json"
    path.write_text(json.dumps(envelope), encoding="utf-8")

    loaded = runner.load_bound_parent_envelope(
        path,
        expected_file_sha256=runner._file_sha256(path),
        expected_envelope_sha256=envelope["envelope_sha256"],
        expected_prompt_set_sha256=prompt_set_sha,
        expected_prompt_ids=["p1"],
        expected_parent_identity_sha256=_sha("parent-identity"),
        expected_parent_unique_parameters=400,
        expected_maximum_new_tokens=256,
    )

    assert loaded == envelope
    with pytest.raises(
        runner.Lfm2SemanticStepBridgeControlError,
        match="differs from the bound",
    ):
        runner.load_bound_parent_envelope(
            path,
            expected_file_sha256=runner._file_sha256(path),
            expected_envelope_sha256=envelope["envelope_sha256"],
            expected_prompt_set_sha256=prompt_set_sha,
            expected_prompt_ids=["p1"],
            expected_parent_identity_sha256=_sha("different-parent"),
            expected_parent_unique_parameters=400,
            expected_maximum_new_tokens=256,
        )


def test_dtype_policy_is_explicit_and_cpu_safe() -> None:
    automatic, automatic_receipt = runner.resolve_runtime_dtype("auto", "cpu")
    half, half_receipt = runner.resolve_runtime_dtype("fp16", "cpu")
    brain, brain_receipt = runner.resolve_runtime_dtype("bf16", "cpu")

    assert automatic is runner.torch.float32
    assert automatic_receipt["resolved_dtype"] == "float32"
    assert half is runner.torch.float32
    assert brain is runner.torch.float32
    assert half_receipt["fallback_recorded"] is True
    assert brain_receipt["fallback_recorded"] is True
    assert "fallback" in half_receipt["resolution_reason"]
    for receipt in (automatic_receipt, half_receipt, brain_receipt):
        supplied = receipt["receipt_sha256"]
        assert supplied == runner.canonical_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
    with pytest.raises(runner.Lfm2SemanticStepBridgeControlError, match="Unsupported"):
        runner.resolve_runtime_dtype("int8", "cpu")

    assert runner.resolve_checkpoint_provenance_dtype("bf16") is (
        runner.torch.bfloat16
    )
    with pytest.raises(
        runner.Lfm2SemanticStepBridgeControlError,
        match="must be one explicit",
    ):
        runner.resolve_checkpoint_provenance_dtype("auto")


def test_dtype_policy_does_not_treat_emulated_p40_bfloat16_as_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        runner.torch.cuda, "get_device_capability", lambda _device: (6, 1)
    )

    automatic, automatic_receipt = runner.resolve_runtime_dtype("auto", "cuda:0")
    explicit, explicit_receipt = runner.resolve_runtime_dtype("bf16", "cuda:0")

    assert automatic is runner.torch.float32
    assert automatic_receipt["resolved_dtype"] == "float32"
    assert automatic_receipt["cuda_compute_capability"] == [6, 1]
    assert automatic_receipt["native_bfloat16_body_execution"] is False
    assert explicit is runner.torch.float32
    assert explicit_receipt["fallback_recorded"] is True
    assert explicit_receipt["resolution_reason"] == (
        "pre-ampere-bfloat16-body-fallback-to-float32"
    )


def test_dtype_policy_selects_native_low_precision_by_cuda_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    capability = [7, 5]
    monkeypatch.setattr(
        runner.torch.cuda,
        "get_device_capability",
        lambda _device: tuple(capability),
    )

    turing, turing_receipt = runner.resolve_runtime_dtype("auto", "cuda:0")
    capability[:] = [8, 0]
    ampere, ampere_receipt = runner.resolve_runtime_dtype("auto", "cuda:0")

    assert turing is runner.torch.float16
    assert turing_receipt["low_precision_accumulation_dtype"] == "float32"
    assert ampere is runner.torch.bfloat16
    assert ampere_receipt["native_bfloat16_body_execution"] is True


def test_failures_are_append_only_content_addressed_and_keep_tracebacks(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    legacy = run_root / "failure.json"
    legacy.write_text('{"legacy":true}\n', encoding="utf-8")
    records = []
    for message in ("first failure", "second failure"):
        try:
            raise RuntimeError(message)
        except RuntimeError as exc:
            records.append(runner._record_failure(run_root, exc))

    first_path, first = records[0]
    second_path, second = records[1]
    assert first_path != second_path
    assert first_path.is_file() and second_path.is_file()
    assert len(tuple((run_root / "failures").glob("*.json"))) == 2
    assert json.loads(first_path.read_text(encoding="utf-8")) == first
    assert json.loads(second_path.read_text(encoding="utf-8")) == second
    assert "RuntimeError: first failure" in first["traceback"]
    assert "RuntimeError: second failure" in second["traceback"]
    assert legacy.read_text(encoding="utf-8") == '{"legacy":true}\n'

    latest = json.loads(
        (run_root / "latest-failure.json").read_text(encoding="utf-8")
    )
    assert latest["failure_result_sha256"] == second["result_sha256"]
    assert latest["failure_relative_path"] == (
        second_path.relative_to(run_root).as_posix()
    )
    pointer_body = dict(latest)
    pointer_sha256 = pointer_body.pop("pointer_sha256")
    assert pointer_sha256 == canonical_sha256(pointer_body)


def test_runner_has_no_parent_loader_generate_training_or_solver_call() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "generate" not in called_attributes
    assert "backward" not in called_attributes
    assert "step" not in called_attributes
    assert "compile_lfm2_semantic_step_token_fold" not in called_names
    assert "load_authenticated_portable" in called_attributes
    parsed = runner._parser().parse_args(
        [
            "--run-directory",
            "run",
            "--checkpoint-directory",
            "checkpoint",
            "--donor-tokenizer-directory",
            "donor",
            "--target-tokenizer-directory",
            "target",
            "--bridge-directory",
            "bridge",
            "--expected-bridge-artifact-sha256",
            _sha("bridge"),
            "--expected-child-unique-parameters",
            "100",
            "--expected-parent-identity-sha256",
            _sha("parent"),
            "--expected-parent-unique-parameters",
            "400",
            "--prompts-json",
            "prompts.json",
            "--expected-prompts-file-sha256",
            _sha("prompts"),
            "--parent-envelope",
            "parent.json",
            "--expected-parent-envelope-sha256",
            _sha("envelope"),
            "--expected-parent-envelope-file-sha256",
            _sha("envelope-file"),
        ]
    )
    assert parsed.maximum_new_tokens == 256
    assert parsed.console_progress_every == 16
    assert parsed.checkpoint_provenance_device == "cpu"
    assert parsed.checkpoint_provenance_dtype == "bfloat16"
    assert "--parent-model-directory" not in source
    assert "--trace-cache-directory" not in source
