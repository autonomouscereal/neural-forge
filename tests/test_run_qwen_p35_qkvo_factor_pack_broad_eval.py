from __future__ import annotations

import copy
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_qwen_p35_qkvo_factor_pack_broad_eval as runner  # noqa: E402


class TinyTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = True):
        prefix = [1] if add_special_tokens else []
        return {"input_ids": prefix + [ord(value) for value in text]}


def _domain_report(
    behavior: list[tuple[str, bool]],
    choices: list[tuple[str, bool]],
    generation: list[tuple[str, bool]],
    *,
    ppl: float = 1.0,
    param_count: int = 10,
) -> dict:
    behavior_items = [{"id": item_id, "passed": passed} for item_id, passed in behavior]
    choice_items = [{"id": item_id, "correct": passed} for item_id, passed in choices]
    diagnostics = [{"id": item_id, "passed": passed} for item_id, passed in generation]
    generations = [
        {"id": item_id, "prompt": item_id, "output": item_id, "diagnostics": {}}
        for item_id, _passed in generation
    ]
    return {
        "model": "tiny",
        "param_count": param_count,
        "perplexity": {"mean": ppl, "items": [ppl]},
        "behavior_suite": {
            "correct": sum(value for _item, value in behavior),
            "total": len(behavior),
            "pass_rate": sum(value for _item, value in behavior)
            / max(1, len(behavior)),
            "items": behavior_items,
        },
        "choice_accuracy": {
            "correct": sum(value for _item, value in choices),
            "total": len(choices),
            "accuracy": sum(value for _item, value in choices) / max(1, len(choices)),
            "items": choice_items,
        },
        "generations": generations,
        "generation_diagnostics": diagnostics,
        "generation_health": {
            "correct": sum(value for _item, value in generation),
            "total": len(generation),
            "pass_rate": sum(value for _item, value in generation)
            / max(1, len(generation)),
            "contract": dict(runner.production_runner.GENERATION_HEALTH_CONTRACT),
        },
    }


def test_decontamination_uses_fixture_tokens_then_keeps_only_parent_passes() -> None:
    fixtures = [
        {"id": f"fixture-{index}", "prompt": f"fixture prompt {index}"}
        for index in range(22)
    ]
    suite = {
        "text_evals": ["unrelated ppl text"],
        "behavior_probes": [
            {"id": "b-overlap", "prompt": fixtures[0]["prompt"]},
            {"id": "b-clean-pass", "prompt": "clean behavior"},
            {"id": "b-clean-fail", "prompt": "failed behavior"},
        ],
        "choice_probes": [
            {
                "id": "c-clean-pass",
                "prompt": "clean choice",
                "choices": ["a", "b"],
                "answer": 0,
            },
            {
                "id": "c-overlap",
                "prompt": fixtures[1]["prompt"],
                "choices": ["a", "b"],
                "answer": 0,
            },
        ],
        "generation_prompts": [
            {"id": "g-overlap", "prompt": fixtures[2]["prompt"]},
            {"id": "g-clean-pass", "prompt": "clean generation"},
            {"id": "g-clean-fail", "prompt": "failed generation"},
        ],
    }
    parent = _domain_report(
        [("b-overlap", True), ("b-clean-pass", True), ("b-clean-fail", False)],
        [("c-clean-pass", True), ("c-overlap", True)],
        [("g-overlap", True), ("g-clean-pass", True), ("g-clean-fail", False)],
    )

    subset, audit = runner.derive_prompt_decontaminated_parent_pass_suite(
        TinyTokenizer(),
        suite,
        parent,
        fixtures,
        fixture_provenance_sha256="a" * 64,
    )

    assert [row["id"] for row in subset["behavior_probes"]] == ["b-clean-pass"]
    assert [row["id"] for row in subset["choice_probes"]] == ["c-clean-pass"]
    assert [row["id"] for row in subset["generation_prompts"]] == ["g-clean-pass"]
    assert subset["text_evals"] == []
    assert audit["domains"]["behavior"]["fixture_prompt_p35_pass_overlap_count"] == 1
    assert audit["domains"]["choice"]["fixture_prompt_p35_pass_overlap_count"] == 1
    assert audit["domains"]["generation"]["fixture_prompt_p35_pass_overlap_count"] == 1
    assert audit["candidate_outputs_used_to_build_subset"] is False
    assert audit["live_parent_pass_ids_define_acceptance_subset"] is True
    assert audit["historical_snapshot_diagnostic"]["used_for_acceptance"] is False


def test_live_p35_contract_uses_live_ids_with_bounded_generation_drift(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "EXACT_CURRENT_P35_PASS_COUNTS",
        {"behavior": 1, "choice": 1, "generation": 3},
    )
    monkeypatch.setattr(
        runner,
        "EXACT_CURRENT_P35_PASS_ID_SHA256",
        {"behavior": "a" * 64, "choice": "b" * 64, "generation": "c" * 64},
    )
    monkeypatch.setattr(runner, "MAX_CONSERVATIVE_GENERATION_PASS_COUNT_DRIFT", 2)
    report = _domain_report(
        [("b-live", True)],
        [("c-live", True)],
        [("g-live-1", True), ("g-live-2", True)],
    )

    contract = runner._live_p35_pass_contract(report)

    assert contract["live"]["generation"]["count"] == 2
    assert contract["generation_count_drift_audit"]["signed_delta"] == -1
    assert contract["generation_count_drift_audit"]["passed"] is True
    assert contract["historical_snapshot_diagnostic"]["exact_match"] is False
    assert contract["historical_snapshot_diagnostic"]["used_for_acceptance"] is False

    report["generation_diagnostics"] = []
    with pytest.raises(
        runner.QwenP35QKVOFactorPackBroadEvalError,
        match="generation pass count drift",
    ):
        runner._live_p35_pass_contract(report)


def test_certified_fresh544_decontamination_is_exact_333_250_386() -> None:
    audit = {
        "subset_counts": copy.deepcopy(runner.CERTIFIED_FRESH544_DECONTAMINATED_COUNTS),
        "subset_suite_sha256": (runner.CERTIFIED_FRESH544_DECONTAMINATED_SUITE_SHA256),
        "domains": {
            domain: {"retained_p35_pass_ids_sha256": digest}
            for domain, digest in (
                runner.CERTIFIED_FRESH544_DECONTAMINATED_PASS_ID_SHA256.items()
            )
        },
        "certified_fresh544_dynamic_baseline": {"live_exact_match": True},
    }
    runner._assert_certified_fresh544_decontamination(audit)

    audit["subset_counts"]["generation"] = 388
    with pytest.raises(
        runner.QwenP35QKVOFactorPackBroadEvalError,
        match="333/250/386",
    ):
        runner._assert_certified_fresh544_decontamination(audit)


def test_resumable_p35_loader_requires_exact_file_hash(
    tmp_path: Path, monkeypatch
) -> None:
    report_path = tmp_path / "p35.json"
    certificate_path = tmp_path / "certificate.json"
    report_path.write_text(json.dumps({"report": "live"}), encoding="utf-8")
    certificate_path.write_text(json.dumps({"certificate": True}), encoding="utf-8")
    digest = runner.file_sha256(report_path)
    certificate_digest = runner.file_sha256(certificate_path)
    monkeypatch.setattr(
        runner,
        "EXACT_FRESH544_RESUME_CERTIFICATE_FILE_SHA256",
        certificate_digest,
    )
    monkeypatch.setattr(
        runner,
        "verify_fresh_p35_report",
        lambda *_args, **_kwargs: {"verified": True},
    )
    monkeypatch.setattr(
        runner,
        "_verify_resume_certificate",
        lambda *_args, **_kwargs: {"certificate_verified": True},
    )
    reads: dict[Path, int] = {}
    original_read_bytes = Path.read_bytes

    def counted_read_bytes(path: Path) -> bytes:
        resolved = path.resolve()
        reads[resolved] = reads.get(resolved, 0) + 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)

    report, summary, observed, observed_certificate = (
        runner.load_verified_resumable_p35_report(
            report_path,
            digest,
            certificate_path=certificate_path,
            expected_certificate_file_sha256=certificate_digest,
            suite={},
            bound_suite=SimpleNamespace(),
            execution_context={},
        )
    )

    assert report == {"report": "live"}
    assert observed == digest
    assert observed_certificate == certificate_digest
    assert reads[report_path.resolve()] == 1
    assert reads[certificate_path.resolve()] == 1
    assert summary["resume_source"]["explicit_sha256_match"] is True
    assert summary["resume_source"]["parsed_from_the_exact_hashed_bytes"] is True
    assert summary["resume_source"]["baseline_evaluator_calls_executed_in_resume"] == 0
    with pytest.raises(
        runner.QwenP35QKVOFactorPackBroadEvalError,
        match="file SHA-256",
    ):
        runner.load_verified_resumable_p35_report(
            report_path,
            "0" * 64,
            certificate_path=certificate_path,
            expected_certificate_file_sha256=certificate_digest,
            suite={},
            bound_suite=SimpleNamespace(),
            execution_context={},
        )


def _audited_resume_certificate_case() -> tuple[dict, dict, dict, dict, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    certificate_path = (
        root
        / "experiments"
        / "p35_qkvo_broad_fresh544_resume_certificate_20260711.json"
    )
    report_path = (
        root
        / "runs"
        / "residual-attention-20260711"
        / "broad_evaluations"
        / "p35_qkvo_selected_factor_pack_v3_broad_eval_fp16_p40_gpu3_v2"
        / "p35_current_strict_full.json"
    )
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "replayed_behavior_items_sha256": certificate["report"][
            "behavior_items_sha256"
        ],
        "live_pass_contract": copy.deepcopy(certificate["live_pass_contract"]),
        "bound_suite_sha256": certificate["broad_suite"]["suite_content_sha256"],
    }
    checkpoint = copy.deepcopy(certificate["p35_checkpoint"])
    checkpoint["loader_source_manifest_sha256"] = checkpoint["manifest_file_sha256"]
    context = {
        "p35_checkpoint": checkpoint,
        "tokenizer": copy.deepcopy(certificate["tokenizer"]),
        "environment": copy.deepcopy(certificate["producer_environment"]),
        "current_dependency_sha256": {
            key: value
            for key, value in certificate["producer_sources"].items()
            if key != "broad_runner"
        },
        "current_consumer_runner_sha256": "d" * 64,
    }
    return certificate, report, summary, context, certificate_path, report_path


def _verify_audited_certificate_case(
    certificate: dict,
    report: dict,
    summary: dict,
    context: dict,
    certificate_path: Path,
    report_path: Path,
) -> dict:
    return runner._verify_resume_certificate(
        certificate,
        certificate_file_sha256=runner.file_sha256(certificate_path),
        certificate_byte_count=len(certificate_path.read_bytes()),
        report=report,
        report_file_sha256=runner.file_sha256(report_path),
        report_byte_count=len(report_path.read_bytes()),
        report_summary=summary,
        execution_context=context,
    )


def test_audited_fresh544_certificate_authenticates_choice_and_sources() -> None:
    case = _audited_resume_certificate_case()
    audit = _verify_audited_certificate_case(*case)
    assert audit["choice_items_authenticated_by_producer_certificate"] is True
    assert audit["producer_dependency_sources_bound"] is True
    assert audit["generation_churn"]["observed_total_churn"] == 12


def test_resume_certificate_rejects_choice_source_and_churn_tampering() -> None:
    certificate, report, summary, context, certificate_path, report_path = (
        _audited_resume_certificate_case()
    )

    changed_choice = copy.deepcopy(report)
    changed_choice["choice_accuracy"]["items"][0]["prediction"] = 1
    with pytest.raises(
        runner.QwenP35QKVOFactorPackBroadEvalError,
        match="collections",
    ):
        _verify_audited_certificate_case(
            certificate,
            changed_choice,
            summary,
            context,
            certificate_path,
            report_path,
        )

    changed_context = copy.deepcopy(context)
    changed_context["current_dependency_sha256"]["hf_tools_scorer"] = "0" * 64
    with pytest.raises(
        runner.QwenP35QKVOFactorPackBroadEvalError,
        match="source differs",
    ):
        _verify_audited_certificate_case(
            certificate,
            report,
            summary,
            changed_context,
            certificate_path,
            report_path,
        )

    changed_environment = copy.deepcopy(context)
    changed_environment["environment"]["gpu_uuid"] = "GPU-wrong"
    with pytest.raises(
        runner.QwenP35QKVOFactorPackBroadEvalError,
        match="execution environment differs",
    ):
        _verify_audited_certificate_case(
            certificate,
            report,
            summary,
            changed_environment,
            certificate_path,
            report_path,
        )

    changed_certificate = copy.deepcopy(certificate)
    changed_certificate["generation_pass_churn"]["lost_ids"].extend(
        [f"generation_pruning_{index:04d}" for index in range(20)]
    )
    with pytest.raises(
        runner.QwenP35QKVOFactorPackBroadEvalError,
        match="churn",
    ):
        _verify_audited_certificate_case(
            changed_certificate,
            report,
            summary,
            context,
            certificate_path,
            report_path,
        )


def test_behavior_outputs_are_replayed_under_exact_scorer() -> None:
    probe = {
        "id": "behavior-1",
        "prompt": "Question: 6 + 4? Answer:",
        "kind": "first_number",
        "answer": "10",
        "category": "math",
    }
    stored = runner.hf_tools._score_behavior_probe(  # noqa: SLF001
        probe,
        {"output": "Question: 6 + 4? Answer: 10"},
    )
    report = {"behavior_suite": {"items": [stored]}}
    observed = runner._assert_fresh_behavior_payloads(
        report, {"behavior_probes": [probe]}
    )
    assert observed == runner._canonical_sha256([stored])

    report["behavior_suite"]["items"][0]["passed"] = False
    with pytest.raises(
        runner.QwenP35QKVOFactorPackBroadEvalError,
        match="behavior outputs",
    ):
        runner._assert_fresh_behavior_payloads(report, {"behavior_probes": [probe]})


class FakeExecutor:
    def __init__(self, reports: list[dict]):
        self.reports = list(reports)
        self.calls: list[str] = []

    def evaluate(self, _runtime, _suite, output_path: Path, **_kwargs):
        self.calls.append(output_path.name)
        report = self.reports.pop(0)
        output_path.write_text(json.dumps(report), encoding="utf-8")
        return report


def _suite(ids: tuple[str, str, str]):
    return SimpleNamespace(
        behavior_ids=(ids[0],),
        choice_ids=(ids[1],),
        generation_ids=(ids[2],),
    )


def test_staged_gate_suppresses_full_evaluation_after_any_parent_pass_loss(
    tmp_path: Path,
) -> None:
    identifiers = ("b", "c", "g")
    parent = _domain_report([("b", True)], [("c", True)], [("g", True)])
    failed = _domain_report([("b", False)], [("c", True)], [("g", True)])
    executor = FakeExecutor([failed])

    result = runner.run_staged_candidate_evaluation(
        torch.nn.Linear(1, 1),
        executor=executor,
        full_suite=_suite(identifiers),
        decontaminated_suite=_suite(identifiers),
        p35_full_report=parent,
        p35_projected_report=parent,
        raw_qwen_report=parent,
        output_directory=tmp_path,
        child_unique_parameters=2,
    )

    assert executor.calls == ["candidate_decontaminated.json"]
    assert result["promotion_gate_passed"] is False
    assert result["full"]["executed"] is False
    assert not (tmp_path / "candidate_full_broad.json").exists()


def test_staged_gate_runs_full_only_after_zero_loss_pass(tmp_path: Path) -> None:
    identifiers = ("b", "c", "g")
    projected = _domain_report([("b", True)], [("c", True)], [("g", True)], ppl=1.0)
    p35 = _domain_report([("b", True)], [("c", True)], [("g", True)], ppl=2.0)
    raw = _domain_report([("b", True)], [("c", True)], [("g", True)], ppl=1.0)
    subset_candidate = _domain_report(
        [("b", True)], [("c", True)], [("g", True)], ppl=1.0
    )
    full_candidate = _domain_report(
        [("b", True)], [("c", True)], [("g", True)], ppl=2.2
    )
    executor = FakeExecutor([subset_candidate, full_candidate])

    result = runner.run_staged_candidate_evaluation(
        torch.nn.Linear(1, 1),
        executor=executor,
        full_suite=_suite(identifiers),
        decontaminated_suite=_suite(identifiers),
        p35_full_report=p35,
        p35_projected_report=projected,
        raw_qwen_report=raw,
        output_directory=tmp_path,
        child_unique_parameters=2,
    )

    assert executor.calls == [
        "candidate_decontaminated.json",
        "candidate_full_broad.json",
    ]
    assert result["promotion_gate_passed"] is True
    assert result["full"]["executed"] is True
    assert result["full"]["perplexity_ratios"] == pytest.approx(
        {
            "candidate_over_current_p35": 1.1,
            "candidate_over_raw_qwen": 2.2,
            "current_p35_over_raw_qwen": 2.0,
        }
    )


def test_fresh_p35_verifier_rejects_legacy_null_generation_contract() -> None:
    with pytest.raises(runner.QwenP35QKVOFactorPackBroadEvalError, match="strict"):
        runner.verify_fresh_p35_report(
            {
                "param_count": runner.EXACT_P35_PARAMETER_COUNT,
                "generation_health": {"contract": None},
            },
            {},
            bound_suite=SimpleNamespace(),
        )


def test_raw_to_current_p35_delta_contract_is_diagnostic_only() -> None:
    verdict = {
        "domains": {
            domain: dict(values)
            for domain, values in runner.EXACT_RAW_TO_CURRENT_P35_DELTAS.items()
        }
    }
    diagnostic = runner._raw_to_current_p35_diagnostic(verdict)
    assert diagnostic["historical_exact_match"] is True
    assert diagnostic["used_for_acceptance"] is False
    verdict["domains"]["generation"]["gain_count"] += 1
    diagnostic = runner._raw_to_current_p35_diagnostic(verdict)
    assert diagnostic["historical_exact_match"] is False
    assert diagnostic["used_for_acceptance"] is False


def test_output_directory_cannot_be_inside_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    with pytest.raises(runner.QwenP35QKVOFactorPackBroadEvalError, match="Output"):
        runner._assert_output_disjoint_from_inputs(
            checkpoint / "evaluation",
            files=(),
            directories=(checkpoint,),
        )


class FakeHandle:
    def __init__(self):
        self.is_applied = True
        self.adapter_registry = torch.nn.Linear(1, 1)
        self.report = SimpleNamespace(
            as_dict=lambda: {
                "model_parameter_count_installed": (
                    runner.EXACT_CANDIDATE_PARAMETER_COUNT
                ),
                "adapter_parameter_count": 92_160,
                "accounting": {
                    "net_parameters_saved_vs_incumbent": (runner.EXACT_PARAMETERS_SAVED)
                },
            }
        )

    def assert_live(self):
        assert self.is_applied

    def rollback(self):
        self.is_applied = False


def _patch_consumer_hardware(monkeypatch, handle: FakeHandle, *, fail_stage: bool):
    monkeypatch.setattr(runner.attention_probe, "_registry", lambda _runtime: object())
    monkeypatch.setattr(
        runner, "qwen3_attention_runtime_snapshot", lambda _runtime: "snap"
    )
    monkeypatch.setattr(
        runner.role_search,
        "_runtime_state_identity_sha256",
        lambda _runtime: "identity",
    )
    monkeypatch.setattr(
        runner.attention_surgery, "_capture_installed", lambda _runtime: "parent"
    )
    monkeypatch.setattr(
        runner,
        "qwen3_codebook_parameter_report",
        lambda _runtime: SimpleNamespace(
            unique_parameter_count=runner.EXACT_P35_PARAMETER_COUNT
        ),
    )
    monkeypatch.setattr(
        runner,
        "verify_fresh_p35_report",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        runner,
        "derive_prompt_decontaminated_parent_pass_suite",
        lambda *_args, **_kwargs: (
            {
                "text_evals": [],
                "behavior_probes": [{"id": "b", "prompt": "b"}],
                "choice_probes": [
                    {"id": "c", "prompt": "c", "choices": ["x"], "answer": 0}
                ],
                "generation_prompts": [{"id": "g", "prompt": "g"}],
            },
            {
                "audit_sha256": "d" * 64,
                "subset_suite_sha256": (
                    runner.CERTIFIED_FRESH544_DECONTAMINATED_SUITE_SHA256
                ),
            },
        ),
    )
    monkeypatch.setattr(
        runner.unified_runner,
        "bind_evaluation_suite",
        lambda *_args, **_kwargs: SimpleNamespace(
            suite_sha256=runner.CERTIFIED_FRESH544_DECONTAMINATED_SUITE_SHA256,
            behavior_ids=("b",),
            choice_ids=("c",),
            generation_ids=("g",),
        ),
    )
    monkeypatch.setattr(runner, "project_parent_report_to_suite", lambda *_a, **_k: {})
    monkeypatch.setattr(
        runner,
        "_assert_certified_fresh544_decontamination",
        lambda *_args: None,
    )
    monkeypatch.setattr(runner, "_zero_loss_gate", lambda *_args: {"passed": True})
    monkeypatch.setattr(
        runner,
        "_raw_to_current_p35_diagnostic",
        lambda *_args: {"used_for_acceptance": False},
    )
    registry = SimpleNamespace()
    codebook = SimpleNamespace(registry=registry)
    monkeypatch.setattr(
        runner.attention_probe,
        "_tentative_codebook",
        lambda *_args: (None, None, (), (codebook, ())),
    )
    monkeypatch.setattr(
        runner.o_runner,
        "_redirected_site",
        lambda *_args: SimpleNamespace(as_dict=lambda: {}),
    )
    monkeypatch.setattr(
        runner, "_verify_factor_pack_candidate_contract", lambda *_a, **_k: {"ok": True}
    )
    monkeypatch.setattr(
        runner.attention_surgery, "_install_attention_codebook", lambda *_args: None
    )
    monkeypatch.setattr(
        runner,
        "install_assignment_local_qkvo_residual_adapter",
        lambda *_a, **_k: handle,
    )
    monkeypatch.setattr(
        runner,
        "install_qkvo_factor_pack",
        lambda *_args: runner.replay_runner.EXPECTED_SELECTED_ADAPTER_STATE_SHA256,
    )
    if fail_stage:
        monkeypatch.setattr(
            runner,
            "run_staged_candidate_evaluation",
            lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError("injected stage failure")
            ),
        )
    else:
        monkeypatch.setattr(
            runner,
            "run_staged_candidate_evaluation",
            lambda *_a, **_k: {
                "promotion_gate_passed": True,
                "full": {"executed": True},
            },
        )
    monkeypatch.setattr(runner.o_runner, "_restore_parent_exact", lambda *_args: True)


def test_consumer_rolls_back_and_preserves_pack_on_success_and_failure(
    tmp_path: Path, monkeypatch
) -> None:
    factor = tmp_path / "factors.safetensors"
    manifest = tmp_path / "manifest.json"
    factor.write_bytes(b"factor bytes")
    pack_manifest = {"factor_file_sha256": runner.file_sha256(factor)}
    manifest.write_text(json.dumps(pack_manifest), encoding="utf-8")

    for fail_stage in (False, True):
        output = tmp_path / ("failure" if fail_stage else "success")
        output.mkdir()
        handle = FakeHandle()
        _patch_consumer_hardware(monkeypatch, handle, fail_stage=fail_stage)
        executor = FakeExecutor([{"parent": True}])
        kwargs = dict(
            source_runtime={"manifest_sha256": "m" * 64},
            fixtures=SimpleNamespace(provenance={"provenance_sha256": "f" * 64}),
            fixture_rows=[
                {"id": str(index), "prompt": str(index)} for index in range(22)
            ],
            loaded_factor_pack=SimpleNamespace(manifest=pack_manifest),
            factor_path=factor,
            manifest_path=manifest,
            full_suite_payload={},
            full_suite=SimpleNamespace(),
            raw_qwen_report={},
            raw_qwen_audit={},
            executor=executor,
            output_directory=output,
        )
        if fail_stage:
            with pytest.raises(RuntimeError, match="injected stage failure"):
                runner.consume_factor_pack_broad(
                    torch.nn.Linear(1, 1), TinyTokenizer(), **kwargs
                )
        else:
            result = runner.consume_factor_pack_broad(
                torch.nn.Linear(1, 1), TinyTokenizer(), **kwargs
            )
            assert result["transaction"]["factor_pack_immutable"] is True
            assert result["transaction"]["incumbent_p35_restored_exactly"] is True
        assert handle.is_applied is False
        assert factor.read_bytes() == b"factor bytes"
        assert json.loads(manifest.read_text(encoding="utf-8")) == pack_manifest


def test_consumer_resume_skips_parent_baseline_and_preserves_report(
    tmp_path: Path, monkeypatch
) -> None:
    factor = tmp_path / "factors.safetensors"
    manifest = tmp_path / "manifest.json"
    resume_report = tmp_path / "p35_current_strict_full.json"
    resume_certificate = tmp_path / "resume_certificate.json"
    factor.write_bytes(b"factor bytes")
    pack_manifest = {"factor_file_sha256": runner.file_sha256(factor)}
    manifest.write_text(json.dumps(pack_manifest), encoding="utf-8")
    resume_report.write_text(json.dumps({"parent": "live"}), encoding="utf-8")
    resume_certificate.write_text(json.dumps({"certificate": True}), encoding="utf-8")
    resume_sha256 = runner.file_sha256(resume_report)
    certificate_sha256 = runner.file_sha256(resume_certificate)
    output = tmp_path / "resume-output"
    output.mkdir()
    handle = FakeHandle()
    _patch_consumer_hardware(monkeypatch, handle, fail_stage=False)
    monkeypatch.setattr(
        runner,
        "load_verified_resumable_p35_report",
        lambda *_args, **_kwargs: (
            {"parent": "live"},
            {"verified": True},
            resume_sha256,
            certificate_sha256,
        ),
    )
    executor = FakeExecutor([])

    result = runner.consume_factor_pack_broad(
        torch.nn.Linear(1, 1),
        TinyTokenizer(),
        source_runtime={"manifest_sha256": "m" * 64},
        fixtures=SimpleNamespace(provenance={"provenance_sha256": "f" * 64}),
        fixture_rows=[{"id": str(index), "prompt": str(index)} for index in range(22)],
        loaded_factor_pack=SimpleNamespace(manifest=pack_manifest),
        factor_path=factor,
        manifest_path=manifest,
        full_suite_payload={},
        full_suite=SimpleNamespace(),
        raw_qwen_report={},
        raw_qwen_audit={},
        executor=executor,
        output_directory=output,
        resume_p35_report_path=resume_report,
        resume_p35_report_sha256=resume_sha256,
        resume_p35_certificate_path=resume_certificate,
        resume_p35_certificate_sha256=certificate_sha256,
        resume_execution_context={"verified": True},
    )

    assert executor.calls == []
    assert result["fresh_current_p35"]["baseline_mode"] == (
        "verified-immutable-live-report-resume"
    )
    assert (
        result["fresh_current_p35"]["baseline_forward_evaluation_executed_in_this_run"]
        is False
    )
    assert result["transaction"]["resumed_p35_report_immutable"] is True
    assert result["transaction"]["resumed_p35_certificate_immutable"] is True
    assert runner.file_sha256(resume_report) == resume_sha256
    assert handle.is_applied is False


def test_baseline_evaluator_failure_restores_p35_and_preserves_pack(
    tmp_path: Path, monkeypatch
) -> None:
    factor = tmp_path / "factors.safetensors"
    manifest = tmp_path / "manifest.json"
    factor.write_bytes(b"factor bytes")
    pack_manifest = {"factor_file_sha256": runner.file_sha256(factor)}
    manifest.write_text(json.dumps(pack_manifest), encoding="utf-8")
    output = tmp_path / "baseline-failure"
    output.mkdir()
    handle = FakeHandle()
    _patch_consumer_hardware(monkeypatch, handle, fail_stage=False)
    restored = {"calls": 0}

    def restore(*_args):
        restored["calls"] += 1
        return True

    monkeypatch.setattr(runner.o_runner, "_restore_parent_exact", restore)

    class BaselineFailureExecutor:
        def evaluate(self, *_args, **_kwargs):
            raise RuntimeError("injected baseline failure")

    with pytest.raises(RuntimeError, match="injected baseline failure"):
        runner.consume_factor_pack_broad(
            torch.nn.Linear(1, 1),
            TinyTokenizer(),
            source_runtime={"manifest_sha256": "m" * 64},
            fixtures=SimpleNamespace(provenance={"provenance_sha256": "f" * 64}),
            fixture_rows=[
                {"id": str(index), "prompt": str(index)} for index in range(22)
            ],
            loaded_factor_pack=SimpleNamespace(manifest=pack_manifest),
            factor_path=factor,
            manifest_path=manifest,
            full_suite_payload={},
            full_suite=SimpleNamespace(),
            raw_qwen_report={},
            raw_qwen_audit={},
            executor=BaselineFailureExecutor(),
            output_directory=output,
        )
    assert restored["calls"] == 1
    assert factor.read_bytes() == b"factor bytes"
    assert json.loads(manifest.read_text(encoding="utf-8")) == pack_manifest


def test_runner_has_no_checkpoint_or_factor_write_surface() -> None:
    source = inspect.getsource(runner)
    forbidden = (
        "save_pretrained(",
        "save_qwen3_codebook_checkpoint(",
        "torch.save(",
        "write_qkvo_factor_pack(",
    )
    assert all(value not in source for value in forbidden)
    args = runner.build_parser().parse_args(
        [
            "--codebook-checkpoint",
            "p35",
            "--tokenizer",
            "tokenizer",
            "--factor-pack",
            "pack.safetensors",
            "--factor-manifest",
            "pack.json",
            "--broad-suite",
            "suite.json",
            "--raw-qwen-parent-report",
            "raw.json",
            "--output-directory",
            "fresh-output",
        ]
    )
    assert args.dtype == "float16"
    assert not hasattr(args, "checkpoint_output")


def test_resume_cli_requires_path_and_hash_together() -> None:
    common = [
        "--codebook-checkpoint",
        "p35",
        "--tokenizer",
        "tokenizer",
        "--factor-pack",
        "pack.safetensors",
        "--factor-manifest",
        "pack.json",
        "--broad-suite",
        "suite.json",
        "--raw-qwen-parent-report",
        "raw.json",
        "--output-directory",
        "fresh-output",
    ]
    missing_hash = runner.build_parser().parse_args(
        [*common, "--resume-p35-report", "p35.json"]
    )
    with pytest.raises(
        runner.QwenP35QKVOFactorPackBroadEvalError,
        match="required together",
    ):
        runner._validate_resume_arguments(missing_hash)

    complete = runner.build_parser().parse_args(
        [
            *common,
            "--resume-p35-report",
            "p35.json",
            "--resume-p35-report-sha256",
            "a" * 64,
            "--resume-p35-certificate",
            "certificate.json",
            "--resume-p35-certificate-sha256",
            runner.EXACT_FRESH544_RESUME_CERTIFICATE_FILE_SHA256,
        ]
    )
    runner._validate_resume_arguments(complete)


def test_remote_wrapper_preserves_manifest_bound_factor_filename() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "remote_evaluate_p35_qkvo_factor_pack_broad_p40_gpu3_20260711.sh"
    ).read_text(encoding="utf-8")
    assert '-v "$PACK_ROOT:/factor-input:ro"' in wrapper
    assert "--factor-pack /factor-input/qkvo_factors.safetensors" in wrapper
    assert "--factor-manifest /factor-input/qkvo_factors.manifest.json" in wrapper
    assert "/factor-pack.safetensors" not in wrapper
    assert "5807fc789516ed5548a78fb4657f369a0f1c67c45a0b9e1626aa5813dd0832a8" in wrapper
    assert "6c6e979712e67f862e4574121e500e9c823778f6774ce81647752158ccb2886d" in wrapper


def test_resume_wrapper_hash_locks_preserved_live_baseline() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "remote_resume_p35_qkvo_factor_pack_broad_p40_gpu3_20260711.sh"
    ).read_text(encoding="utf-8")
    assert "Do not launch without review" in wrapper
    assert "403329fea0ec53510ec88449b8512de5331a05665e5229d78ca1ffa8ff29003c" in wrapper
    assert "365768ab766310d62afd7de141546eeb10a3576bd4bc8e79a3b212ea326e10e9" in wrapper
    assert "b3f90234a6e533c8cdd2f5afd843b6b8900508cb002c9347038146e3c8a4f16a" in wrapper
    assert (
        "sha256:883f0554354a0311c37a1e6d4cecb50ea6f2a48a5971d8c99d46a17bea43f2c8"
        in wrapper
    )
    assert '-v "$BASELINE_REPORT:/fresh-p35-baseline.json:ro"' in wrapper
    assert '-v "$BASELINE_CERTIFICATE:/fresh-p35-certificate.json:ro"' in wrapper
    assert "--resume-p35-report /fresh-p35-baseline.json" in wrapper
    assert '--resume-p35-report-sha256 "$BASELINE_REPORT_SHA256"' in wrapper
    assert "--resume-p35-certificate /fresh-p35-certificate.json" in wrapper
    assert '--resume-p35-certificate-sha256 "$BASELINE_CERTIFICATE_SHA256"' in wrapper
    assert '-v "$PACK_ROOT:/factor-input:ro"' in wrapper
    assert "qwen3_4b_qkvo_broad_resume_20260711" in wrapper
    assert '"$IMAGE_ID" \\' in wrapper
    for name in (
        "NEURAL_FORGE_RESUME_CONTAINER_IMAGE_ID",
        "NEURAL_FORGE_RESUME_GPU_UUID",
        "NEURAL_FORGE_RESUME_GPU_COMPUTE_CAPABILITY",
        "NEURAL_FORGE_RESUME_DRIVER_VERSION",
        "NEURAL_FORGE_RESUME_TRANSFORMERS_TREE_SHA256",
        "NEURAL_FORGE_RESUME_HUGGINGFACE_HUB_TREE_SHA256",
    ):
        assert f"-e {name}=" in wrapper
    assert "baseline report must not be reachable through resume RW root" in wrapper
    assert "__RUNNER_SHA256__" not in wrapper


def test_fresh544_certificate_file_hash_is_exact() -> None:
    certificate = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "p35_qkvo_broad_fresh544_resume_certificate_20260711.json"
    )
    assert runner.file_sha256(certificate) == (
        runner.EXACT_FRESH544_RESUME_CERTIFICATE_FILE_SHA256
    )
