from __future__ import annotations

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
        enforce_exact_contract=False,
    )

    assert [row["id"] for row in subset["behavior_probes"]] == ["b-clean-pass"]
    assert [row["id"] for row in subset["choice_probes"]] == ["c-clean-pass"]
    assert [row["id"] for row in subset["generation_prompts"]] == ["g-clean-pass"]
    assert subset["text_evals"] == []
    assert audit["domains"]["behavior"]["fixture_prompt_p35_pass_overlap_count"] == 1
    assert audit["domains"]["choice"]["fixture_prompt_p35_pass_overlap_count"] == 1
    assert audit["domains"]["generation"]["fixture_prompt_p35_pass_overlap_count"] == 1
    assert audit["candidate_outputs_used_to_build_subset"] is False


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
        )


def test_raw_to_current_p35_delta_contract_is_exact() -> None:
    verdict = {
        "domains": {
            domain: dict(values)
            for domain, values in runner.EXACT_RAW_TO_CURRENT_P35_DELTAS.items()
        }
    }
    runner._assert_raw_to_current_p35_comparison(verdict)
    verdict["domains"]["generation"]["gain_count"] += 1
    with pytest.raises(runner.QwenP35QKVOFactorPackBroadEvalError, match="generation"):
        runner._assert_raw_to_current_p35_comparison(verdict)


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
    monkeypatch.setattr(runner, "verify_fresh_p35_report", lambda *_args: {"ok": True})
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
            {"audit_sha256": "d" * 64},
        ),
    )
    monkeypatch.setattr(
        runner.unified_runner,
        "bind_evaluation_suite",
        lambda *_args, **_kwargs: SimpleNamespace(
            suite_sha256=runner.EXACT_DECONTAMINATED_SUITE_SHA256,
            behavior_ids=("b",),
            choice_ids=("c",),
            generation_ids=("g",),
        ),
    )
    monkeypatch.setattr(runner, "project_parent_report_to_suite", lambda *_a, **_k: {})
    monkeypatch.setattr(runner, "_zero_loss_gate", lambda *_args: {"passed": True})
    monkeypatch.setattr(
        runner, "_assert_raw_to_current_p35_comparison", lambda *_args: None
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
