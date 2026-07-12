from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import reproduce_pack_qwen_p35_qkvo_selected as runner  # noqa: E402


SOURCE_REPORT = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "residual-attention-20260711"
    / "p35_L19G4_to_L21G6_qkvo_residual_rank8_h4_s32_fp16_p40_gpu3.json"
)


def _selected_replay_report() -> dict:
    report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    report["training"]["calibration_history"] = report["training"][
        "calibration_history"
    ][:3]
    report["training"]["fit_step_history"] = report["training"]["fit_step_history"][:16]
    return report


def test_source_success_report_is_exact_and_tamper_fails(tmp_path: Path) -> None:
    report = runner.load_exact_source_success_report(SOURCE_REPORT)
    assert report["report_sha256"] == runner.EXPECTED_SOURCE_REPORT_SHA256
    assert report["calibration_winner_lock"]["best_step"] == 16

    tampered = tmp_path / "source.json"
    tampered.write_bytes(SOURCE_REPORT.read_bytes() + b" ")
    with pytest.raises(runner.QwenP35QKVOSelectedReplayPackError, match="file hash"):
        runner.load_exact_source_success_report(tampered)


def test_exact_selected_replay_contract_checks_state_metrics_and_all_factors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _selected_replay_report()
    # Live FunctionalActionEvaluation.as_dict() returns this as a tuple even
    # though the persisted source JSON necessarily reloads it as a list.
    report["training"]["calibration_history"][-1]["evaluation"]["prefix_sources"] = (
        "parent-greedy",
        "student-greedy",
    )
    report["heldout_evaluation"]["locked_qkvo_action_function"]["prefix_sources"] = (
        "parent-greedy",
        "student-greedy",
    )
    monkeypatch.setattr(
        runner,
        "adapter_state_sha256",
        lambda _state: runner.EXPECTED_SELECTED_ADAPTER_STATE_SHA256,
    )
    runner.assert_exact_selected_replay(report, {"locked_state": {}})

    changed = copy.deepcopy(report)
    changed["training"]["calibration_history"][-1]["evaluation"][
        "parent_action_top1_rate"
    ] = 0.9
    with pytest.raises(
        runner.QwenP35QKVOSelectedReplayPackError,
        match="calibration metrics",
    ):
        runner.assert_exact_selected_replay(changed, {"locked_state": {}})

    inactive = copy.deepcopy(report)
    inactive["training"]["parameter_gradient_activation"]["k_right"] = False
    with pytest.raises(runner.QwenP35QKVOSelectedReplayPackError, match="all eight"):
        runner.assert_exact_selected_replay(inactive, {"locked_state": {}})


def test_reload_baseline_is_post_lock_evaluation_not_selection_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _selected_replay_report()
    report["training"]["calibration_history"][-1]["evaluation"]["prefix_sources"] = (
        "parent-greedy",
        "student-greedy",
    )
    report["heldout_evaluation"]["locked_qkvo_action_function"]["prefix_sources"] = (
        "parent-greedy",
        "student-greedy",
    )
    final = report["training"]["final_calibration"]
    final["mean_parent_action_log_probability"] = -0.5122
    final["hard_action_loss"] = 0.5122
    final["distribution_loss"] = 0.6614
    final["total_loss"] = 0.57834
    final["evaluation_sha256"] = "b" * 64
    monkeypatch.setattr(
        runner,
        "adapter_state_sha256",
        lambda _state: runner.EXPECTED_SELECTED_ADAPTER_STATE_SHA256,
    )

    runner.assert_exact_selected_replay(report, {"locked_state": {}})

    assert runner._post_lock_calibration_evaluation(report) == runner._json_value(final)
    assert runner._post_lock_calibration_evaluation(report) != (
        runner.EXPECTED_CALIBRATION_EVALUATION
    )
    assert runner._functional_discrete_contract(
        runner._post_lock_calibration_evaluation(report)
    ) == runner._functional_discrete_contract(runner.EXPECTED_CALIBRATION_EVALUATION)
    diagnostics = runner._continuous_evaluation_diagnostics(
        runner._post_lock_calibration_evaluation(report),
        runner.EXPECTED_CALIBRATION_EVALUATION,
    )
    assert diagnostics["byte_exact"] is False
    assert diagnostics["used_for_identity_or_selection"] is False


def test_reproduction_entrypoint_passes_only_exact_16_step_schedule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict = {}

    class StopHere(RuntimeError):
        pass

    def fake_replay(*_args, **kwargs):
        captured.update(kwargs)
        raise StopHere

    monkeypatch.setattr(
        runner.qkvo_runner,
        "run_qwen_p35_assignment_local_qkvo_residual",
        fake_replay,
    )
    limits = runner.FunctionalDecodeLimits(max_horizon=4)
    with pytest.raises(StopHere):
        runner.reproduce_pack_and_verify(
            torch.nn.Linear(1, 1),
            object(),
            (torch.tensor([[1]]),),
            (torch.tensor([[2]]),),
            (torch.tensor([[3]]),),
            split_provenance={"provenance_sha256": "a" * 64},
            source_success_proof={},
            source_runtime={},
            factor_path=tmp_path / "pack.safetensors",
            manifest_path=tmp_path / "pack.json",
            trace_limits=limits,
            device="cuda:0",
        )
    assert captured["steps"] == 16
    assert captured["calibration_steps"] == (0, 8, 16)
    assert captured["rank"] == 8
    assert captured["horizon"] == 4
    assert captured["micro_batch_size"] == 2
    assert captured["q_learning_rate"] == 0.001
    assert captured["k_learning_rate"] == 0.001
    assert captured["v_learning_rate"] == 0.002
    assert captured["o_learning_rate"] == 0.002
    assert captured["initialization_seed"] == 0
    assert isinstance(captured["artifact_sink"], dict)


def test_runner_has_no_model_checkpoint_or_optimizer_save_path() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "torch.save(" not in source
    assert ".save_pretrained(" not in source
    assert "optimizer.state_dict" not in source
    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--codebook-checkpoint",
            "p35",
            "--tokenizer",
            "tokenizer",
            "--factor-pack",
            "fresh/factors.safetensors",
            "--factor-manifest",
            "fresh/manifest.json",
            "--output",
            "fresh/report.json",
        ]
    )
    assert args.dtype == "float16"
    assert args.device == "cuda:0"


def test_main_never_deletes_concurrently_created_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "reserved-output"
    factor_path = output_dir / "factors.safetensors"
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "report.json"
    fixtures = SimpleNamespace(provenance={})

    monkeypatch.setattr(
        runner,
        "_prepare_replay",
        lambda _args: (
            {},
            fixtures,
            object(),
            object(),
            {},
            {},
            (),
            (),
            (),
            runner.FunctionalDecodeLimits(max_horizon=4),
        ),
    )

    def fake_reproduce(*_args, **_kwargs):
        factor_path.write_bytes(b"owned factor")
        manifest_path.write_text("owned manifest", encoding="utf-8")
        report_path.write_text("external report", encoding="utf-8")
        return {
            "schema": "fake",
            "environment": {},
            "report_sha256": "x",
            "factor_pack": {"factor_file_sha256": runner.file_sha256(factor_path)},
        }

    monkeypatch.setattr(runner, "reproduce_pack_and_verify", fake_reproduce)
    with pytest.raises(
        runner.QwenP35QKVOSelectedReplayPackError, match="already exists"
    ):
        runner.main(
            [
                "--codebook-checkpoint",
                "unused",
                "--tokenizer",
                "unused",
                "--factor-pack",
                str(factor_path),
                "--factor-manifest",
                str(manifest_path),
                "--output",
                str(report_path),
            ]
        )
    assert report_path.read_text(encoding="utf-8") == "external report"
    assert not factor_path.exists()
    assert not manifest_path.exists()


def test_implementation_provenance_and_schedule_are_durably_bound() -> None:
    provenance = runner._implementation_provenance()
    assert provenance["file_count"] == 5
    assert len(provenance["manifest_sha256"]) == 64
    assert {
        "selected_replay_runner",
        "qkvo_experiment_runner",
        "qkvo_adapter_and_trainer",
        "factor_pack_module",
        "functional_fixture_runner",
    } == {row["role"] for row in provenance["files"]}
    source = Path(runner.__file__).read_text(encoding="utf-8")
    for field in (
        '"adam_beta1"',
        '"adam_beta2"',
        '"adam_epsilon"',
        '"hard_action_weight"',
        '"distribution_weight"',
        '"temperature"',
        '"max_gradient_norm"',
        '"horizon"',
        '"dtype"',
        '"reproduced_environment"',
    ):
        assert field in source
