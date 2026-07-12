from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import experiments.launch_detached as detached
import experiments.launch_qwen_pair_compose_run as launcher
from experiments.run_qwen_pair_compose_tiny_smoke import TinyFixture
from experiments.run_qwen_pair_compose_tiny_smoke import build_tiny_command
from experiments.run_qwen_pair_compose_tiny_smoke import offline_environment


def _write_checkpoint(path: Path, *, layers: int) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "num_hidden_layers": layers}) + "\n"
    )
    (path / "model.safetensors").write_bytes(f"weights-{layers}".encode("ascii"))
    (path / "tokenizer.json").write_text("{}\n")
    return path


def _write_fake_repo(path: Path) -> Path:
    package = path / "src" / "neural_forge"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n")
    (package / "cli.py").write_text("def main():\n    return 0\n")
    (package / "hf_qwen_compose.py").write_text("ZERO_GRADIENT = True\n")
    experiments = path / "experiments"
    experiments.mkdir()
    (experiments / "compare_eval_retention.py").write_text("raise SystemExit(0)\n")
    (path / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    return path


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_compose_and_eval_argument_construction_is_closed_form_and_exact(
    tmp_path: Path,
) -> None:
    settings = launcher.ComposeSettings(
        ridge=0.004,
        max_length=96,
        max_tokens_per_layer=512,
        max_anchors=2048,
        max_calibration_texts=32,
        donor_layer_spans="0+1,2",
        fit_gate_validation_fraction=0.4,
        fit_gate_min_validation_rows=16,
        fit_gate_min_relative_improvement=0.03,
        fit_gate_max_cosine_drop=0.01,
        dtype="bfloat16",
        candidate_dtype="float32",
        device="cuda:0",
    )
    command = launcher.build_compose_command(
        python="python-test",
        source=tmp_path / "source",
        blank_candidate=tmp_path / "blank",
        output=tmp_path / "output",
        calibration_file=tmp_path / "calibration.jsonl",
        settings=settings,
    )

    assert command[:4] == [
        "python-test",
        "-m",
        "neural_forge.cli",
        "hf-qwen-pair-compose",
    ]
    assert command[4:7] == [
        str(tmp_path / "source"),
        str(tmp_path / "blank"),
        str(tmp_path / "output"),
    ]
    assert _value_after(command, "--ridge") == "0.004"
    assert _value_after(command, "--max-tokens-per-layer") == "512"
    assert _value_after(command, "--max-anchors") == "2048"
    assert _value_after(command, "--donor-layer-spans") == "0+1,2"
    assert _value_after(command, "--fit-gate-validation-fraction") == "0.4"
    assert _value_after(command, "--fit-gate-max-relative-rmse") == "0.75"
    assert _value_after(command, "--observable-coordinate-ridge") == "1e-06"
    assert _value_after(command, "--path-decomposition-max-rows") == "1024"
    assert (
        _value_after(command, "--path-decomposition-interpolation-mode")
        == "sparse-signature"
    )
    assert _value_after(command, "--rollout-refold-rounds") == "0"
    assert _value_after(command, "--rollout-refold-candidate-alphas") == "1,0.5,0.25"
    assert _value_after(command, "--rollout-gate-min-action-count-ratio") == "0.9"
    assert _value_after(command, "--candidate-dtype") == "float32"

    flags = [token.lower() for token in command if token.startswith("--")]
    assert not any(
        forbidden in flag
        for flag in flags
        for forbidden in (
            "recover",
            "train",
            "optimizer",
            "learning-rate",
            "steps",
            "epochs",
        )
    )

    eval_command = launcher.build_eval_command(
        python="python-test",
        model=tmp_path / "output",
        report=tmp_path / "eval.json",
        eval_suite=tmp_path / "suite.json",
        device="cuda:0",
        dtype="bfloat16",
    )
    assert eval_command.count("hf-eval") == 1
    assert _value_after(eval_command, "--max-behavior-probes") == "1000"
    assert _value_after(eval_command, "--max-generation-prompts") == "1000"
    assert _value_after(eval_command, "--max-choice-probes") == "250"
    with pytest.raises(ValueError, match="ridge"):
        launcher.ComposeSettings(ridge=float("nan")).validate()


def test_compose_command_wires_activation_costed_depth_spans(tmp_path: Path) -> None:
    settings = launcher.ComposeSettings(
        adaptive_depth_spans=True,
        adaptive_depth_critical_quantile=0.7,
        adaptive_depth_causal_magnitude_weight=2.0,
        adaptive_depth_incompatibility_weight=3.0,
        adaptive_depth_reconstruction_weight=4.0,
    )

    command = launcher.build_compose_command(
        python="python-test",
        source=tmp_path / "source",
        blank_candidate=tmp_path / "blank",
        output=tmp_path / "output",
        calibration_file=tmp_path / "calibration.jsonl",
        settings=settings,
    )

    assert "--adaptive-depth-spans" in command
    assert _value_after(command, "--adaptive-depth-critical-quantile") == "0.7"
    assert _value_after(command, "--adaptive-depth-causal-magnitude-weight") == "2.0"
    assert _value_after(command, "--adaptive-depth-incompatibility-weight") == "3.0"
    assert _value_after(command, "--adaptive-depth-reconstruction-weight") == "4.0"

    with pytest.raises(ValueError, match="mutually exclusive"):
        launcher.ComposeSettings(
            adaptive_depth_spans=True, donor_layer_spans="0+1,2"
        ).validate()


def test_parser_accepts_device_uuid_and_compose_settings() -> None:
    args = launcher.build_parser().parse_args(
        [
            "--source",
            "source",
            "--blank-candidate",
            "blank",
            "--run-root",
            "runs",
            "--tag",
            "pair-proof",
            "--calibration-file",
            "calibration.jsonl",
            "--eval-suite",
            "suite.json",
            "--parent-eval",
            "parent.json",
            "--device-uuid",
            "GPU-test",
            "--ridge",
            "0.02",
            "--max-anchors",
            "999",
            "--adaptive-depth-spans",
            "--adaptive-depth-critical-quantile",
            "0.75",
            "--rollout-refold-rounds",
            "2",
            "--rollout-refold-max-prompts",
            "48",
        ]
    )

    assert args.source == Path("source")
    assert args.blank_candidate == Path("blank")
    assert args.device_uuid == "GPU-test"
    assert args.ridge == pytest.approx(0.02)
    assert args.max_anchors == 999
    assert args.donor_layer_spans is None
    assert args.adaptive_depth_spans is True
    assert args.adaptive_depth_critical_quantile == pytest.approx(0.75)
    assert args.observable_coordinates is True
    assert args.path_decomposition is True
    assert args.path_decomposition_interpolation_mode == "sparse-signature"
    assert args.rollout_refold_rounds == 2
    assert args.rollout_refold_max_prompts == 48


def test_lexical_absolute_preserves_venv_executable_symlink(tmp_path: Path) -> None:
    executable_link = tmp_path / "venv-python"
    try:
        executable_link.symlink_to(Path(sys.executable).resolve())
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    lexical = launcher._lexical_absolute(executable_link)

    assert lexical == executable_link.absolute()
    assert lexical != executable_link.resolve()


def test_launcher_snapshots_every_input_and_detaches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = _write_fake_repo(tmp_path / "repo")
    source = _write_checkpoint(tmp_path / "source", layers=4)
    blank = _write_checkpoint(tmp_path / "blank", layers=2)
    run_root = tmp_path / "runs"
    run_root.mkdir()
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text('{"text":"one"}\n{"text":"two"}\n')
    suite = tmp_path / "suite.json"
    suite.write_text(
        '{"behavior_probes":[],"generation_prompts":[],"choice_probes":[]}\n'
    )
    parent = tmp_path / "parent.json"
    parent.write_text('{"behavior_suite":{},"choice_accuracy":{}}\n')
    captured: dict[str, object] = {}

    def fake_launch_detached(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        identity = {
            "schema": "neural-forge-detached-process-v1",
            "pid": 4242,
            "start_ticks": 9001,
            "argv": command,
            "argv_sha256": "identity-hash",
        }
        kwargs["pid_path"].write_text("4242\n")
        kwargs["identity_path"].write_text(json.dumps(identity) + "\n")
        return identity

    monkeypatch.setattr(launcher, "REPO_ROOT", repo)
    monkeypatch.setattr(launcher, "launch_detached", fake_launch_detached)

    result = launcher.main(
        [
            "--source",
            str(source),
            "--blank-candidate",
            str(blank),
            "--run-root",
            str(run_root),
            "--tag",
            "proof",
            "--calibration-file",
            str(calibration),
            "--eval-suite",
            str(suite),
            "--parent-eval",
            str(parent),
            "--device-uuid",
            "GPU-test-uuid",
            "--python",
            sys.executable,
            "--max-calibration-texts",
            "8",
            "--wait-pid",
            "123",
        ]
    )

    assert result == 0
    output = run_root / "proof"
    assert captured["command"] == ["bash", str(output / "run_snapshot.sh")]
    assert captured["cwd"] == output
    assert captured["environment"] == {
        "CUDA_VISIBLE_DEVICES": "GPU-test-uuid",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    assert (
        output / "snapshot" / "inputs" / "source" / "model.safetensors"
    ).read_bytes() == b"weights-4"
    assert (
        output / "snapshot" / "inputs" / "blank_candidate" / "model.safetensors"
    ).read_bytes() == b"weights-2"
    assert (
        output / "snapshot" / "inputs" / "calibration" / calibration.name
    ).read_bytes() == calibration.read_bytes()
    assert (
        output / "snapshot" / "inputs" / "eval_suite" / suite.name
    ).read_bytes() == suite.read_bytes()
    cached_snapshot = (
        output / "snapshot" / "inputs" / "cached_parent_eval" / parent.name
    )
    assert cached_snapshot.read_bytes() == parent.read_bytes()
    assert (
        output / "snapshot" / "code" / "src" / "neural_forge" / "hf_qwen_compose.py"
    ).is_file()

    snapshot_manifest = json.loads(
        (output / "snapshot" / "snapshot_manifest.json").read_text()
    )
    assert snapshot_manifest["tree_sha256"]
    roles = {item["role"] for item in snapshot_manifest["files"]}
    assert {
        "source_checkpoint",
        "blank_candidate_checkpoint",
        "calibration_file",
        "eval_suite",
        "cached_parent_eval",
        "neural_forge_code",
        "retention_code",
        "launcher_code",
    } <= roles
    assert all(len(item["sha256"]) == 64 for item in snapshot_manifest["files"])

    run_manifest = json.loads((output / "run_manifest.json").read_text())
    assert run_manifest["process"]["pid"] == 4242
    assert run_manifest["process"]["start_ticks"] == 9001
    assert (
        run_manifest["cached_parent_policy"] == "required-snapshot-no-parent-evaluation"
    )
    assert run_manifest["exact_eval"] == {
        "behavior_probes": 1000,
        "generation_prompts": 1000,
        "choice_probes": 250,
    }
    assert len((output / "run_manifest.sha256").read_text().split()[0]) == 64

    script = (output / "run_snapshot.sh").read_text()
    assert script.count("hf-qwen-pair-compose") == 1
    assert script.count("hf-eval") == 1
    assert run_manifest["snapshot_manifest_sha256"] in script
    eval_line = next(line for line in script.splitlines() if "hf-eval" in line)
    assert str(output / "model") in eval_line
    assert str(cached_snapshot) not in eval_line
    comparison_line = next(
        line for line in script.splitlines() if "compare_eval_retention.py" in line
    )
    assert str(cached_snapshot) in comparison_line
    assert "--max-behavior-probes 1000" in eval_line
    assert "--max-generation-prompts 1000" in eval_line
    assert "--max-choice-probes 250" in eval_line


def test_launcher_requires_a_completed_cached_parent_before_creating_run(
    tmp_path: Path,
) -> None:
    source = _write_checkpoint(tmp_path / "source", layers=4)
    blank = _write_checkpoint(tmp_path / "blank", layers=2)
    run_root = tmp_path / "runs"
    run_root.mkdir()
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text('{"text":"one"}\n')
    suite = tmp_path / "suite.json"
    suite.write_text("{}\n")
    invalid_parent = tmp_path / "parent.json"
    invalid_parent.write_text('{"incomplete":true}\n')

    with pytest.raises(ValueError, match="completed cached"):
        launcher.main(
            [
                "--source",
                str(source),
                "--blank-candidate",
                str(blank),
                "--run-root",
                str(run_root),
                "--tag",
                "must-not-exist",
                "--calibration-file",
                str(calibration),
                "--eval-suite",
                str(suite),
                "--parent-eval",
                str(invalid_parent),
                "--device-uuid",
                "GPU-test",
            ]
        )
    assert not (run_root / "must-not-exist").exists()


def test_shared_detacher_closes_fds_and_starts_new_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 3210
        returncode = None

        @staticmethod
        def poll():
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(detached.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(detached.time, "sleep", lambda _: None)
    monkeypatch.setattr(detached, "_linux_process_start_ticks", lambda _: 77)
    identity = detached.launch_detached(
        ["bash", "immutable.sh"],
        log_path=tmp_path / "run.log",
        pid_path=tmp_path / "pid",
        identity_path=tmp_path / "identity.json",
        cwd=tmp_path,
        environment={"CUDA_VISIBLE_DEVICES": "GPU-test"},
    )

    assert captured["command"] == ["bash", "immutable.sh"]
    assert captured["stdin"] is subprocess.DEVNULL
    assert captured["stderr"] is subprocess.STDOUT
    assert captured["close_fds"] is True
    assert captured["start_new_session"] is True
    assert identity["pid"] == 3210
    assert identity["start_ticks"] == 77
    assert json.loads((tmp_path / "identity.json").read_text())["argv"] == [
        "bash",
        "immutable.sh",
    ]


def test_tiny_smoke_command_uses_only_local_paths_and_forces_offline(
    tmp_path: Path,
) -> None:
    fixture = TinyFixture(
        root=tmp_path,
        source=tmp_path / "source",
        blank_candidate=tmp_path / "blank",
        calibration_file=tmp_path / "calibration.jsonl",
        output=tmp_path / "output",
        architecture="qwen3",
    )
    command = build_tiny_command(fixture, python="python-test", device="cpu")
    environment = offline_environment({"KEEP": "yes", "HF_HUB_OFFLINE": "0"})

    assert command[3] == "hf-qwen-pair-compose"
    assert command[4:7] == [
        str(fixture.source),
        str(fixture.blank_candidate),
        str(fixture.output),
    ]
    assert _value_after(command, "--calibration-file") == str(fixture.calibration_file)
    assert _value_after(command, "--device") == "cpu"
    assert _value_after(command, "--dtype") == "float32"
    assert not any("Qwen/" in token or "huggingface.co" in token for token in command)
    assert environment["KEEP"] == "yes"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["HF_DATASETS_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
