from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from experiments.launch_detached import _parse_environment
from experiments.launch_detached import launch_detached
from experiments.launch_hf_functional_fold_run import build_parser as build_functional_fold_run_parser
from experiments.launch_qwen4b_path_run import build_parser as build_path_run_parser


def test_parse_environment_rejects_missing_key_or_separator() -> None:
    assert _parse_environment(["GPU=one", "MODE=fold"]) == {"GPU": "one", "MODE": "fold"}
    with pytest.raises(ValueError):
        _parse_environment(["missing"])
    with pytest.raises(ValueError):
        _parse_environment(["=value"])


def test_launch_detached_records_identity_and_does_not_reuse_pid_file(tmp_path: Path) -> None:
    output = tmp_path / "child.json"
    log = tmp_path / "run.log"
    pid_file = tmp_path / "run.pid"
    identity_file = tmp_path / "run.identity.json"
    code = (
        "import json,os,sys,time; "
        f"open({str(output)!r},'w').write(json.dumps({{'env':os.environ.get('NF_TEST'), 'stdin':sys.stdin.read()}})); "
        "time.sleep(0.2)"
    )

    identity = launch_detached(
        [sys.executable, "-c", code],
        log_path=log,
        pid_path=pid_file,
        identity_path=identity_file,
        cwd=tmp_path,
        environment={"NF_TEST": "detached"},
    )

    deadline = time.monotonic() + 5.0
    while not output.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert json.loads(output.read_text()) == {"env": "detached", "stdin": ""}
    assert int(pid_file.read_text().strip()) == identity["pid"]
    assert json.loads(identity_file.read_text())["argv"] == [sys.executable, "-c", code]
    with pytest.raises(FileExistsError):
        launch_detached(
            [sys.executable, "-c", "import time; time.sleep(1)"],
            log_path=log,
            pid_path=pid_file,
            identity_path=identity_file,
            cwd=tmp_path,
            environment={},
        )


def test_path_run_launcher_requires_explicit_target_and_run_root() -> None:
    args = build_path_run_parser().parse_args(
        [
            "--target",
            "target",
            "--run-root",
            "runs",
            "--tag",
            "proof",
            "--gpu-uuid",
            "GPU-test",
            "--hidden-basis-mode",
            "activation-oblique",
            "--hidden-basis-preprocess",
            "uncentered-rowrms",
            "--untie-output-head",
        ]
    )

    assert args.target == Path("target")
    assert args.run_root == Path("runs")
    assert args.hidden_basis_mode == "activation-oblique"
    assert args.untie_output_head is True


def test_functional_fold_run_launcher_exposes_closed_form_controls() -> None:
    args = build_functional_fold_run_parser().parse_args(
        [
            "--candidate",
            "folded",
            "--run-root",
            "runs",
            "--tag",
            "module-fit",
            "--gpu-uuid",
            "GPU-test",
            "--functional-target",
            "module-all",
            "--ridge",
            "0.004",
            "--max-tokens-per-layer",
            "2048",
        ]
    )

    assert args.candidate == Path("folded")
    assert args.run_root == Path("runs")
    assert args.functional_target == "module-all"
    assert args.ridge == pytest.approx(0.004)
    assert args.max_tokens_per_layer == 2048
