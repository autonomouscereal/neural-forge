"""The launch planner is read-only and must not erase or reuse an earlier run."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("preflight_k3_tracked", Path(__file__).with_name("preflight_k3_tracked.py"))
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    for name in ("nf", "chimera", "parent", "corpus"):
        (tmp_path / name).mkdir()
    package = tmp_path / "chimera/src/chimera_omni"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    shard = tmp_path / "parent/shard"
    shard.write_bytes(b"x")
    digest = {"complete": True, "all_verified": True,
              "files": [{"file": "shard", "local_size": 1}] * 98}
    (tmp_path / "provenance").mkdir()
    (tmp_path / "provenance/digest.json").write_text(json.dumps(digest), encoding="utf-8")
    checked = []
    monkeypatch.setattr(plan, "verify_file", lambda path, expected: checked.append((path, expected)))
    args = argparse.Namespace(**{name: str(tmp_path / name) for name in ("nf", "chimera", "parent", "corpus")},
                              digest=str(tmp_path / "provenance/digest.json"), layer0=str(tmp_path / "provenance/layer0.json"),
                              output=str(tmp_path / "new-run"), image="sha256:" + "a" * 64,
                              threads=22, experts_device="cpu", layers="0-1")
    return args, checked


def test_plan_checks_inputs_and_emits_only_one_new_writable_mount(inputs):
    args, checked = inputs
    result = plan.build_plan(args)
    assert result["action"] == "plan_only_no_allocation_or_launch"
    assert not Path(args.output).exists()
    assert len(checked) == len(plan.SOURCE_HASHES) + len(plan.CORPUS_HASHES) + 4
    writable = [mount for mount in result["mounts"] if not mount["read_only"]]
    assert writable == [{"source": args.output, "target": "/run-output", "read_only": False}]
    assert all(Path(mount["source"]).is_dir() for mount in result["mounts"] if mount["read_only"])
    assert not any(mount["target"] == "/run" or mount["target"].startswith("/run/")
                   for mount in result["mounts"])
    argv = result["command_argv"]
    assert "--resume" not in argv and "--body-dir" in argv
    assert argv.count("--review-input-ids") == 2
    assert "OMP_NUM_THREADS=22" in argv
    assert "not a new full hash pass" in result["digest_scope"]


@pytest.mark.parametrize("case", ["prior_run", "source_overlap", "legacy", "mutable_image", "later_window"])
def test_plan_refuses_unsafe_or_unreviewed_start(inputs, case):
    args, _ = inputs
    if case == "prior_run":
        Path(args.output).mkdir()
        (Path(args.output) / "keep.json").write_text("existing", encoding="utf-8")
    elif case == "source_overlap":
        args.output = args.parent
    elif case == "legacy":
        args.output = "/mnt/lexar/omni-campaign/k3/new-child"
    elif case == "mutable_image":
        args.image = "k3-fold:latest"
    else:
        args.layers = "2-12"
    with pytest.raises(ValueError):
        plan.build_plan(args)
    if case == "prior_run":
        assert (Path(args.output) / "keep.json").read_text(encoding="utf-8") == "existing"


def test_current_source_size_mismatch_refuses(inputs):
    args, _ = inputs
    (Path(args.parent) / "shard").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source size changed"):
        plan.build_plan(args)


@pytest.mark.parametrize("name", ["../shard", "/tmp/shard", "sub/shard", r"..\shard", ".", ".."])
def test_digest_filename_traversal_refuses_before_read(inputs, name):
    args, _ = inputs
    path = Path(args.digest)
    content = json.loads(path.read_text(encoding="utf-8"))
    content["files"][0]["file"] = name
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="plain basenames"):
        plan.build_plan(args)


def test_explicit_cuda_experts_and_full_cpu_phase_are_in_plan(inputs):
    args, _ = inputs
    args.threads = 44
    args.experts_device = "cuda:0"
    argv = plan.build_plan(args)["command_argv"]
    assert "OMP_NUM_THREADS=44" in argv
    assert argv[argv.index("--experts-device") + 1] == "cuda:0"


def test_verify_file_checks_actual_bytes(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"known")
    plan.verify_file(path, hashlib.sha256(b"known").hexdigest())
    with pytest.raises(ValueError, match="hash mismatch"):
        plan.verify_file(path, "0" * 64)
