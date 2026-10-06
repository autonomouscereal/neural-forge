"""Actual-byte fixtures for the read-only continuation proof; no CUDA imports."""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location("preflight_k3_continuation", Path(__file__).with_name("preflight_k3_continuation.py"))
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return sha(path)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in ("nf", "parent", "corpus", "accepted", "sealed", "output")}
    for path in paths.values():
        path.mkdir()
    tool = paths["nf"] / preflight.TOOL
    tool.parent.mkdir(parents=True)
    tool.write_bytes(b"native tool pinned bytes")
    lane = paths["nf"] / preflight.LANE
    lane.mkdir(parents=True)
    names = ["k3_common.py", "k3_layer.py", "k3_stream.py", "c14_union_coreset.py", "k3_receipt.py", "k3_review_stream.py"]
    for name in names:
        (lane / name).write_bytes(name.encode())
    config = {"num_hidden_layers": 93, "hidden_size": 2, "num_experts": 2,
              "first_k_dense_replace": 1, "moe_layer_freq": 1, "attn_res_block_size": 16}
    config_sha = write_json(paths["parent"] / "config.json", config)
    metadata, weight_map = {}, {}
    for i in range(96):
        name = f"model-{i:05d}.safetensors"
        path = paths["parent"] / name
        path.write_bytes(f"source{i}".encode())
        stat = path.stat()
        metadata[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        weight_map[f"model.layers.{i%93}.weight.{i}"] = name
    index_sha = write_json(paths["parent"] / "model.safetensors.index.json", {"weight_map": weight_map})
    corpus_refs = []
    for name in ("k3_input_ids.safetensors", "k3_review_input_ids.safetensors", "k3_review_native_input_ids.safetensors"):
        path = paths["corpus"] / name
        path.write_bytes(name.encode())
        corpus_refs.append({"path": "/inputs/corpus/" + name, "sha256": sha(path)})
    runtime = {"devices": [{"capability": "sm75", "index": 0, "name": "Quadro RTX 8000", "total_GB": 47.6}],
               "device": "cuda:0", "experts_device": "cuda:0", "resid_device": "cpu", "kda_backend": "torch",
               "num_threads": 22, "compute_dtype": "auto", "k_pool_factor": 1, "row_chunk": 1,
               "token_chunk": 1024, "chunk": 1024, "holdout_blocks": 24, "divergence_rows": 0,
               "review_topk": 16, "float32_matmul_precision": "highest", "backend_flags": {"allow_tf32": False}}
    software = {"tool_sha256": sha(tool), "lane_sha256": {name: sha(lane / name) for name in names},
                "torch": "2.11.0+cu128", "cuda": "12.8", "packages": {"triton": "3.6.0", "flash-linear-attention": None}}
    context = {"version": "k3_dense_pass.resume_context.v1", "corpus": "a" * 64,
               "model": {"source_directory": "/parent", "config_sha256": config_sha, "index_sha256": index_sha,
                         "source_file_metadata": metadata}, "runtime": runtime, "software": software,
               "source_identity_scope": "config/index hashes and shard size/mtime; not a full shard rehash"}
    acceptance = {"version": "k3_dense_pass.acceptance.v1", "constants": preflight.CONSTANTS, "verdict": "accepted"}
    receipt = {"resume_context": context, "corpus": {"corpus_sha256": context["corpus"]},
               "arguments": {"rung": 17408}, "layers": {}}
    plan = {"schema": "neural-forge.c17-k3-capture-plan.v1", "fixture": False, "layers": [3, 4],
            "input_ids": corpus_refs[0], "review_input_ids": corpus_refs[1:], "accepted_layers": {}}
    installed = {}
    for i in range(1, 5):
        base = f"fold/dense/L{i:02d}"
        directory = paths["accepted"] / base
        directory.mkdir(parents=True)
        ffn = directory / "ffn.safetensors"
        ffn.write_bytes(f"accepted FFN {i}".encode())
        digest = sha(ffn)
        logical = "/run-output/" + base + "/ffn.safetensors"
        lineage = {"version": preflight.PREFIX_VERSION, "upstream_ffn_sha256": dict(installed)}
        done = {"layer": i, "moe": True, "artefact": logical, "artefact_sha256": digest,
                "resume_context": context, "corpus_sha256": context["corpus"], "rung": 17408,
                "prefix_lineage": lineage, "fold_metrics": {"acceptance": acceptance}}
        done_sha = write_json(directory / "done.json", done)
        refs = {"ffn": {"path": "/accepted/" + base + "/ffn.safetensors", "sha256": digest},
                "done": {"path": "/accepted/" + base + "/done.json", "sha256": done_sha}}
        plan["accepted_layers"][str(i)] = refs
        receipt["layers"][str(i)] = {"ffn": {"path": logical, "sha256": digest, "done_marker_sha256": done_sha},
                                      "prefix_lineage": lineage, "acceptance": acceptance}
        installed[str(i)] = digest
    manifest = {"tool": "k3_dense_pass", "files": {}}
    snapshots = {}
    for stream in ("teacher", "folded"):
        name = "before_L05" + ("_folded" if stream == "folded" else "") + ".safetensors"
        path = paths["accepted"] / "fold/snapshots" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"layer": "5", "stream": stream, "tool": "k3_dense_pass", "tool_version": "1.2.4", "run_id": "original",
                "rung": "17408", "input_ids_sha256": context["corpus"], "resume_context": preflight.canonical(context),
                "prefix_lineage_version": preflight.PREFIX_VERSION, "installed_ffn_sha256": preflight.canonical(installed),
                "installed_layers": "4", "blocks": "1"}
        head = {"__metadata__": meta, "prefix_sum": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
                "block_residual": {"dtype": "F32", "shape": [2, 1, 2], "data_offsets": [16, 32]}}
        raw = json.dumps(head).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(32))
        logical = "/run-output/fold/snapshots/" + name
        ref = {"path": "/accepted/fold/snapshots/" + name, "manifest_key": logical}
        plan["end_" + stream] = ref
        snapshots["end_" + stream] = {**ref, "sha256": sha(path), "metadata": meta}
        manifest["files"][logical] = {"layer": 5, "stream": stream, "installed": dict(installed),
                                      "corpus_sha256": context["corpus"], "rung": 17408, "run_id": "original", "sha256": sha(path)}
    receipt_sha = write_json(paths["sealed"] / "k3_dense_pass_receipt.json", receipt)
    manifest_sha = write_json(paths["sealed"] / "snapshots_manifest.json", manifest)
    plan["accepted_receipt"] = {"path": "/sealed/k3_dense_pass_receipt.json", "sha256": receipt_sha}
    plan["snapshot_manifest"] = {"path": "/sealed/snapshots_manifest.json", "sha256": manifest_sha}
    plan_path = tmp_path / "capture_plan.json"
    plan_sha = write_json(plan_path, plan)
    capture = {"schema": "neural-forge.c17-k3-boundaries.v1", "fixture": False, "accepted_prefix_replay_exact": True,
               "full_corpus_replayed": True, "capture_plan": {"sha256": plan_sha},
               "accepted_receipt": plan["accepted_receipt"], "snapshot_manifest": plan["snapshot_manifest"],
               "accepted_resume_context": context, "corpus_sha256": context["corpus"],
               "corpus": {"tokens": 2, "segments": [{"path": ref["path"]} for ref in corpus_refs]},
               "installed_ffn_sha256": installed, "snapshots": snapshots}
    capture_path = tmp_path / "capture.json"
    capture_sha = write_json(capture_path, capture)
    digest_path = tmp_path / "digest.json"
    digest_sha = write_json(digest_path, {"complete": True, "all_verified": True,
        "files": [{"file": p.name, "local_size": p.stat().st_size} for p in paths["parent"].iterdir()]})
    image = "sha256:" + "b" * 64
    observed = {"schema": "neural-forge.k3-runtime-observation.v1", "image_id": image,
                "gpu_uuid": preflight.GPU_UUID, "runtime": runtime, "software": software,
                "observed_utc": datetime.now(timezone.utc).isoformat()}
    runtime_path = tmp_path / "runtime.json"
    runtime_sha = write_json(runtime_path, observed)
    shutil.copytree(paths["accepted"], paths["output"], dirs_exist_ok=True)
    (paths["output"] / "receipts").mkdir()
    shutil.copyfile(paths["sealed"] / "k3_dense_pass_receipt.json", paths["output"] / "receipts/k3_dense_pass_receipt.json")
    shutil.copyfile(paths["sealed"] / "snapshots_manifest.json", paths["output"] / "fold/snapshots/snapshots_manifest.json")
    for name, value in (("CAPTURE_SHA", capture_sha), ("PLAN_SHA", plan_sha), ("DIGEST_SHA", digest_sha)):
        monkeypatch.setattr(preflight, name, value)
    args = argparse.Namespace(**{k: str(v) for k, v in paths.items()}, capture=str(capture_path),
        capture_plan=str(plan_path), digest=str(digest_path), runtime=str(runtime_path), runtime_sha256=runtime_sha, image=image)
    return args, paths, context, observed, monkeypatch


def tree_state(root):
    return {str(p.relative_to(root)): (sha(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_valid_exact_context_is_read_only_and_only_plans_next_window(inputs):
    args, paths, _, _, _ = inputs
    before = tree_state(paths["output"].parent)
    result = preflight.build_plan(args)
    assert tree_state(paths["output"].parent) == before
    assert result["action"] == "plan_only_no_allocation_or_launch"
    assert result["next_layers"] == "5-12" and result["remaining_moe_layers"] == 88
    assert result["later_windows_conditional_on_new_receipt_and_snapshot_review"][-1] == "85-92"
    argv = result["command_argv"]
    assert "--resume" in argv and argv[argv.index("--acceptance") + 1] == "gate"
    assert argv[argv.index("--experts-device") + 1] == "cuda:0"
    assert [m["source"] for m in result["mounts"] if not m["read_only"]] == [args.output]


@pytest.mark.parametrize("name", ["config", "index", "corpus", "code", "ffn", "done", "snapshot", "clone", "manifest", "receipt"])
def test_stale_bound_bytes_refuse_without_writes(inputs, name):
    args, paths, _, _, _ = inputs
    target = {"config": paths["parent"] / "config.json", "index": paths["parent"] / "model.safetensors.index.json",
              "corpus": paths["corpus"] / "k3_input_ids.safetensors", "code": paths["nf"] / preflight.TOOL,
              "ffn": paths["accepted"] / "fold/dense/L02/ffn.safetensors", "done": paths["accepted"] / "fold/dense/L02/done.json",
              "snapshot": paths["accepted"] / "fold/snapshots/before_L05_folded.safetensors",
              "clone": paths["output"] / "fold/dense/L02/ffn.safetensors", "manifest": paths["sealed"] / "snapshots_manifest.json",
              "receipt": paths["sealed"] / "k3_dense_pass_receipt.json"}[name]
    target.write_bytes(target.read_bytes() + b"stale")
    before = tree_state(paths["output"].parent)
    with pytest.raises(ValueError, match="hash mismatch"):
        preflight.build_plan(args)
    assert tree_state(paths["output"].parent) == before


def test_source_mtime_change_refuses_even_same_bytes(inputs):
    args, paths, _, _, _ = inputs
    path = paths["parent"] / "model-00000.safetensors"
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    with pytest.raises(ValueError, match="source metadata changed"):
        preflight.build_plan(args)


@pytest.mark.parametrize("case", ["3090", "uuid", "software", "dtype", "stale", "future", "missing"])
def test_runtime_refusals(inputs, case):
    args, _, _, observed, _ = inputs
    observed = copy.deepcopy(observed)
    if case == "3090":
        observed["runtime"]["devices"][0].update(name="NVIDIA GeForce RTX 3090", capability="sm86", total_GB=24.0)
    elif case == "uuid": observed["gpu_uuid"] = "GPU-other"
    elif case == "software": observed["software"]["torch"] = "2.13.0+cpu"
    elif case == "dtype": observed["runtime"]["compute_dtype"] = "bf16"
    elif case in ("stale", "future"):
        observed["observed_utc"] = (datetime.now(timezone.utc) + timedelta(seconds=-301 if case == "stale" else 60)).isoformat()
    else: del observed["runtime"]
    args.runtime_sha256 = write_json(Path(args.runtime), observed)
    with pytest.raises(ValueError, match="runtime observation"):
        preflight.build_plan(args)


@pytest.mark.parametrize("case", ["overlap", "later_done", "later_snapshot", "partial", "hardlink", "missing_snapshot"])
def test_prepared_copy_cannot_alias_or_select_unproved_state(inputs, case):
    args, paths, _, _, _ = inputs
    if case == "overlap": args.output = args.accepted
    elif case == "later_done": write_json(paths["output"] / "fold/dense/L05/done.json", {})
    elif case == "later_snapshot": (paths["output"] / "fold/snapshots/before_L06.safetensors").write_bytes(b"unproved")
    elif case == "partial": write_json(paths["output"] / "receipts/k3_dense_pass_receipt.partial.json", {})
    elif case == "hardlink":
        dest = paths["output"] / "fold/dense/L01/ffn.safetensors"
        dest.unlink()
        os.link(paths["accepted"] / "fold/dense/L01/ffn.safetensors", dest)
    else: (paths["output"] / "fold/snapshots/before_L05.safetensors").unlink()
    before = tree_state(paths["output"].parent)
    with pytest.raises((ValueError, OSError)):
        preflight.build_plan(args)
    assert tree_state(paths["output"].parent) == before


def test_source_alias_refused_even_with_new_fixture_anchor(inputs):
    args, paths, _, _, monkeypatch = inputs
    path = Path(args.capture)
    value = json.loads(path.read_text())
    value["accepted_resume_context"]["model"]["source_directory"] = "/guessed-kimi-k3"
    receipt_path = paths["sealed"] / "k3_dense_pass_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["resume_context"] = value["accepted_resume_context"]
    receipt_sha = write_json(receipt_path, receipt)
    value["accepted_receipt"]["sha256"] = receipt_sha
    plan_path = Path(args.capture_plan)
    plan = json.loads(plan_path.read_text())
    plan["accepted_receipt"]["sha256"] = receipt_sha
    new_plan_sha = write_json(plan_path, plan)
    value["capture_plan"]["sha256"] = new_plan_sha
    monkeypatch.setattr(preflight, "PLAN_SHA", new_plan_sha)
    monkeypatch.setattr(preflight, "CAPTURE_SHA", write_json(path, value))
    with pytest.raises(ValueError, match="aliased parent"):
        preflight.build_plan(args)


@pytest.mark.parametrize("case", ["ancestor", "artifact_alias", "acceptance", "rung"])
def test_marker_semantics_refuse_even_when_fixture_file_hashes_are_rebound(inputs, case):
    args, paths, _, _, monkeypatch = inputs
    done_path = paths["accepted"] / "fold/dense/L02/done.json"
    done = json.loads(done_path.read_text())
    if case == "ancestor": done["prefix_lineage"]["upstream_ffn_sha256"] = {}
    elif case == "artifact_alias": done["artefact"] = "/guessed-fold/dense/L02/ffn.safetensors"
    elif case == "acceptance": done["fold_metrics"]["acceptance"]["constants"]["G2_memorisation_ratio_max"] = 999
    else: done["rung"] = True
    done_sha = write_json(done_path, done)
    plan_path, capture_path = Path(args.capture_plan), Path(args.capture)
    plan, capture = json.loads(plan_path.read_text()), json.loads(capture_path.read_text())
    plan["accepted_layers"]["2"]["done"]["sha256"] = done_sha
    receipt_path = paths["sealed"] / "k3_dense_pass_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["layers"]["2"]["ffn"]["done_marker_sha256"] = done_sha
    plan["accepted_receipt"]["sha256"] = write_json(receipt_path, receipt)
    capture["accepted_receipt"] = plan["accepted_receipt"]
    capture["capture_plan"]["sha256"] = write_json(plan_path, plan)
    monkeypatch.setattr(preflight, "PLAN_SHA", capture["capture_plan"]["sha256"])
    monkeypatch.setattr(preflight, "CAPTURE_SHA", write_json(capture_path, capture))
    with pytest.raises(ValueError, match={"ancestor": "ancestry", "artifact_alias": "alias",
                                        "acceptance": "acceptance", "rung": "rung"}[case]):
        preflight.build_plan(args)


def test_no_import_of_torch_or_process_execution_in_planner():
    result = subprocess.run([sys.executable, "-B", "-c",
        "import runpy,sys; runpy.run_path(sys.argv[1],run_name='not_main'); assert 'torch' not in sys.modules", str(Path(preflight.__file__))],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_duplicate_json_refuses(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"rung":17408,"rung":17408}')
    with pytest.raises(ValueError, match="duplicate"):
        preflight.read_json(path)
