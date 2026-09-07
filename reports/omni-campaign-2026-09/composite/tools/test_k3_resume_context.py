"""A carried gate verdict belongs to the source, runtime and evaluation masks that earned it."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_k3_dense_pass import (  # noqa: E402,F401
    corpus_files, fixture, work, lineage_markers, make_run, kdp,
    load_file, save_file, base_argv, run_pass, uniform_fixture,
)


def prepared_resume(lineage_markers):
    run, argv, entries = lineage_markers
    pair = kdp.Pair(torch.zeros(2, 64), torch.zeros(2, 0, 64))
    run.save_dual_snapshot(3, pair, pair.clone())
    run.receipt["layers"] = entries
    kdp.write_receipt(str(Path(run.receipt_dir) / kdp.PARTIAL_RECEIPT_NAME), run.receipt)
    return run, argv


@pytest.mark.parametrize("flag,value", [
    ("--compute-dtype", "bf16"), ("--kda-backend", "fla"),
    ("--divergence-rows", "1"), ("--row-chunk", "2"),
    ("--holdout-blocks", "12"), ("--k-pool-factor", "2"),
])
def test_changed_runtime_or_scoring_does_not_inherit_gate(lineage_markers, flag, value):
    original, argv = prepared_resume(lineage_markers)
    argv = list(argv)
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv += [flag, value]
    resumed = make_run(argv + ["--resume"])
    assert resumed.read_done_marker(1) is None
    assert not resumed.snapshot_provenance(3)[0]
    assert not resumed.receipt["layers"]
    assert original.read_done_marker(1) is not None


@pytest.mark.parametrize("field", ["held_mask", "real_lengths", "split"])
def test_corpus_identity_binds_evaluation_masks_and_lengths(tmp_path, corpus_files, field):
    path = tmp_path / "ids.safetensors"
    tensors = load_file(str(corpus_files["evidence"]))
    save_file(tensors, str(path))
    before = kdp.Corpus(str(path), [])
    tensors[field] = tensors[field].clone()
    if field == "held_mask":
        tensors[field][-1, 0] = ~tensors[field][-1, 0]
    elif field == "real_lengths":
        tensors[field][-1] -= 1
    else:
        tensors[field][-1] = 2
    save_file(tensors, str(path))
    after = kdp.Corpus(str(path), [])
    assert after.sha256 != before.sha256


@pytest.mark.parametrize("part", ["model", "runtime", "software", "corpus"])
def test_changed_or_missing_context_refuses_all_resume_surfaces(lineage_markers, part):
    run, argv = prepared_resume(lineage_markers)
    context = copy.deepcopy(run.receipt.get("resume_context", {}))
    context[part] = {"changed": True}
    marker_path = Path(run.done_marker_path(1))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["resume_context"] = context
    kdp.write_receipt(str(marker_path), marker)
    assert run.read_done_marker(1) is None
    assert not run.snapshot_provenance(3)[0]
    prior = copy.deepcopy(run.receipt)
    prior["resume_context"] = context
    kdp.write_receipt(str(Path(run.receipt_dir) / kdp.PARTIAL_RECEIPT_NAME), prior)
    resumed = make_run(argv + ["--resume"])
    assert not resumed.receipt["layers"]


def test_unchanged_context_retains_marker_snapshot_and_receipt(lineage_markers):
    run, argv = prepared_resume(lineage_markers)
    resumed = make_run(argv + ["--resume"])
    assert resumed.read_done_marker(2) is not None
    assert resumed.snapshot_provenance(3)[0]
    assert set(resumed.receipt["layers"]) == {"1", "2"}


def test_resume_context_records_scope_without_claiming_shard_digest_verification(lineage_markers):
    run, _, _ = lineage_markers
    context = run.receipt["resume_context"]
    assert context["model"]["config_sha256"] == run.receipt["model"]["config_sha256"]
    assert context["model"]["index_sha256"] == run.receipt["model"]["index_sha256"]
    assert context["runtime"]["compute_dtype"] == "fp32"
    assert context["software"]["lane_sha256"] == run.receipt["lane_sha256"]
    assert "not a full shard digest" in context["source_identity_scope"]
