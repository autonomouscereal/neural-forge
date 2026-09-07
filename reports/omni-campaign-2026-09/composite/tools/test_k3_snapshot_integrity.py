"""Synthetic snapshot integrity tests; no real parent or GPU invocation."""
import json
import os
from pathlib import Path
import sys

import pytest
import torch
import safetensors.torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_k3_dense_pass import (kdp, work, fixture, corpus_files, lineage_markers)  # noqa: F401


def saved_pair(lineage_markers):
    run, _, _ = lineage_markers
    pair = kdp.Pair(torch.arange(128).reshape(2, 64).float(), torch.zeros(2, 0, 64))
    run.save_dual_snapshot(3, pair, pair.clone())
    return run, pair


@pytest.mark.parametrize("stream", ["teacher", "folded"])
def test_payload_mutation_preserving_all_metadata_is_refused_before_loading(lineage_markers, monkeypatch, stream):
    run, _ = saved_pair(lineage_markers)
    path = run.snapshot_path(3, stream)
    metadata = kdp.read_metadata(path)
    data = safetensors.torch.load_file(path)
    data["prefix_sum"][0, 0] += 1
    safetensors.torch.save_file(data, path, metadata=metadata)
    assert run.snapshot_provenance(3)[0], "metadata discovery deliberately does not hash every candidate"
    monkeypatch.setattr(safetensors.torch, "load_file", lambda *a, **k: pytest.fail("unverified snapshot deserialized"))
    with pytest.raises(SystemExit, match="payload SHA256"):
        run.load_dual_snapshot(3)


@pytest.mark.parametrize("bad", ["missing", "digest", "layer", "stream", "corpus", "rung", "run", "installed"])
def test_missing_or_wrong_manifest_binding_is_not_eligible(lineage_markers, bad):
    run, _ = saved_pair(lineage_markers)
    path = run.snapshot_path(3, "teacher")
    record = run.snapshot_manifest["files"][path]
    if bad == "missing":
        del run.snapshot_manifest["files"][path]
    elif bad == "digest":
        record["sha256"] = "not-a-sha"
    elif bad == "layer":
        record["layer"] = 2
    elif bad == "stream":
        record["stream"] = "folded"
    elif bad == "corpus":
        record["corpus_sha256"] = "0" * 64
    elif bad == "rung":
        record["rung"] = 7
    elif bad == "run":
        record["run_id"] = "foreign"
    else:
        record["installed"] = {}
    assert not run.snapshot_provenance(3)[0]
    with pytest.raises(SystemExit, match="snapshot|manifest"):
        run.load_dual_snapshot(3)


def test_selected_snapshots_are_hashed_once_and_match_before_deserialization(lineage_markers, monkeypatch):
    run, pair = saved_pair(lineage_markers)
    hashed, loaded = [], []
    original_sha, original_load = kdp.sha256_file, safetensors.torch.load_file

    def sha(path):
        if "before_L03" in path:
            hashed.append(path)
        return original_sha(path)

    def load(path, *args, **kwargs):
        assert len(hashed) == 2, "both payloads must match before either is loaded"
        loaded.append(path)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(kdp, "sha256_file", sha)
    monkeypatch.setattr(safetensors.torch, "load_file", load)
    for _ in range(3):
        assert run.snapshot_provenance(3)[0]
    assert not hashed
    teacher, folded, record = run.load_dual_snapshot(3)
    assert len(hashed) == len(set(hashed)) == len(loaded) == 2
    assert torch.equal(teacher.prefix, pair.prefix) and torch.equal(folded.prefix, pair.prefix)
    assert all(row["manifest_sha256_matched"] for row in record["files"].values())


@pytest.mark.parametrize("when", ["hash", "load"])
def test_snapshot_changed_during_io_is_refused(lineage_markers, monkeypatch, when):
    run, _ = saved_pair(lineage_markers)
    path = run.snapshot_path(3, "teacher")

    def changed():
        stat = os.stat(path)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    if when == "hash":
        original = kdp.sha256_file
        def sha(candidate):
            result = original(candidate)
            if candidate == path:
                changed()
            return result
        monkeypatch.setattr(kdp, "sha256_file", sha)
    else:
        original = safetensors.torch.load_file
        def load(candidate, *args, **kwargs):
            result = original(candidate, *args, **kwargs)
            if candidate == path:
                changed()
            return result
        monkeypatch.setattr(safetensors.torch, "load_file", load)
    with pytest.raises(SystemExit, match="changed"):
        run.load_dual_snapshot(3)
