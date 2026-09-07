"""A carried gate verdict belongs to the source, runtime and evaluation masks that earned it."""
import copy
import json
import os
import shutil
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


@pytest.mark.parametrize("stream", ["teacher", "folded"])
@pytest.mark.parametrize("stamp", [None, "{}", "not json"])
def test_snapshot_context_is_checked_independently_of_valid_markers(lineage_markers, stream, stamp):
    run, _ = prepared_resume(lineage_markers)
    path = run.snapshot_path(3, stream)
    metadata = kdp.read_metadata(path)
    if stamp is None:
        metadata.pop("resume_context")
    else:
        metadata["resume_context"] = stamp
    save_file(load_file(path), path, metadata=metadata)
    assert run.read_done_marker(2) is not None
    assert not run.snapshot_provenance(3)[0]


def window_numbers(kl=0.02, top1=0.8):
    return {"held": {"kl_parent_child_mean": kl, "top1_agreement": top1}}


@pytest.mark.parametrize("mode", ["gate", "record"])
@pytest.mark.parametrize("corruption", ["context", "missing_context", "version", "constants", "prefix",
                                        "nan", "negative", "non_object"])
def test_g4_does_not_use_or_overwrite_incompatible_calibration(lineage_markers, mode, corruption):
    run, _, entries = lineage_markers
    run.receipt["layers"] = entries
    run.args.acceptance = mode
    first = run.window_acceptance(window_numbers())
    assert first["verdict"] == "calibrated"
    path = Path(first["calibration_written"])
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if corruption == "context":
        calibration["resume_context"] = {"old": "fp32 diagnostic"}
    elif corruption == "missing_context":
        calibration.pop("resume_context", None)
    elif corruption == "version":
        calibration["version"] = "old"
    elif corruption == "constants":
        calibration["constants"] = {}
    elif corruption == "prefix":
        calibration["installed_ffn_sha256"] = {"1": "f" * 64}
    elif corruption == "nan":
        calibration["kl_per_fitted_layer"] = float("nan")
    elif corruption == "negative":
        calibration["held_top1_agreement"] = -1.0
    else:
        calibration = []
    kdp.write_receipt(str(path), calibration)
    before = path.read_bytes()
    result = run.window_acceptance(window_numbers())
    assert result["verdict"] == ("refused" if mode == "gate" else "not_evaluated")
    assert "incompatible" in result["note"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("kl,top1", [(None, 0.8), (0.02, None), (float("nan"), 0.8),
                                    (-0.1, 0.8), (0.02, float("inf")), (0.02, 1.1)])
def test_g4_gate_fails_closed_on_unavailable_or_invalid_held_scores(lineage_markers, kl, top1):
    run, _, entries = lineage_markers
    run.receipt["layers"] = entries
    run.args.acceptance = "gate"
    got = run.window_acceptance(window_numbers(kl, top1))
    assert got["verdict"] == "refused"
    assert not (Path(run.receipt_dir) / kdp.ACCEPTANCE_CALIBRATION_NAME).exists()


def test_g4_unchanged_calibrated_prefix_can_extend_without_raising_thresholds(lineage_markers):
    run, _, entries = lineage_markers
    complete_installed = dict(run.installed)
    run.installed = {1: complete_installed[1]}
    run.receipt["layers"] = {"1": entries["1"]}
    first = run.window_acceptance(window_numbers())
    path = Path(first["calibration_written"])
    before = path.read_bytes()
    run.installed = complete_installed
    run.receipt["layers"] = entries
    run.args.acceptance = "gate"
    later = run.window_acceptance(window_numbers())
    assert later["verdict"] == "accepted"
    assert path.read_bytes() == before
    assert later["calibration"]["layers_folded"] == [1]


@pytest.mark.parametrize("changed", ["config", "index", "shard_metadata"])
def test_fresh_invocation_detects_changed_source_files(lineage_markers, tmp_path, changed):
    previous, argv, entries = lineage_markers
    source = tmp_path / "source-copy"
    shutil.copytree(previous.args.src_dir, source)
    argv = list(argv)
    argv[argv.index("--src-dir") + 1] = str(source)
    run = make_run(argv)
    run.installed = dict(previous.installed)
    for layer in (1, 2):
        path = Path(run.done_marker_path(layer))
        marker = json.loads(path.read_text(encoding="utf-8"))
        run.write_done_marker(layer, marker["artefact"], marker["artefact_sha256"],
                              {"moe": True, "fold_metrics": marker["fold_metrics"]})
    pair = kdp.Pair(torch.zeros(2, 64), torch.zeros(2, 0, 64))
    run.save_dual_snapshot(3, pair, pair.clone())
    run.receipt["layers"] = entries
    kdp.write_receipt(str(Path(run.receipt_dir) / kdp.PARTIAL_RECEIPT_NAME), run.receipt)
    if changed == "shard_metadata":
        path = source / next(iter(run.index.weight_map.values()))
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    else:
        path = source / ("config.json" if changed == "config" else "model.safetensors.index.json")
        content = json.loads(path.read_text(encoding="utf-8"))
        content["resume_context_test"] = "changed"
        path.write_text(json.dumps(content), encoding="utf-8")
    resumed = make_run(argv + ["--resume"])
    assert resumed.resume_context != run.resume_context
    assert resumed.read_done_marker(1) is None
    assert not resumed.snapshot_provenance(3)[0]
    assert not resumed.receipt["layers"]


@pytest.mark.parametrize("changed", ["package", "backend_flag"])
def test_kernel_package_and_backend_settings_change_resume_key(lineage_markers, monkeypatch, changed):
    run, argv = prepared_resume(lineage_markers)
    if changed == "package":
        monkeypatch.setattr(kdp.importlib.metadata, "version", lambda name: "changed-test-version")
    else:
        before = torch.backends.cuda.math_sdp_enabled()
        monkeypatch.setattr(torch.backends.cuda, "math_sdp_enabled", lambda: not before)
    resumed = make_run(argv + ["--resume"])
    assert resumed.resume_context != run.resume_context
    assert resumed.read_done_marker(1) is None
    assert not resumed.snapshot_provenance(3)[0]


def test_gate_mode_surfaces_g4_context_refusal_in_partial_receipt(tmp_path, uniform_fixture, corpus_files):
    out, receipts = tmp_path / "out", tmp_path / "receipts"
    argv = base_argv(uniform_fixture["dir"], corpus_files, out, receipts, rung=512, acceptance="gate")
    first = run_pass(argv)
    calibration = Path(first["window_acceptance"]["calibration_written"])
    content = json.loads(calibration.read_text(encoding="utf-8"))
    content["resume_context"] = {"old": "different runtime"}
    kdp.write_receipt(str(calibration), content)
    before = calibration.read_bytes()
    with pytest.raises(kdp.Refusal, match="incompatible calibration"):
        run_pass(argv + ["--resume"])
    partial = json.loads((receipts / kdp.PARTIAL_RECEIPT_NAME).read_text(encoding="utf-8"))
    assert partial["refusal"]["gate"] == "G4_window_intelligence"
    assert partial["window_acceptance"]["verdict"] == "refused"
    assert calibration.read_bytes() == before


def test_g4_counts_validated_installed_prefix_when_prior_receipt_is_missing(lineage_markers):
    run, _, entries = lineage_markers
    run.receipt["layers"] = {"2": entries["2"]}
    result = run.window_acceptance(window_numbers(kl=0.04))
    assert result["layers_folded"] == [1, 2]
    assert result["receipt_layers_missing"] == [1]
    assert result["kl_per_fitted_layer"] == pytest.approx(0.02)
    assert result["calibration"]["installed_ffn_sha256"] == {str(k): v for k, v in run.installed.items()}


def test_g4_preserves_and_normalizes_only_negative_fp32_kl_roundoff(lineage_markers):
    run, _, entries = lineage_markers
    run.receipt["layers"] = entries
    run.args.acceptance = "gate"
    # Reproduced from kl_top_agreement on nearly identical FP32 logits.
    raw_kl = -3.5287033739450635e-08
    result = run.window_acceptance(window_numbers(raw_kl, 1.0))
    assert result["verdict"] == "calibrated"
    assert result["raw_held_kl_parent_child_mean"] == raw_kl
    assert result["held_kl_parent_child_mean"] == 0.0
    assert result["kl_roundoff_normalized"] is True
    assert result["calibration"]["raw_held_kl_parent_child_mean"] == raw_kl
    assert result["calibration"]["kl_per_fitted_layer"] == 0.0
    assert run.window_acceptance(window_numbers(-1e-4, 1.0))["verdict"] == "refused"


@pytest.mark.parametrize("rung", [None, "64", "bad", True, 64.0, 128])
def test_moe_marker_requires_an_explicit_matching_integer_rung(lineage_markers, rung):
    run, _, _ = lineage_markers
    path = Path(run.done_marker_path(1))
    marker = json.loads(path.read_text(encoding="utf-8"))
    if rung is None:
        marker.pop("rung")
    else:
        marker["rung"] = rung
    kdp.write_receipt(str(path), marker)
    assert run.read_done_marker(1) is None
    assert run.read_done_marker(2) is None


@pytest.mark.parametrize("stream", ["teacher", "folded"])
@pytest.mark.parametrize("rung", [None, "bad", "064", "128"])
def test_snapshot_with_folds_requires_both_matching_rung_stamps(lineage_markers, stream, rung):
    run, _ = prepared_resume(lineage_markers)
    path = run.snapshot_path(3, stream)
    metadata = kdp.read_metadata(path)
    if rung is None:
        metadata.pop("rung")
    else:
        metadata["rung"] = rung
    save_file(load_file(path), path, metadata=metadata)
    assert run.read_done_marker(2) is not None
    assert not run.snapshot_provenance(3)[0]


def test_g4_refuses_calibration_that_omits_an_installed_ancestor(lineage_markers):
    run, _, entries = lineage_markers
    run.receipt["layers"] = entries
    result = run.window_acceptance(window_numbers())
    path = Path(result["calibration_written"])
    calibration = json.loads(path.read_text(encoding="utf-8"))
    calibration["layers_folded"] = [2]
    calibration["installed_ffn_sha256"] = {"2": run.installed[2]}
    kdp.write_receipt(str(path), calibration)
    run.args.acceptance = "gate"
    assert run.window_acceptance(window_numbers())["verdict"] == "refused"
