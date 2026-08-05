"""Seal and contract tests for the chimera-omni exact-4x toy fold runner.

Two things are checked here.  The first is the runner's own machinery: its
receipt seal, its exact-4x accounting, its stimulus/row helpers, and the
refusals it owes a caller.  The second is the *sealed report itself* --
``reports/chimera-omni-fold-readiness-20260804`` -- re-verified from disk:
every receipt reseals, every file digest in the manifest still matches the
bytes on disk, and every emitted artifact is nonempty.  A report that drifts
from its manifest fails here rather than being quoted in a summary.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "run_chimera_omni_toy_fold.py"
REPORT = ROOT / "reports" / "chimera-omni-fold-readiness-20260804"
RECEIPTS = (
    "configuration.json",
    "parent.json",
    "inspection.json",
    "budget.json",
    "plan.json",
    "fold.json",
    "behavior.json",
    "complete.json",
    "manifest.json",
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_chimera_omni_toy_fold", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


# ---------------------------------------------------------------------------
# Runner machinery
# ---------------------------------------------------------------------------


def test_receipts_seal_and_detect_tampering() -> None:
    sealed = runner.seal_receipt({"stage": "x", "value": 3})

    assert runner.verify_receipt(sealed)
    assert not runner.verify_receipt({**sealed, "value": 4})
    assert not runner.verify_receipt({**sealed, "receipt_sha256": "0" * 64})


def test_the_declared_compaction_is_exactly_four_times() -> None:
    assert runner.PARENT_HIDDEN_SIZE == runner.COMPACTION_FACTOR * (
        runner.CHILD_HIDDEN_SIZE
    )


def test_gap_closed_is_parent_relative_and_refuses_a_degenerate_span() -> None:
    assert runner.gap_closed(1.0, 0.5, 0.0) == pytest.approx(0.5)
    assert runner.gap_closed(1.0, 0.0, 0.0) == pytest.approx(1.0)
    assert runner.gap_closed(0.5, 0.4, 0.5) != runner.gap_closed(0.5, 0.4, 0.5)


def test_relative_rmse_is_zero_for_an_exact_match() -> None:
    target = torch.arange(12, dtype=torch.float64).reshape(3, 4)

    assert runner.relative_rmse(target.clone(), target) == pytest.approx(0.0)
    assert runner.relative_rmse(torch.zeros_like(target), target) == pytest.approx(1.0)


def test_tree_manifest_digests_every_file_and_honours_skips(tmp_path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.txt").write_bytes(b"alpha")
    (tmp_path / "nested" / "b.txt").write_bytes(b"beta")
    (tmp_path / "skip.txt").write_bytes(b"gamma")

    rows = runner.tree_manifest(tmp_path, skip=("skip.txt",))

    assert [row["path"] for row in rows] == ["a.txt", "nested/b.txt"]
    assert rows[0]["bytes"] == 5
    assert rows[0]["sha256"] == runner.file_sha256(tmp_path / "a.txt")


def test_a_missing_chimera_checkout_fails_closed_with_the_override(tmp_path) -> None:
    with pytest.raises(runner.ChimeraOmniFoldError) as error:
        runner.import_chimera_omni(tmp_path)
    assert "not a chimera-omni source tree" in str(error.value)


def test_parameter_map_refuses_models_with_different_tensor_names() -> None:
    left = torch.nn.Linear(4, 4, bias=False)
    right = torch.nn.Linear(4, 4, bias=True)

    with pytest.raises(runner.ChimeraOmniFoldError) as error:
        runner.parameter_map(left, right)
    assert "not a pure residual-width contraction" in str(error.value)


# ---------------------------------------------------------------------------
# The sealed report
# ---------------------------------------------------------------------------


needs_report = pytest.mark.skipif(
    not (REPORT / "complete.json").is_file(),
    reason=(
        "run experiments/run_chimera_omni_toy_fold.py to produce "
        f"{REPORT.name} before this check can re-verify it"
    ),
)


def _document(name: str) -> dict:
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


@needs_report
def test_every_sealed_receipt_reseals() -> None:
    for name in RECEIPTS:
        document = _document(name)
        assert runner.verify_receipt(document), name
        assert document["schema"] == runner.RUNNER_SCHEMA


@needs_report
def test_the_manifest_still_matches_the_bytes_on_disk() -> None:
    manifest = _document("manifest.json")
    rows = {row["path"]: row for row in manifest["files"]}
    assert rows, "the manifest is empty"
    for name, row in rows.items():
        path = REPORT / name
        assert path.is_file(), name
        assert int(path.stat().st_size) == row["bytes"], name
        assert runner.file_sha256(path) == row["sha256"], name
    present = {
        path.relative_to(REPORT).as_posix()
        for path in REPORT.rglob("*")
        if path.is_file()
    }
    assert present - set(rows) <= set(manifest["skipped"])


@needs_report
def test_the_report_states_an_exact_four_times_compaction() -> None:
    budget = _document("budget.json")
    complete = _document("complete.json")

    assert budget["exact_four_times"] is True
    assert budget["residual_from_exact"] == 0
    assert budget["parent_unique_parameters"] == 4 * budget[
        "child_unique_parameters"
    ]
    assert budget["quantization"] is False
    assert budget["sparsity"] is False
    assert budget["storage_sharing"] is False
    assert budget["tensors_exactly_four_times"] == budget["tensor_count"]
    assert complete["exact_four_times"] is True
    assert complete["parent_unique_parameters"] == budget[
        "parent_unique_parameters"
    ]


@needs_report
def test_the_fold_installed_every_tensor_without_a_gradient_step() -> None:
    fold = _document("fold.json")

    assert fold["fold_only_guard"]["clean"] is True
    assert fold["fold_only_guard"]["optimizer_steps"] == 0
    assert fold["fold_only_guard"]["backward_calls"] == 0
    assert fold["fold_only_guard"]["autograd_grad_calls"] == 0
    assert fold["parent_state_unchanged"] is True
    assert fold["every_child_tensor_installed_once"] is True
    assert fold["blank_child_state_sha256"] != fold["folded_child_state_sha256"]
    assert fold["parent_forward_calls"] == 1
    assert set(fold["install_methods"]) <= {
        "chart_transport_exact",
        "closed_form_ridge_linear",
        "closed_form_diagonal_rms_norm",
        "unexercised_module_deterministic_install",
    }
    targets = [row["target_parameter"] for row in fold["actions"]]
    assert len(targets) == len(set(targets)) == fold["installed_tensor_count"]
    assert fold["chart"]["orthonormality_defect"] < 1.0e-9
    assert fold["chart"]["per_boundary_chart_set"] is False


@needs_report
def test_the_report_names_the_families_the_fold_did_not_attempt() -> None:
    configuration = _document("configuration.json")
    complete = _document("complete.json")

    assert set(configuration["avoided_block_families"]) == {
        "kda",
        "csa",
        "hca",
        "gated_mla",
    }
    assert configuration["avoided_channel_mixers"] == ["stable_latent_moe"]
    assert complete["avoided_block_families"] == configuration[
        "avoided_block_families"
    ]
    inspection = _document("inspection.json")
    assert inspection["parent_metadata"]["kernel_gap_block_kinds"] == ""
    assert inspection["parent_metadata"]["block_kinds"] == "local,local,local,local"


@needs_report
def test_the_planner_refusal_is_recorded_rather_than_worked_around() -> None:
    plan = _document("plan.json")

    assert plan["planner_ready"] is False
    assert plan["gaps"]
    assert "hf-causal-decoder" in plan["interpretation"]


@needs_report
def test_the_child_emits_real_nonempty_media_and_beats_a_blank_child() -> None:
    behavior = _document("behavior.json")
    inspection = _document("inspection.json")

    assert inspection["emitted_media_interface_count"] == 6
    assert inspection["modality_output_nodes"] == [
        "output.native_media.emission_head"
    ]
    assert behavior["emitted_output_nonempty"] is True
    assert behavior["folded_beats_blank_on_native_media"] is True
    stimuli = {
        behavior["emitted_output_evidence"][name]["stimulus_sha256"]
        for name in ("parent", "folded_child", "blank_child")
    }
    assert len(stimuli) == 1, "the three models must answer the same stimulus"
    for name in ("parent", "folded_child", "blank_child"):
        evidence = behavior["emitted_output_evidence"][name]
        assert evidence["empty_files"] == []
        assert evidence["sizes"], name
        assert all(int(value) > 0 for value in evidence["sizes"].values())
    folded = behavior["folded_child_versus_parent"]
    blank = behavior["blank_child_versus_parent"]
    assert (
        folded["emitted_media_relative_rmse_vs_parent"]
        < blank["emitted_media_relative_rmse_vs_parent"]
    )
    assert folded["next_token_top1_agreement"] > blank["next_token_top1_agreement"]


@needs_report
def test_the_parent_actually_trained_under_chimeras_own_criterion() -> None:
    parent = _document("parent.json")

    assert parent["training"]["chimera_tiny_proof_status"] == "passed"
    assert parent["training"]["mean_last_10_loss"] < parent["training"][
        "mean_first_10_loss"
    ]
    assert parent["blank_parent_state_sha256"] != parent[
        "trained_parent_state_sha256"
    ]


@needs_report
def test_the_report_binds_the_chimera_omni_sources_it_read() -> None:
    configuration = _document("configuration.json")
    chimera = configuration["chimera_omni"]

    assert chimera["vendored"] is False
    assert chimera["read_only"] is True
    names = {row["path"] for row in chimera["files"]}
    assert "src/chimera_omni/modeling/chimera.py" in names
    assert "src/chimera_omni/modeling/layers.py" in names
    assert "architecture_manifest.json" in names
    for row in chimera["files"]:
        assert len(row["sha256"]) == 64
        assert row["bytes"] > 0
