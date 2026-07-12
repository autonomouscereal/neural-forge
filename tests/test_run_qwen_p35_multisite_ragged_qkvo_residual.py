from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_qwen_p35_multisite_ragged_qkvo_residual as runner  # noqa: E402
from neural_forge.hf_assignment_local_qkvo_residual_adapter import (  # noqa: E402
    SparseAssignmentLocalQKVOResidualRegistry,
)
from neural_forge.hf_functional_residual_adapter import (  # noqa: E402
    FunctionalActionEvaluation,
)
from neural_forge.hf_qkvo_factor_pack import (  # noqa: E402
    QKVO_ROLE_SCALE_PROVENANCE_SCHEMA,
    QKVO_SOURCE_PROVENANCE_SCHEMA,
    adapter_state_sha256,
    file_sha256,
    load_qkvo_factor_pack,
    write_qkvo_factor_pack,
)
from neural_forge.hf_ragged_assignment_local_qkvo_residual_adapter import (  # noqa: E402
    RaggedAssignmentLocalQKVOAccounting,
    RaggedAssignmentLocalQKVOResidualRegistry,
)


def test_exact_eight_redirect_lineage_and_uneven_schedule_budget() -> None:
    assert [value.source for value in runner.P35_TO_P34_REDIRECTS] == [
        (10, 2),
        (13, 2),
        (15, 2),
        (19, 4),
        (21, 3),
        (25, 1),
        (25, 5),
        (30, 1),
    ]
    assert [value.target for value in runner.P35_TO_P34_REDIRECTS] == [
        (15, 5),
        (35, 0),
        (19, 5),
        (21, 6),
        (7, 6),
        (26, 4),
        (20, 6),
        (14, 4),
    ]
    assert runner.DEFAULT_RANK_SCHEDULES["anchor-economy-r36"] == (
        4,
        4,
        4,
        8,
        4,
        4,
        4,
        4,
    )
    accounting = RaggedAssignmentLocalQKVOAccounting.build(
        site_ranks=runner.DEFAULT_RANK_SCHEDULES["anchor-economy-r36"],
        hidden_size=2560,
        query_group_width=512,
        head_dim=128,
        logical_layers=36,
        physical_groups=8,
        checkpoint_element_size=2,
        role_scale_table_new_for_redirects=True,
    )
    assert accounting.removed_parameters == 26_214_400
    assert accounting.role_scale_parameters == 1_152
    assert accounting.adapter_parameters == 414_720
    assert accounting.net_parameters_saved == 25_798_528
    assert accounting.net_tensor_bytes_saved == 51_597_056
    report = SimpleNamespace(
        accounting=accounting,
        adapter_parameter_count=accounting.adapter_parameters,
        model_parameter_count_before=3_969_356_928,
        model_parameter_count_installed=3_969_771_648,
    )
    assert runner._assert_candidate_accounting(SimpleNamespace(report=report)) is (
        accounting
    )


def test_multisite_lineage_requires_exact_280_to_272_and_permutations() -> None:
    removed = {
        value.source_layer * 8 + value.source_group
        for value in runner.P35_TO_P34_REDIRECTS
    }
    parent_sources = set(range(288)).difference(set(range(1, 9)))
    assert len(parent_sources) == 280
    assert removed.issubset(parent_sources)
    parent = SimpleNamespace(
        key_value_heads=8,
        logical_layers=36,
        prototype_count=280,
        prototype_source_indices=tuple(sorted(parent_sources)),
    )
    proposed = SimpleNamespace(
        prototype_count=272,
        prototype_source_indices=tuple(sorted(parent_sources - removed)),
    )
    permutation = torch.arange(8).repeat(36, 1)
    parent_registry = SimpleNamespace(group_permutations=permutation)
    proposed_registry = SimpleNamespace(group_permutations=permutation.clone())

    audit = runner._assert_multisite_lineage(
        parent,
        proposed,
        parent_registry,
        proposed_registry,
        runner.P35_TO_P34_REDIRECTS,
    )
    assert audit["parent_prototype_count"] == 280
    assert audit["proposed_prototype_count"] == 272
    assert audit["source_set_is_parent_minus_exactly_eight"] is True

    proposed_registry.group_permutations[0, 0] = 1
    with pytest.raises(runner.QwenP35MultisiteRaggedQKVOError, match="lineage"):
        runner._assert_multisite_lineage(
            parent,
            proposed,
            parent_registry,
            proposed_registry,
            runner.P35_TO_P34_REDIRECTS,
        )


def _anchor_pack(tmp_path: Path):
    registry = SparseAssignmentLocalQKVOResidualRegistry(
        [[19, 4]],
        logical_layers=36,
        physical_groups=8,
        hidden_size=2560,
        query_group_width=512,
        head_dim=128,
        rank=8,
        dtype=torch.float16,
        device="cpu",
    )
    with torch.no_grad():
        for role in registry._ROLES:
            getattr(registry, f"{role}_left").fill_(0.001)
    state = registry.state_dict()
    state_sha = adapter_state_sha256(state)
    factor = tmp_path / "anchor.safetensors"
    manifest = tmp_path / "anchor.json"
    write_qkvo_factor_pack(
        factor,
        manifest,
        state,
        selected_adapter_state_sha256=state_sha,
        site={"layer_index": 19, "physical_group": 4},
        rank=8,
        registry_geometry={"logical_layers": 36, "physical_groups": 8},
        role_scale_provenance={
            "schema": QKVO_ROLE_SCALE_PROVENANCE_SCHEMA,
            "q": 1.0,
            "k": 1.0,
            "v": 1.0,
            "o": 1.0,
            "role_scales_trainable": False,
            "table_sha256": "a" * 64,
        },
        provenance={
            "schema": QKVO_SOURCE_PROVENANCE_SCHEMA,
            "source_report_file_sha256": (
                runner.replay_runner.EXPECTED_SOURCE_REPORT_FILE_SHA256
            ),
            "source_report_sha256": (
                runner.replay_runner.EXPECTED_SOURCE_REPORT_SHA256
            ),
            "fixture_provenance_sha256": "d" * 64,
            "calibration_evaluation_sha256": "e" * 64,
            "heldout_evaluation_sha256": "f" * 64,
            "source_runtime": {
                "manifest_sha256": (
                    runner.replay_runner.EXPECTED_CODEBOOK_MANIFEST_SHA256
                )
            },
        },
    )
    return load_qkvo_factor_pack(factor, manifest), state_sha


def test_exact_anchor_pack_seeds_only_rank8_anchor_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    anchor, state_sha = _anchor_pack(tmp_path)
    monkeypatch.setattr(runner, "EXPECTED_ANCHOR_STATE_SHA256", state_sha)
    ragged = RaggedAssignmentLocalQKVOResidualRegistry(
        [[15, 2], [19, 4]],
        (4, 8),
        logical_layers=36,
        physical_groups=8,
        hidden_size=2560,
        query_group_width=512,
        head_dim=128,
        dtype=torch.float16,
        device="cpu",
        initialization_seed=99,
    )
    first_before = {
        name: getattr(ragged, name)[0].detach().clone()
        for name in (
            "q_left",
            "q_right",
            "k_left",
            "k_right",
            "v_left",
            "v_right",
            "o_left",
            "o_right",
        )
    }
    runner._seed_anchor_site(SimpleNamespace(adapter_registry=ragged), anchor)
    for name, value in first_before.items():
        assert torch.equal(getattr(ragged, name)[0], value)
    for name, value in anchor.factors.items():
        assert torch.equal(
            getattr(ragged, name)[1],
            value.squeeze(0),
        )


def test_schedule_selection_prefers_behavior_then_lower_parameter_count() -> None:
    evaluation = FunctionalActionEvaluation(
        split="calibration-selection",
        example_count=2,
        total_loss=1.0,
        hard_action_loss=1.0,
        distribution_loss=0.0,
        parent_action_top1_rate=0.75,
        mean_parent_action_log_probability=-1.0,
        mean_parent_action_rank=1.5,
        used_for_selection=True,
        prefix_sources=("parent-greedy", "student-greedy"),
    )
    assert runner._schedule_key(evaluation, 100) < runner._schedule_key(evaluation, 200)
    different_continuous_reduction = replace(
        evaluation,
        total_loss=1.25,
        hard_action_loss=1.2,
        distribution_loss=0.5,
        mean_parent_action_log_probability=-1.2,
    )
    assert runner._discrete_evaluation_contract(
        evaluation
    ) == runner._discrete_evaluation_contract(different_continuous_reduction)
    different_decision = replace(evaluation, parent_action_top1_rate=0.5)
    assert runner._discrete_evaluation_contract(
        evaluation
    ) != runner._discrete_evaluation_contract(different_decision)
    rows = (
        {
            "schedule": "calibration-best-but-fit-failed",
            "fit_gate_passed": False,
            "all_parameter_tensors_received_nonzero_gradient": True,
            "selection_key": [-1.0, 1.0, 1],
        },
        {
            "schedule": "fit-passed",
            "fit_gate_passed": True,
            "all_parameter_tensors_received_nonzero_gradient": True,
            "selection_key": [-0.5, 2.0, 2],
        },
    )
    assert [
        value["schedule"] for value in runner._fit_eligible_candidate_rows(rows)
    ] == ["fit-passed"]


def test_runner_has_no_full_checkpoint_save_and_heldout_follows_lock() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "torch.save(" not in source
    assert ".save_pretrained(" not in source
    assert "optimizer.state_dict" not in source
    assert source.index('event("winner-schedule-locked"') < source.index(
        'event("locked-heldout-drivers", heldout=True)'
    )
    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--codebook-checkpoint",
            "p35",
            "--tokenizer",
            "tokenizer",
            "--anchor-factor-pack",
            "anchor.safetensors",
            "--anchor-factor-manifest",
            "anchor.json",
            "--bundle-output",
            "bundle",
            "--output",
            "report.json",
        ]
    )
    assert args.dtype == "float16"
    assert args.device == "cuda:0"


def test_exact_source_contract_binds_anchor_checkpoint_fixture_and_environment(
    tmp_path: Path,
) -> None:
    anchor, _state_sha = _anchor_pack(tmp_path)
    split = {"provenance_sha256": "d" * 64}
    source = {"manifest_sha256": runner.replay_runner.EXPECTED_CODEBOOK_MANIFEST_SHA256}
    environment = {
        "python": runner.replay_runner.EXPECTED_PYTHON_VERSION,
        "torch": runner.replay_runner.EXPECTED_TORCH_VERSION,
        "device": "cuda:0",
        "dtype": "float16",
        "gpu_name": runner.replay_runner.EXPECTED_GPU_NAME,
    }
    audit = runner._assert_exact_source_contract(
        anchor,
        split_provenance=split,
        source_runtime=source,
        execution_environment=environment,
    )
    assert audit["base_codebook_manifest_sha256"] == source["manifest_sha256"]

    with pytest.raises(runner.QwenP35MultisiteRaggedQKVOError, match="environment"):
        runner._assert_exact_source_contract(
            anchor,
            split_provenance=split,
            source_runtime=source,
            execution_environment={**environment, "gpu_name": "wrong"},
        )


def test_output_preflight_requires_absent_disjoint_sibling_paths(
    tmp_path: Path,
) -> None:
    runner._preflight_output_paths(tmp_path / "bundle", tmp_path / "report.json")
    with pytest.raises(runner.QwenP35MultisiteRaggedQKVOError, match="disjoint"):
        runner._preflight_output_paths(
            tmp_path / "bundle", tmp_path / "bundle" / "report.json"
        )


def test_owned_bundle_cleanup_refuses_foreign_bytes(tmp_path: Path) -> None:
    directory = tmp_path / "bundle"
    directory.mkdir()
    factor = directory / "site_00.safetensors"
    site_manifest = directory / "site_00.manifest.json"
    factor.write_bytes(b"factor")
    site_manifest.write_bytes(b"manifest")
    manifest = {
        "sites": [
            {
                "factor_file": factor.name,
                "factor_file_sha256": file_sha256(factor),
                "manifest_file": site_manifest.name,
                "manifest_file_sha256": file_sha256(site_manifest),
            }
        ]
    }
    (directory / "bundle.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    factor.write_bytes(b"foreign mutation")
    assert runner._remove_owned_factor_bundle(directory, manifest) is False
    assert directory.exists()

    factor.write_bytes(b"factor")
    assert runner._remove_owned_factor_bundle(directory, manifest) is True
    assert not directory.exists()


def test_prepared_p40_wrapper_uses_read_only_exact_inputs_and_fresh_outputs() -> None:
    wrapper = (
        Path(runner.__file__).parent
        / "remote_run_p35_multisite_ragged_qkvo_factor_bundle_p40_20260711.sh"
    ).read_text(encoding="utf-8")
    assert "Prepared only" in wrapper
    assert 'test "$GPU_MEMORY_USED" -lt 4096' in wrapper
    assert '"Tesla P40"' in wrapper
    assert '"$CHECKPOINT:/codebook:ro"' in wrapper
    assert '"$ANCHOR_PACK:/anchor.safetensors:ro"' in wrapper
    assert '"$ANCHOR_MANIFEST:/anchor.manifest.json:ro"' in wrapper
    assert 'test ! -e "$BUNDLE"' in wrapper
    assert 'test ! -e "$REPORT"' in wrapper
    assert "--bundle-output" in wrapper
    assert "--local-files-only" in wrapper


def test_main_removes_owned_bundle_when_post_run_report_mutation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = SimpleNamespace(
        bundle_output=tmp_path / "bundle",
        output=tmp_path / "report.json",
        fit_prompts_json=tmp_path / "fit.json",
        calibration_selection_prompts_json=tmp_path / "cal.json",
        heldout_evaluation_prompts_json=tmp_path / "heldout.json",
        fit_prompt_manifest=tmp_path / "fit.manifest.json",
        audited_prompt_manifest=tmp_path / "audited.manifest.json",
        transition_inventory=tmp_path / "transitions.json",
        anchor_factor_pack=tmp_path / "anchor.safetensors",
        anchor_factor_manifest=tmp_path / "anchor.manifest.json",
        max_prompt_tokens=256,
        max_vocabulary=200_000,
        max_logit_bytes=256 * 1024 * 1024,
        device="cuda:0",
        dtype="float16",
    )
    parser = SimpleNamespace(parse_args=lambda _argv: args)
    fixtures = SimpleNamespace(
        fit=("fit",),
        calibration=("cal",),
        heldout=("heldout",),
        provenance={"verified_exact_fixture_provenance": True},
    )
    environment = {
        "python": runner.replay_runner.EXPECTED_PYTHON_VERSION,
        "torch": runner.replay_runner.EXPECTED_TORCH_VERSION,
        "device": "cuda:0",
        "dtype": "float16",
        "gpu_name": runner.replay_runner.EXPECTED_GPU_NAME,
    }

    def fake_run(*_args, bundle_output: Path, **_kwargs):
        bundle_output.mkdir()
        factor = bundle_output / "site_00.safetensors"
        site_manifest = bundle_output / "site_00.manifest.json"
        factor.write_bytes(b"factor")
        site_manifest.write_bytes(b"manifest")
        manifest = {
            "sites": [
                {
                    "factor_file": factor.name,
                    "factor_file_sha256": file_sha256(factor),
                    "manifest_file": site_manifest.name,
                    "manifest_file_sha256": file_sha256(site_manifest),
                }
            ]
        }
        (bundle_output / "bundle.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return {"factor_bundle": manifest}

    monkeypatch.setattr(runner, "build_parser", lambda: parser)
    monkeypatch.setattr(
        runner.single_runner, "load_verified_prompt_splits", lambda *_a, **_k: fixtures
    )
    monkeypatch.setattr(
        runner.single_runner, "_encode_prompts", lambda _t, prompts, **_k: prompts
    )
    monkeypatch.setattr(
        runner.base,
        "_load_runtime_and_tokenizer",
        lambda _args: (
            object(),
            object(),
            {
                "manifest_sha256": (
                    runner.replay_runner.EXPECTED_CODEBOOK_MANIFEST_SHA256
                )
            },
        ),
    )
    monkeypatch.setattr(
        runner.replay_runner,
        "assert_exact_execution_environment",
        lambda *_a, **_k: environment,
    )
    monkeypatch.setattr(runner, "load_qkvo_factor_pack", lambda *_a: object())
    monkeypatch.setattr(runner, "run_qwen_p35_multisite_ragged_qkvo", fake_run)
    monkeypatch.setattr(
        runner,
        "qwen3_codebook_parameter_report",
        lambda _runtime: (_ for _ in ()).throw(
            RuntimeError("injected mutation failure")
        ),
    )

    with pytest.raises(RuntimeError, match="injected mutation failure"):
        runner.main([])
    assert not args.bundle_output.exists()
    assert not args.output.exists()
