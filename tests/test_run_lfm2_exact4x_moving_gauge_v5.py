from __future__ import annotations

import ast
from pathlib import Path

import pytest

from experiments import run_lfm2_exact4x_moving_gauge_v5 as runner
from neural_forge.dynamic_counted_child_state import (
    ContinuousChildStateAllocation,
    CountedChildStatePlan,
)


RUNNER_PATH = Path(runner.__file__)


def _exact_plan() -> CountedChildStatePlan:
    return CountedChildStatePlan(
        parent_unique_parameters=runner.CURRENT_LFM2_PARENT_PARAMETER_ELEMENTS,
        target_total_parameter_elements=(
            runner.CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS
        ),
        dense_model_parameter_elements=(
            runner.CURRENT_LFM2_CARRIER_DENSE_PARAMETER_ELEMENTS
        ),
        continuous_allocations=(
            ContinuousChildStateAllocation(
                state_id="semantic-carrier-route.c_in",
                parameter_elements=(
                    runner.CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS
                ),
                tensor_sha256="1" * 64,
                artifact_sha256="2" * 64,
            ),
            ContinuousChildStateAllocation(
                state_id="semantic-carrier-route.c_out",
                parameter_elements=(
                    runner.CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS
                ),
                tensor_sha256="3" * 64,
                artifact_sha256="4" * 64,
            ),
        ),
    )


def _request(tmp_path: Path) -> runner.Lfm2Exact4xMovingGaugeV5Request:
    paths = {}
    for name in sorted(runner.REQUIRED_LOCAL_PATH_KEYS):
        path = tmp_path / name
        path.mkdir()
        paths[name] = str(path)
    plan = _exact_plan()
    lineage = {
        name: f"{index + 1:064x}"
        for index, name in enumerate(sorted(runner.REQUIRED_LINEAGE_KEYS))
    }
    lineage["counted_child_state_plan"] = plan.fingerprint
    return runner.Lfm2Exact4xMovingGaugeV5Request(
        local_paths=paths,
        lineage_sha256s=lineage,
        counted_child_state_plan=plan,
        replay_devices=("cuda:0", "cuda:1"),
        recapture_checkpoint_batch_count=16,
    )


def test_phase_plan_is_the_complete_fit_freeze_recapture_heldout_order(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    plan = runner.build_v5_phase_plan(request)

    assert plan["phase_order"] == list(runner.PHASE_ORDER)
    assert plan["phase_count"] == len(runner.PHASE_ORDER) == 9
    assert plan["base_encoder_tensor_path"] == "basis_result.basis"
    assert plan["chart_modes"] == ["E0", "E0R", "E0A"]
    assert plan["target_arms"] == [
        "joint",
        "sequential_no_recapture",
        "sequential_conditioned",
    ]
    assert plan["publishable_candidate_count"] == 6
    assert runner.verify_v5_phase_plan(plan, request)

    phases = {row["name"]: row for row in plan["phases"]}
    assert phases["import_boundary_rows"]["details"]["logical_layer_indices"] == [
        0,
        1,
    ]
    assert phases["import_boundary_rows"]["details"][
        "physical_child_layer_indices"
    ] == [0, 13]
    assert phases["import_boundary_rows"]["reads_splits"] == []
    assert phases["import_feature_rows"]["reads_splits"] == []
    assert phases["import_feature_rows"]["details"]["source_cache_field_patterns"] == [
        "{phase}.layer.000.short_conv.context",
        "{phase}.layer.000.mlp.post_swiglu",
    ]
    assert (
        phases["import_feature_rows"]["details"]["parent_child_dtype_equality_required"]
        is False
    )
    assert phases["freeze_fit_charts"]["reads_splits"] == ["fit"]
    assert phases["freeze_fit_charts"]["details"]["heldout_rows_read"] is False
    assert phases["compare_frozen_candidates_on_heldout"]["reads_splits"] == ["heldout"]
    assert (
        phases["compare_frozen_candidates_on_heldout"]["details"][
            "candidate_hashes_already_frozen"
        ]
        is True
    )
    assert phases["compare_frozen_candidates_on_heldout"]["details"][
        "manual_semantic_review_status"
    ] == "pending"
    assert phases["compare_frozen_candidates_on_heldout"]["details"][
        "manual_semantic_review_performed_in_phase"
    ] is False
    assert (
        phases["build_baseline_target_arms"]["details"]["target_candidates"][
            "sequential_no_recapture"
        ]["publication_eligible"]
        is False
    )
    assert phases["freeze_fit_charts"]["details"]["chart_candidates"]["E0A"] == {
        "kind": "fixed_source_subspace_full_child_coordinate",
        "encoder": "E0@A",
        "intermediate_fit": True,
        "unrestricted_generalized_retained_as_diagnostic": True,
        "unrestricted_generalized_is_target_authority": False,
    }
    assert phases["stage_atomic_candidate_packs"]["details"][
        "replacement_tensor_names"
    ] == list(runner.REPLACEMENT_TENSOR_NAMES)


def test_phase_plan_rejects_a_self_resealed_policy_change(tmp_path: Path) -> None:
    request = _request(tmp_path)
    plan = dict(runner.build_v5_phase_plan(request))
    phases = [dict(value) for value in plan["phases"]]
    chart_phase = dict(phases[3])
    chart_phase["details"] = dict(chart_phase["details"])
    chart_phase["details"]["base_encoder"] = "untrusted.replacement"
    chart_phase.pop("phase_sha256")
    chart_phase["phase_sha256"] = runner.canonical_sha256(chart_phase)
    phases[3] = chart_phase
    plan["phases"] = phases
    plan.pop("plan_sha256")
    plan["plan_sha256"] = runner.canonical_sha256(plan)

    assert not runner.verify_v5_phase_plan(plan, request)


def test_recapture_plan_uses_child_only_canonical_chunks_and_restoration(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    plan = runner.build_v5_phase_plan(request)
    phase = next(
        row
        for row in plan["phases"]
        if row["name"] == "checkpoint_conditioned_child_recapture"
    )
    ranges_by_chart = phase["details"]["checkpoint_ranges_by_chart"]

    assert phase["runtime_apis"] == [
        "child_only_replay",
        "conditioned_recapture_phase",
        "child_state_fingerprint",
    ]
    assert (
        phase["child_forward_policy"]
        == "required_child_only_or_authenticated_checkpoint_reuse"
    )
    assert phase["parent_forward_count"] == 0
    assert phase["details"]["parent_model_supplied"] is False
    assert phase["details"]["full_fingerprint_before_equals_after"] is True
    assert phase["details"]["transport_captured_splits"] == ["fit", "heldout"]
    assert phase["details"]["solver_visible_splits"] == []
    assert set(ranges_by_chart) == set(runner.CHART_MODES)
    assert phase["details"]["checkpoint_count"] == 48
    for chart_mode, ranges in ranges_by_chart.items():
        assert len(ranges) == 16
        assert ranges[0]["batch_start"] == 0
        assert ranges[-1]["batch_stop"] == runner.EXPECTED_BATCH_COUNT
        assert all(row["batch_stop"] - row["batch_start"] == 16 for row in ranges)
        assert all(str(tmp_path / "checkpoint_root") in row["path"] for row in ranges)
        assert all(chart_mode in row["path"] for row in ranges)


def test_request_binds_exact_counted_plan_and_rejects_uri_or_wrong_budget(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    assert (
        request.counted_child_state_plan.fingerprint
        == request.lineage_sha256s["counted_child_state_plan"]
    )
    assert request.verify_authentication()

    wrong = CountedChildStatePlan(
        parent_unique_parameters=runner.CURRENT_LFM2_PARENT_PARAMETER_ELEMENTS,
        target_total_parameter_elements=(
            runner.CURRENT_LFM2_CARRIER_TARGET_PARAMETER_ELEMENTS + 1
        ),
        dense_model_parameter_elements=(
            runner.CURRENT_LFM2_CARRIER_DENSE_PARAMETER_ELEMENTS + 1
        ),
        continuous_allocations=_exact_plan().continuous_allocations,
    )
    with pytest.raises(
        runner.Lfm2Exact4xMovingGaugeV5Error,
        match="exact-4x counted plan",
    ):
        runner.Lfm2Exact4xMovingGaugeV5Request(
            local_paths=request.local_paths,
            lineage_sha256s={
                **request.lineage_sha256s,
                "counted_child_state_plan": wrong.fingerprint,
            },
            counted_child_state_plan=wrong,
        )

    with pytest.raises(
        runner.Lfm2Exact4xMovingGaugeV5Error,
        match="ordinary local path",
    ):
        runner.Lfm2Exact4xMovingGaugeV5Request(
            local_paths={
                **request.local_paths,
                "published_child": "https://example.invalid/model",
            },
            lineage_sha256s=request.lineage_sha256s,
            counted_child_state_plan=request.counted_child_state_plan,
        )


def test_local_preflight_keeps_all_mutable_roots_outside_r12(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    report = runner.authenticate_v5_local_preflight(request)

    assert report["immutable_inputs_exist"] is True
    assert report["mutable_roots_are_outside_immutable_inputs"] is True
    assert report["hugging_face_loader_used"] is False
    assert report["r12_mutation_allowed"] is False

    nested = Path(request.local_paths["published_child"]) / "bad-output"
    bad_paths = {**request.local_paths, "output_root": str(nested)}
    bad_request = runner.Lfm2Exact4xMovingGaugeV5Request(
        local_paths=bad_paths,
        lineage_sha256s=request.lineage_sha256s,
        counted_child_state_plan=request.counted_child_state_plan,
    )
    with pytest.raises(
        runner.Lfm2Exact4xMovingGaugeV5Error,
        match="would alter immutable input",
    ):
        runner.authenticate_v5_local_preflight(bad_request)


def test_phase_machine_rejects_missing_handlers_before_any_execution(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    called = False

    def handler(_phase, _completed):
        nonlocal called
        called = True
        raise AssertionError("must not run")

    with pytest.raises(
        runner.Lfm2Exact4xMovingGaugeV5RuntimeUnavailableError,
        match="Every v5 runtime handler",
    ):
        runner.run_v5_phase_machine(
            request,
            {runner.PHASE_ORDER[0]: handler},
            logger=lambda *_args, **_kwargs: None,
        )
    assert called is False


def test_phase_machine_enforces_child_only_recapture_and_heldout_barrier(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    calls: list[str] = []
    handlers = {}

    for name in runner.PHASE_ORDER:

        def handler(phase, completed, *, name=name):
            calls.append(name)
            recapture = name == "checkpoint_conditioned_child_recapture"
            heldout = name == "compare_frozen_candidates_on_heldout"
            ready = name == "stage_atomic_candidate_packs"
            return runner.V5PhaseOutcome(
                phase_name=name,
                phase_sha256=phase["phase_sha256"],
                upstream_artifact_sha256s={
                    dependency: completed[dependency].artifact_sha256
                    for dependency in phase["dependencies"]
                },
                evidence={"phase": name, "authenticated": True},
                child_forward_count=256 if recapture else 0,
                heldout_rows_read=heldout,
                temporary_child_mutation_performed=recapture,
                child_state_restored=recapture,
                child_state_before_sha256="a" * 64 if recapture else None,
                child_state_after_sha256="a" * 64 if recapture else None,
                ready_for_atomic_apply=ready,
            )

        handlers[name] = handler

    outcomes, receipt = runner.run_v5_phase_machine(
        request,
        handlers,
        logger=lambda *_args, **_kwargs: None,
    )

    assert calls == list(runner.PHASE_ORDER)
    assert tuple(outcomes) == runner.PHASE_ORDER
    assert receipt["all_phases_executed_once"] is True
    assert receipt["source_r12_mutated"] is False
    assert receipt["candidate_packs_installed"] is False
    assert receipt["candidate_packs_ready_for_atomic_apply"] is True
    assert receipt["operation_audit"] == dict(runner.FORBIDDEN_OPERATION_AUDIT)


def test_phase_machine_rejects_heldout_read_before_candidate_freeze(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    handlers = {}

    for name in runner.PHASE_ORDER:

        def handler(phase, completed, *, name=name):
            recapture = name == "checkpoint_conditioned_child_recapture"
            return runner.V5PhaseOutcome(
                phase_name=name,
                phase_sha256=phase["phase_sha256"],
                upstream_artifact_sha256s={
                    dependency: completed[dependency].artifact_sha256
                    for dependency in phase["dependencies"]
                },
                evidence={"phase": name},
                child_forward_count=1 if recapture else 0,
                heldout_rows_read=name == "freeze_fit_charts",
                temporary_child_mutation_performed=recapture,
                child_state_restored=recapture,
                child_state_before_sha256="b" * 64 if recapture else None,
                child_state_after_sha256="b" * 64 if recapture else None,
                ready_for_atomic_apply=(name == "stage_atomic_candidate_packs"),
            )

        handlers[name] = handler

    with pytest.raises(
        runner.Lfm2Exact4xMovingGaugeV5Error,
        match="FIT/heldout freeze barrier",
    ):
        runner.run_v5_phase_machine(
            request,
            handlers,
            logger=lambda *_args, **_kwargs: None,
        )


def test_phase_machine_accepts_authenticated_full_checkpoint_hit_without_forward(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    handlers = {}

    for name in runner.PHASE_ORDER:

        def handler(phase, completed, *, name=name):
            recapture = name == "checkpoint_conditioned_child_recapture"
            return runner.V5PhaseOutcome(
                phase_name=name,
                phase_sha256=phase["phase_sha256"],
                upstream_artifact_sha256s={
                    dependency: completed[dependency].artifact_sha256
                    for dependency in phase["dependencies"]
                },
                evidence={"phase": name, "checkpoint_chain_authenticated": recapture},
                child_forward_count=0,
                heldout_rows_read=name == "compare_frozen_candidates_on_heldout",
                authenticated_checkpoint_reuse=recapture,
                ready_for_atomic_apply=(name == "stage_atomic_candidate_packs"),
            )

        handlers[name] = handler

    outcomes, _receipt = runner.run_v5_phase_machine(
        request,
        handlers,
        logger=lambda *_args, **_kwargs: None,
    )

    recapture = outcomes["checkpoint_conditioned_child_recapture"]
    assert recapture.authenticated_checkpoint_reuse is True
    assert recapture.child_forward_count == 0
    assert recapture.temporary_child_mutation_performed is False


def test_candidate_pack_is_atomic_two_tensor_staging_not_install(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    request = _request(tmp_path)
    replacements = {
        runner.REPLACEMENT_TENSOR_NAMES[0]: torch.eye(2, dtype=torch.float32),
        runner.REPLACEMENT_TENSOR_NAMES[1]: torch.ones((2, 3), dtype=torch.float32),
    }
    receipts = {
        name: f"{index + 20:064x}"
        for index, name in enumerate(
            (
                "chart",
                "heldout_comparison",
                "lineage",
                "mlp_solution",
                "operator_solution",
                "target_plan",
            )
        )
    }

    path, receipt = runner.stage_atomic_candidate_pack(
        request,
        chart_mode="E0A",
        target_arm="sequential_conditioned",
        replacements=replacements,
        upstream_receipt_sha256s=receipts,
        child_state_before_sha256="c" * 64,
        child_state_after_sha256="c" * 64,
    )

    assert path.name == receipt["candidate_pack_sha256"]
    assert sorted(value.name for value in path.iterdir()) == [
        "receipt.json",
        "replacement-tensors.safetensors",
    ]
    assert receipt["replacement_tensor_count"] == 2
    assert receipt["replacement_tensor_names"] == list(runner.REPLACEMENT_TENSOR_NAMES)
    assert receipt["installed"] is False
    assert receipt["ready_for_atomic_apply"] is True
    assert receipt["source_child_state_preserved"] is True
    assert receipt["manual_semantic_review_status"] == "pending"

    with pytest.raises(
        runner.Lfm2Exact4xMovingGaugeV5Error,
        match="negative control cannot be packaged",
    ):
        runner.stage_atomic_candidate_pack(
            request,
            chart_mode="E0",
            target_arm="sequential_no_recapture",
            replacements=replacements,
            upstream_receipt_sha256s=receipts,
            child_state_before_sha256="c" * 64,
            child_state_after_sha256="c" * 64,
        )


def test_runner_has_no_parent_loader_network_or_training_substitute() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    imported_roots = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "load_lfm2_direct_local_model" not in called_names
    assert "from_pretrained" not in called_attributes
    assert not {
        "train",
        "backward",
        "grad",
        "prune",
        "quantize",
        "sparsify",
        "share_parameters",
    }.intersection(called_names | called_attributes)
    assert not {"requests", "httpx", "urllib", "huggingface_hub"}.intersection(
        imported_roots
    )
    assert all(
        value in (0, False) for value in runner.FORBIDDEN_OPERATION_AUDIT.values()
    )
