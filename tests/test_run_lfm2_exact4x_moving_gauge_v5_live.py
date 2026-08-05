from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments import run_lfm2_exact4x_moving_gauge_v5 as runner
from experiments import run_lfm2_exact4x_moving_gauge_v5_live as live
from neural_forge.dynamic_counted_child_state import (
    ContinuousChildStateAllocation,
    CountedChildStatePlan,
)
from neural_forge.hf_lfm2_moving_gauge_residual_reclosure import (
    fingerprint_lfm2_moving_gauge_tensor,
)
from neural_forge import (
    hf_lfm2_moving_gauge_conditioned_recapture_phase_cache as conditioned,
)


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
                parameter_elements=runner.CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS,
                tensor_sha256="1" * 64,
                artifact_sha256="2" * 64,
            ),
            ContinuousChildStateAllocation(
                state_id="semantic-carrier-route.c_out",
                parameter_elements=runner.CURRENT_LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS,
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
        replay_devices=("cuda:0",),
        recapture_checkpoint_batch_count=16,
    )


def _provenance() -> dict[str, str]:
    return {
        name: f"{index + 101:064x}"
        for index, name in enumerate(sorted(live.REQUIRED_BOUNDARY_PROVENANCE_KEYS))
    }


def _bindings(**changes: object) -> live.Lfm2Exact4xMovingGaugeV5LiveBindings:
    values = {
        "trace_adapter": object(),
        "basis_phase_authority": SimpleNamespace(key_id="basis"),
        "boundary_phase_authority": SimpleNamespace(key_id="boundary"),
        "feature_phase_authority": SimpleNamespace(key_id="feature"),
        "conditioned_recapture_authority": SimpleNamespace(key_id="conditioned"),
        "provenance_sha256s": _provenance(),
        "v3_first_divergence_request": {},
        "v3_first_divergence_ledger": {},
        "basis_split_by_batch": {
            index: "heldout" if index % 4 == 2 else "fit"
            for index in range(runner.EXPECTED_BATCH_COUNT)
        },
        "parent_storage_dtype": "bfloat16",
        "child_storage_dtype": "float32",
        "child_feature_storage_dtypes": {
            "short_conv.context": "float32",
            "mlp.post_swiglu": "float32",
        },
    }
    values.update(changes)
    return live.Lfm2Exact4xMovingGaugeV5LiveBindings(**values)


def _typed_conditioned_request(
    runtime: live.Lfm2Exact4xMovingGaugeV5LiveRuntime,
    *,
    chart_mode: str,
    operator_solution: torch.Tensor,
    operator_report: dict[str, str],
) -> conditioned.Lfm2ConditionedRecapturePhaseRequest:
    def digest(label: str) -> str:
        return live.canonical_sha256({"label": label})

    descriptor = {
        "dtype": str(operator_solution.dtype).removeprefix("torch."),
        "shape": [int(size) for size in operator_solution.shape],
        "numel": int(operator_solution.numel()),
        "size_bytes": int(
            operator_solution.numel() * operator_solution.element_size()
        ),
        "sha256": fingerprint_lfm2_moving_gauge_tensor(operator_solution),
    }
    source_input_sha256s = tuple(
        digest(f"source-input-{index}")
        for index in range(runner.EXPECTED_BATCH_COUNT)
    )
    source_record_ids = tuple(
        f"record-{index:03d}" for index in range(runner.EXPECTED_BATCH_COUNT)
    )
    source_session = digest("source-session")
    basis_split_by_batch = tuple(
        runtime.bindings.basis_split_by_batch[index]
        for index in range(runner.EXPECTED_BATCH_COUNT)
    )
    split_identity = (
        conditioned.derive_lfm2_conditioned_recapture_split_identity(
            source_input_session_sha256=source_session,
            source_input_sha256s=source_input_sha256s,
            source_record_ids=source_record_ids,
            basis_split_by_batch=basis_split_by_batch,
        )
    )
    base_identity = digest("base-identity")
    base_state = digest("base-state")
    endpoint = digest(f"endpoint-{chart_mode}")
    conditioned_identity = digest(f"conditioned-identity-{chart_mode}")
    conditioned_state = digest(f"conditioned-state-{chart_mode}")
    operator_solution_sha256 = operator_report["report_sha256"]
    frozen_state = (
        conditioned.derive_lfm2_frozen_conditioned_child_state_receipt_sha256(
            base_child_identity_sha256=base_identity,
            base_child_state_dict_sha256=base_state,
            endpoint_chain_receipt_sha256=endpoint,
            operator_replacement_tensor_descriptor=descriptor,
            operator_solution_sha256=operator_solution_sha256,
            conditioned_child_identity_sha256=conditioned_identity,
            conditioned_child_state_dict_sha256=conditioned_state,
        )
    )
    widths = {
        "short_conv.context": 3,
        "operator.output": 2,
        "residual.pre": 2,
        "residual.post_operator": 2,
        "residual.post_layer": 2,
        "mlp.post_swiglu": 4,
        "mlp.down": 2,
    }
    contracts = tuple(
        conditioned.Lfm2ConditionedRecaptureFieldContract(
            field_name=name,
            source_dtype="float32",
            feature_width=widths[name],
        )
        for name in conditioned.LFM2_CONDITIONED_RECAPTURE_REQUIRED_FIELDS
    )
    return conditioned.Lfm2ConditionedRecapturePhaseRequest(
        v5_request_sha256=runtime.request.request_sha256,
        v5_lineage_sha256=live.canonical_sha256(
            dict(runtime.request.lineage_sha256s)
        ),
        base_child_identity_sha256=base_identity,
        base_child_state_dict_sha256=base_state,
        endpoint_chain_receipt_sha256=endpoint,
        operator_replacement_tensor_descriptor=descriptor,
        operator_solution_sha256=operator_solution_sha256,
        frozen_conditioned_child_state_receipt_sha256=frozen_state,
        conditioned_child_identity_sha256=conditioned_identity,
        conditioned_child_state_dict_sha256=conditioned_state,
        child_layer_topology_sha256=digest("topology"),
        chart_mode=chart_mode,
        child_replay_config_sha256=digest("replay-config"),
        child_replay_implementation_sha256=digest("replay-implementation"),
        child_replay_execution_sha256=digest("replay-execution"),
        child_replay_numeric_policy_sha256=digest("numeric-policy"),
        child_replay_capture_mode="prefix_by_prefix_canonical",
        source_input_session_sha256=source_session,
        source_input_sha256s=source_input_sha256s,
        source_record_ids=source_record_ids,
        basis_split_by_batch=basis_split_by_batch,
        split_identity_sha256=split_identity,
        field_contracts=contracts,
        runtime_device_type="cpu",
        runtime_floating_dtypes=("float32",),
        conditioned_recapture_spec_sha256=(
            conditioned.lfm2_conditioned_recapture_spec_sha256()
        ),
        implementation_sha256=(
            conditioned.lfm2_conditioned_recapture_implementation_sha256()
        ),
        checkpoint_hmac_key_id=(
            runtime.bindings.conditioned_recapture_authority.key_id
        ),
        checkpoint_batch_count=runtime.request.recapture_checkpoint_batch_count,
    )


def _overrides() -> tuple[dict[str, live.V5LivePhaseOverride], list[str]]:
    calls: list[str] = []
    overrides: dict[str, live.V5LivePhaseOverride] = {}

    for index, phase_name in enumerate(runner.PHASE_ORDER):

        def execute(
            runtime: live.Lfm2Exact4xMovingGaugeV5LiveRuntime,
            phase: object,
            completed: object,
            *,
            expected_name: str = phase_name,
            expected_index: int = index,
        ) -> live.V5LivePhaseExecution:
            assert phase["name"] == expected_name
            assert list(completed) == list(runner.PHASE_ORDER[:expected_index])
            assert list(runtime.artifacts) == list(
                runner.PHASE_ORDER[:expected_index]
            )
            calls.append(expected_name)
            is_cache_hit = expected_name == (
                "checkpoint_conditioned_child_recapture"
            )
            is_heldout = expected_name == (
                "compare_frozen_candidates_on_heldout"
            )
            is_stage = expected_name == "stage_atomic_candidate_packs"
            return live.V5LivePhaseExecution(
                artifact={"ordinal": expected_index},
                evidence={
                    "schema": live.LIVE_SCHEMA,
                    "phase": expected_name,
                    "observed_prior_artifact_count": len(runtime.artifacts),
                    "network_access": False,
                    "training_steps": 0,
                },
                heldout_rows_read=is_heldout,
                authenticated_checkpoint_reuse=is_cache_hit,
                ready_for_atomic_apply=is_stage,
            )

        overrides[phase_name] = execute
    return overrides, calls


def test_dependency_injected_handlers_preserve_order_data_flow_and_cache_hit(
    tmp_path: Path,
) -> None:
    overrides, calls = _overrides()
    runtime = live.Lfm2Exact4xMovingGaugeV5LiveRuntime(
        _request(tmp_path),
        _bindings(),
        phase_overrides=overrides,
    )
    handlers = runtime.handlers()

    outcomes, receipt = runner.run_v5_phase_machine(
        runtime.request,
        handlers,
        logger=lambda *_args, **_kwargs: None,
    )

    assert list(handlers) == list(runner.PHASE_ORDER)
    assert calls == list(runner.PHASE_ORDER)
    assert runtime.phase_call_order == list(runner.PHASE_ORDER)
    assert list(runtime.artifacts) == list(runner.PHASE_ORDER)
    recapture = outcomes["checkpoint_conditioned_child_recapture"]
    assert recapture.authenticated_checkpoint_reuse is True
    assert recapture.child_forward_count == 0
    assert recapture.temporary_child_mutation_performed is False
    assert outcomes["compare_frozen_candidates_on_heldout"].heldout_rows_read
    assert outcomes["stage_atomic_candidate_packs"].ready_for_atomic_apply
    assert receipt["operation_audit"] == dict(runner.FORBIDDEN_OPERATION_AUDIT)


def test_factory_boundary_has_no_parent_model_loader_forward_or_network_argument(
    tmp_path: Path,
) -> None:
    overrides, _calls = _overrides()
    kwargs = {
        "trace_adapter": object(),
        "basis_phase_authority": SimpleNamespace(key_id="basis"),
        "boundary_phase_authority": SimpleNamespace(key_id="boundary"),
        "feature_phase_authority": SimpleNamespace(key_id="feature"),
        "conditioned_recapture_authority": SimpleNamespace(key_id="conditioned"),
        "provenance_sha256s": _provenance(),
        "v3_first_divergence_request": {},
        "v3_first_divergence_ledger": {},
        "basis_split_by_batch": {
            index: "heldout" if index % 4 == 2 else "fit"
            for index in range(256)
        },
        "parent_storage_dtype": "bfloat16",
        "child_storage_dtype": "float32",
        "child_feature_storage_dtypes": {
            "short_conv.context": "float32",
            "mlp.post_swiglu": "float32",
        },
        "phase_overrides": overrides,
    }
    handlers = live.build_lfm2_exact4x_moving_gauge_v5_live_handlers(
        _request(tmp_path),
        **kwargs,
    )
    signature = inspect.signature(
        live.build_lfm2_exact4x_moving_gauge_v5_live_runtime
    )

    assert set(handlers) == set(runner.PHASE_ORDER)
    assert not {
        "parent_model",
        "parent_loader",
        "parent_forward",
        "network_client",
        "hf_loader",
    }.intersection(signature.parameters)
    assert live.load_lfm2_direct_local_model.__module__ == (
        "neural_forge.hf_lfm2_direct_local"
    )


def test_phase3_reuses_binder_owned_orthogonal_charts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = live.Lfm2Exact4xMovingGaugeV5LiveRuntime(
        _request(tmp_path),
        _bindings(),
        phase_overrides=_overrides()[0],
    )
    calls = {
        "orthogonal": 0,
        "base": 0,
        "generalized_fit": 0,
        "generalized_bind": 0,
    }

    def sealed(label: str) -> dict[str, str]:
        return {"report_sha256": live.canonical_sha256({"label": label})}

    def fit_evidence(
        _runtime: object,
        *,
        boundary_id: str,
        **_kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(boundary_id=boundary_id)

    def bind_orthogonal(evidence: SimpleNamespace) -> SimpleNamespace:
        calls["orthogonal"] += 1
        boundary_chart = SimpleNamespace(
            boundary_id=evidence.boundary_id,
            report=sealed(f"orthogonal-anchor:{evidence.boundary_id}"),
        )
        return SimpleNamespace(
            boundary_chart=boundary_chart,
            report=sealed(f"orthogonal-endpoint:{evidence.boundary_id}"),
        )

    def bind_base(evidence: SimpleNamespace) -> SimpleNamespace:
        calls["base"] += 1
        anchor = SimpleNamespace(
            boundary_id=evidence.boundary_id,
            report=sealed(f"orthogonal-anchor:{evidence.boundary_id}"),
        )
        return SimpleNamespace(
            orthogonal_anchor=anchor,
            report=sealed(f"base-endpoint:{evidence.boundary_id}"),
        )

    def fit_generalized(
        evidence: SimpleNamespace,
        *,
        orthogonal_chart: SimpleNamespace,
    ) -> SimpleNamespace:
        calls["generalized_fit"] += 1
        assert orthogonal_chart.boundary_id == evidence.boundary_id
        return SimpleNamespace(
            orthogonal_chart=orthogonal_chart,
            report=sealed(f"generalized-diagnostic:{evidence.boundary_id}"),
        )

    def bind_generalized(
        diagnostic: SimpleNamespace,
        evidence: SimpleNamespace,
    ) -> SimpleNamespace:
        calls["generalized_bind"] += 1
        assert diagnostic.orthogonal_chart.boundary_id == evidence.boundary_id
        return SimpleNamespace(
            report=sealed(f"generalized-endpoint:{evidence.boundary_id}"),
        )

    monkeypatch.setattr(
        live.Lfm2Exact4xMovingGaugeV5LiveRuntime,
        "_fit_evidence",
        fit_evidence,
    )
    monkeypatch.setattr(
        live,
        "bind_lfm2_orthogonal_causal_endpoint_chart",
        bind_orthogonal,
    )
    monkeypatch.setattr(live, "bind_lfm2_base_e0_causal_endpoint_chart", bind_base)
    monkeypatch.setattr(
        live,
        "fit_lfm2_generalized_moving_gauge_chart_diagnostic",
        fit_generalized,
    )
    monkeypatch.setattr(
        live,
        "bind_lfm2_generalized_e0a_causal_endpoint_chart",
        bind_generalized,
    )

    result = runtime._phase_freeze_fit_charts({}, {})
    artifact = result.artifact

    assert calls == {
        "orthogonal": 4,
        "base": 2,
        "generalized_fit": 2,
        "generalized_bind": 2,
    }
    for name in ("pre", "terminal"):
        assert artifact["orthogonal"][name] is artifact["charts"]["E0"][
            name
        ].boundary_chart
        assert artifact["charts"]["E0"][name] is artifact["charts"]["E0R"][name]
        assert artifact["charts"]["E0"][name] is artifact["charts"]["E0A"][name]
    for name in ("post_operator", "post_layer"):
        assert artifact["orthogonal"][name] is artifact["charts"]["E0"][
            name
        ].orthogonal_anchor
        assert (
            artifact["generalized"][name].orthogonal_chart
            is artifact["orthogonal"][name]
        )
    assert result.evidence["chart_bundle_sha256s"] == runtime.chart_bundle_sha256s
    assert len(set(runtime.chart_bundle_sha256s.values())) == 3


def test_phase0_exact_lineage_gate_rejects_wrong_self_sealed_request(
    tmp_path: Path,
) -> None:
    authentic_request = _request(tmp_path)
    actual_lineage = dict(authentic_request.lineage_sha256s)
    wrong_lineage = dict(actual_lineage)
    wrong_lineage["tokenizer_bridge"] = live.canonical_sha256(
        {"wrong-but-self-sealed": "tokenizer-bridge"}
    )
    wrong_request = runner.Lfm2Exact4xMovingGaugeV5Request(
        local_paths=authentic_request.local_paths,
        lineage_sha256s=wrong_lineage,
        counted_child_state_plan=authentic_request.counted_child_state_plan,
        replay_devices=authentic_request.replay_devices,
        recapture_checkpoint_batch_count=(
            authentic_request.recapture_checkpoint_batch_count
        ),
    )

    assert wrong_request.verify_authentication()
    with pytest.raises(
        live.Lfm2Exact4xMovingGaugeV5LiveError,
        match="Self-sealed v5 request differs from authenticated live lineage",
    ):
        live._verify_exact_request_lineage(wrong_request, actual_lineage)


class _TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Module()
        self.conv.out_proj = torch.nn.Linear(3, 2, bias=False)
        self.feed_forward = torch.nn.Module()
        self.feed_forward.w2 = torch.nn.Linear(4, 2, bias=False)


class _TinyChild(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList((_TinyLayer(),))


def test_temporary_operator_failure_restores_exact_child_and_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = live.Lfm2Exact4xMovingGaugeV5LiveRuntime(
        _request(tmp_path),
        _bindings(),
        phase_overrides=_overrides()[0],
    )
    runtime.child = _TinyChild()
    runtime.routing = object()
    parameter = dict(runtime.child.named_parameters())[
        runner.REPLACEMENT_TENSOR_NAMES[0]
    ]
    original = parameter.detach().clone()
    replacement = original + 7.0

    def child_fingerprint(child: object) -> object:
        values = {
            name: fingerprint_lfm2_moving_gauge_tensor(value.detach())
            for name, value in child.named_parameters()
        }
        return SimpleNamespace(
            identity_sha256="f" * 64,
            state_dict_sha256=live.canonical_sha256(values),
        )

    monkeypatch.setattr(live, "fingerprint_lfm2_parent", child_fingerprint)
    monkeypatch.setattr(
        live,
        "fingerprint_lfm2_carrier_runtime_owned_tensor_state",
        lambda _routing: {"state_sha256": "a" * 64},
    )

    def explode() -> None:
        assert torch.is_inference_mode_enabled()
        assert torch.equal(parameter.detach(), replacement)
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        runtime._with_temporary_operator(replacement, explode)

    assert torch.equal(parameter.detach(), original)
    assert runtime._mutation_observations == [
        {
            "child_before": child_fingerprint(runtime.child).state_dict_sha256,
            "child_after": child_fingerprint(runtime.child).state_dict_sha256,
            "route_before": "a" * 64,
            "route_after": "a" * 64,
        }
    ]


def test_temporary_operator_rejects_unrestored_second_child_tensor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = live.Lfm2Exact4xMovingGaugeV5LiveRuntime(
        _request(tmp_path),
        _bindings(),
        phase_overrides=_overrides()[0],
    )
    runtime.child = _TinyChild()
    runtime.routing = object()
    parameters = dict(runtime.child.named_parameters())
    operator = parameters[runner.REPLACEMENT_TENSOR_NAMES[0]]
    other = parameters[runner.REPLACEMENT_TENSOR_NAMES[1]]
    operator_before = operator.detach().clone()
    other_before = other.detach().clone()

    def child_fingerprint(child: object) -> object:
        values = {
            name: fingerprint_lfm2_moving_gauge_tensor(value.detach())
            for name, value in child.named_parameters()
        }
        return SimpleNamespace(
            identity_sha256="f" * 64,
            state_dict_sha256=live.canonical_sha256(values),
        )

    monkeypatch.setattr(live, "fingerprint_lfm2_parent", child_fingerprint)
    monkeypatch.setattr(
        live,
        "fingerprint_lfm2_carrier_runtime_owned_tensor_state",
        lambda _routing: {"state_sha256": "a" * 64},
    )

    def mutate_unmanaged_tensor() -> None:
        with torch.no_grad():
            other.add_(1.0)

    with pytest.raises(
        live.Lfm2Exact4xMovingGaugeV5LiveError,
        match="failed exact child/route restoration",
    ):
        runtime._with_temporary_operator(operator_before + 2.0, mutate_unmanaged_tensor)

    assert torch.equal(operator.detach(), operator_before)
    assert not torch.equal(other.detach(), other_before)
    assert runtime._mutation_observations == []


def test_phase4_prepares_three_writer_systems_once_and_reuses_each_chart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = live.Lfm2Exact4xMovingGaugeV5LiveRuntime(
        _request(tmp_path),
        _bindings(),
        phase_overrides=_overrides()[0],
    )
    runtime.child = _TinyChild()
    row_count = 5
    residual = torch.arange(row_count * 2, dtype=torch.float32).reshape(
        row_count,
        2,
    )
    context = torch.arange(row_count * 3, dtype=torch.float32).reshape(
        row_count,
        3,
    )
    post_swiglu = torch.arange(row_count * 4, dtype=torch.float32).reshape(
        row_count,
        4,
    )
    rows = {
        "pre": SimpleNamespace(child_rows=residual),
        "post_operator": SimpleNamespace(
            parent_rows=residual + 1,
            child_rows=residual + 2,
        ),
        "post_layer": SimpleNamespace(
            parent_rows=residual + 3,
            child_rows=residual + 4,
        ),
        "writers": SimpleNamespace(
            child_combined_writer_output=residual + 5,
            child_operator_output=residual + 6,
            child_mlp_down=residual + 7,
        ),
        "context": context,
        "post_swiglu": post_swiglu,
        "weights": torch.arange(1, row_count + 1, dtype=torch.float32),
    }
    monkeypatch.setattr(runtime, "_baseline_rows", lambda split: rows)
    plans = {
        mode: SimpleNamespace(
            mode=mode,
            report={"result_sha256": live.canonical_sha256({"plan": mode})},
        )
        for mode in runner.CHART_MODES
    }
    monkeypatch.setattr(runtime, "_causal_plan", lambda mode: plans[mode])
    monkeypatch.setattr(
        live,
        "build_lfm2_moving_gauge_causal_state_receipt",
        lambda plan, **_kwargs: {
            "result_sha256": live.canonical_sha256({"receipt": plan.mode})
        },
    )

    def joint_target(plan: object, **_kwargs: object) -> object:
        return SimpleNamespace(
            target_combined_writer_output=torch.full(
                (row_count, 2),
                float(runner.CHART_MODES.index(plan.mode) + 1),
            ),
            report={"result_sha256": live.canonical_sha256({"joint": plan.mode})},
        )

    def sequential_target(plan: object, **_kwargs: object) -> object:
        chart_value = float(runner.CHART_MODES.index(plan.mode) + 10)
        return SimpleNamespace(
            target_operator_writer_output=torch.full(
                (row_count, 2),
                chart_value,
            ),
            target_mlp_writer_output=torch.full(
                (row_count, 2),
                chart_value + 1,
            ),
            report={"result_sha256": live.canonical_sha256({"sequential": plan.mode})},
        )

    monkeypatch.setattr(live, "build_lfm2_joint_writer_target", joint_target)
    monkeypatch.setattr(
        live,
        "build_lfm2_sequential_writer_target",
        sequential_target,
    )

    events: list[tuple[str, int]] = []
    prepared: list[object] = []
    solve_counts: dict[int, int] = {}

    def prepare(
        design: torch.Tensor,
        seed: torch.Tensor,
        weights: torch.Tensor,
    ) -> object:
        problem = SimpleNamespace(
            index=len(prepared),
            design=design.detach().clone(),
            seed=seed.detach().clone(),
            weights=weights.detach().clone(),
        )
        prepared.append(problem)
        events.append(("prepare", int(design.shape[1])))
        return problem

    def solve(problem: object, target: torch.Tensor) -> tuple[torch.Tensor, object]:
        identity = id(problem)
        solve_counts[identity] = solve_counts.get(identity, 0) + 1
        events.append(("solve", int(problem.design.shape[1])))
        solution = problem.seed.detach().clone()
        return solution, {
            "replacement_weight_sha256": (
                fingerprint_lfm2_moving_gauge_tensor(solution)
            ),
            "report_sha256": live.canonical_sha256(
                {
                    "problem": problem.index,
                    "target": float(target.sum().item()),
                }
            ),
        }

    monkeypatch.setattr(
        live,
        "prepare_lfm2_moving_gauge_anchored_writer_problem",
        prepare,
    )
    monkeypatch.setattr(
        live,
        "solve_lfm2_prepared_anchored_writer_replacement",
        solve,
    )

    def reject_legacy_phase4_solver(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("phase 4 rebuilt an unprepared writer system")

    monkeypatch.setattr(
        live,
        "solve_lfm2_moving_gauge_anchored_writer_replacement",
        reject_legacy_phase4_solver,
    )

    result = runtime._phase_build_baseline_target_arms({}, {})

    assert [event[0] for event in events[:3]] == ["prepare"] * 3
    assert [width for _kind, width in events[:3]] == [7, 3, 4]
    assert len(prepared) == 3
    assert len(events) == 3 + 3 * len(runner.CHART_MODES)
    assert solve_counts == {
        id(problem): len(runner.CHART_MODES) for problem in prepared
    }
    assert set(result.artifact) == set(runner.CHART_MODES)


def test_phase6_uses_fresh_conditioned_post_swiglu_not_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = live.Lfm2Exact4xMovingGaugeV5LiveRuntime(
        _request(tmp_path),
        _bindings(),
        phase_overrides=_overrides()[0],
    )
    runtime.child = _TinyChild()
    row_count = 3
    baseline_post_swiglu = torch.full((row_count, 4), -99.0)
    fresh_post_swiglu = torch.arange(row_count * 4, dtype=torch.float32).reshape(
        row_count,
        4,
    )
    residual = torch.arange(row_count * 2, dtype=torch.float32).reshape(
        row_count,
        2,
    )
    runtime.baseline_fit_rows = {
        "pre": SimpleNamespace(child_rows=residual),
        "post_operator": SimpleNamespace(
            parent_rows=residual,
            child_rows=residual + 1,
        ),
        "post_layer": SimpleNamespace(
            parent_rows=residual,
            child_rows=residual + 2,
        ),
        "writers": SimpleNamespace(
            child_operator_output=residual + 3,
            child_mlp_down=residual + 4,
        ),
        "post_swiglu": baseline_post_swiglu,
        "weights": torch.ones(row_count),
    }
    operator = dict(runtime.child.named_parameters())[
        runner.REPLACEMENT_TENSOR_NAMES[0]
    ].detach().clone()
    operator_report = {
        "replacement_weight_sha256": fingerprint_lfm2_moving_gauge_tensor(operator),
        "report_sha256": "b" * 64,
    }
    runtime.baseline_targets = {
        mode: {
            "plan": SimpleNamespace(report={"result_sha256": f"{i + 1:064x}"}),
            "sequential_no_recapture": {
                "operator_solution": operator,
                "operator_report": operator_report,
                "operator_source_receipt": {"result_sha256": "c" * 64},
            },
        }
        for i, mode in enumerate(runner.CHART_MODES)
    }
    runtime.conditioned_recaptures = {
        mode: {
            "receipt_sha256": f"{i + 20:064x}",
            "rows": {
                "fit": {
                    "residual.post_operator": residual + 10,
                    "residual.post_layer": residual + 11,
                    "mlp.down": residual + 12,
                    "mlp.post_swiglu": fresh_post_swiglu + i,
                }
            },
        }
        for i, mode in enumerate(runner.CHART_MODES)
    }
    designs: list[torch.Tensor] = []

    monkeypatch.setattr(
        live,
        "build_lfm2_moving_gauge_causal_state_receipt",
        lambda *_args, **_kwargs: {"result_sha256": "d" * 64},
    )

    def target_builder(*_args: object, **kwargs: object) -> object:
        assert torch.equal(kwargs["fresh_child_post_operator"], residual + 10)
        assert torch.equal(kwargs["fresh_child_post_layer"], residual + 11)
        assert torch.equal(kwargs["existing_mlp_writer_output"], residual + 12)
        return SimpleNamespace(
            target_mlp_writer_output=torch.ones(row_count, 2),
            report={"result_sha256": "e" * 64},
        )

    monkeypatch.setattr(live, "build_lfm2_sequential_writer_target", target_builder)

    def solver(
        design: torch.Tensor,
        _target: torch.Tensor,
        seed: torch.Tensor,
        _weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        designs.append(design.detach().clone())
        solution = seed.detach().clone()
        return solution, {
            "replacement_weight_sha256": (
                fingerprint_lfm2_moving_gauge_tensor(solution)
            ),
            "report_sha256": "f" * 64,
        }

    monkeypatch.setattr(
        live,
        "solve_lfm2_moving_gauge_anchored_writer_replacement",
        solver,
    )

    result = runtime._phase_freeze_conditioned_targets_and_solutions({}, {})

    assert len(designs) == len(runner.CHART_MODES)
    assert all(
        torch.equal(design, fresh_post_swiglu + index)
        for index, design in enumerate(designs)
    )
    assert all(not torch.equal(design, baseline_post_swiglu) for design in designs)
    assert result.evidence["fresh_post_swiglu_used_for_conditioned_mlp"] is True
    assert result.evidence["baseline_post_swiglu_used_for_conditioned_mlp"] is False


@pytest.mark.parametrize("cold", (False, True), ids=("cache-hit", "cold"))
def test_phase5_uses_exact_conditioned_loader_signature_and_cache_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cold: bool,
) -> None:
    loader_calls: list[dict[str, object]] = []
    session_counts = {"opened": 0, "closed": 0}

    class FakeSessionContext:
        def __enter__(self) -> object:
            session_counts["opened"] += 1
            return SimpleNamespace(
                evidence_sha256=_provenance()["child_selective_session"]
            )

        def __exit__(self, *_args: object) -> None:
            session_counts["closed"] += 1

    class FakeAuthenticatedPhase:
        def __init__(self, chart_mode: str) -> None:
            self.chart_mode = chart_mode
            self.receipt_sha256 = live.canonical_sha256(
                {"conditioned-chart": chart_mode}
            )

        def verify_authentication(self) -> bool:
            return True

        def field_rows(
            self,
            field_name: str,
            *,
            split: str,
            heldout_capability: object | None = None,
        ) -> object:
            assert split == "fit"
            assert heldout_capability is None
            assert field_name in {
                "residual.post_operator",
                "residual.post_layer",
                "mlp.down",
                "mlp.post_swiglu",
            }
            return SimpleNamespace(rows=torch.ones(2, 2))

    def exact_loader(
        cache: live.DynamicFoldPhaseCache,
        request: conditioned.Lfm2ConditionedRecapturePhaseRequest,
        *,
        authority: object,
        checkpoint_root: str | Path,
        range_provider: object | None = None,
        maximum_new_checkpoints: int | None = None,
        events: object | None = None,
    ) -> object:
        assert isinstance(cache, live.DynamicFoldPhaseCache)
        assert isinstance(
            request,
            conditioned.Lfm2ConditionedRecapturePhaseRequest,
        )
        assert authority is runtime.bindings.conditioned_recapture_authority
        assert Path(checkpoint_root) == (
            Path(runtime.request.local_paths["checkpoint_root"])
            / runtime.request.request_sha256
            / "conditioned-child-recapture"
            / request.chart_mode
        )
        assert callable(range_provider)
        assert maximum_new_checkpoints is None
        assert events is None
        loader_calls.append(
            {
                "cache": cache,
                "request": request,
                "authority": authority,
                "checkpoint_root": Path(checkpoint_root),
                "range_provider": range_provider,
            }
        )
        if cold:
            first = range_provider(0, 16)
            second = range_provider(16, runner.EXPECTED_BATCH_COUNT)
            assert first.metadata["forward_counts"]["child"] > 0
            assert second.metadata["forward_counts"]["child"] > 0
        return FakeAuthenticatedPhase(request.chart_mode)

    runtime = live.Lfm2Exact4xMovingGaugeV5LiveRuntime(
        _request(tmp_path),
        _bindings(conditioned_recapture_loader=exact_loader),
        phase_overrides=_overrides()[0],
    )
    runtime.child_cache = SimpleNamespace(
        selective_authenticated_session=FakeSessionContext
    )
    runtime.child_fingerprint = SimpleNamespace(
        identity_sha256="a" * 64,
        state_dict_sha256="d" * 64,
    )
    runtime.boundary = SimpleNamespace(live_child_recapture_sha256="9" * 64)
    operator = torch.ones(2, 3)
    operator_report = {
        "replacement_weight_sha256": (
            fingerprint_lfm2_moving_gauge_tensor(operator)
        ),
        "report_sha256": "8" * 64,
    }
    runtime.baseline_targets = {
        mode: {
            "plan": SimpleNamespace(
                report={"result_sha256": f"{index + 20:064x}"}
            ),
            "sequential_no_recapture": {
                "operator_solution": operator,
                "operator_report": operator_report,
            },
        }
        for index, mode in enumerate(runner.CHART_MODES)
    }
    requests = {
        mode: _typed_conditioned_request(
            runtime,
            chart_mode=mode,
            operator_solution=operator,
            operator_report=operator_report,
        )
        for mode in runner.CHART_MODES
    }

    def request_builder(
        *,
        module: object,
        chart_mode: str,
        operator_solution: torch.Tensor,
        operator_report: object,
    ) -> tuple[object, str]:
        assert module is conditioned
        assert operator_solution is operator
        assert operator_report is runtime.baseline_targets[chart_mode][
            "sequential_no_recapture"
        ]["operator_report"]
        request = requests[chart_mode]
        return request, request.frozen_conditioned_child_state_receipt_sha256

    capture_sessions: list[object] = []

    def capture_range(
        *,
        session: object,
        operator_solution: torch.Tensor,
        batch_start: int,
        batch_stop: int,
    ) -> object:
        assert operator_solution is operator
        assert 0 <= batch_start < batch_stop <= runner.EXPECTED_BATCH_COUNT
        capture_sessions.append(session)
        runtime._mutation_observations.append(
            {
                "child_before": "d" * 64,
                "child_after": "d" * 64,
                "route_before": "e" * 64,
                "route_after": "e" * 64,
            }
        )
        return SimpleNamespace(
            metadata={
                "forward_counts": {"child": batch_stop - batch_start}
            }
        )

    monkeypatch.setattr(
        conditioned,
        "AuthenticatedLfm2ConditionedRecapturePhase",
        FakeAuthenticatedPhase,
    )
    monkeypatch.setattr(runtime, "_conditioned_phase_request", request_builder)
    monkeypatch.setattr(runtime, "_capture_replay_range", capture_range)

    result = runtime._phase_checkpoint_conditioned_child_recapture({}, {})

    assert len(loader_calls) == len(runner.CHART_MODES)
    assert [row["request"] for row in loader_calls] == [
        requests[mode] for mode in runner.CHART_MODES
    ]
    assert all(callable(row["range_provider"]) for row in loader_calls)
    assert runtime.conditioned_requests == requests
    if cold:
        assert result.child_forward_count == (
            len(runner.CHART_MODES) * runner.EXPECTED_BATCH_COUNT
        )
        assert result.temporary_child_mutation_performed is True
        assert result.child_state_restored is True
        assert result.child_state_before_sha256 == "d" * 64
        assert result.child_state_after_sha256 == "d" * 64
        assert result.authenticated_checkpoint_reuse is False
        assert session_counts == {"opened": 1, "closed": 1}
        assert len(capture_sessions) == len(runner.CHART_MODES) * 2
        assert all(value is capture_sessions[0] for value in capture_sessions)
    else:
        assert result.child_forward_count == 0
        assert result.temporary_child_mutation_performed is False
        assert result.child_state_restored is False
        assert result.child_state_before_sha256 is None
        assert result.child_state_after_sha256 is None
        assert result.authenticated_checkpoint_reuse is True
        assert session_counts == {"opened": 0, "closed": 0}
        assert capture_sessions == []


def test_cold_recapture_uses_cache_authenticated_profile_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = live.Lfm2Exact4xMovingGaugeV5LiveRuntime(
        _request(tmp_path),
        _bindings(),
        phase_overrides=_overrides()[0],
    )
    runtime.child = object()
    runtime.v3_request = {"runtime_device": "cpu"}
    released: list[object] = []
    references = tuple(
        SimpleNamespace(
            ordered_tensor_names=("input.input_ids", "ignored.output")
        )
        for _index in range(2)
    )

    class FakeSelectiveSession:
        evidence_sha256 = _provenance()["child_selective_session"]
        batch_references = references

        def load_selected_tensors(
            self,
            ordinal: int,
            names: tuple[str, ...],
        ) -> object:
            assert names == ("input.input_ids",)
            return SimpleNamespace(
                tensors={
                    "input.input_ids": torch.tensor([[ordinal]], dtype=torch.long)
                }
            )

        def release_selected_tensors(self, payload: object) -> None:
            released.append(payload)

    def capture(
        child: object,
        replay_batches: object,
        *,
        selected_layer_indices: tuple[int, ...],
        config: object,
        input_device: torch.device,
        replay_capture_mode: str,
        child_evidence_profile: object,
        replay_devices: tuple[str, ...],
    ) -> object:
        assert child is runtime.child
        assert len(replay_batches) == 2
        assert selected_layer_indices == (0,)
        assert config.decode_trace_mode == "full_prefix_canonical"
        assert input_device == torch.device("cpu")
        assert replay_capture_mode == "prefix_by_prefix_canonical"
        assert child_evidence_profile.profile_role == (
            conditioned.LFM2_CONDITIONED_RECAPTURE_PROFILE_ROLE
        )
        assert child_evidence_profile.profile_role == (
            live.CONDITIONED_RECAPTURE_PROFILE_ROLE
        )
        assert child_evidence_profile.layer_suffixes == (
            (0, tuple(sorted(conditioned.LFM2_CONDITIONED_RECAPTURE_REQUIRED_FIELDS))),
        )
        assert replay_devices == runtime.request.replay_devices
        assert torch.is_inference_mode_enabled()
        return SimpleNamespace(metadata={"forward_counts": {"child": 2}})

    monkeypatch.setattr(live, "capture_lfm2_child_replay", capture)
    monkeypatch.setattr(
        runtime,
        "_with_temporary_operator",
        lambda _replacement, callback: callback(),
    )

    replay = runtime._capture_replay_range(
        session=FakeSelectiveSession(),
        operator_solution=torch.ones(2, 3),
        batch_start=0,
        batch_stop=2,
    )

    assert replay.metadata["forward_counts"]["child"] == 2
    assert len(released) == 2


def test_phase7_issues_frozen_capability_before_any_heldout_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = live.Lfm2Exact4xMovingGaugeV5LiveRuntime(
        _request(tmp_path),
        _bindings(),
        phase_overrides=_overrides()[0],
    )
    events: list[str] = []
    row_count = 3
    residual = torch.arange(1, row_count * 2 + 1, dtype=torch.float32).reshape(
        row_count,
        2,
    )
    context = torch.arange(row_count * 3, dtype=torch.float32).reshape(
        row_count,
        3,
    )
    post_swiglu = torch.arange(row_count * 4, dtype=torch.float32).reshape(
        row_count,
        4,
    )
    heldout_rows = {
        "pre": SimpleNamespace(child_rows=residual),
        "post_operator": SimpleNamespace(
            parent_rows=residual + 1,
            child_rows=residual + 2,
        ),
        "post_layer": SimpleNamespace(
            parent_rows=residual + 3,
            child_rows=residual + 4,
        ),
        "writers": SimpleNamespace(
            child_combined_writer_output=residual + 5,
            child_operator_output=residual + 6,
            child_mlp_down=residual + 7,
        ),
        "context": context,
        "post_swiglu": post_swiglu,
        "weights": torch.ones(row_count, dtype=torch.float32),
    }

    def baseline_rows(split: str) -> object:
        assert split == "heldout"
        events.append("baseline-heldout")
        return heldout_rows

    monkeypatch.setattr(runtime, "_baseline_rows", baseline_rows)
    def encode_heldout(value: torch.Tensor) -> torch.Tensor:
        assert value.dtype == torch.float64
        return value

    identity_chart = SimpleNamespace(
        chart=SimpleNamespace(encode_shaped=encode_heldout)
    )
    runtime.fit_charts = {
        mode: {
            "post_operator": identity_chart,
            "post_layer": identity_chart,
        }
        for mode in runner.CHART_MODES
    }
    operator = torch.arange(6, dtype=torch.float64).reshape(2, 3) / 10
    mlp = torch.arange(8, dtype=torch.float64).reshape(2, 4) / 10
    combined = torch.cat((operator, mlp), dim=1)
    combined_report = {
        "replacement_weight_sha256": (
            fingerprint_lfm2_moving_gauge_tensor(combined)
        ),
        "report_sha256": "1" * 64,
    }
    operator_report = {
        "replacement_weight_sha256": (
            fingerprint_lfm2_moving_gauge_tensor(operator)
        ),
        "report_sha256": "2" * 64,
    }
    mlp_report = {
        "replacement_weight_sha256": fingerprint_lfm2_moving_gauge_tensor(mlp),
        "report_sha256": "3" * 64,
    }
    runtime.baseline_targets = {
        mode: {
            "joint": {
                "operator_solution": operator,
                "mlp_solution": mlp,
                "solve_report": combined_report,
            },
            "sequential_no_recapture": {
                "operator_solution": operator,
                "mlp_solution": mlp,
            },
        }
        for mode in runner.CHART_MODES
    }
    runtime.conditioned_targets = {
        mode: {
            "operator_solution": operator,
            "operator_report": operator_report,
            "mlp_solution": mlp,
            "mlp_report": mlp_report,
        }
        for mode in runner.CHART_MODES
    }

    class CapabilityArtifact:
        def __init__(self, chart_mode: str) -> None:
            self.chart_mode = chart_mode
            self.capability: object | None = None
            self.reads: list[str] = []

        def issue_heldout_access_capability(
            self,
            *,
            frozen_candidate_barrier_sha256: str,
        ) -> object:
            events.append(f"issue:{self.chart_mode}")
            self.capability = SimpleNamespace(
                capability_hmac_sha256=live.canonical_sha256(
                    {
                        "chart_mode": self.chart_mode,
                        "barrier": frozen_candidate_barrier_sha256,
                    }
                )
            )
            return self.capability

        def field_rows(
            self,
            field_name: str,
            *,
            split: str,
            heldout_capability: object | None = None,
        ) -> object:
            if (
                split != "heldout"
                or heldout_capability is None
                or heldout_capability is not self.capability
            ):
                raise PermissionError("HELDOUT capability required")
            events.append(f"read:{self.chart_mode}:{field_name}")
            self.reads.append(field_name)
            values = {
                "residual.post_operator": residual + 8,
                "residual.post_layer": residual + 9,
                "mlp.down": residual + 10,
                "mlp.post_swiglu": post_swiglu + 1,
            }
            return SimpleNamespace(rows=values[field_name])

    artifacts = {
        mode: CapabilityArtifact(mode) for mode in runner.CHART_MODES
    }
    runtime.conditioned_recaptures = artifacts
    runtime.chart_bundle_sha256s = {
        mode: f"{index + 50:064x}"
        for index, mode in enumerate(runner.CHART_MODES)
    }
    with pytest.raises(PermissionError, match="capability required"):
        live._conditioned_rows(
            artifacts["E0"],
            split="heldout",
            field_name="residual.post_operator",
        )
    events.clear()

    result = runtime._phase_compare_frozen_candidates_on_heldout(
        {},
        {
            "freeze_conditioned_targets_and_solutions": SimpleNamespace(
                artifact_sha256="4" * 64
            )
        },
    )

    assert result.heldout_rows_read is True
    assert events[: len(runner.CHART_MODES)] == [
        f"issue:{mode}" for mode in runner.CHART_MODES
    ]
    assert events[len(runner.CHART_MODES)] == "baseline-heldout"
    assert all(len(artifact.reads) == 4 for artifact in artifacts.values())
    barriers = result.evidence["frozen_candidate_barrier_sha256s"]
    assert set(barriers) == set(runner.CHART_MODES)
    assert len(set(barriers.values())) == len(runner.CHART_MODES)
    assert result.evidence["freeze_phase_artifact_sha256"] == "4" * 64


def test_phase8_stages_exactly_six_fp32_two_tensor_packs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged: list[tuple[str, str]] = []
    joint_upstream_receipts: list[tuple[str, str]] = []

    def stage(
        _request: object,
        *,
        chart_mode: str,
        target_arm: str,
        replacements: object,
        upstream_receipt_sha256s: object,
        child_state_before_sha256: str,
        child_state_after_sha256: str,
    ) -> tuple[Path, dict[str, str]]:
        assert set(replacements) == set(runner.REPLACEMENT_TENSOR_NAMES)
        assert all(value.dtype == torch.float32 for value in replacements.values())
        assert set(upstream_receipt_sha256s) == {
            "chart",
            "heldout_comparison",
            "lineage",
            "mlp_solution",
            "operator_solution",
            "target_plan",
        }
        assert child_state_before_sha256 == child_state_after_sha256 == "c" * 64
        if target_arm == "joint":
            operator_receipt = upstream_receipt_sha256s["operator_solution"]
            mlp_receipt = upstream_receipt_sha256s["mlp_solution"]
            assert operator_receipt != mlp_receipt
            assert operator_receipt != "2" * 64
            assert mlp_receipt != "2" * 64
            joint_upstream_receipts.append((operator_receipt, mlp_receipt))
        staged.append((chart_mode, target_arm))
        digest = f"{len(staged) + 30:064x}"
        return tmp_path / digest, {"candidate_pack_sha256": digest}

    runtime = live.Lfm2Exact4xMovingGaugeV5LiveRuntime(
        _request(tmp_path),
        _bindings(stage_candidate_pack=stage),
        phase_overrides=_overrides()[0],
    )
    runtime.child = _TinyChild()
    runtime.routing = object()
    runtime.child_fingerprint = SimpleNamespace(
        identity_sha256="a" * 64,
        state_dict_sha256="c" * 64,
    )
    runtime.route_state = {"state_sha256": "b" * 64}
    monkeypatch.setattr(
        live,
        "fingerprint_lfm2_parent",
        lambda _child: SimpleNamespace(
            identity_sha256="a" * 64,
            state_dict_sha256="c" * 64,
        ),
    )
    monkeypatch.setattr(
        live,
        "fingerprint_lfm2_carrier_runtime_owned_tensor_state",
        lambda _routing: {"state_sha256": "b" * 64},
    )
    runtime.chart_bundle_sha256s = {
        mode: f"{index + 40:064x}"
        for index, mode in enumerate(runner.CHART_MODES)
    }
    operator = torch.ones(2, 3, dtype=torch.bfloat16)
    mlp = torch.ones(2, 4, dtype=torch.bfloat16)
    combined = torch.cat((operator, mlp), dim=1)
    combined_report = {
        "replacement_weight_sha256": (
            fingerprint_lfm2_moving_gauge_tensor(combined)
        ),
        "report_sha256": "2" * 64,
    }
    split_receipts = live._joint_split_solution_receipts(
        operator,
        mlp,
        combined_report,
    )
    runtime.baseline_targets = {
        mode: {
            "joint": {
                "target": SimpleNamespace(report={"result_sha256": "1" * 64}),
                "operator_solution": operator,
                "mlp_solution": mlp,
                "solve_report": combined_report,
                "split_receipt_sha256s": split_receipts,
            }
        }
        for mode in runner.CHART_MODES
    }
    runtime.conditioned_targets = {
        mode: {
            "target": SimpleNamespace(report={"result_sha256": "3" * 64}),
            "operator_solution": operator,
            "mlp_solution": mlp,
            "operator_report": {"report_sha256": "4" * 64},
            "mlp_report": {"report_sha256": "5" * 64},
        }
        for mode in runner.CHART_MODES
    }
    completed = {
        "authenticate_lineage": SimpleNamespace(artifact_sha256="6" * 64),
        "compare_frozen_candidates_on_heldout": SimpleNamespace(
            artifact_sha256="7" * 64
        ),
    }

    result = runtime._phase_stage_atomic_candidate_packs({}, completed)

    assert set(staged) == {
        (mode, arm)
        for mode in runner.CHART_MODES
        for arm in runner.PUBLISHABLE_TARGET_ARMS
    }
    assert len(staged) == 6
    assert len(joint_upstream_receipts) == len(runner.CHART_MODES)
    assert all(arm != "sequential_no_recapture" for _mode, arm in staged)
    assert result.ready_for_atomic_apply is True
    assert result.evidence["candidate_pack_count"] == 6
    assert result.evidence["replacement_dtype"] == "float32"


def test_conditioned_cache_api_absence_fails_closed_without_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def absent(name: str) -> object:
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(live.importlib, "import_module", absent)

    with pytest.raises(
        runner.Lfm2Exact4xMovingGaugeV5RuntimeUnavailableError,
        match="Conditioned-recapture cache API is unavailable",
    ):
        live._resolve_conditioned_recapture_loader()
