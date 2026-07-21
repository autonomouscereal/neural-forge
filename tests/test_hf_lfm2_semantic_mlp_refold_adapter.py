from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_mlp_refold_adapter as adapter  # noqa: E402
import neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter as stack_adapter  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _Config:
    model_type = "lfm2"
    norm_eps = 1e-5

    def to_dict(self):
        return {
            "model_type": "lfm2",
            "hidden_size": 4,
            "intermediate_size": 3,
            "num_hidden_layers": 1,
            "norm_eps": self.norm_eps,
        }


class _Mlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
        self.w3 = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
        self.w2 = torch.nn.Linear(3, 4, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.w1.weight.copy_(
                torch.tensor(
                    [
                        [0.50, -0.25, 0.10, 0.20],
                        [-0.20, 0.40, 0.30, -0.10],
                        [0.15, 0.05, -0.35, 0.45],
                    ],
                    dtype=torch.bfloat16,
                )
            )
            self.w3.weight.copy_(
                torch.tensor(
                    [
                        [0.30, 0.10, -0.20, 0.40],
                        [0.25, -0.30, 0.45, 0.15],
                        [-0.10, 0.35, 0.20, 0.50],
                    ],
                    dtype=torch.bfloat16,
                )
            )
            self.w2.weight.copy_(
                torch.tensor(
                    [
                        [0.30, -0.20, 0.10],
                        [-0.10, 0.25, 0.35],
                        [0.20, 0.15, -0.30],
                        [0.05, -0.40, 0.25],
                    ],
                    dtype=torch.bfloat16,
                )
            )


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feed_forward = _Mlp()


class _Inner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer()])


class _Target(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.model = _Inner()


def _source() -> adapter.Lfm2DirectLocalMlpSourceInventory:
    gate = torch.tensor(
        [
            [0.20, -0.10, 0.30, 0.15],
            [-0.25, 0.35, 0.10, -0.20],
            [0.40, 0.05, -0.15, 0.25],
            [0.10, 0.30, 0.20, -0.35],
            [-0.30, 0.20, 0.45, 0.05],
        ],
        dtype=torch.bfloat16,
    )
    up = torch.tensor(
        [
            [0.10, 0.25, -0.20, 0.35],
            [0.30, -0.15, 0.40, 0.10],
            [-0.20, 0.45, 0.15, -0.10],
            [0.35, 0.05, -0.30, 0.20],
            [0.15, -0.25, 0.30, 0.40],
        ],
        dtype=torch.bfloat16,
    )
    down = torch.tensor(
        [
            [0.20, -0.10, 0.30, 0.15, -0.20],
            [-0.15, 0.25, 0.05, 0.35, 0.10],
            [0.30, 0.20, -0.25, 0.10, 0.15],
            [0.05, -0.30, 0.20, 0.25, 0.40],
        ],
        dtype=torch.bfloat16,
    )
    return adapter.Lfm2DirectLocalMlpSourceInventory(
        layer_index=0,
        parent_model_identity_sha256=_sha("parent"),
        parent_checkpoint_manifest_sha256=_sha("manifest"),
        direct_local_loader_receipt_sha256=_sha("loader"),
        residual_assignment_sha256=_sha("residual-chart"),
        compiler_lineage_sha256=_sha("compiler-lineage"),
        gate_weight=gate,
        up_weight=up,
        down_weight=down,
    )


def _evidence(
    model: _Target,
    *,
    batches: int = 4,
    total_phases: int | None = None,
    terminal: bool = True,
) -> tuple[adapter.Lfm2SemanticMlpEndpointEvidence, ...]:
    source = _source()
    state = fingerprint_lfm2_parent(model).identity_sha256
    mlp = model.model.layers[0].feed_forward
    child_input = torch.tensor([[0.75, -0.50, 0.25, 1.00]], dtype=torch.bfloat16)
    child_pre = torch.tensor([[0.20, -0.10, 0.30, 0.05]], dtype=torch.bfloat16)
    desired_down = mlp.w2.weight.detach().to(torch.float64) + torch.tensor(
        [
            [0.08, -0.02, 0.04],
            [-0.03, 0.05, 0.06],
            [0.04, 0.03, -0.05],
            [0.02, -0.06, 0.05],
        ],
        dtype=torch.float64,
    )
    inputs64 = child_input.to(torch.float64)
    child_post = torch.nn.functional.silu(
        inputs64 @ mlp.w1.weight.detach().to(torch.float64).T
    ) * (inputs64 @ mlp.w3.weight.detach().to(torch.float64).T)
    desired_endpoint = (child_pre.to(torch.float64) + child_post @ desired_down.T).to(
        torch.bfloat16
    )
    parent_input = torch.tensor([[0.40, -0.20, 0.60, 0.10]], dtype=torch.bfloat16)
    parent_gate = (
        parent_input.to(torch.float64) @ source.gate_weight.to(torch.float64).T
    ).to(torch.bfloat16)
    parent_up = (
        parent_input.to(torch.float64) @ source.up_weight.to(torch.float64).T
    ).to(torch.bfloat16)
    parent_post = (
        torch.nn.functional.silu(parent_gate.to(torch.float64))
        * parent_up.to(torch.float64)
    ).to(torch.bfloat16)
    if total_phases is None:
        per_batch = [1] * batches
    else:
        quotient, remainder = divmod(total_phases, batches)
        per_batch = [quotient + int(index < remainder) for index in range(batches)]
    values = []
    phase_global = 0
    for batch_index, phase_count in enumerate(per_batch):
        for phase_index in range(phase_count):
            values.append(
                adapter.Lfm2SemanticMlpEndpointEvidence(
                    batch_index=batch_index,
                    trajectory_sha256=_sha(f"trajectory:{batch_index}"),
                    phase_index=phase_index,
                    phase_sha256=_sha(f"phase:{batch_index}:{phase_index}"),
                    layer_index=0,
                    alignment_sha256=_sha(f"alignment:{batch_index}:{phase_index}"),
                    cache_key=_sha("cache"),
                    cache_manifest_sha256=_sha("cache-manifest"),
                    bridge_artifact_sha256=_sha("bridge"),
                    captured_child_model_state_sha256=state,
                    source_inventory=source,
                    parent_gate=parent_gate,
                    parent_up=parent_up,
                    parent_post_swiglu=parent_post,
                    actual_child_mlp_input=child_input,
                    actual_child_pre_mlp_residual=child_pre,
                    parent_post_mlp_residual_target_space=desired_endpoint,
                    parent_final_norm_output_target_space=(
                        desired_endpoint if terminal else None
                    ),
                )
            )
            phase_global += 1
    assert phase_global == (total_phases if total_phases is not None else batches)
    return tuple(values)


def _prepare(
    model: _Target,
    evidence: tuple[adapter.Lfm2SemanticMlpEndpointEvidence, ...],
    *,
    terminal: bool = False,
) -> adapter.Lfm2PreparedSemanticMlpRefold:
    return adapter.prepare_lfm2_semantic_mlp_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key=_sha("cache"),
        cache_manifest_sha256=_sha("cache-manifest"),
        bridge_artifact_sha256=_sha("bridge"),
        maximum_delta_ratio=1.0,
        rmse_relative_tolerance=1e-4,
        cosine_tolerance=1e-5,
        require_terminal_action_evidence=terminal,
    )


def _material_full_win_case(model: _Target):
    """Build repeated heldouts where W2 helps but changed producers help more."""

    source = _source()
    state_sha = fingerprint_lfm2_parent(model).identity_sha256
    mlp = model.model.layers[0].feed_forward
    desired = {
        "w1.weight": (
            mlp.w1.weight.detach().to(torch.float32)
            + torch.tensor(
                [
                    [0.12, -0.08, 0.06, 0.04],
                    [-0.05, 0.10, 0.03, -0.07],
                    [0.08, 0.02, -0.09, 0.11],
                ]
            )
        ).to(torch.bfloat16),
        "w3.weight": (
            mlp.w3.weight.detach().to(torch.float32)
            + torch.tensor(
                [
                    [-0.07, 0.09, 0.04, -0.05],
                    [0.08, 0.03, -0.06, 0.10],
                    [-0.04, 0.07, 0.12, -0.08],
                ]
            )
        ).to(torch.bfloat16),
        "w2.weight": (
            mlp.w2.weight.detach().to(torch.float32)
            + torch.tensor(
                [
                    [0.03, -0.02, 0.01],
                    [-0.01, 0.02, 0.03],
                    [0.02, 0.01, -0.025],
                    [0.01, -0.03, 0.02],
                ]
            )
        ).to(torch.bfloat16),
    }
    inputs = (
        torch.tensor([[0.75, -0.50, 0.25, 1.00]], dtype=torch.bfloat16),
        torch.tensor([[-0.40, 0.80, 0.50, -0.20]], dtype=torch.bfloat16),
        torch.tensor([[0.30, 0.20, -0.90, 0.70]], dtype=torch.bfloat16),
        torch.tensor([[1.00, 0.40, 0.10, -0.60]], dtype=torch.bfloat16),
    )
    child_pre = torch.tensor([[0.20, -0.10, 0.30, 0.05]], dtype=torch.bfloat16)
    values = []
    for batch_index in range(8):
        child_input = inputs[batch_index % len(inputs)]
        input64 = child_input.to(torch.float64)
        desired_post = torch.nn.functional.silu(
            input64 @ desired["w1.weight"].to(torch.float64).T
        ) * (input64 @ desired["w3.weight"].to(torch.float64).T)
        endpoint = (
            child_pre.to(torch.float64)
            + desired_post @ desired["w2.weight"].to(torch.float64).T
        ).to(torch.bfloat16)
        parent_gate = (input64 @ source.gate_weight.to(torch.float64).T).to(
            torch.bfloat16
        )
        parent_up = (input64 @ source.up_weight.to(torch.float64).T).to(torch.bfloat16)
        parent_post = (
            torch.nn.functional.silu(parent_gate.to(torch.float64))
            * parent_up.to(torch.float64)
        ).to(torch.bfloat16)
        values.append(
            adapter.Lfm2SemanticMlpEndpointEvidence(
                batch_index=batch_index,
                trajectory_sha256=_sha(f"full-win-trajectory:{batch_index}"),
                phase_index=0,
                phase_sha256=_sha(f"full-win-phase:{batch_index}"),
                layer_index=0,
                alignment_sha256=_sha(f"full-win-alignment:{batch_index}"),
                cache_key=_sha("cache"),
                cache_manifest_sha256=_sha("cache-manifest"),
                bridge_artifact_sha256=_sha("bridge"),
                captured_child_model_state_sha256=state_sha,
                source_inventory=source,
                parent_gate=parent_gate,
                parent_up=parent_up,
                parent_post_swiglu=parent_post,
                actual_child_mlp_input=child_input,
                actual_child_pre_mlp_residual=child_pre,
                parent_post_mlp_residual_target_space=endpoint,
            )
        )
    return tuple(values), desired


def _cold_start_parent_endpoint_evidence(
    model: _Target,
) -> tuple[adapter.Lfm2SemanticMlpEndpointEvidence, ...]:
    source = _source()
    state_sha = fingerprint_lfm2_parent(model).identity_sha256
    inputs = torch.tensor(
        [
            [0.75, -0.50, 0.25, 1.00],
            [-0.40, 0.80, 0.50, -0.20],
            [0.30, 0.20, -0.90, 0.70],
            [1.00, 0.40, 0.10, -0.60],
            [-0.70, -0.30, 0.80, 0.20],
            [0.20, 1.00, -0.40, 0.60],
            [0.60, -0.80, -0.10, 0.50],
            [-0.20, 0.30, 0.90, -0.70],
        ],
        dtype=torch.bfloat16,
    )
    child_pre = torch.tensor(
        [[0.20, -0.10, 0.30, 0.05]],
        dtype=torch.bfloat16,
    )
    values = []
    for batch_index in range(4):
        for phase_index, input_row in enumerate(inputs):
            child_input = input_row.unsqueeze(0)
            input64 = child_input.to(torch.float64)
            parent_gate = input64 @ source.gate_weight.to(torch.float64).T
            parent_up = input64 @ source.up_weight.to(torch.float64).T
            parent_post = torch.nn.functional.silu(parent_gate) * parent_up
            parent_endpoint = (
                child_pre.to(torch.float64)
                + parent_post @ source.down_weight.to(torch.float64).T
            )
            values.append(
                adapter.Lfm2SemanticMlpEndpointEvidence(
                    batch_index=batch_index,
                    trajectory_sha256=_sha(f"cold-trajectory:{batch_index}"),
                    phase_index=phase_index,
                    phase_sha256=_sha(f"cold-phase:{batch_index}:{phase_index}"),
                    layer_index=0,
                    alignment_sha256=_sha(
                        f"cold-alignment:{batch_index}:{phase_index}"
                    ),
                    cache_key=_sha("cache"),
                    cache_manifest_sha256=_sha("cache-manifest"),
                    bridge_artifact_sha256=_sha("bridge"),
                    captured_child_model_state_sha256=state_sha,
                    source_inventory=source,
                    parent_gate=parent_gate.to(torch.bfloat16),
                    parent_up=parent_up.to(torch.bfloat16),
                    parent_post_swiglu=parent_post.to(torch.bfloat16),
                    actual_child_mlp_input=child_input,
                    actual_child_pre_mlp_residual=child_pre,
                    parent_post_mlp_residual_target_space=(
                        parent_endpoint.to(torch.bfloat16)
                    ),
                )
            )
    return tuple(values)


def test_typed_source_and_endpoint_seals_reject_inserted_rows() -> None:
    model = _Target()
    source = _source()
    assert source.verify()
    assert source.authentication_body()["loader_mode"] == "direct-local-safetensors"
    valid = _evidence(model)[0]
    assert valid.verify()
    assert valid.authentication_body()["parent_exact_semantic_endpoint"] is True
    with pytest.raises(adapter.Lfm2SemanticMlpRefoldError, match="noninserted"):
        adapter.Lfm2SemanticMlpEndpointEvidence(
            batch_index=valid.batch_index,
            trajectory_sha256=valid.trajectory_sha256,
            phase_index=valid.phase_index,
            phase_sha256=valid.phase_sha256,
            layer_index=valid.layer_index,
            alignment_sha256=valid.alignment_sha256,
            cache_key=valid.cache_key,
            cache_manifest_sha256=valid.cache_manifest_sha256,
            bridge_artifact_sha256=valid.bridge_artifact_sha256,
            captured_child_model_state_sha256=(valid.captured_child_model_state_sha256),
            source_inventory=valid.source_inventory,
            parent_gate=valid.parent_gate,
            parent_up=valid.parent_up,
            parent_post_swiglu=valid.parent_post_swiglu,
            actual_child_mlp_input=valid.actual_child_mlp_input,
            actual_child_pre_mlp_residual=valid.actual_child_pre_mlp_residual,
            parent_post_mlp_residual_target_space=(
                valid.parent_post_mlp_residual_target_space
            ),
            inserted_target_row=True,
        )


def test_prepare_freezes_two_fit_only_candidates_and_prefers_writer() -> None:
    model = _Target()
    before = fingerprint_lfm2_parent(model).identity_sha256
    prepared = _prepare(model, _evidence(model))
    assert prepared.verify()
    assert fingerprint_lfm2_parent(model).identity_sha256 == before
    assert prepared.selected_role == "incumbent-producer-w2"
    assert prepared.writer_candidate.solver_receipt["cluster_solve"] is False
    assert prepared.full_candidate.solver_receipt["cluster_solve"] is True
    assert prepared.full_candidate.solver_receipt["full_source_coverage"] is True
    assert prepared.selection_receipt["fit_partition_only_used_for_solves"] is True
    metrics = prepared.selection_receipt["endpoint_metrics"]
    for role in ("incumbent", "incumbent-producer-w2", "full-w1-w3-w2"):
        for partition in ("fit", "fit_check", "selection", "validation"):
            assert metrics[role][partition]["endpoint_rows"] > 0
            assert "endpoint_rmse" in metrics[role][partition]
            assert "endpoint_maximum_absolute_error" in metrics[role][partition]
            assert "mean_endpoint_cosine" in metrics[role][partition]
    heldout = prepared.selection_receipt["heldout_nonregression"][
        "incumbent-producer-w2"
    ]
    assert heldout == {
        "fit_check": True,
        "selection": True,
        "validation": True,
    }


def test_exact_zero_incumbent_uses_unclipped_authenticated_structural_escape() -> None:
    model = _Target()
    mlp = model.model.layers[0].feed_forward
    with torch.no_grad():
        for parameter in mlp.parameters():
            parameter.zero_()
    before = fingerprint_lfm2_parent(model).identity_sha256
    prepared = _prepare(model, _cold_start_parent_endpoint_evidence(model))

    assert prepared.verify()
    assert fingerprint_lfm2_parent(model).identity_sha256 == before
    assert prepared.selected_role == "full-w1-w3-w2"
    assert prepared.selection_receipt["full_changed_roles"] == [
        "w1.weight",
        "w3.weight",
        "w2.weight",
    ]
    assert all(
        int(torch.count_nonzero(value).item()) > 0
        for value in (
            prepared.full_candidate.gate_weight,
            prepared.full_candidate.up_weight,
            prepared.full_candidate.down_weight,
        )
    )
    cold = prepared.full_candidate.solver_receipt["cold_start_escape"]
    assert cold == {
        "incumbent_w1_w3_w2_exactly_zero": True,
        "requested_maximum_delta_ratio": 1.0,
        "effective_full_structural_maximum_delta_ratio": None,
        "relative_incumbent_trust_cap_applied": False,
        "authenticated_full_source_structural_solve_used": True,
        "random_initialization_used": False,
        "gradient_training_used": False,
    }
    assert prepared.selection_receipt["cold_start_escape"] == cold
    planting = prepared.full_candidate.solver_receipt["source_derived_planting"]
    assert planting == {
        "source_inventory_sha256": _source().inventory_sha256,
        "compiler_lineage_sha256": _sha("compiler-lineage"),
        "residual_assignment_sha256": _sha("residual-chart"),
        "full_source_coverage": True,
        "planted_roles": ["w1.weight", "w3.weight", "w2.weight"],
        "all_dense_w1_w3_w2_roles_planted": True,
        "fit_partition_only_used_for_solve": True,
    }
    assert prepared.selection_receipt["source_derived_planting"] == planting
    corrections = prepared.full_candidate.solver_receipt["prior_correction_reports"]
    assert {report["role"] for report in corrections} == {
        "mlp_gate",
        "mlp_up",
        "mlp_down",
    }
    assert all(report["prior_norm"] == 0.0 for report in corrections)
    assert all(report["update_scale"] == 1.0 for report in corrections)
    assert set(prepared.full_candidate.solver_receipt["operation_audit"].values()) == {
        0
    }

    result = adapter.install_lfm2_prepared_semantic_mlp_postfold(
        target_model=model,
        prepared=prepared,
    )
    assert result.receipt["selected_role"] == "full-w1-w3-w2"
    assert result.receipt["cold_start_escape"] == cold
    assert result.receipt["source_derived_planting"] == planting
    assert all(
        int(torch.count_nonzero(parameter).item()) > 0 for parameter in mlp.parameters()
    )


def test_blank_multi_weight_plant_matches_canonical_layer_stack_diff() -> None:
    model = _Target()
    model.untouched = torch.nn.Parameter(
        torch.tensor([1.25], dtype=torch.bfloat16),
    )
    mlp = model.model.layers[0].feed_forward
    with torch.no_grad():
        for parameter in mlp.parameters():
            parameter.zero_()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    evidence = _cold_start_parent_endpoint_evidence(model)
    before_hashes = stack_adapter._state_tensor_hashes(model)
    before_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    identities = {
        name: id(parameter) for name, parameter in model.named_parameters()
    }
    pointers = {
        name: parameter.data_ptr()
        for name, parameter in model.named_parameters()
    }
    gradient_ids = {
        name: id(parameter.grad)
        for name, parameter in model.named_parameters()
    }
    gradient_state = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }

    receipt, terminal_evidence = stack_adapter._install_mlp(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache=SimpleNamespace(
            cache_key=_sha("cache"),
            manifest={"manifest_sha256": _sha("cache-manifest")},
        ),
        bridge=SimpleNamespace(artifact_sha256=_sha("bridge")),
        installer=adapter.compile_install_lfm2_semantic_mlp_postfold,
        final_layer=False,
        parameter_count=stack_adapter._parameter_count(model),
    )

    live_changed = stack_adapter._changed_state_keys(
        before_hashes,
        stack_adapter._state_tensor_hashes(model),
    )
    prefix = "model.layers.0.feed_forward."
    assert receipt["installed"] is True
    assert receipt["selected_role"] == "full-w1-w3-w2"
    assert receipt["changed_parameter_roles"] == [
        name.removeprefix(prefix) for name in live_changed
    ]
    assert receipt["changed_parameter_roles"] == [
        "w1.weight",
        "w2.weight",
        "w3.weight",
    ]
    assert live_changed == tuple(sorted(live_changed))
    assert terminal_evidence == {}
    assert torch.equal(model.untouched.detach().cpu(), before_state["untouched"])
    assert all(
        torch.equal(value.detach().cpu(), before_state[name])
        for name, value in model.state_dict().items()
        if name not in live_changed
    )
    assert {
        name: id(parameter) for name, parameter in model.named_parameters()
    } == identities
    assert {
        name: parameter.data_ptr()
        for name, parameter in model.named_parameters()
    } == pointers
    assert {
        name: id(parameter.grad)
        for name, parameter in model.named_parameters()
    } == gradient_ids
    assert all(
        torch.equal(parameter.grad.detach().cpu(), gradient_state[name])
        for name, parameter in model.named_parameters()
    )
    assert all(
        int(torch.count_nonzero(parameter).item()) > 0
        for parameter in mlp.parameters()
    )


def test_exact_zero_incumbent_fails_closed_when_no_full_plant_passes() -> None:
    model = _Target()
    with torch.no_grad():
        for parameter in model.model.layers[0].feed_forward.parameters():
            parameter.zero_()

    with pytest.raises(
        adapter.Lfm2SemanticMlpRefoldError,
        match="no fully planted nonzero structural W1/W3/W2 candidate",
    ):
        _prepare(model, _evidence(model))


def test_nonzero_incumbent_preserves_requested_relative_trust_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Target()
    observed_caps = []
    real_compile = adapter.compile_lfm2_structural_residual_mlp_fold

    def recording_compile(**kwargs):
        observed_caps.append(kwargs["max_prior_delta_ratio"])
        return real_compile(**kwargs)

    monkeypatch.setattr(
        adapter,
        "compile_lfm2_structural_residual_mlp_fold",
        recording_compile,
    )
    prepared = _prepare(model, _evidence(model))

    assert observed_caps == [1.0]
    assert "cold_start_escape" not in prepared.full_candidate.solver_receipt
    assert "cold_start_escape" not in prepared.selection_receipt
    assert prepared.selected_role == "incumbent-producer-w2"


def test_materially_better_full_candidate_wins_after_frozen_selection(
    monkeypatch,
) -> None:
    model = _Target()
    evidence, desired = _material_full_win_case(model)

    def exact_full_candidate(*_args, **_kwargs):
        return desired, {
            "solver": "test-authenticated-exact-full-candidate",
            "cluster_solve": True,
            "operation_audit": dict(adapter._ZERO_OPERATIONS),
        }

    monkeypatch.setattr(adapter, "_full_candidate_state", exact_full_candidate)
    prepared = _prepare(model, evidence)
    gates = prepared.selection_receipt
    assert gates["heldout_nonregression"]["incumbent-producer-w2"] == {
        "fit_check": True,
        "selection": True,
        "validation": True,
    }
    assert gates["writer_selection_nonregressing_versus_full"] is False
    assert prepared.selected_role == "full-w1-w3-w2"
    assert (
        prepared.full_candidate.metrics["selection"].endpoint_maximum_absolute_error
        < prepared.writer_candidate.metrics["selection"].endpoint_maximum_absolute_error
    )
    assert gates["candidate_frozen_before_validation"] is True
    assert gates["validation_triggered_refit"] is False


def test_transactional_install_preserves_topology_identity_and_gradients() -> None:
    model = _Target()
    mlp = model.model.layers[0].feed_forward
    for parameter in mlp.parameters():
        parameter.grad = torch.ones_like(parameter)
    evidence = _evidence(model)
    prepared = _prepare(model, evidence)
    count = sum(parameter.numel() for parameter in model.parameters())
    identities = tuple(id(parameter) for parameter in model.parameters())
    pointers = tuple(parameter.data_ptr() for parameter in model.parameters())
    gradient_ids = tuple(id(parameter.grad) for parameter in model.parameters())
    before = fingerprint_lfm2_parent(model).identity_sha256
    result = adapter.install_lfm2_prepared_semantic_mlp_postfold(
        target_model=model,
        prepared=prepared,
    )
    assert result.receipt["installed"] is True
    assert result.receipt["selected_role"] == "incumbent-producer-w2"
    assert result.receipt["changed_parameter_roles"] == ["w2.weight"]
    assert result.receipt["model_state_before_sha256"] == before
    assert result.receipt["model_state_after_sha256"] != before
    assert result.receipt["parameter_count_changed"] is False
    assert sum(parameter.numel() for parameter in model.parameters()) == count
    assert tuple(id(parameter) for parameter in model.parameters()) == identities
    assert tuple(parameter.data_ptr() for parameter in model.parameters()) == pointers
    assert tuple(id(parameter.grad) for parameter in model.parameters()) == gradient_ids
    assert all(value == 0 for value in result.receipt["operation_audit"].values())


def test_regressing_candidates_fall_back_to_authenticated_incumbent_noop(
    monkeypatch,
) -> None:
    model = _Target()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    evidence = _evidence(model)
    before = fingerprint_lfm2_parent(model).identity_sha256
    identities = tuple(id(parameter) for parameter in model.parameters())
    pointers = tuple(parameter.data_ptr() for parameter in model.parameters())
    gradient_ids = tuple(id(parameter.grad) for parameter in model.parameters())

    def regressing_writer(_fit, incumbent, **_policy):
        state = {name: value.detach().cpu().clone() for name, value in incumbent.items()}
        state["w2.weight"] = torch.full_like(state["w2.weight"], 64.0)
        return state, {
            "solver": "test-regressing-writer",
            "operation_audit": dict(adapter._ZERO_OPERATIONS),
        }

    def regressing_full(_fit, _inventory, incumbent, **_policy):
        state = {
            name: torch.full_like(value, -64.0)
            for name, value in incumbent.items()
        }
        return state, {
            "solver": "test-regressing-full",
            "operation_audit": dict(adapter._ZERO_OPERATIONS),
        }

    monkeypatch.setattr(adapter, "_solve_writer", regressing_writer)
    monkeypatch.setattr(adapter, "_full_candidate_state", regressing_full)
    prepared = _prepare(model, evidence)
    assert prepared.selected_role == "incumbent"
    assert prepared.selection_receipt["incumbent_noop_fallback_used"] is True
    assert prepared.selection_receipt["heldout_nonregression"][
        "incumbent-producer-w2"
    ]["selection"] is False

    result = adapter.install_lfm2_prepared_semantic_mlp_postfold(
        target_model=model,
        prepared=prepared,
    )
    receipt = dict(result.receipt)
    seal = receipt.pop("result_sha256")
    assert adapter.canonical_sha256(receipt) == seal
    assert receipt["status"] == "heldout-selected-incumbent-noop"
    assert receipt["incumbent_selected_by_heldout_gates"] is True
    assert receipt["fit_check_nonregression_passed"] is True
    assert receipt["selection_nonregression_passed"] is True
    assert receipt["validation_nonregression_passed"] is True
    assert stack_adapter._authenticated_heldout_incumbent_noop(
        receipt,
        before_sha256=receipt["model_state_before_sha256"],
        after_sha256=receipt["model_state_after_sha256"],
        changed_state_keys=receipt["changed_parameter_roles"],
    )
    assert receipt["installed"] is False
    assert receipt["install_count"] == 0
    assert receipt["changed_parameter_roles"] == []
    assert receipt["exact_noop_state_proven"] is True
    assert receipt["parameter_identities_preserved"] is True
    assert receipt["parameter_storage_preserved"] is True
    assert receipt["gradient_state_preserved"] is True
    assert receipt["source_inventory_sha256"] == evidence[0].source_inventory.inventory_sha256
    assert receipt["compiler_lineage_sha256"] == evidence[
        0
    ].source_inventory.compiler_lineage_sha256
    assert receipt["cache_key"] == _sha("cache")
    assert receipt["cache_manifest_sha256"] == _sha("cache-manifest")
    assert receipt["bridge_artifact_sha256"] == _sha("bridge")
    assert receipt["evidence_sha256"] == [item.evidence_sha256 for item in evidence]
    assert all(value == 0 for value in receipt["operation_audit"].values())
    assert fingerprint_lfm2_parent(model).identity_sha256 == before
    assert tuple(id(parameter) for parameter in model.parameters()) == identities
    assert tuple(parameter.data_ptr() for parameter in model.parameters()) == pointers
    assert tuple(id(parameter.grad) for parameter in model.parameters()) == gradient_ids
    assert not result.final_layer_mlp_evidence


def test_identical_mlp_candidates_select_incumbent_and_perform_no_copy(
    monkeypatch,
) -> None:
    model = _Target()
    evidence = _evidence(model)
    before = fingerprint_lfm2_parent(model).identity_sha256

    def identical_writer(_fit, incumbent, **_policy):
        return {
            name: value.detach().cpu().clone() for name, value in incumbent.items()
        }, {
            "solver": "test-identical-writer",
            "operation_audit": dict(adapter._ZERO_OPERATIONS),
        }

    def identical_full(_fit, _inventory, incumbent, **_policy):
        return {
            name: value.detach().cpu().clone() for name, value in incumbent.items()
        }, {
            "solver": "test-identical-full",
            "operation_audit": dict(adapter._ZERO_OPERATIONS),
        }

    monkeypatch.setattr(adapter, "_solve_writer", identical_writer)
    monkeypatch.setattr(adapter, "_full_candidate_state", identical_full)
    prepared = _prepare(model, evidence)
    assert prepared.verify()
    assert prepared.selected_role == "incumbent"
    assert prepared.selection_receipt["writer_changed_roles"] == []
    assert prepared.selection_receipt["full_changed_roles"] == []
    assert prepared.selection_receipt["heldout_selection_improvement"] == {
        "incumbent-producer-w2": False,
        "full-w1-w3-w2": False,
    }

    result = adapter.install_lfm2_prepared_semantic_mlp_postfold(
        target_model=model,
        prepared=prepared,
    )
    assert result.receipt["installed"] is False
    assert result.receipt["install_count"] == 0
    assert result.receipt["selected_role"] == "incumbent"
    assert result.receipt["selected_install_roles_identical_to_live_state"] is True
    assert result.receipt["incumbent_noop_fallback_used"] is True
    assert fingerprint_lfm2_parent(model).identity_sha256 == before


def test_final_w2_deferral_exports_four_banks_and_performs_no_install() -> None:
    model = _Target()
    evidence = _evidence(model, terminal=True)
    before = fingerprint_lfm2_parent(model).identity_sha256
    prepared = _prepare(model, evidence, terminal=True)
    result = adapter.install_lfm2_prepared_semantic_mlp_postfold(
        target_model=model,
        prepared=prepared,
        defer_down_weight=True,
    )
    assert result.receipt["status"] == "prepared-final-w2-deferred"
    assert result.receipt["installed"] is False
    assert result.receipt["install_count"] == 0
    assert result.receipt["final_layer_mlp_w2_deferred_to_fused_terminal"] is True
    assert fingerprint_lfm2_parent(model).identity_sha256 == before
    assert result.receipt["cache_key"] == _sha("cache")
    assert result.receipt["cache_manifest_sha256"] == _sha("cache-manifest")
    assert result.receipt["evidence_sha256"] == [
        item.evidence_sha256 for item in evidence
    ]
    assert set(result.final_layer_mlp_evidence) == {
        "fit",
        "fit_check",
        "selection",
        "validation",
    }
    for partition, bank in result.final_layer_mlp_evidence.items():
        assert bank.split == partition
        assert bank.verify_authentication()
        assert (
            result.receipt["terminal_action_evidence_sha256"][partition]
            == bank.authentication_sha256
        )


def test_stale_replay_fails_closed_and_copy_failure_rolls_back_exactly(
    monkeypatch,
) -> None:
    stale_model = _Target()
    stale_prepared = _prepare(stale_model, _evidence(stale_model))
    with torch.no_grad():
        stale_model.model.layers[0].feed_forward.w1.weight.add_(0.01)
    with pytest.raises(adapter.Lfm2SemanticMlpRefoldError, match="stale"):
        adapter.install_lfm2_prepared_semantic_mlp_postfold(
            target_model=stale_model,
            prepared=stale_prepared,
        )

    model = _Target()
    prepared = _prepare(model, _evidence(model))
    before = fingerprint_lfm2_parent(model).identity_sha256
    target_w2 = model.model.layers[0].feed_forward.w2.weight
    original = adapter._copy_candidate

    def copy_then_fail(parameter, value):
        original(parameter, value)
        if parameter is target_w2:
            raise RuntimeError("injected after W2 copy")

    monkeypatch.setattr(adapter, "_copy_candidate", copy_then_fail)
    with pytest.raises(RuntimeError, match="injected"):
        adapter.install_lfm2_prepared_semantic_mlp_postfold(
            target_model=model,
            prepared=prepared,
        )
    assert fingerprint_lfm2_parent(model).identity_sha256 == before


def test_repeated_complete_cache_scale_accepts_all_365_phases() -> None:
    model = _Target()
    evidence = _evidence(model, batches=16, total_phases=365)
    prepared = _prepare(model, evidence)
    assert len(prepared.evidence_sha256) == 365
    assert (
        sum(
            prepared.incumbent.metrics[name].endpoint_rows
            for name in ("fit", "fit_check", "selection", "validation")
        )
        == 365
    )
    assert set(prepared.partition_batch_indices) == {
        "fit",
        "fit_check",
        "selection",
        "validation",
    }
    assert prepared.selection_receipt["candidate_frozen_before_validation"] is True


def test_module_ast_contains_no_execution_or_training_calls() -> None:
    path = Path(adapter.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_attributes = {
        "forward",
        "generate",
        "backward",
        "step",
        "zero_grad",
        "grad",
        "from_pretrained",
    }
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called.isdisjoint(forbidden_attributes)
