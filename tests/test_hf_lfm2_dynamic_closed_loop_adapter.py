from __future__ import annotations

import gc
import hashlib
import json
import math
import weakref
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_dynamic_closed_loop_adapter as adapter_module  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_closed_loop_fold import (  # noqa: E402
    RUN_SCHEMA,
    CapturePoint,
    ChildClosedLoopCompiler,
    ClosedLoopAudit,
    ClosedLoopCleanupError,
    ClosedLoopCompilerConfig,
    ClosedLoopFoldError,
    FoldScope,
    TrajectoryRecord,
    deterministic_whole_trajectory_split,
)
from neural_forge.hf_lfm2_closed_loop_restart_capsule import (  # noqa: E402
    Lfm2ClosedLoopRestartCapsuleWriter,
    discover_lfm2_closed_loop_restart_capsules,
    lfm2_restart_capsule_adapter_identity,
    lfm2_restart_capsule_split_sha256,
    load_lfm2_closed_loop_restart_capsule,
    portable_child_state_sha256,
    portable_child_topology_sha256,
)
from neural_forge.hf_closed_form_fit import FitMetrics  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceCaptureConfig,
    Lfm2CircuitTraceSelectiveAuthenticatedSession,
    build_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_balanced_circuit_basis import (  # noqa: E402
    build_lfm2_circuit_balanced_basis,
)
from neural_forge.hf_lfm2_dynamic_closed_loop_adapter import (  # noqa: E402
    LFM2_BODY_RECLOSURE_MOVING_GAUGE_V3,
    Lfm2DynamicClosedLoopAdapter,
    Lfm2DynamicClosedLoopAdapterConfig,
    Lfm2DynamicClosedLoopAdapterError,
)
from neural_forge.hf_lfm2_weight_complete_basis import (  # noqa: E402
    build_lfm2_weight_complete_basis,
)
from neural_forge.hf_lfm2_macro_cell_attractor_gate import (  # noqa: E402
    Lfm2MacroCellAttractorGate,
)


def _model(*, hidden: int, intermediate: int, heads: int, seed: int):
    config = transformers.Lfm2Config(
        vocab_size=41,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=2,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return transformers.Lfm2ForCausalLM(config)


def _batch(prompt, forced):
    input_ids = torch.tensor([list(prompt)], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "forced_decode_ids": torch.tensor([list(forced)], dtype=torch.long),
    }


class _DenseMap:
    def __init__(self, source_width: int, target_width: int) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1901)
            raw = torch.randn(
                source_width, target_width, dtype=torch.float64
            )
        self.basis = torch.linalg.qr(raw, mode="reduced").Q.contiguous()
        self.source_width = source_width
        self.target_width = target_width
        self.map_sha256 = hashlib.sha256(
            self.basis.reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest()

    def encode(self, source_rows):
        return (
            source_rows.detach().to(device="cpu", dtype=torch.float64)
            @ self.basis
        ).contiguous()

    def fold_consumer_weight(self, source_weight):
        return self.encode(source_weight)

    def materialize_dense(
        self, *, max_elements, dtype=None, device="cpu"
    ):
        assert self.basis.numel() <= max_elements
        return self.basis.to(
            device=device,
            dtype=self.basis.dtype if dtype is None else dtype,
        ).contiguous()


def _fold_norm(source, target, basis):
    value = torch.sqrt(
        basis.square().T
        @ source.weight.detach().to(dtype=torch.float64).square()
    )
    target.weight.copy_(value.to(dtype=target.weight.dtype))


def _group_rows(value, target_rows):
    source_rows = int(value.shape[0])
    assert source_rows % target_rows == 0
    return value.reshape(
        target_rows, source_rows // target_rows, int(value.shape[1])
    ).mean(dim=1)


def _seed_child_from_basis(parent, child, mapping):
    basis = mapping.basis
    with torch.no_grad():
        child.model.embed_tokens.weight.copy_(
            (
                parent.model.embed_tokens.weight.detach().to(torch.float64)
                @ basis
            ).to(child.model.embed_tokens.weight.dtype)
        )
        child.lm_head.weight.copy_(
            (
                parent.lm_head.weight.detach().to(torch.float64) @ basis
            ).to(child.lm_head.weight.dtype)
        )
        for source_layer, target_layer in zip(
            parent.model.layers, child.model.layers
        ):
            _fold_norm(
                source_layer.operator_norm,
                target_layer.operator_norm,
                basis,
            )
            _fold_norm(
                source_layer.ffn_norm,
                target_layer.ffn_norm,
                basis,
            )
            source_mlp = source_layer.feed_forward
            target_mlp = target_layer.feed_forward
            for source_weight, target_weight in (
                (source_mlp.w1.weight, target_mlp.w1.weight),
                (source_mlp.w3.weight, target_mlp.w3.weight),
            ):
                folded = source_weight.detach().to(torch.float64) @ basis
                folded = _group_rows(
                    folded, int(target_weight.shape[0])
                )
                target_weight.copy_(folded.to(target_weight.dtype))
            source_down = (
                basis.T
                @ source_mlp.w2.weight.detach().to(torch.float64)
            )
            grouped_down = source_down.reshape(
                int(source_down.shape[0]),
                int(target_mlp.w2.weight.shape[1]),
                int(source_down.shape[1])
                // int(target_mlp.w2.weight.shape[1]),
            ).mean(dim=2)
            target_mlp.w2.weight.copy_(
                grouped_down.to(target_mlp.w2.weight.dtype)
            )
            if hasattr(source_layer, "self_attn"):
                source = source_layer.self_attn
                target = target_layer.self_attn
                source_q = source.q_proj.weight.detach().to(torch.float64)
                folded_q = _group_rows(
                    source_q @ basis, int(target.q_proj.weight.shape[0])
                )
                target.q_proj.weight.copy_(
                    folded_q.to(target.q_proj.weight.dtype)
                )
                for source_weight, target_weight in (
                    (source.k_proj.weight, target.k_proj.weight),
                    (source.v_proj.weight, target.v_proj.weight),
                ):
                    folded = source_weight.detach().to(torch.float64) @ basis
                    target_weight.copy_(
                        _group_rows(
                            folded,
                            int(target_weight.shape[0]),
                        ).to(target_weight.dtype)
                    )
                source_out = source.out_proj.weight.detach().to(
                    torch.float64
                )
                head_dim = int(source.head_dim)
                source_heads = int(source_q.shape[0]) // head_dim
                context_fold = source_out.reshape(
                    int(source_out.shape[0]), source_heads, head_dim
                ).mean(dim=1)
                target.out_proj.weight.copy_(
                    (basis.T @ context_fold).to(
                        target.out_proj.weight.dtype
                    )
                )
                if hasattr(source, "q_layernorm"):
                    target.q_layernorm.weight.copy_(
                        source.q_layernorm.weight
                    )
                if hasattr(source, "k_layernorm"):
                    target.k_layernorm.weight.copy_(
                        source.k_layernorm.weight
                    )
            else:
                source = source_layer.conv
                target = target_layer.conv
                chunks = source.in_proj.weight.detach().to(
                    torch.float64
                ).chunk(3, dim=0)
                target.in_proj.weight.copy_(
                    torch.cat(
                        [basis.T @ chunk @ basis for chunk in chunks],
                        dim=0,
                    ).to(target.in_proj.weight.dtype)
                )
                channel_weights = basis.square()
                channel_weights = channel_weights / channel_weights.sum(
                    dim=0, keepdim=True
                )
                target.conv.weight.copy_(
                    torch.einsum(
                        "st,scl->tcl",
                        channel_weights,
                        source.conv.weight.detach().to(torch.float64),
                    ).to(target.conv.weight.dtype)
                )
                target.out_proj.weight.copy_(
                    (
                        basis.T
                        @ source.out_proj.weight.detach().to(torch.float64)
                        @ basis
                    ).to(target.out_proj.weight.dtype)
                )
        _fold_norm(
            parent.model.embedding_norm,
            child.model.embedding_norm,
            basis,
        )


@pytest.fixture(scope="module")
def sealed_parent(tmp_path_factory):
    parent = _model(hidden=8, intermediate=12, heads=2, seed=1801)
    batches = [
        _batch((1, 2, 3, 4), (5,)),
        _batch((6, 7, 8, 9), (10,)),
        _batch((11, 12, 13, 14), (15,)),
        _batch((16, 17, 18, 19), (20,)),
    ]
    config = Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=2,
        max_decode_steps=2,
        max_capture_bytes=64 * 1024 * 1024,
        decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
    )
    cache = build_lfm2_circuit_trace_cache(
        tmp_path_factory.mktemp("lfm2-adapter-parent"),
        parent,
        batches,
        config=config,
        input_device="cpu",
    )
    return parent, cache, config


def _adapter(
    sealed_parent,
    *,
    minimum_improvement=-1_000_000.0,
    minimum_output_improvement=-1_000_000.0,
    output_validation_tolerance=1_000_000.0,
    performance_metrics_diagnostic_only=False,
    replay_capture_mode=CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    attention_consumer_mode="output-only",
    output_interface_mode="fit-transform",
    output_max_prior_delta_ratio=0.25,
    macro_cell_attractor_gate=None,
    attention_score_candidate_gate_enabled=False,
    attention_score_maximum_softmax_fisher_error=None,
    attention_score_minimum_fraction=0.0,
    selective_session=None,
    existing_child=None,
    existing_mapping=None,
    body_reclosure_mode="disabled",
    parent_residual_operators=None,
    progress_sink=None,
    replay_devices=None,
):
    parent, cache, replay_config = sealed_parent
    mapping = _DenseMap(8, 4) if existing_mapping is None else existing_mapping
    child = (
        _model(hidden=4, intermediate=6, heads=1, seed=1802)
        if existing_child is None
        else existing_child
    )
    if existing_child is None:
        _seed_child_from_basis(parent, child, mapping)
    adapter = Lfm2DynamicClosedLoopAdapter(
        source_model=parent,
        child_model=child,
        parent_cache=cache,
        coordinate_map=mapping,
        initial_child_basis_sha256=mapping.map_sha256,
        body_reclosure_mode=body_reclosure_mode,
        parent_residual_operators=parent_residual_operators,
        replay_config=replay_config,
        input_device="cpu",
        config=Lfm2DynamicClosedLoopAdapterConfig(
            ridge=1e-5,
            norm_ridge=1e-5,
            max_prior_delta_ratio=4.0,
            cluster_max_iterations=4,
            cluster_distance_chunk_size=16,
            minimum_layer_relative_rmse_improvement=(
                minimum_improvement
            ),
            minimum_final_norm_relative_rmse_improvement=(
                minimum_improvement
            ),
            minimum_output_kl_improvement=(
                minimum_output_improvement
            ),
            minimum_output_known_sequence_nll_improvement=(
                minimum_output_improvement
            ),
            output_max_prior_delta_ratio=(
                output_max_prior_delta_ratio
            ),
            output_validation_tolerance=(
                output_validation_tolerance
            ),
            performance_metrics_diagnostic_only=(
                performance_metrics_diagnostic_only
            ),
            replay_capture_mode=replay_capture_mode,
            attention_consumer_mode=attention_consumer_mode,
            attention_score_candidate_gate_enabled=(
                attention_score_candidate_gate_enabled
            ),
            attention_score_maximum_softmax_fisher_error=(
                attention_score_maximum_softmax_fisher_error
            ),
            attention_score_minimum_top1_key_agreement_fraction=(
                attention_score_minimum_fraction
            ),
            attention_score_minimum_top_k_pair_order_agreement_fraction=(
                attention_score_minimum_fraction
            ),
            attention_score_minimum_winner_margin_sign_agreement_fraction=(
                attention_score_minimum_fraction
            ),
            attention_score_minimum_relative_margin_floor_retention_fraction=(
                attention_score_minimum_fraction
            ),
            output_interface_mode=output_interface_mode,
            replay_devices=replay_devices,
        ),
        macro_cell_attractor_gate=macro_cell_attractor_gate,
        selective_session=selective_session,
        progress_sink=progress_sink,
    )
    return adapter, child, mapping


class _ScriptedAttractorSnapshot:
    def __init__(self, phase):
        self.phase = phase
        self.snapshot_sha256 = hashlib.sha256(
            phase.encode("utf-8")
        ).hexdigest()
        self.metrics = {
            "mean_entropy_parent_distance": 0.01,
            "mean_repetition_excess": 0.02,
            "mean_dominant_token_excess": 0.03,
            "mean_eos_position_error": 0.0,
            "collapsed_record_count": 0,
        }


class _ScriptedAttractorDecision:
    def __init__(self, accepted):
        self.accepted = bool(accepted)
        self.failed_checks = (
            () if accepted else ("new_collapsed_records",)
        )
        self.decision_sha256 = hashlib.sha256(
            f"decision:{accepted}".encode("ascii")
        ).hexdigest()

    def as_dict(self):
        return {
            "new_collapsed_record_ids": (
                [] if self.accepted else ["batch-000000-row-000000"]
            ),
            "checks": {
                "new_collapsed_records": {
                    "observed": 0 if self.accepted else 1,
                    "maximum": 0,
                    "passed": self.accepted,
                }
            },
        }


class _ScriptedAttractorGate(Lfm2MacroCellAttractorGate):
    def __init__(self, *, accepted):
        self.accepted = bool(accepted)
        self.phases = []

    def capture(self, child_model, *, phase):
        del child_model
        self.phases.append(phase)
        return _ScriptedAttractorSnapshot(phase)

    def compare(self, incumbent, candidate):
        assert incumbent.phase.startswith("pre-macro-cell-layer-")
        assert candidate.phase.startswith("post-macro-cell-layer-")
        return _ScriptedAttractorDecision(self.accepted)


def test_adapter_rejects_a_child_bound_to_another_basis(
    sealed_parent,
) -> None:
    parent, cache, replay_config = sealed_parent
    mapping = _DenseMap(8, 4)
    child = _model(hidden=4, intermediate=6, heads=1, seed=1803)
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="not bound",
    ):
        Lfm2DynamicClosedLoopAdapter(
            source_model=parent,
            child_model=child,
            parent_cache=cache,
            coordinate_map=mapping,
            initial_child_basis_sha256="0" * 64,
            replay_config=replay_config,
            input_device="cpu",
        )


def test_native_short_conv_reuses_exact_legacy_residual_assignment(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    short_conv_indices = tuple(
        index
        for index, contract in enumerate(adapter._source_contracts)
        if contract.kind == "short_conv"
    )

    assert short_conv_indices
    assert adapter._short_conv_assignment is (
        adapter._short_conv_residual_assignment
    )
    assert any(
        coefficient != 1.0
        for coefficient in adapter._short_conv_residual_assignment.coefficients
    )
    assert all(
        adapter._short_conv_assignments[index]
        is adapter._short_conv_residual_assignment
        for index in short_conv_indices
    )


def test_outer_transaction_restores_exact_state_parameter_and_gradient(
    sealed_parent,
) -> None:
    adapter, child, _mapping = _adapter(sealed_parent)
    parameter = child.model.layers[0].feed_forward.w1.weight
    parameter.grad = torch.arange(
        parameter.numel(), dtype=parameter.dtype
    ).reshape_as(parameter)
    state_before = adapter.child_state_fingerprint()
    topology_before = adapter.child_topology_fingerprint()
    parameter_id = id(parameter)
    gradient_id = id(parameter.grad)
    gradient_before = parameter.grad.detach().clone()
    transaction = adapter.begin_child_transaction(ClosedLoopAudit())
    with torch.no_grad():
        parameter.add_(7.0)
    parameter.grad.add_(3.0)
    receipt = transaction.rollback()
    assert receipt.exact
    assert not receipt.accepted
    assert adapter.child_state_fingerprint() == state_before
    assert adapter.child_topology_fingerprint() == topology_before
    assert id(parameter) == parameter_id
    assert id(parameter.grad) == gradient_id
    assert torch.equal(parameter.grad, gradient_before)


def test_moving_gauge_v3_publishes_complete_causal_plan_in_place(
    sealed_parent,
) -> None:
    parent, _cache, _replay_config = sealed_parent
    basis = build_lfm2_weight_complete_basis(parent, 4)
    progress_events = []
    adapter, child, _mapping = _adapter(
        sealed_parent,
        existing_mapping=basis,
        body_reclosure_mode=LFM2_BODY_RECLOSURE_MOVING_GAUGE_V3,
        parent_residual_operators=basis.inventory.operators,
        output_interface_mode="identity-skip",
        progress_sink=(
            lambda stage, metadata: progress_events.append(
                (stage, dict(metadata))
            )
        ),
    )
    original_internal_recapture = adapter._body_internal_recapture
    original_replay_and_capture = adapter.replay_and_capture_child
    internal_tensor_refs = []
    internal_selections = []
    internal_roles = []
    external_body_selections = []
    prior_tensor_ref = None
    prior_tensor_released = []
    initial_body_tensor_ref = None

    def observed_replay_and_capture(self, scope, point, records, audit):
        nonlocal initial_body_tensor_ref
        capture = original_replay_and_capture(scope, point, records, audit)
        if scope.kind == "body_reclosure":
            external_body_selections.append(
                capture.payload.selected_layer_indices
            )
        if point is CapturePoint.PRE_BODY_RECLOSURE:
            payload = self._capture_payload(capture)
            tensor = next(
                value
                for name, value in payload.paired.batches[0].child.tensors.items()
                if ".layer." in name and value.is_floating_point()
            )
            initial_body_tensor_ref = weakref.ref(tensor)
        return capture

    def observed_internal_recapture(self, records, audit, **kwargs):
        nonlocal prior_tensor_ref
        internal_selections.append(
            tuple(kwargs["selected_layer_indices"])
        )
        internal_roles.append(kwargs["body_internal_role"])
        if not internal_tensor_refs:
            assert initial_body_tensor_ref is not None
            assert initial_body_tensor_ref() is None
        if prior_tensor_ref is not None:
            released = prior_tensor_ref() is None
            prior_tensor_released.append(released)
            assert released
        capture = original_internal_recapture(records, audit, **kwargs)
        payload = self._capture_payload(capture)
        tensor = next(
            value
            for name, value in payload.paired.batches[0].child.tensors.items()
            if ".layer." in name and value.is_floating_point()
        )
        prior_tensor_ref = weakref.ref(tensor)
        internal_tensor_refs.append(prior_tensor_ref)
        return capture

    adapter._body_internal_recapture = MethodType(
        observed_internal_recapture,
        adapter,
    )
    adapter.replay_and_capture_child = MethodType(
        observed_replay_and_capture,
        adapter,
    )
    parameters_before = tuple(
        (
            name,
            id(parameter),
            int(parameter.data_ptr()),
            tuple(parameter.shape),
            parameter.dtype,
        )
        for name, parameter in child.named_parameters(
            remove_duplicate=False
        )
    )
    exact_count_before = sum(
        int(parameter.numel())
        for parameter in {
            id(value): value for value in child.parameters()
        }.values()
    )

    report = ChildClosedLoopCompiler(
        adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.25,
            split_seed="lfm2-moving-gauge-v3-contract-v1",
            fold_body_reclosure=True,
            fold_output_interface=False,
        ),
    ).run()

    body = report["body_reclosure"]
    assert body["compiled"]["metadata"]["body_reclosure_mode"] == (
        LFM2_BODY_RECLOSURE_MOVING_GAUGE_V3
    )
    assert body["compiled"]["metadata"][
        "body_coordinate_completion_result_sha256"
    ]
    assert body["compiled"]["metadata"][
        "input_producer_reclosure_state_sha256"
    ]
    assert body["compiled"]["metadata"]["causal_w2_reclosure_count"] == 2
    assert body["compiled"]["metadata"][
        "causal_operator_consumer_reclosure_count"
    ] == 2
    assert body["install"]["metadata"][
        "fresh_primal_w2_reclosure_count"
    ] == 2
    assert body["install"]["metadata"][
        "fresh_primal_operator_consumer_reclosure_count"
    ] == 2
    assert body["install"]["metadata"]["causal_installation_step_count"] == 2
    assert body["validation"]["metrics"][
        "input_producer_reclosure_complete"
    ] is True
    assert body["validation"]["metrics"][
        "per_layer_causal_recapture_complete"
    ] is True
    assert body["runs_before_final_normalization"] is True
    assert report["audit"]["body_reclosure_compile_calls"] == 1
    assert report["audit"]["parent_forward_calls"] == 0
    assert report["audit"]["training_steps"] == 0
    assert report["audit"]["backward_calls"] == 0
    assert report["audit"]["optimizer_steps"] == 0
    parameters_after = tuple(
        (
            name,
            id(parameter),
            int(parameter.data_ptr()),
            tuple(parameter.shape),
            parameter.dtype,
        )
        for name, parameter in child.named_parameters(
            remove_duplicate=False
        )
    )
    exact_count_after = sum(
        int(parameter.numel())
        for parameter in {
            id(value): value for value in child.parameters()
        }.values()
    )
    assert parameters_after == parameters_before
    assert exact_count_after == exact_count_before
    assert len(internal_tensor_refs) == 18
    assert initial_body_tensor_ref is not None
    assert initial_body_tensor_ref() is None
    assert prior_tensor_released == [True] * 17
    assert all(reference() is None for reference in internal_tensor_refs)
    assert external_body_selections == [(0, 1), (1,)]
    assert internal_selections == [
        *((0,),) * 4,
        *((1,),) * 5,
        *((0,),) * 4,
        *((1,),) * 5,
    ]
    assert internal_roles == [
        *(
            "pre_operator",
            "post_operator_producer",
            "post_operator_consumer",
            "post_mlp_producer",
            "pre_operator",
            "post_operator_producer",
            "post_operator_consumer",
            "post_mlp_producer",
            "terminal_receipt",
        ),
        *(
            "pre_operator",
            "post_operator_producer",
            "post_operator_consumer",
            "post_mlp_producer",
            "pre_operator",
            "post_operator_producer",
            "post_operator_consumer",
            "post_mlp_producer",
            "terminal_receipt",
        ),
    ]

    preview_events = [
        (stage, metadata)
        for stage, metadata in progress_events
        if stage.startswith("moving_gauge_preview_recapture_")
    ]
    install_events = [
        (stage, metadata)
        for stage, metadata in progress_events
        if stage.startswith("moving_gauge_install_recapture_")
    ]
    assert len(preview_events) == 18
    assert len(install_events) == 18
    for events, prefix in (
        (preview_events, "moving_gauge_preview_recapture_"),
        (install_events, "moving_gauge_install_recapture_"),
    ):
        assert [stage for stage, _metadata in events] == [
            prefix + suffix
            for _index in range(9)
            for suffix in ("started", "completed")
        ]
        assert all(
            set(metadata).issubset(
                {"layer_index", "capture_boundary", "capture_sha256"}
            )
            for _stage, metadata in events
        )

    payload = adapter._body_reclosure_payload
    assert payload is not None
    assert payload.causal_operator_parameter_ids == (
        "model.layers.0.conv.out_proj.weight",
        "model.layers.1.self_attn.out_proj.weight",
    )
    assert len(payload.causal_operator_replacement_weights) == 2
    steps = payload.causal_w2_step_receipts
    assert len(steps) == 2
    for layer_index, step in enumerate(steps):
        assert step["layer_index"] == layer_index
        assert step[
            "operator_consumer_refreshed_after_producer_recapture"
        ] is True
        assert step[
            "mlp_producers_fit_after_operator_consumer_recapture"
        ] is True
        assert step["w2_solved_after_all_upstream_recaptures"] is True
        assert step["post_operator_producer_capture_sha256"]
        assert step["post_operator_consumer_capture_sha256"]
        assert step["post_mlp_producer_capture_sha256"]
        assert step["post_w2_capture_sha256"]
    assert steps[1]["pre_operator_capture_sha256"] == steps[0][
        "post_w2_capture_sha256"
    ]


def test_moving_gauge_install_rejects_a_stale_causal_capture_plan(
    sealed_parent,
) -> None:
    parent, _cache, _replay_config = sealed_parent
    basis = build_lfm2_weight_complete_basis(parent, 4)
    adapter, _child, _mapping = _adapter(
        sealed_parent,
        existing_mapping=basis,
        body_reclosure_mode=LFM2_BODY_RECLOSURE_MOVING_GAUGE_V3,
        parent_residual_operators=basis.inventory.operators,
        output_interface_mode="identity-skip",
    )
    state_before = adapter.child_state_fingerprint()
    original_recapture = adapter._body_internal_recapture
    tampered = False

    def stale_install_recapture(self, records, audit, **kwargs):
        nonlocal tampered
        capture = original_recapture(records, audit, **kwargs)
        if (
            not tampered
            and self._body_reclosure_payload is not None
            and self._body_reclosure_installed_artifact_sha256 is None
        ):
            tampered = True
            replacement_sha256 = (
                "1" * 64
                if capture.capture_sha256 == "0" * 64
                else "0" * 64
            )
            return replace(
                capture,
                capture_sha256=replacement_sha256,
            )
        return capture

    adapter._body_internal_recapture = MethodType(
        stale_install_recapture,
        adapter,
    )
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="capture precondition lost authentication",
    ):
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.25,
                split_seed="lfm2-moving-gauge-v3-stale-capture-v1",
                fold_body_reclosure=True,
                fold_output_interface=False,
            ),
        ).run()

    assert tampered is True
    assert adapter.child_state_fingerprint() == state_before


def test_moving_gauge_cleanup_releases_every_capture_and_rolls_back(
    sealed_parent,
) -> None:
    parent, _cache, _replay_config = sealed_parent
    basis = build_lfm2_weight_complete_basis(parent, 4)
    adapter, _child, _mapping = _adapter(
        sealed_parent,
        existing_mapping=basis,
        body_reclosure_mode=LFM2_BODY_RECLOSURE_MOVING_GAUGE_V3,
        parent_residual_operators=basis.inventory.operators,
        output_interface_mode="identity-skip",
    )
    state_before = adapter.child_state_fingerprint()
    topology_before = adapter.child_topology_fingerprint()
    original_completion = adapter._compile_moving_gauge_causal_completion
    original_release = adapter._release_capture_lease
    inside_completion = False
    cleanup_attempts = []
    injected = False

    def observed_completion(self, **kwargs):
        nonlocal inside_completion
        inside_completion = True
        try:
            return original_completion(**kwargs)
        finally:
            inside_completion = False

    def fail_first_cleanup(self, capture):
        nonlocal injected
        already_released = capture.payload.value is None
        original_release(capture)
        if inside_completion and already_released:
            cleanup_attempts.append(id(capture))
            if not injected:
                injected = True
                raise RuntimeError(
                    "scripted first moving-gauge cleanup failure"
                )

    adapter._compile_moving_gauge_causal_completion = MethodType(
        observed_completion,
        adapter,
    )
    adapter._release_capture_lease = MethodType(
        fail_first_cleanup,
        adapter,
    )

    with pytest.raises(
        ClosedLoopCleanupError,
        match="Moving-gauge causal preview cleanup failed",
    ):
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.25,
                split_seed="lfm2-moving-gauge-cleanup-v1",
                fold_body_reclosure=True,
                fold_output_interface=False,
            ),
        ).run()

    assert injected is True
    assert len(cleanup_attempts) > 1
    assert adapter.child_state_fingerprint() == state_before
    assert adapter.child_topology_fingerprint() == topology_before


def test_tensor_sha256_memoryview_preserves_the_legacy_digest_contract() -> None:
    value = torch.tensor(
        [[1.25, -2.5], [3.75, 4.5]],
        dtype=torch.float32,
    )
    expected = hashlib.sha256()
    expected.update(str(value.dtype).encode("ascii"))
    expected.update(b"\0")
    expected.update(b"[2,2]")
    expected.update(b"\0")
    expected.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())

    assert adapter_module._tensor_sha256(value) == expected.hexdigest()


def test_moving_gauge_factor_writer_uses_cached_output_action_evidence() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1917)
        encoder = torch.nn.Parameter(
            torch.randn(1, 5, dtype=torch.bfloat16),
            requires_grad=False,
        )
        decoder = torch.nn.Parameter(
            torch.randn(4, 1, dtype=torch.bfloat16),
            requires_grad=False,
        )
        requested = torch.randn(4, 5, dtype=torch.bfloat16)
        design = torch.randn(32, 5, dtype=torch.float64)
        desired = design @ requested.to(dtype=torch.float64).T
        weights = torch.arange(1, 33, dtype=torch.float64)
        weights /= weights.sum()
    parameter_id = "model.layers.0.conv.out_proj.weight"
    site = adapter_module._MovingGaugeWriterSite(
        parameter_id=parameter_id,
        topology="factorized",
        writer_rank=1,
        writer_contract_sha256="2" * 64,
        seed_physical_state_sha256="3" * 64,
        physical_parameter_ids=(
            "model.layers.0.conv.out_proj.0.weight",
            "model.layers.0.conv.out_proj.1.weight",
        ),
        physical_parameters=(encoder, decoder),
    )

    action_aware = adapter_module._compile_moving_gauge_factor_writer_artifact(
        site,
        requested,
        output_action_rows=desired,
        row_weights=weights,
    )
    repeated = adapter_module._compile_moving_gauge_factor_writer_artifact(
        site,
        requested,
        output_action_rows=desired,
        row_weights=weights,
    )
    plain = adapter_module._compile_moving_gauge_factor_writer_artifact(
        site,
        requested,
    )

    assert action_aware.factorization_method == (
        "deterministic-structural-cpu-fp64-weighted-output-action-subspace"
    )
    assert action_aware.action_evidence_sha256
    assert action_aware.artifact_sha256 == repeated.artifact_sha256
    assert torch.equal(action_aware.encoder_weight, repeated.encoder_weight)
    assert torch.equal(action_aware.decoder_weight, repeated.decoder_weight)
    action_effective = (
        action_aware.decoder_weight.to(torch.float64)
        @ action_aware.encoder_weight.to(torch.float64)
    )
    plain_effective = (
        plain.decoder_weight.to(torch.float64)
        @ plain.encoder_weight.to(torch.float64)
    )
    action_error = (
        ((design @ action_effective.T - desired).square().sum(dim=1) * weights)
        .sum()
        .item()
    )
    plain_error = (
        ((design @ plain_effective.T - desired).square().sum(dim=1) * weights)
        .sum()
        .item()
    )
    assert action_error <= plain_error


def test_moving_gauge_writer_rows_normalize_authenticated_attention_layout(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    operator = adapter._child_contracts[1].operator
    query_heads = operator.query_heads
    head_dim = operator.head_dim
    input_width = query_heads * head_dim

    for sampled_tokens in (1, 3):
        attention_context = torch.arange(
            2 * query_heads * sampled_tokens * head_dim,
            dtype=torch.float32,
        ).reshape(2, query_heads, sampled_tokens, head_dim)
        paired = SimpleNamespace(
            load_batch=lambda _index, value=attention_context: SimpleNamespace(
                child=SimpleNamespace(
                    tensors={"prefill.layer.001.attention.context": value}
                )
            )
        )
        manifest = (
            {
                "batch_index": 0,
                "phase": "prefill",
                "leading_shape": (2, sampled_tokens),
            },
        )

        rows = adapter._moving_gauge_child_writer_rows(
            paired,
            manifest,
            layer_index=1,
            suffix="attention.context",
            input_width=input_width,
        )
        expected = (
            attention_context.permute(0, 2, 1, 3)
            .to(dtype=torch.float64)
            .reshape(2 * sampled_tokens, input_width)
        )

        assert rows.shape == (2 * sampled_tokens, input_width)
        assert torch.equal(rows, expected)


def test_real_adapter_runs_all_layers_with_all_batch_live_child_recaptures(
    sealed_parent, monkeypatch
) -> None:
    adapter, child, mapping = _adapter(sealed_parent)
    parent = adapter.source_model
    tied_parameter = child.get_input_embeddings().weight
    assert child.get_output_embeddings().weight is tied_parameter
    tied_parameter_id = id(tied_parameter)
    parent_forward_calls = 0

    def forbidden_parent_forward(self, *_args, **_kwargs):
        nonlocal parent_forward_calls
        parent_forward_calls += 1
        raise AssertionError("closed-loop adapter executed the parent")

    parent.forward = MethodType(forbidden_parent_forward, parent)

    captures = []
    original_capture = adapter_module.capture_lfm2_child_replay

    def observed_capture(child_model, replay_batches, **kwargs):
        batches = tuple(replay_batches)
        assert child_model is child
        assert len(batches) == len(adapter.parent_cache)
        result = original_capture(child_model, batches, **kwargs)
        captures.append(result)
        return result

    monkeypatch.setattr(
        adapter_module,
        "capture_lfm2_child_replay",
        observed_capture,
    )
    report = ChildClosedLoopCompiler(
        adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.25,
            split_seed="lfm2-adapter-test-v1",
        ),
    ).run()

    assert report["selected_layers"] == [0, 1]
    assert report["layers_executed_in_ascending_order"]
    assert report["final_normalization"][
        "runs_after_all_ascending_layers"
    ]
    assert len(captures) == 13
    assert all(
        len(replay) == len(adapter.parent_cache) for replay in captures
    )
    assert all(
        replay.metadata["runtime"]["parent_forward_count"] == 0
        for replay in captures
    )
    assert all(
        replay.metadata["selection"]["capture"]["decode_trace_mode"]
        == FULL_PREFIX_CANONICAL_TRACE_MODE
        for replay in captures
    )
    assert all(
        replay.metadata["selection"]["replay_capture_mode"]
        == CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
        for replay in captures
    )
    assert [replay.selected_layer_indices for replay in captures] == [
        *((0,),) * 4,
        *((1,),) * 9,
    ]
    assert parent_forward_calls == 0
    assert report["audit"]["parent_forward_calls"] == 0
    assert report["audit"]["parent_trace_regeneration_calls"] == 0
    assert report["audit"]["training_steps"] == 0
    assert report["audit"]["backward_calls"] == 0
    assert report["audit"]["optimizer_steps"] == 0
    assert report["audit"]["child_replay_capture_calls"] == 13
    assert report["audit"]["output_interface_compile_calls"] == 1
    assert report["child_topology_preserved"]
    assert report["output_interface"][
        "post_install_full_live_child_replayed"
    ]
    assert report["output_interface"][
        "teacher_forced_terminal_acceptance_is_necessary_not_sufficient"
    ]
    assert report["output_interface"][
        "external_whole_output_behavior_gate_required_for_promotion"
    ]
    assert child.get_input_embeddings().weight is tied_parameter
    assert child.get_output_embeddings().weight is tied_parameter
    assert id(tied_parameter) == tied_parameter_id
    output_metrics = report["output_interface"]["validation"]["metrics"]
    assert output_metrics["full_vocabulary_metrics"]
    assert output_metrics["tied_embedding_feedback_included"]
    assert output_metrics[
        "kl_known_sequence_nll_and_eos_are_terminal_acceptance_gates"
    ]
    assert output_metrics[
        "parent_distribution_cross_entropy_is_redundant_with_kl"
    ]
    assert output_metrics["live_candidate_vocabulary_size"] == 41
    assert output_metrics["live_candidate_rows"] > 0
    assert mapping.map_sha256 == adapter.map_sha256
    for layer in report["layer_reports"]:
        metrics = layer["validation"]["metrics"]
        assert metrics[
            "heldout_endpoint_compared_to_projected_parent"
        ]
        assert layer["consumer"][
            "fitted_after_installed_producer_recapture"
        ]
        assert layer["downstream"][
            "compiled_after_installed_consumer_recapture"
        ]


def test_scope_local_profiles_and_capture_hash_bind_exact_selection(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    last = len(adapter.child_layers) - 1
    assert adapter._capture_layer_selection(
        FoldScope("layer", 0), CapturePoint.PRE_PRODUCER
    ) == (0,)
    assert adapter._capture_layer_selection(
        FoldScope("body_reclosure"), CapturePoint.PRE_BODY_RECLOSURE
    ) == tuple(range(len(adapter.child_layers)))
    assert adapter._capture_layer_selection(
        FoldScope("body_reclosure"), CapturePoint.POST_BODY_RECLOSURE
    ) == (last,)
    assert adapter._capture_layer_selection(
        FoldScope("final_normalization"),
        CapturePoint.PRE_FINAL_NORMALIZATION,
    ) == (last,)
    assert adapter._capture_layer_selection(
        FoldScope("output_interface"), CapturePoint.POST_OUTPUT_INTERFACE
    ) == (last,)

    pre = adapter._child_evidence_profile(
        FoldScope("layer", 0),
        CapturePoint.PRE_PRODUCER,
        (0,),
        body_internal_role=None,
    )
    assert pre is not None
    assert pre.profile_role == "layer.pre_producer"
    assert pre.suffixes_for_layer(0) == (
        "residual.post_layer",
        "residual.pre",
    )
    producer = adapter._child_evidence_profile(
        FoldScope("layer", 0),
        CapturePoint.POST_PRODUCER,
        (0,),
        body_internal_role=None,
    )
    assert producer is not None
    assert producer.suffixes_for_layer(0) == (
        "residual.post_operator",
        "residual.pre",
        "short_conv.context",
    )
    consumer = adapter._child_evidence_profile(
        FoldScope("layer", 1),
        CapturePoint.POST_CONSUMER,
        (1,),
        body_internal_role=None,
    )
    assert consumer is not None
    assert consumer.suffixes_for_layer(1) == (
        "attention.context",
        "residual.post_layer",
        "residual.post_operator",
        "residual.pre",
    )
    body_pre = adapter._child_evidence_profile(
        FoldScope("body_reclosure"),
        CapturePoint.PRE_BODY_RECLOSURE,
        (0, 1),
        body_internal_role=None,
    )
    assert body_pre is not None
    assert body_pre.suffixes_for_layer(0) == (
        "mlp.input",
        "mlp.post_swiglu",
        "operator.input",
        "residual.post_layer",
        "residual.pre",
        "short_conv.context",
    )
    assert body_pre.suffixes_for_layer(1) == (
        "attention.context",
        "mlp.input",
        "mlp.post_swiglu",
        "operator.input",
        "residual.post_layer",
        "residual.pre",
    )
    expected_internal = {
        "pre_operator": ("operator.input", "operator.output", "residual.pre"),
        "post_operator_producer": ("residual.pre", "short_conv.context"),
        "post_operator_consumer": ("mlp.input", "residual.pre"),
        "post_mlp_producer": (
            "mlp.post_swiglu",
            "residual.post_operator",
            "residual.pre",
        ),
        "terminal_receipt": ("residual.pre",),
    }
    for role, expected_suffixes in expected_internal.items():
        profile = adapter._child_evidence_profile(
            FoldScope("body_reclosure"),
            CapturePoint.POST_BODY_RECLOSURE,
            (0,),
            body_internal_role=role,
        )
        assert profile is not None
        assert profile.profile_role == f"body_reclosure.internal.{role}"
        assert profile.suffixes_for_layer(0) == expected_suffixes
    output = adapter._child_evidence_profile(
        FoldScope("output_interface"),
        CapturePoint.PRE_OUTPUT_INTERFACE,
        (last,),
        body_internal_role=None,
    )
    assert output is not None
    assert output.retain_logits is True
    assert output.retain_lm_head_input is True
    assert output.retain_phase_forced_token_ids is False

    audit = ClosedLoopAudit()
    records = tuple(adapter.parent_trajectory_records(audit))
    capture = adapter.replay_and_capture_child(
        FoldScope("layer", 0),
        CapturePoint.PRE_PRODUCER,
        records[:1],
        audit,
    )
    payload = capture.payload.get()
    assert payload.child_evidence_profile_role == "layer.pre_producer"
    assert payload.child_evidence_profile_sha256 == pre.profile_sha256
    assert payload.paired.child_evidence_profile_sha256 == pre.profile_sha256
    assert payload.child_evidence_profile is not None
    mutable_copy = payload.child_evidence_profile.as_dict()
    mutable_copy["layers"][0]["retained_suffixes"].append("operator.output")
    assert mutable_copy != payload.child_evidence_profile.as_dict()
    assert adapter._capture_payload(capture) is payload
    forged = replace(
        capture,
        payload=adapter_module._CapturePayloadLease(
            replace(payload, selected_layer_indices=(0, 1))
        ),
    )
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="canonical all-batch child replay",
    ):
        adapter._capture_payload(forged)
    forged_profile = replace(
        capture,
        payload=adapter_module._CapturePayloadLease(
            replace(payload, child_evidence_profile_sha256="0" * 64)
        ),
    )
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="canonical all-batch child replay",
    ):
        adapter._capture_payload(forged_profile)


def test_shared_selective_session_is_evidence_equivalent_without_full_reads(
    sealed_parent,
    monkeypatch,
) -> None:
    parent, cache, _replay_config = sealed_parent
    baseline_basis = build_lfm2_circuit_balanced_basis(
        parent,
        cache,
        4,
    )
    baseline_adapter, child, mapping = _adapter(sealed_parent)
    initial_state = {
        name: value.detach().clone() for name, value in child.state_dict().items()
    }
    compiler_config = ClosedLoopCompilerConfig(
        validation_fraction=0.25,
        split_seed="lfm2-selective-session-equivalence-v1",
    )
    with cache.selective_authenticated_session() as selective_session:
        selective_adapter, _selective_child, _mapping = _adapter(
            sealed_parent,
            selective_session=selective_session,
            existing_child=child,
            existing_mapping=mapping,
        )
        assert (
            baseline_adapter.child_state_fingerprint()
            == selective_adapter.child_state_fingerprint()
        )
        baseline_report = ChildClosedLoopCompiler(
            baseline_adapter,
            compiler_config,
        ).run()
        child.load_state_dict(initial_state, strict=True)
        assert (
            baseline_report["child_state_sha256_before"]
            == selective_adapter.child_state_fingerprint()
        )

        session_type = Lfm2CircuitTraceSelectiveAuthenticatedSession
        original_selected_load = session_type.load_selected_tensors
        selected_payloads = []

        def observed_selected_load(self, index, tensor_names):
            payload = original_selected_load(self, index, tensor_names)
            selected_payloads.append(payload)
            return payload

        def forbidden_full_read(*_args, **_kwargs):
            raise AssertionError("selective consumer performed a full cache read")

        monkeypatch.setattr(
            session_type,
            "load_selected_tensors",
            observed_selected_load,
        )
        monkeypatch.setattr(
            Lfm2CircuitTraceCache,
            "verify",
            forbidden_full_read,
        )
        monkeypatch.setattr(
            Lfm2CircuitTraceCache,
            "iter_batches",
            forbidden_full_read,
        )
        monkeypatch.setattr(
            Lfm2CircuitTraceCache,
            "load_batch",
            forbidden_full_read,
        )
        selective_basis = build_lfm2_circuit_balanced_basis(
            parent,
            cache,
            4,
            selective_session=selective_session,
        )
        selective_report = ChildClosedLoopCompiler(
            selective_adapter,
            compiler_config,
        ).run()

        assert selective_session.verify() is selective_session
        assert baseline_basis.as_dict() == selective_basis.as_dict()
        assert torch.equal(baseline_basis.basis, selective_basis.basis)
        assert baseline_report == selective_report
        assert selected_payloads
        assert all(payload.released for payload in selected_payloads)


def test_selected_parent_profiles_are_exact_refcounted_and_switch_fail_closed(
    sealed_parent,
    monkeypatch,
) -> None:
    parent, cache, _replay_config = sealed_parent
    with cache.selective_authenticated_session() as selective_session:
        adapter, _child, _mapping = _adapter(
            sealed_parent,
            selective_session=selective_session,
        )

        def forbidden_parent_or_full_read(*_args, **_kwargs):
            raise AssertionError("selected profile escaped cache-only loading")

        monkeypatch.setattr(parent, "forward", forbidden_parent_or_full_read)
        monkeypatch.setattr(
            Lfm2CircuitTraceCache,
            "iter_batches",
            forbidden_parent_or_full_read,
        )
        monkeypatch.setattr(
            Lfm2CircuitTraceCache,
            "load_batch",
            forbidden_parent_or_full_read,
        )

        first = adapter._acquire_selected_parent_profile(
            FoldScope("layer", 0),
            CapturePoint.PRE_PRODUCER,
            (0,),
            body_internal=False,
        )
        profile = first.profile
        first_payloads = tuple(profile.payloads)
        assert profile.borrow_count == 1
        assert profile.selected_layer_indices == (0,)
        for names in profile.ordered_tensor_names_by_batch:
            assert not any(name.endswith(".logits") for name in names)
            assert not any(".layer.001." in name for name in names)
            assert all(
                name in adapter_module._SELECTIVE_INPUT_TENSORS
                or ".layer.000." in name
                for name in names
            )
            assert not any(
                name.endswith("attention.key_cache")
                or name.endswith("attention.value_cache")
                for name in names
            )

        second = adapter._acquire_selected_parent_profile(
            FoldScope("layer", 0),
            CapturePoint.POST_PRODUCER,
            (0,),
            body_internal=False,
        )
        assert second.profile is profile
        assert first.borrow_sha256 != second.borrow_sha256
        assert profile.borrow_count == 2
        with pytest.raises(
            Lfm2DynamicClosedLoopAdapterError,
            match="incompatible capture borrow is live",
        ):
            adapter._acquire_selected_parent_profile(
                FoldScope("layer", 1),
                CapturePoint.PRE_PRODUCER,
                (1,),
                body_internal=False,
            )
        first.release()
        second.release()
        assert profile.borrow_count == 0

        switched = adapter._acquire_selected_parent_profile(
            FoldScope("layer", 1),
            CapturePoint.PRE_PRODUCER,
            (1,),
            body_internal=False,
        )
        assert switched.profile is not profile
        assert all(payload.released for payload in first_payloads)
        switched.release()

        last = len(adapter.child_layers) - 1
        body_pre = adapter._acquire_selected_parent_profile(
            FoldScope("body_reclosure"),
            CapturePoint.PRE_BODY_RECLOSURE,
            tuple(range(len(adapter.child_layers))),
            body_internal=False,
        )
        for names in body_pre.profile.ordered_tensor_names_by_batch:
            assert not any(name.endswith(".logits") for name in names)
            assert all(
                name in adapter_module._SELECTIVE_INPUT_TENSORS
                or name.endswith("residual.pre")
                or name.endswith("residual.post_layer")
                for name in names
            )
            for layer_index in range(len(adapter.child_layers)):
                assert any(
                    f".layer.{layer_index:03d}.residual.pre" in name
                    for name in names
                )
                assert any(
                    f".layer.{layer_index:03d}.residual.post_layer" in name
                    for name in names
                )
        body_pre.release()

        body_internal = adapter._acquire_selected_parent_profile(
            FoldScope("body_reclosure"),
            CapturePoint.POST_BODY_RECLOSURE,
            (0,),
            body_internal=True,
        )
        for names in body_internal.profile.ordered_tensor_names_by_batch:
            assert all(
                name in adapter_module._SELECTIVE_INPUT_TENSORS
                or ".layer.000.residual.pre" in name
                for name in names
            )
            assert not any(name.endswith("residual.post_layer") for name in names)
        body_internal.release()

        final_profile = adapter._acquire_selected_parent_profile(
            FoldScope("final_normalization"),
            CapturePoint.PRE_FINAL_NORMALIZATION,
            (last,),
            body_internal=False,
        )
        for names in final_profile.profile.ordered_tensor_names_by_batch:
            assert not any(name.endswith(".logits") for name in names)
            assert all(
                name in adapter_module._SELECTIVE_INPUT_TENSORS
                or f".layer.{last:03d}.residual.pre" in name
                or f".layer.{last:03d}.residual.post_layer" in name
                for name in names
            )
        final_profile.release()

        output_pre = adapter._acquire_selected_parent_profile(
            FoldScope("output_interface"),
            CapturePoint.PRE_OUTPUT_INTERFACE,
            (last,),
            body_internal=False,
        )
        output_post = adapter._acquire_selected_parent_profile(
            FoldScope("output_interface"),
            CapturePoint.POST_OUTPUT_INTERFACE,
            (last,),
            body_internal=False,
        )
        assert output_pre.profile is output_post.profile
        assert output_pre.profile.borrow_count == 2
        for names in output_pre.profile.ordered_tensor_names_by_batch:
            assert any(name.endswith(".logits") for name in names)
            assert any(name.endswith("residual.pre") for name in names)
            assert not any(name.endswith("residual.post_layer") for name in names)
            assert all(
                name in adapter_module._SELECTIVE_INPUT_TENSORS
                or f".layer.{last:03d}.residual.pre" in name
                or name.endswith(".logits")
                for name in names
            )
        output_pre.release()
        output_post.release()
        adapter._release_idle_selected_parent_profile()
        assert adapter._active_selected_parent_profile is None
        assert not selective_session.closed


def test_selected_parent_profile_partial_load_releases_and_tamper_fails_closed(
    sealed_parent,
    monkeypatch,
) -> None:
    _parent, cache, _replay_config = sealed_parent
    with cache.selective_authenticated_session() as selective_session:
        adapter, _child, _mapping = _adapter(
            sealed_parent,
            selective_session=selective_session,
        )
        session_type = Lfm2CircuitTraceSelectiveAuthenticatedSession
        original_load = session_type.load_selected_tensors
        partial_payloads = []

        def fail_second_load(self, index, tensor_names):
            if self is selective_session and index == 1:
                raise RuntimeError("scripted selected load failure")
            payload = original_load(self, index, tensor_names)
            if self is selective_session:
                partial_payloads.append(payload)
            return payload

        monkeypatch.setattr(
            session_type,
            "load_selected_tensors",
            fail_second_load,
        )
        with pytest.raises(RuntimeError, match="scripted selected load failure"):
            adapter._acquire_selected_parent_profile(
                FoldScope("layer", 0),
                CapturePoint.PRE_PRODUCER,
                (0,),
                body_internal=False,
            )
        assert partial_payloads
        assert all(payload.released for payload in partial_payloads)
        assert adapter._active_selected_parent_profile is None
        assert not selective_session.closed

        monkeypatch.setattr(
            session_type,
            "load_selected_tensors",
            original_load,
        )
        borrow = adapter._acquire_selected_parent_profile(
            FoldScope("layer", 0),
            CapturePoint.PRE_PRODUCER,
            (0,),
            body_internal=False,
        )
        payload = borrow.profile.payloads[0]
        selection_sha256 = payload.selection_sha256
        payload.selection_sha256 = "0" * 64
        with pytest.raises(
            Lfm2DynamicClosedLoopAdapterError,
            match="payload changed after authenticated load",
        ):
            adapter._acquire_selected_parent_profile(
                FoldScope("layer", 0),
                CapturePoint.POST_PRODUCER,
                (0,),
                body_internal=False,
            )
        payload.selection_sha256 = selection_sha256
        borrow.release()
        adapter._release_idle_selected_parent_profile()
        assert payload.released
        assert not selective_session.closed


def test_selected_parent_profile_failed_admission_releases_every_payload(
    sealed_parent,
    monkeypatch,
) -> None:
    _parent, cache, _replay_config = sealed_parent
    with cache.selective_authenticated_session() as selective_session:
        adapter, _child, _mapping = _adapter(
            sealed_parent,
            selective_session=selective_session,
        )
        session_type = Lfm2CircuitTraceSelectiveAuthenticatedSession
        original_load = session_type.load_selected_tensors
        loaded_payloads = []

        def load_malformed_profile(self, index, tensor_names):
            payload = original_load(self, index, tensor_names)
            if self is selective_session:
                loaded_payloads.append(payload)
                if index == 0:
                    payload.batch_index = 999
            return payload

        monkeypatch.setattr(
            session_type,
            "load_selected_tensors",
            load_malformed_profile,
        )
        with pytest.raises(
            Lfm2DynamicClosedLoopAdapterError,
            match="payload changed after authenticated load",
        ):
            adapter._acquire_selected_parent_profile(
                FoldScope("layer", 0),
                CapturePoint.PRE_PRODUCER,
                (0,),
                body_internal=False,
            )

        assert len(loaded_payloads) == len(selective_session.batch_references)
        assert all(payload.released for payload in loaded_payloads)
        assert adapter._active_selected_parent_profile is None
        assert not selective_session.closed


def test_parent_profile_and_operator_rows_are_collected_at_exact_boundaries(
    sealed_parent,
) -> None:
    _parent, cache, _replay_config = sealed_parent
    profile_tensor_ref = None
    operator_tensor_ref = None
    operator_rows_released = False
    profile_tensor_released = False
    adapter = None

    def observe_progress(stage, metadata):
        nonlocal profile_tensor_ref
        nonlocal operator_tensor_ref
        nonlocal operator_rows_released
        nonlocal profile_tensor_released
        if (
            stage == "child_closed_loop_layer_producer_compiled"
            and metadata["layer_index"] == 0
        ):
            profile = adapter._active_selected_parent_profile
            assert profile is not None
            profile_tensor_ref = weakref.ref(
                next(iter(profile.payloads[0].tensors.values()))
            )
            case = adapter._producer_payloads[0].operator_cases.fit_cases[0]
            operator_tensor_ref = weakref.ref(case.target_input)
        elif (
            stage == "child_closed_loop_layer_downstream_compiled"
            and metadata["layer_index"] == 0
        ):
            gc.collect()
            assert operator_tensor_ref is not None
            operator_rows_released = operator_tensor_ref() is None
        elif (
            stage == "child_closed_loop_layer_capture_completed"
            and metadata["layer_index"] == 1
            and metadata["capture_point"] == CapturePoint.PRE_PRODUCER.value
        ):
            gc.collect()
            assert profile_tensor_ref is not None
            profile_tensor_released = profile_tensor_ref() is None

    with cache.selective_authenticated_session() as selective_session:
        adapter, _child, _mapping = _adapter(
            sealed_parent,
            selective_session=selective_session,
        )
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.25,
                split_seed="lfm2-selected-parent-release-boundaries-v1",
            ),
            progress_sink=observe_progress,
        ).run()

        assert operator_rows_released is True
        assert profile_tensor_released is True
        assert adapter._active_selected_parent_profile is None


def test_output_install_exception_releases_all_captures_and_parent_payloads(
    sealed_parent,
    monkeypatch,
) -> None:
    _parent, cache, _replay_config = sealed_parent
    with cache.selective_authenticated_session() as selective_session:
        adapter, _child, _mapping = _adapter(
            sealed_parent,
            selective_session=selective_session,
        )
        state_before = adapter.child_state_fingerprint()
        topology_before = adapter.child_topology_fingerprint()
        session_type = Lfm2CircuitTraceSelectiveAuthenticatedSession
        original_load = session_type.load_selected_tensors
        loaded_payloads = []

        def observed_load(self, index, tensor_names):
            payload = original_load(self, index, tensor_names)
            if self is selective_session:
                loaded_payloads.append(payload)
            return payload

        def fail_output_install(self, scope, compiled, audit):
            del self, scope, compiled, audit
            raise RuntimeError("scripted output install failure")

        monkeypatch.setattr(
            session_type,
            "load_selected_tensors",
            observed_load,
        )
        adapter.install_output_interface = MethodType(
            fail_output_install,
            adapter,
        )
        compiler = ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.25,
                split_seed="lfm2-output-install-capture-cleanup-v1",
            ),
        )

        with pytest.raises(RuntimeError, match="scripted output install failure"):
            compiler.run()

        assert loaded_payloads
        assert all(payload.released for payload in loaded_payloads)
        assert compiler._owned_captures == {}
        assert adapter._active_selected_parent_profile is None
        assert adapter.child_state_fingerprint() == state_before
        assert adapter.child_topology_fingerprint() == topology_before


def test_dual_output_cleanup_continues_after_first_release_error(
    sealed_parent,
) -> None:
    _parent, cache, _replay_config = sealed_parent
    with cache.selective_authenticated_session() as selective_session:
        adapter, _child, _mapping = _adapter(
            sealed_parent,
            selective_session=selective_session,
        )
        state_before = adapter.child_state_fingerprint()
        topology_before = adapter.child_topology_fingerprint()
        original_replay = adapter.replay_and_capture_child
        original_release = adapter._release_capture_lease
        output_pre_captures = []
        output_post_captures = []
        injected = False

        def observed_replay(self, scope, point, records, audit):
            capture = original_replay(scope, point, records, audit)
            if scope.kind == "output_interface":
                if point is CapturePoint.PRE_OUTPUT_INTERFACE:
                    output_pre_captures.append(capture)
                elif point is CapturePoint.POST_OUTPUT_INTERFACE:
                    output_post_captures.append(capture)
            return capture

        def fail_validation_baseline_release(self, capture):
            nonlocal injected
            original_release(capture)
            if (
                len(output_pre_captures) == 2
                and capture is output_pre_captures[1]
                and not injected
            ):
                injected = True
                raise RuntimeError("scripted first output cleanup failure")

        adapter.replay_and_capture_child = MethodType(
            observed_replay,
            adapter,
        )
        adapter._release_capture_lease = MethodType(
            fail_validation_baseline_release,
            adapter,
        )

        with pytest.raises(
            ClosedLoopCleanupError,
            match="validate_output_interface capture cleanup failed",
        ):
            ChildClosedLoopCompiler(
                adapter,
                ClosedLoopCompilerConfig(
                    validation_fraction=0.25,
                    split_seed="lfm2-dual-output-capture-cleanup-v1",
                ),
            ).run()

        assert injected is True
        assert len(output_pre_captures) == 2
        assert len(output_post_captures) == 1
        assert all(
            capture.payload.value is None
            for capture in output_pre_captures + output_post_captures
        )
        assert adapter._active_selected_parent_profile is None
        assert adapter.child_state_fingerprint() == state_before
        assert adapter.child_topology_fingerprint() == topology_before


def test_macro_cell_attractor_gate_is_reported_for_every_layer(
    sealed_parent,
) -> None:
    gate = _ScriptedAttractorGate(accepted=True)
    adapter, _child, _mapping = _adapter(
        sealed_parent,
        macro_cell_attractor_gate=gate,
    )

    report = ChildClosedLoopCompiler(
        adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.25,
            split_seed="lfm2-macro-cell-attractor-pass-v1",
        ),
    ).run()

    assert gate.phases == [
        "pre-macro-cell-layer-000000",
        "post-macro-cell-layer-000000",
        "pre-macro-cell-layer-000001",
        "post-macro-cell-layer-000001",
    ]
    for layer in report["layer_reports"]:
        compiled = layer["producer"]["compiled"]
        metrics = layer["validation"]["metrics"]
        assert compiled["metadata"][
            "macro_cell_attractor_gate_enabled"
        ]
        assert metrics["macro_cell_attractor_gate_enabled"]
        assert metrics["macro_cell_attractor_gate_accepted"]
        assert metrics[
            "macro_cell_exact_token_metrics_used_for_acceptance"
        ] is False
        assert metrics["macro_cell_parent_forward_calls"] == 0
        assert metrics["macro_cell_training_steps"] == 0


def test_macro_cell_attractor_rejection_rolls_back_whole_child_exactly(
    sealed_parent,
) -> None:
    gate = _ScriptedAttractorGate(accepted=False)
    adapter, child, _mapping = _adapter(
        sealed_parent,
        macro_cell_attractor_gate=gate,
    )
    parameter = child.model.layers[0].feed_forward.w1.weight
    parameter.grad = torch.arange(
        parameter.numel(), dtype=parameter.dtype
    ).reshape_as(parameter)
    parameter_id = id(parameter)
    gradient_id = id(parameter.grad)
    gradient_before = parameter.grad.detach().clone()
    state_before = adapter.child_state_fingerprint()
    topology_before = adapter.child_topology_fingerprint()

    with pytest.raises(
        ClosedLoopFoldError,
        match=(
            "macro-cell free-rollout attractor gate rejected: "
            "new_collapsed_records=1>0"
        ),
    ):
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.25,
                split_seed="lfm2-macro-cell-attractor-reject-v1",
            ),
        ).run()

    assert gate.phases == [
        "pre-macro-cell-layer-000000",
        "post-macro-cell-layer-000000",
    ]
    assert adapter.child_state_fingerprint() == state_before
    assert adapter.child_topology_fingerprint() == topology_before
    assert id(parameter) == parameter_id
    assert id(parameter.grad) == gradient_id
    assert torch.equal(parameter.grad, gradient_before)


def test_adapter_opt_in_coupled_attention_value_output_path(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(
        sealed_parent,
        attention_consumer_mode="coupled-value-output",
        attention_score_candidate_gate_enabled=True,
        attention_score_minimum_fraction=0.0,
    )
    report = ChildClosedLoopCompiler(
        adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.25,
            split_seed="lfm2-adapter-coupled-value-output-v1",
        ),
    ).run()

    producer = adapter._producer_payloads[1]
    attention_consumer = adapter._consumer_payloads[1].compiled
    assert producer.attention_score_fit is not None
    assert producer.attention_score_validation is not None
    assert producer.attention_score_validation.accepted
    assert attention_consumer.fixed_query_key_sha256 == (
        producer.compiled.score_state_sha256
    )
    assert attention_consumer.as_dict()["score_path_fixed"]
    assert attention_consumer.value_context_solver == (
        "exact-live-context-head-local-value-transport"
    )
    assert attention_consumer.fit_nonincreasing
    assert report["audit"]["training_steps"] == 0
    assert report["audit"]["backward_calls"] == 0
    assert report["audit"]["optimizer_steps"] == 0
    attention_layer = report["layer_reports"][1]
    score_metadata = attention_layer["producer"]["compiled"]["metadata"]
    assert score_metadata["attention_score_candidate_gate_enabled"]
    assert score_metadata["attention_score_candidate_gate_accepted"]
    assert score_metadata["attention_score_parent_forward_calls"] == 0
    assert score_metadata[
        "attention_score_numeric_weight_matching_used"
    ] is False


def test_attention_score_gate_rejection_rolls_back_whole_child_exactly(
    sealed_parent,
) -> None:
    adapter, child, _mapping = _adapter(
        sealed_parent,
        attention_consumer_mode="coupled-value-output",
        attention_score_candidate_gate_enabled=True,
        attention_score_maximum_softmax_fisher_error=0.0,
        attention_score_minimum_fraction=0.0,
    )
    parameter = child.model.layers[0].feed_forward.w1.weight
    parameter.grad = torch.arange(
        parameter.numel(), dtype=parameter.dtype
    ).reshape_as(parameter)
    parameter_id = id(parameter)
    gradient_id = id(parameter.grad)
    gradient_before = parameter.grad.detach().clone()
    state_before = adapter.child_state_fingerprint()
    topology_before = adapter.child_topology_fingerprint()

    with pytest.raises(
        ClosedLoopFoldError,
        match="Held-out attention score/order/margin gate rejected",
    ):
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.25,
                split_seed="lfm2-attention-score-gate-reject-v1",
            ),
        ).run()

    assert adapter.child_state_fingerprint() == state_before
    assert adapter.child_topology_fingerprint() == topology_before
    assert id(parameter) == parameter_id
    assert id(parameter.grad) == gradient_id
    assert torch.equal(parameter.grad, gradient_before)


def test_adapter_single_pass_is_explicit_and_audits_forward_savings(
    sealed_parent,
    monkeypatch,
) -> None:
    adapter, child, _mapping = _adapter(
        sealed_parent,
        replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    )
    captures = []
    original_capture = adapter_module.capture_lfm2_child_replay

    def observed_capture(child_model, replay_batches, **kwargs):
        assert child_model is child
        assert (
            kwargs["replay_capture_mode"]
            == CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
        )
        result = original_capture(
            child_model,
            tuple(replay_batches),
            **kwargs,
        )
        captures.append(result)
        return result

    monkeypatch.setattr(
        adapter_module,
        "capture_lfm2_child_replay",
        observed_capture,
    )
    report = ChildClosedLoopCompiler(
        adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.25,
            split_seed="lfm2-adapter-single-pass-test-v1",
        ),
    ).run()

    assert report["selected_layers"] == [0, 1]
    assert len(captures) == 13
    for replay in captures:
        assert replay.metadata["selection"]["replay_capture_mode"] == (
            CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
        )
        assert replay.metadata["forward_counts"] == {
            "child": len(adapter.parent_cache),
            "parent": 0,
        }
        assert replay.metadata["replay_benchmark"] == {
            "batch_count": 4,
            "phase_count": 8,
            "actual_child_forwards": 4,
            "canonical_prefix_child_forwards": 8,
            "saved_child_forwards": 4,
        }
        contracts = {
            batch.capture["causal_equivalence_contract"][
                "contract_sha256"
            ]
            for batch in replay.iter_batches()
        }
        assert len(contracts) == 1
    assert report["audit"]["child_replay_capture_calls"] == 13
    assert report["audit"]["parent_forward_calls"] == 0
    assert report["audit"]["training_steps"] == 0
    assert report["audit"]["backward_calls"] == 0
    assert report["audit"]["optimizer_steps"] == 0


def test_output_interface_rejection_restores_exact_tied_child(
    sealed_parent,
) -> None:
    adapter, child, _mapping = _adapter(
        sealed_parent,
        minimum_output_improvement=1_000_000_000.0,
        output_validation_tolerance=0.0,
    )
    parameter = child.get_input_embeddings().weight
    assert child.get_output_embeddings().weight is parameter
    parameter_id = id(parameter)
    state_before = adapter.child_state_fingerprint()
    topology_before = adapter.child_topology_fingerprint()
    value_before = parameter.detach().clone()

    with pytest.raises(
        ClosedLoopFoldError,
        match="output_interface validation rejected",
    ):
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.25,
                split_seed="lfm2-output-interface-rollback-v1",
            ),
        ).run()

    assert adapter.child_state_fingerprint() == state_before
    assert adapter.child_topology_fingerprint() == topology_before
    assert child.get_input_embeddings().weight is parameter
    assert child.get_output_embeddings().weight is parameter
    assert id(parameter) == parameter_id
    assert torch.equal(parameter, value_before)


def test_output_interface_identity_skip_is_exact_with_zero_delta_cap(
    sealed_parent,
) -> None:
    adapter, child, _mapping = _adapter(
        sealed_parent,
        output_interface_mode="identity-skip",
        output_max_prior_delta_ratio=0.0,
    )
    parameter = child.get_input_embeddings().weight
    assert child.get_output_embeddings().weight is parameter
    parameter_id = id(parameter)
    value_before = parameter.detach().clone()

    report = ChildClosedLoopCompiler(
        adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.25,
            split_seed="lfm2-output-interface-identity-skip-v1",
        ),
    ).run()

    output = report["output_interface"]
    assert output["compiled"]["metadata"]["identity_skip"]
    assert output["install"]["metadata"]["parameter_write_calls"] == 0
    assert (
        output["install"]["state_sha256_before"]
        == output["install"]["state_sha256_after"]
    )
    assert output["validation"]["metrics"]["identity_replay_exact"]
    assert child.get_input_embeddings().weight is parameter
    assert child.get_output_embeddings().weight is parameter
    assert id(parameter) == parameter_id
    assert torch.equal(parameter, value_before)


def test_output_interface_required_improvement_rejects_live_candidate(
    sealed_parent,
) -> None:
    adapter, child, _mapping = _adapter(
        sealed_parent,
        minimum_output_improvement=1.0,
        output_validation_tolerance=1e-8,
    )
    state_before = adapter.child_state_fingerprint()
    topology_before = adapter.child_topology_fingerprint()
    parameter = child.get_input_embeddings().weight
    parameter_id = id(parameter)
    value_before = parameter.detach().clone()

    with pytest.raises(
        ClosedLoopFoldError,
        match=(
            "full_vocabulary_parent_kl,"
            "known_sequence_token_nll"
        ),
    ):
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.25,
                split_seed="lfm2-output-interface-default-gates-v1",
            ),
        ).run()

    assert adapter.child_state_fingerprint() == state_before
    assert adapter.child_topology_fingerprint() == topology_before
    assert child.get_input_embeddings().weight is parameter
    assert child.get_output_embeddings().weight is parameter
    assert id(parameter) == parameter_id
    assert torch.equal(parameter, value_before)


def test_performance_diagnostic_only_accepts_finite_layer_final_and_output(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(
        sealed_parent,
        minimum_improvement=1_000_000_000.0,
        minimum_output_improvement=1_000_000_000.0,
        output_validation_tolerance=0.0,
        performance_metrics_diagnostic_only=True,
    )

    report = ChildClosedLoopCompiler(
        adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.25,
            split_seed="lfm2-performance-diagnostic-only-v1",
        ),
    ).run()

    decisions = [
        layer["validation"] for layer in report["layer_reports"]
    ]
    decisions.append(report["final_normalization"]["validation"])
    decisions.append(report["output_interface"]["validation"])
    for decision in decisions:
        metrics = decision["metrics"]
        assert decision["accepted"]
        assert metrics["performance_metrics_diagnostic_only"]
        assert metrics["performance_metrics_are_decision_authority"] is False
        assert metrics["performance_metrics_finite"]
        assert metrics["counterfactual_performance_threshold_passed"] is False
    output_metrics = report["output_interface"]["validation"]["metrics"]
    assert output_metrics["failed_gate_count"] >= 2
    assert output_metrics[
        "kl_known_sequence_nll_and_eos_are_terminal_acceptance_gates"
    ] is False


@pytest.mark.parametrize(
    ("scope", "endpoint"),
    (
        (FoldScope("layer", 0), "layer-output"),
        (FoldScope("final_normalization"), "final-normalization"),
    ),
)
def test_performance_diagnostic_only_never_accepts_nonfinite_metrics(
    scope,
    endpoint,
) -> None:
    adapter = object.__new__(Lfm2DynamicClosedLoopAdapter)
    adapter.config = Lfm2DynamicClosedLoopAdapterConfig(
        performance_metrics_diagnostic_only=True
    )
    baseline = FitMetrics(
        rows=1,
        output_width=1,
        rmse=1.0,
        target_rms=1.0,
        relative_rmse=1.0,
        mean_cosine_similarity=1.0,
        finite=True,
    )
    candidate = FitMetrics(
        rows=1,
        output_width=1,
        rmse=math.nan,
        target_rms=1.0,
        relative_rmse=math.nan,
        mean_cosine_similarity=math.nan,
        finite=False,
    )
    record = TrajectoryRecord(
        trajectory_id="trajectory",
        path_id="path",
        record_sha256="a" * 64,
    )

    decision = adapter._validation_decision(
        scope=scope,
        records=(record,),
        baseline=baseline,
        candidate=candidate,
        minimum_improvement=0.0,
        endpoint=endpoint,
    )
    metrics = dict(decision.metrics)

    assert decision.accepted is False
    assert "nonfinite" in decision.reason
    assert metrics["performance_metrics_diagnostic_only"]
    assert metrics["performance_metrics_finite"] is False
    assert metrics["counterfactual_performance_threshold_passed"] is False
    assert metrics["candidate_relative_rmse"] is None
    assert metrics["candidate_mean_cosine_similarity"] is None
    assert metrics["relative_rmse_improvement"] is None


def test_output_performance_diagnostic_only_rejects_nonfinite_json_safely(
    sealed_parent,
    monkeypatch,
) -> None:
    adapter, child, _mapping = _adapter(
        sealed_parent,
        performance_metrics_diagnostic_only=True,
    )
    original_output_metrics = adapter._output_metrics
    calls = 0

    def scripted_output_metrics(_self, *args, **kwargs):
        nonlocal calls
        calls += 1
        metrics = dict(original_output_metrics(*args, **kwargs))
        if calls == 3:
            metrics["full_vocabulary_parent_kl"] = math.nan
        return metrics

    monkeypatch.setattr(
        adapter,
        "_output_metrics",
        MethodType(scripted_output_metrics, adapter),
    )
    state_before = adapter.child_state_fingerprint()
    topology_before = adapter.child_topology_fingerprint()

    with pytest.raises(ClosedLoopFoldError, match="nonfinite metrics"):
        ChildClosedLoopCompiler(
            adapter,
            ClosedLoopCompilerConfig(
                validation_fraction=0.25,
                split_seed="lfm2-output-nonfinite-diagnostic-only-v1",
            ),
        ).run()

    assert calls == 3
    assert adapter.child_state_fingerprint() == state_before
    assert adapter.child_topology_fingerprint() == topology_before


def test_adapter_config_rejects_unknown_replay_mode() -> None:
    with pytest.raises(ValueError, match="replay_capture_mode"):
        Lfm2DynamicClosedLoopAdapterConfig(
            replay_capture_mode="not-a-replay-mode"
        )


def test_adapter_config_rejects_unknown_opt_in_modes() -> None:
    defaults = Lfm2DynamicClosedLoopAdapterConfig()
    assert defaults.attention_score_candidate_gate_enabled is False
    assert defaults.attention_score_top_k == 8
    with pytest.raises(ValueError, match="attention_consumer_mode"):
        Lfm2DynamicClosedLoopAdapterConfig(
            attention_consumer_mode="not-an-attention-mode"
        )
    with pytest.raises(ValueError, match="output_interface_mode"):
        Lfm2DynamicClosedLoopAdapterConfig(
            output_interface_mode="not-an-output-mode"
        )
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="attention_score_top_k must be at least two",
    ):
        Lfm2DynamicClosedLoopAdapterConfig(attention_score_top_k=1)
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="must be within",
    ):
        Lfm2DynamicClosedLoopAdapterConfig(
            attention_score_minimum_top1_key_agreement_fraction=1.01
        )


# ---------------------------------------------------------------------------
# Multi-device child-replay sharding
# ---------------------------------------------------------------------------


def _capture_one_point(adapter, *, scope, point, selective=False):
    """Capture one point and return its seals plus the payload identity."""

    audit = ClosedLoopAudit()
    records = tuple(adapter.parent_trajectory_records(audit))
    capture = adapter.replay_and_capture_child(scope, point, records, audit)
    try:
        payload = adapter._capture_payload(capture)
        seals = {
            "capture_sha256": capture.capture_sha256,
            "replay_sha256": payload.replay_sha256,
            "selection_sha256": payload.selection_sha256,
            "child_evidence_profile_role": (
                payload.child_evidence_profile_role
            ),
            "child_evidence_profile_sha256": (
                payload.child_evidence_profile_sha256
            ),
            "pairing_sha256": payload.paired.pairing_sha256,
            "actual_child_forwards": payload.actual_child_forwards,
        }
    finally:
        adapter._release_capture_lease(capture)
        if selective:
            adapter._release_idle_selected_parent_profile()
    return seals


def test_two_cpu_shard_workers_seal_the_adapter_capture_bytes(
    sealed_parent,
) -> None:
    scope = FoldScope("layer", 0)
    point = CapturePoint.PRE_PRODUCER
    serial_adapter, _serial_child, mapping = _adapter(sealed_parent)
    serial = _capture_one_point(serial_adapter, scope=scope, point=point)

    events = []
    sharded_adapter, _sharded_child, _mapping = _adapter(
        sealed_parent,
        existing_mapping=mapping,
        replay_devices=("cpu", "cpu"),
        progress_sink=lambda stage, metadata: events.append(
            (stage, dict(metadata))
        ),
    )
    assert sharded_adapter.config.replay_devices == ("cpu", "cpu")
    sharded = _capture_one_point(sharded_adapter, scope=scope, point=point)

    # A shard plan changes the wall clock, never the seal: the envelope, the
    # replay, the paired evidence, and the forward accounting are identical.
    assert sharded == serial

    resolved_events = [
        metadata
        for stage, metadata in events
        if stage == adapter_module.LFM2_REPLAY_SHARDING_RESOLVED_STAGE
    ]
    completed_events = [
        metadata
        for stage, metadata in events
        if stage == adapter_module.LFM2_REPLAY_SHARDING_COMPLETED_STAGE
    ]
    assert len(resolved_events) == 1
    assert len(completed_events) == 1
    assert resolved_events[0]["replay_sharding_enabled"] is True
    assert resolved_events[0]["replay_sharding_requested_devices"] == "cpu,cpu"
    assert resolved_events[0]["replay_sharding_resolved_devices"] == "cpu,cpu"
    assert resolved_events[0]["replay_sharding_worker_count"] == 2
    assert resolved_events[0]["replay_sharding_fallback_reason"] is None
    assert resolved_events[0]["replay_sharding_batch_count"] == 4
    assert completed_events[0]["replay_sharding_worker_count"] == 2
    assert completed_events[0]["replay_sharding_devices"] == "cpu,cpu"
    assert completed_events[0]["replay_sharding_materialized_replica_count"] == 1
    assert completed_events[0]["replay_sharding_worker_replay_seconds_total"] > 0.0
    assert completed_events[0]["replay_sharding_observed_speedup"] > 0.0
    # Every emitted field stays a JSON scalar so the run journal accepts it.
    for _stage, metadata in events:
        assert all(
            value is None or isinstance(value, (bool, int, float, str))
            for value in metadata.values()
        )


def test_sharded_adapter_capture_matches_serial_under_a_selective_session(
    sealed_parent,
) -> None:
    _parent, cache, _replay_config = sealed_parent
    scope = FoldScope("layer", 1)
    point = CapturePoint.POST_DOWNSTREAM
    with cache.selective_authenticated_session() as serial_session:
        serial_adapter, _child, mapping = _adapter(
            sealed_parent,
            selective_session=serial_session,
        )
        serial = _capture_one_point(
            serial_adapter,
            scope=scope,
            point=point,
            selective=True,
        )
    with cache.selective_authenticated_session() as sharded_session:
        sharded_adapter, _sharded_child, _mapping = _adapter(
            sealed_parent,
            existing_mapping=mapping,
            selective_session=sharded_session,
            replay_devices=("cpu", "cpu"),
        )
        sharded = _capture_one_point(
            sharded_adapter,
            scope=scope,
            point=point,
            selective=True,
        )

    # The selected-parent evidence profile and the scope-local child evidence
    # profile are both unchanged by sharding: only the replay is distributed.
    assert sharded == serial
    assert serial["child_evidence_profile_role"] == "layer.post_downstream"


@pytest.mark.parametrize(
    ("devices", "reason"),
    [
        (("cpu", "cuda:0"), "mixed_device_types"),
        (("cuda:0", "cuda:1"), "mixed_device_types"),
        (("cpu",), "single_worker_after_resolution"),
        (("all",), "all_devices_requires_cuda_primary"),
    ],
)
def test_unsafe_replay_device_requests_fall_back_and_record_the_reason(
    sealed_parent,
    devices,
    reason,
) -> None:
    scope = FoldScope("layer", 0)
    point = CapturePoint.PRE_PRODUCER
    serial_adapter, _serial_child, mapping = _adapter(sealed_parent)
    serial = _capture_one_point(serial_adapter, scope=scope, point=point)

    events = []
    adapter, _child, _mapping = _adapter(
        sealed_parent,
        existing_mapping=mapping,
        replay_devices=devices,
        progress_sink=lambda stage, metadata: events.append(
            (stage, dict(metadata))
        ),
    )
    observed = _capture_one_point(adapter, scope=scope, point=point)

    # An unusable request degrades the run's speed, never its result.
    assert observed == serial
    resolved_events = [
        metadata
        for stage, metadata in events
        if stage == adapter_module.LFM2_REPLAY_SHARDING_RESOLVED_STAGE
    ]
    assert len(resolved_events) == 1
    assert resolved_events[0]["replay_sharding_enabled"] is False
    assert resolved_events[0]["replay_sharding_fallback_reason"] == reason
    assert resolved_events[0]["replay_sharding_worker_count"] == 1
    assert resolved_events[0]["replay_sharding_resolved_devices"] is None
    assert not [
        metadata
        for stage, metadata in events
        if stage == adapter_module.LFM2_REPLAY_SHARDING_COMPLETED_STAGE
    ]


def test_an_instrumented_child_is_never_replicated(sealed_parent) -> None:
    events = []
    adapter, child, _mapping = _adapter(
        sealed_parent,
        replay_devices=("cpu", "cpu"),
        progress_sink=lambda stage, metadata: events.append(
            (stage, dict(metadata))
        ),
    )
    handle = child.register_forward_hook(lambda *arguments: None)
    try:
        _capture_one_point(
            adapter,
            scope=FoldScope("layer", 0),
            point=CapturePoint.PRE_PRODUCER,
        )
    finally:
        handle.remove()

    # A replica is a deep copy, so a caller-installed hook would be duplicated
    # onto it while its closure still points at the primary (NOTES-2B rule 7).
    resolved_events = [
        metadata
        for stage, metadata in events
        if stage == adapter_module.LFM2_REPLAY_SHARDING_RESOLVED_STAGE
    ]
    assert len(resolved_events) == 1
    assert resolved_events[0]["replay_sharding_enabled"] is False
    assert resolved_events[0]["replay_sharding_fallback_reason"] == (
        "child_carries_hooks"
    )


def test_sharded_captures_reproduce_the_serial_closed_loop_fold(
    sealed_parent,
) -> None:
    compiler_config = ClosedLoopCompilerConfig(
        validation_fraction=0.25,
        split_seed="lfm2-replay-shard-fold-v1",
    )
    serial_adapter, serial_child, mapping = _adapter(sealed_parent)
    serial = ChildClosedLoopCompiler(serial_adapter, compiler_config).run()

    events = []
    sharded_adapter, sharded_child, _mapping = _adapter(
        sealed_parent,
        existing_mapping=mapping,
        replay_devices=("cpu", "cpu"),
        progress_sink=lambda stage, metadata: events.append(
            (stage, dict(metadata))
        ),
    )
    sharded = ChildClosedLoopCompiler(sharded_adapter, compiler_config).run()

    # Process-local digests mix Parameter object identity, so the committed
    # children are compared through the portable capsule digests.
    assert portable_child_state_sha256(
        sharded_child
    ) == portable_child_state_sha256(serial_child)
    assert portable_child_topology_sha256(
        sharded_child
    ) == portable_child_topology_sha256(serial_child)
    assert sharded["audit"] == serial["audit"]

    for serial_layer, sharded_layer in zip(
        serial["layer_reports"], sharded["layer_reports"], strict=True
    ):
        assert set(sharded_layer["captures"]) == set(serial_layer["captures"])
        for name, record in sharded_layer["captures"].items():
            # Every capture point, not only the one a focused test drives,
            # seals the same envelope under a shard plan.
            assert record["capture_sha256"] == (
                serial_layer["captures"][name]["capture_sha256"]
            )
        assert sharded_layer["validation"]["decision_sha256"] == (
            serial_layer["validation"]["decision_sha256"]
        )
    for section in ("final_normalization", "output_interface"):
        assert json.dumps(
            sharded[section]["captures"], sort_keys=True
        ) == json.dumps(serial[section]["captures"], sort_keys=True)

    resolved_events = [
        metadata
        for stage, metadata in events
        if stage == adapter_module.LFM2_REPLAY_SHARDING_RESOLVED_STAGE
    ]
    # One resolution per capture point, and every one of them sharded.
    assert len(resolved_events) == serial["audit"]["child_replay_capture_calls"]
    assert all(
        metadata["replay_sharding_enabled"] is True
        for metadata in resolved_events
    )


def test_replay_device_requests_are_normalized_or_rejected() -> None:
    assert Lfm2DynamicClosedLoopAdapterConfig().replay_devices is None
    assert Lfm2DynamicClosedLoopAdapterConfig(
        replay_devices=[" cuda:0 ", "cuda:1"]
    ).replay_devices == ("cuda:0", "cuda:1")
    # An empty request is the serial default, not an error.
    assert Lfm2DynamicClosedLoopAdapterConfig(replay_devices=()).replay_devices is None
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="sequence of device strings",
    ):
        Lfm2DynamicClosedLoopAdapterConfig(replay_devices="cuda:0,cuda:1")
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="nonempty device strings",
    ):
        Lfm2DynamicClosedLoopAdapterConfig(replay_devices=("cuda:0", " "))


# ---------------------------------------------------------------------------
# Merged same-state layer captures
# ---------------------------------------------------------------------------


def test_merged_layer_captures_reproduce_the_unmerged_fold(
    sealed_parent,
) -> None:
    baseline_adapter, baseline_child, _baseline_mapping = _adapter(sealed_parent)
    baseline = ChildClosedLoopCompiler(
        baseline_adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.25,
            split_seed="lfm2-merged-capture-v1",
        ),
    ).run()

    merged_adapter, merged_child, _merged_mapping = _adapter(sealed_parent)
    assert merged_adapter.supports_merged_child_captures() is True
    merged = ChildClosedLoopCompiler(
        merged_adapter,
        ClosedLoopCompilerConfig(
            validation_fraction=0.25,
            split_seed="lfm2-merged-capture-v1",
            merge_adjacent_layer_captures=True,
        ),
    ).run()

    assert baseline["merge_adjacent_layer_captures_enabled"] is False
    assert merged["merge_adjacent_layer_captures_enabled"] is True
    # Two layers share one boundary, so exactly one full replay disappears.
    assert baseline["audit"]["child_replay_capture_calls"] == 13
    assert merged["audit"]["child_replay_capture_calls"] == 12

    # Process-local digests mix Parameter object identity, so the committed
    # children are compared through the portable capsule digests.
    assert portable_child_state_sha256(
        merged_child
    ) == portable_child_state_sha256(baseline_child)
    assert portable_child_topology_sha256(
        merged_child
    ) == portable_child_topology_sha256(baseline_child)

    merge_reports = [
        value["merged_captures"] for value in merged["layer_reports"]
    ]
    assert [
        value["post_downstream_served_next_layer_pre_producer"]
        for value in merge_reports
    ] == [True, False]
    assert [
        value["pre_producer_served_by_merged_replay"] for value in merge_reports
    ] == [False, True]

    for baseline_layer, merged_layer in zip(
        baseline["layer_reports"], merged["layer_reports"], strict=True
    ):
        assert set(baseline_layer["captures"]) == set(merged_layer["captures"])
        for name, record in merged_layer["captures"].items():
            reference = baseline_layer["captures"][name]
            # Sealed envelope semantics are schema-compatible per capture point.
            assert set(record) == set(reference)
            assert record["scope"] == reference["scope"]
            assert record["point"] == reference["point"]
            assert record["record_keys"] == reference["record_keys"]
        seals = {
            record["capture_sha256"]
            for record in merged_layer["captures"].values()
        }
        assert len(seals) == len(merged_layer["captures"])
        assert baseline_layer["validation"]["accepted"] is True
        assert merged_layer["validation"]["accepted"] is True

    # The merged boundary retains the union of both points' evidence profiles.
    boundary = merged["layer_reports"][1]["captures"]["pre_producer_fit"]
    assert boundary["scope"]["layer_index"] == 1
    assert merged["audit"]["parent_forward_calls"] == 0
    assert merged["audit"]["training_steps"] == 0
    assert merged["audit"]["backward_calls"] == 0
    assert merged["audit"]["optimizer_steps"] == 0


def test_merged_capture_envelope_matches_the_payload_authenticator(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    audit = ClosedLoopAudit()
    records = tuple(adapter.parent_trajectory_records(audit))
    split = deterministic_whole_trajectory_split(
        records,
        validation_fraction=0.25,
        split_seed="lfm2-merged-envelope-v1",
    )
    post_downstream, pre_producer = adapter.replay_and_capture_child_merged(
        (
            (FoldScope("layer", 0), CapturePoint.POST_DOWNSTREAM, split.validation),
            (FoldScope("layer", 1), CapturePoint.PRE_PRODUCER, split.fit),
        ),
        audit,
    )
    try:
        primary = adapter._capture_payload(post_downstream)
        secondary = adapter._capture_payload(pre_producer)
        # One replay, one paired replay, two independently sealed envelopes.
        assert primary is secondary
        assert primary.selected_layer_indices == (0, 1)
        assert primary.paired.selected_layer_indices == (0, 1)
        assert post_downstream.capture_sha256 != pre_producer.capture_sha256
        # Drift guard: the shared envelope helper reproduces the seal that
        # `_replay_and_capture_child_with_parent_profile` built inline.
        assert adapter_module._capture_envelope_sha256(
            scope=FoldScope("layer", 0),
            point=CapturePoint.POST_DOWNSTREAM,
            record_keys=post_downstream.record_keys,
            payload=primary,
        ) == post_downstream.capture_sha256

        profile = primary.child_evidence_profile
        assert profile is not None
        assert profile.selected_layer_indices == (0, 1)
        expected = {}
        for layer_index, point in (
            (0, CapturePoint.POST_DOWNSTREAM),
            (1, CapturePoint.PRE_PRODUCER),
        ):
            single = adapter._child_evidence_profile(
                FoldScope("layer", layer_index),
                point,
                (layer_index,),
                body_internal_role=None,
            )
            expected[layer_index] = single.suffixes_for_layer(layer_index)
        assert profile.suffixes_for_layer(0) == expected[0]
        assert profile.suffixes_for_layer(1) == expected[1]
    finally:
        adapter._release_capture_lease(post_downstream)
        adapter._release_capture_lease(pre_producer)


def test_merged_captures_are_declined_for_single_pass_causal_replay(
    sealed_parent,
) -> None:
    # Single-pass causal replay certifies one exact selection against a
    # canonical reference replay, so a union selection would buy a second
    # reference replay instead of saving one.  That decline stays.
    adapter, _child, _mapping = _adapter(
        sealed_parent,
        replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    )
    assert adapter.supports_merged_child_captures() is False
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="cannot merge child captures",
    ):
        adapter.replay_and_capture_child_merged((), ClosedLoopAudit())


def test_merged_captures_take_a_two_role_selected_parent_profile(
    sealed_parent,
) -> None:
    _parent, cache, _replay_config = sealed_parent
    with cache.selective_authenticated_session() as session:
        adapter, _child, _mapping = _adapter(
            sealed_parent,
            selective_session=session,
        )
        assert adapter.supports_merged_child_captures() is True
        audit = ClosedLoopAudit()
        records = tuple(adapter.parent_trajectory_records(audit))
        split = deterministic_whole_trajectory_split(
            records,
            validation_fraction=0.25,
            split_seed="lfm2-merged-selective-profile-v1",
        )
        post_downstream, pre_producer = (
            adapter.replay_and_capture_child_merged(
                (
                    (
                        FoldScope("layer", 0),
                        CapturePoint.POST_DOWNSTREAM,
                        split.validation,
                    ),
                    (
                        FoldScope("layer", 1),
                        CapturePoint.PRE_PRODUCER,
                        split.fit,
                    ),
                ),
                audit,
            )
        )
        try:
            profile = adapter._active_selected_parent_profile
            assert profile is not None
            # Role plus capture point still names exactly one addressed layer.
            assert profile.profile_role == (
                "layer_merge.000.post_downstream+001.pre_producer"
            )
            assert profile.selected_layer_indices == (0, 1)
            assert profile.allowed_capture_points == (
                CapturePoint.POST_DOWNSTREAM.value,
                CapturePoint.PRE_PRODUCER.value,
            )
            # Two consumers, two counted borrows over one materialization.
            assert profile.borrow_count == 2
            assert len(adapter._deferred_merged_parent_borrows) == 1
            assert (
                adapter._deferred_merged_parent_borrows[0].capture_point
                == CapturePoint.PRE_PRODUCER.value
            )

            # The union inventory is exactly the union of the two exact layer
            # inventories the unmerged captures would have addressed.
            merged_names = set(profile.ordered_tensor_names_by_batch[0])
            unmerged = {}
            for layer_index in (0, 1):
                role, _points = adapter._selected_parent_profile_role(
                    FoldScope("layer", layer_index),
                    CapturePoint.PRE_PRODUCER,
                    (layer_index,),
                    body_internal=False,
                )
                unmerged[layer_index] = set(
                    adapter._selective_parent_tensor_names(
                        session.batch_references[0],
                        profile_role=role,
                        selected_layer_indices=(layer_index,),
                    )
                )
            assert merged_names == unmerged[0] | unmerged[1]

            primary = adapter._capture_payload(post_downstream)
            secondary = adapter._capture_payload(pre_producer)
            assert primary is secondary
            assert primary.selected_layer_indices == (0, 1)
            assert post_downstream.capture_sha256 != pre_producer.capture_sha256

            # Releasing the layer-local consumer leaves the deferred borrow,
            # and the layer-terminal idle release tolerates exactly that.
            adapter._release_capture_lease(post_downstream)
            adapter._release_idle_selected_parent_profile()
            assert adapter._active_selected_parent_profile is profile
            assert profile.borrow_count == 1
        finally:
            adapter._release_capture_lease(post_downstream)
            adapter._release_capture_lease(pre_producer)
        adapter._release_idle_selected_parent_profile()
        assert adapter._active_selected_parent_profile is None
        assert adapter._deferred_merged_parent_borrows == []


def test_a_leaked_capture_borrow_still_fails_the_terminal_idle_release(
    sealed_parent,
) -> None:
    _parent, cache, _replay_config = sealed_parent
    with cache.selective_authenticated_session() as session:
        adapter, _child, _mapping = _adapter(
            sealed_parent,
            selective_session=session,
        )
        audit = ClosedLoopAudit()
        records = tuple(adapter.parent_trajectory_records(audit))
        capture = adapter.replay_and_capture_child(
            FoldScope("layer", 0),
            CapturePoint.PRE_PRODUCER,
            records,
            audit,
        )
        try:
            # An ordinary capture that was never released is still a leak.
            with pytest.raises(
                Lfm2DynamicClosedLoopAdapterError,
                match="still has live borrows",
            ):
                adapter._release_idle_selected_parent_profile()
        finally:
            adapter._release_capture_lease(capture)
        adapter._release_idle_selected_parent_profile()
        assert adapter._active_selected_parent_profile is None


def test_merged_selective_captures_reproduce_the_unmerged_selective_fold(
    sealed_parent,
) -> None:
    _parent, cache, _replay_config = sealed_parent
    compiler_config = ClosedLoopCompilerConfig(
        validation_fraction=0.25,
        split_seed="lfm2-merged-selective-fold-v1",
    )
    with cache.selective_authenticated_session() as baseline_session:
        baseline_adapter, baseline_child, mapping = _adapter(
            sealed_parent,
            selective_session=baseline_session,
        )
        baseline = ChildClosedLoopCompiler(
            baseline_adapter,
            compiler_config,
        ).run()
        baseline_adapter.release_transient_fold_evidence()

    with cache.selective_authenticated_session() as merged_session:
        merged_adapter, merged_child, _mapping = _adapter(
            sealed_parent,
            existing_mapping=mapping,
            selective_session=merged_session,
        )
        assert merged_adapter.supports_merged_child_captures() is True
        merged = ChildClosedLoopCompiler(
            merged_adapter,
            replace(compiler_config, merge_adjacent_layer_captures=True),
        ).run()
        merged_adapter.release_transient_fold_evidence()

    assert baseline["merge_adjacent_layer_captures_enabled"] is False
    assert merged["merge_adjacent_layer_captures_enabled"] is True
    # Two layers share one boundary, so exactly one full replay disappears.
    assert baseline["audit"]["child_replay_capture_calls"] == 13
    assert merged["audit"]["child_replay_capture_calls"] == 12

    # Process-local digests mix Parameter object identity, so the committed
    # children are compared through the portable capsule digests.
    assert portable_child_state_sha256(
        merged_child
    ) == portable_child_state_sha256(baseline_child)
    assert portable_child_topology_sha256(
        merged_child
    ) == portable_child_topology_sha256(baseline_child)

    merge_reports = [
        value["merged_captures"] for value in merged["layer_reports"]
    ]
    assert [
        value["post_downstream_served_next_layer_pre_producer"]
        for value in merge_reports
    ] == [True, False]
    assert [
        value["pre_producer_served_by_merged_replay"] for value in merge_reports
    ] == [False, True]

    for baseline_layer, merged_layer in zip(
        baseline["layer_reports"], merged["layer_reports"], strict=True
    ):
        assert set(baseline_layer["captures"]) == set(merged_layer["captures"])
        seals = {
            record["capture_sha256"]
            for record in merged_layer["captures"].values()
        }
        assert len(seals) == len(merged_layer["captures"])
        assert baseline_layer["validation"]["accepted"] is True
        assert merged_layer["validation"]["accepted"] is True
        assert merged_layer["validation"]["decision_sha256"] == (
            baseline_layer["validation"]["decision_sha256"]
        )
    assert merged["audit"]["parent_forward_calls"] == 0
    assert merged["audit"]["parent_trace_regeneration_calls"] == 0
    assert merged["audit"]["training_steps"] == 0
    assert merged["audit"]["backward_calls"] == 0
    assert merged["audit"]["optimizer_steps"] == 0


def test_merged_capture_requires_two_distinct_adjacent_points(
    sealed_parent,
) -> None:
    adapter, _child, _mapping = _adapter(sealed_parent)
    audit = ClosedLoopAudit()
    records = tuple(adapter.parent_trajectory_records(audit))
    split = deterministic_whole_trajectory_split(
        records,
        validation_fraction=0.25,
        split_seed="lfm2-merged-guard-v1",
    )
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="post-downstream point",
    ):
        adapter.replay_and_capture_child_merged(
            (
                (FoldScope("layer", 0), CapturePoint.PRE_PRODUCER, split.fit),
                (
                    FoldScope("layer", 1),
                    CapturePoint.POST_DOWNSTREAM,
                    split.validation,
                ),
            ),
            audit,
        )
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="two distinct selected",
    ):
        adapter.replay_and_capture_child_merged(
            (
                (
                    FoldScope("layer", 0),
                    CapturePoint.POST_DOWNSTREAM,
                    split.validation,
                ),
                (FoldScope("layer", 0), CapturePoint.PRE_PRODUCER, split.fit),
            ),
            audit,
        )


# ---------------------------------------------------------------------------
# Accepted-layer restart capsules
# ---------------------------------------------------------------------------


def test_restart_capsule_identity_bindings_are_sealed(sealed_parent) -> None:
    adapter, _child, mapping = _adapter(sealed_parent)
    identity = adapter.restart_capsule_identity_bindings()

    assert identity["coordinate_map_sha256"] == mapping.map_sha256
    assert identity["child_basis_sha256"] == mapping.map_sha256
    assert identity["parent_cache_key"] == adapter.parent_cache.cache_key
    assert identity["child_layer_count"] == len(adapter.child_layers)
    assert identity["selected_layer_indices"] == [0, 1]
    body = {
        name: value
        for name, value in identity.items()
        if name != "adapter_identity_sha256"
    }
    assert identity["adapter_identity_sha256"] == canonical_sha256(body)
    # The capsule module authenticates the same seal it will pin.
    assert dict(
        lfm2_restart_capsule_adapter_identity(adapter)
    ) == dict(identity)


def test_mid_body_resume_is_refused_when_body_reclosure_needs_the_stack(
    sealed_parent,
) -> None:
    parent, _cache, _replay_config = sealed_parent
    basis = build_lfm2_weight_complete_basis(parent, 4)
    adapter, _child, _mapping = _adapter(
        sealed_parent,
        existing_mapping=basis,
        body_reclosure_mode=LFM2_BODY_RECLOSURE_MOVING_GAUGE_V3,
        parent_residual_operators=basis.inventory.operators,
        output_interface_mode="identity-skip",
    )
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="mid-body restart resume cannot satisfy",
    ):
        adapter.assert_mid_body_resume_supported(
            completed_layer_indices=(0,),
            remaining_layer_indices=(1,),
            body_reclosure_enabled=True,
        )
    admission = adapter.assert_mid_body_resume_supported(
        completed_layer_indices=(0,),
        remaining_layer_indices=(1,),
        body_reclosure_enabled=False,
    )
    assert admission["body_reclosure_requires_full_installed_stack"] is True
    assert admission["skipped_layers_are_not_replayed"] is True
    body = {
        name: value
        for name, value in admission.items()
        if name != "admission_sha256"
    }
    assert admission["admission_sha256"] == canonical_sha256(body)

    disabled_adapter, _c, _m = _adapter(sealed_parent)
    enabled = disabled_adapter.assert_mid_body_resume_supported(
        completed_layer_indices=(0,),
        remaining_layer_indices=(1,),
        body_reclosure_enabled=True,
    )
    assert enabled["body_reclosure_requires_full_installed_stack"] is False


def test_restart_capsule_resume_reproduces_the_uninterrupted_fold(
    sealed_parent,
    tmp_path,
) -> None:
    config = ClosedLoopCompilerConfig(
        validation_fraction=0.25,
        split_seed="lfm2-restart-capsule-v1",
    )
    capsule_root = tmp_path / "restart-capsules"
    capsule_root.mkdir()

    cold_adapter, cold_child, mapping = _adapter(sealed_parent)
    split_sha256 = lfm2_restart_capsule_split_sha256(cold_adapter, config)
    cold_writer = Lfm2ClosedLoopRestartCapsuleWriter(
        adapter=cold_adapter,
        directory=capsule_root,
        run_id="lfm2-restart-capsule-run",
        child_name="lfm2-restart-capsule-child",
        trajectory_split_sha256=split_sha256,
        run_schema=RUN_SCHEMA,
    )
    cold = ChildClosedLoopCompiler(
        cold_adapter,
        config,
        restart_capsule_config=cold_writer,
    ).run()

    assert cold["restart_capsule_enabled"] is True
    assert cold["restart_capsule"]["capsules_written"] == 2
    assert cold["restart_capsule_resumed_layers"] == []
    assert cold["layers_executed_in_this_process"] == [0, 1]
    cold_state = portable_child_state_sha256(cold_child)

    # Take the capsule that layer 0 sealed, exactly as a killed run would.
    references = discover_lfm2_closed_loop_restart_capsules(capsule_root)
    assert [value.layer_index for value in references] == [0, 1]
    layer_zero = references[0]

    resumed_child = _model(hidden=4, intermediate=6, heads=1, seed=1802)
    resume = load_lfm2_closed_loop_restart_capsule(
        layer_zero,
        child_model=resumed_child,
    )
    assert resume.completed_layer_indices == (0,)
    assert resume.resume_layer_index == 1

    resumed_adapter, _resumed_child, _mapping = _adapter(
        sealed_parent,
        existing_child=resumed_child,
        existing_mapping=mapping,
    )
    resumed_root = tmp_path / "resumed-capsules"
    resumed_root.mkdir()
    resumed_writer = Lfm2ClosedLoopRestartCapsuleWriter(
        adapter=resumed_adapter,
        directory=resumed_root,
        run_id="lfm2-restart-capsule-run",
        child_name="lfm2-restart-capsule-child",
        trajectory_split_sha256=split_sha256,
        resume=resume,
        run_schema=RUN_SCHEMA,
    )
    resumed = ChildClosedLoopCompiler(
        resumed_adapter,
        config,
        restart_capsule_config=resumed_writer,
    ).run()

    assert resumed["restart_capsule_resumed_layers"] == [0]
    assert resumed["layers_executed_in_this_process"] == [1]
    assert resumed["restart_capsule"]["capsules_written"] == 1
    assert resumed["restart_capsule"]["resume_admission"][
        "resume_layer_index"
    ] == 1
    assert resumed["layer_reports"][0]["executed_in_this_process"] is False
    assert resumed["layer_reports"][1]["executed_in_this_process"] is True
    # Layer 0 ran no capture, compile, install, or validation stage.
    assert cold["audit"]["child_replay_capture_calls"] == 13
    assert resumed["audit"]["child_replay_capture_calls"] == 9
    assert resumed["audit"]["producer_compile_calls"] == 1
    assert resumed["audit"]["install_calls"] == 5
    assert resumed["audit"]["parent_forward_calls"] == 0
    assert resumed["audit"]["training_steps"] == 0
    assert resumed["audit"]["backward_calls"] == 0
    assert resumed["audit"]["optimizer_steps"] == 0

    # The resumed child reproduces the child the uninterrupted run committed.
    assert portable_child_state_sha256(resumed_child) == cold_state
    assert portable_child_topology_sha256(
        resumed_child
    ) == portable_child_topology_sha256(cold_child)
