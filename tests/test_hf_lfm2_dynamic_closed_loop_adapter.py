from __future__ import annotations

import hashlib
import math
import weakref
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_dynamic_closed_loop_adapter as adapter_module  # noqa: E402
from neural_forge.dynamic_closed_loop_fold import (  # noqa: E402
    CapturePoint,
    ChildClosedLoopCompiler,
    ClosedLoopAudit,
    ClosedLoopCompilerConfig,
    ClosedLoopFoldError,
    FoldScope,
    TrajectoryRecord,
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
    prior_tensor_ref = None
    prior_tensor_released = []
    initial_body_tensor_ref = None

    def observed_replay_and_capture(self, scope, point, records, audit):
        nonlocal initial_body_tensor_ref
        capture = original_replay_and_capture(scope, point, records, audit)
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
