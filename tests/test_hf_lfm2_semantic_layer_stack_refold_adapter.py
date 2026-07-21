from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter as stack  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (  # noqa: E402
    Lfm2CpcfTerminalActionEvidence,
)
from neural_forge.hf_lfm2_semantic_mlp_refold_adapter import (  # noqa: E402
    Lfm2DirectLocalMlpSourceInventory,
)


def _terminal_banks():
    return {
        split: Lfm2CpcfTerminalActionEvidence(
            split=split,
            row_ids=(f"{split}:row",),
            mlp_input=torch.ones((1, 2), dtype=torch.bfloat16),
            pre_mlp_residual=torch.zeros((1, 2), dtype=torch.bfloat16),
            parent_final_norm_output=torch.ones((1, 2), dtype=torch.bfloat16),
            sample_weights=torch.ones((1,), dtype=torch.float64),
        )
        for split in stack._PARTITIONS
    }


class _TinyTarget(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mutable = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        self.final_w2 = torch.nn.Parameter(torch.tensor([7.0], dtype=torch.float64))
        self._neural_forge_quarantine_tokenizer = object()


def _state_identity(model: _TinyTarget) -> str:
    return canonical_sha256(
        {
            name: stack._tensor_sha256(value)
            for name, value in sorted(model.state_dict().items())
        }
    )


def _patch_complete_transaction(
    monkeypatch,
    *,
    failing_layer: int | None = None,
    noop_operator_layers: frozenset[int] = frozenset(),
    noop_mlp_layers: frozenset[int] = frozenset(),
):
    events: list[str] = []

    class Cache:
        cache_key = "a" * 64
        parent_fingerprint = "b" * 64
        layer_topology_fingerprint = "c" * 64
        batch_count = 4
        manifest = {"manifest_sha256": "d" * 64}

        def selective_authenticated_session(self):
            class Session:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return None

                def verify(self):
                    return self

                def as_dict(self):
                    return {"session_sha256": "e" * 64}

            return Session()

    class Bridge:
        artifact_sha256 = "f" * 64

        def verify_live_tokenizers(self, _donor, _target):
            return self

    class ParentSession:
        manifest_sha256 = "1" * 64
        open_receipt_sha256 = "2" * 64

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def tensor(self, _key):
            return torch.ones((2, 1, 2), dtype=torch.bfloat16)

        def optional_tensor(self, _key):
            return None

        def receipt(self):
            return {"session_sha256": "3" * 64}

    monkeypatch.setattr(stack, "Lfm2CircuitTraceCache", Cache)
    monkeypatch.setattr(stack, "StrictTokenizerBridgeArtifact", Bridge)
    monkeypatch.setattr(
        stack,
        "fingerprint_hf_tokenizer_semantics",
        lambda _value: "4" * 64,
    )
    monkeypatch.setattr(
        stack,
        "fingerprint_lfm2_parent",
        lambda model: SimpleNamespace(identity_sha256=_state_identity(model)),
    )
    monkeypatch.setattr(
        stack,
        "_validated_operator_topology",
        lambda *_args: tuple(
            "full_attention"
            if index in stack._EXPECTED_ATTENTION_INDICES
            else "short_conv"
            for index in range(14)
        ),
    )
    monkeypatch.setattr(
        stack,
        "_validated_interface_lineage",
        lambda *_args: {
            "interface_report_sha256": "5" * 64,
            "producer_encoder_sha256": "6" * 64,
        },
    )
    monkeypatch.setattr(
        stack,
        "_resolve_short_conv_adapter",
        lambda: (object, lambda **_kwargs: None),
    )
    monkeypatch.setattr(
        stack,
        "_resolve_mlp_adapter",
        lambda: (object, object, lambda **_kwargs: None),
    )
    monkeypatch.setattr(stack, "_final_w2_parameter", lambda model: model.final_w2)
    monkeypatch.setattr(
        stack,
        "_target_rope",
        lambda _model: (torch.ones((1,), dtype=torch.float64), 1.0),
    )
    monkeypatch.setattr(stack, "_DirectParentSafetensorsSession", ParentSession)
    alignments = tuple(
        SimpleNamespace(source_batch_index=index, alignment_sha256=str(index) * 64)
        for index in range(4)
    )
    monkeypatch.setattr(
        stack,
        "_semantic_alignments_from_session",
        lambda **_kwargs: alignments,
    )
    monkeypatch.setattr(
        stack,
        "audit_lfm2_semantic_span_cpcf_refinement",
        lambda *_args, **_kwargs: {
            "probability_preserving_attention_transport_go": True,
            "blocking_reasons": [],
            "report_sha256": "7" * 64,
        },
    )
    monkeypatch.setattr(
        stack,
        "_validate_direct_attention_sources",
        lambda *_args: {
            index: {"result_sha256": canonical_sha256(["qkvo", index])}
            for index in stack._EXPECTED_ATTENTION_INDICES
        },
    )
    monkeypatch.setattr(
        stack,
        "_initial_residual_bank",
        lambda *_args: MappingProxyType({0: "initial"}),
    )
    replay_count = {index: 0 for index in range(14)}

    def replay(model, *, layer_index, **_kwargs):
        ordinal = replay_count[layer_index]
        replay_count[layer_index] += 1
        events.append(f"{layer_index}:replay:{ordinal}")
        return SimpleNamespace(
            captured_child_model_state_sha256=_state_identity(model),
            layer_index=layer_index,
            bank_sha256=canonical_sha256(["bank", layer_index, ordinal]),
        )

    monkeypatch.setattr(stack, "_replay_current_layer", replay)

    def load_parent(_session, *, layer_index, **_kwargs):
        events.append(f"{layer_index}:parent")
        return {
            0: SimpleNamespace(
                evidence_sha256=canonical_sha256(["parent", layer_index])
            )
        }

    monkeypatch.setattr(stack, "_load_parent_layer_trajectories", load_parent)

    def chart(*, layer_index, **_kwargs):
        events.append(f"{layer_index}:chart")
        body = {"layer_index": layer_index, "fit_only": True}
        return torch.eye(2, dtype=torch.float64), {
            **body,
            "result_sha256": canonical_sha256(body),
        }

    monkeypatch.setattr(stack, "_fit_residual_chart", chart)

    def evidence(label: str, layer_index: int):
        return (
            SimpleNamespace(evidence_sha256=canonical_sha256([label, layer_index])),
        )

    def build_attention(*, layer_index, **_kwargs):
        events.append(f"{layer_index}:build:full_attention")
        return evidence("attention", layer_index)

    def build_short(*, layer_index, **_kwargs):
        events.append(f"{layer_index}:build:short_conv")
        return evidence("short", layer_index)

    monkeypatch.setattr(stack, "_build_attention_evidence", build_attention)
    monkeypatch.setattr(stack, "_build_short_conv_evidence", build_short)

    def mutate_operator(model, layer_index: int, kind: str):
        before = _state_identity(model)
        events.append(f"{layer_index}:install:{kind}")
        installed = layer_index not in noop_operator_layers
        if installed:
            with torch.no_grad():
                model.mutable.add_(1.0)
        if failing_layer == layer_index:
            raise RuntimeError("injected partial operator failure")
        after = _state_identity(model)
        return {
            "status": (
                "installed-analytic-candidate"
                if installed
                else "heldout-selected-incumbent-noop"
            ),
            "installed": installed,
            "install_count": int(installed),
            "model_state_before_sha256": before,
            "model_state_after_sha256": after,
            "result_sha256": canonical_sha256(["operator", layer_index, after]),
        }

    monkeypatch.setattr(
        stack,
        "_install_attention_operator",
        lambda target_model, layer_index, **_kwargs: mutate_operator(
            target_model, layer_index, "full_attention"
        ),
    )
    monkeypatch.setattr(
        stack,
        "_install_short_conv_operator",
        lambda target_model, layer_index, **_kwargs: mutate_operator(
            target_model, layer_index, "short_conv"
        ),
    )

    def validate_fir(*, layer_index, **_kwargs):
        events.append(f"{layer_index}:fir")
        body = {"layer_index": layer_index}
        return {**body, "result_sha256": canonical_sha256(body)}

    monkeypatch.setattr(stack, "_validate_cached_fir", validate_fir)

    def inventory(*, layer_index, **_kwargs):
        events.append(f"{layer_index}:inventory")
        body = {
            "layer_index": layer_index,
            "direct_parent_tensors_functionally_bound_to_cache": True,
        }
        return object(), {
            **body,
            "result_sha256": canonical_sha256(body),
        }

    monkeypatch.setattr(stack, "_direct_mlp_inventory", inventory)

    def build_mlp(*, layer_index, **_kwargs):
        events.append(f"{layer_index}:build:mlp")
        return evidence("mlp", layer_index)

    monkeypatch.setattr(stack, "_build_mlp_evidence", build_mlp)

    def install_mlp(*, target_model, layer_index, **_kwargs):
        before = _state_identity(target_model)
        events.append(f"{layer_index}:install:mlp")
        installed = layer_index < 13 and layer_index not in noop_mlp_layers
        if installed:
            with torch.no_grad():
                target_model.mutable.add_(1.0)
        after = _state_identity(target_model)
        receipt = {
            "status": (
                "installed-frozen-semantic-mlp-candidate"
                if installed
                else "heldout-selected-incumbent-noop"
            ),
            "installed": installed,
            "install_count": int(installed),
            "model_state_before_sha256": before,
            "model_state_after_sha256": after,
            "result_sha256": canonical_sha256(["mlp", layer_index, after]),
        }
        return receipt, (_terminal_banks() if layer_index == 13 else {})

    monkeypatch.setattr(stack, "_install_mlp", install_mlp)

    def next_bank(bank):
        events.append(f"{bank.layer_index}:next")
        return MappingProxyType({0: f"after:{bank.layer_index}"})

    monkeypatch.setattr(stack, "_next_residual_bank", next_bank)
    target = _TinyTarget()
    return target, Cache(), Bridge(), events


def test_complete_transaction_executes_exact_causal_order(monkeypatch) -> None:
    target, cache, bridge, events = _patch_complete_transaction(monkeypatch)
    before_w2 = target.final_w2.detach().clone()

    result = stack.compile_install_lfm2_semantic_layer_stack_postfold(
        cache=cache,
        bridge=bridge,
        target_model=target,
        interface=SimpleNamespace(producer_encoder=torch.eye(2)),
        compiler_lineage={
            index: object() for index in stack._EXPECTED_ATTENTION_INDICES
        },
        parent_model_directory="unused",
        donor_tokenizer=object(),
        target_tokenizer=object(),
        attention_installer=lambda **_kwargs: None,
    )

    assert result.receipt["complete_causal_layer_stack_adjudicated"] is True
    assert result.receipt["complete_causal_layer_stack_installed"] is False
    assert result.receipt["attention_operator_install_count"] == 6
    assert result.receipt["short_conv_operator_install_count"] == 8
    assert result.receipt["replay_fresh_mlp_evidence_compile_count"] == 14
    assert result.receipt["final_layer_mlp_w2_install_count"] == 0
    torch.testing.assert_close(target.final_w2, before_w2, rtol=0, atol=0)
    for layer_index in range(14):
        kind = (
            "full_attention"
            if layer_index in stack._EXPECTED_ATTENTION_INDICES
            else "short_conv"
        )
        observed = [value for value in events if value.startswith(f"{layer_index}:")]
        expected = [
            f"{layer_index}:replay:0",
            f"{layer_index}:parent",
            f"{layer_index}:chart",
            *([] if kind == "full_attention" else [f"{layer_index}:fir"]),
            f"{layer_index}:build:{kind}",
            f"{layer_index}:install:{kind}",
            f"{layer_index}:replay:1",
            f"{layer_index}:inventory",
            f"{layer_index}:build:mlp",
            f"{layer_index}:install:mlp",
            f"{layer_index}:replay:2",
            f"{layer_index}:next",
        ]
        assert observed == expected


def test_complete_transaction_truthfully_counts_mixed_incumbent_noops(
    monkeypatch,
) -> None:
    target, cache, bridge, _events = _patch_complete_transaction(
        monkeypatch,
        noop_operator_layers=frozenset({0, 2}),
        noop_mlp_layers=frozenset({1}),
    )

    result = stack.compile_install_lfm2_semantic_layer_stack_postfold(
        cache=cache,
        bridge=bridge,
        target_model=target,
        interface=SimpleNamespace(producer_encoder=torch.eye(2)),
        compiler_lineage={
            index: object() for index in stack._EXPECTED_ATTENTION_INDICES
        },
        parent_model_directory="unused",
        donor_tokenizer=object(),
        target_tokenizer=object(),
        attention_installer=lambda **_kwargs: None,
    )

    receipt = result.receipt
    assert receipt["attention_operator_decision_count"] == 6
    assert receipt["attention_operator_install_count"] == 5
    assert receipt["attention_operator_incumbent_noop_count"] == 1
    assert receipt["short_conv_operator_decision_count"] == 8
    assert receipt["short_conv_operator_install_count"] == 7
    assert receipt["short_conv_operator_incumbent_noop_count"] == 1
    assert receipt["nonterminal_mlp_writer_decision_count"] == 13
    assert receipt["nonterminal_mlp_writer_install_count"] == 12
    assert receipt["nonterminal_mlp_writer_incumbent_noop_count"] == 1
    assert receipt["complete_causal_layer_stack_adjudicated"] is True
    assert receipt["complete_causal_layer_stack_installed"] is False
    assert receipt["whole_stage_contains_meaningful_parameter_changes"] is True


def test_complete_transaction_rejects_an_all_incumbent_noop_stack(
    monkeypatch,
) -> None:
    target, cache, bridge, _events = _patch_complete_transaction(
        monkeypatch,
        noop_operator_layers=frozenset(range(14)),
        noop_mlp_layers=frozenset(range(13)),
    )
    before = _state_identity(target)

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="exact entry bytes were restored",
    ):
        stack.compile_install_lfm2_semantic_layer_stack_postfold(
            cache=cache,
            bridge=bridge,
            target_model=target,
            interface=SimpleNamespace(producer_encoder=torch.eye(2)),
            compiler_lineage={
                index: object() for index in stack._EXPECTED_ATTENTION_INDICES
            },
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
            attention_installer=lambda **_kwargs: None,
        )

    assert _state_identity(target) == before


def test_partial_layer_failure_restores_exact_whole_model(monkeypatch) -> None:
    target, cache, bridge, _events = _patch_complete_transaction(
        monkeypatch,
        failing_layer=3,
    )
    before = {
        name: value.detach().clone() for name, value in target.state_dict().items()
    }
    before_identity = _state_identity(target)

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="exact entry bytes were restored",
    ):
        stack.compile_install_lfm2_semantic_layer_stack_postfold(
            cache=cache,
            bridge=bridge,
            target_model=target,
            interface=SimpleNamespace(producer_encoder=torch.eye(2)),
            compiler_lineage={
                index: object() for index in stack._EXPECTED_ATTENTION_INDICES
            },
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
            attention_installer=lambda **_kwargs: None,
        )

    assert _state_identity(target) == before_identity
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def _chart_inputs(*, reject_validation: bool):
    alignments = []
    parent = {}
    child = {}
    for batch in range(8):
        phase = SimpleNamespace(
            target_to_donor_boundary_positions=(0, 1),
        )
        alignments.append(SimpleNamespace(source_batch_index=batch, phases=(phase,)))
        source = torch.tensor(
            [[[batch + 1.0, 0.0, 0.0], [batch + 2.0, 0.0, 0.0]]],
            dtype=torch.float64,
        )
        factor = -2.0 if reject_validation and batch >= 6 else 2.0
        parent[batch] = SimpleNamespace(final_histories={"residual.pre": source})
        child[batch] = source * factor
    bank = SimpleNamespace(
        captured_child_model_state_sha256="8" * 64,
        pre_residual_by_batch=child,
    )
    return tuple(alignments), parent, bank


def test_residual_chart_preserves_unsupported_subspace_and_uses_four_gates() -> None:
    alignments, parent, bank = _chart_inputs(reject_validation=False)

    chart, receipt = stack._fit_residual_chart(
        alignments=alignments,
        parent=parent,
        child_bank=bank,
        producer_encoder=torch.eye(3, dtype=torch.float64),
        layer_index=0,
    )

    assert receipt["selected_candidate"] != "base_identity"
    assert receipt["supported_rank"] == 1
    assert receipt["candidate_family_constructed_from_fit_only"] is True
    assert receipt["candidate_frozen_before_validation"] is True
    assert receipt["validation_triggered_refit"] is False
    assert set(receipt["partition_batch_indices"]) == set(stack._PARTITIONS)
    torch.testing.assert_close(
        chart[1:],
        torch.eye(3, dtype=torch.float64)[1:],
        rtol=0,
        atol=1e-12,
    )


def test_residual_chart_validation_can_only_reject_back_to_base() -> None:
    alignments, parent, bank = _chart_inputs(reject_validation=True)

    chart, receipt = stack._fit_residual_chart(
        alignments=alignments,
        parent=parent,
        child_bank=bank,
        producer_encoder=torch.eye(3, dtype=torch.float64),
        layer_index=0,
    )

    assert receipt["selected_before_validation"] != "base_identity"
    assert receipt["selected_candidate"] == "base_identity"
    assert receipt["validation_nonregression_passed"] is False
    assert receipt["validation_rejection_fell_back_to_base"] is True
    torch.testing.assert_close(
        chart,
        torch.eye(3, dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_attention_decode_uses_reconstructed_history_not_suffix_rows(
    monkeypatch,
) -> None:
    phases = (
        SimpleNamespace(
            phase="prefill",
            donor_full_prefix_tokens=2,
            target_full_prefix_tokens=2,
            alignment_sha256="9" * 64,
        ),
        SimpleNamespace(
            phase="decode.000",
            donor_full_prefix_tokens=3,
            target_full_prefix_tokens=3,
            alignment_sha256="0" * 64,
        ),
    )
    alignment = SimpleNamespace(
        source_batch_index=0,
        phase_order=("prefill", "decode.000"),
        phases=phases,
    )
    reference = SimpleNamespace(index=0)
    tensors = {
        "prefill.layer.002.attention.q": torch.ones((1, 1, 2, 1)),
        "decode.000.layer.002.attention.q": torch.ones((1, 1, 1, 1)),
        "prefill.layer.002.attention.k": torch.ones((1, 1, 2, 1)),
        "decode.000.layer.002.attention.k": torch.ones((1, 1, 1, 1)),
    }

    class Session:
        batch_references = (reference,)

        def load_selected_tensors(self, *_args):
            return SimpleNamespace(tensors=tensors)

        def release_selected_tensors(self, _payload):
            pass

    class Capability:
        capability_sha256 = "1" * 64

        def __init__(self, **_kwargs):
            pass

        def require_exact_history(self):
            return self

        def to_evidence_dict(self):
            return {"capability_sha256": self.capability_sha256}

    class History:
        def __init__(
            self,
            query_normalized_pre_rope,
            key_normalized_pre_rope,
            value,
            source_capability_sha256="1" * 64,
            source_tensor_names=(),
        ):
            self.query_normalized_pre_rope = query_normalized_pre_rope
            self.key_normalized_pre_rope = key_normalized_pre_rope
            self.value = value
            self.source_capability_sha256 = source_capability_sha256
            self.source_tensor_names = source_tensor_names

        def to_evidence_dict(self):
            return {"history_sha256": "2" * 64}

    complete_history = History(
        torch.ones((1, 1, 3, 1)),
        torch.ones((1, 1, 3, 1)),
        torch.ones((1, 1, 3, 1)),
        source_tensor_names=(
            "prefill.layer.002.attention.q",
            "decode.000.layer.002.attention.q",
        ),
    )

    def transport(phase, **_kwargs):
        if phase.phase == "prefill":
            return SimpleNamespace(
                query_target_positions=(0, 1),
                query_donor_positions=(0, 1),
            )
        return SimpleNamespace(
            query_target_positions=(0, 2),
            query_donor_positions=(0, 2),
        )

    endpoints = []

    class Endpoint:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            endpoints.append(self)

    class Evidence:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.evidence_sha256 = canonical_sha256(
                ["attention", kwargs["alignment_sha256"]]
            )

    monkeypatch.setattr(stack, "_attention_tensor_names", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        stack,
        "inspect_lfm2_semantic_attention_cache_capability",
        lambda *_args, **_kwargs: Capability(),
    )
    monkeypatch.setattr(
        stack,
        "assemble_lfm2_explicit_pre_rope_history",
        lambda *_args, **_kwargs: complete_history,
    )
    monkeypatch.setattr(stack, "Lfm2SemanticAttentionCacheCapability", Capability)
    monkeypatch.setattr(stack, "SemanticAttentionPreRopeHistory", History)
    monkeypatch.setattr(
        stack,
        "transport_lfm2_phase_probability_preserving_attention",
        transport,
    )
    monkeypatch.setattr(stack, "Lfm2SemanticAttentionReplayEndpoint", Endpoint)
    monkeypatch.setattr(stack, "Lfm2SemanticAttentionTransportEvidence", Evidence)
    monkeypatch.setattr(
        stack,
        "_attention_tensor_names",
        lambda *_args, **_kwargs: tuple(tensors),
    )
    full_post = torch.tensor(
        [[[10.0, 100.0], [11.0, 110.0], [12.0, 120.0]]],
        dtype=torch.float32,
    )
    parent = {
        0: SimpleNamespace(
            phase_rows={
                "prefill": {"residual.post_operator": full_post[:, :2]},
                "decode.000": {"residual.post_operator": full_post[:, 2:]},
            },
            phase_histories={
                "prefill": {"residual.post_operator": full_post[:, :2]},
                "decode.000": {"residual.post_operator": full_post},
            },
        )
    }
    replay = SimpleNamespace(operator_input=torch.ones((1, 3, 2)))
    child_bank = SimpleNamespace(
        captured_child_model_state_sha256="3" * 64,
        bank_sha256="4" * 64,
        replay_by_batch={0: replay},
        pre_residual_by_batch={0: torch.zeros((1, 3, 2))},
    )
    lineage = SimpleNamespace(
        source_inventory=object(),
        query_groups=((0,),),
        query_group_weights=((1.0,),),
        compiler_lineage_sha256="5" * 64,
    )

    result = stack._build_attention_evidence(
        cache=SimpleNamespace(cache_key="6" * 64),
        session=Session(),
        bridge=SimpleNamespace(artifact_sha256="7" * 64),
        alignments=(alignment,),
        parent=parent,
        child_bank=child_bank,
        chart=torch.eye(2, dtype=torch.float64),
        chart_receipt={"result_sha256": "8" * 64},
        layer_index=2,
        lineage=lineage,
        target_rope_inv_freq=torch.ones((1,)),
        target_rope_attention_scaling=1.0,
    )

    assert len(result) == 2
    torch.testing.assert_close(
        endpoints[1].parent_post_attention_residual_target_space,
        full_post.index_select(1, torch.tensor((0, 2))).to(torch.float32),
        rtol=0,
        atol=0,
    )


def _parent_mlp_provenance_fixture(*, corrupt_down: bool = False):
    torch.manual_seed(913)
    hidden = 4
    intermediate = 6
    residual = torch.randn((1, 1, hidden), dtype=torch.bfloat16)
    norm = torch.randn((hidden,), dtype=torch.bfloat16)
    gate = torch.randn((intermediate, hidden), dtype=torch.bfloat16)
    up = torch.randn((intermediate, hidden), dtype=torch.bfloat16)
    down = torch.randn((hidden, intermediate), dtype=torch.bfloat16)
    epsilon = 1e-5
    residual_float = residual.to(torch.float32)
    mlp_input = norm * (
        residual_float
        * torch.rsqrt(residual_float.square().mean(dim=-1, keepdim=True) + epsilon)
    ).to(torch.bfloat16)
    expected_gate = torch.nn.functional.linear(mlp_input, gate)
    expected_up = torch.nn.functional.linear(mlp_input, up)
    expected_post = torch.nn.functional.silu(expected_gate) * expected_up
    expected_layer = residual + torch.nn.functional.linear(expected_post, down)
    phase = SimpleNamespace(
        phase="prefill",
        semantic_phase_sha256="a" * 64,
    )
    alignment = SimpleNamespace(
        source_batch_index=0,
        phases=(phase,),
    )
    trajectory = SimpleNamespace(
        evidence_sha256="b" * 64,
        phase_rows={
            "prefill": {
                "residual.post_operator": residual,
                "mlp.gate": expected_gate,
                "mlp.up": expected_up,
                "mlp.post_swiglu": expected_post,
                "residual.post_layer": expected_layer,
            }
        },
    )
    direct_down = down.clone()
    if corrupt_down:
        direct_down[0, 0] += torch.tensor(1.0, dtype=torch.bfloat16)
    tensors = {
        "model.layers.0.feed_forward.w1.weight": gate,
        "model.layers.0.feed_forward.w3.weight": up,
        "model.layers.0.feed_forward.w2.weight": direct_down,
        "model.layers.0.ffn_norm.weight": norm,
    }

    class ParentSession:
        manifest_sha256 = "c" * 64
        open_receipt_sha256 = "d" * 64
        config_file_sha256 = "e" * 64
        norm_epsilon = epsilon

        def tensor(self, key):
            return tensors[key].detach().cpu().contiguous().clone()

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.ones((1,), dtype=torch.bfloat16))

    target = SimpleNamespace(
        model=SimpleNamespace(layers=(Layer(),)),
    )
    cache = SimpleNamespace(
        cache_key="f" * 64,
        parent_fingerprint="0" * 64,
        layer_topology_fingerprint="1" * 64,
        manifest={"manifest_sha256": "2" * 64},
    )
    bridge = SimpleNamespace(artifact_sha256="3" * 64)
    chart_body = {"schema": "chart"}
    chart_receipt = {
        **chart_body,
        "result_sha256": canonical_sha256(chart_body),
    }
    return (
        target,
        cache,
        bridge,
        ParentSession(),
        (alignment,),
        {0: trajectory},
        chart_receipt,
    )


def test_direct_parent_mlp_inventory_is_functionally_bound_to_cached_endpoints() -> (
    None
):
    target, cache, bridge, session, alignments, parent, chart = (
        _parent_mlp_provenance_fixture()
    )

    inventory, receipt = stack._direct_mlp_inventory(
        inventory_type=Lfm2DirectLocalMlpSourceInventory,
        parent_session=session,
        cache=cache,
        bridge=bridge,
        layer_index=0,
        operator_kind="short_conv",
        chart_receipt=chart,
        attention_lineage=None,
        alignments=alignments,
        parent=parent,
        target_model=target,
    )

    assert receipt["all_cached_endpoints_bit_exact"] is True
    assert receipt["direct_parent_tensors_functionally_bound_to_cache"] is True
    assert receipt["semantic_endpoint_rows"] == 1
    assert inventory.direct_local_loader_receipt_sha256 == receipt["result_sha256"]
    assert inventory.verify() is True


def test_direct_parent_mlp_inventory_rejects_self_labeled_wrong_w2() -> None:
    target, cache, bridge, session, alignments, parent, chart = (
        _parent_mlp_provenance_fixture(corrupt_down=True)
    )

    with pytest.raises(
        stack.Lfm2SemanticLayerStackRefoldError,
        match="do not exactly reproduce cached BF16",
    ):
        stack._direct_mlp_inventory(
            inventory_type=Lfm2DirectLocalMlpSourceInventory,
            parent_session=session,
            cache=cache,
            bridge=bridge,
            layer_index=0,
            operator_kind="short_conv",
            chart_receipt=chart,
            attention_lineage=None,
            alignments=alignments,
            parent=parent,
            target_model=target,
        )


def test_heldout_selected_incumbent_noop_contract_is_strict() -> None:
    state = "4" * 64
    receipt = {
        "status": "heldout-selected-incumbent-noop",
        "installed": False,
        "incumbent_selected_by_heldout_gates": True,
        "fit_check_nonregression_passed": True,
        "selection_nonregression_passed": True,
        "validation_nonregression_passed": True,
        "candidate_frozen_before_validation": True,
        "validation_triggered_refit": False,
        "model_state_before_sha256": state,
        "model_state_after_sha256": state,
        "changed_state_keys": [],
        "parameter_count_changed": False,
        "topology_changed": False,
        "parameter_identities_preserved": True,
        "parameter_storage_preserved": True,
        "gradient_state_preserved": True,
        "exact_noop_state_proven": True,
        "analytic_cached_evidence_only": True,
        "operation_audit": dict(stack._ZERO_OPERATIONS),
    }

    assert stack._authenticated_heldout_incumbent_noop(
        receipt,
        before_sha256=state,
        after_sha256=state,
        changed_state_keys=(),
    )
    assert stack._authenticated_heldout_incumbent_noop(
        {
            **receipt,
            "operation_audit": {
                **receipt["operation_audit"],
                "parameters_sparsified": 0,
                "parameter_aliases_created": 0,
            },
        },
        before_sha256=state,
        after_sha256=state,
        changed_state_keys=(),
    )
    assert not stack._authenticated_heldout_incumbent_noop(
        {**receipt, "validation_nonregression_passed": False},
        before_sha256=state,
        after_sha256=state,
        changed_state_keys=(),
    )
    assert not stack._authenticated_heldout_incumbent_noop(
        {
            **receipt,
            "operation_audit": {
                **receipt["operation_audit"],
                "parameters_sparsified": 1,
            },
        },
        before_sha256=state,
        after_sha256=state,
        changed_state_keys=(),
    )


def test_stack_never_implements_its_own_cuda_bf16_router() -> None:
    source = stack.__file__
    text = open(source, encoding="utf-8").read()
    assert "torch.cuda.is_bf16_supported" not in text
    assert "cuda.get_device_capability" not in text
