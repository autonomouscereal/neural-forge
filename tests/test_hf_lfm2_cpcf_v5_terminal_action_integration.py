from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import neural_forge.hf_lfm2_cpcf_v5_terminal_action_integration as integration  # noqa: E402
import neural_forge.hf_lfm2_cpcf_v5_model_compiler as v5  # noqa: E402
from neural_forge.hf_lfm2_cpcf_installability import (  # noqa: E402
    Lfm2CpcfInstallableLayerEvidence,
    Lfm2CpcfInstallablePhaseEvidence,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (  # noqa: E402
    Lfm2CpcfTargetPhase,
    build_lfm2_cpcf_target_phases,
    replay_lfm2_cpcf_phase_bank,
    seed_lfm2_cpcf_phase_states,
)
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (  # noqa: E402
    Lfm2CpcfTerminalActionPolicy,
)
from neural_forge.hf_tied_causal_interface_fold import (  # noqa: E402
    TiedCausalInterfaceFold,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _TraceAdapter:
    def __init__(self, records) -> None:
        self.records = tuple(records)

    def verify(self):
        return self


def _copy_phase(
    phase: Lfm2CpcfTargetPhase,
    evidence: Lfm2CpcfInstallablePhaseEvidence,
) -> Lfm2CpcfTargetPhase:
    return Lfm2CpcfTargetPhase(
        batch_index=phase.batch_index,
        record_id=phase.record_id,
        split=phase.split,
        phase=phase.phase,
        donor_input_sha256=phase.donor_input_sha256,
        target_alignment_sha256=phase.target_alignment_sha256,
        input_ids=phase.input_ids,
        attention_mask=phase.attention_mask,
        position_ids=phase.position_ids,
        sampled_suffix_tokens=phase.sampled_suffix_tokens,
        parent_evidence=evidence,
        phase_sha256=phase.phase_sha256,
    )


def _phase_evidence(
    batch_index: int,
    *,
    final_norm_output=None,
    sampled_residual=None,
) -> Lfm2CpcfInstallablePhaseEvidence:
    layer = Lfm2CpcfInstallableLayerEvidence(
        layer_index=0,
        operator_kind="full_attention",
        source_width=4,
        tensors=(
            {}
            if sampled_residual is None
            else {"residual.pre": sampled_residual}
        ),
        evidence_sha256=_sha(f"layer-{batch_index}"),
    )
    return Lfm2CpcfInstallablePhaseEvidence(
        cache_key=_sha("terminal-cache"),
        batch_index=batch_index,
        input_sha256=_sha(f"input-{batch_index}"),
        phase="prefill",
        layers=(layer,),
        final_norm_tensors=(
            {}
            if final_norm_output is None
            else {"output.final_norm_output": final_norm_output}
        ),
        evidence_sha256=_sha(f"phase-{batch_index}"),
    )


def _model_and_capture():
    config = transformers.Lfm2Config(
        vocab_size=16,
        hidden_size=4,
        intermediate_size=6,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["full_attention"],
        conv_L_cache=2,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    model = transformers.Lfm2ForCausalLM(config).to(torch.bfloat16).eval()
    generator = torch.Generator().manual_seed(831)
    gate = (torch.randn(6, 4, generator=generator) * 0.55).to(torch.bfloat16)
    up = (torch.randn(6, 4, generator=generator) * 0.55).to(torch.bfloat16)
    incumbent = (torch.randn(4, 6, generator=generator) * 0.12).to(
        torch.bfloat16
    )
    target_delta = torch.zeros(4, 6, dtype=torch.float32)
    target_delta[0] = torch.tensor([0.20, -0.16, 0.18, -0.14, 0.12, -0.10])
    target_delta[1] = torch.tensor([-0.09, 0.07, -0.08, 0.06, -0.05, 0.04])
    desired = (incumbent.to(torch.float32) + target_delta).to(torch.bfloat16)
    tied = (torch.randn(16, 4, generator=generator) * 0.65).to(torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        layer = model.model.layers[0]
        layer.operator_norm.weight.fill_(1.0)
        layer.ffn_norm.weight.fill_(1.0)
        layer.self_attn.q_layernorm.weight.fill_(1.0)
        layer.self_attn.k_layernorm.weight.fill_(1.0)
        model.model.embedding_norm.weight.fill_(1.0)
        layer.feed_forward.w1.weight.copy_(gate)
        layer.feed_forward.w3.weight.copy_(up)
        layer.feed_forward.w2.weight.copy_(incumbent)
        model.get_input_embeddings().weight.copy_(tied)

    split_names = (
        "selection",
        "selection",
        "fit",
        "fit",
        "fit",
        "fit",
        "fit",
        "fit",
        "fit",
        "fit",
        "validation",
        "validation",
        "validation",
    )
    records = []
    parent = []
    for batch_index, split in enumerate(split_names):
        records.append(
            SimpleNamespace(
                ordinal=batch_index,
                record_id=f"trajectory-{batch_index:03d}",
                split=split,
                donor_cache_input_sha256=_sha(f"input-{batch_index}"),
                alignment_sha256=_sha(f"alignment-{batch_index}"),
                target_input_ids=torch.tensor([[batch_index % 16]]),
                attention_mask=torch.ones(1, 1, dtype=torch.long),
                target_forced_decode_ids=torch.tensor([[(batch_index + 1) % 16]]),
                position_ids=None,
            )
        )
        parent.append(
            _phase_evidence(
                batch_index,
                sampled_residual=torch.zeros(1, 1, 4, dtype=torch.bfloat16),
            )
        )
    materialized = build_lfm2_cpcf_target_phases(_TraceAdapter(records), parent)
    shells = tuple(
        _copy_phase(phase, _phase_evidence(phase.batch_index))
        for phase in materialized
    )
    input_bank = seed_lfm2_cpcf_phase_states(model, shells)
    incumbent_replay = replay_lfm2_cpcf_phase_bank(
        model,
        layer_index=0,
        input_bank=input_bank,
    )
    capture = integration.capture_lfm2_cpcf_v5_terminal_action_inputs(
        layer_index=0,
        input_bank=input_bank,
        final_replay=incumbent_replay,
    )
    final_phases = []
    epsilon = float(model.model.embedding_norm.variance_epsilon)
    for phase in shells:
        replay = incumbent_replay.replays[phase.identity]
        mlp = phase.suffix(replay.mlp_input).reshape(-1, 4).to(torch.float32)
        pre = phase.suffix(replay.post_operator_residual).reshape(-1, 4).to(
            torch.float64
        )
        features = torch.nn.functional.silu(mlp @ gate.to(torch.float32).T) * (
            mlp @ up.to(torch.float32).T
        )
        target_post = pre + features.to(torch.float64) @ desired.to(torch.float64).T
        parent_final = target_post / (
            target_post.square().mean(dim=1, keepdim=True) + epsilon
        ).sqrt()
        final_phases.append(
            _copy_phase(
                phase,
                _phase_evidence(
                    phase.batch_index,
                    final_norm_output=parent_final.reshape(1, 1, 4).to(
                        torch.bfloat16
                    ),
                ),
            )
        )
    interface = TiedCausalInterfaceFold(
        target_weight=tied,
        producer_encoder=torch.eye(4, dtype=torch.float64),
        report={"report_sha256": _sha("test-interface")},
    )
    return (
        model,
        capture,
        incumbent_replay.post_layer_bank,
        tuple(final_phases),
        interface,
        tied,
        epsilon,
        incumbent,
    )


def test_capture_is_suffix_only_and_fit_check_partition_is_whole_trajectory() -> None:
    model, capture, incumbent_bank, phases, interface, tied, epsilon, incumbent = (
        _model_and_capture()
    )
    del model, incumbent_bank, interface, tied, epsilon, incumbent

    assert capture.verify_authentication()
    assert all(
        tuple(capture.mlp_input[phase.identity].shape)
        == (phase.batch_size, phase.sampled_suffix_tokens, 4)
        for phase in phases
    )
    bundle = integration.build_lfm2_cpcf_v5_terminal_action_evidence(
        capture,
        phases,
    )
    assert bundle.report["fit_check_fraction"] == {
        "numerator": 1,
        "denominator": 8,
    }
    assert bundle.report["fit_check_count_rule"] == (
        "ceil(source_fit_trajectory_count / 8)"
    )
    assert bundle.report["trajectory_counts"] == {
        "fit": 7,
        "fit_check": 1,
        "selection": 2,
        "validation": 3,
    }
    assert not set(bundle.fit.row_ids) & set(bundle.fit_check.row_ids)
    second = integration.build_lfm2_cpcf_v5_terminal_action_evidence(
        capture,
        phases,
    )
    assert second.bundle_sha256 == bundle.bundle_sha256


def test_real_corpus_fit_check_contract_is_exactly_134_plus_20() -> None:
    fit_check = integration._fit_check_trajectory_count(154)

    assert fit_check == 20
    assert 154 - fit_check == 134


def test_v5_runtime_policy_keeps_terminal_action_strictly_opt_in() -> None:
    model_policy = v5.legacy_model_fold.Lfm2CpcfModelFoldPolicy()
    disabled = v5.Lfm2CpcfV5RuntimePolicy(
        model_fold_policy=model_policy,
        maximum_resident_evidence_bytes=1,
        full_phase_tree_logical_bytes=1,
    )
    terminal_policy = Lfm2CpcfTerminalActionPolicy()
    enabled = v5.Lfm2CpcfV5RuntimePolicy(
        model_fold_policy=model_policy,
        maximum_resident_evidence_bytes=1,
        full_phase_tree_logical_bytes=1,
        terminal_action_residual_policy=terminal_policy,
    )

    assert "terminal_action_residual_policy" not in disabled.as_dict()
    assert enabled.as_dict()["terminal_action_residual_policy"] == (
        terminal_policy.as_dict()
    )
    implementation_files = {
        row["filename"] for row in v5.lfm2_cpcf_v5_compiler_implementation()[
            "ordered_source_files"
        ]
    }
    assert "hf_lfm2_cpcf_terminal_action_residual.py" in implementation_files
    assert "hf_lfm2_cpcf_v5_terminal_action_integration.py" in implementation_files


def test_live_install_preserves_parameter_and_publishes_refreshed_bank() -> None:
    model, capture, incumbent_bank, phases, interface, tied, epsilon, incumbent = (
        _model_and_capture()
    )
    parameter = model.model.layers[0].feed_forward.w2.weight
    identity = (id(parameter), int(parameter.data_ptr()), tuple(parameter.shape))

    result = integration.compile_install_lfm2_cpcf_v5_terminal_action(
        target_model=model,
        capture=capture,
        incumbent_final_state_bank=incumbent_bank,
        final_phases=phases,
        interface=interface,
        source_tied_table=tied,
        norm_epsilon=epsilon,
        policy=Lfm2CpcfTerminalActionPolicy(),
    )

    # Since "Defer fold quality decisions to manual review" (dbb87016) the pure
    # terminal-action compiler always publishes the incumbent writer and its
    # result authentication *hard-requires* ``accepted is False`` (see
    # hf_lfm2_cpcf_terminal_action_residual
    # ._authenticate_deferred_structural_frontier_compiled_result).  An accepted
    # TARC result can therefore no longer exist, so the integration must publish
    # nothing, install nothing, and leave the live target byte-identical.
    report = result.report
    assert result.accepted is False
    assert report["pure_tarc_accepted"] is False
    assert report["pure_tarc_changed"] is False
    assert report["writer_installed_in_place"] is False
    assert report["published_is_incumbent"] is True
    assert report["live_final_layer_replay_performed"] is False
    assert report["live_gate_passed"] is False
    assert report["rollback_performed"] is False
    assert report["rollback_exact"] is True
    assert report["operation_audit"]["target_full_model_forward_calls"] == 0
    assert report["operation_audit"]["target_final_layer_replay_calls"] == 0

    # The deferral is a review hand-off, not an empty frontier: the compiler
    # still had structurally eligible non-incumbent candidates to offer.
    pure = report["pure_tarc_report"]
    assert pure["published_writer_is_incumbent_pending_manual_review"] is True
    assert pure["manual_whole_response_review_required"] is True
    assert pure["automatic_candidate_acceptance_performed"] is False
    assert pure["structurally_eligible_frontier_count"] > 0

    # The live parent stays exactly where it was, in the same Parameter object.
    assert identity == (id(parameter), int(parameter.data_ptr()), tuple(parameter.shape))
    assert torch.equal(parameter.detach().cpu(), incumbent)
    assert report["writer_parameter_identity_preserved"] is True
    assert result.final_state_bank.state_bank_sha256 == (
        incumbent_bank.state_bank_sha256
    )


def test_live_gate_rejection_restores_writer_and_exact_local_replay(monkeypatch) -> None:
    model, capture, incumbent_bank, phases, interface, tied, epsilon, incumbent = (
        _model_and_capture()
    )
    parameter = model.model.layers[0].feed_forward.w2.weight
    identity = (id(parameter), int(parameter.data_ptr()), tuple(parameter.shape))
    original_gate = integration._live_metric_gates
    live_gate_calls: list[bool] = []

    def reject_selection(candidate, incumbent_metrics, policy, *, selection):
        live_gate_calls.append(bool(selection))
        gates = original_gate(
            candidate,
            incumbent_metrics,
            policy,
            selection=selection,
        )
        if selection:
            gates["forced_test_live_rejection"] = False
        return gates

    monkeypatch.setattr(integration, "_live_metric_gates", reject_selection)
    result = integration.compile_install_lfm2_cpcf_v5_terminal_action(
        target_model=model,
        capture=capture,
        incumbent_final_state_bank=incumbent_bank,
        final_phases=phases,
        interface=interface,
        source_tied_table=tied,
        norm_epsilon=epsilon,
        policy=Lfm2CpcfTerminalActionPolicy(),
    )

    # A deferred TARC result must never reach the live metric gate at all: the
    # integration only installs and re-replays when the pure compiler both
    # accepted and changed the writer, and since dbb87016 it can do neither.
    # The forced rejection installed above is therefore unreachable, and the
    # writer is restored trivially because it was never installed at all.
    report = result.report
    assert live_gate_calls == []
    assert result.accepted is False
    assert report["pure_tarc_accepted"] is False
    assert report["pure_tarc_changed"] is False
    assert report["writer_installed_in_place"] is False
    assert report["live_gates"] == {}
    assert report["live_upstream_endpoint_records"] == []
    assert report["rollback_performed"] is False
    assert report["rollback_exact"] is True
    assert report["published_is_incumbent"] is True
    assert report["operation_audit"]["target_final_layer_replay_calls"] == 0
    assert result.final_state_bank.state_bank_sha256 == incumbent_bank.state_bank_sha256
    assert torch.equal(parameter.detach().cpu(), incumbent)
    assert identity == (id(parameter), int(parameter.data_ptr()), tuple(parameter.shape))
