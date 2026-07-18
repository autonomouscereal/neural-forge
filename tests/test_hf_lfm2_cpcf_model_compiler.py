from __future__ import annotations

import hashlib
import json

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.hf_lfm2_cpcf_installability import (  # noqa: E402
    Lfm2CpcfInstallableLayerEvidence,
    Lfm2CpcfInstallablePhaseEvidence,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (  # noqa: E402
    Lfm2CpcfLayerReplayBank,
    Lfm2CpcfModelCompilerError,
    build_lfm2_cpcf_target_phases,
    install_lfm2_cpcf_neutral_norms,
    install_lfm2_cpcf_tied_interface,
    replay_lfm2_cpcf_phase_bank,
    seed_lfm2_cpcf_phase_states,
)
from neural_forge.hf_tied_causal_interface_fold import (  # noqa: E402
    TiedCausalInterfaceFold,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha(value) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _target():
    config = transformers.Lfm2Config(
        vocab_size=24,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    model = transformers.Lfm2ForCausalLM(config)
    model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def _interface() -> TiedCausalInterfaceFold:
    generator = torch.Generator().manual_seed(1701)
    return TiedCausalInterfaceFold(
        target_weight=torch.randn(24, 32, generator=generator),
        producer_encoder=torch.randn(40, 32, generator=generator),
        report={"report_sha256": _sha("interface")},
    )


def _layer_evidence(index: int, sampled: int):
    tensors = {
        "residual.pre": torch.zeros(1, sampled, 40),
    }
    return Lfm2CpcfInstallableLayerEvidence(
        layer_index=index,
        operator_kind="short_conv" if index == 0 else "full_attention",
        source_width=40,
        tensors=tensors,
        evidence_sha256=_sha(f"layer-{index}"),
    )


def _phase_evidence(batch: int, phase: str, sampled: int):
    return Lfm2CpcfInstallablePhaseEvidence(
        cache_key=_sha("cache"),
        batch_index=batch,
        input_sha256=_sha(f"donor-{batch}"),
        phase=phase,
        layers=(
            _layer_evidence(0, sampled),
            _layer_evidence(1, sampled),
        ),
        final_norm_tensors={},
        evidence_sha256=_sha(f"phase-{batch}-{phase}"),
    )


class _Record:
    ordinal = 0
    record_id = "row-0"
    split = "fit"
    donor_cache_input_sha256 = _sha("donor-0")
    alignment_sha256 = _sha("alignment")
    target_input_ids = torch.tensor([[3, 4, 5, 6]])
    attention_mask = torch.ones(1, 4, dtype=torch.long)
    target_forced_decode_ids = torch.tensor([[7, 8]])
    position_ids = None


class _Adapter:
    records = (_Record(),)

    def verify(self):
        return self


def _phases():
    return build_lfm2_cpcf_target_phases(
        _Adapter(),
        (
            _phase_evidence(0, "prefill", 3),
            _phase_evidence(0, "decode.000", 1),
            _phase_evidence(0, "decode.001", 1),
        ),
    )


def _assert_cpu_archived_layer_bank(bank) -> None:
    assert all(
        state.device.type == "cpu"
        for state in bank.post_layer_bank.states.values()
    )
    for replay in bank.replays.values():
        assert all(
            getattr(replay, name).device.type == "cpu"
            for name in (
                "operator_input",
                "operator_delta",
                "post_operator_residual",
                "mlp_input",
                "mlp_delta",
                "post_layer_residual",
            )
        )


def test_target_phase_alignment_builds_full_decode_prefixes() -> None:
    phases = _phases()

    assert [phase.full_prefix_tokens for phase in phases] == [4, 5, 6]
    assert [phase.sampled_suffix_tokens for phase in phases] == [3, 1, 1]
    assert phases[1].input_ids.tolist() == [[3, 4, 5, 6, 7]]
    assert phases[2].input_ids.tolist() == [[3, 4, 5, 6, 7, 8]]
    assert phases[2].position_ids.tolist() == [[0, 1, 2, 3, 4, 5]]
    assert phases[0].donor_input_sha256 != phases[0].target_alignment_sha256


def test_tied_install_and_full_prefix_replay_propagate_actual_child_states() -> None:
    target = _target()
    interface = _interface()
    phases = _phases()

    tied_report = install_lfm2_cpcf_tied_interface(target, interface)
    norm_report = install_lfm2_cpcf_neutral_norms(target)
    input_bank = seed_lfm2_cpcf_phase_states(target, phases)
    conv = replay_lfm2_cpcf_phase_bank(
        target,
        layer_index=0,
        input_bank=input_bank,
    )
    attention = replay_lfm2_cpcf_phase_bank(
        target,
        layer_index=1,
        input_bank=conv.post_layer_bank,
    )

    assert tied_report["native_tie_preserved"] is True
    assert norm_report["installed_norm_count"] == 7
    assert input_bank.states[(0, "decode.001")].shape == (1, 6, 32)
    assert all(state.device.type == "cpu" for state in input_bank.states.values())
    _assert_cpu_archived_layer_bank(conv)
    _assert_cpu_archived_layer_bank(attention)
    for phase in phases:
        if phase.phase == "prefill":
            assert (
                conv.post_layer_bank.states[phase.identity].data_ptr()
                == conv.replays[phase.identity].post_layer_residual.data_ptr()
            )
            assert (
                attention.post_layer_bank.states[phase.identity].data_ptr()
                == attention.replays[
                    phase.identity
                ].post_layer_residual.data_ptr()
            )
        else:
            # Decode banks retain the complete raw replay in the receipt and
            # separately stitch only its new suffix onto the already
            # authenticated causal prefix.
            assert (
                conv.post_layer_bank.states[phase.identity].data_ptr()
                != conv.replays[phase.identity].post_layer_residual.data_ptr()
            )
            assert (
                attention.post_layer_bank.states[phase.identity].data_ptr()
                != attention.replays[
                    phase.identity
                ].post_layer_residual.data_ptr()
            )
        # Every branch writer is still the compiler-owned zero blank.  The
        # replay therefore proves exact identity propagation of the installed
        # tied embeddings through both real target macrocell topologies.
        torch.testing.assert_close(
            conv.post_layer_bank.state(phase),
            input_bank.state(phase),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            attention.post_layer_bank.state(phase),
            input_bank.state(phase),
            rtol=0,
            atol=0,
        )
    assert attention.report["actual_folded_upstream_input"] is True
    assert attention.report["child_model_forward_count"] == 0
    assert (
        attention.report["canonical_causal_post_layer_state_stitching"]
        is True
    )
    assert attention.report["post_layer_state_assembly"] == (
        "complete_prefill_plus_ordered_decode_suffixes"
    )
    assert attention.report["raw_recomputed_historical_prefix_used"] is False
    assert attention.report["causal_trajectory_count"] == 1
    assert attention.report["decode_suffix_stitch_count"] == 2
    assert [
        record["source"]
        for record in attention.report["causal_stitch_records"]
    ] == [
        "complete_prefill_replay",
        "ordered_decode_terminal_suffix",
        "ordered_decode_terminal_suffix",
    ]
    assert torch.equal(
        attention.post_layer_bank.states[(0, "decode.000")][:, :4, :],
        attention.post_layer_bank.states[(0, "prefill")],
    )
    assert torch.equal(
        attention.post_layer_bank.states[(0, "decode.001")][:, :5, :],
        attention.post_layer_bank.states[(0, "decode.000")],
    )
    assert attention.post_layer_bank.layer_index == 2


def test_nonzero_attention_stitches_canonical_prefix_and_raw_suffix() -> None:
    target = _target()
    install_lfm2_cpcf_tied_interface(target, _interface())
    install_lfm2_cpcf_neutral_norms(target)
    generator = torch.Generator().manual_seed(20260717)
    with torch.no_grad():
        for name, parameter in target.model.layers[1].named_parameters():
            if (
                name.startswith("self_attn.")
                and name.endswith("_proj.weight")
            ):
                parameter.copy_(
                    torch.randn(
                        parameter.shape,
                        generator=generator,
                        dtype=parameter.dtype,
                    )
                    * 0.05
                )
    phases = _phases()
    input_bank = seed_lfm2_cpcf_phase_states(target, phases)
    conv = replay_lfm2_cpcf_phase_bank(
        target,
        layer_index=0,
        input_bank=input_bank,
    )
    replay = replay_lfm2_cpcf_phase_bank(
        target,
        layer_index=1,
        input_bank=conv.post_layer_bank,
    )

    assert int(
        torch.count_nonzero(
            replay.replays[(0, "prefill")].operator_delta
        ).item()
    ) > 0
    assert not torch.equal(
        replay.post_layer_bank.states[(0, "prefill")],
        conv.post_layer_bank.states[(0, "prefill")],
    )
    for previous, current in zip(phases, phases[1:]):
        canonical = replay.post_layer_bank.state(current)
        prior = replay.post_layer_bank.state(previous)
        raw = replay.replays[current.identity].post_layer_residual
        torch.testing.assert_close(
            canonical[:, : previous.full_prefix_tokens, :],
            prior,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            canonical[:, -current.sampled_suffix_tokens :, :],
            raw[:, -current.sampled_suffix_tokens :, :],
            rtol=0,
            atol=0,
        )


def test_replay_rejects_corrupted_decode_input_prefix() -> None:
    target = _target()
    install_lfm2_cpcf_tied_interface(target, _interface())
    install_lfm2_cpcf_neutral_norms(target)
    phases = _phases()
    bank = seed_lfm2_cpcf_phase_states(target, phases)
    with torch.no_grad():
        bank.states[phases[2].identity][:, 0, 0].add_(1.0)

    with pytest.raises(
        Lfm2CpcfModelCompilerError,
        match="changed or skipped its authenticated causal prefix",
    ):
        replay_lfm2_cpcf_phase_bank(
            target,
            layer_index=0,
            input_bank=bank,
        )


def test_replay_archive_rejects_resealed_drift_bound_loosening() -> None:
    target = _target()
    install_lfm2_cpcf_tied_interface(target, _interface())
    install_lfm2_cpcf_neutral_norms(target)
    replay = replay_lfm2_cpcf_phase_bank(
        target,
        layer_index=0,
        input_bank=seed_lfm2_cpcf_phase_states(target, _phases()),
    )
    with pytest.raises(TypeError):
        replay.report["causal_stitch_records"][1][
            "raw_prefix_permitted_relative_error"
        ] = 1.0
    report = replay.as_dict()
    report["causal_stitch_records"][1][
        "raw_prefix_permitted_relative_error"
    ] = 1.0
    report["causal_stitch_records"][1][
        "raw_prefix_permitted_maximum_absolute_error"
    ] = 1e9
    report["causal_stitch_bundle_sha256"] = _canonical_sha(
        report["causal_stitch_records"]
    )
    body = dict(report)
    body.pop("report_sha256")
    report["report_sha256"] = _canonical_sha(body)

    with pytest.raises(
        Lfm2CpcfModelCompilerError,
        match="unauthenticated drift metrics",
    ):
        Lfm2CpcfLayerReplayBank(
            layer_index=replay.layer_index,
            input_bank_sha256=replay.input_bank_sha256,
            replays=replay.replays,
            post_layer_bank=replay.post_layer_bank,
            report=report,
        )


def test_replay_rejects_archive_dtype_drift_instead_of_casting() -> None:
    target = _target()
    install_lfm2_cpcf_tied_interface(target, _interface())
    install_lfm2_cpcf_neutral_norms(target)
    bank = seed_lfm2_cpcf_phase_states(target, _phases())
    assert {state.dtype for state in bank.states.values()} == {torch.float32}
    target.to(dtype=torch.bfloat16)

    with pytest.raises(
        Lfm2CpcfModelCompilerError,
        match="dtype differs",
    ):
        replay_lfm2_cpcf_phase_bank(
            target,
            layer_index=0,
            input_bank=bank,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to prove CPU archives with a GPU target.",
)
def test_sequential_gpu_layer_replay_retains_zero_cuda_archive_storage() -> None:
    target = _target().to("cuda")
    install_lfm2_cpcf_tied_interface(target, _interface())
    install_lfm2_cpcf_neutral_norms(target)
    phases = _phases()
    input_bank = seed_lfm2_cpcf_phase_states(target, phases)
    conv = replay_lfm2_cpcf_phase_bank(
        target,
        layer_index=0,
        input_bank=input_bank,
    )
    attention = replay_lfm2_cpcf_phase_bank(
        target,
        layer_index=1,
        input_bank=conv.post_layer_bank,
    )

    assert next(target.parameters()).device.type == "cuda"
    assert all(state.device.type == "cpu" for state in input_bank.states.values())
    _assert_cpu_archived_layer_bank(conv)
    _assert_cpu_archived_layer_bank(attention)
    assert conv.report["execution_device"].startswith("cuda")
    assert attention.report["execution_device"].startswith("cuda")
    assert attention.post_layer_bank.layer_index == 2
    for phase in phases:
        torch.testing.assert_close(
            attention.post_layer_bank.state(phase),
            input_bank.state(phase),
            rtol=0,
            atol=0,
        )


def test_phase_bank_rejects_skipping_a_sequential_layer() -> None:
    target = _target()
    install_lfm2_cpcf_tied_interface(target, _interface())
    install_lfm2_cpcf_neutral_norms(target)
    bank = seed_lfm2_cpcf_phase_states(target, _phases())

    with pytest.raises(
        Lfm2CpcfModelCompilerError,
        match="next sequential layer",
    ):
        replay_lfm2_cpcf_phase_bank(
            target,
            layer_index=1,
            input_bank=bank,
        )
