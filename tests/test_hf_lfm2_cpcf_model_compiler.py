from __future__ import annotations

import hashlib

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.hf_lfm2_cpcf_installability import (  # noqa: E402
    Lfm2CpcfInstallableLayerEvidence,
    Lfm2CpcfInstallablePhaseEvidence,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (  # noqa: E402
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
        # The next-boundary bank and the replay receipt intentionally share
        # the one authenticated CPU post-layer archive.  There is no second
        # retained host copy of the same endpoint.
        assert (
            conv.post_layer_bank.states[phase.identity].data_ptr()
            == conv.replays[phase.identity].post_layer_residual.data_ptr()
        )
        assert (
            attention.post_layer_bank.states[phase.identity].data_ptr()
            == attention.replays[phase.identity].post_layer_residual.data_ptr()
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
    assert attention.post_layer_bank.layer_index == 2


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
