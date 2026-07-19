from __future__ import annotations

from types import MappingProxyType

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_cpcf_model_compiler as model_compiler  # noqa: E402
from neural_forge.hf_lfm2_cpcf_installability import (  # noqa: E402
    Lfm2CpcfInstallableLayerEvidence,
    Lfm2CpcfInstallablePhaseEvidence,
)


def _digest(label: str) -> str:
    return model_compiler._sha({"label": label})


def _phase(
    *,
    batch_index: int,
    record_id: str,
    split: str,
    phase_name: str,
    token_ids: tuple[int, ...],
) -> model_compiler.Lfm2CpcfTargetPhase:
    sampled = len(token_ids) if phase_name in {"prefill", "cached_prefill"} else 1
    donor_sha = _digest(f"donor:{record_id}")
    layer = Lfm2CpcfInstallableLayerEvidence(
        layer_index=0,
        operator_kind="short_conv",
        source_width=3,
        tensors={
            "residual.pre": torch.zeros(
                1,
                sampled,
                3,
                dtype=torch.float32,
            )
        },
        evidence_sha256=_digest(f"layer:{record_id}:{phase_name}"),
    )
    parent = Lfm2CpcfInstallablePhaseEvidence(
        cache_key=_digest("cache"),
        batch_index=batch_index,
        input_sha256=donor_sha,
        phase=phase_name,
        layers=(layer,),
        final_norm_tensors={},
        evidence_sha256=_digest(f"parent:{record_id}:{phase_name}"),
    )
    ids = torch.tensor((token_ids,), dtype=torch.long)
    mask = torch.ones_like(ids)
    positions = torch.arange(len(token_ids), dtype=torch.long).unsqueeze(0)
    alignment_sha = _digest(f"alignment:{record_id}")
    body = {
        "schema": model_compiler.LFM2_CPCF_TARGET_PHASE_SCHEMA,
        "batch_index": batch_index,
        "record_id": record_id,
        "split": split,
        "phase": phase_name,
        "donor_input_sha256": donor_sha,
        "target_alignment_sha256": alignment_sha,
        "input_ids_sha256": model_compiler._tensor_sha256(ids),
        "attention_mask_sha256": model_compiler._tensor_sha256(mask),
        "position_ids_sha256": model_compiler._tensor_sha256(positions),
        "batch_size": 1,
        "full_prefix_tokens": len(token_ids),
        "sampled_suffix_tokens": sampled,
        "parent_evidence_sha256": parent.evidence_sha256,
        "numeric_token_id_equality_required": False,
        "actual_child_full_prefix_required": True,
    }
    return model_compiler.Lfm2CpcfTargetPhase(
        batch_index=batch_index,
        record_id=record_id,
        split=split,
        phase=phase_name,
        donor_input_sha256=donor_sha,
        target_alignment_sha256=alignment_sha,
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        sampled_suffix_tokens=sampled,
        parent_evidence=parent,
        phase_sha256=model_compiler._sha(body),
    )


def _trajectory(
    batch_index: int,
    split: str,
) -> tuple[model_compiler.Lfm2CpcfTargetPhase, ...]:
    record_id = f"record-{batch_index}"
    return (
        _phase(
            batch_index=batch_index,
            record_id=record_id,
            split=split,
            phase_name="prefill",
            token_ids=(batch_index + 1, batch_index + 2),
        ),
        _phase(
            batch_index=batch_index,
            record_id=record_id,
            split=split,
            phase_name="decode.000",
            token_ids=(batch_index + 1, batch_index + 2, batch_index + 3),
        ),
        _phase(
            batch_index=batch_index,
            record_id=record_id,
            split=split,
            phase_name="decode.001",
            token_ids=(
                batch_index + 1,
                batch_index + 2,
                batch_index + 3,
                batch_index + 4,
            ),
        ),
    )


def _state(phase: model_compiler.Lfm2CpcfTargetPhase) -> torch.Tensor:
    values = phase.input_ids.to(dtype=torch.float64)
    return torch.stack((values, values.square()), dim=-1)


def _bank(*, validation_offset: float = 0.0):
    phases = (
        *_trajectory(0, "selection"),
        *_trajectory(1, "fit"),
        *_trajectory(2, "validation"),
    )
    states = {}
    for phase in phases:
        value = _state(phase)
        if phase.split == "validation":
            value = value + validation_offset
        states[phase.identity] = value
    return model_compiler._phase_state_bank(
        "layer.003.post_layer",
        4,
        phases,
        states,
    )


def test_split_selection_reconstructs_content_and_reseals_independent_archive() -> None:
    bank = _bank()
    source_sha = bank.state_bank_sha256
    source_phases = bank.phases
    source_values = {
        identity: value.clone()
        for identity, value in bank.states.items()
    }

    selected = model_compiler.select_lfm2_cpcf_phase_state_bank(
        bank,
        splits=("fit", "validation"),
    )

    assert selected.boundary == bank.boundary
    assert selected.layer_index == bank.layer_index
    assert tuple(phase.split for phase in selected.phases) == (
        "fit",
        "fit",
        "fit",
        "validation",
        "validation",
        "validation",
    )
    assert isinstance(selected.states, MappingProxyType)
    assert set(selected.states) == {
        phase.identity for phase in selected.phases
    }
    inventory = [
        {
            "identity": [phase.batch_index, phase.phase],
            "phase_sha256": phase.phase_sha256,
            "state_sha256": model_compiler._tensor_sha256(
                selected.states[phase.identity]
            ),
        }
        for phase in selected.phases
    ]
    expected_body = {
        "schema": model_compiler.LFM2_CPCF_TARGET_RUNTIME_SCHEMA,
        "scope": "phase_state_bank",
        "boundary": bank.boundary,
        "layer_index": bank.layer_index,
        "hidden_size": bank.hidden_size,
        "ordered_states": inventory,
        "actual_folded_upstream_states": True,
    }
    assert selected.state_bank_sha256 == model_compiler._sha(expected_body)
    assert selected.state_bank_sha256 != source_sha
    for phase in selected.phases:
        torch.testing.assert_close(
            selected.states[phase.identity],
            bank.states[phase.identity],
            rtol=0,
            atol=0,
        )
        assert (
            selected.states[phase.identity].data_ptr()
            != bank.states[phase.identity].data_ptr()
        )

    assert bank.state_bank_sha256 == source_sha
    assert bank.phases == source_phases
    for identity, expected in source_values.items():
        torch.testing.assert_close(
            bank.states[identity],
            expected,
            rtol=0,
            atol=0,
        )
    with pytest.raises(TypeError):
        selected.states[(99, "prefill")] = torch.zeros(1, 1, 2)
    returned = selected.state(selected.phases[0])
    returned.add_(1000)
    assert not torch.equal(
        returned,
        selected.states[selected.phases[0].identity],
    )


def test_split_selection_digest_is_isolated_from_excluded_split_states() -> None:
    first = _bank(validation_offset=0.0)
    second = _bank(validation_offset=50.0)

    first_fit = model_compiler.select_lfm2_cpcf_phase_state_bank(
        first,
        splits=("fit",),
    )
    second_fit = model_compiler.select_lfm2_cpcf_phase_state_bank(
        second,
        splits=("fit",),
    )
    first_validation = model_compiler.select_lfm2_cpcf_phase_state_bank(
        first,
        splits=("validation",),
    )
    second_validation = model_compiler.select_lfm2_cpcf_phase_state_bank(
        second,
        splits=("validation",),
    )

    assert first_fit.state_bank_sha256 == second_fit.state_bank_sha256
    assert (
        first_validation.state_bank_sha256
        != second_validation.state_bank_sha256
    )
    assert all(phase.split == "fit" for phase in first_fit.phases)
    assert set(first_fit.states).isdisjoint(first_validation.states)


@pytest.mark.parametrize(
    ("splits", "error"),
    (
        ("fit", TypeError),
        ((), model_compiler.Lfm2CpcfModelCompilerError),
        (("fit", "fit"), model_compiler.Lfm2CpcfModelCompilerError),
        (("absent",), model_compiler.Lfm2CpcfModelCompilerError),
        ((1,), model_compiler.Lfm2CpcfModelCompilerError),
    ),
)
def test_split_selection_rejects_invalid_split_requests(splits, error) -> None:
    with pytest.raises(error):
        model_compiler.select_lfm2_cpcf_phase_state_bank(
            _bank(),
            splits=splits,
        )


def test_split_selection_rejects_incomplete_or_cross_split_trajectories() -> None:
    complete = _trajectory(1, "fit")
    missing_prefill = model_compiler._phase_state_bank(
        "input_embeddings",
        0,
        complete[1:],
        {phase.identity: _state(phase) for phase in complete[1:]},
    )
    with pytest.raises(
        model_compiler.Lfm2CpcfModelCompilerError,
        match="no complete causal prefill",
    ):
        model_compiler.select_lfm2_cpcf_phase_state_bank(
            missing_prefill,
            splits=("fit",),
        )

    skipped_decode = (complete[0], complete[2])
    skipped_bank = model_compiler._phase_state_bank(
        "input_embeddings",
        0,
        skipped_decode,
        {phase.identity: _state(phase) for phase in skipped_decode},
    )
    with pytest.raises(
        model_compiler.Lfm2CpcfModelCompilerError,
        match="skips or reorders",
    ):
        model_compiler.select_lfm2_cpcf_phase_state_bank(
            skipped_bank,
            splits=("fit",),
        )

    crossed = (
        complete[0],
        _phase(
            batch_index=1,
            record_id="record-1",
            split="validation",
            phase_name="decode.000",
            token_ids=(2, 3, 4),
        ),
    )
    crossed_bank = model_compiler._phase_state_bank(
        "input_embeddings",
        0,
        crossed,
        {phase.identity: _state(phase) for phase in crossed},
    )
    with pytest.raises(
        model_compiler.Lfm2CpcfModelCompilerError,
        match="crosses phase-state splits",
    ):
        model_compiler.select_lfm2_cpcf_phase_state_bank(
            crossed_bank,
            splits=("fit",),
        )
