from __future__ import annotations

import copy
import json
from typing import Any

import pytest


torch = pytest.importorskip("torch")

from neural_forge import (  # noqa: E402
    hf_lfm2_semantic_span_cpcf_alignment as alignment_module,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_semantic_span_cpcf_alignment import (  # noqa: E402
    Lfm2SemanticSpanCpcfAlignmentError,
    audit_lfm2_semantic_span_cpcf_refinement,
    build_lfm2_semantic_span_cpcf_alignment,
)
from neural_forge.hf_lfm2_semantic_span_phase_bridge import (  # noqa: E402
    Lfm2SemanticSpanPhaseBridge,
    _build_phases,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
)


class _GreedyTokenizer:
    def __init__(self, pieces: list[str]) -> None:
        self.pieces = list(pieces)
        self.bos_token_id = None
        self.eos_token_id = None
        self.pad_token_id = None
        self.unk_token_id = None
        self.all_special_ids: list[int] = []

    def __len__(self) -> int:
        return len(self.pieces)

    def get_vocab(self) -> dict[str, int]:
        return {piece: index for index, piece in enumerate(self.pieces)}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.pieces[token_id]

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(self.pieces[token_id] for token_id in token_ids)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        result: list[int] = []
        position = 0
        ranked = sorted(
            enumerate(self.pieces),
            key=lambda item: (-len(item[1]), item[0]),
        )
        while position < len(text):
            for token_id, piece in ranked:
                if piece and text.startswith(piece, position):
                    result.append(token_id)
                    position += len(piece)
                    break
            else:
                raise KeyError(text[position:])
        return result


def _bridge(
    donor: _GreedyTokenizer,
    target: _GreedyTokenizer,
    *,
    prompt: tuple[int, ...] = (0, 1),
    forced: tuple[int, ...] = (2, 3),
    batch_index: int = 0,
) -> Lfm2SemanticSpanPhaseBridge:
    return Lfm2SemanticSpanPhaseBridge(
        source_cache_key="1" * 64,
        source_manifest_sha256="2" * 64,
        source_batch_index=batch_index,
        source_input_sha256="3" * 64,
        source_record_sha256="4" * 64,
        donor_tokenizer_semantics_sha256=fingerprint_hf_tokenizer_semantics(donor),
        target_tokenizer_semantics_sha256=fingerprint_hf_tokenizer_semantics(target),
        source_prompt_token_ids=prompt,
        source_forced_decode_token_ids=forced,
        target_position_origin=0,
        phases=_build_phases(
            prompt_ids=prompt,
            forced_ids=forced,
            position_origin=0,
            donor_tokenizer=donor,
            target_tokenizer=target,
        ),
    )


def _tokenizers() -> tuple[_GreedyTokenizer, _GreedyTokenizer]:
    donor = _GreedyTokenizer(["<B>", "a", "b", " c", "!", "z"])
    target = _GreedyTokenizer(["<B>", "a", "ab", " ", "c", "!", "z", "b"])
    return donor, target


def _alignment() -> tuple[Any, _GreedyTokenizer, _GreedyTokenizer]:
    donor, target = _tokenizers()
    bridge = _bridge(donor, target)
    alignment = build_lfm2_semantic_span_cpcf_alignment(
        bridge,
        donor,
        target,
        parent_sampled_suffix_tokens={
            "prefill": 2,
            "decode.000": 1,
            "decode.001": 1,
        },
    )
    return alignment, donor, target


def test_fold_replay_ids_are_canonical_inference_ids_even_for_suffix_rewrite() -> None:
    alignment, donor, target = _alignment()
    bridge = _bridge(donor, target)

    for semantic_phase, aligned_phase in zip(
        bridge.phases,
        alignment.phases,
        strict=True,
    ):
        if semantic_phase.phase == "prefill":
            donor_ids = bridge.source_prompt_token_ids
        else:
            assert semantic_phase.parent_decode_step is not None
            donor_ids = (
                *bridge.source_prompt_token_ids,
                *bridge.source_forced_decode_token_ids[
                    : semantic_phase.parent_decode_step + 1
                ],
            )
        text = donor.decode(list(donor_ids))
        inference_ids = target.encode(text, add_special_tokens=False)
        replay = aligned_phase.canonical_target_replay_batch()

        assert replay["input_ids"].tolist() == [inference_ids]
        assert replay["attention_mask"].tolist() == [[1] * len(inference_ids)]
        assert replay["position_ids"].tolist() == [list(range(len(inference_ids)))]

    rewritten = alignment.get_phase("decode.000")
    assert rewritten.target_full_prefix_token_ids == (0, 2)
    assert rewritten.target_to_donor_boundary_positions == (0, 2)
    assert rewritten.omitted_donor_boundary_positions == (1,)
    assert rewritten.requires_full_prefix_replay is True
    with pytest.raises(
        Lfm2SemanticSpanCpcfAlignmentError,
        match="append-only KV-cache transition is invalid",
    ):
        rewritten.require_incremental_target_transition()


def test_contraction_gathers_exact_endpoint_rows_but_rejects_cached_attention() -> None:
    alignment, _donor, _target = _alignment()
    phase = alignment.get_phase("decode.000")
    history = torch.tensor(
        [[[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]]],
        dtype=torch.float64,
    )
    sampled = torch.tensor([[[7.0, 8.0]]], dtype=torch.float64)
    raw_scores = torch.tensor([[[[11.0, 12.0, 13.0]]]], dtype=torch.float64)

    assert phase.gather_parent_history_rows(history, sequence_axis=2).tolist() == [
        [[[1.0, 10.0], [3.0, 30.0]]]
    ]
    assert phase.gather_parent_sampled_rows(sampled, sequence_axis=1).tolist() == [
        [[7.0, 8.0]]
    ]
    assert phase.gather_parent_attention_matrix(raw_scores).tolist() == [
        [[[11.0, 13.0]]]
    ]
    assert phase.exact_endpoint_coverage is True
    assert phase.endpoint_row_gather_supported is True
    assert phase.exact_cached_attention_lattice is False
    assert phase.probability_preserving_attention_transport_supported is False
    assert phase.attention_transport_kind == "contraction-not-exact"
    with pytest.raises(
        Lfm2SemanticSpanCpcfAlignmentError,
        match="cached post-RoPE scores, probabilities, and context are not exact",
    ):
        phase.require_exact_cached_attention_lattice()


def test_expansion_without_parent_boundary_and_span_logits_fails_closed() -> None:
    alignment, _donor, _target = _alignment()
    phase = alignment.get_phase("decode.001")

    assert phase.target_full_prefix_token_ids == (0, 2, 3, 4)
    assert phase.target_to_donor_boundary_positions == (0, 2, None, 3)
    assert phase.target_to_donor_span_owner_positions == (0, 2, 3, 3)
    assert phase.donor_span_multiplicities == (1, 0, 1, 2)
    assert phase.missing_target_boundary_positions == (2,)
    assert phase.endpoint_row_gather_supported is False
    assert phase.probability_preserving_attention_transport_supported is False
    with pytest.raises(
        Lfm2SemanticSpanCpcfAlignmentError,
        match="no cached parent endpoint activation",
    ):
        phase.require_endpoint_row_gather_support()
    with pytest.raises(
        Lfm2SemanticSpanCpcfAlignmentError,
        match="intermediate target actions",
    ):
        phase.align_terminal_logits(torch.zeros((1, 1, 8)))
    assert phase.align_terminal_logits(
        torch.zeros((1, 1, 8)),
        require_every_target_action=False,
    ).shape == (1, 1, 8)


def test_identical_semantic_boundaries_allow_exact_cached_attention() -> None:
    tokenizer = _GreedyTokenizer(["a", "b", " "])
    bridge = _bridge(
        tokenizer,
        tokenizer,
        prompt=(0,),
        forced=(1,),
    )
    alignment = build_lfm2_semantic_span_cpcf_alignment(
        bridge,
        tokenizer,
        tokenizer,
        parent_sampled_suffix_tokens={"prefill": 1, "decode.000": 1},
    )

    assert all(phase.exact_cached_attention_lattice for phase in alignment.phases)
    assert all(not phase.requires_full_prefix_replay for phase in alignment.phases)
    alignment.get_phase("decode.000").require_exact_cached_attention_lattice()
    alignment.get_phase("decode.000").require_incremental_target_transition()
    alignment.get_phase("decode.000").require_every_target_action_logit()


def test_pure_span_expansion_duplicates_kv_rows_without_inventing_endpoints() -> None:
    donor = _GreedyTokenizer(["a", "bc", "x"])
    target = _GreedyTokenizer(["a", "b", "c", "x"])
    bridge = _bridge(donor, target, prompt=(0,), forced=(1,))
    alignment = build_lfm2_semantic_span_cpcf_alignment(
        bridge,
        donor,
        target,
        parent_sampled_suffix_tokens={"prefill": 1, "decode.000": 1},
    )
    phase = alignment.get_phase("decode.000")

    assert phase.target_to_donor_boundary_positions == (0, None, 1)
    assert phase.target_to_donor_span_owner_positions == (0, 1, 1)
    assert phase.donor_span_multiplicities == (1, 2)
    assert phase.target_span_sampled_suffix_start == 1
    assert phase.target_span_sampled_suffix_tokens == 2
    assert phase.parent_sampled_span_gather_indices == (0, 0)
    assert phase.exact_endpoint_coverage is False
    assert phase.probability_preserving_attention_transport_supported is True
    assert phase.attention_transport_kind == "probability-preserving-span-expansion"
    phase.require_probability_preserving_attention_transport()
    parent_kv = torch.tensor([[[10.0], [20.0]]], dtype=torch.float64)
    assert phase.expand_parent_history_rows(
        parent_kv,
        sequence_axis=1,
    ).tolist() == [[[10.0], [20.0], [20.0]]]
    assert phase.expand_parent_sampled_rows(
        parent_kv[:, -1:, :],
        sequence_axis=1,
    ).tolist() == [[[20.0], [20.0]]]
    with pytest.raises(
        Lfm2SemanticSpanCpcfAlignmentError,
        match="intermediate target actions",
    ):
        phase.require_every_target_action_logit()


def test_alignment_is_sealed_and_has_zero_model_or_training_operations() -> None:
    alignment, _donor, _target = _alignment()
    evidence = alignment.to_evidence_dict()

    assert evidence["coverage"] == {
        "all_target_endpoints_available": False,
        "all_cached_attention_lattices_exact": False,
        "all_probability_preserving_attention_transports_supported": False,
        "suffix_rewrite_phase_count": 1,
    }
    assert all(value == 0 for value in evidence["operation_audit"].values())
    assert evidence["alignment_sha256"] == alignment.alignment_sha256


def test_cached_multiplicities_preserve_legacy_evidence_bytes_and_seals() -> None:
    alignment, _donor, _target = _alignment()
    observed = alignment.to_evidence_dict()
    legacy = copy.deepcopy(observed)

    for phase, legacy_phase in zip(
        alignment.phases,
        legacy["phases"],
        strict=True,
    ):
        legacy_multiplicities = tuple(
            sum(
                owner == donor_index
                for owner in phase.target_to_donor_span_owner_positions
            )
            for donor_index in range(phase.donor_full_prefix_tokens)
        )
        assert phase.donor_span_multiplicities == legacy_multiplicities
        legacy_phase["donor_span_multiplicities"] = list(legacy_multiplicities)
        legacy_phase_body = dict(legacy_phase)
        legacy_phase_seal = legacy_phase_body.pop("alignment_sha256")
        assert canonical_sha256(legacy_phase_body) == legacy_phase_seal

    assert legacy == observed
    assert json.dumps(legacy, sort_keys=True, separators=(",", ":")) == json.dumps(
        observed,
        sort_keys=True,
        separators=(",", ":"),
    )
    legacy_body = dict(legacy)
    legacy_seal = legacy_body.pop("alignment_sha256")
    assert canonical_sha256(legacy_body) == legacy_seal
    assert legacy_seal == alignment.alignment_sha256


def test_multiplicities_are_computed_once_per_frozen_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = alignment_module._compute_donor_span_multiplicities
    calls = 0

    def counted(
        owners: tuple[int | None, ...],
        *,
        donor_count: int,
    ) -> tuple[int, ...]:
        nonlocal calls
        calls += 1
        return original(owners, donor_count=donor_count)

    monkeypatch.setattr(
        alignment_module,
        "_compute_donor_span_multiplicities",
        counted,
    )
    alignment, _donor, _target = _alignment()
    construction_calls = len(alignment.phases)
    assert calls == construction_calls

    for _ in range(3):
        for phase in alignment.phases:
            assert phase.donor_span_multiplicities
            _ = phase.probability_preserving_attention_transport_supported
            _ = phase.attention_transport_kind
            phase.to_evidence_dict()
        alignment.to_evidence_dict()
        audit_lfm2_semantic_span_cpcf_refinement(
            [alignment],
            expected_batch_count=1,
        )

    assert calls == construction_calls


def test_all_16_refinement_audit_is_explicit_long_horizon_go_no_go() -> None:
    donor = _GreedyTokenizer(["a", "bc", "x"])
    target = _GreedyTokenizer(["a", "b", "c", "x"])
    alignments = []
    for batch_index in range(16):
        bridge = _bridge(
            donor,
            target,
            prompt=(0,),
            forced=(1,),
            batch_index=batch_index,
        )
        alignments.append(
            build_lfm2_semantic_span_cpcf_alignment(
                bridge,
                donor,
                target,
                parent_sampled_suffix_tokens={"prefill": 1, "decode.000": 1},
            )
        )

    report = audit_lfm2_semantic_span_cpcf_refinement(alignments)

    assert report["observed_batch_count"] == 16
    assert report["phase_count"] == 32
    assert report["phase_transition_count"] == 16
    assert report["total_donor_node_occurrences"] == 48
    assert report["total_target_node_occurrences"] == 64
    assert report["maximum_target_nodes_per_donor_node"] == 2
    assert report["expanded_donor_node_occurrences"] == 16
    assert report["contracted_donor_node_occurrences"] == 0
    assert report["exact_endpoint_coverage"] == {
        "covered_target_node_occurrences": 48,
        "total_target_node_occurrences": 64,
        "rate": 0.75,
    }
    assert report["every_donor_node_owns_target_node"] is True
    assert report["no_contractions"] is True
    assert report["phase_boundaries_monotonic"] is True
    assert report["parent_attention_history_coverage"] == {
        "all_phase_histories_reconstructable": True,
        "missing_donor_node_occurrences": 0,
        "maximum_missing_donor_nodes_in_one_phase": 0,
        "final_phase_missing_donor_nodes_by_batch": [
            {"batch_index": index, "missing_donor_nodes": 0} for index in range(16)
        ],
        "rolling_suffixes_may_extend_but_not_repair_prefill": True,
    }
    assert report["probability_preserving_attention_transport_go"] is True
    assert report["blocking_reasons"] == []
    assert all(value == 0 for value in report["model_operations"].values())


def test_refinement_audit_blocks_contraction_and_wrong_batch_count() -> None:
    alignment, _donor, _target = _alignment()
    report = audit_lfm2_semantic_span_cpcf_refinement(
        [alignment],
        expected_batch_count=16,
    )

    assert report["contracted_donor_node_occurrences"] > 0
    assert report["probability_preserving_attention_transport_go"] is False
    assert report["blocking_reasons"] == [
        "expected-batch-count-not-covered",
        "donor-node-contraction",
        "nonmonotonic-target-phase-boundary",
    ]


def test_refinement_audit_fails_closed_on_truncated_prefill_history() -> None:
    tokenizer = _GreedyTokenizer(["a", "b", "c", "d", "e"])
    bridge = _bridge(
        tokenizer,
        tokenizer,
        prompt=(0, 1, 2),
        forced=(3, 4),
    )
    alignment = build_lfm2_semantic_span_cpcf_alignment(
        bridge,
        tokenizer,
        tokenizer,
        parent_sampled_suffix_tokens={
            "prefill": 1,
            "decode.000": 1,
            "decode.001": 1,
        },
    )

    report = audit_lfm2_semantic_span_cpcf_refinement(
        [alignment],
        expected_batch_count=1,
    )

    assert report["every_donor_node_owns_target_node"] is True
    assert report["phase_boundaries_monotonic"] is True
    assert report["parent_attention_history_coverage"] == {
        "all_phase_histories_reconstructable": False,
        "missing_donor_node_occurrences": 6,
        "maximum_missing_donor_nodes_in_one_phase": 2,
        "final_phase_missing_donor_nodes_by_batch": [
            {"batch_index": 0, "missing_donor_nodes": 2}
        ],
        "rolling_suffixes_may_extend_but_not_repair_prefill": True,
    }
    assert report["probability_preserving_attention_transport_go"] is False
    assert report["blocking_reasons"] == ["parent-attention-history-incomplete"]


def test_all_16_old_style_last8_prefills_are_an_explicit_no_go() -> None:
    tokenizer = _GreedyTokenizer([f"<{index:02d}>" for index in range(20)])
    alignments = []
    for batch_index in range(16):
        bridge = _bridge(
            tokenizer,
            tokenizer,
            prompt=tuple(range(18)),
            forced=(18, 19),
            batch_index=batch_index,
        )
        alignments.append(
            build_lfm2_semantic_span_cpcf_alignment(
                bridge,
                tokenizer,
                tokenizer,
                parent_sampled_suffix_tokens={
                    "prefill": 8,
                    "decode.000": 1,
                    "decode.001": 1,
                },
                parent_history_suffix_tokens={
                    "prefill": 8,
                    "decode.000": 8,
                    "decode.001": 8,
                },
            )
        )

    report = audit_lfm2_semantic_span_cpcf_refinement(alignments)

    assert report["observed_batch_count"] == 16
    assert report["every_donor_node_owns_target_node"] is True
    assert report["phase_boundaries_monotonic"] is True
    assert report["parent_attention_history_coverage"] == {
        "all_phase_histories_reconstructable": False,
        "missing_donor_node_occurrences": 480,
        "maximum_missing_donor_nodes_in_one_phase": 10,
        "final_phase_missing_donor_nodes_by_batch": [
            {"batch_index": index, "missing_donor_nodes": 10} for index in range(16)
        ],
        "rolling_suffixes_may_extend_but_not_repair_prefill": True,
    }
    assert report["probability_preserving_attention_transport_go"] is False
    assert report["blocking_reasons"] == ["parent-attention-history-incomplete"]
