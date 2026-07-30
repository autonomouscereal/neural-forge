from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_semantic_step_span_adapter import (
    LFM2_SEMANTIC_STEP_SPAN_ADAPTER_PROGRESS_SCHEMA,
    Lfm2SemanticStepSpanAdapterError,
    compile_lfm2_semantic_step_span_evidence_from_local_tokenizers,
    lfm2_semantic_step_span_adapter_cache_contract,
)
from neural_forge.hf_lfm2_semantic_step_token_fold import (
    LFM2_SEMANTIC_STEP_SPAN_EVIDENCE_SCHEMA,
)
from neural_forge.hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    seal_strict_tokenizer_bridge_artifact,
)


class _LocalTokenizer:
    def __init__(
        self,
        raw_tokens: Sequence[str],
        decoded_rows: Sequence[str],
        encodings: Mapping[str, Sequence[int]],
        *,
        eos_token_id: int = 0,
        forbidden_batch_decode_ids: Sequence[int] = (),
    ) -> None:
        assert len(raw_tokens) == len(decoded_rows)
        self.raw_tokens = tuple(raw_tokens)
        self.decoded_rows = list(decoded_rows)
        self.encodings = {
            text: tuple(int(item) for item in values)
            for text, values in encodings.items()
        }
        self.eos_token_id = eos_token_id
        self.bos_token_id = None
        self.pad_token_id = None
        self.unk_token_id = None
        self.all_special_ids = [eos_token_id]
        self.forbidden_batch_decode_ids = frozenset(forbidden_batch_decode_ids)
        self.decode_calls = 0
        self.encode_calls = 0
        self.get_vocab_calls = 0
        self.convert_ids_to_tokens_calls = 0
        self.converted_ids: list[int] = []
        self.batch_decode_rows: list[tuple[tuple[int, ...], ...]] = []
        self.batch_encode_texts: list[tuple[str, ...]] = []

    def __len__(self) -> int:
        return len(self.raw_tokens)

    def get_vocab(self) -> dict[str, int]:
        self.get_vocab_calls += 1
        return {token: token_id for token_id, token in enumerate(self.raw_tokens)}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        self.convert_ids_to_tokens_calls += 1
        self.converted_ids.append(token_id)
        return self.raw_tokens[token_id]

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        self.decode_calls += 1
        return "".join(self.decoded_rows[int(item)] for item in token_ids)

    def batch_decode(
        self,
        rows: Sequence[Sequence[int]],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> list[str]:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        normalized = tuple(tuple(int(item) for item in row) for row in rows)
        if any(
            item in self.forbidden_batch_decode_ids
            for row in normalized
            for item in row
        ):
            raise AssertionError("a direct strict anchor reached lossy batch decode")
        self.batch_decode_rows.append(normalized)
        return ["".join(self.decoded_rows[item] for item in row) for row in normalized]

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        assert add_special_tokens is False
        self.encode_calls += 1
        return list(self.encodings[text])

    def __call__(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = False,
        padding: bool = False,
        truncation: bool = False,
    ) -> dict[str, list[list[int]]]:
        assert add_special_tokens is False
        assert padding is False
        assert truncation is False
        values = tuple(texts)
        self.batch_encode_texts.append(values)
        return {"input_ids": [list(self.encodings[text]) for text in values]}

    def reset_observation_counters(self) -> None:
        self.decode_calls = 0
        self.encode_calls = 0
        self.get_vocab_calls = 0
        self.convert_ids_to_tokens_calls = 0
        self.converted_ids.clear()
        self.batch_decode_rows.clear()
        self.batch_encode_texts.clear()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _tokenizer_pair(
    *,
    lossy_composed_decode: bool = False,
    unsegmentable_composed: bool = False,
    target_roundtrip_mismatch: bool = False,
    ambiguous_target_anchor: bool = False,
) -> tuple[_LocalTokenizer, _LocalTokenizer]:
    donor_decoded = [
        "",
        "a",
        "b",
        "\ufffd" if lossy_composed_decode else "ab",
    ]
    donor_tokens = ["<eos>", "a", "b", "ab"]
    donor_encodings: dict[str, Sequence[int]] = {
        "": (),
        "a": (1,),
        "b": (2,),
        "ab": (3,),
    }
    if unsegmentable_composed:
        donor_decoded.append("c")
        donor_tokens.append("c")
        donor_encodings["c"] = (4,)
    donor = _LocalTokenizer(
        donor_tokens,
        donor_decoded,
        donor_encodings,
        forbidden_batch_decode_ids=(0, 1, 2),
    )
    target_tokens = ["<eos>", "a", "b"]
    target_decoded = ["", "a", "b"]
    if ambiguous_target_anchor:
        target_tokens.append("a-alias")
        target_decoded.append("a")
    target = _LocalTokenizer(
        target_tokens,
        target_decoded,
        {
            "": (),
            "a": (1,),
            "b": (2,),
            "ab": (1,) if target_roundtrip_mismatch else (1, 2),
        },
    )
    return donor, target


def _sealed_bridge(
    root: Path,
    donor: _LocalTokenizer,
    target: _LocalTokenizer,
) -> StrictTokenizerBridgeArtifact:
    root.mkdir()
    (root / "tokenizer.json").write_text(
        '{"local":"test-tokenizer"}\n', encoding="utf-8"
    )
    return seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        root,
        donor_row_capacity=len(donor),
        target_row_capacity=len(target),
    )


def test_global_adapter_uses_direct_anchors_and_composes_every_other_row(
    tmp_path: Path,
) -> None:
    donor, target = _tokenizer_pair()
    bridge = _sealed_bridge(tmp_path / "target", donor, target)
    donor.reset_observation_counters()
    target.reset_observation_counters()

    result = compile_lfm2_semantic_step_span_evidence_from_local_tokenizers(
        bridge,
        donor,
        target,
        maximum_donor_rows_per_chunk=4,
    )

    assert result.donor_target_spans == ((0,), (1,), (2,), (1, 2))
    assert result.direct_anchor_donor_ids == (0, 1, 2)
    assert result.direct_anchor_mask == (True, True, True, False)
    assert result.span_evidence["schema"] == LFM2_SEMANTIC_STEP_SPAN_EVIDENCE_SCHEMA
    assert result.span_evidence["donor_row_count"] == len(donor)
    assert result.span_evidence["all_spans_nonempty"] is True
    assert tuple(
        tuple(row["target_token_ids"])
        for row in result.span_evidence["spans"]
    ) == result.donor_target_spans
    assert result.receipt["direct_anchor_donor_ids"] == (0, 1, 2)
    assert result.receipt["direct_anchor_count"] == 3
    assert result.receipt["composed_span_count"] == 1
    assert result.receipt["enumerated_per_causal_root"] is False
    assert result.receipt["global_donor_row_pass_count"] == 1
    assert result.receipt["operation_audit"]["model_forward_calls"] == 0
    assert result.receipt["operation_audit"]["model_load_calls"] == 0
    assert result.receipt["operation_audit"]["network_access"] is False
    assert donor.batch_decode_rows == []
    assert target.batch_encode_texts == []
    assert target.batch_decode_rows == []
    audit = result.receipt["semantic_compiler_tokenizer_operation_audit"]
    assert audit["donor_convert_ids_to_tokens_calls"] == 1
    assert audit["donor_rows_converted_to_raw_tokens"] == 1
    assert audit["donor_decode_calls"] == 0
    assert audit["target_decode_calls"] == 0
    assert audit["target_encode_calls"] == 0
    assert result.receipt["raw_unicode_equality_required"] is False
    assert result.receipt["unicode_normalization_applied"] is False
    assert result.receipt["external_decode_boundary"] == (
        "whole-target-sequence-only"
    )
    assert result.receipt["internal_semantics"] == (
        "analytic-target-token-code-spans"
    )


def test_chunk_size_and_callback_do_not_change_semantic_inventory(
    tmp_path: Path,
) -> None:
    donor, target = _tokenizer_pair()
    bridge = _sealed_bridge(tmp_path / "target", donor, target)
    observed: list[Mapping[str, Any]] = []

    row_chunks = compile_lfm2_semantic_step_span_evidence_from_local_tokenizers(
        bridge,
        donor,
        target,
        maximum_donor_rows_per_chunk=1,
        progress_callback=observed.append,
    )
    one_chunk = compile_lfm2_semantic_step_span_evidence_from_local_tokenizers(
        bridge,
        donor,
        target,
        maximum_donor_rows_per_chunk=64,
    )

    assert row_chunks.donor_target_spans == one_chunk.donor_target_spans
    assert row_chunks.direct_anchor_donor_ids == one_chunk.direct_anchor_donor_ids
    assert _plain(row_chunks.span_evidence) == _plain(one_chunk.span_evidence)
    assert len(observed) == len(donor) + 2
    assert tuple(record["stage"] for record in observed) == (
        "start",
        "chunk",
        "chunk",
        "chunk",
        "chunk",
        "final",
    )


def test_progress_records_are_canonical_hash_chain(tmp_path: Path) -> None:
    donor, target = _tokenizer_pair()
    bridge = _sealed_bridge(tmp_path / "target", donor, target)
    observed: list[Mapping[str, Any]] = []

    result = compile_lfm2_semantic_step_span_evidence_from_local_tokenizers(
        bridge,
        donor,
        target,
        maximum_donor_rows_per_chunk=2,
        progress_callback=observed.append,
    )

    assert tuple(observed) == result.progress_records
    previous = None
    for ordinal, record in enumerate(observed):
        body = _plain(record)
        supplied = body.pop("progress_sha256")
        assert body["schema"] == LFM2_SEMANTIC_STEP_SPAN_ADAPTER_PROGRESS_SCHEMA
        assert body["ordinal"] == ordinal
        assert body["previous_progress_sha256"] == previous
        assert supplied == canonical_sha256(body)
        previous = supplied
    assert result.receipt["progress_final_sha256"] == previous


def test_composed_rows_are_converted_and_segmented_exactly_once(tmp_path: Path) -> None:
    donor, target = _tokenizer_pair()
    donor.raw_tokens = (*donor.raw_tokens, "ba", "aba")
    donor.decoded_rows.extend(("ba", "aba"))
    donor.encodings.update({"ba": (4,), "aba": (5,)})
    target.encodings.update({"ba": (2, 1), "aba": (1, 2, 1)})
    bridge = _sealed_bridge(tmp_path / "target", donor, target)
    donor.reset_observation_counters()
    target.reset_observation_counters()

    result = compile_lfm2_semantic_step_span_evidence_from_local_tokenizers(
        bridge,
        donor,
        target,
        maximum_donor_rows_per_chunk=2,
    )

    assert result.donor_target_spans[3:] == ((1, 2), (2, 1), (1, 2, 1))
    audit = result.receipt["semantic_compiler_tokenizer_operation_audit"]
    assert audit["donor_convert_ids_to_tokens_calls"] == 3
    assert audit["donor_rows_converted_to_raw_tokens"] == 3
    assert audit["donor_get_vocab_calls"] == 1
    assert audit["target_get_vocab_calls"] == 1
    assert audit["donor_decode_calls"] == 0
    assert audit["target_decode_calls"] == 0
    assert audit["target_encode_calls"] == 0
    assert donor.batch_decode_rows == []
    assert target.batch_encode_texts == []
    assert target.batch_decode_rows == []


@pytest.mark.parametrize("tamper", ["live-fingerprint", "sealed-file"])
def test_live_fingerprint_and_file_tampering_fail_closed(
    tmp_path: Path,
    tamper: str,
) -> None:
    donor, target = _tokenizer_pair()
    bridge = _sealed_bridge(tmp_path / "target", donor, target)
    if tamper == "live-fingerprint":
        donor.decoded_rows[3] = "changed"
    else:
        (bridge.root_dir / "tokenizer.json").write_text(
            '{"tampered":true}\n', encoding="utf-8"
        )

    with pytest.raises(
        Lfm2SemanticStepSpanAdapterError,
        match="failed strict bridge authentication",
    ) as raised:
        compile_lfm2_semantic_step_span_evidence_from_local_tokenizers(
            bridge,
            donor,
            target,
        )
    assert raised.value.classification == "bridge-authentication-failed"


def test_expected_fingerprint_and_cache_contract_are_authenticated(
    tmp_path: Path,
) -> None:
    donor, target = _tokenizer_pair()
    bridge = _sealed_bridge(tmp_path / "target", donor, target)
    contract = lfm2_semantic_step_span_adapter_cache_contract(bridge)
    body = dict(contract)
    supplied = body.pop("contract_sha256")
    assert supplied == canonical_sha256(body)
    assert contract["dependencies"]["bridge_artifact_sha256"] == (
        bridge.artifact_sha256
    )

    with pytest.raises(
        Lfm2SemanticStepSpanAdapterError,
        match="differs from the authenticated strict bridge",
    ) as raised:
        compile_lfm2_semantic_step_span_evidence_from_local_tokenizers(
            bridge,
            donor,
            target,
            expected_bridge_artifact_sha256="0" * 64,
        )
    assert raised.value.classification == "expected-fingerprint-mismatch"


def test_ambiguous_inverse_strict_anchor_fails_closed(tmp_path: Path) -> None:
    donor, target = _tokenizer_pair(ambiguous_target_anchor=True)
    bridge = _sealed_bridge(tmp_path / "target", donor, target)

    with pytest.raises(
        Lfm2SemanticStepSpanAdapterError,
        match="ambiguous direct strict-bridge anchors",
    ) as raised:
        compile_lfm2_semantic_step_span_evidence_from_local_tokenizers(
            bridge,
            donor,
            target,
        )
    assert raised.value.classification == "ambiguous-direct-anchor"
    assert raised.value.donor_ids == (1,)
    assert raised.value.diagnostics["target_ids"] == [1, 3]


def test_unsegmentable_raw_token_has_explicit_row_diagnostics(
    tmp_path: Path,
) -> None:
    donor, target = _tokenizer_pair(unsegmentable_composed=True)
    bridge = _sealed_bridge(tmp_path / "target", donor, target)

    with pytest.raises(
        Lfm2SemanticStepSpanAdapterError,
        match="cannot be segmented",
    ) as raised:
        compile_lfm2_semantic_step_span_evidence_from_local_tokenizers(
            bridge,
            donor,
            target,
            maximum_donor_rows_per_chunk=1,
        )
    assert raised.value.classification == "unsegmentable-donor-raw-token"
    assert raised.value.donor_ids == (4,)
    assert len(raised.value.diagnostics["donor_raw_token_sha256"]) == 64


def test_lossy_replacement_character_singleton_decode_is_rescued_by_raw_code(
    tmp_path: Path,
) -> None:
    donor, target = _tokenizer_pair(lossy_composed_decode=True)
    bridge = _sealed_bridge(tmp_path / "target", donor, target)

    result = compile_lfm2_semantic_step_span_evidence_from_local_tokenizers(
        bridge,
        donor,
        target,
    )
    assert donor.decode([3]) == "\ufffd"
    assert target.decode([1, 2]) == "ab"
    assert result.donor_target_spans[3] == (1, 2)
    assert result.receipt["singleton_unicode_decode_used_for_composition"] is False
    assert result.receipt["singleton_unicode_decode_loss_can_change_composition"] is False


def test_target_encoder_is_not_used_as_internal_code_semantics(
    tmp_path: Path,
) -> None:
    donor, target = _tokenizer_pair(target_roundtrip_mismatch=True)
    bridge = _sealed_bridge(tmp_path / "target", donor, target)

    target.reset_observation_counters()
    result = compile_lfm2_semantic_step_span_evidence_from_local_tokenizers(
        bridge,
        donor,
        target,
    )
    assert target.encode("ab", add_special_tokens=False) == [1]
    assert result.donor_target_spans[3] == (1, 2)
    assert result.receipt["target_encode_used_for_composition"] is False
    assert result.receipt["semantic_compiler_tokenizer_operation_audit"][
        "target_encode_calls"
    ] == 0
