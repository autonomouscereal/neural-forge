from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (  # noqa: E402
    Lfm2StrictTokenizerBridgeTraceError,
    Lfm2TargetTokenizedRecord,
    authenticate_lfm2_bridge_input_pair,
    build_lfm2_strict_tokenizer_bridge_trace_adapter,
    project_lfm2_cached_parent_logits,
)
from neural_forge.hf_token_align import StrictTokenRowMapping  # noqa: E402
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    StrictTokenizerBridgeArtifact,
    seal_strict_tokenizer_bridge_artifact,
)


class _Tokenizer:
    def __init__(
        self,
        surfaces: list[str],
        *,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        unk_token_id: int | None = None,
    ) -> None:
        self.surfaces = list(surfaces)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.all_special_ids = sorted(
            {
                value
                for value in (
                    bos_token_id,
                    eos_token_id,
                    pad_token_id,
                    unk_token_id,
                )
                if value is not None
            }
        )

    def __len__(self) -> int:
        return len(self.surfaces)

    def get_vocab(self) -> dict[str, int]:
        return {
            f"raw-{token_id}-{surface}": token_id
            for token_id, surface in enumerate(self.surfaces)
        }

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"raw-{token_id}-{self.surfaces[token_id]}"

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(self.surfaces[token_id] for token_id in token_ids)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        matches = [
            token_id
            for token_id, surface in enumerate(self.surfaces)
            if surface == text
        ]
        if len(matches) != 1:
            raise KeyError(text)
        return matches


def _target_files(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "test"}}),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({"clean_up_tokenization_spaces": False}),
        encoding="utf-8",
    )


def _cross_bridge(
    tmp_path: Path,
) -> StrictTokenizerBridgeArtifact:
    donor = _Tokenizer(
        ["<pad>", "<bos>", "A", "B", "<eos>", "C"],
        bos_token_id=1,
        eos_token_id=4,
        pad_token_id=0,
    )
    target = _Tokenizer(
        ["<bos>", "B", "<eos>", "<pad>"],
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=3,
    )
    root = tmp_path / "cross-target"
    _target_files(root)
    return seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        root,
        donor_row_capacity=8,
        target_row_capacity=4,
    )


def _cross_records() -> tuple[Lfm2TargetTokenizedRecord, ...]:
    return (
        Lfm2TargetTokenizedRecord(
            record_id="cal-0",
            split="calibration",
            input_ids=torch.tensor(
                [[0, 1, 2, 3], [0, 1, 3, 3]],
                dtype=torch.int32,
            ),
            attention_mask=torch.tensor(
                [[1, 1, 1, 0], [1, 1, 0, 0]],
                dtype=torch.bool,
            ),
            position_ids=torch.tensor(
                [[0, 1, 2, 0], [5, 6, 0, 0]],
                dtype=torch.int16,
            ),
            forced_decode_ids=torch.tensor([[1, 2], [2, 3]]),
        ),
        Lfm2TargetTokenizedRecord(
            record_id="gen-0",
            split="generation",
            input_ids=torch.tensor([[0, 1, 2]]),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            position_ids=torch.tensor([[10, 11, 12]]),
            forced_decode_ids=torch.tensor([[1, 2, 3]]),
        ),
    )


def test_maps_target_records_and_preserves_all_sequence_evidence(
    tmp_path: Path,
) -> None:
    bridge = _cross_bridge(tmp_path)
    assert bridge.transport_profile.target_to_donor_row_indices == (1, 3, 4, 0)
    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        bridge,
        _cross_records(),
        expected_artifact_sha256=bridge.artifact_sha256,
    )

    assert adapter.record_count == 2
    assert (
        adapter.donor_tokenizer_semantics_sha256
        == bridge.donor_tokenizer_semantics_sha256
    )
    assert (
        adapter.target_tokenizer_semantics_sha256
        == bridge.target_tokenizer_semantics_sha256
    )
    assert len(adapter.target_tokenizer_package_fingerprint) == 64
    assert [record.split for record in adapter.records] == [
        "calibration",
        "generation",
    ]
    assert adapter.records_for_split("generation")[0].record_id == "gen-0"
    record = adapter.get_record("cal-0")
    assert torch.equal(
        record.donor_input_ids,
        torch.tensor([[1, 3, 4, 0], [1, 3, 0, 0]]),
    )
    assert torch.equal(
        record.donor_forced_decode_ids,
        torch.tensor([[3, 4], [4, 0]]),
    )
    assert torch.equal(
        record.attention_mask,
        torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
    )
    assert torch.equal(
        record.position_ids,
        torch.tensor([[0, 1, 2, 0], [5, 6, 0, 0]]),
    )
    assert record.special_token_role_counts == (
        ("BOS", 2),
        ("EOS", 3),
        ("PAD", 4),
    )
    assert record.numerically_different_token_count == 12
    assert record.to_evidence_dict()["numeric_token_id_equality_required"] is False
    assert adapter.to_evidence_dict()["model_operations"] == {
        "parent_model_calls": 0,
        "child_model_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "pruned_parameters": 0,
        "quantized_parameters": 0,
    }
    adapter.verify()


def test_actual_parent_child_pair_uses_mapping_not_numeric_id_equality(
    tmp_path: Path,
) -> None:
    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        _cross_bridge(tmp_path),
        _cross_records(),
    )
    record = adapter.get_record("cal-0")
    parent = record.as_parent_cache_batch()
    child = record.as_child_replay_batch()
    assert not torch.equal(parent["input_ids"], child["input_ids"])

    evidence = authenticate_lfm2_bridge_input_pair(
        adapter,
        record_id="cal-0",
        parent_cache_batch=parent,
        child_replay_batch=child,
    )

    assert evidence.numeric_token_id_equality_required is False
    assert evidence.numerically_different_token_count == 12
    assert evidence.masks_equal
    assert evidence.positions_equal
    assert evidence.forced_decode_mapped
    assert evidence.donor_cache_input_sha256 == record.donor_cache_input_sha256
    assert evidence.target_cache_input_sha256 == record.target_cache_input_sha256


@pytest.mark.parametrize(
    ("side", "field", "index"),
    [
        ("parent", "input_ids", (0, 1)),
        ("child", "attention_mask", (0, 2)),
        ("child", "position_ids", (0, 1)),
        ("parent", "forced_decode_ids", (1, 0)),
    ],
)
def test_pair_authentication_fails_closed_on_any_boundary_tampering(
    tmp_path: Path,
    side: str,
    field: str,
    index: tuple[int, int],
) -> None:
    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        _cross_bridge(tmp_path),
        _cross_records(),
    )
    record = adapter.get_record("cal-0")
    parent = record.as_parent_cache_batch()
    child = record.as_child_replay_batch()
    selected = parent if side == "parent" else child
    if field == "attention_mask":
        selected[field][index] = 0
    else:
        selected[field][index] += 1

    with pytest.raises(
        Lfm2StrictTokenizerBridgeTraceError,
        match="differ|do not satisfy",
    ):
        authenticate_lfm2_bridge_input_pair(
            adapter,
            record_id="cal-0",
            parent_cache_batch=parent,
            child_replay_batch=child,
        )


def test_padded_physical_vocabulary_rows_are_mapped_and_projected(
    tmp_path: Path,
) -> None:
    tokenizer = _Tokenizer(["zero", "one", "two", "three"])
    root = tmp_path / "padded-target"
    _target_files(root)
    bridge = seal_strict_tokenizer_bridge_artifact(
        tokenizer,
        tokenizer,
        root,
        donor_row_capacity=6,
        target_row_capacity=6,
    )
    record = Lfm2TargetTokenizedRecord(
        record_id="padded",
        split="calibration",
        input_ids=torch.tensor([[0, 4, 5]]),
        attention_mask=torch.tensor([[1, 1, 0]]),
        forced_decode_ids=torch.tensor([[5, 2]]),
    )
    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(bridge, (record,))
    aligned = adapter.get_record("padded")
    assert adapter.mapping == (0, 1, 2, 3, 4, 5)
    assert aligned.physical_padding_token_count == 3
    assert torch.equal(aligned.target_input_ids, aligned.donor_input_ids)

    logits = torch.tensor([[[10.0, 9.0, 8.0, 7.0, 6.0, 5.0]]])
    projected = project_lfm2_cached_parent_logits(
        adapter,
        record_id="padded",
        parent_cache_input_sha256=aligned.donor_cache_input_sha256,
        cache_key="a" * 64,
        cache_batch_index=0,
        phase="prefill",
        donor_logits=logits,
        top_k=6,
    )
    assert torch.equal(projected.target_logits, logits)
    assert projected.topk_target_ids.tolist() == [[[0, 1, 2, 3, 4, 5]]]
    assert projected.winner_to_topk_margins.tolist() == [
        [[1.0, 2.0, 3.0, 4.0, 5.0]]
    ]


def test_projects_full_cached_logits_then_computes_target_topk_margins(
    tmp_path: Path,
) -> None:
    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        _cross_bridge(tmp_path),
        _cross_records(),
    )
    record = adapter.get_record("cal-0")
    logits = torch.tensor(
        [
            [[0.0, 5.0, -1.0, 9.0, 7.0, 2.0, 100.0, 50.0]],
            [[1.0, 4.0, -2.0, 4.0, 3.0, 0.0, 99.0, 98.0]],
        ],
        dtype=torch.float32,
    )

    result = project_lfm2_cached_parent_logits(
        adapter,
        record_id="cal-0",
        parent_cache_input_sha256=record.donor_cache_input_sha256,
        cache_key="b" * 64,
        cache_batch_index=0,
        phase="decode.001",
        donor_logits=logits,
        top_k=4,
    )

    expected = logits[..., [1, 3, 4, 0]]
    assert torch.equal(result.target_logits, expected)
    assert result.topk_target_ids.tolist() == [
        [[1, 2, 0, 3]],
        [[0, 1, 2, 3]],
    ]
    assert result.topk_values.tolist() == [
        [[9.0, 7.0, 5.0, 0.0]],
        [[4.0, 4.0, 3.0, 1.0]],
    ]
    assert result.winner_to_topk_margins.tolist() == [
        [[2.0, 4.0, 9.0]],
        [[0.0, 1.0, 3.0]],
    ]
    assert result.winner_margin.tolist() == [[2.0], [0.0]]
    assert (
        result.to_evidence_dict()["projection_contract"]["tie_break"]
        == "lower-target-row-first"
    )
    result.verify(adapter)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("parent_cache_input_sha256", "c" * 64, "not the mapped"),
        ("cache_batch_index", 1, "record ordering"),
        ("phase", "decode.999", "exceeds"),
        ("top_k", 1, "at least two"),
    ],
)
def test_logit_projection_rejects_wrong_cache_binding(
    tmp_path: Path,
    field: str,
    value: Any,
    match: str,
) -> None:
    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        _cross_bridge(tmp_path),
        _cross_records(),
    )
    record = adapter.get_record("cal-0")
    kwargs = {
        "record_id": "cal-0",
        "parent_cache_input_sha256": record.donor_cache_input_sha256,
        "cache_key": "d" * 64,
        "cache_batch_index": 0,
        "phase": "decode.000",
        "donor_logits": torch.zeros((2, 1, 8)),
        "top_k": 4,
    }
    kwargs[field] = value
    with pytest.raises(Lfm2StrictTokenizerBridgeTraceError, match=match):
        project_lfm2_cached_parent_logits(adapter, **kwargs)


def test_logit_projection_rejects_truncated_or_nonfinite_donor_columns(
    tmp_path: Path,
) -> None:
    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        _cross_bridge(tmp_path),
        _cross_records(),
    )
    record = adapter.get_record("cal-0")
    kwargs = {
        "record_id": "cal-0",
        "parent_cache_input_sha256": record.donor_cache_input_sha256,
        "cache_key": "e" * 64,
        "cache_batch_index": 0,
        "phase": "decode.000",
        "top_k": 4,
    }
    with pytest.raises(
        Lfm2StrictTokenizerBridgeTraceError,
        match="donor row capacity",
    ):
        project_lfm2_cached_parent_logits(
            adapter,
            donor_logits=torch.zeros((2, 1, 7)),
            **kwargs,
        )
    corrupt = torch.zeros((2, 1, 8))
    corrupt[0, 0, 0] = torch.nan
    with pytest.raises(
        Lfm2StrictTokenizerBridgeTraceError,
        match="NaN or infinity",
    ):
        project_lfm2_cached_parent_logits(
            adapter,
            donor_logits=corrupt,
            **kwargs,
        )
    prefill_kwargs = dict(kwargs)
    prefill_kwargs["phase"] = "prefill"
    with pytest.raises(
        Lfm2StrictTokenizerBridgeTraceError,
        match="prompt boundary",
    ):
        project_lfm2_cached_parent_logits(
            adapter,
            donor_logits=torch.zeros((2, 5, 8)),
            **prefill_kwargs,
        )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def test_rejects_span_only_row_even_inside_self_consistent_artifact(
    tmp_path: Path,
) -> None:
    base = _cross_bridge(tmp_path)
    rows = list(base.transport_profile.row_mappings)
    rows[1] = StrictTokenRowMapping(
        target_row_index=rows[1].target_row_index,
        donor_row_index=rows[1].donor_row_index,
        resolution="utf8-byte-span",
        special_token_roles=rows[1].special_token_roles,
    )
    profile = dataclasses.replace(
        base.transport_profile,
        row_mappings=tuple(rows),
    )
    body = base._body()
    body["profile_fingerprint"] = profile.fingerprint()
    body["transport_profile"] = profile.to_dict()
    span_artifact = StrictTokenizerBridgeArtifact(
        root_dir=base.root_dir,
        transport_profile=profile,
        donor_tokenizer_semantics_sha256=(
            base.donor_tokenizer_semantics_sha256
        ),
        target_tokenizer_semantics_sha256=(
            base.target_tokenizer_semantics_sha256
        ),
        target_files=base.target_files,
        artifact_sha256=_canonical_sha256(body),
    )

    with pytest.raises(
        Lfm2StrictTokenizerBridgeTraceError,
        match="one-row donor resolution",
    ):
        build_lfm2_strict_tokenizer_bridge_trace_adapter(
            span_artifact,
            _cross_records(),
        )


def test_adapter_and_projection_detach_all_mutable_caller_tensors(
    tmp_path: Path,
) -> None:
    input_ids = torch.tensor([[0, 1, 2]])
    mask = torch.ones_like(input_ids)
    positions = torch.tensor([[7, 8, 9]])
    forced = torch.tensor([[1, 2]])
    source = Lfm2TargetTokenizedRecord(
        record_id="immutable",
        split="holdout",
        input_ids=input_ids,
        attention_mask=mask,
        position_ids=positions,
        forced_decode_ids=forced,
    )
    input_ids.fill_(3)
    mask.zero_()
    positions.zero_()
    forced.zero_()
    assert source.input_ids.tolist() == [[0, 1, 2]]
    assert source.attention_mask.tolist() == [[1, 1, 1]]
    assert source.position_ids.tolist() == [[7, 8, 9]]
    assert source.forced_decode_ids.tolist() == [[1, 2]]

    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        _cross_bridge(tmp_path),
        (source,),
    )
    record = adapter.get_record("immutable")
    returned = record.as_parent_cache_batch()
    returned["input_ids"].zero_()
    returned["attention_mask"].zero_()
    record.target_input_ids.zero_()
    assert record.donor_input_ids.tolist() == [[1, 3, 4]]
    assert record.attention_mask.tolist() == [[1, 1, 1]]
    adapter.verify()

    logits = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    projection = project_lfm2_cached_parent_logits(
        adapter,
        record_id="immutable",
        parent_cache_input_sha256=record.donor_cache_input_sha256,
        cache_key="f" * 64,
        cache_batch_index=0,
        phase="decode.000",
        donor_logits=logits,
        top_k=4,
    )
    expected = projection.target_logits
    logits.zero_()
    projection.target_logits.zero_()
    assert torch.equal(projection.target_logits, expected)
    projection.verify(adapter)


def test_private_tensor_tampering_is_detected_by_evidence_seals(
    tmp_path: Path,
) -> None:
    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        _cross_bridge(tmp_path),
        _cross_records(),
    )
    record = adapter.get_record("cal-0")
    projection = project_lfm2_cached_parent_logits(
        adapter,
        record_id="cal-0",
        parent_cache_input_sha256=record.donor_cache_input_sha256,
        cache_key="1" * 64,
        cache_batch_index=0,
        phase="decode.000",
        donor_logits=torch.zeros((2, 1, 8)),
        top_k=4,
    )
    projection._target_logits[0, 0, 0] = 1.0
    with pytest.raises(
        Lfm2StrictTokenizerBridgeTraceError,
        match="seal is invalid",
    ):
        projection.verify(adapter)

    second = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        _cross_bridge(tmp_path / "second"),
        _cross_records(),
    )
    second.get_record("cal-0")._donor_input_ids[0, 0] = 0
    with pytest.raises(
        Lfm2StrictTokenizerBridgeTraceError,
        match="no longer follows",
    ):
        second.verify()


def test_adapter_reauthenticates_bridge_files_and_rejects_duplicate_records(
    tmp_path: Path,
) -> None:
    bridge = _cross_bridge(tmp_path)
    records = _cross_records()
    with pytest.raises(
        Lfm2StrictTokenizerBridgeTraceError,
        match="unique",
    ):
        build_lfm2_strict_tokenizer_bridge_trace_adapter(
            bridge,
            (records[0], records[0]),
        )

    adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        bridge,
        records,
    )
    (bridge.root_dir / "tokenizer.json").write_text(
        "tampered",
        encoding="utf-8",
    )
    with pytest.raises(
        Lfm2StrictTokenizerBridgeTraceError,
        match="file authentication",
    ):
        adapter.verify()


def test_record_validation_rejects_capacity_and_shape_violations(
    tmp_path: Path,
) -> None:
    bridge = _cross_bridge(tmp_path)
    out_of_capacity = Lfm2TargetTokenizedRecord(
        record_id="bad-id",
        split="calibration",
        input_ids=torch.tensor([[0, 4]]),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        forced_decode_ids=torch.tensor([[1]]),
    )
    with pytest.raises(
        Lfm2StrictTokenizerBridgeTraceError,
        match="outside target row capacity",
    ):
        build_lfm2_strict_tokenizer_bridge_trace_adapter(
            bridge,
            (out_of_capacity,),
        )
    with pytest.raises(
        Lfm2StrictTokenizerBridgeTraceError,
        match="batch boundary",
    ):
        Lfm2TargetTokenizedRecord(
            record_id="bad-batch",
            split="generation",
            input_ids=torch.tensor([[0, 1]]),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            forced_decode_ids=torch.tensor([[1], [2]]),
        )
    with pytest.raises(
        Lfm2StrictTokenizerBridgeTraceError,
        match="at least one active",
    ):
        Lfm2TargetTokenizedRecord(
            record_id="bad-mask",
            split="generation",
            input_ids=torch.tensor([[0, 1]]),
            attention_mask=torch.zeros((1, 2), dtype=torch.long),
            forced_decode_ids=torch.tensor([[1]]),
        )
