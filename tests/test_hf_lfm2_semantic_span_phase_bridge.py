from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest


torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    BATCH_SCHEMA,
    CACHE_SCHEMA,
    Lfm2CircuitTraceBatch,
    _tensor_descriptor,
)
from neural_forge.hf_lfm2_semantic_span_phase_bridge import (  # noqa: E402
    Lfm2SemanticSpanPhaseBridgeError,
    build_lfm2_semantic_span_phase_bridge,
)
from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (  # noqa: E402
    _cache_input_sha256,
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


class _BrokenRoundtripTokenizer(_GreedyTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        values = super().encode(text, add_special_tokens=add_special_tokens)
        return values[:-1] if len(values) > 1 else values


def _sealed(body: dict[str, Any], name: str) -> dict[str, Any]:
    result = dict(body)
    result[name] = canonical_sha256(result)
    return result


def _fixture(
    forced_token_ids: list[int] | None = None,
    *,
    position_origin: int | None = None,
) -> tuple[Lfm2CircuitTraceBatch, dict[str, Any]]:
    forced = [2, 3, 4] if forced_token_ids is None else forced_token_ids
    input_ids = torch.tensor([[0, 1]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    forced_ids = torch.tensor([forced], dtype=torch.long)
    tensors: dict[str, Any] = {
        "input.input_ids": input_ids,
        "input.attention_mask": attention_mask,
        "input.forced_decode_ids": forced_ids,
        "prefill.logits": torch.zeros((1, 1, 6)),
    }
    phase_diagnostics: dict[str, Any] = {"prefill": {}}
    for step, token_id in enumerate(forced):
        phase = f"decode.{step:03d}"
        tensors[f"{phase}.logits"] = torch.zeros((1, 1, 6))
        tensors[f"{phase}.forced_token_ids"] = torch.tensor([[token_id]])
        phase_diagnostics[phase] = {}
    position_ids = None
    if position_origin is not None:
        position_ids = torch.tensor(
            [[position_origin, position_origin + 1]],
            dtype=torch.long,
        )
        tensors["input.position_ids"] = position_ids
    input_sha256 = _cache_input_sha256(
        input_ids=input_ids,
        attention_mask=attention_mask,
        forced_decode_ids=forced_ids,
        position_ids=position_ids,
    )
    corpus_body = {
        "schema": "neural-forge-lfm2-circuit-corpus-v1",
        "batch_count": 1,
        "ordered_batch_input_sha256": [input_sha256],
        "prompt_token_counts": [2],
        "forced_decode_token_counts": [len(forced)],
    }
    corpus = {**corpus_body, "corpus_sha256": canonical_sha256(corpus_body)}
    implementation = {
        "implementation_sha256": canonical_sha256(
            {"fixture": "semantic-span-implementation"}
        )
    }
    topology = {
        "layer_topology_sha256": canonical_sha256({"fixture": "semantic-span-topology"})
    }
    capture = {
        "selection_sha256": canonical_sha256({"fixture": "semantic-span-capture"})
    }
    execution = {
        "execution_sha256": canonical_sha256({"fixture": "semantic-span-execution"})
    }
    runtime = {
        "input_device": "cpu",
        "network_access_performed": False,
        "local_loaded_parent_only": True,
    }
    parent = {"identity_sha256": canonical_sha256({"fixture": "semantic-span-parent"})}
    identity = {
        "schema": CACHE_SCHEMA,
        "parent_identity_sha256": parent["identity_sha256"],
        "corpus_sha256": corpus["corpus_sha256"],
        "implementation_sha256": implementation["implementation_sha256"],
        "layer_topology_sha256": topology["layer_topology_sha256"],
        "selection_sha256": capture["selection_sha256"],
        "execution_sha256": execution["execution_sha256"],
        "runtime": runtime,
    }
    cache_key = canonical_sha256(identity)
    manifest = _sealed(
        {
            "schema": CACHE_SCHEMA,
            "cache_key": cache_key,
            "parent": parent,
            "corpus": corpus,
            "implementation": implementation,
            "layer_topology": topology,
            "capture": capture,
            "runtime": runtime,
            "execution": execution,
            "storage": {"content_addressed": True},
        },
        "manifest_sha256",
    )
    inventory = {
        name: _tensor_descriptor(torch, value)
        for name, value in sorted(tensors.items())
    }
    record = _sealed(
        {
            "schema": BATCH_SCHEMA,
            "cache_key": cache_key,
            "batch_index": 0,
            "input_sha256": input_sha256,
            "tensor_count": len(inventory),
            "tensors": inventory,
            "capture": {
                "decode_steps": len(forced),
                "phase_diagnostics": phase_diagnostics,
            },
        },
        "record_sha256",
    )
    return (
        Lfm2CircuitTraceBatch(
            index=0,
            input_sha256=input_sha256,
            _tensors=MappingProxyType(tensors),
            _record=MappingProxyType(record),
        ),
        manifest,
    )


def _tokenizers() -> tuple[_GreedyTokenizer, _GreedyTokenizer]:
    donor = _GreedyTokenizer(["<B>", "a", "b", " c", "!", "z"])
    target = _GreedyTokenizer(["<B>", "a", "ab", " ", "c", "!", "z", "b"])
    return donor, target


def _build(
    batch: Lfm2CircuitTraceBatch,
    manifest: dict[str, Any],
    donor: Any,
    target: Any,
    **overrides: Any,
) -> Any:
    values = {
        "expected_source_cache_key": manifest["cache_key"],
        "expected_source_manifest_sha256": manifest["manifest_sha256"],
        "expected_source_input_sha256": batch.input_sha256,
        "expected_donor_tokenizer_fingerprint": (
            fingerprint_hf_tokenizer_semantics(donor)
        ),
        "expected_target_tokenizer_fingerprint": (
            fingerprint_hf_tokenizer_semantics(target)
        ),
    }
    values.update(overrides)
    return build_lfm2_semantic_span_phase_bridge(
        batch,
        manifest,
        donor,
        target,
        **values,
    )


def test_builds_exact_variable_span_schedule_and_preserves_parent_phases() -> None:
    batch, manifest = _fixture()
    donor, target = _tokenizers()
    bridge = _build(batch, manifest, donor, target)

    assert bridge.phase_order == (
        "prefill",
        "decode.000",
        "decode.001",
        "decode.002",
    )
    assert [phase.target_full_prefix_token_ids for phase in bridge.phases] == [
        (0, 1),
        (0, 2),
        (0, 2, 3, 4),
        (0, 2, 3, 4, 5),
    ]
    assert [phase.target_span_token_count for phase in bridge.phases[1:]] == [
        1,
        2,
        1,
    ]
    first_decode = bridge.get_phase("decode.000")
    assert first_decode.target_rewrite_start == 1
    assert first_decode.target_replaced_suffix_token_count == 1
    assert first_decode.donor_forced_token_id == 2
    assert first_decode.target_replacement_token_ids == (2,)
    assert first_decode.as_target_full_prefix_batch()["input_ids"].tolist() == [[0, 2]]
    assert (
        bridge.expansion_report["full_horizon"]["retokenized_suffix_phase_count"] == 1
    )
    assert [
        row["phase_count"] for row in bridge.expansion_report["horizon_buckets"]
    ] == [3, 0, 0]
    assert bridge.model_operation_audit == {
        "parent_model_calls": 0,
        "child_model_calls": 0,
        "model_forward_calls": 0,
        "model_parameters_read": 0,
        "model_parameters_written": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "pruned_parameters": 0,
        "quantized_parameters": 0,
    }
    assert bridge.to_evidence_dict()["parent_phase_identity_preserved"] is True
    bridge.verify(donor, target)


def test_exposes_independent_masks_positions_and_preserves_position_origin() -> None:
    batch, manifest = _fixture(position_origin=17)
    donor, target = _tokenizers()
    bridge = _build(batch, manifest, donor, target)
    phase = bridge.get_phase("decode.001")

    assert phase.target_attention_mask.tolist() == [[1, 1, 1, 1]]
    assert phase.target_position_ids.tolist() == [[17, 18, 19, 20]]
    returned = phase.target_input_ids
    returned.zero_()
    assert phase.target_input_ids.tolist() == [[0, 2, 3, 4]]
    assert phase.target_full_prefix_token_ids == (0, 2, 3, 4)


def test_horizon_bucket_boundaries_cover_every_parent_phase_once() -> None:
    batch, manifest = _fixture([2] * 33)
    donor, target = _tokenizers()
    bridge = _build(batch, manifest, donor, target)
    buckets = bridge.expansion_report["horizon_buckets"]

    assert [row["phase_count"] for row in buckets] == [8, 24, 1]
    assert buckets[0]["first_phase"] == "decode.000"
    assert buckets[0]["last_phase"] == "decode.007"
    assert buckets[1]["first_phase"] == "decode.008"
    assert buckets[1]["last_phase"] == "decode.031"
    assert buckets[2]["first_phase"] == "decode.032"
    assert bridge.parent_decode_phase_count == 33


@pytest.mark.parametrize(
    ("kind", "match"),
    [
        ("manifest", "manifest seal"),
        ("cache", "expected cache key"),
        ("input", "input tensors"),
        ("donor_fingerprint", "Donor tokenizer"),
        ("target_fingerprint", "Target tokenizer"),
    ],
)
def test_fails_closed_on_cache_input_or_tokenizer_authentication_tampering(
    kind: str,
    match: str,
) -> None:
    batch, manifest = _fixture()
    donor, target = _tokenizers()
    overrides: dict[str, Any] = {}
    if kind == "manifest":
        manifest["runtime"]["input_device"] = "tampered"
    elif kind == "cache":
        overrides["expected_source_cache_key"] = "a" * 64
    elif kind == "input":
        batch._tensors["input.input_ids"][0, 0] = 1
    elif kind == "donor_fingerprint":
        overrides["expected_donor_tokenizer_fingerprint"] = "b" * 64
    else:
        overrides["expected_target_tokenizer_fingerprint"] = "c" * 64

    with pytest.raises(Lfm2SemanticSpanPhaseBridgeError, match=match):
        _build(batch, manifest, donor, target, **overrides)


def test_rejects_non_exact_target_semantic_roundtrip() -> None:
    batch, manifest = _fixture()
    donor, _target = _tokenizers()
    broken = _BrokenRoundtripTokenizer(["<B>", "a", "ab", " ", "c", "!", "z", "b"])

    with pytest.raises(
        Lfm2SemanticSpanPhaseBridgeError,
        match="exact semantic roundtrip",
    ):
        _build(batch, manifest, donor, broken)


def test_verify_detects_live_tokenizer_drift() -> None:
    batch, manifest = _fixture()
    donor, target = _tokenizers()
    bridge = _build(batch, manifest, donor, target)
    changed_target = _GreedyTokenizer(["<B>", "a", "ab", " ", "c", "?", "z", "b"])

    with pytest.raises(
        Lfm2SemanticSpanPhaseBridgeError,
        match="Live tokenizer semantics differ",
    ):
        bridge.verify(donor, changed_target)
