from __future__ import annotations

import pytest

from neural_forge.dynamic_circuit_ir import (
    CANONICAL_MODALITIES,
    CANONICAL_MODALITY_ORDER,
    MODALITY_AUDIO,
    MODALITY_IMAGE,
    MODALITY_OCR,
    MODALITY_POINTS3D,
    MODALITY_REGISTRY_SCHEMA,
    MODALITY_TEXT,
    MODALITY_VIDEO,
    CircuitPort,
    DynamicCircuitIRError,
    ModalityInterface,
    canonical_modality,
    is_canonical_modality,
    modality_registry,
)
from neural_forge.hf_native_raw_tokens import (
    CANONICAL_RAW_TRANSPORT,
    LANE_COMPATIBILITY_SCHEMA,
    PROTOCOL_NAME as NATIVE_PROTOCOL,
    RawTensorEncodeError,
    VOCABULARY_EXPANSION_PROTOCOL,
    canonical_raw_modality,
    raw_token_lane_compatibility,
)
from neural_forge.raw_modality_tokens import (
    PROTOCOL_NAME as EXPANSION_PROTOCOL,
    RAW_TOKEN_STRINGS,
    RawModality,
    RawTokenEncodeError,
)
from neural_forge.raw_modality_tokens import (
    raw_token_lane_compatibility as reexported_compatibility,
)


def test_registry_is_frozen_ordered_and_self_describing() -> None:
    assert isinstance(CANONICAL_MODALITIES, frozenset)
    assert CANONICAL_MODALITIES == frozenset(CANONICAL_MODALITY_ORDER)
    assert CANONICAL_MODALITY_ORDER == (
        MODALITY_TEXT,
        MODALITY_IMAGE,
        MODALITY_AUDIO,
        MODALITY_VIDEO,
        MODALITY_OCR,
        MODALITY_POINTS3D,
    )
    registry = modality_registry()
    assert registry["schema"] == MODALITY_REGISTRY_SCHEMA
    assert registry["modalities"] == list(CANONICAL_MODALITY_ORDER)
    assert registry["ir_strings_remain_free_form"] is True
    assert registry["new_lanes_must_use_canonical_names"] is True
    assert set(registry["notes"]) == CANONICAL_MODALITIES
    # The 'ocr' collision between a codec modality and the NEO probe category
    # must stay written down rather than remembered.
    assert "category" in registry["notes"][MODALITY_OCR]
    assert registry["task_categories_are_not_modalities"] is True


def test_canonical_modality_validates_and_rejects_private_spellings() -> None:
    for name in CANONICAL_MODALITY_ORDER:
        assert canonical_modality(name) == name
        assert is_canonical_modality(name)
    assert canonical_modality("  audio  ") == MODALITY_AUDIO
    for value in ("Audio", "speech", "sound", "3d.point-cloud", "", None, 7):
        assert not is_canonical_modality(value)
        with pytest.raises(DynamicCircuitIRError):
            canonical_modality(value)


def test_ir_modality_strings_stay_free_form_for_compatibility() -> None:
    # Existing graphs keep validating: the registry is opt-in, not enforced.
    port = CircuitPort(name="pixel_values", kind="modality", modality="depth")
    assert port.modality == "depth"
    interface = ModalityInterface(
        interface_id="iface",
        modality="depth",
        direction="input",
        node_id="node",
        port_name="pixel_values",
        representation="raw",
    )
    assert interface.modality == "depth"
    assert not is_canonical_modality(interface.modality)


def test_raw_modality_enum_uses_the_canonical_registry_strings() -> None:
    assert {member.value for member in RawModality} <= CANONICAL_MODALITIES
    assert RawModality.AUDIO.value is MODALITY_AUDIO
    assert RawModality.IMAGE.value is MODALITY_IMAGE
    assert RawModality.POINTS3D.value is MODALITY_POINTS3D
    # 'text' is canonical and shared with the IR default modality.
    assert RawModality.coerce("text") is RawModality.TEXT
    with pytest.raises(RawTokenEncodeError):
        RawModality.coerce("speech")


def test_canonical_raw_modality_helper_is_opt_in_and_strict() -> None:
    assert canonical_raw_modality("audio") == MODALITY_AUDIO
    with pytest.raises(RawTensorEncodeError):
        canonical_raw_modality("3d.point-cloud")


def test_both_raw_protocols_quote_one_compatibility_statement() -> None:
    assert reexported_compatibility is raw_token_lane_compatibility
    statement = raw_token_lane_compatibility()
    assert statement["schema"] == LANE_COMPATIBILITY_SCHEMA
    assert statement["canonical_transport"] == CANONICAL_RAW_TRANSPORT
    assert CANONICAL_RAW_TRANSPORT == NATIVE_PROTOCOL
    assert VOCABULARY_EXPANSION_PROTOCOL == EXPANSION_PROTOCOL
    assert statement["canonical_modalities"] == list(CANONICAL_MODALITY_ORDER)
    native = statement["protocols"][NATIVE_PROTOCOL]
    expansion = statement["protocols"][EXPANSION_PROTOCOL]
    assert native["added_vocabulary_rows"] == 0
    assert native["changes_tokenizer_fingerprint"] is False
    assert expansion["added_vocabulary_rows"] == len(RAW_TOKEN_STRINGS) == 266
    assert expansion["changes_tokenizer_fingerprint"] is True


def test_fingerprint_equality_lanes_are_closed_to_the_expansion_protocol() -> None:
    statement = raw_token_lane_compatibility()
    matrix = statement["matrix"]
    assert set(matrix) == set(statement["lanes"])
    closed = [
        lane
        for lane, protocols in matrix.items()
        if not protocols[EXPANSION_PROTOCOL]["allowed"]
    ]
    assert set(closed) == {
        "parent_child_tokenizer_fingerprint_equality",
        "cached_parent_behavior_gate",
        "dynamic_fold_plan_execution",
    }
    assert all(
        protocols[NATIVE_PROTOCOL]["allowed"] for protocols in matrix.values()
    )
    assert matrix["vocabulary_expansion_research"][EXPANSION_PROTOCOL]["allowed"]
    for lane, protocols in matrix.items():
        for protocol, entry in protocols.items():
            assert entry["reason"].strip(), (lane, protocol)


def test_compatibility_statement_names_the_missing_bridge() -> None:
    bridge = raw_token_lane_compatibility()["bridge"]
    assert bridge["status"] == "absent"
    assert set(bridge["blocks"]) == {
        "parent_child_tokenizer_fingerprint_equality",
        "cached_parent_behavior_gate",
        "dynamic_fold_plan_execution",
    }
    assert len(bridge["requires"]) >= 3
