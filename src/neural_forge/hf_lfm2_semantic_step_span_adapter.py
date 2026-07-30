"""One-time global semantic-span evidence compilation for LFM2 token bridges.

The strict tokenizer bridge is directed from target rows to donor rows.  A
semantic-step fold needs the inverse operation: one nonempty target token span
for every *defined* donor row.  This module compiles that inventory once,
globally, instead of repeating tokenizer work for every cached causal root.

Compilation is deliberately fail-closed.  Exact strict-bridge anchors are
inverted first.  Every remaining donor row is represented by its authenticated
raw ByteLevel token string and segmented through a deterministic trie/DP over
non-special target vocabulary strings.  The proof is exact token-code
concatenation; it never depends on singleton Unicode decoding, replacement
characters, normalization, or target re-encoding.  Text decoding belongs only
at the external whole-sequence boundary.  No model loading, model execution,
network access, training, pruning, or quantization is performed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_semantic_step_token_fold import (
    LFM2_SEMANTIC_STEP_SPAN_EVIDENCE_SCHEMA,
    seal_lfm2_semantic_step_span_evidence,
)
from neural_forge.hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    StrictTokenizerBridgeError,
)


LFM2_SEMANTIC_STEP_SPAN_ADAPTER_SCHEMA = (
    "neural-forge-lfm2-semantic-step-span-adapter-v2"
)
LFM2_SEMANTIC_STEP_SPAN_ADAPTER_PROGRESS_SCHEMA = (
    "neural-forge-lfm2-semantic-step-span-adapter-progress-v2"
)
LFM2_SEMANTIC_STEP_SPAN_ADAPTER_CACHE_ABI = (
    "neural-forge-lfm2-semantic-step-span-adapter-cache-abi-v2"
)

_ALGORITHM = "strict-anchor-inversion-then-raw-bytelevel-trie-dp-v2"
_SEGMENTATION_TIE_BREAK = "fewest-target-tokens-then-lexicographic-target-ids"
_MODEL_OPERATION_AUDIT = {
    "model_load_calls": 0,
    "model_forward_calls": 0,
    "parent_forward_calls": 0,
    "child_forward_calls": 0,
    "parent_generate_calls": 0,
    "parent_trace_recaptures": 0,
    "training_steps": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "autograd_grad_calls": 0,
    "backward_calls": 0,
    "pruned_parameters": 0,
    "quantized_parameters": 0,
    "network_access": False,
    "hf_hub_calls": 0,
}


class Lfm2SemanticStepSpanAdapterError(RuntimeError):
    """A global donor row cannot be proven to have one exact target code."""

    def __init__(
        self,
        message: str,
        *,
        classification: str = "adapter-contract-error",
        donor_ids: Sequence[int] = (),
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.classification = classification
        self.donor_ids = tuple(int(value) for value in donor_ids)
        self.diagnostics = MappingProxyType(
            {} if diagnostics is None else _plain_json(diagnostics)
        )


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise Lfm2SemanticStepSpanAdapterError(
        f"Compiler metadata contains non-JSON value {type(value).__name__}."
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticStepSpanAdapterError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2SemanticStepSpanAdapterError(f"{name} must be positive.")
    return int(value)


def _defined_row_count(tokenizer: Any, name: str) -> int:
    try:
        value = int(len(tokenizer))
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2SemanticStepSpanAdapterError(
            f"{name} tokenizer length must be a positive integer.",
            classification="invalid-tokenizer-capacity",
        ) from exc
    if value <= 0:
        raise Lfm2SemanticStepSpanAdapterError(
            f"{name} tokenizer length must be a positive integer.",
            classification="invalid-tokenizer-capacity",
        )
    return value


class _RawTokenTrieNode:
    __slots__ = ("children", "target_id")

    def __init__(self) -> None:
        self.children: dict[str, _RawTokenTrieNode] = {}
        self.target_id: int | None = None


def _special_token_ids(
    tokenizer: Any,
    *,
    defined_row_count: int,
    name: str,
) -> frozenset[int]:
    values: list[Any] = list(getattr(tokenizer, "all_special_ids", ()) or ())
    for role in ("bos", "eos", "pad", "unk"):
        value = getattr(tokenizer, f"{role}_token_id", None)
        if value is not None:
            values.append(value)
    result: set[int] = set()
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= defined_row_count
        ):
            raise Lfm2SemanticStepSpanAdapterError(
                f"{name} tokenizer has a special ID outside defined rows.",
                classification="invalid-tokenizer-special-id",
                diagnostics={"token_id": value, "defined_row_count": defined_row_count},
            )
        result.add(int(value))
    return frozenset(result)


def _vocabulary_rows(
    tokenizer: Any,
    *,
    defined_row_count: int,
    name: str,
    audit: dict[str, Any],
) -> tuple[str, ...]:
    getter = getattr(tokenizer, "get_vocab", None)
    if not callable(getter):
        raise Lfm2SemanticStepSpanAdapterError(
            f"{name} tokenizer must expose get_vocab().",
            classification="missing-tokenizer-method",
        )
    try:
        vocabulary = getter()
    except Exception as exc:
        raise Lfm2SemanticStepSpanAdapterError(
            f"{name} tokenizer get_vocab() failed.",
            classification="tokenizer-vocabulary-error",
            diagnostics={"error_type": type(exc).__name__},
        ) from exc
    audit[f"{name}_get_vocab_calls"] += 1
    audit["tokenizer_methods"].add(f"{name}.get_vocab")
    if not isinstance(vocabulary, Mapping) or len(vocabulary) != defined_row_count:
        raise Lfm2SemanticStepSpanAdapterError(
            f"{name} tokenizer vocabulary differs from its defined row count.",
            classification="invalid-tokenizer-vocabulary",
            diagnostics={
                "defined_row_count": defined_row_count,
                "observed_vocabulary_entries": (
                    len(vocabulary) if isinstance(vocabulary, Mapping) else None
                ),
            },
        )
    rows: list[str | None] = [None] * defined_row_count
    for raw_token, token_id in vocabulary.items():
        if not isinstance(raw_token, str):
            raise Lfm2SemanticStepSpanAdapterError(
                f"{name} tokenizer vocabulary contains a non-text token.",
                classification="invalid-tokenizer-vocabulary",
            )
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            or token_id >= defined_row_count
            or rows[token_id] is not None
        ):
            raise Lfm2SemanticStepSpanAdapterError(
                f"{name} tokenizer vocabulary IDs are not contiguous and unique.",
                classification="invalid-tokenizer-vocabulary",
            )
        rows[token_id] = raw_token
    if any(value is None for value in rows):
        raise Lfm2SemanticStepSpanAdapterError(
            f"{name} tokenizer vocabulary has undefined ID holes.",
            classification="invalid-tokenizer-vocabulary",
        )
    return tuple(value for value in rows if value is not None)


def _target_segmentation_trie(
    target_rows: Sequence[str],
    *,
    target_special_ids: frozenset[int],
) -> tuple[_RawTokenTrieNode, int, str]:
    root = _RawTokenTrieNode()
    inventory: list[dict[str, Any]] = []
    included = 0
    for target_id, raw_token in enumerate(target_rows):
        if target_id in target_special_ids:
            continue
        if not raw_token:
            raise Lfm2SemanticStepSpanAdapterError(
                f"Non-special target row {target_id} has an empty raw token string.",
                classification="invalid-target-raw-token",
                diagnostics={"target_id": target_id, "reason": "empty-raw-token"},
            )
        node = root
        for character in raw_token:
            node = node.children.setdefault(character, _RawTokenTrieNode())
        if node.target_id is not None:
            raise Lfm2SemanticStepSpanAdapterError(
                f"Target rows {node.target_id} and {target_id} have the same raw "
                "token string.",
                classification="ambiguous-target-raw-token",
                diagnostics={"target_ids": [node.target_id, target_id]},
            )
        node.target_id = target_id
        inventory.append(
            {
                "target_id": target_id,
                "raw_token_sha256": canonical_sha256(raw_token),
                "raw_token_character_count": len(raw_token),
            }
        )
        included += 1
    if included == 0:
        raise Lfm2SemanticStepSpanAdapterError(
            "Target tokenizer has no non-special raw token strings.",
            classification="empty-target-segmentation-vocabulary",
        )
    return root, included, canonical_sha256(inventory)


def _converted_donor_raw_token(
    tokenizer: Any,
    donor_id: int,
    *,
    expected_raw_token: str,
    audit: dict[str, Any],
) -> str:
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(converter):
        raise Lfm2SemanticStepSpanAdapterError(
            "Donor tokenizer must expose convert_ids_to_tokens().",
            classification="missing-tokenizer-method",
            donor_ids=(donor_id,),
        )
    try:
        value = converter(donor_id)
    except Exception as exc:
        raise Lfm2SemanticStepSpanAdapterError(
            f"Donor convert_ids_to_tokens failed for row {donor_id}.",
            classification="donor-raw-token-conversion-error",
            donor_ids=(donor_id,),
            diagnostics={"error_type": type(exc).__name__},
        ) from exc
    audit["donor_convert_ids_to_tokens_calls"] += 1
    audit["donor_rows_converted_to_raw_tokens"] += 1
    audit["tokenizer_methods"].add("donor.convert_ids_to_tokens")
    if not isinstance(value, str) or not value:
        raise Lfm2SemanticStepSpanAdapterError(
            f"Donor row {donor_id} has no nonempty raw token string.",
            classification="invalid-donor-raw-token",
            donor_ids=(donor_id,),
            diagnostics={
                "converted_type": type(value).__name__,
                "empty": value == "",
            },
        )
    if value != expected_raw_token:
        raise Lfm2SemanticStepSpanAdapterError(
            f"Donor row {donor_id} converter output differs from get_vocab().",
            classification="donor-raw-token-inventory-mismatch",
            donor_ids=(donor_id,),
            diagnostics={
                "converted_raw_token_sha256": canonical_sha256(value),
                "vocabulary_raw_token_sha256": canonical_sha256(expected_raw_token),
            },
        )
    return value


def _segment_raw_token_string(
    raw_token: str,
    trie: _RawTokenTrieNode,
    target_rows: Sequence[str],
    *,
    donor_id: int,
) -> tuple[int, ...]:
    # Dynamic programming is right-to-left.  Every candidate is ranked by the
    # fewest emitted target rows, then by the lexicographically smallest ID
    # sequence, so trie insertion order and chunking cannot change the result.
    best: list[tuple[int, ...] | None] = [None] * (len(raw_token) + 1)
    best[len(raw_token)] = ()
    for start in range(len(raw_token) - 1, -1, -1):
        node = trie
        selected: tuple[int, ...] | None = None
        for stop in range(start, len(raw_token)):
            node = node.children.get(raw_token[stop])
            if node is None:
                break
            if node.target_id is None or best[stop + 1] is None:
                continue
            candidate = (node.target_id, *best[stop + 1])
            if selected is None or (len(candidate), candidate) < (
                len(selected),
                selected,
            ):
                selected = candidate
        best[start] = selected
    result = best[0]
    if not result:
        raise Lfm2SemanticStepSpanAdapterError(
            f"Donor row {donor_id} raw token string cannot be segmented by "
            "non-special target vocabulary rows.",
            classification="unsegmentable-donor-raw-token",
            donor_ids=(donor_id,),
            diagnostics={
                "donor_raw_token_sha256": canonical_sha256(raw_token),
                "donor_raw_token_character_count": len(raw_token),
            },
        )
    reconstructed = "".join(target_rows[target_id] for target_id in result)
    if reconstructed != raw_token:
        raise Lfm2SemanticStepSpanAdapterError(
            f"Donor row {donor_id} raw target-code concatenation verification "
            "failed.",
            classification="raw-token-concatenation-mismatch",
            donor_ids=(donor_id,),
            diagnostics={
                "target_token_ids": list(result),
                "donor_raw_token_sha256": canonical_sha256(raw_token),
                "reconstructed_raw_token_sha256": canonical_sha256(reconstructed),
            },
        )
    return result


def _expected_digest(
    observed: str,
    expected: str | None,
    name: str,
) -> None:
    if expected is not None and observed != _digest(expected, name):
        raise Lfm2SemanticStepSpanAdapterError(
            f"{name} differs from the authenticated strict bridge.",
            classification="expected-fingerprint-mismatch",
            diagnostics={"name": name, "observed": observed, "expected": expected},
        )


def _cache_contract_body(
    bridge: StrictTokenizerBridgeArtifact,
) -> dict[str, Any]:
    return {
        "dependencies": {
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "bridge_profile_sha256": bridge.profile_fingerprint,
            "donor_tokenizer_semantics_sha256": (
                bridge.donor_tokenizer_semantics_sha256
            ),
            "target_tokenizer_semantics_sha256": (
                bridge.target_tokenizer_semantics_sha256
            ),
        },
        "numeric_contract": {
            "floating_point_operations": 0,
            "token_ids": "exact-nonnegative-integers",
            "composition_domain": "authenticated-raw-token-strings",
            "comparison": "exact-concatenation-no-normalization",
            "singleton_unicode_decode_dependency": False,
            "target_encode_dependency": False,
            "external_decode_boundary": "whole-target-sequence-only",
        },
        "abi": {
            "schema": LFM2_SEMANTIC_STEP_SPAN_ADAPTER_CACHE_ABI,
            "algorithm": _ALGORITHM,
            "span_evidence_schema": LFM2_SEMANTIC_STEP_SPAN_EVIDENCE_SCHEMA,
            "adapter_receipt_schema": LFM2_SEMANTIC_STEP_SPAN_ADAPTER_SCHEMA,
            "direct_anchor_policy": (
                "invert-every-defined-target-strict-anchor-fail-on-duplicates"
            ),
            "composed_policy": (
                "donor-raw-token-to-nonspecial-target-trie-dp-exact-concatenation"
            ),
            "segmentation_tie_break": _SEGMENTATION_TIE_BREAK,
        },
    }


def lfm2_semantic_step_span_adapter_cache_contract(
    bridge: StrictTokenizerBridgeArtifact,
) -> dict[str, Any]:
    """Return canonical phase-cache dependencies/numeric contract/ABI."""

    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    try:
        bridge.verify_files()
    except StrictTokenizerBridgeError as exc:
        raise Lfm2SemanticStepSpanAdapterError(
            "Strict tokenizer bridge files failed authentication.",
            classification="bridge-authentication-failed",
        ) from exc
    body = _cache_contract_body(bridge)
    return {**body, "contract_sha256": canonical_sha256(body)}


def _invert_direct_anchors(
    bridge: StrictTokenizerBridgeArtifact,
    *,
    donor_defined_row_count: int,
    target_defined_row_count: int,
) -> tuple[dict[int, tuple[int, str, tuple[str, ...]]], list[dict[str, Any]]]:
    profile = bridge.transport_profile
    candidates: dict[int, list[tuple[int, str, tuple[str, ...]]]] = {}
    for mapping in profile.row_mappings[:target_defined_row_count]:
        target_id = int(mapping.target_row_index)
        donor_id = int(mapping.donor_row_index)
        if target_id >= target_defined_row_count:
            raise Lfm2SemanticStepSpanAdapterError(
                "Strict bridge target mappings are not in defined-row order.",
                classification="bridge-profile-capacity-mismatch",
            )
        if donor_id >= donor_defined_row_count:
            raise Lfm2SemanticStepSpanAdapterError(
                f"Defined target row {target_id} anchors undefined donor row "
                f"{donor_id}.",
                classification="bridge-anchor-outside-defined-donor-rows",
                donor_ids=(donor_id,),
                diagnostics={
                    "target_id": target_id,
                    "donor_defined_row_count": donor_defined_row_count,
                },
            )
        candidates.setdefault(donor_id, []).append(
            (
                target_id,
                str(mapping.resolution),
                tuple(str(role) for role in mapping.special_token_roles),
            )
        )
    for donor_id in sorted(candidates):
        rows = candidates[donor_id]
        if len(rows) != 1:
            target_ids = tuple(row[0] for row in rows)
            raise Lfm2SemanticStepSpanAdapterError(
                f"Donor row {donor_id} has ambiguous direct strict-bridge "
                f"anchors {target_ids}.",
                classification="ambiguous-direct-anchor",
                donor_ids=(donor_id,),
                diagnostics={"target_ids": list(target_ids)},
            )
    direct = {donor_id: rows[0] for donor_id, rows in candidates.items()}
    records = [
        {
            "donor_id": donor_id,
            "target_id": direct[donor_id][0],
            "resolution": direct[donor_id][1],
            "special_token_roles": list(direct[donor_id][2]),
        }
        for donor_id in sorted(direct)
    ]
    return direct, records


def _semantic_tokenizer_audit() -> dict[str, Any]:
    return {
        "tokenizer_methods": set(),
        "donor_get_vocab_calls": 0,
        "target_get_vocab_calls": 0,
        "donor_convert_ids_to_tokens_calls": 0,
        "donor_rows_converted_to_raw_tokens": 0,
        "donor_decode_calls": 0,
        "target_decode_calls": 0,
        "target_encode_calls": 0,
        "unicode_normalization_calls": 0,
    }


def _authentication_methods(receipt: Mapping[str, Any]) -> list[str]:
    shapes = receipt.get("opening_runtime_shape")
    if not isinstance(shapes, Mapping):
        raise Lfm2SemanticStepSpanAdapterError(
            "Tokenizer live-session receipt lacks opening runtime shapes.",
            classification="invalid-live-session-receipt",
        )
    methods = {"bridge.authenticated_live_session"}
    for name in ("donor", "target"):
        shape = shapes.get(name)
        if not isinstance(shape, Mapping):
            raise Lfm2SemanticStepSpanAdapterError(
                "Tokenizer live-session receipt has invalid runtime shapes.",
                classification="invalid-live-session-receipt",
            )
        methods.update({f"{name}.get_vocab", f"{name}.decode"})
        if shape.get("id_converter_callable") is True:
            methods.add(f"{name}.convert_ids_to_tokens")
        if shape.get("backend_serializer_callable") is True:
            methods.add(f"{name}.backend.to_str")
    return sorted(methods)


@dataclass(frozen=True, slots=True)
class Lfm2SemanticStepSpanAdapterResult:
    """Sealed span evidence plus auditable global compiler metadata."""

    donor_target_spans: tuple[tuple[int, ...], ...]
    direct_anchor_donor_ids: tuple[int, ...]
    span_evidence: Mapping[str, Any]
    receipt: Mapping[str, Any]
    progress_records: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        spans = tuple(tuple(int(item) for item in row) for row in self.donor_target_spans)
        direct = tuple(int(item) for item in self.direct_anchor_donor_ids)
        if (
            not spans
            or any(not row for row in spans)
            or direct != tuple(sorted(set(direct)))
            or any(item < 0 or item >= len(spans) for item in direct)
        ):
            raise Lfm2SemanticStepSpanAdapterError(
                "Semantic-step adapter result row inventory is invalid."
            )
        evidence = _plain_json(self.span_evidence)
        receipt = _plain_json(self.receipt)
        progress = tuple(_plain_json(item) for item in self.progress_records)
        if (
            evidence.get("evidence_sha256") != receipt.get("span_evidence_sha256")
            or receipt.get("direct_anchor_donor_ids") != list(direct)
            or receipt.get("span_inventory_sha256")
            != canonical_sha256([list(row) for row in spans])
            or evidence.get("donor_row_count") != len(spans)
        ):
            raise Lfm2SemanticStepSpanAdapterError(
                "Semantic-step adapter result seals do not bind its row inventory."
            )
        supplied = receipt.pop("receipt_sha256", None)
        if supplied != canonical_sha256(receipt):
            raise Lfm2SemanticStepSpanAdapterError(
                "Semantic-step adapter receipt seal is invalid."
            )
        receipt["receipt_sha256"] = supplied
        object.__setattr__(self, "donor_target_spans", spans)
        object.__setattr__(self, "direct_anchor_donor_ids", direct)
        object.__setattr__(self, "span_evidence", _deep_freeze(evidence))
        object.__setattr__(self, "receipt", _deep_freeze(receipt))
        object.__setattr__(
            self,
            "progress_records",
            tuple(_deep_freeze(item) for item in progress),
        )

    @property
    def direct_anchor_mask(self) -> tuple[bool, ...]:
        """Return compact donor-order evidence for stronger solve weights."""

        direct = set(self.direct_anchor_donor_ids)
        return tuple(row_id in direct for row_id in range(len(self.donor_target_spans)))

    def as_dict(self) -> dict[str, Any]:
        return _plain_json(self.receipt)


def compile_lfm2_semantic_step_span_evidence_from_local_tokenizers(
    bridge: StrictTokenizerBridgeArtifact,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    *,
    maximum_donor_rows_per_chunk: int = 1024,
    selective_session_sha256: str | None = None,
    expected_bridge_artifact_sha256: str | None = None,
    expected_bridge_profile_sha256: str | None = None,
    expected_donor_tokenizer_semantics_sha256: str | None = None,
    expected_target_tokenizer_semantics_sha256: str | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> Lfm2SemanticStepSpanAdapterResult:
    """Compile and seal one target semantic code for every donor tokenizer row.

    Complexity is ``O(V_donor + V_target)`` for this adapter invocation.  The
    caller should cache the returned global evidence and reuse it for every
    causal root; no root collection is accepted by this API.
    """

    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise TypeError("bridge must be StrictTokenizerBridgeArtifact.")
    chunk_size = _positive_int(
        maximum_donor_rows_per_chunk,
        "maximum_donor_rows_per_chunk",
    )
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")
    _expected_digest(
        bridge.artifact_sha256,
        expected_bridge_artifact_sha256,
        "expected_bridge_artifact_sha256",
    )
    _expected_digest(
        bridge.profile_fingerprint,
        expected_bridge_profile_sha256,
        "expected_bridge_profile_sha256",
    )
    _expected_digest(
        bridge.donor_tokenizer_semantics_sha256,
        expected_donor_tokenizer_semantics_sha256,
        "expected_donor_tokenizer_semantics_sha256",
    )
    _expected_digest(
        bridge.target_tokenizer_semantics_sha256,
        expected_target_tokenizer_semantics_sha256,
        "expected_target_tokenizer_semantics_sha256",
    )
    donor_defined = _defined_row_count(donor_tokenizer, "donor")
    target_defined = _defined_row_count(target_tokenizer, "target")
    profile = bridge.transport_profile
    if (
        donor_defined != profile.donor_vocab_size
        or target_defined != profile.target_vocab_size
        or donor_defined > profile.donor_row_capacity
        or target_defined > profile.target_row_capacity
    ):
        raise Lfm2SemanticStepSpanAdapterError(
            "Live tokenizer defined rows differ from strict bridge capacities.",
            classification="bridge-profile-capacity-mismatch",
            diagnostics={
                "donor_defined_row_count": donor_defined,
                "target_defined_row_count": target_defined,
                "profile_donor_vocab_size": profile.donor_vocab_size,
                "profile_target_vocab_size": profile.target_vocab_size,
                "donor_row_capacity": profile.donor_row_capacity,
                "target_row_capacity": profile.target_row_capacity,
            },
        )

    cache_contract = _cache_contract_body(bridge)
    cache_contract = {
        **cache_contract,
        "contract_sha256": canonical_sha256(cache_contract),
    }
    session_sha256 = (
        canonical_sha256(
            {
                "schema": "neural-forge-lfm2-semantic-step-span-session-v2",
                "bridge_artifact_sha256": bridge.artifact_sha256,
                "bridge_profile_sha256": bridge.profile_fingerprint,
                "purpose": "one-time-global-semantic-step-span-compilation",
            }
        )
        if selective_session_sha256 is None
        else _digest(selective_session_sha256, "selective_session_sha256")
    )

    progress_records: list[Mapping[str, Any]] = []
    previous_progress_sha256: str | None = None
    processed_rows = 0

    def emit(stage: str, **details: Any) -> None:
        nonlocal previous_progress_sha256
        body = {
            "schema": LFM2_SEMANTIC_STEP_SPAN_ADAPTER_PROGRESS_SCHEMA,
            "ordinal": len(progress_records),
            "previous_progress_sha256": previous_progress_sha256,
            "stage": stage,
            "donor_rows_processed": processed_rows,
            "donor_defined_row_count": donor_defined,
            "target_defined_row_count": target_defined,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            **_plain_json(details),
        }
        seal = canonical_sha256(body)
        record = _deep_freeze({**body, "progress_sha256": seal})
        progress_records.append(record)
        previous_progress_sha256 = seal
        if progress_callback is not None:
            progress_callback(record)

    emit(
        "start",
        algorithm=_ALGORITHM,
        maximum_donor_rows_per_chunk=chunk_size,
    )
    spans: list[tuple[int, ...] | None] = [None] * donor_defined
    direct_records: list[dict[str, Any]] = []
    direct: dict[int, tuple[int, str, tuple[str, ...]]] = {}
    composition_records: list[dict[str, Any]] = []
    target_segmentation_row_count = 0
    target_segmentation_vocabulary_sha256 = ""
    audit = _semantic_tokenizer_audit()
    session_context = bridge.authenticated_live_session(
        donor_tokenizer,
        target_tokenizer,
        selective_session_sha256=session_sha256,
    )
    try:
        with session_context as session:
            session.verify_bound(
                donor_tokenizer,
                target_tokenizer,
                expected_donor_sha256=bridge.donor_tokenizer_semantics_sha256,
                expected_target_sha256=bridge.target_tokenizer_semantics_sha256,
            )
            direct, direct_records = _invert_direct_anchors(
                bridge,
                donor_defined_row_count=donor_defined,
                target_defined_row_count=target_defined,
            )
            for donor_id, (target_id, _resolution, _roles) in direct.items():
                spans[donor_id] = (target_id,)

            # The bridge session authenticates both live tokenizers before this
            # raw-code inventory is observed.  Direct anchors are already
            # complete and never enter the converter or segmentation path.
            donor_raw_rows = _vocabulary_rows(
                donor_tokenizer,
                defined_row_count=donor_defined,
                name="donor",
                audit=audit,
            )
            target_raw_rows = _vocabulary_rows(
                target_tokenizer,
                defined_row_count=target_defined,
                name="target",
                audit=audit,
            )
            target_special_ids = _special_token_ids(
                target_tokenizer,
                defined_row_count=target_defined,
                name="target",
            )
            (
                target_trie,
                target_segmentation_row_count,
                target_segmentation_vocabulary_sha256,
            ) = _target_segmentation_trie(
                target_raw_rows,
                target_special_ids=target_special_ids,
            )

            for start in range(0, donor_defined, chunk_size):
                stop = min(start + chunk_size, donor_defined)
                session.verify_bound(
                    donor_tokenizer,
                    target_tokenizer,
                    expected_donor_sha256=(
                        bridge.donor_tokenizer_semantics_sha256
                    ),
                    expected_target_sha256=(
                        bridge.target_tokenizer_semantics_sha256
                    ),
                )
                composed_ids = tuple(
                    donor_id
                    for donor_id in range(start, stop)
                    if donor_id not in direct
                )
                for donor_id in composed_ids:
                    raw_token = _converted_donor_raw_token(
                        donor_tokenizer,
                        donor_id,
                        expected_raw_token=donor_raw_rows[donor_id],
                        audit=audit,
                    )
                    code = _segment_raw_token_string(
                        raw_token,
                        target_trie,
                        target_raw_rows,
                        donor_id=donor_id,
                    )
                    spans[donor_id] = code
                    composition_records.append(
                        {
                            "donor_id": donor_id,
                            "donor_raw_token_sha256": canonical_sha256(raw_token),
                            "donor_raw_token_character_count": len(raw_token),
                            "target_token_ids": list(code),
                        }
                    )
                processed_rows = stop
                emit(
                    "chunk",
                    donor_row_start=start,
                    donor_row_stop=stop,
                    direct_anchor_rows_in_chunk=(stop - start - len(composed_ids)),
                    raw_segmented_rows_in_chunk=len(composed_ids),
                )
        session_receipt = session_context.as_dict()
    except StrictTokenizerBridgeError as exc:
        raise Lfm2SemanticStepSpanAdapterError(
            "Live donor/target tokenizers failed strict bridge authentication.",
            classification="bridge-authentication-failed",
        ) from exc

    if any(span is None for span in spans):
        missing = tuple(index for index, span in enumerate(spans) if span is None)
        raise Lfm2SemanticStepSpanAdapterError(
            f"Global semantic span inventory is missing donor rows {missing}.",
            classification="incomplete-donor-row-coverage",
            donor_ids=missing,
        )
    compiled_spans = tuple(span for span in spans if span is not None)
    direct_ids = tuple(sorted(direct))
    composed_ids = tuple(
        donor_id for donor_id in range(donor_defined) if donor_id not in direct
    )
    span_inventory_sha256 = canonical_sha256(
        [list(span) for span in compiled_spans]
    )
    direct_inventory_sha256 = canonical_sha256(direct_records)
    raw_composition_inventory_sha256 = canonical_sha256(composition_records)
    stable_lineage = {
        "schema": "neural-forge-lfm2-semantic-step-global-span-lineage-v2",
        "algorithm": _ALGORITHM,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "donor_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "tokenizer_live_session_sha256": session_receipt["receipt_sha256"],
        "selective_session_sha256": session_sha256,
        "donor_defined_row_count": donor_defined,
        "target_defined_row_count": target_defined,
        "donor_row_capacity": profile.donor_row_capacity,
        "target_row_capacity": profile.target_row_capacity,
        "direct_anchor_count": len(direct_ids),
        "composed_span_count": len(composed_ids),
        "direct_anchor_inventory_sha256": direct_inventory_sha256,
        "target_segmentation_vocabulary_sha256": (
            target_segmentation_vocabulary_sha256
        ),
        "raw_composition_inventory_sha256": raw_composition_inventory_sha256,
        "span_inventory_sha256": span_inventory_sha256,
        "composition_domain": "authenticated-raw-bytelevel-token-strings",
        "raw_token_concatenation_verified": True,
        "singleton_unicode_decode_dependency": False,
        "target_encode_dependency": False,
        "unicode_normalization_applied": False,
        "external_decode_boundary": "whole-target-sequence-only",
        "global_inventory_reused_across_causal_roots": True,
    }
    evidence = seal_lfm2_semantic_step_span_evidence(
        compiled_spans,
        target_row_count=profile.target_row_capacity,
        lineage=stable_lineage,
    )
    processed_rows = donor_defined
    emit(
        "final",
        span_evidence_sha256=evidence["evidence_sha256"],
        span_inventory_sha256=span_inventory_sha256,
        direct_anchor_count=len(direct_ids),
        composed_span_count=len(composed_ids),
        raw_composition_inventory_sha256=raw_composition_inventory_sha256,
    )

    compiler_methods = sorted(audit.pop("tokenizer_methods"))
    authentication_methods = _authentication_methods(session_receipt)
    receipt_body = {
        "schema": LFM2_SEMANTIC_STEP_SPAN_ADAPTER_SCHEMA,
        "status": "complete-one-global-span-for-every-defined-donor-row",
        "algorithm": _ALGORITHM,
        "span_evidence_sha256": evidence["evidence_sha256"],
        "span_evidence_schema": LFM2_SEMANTIC_STEP_SPAN_EVIDENCE_SCHEMA,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_profile_sha256": bridge.profile_fingerprint,
        "donor_tokenizer_semantics_sha256": (
            bridge.donor_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            bridge.target_tokenizer_semantics_sha256
        ),
        "selective_session_sha256": session_sha256,
        "tokenizer_live_session_sha256": session_receipt["receipt_sha256"],
        "tokenizer_live_session_receipt": session_receipt,
        "donor_defined_row_count": donor_defined,
        "target_defined_row_count": target_defined,
        "donor_row_capacity": profile.donor_row_capacity,
        "target_row_capacity": profile.target_row_capacity,
        "span_token_count": sum(len(span) for span in compiled_spans),
        "maximum_span_length": max(len(span) for span in compiled_spans),
        "span_inventory_sha256": span_inventory_sha256,
        "direct_anchor_count": len(direct_ids),
        "composed_span_count": len(composed_ids),
        "direct_anchor_donor_ids": list(direct_ids),
        "composed_span_donor_ids": list(composed_ids),
        "direct_anchor_records": direct_records,
        "direct_anchor_inventory_sha256": direct_inventory_sha256,
        "target_special_row_count_excluded_from_segmentation": (
            target_defined - target_segmentation_row_count
        ),
        "target_non_special_segmentation_row_count": (
            target_segmentation_row_count
        ),
        "target_segmentation_vocabulary_sha256": (
            target_segmentation_vocabulary_sha256
        ),
        "raw_composition_inventory_sha256": raw_composition_inventory_sha256,
        "segmentation_algorithm": "deterministic-raw-token-string-trie-dp",
        "segmentation_tie_break": _SEGMENTATION_TIE_BREAK,
        "direct_anchor_mask_available_on_result": True,
        "every_defined_donor_row_has_exactly_one_nonempty_span": True,
        "direct_anchors_bypassed_singleton_decode": True,
        "ambiguous_direct_anchors_accepted": False,
        "all_composed_spans_exactly_concatenate_to_donor_raw_token": True,
        "target_special_rows_used_for_composed_segmentation": False,
        "singleton_unicode_decode_used_for_composition": False,
        "singleton_unicode_decode_loss_can_change_composition": False,
        "target_encode_used_for_composition": False,
        "raw_unicode_equality_required": False,
        "unicode_normalization_applied": False,
        "internal_semantics": "analytic-target-token-code-spans",
        "external_decode_boundary": "whole-target-sequence-only",
        "piecewise_span_decode_is_semantic_evidence": False,
        "enumerated_per_causal_root": False,
        "global_donor_row_pass_count": 1,
        "asymptotic_tokenizer_work": (
            "O(V_donor + V_target + total_raw_token_characters)"
        ),
        "maximum_donor_rows_per_chunk": chunk_size,
        "chunking_changes_span_inventory_or_math": False,
        "progress_record_count": len(progress_records),
        "progress_final_sha256": previous_progress_sha256,
        "progress_records_sha256": canonical_sha256(
            [_plain_json(item) for item in progress_records]
        ),
        "semantic_compiler_tokenizer_methods": compiler_methods,
        "live_authentication_tokenizer_methods": authentication_methods,
        "tokenizer_methods": sorted(set(compiler_methods) | set(authentication_methods)),
        "semantic_compiler_tokenizer_operation_audit": audit,
        "cache_contract": cache_contract,
        "operation_audit": dict(_MODEL_OPERATION_AUDIT),
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": canonical_sha256(receipt_body),
    }
    return Lfm2SemanticStepSpanAdapterResult(
        donor_target_spans=compiled_spans,
        direct_anchor_donor_ids=direct_ids,
        span_evidence=evidence,
        receipt=receipt,
        progress_records=tuple(progress_records),
    )


__all__ = [
    "LFM2_SEMANTIC_STEP_SPAN_ADAPTER_CACHE_ABI",
    "LFM2_SEMANTIC_STEP_SPAN_ADAPTER_PROGRESS_SCHEMA",
    "LFM2_SEMANTIC_STEP_SPAN_ADAPTER_SCHEMA",
    "Lfm2SemanticStepSpanAdapterError",
    "Lfm2SemanticStepSpanAdapterResult",
    "compile_lfm2_semantic_step_span_evidence_from_local_tokenizers",
    "lfm2_semantic_step_span_adapter_cache_contract",
]
