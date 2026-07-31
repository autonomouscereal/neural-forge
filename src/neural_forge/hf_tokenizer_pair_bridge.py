"""Sealed byte-exact code bridge between two independent ByteLevel vocabularies.

The strict bridge in :mod:`neural_forge.hf_tokenizer_bridge` manufactures its
child vocabulary as a subset of the donor's rows, so every target row *is* a
donor row.  Cross-family folding has no such luxury: a Qwen3 parent and an
LFM2-shaped child carry two pre-existing, unrelated vocabularies and no row of
one is a row of the other.  This module compiles the alignment that is still
exact for that case.  Every defined row of one tokenizer is segmented into a
nonempty ordered sequence of rows of the other -- its *code* -- over raw
ByteLevel token strings, in both directions.  Singleton transport is not
attempted: it was proven lossy on the LFM2 lane, and the multi-code readout is
a downstream concern.

This module reads tokenizer vocabularies only.  It never reads, writes, prunes,
quantizes, randomizes, or trains a model parameter.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from neural_forge.hf_tokenizer_bridge import fingerprint_hf_tokenizer_semantics


TOKENIZER_PAIR_BRIDGE_FILENAME = "neural_forge_tokenizer_pair_bridge.json"
TOKENIZER_PAIR_BRIDGE_CODES_FILENAME = (
    "neural_forge_tokenizer_pair_bridge_codes.safetensors"
)
TOKENIZER_PAIR_BRIDGE_SCHEMA = "neural-forge-tokenizer-pair-bridge-v1"
TOKENIZER_PAIR_BRIDGE_KIND = "sealed-independent-tokenizer-pair-bridge"
TOKENIZER_PAIR_TRANSPORT_SCHEMA = "neural-forge-tokenizer-pair-transport-evidence-v1"
TOKENIZER_PAIR_INCIDENCE_SCHEMA = "neural-forge-tokenizer-pair-row-incidence-v1"

PAIR_BRIDGE_ARTIFACT_CONFIG_KEY = "_neural_forge_tokenizer_pair_bridge_artifact_sha256"
PAIR_BRIDGE_PROFILE_CONFIG_KEY = "_neural_forge_tokenizer_pair_profile_sha256"
PAIR_SOURCE_TOKENIZER_SEMANTICS_CONFIG_KEY = (
    "_neural_forge_pair_source_tokenizer_semantics_sha256"
)
PAIR_TARGET_TOKENIZER_SEMANTICS_CONFIG_KEY = (
    "_neural_forge_pair_target_tokenizer_semantics_sha256"
)
TOKENIZER_PAIR_BRIDGE_CONFIG_KEYS = frozenset(
    {
        PAIR_BRIDGE_ARTIFACT_CONFIG_KEY,
        PAIR_BRIDGE_PROFILE_CONFIG_KEY,
        PAIR_SOURCE_TOKENIZER_SEMANTICS_CONFIG_KEY,
        PAIR_TARGET_TOKENIZER_SEMANTICS_CONFIG_KEY,
    }
)

FORWARD_DIRECTION = "source-row-to-target-code"
REVERSE_DIRECTION = "target-row-to-source-code"
SEGMENTATION_ALGORITHM = "greedy-longest-match-bytelevel-token-string-v1"

#: Resolution order applied to every defined row, highest priority first.
CODE_RESOLUTION_ORDER = ("special-role", "anchor-exact", "multi-code")
_CODE_CLASSES = CODE_RESOLUTION_ORDER
_SPECIAL_TOKEN_ROLES = (
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "unk_token_id",
)
_ZERO_MODEL_PARAMETER_OPERATIONS = {
    "training_steps": 0,
    "backward_calls": 0,
    "pruned_parameters": 0,
    "quantized_parameters": 0,
    "random_target_rows": 0,
}
_MAX_ROW_INDEX = 2**31 - 1
_DEFAULT_TRANSPORT_CHUNK_ROWS = 4096

_SENTENCEPIECE_EXTENSION = (
    "The v1 pair bridge aligns two ByteLevel-BPE vocabularies only. A "
    "SentencePiece/Unigram vocabulary requires the SentencePiece byte-fallback "
    "extension -- decompose every piece through its '<0xNN>' byte-fallback rows "
    "into raw bytes before segmentation -- which is deliberately not "
    "implemented in v1 and therefore fails closed here."
)


class TokenizerPairBridgeError(ValueError):
    """Raised when a two-tokenizer code bridge cannot be proven exact."""


@dataclass(frozen=True, slots=True)
class PairBridgeArtifactFile:
    """One immutable sealed payload file record."""

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_path",
            _relative_file_path(self.relative_path),
        )
        object.__setattr__(
            self,
            "size_bytes",
            _nonnegative_int(self.size_bytes, "size_bytes"),
        )
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TokenizerPairCodeTable:
    """Every defined row of one vocabulary as a code over the other's rows."""

    direction: str
    codes: tuple[tuple[int, ...], ...]
    code_classes: tuple[str, ...]
    statistics: Mapping[str, Any]
    array_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.direction not in (FORWARD_DIRECTION, REVERSE_DIRECTION):
            raise TokenizerPairBridgeError(
                "Code table direction must be a declared pair-bridge direction."
            )
        if len(self.codes) != len(self.code_classes):
            raise TokenizerPairBridgeError(
                "Code table classes must cover exactly one code per row."
            )
        if any(code_class not in _CODE_CLASSES for code_class in self.code_classes):
            raise TokenizerPairBridgeError("Code table class is unsupported.")
        if any(not code for code in self.codes):
            raise TokenizerPairBridgeError("Every defined row must carry a code.")

    @property
    def defined_rows(self) -> int:
        return len(self.codes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "statistics": dict(self.statistics),
            "array_sha256": dict(self.array_sha256),
        }


@dataclass(frozen=True, slots=True)
class TokenizerPairCodeProfile:
    """Complete bidirectional code contract between two live vocabularies."""

    source_tokenizer_class: str
    target_tokenizer_class: str
    source_vocab_size: int
    target_vocab_size: int
    source_row_capacity: int
    target_row_capacity: int
    map_special_roles: bool
    special_role_map: tuple[tuple[str, int, int], ...]
    forward: TokenizerPairCodeTable
    reverse: TokenizerPairCodeTable
    source_backend: Mapping[str, Any]
    target_backend: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.forward.direction != FORWARD_DIRECTION:
            raise TokenizerPairBridgeError("forward table has the wrong direction.")
        if self.reverse.direction != REVERSE_DIRECTION:
            raise TokenizerPairBridgeError("reverse table has the wrong direction.")
        if self.forward.defined_rows != self.source_vocab_size:
            raise TokenizerPairBridgeError(
                "forward codes must cover every defined source row."
            )
        if self.reverse.defined_rows != self.target_vocab_size:
            raise TokenizerPairBridgeError(
                "reverse codes must cover every defined target row."
            )
        if self.source_vocab_size > self.source_row_capacity:
            raise TokenizerPairBridgeError(
                "Source vocabulary exceeds its declared row capacity."
            )
        if self.target_vocab_size > self.target_row_capacity:
            raise TokenizerPairBridgeError(
                "Target vocabulary exceeds its declared row capacity."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "profile": "independent-tokenizer-pair-code-v1",
            "source_tokenizer_class": self.source_tokenizer_class,
            "target_tokenizer_class": self.target_tokenizer_class,
            "source_vocab_size": self.source_vocab_size,
            "target_vocab_size": self.target_vocab_size,
            "source_row_capacity": self.source_row_capacity,
            "target_row_capacity": self.target_row_capacity,
            "segmentation": {
                "algorithm": SEGMENTATION_ALGORITHM,
                "resolution_order": list(CODE_RESOLUTION_ORDER),
                "map_special_roles": self.map_special_roles,
                "byte_alphabet_size": 256,
                "target_single_byte_rows_proven": 256,
                "source_single_byte_rows_proven": 256,
                "singleton_unicode_transport_used": False,
            },
            "special_role_map": [
                {
                    "role": role,
                    "source_row_index": source_row,
                    "target_row_index": target_row,
                }
                for role, source_row, target_row in self.special_role_map
            ],
            "forward": self.forward.to_dict(),
            "reverse": self.reverse.to_dict(),
            "source_backend": dict(self.source_backend),
            "target_backend": dict(self.target_backend),
            "model_weight_reads": 0,
            "model_weight_writes": 0,
            "model_weight_payloads_resident": False,
        }

    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class TokenizerPairBridgeArtifact:
    """Verified payload files and the complete bidirectional code profile."""

    root_dir: Path
    profile: TokenizerPairCodeProfile
    source_tokenizer_semantics_sha256: str
    target_tokenizer_semantics_sha256: str
    payload_files: tuple[PairBridgeArtifactFile, ...]
    artifact_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_dir", Path(self.root_dir).resolve())
        if not isinstance(self.profile, TokenizerPairCodeProfile):
            raise TokenizerPairBridgeError(
                "profile must be a TokenizerPairCodeProfile."
            )
        object.__setattr__(
            self,
            "source_tokenizer_semantics_sha256",
            _sha256(
                self.source_tokenizer_semantics_sha256,
                "source_tokenizer_semantics_sha256",
            ),
        )
        object.__setattr__(
            self,
            "target_tokenizer_semantics_sha256",
            _sha256(
                self.target_tokenizer_semantics_sha256,
                "target_tokenizer_semantics_sha256",
            ),
        )
        files = tuple(self.payload_files)
        if not files or any(
            not isinstance(item, PairBridgeArtifactFile) for item in files
        ):
            raise TokenizerPairBridgeError(
                "payload_files must contain typed payload file seals."
            )
        paths = tuple(item.relative_path for item in files)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise TokenizerPairBridgeError(
                "payload_files must be uniquely sorted by relative path."
            )
        object.__setattr__(self, "payload_files", files)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(self.artifact_sha256, "artifact_sha256"),
        )
        if _canonical_sha256(self._body()) != self.artifact_sha256:
            raise TokenizerPairBridgeError(
                "Tokenizer pair bridge artifact seal is invalid."
            )

    @property
    def bridge_kind(self) -> str:
        return TOKENIZER_PAIR_BRIDGE_KIND

    @property
    def profile_fingerprint(self) -> str:
        return self.profile.fingerprint()

    @property
    def source_vocabulary_size(self) -> int:
        return self.profile.source_vocab_size

    @property
    def target_vocabulary_size(self) -> int:
        return self.profile.target_vocab_size

    @property
    def parent_tokenizer_semantics_sha256(self) -> str:
        """Bridge-kind-neutral alias used by the shared operator protocol."""

        return self.source_tokenizer_semantics_sha256

    @property
    def child_tokenizer_semantics_sha256(self) -> str:
        """Bridge-kind-neutral alias used by the shared operator protocol."""

        return self.target_tokenizer_semantics_sha256

    def forward_code(self, source_row_index: int) -> tuple[int, ...]:
        """Return the target-row code of one defined source row."""

        index = _nonnegative_int(source_row_index, "source_row_index")
        if index >= self.profile.source_vocab_size:
            raise TokenizerPairBridgeError(
                "source_row_index is outside the defined source vocabulary."
            )
        return self.profile.forward.codes[index]

    def reverse_code(self, target_row_index: int) -> tuple[int, ...]:
        """Return the source-row code of one defined target row."""

        index = _nonnegative_int(target_row_index, "target_row_index")
        if index >= self.profile.target_vocab_size:
            raise TokenizerPairBridgeError(
                "target_row_index is outside the defined target vocabulary."
            )
        return self.profile.reverse.codes[index]

    def code_table(self, direction: str) -> TokenizerPairCodeTable:
        """Return one direction's table by its ``forward``/``reverse`` name."""

        if direction == "forward":
            return self.profile.forward
        if direction == "reverse":
            return self.profile.reverse
        raise TokenizerPairBridgeError(
            "direction must be either 'forward' or 'reverse'."
        )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": TOKENIZER_PAIR_BRIDGE_SCHEMA,
            "kind": TOKENIZER_PAIR_BRIDGE_KIND,
            "profile_fingerprint": self.profile.fingerprint(),
            "pair_profile": self.profile.to_dict(),
            "source_tokenizer_semantics_sha256": (
                self.source_tokenizer_semantics_sha256
            ),
            "target_tokenizer_semantics_sha256": (
                self.target_tokenizer_semantics_sha256
            ),
            "payload_files": [item.to_dict() for item in self.payload_files],
            "model_parameter_operations": dict(_ZERO_MODEL_PARAMETER_OPERATIONS),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "artifact_sha256": self.artifact_sha256}

    def verify_files(self) -> None:
        """Re-authenticate the payload and reject additions or removals."""

        observed = _payload_file_records(self._verified_root())
        if observed != self.payload_files:
            raise TokenizerPairBridgeError(
                "Pair bridge payload files differ from the sealed artifact."
            )
        self._verify_payload_arrays()

    def verify_packaged_files(self) -> None:
        """Authenticate exactly the sealed payload files inside a package.

        A published child package legitimately carries unrelated files that a
        surrounding artifact certificate authenticates separately, so unlike
        :meth:`verify_files` this tolerates additions while still requiring
        every sealed record to match byte for byte.
        """

        root = self._verified_root()
        for record in self.payload_files:
            path = root / record.relative_path
            if path.is_symlink() or not path.is_file():
                raise TokenizerPairBridgeError(
                    "Sealed pair bridge payload file is missing or unsafe: "
                    f"{record.relative_path}"
                )
            observed = PairBridgeArtifactFile(
                relative_path=record.relative_path,
                size_bytes=path.stat().st_size,
                sha256=_file_sha256(path),
            )
            if observed != record:
                raise TokenizerPairBridgeError(
                    "Pair bridge payload files differ from the sealed artifact."
                )
        self._verify_payload_arrays()

    def verify_live_tokenizers(
        self,
        source_tokenizer: Any,
        target_tokenizer: Any,
        *,
        packaged: bool = False,
    ) -> None:
        """Bind live tokenizer objects back to the stored code contract."""

        if packaged:
            self.verify_packaged_files()
        else:
            self.verify_files()
        source_sha = fingerprint_hf_tokenizer_semantics(source_tokenizer)
        target_sha = fingerprint_hf_tokenizer_semantics(target_tokenizer)
        if source_sha != self.source_tokenizer_semantics_sha256:
            raise TokenizerPairBridgeError(
                "Live source tokenizer differs from the sealed pair bridge."
            )
        if target_sha != self.target_tokenizer_semantics_sha256:
            raise TokenizerPairBridgeError(
                "Live target tokenizer differs from the sealed pair bridge."
            )
        observed = build_hf_tokenizer_pair_code_profile(
            source_tokenizer,
            target_tokenizer,
            source_row_capacity=self.profile.source_row_capacity,
            target_row_capacity=self.profile.target_row_capacity,
            map_special_roles=self.profile.map_special_roles,
        )
        if observed.fingerprint() != self.profile_fingerprint:
            raise TokenizerPairBridgeError(
                "Live tokenizer code tables differ from the sealed profile."
            )
        if (
            observed.forward.codes != self.profile.forward.codes
            or observed.reverse.codes != self.profile.reverse.codes
        ):
            raise TokenizerPairBridgeError(
                "Live tokenizer code tables differ from the sealed profile."
            )

    def _verified_root(self) -> Path:
        if self.root_dir.is_symlink() or not self.root_dir.is_dir():
            raise TokenizerPairBridgeError("Pair bridge root must be a real directory.")
        if _canonical_sha256(self._body()) != self.artifact_sha256:
            raise TokenizerPairBridgeError(
                "Tokenizer pair bridge artifact changed after construction."
            )
        return self.root_dir

    def _verify_payload_arrays(self) -> None:
        payload = _read_code_payload(
            self.root_dir / TOKENIZER_PAIR_BRIDGE_CODES_FILENAME
        )
        for table, prefix in (
            (self.profile.forward, "forward"),
            (self.profile.reverse, "reverse"),
        ):
            observed = _code_table_arrays(table)
            for suffix in ("flat", "offsets", "classes"):
                name = f"{prefix}_code_{suffix}"
                stored = payload.get(name)
                if stored is None:
                    raise TokenizerPairBridgeError(
                        f"Pair bridge payload is missing array {name!r}."
                    )
                if _array_sha256(stored) != table.array_sha256[suffix]:
                    raise TokenizerPairBridgeError(
                        f"Pair bridge payload array {name!r} was tampered with."
                    )
                if _array_sha256(observed[suffix]) != table.array_sha256[suffix]:
                    raise TokenizerPairBridgeError(
                        f"Pair bridge code table {name!r} does not match its seal."
                    )


def tokenizer_pair_bridge_configuration(
    bridge: TokenizerPairBridgeArtifact,
) -> dict[str, str]:
    """Return the exact configuration seal bundle for one verified bridge."""

    if not isinstance(bridge, TokenizerPairBridgeArtifact):
        raise TokenizerPairBridgeError("bridge must be a TokenizerPairBridgeArtifact.")
    bridge.verify_files()
    return {
        PAIR_BRIDGE_ARTIFACT_CONFIG_KEY: bridge.artifact_sha256,
        PAIR_BRIDGE_PROFILE_CONFIG_KEY: bridge.profile_fingerprint,
        PAIR_SOURCE_TOKENIZER_SEMANTICS_CONFIG_KEY: (
            bridge.source_tokenizer_semantics_sha256
        ),
        PAIR_TARGET_TOKENIZER_SEMANTICS_CONFIG_KEY: (
            bridge.target_tokenizer_semantics_sha256
        ),
    }


# ---------------------------------------------------------------------------
# Vocabulary reading and ByteLevel proof
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _VocabularyView:
    """One tokenizer's rows reduced to exact byte strings and role marks."""

    label: str
    tokenizer_class: str
    vocab_size: int
    row_capacity: int
    row_bytes: tuple[bytes, ...]
    added_rows: frozenset[int]
    special_rows: frozenset[int]
    role_rows: tuple[tuple[str, int], ...]
    backend: Mapping[str, Any]

    @property
    def segmentable_rows(self) -> frozenset[int]:
        return frozenset(range(self.vocab_size)) - self.added_rows - self.special_rows


def _byte_level_alphabet_table() -> tuple[str, ...]:
    """Reproduce the canonical ByteLevel byte-to-character table."""

    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    values = list(printable)
    characters = [chr(value) for value in printable]
    shift = 0
    for byte in range(256):
        if byte in printable:
            continue
        values.append(byte)
        characters.append(chr(256 + shift))
        shift += 1
    table = [""] * 256
    for byte, character in zip(values, characters, strict=True):
        table[byte] = character
    if any(not character for character in table) or len(set(table)) != 256:
        raise TokenizerPairBridgeError(
            "The ByteLevel byte-to-character table is not a bijection."
        )
    return tuple(table)


_BYTE_TO_CHARACTER = _byte_level_alphabet_table()
_CHARACTER_TO_BYTE = {
    character: byte for byte, character in enumerate(_BYTE_TO_CHARACTER)
}


def _verify_byte_level_alphabet() -> None:
    try:
        from tokenizers.pre_tokenizers import ByteLevel
    except ImportError as exc:  # pragma: no cover - required HF dependency
        raise TokenizerPairBridgeError(
            "the tokenizers library is required to prove the ByteLevel alphabet."
        ) from exc
    if set(ByteLevel.alphabet()) != set(_BYTE_TO_CHARACTER):
        raise TokenizerPairBridgeError(
            "The derived ByteLevel alphabet differs from the tokenizers library."
        )


def _structured_backend(tokenizer: Any, label: str) -> dict[str, Any]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    serializer = getattr(backend, "to_str", None)
    if backend is None or not callable(serializer):
        raise TokenizerPairBridgeError(
            f"{label} tokenizer must be a Hugging Face fast tokenizer exposing a "
            f"serializable backend. {_SENTENCEPIECE_EXTENSION}"
        )
    try:
        data = json.loads(serializer())
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TokenizerPairBridgeError(
            f"{label} tokenizer backend serialization is not valid JSON."
        ) from exc
    if not isinstance(data, dict):
        raise TokenizerPairBridgeError(
            f"{label} tokenizer backend serialization must be a JSON object."
        )
    return data


def _prove_byte_level_bpe(data: Mapping[str, Any], label: str) -> dict[str, Any]:
    model = data.get("model")
    if not isinstance(model, Mapping):
        raise TokenizerPairBridgeError(
            f"{label} tokenizer backend has no structured model. "
            f"{_SENTENCEPIECE_EXTENSION}"
        )
    model_type = model.get("type")
    if model_type != "BPE":
        raise TokenizerPairBridgeError(
            f"{label} tokenizer uses the {model_type!r} model, not ByteLevel BPE. "
            f"{_SENTENCEPIECE_EXTENSION}"
        )
    if not _contains_component(data.get("pre_tokenizer"), "ByteLevel"):
        raise TokenizerPairBridgeError(
            f"{label} tokenizer does not use ByteLevel pre-tokenization. "
            f"{_SENTENCEPIECE_EXTENSION}"
        )
    if not _contains_component(data.get("decoder"), "ByteLevel"):
        raise TokenizerPairBridgeError(
            f"{label} tokenizer does not use a ByteLevel decoder. "
            f"{_SENTENCEPIECE_EXTENSION}"
        )
    if model.get("byte_fallback"):
        raise TokenizerPairBridgeError(
            f"{label} tokenizer declares BPE byte_fallback rows. "
            f"{_SENTENCEPIECE_EXTENSION}"
        )
    for key in ("continuing_subword_prefix", "end_of_word_suffix"):
        if model.get(key) not in (None, ""):
            raise TokenizerPairBridgeError(
                f"{label} tokenizer sets {key!r}, so a token string is no longer "
                "the exact ByteLevel rendering of its bytes."
            )
    return {
        "model_type": model_type,
        "pre_tokenizer": data.get("pre_tokenizer"),
        "decoder": data.get("decoder"),
        "normalizer": data.get("normalizer"),
        "byte_fallback": bool(model.get("byte_fallback")),
        "ignore_merges": bool(model.get("ignore_merges")),
        "backend_sha256": _canonical_sha256(data),
    }


def _contains_component(component: Any, component_type: str) -> bool:
    if isinstance(component, Mapping):
        if component.get("type") == component_type:
            return True
        return any(
            _contains_component(value, component_type) for value in component.values()
        )
    if isinstance(component, list):
        return any(_contains_component(value, component_type) for value in component)
    return False


def _row_token_strings(tokenizer: Any, label: str) -> tuple[str, ...]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise TokenizerPairBridgeError(f"{label} tokenizer must expose get_vocab().")
    vocabulary = get_vocab()
    try:
        length = int(len(tokenizer))
    except (TypeError, ValueError, OverflowError) as exc:
        raise TokenizerPairBridgeError(
            f"{label} tokenizer length must be a positive integer."
        ) from exc
    if not isinstance(vocabulary, Mapping) or len(vocabulary) != length:
        raise TokenizerPairBridgeError(
            f"{label} tokenizer vocabulary must match its reported length."
        )
    if length <= 0 or length > _MAX_ROW_INDEX:
        raise TokenizerPairBridgeError(
            f"{label} tokenizer length must be a positive 32-bit integer."
        )
    rows: list[str | None] = [None] * length
    for token, raw_token_id in vocabulary.items():
        if not isinstance(token, str):
            raise TokenizerPairBridgeError(
                f"{label} tokenizer vocabulary tokens must be strings."
            )
        token_id = _nonnegative_int(raw_token_id, f"{label} token_id")
        if token_id >= length or rows[token_id] is not None:
            raise TokenizerPairBridgeError(
                f"{label} tokenizer vocabulary IDs must be contiguous from zero."
            )
        rows[token_id] = token
    if any(token is None for token in rows):
        raise TokenizerPairBridgeError(
            f"{label} tokenizer vocabulary IDs must be contiguous from zero."
        )
    return tuple(token for token in rows if token is not None)


def _added_token_contents(
    data: Mapping[str, Any],
    vocab_size: int,
    label: str,
) -> dict[int, str]:
    raw = data.get("added_tokens", ())
    if not isinstance(raw, list):
        raise TokenizerPairBridgeError(
            f"{label} tokenizer added_tokens must be a JSON list."
        )
    contents: dict[int, str] = {}
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise TokenizerPairBridgeError(
                f"{label} tokenizer added_tokens entries must be objects."
            )
        token_id = _nonnegative_int(entry.get("id"), f"{label} added token id")
        content = entry.get("content")
        if not isinstance(content, str) or not content:
            raise TokenizerPairBridgeError(
                f"{label} tokenizer added token content must be nonempty text."
            )
        if token_id >= vocab_size:
            continue
        contents[token_id] = content
    return contents


def _row_byte_strings(
    tokens: Sequence[str],
    added: Mapping[int, str],
    label: str,
) -> tuple[bytes, ...]:
    rows: list[bytes] = []
    for token_id, token in enumerate(tokens):
        content = added.get(token_id)
        if content is not None:
            if content != token:
                raise TokenizerPairBridgeError(
                    f"{label} tokenizer added row {token_id} disagrees with "
                    "get_vocab()."
                )
            raw = content.encode("utf-8")
        else:
            try:
                raw = bytes(_CHARACTER_TO_BYTE[character] for character in token)
            except KeyError as exc:
                raise TokenizerPairBridgeError(
                    f"{label} tokenizer row {token_id} token string contains "
                    f"{exc.args[0]!r}, which is outside the 256-character "
                    "ByteLevel alphabet. "
                    f"{_SENTENCEPIECE_EXTENSION}"
                ) from exc
        if not raw:
            raise TokenizerPairBridgeError(
                f"{label} tokenizer row {token_id} renders to zero bytes."
            )
        rows.append(raw)
    return tuple(rows)


def _special_row_ids(tokenizer: Any, vocab_size: int, label: str) -> frozenset[int]:
    raw = getattr(tokenizer, "all_special_ids", ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TokenizerPairBridgeError(
            f"{label} tokenizer all_special_ids must be a sequence."
        )
    values = {_nonnegative_int(value, f"{label} special token ID") for value in raw}
    if any(value >= vocab_size for value in values):
        raise TokenizerPairBridgeError(
            f"{label} tokenizer special token ID is outside tokenizer capacity."
        )
    return frozenset(values)


def _role_row_ids(
    tokenizer: Any,
    vocab_size: int,
    label: str,
) -> tuple[tuple[str, int], ...]:
    roles: list[tuple[str, int]] = []
    for role in _SPECIAL_TOKEN_ROLES:
        value = getattr(tokenizer, role, None)
        if value is None:
            continue
        token_id = _nonnegative_int(value, f"{label} {role}")
        if token_id >= vocab_size:
            raise TokenizerPairBridgeError(
                f"{label} {role} is outside tokenizer capacity."
            )
        roles.append((role, token_id))
    return tuple(roles)


def _vocabulary_view(
    tokenizer: Any,
    label: str,
    row_capacity: int | None,
) -> _VocabularyView:
    data = _structured_backend(tokenizer, label)
    backend = _prove_byte_level_bpe(data, label)
    tokens = _row_token_strings(tokenizer, label)
    vocab_size = len(tokens)
    added = _added_token_contents(data, vocab_size, label)
    capacity = (
        vocab_size
        if row_capacity is None
        else _positive_int(row_capacity, f"{label}_row_capacity")
    )
    if capacity < vocab_size:
        raise TokenizerPairBridgeError(
            f"{label}_row_capacity is smaller than the {label} vocabulary."
        )
    if capacity > _MAX_ROW_INDEX:
        raise TokenizerPairBridgeError(
            f"{label}_row_capacity must be a positive 32-bit integer."
        )
    return _VocabularyView(
        label=label,
        tokenizer_class=(
            f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
        ),
        vocab_size=vocab_size,
        row_capacity=capacity,
        row_bytes=_row_byte_strings(tokens, added, label),
        added_rows=frozenset(added),
        special_rows=_special_row_ids(tokenizer, vocab_size, label),
        role_rows=_role_row_ids(tokenizer, vocab_size, label),
        backend=backend,
    )


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


class _ByteSegmenter:
    """Deterministic greedy longest-match over one vocabulary's byte strings."""

    __slots__ = ("_rows", "_lengths_by_first_byte", "_single_byte_rows")

    def __init__(self, rows: Mapping[bytes, int]) -> None:
        lengths: dict[int, set[int]] = {}
        for raw in rows:
            lengths.setdefault(raw[0], set()).add(len(raw))
        self._rows = dict(rows)
        self._lengths_by_first_byte = {
            first: tuple(sorted(values, reverse=True))
            for first, values in lengths.items()
        }
        self._single_byte_rows = tuple(
            self._rows.get(bytes((byte,)), -1) for byte in range(256)
        )

    def missing_bytes(self) -> tuple[int, ...]:
        return tuple(byte for byte, row in enumerate(self._single_byte_rows) if row < 0)

    def exact_row(self, raw: bytes) -> int | None:
        return self._rows.get(raw)

    def segment(self, raw: bytes) -> tuple[int, ...]:
        code: list[int] = []
        index = 0
        total = len(raw)
        while index < total:
            candidates = self._lengths_by_first_byte.get(raw[index])
            if candidates is None:
                raise TokenizerPairBridgeError(
                    f"No vocabulary row renders byte 0x{raw[index]:02x}."
                )
            for length in candidates:
                end = index + length
                if end > total:
                    continue
                row = self._rows.get(raw[index:end])
                if row is None:
                    continue
                code.append(row)
                index = end
                break
            else:  # pragma: no cover - the 256 byte rows make this unreachable
                raise TokenizerPairBridgeError(
                    "Greedy segmentation stalled despite complete byte coverage."
                )
        return tuple(code)


def _segmentation_dictionary(view: _VocabularyView) -> _ByteSegmenter:
    rows: dict[bytes, int] = {}
    for row_index in sorted(view.segmentable_rows):
        raw = view.row_bytes[row_index]
        if raw not in rows:
            rows[raw] = row_index
    segmenter = _ByteSegmenter(rows)
    missing = segmenter.missing_bytes()
    if missing:
        raise TokenizerPairBridgeError(
            f"{view.label} tokenizer does not cover the complete 256-row "
            "ByteLevel byte alphabet; single-byte rows are missing for "
            + ", ".join(f"0x{byte:02x}" for byte in missing[:8])
            + ". A pair bridge has no exact fallback without them."
        )
    return segmenter


def _anchor_row(
    raw: bytes,
    source_is_added: bool,
    output_added: Mapping[bytes, int],
    output_model: _ByteSegmenter,
) -> int | None:
    """Resolve one exact byte-string anchor, preferring like row kinds first."""

    added = output_added.get(raw)
    model = output_model.exact_row(raw)
    ordered = (added, model) if source_is_added else (model, added)
    for row in ordered:
        if row is not None:
            return row
    return None


def _added_byte_rows(view: _VocabularyView) -> dict[bytes, int]:
    rows: dict[bytes, int] = {}
    for row_index in sorted(view.added_rows | view.special_rows):
        raw = view.row_bytes[row_index]
        if raw not in rows:
            rows[raw] = row_index
    return rows


def _build_code_table(
    source: _VocabularyView,
    output: _VocabularyView,
    direction: str,
    *,
    map_special_roles: bool,
) -> tuple[TokenizerPairCodeTable, tuple[tuple[str, int, int], ...]]:
    segmenter = _segmentation_dictionary(output)
    output_added = _added_byte_rows(output)
    output_roles = dict(output.role_rows)
    codes: list[tuple[int, ...]] = []
    classes: list[str] = []
    role_map: list[tuple[str, int, int]] = []
    role_by_source_row: dict[int, str] = {}
    if map_special_roles:
        for role, source_row in source.role_rows:
            output_row = output_roles.get(role)
            if output_row is None or source_row in role_by_source_row:
                continue
            role_by_source_row[source_row] = role
            role_map.append((role, source_row, output_row))

    byte_exact_rows = 0
    pure_byte_rows = 0
    single_byte_anchor_rows = 0
    single_byte_output_rows = {
        segmenter.exact_row(bytes((byte,))) for byte in range(256)
    }
    for row_index in range(source.vocab_size):
        raw = source.row_bytes[row_index]
        role = role_by_source_row.get(row_index)
        if role is not None:
            code = (output_roles[role],)
            code_class = "special-role"
        else:
            anchor = _anchor_row(
                raw,
                row_index in source.added_rows or row_index in source.special_rows,
                output_added,
                segmenter,
            )
            if anchor is not None:
                code = (anchor,)
                code_class = "anchor-exact"
            else:
                code = segmenter.segment(raw)
                code_class = "multi-code"
                if len(code) < 2:
                    raise TokenizerPairBridgeError(
                        "A single-row segmentation must resolve as an anchor; "
                        f"row {row_index} of {source.label} did not."
                    )
        if code_class != "special-role":
            rendered = b"".join(output.row_bytes[row] for row in code)
            if rendered != raw:
                raise TokenizerPairBridgeError(
                    f"{source.label} row {row_index} does not render byte-exactly "
                    f"through its {len(code)}-row code."
                )
            byte_exact_rows += 1
            if all(row in single_byte_output_rows for row in code):
                if code_class == "multi-code":
                    pure_byte_rows += 1
                else:
                    single_byte_anchor_rows += 1
        codes.append(code)
        classes.append(code_class)

    statistics = _code_statistics(
        direction,
        source,
        output,
        codes,
        classes,
        byte_exact_rows=byte_exact_rows,
        pure_byte_rows=pure_byte_rows,
        single_byte_anchor_rows=single_byte_anchor_rows,
    )
    arrays = _pack_code_arrays(codes, classes)
    table = TokenizerPairCodeTable(
        direction=direction,
        codes=tuple(codes),
        code_classes=tuple(classes),
        statistics=statistics,
        array_sha256={name: _array_sha256(values) for name, values in arrays.items()},
    )
    return table, tuple(role_map)


def _code_statistics(
    direction: str,
    source: _VocabularyView,
    output: _VocabularyView,
    codes: Sequence[tuple[int, ...]],
    classes: Sequence[str],
    *,
    byte_exact_rows: int,
    pure_byte_rows: int,
    single_byte_anchor_rows: int,
) -> dict[str, Any]:
    histogram: dict[int, int] = {}
    used_rows: set[int] = set()
    total_entries = 0
    for code in codes:
        histogram[len(code)] = histogram.get(len(code), 0) + 1
        total_entries += len(code)
        used_rows.update(code)
    class_counts = {name: 0 for name in _CODE_CLASSES}
    for code_class in classes:
        class_counts[code_class] += 1
    return {
        "direction": direction,
        "input_vocab_size": source.vocab_size,
        "input_row_capacity": source.row_capacity,
        "output_vocab_size": output.vocab_size,
        "output_row_capacity": output.row_capacity,
        "defined_rows": len(codes),
        "unresolved_rows": 0,
        "anchor_exact_rows": class_counts["anchor-exact"],
        "special_role_rows": class_counts["special-role"],
        "multi_code_rows": class_counts["multi-code"],
        "byte_exact_rows": byte_exact_rows,
        "byte_mismatch_rows": 0,
        "single_byte_anchor_rows": single_byte_anchor_rows,
        "pure_byte_fallback_rows": pure_byte_rows,
        "total_code_entries": total_entries,
        "min_code_length": min(histogram) if histogram else 0,
        "max_code_length": max(histogram) if histogram else 0,
        "code_length_histogram": [
            [length, histogram[length]] for length in sorted(histogram)
        ],
        "covered_output_rows": len(used_rows),
        "uncovered_output_rows": output.vocab_size - len(used_rows),
    }


def build_hf_tokenizer_pair_code_profile(
    source_tokenizer: Any,
    target_tokenizer: Any,
    *,
    source_row_capacity: int | None = None,
    target_row_capacity: int | None = None,
    map_special_roles: bool = True,
) -> TokenizerPairCodeProfile:
    """Compile the bidirectional code contract without writing any file.

    Both tokenizers must be independent Hugging Face fast ByteLevel-BPE
    tokenizers.  Anything else fails closed, because the exactness proof this
    module offers rests entirely on a token string being the ByteLevel
    rendering of its own bytes.
    """

    if not isinstance(map_special_roles, bool):
        raise TokenizerPairBridgeError("map_special_roles must be boolean.")
    _verify_byte_level_alphabet()
    source = _vocabulary_view(source_tokenizer, "source", source_row_capacity)
    target = _vocabulary_view(target_tokenizer, "target", target_row_capacity)
    forward, forward_roles = _build_code_table(
        source,
        target,
        FORWARD_DIRECTION,
        map_special_roles=map_special_roles,
    )
    reverse, _reverse_roles = _build_code_table(
        target,
        source,
        REVERSE_DIRECTION,
        map_special_roles=map_special_roles,
    )
    return TokenizerPairCodeProfile(
        source_tokenizer_class=source.tokenizer_class,
        target_tokenizer_class=target.tokenizer_class,
        source_vocab_size=source.vocab_size,
        target_vocab_size=target.vocab_size,
        source_row_capacity=source.row_capacity,
        target_row_capacity=target.row_capacity,
        map_special_roles=map_special_roles,
        special_role_map=forward_roles,
        forward=forward,
        reverse=reverse,
        source_backend=source.backend,
        target_backend=target.backend,
    )


# ---------------------------------------------------------------------------
# Sealed artifact
# ---------------------------------------------------------------------------


def compile_hf_tokenizer_pair_bridge(
    source_tokenizer: Any,
    target_tokenizer: Any,
    output_dir: str | Path,
    *,
    source_row_capacity: int | None = None,
    target_row_capacity: int | None = None,
    map_special_roles: bool = True,
) -> TokenizerPairBridgeArtifact:
    """Compile and seal a code bridge between two independent tokenizers.

    ``output_dir`` is created when absent and must be dedicated to this bridge:
    only the sealed manifest and its content-addressed code payload may live
    there.  Re-compiling an identical bridge is idempotent; a different bridge
    fails closed rather than overwriting the sealed one.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    if root.is_symlink() or not root.is_dir():
        raise TokenizerPairBridgeError("output_dir must be a real directory.")
    manifest_path = root / TOKENIZER_PAIR_BRIDGE_FILENAME
    payload_path = root / TOKENIZER_PAIR_BRIDGE_CODES_FILENAME
    if manifest_path.exists():
        existing = load_hf_tokenizer_pair_bridge_artifact(
            root,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
        )
        candidate = build_hf_tokenizer_pair_code_profile(
            source_tokenizer,
            target_tokenizer,
            source_row_capacity=source_row_capacity,
            target_row_capacity=target_row_capacity,
            map_special_roles=map_special_roles,
        )
        if candidate.fingerprint() != existing.profile_fingerprint:
            raise TokenizerPairBridgeError(
                "A different tokenizer pair bridge is already sealed here."
            )
        return existing
    if payload_path.exists():
        raise TokenizerPairBridgeError(
            "output_dir holds a pair bridge payload without a sealed manifest."
        )
    stray = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if not path.is_dir()
    )
    if stray:
        raise TokenizerPairBridgeError(
            "output_dir must be dedicated to the pair bridge; it also holds "
            + ", ".join(repr(name) for name in stray[:4])
        )

    profile = build_hf_tokenizer_pair_code_profile(
        source_tokenizer,
        target_tokenizer,
        source_row_capacity=source_row_capacity,
        target_row_capacity=target_row_capacity,
        map_special_roles=map_special_roles,
    )
    source_sha = fingerprint_hf_tokenizer_semantics(source_tokenizer)
    target_sha = fingerprint_hf_tokenizer_semantics(target_tokenizer)
    _write_code_payload(payload_path, profile)
    files = _payload_file_records(root)
    body = {
        "schema": TOKENIZER_PAIR_BRIDGE_SCHEMA,
        "kind": TOKENIZER_PAIR_BRIDGE_KIND,
        "profile_fingerprint": profile.fingerprint(),
        "pair_profile": profile.to_dict(),
        "source_tokenizer_semantics_sha256": source_sha,
        "target_tokenizer_semantics_sha256": target_sha,
        "payload_files": [item.to_dict() for item in files],
        "model_parameter_operations": dict(_ZERO_MODEL_PARAMETER_OPERATIONS),
    }
    artifact_sha = _canonical_sha256(body)
    artifact = TokenizerPairBridgeArtifact(
        root_dir=root,
        profile=profile,
        source_tokenizer_semantics_sha256=source_sha,
        target_tokenizer_semantics_sha256=target_sha,
        payload_files=files,
        artifact_sha256=artifact_sha,
    )
    _atomic_write(manifest_path, _canonical_json(artifact.to_dict()) + b"\n")
    return load_hf_tokenizer_pair_bridge_artifact(
        root,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=artifact_sha,
    )


def load_hf_tokenizer_pair_bridge_artifact(
    bridge_dir: str | Path,
    *,
    source_tokenizer: Any | None = None,
    target_tokenizer: Any | None = None,
    expected_artifact_sha256: str | None = None,
    packaged: bool = False,
) -> TokenizerPairBridgeArtifact:
    """Load and re-authenticate a persisted tokenizer pair bridge.

    With ``packaged=True`` the root is treated as a published package: every
    sealed payload record is authenticated byte for byte, but unrelated files
    are tolerated because a surrounding certificate authenticates them.
    """

    root = Path(bridge_dir).resolve()
    if root.is_symlink() or not root.is_dir():
        raise TokenizerPairBridgeError("Pair bridge root must be a real directory.")
    manifest_path = root / TOKENIZER_PAIR_BRIDGE_FILENAME
    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise TokenizerPairBridgeError(
            "Tokenizer pair bridge manifest is missing or unreadable."
        ) from exc
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TokenizerPairBridgeError(
            "Tokenizer pair bridge manifest is not valid JSON."
        ) from exc
    if not isinstance(payload, Mapping):
        raise TokenizerPairBridgeError(
            "Tokenizer pair bridge manifest must be a JSON object."
        )
    expected_keys = {
        "schema",
        "kind",
        "profile_fingerprint",
        "pair_profile",
        "source_tokenizer_semantics_sha256",
        "target_tokenizer_semantics_sha256",
        "payload_files",
        "model_parameter_operations",
        "artifact_sha256",
    }
    if set(payload) != expected_keys:
        raise TokenizerPairBridgeError(
            "Tokenizer pair bridge manifest fields are incomplete or unknown."
        )
    if (
        payload["schema"] != TOKENIZER_PAIR_BRIDGE_SCHEMA
        or payload["kind"] != TOKENIZER_PAIR_BRIDGE_KIND
    ):
        raise TokenizerPairBridgeError(
            "Tokenizer pair bridge schema or kind is unsupported."
        )
    if payload["model_parameter_operations"] != _ZERO_MODEL_PARAMETER_OPERATIONS:
        raise TokenizerPairBridgeError(
            "Tokenizer pair bridge model-parameter operation audit is invalid."
        )
    file_values = payload["payload_files"]
    if not isinstance(file_values, list):
        raise TokenizerPairBridgeError("payload_files must be a JSON list.")
    files = tuple(_file_record_from_dict(value) for value in file_values)
    profile = _profile_from_dict(
        payload["pair_profile"],
        root / TOKENIZER_PAIR_BRIDGE_CODES_FILENAME,
    )
    if payload["profile_fingerprint"] != profile.fingerprint():
        raise TokenizerPairBridgeError("Tokenizer pair profile fingerprint is invalid.")
    artifact = TokenizerPairBridgeArtifact(
        root_dir=root,
        profile=profile,
        source_tokenizer_semantics_sha256=(
            payload["source_tokenizer_semantics_sha256"]
        ),
        target_tokenizer_semantics_sha256=(
            payload["target_tokenizer_semantics_sha256"]
        ),
        payload_files=files,
        artifact_sha256=payload["artifact_sha256"],
    )
    if packaged:
        artifact.verify_packaged_files()
    else:
        artifact.verify_files()
    if expected_artifact_sha256 is not None and artifact.artifact_sha256 != _sha256(
        expected_artifact_sha256,
        "expected_artifact_sha256",
    ):
        raise TokenizerPairBridgeError(
            "Tokenizer pair bridge differs from the expected artifact."
        )
    if (source_tokenizer is None) != (target_tokenizer is None):
        raise TokenizerPairBridgeError(
            "Live verification requires both source and target tokenizers."
        )
    if source_tokenizer is not None:
        artifact.verify_live_tokenizers(
            source_tokenizer,
            target_tokenizer,
            packaged=packaged,
        )
    return artifact


def _profile_from_dict(
    value: Any,
    payload_path: Path,
) -> TokenizerPairCodeProfile:
    if not isinstance(value, Mapping):
        raise TokenizerPairBridgeError("pair_profile must be a JSON object.")
    expected = {
        "schema_version",
        "profile",
        "source_tokenizer_class",
        "target_tokenizer_class",
        "source_vocab_size",
        "target_vocab_size",
        "source_row_capacity",
        "target_row_capacity",
        "segmentation",
        "special_role_map",
        "forward",
        "reverse",
        "source_backend",
        "target_backend",
        "model_weight_reads",
        "model_weight_writes",
        "model_weight_payloads_resident",
    }
    if set(value) != expected:
        raise TokenizerPairBridgeError(
            "Tokenizer pair profile fields are incomplete or unknown."
        )
    if value["schema_version"] != 1 or (
        value["profile"] != "independent-tokenizer-pair-code-v1"
    ):
        raise TokenizerPairBridgeError("Tokenizer pair profile is unsupported.")
    if (
        value["model_weight_reads"] != 0
        or value["model_weight_writes"] != 0
        or value["model_weight_payloads_resident"] is not False
    ):
        raise TokenizerPairBridgeError(
            "Tokenizer pair profile weight-operation audit is invalid."
        )
    segmentation = value["segmentation"]
    if not isinstance(segmentation, Mapping) or (
        segmentation.get("algorithm") != SEGMENTATION_ALGORITHM
    ):
        raise TokenizerPairBridgeError(
            "Tokenizer pair segmentation algorithm is unsupported."
        )
    map_special_roles = segmentation.get("map_special_roles")
    if not isinstance(map_special_roles, bool):
        raise TokenizerPairBridgeError(
            "Tokenizer pair segmentation map_special_roles must be boolean."
        )
    arrays = _read_code_payload(payload_path)
    forward = _table_from_dict(value["forward"], FORWARD_DIRECTION, arrays)
    reverse = _table_from_dict(value["reverse"], REVERSE_DIRECTION, arrays)
    role_map = _role_map_from_list(value["special_role_map"])
    profile = TokenizerPairCodeProfile(
        source_tokenizer_class=_nonempty_text(
            value["source_tokenizer_class"],
            "source_tokenizer_class",
        ),
        target_tokenizer_class=_nonempty_text(
            value["target_tokenizer_class"],
            "target_tokenizer_class",
        ),
        source_vocab_size=_positive_int(
            value["source_vocab_size"],
            "source_vocab_size",
        ),
        target_vocab_size=_positive_int(
            value["target_vocab_size"],
            "target_vocab_size",
        ),
        source_row_capacity=_positive_int(
            value["source_row_capacity"],
            "source_row_capacity",
        ),
        target_row_capacity=_positive_int(
            value["target_row_capacity"],
            "target_row_capacity",
        ),
        map_special_roles=map_special_roles,
        special_role_map=role_map,
        forward=forward,
        reverse=reverse,
        source_backend=value["source_backend"],
        target_backend=value["target_backend"],
    )
    if profile.to_dict() != json.loads(_canonical_json(dict(value))):
        raise TokenizerPairBridgeError(
            "Tokenizer pair profile does not round-trip canonically."
        )
    return profile


def _table_from_dict(
    value: Any,
    direction: str,
    arrays: Mapping[str, Any],
) -> TokenizerPairCodeTable:
    if not isinstance(value, Mapping) or set(value) != {
        "direction",
        "statistics",
        "array_sha256",
    }:
        raise TokenizerPairBridgeError(
            "Tokenizer pair code table fields are incomplete or unknown."
        )
    if value["direction"] != direction:
        raise TokenizerPairBridgeError(
            "Tokenizer pair code table direction is misplaced."
        )
    statistics = value["statistics"]
    if not isinstance(statistics, Mapping):
        raise TokenizerPairBridgeError("Code table statistics must be an object.")
    digests = value["array_sha256"]
    if not isinstance(digests, Mapping) or set(digests) != {
        "flat",
        "offsets",
        "classes",
    }:
        raise TokenizerPairBridgeError(
            "Code table array digests are incomplete or unknown."
        )
    prefix = "forward" if direction == FORWARD_DIRECTION else "reverse"
    stored = {
        suffix: arrays.get(f"{prefix}_code_{suffix}")
        for suffix in ("flat", "offsets", "classes")
    }
    if any(item is None for item in stored.values()):
        raise TokenizerPairBridgeError(
            f"Pair bridge payload lacks the {prefix} code arrays."
        )
    for suffix, values in stored.items():
        if _array_sha256(values) != _sha256(digests[suffix], f"{suffix}_sha256"):
            raise TokenizerPairBridgeError(
                f"Pair bridge {prefix} code array {suffix!r} was tampered with."
            )
    codes, classes = _unpack_code_arrays(
        stored["flat"],
        stored["offsets"],
        stored["classes"],
    )
    return TokenizerPairCodeTable(
        direction=direction,
        codes=codes,
        code_classes=classes,
        statistics=statistics,
        array_sha256={
            suffix: digests[suffix] for suffix in ("flat", "offsets", "classes")
        },
    )


def _role_map_from_list(value: Any) -> tuple[tuple[str, int, int], ...]:
    if not isinstance(value, list):
        raise TokenizerPairBridgeError("special_role_map must be a JSON list.")
    entries: list[tuple[str, int, int]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "role",
            "source_row_index",
            "target_row_index",
        }:
            raise TokenizerPairBridgeError(
                "special_role_map entries are incomplete or unknown."
            )
        role = item["role"]
        if role not in _SPECIAL_TOKEN_ROLES:
            raise TokenizerPairBridgeError("special_role_map role is unsupported.")
        entries.append(
            (
                role,
                _nonnegative_int(item["source_row_index"], "source_row_index"),
                _nonnegative_int(item["target_row_index"], "target_row_index"),
            )
        )
    return tuple(entries)


# ---------------------------------------------------------------------------
# Content-addressed code payload
# ---------------------------------------------------------------------------


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - declared project dependency
        raise TokenizerPairBridgeError(
            "numpy is required for tokenizer pair bridge code payloads."
        ) from exc
    return numpy


def _pack_code_arrays(
    codes: Sequence[tuple[int, ...]],
    classes: Sequence[str],
) -> dict[str, Any]:
    numpy = _numpy()
    offsets = numpy.zeros(len(codes) + 1, dtype="<i4")
    total = 0
    for index, code in enumerate(codes):
        total += len(code)
        offsets[index + 1] = total
    flat = numpy.empty(total, dtype="<i4")
    position = 0
    for code in codes:
        for row in code:
            flat[position] = row
            position += 1
    class_values = numpy.asarray(
        [_CODE_CLASSES.index(name) for name in classes],
        dtype="<i1",
    )
    return {"flat": flat, "offsets": offsets, "classes": class_values}


def _code_table_arrays(table: TokenizerPairCodeTable) -> dict[str, Any]:
    return _pack_code_arrays(table.codes, table.code_classes)


def _unpack_code_arrays(
    flat: Any,
    offsets: Any,
    classes: Any,
) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    numpy = _numpy()
    flat = numpy.asarray(flat)
    offsets = numpy.asarray(offsets)
    classes = numpy.asarray(classes)
    if flat.ndim != 1 or offsets.ndim != 1 or classes.ndim != 1:
        raise TokenizerPairBridgeError("Pair bridge code arrays must be rank one.")
    if int(offsets.shape[0]) != int(classes.shape[0]) + 1:
        raise TokenizerPairBridgeError(
            "Pair bridge code offsets do not match the class array."
        )
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(flat.shape[0]):
        raise TokenizerPairBridgeError(
            "Pair bridge code offsets do not span the flat code array."
        )
    codes: list[tuple[int, ...]] = []
    for index in range(int(classes.shape[0])):
        start = int(offsets[index])
        end = int(offsets[index + 1])
        if end <= start:
            raise TokenizerPairBridgeError(
                "Every defined row must carry a nonempty code."
            )
        codes.append(tuple(int(row) for row in flat[start:end]))
    names: list[str] = []
    for value in classes:
        index = int(value)
        if index < 0 or index >= len(_CODE_CLASSES):
            raise TokenizerPairBridgeError("Pair bridge code class is unsupported.")
        names.append(_CODE_CLASSES[index])
    return tuple(codes), tuple(names)


def _write_code_payload(path: Path, profile: TokenizerPairCodeProfile) -> None:
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:  # pragma: no cover - required HF dependency
        raise TokenizerPairBridgeError(
            "safetensors is required for tokenizer pair bridge payloads."
        ) from exc
    tensors: dict[str, Any] = {}
    for table, prefix in (
        (profile.forward, "forward"),
        (profile.reverse, "reverse"),
    ):
        for suffix, values in _code_table_arrays(table).items():
            tensors[f"{prefix}_code_{suffix}"] = values
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise TokenizerPairBridgeError(
            "Pair bridge payload temporary path already exists."
        )
    try:
        save_file(tensors, str(temporary))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_code_payload(path: Path) -> dict[str, Any]:
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:  # pragma: no cover - required HF dependency
        raise TokenizerPairBridgeError(
            "safetensors is required for tokenizer pair bridge payloads."
        ) from exc
    if path.is_symlink() or not path.is_file():
        raise TokenizerPairBridgeError("Pair bridge code payload is missing or unsafe.")
    try:
        return dict(load_file(str(path)))
    except Exception as exc:
        raise TokenizerPairBridgeError(
            "Pair bridge code payload is not a readable safetensors file."
        ) from exc


def _array_sha256(values: Any) -> str:
    numpy = _numpy()
    array = numpy.asarray(values)
    if array.ndim != 1:
        raise TokenizerPairBridgeError("Pair bridge code arrays must be rank one.")
    if array.dtype == numpy.dtype("int8"):
        dtype = "<i1"
    elif array.dtype in (numpy.dtype("int32"), numpy.dtype("<i4")):
        dtype = "<i4"
    else:
        raise TokenizerPairBridgeError(
            f"Pair bridge code arrays must be int8 or int32, not {array.dtype}."
        )
    digest = hashlib.sha256()
    digest.update(f"neural-forge-pair-code-{dtype}:{int(array.shape[0])}\0".encode())
    digest.update(array.astype(dtype, copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _payload_file_records(root: Path) -> tuple[PairBridgeArtifactFile, ...]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise TokenizerPairBridgeError(
                "Pair bridge artifacts may not contain symbolic links."
            )
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == TOKENIZER_PAIR_BRIDGE_FILENAME:
            continue
        if not path.is_file():
            raise TokenizerPairBridgeError(
                f"Pair bridge artifact entry is not a regular file: {relative}"
            )
        records.append(
            PairBridgeArtifactFile(
                relative_path=relative,
                size_bytes=path.stat().st_size,
                sha256=_file_sha256(path),
            )
        )
    if not records:
        raise TokenizerPairBridgeError(
            "Pair bridge directory contains no sealable payload files."
        )
    return tuple(records)


def _file_record_from_dict(value: Any) -> PairBridgeArtifactFile:
    if not isinstance(value, Mapping) or set(value) != {
        "relative_path",
        "size_bytes",
        "sha256",
    }:
        raise TokenizerPairBridgeError(
            "Pair bridge payload file seal fields are incomplete or unknown."
        )
    return PairBridgeArtifactFile(
        relative_path=value["relative_path"],
        size_bytes=value["size_bytes"],
        sha256=value["sha256"],
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise TokenizerPairBridgeError("Pair bridge temporary path already exists.")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


# ---------------------------------------------------------------------------
# Closed-form embedding-transport evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PairBridgeRowIncidence:
    """One (input row, position-in-code, exact rational weight) incidence."""

    input_row_index: int
    code_position: int
    code_length: int
    weight_numerator: int
    weight_denominator: int

    @property
    def weight(self) -> float:
        return self.weight_numerator / self.weight_denominator

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_row_index": self.input_row_index,
            "code_position": self.code_position,
            "code_length": self.code_length,
            "weight_numerator": self.weight_numerator,
            "weight_denominator": self.weight_denominator,
        }


@dataclass(frozen=True, slots=True)
class PairBridgeIncidenceIndex:
    """Every output row's incidence set under one weighting scheme.

    The index materializes one :class:`PairBridgeRowIncidence` per code entry,
    so it is a diagnostic surface.  The transport compilers stream the same
    incidences instead of building this, and stay linear in memory.
    """

    direction: str
    weighting: str
    input_row_capacity: int
    output_row_capacity: int
    incidences: tuple[tuple[PairBridgeRowIncidence, ...], ...]
    report: Mapping[str, Any]

    def row(self, output_row_index: int) -> tuple[PairBridgeRowIncidence, ...]:
        index = _nonnegative_int(output_row_index, "output_row_index")
        if index >= self.output_row_capacity:
            raise TokenizerPairBridgeError(
                "output_row_index is outside the output row capacity."
            )
        return self.incidences[index]


@dataclass(frozen=True, slots=True)
class PairBridgeTransportEvidence:
    """Closed-form output-row evidence plus its sealed provenance.

    This is evidence, not an install: nothing here writes a model parameter.
    ``evidence_mask`` marks the output rows that received at least one
    incidence; every other row is exactly zero and carries no claim.
    """

    evidence: Any
    evidence_mask: Any
    incidence_counts: Any
    weight_totals: Any
    report: Mapping[str, Any]

    @property
    def covered_rows(self) -> int:
        return int(self.report["covered_output_rows"])


def build_pair_bridge_row_incidences(
    bridge: TokenizerPairBridgeArtifact,
    *,
    direction: str = "forward",
    weighting: str = "length-normalized",
    token_frequencies: Mapping[int, int] | Sequence[int] | None = None,
    include_special_role: bool = True,
) -> PairBridgeIncidenceIndex:
    """Materialize every output row's weighted incidence set."""

    plan = _transport_plan(
        bridge,
        direction=direction,
        weighting=weighting,
        token_frequencies=token_frequencies,
        include_special_role=include_special_role,
    )
    buckets: list[list[PairBridgeRowIncidence]] = [
        [] for _ in range(plan.output_row_capacity)
    ]
    for input_row, position, output_row, numerator, denominator in plan.incidences():
        buckets[output_row].append(
            PairBridgeRowIncidence(
                input_row_index=input_row,
                code_position=position,
                code_length=len(plan.codes[input_row]),
                weight_numerator=numerator,
                weight_denominator=denominator,
            )
        )
    incidences = tuple(tuple(bucket) for bucket in buckets)
    body = {
        "schema": TOKENIZER_PAIR_INCIDENCE_SCHEMA,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "pair_profile_fingerprint": bridge.profile_fingerprint,
        "direction": plan.direction,
        "code_direction": plan.code_direction,
        "weighting": plan.weighting,
        "include_special_role": plan.include_special_role,
        "input_row_capacity": plan.input_row_capacity,
        "output_row_capacity": plan.output_row_capacity,
        "contributing_input_rows": plan.contributing_rows,
        "total_incidences": sum(len(bucket) for bucket in incidences),
        "covered_output_rows": sum(1 for bucket in incidences if bucket),
        "frequency_table_sha256": plan.frequency_sha256,
        **dict(_ZERO_MODEL_PARAMETER_OPERATIONS),
        "model_weight_reads": 0,
        "model_weight_writes": 0,
    }
    return PairBridgeIncidenceIndex(
        direction=plan.direction,
        weighting=plan.weighting,
        input_row_capacity=plan.input_row_capacity,
        output_row_capacity=plan.output_row_capacity,
        incidences=incidences,
        report={**body, "report_sha256": _canonical_sha256(body)},
    )


def compile_pair_bridge_anchor_transport(
    embedding_weight: Any,
    bridge: TokenizerPairBridgeArtifact,
    *,
    direction: str = "forward",
    include_special_role: bool = True,
    chunk_rows: int = _DEFAULT_TRANSPORT_CHUNK_ROWS,
) -> PairBridgeTransportEvidence:
    """Transport only the direct one-code rows into their output rows.

    An output row's evidence is the exact source row when a single direct code
    lands on it.  Multi-code incidences are ignored entirely, so uncovered rows
    stay masked out instead of being guessed at.
    """

    return _compile_pair_bridge_transport(
        embedding_weight,
        bridge,
        direction=direction,
        weighting="anchor-exact",
        token_frequencies=None,
        include_special_role=include_special_role,
        chunk_rows=chunk_rows,
    )


def compile_pair_bridge_length_normalized_transport(
    embedding_weight: Any,
    bridge: TokenizerPairBridgeArtifact,
    *,
    direction: str = "forward",
    include_special_role: bool = True,
    chunk_rows: int = _DEFAULT_TRANSPORT_CHUNK_ROWS,
) -> PairBridgeTransportEvidence:
    """Average every incidence with each input row's mass split across its code.

    An input row of code length ``k`` contributes weight ``1/k`` to each of its
    ``k`` output rows, and each output row is the weight-normalized mean of its
    incidences.  An output row whose only incidence is a direct one-code row
    therefore reproduces :func:`compile_pair_bridge_anchor_transport` exactly.
    """

    return _compile_pair_bridge_transport(
        embedding_weight,
        bridge,
        direction=direction,
        weighting="length-normalized",
        token_frequencies=None,
        include_special_role=include_special_role,
        chunk_rows=chunk_rows,
    )


def compile_pair_bridge_occurrence_weighted_transport(
    embedding_weight: Any,
    bridge: TokenizerPairBridgeArtifact,
    token_frequencies: Mapping[int, int] | Sequence[int],
    *,
    direction: str = "forward",
    include_special_role: bool = True,
    chunk_rows: int = _DEFAULT_TRANSPORT_CHUNK_ROWS,
) -> PairBridgeTransportEvidence:
    """Weight each incidence by ``frequency(input row) / code length``.

    ``token_frequencies`` is an exact nonnegative integer occurrence table over
    the input vocabulary, supplied either as a mapping or as a dense sequence.
    Input rows with zero observed occurrences contribute nothing, so output
    rows reached only by unobserved rows stay masked out.
    """

    return _compile_pair_bridge_transport(
        embedding_weight,
        bridge,
        direction=direction,
        weighting="occurrence-weighted",
        token_frequencies=token_frequencies,
        include_special_role=include_special_role,
        chunk_rows=chunk_rows,
    )


@dataclass(frozen=True, slots=True)
class _TransportPlan:
    direction: str
    code_direction: str
    weighting: str
    include_special_role: bool
    input_row_capacity: int
    output_row_capacity: int
    codes: tuple[tuple[int, ...], ...]
    code_classes: tuple[str, ...]
    frequencies: tuple[int, ...] | None
    frequency_sha256: str | None

    def contributes(self, input_row: int) -> bool:
        """Report whether one input row contributes any weighted incidence."""

        code_class = self.code_classes[input_row]
        if code_class == "special-role" and not self.include_special_role:
            return False
        if self.weighting == "anchor-exact" and code_class == "multi-code":
            return False
        if self.frequencies is not None and self.frequencies[input_row] == 0:
            return False
        return True

    @property
    def contributing_rows(self) -> int:
        return sum(
            1 for input_row in range(len(self.codes)) if self.contributes(input_row)
        )

    def incidences(self):
        """Yield ``(input row, position, output row, numerator, denominator)``."""

        for input_row, code in enumerate(self.codes):
            if not self.contributes(input_row):
                continue
            numerator = 1 if self.frequencies is None else self.frequencies[input_row]
            denominator = len(code)
            for position, output_row in enumerate(code):
                yield input_row, position, output_row, numerator, denominator


def _transport_plan(
    bridge: TokenizerPairBridgeArtifact,
    *,
    direction: str,
    weighting: str,
    token_frequencies: Mapping[int, int] | Sequence[int] | None,
    include_special_role: bool,
) -> _TransportPlan:
    if not isinstance(bridge, TokenizerPairBridgeArtifact):
        raise TokenizerPairBridgeError(
            "bridge must be a verified TokenizerPairBridgeArtifact."
        )
    bridge.verify_files()
    if not isinstance(include_special_role, bool):
        raise TokenizerPairBridgeError("include_special_role must be boolean.")
    if weighting not in (
        "anchor-exact",
        "length-normalized",
        "occurrence-weighted",
    ):
        raise TokenizerPairBridgeError(
            "weighting must be anchor-exact, length-normalized, or occurrence-weighted."
        )
    table = bridge.code_table(direction)
    profile = bridge.profile
    if direction == "forward":
        input_capacity = profile.source_row_capacity
        output_capacity = profile.target_row_capacity
    else:
        input_capacity = profile.target_row_capacity
        output_capacity = profile.source_row_capacity
    frequencies = None
    frequency_sha256 = None
    if weighting == "occurrence-weighted":
        if token_frequencies is None:
            raise TokenizerPairBridgeError(
                "occurrence-weighted transport requires token_frequencies."
            )
        frequencies = _frequency_table(token_frequencies, table.defined_rows)
        frequency_sha256 = _canonical_sha256(
            {
                "schema": "neural-forge-tokenizer-pair-frequency-table-v1",
                "defined_rows": table.defined_rows,
                "counts": list(frequencies),
            }
        )
    elif token_frequencies is not None:
        raise TokenizerPairBridgeError(
            "token_frequencies only applies to occurrence-weighted transport."
        )
    return _TransportPlan(
        direction=direction,
        code_direction=table.direction,
        weighting=weighting,
        include_special_role=include_special_role,
        input_row_capacity=input_capacity,
        output_row_capacity=output_capacity,
        codes=table.codes,
        code_classes=table.code_classes,
        frequencies=frequencies,
        frequency_sha256=frequency_sha256,
    )


def _frequency_table(
    token_frequencies: Mapping[int, int] | Sequence[int],
    defined_rows: int,
) -> tuple[int, ...]:
    counts = [0] * defined_rows
    if isinstance(token_frequencies, Mapping):
        for raw_row, raw_count in token_frequencies.items():
            row = _nonnegative_int(raw_row, "token_frequencies row")
            if row >= defined_rows:
                raise TokenizerPairBridgeError(
                    "token_frequencies references an undefined vocabulary row."
                )
            counts[row] = _nonnegative_int(raw_count, "token_frequencies count")
    elif isinstance(token_frequencies, Sequence) and not isinstance(
        token_frequencies,
        (str, bytes),
    ):
        if len(token_frequencies) != defined_rows:
            raise TokenizerPairBridgeError(
                "A dense token_frequencies table must cover every defined row."
            )
        for row, raw_count in enumerate(token_frequencies):
            counts[row] = _nonnegative_int(raw_count, "token_frequencies count")
    else:
        raise TokenizerPairBridgeError(
            "token_frequencies must be a mapping or a dense integer sequence."
        )
    if not any(counts):
        raise TokenizerPairBridgeError(
            "token_frequencies must observe at least one vocabulary row."
        )
    return tuple(counts)


def _compile_pair_bridge_transport(
    embedding_weight: Any,
    bridge: TokenizerPairBridgeArtifact,
    *,
    direction: str,
    weighting: str,
    token_frequencies: Mapping[int, int] | Sequence[int] | None,
    include_special_role: bool,
    chunk_rows: int,
) -> PairBridgeTransportEvidence:
    torch = _require_torch()
    plan = _transport_plan(
        bridge,
        direction=direction,
        weighting=weighting,
        token_frequencies=token_frequencies,
        include_special_role=include_special_role,
    )
    chunk = _positive_int(chunk_rows, "chunk_rows")
    rows = _float_matrix(torch, embedding_weight, "embedding_weight")
    if int(rows.shape[0]) != plan.input_row_capacity:
        raise TokenizerPairBridgeError(
            "embedding_weight rows differ from the sealed input row capacity."
        )
    width = int(rows.shape[1])

    input_index: list[int] = []
    output_index: list[int] = []
    weights: list[float] = []
    for input_row, _position, output_row, numerator, denominator in plan.incidences():
        input_index.append(input_row)
        output_index.append(output_row)
        weights.append(numerator / denominator)

    with torch.no_grad():
        totals = torch.zeros(
            (plan.output_row_capacity, width),
            dtype=torch.float64,
        )
        weight_totals = torch.zeros(plan.output_row_capacity, dtype=torch.float64)
        counts = torch.zeros(plan.output_row_capacity, dtype=torch.int64)
        if input_index:
            inputs = torch.tensor(input_index, dtype=torch.long)
            outputs = torch.tensor(output_index, dtype=torch.long)
            scale = torch.tensor(weights, dtype=torch.float64)
            for start in range(0, int(inputs.shape[0]), chunk):
                stop = start + chunk
                block = rows.index_select(0, inputs[start:stop])
                block = block * scale[start:stop].unsqueeze(1)
                totals.index_add_(0, outputs[start:stop], block)
            weight_totals.index_add_(0, outputs, scale)
            counts.index_add_(
                0,
                outputs,
                torch.ones(int(outputs.shape[0]), dtype=torch.int64),
            )
        mask = weight_totals > 0.0
        denominator = torch.where(
            mask,
            weight_totals,
            torch.ones((), dtype=torch.float64),
        )
        evidence = totals / denominator.unsqueeze(1)
        evidence = torch.where(
            mask.unsqueeze(1),
            evidence,
            torch.zeros((), dtype=torch.float64),
        )
        if not bool(torch.isfinite(evidence).all()):
            raise TokenizerPairBridgeError(
                "Compiled transport evidence contains NaN or infinity."
            )

    body: dict[str, Any] = {
        "schema": TOKENIZER_PAIR_TRANSPORT_SCHEMA,
        "operation": "compile_pair_bridge_transport",
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "pair_profile_fingerprint": bridge.profile_fingerprint,
        "source_tokenizer_semantics_sha256": (bridge.source_tokenizer_semantics_sha256),
        "target_tokenizer_semantics_sha256": (bridge.target_tokenizer_semantics_sha256),
        "direction": plan.direction,
        "code_direction": plan.code_direction,
        "weighting": plan.weighting,
        "include_special_role": plan.include_special_role,
        "input_row_capacity": plan.input_row_capacity,
        "output_row_capacity": plan.output_row_capacity,
        "width": width,
        "contributing_input_rows": plan.contributing_rows,
        "total_incidences": len(input_index),
        "covered_output_rows": int(mask.sum().item()),
        "uncovered_output_rows": (plan.output_row_capacity - int(mask.sum().item())),
        "frequency_table_sha256": plan.frequency_sha256,
        "accumulation": "cpu-float64-ordered-index-add",
        "evidence_only_no_parameter_install": True,
        "finite": True,
        **dict(_ZERO_MODEL_PARAMETER_OPERATIONS),
        "model_weight_writes": 0,
    }
    return PairBridgeTransportEvidence(
        evidence=evidence.detach(),
        evidence_mask=mask.detach(),
        incidence_counts=counts.detach(),
        weight_totals=weight_totals.detach(),
        report={**body, "report_sha256": _canonical_sha256(body)},
    )


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------


def _float_matrix(torch: Any, value: Any, name: str) -> Any:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise TokenizerPairBridgeError(f"{name} must be a rank-two torch tensor.")
    if not torch.is_floating_point(value) or torch.is_complex(value):
        raise TokenizerPairBridgeError(f"{name} must be a real floating tensor.")
    if int(value.shape[1]) <= 0:
        raise TokenizerPairBridgeError(f"{name} must have a positive width.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise TokenizerPairBridgeError(f"{name} contains NaN or infinity.")
    return result


def _relative_file_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TokenizerPairBridgeError("relative_path must be nonempty text.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise TokenizerPairBridgeError(
            "Pair bridge file paths must be normalized relative paths."
        )
    normalized = path.as_posix()
    if normalized == TOKENIZER_PAIR_BRIDGE_FILENAME:
        raise TokenizerPairBridgeError(
            "Pair bridge file seals may not include their manifest."
        )
    return normalized


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise TokenizerPairBridgeError(f"{name} must be positive.")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TokenizerPairBridgeError(f"{name} must be a nonnegative integer.")
    return value


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TokenizerPairBridgeError(f"{name} must be nonempty text.")
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TokenizerPairBridgeError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TokenizerPairBridgeError(
            "Tokenizer pair bridge payload is not canonical JSON."
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise TokenizerPairBridgeError(
            f"Cannot hash pair bridge artifact file {path.name!r}."
        ) from exc
    return digest.hexdigest()


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "PyTorch is required for tokenizer pair bridge transport evidence."
        ) from exc
    return torch


__all__ = [
    "CODE_RESOLUTION_ORDER",
    "FORWARD_DIRECTION",
    "PAIR_BRIDGE_ARTIFACT_CONFIG_KEY",
    "PAIR_BRIDGE_PROFILE_CONFIG_KEY",
    "PAIR_SOURCE_TOKENIZER_SEMANTICS_CONFIG_KEY",
    "PAIR_TARGET_TOKENIZER_SEMANTICS_CONFIG_KEY",
    "REVERSE_DIRECTION",
    "SEGMENTATION_ALGORITHM",
    "TOKENIZER_PAIR_BRIDGE_CODES_FILENAME",
    "TOKENIZER_PAIR_BRIDGE_CONFIG_KEYS",
    "TOKENIZER_PAIR_BRIDGE_FILENAME",
    "TOKENIZER_PAIR_BRIDGE_KIND",
    "TOKENIZER_PAIR_BRIDGE_SCHEMA",
    "TOKENIZER_PAIR_INCIDENCE_SCHEMA",
    "TOKENIZER_PAIR_TRANSPORT_SCHEMA",
    "PairBridgeArtifactFile",
    "PairBridgeIncidenceIndex",
    "PairBridgeRowIncidence",
    "PairBridgeTransportEvidence",
    "TokenizerPairBridgeArtifact",
    "TokenizerPairBridgeError",
    "TokenizerPairCodeProfile",
    "TokenizerPairCodeTable",
    "build_hf_tokenizer_pair_code_profile",
    "build_pair_bridge_row_incidences",
    "compile_hf_tokenizer_pair_bridge",
    "compile_pair_bridge_anchor_transport",
    "compile_pair_bridge_length_normalized_transport",
    "compile_pair_bridge_occurrence_weighted_transport",
    "load_hf_tokenizer_pair_bridge_artifact",
    "tokenizer_pair_bridge_configuration",
]
