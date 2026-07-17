"""Sealed strict tokenizer compilation and vocabulary-operator transport.

This module closes the reusable boundary between target-vocabulary compilation
and dense blank-target folding.  It does not prune, quantize, train, or mutate
model parameters.  A target tokenizer may contain fewer lexical rows than its
donor; every physical target row must still resolve to exactly one authenticated
donor row before embedding or language-head transport is allowed.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from neural_forge.hf_token_align import (
    StrictTokenRowMapping,
    StrictTokenTransportProfile,
    StrictTokenTransportRejectedError,
    build_strict_token_transport_profile,
)
from neural_forge.hf_tokenizer_prune import prune_hf_tokenizer
from neural_forge.hf_transport_math import CoordinateMap, encode_coordinates


STRICT_TOKENIZER_BRIDGE_FILENAME = "neural_forge_strict_tokenizer_bridge.json"
STRICT_TOKENIZER_BRIDGE_SCHEMA = "neural-forge-strict-tokenizer-bridge-v1"
_STRICT_TOKENIZER_BRIDGE_KIND = "sealed-strict-target-tokenizer"
TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY = "_neural_forge_tokenizer_bridge_artifact_sha256"
TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY = "_neural_forge_tokenizer_transport_profile_sha256"
PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY = (
    "_neural_forge_parent_tokenizer_semantics_sha256"
)
CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY = "_neural_forge_child_tokenizer_semantics_sha256"
TOKENIZER_BRIDGE_CONFIG_KEYS = frozenset(
    {
        TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY,
        TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY,
        PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY,
        CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY,
    }
)


class StrictTokenizerBridgeError(ValueError):
    """Raised when a target-tokenizer bridge cannot be proven authentic."""


@dataclass(frozen=True, slots=True)
class TokenizerArtifactFile:
    """One immutable target-tokenizer file record."""

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        normalized = _relative_file_path(self.relative_path)
        object.__setattr__(self, "relative_path", normalized)
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
class StrictTokenizerBridgeArtifact:
    """Verified target files and their complete one-row transport profile."""

    root_dir: Path
    transport_profile: StrictTokenTransportProfile
    donor_tokenizer_semantics_sha256: str
    target_tokenizer_semantics_sha256: str
    target_files: tuple[TokenizerArtifactFile, ...]
    artifact_sha256: str

    def __post_init__(self) -> None:
        root = Path(self.root_dir).resolve()
        object.__setattr__(self, "root_dir", root)
        if not isinstance(self.transport_profile, StrictTokenTransportProfile):
            raise StrictTokenizerBridgeError(
                "transport_profile must be a StrictTokenTransportProfile."
            )
        object.__setattr__(
            self,
            "donor_tokenizer_semantics_sha256",
            _sha256(
                self.donor_tokenizer_semantics_sha256,
                "donor_tokenizer_semantics_sha256",
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
        files = tuple(self.target_files)
        if not files or any(
            not isinstance(item, TokenizerArtifactFile) for item in files
        ):
            raise StrictTokenizerBridgeError(
                "target_files must contain typed tokenizer file seals."
            )
        paths = tuple(item.relative_path for item in files)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise StrictTokenizerBridgeError(
                "target_files must be uniquely sorted by relative path."
            )
        object.__setattr__(self, "target_files", files)
        object.__setattr__(
            self,
            "artifact_sha256",
            _sha256(self.artifact_sha256, "artifact_sha256"),
        )
        if _canonical_sha256(self._body()) != self.artifact_sha256:
            raise StrictTokenizerBridgeError(
                "Strict tokenizer bridge artifact seal is invalid."
            )

    @property
    def profile_fingerprint(self) -> str:
        return self.transport_profile.fingerprint()

    @property
    def target_vocabulary_size(self) -> int:
        return self.transport_profile.target_row_capacity

    def _body(self) -> dict[str, Any]:
        return {
            "schema": STRICT_TOKENIZER_BRIDGE_SCHEMA,
            "kind": _STRICT_TOKENIZER_BRIDGE_KIND,
            "profile_fingerprint": self.transport_profile.fingerprint(),
            "transport_profile": self.transport_profile.to_dict(),
            "donor_tokenizer_semantics_sha256": (self.donor_tokenizer_semantics_sha256),
            "target_tokenizer_semantics_sha256": (
                self.target_tokenizer_semantics_sha256
            ),
            "target_files": [item.to_dict() for item in self.target_files],
            "model_parameter_operations": {
                "training_steps": 0,
                "backward_calls": 0,
                "pruned_parameters": 0,
                "quantized_parameters": 0,
                "random_target_rows": 0,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "artifact_sha256": self.artifact_sha256}

    def verify_files(self) -> None:
        """Re-authenticate every tokenizer file and reject additions/removals."""

        if self.root_dir.is_symlink() or not self.root_dir.is_dir():
            raise StrictTokenizerBridgeError(
                "Strict tokenizer bridge root must be a real directory."
            )
        observed = _target_file_records(self.root_dir)
        if observed != self.target_files:
            raise StrictTokenizerBridgeError(
                "Target tokenizer files differ from the sealed artifact."
            )
        if _canonical_sha256(self._body()) != self.artifact_sha256:
            raise StrictTokenizerBridgeError(
                "Strict tokenizer bridge artifact changed after construction."
            )

    def verify_live_tokenizers(
        self,
        donor_tokenizer: Any,
        target_tokenizer: Any,
    ) -> None:
        """Bind live tokenizer objects back to the stored row contract."""

        self.verify_files()
        donor_sha = fingerprint_hf_tokenizer_semantics(donor_tokenizer)
        target_sha = fingerprint_hf_tokenizer_semantics(target_tokenizer)
        if donor_sha != self.donor_tokenizer_semantics_sha256:
            raise StrictTokenizerBridgeError(
                "Live donor tokenizer differs from the sealed bridge."
            )
        if target_sha != self.target_tokenizer_semantics_sha256:
            raise StrictTokenizerBridgeError(
                "Live target tokenizer differs from the sealed bridge."
            )
        try:
            observed_profile = build_strict_token_transport_profile(
                donor_tokenizer,
                target_tokenizer,
                donor_row_capacity=self.transport_profile.donor_row_capacity,
                target_row_capacity=self.transport_profile.target_row_capacity,
            )
        except StrictTokenTransportRejectedError as exc:
            raise StrictTokenizerBridgeError(
                "Live tokenizers no longer satisfy strict row transport."
            ) from exc
        if observed_profile.fingerprint() != self.profile_fingerprint:
            raise StrictTokenizerBridgeError(
                "Live tokenizer row mapping differs from the sealed profile."
            )


@dataclass(frozen=True, slots=True)
class CompiledTargetTokenizerBridge:
    """A vocabulary-compiled tokenizer and its persisted strict bridge."""

    tokenizer: Any
    output_dir: Path
    target_vocab_size: int
    selected_donor_token_ids: tuple[int, ...]
    retained_merges: tuple[tuple[str, str], ...]
    artifact: StrictTokenizerBridgeArtifact

    @property
    def transport_profile(self) -> StrictTokenTransportProfile:
        return self.artifact.transport_profile

    @property
    def tokenizer_fingerprint(self) -> str:
        return self.artifact.target_tokenizer_semantics_sha256


@dataclass(frozen=True, slots=True)
class StrictVocabularyOperatorTransportResult:
    """Detached dense target rows ready for an atomic blank-child install."""

    embedding_weight: Any
    lm_head_weight: Any
    lm_head_bias: Any | None
    report: Mapping[str, Any]


class DenseVocabularyProjector(Protocol):
    """Typed residual-coordinate surface used by dense folding bases."""

    source_width: int
    target_width: int
    map_sha256: str

    def fold_consumer_weight(self, source_weight: Any) -> Any: ...


def fingerprint_hf_tokenizer_semantics(tokenizer: Any) -> str:
    """Hash stable row/tokenization semantics without path-local metadata."""

    try:
        length = int(len(tokenizer))
    except (TypeError, ValueError, OverflowError) as exc:
        raise StrictTokenizerBridgeError(
            "Tokenizer length must be a positive integer."
        ) from exc
    if length <= 0:
        raise StrictTokenizerBridgeError("Tokenizer length must be a positive integer.")
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise StrictTokenizerBridgeError("Tokenizer must expose get_vocab().")
    vocabulary = get_vocab()
    if not isinstance(vocabulary, Mapping) or len(vocabulary) != length:
        raise StrictTokenizerBridgeError(
            "Tokenizer vocabulary must match its reported length."
        )
    rows: list[tuple[int, str]] = []
    for token, raw_token_id in vocabulary.items():
        if not isinstance(token, str):
            raise StrictTokenizerBridgeError(
                "Tokenizer vocabulary tokens must be strings."
            )
        token_id = _nonnegative_int(raw_token_id, "token_id")
        rows.append((token_id, token))
    rows.sort()
    if tuple(token_id for token_id, _token in rows) != tuple(range(length)):
        raise StrictTokenizerBridgeError(
            "Tokenizer vocabulary IDs must be contiguous from zero."
        )

    raw_rows = []
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(converter):
        for token_id in range(length):
            value = converter(token_id)
            if value is not None and not isinstance(value, str):
                raise StrictTokenizerBridgeError(
                    "Tokenizer raw token rows must be strings or None."
                )
            raw_rows.append(value)

    decoded_rows = []
    decoder = getattr(tokenizer, "decode", None)
    if not callable(decoder):
        raise StrictTokenizerBridgeError("Tokenizer must expose decode().")
    for token_id in range(length):
        try:
            value = decoder(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            value = decoder([token_id])
        if not isinstance(value, str):
            raise StrictTokenizerBridgeError("Tokenizer decoded rows must be strings.")
        decoded_rows.append(value)

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    backend_sha256 = None
    if backend is not None:
        serializer = getattr(backend, "to_str", None)
        if not callable(serializer):
            raise StrictTokenizerBridgeError("Tokenizer backend must expose to_str().")
        serialized = serializer()
        if not isinstance(serialized, str):
            raise StrictTokenizerBridgeError(
                "Tokenizer backend serialization must be text."
            )
        try:
            backend_payload = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise StrictTokenizerBridgeError(
                "Tokenizer backend serialization must be valid JSON."
            ) from exc
        backend_sha256 = _canonical_sha256(backend_payload)

    special_roles: dict[str, int | None] = {}
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        value = getattr(tokenizer, name, None)
        if value is None:
            special_roles[name] = None
        else:
            token_id = _nonnegative_int(value, name)
            if token_id >= length:
                raise StrictTokenizerBridgeError(
                    f"{name} is outside tokenizer capacity."
                )
            special_roles[name] = token_id
    all_special_ids = getattr(tokenizer, "all_special_ids", ())
    if isinstance(all_special_ids, (str, bytes)) or not isinstance(
        all_special_ids, Sequence
    ):
        raise StrictTokenizerBridgeError(
            "Tokenizer all_special_ids must be a sequence."
        )
    normalized_special_ids = tuple(
        sorted(
            {_nonnegative_int(value, "special_token_id") for value in all_special_ids}
        )
    )
    if normalized_special_ids and normalized_special_ids[-1] >= length:
        raise StrictTokenizerBridgeError(
            "Tokenizer special token ID is outside tokenizer capacity."
        )
    payload = {
        "schema": "neural-forge-tokenizer-semantics-v1",
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "length": length,
        "vocabulary_rows": [[token_id, token] for token_id, token in rows],
        "raw_rows_sha256": _canonical_sha256(raw_rows),
        "decoded_rows_sha256": _canonical_sha256(decoded_rows),
        "backend_sha256": backend_sha256,
        "special_roles": special_roles,
        "all_special_ids": list(normalized_special_ids),
    }
    return _canonical_sha256(payload)


def strict_tokenizer_bridge_configuration(
    bridge: StrictTokenizerBridgeArtifact,
) -> dict[str, str]:
    """Return the exact configuration seal bundle for one verified bridge."""

    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise StrictTokenizerBridgeError(
            "bridge must be a StrictTokenizerBridgeArtifact."
        )
    bridge.verify_files()
    return {
        TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY: bridge.artifact_sha256,
        TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY: bridge.profile_fingerprint,
        PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY: (
            bridge.donor_tokenizer_semantics_sha256
        ),
        CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY: (
            bridge.target_tokenizer_semantics_sha256
        ),
    }


def seal_strict_tokenizer_bridge_artifact(
    donor_tokenizer: Any,
    target_tokenizer: Any,
    target_tokenizer_dir: str | Path,
    *,
    donor_row_capacity: int | None = None,
    target_row_capacity: int | None = None,
) -> StrictTokenizerBridgeArtifact:
    """Persist a canonical bridge manifest next to a saved target tokenizer."""

    root = Path(target_tokenizer_dir).resolve()
    if root.is_symlink() or not root.is_dir():
        raise StrictTokenizerBridgeError(
            "target_tokenizer_dir must be a real existing directory."
        )
    try:
        profile = build_strict_token_transport_profile(
            donor_tokenizer,
            target_tokenizer,
            donor_row_capacity=donor_row_capacity,
            target_row_capacity=target_row_capacity,
        )
    except StrictTokenTransportRejectedError as exc:
        raise StrictTokenizerBridgeError(
            "Target tokenizer lacks a complete strict donor-row mapping."
        ) from exc
    files = _target_file_records(root)
    donor_sha = fingerprint_hf_tokenizer_semantics(donor_tokenizer)
    target_sha = fingerprint_hf_tokenizer_semantics(target_tokenizer)
    body = _artifact_body(profile, donor_sha, target_sha, files)
    artifact_sha = _canonical_sha256(body)
    artifact = StrictTokenizerBridgeArtifact(
        root_dir=root,
        transport_profile=profile,
        donor_tokenizer_semantics_sha256=donor_sha,
        target_tokenizer_semantics_sha256=target_sha,
        target_files=files,
        artifact_sha256=artifact_sha,
    )
    manifest_path = root / STRICT_TOKENIZER_BRIDGE_FILENAME
    if manifest_path.exists():
        existing = load_strict_tokenizer_bridge_artifact(
            root,
            donor_tokenizer=donor_tokenizer,
            target_tokenizer=target_tokenizer,
        )
        if existing.artifact_sha256 != artifact.artifact_sha256:
            raise StrictTokenizerBridgeError(
                "A different strict tokenizer bridge is already sealed here."
            )
        return existing

    temporary = root / (f".{STRICT_TOKENIZER_BRIDGE_FILENAME}.{os.getpid()}.tmp")
    if temporary.exists():
        raise StrictTokenizerBridgeError(
            "Tokenizer bridge temporary path already exists."
        )
    try:
        payload = _canonical_json(artifact.to_dict()) + b"\n"
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return load_strict_tokenizer_bridge_artifact(
        root,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=artifact_sha,
    )


def load_strict_tokenizer_bridge_artifact(
    target_tokenizer_dir: str | Path,
    *,
    donor_tokenizer: Any | None = None,
    target_tokenizer: Any | None = None,
    expected_artifact_sha256: str | None = None,
) -> StrictTokenizerBridgeArtifact:
    """Load and re-authenticate a persisted strict tokenizer bridge."""

    root = Path(target_tokenizer_dir).resolve()
    if root.is_symlink() or not root.is_dir():
        raise StrictTokenizerBridgeError(
            "Strict tokenizer bridge root must be a real directory."
        )
    manifest_path = root / STRICT_TOKENIZER_BRIDGE_FILENAME
    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise StrictTokenizerBridgeError(
            "Strict tokenizer bridge manifest is missing or unreadable."
        ) from exc
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StrictTokenizerBridgeError(
            "Strict tokenizer bridge manifest is not valid JSON."
        ) from exc
    if not isinstance(payload, Mapping):
        raise StrictTokenizerBridgeError(
            "Strict tokenizer bridge manifest must be a JSON object."
        )
    expected_keys = {
        "schema",
        "kind",
        "profile_fingerprint",
        "transport_profile",
        "donor_tokenizer_semantics_sha256",
        "target_tokenizer_semantics_sha256",
        "target_files",
        "model_parameter_operations",
        "artifact_sha256",
    }
    if set(payload) != expected_keys:
        raise StrictTokenizerBridgeError(
            "Strict tokenizer bridge manifest fields are incomplete or unknown."
        )
    if (
        payload["schema"] != STRICT_TOKENIZER_BRIDGE_SCHEMA
        or payload["kind"] != _STRICT_TOKENIZER_BRIDGE_KIND
    ):
        raise StrictTokenizerBridgeError(
            "Strict tokenizer bridge schema or kind is unsupported."
        )
    profile = _strict_profile_from_dict(payload["transport_profile"])
    if payload["profile_fingerprint"] != profile.fingerprint():
        raise StrictTokenizerBridgeError(
            "Strict tokenizer transport profile fingerprint is invalid."
        )
    file_values = payload["target_files"]
    if not isinstance(file_values, list):
        raise StrictTokenizerBridgeError("target_files must be a JSON list.")
    files = tuple(_file_record_from_dict(value) for value in file_values)
    operations = payload["model_parameter_operations"]
    expected_operations = {
        "training_steps": 0,
        "backward_calls": 0,
        "pruned_parameters": 0,
        "quantized_parameters": 0,
        "random_target_rows": 0,
    }
    if operations != expected_operations:
        raise StrictTokenizerBridgeError(
            "Tokenizer bridge model-parameter operation audit is invalid."
        )
    artifact = StrictTokenizerBridgeArtifact(
        root_dir=root,
        transport_profile=profile,
        donor_tokenizer_semantics_sha256=payload["donor_tokenizer_semantics_sha256"],
        target_tokenizer_semantics_sha256=payload["target_tokenizer_semantics_sha256"],
        target_files=files,
        artifact_sha256=payload["artifact_sha256"],
    )
    artifact.verify_files()
    if expected_artifact_sha256 is not None and artifact.artifact_sha256 != _sha256(
        expected_artifact_sha256,
        "expected_artifact_sha256",
    ):
        raise StrictTokenizerBridgeError(
            "Strict tokenizer bridge differs from the expected artifact."
        )
    if (donor_tokenizer is None) != (target_tokenizer is None):
        raise StrictTokenizerBridgeError(
            "Live verification requires both donor and target tokenizers."
        )
    if donor_tokenizer is not None:
        artifact.verify_live_tokenizers(donor_tokenizer, target_tokenizer)
    return artifact


def compile_hf_target_tokenizer_bridge(
    donor_tokenizer: Any,
    output_dir: str | Path,
    *,
    target_vocab_size: int,
    calibration_texts: Sequence[str] | None = None,
    donor_row_capacity: int | None = None,
) -> CompiledTargetTokenizerBridge:
    """Compile a smaller ByteLevel-BPE target vocabulary and seal its bridge.

    The existing deterministic vocabulary selector is reused internally.  This
    operation changes tokenizer vocabulary rows only; it never removes or edits
    a learned model parameter.
    """

    compilation = prune_hf_tokenizer(
        donor_tokenizer,
        output_dir,
        target_vocab_size=target_vocab_size,
        calibration_texts=calibration_texts,
    )
    artifact = seal_strict_tokenizer_bridge_artifact(
        donor_tokenizer,
        compilation.tokenizer,
        compilation.output_dir,
        donor_row_capacity=donor_row_capacity,
        target_row_capacity=compilation.target_vocab_size,
    )
    if (
        artifact.transport_profile.target_to_donor_row_indices
        != compilation.selected_donor_token_ids
    ):
        raise StrictTokenizerBridgeError(
            "Sealed bridge mapping differs from vocabulary compilation selection."
        )
    return CompiledTargetTokenizerBridge(
        tokenizer=compilation.tokenizer,
        output_dir=compilation.output_dir,
        target_vocab_size=compilation.target_vocab_size,
        selected_donor_token_ids=compilation.selected_donor_token_ids,
        retained_merges=compilation.retained_merges,
        artifact=artifact,
    )


def compile_strict_vocabulary_operators(
    source_embedding_weight: Any,
    source_lm_head_weight: Any,
    bridge: StrictTokenizerBridgeArtifact,
    embedding_coordinates: CoordinateMap | DenseVocabularyProjector,
    lm_head_coordinates: CoordinateMap | DenseVocabularyProjector,
    *,
    source_lm_head_bias: Any | None = None,
    require_tied_target: bool = False,
) -> StrictVocabularyOperatorTransportResult:
    """Compile authenticated donor rows into dense blank-child coordinates."""

    torch = _require_torch()
    if not isinstance(bridge, StrictTokenizerBridgeArtifact):
        raise StrictTokenizerBridgeError(
            "bridge must be a verified StrictTokenizerBridgeArtifact."
        )
    bridge.verify_files()
    if not isinstance(require_tied_target, bool):
        raise StrictTokenizerBridgeError("require_tied_target must be boolean.")
    profile = bridge.transport_profile
    embedding = _float_matrix(
        torch,
        source_embedding_weight,
        "source_embedding_weight",
    )
    head = _float_matrix(torch, source_lm_head_weight, "source_lm_head_weight")
    if int(embedding.shape[0]) != profile.donor_row_capacity:
        raise StrictTokenizerBridgeError(
            "Source embedding rows differ from sealed donor row capacity."
        )
    if int(head.shape[0]) != profile.donor_row_capacity:
        raise StrictTokenizerBridgeError(
            "Source language-head rows differ from sealed donor row capacity."
        )
    embedding_source_dim, embedding_target_dim, embedding_map_sha256 = (
        _projector_contract(embedding_coordinates, "embedding_coordinates")
    )
    head_source_dim, head_target_dim, head_map_sha256 = _projector_contract(
        lm_head_coordinates,
        "lm_head_coordinates",
    )
    if int(embedding.shape[1]) != embedding_source_dim:
        raise StrictTokenizerBridgeError(
            "Source embedding width differs from embedding coordinates."
        )
    if int(head.shape[1]) != head_source_dim:
        raise StrictTokenizerBridgeError(
            "Source language-head width differs from head coordinates."
        )
    bias = None
    if source_lm_head_bias is not None:
        bias = _float_vector(
            torch,
            source_lm_head_bias,
            "source_lm_head_bias",
            profile.donor_row_capacity,
        )

    with torch.no_grad():
        indices = torch.tensor(
            profile.target_to_donor_row_indices,
            dtype=torch.long,
        )
        target_source_embedding = embedding.index_select(0, indices)
        target_source_head = head.index_select(0, indices)
        target_embedding = _project_vocabulary_rows(
            target_source_embedding,
            embedding_coordinates,
            "embedding_coordinates",
        )
        target_head = _project_vocabulary_rows(
            target_source_head,
            lm_head_coordinates,
            "lm_head_coordinates",
        )
        target_bias = (
            None if bias is None else bias.index_select(0, indices).contiguous()
        )
        expected_embedding = (
            profile.target_row_capacity,
            embedding_target_dim,
        )
        expected_head = (
            profile.target_row_capacity,
            head_target_dim,
        )
        if tuple(target_embedding.shape) != expected_embedding:
            raise StrictTokenizerBridgeError("Compiled target embedding is incomplete.")
        if tuple(target_head.shape) != expected_head:
            raise StrictTokenizerBridgeError(
                "Compiled target language head is incomplete."
            )
        if target_bias is not None and tuple(target_bias.shape) != (
            profile.target_row_capacity,
        ):
            raise StrictTokenizerBridgeError(
                "Compiled target language-head bias is incomplete."
            )
        if require_tied_target:
            if tuple(target_embedding.shape) != tuple(
                target_head.shape
            ) or not torch.equal(
                target_embedding,
                target_head,
            ):
                raise StrictTokenizerBridgeError(
                    "A tied target requires exactly identical compiled embedding "
                    "and language-head weights."
                )
            target_head = target_embedding

    selected = profile.target_to_donor_row_indices
    report: dict[str, Any] = {
        "schema": "neural-forge-strict-vocabulary-operator-transport-v1",
        "operation": "compile_strict_vocabulary_operators",
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "transport_profile_fingerprint": bridge.profile_fingerprint,
        "donor_row_capacity": profile.donor_row_capacity,
        "target_row_capacity": profile.target_row_capacity,
        "selected_unique_donor_rows": len(set(selected)),
        "selected_donor_row_fraction": len(set(selected)) / profile.donor_row_capacity,
        "embedding_shape": list(target_embedding.shape),
        "lm_head_shape": list(target_head.shape),
        "embedding_coordinate_sha256": embedding_map_sha256,
        "lm_head_coordinate_sha256": head_map_sha256,
        "lm_head_bias_present": target_bias is not None,
        "target_tied_storage_ready": require_tied_target,
        "dense_target_rows": True,
        "random_target_rows": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "pruned_model_parameters": 0,
        "quantized_model_parameters": 0,
        "finite": True,
    }
    report["report_sha256"] = _canonical_sha256(report)
    detached_embedding = target_embedding.detach()
    detached_head = detached_embedding if require_tied_target else target_head.detach()
    return StrictVocabularyOperatorTransportResult(
        embedding_weight=detached_embedding,
        lm_head_weight=detached_head,
        lm_head_bias=None if target_bias is None else target_bias.detach(),
        report=report,
    )


def _projector_contract(
    value: CoordinateMap | DenseVocabularyProjector,
    name: str,
) -> tuple[int, int, str]:
    if isinstance(value, CoordinateMap):
        source = _positive_int(value.source_dim, f"{name}.source_dim")
        target = _positive_int(value.target_dim, f"{name}.target_dim")
        return source, target, _tensor_sha256(value.decode)
    source = _positive_int(getattr(value, "source_width", None), f"{name}.source_width")
    target = _positive_int(getattr(value, "target_width", None), f"{name}.target_width")
    digest = _sha256(getattr(value, "map_sha256", None), f"{name}.map_sha256")
    if not callable(getattr(value, "fold_consumer_weight", None)):
        raise StrictTokenizerBridgeError(f"{name} must expose fold_consumer_weight().")
    return source, target, digest


def _project_vocabulary_rows(
    rows: Any,
    coordinates: CoordinateMap | DenseVocabularyProjector,
    name: str,
) -> Any:
    torch = _require_torch()
    if isinstance(coordinates, CoordinateMap):
        result = encode_coordinates(rows, coordinates)
    else:
        try:
            result = coordinates.fold_consumer_weight(rows)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise StrictTokenizerBridgeError(
                f"{name} rejected vocabulary rows."
            ) from exc
    if (
        not isinstance(result, torch.Tensor)
        or result.ndim != 2
        or not torch.is_floating_point(result)
        or torch.is_complex(result)
    ):
        raise StrictTokenizerBridgeError(
            f"{name} returned an invalid projected matrix."
        )
    result = result.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise StrictTokenizerBridgeError(f"{name} returned NaN or infinity.")
    return result


def _artifact_body(
    profile: StrictTokenTransportProfile,
    donor_sha: str,
    target_sha: str,
    files: tuple[TokenizerArtifactFile, ...],
) -> dict[str, Any]:
    return {
        "schema": STRICT_TOKENIZER_BRIDGE_SCHEMA,
        "kind": _STRICT_TOKENIZER_BRIDGE_KIND,
        "profile_fingerprint": profile.fingerprint(),
        "transport_profile": profile.to_dict(),
        "donor_tokenizer_semantics_sha256": donor_sha,
        "target_tokenizer_semantics_sha256": target_sha,
        "target_files": [item.to_dict() for item in files],
        "model_parameter_operations": {
            "training_steps": 0,
            "backward_calls": 0,
            "pruned_parameters": 0,
            "quantized_parameters": 0,
            "random_target_rows": 0,
        },
    }


def _target_file_records(root: Path) -> tuple[TokenizerArtifactFile, ...]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise StrictTokenizerBridgeError(
                "Tokenizer artifacts may not contain symbolic links."
            )
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == STRICT_TOKENIZER_BRIDGE_FILENAME:
            continue
        if not path.is_file():
            raise StrictTokenizerBridgeError(
                f"Tokenizer artifact entry is not a regular file: {relative}"
            )
        records.append(
            TokenizerArtifactFile(
                relative_path=relative,
                size_bytes=path.stat().st_size,
                sha256=_file_sha256(path),
            )
        )
    if not records:
        raise StrictTokenizerBridgeError(
            "Target tokenizer directory contains no sealable files."
        )
    return tuple(records)


def _file_record_from_dict(value: Any) -> TokenizerArtifactFile:
    if not isinstance(value, Mapping) or set(value) != {
        "relative_path",
        "size_bytes",
        "sha256",
    }:
        raise StrictTokenizerBridgeError(
            "Tokenizer file seal fields are incomplete or unknown."
        )
    return TokenizerArtifactFile(
        relative_path=value["relative_path"],
        size_bytes=value["size_bytes"],
        sha256=value["sha256"],
    )


def _strict_profile_from_dict(value: Any) -> StrictTokenTransportProfile:
    if not isinstance(value, Mapping):
        raise StrictTokenizerBridgeError("transport_profile must be a JSON object.")
    expected = {
        "schema_version",
        "profile",
        "direction",
        "donor_tokenizer_class",
        "target_tokenizer_class",
        "donor_vocab_size",
        "target_vocab_size",
        "donor_row_capacity",
        "target_row_capacity",
        "alignment_fingerprint",
        "alignment_resolution",
        "target_to_donor_row_indices",
        "row_mappings",
    }
    if set(value) != expected:
        raise StrictTokenizerBridgeError(
            "Strict transport profile fields are incomplete or unknown."
        )
    if (
        value["schema_version"] != 1
        or value["profile"] != "strict-cross-tokenizer-one-row-v1"
        or value["direction"] != "target-row-to-donor-row"
    ):
        raise StrictTokenizerBridgeError(
            "Strict transport profile schema is unsupported."
        )
    mappings_value = value["row_mappings"]
    if not isinstance(mappings_value, list):
        raise StrictTokenizerBridgeError("row_mappings must be a JSON list.")
    mappings = []
    for item in mappings_value:
        if not isinstance(item, Mapping) or set(item) != {
            "target_row_index",
            "donor_row_index",
            "resolution",
            "special_token_roles",
        }:
            raise StrictTokenizerBridgeError(
                "Strict row mapping fields are incomplete or unknown."
            )
        roles = item["special_token_roles"]
        if not isinstance(roles, list) or any(
            not isinstance(role, str) or not role for role in roles
        ):
            raise StrictTokenizerBridgeError(
                "special_token_roles must contain nonempty strings."
            )
        resolution = item["resolution"]
        if not isinstance(resolution, str) or not resolution:
            raise StrictTokenizerBridgeError(
                "Strict row mapping resolution must be nonempty."
            )
        mappings.append(
            StrictTokenRowMapping(
                target_row_index=_nonnegative_int(
                    item["target_row_index"],
                    "target_row_index",
                ),
                donor_row_index=_nonnegative_int(
                    item["donor_row_index"],
                    "donor_row_index",
                ),
                resolution=resolution,
                special_token_roles=tuple(roles),
            )
        )
    donor_class = _nonempty_text(
        value["donor_tokenizer_class"],
        "donor_tokenizer_class",
    )
    target_class = _nonempty_text(
        value["target_tokenizer_class"],
        "target_tokenizer_class",
    )
    alignment_resolution = _nonempty_text(
        value["alignment_resolution"],
        "alignment_resolution",
    )
    profile = StrictTokenTransportProfile(
        donor_tokenizer_class=donor_class,
        target_tokenizer_class=target_class,
        donor_vocab_size=_positive_int(
            value["donor_vocab_size"],
            "donor_vocab_size",
        ),
        target_vocab_size=_positive_int(
            value["target_vocab_size"],
            "target_vocab_size",
        ),
        donor_row_capacity=_positive_int(
            value["donor_row_capacity"],
            "donor_row_capacity",
        ),
        target_row_capacity=_positive_int(
            value["target_row_capacity"],
            "target_row_capacity",
        ),
        alignment_fingerprint=_sha256(
            value["alignment_fingerprint"],
            "alignment_fingerprint",
        ),
        row_mappings=tuple(mappings),
        alignment_resolution=alignment_resolution,
    )
    if profile.donor_vocab_size > profile.donor_row_capacity:
        raise StrictTokenizerBridgeError(
            "Donor tokenizer vocabulary exceeds donor row capacity."
        )
    if profile.target_vocab_size > profile.target_row_capacity:
        raise StrictTokenizerBridgeError(
            "Target tokenizer vocabulary exceeds target row capacity."
        )
    indices = value["target_to_donor_row_indices"]
    if not isinstance(indices, list) or tuple(indices) != (
        profile.target_to_donor_row_indices
    ):
        raise StrictTokenizerBridgeError(
            "Strict transport index summary differs from row mappings."
        )
    if profile.to_dict() != dict(value):
        raise StrictTokenizerBridgeError(
            "Strict transport profile does not round-trip canonically."
        )
    return profile


def _float_matrix(torch: Any, value: Any, name: str) -> Any:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise StrictTokenizerBridgeError(f"{name} must be a rank-two torch tensor.")
    if not torch.is_floating_point(value) or torch.is_complex(value):
        raise StrictTokenizerBridgeError(f"{name} must be a real floating tensor.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise StrictTokenizerBridgeError(f"{name} contains NaN or infinity.")
    return result


def _float_vector(torch: Any, value: Any, name: str, size: int) -> Any:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (size,):
        raise StrictTokenizerBridgeError(f"{name} must have shape ({size},).")
    if not torch.is_floating_point(value) or torch.is_complex(value):
        raise StrictTokenizerBridgeError(f"{name} must be a real floating tensor.")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise StrictTokenizerBridgeError(f"{name} contains NaN or infinity.")
    return result


def _relative_file_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise StrictTokenizerBridgeError("relative_path must be nonempty text.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise StrictTokenizerBridgeError(
            "Tokenizer file paths must be normalized relative paths."
        )
    normalized = path.as_posix()
    if normalized == STRICT_TOKENIZER_BRIDGE_FILENAME:
        raise StrictTokenizerBridgeError(
            "Tokenizer file seals may not include their manifest."
        )
    return normalized


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise StrictTokenizerBridgeError(f"{name} must be positive.")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrictTokenizerBridgeError(f"{name} must be a nonnegative integer.")
    return value


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrictTokenizerBridgeError(f"{name} must be nonempty text.")
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StrictTokenizerBridgeError(f"{name} must be a lowercase SHA-256 digest.")
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
        raise StrictTokenizerBridgeError(
            "Tokenizer bridge payload is not canonical JSON."
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
        raise StrictTokenizerBridgeError(
            f"Cannot hash tokenizer artifact file {path.name!r}."
        ) from exc
    return digest.hexdigest()


def _tensor_sha256(value: Any) -> str:
    torch = _require_torch()
    if (
        not isinstance(value, torch.Tensor)
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
    ):
        raise StrictTokenizerBridgeError(
            "Coordinate basis must be a real floating tensor."
        )
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise StrictTokenizerBridgeError("Coordinate basis contains NaN or infinity.")
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "PyTorch is required for strict vocabulary operator transport."
        ) from exc
    return torch


__all__ = [
    "CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY",
    "CompiledTargetTokenizerBridge",
    "DenseVocabularyProjector",
    "PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY",
    "STRICT_TOKENIZER_BRIDGE_FILENAME",
    "STRICT_TOKENIZER_BRIDGE_SCHEMA",
    "StrictTokenizerBridgeArtifact",
    "StrictTokenizerBridgeError",
    "StrictVocabularyOperatorTransportResult",
    "TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY",
    "TOKENIZER_BRIDGE_CONFIG_KEYS",
    "TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY",
    "TokenizerArtifactFile",
    "compile_hf_target_tokenizer_bridge",
    "compile_strict_vocabulary_operators",
    "fingerprint_hf_tokenizer_semantics",
    "load_strict_tokenizer_bridge_artifact",
    "seal_strict_tokenizer_bridge_artifact",
    "strict_tokenizer_bridge_configuration",
]
