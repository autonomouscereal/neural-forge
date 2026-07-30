"""Authenticated local tokenizer control-token protection.

This module deliberately reads tokenizer artifacts as local files.  It does
not instantiate a Transformers tokenizer, consult ``all_special_ids``, or
perform any network access.  The compiler protects structural token IDs before
semantic carrier allocation can consider reclaiming vocabulary rows.

Reserved placeholders are treated differently from active controls.  They are
protected by default, but an independently authenticated policy may release an
explicit list of inactive reserved IDs.  Active template, modality, BOS, EOS,
and PAD controls can never be released by that policy.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256


TOKENIZER_CONTROL_CONTRACT_SCHEMA = "neural-forge-tokenizer-control-contract-v1"
TOKENIZER_CONTROL_SOURCE_INVENTORY_SCHEMA = (
    "neural-forge-tokenizer-control-source-inventory-v1"
)
TOKENIZER_CONTROL_MANIFEST_SCHEMA = "neural-forge-tokenizer-control-manifest-v1"
TOKENIZER_RESERVED_RECLAIM_POLICY_SCHEMA = (
    "neural-forge-tokenizer-reserved-reclaim-policy-v1"
)

_MANDATORY_SOURCE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RESERVED_NAME = re.compile(
    r"(?:^|[^a-z0-9])(?:reserved|unused|placeholder)(?:\d+)?"
    r"(?:[^a-z0-9]|$)",
    re.IGNORECASE,
)
_EXPECTED_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "manifest_kind",
        "active_control_tokens",
        "reserved_placeholder_tokens",
    }
)
_EXPECTED_RECLAIM_POLICY_KEYS = frozenset(
    {
        "schema",
        "source_inventory_sha256",
        "release_reserved_token_ids",
        "reason",
    }
)
_ADDED_TOKEN_REFERENCE_KEYS = frozenset(
    {
        "id",
        "content",
        "special",
        "single_word",
        "lstrip",
        "rstrip",
        "normalized",
        "__type",
    }
)


class TokenizerControlContractError(ValueError):
    """A tokenizer source, control classification, or policy is unsafe."""


@dataclass(frozen=True)
class LocalTokenizerControlContract:
    """Immutable values and a sealed receipt for one local tokenizer package."""

    _protected_token_id_values: tuple[int, ...]
    _default_protected_token_id_values: tuple[int, ...]
    _mandatory_active_token_id_values: tuple[int, ...]
    _reserved_placeholder_token_id_values: tuple[int, ...]
    _reclaimed_token_id_values: tuple[int, ...]
    _receipt_json: str

    @property
    def protected_ids(self) -> torch.Tensor:
        """Return a fresh, mutable-safe CPU int64 tensor of protected IDs."""

        return torch.tensor(
            self._protected_token_id_values,
            dtype=torch.int64,
            device="cpu",
        )

    @property
    def protected_token_ids(self) -> torch.Tensor:
        """Alias spelling suitable for allocator call sites."""

        return self.protected_ids

    @property
    def protected_id_values(self) -> tuple[int, ...]:
        return self._protected_token_id_values

    @property
    def default_protected_id_values(self) -> tuple[int, ...]:
        return self._default_protected_token_id_values

    @property
    def mandatory_active_id_values(self) -> tuple[int, ...]:
        return self._mandatory_active_token_id_values

    @property
    def reserved_placeholder_id_values(self) -> tuple[int, ...]:
        return self._reserved_placeholder_token_id_values

    @property
    def reclaimed_id_values(self) -> tuple[int, ...]:
        return self._reclaimed_token_id_values

    @property
    def receipt(self) -> Mapping[str, Any]:
        """Return an isolated copy of the authenticated receipt."""

        payload = json.loads(self._receipt_json)
        return MappingProxyType(payload)

    @property
    def receipt_sha256(self) -> str:
        return str(json.loads(self._receipt_json)["receipt_sha256"])

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the receipt."""

        return json.loads(self._receipt_json)


@dataclass
class _TokenFacts:
    token_id: int
    content: str
    classifications: set[str]
    evidence: set[str]


def inventory_local_tokenizer_control_sources(
    artifact_dir: str | Path,
    *,
    manifest_files: Sequence[str | Path] = (),
) -> dict[str, str]:
    """Hash the exact local source set expected by the contract compiler.

    This helper is intended for an artifact producer.  A consumer should get
    these digests through an independently authenticated artifact manifest and
    pass them to :func:`compile_local_tokenizer_control_contract`.
    """

    root = _ordinary_directory(artifact_dir)
    names = _source_names(manifest_files)
    return {
        name: hashlib.sha256(_read_stable_local_file(root, name)).hexdigest()
        for name in names
    }


def compile_local_tokenizer_control_contract(
    artifact_dir: str | Path,
    *,
    expected_file_sha256: Mapping[str, str],
    manifest_files: Sequence[str | Path] = (),
    reclaim_policy_file: str | Path | None = None,
    expected_reclaim_policy_sha256: str | None = None,
) -> LocalTokenizerControlContract:
    """Compile a fail-closed control protection contract from local artifacts.

    ``expected_file_sha256`` must contain exactly the three mandatory source
    filenames plus every explicitly supplied manifest filename.  Reclaiming a
    reserved row requires a separate policy file, its externally supplied
    SHA-256 digest, and a binding to this source inventory.
    """

    root = _ordinary_directory(artifact_dir)
    source_names = _source_names(manifest_files)
    expected = _validated_expected_hashes(expected_file_sha256, source_names)
    source_bytes: dict[str, bytes] = {}
    source_records: list[dict[str, Any]] = []
    for name in source_names:
        payload = _read_stable_local_file(root, name)
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected[name]:
            raise TokenizerControlContractError(
                f"Authenticated source hash mismatch for {name!r}."
            )
        source_bytes[name] = payload
        source_records.append(
            {"path": name, "sha256": actual, "byte_length": len(payload)}
        )

    source_inventory = {
        "schema": TOKENIZER_CONTROL_SOURCE_INVENTORY_SCHEMA,
        "files": source_records,
    }
    source_inventory_sha256 = canonical_sha256(source_inventory)

    tokenizer = _strict_json_bytes(source_bytes["tokenizer.json"], "tokenizer.json")
    config = _strict_json_bytes(
        source_bytes["tokenizer_config.json"],
        "tokenizer_config.json",
    )
    if not isinstance(tokenizer, dict) or not isinstance(config, dict):
        raise TokenizerControlContractError(
            "Tokenizer JSON and tokenizer config roots must be objects."
        )

    content_to_id, id_to_content, added_records = _token_identity_inventory(
        tokenizer,
        config,
    )
    facts: dict[int, _TokenFacts] = {}

    def classify(
        token_id: int,
        classification: str,
        evidence: str,
    ) -> None:
        content = id_to_content.get(token_id)
        if content is None:
            raise TokenizerControlContractError(
                f"Control token ID {token_id} is absent from the tokenizer inventory."
            )
        item = facts.setdefault(
            token_id,
            _TokenFacts(token_id, content, set(), set()),
        )
        item.classifications.add(classification)
        item.evidence.add(evidence)

    for token_id, content, special, source in added_records:
        if special:
            classify(token_id, "added-token-special", source)
        if _is_explicit_reserved_name(content):
            classify(token_id, "reserved-placeholder-name", source)

    for token_id, content in sorted(id_to_content.items()):
        if _is_explicit_reserved_name(content):
            classify(
                token_id,
                "reserved-placeholder-name",
                "tokenizer-token-content",
            )

    _classify_declared_controls(config, content_to_id, id_to_content, classify)
    template_sources = _template_sources(
        source_bytes["chat_template.jinja"],
        config,
    )
    added_ids = {record[0] for record in added_records}
    for label, template in template_sources:
        for token_id, content in sorted(id_to_content.items()):
            if (
                content
                and content in template
                and (token_id in added_ids or _looks_like_control_marker(content))
            ):
                classify(token_id, "chat-template-literal-control", label)

    for manifest_name in source_names[len(_MANDATORY_SOURCE_FILES) :]:
        manifest = _strict_json_bytes(source_bytes[manifest_name], manifest_name)
        _classify_manifest(
            manifest,
            manifest_name,
            content_to_id,
            id_to_content,
            classify,
        )

    default_ids = tuple(sorted(facts))
    mandatory_ids = tuple(
        token_id for token_id in default_ids if _is_mandatory_active(facts[token_id])
    )
    reserved_ids = tuple(
        token_id for token_id in default_ids if _is_reserved(facts[token_id])
    )
    mandatory_id_set = frozenset(mandatory_ids)
    reclaimable_ids = tuple(
        token_id for token_id in reserved_ids if token_id not in mandatory_id_set
    )
    policy_receipt, reclaimed_ids = _load_reclaim_policy(
        root=root,
        policy_file=reclaim_policy_file,
        expected_sha256=expected_reclaim_policy_sha256,
        source_inventory_sha256=source_inventory_sha256,
        reclaimable_ids=frozenset(reclaimable_ids),
        default_protected_ids=frozenset(default_ids),
    )
    reclaimed_id_set = frozenset(reclaimed_ids)
    protected_ids = tuple(
        token_id for token_id in default_ids if token_id not in reclaimed_id_set
    )

    records: list[dict[str, Any]] = []
    for token_id in default_ids:
        item = facts[token_id]
        reserved = _is_reserved(item)
        mandatory = _is_mandatory_active(item)
        reclaimed = token_id in reclaimed_id_set
        records.append(
            {
                "token_id": token_id,
                "content": item.content,
                "classifications": sorted(item.classifications),
                "evidence": sorted(item.evidence),
                "mandatory_active_structural_control": mandatory,
                "reserved_placeholder": reserved,
                "reclaim_eligible": reserved and not mandatory,
                "reclaimed": reclaimed,
                "protected": not reclaimed,
            }
        )

    body = {
        "schema": TOKENIZER_CONTROL_CONTRACT_SCHEMA,
        "source_inventory": source_inventory,
        "source_inventory_sha256": source_inventory_sha256,
        "classification_algorithm": "local-tokenizer-control-classifier-v1",
        "known_token_count": len(id_to_content),
        "token_id_upper_bound_exclusive": max(id_to_content, default=-1) + 1,
        "default_protected_token_ids": list(default_ids),
        "mandatory_active_token_ids": list(mandatory_ids),
        "reserved_placeholder_token_ids": list(reserved_ids),
        "reclaim_eligible_token_ids": list(reclaimable_ids),
        "reclaim_policy": policy_receipt,
        "reclaimed_token_ids": list(reclaimed_ids),
        "protected_token_ids": list(protected_ids),
        "token_records": records,
        "tokenizer_all_special_ids_consulted": False,
        "network_access_performed": False,
        "hugging_face_loader_used": False,
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    receipt_json = json.dumps(
        receipt,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return LocalTokenizerControlContract(
        _protected_token_id_values=protected_ids,
        _default_protected_token_id_values=default_ids,
        _mandatory_active_token_id_values=mandatory_ids,
        _reserved_placeholder_token_id_values=reserved_ids,
        _reclaimed_token_id_values=reclaimed_ids,
        _receipt_json=receipt_json,
    )


def _ordinary_directory(value: str | Path) -> Path:
    path = Path(value)
    if not path.exists() or not path.is_dir() or path.is_symlink():
        raise TokenizerControlContractError(
            "Tokenizer artifact directory must be an existing ordinary directory."
        )
    return path.resolve()


def _source_names(manifest_files: Sequence[str | Path]) -> tuple[str, ...]:
    names = list(_MANDATORY_SOURCE_FILES)
    seen = set(names)
    normalized_manifests: list[str] = []
    for value in manifest_files:
        path = Path(value)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise TokenizerControlContractError(
                "Manifest filenames must be relative paths inside the artifact."
            )
        name = path.as_posix()
        if name in seen:
            raise TokenizerControlContractError(
                f"Duplicate tokenizer control source filename {name!r}."
            )
        seen.add(name)
        normalized_manifests.append(name)
    names.extend(sorted(normalized_manifests))
    return tuple(names)


def _validated_expected_hashes(
    value: Mapping[str, str],
    source_names: tuple[str, ...],
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TokenizerControlContractError(
            "expected_file_sha256 must be a filename-to-SHA mapping."
        )
    expected = dict(value)
    if set(expected) != set(source_names):
        raise TokenizerControlContractError(
            "Expected source hashes must name exactly every consumed source file."
        )
    for name, digest in expected.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise TokenizerControlContractError("Source hashes must be strings.")
        if not _SHA256.fullmatch(digest):
            raise TokenizerControlContractError(
                f"Expected SHA-256 for {name!r} is not lowercase hexadecimal."
            )
    return expected


def _read_stable_local_file(root: Path, relative_name: str) -> bytes:
    candidate = root / Path(relative_name)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise TokenizerControlContractError(
            f"Local tokenizer source {relative_name!r} is missing or escapes its root."
        ) from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise TokenizerControlContractError(
            f"Local tokenizer source {relative_name!r} must be an ordinary file."
        )
    before = resolved.stat()
    payload = resolved.read_bytes()
    after = resolved.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != after.st_size:
        raise TokenizerControlContractError(
            f"Local tokenizer source {relative_name!r} changed while being read."
        )
    return payload


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TokenizerControlContractError(
                f"Authenticated JSON contains duplicate key {key!r}."
            )
        result[key] = value
    return result


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    try:
        text = payload.decode("utf-8")
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except TokenizerControlContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TokenizerControlContractError(
            f"Authenticated source {label!r} is not strict UTF-8 JSON."
        ) from exc


def _nonnegative_token_id(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TokenizerControlContractError(
            f"{label} must be a non-negative integer token ID."
        )
    return value


def _token_content(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TokenizerControlContractError(
            f"{label} must be a non-empty token string."
        )
    return value


def _token_identity_inventory(
    tokenizer: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, int], dict[int, str], list[tuple[int, str, bool, str]]]:
    content_to_id: dict[str, int] = {}
    id_to_content: dict[int, str] = {}
    added: list[tuple[int, str, bool, str]] = []

    def add_identity(token_id_value: Any, content_value: Any, source: str) -> None:
        token_id = _nonnegative_token_id(token_id_value, f"{source} ID")
        content = _token_content(content_value, f"{source} content")
        prior_id = content_to_id.get(content)
        prior_content = id_to_content.get(token_id)
        if (prior_id is not None and prior_id != token_id) or (
            prior_content is not None and prior_content != content
        ):
            raise TokenizerControlContractError(
                f"Tokenizer token identity conflicts at {source}."
            )
        content_to_id[content] = token_id
        id_to_content[token_id] = content

    model = tokenizer.get("model")
    if not isinstance(model, dict):
        raise TokenizerControlContractError("tokenizer.json lacks a model object.")
    vocab = model.get("vocab")
    if isinstance(vocab, dict):
        for content, token_id in vocab.items():
            add_identity(token_id, content, "tokenizer.json model.vocab")
    elif isinstance(vocab, list):
        for token_id, entry in enumerate(vocab):
            content = entry[0] if isinstance(entry, list) and entry else entry
            add_identity(token_id, content, "tokenizer.json model.vocab")
    else:
        raise TokenizerControlContractError(
            "tokenizer.json model.vocab must be an object or list."
        )

    raw_added = tokenizer.get("added_tokens", [])
    if not isinstance(raw_added, list):
        raise TokenizerControlContractError(
            "tokenizer.json added_tokens must be a list."
        )
    for index, record in enumerate(raw_added):
        if not isinstance(record, dict):
            raise TokenizerControlContractError(
                "Every tokenizer.json AddedToken must be an object."
            )
        source = f"tokenizer.json added_tokens[{index}]"
        token_id = _nonnegative_token_id(record.get("id"), f"{source} ID")
        content = _token_content(record.get("content"), f"{source} content")
        special = record.get("special", False)
        if not isinstance(special, bool):
            raise TokenizerControlContractError(
                f"{source} special flag must be boolean."
            )
        add_identity(token_id, content, source)
        added.append((token_id, content, special, source))

    decoder = config.get("added_tokens_decoder", {})
    if not isinstance(decoder, dict):
        raise TokenizerControlContractError(
            "tokenizer_config.json added_tokens_decoder must be an object."
        )
    for raw_id, record in decoder.items():
        try:
            token_id_value = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise TokenizerControlContractError(
                "added_tokens_decoder keys must be decimal token IDs."
            ) from exc
        if str(token_id_value) != str(raw_id):
            raise TokenizerControlContractError(
                "added_tokens_decoder keys must use canonical decimal token IDs."
            )
        if not isinstance(record, dict):
            raise TokenizerControlContractError(
                "Every configured AddedToken decoder entry must be an object."
            )
        source = f"tokenizer_config.json added_tokens_decoder[{raw_id}]"
        content = _token_content(record.get("content"), f"{source} content")
        special = record.get("special", False)
        if not isinstance(special, bool):
            raise TokenizerControlContractError(
                f"{source} special flag must be boolean."
            )
        add_identity(token_id_value, content, source)
        added.append((token_id_value, content, special, source))

    return content_to_id, id_to_content, added


def _resolve_token_reference(
    value: Any,
    content_to_id: Mapping[str, int],
    id_to_content: Mapping[int, str],
    label: str,
) -> int:
    token_id: int | None = None
    content: str | None = None
    if isinstance(value, bool):
        raise TokenizerControlContractError(f"{label} is not a valid token reference.")
    if isinstance(value, int):
        token_id = _nonnegative_token_id(value, label)
    elif isinstance(value, str):
        content = _token_content(value, label)
    elif isinstance(value, dict):
        if not value or not set(value).issubset(_ADDED_TOKEN_REFERENCE_KEYS):
            raise TokenizerControlContractError(
                f"{label} token object contains unknown AddedToken fields."
            )
        for flag in (
            "special",
            "single_word",
            "lstrip",
            "rstrip",
            "normalized",
        ):
            if flag in value and not isinstance(value[flag], bool):
                raise TokenizerControlContractError(
                    f"{label} AddedToken {flag} flag must be boolean."
                )
        if "__type" in value and value["__type"] != "AddedToken":
            raise TokenizerControlContractError(
                f"{label} __type must identify an AddedToken."
            )
        if "id" in value:
            token_id = _nonnegative_token_id(value["id"], f"{label} ID")
        if "content" in value:
            content = _token_content(value["content"], f"{label} content")
    else:
        raise TokenizerControlContractError(f"{label} is not a valid token reference.")
    if content is not None:
        resolved_id = content_to_id.get(content)
        if resolved_id is None:
            raise TokenizerControlContractError(
                f"{label} content is absent from the authenticated tokenizer."
            )
        if token_id is not None and token_id != resolved_id:
            raise TokenizerControlContractError(
                f"{label} token ID and content disagree."
            )
        token_id = resolved_id
    if token_id is None or token_id not in id_to_content:
        raise TokenizerControlContractError(
            f"{label} ID is absent from the authenticated tokenizer."
        )
    return token_id


def _references(value: Any, label: str) -> list[Any]:
    if isinstance(value, list):
        if not value:
            raise TokenizerControlContractError(f"{label} must not be an empty list.")
        return value
    return [value]


def _classify_declared_controls(
    config: dict[str, Any],
    content_to_id: Mapping[str, int],
    id_to_content: Mapping[int, str],
    classify: Any,
) -> None:
    for role in ("bos", "eos", "pad"):
        token_value = config.get(f"{role}_token")
        id_value = config.get(f"{role}_token_id")
        if token_value is None and id_value is None:
            continue
        token_refs = (
            []
            if token_value is None
            else _references(
                token_value,
                f"declared {role.upper()} token",
            )
        )
        resolved = {
            _resolve_token_reference(
                ref,
                content_to_id,
                id_to_content,
                f"declared {role.upper()} token",
            )
            for ref in token_refs
        }
        if id_value is not None:
            id_refs = _references(id_value, f"declared {role.upper()} token ID")
            id_resolved = {
                _resolve_token_reference(
                    ref,
                    content_to_id,
                    id_to_content,
                    f"declared {role.upper()} token ID",
                )
                for ref in id_refs
            }
            if resolved and resolved != id_resolved:
                raise TokenizerControlContractError(
                    f"Declared {role.upper()} token content and ID disagree."
                )
            resolved.update(id_resolved)
        for token_id in resolved:
            classify(
                token_id,
                f"declared-{role}",
                f"tokenizer_config.json {role}_token",
            )

    additional = config.get("additional_special_tokens", [])
    if additional is None:
        additional = []
    if not isinstance(additional, list):
        raise TokenizerControlContractError(
            "additional_special_tokens must be a list when present."
        )
    for index, ref in enumerate(additional):
        token_id = _resolve_token_reference(
            ref,
            content_to_id,
            id_to_content,
            f"additional_special_tokens[{index}]",
        )
        classify(
            token_id,
            "configured-additional-special-token",
            f"tokenizer_config.json additional_special_tokens[{index}]",
        )


def _template_sources(
    template_file_bytes: bytes,
    config: dict[str, Any],
) -> list[tuple[str, str]]:
    try:
        file_template = template_file_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TokenizerControlContractError(
            "chat_template.jinja is not strict UTF-8 text."
        ) from exc
    sources = [("chat_template.jinja", file_template)]
    inline = config.get("chat_template")
    if inline is None:
        return sources
    if isinstance(inline, str):
        sources.append(("tokenizer_config.json chat_template", inline))
        return sources
    if isinstance(inline, list):
        for index, item in enumerate(inline):
            if (
                not isinstance(item, dict)
                or set(item) != {"name", "template"}
                or not isinstance(item["name"], str)
                or not item["name"]
                or not isinstance(item["template"], str)
            ):
                raise TokenizerControlContractError(
                    "Named chat templates must contain exactly name and template strings."
                )
            sources.append(
                (
                    f"tokenizer_config.json chat_template[{index}]/{item['name']}",
                    item["template"],
                )
            )
        return sources
    raise TokenizerControlContractError(
        "tokenizer_config.json chat_template must be text or a named-template list."
    )


def _looks_like_control_marker(content: str) -> bool:
    if len(content) < 3:
        return False
    pairs = (("<", ">"), ("[", "]"), ("{", "}"))
    return any(
        content.startswith(left) and content.endswith(right) for left, right in pairs
    )


def _is_explicit_reserved_name(content: str) -> bool:
    normalized = content.replace("|", "_").replace("<", "_").replace(">", "_")
    normalized = normalized.replace("[", "_").replace("]", "_")
    return _RESERVED_NAME.search(normalized) is not None


def _classify_manifest(
    manifest: Any,
    manifest_name: str,
    content_to_id: Mapping[str, int],
    id_to_content: Mapping[int, str],
    classify: Any,
) -> None:
    if not isinstance(manifest, dict) or set(manifest) != _EXPECTED_MANIFEST_KEYS:
        raise TokenizerControlContractError(
            f"Control manifest {manifest_name!r} has incomplete or unknown fields."
        )
    if manifest.get("schema") != TOKENIZER_CONTROL_MANIFEST_SCHEMA:
        raise TokenizerControlContractError(
            f"Control manifest {manifest_name!r} has an unsupported schema."
        )
    if manifest.get("manifest_kind") not in {"modality", "template", "combined"}:
        raise TokenizerControlContractError(
            f"Control manifest {manifest_name!r} has an invalid manifest_kind."
        )
    for key, classification in (
        ("active_control_tokens", "manifest-active-control"),
        ("reserved_placeholder_tokens", "manifest-reserved-placeholder"),
    ):
        refs = manifest.get(key)
        if not isinstance(refs, list):
            raise TokenizerControlContractError(
                f"Control manifest {manifest_name!r} {key} must be a list."
            )
        seen: set[int] = set()
        for index, ref in enumerate(refs):
            token_id = _resolve_token_reference(
                ref,
                content_to_id,
                id_to_content,
                f"{manifest_name} {key}[{index}]",
            )
            if token_id in seen:
                raise TokenizerControlContractError(
                    f"Control manifest {manifest_name!r} repeats token ID {token_id}."
                )
            seen.add(token_id)
            classify(token_id, classification, f"{manifest_name} {key}[{index}]")


def _is_reserved(facts: _TokenFacts) -> bool:
    return bool(
        facts.classifications
        & {"reserved-placeholder-name", "manifest-reserved-placeholder"}
    )


def _is_mandatory_active(facts: _TokenFacts) -> bool:
    explicit_active = any(
        classification.startswith("declared-")
        or classification
        in {"chat-template-literal-control", "manifest-active-control"}
        for classification in facts.classifications
    )
    if explicit_active:
        return True
    protected_special = bool(
        facts.classifications
        & {"added-token-special", "configured-additional-special-token"}
    )
    return protected_special and not _is_reserved(facts)


def _load_reclaim_policy(
    *,
    root: Path,
    policy_file: str | Path | None,
    expected_sha256: str | None,
    source_inventory_sha256: str,
    reclaimable_ids: frozenset[int],
    default_protected_ids: frozenset[int],
) -> tuple[dict[str, Any], tuple[int, ...]]:
    if policy_file is None:
        if expected_sha256 is not None:
            raise TokenizerControlContractError(
                "A reclaim-policy SHA cannot be supplied without a policy file."
            )
        body = {
            "mode": "protect-all",
            "policy_file": None,
            "policy_file_sha256": None,
            "policy_body_sha256": None,
            "source_inventory_sha256": source_inventory_sha256,
            "release_reserved_token_ids": [],
            "reason": "No authenticated reserved-token reclaim policy supplied.",
        }
        return {**body, "policy_receipt_sha256": canonical_sha256(body)}, ()

    if expected_sha256 is None or not isinstance(expected_sha256, str):
        raise TokenizerControlContractError(
            "A reclaim policy requires an externally supplied SHA-256 digest."
        )
    if not _SHA256.fullmatch(expected_sha256):
        raise TokenizerControlContractError(
            "Expected reclaim-policy SHA-256 is not lowercase hexadecimal."
        )
    policy_path = Path(policy_file)
    if policy_path.is_absolute() or not policy_path.parts or ".." in policy_path.parts:
        raise TokenizerControlContractError(
            "Reclaim policy must be a relative ordinary file inside the artifact."
        )
    policy_name = policy_path.as_posix()
    payload = _read_stable_local_file(root, policy_name)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise TokenizerControlContractError(
            "Authenticated reserved-token reclaim-policy hash mismatch."
        )
    policy = _strict_json_bytes(payload, policy_name)
    if not isinstance(policy, dict) or set(policy) != _EXPECTED_RECLAIM_POLICY_KEYS:
        raise TokenizerControlContractError(
            "Reserved-token reclaim policy has incomplete or unknown fields."
        )
    if policy.get("schema") != TOKENIZER_RESERVED_RECLAIM_POLICY_SCHEMA:
        raise TokenizerControlContractError(
            "Reserved-token reclaim policy schema is invalid."
        )
    if policy.get("source_inventory_sha256") != source_inventory_sha256:
        raise TokenizerControlContractError(
            "Reserved-token reclaim policy is not bound to this source inventory."
        )
    reason = policy.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise TokenizerControlContractError(
            "Reserved-token reclaim policy requires a non-empty reason."
        )
    released = policy.get("release_reserved_token_ids")
    if not isinstance(released, list):
        raise TokenizerControlContractError(
            "release_reserved_token_ids must be an explicit list of IDs."
        )
    normalized: list[int] = []
    seen: set[int] = set()
    for index, value in enumerate(released):
        token_id = _nonnegative_token_id(
            value,
            f"release_reserved_token_ids[{index}]",
        )
        if token_id in seen:
            raise TokenizerControlContractError(
                "Reserved-token reclaim policy repeats a token ID."
            )
        seen.add(token_id)
        if token_id not in default_protected_ids:
            raise TokenizerControlContractError(
                f"Reclaim policy token ID {token_id} is not protected by this contract."
            )
        if token_id not in reclaimable_ids:
            raise TokenizerControlContractError(
                f"Reclaim policy token ID {token_id} is an active or non-reserved control."
            )
        normalized.append(token_id)
    normalized_tuple = tuple(sorted(normalized))
    body = {
        "mode": "authenticated-explicit-reserved-id-release",
        "policy_file": policy_name,
        "policy_file_sha256": actual_sha256,
        "policy_body_sha256": canonical_sha256(policy),
        "source_inventory_sha256": source_inventory_sha256,
        "release_reserved_token_ids": list(normalized_tuple),
        "reason": reason.strip(),
    }
    return (
        {**body, "policy_receipt_sha256": canonical_sha256(body)},
        normalized_tuple,
    )
