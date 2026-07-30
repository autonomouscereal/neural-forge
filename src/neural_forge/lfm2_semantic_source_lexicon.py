"""Package the discrete semantic-source tokenizer inside an LFM2 child.

Semantic-step generation advances in the source vocabulary even when the
deployed response is retokenized through the child's target tokenizer.  This
module copies only the canonical tokenizer files into a scanner-safe child
subdirectory, seals every byte, and fresh-loads the tokenizer locally.  No
model weights or continuous parameter credit are involved.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_sha256
from .hf_lfm2_direct_local import load_lfm2_direct_local_tokenizer
from .hf_tokenizer_bridge import (
    LOCAL_TOKENIZER_PACKAGE_FILENAMES,
    fingerprint_hf_tokenizer_semantics,
)


SCHEMA = "neural-forge-lfm2-semantic-source-lexicon-v1"
SEMANTIC_SOURCE_LEXICON_DIRECTORY = "semantic-source-lexicon"
SEMANTIC_SOURCE_LEXICON_RECEIPT = "lexicon-receipt.json"


class Lfm2SemanticSourceLexiconError(ValueError):
    """A packaged semantic-source tokenizer is incomplete or changed."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SemanticSourceLexiconError(f"{name} must be a lowercase SHA-256.")
    return value


def _ordinary_directory(path: str | os.PathLike[str], name: str) -> Path:
    value = Path(path)
    try:
        resolved = value.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise Lfm2SemanticSourceLexiconError(f"{name} is missing.") from exc
    if value.is_symlink() or not resolved.is_dir():
        raise Lfm2SemanticSourceLexiconError(f"{name} must be a real directory.")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise Lfm2SemanticSourceLexiconError(
                    f"Semantic-source lexicon receipt repeats key {key!r}."
                )
            value[key] = item
        return value

    if not path.is_file() or path.is_symlink():
        raise Lfm2SemanticSourceLexiconError("Lexicon receipt is missing or unsafe.")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except Lfm2SemanticSourceLexiconError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2SemanticSourceLexiconError(
            "Lexicon receipt is not strict UTF-8 JSON."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2SemanticSourceLexiconError("Lexicon receipt must be an object.")
    return value


@dataclass(frozen=True, slots=True)
class AuthenticatedLfm2SemanticSourceLexicon:
    root: Path
    tokenizer: Any
    receipt: Mapping[str, Any]

    @property
    def package_sha256(self) -> str:
        return str(self.receipt["package_sha256"])


def _records(root: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for name in LOCAL_TOKENIZER_PACKAGE_FILENAMES:
        path = root / name
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file() or path.parent != root:
            raise Lfm2SemanticSourceLexiconError(
                f"Semantic-source tokenizer file {name!r} is unsafe."
            )
        if getattr(path.stat(), "st_nlink", 1) != 1:
            raise Lfm2SemanticSourceLexiconError(
                f"Semantic-source tokenizer file {name!r} is hard-linked."
            )
        rows.append(
            {
                "relative_path": name,
                "size_bytes": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        )
    if not rows or not any(value["relative_path"] == "tokenizer.json" for value in rows):
        raise Lfm2SemanticSourceLexiconError(
            "Semantic-source tokenizer must include tokenizer.json."
        )
    return tuple(rows)


def load_packaged_lfm2_semantic_source_lexicon(
    package_or_child_directory: str | os.PathLike[str],
    *,
    expected_tokenizer_semantics_sha256: str,
) -> AuthenticatedLfm2SemanticSourceLexicon:
    """Authenticate and locally load the child-packaged semantic lexicon."""

    supplied = _ordinary_directory(
        package_or_child_directory,
        "semantic-source lexicon path",
    )
    direct_payload = (
        (supplied / SEMANTIC_SOURCE_LEXICON_RECEIPT).is_file()
        and not (supplied / SEMANTIC_SOURCE_LEXICON_RECEIPT).is_symlink()
    )
    root = (
        supplied
        if supplied.name == SEMANTIC_SOURCE_LEXICON_DIRECTORY or direct_payload
        else _ordinary_directory(
            supplied / SEMANTIC_SOURCE_LEXICON_DIRECTORY,
            "packaged semantic-source lexicon",
        )
    )
    receipt = _strict_json(root / SEMANTIC_SOURCE_LEXICON_RECEIPT)
    seal = receipt.pop("package_sha256", None)
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("standalone_local_resource") is not True
        or receipt.get("continuous_parameter_elements") != 0
        or receipt.get("tokenizer_semantics_sha256")
        != _digest(
            expected_tokenizer_semantics_sha256,
            "expected semantic-source tokenizer semantics",
        )
        or seal != canonical_sha256(receipt)
    ):
        raise Lfm2SemanticSourceLexiconError(
            "Semantic-source lexicon receipt or self-seal is invalid."
        )
    observed_records = _records(root)
    if receipt.get("files") != list(observed_records):
        raise Lfm2SemanticSourceLexiconError(
            "Semantic-source lexicon files differ from the sealed inventory."
        )
    allowed = {value["relative_path"] for value in observed_records} | {
        SEMANTIC_SOURCE_LEXICON_RECEIPT
    }
    observed = {
        path.name
        for path in root.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if observed != allowed or any(
        path.is_dir()
        or path.is_symlink()
        or (path.is_file() and getattr(path.stat(), "st_nlink", 1) != 1)
        for path in root.iterdir()
    ):
        raise Lfm2SemanticSourceLexiconError(
            "Semantic-source lexicon contains unsealed additions."
        )
    tokenizer = load_lfm2_direct_local_tokenizer(root)
    if (
        fingerprint_hf_tokenizer_semantics(tokenizer)
        != receipt["tokenizer_semantics_sha256"]
    ):
        raise Lfm2SemanticSourceLexiconError(
            "Fresh semantic-source tokenizer semantics changed."
        )
    sealed = {**receipt, "package_sha256": seal}
    return AuthenticatedLfm2SemanticSourceLexicon(
        root=root,
        tokenizer=tokenizer,
        receipt=MappingProxyType(sealed),
    )


def package_lfm2_semantic_source_lexicon(
    tokenizer_directory: str | os.PathLike[str],
    child_directory: str | os.PathLike[str],
    *,
    expected_tokenizer_semantics_sha256: str,
) -> AuthenticatedLfm2SemanticSourceLexicon:
    """Copy, seal, and fresh-load only canonical tokenizer resources."""

    source = _ordinary_directory(tokenizer_directory, "semantic-source tokenizer")
    child = _ordinary_directory(child_directory, "standalone child directory")
    destination = child / SEMANTIC_SOURCE_LEXICON_DIRECTORY
    if destination.exists() or destination.is_symlink():
        raise Lfm2SemanticSourceLexiconError(
            "Standalone child already contains a semantic-source lexicon."
        )
    expected_semantics = _digest(
        expected_tokenizer_semantics_sha256,
        "expected semantic-source tokenizer semantics",
    )
    source_tokenizer = load_lfm2_direct_local_tokenizer(source)
    if fingerprint_hf_tokenizer_semantics(source_tokenizer) != expected_semantics:
        raise Lfm2SemanticSourceLexiconError(
            "Registered semantic-source tokenizer semantics do not match."
        )
    temporary = Path(
        tempfile.mkdtemp(prefix=".semantic-source-lexicon.", dir=child)
    ).resolve(strict=True)
    try:
        for name in LOCAL_TOKENIZER_PACKAGE_FILENAMES:
            source_path = source / name
            if not source_path.exists():
                continue
            if source_path.is_symlink() or not source_path.is_file():
                raise Lfm2SemanticSourceLexiconError(
                    f"Semantic-source tokenizer file {name!r} is unsafe."
                )
            shutil.copyfile(source_path, temporary / name)
        rows = _records(temporary)
        body = {
            "schema": SCHEMA,
            "status": "complete",
            "tokenizer_semantics_sha256": expected_semantics,
            "files": list(rows),
            "standalone_local_resource": True,
            "continuous_parameter_elements": 0,
            "parameter_count_credit": False,
        }
        receipt = {**body, "package_sha256": canonical_sha256(body)}
        (temporary / SEMANTIC_SOURCE_LEXICON_RECEIPT).write_text(
            json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        authenticated = load_packaged_lfm2_semantic_source_lexicon(
            temporary,
            expected_tokenizer_semantics_sha256=expected_semantics,
        )
        os.rename(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    final = load_packaged_lfm2_semantic_source_lexicon(
        destination,
        expected_tokenizer_semantics_sha256=expected_semantics,
    )
    if final.package_sha256 != authenticated.package_sha256:
        raise Lfm2SemanticSourceLexiconError(
            "Semantic-source lexicon changed during atomic publication."
        )
    return final


__all__ = [
    "SCHEMA",
    "SEMANTIC_SOURCE_LEXICON_DIRECTORY",
    "SEMANTIC_SOURCE_LEXICON_RECEIPT",
    "AuthenticatedLfm2SemanticSourceLexicon",
    "Lfm2SemanticSourceLexiconError",
    "load_packaged_lfm2_semantic_source_lexicon",
    "package_lfm2_semantic_source_lexicon",
]
