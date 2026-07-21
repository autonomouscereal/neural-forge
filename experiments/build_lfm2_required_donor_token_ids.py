#!/usr/bin/env python3
"""Build one authenticated donor-token union from sealed LFM2 trace caches.

Only the prompt and forced-decode token tensors are materialized.  The cache
session authenticates every sealed trace file once, and the resulting manifest
binds the cache keys, parent identity, session evidence, and every bounded
selection used to derive the union.  No model is loaded or executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    from experiments.compile_lfm2_target_tokenizer_bridge import (
        REQUIRED_DONOR_TOKEN_IDS_SCHEMA,
    )
except ImportError:  # Direct execution from the experiments directory.
    from compile_lfm2_target_tokenizer_bridge import (
        REQUIRED_DONOR_TOKEN_IDS_SCHEMA,
    )


SELECTED_TENSOR_NAMES = (
    "input.forced_decode_ids",
    "input.input_ids",
)


class Lfm2RequiredDonorTokenIdsError(ValueError):
    """A cache union cannot be authenticated or represented safely."""


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
        raise Lfm2RequiredDonorTokenIdsError(
            "Required donor token ID evidence is not canonical JSON."
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
        raise Lfm2RequiredDonorTokenIdsError(
            f"Cannot hash required donor token ID manifest {path}."
        ) from exc
    return digest.hexdigest()


def _sha256_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2RequiredDonorTokenIdsError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _normalized_cache_roots(
    values: Sequence[str | os.PathLike[str]],
) -> tuple[Path, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise Lfm2RequiredDonorTokenIdsError(
            "trace_cache_roots must be a nonempty sequence."
        )
    roots: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        try:
            root = Path(value).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, TypeError) as exc:
            raise Lfm2RequiredDonorTokenIdsError(
                f"Trace cache root is unavailable: {value!r}."
            ) from exc
        if root.is_symlink() or not root.is_dir():
            raise Lfm2RequiredDonorTokenIdsError(
                f"Trace cache root must be a real local directory: {root}."
            )
        if root in seen:
            raise Lfm2RequiredDonorTokenIdsError(
                f"Trace cache root is duplicated: {root}."
            )
        seen.add(root)
        roots.append(root)
    if not roots:
        raise Lfm2RequiredDonorTokenIdsError(
            "At least one trace cache root is required."
        )
    return tuple(sorted(roots, key=lambda path: str(path).casefold()))


def _open_trace_cache(path: Path) -> Any:
    from neural_forge.hf_lfm2_circuit_trace_cache import (
        open_lfm2_circuit_trace_cache,
    )

    return open_lfm2_circuit_trace_cache(path, verify=False)


def _tensor_token_ids(tensor: Any, name: str) -> tuple[int, ...]:
    try:
        raw_values = tensor.detach().cpu().reshape(-1).tolist()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise Lfm2RequiredDonorTokenIdsError(
            f"Selected tensor {name!r} is not a readable integer tensor."
        ) from exc
    if not isinstance(raw_values, list):
        raise Lfm2RequiredDonorTokenIdsError(
            f"Selected tensor {name!r} did not flatten to a list."
        )
    values: list[int] = []
    for value in raw_values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Lfm2RequiredDonorTokenIdsError(
                f"Selected tensor {name!r} contains an invalid token ID."
            )
        values.append(value)
    return tuple(values)


def _extract_one_cache(root: Path) -> tuple[set[int], dict[str, Any]]:
    cache = _open_trace_cache(root)
    if Path(cache.path).resolve(strict=True) != root:
        raise Lfm2RequiredDonorTokenIdsError(
            "Opened trace cache path differs from the requested root."
        )
    manifest = cache.manifest
    manifest_sha256 = _sha256_digest(
        manifest.get("manifest_sha256"),
        "cache manifest_sha256",
    )
    parent = manifest.get("parent")
    corpus = manifest.get("corpus")
    if not isinstance(parent, Mapping) or not isinstance(corpus, Mapping):
        raise Lfm2RequiredDonorTokenIdsError(
            "Trace cache manifest lacks parent or corpus identity."
        )
    parent_identity = _sha256_digest(
        parent.get("identity_sha256"),
        "cache parent identity_sha256",
    )
    corpus_sha256 = _sha256_digest(
        corpus.get("corpus_sha256"),
        "cache corpus_sha256",
    )
    cache_key = _sha256_digest(cache.cache_key, "cache key")
    if cache.parent_fingerprint != parent_identity:
        raise Lfm2RequiredDonorTokenIdsError(
            "Trace cache parent property differs from its manifest."
        )
    if cache.corpus_fingerprint != corpus_sha256:
        raise Lfm2RequiredDonorTokenIdsError(
            "Trace cache corpus property differs from its manifest."
        )

    cache_token_ids: set[int] = set()
    selected_batches: list[dict[str, Any]] = []
    with cache.selective_authenticated_session() as session:
        if session.cache_key != cache_key or session.batch_count != cache.batch_count:
            raise Lfm2RequiredDonorTokenIdsError(
                "Selective session differs from its opened trace cache."
            )
        for batch_index in range(session.batch_count):
            payload = session.load_selected_tensors(
                batch_index,
                SELECTED_TENSOR_NAMES,
            )
            try:
                if payload.ordered_tensor_names != SELECTED_TENSOR_NAMES:
                    raise Lfm2RequiredDonorTokenIdsError(
                        "Selective cache payload returned an unexpected tensor set."
                    )
                for name in SELECTED_TENSOR_NAMES:
                    cache_token_ids.update(
                        _tensor_token_ids(payload.tensors[name], name)
                    )
                selected_batches.append(
                    {
                        "batch_index": payload.batch_index,
                        "input_sha256": _sha256_digest(
                            payload.input_sha256,
                            "selected input_sha256",
                        ),
                        "ordered_tensor_names": list(
                            payload.ordered_tensor_names
                        ),
                        "logical_bytes": payload.logical_bytes,
                        "selection_sha256": _sha256_digest(
                            payload.selection_sha256,
                            "selection_sha256",
                        ),
                    }
                )
            finally:
                session.release_selected_tensors(payload)
        session.verify()
        session_evidence = session.as_dict()
        session_sha256 = _sha256_digest(
            session_evidence.get("session_sha256"),
            "selective session_sha256",
        )
        if session_sha256 != session.evidence_sha256:
            raise Lfm2RequiredDonorTokenIdsError(
                "Selective session evidence property differs from its seal."
            )
        if (
            session_evidence.get("cache_key") != cache_key
            or session_evidence.get("manifest_sha256") != manifest_sha256
            or session_evidence.get("batch_count") != cache.batch_count
        ):
            raise Lfm2RequiredDonorTokenIdsError(
                "Selective session evidence is not bound to the trace cache."
            )

    ordered_ids = sorted(cache_token_ids)
    source = {
        "trace_cache_root": str(root),
        "cache_key": cache_key,
        "manifest_sha256": manifest_sha256,
        "completion_sha256": _sha256_digest(
            session_evidence.get("completion_sha256"),
            "cache completion_sha256",
        ),
        "parent_identity_sha256": parent_identity,
        "corpus_sha256": corpus_sha256,
        "batch_count": cache.batch_count,
        "selective_session_sha256": session_sha256,
        "selected_batches": selected_batches,
        "selected_logical_bytes": sum(
            int(batch["logical_bytes"]) for batch in selected_batches
        ),
        "unique_donor_token_ids": ordered_ids,
        "unique_donor_token_count": len(ordered_ids),
        "unique_donor_token_ids_sha256": _canonical_sha256(ordered_ids),
    }
    return cache_token_ids, source


def build_required_donor_token_ids_manifest(
    trace_cache_roots: Sequence[str | os.PathLike[str]],
) -> dict[str, Any]:
    """Authenticate cache roots and return their canonical donor-ID union."""

    roots = _normalized_cache_roots(trace_cache_roots)
    union: set[int] = set()
    sources: list[dict[str, Any]] = []
    parent_identities: set[str] = set()
    for root in roots:
        token_ids, source = _extract_one_cache(root)
        union.update(token_ids)
        sources.append(source)
        parent_identities.add(str(source["parent_identity_sha256"]))
    if len(parent_identities) != 1:
        raise Lfm2RequiredDonorTokenIdsError(
            "Trace caches do not share one authenticated parent identity."
        )
    ordered_ids = sorted(union)
    body = {
        "schema": REQUIRED_DONOR_TOKEN_IDS_SCHEMA,
        "source_cache_count": len(sources),
        "source_caches": sources,
        "parent_identity_sha256": next(iter(parent_identities)),
        "selected_tensor_names": list(SELECTED_TENSOR_NAMES),
        "required_donor_token_ids": ordered_ids,
        "required_donor_token_count": len(ordered_ids),
        "required_donor_token_ids_sha256": _canonical_sha256(ordered_ids),
        "extraction_contract": {
            "sealed_trace_caches_only": True,
            "selective_authenticated_sessions": True,
            "unselected_tensor_payloads_loaded": False,
            "model_loaded": False,
            "parent_forward_calls": 0,
            "parent_generation_calls": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "network_access": False,
        },
    }
    return {**body, "manifest_sha256": _canonical_sha256(body)}


def write_required_donor_token_ids_manifest(
    trace_cache_roots: Sequence[str | os.PathLike[str]],
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Write an immutable canonical manifest and return its external seal."""

    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise Lfm2RequiredDonorTokenIdsError(
            f"Output manifest already exists: {output}."
        )
    manifest = build_required_donor_token_ids_manifest(trace_cache_roots)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise Lfm2RequiredDonorTokenIdsError(
            f"Output temporary path already exists: {temporary}."
        )
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_json(manifest) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "schema": REQUIRED_DONOR_TOKEN_IDS_SCHEMA,
        "status": "written",
        "output_path": str(output),
        "output_size_bytes": output.stat().st_size,
        "output_file_sha256": _file_sha256(output),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_cache_count": manifest["source_cache_count"],
        "required_donor_token_count": manifest[
            "required_donor_token_count"
        ],
        "required_donor_token_ids_sha256": manifest[
            "required_donor_token_ids_sha256"
        ],
        "model_loaded": False,
        "parent_forward_calls": 0,
        "training_steps": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an authenticated donor-token union from sealed LFM2 "
            "trace caches without loading or executing a model."
        )
    )
    parser.add_argument(
        "--trace-cache",
        action="append",
        required=True,
        type=Path,
        help="Sealed LFM2 circuit trace cache root; may be repeated.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = write_required_donor_token_ids_manifest(
            args.trace_cache,
            args.output,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": REQUIRED_DONOR_TOKEN_IDS_SCHEMA,
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
