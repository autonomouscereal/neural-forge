#!/usr/bin/env python3
"""Build one immutable dense mapped-root capsule from the sealed LFM2 cache.

This command never accepts or loads a parent model.  It authenticates a local
completed trace cache, the original causal-v2 trie/terminal receipts, the
strict tokenizer bridge, and both live local tokenizers.  Calibration records
are deliberately irrelevant: donor-canonical cache batches are authenticated
directly against the terminal receipt.  The command opens exactly one
selective cache session, projects every target vocabulary row by strict
donor-row index selection, persists only the
``[causal_roots, target_vocabulary]`` tensor, and writes a sealed reference
document accepted directly by ``resume_lfm2_pre_causal_body.py``.
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


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from neural_forge.dynamic_circuit_ir import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    Lfm2CircuitTraceCache,
)
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_interface_frontier import (  # noqa: E402
    LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA,
    LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA,
)
from neural_forge.hf_lfm2_mapped_causal_target_capsule import (  # noqa: E402
    collect_lfm2_mapped_causal_target_evidence,
    load_lfm2_mapped_causal_target_capsule,
    publish_lfm2_mapped_causal_target_capsule,
)
from neural_forge.hf_lfm2_pre_causal_resume_driver import (  # noqa: E402
    restore_lfm2_causal_trie_evidence_from_receipt,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    load_strict_tokenizer_bridge_artifact,
)


MAPPED_TARGET_REFERENCE_DOCUMENT_SCHEMA = (
    "neural-forge-lfm2-mapped-causal-target-reference-document-v1"
)
PRODUCER_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-mapped-causal-target-producer-run-v1"
)


class MappedTargetProducerError(RuntimeError):
    """One external pin, cache lineage, or publication boundary failed."""


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise MappedTargetProducerError(
            f"{label} must be one nonzero lowercase SHA-256 digest."
        )
    return value


def _ordinary_file(path: Path, label: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise MappedTargetProducerError(f"{label} must be one real local file.")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise MappedTargetProducerError(f"{label} cannot be resolved.") from exc
    if resolved != lexical:
        raise MappedTargetProducerError(f"{label} path cannot traverse a link.")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pinned_json(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_seal: str,
    seal_field: str,
    schema: str,
    label: str,
) -> dict[str, Any]:
    source = _ordinary_file(path, label)
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _digest(
        expected_file_sha256,
        f"{label} file pin",
    ):
        raise MappedTargetProducerError(f"{label} differs from its file pin.")
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MappedTargetProducerError(f"{label} is not valid UTF-8 JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise MappedTargetProducerError(f"{label} must contain one JSON object.")
    value = dict(parsed)
    body = dict(value)
    supplied = body.pop(seal_field, None)
    if (
        supplied != _digest(expected_seal, f"{label} semantic pin")
        or supplied != canonical_sha256(body)
        or value.get("schema") != schema
    ):
        raise MappedTargetProducerError(f"{label} seal, schema, or pin is invalid.")
    return value


def _write_exclusive_canonical_json(path: Path, value: Mapping[str, Any]) -> str:
    output = path.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise MappedTargetProducerError(f"Refusing to replace {output}.")
    parent = output.parent
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as exc:
        raise MappedTargetProducerError(
            "Reference-document parent cannot be resolved."
        ) from exc
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or resolved_parent != parent
    ):
        raise MappedTargetProducerError(
            "Reference-document parent must be one real existing directory."
        )
    payload = canonical_json_bytes(dict(value)) + b"\n"
    try:
        with output.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise MappedTargetProducerError(
            "Could not exclusively publish the mapped-target reference document."
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def build(
    *,
    trace_cache_directory: Path,
    expected_cache_key: str,
    expected_parent_identity_sha256: str,
    expected_cache_manifest_sha256: str,
    expected_cache_completion_sha256: str,
    terminal_evidence_json: Path,
    expected_terminal_evidence_file_sha256: str,
    expected_terminal_evidence_sha256: str,
    causal_trie_json: Path,
    expected_causal_trie_file_sha256: str,
    expected_causal_trie_sha256: str,
    donor_tokenizer_directory: Path,
    target_tokenizer_directory: Path,
    bridge_directory: Path,
    expected_bridge_artifact_sha256: str,
    expected_bridge_profile_sha256: str,
    capsule_directory: Path,
    reference_document: Path,
) -> dict[str, Any]:
    """Authenticate all pins, collect through one session, and publish."""

    capsule_output = capsule_directory.expanduser().absolute()
    reference_output = reference_document.expanduser().absolute()
    if (
        capsule_output.exists()
        or capsule_output.is_symlink()
        or reference_output.exists()
        or reference_output.is_symlink()
    ):
        raise MappedTargetProducerError(
            "Capsule and reference outputs must both be new paths."
        )
    expected_cache_key = _digest(expected_cache_key, "cache key")
    expected_parent_identity_sha256 = _digest(
        expected_parent_identity_sha256,
        "parent identity",
    )
    expected_cache_manifest_sha256 = _digest(
        expected_cache_manifest_sha256,
        "cache manifest",
    )
    expected_cache_completion_sha256 = _digest(
        expected_cache_completion_sha256,
        "cache completion",
    )
    terminal = _load_pinned_json(
        terminal_evidence_json,
        expected_file_sha256=expected_terminal_evidence_file_sha256,
        expected_seal=expected_terminal_evidence_sha256,
        seal_field="evidence_sha256",
        schema=LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA,
        label="terminal cache evidence",
    )
    trie_receipt = _load_pinned_json(
        causal_trie_json,
        expected_file_sha256=expected_causal_trie_file_sha256,
        expected_seal=expected_causal_trie_sha256,
        seal_field="result_sha256",
        schema=LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA,
        label="original causal-v2 trie evidence",
    )
    if (
        terminal.get("cache_key") != expected_cache_key
        or terminal.get("parent_identity_sha256")
        != expected_parent_identity_sha256
        or terminal.get("manifest_sha256") != expected_cache_manifest_sha256
        or terminal.get("completion_sha256")
        != expected_cache_completion_sha256
        or trie_receipt.get("parent_terminal_evidence_sha256")
        != terminal.get("evidence_sha256")
    ):
        raise MappedTargetProducerError(
            "Terminal/trie lineage differs from the external cache pins."
        )
    trie = restore_lfm2_causal_trie_evidence_from_receipt(trie_receipt)

    donor_tokenizer = load_lfm2_direct_local_tokenizer(
        donor_tokenizer_directory
    )
    target_tokenizer = load_lfm2_direct_local_tokenizer(
        target_tokenizer_directory
    )
    bridge = load_strict_tokenizer_bridge_artifact(
        bridge_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=_digest(
            expected_bridge_artifact_sha256,
            "bridge artifact",
        ),
    )
    if bridge.profile_fingerprint != _digest(
        expected_bridge_profile_sha256,
        "bridge profile",
    ):
        raise MappedTargetProducerError(
            "Live strict bridge differs from the external profile pin."
        )
    if (
        terminal.get("bridge_artifact_sha256") != bridge.artifact_sha256
        or terminal.get("bridge_profile_sha256") != bridge.profile_fingerprint
        or terminal.get("donor_tokenizer_semantics_sha256")
        != bridge.donor_tokenizer_semantics_sha256
        or terminal.get("target_tokenizer_semantics_sha256")
        != bridge.target_tokenizer_semantics_sha256
        or terminal.get("donor_vocabulary_size")
        != bridge.transport_profile.donor_row_capacity
        or terminal.get("target_vocabulary_size")
        != bridge.transport_profile.target_row_capacity
    ):
        raise MappedTargetProducerError(
            "Terminal vocabulary lineage differs from the live strict bridge."
        )

    # ``verify=False`` avoids a redundant full payload scan.  The collector's
    # one selective authenticated session hashes every cache file exactly once
    # and authenticates only the logit tensors it materializes.
    cache = Lfm2CircuitTraceCache.open(trace_cache_directory, verify=False)
    manifest = cache.manifest
    corpus = manifest.get("corpus")
    ordered_inputs = (
        corpus.get("ordered_batch_input_sha256")
        if isinstance(corpus, Mapping)
        else None
    )
    terminal_inputs = [
        row.get("input_sha256")
        for row in terminal.get("selected_batches", ())
        if isinstance(row, Mapping)
    ]
    if (
        cache.cache_key != expected_cache_key
        or cache.parent_fingerprint != expected_parent_identity_sha256
        or cache.batch_count != len(terminal_inputs)
        or not isinstance(ordered_inputs, list)
        or ordered_inputs != terminal_inputs
    ):
        raise MappedTargetProducerError(
            "Donor-canonical cache corpus and terminal batch order differ."
        )

    evidence = collect_lfm2_mapped_causal_target_evidence(
        cache,
        terminal_evidence=terminal,
        causal_trie_evidence=trie,
        bridge=bridge,
    )
    audit = evidence.receipt["operation_audit"]
    if (
        audit.get("parent_cache_selective_authenticated_sessions") != 1
        or audit.get("parent_model_objects_constructed") != 0
        or audit.get("parent_model_forward_calls") != 0
        or audit.get("parent_model_generate_calls") != 0
        or audit.get("parent_trace_recaptures") != 0
        or audit.get("training_steps") != 0
    ):
        raise MappedTargetProducerError(
            "Mapped-target producer operation audit is not cache-only."
        )
    reference = publish_lfm2_mapped_causal_target_capsule(
        evidence,
        output_directory=capsule_output,
    )
    # Fresh authentication after the atomic rename proves the emitted
    # reference names the exact bytes now visible to the resume command.
    load_lfm2_mapped_causal_target_capsule(reference)
    reference_body = {
        "schema": MAPPED_TARGET_REFERENCE_DOCUMENT_SCHEMA,
        "reference": reference.as_dict(),
        "producer_operation_audit": audit,
        "exactly_one_selective_cache_session": True,
        "parent_model_execution_performed": False,
        "parent_trace_recapture_performed": False,
        "training_performed": False,
    }
    reference_value = {
        **reference_body,
        "reference_sha256": canonical_sha256(reference_body),
    }
    reference_file_sha256 = _write_exclusive_canonical_json(
        reference_output,
        reference_value,
    )
    result_body = {
        "schema": PRODUCER_RECEIPT_SCHEMA,
        "status": "immutable-mapped-causal-target-capsule-published",
        "capsule_reference": reference.as_dict(),
        "reference_document": str(reference_output),
        "reference_document_file_sha256": reference_file_sha256,
        "reference_document_sha256": reference_value["reference_sha256"],
        "mapped_target_evidence_sha256": evidence.receipt["evidence_sha256"],
        "root_row_count": evidence.receipt["root_row_count"],
        "target_row_count": evidence.receipt["target_row_count"],
        "operation_audit": audit,
        "resume_cli_arguments": {
            "--mapped-target-reference-json": str(
                reference_output
            ),
            "--expected-mapped-target-reference-file-sha256": (
                reference_file_sha256
            ),
            "--expected-mapped-target-reference-sha256": reference_value[
                "reference_sha256"
            ],
        },
    }
    return {**result_body, "result_sha256": canonical_sha256(result_body)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-cache-directory", required=True, type=Path)
    parser.add_argument("--expected-cache-key", required=True)
    parser.add_argument("--expected-parent-identity-sha256", required=True)
    parser.add_argument("--expected-cache-manifest-sha256", required=True)
    parser.add_argument("--expected-cache-completion-sha256", required=True)
    parser.add_argument("--terminal-evidence-json", required=True, type=Path)
    parser.add_argument("--expected-terminal-evidence-file-sha256", required=True)
    parser.add_argument("--expected-terminal-evidence-sha256", required=True)
    parser.add_argument("--causal-trie-json", required=True, type=Path)
    parser.add_argument("--expected-causal-trie-file-sha256", required=True)
    parser.add_argument("--expected-causal-trie-sha256", required=True)
    parser.add_argument("--donor-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--target-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument("--expected-bridge-artifact-sha256", required=True)
    parser.add_argument("--expected-bridge-profile-sha256", required=True)
    parser.add_argument("--capsule-directory", required=True, type=Path)
    parser.add_argument("--reference-document", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build(
        trace_cache_directory=args.trace_cache_directory,
        expected_cache_key=args.expected_cache_key,
        expected_parent_identity_sha256=args.expected_parent_identity_sha256,
        expected_cache_manifest_sha256=args.expected_cache_manifest_sha256,
        expected_cache_completion_sha256=args.expected_cache_completion_sha256,
        terminal_evidence_json=args.terminal_evidence_json,
        expected_terminal_evidence_file_sha256=(
            args.expected_terminal_evidence_file_sha256
        ),
        expected_terminal_evidence_sha256=(
            args.expected_terminal_evidence_sha256
        ),
        causal_trie_json=args.causal_trie_json,
        expected_causal_trie_file_sha256=args.expected_causal_trie_file_sha256,
        expected_causal_trie_sha256=args.expected_causal_trie_sha256,
        donor_tokenizer_directory=args.donor_tokenizer_directory,
        target_tokenizer_directory=args.target_tokenizer_directory,
        bridge_directory=args.bridge_directory,
        expected_bridge_artifact_sha256=args.expected_bridge_artifact_sha256,
        expected_bridge_profile_sha256=args.expected_bridge_profile_sha256,
        capsule_directory=args.capsule_directory,
        reference_document=args.reference_document,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
