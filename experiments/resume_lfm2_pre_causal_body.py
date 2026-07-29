#!/usr/bin/env python3
"""Resume only the causal-output stage of one pinned exact-4x LFM2 body.

The command consumes an externally pinned pre-causal full-body capsule and
the immutable terminal/trie/sequence receipts emitted by the original whole
fold.  It restores the typed trie directly from that receipt, so no parent
model or parent trace cache is opened.  An optional externally pinned mapped
target capsule can complete every causal root before child replay; loading it
also opens no parent model or trace cache.  The target-native child prefixes
are replayed, the requested analytic residual-head policy is installed, and
a standalone direct-local checkpoint plus complete decoded review dossier is
published.

This production entrypoint is intentionally fixed to the live A0031
materialization contract: dense FP32 on CUDA:0 for both replay and analytic
solve.  The package API remains usable by tiny CPU tests, but the production
CLI does not advertise dtype/device choices which its pinned capsule rejects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
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
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    Lfm2DirectLocalCheckpointIO,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_exact4x_composite_interface import (  # noqa: E402
    LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA,
)
from neural_forge.hf_lfm2_interface_frontier import (  # noqa: E402
    LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA,
    LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA,
    LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA,
    LFM2_UNSUPPORTED_LOGIT_POLICIES,
    LFM2_UNSUPPORTED_LOGIT_POLICY_ADAPTIVE_STRONGEST_SURVIVING_BACKGROUND_SEAL,
)
from neural_forge.hf_lfm2_mapped_causal_target_capsule import (  # noqa: E402
    Lfm2MappedCausalTargetReference,
)
from neural_forge.hf_lfm2_pre_causal_body_resume_capsule import (  # noqa: E402
    Lfm2PreCausalBodyResumeReference,
)
from neural_forge.hf_lfm2_pre_causal_resume_driver import (  # noqa: E402
    Lfm2PinnedCausalResumeEvidence,
    restore_lfm2_causal_trie_evidence_from_receipt,
    resume_lfm2_pre_causal_body,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    load_strict_tokenizer_bridge_artifact,
)
from neural_forge.lfm2_decoded_review import load_review_prompts  # noqa: E402
from neural_forge.lfm2_exact4x_contract import (  # noqa: E402
    CAUSAL_HEAD_RECIPE,
)


REFERENCE_DOCUMENT_SCHEMA = (
    "neural-forge-lfm2-pre-causal-body-resume-reference-v1"
)
MAPPED_TARGET_REFERENCE_DOCUMENT_SCHEMA = (
    "neural-forge-lfm2-mapped-causal-target-reference-document-v1"
)
_A0031_DEVICE = torch.device("cuda:0")
_A0031_DTYPE = torch.float32


class ResumeEntrypointError(RuntimeError):
    """A local external pin or CLI request is invalid."""


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ResumeEntrypointError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _ordinary_file(path: Path, label: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise ResumeEntrypointError(f"{label} must be one real local file.")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise ResumeEntrypointError(f"{label} cannot be resolved.") from exc
    if resolved != lexical:
        raise ResumeEntrypointError(f"{label} path cannot traverse a link.")
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
        raise ResumeEntrypointError(f"{label} differs from its external file pin.")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumeEntrypointError(f"{label} is not valid UTF-8 JSON.") from exc
    if not isinstance(value, Mapping):
        raise ResumeEntrypointError(f"{label} must contain one JSON object.")
    result = dict(value)
    body = dict(result)
    supplied = body.pop(seal_field, None)
    if (
        supplied != _digest(expected_seal, f"{label} semantic pin")
        or supplied != canonical_sha256(body)
        or result.get("schema") != schema
    ):
        raise ResumeEntrypointError(f"{label} seal, pin, or schema is invalid.")
    return result


def _reference_from_document(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_reference_sha256: str,
) -> Lfm2PreCausalBodyResumeReference:
    document = _load_pinned_json(
        path,
        expected_file_sha256=expected_file_sha256,
        expected_seal=expected_reference_sha256,
        seal_field="reference_sha256",
        schema=REFERENCE_DOCUMENT_SCHEMA,
        label="pre-causal body reference",
    )
    raw = document.get("reference")
    if not isinstance(raw, Mapping):
        raise ResumeEntrypointError(
            "Pre-causal body reference document has no typed reference."
        )
    try:
        return Lfm2PreCausalBodyResumeReference.from_mapping(raw)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ResumeEntrypointError(
            "Pre-causal body reference fields are missing or invalid."
        ) from exc


def _mapped_target_reference_from_document(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_reference_sha256: str,
) -> Lfm2MappedCausalTargetReference:
    document = _load_pinned_json(
        path,
        expected_file_sha256=expected_file_sha256,
        expected_seal=expected_reference_sha256,
        seal_field="reference_sha256",
        schema=MAPPED_TARGET_REFERENCE_DOCUMENT_SCHEMA,
        label="mapped causal target reference",
    )
    raw = document.get("reference")
    if not isinstance(raw, Mapping):
        raise ResumeEntrypointError(
            "Mapped causal target reference document has no typed reference."
        )
    try:
        return Lfm2MappedCausalTargetReference.from_mapping(raw)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ResumeEntrypointError(
            "Mapped causal target reference fields are missing or invalid."
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--pre-causal-reference-json", required=True, type=Path)
    parser.add_argument(
        "--expected-pre-causal-reference-file-sha256",
        required=True,
    )
    parser.add_argument("--expected-pre-causal-reference-sha256", required=True)
    parser.add_argument("--mapped-target-reference-json", type=Path)
    parser.add_argument("--expected-mapped-target-reference-file-sha256")
    parser.add_argument("--expected-mapped-target-reference-sha256")
    parser.add_argument("--terminal-evidence-json", required=True, type=Path)
    parser.add_argument("--expected-terminal-evidence-file-sha256", required=True)
    parser.add_argument("--expected-terminal-evidence-sha256", required=True)
    parser.add_argument("--causal-trie-json", required=True, type=Path)
    parser.add_argument("--expected-causal-trie-file-sha256", required=True)
    parser.add_argument("--expected-causal-trie-sha256", required=True)
    parser.add_argument("--sequence-plan-json", required=True, type=Path)
    parser.add_argument("--expected-sequence-plan-file-sha256", required=True)
    parser.add_argument("--expected-sequence-plan-sha256", required=True)
    parser.add_argument("--structural-interface-json", required=True, type=Path)
    parser.add_argument(
        "--expected-structural-interface-file-sha256",
        required=True,
    )
    parser.add_argument("--expected-structural-interface-sha256", required=True)
    parser.add_argument("--donor-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--target-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument("--expected-bridge-artifact-sha256", required=True)
    parser.add_argument("--prompts-json", required=True, type=Path)
    parser.add_argument("--expected-prompts-file-sha256", required=True)
    parser.add_argument(
        "--unsupported-logit-policy",
        choices=tuple(sorted(LFM2_UNSUPPORTED_LOGIT_POLICIES)),
        default=(
            LFM2_UNSUPPORTED_LOGIT_POLICY_ADAPTIVE_STRONGEST_SURVIVING_BACKGROUND_SEAL
        ),
    )
    parser.add_argument(
        "--ridge-fraction",
        type=float,
        default=float(CAUSAL_HEAD_RECIPE["ridge_fraction"]),
    )
    parser.add_argument(
        "--residual-scale-ppm",
        type=int,
        default=int(CAUSAL_HEAD_RECIPE["residual_scale_ppm"]),
    )
    parser.add_argument(
        "--row-norm-calibration-ppm",
        type=int,
        default=int(CAUSAL_HEAD_RECIPE["row_norm_calibration_ppm"]),
    )
    parser.add_argument(
        "--protect-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=bool(CAUSAL_HEAD_RECIPE["protect_specials"]),
    )
    parser.add_argument("--maximum-prompt-tokens", type=int, default=512)
    parser.add_argument("--maximum-new-tokens", type=int, default=256)
    parser.add_argument("--minimum-parameter-compaction-ratio", type=float, default=4.0)
    parser.add_argument(
        "--candidate-id",
        default="exact4x-parent-free-causal-resume",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ResumeEntrypointError(
            "The A0031 resume CLI is fixed to the authenticated FP32 CUDA:0 "
            "materialization contract, but CUDA:0 is unavailable."
        )
    reference = _reference_from_document(
        args.pre_causal_reference_json,
        expected_file_sha256=args.expected_pre_causal_reference_file_sha256,
        expected_reference_sha256=args.expected_pre_causal_reference_sha256,
    )
    mapped_reference_arguments = (
        args.mapped_target_reference_json,
        args.expected_mapped_target_reference_file_sha256,
        args.expected_mapped_target_reference_sha256,
    )
    if any(value is not None for value in mapped_reference_arguments) and not all(
        value is not None for value in mapped_reference_arguments
    ):
        raise ResumeEntrypointError(
            "Mapped target reference path, file pin, and semantic pin must be "
            "supplied together."
        )
    mapped_target_reference = (
        _mapped_target_reference_from_document(
            args.mapped_target_reference_json,
            expected_file_sha256=(
                args.expected_mapped_target_reference_file_sha256
            ),
            expected_reference_sha256=(
                args.expected_mapped_target_reference_sha256
            ),
        )
        if args.mapped_target_reference_json is not None
        else None
    )
    terminal = _load_pinned_json(
        args.terminal_evidence_json,
        expected_file_sha256=args.expected_terminal_evidence_file_sha256,
        expected_seal=args.expected_terminal_evidence_sha256,
        seal_field="evidence_sha256",
        schema=LFM2_TERMINAL_CACHE_EVIDENCE_SCHEMA,
        label="terminal cache evidence",
    )
    trie_receipt = _load_pinned_json(
        args.causal_trie_json,
        expected_file_sha256=args.expected_causal_trie_file_sha256,
        expected_seal=args.expected_causal_trie_sha256,
        seal_field="result_sha256",
        schema=LFM2_CAUSAL_TRIE_EVIDENCE_SCHEMA,
        label="causal trie evidence",
    )
    sequence = _load_pinned_json(
        args.sequence_plan_json,
        expected_file_sha256=args.expected_sequence_plan_file_sha256,
        expected_seal=args.expected_sequence_plan_sha256,
        seal_field="plan_sha256",
        schema=LFM2_INTERFACE_SEQUENCE_PLAN_SCHEMA,
        label="interface sequence plan",
    )
    structural = _load_pinned_json(
        args.structural_interface_json,
        expected_file_sha256=args.expected_structural_interface_file_sha256,
        expected_seal=args.expected_structural_interface_sha256,
        seal_field="result_sha256",
        schema=LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA,
        label="structural interface install",
    )
    prompts_path = _ordinary_file(args.prompts_json, "manual-review prompts")
    if _file_sha256(prompts_path) != _digest(
        args.expected_prompts_file_sha256,
        "manual-review prompts file pin",
    ):
        raise ResumeEntrypointError(
            "Manual-review prompts differ from their external file pin."
        )
    prompts = load_review_prompts(prompts_path)

    donor_tokenizer = load_lfm2_direct_local_tokenizer(
        args.donor_tokenizer_directory
    )
    target_tokenizer = load_lfm2_direct_local_tokenizer(
        args.target_tokenizer_directory
    )
    bridge = load_strict_tokenizer_bridge_artifact(
        args.bridge_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=_digest(
            args.expected_bridge_artifact_sha256,
            "bridge artifact pin",
        ),
    )
    checkpoint_io = Lfm2DirectLocalCheckpointIO(
        tokenizer_bridge=bridge,
        device=_A0031_DEVICE,
        dtype=_A0031_DTYPE,
    )
    evidence = Lfm2PinnedCausalResumeEvidence(
        terminal_receipt=terminal,
        causal_trie=restore_lfm2_causal_trie_evidence_from_receipt(trie_receipt),
        sequence_plan=sequence,
        structural_interface_receipt=structural,
    )
    result = resume_lfm2_pre_causal_body(
        body_reference=reference,
        checkpoint_io=checkpoint_io,
        evidence=evidence,
        target_tokenizer=target_tokenizer,
        prompts=prompts,
        output_directory=args.run_directory,
        unsupported_logit_policy=args.unsupported_logit_policy,
        model_device=_A0031_DEVICE,
        solve_device=_A0031_DEVICE,
        ridge_fraction=args.ridge_fraction,
        residual_scale_ppm=args.residual_scale_ppm,
        row_norm_calibration_ppm=args.row_norm_calibration_ppm,
        protect_special_tokens=args.protect_special_tokens,
        maximum_prompt_tokens=args.maximum_prompt_tokens,
        maximum_new_tokens=args.maximum_new_tokens,
        minimum_parameter_compaction_ratio=(
            args.minimum_parameter_compaction_ratio
        ),
        candidate_id=args.candidate_id,
        mapped_target_reference=mapped_target_reference,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
