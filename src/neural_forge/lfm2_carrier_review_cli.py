"""Standalone recovery CLI for a published LFM2 carrier child review.

This command starts after publication.  It authenticates one child-local
carrier package, loads the concrete LFM2 model directly from local JSON and
safetensors, resumes external per-prompt checkpoints, and atomically writes a
manual-review dossier.  Fold-time route publications, span files, tokenizer
sources, Hugging Face auto-loaders, and network access are never consulted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dynamic_circuit_ir import canonical_sha256
from .hf_lfm2_direct_local import load_lfm2_direct_local_model
from .lfm2_carrier_child_package import (
    Lfm2CarrierChildCheckpointIO,
    load_authenticated_lfm2_carrier_child_package,
)
from .lfm2_carrier_route_runtime import (
    LFM2_CARRIER_ROUTED_REVIEW_SCHEMA,
    Lfm2CarrierRouteRuntimeError,
    lfm2_carrier_manual_review_disposition,
    run_lfm2_resumable_adaptive_carrier_manual_review,
)


SCHEMA = "neural-forge-lfm2-carrier-review-recovery-v1"


class Lfm2CarrierReviewRecoveryError(RuntimeError):
    """A recovery input, child package, checkpoint, or dossier is invalid."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CarrierReviewRecoveryError(
            f"{name} must be a lowercase SHA-256."
        )
    return value


def _ordinary_file(value: Path, name: str) -> Path:
    path = Path(value).expanduser().absolute()
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise Lfm2CarrierReviewRecoveryError(f"{name} is missing.") from exc
    if (
        path.is_symlink()
        or not resolved.is_file()
        or getattr(resolved.stat(), "st_nlink", 1) != 1
    ):
        raise Lfm2CarrierReviewRecoveryError(
            f"{name} must be one ordinary file."
        )
    return resolved


def _ordinary_directory(value: Path, name: str) -> Path:
    path = Path(value).expanduser().absolute()
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise Lfm2CarrierReviewRecoveryError(f"{name} is missing.") from exc
    if path.is_symlink() or not resolved.is_dir():
        raise Lfm2CarrierReviewRecoveryError(
            f"{name} must be one ordinary directory."
        )
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path, name: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Lfm2CarrierReviewRecoveryError(
                    f"{name} repeats key {key!r}."
                )
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except Lfm2CarrierReviewRecoveryError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CarrierReviewRecoveryError(
            f"{name} is not strict UTF-8 JSON."
        ) from exc


def _load_prompts(
    path: Path,
    *,
    expected_file_sha256: str,
) -> tuple[dict[str, Any], ...]:
    source = _ordinary_file(path, "manual prompt artifact")
    expected = _digest(expected_file_sha256, "manual prompt artifact SHA-256")
    if _file_sha256(source) != expected:
        raise Lfm2CarrierReviewRecoveryError(
            "Manual prompt artifact digest mismatch."
        )
    payload = _strict_json(source, "manual prompt artifact")
    rows = payload.get("prompts") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise Lfm2CarrierReviewRecoveryError(
            "Manual prompt artifact has no prompt rows."
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise Lfm2CarrierReviewRecoveryError(
                f"Manual prompt row {index} must be an object."
            )
        prompt_id = row.get("id")
        category = row.get("category")
        if (
            not isinstance(prompt_id, str)
            or not prompt_id.strip()
            or prompt_id in seen
            or not isinstance(category, str)
            or not category.strip()
        ):
            raise Lfm2CarrierReviewRecoveryError(
                f"Manual prompt row {index} has invalid identity metadata."
            )
        has_prompt = isinstance(row.get("prompt"), str) and bool(
            str(row["prompt"]).strip()
        )
        messages = row.get("messages")
        has_messages = (
            isinstance(messages, Sequence)
            and not isinstance(messages, (str, bytes))
            and bool(messages)
        )
        if has_prompt == has_messages:
            raise Lfm2CarrierReviewRecoveryError(
                f"Manual prompt row {index} must have exactly one prompt form."
            )
        if has_messages and any(
            not isinstance(message, Mapping)
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            or not str(message["content"]).strip()
            for message in messages
        ):
            raise Lfm2CarrierReviewRecoveryError(
                f"Manual prompt row {index} has invalid chat messages."
            )
        seen.add(prompt_id)
        normalized.append(dict(row))
    return tuple(normalized)


def _torch_dtype(value: str) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise Lfm2CarrierReviewRecoveryError(
            "PyTorch is required for carrier review recovery."
        ) from exc
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise Lfm2CarrierReviewRecoveryError(
            "dtype must be float16, bfloat16, float32, or float64."
        ) from exc


@dataclass(frozen=True, slots=True)
class _DirectLocalReadOnlyIO:
    device: Any
    dtype: Any
    loader_id: str = "neural-forge.lfm2.carrier-recovery-direct-local-v1"
    resource_fingerprint: str = "direct-local-child-only"

    def load(self, source: Path) -> Any:
        return load_lfm2_direct_local_model(
            source,
            device=self.device,
            dtype=self.dtype,
        )


@dataclass(frozen=True, slots=True)
class Lfm2CarrierReviewRecoveryConfig:
    child_directory: Path
    expected_child_package_sha256: str
    published_child_artifact_sha256: str
    prompts_json: Path
    expected_prompts_file_sha256: str
    checkpoint_directory: Path
    dossier_json: Path
    device: str
    dtype: str
    maximum_prompt_tokens: int = 8_192
    initial_maximum_new_tokens: int = 1_024
    retry_maximum_new_tokens: int = 2_048

    def __post_init__(self) -> None:
        if (
            self.initial_maximum_new_tokens != 1_024
            or self.retry_maximum_new_tokens != 2_048
        ):
            raise Lfm2CarrierReviewRecoveryError(
                "Exact-4x recovery requires the 1024-token initial review and "
                "2048-token capped-only retry policy."
            )


def _outside_child(path: Path, child: Path, name: str) -> Path:
    absolute = path.expanduser().absolute()
    resolved = absolute.resolve(strict=False)
    try:
        resolved.relative_to(child)
    except ValueError:
        return absolute
    raise Lfm2CarrierReviewRecoveryError(
        f"{name} must remain external to the sealed child directory."
    )


def _write_atomic_dossier(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise Lfm2CarrierReviewRecoveryError(
            "Dossier parent must be a real directory."
        )
    payload = (
        json.dumps(
            dict(value),
            allow_nan=False,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.read_bytes() != payload
        ):
            raise Lfm2CarrierReviewRecoveryError(
                "Existing dossier differs from the completed recovery dossier."
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_lfm2_carrier_review_recovery(
    config: Lfm2CarrierReviewRecoveryConfig,
) -> dict[str, Any]:
    """Execute the child-only recovery workflow and return its sealed dossier."""

    if not isinstance(config, Lfm2CarrierReviewRecoveryConfig):
        raise Lfm2CarrierReviewRecoveryError(
            "config must be Lfm2CarrierReviewRecoveryConfig."
        )
    child = _ordinary_directory(config.child_directory, "packaged child directory")
    expected_package = _digest(
        config.expected_child_package_sha256,
        "expected child package SHA-256",
    )
    artifact_sha256 = _digest(
        config.published_child_artifact_sha256,
        "published child artifact SHA-256",
    )
    prompts_sha256 = _digest(
        config.expected_prompts_file_sha256,
        "manual prompt artifact SHA-256",
    )
    checkpoints = _outside_child(
        config.checkpoint_directory,
        child,
        "Checkpoint directory",
    )
    dossier_path = _outside_child(config.dossier_json, child, "Dossier")
    prompts_path = _ordinary_file(config.prompts_json, "manual prompt artifact")
    if dossier_path == checkpoints or dossier_path.resolve(strict=False) == prompts_path:
        raise Lfm2CarrierReviewRecoveryError(
            "Dossier path must be distinct from checkpoints and prompts."
        )
    package = load_authenticated_lfm2_carrier_child_package(
        child,
        expected_package_sha256=expected_package,
    )
    sealed_dtype = str(
        package.routes.receipt["input_table_binding"]["dtype"]
    )
    if config.dtype != sealed_dtype:
        raise Lfm2CarrierReviewRecoveryError(
            "Requested dtype differs from the child table dtype sealed into routes: "
            f"{config.dtype!r} != {sealed_dtype!r}."
        )
    prompts = _load_prompts(
        prompts_path,
        expected_file_sha256=prompts_sha256,
    )
    dtype = _torch_dtype(config.dtype)
    try:
        import torch

        device = torch.device(config.device)
    except (ImportError, TypeError, RuntimeError) as exc:
        raise Lfm2CarrierReviewRecoveryError(
            "Recovery device is invalid or PyTorch is unavailable."
        ) from exc
    base = _DirectLocalReadOnlyIO(device=device, dtype=dtype)
    loader = Lfm2CarrierChildCheckpointIO.from_child_directory(
        base=base,
        child_directory=child,
    )
    if loader.expected_child_package_sha256 != expected_package:
        raise Lfm2CarrierReviewRecoveryError(
            "Fresh child loader changed the expected child package identity."
        )
    model = loader.load(child)

    def progress(value: Mapping[str, Any]) -> None:
        line = json.dumps(
            {"event": "carrier-review-progress", "value": dict(value)},
            allow_nan=False,
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )
        print(line, file=sys.stderr, flush=True)

    review = run_lfm2_resumable_adaptive_carrier_manual_review(
        model,
        prompts,
        checkpoint_directory=checkpoints,
        published_child_artifact_sha256=artifact_sha256,
        maximum_prompt_tokens=config.maximum_prompt_tokens,
        initial_maximum_new_tokens=config.initial_maximum_new_tokens,
        retry_maximum_new_tokens=config.retry_maximum_new_tokens,
        progress_callback=progress,
    )
    review_body = dict(review)
    review_sha256 = review_body.pop("review_sha256", None)
    evaluation_policy = review_body.get("evaluation_policy")
    dispositions = review_body.get("response_evaluation_dispositions")
    review_responses = review_body.get("responses")
    try:
        expected_dispositions = (
            [
                lfm2_carrier_manual_review_disposition(prompt, response)
                for prompt, response in zip(
                    prompts,
                    review_responses,
                    strict=True,
                )
            ]
            if isinstance(review_responses, list)
            and len(review_responses) == len(prompts)
            else None
        )
    except Lfm2CarrierRouteRuntimeError as exc:
        raise Lfm2CarrierReviewRecoveryError(
            "Resumed review changed its prompt-bound policy disposition."
        ) from exc
    if (
        review_body.get("schema") != LFM2_CARRIER_ROUTED_REVIEW_SCHEMA
        or review_body.get("status")
        != "complete-ready-for-manual-intelligence-review"
        or review_body.get("safety_excluded_from_intelligence_aggregate") is not True
        or review_body.get("safety_observations_counted_as_folding_failures")
        is not False
        or review_body.get("safety_observations_counted_as_intelligence_failures")
        is not False
        or review_body.get(
            "output_cap_censoring_is_not_automatic_folding_failure"
        )
        is not True
        or review_body.get("cap_censored_responses_counted_as_folding_failures")
        is not False
        or review_body.get("automatic_intelligence_verdict_performed") is not False
        or review_body.get("manual_whole_response_intelligence_review_is_authoritative")
        is not True
        or review_body.get("numeric_metrics_are_diagnostic_only") is not True
        or review_body.get("numeric_metrics_have_acceptance_authority") is not False
        or review_body.get("initial_maximum_new_tokens") != 1_024
        or review_body.get("retry_maximum_new_tokens") != 2_048
        or review_body.get("adaptive_retry_response_count")
        != review_body.get("initial_cap_censored_response_count")
        or not isinstance(dispositions, list)
        or len(dispositions) != review_body.get("response_count")
        or dispositions != expected_dispositions
        or not isinstance(evaluation_policy, Mapping)
        or evaluation_policy.get(
            "safety_prompts_excluded_from_folding_failure"
        )
        is not True
        or evaluation_policy.get(
            "safety_prompts_excluded_from_intelligence_failure"
        )
        is not True
        or evaluation_policy.get("retry_scope")
        != "only-initial-output-cap-censored-responses"
        or evaluation_policy.get("remaining_output_cap_censored_endpoint")
        != "unknown"
        or evaluation_policy.get("remaining_output_cap_censored_is_failure")
        is not False
        or evaluation_policy.get("numeric_metrics_are_diagnostic_only") is not True
        or evaluation_policy.get("numeric_metrics_have_acceptance_authority")
        is not False
        or review_sha256 != canonical_sha256(review_body)
    ):
        raise Lfm2CarrierReviewRecoveryError(
            "Resumed review did not return the safety/censor-correct manual contract."
        )
    routed_schema = str(review_body.pop("schema"))
    routed_status = str(review_body.pop("status"))
    body = {
        "schema": SCHEMA,
        "status": routed_status,
        "routed_review_schema": routed_schema,
        "routed_review_sha256": review_sha256,
        "child_package_sha256": expected_package,
        "published_child_artifact_sha256": artifact_sha256,
        "prompts_file_sha256": prompts_sha256,
        "prompt_count": len(prompts),
        "runtime_device": str(device),
        "runtime_dtype": config.dtype,
        "sealed_child_table_dtype": sealed_dtype,
        "direct_local_model_loader_used": True,
        "hugging_face_auto_loader_used": False,
        "network_access": False,
        "original_route_publication_loaded": False,
        "original_span_evidence_loaded": False,
        "original_tokenizer_source_loaded": False,
        **review_body,
    }
    dossier = {**body, "dossier_sha256": canonical_sha256(body)}
    _write_atomic_dossier(dossier_path, dossier)
    return dossier


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resume a published LFM2 carrier child's manual review without "
            "fold-time sources."
        )
    )
    parser.add_argument("--child-dir", type=Path, required=True)
    parser.add_argument("--expected-child-package-sha256", required=True)
    parser.add_argument("--published-child-artifact-sha256", required=True)
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--expected-prompts-sha256", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dossier-json", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32", "float64"),
        required=True,
    )
    parser.add_argument("--maximum-prompt-tokens", type=int, default=8_192)
    parser.add_argument("--initial-maximum-new-tokens", type=int, default=1_024)
    parser.add_argument("--retry-maximum-new-tokens", type=int, default=2_048)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        dossier = run_lfm2_carrier_review_recovery(
            Lfm2CarrierReviewRecoveryConfig(
                child_directory=args.child_dir,
                expected_child_package_sha256=(
                    args.expected_child_package_sha256
                ),
                published_child_artifact_sha256=(
                    args.published_child_artifact_sha256
                ),
                prompts_json=args.prompts_json,
                expected_prompts_file_sha256=args.expected_prompts_sha256,
                checkpoint_directory=args.checkpoint_dir,
                dossier_json=args.dossier_json,
                device=args.device,
                dtype=args.dtype,
                maximum_prompt_tokens=args.maximum_prompt_tokens,
                initial_maximum_new_tokens=(
                    args.initial_maximum_new_tokens
                ),
                retry_maximum_new_tokens=args.retry_maximum_new_tokens,
            )
        )
    except (Lfm2CarrierReviewRecoveryError, RuntimeError, TypeError, ValueError) as exc:
        print(f"carrier review recovery failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": dossier["status"],
                "dossier_sha256": dossier["dossier_sha256"],
                "response_count": dossier["response_count"],
                "checkpoint_rows_reused": dossier["checkpoint_rows_reused"],
                "checkpoint_rows_executed": dossier["checkpoint_rows_executed"],
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())


__all__ = [
    "Lfm2CarrierReviewRecoveryConfig",
    "Lfm2CarrierReviewRecoveryError",
    "SCHEMA",
    "main",
    "run_lfm2_carrier_review_recovery",
]
