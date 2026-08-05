#!/usr/bin/env python3
"""Run one sealed on-policy evidence extension round against a published child.

``run_lfm2_dynamic_catalog.py`` can only produce rounds for the child *its own*
fold pass just published, and it builds the round's capture configuration from
its CLI flags rather than from the sealed base cache.  Neither is usable for the
exact-4x carrier lane after the fact:

* r12's child is already published, so re-running the catalog runner to reach
  ``--on-policy-evidence-rounds`` would re-execute the whole closed-loop fold;
  and
* the sealed base cache
  ``v4096_modelcap_v2_broad256_cpcf_decision_v5/parent_trace_cache/76ff3cbc...``
  was captured with ``require_top1_agreement`` **and**
  ``allow_top1_disagreement_within_numeric_error_envelope`` both true
  (``selection_sha256 = f229e5f3...``).  The runner never sets the second flag,
  so no combination of its CLI arguments reproduces that selection and
  ``run_lfm2_on_policy_evidence_round`` fails closed on the comparable-capture
  clause of the extension contract.

This runner therefore rebuilds the capture configuration **from the sealed base
manifest itself** and proves the reconstruction reproduces the sealed
``selection_sha256`` before anything is captured.  Everything else is the
library contract: the freshly reloaded published child is the only model run
during the rollout, the parent is evaluated exactly once per surviving batch,
no optimizer exists, no gradient is produced, and the round seals its own
manifest.  On a bridged lane the round additionally seals the target-token
preimage of every capture batch, which is what a later refold replays.

There is no fold here.  This runner never writes into the source run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:  # pragma: no cover - import shape depends on the deployed PYTHONPATH
    from experiments import run_lfm2_residual_attention_fold as residual
except ImportError:  # pragma: no cover
    import run_lfm2_residual_attention_fold as residual  # type: ignore[no-redef]

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    CACHED_VERIFIED_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    _execution_identity,
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_on_policy_evidence_round import (  # noqa: E402
    Lfm2OnPolicyEvidenceRoundError,
    Lfm2OnPolicyRolloutConfig,
    _deep_thaw,
    run_lfm2_on_policy_evidence_round,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    load_strict_tokenizer_bridge_artifact,
)


RUNNER_SCHEMA = "neural-forge-lfm2-r12-on-policy-round-runner-v1"


class R12OnPolicyRoundError(RuntimeError):
    """The staged round request is unusable and nothing was captured."""


def _load_fit_prompts(path: Path) -> tuple[str, ...]:
    """Load the explicit JSON fit-prompt corpus, same shape the runner reads."""

    resolved = path.expanduser().resolve(strict=True)
    try:
        document: Any = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise R12OnPolicyRoundError(
            f"On-policy fit prompts are not readable JSON: {resolved}."
        ) from exc
    if isinstance(document, dict):
        document = document.get("prompts")
    if not isinstance(document, list) or not document:
        raise R12OnPolicyRoundError(
            "On-policy fit prompts must be a nonempty JSON list of prompt "
            "strings or rows carrying a nonempty 'prompt' string."
        )
    prompts: list[str] = []
    for row in document:
        value = row.get("prompt") if isinstance(row, dict) else row
        if not isinstance(value, str) or not value.strip():
            raise R12OnPolicyRoundError(
                "Every on-policy fit prompt must be a nonempty string."
            )
        prompts.append(value)
    return tuple(prompts)


def _capture_config_from_sealed_manifest(
    capture: dict[str, Any],
) -> Lfm2CircuitTraceCaptureConfig:
    """Rebuild the base cache's exact capture selection from its own manifest.

    The extension contract requires the round to capture under the same
    selection policy as the sealed base cache.  Deriving the configuration from
    the sealed document and then re-deriving its digest is the only form of
    that statement which cannot drift from what is actually on disk.
    """

    try:
        rebuilt = Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=int(capture["sample_last_tokens"]),
            max_decode_steps=int(capture["max_decode_steps"]),
            max_capture_bytes=int(capture["max_capture_bytes"]),
            logit_rtol=float(capture["logit_rtol"]),
            logit_atol=float(capture["logit_atol"]),
            require_top1_agreement=bool(capture["require_top1_agreement"]),
            allow_top1_disagreement_within_numeric_error_envelope=bool(
                capture["allow_top1_disagreement_within_numeric_error_envelope"]
            ),
            decode_trace_mode=str(
                capture.get("decode_trace_mode", CACHED_VERIFIED_TRACE_MODE)
            ),
            capture_causal_circuit_evidence=bool(
                capture.get("capture_causal_circuit_evidence", False)
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise R12OnPolicyRoundError(
            "The sealed base cache capture block is not a reconstructable "
            f"capture selection: {exc}"
        ) from exc
    sealed_selection = capture.get("selection_sha256")
    rebuilt_selection = rebuilt.as_dict()["selection_sha256"]
    if rebuilt_selection != sealed_selection:
        raise R12OnPolicyRoundError(
            "Rebuilt capture selection "
            f"{rebuilt_selection} does not reproduce the sealed base cache "
            f"selection {sealed_selection}; extension evidence captured under "
            "it would not be comparable with the base corpus."
        )
    return rebuilt


def _load_parent_matching_fingerprint(
    directory: Path,
    *,
    device: torch.device,
    expected_fingerprint: str,
    requested_dtype: str,
) -> tuple[Any, str]:
    """Load the local parent in the dtype that reproduces its sealed identity.

    The round refuses a parent whose fingerprint differs from the sealed base
    cache's, and the identity depends on the load dtype, so the requested dtype
    is tried first and the remaining representable candidates after it.  Every
    candidate is released before the next is built.
    """

    candidates: list[str] = []
    for name in (requested_dtype, "float32", "bfloat16", "float16"):
        if name not in candidates:
            candidates.append(name)
    observed: list[tuple[str, str]] = []
    for name in candidates:
        try:
            dtype = residual._torch_dtype(name)
        except Exception:
            continue
        model = residual.load_local_lfm2_model(
            directory,
            device=device,
            dtype=dtype,
        )
        fingerprint = fingerprint_lfm2_parent(model).identity_sha256
        if fingerprint == expected_fingerprint:
            return model, name
        observed.append((name, fingerprint))
        del model
    raise R12OnPolicyRoundError(
        "No local parent dtype reproduces the sealed base cache parent "
        f"fingerprint {expected_fingerprint}; observed {observed}."
    )


def _bind_sealed_execution_identity(
    parent: Any,
    *,
    device: torch.device,
    sealed_execution: dict[str, Any],
) -> dict[str, Any]:
    """Reproduce the sealed base cache's execution identity, or refuse.

    Extension evidence must be captured under the same execution identity as
    the base corpus, and the identity binds the attention backend and the input
    device type.  Transformers auto-selects SDPA even for a strictly local
    load, while the PACK's base cache was captured under the explicit
    Q/K/softmax/V path, so the backend is set here -- before any forward, and
    without touching learned state or the serialized identity, exactly as
    ``prepare_lfm2_tokenizer_bridge_evidence`` set it when the cache was
    sealed.  The device type cannot be repaired here: a CPU round against a
    CUDA-captured base corpus is a real incomparability and is refused.
    """

    config = getattr(parent, "config", None)
    if config is None:
        raise R12OnPolicyRoundError(
            "Loaded LFM2 parent does not expose its runtime configuration."
        )
    expected_backend = sealed_execution.get("attention_backend")
    if isinstance(expected_backend, str) and expected_backend:
        config._attn_implementation = expected_backend
        if getattr(config, "_attn_implementation", None) != expected_backend:
            raise R12OnPolicyRoundError(
                f"Unable to select the {expected_backend!r} attention backend "
                "the sealed base cache was captured under."
            )
    observed = _execution_identity(parent, device)
    if observed["execution_sha256"] != sealed_execution.get("execution_sha256"):
        raise R12OnPolicyRoundError(
            "Live execution identity does not reproduce the sealed base "
            f"cache: live={observed}, sealed={dict(sealed_execution)}. The "
            "round would capture evidence that no closed-form solve can "
            "consume together with the base corpus."
        )
    return observed


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_directory = args.base_cache_directory.expanduser().resolve(strict=True)
    # Opened read-only for its sealed identities; the round opens its own
    # handle and owns the lifetime of the one it captures against.
    base_cache = open_lfm2_circuit_trace_cache(
        base_directory,
        verify=args.verify_base_cache,
    )
    base_manifest = dict(base_cache.manifest)
    parent_fingerprint = base_cache.parent_fingerprint
    base_cache_key = base_cache.cache_key
    base_batch_count = len(base_cache)
    del base_cache
    capture_block = base_manifest.get("capture")
    if not isinstance(capture_block, dict):
        raise R12OnPolicyRoundError(
            f"Sealed base cache {base_directory} carries no capture block."
        )
    capture_config = _capture_config_from_sealed_manifest(capture_block)

    prompts = _load_fit_prompts(args.fit_prompts)
    if len(prompts) > args.max_prompts:
        raise R12OnPolicyRoundError(
            f"{len(prompts)} fit prompts exceed --max-prompts "
            f"{args.max_prompts}."
        )
    rollout = Lfm2OnPolicyRolloutConfig(
        max_prompts=args.max_prompts,
        max_new_tokens=args.max_new_tokens,
        max_context_tokens=args.max_context_tokens,
        seed=args.rollout_seed,
        max_capture_batches=args.max_capture_batches,
        truncate_forced_decode_to_capture_budget=(
            args.truncate_forced_decode_to_capture_budget
        ),
        screen_cached_decode_admissibility=(
            args.screen_cached_decode_admissibility
        ),
        min_cached_decode_admissible_batches=(
            args.min_cached_decode_admissible_batches
        ),
    )
    if rollout.max_new_tokens > capture_config.max_decode_steps and not (
        rollout.truncate_forced_decode_to_capture_budget
    ):
        raise R12OnPolicyRoundError(
            f"--max-new-tokens {rollout.max_new_tokens} exceeds the sealed "
            f"capture budget max_decode_steps={capture_config.max_decode_steps}; "
            "lower it or pass --truncate-forced-decode-to-capture-budget."
        )

    device = torch.device(args.device)
    parent, parent_dtype = _load_parent_matching_fingerprint(
        args.parent_model.expanduser().resolve(strict=True),
        device=device,
        expected_fingerprint=parent_fingerprint,
        requested_dtype=args.parent_dtype,
    )
    sealed_execution = base_manifest.get("execution")
    if not isinstance(sealed_execution, dict):
        raise R12OnPolicyRoundError(
            f"Sealed base cache {base_directory} carries no execution block."
        )
    execution = _bind_sealed_execution_identity(
        parent,
        device=device,
        sealed_execution=sealed_execution,
    )
    parent_tokenizer = residual.load_local_tokenizer(
        args.tokenizer_directory.expanduser().resolve(strict=True)
    )

    bridge = None
    if args.target_tokenizer_directory is not None:
        target_directory = (
            args.target_tokenizer_directory.expanduser().resolve(strict=True)
        )
        bridge = load_strict_tokenizer_bridge_artifact(
            target_directory,
            donor_tokenizer=parent_tokenizer,
            target_tokenizer=residual.load_local_tokenizer(target_directory),
            expected_artifact_sha256=args.tokenizer_bridge_artifact_sha256,
        )
        if (
            args.tokenizer_bridge_profile_sha256 is not None
            and bridge.profile_fingerprint
            != args.tokenizer_bridge_profile_sha256
        ):
            raise R12OnPolicyRoundError(
                "Tokenizer bridge transport profile differs from the explicit "
                "profile digest."
            )

    executed = run_lfm2_on_policy_evidence_round(
        child_artifact_directory=(
            args.child_artifact_directory.expanduser().resolve(strict=True)
        ),
        parent_model=parent,
        parent_tokenizer=parent_tokenizer,
        parent_fingerprint=parent_fingerprint,
        fit_prompts=prompts,
        manual_prompt_manifest_path=(
            args.manual_prompt_manifest.expanduser().resolve(strict=True)
        ),
        base_cache_directory=base_directory,
        output_root=args.round_root.expanduser().absolute(),
        capture_config=capture_config,
        round_index=args.round_index,
        max_rounds=args.max_rounds,
        rollout=rollout,
        previous_round_manifest_paths=tuple(
            value.expanduser().resolve(strict=True)
            for value in (args.previous_round_manifests or ())
        ),
        child_device=args.device,
        child_dtype=args.child_dtype,
        input_device=device,
        parent_tokenizer_name=args.parent_name,
        verify_base_cache=args.verify_base_cache,
        tokenizer_bridge=None if bridge is None else bridge,
        expected_tokenizer_bridge_artifact_sha256=(
            args.tokenizer_bridge_artifact_sha256
        ),
    )

    manifest = executed.manifest
    body = {
        "schema": RUNNER_SCHEMA,
        "status": "sealed",
        "base_cache_directory": str(base_directory),
        "base_cache_key": base_cache_key,
        "base_cache_batch_count": base_batch_count,
        "base_capture_selection_sha256": capture_block["selection_sha256"],
        "rebuilt_capture_selection_sha256": (
            capture_config.as_dict()["selection_sha256"]
        ),
        "parent_identity_sha256": parent_fingerprint,
        "parent_load_dtype": parent_dtype,
        "execution_identity": _deep_thaw(execution),
        "sealed_execution_identity_sha256": (
            sealed_execution["execution_sha256"]
        ),
        "child_artifact_directory": str(
            args.child_artifact_directory.expanduser().resolve(strict=True)
        ),
        "fit_prompt_path": str(args.fit_prompts.expanduser().resolve(strict=True)),
        "fit_prompt_file_sha256": hashlib.sha256(
            args.fit_prompts.expanduser().resolve(strict=True).read_bytes()
        ).hexdigest(),
        "fit_prompt_count": len(prompts),
        "prompt_corpus_sha256": (
            manifest["prompt_corpus"]["prompt_corpus_sha256"]
        ),
        "manual_prompt_manifest_path": str(
            args.manual_prompt_manifest.expanduser().resolve(strict=True)
        ),
        # _deep_thaw, not dict(): the loaded manifest is deep-frozen, and a
        # shallow dict() leaves nested mappingproxy rows that json.dumps
        # refuses -- the first round with non-empty nested rows proved it.
        "disjointness": _deep_thaw(manifest["disjointness"]),
        "round_index": executed.round_index,
        "max_rounds": executed.max_rounds,
        "round_manifest_path": str(executed.manifest_path),
        "round_manifest_sha256": executed.manifest_sha256,
        "extension_cache_path": str(executed.extension_cache_path),
        "extension_cache_key": executed.extension_cache_key,
        "extension_batch_count": executed.batch_count,
        "parent_cache_input_mode": executed.parent_cache_input_mode,
        "carries_target_token_preimage": executed.carries_target_preimage,
        "target_preimage_bridge_artifact_sha256": (
            executed.target_preimage_bridge_artifact_sha256
        ),
        "counters": _deep_thaw(manifest["counters"]),
        "rollout": _deep_thaw(manifest["rollout"]),
    }
    admissibility = manifest.get("capture", {}).get(
        "cached_decode_admissibility"
    )
    if admissibility is not None:
        body["cached_decode_admissibility"] = _deep_thaw(admissibility)
    receipt = {**body, "runner_receipt_sha256": canonical_sha256(body)}
    if args.receipt is not None:
        destination = args.receipt.expanduser().absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one sealed on-policy evidence extension round against an "
            "already published standalone child."
        )
    )
    parser.add_argument("--child-artifact-directory", type=Path, required=True)
    parser.add_argument("--parent-model", type=Path, required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument(
        "--target-tokenizer-directory",
        type=Path,
        default=None,
        help=(
            "Sealed strict tokenizer bridge directory. Supplying it captures a "
            "bridged round whose batches are the one-row donor image of the "
            "child's own tokenization and which seals that target preimage."
        ),
    )
    parser.add_argument("--tokenizer-bridge-artifact-sha256", default=None)
    parser.add_argument("--tokenizer-bridge-profile-sha256", default=None)
    parser.add_argument("--base-cache-directory", type=Path, required=True)
    parser.add_argument("--fit-prompts", type=Path, required=True)
    parser.add_argument("--manual-prompt-manifest", type=Path, required=True)
    parser.add_argument("--round-root", type=Path, required=True)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument(
        "--previous-round-manifest",
        dest="previous_round_manifests",
        type=Path,
        action="append",
        default=None,
        help="Already sealed round manifest, repeatable and ordered from 0.",
    )
    parser.add_argument("--max-prompts", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    parser.add_argument("--max-capture-batches", type=int, default=256)
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument(
        "--truncate-forced-decode-to-capture-budget",
        action="store_true",
        help=(
            "Truncate a re-tokenized continuation that exceeds the sealed "
            "capture budget instead of failing the round."
        ),
    )
    parser.add_argument(
        "--screen-cached-decode-admissibility",
        action="store_true",
        help=(
            "Pre-screen each candidate trajectory's cached decode against a "
            "use_cache=False full-prefix recompute with the collector's own "
            "comparison, excluding and recording trajectories the sealed "
            "capture gate would refuse instead of failing the round."
        ),
    )
    parser.add_argument(
        "--min-cached-decode-admissible-batches",
        type=int,
        default=1,
        help=(
            "Fail the round closed if fewer than this many capture batches "
            "survive the cached-decode admissibility screen."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--child-dtype", default="float32")
    parser.add_argument("--parent-dtype", default="float32")
    parser.add_argument("--parent-name", default="parent")
    parser.add_argument("--verify-base-cache", action="store_true")
    parser.add_argument("--receipt", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run(args)
    except (
        R12OnPolicyRoundError,
        Lfm2OnPolicyEvidenceRoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"on-policy round failed closed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
