#!/usr/bin/env python3
"""Archive or safely prune the bounded exact-stage Qwen candidate frontier.

Registration is non-destructive. Dominated retained bundles become
``pending_prune`` only. Conservative v1 implements no recursive deletion;
``--apply-prune`` fails closed. Source runs, the canonical blank, and the
compiler cache are protected inputs, never deletion targets.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from neural_forge.hf_exact_candidate_frontier import (
    ExactCandidateFrontier,
    ExactCandidateFrontierError,
    authenticate_terminal_candidate,
    artifact_tree,
)


def _load_exact_stage_runner() -> Any:
    path = Path(__file__).resolve().with_name("run_qwen_exact_target_native_stage.py")
    spec = importlib.util.spec_from_file_location(
        "neural_forge_exact_stage_runner_for_retention", path
    )
    if spec is None or spec.loader is None:
        raise ExactCandidateFrontierError(
            "Cannot load the canonical exact-stage runner for factor authentication."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _source_pinned_factor_authenticator(
    factor_path: Path, lexical_rank: int | None
) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - production dependency
        raise ExactCandidateFrontierError(
            "Torch is required to authenticate a retained factor pack."
        ) from error
    runner = _load_exact_stage_runner()
    try:
        envelope = torch.load(factor_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ExactCandidateFrontierError(
            "Cannot safely load the retained factor pack."
        ) from error
    (
        _factor_payload,
        factor_state_sha256,
        envelope_provenance,
        factor_pack_schema,
        trainer_factor_pack_schema,
        authenticated_lexical_rank,
    ) = runner._authenticate_continuation_factor_payload(
        envelope,
        lexical_rank=lexical_rank,
    )
    return {
        "factor_state_sha256": factor_state_sha256,
        "envelope_provenance": dict(envelope_provenance),
        "factor_pack_schema": factor_pack_schema,
        "trainer_factor_pack_schema": trainer_factor_pack_schema,
        "lexical_rank": authenticated_lexical_rank,
        "canonical_runner_path": str(Path(runner.__file__).resolve()),
    }


def _canonical_preregistration_authenticator(
    preregistration_path: Path, preregistration_sha256: str
) -> Mapping[str, Any]:
    runner = _load_exact_stage_runner()
    contract = runner.authenticate_preregistration(
        preregistration_path, preregistration_sha256
    )
    return contract.preflight_report()


def _independent_accepted_child_authenticator(
    child_path: Path, exact_unique_parameters: int
) -> Mapping[str, Any]:
    """Freshly load and independently certify an already accepted child."""

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:  # pragma: no cover - production dependency
        raise ExactCandidateFrontierError(
            "Torch and Transformers are required for independent child reload."
        ) from error
    runner = _load_exact_stage_runner()
    reloaded = AutoModelForCausalLM.from_pretrained(
        child_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float16,
        device_map="cpu",
        attn_implementation=runner.CANONICAL_ATTENTION_IMPLEMENTATION,
    ).eval()
    try:
        stage = runner.default_qwen4b_progressive_target_plan().stages[0]
        certificate = runner.certify_native_dense_gqa_target(
            reloaded,
            exact_unique_parameters=exact_unique_parameters,
            stage=stage,
            require_dense_swiglu=True,
        )
        banned_modules = ("ResidualProjection", "ResidualRegistry", "Adapter")
        if any(
            any(word in type(module).__name__ for word in banned_modules)
            for module in reloaded.modules()
        ):
            raise ExactCandidateFrontierError(
                "Independently loaded child retains an adapter/registry wrapper."
            )
        banned_names = ("adapter", "factor", "optimizer", "parent", "sidecar")
        if any(
            any(word in path.name.lower() for word in banned_names)
            for path in child_path.rglob("*")
            if path.is_file()
        ):
            raise ExactCandidateFrontierError(
                "Independently loaded child contains a sidecar-like file."
            )
        tree = artifact_tree(child_path)
        return {
            "fresh_reload_performed": True,
            "loaded_from_candidate_path": True,
            "exact_unique_parameters": certificate.exact_unique_parameters,
            "state_dict_sha256": runner._state_dict_sha256(torch, reloaded),
            "artifact_tree": {
                key: tree[key] for key in ("file_count", "total_bytes", "tree_sha256")
            },
            "native_certificate": certificate.as_dict(),
            "published_adapter_parameter_count": 0,
            "parent_runtime_reference_count": 0,
            "source_sized_sidecar_parameter_count": 0,
        }
    finally:
        del reloaded
        gc.collect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frontier-root",
        type=Path,
        required=True,
        help="Dedicated retention root; it must not overlap the blank or cache.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="One completed exact-stage run to authenticate and register.",
    )
    parser.add_argument(
        "--run-report-sha256",
        help="Caller-provided external SHA-256 root for run_report.json.",
    )
    parser.add_argument(
        "--preregistration-sha256",
        help="Caller-provided external SHA-256 root for the preregistration.",
    )
    parser.add_argument(
        "--max-per-contract",
        type=int,
        default=3,
        help="Bound per comparable evaluation contract (maximum: 3).",
    )
    parser.add_argument(
        "--protected-root",
        type=Path,
        action="append",
        default=[],
        help="Additional immutable root; may be supplied more than once.",
    )
    parser.add_argument(
        "--apply-prune",
        action="store_true",
        help="Unsupported in conservative v1; actual deletion is disabled.",
    )
    parser.add_argument(
        "--prune-only",
        action="store_true",
        help="Unsupported in conservative v1; plans require manual review.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Read and authenticate the frontier without registering or deleting.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = sum(
        bool(value)
        for value in (
            args.run_root is not None,
            args.prune_only,
            args.inspect,
        )
    )
    if selected != 1:
        raise SystemExit(
            "Choose exactly one of --run-root, --prune-only, or --inspect."
        )
    if args.prune_only and not args.apply_prune:
        raise SystemExit("Actual deletion is unsupported in conservative v1.")
    if args.inspect and args.apply_prune:
        raise SystemExit("--inspect cannot be combined with --apply-prune.")
    if args.apply_prune or args.prune_only:
        raise SystemExit(
            "Actual deletion is unsupported in conservative v1; the manifest only "
            "emits a durable pending-prune plan."
        )
    if args.run_root is not None and (
        args.run_report_sha256 is None or args.preregistration_sha256 is None
    ):
        raise SystemExit(
            "--run-root requires external --run-report-sha256 and "
            "--preregistration-sha256 roots."
        )
    frontier = ExactCandidateFrontier(
        args.frontier_root,
        max_per_contract=args.max_per_contract,
        protected_roots=args.protected_root,
    )
    if args.inspect:
        result = frontier.inspect()
    else:
        candidate = authenticate_terminal_candidate(
            args.run_root,
            expected_run_report_sha256=args.run_report_sha256,
            expected_preregistration_sha256=args.preregistration_sha256,
            preregistration_authenticator=(_canonical_preregistration_authenticator),
            factor_authenticator=_source_pinned_factor_authenticator,
            accepted_child_authenticator=(_independent_accepted_child_authenticator),
        )
        result = frontier.register(candidate)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
