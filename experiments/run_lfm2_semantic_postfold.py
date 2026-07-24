#!/usr/bin/env python3
"""Run the saved-child, no-training LFM2 semantic postfold transaction."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
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

from experiments.run_lfm2_residual_attention_fold import (  # noqa: E402
    load_local_tokenizer,
)
from neural_forge.hf_lfm2_semantic_postfold import (  # noqa: E402
    Lfm2SemanticPostfoldBlockedError,
    SEMANTIC_ATTENTION_INSTALLER_CALLABLE,
    SEMANTIC_LAYER_STACK_INSTALLER_CALLABLE,
    TIED_RESIDUAL_CONGRUENCE_MODES,
    TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
    run_lfm2_semantic_postfold,
)
from neural_forge.hf_lfm2_semantic_body_recipe import (  # noqa: E402
    LFM2_SEMANTIC_BODY_RECIPE_IDS,
)
from neural_forge.hf_lfm2_producer_aware_semantic_integration import (  # noqa: E402
    Lfm2ProducerAwareCorrectionScalePolicy,
)


def _new_file(value: str, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise ValueError(f"{name} already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError(f"{name} parent must be a real existing directory.")
    return path


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_cli_path_disjointness(args: argparse.Namespace) -> tuple[Path, Path]:
    """Reject write/read ancestry collisions before loading either tokenizer."""

    output = Path(args.output_checkpoint).expanduser().resolve()
    receipt = Path(args.receipt_json).expanduser().resolve()
    capsule = (
        None
        if args.preterminal_capsule is None
        else Path(args.preterminal_capsule).expanduser().resolve()
    )
    readonly = {
        "input checkpoint": Path(args.input_checkpoint).expanduser().resolve(),
        "trace cache": Path(args.trace_cache).expanduser().resolve(),
        "tokenizer bridge": Path(args.bridge_directory).expanduser().resolve(),
        "donor tokenizer": Path(args.donor_tokenizer_directory).expanduser().resolve(),
        "target tokenizer": Path(args.target_tokenizer_directory)
        .expanduser()
        .resolve(),
        "parent checkpoint": Path(args.parent_model_directory).expanduser().resolve(),
        "compiler diagnostic": Path(args.compiler_diagnostic_json)
        .expanduser()
        .resolve(),
    }
    if args.interface_statistics_directory is not None:
        readonly["interface statistics"] = (
            Path(args.interface_statistics_directory).expanduser().resolve()
        )
    if args.interface_frontier_companion_json is not None:
        readonly["interface frontier companion"] = (
            Path(args.interface_frontier_companion_json).expanduser().resolve()
        )
    writes = {"final output": output, "receipt JSON": receipt}
    if capsule is not None:
        writes["preterminal capsule"] = capsule
    write_items = tuple(writes.items())
    collisions = []
    for index, (left_label, left) in enumerate(write_items):
        for right_label, right in write_items[index + 1 :]:
            if _paths_overlap(left, right):
                collisions.append(f"{left_label}/{right_label}")
        for right_label, right in readonly.items():
            if _paths_overlap(left, right):
                collisions.append(f"{left_label}/{right_label}")
    if collisions:
        raise ValueError(
            "Output, capsule, receipt, and read-only artifact paths must be "
            f"disjoint; overlaps {', '.join(collisions)}."
        )
    return output, receipt


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Publish with create-if-absent semantics so a concurrent receipt can
        # never be replaced after the preflight existence check.
        os.link(temporary, path, follow_symlinks=False)
    finally:
        if temporary.exists():
            temporary.unlink()


def _installer(module_name: str | None, callable_name: str) -> Any | None:
    if module_name is None:
        return None
    module = importlib.import_module(module_name)
    value = getattr(module, callable_name, None)
    if not callable(value):
        raise RuntimeError(f"{module_name}.{callable_name} is missing.")
    return value


def _correction_scale_frontier(value: str) -> tuple[float, ...]:
    try:
        frontier = tuple(float(item.strip()) for item in value.split(","))
        return Lfm2ProducerAwareCorrectionScalePolicy(
            materialization_frontier=frontier,
            materialization_scale=frontier[0],
        ).materialization_frontier
    except (TypeError, ValueError, RuntimeError, IndexError) as exc:
        raise argparse.ArgumentTypeError(
            "correction-scale frontier must be sorted unique nonnegative CSV floats"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a complete causal semantic layer-stack refold and one fused "
            "final-MLP/terminal/lexical W2 solve to a saved standalone CPCF-v5 "
            "LFM2 child without training."
        )
    )
    parser.add_argument("--input-checkpoint", required=True)
    parser.add_argument("--expected-input-artifact-sha256", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument(
        "--preterminal-capsule",
        help=(
            "Optional new directory for an authenticated resume capsule saved "
            "after the causal layer stack and before fused terminal compilation."
        ),
    )
    parser.add_argument("--receipt-json", required=True)
    parser.add_argument(
        "--trace-cache",
        required=True,
        help=(
            "Complete semantic-evidence cache. The distinct c61 compiler cache "
            "is reopened from the pinned compiler diagnostic."
        ),
    )
    parser.add_argument(
        "--expected-cache-key",
        required=True,
        help="Externally pinned identity of the semantic-evidence cache.",
    )
    parser.add_argument("--bridge-directory", required=True)
    parser.add_argument("--expected-bridge-artifact-sha256", required=True)
    parser.add_argument("--donor-tokenizer-directory", required=True)
    parser.add_argument("--target-tokenizer-directory", required=True)
    parser.add_argument("--parent-model-directory", required=True)
    parser.add_argument("--compiler-diagnostic-json", required=True)
    parser.add_argument(
        "--expected-compiler-diagnostic-sha256",
        required=True,
    )
    parser.add_argument(
        "--interface-statistics-directory",
        help=(
            "Optional authenticated compact interface-statistics artifact for "
            "semantic-foundation ancestry. Must be paired with its SHA-256 pin."
        ),
    )
    parser.add_argument(
        "--expected-interface-statistics-sha256",
        help=(
            "External artifact identity pin paired with "
            "--interface-statistics-directory."
        ),
    )
    parser.add_argument(
        "--interface-frontier-companion-json",
        help=(
            "Canonical persisted frontier-migration companion. Requires both "
            "external SHA pins and --anchor-candidate-index."
        ),
    )
    parser.add_argument(
        "--expected-interface-frontier-companion-file-sha256",
        help="External SHA-256 pin for the exact canonical companion file bytes.",
    )
    parser.add_argument(
        "--expected-interface-frontier-companion-sha256",
        help="External pin for the companion receipt's canonical body seal.",
    )
    parser.add_argument(
        "--anchor-candidate-index",
        type=int,
        help=(
            "One lazy anchor candidate to materialize after authenticating the "
            "saved compatibility-default child."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--fold-compute-device",
        default="cpu",
        help=(
            "Closed-form folding backend. Use cpu or an explicitly indexed "
            "CUDA device such as cuda:0; arithmetic remains float64."
        ),
    )
    parser.add_argument(
        "--global-residual-closure-version",
        choices=("producer_aware_v2a", "output_writer_v1_diagnostic"),
        default="producer_aware_v2a",
    )
    parser.add_argument(
        "--correction-scale-frontier",
        type=_correction_scale_frontier,
        default=(1.0,),
    )
    parser.add_argument("--correction-scale", type=float, default=1.0)
    parser.add_argument(
        "--global-residual-backend",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--target-dtype-representability-level",
        choices=(
            "requested",
            "first_effectful",
            "mass_0p01_percent",
            "mass_0p1_percent",
            "mass_1_percent",
            "mass_10_percent",
            "mass_50_percent",
        ),
        default="first_effectful",
    )
    parser.add_argument(
        "--body-recipe",
        choices=LFM2_SEMANTIC_BODY_RECIPE_IDS,
        default="fit_guided",
        help=(
            "Authenticated complete-body folding lane. Run the same sealed "
            "foundation once per recipe to lazily retain every body candidate "
            "for decoded manual review; metrics never authorize promotion."
        ),
    )
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument(
        "--parent-final-norm-tensor-key",
        default="model.embedding_norm.weight",
    )
    parser.add_argument("--parent-norm-epsilon-config-key", default="norm_eps")
    parser.add_argument(
        "--parent-tied-tensor-key",
        default="model.embed_tokens.weight",
    )
    parser.add_argument(
        "--tied-interface-mode",
        choices=TIED_RESIDUAL_CONGRUENCE_MODES,
        default=TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
        help=(
            "Select the historical self-dual output-factor control or install "
            "the strong mapped-parent input/residual/output congruence."
        ),
    )
    parser.add_argument(
        "--attention-installer-module",
        help=(
            "Module exporting the exact callable "
            f"{SEMANTIC_ATTENTION_INSTALLER_CALLABLE}; omitted means use the "
            "canonical built-in resolver and fail closed if unavailable."
        ),
    )
    parser.add_argument(
        "--semantic-layer-stack-compiler-module",
        help=(
            "Module exporting the complete causal callable "
            f"{SEMANTIC_LAYER_STACK_INSTALLER_CALLABLE}; omitted means use "
            "the canonical resolver and fail closed if unavailable."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    statistics_requested = args.interface_statistics_directory is not None
    if statistics_requested != (args.expected_interface_statistics_sha256 is not None):
        raise ValueError(
            "--interface-statistics-directory and "
            "--expected-interface-statistics-sha256 must be supplied together."
        )
    companion_values = (
        args.interface_frontier_companion_json,
        args.expected_interface_frontier_companion_file_sha256,
        args.expected_interface_frontier_companion_sha256,
        args.anchor_candidate_index,
    )
    companion_requested = any(value is not None for value in companion_values)
    if companion_requested and any(value is None for value in companion_values):
        raise ValueError(
            "--interface-frontier-companion-json, its file and canonical SHA-256 "
            "pins, and --anchor-candidate-index must be supplied together."
        )
    if companion_requested and not statistics_requested:
        raise ValueError(
            "A frontier companion requires --interface-statistics-directory and "
            "its pin."
        )
    if args.anchor_candidate_index is not None and args.anchor_candidate_index < 0:
        raise ValueError("--anchor-candidate-index must be nonnegative.")
    output, receipt_path = _validate_cli_path_disjointness(args)
    receipt = _new_file(str(receipt_path), "receipt JSON")
    if output.exists() or output.is_symlink():
        raise ValueError(f"output checkpoint already exists: {output}")
    donor = load_local_tokenizer(args.donor_tokenizer_directory)
    target = load_local_tokenizer(args.target_tokenizer_directory)
    try:
        result = run_lfm2_semantic_postfold(
            input_checkpoint_directory=args.input_checkpoint,
            expected_input_artifact_sha256=(args.expected_input_artifact_sha256),
            output_checkpoint_directory=output,
            cache_directory=args.trace_cache,
            expected_cache_key=args.expected_cache_key,
            bridge_directory=args.bridge_directory,
            expected_bridge_artifact_sha256=(args.expected_bridge_artifact_sha256),
            donor_tokenizer=donor,
            target_tokenizer=target,
            parent_model_directory=args.parent_model_directory,
            compiler_diagnostic_json=args.compiler_diagnostic_json,
            expected_compiler_diagnostic_sha256=(
                args.expected_compiler_diagnostic_sha256
            ),
            device=args.device,
            dtype=args.dtype,
            fold_compute_device=args.fold_compute_device,
            global_residual_closure_version=args.global_residual_closure_version,
            correction_scale_frontier=args.correction_scale_frontier,
            correction_scale=args.correction_scale,
            global_residual_backend=args.global_residual_backend,
            target_dtype_representability_level=(
                args.target_dtype_representability_level
            ),
            body_recipe=args.body_recipe,
            parent_final_norm_tensor_key=(args.parent_final_norm_tensor_key),
            parent_norm_epsilon_config_key=(args.parent_norm_epsilon_config_key),
            parent_tied_tensor_key=args.parent_tied_tensor_key,
            tied_interface_mode=args.tied_interface_mode,
            **(
                {
                    "interface_statistics_directory": (
                        args.interface_statistics_directory
                    ),
                    "expected_interface_statistics_sha256": (
                        args.expected_interface_statistics_sha256
                    ),
                    **(
                        {
                            "interface_frontier_companion_json": (
                                args.interface_frontier_companion_json
                            ),
                            "expected_interface_frontier_companion_file_sha256": (
                                args.expected_interface_frontier_companion_file_sha256
                            ),
                            "expected_interface_frontier_companion_sha256": (
                                args.expected_interface_frontier_companion_sha256
                            ),
                            "anchor_candidate_index": args.anchor_candidate_index,
                        }
                        if companion_requested
                        else {}
                    ),
                }
                if statistics_requested
                else {}
            ),
            attention_installer=_installer(
                args.attention_installer_module,
                SEMANTIC_ATTENTION_INSTALLER_CALLABLE,
            ),
            semantic_layer_stack_compiler=_installer(
                args.semantic_layer_stack_compiler_module,
                SEMANTIC_LAYER_STACK_INSTALLER_CALLABLE,
            ),
            preterminal_capsule_directory=args.preterminal_capsule,
        )
    except Lfm2SemanticPostfoldBlockedError as exc:
        _atomic_json(receipt, dict(exc.report))
        print(json.dumps(dict(exc.report), indent=2, sort_keys=True))
        return 2
    _atomic_json(receipt, dict(result))
    print(json.dumps(dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
