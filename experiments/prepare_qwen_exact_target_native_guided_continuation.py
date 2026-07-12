#!/usr/bin/env python3
"""Prepare the model-free authenticated guided-QKVO continuation contract.

The builder consumes the exact terminal lexical run and its passed
output-Jacobian audit.  It does not restart compilation, synthesize a target,
load model weights, run a forward pass, train, or write a checkpoint.  The
prior 120-prompt fit and exposed 24-prompt candidate become a 144-prompt fit;
a deterministic fresh 24-prompt candidate is selected only after excluding
every historical prompt/prefix; the exact untouched 32-prompt sealed artifact
is reused by path and hash.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

import prepare_qwen_exact_target_native_continuation_splits as split_tools
import run_qwen_exact_target_native_guided_continuation as guided
import run_qwen_exact_target_native_stage as stage
from replay_qwen_unified_v8_commit import _artifact_tree, _publish_staged_checkpoint


BUILD_SCHEMA = "neural-forge-qwen-exact-target-native-guided-build-v1"
SPLIT_MANIFEST_SCHEMA = "neural-forge-guided-continuation-split-manifest-v1"
SPLIT_SELECTION_SCHEMA = "neural-forge-guided-continuation-selection-v1"
HISTORICAL_AUDIT_COMMIT = "e57d0b21da2c5d9d815375b35acfcb9305b12259"
DEFAULT_AUDIT_RUN_REPORT = Path(
    "/mnt/ai_data/neural-forge/runs/"
    "qwen_exact_target_native_stage1_step24_output_jacobian_audit_20260712/"
    "run_report.json"
)
DEFAULT_AUDIT_PREREGISTRATION = Path(
    "/mnt/ai_data/neural-forge/prereg/"
    "qwen_exact_target_native_stage1_step24_output_jacobian_audit_20260712/"
    "preregistration.json"
)
DEFAULT_OUTPUT = Path(
    "/mnt/ai_data/neural-forge/prereg/"
    "qwen_exact_target_native_stage1_step24_guided_qkvo_continuation_20260712"
)


class GuidedPreparationError(RuntimeError):
    """The immutable audit/source/split chain cannot be reproduced."""


def _remove_owned_staging(
    staging: Path,
    *,
    output: Path,
    expected_tree: Mapping[str, Any] | None = None,
) -> bool:
    """Remove only this builder's exact staging tree, optionally tree-gated."""

    if not staging.exists() and not staging.is_symlink():
        return False
    if (
        staging.parent != output.parent
        or not staging.name.startswith(f".{output.name}.")
        or not staging.name.endswith(".staging")
        or staging.is_symlink()
        or not staging.is_dir()
        or staging.resolve() != staging
    ):
        raise GuidedPreparationError("Refusing to remove an unowned staging path.")
    if expected_tree is not None and _artifact_tree(staging) != dict(expected_tree):
        raise GuidedPreparationError(
            "Refusing to remove fallback staging because its tree changed."
        )
    shutil.rmtree(staging)
    if staging.exists() or staging.is_symlink():
        raise GuidedPreparationError("Owned staging cleanup did not complete.")
    return True


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _prompt_records(binding: stage.SplitBinding) -> tuple[Any, ...]:
    prompts = binding.payload.get("prompts")
    if not isinstance(prompts, list):
        raise GuidedPreparationError(f"Source {binding.name} prompts are malformed.")
    try:
        return tuple(
            split_tools._prompt_record(value, f"source.{binding.name}.prompts[{index}]")
            for index, value in enumerate(prompts)
        )
    except split_tools.ContinuationSplitPreparationError as error:
        raise GuidedPreparationError(
            "Source behavior prompt failed authentication."
        ) from error


def _implementation_pins() -> dict[str, Any]:
    builder = Path(__file__).resolve()
    runner = builder.with_name("run_qwen_exact_target_native_guided_continuation.py")
    return {
        "runner_sha256": stage._file_sha256(runner),
        "builder_sha256": stage._file_sha256(builder),
        "base_runner_sha256": stage._file_sha256(Path(stage.__file__).resolve()),
        "trainer_module": guided.GUIDED_TRAINER_MODULE,
        "trainer_sha256": guided._module_sha256(guided.GUIDED_TRAINER_MODULE),
        "guided_adapter_sha256": guided._module_sha256(guided.GUIDED_ADAPTER_MODULE),
        "progressive_bridge_sha256": guided._module_sha256(
            "neural_forge.hf_progressive_target_bridge"
        ),
        "compiled_child_cache_sha256": guided._module_sha256(
            "neural_forge.hf_compiled_child_cache"
        ),
    }


def _source_and_audit_rows(
    *,
    audit_run_report: Path,
    audit_preregistration: Path,
    historical_git_revision: str,
) -> tuple[dict[str, Any], dict[str, Any], stage.AuthenticatedPreRegistration]:
    run_path = audit_run_report.expanduser().resolve()
    prereg_path = audit_preregistration.expanduser().resolve()
    if not run_path.is_file() or not prereg_path.is_file():
        raise GuidedPreparationError("Audit report/preregistration is absent.")
    run_sha = stage._file_sha256(run_path)
    prereg_sha = stage._file_sha256(prereg_path)
    report = stage._strict_json(run_path)
    if not isinstance(report, Mapping):
        raise GuidedPreparationError("Audit run report is malformed.")
    historical = report.get("implementation_authentication")
    try:
        contract = stage.authenticate_preregistration(
            prereg_path,
            prereg_sha,
            historical_git_revision=historical_git_revision,
            historical_implementation_authentication=historical,
        )
    except stage.ExactTargetNativeStageError as error:
        raise GuidedPreparationError(
            "Historical audit preregistration/source blobs do not authenticate."
        ) from error
    continuation = contract.continuation
    cache = contract.compiler_cache
    if continuation is None or cache is None or cache.require_hit is not True:
        raise GuidedPreparationError("Audit source lacks its v2/cache continuation.")
    basis_binding = report.get("guided_basis_pack")
    audit_report = report.get("output_jacobian_audit_report")
    if not isinstance(basis_binding, Mapping) or not isinstance(audit_report, Mapping):
        raise GuidedPreparationError("Audit report omits its basis evidence.")
    basis_path = Path(str(basis_binding.get("path", ""))).expanduser().resolve()
    basis_file_sha = stage._file_sha256(basis_path)
    if basis_binding.get("sha256") != basis_file_sha:
        raise GuidedPreparationError("Guided basis envelope file SHA changed.")
    try:
        envelope = torch.load(basis_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise GuidedPreparationError(
            "Guided basis envelope cannot be read safely."
        ) from error
    basis = envelope.get("basis_pack") if isinstance(envelope, Mapping) else None
    if not isinstance(basis, Mapping):
        raise GuidedPreparationError("Guided basis payload is absent.")
    factors = basis.get("factors")
    if not isinstance(factors, list):
        raise GuidedPreparationError("Guided basis tensor manifest is absent.")
    names = {
        "q_orthogonal_right": "q",
        "k_orthogonal_right": "k",
        "v_orthogonal_right": "v",
        "o_orthogonal_right": "o",
    }
    tensor_hashes = {
        names[str(row.get("name"))]: stage._require_sha256(
            row.get("tensor_sha256"), f"guided_basis.{row.get('name')}.sha256"
        )
        for row in factors
        if isinstance(row, Mapping) and str(row.get("name")) in names
    }
    parent_tree = contract.artifact_authentication["parent"].get("tree")
    target_tree = contract.artifact_authentication["blank_target"].get("tree")
    if not isinstance(parent_tree, Mapping) or not isinstance(target_tree, Mapping):
        raise GuidedPreparationError("Source artifact trees are incomplete.")
    source = {
        "run_report": {
            "path": str(continuation.source_run_report_path),
            "sha256": continuation.source_run_report_sha256,
        },
        "factor_pack": {
            "path": str(continuation.factor_pack_path),
            "sha256": continuation.factor_pack_sha256,
        },
        "factor_state_sha256": continuation.factor_state_sha256,
        "source_preregistration_sha256": (continuation.source_preregistration_sha256),
        "source_evaluation_contract_sha256": (
            continuation.source_evaluation_contract_sha256
        ),
        "compiled_state_sha256": continuation.compiled_state_sha256,
        "compiler_report_sha256": continuation.compiler_report_sha256,
        "parent_tree_sha256": parent_tree["tree_sha256"],
        "blank_target_tree_sha256": target_tree["tree_sha256"],
        "compiler_cache": {
            "compiler_identity_sha256": cache.expected_compiler_identity_sha256,
            "receipt_sha256": cache.expected_receipt_sha256,
            "compiler_report_sha256": cache.expected_compiler_report_sha256,
            "state_dict_sha256": cache.expected_state_dict_sha256,
            "artifact_tree_sha256": cache.expected_artifact_tree_sha256,
        },
    }
    audit = {
        "preregistration": {"path": str(prereg_path), "sha256": prereg_sha},
        "historical_git_revision": historical_git_revision,
        "run_report": {"path": str(run_path), "sha256": run_sha},
        "guided_basis_pack": {
            "path": str(basis_path),
            "sha256": basis_file_sha,
        },
        "basis_envelope_file_sha256": basis_file_sha,
        "basis_domain_artifact_sha256": stage._require_sha256(
            audit_report.get("basis_artifact_sha256"),
            "audit_report.basis_artifact_sha256",
        ),
        "basis_payload_sha256": guided._guided_basis_payload_sha256(basis),
        "audit_report_self_sha256": stage._require_sha256(
            audit_report.get("report_sha256"), "audit_report.report_sha256"
        ),
        "audit_report_canonical_sha256": stage._canonical_sha256(audit_report),
        "basis_manifest_sha256": stage._require_sha256(
            audit_report.get("basis_manifest_sha256"),
            "audit_report.basis_manifest_sha256",
        ),
        "authorized_roles": list(guided.GUIDED_ROLES),
        "basis_tensor_sha256s": tensor_hashes,
    }
    # Reuse the production persisted-audit verifier before selecting any new data.
    try:
        authenticated_contract, _artifacts = guided._authenticate_persisted_audit(
            audit, source, base=run_path.parent
        )
    except guided.GuidedContinuationError as error:
        raise GuidedPreparationError(
            "Persisted audit evidence failed replay."
        ) from error
    return source, audit, authenticated_contract


def _prepare_splits(
    *,
    staging: Path,
    source_contract: stage.AuthenticatedPreRegistration,
    corpus_path: Path,
    tokenizer_path: Path,
    tokenizer_loader: Callable[[Path], Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if {
        name: len(binding.row_ids) for name, binding in source_contract.splits.items()
    } != {
        "fit": 120,
        "candidate": 24,
        "sealed": 32,
    }:
        raise GuidedPreparationError("Audit behavior source is not exact 120/24/32.")
    source = {
        name: _prompt_records(source_contract.splits[name])
        for name in ("fit", "candidate", "sealed")
    }
    try:
        split_tools._assert_globally_disjoint(source, label="guided historical source")
        corpus = split_tools._load_corpus(
            corpus_path.expanduser().resolve(),
            expected_sha256=split_tools.PINNED_CORPUS_SHA256,
            expected_line_count=split_tools.PINNED_CORPUS_LINE_COUNT,
        )
        tokenizer_root = tokenizer_path.expanduser().resolve()
        tokenizer_json = tokenizer_root / "tokenizer.json"
        if (
            not tokenizer_json.is_file()
            or split_tools._file_sha256(tokenizer_json)
            != split_tools.PINNED_TOKENIZER_JSON_SHA256
        ):
            raise GuidedPreparationError("Pinned tokenizer JSON changed.")
        tokenizer = tokenizer_loader(tokenizer_root)
        fresh_candidate, selection_stats = split_tools._select_fresh_candidate(
            corpus,
            tokenizer,
            excluded_prompts=tuple(
                prompt for values in source.values() for prompt in values
            ),
            max_prompt_tokens=split_tools.MAX_PROMPT_TOKENS,
            expected_count=24,
        )
    except split_tools.ContinuationSplitPreparationError as error:
        raise GuidedPreparationError(
            "Fresh guided candidate selection failed."
        ) from error

    output_records = {
        "fit": (*source["fit"], *source["candidate"]),
        "candidate": tuple(fresh_candidate),
        "sealed": source["sealed"],
    }
    if {name: len(values) for name, values in output_records.items()} != (
        guided.GUIDED_SPLIT_COUNTS
    ):
        raise GuidedPreparationError("Guided split output is not exact 144/24/32.")
    try:
        split_tools._assert_globally_disjoint(
            output_records, label="guided active contract"
        )
    except split_tools.ContinuationSplitPreparationError as error:
        raise GuidedPreparationError("Guided active splits overlap.") from error
    source_hashes = {
        name: source_contract.splits[name].sha256
        for name in ("fit", "candidate", "sealed")
    }
    fit_payloads = [dict(value.payload) for value in output_records["fit"]]
    candidate_payloads = [dict(value.payload) for value in fresh_candidate]
    component_hashes = {
        "prior_fit": split_tools._component_sha256(
            [dict(value.payload) for value in source["fit"]]
        ),
        "exposed_candidate_to_fit": split_tools._component_sha256(
            [dict(value.payload) for value in source["candidate"]]
        ),
        "fresh_candidate": split_tools._component_sha256(candidate_payloads),
        "reused_sealed": split_tools._component_sha256(
            [dict(value.payload) for value in source["sealed"]]
        ),
    }
    payloads = {
        "fit": split_tools._split_payload(
            "fit",
            fit_payloads,
            original_hashes=source_hashes,
            corpus_sha256=split_tools.PINNED_CORPUS_SHA256,
            corpus_line_count=split_tools.PINNED_CORPUS_LINE_COUNT,
            tokenizer_sha256=split_tools.PINNED_TOKENIZER_JSON_SHA256,
            components=(
                {
                    "name": "prior-fit-training-replay",
                    "count": 120,
                    "prompt_manifest_sha256": component_hashes["prior_fit"],
                },
                {
                    "name": "exposed-prior-candidate-training-only",
                    "count": 24,
                    "prompt_manifest_sha256": component_hashes[
                        "exposed_candidate_to_fit"
                    ],
                },
            ),
            selection_schema=SPLIT_SELECTION_SCHEMA,
        ),
        "candidate": split_tools._split_payload(
            "candidate",
            candidate_payloads,
            original_hashes=source_hashes,
            corpus_sha256=split_tools.PINNED_CORPUS_SHA256,
            corpus_line_count=split_tools.PINNED_CORPUS_LINE_COUNT,
            tokenizer_sha256=split_tools.PINNED_TOKENIZER_JSON_SHA256,
            components=(
                {
                    "name": "fresh-guided-candidate",
                    "count": 24,
                    "prompt_manifest_sha256": component_hashes["fresh_candidate"],
                },
            ),
            selection_schema=SPLIT_SELECTION_SCHEMA,
        ),
    }
    split_dir = staging / "splits"
    split_dir.mkdir()
    for name, payload in payloads.items():
        _write_json_exclusive(split_dir / f"{name}.json", payload)
    manifest = {
        "schema": SPLIT_MANIFEST_SCHEMA,
        "source_split_sha256s": source_hashes,
        "counts": {
            "source": {"fit": 120, "candidate": 24, "sealed": 32},
            "active": dict(guided.GUIDED_SPLIT_COUNTS),
        },
        "components": component_hashes,
        "selection": {
            "schema": SPLIT_SELECTION_SCHEMA,
            "global_order": "source_prompt_sha256, then source_line_number",
            "minimum_meaningful_prefix_tokens": 8,
            "statistics": selection_stats,
        },
        "contract": {
            "fit": "prior fit 120 + exposed candidate 24",
            "candidate": "24 entirely fresh prompts",
            "sealed": "exact prior locked 32-prompt artifact reused by path/hash",
            "audit_basis_candidate_eligible_only_for_fit": True,
            "fresh_candidate_excludes_all_historical_prompts_and_prefixes": True,
            "sealed_copied_or_rewritten": False,
        },
        "model_access": {
            "model_weights_loaded": False,
            "model_forward_calls": 0,
            "tokenizer_loaded": True,
        },
    }
    manifest["manifest_content_sha256"] = stage._canonical_sha256(manifest)
    _write_json_exclusive(staging / "guided_split_manifest.json", manifest)
    bindings = {
        "fit": {
            "path": "splits/fit.json",
            "sha256": stage._file_sha256(split_dir / "fit.json"),
            "expected_count": 144,
        },
        "candidate": {
            "path": "splits/candidate.json",
            "sha256": stage._file_sha256(split_dir / "candidate.json"),
            "expected_count": 24,
        },
        "sealed": {
            "path": str(source_contract.splits["sealed"].path),
            "sha256": source_contract.splits["sealed"].sha256,
            "expected_count": 32,
        },
    }
    return bindings, manifest


def _guided_training(source: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return guided.guided_training_schedule_from_source(source)
    except guided.GuidedContinuationError as error:
        raise GuidedPreparationError("Guided 64-step schedule is invalid.") from error


def prepare_guided_continuation(
    *,
    output_dir: Path,
    audit_run_report: Path = DEFAULT_AUDIT_RUN_REPORT,
    audit_preregistration: Path = DEFAULT_AUDIT_PREREGISTRATION,
    historical_git_revision: str = HISTORICAL_AUDIT_COMMIT,
    corpus_path: Path = split_tools.DEFAULT_CORPUS,
    tokenizer_loader: Callable[[Path], Any] = split_tools._load_tokenizer,
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise GuidedPreparationError("output_dir must not already exist.")
    source, audit, source_contract = _source_and_audit_rows(
        audit_run_report=audit_run_report,
        audit_preregistration=audit_preregistration,
        historical_git_revision=historical_git_revision,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".staging", dir=output.parent
        )
    )
    try:
        split_bindings, split_manifest = _prepare_splits(
            staging=staging,
            source_contract=source_contract,
            corpus_path=corpus_path,
            tokenizer_path=source_contract.parent.path,
            tokenizer_loader=tokenizer_loader,
        )
        thresholds = {
            name: source_contract.thresholds[name].as_dict()
            for name in ("fit", "candidate", "sealed")
        }
        thresholds["fit"].update(
            {"minimum_action_count": 1_152, "minimum_rollout_count": 144}
        )
        training = _guided_training(source_contract.training)
        implementation = _implementation_pins()
        runtime = {
            "dtype": "float16",
            "attention_implementation": stage.CANONICAL_ATTENTION_IMPLEMENTATION,
            "explicit_execute_required": True,
        }
        # Load only the tiny split JSON objects through the production parser so
        # the evaluation hash has one implementation on builder and runner paths.
        parsed_splits = {
            name: stage._load_split(split_bindings[name], name=name, base=staging)
            for name in ("fit", "candidate", "sealed")
        }
        threshold_objects = {
            name: stage.BehaviorThresholds.from_mapping(
                thresholds[name], f"thresholds.{name}"
            )
            for name in ("fit", "candidate", "sealed")
        }
        _source_contract, artifacts = guided._authenticate_persisted_audit(
            audit, source, base=staging
        )
        evaluation_sha = stage._canonical_sha256(
            guided._guided_evaluation_preimage(
                artifacts=artifacts,
                splits=parsed_splits,
                thresholds=threshold_objects,
                training=training,
                implementation=implementation,
                runtime=runtime,
                method_gate=guided.GUIDED_METHOD_GATE,
            )
        )
        preregistration = {
            "schema": guided.PREREGISTRATION_SCHEMA,
            "sealed_before_model_access": True,
            "seed": source_contract.seed,
            "implementation": implementation,
            "source": source,
            "audit": audit,
            "splits": split_bindings,
            "thresholds": thresholds,
            "training": training,
            "method_gate": dict(guided.GUIDED_METHOD_GATE),
            "runtime": runtime,
            "evaluation_contract_sha256": evaluation_sha,
        }
        prereg_path = staging / "preregistration.json"
        _write_json_exclusive(prereg_path, preregistration)
        prereg_sha = stage._file_sha256(prereg_path)
        try:
            contract = guided.authenticate_guided_preregistration(
                prereg_path, prereg_sha
            )
        except guided.GuidedContinuationError as error:
            raise GuidedPreparationError(
                "Production guided preflight rejected the built contract."
            ) from error
        report = {
            "schema": BUILD_SCHEMA,
            "mode": "authenticated-guided-qkvo-continuation",
            "preregistration": str(output / "preregistration.json"),
            "preregistration_sha256": prereg_sha,
            "evaluation_contract_sha256": contract.evaluation_contract_sha256,
            "historical_audit_commit": historical_git_revision,
            "source_run_report_sha256": artifacts.source_run_report_sha256,
            "audit_run_report_sha256": artifacts.audit_run_report_sha256,
            "audit_report_self_sha256": artifacts.audit_report_self_sha256,
            "audit_report_canonical_sha256": (artifacts.audit_report_canonical_sha256),
            "basis_envelope_file_sha256": artifacts.basis_envelope_file_sha256,
            "basis_domain_artifact_sha256": (artifacts.basis_domain_artifact_sha256),
            "basis_payload_sha256": artifacts.basis_payload_sha256,
            "basis_manifest_sha256": artifacts.basis_manifest_sha256,
            "split_manifest_sha256": stage._file_sha256(
                staging / "guided_split_manifest.json"
            ),
            "split_manifest_content_sha256": split_manifest["manifest_content_sha256"],
            "split_counts": dict(guided.GUIDED_SPLIT_COUNTS),
            "model_weights_loaded": False,
            "model_forward_calls": 0,
            "model_checkpoint_created": False,
            "training_launched": False,
            "ready_for_explicit_preflight": True,
        }
        _write_json_exclusive(staging / "build_report.json", report)
        if output.exists():
            raise GuidedPreparationError("output_dir appeared during preparation.")
        complete_tree = _artifact_tree(staging)
        publication = _publish_staged_checkpoint(
            staging,
            output,
            expected_tree=complete_tree,
        )
        if _artifact_tree(output) != complete_tree:
            raise GuidedPreparationError("Published preregistration tree changed.")
        if publication.get("mode") == "verified-copy-fallback":
            _remove_owned_staging(
                staging,
                output=output,
                expected_tree=complete_tree,
            )
        elif staging.exists() or staging.is_symlink():
            raise GuidedPreparationError(
                "Atomic preregistration publication unexpectedly retained staging."
            )
        report = {**report, "publication": publication}
        return report
    except BaseException:
        if staging.exists():
            _remove_owned_staging(staging, output=output)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--audit-run-report", type=Path, default=DEFAULT_AUDIT_RUN_REPORT
    )
    parser.add_argument(
        "--audit-preregistration", type=Path, default=DEFAULT_AUDIT_PREREGISTRATION
    )
    parser.add_argument("--historical-git-revision", default=HISTORICAL_AUDIT_COMMIT)
    parser.add_argument("--corpus", type=Path, default=split_tools.DEFAULT_CORPUS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_guided_continuation(
        output_dir=args.output_dir,
        audit_run_report=args.audit_run_report,
        audit_preregistration=args.audit_preregistration,
        historical_git_revision=args.historical_git_revision,
        corpus_path=args.corpus,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
