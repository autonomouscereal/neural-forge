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
    GUIDED_PREREGISTRATION_SCHEMA,
    GUIDED_WINNER_PUBLICATION_PREREGISTRATION_SCHEMA,
    authenticate_terminal_candidate,
    artifact_tree,
    strict_json,
)


GUIDED_FACTOR_ENVELOPE_SCHEMA = (
    "neural-forge-rejected-target-native-guided-factor-pack-v1"
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
    sibling_path = str(path.parent)
    inserted_sibling_path = sibling_path not in sys.path
    if inserted_sibling_path:
        sys.path.insert(0, sibling_path)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    finally:
        if inserted_sibling_path:
            sys.path.remove(sibling_path)
    return module


def _load_guided_runner() -> Any:
    path = (
        Path(__file__)
        .resolve()
        .with_name("run_qwen_exact_target_native_guided_continuation.py")
    )
    spec = importlib.util.spec_from_file_location(
        "neural_forge_guided_stage_runner_for_retention", path
    )
    if spec is None or spec.loader is None:
        raise ExactCandidateFrontierError(
            "Cannot load the canonical guided runner for retention authentication."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sibling_path = str(path.parent)
    inserted_sibling_path = sibling_path not in sys.path
    if inserted_sibling_path:
        sys.path.insert(0, sibling_path)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    finally:
        if inserted_sibling_path:
            sys.path.remove(sibling_path)
    return module


def _load_guided_winner_publication_runner() -> Any:
    path = (
        Path(__file__)
        .resolve()
        .with_name("run_qwen_exact_target_native_guided_winner_publication.py")
    )
    spec = importlib.util.spec_from_file_location(
        "neural_forge_guided_winner_publication_runner_for_retention", path
    )
    if spec is None or spec.loader is None:
        raise ExactCandidateFrontierError(
            "Cannot load the guided winner-publication runner for retention "
            "authentication."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sibling_path = str(path.parent)
    inserted_sibling_path = sibling_path not in sys.path
    if inserted_sibling_path:
        sys.path.insert(0, sibling_path)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    finally:
        if inserted_sibling_path:
            sys.path.remove(sibling_path)
    return module


def _guided_source_pinned_factor_authentication(
    envelope: Mapping[str, Any], lexical_rank: int | None, torch: Any
) -> Mapping[str, Any]:
    """Strictly authenticate the outer guided envelope and inner v3 pack."""

    guided = _load_guided_runner()
    if guided.FACTOR_ENVELOPE_SCHEMA != GUIDED_FACTOR_ENVELOPE_SCHEMA:
        raise ExactCandidateFrontierError(
            "Canonical guided factor-envelope schema changed."
        )
    expected_outer = {
        "schema",
        "reason",
        "preregistration_sha256",
        "evaluation_contract_sha256",
        "source_factor_state_sha256",
        "audit_report_self_sha256",
        "audit_report_canonical_sha256",
        "basis_envelope_file_sha256",
        "basis_domain_artifact_sha256",
        "basis_payload_sha256",
        "factor_pack",
        "contains_model_checkpoint",
        "contains_parent_parameters",
        "contains_activations_or_hidden_states",
    }
    if (
        set(envelope) != expected_outer
        or envelope.get("schema") != guided.FACTOR_ENVELOPE_SCHEMA
        or envelope.get("reason") != "selected"
        or any(
            envelope.get(name) is not False
            for name in (
                "contains_model_checkpoint",
                "contains_parent_parameters",
                "contains_activations_or_hidden_states",
            )
        )
    ):
        raise ExactCandidateFrontierError(
            "Guided factor envelope has an unsupported schema or fields."
        )
    for name in (
        "preregistration_sha256",
        "evaluation_contract_sha256",
        "source_factor_state_sha256",
        "audit_report_self_sha256",
        "audit_report_canonical_sha256",
        "basis_envelope_file_sha256",
        "basis_domain_artifact_sha256",
        "basis_payload_sha256",
    ):
        guided._require_sha256(envelope.get(name), f"guided_factor.{name}")
    payload = envelope.get("factor_pack")
    expected_payload = {
        "schema",
        "factor_count",
        "factors",
        "factor_state_sha256",
        "parent_state_included",
        "native_child_state_included",
        "optimizer_state_included",
        "factor_tensors",
        "lexical_rank",
        "guided_source_binding",
        "guided_source_binding_sha256",
        "inherited_product_regrouped",
        "guided_product_regrouped",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected_payload
        or payload.get("schema") != guided.TRAINER_FACTOR_PACK_V3_SCHEMA
        or payload.get("factor_count") != guided.GUIDED_TOTAL_FACTOR_COUNT
        or payload.get("lexical_rank") != lexical_rank
        or any(
            payload.get(name) is not False
            for name in (
                "parent_state_included",
                "native_child_state_included",
                "optimizer_state_included",
                "inherited_product_regrouped",
                "guided_product_regrouped",
            )
        )
    ):
        raise ExactCandidateFrontierError(
            "Guided factor payload is not the exact factor-only v3 artifact."
        )
    trainer = importlib.import_module(guided.GUIDED_TRAINER_MODULE)
    binding_type = getattr(trainer, "GuidedFactorSourceBinding", None)
    if not callable(binding_type):
        raise ExactCandidateFrontierError(
            "Canonical trainer lacks its guided source-binding authenticator."
        )
    raw_binding = payload.get("guided_source_binding")
    if not isinstance(raw_binding, Mapping):
        raise ExactCandidateFrontierError("Guided source binding is malformed.")
    try:
        source_binding = binding_type(**dict(raw_binding))
    except (TypeError, ValueError, RuntimeError) as error:
        raise ExactCandidateFrontierError(
            "Guided source binding failed canonical trainer authentication."
        ) from error
    if payload.get("guided_source_binding_sha256") != source_binding.binding_sha256:
        raise ExactCandidateFrontierError(
            "Guided source-binding SHA-256 does not authenticate."
        )
    if lexical_rank is None:
        raise ExactCandidateFrontierError(
            "The guided v3 pack requires the pinned lexical rank."
        )
    expected_shapes = dict(
        guided.stage._expected_continuation_factor_shapes(lexical_rank=lexical_rank)
    )
    sites = guided.stage.QKVO_SITE_COUNT
    rank = guided.stage.QKVO_RANK
    hidden = guided.stage.TARGET_HIDDEN_SIZE
    query_group = guided.stage.TARGET_QUERY_GROUP_WIDTH
    head = guided.stage.TARGET_HEAD_DIM
    expected_shapes.update(
        {
            "guided_qkvo.q_left": (sites, query_group, rank),
            "guided_qkvo.q_right": (sites, rank, hidden),
            "guided_qkvo.k_left": (sites, head, rank),
            "guided_qkvo.k_right": (sites, rank, hidden),
            "guided_qkvo.v_left": (sites, head, rank),
            "guided_qkvo.v_right": (sites, rank, hidden),
            "guided_qkvo.o_left": (sites, hidden, rank),
            "guided_qkvo.o_right": (sites, rank, query_group),
        }
    )
    rows = payload.get("factors")
    tensors = payload.get("factor_tensors")
    names = tuple(sorted(expected_shapes))
    if (
        len(names) != guided.GUIDED_TOTAL_FACTOR_COUNT
        or not isinstance(rows, list)
        or len(rows) != len(names)
        or not isinstance(tensors, Mapping)
        or set(tensors) != set(names)
    ):
        raise ExactCandidateFrontierError(
            "Guided factor manifest does not contain exactly 26 tensors."
        )
    canonical_rows: list[dict[str, Any]] = []
    total_parameters = 0
    for index, (name, row) in enumerate(zip(names, rows, strict=True)):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"name", "shape", "dtype", "tensor_sha256"}
            or row.get("name") != name
        ):
            raise ExactCandidateFrontierError(
                f"Guided factor manifest row {index} is malformed."
            )
        tensor = tensors[name]
        expected_shape = expected_shapes[name]
        try:
            exact_owned_storage = (
                tensor.is_contiguous()
                and tensor.storage_offset() == 0
                and tensor.untyped_storage().nbytes()
                == tensor.numel() * tensor.element_size()
            )
        except (AttributeError, RuntimeError, TypeError):
            exact_owned_storage = False
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.layout != torch.strided
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != tuple(expected_shape)
            or not exact_owned_storage
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ExactCandidateFrontierError(
                f"Guided factor {name!r} is not finite CPU FP32 with exact owned "
                "storage at its exact shape."
            )
        digest = guided.stage._factor_tensor_sha256(tensor.detach().contiguous())
        expected_row = {
            "name": name,
            "shape": list(expected_shape),
            "dtype": str(tensor.dtype),
            "tensor_sha256": digest,
        }
        if dict(row) != expected_row:
            raise ExactCandidateFrontierError(
                f"Guided factor {name!r} failed manifest authentication."
            )
        canonical_rows.append(expected_row)
        total_parameters += int(tensor.numel())
    if total_parameters != guided.GUIDED_TOTAL_FACTOR_PARAMETERS:
        raise ExactCandidateFrontierError(
            "Guided factor parameter count is not the exact v3 total."
        )
    state_preimage = {
        "schema": guided.TRAINER_FACTOR_PACK_V3_SCHEMA,
        "lexical_rank": lexical_rank,
        "factors": canonical_rows,
        "guided_source_binding": source_binding.as_dict(),
        "guided_source_binding_sha256": source_binding.binding_sha256,
        "inherited_product_regrouped": False,
        "guided_product_regrouped": False,
    }
    state_sha = guided._require_sha256(
        payload.get("factor_state_sha256"), "guided_factor.factor_state_sha256"
    )
    if guided.stage._canonical_sha256(state_preimage) != state_sha:
        raise ExactCandidateFrontierError(
            "Guided v3 factor-state SHA-256 does not authenticate its 26 tensors."
        )
    return {
        "factor_state_sha256": state_sha,
        "envelope_provenance": {
            name: envelope[name]
            for name in (
                "reason",
                "preregistration_sha256",
                "evaluation_contract_sha256",
                "source_factor_state_sha256",
                "audit_report_self_sha256",
                "audit_report_canonical_sha256",
                "basis_envelope_file_sha256",
                "basis_domain_artifact_sha256",
                "basis_payload_sha256",
            )
        },
        "factor_pack_schema": guided.FACTOR_ENVELOPE_SCHEMA,
        "trainer_factor_pack_schema": guided.TRAINER_FACTOR_PACK_V3_SCHEMA,
        "lexical_rank": lexical_rank,
        "factor_count": len(names),
        "factor_parameters": total_parameters,
        "guided_source_binding": source_binding.as_dict(),
        "canonical_runner_path": str(Path(guided.__file__).resolve()),
    }


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
    if isinstance(envelope, Mapping):
        if envelope.get("schema") == GUIDED_FACTOR_ENVELOPE_SCHEMA:
            return _guided_source_pinned_factor_authentication(
                envelope, lexical_rank, torch
            )
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
    raw = strict_json(preregistration_path)
    if (
        isinstance(raw, Mapping)
        and raw.get("schema")
        == GUIDED_WINNER_PUBLICATION_PREREGISTRATION_SCHEMA
    ):
        publication = _load_guided_winner_publication_runner()
        contract = publication.authenticate_winner_publication_preregistration(
            preregistration_path, preregistration_sha256
        )
        return contract.preflight_report()
    if isinstance(raw, Mapping) and raw.get("schema") == GUIDED_PREREGISTRATION_SCHEMA:
        guided = _load_guided_runner()
        contract = guided.authenticate_guided_preregistration(
            preregistration_path, preregistration_sha256
        )
        base = contract.execution_contract
        continuation = base.continuation
        cache = base.compiler_cache
        if continuation is None or cache is None:
            raise ExactCandidateFrontierError(
                "Guided retention requires its authenticated v2 source and cache."
            )
        lexical = base.lexical_residual
        if lexical is None:
            raise ExactCandidateFrontierError(
                "Guided retention requires its authenticated lexical residual."
            )
        source_rows = continuation.factor_payload.get("factors")
        if not isinstance(source_rows, list):
            raise ExactCandidateFrontierError(
                "Guided retention source v2 manifest is absent."
            )
        source_hashes = {
            str(row.get("name")): str(row.get("tensor_sha256"))
            for row in source_rows
            if isinstance(row, Mapping)
        }
        artifacts = contract.artifacts
        preflight = contract.preflight_report()
        preflight["retention_authentication"] = {
            "artifact_authentication": dict(base.artifact_authentication),
            "parent": {
                "path": str(base.parent.path),
                "exact_unique_parameters": base.parent.exact_unique_parameters,
            },
            "blank_target": {
                "path": str(base.target.path),
                "exact_unique_parameters": base.target.exact_unique_parameters,
            },
            "compiler_cache_root": str(cache.root),
            "lexical_rank": lexical.rank,
            "source_v2_tensor_sha256s": source_hashes,
            "protected_sources": {
                "source_run_report": str(artifacts.source_run_report_path),
                "source_factor_pack": str(artifacts.source_factor_pack_path),
                "audit_preregistration": str(artifacts.audit_preregistration_path),
                "audit_run_report": str(artifacts.audit_run_report_path),
                "guided_basis_pack": str(artifacts.guided_basis_pack_path),
            },
        }
        return preflight
    runner = _load_exact_stage_runner()
    contract = runner.authenticate_preregistration(
        preregistration_path, preregistration_sha256
    )
    return contract.preflight_report()


def _source_pinned_method_gate_authenticator(
    evidence: Mapping[str, Any], metrics: Mapping[str, Any] | None
) -> tuple[bool, Mapping[str, Any]]:
    guided = _load_guided_runner()
    try:
        passed, decision = (
            guided._guided_method_gate(evidence)
            if metrics is None
            else guided._post_bake_guided_method_gate(metrics, evidence)
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise ExactCandidateFrontierError(
            "Canonical guided method-gate authentication failed."
        ) from error
    if not isinstance(passed, bool) or not isinstance(decision, Mapping):
        raise ExactCandidateFrontierError(
            "Canonical guided method-gate returned a malformed result."
        )
    return passed, dict(decision)


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
            method_gate_authenticator=(_source_pinned_method_gate_authenticator),
        )
        result = frontier.register(candidate)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
