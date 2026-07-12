from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_forge import hf_exact_candidate_frontier as frontier


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "retain_qwen_exact_target_candidates.py"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return frontier.file_sha256(path)


def _metrics(
    *,
    exact: int = 0,
    lcp4: int = 0,
    lcp: int = 1,
    tokens: int = 2,
    actions: int = 3,
    repeat: float = 0.25,
    adjacent: float = 0.25,
) -> dict:
    lcp = max(lcp, exact * 8, lcp4 * 4)
    tokens = max(tokens, lcp, exact * 8)
    actions = max(actions, exact * 8)
    rollout_count = 24
    rollout_tokens = 192
    action_count = 192
    return {
        "split": "calibration-selection",
        "action_count": action_count,
        "action_agreement_count": actions,
        "action_accuracy": actions / action_count,
        "parent_action_mean_logprob": -4.0,
        "parent_action_topk_fraction": 0.25,
        "parent_action_mean_rank": 10.0,
        "parent_action_worst_rank": 20,
        "rollout_count": rollout_count,
        "exact_rollout_count": exact,
        "exact_rollout_fraction": exact / rollout_count,
        "rollout_token_count": rollout_tokens,
        "token_match_count": tokens,
        "token_match_fraction": tokens / rollout_tokens,
        "lcp_token_count": lcp,
        "lcp_fraction": lcp / rollout_tokens,
        "lcp_at_least_4_count": lcp4,
        "repeat_fraction_excess": repeat,
        "adjacent_repeat_fraction_excess": adjacent,
        "longest_run_excess": 1,
        "report_sha256": _sha(
            f"metrics:{exact}:{lcp4}:{lcp}:{tokens}:{actions}:{repeat}:{adjacent}"
        ),
    }


def _artifact_auth(name: str) -> dict:
    return {
        "path": f"/not-used/{name}",
        "required_files": [
            {
                "path": "model.safetensors",
                "sha256": _sha(f"{name}:model"),
                "size_bytes": 7,
            }
        ],
        "tree": None,
        "authenticated_before_model_access": True,
    }


@dataclass(frozen=True)
class _RunCase:
    run: Path
    run_report_sha256: str
    preregistration_sha256: str
    factor_authenticator: frontier.FactorAuthenticator
    preregistration_authenticator: frontier.PreregistrationAuthenticator
    accepted_child_authenticator: frontier.AcceptedChildAuthenticator
    method_gate_authenticator: frontier.MethodGateAuthenticator | None = None

    def candidate(
        self,
        *,
        expected_run_report_sha256: str | None = None,
        expected_preregistration_sha256: str | None = None,
        include_factor_authenticator: bool = True,
        include_child_authenticator: bool = True,
    ) -> frontier.TerminalCandidate:
        return frontier.authenticate_terminal_candidate(
            self.run,
            expected_run_report_sha256=(
                self.run_report_sha256
                if expected_run_report_sha256 is None
                else expected_run_report_sha256
            ),
            expected_preregistration_sha256=(
                self.preregistration_sha256
                if expected_preregistration_sha256 is None
                else expected_preregistration_sha256
            ),
            preregistration_authenticator=self.preregistration_authenticator,
            factor_authenticator=(
                self.factor_authenticator if include_factor_authenticator else None
            ),
            accepted_child_authenticator=(
                self.accepted_child_authenticator
                if include_child_authenticator
                else None
            ),
            method_gate_authenticator=self.method_gate_authenticator,
        )


def _make_run(
    tmp_path: Path,
    name: str,
    *,
    metrics: dict | None = None,
    step0_metrics: dict | None = None,
    status: str = "rejected",
    comparison: str | None = None,
    accepted_factor_fallback: bool = False,
) -> _RunCase:
    run = tmp_path / "runs" / name
    run.mkdir(parents=True)
    blank = tmp_path / "canonical_blank"
    cache = tmp_path / "compiler_cache"
    blank.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    blank_file = blank / "model.safetensors"
    cache_file = cache / "receipt.json"
    if not blank_file.exists():
        blank_file.write_bytes(b"one immutable blank")
    if not cache_file.exists():
        cache_file.write_bytes(b"one immutable compiler cache")
    comparison_sha = _sha("comparison") if comparison is None else comparison
    runner_sha = _sha("canonical runner source")
    trainer_sha = _sha("canonical trainer source")
    preregistration = {
        "schema": frontier.PREREGISTRATION_SCHEMA,
        "sealed_before_model_access": True,
        "parent": {
            "path": str(tmp_path / "parent"),
            "exact_unique_parameters": 4_000,
        },
        "blank_target": {
            "path": str(blank),
            "exact_unique_parameters": 1_400,
        },
        "implementation": {
            "runner_sha256": runner_sha,
            "trainer_module": "neural_forge.test_output_trainer",
            "trainer_sha256": trainer_sha,
        },
        "training": {"evaluation_horizon": 8},
    }
    prereg_path = run / "preregistration.json"
    prereg_sha = _write_json(prereg_path, preregistration)
    selected_metrics = (
        _metrics(lcp=2, tokens=3, actions=4) if metrics is None else metrics
    )
    baseline_metrics = (
        _metrics(lcp=0, tokens=0, actions=0) if step0_metrics is None else step0_metrics
    )
    step0_sha = _sha(f"{name}:step0")
    selected_sha = _sha(f"{name}:selected")
    evidence = {
        "selected_step": 8,
        "checkpoints": [
            {
                "step": 0,
                "factor_state_sha256": step0_sha,
                "metrics": baseline_metrics,
            },
            {
                "step": 8,
                "factor_state_sha256": selected_sha,
                "metrics": selected_metrics,
            },
        ],
        "candidate_metrics": selected_metrics,
        "factor_pack_manifest": {
            "schema": "test-factor-schema",
            "factor_state_sha256": selected_sha,
        },
        "selection_policy": "rollout-first-acquisition",
        "internal_numeric_matching_used": False,
        "selection_uses_internal_numeric_targets": False,
        "candidate_gate_passed": status == "accepted",
        "sealed_evaluation_unlocked": status == "accepted",
    }
    decisions = {
        "fit": {"passed": True, "metrics": selected_metrics},
        "candidate": {
            "passed": status == "accepted",
            "metrics": selected_metrics,
        },
        "sealed": (
            {"passed": True, "metrics": selected_metrics}
            if status == "accepted"
            else None
        ),
    }
    report = {
        "schema": frontier.RUN_SCHEMA,
        "status": status,
        "preregistration": str(prereg_path),
        "preregistration_sha256": prereg_sha,
        "evaluation_contract_sha256": comparison_sha,
        "artifact_authentication": {
            "parent": _artifact_auth("parent"),
            "blank_target": _artifact_auth("blank"),
        },
        "implementation_authentication": {
            "runner": {
                "path": str(tmp_path / "runner.py"),
                "sha256": runner_sha,
            },
            "trainer": {
                "module": "neural_forge.test_output_trainer",
                "path": str(tmp_path / "trainer.py"),
                "sha256": trainer_sha,
            },
            "authenticated_before_model_access": True,
        },
        "splits": {
            role: {
                "path": str(run / f"{role}.json"),
                "sha256": _sha(f"{name}:{role}"),
                "row_count": count,
            }
            for role, count in (("fit", 120), ("candidate", 24), ("sealed", 32))
        },
        "compiler_split": {
            "path": str(run / "compiler.json"),
            "sha256": _sha(f"{name}:compiler"),
            "row_count": 120,
            "reuses_behavior_fit": False,
        },
        "thresholds": {
            role: {"minimum_action_accuracy": 0.5}
            for role in ("fit", "candidate", "sealed")
        },
        "compiler_cache": {"root": str(cache)},
        "trainer_evidence": evidence,
        "trainer_evidence_sha256": frontier.canonical_sha256(evidence),
        "factor_acquisition": {
            "selected_step": 8,
            "step0_factor_state_sha256": step0_sha,
            "selected_factor_state_sha256": selected_sha,
            "target_native_factors_acquired_behavior": True,
        },
        "behavior_decisions": decisions,
        "pre_bake_behavior_decisions": decisions,
    }
    if status == "rejected":
        factor_path = run / "rejected_factor_pack.pt"
        factor_path.write_bytes(f"factor:{selected_sha}".encode())
        report.update(
            {
                "reason": "candidate-gate-rejected",
                "standalone_compaction_passed": False,
                "model_checkpoint_written": False,
                "rollback_exact": True,
                "factor_pack": {
                    "path": str(factor_path),
                    "sha256": frontier.file_sha256(factor_path),
                },
            }
        )
    else:
        child = run / "accepted_child"
        child.mkdir()
        (child / "model.safetensors").write_bytes(b"tiny accepted child")
        (child / "config.json").write_text("{}\n", encoding="utf-8")
        tree = frontier.artifact_tree(child)
        accepted_decisions = {
            role: {"passed": True, "metrics": selected_metrics}
            for role in ("fit", "candidate", "sealed")
        }
        report.update(
            {
                "standalone_compaction_passed": True,
                "model_checkpoint_written": True,
                "rollback_exact": None,
                "pre_bake_behavior_decisions": accepted_decisions,
                "behavior_decisions": accepted_decisions,
                "post_bake_behavior_decisions": accepted_decisions,
                "fresh_reload_certificate": {
                    "fresh_reload_performed": True,
                    "exact_unique_parameters": 1_400,
                    "exact_budget_match": True,
                    "state_keys_match_pre_save": True,
                    "state_bytes_match_pre_save": True,
                    "state_dict_sha256": _sha(f"{name}:child-state"),
                    "published_adapter_parameter_count": 0,
                    "published_parent_runtime_reference_count": 0,
                    "published_source_sized_sidecar_parameter_count": 0,
                    "artifact_tree": {
                        key: tree[key]
                        for key in ("file_count", "total_bytes", "tree_sha256")
                    },
                    "path": str(child),
                },
            }
        )
        if accepted_factor_fallback:
            factor_path = run / "selected_factor_pack.pt"
            factor_path.write_bytes(f"factor:{selected_sha}".encode())
            report["selected_factor_pack"] = {
                "path": str(factor_path),
                "sha256": frontier.file_sha256(factor_path),
            }
    _write_json(run / "run_report.json", report)
    report_sha = frontier.file_sha256(run / "run_report.json")

    def authenticate(path: Path, lexical_rank: int | None) -> dict:
        expected_path = run / (
            "selected_factor_pack.pt"
            if status == "accepted" and accepted_factor_fallback
            else "rejected_factor_pack.pt"
        )
        assert path == expected_path
        assert lexical_rank is None
        return {
            "factor_state_sha256": selected_sha,
            "envelope_provenance": {
                "reason": (
                    "accepted-selected-factor"
                    if status == "accepted"
                    else "candidate-gate-rejected"
                ),
                "preregistration_sha256": prereg_sha,
                "evaluation_contract_sha256": comparison_sha,
            },
        }

    preflight = {
        "status": "preflight-passed",
        "preregistration": str(prereg_path),
        "preregistration_sha256": prereg_sha,
        "evaluation_contract_sha256": comparison_sha,
        "artifact_authentication": report["artifact_authentication"],
        "implementation_authentication": report["implementation_authentication"],
        "splits": report["splits"],
        "compiler_split": report["compiler_split"],
        "thresholds": report["thresholds"],
    }

    def authenticate_preregistration(path: Path, digest: str) -> dict:
        assert path == prereg_path
        assert digest == prereg_sha
        return preflight

    def authenticate_child(path: Path, expected_parameters: int) -> dict:
        assert path == run / "accepted_child"
        assert expected_parameters == 1_400
        tree = frontier.artifact_tree(path)
        certificate = report["fresh_reload_certificate"]
        return {
            "fresh_reload_performed": True,
            "loaded_from_candidate_path": True,
            "exact_unique_parameters": 1_400,
            "state_dict_sha256": certificate["state_dict_sha256"],
            "artifact_tree": {
                key: tree[key] for key in ("file_count", "total_bytes", "tree_sha256")
            },
            "published_adapter_parameter_count": 0,
            "parent_runtime_reference_count": 0,
            "source_sized_sidecar_parameter_count": 0,
        }

    return _RunCase(
        run=run,
        run_report_sha256=report_sha,
        preregistration_sha256=prereg_sha,
        factor_authenticator=authenticate,
        preregistration_authenticator=authenticate_preregistration,
        accepted_child_authenticator=authenticate_child,
    )


_GUIDED_METHOD_GATE = {
    "minimum_selected_step": 1,
    "minimum_candidate_lcp_at_least_4_count": 1,
    "require_candidate_repeat_fraction_excess_nonregression_vs_step0": True,
    "require_candidate_adjacent_repeat_fraction_excess_nonregression_vs_step0": True,
}


def _guided_decision(step0: dict, candidate: dict, *, selected_step: int = 8) -> dict:
    checks = {
        "selected_step_acquired": selected_step >= 1,
        "candidate_has_one_lcp_at_least_4_rollout": (
            candidate["lcp_at_least_4_count"] >= 1
        ),
        "candidate_repeat_fraction_nonregression_vs_step0": (
            candidate["repeat_fraction_excess"] <= step0["repeat_fraction_excess"]
        ),
        "candidate_adjacent_repeat_fraction_nonregression_vs_step0": (
            candidate["adjacent_repeat_fraction_excess"]
            <= step0["adjacent_repeat_fraction_excess"]
        ),
    }
    trainer_gate = {
        "selected_step": selected_step,
        "selected_step_is_promoted": checks["selected_step_acquired"],
        "minimum_lcp_at_least_4_count": 1,
        "observed_lcp_at_least_4_count": candidate["lcp_at_least_4_count"],
        "baseline_repeat_fraction_excess": step0["repeat_fraction_excess"],
        "candidate_repeat_fraction_excess": candidate["repeat_fraction_excess"],
        "repeat_fraction_nonregression": checks[
            "candidate_repeat_fraction_nonregression_vs_step0"
        ],
        "baseline_adjacent_repeat_fraction_excess": step0[
            "adjacent_repeat_fraction_excess"
        ],
        "candidate_adjacent_repeat_fraction_excess": candidate[
            "adjacent_repeat_fraction_excess"
        ],
        "adjacent_repeat_fraction_nonregression": checks[
            "candidate_adjacent_repeat_fraction_nonregression_vs_step0"
        ],
        "passed": all(checks.values()),
    }
    decision = {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": dict(_GUIDED_METHOD_GATE),
        "selected_step": selected_step,
        "step0_metrics": {
            name: step0[name]
            for name in (
                "lcp_at_least_4_count",
                "repeat_fraction_excess",
                "adjacent_repeat_fraction_excess",
            )
        },
        "candidate_metrics": {
            name: candidate[name]
            for name in (
                "lcp_at_least_4_count",
                "repeat_fraction_excess",
                "adjacent_repeat_fraction_excess",
            )
        },
        "four_isolated_one_token_lcps_cannot_substitute": True,
        "internal_numeric_matching_consulted": False,
        "trainer_gate": trainer_gate,
    }
    decision["decision_sha256"] = frontier.canonical_sha256(decision)
    return decision


def _make_guided_run(
    tmp_path: Path,
    name: str,
    *,
    status: str = "rejected",
    method_positive: bool = True,
) -> _RunCase:
    selected_metrics = _metrics(
        lcp4=1 if method_positive else 0,
        lcp=4 if method_positive else 3,
        tokens=5,
        actions=6,
        repeat=0.2 if method_positive else 0.4,
        adjacent=0.2 if method_positive else 0.4,
    )
    step0_metrics = _metrics(
        lcp4=0,
        lcp=0,
        tokens=1,
        actions=1,
        repeat=0.3,
        adjacent=0.3,
    )
    base = _make_run(
        tmp_path,
        name,
        status=status,
        metrics=selected_metrics,
        step0_metrics=step0_metrics,
        accepted_factor_fallback=status == "accepted",
    )
    run = base.run
    old_prereg_path = run / "preregistration.json"
    old_prereg = frontier.strict_json(old_prereg_path)
    report_path = run / "run_report.json"
    report = frontier.strict_json(report_path)
    artifact_authentication = report.pop("artifact_authentication")
    compiler_cache_root = report.pop("compiler_cache")["root"]
    report.pop("behavior_decisions", None)
    selected_sha = report["factor_acquisition"]["selected_factor_state_sha256"]
    source_state_sha = _sha(f"{name}:source-v2-state")
    source_hashes = {
        f"source.factor.{index:02d}": _sha(f"{name}:source:{index}")
        for index in range(18)
    }
    basis_hashes = {role: _sha(f"{name}:basis:{role}") for role in ("q", "k", "v", "o")}
    audit = {
        "audit_report_self_sha256": _sha(f"{name}:audit-self"),
        "audit_report_canonical_sha256": _sha(f"{name}:audit-canonical"),
        "basis_envelope_file_sha256": _sha(f"{name}:basis-envelope"),
        "basis_domain_artifact_sha256": _sha(f"{name}:basis-domain"),
        "basis_payload_sha256": _sha(f"{name}:basis-payload"),
        "basis_manifest_sha256": _sha(f"{name}:basis-manifest"),
        "basis_tensor_sha256s": basis_hashes,
    }
    installation_binding = {
        "inherited_installation_sha256": _sha(f"{name}:inherited-installation"),
        "inherited_factor_state_sha256_at_install": _sha(
            f"{name}:inherited-factor-state-at-install"
        ),
        "adapter_site_indices_sha256": _sha(f"{name}:adapter-sites"),
        "audit_site_indices_sha256": _sha(f"{name}:audit-sites"),
    }
    source = {
        "factor_state_sha256": source_state_sha,
        "source_evaluation_contract_sha256": _sha(f"{name}:source-evaluation"),
    }
    runner_sha = _sha(f"{name}:guided-runner")
    trainer_sha = _sha(f"{name}:guided-trainer")
    implementation = {
        "runner_sha256": runner_sha,
        "trainer_module": "neural_forge.hf_target_native_behavior_distillation",
        "trainer_sha256": trainer_sha,
        "builder_sha256": _sha(f"{name}:guided-builder"),
        "base_runner_sha256": _sha(f"{name}:base-runner"),
        "guided_adapter_sha256": _sha(f"{name}:guided-adapter"),
        "progressive_bridge_sha256": _sha(f"{name}:bridge"),
        "compiled_child_cache_sha256": _sha(f"{name}:cache-module"),
    }
    implementation_authentication = dict(implementation)
    comparison_sha = _sha(f"{name}:guided-evaluation-contract")
    report["evaluation_contract_sha256"] = comparison_sha
    guided_prereg = {
        "schema": frontier.GUIDED_PREREGISTRATION_SCHEMA,
        "sealed_before_model_access": True,
        "implementation": implementation,
        "source": source,
        "audit": audit,
        "splits": report["splits"],
        "thresholds": report["thresholds"],
        "training": old_prereg["training"],
        "method_gate": dict(_GUIDED_METHOD_GATE),
        "runtime": {"dtype": "float16"},
        "evaluation_contract_sha256": comparison_sha,
    }
    prereg_sha = _write_json(old_prereg_path, guided_prereg)
    decision = _guided_decision(step0_metrics, selected_metrics)
    report.update(
        {
            "schema": frontier.GUIDED_RUN_SCHEMA,
            "preregistration_sha256": prereg_sha,
            "parent": {
                "path": old_prereg["parent"]["path"],
                "exact_unique_parameters": old_prereg["parent"][
                    "exact_unique_parameters"
                ],
            },
            "target": {
                "path": old_prereg["blank_target"]["path"],
                "exact_unique_parameters": old_prereg["blank_target"][
                    "exact_unique_parameters"
                ],
            },
            "guided_qkvo": {
                "new_factor_count": 8,
                "combined_factor_count": 26,
                "separate_product_no_regroup": True,
            },
            "source": source,
            "audit": audit,
            "implementation_authentication": implementation_authentication,
            "selection_domain": "causal-output-actions-and-free-rollouts-only",
            "method_gate": dict(_GUIDED_METHOD_GATE),
            "internal_numeric_matching_can_promote": False,
            "sealed_evaluation_locked_until_candidate_passes": True,
            "guided_method_decision": decision,
            "guided_installation": dict(installation_binding),
            "models_loaded": True,
            "gpu_work_launched": True,
            "output_root": str(run),
            "compiler_report": {"schema": "test-guided-compiler-report-v1"},
            "inherited_installation": {
                "installation_sha256": installation_binding[
                    "inherited_installation_sha256"
                ]
            },
            "source_v2_restore": {
                "factor_state_sha256": source_state_sha,
            },
            "last_completed_phase": "guided-zero-left-installation",
            "selection_used_internal_numeric_matching": False,
        }
    )
    evidence = report["trainer_evidence"]
    evidence["guided_candidate_gate"] = decision["trainer_gate"]
    evidence["candidate_gate_passed"] = (
        report["pre_bake_behavior_decisions"]["candidate"]["passed"]
        and decision["passed"]
    )
    evidence["sealed_evaluation_unlocked"] = evidence["candidate_gate_passed"]
    report["trainer_evidence_sha256"] = frontier.canonical_sha256(evidence)
    old_factor = run / (
        "selected_factor_pack.pt" if status == "accepted" else "rejected_factor_pack.pt"
    )
    guided_factor = run / "selected_guided_factor_pack.pt"
    old_factor.replace(guided_factor)
    report.pop("factor_pack", None)
    report["selected_factor_pack"] = {
        "path": str(guided_factor),
        "sha256": frontier.file_sha256(guided_factor),
    }
    if status == "accepted":
        post_decision = _guided_decision(step0_metrics, selected_metrics)
        report["post_bake_guided_method_decision"] = post_decision
        report["fresh_reload_guided_method_gate"] = {
            "passed": True,
            "post_bake_decision_sha256": post_decision["decision_sha256"],
            "state_bytes_match_pre_save": True,
            "internal_numeric_matching_consulted": False,
        }
        report.update(
            {
                "guided_publication_handoff": {
                    "separate_product_no_regroup": True,
                },
                "publication": {"baked": True},
                "model_save_count": 1,
                "retained_artifacts": [
                    "selected_guided_factor_pack.pt",
                    "accepted_child",
                    "run_report.json",
                ],
            }
        )
    else:
        report.update(
            {
                "rollback_state_sha256": _sha(f"{name}:rollback-state"),
                "retained_artifacts": [
                    "selected_guided_factor_pack.pt",
                    "run_report.json",
                ],
            }
        )
    report_sha = _write_json(report_path, report)

    protected_dir = run / "guided_sources"
    protected_dir.mkdir()
    protected_sources: dict[str, str] = {}
    for label in (
        "source_run_report",
        "source_factor_pack",
        "audit_preregistration",
        "audit_run_report",
        "guided_basis_pack",
    ):
        path = protected_dir / f"{label}.bin"
        path.write_bytes(label.encode())
        protected_sources[label] = str(path)
    retention_authentication = {
        "artifact_authentication": artifact_authentication,
        "parent": old_prereg["parent"],
        "blank_target": old_prereg["blank_target"],
        "compiler_cache_root": compiler_cache_root,
        "lexical_rank": 32,
        "source_v2_tensor_sha256s": source_hashes,
        "protected_sources": protected_sources,
    }
    preflight = {
        "status": "preflight-passed",
        "preregistration": str(old_prereg_path),
        "preregistration_sha256": prereg_sha,
        "evaluation_contract_sha256": comparison_sha,
        "implementation_authentication": implementation_authentication,
        "splits": report["splits"],
        "compiler_split": report["compiler_split"],
        "thresholds": report["thresholds"],
        "parent": report["parent"],
        "target": report["target"],
        "guided_qkvo": report["guided_qkvo"],
        "source": source,
        "audit": audit,
        "selection_domain": report["selection_domain"],
        "method_gate": report["method_gate"],
        "internal_numeric_matching_can_promote": False,
        "sealed_evaluation_locked_until_candidate_passes": True,
        "retention_authentication": retention_authentication,
    }

    def authenticate_preregistration(path: Path, digest: str) -> dict:
        assert path == old_prereg_path and digest == prereg_sha
        return preflight

    guided_binding = {
        "source_v2_factor_state_sha256": source_state_sha,
        "source_v2_tensor_sha256s": source_hashes,
        "audit_report_sha256": audit["audit_report_self_sha256"],
        "basis_envelope_file_sha256": audit["basis_envelope_file_sha256"],
        "basis_payload_sha256": audit["basis_payload_sha256"],
        "basis_manifest_sha256": audit["basis_manifest_sha256"],
        "source_basis_tensor_sha256s": basis_hashes,
        **installation_binding,
    }

    def authenticate_factor(path: Path, lexical_rank: int | None) -> dict:
        assert path == guided_factor and lexical_rank == 32
        return {
            "factor_state_sha256": selected_sha,
            "envelope_provenance": {
                "reason": "selected",
                "preregistration_sha256": prereg_sha,
                "evaluation_contract_sha256": comparison_sha,
                "source_factor_state_sha256": source_state_sha,
                **{
                    name: audit[name]
                    for name in (
                        "audit_report_self_sha256",
                        "audit_report_canonical_sha256",
                        "basis_envelope_file_sha256",
                        "basis_domain_artifact_sha256",
                        "basis_payload_sha256",
                    )
                },
            },
            "factor_pack_schema": (
                "neural-forge-rejected-target-native-guided-factor-pack-v1"
            ),
            "trainer_factor_pack_schema": ("neural-forge-target-native-factor-pack-v3"),
            "factor_count": 26,
            "factor_parameters": 28_868_608,
            "lexical_rank": 32,
            "guided_source_binding": guided_binding,
        }

    def authenticate_method(_evidence: dict, metrics: dict | None):
        selected = (
            report.get("post_bake_guided_method_decision")
            if metrics is not None
            else decision
        )
        assert isinstance(selected, dict)
        return bool(selected["passed"]), selected

    return _RunCase(
        run=run,
        run_report_sha256=report_sha,
        preregistration_sha256=prereg_sha,
        factor_authenticator=authenticate_factor,
        preregistration_authenticator=authenticate_preregistration,
        accepted_child_authenticator=base.accepted_child_authenticator,
        method_gate_authenticator=authenticate_method,
    )


def test_authentication_truthfully_distinguishes_factor_and_accepted_child(
    tmp_path: Path,
) -> None:
    rejected = _make_run(tmp_path, "rejected")
    research = rejected.candidate()
    assert research.truth_label == "research-factor-only-candidate"
    assert research.fresh_reload_verified is False
    assert research.standalone_compaction_passed is False
    assert research.deployment_accepted is False

    accepted = _make_run(
        tmp_path,
        "accepted",
        status="accepted",
        metrics=_metrics(exact=1, lcp4=1, lcp=8, tokens=8, actions=8),
    )
    standalone = accepted.candidate()
    assert standalone.truth_label == "accepted-standalone"
    assert standalone.fresh_reload_verified is True
    assert standalone.standalone_compaction_passed is True
    assert standalone.deployment_accepted is True
    assert standalone.source_artifact_tree is not None


def test_factor_and_child_tampering_fail_before_frontier_mutation(
    tmp_path: Path,
) -> None:
    case = _make_run(tmp_path, "factor-tamper")
    (case.run / "rejected_factor_pack.pt").write_bytes(b"changed")
    with pytest.raises(frontier.ExactCandidateFrontierError, match="file hash"):
        case.candidate()

    child_case = _make_run(tmp_path, "child-tamper", status="accepted")
    (child_case.run / "accepted_child" / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(
        frontier.ExactCandidateFrontierError,
        match="failed independent authentication",
    ):
        child_case.candidate()


def test_constructor_and_inspection_are_zero_write(tmp_path: Path) -> None:
    root = tmp_path / "not-created-by-read"
    manager = frontier.ExactCandidateFrontier(root)
    assert not root.exists()
    assert manager.inspect()["retained_record_count"] == 0
    assert not root.exists()


def test_preexisting_unowned_deterministic_dirs_and_extras_are_never_adopted(
    tmp_path: Path,
) -> None:
    case = _make_run(tmp_path, "orphan")
    candidate = case.candidate()
    root = tmp_path / "orphan-frontier"
    artifact = (
        root / "artifacts" / f"factor-file-{candidate.source_artifact_identity_sha256}"
    )
    artifact.mkdir(parents=True)
    (root / "evidence").mkdir()
    (artifact / "factor_pack.pt").write_bytes(candidate.source_artifact.read_bytes())
    manager = frontier.ExactCandidateFrontier(root)
    with pytest.raises(frontier.ExactCandidateFrontierError, match="manifestless"):
        manager.register(candidate)
    assert not (root / "frontier_manifest.json").exists()
    assert candidate.source_artifact.is_file()

    extra_root = tmp_path / "extra-frontier"
    extra_root.mkdir()
    (extra_root / "unexpected.txt").write_text("unowned", encoding="utf-8")
    with pytest.raises(frontier.ExactCandidateFrontierError, match="unowned"):
        frontier.ExactCandidateFrontier(extra_root).inspect()


def test_source_run_and_artifacts_are_protected_before_frontier_creation(
    tmp_path: Path,
) -> None:
    case = _make_run(tmp_path, "source-overlap")
    nested_frontier = case.run / "candidate-frontier"
    manager = frontier.ExactCandidateFrontier(nested_frontier)
    assert not nested_frontier.exists()
    with pytest.raises(frontier.ExactCandidateFrontierError, match="overlaps"):
        manager.register(case.candidate())
    assert not nested_frontier.exists()
    assert case.run.is_dir()
    assert (case.run / "rejected_factor_pack.pt").is_file()


def test_external_hash_roots_and_canonical_source_pins_fail_before_any_write(
    tmp_path: Path,
) -> None:
    case = _make_run(tmp_path, "external-roots")
    frontier_root = tmp_path / "frontier"
    frontier.ExactCandidateFrontier(frontier_root)
    with pytest.raises(frontier.ExactCandidateFrontierError, match="external root"):
        case.candidate(expected_run_report_sha256="0" * 64)
    assert not frontier_root.exists()
    with pytest.raises(
        frontier.ExactCandidateFrontierError, match="external and report"
    ):
        case.candidate(expected_preregistration_sha256="1" * 64)
    assert not frontier_root.exists()

    def forged_preregistration(_path: Path, _digest: str) -> dict:
        value = case.preregistration_authenticator(
            case.run / "preregistration.json", case.preregistration_sha256
        )
        forged = json.loads(json.dumps(value))
        forged["implementation_authentication"]["runner"]["sha256"] = "2" * 64
        return forged

    with pytest.raises(
        frontier.ExactCandidateFrontierError, match="preflight disagrees"
    ):
        frontier.authenticate_terminal_candidate(
            case.run,
            expected_run_report_sha256=case.run_report_sha256,
            expected_preregistration_sha256=case.preregistration_sha256,
            preregistration_authenticator=forged_preregistration,
            factor_authenticator=case.factor_authenticator,
        )
    assert not frontier_root.exists()


def test_candidate_must_strictly_improve_authenticated_step_zero(
    tmp_path: Path,
) -> None:
    tied = _metrics(lcp=2, tokens=3, actions=4)
    case = _make_run(
        tmp_path,
        "step-zero-tie",
        metrics=tied,
        step0_metrics=tied,
    )
    with pytest.raises(frontier.ExactCandidateFrontierError, match="strictly improve"):
        case.candidate()


def test_accepted_child_needs_independent_reload_or_factor_only_fallback(
    tmp_path: Path,
) -> None:
    no_fallback = _make_run(tmp_path, "accepted-no-fallback", status="accepted")
    with pytest.raises(
        frontier.ExactCandidateFrontierError,
        match="no authenticated factor-only fallback",
    ):
        no_fallback.candidate(include_child_authenticator=False)

    fallback = _make_run(
        tmp_path,
        "accepted-factor-fallback",
        status="accepted",
        accepted_factor_fallback=True,
    )
    retained = fallback.candidate(include_child_authenticator=False)
    assert retained.truth_label == "research-factor-only-candidate"
    assert retained.terminal_status == "accepted"
    assert retained.fresh_reload_verified is False

    def forged_child(_path: Path, _count: int) -> dict:
        return {"fresh_reload_performed": True, "loaded_from_candidate_path": True}

    with pytest.raises(
        frontier.ExactCandidateFrontierError,
        match="no authenticated factor-only fallback",
    ):
        frontier.authenticate_terminal_candidate(
            no_fallback.run,
            expected_run_report_sha256=no_fallback.run_report_sha256,
            expected_preregistration_sha256=no_fallback.preregistration_sha256,
            preregistration_authenticator=no_fallback.preregistration_authenticator,
            factor_authenticator=no_fallback.factor_authenticator,
            accepted_child_authenticator=forged_child,
        )


def test_rollout_first_ranking_rejects_internal_numeric_metrics() -> None:
    better_rollout = _metrics(exact=1, lcp4=1, lcp=8, tokens=8, actions=8)
    better_actions = _metrics(exact=0, lcp=7, tokens=7, actions=8)
    assert frontier.rollout_first_rank(better_rollout) < frontier.rollout_first_rank(
        better_actions
    )
    polluted = {**_metrics(), "weight_similarity": 1.0}
    with pytest.raises(frontier.ExactCandidateFrontierError, match="internal"):
        frontier.rollout_first_rank(polluted)
    inconsistent = {**_metrics(), "lcp_at_least_4_count": 5}
    with pytest.raises(frontier.ExactCandidateFrontierError, match="exceeds"):
        frontier.rollout_first_rank(inconsistent)

    impossible_exact = _metrics(exact=1, lcp4=1)
    impossible_exact.update(
        {
            "lcp_token_count": 0,
            "lcp_fraction": 0.0,
            "token_match_count": 0,
            "token_match_fraction": 0.0,
            "action_agreement_count": 0,
            "action_accuracy": 0.0,
        }
    )
    with pytest.raises(frontier.ExactCandidateFrontierError):
        frontier.rollout_first_rank(impossible_exact)


def test_candidate_counts_are_bound_to_authenticated_split_and_horizon(
    tmp_path: Path,
) -> None:
    malformed = _metrics(lcp=2, tokens=3, actions=4)
    malformed.update(
        {
            "rollout_count": 4,
            "rollout_token_count": 8,
            "action_count": 8,
            "exact_rollout_fraction": 0.0,
            "token_match_fraction": 3 / 8,
            "lcp_fraction": 2 / 8,
            "action_accuracy": 4 / 8,
            "report_sha256": _sha("malformed-denominators"),
        }
    )
    case = _make_run(tmp_path, "bad-denominators", metrics=malformed)
    with pytest.raises(
        frontier.ExactCandidateFrontierError,
        match="authenticated split and evaluation horizon",
    ):
        case.candidate()


def test_artifact_tree_rejects_nested_mount_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "child"
    nested = root / "mounted"
    nested.mkdir(parents=True)
    (nested / "model.safetensors").write_bytes(b"outside mounted content")
    real_ismount = frontier.os.path.ismount
    monkeypatch.setattr(
        frontier.os.path,
        "ismount",
        lambda value: Path(value) == nested or real_ismount(value),
    )
    with pytest.raises(frontier.ExactCandidateFrontierError, match="nested mount"):
        frontier.artifact_tree(root)


def test_max_three_dry_run_then_explicit_prune_never_touches_sources_or_protected(
    tmp_path: Path,
) -> None:
    frontier_root = tmp_path / "candidate_frontier"
    runs: list[Path] = []
    blank = tmp_path / "canonical_blank" / "model.safetensors"
    cache = tmp_path / "compiler_cache" / "receipt.json"
    for index, lcp in enumerate((1, 2, 3, 4)):
        case = _make_run(
            tmp_path,
            f"rank-{index}",
            metrics=_metrics(lcp=lcp, tokens=max(lcp, 2), actions=3 + index),
        )
        runs.append(case.run)
        candidate = case.candidate()
        manager = frontier.ExactCandidateFrontier(frontier_root)
        manager.register(candidate)
    manager = frontier.ExactCandidateFrontier(frontier_root)
    inspection = manager.inspect()
    assert inspection["retained_record_count"] == 3
    assert inspection["pending_prune_count"] == 1
    assert all(run.is_dir() for run in runs)
    assert len(list((frontier_root / "artifacts").iterdir())) == 4
    blank_before = blank.read_bytes()
    cache_before = cache.read_bytes()

    with pytest.raises(frontier.ExactCandidateFrontierError, match="unsupported"):
        manager.apply_pending_prune()
    assert len(list((frontier_root / "artifacts").iterdir())) == 4
    assert all(run.is_dir() for run in runs)
    assert blank.read_bytes() == blank_before
    assert cache.read_bytes() == cache_before


def test_dominated_candidate_is_not_copied_and_contracts_are_bounded_separately(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frontier"
    manager = frontier.ExactCandidateFrontier(root)
    best = _make_run(
        tmp_path,
        "best",
        metrics=_metrics(exact=2, lcp4=2, lcp=16, tokens=16, actions=16),
    )
    best_result = manager.register(best.candidate())
    assert best_result["candidate_retained"] is True
    artifacts_before = {path.name for path in (root / "artifacts").iterdir()}
    for index in range(2):
        case = _make_run(
            tmp_path,
            f"other-{index}",
            metrics=_metrics(lcp=2 + index),
        )
        manager.register(case.candidate())
    dominated = _make_run(
        tmp_path,
        "dominated",
        metrics=_metrics(lcp=0, tokens=1, actions=1, repeat=1.0, adjacent=1.0),
    )
    result = manager.register(dominated.candidate())
    assert result["candidate_retained"] is False
    assert len(list((root / "artifacts").iterdir())) == len(artifacts_before) + 2

    different = _make_run(
        tmp_path,
        "different-contract",
        comparison=_sha("different comparison"),
    )
    manager.register(different.candidate())
    inspection = manager.inspect()
    assert sorted(inspection["retained_per_contract"].values()) == [1, 3]


def test_best_accepted_child_keeps_one_slot_even_when_three_research_keys_win(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frontier"
    manager = frontier.ExactCandidateFrontier(root)
    accepted = _make_run(
        tmp_path,
        "accepted-fallback",
        status="accepted",
        metrics=_metrics(lcp=1, tokens=1, actions=1),
    )
    manager.register(accepted.candidate())
    for index in range(3):
        case = _make_run(
            tmp_path,
            f"better-research-{index}",
            metrics=_metrics(
                exact=1,
                lcp4=1,
                lcp=5 + index,
                tokens=6 + index,
                actions=5 + index,
            ),
        )
        manager.register(case.candidate())
    manifest = frontier.strict_json(root / "frontier_manifest.json")
    labels = [record["truth_label"] for record in manifest["records"]]
    assert len(labels) == 3
    assert labels.count("accepted-standalone") == 1


def test_manifest_publication_failure_deletes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "frontier"
    manager = frontier.ExactCandidateFrontier(root, max_per_contract=1)
    first = _make_run(tmp_path, "first", metrics=_metrics(lcp=1))
    manager.register(first.candidate())
    prior_artifacts = {path.name for path in (root / "artifacts").iterdir()}
    better = _make_run(tmp_path, "better", metrics=_metrics(lcp=4))
    candidate = better.candidate()

    def fail_manifest(_path: Path, _payload: dict) -> None:
        raise OSError("injected manifest failure")

    monkeypatch.setattr(frontier, "_atomic_replace_json", fail_manifest)
    with pytest.raises(OSError, match="injected manifest"):
        manager.register(candidate)
    assert prior_artifacts.issubset(
        {path.name for path in (root / "artifacts").iterdir()}
    )
    assert first.run.is_dir()
    assert better.run.is_dir()


def test_pending_prune_is_durable_and_actual_deletion_is_unsupported(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frontier"
    manager = frontier.ExactCandidateFrontier(root, max_per_contract=1)
    first = _make_run(tmp_path, "prune-old", metrics=_metrics(lcp=1))
    second = _make_run(tmp_path, "prune-new", metrics=_metrics(lcp=4))
    manager.register(first.candidate())
    manager.register(second.candidate())
    before = {path.name for path in (root / "artifacts").iterdir()}
    manifest_before = (root / "frontier_manifest.json").read_bytes()
    with pytest.raises(frontier.ExactCandidateFrontierError, match="unsupported"):
        manager.apply_pending_prune()
    assert {path.name for path in (root / "artifacts").iterdir()} == before
    assert (root / "frontier_manifest.json").read_bytes() == manifest_before
    assert first.run.is_dir() and second.run.is_dir()


def test_changed_pending_tree_and_path_escape_refuse_all_deletion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frontier"
    manager = frontier.ExactCandidateFrontier(root, max_per_contract=1)
    first = _make_run(tmp_path, "old", metrics=_metrics(lcp=1))
    second = _make_run(tmp_path, "new", metrics=_metrics(lcp=4))
    manager.register(first.candidate())
    manager.register(second.candidate())
    manifest = frontier.strict_json(root / "frontier_manifest.json")
    pending = manifest["pending_prune"][0]
    pending_bundle = root / pending["artifact_path"]
    (pending_bundle / "factor_pack.pt").write_bytes(b"tampered pending bytes")
    retained_paths = [root / record["artifact_path"] for record in manifest["records"]]
    with pytest.raises(frontier.ExactCandidateFrontierError, match="changed"):
        manager.inspect()
    assert all(path.is_dir() for path in retained_paths)
    assert first.run.is_dir() and second.run.is_dir()

    # Restore a clean frontier, then forge every hash while retaining an escape.
    clean_root = tmp_path / "clean-frontier"
    clean_manager = frontier.ExactCandidateFrontier(clean_root)
    case = _make_run(tmp_path, "escape", metrics=_metrics(lcp=3))
    clean_manager.register(case.candidate())
    forged = frontier.strict_json(clean_root / "frontier_manifest.json")
    record = forged["records"][0]
    record["artifact_path"] = "../canonical_blank"
    record["record_sha256"] = frontier._record_hash(record, "record_sha256")
    forged["manifest_sha256"] = frontier._record_hash(forged, "manifest_sha256")
    _write_json(clean_root / "frontier_manifest.json", forged)
    with pytest.raises(frontier.ExactCandidateFrontierError, match="escapes"):
        clean_manager.inspect()


def test_frontier_cannot_overlap_blank_or_cache_and_rejects_symlink_artifacts(
    tmp_path: Path,
) -> None:
    blank = tmp_path / "blank"
    blank.mkdir()
    with pytest.raises(frontier.ExactCandidateFrontierError, match="overlaps"):
        frontier.ExactCandidateFrontier(blank / "frontier", protected_roots=[blank])

    case = _make_run(tmp_path, "symlink")
    factor = case.run / "rejected_factor_pack.pt"
    real = case.run / "real-factor.pt"
    real.write_bytes(factor.read_bytes())
    factor.unlink()
    try:
        os.symlink(real, factor)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this host")
    with pytest.raises(
        frontier.ExactCandidateFrontierError,
        match="canonical direct run artifact|ordinary file",
    ):
        case.candidate()


def test_cli_disables_apply_prune_before_creating_frontier(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("retention_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    root = tmp_path / "cli-frontier"
    args = parser.parse_args(["--frontier-root", str(root), "--inspect"])
    assert args.apply_prune is False
    module.main(["--frontier-root", str(root), "--inspect"])
    assert not root.exists()
    with pytest.raises(SystemExit, match="unsupported"):
        module.main(
            [
                "--frontier-root",
                str(root),
                "--prune-only",
                "--apply-prune",
            ]
        )
    assert not root.exists()


def test_conservative_v1_contains_no_recursive_delete_implementation() -> None:
    source = Path(frontier.__file__).read_text(encoding="utf-8")
    cli_source = SCRIPT.read_text(encoding="utf-8")
    assert "rmtree" not in source
    assert "Remove-Item" not in source
    assert "rmtree" not in cli_source


def test_guided_terminal_retains_exact_selected_pack_and_dynamic_behavior_rank(
    tmp_path: Path,
) -> None:
    case = _make_guided_run(tmp_path, "guided-rejected")
    candidate = case.candidate()
    assert candidate.terminal_run_schema == frontier.GUIDED_RUN_SCHEMA
    assert candidate.source_artifact.name == "selected_guided_factor_pack.pt"
    assert candidate.truth_label == "research-factor-only-candidate"
    assert candidate.dynamic_method_gate_decision is not None
    assert candidate.dynamic_method_gate_decision["passed"] is True
    assert candidate.rollout_first_rank_key[:2] == (-1, -4)
    assert candidate.rollout_first_rank_key == frontier.behavior_rank(
        candidate.candidate_metrics,
        candidate.dynamic_method_gate_decision,
    )
    root = tmp_path / "guided-frontier"
    frontier.ExactCandidateFrontier(root).register(candidate)
    manifest = frontier.strict_json(root / "frontier_manifest.json")
    assert len(manifest["records"]) == 1
    assert manifest["records"][0]["terminal_run_schema"] == (frontier.GUIDED_RUN_SCHEMA)
    assert manifest["records"][0]["dynamic_method_gate_decision"]["passed"] is True
    assert frontier.ExactCandidateFrontier(root).inspect()["retained_record_count"] == 1


def test_guided_method_and_v3_source_binding_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    case = _make_guided_run(tmp_path, "guided-tamper")
    report_path = case.run / "run_report.json"
    report = frontier.strict_json(report_path)
    report["guided_method_decision"]["checks"][
        "candidate_has_one_lcp_at_least_4_rollout"
    ] = False
    tampered_report_sha = _write_json(report_path, report)
    with pytest.raises(
        frontier.ExactCandidateFrontierError,
        match="method decision does not authenticate",
    ):
        frontier.authenticate_terminal_candidate(
            case.run,
            expected_run_report_sha256=tampered_report_sha,
            expected_preregistration_sha256=case.preregistration_sha256,
            preregistration_authenticator=case.preregistration_authenticator,
            factor_authenticator=case.factor_authenticator,
            method_gate_authenticator=case.method_gate_authenticator,
        )

    source_case = _make_guided_run(tmp_path, "guided-source-tamper")

    def forged_source_factor(path: Path, lexical_rank: int | None) -> dict:
        authenticated = dict(source_case.factor_authenticator(path, lexical_rank))
        binding = dict(authenticated["guided_source_binding"])
        hashes = dict(binding["source_v2_tensor_sha256s"])
        hashes[next(iter(hashes))] = "f" * 64
        binding["source_v2_tensor_sha256s"] = hashes
        authenticated["guided_source_binding"] = binding
        return authenticated

    with pytest.raises(
        frontier.ExactCandidateFrontierError,
        match="v3 factor sources",
    ):
        frontier.authenticate_terminal_candidate(
            source_case.run,
            expected_run_report_sha256=source_case.run_report_sha256,
            expected_preregistration_sha256=source_case.preregistration_sha256,
            preregistration_authenticator=(source_case.preregistration_authenticator),
            factor_authenticator=forged_source_factor,
            method_gate_authenticator=source_case.method_gate_authenticator,
        )


def test_guided_terminal_report_schema_rejects_unknown_fields(
    tmp_path: Path,
) -> None:
    case = _make_guided_run(tmp_path, "guided-report-schema")
    report_path = case.run / "run_report.json"
    report = frontier.strict_json(report_path)
    report["parent_state"] = {"must_not_be_retained": True}
    report_sha = _write_json(report_path, report)
    with pytest.raises(
        frontier.ExactCandidateFrontierError,
        match="exact v1 field schema",
    ):
        frontier.authenticate_terminal_candidate(
            case.run,
            expected_run_report_sha256=report_sha,
            expected_preregistration_sha256=case.preregistration_sha256,
            preregistration_authenticator=case.preregistration_authenticator,
            factor_authenticator=case.factor_authenticator,
            method_gate_authenticator=case.method_gate_authenticator,
        )


@pytest.mark.parametrize(
    "binding_field",
    (
        "inherited_installation_sha256",
        "inherited_factor_state_sha256_at_install",
        "adapter_site_indices_sha256",
        "audit_site_indices_sha256",
    ),
)
def test_guided_runtime_installation_binding_tampering_fails_closed(
    tmp_path: Path,
    binding_field: str,
) -> None:
    case = _make_guided_run(tmp_path, f"guided-install-{binding_field}")

    def forged_factor(path: Path, lexical_rank: int | None) -> dict:
        # The factor authenticator is the trust boundary for the inner v3
        # envelope.  Returning a new value here models a pack whose source
        # binding and factor-state hashes were both recomputed around the
        # forged installation-domain SHA.
        authenticated = dict(case.factor_authenticator(path, lexical_rank))
        binding = dict(authenticated["guided_source_binding"])
        binding[binding_field] = _sha(f"forged:{binding_field}")
        authenticated["guided_source_binding"] = binding
        return authenticated

    with pytest.raises(
        frontier.ExactCandidateFrontierError,
        match="v3 factor sources",
    ):
        frontier.authenticate_terminal_candidate(
            case.run,
            expected_run_report_sha256=case.run_report_sha256,
            expected_preregistration_sha256=case.preregistration_sha256,
            preregistration_authenticator=case.preregistration_authenticator,
            factor_authenticator=forged_factor,
            method_gate_authenticator=case.method_gate_authenticator,
        )


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("factor_count", 25),
        ("factor_parameters", 28_868_607),
        ("lexical_rank", 31),
    ),
)
def test_guided_exact_v3_topology_totals_fail_closed(
    tmp_path: Path,
    field: str,
    forged_value: int,
) -> None:
    case = _make_guided_run(tmp_path, f"guided-total-{field}")

    def forged_factor(path: Path, lexical_rank: int | None) -> dict:
        authenticated = dict(case.factor_authenticator(path, lexical_rank))
        authenticated[field] = forged_value
        return authenticated

    with pytest.raises(
        frontier.ExactCandidateFrontierError,
        match="v3 factor sources",
    ):
        frontier.authenticate_terminal_candidate(
            case.run,
            expected_run_report_sha256=case.run_report_sha256,
            expected_preregistration_sha256=case.preregistration_sha256,
            preregistration_authenticator=case.preregistration_authenticator,
            factor_authenticator=forged_factor,
            method_gate_authenticator=case.method_gate_authenticator,
        )


def test_guided_accepted_child_still_requires_independent_fresh_reload(
    tmp_path: Path,
) -> None:
    case = _make_guided_run(tmp_path, "guided-accepted", status="accepted")
    standalone = case.candidate()
    assert standalone.truth_label == "accepted-standalone"
    assert standalone.fresh_reload_verified is True
    assert standalone.dynamic_method_gate_decision is not None
    assert standalone.dynamic_method_gate_decision["passed"] is True

    factor_fallback = case.candidate(include_child_authenticator=False)
    assert factor_fallback.truth_label == "research-factor-only-candidate"
    assert factor_fallback.source_artifact.name == "selected_guided_factor_pack.pt"
    assert factor_fallback.deployment_accepted is False


def test_legacy_and_guided_reports_remain_compatible_and_pruning_is_plan_only(
    tmp_path: Path,
) -> None:
    manager = frontier.ExactCandidateFrontier(tmp_path / "mixed-frontier")
    legacy = _make_run(tmp_path, "legacy-compatible")
    guided = _make_guided_run(tmp_path, "guided-compatible")
    manager.register(legacy.candidate())
    manager.register(guided.candidate())
    inspection = manager.inspect()
    assert inspection["retained_record_count"] == 2
    assert sorted(inspection["retained_per_contract"].values()) == [1, 1]
    with pytest.raises(frontier.ExactCandidateFrontierError, match="unsupported"):
        manager.apply_pending_prune()
    assert legacy.run.is_dir() and guided.run.is_dir()


def test_cli_strictly_authenticates_outer_guided_envelope_and_inner_v3_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from neural_forge.hf_target_native_behavior_distillation import (
        GuidedFactorSourceBinding,
    )

    spec = importlib.util.spec_from_file_location("retention_cli_guided_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inherited_names = {
        *(
            f"qkvo.{role}_{side}"
            for role in ("q", "k", "v", "o")
            for side in ("left", "right")
        ),
        *(
            f"gud.{role}_{side}"
            for role in ("gate", "up", "down")
            for side in ("left", "right")
        ),
        *(
            f"lexical.{role}_{side}"
            for role in ("embedding", "lm_head")
            for side in ("left", "right")
        ),
    }

    def tensor_sha(value: torch.Tensor) -> str:
        return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()

    fake_stage = SimpleNamespace(
        QKVO_SITE_COUNT=1,
        QKVO_RANK=1,
        TARGET_HIDDEN_SIZE=1,
        TARGET_QUERY_GROUP_WIDTH=1,
        TARGET_HEAD_DIM=1,
        _expected_continuation_factor_shapes=lambda *, lexical_rank: {
            name: (1,) for name in inherited_names
        },
        _factor_tensor_sha256=tensor_sha,
        _canonical_sha256=frontier.canonical_sha256,
    )

    def require_sha(value, name):
        return frontier.require_sha256(value, name)

    fake_guided = SimpleNamespace(
        FACTOR_ENVELOPE_SCHEMA=(
            "neural-forge-rejected-target-native-guided-factor-pack-v1"
        ),
        TRAINER_FACTOR_PACK_V3_SCHEMA=("neural-forge-target-native-factor-pack-v3"),
        GUIDED_TOTAL_FACTOR_COUNT=26,
        GUIDED_TOTAL_FACTOR_PARAMETERS=26,
        GUIDED_TRAINER_MODULE=("neural_forge.hf_target_native_behavior_distillation"),
        stage=fake_stage,
        _require_sha256=require_sha,
        __file__=str(SCRIPT),
    )
    monkeypatch.setattr(module, "_load_guided_runner", lambda: fake_guided)
    source_hashes = {name: _sha(f"source:{name}") for name in inherited_names}
    basis_hashes = {role: _sha(f"basis:{role}") for role in ("q", "k", "v", "o")}
    binding = GuidedFactorSourceBinding(
        source_v2_factor_state_sha256=_sha("source-state"),
        source_v2_tensor_sha256s=source_hashes,
        audit_report_sha256=_sha("audit"),
        basis_envelope_file_sha256=_sha("basis-envelope"),
        basis_payload_sha256=_sha("basis-payload"),
        basis_manifest_sha256=_sha("basis-manifest"),
        source_basis_tensor_sha256s=basis_hashes,
        inherited_installation_sha256=_sha("inherited-install"),
        inherited_factor_state_sha256_at_install=_sha("inherited-state"),
        adapter_site_indices_sha256=_sha("adapter-sites"),
        audit_site_indices_sha256=_sha("audit-sites"),
    )
    guided_names = {
        f"guided_qkvo.{role}_{side}"
        for role in ("q", "k", "v", "o")
        for side in ("left", "right")
    }
    names = tuple(sorted(inherited_names | guided_names))
    test_shapes = {
        name: ((1, 1, 1) if name in guided_names else (1,)) for name in names
    }
    tensors = {
        name: torch.full(test_shapes[name], float(index + 1), dtype=torch.float32)
        for index, name in enumerate(names)
    }
    rows = [
        {
            "name": name,
            "shape": list(test_shapes[name]),
            "dtype": "torch.float32",
            "tensor_sha256": tensor_sha(tensors[name]),
        }
        for name in names
    ]
    state_preimage = {
        "schema": fake_guided.TRAINER_FACTOR_PACK_V3_SCHEMA,
        "lexical_rank": 32,
        "factors": rows,
        "guided_source_binding": binding.as_dict(),
        "guided_source_binding_sha256": binding.binding_sha256,
        "inherited_product_regrouped": False,
        "guided_product_regrouped": False,
    }
    payload = {
        **state_preimage,
        "factor_count": 26,
        "factor_state_sha256": frontier.canonical_sha256(state_preimage),
        "parent_state_included": False,
        "native_child_state_included": False,
        "optimizer_state_included": False,
        "factor_tensors": tensors,
    }
    envelope = {
        "schema": fake_guided.FACTOR_ENVELOPE_SCHEMA,
        "reason": "selected",
        "preregistration_sha256": _sha("prereg"),
        "evaluation_contract_sha256": _sha("evaluation"),
        "source_factor_state_sha256": binding.source_v2_factor_state_sha256,
        "audit_report_self_sha256": binding.audit_report_sha256,
        "audit_report_canonical_sha256": _sha("audit-canonical"),
        "basis_envelope_file_sha256": binding.basis_envelope_file_sha256,
        "basis_domain_artifact_sha256": _sha("basis-domain"),
        "basis_payload_sha256": binding.basis_payload_sha256,
        "factor_pack": payload,
        "contains_model_checkpoint": False,
        "contains_parent_parameters": False,
        "contains_activations_or_hidden_states": False,
    }
    authenticated = module._guided_source_pinned_factor_authentication(
        envelope, 32, torch
    )
    assert authenticated["factor_count"] == 26
    assert authenticated["factor_parameters"] == 26
    assert authenticated["guided_source_binding"] == binding.as_dict()

    tampered = copy.deepcopy(envelope)
    tampered["factor_pack"]["factor_tensors"][names[0]][0] += 1.0
    with pytest.raises(
        frontier.ExactCandidateFrontierError,
        match="manifest authentication",
    ):
        module._guided_source_pinned_factor_authentication(tampered, 32, torch)

    oversized = copy.deepcopy(envelope)
    oversized_name = names[0]
    logical = oversized["factor_pack"]["factor_tensors"][oversized_name]
    backing = torch.empty(logical.numel() + 1, dtype=torch.float32)
    oversized_tensor = backing[: logical.numel()].view(logical.shape)
    oversized_tensor.copy_(logical)
    assert oversized_tensor.untyped_storage().nbytes() > (
        oversized_tensor.numel() * oversized_tensor.element_size()
    )
    oversized["factor_pack"]["factor_tensors"][oversized_name] = oversized_tensor
    oversized_rows = oversized["factor_pack"]["factors"]
    oversized_row = next(row for row in oversized_rows if row["name"] == oversized_name)
    oversized_row["tensor_sha256"] = tensor_sha(oversized_tensor)
    oversized_preimage = {
        "schema": fake_guided.TRAINER_FACTOR_PACK_V3_SCHEMA,
        "lexical_rank": 32,
        "factors": oversized_rows,
        "guided_source_binding": binding.as_dict(),
        "guided_source_binding_sha256": binding.binding_sha256,
        "inherited_product_regrouped": False,
        "guided_product_regrouped": False,
    }
    oversized["factor_pack"]["factor_state_sha256"] = frontier.canonical_sha256(
        oversized_preimage
    )
    with pytest.raises(
        frontier.ExactCandidateFrontierError,
        match="exact owned storage",
    ):
        module._guided_source_pinned_factor_authentication(oversized, 32, torch)
