#!/usr/bin/env python3
"""Build and score the exact-50% cached-causal LFM2 value fold.

The parent is loaded locally only to pack its immutable state.  Candidate
allocation consumes an already sealed cached-causal frontier; it performs no
parent forward or response generation.  The independent 64-case parent
responses are loaded from their existing sealed envelope and the freshly
reloaded candidate is the only model that generates during scoring.

Open-ended cases use paired parent/candidate ratings.  Abilities absent in the
parent are neutralized, absolute scores remain diagnostics, and promotion
requires at least 90% demonstrated-capability retention with no clear manual
regression.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from experiments import evaluate_lfm2_intelligence_holdout as holdout_eval
    from experiments import evaluate_lfm2_intelligence_retention as posthoc
    from experiments import run_lfm2_residual_attention_fold as direct_lfm2
    from experiments import run_lfm2_value_fold_control as control_runner
except ImportError:  # Direct execution from the experiments directory.
    import evaluate_lfm2_intelligence_holdout as holdout_eval
    import evaluate_lfm2_intelligence_retention as posthoc
    import run_lfm2_residual_attention_fold as direct_lfm2
    import run_lfm2_value_fold_control as control_runner

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_causal_value_allocator import (
    export_lfm2_causal_value_fold,
    inspect_lfm2_causal_value_fold,
    load_lfm2_causal_value_fold,
    load_lfm2_causal_value_tokenizer,
)
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_intelligence_holdout import (
    load_lfm2_intelligence_holdout_parent_envelope,
    seal_lfm2_intelligence_holdout_candidate_outputs,
)
from neural_forge.hf_lfm2_parent_relative_manual_gate import (
    MINIMUM_RETENTION,
    build_parent_relative_manual_packet,
    evaluate_parent_relative_manual_retention,
)


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

RUNNER_SCHEMA = "neural-forge-lfm2-causal-value-allocator-run-v1"
TARGET_REPORT_SCHEMA = "neural-forge-lfm2-causal-value-target-report-v1"


class Lfm2CausalValueRunnerError(RuntimeError):
    """A local identity, sealed input, or parent-relative gate failed."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CausalValueRunnerError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _read_json(
    path: Path,
    *,
    expected_file_sha256: str,
    name: str,
) -> dict[str, Any]:
    if control_runner._file_sha256(path) != _digest(
        expected_file_sha256,
        f"{name}_file_sha256",
    ):
        raise Lfm2CausalValueRunnerError(f"{name} file SHA-256 changed.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Lfm2CausalValueRunnerError(f"{name} is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise Lfm2CausalValueRunnerError(f"{name} must contain an object.")
    return value


def _probe(
    checkpoint: Path,
    tokenizer: Any,
    prompt: str,
    *,
    device: str,
) -> tuple[torch.Tensor, str]:
    model = load_lfm2_causal_value_fold(checkpoint, device=device)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    runtime_device = control_runner._runtime_device(model)
    input_ids = encoded["input_ids"].to(runtime_device, dtype=torch.long)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(runtime_device, dtype=torch.long)
    with torch.inference_mode():
        logits = (
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
            .logits.detach()
            .to("cpu")
        )
    manifest_sha = model.neural_forge_load_report["manifest_sha256"]
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits, manifest_sha


def _parent_relative_decision(
    intelligence: Mapping[str, Any],
    manual_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    gate = intelligence.get("core_intelligence_gate")
    policy = intelligence.get("policy")
    if not isinstance(gate, Mapping) or not isinstance(policy, Mapping):
        raise Lfm2CausalValueRunnerError(
            "Automatic intelligence report lacks its sealed gate/policy."
        )
    components = gate.get("components")
    required = policy.get("required_components")
    if (
        not isinstance(components, list)
        or not isinstance(required, list)
        or not required
    ):
        raise Lfm2CausalValueRunnerError(
            "Automatic intelligence component coverage is invalid."
        )
    by_name = {
        str(value.get("name")): value
        for value in components
        if isinstance(value, Mapping)
    }
    if not set(required).issubset(by_name):
        raise Lfm2CausalValueRunnerError(
            "Automatic intelligence report omits a required component."
        )
    automatic_reasons = []
    headline_values = {}
    for name in required:
        component = by_name[name]
        retention = component.get("retention")
        if (
            isinstance(retention, bool)
            or not isinstance(retention, (int, float))
            or not 0.0 <= float(retention) <= 1.0
        ):
            automatic_reasons.append(f"{name}:missing_reliable_coverage")
        else:
            headline_values[name] = float(retention)
            if component.get("passed") is not True:
                automatic_reasons.append(f"{name}:threshold_failed")
    task_absolute = intelligence.get("parent_relative_task_retention", {}).get(
        "absolute_accuracy_retention"
    )
    if isinstance(task_absolute, (int, float)) and not isinstance(task_absolute, bool):
        headline_values["task_absolute_accuracy_retention"] = float(task_absolute)
    else:
        automatic_reasons.append("task_absolute_accuracy:missing")
    manual_pending = manual_report is None
    manual_passed = bool(
        manual_report is not None and manual_report.get("passed") is True
    )
    if manual_report is not None:
        retention = manual_report.get("retention")
        if (
            isinstance(retention, bool)
            or not isinstance(retention, (int, float))
            or not 0.0 <= float(retention) <= 1.0
        ):
            raise Lfm2CausalValueRunnerError(
                "Parent-relative manual report retention is invalid."
            )
        headline_values["parent_relative_manual"] = float(retention)
    reasons = list(automatic_reasons)
    if manual_pending:
        reasons.append("parent_relative_manual:pending")
    elif not manual_passed:
        reasons.append("parent_relative_manual:threshold_or_regression_failed")
    promoted = not reasons
    lower_bound = min(headline_values.values()) if promoted else None
    return {
        "promoted": promoted,
        "status": (
            "promoted"
            if promoted
            else (
                "pending_parent_relative_manual"
                if manual_pending and not automatic_reasons
                else "rejected"
            )
        ),
        "reasons": reasons,
        "certified_parent_relative_lower_bound": lower_bound,
        "headline_inputs": headline_values,
        "automatic_components_passed": not automatic_reasons,
        "parent_relative_manual_status": (
            "pending"
            if manual_pending
            else ("accepted" if manual_passed else "rejected")
        ),
        "minimum_component_retention": MINIMUM_RETENTION,
        "absolute_manual_scores_are_diagnostics_only": True,
        "abilities_absent_in_parent_are_neutralized": True,
        "clear_manual_regressions_are_rejected": True,
    }


@dataclass(frozen=True, slots=True)
class CausalValueRunConfig:
    parent_model_directory: Path
    tokenizer_directory: Path
    output_root: Path
    cached_causal_frontier: Path
    expected_cached_causal_frontier_file_sha256: str
    parent_envelope: Path
    expected_parent_envelope_file_sha256: str
    expected_parent_envelope_sha256: str
    expected_manifest_sha256: str
    expected_parent_state_fingerprint: str
    expected_parent_model_fingerprint: str
    expected_tokenizer_fingerprint: str
    calibration_token_selection: Path | None = None
    expected_calibration_token_selection_file_sha256: str | None = None
    parent_relative_manual_ratings: Path | None = None
    expected_parent_relative_manual_ratings_file_sha256: str | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    group_size: int = 256
    analysis_chunk_groups: int = 4096

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_model_directory",
            control_runner._local_directory(
                self.parent_model_directory,
                "parent_model_directory",
            ),
        )
        object.__setattr__(
            self,
            "tokenizer_directory",
            control_runner._local_directory(
                self.tokenizer_directory,
                "tokenizer_directory",
            ),
        )
        for field in ("cached_causal_frontier", "parent_envelope"):
            object.__setattr__(
                self,
                field,
                control_runner._local_file(getattr(self, field), field),
            )
        output = Path(self.output_root).expanduser().absolute()
        if output.exists():
            raise Lfm2CausalValueRunnerError(
                f"Refusing to replace existing output root {output}."
            )
        object.__setattr__(self, "output_root", output)
        for field in (
            "expected_cached_causal_frontier_file_sha256",
            "expected_parent_envelope_file_sha256",
            "expected_parent_envelope_sha256",
            "expected_manifest_sha256",
            "expected_parent_state_fingerprint",
            "expected_parent_model_fingerprint",
            "expected_tokenizer_fingerprint",
        ):
            _digest(getattr(self, field), field)
        for path_field, digest_field in (
            (
                "calibration_token_selection",
                "expected_calibration_token_selection_file_sha256",
            ),
            (
                "parent_relative_manual_ratings",
                "expected_parent_relative_manual_ratings_file_sha256",
            ),
        ):
            path = getattr(self, path_field)
            digest = getattr(self, digest_field)
            if (path is None) != (digest is None):
                raise Lfm2CausalValueRunnerError(
                    f"{path_field} and {digest_field} must be supplied together."
                )
            if path is not None:
                object.__setattr__(
                    self,
                    path_field,
                    control_runner._local_file(path, path_field),
                )
                _digest(digest, digest_field)
        if self.dtype not in {"float16", "bfloat16"}:
            raise Lfm2CausalValueRunnerError("dtype must be float16 or bfloat16.")


def run_lfm2_causal_value_allocator(
    config: CausalValueRunConfig,
) -> dict[str, Any]:
    if not isinstance(config, CausalValueRunConfig):
        raise TypeError("config must be CausalValueRunConfig.")
    frontier_document = _read_json(
        config.cached_causal_frontier,
        expected_file_sha256=(config.expected_cached_causal_frontier_file_sha256),
        name="cached_causal_frontier",
    )
    parent_document = _read_json(
        config.parent_envelope,
        expected_file_sha256=config.expected_parent_envelope_file_sha256,
        name="parent_envelope",
    )
    envelope = load_lfm2_intelligence_holdout_parent_envelope(
        parent_document,
        expected_envelope_sha256=config.expected_parent_envelope_sha256,
        expected_manifest_sha256=config.expected_manifest_sha256,
        expected_parent_state_fingerprint=(config.expected_parent_state_fingerprint),
        expected_parent_model_fingerprint=(config.expected_parent_model_fingerprint),
        expected_tokenizer_fingerprint=config.expected_tokenizer_fingerprint,
    )
    tokenizer_fingerprint, _files = posthoc.fingerprint_bound_local_tokenizer(
        config.tokenizer_directory
    )
    if tokenizer_fingerprint != config.expected_tokenizer_fingerprint:
        raise Lfm2CausalValueRunnerError(
            "Local tokenizer differs from the sealed parent tokenizer."
        )
    special = posthoc.load_bound_special_token_ids(
        config.tokenizer_directory,
        expected_tokenizer_fingerprint=tokenizer_fingerprint,
    )
    eos_token_ids = tuple(special["special_token_ids"]["eos"])
    if eos_token_ids != envelope.eos_token_ids:
        raise Lfm2CausalValueRunnerError(
            "Tokenizer EOS IDs differ from the parent envelope."
        )
    calibration_document = None
    if config.calibration_token_selection is not None:
        calibration_document = _read_json(
            config.calibration_token_selection,
            expected_file_sha256=(
                config.expected_calibration_token_selection_file_sha256
            ),
            name="calibration_token_selection",
        )
    ratings_document = None
    if config.parent_relative_manual_ratings is not None:
        ratings_document = _read_json(
            config.parent_relative_manual_ratings,
            expected_file_sha256=(
                config.expected_parent_relative_manual_ratings_file_sha256
            ),
            name="parent_relative_manual_ratings",
        )

    config.output_root.mkdir(parents=True)
    checkpoint = config.output_root / "checkpoint"
    parent = direct_lfm2.load_local_lfm2_model(
        config.parent_model_directory,
        device=config.device,
        dtype=direct_lfm2._torch_dtype(config.dtype),
    )
    parent.eval()
    parent_state_before = direct_lfm2.model_state_sha256(parent)
    parent_identity = fingerprint_lfm2_parent(parent).identity_sha256
    if (
        parent_state_before != config.expected_parent_state_fingerprint
        or parent_identity != config.expected_parent_model_fingerprint
        or any(parameter.grad is not None for parameter in parent.parameters())
    ):
        raise Lfm2CausalValueRunnerError(
            "Loaded local parent differs from its sealed identity."
        )
    manifest = export_lfm2_causal_value_fold(
        parent,
        checkpoint,
        tokenizer_directory=config.tokenizer_directory,
        frontier_document=frontier_document,
        eos_token_ids=eos_token_ids,
        calibration_token_selection=calibration_document,
        group_size=config.group_size,
        analysis_chunk_groups=config.analysis_chunk_groups,
    )
    if direct_lfm2.model_state_sha256(parent) != parent_state_before or any(
        parameter.grad is not None for parameter in parent.parameters()
    ):
        raise Lfm2CausalValueRunnerError(
            "Packing mutated the parent state or created gradients."
        )
    del parent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = load_lfm2_causal_value_tokenizer(checkpoint)
    if (
        tokenizer.neural_forge_tokenizer_fingerprint
        != config.expected_tokenizer_fingerprint
    ):
        raise Lfm2CausalValueRunnerError(
            "Fresh checkpoint tokenizer differs from the sealed tokenizer."
        )
    probe_prompt = envelope.manifest.cases[0].prompt
    first_logits, first_manifest = _probe(
        checkpoint,
        tokenizer,
        probe_prompt,
        device=config.device,
    )
    candidate = load_lfm2_causal_value_fold(
        checkpoint,
        device=config.device,
    )
    encoded = tokenizer(
        probe_prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    runtime_device = control_runner._runtime_device(candidate)
    input_ids = encoded["input_ids"].to(runtime_device, dtype=torch.long)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(runtime_device, dtype=torch.long)
    with torch.inference_mode():
        second_logits = (
            candidate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
            .logits.detach()
            .to("cpu")
        )
    if first_manifest != manifest["manifest_sha256"] or not torch.equal(
        first_logits,
        second_logits,
    ):
        raise Lfm2CausalValueRunnerError(
            "Two fresh causal-value reloads are not logit-identical."
        )
    candidate_rows = control_runner._generate_candidate_rows(
        candidate,
        tokenizer,
        envelope.manifest,
    )
    candidate_document = seal_lfm2_intelligence_holdout_candidate_outputs(
        envelope.manifest,
        candidate_rows,
        candidate_state_fingerprint=manifest["manifest_sha256"],
        tokenizer_fingerprint=config.expected_tokenizer_fingerprint,
        eos_token_ids=eos_token_ids,
    )
    candidate_path = config.output_root / "candidate_outputs.json"
    control_runner._write_new_json(candidate_path, candidate_document)
    candidate_file_sha = control_runner._file_sha256(candidate_path)
    del candidate
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    intelligence_path = config.output_root / "intelligence_report.json"
    standard_packet_path = config.output_root / "manual_review_packet.json"
    intelligence = holdout_eval.evaluate_lfm2_intelligence_holdout(
        holdout_eval.HoldoutEvaluationConfig(
            parent_envelope=config.parent_envelope,
            expected_parent_envelope_file_sha256=(
                config.expected_parent_envelope_file_sha256
            ),
            expected_parent_envelope_sha256=(config.expected_parent_envelope_sha256),
            expected_manifest_sha256=config.expected_manifest_sha256,
            expected_parent_state_fingerprint=(
                config.expected_parent_state_fingerprint
            ),
            expected_parent_model_fingerprint=(
                config.expected_parent_model_fingerprint
            ),
            tokenizer_directory=checkpoint,
            expected_tokenizer_fingerprint=(config.expected_tokenizer_fingerprint),
            candidate_report=candidate_path,
            expected_candidate_report_sha256=candidate_file_sha,
            report_path=intelligence_path,
            manual_review_packet_path=standard_packet_path,
            expected_candidate_outputs_sha256=candidate_document[
                "candidate_outputs_sha256"
            ],
            expected_candidate_state_fingerprint=manifest["manifest_sha256"],
        )
    )
    standard_packet = json.loads(standard_packet_path.read_text(encoding="utf-8"))
    parent_relative_packet = build_parent_relative_manual_packet(standard_packet)
    parent_relative_packet_path = (
        config.output_root / "parent_relative_manual_packet.json"
    )
    control_runner._write_new_json(
        parent_relative_packet_path,
        parent_relative_packet,
    )
    manual_report = None
    if ratings_document is not None:
        manual_report = evaluate_parent_relative_manual_retention(
            parent_relative_packet,
            ratings_document,
        )
        control_runner._write_new_json(
            config.output_root / "parent_relative_manual_report.json",
            manual_report,
        )
    decision = _parent_relative_decision(intelligence, manual_report)
    accounting = inspect_lfm2_causal_value_fold(checkpoint)
    core = {
        "schema": TARGET_REPORT_SCHEMA,
        "status": decision["status"],
        "promoted": decision["promoted"],
        "decision": decision,
        "checkpoint": accounting,
        "fresh_reload_equivalence": {
            "reload_count": 2,
            "probe_logits_bitwise_equal": True,
            "manifest_sha256": manifest["manifest_sha256"],
            "exact_source_dtype_overlays_active": True,
        },
        "holdout": {
            "suite": "sealed-independent-64-case-lfm2-holdout",
            "case_count": len(candidate_rows),
            "candidate_outputs_sha256": candidate_document["candidate_outputs_sha256"],
            "candidate_output_file_sha256": candidate_file_sha,
            "automatic_intelligence_report_sha256": intelligence["report_sha256"],
            "standard_absolute_manual_status": intelligence["manual_review"]["status"],
            "standard_absolute_manual_promotion_eligible": False,
            "parent_relative_manual_packet_sha256": parent_relative_packet[
                "packet_sha256"
            ],
            "parent_relative_manual_report_sha256": (
                None if manual_report is None else manual_report["report_sha256"]
            ),
            "parent_model_loaded_during_scoring": False,
            "new_parent_responses": 0,
        },
        "parent_binding": {
            "parent_state_fingerprint": (config.expected_parent_state_fingerprint),
            "parent_model_fingerprint": (config.expected_parent_model_fingerprint),
            "parent_envelope_sha256": config.expected_parent_envelope_sha256,
            "cached_causal_frontier_file_sha256": (
                config.expected_cached_causal_frontier_file_sha256
            ),
            "frontier_sha256": accounting["frontier_sha256"],
            "tokenizer_fingerprint": config.expected_tokenizer_fingerprint,
            "parent_state_unchanged_by_export": True,
        },
        "execution_audit": {
            "network_access": False,
            "hub_loader_used": False,
            "training_used": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "parent_forward_calls_during_allocation": 0,
            "parent_generate_invocations": 0,
            "candidate_generate_invocations": len(candidate_rows),
            "independent_holdout_used_for_allocation": False,
        },
    }
    report = {**core, "report_sha256": canonical_sha256(core)}
    control_runner._write_new_json(config.output_root / "report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and score the exact-50% cached-causal mixed-precision LFM2 "
            "checkpoint from local models and sealed caches only."
        )
    )
    parser.add_argument("--parent-model-directory", type=Path, required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cached-causal-frontier", type=Path, required=True)
    parser.add_argument(
        "--expected-cached-causal-frontier-file-sha256",
        required=True,
    )
    parser.add_argument("--parent-envelope", type=Path, required=True)
    parser.add_argument(
        "--expected-parent-envelope-file-sha256",
        required=True,
    )
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-parent-state-fingerprint", required=True)
    parser.add_argument("--expected-parent-model-fingerprint", required=True)
    parser.add_argument("--expected-tokenizer-fingerprint", required=True)
    parser.add_argument("--calibration-token-selection", type=Path)
    parser.add_argument(
        "--expected-calibration-token-selection-file-sha256",
    )
    parser.add_argument("--parent-relative-manual-ratings", type=Path)
    parser.add_argument(
        "--expected-parent-relative-manual-ratings-file-sha256",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--analysis-chunk-groups", type=int, default=4096)
    return parser


def config_from_args(args: argparse.Namespace) -> CausalValueRunConfig:
    return CausalValueRunConfig(**vars(args))


def main(argv: Sequence[str] | None = None) -> int:
    report = run_lfm2_causal_value_allocator(
        config_from_args(build_parser().parse_args(argv))
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "promoted": report["promoted"],
                "report_sha256": report["report_sha256"],
                "compressed_checkpoint_bytes": report["checkpoint"][
                    "compressed_checkpoint_bytes"
                ],
                "parent_relative_lower_bound": report["decision"][
                    "certified_parent_relative_lower_bound"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
