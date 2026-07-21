#!/usr/bin/env python3
"""Build and score the exact-50% LFM2 protected-weight-error ablation.

This runner requires no cached-causal frontier.  It loads the local parent only
to pack immutable weights, then releases it before scoring.  Candidate-only
generation is compared with the existing sealed independent64 parent envelope.
The allocation is explicitly labeled noncausal and uses no parent forward,
parent generation, optimizer, backward call, or training operation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from experiments import evaluate_lfm2_intelligence_holdout as holdout_eval
    from experiments import evaluate_lfm2_intelligence_retention as posthoc
    from experiments import run_lfm2_causal_value_allocator as shared_runner
    from experiments import run_lfm2_residual_attention_fold as direct_lfm2
    from experiments import run_lfm2_value_fold_control as control_runner
except ImportError:  # Direct execution from the experiments directory.
    import evaluate_lfm2_intelligence_holdout as holdout_eval
    import evaluate_lfm2_intelligence_retention as posthoc
    import run_lfm2_causal_value_allocator as shared_runner
    import run_lfm2_residual_attention_fold as direct_lfm2
    import run_lfm2_value_fold_control as control_runner

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_intelligence_holdout import (
    load_lfm2_intelligence_holdout_parent_envelope,
    seal_lfm2_intelligence_holdout_candidate_outputs,
)
from neural_forge.hf_lfm2_parent_relative_manual_gate import (
    evaluate_parent_relative_manual_retention,
)
from neural_forge.hf_lfm2_protected_weight_error_allocator import (
    POLICY_ID,
    export_lfm2_protected_weight_error_fold,
    inspect_lfm2_protected_weight_error_fold,
    load_lfm2_protected_weight_error_fold,
    load_lfm2_protected_weight_error_tokenizer,
)


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

LEGACY_RUNNER_SCHEMA = "neural-forge-lfm2-protected-weight-error-run-v1"
RUNNER_SCHEMA = "neural-forge-lfm2-protected-weight-error-run-v2"
LEGACY_TARGET_REPORT_SCHEMA = (
    "neural-forge-lfm2-protected-weight-error-target-report-v1"
)
TARGET_REPORT_SCHEMA = "neural-forge-lfm2-protected-weight-error-target-report-v2"


class Lfm2ProtectedWeightErrorRunnerError(RuntimeError):
    """A local source, sealed holdout, reload, or review binding failed."""


def _digest(value: Any, name: str) -> str:
    try:
        return shared_runner._digest(value, name)
    except shared_runner.Lfm2CausalValueRunnerError as exc:
        raise Lfm2ProtectedWeightErrorRunnerError(str(exc)) from exc


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
        raise Lfm2ProtectedWeightErrorRunnerError(f"{name} file SHA-256 changed.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Lfm2ProtectedWeightErrorRunnerError(f"{name} is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise Lfm2ProtectedWeightErrorRunnerError(f"{name} must contain an object.")
    return value


def _probe(
    checkpoint: Path,
    tokenizer: Any,
    prompt: str,
    *,
    device: str,
) -> tuple[torch.Tensor, str]:
    model = load_lfm2_protected_weight_error_fold(
        checkpoint,
        device=device,
    )
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


@dataclass(frozen=True, slots=True)
class ProtectedWeightErrorRunConfig:
    parent_model_directory: Path
    tokenizer_directory: Path
    output_root: Path
    parent_envelope: Path
    expected_parent_envelope_file_sha256: str
    expected_parent_envelope_sha256: str
    expected_manifest_sha256: str
    expected_parent_state_fingerprint: str
    expected_parent_model_fingerprint: str
    expected_tokenizer_fingerprint: str
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
        object.__setattr__(
            self,
            "parent_envelope",
            control_runner._local_file(
                self.parent_envelope,
                "parent_envelope",
            ),
        )
        output = Path(self.output_root).expanduser().absolute()
        if output.exists():
            raise Lfm2ProtectedWeightErrorRunnerError(
                f"Refusing to replace existing output root {output}."
            )
        object.__setattr__(self, "output_root", output)
        for field in (
            "expected_parent_envelope_file_sha256",
            "expected_parent_envelope_sha256",
            "expected_manifest_sha256",
            "expected_parent_state_fingerprint",
            "expected_parent_model_fingerprint",
            "expected_tokenizer_fingerprint",
        ):
            _digest(getattr(self, field), field)
        ratings = self.parent_relative_manual_ratings
        ratings_sha = self.expected_parent_relative_manual_ratings_file_sha256
        if (ratings is None) != (ratings_sha is None):
            raise Lfm2ProtectedWeightErrorRunnerError(
                "parent_relative_manual_ratings and its expected file SHA-256 "
                "must be supplied together."
            )
        if ratings is not None:
            object.__setattr__(
                self,
                "parent_relative_manual_ratings",
                control_runner._local_file(
                    ratings,
                    "parent_relative_manual_ratings",
                ),
            )
            _digest(
                ratings_sha,
                "expected_parent_relative_manual_ratings_file_sha256",
            )
        if self.dtype not in {"float16", "bfloat16"}:
            raise Lfm2ProtectedWeightErrorRunnerError(
                "dtype must be float16 or bfloat16."
            )


def run_lfm2_protected_weight_error_allocator(
    config: ProtectedWeightErrorRunConfig,
) -> dict[str, Any]:
    if not isinstance(config, ProtectedWeightErrorRunConfig):
        raise TypeError("config must be ProtectedWeightErrorRunConfig.")
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
        raise Lfm2ProtectedWeightErrorRunnerError(
            "Local tokenizer differs from the sealed parent tokenizer."
        )
    special = posthoc.load_bound_special_token_ids(
        config.tokenizer_directory,
        expected_tokenizer_fingerprint=tokenizer_fingerprint,
    )
    eos_token_ids = tuple(special["special_token_ids"]["eos"])
    if eos_token_ids != envelope.eos_token_ids:
        raise Lfm2ProtectedWeightErrorRunnerError(
            "Tokenizer EOS IDs differ from the parent envelope."
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
        raise Lfm2ProtectedWeightErrorRunnerError(
            "Loaded local parent differs from its sealed identity."
        )
    manifest = export_lfm2_protected_weight_error_fold(
        parent,
        checkpoint,
        tokenizer_directory=config.tokenizer_directory,
        eos_token_ids=eos_token_ids,
        group_size=config.group_size,
        analysis_chunk_groups=config.analysis_chunk_groups,
    )
    if direct_lfm2.model_state_sha256(parent) != parent_state_before or any(
        parameter.grad is not None for parameter in parent.parameters()
    ):
        raise Lfm2ProtectedWeightErrorRunnerError(
            "Packing mutated the parent state or created gradients."
        )
    del parent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = load_lfm2_protected_weight_error_tokenizer(checkpoint)
    if (
        tokenizer.neural_forge_tokenizer_fingerprint
        != config.expected_tokenizer_fingerprint
    ):
        raise Lfm2ProtectedWeightErrorRunnerError(
            "Fresh checkpoint tokenizer differs from the sealed tokenizer."
        )
    probe_prompt = envelope.manifest.cases[0].prompt
    first_logits, first_manifest = _probe(
        checkpoint,
        tokenizer,
        probe_prompt,
        device=config.device,
    )
    candidate = load_lfm2_protected_weight_error_fold(
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
        raise Lfm2ProtectedWeightErrorRunnerError(
            "Two fresh protected-weight-error reloads are not logit-identical."
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
    parent_relative_packet = shared_runner._build_exact_all_case_parent_relative_packet(
        standard_packet,
        envelope,
        candidate_rows,
    )
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
    decision = shared_runner._parent_relative_decision(
        intelligence,
        manual_report,
        parent_relative_packet,
    )
    accounting = inspect_lfm2_protected_weight_error_fold(checkpoint)
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
            "parent_relative_manual_packet_case_count": parent_relative_packet[
                "case_count"
            ],
            "parent_relative_manual_review_scope": parent_relative_packet[
                "review_scope"
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
            "tokenizer_fingerprint": config.expected_tokenizer_fingerprint,
            "parent_state_unchanged_by_export": True,
        },
        "allocation_binding": {
            "ablation_label": "protected_weight_error",
            "allocator_policy_id": POLICY_ID,
            "allocation_sha256": accounting["allocation_sha256"],
            "parent_identity_sha256": accounting["parent_identity_sha256"],
            "fixed_causal_multiplier": 1.0,
            "cached_causal_frontier_used": False,
            "causal_evidence_used": False,
            "independent_holdout_used_for_allocation": False,
        },
        "execution_audit": {
            "network_access": False,
            "hub_loader_used": False,
            "training_used": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "frontier_reads": 0,
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
            "Build and score the exact-50% protected-weight-error LFM2 "
            "ablation from a local parent and sealed independent64 cache."
        )
    )
    parser.add_argument("--parent-model-directory", type=Path, required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
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
    parser.add_argument("--parent-relative-manual-ratings", type=Path)
    parser.add_argument(
        "--expected-parent-relative-manual-ratings-file-sha256",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--analysis-chunk-groups", type=int, default=4096)
    return parser


def config_from_args(
    args: argparse.Namespace,
) -> ProtectedWeightErrorRunConfig:
    return ProtectedWeightErrorRunConfig(**vars(args))


def main(argv: Sequence[str] | None = None) -> int:
    report = run_lfm2_protected_weight_error_allocator(
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
                "ablation_label": "protected_weight_error",
                "causal_evidence_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
