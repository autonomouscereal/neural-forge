#!/usr/bin/env python3
"""Build and score the standalone exact-50% uniform-int8 LFM2 control.

The runner is local-only and zero-training.  It exports one all-8-bit
checkpoint, proves two fresh reloads are bitwise logit-identical, generates
candidate responses for all 64 sealed cases, and scores them only against the
existing authenticated parent-response envelope.  It has no code path that
can create or replace parent responses.
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
    from experiments import run_lfm2_causal_value_allocator as causal_runner
    from experiments import run_lfm2_residual_attention_fold as direct_lfm2
    from experiments import run_lfm2_value_fold_control as control_runner
except ImportError:  # Direct execution from the experiments directory.
    import evaluate_lfm2_intelligence_holdout as holdout_eval
    import evaluate_lfm2_intelligence_retention as posthoc
    import run_lfm2_causal_value_allocator as causal_runner
    import run_lfm2_residual_attention_fold as direct_lfm2
    import run_lfm2_value_fold_control as control_runner

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_exact50_all8 import (
    export_lfm2_exact50_all8,
    inspect_lfm2_exact50_all8,
    load_lfm2_exact50_all8,
    load_lfm2_exact50_all8_tokenizer,
)
from neural_forge.hf_lfm2_intelligence_holdout import (
    load_lfm2_intelligence_holdout_parent_envelope,
    seal_lfm2_intelligence_holdout_candidate_outputs,
)


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

LEGACY_RUNNER_SCHEMA = "neural-forge-lfm2-exact50-all8-run-v1"
RUNNER_SCHEMA = "neural-forge-lfm2-exact50-all8-run-v2"
LEGACY_TARGET_REPORT_SCHEMA = "neural-forge-lfm2-exact50-all8-target-report-v1"
TARGET_REPORT_SCHEMA = "neural-forge-lfm2-exact50-all8-target-report-v2"


class Lfm2Exact50All8RunnerError(RuntimeError):
    """A local identity, immutable artifact, or all8 gate failed closed."""


_generate_candidate_rows = control_runner._generate_candidate_rows
_parent_relative_decision = causal_runner._parent_relative_decision


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2Exact50All8RunnerError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _local_directory(value: str | os.PathLike[str], name: str) -> Path:
    try:
        return control_runner._local_directory(value, name)
    except control_runner.Lfm2ValueFoldControlRunnerError as exc:
        raise Lfm2Exact50All8RunnerError(str(exc)) from exc


def _local_file(value: str | os.PathLike[str], name: str) -> Path:
    try:
        return control_runner._local_file(value, name)
    except control_runner.Lfm2ValueFoldControlRunnerError as exc:
        raise Lfm2Exact50All8RunnerError(str(exc)) from exc


def _reload_probe(
    checkpoint: Path,
    tokenizer: Any,
    prompt: str,
    *,
    device: str,
) -> tuple[torch.Tensor, str]:
    model = load_lfm2_exact50_all8(checkpoint, device=device)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(
        control_runner._runtime_device(model),
        dtype=torch.long,
    )
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(
            control_runner._runtime_device(model), dtype=torch.long
        )
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        )
    logits = outputs.logits.detach().to("cpu")
    manifest_sha = model.neural_forge_load_report["manifest_sha256"]
    del outputs, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits, manifest_sha


@dataclass(frozen=True, slots=True)
class Exact50All8RunConfig:
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
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    group_size: int = 256
    analysis_chunk_groups: int = 4096

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_model_directory",
            _local_directory(self.parent_model_directory, "parent_model_directory"),
        )
        object.__setattr__(
            self,
            "tokenizer_directory",
            _local_directory(self.tokenizer_directory, "tokenizer_directory"),
        )
        object.__setattr__(
            self,
            "parent_envelope",
            _local_file(self.parent_envelope, "parent_envelope"),
        )
        output = Path(self.output_root).expanduser().absolute()
        if output.exists():
            raise Lfm2Exact50All8RunnerError(
                f"Refusing to replace existing output root {output}."
            )
        object.__setattr__(self, "output_root", output)
        for name in (
            "expected_parent_envelope_file_sha256",
            "expected_parent_envelope_sha256",
            "expected_manifest_sha256",
            "expected_parent_state_fingerprint",
            "expected_parent_model_fingerprint",
            "expected_tokenizer_fingerprint",
        ):
            _digest(getattr(self, name), name)
        if self.dtype not in {"float16", "bfloat16"}:
            raise Lfm2Exact50All8RunnerError("dtype must be float16 or bfloat16.")
        if (
            isinstance(self.group_size, bool)
            or not isinstance(self.group_size, int)
            or self.group_size < 8
            or self.group_size % 8
        ):
            raise Lfm2Exact50All8RunnerError(
                "group_size must be an integer multiple of eight."
            )
        if (
            isinstance(self.analysis_chunk_groups, bool)
            or not isinstance(self.analysis_chunk_groups, int)
            or self.analysis_chunk_groups < 1
        ):
            raise Lfm2Exact50All8RunnerError("analysis_chunk_groups must be positive.")


def run_lfm2_exact50_all8(config: Exact50All8RunConfig) -> dict[str, Any]:
    """Export, two-reload-probe, and independently score exact50 all8."""

    if not isinstance(config, Exact50All8RunConfig):
        raise TypeError("config must be Exact50All8RunConfig.")
    if control_runner._file_sha256(config.parent_envelope) != (
        config.expected_parent_envelope_file_sha256
    ):
        raise Lfm2Exact50All8RunnerError(
            "Parent envelope file differs from its expected SHA-256."
        )
    parent_document = json.loads(config.parent_envelope.read_text(encoding="utf-8"))
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
        raise Lfm2Exact50All8RunnerError(
            "Local tokenizer differs from the sealed parent tokenizer."
        )
    special = posthoc.load_bound_special_token_ids(
        config.tokenizer_directory,
        expected_tokenizer_fingerprint=tokenizer_fingerprint,
    )
    eos_token_ids = special["special_token_ids"]["eos"]
    if tuple(eos_token_ids) != envelope.eos_token_ids:
        raise Lfm2Exact50All8RunnerError(
            "Tokenizer EOS IDs differ from the sealed parent envelope."
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
    parent_model_identity = fingerprint_lfm2_parent(parent).identity_sha256
    if (
        parent_state_before != config.expected_parent_state_fingerprint
        or parent_model_identity != config.expected_parent_model_fingerprint
        or any(parameter.grad is not None for parameter in parent.parameters())
    ):
        raise Lfm2Exact50All8RunnerError(
            "Loaded local parent differs from the sealed parent or has gradients."
        )
    manifest = export_lfm2_exact50_all8(
        parent,
        checkpoint,
        tokenizer_directory=config.tokenizer_directory,
        group_size=config.group_size,
        analysis_chunk_groups=config.analysis_chunk_groups,
    )
    parent_state_after = direct_lfm2.model_state_sha256(parent)
    if parent_state_after != parent_state_before or any(
        parameter.grad is not None for parameter in parent.parameters()
    ):
        raise Lfm2Exact50All8RunnerError(
            "All8 export mutated the parent state or created gradients."
        )
    del parent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = load_lfm2_exact50_all8_tokenizer(checkpoint)
    if (
        tokenizer.neural_forge_tokenizer_fingerprint
        != config.expected_tokenizer_fingerprint
    ):
        raise Lfm2Exact50All8RunnerError(
            "Compressed checkpoint tokenizer differs from the sealed tokenizer."
        )
    probe_prompt = envelope.manifest.cases[0].prompt
    first_logits, first_manifest = _reload_probe(
        checkpoint,
        tokenizer,
        probe_prompt,
        device=config.device,
    )
    candidate = load_lfm2_exact50_all8(checkpoint, device=config.device)
    encoded = tokenizer(
        probe_prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(
        control_runner._runtime_device(candidate), dtype=torch.long
    )
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(
            control_runner._runtime_device(candidate), dtype=torch.long
        )
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
        first_logits, second_logits
    ):
        raise Lfm2Exact50All8RunnerError(
            "Two fresh all8 reloads are not logit-identical."
        )
    candidate_rows = _generate_candidate_rows(
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
    manual_path = config.output_root / "manual_review_packet.json"
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
            tokenizer_directory=config.tokenizer_directory,
            expected_tokenizer_fingerprint=(config.expected_tokenizer_fingerprint),
            candidate_report=candidate_path,
            expected_candidate_report_sha256=candidate_file_sha,
            report_path=intelligence_path,
            manual_review_packet_path=manual_path,
            expected_candidate_outputs_sha256=candidate_document[
                "candidate_outputs_sha256"
            ],
            expected_candidate_state_fingerprint=manifest["manifest_sha256"],
        )
    )
    standard_packet = json.loads(manual_path.read_text(encoding="utf-8"))
    parent_relative_packet = causal_runner._build_exact_all_case_parent_relative_packet(
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
    decision = _parent_relative_decision(
        intelligence,
        None,
        parent_relative_packet,
    )
    accounting = inspect_lfm2_exact50_all8(checkpoint)
    target_core = {
        "schema": TARGET_REPORT_SCHEMA,
        "status": decision["status"],
        "promoted": decision["promoted"],
        "decision": decision,
        "control_label": accounting["control_label"],
        "requested_reduction": 0.5,
        "structural_parameter_count_compaction": False,
        "checkpoint": accounting,
        "fresh_reload_equivalence": {
            "reload_count": 2,
            "probe_logits_bitwise_equal": True,
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "holdout": {
            "suite": "sealed-independent-64-case-lfm2-holdout",
            "case_count": len(candidate_rows),
            "candidate_outputs_sha256": candidate_document["candidate_outputs_sha256"],
            "candidate_output_file_sha256": candidate_file_sha,
            "automatic_intelligence_report_sha256": intelligence["report_sha256"],
            "automatic_status": intelligence["status"],
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
            "parent_relative_manual_report_sha256": None,
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
        "execution_audit": {
            "network_access": False,
            "hub_loader_used": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "parent_forward_passes": 0,
            "parent_generate_invocations": 0,
            "candidate_generate_invocations": len(candidate_rows),
        },
    }
    target_report = {
        **target_core,
        "report_sha256": canonical_sha256(target_core),
    }
    control_runner._write_new_json(
        config.output_root / "target_report.json", target_report
    )
    core = {
        "schema": RUNNER_SCHEMA,
        "status": "completed_exact50_all8_control",
        "control_label": (
            "UNIFORM-INT8 LOSSLESS-CONTAINER CONTROL; "
            "NOT STRUCTURAL PARAMETER COMPACTION"
        ),
        "target_report": target_report,
        "parent_binding": {
            "parent_envelope_sha256": config.expected_parent_envelope_sha256,
            "parent_state_fingerprint": (config.expected_parent_state_fingerprint),
            "parent_model_fingerprint": (config.expected_parent_model_fingerprint),
            "tokenizer_fingerprint": config.expected_tokenizer_fingerprint,
            "parent_state_unchanged_by_export": True,
            "new_parent_responses": 0,
        },
        "execution_audit": {
            "target_reduction": 0.5,
            "training_used": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "network_access": False,
            "hub_loader_used": False,
        },
    }
    report = {**core, "report_sha256": canonical_sha256(core)}
    control_runner._write_new_json(config.output_root / "report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the exact-50% uniform-int8 LFM2 checkpoint, prove two "
            "fresh reloads, and score the sealed independent 64-case suite."
        )
    )
    parser.add_argument("--parent-model-directory", type=Path, required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-envelope", type=Path, required=True)
    parser.add_argument("--expected-parent-envelope-file-sha256", required=True)
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-parent-state-fingerprint", required=True)
    parser.add_argument("--expected-parent-model-fingerprint", required=True)
    parser.add_argument("--expected-tokenizer-fingerprint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--analysis-chunk-groups", type=int, default=4096)
    return parser


def config_from_args(args: argparse.Namespace) -> Exact50All8RunConfig:
    return Exact50All8RunConfig(**vars(args))


def main(argv: Sequence[str] | None = None) -> int:
    report = run_lfm2_exact50_all8(config_from_args(build_parser().parse_args(argv)))
    target = report["target_report"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_sha256": report["report_sha256"],
                "compressed_checkpoint_bytes": target["checkpoint"][
                    "compressed_checkpoint_bytes"
                ],
                "packed_weight_runtime_bytes": target["checkpoint"][
                    "packed_weight_runtime_bytes"
                ],
                "holdout_status": target["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
