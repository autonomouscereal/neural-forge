#!/usr/bin/env python3
"""Re-evaluate an existing packed LFM2 checkpoint with the current loader.

This cached-parent diagnostic exists to isolate runtime-loader corrections from
the immutable packed weight bytes. It never reloads or generates with the
parent and never rewrites the checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

try:
    from experiments import evaluate_lfm2_intelligence_holdout as holdout_eval
    from experiments import evaluate_lfm2_intelligence_retention as posthoc
    from experiments import run_lfm2_causal_value_allocator as causal_runner
    from experiments import run_lfm2_value_fold_control as control_runner
except ImportError:
    import evaluate_lfm2_intelligence_holdout as holdout_eval
    import evaluate_lfm2_intelligence_retention as posthoc
    import run_lfm2_causal_value_allocator as causal_runner
    import run_lfm2_value_fold_control as control_runner

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_intelligence_holdout import (
    load_lfm2_intelligence_holdout_parent_envelope,
    seal_lfm2_intelligence_holdout_candidate_outputs,
)
from neural_forge.hf_lfm2_protected_weight_error_allocator import (
    inspect_lfm2_protected_weight_error_fold,
    load_lfm2_protected_weight_error_fold,
)
from neural_forge.hf_lfm2_value_fold_control import (
    inspect_lfm2_value_fold_control,
    load_lfm2_value_fold_control,
    load_lfm2_value_fold_tokenizer,
)


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

LEGACY_SCHEMA = "neural-forge-lfm2-packed-checkpoint-reevaluation-v1"
SCHEMA = "neural-forge-lfm2-packed-checkpoint-reevaluation-v2"


class PackedCheckpointReevaluationError(RuntimeError):
    """The checkpoint, loader, or cached evaluation failed closed."""


def _digest(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PackedCheckpointReevaluationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _local_file(value: str | os.PathLike[str], name: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise PackedCheckpointReevaluationError(
            f"{name} is not a regular local file: {path}."
        )
    return path


def _local_directory(value: str | os.PathLike[str], name: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink():
        raise PackedCheckpointReevaluationError(
            f"{name} is not a regular local directory: {path}."
        )
    return path


def _loader_and_inspector(
    kind: str,
) -> tuple[Callable[..., nn.Module], Callable[[Path], dict[str, Any]]]:
    if kind == "value_control":
        return load_lfm2_value_fold_control, inspect_lfm2_value_fold_control
    if kind == "protected_weight_error":
        return (
            load_lfm2_protected_weight_error_fold,
            inspect_lfm2_protected_weight_error_fold,
        )
    raise PackedCheckpointReevaluationError(f"Unsupported checkpoint kind: {kind}.")


def _probe(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
) -> torch.Tensor:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    device = control_runner._runtime_device(model)
    kwargs: dict[str, Any] = {
        "input_ids": encoded["input_ids"].to(device=device, dtype=torch.long),
        "use_cache": False,
        "logits_to_keep": 1,
    }
    if encoded.get("attention_mask") is not None:
        kwargs["attention_mask"] = encoded["attention_mask"].to(
            device=device,
            dtype=torch.long,
        )
    with torch.inference_mode():
        return model(**kwargs).logits.detach().cpu()


def execute(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _local_directory(args.checkpoint, "checkpoint")
    envelope_path = _local_file(args.parent_envelope, "parent_envelope")
    tokenizer_directory = _local_directory(
        args.tokenizer_directory,
        "tokenizer_directory",
    )
    output_root = Path(args.output_root).expanduser().absolute()
    if output_root.exists():
        raise PackedCheckpointReevaluationError(
            f"Refusing to replace output_root {output_root}."
        )
    for name in (
        "expected_checkpoint_manifest_sha256",
        "expected_parent_envelope_file_sha256",
        "expected_parent_envelope_sha256",
        "expected_manifest_sha256",
        "expected_parent_state_fingerprint",
        "expected_parent_model_fingerprint",
        "expected_tokenizer_fingerprint",
    ):
        _digest(getattr(args, name), name)
    if (
        control_runner._file_sha256(envelope_path)
        != args.expected_parent_envelope_file_sha256
    ):
        raise PackedCheckpointReevaluationError(
            "Parent envelope file differs from its expected SHA-256."
        )
    envelope = load_lfm2_intelligence_holdout_parent_envelope(
        json.loads(envelope_path.read_text(encoding="utf-8")),
        expected_envelope_sha256=args.expected_parent_envelope_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_parent_state_fingerprint=args.expected_parent_state_fingerprint,
        expected_parent_model_fingerprint=args.expected_parent_model_fingerprint,
        expected_tokenizer_fingerprint=args.expected_tokenizer_fingerprint,
    )
    tokenizer_fingerprint, _files = posthoc.fingerprint_bound_local_tokenizer(
        tokenizer_directory
    )
    if tokenizer_fingerprint != args.expected_tokenizer_fingerprint:
        raise PackedCheckpointReevaluationError(
            "Local tokenizer differs from its sealed fingerprint."
        )
    special = posthoc.load_bound_special_token_ids(
        tokenizer_directory,
        expected_tokenizer_fingerprint=tokenizer_fingerprint,
    )
    eos_token_ids = tuple(special["special_token_ids"]["eos"])
    if eos_token_ids != envelope.eos_token_ids:
        raise PackedCheckpointReevaluationError(
            "Tokenizer EOS IDs differ from the sealed envelope."
        )
    loader, inspector = _loader_and_inspector(args.checkpoint_kind)
    accounting = inspector(checkpoint)
    if accounting["manifest_sha256"] != args.expected_checkpoint_manifest_sha256:
        raise PackedCheckpointReevaluationError(
            "Checkpoint manifest differs from its expected seal."
        )
    tokenizer = load_lfm2_value_fold_tokenizer(checkpoint)
    if (
        tokenizer.neural_forge_tokenizer_fingerprint
        != args.expected_tokenizer_fingerprint
    ):
        raise PackedCheckpointReevaluationError(
            "Checkpoint tokenizer differs from its expected fingerprint."
        )

    output_root.mkdir(parents=True)
    first = loader(checkpoint, device=args.device)
    first_manifest = first.neural_forge_load_report["manifest_sha256"]
    rotary_buffers = dict(first.named_buffers(remove_duplicate=False))
    for name in (
        "model.rotary_emb.inv_freq",
        "model.rotary_emb.original_inv_freq",
    ):
        value = rotary_buffers.get(name)
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16:
            raise PackedCheckpointReevaluationError(
                f"Corrected packed runtime did not restore BF16 {name}."
            )
    probe_prompt = envelope.manifest.cases[0].prompt
    first_logits = _probe(first, tokenizer, probe_prompt)
    del first
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    candidate = loader(checkpoint, device=args.device)
    second_manifest = candidate.neural_forge_load_report["manifest_sha256"]
    second_logits = _probe(candidate, tokenizer, probe_prompt)
    if (
        first_manifest != args.expected_checkpoint_manifest_sha256
        or second_manifest != first_manifest
        or not torch.equal(first_logits, second_logits)
    ):
        raise PackedCheckpointReevaluationError(
            "Two corrected fresh reloads are not bitwise logit-identical."
        )
    candidate_rows = control_runner._generate_candidate_rows(
        candidate,
        tokenizer,
        envelope.manifest,
    )
    candidate_document = seal_lfm2_intelligence_holdout_candidate_outputs(
        envelope.manifest,
        candidate_rows,
        candidate_state_fingerprint=first_manifest,
        tokenizer_fingerprint=args.expected_tokenizer_fingerprint,
        eos_token_ids=eos_token_ids,
    )
    candidate_path = output_root / "candidate_outputs.json"
    control_runner._write_new_json(candidate_path, candidate_document)
    candidate_file_sha = control_runner._file_sha256(candidate_path)
    del candidate
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    intelligence_path = output_root / "intelligence_report.json"
    manual_path = output_root / "manual_review_packet.json"
    intelligence = holdout_eval.evaluate_lfm2_intelligence_holdout(
        holdout_eval.HoldoutEvaluationConfig(
            parent_envelope=envelope_path,
            expected_parent_envelope_file_sha256=(
                args.expected_parent_envelope_file_sha256
            ),
            expected_parent_envelope_sha256=args.expected_parent_envelope_sha256,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_parent_state_fingerprint=(args.expected_parent_state_fingerprint),
            expected_parent_model_fingerprint=(args.expected_parent_model_fingerprint),
            tokenizer_directory=tokenizer_directory,
            expected_tokenizer_fingerprint=args.expected_tokenizer_fingerprint,
            candidate_report=candidate_path,
            expected_candidate_report_sha256=candidate_file_sha,
            report_path=intelligence_path,
            manual_review_packet_path=manual_path,
            expected_candidate_outputs_sha256=candidate_document[
                "candidate_outputs_sha256"
            ],
            expected_candidate_state_fingerprint=first_manifest,
        )
    )
    standard_packet = json.loads(manual_path.read_text(encoding="utf-8"))
    parent_relative_packet = causal_runner._build_exact_all_case_parent_relative_packet(
        standard_packet,
        envelope,
        candidate_rows,
    )
    control_runner._write_new_json(
        output_root / "parent_relative_manual_packet.json",
        parent_relative_packet,
    )
    decision = causal_runner._parent_relative_decision(
        intelligence,
        None,
        parent_relative_packet,
    )
    core = {
        "schema": SCHEMA,
        "status": decision["status"],
        "promoted": decision["promoted"],
        "checkpoint_kind": args.checkpoint_kind,
        "checkpoint": {
            "directory": str(checkpoint),
            "manifest_sha256": first_manifest,
            "accounting": accounting,
            "bytes_unchanged": True,
        },
        "loader_correction": {
            "source_runtime_buffer_dtype": "torch.bfloat16",
            "fresh_reload_count": 2,
            "probe_logits_bitwise_equal": True,
        },
        "holdout": {
            "candidate_outputs_sha256": candidate_document["candidate_outputs_sha256"],
            "candidate_output_file_sha256": candidate_file_sha,
            "intelligence_report_sha256": intelligence["report_sha256"],
            "parent_relative_manual_packet_sha256": parent_relative_packet[
                "packet_sha256"
            ],
            "parent_relative_manual_packet_case_count": parent_relative_packet[
                "case_count"
            ],
            "parent_relative_manual_review_scope": parent_relative_packet[
                "review_scope"
            ],
            "new_parent_responses": 0,
        },
        "decision": decision,
        "execution_audit": {
            "parent_model_loaded": False,
            "parent_forward_calls": 0,
            "parent_generate_invocations": 0,
            "candidate_generate_invocations": len(candidate_rows),
            "training_used": False,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "network_access": False,
            "hub_loader_used": False,
        },
    }
    report = {**core, "report_sha256": canonical_sha256(core)}
    control_runner._write_new_json(output_root / "report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-kind",
        choices=("value_control", "protected_weight_error"),
        required=True,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-directory", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--parent-envelope", required=True)
    parser.add_argument("--expected-checkpoint-manifest-sha256", required=True)
    parser.add_argument("--expected-parent-envelope-file-sha256", required=True)
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-parent-state-fingerprint", required=True)
    parser.add_argument("--expected-parent-model-fingerprint", required=True)
    parser.add_argument("--expected-tokenizer-fingerprint", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> int:
    report = execute(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "promoted": report["promoted"],
                "checkpoint_kind": report["checkpoint_kind"],
                "report_sha256": report["report_sha256"],
                "headline_inputs": report["decision"]["headline_inputs"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
