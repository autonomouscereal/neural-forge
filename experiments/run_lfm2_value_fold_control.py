#!/usr/bin/env python3
"""Build and score large packed-value LFM2 compression controls.

This runner is intentionally not a structural folding experiment.  It exports
exact serialized-directory ceilings near 50% and 75% reduction while
preserving every logical LFM2 operator.  Candidate behavior is evaluated on
the independent 64-case suite against an already sealed parent envelope.
The runner has no code path that can generate or replace parent responses.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from experiments import evaluate_lfm2_intelligence_holdout as holdout_eval
    from experiments import evaluate_lfm2_intelligence_retention as posthoc
    from experiments import run_lfm2_residual_attention_fold as direct_lfm2
except ImportError:  # Direct execution from the experiments directory.
    import evaluate_lfm2_intelligence_holdout as holdout_eval
    import evaluate_lfm2_intelligence_retention as posthoc
    import run_lfm2_residual_attention_fold as direct_lfm2

from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent
from neural_forge.hf_lfm2_intelligence_gate import (
    IntelligenceHoldoutManifest,
)
from neural_forge.hf_lfm2_intelligence_holdout import (
    load_lfm2_intelligence_holdout_parent_envelope,
    seal_lfm2_intelligence_holdout_candidate_outputs,
)
from neural_forge.hf_lfm2_value_fold_control import (
    export_lfm2_value_fold_control,
    inspect_lfm2_value_fold_control,
    load_lfm2_value_fold_control,
    load_lfm2_value_fold_tokenizer,
)


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

RUNNER_SCHEMA = "neural-forge-lfm2-packed-value-fold-control-run-v1"
TARGET_REPORT_SCHEMA = "neural-forge-lfm2-packed-value-fold-control-target-report-v1"


class Lfm2ValueFoldControlRunnerError(RuntimeError):
    """A local identity, immutable artifact, or control gate failed closed."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ValueFoldControlRunnerError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _local_directory(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2ValueFoldControlRunnerError(
            f"{name} must be an explicit local directory."
        )
    try:
        path = Path(raw).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2ValueFoldControlRunnerError(f"{name} does not exist.") from exc
    if not path.is_dir() or path.is_symlink():
        raise Lfm2ValueFoldControlRunnerError(
            f"{name} is missing, not a directory, or a symlink."
        )
    return path


def _local_file(value: str | os.PathLike[str], name: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2ValueFoldControlRunnerError(f"{name} does not exist.") from exc
    if not path.is_file() or path.is_symlink():
        raise Lfm2ValueFoldControlRunnerError(
            f"{name} is missing, not a file, or a symlink."
        )
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise Lfm2ValueFoldControlRunnerError(
            f"Refusing to replace existing output {destination}."
        )
    payload = canonical_json_bytes(dict(value)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _runtime_device(model: torch.nn.Module) -> torch.device:
    registry = getattr(model, "neural_forge_packed_registry", None)
    if (
        not isinstance(registry, torch.nn.ModuleList)
        or not registry
        or not isinstance(registry[0].scales, torch.Tensor)
    ):
        raise Lfm2ValueFoldControlRunnerError(
            "Candidate runtime lacks its packed registry."
        )
    return registry[0].scales.device


def _generate_candidate_rows(
    model: torch.nn.Module,
    tokenizer: Any,
    manifest: IntelligenceHoldoutManifest,
) -> tuple[dict[str, Any], ...]:
    """Generate only candidate responses for the sealed 64-case manifest."""

    if not isinstance(manifest, IntelligenceHoldoutManifest):
        raise TypeError("manifest must be IntelligenceHoldoutManifest.")
    device = _runtime_device(model)
    was_training = bool(model.training)
    model.eval()
    rows = []
    try:
        with torch.inference_mode():
            for spec in manifest.cases:
                encoded = tokenizer(
                    spec.prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                if (
                    not isinstance(encoded, Mapping)
                    or not isinstance(encoded.get("input_ids"), torch.Tensor)
                    or encoded["input_ids"].ndim != 2
                    or int(encoded["input_ids"].shape[0]) != 1
                ):
                    raise Lfm2ValueFoldControlRunnerError(
                        f"Tokenizer output is invalid for {spec.case_id}."
                    )
                input_ids = encoded["input_ids"].to(
                    device=device,
                    dtype=torch.long,
                )
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    if not isinstance(attention_mask, torch.Tensor):
                        raise Lfm2ValueFoldControlRunnerError(
                            f"Attention mask is invalid for {spec.case_id}."
                        )
                    attention_mask = attention_mask.to(
                        device=device,
                        dtype=torch.long,
                    )
                kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "max_new_tokens": spec.generation_limit,
                    "do_sample": False,
                    "use_cache": True,
                }
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask
                pad_token_id = getattr(tokenizer, "pad_token_id", None)
                if pad_token_id is not None:
                    kwargs["pad_token_id"] = int(pad_token_id)
                generated = model.generate(**kwargs)
                if (
                    not isinstance(generated, torch.Tensor)
                    or generated.ndim != 2
                    or int(generated.shape[0]) != 1
                    or int(generated.shape[1]) < int(input_ids.shape[1])
                ):
                    raise Lfm2ValueFoldControlRunnerError(
                        f"Candidate generation is invalid for {spec.case_id}."
                    )
                continuation = generated[0, int(input_ids.shape[1]) :]
                output_ids = [
                    int(value) for value in continuation.detach().to("cpu").tolist()
                ]
                if len(output_ids) > spec.generation_limit:
                    raise Lfm2ValueFoldControlRunnerError(
                        f"Candidate exceeded the limit for {spec.case_id}."
                    )
                output = tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                )
                if not isinstance(output, str):
                    raise Lfm2ValueFoldControlRunnerError(
                        f"Decoded candidate output is invalid for {spec.case_id}."
                    )
                rows.append(
                    {
                        "case_id": spec.case_id,
                        "output": output,
                        "output_token_ids": output_ids,
                        "generation_limit": spec.generation_limit,
                        "topical_relevance": None,
                    }
                )
    finally:
        model.train(was_training)
    if len(rows) != 64 or len(rows) != len(manifest.cases):
        raise Lfm2ValueFoldControlRunnerError(
            "Candidate generation did not cover the exact 64-case manifest."
        )
    return tuple(rows)


def _reload_probe(
    checkpoint: Path,
    tokenizer: Any,
    prompt: str,
    *,
    device: str,
) -> tuple[torch.Tensor, str]:
    model = load_lfm2_value_fold_control(checkpoint, device=device)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(
        _runtime_device(model),
        dtype=torch.long,
    )
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(_runtime_device(model), dtype=torch.long)
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
class ValueFoldControlRunConfig:
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
    target_reductions: tuple[float, ...] = (0.50, 0.75)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_model_directory",
            _local_directory(
                self.parent_model_directory,
                "parent_model_directory",
            ),
        )
        object.__setattr__(
            self,
            "tokenizer_directory",
            _local_directory(
                self.tokenizer_directory,
                "tokenizer_directory",
            ),
        )
        object.__setattr__(
            self,
            "parent_envelope",
            _local_file(self.parent_envelope, "parent_envelope"),
        )
        output = Path(self.output_root).expanduser().absolute()
        if output.exists():
            raise Lfm2ValueFoldControlRunnerError(
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
        if tuple(self.target_reductions) != (0.50, 0.75):
            raise Lfm2ValueFoldControlRunnerError(
                "The large control must run exactly the 50% and 75% targets."
            )
        if self.dtype not in {"float16", "bfloat16"}:
            raise Lfm2ValueFoldControlRunnerError("dtype must be float16 or bfloat16.")


def run_lfm2_value_fold_controls(
    config: ValueFoldControlRunConfig,
) -> dict[str, Any]:
    """Export controls, fresh-reload them, and score the cached holdout."""

    if not isinstance(config, ValueFoldControlRunConfig):
        raise TypeError("config must be ValueFoldControlRunConfig.")
    if _file_sha256(config.parent_envelope) != (
        config.expected_parent_envelope_file_sha256
    ):
        raise Lfm2ValueFoldControlRunnerError(
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
        raise Lfm2ValueFoldControlRunnerError(
            "Local tokenizer differs from the sealed parent tokenizer."
        )
    special = posthoc.load_bound_special_token_ids(
        config.tokenizer_directory,
        expected_tokenizer_fingerprint=tokenizer_fingerprint,
    )
    eos_token_ids = special["special_token_ids"]["eos"]
    if tuple(eos_token_ids) != envelope.eos_token_ids:
        raise Lfm2ValueFoldControlRunnerError(
            "Tokenizer EOS IDs differ from the sealed parent envelope."
        )
    config.output_root.mkdir(parents=True)
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
        raise Lfm2ValueFoldControlRunnerError(
            "Loaded local parent differs from the sealed parent or has gradients."
        )
    checkpoints = []
    for reduction in config.target_reductions:
        label = f"target-{int(round(reduction * 100)):02d}pct"
        target_root = config.output_root / label
        target_root.mkdir()
        checkpoint = target_root / "checkpoint"
        manifest = export_lfm2_value_fold_control(
            parent,
            checkpoint,
            tokenizer_directory=config.tokenizer_directory,
            target_reduction=reduction,
            group_size=config.group_size,
            analysis_chunk_groups=config.analysis_chunk_groups,
        )
        checkpoints.append((reduction, target_root, checkpoint, manifest))
    parent_state_after = direct_lfm2.model_state_sha256(parent)
    if parent_state_after != parent_state_before or any(
        parameter.grad is not None for parameter in parent.parameters()
    ):
        raise Lfm2ValueFoldControlRunnerError(
            "Packing mutated the parent state or created gradients."
        )
    del parent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target_reports = []
    for reduction, target_root, checkpoint, manifest in checkpoints:
        tokenizer = load_lfm2_value_fold_tokenizer(checkpoint)
        if (
            tokenizer.neural_forge_tokenizer_fingerprint
            != config.expected_tokenizer_fingerprint
        ):
            raise Lfm2ValueFoldControlRunnerError(
                "Copied checkpoint tokenizer differs from the sealed tokenizer."
            )
        probe_prompt = envelope.manifest.cases[0].prompt
        first_logits, first_manifest = _reload_probe(
            checkpoint,
            tokenizer,
            probe_prompt,
            device=config.device,
        )
        candidate = load_lfm2_value_fold_control(
            checkpoint,
            device=config.device,
        )
        encoded = tokenizer(
            probe_prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(
            _runtime_device(candidate),
            dtype=torch.long,
        )
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(
                _runtime_device(candidate),
                dtype=torch.long,
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
            raise Lfm2ValueFoldControlRunnerError(
                "Two fresh packed reloads are not logit-identical."
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
        candidate_path = target_root / "candidate_outputs.json"
        _write_new_json(candidate_path, candidate_document)
        candidate_file_sha = _file_sha256(candidate_path)
        del candidate
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        intelligence_path = target_root / "intelligence_report.json"
        manual_path = target_root / "manual_review_packet.json"
        intelligence = holdout_eval.evaluate_lfm2_intelligence_holdout(
            holdout_eval.HoldoutEvaluationConfig(
                parent_envelope=config.parent_envelope,
                expected_parent_envelope_file_sha256=(
                    config.expected_parent_envelope_file_sha256
                ),
                expected_parent_envelope_sha256=(
                    config.expected_parent_envelope_sha256
                ),
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
                manual_review_packet_path=manual_path,
                expected_candidate_outputs_sha256=candidate_document[
                    "candidate_outputs_sha256"
                ],
                expected_candidate_state_fingerprint=manifest["manifest_sha256"],
            )
        )
        accounting = inspect_lfm2_value_fold_control(checkpoint)
        core = {
            "schema": TARGET_REPORT_SCHEMA,
            "control_label": accounting["control_label"],
            "requested_reduction": reduction,
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
                "candidate_outputs_sha256": candidate_document[
                    "candidate_outputs_sha256"
                ],
                "candidate_output_file_sha256": candidate_file_sha,
                "intelligence_report_sha256": intelligence["report_sha256"],
                "status": intelligence["status"],
                "manual_review_status": intelligence["manual_review"]["status"],
                "parent_model_loaded_during_scoring": False,
                "new_parent_responses": 0,
            },
            "execution_audit": {
                "network_access": False,
                "hub_loader_used": False,
                "optimizer_constructed": False,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "parent_generate_invocations": 0,
                "candidate_generate_invocations": len(candidate_rows),
            },
        }
        target_report = {
            **core,
            "report_sha256": canonical_sha256(core),
        }
        _write_new_json(target_root / "target_report.json", target_report)
        target_reports.append(target_report)
    core = {
        "schema": RUNNER_SCHEMA,
        "status": "completed_controls",
        "control_label": (
            "PACKED VALUE-QUANTIZATION BASELINES; NOT STRUCTURAL PARAMETER COMPACTION"
        ),
        "target_reports": target_reports,
        "parent_binding": {
            "parent_envelope_sha256": config.expected_parent_envelope_sha256,
            "parent_state_fingerprint": (config.expected_parent_state_fingerprint),
            "parent_model_fingerprint": (config.expected_parent_model_fingerprint),
            "tokenizer_fingerprint": config.expected_tokenizer_fingerprint,
            "parent_state_unchanged_by_export": True,
            "new_parent_responses": 0,
        },
        "execution_audit": {
            "target_reductions": list(config.target_reductions),
            "training_used": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "network_access": False,
            "hub_loader_used": False,
        },
    }
    report = {**core, "report_sha256": canonical_sha256(core)}
    _write_new_json(config.output_root / "report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the exact 50%/75% LFM2 packed-value controls and score "
            "candidate outputs against the existing sealed 64-case parent cache."
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--analysis-chunk-groups", type=int, default=4096)
    return parser


def config_from_args(args: argparse.Namespace) -> ValueFoldControlRunConfig:
    return ValueFoldControlRunConfig(**vars(args))


def main(argv: Sequence[str] | None = None) -> int:
    report = run_lfm2_value_fold_controls(
        config_from_args(build_parser().parse_args(argv))
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_sha256": report["report_sha256"],
                "targets": [
                    {
                        "requested_reduction": value["requested_reduction"],
                        "compressed_checkpoint_bytes": value["checkpoint"][
                            "compressed_checkpoint_bytes"
                        ],
                        "holdout_status": value["holdout"]["status"],
                    }
                    for value in report["target_reports"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
