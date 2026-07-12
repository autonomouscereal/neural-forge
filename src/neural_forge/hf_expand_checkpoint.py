"""Fail-closed publication of function-preserving HF expansions."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from neural_forge.hf_arch_layout import build_decoder_layout
from neural_forge.hf_expand import (
    NoExactExpansionPlanError,
    count_unique_parameters,
    expand_hf_model,
)


REPORT_FILENAME = "neural_forge_hf_expand_report.json"
REPORT_SCHEMA = "neural-forge-hf-expand-checkpoint-v1"
DEFAULT_VALIDATION_PROMPTS = (
    "The exact expansion preserves this prediction.",
    "A deterministic checkpoint reload should match.",
)


class HFExpandCheckpointError(ValueError):
    """Raised when an expanded checkpoint cannot be proven publishable."""


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise HFExpandCheckpointError(f"{name} must be boolean.")
    return value


def _target_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HFExpandCheckpointError(
            "target_parameter_count must be a positive integer."
        )
    return value


def _dtype(torch: Any, value: Any) -> Any:
    if value == "auto":
        return "auto"
    if value in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        return value
    if isinstance(value, str):
        name = value.strip().lower().replace("torch.", "")
        name = {
            "fp16": "float16",
            "bf16": "bfloat16",
            "fp32": "float32",
            "fp64": "float64",
        }.get(name, name)
        if name in {"float16", "bfloat16", "float32", "float64"}:
            return getattr(torch, name)
    raise HFExpandCheckpointError(
        "dtype must be auto, float16, bfloat16, float32, or float64."
    )


def _device(torch: Any, value: Any) -> Any:
    try:
        resolved = torch.device(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HFExpandCheckpointError(f"Invalid device {value!r}.") from exc
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise HFExpandCheckpointError(
            "A CUDA device was requested but CUDA is unavailable."
        )
    return resolved


def _canonical_json(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise HFExpandCheckpointError(
            "The expansion report contains invalid JSON data."
        ) from exc
    return encoded.encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_sha256(torch: Any, tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _state_dict_sha256(torch: Any, model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_files(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name != REPORT_FILENAME
    ]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _prompt_values(values: Sequence[str] | None, name: str) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        raise HFExpandCheckpointError(f"{name} must be a sequence of strings.")
    result = list(values)
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in result):
        raise HFExpandCheckpointError(f"{name} must contain nonempty strings.")
    return result


def _file_prompts(path: str | os.PathLike[str] | None) -> tuple[list[str], Any]:
    if path is None:
        return [], None
    prompt_path = Path(path).expanduser().resolve()
    try:
        content = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HFExpandCheckpointError(
            f"Could not read validation prompts file {prompt_path}."
        ) from exc
    if prompt_path.suffix.lower() == ".json" or content.lstrip().startswith("["):
        try:
            values = json.loads(content)
        except json.JSONDecodeError as exc:
            raise HFExpandCheckpointError(
                f"Validation prompts file {prompt_path} is invalid JSON."
            ) from exc
        if not isinstance(values, list):
            raise HFExpandCheckpointError(
                "JSON validation prompts file must contain an array of strings."
            )
        prompts = _prompt_values(values, "validation prompts file")
        file_format = "json-array"
    else:
        prompts = [line.strip() for line in content.splitlines() if line.strip()]
        file_format = "nonempty-lines"
    if not prompts:
        raise HFExpandCheckpointError("Validation prompts file contains no prompts.")
    return prompts, {
        "path": str(prompt_path),
        "format": file_format,
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def _validation_prompts(
    inline: Sequence[str] | None,
    prompt_file: str | os.PathLike[str] | None,
) -> tuple[tuple[str, ...], Any]:
    prompts = _prompt_values(inline, "validation_prompts")
    from_file, file_record = _file_prompts(prompt_file)
    prompts.extend(from_file)
    if not prompts:
        prompts.extend(DEFAULT_VALIDATION_PROMPTS)
    return tuple(prompts), file_record


def _accepted_forward_keys(model: Any) -> set[str] | None:
    try:
        parameters = inspect.signature(model.forward).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return None
    return {parameter.name for parameter in parameters}


def _tokenize(
    torch: Any, tokenizer: Any, model: Any, prompt: str, device: Any
) -> dict[str, Any]:
    try:
        encoded = tokenizer(prompt, return_tensors="pt")
    except Exception as exc:
        raise HFExpandCheckpointError(
            "Tokenizer failed on a validation prompt."
        ) from exc
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise HFExpandCheckpointError("Tokenizer did not return input_ids.")
    accepted = _accepted_forward_keys(model)
    result = {
        name: value.to(device)
        for name, value in encoded.items()
        if isinstance(value, torch.Tensor) and (accepted is None or name in accepted)
    }
    if "input_ids" not in result or int(result["input_ids"].numel()) == 0:
        raise HFExpandCheckpointError("Validation prompt produced no input tokens.")
    return result


def _logits(torch: Any, model: Any, inputs: Mapping[str, Any]) -> Any:
    model.eval()
    kwargs = dict(inputs)
    accepted = _accepted_forward_keys(model)
    if accepted is None or "use_cache" in accepted:
        kwargs["use_cache"] = False
    with torch.inference_mode():
        output = model(**kwargs)
    logits = getattr(output, "logits", None)
    if logits is None:
        raise HFExpandCheckpointError("Validation forward pass returned no logits.")
    value = logits.detach().cpu().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise HFExpandCheckpointError("Validation logits must be finite.")
    return value


def _max_error(torch: Any, left: Any, right: Any) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise HFExpandCheckpointError(
            f"Validation logits changed shape from {tuple(left.shape)} to "
            f"{tuple(right.shape)}."
        )
    return float((left.to(torch.float64) - right.to(torch.float64)).abs().max())


def _gradient_count(model: Any) -> int:
    return sum(int(parameter.grad is not None) for parameter in model.parameters())


def _assert_exact_state(torch: Any, expected: Any, actual: Any) -> int:
    expected_state = expected.state_dict()
    actual_state = actual.state_dict()
    if tuple(sorted(expected_state)) != tuple(sorted(actual_state)):
        raise HFExpandCheckpointError(
            "Reloaded checkpoint state keys differ from the expanded candidate."
        )
    checked = 0
    for name in sorted(expected_state):
        left = expected_state[name].detach().cpu()
        right = actual_state[name].detach().cpu()
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            raise HFExpandCheckpointError(
                f"Reloaded tensor metadata differs for {name!r}."
            )
        if not torch.equal(left, right):
            raise HFExpandCheckpointError(
                f"Reloaded tensor values differ for {name!r}."
            )
        checked += 1
    return checked


def _assert_exact_inputs(
    torch: Any,
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
) -> int:
    if len(expected) != len(actual):
        raise HFExpandCheckpointError(
            "Reloaded tokenizer produced a different validation batch count."
        )
    checked = 0
    for index, (left, right) in enumerate(zip(expected, actual)):
        if tuple(sorted(left)) != tuple(sorted(right)):
            raise HFExpandCheckpointError(
                f"Reloaded tokenizer changed validation input keys for prompt {index}."
            )
        for name in sorted(left):
            if not torch.equal(left[name].detach().cpu(), right[name].detach().cpu()):
                raise HFExpandCheckpointError(
                    "Reloaded tokenizer changed validation tensor "
                    f"{name!r} for prompt {index}."
                )
            checked += 1
    return checked


def _source_record(source_model: str | os.PathLike[str]) -> dict[str, Any]:
    requested = os.fspath(source_model)
    candidate = Path(requested).expanduser()
    return {
        "requested": requested,
        "resolved_local_path": str(candidate.resolve()) if candidate.exists() else None,
    }


def expand_hf_checkpoint(
    source_model: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    target_parameter_count: int,
    *,
    dtype: Any = "float32",
    device: Any = "cpu",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    validation_prompts: Sequence[str] | None = None,
    validation_prompts_file: str | os.PathLike[str] | None = None,
    max_equivalence_error: float = 1e-5,
) -> dict[str, Any]:
    """Expand, independently reload, verify, and atomically publish an HF model."""

    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "HF checkpoint expansion requires torch, Transformers, and safetensors."
        ) from exc

    target_count = _target_count(target_parameter_count)
    trust = _strict_bool(trust_remote_code, "trust_remote_code")
    local_only = _strict_bool(local_files_only, "local_files_only")
    if (
        isinstance(max_equivalence_error, bool)
        or not isinstance(max_equivalence_error, (int, float))
        or not math.isfinite(max_equivalence_error)
        or max_equivalence_error < 0
    ):
        raise HFExpandCheckpointError(
            "max_equivalence_error must be finite and nonnegative."
        )
    error_limit = float(max_equivalence_error)
    resolved_dtype = _dtype(torch, dtype)
    resolved_device = _device(torch, device)
    prompts, prompt_file_record = _validation_prompts(
        validation_prompts, validation_prompts_file
    )
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise HFExpandCheckpointError("output_dir must not already exist.")

    load_kwargs = {
        "trust_remote_code": trust,
        "local_files_only": local_only,
        "torch_dtype": resolved_dtype,
    }
    source = transformers.AutoModelForCausalLM.from_pretrained(
        source_model, **load_kwargs
    )
    source.to(resolved_device)
    source.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        source_model,
        trust_remote_code=trust,
        local_files_only=local_only,
    )
    if not callable(getattr(tokenizer, "save_pretrained", None)):
        raise HFExpandCheckpointError("Loaded tokenizer cannot be saved.")
    if _gradient_count(source):
        raise HFExpandCheckpointError("Source model contains populated gradients.")

    source_layout = build_decoder_layout(source)
    source_count = count_unique_parameters(source)
    source_fingerprint = _state_dict_sha256(torch, source)
    validation_inputs = [
        _tokenize(torch, tokenizer, source, prompt, resolved_device)
        for prompt in prompts
    ]
    source_logits = [
        _logits(torch, source, prompt_inputs) for prompt_inputs in validation_inputs
    ]

    try:
        expansion = expand_hf_model(
            source,
            target_count,
            validation_inputs=validation_inputs[0],
            max_equivalence_error=error_limit,
        )
    except NoExactExpansionPlanError as exc:
        raise HFExpandCheckpointError(
            f"Exact target parameter count {target_count} is unreachable: {exc}"
        ) from exc

    candidate = expansion.model
    candidate.eval()
    if _gradient_count(candidate):
        raise HFExpandCheckpointError("Expanded model contains populated gradients.")
    candidate_logits = [
        _logits(torch, candidate, prompt_inputs) for prompt_inputs in validation_inputs
    ]
    source_candidate_errors = [
        _max_error(torch, before, after)
        for before, after in zip(source_logits, candidate_logits)
    ]
    function_error = max(source_candidate_errors)
    if function_error > error_limit:
        raise HFExpandCheckpointError(
            f"Function-equivalence error {function_error:.9g} exceeds "
            f"limit {error_limit:.9g}."
        )

    plan = expansion.report.plan
    candidate_layout = build_decoder_layout(candidate)
    candidate_count = count_unique_parameters(candidate)
    if (
        candidate_count != target_count
        or candidate_layout.layer_count != plan.target_layers
        or candidate_layout.intermediate_size != plan.target_intermediate_size
    ):
        raise HFExpandCheckpointError(
            "Expanded candidate count or decoder shape differs from its exact plan."
        )
    candidate_fingerprint = _state_dict_sha256(torch, candidate)
    actual_dtype = next(candidate.parameters()).dtype

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        candidate.save_pretrained(stage, safe_serialization=True)
        tokenizer.save_pretrained(stage)
        safetensors = sorted(stage.rglob("*.safetensors"))
        unsafe_weights = sorted(stage.rglob("*.bin"))
        if not safetensors or unsafe_weights:
            raise HFExpandCheckpointError(
                "Expanded checkpoint was not saved exclusively as safetensors."
            )

        reloaded = transformers.AutoModelForCausalLM.from_pretrained(
            stage,
            trust_remote_code=trust,
            local_files_only=True,
            torch_dtype=resolved_dtype,
        )
        reloaded.to(resolved_device)
        reloaded.eval()
        reloaded_tokenizer = transformers.AutoTokenizer.from_pretrained(
            stage,
            trust_remote_code=trust,
            local_files_only=True,
        )
        reloaded_validation_inputs = [
            _tokenize(
                torch,
                reloaded_tokenizer,
                reloaded,
                prompt,
                resolved_device,
            )
            for prompt in prompts
        ]
        checked_input_tensors = _assert_exact_inputs(
            torch, validation_inputs, reloaded_validation_inputs
        )
        if _gradient_count(reloaded):
            raise HFExpandCheckpointError(
                "Reloaded model contains populated gradients."
            )
        reloaded_count = count_unique_parameters(reloaded)
        reloaded_layout = build_decoder_layout(reloaded)
        if reloaded_count != target_count:
            raise HFExpandCheckpointError(
                f"Reloaded unique parameter count {reloaded_count} does not equal "
                f"exact target {target_count}."
            )
        if (
            reloaded_layout.layer_count != plan.target_layers
            or reloaded_layout.intermediate_size != plan.target_intermediate_size
        ):
            raise HFExpandCheckpointError(
                "Reloaded decoder depth or width differs from the exact plan."
            )
        checked_tensors = _assert_exact_state(torch, candidate, reloaded)
        reloaded_fingerprint = _state_dict_sha256(torch, reloaded)
        if reloaded_fingerprint != candidate_fingerprint:
            raise HFExpandCheckpointError(
                "Reloaded state fingerprint differs from the expanded candidate."
            )

        reloaded_logits = [
            _logits(torch, reloaded, prompt_inputs)
            for prompt_inputs in reloaded_validation_inputs
        ]
        source_reload_errors = [
            _max_error(torch, before, after)
            for before, after in zip(source_logits, reloaded_logits)
        ]
        candidate_reload_errors = [
            _max_error(torch, before, after)
            for before, after in zip(candidate_logits, reloaded_logits)
        ]
        reload_function_error = max(source_reload_errors)
        reload_serialization_error = max(candidate_reload_errors)
        if max(reload_function_error, reload_serialization_error) > error_limit:
            raise HFExpandCheckpointError(
                "Reloaded validation logits exceed max_equivalence_error: "
                f"source={reload_function_error:.9g}, "
                f"candidate={reload_serialization_error:.9g}, "
                f"limit={error_limit:.9g}."
            )

        validation_records = []
        for index, prompt in enumerate(prompts):
            validation_records.append(
                {
                    "index": index,
                    "prompt": prompt,
                    "input_ids_sha256": _tensor_sha256(
                        torch, validation_inputs[index]["input_ids"]
                    ),
                    "source_logits_sha256": _tensor_sha256(torch, source_logits[index]),
                    "candidate_logits_sha256": _tensor_sha256(
                        torch, candidate_logits[index]
                    ),
                    "reloaded_logits_sha256": _tensor_sha256(
                        torch, reloaded_logits[index]
                    ),
                    "source_candidate_max_error": source_candidate_errors[index],
                    "source_reload_max_error": source_reload_errors[index],
                    "candidate_reload_max_error": candidate_reload_errors[index],
                    "candidate_reload_exact": bool(
                        torch.equal(candidate_logits[index], reloaded_logits[index])
                    ),
                }
            )

        report: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "source": {
                **_source_record(source_model),
                "model_class": type(source).__name__,
                "config_class": type(source.config).__name__,
                "unique_parameter_count": source_count,
                "decoder_depth": source_layout.layer_count,
                "intermediate_size": source_layout.intermediate_size,
                "state_dict_sha256": source_fingerprint,
            },
            "request": {
                "target_parameter_count": target_count,
                "requested_dtype": str(dtype).replace("torch.", ""),
                "materialized_dtype": str(actual_dtype).replace("torch.", ""),
                "device": str(resolved_device),
                "trust_remote_code": trust,
                "local_files_only": local_only,
                "max_equivalence_error": error_limit,
            },
            "plan": asdict(plan),
            "provenance": [
                asdict(item) for item in expansion.report.inserted_layer_provenance
            ],
            "expansion_report": asdict(expansion.report),
            "validation": {
                "prompt_count": len(prompts),
                "prompts_sha256": _json_sha256(list(prompts)),
                "prompts_file": prompt_file_record,
                "function_equivalence_max_error": function_error,
                "reload_function_equivalence_max_error": reload_function_error,
                "reload_serialization_max_error": reload_serialization_error,
                "max_equivalence_error": error_limit,
                "prompts": validation_records,
            },
            "checkpoint_verification": {
                "performed": True,
                "loader": "transformers.AutoModelForCausalLM.from_pretrained",
                "tokenizer_loader": "transformers.AutoTokenizer.from_pretrained",
                "local_reload": True,
                "unique_parameter_count": reloaded_count,
                "decoder_depth": reloaded_layout.layer_count,
                "intermediate_size": reloaded_layout.intermediate_size,
                "exact_state_equality": True,
                "state_tensors_checked": checked_tensors,
                "tokenizer_input_tensors_checked": checked_input_tensors,
                "candidate_state_dict_sha256": candidate_fingerprint,
                "reloaded_state_dict_sha256": reloaded_fingerprint,
            },
            "execution": {
                "training_performed": False,
                "training_steps": 0,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "source_gradients": 0,
                "candidate_gradients": 0,
                "reloaded_gradients": 0,
            },
            "software": {
                "torch": str(torch.__version__),
                "transformers": str(transformers.__version__),
            },
        }
        report["artifacts"] = {
            "report_filename": REPORT_FILENAME,
            "safetensors_files": [
                path.relative_to(stage).as_posix() for path in safetensors
            ],
            "files": _artifact_files(stage),
        }
        normalized = json.loads(_canonical_json(report))
        _write_json(stage / REPORT_FILENAME, normalized)
        stage.replace(output)
        return normalized
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = [
    "DEFAULT_VALIDATION_PROMPTS",
    "HFExpandCheckpointError",
    "REPORT_FILENAME",
    "REPORT_SCHEMA",
    "expand_hf_checkpoint",
]
