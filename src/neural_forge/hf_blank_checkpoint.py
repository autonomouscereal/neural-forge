"""Fail-closed checkpoint publication for blank decoder bootstrap."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from neural_forge.hf_arch_layout import DecoderLayout, build_decoder_layout
from neural_forge.hf_blank_bootstrap import bootstrap_blank_decoder_models
from neural_forge.hf_factorized_vocab import exact_parameter_report


REPORT_FILENAME = "neural_forge_blank_checkpoint_report.json"
REPORT_SCHEMA = "neural-forge-blank-checkpoint-v1"
TargetFamily = Literal["gpt2", "llama", "qwen3"]


class BlankCheckpointError(ValueError):
    """Raised when a blank checkpoint request cannot be proven safe."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BlankCheckpointError(f"{name} must be a positive integer.")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise BlankCheckpointError(f"{name} must be boolean.")
    return value


def _dtype(torch: Any, value: Any) -> Any:
    if value in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        return value
    if isinstance(value, str):
        name = value.strip().lower().replace("torch.", "")
        name = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}.get(
            name, name
        )
        if name in {"float16", "bfloat16", "float32", "float64"}:
            return getattr(torch, name)
    raise BlankCheckpointError("dtype must be float16, bfloat16, float32, or float64.")


def _device(torch: Any, value: Any) -> Any:
    try:
        resolved = torch.device(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise BlankCheckpointError(f"Invalid device {value!r}.") from exc
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise BlankCheckpointError(
            "A CUDA device was requested but CUDA is unavailable."
        )
    return resolved


def _canonical_json(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise BlankCheckpointError(
            "The checkpoint report contains invalid JSON data."
        ) from exc
    return text.encode("utf-8")


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
    result = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == REPORT_FILENAME:
            continue
        result.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return result


def _parameter_counts(model: Any) -> dict[str, Any]:
    exact = exact_parameter_report(model)
    logical = int(exact["logical_parameter_count"])
    unique = int(exact["unique_parameter_count"])
    return {
        "dense_parameter_count": logical,
        "logical_parameter_count": logical,
        "unique_parameter_count": unique,
        "shared_parameter_references": logical - unique,
        "logical_parameter_tensors": int(exact["logical_parameter_tensors"]),
        "unique_parameter_tensors": int(exact["unique_parameter_tensors"]),
    }


def _compression(source: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in ("dense", "logical", "unique"):
        source_count = int(source[f"{label}_parameter_count"])
        child_count = int(child[f"{label}_parameter_count"])
        result[label] = {
            "source_to_child_ratio": source_count / child_count,
            "child_fraction_of_source": child_count / source_count,
            "parameters_removed": source_count - child_count,
        }
    return result


def _layout_identity(layout: DecoderLayout, model: Any) -> dict[str, Any]:
    config = model.config
    return {
        "family": layout.family,
        "model_type": layout.model_type,
        "model_class": layout.model_class,
        "config_class": type(config).__name__,
        "name_or_path": str(getattr(config, "_name_or_path", "")),
        "vocab_size": layout.embeddings.vocab_size,
        "hidden_size": layout.hidden_size,
        "intermediate_size": layout.intermediate_size,
        "num_hidden_layers": layout.layer_count,
        "num_attention_heads": layout.blocks[0].attention.query_heads,
        "num_key_value_heads": layout.blocks[0].attention.key_value_heads,
        "head_dim": layout.blocks[0].attention.head_dim,
        "max_position_embeddings": layout.position.max_positions,
        "tie_word_embeddings": layout.tie_word_embeddings,
    }


def _normal_special_id(
    value: Any, name: str, vocab_size: int
) -> int | list[int] | None:
    if value is None:
        return None
    values = value if isinstance(value, (tuple, list)) else [value]
    normalized = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int):
            raise BlankCheckpointError(f"{name} must contain only integer token IDs.")
        if item < 0 or item >= vocab_size:
            raise BlankCheckpointError(f"{name} contains an out-of-range token ID.")
        normalized.append(item)
    if not normalized:
        raise BlankCheckpointError(f"{name} cannot be an empty sequence.")
    return normalized if isinstance(value, (tuple, list)) else normalized[0]


def _ids_compatible(model_value: Any, tokenizer_value: Any) -> bool:
    if model_value is None or tokenizer_value is None:
        return True
    model_values = model_value if isinstance(model_value, list) else [model_value]
    return tokenizer_value in model_values


def _special_token_ids(model: Any, tokenizer: Any, vocab_size: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        model_value = _normal_special_id(
            getattr(model.config, name, None), name, vocab_size
        )
        tokenizer_value = _normal_special_id(
            getattr(tokenizer, name, None), name, vocab_size
        )
        if not _ids_compatible(model_value, tokenizer_value):
            raise BlankCheckpointError(
                f"Donor config and tokenizer disagree on {name}: "
                f"{model_value!r} != {tokenizer_value!r}."
            )
        result[name] = model_value if model_value is not None else tokenizer_value

    if result["pad_token_id"] is None:
        eos = result["eos_token_id"]
        eos_id = eos[0] if isinstance(eos, list) else eos
        if eos_id is None:
            raise BlankCheckpointError(
                "A pad token or scalar EOS token is required for deterministic batching."
            )
        token = tokenizer.convert_ids_to_tokens(eos_id)
        if token is None:
            raise BlankCheckpointError("The fallback EOS token cannot be resolved.")
        tokenizer.pad_token = token
        if tokenizer.pad_token_id != eos_id:
            raise BlankCheckpointError(
                "Failed to assign EOS as the tokenizer pad token."
            )
        result["pad_token_id"] = eos_id
    return result


def _tokenizer_identity(
    tokenizer: Any, special_ids: Mapping[str, Any]
) -> dict[str, Any]:
    if not callable(getattr(tokenizer, "get_vocab", None)):
        raise BlankCheckpointError("The tokenizer must expose get_vocab().")
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, Mapping) or not vocab:
        raise BlankCheckpointError(
            "The tokenizer vocabulary must be a non-empty mapping."
        )
    normalized_vocab = {str(token): int(index) for token, index in vocab.items()}
    return {
        "class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "length": int(len(tokenizer)),
        "vocab_entries": len(normalized_vocab),
        "vocab_sha256": _json_sha256(normalized_vocab),
        "special_token_ids": dict(special_ids),
    }


def _load_tokenizer(
    transformers: Any,
    tokenizer_choice: Any,
    donor_reference: str | None,
    *,
    trust_remote_code: bool,
    local_files_only: bool,
) -> tuple[Any, dict[str, Any]]:
    choice = tokenizer_choice
    defaulted = choice is None
    if choice is None:
        choice = donor_reference
    if choice is None or (isinstance(choice, str) and not choice.strip()):
        raise BlankCheckpointError(
            "tokenizer must be supplied when the donor model has no checkpoint identity."
        )
    if callable(getattr(choice, "save_pretrained", None)):
        return choice, {
            "loaded_from_pretrained": False,
            "defaulted_to_donor": defaulted,
        }
    if not isinstance(choice, (str, os.PathLike)):
        raise BlankCheckpointError(
            "tokenizer must be a tokenizer object or model/path."
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(choice),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    return tokenizer, {
        "loaded_from_pretrained": True,
        "defaulted_to_donor": defaulted,
        "reference": str(choice),
    }


def _load_donor(
    torch: Any,
    transformers: Any,
    donor: Any,
    *,
    dtype: Any,
    device: Any,
    trust_remote_code: bool,
    local_files_only: bool,
) -> tuple[Any, str | None, int]:
    if isinstance(donor, torch.nn.Module):
        model = donor
        reference = str(getattr(getattr(model, "config", None), "_name_or_path", ""))
        reference = reference or None
        load_calls = 0
    elif isinstance(donor, (str, os.PathLike)):
        reference = str(donor)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            reference,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
        load_calls = 1
    else:
        raise BlankCheckpointError("donor must be a causal LM object or model/path.")
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, reference, load_calls


def _validate_shape(
    family: Any,
    *,
    hidden_size: Any,
    intermediate_size: Any,
    num_hidden_layers: Any,
    num_attention_heads: Any,
    num_key_value_heads: Any,
    head_dim: Any,
    max_position_embeddings: Any,
) -> dict[str, int | str]:
    if not isinstance(family, str) or family.lower() not in {"gpt2", "llama", "qwen3"}:
        raise BlankCheckpointError("target_family must be gpt2, llama, or qwen3.")
    result: dict[str, int | str] = {"target_family": family.lower()}
    for name, value in (
        ("hidden_size", hidden_size),
        ("intermediate_size", intermediate_size),
        ("num_hidden_layers", num_hidden_layers),
        ("num_attention_heads", num_attention_heads),
        ("num_key_value_heads", num_key_value_heads),
        ("head_dim", head_dim),
        ("max_position_embeddings", max_position_embeddings),
    ):
        result[name] = _positive_int(value, name)
    heads = int(result["num_attention_heads"])
    kv_heads = int(result["num_key_value_heads"])
    if heads % kv_heads:
        raise BlankCheckpointError(
            "num_attention_heads must be divisible by num_key_value_heads."
        )
    if result["target_family"] == "gpt2":
        if kv_heads != heads:
            raise BlankCheckpointError(
                "GPT-2 requires num_key_value_heads == num_attention_heads."
            )
        if int(result["hidden_size"]) != heads * int(result["head_dim"]):
            raise BlankCheckpointError(
                "GPT-2 requires hidden_size == num_attention_heads * head_dim."
            )
    elif int(result["head_dim"]) % 2:
        raise BlankCheckpointError("Llama and Qwen3 require an even head_dim for RoPE.")
    if result["target_family"] == "qwen3" and not 1 < kv_heads < heads:
        raise BlankCheckpointError(
            "Qwen3 requires more than one num_key_value_heads and fewer "
            "num_key_value_heads than num_attention_heads."
        )
    return result


def _target_model(
    torch: Any,
    transformers: Any,
    shape: Mapping[str, Any],
    *,
    vocab_size: int,
    special_ids: Mapping[str, Any],
    tied: bool,
    seed: int,
    dtype: Any,
    device: Any,
) -> tuple[Any, str]:
    common = {
        "vocab_size": vocab_size,
        "bos_token_id": special_ids["bos_token_id"],
        "eos_token_id": special_ids["eos_token_id"],
        "pad_token_id": special_ids["pad_token_id"],
        "tie_word_embeddings": tied,
        "use_cache": False,
    }
    family = shape["target_family"]
    if family == "gpt2":
        config = transformers.GPT2Config(
            **common,
            n_embd=shape["hidden_size"],
            n_inner=shape["intermediate_size"],
            n_layer=shape["num_hidden_layers"],
            n_head=shape["num_attention_heads"],
            n_positions=shape["max_position_embeddings"],
            n_ctx=shape["max_position_embeddings"],
        )
        model_class = transformers.GPT2LMHeadModel
    elif family == "llama":
        config = transformers.LlamaConfig(
            **common,
            hidden_size=shape["hidden_size"],
            intermediate_size=shape["intermediate_size"],
            num_hidden_layers=shape["num_hidden_layers"],
            num_attention_heads=shape["num_attention_heads"],
            num_key_value_heads=shape["num_key_value_heads"],
            head_dim=shape["head_dim"],
            max_position_embeddings=shape["max_position_embeddings"],
            attention_bias=False,
            mlp_bias=False,
        )
        model_class = transformers.LlamaForCausalLM
    else:
        if not hasattr(transformers, "Qwen3ForCausalLM"):
            raise BlankCheckpointError(
                "The installed Transformers lacks Qwen3 support."
            )
        config = transformers.Qwen3Config(
            **common,
            hidden_size=shape["hidden_size"],
            intermediate_size=shape["intermediate_size"],
            num_hidden_layers=shape["num_hidden_layers"],
            num_attention_heads=shape["num_attention_heads"],
            num_key_value_heads=shape["num_key_value_heads"],
            head_dim=shape["head_dim"],
            max_position_embeddings=shape["max_position_embeddings"],
            attention_bias=False,
            use_sliding_window=False,
        )
        model_class = transformers.Qwen3ForCausalLM

    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(seed)
        with torch.no_grad():
            model = model_class(config)
            model.to(device=device, dtype=dtype)
    model.eval()
    return model, f"{model_class.__name__}({type(config).__name__}(...))"


def _prompt_sequence(values: Any, name: str) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise BlankCheckpointError(f"{name} must be a non-empty sequence of strings.")
    prompts = list(values)
    if not prompts or any(not isinstance(item, str) for item in prompts):
        raise BlankCheckpointError(f"{name} must be a non-empty sequence of strings.")
    return prompts


def _tokenize(
    torch: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_length: int,
    device: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    had_padding_side = hasattr(tokenizer, "padding_side")
    original_padding_side = getattr(tokenizer, "padding_side", None)
    if had_padding_side:
        tokenizer.padding_side = "right"
    try:
        encoded = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False,
        )
    finally:
        if had_padding_side:
            tokenizer.padding_side = original_padding_side
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise BlankCheckpointError("Tokenizer output must contain input_ids.")
    input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
    attention_mask = torch.as_tensor(
        encoded.get("attention_mask", torch.ones_like(input_ids)), dtype=torch.long
    )
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise BlankCheckpointError(
            "Tokenized input_ids and attention_mask must have the same rank-2 shape."
        )
    if not input_ids.numel() or bool((attention_mask.sum(dim=1) == 0).any()):
        raise BlankCheckpointError(
            "Every prompt must produce at least one valid token."
        )
    payload = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }
    record = {
        "prompt_count": len(prompts),
        "prompt_sha256": _json_sha256(list(prompts)),
        "input_shape": list(input_ids.shape),
        "valid_token_count": int(attention_mask.sum().item()),
        "input_ids_sha256": _tensor_sha256(torch, input_ids),
        "attention_mask_sha256": _tensor_sha256(torch, attention_mask),
        "padding": "longest-right",
        "truncation": True,
        "max_length": max_length,
    }
    return payload, record


def _assert_no_gradients(model: Any, name: str) -> int:
    count = sum(int(parameter.grad is not None) for parameter in model.parameters())
    if count:
        raise BlankCheckpointError(f"{name} contains populated parameter gradients.")
    return count


def _validation_logits(torch: Any, model: Any, values: Mapping[str, Any]) -> Any:
    with torch.inference_mode():
        output = model(**values, use_cache=False)
    logits = getattr(output, "logits", None)
    if logits is None:
        raise BlankCheckpointError("Validation forward pass did not return logits.")
    return logits.detach().cpu().contiguous()


def _write_json(path: Path, report: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _publish(
    output: Path, report: dict[str, Any], save: Any | None = None
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        if save is not None:
            save(stage)
        report["artifacts"] = {
            "report_filename": REPORT_FILENAME,
            "files": _artifact_files(stage),
        }
        normalized = json.loads(_canonical_json(report))
        _write_json(stage / REPORT_FILENAME, normalized)
        stage.replace(output)
        return normalized
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def bootstrap_blank_decoder_checkpoint(
    donor: Any,
    *,
    target_family: TargetFamily,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    max_position_embeddings: int,
    output_dir: str | os.PathLike[str],
    calibration_prompts: Sequence[str],
    validation_prompts: Sequence[str],
    tokenizer: Any = None,
    seed: int = 0,
    dtype: Any = "float32",
    donor_dtype: Any | None = None,
    target_dtype: Any | None = None,
    device: Any = "cpu",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    exact_parameter_budget: int | None = None,
    trace_pooling: Literal["token_rows", "mean", "last_valid"] = "token_rows",
    residual_map_method: Literal["sparse_fold", "pca"] = "sparse_fold",
    ridge: float = 1e-5,
    max_operator_delta_ratio: float = 4.0,
    min_final_state_relative_improvement: float = 0.0,
    min_logit_relative_improvement: float = 0.0,
    min_top1_agreement_gain: float = 0.0,
    fit_full_lm_head: bool = True,
    max_full_head_fit_elements: int = 10_000_000,
    max_coordinate_rows: int = 4096,
    vocab_projection_chunk_size: int = 2048,
) -> dict[str, Any]:
    """Bootstrap and atomically publish one config-created decoder checkpoint."""

    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "Blank checkpoint publication requires torch and Transformers."
        ) from exc

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise BlankCheckpointError("output_dir must not already exist.")
    trust = _strict_bool(trust_remote_code, "trust_remote_code")
    local_only = _strict_bool(local_files_only, "local_files_only")
    seed_value = seed
    if isinstance(seed_value, bool) or not isinstance(seed_value, int):
        raise BlankCheckpointError("seed must be an integer.")
    shape = _validate_shape(
        target_family,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=max_position_embeddings,
    )
    calibration = _prompt_sequence(calibration_prompts, "calibration_prompts")
    validation = _prompt_sequence(validation_prompts, "validation_prompts")
    resolved_dtype = _dtype(torch, dtype)
    resolved_donor_dtype = (
        resolved_dtype if donor_dtype is None else _dtype(torch, donor_dtype)
    )
    resolved_target_dtype = (
        resolved_dtype if target_dtype is None else _dtype(torch, target_dtype)
    )
    resolved_device = _device(torch, device)

    source, donor_reference, donor_load_calls = _load_donor(
        torch,
        transformers,
        donor,
        dtype=resolved_donor_dtype,
        device=resolved_device,
        trust_remote_code=trust,
        local_files_only=local_only,
    )
    source_layout = build_decoder_layout(source)
    if int(shape["max_position_embeddings"]) > source_layout.position.max_positions:
        raise BlankCheckpointError(
            "max_position_embeddings cannot exceed the donor position limit."
        )
    source_counts = _parameter_counts(source)
    source_fingerprint_before = _state_dict_sha256(torch, source)
    _assert_no_gradients(source, "donor")

    resolved_tokenizer, tokenizer_load = _load_tokenizer(
        transformers,
        tokenizer,
        donor_reference,
        trust_remote_code=trust,
        local_files_only=local_only,
    )
    vocab_size = source_layout.embeddings.vocab_size
    if len(resolved_tokenizer) > vocab_size:
        raise BlankCheckpointError("Tokenizer vocabulary exceeds the donor vocabulary.")
    special_ids = _special_token_ids(source, resolved_tokenizer, vocab_size)
    tokenizer_report = {
        **_tokenizer_identity(resolved_tokenizer, special_ids),
        **tokenizer_load,
    }

    target, constructor = _target_model(
        torch,
        transformers,
        shape,
        vocab_size=vocab_size,
        special_ids=special_ids,
        tied=source_layout.tie_word_embeddings,
        seed=seed_value,
        dtype=resolved_target_dtype,
        device=resolved_device,
    )
    target_layout = build_decoder_layout(target)
    blank_counts = _parameter_counts(target)
    blank_fingerprint = _state_dict_sha256(torch, target)
    _assert_no_gradients(target, "blank target")
    if blank_counts["unique_parameter_count"] > source_counts["unique_parameter_count"]:
        raise BlankCheckpointError(
            "The blank target cannot have more unique parameters than the donor."
        )
    if exact_parameter_budget is not None:
        budget = _positive_int(exact_parameter_budget, "exact_parameter_budget")
        if blank_counts["unique_parameter_count"] != budget:
            raise BlankCheckpointError(
                "The config-created target does not match exact_parameter_budget: "
                f"{blank_counts['unique_parameter_count']} != {budget}."
            )
    else:
        budget = None

    max_length = int(shape["max_position_embeddings"])
    calibration_inputs, calibration_report = _tokenize(
        torch,
        resolved_tokenizer,
        calibration,
        max_length=max_length,
        device=resolved_device,
    )
    validation_inputs, validation_report = _tokenize(
        torch,
        resolved_tokenizer,
        validation,
        max_length=max_length,
        device=resolved_device,
    )
    if int(calibration_report["valid_token_count"]) < target_layout.hidden_size:
        raise BlankCheckpointError(
            "Calibration must provide at least hidden_size valid token rows."
        )

    result = bootstrap_blank_decoder_models(
        source,
        target,
        calibration_inputs,
        validation_inputs=validation_inputs,
        trace_pooling=trace_pooling,
        residual_map_method=residual_map_method,
        ridge=ridge,
        max_operator_delta_ratio=max_operator_delta_ratio,
        min_final_state_relative_improvement=min_final_state_relative_improvement,
        min_logit_relative_improvement=min_logit_relative_improvement,
        min_top1_agreement_gain=min_top1_agreement_gain,
        fit_full_lm_head=fit_full_lm_head,
        max_full_head_fit_elements=max_full_head_fit_elements,
        max_coordinate_rows=max_coordinate_rows,
        vocab_projection_chunk_size=vocab_projection_chunk_size,
    )
    source_fingerprint_after = _state_dict_sha256(torch, source)
    if source_fingerprint_after != source_fingerprint_before:
        raise BlankCheckpointError("Bootstrap mutated the donor checkpoint.")
    _assert_no_gradients(source, "donor")
    _assert_no_gradients(target, "target")

    accepted = bool(result.transaction.accepted)
    target_fingerprint = _state_dict_sha256(torch, target)
    if not accepted and target_fingerprint != blank_fingerprint:
        raise BlankCheckpointError(
            "Rejected bootstrap did not restore the blank target."
        )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "operation": "bootstrap_blank_decoder_checkpoint",
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "output_dir": str(output),
        "manual_eval_needed": accepted,
        "manual_evaluation_needed": accepted,
        "gate": result.transaction.report["gate"],
        "source": {
            "architecture": _layout_identity(source_layout, source),
            "parameter_counts": source_counts,
            "state_dict_sha256": source_fingerprint_before,
            "state_dict_sha256_after": source_fingerprint_after,
            "unchanged": True,
            "pretrained_load_calls": donor_load_calls,
        },
        "blank_target": {
            "architecture": _layout_identity(target_layout, target),
            "requested_shape": dict(shape),
            "constructor": constructor,
            "pretrained_child_weights_loaded": False,
            "seed": seed_value,
            "state_dict_sha256": blank_fingerprint,
            "parameter_counts": blank_counts,
            "exact_parameter_budget": budget,
            "exact_parameter_budget_matches": budget is None
            or blank_counts["unique_parameter_count"] == budget,
        },
        "candidate": (
            {
                "state_dict_sha256": target_fingerprint,
                "parameter_counts": _parameter_counts(target),
                "saved": accepted,
            }
            if accepted
            else {"saved": False, "state_restored_to_blank": True}
        ),
        "compression": _compression(source_counts, blank_counts),
        "tokenizer": tokenizer_report,
        "tokenization": {
            "calibration": calibration_report,
            "validation": validation_report,
        },
        "bootstrap": {
            "gate": result.transaction.report["gate"],
            "coverage": result.transaction.report["coverage"],
            "rollback": result.transaction.report["rollback"],
            "donor_layer_spans": [list(span) for span in result.donor_layer_spans],
            "fit_reports": dict(result.fit_reports),
            "residual_coordinates": (
                result.residual_coordinates.report.as_dict()
                if getattr(result.residual_coordinates, "report", None) is not None
                else {"method": residual_map_method}
            ),
            "knobs": {
                "trace_pooling": trace_pooling,
                "residual_map_method": residual_map_method,
                "ridge": ridge,
                "max_operator_delta_ratio": max_operator_delta_ratio,
                "min_final_state_relative_improvement": min_final_state_relative_improvement,
                "min_logit_relative_improvement": min_logit_relative_improvement,
                "min_top1_agreement_gain": min_top1_agreement_gain,
                "fit_full_lm_head": fit_full_lm_head,
                "max_full_head_fit_elements": max_full_head_fit_elements,
                "max_coordinate_rows": max_coordinate_rows,
                "vocab_projection_chunk_size": vocab_projection_chunk_size,
            },
        },
        "execution": {
            "zero_gradients": True,
            "training_performed": False,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "recovery": {"performed": False, "steps": 0},
            "dtype": str(resolved_target_dtype).replace("torch.", ""),
            "donor_dtype": str(resolved_donor_dtype).replace("torch.", ""),
            "target_dtype": str(resolved_target_dtype).replace("torch.", ""),
            "device": str(resolved_device),
            "trust_remote_code": trust,
            "local_files_only": local_only,
        },
        "software": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
        },
    }

    if not accepted:
        report["checkpoint_verification"] = {
            "performed": False,
            "reason": "candidate-rejected",
        }
        return _publish(output, report)

    expected_logits = _validation_logits(torch, target, validation_inputs)
    expected_logits_sha256 = _tensor_sha256(torch, expected_logits)

    def save_and_verify(stage: Path) -> None:
        target.save_pretrained(stage, safe_serialization=True)
        resolved_tokenizer.save_pretrained(stage)
        safetensors = list(stage.glob("*.safetensors"))
        if not safetensors or list(stage.glob("*.bin")):
            raise BlankCheckpointError(
                "Accepted checkpoint was not saved only as safetensors."
            )
        reloaded = type(target).from_pretrained(
            stage,
            local_files_only=True,
            torch_dtype=resolved_target_dtype,
        )
        reloaded.to(resolved_device)
        reloaded.eval()
        actual_logits = _validation_logits(torch, reloaded, validation_inputs)
        actual_sha256 = _tensor_sha256(torch, actual_logits)
        exact = torch.equal(expected_logits, actual_logits)
        reloaded_fingerprint = _state_dict_sha256(torch, reloaded)
        if not exact or reloaded_fingerprint != target_fingerprint:
            raise BlankCheckpointError(
                "Reloaded checkpoint does not exactly match the accepted candidate."
            )
        report["checkpoint_verification"] = {
            "performed": True,
            "local_reload": True,
            "exact_output_equality": True,
            "validation_logits_sha256_before_save": expected_logits_sha256,
            "validation_logits_sha256_after_reload": actual_sha256,
            "state_dict_sha256_after_reload": reloaded_fingerprint,
        }

    return _publish(output, report, save_and_verify)


__all__ = [
    "REPORT_FILENAME",
    "REPORT_SCHEMA",
    "BlankCheckpointError",
    "bootstrap_blank_decoder_checkpoint",
]
