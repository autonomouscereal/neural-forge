"""Bounded Hugging Face checkpoint integration for raw-modality tokens.

This module only expands tokenizer/model vocabulary rows.  It does not add a
learned modality path, run a forward pass, create an optimizer, or perform
backpropagation or training.
"""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from neural_forge.raw_modality_tokens import (
    RAW_TOKEN_STRINGS,
    expand_hf_raw_token_vocabulary,
    plan_hf_raw_token_expansion,
)


RAW_MODALITY_REPORT_FILENAME = "neural_forge_hf_raw_modality_report.json"
_ALLOWED_CONFIG_CHANGE_KEYS = frozenset({"tie_word_embeddings", "vocab_size"})


def expand_hf_raw_token_model_in_memory(
    model: Any,
    tokenizer: Any,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Untie and expand an already loaded HF model without checkpoint I/O.

    The model and tokenizer are mutated in place.  Strict inventories ensure
    that only input-embedding/output-head row dimensions change.
    """

    torch = _import_torch()
    normalized_seed = _normalize_seed(seed)
    _validate_model_interface(model)

    modules_before = _module_inventory(model)
    modes_before = _module_training_modes(model)
    state_before = _state_inventory(model)
    config_before = _config_snapshot(model)
    model_class_before = _type_name(model)
    gradients_before = _gradient_inventory(torch, model)
    _assert_zero_gradients(gradients_before, "before expansion")

    input_before = model.get_input_embeddings()
    output_before = model.get_output_embeddings()
    input_weight_before = input_before.weight
    output_weight_before = output_before.weight
    output_bias_before = getattr(output_before, "bias", None)
    weights_tied_before = _weights_tied(input_weight_before, output_weight_before)
    parameter_count_before = _parameter_count(model)

    with _seeded_rng(torch, model, normalized_seed), torch.no_grad():
        from neural_forge.hf_xfamily import _untie_output_head

        untie_helper_changed = bool(_untie_output_head(torch, model))
        _restore_module_training_modes(model, modes_before)
        _assert_module_inventory_unchanged(
            modules_before,
            _module_inventory(model),
            stage="output-head untying",
        )
        if hasattr(model, "config"):
            model.config.tie_word_embeddings = False
        if _weights_tied(
            model.get_input_embeddings().weight,
            model.get_output_embeddings().weight,
        ):
            raise AssertionError("Output head remains tied after explicit untying.")

        parameter_count_after_untie = _parameter_count(model)
        expected_untie_delta = (
            int(output_weight_before.numel())
            if input_weight_before is output_weight_before
            else 0
        )
        actual_untie_delta = parameter_count_after_untie - parameter_count_before
        if actual_untie_delta != expected_untie_delta:
            raise AssertionError(
                "Output-head untying changed the parameter count by "
                f"{actual_untie_delta}, expected {expected_untie_delta}."
            )

        plan = plan_hf_raw_token_expansion(tokenizer, model)
        target_rows = _expected_target_rows(plan, model)
        expected_delta, expected_delta_components = _expected_vocabulary_delta(
            model, target_rows
        )
        state_after_untie = _state_inventory(model)
        allowed_row_state_names = _allowed_row_state_names(model)

        expansion = expand_hf_raw_token_vocabulary(tokenizer, model)
        _restore_module_training_modes(model, modes_before)

    modules_after = _module_inventory(model)
    modes_after = _module_training_modes(model)
    state_after = _state_inventory(model)
    config_after = _config_snapshot(model)
    gradients_after = _gradient_inventory(torch, model)
    _assert_zero_gradients(gradients_after, "after expansion")

    if _type_name(model) != model_class_before:
        raise AssertionError("Model class changed during raw-token expansion.")
    module_comparison = _assert_module_inventory_unchanged(
        modules_before,
        modules_after,
        stage="raw-token expansion",
    )
    if modes_after != modes_before:
        raise AssertionError(
            "Module training modes changed during raw-token expansion."
        )
    state_comparison = _assert_only_row_state_changes(
        state_after_untie,
        state_after,
        allowed_names=allowed_row_state_names,
        target_rows=target_rows,
    )
    loaded_state_comparison = _assert_only_row_state_changes(
        state_before,
        state_after,
        allowed_names=allowed_row_state_names,
        target_rows=target_rows,
    )

    parameter_count_after = _parameter_count(model)
    actual_vocabulary_delta = parameter_count_after - parameter_count_after_untie
    if actual_vocabulary_delta != expected_delta:
        raise AssertionError(
            "Vocabulary expansion changed the parameter count by "
            f"{actual_vocabulary_delta}, expected {expected_delta}."
        )
    if _weights_tied(
        model.get_input_embeddings().weight,
        model.get_output_embeddings().weight,
    ):
        raise AssertionError("Output head became tied during raw-token expansion.")

    config_changes = _mapping_changes(config_before, config_after)
    unexpected_config_changes = [
        change
        for change in config_changes
        if change["path"].rsplit(".", 1)[-1] not in _ALLOWED_CONFIG_CHANGE_KEYS
    ]
    if unexpected_config_changes:
        paths = ", ".join(change["path"] for change in unexpected_config_changes)
        raise AssertionError(
            f"Architecture config changed outside vocabulary fields: {paths}"
        )

    raw_token_ids = {
        token: int(expansion["raw_token_ids"][token]) for token in RAW_TOKEN_STRINGS
    }
    token_id_hash = _token_id_hash(raw_token_ids)
    row_hashes = _raw_row_hashes(torch, model, raw_token_ids)
    input_after = model.get_input_embeddings()
    output_after = model.get_output_embeddings()

    return {
        "operation": "expand_hf_raw_token_model_in_memory",
        "status": "applied",
        "seed": normalized_seed,
        "parameter_count_before": parameter_count_before,
        "parameter_count_after_untie": parameter_count_after_untie,
        "parameter_count_after": parameter_count_after,
        "untie_parameter_delta": actual_untie_delta,
        "expected_untie_parameter_delta": expected_untie_delta,
        "expected_vocabulary_only_parameter_delta": expected_delta,
        "actual_vocabulary_only_parameter_delta": actual_vocabulary_delta,
        "total_parameter_delta": parameter_count_after - parameter_count_before,
        "vocabulary_delta_components": expected_delta_components,
        "target_vocabulary_rows": target_rows,
        "tokenizer_size_before": int(expansion["tokenizer_size_before"]),
        "tokenizer_size_after": int(expansion["tokenizer_size_after"]),
        "added_token_count": int(expansion["added_token_count"]),
        "raw_token_ids": raw_token_ids,
        "raw_token_ids_sha256": token_id_hash,
        "token_id_hash": {
            "algorithm": "sha256",
            "canonical_form": "ordered-json-token-id-pairs",
            "value": token_id_hash,
        },
        "row_initialization": {
            "seed": normalized_seed,
            "seeded_deterministic": True,
            "torch_rng_seeded": True,
            "torch_rng_state_restored": True,
            "raw_row_initializer": expansion["initialization"]["method"],
            "raw_row_initializer_seed_mode": "token-hash-deterministic",
            "input_token_ids": list(expansion["initialization"]["input_token_ids"]),
            "output_token_ids": list(expansion["initialization"]["output_token_ids"]),
            "output_bias_token_ids": list(
                expansion["initialization"]["output_bias_token_ids"]
            ),
            **row_hashes,
        },
        "output_head": {
            "was_tied": weights_tied_before,
            "untie_helper_called": True,
            "untie_helper_changed": untie_helper_changed,
            "is_untied_after": True,
        },
        "row_counts": {
            "input_embedding_before": int(input_weight_before.shape[0]),
            "input_embedding_after": int(input_after.weight.shape[0]),
            "output_head_before": int(output_weight_before.shape[0]),
            "output_head_after": int(output_after.weight.shape[0]),
            "output_bias_before": (
                int(output_bias_before.shape[0])
                if output_bias_before is not None
                else None
            ),
            "output_bias_after": (
                int(output_after.bias.shape[0])
                if getattr(output_after, "bias", None) is not None
                else None
            ),
        },
        "architecture_inventory": {
            "model_class_before": model_class_before,
            "model_class_after": _type_name(model),
            "module_count": len(modules_before),
            "module_names_unchanged": True,
            "module_types_unchanged": True,
            "unchanged_modules": modules_before,
            "added_modules": module_comparison["added_modules"],
            "removed_modules": module_comparison["removed_modules"],
            "module_type_changes": module_comparison["module_type_changes"],
            "module_training_modes_unchanged": modes_after == modes_before,
            "state_tensor_names_unchanged": state_comparison[
                "state_tensor_names_unchanged"
            ],
            "allowed_row_state_names": sorted(allowed_row_state_names),
            "vocabulary_row_shape_changes": state_comparison["shape_changes"],
            "loaded_to_final_row_shape_changes": loaded_state_comparison[
                "shape_changes"
            ],
            "only_embedding_and_head_rows_changed": True,
            "config_changes": config_changes,
            "unexpected_config_changes": [],
        },
        "gradients_before": gradients_before,
        "gradients_after": gradients_after,
        "zero_gradients": True,
        "operation_bounds": {
            "learned_modality_modules_added": [],
            "encoder_added": False,
            "decoder_added": False,
            "projector_added": False,
            "adapter_added": False,
            "bridge_added": False,
            "modality_specific_tower_added": False,
            "optimizer_created": False,
            "backward_called": False,
            "training_performed": False,
            "forward_called": False,
            "only_head_untie_tokenizer_and_vocabulary_rows_mutated": True,
        },
        "raw_expansion": expansion,
    }


def expand_hf_raw_token_checkpoint(
    model_id_or_path: str | Path,
    output_dir: str | Path,
    *,
    tokenizer_model: str | Path | None = None,
    torch_dtype: str = "auto",
    device: str = "cpu",
    seed: int = 0,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Load, expand, and save one causal-LM checkpoint plus its tokenizer."""

    torch = _import_torch()
    transformers = _import_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    model_source = str(model_id_or_path)
    tokenizer_source = str(tokenizer_model or model_id_or_path)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {
        "device_map": None,
        "trust_remote_code": trust_remote_code,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_source,
        **model_kwargs,
    )
    if str(device).lower() != "auto":
        model.to(device)

    report = expand_hf_raw_token_model_in_memory(
        model,
        tokenizer,
        seed=seed,
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)

    report.update(
        {
            "operation": "expand_hf_raw_token_checkpoint",
            "source_model": model_source,
            "source_tokenizer": tokenizer_source,
            "output_dir": str(target),
            "report_file": str(target / RAW_MODALITY_REPORT_FILENAME),
            "requested_dtype": str(torch_dtype),
            "requested_device": str(device),
            "actual_parameter_dtypes": sorted(
                {str(parameter.dtype) for parameter in model.parameters()}
            ),
            "actual_parameter_devices": sorted(
                {str(parameter.device) for parameter in model.parameters()}
            ),
            "trust_remote_code": bool(trust_remote_code),
            "safe_serialization": True,
            "saved_artifacts": sorted(
                {
                    *(path.name for path in target.iterdir()),
                    RAW_MODALITY_REPORT_FILENAME,
                }
            ),
        }
    )
    _write_json(target / RAW_MODALITY_REPORT_FILENAME, report)
    return report


# Short alias for callers that already hold model/tokenizer objects.
expand_hf_raw_token_model = expand_hf_raw_token_model_in_memory


def _validate_model_interface(model: Any) -> None:
    required = (
        "get_input_embeddings",
        "get_output_embeddings",
        "set_output_embeddings",
        "resize_token_embeddings",
        "named_modules",
        "state_dict",
        "parameters",
    )
    missing = [name for name in required if not callable(getattr(model, name, None))]
    if missing:
        raise TypeError(f"Model is missing required methods: {', '.join(missing)}")
    input_module = model.get_input_embeddings()
    output_module = model.get_output_embeddings()
    if input_module is None or output_module is None:
        raise ValueError("Model must expose input embeddings and an output head.")
    if getattr(input_module, "weight", None) is None:
        raise ValueError("Input embedding module must expose a weight tensor.")
    if getattr(output_module, "weight", None) is None:
        raise ValueError("Output head must expose a weight tensor.")


def _expected_target_rows(plan: Mapping[str, Any], model: Any) -> int:
    proposed_ids = [int(value) for value in plan["proposed_token_ids"].values()]
    return max(
        int(plan["tokenizer_size_after_expected"]),
        max(proposed_ids) + 1,
        int(model.get_input_embeddings().weight.shape[0]),
        int(model.get_output_embeddings().weight.shape[0]),
    )


def _expected_vocabulary_delta(
    model: Any,
    target_rows: int,
) -> tuple[int, list[dict[str, Any]]]:
    input_weight = model.get_input_embeddings().weight
    output_module = model.get_output_embeddings()
    output_weight = output_module.weight
    components = [
        _row_delta_component("input_embedding.weight", input_weight, target_rows),
        _row_delta_component("output_head.weight", output_weight, target_rows),
    ]
    output_bias = getattr(output_module, "bias", None)
    if output_bias is not None:
        if output_bias.ndim != 1 or int(output_bias.shape[0]) != int(
            output_weight.shape[0]
        ):
            raise AssertionError("Output-head bias rows do not match output-head rows.")
        components.append(
            _row_delta_component("output_head.bias", output_bias, target_rows)
        )
    return sum(int(item["parameter_delta"]) for item in components), components


def _row_delta_component(
    name: str,
    tensor: Any,
    target_rows: int,
) -> dict[str, Any]:
    rows_before = int(tensor.shape[0])
    if target_rows < rows_before:
        raise AssertionError("Raw-token expansion must not shrink vocabulary rows.")
    values_per_row = int(tensor.numel()) // rows_before
    return {
        "name": name,
        "rows_before": rows_before,
        "rows_after_expected": target_rows,
        "values_per_row": values_per_row,
        "parameter_delta": (target_rows - rows_before) * values_per_row,
    }


def _allowed_row_state_names(model: Any) -> set[str]:
    input_module = model.get_input_embeddings()
    output_module = model.get_output_embeddings()
    names = {_qualify(path, "weight") for path in _module_paths(model, input_module)}
    names.update(
        _qualify(path, "weight") for path in _module_paths(model, output_module)
    )
    if getattr(output_module, "bias", None) is not None:
        names.update(
            _qualify(path, "bias") for path in _module_paths(model, output_module)
        )
    return names


def _assert_only_row_state_changes(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    *,
    allowed_names: set[str],
    target_rows: int,
) -> dict[str, Any]:
    before_names = set(before)
    after_names = set(after)
    if before_names != after_names:
        added = sorted(after_names - before_names)
        removed = sorted(before_names - after_names)
        raise AssertionError(
            f"State tensor inventory changed; added={added}, removed={removed}."
        )
    shape_changes: list[dict[str, Any]] = []
    for name in sorted(before):
        old = before[name]
        new = after[name]
        if old["dtype"] != new["dtype"]:
            raise AssertionError(f"State tensor dtype changed for {name}.")
        if old["shape"] == new["shape"]:
            continue
        if name not in allowed_names:
            raise AssertionError(f"Non-vocabulary tensor shape changed for {name}.")
        old_shape = list(old["shape"])
        new_shape = list(new["shape"])
        if not old_shape or old_shape[1:] != new_shape[1:]:
            raise AssertionError(f"Non-row dimensions changed for {name}.")
        if new_shape[0] != target_rows or old_shape[0] > new_shape[0]:
            raise AssertionError(f"Unexpected vocabulary row change for {name}.")
        shape_changes.append(
            {"name": name, "shape_before": old_shape, "shape_after": new_shape}
        )
    return {
        "state_tensor_names_unchanged": True,
        "shape_changes": shape_changes,
    }


def _assert_module_inventory_unchanged(
    before: list[dict[str, str]],
    after: list[dict[str, str]],
    *,
    stage: str,
) -> dict[str, Any]:
    before_by_name = {item["name"]: item["type"] for item in before}
    after_by_name = {item["name"]: item["type"] for item in after}
    added = sorted(set(after_by_name) - set(before_by_name))
    removed = sorted(set(before_by_name) - set(after_by_name))
    type_changes = [
        {
            "name": name,
            "type_before": before_by_name[name],
            "type_after": after_by_name[name],
        }
        for name in sorted(set(before_by_name) & set(after_by_name))
        if before_by_name[name] != after_by_name[name]
    ]
    if added or removed or type_changes:
        raise AssertionError(
            f"Module inventory changed during {stage}; added={added}, "
            f"removed={removed}, type_changes={type_changes}."
        )
    return {
        "added_modules": added,
        "removed_modules": removed,
        "module_type_changes": type_changes,
    }


def _module_inventory(model: Any) -> list[dict[str, str]]:
    return [
        {"name": name, "type": _type_name(module)}
        for name, module in _named_modules(model)
    ]


def _module_training_modes(model: Any) -> dict[str, bool]:
    return {name: bool(module.training) for name, module in _named_modules(model)}


def _restore_module_training_modes(model: Any, modes: Mapping[str, bool]) -> None:
    modules = dict(_named_modules(model))
    if set(modules) != set(modes):
        raise AssertionError("Cannot restore modes after module inventory changed.")
    for name, training in modes.items():
        modules[name].training = bool(training)


def _named_modules(model: Any) -> list[tuple[str, Any]]:
    try:
        return list(model.named_modules(remove_duplicate=False))
    except TypeError:  # pragma: no cover - compatibility with older torch
        return list(model.named_modules())


def _module_paths(model: Any, target: Any) -> list[str]:
    paths = [name for name, module in _named_modules(model) if module is target]
    if not paths:
        raise AssertionError("Embedding/head module is absent from model inventory.")
    return paths


def _qualify(module_name: str, field_name: str) -> str:
    return f"{module_name}.{field_name}" if module_name else field_name


def _state_inventory(model: Any) -> dict[str, dict[str, Any]]:
    return {
        str(name): {
            "shape": [int(value) for value in tensor.shape],
            "dtype": str(tensor.dtype),
        }
        for name, tensor in model.state_dict().items()
    }


def _config_snapshot(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        return {}
    to_dict = getattr(config, "to_dict", None)
    source = to_dict() if callable(to_dict) else vars(config)
    normalized = _json_value(source)
    if not isinstance(normalized, dict):
        raise TypeError("Model config must serialize to a mapping.")
    return normalized


def _mapping_changes(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    prefix: str = "",
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for key in sorted(set(before) | set(after)):
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in before:
            changes.append({"path": path, "before": None, "after": after[key]})
            continue
        if key not in after:
            changes.append({"path": path, "before": before[key], "after": None})
            continue
        left = before[key]
        right = after[key]
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            changes.extend(_mapping_changes(left, right, path))
        elif left != right:
            changes.append({"path": path, "before": left, "after": right})
    return changes


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _gradient_inventory(torch: Any, model: Any) -> dict[str, Any]:
    gradient_tensor_count = 0
    gradient_element_count = 0
    nonzero_gradient_element_count = 0
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient_tensor_count += 1
        gradient_element_count += int(gradient.numel())
        values = (
            gradient.detach().coalesce().values() if gradient.is_sparse else gradient
        )
        nonzero_gradient_element_count += int(torch.count_nonzero(values).item())
    return {
        "gradient_tensor_count": gradient_tensor_count,
        "gradient_element_count": gradient_element_count,
        "nonzero_gradient_element_count": nonzero_gradient_element_count,
        "all_gradient_values_zero": nonzero_gradient_element_count == 0,
    }


def _assert_zero_gradients(inventory: Mapping[str, Any], stage: str) -> None:
    if int(inventory["nonzero_gradient_element_count"]) != 0:
        raise AssertionError(f"Model has nonzero gradients {stage}.")


def _token_id_hash(raw_token_ids: Mapping[str, int]) -> str:
    ordered = [[token, int(raw_token_ids[token])] for token in RAW_TOKEN_STRINGS]
    payload = json.dumps(
        ordered,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _raw_row_hashes(
    torch: Any,
    model: Any,
    raw_token_ids: Mapping[str, int],
) -> dict[str, Any]:
    token_ids = [int(raw_token_ids[token]) for token in RAW_TOKEN_STRINGS]
    tensors = {
        "input_embedding": model.get_input_embeddings().weight,
        "output_head": model.get_output_embeddings().weight,
    }
    output_bias = getattr(model.get_output_embeddings(), "bias", None)
    if output_bias is not None:
        tensors["output_bias"] = output_bias

    component_hashes: dict[str, str] = {}
    combined = hashlib.sha256()
    for name, tensor in tensors.items():
        digest = _tensor_rows_hash(torch, tensor, token_ids)
        component_hashes[f"{name}_raw_rows_sha256"] = digest
        combined.update(name.encode("ascii"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\0")
    return {
        **component_hashes,
        "combined_raw_rows_sha256": combined.hexdigest(),
        "hash_algorithm": "sha256",
        "hash_token_order": "RAW_TOKEN_STRINGS",
    }


def _tensor_rows_hash(torch: Any, tensor: Any, token_ids: list[int]) -> str:
    indices = torch.tensor(token_ids, dtype=torch.long, device=tensor.device)
    rows = tensor.detach().index_select(0, indices).contiguous().cpu()
    byte_payload = rows.view(torch.uint8).numpy().tobytes()
    metadata = json.dumps(
        {"dtype": str(rows.dtype), "shape": list(rows.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(b"\0")
    digest.update(byte_payload)
    return digest.hexdigest()


@contextmanager
def _seeded_rng(torch: Any, model: Any, seed: int) -> Iterator[None]:
    cuda_devices = sorted(
        {
            int(parameter.device.index)
            for parameter in model.parameters()
            if parameter.device.type == "cuda" and parameter.device.index is not None
        }
    )
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(seed)
        yield


def _normalize_seed(seed: Any) -> int:
    if isinstance(seed, bool):
        raise TypeError("seed must be an integer, not bool.")
    try:
        value = operator.index(seed)
    except TypeError as exc:
        raise TypeError("seed must be an integer.") from exc
    if not 0 <= value < 2**63:
        raise ValueError("seed must be between 0 and 2**63 - 1.")
    return value


def _parameter_count(model: Any) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def _weights_tied(left: Any, right: Any) -> bool:
    if left is right:
        return True
    left_pointer = getattr(left, "data_ptr", None)
    right_pointer = getattr(right, "data_ptr", None)
    return bool(
        callable(left_pointer)
        and callable(right_pointer)
        and int(left_pointer()) == int(right_pointer())
    )


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _dtype_from_name(torch: Any, name: str) -> Any | None:
    lowered = str(name).lower()
    if lowered in {"auto", "source"}:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[lowered]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("PyTorch is required for HF raw-token expansion.") from exc
    return torch


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "Transformers is required for HF raw-token checkpoint expansion."
        ) from exc
    return transformers


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "RAW_MODALITY_REPORT_FILENAME",
    "expand_hf_raw_token_checkpoint",
    "expand_hf_raw_token_model",
    "expand_hf_raw_token_model_in_memory",
]
