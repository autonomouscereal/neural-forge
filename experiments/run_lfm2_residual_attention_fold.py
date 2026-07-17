#!/usr/bin/env python3
"""Compile Liquid LFM2.5-230M into a smaller residual-attention-aware child.

This runner is intentionally a folding compiler, not a recovery trainer:

* the parent is traced once into an immutable content-addressed cache;
* complete GQA groups and coupled SwiGLU neurons are selected as real medoids;
* provisional producers and aggregated consumer priors overwrite every blank
  target tensor without consulting random target values;
* true compact-child attention contexts, residual endpoints, and post-SwiGLU
  activations are recollected on the exact cached prefixes;
* Q/K/V/O and gate/up/down are closed-loop fitted as one causal layer circuit;
* each final layer candidate is committed or rolled back against cached/full
  decode, residual-path, and child-generated greedy trajectory gates.

No optimizer is constructed, no loss is minimized by iteration, and no
parameter gradient is written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

# This compiler only accepts explicit local model directories.  Force the
# underlying libraries offline before importing Torch/Transformers-backed
# Neural Forge modules so no future regression can resolve a hub model ID.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neural_forge.hf_activation_cluster import (
    GatedMLPActivationClusterPlan,
    GQAActivationClusterPlan,
    build_gated_mlp_activation_cluster_plan,
    build_gqa_activation_cluster_plan,
)
from neural_forge.hf_lfm2_canonical import (
    Lfm2AttentionContract,
    Lfm2ShortConvContract,
    inspect_lfm2_layer,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_closed_loop_fold import (
    Lfm2AttentionFoldEvidence,
    Lfm2ClosedLoopFitRejected,
    Lfm2ClosedLoopFoldConfig,
    Lfm2ClosedLoopFoldEvidence,
    Lfm2ClosedLoopFoldPlan,
    Lfm2MLPFoldEvidence,
    build_lfm2_closed_loop_fold_plan,
    install_lfm2_closed_loop_fold_plan,
    restore_lfm2_layer_snapshot,
    snapshot_lfm2_layer,
)


RUN_SCHEMA = "neural-forge-lfm2-residual-attention-fold-run-v1"
STATE_SCHEMA = "neural-forge-lfm2-residual-attention-fold-state-v1"
TARGET_SCHEMA = "neural-forge-lfm2-fixed-225957632-target-v1"
DYNAMIC_TARGET_SCHEMA = "neural-forge-lfm2-dynamic-target-v1"
TRACE_ROWS_SCHEMA = "neural-forge-lfm2-circuit-rows-v1"

SOURCE_QUERY_HEADS = 16
SOURCE_KEY_VALUE_HEADS = 8
TARGET_QUERY_HEADS = 14
TARGET_KEY_VALUE_HEADS = 7
FIXED_HEAD_DIM = 64
FIXED_HIDDEN_SIZE = 1024
SOURCE_INTERMEDIATE_SIZE = 2560
TARGET_INTERMEDIATE_SIZE = 2528
EXPECTED_PARENT_UNIQUE_PARAMETERS = 229_693_184
EXPECTED_TARGET_UNIQUE_PARAMETERS = 225_957_632
EXPECTED_PARAMETER_SAVINGS = 3_735_552

_PHASE_PATTERN = re.compile(r"^(prefill|decode\.(\d{3}))\.")
_LAYER_PATTERN = re.compile(r"^(prefill|decode\.\d{3})\.layer\.(\d{3})\.(.+)$")
_TRACE_FIELDS = frozenset(
    {
        "attention.context",
        "attention.k",
        "attention.q",
        "attention.v",
        "mlp.gate",
        "mlp.post_swiglu",
        "mlp.up",
        "residual.post_layer",
        "residual.post_operator",
        "residual.pre",
    }
)
_ATTENTION_FOLDED_KEYS = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.out_proj.weight",
)
_MLP_FOLDED_KEYS = (
    "feed_forward.w1.weight",
    "feed_forward.w3.weight",
    "feed_forward.w2.weight",
)


class Lfm2ResidualAttentionRunnerError(RuntimeError):
    """The zero-gradient LFM2 folding run cannot prove a required invariant."""


@dataclass(frozen=True, slots=True)
class Lfm2TargetSpec:
    """One automatically selected target topology for the local LFM parent."""

    query_heads: int
    key_value_heads: int
    intermediate_size: int

    def __post_init__(self) -> None:
        for name in ("query_heads", "key_value_heads", "intermediate_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise Lfm2ResidualAttentionRunnerError(
                    f"target {name} must be a positive integer."
                )
        if self.query_heads % self.key_value_heads:
            raise Lfm2ResidualAttentionRunnerError(
                "Target query heads must be divisible by key/value heads."
            )

    def as_dict(self) -> dict[str, int]:
        return {
            "query_heads": self.query_heads,
            "key_value_heads": self.key_value_heads,
            "intermediate_size": self.intermediate_size,
        }


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2ResidualAttentionRunnerError(
            "A report value is not canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path, *, chunk_size: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor, *, chunk_size: int = 8 * 1024**2) -> str:
    tensor = value.detach()
    if tensor.device.type == "meta" or tensor.layout != torch.strided:
        raise Lfm2ResidualAttentionRunnerError(
            "Cannot hash a meta or non-strided tensor."
        )
    tensor = tensor.contiguous()
    raw = tensor.view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    digest.update(
        _canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    )
    for start in range(0, int(raw.numel()), chunk_size):
        piece = raw[start : start + chunk_size].to(device="cpu").contiguous()
        digest.update(piece.numpy().tobytes())
    return digest.hexdigest()


def model_state_sha256(model: nn.Module) -> str:
    """Hash every named state tensor, including tied state-dict aliases."""

    rows = []
    for name, value in sorted(model.state_dict().items()):
        if not isinstance(value, Tensor):
            raise Lfm2ResidualAttentionRunnerError(
                f"Model state entry is not a tensor: {name}."
            )
        rows.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _tensor_sha256(value),
            }
        )
    if not rows:
        raise Lfm2ResidualAttentionRunnerError("Model state is empty.")
    return _json_sha256(rows)


def unique_parameter_count(model: nn.Module) -> int:
    """Count learned scalars once even when embeddings and LM head are tied."""

    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        total += int(parameter.numel())
    return total


def _gradient_audit(model: nn.Module) -> dict[str, Any]:
    parameters = tuple(model.parameters())
    return {
        "parameter_count": len(parameters),
        "parameters_with_grad": sum(value.grad is not None for value in parameters),
        "nonfinite_gradients": sum(
            value.grad is not None
            and not bool(torch.isfinite(value.grad.detach()).all().item())
            for value in parameters
        ),
        "zero_gradient": all(value.grad is None for value in parameters),
    }


def fixed_contract_accounting() -> dict[str, Any]:
    """Return the independently verified real LFM2.5-230M target accounting."""

    return {
        "schema": TARGET_SCHEMA,
        "source_unique_parameters": EXPECTED_PARENT_UNIQUE_PARAMETERS,
        "target_unique_parameters": EXPECTED_TARGET_UNIQUE_PARAMETERS,
        "learned_parameter_savings": EXPECTED_PARAMETER_SAVINGS,
        "fraction_removed": (
            EXPECTED_PARAMETER_SAVINGS / EXPECTED_PARENT_UNIQUE_PARAMETERS
        ),
        "source_query_heads": SOURCE_QUERY_HEADS,
        "source_key_value_heads": SOURCE_KEY_VALUE_HEADS,
        "target_query_heads": TARGET_QUERY_HEADS,
        "target_key_value_heads": TARGET_KEY_VALUE_HEADS,
        "head_dim": FIXED_HEAD_DIM,
        "source_intermediate_size": SOURCE_INTERMEDIATE_SIZE,
        "target_intermediate_size": TARGET_INTERMEDIATE_SIZE,
        "hidden_size": FIXED_HIDDEN_SIZE,
    }


def _config_value(config: Any, name: str) -> Any:
    if not hasattr(config, name):
        raise Lfm2ResidualAttentionRunnerError(
            f"Parent LFM2 config is missing {name!r}."
        )
    return getattr(config, name)


def build_lfm2_target_config(
    parent_config: Any,
    target: Lfm2TargetSpec,
) -> Any:
    """Clone the local Liquid config and apply a validated compact topology."""

    if str(_config_value(parent_config, "model_type")) != "lfm2":
        raise Lfm2ResidualAttentionRunnerError(
            "The dynamic compiler accepts only model_type='lfm2'."
        )
    if not isinstance(target, Lfm2TargetSpec):
        raise Lfm2ResidualAttentionRunnerError("target must be an Lfm2TargetSpec.")
    observed = {
        "hidden_size": int(_config_value(parent_config, "hidden_size")),
        "intermediate_size": int(_config_value(parent_config, "intermediate_size")),
        "num_attention_heads": int(_config_value(parent_config, "num_attention_heads")),
        "num_key_value_heads": int(_config_value(parent_config, "num_key_value_heads")),
    }
    expected = {
        "hidden_size": FIXED_HIDDEN_SIZE,
        "intermediate_size": SOURCE_INTERMEDIATE_SIZE,
        "num_attention_heads": SOURCE_QUERY_HEADS,
        "num_key_value_heads": SOURCE_KEY_VALUE_HEADS,
    }
    if observed != expected:
        raise Lfm2ResidualAttentionRunnerError(
            f"Parent topology is not the validated LFM2.5-230M contract: "
            f"{observed} != {expected}."
        )
    parent_head_dim = getattr(parent_config, "head_dim", FIXED_HEAD_DIM)
    if parent_head_dim is not None and int(parent_head_dim) != FIXED_HEAD_DIM:
        raise Lfm2ResidualAttentionRunnerError(
            "The parent runtime head dimension is not 64."
        )
    if target.query_heads > observed["num_attention_heads"]:
        raise Lfm2ResidualAttentionRunnerError(
            "Target query heads cannot exceed the parent."
        )
    if target.key_value_heads > observed["num_key_value_heads"]:
        raise Lfm2ResidualAttentionRunnerError(
            "Target key/value heads cannot exceed the parent."
        )
    source_ratio = observed["num_attention_heads"] // observed["num_key_value_heads"]
    target_ratio = target.query_heads // target.key_value_heads
    if source_ratio != target_ratio:
        raise Lfm2ResidualAttentionRunnerError(
            "The current coupled GQA operator requires the parent and target "
            "to keep the same query-heads-per-KV-group ratio."
        )
    if target.intermediate_size > observed["intermediate_size"]:
        raise Lfm2ResidualAttentionRunnerError(
            "Target intermediate size cannot exceed the parent."
        )

    result = copy.deepcopy(parent_config)
    result.num_attention_heads = target.query_heads
    result.num_key_value_heads = target.key_value_heads
    result.num_heads = target.query_heads
    result.head_dim = FIXED_HEAD_DIM
    result.intermediate_size = target.intermediate_size
    result.block_ff_dim = target.intermediate_size
    result.block_auto_adjust_ff_dim = False
    payload = result.to_dict()
    required = {
        "num_attention_heads": target.query_heads,
        "num_key_value_heads": target.key_value_heads,
        "num_heads": target.query_heads,
        "head_dim": FIXED_HEAD_DIM,
        "intermediate_size": target.intermediate_size,
        "block_ff_dim": target.intermediate_size,
        "block_auto_adjust_ff_dim": False,
    }
    if any(payload.get(name) != value for name, value in required.items()):
        raise Lfm2ResidualAttentionRunnerError(
            "The serialized target config dropped a required modern/legacy alias."
        )
    return result


def build_fixed_lfm2_target_config(parent_config: Any) -> Any:
    """Compatibility wrapper for the original 14Q/7KV, 2528-FFN target."""

    return build_lfm2_target_config(
        parent_config,
        Lfm2TargetSpec(
            query_heads=TARGET_QUERY_HEADS,
            key_value_heads=TARGET_KEY_VALUE_HEADS,
            intermediate_size=TARGET_INTERMEDIATE_SIZE,
        ),
    )


def target_config_identity(config: Any) -> dict[str, Any]:
    payload = config.to_dict()
    return {
        "schema": DYNAMIC_TARGET_SCHEMA,
        "config": payload,
        "config_sha256": _json_sha256(payload),
    }


def _read_local_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise Lfm2ResidualAttentionRunnerError(f"Local {name} file is missing: {path}.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Lfm2ResidualAttentionRunnerError(
            f"Could not read local {name}: {path}."
        ) from exc
    if not isinstance(value, Mapping):
        raise Lfm2ResidualAttentionRunnerError(
            f"Local {name} must contain a JSON object."
        )
    return dict(value)


def _resolve_local_model_directory(
    model_directory: str | os.PathLike[str],
) -> Path:
    raw = os.fspath(model_directory)
    if "://" in raw:
        raise Lfm2ResidualAttentionRunnerError(
            "Model sources must be explicit local directories, not URLs."
        )
    directory = Path(raw).expanduser()
    if not directory.is_dir():
        raise Lfm2ResidualAttentionRunnerError(
            f"Local model directory is missing: {directory}."
        )
    return directory.resolve(strict=True)


def load_local_lfm2_config(model_directory: str | os.PathLike[str]) -> Any:
    """Construct concrete Lfm2Config directly from local config.json."""

    from transformers import Lfm2Config

    directory = _resolve_local_model_directory(model_directory)
    payload = _read_local_json(directory / "config.json", "LFM2 config")
    if payload.get("model_type") != "lfm2":
        raise Lfm2ResidualAttentionRunnerError(
            "Local config.json is not an LFM2 configuration."
        )
    return Lfm2Config(**payload)


def _local_safetensor_shards(directory: Path) -> tuple[Path, ...]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_local_json(index_path, "safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise Lfm2ResidualAttentionRunnerError(
                "Local safetensors index has no weight_map."
            )
        names = tuple(sorted({str(value) for value in weight_map.values()}))
        shards = tuple(directory / name for name in names)
    else:
        single = directory / "model.safetensors"
        shards = (
            (single,)
            if single.is_file()
            else tuple(sorted(directory.glob("model-*.safetensors")))
        )
    if not shards or any(not path.is_file() for path in shards):
        raise Lfm2ResidualAttentionRunnerError(
            f"Local LFM2 safetensors are incomplete in {directory}."
        )
    return shards


def _storage_identity(value: Tensor) -> tuple[int, int, tuple[int, ...]]:
    return (
        int(value.untyped_storage().data_ptr()),
        int(value.storage_offset()),
        tuple(int(size) for size in value.shape),
    )


def load_local_lfm2_model(
    model_directory: str | os.PathLike[str],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> nn.Module:
    """Load concrete Lfm2ForCausalLM from local JSON and safetensors only."""

    from safetensors import safe_open
    from transformers import GenerationConfig, Lfm2ForCausalLM

    directory = _resolve_local_model_directory(model_directory)
    config = load_local_lfm2_config(directory)
    # Construct with PyTorch's normal FP32 initialization semantics, then cast
    # the complete module before copying local BF16/FP16 weights.  LFM2 owns
    # non-persistent runtime buffers whose initialization can depend on the
    # global default dtype; changing that default silently changes execution
    # even when every saved parameter byte is identical.
    model = Lfm2ForCausalLM(config)
    model.to(dtype=dtype)
    model.tie_weights()
    target_state = model.state_dict()
    loaded: set[str] = set()
    unexpected: list[str] = []
    with torch.no_grad():
        for shard in _local_safetensor_shards(directory):
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in loaded:
                        raise Lfm2ResidualAttentionRunnerError(
                            f"Duplicate local tensor across shards: {name}."
                        )
                    if name not in target_state:
                        unexpected.append(name)
                        continue
                    source = handle.get_tensor(name)
                    target = target_state[name]
                    if tuple(source.shape) != tuple(target.shape):
                        raise Lfm2ResidualAttentionRunnerError(
                            f"Local tensor shape mismatch for {name}: "
                            f"{tuple(source.shape)} != {tuple(target.shape)}."
                        )
                    target.copy_(source.to(dtype=target.dtype))
                    loaded.add(name)
    if unexpected:
        raise Lfm2ResidualAttentionRunnerError(
            f"Local safetensors contain unexpected state keys: {unexpected}."
        )
    loaded_storage = {_storage_identity(target_state[name]) for name in loaded}
    missing = sorted(
        name
        for name, value in target_state.items()
        if name not in loaded and _storage_identity(value) not in loaded_storage
    )
    if missing:
        raise Lfm2ResidualAttentionRunnerError(
            f"Local safetensors are missing non-aliased state keys: {missing}."
        )
    generation_path = directory / "generation_config.json"
    if generation_path.is_file():
        model.generation_config = GenerationConfig(
            **_read_local_json(generation_path, "generation config")
        )
    model.to(device=torch.device(device), dtype=dtype)
    model.eval()
    return model


def _special_token_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "content" in value:
        return str(value["content"])
    return value


def load_local_tokenizer(model_directory: str | os.PathLike[str]) -> Any:
    """Construct the fast tokenizer directly from local tokenizer artifacts."""

    from transformers import PreTrainedTokenizerFast

    directory = _resolve_local_model_directory(model_directory)
    tokenizer_path = directory / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise Lfm2ResidualAttentionRunnerError(
            f"Local tokenizer.json is missing: {tokenizer_path}."
        )
    kwargs: dict[str, Any] = {}
    config_path = directory / "tokenizer_config.json"
    if config_path.is_file():
        kwargs.update(_read_local_json(config_path, "tokenizer config"))
    special_path = directory / "special_tokens_map.json"
    if special_path.is_file():
        kwargs.update(_read_local_json(special_path, "special-token map"))
    for name in (
        "tokenizer_class",
        "backend",
        "is_local",
        "use_fast",
        "sp_model_kwargs",
    ):
        kwargs.pop(name, None)
    for name in (
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ):
        if name in kwargs:
            kwargs[name] = _special_token_value(kwargs[name])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        **kwargs,
    )
    chat_template_path = directory / "chat_template.jinja"
    if chat_template_path.is_file():
        tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    return tokenizer


def direct_local_io_audit(
    model_directory: str | os.PathLike[str],
) -> dict[str, Any]:
    directory = _resolve_local_model_directory(model_directory)
    shards = _local_safetensor_shards(directory)
    return {
        "model_directory": str(directory),
        "config_loader": "direct_local_json_to_concrete_lfm2_config",
        "model_loader": "direct_local_safetensors_to_concrete_lfm2_model",
        "tokenizer_loader": "direct_local_tokenizer_json",
        "checkpoint_writer": "direct_local_json_and_safetensors",
        "safetensor_files": [path.name for path in shards],
        "auto_loader_used": False,
        "from_pretrained_used": False,
        "hub_download_used": False,
        "network_allowed": False,
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "transformers_offline": (os.environ.get("TRANSFORMERS_OFFLINE") == "1"),
    }


def validate_target_aliases(
    config: Any,
    target: Lfm2TargetSpec,
) -> dict[str, Any]:
    if not isinstance(target, Lfm2TargetSpec):
        raise Lfm2ResidualAttentionRunnerError("target must be an Lfm2TargetSpec.")
    observed = {
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "num_heads": getattr(config, "num_heads", None),
        "head_dim": getattr(config, "head_dim", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "block_ff_dim": getattr(config, "block_ff_dim", None),
        "block_auto_adjust_ff_dim": getattr(config, "block_auto_adjust_ff_dim", None),
    }
    expected = {
        "num_attention_heads": target.query_heads,
        "num_key_value_heads": target.key_value_heads,
        "num_heads": target.query_heads,
        "head_dim": FIXED_HEAD_DIM,
        "intermediate_size": target.intermediate_size,
        "block_ff_dim": target.intermediate_size,
        "block_auto_adjust_ff_dim": False,
    }
    # Transformers 5.12.1 consumes ``block_ff_dim`` while constructing
    # Lfm2Config and may not retain the legacy attribute in memory even when
    # config.json contains it.  The modern runtime width remains authoritative;
    # callers that load a checkpoint restore the alias before validation so a
    # subsequent save remains shape-stable.
    if observed != expected:
        raise Lfm2ResidualAttentionRunnerError(
            f"Dynamic target aliases changed: {observed} != {expected}."
        )
    return {"observed": observed, "expected": expected, "exact": True}


def validate_fixed_target_aliases(config: Any) -> dict[str, Any]:
    """Compatibility wrapper for the original fixed topology."""

    return validate_target_aliases(
        config,
        Lfm2TargetSpec(
            query_heads=TARGET_QUERY_HEADS,
            key_value_heads=TARGET_KEY_VALUE_HEADS,
            intermediate_size=TARGET_INTERMEDIATE_SIZE,
        ),
    )


@dataclass(frozen=True, slots=True)
class TraceRowIndex:
    """Only the circuit tensors needed by folding, aligned by causal path."""

    paths: tuple[str, ...]
    path_splits: tuple[str, ...]
    layer_kinds: tuple[str, ...]
    layer_fields: Mapping[int, Mapping[str, Tensor]]
    cache_key: str
    corpus_fingerprint: str
    trace_rows_sha256: str

    def field(self, layer_index: int, name: str) -> Tensor:
        try:
            value = self.layer_fields[layer_index][name]
        except KeyError as exc:
            raise Lfm2ResidualAttentionRunnerError(
                f"Trace field is missing: layer={layer_index}, name={name!r}."
            ) from exc
        return value

    def assert_aligned(self, other: "TraceRowIndex") -> None:
        if self.paths != other.paths:
            raise Lfm2ResidualAttentionRunnerError(
                "Parent and child circuit paths are not exactly aligned."
            )
        if self.path_splits != other.path_splits:
            raise Lfm2ResidualAttentionRunnerError(
                "Parent and child circuit split labels are not aligned."
            )
        if self.layer_kinds != other.layer_kinds:
            raise Lfm2ResidualAttentionRunnerError(
                "Parent and child layer-kind topology differs."
            )
        if self.corpus_fingerprint != other.corpus_fingerprint:
            raise Lfm2ResidualAttentionRunnerError(
                "Parent and child traces were not collected on identical inputs."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TRACE_ROWS_SCHEMA,
            "cache_key": self.cache_key,
            "corpus_fingerprint": self.corpus_fingerprint,
            "path_count": len(self.paths),
            "path_split_counts": {
                split: self.path_splits.count(split)
                for split in sorted(set(self.path_splits))
            },
            "layer_kinds": list(self.layer_kinds),
            "trace_rows_sha256": self.trace_rows_sha256,
        }


def _phase_sort_key(value: str) -> tuple[int, int]:
    if value == "prefill":
        return (0, 0)
    match = re.fullmatch(r"decode\.(\d{3})", value)
    if match is None:
        raise Lfm2ResidualAttentionRunnerError(f"Unsupported trace phase: {value!r}.")
    return (1, int(match.group(1)))


def _row_major_trace(value: Tensor, field: str) -> Tensor:
    tensor = value.detach().to(device="cpu").contiguous()
    if field.startswith("attention."):
        if tensor.ndim != 4 or int(tensor.shape[0]) != 1:
            raise Lfm2ResidualAttentionRunnerError(
                f"{field} must have shape [1, heads, sequence, head_dim]."
            )
        return tensor.permute(0, 2, 1, 3).reshape(
            int(tensor.shape[2]), int(tensor.shape[1]), int(tensor.shape[3])
        )
    if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
        raise Lfm2ResidualAttentionRunnerError(
            f"{field} must have shape [1, sequence, width]."
        )
    return tensor.reshape(int(tensor.shape[1]), int(tensor.shape[2]))


def extract_trace_rows(
    cache: Lfm2CircuitTraceCache,
    *,
    batch_splits: Sequence[str] | None = None,
) -> TraceRowIndex:
    """Materialize bounded circuit rows without retaining vocabulary logits."""

    manifest = cache.manifest
    if batch_splits is not None and len(batch_splits) != len(cache):
        raise Lfm2ResidualAttentionRunnerError(
            "Circuit-cache batch split count does not match the cache."
        )
    topology = manifest["layer_topology"]["layers"]
    layer_kinds = tuple(str(row["contract"]["kind"]) for row in topology)
    pieces: dict[int, dict[str, list[Tensor]]] = {
        index: {} for index in range(len(layer_kinds))
    }
    paths: list[str] = []
    path_splits: list[str] = []
    descriptors: list[dict[str, Any]] = []

    for batch in cache.iter_batches():
        split = (
            "unspecified" if batch_splits is None else str(batch_splits[batch.index])
        )
        tensors = batch.tensors
        phases = sorted(
            {
                match.group(1)
                for name in tensors
                if (match := _PHASE_PATTERN.match(name)) is not None
                and not name.startswith("cached_prefill.")
            },
            key=_phase_sort_key,
        )
        for phase in phases:
            anchor_name = f"{phase}.layer.000.residual.pre"
            if anchor_name not in tensors:
                continue
            anchor = _row_major_trace(tensors[anchor_name], "residual.pre")
            phase_paths = [
                f"batch.{batch.index:06d}/{phase}/row.{row:04d}"
                for row in range(int(anchor.shape[0]))
            ]
            paths.extend(phase_paths)
            path_splits.extend([split] * len(phase_paths))
            for layer_index in range(len(layer_kinds)):
                prefix = f"{phase}.layer.{layer_index:03d}."
                found = {
                    name[len(prefix) :]: value
                    for name, value in tensors.items()
                    if name.startswith(prefix) and name[len(prefix) :] in _TRACE_FIELDS
                }
                required = {
                    "mlp.gate",
                    "mlp.post_swiglu",
                    "mlp.up",
                    "residual.post_layer",
                    "residual.post_operator",
                    "residual.pre",
                }
                if layer_kinds[layer_index] == "full_attention":
                    required.update(
                        {
                            "attention.context",
                            "attention.k",
                            "attention.q",
                            "attention.v",
                        }
                    )
                missing = sorted(required.difference(found))
                if missing:
                    raise Lfm2ResidualAttentionRunnerError(
                        f"Trace phase {phase}, layer {layer_index} is missing "
                        f"fields: {missing}."
                    )
                for field in sorted(required):
                    rows = _row_major_trace(found[field], field)
                    if int(rows.shape[0]) != len(phase_paths):
                        raise Lfm2ResidualAttentionRunnerError(
                            f"Trace row count differs for {phase}, layer "
                            f"{layer_index}, field {field}."
                        )
                    pieces[layer_index].setdefault(field, []).append(rows)

    if not paths:
        raise Lfm2ResidualAttentionRunnerError(
            "The circuit cache contained no foldable phases."
        )
    materialized: dict[int, Mapping[str, Tensor]] = {}
    for layer_index, fields in pieces.items():
        joined: dict[str, Tensor] = {}
        for field, values in fields.items():
            tensor = torch.cat(values, dim=0).contiguous()
            if int(tensor.shape[0]) != len(paths):
                raise Lfm2ResidualAttentionRunnerError(
                    f"Layer {layer_index} field {field} is not path aligned."
                )
            joined[field] = tensor
            descriptors.append(
                {
                    "layer": layer_index,
                    "field": field,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "sha256": _tensor_sha256(tensor),
                }
            )
        materialized[layer_index] = MappingProxyType(joined)
    identity = {
        "schema": TRACE_ROWS_SCHEMA,
        "cache_key": cache.cache_key,
        "corpus_fingerprint": cache.corpus_fingerprint,
        "paths": paths,
        "path_splits": path_splits,
        "tensors": sorted(descriptors, key=lambda row: (row["layer"], row["field"])),
    }
    return TraceRowIndex(
        paths=tuple(paths),
        path_splits=tuple(path_splits),
        layer_kinds=layer_kinds,
        layer_fields=MappingProxyType(materialized),
        cache_key=cache.cache_key,
        corpus_fingerprint=cache.corpus_fingerprint,
        trace_rows_sha256=_json_sha256(identity),
    )


def _bounded_indices(row_count: int, maximum: int) -> tuple[int, ...]:
    if row_count <= 0 or maximum <= 0:
        raise Lfm2ResidualAttentionRunnerError(
            "Row counts and bounds must be positive."
        )
    if row_count <= maximum:
        return tuple(range(row_count))
    return tuple((index * row_count) // maximum for index in range(maximum))


def _index_rows(value: Tensor, indices: Sequence[int]) -> Tensor:
    index = torch.tensor(tuple(indices), dtype=torch.long)
    return value.index_select(0, index).contiguous()


def _attention_contributions(
    context_rows: Tensor,
    output_weight: Tensor,
    query_heads: int,
    head_dim: int,
) -> Tensor:
    hidden_size = int(output_weight.shape[0])
    blocks = (
        output_weight.detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(hidden_size, query_heads, head_dim)
    )
    context = context_rows.detach().to(device="cpu", dtype=torch.float64)
    return torch.einsum("rqd,hqd->rqh", context, blocks).contiguous()


@dataclass(frozen=True, slots=True)
class LayerProducerPlan:
    """Real-medoid producers plus consumer priors for one compact layer shell."""

    layer_index: int
    kind: str
    selection_indices: tuple[int, ...]
    fit_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    attention: GQAActivationClusterPlan | None
    mlp: GatedMLPActivationClusterPlan
    query_weight: Tensor | None
    key_weight: Tensor | None
    value_weight: Tensor | None
    output_prior: Tensor | None
    gate_weight: Tensor
    up_weight: Tensor
    down_prior: Tensor
    plan_sha256: str

    def local_state_overrides(self) -> Mapping[str, Tensor]:
        result = {
            "feed_forward.w1.weight": self.gate_weight,
            "feed_forward.w3.weight": self.up_weight,
            "feed_forward.w2.weight": self.down_prior,
        }
        if self.attention is not None:
            assert self.query_weight is not None
            assert self.key_weight is not None
            assert self.value_weight is not None
            assert self.output_prior is not None
            result.update(
                {
                    "self_attn.q_proj.weight": self.query_weight,
                    "self_attn.k_proj.weight": self.key_weight,
                    "self_attn.v_proj.weight": self.value_weight,
                    "self_attn.out_proj.weight": self.output_prior,
                }
            )
        return MappingProxyType(result)

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "kind": self.kind,
            "selection_indices": list(self.selection_indices),
            "fit_indices": list(self.fit_indices),
            "validation_indices": list(self.validation_indices),
            "validation_disjoint_from_selection": not bool(
                set(self.selection_indices).intersection(self.validation_indices)
            ),
            "validation_disjoint_from_fit": not bool(
                set(self.fit_indices).intersection(self.validation_indices)
            ),
            "selection_disjoint_from_fit": not bool(
                set(self.selection_indices).intersection(self.fit_indices)
            ),
            "attention": (
                None
                if self.attention is None
                else {
                    "group_medoids": list(self.attention.group_medoids),
                    "group_assignment": list(self.attention.group_assignment),
                    "query_head_medoids": list(self.attention.query_head_medoids),
                    "query_head_assignment": list(self.attention.query_head_assignment),
                    "full_source_coverage": (
                        self.attention.report.full_source_coverage
                    ),
                    "real_medoids": True,
                }
            ),
            "mlp": {
                "medoids": list(self.mlp.medoid_source_neurons),
                "assignment": list(self.mlp.source_to_target),
                "full_source_coverage": self.mlp.report.full_source_coverage,
                "real_medoids": True,
            },
            "producer_averaging": False,
            "blank_values_read": False,
            "zero_gradient": True,
            "plan_sha256": self.plan_sha256,
        }


def derive_layer_producer_plan(
    source_layer: nn.Module,
    target_layer: nn.Module,
    parent_rows: TraceRowIndex,
    layer_index: int,
    *,
    fold_config: Lfm2ClosedLoopFoldConfig,
    selection_indices: Sequence[int],
    fit_indices: Sequence[int],
    validation_indices: Sequence[int],
) -> LayerProducerPlan:
    """Derive the deterministic first-stage medoids from immutable parent rows."""

    source = inspect_lfm2_layer(source_layer)
    target = inspect_lfm2_layer(target_layer)
    if source.kind != target.kind or source.hidden_size != target.hidden_size:
        raise Lfm2ResidualAttentionRunnerError(
            "Source and compact target layer shells are incompatible."
        )
    selection = tuple(int(value) for value in selection_indices)
    fit = tuple(int(value) for value in fit_indices)
    validation = tuple(int(value) for value in validation_indices)
    if not selection or not fit or not validation:
        raise Lfm2ResidualAttentionRunnerError(
            "Producer planning requires nonempty selection, fit, and validation paths."
        )
    if set(validation).intersection(selection) or set(validation).intersection(fit):
        raise Lfm2ResidualAttentionRunnerError(
            "Validation paths must be invisible to producer selection and endpoint fit."
        )
    if set(selection).intersection(fit):
        raise Lfm2ResidualAttentionRunnerError(
            "Producer medoid selection paths must be disjoint from endpoint-fit paths."
        )
    gate = _index_rows(parent_rows.field(layer_index, "mlp.gate"), selection)
    up = _index_rows(parent_rows.field(layer_index, "mlp.up"), selection)
    post = _index_rows(parent_rows.field(layer_index, "mlp.post_swiglu"), selection)
    source_down = source_layer.feed_forward.w2.weight.detach().to(
        device="cpu", dtype=torch.float64
    )
    contribution = post.to(torch.float64) * torch.linalg.vector_norm(
        source_down, dim=0
    ).unsqueeze(0)
    mlp = build_gated_mlp_activation_cluster_plan(
        gate,
        up,
        post,
        contribution,
        target_neurons=target.ffn_intermediate_size,
        source_neurons=source.ffn_intermediate_size,
        ridge=fold_config.cluster_ridge,
        max_iterations=fold_config.cluster_max_iterations,
        distance_chunk_size=fold_config.cluster_distance_chunk_size,
    )
    gate_weight = mlp.select_gate_producer(source_layer.feed_forward.w1.weight, axis=0)
    up_weight = mlp.select_up_producer(source_layer.feed_forward.w3.weight, axis=0)
    down_prior = mlp.aggregate_down_consumer(
        source_layer.feed_forward.w2.weight, axis=1
    )

    attention: GQAActivationClusterPlan | None = None
    query_weight = key_weight = value_weight = output_prior = None
    if source.kind == "full_attention":
        if not isinstance(source.operator, Lfm2AttentionContract) or not isinstance(
            target.operator, Lfm2AttentionContract
        ):
            raise Lfm2ResidualAttentionRunnerError(
                "Canonical attention contracts are unavailable."
            )
        query = _index_rows(parent_rows.field(layer_index, "attention.q"), selection)
        key = _index_rows(parent_rows.field(layer_index, "attention.k"), selection)
        value = _index_rows(parent_rows.field(layer_index, "attention.v"), selection)
        context = _index_rows(
            parent_rows.field(layer_index, "attention.context"), selection
        )
        contributions = _attention_contributions(
            context,
            source_layer.self_attn.out_proj.weight,
            source.operator.query_heads,
            source.operator.head_dim,
        )
        attention = build_gqa_activation_cluster_plan(
            query.movedim(1, 0).contiguous(),
            key.movedim(1, 0).contiguous(),
            value.movedim(1, 0).contiguous(),
            context.movedim(1, 0).contiguous(),
            contributions.movedim(1, 0).contiguous(),
            source_query_heads=source.operator.query_heads,
            source_key_value_heads=source.operator.key_value_heads,
            target_query_heads=target.operator.query_heads,
            target_key_value_heads=target.operator.key_value_heads,
            source_head_dim=source.operator.head_dim,
            target_head_dim=target.operator.head_dim,
            ridge=fold_config.cluster_ridge,
            max_iterations=fold_config.cluster_max_iterations,
            distance_chunk_size=fold_config.cluster_distance_chunk_size,
        )
        query_weight = attention.select_query_producer(
            source_layer.self_attn.q_proj.weight, axis=0
        )
        key_weight = attention.select_key_producer(
            source_layer.self_attn.k_proj.weight, axis=0
        )
        value_weight = attention.select_value_producer(
            source_layer.self_attn.v_proj.weight, axis=0
        )
        output_prior = attention.aggregate_output_consumer(
            source_layer.self_attn.out_proj.weight, axis=1
        )
    elif not isinstance(source.operator, Lfm2ShortConvContract):
        raise Lfm2ResidualAttentionRunnerError(
            "Canonical short-conv contract is unavailable."
        )

    metadata = {
        "layer_index": layer_index,
        "kind": source.kind,
        "selection_indices": list(selection),
        "fit_indices": list(fit),
        "validation_indices": list(validation),
        "attention_medoids": (
            None
            if attention is None
            else {
                "groups": list(attention.group_medoids),
                "queries": list(attention.query_head_medoids),
            }
        ),
        "mlp_medoids": list(mlp.medoid_source_neurons),
        "weights": {
            "q": None if query_weight is None else _tensor_sha256(query_weight),
            "k": None if key_weight is None else _tensor_sha256(key_weight),
            "v": None if value_weight is None else _tensor_sha256(value_weight),
            "o_prior": (None if output_prior is None else _tensor_sha256(output_prior)),
            "gate": _tensor_sha256(gate_weight),
            "up": _tensor_sha256(up_weight),
            "down_prior": _tensor_sha256(down_prior),
        },
        "zero_gradient": True,
        "blank_values_read": False,
    }
    return LayerProducerPlan(
        layer_index=layer_index,
        kind=source.kind,
        selection_indices=selection,
        fit_indices=fit,
        validation_indices=validation,
        attention=attention,
        mlp=mlp,
        query_weight=(
            None
            if query_weight is None
            else query_weight.detach().to(device="cpu").contiguous().clone()
        ),
        key_weight=(
            None
            if key_weight is None
            else key_weight.detach().to(device="cpu").contiguous().clone()
        ),
        value_weight=(
            None
            if value_weight is None
            else value_weight.detach().to(device="cpu").contiguous().clone()
        ),
        output_prior=(
            None
            if output_prior is None
            else output_prior.detach().to(device="cpu").contiguous().clone()
        ),
        gate_weight=gate_weight.detach().to(device="cpu").contiguous().clone(),
        up_weight=up_weight.detach().to(device="cpu").contiguous().clone(),
        down_prior=down_prior.detach().to(device="cpu").contiguous().clone(),
        plan_sha256=_json_sha256(metadata),
    )


def _lfm2_layers(model: nn.Module) -> tuple[nn.Module, ...]:
    body = getattr(model, "model", None)
    layers = getattr(body, "layers", None)
    if layers is None:
        raise Lfm2ResidualAttentionRunnerError(
            "Expected model.model.layers on the loaded LFM2 causal model."
        )
    result = tuple(layers)
    if not result:
        raise Lfm2ResidualAttentionRunnerError("The LFM2 model has no layers.")
    return result


def derive_all_producer_plans(
    parent: nn.Module,
    child_shell: nn.Module,
    parent_rows: TraceRowIndex,
    *,
    fold_config: Lfm2ClosedLoopFoldConfig,
    max_selection_rows: int,
    max_fit_rows: int,
    max_validation_rows: int,
    validation_fraction: float,
) -> tuple[LayerProducerPlan, ...]:
    source_layers = _lfm2_layers(parent)
    target_layers = _lfm2_layers(child_shell)
    if len(source_layers) != len(target_layers):
        raise Lfm2ResidualAttentionRunnerError(
            "The compact target changed the layer count."
        )
    partition = deterministic_path_partition(
        parent_rows.paths,
        path_splits=parent_rows.path_splits,
        max_selection_rows=max_selection_rows,
        max_fit_rows=max_fit_rows,
        max_validation_rows=max_validation_rows,
        validation_fraction=validation_fraction,
    )
    return tuple(
        derive_layer_producer_plan(
            source_layer,
            target_layer,
            parent_rows,
            layer_index,
            fold_config=fold_config,
            selection_indices=partition.selection_indices,
            fit_indices=partition.fit_indices,
            validation_indices=partition.validation_indices,
        )
        for layer_index, (source_layer, target_layer) in enumerate(
            zip(source_layers, target_layers)
        )
    )


def producer_bundle_identity(
    plans: Sequence[LayerProducerPlan],
) -> dict[str, Any]:
    payload = [plan.as_dict() for plan in plans]
    return {
        "layer_count": len(payload),
        "plans": payload,
        "producer_bundle_sha256": _json_sha256(payload),
        "zero_gradient": True,
        "blank_values_read": False,
    }


def _layer_prefixes(model: nn.Module) -> Mapping[int, str]:
    layers = _lfm2_layers(model)
    by_id = {id(layer): index for index, layer in enumerate(layers)}
    prefixes: dict[int, str] = {}
    for name, module in model.named_modules():
        index = by_id.get(id(module))
        if index is not None:
            prefixes[index] = name
    if set(prefixes) != set(range(len(layers))):
        raise Lfm2ResidualAttentionRunnerError(
            "Could not bind every LFM2 layer to its state-dict prefix."
        )
    return MappingProxyType(prefixes)


def install_complete_provisional_state(
    parent: nn.Module,
    child: nn.Module,
    plans: Sequence[LayerProducerPlan],
) -> dict[str, Any]:
    """Overwrite every compact state tensor from parent data and medoid plans."""

    parent_state = parent.state_dict()
    child_state = child.state_dict()
    if set(parent_state) != set(child_state):
        raise Lfm2ResidualAttentionRunnerError(
            "Parent and fixed target state keys differ."
        )
    prefixes = _layer_prefixes(child)
    overrides: dict[str, Tensor] = {}
    for plan in plans:
        prefix = prefixes[plan.layer_index]
        for local_name, value in plan.local_state_overrides().items():
            overrides[f"{prefix}.{local_name}"] = value

    copied: list[str] = []
    installed: list[str] = []
    with torch.no_grad():
        for name, target in child_state.items():
            source = parent_state[name]
            if name in overrides:
                candidate = overrides[name]
                if tuple(candidate.shape) != tuple(target.shape):
                    raise Lfm2ResidualAttentionRunnerError(
                        f"Provisional folded shape mismatch for {name}: "
                        f"{tuple(candidate.shape)} != {tuple(target.shape)}."
                    )
                target.copy_(candidate.to(device=target.device, dtype=target.dtype))
                installed.append(name)
                continue
            if tuple(source.shape) != tuple(target.shape):
                raise Lfm2ResidualAttentionRunnerError(
                    f"Compact state {name} was neither same-shape nor folded."
                )
            if source.dtype != target.dtype:
                raise Lfm2ResidualAttentionRunnerError(
                    f"Parent/child dtype differs for same-shape state {name}."
                )
            target.copy_(source.to(device=target.device))
            copied.append(name)

    covered = set(copied).union(installed)
    if covered != set(child_state):
        raise Lfm2ResidualAttentionRunnerError(
            "The provisional compiler did not overwrite every target state tensor."
        )
    if any(parameter.grad is not None for parameter in child.parameters()):
        raise Lfm2ResidualAttentionRunnerError(
            "Provisional installation wrote parameter gradients."
        )
    return {
        "same_shape_parent_tensors": len(copied),
        "folded_tensors": len(installed),
        "complete_target_state_coverage": True,
        "blank_values_read": False,
        "producer_averaging": False,
        "state_sha256": model_state_sha256(child),
        "zero_gradient": True,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }


def audit_global_transplant(parent: nn.Module, child: nn.Module) -> dict[str, Any]:
    """Prove embeddings, final norm, and tied LM head came from the parent."""

    parent_input = parent.get_input_embeddings()
    child_input = child.get_input_embeddings()
    parent_output = parent.get_output_embeddings()
    child_output = child.get_output_embeddings()
    if any(
        value is None
        for value in (parent_input, child_input, parent_output, child_output)
    ):
        raise Lfm2ResidualAttentionRunnerError(
            "The causal models must expose input and output embeddings."
        )
    input_exact = torch.equal(parent_input.weight.detach(), child_input.weight.detach())
    output_exact = torch.equal(
        parent_output.weight.detach(), child_output.weight.detach()
    )
    parent_tied = parent_input.weight is parent_output.weight
    child_tied = child_input.weight is child_output.weight
    if not input_exact or not output_exact or parent_tied != child_tied:
        raise Lfm2ResidualAttentionRunnerError(
            "Embedding/LM-head transplant or tie topology is not exact."
        )
    parent_state = parent.state_dict()
    child_state = child.state_dict()
    final_norm_keys = [
        name
        for name in child_state
        if name.endswith("norm.weight") and ".layers." not in name
    ]
    if not final_norm_keys:
        raise Lfm2ResidualAttentionRunnerError(
            "Could not identify the final model norm state."
        )
    final_norm_exact = all(
        name in parent_state
        and torch.equal(
            parent_state[name].detach(),
            child_state[name].detach(),
        )
        for name in final_norm_keys
    )
    if not final_norm_exact:
        raise Lfm2ResidualAttentionRunnerError(
            "The final model norm was not transplanted exactly."
        )
    return {
        "input_embeddings_exact": input_exact,
        "lm_head_exact": output_exact,
        "parent_tied_embeddings": parent_tied,
        "child_tied_embeddings": child_tied,
        "tie_topology_preserved": parent_tied == child_tied,
        "final_norm_keys": final_norm_keys,
        "final_norm_exact": final_norm_exact,
    }


def _model_device(model: nn.Module) -> torch.device:
    devices = {parameter.device for parameter in model.parameters()}
    if len(devices) != 1:
        raise Lfm2ResidualAttentionRunnerError(
            "The bounded proof runner requires one model device."
        )
    return next(iter(devices))


def _model_dtype(model: nn.Module) -> torch.dtype:
    dtypes = {parameter.dtype for parameter in model.parameters()}
    if len(dtypes) != 1:
        raise Lfm2ResidualAttentionRunnerError(
            "The bounded proof runner requires one parameter dtype."
        )
    return next(iter(dtypes))


def instantiate_blank_child(
    parent: nn.Module,
    target_config: Any,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Create a genuine random target shell whose values will all be erased."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise Lfm2ResidualAttentionRunnerError(
            "Blank child seed must be a nonnegative integer."
        )
    cuda_devices: list[int] = []
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        cuda_devices = [index]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        child = type(parent)(copy.deepcopy(target_config))
    child.to(device=device, dtype=dtype)
    child.eval()
    return child


def audit_blank_seed_erasure(
    parent: nn.Module,
    target_config: Any,
    plans: Sequence[LayerProducerPlan],
    *,
    seeds: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    primary_child: nn.Module | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Prove two or more random shells compile to one byte-identical child."""

    normalized = tuple(int(seed) for seed in seeds)
    if len(normalized) < 2 or len(set(normalized)) != len(normalized):
        raise Lfm2ResidualAttentionRunnerError(
            "Seed-erasure audit requires at least two distinct seeds."
        )
    rows = []
    final_hashes: list[str] = []
    retained: nn.Module | None = primary_child
    for index, seed in enumerate(normalized):
        child = (
            retained
            if index == 0 and retained is not None
            else instantiate_blank_child(
                parent,
                target_config,
                seed=seed,
                device=device,
                dtype=dtype,
            )
        )
        blank_hash = model_state_sha256(child)
        provisional = install_complete_provisional_state(parent, child, plans)
        compiled_hash = provisional["state_sha256"]
        final_hashes.append(compiled_hash)
        rows.append(
            {
                "seed": seed,
                "blank_state_sha256": blank_hash,
                "compiled_state_sha256": compiled_hash,
                "provisional": provisional,
            }
        )
        if index == 0:
            retained = child
        else:
            del child
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if len(set(row["blank_state_sha256"] for row in rows)) < 2:
        raise Lfm2ResidualAttentionRunnerError(
            "Distinct blank seeds did not produce distinct blank states."
        )
    if len(set(final_hashes)) != 1:
        raise Lfm2ResidualAttentionRunnerError(
            "The provisional fold retained random blank-seed information."
        )
    assert retained is not None
    return retained, {
        "seed_count": len(rows),
        "seeds": rows,
        "blank_states_distinct": True,
        "compiled_states_identical": True,
        "seed_erasure_proven": True,
        "compiled_state_sha256": final_hashes[0],
        "blank_values_read": False,
    }


def calibration_batches_from_cache(
    cache: Lfm2CircuitTraceCache,
) -> tuple[dict[str, Tensor], ...]:
    """Recreate exact detached inputs for every compact-child recollection."""

    result = []
    for batch in cache.iter_batches():
        value = {
            "input_ids": batch.input_ids.clone(),
            "attention_mask": batch.attention_mask.clone(),
            "forced_decode_ids": batch.forced_decode_ids.clone(),
        }
        position = batch.tensors.get("input.position_ids")
        if position is not None:
            value["position_ids"] = position.clone()
        result.append(value)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class EvidenceSplit:
    fit_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "fit_indices": list(self.fit_indices),
            "validation_indices": list(self.validation_indices),
            "disjoint": not bool(
                set(self.fit_indices).intersection(self.validation_indices)
            ),
        }


@dataclass(frozen=True, slots=True)
class EvidencePartition:
    """Batch-separated discovery/fit/validation paths shared by every layer."""

    selection_indices: tuple[int, ...]
    fit_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    validation_batches: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "selection_indices": list(self.selection_indices),
            "fit_indices": list(self.fit_indices),
            "validation_indices": list(self.validation_indices),
            "validation_batches": list(self.validation_batches),
            "validation_disjoint_from_selection": not bool(
                set(self.validation_indices).intersection(self.selection_indices)
            ),
            "validation_disjoint_from_fit": not bool(
                set(self.validation_indices).intersection(self.fit_indices)
            ),
            "selection_disjoint_from_fit": not bool(
                set(self.selection_indices).intersection(self.fit_indices)
            ),
        }


def deterministic_path_partition(
    paths: Sequence[str],
    *,
    path_splits: Sequence[str] | None = None,
    max_selection_rows: int,
    max_fit_rows: int,
    max_validation_rows: int,
    validation_fraction: float,
) -> EvidencePartition:
    """Reserve whole trailing batches for validation whenever possible."""

    if len(paths) < 2:
        raise Lfm2ResidualAttentionRunnerError(
            "At least two circuit rows are required for held-out validation."
        )
    groups: dict[str, list[int]] = {}
    for index, path in enumerate(paths):
        batch = path.split("/", 1)[0]
        groups.setdefault(batch, []).append(index)
    ordered_batches = sorted(groups)
    target_validation = max(1, round(len(paths) * validation_fraction))
    target_validation = min(target_validation, max_validation_rows, len(paths) - 1)
    validation_pool: list[int] = []
    validation_batches: list[str] = []
    labeled = (
        path_splits is not None
        and len(path_splits) == len(paths)
        and set(path_splits) != {"unspecified"}
    )
    if labeled:
        normalized = tuple(str(value).strip().lower() for value in path_splits)
        unexpected = sorted(
            set(normalized).difference({"selection", "fit", "validation"})
        )
        if unexpected:
            raise Lfm2ResidualAttentionRunnerError(
                f"Unsupported calibration split labels: {unexpected}."
            )
        selection_pool = [
            index for index, split in enumerate(normalized) if split == "selection"
        ]
        fit_pool = [index for index, split in enumerate(normalized) if split == "fit"]
        validation_pool = [
            index for index, split in enumerate(normalized) if split == "validation"
        ]
        if not selection_pool or not fit_pool:
            raise Lfm2ResidualAttentionRunnerError(
                "Calibration requires nonempty selection and fit batches."
            )
        if validation_pool:
            validation_batches = sorted(
                {paths[index].split("/", 1)[0] for index in validation_pool}
            )
            discovery_pool = fit_pool
        else:
            fit_batches: dict[str, list[int]] = {}
            for index in fit_pool:
                fit_batches.setdefault(paths[index].split("/", 1)[0], []).append(index)
            ordered_fit_batches = sorted(fit_batches)
            if len(ordered_fit_batches) >= 2:
                validation_batch = ordered_fit_batches[-1]
                validation_pool = fit_batches[validation_batch]
                validation_batches = [validation_batch]
                discovery_pool = [
                    index
                    for batch in ordered_fit_batches[:-1]
                    for index in fit_batches[batch]
                ]
            else:
                validation_count = min(target_validation, max(1, len(fit_pool) - 1))
                validation_pool = fit_pool[-validation_count:]
                discovery_pool = fit_pool[:-validation_count]
                validation_batches = [ordered_fit_batches[0] + ":held-out-tail"]
    elif len(ordered_batches) >= 2:
        for batch in reversed(ordered_batches):
            if len(validation_batches) >= len(ordered_batches) - 1:
                break
            validation_batches.append(batch)
            validation_pool.extend(groups[batch])
            if len(validation_pool) >= target_validation:
                break
        discovery_pool = [
            index
            for batch in ordered_batches
            if batch not in set(validation_batches)
            for index in groups[batch]
        ]
    else:
        validation_pool = list(range(len(paths) - target_validation, len(paths)))
        discovery_pool = list(range(0, len(paths) - target_validation))
        validation_batches = [ordered_batches[0] + ":held-out-tail"]
    if not discovery_pool or not validation_pool:
        raise Lfm2ResidualAttentionRunnerError(
            "Could not form nonempty discovery and validation path sets."
        )

    def bounded_pool(pool: Sequence[int], maximum: int) -> tuple[int, ...]:
        selected = _bounded_indices(len(pool), min(len(pool), maximum))
        return tuple(pool[index] for index in selected)

    selection_source = selection_pool if labeled else discovery_pool
    selection = bounded_pool(selection_source, max_selection_rows)
    fit = bounded_pool(discovery_pool, max_fit_rows)
    validation = bounded_pool(validation_pool, max_validation_rows)
    result = EvidencePartition(
        selection_indices=selection,
        fit_indices=fit,
        validation_indices=validation,
        validation_batches=tuple(sorted(validation_batches)),
    )
    report = result.as_dict()
    if (
        not report["validation_disjoint_from_selection"]
        or not report["validation_disjoint_from_fit"]
    ):
        raise Lfm2ResidualAttentionRunnerError(
            "Held-out validation partition overlaps discovery paths."
        )
    if labeled and not report["selection_disjoint_from_fit"]:
        raise Lfm2ResidualAttentionRunnerError(
            "Explicit selection and fit calibration batches overlap."
        )
    return result


def deterministic_evidence_split(
    row_count: int,
    *,
    max_fit_rows: int,
    max_validation_rows: int,
    validation_fraction: float,
) -> EvidenceSplit:
    if row_count <= 0:
        raise Lfm2ResidualAttentionRunnerError(
            "Evidence requires at least one aligned path row."
        )
    bounded = _bounded_indices(
        row_count, min(row_count, max_fit_rows + max_validation_rows)
    )
    if len(bounded) == 1 or validation_fraction <= 0.0:
        return EvidenceSplit(fit_indices=bounded, validation_indices=())
    validation_count = max(1, round(len(bounded) * validation_fraction))
    validation_count = min(validation_count, max_validation_rows, len(bounded) - 1)
    fit_count = min(len(bounded) - validation_count, max_fit_rows)
    fit = tuple(bounded[:fit_count])
    validation = tuple(bounded[-validation_count:])
    if set(fit).intersection(validation):
        raise Lfm2ResidualAttentionRunnerError(
            "Fit and validation evidence paths overlap."
        )
    return EvidenceSplit(fit_indices=fit, validation_indices=validation)


def _optional_index(value: Tensor, indices: Sequence[int]) -> Tensor | None:
    return None if not indices else _index_rows(value, indices)


def build_cached_layer_evidence(
    parent_rows: TraceRowIndex,
    child_rows: TraceRowIndex,
    *,
    layer_index: int,
    selection_indices: Sequence[int],
    fit_indices: Sequence[int],
    validation_indices: Sequence[int],
) -> tuple[Lfm2ClosedLoopFoldEvidence, EvidenceSplit]:
    """Assemble one layer's evidence using only immutable cached parent rows."""

    parent_rows.assert_aligned(child_rows)
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or not 0 <= layer_index < len(parent_rows.layer_kinds)
    ):
        raise Lfm2ResidualAttentionRunnerError(
            "Evidence layer_index is outside the cached LFM2 topology."
        )
    selection = tuple(int(value) for value in selection_indices)
    fit = tuple(int(value) for value in fit_indices)
    validation = tuple(int(value) for value in validation_indices)
    if not selection or not fit:
        raise Lfm2ResidualAttentionRunnerError(
            "Cached layer evidence requires selection and fit rows."
        )
    if set(selection).intersection(validation) or set(fit).intersection(validation):
        raise Lfm2ResidualAttentionRunnerError(
            "Cached validation evidence overlaps discovery rows."
        )
    split = EvidenceSplit(
        fit_indices=fit,
        validation_indices=validation,
    )

    mlp = Lfm2MLPFoldEvidence(
        parent_gate_rows=_index_rows(
            parent_rows.field(layer_index, "mlp.gate"), selection
        ),
        parent_up_rows=_index_rows(parent_rows.field(layer_index, "mlp.up"), selection),
        parent_post_swiglu_rows=_index_rows(
            parent_rows.field(layer_index, "mlp.post_swiglu"), selection
        ),
        parent_contribution_rows=None,
        child_post_swiglu_rows=_index_rows(
            child_rows.field(layer_index, "mlp.post_swiglu"), fit
        ),
        parent_branch_output_rows=_index_rows(
            parent_rows.field(layer_index, "residual.post_layer"), fit
        ),
        child_pre_branch_residual_rows=_index_rows(
            child_rows.field(layer_index, "residual.post_operator"), fit
        ),
        validation_child_post_swiglu_rows=_optional_index(
            child_rows.field(layer_index, "mlp.post_swiglu"), validation
        ),
        validation_parent_branch_output_rows=_optional_index(
            parent_rows.field(layer_index, "residual.post_layer"), validation
        ),
        validation_child_pre_branch_residual_rows=_optional_index(
            child_rows.field(layer_index, "residual.post_operator"), validation
        ),
    )
    attention = None
    if parent_rows.layer_kinds[layer_index] == "full_attention":
        attention = Lfm2AttentionFoldEvidence(
            parent_query_rows=_index_rows(
                parent_rows.field(layer_index, "attention.q"), selection
            ),
            parent_key_rows=_index_rows(
                parent_rows.field(layer_index, "attention.k"), selection
            ),
            parent_value_rows=_index_rows(
                parent_rows.field(layer_index, "attention.v"), selection
            ),
            parent_context_rows=_index_rows(
                parent_rows.field(layer_index, "attention.context"), selection
            ),
            parent_contribution_rows=None,
            child_context_rows=_index_rows(
                child_rows.field(layer_index, "attention.context"), fit
            ),
            parent_branch_output_rows=_index_rows(
                parent_rows.field(layer_index, "residual.post_operator"), fit
            ),
            child_pre_branch_residual_rows=_index_rows(
                child_rows.field(layer_index, "residual.pre"), fit
            ),
            validation_child_context_rows=_optional_index(
                child_rows.field(layer_index, "attention.context"), validation
            ),
            validation_parent_branch_output_rows=_optional_index(
                parent_rows.field(layer_index, "residual.post_operator"),
                validation,
            ),
            validation_child_pre_branch_residual_rows=_optional_index(
                child_rows.field(layer_index, "residual.pre"), validation
            ),
        )
    return Lfm2ClosedLoopFoldEvidence(mlp=mlp, attention=attention), split


def build_layer_evidence(
    source_layer: nn.Module,
    parent_rows: TraceRowIndex,
    child_rows: TraceRowIndex,
    producer_plan: LayerProducerPlan,
    *,
    max_fit_rows: int,
    max_validation_rows: int,
    validation_fraction: float,
) -> tuple[Lfm2ClosedLoopFoldEvidence, EvidenceSplit]:
    """Build endpoint evidence from true compact-child causal states."""

    layer_index = producer_plan.layer_index
    del max_fit_rows, max_validation_rows, validation_fraction
    source_contract = inspect_lfm2_layer(source_layer)
    if source_contract.kind != parent_rows.layer_kinds[layer_index]:
        raise Lfm2ResidualAttentionRunnerError(
            "Live source layer kind differs from the cached parent topology."
        )
    return build_cached_layer_evidence(
        parent_rows,
        child_rows,
        layer_index=layer_index,
        selection_indices=producer_plan.selection_indices,
        fit_indices=producer_plan.fit_indices,
        validation_indices=producer_plan.validation_indices,
    )


def assert_closed_loop_preserves_producer_plan(
    closed_loop: Lfm2ClosedLoopFoldPlan,
    producer: LayerProducerPlan,
) -> None:
    if tuple(closed_loop.mlp.cluster_plan.medoid_source_neurons) != tuple(
        producer.mlp.medoid_source_neurons
    ) or tuple(closed_loop.mlp.cluster_plan.source_to_target) != tuple(
        producer.mlp.source_to_target
    ):
        raise Lfm2ResidualAttentionRunnerError(
            "Closed-loop MLP planning drifted from the cached parent medoids."
        )
    if producer.attention is None:
        if closed_loop.attention is not None:
            raise Lfm2ResidualAttentionRunnerError(
                "A short-conv layer unexpectedly acquired attention."
            )
        return
    if closed_loop.attention is None:
        raise Lfm2ResidualAttentionRunnerError(
            "The closed-loop plan omitted an attention merge."
        )
    observed = closed_loop.attention.cluster_plan
    if (
        tuple(observed.group_medoids) != tuple(producer.attention.group_medoids)
        or tuple(observed.group_assignment)
        != tuple(producer.attention.group_assignment)
        or tuple(observed.query_head_medoids)
        != tuple(producer.attention.query_head_medoids)
        or tuple(observed.query_head_assignment)
        != tuple(producer.attention.query_head_assignment)
    ):
        raise Lfm2ResidualAttentionRunnerError(
            "Closed-loop attention planning drifted from cached parent medoids."
        )


def _copy_plan_state(layer: nn.Module, plan: Lfm2ClosedLoopFoldPlan) -> None:
    current = layer.state_dict()
    candidate = plan.candidate_state_dict()
    if set(current) != set(candidate):
        raise Lfm2ResidualAttentionRunnerError(
            "Closed-loop candidate does not cover the complete layer state."
        )
    with torch.no_grad():
        for name, target in current.items():
            target.copy_(candidate[name].to(device=target.device, dtype=target.dtype))
    if any(parameter.grad is not None for parameter in layer.parameters()):
        raise Lfm2ResidualAttentionRunnerError(
            "A closed-loop provisional refinement wrote gradients."
        )


def _logit_metrics(parent: Tensor, child: Tensor) -> dict[str, float]:
    left = (
        parent.detach()
        .to(device="cpu", dtype=torch.float32)
        .reshape(-1, int(parent.shape[-1]))
    )
    right = (
        child.detach()
        .to(device="cpu", dtype=torch.float32)
        .reshape(-1, int(child.shape[-1]))
    )
    if left.shape != right.shape:
        raise Lfm2ResidualAttentionRunnerError(
            "Parent and child logit trace shapes differ."
        )
    top1 = float(
        (left.argmax(dim=-1) == right.argmax(dim=-1)).to(torch.float64).mean().item()
    )
    cosine = float(
        F.cosine_similarity(left, right, dim=-1, eps=1e-12)
        .to(torch.float64)
        .mean()
        .item()
    )
    parent_probability = F.softmax(left, dim=-1)
    child_log_probability = F.log_softmax(right, dim=-1)
    kl = float(
        F.kl_div(
            child_log_probability,
            parent_probability,
            reduction="batchmean",
        ).item()
    )
    return {"top1_agreement": top1, "mean_cosine": cosine, "kl": kl}


def _residual_metrics(parent: Tensor, child: Tensor) -> dict[str, float]:
    left = (
        parent.detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(int(parent.shape[0]), -1)
    )
    right = (
        child.detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(int(child.shape[0]), -1)
    )
    if int(left.shape[0]) != int(right.shape[0]):
        raise Lfm2ResidualAttentionRunnerError(
            "Parent and child residual trace row counts differ."
        )
    if int(left.shape[1]) != int(right.shape[1]):
        centered_left = left - left.mean(dim=0, keepdim=True)
        centered_right = right - right.mean(dim=0, keepdim=True)

        def linear_cka(
            left_rows: Tensor,
            right_rows: Tensor,
        ) -> tuple[float, float]:
            cross_energy = torch.sum(
                (left_rows.T @ right_rows).square()
            )
            left_energy = torch.sum((left_rows.T @ left_rows).square())
            right_energy = torch.sum((right_rows.T @ right_rows).square())
            denominator = torch.sqrt(left_energy * right_energy)
            return float(cross_energy.item()), float(denominator.item())

        numerator, denominator = linear_cka(
            centered_left,
            centered_right,
        )
        comparison_mode = "cross_width_centered_linear_cka"
        if denominator <= 1e-24:
            numerator, denominator = linear_cka(left, right)
            comparison_mode = "cross_width_uncentered_linear_cka"
        if denominator <= 1e-24:
            left_zero = bool(torch.count_nonzero(left).item() == 0)
            right_zero = bool(torch.count_nonzero(right).item() == 0)
            cka = 1.0 if left_zero and right_zero else 0.0
        else:
            cka = min(1.0, max(0.0, numerator / denominator))
        normalized_gram_distance = math.sqrt(max(0.0, 2.0 - 2.0 * cka))
        return {
            # Preserve the legacy trajectory-gate field names while making
            # their cross-width representation meaning explicit.
            "rmse": normalized_gram_distance,
            "target_rms": 1.0,
            "relative_rmse": normalized_gram_distance,
            "mean_cosine": cka,
            "comparison_mode": comparison_mode,
            "parent_width": int(left.shape[1]),
            "child_width": int(right.shape[1]),
            "centered_linear_cka": cka,
            "normalized_gram_distance": normalized_gram_distance,
        }
    error = right - left
    rmse = torch.sqrt(torch.mean(error.square()))
    target_rms = torch.sqrt(torch.mean(left.square()))
    relative = float((rmse / target_rms.clamp_min(1e-12)).item())
    cosine = float(F.cosine_similarity(left, right, dim=-1, eps=1e-12).mean().item())
    return {
        "rmse": float(rmse.item()),
        "target_rms": float(target_rms.item()),
        "relative_rmse": relative,
        "mean_cosine": cosine,
        "comparison_mode": "same_width_row_cosine_rmse",
        "parent_width": int(left.shape[1]),
        "child_width": int(right.shape[1]),
    }


def _output_value(output: Any, name: str) -> Any:
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


def _candidate_free_rollout(
    model: nn.Module,
    batch: Any,
    *,
    steps: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    device = _model_device(model)
    input_ids = batch.input_ids.to(device=device, dtype=torch.long)
    attention_mask = batch.attention_mask.to(device=device, dtype=torch.long)
    position = batch.tensors.get("input.position_ids")
    position_ids = (
        None if position is None else position.to(device=device, dtype=torch.long)
    )
    prefix_ids = input_ids
    prefix_mask = attention_mask
    prefix_positions = position_ids
    generated: list[int] = []
    consistency: list[dict[str, Any]] = []
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            cached = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
            past = _output_value(cached, "past_key_values")
            if past is None:
                raise Lfm2ResidualAttentionRunnerError(
                    "Candidate free rollout did not return a cache."
                )
            for step in range(steps):
                cached_logits = _output_value(cached, "logits")[:, -1:, :]
                full = model(
                    input_ids=prefix_ids,
                    attention_mask=prefix_mask,
                    position_ids=prefix_positions,
                    use_cache=False,
                    return_dict=True,
                )
                full_logits = _output_value(full, "logits")[:, -1:, :]
                close = bool(
                    torch.allclose(
                        cached_logits,
                        full_logits,
                        rtol=rtol,
                        atol=atol,
                    )
                )
                top1 = bool(
                    torch.equal(
                        cached_logits.argmax(dim=-1),
                        full_logits.argmax(dim=-1),
                    )
                )
                consistency.append(
                    {
                        "step": step,
                        "numerically_close": close,
                        "top1_agreement": top1,
                    }
                )
                token = cached_logits.argmax(dim=-1).to(torch.long)
                generated.append(int(token.item()))
                prefix_ids = torch.cat((prefix_ids, token), dim=1)
                prefix_mask = torch.cat(
                    (
                        prefix_mask,
                        torch.ones(
                            (1, 1),
                            device=device,
                            dtype=prefix_mask.dtype,
                        ),
                    ),
                    dim=1,
                )
                step_positions = None
                if prefix_positions is not None:
                    step_positions = prefix_positions[:, -1:] + 1
                    prefix_positions = torch.cat(
                        (prefix_positions, step_positions), dim=1
                    )
                if step + 1 < steps:
                    cached = model(
                        input_ids=token,
                        attention_mask=prefix_mask,
                        position_ids=step_positions,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    past = _output_value(cached, "past_key_values")
                    if past is None:
                        raise Lfm2ResidualAttentionRunnerError(
                            "Candidate free rollout lost its recurrent cache."
                        )
    finally:
        model.train(was_training)
    return {
        "tokens": generated,
        "cached_full_checks": consistency,
        "all_cached_full_numerically_close": all(
            row["numerically_close"] for row in consistency
        ),
        "all_cached_full_top1_agree": all(row["top1_agreement"] for row in consistency),
    }


@dataclass(frozen=True, slots=True)
class TrajectoryGateConfig:
    """Acceptance thresholds for a compact layer transaction."""

    min_teacher_forced_top1: float = 0.05
    min_free_run_token_match: float = 0.05
    min_mean_residual_cosine: float = 0.0
    max_score_regression: float = 0.0
    max_layer_relative_rmse_regression: float = 0.0
    require_layer_nonregression: bool = True
    logit_rtol: float = 5e-2
    logit_atol: float = 5e-1

    def __post_init__(self) -> None:
        for name in (
            "min_teacher_forced_top1",
            "min_free_run_token_match",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
            object.__setattr__(self, name, value)
        cosine = float(self.min_mean_residual_cosine)
        if not -1.0 <= cosine <= 1.0:
            raise ValueError("min_mean_residual_cosine must be in [-1, 1].")
        object.__setattr__(self, "min_mean_residual_cosine", cosine)
        for name in (
            "max_score_regression",
            "max_layer_relative_rmse_regression",
            "logit_rtol",
            "logit_atol",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_teacher_forced_top1": self.min_teacher_forced_top1,
            "min_free_run_token_match": self.min_free_run_token_match,
            "min_mean_residual_cosine": self.min_mean_residual_cosine,
            "max_score_regression": self.max_score_regression,
            "max_layer_relative_rmse_regression": (
                self.max_layer_relative_rmse_regression
            ),
            "require_layer_nonregression": self.require_layer_nonregression,
            "logit_rtol": self.logit_rtol,
            "logit_atol": self.logit_atol,
        }


def evaluate_trajectory_gate(
    parent_cache: Lfm2CircuitTraceCache,
    child_cache: Lfm2CircuitTraceCache,
    child_model: nn.Module,
    parent_rows: TraceRowIndex,
    child_rows: TraceRowIndex,
    *,
    config: TrajectoryGateConfig,
    layer_index: int | None = None,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Judge forced decode, residual depth, and true child-generated prefixes."""

    parent_rows.assert_aligned(child_rows)
    if len(parent_cache) != len(child_cache):
        raise Lfm2ResidualAttentionRunnerError(
            "Parent and child trace-cache batch counts differ."
        )
    logit_pairs: list[tuple[Tensor, Tensor]] = []
    free_matches = 0
    free_total = 0
    free_lcp_total = 0
    parent_reference_valid = True
    cached_full_numerically_close = True
    cached_full_top1_checks: list[bool] = []
    free_reports = []
    child_manifest = getattr(child_cache, "manifest", {})
    child_capture = (
        child_manifest.get("capture", {})
        if isinstance(child_manifest, Mapping)
        else {}
    )
    child_full_prefix_canonical = bool(
        isinstance(child_capture, Mapping)
        and child_capture.get("decode_trace_mode")
        == FULL_PREFIX_CANONICAL_TRACE_MODE
    )
    if child_full_prefix_canonical and (
        child_capture.get("cached_logits_accepted") is not False
        or child_capture.get("forced_decode_use_cache") is not False
    ):
        raise Lfm2ResidualAttentionRunnerError(
            "Canonical child traces do not prove cached logits were excluded."
        )

    for batch_index in range(len(parent_cache)):
        parent_batch = parent_cache.load_batch(batch_index)
        child_batch = child_cache.load_batch(batch_index)
        if not torch.equal(
            parent_batch.input_ids, child_batch.input_ids
        ) or not torch.equal(
            parent_batch.forced_decode_ids, child_batch.forced_decode_ids
        ):
            raise Lfm2ResidualAttentionRunnerError(
                "Parent and child cache inputs differ."
            )
        parent_tensors = parent_batch.tensors
        child_tensors = child_batch.tensors
        phase_pairs = [
            (
                "cached_prefill.logits",
                (
                    "prefill.logits"
                    if child_full_prefix_canonical
                    else "cached_prefill.logits"
                ),
            )
        ]
        phase_pairs.extend(
            (f"decode.{step:03d}.logits", f"decode.{step:03d}.logits")
            for step in range(int(parent_batch.forced_decode_ids.shape[1]))
        )
        for parent_key, child_key in phase_pairs:
            if (
                parent_key not in parent_tensors
                or child_key not in child_tensors
            ):
                raise Lfm2ResidualAttentionRunnerError(
                    "Missing canonical trajectory logits for "
                    f"parent={parent_key}, child={child_key}."
                )
            logit_pairs.append(
                (parent_tensors[parent_key], child_tensors[child_key])
            )
            if (
                child_key.startswith("decode.")
                and not child_full_prefix_canonical
            ):
                full_key = child_key.replace(".logits", ".full_prefix_logits")
                if full_key not in child_tensors:
                    raise Lfm2ResidualAttentionRunnerError(
                        f"Missing child full-prefix logits for {child_key}."
                    )
                cached_full_numerically_close = cached_full_numerically_close and bool(
                    torch.allclose(
                        child_tensors[child_key],
                        child_tensors[full_key],
                        rtol=config.logit_rtol,
                        atol=config.logit_atol,
                    )
                )
                cached_full_top1_checks.append(
                    bool(
                        torch.equal(
                            child_tensors[child_key].argmax(dim=-1),
                            child_tensors[full_key].argmax(dim=-1),
                        )
                    )
                )

        reference = [
            int(value) for value in parent_batch.forced_decode_ids.reshape(-1).tolist()
        ]
        for step, expected in enumerate(reference):
            predictor = (
                parent_tensors["cached_prefill.logits"][:, -1:, :]
                if step == 0
                else parent_tensors[f"decode.{step - 1:03d}.logits"]
            )
            if int(predictor.argmax(dim=-1).item()) != expected:
                parent_reference_valid = False
        rollout = _candidate_free_rollout(
            child_model,
            child_batch,
            steps=len(reference),
            rtol=config.logit_rtol,
            atol=config.logit_atol,
        )
        generated = rollout["tokens"]
        matches = sum(left == right for left, right in zip(generated, reference))
        lcp = 0
        for left, right in zip(generated, reference):
            if left != right:
                break
            lcp += 1
        free_matches += matches
        free_total += len(reference)
        free_lcp_total += lcp
        cached_full_numerically_close = cached_full_numerically_close and bool(
            rollout["all_cached_full_numerically_close"]
        )
        cached_full_top1_checks.extend(
            bool(row["top1_agreement"]) for row in rollout["cached_full_checks"]
        )
        free_reports.append(
            {
                "batch_index": batch_index,
                "parent_reference_tokens": reference,
                "child_generated_tokens": generated,
                "token_matches": matches,
                "longest_common_prefix": lcp,
                "cached_full": rollout["cached_full_checks"],
            }
        )

    parent_logits = torch.cat(
        [left.reshape(-1, int(left.shape[-1])) for left, _right in logit_pairs],
        dim=0,
    )
    child_logits = torch.cat(
        [right.reshape(-1, int(right.shape[-1])) for _left, right in logit_pairs],
        dim=0,
    )
    logits = _logit_metrics(parent_logits, child_logits)
    layer_metrics = []
    for index in range(len(parent_rows.layer_kinds)):
        metrics = _residual_metrics(
            parent_rows.field(index, "residual.post_layer"),
            child_rows.field(index, "residual.post_layer"),
        )
        layer_metrics.append({"layer_index": index, **metrics})
    mean_residual_cosine = sum(row["mean_cosine"] for row in layer_metrics) / len(
        layer_metrics
    )
    mean_relative_rmse = sum(row["relative_rmse"] for row in layer_metrics) / len(
        layer_metrics
    )
    free_match = 1.0 if free_total == 0 else free_matches / free_total
    free_lcp = 1.0 if free_total == 0 else free_lcp_total / free_total
    score = (
        0.35 * logits["top1_agreement"]
        + 0.25 * free_match
        + 0.15 * free_lcp
        + 0.10 * ((logits["mean_cosine"] + 1.0) / 2.0)
        + 0.15 * ((mean_residual_cosine + 1.0) / 2.0)
    )

    reasons = []
    finite_values = (
        logits["top1_agreement"],
        logits["mean_cosine"],
        logits["kl"],
        free_match,
        free_lcp,
        mean_residual_cosine,
        mean_relative_rmse,
        score,
    )
    if not all(math.isfinite(float(value)) for value in finite_values):
        reasons.append("candidate trajectory diagnostics contain non-finite values")
    if mean_residual_cosine < config.min_mean_residual_cosine:
        reasons.append("mean residual-path cosine is below threshold")
    if previous is not None:
        if layer_index is not None and config.require_layer_nonregression:
            previous_layer = previous["residual_layers"][layer_index]
            candidate_layer = layer_metrics[layer_index]
            if (
                candidate_layer["relative_rmse"]
                > previous_layer["relative_rmse"]
                + config.max_layer_relative_rmse_regression
            ):
                reasons.append("the folded layer residual endpoint regressed")

    return {
        "schema": "neural-forge-lfm2-multistep-trajectory-gate-v1",
        "passed": not reasons,
        "reasons": reasons,
        "teacher_forced": logits,
        "free_run": {
            "token_match_fraction": free_match,
            "lcp_fraction": free_lcp,
            "reports": free_reports,
            "child_generated_prefixes": True,
            "parent_forward_passes": 0,
        },
        "cached_full_self_consistent": cached_full_numerically_close,
        "cached_full_top1_diagnostic_fraction": (
            1.0
            if not cached_full_top1_checks
            else sum(cached_full_top1_checks) / len(cached_full_top1_checks)
        ),
        "cached_parent_trajectory_is_canonical": True,
        "child_trace_mode": (
            FULL_PREFIX_CANONICAL_TRACE_MODE
            if child_full_prefix_canonical
            else "cached_verified"
        ),
        "child_trace_logits_source": (
            "use_cache_false_full_prefix"
            if child_full_prefix_canonical
            else "verified_one_token_cache"
        ),
        "child_cached_logits_accepted": not child_full_prefix_canonical,
        "full_prefix_top1_is_diagnostic_only": True,
        "full_prefix_numerics_are_diagnostic_only": True,
        "parent_greedy_reference_verified": parent_reference_valid,
        "parent_reference_token_check_is_diagnostic_only": True,
        "mean_residual_cosine": mean_residual_cosine,
        "mean_residual_relative_rmse": mean_relative_rmse,
        "residual_layers": layer_metrics,
        "score": score,
        "thresholds": config.as_dict(),
        "exact_token_and_lcp_metrics_are_diagnostic_only": True,
        "teacher_forced_top1_is_not_an_acceptance_criterion": True,
        "free_run_token_match_is_not_an_acceptance_criterion": True,
        "score_is_not_an_acceptance_criterion": True,
        "internal_acceptance_scope": (
            "data-integrity, finite execution, and residual-path safety only"
        ),
        "zero_gradient": True,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }


MANUAL_RUBRIC = (
    "factual_task_correctness",
    "relevance_instruction_following",
    "coherence_readability",
    "reasoning_continuity_consistency",
    "stability_no_degeneration",
)
MANUAL_CRITICAL_FAILS = (
    "incoherent_or_collapsed",
    "unsafe_nonsense",
    "task_not_attempted",
)


@dataclass(frozen=True, slots=True)
class ReviewPrompt:
    prompt_id: str
    prompt: str
    context: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.prompt_id,
            "prompt": self.prompt,
            "context": self.context,
        }


def load_review_prompts(path: Path) -> tuple[ReviewPrompt, ...]:
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
        rows = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    result = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise Lfm2ResidualAttentionRunnerError(
                f"Review prompt row {index} is not an object."
            )
        prompt = str(row.get("prompt", "")).strip()
        prompt_id = str(row.get("id", f"review-{index:04d}")).strip()
        if not prompt or not prompt_id or prompt_id in seen:
            raise Lfm2ResidualAttentionRunnerError(
                "Review prompts require unique nonempty ids and prompt text."
            )
        seen.add(prompt_id)
        context_value = row.get("context")
        result.append(
            ReviewPrompt(
                prompt_id=prompt_id,
                prompt=prompt,
                context=(
                    None
                    if context_value is None
                    else str(context_value).strip() or None
                ),
            )
        )
    if not result:
        raise Lfm2ResidualAttentionRunnerError(
            "The held-out manual-review prompt set is empty."
        )
    return tuple(result)


def review_prompt_identity(prompts: Sequence[ReviewPrompt]) -> dict[str, Any]:
    rows = [value.as_dict() for value in prompts]
    return {
        "prompts": rows,
        "prompt_count": len(rows),
        "review_prompts_sha256": _json_sha256(rows),
    }


def _repetition_diagnostics(token_ids: Sequence[int], text: str) -> dict[str, Any]:
    tokens = tuple(int(value) for value in token_ids)
    bigrams = tuple(zip(tokens, tokens[1:]))
    maximum_bigram_share = 0.0
    if bigrams:
        counts: dict[tuple[int, int], int] = {}
        for bigram in bigrams:
            counts[bigram] = counts.get(bigram, 0) + 1
        maximum_bigram_share = max(counts.values()) / len(bigrams)
    return {
        "characters": len(text),
        "tokens": len(tokens),
        "unique_token_ratio": (1.0 if not tokens else len(set(tokens)) / len(tokens)),
        "maximum_repeated_bigram_share": maximum_bigram_share,
        "empty": not bool(text.strip()),
        "fragment_like": len(text.strip()) < 12,
    }


def generate_independent_responses(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[ReviewPrompt],
    *,
    max_new_tokens: int,
) -> tuple[dict[str, Any], ...]:
    """Generate decoded semantic artifacts without a reference-token objective."""

    if not 64 <= max_new_tokens <= 96:
        raise Lfm2ResidualAttentionRunnerError(
            "Manual-review generations must be bounded to 64-96 new tokens."
        )
    device = _model_device(model)
    was_training = bool(model.training)
    model.eval()
    rows = []
    try:
        with torch.inference_mode():
            for prompt in prompts:
                encoded = tokenizer(
                    prompt.prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                input_ids = encoded["input_ids"].to(device=device, dtype=torch.long)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device=device, dtype=torch.long)
                kwargs = {
                    "input_ids": input_ids,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "use_cache": True,
                }
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask
                pad_token_id = getattr(tokenizer, "pad_token_id", None)
                if pad_token_id is not None:
                    kwargs["pad_token_id"] = int(pad_token_id)
                generated = model.generate(**kwargs)
                continuation = generated[:, int(input_ids.shape[1]) :]
                token_ids = [
                    int(value) for value in continuation[0].detach().cpu().tolist()
                ]
                decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                rows.append(
                    {
                        "id": prompt.prompt_id,
                        "prompt": prompt.prompt,
                        "context": prompt.context,
                        "response": decoded,
                        "response_token_ids": token_ids,
                        "diagnostics": _repetition_diagnostics(token_ids, decoded),
                    }
                )
    finally:
        model.train(was_training)
    return tuple(rows)


def _response_map(
    responses: Sequence[Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    result = {str(row["id"]): row for row in responses}
    if len(result) != len(responses):
        raise Lfm2ResidualAttentionRunnerError(
            "Response artifact contains duplicate prompt ids."
        )
    return MappingProxyType(result)


def build_manual_review_packet(
    *,
    layer_index: int,
    candidate_state_sha256: str,
    parent_fingerprint: str,
    prompt_identity: Mapping[str, Any],
    parent_responses: Sequence[Mapping[str, Any]],
    incumbent_responses: Sequence[Mapping[str, Any]],
    candidate_responses: Sequence[Mapping[str, Any]],
    internal_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    parent = _response_map(parent_responses)
    incumbent = _response_map(incumbent_responses)
    candidate = _response_map(candidate_responses)
    prompt_ids = [row["id"] for row in prompt_identity["prompts"]]
    if (
        set(prompt_ids) != set(parent)
        or set(prompt_ids) != set(incumbent)
        or set(prompt_ids) != set(candidate)
    ):
        raise Lfm2ResidualAttentionRunnerError(
            "Parent/incumbent/candidate response ids do not match review prompts."
        )
    comparisons = []
    for prompt_id in prompt_ids:
        comparisons.append(
            {
                "id": prompt_id,
                "prompt": candidate[prompt_id]["prompt"],
                "context": candidate[prompt_id].get("context"),
                "parent_response_context_only": parent[prompt_id]["response"],
                "incumbent_child_response": incumbent[prompt_id]["response"],
                "candidate_child_response": candidate[prompt_id]["response"],
                "parent_diagnostics": parent[prompt_id]["diagnostics"],
                "incumbent_diagnostics": incumbent[prompt_id]["diagnostics"],
                "candidate_diagnostics": candidate[prompt_id]["diagnostics"],
                "parent_wording_is_not_a_reference_string": True,
            }
        )
    body = {
        "schema": "neural-forge-lfm2-manual-intelligence-review-v1",
        "layer_index": layer_index,
        "candidate_state_sha256": candidate_state_sha256,
        "parent_fingerprint": parent_fingerprint,
        "review_prompts_sha256": prompt_identity["review_prompts_sha256"],
        "rubric": {
            "score_range": [0, 4],
            "dimensions": list(MANUAL_RUBRIC),
            "critical_fails": list(MANUAL_CRITICAL_FAILS),
            "promotion_suggestion": {
                "no_critical_fails": True,
                "minimum_median_score": 3.0,
                "minimum_mean_score": 2.8,
                "candidate_semantic_pass_count_may_trail_incumbent_by": 1,
            },
        },
        "comparisons": comparisons,
        "internal_diagnostics": dict(internal_diagnostics),
        "exact_token_match_is_not_a_review_objective": True,
        "manual_decision_required": True,
    }
    return {**body, "packet_sha256": _json_sha256(body)}


def load_manual_scores(path: Path | None) -> Mapping[str, Mapping[str, Any]]:
    if path is None or not path.is_file():
        return MappingProxyType({})
    parsed = json.loads(path.read_text(encoding="utf-8"))
    rows = parsed.get("reviews", parsed) if isinstance(parsed, Mapping) else parsed
    if isinstance(rows, Mapping):
        values = [{**dict(value), "packet_sha256": key} for key, value in rows.items()]
    elif isinstance(rows, list):
        values = rows
    else:
        raise Lfm2ResidualAttentionRunnerError(
            "Manual score file must contain a review list or hash mapping."
        )
    result = {}
    for row in values:
        if not isinstance(row, Mapping):
            raise Lfm2ResidualAttentionRunnerError(
                "Manual review score entry is not an object."
            )
        packet_sha = str(row.get("packet_sha256", ""))
        if len(packet_sha) != 64 or packet_sha in result:
            raise Lfm2ResidualAttentionRunnerError(
                "Manual review entries require unique packet_sha256 values."
            )
        result[packet_sha] = dict(row)
    return MappingProxyType(result)


def evaluate_manual_score(
    packet: Mapping[str, Any],
    score: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a human decision and compute the fixed promotion suggestion."""

    if score.get("packet_sha256") != packet.get("packet_sha256"):
        raise Lfm2ResidualAttentionRunnerError(
            "Manual score is not bound to this candidate review packet."
        )
    decision = str(score.get("decision", "")).strip().lower()
    if decision not in {"commit", "reject"}:
        raise Lfm2ResidualAttentionRunnerError(
            "Manual review decision must be commit or reject."
        )
    prompt_scores = score.get("prompt_scores")
    if not isinstance(prompt_scores, list):
        raise Lfm2ResidualAttentionRunnerError("Manual review requires prompt_scores.")
    expected_ids = {row["id"] for row in packet["comparisons"]}
    observed_ids = {str(row.get("id", "")) for row in prompt_scores}
    if observed_ids != expected_ids:
        raise Lfm2ResidualAttentionRunnerError(
            "Manual score prompt ids do not match the review packet."
        )
    all_scores: list[float] = []
    critical_fail_count = 0
    incumbent_pass_count = 0
    candidate_pass_count = 0
    normalized_rows = []
    for row in prompt_scores:
        dimensions = row.get("scores")
        critical = row.get("critical_fails", {})
        if not isinstance(dimensions, Mapping) or set(dimensions) != set(MANUAL_RUBRIC):
            raise Lfm2ResidualAttentionRunnerError(
                "Every prompt must score every fixed rubric dimension."
            )
        values = []
        for name in MANUAL_RUBRIC:
            value = float(dimensions[name])
            if not 0.0 <= value <= 4.0:
                raise Lfm2ResidualAttentionRunnerError(
                    "Manual rubric scores must be in [0, 4]."
                )
            values.append(value)
            all_scores.append(value)
        if not isinstance(critical, Mapping) or set(critical) != set(
            MANUAL_CRITICAL_FAILS
        ):
            raise Lfm2ResidualAttentionRunnerError(
                "Every prompt must mark every fixed critical-fail field."
            )
        critical_values = {name: bool(critical[name]) for name in MANUAL_CRITICAL_FAILS}
        critical_fail_count += sum(critical_values.values())
        incumbent_pass = bool(row.get("incumbent_semantic_pass"))
        candidate_pass = bool(row.get("candidate_semantic_pass"))
        incumbent_pass_count += int(incumbent_pass)
        candidate_pass_count += int(candidate_pass)
        normalized_rows.append(
            {
                "id": str(row["id"]),
                "scores": dict(zip(MANUAL_RUBRIC, values)),
                "critical_fails": critical_values,
                "incumbent_semantic_pass": incumbent_pass,
                "candidate_semantic_pass": candidate_pass,
                "notes": str(row.get("notes", "")),
            }
        )
    mean_score = sum(all_scores) / len(all_scores)
    median_score = float(statistics.median(all_scores))
    semantic_nonregression = candidate_pass_count >= incumbent_pass_count - 1
    suggestion = (
        critical_fail_count == 0
        and median_score >= 3.0
        and mean_score >= 2.8
        and semantic_nonregression
    )
    effective_commit = decision == "commit" and suggestion
    return {
        "packet_sha256": packet["packet_sha256"],
        "reviewer": str(score.get("reviewer", "")).strip(),
        "decision": decision,
        "prompt_scores": normalized_rows,
        "mean_score": mean_score,
        "median_score": median_score,
        "critical_fail_count": critical_fail_count,
        "incumbent_semantic_pass_count": incumbent_pass_count,
        "candidate_semantic_pass_count": candidate_pass_count,
        "semantic_nonregression": semantic_nonregression,
        "promotion_suggested": suggestion,
        "effective_commit": effective_commit,
        "manual_decision_overrides_exact_token_diagnostics": True,
    }


@dataclass(frozen=True, slots=True)
class ManualReviewContext:
    tokenizer: Any
    prompts: tuple[ReviewPrompt, ...]
    prompt_identity: Mapping[str, Any]
    parent_responses: tuple[Mapping[str, Any], ...]
    score_records: Mapping[str, Mapping[str, Any]]
    packet_directory: Path
    max_new_tokens: int = 80


@dataclass(frozen=True, slots=True)
class LayerFoldRuntimeConfig:
    max_fit_rows: int = 512
    max_validation_rows: int = 128
    validation_fraction: float = 0.20
    endpoint_refinement_passes: int = 2

    def __post_init__(self) -> None:
        for name in (
            "max_fit_rows",
            "max_validation_rows",
            "endpoint_refinement_passes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        fraction = float(self.validation_fraction)
        if not 0.0 <= fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1).")
        object.__setattr__(self, "validation_fraction", fraction)


def build_child_trace_cache(
    cache_root: Path,
    child: nn.Module,
    calibration_batches: Sequence[Mapping[str, Tensor]],
    *,
    capture_config: Lfm2CircuitTraceCaptureConfig,
    batch_splits: Sequence[str] | None = None,
) -> tuple[Lfm2CircuitTraceCache, TraceRowIndex]:
    cache = build_lfm2_circuit_trace_cache(
        cache_root,
        child,
        calibration_batches,
        config=capture_config,
        input_device=_model_device(child),
    )
    return cache, extract_trace_rows(cache, batch_splits=batch_splits)


def fold_one_layer_transaction(
    parent: nn.Module,
    child: nn.Module,
    producer: LayerProducerPlan,
    parent_cache: Lfm2CircuitTraceCache,
    parent_rows: TraceRowIndex,
    current_child_cache: Lfm2CircuitTraceCache,
    current_child_rows: TraceRowIndex,
    current_gate: Mapping[str, Any],
    calibration_batches: Sequence[Mapping[str, Tensor]],
    calibration_splits: Sequence[str],
    *,
    child_cache_root: Path,
    capture_config: Lfm2CircuitTraceCaptureConfig,
    fold_config: Lfm2ClosedLoopFoldConfig,
    runtime_config: LayerFoldRuntimeConfig,
    gate_config: TrajectoryGateConfig,
    manual_review: ManualReviewContext | None = None,
) -> tuple[
    Lfm2CircuitTraceCache,
    TraceRowIndex,
    dict[str, Any],
    dict[str, Any],
]:
    """Run endpoint feedback, then commit or exactly roll back one layer."""

    layer_index = producer.layer_index
    source_layer = _lfm2_layers(parent)[layer_index]
    target_layer = _lfm2_layers(child)[layer_index]
    baseline_snapshot = snapshot_lfm2_layer(target_layer)
    baseline_hash = baseline_snapshot.state_sha256
    incumbent_responses = (
        None
        if manual_review is None
        else generate_independent_responses(
            child,
            manual_review.tokenizer,
            manual_review.prompts,
            max_new_tokens=manual_review.max_new_tokens,
        )
    )
    working_cache = current_child_cache
    working_rows = current_child_rows
    refinement_reports = []
    final_evidence: Lfm2ClosedLoopFoldEvidence | None = None
    final_split: EvidenceSplit | None = None

    try:
        for pass_index in range(runtime_config.endpoint_refinement_passes):
            evidence, split = build_layer_evidence(
                source_layer,
                parent_rows,
                working_rows,
                producer,
                max_fit_rows=runtime_config.max_fit_rows,
                max_validation_rows=runtime_config.max_validation_rows,
                validation_fraction=runtime_config.validation_fraction,
            )
            if pass_index + 1 == runtime_config.endpoint_refinement_passes:
                final_evidence = evidence
                final_split = split
                break
            provisional_closed_loop = build_lfm2_closed_loop_fold_plan(
                source_layer,
                target_layer,
                evidence,
                config=fold_config,
            )
            assert_closed_loop_preserves_producer_plan(
                provisional_closed_loop, producer
            )
            _copy_plan_state(target_layer, provisional_closed_loop)
            working_cache, working_rows = build_child_trace_cache(
                child_cache_root,
                child,
                calibration_batches,
                capture_config=capture_config,
                batch_splits=calibration_splits,
            )
            refinement_reports.append(
                {
                    "pass_index": pass_index,
                    "plan": provisional_closed_loop.as_dict(),
                    "child_cache_key": working_cache.cache_key,
                    "child_trace_rows_sha256": (working_rows.trace_rows_sha256),
                    "transactional_final_install": False,
                    "provisional_endpoint_feedback": True,
                }
            )
        assert final_evidence is not None and final_split is not None
        if not final_split.validation_indices:
            raise Lfm2ResidualAttentionRunnerError(
                "A layer cannot be committed without disjoint validation paths."
            )

        if not restore_lfm2_layer_snapshot(target_layer, baseline_snapshot):
            raise Lfm2ResidualAttentionRunnerError(
                "Could not restore the pre-refinement layer before final planning."
            )
        final_plan = build_lfm2_closed_loop_fold_plan(
            source_layer,
            target_layer,
            final_evidence,
            config=fold_config,
        )
        assert_closed_loop_preserves_producer_plan(final_plan, producer)
        transaction = install_lfm2_closed_loop_fold_plan(target_layer, final_plan)
        candidate_cache, candidate_rows = build_child_trace_cache(
            child_cache_root,
            child,
            calibration_batches,
            capture_config=capture_config,
            batch_splits=calibration_splits,
        )
        candidate_gate = evaluate_trajectory_gate(
            parent_cache,
            candidate_cache,
            child,
            parent_rows,
            candidate_rows,
            config=gate_config,
            layer_index=layer_index,
            previous=current_gate,
        )
        manual_packet = None
        manual_evaluation = None
        manual_pending = False
        if candidate_gate["passed"] and manual_review is not None:
            assert incumbent_responses is not None
            candidate_responses = generate_independent_responses(
                child,
                manual_review.tokenizer,
                manual_review.prompts,
                max_new_tokens=manual_review.max_new_tokens,
            )
            manual_packet = build_manual_review_packet(
                layer_index=layer_index,
                candidate_state_sha256=model_state_sha256(child),
                parent_fingerprint=parent_cache.parent_fingerprint,
                prompt_identity=manual_review.prompt_identity,
                parent_responses=manual_review.parent_responses,
                incumbent_responses=incumbent_responses,
                candidate_responses=candidate_responses,
                internal_diagnostics=candidate_gate,
            )
            manual_review.packet_directory.mkdir(parents=True, exist_ok=True)
            packet_path = (
                manual_review.packet_directory
                / f"layer-{layer_index:03d}-{manual_packet['packet_sha256']}.json"
            )
            _atomic_write_json(packet_path, manual_packet)
            score = manual_review.score_records.get(manual_packet["packet_sha256"])
            if score is None:
                manual_pending = True
            else:
                manual_evaluation = evaluate_manual_score(manual_packet, score)
        manual_commit = manual_review is None or (
            manual_evaluation is not None and manual_evaluation["effective_commit"]
        )
        if candidate_gate["passed"] and manual_commit:
            transaction_report = transaction.commit().as_dict()
            result_cache = candidate_cache
            result_rows = candidate_rows
            result_gate = candidate_gate
            accepted = True
        else:
            transaction_report = transaction.rollback().as_dict()
            restored_hash = snapshot_lfm2_layer(target_layer).state_sha256
            if restored_hash != baseline_hash:
                raise Lfm2ResidualAttentionRunnerError(
                    "Rejected layer transaction did not roll back exactly."
                )
            result_cache = current_child_cache
            result_rows = current_child_rows
            result_gate = dict(current_gate)
            accepted = False
    except BaseException as exc:
        try:
            restored = restore_lfm2_layer_snapshot(
                target_layer,
                baseline_snapshot,
            )
            rollback_snapshot = snapshot_lfm2_layer(target_layer)
        except BaseException as rollback_exc:
            raise Lfm2ResidualAttentionRunnerError(
                "Interrupted layer transaction could not authenticate rollback."
            ) from rollback_exc
        if (
            not restored
            or rollback_snapshot.state_sha256 != baseline_hash
            or rollback_snapshot.parameter_ids != baseline_snapshot.parameter_ids
            or rollback_snapshot.gradient_signature
            != baseline_snapshot.gradient_signature
        ):
            raise Lfm2ResidualAttentionRunnerError(
                "Interrupted layer transaction did not roll back exact state and "
                "Parameter objects."
            ) from exc
        raise

    report = {
        "layer_index": layer_index,
        "kind": producer.kind,
        "accepted": accepted,
        "producer_plan": producer.as_dict(),
        "evidence_split": final_split.as_dict(),
        "evidence_binding": {
            "parent_cache_key": parent_cache.cache_key,
            "parent_fingerprint": parent_cache.parent_fingerprint,
            "corpus_fingerprint": parent_cache.corpus_fingerprint,
            "parent_trace_rows_sha256": parent_rows.trace_rows_sha256,
            "child_trace_cache_key": working_cache.cache_key,
            "child_trace_rows_sha256": working_rows.trace_rows_sha256,
            "producer_plan_sha256": producer.plan_sha256,
            "layer_index": layer_index,
            "ordered_path_sha256": _json_sha256(list(parent_rows.paths)),
            "ordered_split_sha256": _json_sha256(list(parent_rows.path_splits)),
            "selection_path_sha256": _json_sha256(
                [parent_rows.paths[index] for index in producer.selection_indices]
            ),
            "fit_path_sha256": _json_sha256(
                [parent_rows.paths[index] for index in final_split.fit_indices]
            ),
            "validation_path_sha256": _json_sha256(
                [parent_rows.paths[index] for index in final_split.validation_indices]
            ),
            "batch_token_order_bound": True,
        },
        "endpoint_refinement_passes": (runtime_config.endpoint_refinement_passes),
        "refinements": refinement_reports,
        "final_plan": final_plan.as_dict(),
        "candidate_gate": candidate_gate,
        "manual_review": {
            "required": manual_review is not None,
            "pending": manual_pending,
            "packet": manual_packet,
            "evaluation": manual_evaluation,
            "exact_token_match_used_for_decision": False,
        },
        "manual_review_pending": manual_pending,
        "transaction": transaction_report,
        "baseline_layer_state_sha256": baseline_hash,
        "result_layer_state_sha256": snapshot_lfm2_layer(target_layer).state_sha256,
        "rollback_exact": (
            accepted or snapshot_lfm2_layer(target_layer).state_sha256 == baseline_hash
        ),
        "child_trace_refreshed_after_accept": accepted,
        "zero_gradient": True,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    return result_cache, result_rows, result_gate, report


def _managed_child_path(root: Path, path: Path) -> None:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved.parent != resolved_root:
        raise Lfm2ResidualAttentionRunnerError(
            f"Managed checkpoint path escaped output root: {resolved}."
        )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(dict(value)) + b"\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_file_manifest(path: Path) -> dict[str, Any]:
    files = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": file_path.stat().st_size,
                "sha256": _file_sha256(file_path),
            }
        )
    forbidden = [
        row["path"]
        for row in files
        if Path(row["path"]).suffix.lower() in {".pt", ".pth", ".bin"}
    ]
    if forbidden:
        raise Lfm2ResidualAttentionRunnerError(
            f"Checkpoint contains forbidden parameter sidecars: {forbidden}."
        )
    safetensors = [row["path"] for row in files if row["path"].endswith(".safetensors")]
    if not safetensors:
        raise Lfm2ResidualAttentionRunnerError(
            "Checkpoint did not contain safetensors weights."
        )
    return {
        "files": files,
        "file_manifest_sha256": _json_sha256(files),
        "safetensors_files": safetensors,
        "parameter_sidecars": [],
        "no_parameter_sidecars": True,
    }


def _local_checkpoint_state(model: nn.Module) -> dict[str, Tensor]:
    """Return CPU-contiguous tensors with exact tied aliases deduplicated."""

    stored: dict[str, Tensor] = {}
    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    for name, value in model.state_dict().items():
        identity = _storage_identity(value)
        if identity in seen:
            continue
        seen.add(identity)
        stored[name] = value.detach().cpu().contiguous()
    return stored


def atomic_save_local_checkpoint(
    model: nn.Module,
    destination: Path,
    *,
    output_root: Path,
) -> dict[str, Any]:
    """Write one replaceable local JSON+safetensors checkpoint directly."""

    from safetensors.torch import save_file

    output_root.mkdir(parents=True, exist_ok=True)
    _managed_child_path(output_root, destination)
    stage = output_root / f".{destination.name}.stage"
    backup = output_root / f".{destination.name}.backup"
    _managed_child_path(output_root, stage)
    _managed_child_path(output_root, backup)
    for stale in (stage, backup):
        if stale.exists():
            shutil.rmtree(stale)
    stage.mkdir(parents=True, exist_ok=False)
    _atomic_write_json(stage / "config.json", model.config.to_dict())
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        if hasattr(generation_config, "to_diff_dict"):
            generation_payload = generation_config.to_diff_dict()
        else:
            generation_payload = generation_config.to_dict()
        _atomic_write_json(stage / "generation_config.json", generation_payload)
    save_file(
        _local_checkpoint_state(model),
        str(stage / "model.safetensors"),
        metadata={"format": "pt"},
    )
    manifest = _checkpoint_file_manifest(stage)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(stage, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    return {
        "path": str(destination),
        "state_sha256": model_state_sha256(model),
        "unique_parameters": unique_parameter_count(model),
        **manifest,
    }


def load_verified_saved_child(
    model: nn.Module,
    checkpoint: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    """Reload the direct local checkpoint and prove exact identity."""

    expected_state = model_state_sha256(model)
    expected_count = unique_parameter_count(model)
    serialized_config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )
    target = Lfm2TargetSpec(
        query_heads=int(model.config.num_attention_heads),
        key_value_heads=int(model.config.num_key_value_heads),
        intermediate_size=int(model.config.intermediate_size),
    )
    serialized_expected = {
        "num_attention_heads": target.query_heads,
        "num_key_value_heads": target.key_value_heads,
        "num_heads": target.query_heads,
        "head_dim": FIXED_HEAD_DIM,
        "intermediate_size": target.intermediate_size,
        "block_ff_dim": target.intermediate_size,
        "block_auto_adjust_ff_dim": False,
    }
    serialized_observed = {
        name: serialized_config.get(name) for name in serialized_expected
    }
    if serialized_observed != serialized_expected:
        raise Lfm2ResidualAttentionRunnerError(
            "Saved config.json did not preserve the dynamic target aliases: "
            f"{serialized_observed} != {serialized_expected}."
        )
    reloaded = load_local_lfm2_model(
        checkpoint,
        device="cpu",
        dtype=_model_dtype(model),
    )
    if getattr(reloaded.config, "block_ff_dim", None) is None:
        reloaded.config.block_ff_dim = int(reloaded.config.intermediate_size)
    observed_state = model_state_sha256(reloaded)
    observed_count = unique_parameter_count(reloaded)
    expected_config = model.config.to_dict()
    observed_config = reloaded.config.to_dict()
    # This provenance-only field does not affect architecture or execution.
    for payload in (expected_config, observed_config):
        payload.pop("_name_or_path", None)
    config_match = observed_config == expected_config
    aliases = validate_target_aliases(reloaded.config, target)
    dtype_match = _model_dtype(reloaded) == _model_dtype(model)
    gradients = _gradient_audit(reloaded)
    if (
        observed_state != expected_state
        or observed_count != expected_count
        or not config_match
        or not dtype_match
        or not gradients["zero_gradient"]
    ):
        raise Lfm2ResidualAttentionRunnerError(
            "Saved compact child did not reload identically."
        )
    return reloaded, {
        "state_sha256": observed_state,
        "unique_parameters": observed_count,
        "config_exact": config_match,
        "ignored_config_fields": ["_name_or_path"],
        "serialized_aliases": {
            "observed": serialized_observed,
            "expected": serialized_expected,
            "exact": True,
        },
        "aliases": aliases,
        "dtype_exact": dtype_match,
        "zero_gradient": gradients["zero_gradient"],
        "save_reload_verified": True,
    }


def verify_saved_reload(
    model: nn.Module,
    checkpoint: Path,
) -> dict[str, Any]:
    reloaded, report = load_verified_saved_child(model, checkpoint)
    del reloaded
    return report


def validate_lfm2_accounting(
    parent: nn.Module,
    child: nn.Module,
) -> dict[str, Any]:
    parent_count = unique_parameter_count(parent)
    child_count = unique_parameter_count(child)
    if parent_count <= 0 or child_count <= 0:
        raise Lfm2ResidualAttentionRunnerError(
            "Parent and child parameter counts must be positive."
        )
    if child_count >= parent_count:
        raise Lfm2ResidualAttentionRunnerError(
            "The dynamic LFM target must remove at least one learned parameter."
        )
    if _model_dtype(parent) != torch.bfloat16 or _model_dtype(child) != torch.bfloat16:
        raise Lfm2ResidualAttentionRunnerError(
            "The validated real proof requires parent and child BF16 parameters."
        )
    target = Lfm2TargetSpec(
        query_heads=int(child.config.num_attention_heads),
        key_value_heads=int(child.config.num_key_value_heads),
        intermediate_size=int(child.config.intermediate_size),
    )
    aliases = validate_target_aliases(child.config, target)
    savings = parent_count - child_count
    return {
        "schema": DYNAMIC_TARGET_SCHEMA,
        "source_unique_parameters": parent_count,
        "target_unique_parameters": child_count,
        "learned_parameter_savings": savings,
        "fraction_removed": savings / parent_count,
        "compression_ratio": parent_count / child_count,
        "source_query_heads": int(parent.config.num_attention_heads),
        "source_key_value_heads": int(parent.config.num_key_value_heads),
        "target_query_heads": target.query_heads,
        "target_key_value_heads": target.key_value_heads,
        "head_dim": int(child.config.head_dim),
        "source_intermediate_size": int(parent.config.intermediate_size),
        "target_intermediate_size": target.intermediate_size,
        "hidden_size": int(parent.config.hidden_size),
        "observed_parent_unique_parameters": parent_count,
        "observed_target_unique_parameters": child_count,
        "parent_dtype": str(_model_dtype(parent)),
        "target_dtype": str(_model_dtype(child)),
        "target_aliases": aliases,
        "exact": True,
    }


def validate_fixed_accounting(parent: nn.Module, child: nn.Module) -> dict[str, Any]:
    """Compatibility check for the original exact fixed target."""

    report = validate_lfm2_accounting(parent, child)
    if (
        report["source_unique_parameters"] != EXPECTED_PARENT_UNIQUE_PARAMETERS
        or report["target_unique_parameters"] != EXPECTED_TARGET_UNIQUE_PARAMETERS
        or report["learned_parameter_savings"] != EXPECTED_PARAMETER_SAVINGS
    ):
        raise Lfm2ResidualAttentionRunnerError(
            "The original fixed-target parameter accounting changed."
        )
    return {
        **fixed_contract_accounting(),
        **report,
        "schema": TARGET_SCHEMA,
    }


@dataclass(frozen=True, slots=True)
class CalibrationCorpus:
    batches: tuple[Mapping[str, Tensor], ...]
    splits: tuple[str, ...]
    source_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_count": len(self.batches),
            "splits": list(self.splits),
            "split_counts": {
                split: self.splits.count(split) for split in sorted(set(self.splits))
            },
            "source_sha256": self.source_sha256,
        }


def load_calibration_batches(path: Path) -> CalibrationCorpus:
    """Load tokenized, parent-greedy forced trajectories from JSON or JSONL."""

    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
        rows = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    result = []
    splits = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise Lfm2ResidualAttentionRunnerError(
                f"Calibration row {index} is not an object."
            )
        if "input_ids" not in row or "forced_decode_ids" not in row:
            raise Lfm2ResidualAttentionRunnerError(
                "Each calibration row must contain input_ids and "
                "forced_decode_ids. The forced sequence must be a cached "
                "parent-greedy trajectory."
            )
        split = str(row.get("split", "")).strip().lower()
        if split not in {"selection", "fit", "validation"}:
            raise Lfm2ResidualAttentionRunnerError(
                "Each calibration row requires split=selection, fit, or "
                "validation. Held-out intelligence prompts do not belong here."
            )
        input_values = row["input_ids"]
        forced_values = row["forced_decode_ids"]
        input_ids = torch.tensor(input_values, dtype=torch.long)
        forced = torch.tensor(forced_values, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if forced.ndim == 1:
            forced = forced.unsqueeze(0)
        batch = {
            "input_ids": input_ids,
            "forced_decode_ids": forced,
        }
        if "attention_mask" in row:
            mask = torch.tensor(row["attention_mask"], dtype=torch.long)
            batch["attention_mask"] = mask.unsqueeze(0) if mask.ndim == 1 else mask
        if "position_ids" in row:
            position = torch.tensor(row["position_ids"], dtype=torch.long)
            batch["position_ids"] = (
                position.unsqueeze(0) if position.ndim == 1 else position
            )
        result.append(batch)
        splits.append(split)
    if not result:
        raise Lfm2ResidualAttentionRunnerError("Calibration input contained no rows.")
    if "selection" not in splits or "fit" not in splits:
        raise Lfm2ResidualAttentionRunnerError(
            "Calibration must contain at least one selection and one fit row."
        )
    return CalibrationCorpus(
        batches=tuple(result),
        splits=tuple(splits),
        source_sha256=_file_sha256(path),
    )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    parent_model: str
    calibration_path: Path
    cache_root: Path
    output_root: Path
    target_query_heads: int = TARGET_QUERY_HEADS
    target_key_value_heads: int = TARGET_KEY_VALUE_HEADS
    target_intermediate_size: int = TARGET_INTERMEDIATE_SIZE
    review_prompts_path: Path | None = None
    manual_scores_path: Path | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    blank_seeds: tuple[int, ...] = (17, 29)
    max_layers: int | None = None
    max_selection_rows: int = 256
    sample_last_tokens: int = 8
    max_decode_steps: int = 16
    max_capture_bytes: int = 512 * 1024 * 1024
    capture_logit_rtol: float = 5e-2
    capture_logit_atol: float = 5e-1
    endpoint_refinement_passes: int = 2
    max_fit_rows: int = 512
    max_validation_rows: int = 128
    validation_fraction: float = 0.20
    endpoint_ridge: float = 1e-6
    max_endpoint_relative_rmse: float = 0.75
    min_endpoint_cosine_similarity: float = 0.20
    resume: bool = False
    save_blank: bool = True
    manual_review: bool = False
    manual_review_scope: str = "final"
    review_max_new_tokens: int = 80

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_model": self.parent_model,
            "calibration_path": str(self.calibration_path),
            "cache_root": str(self.cache_root),
            "output_root": str(self.output_root),
            "target_query_heads": self.target_query_heads,
            "target_key_value_heads": self.target_key_value_heads,
            "target_intermediate_size": self.target_intermediate_size,
            "review_prompts_path": (
                None
                if self.review_prompts_path is None
                else str(self.review_prompts_path)
            ),
            "manual_scores_path": (
                None
                if self.manual_scores_path is None
                else str(self.manual_scores_path)
            ),
            "device": self.device,
            "dtype": self.dtype,
            "blank_seeds": list(self.blank_seeds),
            "max_layers": self.max_layers,
            "max_selection_rows": self.max_selection_rows,
            "sample_last_tokens": self.sample_last_tokens,
            "max_decode_steps": self.max_decode_steps,
            "max_capture_bytes": self.max_capture_bytes,
            "capture_logit_rtol": self.capture_logit_rtol,
            "capture_logit_atol": self.capture_logit_atol,
            "endpoint_refinement_passes": self.endpoint_refinement_passes,
            "max_fit_rows": self.max_fit_rows,
            "max_validation_rows": self.max_validation_rows,
            "validation_fraction": self.validation_fraction,
            "endpoint_ridge": self.endpoint_ridge,
            "max_endpoint_relative_rmse": self.max_endpoint_relative_rmse,
            "min_endpoint_cosine_similarity": (self.min_endpoint_cosine_similarity),
            "resume": self.resume,
            "save_blank": self.save_blank,
            "manual_review": self.manual_review,
            "manual_review_scope": self.manual_review_scope,
            "review_max_new_tokens": self.review_max_new_tokens,
        }

    @property
    def target_spec(self) -> Lfm2TargetSpec:
        return Lfm2TargetSpec(
            query_heads=self.target_query_heads,
            key_value_heads=self.target_key_value_heads,
            intermediate_size=self.target_intermediate_size,
        )


def _torch_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    values = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return values[normalized]
    except KeyError as exc:
        raise Lfm2ResidualAttentionRunnerError(f"Unsupported dtype: {name!r}.") from exc


def _load_parent(config: ExperimentConfig) -> nn.Module:
    device = torch.device(config.device)
    dtype = _torch_dtype(config.dtype)
    parent = load_local_lfm2_model(
        config.parent_model,
        device=device,
        dtype=dtype,
    )
    return parent


def load_or_build_parent_review_responses(
    path: Path,
    parent: nn.Module,
    tokenizer: Any,
    prompts: Sequence[ReviewPrompt],
    *,
    parent_fingerprint: str,
    max_new_tokens: int,
) -> tuple[Mapping[str, Any], ...]:
    prompt_identity = review_prompt_identity(prompts)
    if path.is_file():
        document = json.loads(path.read_text(encoding="utf-8"))
        body = dict(document)
        seal = body.pop("artifact_sha256", None)
        if (
            seal != _json_sha256(body)
            or body.get("parent_fingerprint") != parent_fingerprint
            or body.get("review_prompts_sha256")
            != prompt_identity["review_prompts_sha256"]
            or int(body.get("max_new_tokens", -1)) != max_new_tokens
        ):
            raise Lfm2ResidualAttentionRunnerError(
                "Cached parent whole-response review artifact changed identity."
            )
        responses = body.get("responses")
        if not isinstance(responses, list):
            raise Lfm2ResidualAttentionRunnerError(
                "Cached parent review responses are invalid."
            )
        return tuple(dict(row) for row in responses)
    responses = generate_independent_responses(
        parent,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
    )
    body = {
        "schema": "neural-forge-lfm2-parent-review-responses-v1",
        "parent_fingerprint": parent_fingerprint,
        "review_prompts_sha256": prompt_identity["review_prompts_sha256"],
        "max_new_tokens": max_new_tokens,
        "responses": list(responses),
        "parent_responses_generated_once": True,
        "parent_wording_is_context_only": True,
    }
    _atomic_write_json(path, {**body, "artifact_sha256": _json_sha256(body)})
    return tuple(responses)


def _base_state(
    config: ExperimentConfig,
    *,
    parent_cache: Lfm2CircuitTraceCache,
    target_identity: Mapping[str, Any],
    producer_identity: Mapping[str, Any],
    seed_audit: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    baseline_gate: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "status": "running",
        "config": config.as_dict(),
        "parent_cache_key": parent_cache.cache_key,
        "parent_fingerprint": parent_cache.parent_fingerprint,
        "corpus_fingerprint": parent_cache.corpus_fingerprint,
        "target_config_sha256": target_identity["config_sha256"],
        "producer_bundle_sha256": producer_identity["producer_bundle_sha256"],
        "seed_erasure": dict(seed_audit),
        "current_checkpoint": dict(checkpoint),
        "completed_layers": [],
        "accepted_layers": [],
        "rejected_layers": [],
        "layer_reports": [],
        "current_gate": dict(baseline_gate),
        "zero_gradient": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "training_steps": 0,
    }


def run_experiment(
    config: ExperimentConfig,
    *,
    gate_config: TrajectoryGateConfig | None = None,
) -> dict[str, Any]:
    """Execute or resume the bounded real LFM2 residual-attention fold proof."""

    selected_gate = gate_config or TrajectoryGateConfig(
        logit_rtol=config.capture_logit_rtol,
        logit_atol=config.capture_logit_atol,
    )
    if config.manual_review_scope not in {"layer", "final"}:
        raise Lfm2ResidualAttentionRunnerError(
            "manual_review_scope must be 'layer' or 'final'."
        )
    output_root = config.output_root.expanduser().absolute()
    cache_root = config.cache_root.expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "run_state.json"
    report_path = output_root / "report.json"
    current_checkpoint_path = output_root / "current-child"
    final_checkpoint_path = output_root / "final-child"
    blank_checkpoint_path = output_root / "blank-child"

    local_model_io = direct_local_io_audit(config.parent_model)
    parent = _load_parent(config)
    device = _model_device(parent)
    dtype = _model_dtype(parent)
    calibration = load_calibration_batches(config.calibration_path)
    capture_config = Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=config.sample_last_tokens,
        max_decode_steps=config.max_decode_steps,
        max_capture_bytes=config.max_capture_bytes,
        logit_rtol=config.capture_logit_rtol,
        logit_atol=config.capture_logit_atol,
        require_top1_agreement=False,
    )
    parent_cache = build_lfm2_circuit_trace_cache(
        cache_root / "parent",
        parent,
        calibration.batches,
        config=capture_config,
        input_device=device,
    )
    parent_rows = extract_trace_rows(parent_cache, batch_splits=calibration.splits)
    cached_batches = calibration_batches_from_cache(parent_cache)
    manual_context = None
    if config.manual_review:
        if config.review_prompts_path is None:
            raise Lfm2ResidualAttentionRunnerError(
                "Manual intelligence review requires --review-prompts."
            )
        if (
            config.review_prompts_path.expanduser().resolve()
            == config.calibration_path.expanduser().resolve()
        ):
            raise Lfm2ResidualAttentionRunnerError(
                "Held-out review prompts must be outside the circuit calibration artifact."
            )
        tokenizer = load_local_tokenizer(config.parent_model)
        prompts = load_review_prompts(config.review_prompts_path)
        prompt_identity = review_prompt_identity(prompts)
        parent_responses = load_or_build_parent_review_responses(
            output_root / "parent-review-responses.json",
            parent,
            tokenizer,
            prompts,
            parent_fingerprint=parent_cache.parent_fingerprint,
            max_new_tokens=config.review_max_new_tokens,
        )
        manual_context = ManualReviewContext(
            tokenizer=tokenizer,
            prompts=prompts,
            prompt_identity=MappingProxyType(prompt_identity),
            parent_responses=parent_responses,
            score_records=load_manual_scores(config.manual_scores_path),
            packet_directory=output_root / "manual-review",
            max_new_tokens=config.review_max_new_tokens,
        )
    target_config = build_lfm2_target_config(
        parent.config,
        config.target_spec,
    )
    target_identity = target_config_identity(target_config)
    primary_shell = instantiate_blank_child(
        parent,
        target_config,
        seed=config.blank_seeds[0],
        device=device,
        dtype=dtype,
    )
    blank_hash = model_state_sha256(primary_shell)
    if config.save_blank and not config.resume:
        blank_checkpoint = atomic_save_local_checkpoint(
            primary_shell,
            blank_checkpoint_path,
            output_root=output_root,
        )
    else:
        blank_checkpoint = {
            "path": str(blank_checkpoint_path),
            "state_sha256": blank_hash,
            "saved": False,
        }

    fold_config = Lfm2ClosedLoopFoldConfig(
        endpoint_ridge=config.endpoint_ridge,
        max_endpoint_relative_rmse=config.max_endpoint_relative_rmse,
        min_endpoint_cosine_similarity=(config.min_endpoint_cosine_similarity),
    )
    producer_plans = derive_all_producer_plans(
        parent,
        primary_shell,
        parent_rows,
        fold_config=fold_config,
        max_selection_rows=config.max_selection_rows,
        max_fit_rows=config.max_fit_rows,
        max_validation_rows=config.max_validation_rows,
        validation_fraction=config.validation_fraction,
    )
    producer_identity = producer_bundle_identity(producer_plans)

    if config.resume:
        if not state_path.is_file() or not current_checkpoint_path.is_dir():
            raise Lfm2ResidualAttentionRunnerError(
                "--resume requires run_state.json and current-child."
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        checks = {
            "schema": state.get("schema") == STATE_SCHEMA,
            "parent": state.get("parent_cache_key") == parent_cache.cache_key,
            "target": (
                state.get("target_config_sha256") == target_identity["config_sha256"]
            ),
            "producers": (
                state.get("producer_bundle_sha256")
                == producer_identity["producer_bundle_sha256"]
            ),
        }
        if not all(checks.values()):
            raise Lfm2ResidualAttentionRunnerError(
                f"Resume identity check failed: {checks}."
            )
        child = load_local_lfm2_model(
            current_checkpoint_path,
            device=device,
            dtype=dtype,
        )
        if getattr(child.config, "block_ff_dim", None) is None:
            child.config.block_ff_dim = int(child.config.intermediate_size)
        if model_state_sha256(child) != state["current_checkpoint"]["state_sha256"]:
            raise Lfm2ResidualAttentionRunnerError(
                "Resume checkpoint state hash changed."
            )
        del primary_shell
        seed_audit = state["seed_erasure"]
        blank_checkpoint = state.get("blank_checkpoint", blank_checkpoint)
    else:
        child, seed_audit = audit_blank_seed_erasure(
            parent,
            target_config,
            producer_plans,
            seeds=config.blank_seeds,
            device=device,
            dtype=dtype,
            primary_child=primary_shell,
        )
        state_global_transplant = audit_global_transplant(parent, child)
        checkpoint = atomic_save_local_checkpoint(
            child,
            current_checkpoint_path,
            output_root=output_root,
        )
        child_cache, child_rows = build_child_trace_cache(
            cache_root / "children",
            child,
            cached_batches,
            capture_config=capture_config,
            batch_splits=calibration.splits,
        )
        baseline_gate = evaluate_trajectory_gate(
            parent_cache,
            child_cache,
            child,
            parent_rows,
            child_rows,
            config=selected_gate,
        )
        state = _base_state(
            config,
            parent_cache=parent_cache,
            target_identity=target_identity,
            producer_identity=producer_identity,
            seed_audit=seed_audit,
            checkpoint=checkpoint,
            baseline_gate=baseline_gate,
        )
        state["global_transplant"] = state_global_transplant
        state["blank_checkpoint"] = blank_checkpoint
        state["local_model_io"] = local_model_io
        _atomic_write_json(state_path, state)

    state["local_model_io"] = local_model_io
    child_cache, child_rows = build_child_trace_cache(
        cache_root / "children",
        child,
        cached_batches,
        capture_config=capture_config,
        batch_splits=calibration.splits,
    )
    current_gate = evaluate_trajectory_gate(
        parent_cache,
        child_cache,
        child,
        parent_rows,
        child_rows,
        config=selected_gate,
    )
    completed = {int(value) for value in state.get("completed_layers", [])}
    runtime_config = LayerFoldRuntimeConfig(
        max_fit_rows=config.max_fit_rows,
        max_validation_rows=config.max_validation_rows,
        validation_fraction=config.validation_fraction,
        endpoint_refinement_passes=config.endpoint_refinement_passes,
    )
    layer_order = list(range(len(producer_plans)))
    if config.max_layers is not None:
        layer_order = layer_order[: config.max_layers]

    for layer_index in layer_order:
        if layer_index in completed:
            continue
        try:
            (
                child_cache,
                child_rows,
                current_gate,
                layer_report,
            ) = fold_one_layer_transaction(
                parent,
                child,
                producer_plans[layer_index],
                parent_cache,
                parent_rows,
                child_cache,
                child_rows,
                current_gate,
                cached_batches,
                calibration.splits,
                child_cache_root=cache_root / "children",
                capture_config=capture_config,
                fold_config=fold_config,
                runtime_config=runtime_config,
                gate_config=selected_gate,
                manual_review=(
                    manual_context
                    if config.manual_review and config.manual_review_scope == "layer"
                    else None
                ),
            )
        except Lfm2ClosedLoopFitRejected as exc:
            layer_report = {
                "layer_index": layer_index,
                "kind": producer_plans[layer_index].kind,
                "accepted": False,
                "fit_rejected_before_commit": True,
                "reason": str(exc),
                "producer_plan": producer_plans[layer_index].as_dict(),
                "rollback_exact": True,
                "child_trace_refreshed_after_accept": False,
                "zero_gradient": True,
                "optimizer_steps": 0,
                "backward_calls": 0,
            }
        if layer_report.get("manual_review_pending"):
            state["status"] = "manual_review_pending"
            state["pending_review"] = layer_report
            state["zero_gradient_audit"] = _gradient_audit(child)
            _atomic_write_json(state_path, state)
            pending_report = {
                "schema": RUN_SCHEMA,
                "status": "manual_review_pending",
                "config": config.as_dict(),
                "local_model_io": local_model_io,
                "parent_cache_key": parent_cache.cache_key,
                "target": target_identity,
                "accounting": validate_lfm2_accounting(parent, child),
                "producer_plans": producer_identity,
                "completed_layers": state.get("completed_layers", []),
                "accepted_layers": state.get("accepted_layers", []),
                "rejected_layers": state.get("rejected_layers", []),
                "pending_review": layer_report["manual_review"],
                "next_action": (
                    "Score the emitted packet with the fixed 0-4 rubric, "
                    "bind it by packet_sha256 in --manual-scores, then rerun "
                    "with --resume."
                ),
                "folding_only": True,
                "exact_token_match_used_for_promotion": False,
            }
            pending_report["report_sha256"] = _json_sha256(pending_report)
            _atomic_write_json(report_path, pending_report)
            return pending_report
        state.pop("pending_review", None)
        completed.add(layer_index)
        state["completed_layers"] = sorted(completed)
        state["layer_reports"].append(layer_report)
        bucket = "accepted_layers" if layer_report["accepted"] else "rejected_layers"
        state[bucket] = sorted(
            set(int(value) for value in state.get(bucket, [])) | {layer_index}
        )
        state["current_gate"] = current_gate
        if layer_report["accepted"]:
            state["current_checkpoint"] = atomic_save_local_checkpoint(
                child,
                current_checkpoint_path,
                output_root=output_root,
            )
        state["zero_gradient_audit"] = _gradient_audit(child)
        if not state["zero_gradient_audit"]["zero_gradient"]:
            raise Lfm2ResidualAttentionRunnerError(
                "A layer transaction wrote a parameter gradient."
            )
        _atomic_write_json(state_path, state)

    accounting = validate_lfm2_accounting(parent, child)
    global_transplant = audit_global_transplant(parent, child)
    final_checkpoint = atomic_save_local_checkpoint(
        child,
        final_checkpoint_path,
        output_root=output_root,
    )
    in_memory_final_gate = evaluate_trajectory_gate(
        parent_cache,
        child_cache,
        child,
        parent_rows,
        child_rows,
        config=selected_gate,
    )
    reloaded_child, reload_report = load_verified_saved_child(
        child, final_checkpoint_path
    )
    reloaded_child.to(device=device, dtype=dtype)
    reloaded_child.eval()
    reload_cache, reload_rows = build_child_trace_cache(
        cache_root / "reload-replay",
        reloaded_child,
        cached_batches,
        capture_config=capture_config,
        batch_splits=calibration.splits,
    )
    final_gate = evaluate_trajectory_gate(
        parent_cache,
        reload_cache,
        reloaded_child,
        parent_rows,
        reload_rows,
        config=selected_gate,
    )
    trajectory_keys = (
        "teacher_forced",
        "free_run",
        "mean_residual_cosine",
        "mean_residual_relative_rmse",
        "residual_layers",
        "score",
    )
    in_memory_trajectory = {
        name: in_memory_final_gate[name] for name in trajectory_keys
    }
    reload_trajectory = {name: final_gate[name] for name in trajectory_keys}
    reload_trajectory_equivalent = _json_sha256(in_memory_trajectory) == _json_sha256(
        reload_trajectory
    )
    if not reload_trajectory_equivalent:
        raise Lfm2ResidualAttentionRunnerError(
            "Reloaded child trajectory differs from the saved in-memory child."
        )
    final_manual_packet = None
    final_manual_evaluation = None
    final_manual_pending = False
    if (
        config.manual_review
        and config.manual_review_scope == "final"
        and manual_context is not None
    ):
        candidate_responses = generate_independent_responses(
            reloaded_child,
            manual_context.tokenizer,
            manual_context.prompts,
            max_new_tokens=manual_context.max_new_tokens,
        )
        final_manual_packet = build_manual_review_packet(
            layer_index=len(producer_plans),
            candidate_state_sha256=model_state_sha256(reloaded_child),
            parent_fingerprint=parent_cache.parent_fingerprint,
            prompt_identity=manual_context.prompt_identity,
            parent_responses=manual_context.parent_responses,
            incumbent_responses=manual_context.parent_responses,
            candidate_responses=candidate_responses,
            internal_diagnostics=final_gate,
        )
        manual_context.packet_directory.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            manual_context.packet_directory
            / (f"final-{final_manual_packet['packet_sha256']}.json"),
            final_manual_packet,
        )
        score = manual_context.score_records.get(final_manual_packet["packet_sha256"])
        if score is None:
            final_manual_pending = True
        else:
            final_manual_evaluation = evaluate_manual_score(
                final_manual_packet,
                score,
            )
    del reloaded_child
    final_status = "complete"
    if final_manual_pending:
        final_status = "final_manual_review_pending"
    elif (
        final_manual_evaluation is not None
        and not final_manual_evaluation["effective_commit"]
    ):
        final_status = "final_manual_review_rejected"
    report = {
        "schema": RUN_SCHEMA,
        "status": final_status,
        "config": config.as_dict(),
        "local_model_io": local_model_io,
        "parent": {
            "model": config.parent_model,
            "cache": parent_cache.manifest,
            "trace_rows": parent_rows.as_dict(),
            "calibration": calibration.as_dict(),
            "forward_traces_reused_after_cache": True,
        },
        "target": target_identity,
        "accounting": accounting,
        "global_transplant": global_transplant,
        "producer_plans": producer_identity,
        "seed_erasure": seed_audit,
        "blank_checkpoint": blank_checkpoint,
        "completed_layers": state["completed_layers"],
        "accepted_layers": state["accepted_layers"],
        "rejected_layers": state["rejected_layers"],
        "layer_reports": state["layer_reports"],
        "final_gate": final_gate,
        "final_manual_review": {
            "required": (
                config.manual_review and config.manual_review_scope == "final"
            ),
            "pending": final_manual_pending,
            "packet": final_manual_packet,
            "evaluation": final_manual_evaluation,
            "intermediate_manual_pauses": False,
            "exact_token_match_used_for_decision": False,
        },
        "final_checkpoint": final_checkpoint,
        "reload": reload_report,
        "reload_parent_free_replay": {
            "parent_forward_passes": 0,
            "reloaded_child_trace_cache_key": reload_cache.cache_key,
            "reloaded_child_trace_rows_sha256": reload_rows.trace_rows_sha256,
            "trajectory_equivalent_to_in_memory_child": (reload_trajectory_equivalent),
            "trajectory_sha256": _json_sha256(reload_trajectory),
        },
        "zero_gradient_audit": _gradient_audit(child),
        "folding_only": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "training_steps": 0,
        "parameter_gradients_written": False,
        "no_parameter_sidecars": True,
    }
    report["report_sha256"] = _json_sha256(report)
    _atomic_write_json(report_path, report)
    state.update(
        {
            "status": final_status,
            "final_checkpoint": final_checkpoint,
            "final_gate": final_gate,
            "report_path": str(report_path),
            "report_sha256": report["report_sha256"],
        }
    )
    _atomic_write_json(state_path, state)
    return report


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Blank seeds must be comma-separated integers."
        ) from exc
    if len(result) < 2 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Provide at least two distinct blank seeds.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-gradient residual-attention-aware folding of the local "
            "LiquidAI LFM2.5-230M parent into an automatically specified "
            "smaller LFM2 topology. The compatibility default remains the "
            "225,957,632-parameter 14Q/7KV, 2528-FFN target."
        )
    )
    parser.add_argument("--parent-model", required=True)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument(
        "--target-query-heads",
        type=int,
        default=TARGET_QUERY_HEADS,
    )
    parser.add_argument(
        "--target-key-value-heads",
        type=int,
        default=TARGET_KEY_VALUE_HEADS,
    )
    parser.add_argument(
        "--target-intermediate-size",
        type=int,
        default=TARGET_INTERMEDIATE_SIZE,
    )
    parser.add_argument(
        "--review-prompts",
        type=Path,
        help="Held-out whole-response prompts; never used for folding decisions.",
    )
    parser.add_argument(
        "--manual-scores",
        type=Path,
        help="JSON manual-review decisions bound to packet_sha256.",
    )
    parser.add_argument(
        "--manual-review",
        action="store_true",
        default=False,
        help="Emit a fixed-rubric whole-response review packet.",
    )
    parser.add_argument(
        "--manual-review-scope",
        choices=("final", "layer"),
        default="final",
        help="Default final-only; layer mode preserves the legacy pause behavior.",
    )
    parser.add_argument("--review-max-new-tokens", type=int, default=80)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--blank-seeds", type=_parse_seeds, default=(17, 29))
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--max-selection-rows", type=int, default=256)
    parser.add_argument("--sample-last-tokens", type=int, default=8)
    parser.add_argument("--max-decode-steps", type=int, default=16)
    parser.add_argument("--max-capture-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--capture-logit-rtol", type=float, default=5e-2)
    parser.add_argument("--capture-logit-atol", type=float, default=5e-1)
    parser.add_argument("--endpoint-refinement-passes", type=int, default=2)
    parser.add_argument("--max-fit-rows", type=int, default=512)
    parser.add_argument("--max-validation-rows", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--endpoint-ridge", type=float, default=1e-6)
    parser.add_argument("--max-endpoint-relative-rmse", type=float, default=0.75)
    parser.add_argument("--min-endpoint-cosine-similarity", type=float, default=0.20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-save-blank", action="store_true")
    parser.add_argument("--min-teacher-top1", type=float, default=0.05)
    parser.add_argument("--min-free-run-match", type=float, default=0.05)
    parser.add_argument("--min-residual-cosine", type=float, default=0.0)
    parser.add_argument("--max-score-regression", type=float, default=0.0)
    parser.add_argument("--max-layer-rmse-regression", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(
        parent_model=args.parent_model,
        calibration_path=args.calibration,
        cache_root=args.cache_root,
        output_root=args.output_root,
        target_query_heads=args.target_query_heads,
        target_key_value_heads=args.target_key_value_heads,
        target_intermediate_size=args.target_intermediate_size,
        review_prompts_path=args.review_prompts,
        manual_scores_path=args.manual_scores,
        device=args.device,
        dtype=args.dtype,
        blank_seeds=args.blank_seeds,
        max_layers=args.max_layers,
        max_selection_rows=args.max_selection_rows,
        sample_last_tokens=args.sample_last_tokens,
        max_decode_steps=args.max_decode_steps,
        max_capture_bytes=args.max_capture_bytes,
        capture_logit_rtol=args.capture_logit_rtol,
        capture_logit_atol=args.capture_logit_atol,
        endpoint_refinement_passes=args.endpoint_refinement_passes,
        max_fit_rows=args.max_fit_rows,
        max_validation_rows=args.max_validation_rows,
        validation_fraction=args.validation_fraction,
        endpoint_ridge=args.endpoint_ridge,
        max_endpoint_relative_rmse=args.max_endpoint_relative_rmse,
        min_endpoint_cosine_similarity=(args.min_endpoint_cosine_similarity),
        resume=args.resume,
        save_blank=not args.no_save_blank,
        manual_review=args.manual_review,
        manual_review_scope=args.manual_review_scope,
        review_max_new_tokens=args.review_max_new_tokens,
    )
    gate = TrajectoryGateConfig(
        min_teacher_forced_top1=args.min_teacher_top1,
        min_free_run_token_match=args.min_free_run_match,
        min_mean_residual_cosine=args.min_residual_cosine,
        max_score_regression=args.max_score_regression,
        max_layer_relative_rmse_regression=(args.max_layer_rmse_regression),
    )
    report = run_experiment(config, gate_config=gate)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
