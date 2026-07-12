"""Deterministic, architecture-neutral on-policy probe corpus construction.

Only the child causal LM is executed.  Its greedy prefixes are decoded to
exact UTF-8 text, content-addressed, deduplicated, and independently tokenized
for the child and every parent tokenizer.  The resulting tuple-backed batches
can be materialized as fresh tensors for ``hf_fold_trace_cache`` without
sharing mutable token tensors between models.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import operator
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


PROBE_SCHEMA = "neural-forge-on-policy-probe-corpus-v1"
GENERATION_CACHE_SCHEMA = "neural-forge-on-policy-generation-cache-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class OnPolicyProbeError(RuntimeError):
    """Base error for invalid probe inputs or unsafe execution."""


class OnPolicyProbeIntegrityError(OnPolicyProbeError):
    """Raised when cached or runtime evidence fails strict verification."""


class OnPolicyProbeConflictError(OnPolicyProbeError):
    """Raised when an immutable cache belongs to a different request."""


@dataclass(frozen=True, slots=True)
class GreedyProbeSettings:
    """Fixed generation and trace batching contract."""

    max_new_tokens: int = 8
    max_context_tokens: int = 256
    trace_batch_size: int = 8
    seed: int = 0
    strict_byte_identity: bool = True
    include_seed_prefix: bool = True

    def __post_init__(self) -> None:
        _positive_int(self.max_new_tokens, "max_new_tokens")
        context = _positive_int(self.max_context_tokens, "max_context_tokens")
        _positive_int(self.trace_batch_size, "trace_batch_size")
        if context <= self.max_new_tokens:
            raise ValueError("max_context_tokens must exceed max_new_tokens")
        if isinstance(self.seed, bool):
            raise ValueError("seed must be a non-negative integer")
        try:
            seed = operator.index(self.seed)
        except TypeError as exc:
            raise ValueError("seed must be a non-negative integer") from exc
        if seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.strict_byte_identity, bool):
            raise TypeError("strict_byte_identity must be a bool")
        if not isinstance(self.include_seed_prefix, bool):
            raise TypeError("include_seed_prefix must be a bool")

    def generation_contract(self) -> dict[str, Any]:
        return {
            "algorithm": "greedy-argmax",
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": self.max_new_tokens,
            "max_context_tokens": self.max_context_tokens,
            "seed": self.seed,
            "include_seed_prefix": self.include_seed_prefix,
            "decode": {
                "skip_special_tokens": False,
                "clean_up_tokenization_spaces": False,
            },
        }


@dataclass(frozen=True, slots=True)
class ProbeOrigin:
    """One seed/step that produced a deduplicated UTF-8 row."""

    prompt_index: int
    prompt_utf8_sha256: str
    prefix_step: int
    prompt_token_count: int
    generated_token_count: int
    source_child_token_ids_sha256: str


@dataclass(frozen=True, slots=True)
class ProbeRowIdentity:
    """Immutable identity of one exact decoded child prefix."""

    row_id: str
    utf8_sha256: str
    text: str
    utf8_bytes: bytes
    origins: tuple[ProbeOrigin, ...]


@dataclass(frozen=True, slots=True)
class TokenizedProbeRow:
    """One tokenizer's independent representation of a UTF-8 row."""

    row_id: str
    utf8_sha256: str
    token_ids: tuple[int, ...]
    token_ids_sha256: str


@dataclass(frozen=True, slots=True)
class ProbeTraceBatch:
    """Tuple-backed trace input that creates fresh tensors on demand."""

    tokenizer_name: str
    start: int
    row_ids: tuple[str, ...]
    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    sequence_lengths: tuple[int, ...]
    pad_token_id: int
    input_sha256: str

    @property
    def count(self) -> int:
        return len(self.row_ids)

    def as_model_inputs(
        self,
        *,
        device: Any = "cpu",
        torch_module: Any | None = None,
    ) -> Mapping[str, Any]:
        """Materialize detached tensors accepted by the generic trace cache."""

        torch = _torch(torch_module)
        input_ids = torch.tensor(self.input_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(
            self.attention_mask, dtype=torch.long, device=device
        )
        return MappingProxyType(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )

    def as_trace_cache_batch(
        self,
        *,
        device: Any = "cpu",
        torch_module: Any | None = None,
    ) -> Mapping[str, Any]:
        return self.as_model_inputs(device=device, torch_module=torch_module)


@dataclass(frozen=True, slots=True)
class OnPolicyProbeCorpus:
    """Immutable cross-tokenizer rows, trace batches, and audit evidence."""

    rows: tuple[ProbeRowIdentity, ...]
    tokenizations: Mapping[str, tuple[TokenizedProbeRow, ...]]
    trace_batches: Mapping[str, tuple[ProbeTraceBatch, ...]]
    report: Mapping[str, Any]

    def batches_for(self, tokenizer_name: str) -> tuple[ProbeTraceBatch, ...]:
        try:
            return self.trace_batches[tokenizer_name]
        except KeyError as exc:
            raise KeyError(f"unknown tokenizer {tokenizer_name!r}") from exc


def _torch(value: Any | None = None) -> Any:
    if value is not None:
        return value
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("on-policy probes require the 'hf' dependencies") from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(normalized)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OnPolicyProbeError("probe metadata must be canonical JSON") from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OnPolicyProbeIntegrityError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_cache(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise OnPolicyProbeIntegrityError(
            f"generation cache is not a regular file: {path}"
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_strict_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise OnPolicyProbeIntegrityError(
            f"cannot parse generation cache {path}"
        ) from exc
    if not isinstance(value, dict):
        raise OnPolicyProbeIntegrityError("generation cache must contain one object")
    if raw != _canonical_json(value) + b"\n":
        raise OnPolicyProbeIntegrityError(
            "generation cache is noncanonical or was modified"
        )
    supplied = value.get("cache_sha256")
    _validate_sha256(supplied, "cache_sha256")
    core = {key: item for key, item in value.items() if key != "cache_sha256"}
    if _json_sha256(core) != supplied:
        raise OnPolicyProbeIntegrityError(
            "cache_sha256 does not match generation cache contents"
        )
    _validate_cache_core(core)
    return value


def _write_new_cache(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise OnPolicyProbeConflictError(
            f"immutable generation cache already exists: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise OnPolicyProbeError("generation cache parent must be a local directory")
    temporary = path.with_name(f".{path.name}.stage-{uuid.uuid4().hex}")
    payload = _canonical_json(dict(value)) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.rename(temporary, path)
        except FileExistsError as exc:
            raise OnPolicyProbeConflictError(
                f"immutable generation cache already exists: {path}"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _vocabulary_snapshot(tokenizer: Any, name: str) -> dict[str, Any]:
    getter = getattr(tokenizer, "get_vocab", None)
    if not callable(getter):
        raise OnPolicyProbeError(f"{name} tokenizer must expose get_vocab()")
    try:
        raw = getter()
    except Exception as exc:
        raise OnPolicyProbeError(f"cannot read {name} tokenizer vocabulary") from exc
    if not isinstance(raw, Mapping) or not raw:
        raise OnPolicyProbeError(f"{name} tokenizer vocabulary must be nonempty")
    vocabulary: dict[str, int] = {}
    for token, token_id in raw.items():
        if not isinstance(token, str):
            raise OnPolicyProbeError(f"{name} vocabulary keys must be strings")
        if isinstance(token_id, bool):
            raise OnPolicyProbeError(f"{name} vocabulary IDs must be integers")
        try:
            normalized = operator.index(token_id)
        except TypeError as exc:
            raise OnPolicyProbeError(
                f"{name} vocabulary IDs must be integers"
            ) from exc
        if normalized < 0:
            raise OnPolicyProbeError(f"{name} vocabulary IDs cannot be negative")
        vocabulary[token] = int(normalized)
    try:
        tokenizer_length = operator.index(len(tokenizer))
    except (TypeError, ValueError, OverflowError) as exc:
        raise OnPolicyProbeError(f"{name} tokenizer must expose a valid length") from exc
    if tokenizer_length <= 0 or max(vocabulary.values()) >= tokenizer_length:
        raise OnPolicyProbeError(
            f"{name} tokenizer length does not cover its vocabulary IDs"
        )
    core = {
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "length": int(tokenizer_length),
        "vocabulary": vocabulary,
        "special_token_ids": {
            key: _json_token_id(getattr(tokenizer, key, None), f"{name}.{key}")
            for key in ("bos_token_id", "eos_token_id", "pad_token_id")
        },
    }
    core["vocabulary_sha256"] = _json_sha256(
        {"length": core["length"], "vocabulary": vocabulary}
    )
    core["tokenizer_sha256"] = _json_sha256(core)
    return core


def _json_token_id(value: Any, name: str) -> int | list[int] | None:
    if value is None:
        return None
    values = (
        list(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else [value]
    )
    normalized: list[int] = []
    for item in values:
        if isinstance(item, bool):
            raise OnPolicyProbeError(f"{name} must contain integer token IDs")
        try:
            token_id = operator.index(item)
        except TypeError as exc:
            raise OnPolicyProbeError(
                f"{name} must contain integer token IDs"
            ) from exc
        if token_id < 0:
            raise OnPolicyProbeError(f"{name} cannot contain negative IDs")
        normalized.append(int(token_id))
    return normalized[0] if len(normalized) == 1 else normalized


def _tensor_sha256(torch: Any, tensor: Any) -> str:
    if not torch.is_tensor(tensor) or tensor.layout != torch.strided:
        raise OnPolicyProbeError("model state must contain dense tensors")
    if tensor.device.type == "meta":
        raise OnPolicyProbeError("model state tensors must be materialized")
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(b"neural-forge-on-policy-state-tensor-v1\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json([int(size) for size in value.shape]))
    digest.update(b"\0")
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _model_fingerprint(torch: Any, model: Any) -> dict[str, Any]:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise OnPolicyProbeError("child model must expose state_dict()")
    try:
        state = state_dict()
    except Exception as exc:
        raise OnPolicyProbeError("cannot read child model state") from exc
    if not isinstance(state, Mapping):
        raise OnPolicyProbeError("child state_dict() must return a mapping")
    digest = hashlib.sha256()
    digest.update(b"neural-forge-on-policy-child-state-v1\0")
    for name in sorted(state):
        if not isinstance(name, str):
            raise OnPolicyProbeError("child state names must be strings")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_sha256(torch, state[name]).encode("ascii"))
        digest.update(b"\0")
    config_value: Any = None
    config = getattr(model, "config", None)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        try:
            config_value = to_dict()
            _canonical_json(config_value)
        except Exception as exc:
            raise OnPolicyProbeError("cannot serialize child model config") from exc
    core = {
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "state_dict_sha256": digest.hexdigest(),
        "state_tensor_count": len(state),
        "config_sha256": None if config_value is None else _json_sha256(config_value),
    }
    core["model_sha256"] = _json_sha256(core)
    return core


def _parameter_runtime_snapshot(torch: Any, model: Any) -> tuple[Any, ...]:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        return ()
    result = []
    for name, parameter in named_parameters():
        grad = getattr(parameter, "grad", None)
        result.append(
            (
                str(name),
                id(parameter),
                int(getattr(parameter, "_version", -1)),
                None if grad is None else id(grad),
                None if grad is None else _tensor_sha256(torch, grad),
            )
        )
    return tuple(result)


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except StopIteration:
            pass
    return "cpu"


def _encode_text(tokenizer: Any, text: str, name: str) -> tuple[int, ...]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise OnPolicyProbeError(f"{name} tokenizer must expose encode()")
    try:
        try:
            raw = encode(text, add_special_tokens=False)
        except TypeError:
            raw = encode(text)
    except Exception as exc:
        raise OnPolicyProbeError(f"{name} tokenizer encode() failed") from exc
    if hasattr(raw, "ids"):
        raw = raw.ids
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise OnPolicyProbeError(
            f"{name} tokenizer must produce a nonempty token-ID sequence"
        )
    result = []
    for item in raw:
        if isinstance(item, bool):
            raise OnPolicyProbeError(f"{name} tokenizer returned non-integer IDs")
        try:
            token_id = operator.index(item)
        except TypeError as exc:
            raise OnPolicyProbeError(
                f"{name} tokenizer returned non-integer IDs"
            ) from exc
        if token_id < 0:
            raise OnPolicyProbeError(f"{name} tokenizer returned a negative ID")
        result.append(int(token_id))
    _validate_tokenizer_id_range(tokenizer, result, name)
    return tuple(result)


def _decode_ids(tokenizer: Any, token_ids: Sequence[int], name: str) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise OnPolicyProbeError(f"{name} tokenizer must expose decode()")
    _validate_tokenizer_id_range(tokenizer, token_ids, name)
    try:
        try:
            text = decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            text = decode(list(token_ids))
    except Exception as exc:
        raise OnPolicyProbeError(f"{name} tokenizer decode() failed") from exc
    if not isinstance(text, str):
        raise OnPolicyProbeError(f"{name} tokenizer decode() must return text")
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise OnPolicyProbeError(f"{name} tokenizer decoded invalid Unicode") from exc
    return text


def _validate_tokenizer_id_range(
    tokenizer: Any, token_ids: Sequence[int], name: str
) -> None:
    try:
        capacity = operator.index(len(tokenizer))
    except (TypeError, ValueError, OverflowError) as exc:
        raise OnPolicyProbeError(f"{name} tokenizer must expose a valid length") from exc
    for token_id in token_ids:
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            or token_id >= capacity
        ):
            raise OnPolicyProbeIntegrityError(
                f"{name} token ID {token_id!r} is outside tokenizer capacity {capacity}"
            )


def _roundtrip_tokenize(
    tokenizer: Any,
    name: str,
    text: str,
    *,
    strict_byte_identity: bool,
    max_context_tokens: int,
) -> TokenizedProbeRow:
    raw_bytes = text.encode("utf-8")
    utf8_sha256 = _bytes_sha256(raw_bytes)
    token_ids = _encode_text(tokenizer, text, name)
    if len(token_ids) > max_context_tokens:
        raise OnPolicyProbeError(
            f"{name} tokenization length {len(token_ids)} exceeds "
            f"max_context_tokens={max_context_tokens}"
        )
    if strict_byte_identity:
        reconstructed = _decode_ids(tokenizer, token_ids, name).encode("utf-8")
        if reconstructed != raw_bytes:
            raise OnPolicyProbeIntegrityError(
                f"{name} tokenizer loses UTF-8 bytes during decode/re-encode"
            )
    return TokenizedProbeRow(
        row_id=f"utf8-sha256:{utf8_sha256}",
        utf8_sha256=utf8_sha256,
        token_ids=token_ids,
        token_ids_sha256=_json_sha256(list(token_ids)),
    )


def _normalize_prompts(seed_prompts: Sequence[str]) -> tuple[str, ...]:
    if isinstance(seed_prompts, (str, bytes)) or not isinstance(
        seed_prompts, Sequence
    ):
        raise TypeError("seed_prompts must be a sequence of strings")
    prompts = tuple(seed_prompts)
    if not prompts:
        raise ValueError("seed_prompts must not be empty")
    if any(not isinstance(prompt, str) for prompt in prompts):
        raise TypeError("seed_prompts must contain only strings")
    if any(not prompt for prompt in prompts):
        raise ValueError("seed_prompts must not contain empty text")
    for prompt in prompts:
        try:
            prompt.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError("seed_prompts must contain valid Unicode") from exc
    return prompts


def _normalize_parent_tokenizers(
    parent_tokenizers: Mapping[str, Any] | Sequence[Any],
) -> tuple[tuple[str, Any], ...]:
    if isinstance(parent_tokenizers, Mapping):
        values = list(parent_tokenizers.items())
    elif isinstance(parent_tokenizers, Sequence) and not isinstance(
        parent_tokenizers, (str, bytes)
    ):
        values = [(f"parent_{index:03d}", value) for index, value in enumerate(parent_tokenizers)]
    else:
        raise TypeError("parent_tokenizers must be a mapping or sequence")
    if not values:
        raise ValueError("at least one parent tokenizer is required")
    normalized = []
    seen = set()
    for raw_name, tokenizer in values:
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("parent tokenizer names must be nonempty strings")
        name = raw_name.strip()
        if name == "child" or name in seen:
            raise ValueError(f"duplicate or reserved tokenizer name {name!r}")
        seen.add(name)
        normalized.append((name, tokenizer))
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _invoke_generate(model: Any, kwargs: dict[str, Any]) -> Any:
    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise OnPolicyProbeError("child model must expose generate()")
    try:
        signature = inspect.signature(generate)
    except (TypeError, ValueError):
        selected = kwargs
    else:
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        selected = (
            kwargs
            if accepts_kwargs
            else {key: value for key, value in kwargs.items() if key in signature.parameters}
        )
    required = {"input_ids", "attention_mask", "max_new_tokens", "do_sample", "num_beams"}
    missing = sorted(required.difference(selected))
    if missing:
        raise OnPolicyProbeError(
            "child generate() cannot accept the fixed greedy controls: "
            f"{missing!r}"
        )
    return generate(**selected)


def _generated_row(torch: Any, value: Any) -> tuple[int, ...]:
    value = getattr(value, "sequences", value)
    if not torch.is_tensor(value):
        try:
            value = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise OnPolicyProbeError("child generate() output must be tensor-like") from exc
    value = value.detach().cpu()
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or int(value.shape[0]) != 1:
        raise OnPolicyProbeError("child generate() must return exactly one rank-2 row")
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise OnPolicyProbeError("child generate() must return integer token IDs")
    return tuple(int(item) for item in value[0].tolist())


def _generation_kwargs(
    torch: Any,
    token_ids: tuple[int, ...],
    tokenizer: Any,
    settings: GreedyProbeSettings,
    device: Any,
) -> dict[str, Any]:
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }
    for name in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            kwargs[name] = value
    return kwargs


def _generate_unique_prefixes(
    torch: Any,
    model: Any,
    tokenizer: Any,
    prompts: tuple[str, ...],
    settings: GreedyProbeSettings,
    *,
    device: Any,
) -> tuple[list[dict[str, Any]], int]:
    modes = []
    modules = getattr(model, "modules", None)
    if callable(modules):
        modes = [(module, bool(module.training)) for module in modules()]
    elif isinstance(getattr(model, "training", None), bool):
        modes = [(model, bool(model.training))]
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    cuda_devices = []
    resolved = torch.device(device)
    if resolved.type == "cuda":
        cuda_devices = [resolved.index or torch.cuda.current_device()]
    by_hash: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    calls = 0
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.manual_seed(settings.seed)
            evaluate = getattr(model, "eval", None)
            if callable(evaluate):
                evaluate()
            inference_context = getattr(torch, "inference_mode", torch.no_grad)
            with inference_context():
                for prompt_index, prompt in enumerate(prompts):
                    prompt_ids = _encode_text(tokenizer, prompt, "child")
                    if settings.strict_byte_identity:
                        decoded_prompt = _decode_ids(
                            tokenizer, prompt_ids, "child"
                        ).encode("utf-8")
                        if decoded_prompt != prompt.encode("utf-8"):
                            raise OnPolicyProbeIntegrityError(
                                f"child tokenizer loses UTF-8 bytes for seed prompt {prompt_index}"
                            )
                    if len(prompt_ids) + settings.max_new_tokens > settings.max_context_tokens:
                        raise OnPolicyProbeError(
                            f"seed prompt {prompt_index} plus max_new_tokens exceeds "
                            f"max_context_tokens={settings.max_context_tokens}"
                        )
                    generated = _generated_row(
                        torch,
                        _invoke_generate(
                            model,
                            _generation_kwargs(
                                torch, prompt_ids, tokenizer, settings, device
                            ),
                        ),
                    )
                    calls += 1
                    if len(generated) < len(prompt_ids) or generated[: len(prompt_ids)] != prompt_ids:
                        raise OnPolicyProbeIntegrityError(
                            "child generate() did not preserve exact prompt token IDs"
                        )
                    suffix = generated[len(prompt_ids) :]
                    if len(suffix) > settings.max_new_tokens:
                        raise OnPolicyProbeIntegrityError(
                            "child generate() exceeded max_new_tokens"
                        )
                    if len(generated) > settings.max_context_tokens:
                        raise OnPolicyProbeError("generated child prefix exceeds context")
                    first_step = 0 if settings.include_seed_prefix else 1
                    if not suffix and first_step == 1:
                        first_step = 0
                    for step in range(first_step, len(suffix) + 1):
                        source_ids = prompt_ids + suffix[:step]
                        text = _decode_ids(tokenizer, source_ids, "child")
                        raw = text.encode("utf-8")
                        digest = _bytes_sha256(raw)
                        origin = {
                            "prompt_index": prompt_index,
                            "prompt_utf8_sha256": _bytes_sha256(prompt.encode("utf-8")),
                            "prefix_step": step,
                            "prompt_token_count": len(prompt_ids),
                            "generated_token_count": step,
                            "source_child_token_ids_sha256": _json_sha256(list(source_ids)),
                        }
                        existing = by_hash.get(digest)
                        if existing is not None:
                            if bytes.fromhex(existing["utf8_hex"]) != raw:
                                raise OnPolicyProbeIntegrityError("SHA-256 collision in probe rows")
                            existing["origins"].append(origin)
                            continue
                        record = {
                            "row_index": len(ordered),
                            "row_id": f"utf8-sha256:{digest}",
                            "utf8_sha256": digest,
                            "utf8_hex": raw.hex(),
                            "text": text,
                            "source_child_token_ids": list(source_ids),
                            "source_child_token_ids_sha256": _json_sha256(list(source_ids)),
                            "origins": [origin],
                        }
                        by_hash[digest] = record
                        ordered.append(record)
    finally:
        torch.use_deterministic_algorithms(
            deterministic_before, warn_only=warn_only_before
        )
        for module, was_training in modes:
            module.training = was_training
    if not ordered:
        raise OnPolicyProbeError("generation produced no probe prefixes")
    return ordered, calls


def _validate_cache_core(core: Mapping[str, Any]) -> None:
    if core.get("schema") != GENERATION_CACHE_SCHEMA:
        raise OnPolicyProbeIntegrityError("unsupported generation cache schema")
    _validate_sha256(core.get("request_sha256"), "request_sha256")
    request = core.get("request")
    if not isinstance(request, Mapping) or _json_sha256(request) != core["request_sha256"]:
        raise OnPolicyProbeIntegrityError("generation cache request hash is invalid")
    records = core.get("records")
    if not isinstance(records, list) or not records:
        raise OnPolicyProbeIntegrityError("generation cache records must be nonempty")
    if core.get("record_count") != len(records):
        raise OnPolicyProbeIntegrityError("generation cache record_count is invalid")
    row_ids = set()
    utf8_hashes = set()
    origin_keys = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("row_index") != index:
            raise OnPolicyProbeIntegrityError("generation cache row order is invalid")
        digest = _validate_sha256(record.get("utf8_sha256"), "row utf8_sha256")
        row_id = record.get("row_id")
        if row_id != f"utf8-sha256:{digest}" or row_id in row_ids or digest in utf8_hashes:
            raise OnPolicyProbeIntegrityError("generation cache contains duplicate rows")
        row_ids.add(row_id)
        utf8_hashes.add(digest)
        try:
            raw = bytes.fromhex(record.get("utf8_hex"))
        except (TypeError, ValueError) as exc:
            raise OnPolicyProbeIntegrityError("cached utf8_hex is invalid") from exc
        text = record.get("text")
        if not isinstance(text, str) or text.encode("utf-8") != raw or _bytes_sha256(raw) != digest:
            raise OnPolicyProbeIntegrityError("cached UTF-8 row identity is invalid")
        token_ids = _cached_token_ids(record.get("source_child_token_ids"))
        if _json_sha256(list(token_ids)) != record.get("source_child_token_ids_sha256"):
            raise OnPolicyProbeIntegrityError("cached source token hash is invalid")
        origins = record.get("origins")
        if not isinstance(origins, list) or not origins:
            raise OnPolicyProbeIntegrityError("cached row origins must be nonempty")
        local_origins = set()
        for origin in origins:
            if not isinstance(origin, Mapping):
                raise OnPolicyProbeIntegrityError("cached row origin is invalid")
            prompt_index = origin.get("prompt_index")
            prefix_step = origin.get("prefix_step")
            key = (prompt_index, prefix_step)
            if key in local_origins or key in origin_keys:
                raise OnPolicyProbeIntegrityError("generation cache has duplicate origins")
            local_origins.add(key)
            origin_keys.add(key)
            if (
                not isinstance(prompt_index, int)
                or isinstance(prompt_index, bool)
                or prompt_index < 0
                or not isinstance(prefix_step, int)
                or isinstance(prefix_step, bool)
                or prefix_step < 0
            ):
                raise OnPolicyProbeIntegrityError("cached origin indices are invalid")
            _validate_sha256(origin.get("prompt_utf8_sha256"), "prompt_utf8_sha256")
            _validate_sha256(
                origin.get("source_child_token_ids_sha256"),
                "origin source_child_token_ids_sha256",
            )
    if _json_sha256(records) != core.get("records_sha256"):
        raise OnPolicyProbeIntegrityError("generation cache records hash is invalid")


def _cached_token_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise OnPolicyProbeIntegrityError("cached token IDs must be nonempty")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise OnPolicyProbeIntegrityError("cached token IDs are invalid")
        result.append(item)
    return tuple(result)


def _rows_from_records(
    records: Sequence[Mapping[str, Any]], child_tokenizer: Any
) -> tuple[ProbeRowIdentity, ...]:
    rows = []
    for record in records:
        source_ids = _cached_token_ids(record["source_child_token_ids"])
        decoded = _decode_ids(child_tokenizer, source_ids, "child")
        if decoded != record["text"]:
            raise OnPolicyProbeConflictError(
                "child tokenizer decode behavior differs from the generation cache"
            )
        origins = tuple(
            ProbeOrigin(
                prompt_index=int(origin["prompt_index"]),
                prompt_utf8_sha256=str(origin["prompt_utf8_sha256"]),
                prefix_step=int(origin["prefix_step"]),
                prompt_token_count=int(origin["prompt_token_count"]),
                generated_token_count=int(origin["generated_token_count"]),
                source_child_token_ids_sha256=str(
                    origin["source_child_token_ids_sha256"]
                ),
            )
            for origin in record["origins"]
        )
        rows.append(
            ProbeRowIdentity(
                row_id=str(record["row_id"]),
                utf8_sha256=str(record["utf8_sha256"]),
                text=str(record["text"]),
                utf8_bytes=bytes.fromhex(str(record["utf8_hex"])),
                origins=origins,
            )
        )
    return tuple(rows)


def _pad_token_id(tokenizer: Any, vocabulary: Mapping[str, Any]) -> int:
    values = vocabulary["vocabulary"]
    valid_ids = set(values.values())
    for name in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value in valid_ids:
            return int(value)
    return min(valid_ids)


def _build_batches(
    name: str,
    tokenizer: Any,
    vocabulary: Mapping[str, Any],
    tokenizations: tuple[TokenizedProbeRow, ...],
    batch_size: int,
) -> tuple[ProbeTraceBatch, ...]:
    pad_id = _pad_token_id(tokenizer, vocabulary)
    result = []
    for start in range(0, len(tokenizations), batch_size):
        chunk = tokenizations[start : start + batch_size]
        width = max(len(row.token_ids) for row in chunk)
        ids = []
        masks = []
        lengths = []
        for row in chunk:
            length = len(row.token_ids)
            padding = width - length
            ids.append(row.token_ids + (pad_id,) * padding)
            masks.append((1,) * length + (0,) * padding)
            lengths.append(length)
        core = {
            "tokenizer_name": name,
            "start": start,
            "row_ids": [row.row_id for row in chunk],
            "input_ids": [list(row) for row in ids],
            "attention_mask": [list(row) for row in masks],
        }
        result.append(
            ProbeTraceBatch(
                tokenizer_name=name,
                start=start,
                row_ids=tuple(row.row_id for row in chunk),
                input_ids=tuple(ids),
                attention_mask=tuple(masks),
                sequence_lengths=tuple(lengths),
                pad_token_id=pad_id,
                input_sha256=_json_sha256(core),
            )
        )
    return tuple(result)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def load_on_policy_generation_cache(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    """Load and verify an immutable cached child-generation document."""

    return _deep_freeze(_read_cache(Path(path).expanduser().absolute()))


def build_on_policy_probe_corpus(
    child_model: Any,
    child_tokenizer: Any,
    parent_tokenizers: Mapping[str, Any] | Sequence[Any],
    seed_prompts: Sequence[str],
    *,
    settings: GreedyProbeSettings | None = None,
    generation_cache_path: str | os.PathLike[str] | None = None,
    device: Any | None = None,
    torch_module: Any | None = None,
) -> OnPolicyProbeCorpus:
    """Generate or reuse child prefixes and align them by exact UTF-8 identity."""

    torch = _torch(torch_module)
    selected = settings or GreedyProbeSettings()
    if not isinstance(selected, GreedyProbeSettings):
        raise TypeError("settings must be GreedyProbeSettings or None")
    prompts = _normalize_prompts(seed_prompts)
    parents = _normalize_parent_tokenizers(parent_tokenizers)
    tokenizers = (("child", child_tokenizer),) + parents

    vocabularies_before = {
        name: _vocabulary_snapshot(tokenizer, name)
        for name, tokenizer in tokenizers
    }
    model_before = _model_fingerprint(torch, child_model)
    runtime_before = _parameter_runtime_snapshot(torch, child_model)
    request = {
        "schema": "neural-forge-on-policy-generation-request-v1",
        "child_model": model_before,
        "child_tokenizer": vocabularies_before["child"],
        "seed_prompts": [
            {
                "index": index,
                "text": prompt,
                "utf8_sha256": _bytes_sha256(prompt.encode("utf-8")),
            }
            for index, prompt in enumerate(prompts)
        ],
        "generation": selected.generation_contract(),
    }
    request_sha256 = _json_sha256(request)
    cache_path = (
        None
        if generation_cache_path is None
        else Path(generation_cache_path).expanduser().absolute()
    )

    cache_hit = False
    generation_calls = 0
    if cache_path is not None and cache_path.exists():
        cache = _read_cache(cache_path)
        if cache["request_sha256"] != request_sha256 or cache["request"] != request:
            raise OnPolicyProbeConflictError(
                "generation cache belongs to a different child, tokenizer, prompts, or settings"
            )
        records = cache["records"]
        cache_hit = True
    else:
        records, generation_calls = _generate_unique_prefixes(
            torch,
            child_model,
            child_tokenizer,
            prompts,
            selected,
            device=_model_device(child_model) if device is None else device,
        )
        model_after_generation = _model_fingerprint(torch, child_model)
        runtime_after_generation = _parameter_runtime_snapshot(torch, child_model)
        if model_after_generation != model_before or runtime_after_generation != runtime_before:
            raise OnPolicyProbeIntegrityError(
                "child model parameters or gradients changed during generation"
            )
        child_vocab_after_generation = _vocabulary_snapshot(child_tokenizer, "child")
        if child_vocab_after_generation != vocabularies_before["child"]:
            raise OnPolicyProbeIntegrityError(
                "child tokenizer vocabulary changed during generation"
            )
        cache_core = {
            "schema": GENERATION_CACHE_SCHEMA,
            "request": request,
            "request_sha256": request_sha256,
            "record_count": len(records),
            "records": records,
            "records_sha256": _json_sha256(records),
        }
        cache = {**cache_core, "cache_sha256": _json_sha256(cache_core)}
        _validate_cache_core(cache_core)
        if cache_path is not None:
            _write_new_cache(cache_path, cache)

    _validate_cache_core(
        {key: item for key, item in cache.items() if key != "cache_sha256"}
    )
    rows = _rows_from_records(records, child_tokenizer)

    tokenization_map: dict[str, tuple[TokenizedProbeRow, ...]] = {}
    batch_map: dict[str, tuple[ProbeTraceBatch, ...]] = {}
    for name, tokenizer in tokenizers:
        tokenized = tuple(
            _roundtrip_tokenize(
                tokenizer,
                name,
                row.text,
                strict_byte_identity=selected.strict_byte_identity,
                max_context_tokens=selected.max_context_tokens,
            )
            for row in rows
        )
        if tuple(item.row_id for item in tokenized) != tuple(row.row_id for row in rows):
            raise OnPolicyProbeIntegrityError("cross-tokenizer UTF-8 row alignment failed")
        tokenization_map[name] = tokenized
        batch_map[name] = _build_batches(
            name,
            tokenizer,
            vocabularies_before[name],
            tokenized,
            selected.trace_batch_size,
        )

    vocabularies_after = {
        name: _vocabulary_snapshot(tokenizer, name)
        for name, tokenizer in tokenizers
    }
    if vocabularies_after != vocabularies_before:
        raise OnPolicyProbeIntegrityError(
            "a tokenizer vocabulary changed during probe construction"
        )
    model_after = _model_fingerprint(torch, child_model)
    runtime_after = _parameter_runtime_snapshot(torch, child_model)
    if model_after != model_before or runtime_after != runtime_before:
        raise OnPolicyProbeIntegrityError(
            "child model parameters or gradients changed during probe construction"
        )

    corpus_core = {
        "rows": [
            {
                "row_id": row.row_id,
                "utf8_sha256": row.utf8_sha256,
                "origin_count": len(row.origins),
            }
            for row in rows
        ],
        "tokenizations": {
            name: [
                {
                    "row_id": row.row_id,
                    "token_ids_sha256": row.token_ids_sha256,
                    "token_count": len(row.token_ids),
                }
                for row in values
            ]
            for name, values in tokenization_map.items()
        },
    }
    report: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "corpus_sha256": _json_sha256(corpus_core),
        "request_sha256": request_sha256,
        "generation_cache": {
            "path": None if cache_path is None else str(cache_path),
            "hit": cache_hit,
            "cache_sha256": cache["cache_sha256"],
            "generation_calls": generation_calls,
            "record_count": len(rows),
            "deduplicated": True,
        },
        "execution": {
            "greedy": True,
            "do_sample": False,
            "deterministic_algorithms": True,
            "inference_mode": True,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "parent_models_supplied": False,
            "parent_forward_calls": 0,
            "child_generate_calls": generation_calls,
        },
        "immutability": {
            "child_model_before_sha256": model_before["model_sha256"],
            "child_model_after_sha256": model_after["model_sha256"],
            "child_parameters_unchanged": True,
            "child_gradients_unchanged": True,
            "tokenizer_vocabularies_unchanged": True,
            "tokenizer_vocabulary_before_sha256": {
                name: value["vocabulary_sha256"]
                for name, value in vocabularies_before.items()
            },
            "tokenizer_vocabulary_after_sha256": {
                name: value["vocabulary_sha256"]
                for name, value in vocabularies_after.items()
            },
        },
        "alignment": {
            "identity": "exact-decoded-utf8-sha256",
            "strict_byte_identity": selected.strict_byte_identity,
            "tokenizers": [name for name, _ in tokenizers],
            "row_count": len(rows),
            "origin_count": sum(len(row.origins) for row in rows),
        },
        "settings": asdict(selected),
        "corpus": corpus_core,
    }
    report["report_sha256"] = _json_sha256(report)
    return OnPolicyProbeCorpus(
        rows=rows,
        tokenizations=MappingProxyType(dict(tokenization_map)),
        trace_batches=MappingProxyType(dict(batch_map)),
        report=_deep_freeze(report),
    )


generate_on_policy_probe_corpus = build_on_policy_probe_corpus
build_on_policy_probes = build_on_policy_probe_corpus


__all__ = [
    "GENERATION_CACHE_SCHEMA",
    "PROBE_SCHEMA",
    "GreedyProbeSettings",
    "OnPolicyProbeConflictError",
    "OnPolicyProbeCorpus",
    "OnPolicyProbeError",
    "OnPolicyProbeIntegrityError",
    "ProbeOrigin",
    "ProbeRowIdentity",
    "ProbeTraceBatch",
    "TokenizedProbeRow",
    "build_on_policy_probe_corpus",
    "build_on_policy_probes",
    "generate_on_policy_probe_corpus",
    "load_on_policy_generation_cache",
]
