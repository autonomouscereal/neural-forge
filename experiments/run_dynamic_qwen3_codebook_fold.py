"""Run the automatic Qwen3 dense-to-codebook fold from local artifacts only.

The dense parent, both tokenizers, cached parent behavior, and the standalone
codebook evidence checkpoint must already exist on local storage. The runner
never trains and never performs a parent forward pass.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from neural_forge.dynamic_behavior_gate import (
    CachedWholeOutputBehaviorGate,
    CandidateWholeOutput,
    TaskScoreCoveragePolicy,
    WholeOutputCase,
    WholeOutputGateConfig,
    WholeOutputMetadata,
    identify_behavior_callable,
)
from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.dynamic_fold_compiler import DynamicFoldSystemStack
from neural_forge.dynamic_fold_executor import FoldDecision, fingerprint_model_state
from neural_forge.dynamic_fold_snapshot import (
    DynamicFoldSnapshotConfig,
    snapshot_directory_inventory,
)
from neural_forge.dynamic_qwen_adapter import (
    Qwen3CodebookStackFactory,
    Qwen3CodebookTargetSynthesisAdapter,
    build_qwen3_codebook_tokenizer_contract,
    build_qwen3_global_codebook_fold_cache_from_checkpoint,
)
from neural_forge.dynamic_system import (
    AutomaticDynamicFoldingSystem,
    DynamicFoldStackAdapterRegistry,
)
from neural_forge.dynamic_target_synthesis import (
    AdapterMaterialization,
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
    DynamicTargetSynthesisResult,
)
from neural_forge.hf_qwen_codebook_model import (
    load_qwen3_codebook_checkpoint,
    qwen3_codebook_parameter_report,
)
from neural_forge.hf_tied_depth import exact_token_row_mapping


SCHEMA = "neural-forge-dynamic-qwen3-codebook-run-v1"
DEFAULT_TARGET_PARAMETERS = 3_995_570_176
DEFAULT_ATTENTION_PROTOTYPES = 35
DEFAULT_NEURON_PROTOTYPES = 36
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_DEFAULT_DTYPE_LOCK = threading.RLock()


class DynamicQwen3RunnerError(RuntimeError):
    """A local input, runtime invariant, or publication contract failed."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicQwen3RunnerError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DynamicQwen3RunnerError(f"{name} must be a non-negative integer.")
    return value


def _bounded_fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicQwen3RunnerError(f"{name} must be numeric.")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise DynamicQwen3RunnerError(f"{name} must be between zero and one.")
    return result


def _sha256(value: str) -> str:
    normalized = str(value).strip().lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise argparse.ArgumentTypeError(
            "value must be one lowercase 64-character SHA-256 digest"
        )
    return normalized


def _optional_expected_digest(
    actual: str,
    expected: str | None,
    name: str,
) -> None:
    if expected is not None and actual != expected:
        raise DynamicQwen3RunnerError(
            f"{name} fingerprint differs from the expected SHA-256 digest."
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _resolve_local_directory(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise DynamicQwen3RunnerError(
            f"{name} must be an explicit local directory, not a URL."
        )
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise DynamicQwen3RunnerError(f"{name} directory is missing: {path}.")
    return path.resolve(strict=True)


def _resolve_local_file(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise DynamicQwen3RunnerError(
            f"{name} must be an explicit local file, not a URL."
        )
    path = Path(raw).expanduser()
    if not path.is_file():
        raise DynamicQwen3RunnerError(f"{name} file is missing: {path}.")
    return path.resolve(strict=True)


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DynamicQwen3RunnerError(f"Could not read {name}: {path}.") from exc
    if not isinstance(value, Mapping):
        raise DynamicQwen3RunnerError(f"{name} must contain one JSON object.")
    return dict(value)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().absolute()
    if destination.exists():
        raise DynamicQwen3RunnerError(
            f"Run receipt destination already exists: {destination}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DynamicQwen3RunnerError(
            "Run receipt is not finite canonical JSON."
        ) from exc
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _torch_dtype(name: str) -> torch.dtype:
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[str(name).strip().lower()]
    except KeyError as exc:
        raise DynamicQwen3RunnerError(
            "dtype must be bfloat16, float16, or float32."
        ) from exc


def _model_device(model: nn.Module) -> torch.device:
    parameters = tuple(model.parameters())
    if not parameters:
        raise DynamicQwen3RunnerError("Model has no parameters.")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise DynamicQwen3RunnerError(
            "The production runner requires each model on one device."
        )
    return next(iter(devices))


def _storage_identity(value: Tensor) -> tuple[int, int, tuple[int, ...]]:
    return (
        int(value.untyped_storage().data_ptr()),
        int(value.storage_offset()),
        tuple(int(size) for size in value.shape),
    )


def _tensor_equal_across_devices(
    target: Tensor,
    source: Tensor,
    *,
    chunk_bytes: int = 32 * 1024**2,
) -> bool:
    if tuple(target.shape) != tuple(source.shape):
        return False
    target_flat = target.detach().reshape(-1)
    source_flat = source.detach().reshape(-1)
    elements = max(1, chunk_bytes // max(1, int(target.element_size())))
    for start in range(0, int(target_flat.numel()), elements):
        stop = min(int(target_flat.numel()), start + elements)
        expected = source_flat[start:stop].to(
            device="cpu",
            dtype=target.dtype,
        )
        observed = target_flat[start:stop].to(device="cpu")
        if not torch.equal(observed, expected):
            return False
    return True


@contextmanager
def _temporary_default_dtype(dtype: torch.dtype):
    with _DEFAULT_DTYPE_LOCK:
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(previous)


@dataclass(frozen=True, slots=True)
class LocalSafetensorLayout:
    shards: tuple[Path, ...]
    weight_map: Mapping[str, str] | None


def _local_safetensor_layout(directory: Path) -> LocalSafetensorLayout:
    index_path = directory / "model.safetensors.index.json"
    weight_map: dict[str, str] | None = None
    if index_path.is_file():
        index = _read_json_object(index_path, "safetensors index")
        raw_map = index.get("weight_map")
        if not isinstance(raw_map, Mapping) or not raw_map:
            raise DynamicQwen3RunnerError("Local safetensors index has no weight_map.")
        weight_map = {}
        for raw_name, raw_filename in raw_map.items():
            name = str(raw_name)
            filename = str(raw_filename)
            relative = Path(filename)
            if (
                not name
                or not filename
                or relative.is_absolute()
                or relative.name != filename
            ):
                raise DynamicQwen3RunnerError(
                    "Safetensors index contains an unsafe key or shard path."
                )
            weight_map[name] = filename
        shards = tuple(
            directory / filename for filename in sorted(set(weight_map.values()))
        )
    else:
        single = directory / "model.safetensors"
        shards = (
            (single,)
            if single.is_file()
            else tuple(sorted(directory.glob("model-*.safetensors")))
        )
    if not shards or any(not shard.is_file() for shard in shards):
        raise DynamicQwen3RunnerError(
            f"Local Qwen3 safetensors are incomplete in {directory}."
        )
    return LocalSafetensorLayout(shards=shards, weight_map=weight_map)


def load_local_qwen3_config(model_directory: str | os.PathLike[str]) -> Any:
    """Construct concrete Qwen3Config directly from local config.json."""

    from transformers import Qwen3Config

    directory = _resolve_local_directory(model_directory, "parent model")
    payload = _read_json_object(directory / "config.json", "Qwen3 config")
    if payload.get("model_type") != "qwen3":
        raise DynamicQwen3RunnerError("Local config.json is not a Qwen3 configuration.")
    return Qwen3Config(**payload)


def _rebuild_qwen3_runtime_buffers(model: nn.Module, device: torch.device) -> None:
    backbone = getattr(model, "model", None)
    rotary = getattr(backbone, "rotary_emb", None)
    config = getattr(model, "config", None)
    if rotary is None or config is None:
        raise DynamicQwen3RunnerError(
            "Concrete Qwen3 runtime lacks its rotary embedding."
        )
    try:
        replacement = type(rotary)(config, device=device)
    except Exception as exc:
        raise DynamicQwen3RunnerError(
            "Could not reconstruct Qwen3 non-persistent rotary buffers."
        ) from exc
    backbone.rotary_emb = replacement


def load_local_qwen3_model(
    model_directory: str | os.PathLike[str],
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Stream local safetensors into concrete Qwen3ForCausalLM."""

    from safetensors import safe_open
    from transformers import GenerationConfig, Qwen3ForCausalLM

    directory = _resolve_local_directory(model_directory, "parent model")
    config = load_local_qwen3_config(directory)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise DynamicQwen3RunnerError(
            "A CUDA parent device was requested without CUDA."
        )
    with _temporary_default_dtype(dtype), torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    model.to_empty(device=target_device)
    model.tie_weights()
    _rebuild_qwen3_runtime_buffers(model, target_device)
    target_state = model.state_dict()
    layout = _local_safetensor_layout(directory)
    loaded: set[str] = set()
    unexpected: list[str] = []
    loaded_storage: dict[tuple[int, int, tuple[int, ...]], str] = {}
    with torch.no_grad():
        for shard in layout.shards:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in loaded:
                        raise DynamicQwen3RunnerError(
                            f"Duplicate tensor across local shards: {name}."
                        )
                    if (
                        layout.weight_map is not None
                        and layout.weight_map.get(name) != shard.name
                    ):
                        raise DynamicQwen3RunnerError(
                            f"Safetensors index maps {name!r} to the wrong shard."
                        )
                    if name not in target_state:
                        unexpected.append(name)
                        continue
                    target = target_state[name]
                    identity = _storage_identity(target)
                    if identity in loaded_storage:
                        source = handle.get_tensor(name)
                        if not _tensor_equal_across_devices(target, source):
                            raise DynamicQwen3RunnerError(
                                "Serialized aliases disagree for tied Qwen3 storage: "
                                f"{loaded_storage[identity]!r} and {name!r}."
                            )
                        loaded.add(name)
                        continue
                    source = handle.get_tensor(name)
                    if tuple(source.shape) != tuple(target.shape):
                        raise DynamicQwen3RunnerError(
                            f"Local tensor shape mismatch for {name}: "
                            f"{tuple(source.shape)} != {tuple(target.shape)}."
                        )
                    target.copy_(
                        source.to(device=target.device, dtype=target.dtype),
                    )
                    loaded.add(name)
                    loaded_storage[identity] = name
    if unexpected:
        raise DynamicQwen3RunnerError(
            f"Local safetensors contain unexpected Qwen3 keys: {unexpected[:8]}."
        )
    if layout.weight_map is not None and set(layout.weight_map) != loaded:
        raise DynamicQwen3RunnerError(
            "Safetensors index key inventory differs from the loaded shard keys."
        )
    missing = sorted(
        name
        for name, value in target_state.items()
        if name not in loaded and _storage_identity(value) not in loaded_storage
    )
    if missing:
        raise DynamicQwen3RunnerError(
            f"Local safetensors are missing non-aliased Qwen3 keys: {missing[:8]}."
        )
    model.tie_weights()
    generation_path = directory / "generation_config.json"
    if generation_path.is_file():
        model.generation_config = GenerationConfig(
            **_read_json_object(generation_path, "generation config")
        )
    meta_parameters = [
        name for name, value in model.named_parameters() if value.device.type == "meta"
    ]
    meta_buffers = [
        name for name, value in model.named_buffers() if value.device.type == "meta"
    ]
    if meta_parameters or meta_buffers:
        raise DynamicQwen3RunnerError(
            "Direct Qwen3 loading left meta state: "
            f"parameters={meta_parameters[:4]}, buffers={meta_buffers[:4]}."
        )
    model.eval()
    return model


def _special_token_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "content" in value:
        return str(value["content"])
    return value


def load_local_tokenizer(model_directory: str | os.PathLike[str]) -> Any:
    """Construct a fast tokenizer directly from local tokenizer.json."""

    from transformers import PreTrainedTokenizerFast

    directory = _resolve_local_directory(model_directory, "tokenizer")
    tokenizer_path = directory / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise DynamicQwen3RunnerError(
            f"Local tokenizer.json is missing: {tokenizer_path}."
        )
    kwargs: dict[str, Any] = {}
    config_path = directory / "tokenizer_config.json"
    if config_path.is_file():
        kwargs.update(_read_json_object(config_path, "tokenizer config"))
    special_path = directory / "special_tokens_map.json"
    if special_path.is_file():
        kwargs.update(_read_json_object(special_path, "special-token map"))
    for name in (
        "auto_map",
        "backend",
        "is_local",
        "sp_model_kwargs",
        "tokenizer_class",
        "tokenizer_file",
        "use_fast",
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


@dataclass(frozen=True, slots=True)
class LocalTokenizerIdentity:
    path: str
    fingerprint: str
    vocab_size: int
    vocab_sha256: str
    special_token_ids: Mapping[str, int | None]
    file_sha256: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "fingerprint": self.fingerprint,
            "vocab_size": self.vocab_size,
            "vocab_sha256": self.vocab_sha256,
            "special_token_ids": dict(self.special_token_ids),
            "file_sha256": dict(self.file_sha256),
        }


def fingerprint_local_tokenizer(
    model_directory: str | os.PathLike[str],
    tokenizer: Any,
) -> LocalTokenizerIdentity:
    directory = _resolve_local_directory(model_directory, "tokenizer")
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise DynamicQwen3RunnerError("Tokenizer must expose get_vocab().")
    vocabulary = get_vocab()
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise DynamicQwen3RunnerError("Tokenizer vocabulary is empty or malformed.")
    normalized_vocab: dict[str, int] = {}
    for token, raw_index in vocabulary.items():
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or raw_index < 0
        ):
            raise DynamicQwen3RunnerError("Tokenizer vocabulary IDs are invalid.")
        normalized_vocab[str(token)] = raw_index
    if len(set(normalized_vocab.values())) != len(normalized_vocab):
        raise DynamicQwen3RunnerError("Tokenizer vocabulary IDs are not unique.")
    filenames = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    )
    file_hashes = {
        name: _file_sha256(directory / name)
        for name in filenames
        if (directory / name).is_file()
    }
    if "tokenizer.json" not in file_hashes:
        raise DynamicQwen3RunnerError("Tokenizer identity lacks tokenizer.json.")
    vocab_sha256 = hashlib.sha256(canonical_json_bytes(normalized_vocab)).hexdigest()
    special_ids = {
        name: getattr(tokenizer, name, None)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id")
    }
    body = {
        "schema": "neural-forge-local-tokenizer-identity-v1",
        "vocab_size": len(normalized_vocab),
        "vocab_sha256": vocab_sha256,
        "special_token_ids": special_ids,
        "file_sha256": file_hashes,
    }
    return LocalTokenizerIdentity(
        path=str(directory),
        fingerprint=canonical_sha256(body),
        vocab_size=len(normalized_vocab),
        vocab_sha256=vocab_sha256,
        special_token_ids=special_ids,
        file_sha256=file_hashes,
    )


def _surface_repeated_bigram_share(text: str) -> float:
    tokens = re.findall(r"[A-Za-z0-9_]+", str(text).lower())
    bigrams = list(zip(tokens, tokens[1:]))
    if not bigrams:
        return 0.0
    repeated = len(bigrams) - len(set(bigrams))
    return repeated / len(bigrams)


def _fragment_like(text: str, minimum_characters: int) -> bool:
    return len(str(text).strip()) < _positive_int(
        minimum_characters,
        "fragment minimum characters",
    )


def _normalized_rules(
    value: Any,
    case_id: str,
) -> tuple[tuple[str, ...], ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DynamicQwen3RunnerError(
            f"Cached relevance rules for {case_id!r} are malformed."
        )
    result = []
    for group in value:
        if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
            raise DynamicQwen3RunnerError(
                f"Cached relevance rule group for {case_id!r} is malformed."
            )
        patterns = tuple(str(pattern) for pattern in group if str(pattern))
        if not patterns:
            raise DynamicQwen3RunnerError(
                f"Cached relevance rule group for {case_id!r} is empty."
            )
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise DynamicQwen3RunnerError(
                    f"Cached relevance rule for {case_id!r} is invalid."
                ) from exc
        result.append(patterns)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CachedGenerationBaseline:
    source_path: str
    source_sha256: str
    cases: tuple[WholeOutputCase, ...]
    relevance_rules: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...]
    selected_strata: tuple[tuple[str, bool], ...]
    subset_seed: int
    generation_limit: int
    fragment_minimum_characters: int

    @property
    def rules(self) -> dict[str, tuple[tuple[str, ...], ...]]:
        return dict(self.relevance_rules)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": "neural-forge-cached-generation-baseline-v1",
                "source_sha256": self.source_sha256,
                "cases": [case.as_dict() for case in self.cases],
                "relevance_rules": {
                    name: [list(group) for group in groups]
                    for name, groups in self.relevance_rules
                },
                "selected_strata": [
                    [category, parent_pass]
                    for category, parent_pass in self.selected_strata
                ],
                "subset_seed": self.subset_seed,
                "generation_limit": self.generation_limit,
                "fragment_minimum_characters": (self.fragment_minimum_characters),
            }
        )

    def task_score(self, case: WholeOutputCase, output: str) -> bool | None:
        groups = self.rules.get(case.case_id, ())
        if not groups:
            return None
        text = str(output).lower()
        return all(
            any(re.search(pattern, text) is not None for pattern in group)
            for group in groups
        )

    def as_dict(self) -> dict[str, Any]:
        categories: dict[str, int] = {}
        for case in self.cases:
            categories[case.category] = categories.get(case.category, 0) + 1
        return {
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "fingerprint": self.fingerprint,
            "case_count": len(self.cases),
            "case_ids": [case.case_id for case in self.cases],
            "category_counts": categories,
            "task_scored_case_count": sum(
                bool(groups) for _name, groups in self.relevance_rules
            ),
            "selected_strata": [
                {"category": category, "parent_pass": parent_pass}
                for category, parent_pass in self.selected_strata
            ],
            "subset_seed": self.subset_seed,
            "generation_limit": self.generation_limit,
            "fragment_minimum_characters": self.fragment_minimum_characters,
            "parent_forward_passes": 0,
        }


def _stable_row_order(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    stratum: tuple[str, bool],
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(
                f"{seed}\0{stratum[0]}\0{int(stratum[1])}\0{row['id']}".encode("utf-8")
            ).hexdigest(),
            str(row["id"]),
        ),
    )


def load_cached_generation_baseline(
    path: str | os.PathLike[str],
    *,
    subset_size: int,
    subset_seed: int,
    generation_limit: int,
    fragment_minimum_characters: int,
    expected_sha256: str | None = None,
) -> CachedGenerationBaseline:
    """Load a deterministic category/pass-stratified parent generation subset."""

    source = _resolve_local_file(path, "cached parent evaluation")
    source_sha256 = _file_sha256(source)
    _optional_expected_digest(
        source_sha256,
        expected_sha256,
        "cached parent evaluation",
    )
    payload = _read_json_object(source, "cached parent evaluation")
    generations = payload.get("generations")
    diagnostics = payload.get("generation_diagnostics")
    if (
        not isinstance(generations, list)
        or not generations
        or not isinstance(diagnostics, list)
        or not diagnostics
    ):
        raise DynamicQwen3RunnerError(
            "Cached parent evaluation lacks generations and diagnostics."
        )
    diagnostic_map: dict[str, dict[str, Any]] = {}
    for raw in diagnostics:
        if not isinstance(raw, Mapping) or not str(raw.get("id", "")).strip():
            raise DynamicQwen3RunnerError("Cached generation diagnostic is malformed.")
        case_id = str(raw["id"]).strip()
        if case_id in diagnostic_map:
            raise DynamicQwen3RunnerError(
                f"Cached generation diagnostic ID repeats: {case_id}."
            )
        diagnostic_map[case_id] = dict(raw)
    enriched: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in generations:
        if not isinstance(raw, Mapping):
            raise DynamicQwen3RunnerError("Cached generation row is malformed.")
        case_id = str(raw.get("id", "")).strip()
        prompt = raw.get("prompt")
        output = raw.get("output")
        category = str(raw.get("category", "unspecified")).strip()
        if (
            not case_id
            or case_id in seen
            or not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(output, str)
            or not category
            or case_id not in diagnostic_map
        ):
            raise DynamicQwen3RunnerError(
                "Cached generation rows require unique IDs, prompts, outputs, "
                "categories, and matching diagnostics."
            )
        seen.add(case_id)
        row = dict(raw)
        row["id"] = case_id
        row["category"] = category
        row["diagnostic"] = diagnostic_map[case_id]
        enriched.append(row)
    if set(diagnostic_map) != seen:
        raise DynamicQwen3RunnerError(
            "Cached generation and diagnostic ID inventories differ."
        )
    size = _positive_int(subset_size, "subset_size")
    if size > len(enriched):
        raise DynamicQwen3RunnerError(
            "Requested generation subset exceeds the cached evaluation."
        )
    seed = _nonnegative_int(subset_seed, "subset_seed")
    limit = _positive_int(generation_limit, "generation_limit")
    fragment_chars = _positive_int(
        fragment_minimum_characters,
        "fragment_minimum_characters",
    )
    strata: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for row in enriched:
        key = (
            str(row["category"]),
            bool(row["diagnostic"].get("passed", False)),
        )
        strata.setdefault(key, []).append(row)
    if size < len(strata):
        raise DynamicQwen3RunnerError(
            "Subset size must cover every cached category/pass stratum."
        )
    ordered = {
        key: _stable_row_order(rows, seed=seed, stratum=key)
        for key, rows in strata.items()
    }
    selected: list[dict[str, Any]] = []
    offsets = {key: 0 for key in ordered}
    keys = sorted(ordered)
    while len(selected) < size:
        progressed = False
        for key in keys:
            offset = offsets[key]
            rows = ordered[key]
            if offset >= len(rows):
                continue
            selected.append(rows[offset])
            offsets[key] += 1
            progressed = True
            if len(selected) == size:
                break
        if not progressed:
            raise DynamicQwen3RunnerError(
                "Could not fill the deterministic stratified subset."
            )
    cases = []
    rule_rows = []
    selected_strata = []
    for row in selected:
        case_id = str(row["id"])
        prompt = str(row["prompt"])
        output = str(row["output"])
        if output.startswith(prompt):
            continuation = output[len(prompt) :]
        elif isinstance(row.get("continuation"), str):
            continuation = str(row["continuation"])
        else:
            raise DynamicQwen3RunnerError(
                f"Cached output for {case_id!r} does not preserve its prompt prefix."
            )
        raw_diagnostics = row.get("diagnostics")
        if not isinstance(raw_diagnostics, Mapping):
            raw_diagnostics = row["diagnostic"].get("diagnostics")
        if not isinstance(raw_diagnostics, Mapping):
            raise DynamicQwen3RunnerError(
                f"Cached generation metadata is missing for {case_id!r}."
            )
        generated_count = raw_diagnostics.get("generated_token_count")
        repeated_share = raw_diagnostics.get("repeated_bigram_share")
        if (
            isinstance(generated_count, bool)
            or not isinstance(generated_count, int)
            or generated_count < 0
            or generated_count > limit
        ):
            raise DynamicQwen3RunnerError(
                f"Cached generated token count is invalid for {case_id!r}."
            )
        repeated = _bounded_fraction(
            repeated_share,
            f"cached repeated_bigram_share for {case_id}",
        )
        rules = _normalized_rules(row.get("relevance_rules"), case_id)
        cases.append(
            WholeOutputCase(
                case_id=case_id,
                prompt=prompt,
                parent_output=continuation,
                category=str(row["category"]),
                split="heldout",
                parent_fragment_like=_fragment_like(
                    continuation,
                    fragment_chars,
                ),
                parent_metadata=WholeOutputMetadata(
                    generated_token_count=generated_count,
                    generation_limit=limit,
                    maximum_repeated_ngram_share=repeated,
                ),
            )
        )
        rule_rows.append((case_id, rules))
        selected_strata.append(
            (
                str(row["category"]),
                bool(row["diagnostic"].get("passed", False)),
            )
        )
    return CachedGenerationBaseline(
        source_path=str(source),
        source_sha256=source_sha256,
        cases=tuple(cases),
        relevance_rules=tuple(rule_rows),
        selected_strata=tuple(selected_strata),
        subset_seed=seed,
        generation_limit=limit,
        fragment_minimum_characters=fragment_chars,
    )


def _generation_report_maps(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> tuple[
    tuple[str, ...],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    generations = payload.get("generations")
    diagnostics = payload.get("generation_diagnostics")
    if (
        not isinstance(generations, list)
        or not generations
        or not isinstance(diagnostics, list)
        or not diagnostics
    ):
        raise DynamicQwen3RunnerError(
            f"{label} lacks generations and generation_diagnostics."
        )
    row_map: dict[str, dict[str, Any]] = {}
    ordered_ids = []
    for raw in generations:
        if not isinstance(raw, Mapping):
            raise DynamicQwen3RunnerError(f"{label} generation row is malformed.")
        case_id = str(raw.get("id", "")).strip()
        if not case_id or case_id in row_map:
            raise DynamicQwen3RunnerError(
                f"{label} generation IDs are empty or repeated."
            )
        row_map[case_id] = dict(raw)
        ordered_ids.append(case_id)
    diagnostic_map: dict[str, dict[str, Any]] = {}
    diagnostic_ids = []
    for raw in diagnostics:
        if not isinstance(raw, Mapping):
            raise DynamicQwen3RunnerError(
                f"{label} generation diagnostic is malformed."
            )
        case_id = str(raw.get("id", "")).strip()
        if not case_id or case_id in diagnostic_map:
            raise DynamicQwen3RunnerError(
                f"{label} generation diagnostic IDs are empty or repeated."
            )
        diagnostic_map[case_id] = dict(raw)
        diagnostic_ids.append(case_id)
    if tuple(diagnostic_ids) != tuple(ordered_ids):
        raise DynamicQwen3RunnerError(
            f"{label} generation and diagnostic ordering differs."
        )
    return tuple(ordered_ids), row_map, diagnostic_map


def _generation_contract_rows(
    ordered_ids: Sequence[str],
    rows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    result = []
    for case_id in ordered_ids:
        row = rows[case_id]
        prompt = row.get("prompt")
        category = row.get("category")
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(category, str)
            or not category.strip()
        ):
            raise DynamicQwen3RunnerError(
                f"Generation contract row {case_id!r} is malformed."
            )
        result.append(
            {
                "id": case_id,
                "prompt": prompt,
                "category": category,
                "relevance_rules": [
                    list(group)
                    for group in _normalized_rules(
                        row.get("relevance_rules"),
                        case_id,
                    )
                ],
            }
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CachedCandidatePreflight:
    candidate_path: str
    candidate_sha256: str
    checkpoint_manifest_sha256: str
    parent_eval_sha256: str
    binding_fingerprint: str
    certification_context_sha256: str
    decision: FoldDecision
    diagnostics: Mapping[str, Any]

    @property
    def accepted(self) -> bool:
        return self.decision.accepted

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "neural-forge-cached-candidate-preflight-v1",
            "candidate_path": self.candidate_path,
            "candidate_sha256": self.candidate_sha256,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "parent_eval_sha256": self.parent_eval_sha256,
            "binding_fingerprint": self.binding_fingerprint,
            "certification_context_sha256": self.certification_context_sha256,
            "decision": self.decision.as_dict(),
            "diagnostics": dict(self.diagnostics),
            "model_loading_performed": False,
            "gpu_loading_performed": False,
            "parent_forward_passes": 0,
        }


def preflight_cached_candidate(
    baseline: CachedGenerationBaseline,
    *,
    candidate_eval: str | os.PathLike[str],
    expected_candidate_sha256: str,
    checkpoint_manifest_sha256: str,
    target_parameters: int,
    attention_prototypes: int,
    neuron_prototypes: int,
    maximum_new_empty_responses: int,
    maximum_new_fragment_responses: int,
    minimum_parent_nonempty_retention: float,
    minimum_task_score_retention: float,
    maximum_new_generation_limit_saturations: int,
    maximum_repeated_bigram_share_regressions: int,
    maximum_repeated_bigram_share_delta: float,
) -> CachedCandidatePreflight:
    """Authenticate and gate a cached candidate without loading any model."""

    candidate_path = _resolve_local_file(
        candidate_eval,
        "cached candidate evaluation",
    )
    candidate_sha256 = _file_sha256(candidate_path)
    _optional_expected_digest(
        candidate_sha256,
        expected_candidate_sha256,
        "cached candidate evaluation",
    )
    parent_payload = _read_json_object(
        Path(baseline.source_path),
        "cached parent evaluation",
    )
    candidate_payload = _read_json_object(
        candidate_path,
        "cached candidate evaluation",
    )
    parent_suite = parent_payload.get("eval_suite")
    candidate_suite = candidate_payload.get("eval_suite")
    if (
        not isinstance(parent_suite, Mapping)
        or not isinstance(candidate_suite, Mapping)
        or canonical_json_bytes(parent_suite) != canonical_json_bytes(candidate_suite)
    ):
        raise DynamicQwen3RunnerError(
            "Cached parent and candidate evaluation-suite contracts differ."
        )
    parent_ids, parent_rows, _parent_diagnostics = _generation_report_maps(
        parent_payload,
        label="Cached parent evaluation",
    )
    candidate_ids, candidate_rows, candidate_diagnostics = _generation_report_maps(
        candidate_payload,
        label="Cached candidate evaluation",
    )
    if candidate_ids != parent_ids:
        raise DynamicQwen3RunnerError(
            "Cached parent and candidate generation ID order differs."
        )
    parent_contract_rows = _generation_contract_rows(parent_ids, parent_rows)
    candidate_contract_rows = _generation_contract_rows(
        candidate_ids,
        candidate_rows,
    )
    if candidate_contract_rows != parent_contract_rows:
        raise DynamicQwen3RunnerError(
            "Cached candidate prompts/categories/rules differ from the parent suite."
        )
    expected_model = (
        f"qwen3-codebook:pn{_positive_int(neuron_prototypes, 'neuron_prototypes')}:"
        f"pa{_positive_int(attention_prototypes, 'attention_prototypes')}"
    )
    if candidate_payload.get("model") != expected_model:
        raise DynamicQwen3RunnerError(
            "Cached candidate model label differs from the requested checkpoint "
            f"context: {candidate_payload.get('model')!r} != {expected_model!r}."
        )
    if candidate_payload.get("param_count") != _positive_int(
        target_parameters,
        "target_parameters",
    ):
        raise DynamicQwen3RunnerError(
            "Cached candidate parameter count differs from the requested target."
        )
    embedded_context = candidate_payload.get("certification_context")
    if embedded_context is not None and not isinstance(embedded_context, Mapping):
        raise DynamicQwen3RunnerError(
            "Cached candidate certification_context is malformed."
        )
    derived_context = {
        "schema": "neural-forge-derived-candidate-certification-context-v1",
        "model": expected_model,
        "param_count": target_parameters,
        "eval_suite": dict(candidate_suite),
        "generation_contract_sha256": canonical_sha256(list(candidate_contract_rows)),
        "generation_ids_sha256": canonical_sha256(list(candidate_ids)),
    }
    if isinstance(embedded_context, Mapping):
        validated = embedded_context.get("validated_candidate")
        if validated is not None:
            if not isinstance(validated, Mapping):
                raise DynamicQwen3RunnerError(
                    "Embedded validated_candidate context is malformed."
                )
            validated_model = validated.get("model")
            if validated_model is not None and validated_model != expected_model:
                raise DynamicQwen3RunnerError(
                    "Embedded candidate model provenance conflicts with the "
                    "requested checkpoint."
                )
            eval_counts = validated.get("eval_counts")
            if isinstance(eval_counts, Mapping):
                generation_count = eval_counts.get("generation_prompts")
                if generation_count is not None and generation_count != len(
                    candidate_ids
                ):
                    raise DynamicQwen3RunnerError(
                        "Embedded candidate generation count is inconsistent."
                    )
            prompt_ids = validated.get("prompt_ids")
            if isinstance(prompt_ids, Mapping):
                generation_ids = prompt_ids.get("generation_prompts")
                if (
                    generation_ids is not None
                    and tuple(generation_ids) != candidate_ids
                ):
                    raise DynamicQwen3RunnerError(
                        "Embedded candidate generation IDs are inconsistent."
                    )
        certification_context_sha256 = canonical_sha256(dict(embedded_context))
    else:
        certification_context_sha256 = canonical_sha256(derived_context)
    binding = {
        "schema": "neural-forge-checkpoint-candidate-binding-v1",
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "candidate_eval_sha256": candidate_sha256,
        "parent_eval_sha256": baseline.source_sha256,
        "certification_context_sha256": certification_context_sha256,
        "candidate_model": expected_model,
        "target_parameters": target_parameters,
        "attention_prototypes": attention_prototypes,
        "neuron_prototypes": neuron_prototypes,
        "generation_contract_sha256": derived_context["generation_contract_sha256"],
    }
    outputs: dict[str, CandidateWholeOutput] = {}
    repetition_regressions = []
    severe_repetition_regressions = []
    task_regressions = []
    pruning_task_regressions = []
    parent_rules = baseline.rules
    for case in baseline.cases:
        row = candidate_rows.get(case.case_id)
        diagnostic = candidate_diagnostics.get(case.case_id)
        if row is None or diagnostic is None:
            raise DynamicQwen3RunnerError(
                f"Cached candidate lacks selected case {case.case_id!r}."
            )
        candidate_prompt = row.get("prompt")
        if (
            not isinstance(candidate_prompt, str)
            or candidate_prompt.strip() != case.prompt
            or row.get("category") != case.category
            or _normalized_rules(row.get("relevance_rules"), case.case_id)
            != parent_rules.get(case.case_id, ())
        ):
            raise DynamicQwen3RunnerError(
                f"Cached candidate selected case {case.case_id!r} changed contract."
            )
        output = row.get("output")
        if not isinstance(output, str):
            raise DynamicQwen3RunnerError(
                f"Cached candidate output for {case.case_id!r} is malformed."
            )
        if output.startswith(candidate_prompt):
            continuation = output[len(candidate_prompt) :]
        elif isinstance(row.get("continuation"), str):
            continuation = str(row["continuation"])
        else:
            raise DynamicQwen3RunnerError(
                f"Cached candidate output for {case.case_id!r} lost its prompt prefix."
            )
        raw_metadata = row.get("diagnostics")
        if not isinstance(raw_metadata, Mapping):
            raw_metadata = diagnostic.get("diagnostics")
        if not isinstance(raw_metadata, Mapping):
            raise DynamicQwen3RunnerError(
                f"Cached candidate metadata for {case.case_id!r} is missing."
            )
        generated_count = raw_metadata.get("generated_token_count")
        repeated_share = raw_metadata.get("repeated_bigram_share")
        if (
            isinstance(generated_count, bool)
            or not isinstance(generated_count, int)
            or generated_count < 0
            or generated_count > baseline.generation_limit
        ):
            raise DynamicQwen3RunnerError(
                f"Cached candidate token count for {case.case_id!r} is invalid."
            )
        repeated = _bounded_fraction(
            repeated_share,
            f"cached candidate repeated_bigram_share for {case.case_id}",
        )
        outputs[case.case_id] = CandidateWholeOutput(
            output=continuation,
            fragment_like=_fragment_like(
                continuation,
                baseline.fragment_minimum_characters,
            ),
            metadata=WholeOutputMetadata(
                generated_token_count=generated_count,
                generation_limit=baseline.generation_limit,
                maximum_repeated_ngram_share=repeated,
            ),
        )
        parent_metadata = case.parent_metadata
        if parent_metadata is None:
            raise AssertionError("Cached parent metadata vanished.")
        parent_repetition = parent_metadata.maximum_repeated_ngram_share
        if parent_repetition is None:
            raise AssertionError("Cached parent repetition metadata vanished.")
        repetition_delta = repeated - parent_repetition
        if repetition_delta > maximum_repeated_bigram_share_delta + 1e-12:
            repetition_regressions.append(case.case_id)
            if repeated >= 0.80:
                severe_repetition_regressions.append(case.case_id)
        parent_score = baseline.task_score(case, case.parent_output)
        candidate_score = baseline.task_score(case, continuation)
        if parent_score is True and candidate_score is False:
            task_regressions.append(case.case_id)
            if case.case_id.startswith("generation_pruning_"):
                pruning_task_regressions.append(case.case_id)
    gate = build_cached_whole_output_gate(
        baseline,
        generator=lambda _model, _cases: outputs,
        maximum_new_empty_responses=maximum_new_empty_responses,
        maximum_new_fragment_responses=maximum_new_fragment_responses,
        minimum_parent_nonempty_retention=minimum_parent_nonempty_retention,
        minimum_task_score_retention=minimum_task_score_retention,
        maximum_new_generation_limit_saturations=(
            maximum_new_generation_limit_saturations
        ),
        maximum_repeated_bigram_share_regressions=(
            maximum_repeated_bigram_share_regressions
        ),
        maximum_repeated_bigram_share_delta=(maximum_repeated_bigram_share_delta),
    )
    decision = gate.evaluate_model(
        object(),
        verify_model_execution=False,
    )
    diagnostics = {
        "selected_case_count": len(baseline.cases),
        "repetition_regression_count": len(repetition_regressions),
        "repetition_regression_ids": repetition_regressions,
        "severe_repetition_regression_count": len(severe_repetition_regressions),
        "severe_repetition_regression_ids": severe_repetition_regressions,
        "task_regression_count": len(task_regressions),
        "task_regression_ids": task_regressions,
        "pruning_task_regression_count": len(pruning_task_regressions),
        "pruning_task_regression_ids": pruning_task_regressions,
        "exact_parent_subset_reused": True,
        "whole_output_gate_reused": True,
    }
    return CachedCandidatePreflight(
        candidate_path=str(candidate_path),
        candidate_sha256=candidate_sha256,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        parent_eval_sha256=baseline.source_sha256,
        binding_fingerprint=canonical_sha256(binding),
        certification_context_sha256=certification_context_sha256,
        decision=decision,
        diagnostics=diagnostics,
    )


class GreedyWholeOutputGenerator:
    """Generate independent deterministic child continuations."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        max_new_tokens: int,
        maximum_prompt_tokens: int,
        fragment_minimum_characters: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_new_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        self.maximum_prompt_tokens = _positive_int(
            maximum_prompt_tokens,
            "maximum_prompt_tokens",
        )
        self.fragment_minimum_characters = _positive_int(
            fragment_minimum_characters,
            "fragment_minimum_characters",
        )
        self.invocations: list[dict[str, Any]] = []

    def __call__(
        self,
        model: nn.Module,
        cases: tuple[WholeOutputCase, ...],
    ) -> Mapping[str, CandidateWholeOutput]:
        device = _model_device(model)
        was_training = bool(model.training)
        model.eval()
        outputs: dict[str, CandidateWholeOutput] = {}
        summaries = []
        try:
            with torch.inference_mode():
                for case in cases:
                    encoded = self.tokenizer(
                        case.prompt,
                        return_tensors="pt",
                        add_special_tokens=True,
                    )
                    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
                        raise DynamicQwen3RunnerError(
                            "Tokenizer did not return input_ids."
                        )
                    input_ids = encoded["input_ids"]
                    if (
                        not isinstance(input_ids, Tensor)
                        or input_ids.ndim != 2
                        or input_ids.shape[0] != 1
                        or input_ids.shape[1] <= 0
                        or input_ids.shape[1] > self.maximum_prompt_tokens
                    ):
                        raise DynamicQwen3RunnerError(
                            f"Prompt tokenization is invalid for {case.case_id!r}."
                        )
                    input_ids = input_ids.to(device=device, dtype=torch.long)
                    attention_mask = encoded.get("attention_mask")
                    if attention_mask is not None:
                        if not isinstance(attention_mask, Tensor):
                            raise DynamicQwen3RunnerError(
                                "Tokenizer attention_mask is not a tensor."
                            )
                        attention_mask = attention_mask.to(
                            device=device,
                            dtype=torch.long,
                        )
                    kwargs: dict[str, Any] = {
                        "input_ids": input_ids,
                        "max_new_tokens": self.max_new_tokens,
                        "do_sample": False,
                        "use_cache": True,
                    }
                    if attention_mask is not None:
                        kwargs["attention_mask"] = attention_mask
                    pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
                    if pad_token_id is not None:
                        kwargs["pad_token_id"] = int(pad_token_id)
                    generated = model.generate(**kwargs)
                    if (
                        not isinstance(generated, Tensor)
                        or generated.ndim != 2
                        or generated.shape[0] != 1
                        or generated.shape[1] < input_ids.shape[1]
                    ):
                        raise DynamicQwen3RunnerError(
                            f"Greedy generation is invalid for {case.case_id!r}."
                        )
                    continuation = generated[:, int(input_ids.shape[1]) :]
                    token_ids = [
                        int(value)
                        for value in continuation[0].detach().to(device="cpu").tolist()
                    ]
                    text = self.tokenizer.decode(
                        token_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    repeated_share = _surface_repeated_bigram_share(text)
                    fragment = _fragment_like(
                        text,
                        self.fragment_minimum_characters,
                    )
                    outputs[case.case_id] = CandidateWholeOutput(
                        output=text,
                        fragment_like=fragment,
                        metadata=WholeOutputMetadata(
                            generated_token_count=len(token_ids),
                            generation_limit=self.max_new_tokens,
                            maximum_repeated_ngram_share=repeated_share,
                        ),
                    )
                    summaries.append(
                        {
                            "case_id": case.case_id,
                            "generated_token_count": len(token_ids),
                            "empty": not bool(text.strip()),
                            "fragment_like": fragment,
                            "generation_limit_saturated": (
                                len(token_ids) == self.max_new_tokens
                            ),
                            "repeated_bigram_share": repeated_share,
                        }
                    )
        finally:
            model.train(was_training)
        self.invocations.append(
            {
                "device": str(device),
                "case_count": len(cases),
                "empty_count": sum(row["empty"] for row in summaries),
                "fragment_count": sum(row["fragment_like"] for row in summaries),
                "generation_limit_saturation_count": sum(
                    row["generation_limit_saturated"] for row in summaries
                ),
                "maximum_repeated_bigram_share": max(
                    (row["repeated_bigram_share"] for row in summaries),
                    default=0.0,
                ),
            }
        )
        return outputs

    def receipt(self) -> dict[str, Any]:
        return {
            "kind": "deterministic-greedy-independent-output",
            "invocation_count": len(self.invocations),
            "invocations": list(self.invocations),
            "max_new_tokens": self.max_new_tokens,
            "maximum_prompt_tokens": self.maximum_prompt_tokens,
            "fragment_minimum_characters": self.fragment_minimum_characters,
            "parent_forward_passes": 0,
        }


def build_cached_whole_output_gate(
    baseline: CachedGenerationBaseline,
    *,
    generator: Callable[
        [nn.Module, tuple[WholeOutputCase, ...]],
        Mapping[str, CandidateWholeOutput],
    ],
    maximum_new_empty_responses: int,
    maximum_new_fragment_responses: int,
    minimum_parent_nonempty_retention: float,
    minimum_task_score_retention: float,
    maximum_new_generation_limit_saturations: int,
    maximum_repeated_bigram_share_regressions: int,
    maximum_repeated_bigram_share_delta: float,
) -> CachedWholeOutputBehaviorGate:
    required_categories = tuple(
        sorted(
            {
                case.category
                for case in baseline.cases
                if baseline.task_score(case, case.parent_output) is True
            }
        )
    )
    generator_identity = identify_behavior_callable(
        generator,
        implementation_id=("neural_forge.qwen3_codebook.greedy_whole_output_generator"),
        version=1,
    )
    scorer_identity = identify_behavior_callable(
        baseline.task_score,
        implementation_id="neural_forge.qwen3_codebook.regex_task_scorer",
        version=1,
    )
    return CachedWholeOutputBehaviorGate(
        baseline.cases,
        generator=generator,
        config=WholeOutputGateConfig(
            maximum_new_empty_responses=_nonnegative_int(
                maximum_new_empty_responses,
                "maximum_new_empty_responses",
            ),
            maximum_new_fragment_responses=_nonnegative_int(
                maximum_new_fragment_responses,
                "maximum_new_fragment_responses",
            ),
            minimum_parent_nonempty_retention=_bounded_fraction(
                minimum_parent_nonempty_retention,
                "minimum_parent_nonempty_retention",
            ),
            minimum_task_score_retention=_bounded_fraction(
                minimum_task_score_retention,
                "minimum_task_score_retention",
            ),
            maximum_new_generation_limit_saturations=_nonnegative_int(
                maximum_new_generation_limit_saturations,
                "maximum_new_generation_limit_saturations",
            ),
            maximum_repeated_ngram_share_regressions=_nonnegative_int(
                maximum_repeated_bigram_share_regressions,
                "maximum_repeated_bigram_share_regressions",
            ),
            maximum_repeated_ngram_share_delta=_bounded_fraction(
                maximum_repeated_bigram_share_delta,
                "maximum_repeated_bigram_share_delta",
            ),
            task_score_coverage=TaskScoreCoveragePolicy.production(
                required_splits=("heldout",),
                required_categories=required_categories,
                require_split_category_cross_product=bool(required_categories),
            ),
        ),
        task_scorer=baseline.task_score,
        gate_id="neural_forge.qwen3_codebook.whole_output_behavior",
        gate_version=2,
        generator_identity=generator_identity,
        task_scorer_identity=scorer_identity,
        baseline_binding={
            "source_sha256": baseline.source_sha256,
            "baseline_fingerprint": baseline.fingerprint,
            "selected_strata": [list(value) for value in baseline.selected_strata],
            "subset_seed": baseline.subset_seed,
            "generation_limit": baseline.generation_limit,
            "fragment_minimum_characters": (baseline.fragment_minimum_characters),
        },
    )


class ParentReleaseBeforeFreshLoad:
    """Move the no-longer-needed parent to CPU before fresh child loading."""

    def __init__(
        self,
        *,
        parent: nn.Module,
        expected_parent_fingerprint: str,
        enabled: bool,
        reload_device: str | torch.device,
    ) -> None:
        self.parent = parent
        self.expected_parent_fingerprint = expected_parent_fingerprint
        self.enabled = bool(enabled)
        self.reload_device = torch.device(reload_device)
        self.called = False
        self.valid_load_count = 0
        self.report: dict[str, Any] = {
            "enabled": self.enabled,
            "performed": False,
            "parent_values_unchanged": None,
            "reload_device": str(self.reload_device),
        }

    def wrap(self, loader: Callable[[Path], nn.Module]) -> Callable[[Path], nn.Module]:
        if not callable(loader):
            raise DynamicQwen3RunnerError("Fresh child loader must be callable.")

        def load(path: Path) -> nn.Module:
            source = Path(path).expanduser()
            if source.is_dir() and not source.is_symlink():
                try:
                    empty_path_challenge = next(source.iterdir(), None) is None
                except OSError:
                    empty_path_challenge = False
                if empty_path_challenge:
                    return loader(source)
            if not self.called:
                self.called = True
                before_devices = sorted(
                    {str(parameter.device) for parameter in self.parent.parameters()}
                )
                before = fingerprint_model_state(self.parent)
                if before != self.expected_parent_fingerprint:
                    raise DynamicQwen3RunnerError(
                        "Parent state changed before the fresh-load release boundary."
                    )
                if self.enabled:
                    self.parent.to(device=torch.device("cpu"))
                    gc.collect()
                    for raw_device in before_devices:
                        device = torch.device(raw_device)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                after = fingerprint_model_state(self.parent)
                unchanged = after == before == self.expected_parent_fingerprint
                if not unchanged:
                    raise DynamicQwen3RunnerError(
                        "Moving the parent before fresh reload changed its values."
                    )
                self.report = {
                    "enabled": self.enabled,
                    "performed": self.enabled,
                    "parent_devices_before": before_devices,
                    "parent_devices_after": sorted(
                        {
                            str(parameter.device)
                            for parameter in self.parent.parameters()
                        }
                    ),
                    "parent_state_before": before,
                    "parent_state_after": after,
                    "parent_values_unchanged": unchanged,
                    "reload_device": str(self.reload_device),
                    "valid_load_count": 0,
                }
            reloaded = loader(path)
            observed_reload_device = _model_device(reloaded)
            if observed_reload_device != self.reload_device:
                raise DynamicQwen3RunnerError(
                    "Fresh child loaded on an unexpected device: "
                    f"{observed_reload_device} != {self.reload_device}."
                )
            self.valid_load_count += 1
            self.report["valid_load_count"] = self.valid_load_count
            self.report["reload_device"] = str(observed_reload_device)
            return reloaded

        return load


class CheckpointTopologyQwen3TargetSynthesisAdapter:
    """Create a seeded blank from authenticated checkpoint topology, without fit."""

    adapter_id = Qwen3CodebookTargetSynthesisAdapter.adapter_id
    version = Qwen3CodebookTargetSynthesisAdapter.version
    priority = Qwen3CodebookTargetSynthesisAdapter.priority + 100
    target_families = Qwen3CodebookTargetSynthesisAdapter.target_families
    target_modalities = Qwen3CodebookTargetSynthesisAdapter.target_modalities

    def __init__(
        self,
        *,
        checkpoint: Path,
        cache: Any,
        device: str | torch.device,
        dtype: torch.dtype,
        manifest_sha256: str,
    ) -> None:
        self.checkpoint = checkpoint
        self.cache = cache
        self.device = torch.device(device)
        self.dtype = dtype
        self.manifest_sha256 = manifest_sha256

    def supports(
        self,
        request: DynamicTargetRequest,
        parent_graphs: Mapping[str, Any],
    ) -> bool:
        return (
            request.target_family in {"auto", "qwen3_codebook"}
            and request.target_modalities == ("text",)
            and any(graph.family == "qwen3" for graph in parent_graphs.values())
        )

    def materialize(
        self,
        request: DynamicTargetRequest,
        parent_models: Mapping[str, Any],
        parent_graphs: Mapping[str, Any],
        inspectors: Any,
    ) -> AdapterMaterialization:
        configured_parent = dict(request.configuration).get("source_parent")
        if (
            not isinstance(configured_parent, str)
            or configured_parent not in parent_models
            or parent_graphs[configured_parent].family != "qwen3"
        ):
            raise DynamicQwen3RunnerError(
                "Checkpoint-topology target requires one configured Qwen3 parent."
            )
        parent = parent_models[configured_parent]
        self.cache.validate_parent(configured_parent, parent)
        blank = load_qwen3_codebook_checkpoint(
            self.checkpoint,
            device=self.device,
        )
        report = qwen3_codebook_parameter_report(blank)
        if (
            report.unique_parameter_count != request.exact_unique_parameters
            or report.unique_parameter_count != self.cache.expected_unique_parameters
            or report.attention_prototypes_per_group
            != self.cache.attention_prototypes_per_group
            or report.neuron_prototypes_per_neuron
            != self.cache.neuron_prototypes_per_neuron
        ):
            raise DynamicQwen3RunnerError(
                "Checkpoint topology differs from the requested cached target."
            )
        parameter_dtypes = {parameter.dtype for parameter in blank.parameters()}
        if parameter_dtypes != {self.dtype}:
            raise DynamicQwen3RunnerError(
                "Requested child dtype differs from the standalone checkpoint: "
                f"{sorted(map(str, parameter_dtypes))} != {self.dtype}."
            )
        self.cache.assert_parameter_contract(blank)
        self.cache.assert_topology(blank)
        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices = [
                (
                    torch.cuda.current_device()
                    if self.device.index is None
                    else self.device.index
                )
            ]
        initializer_range = float(
            getattr(getattr(blank, "config", None), "initializer_range", 0.02)
        )
        if not 0.0 < initializer_range < 1.0:
            raise DynamicQwen3RunnerError(
                "Qwen3 checkpoint has an invalid initializer_range."
            )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(request.blank_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(request.blank_seed)
            with torch.no_grad():
                for parameter in blank.parameters():
                    torch.nn.init.normal_(
                        parameter,
                        mean=0.0,
                        std=initializer_range,
                    )
                    parameter.grad = None
        blank.eval()
        if self.cache.matches_parameters(blank):
            raise DynamicQwen3RunnerError(
                "Seed erasure failed to create a real blank child."
            )
        self.cache.assert_topology(blank)
        graph = inspectors.inspect(
            blank,
            model_name=request.child_name,
            tokenizer_fingerprint=request.tokenizer_fingerprint,
        )
        if graph.family != "qwen3_codebook":
            raise DynamicQwen3RunnerError(
                "Checkpoint-topology blank was not identified as Qwen3 codebook."
            )
        blank_state = fingerprint_model_state(blank)
        topology_fingerprint = canonical_sha256(
            {
                "family": graph.family,
                "graph_fingerprint": graph.fingerprint,
                "cached_topology_fingerprint": self.cache.topology_fingerprint,
                "manifest_sha256": self.manifest_sha256,
                "blank_materialization": "checkpoint-topology-seed-erasure",
            }
        )
        return AdapterMaterialization(
            model=blank,
            graph=graph,
            topology_fingerprint=topology_fingerprint,
            report={
                "source_parent": configured_parent,
                "blank_seed": request.blank_seed,
                "blank_state_fingerprint": blank_state,
                "cached_fold_fingerprint": self.cache.fingerprint,
                "cached_topology_fingerprint": self.cache.topology_fingerprint,
                "checkpoint_manifest_sha256": self.manifest_sha256,
                "blank_learned_parameters_parent_derived": False,
                "topology_loaded_from_immutable_checkpoint": True,
                "fresh_codebook_fit_performed": False,
                "optimizer_constructions": 0,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "autograd_grad_calls": 0,
            },
        )


class RunnerQwen3CodebookStackAdapter:
    """Wrap the persistent Qwen stack with the fresh-load release boundary."""

    adapter_id = "neural_forge.qwen3.codebook.stack.production_runner"
    version = 1
    priority = 300
    target_families = ("qwen3_codebook",)
    target_modalities = ("text",)
    method_ids = Qwen3CodebookStackFactory.method_ids

    def __init__(
        self,
        *,
        delegate: Qwen3CodebookStackFactory,
        parent_release: ParentReleaseBeforeFreshLoad,
    ) -> None:
        self.delegate = delegate
        self.parent_release = parent_release

    def supports(self, target: DynamicTargetSynthesisResult) -> bool:
        return self.delegate.supports(target)

    def build(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> DynamicFoldSystemStack:
        stack = self.delegate.build(target)
        return replace(
            stack,
            model_loader=self.parent_release.wrap(stack.model_loader),
        )


def _checkpoint_contract(
    checkpoint: str | os.PathLike[str],
    expected_manifest_sha256: str,
) -> tuple[Path, dict[str, Any], str]:
    directory = _resolve_local_directory(checkpoint, "fold checkpoint")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise DynamicQwen3RunnerError("Fold checkpoint lacks manifest.json.")
    manifest_sha256 = _file_sha256(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise DynamicQwen3RunnerError(
            "Fold checkpoint manifest differs from the required SHA-256 pin."
        )
    contract = _read_json_object(
        directory / "tokenizer_contract.json",
        "fold checkpoint tokenizer contract",
    )
    return directory, contract, manifest_sha256


def _verify_runtime_tokenizer_contract(
    *,
    stored_contract: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
) -> None:
    keys = (
        "tokenizer_name_or_path",
        "vocab_size",
        "vocab_sha256",
        "special_token_ids",
        "parent_token_rows_sha256",
    )
    observed = {key: runtime_contract.get(key) for key in keys}
    expected = {key: stored_contract.get(key) for key in keys}
    if observed != expected:
        raise DynamicQwen3RunnerError(
            "Runtime target tokenizer differs from the checkpoint contract."
        )


def _verify_cached_parent_rows(
    parent_tokenizer: Any,
    target_tokenizer: Any,
    cache: Any,
) -> str:
    observed = exact_token_row_mapping(parent_tokenizer, target_tokenizer)
    expected_tensor = dict(cache.topology_buffers).get("parent_token_rows")
    if not isinstance(expected_tensor, Tensor):
        raise DynamicQwen3RunnerError("Fold cache lacks parent_token_rows.")
    expected = tuple(
        int(value)
        for value in expected_tensor.detach()
        .to(device="cpu", dtype=torch.int64)
        .tolist()
    )
    if observed != expected:
        raise DynamicQwen3RunnerError(
            "Supplied parent/target tokenizers do not reproduce cached rows."
        )
    return hashlib.sha256(canonical_json_bytes(list(observed))).hexdigest()


def _direct_local_model_audit(directory: Path) -> dict[str, Any]:
    layout = _local_safetensor_layout(directory)
    config_path = directory / "config.json"
    return {
        "path": str(directory),
        "config_sha256": _file_sha256(config_path),
        "safetensor_files": [
            {"name": shard.name, "bytes": shard.stat().st_size}
            for shard in layout.shards
        ],
        "config_loader": "direct-local-json-to-Qwen3Config",
        "model_loader": "direct-local-safetensors-to-Qwen3ForCausalLM",
        "generic_model_loader_used": False,
        "hub_download_used": False,
        "network_allowed": False,
    }


def _receipt_path(args: argparse.Namespace, output: Path) -> Path:
    if args.receipt is not None:
        return Path(args.receipt).expanduser().absolute()
    return output.parent / f"{output.name}.run_receipt.json"


def _snapshot_root(args: argparse.Namespace, output: Path) -> Path:
    if args.snapshot_root is not None:
        return Path(args.snapshot_root).expanduser().absolute()
    return output.parent / f".{output.name}.fold-snapshots"


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _preflight_rejected_receipt(
    *,
    output: Path,
    receipt_path: Path,
    checkpoint: Path,
    preflight: CachedCandidatePreflight,
    baseline: CachedGenerationBaseline,
    args: argparse.Namespace,
) -> dict[str, Any]:
    receipt = {
        "schema": SCHEMA,
        "status": "cached_candidate_preflight_rejected",
        "complete": False,
        "output": str(output),
        "receipt": str(receipt_path),
        "inputs": {
            "fold_checkpoint": {
                "path": str(checkpoint),
                "manifest_sha256": preflight.checkpoint_manifest_sha256,
            },
            "cached_parent_behavior": baseline.as_dict(),
            "cached_candidate_preflight": preflight.as_dict(),
        },
        "target": {
            "child_name": args.child_name,
            "attention_prototypes_per_group": args.attention_prototypes,
            "neuron_prototypes_per_neuron": args.neuron_prototypes,
            "requested_unique_parameters": args.target_parameters,
        },
        "execution": {
            "model_loading_performed": False,
            "gpu_loading_performed": False,
            "tokenizer_loading_performed": False,
            "parent_forward_passes": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "autograd_grad_calls": 0,
        },
        "artifact": {
            "published": False,
            "atomic_output_created": False,
        },
        "io_contract": {
            "checkpoint_and_eval_sha256_authenticated": True,
            "network_allowed": False,
            "training_used": False,
        },
    }
    _atomic_write_json(receipt_path, receipt)
    return receipt


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    output = Path(args.output).expanduser().absolute()
    if output.exists():
        raise DynamicQwen3RunnerError(
            f"Atomic output destination already exists: {output}."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = _receipt_path(args, output)
    if receipt_path.exists():
        raise DynamicQwen3RunnerError(f"Run receipt already exists: {receipt_path}.")
    snapshot_root = _snapshot_root(args, output)
    if receipt_path.is_relative_to(output):
        raise DynamicQwen3RunnerError(
            "Run receipt must remain outside the certified artifact directory."
        )
    if _paths_overlap(snapshot_root, output):
        raise DynamicQwen3RunnerError(
            "Snapshot root and atomic artifact destination must be disjoint."
        )
    checkpoint, stored_contract, manifest_sha256 = _checkpoint_contract(
        args.fold_checkpoint,
        args.checkpoint_manifest_sha256,
    )
    baseline = load_cached_generation_baseline(
        args.cached_parent_eval,
        subset_size=args.behavior_subset_size,
        subset_seed=args.behavior_subset_seed,
        generation_limit=args.max_new_tokens,
        fragment_minimum_characters=args.fragment_minimum_characters,
        expected_sha256=args.cached_parent_eval_sha256,
    )
    candidate_preflight = preflight_cached_candidate(
        baseline,
        candidate_eval=args.cached_candidate_eval,
        expected_candidate_sha256=args.cached_candidate_eval_sha256,
        checkpoint_manifest_sha256=manifest_sha256,
        target_parameters=args.target_parameters,
        attention_prototypes=args.attention_prototypes,
        neuron_prototypes=args.neuron_prototypes,
        maximum_new_empty_responses=args.maximum_new_empty_responses,
        maximum_new_fragment_responses=args.maximum_new_fragment_responses,
        minimum_parent_nonempty_retention=args.minimum_parent_nonempty_retention,
        minimum_task_score_retention=args.minimum_task_score_retention,
        maximum_new_generation_limit_saturations=(
            args.maximum_new_generation_limit_saturations
        ),
        maximum_repeated_bigram_share_regressions=(
            args.maximum_repeated_bigram_share_regressions
        ),
        maximum_repeated_bigram_share_delta=(args.maximum_repeated_bigram_share_delta),
    )
    if not candidate_preflight.accepted:
        return _preflight_rejected_receipt(
            output=output,
            receipt_path=receipt_path,
            checkpoint=checkpoint,
            preflight=candidate_preflight,
            baseline=baseline,
            args=args,
        )
    snapshot_root.mkdir(parents=True, exist_ok=True)
    dtype = _torch_dtype(args.dtype)
    parent_directory = _resolve_local_directory(args.parent_model, "parent model")
    parent_tokenizer_directory = _resolve_local_directory(
        args.parent_tokenizer,
        "parent tokenizer",
    )
    target_tokenizer_directory = _resolve_local_directory(
        args.target_tokenizer,
        "target tokenizer",
    )
    parent_tokenizer = load_local_tokenizer(parent_tokenizer_directory)
    target_tokenizer = load_local_tokenizer(target_tokenizer_directory)
    parent_tokenizer_identity = fingerprint_local_tokenizer(
        parent_tokenizer_directory,
        parent_tokenizer,
    )
    target_tokenizer_identity = fingerprint_local_tokenizer(
        target_tokenizer_directory,
        target_tokenizer,
    )
    _optional_expected_digest(
        parent_tokenizer_identity.fingerprint,
        args.expected_parent_tokenizer_sha256,
        "parent tokenizer",
    )
    _optional_expected_digest(
        target_tokenizer_identity.fingerprint,
        args.expected_target_tokenizer_sha256,
        "target tokenizer",
    )
    parent = load_local_qwen3_model(
        parent_directory,
        device=args.parent_device,
        dtype=dtype,
    )
    parent_state_fingerprint = fingerprint_model_state(parent)
    _optional_expected_digest(
        parent_state_fingerprint,
        args.expected_parent_state_sha256,
        "parent model state",
    )
    cache = build_qwen3_global_codebook_fold_cache_from_checkpoint(
        checkpoint=checkpoint,
        source_parent=args.parent_name,
        live_parent_model=parent,
        attention_prototypes_per_group=args.attention_prototypes,
        neuron_prototypes_per_neuron=args.neuron_prototypes,
        tokenizer_contract=stored_contract,
        expected_manifest_sha256=manifest_sha256,
        cache_device=args.cache_device,
    )
    if cache.expected_unique_parameters != args.target_parameters:
        raise DynamicQwen3RunnerError(
            "Checkpoint budget differs from --target-parameters: "
            f"{cache.expected_unique_parameters} != {args.target_parameters}."
        )
    token_row_fingerprint = _verify_cached_parent_rows(
        parent_tokenizer,
        target_tokenizer,
        cache,
    )
    tokenizer_name = stored_contract.get("tokenizer_name_or_path")
    if not isinstance(tokenizer_name, str) or not tokenizer_name.strip():
        raise DynamicQwen3RunnerError(
            "Checkpoint tokenizer contract lacks tokenizer_name_or_path."
        )
    tokenizer_contract = build_qwen3_codebook_tokenizer_contract(
        target_tokenizer=target_tokenizer,
        cache=cache,
        tokenizer_name_or_path=tokenizer_name,
    )
    _verify_runtime_tokenizer_contract(
        stored_contract=stored_contract,
        runtime_contract=tokenizer_contract,
    )
    generator = GreedyWholeOutputGenerator(
        tokenizer=target_tokenizer,
        max_new_tokens=args.max_new_tokens,
        maximum_prompt_tokens=args.maximum_prompt_tokens,
        fragment_minimum_characters=args.fragment_minimum_characters,
    )
    behavior_gate = build_cached_whole_output_gate(
        baseline,
        generator=generator,
        maximum_new_empty_responses=args.maximum_new_empty_responses,
        maximum_new_fragment_responses=args.maximum_new_fragment_responses,
        minimum_parent_nonempty_retention=args.minimum_parent_nonempty_retention,
        minimum_task_score_retention=args.minimum_task_score_retention,
        maximum_new_generation_limit_saturations=(
            args.maximum_new_generation_limit_saturations
        ),
        maximum_repeated_bigram_share_regressions=(
            args.maximum_repeated_bigram_share_regressions
        ),
        maximum_repeated_bigram_share_delta=(args.maximum_repeated_bigram_share_delta),
    )
    target_adapter = CheckpointTopologyQwen3TargetSynthesisAdapter(
        checkpoint=checkpoint,
        cache=cache,
        device=args.child_device,
        dtype=dtype,
        manifest_sha256=manifest_sha256,
    )
    delegate_stack = Qwen3CodebookStackFactory(
        cache=cache,
        final_evaluator=behavior_gate,
        tokenizer_contract=tokenizer_contract,
        fresh_reload_evaluator=behavior_gate,
        load_device=args.reload_device,
    )
    parent_release = ParentReleaseBeforeFreshLoad(
        parent=parent,
        expected_parent_fingerprint=parent_state_fingerprint,
        enabled=args.release_parent_before_reload,
        reload_device=args.reload_device,
    )
    stack_adapter = RunnerQwen3CodebookStackAdapter(
        delegate=delegate_stack,
        parent_release=parent_release,
    )
    system = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((target_adapter,)),
        stack_adapters=DynamicFoldStackAdapterRegistry((stack_adapter,)),
        snapshot_config=DynamicFoldSnapshotConfig.disk(
            snapshot_root,
            chunk_bytes=args.snapshot_chunk_mib * 1024**2,
        ),
    )
    request = DynamicTargetRequest(
        child_name=args.child_name,
        target_family="qwen3_codebook",
        target_modalities=("text",),
        exact_unique_parameters=args.target_parameters,
        maximum_parameter_bytes=args.maximum_parameter_bytes,
        tokenizer_fingerprint=target_tokenizer_identity.fingerprint,
        adapter_id=CheckpointTopologyQwen3TargetSynthesisAdapter.adapter_id,
        blank_seed=args.blank_seed,
        configuration=(("source_parent", args.parent_name),),
    )
    result = system.run(
        parent_models={args.parent_name: parent},
        target_request=request,
        evidence_fingerprint=cache.fingerprint,
        destination=output,
        parent_tokenizer_fingerprints={
            args.parent_name: parent_tokenizer_identity.fingerprint
        },
    )
    run_result = result.published.compilation.run
    certificate = result.published.artifact
    if (
        not result.complete
        or run_result.optimizer_constructions
        or run_result.optimizer_steps
        or run_result.backward_calls
        or run_result.autograd_grad_calls
        or certificate.optimizer_constructions
        or certificate.optimizer_steps
        or certificate.backward_calls
        or certificate.autograd_grad_calls
    ):
        raise DynamicQwen3RunnerError(
            "Dynamic Qwen3 fold did not complete under the zero-training contract."
        )
    if generator.receipt()["invocation_count"] != 3:
        raise DynamicQwen3RunnerError(
            "Whole-output behavior gate did not run in-memory, after staging "
            "reload, and from the retained final destination."
        )
    if snapshot_directory_inventory(snapshot_root):
        raise DynamicQwen3RunnerError(
            "Disk-backed executor snapshots were not cleaned."
        )
    final_decision = run_result.final_decision
    if final_decision is None or not final_decision.accepted:
        raise DynamicQwen3RunnerError(
            "Completed run lacks an accepted final behavior decision."
        )
    receipt = {
        "schema": SCHEMA,
        "status": "complete",
        "complete": True,
        "output": str(output),
        "receipt": str(receipt_path),
        "inputs": {
            "parent_model": _direct_local_model_audit(parent_directory),
            "parent_state_fingerprint": parent_state_fingerprint,
            "parent_tokenizer": parent_tokenizer_identity.as_dict(),
            "target_tokenizer": target_tokenizer_identity.as_dict(),
            "fold_checkpoint": {
                "path": str(checkpoint),
                "manifest_sha256": manifest_sha256,
                "cache_fingerprint": cache.fingerprint,
                "parameter_fingerprint": cache.parameter_fingerprint,
                "topology_fingerprint": cache.topology_fingerprint,
                "token_row_fingerprint": token_row_fingerprint,
            },
            "cached_parent_behavior": baseline.as_dict(),
            "cached_candidate_preflight": candidate_preflight.as_dict(),
        },
        "devices": {
            "parent": args.parent_device,
            "child": args.child_device,
            "reload": args.reload_device,
            "cache": args.cache_device,
            "dtype": args.dtype,
            "parent_release_before_reload": parent_release.report,
        },
        "target": {
            "child_name": args.child_name,
            "blank_seed": args.blank_seed,
            "attention_prototypes_per_group": args.attention_prototypes,
            "neuron_prototypes_per_neuron": args.neuron_prototypes,
            "requested_unique_parameters": args.target_parameters,
            "realized_unique_parameters": (
                result.target.resource_profile.unique_parameters
            ),
            "initial_blank_state_fingerprint": (run_result.initial_state_fingerprint),
            "folded_state_fingerprint": run_result.final_state_fingerprint,
            "blank_materialization": {
                "topology_loaded_from_immutable_checkpoint": (
                    result.target.adapter_report[
                        "topology_loaded_from_immutable_checkpoint"
                    ]
                ),
                "fresh_codebook_fit_performed": result.target.adapter_report[
                    "fresh_codebook_fit_performed"
                ],
                "blank_learned_parameters_parent_derived": (
                    result.target.adapter_report[
                        "blank_learned_parameters_parent_derived"
                    ]
                ),
            },
        },
        "execution": {
            "status": run_result.status,
            "plan_fingerprint": run_result.plan_fingerprint,
            "run_fingerprint": run_result.fingerprint,
            "action_count": len(result.published.compilation.plan.actions),
            "accepted_action_ids": list(run_result.accepted_action_ids),
            "snapshot_backend": run_result.whole_model_snapshot_backend,
            "snapshot_root": str(snapshot_root),
            "snapshot_cleanup_complete": True,
            "final_behavior_decision": final_decision.as_dict(),
            "fresh_reload_behavior_decision": (
                certificate.fresh_reload_decision.as_dict()
            ),
            "behavior_generator": generator.receipt(),
            "cached_candidate_preflight_accepted": True,
            "parent_forward_passes": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "autograd_grad_calls": 0,
        },
        "artifact": {
            "certificate_sha256": certificate.certificate_sha256,
            "artifact_tree_sha256": certificate.artifact_tree_sha256,
            "child_graph_fingerprint": certificate.child_graph_fingerprint,
            "child_state_fingerprint": certificate.child_state_fingerprint,
            "exact_unique_parameters": certificate.exact_unique_parameters,
            "standalone_child_directory": str(output / "child"),
            "atomic_publication": True,
            "fresh_reload_performed": True,
        },
        "io_contract": {
            "direct_local_parent_config_and_safetensors": True,
            "direct_local_tokenizer_json": True,
            "hub_download_used": False,
            "network_allowed": False,
            "training_used": False,
        },
    }
    _atomic_write_json(receipt_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fold one local dense Qwen3 parent into an authenticated standalone "
            "global-codebook child without training."
        )
    )
    parser.add_argument("--parent-model", type=Path, required=True)
    parser.add_argument("--parent-tokenizer", type=Path, required=True)
    parser.add_argument("--target-tokenizer", type=Path, required=True)
    parser.add_argument("--fold-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-manifest-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument("--cached-parent-eval", type=Path, required=True)
    parser.add_argument("--cached-parent-eval-sha256", type=_sha256)
    parser.add_argument("--cached-candidate-eval", type=Path, required=True)
    parser.add_argument(
        "--cached-candidate-eval-sha256",
        type=_sha256,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--parent-name", default="parent")
    parser.add_argument("--child-name", default="dynamic-qwen3-codebook-child")
    parser.add_argument(
        "--target-parameters",
        type=int,
        default=DEFAULT_TARGET_PARAMETERS,
    )
    parser.add_argument(
        "--maximum-parameter-bytes",
        type=int,
    )
    parser.add_argument(
        "--attention-prototypes",
        type=int,
        default=DEFAULT_ATTENTION_PROTOTYPES,
    )
    parser.add_argument(
        "--neuron-prototypes",
        type=int,
        default=DEFAULT_NEURON_PROTOTYPES,
    )
    parser.add_argument("--blank-seed", type=int, default=20_260_716)
    parser.add_argument("--parent-device", default="cuda:0")
    parser.add_argument("--child-device", default="cuda:1")
    parser.add_argument("--reload-device", default="cuda:0")
    parser.add_argument("--cache-device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--snapshot-chunk-mib", type=int, default=64)
    parser.add_argument("--behavior-subset-size", type=int, default=25)
    parser.add_argument("--behavior-subset-seed", type=int, default=20_260_716)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--maximum-prompt-tokens", type=int, default=512)
    parser.add_argument(
        "--fragment-minimum-characters",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--maximum-new-empty-responses",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--maximum-new-fragment-responses",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--minimum-parent-nonempty-retention",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--minimum-task-score-retention",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--maximum-new-generation-limit-saturations",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--maximum-repeated-bigram-share-regressions",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--maximum-repeated-bigram-share-delta",
        type=float,
        default=0.05,
    )
    parser.add_argument("--expected-parent-state-sha256", type=_sha256)
    parser.add_argument("--expected-parent-tokenizer-sha256", type=_sha256)
    parser.add_argument("--expected-target-tokenizer-sha256", type=_sha256)
    parser.add_argument(
        "--release-parent-before-reload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run(args)
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))
    return 0 if receipt.get("complete") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
