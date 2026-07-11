"""In-memory Qwen3 causal-LM runtime backed by global depth codebooks.

The conversion in this module is deliberately in place.  A multi-billion
parameter dense source cannot be copied merely to replace most of the copy
with shared storage.  Fitting finishes before mutation, then the existing
Qwen3 decoder blocks keep their residual topology, rotary attention runtime,
and layer-specific norms while their dense projections are replaced by
parameterless views into two globally owned registries.

Vocabulary transport is exact: every target-tokenizer row selects one unique
raw-token-identical parent row, and one parameter is shared by the input
embedding and output head.  No parent model or tokenizer is retained by the
resulting runtime.
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from neural_forge.hf_attention_codebook import (
    DEFAULT_SKETCH_DIM,
    GQAAttentionBundle,
    GlobalAttentionCodebook,
    fit_global_attention_codebook,
)
from neural_forge.hf_neuron_codebook import (
    DEFAULT_WEIGHT_SKETCH_SIZE,
    GatedMLPBundle,
    GlobalNeuronCodebook,
    fit_global_neuron_codebook,
)
from neural_forge.hf_tied_depth import TiedDepthError, exact_token_row_mapping


RUNTIME_SCHEMA = "neural-forge-qwen3-codebook-runtime-v1"
RUNTIME_METHOD = "exact-token-rows-plus-global-attention-and-neuron-codebooks"
CHECKPOINT_SCHEMA = "neural-forge-qwen3-codebook-checkpoint-v1"
_REPORT_ATTRIBUTE = "neural_forge_codebook_report"
_ATTENTION_ATTRIBUTE = "attention_codebook"
_NEURON_ATTRIBUTE = "neuron_codebook"
_PARENT_ROWS_BUFFER = "parent_token_rows"
_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
_LAYER_NORMS = (
    ("layer", "input_layernorm"),
    ("layer", "post_attention_layernorm"),
    ("attention", "q_norm"),
    ("attention", "k_norm"),
)

PermutationRows = Tensor | Sequence[Tensor | Sequence[int]]


class Qwen3CodebookError(ValueError):
    """A dense source cannot be converted to the codebook runtime."""


class Qwen3CodebookStateError(Qwen3CodebookError):
    """The converted runtime violates an ownership or accounting invariant."""


@dataclass(frozen=True, slots=True)
class Qwen3CodebookParameterReport:
    """Physical storage, logical dense execution, and index accounting."""

    schema: str
    method: str
    logical_layers: int
    residual_block_count: int
    layer_specific_norm_count: int
    hidden_size: int
    intermediate_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    attention_prototypes_per_group: int
    neuron_prototypes_per_neuron: int
    neuron_prototype_count: int
    source_vocab_size: int
    target_vocab_size: int
    token_row_fingerprint: str
    source_unique_parameter_count: int
    source_logical_parameter_count: int
    unique_parameter_count: int
    logical_parameter_count: int
    registered_logical_parameter_count: int
    attention_unique_parameter_count: int
    attention_logical_parameter_count: int
    neuron_unique_parameter_count: int
    neuron_logical_parameter_count: int
    retained_fixed_parameter_count: int
    retained_layer_specific_norm_parameter_count: int
    vocabulary_unique_parameter_count: int
    vocabulary_logical_parameter_count: int
    attention_index_bytes: int
    neuron_index_bytes: int
    token_row_index_bytes: int
    index_buffer_bytes: int
    full_logical_depth: bool
    exact_parent_token_rows: bool
    tied_input_output_embeddings: bool
    parameterless_layer_views: bool
    single_registry_owner_per_codebook: bool
    fitting_requires_grad: bool
    parent_runtime_reference_count: int

    @property
    def unique_learned_parameters(self) -> int:
        return self.unique_parameter_count

    @property
    def unique_learned_parameter_count(self) -> int:
        return self.unique_parameter_count

    @property
    def logical_dense_parameters(self) -> int:
        return self.logical_parameter_count

    @property
    def logical_dense_parameter_count(self) -> int:
        return self.logical_parameter_count

    @property
    def index_bytes(self) -> int:
        return self.index_buffer_bytes

    @property
    def parameters_saved(self) -> int:
        return self.logical_parameter_count - self.unique_parameter_count

    @property
    def logical_to_unique_ratio(self) -> float:
        return self.logical_parameter_count / self.unique_parameter_count

    @property
    def source_to_runtime_ratio(self) -> float:
        return self.source_unique_parameter_count / self.unique_parameter_count

    def as_dict(self) -> dict[str, int | float | str | bool]:
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result.update(
            {
                "unique_learned_parameters": self.unique_learned_parameters,
                "unique_learned_parameter_count": (self.unique_learned_parameter_count),
                "logical_dense_parameters": self.logical_dense_parameters,
                "logical_dense_parameter_count": (self.logical_dense_parameter_count),
                "index_bytes": self.index_bytes,
                "parameters_saved": self.parameters_saved,
                "logical_to_unique_ratio": self.logical_to_unique_ratio,
                "source_to_runtime_ratio": self.source_to_runtime_ratio,
            }
        )
        return result

    def __getitem__(self, key: str) -> int | float | str | bool:
        return self.as_dict()[key]


@dataclass(frozen=True, slots=True)
class _DenseQwen3Source:
    layers: nn.ModuleList
    embedding: nn.Embedding
    output: nn.Linear
    attention_bundles: tuple[GQAAttentionBundle, ...]
    neuron_bundles: tuple[GatedMLPBundle, ...]
    residual_block_ids: tuple[int, ...]
    norm_modules: tuple[nn.Module, ...]
    norm_parameter_count: int
    source_unique_parameters: int
    source_logical_parameters: int


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise Qwen3CodebookError(f"{name} must be a positive integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Qwen3CodebookError(f"{name} must be a positive integer.") from exc
    if result <= 0:
        raise Qwen3CodebookError(f"{name} must be a positive integer.")
    return result


def _parameter_counts(module: nn.Module) -> tuple[int, int]:
    unique = sum(int(parameter.numel()) for parameter in module.parameters())
    logical = sum(
        int(parameter.numel())
        for _, parameter in module.named_parameters(remove_duplicate=False)
    )
    return unique, logical


def _token_row_fingerprint(parent_rows: tuple[int, ...]) -> str:
    payload = json.dumps(
        parent_rows,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise Qwen3CodebookError("Checkpoint hashes require tensors.")
    cpu = value.detach().to(device="cpu").contiguous()
    payload = {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "bytes": hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safetensors_io() -> tuple[Any, Any]:
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise Qwen3CodebookError(
            "Compact Qwen3 codebook checkpoints require safetensors."
        ) from exc
    return load_file, save_file


def _qwen3_type() -> type[nn.Module]:
    try:
        from transformers import Qwen3ForCausalLM
    except ImportError as exc:
        raise Qwen3CodebookError(
            "Conversion requires Transformers with Qwen3 support."
        ) from exc
    return Qwen3ForCausalLM


def _module(value: Any, name: str) -> nn.Module:
    if not isinstance(value, nn.Module):
        raise Qwen3CodebookError(f"Dense Qwen3 source lacks module {name}.")
    return value


def _linear(value: Any, name: str) -> nn.Linear:
    if not isinstance(value, nn.Linear):
        raise Qwen3CodebookError(f"Dense Qwen3 source requires nn.Linear at {name}.")
    return value


def _layer_norm_modules(layer: nn.Module, layer_index: int) -> tuple[nn.Module, ...]:
    result: list[nn.Module] = []
    attention = _module(getattr(layer, "self_attn", None), "self_attn")
    for owner_name, norm_name in _LAYER_NORMS:
        owner = layer if owner_name == "layer" else attention
        result.append(
            _module(
                getattr(owner, norm_name, None),
                f"model.layers.{layer_index}.{norm_name}",
            )
        )
    return tuple(result)


def _validate_dense_source(model: nn.Module) -> _DenseQwen3Source:
    qwen_type = _qwen3_type()
    if not isinstance(model, qwen_type):
        raise Qwen3CodebookError(
            "model must be a Transformers Qwen3ForCausalLM instance."
        )
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "qwen3":
        raise Qwen3CodebookError("model.config.model_type must be 'qwen3'.")
    for attribute in (
        _REPORT_ATTRIBUTE,
        _ATTENTION_ATTRIBUTE,
        _NEURON_ATTRIBUTE,
        _PARENT_ROWS_BUFFER,
    ):
        if hasattr(model, attribute):
            raise Qwen3CodebookError("Model is already a codebook runtime.")

    backbone = _module(getattr(model, "model", None), "model")
    layers = getattr(backbone, "layers", None)
    expected_layers = _positive_int(
        getattr(config, "num_hidden_layers", None),
        "config.num_hidden_layers",
    )
    if not isinstance(layers, nn.ModuleList) or len(layers) != expected_layers:
        raise Qwen3CodebookError(
            "model.layers must contain every configured Qwen3 decoder block."
        )

    embedding = model.get_input_embeddings()
    output = model.get_output_embeddings()
    if not isinstance(embedding, nn.Embedding) or not isinstance(output, nn.Linear):
        raise Qwen3CodebookError(
            "Dense source must expose nn.Embedding input and nn.Linear output."
        )
    if embedding.weight is not output.weight:
        raise Qwen3CodebookError(
            "Dense source input and output vocabulary weights must be tied."
        )
    if output.bias is not None:
        raise Qwen3CodebookError("Qwen3 output head must be bias-free.")
    if embedding.weight.device.type == "meta":
        raise Qwen3CodebookError("Dense source weights cannot be on the meta device.")

    query_heads = _positive_int(
        getattr(config, "num_attention_heads", None),
        "config.num_attention_heads",
    )
    key_value_heads = _positive_int(
        getattr(config, "num_key_value_heads", None),
        "config.num_key_value_heads",
    )
    head_dim = _positive_int(getattr(config, "head_dim", None), "config.head_dim")
    attention_bundles: list[GQAAttentionBundle] = []
    neuron_bundles: list[GatedMLPBundle] = []
    norm_modules: list[nn.Module] = []
    for layer_index, layer in enumerate(layers):
        attention = _module(
            getattr(layer, "self_attn", None),
            f"model.layers.{layer_index}.self_attn",
        )
        mlp = _module(
            getattr(layer, "mlp", None),
            f"model.layers.{layer_index}.mlp",
        )
        projections = {
            name: _linear(
                getattr(attention, name, None),
                f"model.layers.{layer_index}.self_attn.{name}",
            )
            for name in _ATTENTION_PROJECTIONS
        }
        attention_bundles.append(
            GQAAttentionBundle.from_linears(
                projections["q_proj"],
                projections["k_proj"],
                projections["v_proj"],
                projections["o_proj"],
                query_heads=query_heads,
                key_value_heads=key_value_heads,
                head_dim=head_dim,
            )
        )
        neuron_bundles.append(
            GatedMLPBundle.from_linears(
                _linear(
                    getattr(mlp, "gate_proj", None),
                    f"model.layers.{layer_index}.mlp.gate_proj",
                ),
                _linear(
                    getattr(mlp, "up_proj", None),
                    f"model.layers.{layer_index}.mlp.up_proj",
                ),
                _linear(
                    getattr(mlp, "down_proj", None),
                    f"model.layers.{layer_index}.mlp.down_proj",
                ),
            )
        )
        norm_modules.extend(_layer_norm_modules(layer, layer_index))

    source_unique, source_logical = _parameter_counts(model)
    norm_parameter_count = sum(
        int(parameter.numel())
        for norm in norm_modules
        for parameter in norm.parameters()
    )
    return _DenseQwen3Source(
        layers=layers,
        embedding=embedding,
        output=output,
        attention_bundles=tuple(attention_bundles),
        neuron_bundles=tuple(neuron_bundles),
        residual_block_ids=tuple(id(layer) for layer in layers),
        norm_modules=tuple(norm_modules),
        norm_parameter_count=norm_parameter_count,
        source_unique_parameters=source_unique,
        source_logical_parameters=source_logical,
    )


def _special_token_ids(tokenizer: Any, target_size: int) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, name, None)
        if value is None:
            result[name] = None
            continue
        if isinstance(value, bool):
            raise Qwen3CodebookError(f"target_tokenizer.{name} must be an integer.")
        try:
            token_id = operator.index(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise Qwen3CodebookError(
                f"target_tokenizer.{name} must be an integer."
            ) from exc
        if not 0 <= token_id < target_size:
            raise Qwen3CodebookError(
                f"target_tokenizer.{name} is outside the target vocabulary."
            )
        result[name] = token_id
    return result


def _exact_target_vocabulary(
    source: _DenseQwen3Source,
    parent_tokenizer: Any,
    target_tokenizer: Any,
) -> tuple[nn.Embedding, nn.Linear, tuple[int, ...], dict[str, int | None]]:
    try:
        parent_rows = exact_token_row_mapping(parent_tokenizer, target_tokenizer)
    except TiedDepthError as exc:
        raise Qwen3CodebookError(
            f"Strict target-tokenizer row mapping failed: {exc}"
        ) from exc
    if not parent_rows or max(parent_rows) >= source.embedding.num_embeddings:
        raise Qwen3CodebookError(
            "Strict target-tokenizer rows exceed the parent embedding table."
        )
    target_size = len(parent_rows)
    special_ids = _special_token_ids(target_tokenizer, target_size)
    row_indices = torch.tensor(
        parent_rows,
        device=source.embedding.weight.device,
        dtype=torch.long,
    )
    selected = source.embedding.weight.detach().index_select(0, row_indices)
    if not bool(torch.isfinite(selected).all()):
        raise Qwen3CodebookError(
            "Selected parent vocabulary rows contain NaN or infinity."
        )

    target_embedding = nn.Embedding(
        target_size,
        source.embedding.embedding_dim,
        padding_idx=special_ids["pad_token_id"],
        max_norm=source.embedding.max_norm,
        norm_type=source.embedding.norm_type,
        scale_grad_by_freq=source.embedding.scale_grad_by_freq,
        sparse=source.embedding.sparse,
        device="meta",
        dtype=selected.dtype,
    )
    weight = nn.Parameter(
        selected,
        requires_grad=source.embedding.weight.requires_grad,
    )
    target_embedding.weight = weight
    target_output = nn.Linear(
        source.output.in_features,
        target_size,
        bias=False,
        device="meta",
        dtype=selected.dtype,
    )
    target_output.weight = weight
    target_embedding.train(source.embedding.training)
    target_output.train(source.output.training)
    return target_embedding, target_output, parent_rows, special_ids


def _resolve_group_permutations(
    group_permutations: PermutationRows | None,
    gqa_group_permutations: PermutationRows | None,
) -> PermutationRows | None:
    if group_permutations is not None and gqa_group_permutations is not None:
        raise Qwen3CodebookError(
            "Specify group_permutations or gqa_group_permutations, not both."
        )
    return (
        group_permutations if group_permutations is not None else gqa_group_permutations
    )


def _update_vocabulary_metadata(
    model: nn.Module,
    target_size: int,
    special_ids: Mapping[str, int | None],
) -> None:
    model.config.vocab_size = target_size
    model.config.tie_word_embeddings = True
    if hasattr(model, "vocab_size"):
        model.vocab_size = target_size
    backbone = getattr(model, "model", None)
    if backbone is not None:
        if hasattr(backbone, "vocab_size"):
            backbone.vocab_size = target_size
        if hasattr(backbone, "padding_idx"):
            backbone.padding_idx = special_ids["pad_token_id"]
    generation_config = getattr(model, "generation_config", None)
    for name, token_id in special_ids.items():
        setattr(model.config, name, token_id)
        if generation_config is not None:
            setattr(generation_config, name, token_id)


def _registered_owner_count(model: nn.Module, target: nn.Module) -> int:
    return sum(
        module is target for _, module in model.named_modules(remove_duplicate=False)
    )


def _all_views_are_parameterless(
    attention: GlobalAttentionCodebook,
    neurons: GlobalNeuronCodebook,
) -> bool:
    return all(
        not list(view.parameters()) and not view.state_dict()
        for view in (*attention.layers, *neurons.layers)
    )


def _rewire_layers(
    source: _DenseQwen3Source,
    attention: GlobalAttentionCodebook,
    neurons: GlobalNeuronCodebook,
) -> None:
    for layer_index, layer in enumerate(source.layers):
        attention_view = attention[layer_index]
        self_attention = layer.self_attn
        self_attention.q_proj = attention_view.q_proj
        self_attention.k_proj = attention_view.k_proj
        self_attention.v_proj = attention_view.v_proj
        self_attention.o_proj = attention_view.o_proj
        layer.mlp = neurons[layer_index]


def _assert_preserved_topology(
    source: _DenseQwen3Source,
    attention: GlobalAttentionCodebook,
    neurons: GlobalNeuronCodebook,
) -> None:
    if tuple(id(layer) for layer in source.layers) != source.residual_block_ids:
        raise Qwen3CodebookStateError("Conversion replaced a Qwen3 residual block.")
    current_norms = tuple(
        norm
        for layer_index, layer in enumerate(source.layers)
        for norm in _layer_norm_modules(layer, layer_index)
    )
    if any(
        current is not original
        for current, original in zip(current_norms, source.norm_modules)
    ):
        raise Qwen3CodebookStateError(
            "Conversion replaced a layer-specific Qwen3 norm."
        )
    for layer_index, layer in enumerate(source.layers):
        attention_view = attention[layer_index]
        if any(
            getattr(layer.self_attn, name) is not getattr(attention_view, name)
            for name in _ATTENTION_PROJECTIONS
        ):
            raise Qwen3CodebookStateError(
                "A Qwen3 attention projection is not its codebook view."
            )
        if layer.mlp is not neurons[layer_index]:
            raise Qwen3CodebookStateError(
                "A Qwen3 MLP is not its neuron-codebook view."
            )


def _runtime_report(
    model: nn.Module,
    source: _DenseQwen3Source,
    attention: GlobalAttentionCodebook,
    neurons: GlobalNeuronCodebook,
    parent_rows: tuple[int, ...],
) -> Qwen3CodebookParameterReport:
    attention_report = attention.report
    neuron_report = neurons.report
    neuron_prototypes_per_neuron = neuron_report.prototypes_per_neuron
    if neuron_prototypes_per_neuron is None:
        raise Qwen3CodebookStateError(
            "Qwen3 neuron codebook lacks uniform per-neuron depth capacity."
        )
    unique_parameters, registered_logical = _parameter_counts(model)
    vocabulary_unique = int(model.get_input_embeddings().weight.numel())
    vocabulary_logical = 2 * vocabulary_unique
    retained_fixed = (
        unique_parameters
        - attention_report.unique_learned_parameters
        - neuron_report.unique_learned_parameters
        - vocabulary_unique
    )
    if retained_fixed < source.norm_parameter_count:
        raise Qwen3CodebookStateError(
            "Runtime parameter accounting lost layer-specific norm parameters."
        )
    logical_parameters = (
        retained_fixed
        + attention_report.logical_dense_parameters
        + neuron_report.logical_dense_parameters
        + vocabulary_logical
    )
    token_row_index_bytes = (
        len(parent_rows) * torch.tensor([], dtype=torch.int32).element_size()
    )
    index_bytes = (
        attention_report.index_buffer_bytes
        + neuron_report.index_buffer_bytes
        + token_row_index_bytes
    )
    attention_owners = _registered_owner_count(model, attention.registry)
    neuron_owners = _registered_owner_count(model, neurons.registry)
    owner_invariant = attention_owners == neuron_owners == 1
    parameterless = _all_views_are_parameterless(attention, neurons)
    if not owner_invariant:
        raise Qwen3CodebookStateError(
            "Each codebook registry must have exactly one registered owner."
        )
    if not parameterless:
        raise Qwen3CodebookStateError(
            "Every logical codebook layer view must be parameterless."
        )
    return Qwen3CodebookParameterReport(
        schema=RUNTIME_SCHEMA,
        method=RUNTIME_METHOD,
        logical_layers=len(source.layers),
        residual_block_count=len(source.layers),
        layer_specific_norm_count=len(source.norm_modules),
        hidden_size=attention_report.input_width,
        intermediate_size=neuron_report.intermediate_width,
        query_heads=attention_report.query_heads,
        key_value_heads=attention_report.key_value_heads,
        head_dim=attention_report.head_dim,
        attention_prototypes_per_group=(attention_report.prototypes_per_group),
        neuron_prototypes_per_neuron=neuron_prototypes_per_neuron,
        neuron_prototype_count=neuron_report.prototype_count,
        source_vocab_size=source.embedding.num_embeddings,
        target_vocab_size=len(parent_rows),
        token_row_fingerprint=_token_row_fingerprint(parent_rows),
        source_unique_parameter_count=source.source_unique_parameters,
        source_logical_parameter_count=source.source_logical_parameters,
        unique_parameter_count=unique_parameters,
        logical_parameter_count=logical_parameters,
        registered_logical_parameter_count=registered_logical,
        attention_unique_parameter_count=(attention_report.unique_learned_parameters),
        attention_logical_parameter_count=(attention_report.logical_dense_parameters),
        neuron_unique_parameter_count=neuron_report.unique_learned_parameters,
        neuron_logical_parameter_count=neuron_report.logical_dense_parameters,
        retained_fixed_parameter_count=retained_fixed,
        retained_layer_specific_norm_parameter_count=(source.norm_parameter_count),
        vocabulary_unique_parameter_count=vocabulary_unique,
        vocabulary_logical_parameter_count=vocabulary_logical,
        attention_index_bytes=attention_report.index_buffer_bytes,
        neuron_index_bytes=neuron_report.index_buffer_bytes,
        token_row_index_bytes=token_row_index_bytes,
        index_buffer_bytes=index_bytes,
        full_logical_depth=True,
        exact_parent_token_rows=True,
        tied_input_output_embeddings=True,
        parameterless_layer_views=parameterless,
        single_registry_owner_per_codebook=owner_invariant,
        fitting_requires_grad=False,
        parent_runtime_reference_count=0,
    )


def prepare_qwen3_codebook_model(
    model: nn.Module,
    parent_tokenizer: Any,
    target_tokenizer: Any,
    *,
    neuron_permutations: PermutationRows | None = None,
    group_permutations: PermutationRows | None = None,
    gqa_group_permutations: PermutationRows | None = None,
    attention_prototypes_per_group: int = 1,
    neuron_prototypes_per_neuron: int = 1,
    attention_use_scales: bool = False,
    neuron_use_scales: bool = False,
    attention_sketch_dim: int = DEFAULT_SKETCH_DIM,
    neuron_weight_sketch_size: int = DEFAULT_WEIGHT_SKETCH_SIZE,
) -> nn.Module:
    """Convert a tied dense Qwen3 causal LM to the standalone runtime in place.

    ``neuron_permutations[layer, canonical_neuron]`` and
    ``group_permutations[layer, canonical_kv_group]`` map canonical positions
    to physical source positions.  Rows must be bijections; omitted matrices
    are identities.  Neurons and attention can retain one or more real depth
    prototypes per aligned canonical neuron or GQA group, respectively.

    Fitting and exact vocabulary selection run under ``torch.no_grad()`` and
    finish before the first source module is replaced.  The returned object is
    the same Hugging Face model instance, now with no dense parent projection
    storage or parent runtime reference.
    """

    prototypes_per_group = _positive_int(
        attention_prototypes_per_group,
        "attention_prototypes_per_group",
    )
    prototypes_per_neuron = _positive_int(
        neuron_prototypes_per_neuron,
        "neuron_prototypes_per_neuron",
    )
    attention_sketch = _positive_int(
        attention_sketch_dim,
        "attention_sketch_dim",
    )
    neuron_sketch = _positive_int(
        neuron_weight_sketch_size,
        "neuron_weight_sketch_size",
    )
    gqa_permutations = _resolve_group_permutations(
        group_permutations,
        gqa_group_permutations,
    )
    source = _validate_dense_source(model)
    config = model.config
    was_training = model.training

    with torch.no_grad():
        # Exact tokenizer/vocabulary validation is cheap and must precede fitting.
        target_embedding, target_output, parent_rows, special_ids = (
            _exact_target_vocabulary(
                source,
                parent_tokenizer,
                target_tokenizer,
            )
        )
        attention = fit_global_attention_codebook(
            source.attention_bundles,
            prototypes_per_group=prototypes_per_group,
            query_heads=int(config.num_attention_heads),
            key_value_heads=int(config.num_key_value_heads),
            head_dim=int(config.head_dim),
            group_permutations=gqa_permutations,
            use_scales=attention_use_scales,
            sketch_dim=attention_sketch,
        )
        neurons = fit_global_neuron_codebook(
            source.neuron_bundles,
            prototypes_per_neuron=prototypes_per_neuron,
            activation=str(config.hidden_act),
            use_scales=neuron_use_scales,
            weight_sketch_size=neuron_sketch,
            assignment_strategy="aligned_index",
            neuron_permutations=neuron_permutations,
        )
        setattr(model, _ATTENTION_ATTRIBUTE, attention)
        setattr(model, _NEURON_ATTRIBUTE, neurons)
        _rewire_layers(source, attention, neurons)
        model.set_input_embeddings(target_embedding)
        model.set_output_embeddings(target_output)
        model.register_buffer(
            _PARENT_ROWS_BUFFER,
            torch.tensor(parent_rows, dtype=torch.int32),
            persistent=True,
        )
        _update_vocabulary_metadata(model, len(parent_rows), special_ids)
        model.train(was_training)
        _assert_preserved_topology(source, attention, neurons)
        if (
            model.get_input_embeddings().weight
            is not model.get_output_embeddings().weight
        ):
            raise Qwen3CodebookStateError(
                "Converted input embedding and output head lost their weight tie."
            )
        report = _runtime_report(
            model,
            source,
            attention,
            neurons,
            parent_rows,
        )
        setattr(model, _REPORT_ATTRIBUTE, report)
    return model


def qwen3_codebook_parameter_report(
    model: nn.Module,
) -> Qwen3CodebookParameterReport:
    """Return the verified report attached by the in-memory conversion."""

    report = getattr(model, _REPORT_ATTRIBUTE, None)
    if not isinstance(report, Qwen3CodebookParameterReport):
        raise Qwen3CodebookError("Model is not a prepared Qwen3 codebook runtime.")
    return report


def _checkpoint_state(model: nn.Module) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, value in model.state_dict().items():
        if not isinstance(value, Tensor) or value.device.type == "meta":
            raise Qwen3CodebookStateError(
                f"Checkpoint state {name!r} is not a materialized tensor."
            )
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise Qwen3CodebookStateError(
                f"Checkpoint state {name!r} contains NaN or infinity."
            )
        # safetensors rejects tied storage; cloned values preserve the explicit tie
        # reconstruction step while keeping the artifact pickle-free.
        state[name] = value.detach().to(device="cpu").contiguous().clone()
    return state


def _checkpoint_grouped_keys(state: Mapping[str, Tensor]) -> dict[str, list[str]]:
    return {
        "attention_registry": sorted(
            name for name in state if name.startswith("attention_codebook.registry.")
        ),
        "neuron_registry": sorted(
            name for name in state if name.startswith("neuron_codebook.registry.")
        ),
        "vocabulary": sorted(
            name
            for name in state
            if name in {"model.embed_tokens.weight", "lm_head.weight", _PARENT_ROWS_BUFFER}
        ),
        "norms_and_fixed": sorted(
            name
            for name in state
            if not name.startswith(("attention_codebook.", "neuron_codebook."))
            and name
            not in {"model.embed_tokens.weight", "lm_head.weight", _PARENT_ROWS_BUFFER}
        ),
    }


def _codebook_checkpoint_hashes(state: Mapping[str, Tensor]) -> dict[str, dict[str, str]]:
    def selected(suffix: str) -> dict[str, str]:
        return {
            name: _tensor_sha256(value)
            for name, value in sorted(state.items())
            if name.endswith(suffix)
        }

    return {
        "assignment_hashes": {
            **selected(".assignments"),
            **selected(".group_permutations"),
        },
        "prototype_source_hashes": selected(".prototype_source_indices"),
    }


def save_qwen3_codebook_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    tokenizer_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically save a standalone, pickle-free Qwen3 codebook runtime."""

    report = qwen3_codebook_parameter_report(model)
    if not isinstance(tokenizer_contract, Mapping) or not tokenizer_contract:
        raise Qwen3CodebookError("tokenizer_contract must be a non-empty mapping.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    state = _checkpoint_state(model)
    grouped_keys = _checkpoint_grouped_keys(state)
    required_groups = ("attention_registry", "neuron_registry", "vocabulary")
    if any(not grouped_keys[name] for name in required_groups):
        raise Qwen3CodebookStateError("Checkpoint is missing codebook or vocabulary state.")
    load_file, save_file = _safetensors_io()
    del load_file
    tensor_path = destination / "model.safetensors"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination,
        prefix=".model.safetensors.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(state, str(temporary), metadata={"format": CHECKPOINT_SCHEMA})
        # Windows rejects fsync on a read-only descriptor even though POSIX accepts it.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, tensor_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    config_payload = {
        "model_config": model.config.to_dict(),
        "generation_config": (
            model.generation_config.to_dict()
            if getattr(model, "generation_config", None) is not None
            else None
        ),
        "training": bool(model.training),
    }
    config_path = destination / "config.json"
    tokenizer_path = destination / "tokenizer_contract.json"
    _atomic_write_json(config_path, config_payload)
    _atomic_write_json(tokenizer_path, dict(tokenizer_contract))
    hashes = _codebook_checkpoint_hashes(state)
    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "format": "safetensors",
        "tensor_file": tensor_path.name,
        "config_file": config_path.name,
        "tokenizer_contract_file": tokenizer_path.name,
        "files": {
            tensor_path.name: _sha256_file(tensor_path),
            config_path.name: _sha256_file(config_path),
            tokenizer_path.name: _sha256_file(tokenizer_path),
        },
        "state_key_sha256": hashlib.sha256(
            _canonical_json_bytes(sorted(state))
        ).hexdigest(),
        "state_tensor_count": len(state),
        "state_groups": grouped_keys,
        "parameter_report": report.as_dict(),
        "token_row_fingerprint": report.token_row_fingerprint,
        **hashes,
    }
    _atomic_write_json(destination / "manifest.json", manifest)
    return manifest


def _load_checkpoint_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Qwen3CodebookError(f"Cannot load checkpoint {label}.") from exc
    if not isinstance(value, dict):
        raise Qwen3CodebookError(f"Checkpoint {label} must be a JSON object.")
    return value


def _checkpoint_substate(
    state: Mapping[str, Tensor], prefix: str
) -> dict[str, Tensor]:
    return {
        name[len(prefix) :]: value
        for name, value in state.items()
        if name.startswith(prefix)
    }


def _materialize_checkpoint_meta_buffers(model: nn.Module) -> None:
    """Restore small non-persistent buffers omitted from safetensors state."""

    for module in model.modules():
        for name, value in tuple(module._buffers.items()):
            if not isinstance(value, Tensor) or value.device.type != "meta":
                continue
            replacement: Tensor | None = None
            config = getattr(module, "config", None)
            if config is not None:
                try:
                    fresh_module = type(module)(config, device="cpu")
                except (TypeError, ValueError, RuntimeError):
                    fresh_module = None
                fresh_value = (
                    getattr(fresh_module, name, None)
                    if fresh_module is not None
                    else None
                )
                if isinstance(fresh_value, Tensor):
                    replacement = fresh_value.detach().clone()
            if name == "inv_freq" and callable(getattr(module, "rope_init_fn", None)):
                result = module.rope_init_fn(module.config, device="cpu")
                if isinstance(result, tuple) and isinstance(result[0], Tensor):
                    replacement = result[0]
            if replacement is None:
                raise Qwen3CodebookStateError(
                    f"Cannot reconstruct derived checkpoint buffer {name!r} "
                    f"for {type(module).__name__}."
                )
            module._buffers[name] = replacement
            if name == "inv_freq" and isinstance(
                getattr(module, "original_inv_freq", None), Tensor
            ):
                module.original_inv_freq = replacement.detach().clone()


def load_qwen3_codebook_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load a compact runtime from safetensors without a dense parent checkpoint."""

    destination = Path(path)
    manifest = _load_checkpoint_json(destination / "manifest.json", "manifest")
    if manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise Qwen3CodebookError("Checkpoint schema is not a Qwen3 codebook artifact.")
    for name in ("tensor_file", "config_file", "tokenizer_contract_file"):
        if not isinstance(manifest.get(name), str):
            raise Qwen3CodebookError(f"Checkpoint manifest lacks {name}.")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise Qwen3CodebookError("Checkpoint manifest lacks file hashes.")
    for filename in (
        manifest["tensor_file"],
        manifest["config_file"],
        manifest["tokenizer_contract_file"],
    ):
        candidate = destination / filename
        if candidate.parent != destination or not candidate.is_file():
            raise Qwen3CodebookError("Checkpoint manifest has an invalid artifact path.")
        if files.get(filename) != _sha256_file(candidate):
            raise Qwen3CodebookError(f"Checkpoint hash mismatch for {filename}.")
    config_payload = _load_checkpoint_json(
        destination / manifest["config_file"], "config"
    )
    _load_checkpoint_json(destination / manifest["tokenizer_contract_file"], "tokenizer contract")
    model_config = config_payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise Qwen3CodebookError("Checkpoint config lacks a model_config object.")
    load_file, _save_file = _safetensors_io()
    state = load_file(str(destination / manifest["tensor_file"]), device="cpu")
    if not isinstance(state, Mapping) or not state:
        raise Qwen3CodebookError("Checkpoint safetensors state is empty.")
    if hashlib.sha256(_canonical_json_bytes(sorted(state))).hexdigest() != manifest.get(
        "state_key_sha256"
    ):
        raise Qwen3CodebookError("Checkpoint state key manifest hash mismatches.")
    hashes = _codebook_checkpoint_hashes(state)
    if (
        hashes["assignment_hashes"] != manifest.get("assignment_hashes")
        or hashes["prototype_source_hashes"] != manifest.get("prototype_source_hashes")
    ):
        raise Qwen3CodebookError("Checkpoint assignment/provenance hashes mismatch.")
    attention = GlobalAttentionCodebook.from_state_dict(
        _checkpoint_substate(state, "attention_codebook.")
    )
    neurons = GlobalNeuronCodebook.from_state_dict(
        _checkpoint_substate(state, "neuron_codebook.")
    )
    if attention.logical_layers != neurons.logical_layers:
        raise Qwen3CodebookStateError("Checkpoint codebook depths do not agree.")
    try:
        from transformers import GenerationConfig, Qwen3Config
    except ImportError as exc:
        raise Qwen3CodebookError(
            "Loading requires Transformers with Qwen3 support."
        ) from exc
    config = Qwen3Config.from_dict(dict(model_config))
    qwen_type = _qwen3_type()
    with torch.device("meta"):
        model = qwen_type(config)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) != attention.logical_layers:
        raise Qwen3CodebookStateError("Checkpoint Qwen3 topology does not match codebooks.")
    setattr(model, _ATTENTION_ATTRIBUTE, attention)
    setattr(model, _NEURON_ATTRIBUTE, neurons)
    for layer_index, layer in enumerate(layers):
        attention_view = attention[layer_index]
        layer.self_attn.q_proj = attention_view.q_proj
        layer.self_attn.k_proj = attention_view.k_proj
        layer.self_attn.v_proj = attention_view.v_proj
        layer.self_attn.o_proj = attention_view.o_proj
        layer.mlp = neurons[layer_index]
    embedding_weight = state.get("model.embed_tokens.weight")
    parent_rows = state.get(_PARENT_ROWS_BUFFER)
    if not isinstance(embedding_weight, Tensor) or not isinstance(parent_rows, Tensor):
        raise Qwen3CodebookError("Checkpoint lacks target vocabulary tensors.")
    embedding = nn.Embedding(
        int(embedding_weight.shape[0]),
        int(embedding_weight.shape[1]),
        padding_idx=getattr(config, "pad_token_id", None),
        device="meta",
        dtype=embedding_weight.dtype,
    )
    embedding.weight = nn.Parameter(embedding_weight.detach().clone())
    output = nn.Linear(
        int(embedding_weight.shape[1]),
        int(embedding_weight.shape[0]),
        bias=False,
        device="meta",
        dtype=embedding_weight.dtype,
    )
    output.weight = embedding.weight
    model.set_input_embeddings(embedding)
    model.set_output_embeddings(output)
    model.register_buffer(
        _PARENT_ROWS_BUFFER,
        parent_rows.detach().to(dtype=torch.int32).clone(),
        persistent=True,
    )
    special_ids = {
        name: getattr(config, name, None)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id")
    }
    _update_vocabulary_metadata(model, int(embedding_weight.shape[0]), special_ids)
    incompatible = model.load_state_dict(dict(state), strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise Qwen3CodebookStateError("Checkpoint state does not exactly match runtime.")
    model.get_output_embeddings().weight = model.get_input_embeddings().weight
    generation_payload = config_payload.get("generation_config")
    if isinstance(generation_payload, Mapping):
        model.generation_config = GenerationConfig.from_dict(dict(generation_payload))
    saved_report = manifest.get("parameter_report")
    if not isinstance(saved_report, Mapping):
        raise Qwen3CodebookError("Checkpoint manifest lacks parameter accounting.")
    report = Qwen3CodebookParameterReport(
        **{field.name: saved_report[field.name] for field in fields(Qwen3CodebookParameterReport)}
    )
    setattr(model, _REPORT_ATTRIBUTE, report)
    model.train(bool(config_payload.get("training", False)))
    _materialize_checkpoint_meta_buffers(model)
    meta_parameters = [
        name for name, parameter in model.named_parameters() if parameter.device.type == "meta"
    ]
    if meta_parameters:
        raise Qwen3CodebookStateError(
            f"Checkpoint reload left meta parameters: {meta_parameters[:5]}."
        )
    model.to(device)
    if qwen3_codebook_parameter_report(model) != report:
        raise Qwen3CodebookStateError("Checkpoint parameter accounting changed on load.")
    return model


build_qwen3_codebook_model = prepare_qwen3_codebook_model
fit_qwen3_codebook_model = prepare_qwen3_codebook_model


__all__ = [
    "RUNTIME_METHOD",
    "RUNTIME_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "Qwen3CodebookError",
    "Qwen3CodebookParameterReport",
    "Qwen3CodebookStateError",
    "build_qwen3_codebook_model",
    "fit_qwen3_codebook_model",
    "load_qwen3_codebook_checkpoint",
    "prepare_qwen3_codebook_model",
    "qwen3_codebook_parameter_report",
    "save_qwen3_codebook_checkpoint",
]
