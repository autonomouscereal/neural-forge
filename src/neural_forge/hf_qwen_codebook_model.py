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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
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
        target_embedding, target_output, parent_rows, special_ids = (
            _exact_target_vocabulary(
                source,
                parent_tokenizer,
                target_tokenizer,
            )
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


build_qwen3_codebook_model = prepare_qwen3_codebook_model
fit_qwen3_codebook_model = prepare_qwen3_codebook_model


__all__ = [
    "RUNTIME_METHOD",
    "RUNTIME_SCHEMA",
    "Qwen3CodebookError",
    "Qwen3CodebookParameterReport",
    "Qwen3CodebookStateError",
    "build_qwen3_codebook_model",
    "fit_qwen3_codebook_model",
    "prepare_qwen3_codebook_model",
    "qwen3_codebook_parameter_report",
]
