"""Dynamic-target adapter for strict local Hugging Face causal blanks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .dynamic_circuit_ir import CircuitKind, ModelCircuitGraph, canonical_sha256
from .dynamic_parameter_frontier import (
    COMPONENT_ALLOCATION_POLICY_CONFIG_KEY,
    COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY,
    COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY,
    DENSE_TOPOLOGY_CATALOG_CONFIG_KEY,
    HF_LLAMA_CATALOG_ID,
    HF_LLAMA_ORDERED_SOURCE_SPANS_CONFIG_KEY,
    HF_LLAMA_PARAMETER_ALIAS_POLICY_CONFIG_KEY,
    HF_LLAMA_PARAMETER_ALIAS_POLICY_NONE,
    HF_LLAMA_SOURCE_FAMILY_CONFIG_KEY,
    HF_LLAMA_SOURCE_LAYER_COUNT_CONFIG_KEY,
    HF_LLAMA_TARGET_SCHEMA,
    HF_LLAMA_TARGET_SCHEMA_CONFIG_KEY,
    component_allocation_topology_fingerprint,
)
from .dynamic_target_synthesis import (
    AdapterMaterialization,
    DynamicTargetRequest,
    DynamicTargetSynthesisError,
)
from .hf_causal_blank_target import (
    HFCausalBlankTargetError,
    materialize_hf_causal_blank_target,
    normalize_hf_causal_blank_target_configuration,
)
from .hf_tokenizer_bridge import (
    CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY,
    PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY,
    TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY,
    TOKENIZER_BRIDGE_CONFIG_KEYS,
    TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY,
)


class HFCausalBlankTargetAdapterError(DynamicTargetSynthesisError):
    """The HF causal target adapter rejected an unsafe blank contract."""


_CATALOG_PROVENANCE_KEYS = frozenset(
    {
        DENSE_TOPOLOGY_CATALOG_CONFIG_KEY,
        HF_LLAMA_TARGET_SCHEMA_CONFIG_KEY,
        HF_LLAMA_PARAMETER_ALIAS_POLICY_CONFIG_KEY,
        HF_LLAMA_ORDERED_SOURCE_SPANS_CONFIG_KEY,
        HF_LLAMA_SOURCE_FAMILY_CONFIG_KEY,
        HF_LLAMA_SOURCE_LAYER_COUNT_CONFIG_KEY,
    }
)
_RECEIPT_PROVENANCE_KEYS = frozenset(
    {
        COMPONENT_ALLOCATION_POLICY_CONFIG_KEY,
        COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY,
        COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY,
    }
)
_ALLOWED_COMPILER_PROVENANCE_KEYS = (
    _CATALOG_PROVENANCE_KEYS
    | _RECEIPT_PROVENANCE_KEYS
    | TOKENIZER_BRIDGE_CONFIG_KEYS
)


def _constructor_configuration(
    configuration: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate compiler provenance seals from model-constructor keys."""

    constructor: dict[str, Any] = {}
    compiler: dict[str, Any] = {}
    for key, value in configuration.items():
        if key.startswith("_neural_forge_"):
            if key not in _ALLOWED_COMPILER_PROVENANCE_KEYS:
                raise HFCausalBlankTargetAdapterError(
                    f"Unknown HF causal compiler-provenance key {key!r}."
                )
            compiler[key] = value
        else:
            constructor[key] = value
    return constructor, compiler


def _decode_source_spans(value: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, str) or not value:
        raise HFCausalBlankTargetAdapterError(
            "Llama catalog source spans must be a nonempty canonical string."
        )
    try:
        spans = tuple(
            tuple(int(item) for item in span.split(","))
            for span in value.split(";")
        )
    except ValueError as exc:
        raise HFCausalBlankTargetAdapterError(
            "Llama catalog source spans contain a non-integer index."
        ) from exc
    canonical = ";".join(
        ",".join(str(index) for index in span) for span in spans
    )
    flattened = tuple(index for span in spans for index in span)
    if (
        canonical != value
        or any(not span for span in spans)
        or any(index < 0 for index in flattened)
        or flattened != tuple(range(len(flattened)))
    ):
        raise HFCausalBlankTargetAdapterError(
            "Llama catalog source spans must cover source layers once in order."
        )
    return spans


def _sha256_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HFCausalBlankTargetAdapterError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _source_vocabulary_size(source: ModelCircuitGraph) -> int:
    embeddings = source.nodes_of_kind(CircuitKind.TOKEN_EMBEDDING)
    heads = source.nodes_of_kind(CircuitKind.OUTPUT_HEAD)
    if len(embeddings) != 1 or len(heads) != 1:
        raise HFCausalBlankTargetAdapterError(
            "Tokenizer-bridge provenance requires one source token embedding "
            "and one source output head."
        )
    embedding_vocab = dict(embeddings[0].attributes).get("vocabulary_size")
    head_vocab = dict(heads[0].attributes).get("vocabulary_size")
    if (
        isinstance(embedding_vocab, bool)
        or not isinstance(embedding_vocab, int)
        or embedding_vocab <= 0
        or head_vocab != embedding_vocab
    ):
        raise HFCausalBlankTargetAdapterError(
            "Tokenizer-bridge provenance disagrees with the source vocabulary."
        )
    return embedding_vocab


def _validate_compiler_provenance(
    *,
    request: DynamicTargetRequest,
    constructor: Mapping[str, Any],
    compiler: Mapping[str, Any],
    parent_graphs: Mapping[str, ModelCircuitGraph],
) -> dict[str, str | bool]:
    if not compiler:
        if request.component_allocation_receipt is not None:
            raise HFCausalBlankTargetAdapterError(
                "A component-allocation receipt requires complete Llama catalog "
                "provenance."
            )
        return {}
    present = frozenset(compiler)
    bridge_keys = present.intersection(TOKENIZER_BRIDGE_CONFIG_KEYS)
    if bridge_keys and bridge_keys != TOKENIZER_BRIDGE_CONFIG_KEYS:
        missing = sorted(TOKENIZER_BRIDGE_CONFIG_KEYS - bridge_keys)
        raise HFCausalBlankTargetAdapterError(
            "Llama tokenizer-bridge provenance is incomplete; missing: "
            + ", ".join(missing)
        )
    required_provenance = (
        _CATALOG_PROVENANCE_KEYS | _RECEIPT_PROVENANCE_KEYS
    )
    if not required_provenance.issubset(present):
        missing = sorted(required_provenance - present)
        raise HFCausalBlankTargetAdapterError(
            "Llama compiler provenance must contain its complete catalog and "
            "receipt bindings; missing: " + ", ".join(missing)
        )
    if (
        compiler[DENSE_TOPOLOGY_CATALOG_CONFIG_KEY] != HF_LLAMA_CATALOG_ID
        or compiler[HF_LLAMA_TARGET_SCHEMA_CONFIG_KEY]
        != HF_LLAMA_TARGET_SCHEMA
        or compiler[HF_LLAMA_PARAMETER_ALIAS_POLICY_CONFIG_KEY]
        != HF_LLAMA_PARAMETER_ALIAS_POLICY_NONE
    ):
        raise HFCausalBlankTargetAdapterError(
            "Llama catalog/schema/parameter-alias provenance is invalid."
        )
    receipt = request.component_allocation_receipt
    if receipt is None:
        raise HFCausalBlankTargetAdapterError(
            "Llama catalog provenance requires its component-allocation receipt."
        )
    if (
        compiler[COMPONENT_ALLOCATION_POLICY_CONFIG_KEY] != receipt.policy
        or compiler[COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY]
        != receipt.fingerprint
        or compiler[COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY]
        != receipt.topology_configuration_sha256
    ):
        raise HFCausalBlankTargetAdapterError(
            "Llama compiler receipt seals do not match the bound receipt."
        )
    matching_sources = tuple(
        graph
        for graph in parent_graphs.values()
        if graph.fingerprint == receipt.source_graph_fingerprint
    )
    if not matching_sources:
        raise HFCausalBlankTargetAdapterError(
            "The receipt source graph is not one of the loaded parents."
        )
    source = matching_sources[0]
    if (
        source.unique_parameter_count != receipt.source_unique_parameters
        or source.logical_parameter_count != receipt.source_logical_parameters
        or receipt.target_unique_parameters != request.exact_unique_parameters
        or receipt.target_logical_parameters != request.exact_unique_parameters
    ):
        raise HFCausalBlankTargetAdapterError(
            "Llama receipt counts differ from its source or requested dense target."
        )
    if compiler[HF_LLAMA_SOURCE_FAMILY_CONFIG_KEY] != source.family:
        raise HFCausalBlankTargetAdapterError(
            "Llama source-family provenance differs from the receipt source."
        )
    source_vocabulary_size = _source_vocabulary_size(source)
    target_vocabulary_size = constructor.get("vocab_size")
    if (
        isinstance(target_vocabulary_size, bool)
        or not isinstance(target_vocabulary_size, int)
        or target_vocabulary_size <= 0
    ):
        raise HFCausalBlankTargetAdapterError(
            "Llama target vocabulary provenance is not a positive integer."
        )
    bridge_required = source_vocabulary_size != target_vocabulary_size
    bridge_supplied = bridge_keys == TOKENIZER_BRIDGE_CONFIG_KEYS
    if bridge_required != bridge_supplied:
        raise HFCausalBlankTargetAdapterError(
            "Llama changed-vocabulary provenance requires exactly one complete "
            "tokenizer bridge, and same-vocabulary provenance forbids one."
        )
    bridge_report: dict[str, str | bool] = {
        "tokenizer_bridge_required": bridge_required,
        "tokenizer_bridge_authenticated": bridge_supplied,
    }
    if bridge_supplied:
        bridge_values = {
            key: _sha256_digest(compiler[key], key)
            for key in TOKENIZER_BRIDGE_CONFIG_KEYS
        }
        if (
            bridge_values[PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY]
            != source.tokenizer_fingerprint
        ):
            raise HFCausalBlankTargetAdapterError(
                "Llama tokenizer-bridge parent semantics differ from the "
                "receipt source graph."
            )
        if (
            bridge_values[CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY]
            != request.tokenizer_fingerprint
        ):
            raise HFCausalBlankTargetAdapterError(
                "Llama tokenizer-bridge child semantics differ from the target "
                "request."
            )
        bridge_report.update(
            {
                "tokenizer_bridge_artifact_sha256": bridge_values[
                    TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY
                ],
                "tokenizer_bridge_profile_sha256": bridge_values[
                    TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY
                ],
                "parent_tokenizer_semantics_sha256": bridge_values[
                    PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY
                ],
                "child_tokenizer_semantics_sha256": bridge_values[
                    CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY
                ],
            }
        )
    elif (
        request.tokenizer_fingerprint is not None
        and request.tokenizer_fingerprint != source.tokenizer_fingerprint
    ):
        raise HFCausalBlankTargetAdapterError(
            "Llama target tokenizer semantics differ from the source without "
            "an authenticated tokenizer bridge."
        )
    spans = _decode_source_spans(
        compiler[HF_LLAMA_ORDERED_SOURCE_SPANS_CONFIG_KEY]
    )
    flattened_spans = tuple(index for span in spans for index in span)
    source_layer_count = compiler[HF_LLAMA_SOURCE_LAYER_COUNT_CONFIG_KEY]
    target_depth = constructor.get("num_hidden_layers")
    if (
        isinstance(target_depth, bool)
        or not isinstance(target_depth, int)
        or len(spans) != target_depth
        or isinstance(source_layer_count, bool)
        or not isinstance(source_layer_count, int)
        or source_layer_count <= 0
        or source_layer_count != len(flattened_spans)
        or source_layer_count != len(source.layer_indices)
        or flattened_spans != tuple(range(source_layer_count))
    ):
        raise HFCausalBlankTargetAdapterError(
            "Llama source spans/layer count disagree with target depth or "
            "receipt source depth."
        )
    if (
        component_allocation_topology_fingerprint(
            {**constructor, **compiler}
        )
        != receipt.topology_configuration_sha256
    ):
        raise HFCausalBlankTargetAdapterError(
            "Llama compiler provenance is not bound to the receipt topology."
        )
    return bridge_report


class HFCausalBlankTargetSynthesisAdapter:
    """Materialize an exact dense HF child without consulting parent classes.

    Llama is admitted through the unchanged generic target synthesizer.  GPT-2
    and Qwen3 remain available from the public materializer, but are not
    advertised here until their complete generic IR/runtime-buffer contracts
    are independently authenticated.
    """

    adapter_id = "neural_forge.hf.causal_dense_blank"
    version = 1
    priority = 90
    target_families = ("llama",)
    target_modalities = ("text",)

    def __init__(
        self,
        *,
        device: Any = "cpu",
        dtype: Any | None = None,
    ) -> None:
        self.device = device
        self.dtype = dtype

    def supports(
        self,
        request: DynamicTargetRequest,
        _parent_graphs: Mapping[str, ModelCircuitGraph],
    ) -> bool:
        if request.target_modalities != ("text",):
            return False
        try:
            constructor, compiler = _constructor_configuration(
                dict(request.configuration)
            )
            _validate_compiler_provenance(
                request=request,
                constructor=constructor,
                compiler=compiler,
                parent_graphs=_parent_graphs,
            )
            family, _normalized = (
                normalize_hf_causal_blank_target_configuration(
                    target_family=request.target_family,
                    configuration=constructor,
                )
            )
        except (HFCausalBlankTargetError, HFCausalBlankTargetAdapterError):
            return False
        return family in self.target_families

    def materialize(
        self,
        request: DynamicTargetRequest,
        _parent_models: Mapping[str, Any],
        _parent_graphs: Mapping[str, ModelCircuitGraph],
        inspectors: Any,
    ) -> AdapterMaterialization:
        configuration = dict(request.configuration)
        constructor, compiler = _constructor_configuration(configuration)
        provenance_report = _validate_compiler_provenance(
            request=request,
            constructor=constructor,
            compiler=compiler,
            parent_graphs=_parent_graphs,
        )
        try:
            blank = materialize_hf_causal_blank_target(
                target_family=request.target_family,
                configuration=constructor,
                exact_unique_parameters=request.exact_unique_parameters,
                seed=request.blank_seed,
                device=self.device,
                dtype=self.dtype,
            )
        except HFCausalBlankTargetError as exc:
            raise HFCausalBlankTargetAdapterError(
                "Could not materialize the requested local HF causal blank."
            ) from exc
        if blank.family not in self.target_families:
            raise HFCausalBlankTargetAdapterError(
                "The requested family lacks an authenticated generic runtime "
                "buffer policy."
            )
        graph = inspectors.inspect(
            blank.model,
            model_name=request.child_name,
            tokenizer_fingerprint=request.tokenizer_fingerprint,
        )
        if (
            graph.family != blank.family
            or graph.logical_parameter_count != request.exact_unique_parameters
            or graph.unique_parameter_count != request.exact_unique_parameters
            or graph.logical_parameter_count != graph.unique_parameter_count
        ):
            raise HFCausalBlankTargetAdapterError(
                "Independent circuit inspection disagrees with the exact dense "
                "untied blank proof."
            )
        receipt = request.component_allocation_receipt
        allocation_report: dict[str, str | bool] = {}
        if receipt is not None:
            if (
                configuration.get(COMPONENT_ALLOCATION_POLICY_CONFIG_KEY)
                != receipt.policy
                or configuration.get(COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY)
                != receipt.fingerprint
                or configuration.get(COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY)
                != receipt.topology_configuration_sha256
                or receipt.target_unique_parameters
                != request.exact_unique_parameters
                or receipt.target_logical_parameters
                != request.exact_unique_parameters
            ):
                raise HFCausalBlankTargetAdapterError(
                    "The component-allocation receipt differs from the materialized "
                    "dense untied target contract."
                )
            allocation_report = {
                "component_allocation_receipt_sha256": receipt.fingerprint,
                "component_allocation_topology_sha256": (
                    receipt.topology_configuration_sha256
                ),
                "component_allocation_verified_by_target_adapter": True,
            }
        return AdapterMaterialization(
            model=blank.model,
            graph=graph,
            topology_fingerprint=canonical_sha256(
                {
                    "adapter_id": self.adapter_id,
                    "adapter_version": self.version,
                    "request_fingerprint": request.fingerprint,
                    "graph_fingerprint": graph.fingerprint,
                    "family": blank.family,
                    "architecture": blank.architecture,
                    "exact_unique_parameters": blank.unique_parameter_count,
                }
            ),
            report={
                "architecture": blank.architecture,
                "constructor": blank.constructor,
                "target_family": blank.family,
                "target_depth": int(blank.configuration["num_hidden_layers"]),
                "target_hidden_size": int(blank.configuration["hidden_size"]),
                "target_intermediate_size": int(
                    blank.configuration["intermediate_size"]
                ),
                "target_query_heads": int(
                    blank.configuration["num_attention_heads"]
                ),
                "target_key_value_heads": int(
                    blank.configuration["num_key_value_heads"]
                ),
                "target_head_dim": int(blank.configuration["head_dim"]),
                "target_vocab_size": int(blank.configuration["vocab_size"]),
                "target_logical_parameters": blank.logical_parameter_count,
                "target_unique_parameters": blank.unique_parameter_count,
                "unique_parameter_tensors": blank.unique_parameter_tensors,
                "unique_parameter_storages": blank.unique_parameter_storages,
                "compiler_metadata_key_count": len(compiler),
                **provenance_report,
                "local_config_instantiation": True,
                "pretrained_load_calls": 0,
                "network_loader_used": False,
                "parent_weights_read": False,
                "parent_family_independent": True,
                "dense_parameters": True,
                "all_parameters_exact_zero": True,
                "tied_parameters": False,
                "parameter_sharing_used": False,
                "training_used": False,
                "pruning_used": False,
                "quantization_used": False,
                "sparsity_used": False,
                "sidecar_used": False,
                **allocation_report,
            },
            adapter_payload=blank,
        )


__all__ = [
    "HFCausalBlankTargetAdapterError",
    "HFCausalBlankTargetSynthesisAdapter",
]
