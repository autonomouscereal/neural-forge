"""Standalone LFM2 decoding on the donor semantic-token clock.

The semantic-step token fold stores one compact target-token table in the
child.  A donor token can be spelled by several target tokens, but those
spelling tokens must not become transformer time steps.  This runtime keeps
the two clocks separate:

* prompts are rendered and tokenized by the donor tokenizer;
* donor embeddings are reconstructed from the child's existing tied table;
* the LFM2 body advances once for each donor semantic token;
* only defined donor rows are scored; and
* the complete decoded response string is retokenized once by the target
  tokenizer after generation.

No parent model, ``generate()``, gradient, optimizer, training, parameter
copy, or registered runtime parameter is used here.  The routing object is
consumed through a small duck-typed API so the model runtime does not depend
on a particular fold implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.lfm2_decoded_review import response_termination_evidence


LFM2_SEMANTIC_STEP_MODEL_RUNTIME_SCHEMA = (
    "neural-forge-lfm2-semantic-step-model-runtime-v1"
)
LFM2_SEMANTIC_STEP_MODEL_PROGRESS_SCHEMA = (
    "neural-forge-lfm2-semantic-step-model-progress-v1"
)


class Lfm2SemanticStepModelRuntimeError(RuntimeError):
    """The standalone semantic-step decode contract was violated."""


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise Lfm2SemanticStepModelRuntimeError(
        f"Runtime evidence contains non-JSON value {type(value).__name__}."
    )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2SemanticStepModelRuntimeError(f"{name} must be positive.")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2SemanticStepModelRuntimeError(
            f"{name} must be a nonnegative integer."
        )
    return int(value)


class _ProgressEmitter:
    def __init__(
        self,
        callback: Callable[[Mapping[str, Any]], None] | None,
    ) -> None:
        if callback is not None and not callable(callback):
            raise TypeError("progress_callback must be callable or None.")
        self._callback = callback
        self.count = 0
        self.final_sha256: str | None = None

    def emit(self, stage: str, **fields: Any) -> None:
        body = {
            "schema": LFM2_SEMANTIC_STEP_MODEL_PROGRESS_SCHEMA,
            "ordinal": self.count,
            "previous_progress_sha256": self.final_sha256,
            "stage": str(stage),
            **_plain_json(fields),
        }
        seal = canonical_sha256(body)
        record = MappingProxyType({**body, "progress_sha256": seal})
        self.count += 1
        self.final_sha256 = seal
        if self._callback is not None:
            self._callback(record)


def _vocabulary_ids(tokenizer: Any, *, name: str) -> tuple[int, ...]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise Lfm2SemanticStepModelRuntimeError(
            f"{name} tokenizer must expose get_vocab()."
        )
    vocabulary = get_vocab()
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise Lfm2SemanticStepModelRuntimeError(
            f"{name} tokenizer returned an empty or invalid vocabulary."
        )
    ids = []
    for token, value in vocabulary.items():
        if (
            not isinstance(token, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise Lfm2SemanticStepModelRuntimeError(
                f"{name} tokenizer vocabulary must map strings to integer IDs."
            )
        ids.append(int(value))
    unique = tuple(sorted(set(ids)))
    if len(unique) != len(ids) or unique != tuple(range(len(unique))):
        raise Lfm2SemanticStepModelRuntimeError(
            f"{name} tokenizer defined IDs must be unique and contiguous from zero."
        )
    return unique


def _routing_contract(routing: Any) -> dict[str, Any]:
    required_methods = (
        "span_for_donor",
        "reconstruct_selected_donor_embeddings",
        "score_target_tokens",
        "compute_all_donor_logits_from_target_scores",
    )
    missing = [
        name for name in required_methods if not callable(getattr(routing, name, None))
    ]
    if missing:
        raise Lfm2SemanticStepModelRuntimeError(
            "Semantic routing is missing required operations: " + ", ".join(missing)
        )
    if callable(getattr(routing, "parameters", None)) or hasattr(
        routing, "target_table"
    ):
        raise Lfm2SemanticStepModelRuntimeError(
            "Semantic routing must be parameterless and must not own a target table."
        )
    try:
        donor_count = _positive_int(routing.donor_row_count, "routing donor_row_count")
        target_count = _positive_int(
            routing.target_row_count, "routing target_row_count"
        )
        hidden_size = _positive_int(routing.hidden_size, "routing hidden_size")
    except AttributeError as exc:
        raise Lfm2SemanticStepModelRuntimeError(
            "Semantic routing must expose donor_row_count, target_row_count, and "
            "hidden_size."
        ) from exc
    as_dict = getattr(routing, "as_dict", None)
    if not callable(as_dict):
        raise Lfm2SemanticStepModelRuntimeError(
            "Semantic routing must expose its authenticated as_dict() receipt."
        )
    receipt = as_dict()
    if not isinstance(receipt, Mapping):
        raise Lfm2SemanticStepModelRuntimeError(
            "Semantic routing receipt must be a mapping."
        )
    contract = _plain_json(receipt)
    if (
        contract.get("owned_learned_parameter_count") != 0
        or contract.get("owns_or_registers_target_table") is not False
    ):
        raise Lfm2SemanticStepModelRuntimeError(
            "Semantic routing receipt does not prove an unregistered fixed route."
        )
    integer_only = contract.get("routing_metadata_is_integer_only")
    if integer_only is True:
        continuous_state_count = 0
        input_continuous_state_count = 0
        output_continuous_state_count = 0
        input_output_coefficient_storage_shared = True
        if (
            contract.get("owned_fixed_continuous_state_count", 0) != 0
            or contract.get("continuous_semantic_code_coefficient_count", 0) != 0
        ):
            raise Lfm2SemanticStepModelRuntimeError(
                "Integer-only semantic routing claims continuous child state."
            )
    elif integer_only is False:
        raw_count = contract.get("continuous_semantic_code_coefficient_count")
        routing_mode = contract.get("semantic_routing_mode")
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count <= 0
            or contract.get("owned_fixed_continuous_state_count") != raw_count
            or contract.get("semantic_code_coefficients_counted_as_child_state")
            is not True
            or contract.get("semantic_code_coefficient_dtype") != "float32"
            or routing_mode
            not in {
                "dual-channel-sparse-semantic-code",
                "emission-plus-separate-input-output-sparse-semantic-codes",
            }
            or contract.get("emission_spans_are_semantic_coefficients") is not False
        ):
            raise Lfm2SemanticStepModelRuntimeError(
                "Floating semantic routing does not authenticate and count every "
                "continuous coefficient as child state."
            )
        continuous_state_count = int(raw_count)
        if routing_mode == "dual-channel-sparse-semantic-code":
            input_continuous_state_count = continuous_state_count
            output_continuous_state_count = continuous_state_count
            input_output_coefficient_storage_shared = True
        else:
            input_count = contract.get(
                "input_continuous_semantic_code_coefficient_count"
            )
            output_count = contract.get(
                "output_continuous_semantic_code_coefficient_count"
            )
            if (
                isinstance(input_count, bool)
                or not isinstance(input_count, int)
                or input_count <= 0
                or isinstance(output_count, bool)
                or not isinstance(output_count, int)
                or output_count <= 0
                or input_count + output_count != continuous_state_count
                or contract.get("input_output_coefficient_storage_shared") is not False
            ):
                raise Lfm2SemanticStepModelRuntimeError(
                    "Separate semantic routes do not count C_in and C_out exactly."
                )
            input_continuous_state_count = int(input_count)
            output_continuous_state_count = int(output_count)
            input_output_coefficient_storage_shared = False
    else:
        raise Lfm2SemanticStepModelRuntimeError(
            "Semantic routing receipt has no authenticated metadata type."
        )
    if (
        contract.get("donor_row_count") != donor_count
        or contract.get("target_row_count") != target_count
        or contract.get("hidden_size") != hidden_size
    ):
        raise Lfm2SemanticStepModelRuntimeError(
            "Semantic routing properties differ from its receipt."
        )
    return {
        "donor_row_count": donor_count,
        "target_row_count": target_count,
        "hidden_size": hidden_size,
        "continuous_state_count": continuous_state_count,
        "input_continuous_state_count": input_continuous_state_count,
        "output_continuous_state_count": output_continuous_state_count,
        "input_output_coefficient_storage_shared": (
            input_output_coefficient_storage_shared
        ),
        "receipt": contract,
        "receipt_sha256": canonical_sha256(contract),
    }


def _live_tied_table(
    child_model: Any,
    *,
    target_row_count: int,
    hidden_size: int,
) -> Tensor:
    get_input = getattr(child_model, "get_input_embeddings", None)
    get_output = getattr(child_model, "get_output_embeddings", None)
    if not callable(get_input) or not callable(get_output):
        raise Lfm2SemanticStepModelRuntimeError(
            "LFM2 child must expose input and output embedding accessors."
        )
    input_module = get_input()
    output_module = get_output()
    input_weight = getattr(input_module, "weight", None)
    output_weight = getattr(output_module, "weight", None)
    if not isinstance(input_weight, Tensor) or not isinstance(output_weight, Tensor):
        raise Lfm2SemanticStepModelRuntimeError(
            "LFM2 child input/output modules must expose tensor weights."
        )
    if input_weight is not output_weight:
        raise Lfm2SemanticStepModelRuntimeError(
            "Same-body semantic-step baseline requires one live tied input/readout "
            "table."
        )
    if tuple(int(size) for size in input_weight.shape) != (
        target_row_count,
        hidden_size,
    ):
        raise Lfm2SemanticStepModelRuntimeError(
            "Live tied child table differs from semantic routing dimensions."
        )
    if (
        input_weight.ndim != 2
        or not torch.is_floating_point(input_weight)
        or input_weight.is_complex()
        or input_weight.layout != torch.strided
        or input_weight.device.type == "meta"
    ):
        raise Lfm2SemanticStepModelRuntimeError(
            "Live tied child table must be one materialized dense floating matrix."
        )
    return input_weight


def _normalize_eos_ids(tokenizer: Any, *, donor_row_count: int) -> frozenset[int]:
    raw = getattr(tokenizer, "eos_token_id", None)
    if raw is None:
        return frozenset()
    values: Sequence[Any]
    if isinstance(raw, int) and not isinstance(raw, bool):
        values = (raw,)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = raw
    else:
        raise Lfm2SemanticStepModelRuntimeError(
            "Donor tokenizer eos_token_id must be an integer sequence or None."
        )
    result = frozenset(
        _nonnegative_int(value, "donor EOS token ID") for value in values
    )
    if any(value >= donor_row_count for value in result):
        raise Lfm2SemanticStepModelRuntimeError(
            "Donor EOS token ID leaves the defined donor vocabulary."
        )
    return result


def _normalize_prompts(
    prompts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if (
        isinstance(prompts, (str, bytes))
        or not isinstance(prompts, Sequence)
        or not prompts
    ):
        raise Lfm2SemanticStepModelRuntimeError(
            "prompts must be one nonempty sequence of review rows."
        )
    result = []
    seen: set[str] = set()
    for index, row in enumerate(prompts):
        if not isinstance(row, Mapping):
            raise Lfm2SemanticStepModelRuntimeError(
                "Every review prompt must be a mapping."
            )
        prompt_id = str(row.get("id", f"review-{index:04d}")).strip()
        if not prompt_id or prompt_id in seen:
            raise Lfm2SemanticStepModelRuntimeError(
                "Review prompt IDs must be unique and nonempty."
            )
        seen.add(prompt_id)
        has_prompt = isinstance(row.get("prompt"), str) and bool(
            str(row["prompt"]).strip()
        )
        messages = row.get("messages")
        has_messages = (
            isinstance(messages, Sequence)
            and not isinstance(messages, (str, bytes))
            and bool(messages)
        )
        if has_prompt == has_messages:
            raise Lfm2SemanticStepModelRuntimeError(
                f"Prompt {prompt_id!r} must contain exactly one of prompt or messages."
            )
        normalized = dict(row)
        normalized["id"] = prompt_id
        normalized["category"] = str(row.get("category", "general"))
        if has_messages:
            normalized_messages = []
            for message in messages:
                if (
                    not isinstance(message, Mapping)
                    or message.get("role") not in {"system", "user", "assistant"}
                    or not isinstance(message.get("content"), str)
                    or not str(message["content"]).strip()
                ):
                    raise Lfm2SemanticStepModelRuntimeError(
                        f"Prompt {prompt_id!r} contains an invalid message."
                    )
                normalized_messages.append(
                    {
                        "role": str(message["role"]),
                        "content": str(message["content"]),
                    }
                )
            normalized["messages"] = normalized_messages
        result.append(normalized)
    return tuple(result)


def _extract_tokenized(value: Any, *, name: str) -> tuple[Tensor, Tensor | None]:
    if isinstance(value, Mapping):
        input_ids = value.get("input_ids")
        attention_mask = value.get("attention_mask")
    else:
        input_ids = value
        attention_mask = None
    if not isinstance(input_ids, Tensor) or input_ids.ndim not in {1, 2}:
        raise Lfm2SemanticStepModelRuntimeError(
            f"{name} tokenizer returned invalid input IDs."
        )
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if int(input_ids.shape[0]) != 1 or int(input_ids.shape[1]) == 0:
        raise Lfm2SemanticStepModelRuntimeError(
            f"{name} tokenizer must return one nonempty prompt row."
        )
    if attention_mask is not None:
        if not isinstance(attention_mask, Tensor) or attention_mask.ndim not in {1, 2}:
            raise Lfm2SemanticStepModelRuntimeError(
                f"{name} tokenizer returned an invalid attention mask."
            )
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if tuple(attention_mask.shape) != tuple(input_ids.shape) or not bool(
            torch.all(attention_mask != 0)
        ):
            raise Lfm2SemanticStepModelRuntimeError(
                "Semantic-step review supports one unpadded donor prompt at a time."
            )
    return input_ids.detach().to(device="cpu", dtype=torch.int64), attention_mask


def _donor_prompt_ids(tokenizer: Any, prompt: Mapping[str, Any]) -> Tensor:
    if "messages" in prompt:
        template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(template):
            raise Lfm2SemanticStepModelRuntimeError(
                "Donor tokenizer cannot render chat-message review prompts."
            )
        rendered = template(
            prompt["messages"],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        rendered = tokenizer(
            prompt["prompt"],
            return_tensors="pt",
            add_special_tokens=True,
        )
    input_ids, _attention_mask = _extract_tokenized(rendered, name="donor")
    return input_ids


def _body_output(value: Any, *, expected_hidden_size: int) -> tuple[Tensor, Any]:
    hidden = getattr(value, "last_hidden_state", None)
    cache = getattr(value, "past_key_values", None)
    if hidden is None and isinstance(value, (tuple, list)) and value:
        hidden = value[0]
        if len(value) > 1:
            cache = value[1]
    if (
        not isinstance(hidden, Tensor)
        or hidden.ndim != 3
        or int(hidden.shape[0]) != 1
        or int(hidden.shape[1]) == 0
        or int(hidden.shape[2]) != expected_hidden_size
        or not torch.is_floating_point(hidden)
        or hidden.is_complex()
        or not bool(torch.isfinite(hidden.detach()).all())
    ):
        raise Lfm2SemanticStepModelRuntimeError(
            "LFM2 body returned invalid semantic hidden states."
        )
    if cache is None:
        raise Lfm2SemanticStepModelRuntimeError(
            "LFM2 body did not return a cache while use_cache=True."
        )
    return hidden.detach(), cache


def _target_retokenize(
    tokenizer: Any,
    text: str,
    *,
    target_row_count: int,
) -> tuple[list[int], str]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise Lfm2SemanticStepModelRuntimeError(
            "Target tokenizer must expose encode() for whole-string retokenization."
        )
    encoded = encode(text, add_special_tokens=False)
    if isinstance(encoded, Tensor):
        encoded = encoded.detach().cpu().reshape(-1).tolist()
    if isinstance(encoded, (str, bytes)) or not isinstance(encoded, Sequence):
        raise Lfm2SemanticStepModelRuntimeError(
            "Target tokenizer returned invalid whole-string token IDs."
        )
    ids = [_nonnegative_int(value, "target response token ID") for value in encoded]
    if any(value >= target_row_count for value in ids):
        raise Lfm2SemanticStepModelRuntimeError(
            "Whole-string target retokenization left the defined target vocabulary."
        )
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise Lfm2SemanticStepModelRuntimeError(
            "Target tokenizer must expose decode()."
        )
    decoded = decode(ids, skip_special_tokens=True)
    if not isinstance(decoded, str):
        raise Lfm2SemanticStepModelRuntimeError(
            "Target tokenizer returned non-text decoded output."
        )
    return ids, decoded


def _session_context(
    routing: Any,
    live_table: Tensor,
    *,
    allow_legacy_per_operation_authentication: bool,
) -> tuple[Any, str]:
    factory = getattr(routing, "authenticated_live_table_session", None)
    if callable(factory):
        context = factory(live_table)
        if not callable(getattr(context, "__enter__", None)) or not callable(
            getattr(context, "__exit__", None)
        ):
            raise Lfm2SemanticStepModelRuntimeError(
                "Routing live-table session must be a context manager."
            )
        return context, "hash-once-live-table-session"
    if not allow_legacy_per_operation_authentication:
        raise Lfm2SemanticStepModelRuntimeError(
            "Semantic routing must expose authenticated_live_table_session(); "
            "repeated full-table authentication is disabled for real review runs."
        )
    return nullcontext(live_table), "legacy-per-operation-authentication"


def _parameter_snapshot(model: Any) -> dict[str, Any]:
    parameters = tuple(model.parameters())
    modules = tuple(model.modules())
    if any(parameter.grad is not None for parameter in parameters):
        raise Lfm2SemanticStepModelRuntimeError(
            "Child model contains gradients before semantic-step inference."
        )
    return {
        "parameter_ids": tuple(id(parameter) for parameter in parameters),
        "parameter_versions": tuple(
            int(parameter._version) for parameter in parameters
        ),
        "module_ids": tuple(id(module) for module in modules),
        "unique_parameter_count": sum(
            int(parameter.numel()) for parameter in parameters
        ),
    }


def _verify_parameter_snapshot(model: Any, before: Mapping[str, Any]) -> None:
    parameters = tuple(model.parameters())
    if (
        tuple(id(parameter) for parameter in parameters) != before["parameter_ids"]
        or tuple(int(parameter._version) for parameter in parameters)
        != before["parameter_versions"]
        or tuple(id(module) for module in model.modules()) != before["module_ids"]
        or sum(int(parameter.numel()) for parameter in parameters)
        != before["unique_parameter_count"]
        or any(parameter.grad is not None for parameter in parameters)
    ):
        raise Lfm2SemanticStepModelRuntimeError(
            "Child parameters, modules, values, or gradients changed during review."
        )


def run_lfm2_semantic_step_model_review(
    child_model: Any,
    semantic_routing: Any,
    donor_tokenizer: Any,
    target_tokenizer: Any,
    prompts: Sequence[Mapping[str, Any]],
    *,
    maximum_prompt_tokens: int = 8192,
    maximum_new_tokens: int = 256,
    maximum_embedding_rows_per_chunk: int = 1024,
    maximum_donor_rows_per_score_chunk: int = 4096,
    allow_legacy_per_operation_authentication: bool = False,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Decode review prompts with donor actions as the only body clock.

    ``semantic_routing`` is deliberately duck typed.  Its required surface is
    documented by :func:`_routing_contract`; newer artifacts may additionally
    expose ``authenticated_live_table_session(table)`` to hash the live table
    only at context entry and exit.
    """

    prompt_limit = _positive_int(maximum_prompt_tokens, "maximum_prompt_tokens")
    new_limit = _positive_int(maximum_new_tokens, "maximum_new_tokens")
    embedding_chunk = _positive_int(
        maximum_embedding_rows_per_chunk,
        "maximum_embedding_rows_per_chunk",
    )
    score_chunk = _positive_int(
        maximum_donor_rows_per_score_chunk,
        "maximum_donor_rows_per_score_chunk",
    )
    if not isinstance(allow_legacy_per_operation_authentication, bool):
        raise TypeError("allow_legacy_per_operation_authentication must be bool.")
    if prompt_limit > 8192:
        raise Lfm2SemanticStepModelRuntimeError(
            "maximum_prompt_tokens cannot exceed 8192."
        )
    if new_limit > 2048:
        raise Lfm2SemanticStepModelRuntimeError(
            "maximum_new_tokens cannot exceed 2048."
        )
    normalized_prompts = _normalize_prompts(prompts)
    routing = _routing_contract(semantic_routing)
    donor_ids = _vocabulary_ids(donor_tokenizer, name="donor")
    target_ids = _vocabulary_ids(target_tokenizer, name="target")
    if len(donor_ids) != routing["donor_row_count"]:
        raise Lfm2SemanticStepModelRuntimeError(
            "Routing donor rows must equal every contiguous defined donor tokenizer "
            "ID; undefined model-capacity rows cannot be scored."
        )
    if len(target_ids) != routing["target_row_count"]:
        raise Lfm2SemanticStepModelRuntimeError(
            "Routing target rows must equal every contiguous defined target tokenizer "
            "ID."
        )
    for donor_id in donor_ids:
        span = semantic_routing.span_for_donor(donor_id)
        if (
            isinstance(span, (str, bytes))
            or not isinstance(span, Sequence)
            or not span
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= routing["target_row_count"]
                for value in span
            )
        ):
            raise Lfm2SemanticStepModelRuntimeError(
                "Every defined donor ID must have one nonempty in-range target span."
            )
    eos_ids = _normalize_eos_ids(
        donor_tokenizer,
        donor_row_count=routing["donor_row_count"],
    )
    live_table = _live_tied_table(
        child_model,
        target_row_count=routing["target_row_count"],
        hidden_size=routing["hidden_size"],
    )
    body = getattr(child_model, "model", None)
    if not callable(body):
        raise Lfm2SemanticStepModelRuntimeError(
            "LFM2 child must expose its callable body as model."
        )

    snapshot = _parameter_snapshot(child_model)
    effective_child_parameter_count = int(snapshot["unique_parameter_count"]) + int(
        routing["continuous_state_count"]
    )
    emitter = _ProgressEmitter(progress_callback)
    session_context, authentication_mode = _session_context(
        semantic_routing,
        live_table,
        allow_legacy_per_operation_authentication=(
            allow_legacy_per_operation_authentication
        ),
    )
    emitter.emit(
        "runtime-start",
        prompt_count=len(normalized_prompts),
        donor_defined_row_count=routing["donor_row_count"],
        target_defined_row_count=routing["target_row_count"],
        maximum_new_tokens=new_limit,
        live_table_device=str(live_table.device),
        live_table_dtype=str(live_table.dtype).removeprefix("torch."),
        routing_authentication_mode=authentication_mode,
        routing_continuous_state_count=routing["continuous_state_count"],
        routing_input_continuous_state_count=routing[
            "input_continuous_state_count"
        ],
        routing_output_continuous_state_count=routing[
            "output_continuous_state_count"
        ],
        routing_input_output_coefficient_storage_shared=routing[
            "input_output_coefficient_storage_shared"
        ],
        effective_child_parameter_count=effective_child_parameter_count,
    )

    responses: list[dict[str, Any]] = []
    prefill_calls = 0
    generation_calls = 0
    prompt_semantic_tokens = 0
    generated_semantic_tokens = 0
    target_span_metadata_tokens = 0
    target_retokenizer_calls = 0
    session: Any = None
    was_training = bool(child_model.training)
    child_model.eval()
    try:
        with torch.inference_mode(), session_context as session:
            for prompt_index, prompt in enumerate(normalized_prompts):
                prompt_id = str(prompt["id"])
                emitter.emit(
                    "prompt-start",
                    prompt_index=prompt_index,
                    prompt_id=prompt_id,
                    category=prompt["category"],
                )
                prompt_tensor = _donor_prompt_ids(donor_tokenizer, prompt)
                prompt_length = int(prompt_tensor.shape[1])
                if prompt_length > prompt_limit:
                    raise Lfm2SemanticStepModelRuntimeError(
                        f"Prompt {prompt_id!r} exceeds maximum_prompt_tokens."
                    )
                raw_prompt_ids = [
                    int(value) for value in prompt_tensor[0].detach().cpu().tolist()
                ]
                if any(
                    value < 0 or value >= routing["donor_row_count"]
                    for value in raw_prompt_ids
                ):
                    raise Lfm2SemanticStepModelRuntimeError(
                        f"Prompt {prompt_id!r} contains an undefined donor token ID."
                    )
                prompt_embeddings = (
                    semantic_routing.reconstruct_selected_donor_embeddings(
                        session,
                        raw_prompt_ids,
                        maximum_rows_per_chunk=embedding_chunk,
                    )
                )
                if tuple(prompt_embeddings.shape) != (
                    prompt_length,
                    routing["hidden_size"],
                ):
                    raise Lfm2SemanticStepModelRuntimeError(
                        "Routing reconstructed an invalid donor prompt embedding matrix."
                    )
                attention_mask = torch.ones(
                    (1, prompt_length),
                    device=live_table.device,
                    dtype=torch.int64,
                )
                output = body(
                    inputs_embeds=prompt_embeddings.unsqueeze(0),
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                hidden, cache = _body_output(
                    output,
                    expected_hidden_size=routing["hidden_size"],
                )
                prefill_calls += 1
                prompt_semantic_tokens += prompt_length
                emitter.emit(
                    "prompt-prefill-complete",
                    prompt_index=prompt_index,
                    prompt_id=prompt_id,
                    donor_prompt_token_count=prompt_length,
                    body_forward_call_count=1,
                    body_semantic_tokens_advanced=prompt_length,
                )

                raw_generated_ids: list[int] = []
                step_audit: list[dict[str, Any]] = []
                ended_with_eos = False
                last_hidden = hidden[:, -1, :]
                for step_index in range(new_limit):
                    target_scores = semantic_routing.score_target_tokens(
                        session,
                        last_hidden,
                    )
                    donor_logits = (
                        semantic_routing.compute_all_donor_logits_from_target_scores(
                            session,
                            target_scores,
                            maximum_donor_rows_per_chunk=score_chunk,
                        )
                    )
                    if (
                        not isinstance(donor_logits, Tensor)
                        or donor_logits.ndim != 2
                        or tuple(int(size) for size in donor_logits.shape)
                        != (1, routing["donor_row_count"])
                        or not torch.is_floating_point(donor_logits)
                        or donor_logits.is_complex()
                        or not bool(torch.isfinite(donor_logits.detach()).all())
                    ):
                        raise Lfm2SemanticStepModelRuntimeError(
                            "Routing must score exactly the contiguous defined donor rows."
                        )
                    selected_id = int(torch.argmax(donor_logits[0]).item())
                    span = tuple(
                        int(value)
                        for value in semantic_routing.span_for_donor(selected_id)
                    )
                    raw_generated_ids.append(selected_id)
                    target_span_metadata_tokens += len(span)
                    ended_with_eos = selected_id in eos_ids
                    reached_generation_limit = step_index + 1 == new_limit
                    terminal_without_future_score = (
                        ended_with_eos or reached_generation_limit
                    )
                    body_forward_calls = 0
                    body_semantic_tokens_advanced = 0
                    if not terminal_without_future_score:
                        semantic_embedding = (
                            semantic_routing.reconstruct_selected_donor_embeddings(
                                session,
                                [selected_id],
                                maximum_rows_per_chunk=1,
                            )
                        )
                        if tuple(semantic_embedding.shape) != (
                            1,
                            routing["hidden_size"],
                        ):
                            raise Lfm2SemanticStepModelRuntimeError(
                                "Routing reconstructed an invalid selected donor "
                                "embedding."
                            )
                        next_total_length = prompt_length + len(raw_generated_ids)
                        attention_mask = torch.ones(
                            (1, next_total_length),
                            device=live_table.device,
                            dtype=torch.int64,
                        )
                        output = body(
                            inputs_embeds=semantic_embedding.unsqueeze(0),
                            attention_mask=attention_mask,
                            past_key_values=cache,
                            use_cache=True,
                        )
                        hidden, cache = _body_output(
                            output,
                            expected_hidden_size=routing["hidden_size"],
                        )
                        last_hidden = hidden[:, -1, :]
                        generation_calls += 1
                        generated_semantic_tokens += 1
                        body_forward_calls = 1
                        body_semantic_tokens_advanced = 1
                    step = {
                        "step_index": step_index,
                        "donor_token_id": selected_id,
                        "target_span_token_ids": list(span),
                        "target_span_length": len(span),
                        "eos_selected": ended_with_eos,
                        "generation_limit_selected": reached_generation_limit,
                        "defined_donor_logits_scored": routing["donor_row_count"],
                        "undefined_donor_logits_scored": 0,
                        "body_forward_calls": body_forward_calls,
                        "body_semantic_tokens_advanced": (
                            body_semantic_tokens_advanced
                        ),
                        "terminal_token_not_fed_to_unused_cache": (
                            terminal_without_future_score
                        ),
                        "target_span_tokens_used_as_body_steps": 0,
                    }
                    step_audit.append(step)
                    emitter.emit(
                        "semantic-step-complete",
                        prompt_index=prompt_index,
                        prompt_id=prompt_id,
                        **step,
                    )
                    if ended_with_eos:
                        break

                donor_decode = getattr(donor_tokenizer, "decode", None)
                if not callable(donor_decode):
                    raise Lfm2SemanticStepModelRuntimeError(
                        "Donor tokenizer must expose decode()."
                    )
                donor_text = donor_decode(
                    raw_generated_ids,
                    skip_special_tokens=True,
                )
                if not isinstance(donor_text, str):
                    raise Lfm2SemanticStepModelRuntimeError(
                        "Donor tokenizer returned non-text decoded output."
                    )
                final_target_ids, final_target_text = _target_retokenize(
                    target_tokenizer,
                    donor_text,
                    target_row_count=routing["target_row_count"],
                )
                target_retokenizer_calls += 1
                generation_limit_reached = (
                    len(raw_generated_ids) == new_limit and not ended_with_eos
                )
                termination_evidence = response_termination_evidence(
                    ended_with_eos=ended_with_eos,
                    generation_limit_reached=generation_limit_reached,
                )
                response = {
                    **dict(prompt),
                    # The target-tokenizer round trip is the deployable child
                    # response.  Preserve donor text separately as diagnostic
                    # evidence when tokenizer normalization changes the text.
                    "response": final_target_text,
                    "response_token_ids": final_target_ids,
                    "donor_response_text": donor_text,
                    "raw_donor_response_token_ids": raw_generated_ids,
                    "final_target_response_token_ids": final_target_ids,
                    "final_target_response_text": final_target_text,
                    "target_roundtrip_exact": final_target_text == donor_text,
                    "semantic_steps": step_audit,
                    "termination_evidence": termination_evidence,
                    "diagnostics": {
                        "donor_prompt_tokens": prompt_length,
                        "donor_response_tokens": len(raw_generated_ids),
                        "final_target_response_tokens": len(final_target_ids),
                        "ended_with_eos": ended_with_eos,
                        "generation_limit_reached": generation_limit_reached,
                        "maximum_new_tokens": new_limit,
                        "finish_reason": termination_evidence["finish_reason"],
                        "completion_status": termination_evidence[
                            "completion_status"
                        ],
                        "censored_by_output_cap": termination_evidence[
                            "censored_by_output_cap"
                        ],
                        "output_cap_hit_is_automatic_folding_failure": False,
                        "loop_observed": None,
                        "loop_observation_source": "not-reviewed",
                        "runtime_error": None,
                        "body_prefill_forward_calls": 1,
                        "body_generated_forward_calls": sum(
                            int(step["body_forward_calls"]) for step in step_audit
                        ),
                        "body_semantic_tokens_advanced": (
                            prompt_length
                            + sum(
                                int(step["body_semantic_tokens_advanced"])
                                for step in step_audit
                            )
                        ),
                        "target_span_tokens_used_as_body_steps": 0,
                        "whole_string_target_retokenizations": 1,
                        "undefined_donor_rows_scored": 0,
                        "authoritative_response_text_source": (
                            "whole-target-sequence-retokenization-decode"
                        ),
                        "target_roundtrip_changed_text": (
                            final_target_text != donor_text
                        ),
                        "interpretation": (
                            "diagnostic-only; manual whole-response intelligence "
                            "review is authoritative"
                        ),
                    },
                }
                responses.append(response)
                emitter.emit(
                    "prompt-complete",
                    prompt_index=prompt_index,
                    prompt_id=prompt_id,
                    donor_response_token_count=len(raw_generated_ids),
                    final_target_response_token_count=len(final_target_ids),
                    ended_with_eos=ended_with_eos,
                    generation_limit_reached=generation_limit_reached,
                    target_roundtrip_exact=final_target_text == donor_text,
                )
    finally:
        child_model.train(was_training)

    _verify_parameter_snapshot(child_model, snapshot)
    session_receipt = None
    full_hash_count = None
    if session is not None:
        session_as_dict = getattr(session, "as_dict", None)
        if callable(session_as_dict):
            raw_session_receipt = session_as_dict()
            if isinstance(raw_session_receipt, Mapping):
                session_receipt = _plain_json(raw_session_receipt)
        raw_hash_count = getattr(session, "full_hash_verification_count", None)
        if isinstance(raw_hash_count, int) and not isinstance(raw_hash_count, bool):
            full_hash_count = int(raw_hash_count)
    if (
        authentication_mode == "hash-once-live-table-session"
        and full_hash_count
        not in {
            None,
            2,
        }
    ):
        raise Lfm2SemanticStepModelRuntimeError(
            "Hash-once routing session must perform exactly entry and exit full hashes."
        )

    operation_audit = {
        "training_steps": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "parent_forward_calls": 0,
        "parent_generate_calls": 0,
        "child_generate_calls": 0,
        "child_lm_head_forward_calls": 0,
        "child_body_prefill_forward_calls": prefill_calls,
        "child_body_generated_forward_calls": generation_calls,
        "child_body_forward_calls": prefill_calls + generation_calls,
        "prompt_donor_semantic_tokens_advanced": prompt_semantic_tokens,
        "generated_donor_semantic_tokens_advanced": generated_semantic_tokens,
        "target_span_metadata_tokens_observed": target_span_metadata_tokens,
        "target_span_tokens_used_as_body_steps": 0,
        "whole_string_target_retokenizer_calls": target_retokenizer_calls,
        "undefined_donor_rows_scored": 0,
        "owned_learned_parameter_count": 0,
        "routing_fixed_continuous_state_count": routing["continuous_state_count"],
        "routing_input_fixed_continuous_state_count": routing[
            "input_continuous_state_count"
        ],
        "routing_output_fixed_continuous_state_count": routing[
            "output_continuous_state_count"
        ],
        "routing_input_output_coefficient_storage_shared": routing[
            "input_output_coefficient_storage_shared"
        ],
        "effective_child_parameter_count": effective_child_parameter_count,
        "semantic_code_coefficients_counted_exactly_once": True,
        "registered_parameter_count_delta": 0,
        "registered_module_count_delta": 0,
        "quantized_parameters": 0,
        "pruned_parameters": 0,
        "network_access": False,
    }
    emitter.emit(
        "runtime-complete",
        prompt_count=len(responses),
        child_body_forward_calls=prefill_calls + generation_calls,
        generated_donor_semantic_tokens=generated_semantic_tokens,
        undefined_donor_rows_scored=0,
        routing_full_hash_verification_count=full_hash_count,
    )
    result = {
        "schema": LFM2_SEMANTIC_STEP_MODEL_RUNTIME_SCHEMA,
        "status": "complete-standalone-semantic-step-whole-response-review",
        "responses": responses,
        "prompt_count": len(responses),
        "maximum_prompt_tokens": prompt_limit,
        "maximum_new_tokens": new_limit,
        "donor_defined_row_count": routing["donor_row_count"],
        "target_defined_row_count": routing["target_row_count"],
        "undefined_donor_capacity_rows_scored": 0,
        "live_table_shape": [
            routing["target_row_count"],
            routing["hidden_size"],
        ],
        "live_table_device": str(live_table.device),
        "live_table_dtype": str(live_table.dtype).removeprefix("torch."),
        "child_unique_parameter_count": snapshot["unique_parameter_count"],
        "routing_fixed_continuous_state_count": routing["continuous_state_count"],
        "routing_input_fixed_continuous_state_count": routing[
            "input_continuous_state_count"
        ],
        "routing_output_fixed_continuous_state_count": routing[
            "output_continuous_state_count"
        ],
        "routing_input_output_coefficient_storage_shared": routing[
            "input_output_coefficient_storage_shared"
        ],
        "effective_child_parameter_count": effective_child_parameter_count,
        "effective_child_parameter_count_includes_semantic_code_coefficients": True,
        "routing_contract_sha256": routing["receipt_sha256"],
        "routing_authentication_mode": authentication_mode,
        "hash_once_routing_authentication_required": (
            not allow_legacy_per_operation_authentication
        ),
        "routing_full_hash_verification_count": full_hash_count,
        "routing_session_receipt": session_receipt,
        "donor_tokens_are_the_only_body_clock": True,
        "target_spans_are_external_metadata_only": True,
        "whole_response_target_retokenization": True,
        "manual_whole_response_intelligence_review_is_authoritative": True,
        "progress_record_count": emitter.count,
        "progress_final_sha256": emitter.final_sha256,
        "operation_audit": operation_audit,
    }
    return {**result, "runtime_sha256": canonical_sha256(result)}


__all__ = [
    "LFM2_SEMANTIC_STEP_MODEL_PROGRESS_SCHEMA",
    "LFM2_SEMANTIC_STEP_MODEL_RUNTIME_SCHEMA",
    "Lfm2SemanticStepModelRuntimeError",
    "run_lfm2_semantic_step_model_review",
]
