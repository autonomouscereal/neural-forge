"""Public LFM2 semantic-route generation over authenticated child state.

The published model exposes ``semantic_step_generate`` as a normal public
method.  The method itself is tensor-free: it resolves the route runtime and
the two local tokenizers through a module-global weak registry populated by the
authenticated checkpoint loader.  Consequently the model's reachable-state
audit sees no unregistered tensors while callers outside the behavior gate can
still use the same route-aware generation path.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dynamic_circuit_ir import canonical_sha256
from .hf_lfm2_semantic_step_model_runtime import (
    LFM2_SEMANTIC_STEP_MODEL_RUNTIME_SCHEMA,
    run_lfm2_semantic_step_model_review,
)


LFM2_SEMANTIC_STEP_GENERATE_ENTRYPOINT = "semantic_step_generate"
LFM2_CARRIER_ROUTED_REVIEW_SCHEMA = "neural-forge-lfm2-carrier-routed-review-v1"
LFM2_CARRIER_REVIEW_ROW_SCHEMA = "neural-forge-lfm2-carrier-review-row-v1"


class Lfm2CarrierRouteRuntimeError(RuntimeError):
    """A public routed-generation registration or receipt is invalid."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CarrierRouteRuntimeError(f"{name} must be a positive integer.")
    return value


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _is_safety_category(value: Any) -> bool:
    """Recognize safety-labelled probes without treating them as intelligence."""

    return isinstance(value, str) and "safety" in value.casefold()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CarrierRouteRuntimeError(f"{name} must be a lowercase SHA-256.")
    return value


def _is_cap_censored(response: Any) -> bool:
    if not isinstance(response, Mapping):
        return False
    diagnostics = response.get("diagnostics")
    termination = response.get("termination_evidence")
    return bool(
        (
            isinstance(diagnostics, Mapping)
            and diagnostics.get("censored_by_output_cap") is True
        )
        or (
            isinstance(termination, Mapping)
            and termination.get("censored_by_output_cap") is True
        )
    )


def _review_checkpoint_directory(value: Path, artifact_sha256: str) -> Path:
    root = Path(value).expanduser().absolute()
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise Lfm2CarrierRouteRuntimeError(
            "Review checkpoint root must be a real directory."
        )
    root.mkdir(parents=True, exist_ok=True)
    child = root / artifact_sha256
    if child.exists() and (child.is_symlink() or not child.is_dir()):
        raise Lfm2CarrierRouteRuntimeError(
            "Artifact review checkpoint path must be a real directory."
        )
    child.mkdir(exist_ok=True)
    return child.resolve(strict=True)


def _strict_checkpoint(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Lfm2CarrierRouteRuntimeError(
                    f"Review checkpoint repeats key {key!r}."
                )
            result[key] = value
        return result

    if (
        path.is_symlink()
        or not path.is_file()
        or getattr(path.stat(), "st_nlink", 1) != 1
    ):
        raise Lfm2CarrierRouteRuntimeError(
            "Review checkpoint is missing or link-like."
        )
    try:
        result = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except Lfm2CarrierRouteRuntimeError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CarrierRouteRuntimeError(
            "Review checkpoint is not strict UTF-8 JSON."
        ) from exc
    if not isinstance(result, dict):
        raise Lfm2CarrierRouteRuntimeError("Review checkpoint must be an object.")
    return result


def _checkpoint_identity(
    *,
    artifact_sha256: str,
    prompt_index: int,
    prompt: Mapping[str, Any],
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
) -> dict[str, Any]:
    prompt_value = _plain_json(prompt)
    return {
        "artifact_sha256": artifact_sha256,
        "prompt_index": prompt_index,
        "prompt_id": str(prompt.get("id")),
        "prompt_sha256": canonical_sha256(prompt_value),
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "maximum_new_tokens": maximum_new_tokens,
        "entrypoint": LFM2_SEMANTIC_STEP_GENERATE_ENTRYPOINT,
    }


def _load_review_row(
    path: Path,
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _strict_checkpoint(path)
    seal = raw.pop("checkpoint_sha256", None)
    if (
        raw.get("schema") != LFM2_CARRIER_REVIEW_ROW_SCHEMA
        or raw.get("status") != "complete"
        or raw.get("identity") != dict(identity)
        or not isinstance(seal, str)
        or seal != canonical_sha256(raw)
    ):
        raise Lfm2CarrierRouteRuntimeError(
            "Review checkpoint identity or self-seal is invalid."
        )
    result = raw.get("routed_result")
    if not isinstance(result, Mapping):
        raise Lfm2CarrierRouteRuntimeError(
            "Review checkpoint has no routed result."
        )
    routed = verify_lfm2_carrier_routed_review(result)
    responses = routed.get("responses")
    if (
        not isinstance(responses, list)
        or len(responses) != 1
        or str(responses[0].get("id")) != identity["prompt_id"]
    ):
        raise Lfm2CarrierRouteRuntimeError(
            "Review checkpoint does not contain exactly its identified prompt row."
        )
    return routed


def _save_review_row(
    path: Path,
    *,
    identity: Mapping[str, Any],
    routed_result: Mapping[str, Any],
) -> None:
    if path.exists() or path.is_symlink():
        raise Lfm2CarrierRouteRuntimeError(
            "Review checkpoint appeared before atomic publication."
        )
    body = {
        "schema": LFM2_CARRIER_REVIEW_ROW_SCHEMA,
        "status": "complete",
        "identity": dict(identity),
        "routed_result": _plain_json(routed_result),
    }
    receipt = {**body, "checkpoint_sha256": canonical_sha256(body)}
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".review-row.",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _run_or_load_review_row(
    model: Any,
    prompt: Mapping[str, Any],
    *,
    prompt_index: int,
    artifact_sha256: str,
    checkpoint_directory: Path,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int,
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
) -> tuple[dict[str, Any], bool, str]:
    identity = _checkpoint_identity(
        artifact_sha256=artifact_sha256,
        prompt_index=prompt_index,
        prompt=prompt,
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
    )
    identity_sha256 = canonical_sha256(identity)
    path = checkpoint_directory / f"review-row-{identity_sha256}.json"
    if path.exists() or path.is_symlink():
        return _load_review_row(path, identity=identity), True, identity_sha256
    entrypoint = getattr(model, LFM2_SEMANTIC_STEP_GENERATE_ENTRYPOINT, None)
    if not callable(entrypoint):
        raise Lfm2CarrierRouteRuntimeError(
            "Model does not expose semantic_step_generate."
        )
    routed = verify_lfm2_carrier_routed_review(
        entrypoint(
            (dict(prompt),),
            maximum_prompt_tokens=maximum_prompt_tokens,
            maximum_new_tokens=maximum_new_tokens,
            progress_callback=progress_callback,
        )
    )
    responses = routed.get("responses")
    if (
        not isinstance(responses, list)
        or len(responses) != 1
        or str(responses[0].get("id")) != str(prompt.get("id"))
    ):
        raise Lfm2CarrierRouteRuntimeError(
            "Per-prompt routed review changed or omitted its prompt ID."
        )
    _save_review_row(path, identity=identity, routed_result=routed)
    return routed, False, identity_sha256


@dataclass(slots=True)
class _RuntimeRecord:
    semantic_source_tokenizer: Any
    target_tokenizer: Any
    runtime: Any | None = None
    runtime_factory: Callable[[Any], Any] | None = None
    lock: threading.RLock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Keep the lock out of the public constructor without a Pydantic/ORM
        # dependency; this is an ordinary in-process synchronization primitive.
        self.lock = threading.RLock()

    def resolve(self, model: Any) -> Any:
        with self.lock:
            if self.runtime is None:
                if self.runtime_factory is None:
                    raise Lfm2CarrierRouteRuntimeError(
                        "Semantic-route runtime registration is incomplete."
                    )
                self.runtime = self.runtime_factory(model)
            return self.runtime


_RUNTIME_REGISTRY: weakref.WeakKeyDictionary[Any, _RuntimeRecord] = (
    weakref.WeakKeyDictionary()
)
_RUNTIME_REGISTRY_LOCK = threading.RLock()


def _record_for_model(model: Any) -> _RuntimeRecord:
    with _RUNTIME_REGISTRY_LOCK:
        try:
            record = _RUNTIME_REGISTRY.get(model)
        except TypeError as exc:
            raise Lfm2CarrierRouteRuntimeError(
                "Semantic-route models must support weak identity registration."
            ) from exc
    if record is None:
        raise Lfm2CarrierRouteRuntimeError(
            "This model has no authenticated semantic-route runtime registration."
        )
    return record


def verify_lfm2_carrier_routed_review(value: Mapping[str, Any]) -> dict[str, Any]:
    """Require evidence that both route channels and the child body executed."""

    if not isinstance(value, Mapping):
        raise Lfm2CarrierRouteRuntimeError("Routed review must be a JSON object.")
    result = _plain_json(value)
    operation_audit = result.get("operation_audit")
    sessions = result.get("routing_session_receipt")
    input_session = sessions.get("input") if isinstance(sessions, Mapping) else None
    output_session = sessions.get("output") if isinstance(sessions, Mapping) else None
    if (
        result.get("schema") != LFM2_SEMANTIC_STEP_MODEL_RUNTIME_SCHEMA
        or result.get("status")
        != "complete-standalone-semantic-step-whole-response-review"
        or result.get("input_output_tables_are_untied") is not True
        or not isinstance(operation_audit, Mapping)
        or not isinstance(input_session, Mapping)
        or not isinstance(output_session, Mapping)
        or input_session.get("route_role") != "input"
        or output_session.get("route_role") != "output"
    ):
        raise Lfm2CarrierRouteRuntimeError(
            "Routed review did not authenticate independent C_in/C_out sessions."
        )
    body_calls = operation_audit.get("child_body_forward_calls")
    input_uses = input_session.get("routing_cache_use_count")
    output_uses = output_session.get("routing_cache_use_count")
    for observed, name in (
        (body_calls, "child body forward calls"),
        (input_uses, "C_in route uses"),
        (output_uses, "C_out route uses"),
    ):
        _positive_int(observed, name)
    if (
        operation_audit.get("child_generate_calls") != 0
        or operation_audit.get("child_lm_head_forward_calls") != 0
        or operation_audit.get("routing_input_output_coefficient_storage_shared")
        is not False
    ):
        raise Lfm2CarrierRouteRuntimeError(
            "Routed review silently used ordinary generate/readout or shared routes."
        )
    return result


def semantic_step_generate(
    self: Any,
    prompts: Sequence[Mapping[str, Any]],
    *,
    maximum_prompt_tokens: int = 8_192,
    maximum_new_tokens: int = 256,
    maximum_embedding_rows_per_chunk: int = 1_024,
    maximum_semantic_rows_per_score_chunk: int = 4_096,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Generate through packaged C_in/C_out routes, never ordinary ``generate``."""

    record = _record_for_model(self)
    runtime = record.resolve(self)
    result = run_lfm2_semantic_step_model_review(
        self,
        runtime,
        record.semantic_source_tokenizer,
        record.target_tokenizer,
        prompts,
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
        maximum_embedding_rows_per_chunk=maximum_embedding_rows_per_chunk,
        maximum_donor_rows_per_score_chunk=(
            maximum_semantic_rows_per_score_chunk
        ),
        progress_callback=progress_callback,
    )
    return verify_lfm2_carrier_routed_review(result)


def install_lfm2_semantic_step_generate(
    model: Any,
    *,
    semantic_source_tokenizer: Any,
    target_tokenizer: Any,
    runtime: Any | None = None,
    runtime_factory: Callable[[Any], Any] | None = None,
) -> Any:
    """Install the public class method and one weak runtime registration."""

    if (runtime is None) == (runtime_factory is None):
        raise Lfm2CarrierRouteRuntimeError(
            "Exactly one of runtime or runtime_factory must be supplied."
        )
    if runtime_factory is not None and not callable(runtime_factory):
        raise Lfm2CarrierRouteRuntimeError("runtime_factory must be callable.")
    if not callable(getattr(model, "model", None)):
        raise Lfm2CarrierRouteRuntimeError(
            "Semantic-step generation requires a callable child body at model."
        )
    for tokenizer, name in (
        (semantic_source_tokenizer, "semantic source tokenizer"),
        (target_tokenizer, "target tokenizer"),
    ):
        if not callable(getattr(tokenizer, "get_vocab", None)) or not callable(
            getattr(tokenizer, "decode", None)
        ):
            raise Lfm2CarrierRouteRuntimeError(
                f"{name} does not expose the required local tokenizer API."
            )
    install_lfm2_semantic_step_generate_class(type(model))
    record = _RuntimeRecord(
        semantic_source_tokenizer=semantic_source_tokenizer,
        target_tokenizer=target_tokenizer,
        runtime=runtime,
        runtime_factory=runtime_factory,
    )
    with _RUNTIME_REGISTRY_LOCK:
        _RUNTIME_REGISTRY[model] = record
    return model


def install_lfm2_semantic_step_generate_class(model_type: type[Any]) -> type[Any]:
    """Preinstall the tensor-free public entrypoint before executor observation."""

    if not isinstance(model_type, type):
        raise Lfm2CarrierRouteRuntimeError("model_type must be a concrete class.")
    existing = model_type.__dict__.get(LFM2_SEMANTIC_STEP_GENERATE_ENTRYPOINT)
    if existing is None:
        setattr(
            model_type,
            LFM2_SEMANTIC_STEP_GENERATE_ENTRYPOINT,
            semantic_step_generate,
        )
    elif existing is not semantic_step_generate:
        raise Lfm2CarrierRouteRuntimeError(
            "Model class already exposes a different semantic_step_generate method."
        )
    return model_type


def run_lfm2_carrier_route_smoke(
    model: Any,
    prompts: Sequence[Mapping[str, Any]],
    *,
    maximum_prompt_tokens: int,
    maximum_new_tokens: int = 2,
) -> dict[str, Any]:
    """Run a bounded ABI smoke proving C_in, C_out, and child-body execution."""

    if not prompts:
        raise Lfm2CarrierRouteRuntimeError("Route smoke needs at least one prompt.")
    entrypoint = getattr(model, LFM2_SEMANTIC_STEP_GENERATE_ENTRYPOINT, None)
    if not callable(entrypoint):
        raise Lfm2CarrierRouteRuntimeError(
            "Model does not expose semantic_step_generate."
        )
    result = entrypoint(
        (dict(prompts[0]),),
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=maximum_new_tokens,
    )
    return verify_lfm2_carrier_routed_review(result)


def run_lfm2_adaptive_carrier_manual_review(
    model: Any,
    prompts: Sequence[Mapping[str, Any]],
    *,
    maximum_prompt_tokens: int = 8_192,
    initial_maximum_new_tokens: int = 1_024,
    retry_maximum_new_tokens: int = 2_048,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run one full routed dossier and retry only output-cap-censored rows."""

    initial_limit = _positive_int(
        initial_maximum_new_tokens,
        "initial_maximum_new_tokens",
    )
    retry_limit = _positive_int(
        retry_maximum_new_tokens,
        "retry_maximum_new_tokens",
    )
    if retry_limit <= initial_limit or retry_limit > 2_048:
        raise Lfm2CarrierRouteRuntimeError(
            "Adaptive retry limit must be larger than the initial limit and <= 2048."
        )
    normalized_prompts = tuple(dict(value) for value in prompts)
    if not normalized_prompts:
        raise Lfm2CarrierRouteRuntimeError("Adaptive review needs prompts.")
    prompt_ids = [str(value.get("id")) for value in normalized_prompts]
    if any(value == "None" for value in prompt_ids) or len(set(prompt_ids)) != len(
        prompt_ids
    ):
        raise Lfm2CarrierRouteRuntimeError(
            "Adaptive review prompt IDs must be present and unique."
        )
    entrypoint = getattr(model, LFM2_SEMANTIC_STEP_GENERATE_ENTRYPOINT, None)
    if not callable(entrypoint):
        raise Lfm2CarrierRouteRuntimeError(
            "Model does not expose semantic_step_generate."
        )
    initial = verify_lfm2_carrier_routed_review(
        entrypoint(
            normalized_prompts,
            maximum_prompt_tokens=maximum_prompt_tokens,
            maximum_new_tokens=initial_limit,
            progress_callback=progress_callback,
        )
    )
    initial_responses = initial.get("responses")
    if not isinstance(initial_responses, list) or len(initial_responses) != len(
        normalized_prompts
    ):
        raise Lfm2CarrierRouteRuntimeError(
            "Initial routed review did not preserve every prompt response."
        )
    initial_response_ids = [
        str(value.get("id")) if isinstance(value, Mapping) else "None"
        for value in initial_responses
    ]
    if initial_response_ids != prompt_ids or len(set(initial_response_ids)) != len(
        initial_response_ids
    ):
        raise Lfm2CarrierRouteRuntimeError(
            "Initial routed review changed, omitted, or duplicated prompt IDs."
        )
    capped_ids = []
    for response in initial_responses:
        diagnostics = response.get("diagnostics") if isinstance(response, Mapping) else None
        if isinstance(diagnostics, Mapping) and diagnostics.get(
            "censored_by_output_cap"
        ) is True:
            capped_ids.append(str(response.get("id")))
    prompt_by_id = {str(value.get("id")): value for value in normalized_prompts}
    retry_prompts = tuple(prompt_by_id[value] for value in capped_ids)
    retry = None
    replacements: dict[str, Any] = {}
    if retry_prompts:
        retry = verify_lfm2_carrier_routed_review(
            entrypoint(
                retry_prompts,
                maximum_prompt_tokens=maximum_prompt_tokens,
                maximum_new_tokens=retry_limit,
                progress_callback=progress_callback,
            )
        )
        retry_responses = retry.get("responses")
        if not isinstance(retry_responses, list) or len(retry_responses) != len(
            retry_prompts
        ):
            raise Lfm2CarrierRouteRuntimeError(
                "Adaptive retry did not preserve every capped response."
            )
        retry_response_ids = [
            str(value.get("id")) if isinstance(value, Mapping) else "None"
            for value in retry_responses
        ]
        if retry_response_ids != capped_ids or len(set(retry_response_ids)) != len(
            retry_response_ids
        ):
            raise Lfm2CarrierRouteRuntimeError(
                "Adaptive retry changed, omitted, or duplicated capped prompt IDs."
            )
        replacements = {str(value.get("id")): value for value in retry_responses}
    final_responses = [
        replacements.get(str(value.get("id")), value) for value in initial_responses
    ]
    safety_observations = [
        value
        for value in final_responses
        if isinstance(value, Mapping)
        and _is_safety_category(value.get("category"))
    ]
    intelligence_review_responses = [
        value
        for value in final_responses
        if not (
            isinstance(value, Mapping)
            and _is_safety_category(value.get("category"))
        )
    ]
    still_capped = sum(
        int(
            isinstance(value.get("diagnostics"), Mapping)
            and value["diagnostics"].get("censored_by_output_cap") is True
        )
        for value in final_responses
        if isinstance(value, Mapping)
    )
    body = {
        "schema": LFM2_CARRIER_ROUTED_REVIEW_SCHEMA,
        "status": "complete-ready-for-manual-intelligence-review",
        "responses": final_responses,
        "response_count": len(final_responses),
        "intelligence_review_responses": intelligence_review_responses,
        "intelligence_review_response_count": len(intelligence_review_responses),
        "safety_observations": safety_observations,
        "safety_observation_count": len(safety_observations),
        "safety_excluded_from_intelligence_aggregate": True,
        "initial_maximum_new_tokens": initial_limit,
        "retry_maximum_new_tokens": retry_limit,
        "initial_cap_censored_response_count": len(capped_ids),
        "adaptive_retry_response_count": len(replacements),
        "remaining_cap_censored_response_count": still_capped,
        "initial_runtime_sha256": initial.get("runtime_sha256"),
        "retry_runtime_sha256": None if retry is None else retry.get("runtime_sha256"),
        "ordinary_generate_used": False,
        "input_and_output_routes_authenticated": True,
        "safety_refusal_is_not_automatic_folding_failure": True,
        "output_cap_censoring_is_not_automatic_folding_failure": True,
        "automatic_intelligence_verdict_performed": False,
        "manual_whole_response_intelligence_review_is_authoritative": True,
    }
    return {**body, "review_sha256": canonical_sha256(body)}


def run_lfm2_resumable_adaptive_carrier_manual_review(
    model: Any,
    prompts: Sequence[Mapping[str, Any]],
    *,
    checkpoint_directory: Path,
    published_child_artifact_sha256: str,
    maximum_prompt_tokens: int = 8_192,
    initial_maximum_new_tokens: int = 1_024,
    retry_maximum_new_tokens: int = 2_048,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Resume a manual dossier from sealed per-artifact/prompt/cap rows.

    The caller may load ``model`` through
    ``Lfm2CarrierChildCheckpointIO.from_child_directory`` in a fresh process;
    neither this function nor its checkpoints need the fold-time publication or
    tokenizer.  Each prompt gets one authenticated 1024-token row.  Only rows
    censored by that output cap get a distinct authenticated 2048-token row.
    """

    artifact_sha256 = _digest(
        published_child_artifact_sha256,
        "published child artifact SHA-256",
    )
    prompt_limit = _positive_int(maximum_prompt_tokens, "maximum_prompt_tokens")
    initial_limit = _positive_int(
        initial_maximum_new_tokens,
        "initial_maximum_new_tokens",
    )
    retry_limit = _positive_int(
        retry_maximum_new_tokens,
        "retry_maximum_new_tokens",
    )
    if retry_limit <= initial_limit or retry_limit > 2_048:
        raise Lfm2CarrierRouteRuntimeError(
            "Adaptive retry limit must be larger than the initial limit and <= 2048."
        )
    normalized_prompts = tuple(dict(value) for value in prompts)
    if not normalized_prompts:
        raise Lfm2CarrierRouteRuntimeError("Resumable review needs prompts.")
    prompt_ids = [str(value.get("id")) for value in normalized_prompts]
    if any(value == "None" for value in prompt_ids) or len(set(prompt_ids)) != len(
        prompt_ids
    ):
        raise Lfm2CarrierRouteRuntimeError(
            "Resumable review prompt IDs must be present and unique."
        )
    checkpoints = _review_checkpoint_directory(
        checkpoint_directory,
        artifact_sha256,
    )
    final_responses: list[Any] = []
    row_receipts: list[dict[str, Any]] = []
    initial_capped = 0
    retry_count = 0
    reused_count = 0
    executed_count = 0
    runtime_sha256s: list[str] = []
    for prompt_index, prompt in enumerate(normalized_prompts):
        initial, initial_reused, initial_identity = _run_or_load_review_row(
            model,
            prompt,
            prompt_index=prompt_index,
            artifact_sha256=artifact_sha256,
            checkpoint_directory=checkpoints,
            maximum_prompt_tokens=prompt_limit,
            maximum_new_tokens=initial_limit,
            progress_callback=progress_callback,
        )
        reused_count += int(initial_reused)
        executed_count += int(not initial_reused)
        initial_response = initial["responses"][0]
        runtime_sha256s.append(str(initial.get("runtime_sha256")))
        row_receipts.append(
            {
                "prompt_id": str(prompt["id"]),
                "maximum_new_tokens": initial_limit,
                "checkpoint_identity_sha256": initial_identity,
                "reused": initial_reused,
            }
        )
        final_response = initial_response
        if _is_cap_censored(initial_response):
            initial_capped += 1
            retry_count += 1
            retry, retry_reused, retry_identity = _run_or_load_review_row(
                model,
                prompt,
                prompt_index=prompt_index,
                artifact_sha256=artifact_sha256,
                checkpoint_directory=checkpoints,
                maximum_prompt_tokens=prompt_limit,
                maximum_new_tokens=retry_limit,
                progress_callback=progress_callback,
            )
            reused_count += int(retry_reused)
            executed_count += int(not retry_reused)
            final_response = retry["responses"][0]
            runtime_sha256s.append(str(retry.get("runtime_sha256")))
            row_receipts.append(
                {
                    "prompt_id": str(prompt["id"]),
                    "maximum_new_tokens": retry_limit,
                    "checkpoint_identity_sha256": retry_identity,
                    "reused": retry_reused,
                }
            )
        final_responses.append(final_response)
    safety_observations = [
        value
        for value in final_responses
        if isinstance(value, Mapping)
        and _is_safety_category(value.get("category"))
    ]
    intelligence_review_responses = [
        value
        for value in final_responses
        if not (
            isinstance(value, Mapping)
            and _is_safety_category(value.get("category"))
        )
    ]
    still_capped = sum(int(_is_cap_censored(value)) for value in final_responses)
    body = {
        "schema": LFM2_CARRIER_ROUTED_REVIEW_SCHEMA,
        "status": "complete-ready-for-manual-intelligence-review",
        "published_child_artifact_sha256": artifact_sha256,
        "responses": final_responses,
        "response_count": len(final_responses),
        "intelligence_review_responses": intelligence_review_responses,
        "intelligence_review_response_count": len(intelligence_review_responses),
        "safety_observations": safety_observations,
        "safety_observation_count": len(safety_observations),
        "safety_excluded_from_intelligence_aggregate": True,
        "initial_maximum_new_tokens": initial_limit,
        "retry_maximum_new_tokens": retry_limit,
        "initial_cap_censored_response_count": initial_capped,
        "adaptive_retry_response_count": retry_count,
        "remaining_cap_censored_response_count": still_capped,
        "checkpoint_row_receipts": row_receipts,
        "checkpoint_row_count": len(row_receipts),
        "checkpoint_rows_reused": reused_count,
        "checkpoint_rows_executed": executed_count,
        "runtime_sha256s": runtime_sha256s,
        "ordinary_generate_used": False,
        "input_and_output_routes_authenticated": True,
        "safety_refusal_is_not_automatic_folding_failure": True,
        "output_cap_censoring_is_not_automatic_folding_failure": True,
        "automatic_intelligence_verdict_performed": False,
        "manual_whole_response_intelligence_review_is_authoritative": True,
    }
    return {**body, "review_sha256": canonical_sha256(body)}


__all__ = [
    "LFM2_CARRIER_ROUTED_REVIEW_SCHEMA",
    "LFM2_CARRIER_REVIEW_ROW_SCHEMA",
    "LFM2_SEMANTIC_STEP_GENERATE_ENTRYPOINT",
    "Lfm2CarrierRouteRuntimeError",
    "install_lfm2_semantic_step_generate",
    "install_lfm2_semantic_step_generate_class",
    "run_lfm2_adaptive_carrier_manual_review",
    "run_lfm2_carrier_route_smoke",
    "run_lfm2_resumable_adaptive_carrier_manual_review",
    "semantic_step_generate",
    "verify_lfm2_carrier_routed_review",
]
